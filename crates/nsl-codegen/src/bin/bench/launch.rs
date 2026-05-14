//! CUDA kernel launch + timing harness for the `nsl-codegen-bench` binary.
//!
//! Per §8.2 of the 2026-05-13 design spec, this module is responsible for:
//!
//! 1. Bracketing each kernel launch with CUDA events.
//! 2. Looping over `iterations` inner launches, collecting per-iter elapsed
//!    times via `cuEventElapsedTime`, and computing the median.
//! 3. Optionally, reading back the M3 skip-decision HBM buffer once at the
//!    end of the loop (decisions are deterministic per-fixture, so a single
//!    snapshot suffices) and computing the kv-tile skip ratio.
//!
//! The launch is performed via cudarc's low-level driver `sys::*` API,
//! mirroring `tests/tier_b_smem_probe.rs`'s style: explicit `cuInit`,
//! primary-context retain, `cuModuleLoadDataEx` with JIT error-log capture
//! so PTX-level failures surface real diagnostics, `cuFuncSetAttribute` for
//! the >48 KB dynamic-SMEM opt-in, and `cuMemAlloc_v2` / `cuMemFree_v2` for
//! per-launch device buffers.
//!
//! Kernel-arg layout mirrors `nsl_flash_attention_csha`'s 37-slot list
//! (`crates/nsl-runtime/src/flash_attention.rs:485`) so the bench launches
//! the same PTX entry point as the production code paths. Note: the
//! current PTX synthesis emits 36 params for `segment_masked=false` and
//! 37 for `segment_masked=true` (the gate fixture uses the latter); the
//! arg array's segment-ids slot is always present here and the kernel
//! ignores it when its prelude didn't declare it.
//!
//! # Safety
//!
//! `time_kernel_launches` is `unsafe` because it dereferences raw CUDA
//! handles (`CUfunction`, `CUdeviceptr`) and a raw kernel-argument pointer
//! array. Callers must guarantee:
//!
//! * `func` is a valid `CUfunction` loaded into the current context.
//! * `args` slots point to live, correctly-sized argument storage for the
//!   duration of the call.
//! * `skip_decisions_buf`, if `Some`, names a live device allocation of at
//!   least the stated byte length.
//! * The current thread has the appropriate CUDA primary context active
//!   (cf. `nsl-runtime`'s `ensure_context`).

use crate::bin::bench::fixtures::Fixture;
use crate::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2_with_seqlen,
    synthesize_flash_attention_ptx_v2_with_tier_b,
};
use crate::pca_segment::SegmentResidency;
use cudarc::driver::sys;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;

#[derive(Debug, Clone)]
pub struct LaunchResult {
    /// Median per-iteration kernel time in microseconds.
    pub median_us: f64,
    /// Ratio of kv-tiles skipped by the Tier B predicate. `0.0` when the
    /// skip-decision buffer is `None` (Tier-B-off run) or all-zeros. The
    /// Tier B PTX-side writeback is wired in task B1.5-3; until then this
    /// reads as 0.0 for Tier-B-on runs too.
    pub skip_ratio: f64,
}

/// Time `iterations` back-to-back launches of `func` with `args`, returning
/// the median per-iter wall time and (optionally) the kv-tile skip ratio
/// read from `skip_decisions_buf`.
///
/// # Returns
///
/// * `Ok(LaunchResult)` on success.
/// * `Err(String)` on CUDA driver failure (event create / record / sync /
///   elapsed-time, kernel launch, or device-to-host memcpy). The error
///   message includes the failing `CUresult` for diagnostics. Callers
///   should map this to bench exit code 2 (CUDA/launch error).
///
/// # Safety
///
/// Caller must guarantee:
///
/// * `func` is a valid `CUfunction` loaded into the current CUDA primary
///   context. The current thread has that context active (cf.
///   `nsl-runtime`'s `ensure_context`).
/// * `args` is a kernel-argument-pointer array shaped per the kernel's
///   parameter list; each entry points to live, correctly-sized argument
///   storage for the duration of this call.
/// * `skip_decisions_buf`, if `Some`, names a live device allocation of
///   at least the stated byte length and remains live until this call
///   returns.
/// * `iterations > 0` (debug-checked).
pub unsafe fn time_kernel_launches(
    func: sys::CUfunction,
    args: &mut [*mut c_void],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shmem_bytes: u32,
    iterations: u32,
    skip_decisions_buf: Option<(sys::CUdeviceptr, usize)>,
) -> Result<LaunchResult, String> {
    debug_assert!(iterations > 0, "iterations must be > 0");

    // --- Event pair: created once, reused across all iterations. ---
    let mut start_evt: sys::CUevent = std::ptr::null_mut();
    let mut stop_evt: sys::CUevent = std::ptr::null_mut();
    let rc = sys::cuEventCreate(&mut start_evt, 0);
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuEventCreate(start) rc={:?}: {}", rc, cu_error_string(rc)));
    }
    let rc = sys::cuEventCreate(&mut stop_evt, 0);
    if rc != sys::CUresult::CUDA_SUCCESS {
        let _ = sys::cuEventDestroy_v2(start_evt);
        return Err(format!("cuEventCreate(stop) rc={:?}: {}", rc, cu_error_string(rc)));
    }

    // --- Per-iter timing loop. ---
    let mut times_ms: Vec<f32> = Vec::with_capacity(iterations as usize);
    for it in 0..iterations {
        let rc = sys::cuEventRecord(start_evt, std::ptr::null_mut());
        if rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuEventDestroy_v2(start_evt);
            let _ = sys::cuEventDestroy_v2(stop_evt);
            return Err(format!(
                "cuEventRecord(start) iter={} rc={:?}: {}",
                it,
                rc,
                cu_error_string(rc)
            ));
        }
        let launch_rc = sys::cuLaunchKernel(
            func,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shmem_bytes,
            std::ptr::null_mut(), // default stream
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if launch_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuEventDestroy_v2(start_evt);
            let _ = sys::cuEventDestroy_v2(stop_evt);
            return Err(format!(
                "cuLaunchKernel iter={} rc={:?}: {}",
                it,
                launch_rc,
                cu_error_string(launch_rc)
            ));
        }
        let rc = sys::cuEventRecord(stop_evt, std::ptr::null_mut());
        if rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuEventDestroy_v2(start_evt);
            let _ = sys::cuEventDestroy_v2(stop_evt);
            return Err(format!(
                "cuEventRecord(stop) iter={} rc={:?}: {}",
                it,
                rc,
                cu_error_string(rc)
            ));
        }
        let rc = sys::cuEventSynchronize(stop_evt);
        if rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuEventDestroy_v2(start_evt);
            let _ = sys::cuEventDestroy_v2(stop_evt);
            return Err(format!(
                "cuEventSynchronize iter={} rc={:?}: {}",
                it,
                rc,
                cu_error_string(rc)
            ));
        }
        let mut ms: f32 = 0.0;
        let rc = sys::cuEventElapsedTime_v2(&mut ms, start_evt, stop_evt);
        if rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuEventDestroy_v2(start_evt);
            let _ = sys::cuEventDestroy_v2(stop_evt);
            return Err(format!(
                "cuEventElapsedTime iter={} rc={:?}: {}",
                it,
                rc,
                cu_error_string(rc)
            ));
        }
        times_ms.push(ms);
    }

    let _ = sys::cuEventDestroy_v2(start_evt);
    let _ = sys::cuEventDestroy_v2(stop_evt);

    // --- Median across iterations (sort + middle slot). ---
    // For even-iter counts we still pick the upper-middle (index n/2) as
    // a single representative sample. This is consistent with the spec
    // §8.1 wording: "median across the inner samples"; no interpolation
    // needed because we want a real observed time, not an average.
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_ms = times_ms[times_ms.len() / 2] as f64;
    let median_us = median_ms * 1000.0;

    // --- Skip-ratio readback (Tier-B-on only). ---
    // The PTX-side `emit_skip_decision_writeback` is wired in task B1.5-3;
    // until then this buffer is all-zeros and skip_ratio reads as 0.0 even
    // for Tier-B-on. The bench machinery is in place so B1.5-3 plugs in
    // with no further infrastructure work.
    let skip_ratio = if let Some((dev_ptr, byte_len)) = skip_decisions_buf {
        let mut host = vec![0u8; byte_len];
        let rc = sys::cuMemcpyDtoH_v2(
            host.as_mut_ptr() as *mut c_void,
            dev_ptr,
            byte_len,
        );
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemcpyDtoH(skip_decisions) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ));
        }
        let set = host.iter().filter(|&&b| b != 0).count();
        (set as f64) / (host.len().max(1) as f64)
    } else {
        0.0
    };

    Ok(LaunchResult {
        median_us,
        skip_ratio,
    })
}

/// Format a CUresult into a human-readable string via `cuGetErrorString`.
/// Returns `<unknown>` on any failure of the introspection itself; never
/// panics or allocates if the driver returns a null name.
unsafe fn cu_error_string(rc: sys::CUresult) -> String {
    let mut name: *const std::os::raw::c_char = std::ptr::null();
    let _ = sys::cuGetErrorString(rc, &mut name);
    if name.is_null() {
        "<unknown>".to_string()
    } else {
        CStr::from_ptr(name).to_string_lossy().into_owned()
    }
}

/// End-to-end run of one fixture: synthesize PTX, allocate device buffers,
/// load + launch, measure, free.
///
/// Returns the `LaunchResult` (median_us + skip_ratio) on success. On any
/// CUDA error returns `Err(msg)` mapped by main() to exit code 2.
///
/// # Safety
///
/// Calls into raw cudarc driver APIs and the runtime's CUDA-init shim.
/// Caller must invoke from a single-threaded context (the CUDA driver's
/// thread-locality model assumes the primary context is current on this
/// thread).
pub unsafe fn run_fixture(
    fixture: &Fixture,
    tier_b_on: bool,
    seed: u64,
    iterations: u32,
) -> Result<LaunchResult, String> {
    // -- Step 1: Ensure CUDA context is current on this thread. --
    let mut ctx: sys::CUcontext = std::ptr::null_mut();
    let _ = sys::cuCtxGetCurrent(&mut ctx);
    if ctx.is_null() {
        let _ = sys::cuInit(0);
        let mut dev: sys::CUdevice = 0;
        let rc = sys::cuDeviceGet(&mut dev, 0);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDeviceGet rc={:?}: {}", rc, cu_error_string(rc)));
        }
        let rc = sys::cuDevicePrimaryCtxRetain(&mut ctx, dev);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuDevicePrimaryCtxRetain rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ));
        }
        let rc = sys::cuCtxSetCurrent(ctx);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuCtxSetCurrent rc={:?}: {}", rc, cu_error_string(rc)));
        }
    }

    // -- Step 2: Synthesize PTX. --
    // Tier B argument: pass Some((seq_len, Shared)) only when the bench
    // is run with `--tier-b on`. `should_emit_tier_b` (the fine-grained
    // budget check in `pca_tilerange`) further admits the emission inside
    // the synthesizer. Note this bypasses the central
    // `flash_attention_v2::should_emit_tier_b` toggle by design — the
    // bench is the calibration tool *for* that toggle.
    let tier_b_arg = if tier_b_on {
        Some((fixture.seq_len, SegmentResidency::Shared))
    } else {
        None
    };
    let mut ptx = synthesize_flash_attention_ptx_v2_with_tier_b(&fixture.config, tier_b_arg);
    // Defensive: strip trailing NULs (the synthesizer ends with `\n\0` for
    // direct cuModuleLoad consumption), ensure trailing newline, then
    // null-terminate for cuModuleLoadDataEx.
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    ptx.push(0);

    // Optional debug: dump the synthesized PTX so a failed launch can be
    // post-mortemed offline. Gated by env var so the production bench path
    // doesn't write to disk per iteration.
    if std::env::var_os("NSL_BENCH_DUMP_PTX").is_some() {
        let p = std::env::temp_dir().join(format!(
            "bench_{}_tier_b_{}.ptx",
            fixture.name,
            if tier_b_on { "on" } else { "off" }
        ));
        let body = &ptx[..ptx.len() - 1]; // strip trailing NUL for readability
        let _ = std::fs::write(&p, body);
        eprintln!("[bench] PTX dumped to: {}", p.display());
    }

    // -- Step 3: Load module with JIT error capture. --
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let mut err_log = vec![0u8; 8192];
    let info_log = vec![0u8; 4096];
    let mut options: [sys::CUjit_option; 4] = [
        sys::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
        sys::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        sys::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER,
        sys::CUjit_option_enum::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut option_values: [*mut c_void; 4] = [
        err_log.as_mut_ptr() as *mut c_void,
        err_log.len() as *mut c_void,
        info_log.as_ptr() as *mut c_void,
        info_log.len() as *mut c_void,
    ];
    let rc = sys::cuModuleLoadDataEx(
        &mut module,
        ptx.as_ptr() as *const c_void,
        options.len() as u32,
        options.as_mut_ptr(),
        option_values.as_mut_ptr(),
    );
    if rc != sys::CUresult::CUDA_SUCCESS {
        let nul = err_log.iter().position(|&b| b == 0).unwrap_or(err_log.len());
        let log = String::from_utf8_lossy(&err_log[..nul]).trim().to_string();
        return Err(format!(
            "cuModuleLoadDataEx rc={:?}: {} | JIT log: {}",
            rc,
            cu_error_string(rc),
            if log.is_empty() { "<empty>".into() } else { log }
        ));
    }

    // -- Step 4: Get function handle. --
    let kernel_name = match CString::new(flash_attention_kernel_name_v2(&fixture.config)) {
        Ok(n) => n,
        Err(_) => {
            let _ = sys::cuModuleUnload(module);
            return Err("kernel name contains embedded NUL".into());
        }
    };
    let mut func: sys::CUfunction = std::ptr::null_mut();
    let rc = sys::cuModuleGetFunction(&mut func, module, kernel_name.as_ptr());
    if rc != sys::CUresult::CUDA_SUCCESS {
        let _ = sys::cuModuleUnload(module);
        return Err(format!(
            "cuModuleGetFunction({}) rc={:?}: {}",
            kernel_name.to_string_lossy(),
            rc,
            cu_error_string(rc)
        ));
    }

    // -- Step 5: Allocate device buffers + populate inputs. --
    // Gate fixture uses heads=1 (the matrix doesn't multiplex over heads
    // because we measure tile-skip behaviour, which is per-(batch, head)
    // independent).
    let heads: u32 = 1;
    let head_dim = fixture.config.head_dim as u32;
    let total_elems = (fixture.batch as usize)
        * (heads as usize)
        * (fixture.seq_len as usize)
        * (head_dim as usize);
    let qkv_bytes_per = (total_elems * std::mem::size_of::<f32>()) as usize;
    // FA2 forward writes f16 to `out`; lse is f32 [B, H, S].
    let out_bytes = (total_elems * std::mem::size_of::<u16>()) as usize;
    let lse_elems =
        (fixture.batch as usize) * (heads as usize) * (fixture.seq_len as usize);
    let lse_bytes = (lse_elems * std::mem::size_of::<f32>()) as usize;
    let seg_bytes_per_batch =
        (fixture.seq_len as usize) * std::mem::size_of::<u16>();

    let mut allocations: Vec<sys::CUdeviceptr> = Vec::new();
    let alloc =
        |bytes: usize, allocations: &mut Vec<sys::CUdeviceptr>| -> Result<sys::CUdeviceptr, String> {
            let mut p: sys::CUdeviceptr = 0;
            let rc = sys::cuMemAlloc_v2(&mut p, bytes);
            if rc != sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "cuMemAlloc_v2({} bytes) rc={:?}: {}",
                    bytes,
                    rc,
                    cu_error_string(rc)
                ));
            }
            allocations.push(p);
            Ok(p)
        };

    let q_dev = alloc(qkv_bytes_per, &mut allocations)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    let k_dev = alloc(qkv_bytes_per, &mut allocations)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    let v_dev = alloc(qkv_bytes_per, &mut allocations)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    let out_dev = alloc(out_bytes, &mut allocations)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    let lse_dev = alloc(lse_bytes, &mut allocations)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    let seg_dev = alloc(seg_bytes_per_batch, &mut allocations)
        .map_err(|e| free_all_and(&allocations, module, e))?;

    // Generate host inputs deterministically from `seed`. Q/K/V values
    // are small (~[-0.5, 0.5)) to keep softmax numerics calm; the actual
    // numerical correctness comes from the parity tests in B1.5-3, not
    // this bench harness.
    let q_host = fill_seeded(total_elems, seed.wrapping_mul(3));
    let k_host = fill_seeded(total_elems, seed.wrapping_mul(5));
    let v_host = fill_seeded(total_elems, seed.wrapping_mul(7));
    let seg_host = generate_segment_mask(fixture.seq_len, fixture.target_sparsity, seed);

    let h2d = |dst: sys::CUdeviceptr, src: *const c_void, n: usize| -> Result<(), String> {
        let rc = sys::cuMemcpyHtoD_v2(dst, src, n);
        if rc != sys::CUresult::CUDA_SUCCESS {
            Err(format!(
                "cuMemcpyHtoD_v2({} bytes) rc={:?}: {}",
                n,
                rc,
                cu_error_string(rc)
            ))
        } else {
            Ok(())
        }
    };

    h2d(q_dev, q_host.as_ptr() as *const c_void, qkv_bytes_per)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    h2d(k_dev, k_host.as_ptr() as *const c_void, qkv_bytes_per)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    h2d(v_dev, v_host.as_ptr() as *const c_void, qkv_bytes_per)
        .map_err(|e| free_all_and(&allocations, module, e))?;
    h2d(
        seg_dev,
        seg_host.as_ptr() as *const c_void,
        seg_bytes_per_batch,
    )
    .map_err(|e| free_all_and(&allocations, module, e))?;

    // Skip-decisions buffer (only on Tier-B-on). Shape per spec §4.3.1:
    // `[batch, head, num_q_tiles, num_kv_tiles] : u8`.
    let skip_decisions_buf: Option<(sys::CUdeviceptr, usize)> = if tier_b_on {
        let num_q = (fixture.seq_len).div_ceil(fixture.config.block_q as u32) as usize;
        let num_kv = (fixture.seq_len).div_ceil(fixture.config.block_kv as u32) as usize;
        let n_slots = (fixture.batch as usize) * (heads as usize) * num_q * num_kv;
        let dev = alloc(n_slots, &mut allocations)
            .map_err(|e| free_all_and(&allocations, module, e))?;
        // Zero the buffer so the readback sees 0 wherever the kernel
        // didn't write (e.g. before B1.5-3 wires the writeback).
        let rc = sys::cuMemsetD8_v2(dev, 0, n_slots);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(free_all_and(
                &allocations,
                module,
                format!(
                    "cuMemsetD8_v2(skip_decisions) rc={:?}: {}",
                    rc,
                    cu_error_string(rc)
                ),
            ));
        }
        Some((dev, n_slots))
    } else {
        None
    };

    // -- Step 6: SMEM opt-in for >48 KB dynamic allocation. --
    // The gate fixture's SMEM allocation is well under the static limit
    // for the chosen tile dims (block 64×64, head_dim=64), but the FA2
    // synthesizer may route to dynamic SMEM depending on
    // `fwd_needs_dynamic_smem` heuristics. Opt in unconditionally; ptxas
    // ignores the attribute set when the kernel uses static SMEM only.
    //
    // When Tier B is on, the kernel's range-table base is computed by
    // `smem_layout::tier_b_range_table_offset(config, Direction::Forward)`
    // — i.e. it rides at the tail of the *forward* total + seg_overhead
    // (~50 KB at the gate fixture's pinned dims). Per design spec §11.4,
    // forward-only kernels MUST pass `Direction::Forward`; passing
    // `Direction::Backward` would inherit the backward-sized offset
    // (~140 KB) and exceed the 99 KB per-CTA dynamic SMEM cap on
    // Blackwell-class hardware.
    //
    // Compute the high-watermark of any SMEM offset the kernel touches:
    //   range_table_base + range_table_bytes  when Tier B is on
    //   forward total + seg_overhead          when Tier B is off
    let shmem_bytes_base = shared_mem_bytes_v2_with_seqlen(
        &fixture.config,
        fixture.seq_len,
        SegmentResidency::Shared,
    );
    let shmem_bytes = if tier_b_on
        && crate::pca_tilerange::should_emit_tier_b(
            &fixture.config,
            fixture.seq_len as u64,
            SegmentResidency::Shared,
        )
    {
        let base = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
            &fixture.config,
            crate::flash_attention_v2::smem_layout::Direction::Forward,
        );
        let tbl =
            crate::pca_tilerange::tier_b_range_table_bytes(&fixture.config, fixture.seq_len);
        shmem_bytes_base.max(base + tbl)
    } else {
        shmem_bytes_base
    };
    const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: sys::CUfunction_attribute =
        sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    let attr_rc = sys::cuFuncSetAttribute(
        func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shmem_bytes as i32,
    );
    // Don't fail on attribute-set for INVALID_VALUE: this is benign for
    // kernels with no dynamic SMEM (the opt-in is a no-op there). Log
    // anyway so diagnostics surface when the size > cap.
    if attr_rc != sys::CUresult::CUDA_SUCCESS {
        eprintln!(
            "[bench] cuFuncSetAttribute({} B) rc={:?}: {}",
            shmem_bytes,
            attr_rc,
            cu_error_string(attr_rc)
        );
    }

    // -- Step 7: Build kernel-args list (37 slots). --
    // Layout mirrors `nsl_flash_attention_csha`
    // (`crates/nsl-runtime/src/flash_attention.rs:485`).
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut a_q: u64 = q_dev;
    let mut a_k: u64 = k_dev;
    let mut a_v: u64 = v_dev;
    let mut a_out: u64 = out_dev;
    let mut a_scale: f32 = scale;
    let mut a_batch: u64 = fixture.batch as u64;
    let mut a_heads: u64 = heads as u64;
    let mut a_seq_len: u64 = fixture.seq_len as u64;
    let mut a_head_dim: u64 = head_dim as u64;
    let mut a_block_table: u64 = 0;
    let mut a_k_pool: u64 = 0;
    let mut a_v_pool: u64 = 0;
    let mut a_block_size: u64 = 0;
    let mut a_cos: u64 = 0;
    let mut a_sin: u64 = 0;
    let mut a_seq_ids: u64 = 0;
    let mut a_seq_lens: u64 = 0;
    let mut a_dfs_enter: u64 = 0;
    let mut a_dfs_exit: u64 = 0;
    let mut a_num_tree_nodes: u64 = 0;
    let mut a_logsumexp: u64 = lse_dev;
    let mut a_csha_x: u64 = 0;
    let mut a_csha_nw: u64 = 0;
    let mut a_csha_wq: u64 = 0;
    let mut a_csha_wk: u64 = 0;
    let mut a_csha_wv: u64 = 0;
    let mut a_csha_wo: u64 = 0;
    let mut a_csha_eps: f32 = 1.0e-5;
    let mut a_csha_active_heads: u32 = 0;
    let mut a_csha_d_model: u32 = 0;
    let mut a_q_proj: u64 = 0;
    let mut a_k_proj: u64 = 0;
    let mut a_v_proj: u64 = 0;
    let mut a_row_max: u64 = 0;
    let mut a_row_sum: u64 = 0;
    let mut a_x_raw: u64 = 0;
    let mut a_seg_ids: u64 = seg_dev;

    let mut args: [*mut c_void; 37] = [
        &mut a_q as *mut _ as *mut c_void,
        &mut a_k as *mut _ as *mut c_void,
        &mut a_v as *mut _ as *mut c_void,
        &mut a_out as *mut _ as *mut c_void,
        &mut a_scale as *mut _ as *mut c_void,
        &mut a_batch as *mut _ as *mut c_void,
        &mut a_heads as *mut _ as *mut c_void,
        &mut a_seq_len as *mut _ as *mut c_void,
        &mut a_head_dim as *mut _ as *mut c_void,
        &mut a_block_table as *mut _ as *mut c_void,
        &mut a_k_pool as *mut _ as *mut c_void,
        &mut a_v_pool as *mut _ as *mut c_void,
        &mut a_block_size as *mut _ as *mut c_void,
        &mut a_cos as *mut _ as *mut c_void,
        &mut a_sin as *mut _ as *mut c_void,
        &mut a_seq_ids as *mut _ as *mut c_void,
        &mut a_seq_lens as *mut _ as *mut c_void,
        &mut a_dfs_enter as *mut _ as *mut c_void,
        &mut a_dfs_exit as *mut _ as *mut c_void,
        &mut a_num_tree_nodes as *mut _ as *mut c_void,
        &mut a_logsumexp as *mut _ as *mut c_void,
        &mut a_csha_x as *mut _ as *mut c_void,
        &mut a_csha_nw as *mut _ as *mut c_void,
        &mut a_csha_wq as *mut _ as *mut c_void,
        &mut a_csha_wk as *mut _ as *mut c_void,
        &mut a_csha_wv as *mut _ as *mut c_void,
        &mut a_csha_wo as *mut _ as *mut c_void,
        &mut a_csha_eps as *mut _ as *mut c_void,
        &mut a_csha_active_heads as *mut _ as *mut c_void,
        &mut a_csha_d_model as *mut _ as *mut c_void,
        &mut a_q_proj as *mut _ as *mut c_void,
        &mut a_k_proj as *mut _ as *mut c_void,
        &mut a_v_proj as *mut _ as *mut c_void,
        &mut a_row_max as *mut _ as *mut c_void,
        &mut a_row_sum as *mut _ as *mut c_void,
        &mut a_x_raw as *mut _ as *mut c_void,
        &mut a_seg_ids as *mut _ as *mut c_void,
    ];

    // -- Step 8: Grid / block dims. --
    // Grid: (seq_len / block_q) along x, (batch * heads) along y. Mirrors
    // `nsl_flash_attention_csha` which is the production launcher for
    // this PTX. 128 threads/block (4 warps × 32 lanes) is the v2 emitter's
    // pinned thread-mapping contract.
    let grid_x = fixture.seq_len.div_ceil(fixture.config.block_q as u32);
    let grid_y = fixture.batch * heads;
    let grid = (grid_x, grid_y, 1);
    let block = (128u32, 1u32, 1u32);

    // -- Step 9: Run the timed loop. --
    let result = time_kernel_launches(
        func,
        &mut args,
        grid,
        block,
        shmem_bytes,
        iterations,
        skip_decisions_buf,
    );

    // -- Step 10: Cleanup. --
    for p in &allocations {
        let _ = sys::cuMemFree_v2(*p);
    }
    let _ = sys::cuModuleUnload(module);

    result
}

/// Free all tracked device allocations + unload the module, then thread
/// `err_msg` through so the original error is the one returned.
unsafe fn free_all_and(
    allocations: &[sys::CUdeviceptr],
    module: sys::CUmodule,
    err_msg: String,
) -> String {
    for p in allocations {
        let _ = sys::cuMemFree_v2(*p);
    }
    let _ = sys::cuModuleUnload(module);
    err_msg
}

/// Deterministic PRNG used to fill Q/K/V tensors. Mirrors the LCG used in
/// `csha_cuda_launch_classic.rs` so per-seed values are reproducible if a
/// reviewer wants to cross-check against the existing test harness.
///
/// Values are mapped to [-0.5, 0.5).
fn fill_seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        let f = (u as f32) / (u32::MAX as f32) - 0.5;
        out.push(f);
    }
    out
}

/// Generate a per-token segment-id vector that produces a kv-tile-disjoint
/// ratio approximately equal to `target_sparsity`.
///
/// Strategy: divide the sequence into `n_segments` equal-sized contiguous
/// segments, where `n_segments` is chosen so the expected fraction of
/// kv-tiles whose segment-id range is disjoint from each q-tile's range
/// approximates `target_sparsity`. At seq_len=4096 with block 64×64,
/// 8 segments of 512 tokens each yields roughly 50% empty tiles under
/// causal + segment masking — close to the gate fixture target.
///
/// This is intentionally simple: the bench harness's job is to drive the
/// kernel through a representative workload, not to match a specific
/// sparsity number exactly. The numerical correctness gate is the parity
/// tier (B1.5-3); the perf gate is the wall-time difference, which is
/// robust to small sparsity-target slippage.
///
/// `seed` is reserved for future shuffling/randomization of segment
/// boundaries; today it's unused but kept in the signature so call sites
/// don't change when B1.5-4 / B1.5-3 want seed-driven variation.
fn generate_segment_mask(seq_len: u32, target_sparsity: f64, _seed: u64) -> Vec<u16> {
    // Map target_sparsity in [0, 1] → segment count via a coarse table.
    // The relationship between #segments and tile-disjoint fraction depends
    // on block_kv and seq_len; for the gate fixture (block_kv=64,
    // seq_len=4096) the table below was hand-tuned to approximate the
    // target. Sensitivity fixtures (B1.5-4) at 10/50/90% will refine this.
    let n_segments: u32 = if target_sparsity <= 0.1 {
        2
    } else if target_sparsity <= 0.3 {
        4
    } else if target_sparsity <= 0.55 {
        8
    } else if target_sparsity <= 0.8 {
        16
    } else {
        32
    };
    let seg_len = (seq_len / n_segments).max(1);
    let mut out = Vec::with_capacity(seq_len as usize);
    for i in 0..seq_len {
        let seg_id = ((i / seg_len) as u16).min(n_segments as u16 - 1);
        out.push(seg_id);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_seeded_is_deterministic() {
        let a = fill_seeded(16, 42);
        let b = fill_seeded(16, 42);
        let c = fill_seeded(16, 43);
        assert_eq!(a, b, "same seed → same output");
        assert_ne!(a, c, "different seed → different output");
        for v in &a {
            assert!(v.is_finite() && (-1.0..=1.0).contains(v), "v={}", v);
        }
    }

    #[test]
    fn segment_mask_size_matches_seq_len() {
        let m = generate_segment_mask(4096, 0.5, 42);
        assert_eq!(m.len(), 4096);
        // Sparsity-50% maps to 8 segments per the table; verify the
        // segment-id pattern.
        let max = *m.iter().max().unwrap();
        assert_eq!(max, 7, "expected 8 segments (ids 0..=7)");
    }

    #[test]
    fn segment_mask_higher_sparsity_has_more_segments() {
        let low = generate_segment_mask(4096, 0.1, 42);
        let mid = generate_segment_mask(4096, 0.5, 42);
        let high = generate_segment_mask(4096, 0.9, 42);
        let n = |v: &[u16]| *v.iter().max().unwrap() as u32 + 1;
        assert!(n(&low) < n(&mid));
        assert!(n(&mid) < n(&high));
    }
}
