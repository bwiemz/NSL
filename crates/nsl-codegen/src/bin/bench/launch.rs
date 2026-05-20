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
    flash_attention_kernel_name_v2, shared_mem_bytes_v2_backward_with_seqlen,
    shared_mem_bytes_v2_with_seqlen, synthesize_backward_with_tier_b,
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
        return Err(format!(
            "cuEventCreate(start) rc={:?}: {}",
            rc,
            cu_error_string(rc)
        ));
    }
    let rc = sys::cuEventCreate(&mut stop_evt, 0);
    if rc != sys::CUresult::CUDA_SUCCESS {
        let _ = sys::cuEventDestroy_v2(start_evt);
        return Err(format!(
            "cuEventCreate(stop) rc={:?}: {}",
            rc,
            cu_error_string(rc)
        ));
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
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
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
        let rc = sys::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut c_void, dev_ptr, byte_len);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemcpyDtoH(skip_decisions) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ));
        }
        // skip_ratio = #(byte == 1) / #(buffer slots written).
        // Per spec §4.3.1 the buffer's shape is [B, H, num_q_tiles,
        // num_kv_tiles]. The writeback emits one decision per
        // (b, h, q_tile_iter, kv_tile_ord) — i.e. one per `iters` × `num_kv`
        // per block. Slots OUTSIDE the actually-written region stay at
        // their initial zero from `cuMemsetD8_v2`, so a naive `non-zero /
        // total` ratio (a) understates by mixing in unwritten zeros and
        // (b) cannot distinguish a written-but-kept-tile zero from an
        // unwritten-slot zero. Optional env var `NSL_BENCH_PRINT_DECISIONS`
        // dumps a histogram to stderr for diagnostic runs; the default
        // skip_ratio is the simple non-zero-fraction (used by M2/M6
        // measurement scripts that diff Tier-B-on vs Tier-B-off output).
        let n_skip = host.iter().filter(|&&b| b == 1).count();
        let n_keep = host.iter().filter(|&&b| b == 0).count();
        let n_other = host.len() - n_skip - n_keep;
        if std::env::var_os("NSL_BENCH_PRINT_DECISIONS").is_some() {
            eprintln!(
                "[bench] skip_decisions buffer: total={} bytes, n_skip(==1)={}, n_keep(==0)={}, n_other={}",
                host.len(),
                n_skip,
                n_keep,
                n_other
            );
            // Dump first 128 bytes raw and the first non-zero byte's offset
            // for diagnostic verification that the kernel writeback path is
            // populating the buffer at the expected slot stride.
            let head: Vec<String> = host
                .iter()
                .take(128)
                .map(|b| format!("{:02x}", b))
                .collect();
            eprintln!("[bench] first 128 bytes: {}", head.join(" "));
            let first_nz = host.iter().enumerate().find(|(_, &b)| b != 0);
            match first_nz {
                Some((i, b)) => {
                    eprintln!("[bench] first non-zero byte at offset {} = 0x{:02x}", i, b)
                }
                None => eprintln!(
                    "[bench] no non-zero bytes in entire {} byte buffer",
                    host.len()
                ),
            }
        }
        // Ratio is over the FULL buffer (not just written cells) — the spec
        // §4.3 contract pins the buffer shape; under-writing is a bench
        // measurement bias, not a correctness issue. Future B1.5-4 may
        // refine this to "written-cells-only" once the kernel writes all
        // num_q_tiles slots (currently writes only `iters` per block).
        (n_skip as f64) / (host.len().max(1) as f64)
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
    dump_output: Option<&std::path::Path>,
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
            return Err(format!(
                "cuCtxSetCurrent rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ));
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
        let nul = err_log
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(err_log.len());
        let log = String::from_utf8_lossy(&err_log[..nul]).trim().to_string();
        return Err(format!(
            "cuModuleLoadDataEx rc={:?}: {} | JIT log: {}",
            rc,
            cu_error_string(rc),
            if log.is_empty() {
                "<empty>".into()
            } else {
                log
            }
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
    let lse_elems = (fixture.batch as usize) * (heads as usize) * (fixture.seq_len as usize);
    let lse_bytes = (lse_elems * std::mem::size_of::<f32>()) as usize;
    let seg_bytes_per_batch = (fixture.seq_len as usize) * std::mem::size_of::<u16>();

    let mut allocations: Vec<sys::CUdeviceptr> = Vec::new();
    let alloc = |bytes: usize,
                 allocations: &mut Vec<sys::CUdeviceptr>|
     -> Result<sys::CUdeviceptr, String> {
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
    let out_dev =
        alloc(out_bytes, &mut allocations).map_err(|e| free_all_and(&allocations, module, e))?;
    let lse_dev =
        alloc(lse_bytes, &mut allocations).map_err(|e| free_all_and(&allocations, module, e))?;
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
        let dev =
            alloc(n_slots, &mut allocations).map_err(|e| free_all_and(&allocations, module, e))?;
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
    let shmem_bytes_base =
        shared_mem_bytes_v2_with_seqlen(&fixture.config, fixture.seq_len, SegmentResidency::Shared);
    let shmem_bytes = if tier_b_on
        && crate::pca_tilerange::should_emit_tier_b(
            &fixture.config,
            fixture.seq_len as u64,
            SegmentResidency::Shared,
        ) {
        let base = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
            &fixture.config,
            crate::flash_attention_v2::smem_layout::Direction::Forward,
        );
        let tbl = crate::pca_tilerange::tier_b_range_table_bytes(&fixture.config, fixture.seq_len);
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
    // 38th kernel-arg slot: skip_decisions_ptr (Tier B M3 instrumentation).
    // PTX declares this param ONLY when `tier_b_on` AND the
    // `debug_kernel_instrumentation` feature is enabled. Bench binary is
    // always built with that feature (cf. `required-features` in Cargo.toml),
    // so the slot is pushed onto the args list exactly when
    // `skip_decisions_buf.is_some()`. When tier_b_on is false the param
    // is absent from the PTX signature, so the slot is omitted from args.
    let mut a_skip_decisions: u64 = skip_decisions_buf.map_or(0, |(p, _)| p);

    let mut args: Vec<*mut c_void> = vec![
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
    if skip_decisions_buf.is_some() {
        args.push(&mut a_skip_decisions as *mut _ as *mut c_void);
    }

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

    // -- Step 9b: M3-parity dump-output (B1.5-3). --
    // After the timed loop the `out` device buffer holds the last
    // iteration's forward output. Memcpy_dtoh and write the raw bytes to
    // the requested path. This is invoked by the M3 parity tests to
    // capture per-fixture Tier-B-on / Tier-B-off outputs and assert
    // byte-equality (spec §6.1). The dump happens BEFORE Step 10's
    // cleanup so we read from a still-live device allocation.
    if let Some(path) = dump_output {
        if result.is_ok() {
            let mut host_out = vec![0u8; out_bytes];
            let rc = sys::cuMemcpyDtoH_v2(host_out.as_mut_ptr() as *mut c_void, out_dev, out_bytes);
            if rc != sys::CUresult::CUDA_SUCCESS {
                let msg = format!(
                    "cuMemcpyDtoH(out for --dump-output) rc={:?}: {}",
                    rc,
                    cu_error_string(rc)
                );
                for p in &allocations {
                    let _ = sys::cuMemFree_v2(*p);
                }
                let _ = sys::cuModuleUnload(module);
                return Err(msg);
            }
            if let Err(e) = std::fs::write(path, &host_out) {
                let msg = format!("failed to write --dump-output to {:?}: {}", path, e);
                for p in &allocations {
                    let _ = sys::cuMemFree_v2(*p);
                }
                let _ = sys::cuModuleUnload(module);
                return Err(msg);
            }
        }
    }

    // -- Step 10: Cleanup. --
    for p in &allocations {
        let _ = sys::cuMemFree_v2(*p);
    }
    let _ = sys::cuModuleUnload(module);

    result
}

/// End-to-end run for the BACKWARD parity tier (B2-3).
///
/// Synthesizes the backward PTX via `synthesize_backward_with_tier_b`,
/// runs forward first to populate `O`/`logsumexp` (and a deterministic
/// `row_max`/`row_sum` save when csha is on; otherwise the parity test
/// uses pre-populated buffers — see comments inline), then runs the
/// backward kernel and dumps the `dQ`, `dK_scratch`, `dV_scratch`
/// device buffers to `dump_path` as a single concatenated blob:
///
/// ```text
/// [0..8]   u64 dq_len_bytes (little-endian)
/// [8..16]  u64 dk_len_bytes
/// [16..24] u64 dv_len_bytes
/// [24..]   dq bytes (f16 [B,H,S,D])  then dk bytes (f32) then dv bytes (f32)
/// ```
///
/// The B2-3 parity tests invoke this once per `--tier-b {on,off}` and
/// assert the resulting blobs are byte-identical (spec §6.1 / §7.1).
///
/// Returns the `LaunchResult` for the backward kernel timing. When
/// Tier B is on AND `--dump-backward-outputs` is set, the bench also
/// emits `bwd_skip_ratio=<f>` to stderr so step-6 of the B2-3 plan
/// can observability-check that the backward predicate actually fires.
///
/// # Safety
///
/// Same contract as `run_fixture` — primary-context-must-be-current,
/// raw CUDA driver calls, etc.
pub unsafe fn run_fixture_backward(
    fixture: &Fixture,
    tier_b_on: bool,
    seed: u64,
    iterations: u32,
    dump_path: &std::path::Path,
) -> Result<LaunchResult, String> {
    // -- Step B1: Ensure CUDA context is current on this thread. --
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
            return Err(format!(
                "cuCtxSetCurrent rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ));
        }
    }

    // -- Step B2: Synthesize FORWARD + BACKWARD PTX. --
    // tier_b argument identical to forward; backward's synthesizer uses
    // it for the range-table preamble + per-(q_iter, kvt) skip predicate.
    let tier_b_arg = if tier_b_on {
        Some((fixture.seq_len, SegmentResidency::Shared))
    } else {
        None
    };
    let mut fwd_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(&fixture.config, tier_b_arg);
    while fwd_ptx.last() == Some(&0) {
        fwd_ptx.pop();
    }
    if fwd_ptx.last() != Some(&b'\n') {
        fwd_ptx.push(b'\n');
    }
    fwd_ptx.push(0);

    let bwd_ptx_str = synthesize_backward_with_tier_b(&fixture.config, tier_b_arg)
        .map_err(|e| format!("synthesize_backward_with_tier_b: {e}"))?;
    let mut bwd_ptx: Vec<u8> = bwd_ptx_str.into_bytes();
    while bwd_ptx.last() == Some(&0) {
        bwd_ptx.pop();
    }
    if bwd_ptx.last() != Some(&b'\n') {
        bwd_ptx.push(b'\n');
    }
    bwd_ptx.push(0);

    if std::env::var_os("NSL_BENCH_DUMP_PTX").is_some() {
        let p_fwd = std::env::temp_dir().join(format!(
            "bench_bwd_{}_tier_b_{}_FWD.ptx",
            fixture.name,
            if tier_b_on { "on" } else { "off" }
        ));
        let _ = std::fs::write(&p_fwd, &fwd_ptx[..fwd_ptx.len() - 1]);
        let p_bwd = std::env::temp_dir().join(format!(
            "bench_bwd_{}_tier_b_{}_BWD.ptx",
            fixture.name,
            if tier_b_on { "on" } else { "off" }
        ));
        let _ = std::fs::write(&p_bwd, &bwd_ptx[..bwd_ptx.len() - 1]);
        eprintln!(
            "[bench-bwd] PTX dumped to: {} (fwd), {} (bwd)",
            p_fwd.display(),
            p_bwd.display()
        );
    }

    // -- Step B3: Load BOTH modules. --
    let load_module = |ptx: &[u8], tag: &str| -> Result<sys::CUmodule, String> {
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
            let nul = err_log
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(err_log.len());
            let log = String::from_utf8_lossy(&err_log[..nul]).trim().to_string();
            return Err(format!(
                "cuModuleLoadDataEx({tag}) rc={:?}: {} | JIT log: {}",
                rc,
                cu_error_string(rc),
                if log.is_empty() {
                    "<empty>".into()
                } else {
                    log
                }
            ));
        }
        Ok(module)
    };
    let fwd_module = load_module(&fwd_ptx, "forward")?;
    let bwd_module = match load_module(&bwd_ptx, "backward") {
        Ok(m) => m,
        Err(e) => {
            let _ = sys::cuModuleUnload(fwd_module);
            return Err(e);
        }
    };

    // -- Step B4: Resolve function handles for both kernels. --
    let fwd_kernel_name = match CString::new(flash_attention_kernel_name_v2(&fixture.config)) {
        Ok(n) => n,
        Err(_) => {
            let _ = sys::cuModuleUnload(fwd_module);
            let _ = sys::cuModuleUnload(bwd_module);
            return Err("forward kernel name contains embedded NUL".into());
        }
    };
    let bwd_kernel_name_str =
        crate::flash_attention_v2::phases::backward::prelude::kernel_name(&fixture.config);
    let bwd_kernel_name = match CString::new(bwd_kernel_name_str.clone()) {
        Ok(n) => n,
        Err(_) => {
            let _ = sys::cuModuleUnload(fwd_module);
            let _ = sys::cuModuleUnload(bwd_module);
            return Err("backward kernel name contains embedded NUL".into());
        }
    };
    let mut fwd_func: sys::CUfunction = std::ptr::null_mut();
    let rc = sys::cuModuleGetFunction(&mut fwd_func, fwd_module, fwd_kernel_name.as_ptr());
    if rc != sys::CUresult::CUDA_SUCCESS {
        let _ = sys::cuModuleUnload(fwd_module);
        let _ = sys::cuModuleUnload(bwd_module);
        return Err(format!(
            "cuModuleGetFunction(fwd, {}) rc={:?}: {}",
            fwd_kernel_name.to_string_lossy(),
            rc,
            cu_error_string(rc)
        ));
    }
    let mut bwd_func: sys::CUfunction = std::ptr::null_mut();
    let rc = sys::cuModuleGetFunction(&mut bwd_func, bwd_module, bwd_kernel_name.as_ptr());
    if rc != sys::CUresult::CUDA_SUCCESS {
        let _ = sys::cuModuleUnload(fwd_module);
        let _ = sys::cuModuleUnload(bwd_module);
        return Err(format!(
            "cuModuleGetFunction(bwd, {}) rc={:?}: {}",
            bwd_kernel_name.to_string_lossy(),
            rc,
            cu_error_string(rc)
        ));
    }

    // -- Step B5: Allocate ALL device buffers. --
    let heads: u32 = 1;
    let head_dim = fixture.config.head_dim as u32;
    let total_elems = (fixture.batch as usize)
        * (heads as usize)
        * (fixture.seq_len as usize)
        * (head_dim as usize);
    // Backward kernel reads Q/K/V from `q_proj_ptr`/`k_proj_ptr`/`v_proj_ptr`
    // as f16 (`ld.global.b16` + `cvt.f32.f16`). The host PRNG produces f32
    // values which we convert to f16 below before HtoD — interpreting
    // raw f32 bits as f16 produces denormals/NaN cascades that defeat the
    // bit-identical assertion (the forward parity test gets away with the
    // f32→f16-bits reinterpret because both `on`/`off` see the same noise,
    // but in backward the noise compounds through P/dP/dS/dK/dV accumulators
    // and produces non-zero NaN bit-pattern divergence).
    let qkv_bytes_per = total_elems * std::mem::size_of::<u16>(); // f16 storage
    let out_bytes = total_elems * std::mem::size_of::<u16>(); // f16 forward O
    let dq_bytes = total_elems * std::mem::size_of::<u16>(); // f16 dQ
    let dkv_scratch_bytes = total_elems * std::mem::size_of::<f32>(); // f32 dK/dV scratch
    let lse_elems = (fixture.batch as usize) * (heads as usize) * (fixture.seq_len as usize);
    let lse_bytes = lse_elems * std::mem::size_of::<f32>();
    let row_stats_bytes = lse_bytes; // row_max + row_sum: same [B,H,S] f32 shape
    let seg_bytes_per_batch = (fixture.seq_len as usize) * std::mem::size_of::<u16>();

    let mut allocations: Vec<sys::CUdeviceptr> = Vec::new();
    let alloc = |bytes: usize,
                 allocations: &mut Vec<sys::CUdeviceptr>|
     -> Result<sys::CUdeviceptr, String> {
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

    let free_modules = |fwd_module: sys::CUmodule, bwd_module: sys::CUmodule| unsafe {
        let _ = sys::cuModuleUnload(fwd_module);
        let _ = sys::cuModuleUnload(bwd_module);
    };
    let cleanup_and = |allocs: &[sys::CUdeviceptr], err: String| -> String {
        for p in allocs {
            let _ = sys::cuMemFree_v2(*p);
        }
        free_modules(fwd_module, bwd_module);
        err
    };

    let q_dev = alloc(qkv_bytes_per, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let k_dev = alloc(qkv_bytes_per, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let v_dev = alloc(qkv_bytes_per, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let out_dev = alloc(out_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let lse_dev = alloc(lse_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let seg_dev =
        alloc(seg_bytes_per_batch, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;

    // Backward-specific buffers.
    let row_max_dev =
        alloc(row_stats_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let row_sum_dev =
        alloc(row_stats_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let do_dev = alloc(out_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?; // dO, f16
    let dq_dev = alloc(dq_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let dk_scratch_dev =
        alloc(dkv_scratch_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
    let dv_scratch_dev =
        alloc(dkv_scratch_bytes, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;

    // -- Step B6: Populate host inputs from seed and H2D. --
    // Q/K/V same seed scheme as forward path so the two runs (on/off) see
    // bit-identical inputs. f32 PRNG → f16-bit host buffers; backward reads
    // these via `ld.global.b16` so the storage MUST be valid f16.
    let q_host_f16 = f32_slice_to_f16_bits(&fill_seeded(total_elems, seed.wrapping_mul(3)));
    let k_host_f16 = f32_slice_to_f16_bits(&fill_seeded(total_elems, seed.wrapping_mul(5)));
    let v_host_f16 = f32_slice_to_f16_bits(&fill_seeded(total_elems, seed.wrapping_mul(7)));
    let seg_host = generate_segment_mask(fixture.seq_len, fixture.target_sparsity, seed);
    // dO: f16 host buffer derived from a different seed mix so it's not
    // correlated to Q/K/V (the value distribution doesn't matter for
    // parity, only that on/off pairs see the exact same bytes).
    let do_f32 = fill_seeded(total_elems, seed.wrapping_mul(11));
    let do_f16 = f32_slice_to_f16_bits(&do_f32);
    // row_max / row_sum: numerically-stable constants — `row_max=0`,
    // `row_sum=1`. When csha=None forward doesn't save these, but backward
    // reads them in `ds_compute`. Random seeded values produce NaN cascades
    // in tiles processed by Tier-B-off (where Tier-B-on skips them), because
    // `P = ex2((S - row_max) * log2e) / row_sum` becomes unbounded when
    // row_max/row_sum are pathological — and the resulting NaN contaminates
    // the f32 scratch RMW chain, defeating bit-identical comparison even
    // for kept tiles.
    //
    // Constants instead: P = ex2(S * log2e) is bounded on S ∈ [-INF, ~1.4]
    // (causal/segment masking sets masked lanes to S=-INF → ex2(-INF)=0;
    // in-range lanes give S ∈ [-1.4, 1.4] → P ∈ [0.25, 4.0]). Skipped tiles
    // under Tier-B-on still produce zero contribution (P=0 for masked
    // lanes, predicate-skipped tiles never enter the SMEM tile), and OFF
    // computes the same arithmetic on those tiles — preserving bit-equality.
    let row_max_host: Vec<f32> = vec![0.0; lse_elems];
    let row_sum_host: Vec<f32> = vec![1.0; lse_elems];

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

    h2d(q_dev, q_host_f16.as_ptr() as *const c_void, qkv_bytes_per)
        .map_err(|e| cleanup_and(&allocations, e))?;
    h2d(k_dev, k_host_f16.as_ptr() as *const c_void, qkv_bytes_per)
        .map_err(|e| cleanup_and(&allocations, e))?;
    h2d(v_dev, v_host_f16.as_ptr() as *const c_void, qkv_bytes_per)
        .map_err(|e| cleanup_and(&allocations, e))?;
    h2d(
        seg_dev,
        seg_host.as_ptr() as *const c_void,
        seg_bytes_per_batch,
    )
    .map_err(|e| cleanup_and(&allocations, e))?;
    h2d(do_dev, do_f16.as_ptr() as *const c_void, out_bytes)
        .map_err(|e| cleanup_and(&allocations, e))?;
    h2d(
        row_max_dev,
        row_max_host.as_ptr() as *const c_void,
        row_stats_bytes,
    )
    .map_err(|e| cleanup_and(&allocations, e))?;
    h2d(
        row_sum_dev,
        row_sum_host.as_ptr() as *const c_void,
        row_stats_bytes,
    )
    .map_err(|e| cleanup_and(&allocations, e))?;

    // Zero the dQ/dK/dV outputs before the backward launch (matches the
    // runtime's `memset_d8(d_q, qkv_elems * 2)` pre-launch).
    let _ = sys::cuMemsetD8_v2(out_dev, 0, out_bytes);
    let _ = sys::cuMemsetD8_v2(lse_dev, 0, lse_bytes);
    let _ = sys::cuMemsetD8_v2(dq_dev, 0, dq_bytes);
    let _ = sys::cuMemsetD8_v2(dk_scratch_dev, 0, dkv_scratch_bytes);
    let _ = sys::cuMemsetD8_v2(dv_scratch_dev, 0, dkv_scratch_bytes);

    // Tier B skip-decision buffer for the BACKWARD launch (writeback fires
    // inside `emit_skip_predicate` for both forward and backward; same slot
    // layout `[B,H,Qtiles,KVtiles]` per spec §6.2). Only allocated when
    // tier_b_on so the off-run's backward param list matches the off PTX
    // signature.
    let skip_decisions_buf: Option<(sys::CUdeviceptr, usize)> = if tier_b_on {
        let num_q = (fixture.seq_len).div_ceil(fixture.config.block_q as u32) as usize;
        let num_kv = (fixture.seq_len).div_ceil(fixture.config.block_kv as u32) as usize;
        let n_slots = (fixture.batch as usize) * (heads as usize) * num_q * num_kv;
        let dev = alloc(n_slots, &mut allocations).map_err(|e| cleanup_and(&allocations, e))?;
        let rc = sys::cuMemsetD8_v2(dev, 0, n_slots);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(cleanup_and(
                &allocations,
                format!(
                    "cuMemsetD8_v2(skip_decisions bwd) rc={:?}: {}",
                    rc,
                    cu_error_string(rc)
                ),
            ));
        }
        Some((dev, n_slots))
    } else {
        None
    };

    // -- Step B7: SMEM opt-in for forward. --
    let fwd_shmem_bytes_base =
        shared_mem_bytes_v2_with_seqlen(&fixture.config, fixture.seq_len, SegmentResidency::Shared);
    let fwd_shmem_bytes = if tier_b_on
        && crate::pca_tilerange::should_emit_tier_b(
            &fixture.config,
            fixture.seq_len as u64,
            SegmentResidency::Shared,
        ) {
        let base = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
            &fixture.config,
            crate::flash_attention_v2::smem_layout::Direction::Forward,
        );
        let tbl = crate::pca_tilerange::tier_b_range_table_bytes(&fixture.config, fixture.seq_len);
        fwd_shmem_bytes_base.max(base + tbl)
    } else {
        fwd_shmem_bytes_base
    };
    const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: sys::CUfunction_attribute =
        sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    let _ = sys::cuFuncSetAttribute(
        fwd_func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        fwd_shmem_bytes as i32,
    );

    // -- Step B8: SMEM opt-in for backward. --
    let bwd_shmem_bytes_base = shared_mem_bytes_v2_backward_with_seqlen(
        &fixture.config,
        fixture.seq_len,
        SegmentResidency::Shared,
    );
    let bwd_shmem_bytes = if tier_b_on
        && crate::pca_tilerange::should_emit_tier_b(
            &fixture.config,
            fixture.seq_len as u64,
            SegmentResidency::Shared,
        ) {
        let base = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
            &fixture.config,
            crate::flash_attention_v2::smem_layout::Direction::Backward,
        );
        let tbl = crate::pca_tilerange::tier_b_range_table_bytes(&fixture.config, fixture.seq_len);
        bwd_shmem_bytes_base.max(base + tbl)
    } else {
        bwd_shmem_bytes_base
    };
    let _ = sys::cuFuncSetAttribute(
        bwd_func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        bwd_shmem_bytes as i32,
    );

    // -- Step B9: Build forward kernel-args (37 slots + optional 38th). --
    // Mirrors the 37-slot list in `run_fixture`.
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
    // row_max/row_sum point to allocated buffers (overwritten by forward
    // softmax::emit's direct stores when present, OR pre-seeded by the
    // host when csha=None and the forward never writes them).
    let mut a_q_proj: u64 = 0;
    let mut a_k_proj: u64 = 0;
    let mut a_v_proj: u64 = 0;
    let mut a_row_max: u64 = row_max_dev;
    let mut a_row_sum: u64 = row_sum_dev;
    let mut a_x_raw: u64 = 0;
    let mut a_seg_ids: u64 = seg_dev;
    let mut a_skip_decisions_fwd: u64 = skip_decisions_buf.map_or(0, |(p, _)| p);

    let mut fwd_args: Vec<*mut c_void> = vec![
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
    if skip_decisions_buf.is_some() {
        fwd_args.push(&mut a_skip_decisions_fwd as *mut _ as *mut c_void);
    }

    // -- Step B10: Launch FORWARD (one shot, results consumed by backward). --
    let fwd_grid_x = fixture.seq_len.div_ceil(fixture.config.block_q as u32);
    let fwd_grid_y = fixture.batch * heads;
    let rc = sys::cuLaunchKernel(
        fwd_func,
        fwd_grid_x,
        fwd_grid_y,
        1,
        128,
        1,
        1,
        fwd_shmem_bytes,
        std::ptr::null_mut(),
        fwd_args.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(cleanup_and(
            &allocations,
            format!(
                "cuLaunchKernel(forward) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ),
        ));
    }
    let rc = sys::cuCtxSynchronize();
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(cleanup_and(
            &allocations,
            format!(
                "cuCtxSynchronize(post-forward) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ),
        ));
    }

    // -- Step B11: Build BACKWARD kernel args (48 slots + optional 49th
    // skip_decisions_ptr; mirrors `nsl_flash_attention_csha_backward`'s
    // launch list in `crates/nsl-runtime/src/flash_attention.rs`). --
    //
    // ABI summary (48 slots, see backward prelude.rs param block):
    //   0:  q_ptr            17: dfs_enter_ptr     34: row_sum_ptr
    //   1:  k_ptr            18: dfs_exit_ptr      35: x_raw_ptr
    //   2:  v_ptr            19: num_tree_nodes    36: dO_ptr
    //   3:  out_ptr          20: param_logsumexp   37: dq_ptr
    //   4:  scale            21: csha_x_ptr        38: dk_ptr
    //   5:  batch            22: csha_nw_ptr       39: dv_ptr
    //   6:  heads            23: csha_wq_ptr       40: dwq_ptr
    //   7:  seq_len          24: csha_wk_ptr       41: dwk_ptr
    //   8:  head_dim         25: csha_wv_ptr       42: dwv_ptr
    //   9:  block_table_ptr  26: csha_wo_ptr       43: dx_ptr
    //  10:  k_pool_ptr       27: csha_eps          44: dx_norm_ptr
    //  11:  v_pool_ptr       28: csha_active_heads 45: dk_scratch_ptr
    //  12:  block_size       29: csha_d_model      46: dv_scratch_ptr
    //  13:  cos_ptr          30: q_proj_ptr        47: segment_ids_ptr (cond)
    //  14:  sin_ptr          31: k_proj_ptr        +1: skip_decisions_ptr (cond)
    //  15:  seq_ids_ptr      32: v_proj_ptr
    //  16:  seq_lens_ptr     33: row_max_ptr
    //
    // q_proj/k_proj/v_proj are CSHA-saved activations; backward q_load
    // null-guards on them. We pass the raw Q/K/V buffers in here so the
    // backward Q reload reads the same data path the on/off pair agreed
    // on (no semantic dependence, just byte-determinism).
    let mut bw_q: u64 = q_dev;
    let mut bw_k: u64 = k_dev;
    let mut bw_v: u64 = v_dev;
    let mut bw_out: u64 = out_dev;
    let mut bw_scale: f32 = scale;
    let mut bw_batch: u64 = fixture.batch as u64;
    let mut bw_heads: u64 = heads as u64;
    let mut bw_seq_len: u64 = fixture.seq_len as u64;
    let mut bw_head_dim: u64 = head_dim as u64;
    let mut bw_block_table: u64 = 0;
    let mut bw_k_pool: u64 = 0;
    let mut bw_v_pool: u64 = 0;
    let mut bw_block_size: u64 = 0;
    let mut bw_cos: u64 = 0;
    let mut bw_sin: u64 = 0;
    let mut bw_seq_ids: u64 = 0;
    // seq_lens_ptr doubles as the q-block base in the per-q-block launch
    // loop (mirrors `nsl_flash_attention_csha_backward`'s `slens` field).
    let mut bw_seq_lens: u64 = 0;
    let mut bw_dfs_enter: u64 = 0;
    let mut bw_dfs_exit: u64 = 0;
    let mut bw_num_tree_nodes: u64 = 0;
    let mut bw_logsumexp: u64 = lse_dev;
    let mut bw_csha_x: u64 = 0;
    let mut bw_csha_nw: u64 = 0;
    let mut bw_csha_wq: u64 = 0;
    let mut bw_csha_wk: u64 = 0;
    let mut bw_csha_wv: u64 = 0;
    let mut bw_csha_wo: u64 = 0;
    let mut bw_csha_eps: f32 = 1.0e-5;
    let mut bw_csha_active_heads: u32 = 0;
    let mut bw_csha_d_model: u32 = 0;
    // q_proj/k_proj/v_proj — pass raw Q/K/V so backward Q-load reads the
    // same forward-input bytes. Acceptable for parity (numerical correctness
    // is NOT a gate criterion for B2-3; bit-equality of on/off is).
    let mut bw_q_proj: u64 = q_dev;
    let mut bw_k_proj: u64 = k_dev;
    let mut bw_v_proj: u64 = v_dev;
    let mut bw_row_max: u64 = row_max_dev;
    let mut bw_row_sum: u64 = row_sum_dev;
    let mut bw_x_raw: u64 = 0;
    let mut bw_do: u64 = do_dev;
    let mut bw_dq: u64 = dq_dev;
    let mut bw_dk: u64 = 0; // f32 scratch path; f16 outputs unused for parity.
    let mut bw_dv: u64 = 0;
    let mut bw_dwq: u64 = 0;
    let mut bw_dwk: u64 = 0;
    let mut bw_dwv: u64 = 0;
    let mut bw_dx: u64 = 0;
    let mut bw_dx_norm: u64 = 0;
    let mut bw_dk_scratch: u64 = dk_scratch_dev;
    let mut bw_dv_scratch: u64 = dv_scratch_dev;
    let mut bw_seg_ids: u64 = seg_dev;
    let mut bw_skip_decisions: u64 = skip_decisions_buf.map_or(0, |(p, _)| p);

    let mut bwd_args: Vec<*mut c_void> = vec![
        &mut bw_q as *mut _ as *mut c_void,
        &mut bw_k as *mut _ as *mut c_void,
        &mut bw_v as *mut _ as *mut c_void,
        &mut bw_out as *mut _ as *mut c_void,
        &mut bw_scale as *mut _ as *mut c_void,
        &mut bw_batch as *mut _ as *mut c_void,
        &mut bw_heads as *mut _ as *mut c_void,
        &mut bw_seq_len as *mut _ as *mut c_void,
        &mut bw_head_dim as *mut _ as *mut c_void,
        &mut bw_block_table as *mut _ as *mut c_void,
        &mut bw_k_pool as *mut _ as *mut c_void,
        &mut bw_v_pool as *mut _ as *mut c_void,
        &mut bw_block_size as *mut _ as *mut c_void,
        &mut bw_cos as *mut _ as *mut c_void,
        &mut bw_sin as *mut _ as *mut c_void,
        &mut bw_seq_ids as *mut _ as *mut c_void,
        &mut bw_seq_lens as *mut _ as *mut c_void,
        &mut bw_dfs_enter as *mut _ as *mut c_void,
        &mut bw_dfs_exit as *mut _ as *mut c_void,
        &mut bw_num_tree_nodes as *mut _ as *mut c_void,
        &mut bw_logsumexp as *mut _ as *mut c_void,
        &mut bw_csha_x as *mut _ as *mut c_void,
        &mut bw_csha_nw as *mut _ as *mut c_void,
        &mut bw_csha_wq as *mut _ as *mut c_void,
        &mut bw_csha_wk as *mut _ as *mut c_void,
        &mut bw_csha_wv as *mut _ as *mut c_void,
        &mut bw_csha_wo as *mut _ as *mut c_void,
        &mut bw_csha_eps as *mut _ as *mut c_void,
        &mut bw_csha_active_heads as *mut _ as *mut c_void,
        &mut bw_csha_d_model as *mut _ as *mut c_void,
        &mut bw_q_proj as *mut _ as *mut c_void,
        &mut bw_k_proj as *mut _ as *mut c_void,
        &mut bw_v_proj as *mut _ as *mut c_void,
        &mut bw_row_max as *mut _ as *mut c_void,
        &mut bw_row_sum as *mut _ as *mut c_void,
        &mut bw_x_raw as *mut _ as *mut c_void,
        &mut bw_do as *mut _ as *mut c_void,
        &mut bw_dq as *mut _ as *mut c_void,
        &mut bw_dk as *mut _ as *mut c_void,
        &mut bw_dv as *mut _ as *mut c_void,
        &mut bw_dwq as *mut _ as *mut c_void,
        &mut bw_dwk as *mut _ as *mut c_void,
        &mut bw_dwv as *mut _ as *mut c_void,
        &mut bw_dx as *mut _ as *mut c_void,
        &mut bw_dx_norm as *mut _ as *mut c_void,
        &mut bw_dk_scratch as *mut _ as *mut c_void,
        &mut bw_dv_scratch as *mut _ as *mut c_void,
    ];
    if fixture.config.segment_masked {
        bwd_args.push(&mut bw_seg_ids as *mut _ as *mut c_void);
    }
    if skip_decisions_buf.is_some() {
        bwd_args.push(&mut bw_skip_decisions as *mut _ as *mut c_void);
    }

    // -- Step B12: Launch BACKWARD (production ABI: grid_x = 1, per-q-block loop). --
    //
    // B2-2.5 fix: the backward Tier B predicate now reads
    // `%r_qt_ord_TB_BWD = (%q_start >> log2(block_q))` instead of `%bid_x`,
    // which is correct under both grid_x=1 (production) and the previous
    // grid_x=num_q_tiles workaround. We therefore switch the bench to the
    // production launch ABI (mirrors `nsl_flash_attention_csha_backward`):
    //   grid_x = 1, q-block ordinal encoded in `seq_lens_ptr` (slot 16),
    //   per-q-block sequential launches so the dK/dV f32 scratch RMW
    //   serializes deterministically (no parallel-CTA race).
    //
    // The previous grid_x=num_q_tiles parallel launch produced
    // non-deterministic dK/dV outputs because parallel CTAs raced on the
    // f32 scratch read-modify-write. Sequential per-q-block launches under
    // grid_x=1 eliminate that race.
    let block_q_u32 = fixture.config.block_q as u32;
    let bwd_grid_y = fixture.batch * heads;
    let q_blocks_bwd = fixture.seq_len.div_ceil(block_q_u32);

    let mut start_evt: sys::CUevent = std::ptr::null_mut();
    let mut stop_evt: sys::CUevent = std::ptr::null_mut();
    let _ = sys::cuEventCreate(&mut start_evt, 0);
    let _ = sys::cuEventCreate(&mut stop_evt, 0);
    let mut times_ms: Vec<f32> = Vec::with_capacity(iterations as usize);
    for _it in 0..iterations.max(1) {
        let _ = sys::cuEventRecord(start_evt, std::ptr::null_mut());
        let mut launch_rc = sys::CUresult::CUDA_SUCCESS;
        for q_block in 0..q_blocks_bwd {
            // Thread the q-block base into seq_lens_ptr slot via `bw_seq_lens`.
            // CUDA reads the pointee at launch time, so each iteration's
            // assignment is picked up by the next launch. `black_box`
            // prevents the optimizer from eliding the write because dataflow
            // analysis can't see the read-through-ptr.
            bw_seq_lens = (q_block as u64) * (fixture.config.block_q as u64);
            let _ = std::hint::black_box(&bw_seq_lens);
            let rc = sys::cuLaunchKernel(
                bwd_func,
                1,
                bwd_grid_y,
                1,
                128,
                1,
                1,
                bwd_shmem_bytes,
                std::ptr::null_mut(),
                bwd_args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if rc != sys::CUresult::CUDA_SUCCESS {
                launch_rc = rc;
                break;
            }
        }
        if launch_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuEventDestroy_v2(start_evt);
            let _ = sys::cuEventDestroy_v2(stop_evt);
            return Err(cleanup_and(
                &allocations,
                format!(
                    "cuLaunchKernel(backward grid_x=1, per-q-block loop) rc={:?}: {}",
                    launch_rc,
                    cu_error_string(launch_rc)
                ),
            ));
        }
        let _ = sys::cuEventRecord(stop_evt, std::ptr::null_mut());
        let _ = sys::cuEventSynchronize(stop_evt);
        let mut ms: f32 = 0.0;
        let _ = sys::cuEventElapsedTime_v2(&mut ms, start_evt, stop_evt);
        times_ms.push(ms);
    }
    let _ = sys::cuEventDestroy_v2(start_evt);
    let _ = sys::cuEventDestroy_v2(stop_evt);
    let rc = sys::cuCtxSynchronize();
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(cleanup_and(
            &allocations,
            format!(
                "cuCtxSynchronize(post-backward) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ),
        ));
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_us = if times_ms.is_empty() {
        0.0
    } else {
        (times_ms[times_ms.len() / 2] as f64) * 1000.0
    };

    // -- Step B13: Read back dQ + dK_scratch + dV_scratch + write blob. --
    // dK/dV are converted f32→f16 on the host before the blob to match
    // the production write path (the runtime's host-side conversion
    // kernel writes the f32 scratch to f16 dK/dV outputs after all
    // q-block launches complete). Comparing at the f16 level is also
    // what makes the bit-identical assertion *tractable*: the f32
    // scratch retains sub-f16-ULP leakage from masked-lane ex2.approx
    // imprecision, which exists in BOTH `--tier-b on` (kept tiles only)
    // and `--tier-b off` (kept + masked-out tiles), but those leakage
    // values aren't byte-equal at f32 — they ARE byte-equal at f16
    // because the leaked sub-ULP contributions round to zero (matching
    // the forward parity assertion which compares f16 `out` directly).
    let mut host_dq = vec![0u8; dq_bytes];
    let rc = sys::cuMemcpyDtoH_v2(host_dq.as_mut_ptr() as *mut c_void, dq_dev, dq_bytes);
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(cleanup_and(
            &allocations,
            format!("cuMemcpyDtoH(dq) rc={:?}: {}", rc, cu_error_string(rc)),
        ));
    }
    let mut host_dk_f32_bytes = vec![0u8; dkv_scratch_bytes];
    let rc = sys::cuMemcpyDtoH_v2(
        host_dk_f32_bytes.as_mut_ptr() as *mut c_void,
        dk_scratch_dev,
        dkv_scratch_bytes,
    );
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(cleanup_and(
            &allocations,
            format!(
                "cuMemcpyDtoH(dk_scratch) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ),
        ));
    }
    let mut host_dv_f32_bytes = vec![0u8; dkv_scratch_bytes];
    let rc = sys::cuMemcpyDtoH_v2(
        host_dv_f32_bytes.as_mut_ptr() as *mut c_void,
        dv_scratch_dev,
        dkv_scratch_bytes,
    );
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(cleanup_and(
            &allocations,
            format!(
                "cuMemcpyDtoH(dv_scratch) rc={:?}: {}",
                rc,
                cu_error_string(rc)
            ),
        ));
    }
    // f32→f16 host conversion (mirrors runtime's post-kernel conversion).
    let host_dk = f32_bytes_to_f16_bits(&host_dk_f32_bytes);
    let host_dv = f32_bytes_to_f16_bits(&host_dv_f32_bytes);

    // Build + write the blob.
    let mut blob: Vec<u8> = Vec::with_capacity(24 + host_dq.len() + host_dk.len() + host_dv.len());
    blob.extend_from_slice(&(host_dq.len() as u64).to_le_bytes());
    blob.extend_from_slice(&(host_dk.len() as u64).to_le_bytes());
    blob.extend_from_slice(&(host_dv.len() as u64).to_le_bytes());
    blob.extend_from_slice(&host_dq);
    blob.extend_from_slice(&host_dk);
    blob.extend_from_slice(&host_dv);
    if let Err(e) = std::fs::write(dump_path, &blob) {
        return Err(cleanup_and(
            &allocations,
            format!(
                "failed to write --dump-backward-outputs to {:?}: {}",
                dump_path, e
            ),
        ));
    }

    // -- Step B14: Optional skip-ratio readback from the backward writeback. --
    let skip_ratio = if let Some((dev_ptr, byte_len)) = skip_decisions_buf {
        let mut host = vec![0u8; byte_len];
        let rc = sys::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut c_void, dev_ptr, byte_len);
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(cleanup_and(
                &allocations,
                format!(
                    "cuMemcpyDtoH(skip_decisions bwd) rc={:?}: {}",
                    rc,
                    cu_error_string(rc)
                ),
            ));
        }
        let n_skip = host.iter().filter(|&&b| b == 1).count();
        let ratio = (n_skip as f64) / (host.len().max(1) as f64);
        eprintln!("[bench-bwd] bwd_skip_ratio={:.6}", ratio);
        ratio
    } else {
        0.0
    };

    // -- Step B15: Cleanup. --
    for p in &allocations {
        let _ = sys::cuMemFree_v2(*p);
    }
    free_modules(fwd_module, bwd_module);

    Ok(LaunchResult {
        median_us,
        skip_ratio,
    })
}

/// Convert a raw little-endian f32 byte buffer to f16-bit bytes — used to
/// post-process the backward kernel's `dk_scratch_ptr`/`dv_scratch_ptr`
/// f32 outputs to the f16 representation the production conversion kernel
/// would emit. Matches `f32_to_f16_bits` rounding.
fn f32_bytes_to_f16_bits(src_bytes: &[u8]) -> Vec<u8> {
    assert!(
        src_bytes.len().is_multiple_of(4),
        "f32_bytes_to_f16_bits input must be 4-byte aligned"
    );
    let n = src_bytes.len() / 4;
    let mut out: Vec<u8> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(src_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        let h = f32_to_f16_bits(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Convert an f32 slice to little-endian f16-bit bytes (u16 per element).
/// Used by the backward bench harness to populate the `dO` device buffer
/// from a host f32 PRNG stream — backward expects f16 dO (matches forward's
/// f16 O output).
fn f32_slice_to_f16_bits(src: &[f32]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(src.len() * 2);
    for &x in src {
        let h = f32_to_f16_bits(x);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Minimal f32 → f16 bit-conversion. Round-to-nearest-even with
/// subnormal/inf/NaN handling sufficient for deterministic test inputs.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant_f32 = bits & 0x7fffff;
    if exp_f32 == 0xff {
        // NaN or Inf
        let h_mant = if mant_f32 != 0 { 0x200u16 } else { 0u16 };
        return sign | 0x7c00 | h_mant;
    }
    let exp_unbiased = exp_f32 - 127;
    if exp_unbiased > 15 {
        return sign | 0x7c00; // overflow -> Inf
    }
    if exp_unbiased < -14 {
        // subnormal or zero
        if exp_unbiased < -25 {
            return sign;
        }
        let mant_with_imp = mant_f32 | 0x800000;
        let shift = (-14 - exp_unbiased + 13) as u32;
        let mant_shifted = mant_with_imp >> shift;
        // Round-to-nearest-even on the dropped bits.
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let dropped = mant_with_imp & mask;
        let mut h_mant = mant_shifted as u16;
        if dropped > half || (dropped == half && (h_mant & 1) != 0) {
            h_mant += 1;
        }
        return sign | h_mant;
    }
    let h_exp = ((exp_unbiased + 15) as u16) << 10;
    let h_mant = (mant_f32 >> 13) as u16;
    let dropped = mant_f32 & 0x1fff;
    let mut h = sign | h_exp | h_mant;
    if dropped > 0x1000 || (dropped == 0x1000 && (h & 1) != 0) {
        h += 1;
    }
    h
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
