//! V-Bii-SMEM probe for the PCA Tier B Planner spec (single-emission
//! conservative-max sub-variant resolution).
//!
//! Per `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §3,
//! this probe characterizes the **SMEM feasibility envelope** for the
//! single-emission Option B variants. Tier B's range tables are sized at
//! codegen via `compute_range_table_bytes(seq_len, block_q, block_kv)` (see
//! `crates/nsl-codegen/src/pca_tilerange.rs`). Single-emission requires
//! baking a conservative-max `seq_len` into the kernel's SMEM allocation;
//! this probe measures whether the resulting allocation fits the 99 KB
//! Blackwell `.shared` cap (and the 100 KB Ampere cap) across the sweep.
//!
//! Mirrors the original SMEM probe at `crates/nsl-codegen/tests/tier_b_smem_probe.rs`
//! (revision spec §1 / `2026-05-12-pca-tier-b-revision-design.md`) for kernel
//! emission, sentinel-write+readback discipline, ptxas+driver outcome
//! classification, and per-config row recording.
//!
//! Sweep: `MAX ∈ {4096, 8192, 16384}` × `block ∈ {32, 64}` × `arch ∈ {sm_80, sm_120}`
//! = 12 configurations. Outcomes are classified per §3.4 five-outcome decision
//! matrix; SMEM headroom % is recorded per §3.5 ladders.
//!
//! Findings doc: `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
//!
//! Run:
//! ```text
//! cargo test -p nsl-codegen --features cuda --release \
//!   --test tier_b_bii_smem_probe probe_full_sweep -- --nocapture --test-threads=1
//! ```
//! `--test-threads=1` because CUDA primary contexts are thread-local; parallel
//! configs collide on the primary context.

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::pca_tilerange::compute_range_table_bytes;

// Tier-B-off baseline static SMEM region size, matching the original probe's
// largest static configuration (revision spec §1 / `tier_b_smem_probe.rs`).
// The probe kernel uses this as the static `.shared` allocation for
// `seg_smem[]`; the Tier-B-on extern region is sized per config.
const SEG_SMEM_STATIC_BYTES: u32 = 8192;

// SMEM caps in bytes per architecture, per spec §3.1 and revision-spec §1
// findings. Blackwell (sm_120) tops out at 99 KB; Ampere (sm_80) at 100 KB.
fn smem_cap_bytes(sm: u32) -> u32 {
    match sm {
        120 => 99 * 1024,
        _ => 100 * 1024,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeOutcome {
    Pass,
    PtxasRejected(String),
    LaunchFailed(String),
    ValueCorruption { expected: u8, found: u8 },
}

#[derive(Debug)]
#[allow(dead_code)] // Fields surface via the eprintln! sweep summary, not direct reads.
struct ProbeResult {
    max_seq_len: u32,
    block: u32,
    target_sm: u32,
    seg_smem_bytes: u32,
    tier_b_extern_bytes: u32,
    total_smem_bytes: u32,
    cap_bytes: u32,
    utilization_pct: f32,
    outcome: ProbeOutcome,
}

/// Build the PTX for the Tier-B-Bii SMEM probe kernel for a given target SM.
///
/// The kernel allocates two SMEM regions: a static `seg_smem[SEG_SMEM_STATIC_BYTES]`
/// (Tier-B-off baseline; mirrors the original probe's structure) and an
/// extern `shmem[]` (Tier B's range-table region, sized by the launch's
/// dynamic shared-mem bytes parameter to `tier_b_extern_bytes` for the config).
/// Every thread writes a sentinel `0xA5` to both regions, then `bar.sync 0`,
/// then lane-0 reads back from `seg_smem[0]` and writes the byte (zero-extended
/// to u32) to the global output pointer. Readback == 0xA5 confirms both
/// allocation + execution.
fn build_probe_ptx(sm: u32) -> String {
    // PTX 8.7 covers sm_120 (Blackwell, CUDA 12.8+); it's also backward
    // compatible with sm_80 per the original probe's rationale.
    format!(
        r#".version 8.7
.target sm_{sm}
.address_size 64

.shared .align 4 .b8 seg_smem[{seg_bytes}];
.extern .shared .align 16 .b8 shmem[];

.visible .entry tier_b_bii_smem_probe(
    .param .u64 tier_b_bii_smem_probe_param_0
) {{
    .reg .u64 %out_ptr, %wide_seg_addr, %wide_shmem_addr, %slot_off;
    .reg .u32 %t, %n, %sent, %read_back;
    .reg .u16 %byte_a5, %read_u16;
    .reg .pred %p_skip;

    ld.param.u64 %out_ptr, [tier_b_bii_smem_probe_param_0];
    mov.u32 %t, %tid.x;

    // Static region write: every thread writes a sentinel byte at offset
    // tid (bounded by SEG_SMEM_STATIC_BYTES; the launch uses 64 threads so
    // we never exceed 64 bytes -- well inside the {seg_bytes}-byte region).
    mov.u16 %byte_a5, 0xA5;
    cvta.shared.u64 %wide_seg_addr, seg_smem;
    cvt.u64.u32 %slot_off, %t;
    add.u64 %wide_seg_addr, %wide_seg_addr, %slot_off;
    st.u8 [%wide_seg_addr], %byte_a5;

    // Extern region write: every thread writes the same sentinel at
    // offset tid. The host launch sizes the extern region per config via
    // sharedMemBytes; the probe's 64-thread launch only touches the first
    // 64 bytes of the extern region (proves the allocation succeeded).
    cvta.shared.u64 %wide_shmem_addr, shmem;
    add.u64 %wide_shmem_addr, %wide_shmem_addr, %slot_off;
    st.u8 [%wide_shmem_addr], %byte_a5;

    bar.sync 0;

    // Lane-0 readback from seg_smem[0] -> global output[0] (as u32 for an
    // unambiguous host comparison). cvt.u32.u16 zero-extends the low byte.
    setp.ne.u32 %p_skip, %t, 0;
    @%p_skip ret;

    ld.shared.u8 %read_u16, [seg_smem];
    cvt.u32.u16 %read_back, %read_u16;
    st.global.u32 [%out_ptr], %read_back;
    ret;
}}
"#,
        sm = sm,
        seg_bytes = SEG_SMEM_STATIC_BYTES,
    )
}

#[test]
fn probe_ptx_contains_expected_sections() {
    let ptx = build_probe_ptx(120);
    assert!(ptx.contains(".target sm_120"), "PTX must target sm_120");
    assert!(
        ptx.contains(".shared .align 4 .b8 seg_smem[8192]"),
        "PTX must allocate static seg_smem[8192]"
    );
    assert!(
        ptx.contains(".extern .shared .align 16 .b8 shmem[]"),
        "PTX must declare extern shmem[]"
    );
    assert!(
        ptx.contains("st.u8 [%wide_seg_addr], %byte_a5"),
        "PTX must write sentinel 0xA5 to seg_smem"
    );
    assert!(
        ptx.contains("st.u8 [%wide_shmem_addr], %byte_a5"),
        "PTX must write sentinel 0xA5 to extern shmem"
    );
    assert!(ptx.contains("bar.sync 0"), "PTX must barrier-sync before readback");
    assert!(
        ptx.contains("st.global.u32 [%out_ptr], %read_back"),
        "PTX must write readback to global output"
    );

    let ptx_sm80 = build_probe_ptx(80);
    assert!(ptx_sm80.contains(".target sm_80"), "sm_80 variant must target sm_80");
}

/// Execute one (max_seq_len, block, target_sm) probe configuration.
///
/// Computes the Tier B extern-shmem size for the config, launches the
/// kernel with that size via `cuModuleLoadDataEx` + `cuLaunchKernel`,
/// verifies the readback byte equals the sentinel `0xA5`.
fn run_probe_config(max_seq_len: u32, block: u32, target_sm: u32) -> ProbeResult {
    use cudarc::driver::sys;
    use std::os::raw::c_void;

    let tier_b_extern_bytes =
        compute_range_table_bytes(max_seq_len as u64, block as u64, block as u64) as u32;
    let total_smem_bytes = SEG_SMEM_STATIC_BYTES + tier_b_extern_bytes;
    let cap_bytes = smem_cap_bytes(target_sm);
    let utilization_pct = (total_smem_bytes as f32 / cap_bytes as f32) * 100.0;

    let ptx = build_probe_ptx(target_sm);

    let outcome = unsafe {
        // Ensure a CUDA context exists for the calling thread.
        // (Mirrors the cold-binary handshake from `tier_b_smem_probe.rs`.)
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        let query_rc = sys::cuCtxGetCurrent(&mut ctx);
        if query_rc != sys::CUresult::CUDA_SUCCESS || ctx.is_null() {
            let _ = sys::cuInit(0);
            let mut dev: sys::CUdevice = 0;
            let dev_rc = sys::cuDeviceGet(&mut dev, 0);
            if dev_rc != sys::CUresult::CUDA_SUCCESS {
                return ProbeResult {
                    max_seq_len,
                    block,
                    target_sm,
                    seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                    tier_b_extern_bytes,
                    total_smem_bytes,
                    cap_bytes,
                    utilization_pct,
                    outcome: ProbeOutcome::LaunchFailed("no CUDA device found".to_string()),
                };
            }
            let ctx_rc = sys::cuDevicePrimaryCtxRetain(&mut ctx, dev);
            if ctx_rc != sys::CUresult::CUDA_SUCCESS {
                return ProbeResult {
                    max_seq_len,
                    block,
                    target_sm,
                    seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                    tier_b_extern_bytes,
                    total_smem_bytes,
                    cap_bytes,
                    utilization_pct,
                    outcome: ProbeOutcome::LaunchFailed("cuDevicePrimaryCtxRetain failed".to_string()),
                };
            }
            let set_rc = sys::cuCtxSetCurrent(ctx);
            if set_rc != sys::CUresult::CUDA_SUCCESS {
                return ProbeResult {
                    max_seq_len,
                    block,
                    target_sm,
                    seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                    tier_b_extern_bytes,
                    total_smem_bytes,
                    cap_bytes,
                    utilization_pct,
                    outcome: ProbeOutcome::LaunchFailed("cuCtxSetCurrent failed".to_string()),
                };
            }
        }

        let ptx_c = match CString::new(ptx.clone()) {
            Ok(c) => c,
            Err(_) => {
                return ProbeResult {
                    max_seq_len,
                    block,
                    target_sm,
                    seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                    tier_b_extern_bytes,
                    total_smem_bytes,
                    cap_bytes,
                    utilization_pct,
                    outcome: ProbeOutcome::LaunchFailed("PTX contains embedded NUL".to_string()),
                };
            }
        };

        // cuModuleLoadDataEx with explicit ptxas error-log capture so
        // any JIT compilation failure surfaces as a structured message
        // rather than the generic "PTX JIT compilation failed" string.
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
        let mod_rc = sys::cuModuleLoadDataEx(
            &mut module,
            ptx_c.as_ptr() as *const c_void,
            options.len() as u32,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        );
        if mod_rc != sys::CUresult::CUDA_SUCCESS {
            let log_nul = err_log
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(err_log.len());
            let log_str = String::from_utf8_lossy(&err_log[..log_nul])
                .trim()
                .to_string();
            let mut err_name: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(mod_rc, &mut err_name);
            let driver_msg = if !err_name.is_null() {
                std::ffi::CStr::from_ptr(err_name).to_string_lossy().to_string()
            } else {
                format!("cuModuleLoadDataEx rc={:?}", mod_rc)
            };
            let combined = if log_str.is_empty() {
                driver_msg
            } else {
                format!("{driver_msg} | log: {log_str}")
            };
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::PtxasRejected(combined),
            };
        }

        let func_name = CString::new("tier_b_bii_smem_probe").unwrap();
        let mut func: sys::CUfunction = std::ptr::null_mut();
        let func_rc = sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr());
        if func_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuModuleUnload(module);
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed("cuModuleGetFunction failed".to_string()),
            };
        }

        // Single 4-byte device buffer for the readback word.
        let out_size: usize = 4;
        let mut out_buf: sys::CUdeviceptr = 0;
        let alloc_rc = sys::cuMemAlloc_v2(&mut out_buf, out_size);
        if alloc_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuModuleUnload(module);
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed("cuMemAlloc_v2 failed".to_string()),
            };
        }
        let clear_rc = sys::cuMemsetD8_v2(out_buf, 0, out_size);
        if clear_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed("cuMemsetD8_v2 failed".to_string()),
            };
        }

        // Opt in to >48 KB dynamic shared mem (matches original probe
        // discipline; the driver rejects launches with "invalid argument"
        // when sharedMemBytes > 49152 without this attribute).
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: sys::CUfunction_attribute =
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        let attr_rc = sys::cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            tier_b_extern_bytes.max(1) as i32,
        );
        if attr_rc != sys::CUresult::CUDA_SUCCESS {
            let mut err_name: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(attr_rc, &mut err_name);
            let msg = if !err_name.is_null() {
                std::ffi::CStr::from_ptr(err_name).to_string_lossy().to_string()
            } else {
                format!("cuFuncSetAttribute failed with rc={:?}", attr_rc)
            };
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed(format!("cuFuncSetAttribute: {}", msg)),
            };
        }

        let num_threads: u32 = 64;
        let mut out_ptr_arg: u64 = out_buf;
        let mut args: [*mut c_void; 1] = [&mut out_ptr_arg as *mut u64 as *mut c_void];

        let launch_rc = sys::cuLaunchKernel(
            func,
            1, 1, 1,
            num_threads, 1, 1,
            tier_b_extern_bytes,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if launch_rc != sys::CUresult::CUDA_SUCCESS {
            let mut err_name: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(launch_rc, &mut err_name);
            let msg = if !err_name.is_null() {
                std::ffi::CStr::from_ptr(err_name).to_string_lossy().to_string()
            } else {
                format!("cuLaunchKernel failed with rc={:?}", launch_rc)
            };
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed(msg),
            };
        }

        let sync_rc = sys::cuCtxSynchronize();
        if sync_rc != sys::CUresult::CUDA_SUCCESS {
            let mut err_name: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(sync_rc, &mut err_name);
            let msg = if !err_name.is_null() {
                std::ffi::CStr::from_ptr(err_name).to_string_lossy().to_string()
            } else {
                format!("cuCtxSynchronize failed with rc={:?}", sync_rc)
            };
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed(format!("cuCtxSynchronize: {}", msg)),
            };
        }

        let mut host_buf = vec![0u8; out_size];
        let copy_rc = sys::cuMemcpyDtoH_v2(
            host_buf.as_mut_ptr() as *mut c_void,
            out_buf,
            out_size,
        );
        let _ = sys::cuMemFree_v2(out_buf);
        let _ = sys::cuModuleUnload(module);
        if copy_rc != sys::CUresult::CUDA_SUCCESS {
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::LaunchFailed("cuMemcpyDtoH failed".to_string()),
            };
        }

        // Readback word: the lane-0 store wrote a u32 zero-extended from
        // the sentinel byte 0xA5. host_buf[0] should equal 0xA5; the
        // remaining 3 bytes should be the zero-extension padding.
        let found = host_buf[0];
        if found != 0xA5 {
            return ProbeResult {
                max_seq_len,
                block,
                target_sm,
                seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
                tier_b_extern_bytes,
                total_smem_bytes,
                cap_bytes,
                utilization_pct,
                outcome: ProbeOutcome::ValueCorruption { expected: 0xA5, found },
            };
        }

        ProbeOutcome::Pass
    };

    ProbeResult {
        max_seq_len,
        block,
        target_sm,
        seg_smem_bytes: SEG_SMEM_STATIC_BYTES,
        tier_b_extern_bytes,
        total_smem_bytes,
        cap_bytes,
        utilization_pct,
        outcome,
    }
}

/// Smoke fixture for the launch path. Picks the easiest config —
/// MAX=4096, block=64, sm_120 — which yields ~8.6% utilization on
/// Blackwell. If this fails the rest of the sweep is also suspect.
#[test]
fn probe_launch_smoke_sm120_max4096_block64() {
    let result = run_probe_config(4096, 64, 120);
    eprintln!(
        "smoke sm_{} MAX={} block={}: tier_b_extern={}B total={}B util={:.2}% outcome={:?}",
        result.target_sm,
        result.max_seq_len,
        result.block,
        result.tier_b_extern_bytes,
        result.total_smem_bytes,
        result.utilization_pct,
        result.outcome,
    );
    assert_eq!(result.outcome, ProbeOutcome::Pass);
}

