//! Forced-access SMEM probe for PCA Tier B mixed static+extern allocation safety.
//!
//! Per spec §2 of `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`:
//! sweep 18 configurations (3 static-sizes × 3 dynamic-shmem-sizes × 2 architectures)
//! with every thread writing-and-reading both a static `.shared` array AND extern
//! `.shared` shmem. Detects the Blackwell ILLEGAL_ADDRESS failure mode at access
//! time, not declaration time.
//!
//! Outcome rows from the five-outcome decision matrix are recorded in
//! `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`.
//!
//! Run: `cargo test -p nsl-codegen --features cuda --test tier_b_smem_probe -- --nocapture --test-threads=1`
//! (`--test-threads=1` because CUDA context is thread-local; parallel runs
//! across the sweep collide on the primary context.)

#![cfg(feature = "cuda")]

use std::ffi::CString;

const STATIC_SIZES_BYTES: &[u32]  = &[256, 1024, 4096];
const DYNAMIC_SIZES_BYTES: &[u32] = &[16 * 1024, 64 * 1024, 96 * 1024];
const TARGET_ARCHES: &[u32] = &[80, 120]; // sm_80 (Ampere) + sm_120 (Blackwell)

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeOutcome {
    Pass,
    LaunchError(String),
    ValueCorruption { expected_a: u8, expected_b: u8, found_a: u8, found_b: u8 },
}

#[derive(Debug)]
struct ProbeRow {
    static_bytes: u32,
    dynamic_bytes: u32,
    sm: u32,
    outcome: ProbeOutcome,
}

fn build_probe_ptx(static_bytes: u32, sm: u32) -> String {
    // PTX ISA version must define the target family. sm_120
    // (Blackwell) was introduced in PTX 8.7 (CUDA 12.8) — the
    // driver-bundled JIT rejects sm_120 with anything older. 8.7
    // is backward compatible with sm_80, so we use it unconditionally.
    format!(
        r#".version 8.7
.target sm_{sm}
.address_size 64

.shared .align 4 .b8 seg_smem[{static_bytes}];
.extern .shared .align 16 .b8 shmem[];

.visible .entry probe_kernel(
    .param .u64 probe_kernel_param_0,
    .param .u32 probe_kernel_param_1
) {{
    .reg .u64 %out_ptr;
    // NOTE: avoid the name %tid for a user register; ptxas parses
    // `%tid.x` as our user %tid followed by a video selector `.x`
    // and rejects it. Use %t for our local copy.
    .reg .u32 %t, %n;
    .reg .u32 %seg_addr;
    .reg .u64 %wide_seg_addr, %wide_shmem_addr;
    .reg .u64 %slot0_ptr, %slot1_ptr;
    .reg .u16 %byte_aa, %byte_bb, %read_a, %read_b;
    .reg .pred %p_oob;

    ld.param.u64 %out_ptr, [probe_kernel_param_0];
    ld.param.u32 %n,       [probe_kernel_param_1];
    mov.u32 %t, %tid.x;

    setp.ge.u32 %p_oob, %t, %n;
    @%p_oob bra END;

    // Static-shared write at byte offset tid
    mov.u16 %byte_aa, 0xAA;
    cvta.shared.u64 %wide_seg_addr, seg_smem;
    cvt.u64.u32 %slot0_ptr, %t;
    add.u64 %wide_seg_addr, %wide_seg_addr, %slot0_ptr;
    st.u8 [%wide_seg_addr], %byte_aa;

    // Dynamic-shared write at byte offset tid
    mov.u16 %byte_bb, 0xBB;
    cvta.shared.u64 %wide_shmem_addr, shmem;
    add.u64 %wide_shmem_addr, %wide_shmem_addr, %slot0_ptr;
    st.u8 [%wide_shmem_addr], %byte_bb;

    bar.sync 0;

    // Read back via the generic addresses we already produced.
    ld.u8 %read_a, [%wide_seg_addr];
    ld.u8 %read_b, [%wide_shmem_addr];

    // Write to out[2*tid], out[2*tid+1]
    mul.lo.u32 %seg_addr, %t, 2;
    cvt.u64.u32 %slot0_ptr, %seg_addr;
    add.u64 %slot0_ptr, %out_ptr, %slot0_ptr;
    add.u64 %slot1_ptr, %slot0_ptr, 1;
    st.global.u8 [%slot0_ptr], %read_a;
    st.global.u8 [%slot1_ptr], %read_b;

END:
    ret;
}}
"#,
        sm = sm,
        static_bytes = static_bytes,
    )
}

fn run_probe_config(static_bytes: u32, dynamic_bytes: u32, sm: u32) -> ProbeRow {
    use cudarc::driver::sys;
    use std::os::raw::c_void;

    let num_threads: u32 = 64;
    let ptx = build_probe_ptx(static_bytes, sm);

    let outcome = unsafe {
        // Ensure we have a CUDA context.
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        let query_rc = sys::cuCtxGetCurrent(&mut ctx);
        if query_rc != sys::CUresult::CUDA_SUCCESS || ctx.is_null() {
            // cuInit is idempotent; required before cuDeviceGet on a cold
            // test binary where no other code has driven the driver yet.
            let _ = sys::cuInit(0);
            // Try to create one on device 0.
            let mut dev: sys::CUdevice = 0;
            let dev_rc = sys::cuDeviceGet(&mut dev, 0);
            if dev_rc != sys::CUresult::CUDA_SUCCESS {
                return ProbeRow {
                    static_bytes,
                    dynamic_bytes,
                    sm,
                    outcome: ProbeOutcome::LaunchError("no CUDA device found".to_string()),
                };
            }
            let ctx_rc = sys::cuDevicePrimaryCtxRetain(&mut ctx, dev);
            if ctx_rc != sys::CUresult::CUDA_SUCCESS {
                return ProbeRow {
                    static_bytes,
                    dynamic_bytes,
                    sm,
                    outcome: ProbeOutcome::LaunchError("cuDevicePrimaryCtxRetain failed".to_string()),
                };
            }
            let set_rc = sys::cuCtxSetCurrent(ctx);
            if set_rc != sys::CUresult::CUDA_SUCCESS {
                return ProbeRow {
                    static_bytes,
                    dynamic_bytes,
                    sm,
                    outcome: ProbeOutcome::LaunchError("cuCtxSetCurrent failed".to_string()),
                };
            }
        }

        // Load PTX module.
        let ptx_c = match CString::new(ptx.clone()) {
            Ok(c) => c,
            Err(_) => {
                return ProbeRow {
                    static_bytes,
                    dynamic_bytes,
                    sm,
                    outcome: ProbeOutcome::LaunchError("PTX contains embedded NUL".to_string()),
                };
            }
        };

        // Use cuModuleLoadDataEx with CU_JIT_ERROR_LOG_BUFFER so any
        // JIT failure message is captured instead of being collapsed
        // to the generic "a PTX JIT compilation failed" string.
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
                std::ffi::CStr::from_ptr(err_name)
                    .to_string_lossy()
                    .to_string()
            } else {
                format!("cuModuleLoadDataEx rc={:?}", mod_rc)
            };
            let combined = if log_str.is_empty() {
                driver_msg
            } else {
                format!("{driver_msg} | log: {log_str}")
            };
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError(combined),
            };
        }

        // Get function handle.
        let func_name = CString::new("probe_kernel").unwrap();
        let mut func: sys::CUfunction = std::ptr::null_mut();
        let func_rc = sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr());
        if func_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuModuleUnload(module);
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError("cuModuleGetFunction failed".to_string()),
            };
        }

        // Allocate device memory for output (2 bytes per thread).
        let out_size = (num_threads as usize) * 2;
        let mut out_buf: sys::CUdeviceptr = 0;
        let alloc_rc = sys::cuMemAlloc_v2(&mut out_buf, out_size);
        if alloc_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuModuleUnload(module);
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError("cuMemAlloc_v2 failed".to_string()),
            };
        }

        // Clear the output buffer.
        let clear_rc = sys::cuMemsetD8_v2(out_buf, 0, out_size);
        if clear_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError("cuMemsetD8_v2 failed".to_string()),
            };
        }

        // Opt in to dynamic shared > 48 KB. Both Ampere and Blackwell
        // require the kernel to request the larger budget explicitly;
        // without this the driver rejects launches with "invalid
        // argument" once shared_mem_bytes crosses 49152.
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: sys::CUfunction_attribute =
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        let attr_rc = sys::cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_bytes as i32,
        );
        if attr_rc != sys::CUresult::CUDA_SUCCESS {
            let mut err_name: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(attr_rc, &mut err_name);
            let msg = if !err_name.is_null() {
                std::ffi::CStr::from_ptr(err_name)
                    .to_string_lossy()
                    .to_string()
            } else {
                format!("cuFuncSetAttribute failed with rc={:?}", attr_rc)
            };
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError(format!("cuFuncSetAttribute: {}", msg)),
            };
        }

        // Prepare kernel arguments.
        let mut out_ptr_arg: u64 = out_buf;
        let mut n_threads_arg: u32 = num_threads;
        let mut args: [*mut c_void; 2] = [
            &mut out_ptr_arg as *mut u64 as *mut c_void,
            &mut n_threads_arg as *mut u32 as *mut c_void,
        ];

        // Launch kernel.
        let launch_rc = sys::cuLaunchKernel(
            func,
            1, 1, 1,                           // grid_dim
            num_threads, 1, 1,                 // block_dim
            dynamic_bytes,                     // shared_mem_bytes
            std::ptr::null_mut(),              // stream (NULL = default)
            args.as_mut_ptr(),                 // args
            std::ptr::null_mut(),              // extra
        );

        if launch_rc != sys::CUresult::CUDA_SUCCESS {
            let mut err_name: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(launch_rc, &mut err_name);
            let msg = if !err_name.is_null() {
                std::ffi::CStr::from_ptr(err_name)
                    .to_string_lossy()
                    .to_string()
            } else {
                format!("cuLaunchKernel failed with rc={:?}", launch_rc)
            };
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError(msg),
            };
        }

        // Synchronize.
        let sync_rc = sys::cuCtxSynchronize();
        if sync_rc != sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuMemFree_v2(out_buf);
            let _ = sys::cuModuleUnload(module);
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError("cuCtxSynchronize failed".to_string()),
            };
        }

        // Copy result back to host.
        let mut host_buf = vec![0u8; out_size];
        let copy_rc = sys::cuMemcpyDtoH_v2(
            host_buf.as_mut_ptr() as *mut c_void,
            out_buf,
            out_size,
        );

        let _ = sys::cuMemFree_v2(out_buf);
        let _ = sys::cuModuleUnload(module);

        if copy_rc != sys::CUresult::CUDA_SUCCESS {
            return ProbeRow {
                static_bytes,
                dynamic_bytes,
                sm,
                outcome: ProbeOutcome::LaunchError("cuMemcpyDtoH failed".to_string()),
            };
        }

        // Verify sentinel bytes.
        for tid in 0..num_threads as usize {
            let a = host_buf[2 * tid];
            let b = host_buf[2 * tid + 1];
            if a != 0xAA || b != 0xBB {
                return ProbeRow {
                    static_bytes,
                    dynamic_bytes,
                    sm,
                    outcome: ProbeOutcome::ValueCorruption {
                        expected_a: 0xAA,
                        expected_b: 0xBB,
                        found_a: a,
                        found_b: b,
                    },
                };
            }
        }

        ProbeRow {
            static_bytes,
            dynamic_bytes,
            sm,
            outcome: ProbeOutcome::Pass,
        }
    };

    outcome
}

#[test]
fn forced_access_probe_sweep() {
    let mut rows = Vec::new();
    for &sm in TARGET_ARCHES {
        for &static_bytes in STATIC_SIZES_BYTES {
            for &dynamic_bytes in DYNAMIC_SIZES_BYTES {
                let row = run_probe_config(static_bytes, dynamic_bytes, sm);
                eprintln!(
                    "sm_{} N={} M={}: {:?}",
                    row.sm, row.static_bytes, row.dynamic_bytes, row.outcome
                );
                rows.push(row);
            }
        }
    }

    eprintln!("\n=== Probe sweep summary (18 configs) ===");
    for row in &rows {
        eprintln!(
            "  sm_{:>3}  N={:>5}  M={:>6}  {:?}",
            row.sm, row.static_bytes, row.dynamic_bytes, row.outcome
        );
    }
    eprintln!("=== End summary ===\n");
    eprintln!("Transcribe into docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md");
}
