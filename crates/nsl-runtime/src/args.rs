use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Mutex;

use crate::list::{nsl_list_new, nsl_list_push};
use crate::memory::checked_alloc;

/// Wrapper so we can store raw argv pointer in a Mutex (it's only accessed under lock).
struct ArgvPtr(*const *const c_char);
unsafe impl Send for ArgvPtr {}

static ARGS: Mutex<(i32, ArgvPtr)> = Mutex::new((0, ArgvPtr(std::ptr::null())));

#[no_mangle]
pub extern "C" fn nsl_args_init(argc: i32, argv: i64) {
    let mut args = ARGS.lock().unwrap();
    *args = (argc, ArgvPtr(argv as *const *const c_char));

    // Auto-start memory profiler when NSL_PROFILE_MEMORY env var is set.
    if std::env::var("NSL_PROFILE_MEMORY").is_ok() {
        crate::profiling::nsl_profiler_start(0);
    }

    // Auto-start kernel profiler when NSL_PROFILE_KERNELS env var is set.
    if std::env::var("NSL_PROFILE_KERNELS").is_ok() {
        crate::kernel_profiler::nsl_kernel_profiler_start();
    }

    // ELTLS instrumentation: print GPU memory report at program exit when
    // NSL_GPU_MEM_REPORT env var is set. Piggybacks on the C `atexit`
    // function (available on both glibc and MSVCRT/UCRT) so the report
    // fires after the compiled Cranelift `main` returns.
    if std::env::var("NSL_GPU_MEM_REPORT").is_ok() {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_gpu_mem_report_atexit);
        }
    }

    // B.3 Task 5: print fused-adapter kernel-launch count at exit when
    // NSL_KERNEL_LAUNCH_COUNTER=1.  The counter is always live (cheap
    // atomic increment); the env var only gates the report.
    if std::env::var("NSL_KERNEL_LAUNCH_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(crate::fused_adapter::nsl_fused_adapter_launch_count_atexit);
        }
    }

    // B.3 Task 5.6 hardening: GPU-specific launch counter, enabled when
    // NSL_WRGA_GPU_LAUNCH_COUNTER=1.  Distinguishes real-GPU execution
    // from CPU fallback so tests can assert the fused CUDA path actually
    // fired (not just the math came out right).
    if std::env::var("NSL_WRGA_GPU_LAUNCH_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(crate::fused_adapter::nsl_fused_adapter_gpu_launch_count_atexit);
        }
    }

    // p9: fused FASE optimizer-step launch count, enabled when
    // NSL_FASE_FUSED_COUNTER=1. Lets the differential gate assert the fused
    // path actually fired (anti-vacuity), mirroring the WRGA counter pattern.
    // The counter is always live; the env var only gates the report.
    if std::env::var("NSL_FASE_FUSED_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_fase_fused_step_count_atexit);
        }
    }
}

extern "C" fn nsl_fase_fused_step_count_atexit() {
    eprintln!(
        "[fase-fused] optimizer fused-step launches: {}",
        crate::fase_step::nsl_fase_fused_step_count()
    );
}

extern "C" fn nsl_gpu_mem_report_atexit() {
    let epilog_frees = crate::tensor::nsl_debug_epilog_free_count();
    eprintln!("--- GPU memory report ---");
    eprintln!("epilog_frees_total: {epilog_frees}");
    #[cfg(feature = "cuda")]
    {
        // Print live caching-allocator block summary (step=0 so it prints).
        crate::tensor::nsl_debug_gpu_alloc_summary(0);
        // Print driver + allocator stats (step=0 so it prints).
        crate::tensor::nsl_debug_gpu_mem(0);
    }
}

#[no_mangle]
pub extern "C" fn nsl_args() -> i64 {
    let args = ARGS.lock().unwrap();
    let (argc, ref argv_wrapper) = *args;
    let argv = argv_wrapper.0;
    let list = nsl_list_new();
    for i in 0..argc {
        let arg = unsafe { *argv.add(i as usize) };
        let cstr = unsafe { CStr::from_ptr(arg) };
        let bytes = cstr.to_bytes_with_nul();
        let copy = checked_alloc(bytes.len());
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), copy, bytes.len());
        }
        nsl_list_push(list, copy as i64);
    }
    list
}
