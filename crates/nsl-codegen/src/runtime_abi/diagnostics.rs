//! Profiling, VRAM accounting, health and trace surfaces.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_DIAGNOSTICS: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Training diagnostics (temporary)
    ("nsl_debug_train_step", &[types::I64, types::I64, types::I64], None),
    ("nsl_debug_gpu_mem", &[types::I64], None),
    ("nsl_gpu_drain_cache", &[], None),
    ("nsl_gpu_set_persistent_pool", &[], None),
    ("nsl_gpu_set_transient_pool", &[], None),
    // P0.1 per-surface VRAM accounting (tag values: caching_allocator::SurfaceTag)
    ("nsl_gpu_set_alloc_surface", &[types::I8], None),
    ("nsl_gpu_get_alloc_surface", &[], Some(types::I8)),
    // A1 unified accounting: numeric VRAM getters (peak / counts / per-surface)
    // + stable allocation identity. First in-process VRAM-peak API — gates and
    // WGGO read these instead of scraping NSL_MEMSTATS stderr.
    ("nsl_gpu_peak_allocated_bytes", &[], Some(types::I64)),
    ("nsl_flash_bwd_det_routed_count", &[], Some(types::I64)),
    ("nsl_gpu_cumulative_alloc_count", &[], Some(types::I64)),
    ("nsl_gpu_surface_peak_bytes", &[types::I8], Some(types::I64)),
    ("nsl_gpu_surface_at_peak_bytes", &[types::I8], Some(types::I64)),
    ("nsl_gpu_reset_mem_stats", &[], None),
    ("nsl_gpu_set_alloc_identity", &[types::I32, types::I64], None),
    ("nsl_gpu_clear_alloc_identity", &[], None),
    ("nsl_debug_gpu_alloc_summary", &[types::I64], None),
    // Health monitor FFI (dev-tools phase 4)
    ("nsl_health_record_loss", &[types::F64, types::I64], None),
    (
        "nsl_health_record_grad_norm",
        &[types::I64, types::I64, types::I32, types::F64],
        None,
    ),
    (
        "nsl_health_record_weight_norm",
        &[types::I64, types::I64, types::F64, types::I8],
        None,
    ),
    (
        "nsl_health_flush_snapshot",
        &[types::I64, types::I64],
        Some(types::I32),
    ),
    ("nsl_health_set_flush_interval", &[types::I64], None),
    // Inspector FFI (dev-tools phase 5)
    (
        "nsl_tensor_stats",
        &[types::I64, types::I64],
        Some(types::I32),
    ),
    (
        "nsl_inspect_record_stats",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I32),
    ),
    (
        "nsl_inspect_dump_full",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I32),
    ),
    ("nsl_inspect_set_dir", &[types::I64, types::I64], None),
    ("nsl_health_get_last_loss", &[], Some(types::F64)),
    ("nsl_health_get_loss_ema", &[], Some(types::F64)),
    ("nsl_health_get_loss_ema_slope", &[], Some(types::F64)),
    ("nsl_health_get_grad_norm_total", &[], Some(types::F64)),
    (
        "nsl_health_get_nan_inf_count_window",
        &[],
        Some(types::I64),
    ),
    // Timing and allocation tracking
    ("nsl_clock", &[], Some(types::F64)),
    // NSL_PHASE_TIMING train-block instrumentation (deferral-closure
    // 2026-07-14): device sync + per-phase wall-clock report lines.
    ("nsl_cuda_device_synchronize", &[], None),
    ("nsl_phase_fwd_bwd_report", &[types::F64, types::F64], None),
    ("nsl_phase_optim_report", &[types::F64], None),
    ("nsl_alloc_reset", &[], Some(types::I64)),
    ("nsl_alloc_count", &[], Some(types::I64)),
    ("nsl_alloc_bytes", &[], Some(types::I64)),
    (
        "nsl_model_to_device",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // Memory profiler (M25)
    ("nsl_profiler_start", &[types::I64], None),
    ("nsl_profiler_stop", &[], None),
    ("nsl_profiler_dump", &[types::I64, types::I64], None),
    ("nsl_profiler_peak", &[], Some(types::I64)),
    // Dev Tools Phase 2, Task 5: kernel-launch profile hooks.
    // Emitted around every GPU `kernel { ... }` launch when codegen runs
    // with `profile_kernels` enabled. Take a single i32 kernel_id matching
    // the dense ids assigned by ManifestBuilder::reserve_id().
    ("nsl_profile_kernel_begin", &[types::I32], None),
    ("nsl_profile_kernel_end", &[types::I32], None),
    // Kernel profiler (M26) — flush is NOT registered here (Rust-only atexit call)
    ("nsl_kernel_profiler_start", &[], None),
    ("nsl_kernel_profiler_stop", &[], None),
    // Execution fingerprint: (ptr, len) of a .rodata k=v record naming the
    // compile flags that decide training arithmetic. Installed before user
    // code so a checkpoint written later carries it.
    (
        "nsl_set_exec_fingerprint",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M45: Tensor debugger trace ---
    ("nsl_trace_init", &[], Some(types::I64)),
    (
        "nsl_trace_record_op",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_trace_suppress", &[], Some(types::I64)),
    ("nsl_trace_unsuppress", &[], Some(types::I64)),
    ("nsl_trace_breakpoint", &[], Some(types::I64)),
    ("nsl_trace_flush", &[], Some(types::I64)),
    ("nsl_trace_destroy", &[], Some(types::I64)),
    (
        "nsl_trace_nan_warning",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
];
