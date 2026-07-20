use cranelift_codegen::ir::{types, AbiParam, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use std::collections::HashMap;

use crate::error::CodegenError;

/// Runtime function info: (name, params, returns).
const RUNTIME_FUNCTIONS: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Print
    ("nsl_print_int", &[types::I64], None),
    ("nsl_print_float", &[types::F64], None),
    ("nsl_print_str", &[types::I64], None),
    ("nsl_print_bool", &[types::I8], None),
    // Power
    ("nsl_pow_int", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_pow_float", &[types::F64, types::F64], Some(types::F64)),
    // Memory
    ("nsl_alloc", &[types::I64], Some(types::I64)),
    ("nsl_free", &[types::I64], None),
    ("nsl_closure_free", &[types::I64], None),
    // List
    ("nsl_list_new", &[], Some(types::I64)),
    ("nsl_list_push", &[types::I64, types::I64], None),
    ("nsl_list_get", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_list_len", &[types::I64], Some(types::I64)),
    ("nsl_list_set", &[types::I64, types::I64, types::I64], None),
    (
        "nsl_list_contains",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    ("nsl_list_free", &[types::I64], None),
    // String
    (
        "nsl_str_concat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_int_to_str", &[types::I64], Some(types::I64)),
    ("nsl_float_to_str", &[types::F64], Some(types::I64)),
    ("nsl_bool_to_str", &[types::I8], Some(types::I64)),
    // Type conversions
    ("nsl_str_to_int", &[types::I64], Some(types::I64)),
    ("nsl_str_to_float", &[types::I64], Some(types::F64)),
    ("nsl_str_len", &[types::I64], Some(types::I64)),
    // Range
    (
        "nsl_range",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Dict
    ("nsl_dict_new", &[], Some(types::I64)),
    (
        "nsl_dict_set_str",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_dict_get_str",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_dict_len", &[types::I64], Some(types::I64)),
    (
        "nsl_dict_contains",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    ("nsl_dict_keys", &[types::I64], Some(types::I64)),
    ("nsl_dict_free", &[types::I64], None),
    ("nsl_dict_free_tensor_values", &[types::I64], None),
    // String comparison
    ("nsl_str_eq", &[types::I64, types::I64], Some(types::I64)),
    // String repeat & slice
    (
        "nsl_str_repeat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_list_slice",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_str_slice",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // String methods
    ("nsl_str_upper", &[types::I64], Some(types::I64)),
    ("nsl_str_lower", &[types::I64], Some(types::I64)),
    ("nsl_str_strip", &[types::I64], Some(types::I64)),
    ("nsl_str_split", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_str_join", &[types::I64, types::I64], Some(types::I64)),
    (
        "nsl_str_replace",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_str_find", &[types::I64, types::I64], Some(types::I64)),
    (
        "nsl_str_startswith",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    (
        "nsl_str_endswith",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    (
        "nsl_str_contains",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    // Math
    ("nsl_sqrt", &[types::F64], Some(types::F64)),
    ("nsl_log", &[types::F64], Some(types::F64)),
    ("nsl_exp", &[types::F64], Some(types::F64)),
    ("nsl_sin", &[types::F64], Some(types::F64)),
    ("nsl_cos", &[types::F64], Some(types::F64)),
    ("nsl_abs_float", &[types::F64], Some(types::F64)),
    ("nsl_abs_int", &[types::I64], Some(types::I64)),
    ("nsl_min_int", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_max_int", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_min_float", &[types::F64, types::F64], Some(types::F64)),
    ("nsl_max_float", &[types::F64, types::F64], Some(types::F64)),
    // Assert & Exit
    ("nsl_assert", &[types::I8, types::I64], None),
    ("nsl_exit", &[types::I64], None),
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
    ("nsl_gpu_cumulative_alloc_count", &[], Some(types::I64)),
    ("nsl_gpu_surface_peak_bytes", &[types::I8], Some(types::I64)),
    ("nsl_gpu_surface_at_peak_bytes", &[types::I8], Some(types::I64)),
    ("nsl_gpu_reset_mem_stats", &[], None),
    ("nsl_gpu_set_alloc_identity", &[types::I32, types::I64], None),
    ("nsl_gpu_clear_alloc_identity", &[], None),
    ("nsl_debug_gpu_alloc_summary", &[types::I64], None),
    // Stdin I/O
    ("nsl_read_line", &[], Some(types::I64)),
    // File I/O
    ("nsl_read_file", &[types::I64], Some(types::I64)),
    ("nsl_write_file", &[types::I64, types::I64], None),
    ("nsl_append_file", &[types::I64, types::I64], None),
    ("nsl_file_exists", &[types::I64], Some(types::I8)),
    // Command-line args
    ("nsl_args_init", &[types::I32, types::I64], None),
    ("nsl_args", &[], Some(types::I64)),
    // Higher-order functions
    ("nsl_map", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_filter", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_enumerate", &[types::I64], Some(types::I64)),
    ("nsl_zip", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_sorted", &[types::I64], Some(types::I64)),
    ("nsl_reversed", &[types::I64], Some(types::I64)),
    // Tensor creation
    ("nsl_tensor_zeros", &[types::I64], Some(types::I64)),
    ("nsl_tensor_ones", &[types::I64], Some(types::I64)),
    ("nsl_tensor_rand", &[types::I64], Some(types::I64)),
    ("nsl_tensor_randn", &[types::I64], Some(types::I64)),
    // Training mode
    ("nsl_set_training_mode", &[types::I8], None),
    ("nsl_is_training", &[], Some(types::I8)),
    (
        "nsl_tensor_full",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_arange",
        &[types::F64, types::F64, types::F64],
        Some(types::I64),
    ),
    // Tensor element access
    (
        "nsl_tensor_get",
        &[types::I64, types::I64],
        Some(types::F64),
    ),
    (
        "nsl_tensor_set",
        &[types::I64, types::I64, types::F64],
        None,
    ),
    // Tensor shape ops
    ("nsl_tensor_shape", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_shape_dim",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // M28: Dynamic shape assertions
    (
        "nsl_tensor_assert_dim",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_assert_dim_bound",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_ndim", &[types::I64], Some(types::I64)),
    // PCA Stage C: non-aborting shape probe (0 for out-of-range dims).
    ("nsl_tensor_dim_or_zero", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_len", &[types::I64], Some(types::I64)),
    ("nsl_tensor_get_dtype", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_reshape",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_transpose",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Tensor arithmetic (elementwise)
    (
        "nsl_tensor_add",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_sub",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_mul",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_div",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    ("nsl_tensor_neg", &[types::I64], Some(types::I64)),
    // Tensor scalar ops
    (
        "nsl_tensor_add_scalar",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_mul_scalar",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    // FASE fused scaled-add epilogue (p4): m += s * g, in place, void.
    (
        "nsl_tensor_scalar_mul_add_inplace",
        &[types::I64, types::I64, types::F64],
        None,
    ),
    // FASE Deferred bias correction: 1/(1 - base^step).  Scalar, no tensor args.
    (
        "nsl_bias_correction_inv",
        &[types::F64, types::I64],
        Some(types::F64),
    ),
    // FASE Deferred two-phase grad clip: sum of squared elements, in-place scale.
    (
        "nsl_tensor_sum_sq",
        &[types::I64],
        Some(types::F64),
    ),
    (
        "nsl_tensor_mul_scalar_inplace",
        &[types::I64, types::F64],
        None,
    ),
    // Tensor matmul
    (
        "nsl_tensor_matmul",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    // WRGA B.3 Task 4: fused LoRA/IA³ adapter matmul FFIs.
    // LoRA args: (x_ptr, w_ptr, a_ptr, b_ptr, scale_f64, kernel_handle_i64).
    // The scale is f64 at the FFI boundary because NSL FloatLiteral is f64;
    // the runtime narrows to f32 internally.
    (
        "nsl_adapter_fused_lora_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::F64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // IA³ args: (x_ptr, w_ptr, ia3_scale_ptr, kernel_handle_i64)
    (
        "nsl_adapter_fused_ia3_matmul",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // B.3.1 GatedLoRA args: (x_ptr, w_ptr, a_ptr, b_ptr, scale_f64, gate_ptr, kernel_handle_i64).
    // Body is a stub returning the base matmul (x @ W) until Task 5.0.c registers the real
    // fused PTX kernel.  The FFI declaration is needed here so compile_call resolves the
    // callee symbol and falls through to compile_traced_call rather than compile_indirect_call.
    (
        "nsl_adapter_fused_gatedlora_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::F64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // WRGA B.3 Task 5.6: fused-PTX runtime registry registration.
    // Args: (handle_i64, ptx_ptr_i64, ptx_len_i64, name_ptr_i64, name_len_i64).
    // Called from `main` preamble, one call per unique (m,n,k,rank,sm) key.
    (
        "nsl_wrga_register_fused_ptx",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    // Tensor reductions (return scalar tensor ptr, not f64)
    ("nsl_tensor_sum", &[types::I64], Some(types::I64)),
    ("nsl_tensor_mean", &[types::I64], Some(types::I64)),
    // Tensor scalar extraction
    ("nsl_tensor_item", &[types::I64], Some(types::F64)),
    ("nsl_tensor_l2_norm", &[types::I64], Some(types::F64)),
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
    // Tensor display
    ("nsl_tensor_print", &[types::I64], None),
    // Tensor memory
    ("nsl_tensor_clone", &[types::I64], Some(types::I64)),
    ("nsl_tensor_clone_if_valid", &[types::I64], Some(types::I64)),
    ("nsl_tensor_free", &[types::I64], None),
    ("nsl_tensor_free_if_valid", &[types::I64], None),
    ("nsl_tensor_retain", &[types::I64], None),
    ("nsl_tensor_release", &[types::I64], None),
    ("nsl_tensor_scope_begin", &[], None),
    ("nsl_tensor_scope_end", &[types::I64], None),
    // Fused elementwise operations (M31 fusion lowering)
    (
        "nsl_fused_elementwise_2",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_fused_elementwise_1",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_fused_matmul_epilogue",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
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
    // FBIP Phase 2: unconditional in-place variants (compiler-guaranteed single-use)
    ("nsl_tensor_relu_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_exp_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_log_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sqrt_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_abs_inplace", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_sigmoid_inplace",
        &[types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_tanh_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_neg_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sign_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_gelu_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_silu_inplace", &[types::I64], Some(types::I64)),
    // Autodiff tape management
    ("nsl_tape_start", &[types::I64], None),
    ("nsl_tape_stop", &[], None),
    (
        "nsl_tape_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tape_pause", &[], None),
    ("nsl_tape_resume", &[], None),
    // Gradient checkpointing
    (
        "nsl_checkpoint_record",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    // Element-wise tensor ops (M14)
    ("nsl_tensor_exp", &[types::I64], Some(types::I64)),
    ("nsl_tensor_log", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sqrt", &[types::I64], Some(types::I64)),
    ("nsl_tensor_abs", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sign", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_clamp",
        &[types::I64, types::F64, types::F64],
        Some(types::I64),
    ),
    // Dimensional reductions (M14)
    (
        "nsl_tensor_sum_dim",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_mean_dim",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_reduce_max",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_gather",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // CFTP §4.3 / Tier A activation — extract raw device pointer from
    // NslTensor* for use by nsl_packing_metadata_set. Returns 0 when
    // tensor_ptr == 0. See spec 2026-05-17-pca-rope-activation-design.md.
    ("nsl_tensor_data_ptr", &[types::I64], Some(types::I64)),
    // CFTP §4.3 / Tier A activation — thread-local registry for the
    // segment_ids/doc_starts pointers. Train block sets per step;
    // CSHA call sites read.
    ("nsl_packing_metadata_set", &[types::I64, types::I64], None),
    ("nsl_packing_metadata_get_segment_ids", &[], Some(types::I64)),
    ("nsl_packing_metadata_get_doc_starts", &[], Some(types::I64)),
    // PCA Tier A (spec §6.1) — mismatch warning: warns once if a
    // segment-masked module sees no segment_ids in the first N steps.
    ("nsl_pca_packing_mismatch_check", &[types::I64], None),
    // In-place mutation ops (M14)
    ("nsl_tensor_copy_data", &[types::I64, types::I64], None),
    ("nsl_tensor_add_inplace", &[types::I64, types::I64], None),
    ("nsl_tensor_zero_inplace", &[types::I64], None),
    ("nsl_tensor_zeros_like", &[types::I64], Some(types::I64)),
    // CPDT precision-adaptive optimizer: cast / zeros helpers
    ("nsl_tensor_cast", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cast_into", &[types::I64, types::I64], None),
    ("nsl_tensor_zeros_like_dtype", &[types::I64, types::I64], Some(types::I64)),
    // Optimizer-state offload (scaling campaign item 4): host-resident f32
    // zeros with the template's shape, regardless of the template's device.
    ("nsl_tensor_zeros_like_host_f32", &[types::I64], Some(types::I64)),
    // Offload P0.2: async copy-back (CONSUMES src — replaces the emitted
    // copy_data+free pair) + the once-per-step drain point.
    ("nsl_tensor_copy_data_async", &[types::I64, types::I64], None),
    ("nsl_offload_drain", &[], None),
    // Offload P0.3 (offload x reduced-precision composition): host state
    // at the planned dtype + the cross-device quant/dequant envelope.
    ("nsl_tensor_zeros_like_host_dtype", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cast_to_host_into", &[types::I64, types::I64], None),
    ("nsl_tensor_cast_from_host", &[types::I64, types::I64], Some(types::I64)),
    // CPDT §3.2: INT8 blockwise quantization (the headline 4× memory result)
    ("nsl_tensor_quant_int8_blockwise", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_dequant_int8_blockwise", &[types::I64], Some(types::I64)),
    // CFTP v6 forward inline-cast wrappers: src_ptr -> new tensor (scope-tracked).
    ("nsl_tensor_to_bf16", &[types::I64], Some(types::I64)),
    ("nsl_tensor_to_fp16", &[types::I64], Some(types::I64)),
    ("nsl_tensor_to_f32", &[types::I64], Some(types::I64)),
    // Gradient clipping (M14)
    ("nsl_clip_grad_norm", &[types::I64, types::F64], None),
    // Collect all tensor params from a model struct (recursive, magic-probed)
    (
        "nsl_collect_model_params",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Debug training: gradient checksum (--debug-training)
    ("nsl_debug_grad_checksum", &[types::I64, types::I64], None),
    // P0.3 gradient-integrity gate (--grad-integrity)
    ("nsl_grad_integrity_arm", &[], None),
    ("nsl_grad_integrity_check", &[types::I64, types::I64], None),
    ("nsl_grad_integrity_step_begin", &[types::I64], None),
    ("nsl_grad_integrity_note", &[types::I64, types::I64], None),
    ("nsl_grad_integrity_step_end", &[], None),
    // Prefetch tensor to GPU asynchronously
    ("nsl_tensor_prefetch", &[types::I64, types::I64], None),
    // M36: GPU memory slab (compile-time planned device memory arena)
    ("nsl_gpu_slab_init", &[types::I64], Some(types::I64)),
    (
        "nsl_slab_offset",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_gpu_slab_destroy", &[], None),
    ("nsl_gpu_slab_active", &[], Some(types::I64)),
    (
        "nsl_tensor_from_slab",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // M52b: Create tensor from static .rodata data (compile-time constant folded)
    (
        "nsl_tensor_from_static",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // M52c: CSR sparse matmul (row_ptrs, col_indices, values, B, nrows, ncols, nnz) -> C
    (
        "nsl_sparse_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // Checkpoint I/O (M14)
    (
        "nsl_model_save",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_model_load",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // Scalar math (M14)
    ("nsl_floor", &[types::F64], Some(types::F64)),
    // Activation functions (M15)
    ("nsl_tensor_relu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_gelu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_silu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sigmoid", &[types::I64], Some(types::I64)),
    ("nsl_tensor_tanh_act", &[types::I64], Some(types::I64)),
    // Tensor trig (RoPE support)
    ("nsl_tensor_sin", &[types::I64], Some(types::I64)),
    ("nsl_tensor_cos", &[types::I64], Some(types::I64)),
    // Fused rotate_half (RoPE support)
    ("nsl_tensor_rotate_half", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_softmax",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Slice & Cat (M15)
    (
        "nsl_tensor_slice",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_cat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Embedding lookup (M15)
    (
        "nsl_tensor_embedding_lookup",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // LayerNorm & RMSNorm (M15)
    (
        "nsl_tensor_layernorm",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_rmsnorm",
        &[types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // Item 9: fused RMSNorm input-gradient (dy, x, gamma, eps) -> dx.
    (
        "nsl_rmsnorm_dx_backward",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // Source AD: reduce gradient to match parameter shape (matmul broadcast backward)
    (
        "nsl_tensor_reduce_to_shape",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Dropout, Conv2d, MaxPool2d (M15)
    (
        "nsl_tensor_dropout",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_conv2d",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // Source-AD conv2d backward FFIs: (grad, input, weight, sh, sw, ph, pw) -> grad.
    (
        "nsl_conv2d_input_backward",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_conv2d_weight_backward",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_conv2d_bias_backward",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // Reify grad_output to the conv2d output shape once per node, shared by
    // the 3 FFIs above: (grad, input, weight, sh, sw, ph, pw) -> grad.
    (
        "nsl_materialize_conv_output_grad",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_tensor_maxpool2d",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Bias add (M15 — broadcast 1D bias over 2D tensor)
    (
        "nsl_tensor_bias_add",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // String deallocation (M15)
    ("nsl_string_free", &[types::I64], None),
    // Assert functions (M15 test framework)
    (
        "nsl_assert_eq_int",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_assert_eq_float",
        &[types::F64, types::F64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_assert_close",
        &[
            types::I64,
            types::I64,
            types::F64,
            types::F64,
            types::I64,
            types::I64,
        ],
        None,
    ),
    // Tokenizer functions (M15)
    ("nsl_byte_tokenizer_new", &[], Some(types::I64)),
    (
        "nsl_bpe_train",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tokenizer_load", &[types::I64], Some(types::I64)),
    ("nsl_tokenizer_save", &[types::I64, types::I64], None),
    (
        "nsl_tokenizer_encode",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tokenizer_decode",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tokenizer_vocab_size", &[types::I64], Some(types::I64)),
    (
        "nsl_tokenizer_encode_batch",
        &[types::I64, types::I64, types::I8, types::I8, types::I64],
        Some(types::I64),
    ),
    // Quantization (M16)
    (
        "nsl_qtensor_quantize",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_qtensor_dequantize", &[types::I64], Some(types::I64)),
    (
        "nsl_qtensor_matmul_mixed",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_qtensor_free", &[types::I64], None),
    ("nsl_qtensor_addref", &[types::I64], None),
    ("nsl_qtensor_release", &[types::I64], None),
    ("nsl_qtensor_dtype", &[types::I64], Some(types::I64)),
    ("nsl_qtensor_shape", &[types::I64], Some(types::I64)),
    // Tensor creation helpers (M17)
    (
        "nsl_tensor_zeros_on",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // CSHA Gap I.3 (A+F): f16 (dtype=2, 2 bytes/element) zeros allocator.
    // The Tier C backward kernel writes dq/dk/dv/dwq/dwk/dwv via
    // `st.global.u16`; the f32 `_zeros_on` variant over-allocates by 2×
    // and leaves every second byte uninitialised → host-side f32 reads
    // then interpret raw f16 bits as f32 → garbage → weight corruption.
    // `dx` stays on `_zeros_on` because the kernel writes it as f32.
    (
        "nsl_tensor_zeros_f16_on",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_ones_like", &[types::I64], Some(types::I64)),
    // GPU runtime functions (M17)
    ("nsl_cuda_init", &[], Some(types::I64)),
    (
        "nsl_kernel_launch",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // User `kernel` block launch: args array holds NslTensor handles; the
    // runtime extracts each `.data` device pointer and builds the kernelParams.
    (
        "nsl_kernel_launch_tensors",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_tensor_to_device",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_to_device_like",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Shape manipulation ops (M18a)
    (
        "nsl_tensor_unsqueeze",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_select",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_stack",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_expand",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_contiguous", &[types::I64], Some(types::I64)),
    ("nsl_tensor_causal_mask", &[types::I64], Some(types::I64)),
    // Safetensors I/O (M18b)
    (
        "nsl_safetensors_load",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_safetensors_save",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // HuggingFace Hub download + weight loading (M18b)
    (
        "nsl_hf_load",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // Trace infrastructure for ONNX export (M18b Task 7)
    ("nsl_trace_start", &[], None),
    ("nsl_trace_register_input", &[types::I64, types::I64], None),
    ("nsl_trace_register_output", &[types::I64, types::I64], None),
    ("nsl_trace_stop", &[], Some(types::I64)),
    // ONNX export (M18b Tasks 9-10)
    (
        "nsl_onnx_export",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // Sampling primitives (M19)
    ("nsl_manual_seed", &[types::I64], None),
    (
        "nsl_tensor_topk",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_multinomial",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_argmax",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_cumsum",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_lt_scalar",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    // Tensor mutation (M19)
    (
        "nsl_tensor_set_element",
        &[types::I64, types::I64, types::I64, types::F64],
        None,
    ),
    (
        "nsl_tensor_slice_assign",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    // Data sources (M19)
    (
        "nsl_load_jsonl",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_load_csv",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_load_mmap",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // DataLoader (M19)
    (
        "nsl_dataloader_create",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_dataloader_start", &[types::I64], None),
    ("nsl_dataloader_next_batch", &[types::I64], Some(types::I64)),
    ("nsl_dataloader_reset", &[types::I64], None),
    ("nsl_dataloader_stop", &[types::I64], None),
    ("nsl_dataloader_free", &[types::I64], None),
    // Packing efficiency (M19)
    ("nsl_packing_efficiency", &[types::I64], Some(types::F64)),
    // Custom dtype registry (M23)
    (
        "nsl_register_custom_dtype",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        None,
    ),
    ("nsl_finalize_dtype_registry", &[], None),
    (
        "nsl_tensor_to_custom_dtype",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_from_custom_dtype",
        &[types::I64],
        Some(types::I64),
    ),
    // Standalone weight provider and arg parser (M24)
    (
        "nsl_standalone_init_embedded",
        &[types::I64, types::I64],
        None,
    ),
    (
        "nsl_standalone_init_sidecar",
        &[types::I64, types::I64],
        None,
    ),
    ("nsl_standalone_has_weights", &[], Some(types::I64)),
    ("nsl_standalone_args_init", &[types::I64, types::I64], None),
    (
        "nsl_standalone_arg_str",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_str_default",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_int",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_int_default",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_float",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_float_default",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_standalone_args_finish", &[], None),
    // Paged KV-cache (M25)
    (
        "nsl_kv_cache_init",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_init_gpu",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    ("nsl_kv_cache_alloc_seq", &[types::I64], Some(types::I64)),
    (
        "nsl_kv_cache_append",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_k_ptr",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_v_ptr",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_cache_free_seq", &[types::I64, types::I64], None),
    (
        "nsl_kv_cache_seq_len",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_seq_blocks",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_seq_num_blocks",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_cache_utilization", &[types::I64], Some(types::F64)),
    ("nsl_kv_cache_destroy", &[types::I64], None),
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
    // FlashAttention-2 launch wrappers (M27)
    (
        "nsl_flash_attention",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64, // q, k, v, out
            types::I64, // logsumexp (backward aux, 0=skip)
            types::I64, // scale
            types::I64,
            types::I64,
            types::I64,
            types::I64, // batch, heads, seq_len, head_dim
            types::I64,
            types::I64,
            types::I64,
            types::I64, // block_table, k_pool, v_pool, block_size
            types::I64,
            types::I64, // cos, sin (RoPE)
            types::I64,
            types::I64, // seq_ids, seq_lens (M29-ready)
            types::I64, // shared_mem_bytes
            types::I64,
            types::I64, // ptx_ptr, name_ptr
            types::I64,
            types::I64, // block_q, block_kv
            types::I64, // causal (0=false, 1=true)
            // PCA Tier B planner spec §4 — must match the runtime signature
            // (nsl_runtime::flash_attention::nsl_flash_attention, lines 149-150).
            // Non-CSHA path has no Tier-B-on emission, so the caller always
            // passes the disabled sentinel (0, 0). Without these slots in the
            // Cranelift signature, the runtime's `assert_tier_b_sentinels`
            // entry guard reads stack garbage and may abort.
            types::I64, // tier_b_ptx_ptr  (always 0 from non-CSHA caller)
            types::I64, // tier_b_name_ptr (always 0 from non-CSHA caller)
        ],
        Some(types::I64),
    ),
    // CSHA Tier A.1: FA launcher variant carrying per-layer CSHA extras.
    // Same 24-arg prelude as `nsl_flash_attention`, then 9 CSHA args:
    //   x, norm_weight, Wq, Wk, Wv, Wo, rmsnorm_eps_bits, active_heads, d_model.
    // Stub today — forwards to `nsl_flash_attention`; A.2 will light up
    // the CSHA PTX body.
    (
        "nsl_flash_attention_csha",
        &[
            types::I64, types::I64, types::I64, types::I64, // q, k, v, out
            types::I64, // logsumexp
            types::I64, // scale_bits
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, types::I64, types::I64, types::I64, // block_table, k_pool, v_pool, block_size
            types::I64, types::I64, // cos, sin
            types::I64, types::I64, // seq_ids, seq_lens
            types::I64, // shared_mem_bytes
            types::I64, types::I64, // ptx_ptr, name_ptr
            types::I64, types::I64, // block_q, block_kv
            types::I64, // causal
            // CSHA extras:
            types::I64, // x_ptr
            types::I64, // norm_weight_ptr
            types::I64, types::I64, types::I64, types::I64, // Wq, Wk, Wv, Wo
            types::I64, // rmsnorm_eps_bits (f32 as i64)
            types::I64, // active_heads
            types::I64, // d_model
            // PCA Tier A: segment_ids device pointer (0 = unpacked path)
            types::I64, // segment_ids_ptr
            // Tier B extension (planner spec §4):
            types::I64, // tier_b_ptx_ptr
            types::I64, // tier_b_name_ptr
            // PCA §4.3: doc_starts device pointer (0 = identity positions)
            types::I64, // doc_starts_ptr
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — grid_x
            // override when the kernel name carries the `_per_doc_cta`
            // suffix. Pass 0 for all non-per-doc topologies.
            types::I64, // num_docs_or_zero
        ],
        Some(types::I64),
    ),
    // Gap A: CSHA FA launcher with activation-save pointers for source-AD
    // backward. Identical to `nsl_flash_attention_csha` plus 6 trailing
    // save-pointer args (q_proj, k_proj, v_proj, row_max, row_sum, x_raw).
    // Emitted by `compile_flash_attention_call` when
    // `CshaExtras.save_activations_for_backward` is true (i.e. inside a
    // `@train` block with CSHA fused).
    (
        "nsl_flash_attention_csha_with_saves",
        &[
            types::I64, types::I64, types::I64, types::I64, // q, k, v, out
            types::I64, // logsumexp
            types::I64, // scale_bits
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, types::I64, types::I64, types::I64, // block_table, k_pool, v_pool, block_size
            types::I64, types::I64, // cos, sin
            types::I64, types::I64, // seq_ids, seq_lens
            types::I64, // shared_mem_bytes
            types::I64, types::I64, // ptx_ptr, name_ptr
            types::I64, types::I64, // block_q, block_kv
            types::I64, // causal
            // CSHA extras (9):
            types::I64, // x_ptr
            types::I64, // norm_weight_ptr
            types::I64, types::I64, types::I64, types::I64, // Wq, Wk, Wv, Wo
            types::I64, // rmsnorm_eps_bits
            types::I64, // active_heads
            types::I64, // d_model
            // Tier C activation-save pointers (6):
            types::I64, // q_proj_ptr
            types::I64, // k_proj_ptr
            types::I64, // v_proj_ptr
            types::I64, // row_max_ptr
            types::I64, // row_sum_ptr
            types::I64, // x_raw_ptr
            // PCA Tier A: segment_ids device pointer (0 = unpacked path)
            types::I64, // segment_ids_ptr
            // Tier B extension (planner spec §4):
            types::I64, // tier_b_ptx_ptr
            types::I64, // tier_b_name_ptr
            // PCA §4.3: doc_starts device pointer (0 = identity positions)
            types::I64, // doc_starts_ptr
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — grid_x
            // override when the kernel name carries the `_per_doc_cta`
            // suffix. Pass 0 for all non-per-doc topologies.
            types::I64, // num_docs_or_zero
        ],
        Some(types::I64),
    ),
    // Gap A: codegen-side allocator for the 6 CSHA backward-activation
    // HBM buffers. Writes 6 device-pointer i64s contiguously into
    // `out_ptr` (caller-supplied stack-slot). Returns 0 on success.
    (
        "nsl_csha_alloc_backward_activations_into",
        &[
            types::I64, // batch
            types::I64, // heads
            types::I64, // seq
            types::I64, // head_dim
            types::I64, // out_ptr (writes 6 * i64)
        ],
        Some(types::I64),
    ),
    // Gap A: codegen-side free helper. Takes 6 i64 pointers matching the
    // layout written by `nsl_csha_alloc_backward_activations_into`.
    (
        "nsl_csha_free_backward_activations_from",
        &[
            types::I64, // q_proj
            types::I64, // k_proj
            types::I64, // v_proj
            types::I64, // row_max
            types::I64, // row_sum
            types::I64, // x_raw
        ],
        None,
    ),
    // Gap D / Tier C (extended by Gap I.5 Option A): CSHA fused backward
    // launch. i64 args matching the wengert_lower.rs
    // `PrimalOp::FusedCshaBackward` emission order, plus the trailing
    // tier_b2_active flag (CSHA Tier B.2 Phase 3 T6):
    //   36-arg forward-side prelude mirrored off `_with_saves`,
    //   + 6 forward-saved activation pointers,
    //   + dO input + 8 gradient outputs
    //     (dq, dk, dv, dwq, dwk, dwv, dx, dx_norm).
    // First surfaced as "undefined function" in the Gap I.3 smoke once
    // A+F let the backward launch actually fire. Gap I.5 appended the
    // 8th output (`dx_norm`) so the AD-side `RmsNormGammaBackward` gets
    // the correct `dy_norm` input.
    (
        "nsl_flash_attention_csha_backward",
        &[
            types::I64, types::I64, types::I64, // q, k, v
            types::I64, types::I64,             // out, logsumexp
            types::I64,                         // scale_bits
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, types::I64, types::I64, types::I64, // block_table, k_pool, v_pool, block_size
            types::I64, types::I64,             // cos, sin
            types::I64, types::I64,             // seq_ids, seq_lens
            types::I64,                         // shmem_bytes
            types::I64, types::I64,             // bwd_ptx_ptr, bwd_name_ptr
            types::I64, types::I64,             // block_q, block_kv
            types::I64,                         // causal
            types::I64, types::I64,             // x_ptr, norm_weight_ptr
            types::I64, types::I64, types::I64, // wq, wk, wv
            types::I64,                         // wo (null)
            types::I64,                         // rmsnorm_eps_bits
            types::I64, types::I64,             // active_heads, d_model
            // Saved activations (6):
            types::I64, types::I64, types::I64, // q_proj, k_proj, v_proj
            types::I64, types::I64,             // row_max, row_sum
            types::I64,                         // x_raw
            // Gradient outputs (dO + 8):
            types::I64,                         // do_ptr
            types::I64, types::I64, types::I64, // dq, dk, dv
            types::I64, types::I64, types::I64, // dwq, dwk, dwv
            types::I64,                         // dx
            types::I64,                         // dx_norm (Gap I.5)
            types::I64,                         // segment_ids_ptr (PCA Tier A Task 4B)
            types::I64,                         // tier_b_ptx_ptr (planner spec §4)
            types::I64,                         // tier_b_name_ptr (planner spec §4)
            types::I64,                         // doc_starts_ptr (PCA §4.3 Task 3)
            types::I64,                         // tier_b2_active (CSHA Tier B.2 Phase 3 T6)
            types::I64,                         // num_docs_or_zero (PCA per-doc CTA backward, Sprint 5)
        ],
        Some(types::I64),
    ),
    // FlashAttention-2 backward (M27 backward pass)
    // Returns NslList [dQ, dK, dV]. When logsumexp_ptr == 0, auto-computes lse.
    //
    // The trailing Tier-B sentinel pair (planner spec §4) was added to the
    // runtime FFI but this declaration and the wengert_lower call site were
    // never extended — Cranelift emitted 16-arg calls against the 18-param C
    // function, so the runtime read undefined stack/registers for the pair
    // and `assert_tier_b_sentinels` aborted the process at the FIRST plain-
    // SDPA training backward (found by the roadmap-4.2 pretrain e2e; no prior
    // test ran the compiler-EMITTED call — GPU parity tests build FFI args by
    // hand). Keep this in lock-step with `nsl_flash_attention_backward` in
    // nsl-runtime/src/flash_attention.rs.
    (
        "nsl_flash_attention_backward",
        &[
            types::I64, // dout
            types::I64, types::I64, types::I64, // q, k, v
            types::I64, // out (forward output)
            types::I64, // logsumexp (0 = auto-compute)
            types::I64, // scale_bits (f32 as i64)
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, // causal
            types::I64, // phase1_ptx_ptr (0 = CPU fallback)
            types::I64, // phase1_name_ptr
            types::I64, // phase2_ptx_ptr
            types::I64, // phase2_name_ptr
            types::I64, // tier_b_ptx_ptr (planner spec §4 sentinel pair;
            types::I64, // tier_b_name_ptr  both 0 = no Tier-B-on variant)
            types::I64, // segment_ids (PCA Stage C: NslTensor* [b,s], 0 = plain)
        ],
        Some(types::I64),
    ),
    // PCA Stage C: plain fused SDPA forward with saves. Launches the v2
    // scalar forward (csha: None) selected by the wengert lowering's
    // per-head_dim variant table; returns an NslList* [out, lse] or 0 to
    // DECLINE (caller's decomposed fallback runs). segment_ids != 0 selects
    // the segment-masked kernel family (packed attention); the Tier-B pair
    // is the tile-skip variant behind the runtime gate. Keep in lock-step
    // with `nsl_sdpa_fused_forward` in nsl-runtime/src/flash_attention.rs.
    (
        "nsl_sdpa_fused_forward",
        &[
            types::I64, types::I64, types::I64, // q, k, v (NslTensor*)
            types::I64, // scale_bits (f32 as i64)
            types::I64, // causal
            types::I64, // segment_ids (NslTensor* [b,s] or 0)
            types::I64, // ptx_ptr (base kernel; 0 = decline)
            types::I64, // name_ptr
            types::I64, // tier_b_ptx_ptr (sentinel pair, both 0 = none)
            types::I64, // tier_b_name_ptr
            types::I64, // block_q
            types::I64, // block_kv
            types::I64, // shared_mem_bytes
        ],
        Some(types::I64),
    ),
    // PCA Stage C: align packed-batch mask/segment tensors to the params'
    // device at step start (train-block batch prep). No-op for CPU models
    // and unpacked batches. Keep in lock-step with
    // `nsl_packed_batch_align_device` in nsl-runtime/src/packing.rs.
    (
        "nsl_packed_batch_align_device",
        &[
            types::I64, // batch dict (NslDict*)
            types::I64, // param list (NslList*) — device reference
        ],
        Some(types::I64),
    ),
    // Campaign item 5: derive the dense [b,1,s,s] packed mask from
    // segment_ids at the decomposed-fallback site (the DataLoader no
    // longer ships attention_mask by default). Keep in lock-step with
    // `nsl_packed_mask_from_segment_ids` in nsl-runtime/src/packing.rs.
    (
        "nsl_packed_mask_from_segment_ids",
        &[
            types::I64, // segment_ids (NslTensor* [b,s], f32/f64, CPU or GPU)
        ],
        Some(types::I64),
    ),
    // CFTP §4.4 G3 (Sprint 4): fused linear-CE FFI signatures.
    // Sprint v3-2 added trailing `dtype_tag` (0=F32 sentinel preserves
    // pre-v3-2 ABI; 1=F16). Sprint v4-1 extended the sentinel space
    // with 2=Bf16 (same single-i64 trailing arg — no ABI bump).
    // The Cranelift IR call sites in wengert_lower.rs derive the tag
    // from the @fused_lm_ce(dtype=...) decorator via
    // `fused_ce_dtype_for_compiler`. Note: v4-2 wengert refuses
    // tag != 0 pending precision_cast plumbing (see review Finding 2);
    // direct FFI tests with caller-managed 16-bit HBM allocation
    // exercise tags 1 and 2 end-to-end.
    // Forward v1 (small vocab, single CTA per row).
    (
        "nsl_fused_linear_ce_forward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, types::I64, types::I64, types::I64, // x, W, bias, targets (raw device ptrs)
            types::I64, types::I64, // loss_out, lse_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // smem_bytes
            types::I64, // dtype_tag (Sprint v3-2 / extended v4-1; 0=F32, 1=F16, 2=Bf16)
        ],
        Some(types::I64),
    ),
    // Forward large-vocab (Sprint 3 two-kernel path, vocab > 8192).
    (
        "nsl_fused_linear_ce_forward_large",
        &[
            types::I64, // ptx_ptr
            types::I64, // partials_kname_ptr
            types::I64, // finalize_kname_ptr
            types::I64, types::I64, types::I64, types::I64, // x, W, bias, targets
            types::I64, // partials_ptr (caller-owned scratch)
            types::I64, types::I64, // loss_out, lse_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // num_tiles
            types::I64, // smem_bytes
            types::I64, // dtype_tag (Sprint v3-2 / extended v4-1; 0=F32, 1=F16, 2=Bf16)
        ],
        Some(types::I64),
    ),
    // Backward (shared between v1 and large-vocab forward paths).
    (
        "nsl_fused_linear_ce_backward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, // grad_output_bits (f32 bits packed into i64)
            types::I64, types::I64, types::I64, types::I64, // x, W, bias, targets
            types::I64, // lse_ptr
            types::I64, types::I64, types::I64, // dx_out, dW_out, dbias_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // num_valid
            types::I64, // smem_bytes
            types::I64, // dtype_tag (Sprint v3-2 / extended v4-1; 0=F32, 1=F16, 2=Bf16)
        ],
        Some(types::I64),
    ),
    // CPKD: fused KL-CE distillation loss (forward + backward).
    //
    // ABI LOCK-STEP: these declarations, the call sites in
    // wengert_lower.rs (lower_fused_kl_ce_forward / _backward_extract),
    // and the runtime extern "C" fns in
    // crates/nsl-runtime/src/fused_kl_ce.rs must agree on arg count and
    // order BY HAND — there is no compile-time cross-check (see the
    // 16-vs-18-arg Tier-B lesson above nsl_flash_attention_backward).
    // Forward = 20 args; backward = 23 args.
    (
        "nsl_fused_kl_ce_forward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, types::I64, types::I64, // x_s, W_s, bias_s (raw device ptrs)
            types::I64, types::I64, types::I64, // x_t, W_t, bias_t
            types::I64, // targets
            types::I64, // loss_out
            types::I64, types::I64, types::I64, // lse_s1_out, lse_st_out, lse_tt_out
            types::I64, types::I64, types::I64, types::I64, // rows, v, hs, ht
            types::I64, types::I64, // alpha_bits, temp_bits (f32 bits in i64)
            types::I64, // smem_bytes
        ],
        Some(types::I64),
    ),
    (
        "nsl_fused_kl_ce_backward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, // grad_output_bits (f32 bits in i64)
            types::I64, types::I64, types::I64, // x_s, W_s, bias_s
            types::I64, types::I64, types::I64, // x_t, W_t, bias_t
            types::I64, // targets
            types::I64, types::I64, types::I64, // lse_s1, lse_st, lse_tt
            types::I64, types::I64, types::I64, // dxs_out, dws_out, dbs_out (student only — I-11)
            types::I64, types::I64, types::I64, types::I64, // rows, v, hs, ht
            types::I64, types::I64, // alpha_bits, temp_bits
            types::I64, // num_valid
        ],
        Some(types::I64),
    ),
    // M42b: Quantized FlashAttention (KV-cache in INT8/FP8)
    (
        "nsl_flash_attention_quantized",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64, // q, k, v, out, scale
            types::I64,
            types::I64,
            types::I64,
            types::I64, // batch, heads, seq_len, head_dim
            types::I64,
            types::I64,
            types::I64,
            types::I64, // block_table, k_pool, v_pool, block_size
            types::I64,
            types::I64,
            types::I64, // meta_k, meta_v, kv_quant_scheme
            types::I64, // shared_mem_bytes
            types::I64,
            types::I64, // ptx_ptr, name_ptr
            types::I64,
            types::I64, // block_q, block_kv
            // PCA Tier B planner spec §4 — the runtime
            // `nsl_flash_attention_quantized` takes the SAME 2 trailing Tier-B
            // sentinel slots as `nsl_flash_attention` and reads them via
            // `assert_tier_b_sentinels`. They were missing here (declared 21 vs
            // the runtime's 23) — a latent ABI drift: no call site emits this
            // today, but any future caller would push 21 args and the runtime
            // would read 2 slots of stack garbage. Caught by the nsl-abi
            // signature-agreement gate.
            types::I64, // tier_b_ptx_ptr  (disabled sentinel 0)
            types::I64, // tier_b_name_ptr (disabled sentinel 0)
        ],
        Some(types::I64),
    ),
    (
        "nsl_rope_cache_write",
        &[
            types::I64,
            types::I64, // k_projected, v_projected
            types::I64,
            types::I64,
            types::I64, // cos, sin, positions
            types::I64,
            types::I64,
            types::I64, // k_pool, v_pool, block_table
            types::I64,
            types::I64, // seq_ids, seq_lens (M29-ready)
            types::I64,
            types::I64,
            types::I64,
            types::I64, // num_tokens, num_heads, head_dim, block_size
            types::I64,
            types::I64, // ptx_ptr, name_ptr
        ],
        Some(types::I64),
    ),
    // M29: Serving engine
    (
        "nsl_serve_init",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_serve_enqueue",
        &[types::I64, types::I64, types::I64, types::F64, types::F64],
        Some(types::I64),
    ),
    ("nsl_serve_step", &[], Some(types::I64)),
    (
        "nsl_serve_record_token",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_serve_drain_completed", &[], Some(types::I64)),
    ("nsl_serve_has_work", &[], Some(types::I64)),
    ("nsl_serve_completed_count", &[], Some(types::I64)),
    ("nsl_serve_preempt", &[types::I64], Some(types::I64)),
    ("nsl_serve_destroy", &[], Some(types::I64)),
    // --- CFIE: continuous-batching request ring + grammar-table helper ---
    ("nsl_cfie_ring_init", &[types::I64], Some(types::I64)),
    (
        "nsl_cfie_ring_push",
        &[
            types::I64,
            types::I64, // sequence_id, prompt_ptr
            types::I64,
            types::I64, // prompt_len, max_new_tokens
            types::I64,
            types::I64, // grammar_start_state, sampling_packed
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_ring_pop",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64, // six out-pointers mirroring ring_push fields
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_ring_len", &[], Some(types::I64)),
    (
        "nsl_cfie_grammar_transition",
        &[
            types::I64, // table_ptr
            types::I64,
            types::I64, // num_states, vocab_size
            types::I64,
            types::I64, // state, token
        ],
        Some(types::I64),
    ),
    // --- CFIE: KV sequence-slot free-list ---
    (
        "nsl_cfie_kv_slots_init",
        &[types::I64, types::I64], // slot_count, per_slot_tokens
        Some(types::I64),
    ),
    ("nsl_cfie_kv_slot_acquire", &[], Some(types::I64)),
    ("nsl_cfie_kv_slot_release", &[types::I64], Some(types::I64)),
    (
        "nsl_cfie_kv_slot_advance",
        &[types::I64, types::I64], // slot, n_tokens
        Some(types::I64),
    ),
    (
        "nsl_cfie_kv_slot_rollback",
        &[types::I64, types::I64], // slot, n_tokens
        Some(types::I64),
    ),
    ("nsl_cfie_kv_slots_active", &[], Some(types::I64)),
    (
        "nsl_cfie_kv_attach_device",
        &[types::I64, types::I64], // base, bytes
        Some(types::I64),
    ),
    // --- CFIE Cycle 6: compiled-engine registration/lifecycle + launch
    // FFIs (frozen ABI).  All params/returns i64; f32 kernel params are
    // passed as f32::to_bits in the LOW 32 bits.  Kernel kinds:
    // 0=decode_attn, 1=fused_sample, 2=decode_block, 3=spec_verify,
    // 4=spec_reject, 5=quant_attn (layer_idx meaningful only for 5). ---
    (
        "nsl_cfie_register_kernel",
        &[
            types::I64, // kind
            types::I64, // layer_idx
            types::I64, // ptx_ptr
            types::I64, // ptx_len (excludes NUL)
            types::I64, // name_ptr
            types::I64, // name_len (excludes NUL)
            types::I64, // grid_x
            types::I64, // block_x
            types::I64, // smem_dyn_bytes
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_kv_pool_alloc", &[types::I64], Some(types::I64)), // bytes
    ("nsl_cfie_engine_finalize", &[], Some(types::I64)),
    ("nsl_cfie_engine_destroy", &[], Some(types::I64)),
    // --- CFIE Cycle 9: runtime weight binding (production upload FFIs).
    // Cast host f32 [out][in] row-major -> device f16/f32, persistent
    // pool, engine-tracked; reset frees them. ---
    (
        "nsl_cfie_upload_weight_f16",
        &[
            types::I64, // host_f32_ptr
            types::I64, // n_elems
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_upload_weight_f32",
        &[
            types::I64, // host_f32_ptr
            types::I64, // n_elems
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_weights_reset", &[], Some(types::I64)),
    (
        "nsl_cfie_launch_decode_attn",
        &[
            types::I64, // q_ptr
            types::I64, // out_ptr
            types::I64, // layer_idx
            types::I64, // slot_idx
            types::I64, // seq_len
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_fused_sample",
        &[
            types::I64, // hidden_ptr
            types::I64, // norm_w_ptr
            types::I64, // lm_head_ptr
            types::I64, // out_token_ptr
            types::I64, // rng_seed
            types::I64, // grammar_state
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_decode_block",
        &[
            types::I64, // x_in
            types::I64, // x_out
            types::I64, // wq
            types::I64, // wk
            types::I64, // wv
            types::I64, // wo
            types::I64, // w_gate
            types::I64, // w_up
            types::I64, // w_down
            types::I64, // norm1_w
            types::I64, // norm2_w
            types::I64, // layer_idx
            types::I64, // slot_idx
            types::I64, // pos
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_spec_verify",
        &[
            types::I64, // q_ptr
            types::I64, // out_ptr
            types::I64, // layer_idx
            types::I64, // slot_idx
            types::I64, // seq_len
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_spec_reject",
        &[
            types::I64, // target_probs_ptr
            types::I64, // draft_probs_ptr
            types::I64, // draft_tokens_ptr
            types::I64, // rng_seed
            types::I64, // out_accepted_ptr
            types::I64, // out_correction_token_ptr
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_quant_attn",
        &[
            types::I64, // layer_idx (selects the (5, layer) registration)
            types::I64, // q_ptr
            types::I64, // out_ptr
            types::I64, // slot_idx
            types::I64, // seq_len
            types::I64, // k_scale_bits (f32 bits, low 32)
            types::I64, // v_scale_bits (f32 bits, low 32)
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_decode_step",
        &[
            types::I64, // x_buf_a
            types::I64, // x_buf_b
            types::I64, // layer_weights_ptr (host array: n_layers x 9 device ptrs)
            types::I64, // n_layers
            types::I64, // norm_w_ptr (final-norm gamma)
            types::I64, // lm_head_ptr
            types::I64, // slot_idx
            types::I64, // pos
            types::I64, // rng_seed
            types::I64, // grammar_state
            types::I64, // out_token_ptr (device u32)
        ],
        Some(types::I64),
    ),
    // --- CFIE Cycle 10: model binding + generation driver. bind_model
    // resolves an NslModel's host f32 weights by the HF-Llama name
    // convention, uploads them, and records the device weight table;
    // generate drives the decode loop over a prompt; generate_reset
    // clears the binding without freeing the weight buffers. ---
    (
        "nsl_cfie_bind_model",
        &[
            types::I64, // model_handle (NslModel*)
            types::I64, // n_layers
            types::I64, // d_model
            types::I64, // n_heads
            types::I64, // n_kv_heads
            types::I64, // head_dim
            types::I64, // d_ff
            types::I64, // vocab_size
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_generate",
        &[
            types::I64, // prompt_tokens_ptr (host i64 array)
            types::I64, // prompt_len
            types::I64, // max_new_tokens
            types::I64, // eos_token_id
            types::I64, // rng_seed
            types::I64, // out_tokens_ptr (host i64 array)
            types::I64, // out_cap
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_generate_reset", &[], Some(types::I64)),
    // --- CFIE Cycle 12: host token-buffer <-> tokenizer-tensor bridge.
    // tokens_to_tensor turns generate's out-buffer into the 1-D f64
    // tensor nsl_tokenizer_decode consumes (text output); tensor_to_tokens
    // turns nsl_tokenizer_encode's tensor into generate's host i64 prompt
    // array (runtime-encoded prompt). ---
    (
        "nsl_cfie_tokens_to_tensor",
        &[
            types::I64, // tokens_ptr (host i64 array)
            types::I64, // count
        ],
        Some(types::I64), // NslTensor* (1-D f64), or 0 on bad args
    ),
    (
        "nsl_cfie_tensor_to_tokens",
        &[
            types::I64, // tensor_ptr (1-D f64 NslTensor*)
            types::I64, // out_ptr (host i64 buffer)
            types::I64, // cap
        ],
        Some(types::I64), // FULL token count (> cap = truncated), or -1
    ),
    // --- CFIE Cycle 13 (G15 draft-model-in-binary): draft-model binding
    // + engine-held draft KV pool + speculative decode driver.  The
    // serve wiring emits bind_draft_model/draft_pool_alloc at serve init
    // (after the target bind) and speculative_generate from the
    // endpoint's generate() when the speculative draft is configured;
    // the launch FFIs are the kind-6/7/8 wrappers the driver uses
    // internally (registered for ABI completeness + direct testing). ---
    (
        "nsl_cfie_bind_draft_model",
        &[
            types::I64, // model_handle (NslModel*)
            types::I64, // n_layers (draft)
            types::I64, // d_model (draft)
            types::I64, // n_heads (draft)
            types::I64, // n_kv_heads (draft)
            types::I64, // head_dim (draft)
            types::I64, // d_ff (draft)
            types::I64, // vocab_size (MUST equal the target binding's)
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_draft_pool_alloc", &[types::I64], Some(types::I64)), // bytes
    ("nsl_cfie_draft_reset", &[], Some(types::I64)),
    (
        "nsl_cfie_launch_draft_block",
        &[
            types::I64, // x_in
            types::I64, // x_out
            types::I64, // layer_idx (draft weight table is engine-held)
            types::I64, // pos
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_draft_sample",
        &[
            types::I64, // hidden_ptr
            types::I64, // out_token_ptr (device u32)
            types::I64, // out_prob_ptr (device f32)
            types::I64, // rng_seed (accepted-unused; greedy v1)
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_verify_probs",
        &[
            types::I64, // hidden_ptr
            types::I64, // out_probs_ptr (device f32 x vocab)
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_speculative_generate",
        &[
            types::I64, // prompt_tokens_ptr (host i64 array)
            types::I64, // prompt_len
            types::I64, // max_new_tokens
            types::I64, // eos_token_id
            types::I64, // rng_seed
            types::I64, // k_tokens (MUST match the kind-4 kernel's K)
            types::I64, // out_tokens_ptr (host i64 array)
            types::I64, // out_cap
        ],
        Some(types::I64),
    ),
    // --- M41: Disaggregated inference ---
    (
        "nsl_disagg_init",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_disagg_get_role", &[], Some(types::I64)),
    ("nsl_disagg_get_rank", &[], Some(types::I64)),
    ("nsl_disagg_destroy", &[], Some(types::I64)),
    (
        "nsl_disagg_worker_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_disagg_worker_destroy", &[], Some(types::I64)),
    ("nsl_disagg_prefill_loop", &[types::I64], Some(types::I64)),
    ("nsl_disagg_decode_loop", &[types::I64], Some(types::I64)),
    // --- M41b: KV transfer backends ---
    (
        "nsl_kv_transfer_init",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_transfer_send",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_transfer_recv",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_transfer_destroy", &[], Some(types::I64)),
    // --- M30: Tensor parallelism ---
    ("nsl_tp_init", &[], Some(types::I64)),
    ("nsl_tp_rank", &[], Some(types::I64)),
    ("nsl_tp_world_size", &[], Some(types::I64)),
    (
        "nsl_tp_all_reduce_sum",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tp_all_gather",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tp_broadcast",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tp_barrier", &[], Some(types::I64)),
    ("nsl_tp_destroy", &[], Some(types::I64)),
    // --- M32: MoE runtime functions ---
    (
        "nsl_moe_route",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_scatter",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_expert_parallel_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_gather",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_all_to_all",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_aux_loss",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_dispatch_full",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // CPDT Part III v1 production-forward (M32 gap closure): same as v1
    // plus `experts_ptr`, `hidden_dim`, `intermediate_dim` (3 extra i64
    // args, total 8). Returns NslTensor `[total_tokens, intermediate_dim]`
    // (note: trailing dim differs from v1's `[total_tokens, hidden_dim]`
    // identity output). See crates/nsl-runtime/src/moe/ffi.rs.
    (
        "nsl_moe_dispatch_full_v2",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // CPDT Part III v2.2 paper-faithful MoE FFN: per-expert kernel is
    // `up → SiLU → down` instead of v2's single matmul. 10 i64 args:
    // tokens, logits, experts_up, experts_down, num_experts, top_k,
    // capacity_factor_bits, hidden_dim, intermediate_dim, activation_kind,
    // experts_up_bias_ptr, experts_down_bias_ptr (v2.11: bias args are
    // nullable — pass 0 for no bias).
    // Returns NslTensor `[total_tokens, hidden_dim]` (back to hidden,
    // unlike v2's intermediate). See nsl-runtime/src/moe/ffi.rs.
    (
        "nsl_moe_dispatch_full_v3",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // CPDT Part III v2.5+v2.8 Mixtral gated MoE FFN: per-expert kernel is
    // `gate_act(gate) * up → down` where gate_act is selected by
    // gate_activation_kind. 11 i64 args: tokens, logits, experts_gate,
    // experts_up, experts_down, num_experts, top_k, capacity_factor_bits,
    // hidden_dim, intermediate_dim, gate_activation_kind (v2.8: 1=SwiGLU,
    // 2=GeGLU, 3=ReGLU). Output shape matches v3
    // `[total_tokens, hidden_dim]`. See nsl-runtime/src/moe/ffi.rs.
    (
        "nsl_moe_dispatch_full_v4",
        // v2.14: 14 i64 args. Positions 12+13+14 are
        // experts_{gate,up,down}_bias_ptr (nullable, 0 = no bias).
        // Codegen always emits iconst(0) for these in the 5-arg
        // source form; the 8-arg form threads source-supplied bias
        // expressions through (mirrors v3's 4/6 pattern in v2.12).
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // --- M33: Speculative decoding runtime functions ---
    (
        "nsl_speculative_draft",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_speculative_verify",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_speculative_build_tree",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_speculative_verify_tree",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_page_branch",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_page_cow_copy",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tree_attention",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_speculative_cleanup",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_speculative_decode_step",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // --- M34: Context parallelism (ring attention) ---
    // Extern signatures REMOVED across two cycles:
    //   * CPDT Part III v2.22 unlinked the ring FFI chain from codegen
    //     (`nsl_cp_init` / `nsl_sequence_partition` / `nsl_ring_attention`
    //     / `nsl_ring_send_recv` / `nsl_sequence_gather` / `nsl_cp_destroy`)
    //     and fell `@context_parallel` through to naive attention.
    //   * M34 v1 (this cycle) deleted the six runtime stubs themselves from
    //     `crates/nsl-runtime/src/context_parallel/ffi.rs` (they were dead
    //     symbols with wrong positional layouts) and shipped the
    //     single-node ring-attention composer
    //     (`run_ring_attention_full`) verified against `naive_attention`
    //     on a matrix of shapes and ring sizes. Multi-device distribution
    //     is deferred until NCCL/IPC lands.
    // When multi-device distribution lands, a fresh runtime FFI shape gets
    // designed and the extern table + emission + runtime impl all get
    // wired together against the new shape.
    // --- M35: FP8 compute ---
    (
        "nsl_fp8_cast",
        &[types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_fp8_matmul",
        &[types::I64, types::I64, types::F64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_fp8_matmul_training",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_fp8_compute_scale",
        &[types::I64, types::I64],
        Some(types::F64),
    ),
    (
        "nsl_fp8_quantize_e5m2",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    ("nsl_fp8_gradient_scale", &[types::I64], Some(types::F64)),
    ("nsl_fp8_cache_e5m2_ptx", &[types::I64, types::I64], None),
    (
        "nsl_fp8_update_calibration",
        &[types::I64, types::I64, types::F64],
        Some(types::F64),
    ),
    // --- M35: AWQ 4-bit quantization ---
    (
        "nsl_awq_quantize",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_awq_matmul",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_awq_free", &[types::I64], None),
    // AWQ calibration sidecar: apply per-channel scales to weight tensor before quantizing.
    // Signature: (weight_ptr, scales_ptr, scales_len, alpha) -> scaled_weight_ptr
    (
        "nsl_awq_pre_scale_weight",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // --- M35: GPTQ quantization ---
    (
        "nsl_gptq_quantize",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_gptq_quantize_ext",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_gptq_matmul",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_gptq_free", &[types::I64], None),
    ("nsl_gptq_hessian_init", &[types::I64], Some(types::I64)),
    (
        "nsl_gptq_hessian_add_batch",
        &[types::I64],
        Some(types::I64),
    ),
    ("nsl_gptq_hessian_finalize", &[], Some(types::I64)),
    // --- M42: KV-cache compression ---
    (
        "nsl_kv_quantize_and_store",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_kv_sliding_window_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_sliding_window_check",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_sliding_window_destroy", &[], Some(types::I64)),
    (
        "nsl_kv_h2o_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_h2o_accumulate",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_h2o_check",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_h2o_remove_sequence",
        &[types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_h2o_destroy", &[], Some(types::I64)),
    ("nsl_kv_compress_ratio", &[types::I64], Some(types::I64)),
    // --- M44: Constrained decoding (grammar FSM) ---
    (
        "nsl_grammar_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_grammar_step",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_grammar_apply_mask",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_grammar_is_accept", &[types::I64], Some(types::I64)),
    ("nsl_grammar_start_state", &[], Some(types::I64)),
    ("nsl_grammar_destroy", &[], Some(types::I64)),
    // M44b: Constrained decoding serve integration
    (
        "nsl_serve_apply_grammar",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_serve_advance_grammar",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_serve_set_grammar",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M39b: vmap runtime ---
    (
        "nsl_vmap_check_batch",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M40b: Backward context for source-to-source AD (handle-based) ---
    ("nsl_backward_ctx_new", &[types::I64], Some(types::I64)),
    (
        "nsl_backward_ctx_save",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_backward_ctx_load",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_backward_ctx_free", &[types::I64], Some(types::I64)),
    // --- M43: Pipeline parallelism ---
    (
        "nsl_pipeline_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_send",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_recv",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_send_grad",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_recv_grad",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    ("nsl_pipeline_barrier", &[], Some(types::I64)),
    ("nsl_pipeline_destroy", &[], Some(types::I64)),
    // --- M43: ZeRO optimizer (ABI-fixed: match runtime signatures exactly) ---
    ("nsl_zero_init", &[types::I64, types::I64], Some(types::I64)), // (stage, world_size)
    ("nsl_zero_partition", &[types::I64], Some(types::I64)),        // (num_params)
    (
        "nsl_zero_reduce_grads",
        &[types::I64, types::I64],
        Some(types::I64),
    ), // (grad_ptr, num_elems)
    ("nsl_zero_step", &[], Some(types::I64)),                       // ()
    // D3 (ZeRO-1): post-step parameter sync — broadcast each param from
    // its owner rank (idx % world_size) so all ranks hold the full model.
    ("nsl_zero_sync_params", &[types::I64, types::I64], Some(types::I64)), // (param_list, num_params)
    ("nsl_zero_destroy", &[], Some(types::I64)),
    ("nsl_zero_owns_param", &[types::I64], Some(types::I64)), // (param_idx) -> 1 if owned
    // D3 v2: record an owned optimizer-moment allocation's element count so the
    // G3 gate can prove per-rank optimizer state shrank to ~1/world_size.
    ("nsl_zero_note_optim_alloc", &[types::I64], Some(types::I64)), // (tensor_ptr) -> running elems
    // --- M43: Gradient accumulation (ABI-fixed) ---
    (
        "nsl_grad_accumulate_add",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ), // (dst, src, num_elems)
    ("nsl_grad_zero", &[types::I64, types::I64], Some(types::I64)), // (grad_ptr, num_elems)
    (
        "nsl_grad_all_reduce",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M46: Deterministic kernel variants ---
    (
        "nsl_tensor_reduce_sum_deterministic",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_reduce_mean_deterministic",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_scatter_add_deterministic",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M46: Global deterministic mode flag + RNG seeding ---
    ("nsl_set_deterministic", &[types::I64], Some(types::I64)),
    ("nsl_rng_seed", &[types::I64], Some(types::I64)),
    // --- M48: Multimodal primitives ---
    (
        "nsl_patch_embed",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_mel_spectrogram",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Explicit-sample-rate variant; the 4-arg form assumes 16 kHz.
    (
        "nsl_mel_spectrogram_sr",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_cross_attention",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_image_resize",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_image_normalize",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_stft",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_audio_resample",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M50: Sparse tensors ---
    (
        "nsl_sparse_coo",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_sparse_from_dense",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ), // M50b: +threshold
    ("nsl_sparse_to_dense", &[types::I64], Some(types::I64)),
    ("nsl_sparse_nnz", &[types::I64], Some(types::I64)),
    ("nsl_sparse_density", &[types::I64], Some(types::I64)),
    (
        "nsl_sparse_spmm",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_sparse_spmv",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_sparse_coo_to_csr", &[types::I64], Some(types::I64)),
    ("nsl_sparse_coo_to_csc", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csr_to_csc", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csc_to_csr", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csr_to_coo", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csc_to_coo", &[types::I64], Some(types::I64)),
    (
        "nsl_sparse_add",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_sparse_mul",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_sparse_free", &[types::I64], Some(types::I64)),
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
    // --- M62: Legacy Interop — DLPack bridge + C API ---
    ("nsl_dlpack_export", &[types::I64], Some(types::I64)),
    ("nsl_dlpack_import", &[types::I64], Some(types::I64)),
    ("nsl_dlpack_free", &[types::I64], None),
    ("nsl_model_create", &[types::I64], Some(types::I64)),
    ("nsl_model_destroy", &[types::I64], Some(types::I64)),
    (
        "nsl_model_forward",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_model_forward_dlpack",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_model_backward",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_model_get_version", &[], Some(types::I64)),
    ("nsl_get_last_error", &[], Some(types::I64)),
    ("nsl_clear_error", &[], Some(types::I64)),
    // --- M40: Source AD runtime helpers ---
    (
        "nsl_tensor_compare",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_where",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_scalar",
        &[types::F64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_pad_zero",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_scatter_add",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_embedding_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_cross_entropy_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_mse_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_l1_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 2: fused SiLU backward — grad * σ(x)*(1 + x*(1-σ(x))).
    (
        "nsl_tensor_silu_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 3: fused Sigmoid backward — grad * y*(1-y), y = σ output.
    (
        "nsl_tensor_sigmoid_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 3: fused Tanh backward — grad * (1 - y*y), y = tanh output.
    (
        "nsl_tensor_tanh_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 4: source-AD in-place suppression guard (on!=0 enter / on==0 leave)
    // — raised around the source-AD forward so FBIP preserves primal inputs.
    ("nsl_set_inplace_suppressed", &[types::I64], None),
    // p4 GELU fix: fused GELU backward — grad * gelu'(x), per-device derivative.
    (
        "nsl_tensor_gelu_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // p9: fused per-param FASE-Deferred AdamW step — one launch for the whole
    // m/v/θ update. (theta, m, v, m_partial, lr, β₁, 1-β₁, β₂, 1-β₂, ε, wd,
    // bc1_inv, bc2_inv) → void.
    (
        "nsl_fase_fused_adamw_step",
        &[
            types::I64, types::I64, types::I64, types::I64,
            types::F64, types::F64, types::F64, types::F64, types::F64,
            types::F64, types::F64, types::F64, types::F64,
        ],
        None,
    ),
    // D1 (CSLA Stage-2): window-backward anti-vacuity counter — mark once per
    // accumulation window at the head of the buffered backward phase, plus the
    // in-process getter for gates.
    ("nsl_csla_window_mark", &[], None),
    ("nsl_csla_window_count", &[], Some(types::I64)),
    // D1b: one-time pointer-tie guard over param_list (aborts loudly on any
    // aliased pair — per-layer in-place updates would corrupt the alias).
    ("nsl_csla_assert_params_unaliased", &[types::I64], None),
    // LSE tape-carry gates: fused-SDPA launch counter (0 = base fwd kernel,
    // 1 = Tier-B tile-skip) — NSL-callable as sdpa_fused_launch_count(v).
    ("nsl_sdpa_fused_launch_count", &[types::I64], Some(types::I64)),
    // Fused-CE tape-carry gates: fused linear-CE launch counter (0 =
    // forward, 1 = forward_large, 2 = backward) — NSL-callable as
    // fused_lce_launch_count(k).
    ("nsl_fused_lce_launch_count", &[types::I64], Some(types::I64)),
    // Fused-CE targets dtype bridge: the kernels read targets as s64 but
    // NSL GPU labels are f32 — materialize/free a device i64 copy around
    // each fused forward/backward FFI.
    ("nsl_fused_lce_targets_i64_alloc", &[types::I64], Some(types::I64)),
    ("nsl_fused_lce_targets_i64_free", &[types::I64], None),
    // D2b weight streaming: pointer-identity host offload of model params
    // (side-table mirrors; tensor pointers never change).
    ("nsl_weight_stream_register", &[types::I64], None),
    ("nsl_weight_stream_upload", &[types::I64], None),
    ("nsl_weight_stream_evict", &[types::I64, types::I64], None),
    ("nsl_weight_stream_upload_all", &[], None),
    // Item 12: re-evict everything after a scoped `upload_all` around a
    // model-touching callback. Arg = writeback (1 if the callback may mutate).
    ("nsl_weight_stream_reevict_all", &[types::I64], None),
    ("nsl_weight_stream_teardown", &[], None),
    ("nsl_weight_stream_upload_count", &[], Some(types::I64)),
    (
        "nsl_tensor_logsoftmax",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_repeat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_rope_inverse",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // BatchNorm + AvgPool2d (proper implementations replacing approximations)
    (
        "nsl_tensor_batchnorm",
        &[types::I64, types::I64, types::I64, types::F64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_avgpool2d",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M54b: Unikernel runtime ---
    (
        "nsl_unikernel_init",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_unikernel_model_alloc",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_unikernel_kv_alloc",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_unikernel_model_pool_stats", &[], Some(types::I64)),
    ("nsl_unikernel_shutdown", &[], Some(types::I64)),
    ("nsl_unikernel_gpu_init", &[types::I64], Some(types::I64)),
    ("nsl_unikernel_gpu_ready", &[], Some(types::I64)),
    ("nsl_unikernel_gpu_device_id", &[], Some(types::I64)),
    // --- M56 v1 agent runtime FFI (Task 16). Signatures from spec §3.4. ---
    // All raw pointers are I64 per the workspace convention; time: u64 is also I64.
    ("nsl_agent_pool_new", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_agent_pool_destroy", &[types::I64], None),
    ("nsl_agent_pool_acquire", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_agent_pool_release", &[types::I64, types::I64], None),
    ("nsl_agent_scheduler_step", &[types::I64], Some(types::I32)),
    ("nsl_agent_mailbox_write", &[types::I64, types::I64, types::I64], Some(types::I32)),
    ("nsl_agent_mailbox_read", &[types::I64], Some(types::I64)),
];

/// Declare all runtime functions as imports in the module.
pub fn declare_runtime_functions(
    module: &mut ObjectModule,
    call_conv: CallConv,
) -> Result<HashMap<String, (FuncId, Signature)>, CodegenError> {
    let mut fns = HashMap::new();

    for &(name, params, ret) in RUNTIME_FUNCTIONS {
        let mut sig = module.make_signature();
        sig.call_conv = call_conv;
        for &p in params {
            sig.params.push(AbiParam::new(p));
        }
        if let Some(r) = ret {
            sig.returns.push(AbiParam::new(r));
        }

        let func_id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| {
                CodegenError::new(format!("failed to declare runtime fn '{name}': {e}"))
            })?;

        fns.insert(name.to_string(), (func_id, sig));
    }

    // CSHA cycle 19 T1 (variant-B): register the new probe FFI symbol behind
    // the `csha_cycle19_probe` feature. Signature = 54 original i64 params
    // (byte-identical to `nsl_flash_attention_csha_backward`) + 2 trailing
    // i64 probe pointers = 56. Non-default; wired only by c19 probe tests.
    // See `docs/superpowers` c19 T1 spec + project_csha_paper_completion_cycle18.md.
    #[cfg(feature = "csha_cycle19_probe")]
    {
        let mut sig = module.make_signature();
        sig.call_conv = call_conv;
        for _ in 0..56 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function(
                "nsl_flash_attention_csha_backward_probe",
                Linkage::Import,
                &sig,
            )
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare runtime fn 'nsl_flash_attention_csha_backward_probe': {e}"
                ))
            })?;

        fns.insert(
            "nsl_flash_attention_csha_backward_probe".to_string(),
            (func_id, sig),
        );
    }

    Ok(fns)
}

#[cfg(test)]
mod tests {
    use super::RUNTIME_FUNCTIONS;

    #[test]
    fn precision_cast_ops_have_signatures() {
        let names: Vec<&str> = RUNTIME_FUNCTIONS.iter().map(|(n, _, _)| *n).collect();
        assert!(names.contains(&"nsl_tensor_cast"), "nsl_tensor_cast missing");
        assert!(names.contains(&"nsl_tensor_cast_into"), "nsl_tensor_cast_into missing");
        assert!(
            names.contains(&"nsl_tensor_zeros_like_dtype"),
            "nsl_tensor_zeros_like_dtype missing"
        );
    }

    #[test]
    fn int8_blockwise_ops_have_signatures() {
        // CPDT §3.2 — the headline 4× memory result. These signatures must
        // match the runtime exports in nsl-runtime/src/tensor/int8_blockwise.rs
        // and the ownership table in ffi_ownership.rs (both produce new owned
        // tensors).
        let table: Vec<(&str, &[cranelift_codegen::ir::Type], Option<cranelift_codegen::ir::Type>)> =
            RUNTIME_FUNCTIONS
                .iter()
                .filter(|(n, _, _)| {
                    *n == "nsl_tensor_quant_int8_blockwise"
                        || *n == "nsl_tensor_dequant_int8_blockwise"
                })
                .map(|(n, p, r)| (*n, *p, *r))
                .collect();
        assert_eq!(table.len(), 2, "INT8 blockwise op pair missing");
        for (name, params, ret) in &table {
            assert_eq!(*ret, Some(cranelift_codegen::ir::types::I64), "{name} must return i64");
            assert!(params.iter().all(|t| *t == cranelift_codegen::ir::types::I64),
                "{name} params must all be I64");
        }
    }

    /// CFTP v6: forward inline-cast wrapper FFIs are registered with the
    /// correct Cranelift signature ([I64] -> I64). Required so wengert_lower
    /// can emit calls to them from compiled NSL.
    #[test]
    fn cftp_v6_cast_wrappers_have_signatures() {
        use cranelift_codegen::ir::types;
        for &name in &["nsl_tensor_to_bf16", "nsl_tensor_to_fp16", "nsl_tensor_to_f32"] {
            let entry = RUNTIME_FUNCTIONS
                .iter()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} missing from RUNTIME_FUNCTIONS"));
            assert_eq!(
                entry.1,
                &[types::I64],
                "{name}: expected params [I64], got {:?}",
                entry.1
            );
            assert_eq!(
                entry.2,
                Some(types::I64),
                "{name}: expected return I64, got {:?}",
                entry.2
            );
        }
    }

    /// CFIE Cycle 6: the engine registration/lifecycle + launch FFIs
    /// are declared with the frozen ABI's arities — all-i64 params,
    /// i64 return — so `declare_runtime_functions` picks them up and
    /// the serve emission can `compile_call_by_name` them.
    #[test]
    fn cfie_cycle6_engine_ffis_have_frozen_abi_signatures() {
        use cranelift_codegen::ir::types;
        let arity = |name: &str| -> usize {
            let entry = RUNTIME_FUNCTIONS
                .iter()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} missing from RUNTIME_FUNCTIONS"));
            assert!(
                entry.1.iter().all(|&t| t == types::I64),
                "{name}: every param must be I64 (frozen ABI), got {:?}",
                entry.1
            );
            assert_eq!(
                entry.2,
                Some(types::I64),
                "{name}: must return I64, got {:?}",
                entry.2
            );
            entry.1.len()
        };
        assert_eq!(arity("nsl_cfie_register_kernel"), 9);
        assert_eq!(arity("nsl_cfie_kv_pool_alloc"), 1);
        assert_eq!(arity("nsl_cfie_engine_finalize"), 0);
        assert_eq!(arity("nsl_cfie_engine_destroy"), 0);
        assert_eq!(arity("nsl_cfie_upload_weight_f16"), 2);
        assert_eq!(arity("nsl_cfie_upload_weight_f32"), 2);
        assert_eq!(arity("nsl_cfie_weights_reset"), 0);
        assert_eq!(arity("nsl_cfie_launch_decode_attn"), 5);
        assert_eq!(arity("nsl_cfie_launch_fused_sample"), 6);
        assert_eq!(arity("nsl_cfie_launch_decode_block"), 14);
        assert_eq!(arity("nsl_cfie_launch_spec_verify"), 5);
        assert_eq!(arity("nsl_cfie_launch_spec_reject"), 6);
        assert_eq!(arity("nsl_cfie_launch_quant_attn"), 7);
        assert_eq!(arity("nsl_cfie_decode_step"), 11);
        assert_eq!(arity("nsl_cfie_bind_model"), 8);
        assert_eq!(arity("nsl_cfie_generate"), 7);
        assert_eq!(arity("nsl_cfie_generate_reset"), 0);
        assert_eq!(arity("nsl_cfie_tokens_to_tensor"), 2);
        assert_eq!(arity("nsl_cfie_tensor_to_tokens"), 3);
        // CFIE Cycle 13 (G15): draft binding + pool + launch trio +
        // the speculative decode driver — arity-pinned against the
        // frozen all-i64 engine ABI.
        assert_eq!(arity("nsl_cfie_bind_draft_model"), 8);
        assert_eq!(arity("nsl_cfie_draft_pool_alloc"), 1);
        assert_eq!(arity("nsl_cfie_draft_reset"), 0);
        assert_eq!(arity("nsl_cfie_launch_draft_block"), 4);
        assert_eq!(arity("nsl_cfie_launch_draft_sample"), 4);
        assert_eq!(arity("nsl_cfie_launch_verify_probs"), 2);
        assert_eq!(arity("nsl_cfie_speculative_generate"), 8);
    }
}
