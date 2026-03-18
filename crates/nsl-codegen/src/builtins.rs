use std::collections::HashMap;
use cranelift_codegen::ir::{types, AbiParam, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::ObjectModule;

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
    // List
    ("nsl_list_new", &[], Some(types::I64)),
    ("nsl_list_push", &[types::I64, types::I64], None),
    ("nsl_list_get", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_list_len", &[types::I64], Some(types::I64)),
    ("nsl_list_set", &[types::I64, types::I64, types::I64], None),
    ("nsl_list_contains", &[types::I64, types::I64], Some(types::I8)),
    ("nsl_list_free", &[types::I64], None),
    // String
    ("nsl_str_concat", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_int_to_str", &[types::I64], Some(types::I64)),
    ("nsl_float_to_str", &[types::F64], Some(types::I64)),
    ("nsl_bool_to_str", &[types::I8], Some(types::I64)),
    // Type conversions
    ("nsl_str_to_int", &[types::I64], Some(types::I64)),
    ("nsl_str_to_float", &[types::I64], Some(types::F64)),
    ("nsl_str_len", &[types::I64], Some(types::I64)),
    // Range
    ("nsl_range", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // Dict
    ("nsl_dict_new", &[], Some(types::I64)),
    ("nsl_dict_set_str", &[types::I64, types::I64, types::I64], None),
    ("nsl_dict_get_str", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_dict_len", &[types::I64], Some(types::I64)),
    ("nsl_dict_contains", &[types::I64, types::I64], Some(types::I8)),
    ("nsl_dict_keys", &[types::I64], Some(types::I64)),
    ("nsl_dict_free", &[types::I64], None),
    // String comparison
    ("nsl_str_eq", &[types::I64, types::I64], Some(types::I64)),
    // String repeat & slice
    ("nsl_str_repeat", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_list_slice", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_str_slice", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    // String methods
    ("nsl_str_upper", &[types::I64], Some(types::I64)),
    ("nsl_str_lower", &[types::I64], Some(types::I64)),
    ("nsl_str_strip", &[types::I64], Some(types::I64)),
    ("nsl_str_split", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_str_join", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_str_replace", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_str_find", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_str_startswith", &[types::I64, types::I64], Some(types::I8)),
    ("nsl_str_endswith", &[types::I64, types::I64], Some(types::I8)),
    ("nsl_str_contains", &[types::I64, types::I64], Some(types::I8)),
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
    ("nsl_tensor_full", &[types::I64, types::F64], Some(types::I64)),
    ("nsl_tensor_arange", &[types::F64, types::F64, types::F64], Some(types::I64)),
    // Tensor element access
    ("nsl_tensor_get", &[types::I64, types::I64], Some(types::F64)),
    ("nsl_tensor_set", &[types::I64, types::I64, types::F64], None),
    // Tensor shape ops
    ("nsl_tensor_shape", &[types::I64], Some(types::I64)),
    ("nsl_tensor_shape_dim", &[types::I64, types::I64], Some(types::I64)),
    // M28: Dynamic shape assertions
    ("nsl_tensor_assert_dim", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_assert_dim_bound", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_ndim", &[types::I64], Some(types::I64)),
    ("nsl_tensor_reshape", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_transpose", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // Tensor arithmetic (elementwise)
    ("nsl_tensor_add", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_sub", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_mul", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_div", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_neg", &[types::I64], Some(types::I64)),
    // Tensor scalar ops
    ("nsl_tensor_add_scalar", &[types::I64, types::F64], Some(types::I64)),
    ("nsl_tensor_mul_scalar", &[types::I64, types::F64], Some(types::I64)),
    // Tensor matmul
    ("nsl_tensor_matmul", &[types::I64, types::I64], Some(types::I64)),
    // Tensor reductions (return scalar tensor ptr, not f64)
    ("nsl_tensor_sum", &[types::I64], Some(types::I64)),
    ("nsl_tensor_mean", &[types::I64], Some(types::I64)),
    // Tensor scalar extraction
    ("nsl_tensor_item", &[types::I64], Some(types::F64)),
    // Tensor display
    ("nsl_tensor_print", &[types::I64], None),
    // Tensor memory
    ("nsl_tensor_clone", &[types::I64], Some(types::I64)),
    ("nsl_tensor_free", &[types::I64], None),
    // Autodiff tape management
    ("nsl_tape_start", &[types::I64], None),
    ("nsl_tape_stop", &[], None),
    ("nsl_tape_backward", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tape_pause", &[], None),
    ("nsl_tape_resume", &[], None),
    // Element-wise tensor ops (M14)
    ("nsl_tensor_exp", &[types::I64], Some(types::I64)),
    ("nsl_tensor_log", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sqrt", &[types::I64], Some(types::I64)),
    ("nsl_tensor_abs", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sign", &[types::I64], Some(types::I64)),
    ("nsl_tensor_clamp", &[types::I64, types::F64, types::F64], Some(types::I64)),
    // Dimensional reductions (M14)
    ("nsl_tensor_sum_dim", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_mean_dim", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_reduce_max", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_gather", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // In-place mutation ops (M14)
    ("nsl_tensor_copy_data", &[types::I64, types::I64], None),
    ("nsl_tensor_add_inplace", &[types::I64, types::I64], None),
    ("nsl_tensor_zero_inplace", &[types::I64], None),
    ("nsl_tensor_zeros_like", &[types::I64], Some(types::I64)),
    // Gradient clipping (M14)
    ("nsl_clip_grad_norm", &[types::I64, types::F64], None),
    // Checkpoint I/O (M14)
    ("nsl_model_save", &[types::I64, types::I64, types::I64, types::I64], None),
    ("nsl_model_load", &[types::I64, types::I64, types::I64], None),
    // Scalar math (M14)
    ("nsl_floor", &[types::F64], Some(types::F64)),
    // Activation functions (M15)
    ("nsl_tensor_relu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_gelu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_silu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sigmoid", &[types::I64], Some(types::I64)),
    ("nsl_tensor_tanh_act", &[types::I64], Some(types::I64)),
    ("nsl_tensor_softmax", &[types::I64, types::I64], Some(types::I64)),
    // Slice & Cat (M15)
    ("nsl_tensor_slice", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cat", &[types::I64, types::I64], Some(types::I64)),
    // Embedding lookup (M15)
    ("nsl_tensor_embedding_lookup", &[types::I64, types::I64], Some(types::I64)),
    // LayerNorm & RMSNorm (M15)
    ("nsl_tensor_layernorm", &[types::I64, types::I64, types::I64, types::F64], Some(types::I64)),
    ("nsl_tensor_rmsnorm", &[types::I64, types::I64, types::F64], Some(types::I64)),
    // Dropout, Conv2d, MaxPool2d (M15)
    ("nsl_tensor_dropout", &[types::I64, types::F64, types::I8], Some(types::I64)),
    ("nsl_tensor_conv2d", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_maxpool2d", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    // Bias add (M15 — broadcast 1D bias over 2D tensor)
    ("nsl_tensor_bias_add", &[types::I64, types::I64], Some(types::I64)),
    // String deallocation (M15)
    ("nsl_string_free", &[types::I64], None),
    // Assert functions (M15 test framework)
    ("nsl_assert_eq_int", &[types::I64, types::I64, types::I64, types::I64], None),
    ("nsl_assert_eq_float", &[types::F64, types::F64, types::I64, types::I64], None),
    ("nsl_assert_close", &[types::I64, types::I64, types::F64, types::F64, types::I64, types::I64], None),
    // Tokenizer functions (M15)
    ("nsl_byte_tokenizer_new", &[], Some(types::I64)),
    ("nsl_bpe_train", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tokenizer_load", &[types::I64], Some(types::I64)),
    ("nsl_tokenizer_save", &[types::I64, types::I64], None),
    ("nsl_tokenizer_encode", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tokenizer_decode", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tokenizer_vocab_size", &[types::I64], Some(types::I64)),
    ("nsl_tokenizer_encode_batch", &[types::I64, types::I64, types::I8, types::I8, types::I64], Some(types::I64)),
    // Quantization (M16)
    ("nsl_qtensor_quantize", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_qtensor_dequantize", &[types::I64], Some(types::I64)),
    ("nsl_qtensor_matmul_mixed", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_qtensor_free", &[types::I64], None),
    ("nsl_qtensor_addref", &[types::I64], None),
    ("nsl_qtensor_release", &[types::I64], None),
    ("nsl_qtensor_dtype", &[types::I64], Some(types::I64)),
    ("nsl_qtensor_shape", &[types::I64], Some(types::I64)),
    // Tensor creation helpers (M17)
    ("nsl_tensor_zeros_on", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_ones_like", &[types::I64], Some(types::I64)),
    // GPU runtime functions (M17)
    ("nsl_cuda_init", &[], Some(types::I64)),
    ("nsl_kernel_launch", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_to_device", &[types::I64, types::I64], Some(types::I64)),
    // Shape manipulation ops (M18a)
    ("nsl_tensor_unsqueeze", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_select", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_stack", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_expand", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_causal_mask", &[types::I64], Some(types::I64)),
    // Safetensors I/O (M18b)
    ("nsl_safetensors_load", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_safetensors_save", &[types::I64, types::I64, types::I64], None),
    // HuggingFace Hub download + weight loading (M18b)
    ("nsl_hf_load", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    // Trace infrastructure for ONNX export (M18b Task 7)
    ("nsl_trace_start", &[], None),
    ("nsl_trace_register_input", &[types::I64, types::I64], None),
    ("nsl_trace_register_output", &[types::I64, types::I64], None),
    ("nsl_trace_stop", &[], Some(types::I64)),
    // ONNX export (M18b Tasks 9-10)
    ("nsl_onnx_export", &[types::I64, types::I64, types::I64], None),
    // Sampling primitives (M19)
    ("nsl_manual_seed", &[types::I64], None),
    ("nsl_tensor_topk", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_multinomial", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_argmax", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cumsum", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_lt_scalar", &[types::I64, types::F64], Some(types::I64)),
    // Tensor mutation (M19)
    ("nsl_tensor_set_element", &[types::I64, types::I64, types::I64, types::F64], None),
    ("nsl_tensor_slice_assign", &[types::I64, types::I64, types::I64, types::I64], None),
    // Data sources (M19)
    ("nsl_load_jsonl", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_load_csv", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_load_mmap", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // DataLoader (M19)
    ("nsl_dataloader_create", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_dataloader_start", &[types::I64], None),
    ("nsl_dataloader_next_batch", &[types::I64], Some(types::I64)),
    ("nsl_dataloader_reset", &[types::I64], None),
    ("nsl_dataloader_stop", &[types::I64], None),
    ("nsl_dataloader_free", &[types::I64], None),
    // Packing efficiency (M19)
    ("nsl_packing_efficiency", &[types::I64], Some(types::F64)),
    // Custom dtype registry (M23)
    ("nsl_register_custom_dtype", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], None),
    ("nsl_finalize_dtype_registry", &[], None),
    ("nsl_tensor_to_custom_dtype", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_from_custom_dtype", &[types::I64], Some(types::I64)),
    // Standalone weight provider and arg parser (M24)
    ("nsl_standalone_init_embedded",     &[types::I64, types::I64],                                     None),
    ("nsl_standalone_init_sidecar",      &[types::I64, types::I64],                                     None),
    ("nsl_standalone_has_weights",       &[],                                                            Some(types::I64)),
    ("nsl_standalone_args_init",         &[types::I64, types::I64],                                     None),
    ("nsl_standalone_arg_str",           &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_standalone_arg_str_default",   &[types::I64, types::I64, types::I64, types::I64],             Some(types::I64)),
    ("nsl_standalone_arg_int",           &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_standalone_arg_int_default",   &[types::I64, types::I64, types::I64],                         Some(types::I64)),
    ("nsl_standalone_arg_float",         &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_standalone_arg_float_default", &[types::I64, types::I64, types::I64],                         Some(types::I64)),
    ("nsl_standalone_args_finish",       &[],                                                            None),
    // Paged KV-cache (M25)
    ("nsl_kv_cache_init",       &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_cache_init_gpu",   &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_cache_alloc_seq",  &[types::I64],                                                 Some(types::I64)),
    ("nsl_kv_cache_append",     &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_kv_cache_k_ptr",      &[types::I64, types::I64, types::I64],                         Some(types::I64)),
    ("nsl_kv_cache_v_ptr",      &[types::I64, types::I64, types::I64],                         Some(types::I64)),
    ("nsl_kv_cache_free_seq",   &[types::I64, types::I64],                                     None),
    ("nsl_kv_cache_seq_len",    &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_kv_cache_seq_blocks", &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_kv_cache_seq_num_blocks", &[types::I64, types::I64],                                 Some(types::I64)),
    ("nsl_kv_cache_utilization", &[types::I64],                                                Some(types::F64)),
    ("nsl_kv_cache_destroy",    &[types::I64],                                                 None),
    // Memory profiler (M25)
    ("nsl_profiler_start",      &[types::I64],                                                 None),
    ("nsl_profiler_stop",       &[],                                                           None),
    ("nsl_profiler_dump",       &[types::I64, types::I64],                                     None),
    ("nsl_profiler_peak",       &[],                                                           Some(types::I64)),
    // Kernel profiler (M26) — flush is NOT registered here (Rust-only atexit call)
    ("nsl_kernel_profiler_start", &[],                                                         None),
    ("nsl_kernel_profiler_stop",  &[],                                                         None),
    // FlashAttention-2 launch wrappers (M27)
    ("nsl_flash_attention", &[
        types::I64, types::I64, types::I64, types::I64, types::I64,  // q, k, v, out, scale
        types::I64, types::I64, types::I64, types::I64,              // batch, heads, seq_len, head_dim
        types::I64, types::I64, types::I64, types::I64,              // block_table, k_pool, v_pool, block_size
        types::I64, types::I64,                                       // cos, sin (RoPE)
        types::I64, types::I64,                                       // seq_ids, seq_lens (M29-ready)
        types::I64,                                                   // shared_mem_bytes
        types::I64, types::I64,                                       // ptx_ptr, name_ptr
        types::I64, types::I64,                                       // block_q, block_kv
    ], Some(types::I64)),
    ("nsl_rope_cache_write", &[
        types::I64, types::I64,                                       // k_projected, v_projected
        types::I64, types::I64, types::I64,                          // cos, sin, positions
        types::I64, types::I64, types::I64,                          // k_pool, v_pool, block_table
        types::I64, types::I64,                                       // seq_ids, seq_lens (M29-ready)
        types::I64, types::I64, types::I64, types::I64,              // num_tokens, num_heads, head_dim, block_size
        types::I64, types::I64,                                       // ptx_ptr, name_ptr
    ], Some(types::I64)),
    // M29: Serving engine
    ("nsl_serve_init",             &[types::I64, types::I64, types::I64, types::I64],       Some(types::I64)),
    ("nsl_serve_enqueue",          &[types::I64, types::I64, types::I64, types::F64, types::F64],  Some(types::I64)),
    ("nsl_serve_step",             &[],                         Some(types::I64)),
    ("nsl_serve_record_token",     &[types::I64, types::I64],                 Some(types::I64)),
    ("nsl_serve_drain_completed",  &[],                         Some(types::I64)),
    ("nsl_serve_has_work",         &[],                         Some(types::I64)),
    ("nsl_serve_completed_count",  &[],                         Some(types::I64)),
    ("nsl_serve_preempt",          &[types::I64],                      Some(types::I64)),
    ("nsl_serve_destroy",          &[],                         Some(types::I64)),
    // --- M30: Tensor parallelism ---
    ("nsl_tp_init", &[], Some(types::I64)),
    ("nsl_tp_rank", &[], Some(types::I64)),
    ("nsl_tp_world_size", &[], Some(types::I64)),
    ("nsl_tp_all_reduce_sum", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tp_all_gather", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tp_broadcast", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tp_barrier", &[], Some(types::I64)),
    ("nsl_tp_destroy", &[], Some(types::I64)),
    // --- M32: MoE runtime functions ---
    ("nsl_moe_route", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_moe_scatter", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_expert_parallel_matmul", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_moe_gather", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_moe_all_to_all", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_moe_aux_loss", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_moe_dispatch_full", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    // --- M33: Speculative decoding runtime functions ---
    ("nsl_speculative_draft", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_speculative_verify", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_speculative_build_tree", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_speculative_verify_tree", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_page_branch", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_page_cow_copy", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tree_attention", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_speculative_cleanup", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_speculative_decode_step", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    // --- M34: Context parallelism (ring attention) ---
    ("nsl_cp_init", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_sequence_partition", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_ring_attention", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_ring_send_recv", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_sequence_gather", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_cp_destroy", &[types::I64], Some(types::I64)),
    // --- M35: FP8 compute ---
    ("nsl_fp8_cast", &[types::I64, types::I64, types::F64], Some(types::I64)),
    ("nsl_fp8_matmul", &[types::I64, types::I64, types::F64, types::F64], Some(types::I64)),
    ("nsl_fp8_compute_scale", &[types::I64, types::I64], Some(types::F64)),
    ("nsl_fp8_update_calibration", &[types::I64, types::I64, types::F64], Some(types::F64)),
    // --- M35: AWQ 4-bit quantization ---
    ("nsl_awq_quantize", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_awq_matmul", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_awq_free", &[types::I64], None),
    // --- M35: GPTQ quantization ---
    ("nsl_gptq_quantize", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_gptq_matmul", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_gptq_free", &[types::I64], None),
    // --- M36: Memory planning slab ---
    ("nsl_slab_alloc", &[types::I64], Some(types::I64)),
    ("nsl_slab_free", &[types::I64, types::I64], None),
    ("nsl_slab_offset", &[types::I64, types::I64], Some(types::I64)),
    // --- M42: KV-cache compression ---
    ("nsl_kv_quantize_and_store", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_sliding_window_init", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_sliding_window_check", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_sliding_window_destroy", &[], Some(types::I64)),
    ("nsl_kv_h2o_init", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_h2o_accumulate", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_h2o_check", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_kv_h2o_remove_sequence", &[types::I64], Some(types::I64)),
    ("nsl_kv_h2o_destroy", &[], Some(types::I64)),
    ("nsl_kv_compress_ratio", &[types::I64], Some(types::I64)),
    // --- M44: Constrained decoding (grammar FSM) ---
    ("nsl_grammar_init", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_grammar_step", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_grammar_apply_mask", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_grammar_is_accept", &[types::I64], Some(types::I64)),
    ("nsl_grammar_start_state", &[], Some(types::I64)),
    ("nsl_grammar_destroy", &[], Some(types::I64)),
    // --- M39b: vmap runtime ---
    ("nsl_vmap_check_batch", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // --- M40b: Backward context for source-to-source AD ---
    ("nsl_backward_ctx_new", &[types::I64], Some(types::I64)),
    ("nsl_backward_ctx_save", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_backward_ctx_load", &[types::I64], Some(types::I64)),
    ("nsl_backward_ctx_free", &[], Some(types::I64)),
    // --- M43: Pipeline parallelism ---
    ("nsl_pipeline_init", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_pipeline_send", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_pipeline_recv", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_pipeline_send_grad", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_pipeline_recv_grad", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_pipeline_barrier", &[], Some(types::I64)),
    ("nsl_pipeline_destroy", &[], Some(types::I64)),
    // --- M43: ZeRO optimizer ---
    ("nsl_zero_init", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_zero_partition", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_zero_reduce_grads", &[], Some(types::I64)),
    ("nsl_zero_step", &[types::I64], Some(types::I64)),
    ("nsl_zero_destroy", &[], Some(types::I64)),
    // --- M43: Gradient accumulation ---
    ("nsl_grad_accumulate_add", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_grad_zero", &[types::I64], Some(types::I64)),
    ("nsl_grad_all_reduce", &[types::I64, types::I64], Some(types::I64)),
    // --- M46: Deterministic kernel variants ---
    ("nsl_tensor_reduce_sum_deterministic", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_reduce_mean_deterministic", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_scatter_add_deterministic", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // --- M48: Multimodal primitives ---
    ("nsl_patch_embed", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_mel_spectrogram", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_cross_attention", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_image_resize", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_image_normalize", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_stft", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_audio_resample", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // --- M45: Tensor debugger trace ---
    ("nsl_trace_init", &[], Some(types::I64)),
    ("nsl_trace_record_op", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_trace_suppress", &[], Some(types::I64)),
    ("nsl_trace_unsuppress", &[], Some(types::I64)),
    ("nsl_trace_breakpoint", &[], Some(types::I64)),
    ("nsl_trace_flush", &[], Some(types::I64)),
    ("nsl_trace_destroy", &[], Some(types::I64)),
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
            .map_err(|e| CodegenError::new(format!("failed to declare runtime fn '{name}': {e}")))?;

        fns.insert(name.to_string(), (func_id, sig));
    }

    Ok(fns)
}
