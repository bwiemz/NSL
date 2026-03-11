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
    ("nsl_kernel_launch", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_to_device", &[types::I64, types::I64], Some(types::I64)),
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
