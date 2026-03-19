use nsl_ast::Symbol;
use nsl_errors::Span;
use nsl_lexer::Interner;

use crate::scope::{ScopeId, ScopeMap, SymbolInfo};
use crate::types::{Type, Shape, DType, Device};

/// Register all built-in symbols in the root scope.
pub fn register_builtins(scopes: &mut ScopeMap, interner: &mut Interner) {
    let root = ScopeId::ROOT;

    let mut def = |name: &str, ty: Type| {
        let sym = Symbol(interner.get_or_intern(name));
        let _ = scopes.declare(
            root,
            sym,
            SymbolInfo {
                ty,
                def_span: Span::DUMMY,
                is_const: true,
                is_param: false,
                is_used: true,
            },
        );
    };

    // Built-in functions
    def(
        "print",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "range",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Int))),
        },
    );
    def(
        "len",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "abs",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
        },
    );
    def(
        "min",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
        },
    );
    def(
        "max",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
        },
    );

    // Tensor creation functions
    let tensor_ret = Type::Tensor {
        shape: Shape::unknown(),
        dtype: DType::F64,
        device: Device::Cpu,
    };
    for name in &["zeros", "ones", "rand", "randn", "empty"] {
        def(
            name,
            Type::Function {
                params: vec![Type::List(Box::new(Type::Int))],
                ret: Box::new(tensor_ret.clone()),
            },
        );
    }
    // full(shape, value) -> Tensor
    def(
        "full",
        Type::Function {
            params: vec![Type::List(Box::new(Type::Int)), Type::Float],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // arange(start, stop, step) -> Tensor (variadic via Unknown)
    def(
        "arange",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // Activation functions and tensor trig (take tensor, return tensor)
    for name in &["relu", "gelu", "silu", "sigmoid", "tanh", "tensor_sin", "tensor_cos"] {
        def(
            name,
            Type::Function {
                params: vec![tensor_ret.clone()],
                ret: Box::new(tensor_ret.clone()),
            },
        );
    }
    // softmax(tensor, dim) -> tensor
    def(
        "softmax",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // NN layer constructors
    for name in &[
        "Linear",
        "Conv2d",
        "Conv1d",
        "LayerNorm",
        "BatchNorm",
        "Dropout",
        "Embedding",
        "MultiheadAttention",
    ] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Unknown),
            },
        );
    }

    // Training mode
    def(
        "is_training",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Bool),
        },
    );
    def(
        "set_training_mode",
        Type::Function {
            params: vec![Type::Bool],
            ret: Box::new(Type::Void),
        },
    );

    // Utility functions
    def(
        "assert",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "assert_eq",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "assert_close",
        Type::Function {
            params: vec![
                Type::Tensor { shape: Shape::unknown(), dtype: DType::Unknown, device: Device::Unknown },
                Type::Tensor { shape: Shape::unknown(), dtype: DType::Unknown, device: Device::Unknown },
                Type::Float,
                Type::Float,
            ],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "exit",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "read_file",
        Type::Function {
            params: vec![Type::Str],
            ret: Box::new(Type::Str),
        },
    );
    def(
        "write_file",
        Type::Function {
            params: vec![Type::Str, Type::Str],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "append_file",
        Type::Function {
            params: vec![Type::Str, Type::Str],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "file_exists",
        Type::Function {
            params: vec![Type::Str],
            ret: Box::new(Type::Bool),
        },
    );
    def(
        "args",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::List(Box::new(Type::Str))),
        },
    );

    // Optimizer constructors (used in train block DSL: optimizer: SGD(lr=0.01))
    for name in &["SGD", "Adam", "AdamW", "Lion", "Muon", "SOAP"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Unknown),
            },
        );
    }

    // Tensor mutation: copy_data(dest, src) — in-place copy
    def(
        "copy_data",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );

    // embedding_lookup(weight, indices) -> tensor
    def(
        "embedding_lookup",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // layernorm(input, weight, bias, eps) -> tensor
    def(
        "layernorm",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), tensor_ret.clone(), Type::Float],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // rmsnorm(input, weight, eps) -> tensor
    def(
        "rmsnorm",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), Type::Float],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // dropout(tensor, p, training) -> tensor
    def(
        "dropout",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Float, Type::Bool],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w) -> tensor
    def(
        "conv2d",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), tensor_ret.clone(), Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // maxpool2d(input, kernel_h, kernel_w, stride, padding) -> tensor
    def(
        "maxpool2d",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // bias_add(tensor, bias) -> tensor — broadcasts 1D bias over 2D tensor
    def(
        "bias_add",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // Tensor reduction / element-wise builtins (take tensor(s), return tensor)
    for name in &["mean", "sum", "neg", "clamp", "reduce_max", "gather"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Unknown),
            },
        );
    }

    // tensor_slice(tensor, dim, start, end) -> tensor
    def(
        "tensor_slice",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // tensor_cat(list_of_tensors, dim) -> tensor
    def(
        "tensor_cat",
        Type::Function {
            params: vec![Type::List(Box::new(tensor_ret.clone())), Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // M18a shape manipulation free functions
    // unsqueeze(tensor, dim) -> tensor
    def(
        "unsqueeze",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // stack(list_of_tensors, dim) -> tensor
    def(
        "stack",
        Type::Function {
            params: vec![Type::List(Box::new(tensor_ret.clone())), Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // causal_mask(seq_len) -> tensor
    def(
        "causal_mask",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // M27: scaled_dot_product_attention(Q, K, V, scale, causal=true) -> tensor
    // 5th param (causal bool) is optional — semantic checker allows fewer args than declared
    def(
        "scaled_dot_product_attention",
        Type::Function {
            params: vec![
                tensor_ret.clone(),  // Q
                tensor_ret.clone(),  // K
                tensor_ret.clone(),  // V
                Type::Float,         // scale
                Type::Unknown,       // causal (optional, makes it variadic)
            ],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // Tokenizer functions (M15)
    def(
        "byte_tokenizer_new",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "bpe_train",
        Type::Function {
            params: vec![Type::Str, Type::Int, Type::Int, Type::List(Box::new(Type::Str))],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "tokenizer_load",
        Type::Function {
            params: vec![Type::Str],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "tokenizer_save",
        Type::Function {
            params: vec![Type::Int, Type::Str],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "tokenizer_encode",
        Type::Function {
            params: vec![Type::Int, Type::Str],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    def(
        "tokenizer_decode",
        Type::Function {
            params: vec![Type::Int, tensor_ret.clone()],
            ret: Box::new(Type::Str),
        },
    );
    def(
        "tokenizer_vocab_size",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "tokenizer_encode_batch",
        Type::Function {
            params: vec![Type::Int, Type::List(Box::new(Type::Str)), Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::List(Box::new(tensor_ret.clone()))),
        },
    );

    // Quantization functions (M16)
    // nsl_qtensor_quantize(tensor, dtype, granularity, axis, group_size) -> QuantizedTensor
    def(
        "nsl_qtensor_quantize",
        Type::Function {
            params: vec![Type::Unknown, Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::QuantizedTensor),
        },
    );
    // nsl_qtensor_dequantize(qtensor) -> Tensor
    def(
        "nsl_qtensor_dequantize",
        Type::Function {
            params: vec![Type::QuantizedTensor],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // nsl_qtensor_matmul_mixed(tensor, qtensor) -> Tensor
    def(
        "nsl_qtensor_matmul_mixed",
        Type::Function {
            params: vec![Type::Unknown, Type::QuantizedTensor],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    // nsl_qtensor_dtype(qtensor) -> Int
    def(
        "nsl_qtensor_dtype",
        Type::Function {
            params: vec![Type::QuantizedTensor],
            ret: Box::new(Type::Int),
        },
    );
    // nsl_qtensor_shape(qtensor) -> Tensor (1D shape as f64 values)
    def(
        "nsl_qtensor_shape",
        Type::Function {
            params: vec![Type::QuantizedTensor],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // Math functions (always return Float — coerce int args at codegen level)
    for name in &["sqrt", "log", "exp", "sin", "cos", "floor"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Float),
            },
        );
    }

    // Type conversion functions
    def(
        "int",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "float",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Float),
        },
    );
    def(
        "bool",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Bool),
        },
    );
    def(
        "str",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Str),
        },
    );
    def("void", Type::Void);

    // GPU kernel intrinsics (M17)
    for name in &["thread_id", "thread_id_y", "block_id", "block_id_y", "block_dim"] {
        def(
            name,
            Type::Function {
                params: vec![],
                ret: Box::new(Type::Int),
            },
        );
    }
    def(
        "sync_threads",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
        },
    );
    // Device identifier constants (M17)
    def("cuda", Type::Int);
    def("cpu", Type::Int);

    // Dtype identifier constants (M23 BYOD) — used in .to(f32), .to(f64) etc.
    def("f32", Type::Int);
    def("f64", Type::Int);

    // Higher-order functions
    def(
        "map",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
        },
    );
    def(
        "filter",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
        },
    );
    def(
        "enumerate",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
        },
    );
    def(
        "zip",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
        },
    );
    def(
        "sorted",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
        },
    );
    def(
        "reversed",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
        },
    );

    // Interop intrinsics (M18b)
    // load_safetensors(path, device=0) -> Dict[str, Tensor]
    def(
        "load_safetensors",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Dict(Box::new(Type::Str), Box::new(tensor_ret.clone()))),
        },
    );
    // save_safetensors(dict, path) -> void
    def(
        "save_safetensors",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );
    // from_hf(repo_id, model, device=0) -> model
    def(
        "from_hf",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
        },
    );
    // to_onnx(model, input, path) -> void
    def(
        "to_onnx",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );

    // Data pipeline intrinsics (M19)
    def(
        "load_jsonl",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Str))),
        },
    );
    def(
        "load_csv",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Str))),
        },
    );
    def(
        "load_mmap",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    def(
        "DataLoader",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
        },
    );

    // Sampling intrinsics (M19)
    def(
        "topk",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Dict(Box::new(Type::Str), Box::new(tensor_ret.clone()))),
        },
    );
    def(
        "multinomial",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    def(
        "argmax",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    def(
        "manual_seed",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "cumsum",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );
    def(
        "lt_scalar",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // Paged KV-cache functions (M25)
    def(
        "kv_cache_init",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_init_gpu",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_alloc_seq",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_append",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_k_ptr",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_v_ptr",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_free_seq",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "kv_cache_seq_len",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_seq_blocks",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_seq_num_blocks",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
        },
    );
    def(
        "kv_cache_utilization",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Float),
        },
    );
    def(
        "kv_cache_destroy",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Void),
        },
    );

    // Memory profiler functions (M25)
    def(
        "profiler_start",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "profiler_stop",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "profiler_peak",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Int),
        },
    );
    // Kernel profiler builtins (M26)
    def(
        "kernel_profiler_start",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
        },
    );
    def(
        "kernel_profiler_stop",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
        },
    );

    // M32: MoE dispatch intrinsic
    def(
        "moe_dispatch",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
        },
    );

    // M33: Speculative decode step intrinsic
    def(
        "speculative_decode",
        Type::Function {
            params: vec![
                tensor_ret.clone(),  // draft_tokens
                tensor_ret.clone(),  // draft_logits
                tensor_ret.clone(),  // verifier_logits
                Type::Int,           // vocab_size
            ],
            ret: Box::new(tensor_ret.clone()),
        },
    );
}
