use nsl_ast::Symbol;
use nsl_errors::Span;
use nsl_lexer::Interner;

use crate::scope::{ScopeId, ScopeMap, SymbolInfo};
use crate::types::{Type, Shape, DType, Device, Effect};

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
            effect: Effect::Inferred,
        },
    );
    def(
        "range",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Int))),
            effect: Effect::Inferred,
        },
    );
    def(
        "len",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "abs",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
            effect: Effect::Inferred,
        },
    );
    def(
        "min",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
            effect: Effect::Inferred,
        },
    );
    def(
        "max",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
            effect: Effect::Inferred,
        },
    );

    // Benchmarking intrinsics
    def(
        "clock",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Float),
            effect: Effect::Inferred,
        },
    );
    def(
        "alloc_reset",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "alloc_count",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "alloc_bytes",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
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
                effect: Effect::Inferred,
            },
        );
    }
    // full(shape, value) -> Tensor
    def(
        "full",
        Type::Function {
            params: vec![Type::List(Box::new(Type::Int)), Type::Float],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // arange(start, stop, step) -> Tensor (variadic via Unknown)
    def(
        "arange",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // Activation functions, tensor trig, and rotate_half (take tensor, return tensor)
    for name in &["relu", "gelu", "silu", "sigmoid", "tanh", "tensor_sin", "tensor_cos", "rotate_half"] {
        def(
            name,
            Type::Function {
                params: vec![tensor_ret.clone()],
                ret: Box::new(tensor_ret.clone()),
                effect: Effect::Inferred,
            },
        );
    }
    // softmax(tensor, dim) -> tensor
    def(
        "softmax",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // log_softmax(tensor, dim) -> tensor
    def(
        "log_softmax",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
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
                effect: Effect::Inferred,
            },
        );
    }

    // Training mode
    def(
        "is_training",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Bool),
            effect: Effect::Inferred,
        },
    );
    def(
        "set_training_mode",
        Type::Function {
            params: vec![Type::Bool],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );

    // Utility functions
    def(
        "assert",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "assert_eq",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
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
            effect: Effect::Inferred,
        },
    );
    def(
        "exit",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "read_line",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Str),
            effect: Effect::Inferred,
        },
    );
    def(
        "read_file",
        Type::Function {
            params: vec![Type::Str],
            ret: Box::new(Type::Str),
            effect: Effect::Inferred,
        },
    );
    def(
        "write_file",
        Type::Function {
            params: vec![Type::Str, Type::Str],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "append_file",
        Type::Function {
            params: vec![Type::Str, Type::Str],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "file_exists",
        Type::Function {
            params: vec![Type::Str],
            ret: Box::new(Type::Bool),
            effect: Effect::Inferred,
        },
    );
    def(
        "args",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::List(Box::new(Type::Str))),
            effect: Effect::Inferred,
        },
    );

    // Optimizer constructors (used in train block DSL: optimizer: SGD(lr=0.01))
    for name in &["SGD", "Adam", "AdamW", "Lion", "Muon", "SOAP"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Unknown),
                effect: Effect::Inferred,
            },
        );
    }

    // Tensor mutation: copy_data(dest, src) — in-place copy
    def(
        "copy_data",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );

    // embedding_lookup(weight, indices) -> tensor
    def(
        "embedding_lookup",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // layernorm(input, weight, bias, eps) -> tensor
    def(
        "layernorm",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), tensor_ret.clone(), Type::Float],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // rmsnorm(input, weight, eps) -> tensor
    def(
        "rmsnorm",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), Type::Float],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // dropout(tensor, p, training) -> tensor
    def(
        "dropout",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Float, Type::Bool],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w) -> tensor
    def(
        "conv2d",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), tensor_ret.clone(), Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // maxpool2d(input, kernel_h, kernel_w, stride, padding) -> tensor
    def(
        "maxpool2d",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // bias_add(tensor, bias) -> tensor — broadcasts 1D bias over 2D tensor
    def(
        "bias_add",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // Tensor reduction / element-wise builtins (take tensor(s), return tensor)
    for name in &["mean", "sum", "neg", "clamp", "reduce_max", "gather"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Unknown),
                effect: Effect::Inferred,
            },
        );
    }

    // Element-wise tensor `sign` (Tensor -> Tensor). Its runtime FFI
    // (`nsl_tensor_sign`), Cranelift signature, and codegen call-dispatch all
    // already existed; only this semantic registration was missing, which made
    // the stdlib Lion optimizer (`sign(β₁m+(1-β₁)g)`) fail type-checking.
    //
    // It is deliberately typed concretely as Tensor->Tensor (not `Unknown` like
    // `abs`): a `sign(...)` result typed `Unknown` would make a downstream
    // `scalar * result` (e.g. Lion's `lr * update`) mis-dispatch to a scalar
    // multiply, producing an f64 where a tensor pointer is required. Unlike
    // `abs`, `sign` has no scalar-argument codegen path — the scalar version is
    // `nsl.math.sign`, which shadows this builtin via codegen's
    // user-defined-function check when imported (same pattern as `clamp`).
    def(
        "sign",
        Type::Function {
            params: vec![tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // tensor_slice(tensor, dim, start, end) -> tensor
    def(
        "tensor_slice",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // tensor_cat(list_of_tensors, dim) -> tensor
    def(
        "tensor_cat",
        Type::Function {
            params: vec![Type::List(Box::new(tensor_ret.clone())), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // contiguous(tensor) -> tensor — materialize non-contiguous views
    def(
        "contiguous",
        Type::Function {
            params: vec![tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // M18a shape manipulation free functions
    // unsqueeze(tensor, dim) -> tensor
    def(
        "unsqueeze",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // stack(list_of_tensors, dim) -> tensor
    def(
        "stack",
        Type::Function {
            params: vec![Type::List(Box::new(tensor_ret.clone())), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // causal_mask(seq_len) -> tensor
    def(
        "causal_mask",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
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
            effect: Effect::Inferred,
        },
    );

    // Tokenizer functions (M15)
    def(
        "byte_tokenizer_new",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "bpe_train",
        Type::Function {
            params: vec![Type::Str, Type::Int, Type::Int, Type::List(Box::new(Type::Str))],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "tokenizer_load",
        Type::Function {
            params: vec![Type::Str],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "tokenizer_save",
        Type::Function {
            params: vec![Type::Int, Type::Str],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "tokenizer_encode",
        Type::Function {
            params: vec![Type::Int, Type::Str],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    def(
        "tokenizer_decode",
        Type::Function {
            params: vec![Type::Int, tensor_ret.clone()],
            ret: Box::new(Type::Str),
            effect: Effect::Inferred,
        },
    );
    def(
        "tokenizer_vocab_size",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "tokenizer_encode_batch",
        Type::Function {
            params: vec![Type::Int, Type::List(Box::new(Type::Str)), Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::List(Box::new(tensor_ret.clone()))),
            effect: Effect::Inferred,
        },
    );

    // Quantization functions (M16)
    // nsl_qtensor_quantize(tensor, dtype, granularity, axis, group_size) -> QuantizedTensor
    def(
        "nsl_qtensor_quantize",
        Type::Function {
            params: vec![Type::Unknown, Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::QuantizedTensor),
            effect: Effect::Inferred,
        },
    );
    // nsl_qtensor_dequantize(qtensor) -> Tensor
    def(
        "nsl_qtensor_dequantize",
        Type::Function {
            params: vec![Type::QuantizedTensor],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_qtensor_matmul_mixed(tensor, qtensor) -> Tensor
    def(
        "nsl_qtensor_matmul_mixed",
        Type::Function {
            params: vec![Type::Unknown, Type::QuantizedTensor],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_qtensor_dtype(qtensor) -> Int
    def(
        "nsl_qtensor_dtype",
        Type::Function {
            params: vec![Type::QuantizedTensor],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    // nsl_qtensor_shape(qtensor) -> Tensor (1D shape as f64 values)
    def(
        "nsl_qtensor_shape",
        Type::Function {
            params: vec![Type::QuantizedTensor],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // Math functions (always return Float — coerce int args at codegen level)
    for name in &["sqrt", "log", "exp", "sin", "cos", "floor"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Float),
                effect: Effect::Inferred,
            },
        );
    }

    // Type conversion functions
    def(
        "int",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "float",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Float),
            effect: Effect::Inferred,
        },
    );
    def(
        "bool",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Bool),
            effect: Effect::Inferred,
        },
    );
    def(
        "str",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Str),
            effect: Effect::Inferred,
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
                effect: Effect::Inferred,
            },
        );
    }
    def(
        "sync_threads",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
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
            effect: Effect::Inferred,
        },
    );
    def(
        "filter",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
            effect: Effect::Inferred,
        },
    );
    def(
        "enumerate",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
            effect: Effect::Inferred,
        },
    );
    def(
        "zip",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
            effect: Effect::Inferred,
        },
    );
    def(
        "sorted",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
            effect: Effect::Inferred,
        },
    );
    def(
        "reversed",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Unknown))),
            effect: Effect::Inferred,
        },
    );

    // Interop intrinsics (M18b)
    // load_safetensors(path, device=0) -> Dict[str, Tensor]
    def(
        "load_safetensors",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Dict(Box::new(Type::Str), Box::new(tensor_ret.clone()))),
            effect: Effect::Inferred,
        },
    );
    // save_safetensors(dict, path) -> void
    def(
        "save_safetensors",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    // from_hf(repo_id, model, device=0) -> model
    def(
        "from_hf",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Unknown),
            effect: Effect::Inferred,
        },
    );
    // to_onnx(model, input, path) -> void
    def(
        "to_onnx",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );

    // Data pipeline intrinsics (M19)
    def(
        "load_jsonl",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Str))),
            effect: Effect::Inferred,
        },
    );
    def(
        "load_csv",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(Type::Str))),
            effect: Effect::Inferred,
        },
    );
    def(
        "load_mmap",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // DataLoader returns an opaque handle; iterating it yields Dict<Str, Tensor> batches.
    // We type the constructor return as List<Dict<Str, Tensor>> so that for-loop iteration
    // naturally produces Dict<Str, Tensor> as the element type.
    let batch_dict = Type::Dict(Box::new(Type::Str), Box::new(tensor_ret.clone()));
    def(
        "DataLoader",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::List(Box::new(batch_dict))),
            effect: Effect::Inferred,
        },
    );

    // Sampling intrinsics (M19)
    def(
        "topk",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(Type::Dict(Box::new(Type::Str), Box::new(tensor_ret.clone()))),
            effect: Effect::Inferred,
        },
    );
    def(
        "multinomial",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    def(
        "argmax",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    def(
        "manual_seed",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    // CFIE Cycle 11: generate(target_model, prompt, params=...) drives
    // the compiled decode loop inside a CFIE serve endpoint.  Variadic
    // (the trailing `params` / kwargs are accepted for source
    // compatibility); returns the generated token COUNT (Int) in v1.
    def(
        "generate",
        Type::Function {
            params: vec![Type::Unknown],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "cumsum",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    def(
        "lt_scalar",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // Paged KV-cache functions (M25)
    def(
        "kv_cache_init",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_init_gpu",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_alloc_seq",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_append",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_k_ptr",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_v_ptr",
        Type::Function {
            params: vec![Type::Int, Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_free_seq",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_seq_len",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_seq_blocks",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_seq_num_blocks",
        Type::Function {
            params: vec![Type::Int, Type::Int],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_utilization",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Float),
            effect: Effect::Inferred,
        },
    );
    def(
        "kv_cache_destroy",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );

    // Memory profiler functions (M25)
    def(
        "profiler_start",
        Type::Function {
            params: vec![Type::Int],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "profiler_stop",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "profiler_peak",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Int),
            effect: Effect::Inferred,
        },
    );
    // Kernel profiler builtins (M26)
    def(
        "kernel_profiler_start",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "kernel_profiler_stop",
        Type::Function {
            params: vec![],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );

    // M32: MoE dispatch intrinsic
    def(
        "moe_dispatch",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // CPDT Part III v2.3: paper-faithful MoE FFN intrinsic
    // (`moe_dispatch_ffn(tokens, logits, experts_up, experts_down)`).
    // Opt-in 4-arg variant that lowers to `nsl_moe_dispatch_full_v3`
    // (up → SiLU → down). Coexists with the 3-arg `moe_dispatch`
    // (v1/v2 path); users opt into v3 by switching intrinsics.
    def(
        "moe_dispatch_ffn",
        Type::Function {
            params: vec![Type::Unknown, Type::Unknown, Type::Unknown, Type::Unknown],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );

    // CPDT Part III v2.5+v2.8: Mixtral's gated MoE FFN intrinsic
    // (`moe_dispatch_swiglu(tokens, logits, experts_gate, experts_up,
    // experts_down)`). 5-arg opt-in that lowers to
    // `nsl_moe_dispatch_full_v4`. Gate activation selected by
    // `@moe(activation="silu"|"gelu"|"relu")` — default silu→SwiGLU,
    // gelu→GeGLU (v2.8), relu→ReGLU (v2.8). `@moe(activation="identity")`
    // is refused at codegen (use `moe_dispatch_ffn` v3 for a 2-weight
    // FFN). Coexists with v2 / v3 source intrinsics.
    def(
        "moe_dispatch_swiglu",
        Type::Function {
            params: vec![
                Type::Unknown,
                Type::Unknown,
                Type::Unknown,
                Type::Unknown,
                Type::Unknown,
            ],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
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
            effect: Effect::Inferred,
        },
    );

    // Checkpoint I/O (M14): model_save(model, path), model_load(model, path)
    def(
        "model_save",
        Type::Function {
            params: vec![Type::Unknown, Type::Str],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );
    def(
        "model_load",
        Type::Function {
            params: vec![Type::Unknown, Type::Str],
            ret: Box::new(Type::Void),
            effect: Effect::Inferred,
        },
    );

    // Source-AD backward lowering helpers (Task 2 of source-AD training backend)
    // nsl_tensor_compare(a, b, cmp_kind) -> tensor  — elementwise 0/1 comparison
    def(
        "nsl_tensor_compare",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_where(cond, true_val, false_val) -> tensor  — ternary select
    def(
        "nsl_tensor_where",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), tensor_ret.clone()],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_scalar(val) -> tensor  — create 0-dim scalar tensor
    def(
        "nsl_tensor_scalar",
        Type::Function {
            params: vec![Type::Float],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_pad_zero(tensor, dim, pad_before, pad_after) -> tensor
    def(
        "nsl_tensor_pad_zero",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int, Type::Int, Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_scatter_add(src, indices, dim) -> tensor  — embedding backward
    def(
        "nsl_tensor_scatter_add",
        Type::Function {
            params: vec![tensor_ret.clone(), tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_logsoftmax(tensor, dim) -> tensor
    def(
        "nsl_tensor_logsoftmax",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_repeat(tensor, kernel) -> tensor  — spatial repeat for avgpool backward
    def(
        "nsl_tensor_repeat",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
    // nsl_tensor_rope_inverse(tensor, dim) -> tensor  — inverse RoPE rotation
    def(
        "nsl_tensor_rope_inverse",
        Type::Function {
            params: vec![tensor_ret.clone(), Type::Int],
            ret: Box::new(tensor_ret.clone()),
            effect: Effect::Inferred,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The element-wise tensor `sign` builtin must be registered as
    /// `Tensor -> Tensor`. If it is missing, the stdlib Lion optimizer fails to
    /// type-check (`undefined variable sign`). If it is registered as `Unknown`
    /// instead of `Tensor`, a downstream `scalar * sign(...)` (Lion's
    /// `lr * update`) mis-dispatches to a scalar multiply and fails Cranelift
    /// verification with "arg has type f64, expected i64". This test guards both.
    #[test]
    fn sign_is_registered_as_tensor_to_tensor() {
        let mut scopes = ScopeMap::new();
        let mut interner = Interner::new();
        register_builtins(&mut scopes, &mut interner);

        let sym = Symbol(interner.get_or_intern("sign"));
        let (_, info) = scopes
            .lookup(ScopeId::ROOT, sym)
            .expect("`sign` must be a registered builtin");

        match &info.ty {
            Type::Function { params, ret, .. } => {
                assert_eq!(params.len(), 1, "sign takes exactly one argument");
                assert!(
                    matches!(params[0], Type::Tensor { .. }),
                    "sign's parameter must be Tensor (got {:?}); an Unknown param would let \
                     the result stay untyped and mis-dispatch downstream arithmetic",
                    params[0]
                );
                assert!(
                    matches!(**ret, Type::Tensor { .. }),
                    "sign must return Tensor (got {:?}), not Unknown",
                    ret
                );
            }
            other => panic!("`sign` must be a Function type, got {:?}", other),
        }
    }
}
