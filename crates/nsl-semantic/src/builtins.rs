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

    // Activation functions (take tensor, return tensor)
    for name in &["relu", "gelu", "silu", "sigmoid", "tanh", "softmax"] {
        def(
            name,
            Type::Function {
                params: vec![Type::Unknown],
                ret: Box::new(Type::Unknown),
            },
        );
    }

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

    // Utility functions
    def(
        "assert",
        Type::Function {
            params: vec![Type::Unknown],
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

    // Math functions (always return Float — coerce int args at codegen level)
    for name in &["sqrt", "log", "exp", "sin", "cos"] {
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
}
