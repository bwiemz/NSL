//! Type-annotation table tests (roadmap T1): each case is parsed as the
//! annotation of `let x: <type> = 0`, and the test pins the type it
//! becomes in the S-expression form of `common/sexpr.rs`.

#[path = "common/sexpr.rs"]
#[allow(dead_code)]
#[macro_use]
mod sexpr;

use sexpr::ty;

cases! { ty;
    named_int: "int" => "int";
    named_float: "float" => "float";
    wildcard_type: "_" => "_";
    generic_one_argument: "list<int>" => "list<int>";
    generic_two_arguments: "dict<str, float>" => "dict<str float>";
    nested_generic: "list<list<int>>" => "list<list<int>>";
    generic_over_a_type_variable: "Option<T>" => "Option<T>";
    tuple_type: "(int, float)" => "(ttuple int float)";
    union_type: "int | None" => "(union int None)";
    function_type: "(int, int) -> int" => "(fn (int int) int)";
    fixed_array_type: "[Linear; 12]" => "[Linear; 12]";
    borrow_type: "&Tensor" => "&Tensor";
}

cases! { ty;
    tensor_concrete_shape: "Tensor<[2, 3], f32>" => "(Tensor [2 3] f32)";
    tensor_symbolic_and_concrete: "Tensor<[batch, 768], bf16>" => "(Tensor [batch 768] bf16)";
    tensor_on_cuda: "Tensor<[B, T, D], f16, cuda>" => "(Tensor [B T D] f16 cuda)";
    tensor_on_a_numbered_cuda_device: "Tensor<[4], f32, cuda(1)>" => "(Tensor [4] f32 cuda(1))";
    tensor_on_cpu: "Tensor<[4], f32, cpu>" => "(Tensor [4] f32 cpu)";
    tensor_on_metal: "Tensor<[4], f32, metal>" => "(Tensor [4] f32 metal)";
    tensor_on_rocm: "Tensor<[4], f32, rocm>" => "(Tensor [4] f32 rocm)";
    tensor_wildcard_dimension: "Tensor<[_, 3], f32>" => "(Tensor [_ 3] f32)";
    tensor_named_dimensions: "Tensor<[batch=\"B\", heads=12], f32>" => "(Tensor [batch=\"B\" heads=12] f32)";
    tensor_bounded_dimension: "Tensor<[SeqLen < 4096], f32>" => "(Tensor [SeqLen<4096] f32)";
    param_type: "Param<[768, 768], f32>" => "(Param [768 768] f32)";
    buffer_type: "Buffer<[10], i32>" => "(Buffer [10] i32)";
    sparse_type: "Sparse<[100, 100], f32, csr>" => "(Sparse [100 100] f32 csr)";
}
