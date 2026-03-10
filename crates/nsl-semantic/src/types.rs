use std::collections::HashMap;

use nsl_ast::Symbol;

use crate::shapes;

/// Resolved, canonical type representation.
/// Distinct from `nsl_ast::types::TypeExpr` which is syntactic.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitives
    Int,
    Float,
    Bool,
    Str,
    Void,

    // Specific numeric types
    F32,
    F64,
    Fp16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    Int8,
    Int16,
    Int32,
    Int64,
    Int4,
    Uint8,

    // Compound types
    List(Box<Type>),
    Dict(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),
    Optional(Box<Type>),

    // Tensor family
    Tensor {
        shape: Shape,
        dtype: DType,
        device: Device,
    },
    Param {
        shape: Shape,
        dtype: DType,
    },
    Buffer {
        shape: Shape,
        dtype: DType,
    },
    Sparse {
        shape: Shape,
        dtype: DType,
        format: SparseFormat,
    },

    /// Opaque quantized tensor handle (always i64 pointer at IR level).
    QuantizedTensor,

    // Callable
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },

    // Nominal user-defined types
    Struct {
        name: Symbol,
        fields: Vec<(Symbol, Type)>,
    },
    Enum {
        name: Symbol,
        variants: Vec<(Symbol, Vec<Type>)>,
    },
    Model {
        name: Symbol,
        fields: Vec<(Symbol, Type)>,
        methods: Vec<(Symbol, Type)>,
    },

    // Union type
    Union(Vec<Type>),

    // Generic type variable (tracked but not instantiated in M2)
    TypeVar(Symbol),

    /// A module alias (e.g., `import nsl.math as math`).
    /// Contains the exported symbols of the module.
    Module {
        exports: HashMap<Symbol, Box<Type>>,
    },

    /// Type is unknown (inference failed or annotation missing).
    /// Propagates silently without generating diagnostics.
    Unknown,

    /// An error occurred resolving this type. Any operation on Error
    /// produces Error without further diagnostics ("poison value").
    Error,

    /// The type of `none` literal. Unifies with `Optional<T>` for any T.
    NoneType,
}

impl Type {
    /// Returns true if this type is a tensor family type.
    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor { .. } | Type::Param { .. } | Type::Buffer { .. })
    }

    /// Returns true if this is an error or unknown (suppresses cascading).
    pub fn is_indeterminate(&self) -> bool {
        matches!(self, Type::Error | Type::Unknown)
    }
}

/// Resolved dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    /// Known concrete size.
    Concrete(i64),
    /// Symbolic: same name within a scope must unify to same size.
    Symbolic(Symbol),
    /// Named dimension with a label and optional concrete/symbolic size.
    Named { name: Symbol, size: Box<Dim> },
    /// Wildcard: unchecked.
    Wildcard,
}

/// A resolved tensor shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub dims: Vec<Dim>,
}

impl Shape {
    pub fn unknown() -> Self {
        Shape { dims: Vec::new() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn scalar() -> Self {
        Shape { dims: Vec::new() }
    }
}

/// Resolved data type for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    Fp16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    Int64,
    Int32,
    Int16,
    Int8,
    Int4,
    Uint8,
    Bool,
    Unknown,
}

/// Resolved device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(Option<u32>),
    Metal,
    Rocm(Option<u32>),
    Npu(Symbol),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    Coo,
    Csr,
    Csc,
    Bsr,
    Unknown,
}

/// Check if `source` type can be assigned to a location of `target` type.
pub fn is_assignable(source: &Type, target: &Type) -> bool {
    // Error/Unknown are compatible with everything (suppress cascading)
    if source.is_indeterminate() || target.is_indeterminate() {
        return true;
    }
    // Same type
    if source == target {
        return true;
    }
    // Safe widening: int -> float
    if matches!(source, Type::Int) && matches!(target, Type::Float) {
        return true;
    }
    // Generic float is compatible with specific float types (narrowing)
    if matches!(source, Type::Float)
        && matches!(
            target,
            Type::F32 | Type::F64 | Type::Fp16 | Type::Bf16 | Type::Fp8E4m3 | Type::Fp8E5m2
        )
    {
        return true;
    }
    // Generic int is compatible with specific int types (narrowing)
    if matches!(source, Type::Int)
        && matches!(
            target,
            Type::Int4 | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Uint8
        )
    {
        return true;
    }
    // List covariance: List<A> assignable to List<B> if A assignable to B
    if let (Type::List(src_elem), Type::List(tgt_elem)) = (source, target) {
        return is_assignable(src_elem, tgt_elem);
    }
    // NoneType -> Optional<T>
    if matches!(source, Type::NoneType) && matches!(target, Type::Optional(_)) {
        return true;
    }
    // Param assignable to Tensor (subtyping)
    if let (Type::Param { shape: s1, dtype: d1 }, Type::Tensor { shape: s2, dtype: d2, .. }) =
        (source, target)
    {
        return s1 == s2 && d1 == d2;
    }
    // Tensor-to-Tensor assignability with shape checking
    if let (
        Type::Tensor { shape: vs, dtype: vd, device: vdev },
        Type::Tensor { shape: as_, dtype: ad, device: adev },
    ) = (source, target)
    {
        // Unknown shape is always compatible
        if vs.rank() == 0 || as_.rank() == 0 {
            return true;
        }
        if vs.rank() != as_.rank() {
            return false;
        }
        for (v, a) in vs.dims.iter().zip(as_.dims.iter()) {
            if shapes::unify_dim(v, a).is_none() {
                return false;
            }
        }
        // Device: unknown matches anything
        if !matches!(vdev, Device::Unknown) && !matches!(adev, Device::Unknown) && vdev != adev {
            return false;
        }
        // Dtype: must match (dtype widening for tensors is a later feature)
        return vd == ad;
    }
    // Numeric widening: int4 -> int8 -> int16 -> int32 -> int64
    if dtype_rank(source) > 0 && dtype_rank(target) > 0 && dtype_rank(source) <= dtype_rank(target)
    {
        return true;
    }
    false
}

/// Returns a "rank" for numeric types used in widening checks. 0 = not numeric.
fn dtype_rank(ty: &Type) -> u8 {
    match ty {
        Type::Int4 => 1,
        Type::Int8 => 2,
        Type::Int16 => 3,
        Type::Int32 => 4,
        Type::Int64 | Type::Int => 5,
        Type::Fp8E4m3 | Type::Fp8E5m2 => 6,
        Type::Fp16 => 7,
        Type::Bf16 => 8,
        Type::F32 => 9,
        Type::F64 | Type::Float => 10,
        _ => 0,
    }
}

/// Return the wider of two DTypes for binary operation results.
pub fn wider_dtype(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    if matches!(a, DType::Unknown) {
        return b;
    }
    if matches!(b, DType::Unknown) {
        return a;
    }
    // Use precedence: higher rank wins
    let rank_a = dtype_to_rank(a);
    let rank_b = dtype_to_rank(b);
    if rank_a >= rank_b {
        a
    } else {
        b
    }
}

fn dtype_to_rank(d: DType) -> u8 {
    match d {
        DType::Bool => 0,
        DType::Int4 => 1,
        DType::Int8 | DType::Uint8 => 2,
        DType::Int16 => 3,
        DType::Int32 => 4,
        DType::Int64 => 5,
        DType::Fp8E4m3 | DType::Fp8E5m2 => 6,
        DType::Fp16 => 7,
        DType::Bf16 => 8,
        DType::F32 => 9,
        DType::F64 => 10,
        DType::Unknown => 0,
    }
}

/// Format a type for human-readable display.
pub fn display_type(ty: &Type) -> String {
    match ty {
        Type::Int => "int".into(),
        Type::Float => "float".into(),
        Type::Bool => "bool".into(),
        Type::Str => "str".into(),
        Type::Void => "void".into(),
        Type::F32 => "f32".into(),
        Type::F64 => "f64".into(),
        Type::Fp16 => "fp16".into(),
        Type::Bf16 => "bf16".into(),
        Type::Fp8E4m3 => "fp8e4m3".into(),
        Type::Fp8E5m2 => "fp8e5m2".into(),
        Type::Int8 => "int8".into(),
        Type::Int16 => "int16".into(),
        Type::Int32 => "int32".into(),
        Type::Int64 => "int64".into(),
        Type::Int4 => "int4".into(),
        Type::Uint8 => "uint8".into(),
        Type::Tensor { shape, dtype, device } => {
            let shape_str = crate::shapes::fmt_shape(shape);
            let dtype_str = display_dtype(dtype);
            let dev_str = display_device(device);
            if matches!(device, Device::Unknown) {
                format!("Tensor<{}, {}>", shape_str, dtype_str)
            } else {
                format!("Tensor<{}, {}, {}>", shape_str, dtype_str, dev_str)
            }
        }
        Type::Param { shape, dtype } => {
            let shape_str = crate::shapes::fmt_shape(shape);
            let dtype_str = display_dtype(dtype);
            format!("Param<{}, {}>", shape_str, dtype_str)
        }
        Type::Buffer { shape, dtype } => {
            let shape_str = crate::shapes::fmt_shape(shape);
            let dtype_str = display_dtype(dtype);
            format!("Buffer<{}, {}>", shape_str, dtype_str)
        }
        Type::Sparse { shape, dtype, format } => {
            let shape_str = crate::shapes::fmt_shape(shape);
            let dtype_str = display_dtype(dtype);
            let fmt_str = match format {
                SparseFormat::Coo => "coo",
                SparseFormat::Csr => "csr",
                SparseFormat::Csc => "csc",
                SparseFormat::Bsr => "bsr",
                SparseFormat::Unknown => "?",
            };
            format!("Sparse<{}, {}, {}>", shape_str, dtype_str, fmt_str)
        }
        Type::List(inner) => format!("list<{}>", display_type(inner)),
        Type::Dict(k, v) => format!("dict<{}, {}>", display_type(k), display_type(v)),
        Type::Tuple(elems) => {
            let inner: Vec<String> = elems.iter().map(|t| display_type(t)).collect();
            format!("({})", inner.join(", "))
        }
        Type::Optional(inner) => format!("{}?", display_type(inner)),
        Type::Function { params, ret } => {
            let ps: Vec<String> = params.iter().map(|t| display_type(t)).collect();
            format!("({}) -> {}", ps.join(", "), display_type(ret))
        }
        Type::QuantizedTensor => "QuantizedTensor".into(),
        Type::Module { .. } => "module".into(),
        Type::Unknown => "unknown".into(),
        Type::Error => "error".into(),
        Type::NoneType => "None".into(),
        _ => format!("{:?}", ty),
    }
}

fn display_dtype(dtype: &DType) -> String {
    match dtype {
        DType::F64 => "f64".into(),
        DType::F32 => "f32".into(),
        DType::Fp16 => "fp16".into(),
        DType::Bf16 => "bf16".into(),
        DType::Fp8E4m3 => "fp8e4m3".into(),
        DType::Fp8E5m2 => "fp8e5m2".into(),
        DType::Int64 => "int64".into(),
        DType::Int32 => "int32".into(),
        DType::Int16 => "int16".into(),
        DType::Int8 => "int8".into(),
        DType::Int4 => "int4".into(),
        DType::Uint8 => "uint8".into(),
        DType::Bool => "bool".into(),
        DType::Unknown => "?".into(),
    }
}

fn display_device(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".into(),
        Device::Cuda(None) => "cuda".into(),
        Device::Cuda(Some(id)) => format!("cuda:{}", id),
        Device::Metal => "metal".into(),
        Device::Rocm(None) => "rocm".into(),
        Device::Rocm(Some(id)) => format!("rocm:{}", id),
        Device::Npu(_) => "npu".into(),
        Device::Unknown => "?".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_equality() {
        assert_eq!(Type::Int, Type::Int);
        assert_ne!(Type::Int, Type::Float);
    }

    #[test]
    fn assignability_same_type() {
        assert!(is_assignable(&Type::Int, &Type::Int));
        assert!(is_assignable(&Type::Str, &Type::Str));
    }

    #[test]
    fn assignability_widening() {
        assert!(is_assignable(&Type::Int, &Type::Float));
        assert!(!is_assignable(&Type::Float, &Type::Int));
    }

    #[test]
    fn assignability_error_unknown() {
        assert!(is_assignable(&Type::Error, &Type::Int));
        assert!(is_assignable(&Type::Int, &Type::Unknown));
    }

    #[test]
    fn assignability_none_to_optional() {
        assert!(is_assignable(
            &Type::NoneType,
            &Type::Optional(Box::new(Type::Int))
        ));
    }

    #[test]
    fn shape_rank() {
        let s = Shape {
            dims: vec![Dim::Concrete(3), Dim::Concrete(4)],
        };
        assert_eq!(s.rank(), 2);
        assert_eq!(Shape::unknown().rank(), 0);
    }

    #[test]
    fn wider_dtype_same() {
        assert_eq!(wider_dtype(DType::F32, DType::F32), DType::F32);
    }

    #[test]
    fn wider_dtype_mixed() {
        assert_eq!(wider_dtype(DType::Fp16, DType::F32), DType::F32);
        assert_eq!(wider_dtype(DType::F32, DType::Fp16), DType::F32);
    }
}
