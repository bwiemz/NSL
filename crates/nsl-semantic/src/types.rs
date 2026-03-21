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

    /// Fixed-size array of models: [TransformerBlock; 12]
    FixedModelArray {
        element_model: Symbol,
        size: i64,
    },

    /// Immutable borrow: &T — read-only access, cannot be consumed or mutated.
    /// The inner type is the borrowed type (e.g., &Tensor<[N], f32> -> Borrow(Tensor{...})).
    Borrow(Box<Type>),
}

impl Type {
    /// Returns true if this type is a tensor family type.
    pub fn is_tensor(&self) -> bool {
        match self {
            Type::Tensor { .. } | Type::Param { .. } | Type::Buffer { .. } => true,
            Type::Borrow(inner) => inner.is_tensor(),
            _ => false,
        }
    }

    /// Returns true if this is an error or unknown (suppresses cascading).
    pub fn is_indeterminate(&self) -> bool {
        match self {
            Type::Error | Type::Unknown => true,
            Type::Borrow(inner) => inner.is_indeterminate(),
            _ => false,
        }
    }

    /// Normalize Param/Buffer to Tensor for type-checking operations.
    /// Returns (shape, dtype, device) if this is a tensor-family type.
    /// Sees through borrows: `&Tensor<...>` returns the inner tensor parts.
    pub fn as_tensor_parts(&self) -> Option<(&Shape, &DType, Device)> {
        match self {
            Type::Tensor { shape, dtype, device } => Some((shape, dtype, device.clone())),
            Type::Param { shape, dtype } => Some((shape, dtype, Device::Unknown)),
            Type::Buffer { shape, dtype } => Some((shape, dtype, Device::Unknown)),
            Type::Borrow(inner) => inner.as_tensor_parts(),
            _ => None,
        }
    }

    /// Returns the inner type if this is a borrow, otherwise returns self.
    pub fn strip_borrow(&self) -> &Type {
        match self {
            Type::Borrow(inner) => inner.strip_borrow(),
            other => other,
        }
    }

    /// Returns true if this is a borrow type.
    pub fn is_borrow(&self) -> bool {
        matches!(self, Type::Borrow(_))
    }
}

/// Arithmetic expression over symbolic dimensions.
/// Tracks how dimensions compose through reshape/concat/split.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Sym(Symbol),
    Lit(i64),
    Add(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Mod(Box<DimExpr>, Box<DimExpr>),
}

impl DimExpr {
    pub fn as_lit(&self) -> Option<i64> {
        if let DimExpr::Lit(v) = self { Some(*v) } else { None }
    }
    pub fn as_sym(&self) -> Option<Symbol> {
        if let DimExpr::Sym(s) = self { Some(*s) } else { None }
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
    /// Bounded symbolic: resolved at runtime, with compile-time upper bound.
    Bounded { name: Symbol, upper_bound: i64 },
    /// Computed: arithmetic over other dims (e.g. from reshape).
    Computed(Box<DimExpr>),
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

    /// Returns true if any dimension is symbolic, bounded, or computed.
    pub fn has_symbolic(&self) -> bool {
        self.dims.iter().any(|d| matches!(d, Dim::Symbolic(_) | Dim::Bounded { .. } | Dim::Computed(_)))
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
    /// Custom user-defined datatype (M23 BYOD)
    Custom(u16),  // runtime dtype ID (256+)
    Unknown,
}

impl DType {
    /// Returns the size in bytes of one element of this dtype.
    /// Custom and Unknown return 0 (size not known at compile time).
    pub fn byte_width(&self) -> usize {
        match self {
            DType::F64 | DType::Int64 => 8,
            DType::F32 | DType::Int32 => 4,
            DType::Fp16 | DType::Bf16 | DType::Int16 => 2,
            DType::Fp8E4m3 | DType::Fp8E5m2 | DType::Int8 | DType::Uint8 | DType::Bool => 1,
            DType::Int4 => 1, // sub-byte; round up
            DType::Custom(_) | DType::Unknown => 0,
        }
    }

    /// Returns true if this is an FP8 dtype (E4M3 or E5M2).
    pub fn is_fp8(&self) -> bool {
        matches!(self, DType::Fp8E4m3 | DType::Fp8E5m2)
    }
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
    // Borrow assignability rules:
    // - &T -> &T: if inner types are assignable (handled by same-type check above for exact match)
    // - T -> &T: auto-borrow — owned can be passed where borrow is expected
    // - &T -> T: NOT allowed — cannot pass a borrow where owned/consumed is expected
    if let Type::Borrow(target_inner) = target {
        // Target expects a borrow: source can be either &T or owned T
        let source_inner = source.strip_borrow();
        return is_assignable(source_inner, target_inner);
    }
    if let Type::Borrow(source_inner) = source {
        // Source is a borrow but target is owned: read-compatible for non-consuming operations.
        // Allow &T -> T for read-only contexts (method calls, reads), but the ownership
        // checker will catch actual consumption attempts.
        return is_assignable(source_inner, target);
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
    // Numeric widening: only within same family (int->int or float->float)
    let (src_family, src_rank) = dtype_rank(source);
    let (tgt_family, tgt_rank) = dtype_rank(target);
    if src_family > 0 && tgt_family > 0 && src_family == tgt_family && src_rank <= tgt_rank {
        return true;
    }
    false
}

/// Returns (family, rank) for numeric types. Family: 0=non-numeric, 1=int, 2=float.
/// Widening is only allowed within the same family to prevent e.g. Int64 -> Fp8.
/// Sees through borrow types: `&int` has the same rank as `int`.
pub fn dtype_rank(ty: &Type) -> (u8, u8) {
    match ty {
        Type::Borrow(inner) => dtype_rank(inner),
        Type::Int4 => (1, 1),
        Type::Uint8 => (1, 2),
        Type::Int8 => (1, 2),
        Type::Int16 => (1, 3),
        Type::Int32 => (1, 4),
        Type::Int64 | Type::Int => (1, 5),
        Type::Fp8E4m3 | Type::Fp8E5m2 => (2, 1),
        Type::Fp16 => (2, 2),
        Type::Bf16 => (2, 3),
        Type::F32 => (2, 4),
        Type::F64 | Type::Float => (2, 5),
        _ => (0, 0),
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
        DType::Custom(_) => 0,  // custom dtypes don't participate in widening
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
            let inner: Vec<String> = elems.iter().map(display_type).collect();
            format!("({})", inner.join(", "))
        }
        Type::Optional(inner) => format!("{}?", display_type(inner)),
        Type::Function { params, ret } => {
            let ps: Vec<String> = params.iter().map(display_type).collect();
            format!("({}) -> {}", ps.join(", "), display_type(ret))
        }
        Type::QuantizedTensor => "QuantizedTensor".into(),
        Type::Module { .. } => "module".into(),
        Type::Unknown => "unknown".into(),
        Type::Error => "error".into(),
        Type::NoneType => "None".into(),
        Type::FixedModelArray { size, .. } => format!("[model; {}]", size),
        Type::Borrow(inner) => format!("&{}", display_type(inner)),
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
        DType::Custom(id) => format!("custom:{}", id),
        DType::Unknown => "?".into(),
    }
}

pub fn display_device(device: &Device) -> String {
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

    // ── Borrow type tests ──────────────────────────────────────────

    #[test]
    fn borrow_same_type_assignable() {
        let borrow_int = Type::Borrow(Box::new(Type::Int));
        assert!(is_assignable(&borrow_int, &borrow_int));
    }

    #[test]
    fn owned_to_borrow_assignable() {
        // T -> &T: auto-borrow
        let owned = Type::Int;
        let borrow_int = Type::Borrow(Box::new(Type::Int));
        assert!(is_assignable(&owned, &borrow_int));
    }

    #[test]
    fn borrow_to_owned_assignable_for_reads() {
        // &T -> T: allowed for read-compatible contexts
        let owned = Type::Int;
        let borrow_int = Type::Borrow(Box::new(Type::Int));
        assert!(is_assignable(&borrow_int, &owned));
    }

    #[test]
    fn borrow_tensor_auto_borrow() {
        let tensor = Type::Tensor {
            shape: Shape { dims: vec![Dim::Concrete(4)] },
            dtype: DType::F32,
            device: Device::Unknown,
        };
        let borrow_tensor = Type::Borrow(Box::new(tensor.clone()));
        // Owned tensor assignable to &Tensor
        assert!(is_assignable(&tensor, &borrow_tensor));
        // &Tensor assignable to &Tensor
        assert!(is_assignable(&borrow_tensor, &borrow_tensor));
    }

    #[test]
    fn borrow_display() {
        let borrow_int = Type::Borrow(Box::new(Type::Int));
        assert_eq!(display_type(&borrow_int), "&int");

        let borrow_tensor = Type::Borrow(Box::new(Type::Tensor {
            shape: Shape { dims: vec![Dim::Concrete(4)] },
            dtype: DType::F32,
            device: Device::Unknown,
        }));
        assert_eq!(display_type(&borrow_tensor), "&Tensor<[4], f32>");
    }

    #[test]
    fn borrow_is_tensor() {
        let tensor = Type::Tensor {
            shape: Shape { dims: vec![Dim::Concrete(4)] },
            dtype: DType::F32,
            device: Device::Unknown,
        };
        let borrow = Type::Borrow(Box::new(tensor));
        assert!(borrow.is_tensor());
        assert!(borrow.is_borrow());
    }

    #[test]
    fn borrow_strip_borrow() {
        let inner = Type::Int;
        let borrow = Type::Borrow(Box::new(inner.clone()));
        assert_eq!(borrow.strip_borrow(), &inner);
        // Non-borrow returns self
        assert_eq!(inner.strip_borrow(), &inner);
    }

    #[test]
    fn borrow_mismatched_types_not_assignable() {
        let borrow_int = Type::Borrow(Box::new(Type::Int));
        let borrow_str = Type::Borrow(Box::new(Type::Str));
        assert!(!is_assignable(&borrow_int, &borrow_str));
    }
}

#[cfg(test)]
mod dtype_tests {
    use super::DType;

    #[test]
    fn test_byte_width_standard_types() {
        assert_eq!(DType::F64.byte_width(), 8);
        assert_eq!(DType::F32.byte_width(), 4);
        assert_eq!(DType::Fp16.byte_width(), 2);
        assert_eq!(DType::Bf16.byte_width(), 2);
        assert_eq!(DType::Int64.byte_width(), 8);
        assert_eq!(DType::Int32.byte_width(), 4);
        assert_eq!(DType::Int16.byte_width(), 2);
        assert_eq!(DType::Int8.byte_width(), 1);
        assert_eq!(DType::Uint8.byte_width(), 1);
        assert_eq!(DType::Bool.byte_width(), 1);
    }

    #[test]
    fn test_byte_width_small_types() {
        assert_eq!(DType::Fp8E4m3.byte_width(), 1);
        assert_eq!(DType::Fp8E5m2.byte_width(), 1);
        assert_eq!(DType::Int4.byte_width(), 1); // rounds up to 1 byte
    }

    #[test]
    fn test_byte_width_special() {
        assert_eq!(DType::Custom(256).byte_width(), 0);
        assert_eq!(DType::Unknown.byte_width(), 0);
    }
}
