// crates/nsl-codegen/src/kernel_lower.rs
//! M47: AST KernelDef -> KIR lowering.
//!
//! Translates a parsed `KernelDef` AST node into the backend-agnostic `KernelIR`.
//! Handles the portable subset: arithmetic, thread indexing, memory access, barriers.
//! Complex kernel bodies that use NSL features beyond the portable subset should
//! fall back to the existing direct AST->PTX path in `kernel.rs`.
//!
//! ## Type Inference
//! Parameters carry their AST type annotations through to KIR via a type map.
//! Expressions propagate types from the parameter map through variables, binary ops,
//! and subscript dereferences. Mixed-type binary operations promote to the wider type
//! using `promote_types()`.

use std::collections::HashMap;

use nsl_ast::block::KernelDef;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::operator::BinOp;
use nsl_ast::stmt::{StmtKind, Block};
use nsl_ast::types::TypeExprKind;
use nsl_lexer::Interner;

use crate::gpu_target::GpuTarget;
use crate::kernel_ir::*;

// ---------------------------------------------------------------------------
// Type promotion
// ---------------------------------------------------------------------------

/// Type promotion table for binary operations in KIR.
///
/// Rules:
/// 1. Same type -> same type
/// 2. Float + Float -> wider float
/// 3. Int + Float -> float (or wider float if int is 64-bit to avoid precision loss)
/// 4. Int + Int -> wider int
/// 5. Unsigned + Signed of same width -> signed (to preserve sign information)
pub fn promote_types(a: KirType, b: KirType) -> KirType {
    if a == b {
        return a;
    }

    use KirType::*;
    match (a.clone(), b.clone()) {
        // Float promotions (wider wins)
        (F16, F32) | (F32, F16) => F32,
        (F16, F64) | (F64, F16) => F64,
        (F32, F64) | (F64, F32) => F64,
        (Bf16, F32) | (F32, Bf16) => F32,
        (Bf16, F64) | (F64, Bf16) => F64,
        (F16, Bf16) | (Bf16, F16) => F32, // both are 16-bit, promote to f32

        // Int promotions (wider wins)
        (I32, I64) | (I64, I32) => I64,
        (U32, U64) | (U64, U32) => U64,
        (U32, I32) | (I32, U32) => I32,
        (U64, I64) | (I64, U64) => I64,
        (U32, I64) | (I64, U32) => I64,
        (U64, I32) | (I32, U64) => I64, // i32 widens to i64 to hold u64 range

        // Int + Float promotions
        (I32, F32) | (F32, I32) => F32,
        (I32, F64) | (F64, I32) => F64,
        (I64, F32) | (F32, I64) => F64, // i64 + f32 -> f64 (avoid precision loss)
        (I64, F64) | (F64, I64) => F64,
        (U32, F32) | (F32, U32) => F32,
        (U32, F64) | (F64, U32) => F64,
        (U64, F32) | (F32, U64) => F64,
        (U64, F64) | (F64, U64) => F64,
        (I32, F16) | (F16, I32) => F32,
        (I64, F16) | (F16, I64) => F64,
        (U32, F16) | (F16, U32) => F32,
        (U64, F16) | (F16, U64) => F64,
        (I32, Bf16) | (Bf16, I32) => F32,
        (I64, Bf16) | (Bf16, I64) => F64,
        (U32, Bf16) | (Bf16, U32) => F32,
        (U64, Bf16) | (Bf16, U64) => F64,

        // Bool promotes to the other type
        (Bool, other) | (other, Bool) => other,

        // Pointer types and Vec types -- no arithmetic promotion
        _ => panic!("Cannot promote types {:?} and {:?}", a, b),
    }
}

// ---------------------------------------------------------------------------
// AST dtype -> KIR type mapping
// ---------------------------------------------------------------------------

/// Convert an AST dtype string (resolved from Symbol) to a KIR scalar type.
fn dtype_str_to_kir(dtype: &str) -> KirType {
    match dtype {
        "f16" | "fp16" | "float16" => KirType::F16,
        "bf16" | "bfloat16" => KirType::Bf16,
        "f32" | "fp32" | "float32" | "float" => KirType::F32,
        "f64" | "fp64" | "float64" | "double" => KirType::F64,
        "i32" | "int32" | "int" => KirType::I32,
        "i64" | "int64" | "long" => KirType::I64,
        "u32" | "uint32" => KirType::U32,
        "u64" | "uint64" => KirType::U64,
        "bool" => KirType::Bool,
        "fp8" | "f8" => KirType::F16, // FP8 promoted to F16 for compute
        _ => KirType::F32,            // fallback for unknown types
    }
}

/// Build a type map from kernel parameter names to their KIR types.
///
/// Tensor/Param/Buffer parameters become `Ptr(element_type, Global)`.
/// Named scalar types (int, float, etc.) become their scalar KIR type.
/// Parameters without type annotations default to `Ptr(F32, Global)`.
pub fn build_param_type_map(
    kernel: &KernelDef,
    interner: &Interner,
) -> HashMap<String, KirType> {
    let mut map = HashMap::new();

    for param in &kernel.params {
        let pname = interner.resolve(param.name.0).unwrap_or("_p").to_string();
        let kir_type = if let Some(type_ann) = &param.type_ann {
            match &type_ann.kind {
                TypeExprKind::Tensor { dtype, .. }
                | TypeExprKind::Param { dtype, .. }
                | TypeExprKind::Buffer { dtype, .. } => {
                    let dtype_str = interner.resolve(dtype.0).unwrap_or("f32");
                    let elem = dtype_str_to_kir(dtype_str);
                    KirType::Ptr(Box::new(elem), AddressSpace::Global)
                }
                TypeExprKind::Named(sym) => {
                    let name = interner.resolve(sym.0).unwrap_or("f32");
                    dtype_str_to_kir(name)
                }
                _ => KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            }
        } else {
            // No type annotation -- default to Ptr(F32, Global) for backward compat
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
        };
        map.insert(pname, kir_type);
    }

    map
}

// ---------------------------------------------------------------------------
// Kernel lowering
// ---------------------------------------------------------------------------

/// Lower a KernelDef AST node into a KernelIR.
///
/// Parameters:
/// - `kernel`: the parsed kernel AST
/// - `interner`: string interner for resolving symbol names
/// - `_target`: the GPU target (used for feature validation in future)
///
/// Returns a `KernelIR` ready for backend lowering (e.g., `backend_ptx::lower_kir_to_ptx`).
pub fn lower_kernel_to_ir(
    kernel: &KernelDef,
    interner: &Interner,
    _target: GpuTarget,
) -> KernelIR {
    let name = interner.resolve(kernel.name.0).unwrap_or("__kernel").to_string();

    // Build type map from AST type annotations
    let param_type_map = build_param_type_map(kernel, interner);

    let mut lowerer = KernelLowerer::new(&name);

    // Map kernel parameters to KirParams using their AST types.
    for param in &kernel.params {
        let pname = interner.resolve(param.name.0).unwrap_or("_p").to_string();
        let kir_type = param_type_map
            .get(&pname)
            .cloned()
            .unwrap_or(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));

        let address_space = match &kir_type {
            KirType::Ptr(_, space) => *space,
            _ => AddressSpace::Local,
        };

        let var_id = lowerer.builder.add_param(&pname, kir_type.clone(), address_space);
        lowerer.var_map.insert(pname.clone(), var_id);
        lowerer.type_map.insert(pname, kir_type);
    }

    // Create entry block
    let entry = lowerer.builder.new_block();
    lowerer.builder.set_block(entry);

    // Lower body statements
    lower_block(&mut lowerer, &kernel.body, interner);

    // Terminate the current block if not already terminated
    lowerer.builder.terminate(KirTerminator::Return);

    lowerer.builder.set_workgroup_size([256, 1, 1]);
    lowerer.builder.finalize()
}

/// Internal lowering state.
struct KernelLowerer {
    builder: KirBuilder,
    /// Map from variable name -> VarId
    var_map: HashMap<String, VarId>,
    /// Map from variable name -> KirType (for type propagation)
    type_map: HashMap<String, KirType>,
}

impl KernelLowerer {
    fn new(name: &str) -> Self {
        KernelLowerer {
            builder: KirBuilder::new(name),
            var_map: HashMap::new(),
            type_map: HashMap::new(),
        }
    }

    /// Look up the KIR type of a variable by name.
    /// Returns F32 as fallback for unknown variables.
    fn var_type(&self, name: &str) -> KirType {
        self.type_map.get(name).cloned().unwrap_or(KirType::F32)
    }

}

/// Lower a block of statements.
fn lower_block(lowerer: &mut KernelLowerer, block: &Block, interner: &Interner) {
    for stmt in &block.stmts {
        lower_stmt(lowerer, stmt, interner);
    }
}

/// Lower a single statement.
fn lower_stmt(lowerer: &mut KernelLowerer, stmt: &nsl_ast::stmt::Stmt, interner: &Interner) {
    match &stmt.kind {
        StmtKind::VarDecl { pattern, value, .. } => {
            // Extract variable name from pattern
            if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                let name = interner.resolve(sym.0).unwrap_or("_v").to_string();
                if let Some(expr) = value {
                    let (var_id, inferred_type) = lower_expr_typed(lowerer, expr, interner);
                    lowerer.var_map.insert(name.clone(), var_id);
                    // Task 5: record inferred type for this local variable
                    lowerer.type_map.insert(name, inferred_type);
                }
            }
        }
        StmtKind::Expr(expr) => {
            // Expression statement (e.g., function call like sync_threads())
            lower_expr_typed(lowerer, expr, interner);
        }
        _ => {
            // Other statement kinds (if/else, for, etc.) are not yet supported
            // in the portable KIR lowering. They fall back to direct AST->PTX.
        }
    }
}

// ---------------------------------------------------------------------------
// Typed expression lowering
// ---------------------------------------------------------------------------

/// Lower an expression, returning the VarId holding the result and its inferred KIR type.
///
/// Type propagation rules:
/// - Identifiers: look up type from `type_map`
/// - Int literals: U32
/// - Float literals: F32
/// - Subscript (a[idx]): dereferences Ptr(T, _) to T
/// - BinaryOp: promotes operand types via `promote_types()`, inserts Cast ops if needed
/// - Builtins (thread_id, etc.): U32
/// - Unary neg: preserves operand type
fn lower_expr_typed(
    lowerer: &mut KernelLowerer,
    expr: &Expr,
    interner: &Interner,
) -> (VarId, KirType) {
    match &expr.kind {
        ExprKind::IntLiteral(val) => {
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            lowerer.builder.emit(KirOp::Const(dst, KirConst {
                ty: KirType::U32,
                value: ConstValue::U32(*val as u32),
            }));
            (dst, KirType::U32)
        }
        ExprKind::FloatLiteral(val) => {
            let dst = lowerer.builder.new_typed_var(KirType::F32);
            lowerer.builder.emit(KirOp::Const(dst, KirConst {
                ty: KirType::F32,
                value: ConstValue::F32(*val as f32),
            }));
            (dst, KirType::F32)
        }
        ExprKind::Ident(sym) => {
            let name = interner.resolve(sym.0).unwrap_or("_");
            if let Some(&var_id) = lowerer.var_map.get(name) {
                let ty = lowerer.var_type(name);
                (var_id, ty)
            } else {
                // Unknown variable -- allocate a placeholder
                let dst = lowerer.builder.new_typed_var(KirType::U32);
                (dst, KirType::U32)
            }
        }
        ExprKind::BinaryOp { left, op, right } => {
            let (a, a_ty) = lower_expr_typed(lowerer, left, interner);
            let (b, b_ty) = lower_expr_typed(lowerer, right, interner);

            // Comparison operators always produce Bool
            match op {
                BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq | BinOp::Eq | BinOp::NotEq => {
                    let cmp_op = match op {
                        BinOp::Lt => CmpOp::Lt,
                        BinOp::LtEq => CmpOp::Le,
                        BinOp::Gt => CmpOp::Gt,
                        BinOp::GtEq => CmpOp::Ge,
                        BinOp::Eq => CmpOp::Eq,
                        BinOp::NotEq => CmpOp::Ne,
                        _ => unreachable!(),
                    };
                    let cmp_dst = lowerer.builder.new_typed_var(KirType::Bool);
                    lowerer.builder.emit(KirOp::Cmp(cmp_dst, a, b, cmp_op));
                    return (cmp_dst, KirType::Bool);
                }
                _ => {}
            }

            // Arithmetic: promote types
            let result_ty = promote_types(a_ty.clone(), b_ty.clone());

            // Insert casts if needed
            let a = if a_ty != result_ty {
                let cast_dst = lowerer.builder.new_typed_var(result_ty.clone());
                lowerer.builder.emit(KirOp::Cast(cast_dst, a, result_ty.clone()));
                cast_dst
            } else {
                a
            };
            let b = if b_ty != result_ty {
                let cast_dst = lowerer.builder.new_typed_var(result_ty.clone());
                lowerer.builder.emit(KirOp::Cast(cast_dst, b, result_ty.clone()));
                cast_dst
            } else {
                b
            };

            let dst = lowerer.builder.new_typed_var(result_ty.clone());
            let kir_op = match op {
                BinOp::Add => KirOp::Add(dst, a, b),
                BinOp::Sub => KirOp::Sub(dst, a, b),
                BinOp::Mul => KirOp::Mul(dst, a, b),
                BinOp::Div => KirOp::Div(dst, a, b),
                _ => {
                    // Unsupported binary op -- return a as fallback
                    return (a, result_ty);
                }
            };
            lowerer.builder.emit(kir_op);
            (dst, result_ty)
        }
        ExprKind::Call { callee, args: _ } => {
            // Recognize builtins: thread_id(), sync_threads(), block_idx(), etc.
            if let ExprKind::Ident(sym) = &callee.kind {
                let name = interner.resolve(sym.0).unwrap_or("");
                match name {
                    "thread_id" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::ThreadId(dst, 0));
                        return (dst, KirType::U32);
                    }
                    "block_idx" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::BlockIdx(dst, 0));
                        return (dst, KirType::U32);
                    }
                    "block_dim" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::BlockDim(dst, 0));
                        return (dst, KirType::U32);
                    }
                    "global_id" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::GlobalId(dst, 0));
                        return (dst, KirType::U32);
                    }
                    "sync_threads" => {
                        lowerer.builder.emit(KirOp::Barrier);
                        // Barrier has no result value; return a dummy
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        return (dst, KirType::U32);
                    }
                    _ => {}
                }
            }
            // Unrecognized call -- return a placeholder
            let dst = lowerer.builder.new_typed_var(KirType::F32);
            (dst, KirType::F32)
        }
        ExprKind::Subscript { object, index } => {
            // a[tid] -> PtrOffset + Load
            // The loaded value's type comes from dereferencing the pointer type.
            let (base, base_ty) = lower_expr_typed(lowerer, object, interner);

            // Determine the element type by dereferencing the pointer
            let elem_ty = match &base_ty {
                KirType::Ptr(inner, _) => *inner.clone(),
                _ => KirType::F32, // fallback
            };

            if let SubscriptKind::Index(idx_expr) = index.as_ref() {
                let (offset, _) = lower_expr_typed(lowerer, idx_expr, interner);
                let addr = lowerer.builder.new_typed_var(
                    KirType::Ptr(Box::new(elem_ty.clone()), AddressSpace::Global),
                );
                lowerer.builder.emit(KirOp::PtrOffset(addr, base, offset));
                let val = lowerer.builder.new_typed_var(elem_ty.clone());
                lowerer.builder.emit(KirOp::Load(val, addr, AddressSpace::Global));
                (val, elem_ty)
            } else {
                // Unsupported subscript kind
                let dst = lowerer.builder.new_typed_var(elem_ty.clone());
                (dst, elem_ty)
            }
        }
        ExprKind::UnaryOp { op, operand } => {
            let (src, src_ty) = lower_expr_typed(lowerer, operand, interner);
            match op {
                nsl_ast::operator::UnaryOp::Neg => {
                    let dst = lowerer.builder.new_typed_var(src_ty.clone());
                    lowerer.builder.emit(KirOp::Neg(dst, src));
                    (dst, src_ty)
                }
                _ => (src, src_ty),
            }
        }
        _ => {
            // Unsupported expression -- return a placeholder variable
            let dst = lowerer.builder.new_typed_var(KirType::F32);
            (dst, KirType::F32)
        }
    }
}

/// Legacy wrapper: lower an expression returning only the VarId.
/// Used by code that doesn't need the type information.
#[allow(dead_code)]
fn lower_expr(lowerer: &mut KernelLowerer, expr: &Expr, interner: &Interner) -> VarId {
    lower_expr_typed(lowerer, expr, interner).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use string_interner::StringInterner;
    use nsl_ast::block::KernelDef;
    use nsl_ast::decl::Param;
    use nsl_ast::pattern::{Pattern, PatternKind};
    use nsl_ast::stmt::{Block, Stmt, StmtKind};
    use nsl_ast::expr::{Expr, ExprKind};
    use nsl_ast::types::{TypeExpr, TypeExprKind, DimExpr};
    use nsl_ast::NodeId;
    use nsl_errors::Span;

    fn make_interner() -> Interner {
        StringInterner::new()
    }

    fn dummy_span() -> Span {
        Span {
            file_id: nsl_errors::FileId(0),
            start: nsl_errors::BytePos(0),
            end: nsl_errors::BytePos(0),
        }
    }

    fn make_empty_kernel(interner: &mut Interner) -> KernelDef {
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("test_kernel"));
        KernelDef {
            name: name_sym,
            params: Vec::new(),
            return_type: None,
            body: Block { stmts: Vec::new(), span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        }
    }

    fn make_kernel_with_params(interner: &mut Interner) -> KernelDef {
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("add_kernel"));
        let p_a = nsl_ast::Symbol(interner.get_or_intern("a"));
        let p_b = nsl_ast::Symbol(interner.get_or_intern("b"));
        KernelDef {
            name: name_sym,
            params: vec![
                Param { name: p_a, type_ann: None, default: None, is_variadic: false, span: dummy_span() },
                Param { name: p_b, type_ann: None, default: None, is_variadic: false, span: dummy_span() },
            ],
            return_type: None,
            body: Block { stmts: Vec::new(), span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        }
    }

    /// Helper to create a tensor type annotation with the given dtype.
    fn make_tensor_type_ann(interner: &mut Interner, dtype_str: &str) -> TypeExpr {
        let dtype_sym = nsl_ast::Symbol(interner.get_or_intern(dtype_str));
        TypeExpr {
            kind: TypeExprKind::Tensor {
                shape: vec![DimExpr::Concrete(1024)],
                dtype: dtype_sym,
                device: None,
            },
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    /// Helper to create a named (scalar) type annotation.
    fn make_named_type_ann(interner: &mut Interner, name: &str) -> TypeExpr {
        let sym = nsl_ast::Symbol(interner.get_or_intern(name));
        TypeExpr {
            kind: TypeExprKind::Named(sym),
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    // -----------------------------------------------------------------------
    // Task 1: Type promotion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_promotion_same_types() {
        assert_eq!(promote_types(KirType::F32, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::F64, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::I32, KirType::I32), KirType::I32);
        assert_eq!(promote_types(KirType::I64, KirType::I64), KirType::I64);
        assert_eq!(promote_types(KirType::U32, KirType::U32), KirType::U32);
        assert_eq!(promote_types(KirType::F16, KirType::F16), KirType::F16);
    }

    #[test]
    fn test_type_promotion_mixed_float() {
        assert_eq!(promote_types(KirType::F16, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::F32, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::F16, KirType::F64), KirType::F64);
        // Symmetric
        assert_eq!(promote_types(KirType::F32, KirType::F16), KirType::F32);
        assert_eq!(promote_types(KirType::F64, KirType::F32), KirType::F64);
    }

    #[test]
    fn test_type_promotion_int_float() {
        assert_eq!(promote_types(KirType::I32, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::I32, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::I64, KirType::F32), KirType::F64); // precision safety
        assert_eq!(promote_types(KirType::I64, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::I32, KirType::F16), KirType::F32);
        // Symmetric
        assert_eq!(promote_types(KirType::F32, KirType::I32), KirType::F32);
        assert_eq!(promote_types(KirType::F64, KirType::I64), KirType::F64);
    }

    #[test]
    fn test_type_promotion_mixed_int() {
        assert_eq!(promote_types(KirType::I32, KirType::I64), KirType::I64);
        assert_eq!(promote_types(KirType::U32, KirType::U64), KirType::U64);
        assert_eq!(promote_types(KirType::U32, KirType::I32), KirType::I32);
    }

    #[test]
    fn test_type_promotion_bool() {
        assert_eq!(promote_types(KirType::Bool, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::Bool, KirType::I32), KirType::I32);
        assert_eq!(promote_types(KirType::F64, KirType::Bool), KirType::F64);
    }

    #[test]
    #[should_panic(expected = "Cannot promote types")]
    fn test_type_promotion_ptr_panics() {
        let ptr_a = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let ptr_b = KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global);
        promote_types(ptr_a, ptr_b);
    }

    // -----------------------------------------------------------------------
    // Task 2: Parameter type map tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_type_map_tensor_f64() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("f64_kernel"));
        let p_a = nsl_ast::Symbol(interner.get_or_intern("a"));
        let p_b = nsl_ast::Symbol(interner.get_or_intern("b"));
        let p_out = nsl_ast::Symbol(interner.get_or_intern("out"));

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: p_a,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_b,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "i32")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_out,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block { stmts: Vec::new(), span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let type_map = build_param_type_map(&kernel, &interner);

        assert_eq!(type_map["a"], KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global));
        assert_eq!(type_map["b"], KirType::Ptr(Box::new(KirType::I32), AddressSpace::Global));
        assert_eq!(type_map["out"], KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global));
    }

    #[test]
    fn test_param_type_map_scalar() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("scalar_kernel"));
        let p_n = nsl_ast::Symbol(interner.get_or_intern("n"));

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: p_n,
                    type_ann: Some(make_named_type_ann(&mut interner, "int")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block { stmts: Vec::new(), span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let type_map = build_param_type_map(&kernel, &interner);
        assert_eq!(type_map["n"], KirType::I32);
    }

    #[test]
    fn test_param_type_map_no_annotation_defaults_f32() {
        let mut interner = make_interner();
        let kernel = make_kernel_with_params(&mut interner);
        let type_map = build_param_type_map(&kernel, &interner);

        // No type annotation -> Ptr(F32, Global) default
        assert_eq!(type_map["a"], KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        assert_eq!(type_map["b"], KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
    }

    // -----------------------------------------------------------------------
    // Task 3: Type propagation in expressions
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_propagation_f64_subscript() {
        // Kernel with f64 tensors: a[idx] + b[idx] should produce f64
        let mut interner = make_interner();
        let _name_sym = nsl_ast::Symbol(interner.get_or_intern("f64_add"));
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        // Build: a[idx] + b[idx]
        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let mut lowerer = KernelLowerer::new("test");
        // Set up type map: a and b are Ptr(F64, Global)
        let a_var = lowerer.builder.new_typed_var(
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global),
        );
        lowerer.var_map.insert("a".to_string(), a_var);
        lowerer.type_map.insert(
            "a".to_string(),
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global),
        );
        let b_var = lowerer.builder.new_typed_var(
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global),
        );
        lowerer.var_map.insert("b".to_string(), b_var);
        lowerer.type_map.insert(
            "b".to_string(),
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global),
        );
        // idx is U32 (thread index)
        let idx_var = lowerer.builder.new_typed_var(KirType::U32);
        lowerer.var_map.insert("idx".to_string(), idx_var);
        lowerer.type_map.insert("idx".to_string(), KirType::U32);

        // Need a block to emit into
        let entry = lowerer.builder.new_block();
        lowerer.builder.set_block(entry);

        let (_, result_ty) = lower_expr_typed(&mut lowerer, &add_expr, &interner);
        assert_eq!(result_ty, KirType::F64, "a[idx]+b[idx] with F64 ptrs should produce F64");
    }

    #[test]
    fn test_mixed_type_promotion_in_expr() {
        // a is Ptr(I32), b is Ptr(F32)
        // a[idx] + b[idx] should promote to F32
        let mut interner = make_interner();
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let mut lowerer = KernelLowerer::new("test");
        let a_var = lowerer.builder.new_typed_var(
            KirType::Ptr(Box::new(KirType::I32), AddressSpace::Global),
        );
        lowerer.var_map.insert("a".to_string(), a_var);
        lowerer.type_map.insert(
            "a".to_string(),
            KirType::Ptr(Box::new(KirType::I32), AddressSpace::Global),
        );
        let b_var = lowerer.builder.new_typed_var(
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        );
        lowerer.var_map.insert("b".to_string(), b_var);
        lowerer.type_map.insert(
            "b".to_string(),
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        );
        let idx_var = lowerer.builder.new_typed_var(KirType::U32);
        lowerer.var_map.insert("idx".to_string(), idx_var);
        lowerer.type_map.insert("idx".to_string(), KirType::U32);

        let entry = lowerer.builder.new_block();
        lowerer.builder.set_block(entry);

        let (_, result_ty) = lower_expr_typed(&mut lowerer, &add_expr, &interner);
        assert_eq!(result_ty, KirType::F32, "I32 + F32 should promote to F32");
    }

    // -----------------------------------------------------------------------
    // Task 4: Full kernel lowering with AST types
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_empty_kernel() {
        let mut interner = make_interner();
        let kernel = make_empty_kernel(&mut interner);
        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);

        assert_eq!(ir.name, "test_kernel");
        assert_eq!(ir.params.len(), 0);
        assert!(ir.is_well_formed());
        assert_eq!(ir.blocks.len(), 1); // entry block
    }

    #[test]
    fn test_lower_params() {
        let mut interner = make_interner();
        let kernel = make_kernel_with_params(&mut interner);
        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);

        assert_eq!(ir.name, "add_kernel");
        assert_eq!(ir.params.len(), 2);
        assert_eq!(ir.params[0].name, "a");
        assert_eq!(ir.params[1].name, "b");
        // No type annotation -> default Ptr(F32, Global)
        assert_eq!(ir.params[0].ty, KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
    }

    #[test]
    fn test_lower_f64_params() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("f64_kernel"));
        let p_a = nsl_ast::Symbol(interner.get_or_intern("a"));
        let p_out = nsl_ast::Symbol(interner.get_or_intern("out"));

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: p_a,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_out,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block { stmts: Vec::new(), span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);

        assert_eq!(ir.params[0].ty, KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global));
        assert_eq!(ir.params[1].ty, KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global));
    }

    #[test]
    fn test_lower_basic_ops() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("arith_kernel"));
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        // Build: let x = 1.0 + 2.0
        let left = Expr {
            kind: ExprKind::FloatLiteral(1.0),
            span: dummy_span(),
            id: NodeId::next(),
        };
        let right = Expr {
            kind: ExprKind::FloatLiteral(2.0),
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(left),
                op: BinOp::Add,
                right: Box::new(right),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let var_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(x_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(add_expr),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: Vec::new(),
            return_type: None,
            body: Block { stmts: vec![var_decl], span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);

        // Should have ops: Const(1.0), Const(2.0), Add
        assert!(ir.op_count() >= 3, "expected at least 3 ops, got {}", ir.op_count());
    }

    #[test]
    fn test_feature_tracking() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("sync_kernel"));
        let sync_sym = nsl_ast::Symbol(interner.get_or_intern("sync_threads"));

        // Build: sync_threads()
        let call_expr = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(sync_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let expr_stmt = Stmt {
            kind: StmtKind::Expr(call_expr),
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: Vec::new(),
            return_type: None,
            body: Block { stmts: vec![expr_stmt], span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);

        // sync_threads() should set SHARED_MEMORY feature
        assert!(
            ir.required_features.contains(crate::gpu_target::FeatureSet::SHARED_MEMORY),
            "sync_threads() should require SHARED_MEMORY feature"
        );
    }

    // -----------------------------------------------------------------------
    // Task 5: Local variable type inference
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_variable_type_inference() {
        // let x = a[idx]  -> x should be F64 if a is Ptr(F64)
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("local_infer_kernel"));
        let a_sym_param = nsl_ast::Symbol(interner.get_or_intern("a"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        // Build: let x = a[idx]
        let a_subscript = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym_param),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let var_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(x_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(a_subscript),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: a_sym_param,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block { stmts: vec![var_decl], span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);

        // The load from a[idx] should produce an F64 value.
        // Check that the loaded variable has F64 type in var_types.
        let f64_vars: Vec<_> = ir.var_types.iter()
            .filter(|(_, ty)| **ty == KirType::F64)
            .collect();
        assert!(
            !f64_vars.is_empty(),
            "Expected at least one F64 variable from `let x = a[idx]` with F64 tensor 'a'. \
             var_types: {:?}",
            ir.var_types,
        );
    }

    // -----------------------------------------------------------------------
    // Task 6: E2E PTX tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f64_kernel_ptx_uses_f64_ops() {
        // Build an F64 kernel: a[idx] + b[idx] -> out[idx]
        // Verify the PTX output uses .f64 operations.
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("add_f64"));
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let out_sym = nsl_ast::Symbol(interner.get_or_intern("out"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        // Build body:
        //   let idx = global_id()
        //   let val = a[idx] + b[idx]
        // (We don't implement store via statement yet, so test just the typed expressions)

        let global_id_call = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(nsl_ast::Symbol(interner.get_or_intern("global_id"))),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let idx_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(global_id_call),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let val_sym = nsl_ast::Symbol(interner.get_or_intern("val"));
        let val_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(val_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(add_expr),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: a_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: b_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: out_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block { stmts: vec![idx_decl, val_decl], span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);
        let ptx_bytes = crate::backend_ptx::lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]).to_string();

        // F64 kernel: should use .f64 for the add operation
        assert!(
            ptx.contains("add.f64"),
            "F64 kernel should produce add.f64 in PTX. Got:\n{ptx}"
        );
        // The add should NOT be f32
        assert!(
            !ptx.contains("add.f32"),
            "F64 kernel should NOT produce add.f32. Got:\n{ptx}"
        );
        // Load should be f64
        assert!(
            ptx.contains("ld.global.f64"),
            "F64 kernel should load .f64 from global. Got:\n{ptx}"
        );
    }

    #[test]
    fn test_mixed_precision_kernel_ptx() {
        // a is Tensor<[1024], i32>, b is Tensor<[1024], f32>
        // a[idx] + b[idx] should produce cvt (cast) + add.f32
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("mixed_add"));
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        let global_id_call = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(nsl_ast::Symbol(interner.get_or_intern("global_id"))),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let idx_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(global_id_call),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let val_sym = nsl_ast::Symbol(interner.get_or_intern("val"));
        let val_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(val_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(add_expr),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: a_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "i32")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: b_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f32")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block { stmts: vec![idx_decl, val_decl], span: dummy_span() },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda);
        let ptx_bytes = crate::backend_ptx::lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]).to_string();

        // I32 + F32 should promote to F32
        assert!(
            ptx.contains("add.f32"),
            "Mixed I32+F32 kernel should produce add.f32 in PTX. Got:\n{ptx}"
        );
        // Should have a cvt (cast from i32 to f32)
        assert!(
            ptx.contains("cvt.f32.i32"),
            "Mixed I32+F32 kernel should produce cvt.f32.i32 in PTX. Got:\n{ptx}"
        );
        // Load from a should be i32
        assert!(
            ptx.contains("ld.global.i32"),
            "I32 tensor should load as .i32. Got:\n{ptx}"
        );
        // Load from b should be f32
        assert!(
            ptx.contains("ld.global.f32"),
            "F32 tensor should load as .f32. Got:\n{ptx}"
        );
    }

    // -----------------------------------------------------------------------
    // dtype_str_to_kir coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_str_to_kir() {
        assert_eq!(dtype_str_to_kir("f16"), KirType::F16);
        assert_eq!(dtype_str_to_kir("fp16"), KirType::F16);
        assert_eq!(dtype_str_to_kir("bf16"), KirType::Bf16);
        assert_eq!(dtype_str_to_kir("f32"), KirType::F32);
        assert_eq!(dtype_str_to_kir("fp32"), KirType::F32);
        assert_eq!(dtype_str_to_kir("float"), KirType::F32);
        assert_eq!(dtype_str_to_kir("f64"), KirType::F64);
        assert_eq!(dtype_str_to_kir("float64"), KirType::F64);
        assert_eq!(dtype_str_to_kir("double"), KirType::F64);
        assert_eq!(dtype_str_to_kir("i32"), KirType::I32);
        assert_eq!(dtype_str_to_kir("int"), KirType::I32);
        assert_eq!(dtype_str_to_kir("i64"), KirType::I64);
        assert_eq!(dtype_str_to_kir("u32"), KirType::U32);
        assert_eq!(dtype_str_to_kir("u64"), KirType::U64);
        assert_eq!(dtype_str_to_kir("bool"), KirType::Bool);
        assert_eq!(dtype_str_to_kir("fp8"), KirType::F16);
        assert_eq!(dtype_str_to_kir("unknown"), KirType::F32); // fallback
    }
}
