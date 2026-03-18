// crates/nsl-codegen/src/kernel_lower.rs
//! M47: AST KernelDef -> KIR lowering.
//!
//! Translates a parsed `KernelDef` AST node into the backend-agnostic `KernelIR`.
//! Handles the portable subset: arithmetic, thread indexing, memory access, barriers.
//! Complex kernel bodies that use NSL features beyond the portable subset should
//! fall back to the existing direct AST->PTX path in `kernel.rs`.

use std::collections::HashMap;

use nsl_ast::block::KernelDef;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::operator::BinOp;
use nsl_ast::stmt::{StmtKind, Block};
use nsl_lexer::Interner;

use crate::gpu_target::GpuTarget;
use crate::kernel_ir::*;

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
    let mut lowerer = KernelLowerer::new(&name);

    // Map kernel parameters to KirParams.
    // All kernel params are treated as u64 pointers by default (matching existing kernel.rs).
    for param in &kernel.params {
        let pname = interner.resolve(param.name.0).unwrap_or("_p").to_string();
        let var_id = lowerer.builder.add_param(
            &pname,
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        lowerer.var_map.insert(pname, var_id);
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
}

impl KernelLowerer {
    fn new(name: &str) -> Self {
        KernelLowerer {
            builder: KirBuilder::new(name),
            var_map: HashMap::new(),
        }
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
                    let var_id = lower_expr(lowerer, expr, interner);
                    lowerer.var_map.insert(name, var_id);
                }
            }
        }
        StmtKind::Expr(expr) => {
            // Expression statement (e.g., function call like sync_threads())
            lower_expr(lowerer, expr, interner);
        }
        _ => {
            // Other statement kinds (if/else, for, etc.) are not yet supported
            // in the portable KIR lowering. They fall back to direct AST->PTX.
        }
    }
}

/// Lower an expression, returning the VarId holding the result.
fn lower_expr(lowerer: &mut KernelLowerer, expr: &Expr, interner: &Interner) -> VarId {
    match &expr.kind {
        ExprKind::IntLiteral(val) => {
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            lowerer.builder.emit(KirOp::Const(dst, KirConst {
                ty: KirType::U32,
                value: ConstValue::U32(*val as u32),
            }));
            dst
        }
        ExprKind::FloatLiteral(val) => {
            let dst = lowerer.builder.new_typed_var(KirType::F32);
            lowerer.builder.emit(KirOp::Const(dst, KirConst {
                ty: KirType::F32,
                value: ConstValue::F32(*val as f32),
            }));
            dst
        }
        ExprKind::Ident(sym) => {
            let name = interner.resolve(sym.0).unwrap_or("_");
            if let Some(&var_id) = lowerer.var_map.get(name) {
                var_id
            } else {
                // Unknown variable -- allocate a placeholder
                
                lowerer.builder.new_typed_var(KirType::U32)
            }
        }
        ExprKind::BinaryOp { left, op, right } => {
            let a = lower_expr(lowerer, left, interner);
            let b = lower_expr(lowerer, right, interner);
            let dst = lowerer.builder.new_typed_var(KirType::F32);
            let kir_op = match op {
                BinOp::Add => KirOp::Add(dst, a, b),
                BinOp::Sub => KirOp::Sub(dst, a, b),
                BinOp::Mul => KirOp::Mul(dst, a, b),
                BinOp::Div => KirOp::Div(dst, a, b),
                BinOp::Lt => {
                    let cmp_dst = lowerer.builder.new_typed_var(KirType::Bool);
                    lowerer.builder.emit(KirOp::Cmp(cmp_dst, a, b, CmpOp::Lt));
                    return cmp_dst;
                }
                BinOp::Gt => {
                    let cmp_dst = lowerer.builder.new_typed_var(KirType::Bool);
                    lowerer.builder.emit(KirOp::Cmp(cmp_dst, a, b, CmpOp::Gt));
                    return cmp_dst;
                }
                BinOp::Eq => {
                    let cmp_dst = lowerer.builder.new_typed_var(KirType::Bool);
                    lowerer.builder.emit(KirOp::Cmp(cmp_dst, a, b, CmpOp::Eq));
                    return cmp_dst;
                }
                _ => {
                    // Unsupported binary op -- return a as fallback
                    return a;
                }
            };
            lowerer.builder.emit(kir_op);
            dst
        }
        ExprKind::Call { callee, args: _ } => {
            // Recognize builtins: thread_id(), sync_threads(), block_idx(), etc.
            if let ExprKind::Ident(sym) = &callee.kind {
                let name = interner.resolve(sym.0).unwrap_or("");
                match name {
                    "thread_id" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::ThreadId(dst, 0));
                        return dst;
                    }
                    "block_idx" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::BlockIdx(dst, 0));
                        return dst;
                    }
                    "block_dim" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::BlockDim(dst, 0));
                        return dst;
                    }
                    "global_id" => {
                        let dst = lowerer.builder.new_typed_var(KirType::U32);
                        lowerer.builder.emit(KirOp::GlobalId(dst, 0));
                        return dst;
                    }
                    "sync_threads" => {
                        lowerer.builder.emit(KirOp::Barrier);
                        // Barrier has no result value; return a dummy
                        return lowerer.builder.new_typed_var(KirType::U32);
                    }
                    _ => {}
                }
            }
            // Unrecognized call -- return a placeholder
            lowerer.builder.new_typed_var(KirType::F32)
        }
        ExprKind::Subscript { object, index } => {
            // a[tid] -> PtrOffset + Load
            let base = lower_expr(lowerer, object, interner);
            if let SubscriptKind::Index(idx_expr) = index.as_ref() {
                let offset = lower_expr(lowerer, idx_expr, interner);
                let addr = lowerer.builder.new_typed_var(
                    KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
                );
                lowerer.builder.emit(KirOp::PtrOffset(addr, base, offset));
                let val = lowerer.builder.new_typed_var(KirType::F32);
                lowerer.builder.emit(KirOp::Load(val, addr, AddressSpace::Global));
                val
            } else {
                // Unsupported subscript kind
                lowerer.builder.new_typed_var(KirType::F32)
            }
        }
        ExprKind::UnaryOp { op, operand } => {
            let src = lower_expr(lowerer, operand, interner);
            match op {
                nsl_ast::operator::UnaryOp::Neg => {
                    let dst = lowerer.builder.new_typed_var(KirType::F32);
                    lowerer.builder.emit(KirOp::Neg(dst, src));
                    dst
                }
                _ => src,
            }
        }
        _ => {
            // Unsupported expression -- return a placeholder variable
            lowerer.builder.new_typed_var(KirType::F32)
        }
    }
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
        // All params are Ptr(F32, Global) by default
        assert_eq!(ir.params[0].ty, KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
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
}
