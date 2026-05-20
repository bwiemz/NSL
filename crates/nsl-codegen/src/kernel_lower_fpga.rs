//! M57.1 §3.3 — FPGA-specific AST → structured-KIR lowerer.
//!
//! Recognizes the v1 MLP shape:
//!
//!     model TinyMlp:
//!         W<i>: Tensor<[K, N], dtype>
//!         b<i>: Tensor<[N], acc_dtype>
//!         ...
//!         fn forward(self, x: Tensor<[1, K], dtype>) -> Tensor<[1, N], acc_dtype>:
//!             let h<i> = relu(matmul(<prev>, self.W<i>) + self.b<i>)
//!             ...
//!             return relu(matmul(<prev>, self.W<N>) + self.b<N>)
//!
//! Emits 3 structured KIR ops per layer in this order: `KirOp::Matmul`,
//! `KirOp::ElementwiseAdd`, `KirOp::Relu`. Errors on unrecognized shapes per
//! [M57 §2.5]'s op-validated-shape-accepting discipline.
//!
//! Blast-radius isolation (M57.1 spec Q5): dispatched at the AST→KIR entry point
//! via a single additive `if target == Fpga` branch; GPU codegen path untouched.
//!
//! ## Actual AST shape used by the recognizer (verified against `crates/nsl-ast/src/`):
//!
//! - `Module.stmts: Vec<Stmt>` — top-level items live here.
//! - `StmtKind::ModelDef(ModelDef)` — `model TinyMlp: ...` decl.
//! - `ModelDef.members: Vec<ModelMember>` where each member is either
//!   `ModelMember::LayerDecl { name, type_ann, init, decorators, span }` (the
//!   `W<i>: Tensor<[K,N], dtype>` field declarations) or
//!   `ModelMember::Method(FnDef, Vec<Decorator>)` (the `fn forward(...)` method).
//! - `FnDef.body: Block` with `Block.stmts: Vec<Stmt>`. Returns are
//!   `StmtKind::Return(Option<Expr>)` inside the body (no dedicated `return_expr`).
//! - `ExprKind::Call { callee: Box<Expr>, args: Vec<Arg> }` — function calls,
//!   `Arg { name: Option<Symbol>, value: Expr, span }`.
//! - `ExprKind::BinaryOp { left, op: BinOp, right }` — `BinOp::Add` is bias add.
//! - `ExprKind::MemberAccess { object: Box<Expr>, member: Symbol }` — `self.W1`
//!   is a MemberAccess whose `object.kind == ExprKind::SelfRef` and whose
//!   `member` is the interned field name.
//! - `ExprKind::Ident(Symbol)` — bare identifiers (the `matmul` / `relu` callees).
//! - `TypeExprKind::Tensor { shape: Vec<DimExpr>, dtype: Symbol, device }` —
//!   tensor types. Shape is concrete via `DimExpr::Concrete(i64)`; dtype is a
//!   single interned symbol whose name we map to a `KirType` ourselves.

use crate::fpga_error::FpgaLoweringError;
use crate::kernel_ir::{KernelIR, KirBuilder, KirOp, KirTerminator, KirType, VarId};

use nsl_ast::decl::{FnDef, ModelMember};
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::operator::BinOp;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::types::{DimExpr, TypeExpr, TypeExprKind};
use nsl_ast::Module;
use nsl_lexer::Interner;

/// Repeated diagnostic strings — promoted to consts so refactors stay in sync
/// across the multiple error sites that reference the v1 field-naming rule.
const EXPECTED_FIELD_NAMING: &str = "fields named W1, b1, W2, b2, ... per layer";

/// Parse a `W<i>` or `b<i>` field name into its 1-based layer index.
/// Returns `None` if the suffix after the prefix char is not a non-empty
/// run of ASCII digits parseable as `usize`. Rejects `Wfoo` / `bias_alpha`
/// (silent classification under the old `starts_with` predicate).
fn parse_layer_index(name: &str, prefix: char) -> Option<usize> {
    let suffix = name.strip_prefix(prefix)?;
    if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<usize>().ok()
}

/// Entry point — invoked when the AST→KIR pipeline dispatches on
/// `target == GpuTarget::Fpga`.
///
/// Walks the parsed NSL `model` block's `fn forward` body, recognizes the
/// per-layer `relu(matmul(<prev>, self.W<i>) + self.b<i>)` pattern, and emits
/// the corresponding structured KIR ops (`Matmul` → `ElementwiseAdd` → `Relu`
/// per layer, in that order — required by the KIR→HIR bias-as-seed peek-ahead
/// in Task 3.1).
///
/// Returns `UnsupportedV1Shape` on any structural deviation; never panics.
pub fn lower(ast: &Module, interner: &Interner) -> Result<KernelIR, FpgaLoweringError> {
    // ── Locate the single model block ───────────────────────────────────────
    let mut model_iter = ast.stmts.iter().filter_map(|stmt| match &stmt.kind {
        StmtKind::ModelDef(m) => Some(m),
        _ => None,
    });
    let model = model_iter
        .next()
        .ok_or_else(|| FpgaLoweringError::UnsupportedV1Shape {
            found: "no `model` block in source".to_string(),
            expected: "exactly one `model` block defining the v1 MLP",
        })?;
    if model_iter.next().is_some() {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: "multiple `model` blocks in source".to_string(),
            expected: "exactly one `model` block defining the v1 MLP",
        });
    }

    // ── Classify members into (weights, biases, forward-method) ─────────────
    // Tuple shape: (layer_idx, field_name, type_ann). `layer_idx` is parsed
    // numerically from the `W<i>` / `b<i>` suffix so that W10 sorts after W2,
    // not before (lexicographic sort silently misorders at >= 10 layers).
    let mut weights: Vec<(usize, String, &TypeExpr)> = Vec::new();
    let mut biases: Vec<(usize, String, &TypeExpr)> = Vec::new();
    let mut forward: Option<&FnDef> = None;

    for member in &model.members {
        match member {
            ModelMember::LayerDecl { name, type_ann, .. } => {
                let field_name = interner
                    .resolve(name.0)
                    .ok_or_else(|| FpgaLoweringError::UnsupportedV1Shape {
                        found: "unresolved field symbol".to_string(),
                        expected: EXPECTED_FIELD_NAMING,
                    })?;
                if let Some(idx) = parse_layer_index(field_name, 'W') {
                    weights.push((idx, field_name.to_string(), type_ann));
                } else if let Some(idx) = parse_layer_index(field_name, 'b') {
                    biases.push((idx, field_name.to_string(), type_ann));
                } else {
                    return Err(FpgaLoweringError::UnsupportedV1Shape {
                        found: format!(
                            "model field `{field_name}` (expected W<i> or b<i> with digit suffix)"
                        ),
                        expected: EXPECTED_FIELD_NAMING,
                    });
                }
            }
            ModelMember::Method(fn_def, _decorators) => {
                if interner.resolve(fn_def.name.0) == Some("forward") {
                    if forward.is_some() {
                        return Err(FpgaLoweringError::UnsupportedV1Shape {
                            found: "multiple `fn forward` methods in model".to_string(),
                            expected: "exactly one fn forward(self, x)",
                        });
                    }
                    forward = Some(fn_def);
                } else {
                    return Err(FpgaLoweringError::UnsupportedV1Shape {
                        found: format!(
                            "unrecognized method `{}` in model",
                            interner.resolve(fn_def.name.0).unwrap_or("<unresolved>")
                        ),
                        expected: "only fn forward(self, x) — no other methods in v1",
                    });
                }
            }
        }
    }

    // Numerical sort by parsed layer index — `W10` correctly follows `W2`.
    weights.sort_by_key(|(idx, _, _)| *idx);
    biases.sort_by_key(|(idx, _, _)| *idx);

    if weights.is_empty() {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: "no weight fields declared".to_string(),
            expected: "at least one W<i>/b<i> pair (one per layer)",
        });
    }
    if weights.len() != biases.len() {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("{} weights, {} biases", weights.len(), biases.len()),
            expected: "equal non-zero counts of W and b fields (one per layer)",
        });
    }

    // Verify the indices form a contiguous run 1..=N so `W1, W3` without `W2`
    // errors instead of silently treating them as layers 1 and 2.
    for (expected, (got, _, _)) in (1..=weights.len()).zip(weights.iter()) {
        if *got != expected {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: format!(
                    "weight layer indices: {:?}, expected contiguous 1..={}",
                    weights.iter().map(|(i, _, _)| *i).collect::<Vec<_>>(),
                    weights.len()
                ),
                expected: "weight fields W1, W2, ..., WN with contiguous 1-based indices",
            });
        }
    }
    for (expected, (got, _, _)) in (1..=biases.len()).zip(biases.iter()) {
        if *got != expected {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: format!(
                    "bias layer indices: {:?}, expected contiguous 1..={}",
                    biases.iter().map(|(i, _, _)| *i).collect::<Vec<_>>(),
                    biases.len()
                ),
                expected: "bias fields b1, b2, ..., bN with contiguous 1-based indices",
            });
        }
    }

    let forward = forward.ok_or_else(|| FpgaLoweringError::UnsupportedV1Shape {
        found: "no `fn forward` method in model".to_string(),
        expected: "fn forward(self, x: Tensor<[1, K], dtype>) -> Tensor<[1, N], acc_dtype>",
    })?;

    // ── Validate forward signature shape: (self, x) ─────────────────────────
    if forward.params.len() != 2 {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("fn forward has {} params", forward.params.len()),
            expected: "fn forward(self, x: Tensor<[1, K], dtype>)",
        });
    }
    let self_param_name = interner.resolve(forward.params[0].name.0).unwrap_or("");
    if self_param_name != "self" {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("fn forward first param `{self_param_name}`"),
            expected: "fn forward(self, x: ...)",
        });
    }
    let x_param = &forward.params[1];
    let x_ty_expr =
        x_param
            .type_ann
            .as_ref()
            .ok_or_else(|| FpgaLoweringError::UnsupportedV1Shape {
                found: "fn forward `x` parameter has no type annotation".to_string(),
                expected: "x: Tensor<[1, K], dtype>",
            })?;
    let x_kir_type = kir_type_from_tensor(x_ty_expr, interner)?;

    // ── Build KIR ───────────────────────────────────────────────────────────
    let mut builder = KirBuilder::new("tiny_mlp");
    let blk = builder.new_block();
    builder.set_block(blk);

    // Allocate the initial `x` VarId — it's the input to layer 1's matmul.
    let mut prev_var: VarId = builder.new_typed_var(x_kir_type);
    let n_layers = weights.len();

    // Walk the forward body. v1 expects N-1 `let h<i> = relu(matmul(...) + b<i>)`
    // statements followed by `return relu(matmul(...) + b<N>)`.
    let stmts = &forward.body.stmts;
    let (let_stmts, return_expr) = split_return(stmts)?;

    if let_stmts.len() + 1 != n_layers {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!(
                "{} let-bindings + 1 return, but {} layers declared",
                let_stmts.len(),
                n_layers
            ),
            expected: "N-1 let-bindings (one per intermediate layer) + 1 return (final layer)",
        });
    }

    for (i, stmt) in let_stmts.iter().enumerate() {
        prev_var = emit_layer(
            &mut builder,
            stmt,
            &weights[i],
            &biases[i],
            prev_var,
            interner,
        )?;
    }
    // Final layer from the return expression.
    let _final_var = emit_return_layer(
        &mut builder,
        return_expr,
        &weights[n_layers - 1],
        &biases[n_layers - 1],
        prev_var,
        interner,
    )?;

    builder.terminate(KirTerminator::Return);
    Ok(builder.finalize())
}

/// Split a forward-body's statement list into `(let_stmts, return_expr)`.
///
/// v1 expects every statement except the last to be a let-binding, and the
/// last statement to be `return <expr>` with a non-empty expression.
fn split_return(stmts: &[Stmt]) -> Result<(&[Stmt], &Expr), FpgaLoweringError> {
    let (last, head) = stmts
        .split_last()
        .ok_or_else(|| FpgaLoweringError::UnsupportedV1Shape {
            found: "fn forward body is empty".to_string(),
            expected: "let h<i> = ... bindings followed by return relu(...)",
        })?;
    let ret_expr = match &last.kind {
        StmtKind::Return(Some(e)) => e,
        StmtKind::Return(None) => {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: "fn forward final statement is bare `return` with no expression".to_string(),
                expected: "return relu(matmul(<prev>, self.W<N>) + self.b<N>)",
            });
        }
        _ => {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: "fn forward final statement is not a return".to_string(),
                expected: "return relu(matmul(<prev>, self.W<N>) + self.b<N>)",
            });
        }
    };
    Ok((head, ret_expr))
}

/// Emit one layer's KIR ops from a `let h<i> = relu(matmul(...) + self.b<i>)`
/// statement. Returns the layer-output VarId (the ReLU's result).
fn emit_layer(
    builder: &mut KirBuilder,
    stmt: &Stmt,
    weight: &(usize, String, &TypeExpr),
    bias: &(usize, String, &TypeExpr),
    input_var: VarId,
    interner: &Interner,
) -> Result<VarId, FpgaLoweringError> {
    let init = match &stmt.kind {
        StmtKind::VarDecl {
            value: Some(v),
            is_const: false,
            ..
        } => v,
        _ => {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: "non-`let` statement in fn forward body".to_string(),
                expected: "let h<i> = relu(matmul(<prev>, self.W<i>) + self.b<i>)",
            });
        }
    };
    extract_relu_matmul_bias(builder, init, weight, bias, input_var, interner)
}

/// Emit the final layer's KIR ops from the return-expression of `fn forward`.
fn emit_return_layer(
    builder: &mut KirBuilder,
    return_expr: &Expr,
    weight: &(usize, String, &TypeExpr),
    bias: &(usize, String, &TypeExpr),
    input_var: VarId,
    interner: &Interner,
) -> Result<VarId, FpgaLoweringError> {
    extract_relu_matmul_bias(builder, return_expr, weight, bias, input_var, interner)
}

/// Recognize the per-layer expression `relu(matmul(<prev>, self.W<i>) + self.b<i>)`
/// and emit the corresponding `Matmul` → `ElementwiseAdd` → `Relu` ops, in that
/// order (the KIR→HIR pass relies on this order for the bias-as-seed peek-ahead).
fn extract_relu_matmul_bias(
    builder: &mut KirBuilder,
    expr: &Expr,
    weight: &(usize, String, &TypeExpr),
    bias: &(usize, String, &TypeExpr),
    input_var: VarId,
    interner: &Interner,
) -> Result<VarId, FpgaLoweringError> {
    // Unwrap a single layer of parens, if present (`relu(...)`'s arg is plain).
    let expr = strip_paren(expr);

    // Outer must be `relu(<inner>)`.
    let inner = match &expr.kind {
        ExprKind::Call { callee, args }
            if call_name_is(callee, interner, "relu") && args.len() == 1 =>
        {
            &args[0].value
        }
        _ => {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: "layer outer expression is not `relu(...)` with one arg".to_string(),
                expected: "relu(matmul(<prev>, self.W<i>) + self.b<i>)",
            });
        }
    };
    let inner = strip_paren(inner);

    // Inner must be `<matmul-call> + <bias-ref>`.
    let (matmul_call, bias_ref) = match &inner.kind {
        ExprKind::BinaryOp {
            left,
            op: BinOp::Add,
            right,
        } => (left.as_ref(), right.as_ref()),
        _ => {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: "layer inner expression is not `<matmul> + <bias>`".to_string(),
                expected: "matmul(<prev>, self.W<i>) + self.b<i>",
            });
        }
    };
    let matmul_call = strip_paren(matmul_call);
    let bias_ref = strip_paren(bias_ref);

    // The matmul side must be a `matmul(<prev>, self.W<i>)` call with 2 args.
    let (_matmul_x, matmul_w) = match &matmul_call.kind {
        ExprKind::Call { callee, args }
            if call_name_is(callee, interner, "matmul") && args.len() == 2 =>
        {
            (&args[0].value, &args[1].value)
        }
        _ => {
            return Err(FpgaLoweringError::UnsupportedV1Shape {
                found: "layer matmul side is not `matmul(<prev>, <W>)`".to_string(),
                expected: "matmul(<prev>, self.W<i>)",
            });
        }
    };

    // Confirm matmul_w references `self.<weight-name>` and bias_ref references
    // `self.<bias-name>`.
    confirm_self_field(matmul_w, &weight.1, interner)?;
    confirm_self_field(bias_ref, &bias.1, interner)?;

    // Extract weight shape [K, N] and dtypes from the type annotations.
    let w_shape = tensor_shape_rank2(weight.2, &weight.1)?;
    let bias_shape = tensor_shape_rank1(bias.2, &bias.1)?;
    if bias_shape != w_shape[1] {
        return Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!(
                "bias `{}` has shape [{}] but weight `{}` has output dim {}",
                bias.1, bias_shape, weight.1, w_shape[1]
            ),
            expected: "bias rank-1 shape [N] matching weight rank-2 shape [K, N]",
        });
    }

    let a_kir_dtype = builder
        .var_type(input_var)
        .expect("input_var must have a recorded type");
    let b_kir_dtype = kir_type_from_tensor(weight.2, interner)?;
    let out_kir_dtype = kir_type_from_tensor(bias.2, interner)?;

    let matmul_out = builder.new_typed_var(out_kir_dtype.clone());
    let bias_out = builder.new_typed_var(out_kir_dtype.clone());
    let relu_out = builder.new_typed_var(out_kir_dtype.clone());
    let weight_var = builder.new_typed_var(b_kir_dtype.clone());
    let bias_var = builder.new_typed_var(out_kir_dtype.clone());

    let a_shape: [usize; 2] = [1, w_shape[0]];
    let b_shape: [usize; 2] = [w_shape[0], w_shape[1]];
    let elementwise_shape: [usize; 1] = [w_shape[1]];

    builder.emit(KirOp::Matmul {
        a: input_var,
        b: weight_var,
        out: matmul_out,
        a_dtype: a_kir_dtype,
        b_dtype: b_kir_dtype,
        out_dtype: out_kir_dtype.clone(),
        a_shape,
        b_shape,
    });

    builder.emit(KirOp::ElementwiseAdd {
        a: matmul_out,
        b: bias_var,
        out: bias_out,
        dtype: out_kir_dtype.clone(),
        shape: elementwise_shape,
    });

    builder.emit(KirOp::Relu {
        a: bias_out,
        out: relu_out,
        dtype: out_kir_dtype,
        shape: elementwise_shape,
    });

    Ok(relu_out)
}

// ── Helper: small predicates / extractors ───────────────────────────────────

fn strip_paren(expr: &Expr) -> &Expr {
    let mut cur = expr;
    while let ExprKind::Paren(inner) = &cur.kind {
        cur = inner;
    }
    cur
}

fn call_name_is(callee: &Expr, interner: &Interner, expected: &str) -> bool {
    matches!(&strip_paren(callee).kind,
        ExprKind::Ident(sym) if interner.resolve(sym.0) == Some(expected))
}

/// Confirm an expression is `self.<expected_field>` (a `MemberAccess` whose
/// object is `SelfRef` and whose member name matches).
fn confirm_self_field(
    expr: &Expr,
    expected_name: &str,
    interner: &Interner,
) -> Result<(), FpgaLoweringError> {
    let expr = strip_paren(expr);
    match &expr.kind {
        ExprKind::MemberAccess { object, member }
            if matches!(strip_paren(object).kind, ExprKind::SelfRef)
                && interner.resolve(member.0) == Some(expected_name) =>
        {
            Ok(())
        }
        _ => Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("expected `self.{expected_name}` field reference"),
            expected: "self.W<i> / self.b<i> referencing the matching layer field",
        }),
    }
}

/// Pull the `[K, N]` shape pair from a rank-2 `Tensor<[K, N], dtype>` type.
fn tensor_shape_rank2(
    ty: &TypeExpr,
    field_name: &str,
) -> Result<[usize; 2], FpgaLoweringError> {
    match &ty.kind {
        TypeExprKind::Tensor { shape, .. } if shape.len() == 2 => {
            Ok([extract_dim(&shape[0])?, extract_dim(&shape[1])?])
        }
        _ => Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("weight `{field_name}` is not a rank-2 Tensor"),
            expected: "Tensor<[K, N], dtype> (rank-2)",
        }),
    }
}

/// Pull the `[N]` shape from a rank-1 `Tensor<[N], dtype>` type.
fn tensor_shape_rank1(ty: &TypeExpr, field_name: &str) -> Result<usize, FpgaLoweringError> {
    match &ty.kind {
        TypeExprKind::Tensor { shape, .. } if shape.len() == 1 => extract_dim(&shape[0]),
        _ => Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("bias `{field_name}` is not a rank-1 Tensor"),
            expected: "Tensor<[N], dtype> (rank-1)",
        }),
    }
}

fn extract_dim(d: &DimExpr) -> Result<usize, FpgaLoweringError> {
    match d {
        DimExpr::Concrete(n) if *n > 0 => Ok(*n as usize),
        _ => Err(FpgaLoweringError::UnsupportedV1Shape {
            found: "non-literal or non-positive tensor dimension".to_string(),
            expected: "literal positive dimension (K or N)",
        }),
    }
}

/// Map a `TypeExpr` that must be `Tensor<[...], dtype>` to a `KirType` (looking
/// up the dtype-symbol string and converting via `kir_type_from_dtype_name`).
fn kir_type_from_tensor(ty: &TypeExpr, interner: &Interner) -> Result<KirType, FpgaLoweringError> {
    match &ty.kind {
        TypeExprKind::Tensor { dtype, .. } => {
            let name = interner
                .resolve(dtype.0)
                .ok_or_else(|| FpgaLoweringError::UnsupportedV1Shape {
                    found: "unresolved tensor dtype symbol".to_string(),
                    expected: "tensor dtype in {i8, i16, i32, i64}",
                })?;
            kir_type_from_dtype_name(name)
        }
        _ => Err(FpgaLoweringError::UnsupportedV1Shape {
            found: "non-tensor type where Tensor<[...], dtype> was required".to_string(),
            expected: "Tensor<[shape], dtype>",
        }),
    }
}

/// Map an NSL dtype name to a `KirType`. v1 supports the integer dtypes only.
fn kir_type_from_dtype_name(name: &str) -> Result<KirType, FpgaLoweringError> {
    match name {
        "i8" => Ok(KirType::I8),
        "i16" => Ok(KirType::I16),
        "i32" => Ok(KirType::I32),
        "i64" => Ok(KirType::I64),
        other => Err(FpgaLoweringError::UnsupportedV1Shape {
            found: format!("dtype `{other}` is not supported in v1"),
            expected: "i8 / i16 / i32 / i64",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::KirOp;
    use nsl_lexer::Interner as LexerInterner;

    fn parse_source(src: &str) -> (Module, LexerInterner) {
        let mut interner = LexerInterner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let errs: Vec<_> = lex_diags
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .collect();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        let errs: Vec<_> = parsed
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .collect();
        assert!(errs.is_empty(), "parse errors: {errs:?}");
        (parsed.module, interner)
    }

    #[test]
    fn empty_module_returns_unsupported() {
        // Smoke-test only — Task 2.4 ships comprehensive fixture coverage.
        let (ast, interner) = parse_source("");
        let err = lower(&ast, &interner).expect_err("empty module must error");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }

    #[test]
    fn parse_layer_index_rejects_non_digit_suffix() {
        // Unit-level coverage of the helper itself.
        assert_eq!(parse_layer_index("W1", 'W'), Some(1));
        assert_eq!(parse_layer_index("W10", 'W'), Some(10));
        assert_eq!(parse_layer_index("b3", 'b'), Some(3));
        assert_eq!(parse_layer_index("Wfoo", 'W'), None);
        assert_eq!(parse_layer_index("bias_alpha", 'b'), None);
        assert_eq!(parse_layer_index("W", 'W'), None); // empty suffix
        assert_eq!(parse_layer_index("W1a", 'W'), None); // mixed
        assert_eq!(parse_layer_index("b", 'W'), None); // wrong prefix
    }

    #[test]
    fn rejects_non_digit_suffix_field_names() {
        // `Wfoo` is not `W<digits>` — must error, not silently classify as a
        // weight (the old `starts_with('W')` predicate would have accepted it
        // and produced a confusing downstream shape error).
        let src = "\
model TinyMlp:
    Wfoo: Tensor<[784, 128], i8>
    b1: Tensor<[128], i32>

    fn forward(self, x: Tensor<[1, 784], i8>) -> Tensor<[1, 128], i32>:
        return relu(matmul(x, self.Wfoo) + self.b1)
";
        let (ast, interner) = parse_source(src);
        let err = lower(&ast, &interner).expect_err("Wfoo must reject");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }

    #[test]
    fn rejects_skipped_layer_index() {
        // W1 + W3 without W2 — must error, not silently re-index as layers
        // 1 and 2 (lexicographic sort + position-based indexing combo bug).
        let src = "\
model TinyMlp:
    W1: Tensor<[784, 128], i8>
    b1: Tensor<[128], i32>
    W3: Tensor<[128, 10], i32>
    b3: Tensor<[10], i64>

    fn forward(self, x: Tensor<[1, 784], i8>) -> Tensor<[1, 10], i64>:
        let h = relu(matmul(x, self.W1) + self.b1)
        return relu(matmul(h, self.W3) + self.b3)
";
        let (ast, interner) = parse_source(src);
        let err = lower(&ast, &interner).expect_err("skipped index must reject");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }

    #[test]
    fn v1_mlp_source_lowers_to_6_kir_ops() {
        let src = "\
model TinyMlp:
    W1: Tensor<[784, 128], i8>
    b1: Tensor<[128], i32>
    W2: Tensor<[128, 10], i32>
    b2: Tensor<[10], i64>

    fn forward(self, x: Tensor<[1, 784], i8>) -> Tensor<[1, 10], i64>:
        let h = relu(matmul(x, self.W1) + self.b1)
        return relu(matmul(h, self.W2) + self.b2)
";
        let (ast, interner) = parse_source(src);
        let kir = lower(&ast, &interner).expect("v1 MLP source must lower");
        let ops: Vec<&KirOp> = kir.ops().collect();
        assert_eq!(ops.len(), 6, "expected 6 KIR ops (3 per layer x 2 layers)");
        assert!(matches!(ops[0], KirOp::Matmul { .. }));
        assert!(matches!(ops[1], KirOp::ElementwiseAdd { .. }));
        assert!(matches!(ops[2], KirOp::Relu { .. }));
        assert!(matches!(ops[3], KirOp::Matmul { .. }));
        assert!(matches!(ops[4], KirOp::ElementwiseAdd { .. }));
        assert!(matches!(ops[5], KirOp::Relu { .. }));
    }

    #[test]
    fn missing_bias_layer_rejects() {
        // M57.1 §3.3: model fields must include both W<i> and b<i> per layer.
        // A model with only W<i> (no bias) must error with UnsupportedV1Shape.
        let src = "\
model NoBias:
    W1: Tensor<[784, 128], i8>

    fn forward(self, x: Tensor<[1, 784], i8>) -> Tensor<[1, 128], i32>:
        return relu(matmul(x, self.W1))
";
        let (ast, interner) = parse_source(src);
        let err = lower(&ast, &interner).expect_err("missing bias should fail");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }

    #[test]
    fn dtype_aliases_resolve_in_v1_mlp() {
        // M57.1 cross-test: the v1 MLP source uses i8/i32/i64 short-form (Phase 1 §3.1).
        // Confirms parse + lower pipeline accepts the short-form in tensor type
        // annotations and produces the expected 3 KIR ops for a 1-layer MLP.
        let src = "\
model TinyMlp:
    W1: Tensor<[784, 128], i8>
    b1: Tensor<[128], i32>

    fn forward(self, x: Tensor<[1, 784], i8>) -> Tensor<[1, 128], i32>:
        return relu(matmul(x, self.W1) + self.b1)
";
        let (ast, interner) = parse_source(src);
        let kir = lower(&ast, &interner).expect("v1 MLP w/ i8 aliases must lower");
        let n_ops = kir.ops().count();
        assert_eq!(
            n_ops, 3,
            "1-layer MLP = 3 KIR ops (Matmul + ElementwiseAdd + Relu)"
        );
    }

    #[test]
    fn layer_threading_relu_output_feeds_next_matmul_input() {
        // M57.1 §3.3: per-layer threading invariant — layer i's Relu output var
        // must be exactly layer i+1's Matmul input var (SSA continuity).
        let src = "\
model TwoLayer:
    W1: Tensor<[10, 20], i8>
    b1: Tensor<[20], i32>
    W2: Tensor<[20, 5], i32>
    b2: Tensor<[5], i64>

    fn forward(self, x: Tensor<[1, 10], i8>) -> Tensor<[1, 5], i64>:
        let h = relu(matmul(x, self.W1) + self.b1)
        return relu(matmul(h, self.W2) + self.b2)
";
        let (ast, interner) = parse_source(src);
        let kir = lower(&ast, &interner).expect("two-layer MLP must lower");

        // Walk KIR ops in order. Find layer-1 Relu output (ops[2].out) and
        // assert it equals layer-2 Matmul input (ops[3].a).
        let ops: Vec<&KirOp> = kir.ops().collect();
        assert_eq!(ops.len(), 6, "2 layers x 3 ops = 6 KIR ops total");

        let layer1_relu_out = match ops[2] {
            KirOp::Relu { out, .. } => *out,
            other => panic!("ops[2] should be Relu, got {other:?}"),
        };
        let layer2_matmul_in = match ops[3] {
            KirOp::Matmul { a, .. } => *a,
            other => panic!("ops[3] should be Matmul, got {other:?}"),
        };
        assert_eq!(
            layer1_relu_out, layer2_matmul_in,
            "layer 1 Relu.out VarId must thread into layer 2 Matmul.a (SSA continuity)"
        );
    }

    #[test]
    fn rejects_non_relu_activation() {
        // M57.1 §3.3 v1 recognizer only accepts ReLU as the activation function.
        // sigmoid, tanh, etc. must error with UnsupportedV1Shape.
        let src = "\
model BadActivation:
    W1: Tensor<[10, 5], i8>
    b1: Tensor<[5], i32>

    fn forward(self, x: Tensor<[1, 10], i8>) -> Tensor<[1, 5], i32>:
        return sigmoid(matmul(x, self.W1) + self.b1)
";
        let (ast, interner) = parse_source(src);
        let err = lower(&ast, &interner).expect_err("sigmoid should not be recognized");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }

    #[test]
    fn rejects_matmul_with_wrong_arg_count() {
        // matmul must be a 2-arg call (matmul(x, W)). 1-arg matmul must error.
        let src = "\
model BadMatmulArity:
    W1: Tensor<[10, 5], i8>
    b1: Tensor<[5], i32>

    fn forward(self, x: Tensor<[1, 10], i8>) -> Tensor<[1, 5], i32>:
        return relu(matmul(x) + self.b1)
";
        let (ast, interner) = parse_source(src);
        let err = lower(&ast, &interner).expect_err("1-arg matmul should fail");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }
}
