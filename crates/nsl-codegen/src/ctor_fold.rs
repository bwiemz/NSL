//! Item 5 — constructor-argument constant folding for model field dims.
//!
//! `collection.rs` derives `model_field_dims` from LITERAL initializers only:
//! `embed: Tensor = randn([49152, 512]) * full([1], 0.02)` works, but every
//! weight inside a parameterized module — `w_gate: Tensor = randn([d_model,
//! d_ff]) * ...` in `model SwiGLUFFN(d_model: int, d_ff: int, ...)` — is
//! structurally underivable there, because the dims are constructor
//! parameters. On coder50m that is ALL of the block weights, which left the
//! arena's shape propagation with nothing to hang the forward chain on
//! (measured: 4 of 1321 transients placed, the forward stalling at 57 unsized
//! matmuls whose weights had no dims).
//!
//! Every shipped model instantiates its submodules with COMPILE-TIME-CONSTANT
//! arguments (`blocks: [TransformerBlock; 8] = TransformerBlock(512, 8, 4,
//! 64, 1408, 0.1, 1024, 10000.0, 8)`), so the dims are recoverable by
//! walking the instantiation tree from every no-parameter root model,
//! substituting the argument values into each child's field initializers,
//! and evaluating the dim expressions over integer arithmetic.
//!
//! Soundness rules, same posture as `unique_field_dims`:
//! * a dim that does not evaluate to a positive integer derives nothing;
//! * a (type, field) reached twice with DIFFERENT dims (two instantiations
//!   of one module type with different sizes in one compile unit) is
//!   dropped and the bare field name is reported so the caller can veto it —
//!   `model_field_dims` is keyed by TYPE, so per-type dims are only sound
//!   when every instantiation agrees;
//! * a cycle in the instantiation graph stops (models cannot contain
//!   themselves at runtime, so a textual cycle is a broken program, not a
//!   recursion to chase);
//! * named or non-constant constructor arguments stop that edge.

use std::collections::{HashMap, HashSet};

use nsl_ast::decl::ModelMember;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::operator::{BinOp, UnaryOp};
use nsl_ast::Module;
use nsl_lexer::Interner;

/// Per-type dims recovered by folding, plus the bare field names whose dims
/// conflicted across instantiations (the caller must veto those).
#[derive(Debug, Default)]
pub struct FoldedCtorDims {
    pub dims: HashMap<String, HashMap<String, Vec<i64>>>,
    /// VALUES of 1-element `full([1], <const>)` config fields (`_n_heads =
    /// full([1], float(n_heads))`). The attention stack reads its head
    /// counts back out of these at runtime (`int(self._n_heads.item())`)
    /// and builds every reshape target from them — so without the values,
    /// shape propagation dies at the first `q.reshape([b, s, nh, hd])`.
    /// A conflicting value across instantiations drops the entry.
    pub values: HashMap<String, HashMap<String, f64>>,
    pub conflicted: HashSet<String>,
    /// `"type.field"` tombstones so a value dropped for conflicting cannot
    /// be re-proposed by a third agreeing instantiation.
    values_conflicted: HashSet<String>,
}

impl FoldedCtorDims {
    /// Field-level union into an imported-dims map. A field the target
    /// already has with DIFFERENT dims is removed and its bare name vetoed
    /// — two sources disagreeing is the same soundness problem as two
    /// instantiations disagreeing.
    pub fn merge_into(
        self,
        dims: &mut HashMap<String, HashMap<String, Vec<i64>>>,
        values: &mut HashMap<String, HashMap<String, f64>>,
        veto: &mut HashSet<String>,
    ) {
        veto.extend(self.conflicted);
        for (ty, fields) in self.dims {
            let entry = dims.entry(ty).or_default();
            for (f, d) in fields {
                match entry.get(&f) {
                    None => {
                        entry.insert(f, d);
                    }
                    Some(prev) if *prev == d => {}
                    Some(_) => {
                        entry.remove(&f);
                        veto.insert(f);
                    }
                }
            }
        }
        for (ty, fields) in self.values {
            let entry = values.entry(ty).or_default();
            for (f, v) in fields {
                match entry.get(&f) {
                    None => {
                        entry.insert(f, v);
                    }
                    Some(prev) if *prev == v => {}
                    // A conflicting VALUE only loses the value; the veto
                    // set governs dims, not values.
                    Some(_) => {
                        entry.remove(&f);
                    }
                }
            }
        }
    }
}

/// A constant a constructor argument can evaluate to.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConstVal {
    I(i64),
    F(f64),
}

/// The tensor-init builtins whose first argument is the dim list.
const INIT_CALLEES: &[&str] = &["randn", "rand", "zeros", "ones", "full"];

/// The scalar cast builtins that appear inside constructor-argument and
/// dim arithmetic (`float(d_model + d_ff)`, `int(x)`).
const CAST_CALLEES: &[&str] = &["float", "int"];

struct ModelInfo<'a> {
    params: Vec<String>,
    members: &'a [ModelMember],
}

/// Fold constructor arguments through every model instantiation reachable
/// from a no-parameter root, over ALL modules of the compile unit (the
/// definitions and their instantiations routinely live in different files —
/// stdlib modules instantiated by `model.nsl`, imported by the entry).
pub fn fold_constructor_dims(modules: &[&Module], interner: &Interner) -> FoldedCtorDims {
    let mut defs: HashMap<String, ModelInfo> = HashMap::new();
    for module in modules {
        for stmt in &module.stmts {
            if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                let Some(name) = interner.resolve(md.name.0) else {
                    continue;
                };
                let params: Vec<String> = md
                    .params
                    .iter()
                    .filter_map(|p| interner.resolve(p.name.0).map(str::to_string))
                    .collect();
                if params.len() != md.params.len() {
                    continue; // an unresolvable param name poisons the def
                }
                // A duplicate model name across modules is already an
                // ambiguity; keep the first and let conflict detection
                // catch any disagreement through the instantiations.
                defs.entry(name.to_string()).or_insert(ModelInfo {
                    params,
                    members: &md.members,
                });
            }
        }
    }

    let mut out = FoldedCtorDims::default();
    let mut visiting: Vec<String> = Vec::new();
    let mut roots: Vec<String> = defs
        .iter()
        .filter(|(_, info)| info.params.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    roots.sort(); // deterministic traversal -> deterministic conflicts
    for root in roots {
        fold_type(&root, &HashMap::new(), &defs, interner, &mut out, &mut visiting);
    }
    out
}

fn fold_type(
    type_name: &str,
    env: &HashMap<String, ConstVal>,
    defs: &HashMap<String, ModelInfo>,
    interner: &Interner,
    out: &mut FoldedCtorDims,
    visiting: &mut Vec<String>,
) {
    if visiting.iter().any(|t| t == type_name) {
        return; // cycle — a broken program, not a recursion to chase
    }
    let Some(info) = defs.get(type_name) else {
        return;
    };
    visiting.push(type_name.to_string());
    for member in info.members {
        let ModelMember::LayerDecl { name, init: Some(init), .. } = member else {
            continue;
        };
        let Some(field_name) = interner.resolve(name.0) else {
            continue;
        };
        // A tensor initializer whose dims evaluate: propose (type, field).
        if let Some(dim_exprs) = tensor_init_dims(init, interner) {
            let dims: Option<Vec<i64>> = dim_exprs
                .iter()
                .map(|e| match eval_const(e, env, interner) {
                    Some(ConstVal::I(n)) if n > 0 => Some(n),
                    _ => None,
                })
                .collect();
            if let Some(dims) = dims {
                let one_elem = dims == [1];
                propose(out, type_name, field_name, dims);
                // `full([1], <const>)` config fields also carry a VALUE the
                // model reads back at runtime (`int(self._n_heads.item())`).
                if one_elem {
                    if let Some(v) = full_scalar_value(init, env, interner) {
                        propose_value(out, type_name, field_name, v);
                    }
                }
            }
            continue;
        }
        // A submodule instantiation: evaluate the args, recurse.
        if let Some((child, child_env)) = instantiation(init, env, defs, interner) {
            fold_type(&child, &child_env, defs, interner, out, visiting);
        }
    }
    visiting.pop();
}

/// The scalar VALUE of a direct `full([1], <expr>)` initializer, when the
/// expression folds. Deliberately not routed through the `*`-peel: config
/// fields are written as the bare call, and a scaled `full([1], a) * b`
/// would need value multiplication nothing currently reads.
fn full_scalar_value(
    init: &Expr,
    env: &HashMap<String, ConstVal>,
    interner: &Interner,
) -> Option<f64> {
    let ExprKind::Call { callee, args } = &init.kind else {
        return None;
    };
    let ExprKind::Ident(sym) = &callee.kind else {
        return None;
    };
    if interner.resolve(sym.0)? != "full" || args.len() < 2 || args[1].name.is_some() {
        return None;
    }
    Some(as_f64(eval_const(&args[1].value, env, interner)?))
}

fn propose_value(out: &mut FoldedCtorDims, type_name: &str, field: &str, value: f64) {
    let key = format!("{type_name}.{field}");
    if out.values_conflicted.contains(&key) {
        return;
    }
    let fields = out.values.entry(type_name.to_string()).or_default();
    match fields.get(field) {
        None => {
            fields.insert(field.to_string(), value);
        }
        Some(prev) if *prev == value => {}
        Some(_) => {
            fields.remove(field);
            out.values_conflicted.insert(key);
        }
    }
}

fn propose(out: &mut FoldedCtorDims, type_name: &str, field: &str, dims: Vec<i64>) {
    if out.conflicted.contains(field) {
        return;
    }
    let fields = out.dims.entry(type_name.to_string()).or_default();
    match fields.get(field) {
        None => {
            fields.insert(field.to_string(), dims);
        }
        Some(prev) if *prev == dims => {}
        Some(_) => {
            // Two instantiations of this TYPE disagree; per-type dims are
            // then a lie for at least one instance. Drop and veto the NAME
            // (the leaf-name bridges downstream key on it too).
            fields.remove(field);
            out.conflicted.insert(field.to_string());
        }
    }
}

/// Match a `Type(args...)` initializer against a known model def and build
/// the child environment. Named or non-constant args refuse the edge.
fn instantiation(
    init: &Expr,
    env: &HashMap<String, ConstVal>,
    defs: &HashMap<String, ModelInfo>,
    interner: &Interner,
) -> Option<(String, HashMap<String, ConstVal>)> {
    let ExprKind::Call { callee, args } = &init.kind else {
        return None;
    };
    let ExprKind::Ident(sym) = &callee.kind else {
        return None;
    };
    let name = interner.resolve(sym.0)?;
    let info = defs.get(name)?;
    if args.len() != info.params.len() || args.iter().any(|a| a.name.is_some()) {
        return None;
    }
    let mut child_env = HashMap::new();
    for (param, arg) in info.params.iter().zip(args) {
        child_env.insert(param.clone(), eval_const(&arg.value, env, interner)?);
    }
    Some((name.to_string(), child_env))
}

/// The dim-expression list of a tensor initializer: a `randn/rand/zeros/
/// ones/full([d0, d1, ...], ...)` call, possibly scaled by `* <scalar>`
/// factors (the universal init idiom). When BOTH sides of a `*` are tensor
/// inits, the 1-element side (`full([1], ...)`) is the scalar; two real
/// tensor sides refuse.
fn tensor_init_dims<'a>(expr: &'a Expr, interner: &Interner) -> Option<&'a [Expr]> {
    match &expr.kind {
        ExprKind::Paren(inner) => tensor_init_dims(inner, interner),
        ExprKind::BinaryOp { left, op: BinOp::Mul, right } => {
            match (
                tensor_init_dims(left, interner),
                tensor_init_dims(right, interner),
            ) {
                (Some(d), None) => Some(d),
                (None, Some(d)) => Some(d),
                (Some(dl), Some(dr)) => {
                    if is_one_elem(dr) {
                        Some(dl)
                    } else if is_one_elem(dl) {
                        Some(dr)
                    } else {
                        None
                    }
                }
                (None, None) => None,
            }
        }
        ExprKind::Call { callee, args } => {
            let ExprKind::Ident(sym) = &callee.kind else {
                return None;
            };
            let name = interner.resolve(sym.0)?;
            if !INIT_CALLEES.contains(&name) {
                return None;
            }
            let first = args.first()?;
            match &first.value.kind {
                ExprKind::ListLiteral(dims) => Some(dims),
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_one_elem(dims: &[Expr]) -> bool {
    dims.len() == 1 && matches!(dims[0].kind, ExprKind::IntLiteral(1))
}

/// Evaluate a compile-time-constant scalar expression under a constructor
/// environment (param name -> value). Integer arithmetic is exact or
/// refused; `/` with a remainder refuses rather than guessing the
/// language's rounding; `//` floors.
fn eval_const(
    expr: &Expr,
    env: &HashMap<String, ConstVal>,
    interner: &Interner,
) -> Option<ConstVal> {
    use ConstVal::{F, I};
    match &expr.kind {
        ExprKind::IntLiteral(n) => Some(I(*n)),
        ExprKind::FloatLiteral(f) => Some(F(*f)),
        ExprKind::Paren(inner) => eval_const(inner, env, interner),
        ExprKind::UnaryOp { op: UnaryOp::Neg, operand } => {
            match eval_const(operand, env, interner)? {
                I(n) => Some(I(n.checked_neg()?)),
                F(f) => Some(F(-f)),
            }
        }
        ExprKind::BinaryOp { left, op, right } => {
            let l = eval_const(left, env, interner)?;
            let r = eval_const(right, env, interner)?;
            match (l, r) {
                (I(a), I(b)) => match op {
                    BinOp::Add => a.checked_add(b).map(I),
                    BinOp::Sub => a.checked_sub(b).map(I),
                    BinOp::Mul => a.checked_mul(b).map(I),
                    BinOp::Div => {
                        if b != 0 && a % b == 0 {
                            Some(I(a / b))
                        } else {
                            None
                        }
                    }
                    BinOp::FloorDiv => {
                        if b != 0 {
                            Some(I(a.div_euclid(b)))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                (l, r) => {
                    let (a, b) = (as_f64(l), as_f64(r));
                    match op {
                        BinOp::Add => Some(F(a + b)),
                        BinOp::Sub => Some(F(a - b)),
                        BinOp::Mul => Some(F(a * b)),
                        BinOp::Div => Some(F(a / b)),
                        _ => None,
                    }
                }
            }
        }
        ExprKind::Call { callee, args } => {
            let ExprKind::Ident(sym) = &callee.kind else {
                return None;
            };
            let name = interner.resolve(sym.0)?;
            if !CAST_CALLEES.contains(&name) || args.len() != 1 || args[0].name.is_some() {
                return None;
            }
            let v = eval_const(&args[0].value, env, interner)?;
            match name {
                "float" => Some(F(as_f64(v))),
                "int" => match v {
                    I(n) => Some(I(n)),
                    F(f) if f.is_finite() && f.abs() < 9.0e15 => Some(I(f.trunc() as i64)),
                    F(_) => None,
                },
                _ => None,
            }
        }
        ExprKind::Ident(sym) => {
            let name = interner.resolve(sym.0)?;
            env.get(name).copied()
        }
        _ => None,
    }
}

fn as_f64(v: ConstVal) -> f64 {
    match v {
        ConstVal::I(n) => n as f64,
        ConstVal::F(f) => f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::FileId;

    fn fold(src: &str) -> FoldedCtorDims {
        let mut interner = Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex: {lex_diags:?}");
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(parsed.diagnostics.is_empty(), "parse: {:?}", parsed.diagnostics);
        fold_constructor_dims(&[&parsed.module], &interner)
    }

    #[test]
    fn ctor_args_substitute_into_field_dims() {
        let f = fold(
            "model Inner(d: int, f: int):\n    \
                 w: Tensor = randn([d, f]) * full([1], 0.5)\n\n\
             model Outer:\n    \
                 inner: Inner = Inner(512, 1408)\n",
        );
        assert_eq!(f.dims["Inner"]["w"], vec![512, 1408]);
        assert!(f.conflicted.is_empty());
    }

    #[test]
    fn dim_arithmetic_over_params_evaluates() {
        let f = fold(
            "model Attn(d_model: int, n_heads: int, head_dim: int):\n    \
                 wq: Tensor = randn([d_model, n_heads * head_dim])\n\n\
             model Root:\n    \
                 attn: Attn = Attn(512, 8, 64)\n",
        );
        assert_eq!(f.dims["Attn"]["wq"], vec![512, 512]);
    }

    #[test]
    fn nested_instantiation_cascades_the_environment() {
        let f = fold(
            "model Norm(dim: int):\n    \
                 weight: Tensor = ones([dim])\n\n\
             model Block(d: int):\n    \
                 norm: Norm = Norm(d)\n\n\
             model Root:\n    \
                 block: Block = Block(1280)\n",
        );
        assert_eq!(f.dims["Norm"]["weight"], vec![1280]);
    }

    #[test]
    fn conflicting_instantiations_veto_the_field() {
        let f = fold(
            "model Norm(dim: int):\n    \
                 weight: Tensor = ones([dim])\n\n\
             model Root:\n    \
                 a: Norm = Norm(512)\n    \
                 b: Norm = Norm(1280)\n",
        );
        assert!(f.dims.get("Norm").map_or(true, |m| !m.contains_key("weight")));
        assert!(f.conflicted.contains("weight"));
    }

    #[test]
    fn agreeing_instantiations_keep_the_dims() {
        let f = fold(
            "model Norm(dim: int):\n    \
                 weight: Tensor = ones([dim])\n\n\
             model Root:\n    \
                 a: Norm = Norm(512)\n    \
                 b: Norm = Norm(512)\n",
        );
        assert_eq!(f.dims["Norm"]["weight"], vec![512]);
        assert!(f.conflicted.is_empty());
    }

    #[test]
    fn non_constant_args_refuse_the_edge_not_the_world() {
        let f = fold(
            "model Inner(d: int):\n    \
                 w: Tensor = randn([d, 4])\n\n\
             model Root:\n    \
                 lit: Tensor = randn([16, 16])\n    \
                 inner: Inner = Inner(some_runtime_value)\n",
        );
        assert!(f.dims.get("Inner").is_none(), "unresolvable arg derives nothing");
        assert_eq!(f.dims["Root"]["lit"], vec![16, 16]);
    }

    #[test]
    fn scalar_scaled_and_one_elem_full_sides_are_peeled() {
        let f = fold(
            "model M(a: int, b: int, n: int):\n    \
                 w: Tensor = randn([a, b]) * sqrt(full([1], 2.0 / float(a + b))) * full([1], 1.0)\n\n\
             model Root:\n    \
                 m: M = M(1408, 512, 8)\n",
        );
        assert_eq!(f.dims["M"]["w"], vec![1408, 512]);
    }

    #[test]
    fn the_shipped_coder50m_shapes_fold_exactly() {
        // The real three-level chain: NSLCoder -> TransformerBlock ->
        // {RMSNorm, SwiGLUFFN} with the shipped constants.
        let f = fold(
            "model RMSNorm(dim: int):\n    \
                 weight: Tensor = ones([dim])\n\n\
             model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float, n_layers: int):\n    \
                 w_gate: Tensor = randn([d_model, d_ff]) * sqrt(full([1], 2.0 / float(d_model + d_ff)))\n    \
                 w_up: Tensor = randn([d_model, d_ff]) * sqrt(full([1], 2.0 / float(d_model + d_ff)))\n    \
                 _dropout_p: Tensor = full([1], dropout_p)\n\n\
             model TransformerBlock(d_model: int, d_ff: int, dropout_p: float, n_layers: int):\n    \
                 ffn_norm: RMSNorm = RMSNorm(d_model)\n    \
                 ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p, n_layers)\n\n\
             model NSLCoder:\n    \
                 embed: Tensor = randn([49152, 512]) * full([1], 0.02)\n    \
                 blocks: [TransformerBlock; 8] = TransformerBlock(512, 1408, 0.1, 8)\n    \
                 norm: RMSNorm = RMSNorm(512)\n",
        );
        assert_eq!(f.dims["SwiGLUFFN"]["w_gate"], vec![512, 1408]);
        assert_eq!(f.dims["SwiGLUFFN"]["w_up"], vec![512, 1408]);
        assert_eq!(f.dims["SwiGLUFFN"]["_dropout_p"], vec![1]);
        assert_eq!(f.dims["RMSNorm"]["weight"], vec![512]);
        assert_eq!(f.dims["NSLCoder"]["embed"], vec![49152, 512]);
        assert!(f.conflicted.is_empty());
    }
}
