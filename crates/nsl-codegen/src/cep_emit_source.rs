//! CEP Option 2 — SP2: pruned-model source emission (pure text rewrite over the AST).
//!
//! Given the original NSL source + the same `PruneDelta` SP1 consumes, emit a rewritten
//! NSL source whose `(n_heads, n_kv_heads, d_ff)` constructor args reflect the chosen
//! surviving values. Pure: no I/O. v1 supports the **uniform-delta** case only —
//! heterogeneous per-layer deltas refuse with a "deferred to SP3" error.

use nsl_ast::expr::{Arg, Expr, ExprKind};
use nsl_ast::stmt::StmtKind;
use nsl_ast::{Module, Symbol};

use crate::cep_oracle::ModelSpec;
use crate::cep_rewrite::PruneDelta;

type Resolve<'a> = &'a dyn Fn(Symbol) -> String;

/// All surviving layers must agree on these three values for SP2 v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformPrunedSpec {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_ff: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CepEmitError {
    HeterogeneousDelta { detail: String },
    UnrewritableExpr { dim: &'static str, detail: String },
    MissingArg { what: &'static str },
    SpecMismatch { detail: String },
}

impl std::fmt::Display for CepEmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CepEmitError::HeterogeneousDelta { detail } => write!(
                f,
                "CEP emit: heterogeneous per-layer delta cannot be emitted by SP2 v1\n  detail: {detail}\n  (per-layer block-variant emission is deferred to SP3)"
            ),
            CepEmitError::UnrewritableExpr { dim, detail } => write!(
                f,
                "CEP emit: cannot rewrite '{dim}' arg\n  detail: {detail}\n  (expected an int literal or an ident bound to a top-level `const NAME = <int>`)"
            ),
            CepEmitError::MissingArg { what } => write!(
                f,
                "CEP emit: block constructor is missing positional arg: {what}"
            ),
            CepEmitError::SpecMismatch { detail } => write!(
                f,
                "CEP emit: source disagrees with ModelSpec: {detail}"
            ),
        }
    }
}

/// Derive the surviving (n_heads, n_kv_heads, d_ff) implied by `delta`, requiring all
/// surviving layers to agree. The delta is GQA-group-aligned (produced by
/// `plan_to_prune_delta`), so for each layer the number of dropped KV groups is
/// `pruned_heads.len() / group` where `group = baseline.n_heads[l] / baseline.n_kv_heads[l]`.
pub fn derive_uniform_pruned_spec(
    baseline: &ModelSpec,
    delta: &PruneDelta,
) -> Result<UniformPrunedSpec, CepEmitError> {
    if baseline.n_layers == 0 {
        return Err(CepEmitError::SpecMismatch {
            detail: "baseline has zero layers".into(),
        });
    }

    let by_layer: std::collections::HashMap<u32, &crate::cep_rewrite::LayerDelta> =
        delta.per_layer.iter().map(|ld| (ld.layer, ld)).collect();

    let mut first: Option<UniformPrunedSpec> = None;
    for l in 0..baseline.n_layers as usize {
        let base_h = baseline.n_heads[l];
        let base_kv = baseline.n_kv_heads[l].max(1);
        let group = (base_h / base_kv).max(1);

        let ld = by_layer.get(&(l as u32));
        let pruned_heads = ld.map(|d| d.pruned_heads.len() as u32).unwrap_or(0);
        let new_d_ff = ld.and_then(|d| d.new_d_ff).unwrap_or(baseline.d_ff[l]);

        if !pruned_heads.is_multiple_of(group) {
            return Err(CepEmitError::SpecMismatch {
                detail: format!(
                    "layer {l}: pruned_heads.len() = {pruned_heads} is not a multiple of GQA group {group}"
                ),
            });
        }
        let n_heads = base_h - pruned_heads;
        let n_kv_heads = base_kv - (pruned_heads / group);
        let triple = UniformPrunedSpec { n_heads, n_kv_heads, d_ff: new_d_ff };

        match &first {
            None => first = Some(triple),
            Some(prev) if prev == &triple => {}
            Some(prev) => {
                return Err(CepEmitError::HeterogeneousDelta {
                    detail: format!(
                        "layer 0 -> (n_heads={}, n_kv_heads={}, d_ff={}); layer {l} -> (n_heads={}, n_kv_heads={}, d_ff={})",
                        prev.n_heads, prev.n_kv_heads, prev.d_ff,
                        triple.n_heads, triple.n_kv_heads, triple.d_ff,
                    ),
                });
            }
        }
    }
    Ok(first.expect("non-zero layers checked"))
}

/// Find the block model's positional parameter index whose declared NAME matches `param_name`.
/// Returns None if absent. Robust to constructor parameter reordering across NSL versions.
fn block_param_index(
    block_def: &nsl_ast::decl::ModelDef,
    resolve: Resolve,
    param_name: &str,
) -> Option<usize> {
    block_def
        .params
        .iter()
        .position(|p| resolve(p.name) == param_name)
}

/// Locate the top-level `const NAME = <IntLiteral>` initializer expr by name. Returns None
/// when no matching const decl exists or the initializer isn't a plain int literal.
fn find_int_const_init<'a>(
    module: &'a Module,
    resolve: Resolve,
    name: &str,
) -> Option<&'a Expr> {
    for stmt in &module.stmts {
        if let StmtKind::VarDecl { is_const: true, pattern, value: Some(init), .. } = &stmt.kind {
            if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                if resolve(*sym) == name {
                    if matches!(init.kind, ExprKind::IntLiteral(_)) {
                        return Some(init);
                    }
                    return None;
                }
            }
        }
    }
    None
}

/// Compute one `(start_byte, end_byte, replacement)` edit for a single ctor positional arg.
/// `IntLiteral(n)` -> rewrite the arg span. `Ident(sym)` -> walk to its top-level int const
/// init and rewrite that span. Anything else -> refuse.
fn edit_for_arg(
    module: &Module,
    resolve: Resolve,
    dim: &'static str,
    arg: &Expr,
    expected_baseline: u32,
    new_val: u32,
) -> Result<(usize, usize, String), CepEmitError> {
    match &arg.kind {
        ExprKind::IntLiteral(n) => {
            if *n as u32 != expected_baseline {
                return Err(CepEmitError::SpecMismatch {
                    detail: format!(
                        "ctor '{dim}' arg = {n}, but baseline.{dim} = {expected_baseline}"
                    ),
                });
            }
            Ok((arg.span.start.0 as usize, arg.span.end.0 as usize, new_val.to_string()))
        }
        ExprKind::Ident(sym) => {
            let name = resolve(*sym);
            let init = find_int_const_init(module, resolve, &name).ok_or(
                CepEmitError::UnrewritableExpr {
                    dim,
                    detail: format!("ident '{name}' is not a top-level `const NAME = <int literal>`"),
                },
            )?;
            if let ExprKind::IntLiteral(n) = init.kind {
                if n as u32 != expected_baseline {
                    return Err(CepEmitError::SpecMismatch {
                        detail: format!(
                            "const '{name}' = {n}, but baseline.{dim} = {expected_baseline}"
                        ),
                    });
                }
                Ok((init.span.start.0 as usize, init.span.end.0 as usize, new_val.to_string()))
            } else {
                Err(CepEmitError::UnrewritableExpr {
                    dim,
                    detail: format!("const '{name}' init is not an int literal"),
                })
            }
        }
        _ => Err(CepEmitError::UnrewritableExpr {
            dim,
            detail: "expected IntLiteral or Ident, found a more complex expression".to_string(),
        }),
    }
}

/// Rewrite the NSL source so the block constructor call's positional args
/// `n_heads`, `n_kv_heads`, `d_ff` carry the chosen uniform pruned values.
pub fn apply_prune_delta_to_source(
    source: &str,
    module: &Module,
    resolve: Resolve,
    baseline: &ModelSpec,
    delta: &PruneDelta,
) -> Result<String, CepEmitError> {
    let pruned = derive_uniform_pruned_spec(baseline, delta)?;

    // Locate the top model + its block ctor call site (reuse cep_extract).
    let (_top_md, block_type, _n_layers, block_init) =
        crate::cep_extract::find_top_model(module, resolve).ok_or(CepEmitError::SpecMismatch {
            detail: "no top-level model with a `blocks: [Block; N]` field".to_string(),
        })?;

    // The block model definition (for parameter-name lookup).
    let block_def = module.stmts.iter().find_map(|s| match &s.kind {
        StmtKind::ModelDef(md) if resolve(md.name) == block_type => Some(md),
        StmtKind::Decorated { stmt, .. } => match &stmt.kind {
            StmtKind::ModelDef(md) if resolve(md.name) == block_type => Some(md),
            _ => None,
        },
        _ => None,
    }).ok_or(CepEmitError::SpecMismatch {
        detail: format!("block model '{block_type}' not found"),
    })?;

    // Positional args at the block ctor call site.
    let block_args: &[Arg] = match &block_init.kind {
        ExprKind::Call { callee, args } => match &callee.kind {
            ExprKind::Ident(_) => args.as_slice(),
            _ => return Err(CepEmitError::SpecMismatch {
                detail: "block initializer is not a simple identifier call".to_string(),
            }),
        },
        _ => return Err(CepEmitError::SpecMismatch {
            detail: "block initializer is not a constructor call".to_string(),
        }),
    };

    // For each of the three slots, find the positional index by NAME, then the arg expr,
    // then either rewrite the arg literal in place or (if Ident) rewrite the const decl.
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    for (dim, baseline_val, new_val) in [
        ("n_heads", baseline.n_heads[0], pruned.n_heads),
        ("n_kv_heads", baseline.n_kv_heads[0], pruned.n_kv_heads),
        ("d_ff", baseline.d_ff[0], pruned.d_ff),
    ] {
        // No-op if the value isn't changing (preserves bytes; also handles empty-delta case).
        if baseline_val == new_val {
            continue;
        }
        let idx = block_param_index(block_def, resolve, dim).ok_or(CepEmitError::MissingArg {
            what: dim,
        })?;
        let arg: &Arg = block_args.get(idx).ok_or(CepEmitError::MissingArg { what: dim })?;
        edits.push(edit_for_arg(module, resolve, dim, &arg.value, baseline_val, new_val)?);
    }

    // Dedupe (same const may back multiple slots) then apply right-to-left.
    edits.sort_by_key(|e| std::cmp::Reverse(e.0));
    edits.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut out = source.to_string();
    for (start, end, replacement) in edits {
        if end > out.len() || start > end {
            return Err(CepEmitError::SpecMismatch {
                detail: format!("span [{start}, {end}) out of source bounds (len {})", out.len()),
            });
        }
        out.replace_range(start..end, &replacement);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_oracle::{Activation, NormType};
    use crate::cep_rewrite::{LayerDelta, PruneDelta};

    fn parse(src: &str) -> (nsl_ast::Module, nsl_lexer::Interner) {
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _d) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        (parsed.module, interner)
    }

    fn baseline_spec_2layer() -> ModelSpec {
        ModelSpec {
            d_model: 64, n_layers: 2,
            n_heads: vec![4, 4], n_kv_heads: vec![2, 2],
            head_dim: vec![16, 16], d_ff: vec![128, 128],
            vocab: 256, max_seq: 512, batch: 1,
            activation: Activation::SwiGlu, norm: NormType::RmsNorm,
            dtype_bytes: 4,
        }
    }

    #[test]
    fn derive_uniform_passes_when_all_layers_agree() {
        let baseline = baseline_spec_2layer();
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta { layer: 0, pruned_heads: vec![0, 1], new_d_ff: Some(64), drop_layer: false },
                LayerDelta { layer: 1, pruned_heads: vec![0, 1], new_d_ff: Some(64), drop_layer: false },
            ],
        };
        let u = derive_uniform_pruned_spec(&baseline, &delta).unwrap();
        assert_eq!(u, UniformPrunedSpec { n_heads: 2, n_kv_heads: 1, d_ff: 64 });
    }

    #[test]
    fn derive_uniform_refuses_when_layers_disagree() {
        let baseline = baseline_spec_2layer();
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta { layer: 0, pruned_heads: vec![0, 1], new_d_ff: Some(64), drop_layer: false },
                LayerDelta { layer: 1, pruned_heads: vec![], new_d_ff: None, drop_layer: false },
            ],
        };
        let err = derive_uniform_pruned_spec(&baseline, &delta).unwrap_err();
        assert!(matches!(err, CepEmitError::HeterogeneousDelta { .. }), "got: {err}");
        let msg = err.to_string();
        assert!(msg.contains("SP3"), "error message must mention SP3");
    }

    #[test]
    fn derive_uniform_zero_layers_returns_baseline_uniform() {
        let baseline = baseline_spec_2layer();
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta { layer: 0, pruned_heads: vec![], new_d_ff: None, drop_layer: false },
                LayerDelta { layer: 1, pruned_heads: vec![], new_d_ff: None, drop_layer: false },
            ],
        };
        let u = derive_uniform_pruned_spec(&baseline, &delta).unwrap();
        assert_eq!(u, UniformPrunedSpec { n_heads: 4, n_kv_heads: 2, d_ff: 128 });
    }

    const CANONICAL: &str = r#"
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
    fn forward(self, x: Tensor) -> Tensor:
        return silu(x @ self.w_gate)
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn_norm: RMSNorm = RMSNorm(d_model)
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
model TinyCoder:
    embed: Tensor = randn([4096, 384]) * full([1], 0.02)
    blocks: [TransformerBlock; 6] = TransformerBlock(384, 6, 3, 1024, 0.1)
    norm: RMSNorm = RMSNorm(384)
"#;

    fn baseline_canonical() -> ModelSpec {
        ModelSpec {
            d_model: 384, n_layers: 6,
            n_heads: vec![6; 6], n_kv_heads: vec![3; 6],
            head_dim: vec![64; 6], d_ff: vec![1024; 6],
            vocab: 4096, max_seq: 2048, batch: 1,
            activation: Activation::SwiGlu, norm: NormType::RmsNorm,
            dtype_bytes: 4,
        }
    }

    #[test]
    fn direct_literal_round_trip() {
        let (module, interner) = parse(CANONICAL);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let baseline = baseline_canonical();

        let delta = PruneDelta {
            per_layer: (0..6).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![0, 1],
                new_d_ff: Some(512), drop_layer: false,
            }).collect(),
        };

        let out = apply_prune_delta_to_source(CANONICAL, &module, &resolve, &baseline, &delta)
            .expect("rewrite ok");

        let (module2, interner2) = parse(&out);
        let resolve2 = |s: Symbol| interner2.resolve(s.0).unwrap_or("").to_string();
        let spec2 = crate::cep_extract::extract_model_spec(&module2, &resolve2).expect("re-extract");
        assert_eq!(spec2.n_heads, vec![4; 6], "n_heads (6 - 2 pruned)");
        assert_eq!(spec2.n_kv_heads, vec![2; 6], "n_kv_heads (3 - 1 group)");
        assert_eq!(spec2.d_ff, vec![512; 6], "d_ff halved");
        assert_eq!(spec2.d_model, 384);
        assert_eq!(spec2.n_layers, 6);
        assert_eq!(spec2.vocab, 4096);

        assert!(out.contains("blocks: [TransformerBlock; 6] = TransformerBlock(384, 4, 2, 512, 0.1)"),
                "rewritten ctor call must match expected form, got:\n{out}");
        assert!(out.contains("randn([4096, 384])"),
                "embed shape must not be touched");
    }

    #[test]
    fn empty_delta_is_identity() {
        let (module, interner) = parse(CANONICAL);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let baseline = baseline_canonical();

        let delta = PruneDelta {
            per_layer: (0..6).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![], new_d_ff: None, drop_layer: false,
            }).collect(),
        };
        let out = apply_prune_delta_to_source(CANONICAL, &module, &resolve, &baseline, &delta).unwrap();
        assert_eq!(out, CANONICAL, "empty delta must not change a single byte");
    }

    const CONST_BOUND: &str = r#"
const D_MODEL = 256
const N_HEADS = 8
const N_KV_HEADS = 4
const D_FF = 512
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model ConstNet:
    embed: Tensor = randn([1000, 256])
    blocks: [TransformerBlock; 2] = TransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS, D_FF, 0.1)
"#;

    fn baseline_const_bound() -> ModelSpec {
        ModelSpec {
            d_model: 256, n_layers: 2,
            n_heads: vec![8; 2], n_kv_heads: vec![4; 2],
            head_dim: vec![32; 2], d_ff: vec![512; 2],
            vocab: 1000, max_seq: 2048, batch: 1,
            activation: Activation::SwiGlu, norm: NormType::RmsNorm,
            dtype_bytes: 4,
        }
    }

    #[test]
    fn const_bound_round_trip_rewrites_const_decls_not_call_site() {
        let (module, interner) = parse(CONST_BOUND);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let baseline = baseline_const_bound();
        let delta = PruneDelta {
            per_layer: (0..2).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![0, 1, 2, 3],
                new_d_ff: Some(256), drop_layer: false,
            }).collect(),
        };
        let out = apply_prune_delta_to_source(CONST_BOUND, &module, &resolve, &baseline, &delta)
            .expect("rewrite ok");

        let (module2, interner2) = parse(&out);
        let resolve2 = |s: Symbol| interner2.resolve(s.0).unwrap_or("").to_string();
        let spec2 = crate::cep_extract::extract_model_spec(&module2, &resolve2).unwrap();
        assert_eq!(spec2.n_heads, vec![4; 2]);
        assert_eq!(spec2.n_kv_heads, vec![2; 2]);
        assert_eq!(spec2.d_ff, vec![256; 2]);

        assert!(out.contains("const N_HEADS = 4"), "N_HEADS const must be rewritten, got:\n{out}");
        assert!(out.contains("const N_KV_HEADS = 2"), "N_KV_HEADS const must be rewritten");
        assert!(out.contains("const D_FF = 256"), "D_FF const must be rewritten");
        assert!(out.contains("TransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS, D_FF, 0.1)"),
                "call site must keep referring to consts by name");
        assert!(out.contains("const D_MODEL = 256"), "D_MODEL must not be rewritten");
    }

    const ARITHMETIC_ARG: &str = r#"
const N_HEADS = 8
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model Net:
    embed: Tensor = randn([1000, 256])
    blocks: [TransformerBlock; 2] = TransformerBlock(256, N_HEADS * 2, 4, 512, 0.1)
"#;

    #[test]
    fn arithmetic_arg_refuses() {
        let (module, interner) = parse(ARITHMETIC_ARG);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let baseline = ModelSpec {
            d_model: 256, n_layers: 2,
            n_heads: vec![16; 2], n_kv_heads: vec![4; 2],
            head_dim: vec![16; 2], d_ff: vec![512; 2],
            vocab: 1000, max_seq: 2048, batch: 1,
            activation: Activation::SwiGlu, norm: NormType::RmsNorm,
            dtype_bytes: 4,
        };
        let delta = PruneDelta {
            per_layer: (0..2).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![0, 1, 2, 3],
                new_d_ff: None, drop_layer: false,
            }).collect(),
        };
        let err = apply_prune_delta_to_source(ARITHMETIC_ARG, &module, &resolve, &baseline, &delta)
            .unwrap_err();
        assert!(matches!(err, CepEmitError::UnrewritableExpr { dim: "n_heads", .. }),
                "got: {err}");
    }

    const MISSING_ARG: &str = r#"
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, 4, 2, 0.1)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, 128, 0.1)
    norm: RMSNorm = RMSNorm(d_model)
model Net:
    embed: Tensor = randn([1000, 256])
    blocks: [TransformerBlock; 2] = TransformerBlock(256)
"#;

    #[test]
    fn missing_param_refuses() {
        let (module, interner) = parse(MISSING_ARG);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let baseline = ModelSpec {
            d_model: 256, n_layers: 2,
            n_heads: vec![4; 2], n_kv_heads: vec![2; 2],
            head_dim: vec![64; 2], d_ff: vec![128; 2],
            vocab: 1000, max_seq: 2048, batch: 1,
            activation: Activation::SwiGlu, norm: NormType::RmsNorm,
            dtype_bytes: 4,
        };
        let delta = PruneDelta {
            per_layer: (0..2).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![0, 1],
                new_d_ff: Some(64), drop_layer: false,
            }).collect(),
        };
        let err = apply_prune_delta_to_source(MISSING_ARG, &module, &resolve, &baseline, &delta)
            .unwrap_err();
        assert!(matches!(err, CepEmitError::MissingArg { what: "n_heads" }), "got: {err}");
    }

    #[test]
    fn spec_mismatch_refuses_literal_with_wrong_value() {
        let (module, interner) = parse(CANONICAL);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let mut baseline = baseline_canonical();
        baseline.n_heads = vec![8; 6];
        let delta = PruneDelta {
            per_layer: (0..6).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![0, 1],
                new_d_ff: None, drop_layer: false,
            }).collect(),
        };
        let err = apply_prune_delta_to_source(CANONICAL, &module, &resolve, &baseline, &delta)
            .unwrap_err();
        assert!(matches!(err, CepEmitError::SpecMismatch { .. }), "got: {err}");
    }

    /// SP2's emitted source extracts the chosen pruned (n_heads, n_kv_heads, d_ff) triple.
    /// d_model is invariant (CEP design).
    ///
    /// KNOWN LIMITATION (deferred): the recognizer derives `head_dim = d_model / n_heads`
    /// while SP1 slices using the BASELINE head_dim. After SP2 rewrites n_heads, the
    /// recognizer's head_dim therefore disagrees with SP1's sliced projection width
    /// (e.g. for baseline d_model=384/n_heads=6 -> hd=64; rewrite n_heads=4 -> hd=96
    /// per the recognizer but the sliced wq has 4*64=256 cols not 4*96=384). This is a
    /// stdlib GQA limitation (head_dim is implicit, not a parameter), not an SP2 bug;
    /// fixing it requires either making head_dim an explicit GQA arg or teaching the
    /// recognizer to source head_dim from weight shapes. Out of SP2 v1 scope.
    #[test]
    fn sp2_emitted_source_extracts_chosen_dim_triple() {
        let (module, interner) = parse(CANONICAL);
        let resolve = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let baseline = baseline_canonical();
        let delta = PruneDelta {
            per_layer: (0..6).map(|l| LayerDelta {
                layer: l, pruned_heads: vec![0, 1],
                new_d_ff: Some(512), drop_layer: false,
            }).collect(),
        };
        let uniform = derive_uniform_pruned_spec(&baseline, &delta).unwrap();

        let out = apply_prune_delta_to_source(CANONICAL, &module, &resolve, &baseline, &delta).unwrap();
        let (module2, interner2) = parse(&out);
        let resolve2 = |s: Symbol| interner2.resolve(s.0).unwrap_or("").to_string();
        let spec2 = crate::cep_extract::extract_model_spec(&module2, &resolve2).unwrap();

        // d_model invariant
        assert_eq!(spec2.d_model, baseline.d_model);
        assert_eq!(spec2.n_layers, baseline.n_layers);
        // The chosen dim triple round-trips identically.
        assert_eq!(spec2.n_heads[0], uniform.n_heads);
        assert_eq!(spec2.n_kv_heads[0], uniform.n_kv_heads);
        assert_eq!(spec2.d_ff[0], uniform.d_ff);
    }
}
