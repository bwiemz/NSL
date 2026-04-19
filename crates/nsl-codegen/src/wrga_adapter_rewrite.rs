//! WRGA Milestone B.2.1 Tasks 3+4: LoRA / IA³ / GatedLoRA forward-pass AST rewrite.
//!
//! Walks each model-method body and replaces every matmul of the form
//! `x @ self.W` — where `W` is a raw `Tensor<...>` field targeted by an
//! active adapter — with the adapter-specific rewritten expression:
//!
//! ```text
//! LoRA:       x @ self.W + ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * (alpha / rank)
//! IA³:        (x @ self.W) * self.ia3_scale_<site>
//! GatedLoRA:  x @ self.W + sigmoid(self.gate_<site>)
//!                          * ((x @ self.lora_A_<site>) @ self.lora_B_<site>)
//!                          * (alpha / rank)
//! ```
//!
//! # Risk #6 — Step-0 invariant for GatedLoRA (LOAD-BEARING)
//!
//! `gate_<site>` is initialized to zeros, and `sigmoid(0) == 0.5` — NOT 0.
//! The gate is HALF-OPEN at step 0. Base-model equivalence at step 0
//! depends ENTIRELY on `lora_B = 0` zeroing the entire adapter contribution.
//! A refactor that changes `lora_B`'s init without simultaneously changing
//! the gate's init (or vice versa) will silently break the equivalence
//! invariant. Task 5 Build 4 is the load-bearing runtime assertion.
//!
//! Synthesized `MemberAccess` nodes use a sentinel `Symbol` (reusing the
//! original `W` field's Symbol) paired with an override entry in
//! `Compiler::synth_member_names` keyed by the synthesized node's
//! `NodeId`. `compile_member_access` consults this override map first
//! before falling back to `resolve_sym`. This avoids the need for mutable
//! interner access during codegen.

use std::collections::HashMap;

use nsl_ast::expr::{Expr, ExprKind, Arg, FStringPart, SubscriptKind, MatchArm, CompGenerator};
use nsl_ast::operator::BinOp;
use nsl_ast::stmt::{Stmt, StmtKind, Block};
use nsl_ast::{NodeId, Symbol};

use crate::wrga_adapter_inject::AdapterSite;
use crate::AdapterKind;

/// Context for a single model-method rewrite pass.
pub struct RewriteContext<'a> {
    /// Sites targeting THIS model (already filtered by target_model).
    pub sites: Vec<&'a AdapterSite>,
    /// Map from the original `W` field name to its interned `Symbol`, so
    /// synthesized `MemberAccess` nodes can reuse a valid sentinel Symbol.
    /// The actual string resolution at codegen time comes from
    /// `synth_member_names`.
    pub field_symbols: HashMap<String, Symbol>,
    /// Function used to mint a fresh `NodeId` for each synthesized node.
    /// `NodeId::next()` is the default; tests may override.
    /// Populated via closure for flexibility, but we default to `NodeId::next`.
    /// Overrides from the rewrite are accumulated here so the caller can
    /// apply them to `Compiler::synth_member_names`.
    pub synth_overrides: HashMap<NodeId, String>,
    /// Resolved `self` symbol from the enclosing method, used to verify that
    /// `MemberAccess { object: Ident(self_sym), .. }` patterns are actually
    /// `self`-rooted.
    pub self_sym: Option<Symbol>,
    /// B.3 Task 4: CUDA sm version resolved from the compile target.
    /// `None` when the target is plain `cuda` (no sm pinned) or non-CUDA.
    /// The fused single-FFI rewrite path activates only when this is
    /// `Some(sm)` with `sm >= 80`.
    pub target_sm: Option<u32>,
    /// B.3 Task 4: overrides emitted for synthesized `ExprKind::Call`
    /// callees.  Keyed by the callee Ident's `NodeId`; value is the real
    /// FFI name the codegen should dispatch to.  Mirrors `synth_overrides`
    /// but for call sites, not member accesses.
    pub synth_call_overrides: HashMap<NodeId, String>,
    /// B.3 Task 5: deterministic ordering of fused PTX kernel keys, used
    /// to assign a stable `kernel_handle` index per call site.  Sorted by
    /// `(m, n, k, rank, target_sm)`.  Empty when the prescan didn't
    /// populate `Compiler::fused_ptx_kernels` (e.g. non-sm_80 target).
    pub fused_kernel_order: Vec<crate::wrga_fused_ptx::LoraKernelKey>,
    /// B.3.1 Task 5.0.c: parallel ordering for GatedLoRA kernels.
    /// GatedLoRA handles = `fused_kernel_order.len() + position_in_this_vec`.
    /// Sorted by the same key tuple as `fused_kernel_order`.
    pub fused_gatedlora_kernel_order: Vec<crate::wrga_fused_ptx::LoraKernelKey>,
}

impl<'a> RewriteContext<'a> {
    pub fn new(sites: Vec<&'a AdapterSite>) -> Self {
        Self {
            sites,
            field_symbols: HashMap::new(),
            synth_overrides: HashMap::new(),
            self_sym: None,
            target_sm: None,
            synth_call_overrides: HashMap::new(),
            fused_kernel_order: Vec::new(),
            fused_gatedlora_kernel_order: Vec::new(),
        }
    }
}

/// Apply the adapter rewrite to a sequence of statements for a given model.
///
/// Shared between `compile_user_functions` (Cranelift path, see
/// `compiler/functions.rs`) and `wrga_prescan::rewrite_model_method_bodies_\
/// with_adapter_sites` (source-AD path, which re-rewrites
/// `compiler.models.model_method_bodies`). Using this single function
/// prevents drift between the two paths — they must see the same rewritten
/// AST or source-AD and Cranelift will diverge silently. See
/// `docs/superpowers/specs/2026-04-19-wrga-b32-option3-revised-design.md`
/// §4 for the rationale.
///
/// Preconditions: `compiler.adapter_sites` is populated (by
/// `prescan_adapter_sites_from_decorators` or the train-block WRGA pass).
/// The rewrite only fires when `adapter_sites` has at least one site with
/// `target_model == model_name`; otherwise returns the input stmts cloned
/// unchanged (cheap no-op).
///
/// Side effects: on a non-empty rewrite this inserts into
/// `compiler.synth_member_names` and `compiler.synth_call_names` so
/// downstream codegen can resolve the synthesized member accesses and call
/// callees.
pub fn rewrite_stmts_for_model(
    compiler: &mut crate::compiler::Compiler<'_>,
    model_name: &str,
    fn_def: &nsl_ast::decl::FnDef,
    stmts: &[Stmt],
) -> Vec<Stmt> {
    // B.2.1 Task 3: run the LoRA forward-pass AST rewrite over each
    // statement before lowering. Only active when adapter sites target
    // this model class.
    let sites_for_model: Vec<&crate::wrga_adapter_inject::AdapterSite> = compiler
        .adapter_sites
        .iter()
        .filter(|s| s.target_model == model_name)
        .collect();
    if sites_for_model.is_empty() {
        return stmts.to_vec();
    }

    let mut ctx = RewriteContext::new(sites_for_model);
    // B.3 Task 4: expose the compile target's sm version so the rewrite
    // can choose between the fused single-FFI path and the B.2.1 unfused
    // triple.
    ctx.target_sm = compiler.target_sm();
    // B.3 Task 5: deterministic ordering of fused PTX kernel keys, used
    // by the rewrite to assign a stable per-site `kernel_handle`. Sort by
    // the full key tuple.
    let mut order: Vec<crate::wrga_fused_ptx::LoraKernelKey> =
        compiler.fused_ptx_kernels.keys().cloned().collect();
    order.sort_by_key(|k| (k.m, k.n, k.k, k.rank, k.target_sm));
    ctx.fused_kernel_order = order;
    // B.3.1 Task 5.0.c: build GatedLoRA kernel order from the parallel
    // map. Handles for GatedLoRA are assigned at lora_count + idx in this
    // vec.
    let mut gl_order: Vec<crate::wrga_fused_ptx::LoraKernelKey> = compiler
        .fused_gatedlora_ptx_kernels
        .keys()
        .cloned()
        .collect();
    gl_order.sort_by_key(|k| (k.m, k.n, k.k, k.rank, k.target_sm));
    ctx.fused_gatedlora_kernel_order = gl_order;
    // Find the `self` Symbol and populate field_symbols from the model's
    // known fields for matcher support.
    ctx.self_sym = fn_def
        .params
        .iter()
        .find(|p| compiler.resolve_sym(p.name) == "self")
        .map(|p| p.name);
    if let Some(field_map) = compiler
        .models
        .model_field_types
        .get(model_name)
        .cloned()
    {
        for fname in field_map.keys() {
            if let Some(s) = compiler.interner.get(fname) {
                ctx.field_symbols
                    .insert(fname.clone(), nsl_ast::Symbol(s));
            }
        }
    }
    // B.2.1 Task 5.5: also include Tensor-typed fields (whose shape
    // strings live in the separate `model_tensor_field_shapes` map) so
    // the rewrite matcher can recognise `self.w @ ...` for adapted tensor
    // fields.
    if let Some(field_map) = compiler
        .models
        .model_tensor_field_shapes
        .get(model_name)
        .cloned()
    {
        for fname in field_map.keys() {
            if let Some(s) = compiler.interner.get(fname) {
                ctx.field_symbols
                    .insert(fname.clone(), nsl_ast::Symbol(s));
            }
        }
    }
    // B.3.1 Task 5.1: GatedLoRA's synthesize_gatedlora_adapted builds a
    // `sigmoid(self.gate_<site>)` Call node and resolves the callee by
    // looking up `field_symbols["sigmoid"]`. Without this insertion the
    // fallback picks an arbitrary field symbol (e.g. `w`) as the callee,
    // emitting `w(gate_...)` which fails at codegen with "undefined
    // variable 'w'".
    //
    // We insert the symbol unconditionally for all models that have
    // adapter sites — it is a no-op if no GatedLoRA site is present
    // since non-GatedLoRA rewrites never consult `field_symbols["sigmoid"]`.
    if let Some(s) = compiler.interner.get("sigmoid") {
        ctx.field_symbols
            .insert("sigmoid".to_string(), nsl_ast::Symbol(s));
    }
    let out: Vec<Stmt> = stmts
        .iter()
        .cloned()
        .map(|s| rewrite_stmt(s, &mut ctx))
        .collect();
    // Commit synth overrides to the compiler so compile_member_access
    // can resolve them.
    compiler.synth_member_names.extend(ctx.synth_overrides);
    // B.3 Task 4: commit fused-call callee overrides so expr_as_func_name
    // can resolve them.
    compiler.synth_call_names.extend(ctx.synth_call_overrides);
    out
}

/// Is the given expression the `self` reference, given the optional
/// known `self_sym` (from the method's `self` parameter)?
fn is_self_expr(expr: &Expr, self_sym: Option<Symbol>) -> bool {
    match &expr.kind {
        ExprKind::SelfRef => true,
        ExprKind::Ident(sym) => match self_sym {
            Some(ss) => *sym == ss,
            None => false,
        },
        _ => false,
    }
}

/// Attempt to match a `BinaryOp MatMul` whose right operand is
/// `self.<field>` where `<field>` is the `target_field` of an active
/// adapter site (any kind) in `ctx`. On match, returns `(site, lhs_expr)`.
///
/// The matcher is shared across all adapter kinds — the pattern
/// (`x @ self.W`) is identical; only the synthesized replacement differs.
pub fn match_adapter_site<'a, 'b>(
    expr: &'b Expr,
    ctx: &'a RewriteContext<'a>,
) -> Option<(&'a AdapterSite, &'b Expr)> {
    let (left, right) = match &expr.kind {
        ExprKind::BinaryOp { left, op: BinOp::MatMul, right } => (left.as_ref(), right.as_ref()),
        _ => return None,
    };
    let (obj, member) = match &right.kind {
        ExprKind::MemberAccess { object, member } => (object.as_ref(), *member),
        _ => return None,
    };
    if !is_self_expr(obj, ctx.self_sym) {
        return None;
    }
    // Resolve member symbol to a string via the context's field_symbols
    // reverse lookup (the rewrite caller populates this by resolving
    // `self.resolve_sym` once up front).
    let member_name = ctx
        .field_symbols
        .iter()
        .find_map(|(name, sym)| if *sym == member { Some(name.as_str()) } else { None });
    let member_name = member_name?;
    for site in &ctx.sites {
        if site.target_field == member_name
            && site.input_dim > 0
            && site.output_dim > 0
        {
            return Some((*site, left));
        }
    }
    None
}

/// Backwards-compatible alias used by the original Task 3 tests.
pub fn match_lora_site<'a, 'b>(
    expr: &'b Expr,
    ctx: &'a RewriteContext<'a>,
) -> Option<(&'a AdapterSite, &'b Expr)> {
    match match_adapter_site(expr, ctx) {
        Some((site, lhs)) if site.kind == AdapterKind::Lora => Some((site, lhs)),
        _ => None,
    }
}

/// Build a `MemberAccess { self, <field>_name }` expression whose
/// resolution at codegen time will come from `synth_overrides`.
fn make_synth_member_access(
    ctx: &mut RewriteContext<'_>,
    object_self: Expr,
    field_name: &str,
    span: nsl_errors::Span,
    sentinel_sym: Symbol,
) -> Expr {
    let node_id = NodeId::next();
    ctx.synth_overrides
        .insert(node_id, field_name.to_string());
    Expr {
        kind: ExprKind::MemberAccess {
            object: Box::new(object_self),
            member: sentinel_sym,
        },
        span,
        id: node_id,
    }
}

fn make_self_expr(span: nsl_errors::Span) -> Expr {
    Expr {
        kind: ExprKind::SelfRef,
        span,
        id: NodeId::next(),
    }
}

fn make_bin(op: BinOp, left: Expr, right: Expr, span: nsl_errors::Span) -> Expr {
    Expr {
        kind: ExprKind::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        },
        span,
        id: NodeId::next(),
    }
}

fn make_float(v: f64, span: nsl_errors::Span) -> Expr {
    Expr {
        kind: ExprKind::FloatLiteral(v),
        span,
        id: NodeId::next(),
    }
}

/// B.3 Task 4: dispatch between the fused single-FFI path and the B.2.1
/// unfused triple based on the site's `fusion_decision` and target sm.
///
/// * Fused path: `site.fusion_decision == Some(EpilogueFusedLora)` AND
///   `ctx.target_sm >= 80` → emit one Call to `nsl_adapter_fused_lora_matmul`.
/// * Unfused path (B.2.1): any other condition → emit the three-FFI
///   triple `x @ W + ((x @ A) @ B) * scale`.
///
/// Renamed from its B.2.1 name to preserve the dispatch seam; the
/// unfused body now lives in `synthesize_lora_unfused_triple`.
pub fn synthesize_lora_adapted(
    original: Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let is_fused = matches!(
        site.fusion_decision,
        Some(crate::wrga_fusion::FusionTarget::EpilogueFusedLora { .. })
    );
    let sm_ok = ctx.target_sm.map(|sm| sm >= 80).unwrap_or(false);
    if is_fused && sm_ok {
        return synthesize_lora_fused_call(original, lhs, site, ctx);
    }
    synthesize_lora_unfused_triple(original, lhs, site, ctx)
}

/// B.3 Task 4: synthesize a single `Call` to `nsl_adapter_fused_lora_matmul`
/// with args `[lhs, self.W, self.lora_A_<site>, self.lora_B_<site>,
/// FloatLit(scale), IntLit(kernel_handle)]`.  The callee Ident carries a
/// sentinel Symbol; the real FFI name is stashed in
/// `ctx.synth_call_overrides` keyed by the callee's NodeId.
pub fn synthesize_lora_fused_call(
    original: Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let span = original.span;
    let sentinel = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("RewriteContext must have at least one field symbol");

    // Recover `self.W` from the original `x @ self.W` matmul we're replacing.
    let self_w = match &original.kind {
        ExprKind::BinaryOp { right, .. } => (**right).clone(),
        _ => {
            // Shouldn't happen — `match_adapter_site` guarantees a
            // matmul-shaped original.  Fall back to unfused to be safe.
            return synthesize_lora_unfused_triple(original, lhs, site, ctx);
        }
    };

    let a_name = format!("lora_A_{}", site.site_id);
    let b_name = format!("lora_B_{}", site.site_id);
    let self_a = make_self_expr(span);
    let ma_a = make_synth_member_access(ctx, self_a, &a_name, span, sentinel);
    let self_b = make_self_expr(span);
    let ma_b = make_synth_member_access(ctx, self_b, &b_name, span, sentinel);

    let rank = site.rank.max(1) as f64;
    let alpha = site.alpha.max(1) as f64;
    let scale = alpha / rank;

    // B.3 Task 5: derive a deterministic kernel_handle by looking up
    // this site's `LoraKernelKey` in the sorted `fused_kernel_order`.
    // Sites whose key isn't in the order (rank > 16, dims unresolved,
    // non-sm_80 target) get handle = -1 so the runtime can detect a
    // miswiring and fall back deterministically.
    let kernel_handle: i64 = {
        let target_sm = ctx.target_sm.unwrap_or(0);
        // Rank from FusionTarget::EpilogueFusedLora { rank }; fall back
        // to site.rank when the decision shape doesn't carry it.
        let rank = match &site.fusion_decision {
            Some(crate::wrga_fusion::FusionTarget::EpilogueFusedLora { rank }) => *rank as u32,
            _ => site.rank as u32,
        };
        let key = crate::wrga_fused_ptx::LoraKernelKey {
            m: 1,
            n: site.output_dim,
            k: site.input_dim,
            rank,
            target_sm,
        };
        ctx.fused_kernel_order
            .iter()
            .position(|k| k == &key)
            .map(|p| p as i64)
            .unwrap_or(-1)
    };

    // Build the callee Ident with a sentinel symbol + override entry
    // pointing at the FFI name.
    let callee_id = NodeId::next();
    ctx.synth_call_overrides
        .insert(callee_id, "nsl_adapter_fused_lora_matmul".to_string());
    let callee = Expr {
        kind: ExprKind::Ident(sentinel),
        span,
        id: callee_id,
    };

    let args = vec![
        Arg { name: None, value: lhs.clone(), span },
        Arg { name: None, value: self_w, span },
        Arg { name: None, value: ma_a, span },
        Arg { name: None, value: ma_b, span },
        Arg {
            name: None,
            value: make_float(scale, span),
            span,
        },
        Arg {
            name: None,
            value: Expr {
                kind: ExprKind::IntLiteral(kernel_handle),
                span,
                id: NodeId::next(),
            },
            span,
        },
    ];

    Expr {
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args,
        },
        span,
        id: NodeId::next(),
    }
}

/// B.2.1's original body, preserved verbatim.  Synthesize
/// `original + ((lhs @ self.lora_A_<site>) @ self.lora_B_<site>) * scale`.
/// `original` is the already-recursively-rewritten `x @ self.W` matmul.
/// `lhs` is a reference to the (already-rewritten) x expression — it
/// will be cloned for the second matmul chain.
pub fn synthesize_lora_unfused_triple(
    original: Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let span = original.span;
    // Sentinel Symbol: reuse any Symbol from field_symbols (any arbitrary
    // interned symbol works — the override map overrides its resolution).
    let sentinel = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("RewriteContext must have at least one field symbol");

    let a_name = format!("lora_A_{}", site.site_id);
    let b_name = format!("lora_B_{}", site.site_id);

    let self_a = make_self_expr(span);
    let ma_a = make_synth_member_access(ctx, self_a, &a_name, span, sentinel);
    let x_at_a = make_bin(BinOp::MatMul, lhs.clone(), ma_a, span);

    let self_b = make_self_expr(span);
    let ma_b = make_synth_member_access(ctx, self_b, &b_name, span, sentinel);
    let xa_at_b = make_bin(BinOp::MatMul, x_at_a, ma_b, span);

    let rank = site.rank.max(1) as f64;
    let alpha = site.alpha.max(1) as f64;
    let scale = alpha / rank;
    let scaled = make_bin(BinOp::Mul, xa_at_b, make_float(scale, span), span);

    make_bin(BinOp::Add, original, scaled, span)
}

/// Synthesize `(original) * self.ia3_scale_<site>`.
///
/// `original` is the already-recursively-rewritten `x @ self.W` matmul.
/// IA³ multiplies the matmul result elementwise by a per-output-channel
/// scale vector initialized to ones (base-model equivalent at step 0).
/// Broadcast semantics: matmul result has shape `[..., d_out]`, scale has
/// shape `[d_out]`, standard NSL tensor broadcasting applies.
pub fn synthesize_ia3_adapted(
    original: Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let is_fused = matches!(
        site.fusion_decision,
        Some(crate::wrga_fusion::FusionTarget::ActivationFusedIa3)
    );
    let sm_ok = ctx.target_sm.map(|sm| sm >= 80).unwrap_or(false);
    if is_fused && sm_ok {
        return synthesize_ia3_fused_call(original, site, ctx);
    }
    synthesize_ia3_unfused_mul(original, site, ctx)
}

/// B.3 Task 4: single-Call fused IA³ entry —
/// `nsl_adapter_fused_ia3_matmul(x, self.W, self.ia3_scale_<site>, handle)`.
pub fn synthesize_ia3_fused_call(
    original: Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let span = original.span;
    let sentinel = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("RewriteContext must have at least one field symbol");

    let (lhs_expr, self_w) = match &original.kind {
        ExprKind::BinaryOp { left, right, .. } => ((**left).clone(), (**right).clone()),
        _ => return synthesize_ia3_unfused_mul(original, site, ctx),
    };

    let scale_name = format!("ia3_scale_{}", site.site_id);
    let self_s = make_self_expr(span);
    let ma_scale = make_synth_member_access(ctx, self_s, &scale_name, span, sentinel);

    // B.3 Task 5: IA³ has no PTX registry yet (LoRA-only in prescan), so
    // emit -1 as a sentinel.  The Task-4 CPU stub ignores the handle.
    let kernel_handle: i64 = -1;

    let callee_id = NodeId::next();
    ctx.synth_call_overrides
        .insert(callee_id, "nsl_adapter_fused_ia3_matmul".to_string());
    let callee = Expr {
        kind: ExprKind::Ident(sentinel),
        span,
        id: callee_id,
    };

    let args = vec![
        Arg { name: None, value: lhs_expr, span },
        Arg { name: None, value: self_w, span },
        Arg { name: None, value: ma_scale, span },
        Arg {
            name: None,
            value: Expr {
                kind: ExprKind::IntLiteral(kernel_handle),
                span,
                id: NodeId::next(),
            },
            span,
        },
    ];

    Expr {
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args,
        },
        span,
        id: NodeId::next(),
    }
}

/// B.2.1's original IA³ body, preserved verbatim.
pub fn synthesize_ia3_unfused_mul(
    original: Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let span = original.span;
    let sentinel = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("RewriteContext must have at least one field symbol");

    let scale_name = format!("ia3_scale_{}", site.site_id);
    let self_s = make_self_expr(span);
    let ma_scale = make_synth_member_access(ctx, self_s, &scale_name, span, sentinel);

    make_bin(BinOp::Mul, original, ma_scale, span)
}

/// B.3.1: synthesize a single `Call` to `nsl_adapter_fused_gatedlora_matmul`
/// with args `[lhs, self.W, self.lora_A_<site>, self.lora_B_<site>,
/// FloatLit(scale), self.gate_<site>, IntLit(kernel_handle)]`.
///
/// The callee Ident carries a sentinel Symbol; the real FFI name is stashed in
/// `ctx.synth_call_overrides` keyed by the callee's NodeId (same pattern as
/// `synthesize_lora_fused_call`).
///
/// `kernel_handle` is stubbed at 0 until Task 5.0.c registers the real PTX
/// kernel and wires the handle through `ctx.fused_kernel_order`.
pub fn synthesize_gatedlora_fused_call(
    original: Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let span = original.span;
    let sentinel = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("RewriteContext must have at least one field symbol");

    // Recover `self.W` from the original `x @ self.W` matmul.
    let self_w = match &original.kind {
        ExprKind::BinaryOp { right, .. } => (**right).clone(),
        _ => {
            // Shouldn't happen — match_adapter_site guarantees a
            // matmul-shaped original.  Fall back to unfused to be safe.
            return synthesize_gatedlora_unfused(original, lhs, site, ctx);
        }
    };

    let a_name = format!("lora_A_{}", site.site_id);
    let b_name = format!("lora_B_{}", site.site_id);
    let g_name = format!("gate_{}", site.site_id);

    let self_a = make_self_expr(span);
    let ma_a = make_synth_member_access(ctx, self_a, &a_name, span, sentinel);
    let self_b = make_self_expr(span);
    let ma_b = make_synth_member_access(ctx, self_b, &b_name, span, sentinel);
    let self_g = make_self_expr(span);
    let ma_g = make_synth_member_access(ctx, self_g, &g_name, span, sentinel);

    let rank = site.rank.max(1) as f64;
    let alpha = site.alpha.max(1) as f64;
    let scale = alpha / rank;

    // B.3.1 Task 5.0.c: look up the GatedLoRA-specific kernel order.
    // GatedLoRA handles are assigned as `lora_order.len() + idx_in_gatedlora_order`
    // to avoid collisions with LoRA handles that share the same shape.
    let kernel_handle: i64 = {
        let target_sm = ctx.target_sm.unwrap_or(0);
        let rank_u32 = match &site.fusion_decision {
            Some(crate::wrga_fusion::FusionTarget::EpilogueFusedGatedLora { rank }) => {
                *rank as u32
            }
            _ => site.rank as u32,
        };
        let key = crate::wrga_fused_ptx::LoraKernelKey {
            m: 1,
            n: site.output_dim,
            k: site.input_dim,
            rank: rank_u32,
            target_sm,
        };
        let lora_offset = ctx.fused_kernel_order.len() as i64;
        ctx.fused_gatedlora_kernel_order
            .iter()
            .position(|k| k == &key)
            .map(|p| lora_offset + p as i64)
            .unwrap_or(-1)
    };

    let callee_id = NodeId::next();
    ctx.synth_call_overrides
        .insert(callee_id, "nsl_adapter_fused_gatedlora_matmul".to_string());
    let callee = Expr {
        kind: ExprKind::Ident(sentinel),
        span,
        id: callee_id,
    };

    let args = vec![
        Arg { name: None, value: lhs.clone(), span },
        Arg { name: None, value: self_w, span },
        Arg { name: None, value: ma_a, span },
        Arg { name: None, value: ma_b, span },
        Arg { name: None, value: make_float(scale, span), span },
        Arg { name: None, value: ma_g, span },
        Arg {
            name: None,
            value: Expr {
                kind: ExprKind::IntLiteral(kernel_handle),
                span,
                id: NodeId::next(),
            },
            span,
        },
    ];

    Expr {
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args,
        },
        span,
        id: NodeId::next(),
    }
}

/// Synthesize
/// `original + sigmoid(self.gate_<site>) * ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * scale`.
///
/// # Step-0 invariant (LOAD-BEARING — DO NOT REMOVE)
///
/// `gate_<site>` is initialized to zeros. `sigmoid(0) == 0.5`, NOT 0.
/// The gate is HALF-OPEN at step 0. Base-model equivalence at step 0
/// depends ENTIRELY on `lora_B = 0` zeroing the entire adapter
/// contribution. If `lora_B`'s init is changed from zero, the equivalence
/// invariant breaks silently. Task 5 Build 4 is the load-bearing runtime
/// assertion that catches such regressions.
///
/// # Fused-dispatch (B.3.1)
///
/// When `site.fusion_decision == Some(EpilogueFusedGatedLora)` AND
/// `ctx.target_sm >= 80`, emits a single `Call` to
/// `nsl_adapter_fused_gatedlora_matmul(x, W, A, B, scale, gate, kernel_handle)`
/// instead of the AST sigmoid+triple+scale expression.  The kernel
/// handle is currently stubbed at 0; Task 5.0.c wires real registration.
pub fn synthesize_gatedlora_adapted(
    original: Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    // Fused-dispatch branch (B.3.1): mirror synthesize_lora_adapted's pattern.
    let is_fused = matches!(
        site.fusion_decision,
        Some(crate::wrga_fusion::FusionTarget::EpilogueFusedGatedLora { .. })
    );
    let sm_ok = ctx.target_sm.map(|sm| sm >= 80).unwrap_or(false);
    if is_fused && sm_ok {
        return synthesize_gatedlora_fused_call(original, lhs, site, ctx);
    }
    synthesize_gatedlora_unfused(original, lhs, site, ctx)
}

/// Unfused GatedLoRA body, preserved verbatim from B.3.
/// Synthesize
/// `original + sigmoid(self.gate_<site>) * ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * scale`.
pub fn synthesize_gatedlora_unfused(
    original: Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext<'_>,
) -> Expr {
    let span = original.span;
    let sentinel = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("RewriteContext must have at least one field symbol");

    let a_name = format!("lora_A_{}", site.site_id);
    let b_name = format!("lora_B_{}", site.site_id);
    let g_name = format!("gate_{}", site.site_id);

    // LoRA contribution: (x @ A) @ B
    let self_a = make_self_expr(span);
    let ma_a = make_synth_member_access(ctx, self_a, &a_name, span, sentinel);
    let x_at_a = make_bin(BinOp::MatMul, lhs.clone(), ma_a, span);

    let self_b = make_self_expr(span);
    let ma_b = make_synth_member_access(ctx, self_b, &b_name, span, sentinel);
    let xa_at_b = make_bin(BinOp::MatMul, x_at_a, ma_b, span);

    // sigmoid(self.gate_<site>)
    let self_g = make_self_expr(span);
    let ma_g = make_synth_member_access(ctx, self_g, &g_name, span, sentinel);
    let sigmoid_sym = ctx
        .field_symbols
        .values()
        .next()
        .copied()
        .expect("sentinel symbol required");
    // Build sigmoid callee as a plain Ident. Codegen's expr::calls
    // resolves "sigmoid" by name, so we need an Ident whose Symbol
    // stringifies to "sigmoid" — use the compiler's synth_member_names
    // override is not available for Call callees; instead, the caller
    // is expected to resolve the sigmoid symbol via the shared interner
    // and place it in `field_symbols` under the key "sigmoid" BEFORE
    // invoking the rewrite. Production callers (adapter_inject pipeline)
    // must ensure this. In unit tests we register it explicitly.
    let sigmoid_ident_sym = ctx
        .field_symbols
        .get("sigmoid")
        .copied()
        .unwrap_or(sigmoid_sym);
    let sigmoid_callee = Expr {
        kind: ExprKind::Ident(sigmoid_ident_sym),
        span,
        id: NodeId::next(),
    };
    let sigmoid_call = Expr {
        kind: ExprKind::Call {
            callee: Box::new(sigmoid_callee),
            args: vec![Arg { name: None, value: ma_g, span }],
        },
        span,
        id: NodeId::next(),
    };

    // sigmoid(gate) * ((x @ A) @ B)
    let gated = make_bin(BinOp::Mul, sigmoid_call, xa_at_b, span);

    let rank = site.rank.max(1) as f64;
    let alpha = site.alpha.max(1) as f64;
    let scale = alpha / rank;
    let scaled = make_bin(BinOp::Mul, gated, make_float(scale, span), span);

    make_bin(BinOp::Add, original, scaled, span)
}

/// Post-order recursive walk. Recurses into children first, THEN checks
/// pattern and may replace the current node with a synthesized adapted
/// expression. This avoids re-visiting the synthesized output.
pub fn rewrite_expr(mut expr: Expr, ctx: &mut RewriteContext<'_>) -> Expr {
    // Recurse into children first.
    expr.kind = match expr.kind {
        ExprKind::BinaryOp { left, op, right } => ExprKind::BinaryOp {
            left: Box::new(rewrite_expr(*left, ctx)),
            op,
            right: Box::new(rewrite_expr(*right, ctx)),
        },
        ExprKind::UnaryOp { op, operand } => ExprKind::UnaryOp {
            op,
            operand: Box::new(rewrite_expr(*operand, ctx)),
        },
        ExprKind::Pipe { left, right } => ExprKind::Pipe {
            left: Box::new(rewrite_expr(*left, ctx)),
            right: Box::new(rewrite_expr(*right, ctx)),
        },
        ExprKind::MemberAccess { object, member } => ExprKind::MemberAccess {
            object: Box::new(rewrite_expr(*object, ctx)),
            member,
        },
        ExprKind::Subscript { object, index } => ExprKind::Subscript {
            object: Box::new(rewrite_expr(*object, ctx)),
            index: Box::new(rewrite_subscript(*index, ctx)),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: Box::new(rewrite_expr(*callee, ctx)),
            args: args
                .into_iter()
                .map(|a| Arg { name: a.name, value: rewrite_expr(a.value, ctx), span: a.span })
                .collect(),
        },
        ExprKind::ListLiteral(items) => {
            ExprKind::ListLiteral(items.into_iter().map(|e| rewrite_expr(e, ctx)).collect())
        }
        ExprKind::TupleLiteral(items) => {
            ExprKind::TupleLiteral(items.into_iter().map(|e| rewrite_expr(e, ctx)).collect())
        }
        ExprKind::DictLiteral(pairs) => ExprKind::DictLiteral(
            pairs
                .into_iter()
                .map(|(k, v)| (rewrite_expr(k, ctx), rewrite_expr(v, ctx)))
                .collect(),
        ),
        ExprKind::FString(parts) => ExprKind::FString(
            parts
                .into_iter()
                .map(|p| match p {
                    FStringPart::Text(t) => FStringPart::Text(t),
                    FStringPart::Expr(e) => FStringPart::Expr(rewrite_expr(e, ctx)),
                })
                .collect(),
        ),
        ExprKind::Paren(inner) => ExprKind::Paren(Box::new(rewrite_expr(*inner, ctx))),
        ExprKind::Await(inner) => ExprKind::Await(Box::new(rewrite_expr(*inner, ctx))),
        ExprKind::IfExpr { condition, then_expr, else_expr } => ExprKind::IfExpr {
            condition: Box::new(rewrite_expr(*condition, ctx)),
            then_expr: Box::new(rewrite_expr(*then_expr, ctx)),
            else_expr: Box::new(rewrite_expr(*else_expr, ctx)),
        },
        ExprKind::BlockExpr(block) => ExprKind::BlockExpr(rewrite_block(block, ctx)),
        ExprKind::Range { start, end, inclusive } => ExprKind::Range {
            start: start.map(|e| Box::new(rewrite_expr(*e, ctx))),
            end: end.map(|e| Box::new(rewrite_expr(*e, ctx))),
            inclusive,
        },
        ExprKind::ListComp { element, generators } => ExprKind::ListComp {
            element: Box::new(rewrite_expr(*element, ctx)),
            generators: generators
                .into_iter()
                .map(|g| CompGenerator {
                    pattern: g.pattern,
                    iterable: rewrite_expr(g.iterable, ctx),
                    conditions: g.conditions.into_iter().map(|c| rewrite_expr(c, ctx)).collect(),
                })
                .collect(),
        },
        ExprKind::MatchExpr { subject, arms } => ExprKind::MatchExpr {
            subject: Box::new(rewrite_expr(*subject, ctx)),
            arms: arms
                .into_iter()
                .map(|a| MatchArm {
                    pattern: a.pattern,
                    guard: a.guard.map(|g| rewrite_expr(g, ctx)),
                    body: rewrite_block(a.body, ctx),
                    span: a.span,
                })
                .collect(),
        },
        // Catch-all: leaf kinds (literals, idents, SelfRef, Error, Lambda body
        // — we deliberately do not descend into Lambda bodies because the
        // rewrite targets model-method bodies, and lambdas introduce a new
        // scope where `self` may not bind the outer model.)
        other => other,
    };

    // Post-order pattern match on the rewritten node.
    if let Some((site, lhs)) = match_adapter_site(&expr, ctx) {
        // Clone the site handle before we mutate ctx via synth_overrides.
        let site_copy = AdapterSite {
            site_id: site.site_id.clone(),
            kind: site.kind,
            target_param: site.target_param.clone(),
            rank: site.rank,
            alpha: site.alpha,
            synthesized_fields: site.synthesized_fields.clone(),
            input_dim: site.input_dim,
            output_dim: site.output_dim,
            target_model: site.target_model.clone(),
            target_field: site.target_field.clone(),
            fusion_decision: site.fusion_decision.clone(),
        };
        let lhs_clone = lhs.clone();
        return match site_copy.kind {
            AdapterKind::Lora => synthesize_lora_adapted(expr, &lhs_clone, &site_copy, ctx),
            AdapterKind::Ia3 => synthesize_ia3_adapted(expr, &site_copy, ctx),
            AdapterKind::GatedLora => {
                synthesize_gatedlora_adapted(expr, &lhs_clone, &site_copy, ctx)
            }
        };
    }

    expr
}

fn rewrite_subscript(sub: SubscriptKind, ctx: &mut RewriteContext<'_>) -> SubscriptKind {
    match sub {
        SubscriptKind::Index(e) => SubscriptKind::Index(rewrite_expr(e, ctx)),
        SubscriptKind::Slice { lower, upper, step } => SubscriptKind::Slice {
            lower: lower.map(|e| rewrite_expr(e, ctx)),
            upper: upper.map(|e| rewrite_expr(e, ctx)),
            step: step.map(|e| rewrite_expr(e, ctx)),
        },
        SubscriptKind::MultiDim(parts) => {
            SubscriptKind::MultiDim(parts.into_iter().map(|p| rewrite_subscript(p, ctx)).collect())
        }
    }
}

fn rewrite_block(block: Block, ctx: &mut RewriteContext<'_>) -> Block {
    Block {
        stmts: block.stmts.into_iter().map(|s| rewrite_stmt(s, ctx)).collect(),
        span: block.span,
    }
}

pub fn rewrite_stmt(stmt: Stmt, ctx: &mut RewriteContext<'_>) -> Stmt {
    let kind = match stmt.kind {
        StmtKind::Expr(e) => StmtKind::Expr(rewrite_expr(e, ctx)),
        StmtKind::Return(opt) => StmtKind::Return(opt.map(|e| rewrite_expr(e, ctx))),
        StmtKind::Yield(opt) => StmtKind::Yield(opt.map(|e| rewrite_expr(e, ctx))),
        StmtKind::VarDecl { is_const, pattern, type_ann, value } => StmtKind::VarDecl {
            is_const,
            pattern,
            type_ann,
            value: value.map(|e| rewrite_expr(e, ctx)),
        },
        StmtKind::Assign { target, op, value } => StmtKind::Assign {
            target: rewrite_expr(target, ctx),
            op,
            value: rewrite_expr(value, ctx),
        },
        StmtKind::If { condition, then_block, elif_clauses, else_block } => StmtKind::If {
            condition: rewrite_expr(condition, ctx),
            then_block: rewrite_block(then_block, ctx),
            elif_clauses: elif_clauses
                .into_iter()
                .map(|(c, b)| (rewrite_expr(c, ctx), rewrite_block(b, ctx)))
                .collect(),
            else_block: else_block.map(|b| rewrite_block(b, ctx)),
        },
        StmtKind::For { pattern, iterable, body } => StmtKind::For {
            pattern,
            iterable: rewrite_expr(iterable, ctx),
            body: rewrite_block(body, ctx),
        },
        StmtKind::While { condition, body } => StmtKind::While {
            condition: rewrite_expr(condition, ctx),
            body: rewrite_block(body, ctx),
        },
        StmtKind::WhileLet { pattern, expr, body } => StmtKind::WhileLet {
            pattern,
            expr: rewrite_expr(expr, ctx),
            body: rewrite_block(body, ctx),
        },
        StmtKind::Match { subject, arms } => StmtKind::Match {
            subject: rewrite_expr(subject, ctx),
            arms: arms
                .into_iter()
                .map(|a| MatchArm {
                    pattern: a.pattern,
                    guard: a.guard.map(|g| rewrite_expr(g, ctx)),
                    body: rewrite_block(a.body, ctx),
                    span: a.span,
                })
                .collect(),
        },
        other => other,
    };
    Stmt { kind, span: stmt.span, id: stmt.id }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::{BytePos, FileId, Span};
    use nsl_lexer::Interner;

    fn dummy_span() -> Span {
        Span::new(FileId(0), BytePos(0), BytePos(0))
    }

    fn mk_site(target_field: &str) -> AdapterSite {
        AdapterSite {
            site_id: format!("m_{}__lora", target_field),
            kind: AdapterKind::Lora,
            target_param: format!("m.{}", target_field),
            rank: 8,
            alpha: 16,
            synthesized_fields: vec![
                format!("lora_A_m_{}__lora", target_field),
                format!("lora_B_m_{}__lora", target_field),
            ],
            input_dim: 16,
            output_dim: 32,
            target_model: "Toy".to_string(),
            target_field: target_field.to_string(),
            fusion_decision: None,
        }
    }

    fn ident(sym: Symbol) -> Expr {
        Expr { kind: ExprKind::Ident(sym), span: dummy_span(), id: NodeId::next() }
    }

    fn self_ref() -> Expr {
        Expr { kind: ExprKind::SelfRef, span: dummy_span(), id: NodeId::next() }
    }

    fn member(object: Expr, sym: Symbol) -> Expr {
        Expr {
            kind: ExprKind::MemberAccess { object: Box::new(object), member: sym },
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    fn bin(left: Expr, op: BinOp, right: Expr) -> Expr {
        Expr {
            kind: ExprKind::BinaryOp { left: Box::new(left), op, right: Box::new(right) },
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    fn make_ctx<'a>(
        sites: Vec<&'a AdapterSite>,
        interner: &mut Interner,
        field_name: &str,
    ) -> (RewriteContext<'a>, Symbol) {
        let mut ctx = RewriteContext::new(sites);
        let w_sym: Symbol = interner.get_or_intern(field_name).into();
        ctx.field_symbols.insert(field_name.to_string(), w_sym);
        (ctx, w_sym)
    }

    #[test]
    fn matches_x_at_self_w_on_lora_target() {
        let site = mk_site("w");
        let mut interner = Interner::new();
        let x_sym: Symbol = interner.get_or_intern("x").into();
        let (ctx, w_sym) = make_ctx(vec![&site], &mut interner, "w");

        let x = ident(x_sym);
        let sw = member(self_ref(), w_sym);
        let matmul = bin(x, BinOp::MatMul, sw);

        let m = match_lora_site(&matmul, &ctx);
        assert!(m.is_some(), "should match x @ self.w for LoRA target");
    }

    #[test]
    fn does_not_match_non_self_receiver() {
        let site = mk_site("w");
        let mut interner = Interner::new();
        let x_sym: Symbol = interner.get_or_intern("x").into();
        let other_sym: Symbol = interner.get_or_intern("other").into();
        let (ctx, w_sym) = make_ctx(vec![&site], &mut interner, "w");

        let x = ident(x_sym);
        // other.w — receiver is NOT self
        let ow = member(ident(other_sym), w_sym);
        let matmul = bin(x, BinOp::MatMul, ow);

        assert!(match_lora_site(&matmul, &ctx).is_none());
    }

    #[test]
    fn does_not_match_non_matmul_op() {
        let site = mk_site("w");
        let mut interner = Interner::new();
        let x_sym: Symbol = interner.get_or_intern("x").into();
        let (ctx, w_sym) = make_ctx(vec![&site], &mut interner, "w");

        let x = ident(x_sym);
        let sw = member(self_ref(), w_sym);
        // Add, not MatMul
        let add = bin(x, BinOp::Add, sw);

        assert!(match_lora_site(&add, &ctx).is_none());
    }

    fn mk_site_with_kind(target_field: &str, kind: AdapterKind) -> AdapterSite {
        let tag = match kind {
            AdapterKind::Lora => "lora",
            AdapterKind::Ia3 => "ia3",
            AdapterKind::GatedLora => "gatedlora",
        };
        AdapterSite {
            site_id: format!("m_{}__{}", target_field, tag),
            kind,
            target_param: format!("m.{}", target_field),
            rank: 8,
            alpha: 16,
            synthesized_fields: vec![],
            input_dim: 16,
            output_dim: 32,
            target_model: "Toy".to_string(),
            target_field: target_field.to_string(),
            fusion_decision: None,
        }
    }

    #[test]
    fn ia3_rewrite_multiplies_matmul_result_by_scale() {
        let site = mk_site_with_kind("w", AdapterKind::Ia3);
        let mut interner = Interner::new();
        let x_sym: Symbol = interner.get_or_intern("x").into();
        let (mut ctx, w_sym) = make_ctx(vec![&site], &mut interner, "w");

        let x = ident(x_sym);
        let sw = member(self_ref(), w_sym);
        let matmul = bin(x, BinOp::MatMul, sw);

        let out = rewrite_expr(matmul, &mut ctx);
        // Expected: (x @ self.w) * self.ia3_scale_<site>
        match &out.kind {
            ExprKind::BinaryOp { op: BinOp::Mul, left, right } => {
                // left: matmul x @ self.w
                match &left.kind {
                    ExprKind::BinaryOp { op: BinOp::MatMul, .. } => {}
                    other => panic!("expected MatMul on left, got {:?}", other),
                }
                // right: MemberAccess (self, ia3_scale_<site>)
                match &right.kind {
                    ExprKind::MemberAccess { .. } => {}
                    other => panic!("expected MemberAccess on right, got {:?}", other),
                }
            }
            other => panic!("expected top-level Mul for IA3, got {:?}", other),
        }
        // Exactly one synth override (ia3_scale_<site>).
        assert_eq!(ctx.synth_overrides.len(), 1);
        let name = ctx.synth_overrides.values().next().unwrap();
        assert!(name.starts_with("ia3_scale_"), "got {name}");
    }

    #[test]
    fn gatedlora_rewrite_adds_sigmoid_gate_times_lora_contrib() {
        // STEP-0 INVARIANT (keep this comment forever):
        //   `gate_<site>` is initialized to zeros. `sigmoid(0) == 0.5`, NOT 0.
        //   The gate is HALF-OPEN at step 0. Base-model equivalence at step 0
        //   depends ENTIRELY on `lora_B = 0`. A refactor that changes B's init
        //   without simultaneously changing the gate's init will silently break
        //   the equivalence invariant. Task 5 Build 4 is the load-bearing
        //   runtime assertion.
        let site = mk_site_with_kind("w", AdapterKind::GatedLora);
        let mut interner = Interner::new();
        let x_sym: Symbol = interner.get_or_intern("x").into();
        let sig_sym: Symbol = interner.get_or_intern("sigmoid").into();
        let (mut ctx, w_sym) = make_ctx(vec![&site], &mut interner, "w");
        ctx.field_symbols.insert("sigmoid".to_string(), sig_sym);

        let x = ident(x_sym);
        let sw = member(self_ref(), w_sym);
        let matmul = bin(x, BinOp::MatMul, sw);

        let out = rewrite_expr(matmul, &mut ctx);
        // Top-level should be Add (original + scaled gated LoRA).
        match &out.kind {
            ExprKind::BinaryOp { op: BinOp::Add, left, right } => {
                // left: original matmul
                match &left.kind {
                    ExprKind::BinaryOp { op: BinOp::MatMul, .. } => {}
                    other => panic!("expected original MatMul on left, got {:?}", other),
                }
                // right: Mul(..., scale_float)
                match &right.kind {
                    ExprKind::BinaryOp { op: BinOp::Mul, left: inner_left, right: inner_right } => {
                        // inner_right must be a float literal (alpha/rank scale)
                        match &inner_right.kind {
                            ExprKind::FloatLiteral(_) => {}
                            other => panic!("expected scale FloatLiteral, got {:?}", other),
                        }
                        // inner_left: Mul(sigmoid(gate), (x @ A) @ B)
                        match &inner_left.kind {
                            ExprKind::BinaryOp {
                                op: BinOp::Mul,
                                left: gate_side,
                                right: lora_side,
                            } => {
                                // gate_side: sigmoid(self.gate_<site>)
                                match &gate_side.kind {
                                    ExprKind::Call { .. } => {}
                                    other => panic!(
                                        "expected sigmoid Call, got {:?}",
                                        other
                                    ),
                                }
                                // lora_side: (x @ A) @ B
                                match &lora_side.kind {
                                    ExprKind::BinaryOp { op: BinOp::MatMul, .. } => {}
                                    other => panic!(
                                        "expected LoRA chain MatMul, got {:?}",
                                        other
                                    ),
                                }
                            }
                            other => panic!("expected gated Mul, got {:?}", other),
                        }
                    }
                    other => panic!("expected scale Mul on right, got {:?}", other),
                }
            }
            other => panic!("expected top-level Add for GatedLoRA, got {:?}", other),
        }
        // Three synth overrides: lora_A_<site>, lora_B_<site>, gate_<site>.
        assert_eq!(ctx.synth_overrides.len(), 3);
        let names: Vec<&String> = ctx.synth_overrides.values().collect();
        assert!(names.iter().any(|n| n.starts_with("lora_A_")));
        assert!(names.iter().any(|n| n.starts_with("lora_B_")));
        assert!(names.iter().any(|n| n.starts_with("gate_")));
    }

    #[test]
    fn rewrite_expr_replaces_matching_matmul_with_scaled_sum() {
        let site = mk_site("w");
        let mut interner = Interner::new();
        let x_sym: Symbol = interner.get_or_intern("x").into();
        let (mut ctx, w_sym) = make_ctx(vec![&site], &mut interner, "w");

        let x = ident(x_sym);
        let sw = member(self_ref(), w_sym);
        let matmul = bin(x, BinOp::MatMul, sw);

        let out = rewrite_expr(matmul, &mut ctx);
        // Output should be Add (scaled LoRA wrapper).
        match out.kind {
            ExprKind::BinaryOp { op: BinOp::Add, .. } => {}
            other => panic!("expected Add wrapper, got {:?}", other),
        }
        // And we should have registered two synth overrides (A and B fields).
        assert_eq!(ctx.synth_overrides.len(), 2);
        let names: Vec<&String> = ctx.synth_overrides.values().collect();
        assert!(names.iter().any(|n| n.starts_with("lora_A_")));
        assert!(names.iter().any(|n| n.starts_with("lora_B_")));
    }
}
