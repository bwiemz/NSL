//! M40: Source-to-source AD — adjoint generation, dead gradient elimination,
//! and saved tensor analysis.

use crate::ad_rules::{
    apply_ad_rule, csha_dispatch_for_op, AdjointExpr, CshaDispatchDecision, InputAdjoint,
};
use crate::csha_apply::FusionMark;
use crate::wengert::{
    type_for_op, CompareKind, OpId, PrimalOp, SubgraphId, VarId, WengertList, WengertOp,
    WengertType,
};
use std::collections::{HashMap, HashSet};

// Cycle-10 §5.3 (Task 5): semantic-layer CheckpointPolicy. nsl-codegen
// already depends on nsl-semantic via Cargo.toml; the loader (Task 6)
// passes the per-function map collected by `EffectChecker` into
// `WengertExtractor::with_checkpoint_policies`.
use nsl_semantic::effects::CheckpointPolicy;

// M40b: Wengert extraction from typed AST
use nsl_ast::expr::ExprKind;
use nsl_ast::operator::{BinOp, UnaryOp as AstUnaryOp};
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;

// ---------------------------------------------------------------------------
// Adjoint generation
// ---------------------------------------------------------------------------

/// T7.1: CSHA backward dispatch claims threaded into the adjoint generator.
///
/// Built by `collect_chain_dispatch_map` in `csha_apply.rs`.  When
/// present, the reverse walk checks each op against `op_to_chain`
/// before falling through to `apply_ad_rule`.
pub struct CshaBackwardClaims {
    /// Maps Wengert op index → chain index in `chain_marks`.
    pub op_to_chain: HashMap<u32, usize>,
    /// One canonical FusionMark per boundary chain.  All ops in a chain
    /// share the same mark (and its `backward_emitted` Cell).
    pub chain_marks: Vec<FusionMark>,
}

/// T7.2: a single "fused backward would be emitted" event produced by
/// the dispatcher when a claimed chain's config is accepted.
///
/// Records the chain metadata downstream tooling (`nsl profile`, sidecar
/// reports, planner consumers) needs to show the user which layers
/// WOULD go through the fused path — even while the full kernel-launch
/// codegen is deferred behind the forward save-buffer plumbing.
///
/// When the actual kernel-launch emission is wired up (see the
/// architectural-gaps list in `generate`), consumers can diff
/// `csha_fused_events.len()` against a run that uses per-op AD to
/// quantify the fusion coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct CshaFusedBackwardEvent {
    /// Layer prefix (e.g. `blocks.0`) the chain targets.
    pub layer: String,
    /// The chain's output-op Wengert index (RoPE when present, else
    /// the Q/K/V projection matmul). Used by downstream tools to
    /// correlate fused events with primal ops in a profile trace.
    pub output_op_id: u32,
    /// Head dim from the resolved config — useful to tell Track A
    /// (head_dim=32/64) from Track B (head_dim=128) chains apart in
    /// reports without hauling the full FlashAttentionConfig around.
    pub head_dim: i64,
    /// block_q/block_kv pair — recorded so the smoke-config gate can
    /// be audited from downstream tests.
    pub block_q: i64,
    pub block_kv: i64,
    /// `true` when this event fired on the smoke config and the
    /// dispatcher took the `EmitFused` path (stub today; kernel-launch
    /// codegen deferred). `false` when the dispatcher scope-gated the
    /// chain to per-op AD even though the backward validator accepted
    /// it (e.g. head_dim > 32). Kept so tests can pin the scope gate.
    pub smoke_config: bool,
    /// Gap I step K: `true` when this fused emission also emitted the
    /// standalone `NormGammaBackward` adjoint (for `norm_weight_var`
    /// gamma trainables). `false` when the chain had no trainable
    /// gamma (`norm_weight_var = None`) or `x_raw_var` resolution
    /// failed. Load-bearing so the toy-pretrain smoke test can pin
    /// that the gamma gradient is actually wired.
    pub dgamma_emitted: bool,
}

/// Memoization key for `MaterializeConvOutputGrad`. Two sibling gradients
/// (input/weight/bias) of ONE `Conv2d` node may share a reified `grad_output`
/// only when every input to the materialize op matches: the same `grad` VarId
/// (identical contents) reified against the same output geometry — which is a
/// function of `input`/`weight` shapes plus `stride`/`padding`. This is exactly
/// the materialize op's own identity `(grad, input, weight, stride, padding)`,
/// so the cache is CSE scoped to that op: it shares among siblings (identical
/// keys) but keeps DISTINCT `Conv2d` nodes separate even when they happen to
/// share one upstream `grad` VarId (e.g. a broadcast scalar sum-loss seed feeding
/// two convs of different geometry) — where reusing the first node's differently
/// shaped materialized tensor would be silently wrong.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ConvGradMaterializeKey {
    grad: VarId,
    input: VarId,
    weight: VarId,
    stride: usize,
    padding: usize,
}

/// Generates the backward (adjoint) Wengert list from a forward (primal) list.
pub struct AdjointGenerator {
    adjoint_ops: Vec<WengertOp>,
    adjoint_vars: HashMap<VarId, VarId>,
    var_counter: VarId,
    op_counter: OpId,
    /// T7.1: optional CSHA claims for fused backward dispatch.
    csha_claims: Option<CshaBackwardClaims>,
    /// T7.1: diagnostics emitted when a CSHA chain falls back to per-op AD.
    csha_diagnostics: Vec<String>,
    /// T7.2: one event per claimed chain whose config was accepted by the
    /// backward validator (regardless of whether the smoke-config scope
    /// gate caused the dispatcher to actually take the fused path). See
    /// [`CshaFusedBackwardEvent`].
    csha_fused_events: Vec<CshaFusedBackwardEvent>,
    /// Conv2d's 2-3 gradient adjoints (input/weight/bias) all reify the same
    /// `grad_output` to the conv output shape. Memoizes the reified VarId by the
    /// materialize op's full identity ([`ConvGradMaterializeKey`]) so only the
    /// first sibling emits the `MaterializeConvOutputGrad` op; the rest reuse it.
    /// Without this, each sibling reifies (and, for a scalar sum-loss gradient,
    /// allocates) its own copy — up to 3x the necessary allocations per conv2d
    /// node per step. Keying on the full geometry (not `grad` alone) keeps two
    /// different-geometry conv nodes that share an upstream `grad` VarId from
    /// colliding on a mismatched output shape.
    conv_grad_materialize_cache: HashMap<ConvGradMaterializeKey, VarId>,
}

impl AdjointGenerator {
    pub fn new(start_var: VarId) -> Self {
        Self {
            adjoint_ops: Vec::new(),
            adjoint_vars: HashMap::new(),
            var_counter: start_var,
            op_counter: 0,
            csha_claims: None,
            csha_diagnostics: Vec::new(),
            csha_fused_events: Vec::new(),
            conv_grad_materialize_cache: HashMap::new(),
        }
    }

    /// T7.1: attach CSHA backward claims to this generator.  Must be
    /// called before `generate()`.  When set, the reverse walk will
    /// check each op against the claims map and route to the fused
    /// backward path (or skip) instead of per-op AD rules.
    pub fn set_csha_claims(&mut self, claims: CshaBackwardClaims) {
        self.csha_claims = Some(claims);
    }

    /// CSLA D2b part 2: hand the claims back after `generate()`. The
    /// hoisted pipeline runs adjoint generation BEFORE the forward
    /// lowering, but the forward's fused-SDPA claim dispatch still needs
    /// the table (`compiler.csha_backward_claims`); the compiler restores
    /// it from here and clears its slot after the forward, preserving the
    /// old invariant that the ADJOINT lowering never sees claims. The
    /// forward reads only `op_to_chain` / `chain_marks` metadata — never
    /// the `backward_emitted` cells generation flipped — so the ordering
    /// change is invisible to it.
    pub fn take_csha_claims(&mut self) -> Option<CshaBackwardClaims> {
        self.csha_claims.take()
    }

    /// T7.1: return diagnostics collected during the reverse walk for
    /// CSHA chains that fell back to per-op AD.
    pub fn csha_diagnostics(&self) -> &[String] {
        &self.csha_diagnostics
    }

    /// T7.2: return the per-chain "fused backward accepted" events
    /// recorded during the reverse walk. See [`CshaFusedBackwardEvent`].
    pub fn csha_fused_events(&self) -> &[CshaFusedBackwardEvent] {
        &self.csha_fused_events
    }

    fn next_var(&mut self) -> VarId {
        let v = self.var_counter;
        self.var_counter += 1;
        v
    }

    fn next_op(&mut self) -> OpId {
        let o = self.op_counter;
        self.op_counter += 1;
        o
    }

    /// Expose primal→adjoint VarId map for passes that need to translate
    /// primal-space VarIds (e.g. `wrga_prune::backward_live`) into the
    /// adjoint namespace produced by `generate`.
    pub fn adjoint_vars_map(&self) -> &HashMap<VarId, VarId> {
        &self.adjoint_vars
    }

    pub fn get_or_create_adjoint(&mut self, primal_var: VarId) -> VarId {
        if let Some(&adj) = self.adjoint_vars.get(&primal_var) {
            adj
        } else {
            // Allocate a new adjoint VarId. No op is emitted yet —
            // the VarId is a "ghost" that gets populated by accumulate_adjoint
            // when a gradient arrives. If no gradient arrives, the VarId
            // stays unmapped and the lowerer should produce zeros_like(primal).
            let adj = self.next_var();
            self.adjoint_vars.insert(primal_var, adj);
            adj
        }
    }

    /// Generate the backward Wengert list by walking primal ops in reverse.
    pub fn generate(&mut self, primal: &WengertList) -> WengertList {
        let loss_bar = self.next_var();
        let seed_op_id = self.next_op();
        self.adjoint_vars.insert(primal.output, loss_bar);
        self.adjoint_ops.push(WengertOp {
            id: seed_op_id,
            result: loss_bar,
            op: PrimalOp::Constant(1.0),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });

        // T7.2: track chain indices whose fused event has already been
        // recorded in this generate() call. Because the stub resets
        // `backward_emitted = false` after recording the event (so the
        // other claimed ops in the chain still go through per-op AD),
        // we can't rely on the mark's Cell for single-event semantics —
        // local state gives us one event per chain.
        let mut fused_event_emitted: HashSet<usize> = HashSet::new();

        for op in primal.ops.iter().rev() {
            // T7.1: check CSHA backward dispatch before per-op AD.
            if let Some(ref claims) = self.csha_claims {
                if let Some(&chain_idx) = claims.op_to_chain.get(&op.id) {
                    let mark = &claims.chain_marks[chain_idx];
                    match csha_dispatch_for_op(mark, op.id) {
                        CshaDispatchDecision::EmitFused => {
                            // Gap D: real kernel-launch emission for the fused
                            // backward.  See `PrimalOp::FusedCshaBackward` in
                            // wengert.rs + its lowerer arm in wengert_lower.rs
                            // for the mechanical details of the launch.
                            //
                            // Gap D.1 claim-site fix:
                            //   - When the dispatch map was built with a
                            //     Wengert list that contained the SDPA op
                            //     (production path), the primary claim is
                            //     the SDPA op itself — by the time the
                            //     reverse walk hits it, `y_bar(SDPA_output)`
                            //     is already populated by downstream ops
                            //     (output projection, residual add, etc.),
                            //     so we consume that as `dO`. The fused
                            //     kernel's 7 outputs then land on the
                            //     correct VarIds (q_out/k_out/v_out/wq/wk/
                            //     wv/x_norm) via `mark.chain_varids`. The
                            //     Q/K/V matmul ops + RMSNorm + RoPE ops in
                            //     the same chain map to the same chain_idx
                            //     and return `AlreadyEmitted` on subsequent
                            //     reverse-walk visits, suppressing their
                            //     per-op AD (no double-accumulation).
                            //   - When `chain_varids` is absent (structural
                            //     tests, open-coded attention, or a partial
                            //     chain group), we fall back to the legacy
                            //     best-effort routing that treats `op` as a
                            //     matmul. This is the path the T7.x unit
                            //     tests in ad_csha_reverse_walk_wiring.rs
                            //     exercise.
                            let mark_layer = mark.layer.clone();
                            let mark_kind = mark.kind;
                            let cfg = mark.config.clone().expect(
                                "dispatcher accepted EmitFused with None config — \
                                 csha_dispatch_for_op contract violated",
                            );
                            let chain_varids = mark.chain_varids;
                            let is_smoke = cfg.head_dim == 32
                                && cfg.block_q == 32
                                && cfg.block_kv == 32
                                && cfg.csha.as_ref().is_some_and(|c| c.d_model == 32);

                            // One event per chain.
                            if fused_event_emitted.insert(chain_idx) {
                                self.csha_fused_events.push(CshaFusedBackwardEvent {
                                    layer: mark_layer.clone(),
                                    output_op_id: op.id,
                                    head_dim: cfg.head_dim,
                                    block_q: cfg.block_q,
                                    block_kv: cfg.block_kv,
                                    smoke_config: is_smoke,
                                    // Gap I step K: flipped below if the
                                    // chain has a trainable gamma.
                                    dgamma_emitted: false,
                                });
                                let this_event_idx = self.csha_fused_events.len() - 1;

                                eprintln!(
                                    "[nsl] CSHA fused backward: emitting fused launch for \
                                     layer '{}' (hd={}, block_q/kv={}x{}, d_model={}, smoke={})",
                                    mark_layer,
                                    cfg.head_dim,
                                    cfg.block_q,
                                    cfg.block_kv,
                                    cfg.csha.as_ref().map_or(0, |c| c.d_model),
                                    is_smoke,
                                );

                                // Chain-key VarId: the op's result (the SDPA
                                // output when chain_varids is present, or
                                // the matmul/rope output in the legacy
                                // path). All 7 extract ops share this as
                                // their first input so they hit the same
                                // cache entry at lowering time.
                                let chain_key = op.result;

                                // dO (adjoint of the fused kernel's "output"
                                // tensor) — the Gap D.1 routing:
                                //   - With `chain_varids`: read y_bar of the
                                //     SDPA output (already populated by
                                //     downstream reverse-walk iterations).
                                //   - Without: fall back to the current op's
                                //     y_bar (legacy best-effort placeholder).
                                let do_source_var = chain_varids
                                    .as_ref()
                                    .map(|v| v.sdpa_out_var)
                                    .unwrap_or(op.result);
                                let do_var = self.get_or_create_adjoint(do_source_var);

                                // Launch-op inputs (Gap I.4): the lowerer's
                                // `FusedCshaBackward` arm indexes:
                                //
                                //   [0]  chain_key  — cache key; NOT a
                                //                    real tensor read
                                //   [1]  do_ptr    — dO adjoint
                                //   [2]  q_ptr     — SDPA op.inputs[0]
                                //   [3]  k_ptr     — SDPA op.inputs[1]
                                //   [4]  v_ptr     — SDPA op.inputs[2]
                                //   [5]  x_ptr     — RMSNorm output (x_norm)
                                //   [6]  wq_ptr    — Q projection weight
                                //   [7]  wk_ptr    — K projection weight
                                //   [8]  wv_ptr    — V projection weight
                                //   [9]  norm_w_ptr — RMSNorm gamma (or null)
                                //
                                // Pre-Gap-I.4 the launch only carried 5
                                // entries (chain_key, dO, q, k, v). The
                                // lowerer's null-default branches left
                                // wq/wk/wv/x/norm_weight at null, and the
                                // backward PTX's per-weight null-guard
                                // (csha_hooks_backward.rs:348-350) then
                                // skipped `V2_BWD_DPROJ_{WQ,WK,WV}_LOOP`,
                                // returning zero-filled dwq/dwk/dwv. This
                                // is Gap I step J's fix: push the 5 extra
                                // VarIds when chain_varids is populated,
                                // so the kernel receives real device
                                // pointers and weight gradients are no
                                // longer zero.
                                //
                                // When `chain_varids` is None (legacy
                                // per-chain marks, structural tests, or
                                // floating chains without SDPA), stick
                                // to the 5-entry shape so the lowerer's
                                // null-default path kicks in — those
                                // code paths don't populate CSHA saves
                                // and can't safely launch the backward
                                // kernel anyway.
                                // Launch-op inputs contract (from the
                                // doc block above): positions [2..5) carry
                                // **only** q, k, v — the SDPA op's first
                                // three tensor inputs. `op.inputs` can hold
                                // more than 3 entries (scaled_dot_product_attention
                                // has a 4th scalar `scale` VarId, and
                                // optionally a 5th causal-flag VarId;
                                // matmul chains have exactly 2). We clip
                                // to 3 so the x_norm/wq/wk/wv/norm_w
                                // pushes below land at the positions the
                                // FusedCshaBackward lowerer reads
                                // (inputs[5..10]). Passing SDPA's `scale`
                                // scalar at inputs[5] would shift every
                                // subsequent slot and make the lowerer
                                // marshal a scalar as x_ptr, wq's
                                // shape as norm_w_ptr, etc. — the exact
                                // symptom seen in the device-placement
                                // trace (a CU_MEMORYTYPE_DEVICE address
                                // read from what should have been a
                                // 4 KB f32 tensor but was interpreted as
                                // a scale-bits i64 scalar handle).
                                let mut launch_inputs = vec![chain_key, do_var];
                                launch_inputs.extend(op.inputs.iter().copied().take(3));
                                if let Some(v) = chain_varids.as_ref() {
                                    launch_inputs.push(v.x_norm_var); // [5]
                                    launch_inputs.push(v.wq_var); // [6]
                                    launch_inputs.push(v.wk_var); // [7]
                                    launch_inputs.push(v.wv_var); // [8]
                                    // Null (VarId 0) when the RMSNorm
                                    // has no trainable gamma param. The
                                    // backward PTX null-guards on
                                    // `csha_norm_weight_ptr`.
                                    launch_inputs.push(v.norm_weight_var.unwrap_or(0)); // [9]
                                }

                                let launch_result = self.emit_op(
                                    PrimalOp::FusedCshaBackward {
                                        layer: mark_layer.clone(),
                                    },
                                    launch_inputs,
                                );

                                // Gap I.2 + M: each extract op now lists the
                                // launch op's result VarId as its FIRST input
                                // so `eliminate_dead_gradients`' worklist walk
                                // (which traverses `op.inputs` back from
                                // `needed_vars`) reaches the launch op via any
                                // live extract. Without this, the launch's
                                // result is referenced by nothing and the pass
                                // prunes it — leaving the extracts to fail
                                // with "no cache entry".
                                //
                                // The launch's Cranelift Value (inputs[0] in
                                // the lowerer) is a placeholder zero tensor
                                // and is not actually consumed; inputs[1] is
                                // the `chain_key` used to look up the cache.
                                let mut extract_results: [VarId; 8] = [0u32; 8];
                                for component in 0u8..=7u8 {
                                    let r = self.emit_op(
                                        PrimalOp::CshaFusedBackwardExtract { component },
                                        vec![launch_result, chain_key],
                                    );
                                    extract_results[component as usize] = r;
                                }

                                // Route the 8 outputs to the right VarIds.
                                if let Some(v) = chain_varids {
                                    // Gap D.1 primary routing (extended by
                                    // Gap I.5 Option A): outputs land on the
                                    // correct primal VarIds, so downstream
                                    // accumulate_adjoint calls for the
                                    // SUPPRESSED per-op backward
                                    // (matmul/RMSNorm/RoPE) don't run and no
                                    // double-accumulation occurs.
                                    //   0 = dq      → q_out_var (Q matmul or RoPE-Q output)
                                    //   1 = dk      → k_out_var
                                    //   2 = dv      → v_out_var
                                    //   3 = dwq     → wq_var
                                    //   4 = dwk     → wk_var
                                    //   5 = dwv     → wv_var
                                    //   6 = dx_raw  → x_raw_var (pre-RMSNorm input)
                                    //   7 = dx_norm → x_norm_var (RMSNorm output — dy_norm)
                                    //
                                    // Pre-Gap-I.5 Option A the routing sent
                                    // extract[6] (dx_raw) to `x_norm_var`,
                                    // which was semantically wrong but
                                    // benign (nothing downstream read it).
                                    // Now extract[6] goes to `x_raw_var`
                                    // (correct — it's the gradient w.r.t.
                                    // the pre-RMSNorm input) and the new
                                    // extract[7] (dx_norm) goes to
                                    // `x_norm_var` (correct — it's the
                                    // gradient w.r.t. the RMSNorm output).
                                    self.accumulate_adjoint(v.q_out_var, extract_results[0]);
                                    self.accumulate_adjoint(v.k_out_var, extract_results[1]);
                                    self.accumulate_adjoint(v.v_out_var, extract_results[2]);
                                    self.accumulate_adjoint(v.wq_var, extract_results[3]);
                                    self.accumulate_adjoint(v.wk_var, extract_results[4]);
                                    self.accumulate_adjoint(v.wv_var, extract_results[5]);
                                    // Route dx_raw to x_raw_var when known;
                                    // fall back to x_norm_var (legacy
                                    // routing) when the chain didn't
                                    // resolve x_raw_var. This keeps the
                                    // Gap D.1 contract for chains without a
                                    // trainable RMSNorm (no x_raw_var
                                    // resolution is performed in that case).
                                    let x_raw_target =
                                        v.x_raw_var.unwrap_or(v.x_norm_var);
                                    self.accumulate_adjoint(x_raw_target, extract_results[6]);
                                    self.accumulate_adjoint(v.x_norm_var, extract_results[7]);

                                    // Gap I step K: standalone RMSNorm
                                    // gamma gradient. The suppressed per-op
                                    // backward for the RMSNorm op
                                    // (`AlreadyEmitted`) would have
                                    // emitted the RMSNorm gamma adjoint via
                                    // `ad_rules.rs:~388`; we replicate
                                    // that here so gamma receives a real
                                    // gradient instead of cascade-skipping
                                    // out of the lowered grad var set.
                                    //
                                    // Option B from the Gap I design
                                    // doc § K: reuse the existing per-op
                                    // AD lowering rather than adding an
                                    // 8th kernel output. Zero PTX
                                    // changes; zero new primal ops.
                                    //
                                    // Formula: `dgamma = reduce(y_bar * x / rms)`
                                    // matches `RmsNormGammaBackward`'s
                                    // lowering. Gap I.5 Option-A fix: now
                                    // uses `extract_results[7]` (= dx_norm,
                                    // the gradient w.r.t. the RMSNorm
                                    // OUTPUT), which is exactly `dy_norm` —
                                    // the semantically correct input for
                                    // `RmsNormGammaBackward`'s `grad`
                                    // argument. Previously extract[6]
                                    // (dx_raw, post-dRMSNorm) was used,
                                    // producing numerically incorrect
                                    // dgamma values when the CSHA
                                    // dispatcher claim fired on programs
                                    // with trainable gamma.
                                    if let (Some(gamma_var), Some(x_raw_var)) =
                                        (v.norm_weight_var, v.x_raw_var)
                                    {
                                        let dgamma = self.lower_adjoint_expr(
                                            AdjointExpr::RmsNormGammaBackward(
                                                extract_results[7],
                                                x_raw_var,
                                                v.rmsnorm_eps,
                                                gamma_var,
                                            ),
                                        );
                                        self.accumulate_adjoint(gamma_var, dgamma);
                                        if let Some(ev) =
                                            self.csha_fused_events.get_mut(this_event_idx)
                                        {
                                            ev.dgamma_emitted = true;
                                        }
                                        eprintln!(
                                            "[nsl] CSHA fused backward: emitted dgamma \
                                             (NormGammaBackward) for layer '{}' → \
                                             gamma VarId {} (x_raw VarId {}, eps={:e})",
                                            mark_layer,
                                            gamma_var,
                                            x_raw_var,
                                            v.rmsnorm_eps,
                                        );
                                    }
                                } else {
                                    // Legacy best-effort routing: treat `op`
                                    // as a matmul. Wrong for RoPE/norm, but
                                    // kept for the structural unit tests
                                    // that don't construct a full SDPA op.
                                    use crate::csha_boundary::ProjKind;
                                    let (proj_component, weight_component) = match mark_kind {
                                        Some(ProjKind::Q) => (0u8, 3u8),
                                        Some(ProjKind::K) => (1u8, 4u8),
                                        Some(ProjKind::V) => (2u8, 5u8),
                                        None => (0u8, 3u8),
                                    };
                                    self.accumulate_adjoint(
                                        op.result,
                                        extract_results[proj_component as usize],
                                    );
                                    if let Some(&weight_vid) = op.inputs.get(1) {
                                        self.accumulate_adjoint(
                                            weight_vid,
                                            extract_results[weight_component as usize],
                                        );
                                    }
                                    if let Some(&x_vid) = op.inputs.first() {
                                        self.accumulate_adjoint(x_vid, extract_results[6]);
                                    }
                                }

                                // We emitted the fused launch + extracts;
                                // skip per-op AD for this op (the fused
                                // kernel's gradients already cover it).
                                continue;
                            }

                            // Already emitted for this chain — skip per-op AD
                            // so we don't double-accumulate.
                            continue;
                        }
                        CshaDispatchDecision::AlreadyEmitted => {
                            // The fused kernel already handles this op's gradient.
                            // Skip per-op AD entirely.
                            continue;
                        }
                        CshaDispatchDecision::Fallback { diagnostic } => {
                            self.csha_diagnostics.push(diagnostic);
                            // Fall through to per-op AD below.
                        }
                    }
                }
            }

            let output_bar = self.get_or_create_adjoint(op.result);

            // WRGA B.3 fused LoRA forward AD rule.
            //
            // Forward: y = x @ W + scale * (x @ A @ B).
            // Given dy:
            //   dy_sc = dy * scale
            //   dx  += dy @ W.T + dy_sc @ B.T @ A.T
            //   dW  += x.T @ dy
            //   dA  += x.T @ dy_sc @ B.T
            //   dB  += (x @ A).T @ dy_sc
            if let PrimalOp::FusedLoraMatmul { scale, .. } = op.op {
                let x = op.inputs[0];
                let w = op.inputs[1];
                let a = op.inputs[2];
                let b_in = op.inputs[3];
                let dy = output_bar;

                let scale_c = self.emit_constant(scale as f64);
                let dy_sc = self.emit_op(PrimalOp::Mul, vec![dy, scale_c]);
                let xa = self.emit_op(PrimalOp::Matmul, vec![x, a]);

                let x_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![x]);
                let w_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![w]);
                let a_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![a]);
                let b_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![b_in]);
                let xa_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![xa]);

                let dw = self.emit_op(PrimalOp::Matmul, vec![x_t, dy]);
                self.accumulate_adjoint(w, dw);

                let xt_dysc = self.emit_op(PrimalOp::Matmul, vec![x_t, dy_sc]);
                let da = self.emit_op(PrimalOp::Matmul, vec![xt_dysc, b_t]);
                self.accumulate_adjoint(a, da);

                let db = self.emit_op(PrimalOp::Matmul, vec![xa_t, dy_sc]);
                self.accumulate_adjoint(b_in, db);

                let dx_base = self.emit_op(PrimalOp::Matmul, vec![dy, w_t]);
                let dy_sc_bt = self.emit_op(PrimalOp::Matmul, vec![dy_sc, b_t]);
                let dx_ad = self.emit_op(PrimalOp::Matmul, vec![dy_sc_bt, a_t]);
                let dx = self.emit_op(PrimalOp::Add, vec![dx_base, dx_ad]);
                self.accumulate_adjoint(x, dx);
                continue;
            }

            // WRGA B.3 fused IA³ forward AD rule.
            //
            // Forward: y[B, N] = (x[B, K] @ W[K, N]) * gamma[N] (broadcast).
            // Given dy:
            //   dxw = dy * gamma
            //   dx  += dxw @ W.T
            //   dW  += x.T @ dxw
            //   dgamma += ReduceSum(dy * (x @ W), axis=0)
            if let PrimalOp::FusedIa3Matmul { .. } = op.op {
                let x = op.inputs[0];
                let w = op.inputs[1];
                let gamma = op.inputs[2];
                let dy = output_bar;

                let xw = self.emit_op(PrimalOp::Matmul, vec![x, w]);
                let dxw = self.emit_op(PrimalOp::Mul, vec![dy, gamma]);

                let x_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![x]);
                let w_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![w]);

                let dw = self.emit_op(PrimalOp::Matmul, vec![x_t, dxw]);
                self.accumulate_adjoint(w, dw);

                let dx = self.emit_op(PrimalOp::Matmul, vec![dxw, w_t]);
                self.accumulate_adjoint(x, dx);

                let dy_xw = self.emit_op(PrimalOp::Mul, vec![dy, xw]);
                let dgamma = self.emit_op(PrimalOp::Sum { dim: Some(0) }, vec![dy_xw]);
                self.accumulate_adjoint(gamma, dgamma);
                continue;
            }

            // WRGA B.3.2 Option 3 revised: fused GatedLoRA forward AD rule.
            //
            // Forward:
            //   y[B, N] = x[B, K] @ W[K, N]
            //           + sigmoid(gate[N]) ⊙ (x @ A @ B)[B, N] * scale
            //
            // Given upstream adjoint dy[B, N]:
            //   sig[N]          = Sigmoid(gate)
            //   sig_prime[N]    = sig * (1 - sig)
            //   xa[B, R]        = Matmul(x, A)
            //   xab[B, N]       = Matmul(xa, B)
            //   dy_sig[B, N]    = Mul(dy, sig)       (broadcast over batch)
            //   dy_sig_sc[B, N] = Mul(dy_sig, scale) (scale factored once)
            //
            //   dx  += Matmul(dy, W.T)
            //       +  Matmul(Matmul(dy_sig_sc, B.T), A.T)
            //   dW  += Matmul(x.T, dy)
            //   dA  += Matmul(Matmul(x.T, dy_sig_sc), B.T)
            //   dB  += Matmul(xa.T, dy_sig_sc)
            //   dgate += ReduceSum(dy * xab * sig_prime, axis=0) * scale
            //
            // Load-bearing:
            //   - dy_sig is elementwise Mul with broadcast, NOT Matmul.
            //   - dgate reduces over axis=0 (batch); shape drops [B,N] -> [N].
            //   - scale is applied once (into dy_sig_sc) and reused.
            if let PrimalOp::FusedGatedLoraMatmul { scale, .. } = op.op {
                let x = op.inputs[0];
                let w = op.inputs[1];
                let a = op.inputs[2];
                let b_in = op.inputs[3];
                let gate = op.inputs[4];
                let dy = output_bar;

                let sig = self.emit_op(PrimalOp::Sigmoid, vec![gate]);
                let one = self.emit_constant(1.0);
                let one_minus_sig = self.emit_op(PrimalOp::Sub, vec![one, sig]);
                let sig_prime = self.emit_op(PrimalOp::Mul, vec![sig, one_minus_sig]);
                let xa = self.emit_op(PrimalOp::Matmul, vec![x, a]);
                let xab = self.emit_op(PrimalOp::Matmul, vec![xa, b_in]);

                let dy_sig = self.emit_op(PrimalOp::Mul, vec![dy, sig]);
                let scale_c = self.emit_constant(scale as f64);
                let dy_sig_sc = self.emit_op(PrimalOp::Mul, vec![dy_sig, scale_c]);

                let x_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![x]);
                let dw = self.emit_op(PrimalOp::Matmul, vec![x_t, dy]);
                self.accumulate_adjoint(w, dw);

                let b_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![b_in]);
                let xt_dysigsc = self.emit_op(PrimalOp::Matmul, vec![x_t, dy_sig_sc]);
                let da = self.emit_op(PrimalOp::Matmul, vec![xt_dysigsc, b_t]);
                self.accumulate_adjoint(a, da);

                let xa_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![xa]);
                let db = self.emit_op(PrimalOp::Matmul, vec![xa_t, dy_sig_sc]);
                self.accumulate_adjoint(b_in, db);

                let w_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![w]);
                let dx_base = self.emit_op(PrimalOp::Matmul, vec![dy, w_t]);
                let dy_sig_sc_bt = self.emit_op(PrimalOp::Matmul, vec![dy_sig_sc, b_t]);
                let a_t = self.emit_op(PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX }, vec![a]);
                let dx_adapter = self.emit_op(PrimalOp::Matmul, vec![dy_sig_sc_bt, a_t]);
                let dx = self.emit_op(PrimalOp::Add, vec![dx_base, dx_adapter]);
                self.accumulate_adjoint(x, dx);

                let dy_xab = self.emit_op(PrimalOp::Mul, vec![dy, xab]);
                let dy_xab_sp = self.emit_op(PrimalOp::Mul, vec![dy_xab, sig_prime]);
                let reduced = self.emit_op(PrimalOp::Sum { dim: Some(0) }, vec![dy_xab_sp]);
                let dgate = self.emit_op(PrimalOp::Mul, vec![reduced, scale_c]);
                self.accumulate_adjoint(gate, dgate);
                continue;
            }

            let input_adjoints = apply_ad_rule(op, output_bar);

            for InputAdjoint { input_var, expr } in input_adjoints {
                let adj_val = self.lower_adjoint_expr(expr);
                self.accumulate_adjoint(input_var, adj_val);
            }
        }

        // Build var_types for the adjoint graph from its ops.
        let mut adjoint_var_types = HashMap::new();
        for op in &self.adjoint_ops {
            adjoint_var_types.insert(op.result, type_for_op(&op.op));
        }
        WengertList {
            ops: self.adjoint_ops.clone(),
            output: loss_bar,
            var_names: HashMap::new(),
            var_types: adjoint_var_types,
        }
    }

    /// Emit a single intermediate op and return its result VarId.
    fn emit_op(&mut self, op: PrimalOp, inputs: Vec<VarId>) -> VarId {
        let result = self.next_var();
        let op_id = self.next_op();
        self.adjoint_ops.push(WengertOp {
            id: op_id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        });
        result
    }

    /// Emit a constant value and return its VarId.
    fn emit_constant(&mut self, value: f64) -> VarId {
        self.emit_op(PrimalOp::Constant(value), vec![])
    }

    fn lower_adjoint_expr(&mut self, expr: AdjointExpr) -> VarId {
        // Identity: pass-through, no op needed
        if let AdjointExpr::Identity(v) = expr {
            return v;
        }

        match expr {
            AdjointExpr::Identity(v) => {
                // Gradient passes through unchanged — emit a "copy" op
                // so the VarId gets registered in the var_map during lowering.
                self.emit_op(PrimalOp::Broadcast, vec![v])
            }

            // --- Simple pass-through ops (single instruction, mathematically correct) ---
            AdjointExpr::Negate(v) => self.emit_op(PrimalOp::Neg, vec![v]),
            AdjointExpr::MulElementwise(grad, other, target) => {
                // Broadcast-aware backward for elementwise multiply.
                // The raw product `grad * other` carries the broadcast output
                // shape; the gradient accumulating into `target` must be
                // reduced (summed) over any broadcast axes so it matches
                // target's storage shape. Mirrors the broadcast-correct
                // sibling arms MatmulTransposeRight, Expand/ReduceToShape,
                // and the RMSNorm dgamma path.
                let raw = self.emit_op(PrimalOp::Mul, vec![grad, other]);
                self.emit_op(
                    PrimalOp::Passthrough("reduce_to_shape".into()),
                    vec![raw, target],
                )
            }
            AdjointExpr::MatmulTransposeLeft(grad, b) => {
                // d_loss/d_A = grad @ B^T (transpose last two dims for N-D support)
                let b_t = self.emit_op(
                    PrimalOp::Transpose {
                        dim0: usize::MAX - 1,
                        dim1: usize::MAX,
                    },
                    vec![b],
                );
                self.emit_op(PrimalOp::Matmul, vec![grad, b_t])
            }
            AdjointExpr::MatmulTransposeRight(a, grad, b) => {
                // d_loss/d_B = A^T @ grad (transpose last two dims for N-D support)
                // When the forward matmul broadcasts (A has more dims than B),
                // the gradient must be summed over the extra batch dimensions
                // so it matches B's shape.
                let a_t = self.emit_op(
                    PrimalOp::Transpose {
                        dim0: usize::MAX - 1,
                        dim1: usize::MAX,
                    },
                    vec![a],
                );
                let raw_grad = self.emit_op(PrimalOp::Matmul, vec![a_t, grad]);
                // Reduce to match B's shape: nsl_tensor_reduce_to_shape(raw_grad, B)
                self.emit_op(
                    PrimalOp::Passthrough("reduce_to_shape".into()),
                    vec![raw_grad, b],
                )
            }
            AdjointExpr::Scale(v, s) => {
                let scale = self.emit_constant(s);
                self.emit_op(PrimalOp::Mul, vec![v, scale])
            }
            AdjointExpr::Broadcast(v) => self.emit_op(PrimalOp::Broadcast, vec![v]),
            AdjointExpr::ScaleBroadcast(v, n) => {
                let scale = self.emit_constant(n);
                let scaled = self.emit_op(PrimalOp::Mul, vec![v, scale]);
                self.emit_op(PrimalOp::Broadcast, vec![scaled])
            }
            AdjointExpr::Transpose(v, d0, d1) => {
                self.emit_op(PrimalOp::Transpose { dim0: d0, dim1: d1 }, vec![v])
            }
            AdjointExpr::ReshapeLike(v, target) => {
                let shape = self.emit_op(PrimalOp::Passthrough("shape".into()), vec![target]);
                self.emit_op(PrimalOp::Passthrough("reshape".into()), vec![v, shape])
            }
            // --- Expand backward: sum-reduce gradient over broadcast-expanded dims ---
            AdjointExpr::ReduceToShape(grad, target) => self.emit_op(
                PrimalOp::Passthrough("reduce_to_shape".into()),
                vec![grad, target],
            ),

            // --- Exp backward: d(exp(x))/dx = exp(x) = y. grad * y (correct as-is) ---
            AdjointExpr::ExpBackward(y_bar, y) => self.emit_op(PrimalOp::Mul, vec![y_bar, y]),

            // --- ReLU backward: grad * (x > 0), NOT grad * x ---
            AdjointExpr::ReluBackward(y_bar, x) => {
                let zero = self.emit_constant(0.0);
                let cond = self.emit_op(PrimalOp::Condition(CompareKind::Gt), vec![x, zero]);
                self.emit_op(PrimalOp::Select, vec![cond, y_bar, zero])
            }

            // --- Sigmoid backward: grad * σ(x) * (1 - σ(x)), where y = σ(x) ---
            // Fused (Milestone C · p4 slice 3): the three-op expansion (Sub, Mul,
            // Mul → three kernels + two intermediates) collapses to a single
            // `nsl_tensor_sigmoid_backward(grad, y)` launch, BIT-EXACT with the
            // decomposed path — see `SIGMOID_BACKWARD_SRCAD_F32_PTX`.
            AdjointExpr::SigmoidBackward(y_bar, y) => {
                self.emit_op(PrimalOp::Passthrough("sigmoid_backward".into()), vec![y_bar, y])
            }

            // --- Tanh backward: grad * (1 - tanh²(x)), where y = tanh(x) ---
            // Fused (Milestone C · p4 slice 3): the three-op expansion (Mul, Sub,
            // Mul) collapses to a single `nsl_tensor_tanh_backward(grad, y)`
            // launch, BIT-EXACT with the decomposed path (LOAD-BEARING `.rn`
            // blocks the y*y→1-y*y fma-contraction) — see
            // `TANH_BACKWARD_SRCAD_F32_PTX`.
            AdjointExpr::TanhBackward(y_bar, y) => {
                self.emit_op(PrimalOp::Passthrough("tanh_backward".into()), vec![y_bar, y])
            }

            // --- Log backward: grad / x (correct as-is) ---
            AdjointExpr::LogBackward(y_bar, x) => self.emit_op(PrimalOp::Div, vec![y_bar, x]),

            // --- Sqrt backward: grad / (2 * sqrt(x)), where y = sqrt(x) ---
            AdjointExpr::SqrtBackward(y_bar, y) => {
                let two = self.emit_constant(2.0);
                let two_y = self.emit_op(PrimalOp::Mul, vec![two, y]);
                self.emit_op(PrimalOp::Div, vec![y_bar, two_y])
            }

            // --- Div numerator backward: d(a/b)/da = 1/b, so grad / b ---
            AdjointExpr::DivNumeratorBackward(y_bar, b) => {
                self.emit_op(PrimalOp::Div, vec![y_bar, b])
            }

            // --- Div denominator backward: d(a/b)/db = -a/b², so grad * (-a/b²) ---
            AdjointExpr::DivDenominatorBackward(y_bar, a, b) => {
                let b_sq = self.emit_op(PrimalOp::Mul, vec![b, b]);
                let a_over_b_sq = self.emit_op(PrimalOp::Div, vec![a, b_sq]);
                let neg = self.emit_op(PrimalOp::Neg, vec![a_over_b_sq]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, neg])
            }

            // --- Select backward (control flow) ---
            AdjointExpr::SelectTrue(y_bar, cond) => {
                let zero = self.emit_constant(0.0);
                self.emit_op(PrimalOp::Select, vec![cond, y_bar, zero])
            }
            AdjointExpr::SelectFalse(y_bar, cond) => {
                let zero = self.emit_constant(0.0);
                self.emit_op(PrimalOp::Select, vec![cond, zero, y_bar])
            }

            // =================================================================
            // Compound backward rules — multi-instruction lowerings
            // =================================================================

            // --- GELU backward: grad * gelu'(x) ---
            // Fused (Milestone C · p4 GELU fix): a single
            // `nsl_tensor_gelu_backward(grad, x)` launch computing the derivative
            // of the forward each device actually ran (GPU: sigmoid approx
            // σ(1.702x)·(1+1.702x·(1−σ)); CPU: tanh-approx, same as tape-AD).
            //
            // The previous 7-op expansion here was numerically WRONG: its
            // internal temp `kx = Mul(1.702, x)` had refcount 1, and the
            // expansion's own `Sigmoid(kx)` FBIP-mutated it in place during the
            // adjoint pass (where the in-place-suppression guard is deliberately
            // clear), so the later `Mul(kx, 1−s)` read σ(kx) and the result was
            // s·(1+s·(1−s)) — 0.625 instead of 0.5 at x=0. This was the ONLY
            // adjoint expansion creating an internal temp that is consumed by an
            // FBIP-capable unary op and then read again; do not reintroduce that
            // pattern (fuse instead, or keep temps single-use).
            AdjointExpr::GeluBackward(y_bar, x) => {
                self.emit_op(PrimalOp::Passthrough("gelu_backward".into()), vec![y_bar, x])
            }

            // --- SiLU backward: d/dx[x * σ(x)] = σ(x) * (1 + x * (1 - σ(x))) ---
            // Fused (Milestone C · p4 slice 2): the six-op expansion below
            // (Sigmoid, Sub, Mul, Add, Mul, Mul → six kernels + five
            // intermediates) collapses to a single `nsl_tensor_silu_backward`
            // launch. The runtime kernel reproduces this exact operation order,
            // so it is BIT-EXACT with the decomposed path — see
            // `nsl_tensor_silu_backward` / `SILU_BACKWARD_SRCAD_F32_PTX`.
            AdjointExpr::SiluBackward(y_bar, x) => {
                self.emit_op(PrimalOp::Passthrough("silu_backward".into()), vec![y_bar, x])
            }

            // --- Abs backward: grad * sign(x) ---
            AdjointExpr::SignMul(y_bar, x) => {
                let zero = self.emit_constant(0.0);
                let pos_cond = self.emit_op(PrimalOp::Condition(CompareKind::Gt), vec![x, zero]);
                let one = self.emit_constant(1.0);
                let neg_one = self.emit_constant(-1.0);
                let sign = self.emit_op(PrimalOp::Select, vec![pos_cond, one, neg_one]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, sign])
            }

            // --- Clamp backward: grad * (min <= x <= max) ---
            // Derivative is 1 when x is in [min, max], 0 outside.
            // Implemented as chained Selects: pass grad only if x >= min AND x <= max.
            AdjointExpr::ClampBackward(y_bar, x, min, max) => {
                let zero = self.emit_constant(0.0);
                let min_val = self.emit_constant(min);
                let max_val = self.emit_constant(max);
                let ge_min = self.emit_op(PrimalOp::Condition(CompareKind::GtEq), vec![x, min_val]);
                let le_max = self.emit_op(PrimalOp::Condition(CompareKind::LtEq), vec![x, max_val]);
                let pass_if_le_max = self.emit_op(PrimalOp::Select, vec![le_max, y_bar, zero]);
                self.emit_op(PrimalOp::Select, vec![ge_min, pass_if_le_max, zero])
            }

            // --- Softmax backward: s * (grad - dot(grad, s)) where s = softmax output ---
            // The sum along the last dimension (the softmax axis) is emitted
            // as sum_keepdim_last which resolves at lowering time to
            // nsl_tensor_sum_dim(input, ndim-1, keepdim=1). Using keepdim=1
            // preserves the trailing dimension as size-1, so the subsequent
            // Sub broadcasts naturally (e.g. [B,nh,S,1] against [B,nh,S,S]).
            AdjointExpr::SoftmaxBackward(y_bar, y) => {
                let dot = self.emit_op(PrimalOp::Mul, vec![y_bar, y]);
                let dot_sum = self.emit_op(
                    PrimalOp::Passthrough("sum_keepdim_last".into()),
                    vec![dot],
                );
                let diff = self.emit_op(PrimalOp::Sub, vec![y_bar, dot_sum]);
                self.emit_op(PrimalOp::Mul, vec![y, diff])
            }

            // --- LogSoftmax backward: grad - exp(y) * sum(grad) ---
            // Same sum_keepdim_last pattern as softmax backward.
            AdjointExpr::LogSoftmaxBackward(y_bar, y) => {
                let exp_y = self.emit_op(PrimalOp::Exp, vec![y]);
                let grad_sum = self.emit_op(
                    PrimalOp::Passthrough("sum_keepdim_last".into()),
                    vec![y_bar],
                );
                let correction = self.emit_op(PrimalOp::Mul, vec![exp_y, grad_sum]);
                self.emit_op(PrimalOp::Sub, vec![y_bar, correction])
            }

            // --- LayerNorm backward: dx = rstd * (grad - mean(grad) - x_hat * mean(grad * x_hat)) ---
            // Recomputes mean/rstd from input (not op.result) for correctness.
            // Every Mean{dim} reduction MUST be followed by Broadcast before use in
            // Sub/Mul with full-shape tensors to avoid shape mismatch.
            AdjointExpr::LayerNormBackward(y_bar, x, _mean_unused, _rstd_unused, eps_val) => {
                // Recompute mean and rstd from input (standard approach, matches PyTorch).
                // All mean_keepdim_last calls reduce the last dim with keepdim=1,
                // so the result broadcasts naturally against the full-shape tensors
                // (e.g. [B,S,1] against [B,S,D]). No Broadcast op needed.
                let mean = self.emit_op(PrimalOp::Passthrough("mean_keepdim_last".into()), vec![x]);
                let x_centered = self.emit_op(PrimalOp::Sub, vec![x, mean]);
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x_centered, x_centered]);
                let var = self.emit_op(PrimalOp::Passthrough("mean_keepdim_last".into()), vec![x_sq]);
                let eps = self.emit_constant(eps_val);
                let var_eps = self.emit_op(PrimalOp::Add, vec![var, eps]);
                let std = self.emit_op(PrimalOp::Sqrt, vec![var_eps]);
                let one = self.emit_constant(1.0);
                let rstd = self.emit_op(PrimalOp::Div, vec![one, std]);
                let x_hat = self.emit_op(PrimalOp::Mul, vec![x_centered, rstd]);
                // Compute gradient corrections
                let mean_grad = self.emit_op(PrimalOp::Passthrough("mean_keepdim_last".into()), vec![y_bar]);
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                let mean_gxh = self.emit_op(PrimalOp::Passthrough("mean_keepdim_last".into()), vec![grad_x_hat]);
                let correction = self.emit_op(PrimalOp::Mul, vec![x_hat, mean_gxh]);
                let t1 = self.emit_op(PrimalOp::Sub, vec![y_bar, mean_grad]);
                let t2 = self.emit_op(PrimalOp::Sub, vec![t1, correction]);
                self.emit_op(PrimalOp::Mul, vec![t2, rstd])
            }

            // --- BatchNorm backward: same structure as LayerNorm but over batch dim (dim=0) ---
            // Recomputes mean/rstd from input for correctness.
            // Every Mean{dim} reduction MUST be followed by Broadcast before use in
            // Sub/Mul with full-shape tensors to avoid shape mismatch.
            AdjointExpr::BatchNormBackward(y_bar, x, _mean_unused, _rstd_unused, eps_val) => {
                // Recompute mean and rstd from input over batch dimension
                let mean = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![x]);
                let mean_bc = self.emit_op(PrimalOp::Broadcast, vec![mean]);
                let x_centered = self.emit_op(PrimalOp::Sub, vec![x, mean_bc]);
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x_centered, x_centered]);
                let var = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![x_sq]);
                let eps = self.emit_constant(eps_val);
                let var_eps = self.emit_op(PrimalOp::Add, vec![var, eps]);
                let std = self.emit_op(PrimalOp::Sqrt, vec![var_eps]);
                let one = self.emit_constant(1.0);
                let rstd = self.emit_op(PrimalOp::Div, vec![one, std]);
                let rstd_bc = self.emit_op(PrimalOp::Broadcast, vec![rstd]);
                let x_hat = self.emit_op(PrimalOp::Mul, vec![x_centered, rstd_bc]);
                // Compute gradient corrections
                let mean_grad = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![y_bar]);
                let mean_grad_bc = self.emit_op(PrimalOp::Broadcast, vec![mean_grad]);
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                let mean_gxh = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![grad_x_hat]);
                let mean_gxh_bc = self.emit_op(PrimalOp::Broadcast, vec![mean_gxh]);
                let correction = self.emit_op(PrimalOp::Mul, vec![x_hat, mean_gxh_bc]);
                let t1 = self.emit_op(PrimalOp::Sub, vec![y_bar, mean_grad_bc]);
                let t2 = self.emit_op(PrimalOp::Sub, vec![t1, correction]);
                self.emit_op(PrimalOp::Mul, vec![t2, rstd_bc])
            }

            // --- RMSNorm gamma backward: grad * (x / rms) ---
            //
            // RMSNorm's forward is `y = gamma * x / rms` where
            // `rms = sqrt(mean(x^2) + eps)` over the LAST dim.  Unlike
            // LayerNorm, RMSNorm does NOT subtract the per-row mean from x.
            //
            // Therefore dgamma[d] = sum_i (dy[i,d] * x[i,d] / rms[i]).
            //
            // Using `NormGammaBackward`'s `(x - mean(x)) / std` formulation
            // here would yield dgamma = 0 whenever every row of x is constant
            // (e.g. the all-ones smoke input), silently masking the real
            // gradient.  That bug surfaced as "w_norm delta = 0" in the
            // end-to-end CSHA GPU smoke on 2026-04-16.
            AdjointExpr::RmsNormGammaBackward(y_bar, x, eps_val, weight) => {
                // Recompute rms = sqrt(mean(x^2) + eps) over the last dim,
                // keepdim so it broadcasts against x.
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x, x]);
                let mean_sq = self.emit_op(
                    PrimalOp::Passthrough("mean_keepdim_last".into()),
                    vec![x_sq],
                );
                let eps = self.emit_constant(eps_val);
                let ms_eps = self.emit_op(PrimalOp::Add, vec![mean_sq, eps]);
                let rms = self.emit_op(PrimalOp::Sqrt, vec![ms_eps]);
                // x_hat = x / rms (elementwise, with rms broadcasting along
                // the last dim from shape [..., 1] to [..., D]).
                let x_hat = self.emit_op(PrimalOp::Div, vec![x, rms]);
                // dgamma = reduce_to_shape(y_bar * x_hat, weight).
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                self.emit_op(
                    PrimalOp::Passthrough("reduce_to_shape".into()),
                    vec![grad_x_hat, weight],
                )
            }

            // --- LayerNorm / BatchNorm gamma backward: grad * x_hat ---
            // Recomputes x_hat = (x - mean) / std from input to get the correct
            // normalized values (NOT the output, which is gamma * x_hat + beta).
            AdjointExpr::NormGammaBackward(y_bar, x, eps_val, dim, weight) => {
                // When dim == -1 (LayerNorm/RMSNorm), use keepdim variants for
                // correct broadcasting. When dim == 0 (BatchNorm), the standard
                // Mean + Broadcast(clone) pattern works because reducing dim=0
                // of [B,D] gives [D] which broadcasts as [1,D] → [B,D].
                let (mean_op, use_broadcast) = if dim == -1 {
                    (PrimalOp::Passthrough("mean_keepdim_last".into()), false)
                } else {
                    (PrimalOp::Mean { dim: Some(dim) }, true)
                };
                let mean = self.emit_op(mean_op.clone(), vec![x]);
                let mean_val = if use_broadcast {
                    self.emit_op(PrimalOp::Broadcast, vec![mean])
                } else {
                    mean
                };
                let x_centered = self.emit_op(PrimalOp::Sub, vec![x, mean_val]);
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x_centered, x_centered]);
                let var = self.emit_op(mean_op, vec![x_sq]);
                let eps = self.emit_constant(eps_val);
                let var_eps = self.emit_op(PrimalOp::Add, vec![var, eps]);
                let std = self.emit_op(PrimalOp::Sqrt, vec![var_eps]);
                let one = self.emit_constant(1.0);
                let rstd = self.emit_op(PrimalOp::Div, vec![one, std]);
                let rstd_val = if use_broadcast {
                    self.emit_op(PrimalOp::Broadcast, vec![rstd])
                } else {
                    rstd
                };
                let x_hat = self.emit_op(PrimalOp::Mul, vec![x_centered, rstd_val]);
                // dgamma = reduce_to_shape(grad * x_hat, weight)
                // The weight has shape [last_dim], so we sum over all dims except the last.
                // Using reduce_to_shape handles arbitrary input ndim.
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                self.emit_op(
                    PrimalOp::Passthrough("reduce_to_shape".into()),
                    vec![grad_x_hat, weight],
                )
            }

            // --- Dropout backward: grad * mask * (1/(1-p)) ---
            AdjointExpr::DropoutBackward(y_bar, mask, scale) => {
                let masked = self.emit_op(PrimalOp::Mul, vec![y_bar, mask]);
                let inv_keep = self.emit_constant(scale);
                self.emit_op(PrimalOp::Mul, vec![masked, inv_keep])
            }

            // --- Embedding backward: scatter_add(grad, indices) into weight-shaped gradient ---
            // Uses dedicated runtime function that creates a zeros tensor matching
            // the weight's shape and scatter-adds gradient rows at index positions.
            AdjointExpr::EmbeddingBackward(y_bar, indices, weight) => self.emit_op(
                PrimalOp::Passthrough("embedding_backward".into()),
                vec![y_bar, indices, weight],
            ),

            // --- Gather backward: scatter_add(grad, indices, dim) ---
            AdjointExpr::GatherBackward(y_bar, indices, dim) => {
                self.emit_op(PrimalOp::ScatterAdd { dim }, vec![y_bar, indices])
            }

            // --- ScatterAdd src backward: gather(grad, indices, dim) ---
            AdjointExpr::ScatterAddSrcBackward(y_bar, indices, dim) => {
                self.emit_op(PrimalOp::Gather { dim }, vec![y_bar, indices])
            }

            // --- Shape backward ops ---
            AdjointExpr::ConcatSplit(y_bar, dim, offset, size) => self.emit_op(
                PrimalOp::Slice {
                    dim,
                    start: offset as i64,
                    end: (offset + size) as i64,
                    orig_dim_size: 0,
                },
                vec![y_bar],
            ),
            AdjointExpr::SplitConcat(y_bar, _dim) => {
                // Split backward = concat the gradient pieces (handled by accumulate_adjoint)
                self.emit_op(PrimalOp::Reshape { target_ndim: 0 }, vec![y_bar])
            }
            AdjointExpr::SliceBackward(y_bar, dim, start, end, orig_dim_size) => {
                // Slice backward = zero-pad grad into original shape along the sliced dim.
                // pad_before = start (zeros before the slice region)
                // pad_after = orig_dim_size - end (zeros after the slice region)
                let pad_after = orig_dim_size - end;
                self.emit_op(
                    PrimalOp::PadZero {
                        dim,
                        pad_before: start,
                        pad_after,
                    },
                    vec![y_bar],
                )
            }

            // --- Conv/Pool backward ---
            // Conv2d gradients delegate to the runtime `nsl_conv2d_*_backward`
            // FFIs, which wrap the same verified nested-loop `conv2d_backward`
            // the tape path uses — so source-AD conv gradients are byte-identical
            // to the tape gradients. `kind` selects input/weight/bias; inputs are
            // [grad_output, input, weight]. (This replaced an earlier
            // transpose+matmul lowering that was wrong for 4D convolution and was
            // subsequently turned into a hard refusal; it is now implemented.)
            //
            // Conv2d's 2-3 gradients (input/weight/bias) all reify the same
            // `grad` to the conv output shape (e.g. broadcasting a scalar
            // sum-loss gradient). Emit that reification once per (grad, input,
            // weight, stride, padding) tuple and have every sibling gradient
            // consume the shared result, instead of each independently reifying
            // (and, for a scalar gradient, allocating) its own copy. The full
            // tuple — not `grad` alone — is the key so two different-geometry
            // conv nodes sharing one upstream `grad` VarId each get their own
            // correctly shaped materialize, never a collided one.
            AdjointExpr::Conv2dBackward(kind, grad, input, weight, stride, padding) => {
                let key = ConvGradMaterializeKey {
                    grad,
                    input,
                    weight,
                    stride,
                    padding,
                };
                let materialized_grad = if let Some(&v) = self.conv_grad_materialize_cache.get(&key) {
                    v
                } else {
                    let v = self.emit_op(
                        PrimalOp::MaterializeConvOutputGrad { stride, padding },
                        vec![grad, input, weight],
                    );
                    self.conv_grad_materialize_cache.insert(key, v);
                    v
                };
                self.emit_op(
                    PrimalOp::Conv2dBackward {
                        kind,
                        stride,
                        padding,
                    },
                    vec![materialized_grad, input, weight],
                )
            }
            AdjointExpr::MaxPoolBackward(y_bar, indices) => {
                // MaxPool backward: scatter grad to argmax positions
                self.emit_op(PrimalOp::ScatterAdd { dim: 0 }, vec![y_bar, indices])
            }
            AdjointExpr::AvgPoolBackward(y_bar, pool_size) => {
                // Repeat (upsample) each element to cover the pool region,
                // then scale by 1/pool_size to distribute the gradient evenly.
                let repeated = self.emit_op(PrimalOp::Repeat { kernel: pool_size }, vec![y_bar]);
                let scale = self.emit_constant(1.0 / pool_size as f64);
                self.emit_op(PrimalOp::Mul, vec![repeated, scale])
            }

            // --- CrossEntropy backward: (softmax(logits) - one_hot(targets)) * y_bar / N ---
            // Uses dedicated runtime function that handles class-index targets
            // (not one-hot), computes softmax internally, and fuses the /N scaling.
            AdjointExpr::CrossEntropyBackward(y_bar, logits, targets) => self.emit_op(
                PrimalOp::Passthrough("cross_entropy_backward".into()),
                vec![y_bar, logits, targets],
            ),

            // --- MSE backward: d/d(pred_i) = grad_output * 2 * (pred_i - target_i) / N ---
            // Forward `PrimalOp::MSELoss` lowers to a global `mean((pred-target)^2)`
            // (scalar over all N=numel(pred)), so the /N factor must be applied here
            // or the gradient is N× too large (was causing dO inflation in CSHA e2e).
            // Delegated to `nsl_mse_backward` so `N` can be read at runtime from the
            // pred tensor header without plumbing shape info through the Wengert list.
            AdjointExpr::MSEBackward(y_bar, pred, target) => self.emit_op(
                PrimalOp::Passthrough("mse_backward".into()),
                vec![y_bar, pred, target],
            ),

            // --- L1 backward: d/d(pred_i) = grad_output * sign(pred_i - target_i) / N ---
            // Same intercept pattern as MSE: the forward `PrimalOp::L1Loss` lowers to
            // `mean(|pred - target|)` (global mean over N=numel(pred), scalar out), so
            // backward must apply the /N factor or gradients come out N× too large.
            // Delegated to `nsl_l1_backward` so N can be read at runtime from the
            // pred tensor header without plumbing shape info through the Wengert list.
            AdjointExpr::L1Backward(y_bar, pred, target) => self.emit_op(
                PrimalOp::Passthrough("l1_backward".into()),
                vec![y_bar, pred, target],
            ),

            // --- Attention backward: per-component extraction from fused kernel ---
            // Each component (dQ=0, dK=1, dV=2) is extracted via a dedicated op
            // that carries the causal flag so the runtime can apply the correct mask.
            AdjointExpr::AttentionBackwardQ(y_bar, q, k, v, fwd_out, causal) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtract {
                    causal,
                    component: 0,
                },
                vec![y_bar, q, k, v, fwd_out],
            ),
            AdjointExpr::AttentionBackwardK(y_bar, q, k, v, fwd_out, causal) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtract {
                    causal,
                    component: 1,
                },
                vec![y_bar, q, k, v, fwd_out],
            ),
            AdjointExpr::AttentionBackwardV(y_bar, q, k, v, fwd_out, causal) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtract {
                    causal,
                    component: 2,
                },
                vec![y_bar, q, k, v, fwd_out],
            ),

            // PCA Stage C: packed (segment-masked) attention backward — same
            // extract pattern with the segment tensor threaded as a sixth
            // input so the lowering can reach the `_segmask` variant table
            // and the 19-arg backward FFI.
            AdjointExpr::AttentionBackwardQPacked(y_bar, q, k, v, fwd_out, seg, scale) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtractPacked { component: 0 },
                vec![y_bar, q, k, v, fwd_out, seg, scale],
            ),
            AdjointExpr::AttentionBackwardKPacked(y_bar, q, k, v, fwd_out, seg, scale) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtractPacked { component: 1 },
                vec![y_bar, q, k, v, fwd_out, seg, scale],
            ),
            AdjointExpr::AttentionBackwardVPacked(y_bar, q, k, v, fwd_out, seg, scale) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtractPacked { component: 2 },
                vec![y_bar, q, k, v, fwd_out, seg, scale],
            ),

            // CFTP §4.4 G3 (Sprint 4): fused linear-CE backward extract.
            // Lowers to a `FusedLinearCeBackwardExtract` op with the inputs
            // the wengert lowerer expects:
            //   [grad, x, W, bias, targets, fwd_result]
            AdjointExpr::FusedLinearCeBackward {
                grad,
                x,
                w,
                bias,
                targets,
                fwd_result,
                component,
                vocab_size,
                hidden_size,
                batch_size,
                seq_len,
                vocab_tile,
                ignore_index,
            } => self.emit_op(
                PrimalOp::FusedLinearCeBackwardExtract {
                    component,
                    vocab_size,
                    hidden_size,
                    batch_size,
                    seq_len,
                    vocab_tile,
                    ignore_index,
                },
                vec![grad, x, w, bias, targets, fwd_result],
            ),

            // CPKD: fused KL-CE backward extract. Lowers to a
            // `FusedKlCeBackwardExtract` op with the inputs the wengert
            // lowerer expects:
            //   [grad, x_s, W_s, bias_s, x_t, W_t, bias_t, targets, fwd_result]
            AdjointExpr::FusedKlCeBackward {
                grad,
                fwd_inputs,
                fwd_result,
                component,
                vocab_size,
                student_hidden,
                teacher_hidden,
                batch_size,
                seq_len,
                vocab_tile,
                ignore_index,
                alpha_bits,
                temperature_bits,
            } => {
                let mut inputs = Vec::with_capacity(9);
                inputs.push(grad);
                inputs.extend_from_slice(&fwd_inputs);
                inputs.push(fwd_result);
                self.emit_op(
                    PrimalOp::FusedKlCeBackwardExtract {
                        component,
                        vocab_size,
                        student_hidden,
                        teacher_hidden,
                        batch_size,
                        seq_len,
                        vocab_tile,
                        ignore_index,
                        alpha_bits,
                        temperature_bits,
                    },
                    inputs,
                )
            }

            // --- RoPE backward: apply inverse rotation (rotate by -θ) ---
            // RoPE(x, θ) = R(θ)·x, so d/dx = R(θ)^T = R(-θ).
            // Negating the input is WRONG (that's reflection, not inverse rotation).
            // Emit as RoPE with negated sin component.
            AdjointExpr::RoPEBackward(y_bar, dim) => {
                self.emit_op(PrimalOp::RoPEInverse { dim }, vec![y_bar])
            }

            // --- rotate_half backward: -rotate_half(grad) ---
            // forward: y[..h] = -x[h..], y[h..] = x[..h]
            // backward: dx[..h] = dy[h..], dx[h..] = -dy[..h]
            // which equals -rotate_half(dy). Lower as a Passthrough rotate_half
            // followed by Neg.
            AdjointExpr::RotateHalfBackward(y_bar) => {
                let rotated = self.emit_op(
                    PrimalOp::Passthrough("rotate_half".into()),
                    vec![y_bar],
                );
                self.emit_op(PrimalOp::Neg, vec![rotated])
            }
        }
    }

    fn accumulate_adjoint(&mut self, var: VarId, value: VarId) {
        if let Some(&existing) = self.adjoint_vars.get(&var) {
            let sum = self.next_var();
            let op_id = self.next_op();
            self.adjoint_ops.push(WengertOp {
                id: op_id,
                result: sum,
                op: PrimalOp::Add,
                inputs: vec![existing, value],
                saved_for_backward: false,
                checkpointed: false,
            });
            self.adjoint_vars.insert(var, sum);
        } else {
            self.adjoint_vars.insert(var, value);
        }
    }

    /// Get the adjoint variable for a primal variable (after generation).
    pub fn adjoint_of(&self, primal_var: VarId) -> Option<VarId> {
        self.adjoint_vars.get(&primal_var).copied()
    }

    /// Return the VarId of the loss seed (the adjoint of `primal.output`).
    ///
    /// This is the VarId that `generate()` assigns `PrimalOp::Constant(1.0)` to.
    /// Callers that hold an upstream loss gradient (e.g. `dy_handle` from the
    /// L2 backward wrapper — spec §4.2) can pre-populate this VarId in their
    /// `primal_vars` map before calling `compile_wengert_ops`, so the lowerer's
    /// pre-mapped-constant check skips the hardcoded 1.0 and uses the real gradient.
    ///
    /// Returns `None` if `generate()` has not yet been called.
    pub fn loss_seed_var_id(&self, primal_output: VarId) -> Option<VarId> {
        self.adjoint_vars.get(&primal_output).copied()
    }
}

// ---------------------------------------------------------------------------
// Dead gradient elimination
// ---------------------------------------------------------------------------

/// Prune backward ops whose results are never needed by any parameter gradient.
pub fn eliminate_dead_gradients(
    adjoint_ops: &[WengertOp],
    needed_vars: &HashSet<VarId>,
) -> Vec<WengertOp> {
    let mut live_ops = HashSet::new();
    let mut worklist: Vec<VarId> = needed_vars.iter().copied().collect();

    while let Some(var) = worklist.pop() {
        for (i, op) in adjoint_ops.iter().enumerate() {
            if op.result == var && !live_ops.contains(&i) {
                live_ops.insert(i);
                worklist.extend(op.inputs.iter().copied());
            }
        }
    }

    adjoint_ops
        .iter()
        .enumerate()
        .filter(|(i, _)| live_ops.contains(i))
        .map(|(_, op)| op.clone())
        .collect()
}

/// Task 4: WRGA backward-live filter.
///
/// Drop adjoint ops whose **primal** dependency is not in `backward_live`.
/// `backward_live` is the set of *primal* VarIds whose adjoint must be
/// materialised, as computed by `wrga_prune::prune`.  Each adjoint op's
/// `inputs[0]` (the op's seeded adjoint input or primal it differentiates
/// against) must lie on that set, otherwise the adjoint is dead by pruning.
///
/// For ops whose `inputs` are empty (e.g. adjoint constant seeds), keep them;
/// the downstream `eliminate_dead_gradients` pass is the authority on true
/// dead-code elimination.  This pass is coarser: it only removes ops the
/// WRGA prune pass has already proved to be on frozen branches.
pub fn eliminate_by_backward_live(
    ops: &[WengertOp],
    backward_live: &std::collections::BTreeSet<VarId>,
    adjoint_vars: &HashMap<VarId, VarId>,
) -> Vec<WengertOp> {
    let adj_to_primal: HashMap<VarId, VarId> = adjoint_vars
        .iter()
        .map(|(&p, &a)| (a, p))
        .collect();

    let mut kept = Vec::with_capacity(ops.len());
    let mut dropped = 0usize;
    for op in ops {
        match adj_to_primal.get(&op.result) {
            Some(primal) if !backward_live.contains(primal) => {
                dropped += 1;
            }
            _ => kept.push(op.clone()),
        }
    }
    crate::debug_set_adjoint_ops_dropped(dropped);
    kept
}

#[cfg(test)]
mod backward_live_tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::BTreeSet;

    fn mk(id: u32, result: u32, op: PrimalOp, inputs: Vec<u32>) -> WengertOp {
        WengertOp {
            id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    #[test]
    fn retains_only_live_ops() {
        // Three adjoint ops with results 10, 11, 12 corresponding to primals
        // 1, 2, 3 respectively. live = {2}. Adjoint of primal 1 and 3 should
        // be dropped; adjoint of primal 2 kept.
        let ops = vec![
            mk(0, 10, PrimalOp::Neg, vec![1]),
            mk(1, 11, PrimalOp::Neg, vec![2]),
            mk(2, 12, PrimalOp::Neg, vec![3]),
        ];
        let mut live = BTreeSet::new();
        live.insert(2u32);
        let mut adjoint_vars = HashMap::new();
        adjoint_vars.insert(1u32, 10u32);
        adjoint_vars.insert(2u32, 11u32);
        adjoint_vars.insert(3u32, 12u32);
        let kept = eliminate_by_backward_live(&ops, &live, &adjoint_vars);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].result, 11);
    }

    #[test]
    fn keeps_input_less_seed_ops() {
        // Seed ops (result not mapped in adjoint_vars) are always kept — only
        // ops whose result IS mapped to a primal and that primal is not live
        // get dropped.
        let ops = vec![mk(0, 5, PrimalOp::Constant(1.0), vec![])];
        let live: BTreeSet<VarId> = BTreeSet::new();
        let adjoint_vars: HashMap<VarId, VarId> = HashMap::new();
        let kept = eliminate_by_backward_live(&ops, &live, &adjoint_vars);
        assert_eq!(kept.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Saved tensor analysis
// ---------------------------------------------------------------------------

/// Information about a tensor that must be saved from forward for backward.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedTensorInfo {
    pub var: VarId,
    pub checkpointed: bool,
}

/// Identify which forward-pass intermediates must be saved for the backward pass.
pub fn analyze_saved_tensors(primal: &WengertList, adjoint: &WengertList) -> Vec<SavedTensorInfo> {
    let primal_vars: HashSet<VarId> = primal.ops.iter().map(|op| op.result).collect();
    let adjoint_vars: HashSet<VarId> = adjoint.ops.iter().map(|op| op.result).collect();

    let mut saved = Vec::new();
    for adj_op in &adjoint.ops {
        for &input in &adj_op.inputs {
            if primal_vars.contains(&input) && !adjoint_vars.contains(&input) {
                saved.push(SavedTensorInfo {
                    var: input,
                    checkpointed: primal.is_checkpointed(input),
                });
            }
        }
    }

    saved.sort_by_key(|s| s.var);
    saved.dedup_by_key(|s| s.var);
    saved
}

// ---------------------------------------------------------------------------
// M40b: Wengert Extraction from AST
// ---------------------------------------------------------------------------

/// Extracts a WengertList from a sequence of AST statements.
///
/// Returns `Some(WengertList)` if the computation is fully static (no
/// data-dependent control flow). Returns `None` if dynamic control flow
/// is detected, signaling fallback to tape-based AD.
pub struct WengertExtractor<'a> {
    interner: &'a Interner,
    list: WengertList,
    /// Maps AST symbol -> WengertList VarId.
    symbol_to_var: HashMap<nsl_ast::Symbol, VarId>,
    /// Next VarId to allocate.
    next_var: VarId,
    /// Whether this computation graph is fully static.
    is_static: bool,
    /// Symbols that are model parameters (need gradients).
    param_symbols: HashSet<nsl_ast::Symbol>,
    /// Current "self" context prefix for model method inlining (e.g., "m", "m.blocks.0").
    self_context: Option<String>,
    /// Model method bodies: model_type_name -> method_name -> FnDef
    model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
    /// Model field type info: model_type -> field_name -> type_string
    model_field_types: HashMap<String, HashMap<String, String>>,
    /// CFTP v10 (item 5): model field rank info: `model_type -> field_name -> rank`.
    /// Mirrors `Compiler::models.model_field_ranks`; threaded here by
    /// `set_model_field_ranks` from the compile-train-block prologue so
    /// the two model-field `Param` registration sites in `extract_expr`
    /// (`MemberAccess` and pipe-RHS-as-ident) can look up rank BEFORE
    /// pushing a `PrimalOp::Param` and populate `known_ranks` — which
    /// closes the fused-LCE matcher's LATENT 3-D+ RISK on weights
    /// accessed via `self.field` (e.g. an MoE expert stack).
    model_field_ranks: HashMap<String, HashMap<String, usize>>,
    /// Context prefix -> model type name mapping (e.g., "m" -> "NSLCoder", "m.blocks.0" -> "TransformerBlock")
    context_to_model_type: HashMap<String, String>,
    /// Override resolved names for symbols (loop var -> "m.blocks.0")
    symbol_name_overrides: HashMap<nsl_ast::Symbol, String>,
    /// Model instance type for symbols (loop var -> "TransformerBlock")
    model_instance_types: HashMap<nsl_ast::Symbol, String>,
    /// Named parameter VarIds with their compound names (e.g., "m.blocks.0.attn.w_q" -> VarId).
    /// Used for gradient collection since compound params don't have unique AST Symbols.
    named_param_vars: Vec<(String, VarId)>,
    /// WRGA B.3.2 Option 3: the AST rewrite emits synthesized `Call`
    /// nodes whose callee `Ident` carries a sentinel Symbol (e.g., `'w'`
    /// — whatever field symbol the rewriter had on hand). The real FFI
    /// callee name lives in this map, keyed by the Call callee's NodeId.
    /// Parallel to `compiler.synth_call_names` in the Cranelift path.
    synth_call_names: HashMap<nsl_ast::NodeId, String>,
    /// WRGA B.3.2 Option 3: same pattern as `synth_call_names` but for
    /// `MemberAccess` nodes (e.g. the synthesized `self.lora_A_<site>`
    /// accesses whose member Symbol is a sentinel field symbol).
    synth_member_names: HashMap<nsl_ast::NodeId, String>,

    /// Cycle-10 §5.3 Task 5: per-function `@checkpoint(policy=...)` policies
    /// flowed from the semantic layer (`EffectChecker::checkpoint_policies()`)
    /// through the loader wire-up at `crates/nsl-cli/src/loader.rs` (Task 6).
    /// Empty map = no checkpointing = byte-identity preserved at default.
    pub(crate) checkpoint_policies: HashMap<String, CheckpointPolicy>,

    /// Cycle-10 §5.3 Task 5: monotonic allocator for `SubgraphId` values.
    /// Each call to `apply_checkpoint_policy` for a `policy=Full` function
    /// allocates a fresh id and emits exactly one
    /// `PrimalOp::PrologueRecompute { subgraph_id }` marker into the tape.
    pub(crate) next_subgraph_id: u32,
    /// CFTP §4.4 G3 (Sprint 4): the active `@fused_lm_ce(...)` config
    /// for this train block, if any.  When `enabled = true` AND all
    /// shape hints (`vocab_size`, `hidden_size`, `batch_size`, `seq_len`)
    /// are populated, the builtin-recognition pass emits a
    /// `PrimalOp::FusedLinearCe` for matching `fused_linear_ce(...)` calls
    /// instead of stepping into the composite stdlib body.  Otherwise the
    /// composite path is used (preserving v1's safety + the composite-
    /// fallback regression invariant required by Sprint 4 spec).
    fused_ce_config: Option<crate::FusedCeDecoratorConfig>,
    /// CPKD: the active `@fused_kl_ce(...)` config for this distill block
    /// plus the `loss:` section's alpha/temperature for cross-checking the
    /// call-site literals (None outside distill blocks — the builtin arm
    /// then falls through to the stdlib composite, which forces tape AD
    /// and is refused by the distill lowering).
    fused_kl_ce_config: Option<crate::FusedKlCeDecoratorConfig>,
    /// CPKD: (alpha, temperature) from the distill `loss:` section, used
    /// only to refuse a divergent call-site literal (loud consistency
    /// check between the report surface and the kernel constants).
    distill_loss_alpha_temp: (Option<f64>, Option<f64>),
    /// CFTP v10 (item 5): known tensor rank per VarId, when the frontend
    /// could derive it (from an annotated function-parameter type or a
    /// rank-preserving op).  Populated by `register_input_with_rank` /
    /// `register_param_with_rank` when the compiler threads the AST
    /// `TypeExpr` rank through, and by the extractor for rank-preserving
    /// ops emitted downstream (e.g. `Transpose`).
    ///
    /// Consulted by `try_match_fused_linear_ce_pattern` to refuse the
    /// auto-substitution when `W` is known to be non-2D — closes the
    /// LATENT 3-D+ RISK documented on the matcher (rank-3 W stack of a
    /// per-expert MoE layout that would otherwise silently produce
    /// wrong forward logits + wrong dW gradients with no diagnostic).
    ///
    /// Absent-key semantics: the matcher CANNOT prove non-2D from the
    /// current information, so it falls through to the pre-v10 behaviour
    /// (fire).  This preserves backwards compatibility with tests that
    /// use unannotated `Tensor` params while structurally rejecting
    /// programs that DID annotate a non-2D `W`.
    known_ranks: HashMap<VarId, usize>,
    /// CFTP §4.4 G3 (Sprint v3-1, review Finding 1): each successful
    /// `cross_entropy → FusedLinearCe` auto-substitution records its
    /// dead upstream chain (`Transpose → Matmul → Add`) here.  The
    /// prune itself runs at `finalize()` time — by then the entire
    /// tape is complete, so the "is this VarId still consumed by
    /// anything else?" check is exact (no false negatives from
    /// substitution-time scans that haven't yet seen later uses).
    pending_fused_lce_prunes: Vec<FusedLceMatch>,
    /// CPKD (I-11): variable roots whose model fields are FROZEN — the
    /// distill block's teacher instance.  A model-field access whose
    /// compound name is rooted at one of these (e.g. `teacher.wq`,
    /// `teacher.blocks.0.attn.wk`) registers as a `PrimalOp::Input`
    /// leaf instead of a `Param`.  No adjoint is ever generated for an
    /// Input, so the teacher's backward is structurally absent from the
    /// tape (composition-paper invariant I-11 / failure mode F-06: no
    /// teacher gradient buffers, no teacher weight drift).
    frozen_model_roots: HashSet<String>,
    /// CPKD: compound names + VarIds of frozen model-field Input leaves
    /// (parallel to `named_param_vars` but explicitly non-trainable).
    /// The distill lowering resolves these to Cranelift values via
    /// `load_source_ad_named_param` so the teacher forward can read its
    /// weights; they are never handed to the optimizer or gradient
    /// collection.
    frozen_input_vars: Vec<(String, VarId)>,
    /// Dev-tools profile capture: producing AST node per result VarId,
    /// recorded by the `extract_expr` wrapper. `entry().or_insert` keeps
    /// the INNERMOST (producing) expression when an outer expression
    /// passes a var through unchanged (e.g. an ident reference). Used by
    /// `profiling::captures::size_hints_from_var_nodes` to resolve
    /// concrete typed shapes into byte sizes; never read by lowering.
    var_nodes: HashMap<VarId, nsl_ast::NodeId>,
}

/// CFTP §4.4 G3 (Sprint v3-1, review Finding 1):
/// matcher output for `try_match_fused_linear_ce_pattern`.
///
/// Carries both the substitution inputs (`x_var`, `w_var`,
/// `bias_var`) AND the three upstream op result VarIds that produced
/// the now-dead `Transpose(W) → Matmul(x, W^T) → Add(matmul, bias)`
/// chain.  After the substitution emits a `PrimalOp::FusedLinearCe`,
/// `prune_fused_lce_dead_chain` uses the three result-VarIds to
/// surgically remove the upstream ops — the lowerer has no DCE and
/// would otherwise physically run the dead chain, allocating + freeing
/// a `[B*S, V]` tensor per training step (1.5 GB at the v3-2 fp16
/// fixture shape).
#[derive(Debug, Clone, Copy)]
struct FusedLceMatch {
    x_var: VarId,
    w_var: VarId,
    bias_var: VarId,
    transpose_result_var: VarId,
    matmul_result_var: VarId,
    add_result_var: VarId,
}

impl<'a> WengertExtractor<'a> {
    pub fn new(interner: &'a Interner) -> Self {
        WengertExtractor {
            interner,
            list: WengertList {
                ops: Vec::new(),
                output: 0,
                var_names: HashMap::new(),
                var_types: HashMap::new(),
            },
            symbol_to_var: HashMap::new(),
            next_var: 0,
            is_static: true,
            param_symbols: HashSet::new(),
            self_context: None,
            model_method_bodies: HashMap::new(),
            model_field_types: HashMap::new(),
            model_field_ranks: HashMap::new(),
            context_to_model_type: HashMap::new(),
            symbol_name_overrides: HashMap::new(),
            model_instance_types: HashMap::new(),
            named_param_vars: Vec::new(),
            synth_call_names: HashMap::new(),
            synth_member_names: HashMap::new(),
            checkpoint_policies: HashMap::new(),
            next_subgraph_id: 0,
            fused_ce_config: None,
            known_ranks: HashMap::new(),
            pending_fused_lce_prunes: Vec::new(),
            fused_kl_ce_config: None,
            distill_loss_alpha_temp: (None, None),
            frozen_model_roots: HashSet::new(),
            frozen_input_vars: Vec::new(),
            var_nodes: HashMap::new(),
        }
    }

    /// Cycle-10 §5.3 Task 5: builder-style installer for per-function
    /// `@checkpoint(policy=...)` policies. Sibling to the existing
    /// `set_model_method_bodies` / `set_model_field_types` / `set_synth_*`
    /// installers. Wire-up flows from `EffectChecker::checkpoint_policies()`
    /// through `crates/nsl-cli/src/loader.rs` (Task 6).
    ///
    /// Empty map = no checkpointing transformations = byte-identity preserved
    /// at default. Non-empty map activates `apply_checkpoint_policy` after
    /// extraction to perform transitive stamping + `PrologueRecompute` emission.
    pub fn with_checkpoint_policies(
        mut self,
        policies: HashMap<String, CheckpointPolicy>,
    ) -> Self {
        self.checkpoint_policies = policies;
        self
    }

    /// Cycle-10 §5.3 Task 5: allocate a fresh `SubgraphId` for a recompute
    /// subgraph. v1 is singleton-per-`@checkpoint`-fn so a single allocation
    /// per call site suffices. Monotonically increments for tests + multi-fn
    /// compile units.
    pub(crate) fn alloc_subgraph_id(&mut self) -> SubgraphId {
        let id = self.next_subgraph_id;
        self.next_subgraph_id = self.next_subgraph_id.saturating_add(1);
        SubgraphId(id)
    }

    /// Cycle-10 §5.3 Task 5: transitive stamping pass.
    ///
    /// Caller is the per-fn extraction site (`stmt.rs` grad-block or
    /// `binary_codegen.rs` forward-fn). Pre: `extract_stmts(fn_body)` ran
    /// successfully on the extractor.
    ///
    /// Behavior:
    ///   - If `checkpoint_policies` does not contain `fn_name`, no-op.
    ///     This is the byte-identity branch.
    ///   - If keyed with `CheckpointPolicy::Full`, walk the entire current
    ///     tape and stamp every previously-emitted `WengertOp.checkpointed = true`
    ///     (the prologue subgraph in v1 = everything upstream of the
    ///     persisted attention residency). v1 prologue is everything in
    ///     scope because the @checkpoint fn body IS the prologue +
    ///     attention chain, and only forward-input `Input` / `Param` ops
    ///     are exempt (they are entry-of-function args, not recomputed).
    ///   - Allocate a fresh `SubgraphId` via `alloc_subgraph_id` and emit
    ///     exactly one `PrimalOp::PrologueRecompute { subgraph_id }` marker
    ///     onto the tape at the boundary point (the current tape tail).
    ///     The codegen-side dispatch fork at Task 9 consumes this marker.
    ///
    /// W13 invariant: cos/sin saves channel lives at the PTX-param layer
    /// (`flash_attention.rs:1138-1142` cos_ptr/sin_ptr), NOT on
    /// `CshaSavePointers`. This pass therefore does NOT touch
    /// `csha_apply.rs:365-382`. Wengert-level suppression here applies
    /// only to RoPE primal-op intermediates (mul/sub/add/tensor_cat) by
    /// virtue of stamping `checkpointed=true` — the lowerer reads this
    /// flag to skip save-pointer emission.
    pub fn apply_checkpoint_policy(&mut self, fn_name: &str) {
        let Some(policy) = self.checkpoint_policies.get(fn_name).copied() else {
            return;
        };
        match policy {
            // CCR P1.b: Selective is consumed by the train-block CCR gate
            // in stmt.rs (block-granular recompute with matmul outputs
            // saved) — it must NOT trigger the CSHA prologue stamping,
            // which would suppress save-pointer emission for a policy
            // whose whole point is keeping the expensive saves.
            CheckpointPolicy::Selective => {}
            CheckpointPolicy::Full => {
                // Stamp every previously-emitted op as checkpointed EXCEPT
                // function-entry roots (Input / Param). The roots are not
                // "recomputed" — they are the args the caller passed in.
                for op in self.list.ops.iter_mut() {
                    match op.op {
                        PrimalOp::Input(_) | PrimalOp::Param(_) => {}
                        _ => op.checkpointed = true,
                    }
                }

                // Emit exactly one PrologueRecompute marker at the boundary.
                // alloc_subgraph_id() returns a fresh, monotonic id.
                let subgraph_id = self.alloc_subgraph_id();
                let marker_var = self.alloc_var();
                let op_id = self.list.ops.len() as u32;
                self.push_op(WengertOp {
                    id: op_id,
                    result: marker_var,
                    op: PrimalOp::PrologueRecompute { subgraph_id },
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: true,
                });
            }
        }
    }

    /// CFTP §4.4 G3 (Sprint 4): builder method for plumbing the
    /// `@fused_lm_ce` decorator config into builtin recognition.  Callers
    /// (compiler/train-block lowering) thread
    /// `compiler.active_fused_ce_config` (CFTP v10 item 3; pre-v10:
    /// `compiler.fused_ce_configs[0]`) here when the active train block
    /// carries the decorator and v1's opt-in gate is satisfied.
    pub fn with_fused_ce_config(
        mut self,
        cfg: Option<crate::FusedCeDecoratorConfig>,
    ) -> Self {
        self.fused_ce_config = cfg;
        self
    }

    /// CPKD: install the active `@fused_kl_ce` decorator config plus the
    /// distill `loss:` section's (alpha, temperature) for the call-site
    /// literal cross-check.
    pub fn with_fused_kl_ce_config(
        mut self,
        cfg: Option<crate::FusedKlCeDecoratorConfig>,
        loss_alpha: Option<f64>,
        loss_temperature: Option<f64>,
    ) -> Self {
        self.fused_kl_ce_config = cfg;
        self.distill_loss_alpha_temp = (loss_alpha, loss_temperature);
        self
    }

    fn alloc_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    /// Push a WengertOp and auto-tag its result VarId with the correct WengertType.
    fn push_op(&mut self, op: WengertOp) {
        let ty = type_for_op(&op.op);
        self.list.var_types.insert(op.result, ty);
        self.list.ops.push(op);
    }

    /// CFTP §4.4 G3 (Sprint v3-1) — auto-substitution pattern matcher.
    ///
    /// Walks the Wengert list backwards from `logits_var` looking for the
    /// canonical stdlib `fused_linear_ce` decomposition:
    ///
    /// ```text
    ///   logits_var = Add(Matmul(x, Transpose(W, 0, 1)), bias)
    /// ```
    ///
    /// On match, returns `Some((x_var, W_var, bias_var))`. Returns `None` on
    /// any structural mismatch (different op chain, missing transpose,
    /// missing bias add, producers not present in `self.list`, etc.).
    ///
    /// The caller (the `"cross_entropy"` builtin recognition arm) checks
    /// `Some(_)` together with `fused_ce_config.enabled` AND fully-populated
    /// shape hints before emitting a `PrimalOp::FusedLinearCe` substitution.
    /// In all other cases the caller falls through to the standard
    /// `PrimalOp::CrossEntropyLoss` lowering — preserving the regression
    /// invariant that programs without `@fused_lm_ce` keep the composite path.
    ///
    /// Edge cases handled:
    ///   * `bias` is required by v1's fused FFI signature, so a bare
    ///     `Matmul(x, Transpose(W))` (no add) does NOT match.  Sprint v4-3
    ///     deferral: a bias-free PTX variant is feasible but would require
    ///     emitter forks (`emit_fwd_kernel_no_bias` + `emit_bwd_kernel_no_bias`
    ///     etc. — comparable LOC to the bf16 sprint).  A cheaper host-side
    ///     workaround (caller allocates+zeros a `[V]` scratch buffer) was
    ///     considered for v4-3 but not landed because the upstream
    ///     decorator/wengert path has no hook to inject scratch alloc
    ///     today; tracked for v5+.
    ///   * `Transpose` must be over the last two dims of a 2D weight tensor.
    ///     Sprint v4-3 widens this beyond the strict stdlib
    ///     `transpose(W, 0, 1)` rewrite — `transpose(W, 1, 0)` and the
    ///     negative-dim form `transpose(W, -2, -1)` (encoded via
    ///     `encode_transpose_dim` as `(usize::MAX - 1, usize::MAX)`) and its
    ///     swap `transpose(W, -1, -2)` are now all accepted.  All three
    ///     forms are semantically identical on a 2D tensor — the previous
    ///     refusal was over-strict and forced idiomatic NSL code (which
    ///     prefers negative dims for rank-agnostic transposes) onto the
    ///     slower composite path.
    ///
    ///     LATENT 3-D+ RISK (adversarial review Findings 1 + 8): the matcher
    ///     does NOT structurally verify that `W` is rank-2 — it accepts
    ///     `W` via the transpose's first input slot regardless of declared
    ///     rank.  Every PTX emitter indexes `W` as `W[v*H + h]` (rank-2
    ///     `[V, H]`); a user who wrote `matmul(x, transpose(W3D, -2, -1))
    ///     + bias` over a rank-3 `W3D` (e.g. an MoE expert stack `[D, V, H]`
    ///     or a head-major `[heads, V, H]` layout) would have the matcher
    ///     silently fire, the fused FFI stride through `W3D` as if it were
    ///     `[V, H]`, and produce wrong forward logits + wrong dW gradients
    ///     with no diagnostic.
    ///
    ///     RESOLVED (v10, PR #294): structural rank enforcement now exists
    ///     for known-rank `W` — the matcher consults the `known_ranks`
    ///     table (populated via `register_input_with_rank` /
    ///     `register_param_with_rank` and, for model-field weights,
    ///     `set_model_field_ranks` + `resolvable_tensor_rank`) and returns
    ///     `None` when `W`'s declared rank is not exactly 2 (guard at the
    ///     `known_ranks.get(&w_var)` check below).  Only the
    ///     unknown/absent-shape-hint path remains conservative-fire; the
    ///     defensive layers there are the upstream type system, the
    ///     decorator's `(vocab_size, hidden_size)` shape contract, and the
    ///     negative-test coverage in
    ///     `fused_linear_ce_auto_substitution.rs` (rank-3 W → no
    ///     substitution).
    ///   * No-transpose W layout (`Matmul(x, W)` with W already `[H, V]`)
    ///     is NOT accepted.  Sprint v4-3 deferral: the runtime FFI at
    ///     `crates/nsl-runtime/src/fused_linear_ce.rs` (`w_ptr` doc) and
    ///     every PTX emitter indexes W as `[V, H]` (`W[v*H + h]`).  An
    ///     `[H, V]` path needs a new emitter variant + new FFI symbol
    ///     (`nsl_fused_linear_ce_forward_w_hv` or similar) — comparable
    ///     LOC to the bf16 sprint; tracked for v5+.
    ///   * Add operands may appear in either order:
    ///     `Add(Matmul(...), bias)` and `Add(bias, Matmul(...))` both match.
    ///
    /// Returns `FusedLceMatch` carrying both the substitution inputs
    /// (`x_var`, `w_var`, `bias_var`) AND the three upstream op result
    /// VarIds (`logits_var` = add result, `matmul_result_var`,
    /// `w_transposed_var` = transpose result).  The caller uses the
    /// upstream VarIds to prune the now-dead composite chain from
    /// `self.list.ops` — without pruning, the lowerer (which iterates
    /// every op with no DCE — see `wengert_lower.rs:72-94`) would
    /// physically run the transpose + matmul + add producing a
    /// `[B*S, V]` tensor that nothing consumes, wasting ~1.5 GB at
    /// V=49152/H=4096/B=2/S=4096 per training step (review Finding 1).
    fn try_match_fused_linear_ce_pattern(
        &self,
        logits_var: VarId,
    ) -> Option<FusedLceMatch> {
        let add_op = self.list.find_producer(logits_var)?;
        if !matches!(add_op.op, PrimalOp::Add) {
            return None;
        }
        if add_op.inputs.len() != 2 {
            return None;
        }
        // Inspect both Add operands — Matmul may be on either side.
        let lhs = add_op.inputs[0];
        let rhs = add_op.inputs[1];
        let lhs_op = self.list.find_producer(lhs);
        let rhs_op = self.list.find_producer(rhs);
        let (matmul_op, bias_var) = match (
            lhs_op.map(|o| &o.op),
            rhs_op.map(|o| &o.op),
        ) {
            (Some(PrimalOp::Matmul), _) => (lhs_op?, rhs),
            (_, Some(PrimalOp::Matmul)) => (rhs_op?, lhs),
            _ => return None,
        };
        // Review Finding 3: reject `Add(Matmul, Matmul)` and other
        // patterns where the "bias" slot would receive a high-rank
        // tensor.  The arm above picks the FIRST Matmul-producing
        // operand as the matmul leg and treats the OTHER Add operand
        // as `bias_var`.  If `bias_var` is itself the output of a
        // Matmul (e.g. `matmul(x1, W1^T) + matmul(x2, W2^T)`), the
        // substituted fused FFI will dereference its bias_ptr as a
        // dense `[V]` vector — reading garbage from a `[B*S, V]`
        // matmul intermediate.  Same hazard for
        // `ScaledDotProductAttention` (high-rank output) and `Conv2d`
        // (if/when it appears).  Refuse the substitution and fall
        // through to the composite path.
        if let Some(bias_producer) = self.list.find_producer(bias_var) {
            if matches!(
                bias_producer.op,
                PrimalOp::Matmul | PrimalOp::ScaledDotProductAttention { .. }
            ) {
                return None;
            }
        }
        if matmul_op.inputs.len() != 2 {
            return None;
        }
        let matmul_result_var = matmul_op.result;
        let x_var = matmul_op.inputs[0];
        let w_transposed_var = matmul_op.inputs[1];
        let transpose_op = self.list.find_producer(w_transposed_var)?;
        // Sprint v4-3: accept any of the four semantically-equivalent
        // last-two-dim transposes of a 2D weight tensor:
        //   * `transpose(W, 0, 1)` → `{0, 1}` (stdlib form)
        //   * `transpose(W, 1, 0)` → `{1, 0}` (operand swap; same matrix)
        //   * `transpose(W, -2, -1)` → `{usize::MAX - 1, usize::MAX}`
        //     (negative-dim encoding from `encode_transpose_dim`)
        //   * `transpose(W, -1, -2)` → `{usize::MAX, usize::MAX - 1}` (swap)
        //
        // These four forms are bit-identical on a 2D tensor.  The
        // pre-v4-3 matcher accepted only the first form, forcing
        // idiomatic NSL code (which prefers negative-dim transposes for
        // rank-agnostic code, see `source_ad.rs` lines 556-560/597-598/
        // 656-672/746-761 where the AD rules themselves emit
        // `{usize::MAX - 1, usize::MAX}`) onto the slower composite
        // path.  Higher-dim transposes (e.g. `transpose(W, 0, 2)`) are
        // STILL refused — those reshape a >2D tensor in ways that the
        // fused FFI's `[V, H]` indexing cannot honour.
        const NEG_MINUS_2: usize = usize::MAX - 1;
        const NEG_MINUS_1: usize = usize::MAX;
        match transpose_op.op {
            PrimalOp::Transpose { dim0: 0, dim1: 1 }
            | PrimalOp::Transpose { dim0: 1, dim1: 0 }
            | PrimalOp::Transpose {
                dim0: NEG_MINUS_2,
                dim1: NEG_MINUS_1,
            }
            | PrimalOp::Transpose {
                dim0: NEG_MINUS_1,
                dim1: NEG_MINUS_2,
            } => {}
            _ => return None,
        }
        // `transpose(W, 0, 1)` is recognised in `extract_expr` with inputs
        // `[W, dim0_const, dim1_const]` (the two dim args get folded to
        // `PrimalOp::Constant` VarIds even though they aren't used by AD).
        // Pull W from the first input slot; reject if missing.
        if transpose_op.inputs.is_empty() {
            return None;
        }
        let w_var = transpose_op.inputs[0];
        // CFTP v10 (item 5): structural rank-2 enforcement.
        //
        // When the compiler was able to derive the rank of `w_var` from the
        // enclosing AST type annotation (via
        // `register_input_with_rank` / `register_param_with_rank`) OR
        // propagated the rank through a rank-preserving op (Transpose),
        // consult it.  A `known_ranks` entry != 2 means the matcher would
        // otherwise silently fire on a rank-3 W3D (e.g. an MoE expert
        // stack `[D, V, H]`), stride through it as if it were `[V, H]`,
        // and produce wrong forward logits + wrong dW gradients with no
        // diagnostic (documented LATENT 3-D+ RISK).  Refuse the
        // substitution so the composite CE path takes over.
        //
        // Absent-key semantics: we can't PROVE non-2D, so we conservatively
        // fire — matching the pre-v10 behaviour for unannotated
        // `Tensor` params.
        if let Some(&r) = self.known_ranks.get(&w_var) {
            if r != 2 {
                return None;
            }
        }
        Some(FusedLceMatch {
            x_var,
            w_var,
            bias_var,
            transpose_result_var: w_transposed_var,
            matmul_result_var,
            add_result_var: logits_var,
        })
    }

    /// Review Finding 1: prune the upstream `Transpose → Matmul → Add`
    /// chain that produced `logits_var` after auto-substitution succeeds.
    ///
    /// Called by `finalize()` once the entire tape is built, so the
    /// "is this VarId consumed elsewhere?" check is exact — every op
    /// that will ever be pushed has already been pushed.  We deliberately
    /// do NOT consult `self.symbol_to_var`: symbol-bound VarIds that no
    /// op actually consumes are still dead (Cranelift lowering iterates
    /// `list.ops`, not the symbol map).
    ///
    /// Strategy: for each of the three chain ops, surgically remove it
    /// from `list.ops` IF its result VarId has no consumers outside the
    /// chain itself AND isn't the function output.
    ///
    /// Returns the number of ops actually removed.  Conservative on
    /// ambiguity (leaves the op in place) — correctness over pruning.
    fn prune_fused_lce_dead_chain(list: &mut WengertList, m: &FusedLceMatch) -> usize {
        // The three op-result VarIds we'd like to remove.  In-chain
        // consumers don't block removal (Matmul reads transpose, Add
        // reads matmul) since they themselves are about to be removed.
        let chain_results = [
            m.transpose_result_var,
            m.matmul_result_var,
            m.add_result_var,
        ];

        let is_dead = |var: VarId, list: &WengertList| -> bool {
            // Never remove the function output.
            if list.output == var {
                return false;
            }
            // Scan all ops; a consumer that's NOT itself in the chain
            // blocks removal.
            for op in &list.ops {
                if chain_results.contains(&op.result) {
                    continue;
                }
                if op.inputs.contains(&var) {
                    return false;
                }
            }
            true
        };

        let mut removed = 0usize;
        for target in chain_results {
            if !is_dead(target, list) {
                continue;
            }
            // Remove the op producing `target`.  Linear scan is O(N)
            // but we run this at most once per cross_entropy
            // substitution; N is bounded by the per-fn op count.
            let before = list.ops.len();
            list.ops.retain(|op| op.result != target);
            if list.ops.len() < before {
                removed += 1;
                // Leave `var_types` and `var_names` entries in place —
                // they're lookup-only and harmless once the producing
                // op is gone.  Removing them would risk breaking
                // downstream type queries on still-live VarIds that
                // happen to share an entry.
            }
        }
        removed
    }

    fn extract_int_literal(expr: &nsl_ast::expr::Expr) -> Option<i64> {
        match &expr.kind {
            ExprKind::IntLiteral(v) => Some(*v),
            ExprKind::UnaryOp {
                op: AstUnaryOp::Neg,
                operand,
            } => match &operand.kind {
                ExprKind::IntLiteral(v) => Some(-*v),
                _ => None,
            },
            _ => None,
        }
    }

    fn encode_transpose_dim(dim: i64) -> Option<usize> {
        match dim {
            -2 => Some(usize::MAX - 1),
            -1 => Some(usize::MAX),
            value if value >= 0 => Some(value as usize),
            _ => None,
        }
    }

    fn extract_transpose_dims(args: &[nsl_ast::expr::Arg]) -> Option<(usize, usize)> {
        let dim0 = args
            .first()
            .and_then(|arg| Self::extract_int_literal(&arg.value))?;
        let dim1 = args
            .get(1)
            .and_then(|arg| Self::extract_int_literal(&arg.value))?;
        Some((
            Self::encode_transpose_dim(dim0)?,
            Self::encode_transpose_dim(dim1)?,
        ))
    }

    /// Register a parameter symbol (needs gradient).
    pub fn register_param(&mut self, sym: nsl_ast::Symbol) {
        self.register_param_with_rank(sym, None);
    }

    /// CFTP v10 (item 5): register a parameter symbol AND record its
    /// declared tensor rank when the compiler can derive it from the
    /// enclosing AST type annotation.  A `Some(2)` here is what unlocks
    /// the rank-2 acceptance path in
    /// `try_match_fused_linear_ce_pattern`; a `Some(3+)` triggers the
    /// structural refusal.  Pass `None` when rank is unknown to preserve
    /// the pre-v10 behaviour (matcher fires).
    pub fn register_param_with_rank(
        &mut self,
        sym: nsl_ast::Symbol,
        rank: Option<usize>,
    ) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);
        self.param_symbols.insert(sym);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        if let Some(r) = rank {
            self.known_ranks.insert(var, r);
        }
        self.push_op(WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: PrimalOp::Param(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Register an input symbol (data, no gradient).
    pub fn register_input(&mut self, sym: nsl_ast::Symbol) {
        self.register_input_with_rank(sym, None);
    }

    /// CFTP v10 (item 5): register an input symbol AND record its
    /// declared tensor rank when the compiler can derive it from the
    /// enclosing AST type annotation.  See
    /// [`register_param_with_rank`] for the load-bearing invariant this
    /// unlocks.
    pub fn register_input_with_rank(
        &mut self,
        sym: nsl_ast::Symbol,
        rank: Option<usize>,
    ) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        if let Some(r) = rank {
            self.known_ranks.insert(var, r);
        }
        self.push_op(WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: PrimalOp::Input(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Set model method bodies for inline expansion during extraction.
    /// Maps model_type_name -> method_name -> FnDef.
    pub fn set_model_method_bodies(
        &mut self,
        bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
    ) {
        self.model_method_bodies = bodies;
    }

    /// Set model field type info for for-loop unrolling.
    /// Maps model_type -> field_name -> type_string (e.g., "[TransformerBlock;8]").
    pub fn set_model_field_types(&mut self, types: HashMap<String, HashMap<String, String>>) {
        self.model_field_types = types;
    }

    /// CFTP v10 (item 5): install per-model-field rank info so the two
    /// model-field `Param` registration sites in `extract_expr` can
    /// populate `known_ranks` — closing the LATENT 3-D+ RISK for W
    /// operands loaded via `self.field`.  Maps
    /// `model_type -> field_name -> rank`.
    pub fn set_model_field_ranks(
        &mut self,
        ranks: HashMap<String, HashMap<String, usize>>,
    ) {
        self.model_field_ranks = ranks;
    }

    /// WRGA B.3.2 Option 3: install the compiler's synth_call_names map so
    /// the Call extractor can resolve sentinel-Ident callees back to their
    /// real FFI name (e.g. `nsl_adapter_fused_gatedlora_matmul`).
    pub fn set_synth_call_names(&mut self, names: HashMap<nsl_ast::NodeId, String>) {
        self.synth_call_names = names;
    }

    /// WRGA B.3.2 Option 3: install the compiler's synth_member_names map
    /// so MemberAccess extraction resolves sentinel field symbols back to
    /// their real names (e.g. `self.lora_A_<site>`).
    pub fn set_synth_member_names(&mut self, names: HashMap<nsl_ast::NodeId, String>) {
        self.synth_member_names = names;
    }

    /// Register a model instance: maps a variable symbol to a model type name
    /// and sets up the context-to-type mapping for method inlining.
    pub fn register_model_instance(&mut self, sym: nsl_ast::Symbol, model_type: &str) {
        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.model_instance_types
            .insert(sym, model_type.to_string());
        self.context_to_model_type
            .insert(name, model_type.to_string());
    }

    /// Set the current self-context string directly.
    ///
    /// Required when extracting a model method body directly (i.e., not via
    /// method inlining from an outer training loop).  Call this with the name
    /// of the `self` parameter (typically `"self"`) after calling
    /// `register_model_instance`, so that `ExprKind::SelfRef` and
    /// `ExprKind::Pipe` with bare field-name RHS can be resolved correctly.
    pub fn set_self_context(&mut self, ctx: Option<String>) {
        self.self_context = ctx;
    }

    /// CPKD (I-11): mark variable roots (e.g. the distill teacher instance
    /// name) whose model fields must register as non-trainable `Input`
    /// leaves.  Must be called BEFORE `extract_stmts` — freezing is applied
    /// at field-registration time, not retroactively.
    pub fn set_frozen_model_roots(&mut self, roots: HashSet<String>) {
        self.frozen_model_roots = roots;
    }

    /// CPKD: compound-name/VarId pairs for frozen model-field Input leaves
    /// (teacher weights).  The distill lowering resolves each to a Cranelift
    /// value; they never reach the optimizer.
    pub fn frozen_input_var_ids(&self) -> &[(String, VarId)] {
        &self.frozen_input_vars
    }

    /// CPKD: true when a model-field compound name (`teacher.wq`,
    /// `teacher.blocks.0.attn.wk`) is rooted at a frozen model instance.
    fn is_frozen_compound(&self, compound: &str) -> bool {
        if self.frozen_model_roots.is_empty() {
            return false;
        }
        let root = compound.split('.').next().unwrap_or("");
        self.frozen_model_roots.contains(root)
    }

    /// Get named parameter VarIds with their compound names.
    /// Returns (compound_name, VarId) pairs for gradient collection.
    pub fn named_param_var_ids(&self) -> &[(String, VarId)] {
        &self.named_param_vars
    }

    /// Extract statements into the Wengert list.
    /// Returns false if dynamic control flow is detected.
    pub fn extract_stmts(&mut self, stmts: &[nsl_ast::stmt::Stmt]) -> bool {
        for stmt in stmts {
            if !self.extract_stmt(stmt) {
                self.is_static = false;
                eprintln!(
                    "[source-ad] extraction failed at {:?} (line {:?})",
                    std::mem::discriminant(&stmt.kind),
                    stmt.span
                );
                return false;
            }
        }
        true
    }

    fn extract_stmt(&mut self, stmt: &nsl_ast::stmt::Stmt) -> bool {
        match &stmt.kind {
            StmtKind::VarDecl {
                pattern,
                value: Some(val),
                ..
            } => {
                let _var_name = if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    self.interner.resolve(sym.0).unwrap_or("?").to_string()
                } else {
                    "?".to_string()
                };
                if let Some(var) = self.extract_expr(val) {
                    if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                        self.symbol_to_var.insert(*sym, var);
                        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
                        self.list.var_names.insert(var, name);
                    }
                    true
                } else {
                    eprintln!(
                        "[source-ad] VarDecl '{}' extraction failed at expr {:?}",
                        _var_name,
                        std::mem::discriminant(&val.kind)
                    );
                    false
                }
            }

            StmtKind::Return(Some(expr)) => {
                if let Some(var) = self.extract_expr(expr) {
                    self.list.output = var;
                    true
                } else {
                    false
                }
            }

            StmtKind::Expr(expr) => self.extract_expr(expr).is_some(),

            // Dynamic control flow -> not static
            StmtKind::While { .. } => {
                self.is_static = false;
                false
            }

            // For loops: try to unroll if iterating over a model's FixedArray field
            StmtKind::For {
                pattern,
                iterable,
                body,
            } => self.try_unroll_for(pattern, iterable, body),

            // If/else: extract both branches, emit Select ops for variables
            // that differ between branches. The condition is saved for backward.
            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => self.extract_if(condition, then_block, elif_clauses, else_block.as_ref()),

            // Assignment: x = expr (rebind variable to new value)
            StmtKind::Assign { target, op, value } => {
                if let nsl_ast::operator::AssignOp::Assign = op {
                    if let nsl_ast::expr::ExprKind::Ident(sym) = &target.kind {
                        if let Some(var) = self.extract_expr(value) {
                            self.symbol_to_var.insert(*sym, var);
                            return true;
                        }
                        return false;
                    }
                }
                true
            }

            // Other statements pass through (decorated, etc.)
            _ => true,
        }
    }

    /// Dev-tools profile capture: read-only view of the producing AST node
    /// per result VarId (see the `var_nodes` field doc).
    pub fn var_nodes(&self) -> &HashMap<VarId, nsl_ast::NodeId> {
        &self.var_nodes
    }

    /// Extract an expression into a WengertOp, returning its VarId.
    /// Returns None if the expression contains dynamic control flow.
    ///
    /// Thin wrapper over `extract_expr_inner` that additionally attributes
    /// each result VarId to the AST expression that produced it. `or_insert`
    /// keeps the first (innermost/producing) attribution when outer
    /// expressions return an already-mapped var unchanged. Recursive
    /// extraction calls go through this wrapper, so every subexpression is
    /// attributed.
    fn extract_expr(&mut self, expr: &nsl_ast::expr::Expr) -> Option<VarId> {
        let result = self.extract_expr_inner(expr);
        if let Some(var) = result {
            self.var_nodes.entry(var).or_insert(expr.id);
        }
        result
    }

    /// Core expression extraction (formerly `extract_expr`; the attribution
    /// wrapper above now owns that name). Returns None on dynamic control
    /// flow.
    fn extract_expr_inner(&mut self, expr: &nsl_ast::expr::Expr) -> Option<VarId> {
        match &expr.kind {
            ExprKind::Ident(sym) => self.symbol_to_var.get(sym).copied(),

            // Field access: self.w, m.w, block.attn → treat as a parameter reference
            ExprKind::MemberAccess { object, member } => {
                // G6.1.2 — paper §3.2 closure: pre-register all nested
                // model contexts implied by this MemberAccess chain
                // BEFORE computing `is_model_param` below.  Top-down
                // extraction otherwise computes the classification
                // before recursing into the inner object, so
                // `self.blocks[0].attn.wq` would check
                // `is_model_param("m.blocks.0.attn")` against an empty
                // map, fall to the data-access branch, then only
                // register "m.blocks.0" once it recurses all the way
                // down — too late to fix the classification of the
                // ancestor frames.  The pre-pass walks bottom-up so all
                // intermediate contexts are in place before the
                // classification check.
                self.preregister_chain_contexts(expr);

                // WRGA B.3.2 Option 3: synthesized adapter accesses carry a
                // sentinel member Symbol; the real field name lives in
                // synth_member_names. Consult it FIRST so we recover e.g.
                // `lora_A_<site>` instead of the sentinel 'w'.
                let synth_name = self.synth_member_names.get(&expr.id).cloned();
                let member_name_owned = synth_name.unwrap_or_else(|| {
                    self.interner.resolve(member.0).unwrap_or("?").to_string()
                });
                let member_name = member_name_owned.as_str();
                if member_name == "shape" || member_name == "ndim" {
                    if let Some(obj_var) = self.extract_expr(object) {
                        let result = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result,
                            op: PrimalOp::Passthrough(member_name.to_string()),
                            inputs: vec![obj_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(result);
                    }
                }

                // Resolve the object prefix to a string
                let obj_prefix = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(obj_sym) => self
                        .symbol_name_overrides
                        .get(obj_sym)
                        .cloned()
                        .or_else(|| {
                            Some(self.interner.resolve(obj_sym.0).unwrap_or("?").to_string())
                        }),
                    // Nested member access: self.attn.w_q -> resolve recursively
                    ExprKind::MemberAccess { .. } => {
                        // Try to extract the inner object as a compound name
                        // by recursively resolving the member access chain
                        self.resolve_member_access_prefix(object)
                    }
                    // G6.1.2 — paper §3.2 closure: subscript on `[Model; N]`
                    // produces a compound prefix like `"m.blocks.0"` that
                    // matches what `enumerate_model_tensor_paths` emits in
                    // the runtime tensor-path table (stmt.rs ~line 6396).
                    // The `Subscript` arm in `extract_expr` will have
                    // already registered the indexed context into
                    // `context_to_model_type` by the time we reach the
                    // first `.field` access on the result, so the
                    // Param-classification check below sees it.
                    ExprKind::Subscript { .. } => self.resolve_member_access_prefix(object),
                    _ => None,
                };

                // shape/ndim already handled by early check above

                if let Some(prefix) = obj_prefix {
                    // Reuse the synth-aware member_name resolved above.
                    let field_name = member_name_owned.clone();
                    let compound = format!("{}.{}", prefix, field_name);

                    // Check if we already have this compound name registered
                    // (search by name in var_names to handle cross-iteration reuse)
                    for (vid, name) in &self.list.var_names {
                        if name == &compound {
                            return Some(*vid);
                        }
                    }

                    // Determine if this is a model parameter (prefix is a known model context)
                    // or a data access (e.g., batch.input_ids). Only model params get gradients.
                    //
                    // G6.1.2 — paper §3.2 closure: not every field access on a
                    // model context is a tensor Param. `self.blocks` on
                    // `TinyCoder` is a `[TransformerBlock; N]` array, not a
                    // tensor. `self.attn` on `TransformerBlock` is a nested
                    // model, not a tensor. Both must NOT be registered as
                    // Params — otherwise the gradient-collection scan at
                    // stmt.rs ~line 4856 fails to match them against the
                    // runtime `trainable_tensor_param_paths`, surfacing as
                    // the "0/N trainable params connected" silent grad-miss
                    // that regressed the prior partial fix attempt.
                    //
                    // Discriminator: `model_field_types[parent_model_type]`
                    // (built by `compiler/collection.rs:411-728`) is
                    // populated ONLY with non-built-in named types — i.e.
                    // nested model fields and `[Model; N]` array fields.
                    // Tensor / int / float / bool / str fields are NEVER
                    // present (filtered at collection.rs:507-510).  So if
                    // `field_name` IS in the map, it's a nested model or
                    // model-array (NOT a tensor) and must be classified as
                    // a context, not a Param.
                    let parent_model = self.context_to_model_type.get(&prefix).cloned();
                    let nested_field_type = parent_model
                        .as_ref()
                        .and_then(|pt| self.model_field_types.get(pt))
                        .and_then(|fields| fields.get(&field_name))
                        .cloned();
                    let is_model_param = parent_model.is_some() && nested_field_type.is_none();

                    let var = self.alloc_var();
                    self.symbol_to_var.insert(*member, var);
                    self.list.var_names.insert(var, compound.clone());

                    // G6.1.2 — when the field is a SINGULAR nested model
                    // (not a `[Model; N]` array), register the compound as
                    // a new context so the next chain link (`.tensor` or
                    // `.method()`) resolves correctly.  Array fields are
                    // registered per-index by the `Subscript` arm below;
                    // doing it here for `[Model; N]` would register a
                    // mis-shaped "array-context" that wouldn't match any
                    // runtime path.
                    //
                    // Format coupling: `starts_with('[')` here recognizes
                    // the `format!("[{};{}]", elem_name, size)` produced
                    // by `compiler/collection.rs::~line 481` for
                    // `TypeExprKind::FixedArray`.  The two sites must
                    // stay in sync — if the collection-side format ever
                    // grows a leading whitespace or a different bracket
                    // shape, this check silently mis-classifies and
                    // downstream `is_model_param` accepts the array
                    // name as a tensor Param.  No test currently catches
                    // that drift; the coupling is documented here so a
                    // future maintainer touching collection.rs:481 sees
                    // the dependency.
                    if let Some(ref ft) = nested_field_type {
                        if !ft.starts_with('[') {
                            self.context_to_model_type
                                .entry(compound.clone())
                                .or_insert_with(|| ft.clone());
                        }
                    }

                    if is_model_param {
                        // CFTP v10 (item 5): populate `known_ranks` when
                        // the enclosing model recorded a declared rank
                        // for this field (via `collection.rs`).  Without
                        // this, a rank-3 model-field weight (e.g. an MoE
                        // expert stack `self.experts.weight:[D,V,H]`)
                        // would leave `known_ranks` empty and the
                        // fused-LCE matcher would silently fire on it —
                        // reintroducing the LATENT 3-D+ RISK on the
                        // most common weight-access path.
                        if let Some(pt) = parent_model.as_ref() {
                            if let Some(&r) = self
                                .model_field_ranks
                                .get(pt)
                                .and_then(|fields| fields.get(&field_name))
                            {
                                self.known_ranks.insert(var, r);
                            }
                        }
                        if self.is_frozen_compound(&compound) {
                            // CPKD (I-11): frozen (teacher) model field —
                            // register as an Input leaf.  Inputs receive no
                            // adjoint, so the teacher backward is
                            // structurally absent (F-06 guard).
                            self.frozen_input_vars.push((compound.clone(), var));
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result: var,
                                op: PrimalOp::Input(compound),
                                inputs: vec![],
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                        } else {
                            self.param_symbols.insert(*member);
                            self.named_param_vars.push((compound.clone(), var));
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result: var,
                                op: PrimalOp::Param(compound),
                                inputs: vec![],
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                        }
                    } else {
                        // Data access (e.g., batch.input_ids) — emit a dict_get op
                        // that depends on the object value, not a disconnected leaf.
                        // This preserves the computation edge: batch -> dict_get -> tensor.
                        let obj_var = self.extract_expr(object)?;
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: var,
                            op: PrimalOp::Passthrough(format!("dict_get:{}", field_name)),
                            inputs: vec![obj_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                    }
                    Some(var)
                } else {
                    None // Can't resolve object prefix
                }
            }

            ExprKind::BinaryOp { left, op, right } => {
                let l = self.extract_expr(left)?;
                let r = self.extract_expr(right)?;
                let result = self.alloc_var();
                let primal_op = match op {
                    BinOp::Add => PrimalOp::Add,
                    BinOp::Sub => PrimalOp::Sub,
                    BinOp::Mul => PrimalOp::Mul,
                    BinOp::Div => PrimalOp::Div,
                    // Comparison operators (non-differentiable, used for branch conditions)
                    BinOp::Gt => PrimalOp::Condition(CompareKind::Gt),
                    BinOp::Lt => PrimalOp::Condition(CompareKind::Lt),
                    BinOp::GtEq => PrimalOp::Condition(CompareKind::GtEq),
                    BinOp::LtEq => PrimalOp::Condition(CompareKind::LtEq),
                    BinOp::Eq => PrimalOp::Condition(CompareKind::Eq),
                    BinOp::NotEq => PrimalOp::Condition(CompareKind::NotEq),
                    BinOp::MatMul => PrimalOp::Matmul,
                    _ => return None, // Unsupported op for AD
                };
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: vec![l, r],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            // Pipe operator: `x |> f`
            //
            // Semantics depend on `f`:
            //   • `x |> tensor_field`  (bare Ident that resolves to a model Tensor param)
            //     → equivalent to `x @ tensor_field`, i.e. Matmul.
            //   • `x |> func`          (bare Ident that resolves to an input/local var)
            //     → Matmul(x, func_var).
            //   • Anything else        → extract RHS as a full expression, Matmul(x, rhs).
            //
            // The special case for bare model-field identifiers handles forward
            // bodies of the form `x |> q_proj |> k_proj |> …` where `q_proj` etc.
            // are model fields, not function names.  We resolve them via
            // `self_context` + `context_to_model_type` just as `ExprKind::MemberAccess`
            // with `SelfRef` would.
            ExprKind::Pipe { left, right } => {
                let left_var = self.extract_expr(left)?;

                // Try to resolve the RHS.  For a bare Ident we first attempt the
                // model-field path (registers a Param op if needed), then fall back
                // to symbol_to_var lookup.
                let right_var = if let ExprKind::Ident(sym) = &right.kind {
                    // Fast path: already registered.
                    if let Some(&v) = self.symbol_to_var.get(sym) {
                        v
                    } else {
                        // Try model-field path: look up "<self_context>.<ident_name>".
                        let ident_name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
                        let resolved = if let Some(ctx) = self.self_context.clone() {
                            if self.context_to_model_type.contains_key(&ctx) {
                                let compound = format!("{}.{}", ctx, ident_name);
                                // Check if already present in var_names (idempotent).
                                let existing = self
                                    .list
                                    .var_names
                                    .iter()
                                    .find_map(|(&vid, n)| if n == &compound { Some(vid) } else { None });
                                if let Some(vid) = existing {
                                    Some(vid)
                                } else {
                                    // Register as a Param leaf (or a frozen
                                    // Input leaf for CPKD teacher fields).
                                    let var = self.alloc_var();
                                    self.symbol_to_var.insert(*sym, var);
                                    self.list.var_names.insert(var, compound.clone());
                                    // CFTP v10 (item 5): mirror the
                                    // MemberAccess-site rank plumbing so
                                    // pipe-RHS model-field accesses
                                    // (`x |> q_proj`) also populate
                                    // `known_ranks` — otherwise a rank-3
                                    // `q_proj` field would slip past the
                                    // matcher via the pipe form only.
                                    if let Some(pt) = self
                                        .context_to_model_type
                                        .get(&ctx)
                                        .cloned()
                                    {
                                        if let Some(&r) = self
                                            .model_field_ranks
                                            .get(&pt)
                                            .and_then(|fields| fields.get(&ident_name))
                                        {
                                            self.known_ranks.insert(var, r);
                                        }
                                    }
                                    if self.is_frozen_compound(&compound) {
                                        // CPKD (I-11): teacher field via
                                        // pipe — Input leaf, no adjoint.
                                        self.frozen_input_vars
                                            .push((compound.clone(), var));
                                        self.push_op(WengertOp {
                                            id: self.list.ops.len() as u32,
                                            result: var,
                                            op: PrimalOp::Input(compound),
                                            inputs: vec![],
                                            saved_for_backward: false,
                                            checkpointed: false,
                                        });
                                    } else {
                                        self.param_symbols.insert(*sym);
                                        self.named_param_vars
                                            .push((compound.clone(), var));
                                        self.push_op(WengertOp {
                                            id: self.list.ops.len() as u32,
                                            result: var,
                                            op: PrimalOp::Param(compound),
                                            inputs: vec![],
                                            saved_for_backward: false,
                                            checkpointed: false,
                                        });
                                    }
                                    Some(var)
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        // If the model-field path failed (e.g., `relu` is a free
                        // function, not a model field), fall through to extracting
                        // the RHS as a normal expression.  Without this, `x |> relu
                        // |> q_proj` would return None here and collapse the whole
                        // pipe chain to a stub.
                        if let Some(vid) = resolved {
                            vid
                        } else {
                            self.extract_expr(right)?
                        }
                    }
                } else {
                    // Non-Ident RHS (e.g. a function call or nested pipe):
                    // extract it normally.
                    self.extract_expr(right)?
                };

                // Emit Matmul(left, right) — `x |> w` ≡ `x @ w`.
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Matmul,
                    inputs: vec![left_var, right_var],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::UnaryOp { op, operand } => {
                let input = self.extract_expr(operand)?;
                let result = self.alloc_var();
                let primal_op = match op {
                    AstUnaryOp::Neg => PrimalOp::Neg,
                    _ => return None,
                };
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: vec![input],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Call { callee, args } => {
                // Check for model method calls: obj.method(args) → inline the method body
                if let ExprKind::MemberAccess { object, member } = &callee.kind {
                    let method_name = self.interner.resolve(member.0).unwrap_or("?").to_string();

                    // Case 1: obj.method(args) where obj is an Ident (e.g., m.forward_train(...))
                    if let ExprKind::Ident(obj_sym) = &object.kind {
                        let obj_type = self.model_instance_types.get(obj_sym).cloned();
                        if let Some(model_type) = obj_type {
                            let fn_def = self
                                .model_method_bodies
                                .get(&model_type)
                                .and_then(|methods| methods.get(&method_name))
                                .cloned();
                            if let Some(fn_def) = fn_def {
                                return self.inline_method_call(*obj_sym, &fn_def, args);
                            } else {
                                eprintln!("[source-ad] method '{}' not found in model type '{}' (available: {:?})",
                                    method_name, model_type,
                                    self.model_method_bodies.get(&model_type)
                                        .map(|m| m.keys().collect::<Vec<_>>())
                                        .unwrap_or_default());
                            }
                        }
                    }

                    // Case 2: self.sub_model.method(args) — nested model field
                    if let ExprKind::MemberAccess {
                        object: inner_obj,
                        member: field_sym,
                    } = &object.kind
                    {
                        let obj_prefix = match &inner_obj.kind {
                            ExprKind::SelfRef => self.self_context.clone(),
                            ExprKind::Ident(sym) => {
                                self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                                    Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                                })
                            }
                            _ => None,
                        };
                        if let Some(prefix) = obj_prefix {
                            let sub_field_name = self
                                .interner
                                .resolve(field_sym.0)
                                .unwrap_or("?")
                                .to_string();
                            let compound_prefix = format!("{}.{}", prefix, sub_field_name);

                            // Find the sub-model's type
                            let parent_type = self.context_to_model_type.get(&prefix).cloned();
                            let sub_type = parent_type
                                .as_ref()
                                .and_then(|pt| self.model_field_types.get(pt))
                                .and_then(|fields| fields.get(&sub_field_name))
                                .cloned();

                            if let Some(ref sub_type) = sub_type {
                                // Don't try to inline array types
                                if !sub_type.starts_with('[') {
                                    self.context_to_model_type
                                        .insert(compound_prefix.clone(), sub_type.clone());
                                    let fn_def = self
                                        .model_method_bodies
                                        .get(sub_type)
                                        .and_then(|methods| methods.get(&method_name))
                                        .cloned();
                                    if let Some(fn_def) = fn_def {
                                        // Pre-extract args in CALLER's context before switching to callee's context.
                                        // This is critical: args like `self.attn_norm.forward(x)` use `self` referring
                                        // to the caller's model (TransformerBlock), not the callee (GQA).
                                        let extracted = self.pre_extract_args(&fn_def, args)?;
                                        let saved_self = self.self_context.clone();
                                        self.self_context = Some(compound_prefix);
                                        let result = self.inline_method_body(&fn_def, extracted);
                                        self.self_context = saved_self;
                                        return result;
                                    }
                                }
                            }
                        }
                    }

                    // Case 3: self.method(args) — method call on self
                    if let ExprKind::SelfRef = &object.kind {
                        if let Some(ctx) = self.self_context.clone() {
                            let model_type = self.context_to_model_type.get(&ctx).cloned();
                            if let Some(model_type) = model_type {
                                let fn_def = self
                                    .model_method_bodies
                                    .get(&model_type)
                                    .and_then(|methods| methods.get(&method_name))
                                    .cloned();
                                if let Some(fn_def) = fn_def {
                                    // Pre-extract args in caller's context
                                    let extracted = self.pre_extract_args(&fn_def, args)?;
                                    let saved_self = self.self_context.clone();
                                    self.self_context = Some(ctx);
                                    let result = self.inline_method_body(&fn_def, extracted);
                                    self.self_context = saved_self;
                                    return result;
                                }
                            }
                        }
                    }
                }

                // Tensor method calls: tensor.reshape(arg), tensor.transpose(d0, d1), etc.
                if let ExprKind::MemberAccess { object, member } = &callee.kind {
                    let method_name = self.interner.resolve(member.0).unwrap_or("?").to_string();
                    match method_name.as_str() {
                        "transpose" => {
                            let obj = self.extract_expr(object)?;
                            if let Some((dim0, dim1)) = Self::extract_transpose_dims(args) {
                                let result = self.alloc_var();
                                self.push_op(WengertOp {
                                    id: self.list.ops.len() as u32,
                                    result,
                                    op: PrimalOp::Transpose { dim0, dim1 },
                                    inputs: vec![obj],
                                    saved_for_backward: false,
                                    checkpointed: false,
                                });
                                return Some(result);
                            }

                            let mut inputs = vec![obj];
                            for arg in args {
                                inputs.push(self.extract_expr(&arg.value)?);
                            }
                            let result = self.alloc_var();
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result,
                                op: PrimalOp::Passthrough(method_name),
                                inputs,
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                            return Some(result);
                        }
                        "reshape" | "contiguous" | "item" | "expand" | "squeeze" | "unsqueeze" => {
                            let obj = self.extract_expr(object)?;
                            let mut inputs = vec![obj];
                            for arg in args {
                                inputs.push(self.extract_expr(&arg.value)?);
                            }
                            let result = self.alloc_var();
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result,
                                op: PrimalOp::Passthrough(method_name),
                                inputs,
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                            return Some(result);
                        }
                        "shape" | "ndim" => {
                            // Property access compiled as method — non-diff metadata
                            let obj = self.extract_expr(object)?;
                            let result = self.alloc_var();
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result,
                                op: PrimalOp::Passthrough(method_name),
                                inputs: vec![obj],
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                            return Some(result);
                        }
                        _ => {} // fall through to model method handling
                    }
                }

                // Regular function call: extract function name.
                //
                // WRGA B.3.2 Option 3: synth Calls emitted by the adapter
                // rewrite use a sentinel Symbol for the callee Ident. The
                // real FFI name lives in `synth_call_names` keyed by the
                // callee's NodeId — consult it FIRST so the fused-FFI
                // dispatch below can match.
                let func_name = if let Some(synth) = self.synth_call_names.get(&callee.id) {
                    synth.clone()
                } else if let ExprKind::Ident(sym) = &callee.kind {
                    self.interner.resolve(sym.0).unwrap_or("").to_string()
                } else if let ExprKind::MemberAccess { object, member } = &callee.kind {
                    let method = self.interner.resolve(member.0).unwrap_or("?");
                    let obj_desc = match &object.kind {
                        ExprKind::Ident(s) => {
                            format!("Ident({})", self.interner.resolve(s.0).unwrap_or("?"))
                        }
                        ExprKind::SelfRef => "self".to_string(),
                        ExprKind::MemberAccess {
                            object: inner,
                            member: m,
                        } => {
                            let m_name = self.interner.resolve(m.0).unwrap_or("?");
                            match &inner.kind {
                                ExprKind::SelfRef => format!("self.{}", m_name),
                                ExprKind::Ident(s) => format!(
                                    "{}.{}",
                                    self.interner.resolve(s.0).unwrap_or("?"),
                                    m_name
                                ),
                                _ => format!("?.{}", m_name),
                            }
                        }
                        _ => format!("{:?}", std::mem::discriminant(&object.kind)),
                    };
                    eprintln!("[source-ad] unresolved method call: {}.{}() — model type not found in method bodies", obj_desc, method);
                    return None;
                } else {
                    eprintln!(
                        "[source-ad] unsupported callee expression: {:?}",
                        std::mem::discriminant(&callee.kind)
                    );
                    return None; // Complex callee -- can't extract
                };

                // Extract arguments
                let mut input_vars = Vec::new();
                for arg in args {
                    if let Some(var) = self.extract_expr(&arg.value) {
                        input_vars.push(var);
                    } else {
                        return None;
                    }
                }

                let result = self.alloc_var();
                let primal_op = match func_name.as_str() {
                    // Elementwise unary
                    "relu" => PrimalOp::Relu,
                    "sigmoid" => PrimalOp::Sigmoid,
                    "tanh" => PrimalOp::Tanh,
                    "gelu" => PrimalOp::Gelu,
                    "silu" | "swish" => PrimalOp::Silu,
                    "exp" => PrimalOp::Exp,
                    "log" => PrimalOp::Log,
                    "sqrt" => PrimalOp::Sqrt,
                    "abs" => PrimalOp::Abs,
                    // Linear algebra
                    "matmul" => PrimalOp::Matmul,
                    // Reductions — extract dim from second arg if present
                    "softmax" => {
                        let dim = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(-1);
                        PrimalOp::Softmax { dim }
                    }
                    "log_softmax" => {
                        let dim = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(-1);
                        PrimalOp::LogSoftmax { dim }
                    }
                    // Normalization
                    "layer_norm" | "layernorm" => PrimalOp::LayerNorm { eps: 1e-5 },
                    "batch_norm" | "batchnorm" => PrimalOp::BatchNorm {
                        eps: 1e-5,
                        training: true,
                    },
                    "rmsnorm" | "rms_norm" => PrimalOp::RMSNorm { eps: 1e-5 },
                    // CFTP §4.4 G3 (Sprint 4): user-facing `fused_linear_ce`.
                    //
                    // When the active train block carries `@fused_lm_ce(enabled=true)`
                    // AND all four shape hints are populated, recognise the call
                    // as a single `PrimalOp::FusedLinearCe` instead of stepping
                    // into the stdlib composite (matmul + cross_entropy).
                    //
                    // When the decorator is absent or shape hints are missing,
                    // we INTENTIONALLY return `None` so callers fall through to
                    // the regular function-body lowering — preserving the
                    // composite path's bit-identity for inference / disabled
                    // decorators.  This honours the Sprint 4 regression
                    // invariant: a program without `@fused_lm_ce` always emits
                    // the composite.
                    // CPKD: user-facing `fused_kl_ce(x_s, W_s, b_s, x_t, W_t,
                    // b_t, targets, alpha, temperature)`. Recognised as a
                    // single PrimalOp::FusedKlCe when the enclosing distill
                    // block carries `@fused_kl_ce(enabled=true)` with all
                    // five shape hints. alpha/temperature must be numeric
                    // LITERALS in v1 (they bake into the op as compile-time
                    // constants) and must agree with the distill `loss:`
                    // section when it specifies them.
                    "fused_kl_ce" => {
                        if args.len() != 9 {
                            eprintln!(
                                "[source-ad] fused_kl_ce expected 9 args (x_s, W_s, bias_s, \
                                 x_t, W_t, bias_t, targets, alpha, temperature), got {}",
                                args.len()
                            );
                            return None;
                        }
                        let lit_f64 = |e: &nsl_ast::expr::Expr| -> Option<f64> {
                            match e.kind {
                                ExprKind::FloatLiteral(v) => Some(v),
                                ExprKind::IntLiteral(v) => Some(v as f64),
                                _ => None,
                            }
                        };
                        let (Some(alpha), Some(temperature)) =
                            (lit_f64(&args[7].value), lit_f64(&args[8].value))
                        else {
                            eprintln!(
                                "[source-ad] fused_kl_ce: alpha and temperature must be \
                                 numeric literals in v1 (compile-time kernel constants)"
                            );
                            return None;
                        };
                        // Consistency with the distill loss: section — a
                        // divergent literal would make the build report lie.
                        let (loss_alpha, loss_temp) = self.distill_loss_alpha_temp;
                        if let Some(la) = loss_alpha {
                            if (la - alpha).abs() > 1e-12 {
                                eprintln!(
                                    "[source-ad] fused_kl_ce: call-site alpha {alpha} \
                                     != distill loss: section alpha {la}"
                                );
                                return None;
                            }
                        }
                        if let Some(lt) = loss_temp {
                            if (lt - temperature).abs() > 1e-12 {
                                eprintln!(
                                    "[source-ad] fused_kl_ce: call-site temperature \
                                     {temperature} != distill loss: section temperature {lt}"
                                );
                                return None;
                            }
                        }
                        if let Some(cfg) = self.fused_kl_ce_config.as_ref() {
                            if cfg.enabled {
                                if let (Some(v), Some(hs), Some(ht), Some(b), Some(s)) = (
                                    cfg.vocab_size,
                                    cfg.student_hidden,
                                    cfg.teacher_hidden,
                                    cfg.batch_size,
                                    cfg.seq_len,
                                ) {
                                    let vt = cfg.vocab_tile.unwrap_or(
                                        crate::cpkd_fused_loss::FusedKlCeConfig::default()
                                            .vocab_tile,
                                    );
                                    let seven: Vec<VarId> =
                                        input_vars[0..7].to_vec();
                                    self.push_op(WengertOp {
                                        id: self.list.ops.len() as u32,
                                        result,
                                        op: PrimalOp::FusedKlCe {
                                            vocab_size: v,
                                            student_hidden: hs,
                                            teacher_hidden: ht,
                                            batch_size: b,
                                            seq_len: s,
                                            vocab_tile: vt,
                                            ignore_index: -100,
                                            alpha_bits: alpha.to_bits(),
                                            temperature_bits: temperature.to_bits(),
                                        },
                                        inputs: seven,
                                        saved_for_backward: false,
                                        checkpointed: false,
                                    });
                                    return Some(result);
                                }
                            }
                        }
                        // Fall through: composite expansion via the stdlib
                        // `fused_kl_ce` body (tape AD — refused inside
                        // distill blocks, so an incomplete/missing decorator
                        // there surfaces as a loud compile error). Name the
                        // exact precondition so the refusal is actionable
                        // rather than a bare "extraction failed".
                        match self.fused_kl_ce_config.as_ref() {
                            None => eprintln!(
                                "[source-ad] fused_kl_ce called without an active \
                                 @fused_kl_ce decorator config; add \
                                 @fused_kl_ce(enabled=true, vocab_size=, hidden_size=, \
                                 teacher_hidden=, batch_size=, seq_len=) to the distill block"
                            ),
                            Some(cfg) if !cfg.enabled => eprintln!(
                                "[source-ad] fused_kl_ce: @fused_kl_ce is present but \
                                 enabled=false; set enabled=true to activate the fused kernel"
                            ),
                            Some(_) => eprintln!(
                                "[source-ad] fused_kl_ce: @fused_kl_ce is enabled but one of \
                                 the shape hints (vocab_size, hidden_size, teacher_hidden, \
                                 batch_size, seq_len) is missing"
                            ),
                        }
                        return None;
                    }
                    "fused_linear_ce" => {
                        if args.len() != 4 {
                            eprintln!(
                                "[source-ad] fused_linear_ce expected 4 args (x, W, bias, targets), got {}",
                                args.len()
                            );
                            return None;
                        }
                        if let Some(cfg) = self.fused_ce_config.as_ref() {
                            if cfg.enabled {
                                if let (Some(v), Some(h), Some(b), Some(s)) = (
                                    cfg.vocab_size,
                                    cfg.hidden_size,
                                    cfg.batch_size,
                                    cfg.seq_len,
                                ) {
                                    let vt = cfg.vocab_tile.unwrap_or(
                                        crate::fused_linear_ce::FusedLinearCEConfig::default().vocab_tile,
                                    );
                                    let is_large = v
                                        > crate::fused_linear_ce::LARGE_VOCAB_THRESHOLD;
                                    let four = vec![
                                        input_vars[0], input_vars[1],
                                        input_vars[2], input_vars[3],
                                    ];
                                    self.push_op(WengertOp {
                                        id: self.list.ops.len() as u32,
                                        result,
                                        op: PrimalOp::FusedLinearCe {
                                            vocab_size: v,
                                            hidden_size: h,
                                            batch_size: b,
                                            seq_len: s,
                                            vocab_tile: vt,
                                            ignore_index: -100,
                                            is_large,
                                        },
                                        inputs: four,
                                        saved_for_backward: false,
                                        checkpointed: false,
                                    });
                                    return Some(result);
                                }
                            }
                        }
                        // Fall through: composite expansion via the regular
                        // function-body lowering (stdlib `fused_linear_ce`
                        // body in stdlib/nsl/nn/losses.nsl).
                        return None;
                    }
                    // Loss functions
                    //
                    // CFTP §4.4 G3 (Sprint v3-1): auto-substitution into the
                    // fused linear-CE kernel when the surrounding `train` block
                    // carries `@fused_lm_ce(enabled = true, ...)` AND the
                    // `cross_entropy(logits, targets)` input was produced by
                    // the canonical `Add(Matmul(x, Transpose(W, 0, 1)), bias)`
                    // pattern (the same expansion the stdlib's
                    // `fused_linear_ce(x, W, bias, targets)` decomposes to).
                    //
                    // On a successful match we synthesise a single
                    // `PrimalOp::FusedLinearCe` with inputs `[x, W, bias,
                    // targets]` (the same four-input shape Sprint 4's
                    // explicit-call path uses) and return immediately.  The
                    // AD rule in `ad_rules.rs` then produces the three
                    // backward components automatically.
                    //
                    // On ANY of the following, we fall through to the
                    // standard `PrimalOp::CrossEntropyLoss` emission below —
                    // preserving the composite path's bit-identity:
                    //   * decorator absent, or `enabled = false`
                    //   * any of the four shape hints (vocab_size,
                    //     hidden_size, batch_size, seq_len) is `None`
                    //   * the upstream chain doesn't match the canonical
                    //     stdlib decomposition (see
                    //     `try_match_fused_linear_ce_pattern`)
                    //
                    // This is the substitution site that the wengert_lower.rs
                    // Sprint 2.5 comment near `PrimalOp::CrossEntropyLoss`
                    // forward-referenced.
                    "cross_entropy" | "cross_entropy_loss" => {
                        if args.len() == 2 && input_vars.len() == 2 {
                            if let Some(cfg) = self.fused_ce_config.as_ref() {
                                if cfg.enabled {
                                    if let (Some(v), Some(h), Some(b), Some(s)) = (
                                        cfg.vocab_size,
                                        cfg.hidden_size,
                                        cfg.batch_size,
                                        cfg.seq_len,
                                    ) {
                                        let logits_var = input_vars[0];
                                        let targets_var = input_vars[1];
                                        if let Some(m) =
                                            self.try_match_fused_linear_ce_pattern(logits_var)
                                        {
                                            let vt = cfg.vocab_tile.unwrap_or(
                                                crate::fused_linear_ce::FusedLinearCEConfig::default(
                                                ).vocab_tile,
                                            );
                                            let is_large = v
                                                > crate::fused_linear_ce::LARGE_VOCAB_THRESHOLD;
                                            let four = vec![
                                                m.x_var, m.w_var, m.bias_var, targets_var,
                                            ];
                                            self.push_op(WengertOp {
                                                id: self.list.ops.len() as u32,
                                                result,
                                                op: PrimalOp::FusedLinearCe {
                                                    vocab_size: v,
                                                    hidden_size: h,
                                                    batch_size: b,
                                                    seq_len: s,
                                                    vocab_tile: vt,
                                                    ignore_index: -100,
                                                    is_large,
                                                },
                                                inputs: four,
                                                saved_for_backward: false,
                                                checkpointed: false,
                                            });
                                            // Review Finding 1: defer the prune of the
                                            // now-dead upstream composite chain
                                            // (Transpose → Matmul → Add) until finalize().
                                            // By then the whole tape is built, so the
                                            // "is this VarId consumed elsewhere?" check is
                                            // exact — no false-positive prunes from later
                                            // ops that haven't been pushed yet.
                                            self.pending_fused_lce_prunes.push(m);
                                            return Some(result);
                                        }
                                    }
                                }
                            }
                        }
                        PrimalOp::CrossEntropyLoss
                    }
                    "mse_loss" => PrimalOp::MSELoss,
                    "l1_loss" => PrimalOp::L1Loss,
                    // Reductions
                    "sum" => PrimalOp::Sum { dim: None },
                    "mean" => PrimalOp::Mean { dim: None },
                    // Regularization — extract p from second arg if literal.
                    //
                    // CRITICAL: dropout(x, p, training) — when the training
                    // arg statically resolves to a Constant(0.0), we MUST
                    // skip emitting Dropout entirely and pass the input
                    // through. Otherwise the wengert lower hardcodes
                    // training=1 in nsl_tensor_dropout, ablating ~p% of
                    // every activation on every step regardless of the
                    // user's training=false. The model never sees a stable
                    // representation and loss plateaus near random.
                    "dropout" => {
                        // Check the training arg (input_vars[2]) — if it
                        // resolves to PrimalOp::Constant(0.0), this is an
                        // inference-mode dropout call: emit identity.
                        let training_is_false = input_vars.get(2).is_some_and(|&tv| {
                            self.list.ops.iter().any(|op| {
                                op.result == tv
                                    && matches!(op.op, PrimalOp::Constant(c) if c == 0.0)
                            })
                        });
                        if training_is_false {
                            // Skip allocating a fresh result var — return
                            // the input directly so downstream uses see the
                            // passthrough.
                            return Some(input_vars[0]);
                        }
                        let p = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::FloatLiteral(v) => Some(*v),
                                _ => None,
                            })
                            .unwrap_or(0.1);
                        PrimalOp::Dropout { p }
                    }
                    // Indexing
                    "embedding" | "embedding_lookup" => PrimalOp::Embedding,
                    "gather" => {
                        // gather(tensor, dim, indices) — extract dim from second arg
                        let dim = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(0);
                        PrimalOp::Gather { dim }
                    }
                    // Conv2d: conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w).
                    // Extract square stride/padding as compile-time constants and
                    // emit PrimalOp::Conv2d with only the tensor inputs
                    // [input, weight, bias]. Non-constant or non-square stride/
                    // padding is not representable by the single-valued
                    // PrimalOp::Conv2d, so it returns None and falls back to tape AD
                    // (whose conv2d_backward handles the general case).
                    "conv2d" => {
                        if args.len() != 7 {
                            eprintln!(
                                "[source-ad] conv2d expected 7 args (input, weight, bias, \
                                 stride_h, stride_w, pad_h, pad_w), got {}",
                                args.len()
                            );
                            return None;
                        }
                        let as_const_int = |k: &ExprKind| -> Option<i64> {
                            match k {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            }
                        };
                        let (sh, sw, ph, pw) = match (
                            as_const_int(&args[3].value.kind),
                            as_const_int(&args[4].value.kind),
                            as_const_int(&args[5].value.kind),
                            as_const_int(&args[6].value.kind),
                        ) {
                            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                            _ => return None,
                        };
                        if sh != sw || ph != pw || sh < 1 || ph < 0 {
                            return None;
                        }
                        let conv_inputs = vec![input_vars[0], input_vars[1], input_vars[2]];
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result,
                            op: PrimalOp::Conv2d {
                                stride: sh as usize,
                                padding: ph as usize,
                            },
                            inputs: conv_inputs,
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(result);
                    }
                    // Attention — emit as fused ScaledDotProductAttention.
                    // The backward is handled by FlashAttentionBackwardExtract
                    // which calls nsl_flash_attention_backward (CPU reference path).
                    //
                    // Previously this was decomposed into primitive ops (transpose,
                    // matmul, softmax, etc.) — see commit e8d5a76. That approach
                    // worked but prevented using the fused backward kernel.
                    "scaled_dot_product_attention" => {
                        // Causal flag: check whether input_vars[4] resolves to
                        // Constant(1.0) (the extracted form of BoolLiteral(true)).
                        // Default to causal=true to match the prior behavior
                        // when the arg is missing or non-literal.
                        let causal = input_vars.get(4).is_none_or(|&cv| {
                            self.list.ops.iter().any(|op| {
                                op.result == cv
                                    && matches!(op.op, PrimalOp::Constant(c) if c != 0.0)
                            }) || !self.list.ops.iter().any(|op| op.result == cv)
                        });
                        PrimalOp::ScaledDotProductAttention { causal }
                    }
                    // PCA Stage B: masked attention for packed sequences —
                    // DECOMPOSED into primitive ops (transpose/matmul/mul/
                    // add/softmax/matmul), the pre-#347 SDPA shape (commit
                    // e8d5a76). The fused op's FlashAttentionBackwardExtract
                    // backward only understands a `causal` flag; an additive
                    // mask through that kernel would silently produce wrong
                    // dQ/dK/dV. Decomposition gives every step its standard
                    // source-AD adjoint (the mask enters via Add, whose
                    // gradient reduces to the mask shape and is discarded for
                    // non-param masks) at decomposed-forward speed — masked
                    // flash kernels are the Stage C work.
                    "scaled_dot_product_attention_masked" => {
                        if input_vars.len() < 5 {
                            eprintln!(
                                "[source-ad] scaled_dot_product_attention_masked \
                                 requires 5 args (q, k, v, scale, mask)"
                            );
                            return None;
                        }
                        let q_var = input_vars[0];
                        let k_var = input_vars[1];
                        let v_var = input_vars[2];
                        let scale_var = input_vars[3];
                        let mask_var = input_vars[4];

                        // k_t = transpose(k, -2, -1) — the (MAX-1, MAX)
                        // sentinel means "last two dims" in wengert_lower.
                        let k_t = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: k_t,
                            op: PrimalOp::Transpose {
                                dim0: usize::MAX - 1,
                                dim1: usize::MAX,
                            },
                            inputs: vec![k_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        // scores = q @ k_t
                        let scores = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: scores,
                            op: PrimalOp::Matmul,
                            inputs: vec![q_var, k_t],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        // scaled = scores * scale
                        let scaled = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: scaled,
                            op: PrimalOp::Mul,
                            inputs: vec![scores, scale_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        // masked = scaled + mask (additive; REPLACES causal —
                        // a packed block-diagonal mask already encodes
                        // within-doc causality)
                        let masked = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: masked,
                            op: PrimalOp::Add,
                            inputs: vec![scaled, mask_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        // attn = softmax(masked, -1)
                        let attn = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: attn,
                            op: PrimalOp::Softmax { dim: -1 },
                            inputs: vec![masked],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        // out = attn @ v — allocate the result var and return
                        // it directly (the shared tail below would wrap the
                        // WHOLE call in one more op otherwise).
                        let out = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: out,
                            op: PrimalOp::Matmul,
                            inputs: vec![attn, v_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(out);
                    }
                    // PCA Stage C: PACKED attention — a single fused op
                    // (unlike the masked builtin above, which decomposes).
                    // The lowering dispatches to the fused segment-masked
                    // flash kernel on GPU and falls back to the decomposed
                    // additive-mask chain at RUNTIME (CPU tensors, unmatched
                    // head_dim, launch failure). Backward is the `_segmask`
                    // FlashAttentionBackwardExtractPacked family. See the
                    // PrimalOp doc for the mask ≡ causal-within-segment
                    // contract.
                    "scaled_dot_product_attention_packed" => {
                        if input_vars.len() < 5 {
                            eprintln!(
                                "[source-ad] scaled_dot_product_attention_packed \
                                 requires 5 args (q, k, v, scale, segment_ids)"
                            );
                            return None;
                        }
                        PrimalOp::ScaledDotProductAttentionPacked
                    }
                    // Transpose (default: swap last two dims)
                    "transpose" => {
                        let dim_args = args.get(1..).unwrap_or(&[]);
                        let (dim0, dim1) = Self::extract_transpose_dims(dim_args).unwrap_or((0, 1));
                        PrimalOp::Transpose { dim0, dim1 }
                    }
                    // Shape ops (non-differentiable passthrough for AD)
                    "reshape" | "contiguous" | "expand" | "squeeze" | "unsqueeze" => {
                        PrimalOp::Passthrough(func_name.clone())
                    }
                    // Trigonometric (for RoPE)
                    "tensor_cos" | "cos" => PrimalOp::Passthrough("cos".into()),
                    "tensor_sin" | "sin" => PrimalOp::Passthrough("sin".into()),
                    "rotate_half" => PrimalOp::Passthrough("rotate_half".into()),
                    // Tensor construction (non-differentiable)
                    "arange" | "zeros" | "ones" | "full" | "randn" | "zeros_like" | "ones_like" => {
                        PrimalOp::Passthrough(func_name.clone())
                    }
                    // Concatenation
                    "tensor_cat" | "cat" => {
                        // tensor_cat(tensors, dim) — extract dim from last arg
                        let dim = args
                            .last()
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(-1);
                        PrimalOp::Concat { dim }
                    }
                    // Negative
                    "neg" => PrimalOp::Neg,
                    // Clamp
                    "clamp" => PrimalOp::Clamp {
                        min: f64::NEG_INFINITY,
                        max: f64::INFINITY,
                    },
                    // Scalar extraction (non-differentiable)
                    "int" | "float" => PrimalOp::Passthrough(func_name.clone()),
                    // WRGA B.3 fused LoRA forward FFI.
                    //
                    // Call shape: [x, W, A, B, FloatLit(scale), IntLit(kh)]
                    // Forward: y = x @ W + scale * (x @ A @ B).
                    "nsl_adapter_fused_lora_matmul" => {
                        if args.len() != 6 {
                            eprintln!(
                                "[source-ad] nsl_adapter_fused_lora_matmul expected 6 args, got {}",
                                args.len()
                            );
                            return None;
                        }
                        let scale = match &args[4].value.kind {
                            ExprKind::FloatLiteral(v) => *v as f32,
                            ExprKind::IntLiteral(v) => *v as f32,
                            ExprKind::UnaryOp { op: AstUnaryOp::Neg, operand } => match &operand.kind {
                                ExprKind::FloatLiteral(v) => -(*v) as f32,
                                ExprKind::IntLiteral(v) => -(*v) as f32,
                                _ => return None,
                            },
                            _ => return None,
                        };
                        let kernel_handle = match &args[5].value.kind {
                            ExprKind::IntLiteral(v) => *v,
                            ExprKind::UnaryOp { op: AstUnaryOp::Neg, operand } => match &operand.kind {
                                ExprKind::IntLiteral(v) => -*v,
                                _ => return None,
                            },
                            _ => return None,
                        };
                        let four_inputs = vec![
                            input_vars[0], input_vars[1], input_vars[2], input_vars[3],
                        ];
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result,
                            op: PrimalOp::FusedLoraMatmul { scale, kernel_handle },
                            inputs: four_inputs,
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(result);
                    }
                    // WRGA B.3 fused IA³ forward FFI.
                    //
                    // Call shape: [x, W, gamma, IntLit(kh)]
                    // Forward: y = (x @ W) * gamma (gamma broadcasts over out dim).
                    "nsl_adapter_fused_ia3_matmul" => {
                        if args.len() != 4 {
                            eprintln!(
                                "[source-ad] nsl_adapter_fused_ia3_matmul expected 4 args, got {}",
                                args.len()
                            );
                            return None;
                        }
                        let kernel_handle = match &args[3].value.kind {
                            ExprKind::IntLiteral(v) => *v,
                            ExprKind::UnaryOp { op: AstUnaryOp::Neg, operand } => match &operand.kind {
                                ExprKind::IntLiteral(v) => -*v,
                                _ => return None,
                            },
                            _ => return None,
                        };
                        let three_inputs = vec![input_vars[0], input_vars[1], input_vars[2]];
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result,
                            op: PrimalOp::FusedIa3Matmul { kernel_handle },
                            inputs: three_inputs,
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(result);
                    }
                    // WRGA B.3.2 Option 3: fused GatedLoRA forward FFI.
                    //
                    // Call shape (see wrga_adapter_rewrite::synthesize_gatedlora_fused_call):
                    //   args = [x, W, A, B, FloatLit(scale), gate, IntLit(kh)]
                    //
                    // PrimalOp has 5 tensor inputs [x, W, A, B, gate];
                    // scale + kernel_handle carried on the variant.
                    "nsl_adapter_fused_gatedlora_matmul" => {
                        if args.len() != 7 {
                            eprintln!(
                                "[source-ad] nsl_adapter_fused_gatedlora_matmul expected 7 args, got {}",
                                args.len()
                            );
                            return None;
                        }
                        let scale = match &args[4].value.kind {
                            ExprKind::FloatLiteral(v) => *v as f32,
                            ExprKind::IntLiteral(v) => *v as f32,
                            ExprKind::UnaryOp { op: AstUnaryOp::Neg, operand } => match &operand.kind {
                                ExprKind::FloatLiteral(v) => -(*v) as f32,
                                ExprKind::IntLiteral(v) => -(*v) as f32,
                                _ => return None,
                            },
                            _ => return None,
                        };
                        let kernel_handle = match &args[6].value.kind {
                            ExprKind::IntLiteral(v) => *v,
                            ExprKind::UnaryOp { op: AstUnaryOp::Neg, operand } => match &operand.kind {
                                ExprKind::IntLiteral(v) => -*v,
                                _ => return None,
                            },
                            _ => return None,
                        };
                        let five_inputs = vec![
                            input_vars[0], input_vars[1], input_vars[2], input_vars[3], input_vars[5],
                        ];
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result,
                            op: PrimalOp::FusedGatedLoraMatmul { scale, kernel_handle },
                            inputs: five_inputs,
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(result);
                    }
                    _ => {
                        eprintln!(
                            "[source-ad] warning: unrecognized FFI callee '{}' in train block; \
                             falling back to unfused AST evaluation. If you expected a fused kernel, \
                             check that source-AD has a handler for this FFI.",
                            func_name
                        );
                        return None;
                    }
                };

                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: input_vars,
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::IntLiteral(v) => {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(*v as f64),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                // Override type: integer literals are raw i64, not tensor pointers.
                // push_op defaults to Tensor (for adjoint seed constants), but
                // IntLiterals are used for subscript indices and shape dimensions.
                self.list.var_types.insert(result, WengertType::Integer);
                Some(result)
            }

            ExprKind::FloatLiteral(v) => {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(*v),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::BoolLiteral(v) => {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(if *v { 1.0 } else { 0.0 }),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Paren(inner) => self.extract_expr(inner),

            // Subscript: list[index] — non-differentiable metadata access
            ExprKind::Subscript { object, index } => {
                // G6.1.2 — paper §3.2 closure: pre-register all nested
                // model contexts implied by this chain before recursing.
                // See `preregister_chain_contexts` for the rationale —
                // top-down extraction would otherwise compute
                // `is_model_param` before the recursive descent that
                // would register intermediate contexts, locking in the
                // wrong classification.
                self.preregister_chain_contexts(expr);

                let obj = self.extract_expr(object)?;
                if let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() {
                    let idx = self.extract_expr(idx_expr)?;
                    // Note: idx type is NOT overridden here — the subscript lowerer
                    // handles type conversion (extracts i64 from tensor/scalar).
                    let result = self.alloc_var();
                    self.push_op(WengertOp {
                        id: self.list.ops.len() as u32,
                        result,
                        op: PrimalOp::Passthrough("subscript".into()),
                        inputs: vec![obj, idx],
                        saved_for_backward: false,
                        checkpointed: false,
                    });
                    Some(result)
                } else {
                    None
                }
            }

            // List literal: [a, b, c] — non-differentiable
            ExprKind::ListLiteral(elements) => {
                let mut inputs = Vec::new();
                for elem in elements {
                    inputs.push(self.extract_expr(elem)?);
                }
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Passthrough("list".into()),
                    inputs,
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            // Anything else we can't extract -> fallback
            _ => None,
        }
    }

    /// Extract an if/else statement into the Wengert list.
    ///
    /// Strategy: flatten branches into linear SSA by extracting both branches
    /// independently, then emitting `Select(cond, true_val, false_val)` ops
    /// for any variables that differ between branches.
    ///
    /// For `if cond { return x*x } else { return -x }`, this produces:
    ///   cond_var = Condition(Gt, x, 0)
    ///   true_val = Mul(x, x)        // from then-branch
    ///   false_val = Neg(x)           // from else-branch
    ///   result = Select(cond_var, true_val, false_val)
    ///
    /// # Speculative (Eager) Evaluation of Both Branches
    ///
    /// **Both branches are eagerly evaluated**: all ops from both the then-branch
    /// and the else-branch appear unconditionally in the Wengert list. Only the
    /// `Select` op at the end chooses which result to use based on the condition.
    ///
    /// This design is **intentional** for two reasons:
    /// 1. **SSA flattening**: the Wengert list is a linear DAG with no control flow;
    ///    both paths must be materialized so that reverse-mode AD can propagate
    ///    gradients through the chosen branch.
    /// 2. **AD correctness**: the `SelectTrue`/`SelectFalse` adjoint rules mask
    ///    gradient flow to the non-taken branch (multiplying by 0), which is the
    ///    correct sub-gradient for a piecewise-linear selection.
    ///
    /// **Known limitation — side effects and expensive computations**: Because both
    /// branches execute unconditionally, any side-effectful code (I/O, mutations,
    /// random sampling) inside a branch will run on *every* evaluation regardless of
    /// the condition. Similarly, expensive computations in the unused branch still
    /// pay their full cost. This is a deliberate trade-off for AD correctness; users
    /// with side-effectful branches should use the dynamic tape fallback (`@no_static`
    /// or constructs that prevent static extraction) instead.
    fn extract_if(
        &mut self,
        condition: &nsl_ast::expr::Expr,
        then_block: &nsl_ast::stmt::Block,
        elif_clauses: &[(nsl_ast::expr::Expr, nsl_ast::stmt::Block)],
        else_block: Option<&nsl_ast::stmt::Block>,
    ) -> bool {
        // Extract the condition expression
        let cond_var = match self.extract_expr(condition) {
            Some(v) => v,
            None => return false,
        };

        // Snapshot the current symbol -> var mapping before entering branches
        let snapshot = self.symbol_to_var.clone();
        let output_before = self.list.output;

        // --- Extract then-branch ---
        if !self.extract_stmts(&then_block.stmts) {
            return false;
        }
        let then_symbols = self.symbol_to_var.clone();
        let then_output = self.list.output;

        // Restore snapshot for else-branch extraction
        self.symbol_to_var = snapshot.clone();
        self.list.output = output_before;

        // --- Extract else-branch ---
        // Handle elif as nested if/else: `elif c2: B2 else: B3` becomes
        // `else: if c2: B2 else: B3` — we support at most simple else for now.
        // If there are elif clauses, desugar the first one as a nested if.
        if !elif_clauses.is_empty() {
            // Desugar: elif chain becomes nested if/else.
            // `elif c2: B2 elif c3: B3 else: B4` is treated as
            // `else: if c2: B2 elif c3: B3 else: B4`, recursively.
            let (elif_cond, elif_block) = &elif_clauses[0];
            let remaining_elifs = &elif_clauses[1..];
            // Pass the remaining elif clauses into the recursive call so the
            // entire chain is desugared, not just the first level.
            if !self.extract_if(elif_cond, elif_block, remaining_elifs, else_block) {
                return false;
            }
        } else if let Some(else_blk) = else_block {
            if !self.extract_stmts(&else_blk.stmts) {
                return false;
            }
        }
        // If no else block: variables retain their pre-branch values (identity)

        let else_symbols = self.symbol_to_var.clone();
        let else_output = self.list.output;

        // --- Merge branches with Select ops ---
        // Collect all symbols that exist in either branch
        let all_symbols: HashSet<nsl_ast::Symbol> = then_symbols
            .keys()
            .chain(else_symbols.keys())
            .copied()
            .collect();

        for sym in &all_symbols {
            let then_var = then_symbols.get(sym).copied();
            let else_var = else_symbols.get(sym).copied();
            let before_var = snapshot.get(sym).copied();

            let tv = then_var.or(before_var);
            let ev = else_var.or(before_var);

            if let (Some(t), Some(e)) = (tv, ev) {
                if t != e {
                    // Different values in each branch — emit Select
                    let result = self.alloc_var();
                    self.push_op(WengertOp {
                        id: self.list.ops.len() as u32,
                        result,
                        op: PrimalOp::Select,
                        inputs: vec![cond_var, t, e],
                        saved_for_backward: false,
                        checkpointed: false,
                    });
                    self.symbol_to_var.insert(*sym, result);
                    // Propagate variable name
                    if let Some(name) = self
                        .list
                        .var_names
                        .get(&t)
                        .cloned()
                        .or_else(|| self.list.var_names.get(&e).cloned())
                    {
                        self.list.var_names.insert(result, name);
                    }
                } else {
                    // Same value in both branches — no Select needed
                    self.symbol_to_var.insert(*sym, t);
                }
            }
        }

        // Handle output (return value) merging
        if then_output != output_before || else_output != output_before {
            let t_out = if then_output != output_before {
                then_output
            } else {
                output_before
            };
            let e_out = if else_output != output_before {
                else_output
            } else {
                output_before
            };
            if t_out != e_out {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Select,
                    inputs: vec![cond_var, t_out, e_out],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                self.list.output = result;
            } else {
                self.list.output = t_out;
            }
        }

        true
    }

    /// Try to unroll a for-loop over a model's FixedArray field.
    ///
    /// Handles patterns like `for block in self.blocks: x = block.forward(x, training)`
    /// by resolving the iterable to a model field with a known array type (e.g.,
    /// `[TransformerBlock; 8]`), then expanding the loop body N times with the loop
    /// variable bound to each element's context prefix (e.g., "m.blocks.0", "m.blocks.1", ...).
    fn try_unroll_for(
        &mut self,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> bool {
        // 1. Get the loop variable symbol
        let loop_var_sym = match &pattern.kind {
            nsl_ast::pattern::PatternKind::Ident(sym) => *sym,
            _ => {
                self.is_static = false;
                return false;
            }
        };

        // 2. Resolve the iterable to get the context prefix and field name.
        // The iterable is `self.blocks` (MemberAccess { SelfRef, "blocks" })
        // or `m.blocks` (MemberAccess { Ident(m), "blocks" }).
        let (context_prefix, field_name) = match &iterable.kind {
            ExprKind::MemberAccess { object, member } => {
                let ctx = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(sym) => {
                        self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                            Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                        })
                    }
                    _ => None,
                };
                let fname = self.interner.resolve(member.0).unwrap_or("?").to_string();
                match ctx {
                    Some(c) => (c, fname),
                    None => {
                        self.is_static = false;
                        return false;
                    }
                }
            }
            _ => {
                self.is_static = false;
                return false;
            }
        };

        // 3. Look up the model type for this context, then the field type
        let model_type = self.context_to_model_type.get(&context_prefix).cloned();
        let field_info = model_type
            .as_ref()
            .and_then(|mt| self.model_field_types.get(mt))
            .and_then(|fields| fields.get(&field_name))
            .cloned();

        // Parse field type — look for "[ModelType;N]" pattern
        let (element_type, size) = match field_info {
            Some(ref ft) if ft.starts_with('[') && ft.contains(';') => {
                // Parse "[TransformerBlock;8]"
                let inner = ft.trim_start_matches('[').trim_end_matches(']');
                let parts: Vec<&str> = inner.split(';').collect();
                if parts.len() == 2 {
                    let elem = parts[0].trim().to_string();
                    let n = parts[1].trim().parse::<usize>().unwrap_or(0);
                    if n > 0 {
                        (elem, n)
                    } else {
                        self.is_static = false;
                        return false;
                    }
                } else {
                    self.is_static = false;
                    return false;
                }
            }
            _ => {
                self.is_static = false;
                return false;
            }
        };

        // 4. Unroll: for i in 0..size, expand loop body with adjusted context
        for i in 0..size {
            let iter_prefix = format!("{}.{}.{}", context_prefix, field_name, i);

            // Register this iteration's context -> model type
            self.context_to_model_type
                .insert(iter_prefix.clone(), element_type.clone());

            // Map loop variable to this iteration's context
            self.symbol_name_overrides
                .insert(loop_var_sym, iter_prefix.clone());
            self.model_instance_types
                .insert(loop_var_sym, element_type.clone());

            // Save and set self_context for nested self.field resolution
            let saved_self = self.self_context.clone();

            // Extract loop body
            if !self.extract_stmts(&body.stmts) {
                self.self_context = saved_self;
                self.is_static = false;
                return false;
            }

            self.self_context = saved_self;
        }

        // Clean up overrides
        self.symbol_name_overrides.remove(&loop_var_sym);
        true
    }

    /// Inline a model method call, given the model instance symbol and FnDef.
    /// Sets up self_context from the symbol name (or override), pre-extracts
    /// args in the caller's context, then inlines the method body.
    fn inline_method_call(
        &mut self,
        model_sym: nsl_ast::Symbol,
        fn_def: &nsl_ast::decl::FnDef,
        call_args: &[nsl_ast::expr::Arg],
    ) -> Option<VarId> {
        // Pre-extract args in caller's context before switching to callee's
        let extracted = self.pre_extract_args(fn_def, call_args)?;
        let name = self
            .symbol_name_overrides
            .get(&model_sym)
            .cloned()
            .unwrap_or_else(|| {
                self.interner
                    .resolve(model_sym.0)
                    .unwrap_or("?")
                    .to_string()
            });
        let saved_self = self.self_context.clone();
        self.self_context = Some(name);
        let result = self.inline_method_body(fn_def, extracted);
        self.self_context = saved_self;
        result
    }

    /// Inline a model method call's body into the current Wengert list.
    /// Assumes `self.self_context` is already set to the correct prefix.
    ///
    /// Binds call arguments to the method's parameter names (skipping `self`),
    /// then extracts the method body statements. The return value of the method
    /// becomes the result of this expression.
    /// Pre-extract call arguments in the current (caller's) context.
    /// Returns (param_symbol, param_name, VarId) triples for binding.
    fn pre_extract_args(
        &mut self,
        fn_def: &nsl_ast::decl::FnDef,
        call_args: &[nsl_ast::expr::Arg],
    ) -> Option<Vec<(nsl_ast::Symbol, String, VarId)>> {
        let mut result = Vec::new();
        let mut arg_idx = 0;
        for param in &fn_def.params {
            let param_name = self
                .interner
                .resolve(param.name.0)
                .unwrap_or("?")
                .to_string();
            if param_name == "self" {
                continue;
            }
            if arg_idx < call_args.len() {
                if let Some(var) = self.extract_expr(&call_args[arg_idx].value) {
                    result.push((param.name, param_name, var));
                } else {
                    return None;
                }
                arg_idx += 1;
            }
        }
        Some(result)
    }

    /// Inline a method body with pre-extracted argument bindings.
    fn inline_method_body(
        &mut self,
        fn_def: &nsl_ast::decl::FnDef,
        extracted_args: Vec<(nsl_ast::Symbol, String, VarId)>,
    ) -> Option<VarId> {
        // Inline callee locals in an isolated symbol scope so names like
        // `batch` inside a model method do not overwrite caller bindings such
        // as the train-step batch dict.
        let saved_symbols = self.symbol_to_var.clone();

        // Bind the pre-extracted argument VarIds to parameter symbols
        for (sym, name, var) in extracted_args {
            self.symbol_to_var.insert(sym, var);
            self.list.var_names.insert(var, name);
        }

        // Save the output before extraction
        let saved_output = self.list.output;

        // Extract the method body
        if !self.extract_stmts(&fn_def.body.stmts) {
            self.symbol_to_var = saved_symbols;
            self.list.output = saved_output;
            return None;
        }

        // Cycle-10 §5.3 Task 6: if this method is registered as
        // `@checkpoint(policy=Full)` in the semantic-layer-published
        // policy map, stamp the just-extracted prologue ops and emit
        // a `PrologueRecompute` marker at the tape tail. No-op when
        // the policy map is empty (byte-identity preserved).
        let fn_name = self
            .interner
            .resolve(fn_def.name.0)
            .unwrap_or("?")
            .to_string();
        self.apply_checkpoint_policy(&fn_name);

        // The method's return value is the list output (set by Return stmt extraction)
        let result = self.list.output;

        // Restore caller scope now that the callee result has been captured.
        self.symbol_to_var = saved_symbols;
        self.list.output = saved_output;

        // If the method didn't set output (no return statement), check if a variable
        // was assigned that should be the result
        if result == saved_output {
            // No return found — method might use implicit return via last expression
            return None;
        }

        Some(result)
    }

    /// G6.1.2 — paper §3.2 closure: walk a MemberAccess/Subscript chain
    /// bottom-up and eagerly register all nested-model contexts implied
    /// by the chain into `context_to_model_type`.
    ///
    /// Called from `extract_expr` at the top of every `MemberAccess`
    /// and `Subscript` arm.  Without this pre-pass, the top-down
    /// extraction order causes the parent `is_model_param` check to
    /// run BEFORE the recursive descent that would register an
    /// intermediate context, locking in a wrong-classification frame
    /// at every chain depth.
    ///
    /// Registration uses `model_field_types` (populated by
    /// `compiler/collection.rs:411-728`), which contains ONLY non-
    /// built-in named field types (nested models + `[Model; N]`
    /// arrays) — Tensor / int / float / bool / str are excluded at
    /// collection.rs:507-510, so presence in the map is the
    /// discriminator for "this field is a nested model context, not a
    /// tensor Param".
    ///
    /// Indices must be static integer literals; dynamic indices skip
    /// registration so the static-AD path falls back safely (tape-AD
    /// handles dynamic indexing at runtime).
    fn preregister_chain_contexts(&mut self, expr: &nsl_ast::expr::Expr) {
        match &expr.kind {
            ExprKind::MemberAccess { object, member } => {
                // Descend first so deeper contexts are registered before
                // we compute this access's compound.
                self.preregister_chain_contexts(object);

                // If the object resolves to a prefix and the parent
                // model has this field as a SINGULAR nested model,
                // register `{prefix}.{field} -> field_type`.  Skip
                // array fields here — those are handled per-index by
                // the Subscript arm below (registering a bare-array
                // compound would mis-shape the context map).
                if let Some(obj_prefix) = self.resolve_member_access_prefix(object) {
                    let parent_model = self.context_to_model_type.get(&obj_prefix).cloned();
                    let field_name = self.interner.resolve(member.0).unwrap_or("?").to_string();
                    if let Some(parent) = parent_model {
                        if let Some(field_ty) = self
                            .model_field_types
                            .get(&parent)
                            .and_then(|fields| fields.get(&field_name))
                            .cloned()
                        {
                            if !field_ty.starts_with('[') {
                                let compound = format!("{}.{}", obj_prefix, field_name);
                                self.context_to_model_type
                                    .entry(compound)
                                    .or_insert(field_ty);
                            }
                        }
                    }
                }
            }
            ExprKind::Subscript { object, index } => {
                self.preregister_chain_contexts(object);

                // Static-index subscript on a `[Model; N]` field:
                // register the indexed compound -> element model type.
                if let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() {
                    if let ExprKind::IntLiteral(i) = &idx_expr.kind {
                        if *i >= 0 {
                            if let Some(obj_prefix) = self.resolve_member_access_prefix(object) {
                                if let Some((parent_pref, field_name)) =
                                    obj_prefix.rsplit_once('.')
                                {
                                    let parent_model = self
                                        .context_to_model_type
                                        .get(parent_pref)
                                        .cloned();
                                    if let Some(field_ty) = parent_model
                                        .as_ref()
                                        .and_then(|pt| self.model_field_types.get(pt))
                                        .and_then(|fields| fields.get(field_name))
                                        .cloned()
                                    {
                                        // Same format coupling as the
                                        // singular-Model arm above —
                                        // recognises the
                                        // `format!("[{};{}]", elem, n)`
                                        // from
                                        // `compiler/collection.rs::~481`.
                                        // Any drift on either side breaks
                                        // silently; keep these two
                                        // recognizers in sync.
                                        if field_ty.starts_with('[') && field_ty.contains(';') {
                                            let inner = field_ty
                                                .trim_start_matches('[')
                                                .trim_end_matches(']');
                                            if let Some(elem) = inner.split(';').next() {
                                                let indexed_ctx = format!("{}.{}", obj_prefix, i);
                                                self.context_to_model_type
                                                    .entry(indexed_ctx)
                                                    .or_insert_with(|| elem.trim().to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Resolve a nested MemberAccess chain to a compound prefix string.
    /// E.g., `self.attn` -> "m.blocks.0.attn", `block.ffn` -> "m.blocks.0.ffn"
    ///
    /// G6.1.2: also handles `Subscript` of array-of-models fields. For
    /// `self.blocks[0]` this yields `"m.blocks.0"` — matching the
    /// runtime tensor-path format that `enumerate_model_tensor_paths`
    /// (stmt.rs ~line 6396) emits.  Subscript indices must be static
    /// integer literals; dynamic indices return None so the AD path
    /// falls back safely (tape-AD handles dynamic indexing at runtime).
    ///
    /// This is intentionally a PURE STRING WALK with no side effects —
    /// eager registration of intermediate contexts happens in
    /// `preregister_chain_contexts`, called at the top of the
    /// `extract_expr` `MemberAccess` / `Subscript` arms.  Mixing
    /// registration into resolution would re-introduce the
    /// lazy-vs-eager asymmetry that regressed the prior partial fix
    /// attempt.
    fn resolve_member_access_prefix(&self, expr: &nsl_ast::expr::Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::MemberAccess { object, member } => {
                let obj_prefix = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(sym) => {
                        self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                            Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                        })
                    }
                    ExprKind::MemberAccess { .. } => self.resolve_member_access_prefix(object),
                    ExprKind::Subscript { .. } => self.resolve_member_access_prefix(object),
                    _ => None,
                };
                let field_name = self.interner.resolve(member.0).unwrap_or("?").to_string();
                obj_prefix.map(|p| format!("{}.{}", p, field_name))
            }
            // G6.1.2: array-of-models subscript. `self.blocks[0]` -> "m.blocks.0".
            ExprKind::Subscript { object, index } => {
                let obj_prefix = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(sym) => {
                        self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                            Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                        })
                    }
                    ExprKind::MemberAccess { .. } => self.resolve_member_access_prefix(object),
                    ExprKind::Subscript { .. } => self.resolve_member_access_prefix(object),
                    _ => None,
                }?;
                let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() else {
                    return None;
                };
                let ExprKind::IntLiteral(i) = &idx_expr.kind else {
                    return None;
                };
                if *i < 0 {
                    return None;
                }
                Some(format!("{}.{}", obj_prefix, i))
            }
            _ => None,
        }
    }

    /// Finalize extraction. Returns the WengertList if the graph is static.
    ///
    /// Side-effect: drains `pending_fused_lce_prunes` and removes the
    /// dead `Transpose → Matmul → Add` chains the auto-substitution
    /// path queued during extraction (review Finding 1).  Conservative —
    /// each chain is only removed once verified to have no remaining
    /// consumers in the completed tape.
    pub fn finalize(mut self) -> Option<WengertList> {
        if !self.is_static || self.list.ops.is_empty() {
            return None;
        }
        // Review Finding 1: run all pending fused-LCE prunes against
        // the COMPLETED tape so the consumer-scan sees every op.
        let prunes = std::mem::take(&mut self.pending_fused_lce_prunes);
        for m in &prunes {
            Self::prune_fused_lce_dead_chain(&mut self.list, m);
        }
        Some(self.list)
    }

    /// Check if the computation graph is static (no dynamic control flow).
    pub fn is_static_graph(&self) -> bool {
        self.is_static
    }

    /// Get the parameter VarIds (for gradient computation targets).
    pub fn param_vars(&self) -> Vec<VarId> {
        self.param_symbols
            .iter()
            .filter_map(|sym| self.symbol_to_var.get(sym).copied())
            .collect()
    }

    /// Get the symbol-to-VarId mapping for resolving primal vars to Cranelift Values.
    pub fn symbol_var_map(&self) -> &HashMap<nsl_ast::Symbol, VarId> {
        &self.symbol_to_var
    }

    /// Get VarIds for model parameters (unordered).
    pub fn param_var_ids(&self) -> Vec<(nsl_ast::Symbol, VarId)> {
        self.param_symbols
            .iter()
            .filter_map(|sym| self.symbol_to_var.get(sym).map(|vid| (*sym, *vid)))
            .collect()
    }

    /// Get the next available VarId (for AdjointGenerator::new start_var).
    pub fn next_var_id(&self) -> VarId {
        self.next_var
    }

    /// Access the extracted WengertList.
    pub fn wengert_list(&self) -> &WengertList {
        &self.list
    }

    /// Mutably access the extracted WengertList.
    /// Used by `wggo_prune::run()` to perform in-place IR rewrites before
    /// downstream passes (WRGA, source-AD adjoint generation) consume the list.
    pub fn wengert_list_mut(&mut self) -> &mut WengertList {
        &mut self.list
    }

    /// Set the output VarId of the extracted WengertList.
    /// This is needed when the step body uses `let loss = ...` (VarDecl) rather
    /// than `return loss`, since VarDecl extraction does not set list.output.
    pub fn set_output(&mut self, var: VarId) {
        self.list.output = var;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertList, WengertOp};

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    #[test]
    fn test_adjoint_generator_simple_add() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("a".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("b".into()), vec![]),
                make_op(2, 2, PrimalOp::Add, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);
        assert!(!adjoint.ops.is_empty());
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_adjoint_generator_matmul() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("A".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("B".into()), vec![]),
                make_op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let _adjoint = gen.generate(&primal);
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    /// CPDT Part II activation: directly tests `lower_adjoint_expr` against
    /// a single `MulElementwise` adjoint, independent of the full
    /// `AdjointGenerator::generate` flow. Catches a bug class that
    /// `mul_backward_emits_reduce_to_shape_per_arm` cannot — e.g. accidentally
    /// swapping `raw` and `target` inside the Mul lowering's emit_op call
    /// would still produce two ops and a `Passthrough("reduce_to_shape")`
    /// op, but with the wrong target VarId.
    #[test]
    fn lower_adjoint_expr_mul_elementwise_uses_target_as_shape_anchor() {
        let mut gen = AdjointGenerator::new(200);
        // grad=100, other=42, target=7 — chosen to be unambiguous in the
        // result inputs vector.
        let _ = gen.lower_adjoint_expr(AdjointExpr::MulElementwise(100, 42, 7));
        let ops = &gen.adjoint_ops;
        assert_eq!(
            ops.len(),
            2,
            "MulElementwise must lower to exactly 2 ops (Mul + reduce_to_shape), got {} (ops: {:?})",
            ops.len(),
            ops.iter().map(|o| &o.op).collect::<Vec<_>>(),
        );
        assert!(
            matches!(ops[0].op, PrimalOp::Mul),
            "op[0] must be PrimalOp::Mul, got {:?}",
            ops[0].op,
        );
        assert_eq!(
            ops[0].inputs,
            vec![100, 42],
            "Mul inputs must be (grad, other), got {:?}",
            ops[0].inputs,
        );
        let raw_var = ops[0].result;
        match &ops[1].op {
            PrimalOp::Passthrough(name) => assert_eq!(name, "reduce_to_shape"),
            other => panic!("op[1] must be Passthrough(\"reduce_to_shape\"), got {:?}", other),
        }
        assert_eq!(
            ops[1].inputs,
            vec![raw_var, 7],
            "reduce_to_shape inputs must be (raw_mul_result, target), got {:?}",
            ops[1].inputs,
        );
    }

    /// Conv2d gradients are now implemented in source AD (the deferral is
    /// closed). Each `Conv2dBackward` adjoint lowers to a `MaterializeConvOutputGrad`
    /// op (reifying `grad` to the conv output shape) feeding a
    /// `PrimalOp::Conv2dBackward` op carrying the same kind/stride/padding —
    /// which in turn calls the runtime `nsl_conv2d_{input,weight,bias}_backward`
    /// FFI wrapping the verified `conv2d_backward` shared with the tape path.
    /// (Previously both arms refused loudly because the old transpose+matmul
    /// lowering was wrong for 4D convolution.)
    #[test]
    fn lower_adjoint_expr_conv_grads_emit_backward_ops() {
        use crate::wengert::ConvGradKind;
        for (kind, stride, padding) in [
            (ConvGradKind::Input, 1usize, 0usize),
            (ConvGradKind::Weight, 2, 1),
            (ConvGradKind::Bias, 1, 0),
        ] {
            let mut gen = AdjointGenerator::new(200);
            // grad=100, input=5, weight=6 — distinct for an unambiguous inputs check.
            let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(
                kind, 100, 5, 6, stride, padding,
            ));
            let ops = &gen.adjoint_ops;
            assert_eq!(
                ops.len(),
                2,
                "Conv2dBackward must lower to exactly 2 ops (materialize + backward), got {:?}",
                ops.iter().map(|o| &o.op).collect::<Vec<_>>(),
            );
            assert_eq!(
                ops[0].op,
                PrimalOp::MaterializeConvOutputGrad { stride, padding },
                "unexpected materialize op for kind {:?}",
                kind,
            );
            assert_eq!(
                ops[0].inputs,
                vec![100, 5, 6],
                "MaterializeConvOutputGrad inputs must be [grad, input, weight]",
            );
            let materialized = ops[0].result;
            assert_eq!(
                ops[1].op,
                PrimalOp::Conv2dBackward {
                    kind,
                    stride,
                    padding
                },
                "unexpected lowered op for kind {:?}",
                kind,
            );
            assert_eq!(
                ops[1].inputs,
                vec![materialized, 5, 6],
                "Conv2dBackward inputs must be [materialized_grad, input, weight]",
            );
        }
    }

    /// Regression for the hidden-allocation invariant: Conv2d's 2-3 sibling
    /// gradients (input/weight/bias) must share ONE `MaterializeConvOutputGrad`
    /// call for a given `grad` VarId, not one each. Before this memoization,
    /// a scalar sum-loss gradient (the common case — see
    /// `conv2d_source_ad_grad_e2e`) was reified — and allocated — up to 3x per
    /// conv2d node per backward step, a silent violation of the "no dynamic
    /// allocation on the backward hot path" invariant.
    #[test]
    fn conv2d_backward_siblings_share_one_materialize_call() {
        use crate::wengert::ConvGradKind;
        let mut gen = AdjointGenerator::new(200);
        // Same grad=100/input=5/weight=6 for all three sibling gradients,
        // as apply_ad_rule produces for one Conv2d node with a bias.
        let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(
            ConvGradKind::Input,
            100,
            5,
            6,
            1,
            0,
        ));
        let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(
            ConvGradKind::Weight,
            100,
            5,
            6,
            1,
            0,
        ));
        let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(
            ConvGradKind::Bias,
            100,
            5,
            6,
            1,
            0,
        ));

        let materialize_ops: Vec<_> = gen
            .adjoint_ops
            .iter()
            .filter(|op| matches!(op.op, PrimalOp::MaterializeConvOutputGrad { .. }))
            .collect();
        assert_eq!(
            materialize_ops.len(),
            1,
            "expected exactly 1 MaterializeConvOutputGrad shared by all 3 sibling \
             gradients, got {}: {:?}",
            materialize_ops.len(),
            materialize_ops,
        );
        let materialized = materialize_ops[0].result;

        let backward_ops: Vec<_> = gen
            .adjoint_ops
            .iter()
            .filter(|op| matches!(op.op, PrimalOp::Conv2dBackward { .. }))
            .collect();
        assert_eq!(backward_ops.len(), 3, "expected 3 Conv2dBackward ops (input/weight/bias)");
        for op in backward_ops {
            assert_eq!(
                op.inputs[0], materialized,
                "every sibling Conv2dBackward must consume the shared materialized grad, got {:?}",
                op,
            );
        }
    }

    /// Regression for the cache-collision hazard: the memoization key is the
    /// materialize op's FULL identity `(grad, input, weight, stride, padding)`,
    /// not `grad` alone. Two DIFFERENT-geometry `Conv2d` nodes that share one
    /// upstream `grad` VarId (as a broadcast scalar sum-loss seed does when it
    /// feeds two convs) must each reify their own correctly shaped grad — the
    /// conv output shape differs by node, so reusing node A's materialized
    /// tensor for node B would be a silent shape/gradient miscompile. Keyed on
    /// `grad` alone, node B would reuse node A's op; keyed on the full tuple,
    /// it does not.
    #[test]
    fn conv2d_backward_distinct_geometry_shared_grad_does_not_collide() {
        use crate::wengert::ConvGradKind;
        let mut gen = AdjointGenerator::new(300);
        // Node A: grad=100, input=5, weight=6. Node B: SAME grad=100, but
        // input=7, weight=8 (a differently shaped conv). Emit both siblings of
        // each node so the intra-node sharing path is also exercised.
        for kind in [ConvGradKind::Input, ConvGradKind::Weight] {
            let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(kind, 100, 5, 6, 1, 0));
        }
        for kind in [ConvGradKind::Input, ConvGradKind::Weight] {
            let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(kind, 100, 7, 8, 1, 0));
        }

        let materialize_ops: Vec<_> = gen
            .adjoint_ops
            .iter()
            .filter(|op| matches!(op.op, PrimalOp::MaterializeConvOutputGrad { .. }))
            .collect();
        assert_eq!(
            materialize_ops.len(),
            2,
            "two different-geometry conv nodes sharing grad=100 must each emit \
             their own MaterializeConvOutputGrad (no collision), got {}: {:?}",
            materialize_ops.len(),
            materialize_ops,
        );
        // The two materializations reify grad=100 against each node's own
        // input/weight — the inputs must differ, proving they are not aliased.
        assert_eq!(materialize_ops[0].inputs, vec![100, 5, 6]);
        assert_eq!(materialize_ops[1].inputs, vec![100, 7, 8]);
        let mat_a = materialize_ops[0].result;
        let mat_b = materialize_ops[1].result;
        assert_ne!(mat_a, mat_b, "materialized grads must be distinct VarIds");

        // Node A's Conv2dBackward ops consume mat_a; node B's consume mat_b.
        let bwd_inputs: Vec<u32> = gen
            .adjoint_ops
            .iter()
            .filter(|op| matches!(op.op, PrimalOp::Conv2dBackward { .. }))
            .map(|op| op.inputs[0])
            .collect();
        assert_eq!(
            bwd_inputs,
            vec![mat_a, mat_a, mat_b, mat_b],
            "each node's backward ops must consume that node's own materialized grad",
        );
    }

    /// The materialize key includes `stride`/`padding`, not just the tensor
    /// VarIds: two convs over the SAME `grad`/`input`/`weight` but with
    /// different stride (e.g. dilated vs plain) produce different output
    /// shapes, so they must not share a materialized grad either.
    #[test]
    fn conv2d_backward_materialize_key_includes_stride_padding() {
        use crate::wengert::ConvGradKind;
        let mut gen = AdjointGenerator::new(400);
        // Identical grad/input/weight VarIds; stride 1 vs 2.
        let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(
            ConvGradKind::Input,
            100,
            5,
            6,
            1,
            0,
        ));
        let _ = gen.lower_adjoint_expr(AdjointExpr::Conv2dBackward(
            ConvGradKind::Input,
            100,
            5,
            6,
            2,
            0,
        ));
        let materialize_ops: Vec<_> = gen
            .adjoint_ops
            .iter()
            .filter(|op| matches!(op.op, PrimalOp::MaterializeConvOutputGrad { .. }))
            .collect();
        assert_eq!(
            materialize_ops.len(),
            2,
            "differing stride must NOT collide in the materialize cache, got {}: {:?}",
            materialize_ops.len(),
            materialize_ops,
        );
    }

    /// CPDT Part II activation: the source-AD lowering of `Mul`'s backward
    /// must reduce each grad arm to the shape of the input whose adjoint it
    /// feeds, otherwise broadcast multiplies (e.g. `h[2,64] * scale[64]`)
    /// produce mismatched grad shapes and crash at
    /// `nsl_tensor_add_inplace`. This regression test confirms that BOTH
    /// arms of `Mul`'s backward emit a `reduce_to_shape` passthrough whose
    /// second input is the corresponding original primal input — the
    /// shape-anchor used by `nsl_tensor_reduce_to_shape` at runtime.
    #[test]
    fn mul_backward_emits_reduce_to_shape_per_arm() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("a".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("b".into()), vec![]),
                make_op(2, 2, PrimalOp::Mul, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);
        let reduce_ops: Vec<_> = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Passthrough(s) if s == "reduce_to_shape"))
            .collect();
        assert_eq!(
            reduce_ops.len(),
            2,
            "expected one reduce_to_shape per Mul backward arm, got {} (ops: {:?})",
            reduce_ops.len(),
            adjoint.ops.iter().map(|o| &o.op).collect::<Vec<_>>(),
        );
        // Targets are the original Mul inputs, in arm order: 0 then 1.
        let targets: Vec<VarId> = reduce_ops.iter().map(|op| op.inputs[1]).collect();
        assert_eq!(
            targets,
            vec![0, 1],
            "reduce_to_shape targets must match the primal inputs whose grads they shape",
        );
    }

    #[test]
    fn test_dead_gradient_elimination() {
        let ops = vec![
            make_op(0, 10, PrimalOp::Mul, vec![3, 4]),
            make_op(1, 11, PrimalOp::Add, vec![5, 6]),
            make_op(2, 12, PrimalOp::Neg, vec![7]),
            make_op(3, 3, PrimalOp::Add, vec![10, 8]),
            make_op(4, 13, PrimalOp::Mul, vec![9, 10]),
        ];
        let needed = HashSet::from([3_u32]);
        let pruned = eliminate_dead_gradients(&ops, &needed);
        assert!(pruned.len() < ops.len());
        assert!(pruned.iter().any(|op| op.result == 3));
    }

    #[test]
    fn test_dead_gradient_empty_needed() {
        let ops = vec![make_op(0, 10, PrimalOp::Add, vec![0, 1])];
        let pruned = eliminate_dead_gradients(&ops, &HashSet::new());
        assert!(pruned.is_empty());
    }

    /// Gap I.2+M regression: a `FusedCshaBackward` launch op and its
    /// seven `CshaFusedBackwardExtract` consumers form a coupled
    /// multi-result primitive. The launch's result VarId is never read
    /// directly (the extracts pull the real outputs from a side-channel
    /// cache), so the dead-gradient worklist walk would normally prune
    /// the launch.
    ///
    /// The Gap I.2+M fix lists the launch's result VarId as the FIRST
    /// input of every extract op, giving the worklist walk a data-
    /// dependency edge to reach back through. This test pins that
    /// behavior: marking any ONE extract output as needed must keep the
    /// launch op alive in the pruned output.
    #[test]
    fn gap_i2_launch_op_kept_alive_via_extract_dependency() {
        // Layout:
        //   op 0 → launch  (FusedCshaBackward, result = 100, inputs = [chain_key=1])
        //   op 1 → extract(0)  (result = 200, inputs = [launch_result=100, chain_key=1])
        //   op 2..7 → extract(1..6)  (results 201..206, same input shape)
        //   op 8 → Mul(200, 999) → result = 300  (consumer of the dq extract)
        //
        // `needed = {300}` — we need the consumer's adjoint. Dead-grad
        // elim must walk back: 300 → Mul → 200 → extract(0) → launch
        // → chain_key. Without the I.2+M fix, the extract's inputs
        // would be only [chain_key] and the launch would be pruned.
        let mut ops = vec![
            make_op(
                0,
                100,
                PrimalOp::FusedCshaBackward {
                    layer: "blocks.0".into(),
                },
                vec![1 /* chain_key */],
            ),
        ];
        for c in 0u8..=6u8 {
            ops.push(make_op(
                (c as u32) + 1,
                200 + c as VarId,
                PrimalOp::CshaFusedBackwardExtract { component: c },
                vec![100 /* launch_result */, 1 /* chain_key */],
            ));
        }
        // A downstream consumer of the dq extract (component 0).
        ops.push(make_op(8, 300, PrimalOp::Mul, vec![200, 999]));

        let needed = HashSet::from([300_u32]);
        let pruned = eliminate_dead_gradients(&ops, &needed);

        // Launch op MUST survive — it's now reachable via extract(0)'s
        // inputs[0].
        let launch_kept = pruned.iter().any(|o| {
            matches!(
                &o.op,
                PrimalOp::FusedCshaBackward { layer } if layer == "blocks.0"
            )
        });
        assert!(
            launch_kept,
            "I.2+M regression: FusedCshaBackward launch op must survive \
             dead-grad elimination when any extract's output is live; \
             pruned ops = {:?}",
            pruned.iter().map(|o| (o.result, &o.op)).collect::<Vec<_>>()
        );

        // The live extract (component 0) must also be kept.
        let extract0_kept = pruned.iter().any(|o| {
            matches!(
                &o.op,
                PrimalOp::CshaFusedBackwardExtract { component: 0 }
            )
        });
        assert!(
            extract0_kept,
            "extract(0) feeds the consumer at var 300 and must survive"
        );
    }

    #[test]
    fn test_saved_tensor_analysis() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![
                make_op(0, 10, PrimalOp::Constant(1.0), vec![]),
                make_op(1, 11, PrimalOp::Mul, vec![10, 0]),
            ],
            output: 10,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let saved = analyze_saved_tensors(&primal, &adjoint);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].var, 0);
    }

    #[test]
    fn test_saved_tensor_no_cross_reference() {
        let primal = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Input("x".into()), vec![])],
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![make_op(0, 10, PrimalOp::Constant(1.0), vec![])],
            output: 10,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        assert!(analyze_saved_tensors(&primal, &adjoint).is_empty());
    }

    // --- M40b: WengertExtractor tests ---

    use nsl_ast::expr::ExprKind;
    use nsl_ast::stmt::StmtKind;
    use nsl_lexer::Interner;

    fn test_expr(kind: ExprKind) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind,
            span: nsl_errors::Span::dummy(),
            id: nsl_ast::NodeId::dummy(),
        }
    }

    fn test_arg(value: nsl_ast::expr::Expr) -> nsl_ast::expr::Arg {
        nsl_ast::expr::Arg {
            name: None,
            value,
            span: nsl_errors::Span::dummy(),
        }
    }

    #[test]
    fn extract_simple_input() {
        let mut interner = Interner::new();
        // Pre-intern before borrowing into extractor
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let extractor = {
            let mut ext = WengertExtractor::new(&interner);
            ext.register_input(x_sym);
            ext
        };

        assert!(extractor.is_static_graph());
        let list = extractor.finalize();
        assert!(list.is_some());
        assert_eq!(list.unwrap().ops.len(), 1);
    }

    #[test]
    fn extract_registers_params_and_inputs() {
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let w_sym = nsl_ast::Symbol(interner.get_or_intern("W"));

        let extractor = {
            let mut ext = WengertExtractor::new(&interner);
            ext.register_input(x_sym);
            ext.register_param(w_sym);
            ext
        };

        let params = extractor.param_vars();
        assert_eq!(params.len(), 1);
        assert!(extractor.is_static_graph());
    }

    #[test]
    fn extract_method_transpose_as_primal_transpose() {
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let transpose_sym = nsl_ast::Symbol(interner.get_or_intern("transpose"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_input(x_sym);

        let call = test_expr(ExprKind::Call {
            callee: Box::new(test_expr(ExprKind::MemberAccess {
                object: Box::new(test_expr(ExprKind::Ident(x_sym))),
                member: transpose_sym,
            })),
            args: vec![
                test_arg(test_expr(ExprKind::IntLiteral(1))),
                test_arg(test_expr(ExprKind::IntLiteral(2))),
            ],
        });

        let result = extractor.extract_expr(&call);
        assert!(result.is_some());

        let list = extractor
            .finalize()
            .expect("static transpose extraction should succeed");
        let transpose_op = list.ops.last().expect("transpose op should be emitted");
        assert_eq!(transpose_op.op, PrimalOp::Transpose { dim0: 1, dim1: 2 });
        assert_eq!(transpose_op.inputs.len(), 1);
    }

    #[test]
    fn extract_function_transpose_preserves_literal_dims() {
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let transpose_sym = nsl_ast::Symbol(interner.get_or_intern("transpose"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_input(x_sym);

        let call = test_expr(ExprKind::Call {
            callee: Box::new(test_expr(ExprKind::Ident(transpose_sym))),
            args: vec![
                test_arg(test_expr(ExprKind::Ident(x_sym))),
                test_arg(test_expr(ExprKind::IntLiteral(1))),
                test_arg(test_expr(ExprKind::IntLiteral(2))),
            ],
        });

        let result = extractor.extract_expr(&call);
        assert!(result.is_some());

        let list = extractor
            .finalize()
            .expect("static transpose extraction should succeed");
        let transpose_op = list.ops.last().expect("transpose op should be emitted");
        assert_eq!(transpose_op.op, PrimalOp::Transpose { dim0: 1, dim1: 2 });
    }

    #[test]
    fn extract_detects_dynamic_while() {
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let while_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::While {
                condition: nsl_ast::expr::Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span: nsl_errors::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                body: nsl_ast::stmt::Block {
                    stmts: vec![],
                    span: nsl_errors::Span::dummy(),
                },
            },
            span: nsl_errors::Span::dummy(),
            id: nsl_ast::NodeId(1),
        };

        let result = extractor.extract_stmts(&[while_stmt]);
        assert!(!result);
        assert!(!extractor.is_static_graph());
    }

    #[test]
    fn extract_detects_dynamic_for() {
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let for_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::For {
                pattern: nsl_ast::pattern::Pattern {
                    kind: nsl_ast::pattern::PatternKind::Wildcard,
                    span: nsl_errors::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                iterable: nsl_ast::expr::Expr {
                    kind: ExprKind::IntLiteral(10),
                    span: nsl_errors::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                body: nsl_ast::stmt::Block {
                    stmts: vec![],
                    span: nsl_errors::Span::dummy(),
                },
            },
            span: nsl_errors::Span::dummy(),
            id: nsl_ast::NodeId(1),
        };

        let result = extractor.extract_stmts(&[for_stmt]);
        assert!(!result);
    }

    #[test]
    fn extractor_finalize_none_when_empty() {
        let interner = Interner::new();
        let extractor = WengertExtractor::new(&interner);
        assert!(extractor.finalize().is_none());
    }

    // --- Helper: build AST nodes for if/else tests ---

    fn dummy_span() -> nsl_errors::Span {
        nsl_errors::Span::dummy()
    }

    fn make_ident_expr(sym: nsl_ast::Symbol) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::Ident(sym),
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_float_expr(v: f64) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::FloatLiteral(v),
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_binop_expr(
        left: nsl_ast::expr::Expr,
        op: nsl_ast::operator::BinOp,
        right: nsl_ast::expr::Expr,
    ) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_neg_expr(operand: nsl_ast::expr::Expr) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::UnaryOp {
                op: nsl_ast::operator::UnaryOp::Neg,
                operand: Box::new(operand),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_return_stmt(expr: nsl_ast::expr::Expr) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::Return(Some(expr)),
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_let_stmt(sym: nsl_ast::Symbol, value: nsl_ast::expr::Expr) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: nsl_ast::pattern::Pattern {
                    kind: nsl_ast::pattern::PatternKind::Ident(sym),
                    span: dummy_span(),
                    id: nsl_ast::NodeId::next(),
                },
                type_ann: None,
                value: Some(value),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_block(stmts: Vec<nsl_ast::stmt::Stmt>) -> nsl_ast::stmt::Block {
        nsl_ast::stmt::Block {
            stmts,
            span: dummy_span(),
        }
    }

    fn make_if_stmt(
        condition: nsl_ast::expr::Expr,
        then_stmts: Vec<nsl_ast::stmt::Stmt>,
        else_stmts: Option<Vec<nsl_ast::stmt::Stmt>>,
    ) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::If {
                condition,
                then_block: make_block(then_stmts),
                elif_clauses: vec![],
                else_block: else_stmts.map(make_block),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_if_elif_stmt(
        condition: nsl_ast::expr::Expr,
        then_stmts: Vec<nsl_ast::stmt::Stmt>,
        elif_cond: nsl_ast::expr::Expr,
        elif_stmts: Vec<nsl_ast::stmt::Stmt>,
        else_stmts: Option<Vec<nsl_ast::stmt::Stmt>>,
    ) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::If {
                condition,
                then_block: make_block(then_stmts),
                elif_clauses: vec![(elif_cond, make_block(elif_stmts))],
                else_block: else_stmts.map(make_block),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    // ---------------------------------------------------------------
    // Task 1: Source AD accepts simple if/else (no longer marks dynamic)
    // ---------------------------------------------------------------

    #[test]
    fn extract_accepts_simple_if_else() {
        // f(x) = if x > 0: return x else: return 0.0
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let then_stmts = vec![make_return_stmt(make_ident_expr(x_sym))];
        let else_stmts = vec![make_return_stmt(make_float_expr(0.0))];
        let if_stmt = make_if_stmt(cond, then_stmts, Some(else_stmts));

        let result = extractor.extract_stmts(&[if_stmt]);
        assert!(result, "Source AD should accept simple if/else");
        assert!(
            extractor.is_static_graph(),
            "If/else should not make graph dynamic"
        );

        let list = extractor.finalize();
        assert!(list.is_some(), "Should produce a valid WengertList");
    }

    #[test]
    fn extract_still_rejects_while_loops() {
        // Ensure while loops are still dynamic
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let while_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::While {
                condition: nsl_ast::expr::Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span: dummy_span(),
                    id: nsl_ast::NodeId(0),
                },
                body: make_block(vec![]),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId(1),
        };

        assert!(!extractor.extract_stmts(&[while_stmt]));
        assert!(!extractor.is_static_graph());
    }

    #[test]
    fn extract_still_rejects_for_loops() {
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let for_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::For {
                pattern: nsl_ast::pattern::Pattern {
                    kind: nsl_ast::pattern::PatternKind::Wildcard,
                    span: dummy_span(),
                    id: nsl_ast::NodeId(0),
                },
                iterable: nsl_ast::expr::Expr {
                    kind: ExprKind::IntLiteral(10),
                    span: dummy_span(),
                    id: nsl_ast::NodeId(0),
                },
                body: make_block(vec![]),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId(1),
        };

        assert!(!extractor.extract_stmts(&[for_stmt]));
    }

    // ---------------------------------------------------------------
    // Task 2: Forward pass saves branch condition + emits Select
    // ---------------------------------------------------------------

    #[test]
    fn extract_if_else_produces_select_op() {
        // f(x) = if x > 0: return x*x else: return -x
        // Should produce: cond = Condition(Gt, x, 0); t = Mul(x,x); e = Neg(x); out = Select(cond, t, e)
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let then_stmts = vec![make_return_stmt(make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        ))];
        let else_stmts = vec![make_return_stmt(make_neg_expr(make_ident_expr(x_sym)))];
        let if_stmt = make_if_stmt(cond, then_stmts, Some(else_stmts));

        let result = extractor.extract_stmts(&[if_stmt]);
        assert!(result, "Should extract if/else successfully");

        let list = extractor.finalize().expect("Should produce WengertList");

        // Verify a Condition op was emitted
        let has_condition = list
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Condition(CompareKind::Gt)));
        assert!(
            has_condition,
            "Should contain a Condition(Gt) op for `x > 0`"
        );

        // Verify a Select op was emitted
        let select_ops: Vec<_> = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .collect();
        assert!(
            !select_ops.is_empty(),
            "Should contain at least one Select op"
        );

        // The Select op should have 3 inputs: [cond, true_val, false_val]
        for sel in &select_ops {
            assert_eq!(
                sel.inputs.len(),
                3,
                "Select should have 3 inputs (cond, true, false)"
            );
        }

        // The output should be the Select result
        let last_select = select_ops.last().unwrap();
        assert_eq!(
            list.output, last_select.result,
            "Output should be the Select result"
        );
    }

    #[test]
    fn extract_if_else_condition_saved_as_var() {
        // Verify the condition comparison result is captured in the Wengert list
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let then_stmts = vec![make_return_stmt(make_ident_expr(x_sym))];
        let else_stmts = vec![make_return_stmt(make_float_expr(0.0))];
        let if_stmt = make_if_stmt(cond, then_stmts, Some(else_stmts));

        extractor.extract_stmts(&[if_stmt]);
        let list = extractor.finalize().unwrap();

        // Find the Condition op
        let cond_op = list
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .unwrap();
        let cond_var = cond_op.result;

        // Find the Select op that uses this condition
        let select_op = list
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Select) && op.inputs[0] == cond_var);
        assert!(
            select_op.is_some(),
            "Select should reference the saved condition variable"
        );
    }

    // ---------------------------------------------------------------
    // Task 3: Backward pass produces conditional adjoint selection
    // ---------------------------------------------------------------

    #[test]
    fn adjoint_of_select_produces_conditional_gradient() {
        // Primal: x (param), 0.0 (const), cond = x > 0, t = x, e = 0.0, out = Select(cond, t, e)
        // This is relu: f(x) = max(x, 0)
        // Backward: d(out)/dx = cond ? 1 : 0
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Select, vec![2, 0, 1]),
            ],
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // The output (var 3) should have an adjoint
        assert!(gen.adjoint_of(3).is_some(), "Output should have adjoint");

        // The param x (var 0) should have an adjoint (from Select true-branch)
        assert!(
            gen.adjoint_of(0).is_some(),
            "Param x should have adjoint through Select"
        );

        // Verify Select ops appear in adjoint list (for conditional gradient selection)
        let has_select_in_adjoint = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Select));
        assert!(
            has_select_in_adjoint,
            "Backward pass should contain Select ops for conditional adjoint selection"
        );
    }

    #[test]
    fn adjoint_select_has_zero_for_inactive_branch() {
        // For Select(cond, a, b), backward should create:
        //   adj_a = Select(cond, adj_out, 0)
        //   adj_b = Select(cond, 0, adj_out)
        // Verify zero constants are generated
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Select, vec![2, 0, 1]),
            ],
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // Should have Constant(0.0) ops in the adjoint for the zero branches
        let zero_consts: Vec<_> = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Constant(v) if *v == 0.0))
            .collect();
        assert!(
            !zero_consts.is_empty(),
            "Backward pass should contain Constant(0.0) for inactive branch adjoints"
        );
    }

    #[test]
    fn adjoint_quadratic_linear_branch() {
        // f(x) = if x > 0 { x * x } else { -x }
        // Primal ops: x(0), zero(1), cond(2)=x>0, t(3)=x*x, e(4)=-x, out(5)=Select(cond, t, e)
        // d/dx for true branch (x*x): 2x
        // d/dx for false branch (-x): -1
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Mul, vec![0, 0]), // x * x (then-branch)
                make_op(4, 4, PrimalOp::Neg, vec![0]),    // -x (else-branch)
                make_op(5, 5, PrimalOp::Select, vec![2, 3, 4]),
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // x should have adjoint (from both Mul and Neg, gated by Select)
        assert!(gen.adjoint_of(0).is_some(), "x should have an adjoint");
        // The true result and false result should have adjoints
        assert!(
            gen.adjoint_of(3).is_some(),
            "true-branch result (x*x) should have adjoint"
        );
        assert!(
            gen.adjoint_of(4).is_some(),
            "false-branch result (-x) should have adjoint"
        );

        // Both branch results' adjoints should be Select ops (conditional on saved cond)
        let select_count = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(select_count >= 2,
            "Should have at least 2 Select ops in backward (one per branch input), got {select_count}");
    }

    // ---------------------------------------------------------------
    // Task 4: Nested if/else
    // ---------------------------------------------------------------

    #[test]
    fn extract_nested_if_else() {
        // f(x) = if x > 0 { if x > 1 { x*x*x } else { x*x } } else { 0 }
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        // Inner if: if x > 1 { x*x*x } else { x*x }
        let inner_cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(1.0),
        );
        let x_cubed = make_binop_expr(
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Mul,
                make_ident_expr(x_sym),
            ),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let x_squared = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );

        let inner_if = make_if_stmt(
            inner_cond,
            vec![make_return_stmt(x_cubed)],
            Some(vec![make_return_stmt(x_squared)]),
        );

        // Outer if: if x > 0 { <inner_if> } else { 0 }
        let outer_cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let outer_if = make_if_stmt(
            outer_cond,
            vec![inner_if],
            Some(vec![make_return_stmt(make_float_expr(0.0))]),
        );

        let result = extractor.extract_stmts(&[outer_if]);
        assert!(result, "Nested if/else should be extractable");
        assert!(extractor.is_static_graph());

        let list = extractor.finalize().expect("Should produce WengertList");

        // Should have 2 Condition ops (one per if)
        let cond_count = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .count();
        assert_eq!(
            cond_count, 2,
            "Should have 2 Condition ops for nested if/else, got {cond_count}"
        );

        // Should have at least 2 Select ops (one per merge point)
        let select_count = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            select_count >= 2,
            "Should have at least 2 Select ops, got {select_count}"
        );
    }

    #[test]
    fn adjoint_nested_branches_propagates() {
        // Nested: if x > 0 { if x > 1 { x*x } else { x } } else { 0 }
        // We build the flat Wengert for this manually:
        //   x(0), zero(1), one(2)
        //   cond_outer(3) = x > 0
        //   cond_inner(4) = x > 1
        //   t_inner(5) = x * x
        //   inner_select(6) = Select(cond_inner, t_inner, x)  -- inner merge
        //   outer_select(7) = Select(cond_outer, inner_select, zero) -- outer merge
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Constant(1.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                make_op(5, 5, PrimalOp::Mul, vec![0, 0]),
                make_op(6, 6, PrimalOp::Select, vec![4, 5, 0]),
                make_op(7, 7, PrimalOp::Select, vec![3, 6, 1]),
            ],
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // x (var 0) should have an adjoint — gradient propagates through both Select ops
        assert!(
            gen.adjoint_of(0).is_some(),
            "x should have adjoint through nested Select"
        );

        // The adjoint list should contain multiple Select ops (from both nesting levels)
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Adjoint should have >= 2 Select ops for nested branching, got {adj_selects}"
        );
    }

    // ---------------------------------------------------------------
    // Task 5: If-without-else (identity for missing branch)
    // ---------------------------------------------------------------

    #[test]
    fn extract_if_without_else() {
        // f(x):
        //   let y = x
        //   if x > 5: y = x * 2
        //   return y
        // For x > 5: f(x) = 2x, f'(x) = 2
        // For x <= 5: f(x) = x, f'(x) = 1
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let y_sym = nsl_ast::Symbol(interner.get_or_intern("y"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        // let y = x
        let let_y = make_let_stmt(y_sym, make_ident_expr(x_sym));

        // if x > 5: let y = x * 2.0
        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(5.0),
        );
        let x_times_2 = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_float_expr(2.0),
        );
        let reassign_y = make_let_stmt(y_sym, x_times_2);
        let if_stmt = make_if_stmt(cond, vec![reassign_y], None); // No else!

        // return y
        let return_y = make_return_stmt(make_ident_expr(y_sym));

        let result = extractor.extract_stmts(&[let_y, if_stmt, return_y]);
        assert!(result, "If-without-else should be extractable");
        assert!(extractor.is_static_graph());

        let list = extractor.finalize().expect("Should produce WengertList");

        // Should have a Select op that chooses between modified y (x*2) and original y (x)
        let select_ops: Vec<_> = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .collect();
        assert!(
            !select_ops.is_empty(),
            "If-without-else should produce Select op (modified vs identity)"
        );

        // The Select should use the pre-branch value as the false (else) input
        // This is the identity case — when condition is false, y retains its original value
        let sel = select_ops.last().unwrap();
        assert_eq!(sel.inputs.len(), 3);
    }

    #[test]
    fn adjoint_if_without_else_identity_passthrough() {
        // Manual Wengert for: y = x; if x > 5: y = x * 2; return y
        //   x(0), five(1), two(2)
        //   cond(3) = x > 5
        //   modified_y(4) = x * 2
        //   y_merged(5) = Select(cond, modified_y, x)  // else branch is identity (x)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(5.0), vec![]),
                make_op(2, 2, PrimalOp::Constant(2.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(4, 4, PrimalOp::Mul, vec![0, 2]), // x * 2
                make_op(5, 5, PrimalOp::Select, vec![3, 4, 0]), // cond ? x*2 : x
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // x should have an adjoint
        assert!(gen.adjoint_of(0).is_some(), "x should have adjoint");

        // The backward pass should produce Select ops for conditional adjoint
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 1,
            "Backward should use Select for conditional adjoint, got {adj_selects}"
        );
    }

    #[test]
    fn extract_elif_desugars_to_nested() {
        // f(x) = if x > 1: x*x*x  elif x > 0: x*x  else: 0
        // This tests elif desugaring (single elif clause + else)
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let x_cubed = make_binop_expr(
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Mul,
                make_ident_expr(x_sym),
            ),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let x_squared = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );

        let if_stmt = make_if_elif_stmt(
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Gt,
                make_float_expr(1.0),
            ),
            vec![make_return_stmt(x_cubed)],
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Gt,
                make_float_expr(0.0),
            ),
            vec![make_return_stmt(x_squared)],
            Some(vec![make_return_stmt(make_float_expr(0.0))]),
        );

        let result = extractor.extract_stmts(&[if_stmt]);
        assert!(result, "elif should be desugared and extractable");
        assert!(extractor.is_static_graph());

        let list = extractor.finalize().expect("Should produce WengertList");

        // Should have 2 Condition ops (outer and elif)
        let cond_count = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .count();
        assert_eq!(
            cond_count, 2,
            "Should have 2 Condition ops for if/elif/else, got {cond_count}"
        );
    }

    // ---------------------------------------------------------------
    // Integration: end-to-end extraction + adjoint generation
    // ---------------------------------------------------------------

    #[test]
    fn end_to_end_relu_extraction_and_adjoint() {
        // Build AST for: if x > 0 { return x } else { return 0 }
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let if_stmt = make_if_stmt(
            cond,
            vec![make_return_stmt(make_ident_expr(x_sym))],
            Some(vec![make_return_stmt(make_float_expr(0.0))]),
        );

        assert!(extractor.extract_stmts(&[if_stmt]));
        let primal = extractor.finalize().unwrap();

        // Generate adjoint
        let max_primal_var = primal.ops.iter().map(|op| op.result).max().unwrap_or(0);
        let mut gen = AdjointGenerator::new(max_primal_var + 1);
        let adjoint = gen.generate(&primal);

        // Verify the full pipeline works
        assert!(!adjoint.ops.is_empty(), "Adjoint list should not be empty");
        // x should have an adjoint
        let x_var = primal
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(ref n) if n == "x"))
            .map(|op| op.result)
            .unwrap();
        assert!(
            gen.adjoint_of(x_var).is_some(),
            "Param x should have adjoint in relu"
        );
    }

    #[test]
    fn end_to_end_quadratic_branch_extraction_and_adjoint() {
        // Build AST for: if x > 0 { return x*x } else { return -x }
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let x_squared = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let neg_x = make_neg_expr(make_ident_expr(x_sym));
        let if_stmt = make_if_stmt(
            cond,
            vec![make_return_stmt(x_squared)],
            Some(vec![make_return_stmt(neg_x)]),
        );

        assert!(extractor.extract_stmts(&[if_stmt]));
        let primal = extractor.finalize().unwrap();

        let max_var = primal.ops.iter().map(|op| op.result).max().unwrap_or(0);
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        assert!(!adjoint.ops.is_empty());
        // Both branches contribute to the adjoint of x
        let x_var = primal
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(ref n) if n == "x"))
            .map(|op| op.result)
            .unwrap();
        assert!(gen.adjoint_of(x_var).is_some());
    }

    // ---------------------------------------------------------------
    // Task 6: E2E-style tests for branching gradients
    //
    // These tests exercise the full pipeline (extractor → primal list →
    // adjoint generator) for the three canonical patterns from the plan.
    // Full NSL-file E2E tests are not practical without a live runtime,
    // so we use the AST-level API which is exactly what the compiler uses.
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // Task 6.1: Leaky ReLU
    //   f(x) = if x > 0 { x } else { 0.01 * x }
    //   f'(x) = 1.0  when x > 0
    //   f'(x) = 0.01 when x <= 0
    //
    // Primal Wengert:
    //   x(0), zero(1), alpha(2)=0.01
    //   cond(3) = x > 0          Condition(Gt, x, zero)
    //   scaled(4) = alpha * x    Mul(alpha, x)
    //   out(5) = Select(cond, x, scaled)
    //
    // Backward:
    //   adj(x) should come from Select(cond, 1·adj_out, 0.01·adj_out)
    // ---------------------------------------------------------------
    #[test]
    fn leaky_relu_gradient_structure() {
        // Build using the AST extractor to stay close to the real compiler path.
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let alpha_sym = nsl_ast::Symbol(interner.get_or_intern("alpha"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);
        extractor.register_input(alpha_sym);

        // cond: x > 0
        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        // then branch: return x
        let then_branch = vec![make_return_stmt(make_ident_expr(x_sym))];
        // else branch: return alpha * x
        let alpha_times_x = make_binop_expr(
            make_ident_expr(alpha_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let else_branch = vec![make_return_stmt(alpha_times_x)];

        let if_stmt = make_if_stmt(cond, then_branch, Some(else_branch));

        let ok = extractor.extract_stmts(&[if_stmt]);
        assert!(ok, "Leaky ReLU if/else should be extractable by source AD");
        assert!(extractor.is_static_graph(), "Should remain static graph");

        let primal = extractor.finalize().expect("Should produce WengertList");

        // Structural checks on the primal
        let cond_count = primal
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .count();
        assert_eq!(cond_count, 1, "Leaky ReLU needs exactly 1 Condition op");

        let select_count = primal
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert_eq!(select_count, 1, "Leaky ReLU needs exactly 1 Select op");

        // Generate the adjoint
        let max_var = primal.ops.iter().map(|op| op.result).max().unwrap_or(0);
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        // x (the param) must have an adjoint
        let x_var = primal
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(ref n) if n == "x"))
            .map(|op| op.result)
            .expect("x must be in primal");
        assert!(
            gen.adjoint_of(x_var).is_some(),
            "Param x should have an adjoint in leaky ReLU"
        );

        // The adjoint must use Select to conditionally route gradient
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 1,
            "Leaky ReLU backward should use at least 1 Select op for conditional gradient, \
             got {adj_selects}"
        );

        // There must be no tape-style Input ops in the adjoint (source AD is tape-free)
        let tape_ops = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Input(_)))
            .count();
        assert_eq!(
            tape_ops, 0,
            "Source AD adjoint must not contain Input (tape-push) ops, got {tape_ops}"
        );
    }

    // ---------------------------------------------------------------
    // Task 6.2: Nested clamp gradient
    //   f(x, lo, hi) = if x < lo { lo } else { if x > hi { hi } else { x } }
    //   f'(x) = 0  when x < lo  (gradient killed — lo is constant)
    //   f'(x) = 0  when x > hi  (gradient killed — hi is constant)
    //   f'(x) = 1  when lo <= x <= hi (pass-through)
    //
    // We build the flat Wengert directly (as the adjoint generator sees it):
    //   x(0), lo(1), hi(2)
    //   cond_low(3)  = x < lo       Condition(Lt)
    //   cond_high(4) = x > hi       Condition(Gt)
    //   inner(5) = Select(cond_high, hi, x)   -- inner if: x>hi ? hi : x
    //   out(6)   = Select(cond_low, lo, inner) -- outer if: x<lo ? lo : inner
    //
    // In the true branch of cond_low the result is lo (Input, not Param of x),
    // so x gets gradient 0 from that branch.  In the false branch x's gradient
    // flows through inner_select, again gated by cond_high.
    // ---------------------------------------------------------------
    #[test]
    fn nested_clamp_gradient_structure() {
        // x is the param we differentiate w.r.t.; lo and hi are inputs (constants).
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("lo".into()), vec![]),
                make_op(2, 2, PrimalOp::Input("hi".into()), vec![]),
                // cond_low = x < lo
                make_op(3, 3, PrimalOp::Condition(CompareKind::Lt), vec![0, 1]),
                // cond_high = x > hi
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                // inner = cond_high ? hi : x
                make_op(5, 5, PrimalOp::Select, vec![4, 2, 0]),
                // out = cond_low ? lo : inner
                make_op(6, 6, PrimalOp::Select, vec![3, 1, 5]),
            ],
            output: 6,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let max_var = 6;
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        // x must have an adjoint
        assert!(
            gen.adjoint_of(0).is_some(),
            "x should have an adjoint through the nested clamp Select chain"
        );

        // The adjoint should use multiple Select ops to gate the gradient
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Nested clamp backward needs >= 2 Select ops (one per nesting level), \
             got {adj_selects}"
        );

        // lo (var 1) and hi (var 2) are Inputs — no gradient expected
        // (they are not Params so AdjointGenerator will not produce a path to them
        //  that matters for x's gradient, but the adjoint list should still be sound)
        assert!(!adjoint.ops.is_empty(), "Adjoint list must not be empty");
    }

    // ---------------------------------------------------------------
    // Task 6.3: Multiple differentiable variables in branches
    //   f(cond_flag, a, b) = if cond_flag > 0.5 { a * b } else { a + b }
    //
    // Gradient w.r.t. a:
    //   true  branch: d(a*b)/da = b  → adjoint contributes b * adj_out
    //   false branch: d(a+b)/da = 1  → adjoint contributes 1 * adj_out
    //   Combined via Select: adj_a = Select(cond, b·adj_out, adj_out)
    //
    // Gradient w.r.t. b:
    //   true  branch: d(a*b)/db = a  → a * adj_out
    //   false branch: d(a+b)/db = 1  → adj_out
    //   Combined via Select: adj_b = Select(cond, a·adj_out, adj_out)
    //
    // Primal Wengert (flat):
    //   cond_raw(0), a(1), b(2)  — all Params
    //   half(3) = Constant(0.5)
    //   cond(4)  = cond_raw > 0.5
    //   t_mul(5) = a * b
    //   f_add(6) = a + b
    //   out(7)   = Select(cond, t_mul, f_add)
    // ---------------------------------------------------------------
    #[test]
    fn multi_variable_branch_adjoint_structure() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("cond_flag".into()), vec![]),
                make_op(1, 1, PrimalOp::Param("a".into()), vec![]),
                make_op(2, 2, PrimalOp::Param("b".into()), vec![]),
                make_op(3, 3, PrimalOp::Constant(0.5), vec![]),
                // cond = cond_flag > 0.5
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 3]),
                // true branch: a * b
                make_op(5, 5, PrimalOp::Mul, vec![1, 2]),
                // false branch: a + b
                make_op(6, 6, PrimalOp::Add, vec![1, 2]),
                // output: Select(cond, t_mul, f_add)
                make_op(7, 7, PrimalOp::Select, vec![4, 5, 6]),
            ],
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let max_var = 7;
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        // Both a and b must have adjoints
        assert!(
            gen.adjoint_of(1).is_some(),
            "Param a should have an adjoint"
        );
        assert!(
            gen.adjoint_of(2).is_some(),
            "Param b should have an adjoint"
        );

        // The true-branch result and false-branch result should have adjoints
        assert!(
            gen.adjoint_of(5).is_some(),
            "True-branch Mul result should have adjoint"
        );
        assert!(
            gen.adjoint_of(6).is_some(),
            "False-branch Add result should have adjoint"
        );

        // The backward pass must contain Select ops (one per branch input of the primal Select)
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Multi-variable branch backward needs >= 2 Select ops, got {adj_selects}"
        );

        // No tape-style markers in source AD adjoint
        let tape_inputs = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Input(_)))
            .count();
        assert_eq!(
            tape_inputs, 0,
            "Source AD adjoint must be tape-free (no Input ops), got {tape_inputs}"
        );
    }

    // ---------------------------------------------------------------
    // Task 7: Source AD produces no tape push/pop for branching code.
    //
    // Tape-based AD would introduce Input("__tape_push__") / Param markers
    // to record intermediate values at runtime.  Source AD encodes all
    // branching information structurally via Condition + Select ops, so
    // the adjoint Wengert list must be entirely free of tape markers.
    //
    // We verify this for three representative branching functions.
    // ---------------------------------------------------------------
    #[test]
    fn source_ad_no_tape_ops_for_simple_branch() {
        // f(x) = if x > 0 { x * x } else { -x }
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Mul, vec![0, 0]),
                make_op(4, 4, PrimalOp::Neg, vec![0]),
                make_op(5, 5, PrimalOp::Select, vec![2, 3, 4]),
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // Structural: must not contain Input ops (tape-push markers)
        let has_tape_input = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Input(_)));
        assert!(
            !has_tape_input,
            "Source AD adjoint for branching code must not contain tape Input ops"
        );

        // Must not contain Param ops (tape-checkpoint markers) beyond what primal already has
        // (AdjointGenerator never introduces Param ops; it only uses Constant/Select/arithmetic)
        let has_new_param = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Param(_)));
        assert!(
            !has_new_param,
            "Source AD adjoint must not introduce new Param ops"
        );

        // Must contain Select ops to encode the branch gradient
        let has_select = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Select));
        assert!(
            has_select,
            "Source AD adjoint must use Select (not tape push/pop) for branch gradients"
        );

        // x must have an adjoint — the whole point of AD
        assert!(gen.adjoint_of(0).is_some(), "x must have adjoint");
    }

    #[test]
    fn source_ad_no_tape_ops_for_nested_branch() {
        // Nested: if x > 0 { if x > 1 { x*x } else { x } } else { 0 }
        // Flat Wengert (same as adjoint_nested_branches_propagates)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Constant(1.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                make_op(5, 5, PrimalOp::Mul, vec![0, 0]),
                make_op(6, 6, PrimalOp::Select, vec![4, 5, 0]),
                make_op(7, 7, PrimalOp::Select, vec![3, 6, 1]),
            ],
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // No tape-style ops in the backward pass
        assert!(
            !adjoint
                .ops
                .iter()
                .any(|op| matches!(&op.op, PrimalOp::Input(_))),
            "Nested-branch adjoint must be tape-free"
        );
        assert!(
            !adjoint
                .ops
                .iter()
                .any(|op| matches!(&op.op, PrimalOp::Param(_))),
            "Nested-branch adjoint must not introduce Param ops"
        );

        // Must be richer in Select ops than a straight-line function would be
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Nested-branch adjoint should have >= 2 Select ops (source AD encodes branches \
             structurally, not via tape), got {adj_selects}"
        );
    }

    #[test]
    fn source_ad_adjoint_op_count_dominated_by_arithmetic_not_tape() {
        // For a branching function the adjoint ops should be entirely arithmetic /
        // Select / Constant — zero tape-style bookkeeping.
        //
        // Tape AD would add O(branch_depth) Input ops for each checkpoint save.
        // Source AD adds exactly 0.
        //
        // f(x) = if x > 0 { x } else { 0.01 * x }  (leaky ReLU)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.01), vec![]),
                make_op(2, 2, PrimalOp::Constant(0.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                make_op(4, 4, PrimalOp::Mul, vec![1, 0]), // 0.01 * x
                make_op(5, 5, PrimalOp::Select, vec![3, 0, 4]), // cond ? x : 0.01*x
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // Count op types
        let tape_ops = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Input(_) | PrimalOp::Param(_)))
            .count();
        let structural_ops = adjoint
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.op,
                    PrimalOp::Select | PrimalOp::Condition(_) | PrimalOp::Constant(_)
                )
            })
            .count();
        let arithmetic_ops = adjoint
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.op,
                    PrimalOp::Add | PrimalOp::Mul | PrimalOp::Neg | PrimalOp::Sub
                )
            })
            .count();

        assert_eq!(
            tape_ops, 0,
            "Source AD for leaky ReLU must have 0 tape ops, got {tape_ops}"
        );
        assert!(
            structural_ops + arithmetic_ops > 0,
            "Adjoint must have actual gradient computation ops"
        );

        // x must have an adjoint
        assert!(gen.adjoint_of(0).is_some(), "x must have an adjoint");
    }

    // -----------------------------------------------------------------
    // Cycle-10 §5.3 Task 5: WengertExtractor checkpoint policy tests
    // -----------------------------------------------------------------

    #[test]
    fn cycle10_task5_empty_policies_no_transitive_stamping() {
        let interner = Interner::new();
        let mut ext = WengertExtractor::new(&interner);
        // Manually push a synthetic op so we can observe (no extraction
        // needed for the byte-identity contract).
        let v = ext.alloc_var();
        let op_id = ext.list.ops.len() as u32;
        ext.push_op(WengertOp {
            id: op_id,
            result: v,
            op: PrimalOp::Relu,
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });

        // With no policies configured, apply_checkpoint_policy is a no-op.
        ext.apply_checkpoint_policy("any_fn_name");

        assert_eq!(ext.list.ops.len(), 1, "no marker should be emitted");
        assert!(
            !ext.list.ops[0].checkpointed,
            "no stamping should occur with empty policies"
        );
        assert_eq!(ext.next_subgraph_id, 0, "allocator must not advance");
    }

    #[test]
    fn cycle10_task5_full_policy_stamps_and_emits_marker() {
        let interner = Interner::new();
        let mut ext = WengertExtractor::new(&interner);

        // Simulate a prologue: Input (entry root) + Relu + Sigmoid (intermediates).
        let v_in = ext.alloc_var();
        ext.push_op(WengertOp {
            id: 0,
            result: v_in,
            op: PrimalOp::Input("x".to_string()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
        let v1 = ext.alloc_var();
        ext.push_op(WengertOp {
            id: 1,
            result: v1,
            op: PrimalOp::Relu,
            inputs: vec![v_in],
            saved_for_backward: false,
            checkpointed: false,
        });
        let v2 = ext.alloc_var();
        ext.push_op(WengertOp {
            id: 2,
            result: v2,
            op: PrimalOp::Sigmoid,
            inputs: vec![v1],
            saved_for_backward: false,
            checkpointed: false,
        });

        // Configure a Full policy for "csha_fwd".
        let mut policies = HashMap::new();
        policies.insert("csha_fwd".to_string(), CheckpointPolicy::Full);
        let mut ext = ext.with_checkpoint_policies(policies);
        ext.apply_checkpoint_policy("csha_fwd");

        // Expect: Input root NOT stamped, Relu + Sigmoid stamped, ONE
        // PrologueRecompute marker emitted at the tail.
        assert_eq!(
            ext.list.ops.len(),
            4,
            "exactly one marker op should be emitted"
        );
        assert!(
            !ext.list.ops[0].checkpointed,
            "Input root must not be stamped"
        );
        assert!(ext.list.ops[1].checkpointed, "Relu must be stamped");
        assert!(ext.list.ops[2].checkpointed, "Sigmoid must be stamped");
        match ext.list.ops[3].op {
            PrimalOp::PrologueRecompute { subgraph_id } => {
                assert_eq!(subgraph_id, SubgraphId(0), "fresh id starts at 0");
            }
            ref other => panic!("expected PrologueRecompute, got {:?}", other),
        }

        // Allocator advanced exactly once.
        assert_eq!(ext.next_subgraph_id, 1);
    }

    #[test]
    fn cycle10_task5_subgraph_id_allocator_monotonic() {
        let interner = Interner::new();
        let mut ext = WengertExtractor::new(&interner);
        let a = ext.alloc_subgraph_id();
        let b = ext.alloc_subgraph_id();
        let c = ext.alloc_subgraph_id();
        assert_eq!(a, SubgraphId(0));
        assert_eq!(b, SubgraphId(1));
        assert_eq!(c, SubgraphId(2));
    }

    #[test]
    fn cycle10_task6_with_checkpoint_policies_is_byte_identity_when_empty() {
        // Wire-up contract: the loader passes an empty HashMap when no
        // @checkpoint decorators are present. The extractor must be
        // observationally indistinguishable from one constructed without
        // any installer.
        let interner = Interner::new();
        let ext_baseline = WengertExtractor::new(&interner);
        let ext_wired = WengertExtractor::new(&interner)
            .with_checkpoint_policies(HashMap::new());

        assert!(ext_baseline.checkpoint_policies.is_empty());
        assert!(ext_wired.checkpoint_policies.is_empty());
        assert_eq!(ext_baseline.next_subgraph_id, ext_wired.next_subgraph_id);
        assert_eq!(ext_baseline.list.ops.len(), ext_wired.list.ops.len());
    }

    #[test]
    fn cycle10_task6_with_checkpoint_policies_carries_full_for_named_fn() {
        // Simulate the loader → CompileOptions → WengertExtractor flow:
        // a single fn_name keyed to Full lands in the extractor and
        // apply_checkpoint_policy fires only for that name.
        let interner = Interner::new();
        let mut policies = HashMap::new();
        policies.insert("forward".to_string(), CheckpointPolicy::Full);
        let mut ext = WengertExtractor::new(&interner)
            .with_checkpoint_policies(policies);

        // Inject one intermediate op so stamping has something to flip.
        let v = ext.alloc_var();
        ext.push_op(WengertOp {
            id: 0,
            result: v,
            op: PrimalOp::Relu,
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });

        // Applying for a non-keyed fn is a no-op (byte-identity).
        ext.apply_checkpoint_policy("not_forward");
        assert!(!ext.list.ops[0].checkpointed);
        assert_eq!(ext.list.ops.len(), 1);

        // Applying for the keyed fn flips the stamp + emits exactly one
        // PrologueRecompute marker.
        ext.apply_checkpoint_policy("forward");
        assert!(ext.list.ops[0].checkpointed);
        assert_eq!(ext.list.ops.len(), 2);
        assert!(matches!(
            ext.list.ops[1].op,
            PrimalOp::PrologueRecompute { .. }
        ));
    }

    #[test]
    fn cycle10_task5_policy_for_unrelated_fn_is_noop() {
        let interner = Interner::new();
        let mut ext = WengertExtractor::new(&interner);
        let v = ext.alloc_var();
        ext.push_op(WengertOp {
            id: 0,
            result: v,
            op: PrimalOp::Relu,
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });

        // Policy set for "other_fn", but we apply for "this_fn".
        let mut policies = HashMap::new();
        policies.insert("other_fn".to_string(), CheckpointPolicy::Full);
        let mut ext = ext.with_checkpoint_policies(policies);
        ext.apply_checkpoint_policy("this_fn");

        assert_eq!(ext.list.ops.len(), 1, "no marker should be emitted");
        assert!(
            !ext.list.ops[0].checkpointed,
            "no stamping for mismatched fn name"
        );
    }
}
