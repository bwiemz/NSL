//! Section 6d of the train block's source-AD arm: CCR's adjoint-region
//! last-use freeing. Adjoint intermediates (the dx-chain temporaries) are
//! the binding activation wall of a checkpointed build; a `FreeTensor`
//! marker is inserted after each adjoint var's last use, protecting every
//! parameter-gradient adjoint (and the inputs of the ops producing them),
//! with the weight-gradient fusion chains planned on the pre-insertion
//! tape so the inserter keeps them contiguous.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 116 lines,
//! 7 inputs ([`CcrAdjointFreesInputs`]); a pure tape rewrite — no
//! IR is emitted, the adjoint is updated in place. The train-block CLIF
//! snapshots (`tests/train_clif_snapshots.rs`) pin the resulting free
//! placement on the checkpointed fixtures.

use crate::compiler::Compiler;

/// Every binding of `compile_train_block_inner` the last-use freeing
/// reads; names are the driver's.
pub(crate) struct CcrAdjointFreesInputs<'a> {
    /// The adjoint tape; the frees are inserted in place.
    pub(crate) adjoint: &'a mut crate::wengert::WengertList,
    /// The next fresh VarId for the inserted markers (advanced past both tapes first).
    pub(crate) ccr_fresh: crate::wengert::VarId,
    /// The CCR plan; `None` means no last-use freeing.
    pub(crate) ccr_plan: &'a Option<crate::ccr::CcrPlan>,
    pub(crate) effective_primal: &'a crate::wengert::WengertList,
    /// The forward extractor (its named parameter VarIds seed the protected set).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    pub(crate) fase_hook_active: bool,
    pub(crate) generator: &'a crate::source_ad::AdjointGenerator,
}

impl Compiler<'_> {
    /// Insert the CCR adjoint-region last-use frees (see the module header).
    pub(crate) fn insert_ccr_adjoint_frees(&self, inputs: CcrAdjointFreesInputs<'_>) {
        let CcrAdjointFreesInputs {
            adjoint,
            mut ccr_fresh,
            ccr_plan,
            effective_primal,
            extractor,
            fase_hook_active,
            generator,
        } = inputs;

        // 6d. CCR: adjoint-region last-use freeing. The 500M/seq1024
        // per-surface OOM decomposition showed adjoint intermediates
        // (dx-chain temporaries) are the binding activation wall —
        // they all lived to the end-of-backward bulk free. Insert a
        // FreeTensor after each adjoint var's last use. Protected:
        // every param-gradient adjoint (consumed by the FASE hook or
        // by post-lowering grad collection) plus the inputs of the
        // ops producing them (the hook's reduce_to_shape identity
        // path emits its own extra free for those raw grads).
        if ccr_plan.is_some() {
            let mut ccr_protect: std::collections::HashSet<crate::wengert::VarId> =
                std::collections::HashSet::new();
            for (param_name, primal_vid) in extractor.named_param_var_ids() {
                if !self.is_trainable_param_name(param_name) {
                    continue;
                }
                if let Some(adj_vid) = generator.adjoint_of(*primal_vid) {
                    ccr_protect.insert(adj_vid);
                }
            }
            // Item 7 x CCR: the fusable weight-gradient chains, planned
            // on the adjoint BEFORE last-use freeing rewrites it.
            // `wgrad_fusion::plan` requires the chain's three ops to be
            // contiguous, and last-use freeing lands `FreeTensor(a_t)`
            // between the matmul and the reduce — which silently took
            // the fusion count to ZERO on every `--checkpoint-blocks`
            // build. Planning here, on the pre-insertion tape, is what
            // lets the inserter keep those chains adjacent; the lowerer
            // re-derives the same plan from VarIds afterwards.
            //
            // GATED ON `fase_hook_active` TOO, mirroring the lowerer:
            // it plans only when `on_param_grad` is present
            // (wengert_lower.rs), which requires the FASE hook. Without
            // this the two sides diverge on any `--checkpoint-blocks`
            // build that will NOT fuse — accumulation == 1 makes FASE
            // Passthrough for EVERY optimizer, and
            // `--pretrain-optimized` turns `fuse_wgrad_accum` on with no
            // accumulation precondition — and CCR would then delete
            // `FreeTensor(a_t)` markers for chains nobody fuses. That is
            // not a leak (the transpose result is a view, swept by the
            // end-of-backward bulk free) but it pins the base
            // activation's buffer until then, a regression in exactly
            // the pass whose purpose is cutting the adjoint peak.
            // Found by adversarial review, measured at 3 lost markers on
            // an SGD/no-accumulation build.
            //
            // The seed is still the param-adjoint set (before the input
            // expansion below widens it), a SUPERSET of the hook's
            // `param_adj_set`, which additionally requires an accum slot
            // and a primal value — both built later. That residual
            // over-inclusion is bounded and one-directional: at worst a
            // param with no accum slot loses its `a_t` marker to the
            // bulk free. It cannot mis-fuse anything, because the
            // lowerer's plan — not this one — decides what is elided.
            let wgrad_chains = if self.compile_options.fusion.wgrad_accum
                && fase_hook_active
            {
                Some(crate::wgrad_fusion::plan(adjoint, &ccr_protect))
            } else {
                None
            };
            for op in &adjoint.ops {
                if ccr_protect.contains(&op.result) {
                    for input in &op.inputs {
                        ccr_protect.insert(*input);
                    }
                }
            }
            ccr_fresh = ccr_fresh.max(
                effective_primal
                    .ops
                    .iter()
                    .map(|o| o.result)
                    .chain(adjoint.ops.iter().map(|o| o.result))
                    .max()
                    .unwrap_or(0)
                    + 1,
            );
            let n = crate::ccr::insert_adjoint_last_use_frees(
                adjoint,
                &ccr_protect,
                &mut ccr_fresh,
                wgrad_chains.as_ref(),
            );
            if std::env::var("NSL_CCR_DEBUG").is_ok() {
                nsl_runtime::nsl_log!(INFO, "ccr", "[ccr] adjoint last-use frees inserted: {n}");
                if let Some(ref chains) = wgrad_chains {
                    nsl_runtime::nsl_log!(INFO, "ccr", 
                        "[ccr] wgrad chains kept contiguous: {}",
                        chains.by_reduce_result.len()
                    );
                }
            }
            // The invariant this protection exists to hold: a chain the
            // pre-insertion plan admitted must still be admissible
            // afterwards. A drop here is exactly the silent-inertness
            // defect that motivated the fix (the lowerer would re-plan,
            // find fewer chains, and report a smaller count with no
            // error), so say so loudly rather than losing a fusion to a
            // future free-placement change.
            if let Some(ref before) = wgrad_chains {
                let after = crate::wgrad_fusion::plan(adjoint, &ccr_protect);
                if after.by_reduce_result.len() < before.by_reduce_result.len() {
                    nsl_runtime::nsl_log!(WARN, "codegen", 
                        "warning: [ccr] last-use freeing broke {} weight-gradient \
                         fusion chain(s) ({} admissible before, {} after) — \
                         --fuse-wgrad-accum will silently fuse fewer chains on \
                         this build. This is a compiler defect, not a property \
                         of your program; please report it.",
                        before.by_reduce_result.len() - after.by_reduce_result.len(),
                        before.by_reduce_result.len(),
                        after.by_reduce_result.len(),
                    );
                }
            }
        }
    }
}
