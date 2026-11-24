//! Section 6b.6 of the train block's source-AD arm: which fused LM heads
//! compute their gradient in the forward ([`crate::lce_grad_in_forward`]),
//! the adjoint rewrite that goes with it, and the checks the driver runs on
//! the plan around the lowerings that carry it out.
//!
//! Planned on the adjoint after the tape optimizations and BEFORE CCR's
//! splice, so CCR, the last-use frees, the storage plan and the arena all see
//! extracts that no longer read the head's input. A pure tape rewrite: the
//! plan is installed on the compiler (`Compiler::lce_gif`) only for the
//! step's forward and adjoint lowerings.

use std::collections::{HashMap, HashSet};

use crate::compiler::Compiler;
use crate::error::CodegenError;
use crate::lce_grad_in_forward as gif;
use crate::stmt::ParamHookEntry;
use crate::wengert::{VarId, WengertList};
use crate::wengert_lower::LceGifLowering;

/// Every binding of `compile_train_block_inner` the planning reads; names
/// are the driver's.
pub(crate) struct LceGifInputs<'a> {
    /// The adjoint tape; the planned extracts are rewritten in place.
    pub(crate) adjoint: &'a mut WengertList,
    pub(crate) effective_primal: &'a WengertList,
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    pub(crate) generator: &'a crate::source_ad::AdjointGenerator,
    pub(crate) ccr_plan: &'a Option<crate::ccr::CcrPlan>,
    /// `param_paths[i]` is `accum_list[i]`'s parameter.
    pub(crate) param_paths: &'a [String],
    /// The initial VarMap (the parameters' values before the forward).
    pub(crate) primal_vars: &'a crate::wengert_lower::VarMap,
    /// The FASE accumulation buffer list.
    pub(crate) accum_list: Option<cranelift_codegen::ir::Value>,
    /// `1 / grad_accumulation`.
    pub(crate) accum_scale: f64,
    pub(crate) fase_hook_active: bool,
    pub(crate) csla_active: bool,
}

impl Compiler<'_> {
    /// Plan the heads (see the module header). `None` when nothing is
    /// planned: every head keeps its gradient in its backward, exactly as
    /// with `NSL_LCE_GRAD_IN_FORWARD=0`.
    pub(crate) fn plan_lce_grad_in_forward(
        &self,
        inputs: LceGifInputs<'_>,
    ) -> Option<LceGifLowering> {
        let LceGifInputs {
            adjoint,
            effective_primal,
            extractor,
            generator,
            ccr_plan,
            param_paths,
            primal_vars,
            accum_list,
            accum_scale,
            fase_hook_active,
            csla_active,
        } = inputs;
        let heads = effective_primal
            .ops
            .iter()
            .filter(|o| matches!(o.op, crate::wengert::PrimalOp::FusedLinearCe { .. }))
            .count();
        if heads == 0 || !crate::wengert_lower::lce_grad_in_forward_enabled() {
            return None;
        }
        let opts = &self.compile_options;
        // Where the forward can add into the FASE accumulation buffers: the
        // hook lowering with device-resident buffers, every gradient left
        // unobserved (grad-integrity notes each one as the hook receives
        // it), one lowering of the whole tape per micro-batch (CSLA replays
        // the adjoint in windows), no graph capture (the bf16 GEMMs refuse
        // it, and a replayed forward would add twice).
        let refusal = if !fase_hook_active {
            Some(
                "gradients are not accumulated by the FASE hook (grad_accumulation is 1 or FASE is off)",
            )
        } else if csla_active {
            Some("--layerwise-accum replays the backward in windows")
        } else if accum_list.is_none() {
            Some("there are no accumulation buffers")
        } else if opts.train.optim_state_offload {
            Some("--optim-state-offload keeps the accumulation buffers on the host")
        } else if opts.diagnostics.grad_integrity {
            Some("--grad-integrity observes every gradient")
        } else if opts.train.cuda_graphs {
            Some("--cuda-graphs")
        } else if std::env::var("NSL_FUSED_LCE_GEMM").ok().as_deref() == Some("0") {
            Some("NSL_FUSED_LCE_GEMM=0 takes the heads off the GEMM route")
        } else {
            None
        };
        if let Some(why) = refusal {
            nsl_log::nsl_log!(
                INFO,
                "lce-grad-in-forward",
                "[lce-grad-in-forward] 0 of {heads} fused LM head(s) compute their gradient in the forward: {why}"
            );
            return None;
        }
        // The parameter adjoints the hook will consume, with their buffers
        // (the hook's own map is built after the forward; the driver checks
        // that it agrees, see `check_lce_gif_hook_map`).
        let idx_by_path: HashMap<&str, i64> = param_paths
            .iter()
            .enumerate()
            .map(|(i, p)| (p.as_str(), i as i64))
            .collect();
        let mut idx_of: HashMap<VarId, i64> = HashMap::new();
        for (name, vid) in extractor.named_param_var_ids() {
            if !self.is_trainable_param_name(name) || !primal_vars.contains_key(vid) {
                continue;
            }
            let (Some(&idx), Some(adj)) =
                (idx_by_path.get(name.as_str()), generator.adjoint_of(*vid))
            else {
                continue;
            };
            idx_of.insert(adj, idx);
        }
        let param_adj_set: HashSet<VarId> = idx_of.keys().copied().collect();
        // The tied-embedding plan the adjoint lowering will make, under the
        // same conditions (`compile_wengert_ops_range`): its device buffers
        // exist (no offload, checked above), its kill switch is off.
        let (dtype_tag, _) = crate::wengert_lower::fused_ce_dtype_for_compiler(self);
        let tied = if crate::wengert_lower::tied_embed_inplace_enabled() {
            crate::tied_embed_fusion::plan(adjoint, &param_adj_set, dtype_tag == 0)
        } else {
            crate::tied_embed_fusion::TiedEmbedPlan::default()
        };
        // Everything CCR recomputes, compresses or frees on its own: the
        // block segments' ops and the compressed saves.
        let mut ccr_handled: HashSet<VarId> = HashSet::new();
        if let Some(p) = ccr_plan {
            for seg in &p.segments {
                ccr_handled.extend(
                    effective_primal.ops[seg.start..seg.end]
                        .iter()
                        .map(|o| o.result),
                );
            }
            ccr_handled.extend(p.recompute.iter().copied());
            ccr_handled.extend(p.compress.iter().copied());
        }
        let seed = generator.adjoint_of(effective_primal.output)?;
        let (plan, declined) = if dtype_tag == 0 {
            gif::plan(&gif::PlanInputs {
                primal: effective_primal,
                adjoint,
                seed,
                param_adj_set: &param_adj_set,
                tied: &tied,
                ccr_handled: &ccr_handled,
            })
        } else {
            (
                gif::LceGifPlan::default(),
                vec!["a 16-bit @fused_lm_ce dtype= takes the v1 kernels".to_string()],
            )
        };
        let why = if declined.is_empty() {
            String::new()
        } else {
            format!("; kept in the backward: {}", declined.join("; "))
        };
        nsl_log::nsl_log!(
            INFO,
            "lce-grad-in-forward",
            "[lce-grad-in-forward] {} of {heads} fused LM head(s) compute their gradient in the forward (the backward \
             hands on the saved x gradient; no logits recompute){why}",
            plan.by_key.len()
        );
        if plan.is_empty() {
            return None;
        }
        gif::rewrite_adjoint(adjoint, &plan);
        Some(LceGifLowering {
            plan,
            accum_list: accum_list.expect("checked above"),
            idx_of,
            scale: accum_scale,
            forwards: HashSet::new(),
            extracts: HashMap::new(),
            freed_x: 0,
        })
    }
}

/// After CCR's splice, the last-use frees and every later adjoint rewrite: a
/// head's input is freed by the forward only while still no backward op
/// reads it (a marker that frees it there would free it twice).
pub(crate) fn recheck_lce_gif_frees(g: &mut LceGifLowering, adjoint: &WengertList) {
    for head in g.plan.by_key.values_mut() {
        if head.free_x && adjoint.ops.iter().any(|o| o.inputs.contains(&head.x)) {
            head.free_x = false;
        }
    }
}

/// The hook's own parameter map, built after the forward, must give every
/// planned head's parameter the buffer the forward already wrote.
pub(crate) fn check_lce_gif_hook_map(
    g: &LceGifLowering,
    hook: &HashMap<VarId, ParamHookEntry>,
) -> Result<(), CodegenError> {
    for head in g.plan.by_key.values() {
        let planned = g.idx_of.get(&head.param_adjoint).copied();
        let actual = hook.get(&head.param_adjoint).map(|e| e.accum_idx);
        if planned.is_none() || planned != actual {
            return Err(CodegenError::new(format!(
                "internal: LM head {} added its weight gradient into accumulation buffer {planned:?} in \
                 the forward, but the FASE hook gives its parameter buffer {actual:?}",
                head.key
            )));
        }
    }
    Ok(())
}

/// After the adjoint lowering: every planned head ran its forward once, and
/// its backward lowered every extract it was planned with (the first of them
/// checks the upstream gradient) and took its saved x gradient.
pub(crate) fn finish_lce_gif(g: &LceGifLowering, unclaimed_dx: usize) -> Result<(), CodegenError> {
    let missing: Vec<VarId> = g
        .plan
        .by_key
        .keys()
        .filter(|k| !g.forwards.contains(k))
        .copied()
        .collect();
    let unchecked: Vec<VarId> = g
        .plan
        .by_key
        .values()
        .filter(|h| g.extracts.get(&h.key).copied().unwrap_or(0) != 1 + u8::from(h.dx.is_some()))
        .map(|h| h.key)
        .collect();
    if !missing.is_empty() || !unchecked.is_empty() || unclaimed_dx != 0 {
        return Err(CodegenError::new(format!(
            "internal: planned LM head(s) {missing:?} were not lowered in the forward, {unchecked:?} did not \
             lower each of their backward extracts once, or {unclaimed_dx} saved x gradient(s) reached no \
             backward extract: a weight gradient added in a forward would go unchecked"
        )));
    }
    if g.freed_x > 0 {
        nsl_log::nsl_log!(
            INFO,
            "lce-grad-in-forward",
            "[lce-grad-in-forward] {} head input(s) freed by the forward (the backward reads the saved x gradient)",
            g.freed_x
        );
    }
    Ok(())
}
