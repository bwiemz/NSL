//! WGGO pre-main planning pass — the "WGGO before kernel synthesis"
//! restructure (PR #334's tracked follow-up, wggo_overrides.rs §"v1 caveat").
//!
//! # Why this exists
//!
//! Every compile entry point runs `compile_flash_attention_kernels` (and thus
//! `maybe_synthesize_csha_training_ptx`, home of the @pca per-document
//! admission fork) strictly BEFORE `compile_main`, while WGGO planning used
//! to run inside `compile_main` (train-block lowering). Two structural
//! consequences:
//!
//! 1. `Compiler.wggo_overrides` was provably `None` at kernel-synthesis /
//!    admission time — the plan's `packing_mode` could be *validated* after
//!    the fact (PR #334) but never *influence* which kernels get synthesized.
//! 2. FASE recipe selection and the per-param mode table read
//!    `self.wggo_overrides` BEFORE the planner ran in the same
//!    `compile_train_block_inner` invocation — `None` for the first train
//!    block, and (worse) the PREVIOUS block's stale overrides for any later
//!    one.
//!
//! This pass hoists the *solve* only. The WGGO ILP core reads nothing from
//! kernel synthesis (its SMEM budget is a static constant; its input is the
//! Wengert list), so planning earlier is sound. What the in-place path had
//! that a pre-pass lacks is codegen-time state: the enclosing `FuncState`'s
//! variable/type tables used for extractor input registration, and the model
//! variable's resolved type. Both have semantic-level substitutes for the
//! only place train blocks are actually planned today — the TOP LEVEL of the
//! entry module: a linear walk of the preceding `let`/`const` declarations
//! with types from the semantic `TypeMap`.
//!
//! # Preference, not mandate — and honest fallback
//!
//! A pre-plan is an *offer*. `compile_train_block_inner` re-extracts its own
//! Wengert list during codegen (that list is what actually lowers, and
//! `wggo_prune` may rewrite it); it consumes the pre-plan ONLY if the
//! codegen-time extraction has the same graph fingerprint. On any mismatch —
//! or whenever the pre-pass could not extract (non-top-level train block,
//! unresolvable model type, source-AD extraction failure) — the train block
//! plans in place exactly as before this pass existed. Worst case is the
//! status quo, never a silently mismatched plan.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use nsl_ast::expr::ExprKind;
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind};

use crate::compiler::Compiler;

/// One train block's pre-computed plan, keyed by the train block's stmt
/// NodeId (the same key `set_active_fused_ce_config_for_train_block` uses).
pub struct WggoPrePlan {
    pub train_block_stmt_id: nsl_ast::NodeId,
    pub plan: crate::wggo::WggoPlan,
    pub overrides: crate::wggo_overrides::WggoOverrides,
    /// Fingerprint of the Wengert list the plan was solved against. The
    /// consumer must re-fingerprint its own extraction and reject on
    /// mismatch (`wggo-preplan-rejected reason=graph_fingerprint_mismatch`).
    pub graph_fingerprint: u64,
}

/// Structural fingerprint of a Wengert list: op count plus per-op
/// (canonically-renumbered result/input VarIds, op-kind discriminant) plus
/// the Param/Input leaf names the plan's layer matching keys off.
///
/// VarIds are renumbered densely in first-appearance order rather than
/// hashed raw: `register_input_with_rank` eagerly allocates a VarId per
/// PRE-REGISTERED symbol, the in-place path registers every `FuncState`
/// variable in HashMap iteration order while the pre-pass registers the
/// top-level declarations — so raw VarIds differ between two extractions of
/// the very same graph. Canonical renumbering makes the fingerprint depend
/// only on graph shape and leaf identity, which is exactly what the plan
/// depends on. Spans/saved-for-backward bookkeeping are ignored for the
/// same reason.
pub fn fingerprint_wengert(list: &crate::wengert::WengertList) -> u64 {
    if std::env::var("NSL_WGGO_PREPASS_DEBUG").is_ok() {
        eprintln!("[wggo-prepass-debug] ops={}", list.ops.len());
        for (i, op) in list.ops.iter().enumerate() {
            eprintln!("[wggo-prepass-debug] {i}: {:?} <- {:?} ({})", op.result, op.inputs, primal_op_tag(&op.op));
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();

    // Pass 1: leaf ops (Input/Param). Their EMISSION ORDER follows symbol
    // registration order — which on the in-place path is `state.variables`
    // HashMap iteration, i.e. nondeterministic — so leaves are hashed as a
    // name-sorted set and their canonical VarIds derive from that sorted
    // position, not from op order.
    let mut leaves: Vec<(&str, crate::wengert::VarId)> = Vec::new();
    for op in &list.ops {
        match &op.op {
            crate::wengert::PrimalOp::Input(name) | crate::wengert::PrimalOp::Param(name) => {
                leaves.push((name.as_str(), op.result));
            }
            _ => {}
        }
    }
    leaves.sort_by(|a, b| a.0.cmp(b.0));
    let mut canon: HashMap<crate::wengert::VarId, u64> = HashMap::new();
    let mut next: u64 = 0;
    for (name, var) in &leaves {
        name.hash(&mut h);
        canon.entry(*var).or_insert_with(|| {
            let n = next;
            next += 1;
            n
        });
    }

    // Pass 2: non-leaf ops in list order, with canonically-renumbered
    // result/input VarIds (leaves resolve to their name-derived ids).
    list.ops.len().hash(&mut h);
    for op in &list.ops {
        if matches!(
            &op.op,
            crate::wengert::PrimalOp::Input(_) | crate::wengert::PrimalOp::Param(_)
        ) {
            continue;
        }
        for &input in &op.inputs {
            let id = *canon.entry(input).or_insert_with(|| {
                let n = next;
                next += 1;
                n
            });
            id.hash(&mut h);
        }
        let rid = *canon.entry(op.result).or_insert_with(|| {
            let n = next;
            next += 1;
            n
        });
        rid.hash(&mut h);
        std::mem::discriminant(&op.op).hash(&mut h);
    }
    h.finish()
}

/// Module-level packing-preference aggregation for the admission gate.
/// Kernel synthesis is per train block (module-scoped), so the per-layer
/// packing decisions collapse to one preference: `true` iff at least one
/// layer requested packing AND every layer that did chose segment_id
/// (mode 1). Mixed or zero preferences are ambiguous at module scope and
/// must not flip admission.
pub fn plan_prefers_segment_id(overrides: &crate::wggo_overrides::WggoOverrides) -> bool {
    let mut any_nonzero = false;
    let all_segment_id = overrides.per_layer.iter().all(|l| {
        if l.packing_mode == 0 {
            true
        } else {
            any_nonzero = true;
            l.packing_mode == 1
        }
    });
    any_nonzero && all_segment_id
}

fn primal_op_tag(op: &crate::wengert::PrimalOp) -> String {
    let dbg = format!("{op:?}");
    dbg.split(['(', '{', ' ']).next().unwrap_or("?").to_string()
        + match op {
            crate::wengert::PrimalOp::Input(n) | crate::wengert::PrimalOp::Param(n) => {
                return format!("{}:{n}", dbg.split(['(', '{']).next().unwrap_or("?"));
            }
            _ => "",
        }
}

/// Run the pre-pass over the entry module's top-level statements. Called
/// from the compile entry points AFTER the WRGA prescan rewrites
/// `model_method_bodies` (source AD must see adapter-rewritten methods —
/// same input the in-place extraction gets) and BEFORE
/// `compile_flash_attention_kernels`.
///
/// No-op (returns empty) unless source AD and a WGGO mode are both enabled —
/// mirroring the exact gating of the in-place planner so behavior without
/// this pass's preconditions is byte-identical to before.
pub fn run(compiler: &mut Compiler, stmts: &[Stmt]) -> Vec<WggoPrePlan> {
    if !compiler.features.source_ad_enabled {
        return Vec::new();
    }
    let wggo_on = matches!(
        compiler.compile_options.wggo.mode.as_deref(),
        Some(m) if m != "off" && m != "disable" && m != "disabled"
    );
    if !wggo_on {
        return Vec::new();
    }

    // Linear walk: accumulate top-level `let`/`const` types as the
    // semantic-level substitute for the in-place path's
    // `FuncState.variables` registration, planning each train block with
    // exactly the declarations that precede it (main's sequential scope).
    let mut prefix_types: HashMap<nsl_ast::Symbol, nsl_semantic::types::Type> = HashMap::new();
    let mut preplans = Vec::new();
    walk_stmts(compiler, stmts, &mut prefix_types, &mut preplans);
    preplans
}

fn walk_stmts(
    compiler: &mut Compiler,
    stmts: &[Stmt],
    prefix_types: &mut HashMap<nsl_ast::Symbol, nsl_semantic::types::Type>,
    out: &mut Vec<WggoPrePlan>,
) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                if let (PatternKind::Ident(sym), Some(init)) = (&pattern.kind, value) {
                    if let Some(ty) = compiler.type_map.get(&init.id) {
                        prefix_types.insert(*sym, ty.clone());
                    }
                }
            }
            StmtKind::TrainBlock(train) => {
                if let Some(preplan) = plan_train_block(compiler, train, stmt.id, prefix_types) {
                    out.push(preplan);
                }
            }
            StmtKind::Decorated { stmt: inner, .. } => {
                // Train blocks often sit inside test/bench decorators — same
                // recursion `stmts_contain_train_block` uses.
                walk_stmts(compiler, std::slice::from_ref(inner), prefix_types, out);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, VarId, WengertList, WengertOp};

    fn op(id: u64, result: u64, op: PrimalOp, inputs: Vec<u64>) -> WengertOp {
        WengertOp {
            id: id as crate::wengert::OpId,
            result: result as VarId,
            op,
            inputs: inputs.into_iter().map(|v| v as VarId).collect(),
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn list(ops: Vec<WengertOp>, output: u64) -> WengertList {
        WengertList {
            ops,
            output: output as VarId,
            var_names: std::collections::HashMap::new(),
            var_types: std::collections::HashMap::new(),
        }
    }

    /// The load-bearing property: leaf emission order follows symbol
    /// registration order (HashMap iteration on the in-place path), so two
    /// extractions of the same graph legitimately differ in leaf order AND
    /// in every downstream VarId. The fingerprint must be identical anyway.
    #[test]
    fn fingerprint_is_insensitive_to_leaf_order_and_var_ids() {
        let a = list(
            vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.w".into()), vec![]),
                op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            2,
        );
        // Same graph: leaves swapped, all VarIds shifted.
        let b = list(
            vec![
                op(0, 5, PrimalOp::Param("m.w".into()), vec![]),
                op(1, 7, PrimalOp::Input("x".into()), vec![]),
                op(2, 9, PrimalOp::Matmul, vec![7, 5]),
            ],
            9,
        );
        assert_eq!(fingerprint_wengert(&a), fingerprint_wengert(&b));
    }

    #[test]
    fn fingerprint_distinguishes_op_kind_and_leaf_names() {
        let base = list(
            vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.w".into()), vec![]),
                op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            2,
        );
        let different_kind = list(
            vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.w".into()), vec![]),
                op(2, 2, PrimalOp::Add, vec![0, 1]),
            ],
            2,
        );
        let different_leaf = list(
            vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.other".into()), vec![]),
                op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            2,
        );
        assert_ne!(fingerprint_wengert(&base), fingerprint_wengert(&different_kind));
        assert_ne!(fingerprint_wengert(&base), fingerprint_wengert(&different_leaf));
    }

    fn overrides_with_modes(modes: &[u8]) -> crate::wggo_overrides::WggoOverrides {
        crate::wggo_overrides::WggoOverrides {
            per_layer: modes
                .iter()
                .enumerate()
                .map(|(i, &m)| crate::wggo_overrides::PerLayerOverride {
                    layer_index: i as u32,
                    layer_name: format!("blocks.{i}"),
                    active_heads: 0,
                    requested_csha_level: None,
                    adapter_rank: 0,
                    adapter_placement: Default::default(),
                    fase_fused: false,
                    packing_mode: m,
                    shard_factor: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn segment_id_preference_requires_unambiguous_mode_1() {
        // All packing layers chose segment_id → prefer.
        assert!(plan_prefers_segment_id(&overrides_with_modes(&[0, 1, 1])));
        // Mixed 1 and 3 → ambiguous → no flip.
        assert!(!plan_prefers_segment_id(&overrides_with_modes(&[1, 3])));
        // No packing anywhere → no flip.
        assert!(!plan_prefers_segment_id(&overrides_with_modes(&[0, 0])));
        // multi_seq only → no flip (per-doc CTA stays).
        assert!(!plan_prefers_segment_id(&overrides_with_modes(&[3])));
    }
}

fn plan_train_block(
    compiler: &mut Compiler,
    train: &nsl_ast::block::TrainBlock,
    train_block_stmt_id: nsl_ast::NodeId,
    prefix_types: &HashMap<nsl_ast::Symbol, nsl_semantic::types::Type>,
) -> Option<WggoPrePlan> {
    // `train(model = <ident>)` — pure AST, same extraction as the in-place
    // path's config scan.
    let model_sym = train.config.iter().find_map(|arg| {
        let name_sym = arg.name?;
        if compiler.resolve_sym(name_sym) != "model" {
            return None;
        }
        match &arg.value.kind {
            ExprKind::Ident(sym) => Some(*sym),
            _ => None,
        }
    })?;

    // Model type: for-loop model vars first (same priority as in-place),
    // then the semantic type of the model variable's initializer.
    let model_type_name = compiler
        .models
        .model_var_types
        .get(&model_sym)
        .cloned()
        .or_else(|| match prefix_types.get(&model_sym) {
            Some(nsl_semantic::types::Type::Model { name, .. })
            | Some(nsl_semantic::types::Type::Struct { name, .. }) => {
                Some(compiler.resolve_sym(*name).to_string())
            }
            _ => None,
        })?;

    let (step_body, step_param_sym) = train.sections.iter().find_map(|s| match s {
        nsl_ast::block::TrainSection::Step { body, param } => Some((body, *param)),
        _ => None,
    })?;

    // Install THIS block's fused-CE config for the extraction, exactly as
    // `compile_train_block` does, and restore unconditionally.
    let saved_fused_ce =
        compiler.set_active_fused_ce_config_for_train_block(train_block_stmt_id);
    let result = (|| {
        let fused_ce_cfg = compiler.active_fused_ce_config.clone();
        let mut extractor = crate::source_ad::WengertExtractor::new(compiler.interner)
            .with_checkpoint_policies(compiler.compile_options.checkpoint_policies.clone())
            .with_fused_ce_config(fused_ce_cfg);
        extractor.set_model_method_bodies(compiler.models.model_method_bodies.clone());
        extractor.set_model_field_types(compiler.models.model_field_types.clone());
        extractor.set_model_field_ranks(compiler.models.model_field_ranks.clone());
        extractor.set_synth_call_names(compiler.synth_call_names.clone());
        extractor.set_synth_member_names(compiler.synth_member_names.clone());
        extractor.register_model_instance(model_sym, &model_type_name);
        for (&sym, ty) in prefix_types {
            extractor.register_input_with_rank(sym, crate::stmt::resolvable_tensor_rank(ty));
        }
        // The in-place path registers the step parameter as a FuncState
        // variable before extraction, which surfaces as an `Input(batch)`
        // leaf in the extracted graph — mirror it or every fingerprint
        // comparison fails by exactly that one op.
        extractor.register_input_with_rank(step_param_sym, None);

        if !extractor.extract_stmts(&step_body.stmts) {
            // Source-AD extraction failed here; the in-place path may still
            // succeed (it registers codegen-time state we don't model) or
            // fall back to tape AD — either way, planning stays in place.
            return None;
        }

        let mut analysis_config = crate::wggo_weight_analysis::AnalysisConfig::default();
        if let Some(f) = compiler.compile_options.wggo.prune_fraction {
            analysis_config.default_prune_fraction = f.clamp(0.0, 0.9);
        }
        let plan = crate::wggo::run_on_wengert_with_weights(
            extractor.wengert_list(),
            &compiler.compile_options.target,
            compiler.compile_options.wggo.mode.as_deref().unwrap_or("off"),
            compiler.compile_options.world_size,
            compiler.compile_options.wggo.weights.as_deref(),
            analysis_config,
            Some(&compiler.compile_options),
        )?;
        let overrides = crate::wggo_overrides::WggoOverrides::from_applied(&plan.applied);
        let graph_fingerprint = fingerprint_wengert(extractor.wengert_list());
        Some(WggoPrePlan {
            train_block_stmt_id,
            plan,
            overrides,
            graph_fingerprint,
        })
    })();
    compiler.restore_active_fused_ce_config(saved_fused_ce);
    result
}
