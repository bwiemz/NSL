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
    /// True iff this is the FIRST train block in document order — the one
    /// whose plan may govern module-scoped kernel admission. When the first
    /// block fails to plan, NO pre-plan carries this flag: the admission
    /// gate must not fall through to a later block's preferences (kernel
    /// synthesis serves the whole module; attributing it to whichever block
    /// happened to plan first would flip admission on the wrong evidence).
    pub is_first_train_block: bool,
}

/// The exact off-mode interpretation `run` (and the in-place planner) use.
/// Callers printing pre-pass diagnostics must gate on this too — `--wggo
/// off` arrives as `Some("off")`, so `mode.is_some()` is NOT "wggo enabled".
pub fn wggo_mode_enabled(options: &crate::CompileOptions) -> bool {
    matches!(
        options.wggo.mode.as_deref(),
        Some(m) if m != "off" && m != "disable" && m != "disabled"
    )
}

/// True iff the WGGO pre-pass must be DEFERRED because calibrated importance
/// scoring was requested but its calibration sidecar is not yet available.
///
/// The pre-pass scorer (`build_scorer`) needs the calibration sidecar to
/// honour `importance != Magnitude`: without it `importance=Grad` errors (the
/// plan is silently dropped) and `importance=Auto` degrades to magnitude
/// scoring. Either way the pre-pass would score with a DIFFERENT importance
/// than the in-place planner, which runs later (in `compile_main`) once the
/// sidecar exists — so the pre-plan's graph fingerprint would never match and
/// it would be rejected. While the sidecar is absent we therefore refuse the
/// pre-pass and let planning stay in-place.
///
/// Once the calibration harness is sequenced ahead of
/// `compile_flash_attention_kernels` (the `compile_and_calibrate` wrapper does
/// this) the sidecar is populated before the pre-pass runs, this returns
/// `false`, and the pre-pass plans under the same calibrated scorer the
/// in-place planner uses — so the two agree and the pre-plan is consumed.
/// Entry points that never fire the harness leave the sidecar `None`, so this
/// keeps returning `true` there and the pre-pass stays deferred, exactly as
/// before the WGGO-before-kernels restructure.
pub fn wggo_prepass_deferred_pending_sidecar(options: &crate::CompileOptions) -> bool {
    options.calibration_data.is_some()
        && !matches!(options.wggo.importance, crate::WggoImportance::Magnitude)
        && options.calibration_sidecar.is_none()
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

    // Registration noise filter: `register_input_with_rank` EAGERLY pushes
    // one `Input` op per registered symbol — referenced or not — and the
    // in-place path registers every `FuncState` variable (for-loop counters,
    // tuple binds, anything main compiled so far) while the pre-pass mirrors
    // only the top-level declarations. An Input leaf nobody reads is pure
    // registration residue, so it must not participate in the fingerprint:
    // otherwise a `for i in range(3): ...` before the train block rejects
    // every pre-plan for a semantically identical graph. A missed symbol the
    // step body actually READS fails the pre-pass extraction instead — the
    // honest no-preplan fallback.
    let referenced: std::collections::HashSet<crate::wengert::VarId> = list
        .ops
        .iter()
        .flat_map(|op| op.inputs.iter().copied())
        .collect();
    // NOTE deliberately NO `list.output` exemption here: the pre-pass
    // extraction never sets `output` (it stays the default VarId 0 — the
    // extractor only assigns it on `Return` statements, and the in-place
    // flow fills it post-extraction), so an output exemption would exempt
    // whichever symbol RANDOMLY got VarId 0 from registration order — the
    // exact nondeterministic mismatch this filter exists to prevent. A
    // train step's real output is a computed loss (never a bare leaf), so
    // the exemption bought nothing.
    let is_dead_leaf = |op: &crate::wengert::WengertOp| {
        matches!(
            &op.op,
            crate::wengert::PrimalOp::Input(_) | crate::wengert::PrimalOp::Param(_)
        ) && !referenced.contains(&op.result)
    };

    // Pass 1: REFERENCED leaf ops (Input/Param). Their EMISSION ORDER follows
    // symbol registration order — which on the in-place path is
    // `state.variables` HashMap iteration, i.e. nondeterministic — so leaves
    // are hashed as a name-sorted set and their canonical VarIds derive from
    // that sorted position, not from op order.
    let mut leaves: Vec<(&str, crate::wengert::VarId)> = Vec::new();
    let mut hashed_ops: u64 = 0;
    for op in &list.ops {
        if is_dead_leaf(op) {
            continue;
        }
        match &op.op {
            crate::wengert::PrimalOp::Input(name) | crate::wengert::PrimalOp::Param(name) => {
                leaves.push((name.as_str(), op.result));
                hashed_ops += 1;
            }
            _ => hashed_ops += 1,
        }
    }
    leaves.sort_by(|a, b| a.0.cmp(b.0));
    if std::env::var("NSL_WGGO_PREPASS_DEBUG").is_ok() {
        eprintln!(
            "[wggo-prepass-debug] stream: hashed_ops={hashed_ops} output={:?} leaves={:?}",
            list.output,
            leaves.iter().map(|(n, _)| *n).collect::<Vec<_>>()
        );
    }
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
    hashed_ops.hash(&mut h);
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
    let fp = h.finish();
    if std::env::var("NSL_WGGO_PREPASS_DEBUG").is_ok() {
        eprintln!("[wggo-prepass-debug] fingerprint={fp:016x}");
    }
    fp
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
    if !wggo_mode_enabled(&compiler.compile_options) {
        return Vec::new();
    }

    // Linear walk: accumulate top-level `let`/`const` types as the
    // semantic-level substitute for the in-place path's
    // `FuncState.variables` registration, planning each train block with
    // exactly the declarations that precede it (main's sequential scope).
    let mut prefix_types: HashMap<nsl_ast::Symbol, nsl_semantic::types::Type> = HashMap::new();
    let mut preplans = Vec::new();
    let mut train_blocks_seen = 0usize;
    walk_stmts(compiler, stmts, &mut prefix_types, &mut train_blocks_seen, &mut preplans);
    preplans
}

fn walk_stmts(
    compiler: &mut Compiler,
    stmts: &[Stmt],
    prefix_types: &mut HashMap<nsl_ast::Symbol, nsl_semantic::types::Type>,
    train_blocks_seen: &mut usize,
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
                let is_first = *train_blocks_seen == 0;
                *train_blocks_seen += 1;
                if let Some(mut preplan) =
                    plan_train_block(compiler, train, stmt.id, prefix_types)
                {
                    preplan.is_first_train_block = is_first;
                    out.push(preplan);
                }
            }
            StmtKind::Decorated { stmt: inner, .. } => {
                // Train blocks often sit inside test/bench decorators — same
                // recursion `stmts_contain_train_block` uses.
                walk_stmts(
                    compiler,
                    std::slice::from_ref(inner),
                    prefix_types,
                    train_blocks_seen,
                    out,
                );
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

    /// Gate for the calibrated-importance WGGO deferral (calibration-harness
    /// hoist). The pre-pass must be deferred only while the sidecar it needs
    /// is still absent; once the harness has produced a sidecar the pre-pass
    /// plans under the same calibrated scorer the in-place planner uses. The
    /// old proxy (`calibration_data.is_some() && importance != Magnitude`) had
    /// no sidecar term, so hoisting the harness alone would have been a silent
    /// no-op — this pins the sidecar term that makes the hoist effective.
    #[test]
    fn prepass_deferral_tracks_sidecar_presence() {
        use crate::calibration::sidecar::{Sidecar, SIDECAR_VERSION};
        use crate::{CompileOptions, WggoImportance, WggoOptions};

        // Sidecar has no Default derive (byte_map_b64 serde helper), so build
        // a minimal empty one — the gate only checks presence, not contents.
        let empty_sidecar = || Sidecar {
            version: SIDECAR_VERSION,
            checkpoint_sha256: String::new(),
            calibration_data_sha256: String::new(),
            hook_set_sha256: String::new(),
            cache_key_digest: String::new(),
            num_samples_used: 0,
            hooks: std::collections::BTreeMap::new(),
            wggo_head_gradients: None,
        };

        let opts = |imp: WggoImportance, calib: bool, sidecar: Option<Sidecar>| CompileOptions {
            calibration_data: calib.then(|| std::path::PathBuf::from("calib.safetensors")),
            calibration_sidecar: sidecar,
            wggo: WggoOptions {
                importance: imp,
                ..Default::default()
            },
            ..Default::default()
        };

        // Calibrated importance requested, sidecar not yet produced -> defer.
        assert!(wggo_prepass_deferred_pending_sidecar(&opts(
            WggoImportance::Grad,
            true,
            None
        )));
        assert!(wggo_prepass_deferred_pending_sidecar(&opts(
            WggoImportance::Auto,
            true,
            None
        )));

        // Sidecar present (harness hoisted ahead of the pre-pass) -> plan.
        assert!(!wggo_prepass_deferred_pending_sidecar(&opts(
            WggoImportance::Grad,
            true,
            Some(empty_sidecar())
        )));
        assert!(!wggo_prepass_deferred_pending_sidecar(&opts(
            WggoImportance::Auto,
            true,
            Some(empty_sidecar())
        )));

        // Magnitude scoring never needs the sidecar -> plan regardless.
        assert!(!wggo_prepass_deferred_pending_sidecar(&opts(
            WggoImportance::Magnitude,
            true,
            None
        )));

        // No calibration data at all -> the gate is inert -> plan.
        assert!(!wggo_prepass_deferred_pending_sidecar(&opts(
            WggoImportance::Grad,
            false,
            None
        )));
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

    /// Registration is EAGER (one Input op per registered symbol, referenced
    /// or not) and the in-place path registers every FuncState variable —
    /// unreferenced leaves are pure registration residue and must not affect
    /// the fingerprint.
    #[test]
    fn fingerprint_ignores_unreferenced_input_leaves() {
        let clean = list(
            vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.w".into()), vec![]),
                op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            2,
        );
        // Same graph plus two dead registrations (a for-loop counter and a
        // duplicate step-param leaf).
        let noisy = list(
            vec![
                op(0, 0, PrimalOp::Input("i".into()), vec![]),
                op(1, 1, PrimalOp::Input("x".into()), vec![]),
                op(2, 2, PrimalOp::Param("m.w".into()), vec![]),
                op(3, 3, PrimalOp::Input("batch".into()), vec![]),
                op(4, 4, PrimalOp::Matmul, vec![1, 2]),
            ],
            4,
        );
        assert_eq!(fingerprint_wengert(&clean), fingerprint_wengert(&noisy));
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
        // comparison fails by exactly that one op. Guarded: when the step
        // param shadows a top-level declaration the two are the SAME
        // Symbol, and the in-place path holds one `state.variables` entry
        // per key — double registration here would emit a duplicate leaf.
        if !prefix_types.contains_key(&step_param_sym) {
            extractor.register_input_with_rank(step_param_sym, None);
        }

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
            // Stamped by walk_stmts, which tracks document order across
            // failed extractions too.
            is_first_train_block: false,
        })
    })();
    compiler.restore_active_fused_ce_config(saved_fused_ce);
    result
}

#[cfg(test)]
mod repro_tests {
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

    /// Exact op lists dumped from two flaky prepass extractions of the same
    /// two-block fixture (runs a and c) — must fingerprint identically.
    #[test]
    fn dumped_isomorphic_lists_fingerprint_equal() {
        let bad = list(
            vec![
                op(0, 0, PrimalOp::Input("m2".into()), vec![]),
                op(1, 1, PrimalOp::Input("x2".into()), vec![]),
                op(2, 2, PrimalOp::Input("y2".into()), vec![]),
                op(3, 3, PrimalOp::Input("x".into()), vec![]),
                op(4, 4, PrimalOp::Input("y".into()), vec![]),
                op(5, 5, PrimalOp::Input("m".into()), vec![]),
                op(6, 6, PrimalOp::Input("batch".into()), vec![]),
                op(7, 7, PrimalOp::Param("m2.w2".into()), vec![]),
                op(8, 8, PrimalOp::Matmul, vec![1, 7]),
                op(9, 9, PrimalOp::MSELoss, vec![8, 2]),
            ],
            9,
        );
        let good = list(
            vec![
                op(0, 0, PrimalOp::Input("y2".into()), vec![]),
                op(1, 1, PrimalOp::Input("x".into()), vec![]),
                op(2, 2, PrimalOp::Input("y".into()), vec![]),
                op(3, 3, PrimalOp::Input("m".into()), vec![]),
                op(4, 4, PrimalOp::Input("m2".into()), vec![]),
                op(5, 5, PrimalOp::Input("x2".into()), vec![]),
                op(6, 6, PrimalOp::Input("batch".into()), vec![]),
                op(7, 7, PrimalOp::Param("m2.w2".into()), vec![]),
                op(8, 8, PrimalOp::Matmul, vec![5, 7]),
                op(9, 9, PrimalOp::MSELoss, vec![8, 0]),
            ],
            9,
        );
        assert_eq!(fingerprint_wengert(&bad), fingerprint_wengert(&good));
    }
}
