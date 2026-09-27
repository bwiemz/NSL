//! `PASS_ORDER`: the train-block driver's planning stretch as one table
//! (roadmap A1, `TrainPlan` step 4a of
//! `docs/superpowers/specs/2026-09-08-a1-train-plan-ir-design.md`).
//!
//! Until now, the order in which `compile_train_block` plans a train block
//! existed only as the reading order of `driver.rs`: the pre-plan CPDT
//! offer, the FASE contract, the admissions, the parameter facts,
//! extraction, the tape passes, adjoint generation and the adjoint passes.
//! This table writes that order down as a value. The composition work the
//! spec lists (CLASP, CADENCE, CADENZA) needs to reason about it, and
//! `tests/train_pass_order.rs` holds the table to the tree in four ways:
//!
//! - **The driver.** Every step's `entry` call appears in `driver.rs` in
//!   table order, and each is defined in the step's `module`.
//! - **The scheduled sites.** The `(module, pass)` pairs of every
//!   `PassScheduler::schedule` call in the train-block driver's files are
//!   exactly the pairs the table names. Every registry pass that declares
//!   the `TrainBlock` phase has a step.
//! - **The bus.** The table's first-invocation order violates none of the
//!   pass bus's `InvocationOrdered` edges ([`bus_order_violations`]). That
//!   is the same check the pass manager enforces on every compile's ledger,
//!   applied to the declared order instead of an observed one.
//! - **The stages.** The stages never go backwards, and no pass that reads
//!   or rewrites the tape is placed before extraction.
//!
//! This step changes nothing the compiler does. The driver still makes the
//! calls itself: its planning stretch interleaves emission (the CSHA prune
//! and the WRGA adapter sites take the builder), so a `TrainPass` trait
//! object per row would have to carry most of the driver's bindings. Step
//! 4b wraps the rows whose inputs are already plan facts.

use crate::pass_bus::Channel;

/// When the driver runs a step, relative to the tapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrainStage {
    /// Before `WengertExtractor` builds the primal tape.
    PreTape,
    /// After extraction, before adjoint generation.
    Tape,
    /// After `AdjointGenerator` builds the adjoint.
    Adjoint,
}

/// One step of the train-block driver's planning stretch.
#[derive(Debug, Clone, Copy)]
pub struct TrainPassDecl {
    /// The step's name, as the spec's `PASS_ORDER` table calls it.
    pub step: &'static str,
    /// The registry passes (`pass_registry::PASSES`) this step runs under
    /// `PassScheduler::schedule`, in the order it schedules them. Empty for
    /// a step that is not a registered pass.
    pub passes: &'static [&'static str],
    /// When the driver runs it.
    pub stage: TrainStage,
    /// The file the step runs in (its `schedule` sites, if any), relative to
    /// `crates/nsl-codegen/src`.
    pub module: &'static str,
    /// The call the driver makes: a `Compiler` method, a free function or a
    /// `Type::new` constructor.
    pub entry: &'static str,
    /// The file that defines `entry`: `module`, except for a step whose
    /// `schedule` site in `module` calls a bridge defined elsewhere.
    pub defined_in: &'static str,
}

const fn step(
    step: &'static str,
    passes: &'static [&'static str],
    stage: TrainStage,
    module: &'static str,
    entry: &'static str,
) -> TrainPassDecl {
    TrainPassDecl { step, passes, stage, module, entry, defined_in: module }
}

/// A step whose `schedule` site in `module` calls `entry`, defined in
/// `defined_in`.
const fn bridged(step_: TrainPassDecl, defined_in: &'static str) -> TrainPassDecl {
    TrainPassDecl { defined_in, ..step_ }
}

use TrainStage::{Adjoint, PreTape, Tape};

/// The train-block driver's planning steps, in the order it runs them.
pub const PASS_ORDER: &[TrainPassDecl] = &[
    // `compile_train_block`, before the inner driver: the CPDT pre-plan
    // offer (the speculative weights-only precision plan the moment consult
    // reads). CPDT's registry stage is OnWengert, its pipeline position; it
    // reads no tape, so it can run this early.
    bridged(
        step("CpdtPreplanOffer", &["CPDT"], PreTape, "stmt_train/driver.rs", "invoke_cpdt_if_enabled"),
        "stmt_pass_bridges.rs",
    ),
    step("ConfigPass", &["FASE"], PreTape, "stmt_train/contract.rs", "resolve_train_contract"),
    step("WgradAdmission", &[], PreTape, "stmt_admission.rs", "wgrad_fusion_admission"),
    step("CslaZeroAdmission", &[], PreTape, "stmt_admission.rs", "csla_and_zero_admission"),
    step("ParamPlanPass", &[], PreTape, "stmt_train/model_params.rs", "emit_model_params"),
    step("NoDecayAdmission", &[], PreTape, "stmt_admission.rs", "no_decay_composition_admission"),
    step("ExtractPass", &[], Tape, "source_ad.rs", "WengertExtractor::new"),
    // The primal VarMap build hosts the CPKD plan: the driver scans the
    // extracted tape for the fused KL-CE op and hands the result to
    // `cpkd::build_plan`, which itself never sees the tape.
    step("CpkdPass", &["CPKD"], Tape, "stmt_train/primal_vars.rs", "emit_primal_vars"),
    step("WggoPlanPass", &["WGGO"], Tape, "stmt_train/plan_wggo.rs", "plan_wggo"),
    step("CshaPass+WggoPrunePass", &["CSHA"], Tape, "stmt_train/plan_csha_prune.rs", "run_csha_and_wggo_prune"),
    step("WrgaPass+CpdtPass", &["WRGA", "CPDT"], Tape, "stmt_train/plan_wrga_cpdt.rs", "run_wrga_and_plan_cpdt"),
    step("CcrPlanPass", &["CCR"], Tape, "stmt_train/plan_ccr.rs", "fork_wrga_and_plan_ccr"),
    step("AdjointGenPass", &[], Adjoint, "source_ad.rs", "AdjointGenerator::new"),
    step("AdjointTapeOptPass", &[], Adjoint, "stmt_train/adjoint_tape_opt.rs", "optimize_adjoint_tape"),
    step("CcrAdjointFreesPass", &[], Adjoint, "stmt_train/ccr_adjoint_frees.rs", "insert_ccr_adjoint_frees"),
    step(
        "ArenaProjectionPass",
        &["MemoryPlanner"],
        Adjoint,
        "stmt_train/transient_arena_projection.rs",
        "emit_transient_arena_projection",
    ),
    step("CslaPrecomputePass", &[], Adjoint, "stmt_train/csla_precompute.rs", "precompute_csla_schedule"),
];

/// The registry passes of `order`, in first-invocation order: the sequence
/// a compile that ran every step would leave in the pass manager's ledger.
pub fn first_invocations(order: &[TrainPassDecl]) -> Vec<&'static str> {
    let mut seen: Vec<&'static str> = Vec::new();
    for pass in order.iter().flat_map(|s| s.passes.iter().copied()) {
        if !seen.contains(&pass) {
            seen.push(pass);
        }
    }
    seen
}

/// The pass bus's `InvocationOrdered` edges that `order` inverts, as
/// `(producer, consumer, channel)`: the pass manager's per-compile check,
/// applied to a declared order.
pub fn bus_order_violations_in(order: &[TrainPassDecl]) -> Vec<(&'static str, &'static str, Channel)> {
    crate::pass_bus::dependency_order_violations_in(&first_invocations(order))
}

/// [`bus_order_violations_in`] of [`PASS_ORDER`]. Empty, or the table is
/// wrong.
pub fn bus_order_violations() -> Vec<(&'static str, &'static str, Channel)> {
    bus_order_violations_in(PASS_ORDER)
}
