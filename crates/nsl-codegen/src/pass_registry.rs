//! Roadmap item 21: the compiler's optimization passes, as data.
//!
//! # Why this exists
//!
//! NSL has eleven named compiler passes spread over ~120 source files, ~40 CLI
//! flags
//! and a dozen decorators. (Not all eleven are *IR* passes: CEP and CFIE run
//! out of band, the memory planner runs over lowered bodies, and PCA is an AST
//! module scan. The `stage` field records which is which.) Every fact about
//! them — which files implement one,
//! which flags drive it, where it sits in the pipeline — was previously
//! recorded only in prose: `docs/wiki/Optimization-Passes.md`, README tables,
//! `STATUS.md`, and a hand-maintained shell array in
//! `scripts/check-doc-agreement.sh`. Prose cannot be compiled, so it drifts,
//! and the drift is invisible until someone reads the page and happens to know
//! better.
//!
//! That is not hypothetical here. The wiki page described CCR as "Common-kernel
//! Combination Rewriting ... no implementation file exists" for roughly three
//! months while `ccr.rs` was 2000+ lines, wired into `stmt.rs`, and driven by
//! `--checkpoint-blocks`. It had the pass's NAME wrong. Nothing forced a
//! revisit because nothing could.
//!
//! [`PASSES`] makes those facts checkable. The gates in
//! `crates/nsl-codegen/tests/pass_registry_drift.rs` assert that every claim
//! here still holds against the tree, in BOTH directions, so a pass cannot be
//! added, renamed, or have its flags changed without this file moving in the
//! same commit.
//!
//! # What this is NOT
//!
//! Not a pass *manager* — it does not schedule, order, or invoke anything.
//! `compile_train_block` still calls the passes directly. This is a
//! description of that call sequence, and the gates prove the description
//! stays true; wiring the registry into the driver so it becomes the single
//! source of execution order is the natural follow-on, and is deliberately not
//! attempted here (it would change codegen behaviour, which a documentation
//! gate has no business doing).
//!
//! # Adding a pass
//!
//! Add a [`PassDescriptor`]. The drift gates will then tell you what else is
//! missing — a wiki section, an unresolvable flag, a stale source path. If a
//! pass genuinely has no wiki section yet, say so with
//! [`WikiCoverage::Undocumented`] and a reason; the gate requires the reason
//! to be non-empty precisely so "not documented" stays a visible decision
//! rather than a silent omission.

// NOTE: there is deliberately no `status` field.
//
// The roadmap item sketched one (Shipped / Experimental). Every pass here
// would have been `Shipped`, no gate could verify it, and the `Experimental`
// variant would have been dead — an enum implying a distinction the registry
// does not actually make. That is the decorative-metadata defect the
// feature-composition registry (item 20) had to be reworked to remove, so it
// is not reintroduced here. Maturity lives in STATUS.md, which is prose
// because the judgement is a human one.

/// WHICH PHASE of the compiler driver invokes this pass.
///
/// Distinct from [`PipelineStage`], and the distinction is the point.
/// `PipelineStage` says where a pass conceptually *acts* on the IR;
/// `CompilePhase` says when the driver actually *calls* it. Roadmap item 2
/// step 1 showed those are not the same axis, and that conflating them
/// misleads: under `--wggo full` the observed order is
/// `WGGO(OnWengert) -> FASE(PreExtraction)`, which reads as an inversion until
/// you know WGGO is invoked from `compile_flash_attention_kernels` — a phase
/// that runs before any train block is compiled at all.
///
/// `compile_entry_returning_plan` is the de-facto pass manager today: a
/// hand-written sequence of ~20 statements, of which exactly two reach a
/// registered pass. Naming those phases is what lets the trace check that a
/// pass ran where the registry says it does and — more importantly — notice
/// when a pass starts running somewhere new.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilePhase {
    /// `compiler.compile_flash_attention_kernels(...)` — runs BEFORE any
    /// train block. WGGO's prepass and PCA's packing detection live here.
    KernelPrepass,
    /// `compiler.compile_main(...)` -> `compile_train_block_inner`.
    TrainBlock,
    /// The whole-program memory plan in `compile_entry`, which runs between
    /// the two compile phases.
    Lowering,
    /// An ANALYSIS command that runs a pass's planner without compiling:
    /// `nsl check --training-report` builds its report by calling
    /// `fase::plan` and `pca_detect::detect` directly. Not the pipeline, but
    /// not a process-exiting subcommand either.
    Analysis,
    /// NOT part of the compile pipeline: driven by a CLI subcommand or the
    /// serve path, which terminate the process themselves. Expected to be
    /// unattributed in the trace.
    OutOfBand,
}

/// Where a pass sits relative to source-AD extraction. This is the axis that
/// actually constrains ordering: a pass that consumes the `WengertList` cannot
/// run before it exists, and a pass that rewrites the adjoint cannot run
/// before adjoint generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Runs before source-AD extraction builds the `WengertList`.
    PreExtraction,
    /// Scans the module AST directly and synthesises kernels from it, without
    /// ever seeing the `WengertList`. Verified for PCA: no `pca_*.rs` module
    /// mentions `WengertList`, and activation is driven from
    /// `compiler/kernel.rs::compile_flash_attention_kernels(stmts)`.
    ModuleScan,
    /// Consumes/rewrites the primal `WengertList` after extraction.
    OnWengert,
    /// Rewrites the adjoint program, or splices into it.
    OnAdjoint,
    /// Runs during Cranelift lowering or later.
    Lowering,
    /// Not part of `compile_train_block` at all — a separate driver entry
    /// point (inference serving, offline weight surgery, design tools).
    OutOfBand,
}

/// Whether the pass has a section in `docs/wiki/Optimization-Passes.md`.
///
/// Modelled as a sum type rather than `Option<&str>` so that "no section"
/// carries a reason. An `Option::None` would be an escape hatch the gate
/// cannot distinguish from an oversight, which is how hand-maintained tables
/// rot: the exemption outlives the reason for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WikiCoverage {
    /// The exact `### ` heading text under "Per-pass descriptions".
    Documented(&'static str),
    /// No section yet, and why not.
    Undocumented(&'static str),
}

/// An `nsl` subcommand that can carry a pass flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subcommand {
    Check,
    Build,
    Run,
}

impl Subcommand {
    /// The `args.rs` struct backing this subcommand.
    pub fn arg_struct(self) -> &'static str {
        match self {
            Self::Check => "CheckArgs",
            Self::Build => "BuildArgs",
            Self::Run => "RunArgs",
        }
    }
}

/// One CLI flag and the subcommands that accept it.
#[derive(Debug, Clone, Copy)]
pub struct CliFlag {
    /// Kebab-case, WITHOUT the leading `--`, exactly as a user types it.
    pub flag: &'static str,
    /// Every subcommand whose arg struct declares it. Pinned exhaustively:
    /// the gate fails if the flag appears on a subcommand not listed here, so
    /// widening the surface is a deliberate edit.
    pub on: &'static [Subcommand],
}

const fn f(flag: &'static str, on: &'static [Subcommand]) -> CliFlag {
    CliFlag { flag, on }
}

/// Shorthands for the three common placements.
use Subcommand::{Build, Check, Run};
const BR: &[Subcommand] = &[Build, Run];
const CBR: &[Subcommand] = &[Check, Build, Run];
const B_ONLY: &[Subcommand] = &[Build];
const C_ONLY: &[Subcommand] = &[Check];
const CB: &[Subcommand] = &[Check, Build];

/// One compiler pass, as data.
#[derive(Debug, Clone, Copy)]
pub struct PassDescriptor {
    /// Short name as it appears in flags, diagnostics and docs (e.g. "CCR").
    pub name: &'static str,
    /// Expanded name. The CCR drift was partly a WRONG expansion, so this is
    /// pinned too.
    pub full_name: &'static str,
    /// Repo-relative implementation files. Every one must exist; the gate
    /// checks this, which alone would have caught the CCR drift.
    pub source_files: &'static [&'static str],
    /// CLI flags, each pinned to the subcommands that actually accept it.
    ///
    /// `args.rs` has THREE argument structs — `CheckArgs`, `BuildArgs`,
    /// `RunArgs` — and a flag is declared once per struct that accepts it.
    /// Which structs those are varies per flag and is NOT uniform. An earlier
    /// version of this registry assumed every pass flag lived on both `build`
    /// and `run`; that is simply false (`--wrga-analyze` is analysis-only and
    /// lives on `check`; `--cep-prune` is offline weight surgery and lives on
    /// `build`). Recording the real placement means the gate can check
    /// EXACTLY — a flag gaining or losing a subcommand fails until the change
    /// is recorded here, which is the drift signal that matters.
    pub cli_flags: &'static [CliFlag],
    pub stage: PipelineStage,
    /// Every driver phase this pass is invoked from.
    ///
    /// A SET, not one value, because passes genuinely have more than one
    /// driver: `MemoryPlanner` runs both inside `compile_main` (via the
    /// transient arena) and from `compile_entry`'s whole-program memory plan;
    /// `FASE` runs in the train block and again under
    /// `nsl check --training-report`. Declaring a single phase forced a false
    /// claim — an earlier revision declared `MemoryPlanner` as `Lowering`
    /// only, which one `--memory-report` build contradicts.
    ///
    /// The gate asserts observed IN declared. Empty means "expected
    /// unattributed" (an `OutOfBand` subcommand that exits on its own).
    pub phases: &'static [CompilePhase],
    /// Source-level decorators that activate the pass, without the `@`.
    pub decorator_triggers: &'static [&'static str],
    pub wiki: WikiCoverage,
}

/// Every named optimization pass in the compiler.
///
/// Verified against the tree by `pass_registry_drift.rs`.
///
/// `source_files` is a REPRESENTATIVE SUBSET of each pass's own modules, not an
/// exhaustive list — CFIE has 16 modules and 5 are named here. It gives the
/// existence gate something concrete to check and tells a reader where to
/// start. Completeness of the module set is enforced from the other side by
/// `every_codegen_pass_module_belongs_to_a_registered_pass`, which matches by
/// name prefix and so covers the unlisted ones.
pub const PASSES: &[PassDescriptor] = &[
    PassDescriptor {
        name: "CCR",
        full_name: "Compiler-Chosen Recomputation",
        source_files: &["crates/nsl-codegen/src/ccr.rs"],
        cli_flags: &[
            f("checkpoint-blocks", BR),
            f("checkpoint-stride", BR),
            f("checkpoint-budget-mib", BR),
            f("checkpoint-selective", BR),
            f("checkpoint-compress", BR),
        ],
        // Straddles two stages: `ccr::plan` / `select_partition_dp` /
        // `apply_budget` run over the PRIMAL tape (stmt.rs ~8034-8248, before
        // `AdjointGenerator::new` at ~8277); only `apply_to_adjoint`,
        // `splice_decompress` and `insert_adjoint_last_use_frees`
        // (~8408-8460) touch the adjoint. Recorded as OnAdjoint because that
        // is the half that REWRITES; the planning half only reads.
        stage: PipelineStage::OnAdjoint,
        phases: &[CompilePhase::TrainBlock],
        decorator_triggers: &["checkpoint"],
        wiki: WikiCoverage::Documented("CCR — Compiler-Chosen Recomputation"),
    },
    PassDescriptor {
        name: "FASE",
        full_name: "Fused Accumulation + Step + Epilogue",
        source_files: &[
            "crates/nsl-codegen/src/fase.rs",
            "crates/nsl-codegen/src/fase_clip.rs",
            "crates/nsl-codegen/src/fase_codegen_table.rs",
            "crates/nsl-codegen/src/fase_memory.rs",
            "crates/nsl-codegen/src/fase_optimizer.rs",
        ],
        cli_flags: &[],
        stage: PipelineStage::PreExtraction,
        phases: &[CompilePhase::TrainBlock, CompilePhase::Analysis],
        decorator_triggers: &["fase"],
        wiki: WikiCoverage::Documented("FASE — Fused Accumulation + Step + Epilogue"),
    },
    PassDescriptor {
        name: "WGGO",
        full_name: "Wengert Graph Global Optimization",
        source_files: &[
            "crates/nsl-codegen/src/wggo.rs",
            "crates/nsl-codegen/src/wggo_apply.rs",
            "crates/nsl-codegen/src/wggo_graph.rs",
            "crates/nsl-codegen/src/wggo_ilp.rs",
            "crates/nsl-codegen/src/wggo_overrides.rs",
            "crates/nsl-codegen/src/wggo_prepass.rs",
            "crates/nsl-codegen/src/wggo_prune.rs",
            "crates/nsl-codegen/src/wggo_schedule.rs",
        ],
        cli_flags: &[
            f("wggo", BR),
            f("wggo-report", BR),
            f("wggo-importance", BR),
            f("wggo-memory-budget", BR),
            f("wggo-moment-precision", BR),
            f("wggo-prune-fraction", BR),
            f("wggo-weights", BR),
        ],
        stage: PipelineStage::OnWengert,
        phases: &[CompilePhase::KernelPrepass],
        decorator_triggers: &["wggo", "wggo_target"],
        wiki: WikiCoverage::Documented("WGGO — Wengert Graph Global Optimization"),
    },
    PassDescriptor {
        name: "CSHA",
        full_name: "Compiler-Synthesized Holistic Attention",
        source_files: &[
            "crates/nsl-codegen/src/csha.rs",
            "crates/nsl-codegen/src/csha_apply.rs",
            "crates/nsl-codegen/src/csha_boundary.rs",
            "crates/nsl-codegen/src/csha_patterns.rs",
            "crates/nsl-codegen/src/csha_pipeline.rs",
            "crates/nsl-codegen/src/csha_specialize.rs",
        ],
        cli_flags: &[f("csha", CBR), f("csha-report", CBR)],
        stage: PipelineStage::OnWengert,
        phases: &[CompilePhase::TrainBlock],
        decorator_triggers: &["csha"],
        wiki: WikiCoverage::Documented("CSHA — Compiler-Synthesized Holistic Attention"),
    },
    PassDescriptor {
        name: "WRGA",
        full_name: "Wengert-Pruned Roofline-Guided Adaptation",
        source_files: &[
            "crates/nsl-codegen/src/wrga.rs",
            "crates/nsl-codegen/src/wrga_fusion.rs",
            "crates/nsl-codegen/src/wrga_memory.rs",
            "crates/nsl-codegen/src/wrga_prune.rs",
            "crates/nsl-codegen/src/wrga_roofline.rs",
        ],
        cli_flags: &[
            // Analysis lives on `nsl check`; the two that affect codegen live
            // on `nsl build`. Neither group is on `run`.
            f("wrga-target", C_ONLY),
            f("wrga-ablate", C_ONLY),
            f("wrga-analyze", C_ONLY),
            f("wrga-compare", C_ONLY),
            f("wrga-report", B_ONLY),
            f("wrga-fold-allocations", B_ONLY),
        ],
        stage: PipelineStage::OnWengert,
        phases: &[CompilePhase::TrainBlock],
        decorator_triggers: &["wrga"],
        wiki: WikiCoverage::Documented("WRGA — Wengert-Pruned Roofline-Guided Adaptation"),
    },
    PassDescriptor {
        name: "CPDT",
        full_name: "Compile-time Parallelism & Distributed Training",
        source_files: &[
            "crates/nsl-codegen/src/cpdt.rs",
            "crates/nsl-codegen/src/cpdt_comm.rs",
            "crates/nsl-codegen/src/cpdt_decorator.rs",
            "crates/nsl-codegen/src/cpdt_joint.rs",
            "crates/nsl-codegen/src/cpdt_optim.rs",
            "crates/nsl-codegen/src/cpdt_zero.rs",
        ],
        cli_flags: &[
            f("cpdt", BR),
            f("cpdt-report", BR),
            f("cpdt-num-gpus", BR),
            f("cpdt-inter-bw", BR),
            f("cpdt-intra-bw", BR),
        ],
        stage: PipelineStage::OnWengert,
        phases: &[CompilePhase::TrainBlock],
        decorator_triggers: &["cpdt"],
        wiki: WikiCoverage::Documented("CPDT — Compile-time Parallelism & Distributed Training"),
    },
    PassDescriptor {
        name: "PCA",
        full_name: "Packed Causal Attention",
        source_files: &[
            "crates/nsl-codegen/src/pca_detect.rs",
            "crates/nsl-codegen/src/pca_segment.rs",
            "crates/nsl-codegen/src/pca_activation.rs",
            "crates/nsl-codegen/src/pca_per_doc.rs",
            "crates/nsl-codegen/src/pca_tier_b.rs",
        ],
        cli_flags: &[],
        // NOT OnWengert. No `pca_*.rs` module references `WengertList`;
        // activation happens in `compiler/kernel.rs` (lines 548/802/851) via
        // `pca_activation::detect_packing_for_stmts(stmts, ..)`, a scan over
        // the module AST that runs before `compile_main`. An earlier version of
        // this entry said OnWengert, which would have told a reader PCA can
        // consume WGGO's AppliedPlan. It cannot.
        stage: PipelineStage::ModuleScan,
        // Neither member is gate-verified: PCA needs a packing-shaped model no
        // CPU fixture provides. KernelPrepass is where `compiler/kernel.rs`
        // calls it; Analysis is `training_report`'s direct call.
        phases: &[CompilePhase::KernelPrepass, CompilePhase::Analysis],
        decorator_triggers: &["pca"],
        wiki: WikiCoverage::Documented("PCA — Packed Causal Attention"),
    },
    PassDescriptor {
        name: "CPKD",
        full_name: "Compiler-Planned Knowledge Distillation",
        source_files: &[
            "crates/nsl-codegen/src/cpkd.rs",
            "crates/nsl-codegen/src/cpkd_fused_loss.rs",
            "crates/nsl-codegen/src/cpkd_spectral.rs",
            "crates/nsl-codegen/src/cpkd_student.rs",
        ],
        // Design-time analysis only — CPKD's runtime half is driven by the
        // `@fused_kl_ce` decorator on a distill block, not by a flag.
        cli_flags: &[f("cpkd-target", C_ONLY), f("cpkd-design-student", C_ONLY)],
        stage: PipelineStage::OnWengert,
        phases: &[CompilePhase::TrainBlock],
        decorator_triggers: &["fused_kl_ce"],
        wiki: WikiCoverage::Documented("CPKD — Compiler-Planned Knowledge Distillation"),
    },
    PassDescriptor {
        name: "CEP",
        full_name: "Compilation-Evaluated Pruning",
        source_files: &[
            "crates/nsl-codegen/src/cep.rs",
            "crates/nsl-codegen/src/cep_extract.rs",
            "crates/nsl-codegen/src/cep_importance.rs",
            "crates/nsl-codegen/src/cep_rewrite.rs",
            "crates/nsl-codegen/src/cep_search.rs",
            "crates/nsl-codegen/src/cep_slice.rs",
        ],
        cli_flags: &[
            f("cep-target", CB),
            f("cep-out", CB),
            f("cep-search", C_ONLY),
            f("cep-profile", C_ONLY),
            f("cep-prune", B_ONLY),
            f("cep-joint", B_ONLY),
            f("cep-sparsity", B_ONLY),
            f("cep-emit-source", B_ONLY),
            f("cep-emit-weights", B_ONLY),
        ],
        // Structured pruning rewrites the model OFFLINE and emits new weights;
        // it is not part of the train-block pass sequence.
        stage: PipelineStage::OutOfBand,
        phases: &[],
        decorator_triggers: &["cep_prune", "cep_search"],
        wiki: WikiCoverage::Documented("CEP — Compilation-Evaluated Pruning"),
    },
    PassDescriptor {
        name: "CFIE",
        full_name: "Compiler-Fused Inference Engine",
        source_files: &[
            "crates/nsl-codegen/src/cfie.rs",
            "crates/nsl-codegen/src/cfie_kv_plan.rs",
            "crates/nsl-codegen/src/cfie_persistent.rs",
            "crates/nsl-codegen/src/cfie_serve.rs",
            "crates/nsl-codegen/src/cfie_speculative.rs",
        ],
        cli_flags: &[f("cfie", B_ONLY), f("cfie-report", B_ONLY)],
        // Inference serving, not training — a separate driver entry point.
        stage: PipelineStage::OutOfBand,
        phases: &[],
        decorator_triggers: &["cfie"],
        wiki: WikiCoverage::Documented("CFIE — Compiler-Fused Inference Engine"),
    },
    PassDescriptor {
        name: "MemoryPlanner",
        full_name: "Compile-time memory planning",
        source_files: &["crates/nsl-codegen/src/memory_planner.rs"],
        // Build-only today. `nsl run` cannot request slab planning; whether
        // that is deliberate or an oversight is recorded, not asserted.
        cli_flags: &[f("memory", B_ONLY), f("memory-report", B_ONLY)],
        stage: PipelineStage::Lowering,
        // TrainBlock is GATE-VERIFIED (`nsl build --memory-report` observes
        // MemoryPlanner there, via the transient arena). Lowering is scoped in
        // `compile_entry`'s memory-plan block but NOT observed by any fixture:
        // that block needs a plannable allocation set the CPU fixtures never
        // produce. Recorded as a claim, flagged as one — the gate asserts
        // observed IN this set, so an unverified member can only ever widen
        // what is accepted, never assert something false.
        phases: &[CompilePhase::TrainBlock, CompilePhase::Lowering],
        decorator_triggers: &[],
        // Documented under its own top-level heading rather than in the
        // per-pass list, because it does not operate on the WengertList — it
        // runs over already-lowered function bodies.
        wiki: WikiCoverage::Undocumented(
            "covered by the top-level '## Memory planning — the slab allocator' \
             section instead; it is not a WengertList pass and does not belong \
             in the per-pass list",
        ),
    },
];

/// Look a pass up by name.
pub fn pass(name: &str) -> Option<&'static PassDescriptor> {
    PASSES.iter().find(|p| p.name == name)
}
