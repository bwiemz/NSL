//! Item 20: a declarative registry of feature-composition rules.
//!
//! NSL has ~14 composable training features and a large body of pairwise rules
//! governing them — "A requires B", "A does not compose with B". Fifteen are
//! clap attributes (spread over twelve constrained fields); the rest live as
//! hand-written refusals deep in `stmt.rs` / `stmt_fase.rs` / the CLI, each
//! covered (if at all) by a bespoke test. Nothing tied the two halves
//! together, so three things can rot silently:
//!
//! 1. **`nsl run` and `nsl build` drift.** `args.rs` declares every shared flag
//!    TWICE (`BuildArgs`, `RunArgs`). A `conflicts_with` added to one and not
//!    the other means the same composition is refused by one subcommand and
//!    accepted by the other. They agree today; nothing enforces that.
//! 2. **A refusal is deleted or reworded** and the composition becomes
//!    silently permitted. The repo's `Deferral must refuse` doctrine says an
//!    unsupported combination must fail loudly — but a deferral with no gate
//!    is one edit away from gone.
//! 3. **The registry itself lies.** The obvious failure of a hand-maintained
//!    table: it says a rule is enforced, the enforcement is gone, and the
//!    table's existence makes everyone stop checking.
//!
//! The gates in `tests/feature_composition_gate.rs` attack all three. (3) is
//! why nothing here is transcribed on trust: every entry is cross-checked
//! against the actual source — clap rules against the parsed attributes in
//! BOTH arg blocks, codegen rules against the refusal text in the file that
//! is supposed to contain it.
//!
//! # Rules can be enforced at two layers
//!
//! Several compositions are refused by BOTH clap and a source-level check —
//! `--weight-stream requires --layerwise-accum` is a clap attribute *and* a
//! `stmt.rs` refusal. Measured 2026-07-26: clap wins, so the `stmt.rs` arm is
//! unreachable by driving the binary and exists as defense-in-depth for
//! callers that build `CompileOptions` directly. Both are registered on
//! purpose — the source rule still needs the deleted-refusal gate — but only
//! the clap layer can be exercised end-to-end, which is why the subprocess
//! tier covers fewer rules than the registry holds.
//!
//! # What a rule does NOT assert
//!
//! That the composition is *correctly* refused for a good reason. This
//! registry pins the refusal's existence, location and message — not its
//! justification. Reading a rule as "this combination is genuinely
//! impossible" would be over-reading it; several are `yet`-flavoured
//! deferrals that should eventually become supported compositions, at which
//! point the rule is deleted rather than "fixed".

/// Which direction the constraint runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleKind {
    /// `flag` is meaningless or unsound without `other`.
    Requires,
    /// `flag` and `other` cannot both be active.
    Conflicts,
}

/// Where the rule is actually enforced — and therefore how it can be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enforcement {
    /// A clap attribute, rejected at argument-parse time. Must be present and
    /// IDENTICAL in both `BuildArgs` and `RunArgs`.
    Clap,
    /// A hand-written refusal in compiler or CLI code. `file` is repo-relative
    /// and `fragment` is a distinctive, whitespace-normalized slice of the
    /// user-facing message.
    Source {
        file: &'static str,
        fragment: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureRule {
    /// The flag the rule is about, as the user types it (`--weight-stream`).
    pub flag: &'static str,
    pub kind: RuleKind,
    /// The partner flag. For clap rules this is the clap FIELD name
    /// (`weight_stream`), because that is what the attribute contains.
    pub other: &'static str,
    pub enforcement: Enforcement,
}

const fn clap_rule(flag: &'static str, kind: RuleKind, other: &'static str) -> FeatureRule {
    FeatureRule {
        flag,
        kind,
        other,
        enforcement: Enforcement::Clap,
    }
}

const fn src_rule(
    flag: &'static str,
    kind: RuleKind,
    other: &'static str,
    file: &'static str,
    fragment: &'static str,
) -> FeatureRule {
    FeatureRule {
        flag,
        kind,
        other,
        enforcement: Enforcement::Source { file, fragment },
    }
}

const STMT: &str = "crates/nsl-codegen/src/stmt.rs";
const STMT_FASE: &str = "crates/nsl-codegen/src/stmt_fase.rs";
const CLI_RUN: &str = "crates/nsl-cli/src/commands/run.rs";
const CALIB: &str = "crates/nsl-codegen/src/calibration/binary_codegen.rs";
const CLI_CHECK: &str = "crates/nsl-cli/src/commands/check.rs";
const CLI_REPORTS: &str = "crates/nsl-cli/src/commands/build/reports.rs";
const CLI_OPTIONS: &str = "crates/nsl-cli/src/commands/build/options.rs";
const CLI_CEP: &str = "crates/nsl-cli/src/commands/cep.rs";

/// Every composition rule the compiler enforces, measured from the source
/// rather than recalled. Ordered by subsystem so a reader can see the shape
/// of each feature's envelope in one place.
pub const FEATURE_RULES: &[FeatureRule] = &[
    // ── clap-enforced (parse time; must match in BuildArgs AND RunArgs) ────
    clap_rule("--checkpoint-selective", RuleKind::Requires, "checkpoint_blocks"),
    clap_rule("--checkpoint-budget-mib", RuleKind::Requires, "checkpoint_blocks"),
    clap_rule("--checkpoint-stride", RuleKind::Requires, "checkpoint_blocks"),
    clap_rule("--checkpoint-compress", RuleKind::Requires, "checkpoint_selective"),
    clap_rule("--layerwise-accum", RuleKind::Requires, "checkpoint_blocks"),
    clap_rule("--weight-stream", RuleKind::Requires, "layerwise_accum"),
    clap_rule("--stream-arena", RuleKind::Requires, "weight_stream"),
    clap_rule("--stream-prefetch", RuleKind::Requires, "stream_arena"),
    clap_rule("--stream-async-writeback", RuleKind::Requires, "stream_arena"),
    clap_rule("--fuse-rmsnorm-backward", RuleKind::Requires, "source_ad"),
    clap_rule("--pretrain-optimized", RuleKind::Conflicts, "tape_ad"),
    // Item 7 (`--fuse-wgrad-accum`) — added 2026-07-26.
    clap_rule("--fuse-wgrad-accum", RuleKind::Requires, "source_ad"),
    clap_rule("--fuse-wgrad-accum", RuleKind::Conflicts, "grad_integrity"),
    clap_rule("--fuse-wgrad-accum", RuleKind::Conflicts, "optim_state_offload"),
    clap_rule("--fuse-wgrad-accum", RuleKind::Conflicts, "layerwise_accum"),
    // ── CSLA (`--layerwise-accum`) envelope ────────────────────────────────
    src_rule(
        "--weight-stream",
        RuleKind::Requires,
        "--layerwise-accum",
        STMT,
        "--weight-stream requires --layerwise-accum",
    ),
    src_rule(
        "--layerwise-accum",
        RuleKind::Requires,
        "--source-ad",
        STMT,
        "--layerwise-accum requires --source-ad",
    ),
    src_rule(
        "--layerwise-accum",
        RuleKind::Requires,
        "--checkpoint-blocks",
        STMT,
        "--layerwise-accum requires an active checkpoint plan: pass --checkpoint-blocks",
    ),
    src_rule(
        "--layerwise-accum",
        RuleKind::Conflicts,
        "--checkpoint-compress",
        STMT,
        "--layerwise-accum is incompatible with --checkpoint-compress",
    ),
    src_rule(
        "--layerwise-accum",
        RuleKind::Conflicts,
        "--zero-stage",
        STMT,
        "--layerwise-accum is incompatible with --zero-stage 1/2",
    ),
    src_rule(
        "--layerwise-accum",
        RuleKind::Conflicts,
        "--wggo",
        STMT,
        "Drop --wggo overrides or --layerwise-accum",
    ),
    // CORRECTED after review: the guard here is
    // `wrap_precision && optim_state_offload`, NOT `layerwise_accum &&
    // optim_state_offload`. Plain CSLA + offload is a SUPPORTED path (see the
    // staging note above the guard); only the reduced-precision moment plan
    // on top of it is ungated. The original entry claimed a refusal that does
    // not exist — precisely the "registry lies about coverage" rot this
    // module names, found in the registry's own first version.
    src_rule(
        "--optim-state-offload",
        RuleKind::Conflicts,
        "CPDT reduced-precision moment plan",
        STMT,
        "--layerwise-accum with --optim-state-offload does not yet support a CPDT reduced-precision moment plan",
    ),
    // ── ZeRO ───────────────────────────────────────────────────────────────
    src_rule(
        "--zero-stage",
        RuleKind::Requires,
        "--weight-stream",
        STMT,
        "--zero-stage 3 requires --layerwise-accum --weight-stream",
    ),
    src_rule(
        "--zero-stage",
        RuleKind::Conflicts,
        "--optim-state-offload",
        STMT,
        "--zero-stage 3 with --optim-state-offload is not lowered",
    ),
    src_rule(
        "--zero-stage",
        RuleKind::Conflicts,
        "--wggo-moment-precision",
        STMT,
        "Drop --wggo-moment-precision / the CPDT precision plan, or use --zero-stage 2",
    ),
    src_rule(
        "--zero-stage",
        RuleKind::Conflicts,
        "--wggo",
        STMT,
        "Drop --wggo mode overrides or --zero-stage",
    ),
    // ── SR-BF16 (`--param-dtype bf16-sr`) envelope ─────────────────────────
    src_rule(
        "--param-dtype",
        RuleKind::Requires,
        "--weight-stream",
        STMT,
        "--param-dtype bf16-sr requires --weight-stream",
    ),
    src_rule(
        "--param-dtype",
        RuleKind::Requires,
        "FASE-Deferred plan",
        STMT,
        "--param-dtype bf16-sr requires the FASE-Deferred plan",
    ),
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "--zero-stage",
        STMT,
        "--param-dtype bf16-sr does not compose with --zero-stage",
    ),
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "--optim-state-offload",
        STMT,
        "--param-dtype bf16-sr does not compose with --optim-state-offload",
    ),
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "--wggo-moment-precision",
        STMT,
        "--param-dtype bf16-sr does not compose with reduced-precision optimizer moments (drop --wggo-moment-precision",
    ),
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "--training-reference",
        STMT,
        "--param-dtype bf16-sr requires the fused optimizer step, which --training-reference disables",
    ),
    // Found by the completeness sweep below — the registry had CSLA x --wggo
    // and ZeRO x --wggo but not this one.
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "WGGO per-layer FASE overrides",
        STMT,
        "--param-dtype bf16-sr does not compose with WGGO per-layer FASE overrides",
    ),
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "--optim-state-offload",
        STMT_FASE,
        "--param-dtype bf16-sr does not compose with reduced-precision moment plans or --optim-state-offload",
    ),
    // ── Muon envelope ──────────────────────────────────────────────────────
    src_rule(
        "--muon-batch-ns",
        RuleKind::Conflicts,
        "--layerwise-accum",
        STMT,
        "--muon-batch-ns does not compose with --layerwise-accum",
    ),
    src_rule(
        "--muon-batch-ns",
        RuleKind::Conflicts,
        "--optim-state-offload",
        STMT,
        "--muon-batch-ns does not compose with --optim-state-offload",
    ),
    src_rule(
        "--muon-batch-ns",
        RuleKind::Conflicts,
        "--muon-state-dtype",
        STMT,
        "--muon-batch-ns does not compose with --muon-state-dtype bf16",
    ),
    src_rule(
        "--muon-batch-ns",
        RuleKind::Conflicts,
        "--param-dtype",
        STMT,
        "--muon-batch-ns does not compose with --param-dtype bf16-sr",
    ),
    src_rule(
        "--muon-batch-ns",
        RuleKind::Conflicts,
        "--zero-stage",
        STMT,
        "--muon-batch-ns does not compose with --zero-stage",
    ),
    src_rule(
        "--muon-resident-momentum",
        RuleKind::Requires,
        "--optim-state-offload",
        STMT,
        "--muon-resident-momentum is only meaningful with --optim-state-offload",
    ),
    src_rule(
        "--muon-resident-momentum",
        RuleKind::Conflicts,
        "--muon-state-dtype",
        STMT,
        "--muon-resident-momentum does not compose with --muon-state-dtype bf16",
    ),
    src_rule(
        "--muon-resident-momentum",
        RuleKind::Conflicts,
        "--zero-stage",
        STMT,
        "--muon-resident-momentum does not compose with --zero-stage",
    ),
    src_rule(
        "--muon-state-dtype",
        RuleKind::Requires,
        "--layerwise-accum",
        STMT,
        "--muon-state-dtype bf16 requires --layerwise-accum",
    ),
    src_rule(
        "--muon-state-dtype",
        RuleKind::Conflicts,
        "--zero-stage",
        STMT,
        "--muon-state-dtype bf16 does not compose with --zero-stage",
    ),
    src_rule(
        "--muon-state-dtype",
        RuleKind::Conflicts,
        "--optim-state-offload",
        STMT,
        "--muon-state-dtype bf16 does not compose with --optim-state-offload",
    ),
    // CORRECTED after review: the guard is `optimizer_name != "muon"`. This is
    // a flag x OPTIMIZER constraint, not flag x flag — `--wggo-moment-precision`
    // appears only as advice in the message. Recorded with a non-flag `other`
    // rather than deleted, so the refusal keeps its deleted-refusal gate.
    src_rule(
        "--muon-state-dtype",
        RuleKind::Requires,
        "muon optimizer",
        STMT,
        "--muon-state-dtype bf16 applies to the Muon optimizer only",
    ),
    // Second enforcement site for --param-dtype x --training-reference (the
    // stmt.rs one guards the plan, this one the optimizer program shape).
    // Found by the completeness sweep, not by hand.
    src_rule(
        "--param-dtype",
        RuleKind::Conflicts,
        "--training-reference",
        STMT_FASE,
        "NSL_FASE_FUSED_STEP=0 / --training-reference, or drop --param-dtype bf16-sr",
    ),
    // ── CPDT ───────────────────────────────────────────────────────────────
    src_rule(
        "--cpdt",
        RuleKind::Requires,
        "--cpdt-num-gpus",
        CLI_RUN,
        "--cpdt requires --cpdt-num-gpus N",
    ),
    src_rule(
        "--cpdt",
        RuleKind::Requires,
        "--weights",
        CLI_RUN,
        "1. Add --weights <path.safetensors> to this invocation.",
    ),
    // ── nsl check / report subcommands ─────────────────────────────────────
    // Found by widening the sweep's search set beyond the files the registry
    // already cited — a bootstrapping blind spot: a refusal in a file with no
    // registry entry was invisible to the sweep that is supposed to find it.
    src_rule(
        "--gpu",
        RuleKind::Requires,
        "--perf",
        CLI_CHECK,
        "error: --gpu requires --perf",
    ),
    src_rule(
        "--cpkd-target",
        RuleKind::Requires,
        "--cpkd-design-student",
        CLI_CHECK,
        "--cpkd-target requires --cpkd-design-student",
    ),
    src_rule(
        "--wrga-ablate",
        RuleKind::Requires,
        "--wrga-analyze",
        CLI_CHECK,
        "--wrga-ablate requires --wrga-analyze or --wrga-compare",
    ),
    src_rule(
        "--wrga-report",
        RuleKind::Requires,
        "--source-ad",
        CLI_REPORTS,
        "--wrga-report requires --source-ad when WRGA decorators are present",
    ),
    src_rule(
        "--muon-batch-ns",
        RuleKind::Requires,
        "muon optimizer",
        STMT,
        "--muon-batch-ns requires the muon optimizer",
    ),
    // ── build / cep / calibration subcommands ──────────────────────────────
    // All twelve found by the widened sweep, none by hand.
    src_rule(
        "--calibrate",
        RuleKind::Requires,
        "--calibration-data",
        CLI_OPTIONS,
        "requires --calibration-data <PATH>",
    ),
    src_rule(
        "--standalone",
        RuleKind::Requires,
        "--weights",
        CLI_OPTIONS,
        "--standalone requires -w/--weights <path>",
    ),
    src_rule(
        "--cep-prune",
        RuleKind::Conflicts,
        "--cep-joint",
        CLI_OPTIONS,
        "--cep-prune and --cep-joint are mutually exclusive",
    ),
    src_rule(
        "--cep-prune",
        RuleKind::Requires,
        "--weights",
        CLI_CEP,
        "--cep-prune requires --weights <file.safetensors>",
    ),
    src_rule(
        "--cep-joint",
        RuleKind::Requires,
        "--weights",
        CLI_CEP,
        "--cep-joint requires --weights <file.safetensors>",
    ),
    src_rule(
        "--cep-search",
        RuleKind::Conflicts,
        "--cep-profile",
        CLI_CHECK,
        "--cep-search and --cep-profile are mutually exclusive",
    ),
    src_rule(
        "--cpkd-design-student",
        RuleKind::Conflicts,
        "--cep-search",
        CLI_CHECK,
        "--cpkd-design-student is mutually exclusive with --cep-search/--cep-profile",
    ),
    src_rule(
        "--wrga-analyze",
        RuleKind::Conflicts,
        "--wrga-compare",
        CLI_CHECK,
        "--wrga-analyze and --wrga-compare are mutually exclusive",
    ),
    src_rule(
        "--weight-analysis",
        RuleKind::Requires,
        "--weights",
        CLI_CHECK,
        "--weight-analysis requires --weights <path>",
    ),
    // ── CUDA graphs ────────────────────────────────────────────────────────
    src_rule(
        "--cuda-graphs",
        RuleKind::Requires,
        "--source-ad",
        STMT,
        "--cuda-graphs requires --source-ad",
    ),
    src_rule(
        "--cuda-graphs",
        RuleKind::Conflicts,
        "--zero-stage",
        STMT,
        "--cuda-graphs does not compose with --zero-stage",
    ),
    src_rule(
        "--cuda-graphs",
        RuleKind::Conflicts,
        "--cuda-sync",
        CLI_RUN,
        "--cuda-graphs does not compose with --cuda-sync",
    ),
    src_rule(
        "--cuda-graphs",
        RuleKind::Conflicts,
        "--profile-kernels",
        CLI_RUN,
        "--cuda-graphs does not compose with --profile-kernels",
    ),
    // ── Item 7: refusals that clap CANNOT express ──────────────────────────
    // Both are unreachable while the clap conflicts above hold; they exist so
    // the reason survives if a clap attribute is ever dropped, and so the
    // deleted-refusal gate can see them.
    src_rule(
        "--fuse-wgrad-accum",
        RuleKind::Conflicts,
        "CSLA window replay",
        STMT,
        "--fuse-wgrad-accum fired inside the CSLA window replay",
    ),
    src_rule(
        "--fuse-wgrad-accum",
        RuleKind::Conflicts,
        "calibration binary",
        CALIB,
        "--fuse-wgrad-accum cannot be used when emitting a calibration binary",
    ),
];

/// Rules whose enforcement is a clap attribute.
pub fn clap_rules() -> impl Iterator<Item = &'static FeatureRule> {
    FEATURE_RULES
        .iter()
        .filter(|r| matches!(r.enforcement, Enforcement::Clap))
}

/// Rules enforced by a hand-written refusal in compiler / CLI source.
pub fn source_rules() -> impl Iterator<Item = &'static FeatureRule> {
    FEATURE_RULES
        .iter()
        .filter(|r| matches!(r.enforcement, Enforcement::Source { .. }))
}

/// `--flag-name` → the clap field identifier (`flag_name`).
pub fn flag_to_field(flag: &str) -> String {
    flag.trim_start_matches("--").replace('-', "_")
}
