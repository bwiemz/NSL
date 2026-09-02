//! Milestone A: activation contracts — "no advertised feature can be silently
//! inert".
//!
//! # The defect class this closes
//!
//! NSL's request surface (CLI optimization flags, source decorators) is larger
//! than the surface the compiler honours, and the difference has historically
//! been silent: `@cpdt(mode = off)` on a train block compiled to a byte-identical
//! binary on every CLI entry point, `@fase(...)`'s validated config was dropped
//! at its only call site, and `@tie_weights` is advertised in the wiki while
//! appearing in the crates only inside two comments. A user cannot tell a
//! honoured request from an ignored one by exit code.
//!
//! Three of the four pieces needed to close this already existed:
//! [`crate::pass_registry`] holds per-pass `cli_flags` + `decorator_triggers`
//! as gated data; [`crate::pass_trace`] holds the outcome vocabulary
//! ([`PassDisposition`]: applied / declined / advisory) and the execution
//! witness (`NSL_PASS_TRACE=1`). This module is the missing JOIN: at the end
//! of a compile, for every surface the user actually requested, it asks "did
//! the owner record anything?" — and a silent owner becomes a hard error in
//! the CLI instead of a silently-identical binary.
//!
//! # What a contract is
//!
//! One [`Contract`] per advertised surface. The load-bearing field is
//! [`Contract::witness`]: HOW activation is observable. A surface whose
//! activation cannot be observed is not admissible here — that rule is what
//! keeps this table from decaying into decorative metadata (the defect the
//! item-20/21 registries each had to delete a field to fix).
//!
//! Witnesses come in two enforcement tiers:
//!
//! - [`Witness::Disposition`]: the owner is a registered pass that records a
//!   [`PassDisposition`] when reached. Reconciled IN-PROCESS by
//!   [`reconcile`]; a requested-but-silent owner is [`OutcomeState::Unsatisfied`]
//!   and the CLI turns that into an error (`--allow-inert-requests` demotes
//!   it to a warning).
//! - [`Witness::Marker`] / [`Witness::Config`] / [`Witness::Report`]: the
//!   surface is consumed at a named site or emits a named stderr marker.
//!   These cannot be reconciled in-process (a process cannot cheaply observe
//!   its own stderr), so they are enforced by drift gates in
//!   `tests/activation_contract_gate.rs`: the named marker/site must exist in
//!   the tree, and the flag itself must exist on the declared subcommands.
//!
//! Completeness is gated from the OTHER side (item-20 lesson: a registry must
//! gate its own incompleteness): the gate test parses `args.rs` and fails on
//! any optimization-surface flag with neither a contract nor an entry in the
//! explicit [`UNCONTRACTED_FLAGS`] allowlist, so "deliberately not covered"
//! stays a written-down decision rather than a silent omission.

use std::sync::Mutex;

use crate::pass_registry::{self, Subcommand};
use crate::pass_trace::{self, PassDisposition};

/// Decorator names present in the ENTRY module of the compile that just ran.
///
/// Entry module only, deliberately: decorators inside imported/stdlib modules
/// (`@fused_lm_ce` in `stdlib/nsl/nn/losses.nsl`, `@param_role` in
/// `muon.nsl`) are the library author's requests, exercised by the library's
/// own gates — reconciling them against every downstream user compile would
/// hold users answerable for requests they never wrote.
///
/// Same process-global pattern as [`pass_trace`]'s logs, cleared by the same
/// [`pass_trace::reset`] (one reset discipline — see the comment there on why
/// partial clears are never offered).
static REQUESTED_DECORATORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Record the entry module's decorator names for reconciliation. Called at
/// the top of each codegen entry point that receives the entry AST, so every
/// CLI path (normal, shared-lib, standalone, zk) feeds the same log without
/// threading ASTs into the drivers.
pub fn note_entry_module_decorators(module: &nsl_ast::Module, interner: &nsl_lexer::Interner) {
    let mut log = REQUESTED_DECORATORS.lock().unwrap_or_else(|e| e.into_inner());
    for u in nsl_ast::decorator_walk::collect_decorators(module) {
        if let Some(first) = u.deco.name.first() {
            let name = interner.resolve(first.0).unwrap_or("").to_string();
            if !name.is_empty() && !log.contains(&name) {
                log.push(name);
            }
        }
    }
}

/// The decorator names noted by the last compile, in first-seen order.
pub fn requested_decorators() -> Vec<String> {
    REQUESTED_DECORATORS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Clear the request log. Called from [`pass_trace::reset`] ONLY — tests and
/// multi-compile processes reset all compile-scoped globals through one door.
pub(crate) fn reset_requests() {
    REQUESTED_DECORATORS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// A user-visible request surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// A CLI long flag, kebab-case, without the leading `--`, exactly as a
    /// user types it (matches [`pass_registry::CliFlag::flag`]).
    Flag(&'static str),
    /// A source decorator name, without the leading `@`.
    Decorator(&'static str),
}

impl Surface {
    pub fn render(&self) -> String {
        match self {
            Surface::Flag(f) => format!("--{f}"),
            Surface::Decorator(d) => format!("@{d}"),
        }
    }
}

/// HOW a surface's activation is observable. See the module doc for the two
/// enforcement tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Witness {
    /// Owner is a registered pass; activation (or a typed decline) appears in
    /// [`pass_trace::dispositions`]. Reconciled in-process.
    Disposition(&'static str),
    /// Activation emits this exact stderr marker (compile-time stderr for
    /// build-phase features). Enforced by gates, not in-process.
    Marker(&'static str),
    /// The value is consumed unconditionally at the named site — inertness is
    /// impossible by construction (a threshold, a device selector, a seed).
    /// The site is a `file::fn` breadcrumb for the gate and the reader.
    Config(&'static str),
    /// The surface exists to produce a report/artifact; a missing artifact is
    /// already loud. The site names the emitter.
    Report(&'static str),
}

/// One advertised surface and its activation contract.
#[derive(Debug, Clone, Copy)]
pub struct Contract {
    pub surface: Surface,
    /// Subcommands where the surface is accepted. For decorators this is the
    /// set of subcommands whose pipeline can honour the decorator.
    pub on: &'static [Subcommand],
    /// The component answerable for the request: a registered pass name, or a
    /// driver/frontend component name for non-pass surfaces.
    pub owner: &'static str,
    pub witness: Witness,
}

use Subcommand::{Build, Check, Run};
const BR: &[Subcommand] = &[Build, Run];
const CBR: &[Subcommand] = &[Check, Build, Run];
const CB: &[Subcommand] = &[Check, Build];
const CR: &[Subcommand] = &[Check, Run];
const B_ONLY: &[Subcommand] = &[Build];
const C_ONLY: &[Subcommand] = &[Check];
const R_ONLY: &[Subcommand] = &[Run];

const fn deco(
    surface: &'static str,
    on: &'static [Subcommand],
    owner: &'static str,
    witness: Witness,
) -> Contract {
    Contract { surface: Surface::Decorator(surface), on, owner, witness }
}

const fn flag(
    surface: &'static str,
    on: &'static [Subcommand],
    owner: &'static str,
    witness: Witness,
) -> Contract {
    Contract { surface: Surface::Flag(surface), on, owner, witness }
}

/// Contracts for surfaces that are NOT derivable from [`pass_registry`]:
/// driver-owned flags and decorators owned by non-pass components.
///
/// Grown family-by-family; the completeness gate forces every optimization
/// flag to appear either here, in a pass's `cli_flags`, or in
/// [`UNCONTRACTED_FLAGS`] with a reason.
pub static MANUAL_CONTRACTS: &[Contract] = &[
    // ------------------------------------------------------------------
    // Decorators owned by non-pass components (frontend / kernel driver).
    // ------------------------------------------------------------------
    deco("fase", BR, "FASE", Witness::Disposition("FASE")),
    deco("autotune", BR, "autotune", Witness::Marker("[autotune]")),
    // Config, not Marker: the only "[flash-attention]" string in the tree
    // turned out to be registry data (review caught the gate matching its
    // own table); the real consumer is the kernel driver's name match.
    deco("flash_attention", BR, "kernel-driver", Witness::Config("crates/nsl-codegen/src/compiler/kernel.rs")),
    // ------------------------------------------------------------------
    // CLI flags with no pass-registry owner, from the 2026-08-15 consumer
    // audit. `on` mirrors the arg structs that declare the flag and is
    // gated against args.rs; witness strings/sites are gated for existence.
    //
    // The autotune / calibration families (driver-owned, not passes).
    // ------------------------------------------------------------------
    flag("autotune-clean", B_ONLY, "autotune", Witness::Config("crates/nsl-cli/src/commands/build/options.rs")),
    flag("autotune-db", B_ONLY, "autotune", Witness::Marker("[autotune]")),
    flag("autotune-db-sha256", B_ONLY, "autotune", Witness::Config("crates/nsl-codegen/src/autotune.rs")),
    flag("autotune-fresh", B_ONLY, "autotune", Witness::Marker("[autotune]")),
    flag("no-autotune", B_ONLY, "autotune", Witness::Config("crates/nsl-codegen/src/compiler/kernel.rs")),
    flag("calibrate", B_ONLY, "calibration", Witness::Config("crates/nsl-codegen/src/calibration/mod.rs")),
    flag("calibration-batch-size", B_ONLY, "calibration", Witness::Config("crates/nsl-codegen/src/calibration/mod.rs")),
    flag("calibration-data", B_ONLY, "calibration", Witness::Marker("[calibration]")),
    flag("calibration-samples", B_ONLY, "calibration", Witness::Config("crates/nsl-codegen/src/calibration/mod.rs")),
    flag("calibration-timeout", B_ONLY, "calibration", Witness::Config("crates/nsl-codegen/src/calibration/mod.rs")),
    // Gated optimizations with stderr markers (the strongest witnesses).
    flag("cuda-graphs", BR, "cuda-graphs", Witness::Marker("[cuda-graph]")),
    flag("fuse-lm-head", BR, "lm-head-fusion", Witness::Marker("[lm-head-fusion]")),
    flag("fuse-rmsnorm-backward", BR, "source-ad-fusion", Witness::Marker("[fuse]")),
    flag("fuse-wgrad-accum", BR, "wgrad-fusion", Witness::Marker("[wgrad-fusion]")),
    flag("layerwise-accum", BR, "csla", Witness::Marker("[csla]")),
    flag("optim-state-offload", BR, "offload", Witness::Marker("[offload]")),
    flag("param-dtype", BR, "sr-bf16", Witness::Marker("[sr-bf16]")),
    flag("muon-state-dtype", BR, "muon", Witness::Marker("[muon-state]")),
    flag("stream-arena", BR, "weight-stream", Witness::Marker("[weight-stream]")),
    flag("stream-async-writeback", BR, "weight-stream", Witness::Marker("[weight-stream]")),
    flag("stream-prefetch", BR, "weight-stream", Witness::Marker("[weight-stream]")),
    flag("weight-stream", BR, "weight-stream", Witness::Marker("[weight-stream]")),
    flag("transient-arena", BR, "arena", Witness::Marker("[arena]")),
    flag("zero-elementwise", BR, "zero3", Witness::Marker("[zero3]")),
    flag("zero-stage", BR, "zero", Witness::Marker("[zero]")),
    flag("grad-integrity", BR, "grad-integrity", Witness::Marker("[grad-integrity]")),
    flag("gpu-mem-report", R_ONLY, "runtime", Witness::Marker("[gpu-mem]")),
    flag("inspect", R_ONLY, "inspect", Witness::Marker("[inspect]")),
    // Unconditional configuration (consumed at the named site on every
    // compile that accepts the flag; inertness impossible by construction).
    flag("muon-batch-ns", BR, "muon", Witness::Config("crates/nsl-codegen/src/stmt.rs")),
    flag("muon-resident-momentum", BR, "muon", Witness::Config("crates/nsl-codegen/src/stmt_fase.rs")),
    flag("collectives", R_ONLY, "zero", Witness::Config("crates/nsl-cli/src/commands/run.rs")),
    flag("cuda-sync", R_ONLY, "runtime", Witness::Config("crates/nsl-cli/src/commands/build/run.rs")),
    flag("decode-workers", R_ONLY, "serve", Witness::Config("crates/nsl-cli/src/commands/run.rs")),
    flag("prefill-workers", R_ONLY, "serve", Witness::Config("crates/nsl-cli/src/commands/run.rs")),
    flag("health-interval", R_ONLY, "monitor", Witness::Config("crates/nsl-cli/src/commands/run.rs")),
    flag("monitor", R_ONLY, "monitor", Witness::Config("crates/nsl-cli/src/commands/run.rs")),
    flag("debug-training", BR, "driver", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("disable-fusion", BR, "driver", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("source-ad", BR, "driver", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    // `--tape-ad`'s whole effect is clap's source_ad_mode group (tape AD is
    // the default path); the CompileOptions field it once filled was never
    // read and has been removed.
    flag("tape-ad", BR, "clap-group", Witness::Config("crates/nsl-cli/src/args.rs")),
    flag("deterministic", CBR, "driver", Witness::Config("crates/nsl-codegen/src/compiler/main_entry.rs")),
    flag("seed", BR, "driver", Witness::Config("crates/nsl-codegen/src/compiler/main_entry.rs")),
    flag("target", BR, "backend", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("devices", BR, "driver", Witness::Config("crates/nsl-cli/src/commands/build/options.rs")),
    flag("linear-types", CBR, "ownership", Witness::Config("crates/nsl-cli/src/pipeline.rs")),
    flag("pretrain-optimized", BR, "meta-flags", Witness::Config("crates/nsl-cli/src/meta_flags.rs")),
    flag("training-reference", BR, "meta-flags", Witness::Config("crates/nsl-cli/src/meta_flags.rs")),
    flag("trace-ops", R_ONLY, "driver", Witness::Config("crates/nsl-codegen/src/compiler/main_entry.rs")),
    // Weight-aware family.
    flag("weights", CBR, "weight-aware", Witness::Config("crates/nsl-codegen/src/weight_aware.rs")),
    flag("dead-weight-threshold", CB, "weight-aware", Witness::Config("crates/nsl-codegen/src/weight_aware.rs")),
    flag("sparse-threshold", CB, "weight-aware", Witness::Config("crates/nsl-codegen/src/weight_aware.rs")),
    flag("no-constant-fold", B_ONLY, "weight-aware", Witness::Config("crates/nsl-codegen/src/expr/advanced.rs")),
    flag("no-dead-weight", B_ONLY, "weight-aware", Witness::Config("crates/nsl-codegen/src/weight_aware.rs")),
    flag("no-sparse-codegen", B_ONLY, "weight-aware", Witness::Config("crates/nsl-codegen/src/expr/advanced.rs")),
    flag("embed-threshold", B_ONLY, "standalone", Witness::Config("crates/nsl-cli/src/commands/build/standalone.rs")),
    flag("embed-weights", B_ONLY, "standalone", Witness::Config("crates/nsl-cli/src/commands/build/standalone.rs")),
    flag("vram-budget", B_ONLY, "memory-planner", Witness::Config("crates/nsl-codegen/src/memory_planner.rs")),
    // WCET family: build/run only. The six were declared-and-dropped on
    // CheckArgs for a long time; review chose deletion over a bespoke
    // refusal block — clap's own unknown-argument error IS the refusal, and
    // the entry-set gate now enforces the BR scope automatically.
    flag("wcet", BR, "wcet", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("wcet-cert", BR, "wcet", Witness::Report("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("wcet-target", BR, "wcet", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("cpu", BR, "wcet", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("do178c-report", BR, "wcet", Witness::Report("crates/nsl-codegen/src/compiler/mod.rs")),
    flag("fpga-device", BR, "wcet", Witness::Config("crates/nsl-codegen/src/compiler/mod.rs")),
    // Report-only surfaces: the artifact/report is the witness.
    flag("fusion-report", B_ONLY, "fusion", Witness::Report("crates/nsl-codegen/src/fusion_report.rs")),
    flag("nan-analysis", CB, "nan-analysis", Witness::Report("crates/nsl-cli/src/commands/build/normal.rs")),
    flag("perf", C_ONLY, "profile", Witness::Report("crates/nsl-cli/src/profile.rs")),
    flag("gpu", CR, "profile", Witness::Config("crates/nsl-cli/src/profile.rs")),
    flag("shapes", C_ONLY, "shape-debug", Witness::Report("crates/nsl-cli/src/shape_debug.rs")),
    flag("training-report", C_ONLY, "training-report", Witness::Report("crates/nsl-codegen/src/training_report.rs")),
    flag("weight-analysis", C_ONLY, "weight-aware", Witness::Report("crates/nsl-codegen/src/weight_aware.rs")),
    flag("profile", R_ONLY, "profiling", Witness::Report("crates/nsl-cli/src/commands/profile_merge.rs")),
    flag("profile-kernels", R_ONLY, "profiling", Witness::Report("crates/nsl-runtime/src/args.rs")),
    flag("profile-memory", R_ONLY, "profiling", Witness::Report("crates/nsl-runtime/src/args.rs")),
];

/// Flags DELIBERATELY without an activation contract, with the reason
/// written down. The completeness gate fails on any arg-struct field missing
/// from both the contract set and this list, so exclusion is a diff-visible
/// decision (item-20 lesson: a registry must gate its own incompleteness).
pub static UNCONTRACTED_FLAGS: &[(&str, &str)] = &[
    // ---- matmul arithmetic (#583/#586) -------------------------------
    // These select the ARITHMETIC the runtime performs; they activate no
    // compiler pass, so there is no activation contract to write. They are
    // not uncontracted in the sense of being unobserved: each one is
    // rendered into the EXECUTION FINGERPRINT and classified for resume
    // (`nsl-runtime/src/exec_fingerprint.rs`), and each is witnessed by the
    // runtime's cuBLAS math-mode banner. That is a stronger check than an
    // activation contract, and a different one -- it observes what the
    // RUNTIME did, not which pass the compiler ran.
    ("matmul-mode", "runtime GEMM arithmetic; witnessed by the cuBLAS math-mode banner and the `mm` fingerprint key, not by a pass"),
    ("bf16-rounding", "runtime operand-cast rounding; fingerprint key `mmround`, witnessed by the SR banner line"),
    ("bf16-min-ratio", "runtime bf16 eligibility threshold; fingerprint key `mmratio`"),
    ("bf16-cast-cache", "runtime weight-cast cache; fingerprint key `mmcache` (placement class -- bit-preserving, warns on resume)"),
    ("bf16-lt", "runtime cuBLASLt dispatch; fingerprint key `mmlt`"),
    ("bf16-lt-workspace-mib", "runtime Lt workspace cap; fingerprint key `mmltws` (arithmetic class -- the cap filters kernel candidates)"),
    ("no-bf16-lt-tune", "disables Lt timed plan selection; fingerprint key `mmlttune`"),
    ("file", "positional input path, not a feature request"),
    ("args", "positional program arguments forwarded to the compiled binary"),
    ("output", "artifact path plumbing"),
    ("emit-obj", "artifact form plumbing (stop after the object file)"),
    ("dump-ir", "debug dump plumbing"),
    ("dump-tokens", "debug dump plumbing"),
    ("dump-ast", "debug dump plumbing"),
    ("dump-types", "debug dump plumbing"),
    ("shared-lib", "build-flavor selector (dispatches to run_build_shared)"),
    ("standalone", "build-flavor selector (dispatches to run_build_standalone)"),
    ("unikernel", "build-flavor selector (M54)"),
    ("listen", "serve-mode selector"),
    ("trace", "refused on nsl check (see commands/check.rs); nsl debug owns traces"),
    (
        "distribute",
        "REFUSED at dispatch: M43's 3D-parallelism config has no consumer \
         anywhere in the tree — an inert value is an error, not a contract",
    ),
    ("zk-circuit", "build-flavor selector (dispatches to run_build_zk)"),
    ("zk-backend", "zk emission parameter, consumed by the zk build flavor"),
    ("zk-field", "zk emission parameter, consumed by the zk build flavor"),
    ("zk-solidity", "zk emission parameter, consumed by the zk build flavor"),
    ("zk-weights", "zk emission parameter, consumed by the zk build flavor"),
    ("activation-report", "the activation mechanism's own reporting switch"),
    ("allow-inert-requests", "the activation mechanism's own escape hatch"),
    ("allow-unknown-decorators", "the namespace close's own escape hatch"),
];

/// Owners MEASURED to record a [`PassDisposition`] on every compile where
/// their surface is armed — the precondition for runtime enforcement. FASE
/// and CPDT record on EVERY train-block compile (including their mode-off
/// declines); CCR records whenever `--checkpoint-blocks` is set and a train
/// block compiles. Every other pass has additional preconditions (WRGA needs
/// decorators, MemoryPlanner needs plannable single-file allocations, WGGO
/// arms from its own flag family's mode value, ...), so reconciling their
/// silence as "inert" would hard-fail legitimately-quiet paths — review
/// found exactly that for `--wrga-report`'s designed soft path and for
/// `--memory-report` on any multi-file build. Their contracts are
/// consumer-site witnesses instead, enforced by the static gate.
const RUNTIME_ENFORCED_PASSES: &[&str] = &["FASE", "CPDT", "CCR"];
const RUNTIME_ENFORCED_DECORATORS: &[&str] = &["fase", "cpdt"];

/// The full contract set:
/// - pass-registry flags: [`Witness::Disposition`] where the owner is in
///   [`RUNTIME_ENFORCED_PASSES`], else a Config witness on the pass's first
///   source file (activation still declared, verified by gates rather than
///   reconciled at runtime);
/// - the two runtime-enforced decorator triggers;
/// - the manual table;
/// - every remaining KNOWN decorator name from
///   `nsl_semantic::decorator_registry`, as a Config witness on the
///   consumer file that registry row already names — one source of truth,
///   so a decorator can never again reconcile as "plumbing".
///
/// Derived once and memoized (the inputs are all `'static`).
pub fn contracts() -> &'static [Contract] {
    use std::sync::OnceLock;
    static CONTRACTS: OnceLock<Vec<Contract>> = OnceLock::new();
    CONTRACTS.get_or_init(|| {
        let mut out = Vec::new();
        for p in pass_registry::PASSES {
            let runtime = RUNTIME_ENFORCED_PASSES.contains(&p.name);
            for f in p.cli_flags {
                out.push(Contract {
                    surface: Surface::Flag(f.flag),
                    on: f.on,
                    owner: p.name,
                    witness: if runtime {
                        Witness::Disposition(p.name)
                    } else {
                        Witness::Config(p.source_files[0])
                    },
                });
            }
            for d in p.decorator_triggers {
                if RUNTIME_ENFORCED_DECORATORS.contains(d) {
                    out.push(Contract {
                        surface: Surface::Decorator(d),
                        on: BR,
                        owner: p.name,
                        witness: Witness::Disposition(p.name),
                    });
                }
                // Non-runtime triggers fall through to the KNOWN_DECORATORS
                // derivation below, which names the actual consumer.
            }
        }
        for c in MANUAL_CONTRACTS {
            if !out.iter().any(|e| e.surface == c.surface) {
                out.push(*c);
            }
        }
        // Every remaining known decorator: the namespace registry's
        // `read_by` consumer is the witness site. This is what makes "a
        // decorator is never plumbing" structural: closing the namespace
        // over a name automatically gives it an activation contract.
        for d in nsl_semantic::decorator_registry::KNOWN_DECORATORS {
            if !out.iter().any(|e| e.surface == Surface::Decorator(d.name)) {
                out.push(Contract {
                    surface: Surface::Decorator(d.name),
                    on: BR,
                    owner: "frontend",
                    witness: Witness::Config(d.read_by),
                });
            }
        }
        out
    })
}

/// Look up the contract for a requested surface, if any.
pub fn contract_for(surface: &RequestedSurface) -> Option<Contract> {
    contracts().iter().copied().find(|c| match (&c.surface, surface) {
        (Surface::Flag(f), RequestedSurface::Flag(name)) => f == name,
        (Surface::Decorator(d), RequestedSurface::Decorator(name)) => d == name,
        _ => false,
    })
}

/// A surface the user actually requested on THIS invocation: a flag that was
/// explicitly present on the command line, or a decorator present in the
/// program source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestedSurface {
    Flag(String),
    Decorator(String),
}

impl RequestedSurface {
    pub fn render(&self) -> String {
        match self {
            RequestedSurface::Flag(f) => format!("--{f}"),
            RequestedSurface::Decorator(d) => format!("@{d}"),
        }
    }
}

/// What became of one requested surface, after the compile.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub surface: RequestedSurface,
    pub owner: Option<&'static str>,
    pub state: OutcomeState,
}

#[derive(Debug, Clone)]
pub enum OutcomeState {
    /// The owner recorded a disposition — the request was answered, whichever
    /// of the three answers it got. `Declined` IS a satisfied contract: the
    /// user can see why.
    Disposed(PassDisposition),
    /// The witness is structural (Marker/Config/Report) — activation is
    /// enforced by gates, not reconciled in-process. Never an error here.
    ByConstruction(Witness),
    /// The surface's contract says its owner records a disposition when
    /// reached, the surface was requested, and the owner recorded NOTHING.
    /// This is the silent-inertness condition. The CLI turns it into a hard
    /// error (or a warning under `--allow-inert-requests`).
    Unsatisfied { owner: &'static str },
    /// No contract covers this surface. Plumbing flags land here at runtime;
    /// the completeness gate bounds this set statically, so reconcile treats
    /// it as benign rather than re-deriving policy per-process.
    Uncontracted,
    /// The surface has a contract, but not for this subcommand (e.g. a
    /// build-only flag reconciled under check). Reported, never enforced:
    /// clap already rejects genuinely undeclared flags, so this arises only
    /// for surfaces accepted-but-inertly-scoped, which the entry-point
    /// declaration gate pins separately.
    OutOfScope,
}

/// Join the requested surfaces against the contract table and the disposition
/// log of the compile that just ran (the process-global [`pass_trace`] log).
///
/// Enforcement policy (error vs warn vs report) is the CLI's, which knows the
/// entry point and the escape-hatch flags.
pub fn reconcile(requested: &[RequestedSurface], entry: Subcommand) -> Vec<Outcome> {
    reconcile_with(requested, entry, &pass_trace::dispositions())
}

/// [`reconcile`] over an explicit disposition list. Pure: no globals, no I/O,
/// no exit — this is the function unit tests drive, because the global log is
/// process-wide and in-crate tests of other passes write to it concurrently.
pub fn reconcile_with(
    requested: &[RequestedSurface],
    entry: Subcommand,
    dispositions: &[(&'static str, PassDisposition)],
) -> Vec<Outcome> {
    requested
        .iter()
        .map(|req| {
            let Some(c) = contract_for(req) else {
                return Outcome { surface: req.clone(), owner: None, state: OutcomeState::Uncontracted };
            };
            if !c.on.contains(&entry) {
                return Outcome {
                    surface: req.clone(),
                    owner: Some(c.owner),
                    state: OutcomeState::OutOfScope,
                };
            }
            let state = match c.witness {
                Witness::Disposition(pass) => {
                    match dispositions.iter().find(|(p, _)| *p == pass) {
                        Some((_, d)) => OutcomeState::Disposed(*d),
                        None => OutcomeState::Unsatisfied { owner: pass },
                    }
                }
                w => OutcomeState::ByConstruction(w),
            };
            Outcome { surface: req.clone(), owner: Some(c.owner), state }
        })
        .collect()
}

/// The subset of outcomes that must fail the compile (absent the escape
/// hatch): requested, contracted for this entry, owner silent.
pub fn unsatisfied(outcomes: &[Outcome]) -> Vec<&Outcome> {
    outcomes
        .iter()
        .filter(|o| matches!(o.state, OutcomeState::Unsatisfied { .. }))
        .collect()
}

/// Render the full request→outcome table, one `[activation]`-prefixed line
/// per requested surface. Reuses the `[pass-trace]` disposition wording so
/// every outcome category keeps a single spelling in the tree.
pub fn render_report(outcomes: &[Outcome]) -> String {
    if outcomes.is_empty() {
        return "[activation] no optimization surfaces requested\n".to_string();
    }
    let mut s = String::new();
    for o in outcomes {
        let line = match &o.state {
            OutcomeState::Disposed(d) => format!(
                "[activation] {}: {} ({})",
                o.surface.render(),
                pass_trace::render_disposition_line(*d),
                o.owner.unwrap_or("?"),
            ),
            OutcomeState::ByConstruction(w) => format!(
                "[activation] {}: witnessed by construction ({})",
                o.surface.render(),
                match w {
                    Witness::Marker(m) => format!("marker {m}"),
                    Witness::Config(site) => format!("consumed at {site}"),
                    Witness::Report(site) => format!("report from {site}"),
                    // reconcile_with never routes Disposition here; if a
                    // future refactor does, render it rather than panic in
                    // the user's build.
                    Witness::Disposition(p) => format!("disposition owner {p}"),
                },
            ),
            OutcomeState::Unsatisfied { owner } => format!(
                "[activation] {}: UNSATISFIED — {} recorded no disposition",
                o.surface.render(),
                owner,
            ),
            OutcomeState::Uncontracted => {
                format!("[activation] {}: no activation contract (plumbing)", o.surface.render())
            }
            OutcomeState::OutOfScope => format!(
                "[activation] {}: contracted, but not on this subcommand",
                o.surface.render(),
            ),
        };
        s.push_str(&line);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pass-registry flag surfaces as a contract owned by its pass —
    /// Disposition (runtime-reconciled) ONLY for owners measured to record
    /// on every armed compile, Config otherwise. Review found the all-
    /// Disposition version hard-failing designed soft paths
    /// (`--wrga-report` with no WRGA decorators; `--memory-report` on any
    /// multi-file build).
    #[test]
    fn registry_surfaces_become_contracts_with_the_measured_cut() {
        let cs = contracts();
        for p in pass_registry::PASSES {
            let runtime = RUNTIME_ENFORCED_PASSES.contains(&p.name);
            for f in p.cli_flags {
                let c = cs
                    .iter()
                    .find(|c| c.surface == Surface::Flag(f.flag))
                    .unwrap_or_else(|| panic!("--{} has no contract", f.flag));
                assert_eq!(c.owner, p.name, "--{} owner", f.flag);
                if runtime {
                    assert_eq!(c.witness, Witness::Disposition(p.name), "--{} witness", f.flag);
                } else {
                    assert!(
                        matches!(c.witness, Witness::Config(_)),
                        "--{}: non-runtime pass flags carry Config witnesses",
                        f.flag
                    );
                }
            }
            for d in p.decorator_triggers {
                assert!(
                    cs.iter().any(|c| c.surface == Surface::Decorator(d)),
                    "@{d} has no contract"
                );
            }
        }
        // Every KNOWN decorator name has a contract — "a decorator is never
        // plumbing" is structural, not aspirational.
        for d in nsl_semantic::decorator_registry::KNOWN_DECORATORS {
            assert!(
                cs.iter().any(|c| c.surface == Surface::Decorator(d.name)),
                "@{} is a known decorator with no activation contract",
                d.name
            );
        }
    }

    /// A manual entry may not silently shadow a registry-derived one — the
    /// registry wins and the duplicate must be caught by review, so the
    /// derivation keeps first-wins order.
    #[test]
    fn registry_wins_over_manual_duplicates() {
        // @fase is both a FASE decorator_trigger and a MANUAL_CONTRACTS row;
        // the surviving contract must be the registry-derived one.
        let cs = contracts();
        let fase: Vec<_> = cs
            .iter()
            .filter(|c| c.surface == Surface::Decorator("fase"))
            .collect();
        assert_eq!(fase.len(), 1, "exactly one @fase contract must survive dedup");
        assert_eq!(fase[0].witness, Witness::Disposition("FASE"));
    }

    /// The reconciler's three-way split, over an explicit disposition list —
    /// the global log is process-wide and other passes' unit tests write to
    /// it concurrently, so tests never read it.
    #[test]
    fn reconcile_splits_disposed_unsatisfied_uncontracted() {
        let dispositions = [("CCR", PassDisposition::Applied { rewrites: 3 })];
        let requested = vec![
            RequestedSurface::Flag("checkpoint-blocks".into()),  // disposed (CCR spoke)
            RequestedSurface::Flag("checkpoint-stride".into()),  // also CCR: same log entry
            RequestedSurface::Flag("wggo".into()),               // Config witness -> by construction
            RequestedSurface::Flag("output".into()),             // plumbing -> uncontracted
        ];
        let outcomes = reconcile_with(&requested, Subcommand::Build, &dispositions);
        assert!(matches!(outcomes[0].state, OutcomeState::Disposed(_)), "{:?}", outcomes[0]);
        assert!(matches!(outcomes[1].state, OutcomeState::Disposed(_)), "{:?}", outcomes[1]);
        assert!(
            matches!(outcomes[2].state, OutcomeState::ByConstruction(Witness::Config(_))),
            "{:?}",
            outcomes[2]
        );
        assert!(matches!(outcomes[3].state, OutcomeState::Uncontracted), "{:?}", outcomes[3]);
        assert!(unsatisfied(&outcomes).is_empty());

        // The silent-owner arm, isolated: CCR contracted, nothing recorded.
        let outcomes = reconcile_with(
            &[RequestedSurface::Flag("checkpoint-blocks".into())],
            Subcommand::Build,
            &[],
        );
        assert!(
            matches!(outcomes[0].state, OutcomeState::Unsatisfied { owner: "CCR" }),
            "{:?}",
            outcomes[0]
        );
    }

    /// Check-side reconciliation of a build/run-scoped surface reports
    /// OutOfScope rather than Unsatisfied — check runs no codegen, so a
    /// silent owner there is not evidence of inertness.
    #[test]
    fn check_scope_is_reported_not_enforced() {
        let outcomes = reconcile_with(
            &[RequestedSurface::Decorator("cpdt".into())],
            Subcommand::Check,
            &[],
        );
        assert!(
            matches!(outcomes[0].state, OutcomeState::OutOfScope),
            "{:?}",
            outcomes[0]
        );
        assert!(unsatisfied(&outcomes).is_empty());
    }

    /// The report renders one line per requested surface and names the silent
    /// owner in the UNSATISFIED line — that string is the user's only pointer
    /// to WHICH component ignored them.
    #[test]
    fn report_names_the_silent_owner() {
        let outcomes = reconcile_with(
            &[RequestedSurface::Decorator("cpdt".into())],
            Subcommand::Build,
            &[],
        );
        let report = render_report(&outcomes);
        assert!(
            report.contains("@cpdt: UNSATISFIED — CPDT recorded no disposition"),
            "report was: {report}"
        );
    }

    /// A decline IS a satisfied contract — the reconciler must never treat
    /// Declined as inertness, or every legitimately-declined request would
    /// fail the build it correctly declined on.
    #[test]
    fn declined_is_satisfied_not_unsatisfied() {
        let dispositions = [(
            "CCR",
            PassDisposition::Declined { reason: crate::pass_trace::DeclineReason::ModeOff },
        )];
        let outcomes = reconcile_with(
            &[RequestedSurface::Flag("checkpoint-blocks".into())],
            Subcommand::Build,
            &dispositions,
        );
        assert!(matches!(outcomes[0].state, OutcomeState::Disposed(_)), "{:?}", outcomes[0]);
        assert!(unsatisfied(&outcomes).is_empty());
    }
}
