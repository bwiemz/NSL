//! The `optimizer:` / `scheduler:` / `callbacks:` section contract: the
//! training-DSL section namespaces, CLOSED.
//!
//! This extends the Training Configuration Contract (see
//! [`crate::train_config`]) from the `train(...)` header to the sections.
//! Before this module, the section kwargs were the same defect class the
//! header contract removed: codegen matched known optimizer kwargs with
//! `_ => {}` (a typo'd `lrr=0.01` silently trained at the default lr), a
//! known kwarg with the wrong literal shape was silently dropped inside its
//! own match arm (`momentum=1` — an int — kept the 0.0 default), an unknown
//! scheduler NAME fell through to "no change" and trained at constant lr, a
//! typo'd scheduler kwarg silently took a hardcoded default, positional args
//! were skipped before any name matched, duplicate kwargs resolved
//! last-wins in the optimizer but first-wins in the scheduler, and a
//! callback with an unknown name was collected and never emitted. Ignored
//! configuration is not a compatibility strategy — it trains a different
//! model than the source describes.
//!
//! ONE resolver, THREE consumers: the semantic checker calls it (span-
//! labelled diagnostics on `nsl check` and every build, for both train and
//! distill blocks — distill sections travel verbatim into the synthesized
//! train block, cpkd.rs documents this), and codegen calls it again as the
//! backstop and single source of truth for the values it lowers (the
//! standard path, the pipelined twin, and CPDT's AdamW hyperparameter
//! extraction — previously three independent parses with three different
//! accepted sets).
//!
//! Unlike the flat header key set, the optimizer kwarg namespace is
//! per-optimizer: the accepted keys are exactly the stdlib step-function
//! parameters (`sgd_step` has no betas; `lion_step` has no eps;
//! `soap_step` has no weight_decay), so `SGD(beta2=0.5)` is refused rather
//! than accepted-and-never-read. Scheduler kwargs are per-scheduler for the
//! same reason, and defaults live HERE (single source of truth) rather than
//! scattered through the lowering.

use nsl_ast::block::TrainSection;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

use crate::train_config::{fold_int_literal, fold_numeric_literal, TrainConfigPurpose};

/// Every role name `no_decay=[...]` accepts. This is the single source of
/// truth — `nsl-codegen/src/param_roles.rs` re-exports it and unit-tests
/// that the classifier can only produce roles from this set, so the
/// accepted set cannot drift from the set the classifier assigns.
pub const VALID_ROLES: &[&str] = &["vector", "embedding", "head", "hidden"];

/// The six optimizers the training DSL lowers. Anything else refuses at
/// resolve time (previously: passed `nsl check` clean and refused deep in
/// codegen at step-function mangling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerKind {
    Sgd,
    Adam,
    AdamW,
    Lion,
    Muon,
    Soap,
}

impl OptimizerKind {
    /// The lowercase name codegen keys everything on (step-fn mangling,
    /// state-buffer counts, checkpoint gating, FASE planning).
    pub fn as_str(self) -> &'static str {
        match self {
            OptimizerKind::Sgd => "sgd",
            OptimizerKind::Adam => "adam",
            OptimizerKind::AdamW => "adamw",
            OptimizerKind::Lion => "lion",
            OptimizerKind::Muon => "muon",
            OptimizerKind::Soap => "soap",
        }
    }

    /// Spelling used in diagnostics.
    fn display(self) -> &'static str {
        match self {
            OptimizerKind::Sgd => "SGD",
            OptimizerKind::Adam => "Adam",
            OptimizerKind::AdamW => "AdamW",
            OptimizerKind::Lion => "Lion",
            OptimizerKind::Muon => "Muon",
            OptimizerKind::Soap => "SOAP",
        }
    }

    /// Case-insensitive, matching the pre-contract codegen behavior.
    fn parse(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "sgd" => Some(OptimizerKind::Sgd),
            "adam" => Some(OptimizerKind::Adam),
            "adamw" => Some(OptimizerKind::AdamW),
            "lion" => Some(OptimizerKind::Lion),
            "muon" => Some(OptimizerKind::Muon),
            "soap" => Some(OptimizerKind::Soap),
            _ => None,
        }
    }

    /// The accepted kwargs — exactly the stdlib step-function parameters
    /// (plus `no_decay`, the weight-decay exemption, wherever weight decay
    /// itself applies). A key accepted here must be consumed by the emitted
    /// step call; an accepted-but-never-read key is what this module
    /// exists to prevent.
    fn accepted_kwargs(self) -> &'static [&'static str] {
        match self {
            // stdlib/nsl/optim/sgd.nsl: sgd_step(.., lr, momentum,
            // dampening, weight_decay, nesterov)
            OptimizerKind::Sgd => &[
                "lr",
                "momentum",
                "dampening",
                "weight_decay",
                "nesterov",
                "no_decay",
            ],
            // adam.nsl / adamw.nsl: (.., lr, beta1, beta2, eps,
            // weight_decay, t)
            OptimizerKind::Adam | OptimizerKind::AdamW => {
                &["lr", "beta1", "beta2", "eps", "weight_decay", "no_decay"]
            }
            // lion.nsl: (.., lr, beta1, beta2, weight_decay) — no eps.
            OptimizerKind::Lion => &["lr", "beta1", "beta2", "weight_decay", "no_decay"],
            // soap.nsl: (.., lr, beta1, beta2, eps, t) — no weight_decay,
            // so no no_decay either.
            OptimizerKind::Soap => &["lr", "beta1", "beta2", "eps"],
            // muon.nsl: the full set (the mixed step's AdamW arm consumes
            // the betas/eps; adamw_lr/ns_steps are Muon-only knobs).
            OptimizerKind::Muon => &[
                "lr",
                "adamw_lr",
                "momentum",
                "nesterov",
                "ns_steps",
                "weight_decay",
                "beta1",
                "beta2",
                "eps",
                "no_decay",
            ],
        }
    }
}

const OPTIMIZER_NAMES_LABEL: &str = "expected SGD, Adam, AdamW, Lion, Muon, or SOAP";

/// A fully validated optimizer section. Defaults match the pre-contract
/// codegen locals bit-for-bit, including the Muon spec-default backfill
/// (`Muon()` means lr=0.02, momentum=0.95, nesterov=true — spec/10), which
/// previously lived only on the standard lowering path so bare Muon under
/// `@pipeline` silently degraded to momentum-free lr=0.01.
#[derive(Debug, Clone)]
pub struct ResolvedOptimizer {
    pub kind: OptimizerKind,
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Newton-Schulz iteration count; integer-valued (validated), stored as
    /// f64 because the stdlib `muon_step` signature takes float.
    pub ns_steps: f64,
    /// Muon only: the AdamW arm's rate, threaded by codegen as the fixed
    /// ratio adamw_lr/lr of the scheduled lr. `None` → the arm follows
    /// `lr` exactly. Validated positive, and lr > 0 whenever this is set.
    pub adamw_lr: Option<f64>,
    /// Roles named in `no_decay`, minus `"vector"` (mirrors codegen's
    /// `NoDecayScope.static_roles`).
    pub no_decay_static_roles: Vec<String>,
    /// `no_decay` included `"vector"` → exempt anything not rank-2 at
    /// runtime (mirrors `NoDecayScope.exempt_non_rank2`).
    pub no_decay_exempt_non_rank2: bool,
}

/// A fully validated scheduler section, defaults applied. The lowering
/// passes these as f64 constants to `nsl__optim__schedulers__<fn_name>`
/// after the auto-injected `(base_lr, step)` prefix — kwarg order here
/// matches the stdlib signatures exactly.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedScheduler {
    ConstantLr,
    StepLr { step_size: f64, gamma: f64 },
    ExponentialLr { gamma: f64 },
    LinearDecay { total_steps: f64, end_factor: f64 },
    CosineAnneal { t_max: f64, eta_min: f64 },
    WarmupCosine { warmup_steps: f64, total_steps: f64, min_lr: f64 },
    OneCycle { max_lr: f64, total_steps: f64, pct_start: f64 },
}

impl ResolvedScheduler {
    /// The stdlib function name (`stdlib/nsl/optim/schedulers.nsl`) the
    /// lowering mangles and calls.
    pub fn fn_name(&self) -> &'static str {
        match self {
            ResolvedScheduler::ConstantLr => "constant_lr",
            ResolvedScheduler::StepLr { .. } => "step_lr",
            ResolvedScheduler::ExponentialLr { .. } => "exponential_lr",
            ResolvedScheduler::LinearDecay { .. } => "linear_decay",
            ResolvedScheduler::CosineAnneal { .. } => "cosine_anneal",
            ResolvedScheduler::WarmupCosine { .. } => "warmup_cosine",
            ResolvedScheduler::OneCycle { .. } => "one_cycle",
        }
    }
}

const SCHEDULER_NAMES_LABEL: &str = "expected constant_lr, step_lr, exponential_lr, \
     linear_decay, cosine_anneal, warmup_cosine, or one_cycle";

/// Canonicalize a scheduler name — case-insensitive with both the
/// snake_case and CamelCase-collapsed spellings accepted, matching the
/// pre-contract lowering's dual-spelling match.
fn canonical_scheduler_name(name: &str) -> Option<&'static str> {
    match name.to_lowercase().as_str() {
        "constant_lr" | "constantlr" => Some("constant_lr"),
        "step_lr" | "steplr" => Some("step_lr"),
        "exponential_lr" | "exponentiallr" => Some("exponential_lr"),
        "linear_decay" | "lineardecay" => Some("linear_decay"),
        "cosine_anneal" | "cosineanneal" => Some("cosine_anneal"),
        "warmup_cosine" | "warmupcosine" => Some("warmup_cosine"),
        "one_cycle" | "onecycle" => Some("one_cycle"),
        _ => None,
    }
}

/// The callback names codegen actually emits (`on_step` per micro-batch,
/// `on_epoch`/`on_epoch_end` per epoch). Anything else was previously
/// collected and silently never compiled.
pub const CALLBACK_NAMES: &[&str] = &["on_step", "on_epoch", "on_epoch_end"];

const CALLBACK_NAMES_LABEL: &str = "expected on_step, on_epoch, or on_epoch_end";

/// The parameter names codegen binds for each callback — anything else hit
/// a silent `_ => iconst(0)` fallback: `on_step(step, lss)` type-checked
/// clean (the checker typed unknown params as Unknown) and logged a
/// constant 0 forever. Any subset in any order is legal (`on_step(step)`
/// is a committed corpus shape).
fn callback_param_names(callback: &str) -> &'static [&'static str] {
    match callback {
        "on_step" => &["step", "loss"],
        "on_epoch" | "on_epoch_end" => &["epoch", "loss"],
        other => unreachable!("no param table for callback '{other}'"),
    }
}

/// The optimizer and scheduler sections, fully validated.
#[derive(Debug, Clone)]
pub struct ResolvedOptimConfig {
    pub optimizer: ResolvedOptimizer,
    pub scheduler: Option<ResolvedScheduler>,
}

/// True numeric-literal fold plus a bool fold for `nesterov`.
fn fold_bool_literal(kind: &ExprKind) -> Option<bool> {
    match kind {
        ExprKind::BoolLiteral(b) => Some(*b),
        ExprKind::Paren(inner) => fold_bool_literal(&inner.kind),
        _ => None,
    }
}

/// Validate the `optimizer:` / `scheduler:` / `callbacks:` sections of a
/// train (or distill — sections travel verbatim) block and produce the
/// typed config. All violations are collected so one `nsl check` shows
/// every problem. Takes the section list rather than a `TrainBlock` so
/// `check_distill_block` can call it on `DistillBlock.sections` directly.
pub fn resolve_optim_config(
    sections: &[TrainSection],
    block_span: Span,
    resolve_sym: &dyn Fn(Symbol) -> String,
    purpose: TrainConfigPurpose,
) -> Result<ResolvedOptimConfig, Vec<Diagnostic>> {
    let mut diags: Vec<Diagnostic> = Vec::new();

    let mut optimizer: Option<ResolvedOptimizer> = None;
    let mut optimizer_seen = false;
    let mut scheduler_expr: Option<&nsl_ast::expr::Expr> = None;
    let mut scheduler_seen = false;
    let mut seen_callbacks: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for section in sections {
        match section {
            TrainSection::Optimizer(expr) => {
                if optimizer_seen {
                    // Pre-contract, the three parses disagreed here: the
                    // standard lowering took the LAST section, CPDT and the
                    // training report took the FIRST.
                    diags.push(
                        Diagnostic::error("duplicate optimizer: section")
                            .with_label(expr.span, "already specified above"),
                    );
                    continue;
                }
                optimizer_seen = true;
                optimizer = resolve_optimizer_expr(expr, resolve_sym, &mut diags);
            }
            TrainSection::Scheduler(expr) => {
                if scheduler_seen {
                    diags.push(
                        Diagnostic::error("duplicate scheduler: section")
                            .with_label(expr.span, "already specified above"),
                    );
                    continue;
                }
                scheduler_seen = true;
                scheduler_expr = Some(expr);
            }
            TrainSection::Callbacks(cbs) => {
                for cb in cbs {
                    let name = resolve_sym(cb.name);
                    if !CALLBACK_NAMES.contains(&name.as_str()) {
                        // Previously collected and silently never emitted —
                        // documented logging/eval logic just never ran.
                        diags.push(
                            Diagnostic::error(format!("unknown callback '{name}'"))
                                .with_label(cb.span, CALLBACK_NAMES_LABEL),
                        );
                        continue;
                    }
                    // Duplicate DEFINITIONS refuse like duplicate
                    // optimizer:/scheduler: sections do — codegen compiles
                    // every matching definition, so a user redefining
                    // on_step (expecting override) silently got both
                    // bodies run per step. Multiple callbacks: sections
                    // defining DISTINCT callbacks stay legal (union).
                    if !seen_callbacks.insert(name.clone()) {
                        diags.push(
                            Diagnostic::error(format!("duplicate callback '{name}'"))
                                .with_label(cb.span, "already defined above"),
                        );
                        continue;
                    }
                    // Param namespace: codegen binds params BY NAME with a
                    // silent zero fallback for anything unrecognized —
                    // `on_step(step, lss)` logged a constant 0 forever.
                    let accepted = callback_param_names(&name);
                    for param in &cb.params {
                        let pname = resolve_sym(param.name);
                        if !accepted.contains(&pname.as_str()) {
                            let joined = accepted.join(", ");
                            diags.push(
                                Diagnostic::error(format!(
                                    "unknown {name} callback parameter '{pname}'"
                                ))
                                .with_label(
                                    param.span,
                                    format!("{name} provides: {joined}"),
                                ),
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if !optimizer_seen {
        let block_kind = match purpose {
            TrainConfigPurpose::UserTrainBlock => "train",
            TrainConfigPurpose::DistillLowering => "distill",
        };
        diags.push(
            Diagnostic::error(format!(
                "{block_kind} block requires an optimizer section"
            ))
            .with_label(block_span, "add optimizer: <name>(lr=...), e.g. optimizer: AdamW(lr=0.001)"),
        );
    }

    // Scheduler resolution needs the optimizer's lr (one_cycle's max_lr
    // defaults to 10x it), so it runs second.
    let scheduler = scheduler_expr.and_then(|expr| {
        let base_lr = optimizer.as_ref().map(|o| o.lr).unwrap_or(0.01);
        resolve_scheduler_expr(expr, base_lr, resolve_sym, &mut diags)
    });

    if diags.is_empty() {
        Ok(ResolvedOptimConfig {
            // optimizer_seen guarantees Some on the empty-diags path except
            // when the section itself failed to resolve — which pushed a
            // diagnostic, so this branch is unreachable then.
            optimizer: optimizer.expect("optimizer resolved when no diagnostics"),
            scheduler,
        })
    } else {
        Err(diags)
    }
}

/// Numeric-value constraints, so the refusal text and the check cannot
/// drift apart per key.
#[derive(Clone, Copy)]
enum NumRule {
    /// v >= 0 and finite.
    NonNegative,
    /// v > 0 and finite.
    Positive,
    /// 0 <= v < 1 (bias-correction divides by 1 - beta^t; beta = 1 is a
    /// division by zero at t=1).
    Beta,
    /// 0 < v < 1.
    UnitOpen,
}

impl NumRule {
    fn admits(self, v: f64) -> bool {
        match self {
            NumRule::NonNegative => v >= 0.0 && v.is_finite(),
            NumRule::Positive => v > 0.0 && v.is_finite(),
            NumRule::Beta => (0.0..1.0).contains(&v),
            NumRule::UnitOpen => v > 0.0 && v < 1.0,
        }
    }

    fn expectation(self) -> &'static str {
        match self {
            NumRule::NonNegative => "a non-negative finite numeric literal",
            NumRule::Positive => "a positive finite numeric literal",
            NumRule::Beta => "a numeric literal in [0, 1)",
            NumRule::UnitOpen => "a numeric literal in (0, 1)",
        }
    }
}

fn resolve_optimizer_expr(
    expr: &nsl_ast::expr::Expr,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diags: &mut Vec<Diagnostic>,
) -> Option<ResolvedOptimizer> {
    let ExprKind::Call { callee, args } = &expr.kind else {
        // Previously a bare `optimizer: AdamW` passed `nsl check` and then
        // refused at build with the misleading "requires an optimizer
        // section" (the name never parsed, so the section looked absent).
        diags.push(
            Diagnostic::error(
                "optimizer must be a constructor call, e.g. AdamW(lr=0.001)",
            )
            .with_label(expr.span, OPTIMIZER_NAMES_LABEL),
        );
        return None;
    };
    let ExprKind::Ident(name_sym) = &callee.kind else {
        diags.push(
            Diagnostic::error(
                "optimizer must be a constructor call, e.g. AdamW(lr=0.001)",
            )
            .with_label(callee.span, OPTIMIZER_NAMES_LABEL),
        );
        return None;
    };
    let raw_name = resolve_sym(*name_sym);
    let Some(kind) = OptimizerKind::parse(&raw_name) else {
        // Previously refused only at codegen step-fn mangling; `nsl check`
        // was green.
        diags.push(
            Diagnostic::error(format!("unknown optimizer '{raw_name}'"))
                .with_label(callee.span, OPTIMIZER_NAMES_LABEL),
        );
        return None;
    };

    // Generic defaults — identical to the pre-contract codegen locals.
    let mut lr: f64 = 0.01;
    let mut momentum: f64 = 0.0;
    let mut dampening: f64 = 0.0;
    let mut weight_decay: f64 = 0.0;
    let mut nesterov = false;
    let mut beta1: f64 = 0.9;
    let mut beta2: f64 = 0.999;
    let mut eps: f64 = 1e-8;
    let mut ns_steps: f64 = 5.0;
    let mut adamw_lr: Option<f64> = None;
    let mut no_decay_static_roles: Vec<String> = Vec::new();
    let mut no_decay_exempt_non_rank2 = false;
    let mut lr_set = false;
    let mut momentum_set = false;
    let mut nesterov_set = false;

    let display = kind.display();
    let accepted = kind.accepted_kwargs();
    let accepted_label = || {
        let joined = accepted
            .iter()
            .map(|k| format!("{k}="))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{display} accepts {joined}")
    };

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for arg in args {
        let Some(kw_sym) = arg.name else {
            // Previously skipped before any name matched: `SGD(0.01)`
            // compiled and trained at the default lr.
            diags.push(
                Diagnostic::error("optimizer takes named arguments only")
                    .with_label(arg.span, "write key=value, e.g. lr=0.001"),
            );
            continue;
        };
        let key = resolve_sym(kw_sym);

        if !seen.insert(key.clone()) {
            // Previously last-wins (each recognized arm reassigned its
            // local as the loop ran).
            diags.push(
                Diagnostic::error(format!("duplicate optimizer kwarg '{key}'"))
                    .with_label(arg.span, "already specified"),
            );
            continue;
        }

        if !accepted.contains(&key.as_str()) {
            // Both the catch-all `_ => {}` typo case AND the
            // accepted-elsewhere case (`SGD(beta2=..)` parsed fine and was
            // simply never read by the emitted sgd_step call).
            diags.push(
                Diagnostic::error(format!("unknown {display} kwarg '{key}'"))
                    .with_label(arg.span, accepted_label()),
            );
            continue;
        }

        // A named numeric kwarg with a rule; nesterov / ns_steps / no_decay
        // / adamw_lr have bespoke shapes below. Returns whether `target` was
        // actually written — callers use this to gate their `_set` flag the
        // same way the `nesterov` arm below does, so a rule-failing value
        // (which leaves `target` at its prior default and pushes a
        // diagnostic) doesn't mark the knob as user-set and skip the Muon
        // spec-default backfill.
        let numeric = |target: &mut f64, rule: NumRule, diags: &mut Vec<Diagnostic>| -> bool {
            match fold_numeric_literal(&arg.value.kind) {
                Some(v) if rule.admits(v) => {
                    *target = v;
                    true
                }
                Some(v) => {
                    diags.push(
                        Diagnostic::error(format!(
                            "optimizer '{key}' must be {} (got {v})",
                            rule.expectation()
                        ))
                        .with_label(arg.span, format!("expected {}", rule.expectation())),
                    );
                    false
                }
                None => {
                    diags.push(
                        // Previously the silent half of the defect: a known key
                        // with a non-matching literal shape (int where float,
                        // any computed expr, a negative literal — which parses
                        // as UnaryOp(Neg, ..)) fell through its own match arm
                        // and kept the default.
                        Diagnostic::error(format!(
                            "optimizer '{key}' must be a numeric literal"
                        ))
                        .with_label(arg.span, format!("expected {}", rule.expectation())),
                    );
                    false
                }
            }
        };

        match key.as_str() {
            "lr" => {
                lr_set |= numeric(&mut lr, NumRule::NonNegative, diags);
            }
            "momentum" => {
                momentum_set |= numeric(&mut momentum, NumRule::NonNegative, diags);
            }
            "dampening" => {
                numeric(&mut dampening, NumRule::NonNegative, diags);
            }
            "weight_decay" => {
                numeric(&mut weight_decay, NumRule::NonNegative, diags);
            }
            "beta1" => {
                numeric(&mut beta1, NumRule::Beta, diags);
            }
            "beta2" => {
                numeric(&mut beta2, NumRule::Beta, diags);
            }
            "eps" => {
                numeric(&mut eps, NumRule::Positive, diags);
            }
            "adamw_lr" => {
                // A non-literal or non-positive adamw_lr silently falling
                // back to `lr` would train the embedding/head at Muon's
                // ~2e-2 — the exact divergence class the knob exists to
                // prevent.
                match fold_numeric_literal(&arg.value.kind) {
                    Some(v) if v > 0.0 && v.is_finite() => adamw_lr = Some(v),
                    Some(v) => diags.push(
                        Diagnostic::error(format!("adamw_lr must be positive (got {v})"))
                            .with_label(arg.span, "expected a positive numeric literal"),
                    ),
                    None => diags.push(
                        Diagnostic::error("adamw_lr must be a positive numeric literal")
                            .with_label(arg.span, "expected a positive numeric literal"),
                    ),
                }
            }
            "nesterov" => match fold_bool_literal(&arg.value.kind) {
                Some(b) => {
                    nesterov = b;
                    nesterov_set = true;
                }
                None => diags.push(
                    Diagnostic::error("optimizer 'nesterov' must be a bool literal")
                        .with_label(arg.span, "expected true or false"),
                ),
            },
            "ns_steps" => {
                // ns_steps is the Newton-Schulz iteration COUNT — a
                // positive integer by construction (spec/10:
                // `ns_steps: int = 5`). A float would silently truncate in
                // the stdlib `while i < ns_steps` loop and a non-positive
                // value would skip orthogonalization entirely (plain
                // momentum-SGD masquerading as Muon).
                match fold_int_literal(&arg.value.kind) {
                    Some(n) if n >= 1 => ns_steps = n as f64,
                    Some(n) => diags.push(
                        Diagnostic::error(format!(
                            "ns_steps must be a positive integer (got {n}); \
                             ns_steps=0 or negative would skip Newton-Schulz \
                             orthogonalization entirely"
                        ))
                        .with_label(arg.span, "expected an integer literal >= 1"),
                    ),
                    None => match fold_numeric_literal(&arg.value.kind) {
                        Some(f) => diags.push(
                            Diagnostic::error(format!(
                                "ns_steps must be a positive integer (got float {f}); \
                                 the Newton-Schulz iteration count cannot be fractional"
                            ))
                            .with_label(arg.span, "expected an integer literal >= 1"),
                        ),
                        None => diags.push(
                            Diagnostic::error(
                                "ns_steps must be a positive integer literal \
                                 (e.g. ns_steps = 5)",
                            )
                            .with_label(arg.span, "expected an integer literal >= 1"),
                        ),
                    },
                }
            }
            "no_decay" => {
                // Role names whose params are EXEMPT from weight decay. An
                // unknown name is a compile error, not a silently-dropped
                // entry — a typo'd "vectors" would otherwise read as
                // "exclusion configured" while decaying every norm in the
                // model. (Message text is pinned by
                // weight_decay_groups_gate.)
                let ExprKind::ListLiteral(items) = &arg.value.kind else {
                    diags.push(
                        Diagnostic::error(
                            "no_decay must be a list of role-name strings, \
                             e.g. no_decay=[\"vector\"] (norms/biases) or \
                             no_decay=[\"vector\", \"embedding\"]",
                        )
                        .with_label(arg.span, "expected a list of role-name strings"),
                    );
                    continue;
                };
                for item in items {
                    let ExprKind::StringLiteral(s) = &item.kind else {
                        diags.push(
                            Diagnostic::error(
                                "no_decay entries must be string literals \
                                 naming a parameter role",
                            )
                            .with_label(item.span, "expected a string literal role name"),
                        );
                        continue;
                    };
                    if !VALID_ROLES.contains(&s.as_str()) {
                        diags.push(
                            Diagnostic::error(format!(
                                "no_decay: unknown parameter role {s:?}. \
                                 Valid roles are {VALID_ROLES:?}. \"vector\" covers \
                                 anything not rank-2 at runtime (norms, \
                                 biases); \"embedding\"/\"head\" come from \
                                 @param_role or embedding_lookup usage"
                            ))
                            .with_label(item.span, "unknown parameter role"),
                        );
                        continue;
                    }
                    if s == "vector" {
                        no_decay_exempt_non_rank2 = true;
                    } else if !no_decay_static_roles.iter().any(|r| r == s) {
                        no_decay_static_roles.push(s.clone());
                    }
                }
                // Every param's role is exactly one of embedding/head/
                // hidden/vector, so exempting the three non-vector roles
                // exempts everything (a "vector"-role param needs a
                // statically-derivable rank, which real models do not
                // have). That is weight_decay=0.0 spelled obscurely.
                if ["embedding", "head", "hidden"]
                    .iter()
                    .all(|r| no_decay_static_roles.iter().any(|have| have == r))
                {
                    diags.push(
                        Diagnostic::error(
                            "no_decay exempts every role that a parameter \
                             can have (embedding, head, hidden), which \
                             disables weight decay entirely — set \
                             weight_decay=0.0 instead so the intent is \
                             visible at the call site",
                        )
                        .with_label(arg.span, "disables weight decay entirely"),
                    );
                }
            }
            // Unreachable: the accepted-set gate above filters everything
            // else, and every accepted key has an arm.
            other => unreachable!("accepted optimizer kwarg '{other}' has no arm"),
        }
    }

    // Muon spec defaults for knobs the user left unset — Muon(lr=0.02,
    // momentum=0.95, nesterov=true) per spec/10. The generic defaults
    // (lr=0.01, momentum=0.0, nesterov=false) would silently degrade a
    // bare Muon() to orthogonalized momentum-free GD. Lives HERE so the
    // pipelined path (which previously had no backfill) agrees.
    if kind == OptimizerKind::Muon {
        if !lr_set {
            lr = 0.02;
        }
        if !momentum_set {
            momentum = 0.95;
        }
        if !nesterov_set {
            nesterov = true;
        }
        // adamw_lr is threaded as the ratio adamw_lr/lr of the scheduled
        // lr — lr <= 0 would make that inf/NaN at every AdamW-routed
        // update. (Judged AFTER the backfill, so adamw_lr with omitted lr
        // stays legal: lr defaults to 0.02 first.)
        if adamw_lr.is_some() && lr <= 0.0 {
            diags.push(
                Diagnostic::error(format!(
                    "Muon adamw_lr requires a positive lr (got lr={lr}): \
                     the AdamW arm's rate is applied as the fixed ratio \
                     adamw_lr/lr of the scheduled lr"
                ))
                .with_label(expr.span, "set lr > 0 or drop adamw_lr"),
            );
        }
    }

    Some(ResolvedOptimizer {
        kind,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        beta1,
        beta2,
        eps,
        ns_steps,
        adamw_lr,
        no_decay_static_roles,
        no_decay_exempt_non_rank2,
    })
}

/// Per-scheduler kwarg spec: (key, default, rule). Defaults are the values
/// the pre-contract lowering hardcoded at each `.unwrap_or(..)` site —
/// they live here now so the lowering reads resolved values only.
fn scheduler_kwarg_spec(canonical: &str) -> &'static [(&'static str, f64, NumRule)] {
    match canonical {
        "constant_lr" => &[],
        "step_lr" => &[
            ("step_size", 10.0, NumRule::Positive),
            ("gamma", 0.1, NumRule::Positive),
        ],
        "exponential_lr" => &[("gamma", 0.95, NumRule::Positive)],
        "linear_decay" => &[
            ("total_steps", 1000.0, NumRule::Positive),
            ("end_factor", 0.0, NumRule::NonNegative),
        ],
        "cosine_anneal" => &[
            ("t_max", 1000.0, NumRule::Positive),
            ("eta_min", 0.0, NumRule::NonNegative),
        ],
        // NOTE: warmup_steps > total_steps is deliberately legal — the
        // coder-rl 12-step repro fixtures train entirely inside the warmup
        // ramp (warmup_steps=20, total_steps=12).
        "warmup_cosine" => &[
            ("warmup_steps", 100.0, NumRule::NonNegative),
            ("total_steps", 1000.0, NumRule::Positive),
            ("min_lr", 1e-5, NumRule::NonNegative),
        ],
        // max_lr's real default is 10x the optimizer lr — patched by the
        // caller; the sentinel here is never read.
        "one_cycle" => &[
            ("max_lr", f64::NAN, NumRule::Positive),
            ("total_steps", 1000.0, NumRule::Positive),
            ("pct_start", 0.3, NumRule::UnitOpen),
        ],
        other => unreachable!("no kwarg spec for scheduler '{other}'"),
    }
}

fn resolve_scheduler_expr(
    expr: &nsl_ast::expr::Expr,
    base_lr: f64,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diags: &mut Vec<Diagnostic>,
) -> Option<ResolvedScheduler> {
    let ExprKind::Call { callee, args } = &expr.kind else {
        // A bare `scheduler: warmup_cosine` previously left the whole
        // section inert (name never parsed, lowering gate saw it empty).
        diags.push(
            Diagnostic::error(
                "scheduler must be a constructor call, e.g. \
                 warmup_cosine(warmup_steps=100, total_steps=1000)",
            )
            .with_label(expr.span, SCHEDULER_NAMES_LABEL),
        );
        return None;
    };
    let ExprKind::Ident(name_sym) = &callee.kind else {
        diags.push(
            Diagnostic::error(
                "scheduler must be a constructor call, e.g. \
                 warmup_cosine(warmup_steps=100, total_steps=1000)",
            )
            .with_label(callee.span, SCHEDULER_NAMES_LABEL),
        );
        return None;
    };
    let raw_name = resolve_sym(*name_sym);
    let Some(canonical) = canonical_scheduler_name(&raw_name) else {
        // Previously fell through the lowering's name map to
        // `_ => base_lr_val, // fallback: no change` — the program
        // compiled and silently trained at constant lr.
        diags.push(
            Diagnostic::error(format!("unknown scheduler '{raw_name}'"))
                .with_label(callee.span, SCHEDULER_NAMES_LABEL),
        );
        return None;
    };

    let spec = scheduler_kwarg_spec(canonical);
    let accepted_label = || {
        if spec.is_empty() {
            format!("{canonical} takes no kwargs")
        } else {
            let joined = spec
                .iter()
                .map(|(k, _, _)| format!("{k}="))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{canonical} accepts {joined}")
        }
    };

    let mut values: Vec<(&'static str, f64)> = spec
        .iter()
        .map(|(k, default, _)| (*k, *default))
        .collect();
    // one_cycle's max_lr defaults to 10x the optimizer lr.
    if canonical == "one_cycle" {
        values[0].1 = base_lr * 10.0;
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut max_lr_written = false;

    for arg in args {
        let Some(kw_sym) = arg.name else {
            diags.push(
                Diagnostic::error("scheduler takes named arguments only")
                    .with_label(arg.span, "write key=value, e.g. total_steps=1000"),
            );
            continue;
        };
        let key = resolve_sym(kw_sym);

        if !seen.insert(key.clone()) {
            // Previously FIRST-wins (args were appended to a Vec and read
            // via .find()) — the opposite of the optimizer's last-wins.
            diags.push(
                Diagnostic::error(format!("duplicate scheduler kwarg '{key}'"))
                    .with_label(arg.span, "already specified"),
            );
            continue;
        }

        let Some((_, rule)) = spec
            .iter()
            .find(|(k, _, _)| *k == key.as_str())
            .map(|(k, _, r)| (*k, *r))
        else {
            // Previously pushed into scheduler_args, never matched by any
            // `.find()`, and the hardcoded default silently applied.
            diags.push(
                Diagnostic::error(format!("unknown {canonical} kwarg '{key}'"))
                    .with_label(arg.span, accepted_label()),
            );
            continue;
        };

        match fold_numeric_literal(&arg.value.kind) {
            Some(v) if rule.admits(v) => {
                if let Some(slot) = values.iter_mut().find(|(k, _)| *k == key.as_str()) {
                    slot.1 = v;
                }
                if key == "max_lr" {
                    max_lr_written = true;
                }
            }
            Some(v) => diags.push(
                Diagnostic::error(format!(
                    "scheduler '{key}' must be {} (got {v})",
                    rule.expectation()
                ))
                .with_label(arg.span, format!("expected {}", rule.expectation())),
            ),
            None => diags.push(
                // Previously silently dropped (only Float/Int literals were
                // collected) — `total_steps = CONFIG.total_steps` compiled
                // against the hardcoded default.
                Diagnostic::error(format!("scheduler '{key}' must be a numeric literal"))
                    .with_label(arg.span, format!("expected {}", rule.expectation())),
            ),
        }
    }

    // one_cycle's derived default can be invalid even when every written
    // kwarg is valid (lr=0 → max_lr defaults to 0).
    if canonical == "one_cycle" && !max_lr_written {
        let max_lr = values[0].1;
        if !(max_lr > 0.0 && max_lr.is_finite()) {
            diags.push(
                Diagnostic::error(format!(
                    "one_cycle max_lr defaults to 10x the optimizer lr, which \
                     here is {max_lr} — write max_lr= explicitly"
                ))
                .with_label(expr.span, "expected a positive finite max_lr"),
            );
            return None;
        }
    }

    let get = |k: &str| -> f64 {
        values
            .iter()
            .find(|(key, _)| *key == k)
            .map(|(_, v)| *v)
            .expect("spec key present")
    };

    Some(match canonical {
        "constant_lr" => ResolvedScheduler::ConstantLr,
        "step_lr" => ResolvedScheduler::StepLr {
            step_size: get("step_size"),
            gamma: get("gamma"),
        },
        "exponential_lr" => ResolvedScheduler::ExponentialLr { gamma: get("gamma") },
        "linear_decay" => ResolvedScheduler::LinearDecay {
            total_steps: get("total_steps"),
            end_factor: get("end_factor"),
        },
        "cosine_anneal" => ResolvedScheduler::CosineAnneal {
            t_max: get("t_max"),
            eta_min: get("eta_min"),
        },
        "warmup_cosine" => ResolvedScheduler::WarmupCosine {
            warmup_steps: get("warmup_steps"),
            total_steps: get("total_steps"),
            min_lr: get("min_lr"),
        },
        "one_cycle" => ResolvedScheduler::OneCycle {
            max_lr: get("max_lr"),
            total_steps: get("total_steps"),
            pct_start: get("pct_start"),
        },
        other => unreachable!("canonical scheduler '{other}' not constructed"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every optimizer's accepted-kwargs table must contain lr — and every
    /// key in any table must have a resolver arm. This test ACTUALLY
    /// resolves a call carrying each table key with a valid value, so a
    /// deleted/renamed match arm cannot hide behind the table: the
    /// `unreachable!` in resolve_optimizer_expr would abort the test.
    /// (The first version of this test compared the table against a
    /// hardcoded mirror list of arm names — a guard that guarded nothing;
    /// review finding.)
    #[test]
    fn every_accepted_kwarg_has_a_resolver_arm() {
        use OptimizerKind::*;
        for kind in [Sgd, Adam, AdamW, Lion, Muon, Soap] {
            assert!(
                kind.accepted_kwargs().contains(&"lr"),
                "{} table is missing lr",
                kind.as_str()
            );
            for key in kind.accepted_kwargs() {
                let mut interner = nsl_lexer::Interner::new();
                let value = match *key {
                    "nesterov" => mk_expr(ExprKind::BoolLiteral(true)),
                    "ns_steps" => mk_expr(ExprKind::IntLiteral(5)),
                    "no_decay" => mk_expr(ExprKind::ListLiteral(vec![mk_expr(
                        ExprKind::StringLiteral("embedding".to_string()),
                    )])),
                    "beta1" | "beta2" => mk_expr(ExprKind::FloatLiteral(0.9)),
                    _ => mk_expr(ExprKind::FloatLiteral(0.001)),
                };
                let kwarg = mk_kwarg(&mut interner, key, value);
                let call = mk_call(&mut interner, kind.display(), vec![kwarg]);
                let result = resolve(&interner, &[TrainSection::Optimizer(call)]);
                assert!(
                    result.is_ok(),
                    "{}({key}=<valid>) must resolve cleanly, got {:?}",
                    kind.as_str(),
                    result.err().map(|ds| ds
                        .into_iter()
                        .map(|d| d.message)
                        .collect::<Vec<_>>())
                );
            }
        }
    }

    /// OPTIMIZER_NAMES_LABEL ↔ OptimizerKind::parse drift guard — pinned
    /// in BOTH directions by constructing the label from the display names
    /// (a stale extra name in the label fails assert_eq, not just a
    /// missing one; review finding on the one-directional version).
    #[test]
    fn the_optimizer_names_label_lists_exactly_the_accepted_names() {
        use OptimizerKind::*;
        let kinds = [Sgd, Adam, AdamW, Lion, Muon, Soap];
        let names: Vec<&str> = kinds.iter().map(|k| k.display()).collect();
        let (last, init) = names.split_last().unwrap();
        let expected = format!("expected {}, or {last}", init.join(", "));
        assert_eq!(
            OPTIMIZER_NAMES_LABEL, expected,
            "OPTIMIZER_NAMES_LABEL drifted from the accepted set"
        );
        for name in names {
            assert!(
                OptimizerKind::parse(name).is_some(),
                "advertised optimizer '{name}' does not parse"
            );
        }
    }

    /// SCHEDULER_NAMES_LABEL ↔ canonical_scheduler_name drift guard —
    /// both directions via assert_eq on the constructed label — and every
    /// canonical name must have a kwarg spec (the unreachable! in
    /// scheduler_kwarg_spec fires otherwise).
    #[test]
    fn the_scheduler_names_label_lists_exactly_the_accepted_names() {
        let names = [
            "constant_lr",
            "step_lr",
            "exponential_lr",
            "linear_decay",
            "cosine_anneal",
            "warmup_cosine",
            "one_cycle",
        ];
        let (last, init) = names.split_last().unwrap();
        let expected = format!("expected {}, or {last}", init.join(", "));
        // The label wraps across two source lines; compare whitespace-
        // normalized.
        let normalized = SCHEDULER_NAMES_LABEL.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            normalized, expected,
            "SCHEDULER_NAMES_LABEL drifted from the accepted set"
        );
        for name in names {
            assert_eq!(canonical_scheduler_name(name), Some(name));
            // CamelCase-collapsed spelling accepted too.
            let collapsed = name.replace('_', "");
            assert_eq!(canonical_scheduler_name(&collapsed), Some(name));
            let _ = scheduler_kwarg_spec(name);
        }
    }

    /// CALLBACK_NAMES ↔ CALLBACK_NAMES_LABEL drift guard, both directions.
    /// (A substring check was vacuous for "on_epoch" — it is a prefix of
    /// "on_epoch_end"; review finding.)
    #[test]
    fn the_callback_names_label_lists_exactly_the_accepted_names() {
        let (last, init) = CALLBACK_NAMES.split_last().unwrap();
        let expected = format!("expected {}, or {last}", init.join(", "));
        assert_eq!(
            CALLBACK_NAMES_LABEL, expected,
            "CALLBACK_NAMES_LABEL drifted from the accepted set"
        );
        // Every accepted callback has a param table (the unreachable! in
        // callback_param_names fires otherwise).
        for name in CALLBACK_NAMES {
            assert!(!callback_param_names(name).is_empty());
        }
    }

    // ── Value-level behavior (AST fixtures built directly) ──────────────

    use nsl_ast::expr::{Arg, Expr};
    use nsl_ast::NodeId;

    fn mk_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span::dummy(),
            id: NodeId::next(),
        }
    }

    fn mk_call(interner: &mut nsl_lexer::Interner, name: &str, args: Vec<Arg>) -> Expr {
        let sym = Symbol(interner.get_or_intern(name));
        mk_expr(ExprKind::Call {
            callee: Box::new(mk_expr(ExprKind::Ident(sym))),
            args,
        })
    }

    fn mk_kwarg(interner: &mut nsl_lexer::Interner, name: &str, value: Expr) -> Arg {
        Arg {
            name: Some(Symbol(interner.get_or_intern(name))),
            value,
            span: Span::dummy(),
        }
    }

    fn resolve(
        interner: &nsl_lexer::Interner,
        sections: &[TrainSection],
    ) -> Result<ResolvedOptimConfig, Vec<Diagnostic>> {
        resolve_optim_config(
            sections,
            Span::dummy(),
            &|sym| interner.resolve(sym.0).unwrap_or("<unknown>").to_string(),
            TrainConfigPurpose::UserTrainBlock,
        )
    }

    /// Bare `Muon()` gets the spec defaults (lr=0.02, momentum=0.95,
    /// nesterov=true) — the backfill that previously lived only on the
    /// standard lowering path, so bare Muon under @pipeline silently
    /// degraded to momentum-free lr=0.01.
    #[test]
    fn bare_muon_gets_spec_defaults() {
        let mut interner = nsl_lexer::Interner::new();
        let call = mk_call(&mut interner, "Muon", vec![]);
        let cfg = resolve(&interner, &[TrainSection::Optimizer(call)]).expect("valid");
        assert_eq!(cfg.optimizer.kind, OptimizerKind::Muon);
        assert!((cfg.optimizer.lr - 0.02).abs() < 1e-12, "lr = {}", cfg.optimizer.lr);
        assert!((cfg.optimizer.momentum - 0.95).abs() < 1e-12);
        assert!(cfg.optimizer.nesterov);
        // Written knobs are NOT backfilled.
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.005)));
        let call = mk_call(&mut interner, "Muon", vec![lr]);
        let cfg = resolve(&interner, &[TrainSection::Optimizer(call)]).expect("valid");
        assert!((cfg.optimizer.lr - 0.005).abs() < 1e-12);
        assert!((cfg.optimizer.momentum - 0.95).abs() < 1e-12, "unset knob still backfills");
    }

    /// A rule-failing `lr`/`momentum` value must NOT count as "user set" —
    /// otherwise the Muon spec-default backfill is skipped and the knob is
    /// left at the wrong (generic) default instead of the Muon one. Calls
    /// `resolve_optimizer_expr` directly (not the `resolve_optim_config`
    /// wrapper, which discards the optimizer entirely once any diagnostic
    /// is pushed) so the backfilled-but-otherwise-discarded value is
    /// observable. Review finding: `lr_set`/`momentum_set` were set
    /// unconditionally after the `numeric()` call, unlike `nesterov_set`
    /// which is only set on the success arm.
    #[test]
    fn muon_invalid_momentum_still_backfills_spec_default() {
        let mut interner = nsl_lexer::Interner::new();
        let momentum = mk_kwarg(
            &mut interner,
            "momentum",
            mk_expr(ExprKind::UnaryOp {
                op: nsl_ast::operator::UnaryOp::Neg,
                operand: Box::new(mk_expr(ExprKind::FloatLiteral(1.0))),
            }),
        );
        let call = mk_call(&mut interner, "Muon", vec![momentum]);
        let mut diags = Vec::new();
        let resolve_sym = |sym: Symbol| interner.resolve(sym.0).unwrap_or("<unknown>").to_string();
        let resolved = resolve_optimizer_expr(&call, &resolve_sym, &mut diags)
            .expect("resolve_optimizer_expr always returns Some past the name-parse stage");
        assert!(
            !diags.is_empty(),
            "momentum = -1 must be rejected by the NonNegative rule"
        );
        assert!(
            (resolved.momentum - 0.95).abs() < 1e-12,
            "an invalid momentum must not suppress the Muon spec-default \
             backfill (got {}, want 0.95)",
            resolved.momentum
        );
    }

    /// Non-Muon optimizers keep the generic defaults.
    #[test]
    fn non_muon_keeps_generic_defaults() {
        let mut interner = nsl_lexer::Interner::new();
        let call = mk_call(&mut interner, "SGD", vec![]);
        let cfg = resolve(&interner, &[TrainSection::Optimizer(call)]).expect("valid");
        assert!((cfg.optimizer.lr - 0.01).abs() < 1e-12);
        assert!((cfg.optimizer.momentum - 0.0).abs() < 1e-12);
        assert!(!cfg.optimizer.nesterov);
    }

    /// Muon(lr=0, adamw_lr=..) refuses: adamw_lr is threaded as the ratio
    /// adamw_lr/lr. Muon(adamw_lr=..) with lr omitted stays legal — the
    /// spec-default lr=0.02 is applied first (pre-contract behavior).
    #[test]
    fn muon_adamw_lr_requires_positive_lr_after_backfill() {
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.0)));
        let awl = mk_kwarg(&mut interner, "adamw_lr", mk_expr(ExprKind::FloatLiteral(0.001)));
        let call = mk_call(&mut interner, "Muon", vec![lr, awl]);
        let errs = resolve(&interner, &[TrainSection::Optimizer(call)]).unwrap_err();
        assert!(
            errs.iter().any(|d| d.message.contains("Muon adamw_lr requires a positive lr")),
            "got {:?}",
            errs.iter().map(|d| &d.message).collect::<Vec<_>>()
        );

        let mut interner = nsl_lexer::Interner::new();
        let awl = mk_kwarg(&mut interner, "adamw_lr", mk_expr(ExprKind::FloatLiteral(0.001)));
        let call = mk_call(&mut interner, "Muon", vec![awl]);
        let cfg = resolve(&interner, &[TrainSection::Optimizer(call)])
            .expect("adamw_lr with omitted lr is legal (backfilled lr=0.02)");
        assert_eq!(cfg.optimizer.adamw_lr, Some(0.001));
    }

    /// SGD(lr = 0.0) stays legal — the WRGA equivalence gates freeze
    /// weights with a zero lr on purpose.
    #[test]
    fn zero_lr_is_legal() {
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.0)));
        let call = mk_call(&mut interner, "SGD", vec![lr]);
        let cfg = resolve(&interner, &[TrainSection::Optimizer(call)]).expect("valid");
        assert_eq!(cfg.optimizer.lr, 0.0);
    }

    /// Scheduler defaults are the values the pre-contract lowering
    /// hardcoded at its `.unwrap_or(..)` sites; one_cycle's max_lr derives
    /// from the optimizer lr.
    #[test]
    fn scheduler_defaults_match_the_pre_contract_lowering() {
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.002)));
        let opt = mk_call(&mut interner, "AdamW", vec![lr]);
        let sched = mk_call(&mut interner, "warmup_cosine", vec![]);
        let cfg = resolve(
            &interner,
            &[TrainSection::Optimizer(opt), TrainSection::Scheduler(sched)],
        )
        .expect("valid");
        assert_eq!(
            cfg.scheduler,
            Some(ResolvedScheduler::WarmupCosine {
                warmup_steps: 100.0,
                total_steps: 1000.0,
                min_lr: 1e-5,
            })
        );

        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.002)));
        let opt = mk_call(&mut interner, "AdamW", vec![lr]);
        let sched = mk_call(&mut interner, "one_cycle", vec![]);
        let cfg = resolve(
            &interner,
            &[TrainSection::Optimizer(opt), TrainSection::Scheduler(sched)],
        )
        .expect("valid");
        let Some(ResolvedScheduler::OneCycle { max_lr, total_steps, pct_start }) = cfg.scheduler
        else {
            panic!("expected OneCycle");
        };
        assert!((max_lr - 0.02).abs() < 1e-12, "max_lr defaults to 10x lr, got {max_lr}");
        assert!((total_steps - 1000.0).abs() < 1e-12);
        assert!((pct_start - 0.3).abs() < 1e-12);
    }

    /// one_cycle + lr=0: the derived max_lr default (10x lr = 0) is
    /// invalid — refused with an actionable message instead of lowering an
    /// all-zero LR curve.
    #[test]
    fn one_cycle_with_zero_lr_refuses_the_derived_default() {
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.0)));
        let opt = mk_call(&mut interner, "AdamW", vec![lr]);
        let sched = mk_call(&mut interner, "one_cycle", vec![]);
        let errs = resolve(
            &interner,
            &[TrainSection::Optimizer(opt), TrainSection::Scheduler(sched)],
        )
        .unwrap_err();
        assert!(
            errs.iter().any(|d| d.message.contains("write max_lr= explicitly")),
            "got {:?}",
            errs.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// Int literals coerce to float everywhere a numeric is expected —
    /// every committed scheduler call passes int steps.
    #[test]
    fn int_literals_coerce_for_numeric_kwargs() {
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(&mut interner, "lr", mk_expr(ExprKind::FloatLiteral(0.001)));
        let opt = mk_call(&mut interner, "AdamW", vec![lr]);
        let ws = mk_kwarg(&mut interner, "warmup_steps", mk_expr(ExprKind::IntLiteral(20)));
        let ts = mk_kwarg(&mut interner, "total_steps", mk_expr(ExprKind::IntLiteral(12)));
        let sched = mk_call(&mut interner, "warmup_cosine", vec![ws, ts]);
        // warmup_steps > total_steps is deliberately legal (the coder-rl
        // 12-step repro fixtures train entirely inside the warmup ramp).
        let cfg = resolve(
            &interner,
            &[TrainSection::Optimizer(opt), TrainSection::Scheduler(sched)],
        )
        .expect("int kwargs must coerce");
        assert_eq!(
            cfg.scheduler,
            Some(ResolvedScheduler::WarmupCosine {
                warmup_steps: 20.0,
                total_steps: 12.0,
                min_lr: 1e-5,
            })
        );
    }

    /// Optimizer-side int coercion is the BEHAVIOR CHANGE half of the
    /// coercion invariant: pre-contract, `SGD(momentum = 1)` (an
    /// IntLiteral against a FloatLiteral-only arm) was silently dropped
    /// and trained with momentum 0.0. It now resolves to 1.0.
    #[test]
    fn optimizer_int_literals_coerce_instead_of_silently_dropping() {
        let mut interner = nsl_lexer::Interner::new();
        let m = mk_kwarg(&mut interner, "momentum", mk_expr(ExprKind::IntLiteral(1)));
        let call = mk_call(&mut interner, "SGD", vec![m]);
        let cfg = resolve(&interner, &[TrainSection::Optimizer(call)])
            .expect("int momentum must coerce");
        assert!(
            (cfg.optimizer.momentum - 1.0).abs() < 1e-12,
            "momentum = {}",
            cfg.optimizer.momentum
        );
    }

    /// An infinite literal (1e400 lexes to +inf) is refused by the
    /// finiteness rules, not stored.
    #[test]
    fn infinite_values_are_refused() {
        let mut interner = nsl_lexer::Interner::new();
        let lr = mk_kwarg(
            &mut interner,
            "lr",
            mk_expr(ExprKind::FloatLiteral(f64::INFINITY)),
        );
        let call = mk_call(&mut interner, "AdamW", vec![lr]);
        let errs = resolve(&interner, &[TrainSection::Optimizer(call)]).unwrap_err();
        assert!(
            errs.iter().any(|d| d.message.contains("optimizer 'lr' must be")),
            "got {:?}",
            errs.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }
}
