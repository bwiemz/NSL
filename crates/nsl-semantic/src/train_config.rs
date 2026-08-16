//! The `train(...)` configuration contract: the config-key namespace, CLOSED.
//!
//! Before this module, the two layers disagreed about the same header:
//! the semantic checker type-checked config VALUES but never read a key
//! name (`epochz=8` passed `nsl check` clean), and codegen matched seven
//! known keys with `_ => {} // ignore unknown config for forward compat` —
//! so a typo silently trained with defaults. A non-literal
//! `grad_accumulation` was silently clamped to 1; a non-literal `grad_clip`
//! was dropped with no record at all; `epochs = 0` compiled and trained
//! zero epochs; duplicate keys resolved last-wins; positional args were
//! skipped before the key match even ran. Ignored configuration is not a
//! compatibility strategy — it trains a different model than the source
//! describes (the same defect class as the Milestone B dropout-default
//! bug, and the asymmetry `check_distill_block` already refused for
//! distill: "the asymmetry was the bug, not the strictness").
//!
//! ONE resolver, TWO callers. `resolve_train_config` is a pure function:
//! the semantic checker calls it (span-labelled diagnostics on
//! `nsl check` and every build), and codegen calls it again as the
//! backstop and single source of truth for the values it lowers
//! (`compile_train_block_inner` / the pipelined twin). This is the
//! `cftp::validate_fase_decorator` pattern — validated here, invoked from
//! both layers — chosen over the `DATA_SECTION_KEYS` pattern of two
//! comment-synced tables precisely because the tables drift.
//!
//! Forward compatibility is refusal, not silence: a program using a key
//! this compiler does not recognize fails with the key named, which is
//! safer than pretending to support syntax from a newer NSL.

use nsl_ast::block::TrainBlock;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// The complete accepted key set. A key added here must be consumed by
/// codegen in the same change — an accepted-but-ignored key is exactly
/// what this module exists to prevent.
pub const TRAIN_CONFIG_KEYS: &[&str] = &[
    "model",
    "epochs",
    "grad_accumulation",
    "grad_clip",
    "checkpoint_save",
    "checkpoint_every",
    "checkpoint_load",
];

const EXPECTED_KEYS_LABEL: &str = "expected model=, epochs=, grad_accumulation=, \
     grad_clip=, checkpoint_save=, checkpoint_every=, or checkpoint_load=";

/// A fully validated train header. Produced once; every consumer reads
/// these fields instead of re-scanning `TrainBlock.config`.
#[derive(Debug, Clone)]
pub struct ResolvedTrainConfig {
    /// `None` only under [`TrainConfigPurpose::DistillLowering`] — a
    /// user-written train block without `model=` is refused.
    pub model: Option<Symbol>,
    /// >= 1 (validated).
    pub epochs: i64,
    /// >= 1 (validated). 1 when omitted.
    pub grad_accumulation: i64,
    /// Whether `grad_accumulation=` was written at all — drives the
    /// wgrad-fusion decline diagnostic's wording, nothing else.
    pub grad_accumulation_explicit: bool,
    /// `Some(v)` with v > 0 when `grad_clip=` was written. 0 would zero
    /// every gradient each step (training silently frozen) and a negative
    /// value flips gradient signs — both refused, not stored.
    pub grad_clip: Option<f64>,
    pub checkpoint_save: Option<String>,
    /// Optimizer steps between checkpoints; 0 = checkpointing off (only
    /// when `checkpoint_save` is absent — the pairing is validated).
    pub checkpoint_every: i64,
    pub checkpoint_load: Option<String>,
}

/// Who is asking. Distill lowering synthesizes a `TrainBlock` whose model
/// and epochs travel as typed `DistillAsTrain` fields (cpkd.rs documents
/// why: a second copy in config is "a divergence waiting for one side to
/// be edited"), so `model=` is legitimately absent there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainConfigPurpose {
    UserTrainBlock,
    DistillLowering,
}

/// Fold an integer literal INCLUDING a leading unary minus: `-3` parses as
/// `UnaryOp(Neg, IntLiteral(3))`, and telling its author "-3 must be an
/// integer literal" reads as nonsense — the range message is the right one.
fn fold_int_literal(kind: &ExprKind) -> Option<i64> {
    match kind {
        ExprKind::IntLiteral(n) => Some(*n),
        ExprKind::UnaryOp {
            op: nsl_ast::operator::UnaryOp::Neg,
            operand,
        } => match &operand.kind {
            ExprKind::IntLiteral(n) => Some(-*n),
            _ => None,
        },
        _ => None,
    }
}

/// Same folding for a numeric (float-or-int) literal.
fn fold_numeric_literal(kind: &ExprKind) -> Option<f64> {
    match kind {
        ExprKind::FloatLiteral(f) => Some(*f),
        ExprKind::IntLiteral(n) => Some(*n as f64),
        ExprKind::UnaryOp {
            op: nsl_ast::operator::UnaryOp::Neg,
            operand,
        } => match &operand.kind {
            ExprKind::FloatLiteral(f) => Some(-*f),
            ExprKind::IntLiteral(n) => Some(-(*n as f64)),
            _ => None,
        },
        _ => None,
    }
}

/// Validate the config args of a train block against the closed key set
/// and produce the typed config. All violations are collected (not
/// first-error-wins) so one `nsl check` shows every problem in the header.
pub fn resolve_train_config(
    train: &TrainBlock,
    resolve_sym: &dyn Fn(Symbol) -> String,
    purpose: TrainConfigPurpose,
) -> Result<ResolvedTrainConfig, Vec<Diagnostic>> {
    let mut diags: Vec<Diagnostic> = Vec::new();

    let mut model: Option<Symbol> = None;
    let mut epochs: i64 = 1;
    let mut grad_accumulation: i64 = 1;
    let mut grad_accumulation_explicit = false;
    let mut grad_clip: Option<f64> = None;
    let mut checkpoint_save: Option<String> = None;
    let mut checkpoint_every: i64 = 0;
    let mut checkpoint_load: Option<String> = None;

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Spans for the pairing diagnostics below, so the label lands on the
    // key that needs the counterpart rather than on the whole header.
    let mut save_span = None;
    let mut every_span = None;

    for arg in &train.config {
        let Some(name_sym) = arg.name else {
            // The old parsers guarded `if let Some(name) = arg.name` and
            // silently skipped positional args — `train(m, 5):` compiled
            // and ignored both.
            diags.push(
                Diagnostic::error("train config takes named arguments only")
                    .with_label(arg.span, "write key=value, e.g. model=m or epochs=5"),
            );
            continue;
        };
        let name = resolve_sym(name_sym);

        if !seen.insert(name.clone()) {
            // The old codegen loop reassigned locals: last occurrence
            // silently won.
            diags.push(
                Diagnostic::error(format!("duplicate train config key '{name}'"))
                    .with_label(arg.span, "already specified"),
            );
            continue;
        }

        match name.as_str() {
            "model" => match &arg.value.kind {
                ExprKind::Ident(sym) => model = Some(*sym),
                _ => diags.push(
                    Diagnostic::error("train 'model' arg must be an identifier")
                        .with_label(arg.span, "expected a model variable, e.g. model=m"),
                ),
            },
            "epochs" => match fold_int_literal(&arg.value.kind) {
                Some(n) if n >= 1 => epochs = n,
                Some(n) => diags.push(
                    Diagnostic::error(format!(
                        "train 'epochs' must be >= 1 (got {n}) — the epoch loop \
                         would run zero iterations and silently train nothing"
                    ))
                    .with_label(arg.span, "expected an integer literal >= 1"),
                ),
                None => diags.push(
                    Diagnostic::error("train 'epochs' must be an integer literal")
                        .with_label(arg.span, "expected an integer literal >= 1"),
                ),
            },
            "grad_accumulation" => {
                grad_accumulation_explicit = true;
                match fold_int_literal(&arg.value.kind) {
                    Some(n) if n >= 1 => grad_accumulation = n,
                    _ => diags.push(
                        // The old behavior silently clamped non-literals to 1
                        // — accumulation OFF while the source plainly asked
                        // for it. Same contract as distill's (which always
                        // refused; the asymmetry was the bug).
                        Diagnostic::error(
                            "train 'grad_accumulation' must be a positive \
                             integer literal",
                        )
                        .with_label(arg.span, "expected an integer literal >= 1"),
                    ),
                }
            }
            "grad_clip" => {
                let lit = fold_numeric_literal(&arg.value.kind);
                match lit {
                    Some(v) if v > 0.0 => grad_clip = Some(v),
                    Some(v) => diags.push(
                        Diagnostic::error(format!(
                            "train 'grad_clip' must be > 0 (got {v}) — the \
                             runtime scales gradients by max_norm/norm, so 0 \
                             zeroes every gradient each step and a negative \
                             value flips their signs"
                        ))
                        .with_label(arg.span, "expected a positive numeric literal"),
                    ),
                    None => diags.push(
                        // Previously the ONLY key with a fully silent bad-value
                        // path: no else arm at all.
                        Diagnostic::error(
                            "train 'grad_clip' must be a numeric literal",
                        )
                        .with_label(arg.span, "expected a positive numeric literal"),
                    ),
                }
            }
            "checkpoint_save" => match &arg.value.kind {
                ExprKind::StringLiteral(s) if !s.is_empty() => {
                    checkpoint_save = Some(s.clone());
                    save_span = Some(arg.span);
                }
                _ => diags.push(
                    Diagnostic::error(
                        "train 'checkpoint_save' arg must be a non-empty string \
                         literal (file path)",
                    )
                    .with_label(arg.span, "expected a string literal path"),
                ),
            },
            "checkpoint_every" => match fold_int_literal(&arg.value.kind) {
                Some(n) if n >= 1 => {
                    checkpoint_every = n;
                    every_span = Some(arg.span);
                }
                _ => diags.push(
                    Diagnostic::error(
                        "train 'checkpoint_every' must be a positive integer \
                         literal (optimizer steps between checkpoints)",
                    )
                    .with_label(arg.span, "expected an integer literal >= 1"),
                ),
            },
            "checkpoint_load" => match &arg.value.kind {
                ExprKind::StringLiteral(s) if !s.is_empty() => {
                    checkpoint_load = Some(s.clone());
                }
                _ => diags.push(
                    Diagnostic::error(
                        "train 'checkpoint_load' arg must be a non-empty string \
                         literal (file path)",
                    )
                    .with_label(arg.span, "expected a string literal path"),
                ),
            },
            other => diags.push(
                Diagnostic::error(format!("unknown train config key '{other}'"))
                    .with_label(arg.span, EXPECTED_KEYS_LABEL),
            ),
        }
    }

    // Pairing: a save path with no cadence would never fire, and a cadence
    // with no path is inert — both are the accepted-but-ignored shape this
    // contract refuses.
    if checkpoint_save.is_some() && checkpoint_every == 0 {
        let mut d = Diagnostic::error(
            "train 'checkpoint_save' requires checkpoint_every=<N optimizer \
             steps> (a positive integer literal)",
        );
        if let Some(span) = save_span {
            d = d.with_label(span, "checkpoint_save has no cadence");
        }
        diags.push(d);
    }
    if checkpoint_every > 0 && checkpoint_save.is_none() {
        let mut d = Diagnostic::error(
            "train 'checkpoint_every' without checkpoint_save= is inert — \
             nothing would be saved",
        );
        if let Some(span) = every_span {
            d = d.with_label(span, "add checkpoint_save=\"path\" or remove this");
        }
        diags.push(d);
    }

    if model.is_none() && purpose == TrainConfigPurpose::UserTrainBlock {
        diags.push(
            Diagnostic::error("train block requires a 'model' config arg")
                .with_label(train.span, "add model=<your model variable>"),
        );
    }

    if diags.is_empty() {
        Ok(ResolvedTrainConfig {
            model,
            epochs,
            grad_accumulation,
            grad_accumulation_explicit,
            grad_clip,
            checkpoint_save,
            checkpoint_every,
            checkpoint_load,
        })
    } else {
        Err(diags)
    }
}
