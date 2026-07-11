//! CPKD — Compiler-Planned Knowledge Distillation (v1).
//!
//! Treats knowledge distillation as a joint compilation problem instead of
//! two independent model executions.  The `distill(teacher=t, student=s,
//! epochs=N)` block compiles to a training loop in which:
//!
//! - the **teacher is structurally frozen** (composition-paper invariant
//!   I-11): every teacher model field registers as a `PrimalOp::Input` leaf
//!   in the Wengert list, so no adjoint is ever generated for it and the
//!   teacher backward is *physically absent* from the compiled step — no
//!   teacher gradient buffers exist to blow up memory (failure mode F-06);
//! - the **fused KL-CE distillation loss** (`fused_kl_ce(...)`, gated by the
//!   `@fused_kl_ce` decorator) computes both LM-head matmuls, the
//!   temperature-scaled KL term, and the hard-label CE term in one kernel
//!   without materializing either logit tensor in HBM;
//! - remaining paper innovations surface as **advisory** plan entries in v1
//!   (repo doctrine: opt-in, gate-off byte-identical, advisory before
//!   consumed): spectral logit-compression rank (`cpkd_spectral`), WGGO
//!   per-layer feature-match/attn-transfer/teacher-stream choices
//!   (`wggo_cpkd`), and CEP-guided student design (`cpkd_student`).
//!
//! v1 deferrals refuse loudly (never degrade silently):
//! - two-stream teacher/student overlap: the runtime has no stream plumbing
//!   (all launches target the NULL stream; ~86 unconditional
//!   `cuCtxSynchronize` sites; stream-oblivious caching allocator) — the
//!   schedule is compile-time sequential and the plan says so;
//! - attention transfer: fused attention kernels never materialize
//!   post-softmax attention weights in HBM — `attn_transfer = true` is a
//!   semantic error;
//! - tape-fallback distillation: the teacher-freeze guarantee is
//!   source-AD-structural, so a failed extraction is a hard compile error
//!   rather than a silent fall back onto a tape that would record teacher
//!   ops (the concrete F-06 failure path).

use serde::Serialize;

/// Which teacher layers feed the (advisory in v1) feature-matching loss.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub enum FeatureLayers {
    /// WGGO/importance-driven selection (advisory in v1).
    #[default]
    Auto,
    /// Explicit teacher layer indices from the `loss:` section.
    Explicit(Vec<i64>),
}

/// Parsed `loss:` section of a distill block (semantic layer has already
/// validated keys, literal-ness, and ranges; attn_transfer=true is refused
/// there).
#[derive(Debug, Clone, Serialize)]
pub struct DistillLossConfig {
    /// Hard-label CE weight; soft KL term gets `1 - alpha`.
    pub alpha: f64,
    /// KL softmax temperature (classic Hinton scaling: the KL term is
    /// multiplied by `T^2`).
    pub temperature: f64,
    /// Weight on the (user-composed) feature-matching loss. Advisory
    /// metadata in v1 — the step body owns the actual feature loss.
    pub feature_weight: f64,
    /// Teacher layers for feature matching (advisory in v1).
    pub feature_layers: FeatureLayers,
}

impl Default for DistillLossConfig {
    fn default() -> Self {
        DistillLossConfig {
            alpha: 0.5,
            temperature: 2.0,
            feature_weight: 0.1,
            feature_layers: FeatureLayers::Auto,
        }
    }
}

/// Compiler-side context for the distill block currently being lowered.
///
/// Installed by `compile_distill_block` around its `compile_train_block`
/// delegation (mirroring `active_fused_ce_config`'s install/restore
/// discipline) and consumed inside `compile_train_block_inner`:
/// - `student_sym`/`epochs` seed the config normally read from
///   `train(model=..., epochs=...)`;
/// - `teacher_sym` is registered as a model instance for method inlining
///   AND as a frozen root (I-11) on the Wengert extractor.
/// Facts collected while lowering one distill block, rendered as the
/// CPKD Distillation Build Report (CFIE-style numbered optimizations,
/// stderr at compile time — the report never lies: advisory entries say
/// "advisory", deferred entries say "deferred").
#[derive(Debug, Clone, Serialize)]
pub struct CpkdPlan {
    pub teacher_name: String,
    pub student_name: String,
    pub epochs: i64,
    pub loss: DistillLossConfig,
    /// Trainable (student) tensor params registered on the Wengert list.
    pub trainable_params: usize,
    /// Frozen teacher-field Input leaves (I-11 evidence: these received
    /// no adjoints and never reach the optimizer).
    pub frozen_teacher_inputs: usize,
    /// Whether the fused KL-CE op fired (decorator enabled + shape hints
    /// complete + call recognised).
    pub fused_kl_ce_fired: bool,
    /// Shape of the fused kernel when fired: (vocab, student_hidden,
    /// teacher_hidden, rows).
    pub fused_shape: Option<(u32, u32, u32, u32)>,
    /// HBM bytes saved by never materializing the two [rows, vocab] f32
    /// logit tensors (only meaningful when the fused op fired).
    pub logit_bytes_eliminated: u64,
}

impl CpkdPlan {
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "=== CPKD Distillation Build Report ===");
        let _ = writeln!(
            s,
            "Teacher: {} (frozen; {} weight tensors as no-grad inputs)",
            self.teacher_name, self.frozen_teacher_inputs
        );
        let _ = writeln!(
            s,
            "Student: {} (trainable; {} tensor params)",
            self.student_name, self.trainable_params
        );
        let _ = writeln!(
            s,
            "Loss: alpha={} temperature={} feature_weight={}",
            self.loss.alpha, self.loss.temperature, self.loss.feature_weight
        );
        let _ = writeln!(s, "Epochs: {}", self.epochs);
        let _ = writeln!(s, "Optimizations:");
        if self.fused_kl_ce_fired {
            let (v, hs, ht, rows) = self.fused_shape.unwrap_or((0, 0, 0, 0));
            let _ = writeln!(
                s,
                "  [1] Fused KL-CE: teacher+student logits never materialized \
                 (V={v}, HS={hs}, HT={ht}, rows={rows}; saves {:.1} MB HBM)",
                self.logit_bytes_eliminated as f64 / (1024.0 * 1024.0)
            );
        } else {
            let _ = writeln!(
                s,
                "  [1] Fused KL-CE: NOT active (add @fused_kl_ce(enabled=true, \
                 <shape hints>) and call fused_kl_ce(...) in the step body)"
            );
        }
        let _ = writeln!(
            s,
            "  [2] Teacher freeze (I-11): teacher backward structurally absent \
             ({} frozen inputs, 0 teacher gradient buffers)",
            self.frozen_teacher_inputs
        );
        let _ = writeln!(
            s,
            "  [3] Teacher scheduling: sequential (two-stream overlap is \
             advisory-only in v1; the runtime is single-stream)"
        );
        let _ = writeln!(
            s,
            "  [4] Spectral logit compression: advisory-only in v1 \
             (see `nsl check --cpkd-design-student` / cpkd_spectral)"
        );
        s
    }
}

#[derive(Debug, Clone)]
pub struct DistillContext {
    pub teacher_sym: nsl_ast::Symbol,
    pub student_sym: nsl_ast::Symbol,
    pub epochs: i64,
    pub loss: DistillLossConfig,
    /// The `@fused_kl_ce` decorator config for THIS distill block (looked
    /// up by `distill_block_stmt_id` in `compile_distill_block`), threaded
    /// into the Wengert extractor so `fused_kl_ce(...)` calls in the step
    /// body are recognised as a single fused op.
    pub fused_kl_ce: Option<crate::FusedKlCeDecoratorConfig>,
    /// alpha as EXPLICITLY written in the `loss:` section (`None` when the
    /// user omitted it). Used for the call-site literal cross-check only —
    /// `loss.alpha` always holds the effective (defaulted) value for the
    /// report, and defaults must not veto a call-site literal.
    pub loss_alpha_explicit: Option<f64>,
    /// temperature as explicitly written in the `loss:` section.
    pub loss_temperature_explicit: Option<f64>,
}
