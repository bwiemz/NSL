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
//!   consumed): WGGO per-layer feature-match/attn-transfer/teacher-stream
//!   choices (`wggo_cpkd`), and CEP-guided student design
//!   (`cpkd_student`). Spectral top-k logit compression is not one of
//!   them: the rank of the LM head does not give a top-k (it bounds the
//!   dimension of the logit subspace, not the number of tokens that carry
//!   probability), and its selector was removed.
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

/// The loss a distill step computes, as the lowering found it.
///
/// The block's `loss:` section does not define the loss: the step body
/// does. The section may state `alpha` and `temperature`, and they are then
/// checked against the `fused_kl_ce(...)` call that must consume them. This
/// is what the build report prints, so it names what runs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DistillLoss {
    /// The step body calls `fused_kl_ce(...)`.
    KlCe {
        /// Hard-label CE weight; the KL term gets `1 - alpha`. `None` when
        /// the call passes something other than a numeric literal.
        alpha: Option<f64>,
        /// KL softmax temperature (the KL term is multiplied by `T^2`).
        temperature: Option<f64>,
        /// `true`: the fused kernel computes it. `false`: the
        /// standard-library composite does.
        fused: bool,
    },
    /// The step body computes its own loss, with no `fused_kl_ce` call.
    StepBody,
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
    /// The accumulation window this block lowered with (1 = none).
    ///
    /// NECESSARY BUT NOT SUFFICIENT for a FASE-Deferred envelope, which is
    /// what the CPDT optimizer-moment precision plan lowers into: `fase::plan`
    /// returns `Passthrough` unconditionally at a window of 1, and at a window
    /// >= 2 it still needs an optimizer it can decompose incrementally. `Lion`
    /// (sign(m + g)) and every unrecognised optimizer name resolve to
    /// `FullBuffer` at ANY window. Read `fase_mode` for the answer; this field
    /// is the input, not the conclusion. Reasoning "window > 1, therefore
    /// Deferred" is exactly what put a wrong caveat in `render_report`.
    pub grad_accumulation: i64,
    /// The FASE mode the lowering actually selected, as its `Debug` spelling
    /// (`Deferred` / `Passthrough` / `FullBuffer`). Recorded rather than
    /// re-derived: a report that re-computed the mode from the window could
    /// disagree with the emitted code.
    pub fase_mode: String,
    pub loss: DistillLoss,
    /// Trainable (student) tensor params registered on the Wengert list.
    pub trainable_params: usize,
    /// Frozen teacher-field Input leaves (I-11 evidence: these received
    /// no adjoints and never reach the optimizer).
    pub frozen_teacher_inputs: usize,
    /// Shape of the fused kernel when the fused KL-CE op fired (decorator
    /// enabled + shape hints complete + call recognised): (vocab,
    /// student_hidden, teacher_hidden, rows). `Some` IS "it fired" — a
    /// separate bool was redundant state that could only ever agree or
    /// silently disagree with this field (review finding).
    pub fused_shape: Option<(u32, u32, u32, u32)>,
    /// HBM bytes saved by never materializing the two [rows, vocab] f32
    /// logit tensors (only meaningful when the fused op fired).
    pub logit_bytes_eliminated: u64,
}

/// The CPKD pass entry (Milestone C).
///
/// CPKD was the one registered pass with no module entry point — the driver
/// built the plan inline in `stmt.rs` and recorded the trace there, which the
/// old inline comment named as "an instance of the pass-bus tangle this
/// roadmap item exists to unpick". Recording HERE gives `pass_trace::record`
/// its callee-side choke point (the step-2 lesson: a record at call sites
/// misses drivers nobody wrapped) and gives the scheduler a body to
/// schedule. The plan is the caller's struct literal passed through — an
/// earlier revision mirrored every field into a `DistillLoweringFacts`
/// struct and transcribed it here field-by-field, a drift surface where a
/// mis-paired same-type copy compiles clean and reports wrong (review
/// finding).
///
/// Deliberately takes NO `WengertList`: the registry declares
/// `TapeAccess::None` for CPKD and `tape_access_drift.rs` enforces zero
/// code-level `WengertList` mentions across the cpkd* module family — so the
/// driver performs the one tape read this plan consumes (the FusedKlCe shape
/// scan) in `stmt.rs` and passes the RESULT in as `fused_shape`.
pub fn build_plan(plan: CpkdPlan) -> CpkdPlan {
    crate::pass_trace::record("CPKD");
    // AdvisoryOnly, and not a hedge: the plan built here is consumed by
    // exactly one thing, `render_report` (see `compile_distill_block`). The
    // fusion it reports on was performed by the `@fused_kl_ce` decorator
    // during extraction, not by this code — CPKD rewrites nothing.
    crate::pass_trace::record_disposition("CPKD", crate::pass_trace::PassDisposition::AdvisoryOnly);
    plan
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
        let shown = |v: Option<f64>| match v {
            Some(v) => v.to_string(),
            None => "<not a literal>".to_string(),
        };
        let _ = match &self.loss {
            DistillLoss::KlCe { alpha, temperature, fused } => writeln!(
                s,
                "Loss: fused_kl_ce(alpha={}, temperature={}), computed by {}",
                shown(*alpha),
                shown(*temperature),
                if *fused { "the fused kernel" } else { "the standard-library composite" }
            ),
            DistillLoss::StepBody => writeln!(
                s,
                "Loss: the step body's own expression (no fused_kl_ce call)"
            ),
        };
        let _ = writeln!(s, "Epochs: {}", self.epochs);
        // The caveat keys on the MODE, not on the window. Keying it on
        // `grad_accumulation > 1` was wrong in the one case that matters:
        // `Lion` (or any optimizer `FaseOptimizer::parse` does not recognise)
        // resolves to FullBuffer at ANY window, so a block with
        // `grad_accumulation = 4` printed a clean line while having no
        // Deferred envelope at all — which is exactly the state the caveat
        // exists to announce.
        let _ = writeln!(
            s,
            "Accumulation: grad_accumulation={} (FASE {}){}",
            self.grad_accumulation,
            self.fase_mode,
            if self.fase_mode == "Deferred" {
                String::new()
            } else if self.grad_accumulation > 1 {
                " — a window, but not a Deferred envelope (the optimizer \
                 decides), so a CPDT optimizer-moment precision plan has \
                 nothing to lower into"
                    .to_string()
            } else {
                " — no window, so no Deferred envelope for a CPDT \
                 optimizer-moment precision plan to lower into"
                    .to_string()
            }
        );
        let _ = writeln!(s, "Optimizations:");
        if let Some((v, hs, ht, rows)) = self.fused_shape {
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
        s
    }
}

#[derive(Debug, Clone)]
pub struct DistillContext {
    pub teacher_sym: nsl_ast::Symbol,
    pub student_sym: nsl_ast::Symbol,
    pub epochs: i64,
    /// The `@fused_kl_ce` decorator config for THIS distill block (looked
    /// up by `distill_block_stmt_id` in `compile_distill_block`), threaded
    /// into the Wengert extractor so `fused_kl_ce(...)` calls in the step
    /// body are recognised as a single fused op.
    pub fused_kl_ce: Option<crate::FusedKlCeDecoratorConfig>,
    /// alpha as written in the `loss:` section (`None` when the user
    /// omitted it). The `fused_kl_ce(...)` call in the step body must pass
    /// the same literal, and there must be such a call: a declared value
    /// nothing reads is refused.
    pub loss_alpha_explicit: Option<f64>,
    /// temperature as explicitly written in the `loss:` section.
    pub loss_temperature_explicit: Option<f64>,
}

// ─── distill → train presentation ────────────────────────────────────────

/// The single authority on how a `distill(...)` header presents as a
/// `train(...)` header.
///
/// A distill block IS a training loop over the student — `compile_distill_block`
/// says so structurally, by building a synthetic [`nsl_ast::block::TrainBlock`]
/// and delegating to the train driver. But that synthesis happens deep inside
/// codegen, so every module-level AST scan that asks "what does this module's
/// training loop look like?" — `training_report::build_report`,
/// `pca_activation::detect_packing_for_stmts` — pattern-matches
/// `StmtKind::TrainBlock` and sees *nothing at all* for a distill block. The
/// answers they compute from that silence are not conservative defaults, they
/// are wrong: an empty training report for a block that trains, and
/// `segment_masked = false` for a packed corpus, which is cross-document
/// attention contamination on a program that asked for packing.
///
/// Every consumer that needs a distill block's HEADER now reads it here — the
/// lowering path (`compile_distill_block`) and the training report — so they
/// cannot disagree about the accumulation window or anything else in it. A
/// consumer that needs only the block's SECTIONS does not come through this
/// function at all: `pca_activation` matches `StmtKind::DistillBlock` and reads
/// `distill.sections` directly, under the verbatim guarantee stated below,
/// rather than cloning a step body to answer a boolean.
///
/// **Which keys travel as train config, and why each is justified alone.**
/// Blanket-forwarding is refused: `compile_train_block_inner`'s config loop
/// ignores keys it does not recognise, so a distill-only key that later became
/// a train key would silently activate a train feature nobody declared for
/// distillation.
///
/// * `grad_accumulation` — travels as config, because config is where the
///   readers look for it: `compile_train_block_inner`'s loop (the FASE plan's
///   `accumulation` field and the accumulation gate) and, since this change,
///   `training_report::extract_fase_config`, which runs the same planner over
///   the same presented block. Carrying it is what lets a distill block reach
///   FASE-Deferred at all (PR #484); without it every distill block was pinned
///   to Passthrough and the CPDT optimizer-moment precision plan arbitrated to
///   nothing.
/// * `teacher` / `student` / `epochs` — do NOT travel as config. `train` has
///   no teacher/student keys, and on the lowering path [`DistillContext`] is
///   the single source of truth for the student and the epoch count (it seeds
///   them *after* the train config loop runs). A second copy in the config
///   would be a divergence waiting for one of the two to be edited. They are
///   returned as typed fields instead, so an analysis consumer with no
///   `DistillContext` — the training report, which must name the model it is
///   reporting on — can still get the right answer from the same place.
///
/// **`sections` travel verbatim** — a distill block's `data:`/`optimizer:`/
/// `step:`/`callbacks:` sections are literally the train sections
/// ([`nsl_ast::block::TrainSection`] is the shared type), so a scan that needs
/// only sections may match `StmtKind::DistillBlock` and read `distill.sections`
/// directly rather than paying for the clone this function makes.
///
/// # Errors
///
/// Returns the message a caller should surface when a header key cannot be
/// presented soundly. `epochs` is the key that used to be *silently* wrong:
/// `compile_distill_block` matched `IntLiteral` and left `epochs = 1` for
/// every other expression, so `distill(..., epochs = n)` compiled clean,
/// printed "Epochs: 1" and trained one epoch, where the identical
/// `train(model = m, epochs = n)` refuses.
///
/// Note WHICH LAYER refuses on each side. Train's is a codegen error
/// (`stmt.rs`: "train 'epochs' arg must be an integer literal"), so
/// `nsl check` on a train block with a non-literal `epochs` is GREEN.
/// Distill's is a span-labelled semantic diagnostic, so `nsl check` catches
/// it, and this function is only the backstop for a caller that lowers an
/// unchecked AST. Both refuse before anything is emitted; "distill is now
/// exactly as strict as train" is true of the compile, not of `nsl check`.
pub fn distill_as_train_block(
    distill: &nsl_ast::block::DistillBlock,
    interner: &nsl_lexer::Interner,
) -> Result<DistillAsTrain, String> {
    use nsl_ast::expr::{Arg, ExprKind};

    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("");
    let mut config: Vec<Arg> = Vec::new();
    let mut teacher_sym: Option<nsl_ast::Symbol> = None;
    let mut student_sym: Option<nsl_ast::Symbol> = None;
    let mut epochs: i64 = 1;

    for arg in &distill.config {
        let Some(name_sym) = arg.name else { continue };
        match resolve(name_sym) {
            "teacher" => {
                if let ExprKind::Ident(sym) = &arg.value.kind {
                    teacher_sym = Some(*sym);
                }
            }
            "student" => {
                if let ExprKind::Ident(sym) = &arg.value.kind {
                    student_sym = Some(*sym);
                }
            }
            "epochs" => match arg.value.kind {
                ExprKind::IntLiteral(n) if n >= 1 => epochs = n,
                _ => {
                    return Err(
                        "distill 'epochs' must be a positive integer literal: the \
                         epoch count is a compile-time constant, exactly as on a \
                         train block (a non-literal used to leave the loop at 1, \
                         and 0 would run it zero times — both with no diagnostic)"
                            .to_string(),
                    )
                }
            },
            "grad_accumulation" => {
                // Refused by VALUE, not clamped. The train path silently
                // clamps a non-literal to 1; on distill that clamp is
                // byte-for-byte the pre-#484 failure — FASE Passthrough,
                // hence no CPDT moment precision — with no diagnostic to
                // explain why the envelope never engaged.
                match arg.value.kind {
                    ExprKind::IntLiteral(n) if n >= 1 => config.push(arg.clone()),
                    _ => {
                        return Err(
                            "distill 'grad_accumulation' must be a positive integer \
                             literal"
                                .to_string(),
                        )
                    }
                }
            }
            _ => {}
        }
    }

    Ok(DistillAsTrain {
        train: nsl_ast::block::TrainBlock {
            config,
            sections: distill.sections.clone(),
            span: distill.span,
        },
        teacher_sym,
        student_sym,
        epochs,
    })
}

/// A distill header presented as a train header — see [`distill_as_train_block`].
#[derive(Debug, Clone)]
pub struct DistillAsTrain {
    /// The synthetic train block: the distill block's sections verbatim, plus
    /// exactly the header keys `train` also understands.
    pub train: nsl_ast::block::TrainBlock,
    /// `teacher = <ident>`. `None` when the header omitted it or wrote a
    /// non-identifier — the semantic checker refuses both, and the lowering
    /// path turns `None` into its own error naming the missing key.
    pub teacher_sym: Option<nsl_ast::Symbol>,
    /// `student = <ident>` — the model this block trains. Same `None`
    /// contract as `teacher_sym`.
    pub student_sym: Option<nsl_ast::Symbol>,
    /// `epochs = <int literal>`, defaulting to 1 when the header omits it
    /// (matching `train`). Non-literals are an error, not a default.
    pub epochs: i64,
}
