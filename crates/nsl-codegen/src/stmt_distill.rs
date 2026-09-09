//! The `distill` block lowering: `compile_distill_block` builds a synthetic
//! `TrainBlock` from the distill sections, installs the
//! `active_distill_context` (student/teacher symbols, epochs, the frozen
//! teacher instance) on the compiler and delegates to the train-block
//! driver, then renders the CPKD report.
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1); the statement
//! dispatch (`stmt.rs::compile_stmt_dispatch`) still calls it.

use cranelift_frontend::FunctionBuilder;

use nsl_ast::expr::ExprKind;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    /// CPKD: lower a `distill(teacher=t, student=s, epochs=N):` block.
    ///
    /// v1 strategy: distillation IS a training loop over the student, so we
    /// delegate to `compile_train_block` with (a) a synthetic `TrainBlock`
    /// carrying the distill sections and (b) an `active_distill_context`
    /// installed on the compiler.  Inside `compile_train_block_inner` the
    /// context: seeds `model_sym = student` / `epochs`, registers the
    /// teacher instance for method inlining with its fields FROZEN on the
    /// Wengert extractor (I-11: teacher fields become Input leaves → no
    /// adjoints → teacher backward structurally absent), forbids the tape
    /// fallback (F-06: a tape would record teacher ops and allocate teacher
    /// grad buffers), and resolves teacher-field Input leaves to Cranelift
    /// values via `load_source_ad_named_param`.
    pub(crate) fn compile_distill_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        distill: &nsl_ast::block::DistillBlock,
        distill_block_stmt_id: nsl_ast::NodeId,
    ) -> Result<(), CodegenError> {
        // Deferred compositions refuse loudly rather than degrade.
        if self.features.pipeline_config.is_some() {
            return Err(CodegenError::new(
                "distill blocks do not support pipeline-parallel training in CPKD v1 \
                 (remove the pipeline configuration or use a train block)",
            ));
        }
        if !self.features.source_ad_enabled {
            return Err(CodegenError::new(
                "distill blocks require source AD (build with --source-ad): the \
                 teacher-freeze guarantee (I-11) is enforced structurally on the \
                 Wengert list; tape AD would record teacher ops and allocate \
                 teacher gradient buffers (F-06)",
            ));
        }

        // ── Present the distill header as a train header ────────────────
        // `cpkd::distill_as_train_block` is the ONE place that says which
        // header keys travel and how, shared with `training_report` — which
        // is the point: a distill block is a training loop, and before this
        // the module-level AST scans matched `StmtKind::TrainBlock` and saw
        // nothing at all, reporting "Training blocks found: 0" for a block
        // that trains and computing `segment_masked = false` for a packed
        // corpus. Keeping the extraction here, private to codegen, is what
        // left those scans nothing to read.
        let presented = crate::cpkd::distill_as_train_block(distill, self.interner)
            .map_err(CodegenError::new)?;
        let epochs = presented.epochs;
        let teacher_sym = presented.teacher_sym.ok_or_else(|| {
            CodegenError::new("distill block requires 'teacher=<model ident>'")
        })?;
        let student_sym = presented.student_sym.ok_or_else(|| {
            CodegenError::new("distill block requires 'student=<model ident>'")
        })?;

        // ── Parse loss: section into DistillLossConfig ──────────────────
        let mut loss_cfg = crate::cpkd::DistillLossConfig::default();
        let mut loss_alpha_explicit: Option<f64> = None;
        let mut loss_temperature_explicit: Option<f64> = None;
        for entry in &distill.loss {
            let Some(name_sym) = entry.name else { continue };
            let key = self.resolve_sym(name_sym).to_string();
            let as_f64 = |e: &nsl_ast::expr::Expr| -> Option<f64> {
                match e.kind {
                    ExprKind::FloatLiteral(v) => Some(v),
                    ExprKind::IntLiteral(v) => Some(v as f64),
                    _ => None,
                }
            };
            match key.as_str() {
                "alpha" => {
                    if let Some(v) = as_f64(&entry.value) {
                        loss_cfg.alpha = v;
                        loss_alpha_explicit = Some(v);
                    }
                }
                "temperature" => {
                    if let Some(v) = as_f64(&entry.value) {
                        loss_cfg.temperature = v;
                        loss_temperature_explicit = Some(v);
                    }
                }
                "feature_weight" => {
                    if let Some(v) = as_f64(&entry.value) {
                        loss_cfg.feature_weight = v;
                    }
                }
                "feature_layers" => match &entry.value.kind {
                    ExprKind::StringLiteral(s) if s == "auto" => {
                        loss_cfg.feature_layers = crate::cpkd::FeatureLayers::Auto;
                    }
                    ExprKind::ListLiteral(items) => {
                        let mut layers = Vec::with_capacity(items.len());
                        for item in items {
                            if let ExprKind::IntLiteral(n) = item.kind {
                                layers.push(n);
                            }
                        }
                        loss_cfg.feature_layers =
                            crate::cpkd::FeatureLayers::Explicit(layers);
                    }
                    _ => {}
                },
                // attn_transfer=true was refused at the semantic layer;
                // false is the only value that reaches codegen and it is
                // the default (no state to record).
                "attn_transfer" => {}
                _ => {}
            }
        }

        // The synthetic TrainBlock built above by the shared presentation.
        // Which keys travel — and why `teacher`/`student`/`epochs` stay out
        // of the config while `grad_accumulation` goes in — is documented on
        // `cpkd::distill_as_train_block`. Every consumer that needs a distill
        // block's HEADER builds it there, this site included, so they cannot
        // drift apart. `pca_activation` is not one of them: it reads only the
        // verbatim `sections`, so it matches `StmtKind::DistillBlock` directly
        // rather than cloning a step body to answer a boolean.
        let synthetic = presented.train;

        // Per-block @fused_kl_ce dispatch (mirrors CFTP v10 item 3's
        // per-train-block @fused_lm_ce lookup by stmt id).
        let fused_kl_ce = self
            .fused_kl_ce_configs
            .iter()
            .find(|c| c.distill_block_stmt_id == distill_block_stmt_id)
            .cloned();

        let saved_context = self.active_distill_context.replace(crate::cpkd::DistillContext {
            teacher_sym,
            student_sym,
            epochs,
            loss: loss_cfg,
            fused_kl_ce,
            loss_alpha_explicit,
            loss_temperature_explicit,
        });
        let result = self.compile_train_block(builder, state, &synthetic, distill_block_stmt_id);
        self.active_distill_context = saved_context;

        // Render the Distillation Build Report (facts collected during the
        // source-AD extraction inside the inner lowering). Stderr, CFIE
        // convention for in-codegen build reports.
        if result.is_ok()
            && let Some(plan) = self.bus.take_cpkd_plan()
        {
            eprint!("{}", plan.render_report());
        }
        result
    }
}
