//! Section 1 of the train block: extract the training configuration from
//! the `train(...)` header — the model, epochs, gradient accumulation and
//! clipping, the checkpoint save/load/cadence — arm the CUDA-graphs
//! runtime when it is on, and apply the distill context's overrides.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1).
//! The resolver itself lives in `nsl-semantic` (`train_config`), the ONE
//! owner of the closed key set and its validation; this section is the
//! codegen-side call to it plus the handful of bindings the rest of the
//! train block reads, which is what [`TrainConfigSection`] carries. The
//! driver destructures it, so every binding keeps the name it had as a
//! local, and the rest of the function is untouched.
//!
//! The accept path is pinned by `tests/train_clif_snapshots.rs` (the
//! `cuda_graphs` fixtures pin the `nsl_cuda_graphs_enable` call); the two
//! refusals here are reached through `nsl check` first (`train_config`'s
//! own tests) and are the backstop for paths that bypass it.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use nsl_semantic::train_config::TrainConfigPurpose;

use crate::compiler::Compiler;
use crate::error::CodegenError;
use crate::stmt::GradAccumulationDecl;

/// What section 1 resolves. Field names are the driver's binding names.
pub(crate) struct TrainConfigSection {
    /// Who asked: a user-written train block, or the distill lowering's
    /// synthesized one (which legitimately has no `model=`).
    pub(crate) purpose: TrainConfigPurpose,
    pub(crate) model_sym: nsl_ast::Symbol,
    pub(crate) epochs: i64,
    pub(crate) grad_accumulation_steps: i64,
    pub(crate) grad_accumulation_decl: GradAccumulationDecl,
    /// `f64::MAX` when `grad_clip=` was not written (no clipping).
    pub(crate) grad_clip: f64,
    pub(crate) checkpoint_save_path: Option<String>,
    pub(crate) checkpoint_every: i64,
    pub(crate) checkpoint_load_path: Option<String>,
}

impl Compiler<'_> {
    /// Extract the train block's configuration (see the module header).
    pub(crate) fn extract_train_config(
        &mut self,
        builder: &mut FunctionBuilder,
        train: &nsl_ast::block::TrainBlock,
    ) -> Result<TrainConfigSection, CodegenError> {
        // ── 1. Extract config from train(...) args ──────────────────────
        // The Training Configuration Contract: ONE resolver (in
        // nsl-semantic, also called by `check_train_block` with spans)
        // owns the closed key set, duplicate/positional refusals, and
        // literal/range validation. The old inline match here ended in
        // `_ => {} // ignore unknown config for forward compat` — a typo'd
        // key silently trained with defaults, a non-literal
        // grad_accumulation silently clamped to 1, a non-literal grad_clip
        // vanished without a trace, and epochs=0 trained zero epochs.
        // This call is the backstop for paths that bypass `nsl check`
        // (notably the distill lowering's synthesized TrainBlock, which
        // legitimately has no model= — the context carries it).
        let purpose = if self.active_distill_context.is_some() {
            nsl_semantic::train_config::TrainConfigPurpose::DistillLowering
        } else {
            nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock
        };
        let cfg = nsl_semantic::train_config::resolve_train_config(
            train,
            &|sym| self.resolve_sym(sym).to_string(),
            purpose,
        )
        .map_err(|diags| {
            let msgs: Vec<String> = diags.into_iter().map(|d| d.message).collect();
            CodegenError::new(format!(
                "train config refused: {}",
                msgs.join("; ")
            ))
        })?;

        let mut model_sym: Option<nsl_ast::Symbol> = cfg.model;
        let mut epochs: i64 = cfg.epochs;
        let grad_accumulation_steps: i64 = cfg.grad_accumulation;
        let grad_accumulation_decl = if cfg.grad_accumulation_explicit {
            GradAccumulationDecl::Literal
        } else {
            GradAccumulationDecl::Omitted
        };
        let grad_clip: f64 = cfg.grad_clip.unwrap_or(f64::MAX); // MAX = no clipping
        // Milestone B: full-train-state checkpointing (θ + m/v + step).
        // Pairing (save↔every) already validated by the resolver.
        let checkpoint_save_path: Option<String> = cfg.checkpoint_save;
        let checkpoint_every: i64 = cfg.checkpoint_every;
        let checkpoint_load_path: Option<String> = cfg.checkpoint_load;

        // P5 item 19: arm the cuda-graph runtime (its enable() re-checks the
        // runtime-only incompatibilities: NSL_CUDA_SYNC, kernel profiler,
        // legacy NULL stream). The accumulation window rides along — each
        // micro-batch phase within a window has its own self-consistent
        // allocator state, so the runtime captures one graph per
        // (region, phase) instead of requiring a phase-free digest.
        if self.compile_options.cuda_graphs {
            let win = builder
                .ins()
                .iconst(cl_types::I64, grad_accumulation_steps.max(1));
            self.compile_call_by_name(builder, "nsl_cuda_graphs_enable", &[win])?;
        }

        // CPKD: a distill block delegates here with a config carrying only
        // the keys whose meaning is identical on both blocks — today just
        // `grad_accumulation`, read by the loop above. The `model=` role and
        // `epochs` deliberately do NOT travel that way: the context is their
        // single source of truth, and a second copy is a divergence waiting
        // for one side to be edited. (`compile_distill_block` installed it.)
        if let Some(distill) = &self.active_distill_context {
            model_sym = Some(distill.student_sym);
            epochs = distill.epochs;
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("train block requires 'model=<ident>' config argument")
        })?;

        Ok(TrainConfigSection {
            purpose,
            model_sym,
            epochs,
            grad_accumulation_steps,
            grad_accumulation_decl,
            grad_clip,
            checkpoint_save_path,
            checkpoint_every,
            checkpoint_load_path,
        })
    }
}
