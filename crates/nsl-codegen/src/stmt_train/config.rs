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
    /// The scheduler advances once per optimizer step, not once per
    /// micro-batch: true unless the header says `schedule_unit =
    /// "micro_batch"`.
    pub(crate) schedule_in_optimizer_steps: bool,
    /// `precision="bf16"`: bf16 activation storage for this step.
    pub(crate) precision_bf16: bool,
    /// `param_dtype=` when it is not f32 (admitted): the weight matrices
    /// the step's GEMMs read are stored bf16 (`param_storage`).
    pub(crate) param_dtype: Option<nsl_semantic::train_config::ParamDtype>,
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
        let schedule_in_optimizer_steps = {
            use nsl_semantic::train_config::ScheduleUnit;
            ScheduleUnit::resolve(cfg.schedule_unit) == ScheduleUnit::OptimizerStep
        };
        let precision_bf16 = cfg.precision == Some(nsl_semantic::train_config::Precision::Bf16);
        if precision_bf16 {
            self.admit_bf16_activation_storage()?;
        }
        let param_dtype = cfg
            .param_dtype
            .filter(|p| *p != nsl_semantic::train_config::ParamDtype::F32);
        if let Some(pd) = param_dtype {
            self.admit_param_storage(pd, precision_bf16)?;
        }

        // P5 item 19: arm the cuda-graph runtime (its enable() re-checks the
        // runtime-only incompatibilities: NSL_CUDA_SYNC, kernel profiler,
        // legacy NULL stream). The accumulation window rides along — each
        // micro-batch phase within a window has its own self-consistent
        // allocator state, so the runtime captures one graph per
        // (region, phase) instead of requiring a phase-free digest.
        if self.compile_options.train.cuda_graphs {
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
            schedule_in_optimizer_steps,
            precision_bf16,
            param_dtype,
        })
    }

    /// The compositions `param_dtype=` admits (design `srbf16-design.md`
    /// 6.2). The bf16 parameters are read by the storage GEMMs of
    /// `precision="bf16"` and stepped only by the fused multi-tensor AdamW
    /// launch, on one device, with the optimizer state resident; each
    /// refusal names a setting that takes another path. The optimizer itself
    /// (AdamW/Adam with the fused multi step) is checked where the step is
    /// emitted (`optimizer_step.rs`).
    fn admit_param_storage(
        &self,
        pd: nsl_semantic::train_config::ParamDtype,
        precision_bf16: bool,
    ) -> Result<(), CodegenError> {
        let name = pd.as_str();
        if !precision_bf16 {
            return Err(CodegenError::new(format!(
                "param_dtype=\"{name}\" requires precision=\"bf16\": the bf16 storage plan's GEMMs are what read a \
                 bf16 parameter without a cast. Add precision=\"bf16\" or drop param_dtype="
            )));
        }
        if self.param_storage != Some(pd) {
            return Err(CodegenError::new(format!(
                "param_dtype=\"{name}\" is supported on a top-level train block only: the program's execution \
                 fingerprint and banner are decided from it before `main` runs"
            )));
        }
        if pd.is_test_only()
            && std::env::var("NSL_PARAM_DTYPE_TEST_MODES").ok().as_deref() != Some("1")
        {
            return Err(CodegenError::new(format!(
                "param_dtype=\"{name}\" is a test mode (the f32 master kept whole, to check the bf16 parameter \
                 plumbing bit for bit against f32 parameters); build with NSL_PARAM_DTYPE_TEST_MODES=1 to use it, \
                 or write param_dtype=\"bf16-split\""
            )));
        }
        let o = &self.compile_options;
        let refused: [(bool, &str); 8] = [
            (
                std::env::var("NSL_FASE_FUSED_STEP").ok().as_deref() == Some("0"),
                "NSL_FASE_FUSED_STEP=0",
            ),
            (
                std::env::var("NSL_FASE_MULTI_STEP").ok().as_deref() == Some("0"),
                "NSL_FASE_MULTI_STEP=0",
            ),
            (o.diagnostics.training_reference, "--training-reference"),
            (o.train.optim_state_offload, "--optim-state-offload"),
            (self.features.zero_stage.is_some(), "--zero-stage"),
            (self.features.world_size > 1, "--devices > 1"),
            (o.train.param_dtype_bf16sr, "--param-dtype bf16-sr"),
            (
                std::env::var("NSL_FASE_FUSED_OVERRIDE").is_ok_and(|v| !v.is_empty()),
                "NSL_FASE_FUSED_OVERRIDE (a per-layer FASE mode table)",
            ),
        ];
        if let Some((_, flag)) = refused.iter().find(|(on, _)| *on) {
            return Err(CodegenError::new(format!(
                "param_dtype=\"{name}\" does not compose with {flag}: the bf16 parameters are stepped only by the \
                 fused multi-tensor AdamW launch, on one device, with the optimizer state resident, and {flag} \
                 takes another path. Drop {flag}, or drop param_dtype="
            )));
        }
        Ok(())
    }

    /// The compositions `precision="bf16"` admits in this phase. Each refusal
    /// names a feature that has not been checked against bf16 activations:
    /// a refusal is a compile error, never a silent fall back to f32.
    fn admit_bf16_activation_storage(&self) -> Result<(), CodegenError> {
        if !self.activation_bf16 {
            // The pre-scan of `main` (which sets the program's matmul mode
            // and fingerprint from this key) did not see this block.
            return Err(CodegenError::new(
                "precision=\"bf16\" is supported on a top-level train block only: the program's \
                 matmul mode and execution fingerprint are decided from it before `main` runs",
            ));
        }
        if !self.features.source_ad_enabled {
            return Err(CodegenError::new(
                "precision=\"bf16\" requires --source-ad: bf16 activation storage is planned over \
                 the source-AD step; the tape path computes and stores f32",
            ));
        }
        let o = &self.compile_options;
        let refused: [(bool, &str); 8] = [
            (o.train.cuda_graphs, "--cuda-graphs"),
            (o.train.layerwise_accum, "--layerwise-accum"),
            (o.train.param_dtype_bf16sr, "--param-dtype bf16-sr"),
            (o.checkpoint.compress.is_some(), "--checkpoint-compress"),
            (o.memory.transient_arena, "--transient-arena"),
            (o.weight_stream.enabled, "--weight-stream"),
            (o.cpdt.mode == crate::cpdt::CpdtMode::Full, "--cpdt full (its precision plan)"),
            (o.wggo.moment_precision, "the WGGO moment-precision plan"),
        ];
        if let Some((_, flag)) = refused.iter().find(|(on, _)| *on) {
            return Err(CodegenError::new(format!(
                "precision=\"bf16\" does not compose with {flag} yet: that feature sizes, copies or \
                 re-types activations assuming f32 storage and has not been checked against the \
                 bf16 storage plan. Drop {flag}, or train with precision=\"f32\""
            )));
        }
        Ok(())
    }
}

/// The execution-fingerprint key of a program's parameter storage
/// (`param_dtype=`); `None` for f32 parameters, which emit no key.
pub(crate) fn param_storage_fingerprint(
    pd: nsl_semantic::train_config::ParamDtype,
) -> Option<&'static str> {
    use nsl_semantic::train_config::ParamDtype;
    match pd {
        ParamDtype::F32 => None,
        ParamDtype::Bf16Split => Some(",pdt=bf16split"),
        ParamDtype::Bf16Master => Some(",pdt=bf16master"),
        ParamDtype::Bf16MasterTz => Some(",pdt=bf16mastertz"),
    }
}

/// The runtime's code for a parameter storage (`nsl_abi::wire::param_storage`);
/// `None` for f32 parameters.
pub(crate) fn param_storage_wire(pd: nsl_semantic::train_config::ParamDtype) -> Option<i64> {
    use nsl_abi::wire::param_storage as wire;
    use nsl_semantic::train_config::ParamDtype;
    match pd {
        ParamDtype::F32 => None,
        ParamDtype::Bf16Split => Some(wire::SPLIT),
        ParamDtype::Bf16Master => Some(wire::MASTER),
        ParamDtype::Bf16MasterTz => Some(wire::MASTER_TZ),
    }
}
