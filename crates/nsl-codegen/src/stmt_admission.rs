//! Admission checks for the train block's window-buffered and sharded
//! schedules (`--layerwise-accum`, `--weight-stream`, `--zero-stage`).
//!
//! These compose only with the exact set of paths they were validated on, so
//! every other combination refuses loudly — the repo's deferral-must-refuse
//! rule — rather than silently running the interleaved baseline under a flag
//! that claims otherwise.
//!
//! Moved out of the body of `compile_train_block_inner` byte-for-byte
//! (roadmap A1). The block was an unusually clean seam: 169 lines that read
//! configuration, write no `self` field, emit no IR, and produce exactly one
//! value (`csla_active`). Naming its inputs in a signature is most of the
//! point — in `stmt.rs` they were five of the ~200 locals live at that
//! point, and nothing said which ones the refusals actually depend on.
//!
//! The accept path is pinned by `tests/train_clif_snapshots.rs` (28
//! snapshots, byte-identical across this move). The refusal paths are pinned
//! by the CLI composition gates — `feature_composition_gate.rs`,
//! `zero3_gate.rs`, `zero_spmd_gate.rs` and `muon_state_gate.rs` — which
//! assert the message text, so a dropped or reordered check fails there.
//!
//! The Item 7 (`--fuse-wgrad-accum`) admission block that sat immediately
//! above it in `compile_train_block_inner` followed by the same route
//! (`wgrad_fusion_admission` below): it only RECORDS — the refusal it feeds
//! is compile-scoped, in `Compiler::finish_wgrad_admission` — so its accept
//! path is the same 28 snapshots and its decline text is pinned by
//! `wgrad_accum_fusion_gate.rs` and `cpkd_distill_e2e.rs`.
//!
//! Remaining admission blocks in `compile_train_block_inner` belong here as
//! they are extracted.

use crate::compiler::Compiler;
use crate::param_roles::NoDecayScope;
use crate::stmt::GradAccumulationDecl;
use crate::CodegenError;

impl Compiler<'_> {
    /// Resolve `--layerwise-accum` admission and the `--zero-stage`
    /// composition rules, returning whether CSLA Stage-2 is active.
    pub(crate) fn csla_and_zero_admission(
        &self,
        grad_accumulation_steps: i64,
        grad_clip: f64,
        optimizer_name: &str,
        fase_deferred: bool,
        fase_plan_mode: crate::fase::FaseMode,
    ) -> Result<bool, CodegenError> {
    // ── CSLA Stage-2 (`--layerwise-accum`) admission ────────────────
    // Window-buffered scheduling only composes with the exact set of
    // paths it was validated on; everything else refuses loudly (the
    // repo's deferral-must-refuse rule) instead of silently running the
    // interleaved baseline under a flag that claims otherwise.
    let csla_active = self.compile_options.layerwise_accum;
    if self.compile_options.weight_stream && !csla_active {
        return Err(CodegenError::new(
            "--weight-stream requires --layerwise-accum (the window-scoped \
             eviction cycle is defined by the layer-major schedule)",
        ));
    }
    if csla_active {
        if !self.features.source_ad_enabled {
            return Err(CodegenError::new(
                "--layerwise-accum requires --source-ad: the window-buffered \
                 schedule replays the compile-time adjoint tape; the runtime \
                 tape-AD backward cannot be partitioned",
            ));
        }
        if grad_accumulation_steps <= 1 {
            return Err(CodegenError::new(format!(
                "--layerwise-accum requires grad_accumulation >= 2 in the {}: \
                 with a single-micro-batch window there is no accumulation \
                 window to buffer",
                self.training_block_noun()
            )));
        }
        if !fase_deferred {
            return Err(CodegenError::new(format!(
                "--layerwise-accum requires a FASE-Deferred plan (AdamW/Adam \
                 + grad_accumulation >= 2, or Muon's separate-accumulator \
                 window mode); this {} resolved to {:?}. Use \
                 AdamW, Adam, or Muon, or drop --layerwise-accum",
                self.training_block_noun(),
                fase_plan_mode
            )));
        }
        if grad_clip < f64::MAX {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with grad_clip: two-phase \
                 clipping needs the GLOBAL L2 norm over every parameter's \
                 completed m_partial before any update, which the layerwise \
                 schedule never materializes. Remove grad_clip or drop \
                 --layerwise-accum",
            ));
        }
        // D2a: --optim-state-offload NOW COMPOSES. The D1a refusal
        // protected against the P3 m_partial-staging half of the flag,
        // which is structurally moot under csla (the accumulator branch
        // checks csla FIRST, so slots stay NULL/device — host m_partial
        // cannot resurrect). What remains is exactly D2a's target: m/v
        // allocate host-pinned (the existing P0.2 path) and stage
        // per LAYER through fase_emit_final_step's wrap_offload
        // envelope at the per-layer update sites, with a drain after
        // each group update. The window's own accumulate hook stays
        // device-resident (wrap_offload=false at the hook site).
        if self.compile_options.checkpoint_compress.is_some() {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with --checkpoint-compress: \
                 the layerwise gate is bit-exact and compressed saves are not",
            ));
        }
        // P3 ZeRO-3: stage 3 is BUILT ON the layerwise schedule (its
        // per-layer group updates host the gradient all-reduce + the
        // owner-gated step, and its upload/evict sites drive the JIT
        // gather/release) — only stages 1/2 keep the M43 incompatibility:
        // their hooks read every accum slot after the backward, but the
        // layerwise schedule frees per-layer accumulators before the
        // (bypassed) optimizer gate.
        if self.features.zero_stage.filter(|&s| (1..=2).contains(&s)).is_some() {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with --zero-stage 1/2: the \
                 M43 ZeRO hooks read every accum slot after the backward, but \
                 the layerwise schedule's per-layer accumulators are freed \
                 before the optimizer gate. Drop one (or use --zero-stage 3, \
                 which composes with the layerwise schedule)",
            ));
        }
    }

    // D3 v1: only ZeRO-1 (optimizer-state sharding: owner-gated updates
    // + post-step parameter broadcast over the CPU-shm SimulatedBackend)
    // is lowered. Grad sharding (stage 2: per-shard reduce-scatter into
    // sharded accumulators) and param sharding (stage 3: JIT all-gather
    // in the forward) need sharded BUFFERS, not just sharded compute —
    // refuse loudly rather than silently running stage-1 semantics.
    if let Some(s) = self.features.zero_stage {
        // P4 item 16: stage 2 (gradient partitioning via owner-segmented
        // reduce_scatter) is lowered — same emission points as stage 1;
        // the runtime dispatches on the baked stage.
        //
        // P3 ZeRO-3 (items 12-14): stage 3 (tensor-granular parameter
        // partitioning + JIT gather) is lowered ON the layerwise
        // schedule: it requires --layerwise-accum + --weight-stream so
        // the per-layer residency sites exist (in zero3 mode their
        // backend is the collective broadcast, not host mirrors), and
        // the per-layer group updates host the gradient all-reduce +
        // owner-gated step. Anything else refuses.
        if s >= 4 {
            return Err(CodegenError::new(format!(
                "--zero-stage {s} does not exist: stages are 1 (optimizer \
                 sharding), 2 (+ gradient partitioning), 3 (+ parameter \
                 partitioning)",
            )));
        }
        // Item 11: refuse rather than silently ignore — stages 1/2 never
        // shard parameters, so the flag would be inert (the
        // declared-but-inert class the composition registry exists to
        // kill).
        if s != 3 && self.features.zero_elementwise {
            return Err(CodegenError::new(format!(
                "--zero-elementwise requires --zero-stage 3 (got stage \
                 {s}): stages 1/2 shard optimizer state and gradients, \
                 never parameters — there is nothing to slice",
            )));
        }
        if s == 3 && !(csla_active && self.compile_options.weight_stream) {
            return Err(CodegenError::new(
                "--zero-stage 3 requires --layerwise-accum --weight-stream \
                 (+ --checkpoint-blocks --source-ad): parameter sharding \
                 rides the layer-major residency schedule — its per-layer \
                 upload/evict sites become the JIT gather/release and its \
                 group updates host the owner-gated step. Add those flags \
                 or use --zero-stage 2",
            ));
        }
        if s == 3 && self.compile_options.optim_state_offload {
            return Err(CodegenError::new(
                "--zero-stage 3 with --optim-state-offload is not lowered: \
                 host-resident moments x owner-gated sharded updates is an \
                 untested composition (deferral-must-refuse). Drop one",
            ));
        }
        // Item 11: elementwise slices are stepped by the fused-AdamW
        // kernel on every rank — Muon's Newton-Schulz needs the WHOLE
        // matrix (gather-before-NS is the documented follow-up), and
        // SGD/Lion have no fused elementwise twin yet.
        if s == 3
            && self.features.zero_elementwise
            && !matches!(optimizer_name, "adamw" | "adam")
        {
            return Err(CodegenError::new(format!(
                "--zero-elementwise requires the AdamW/Adam optimizer \
                 (train block uses '{optimizer_name}'): every rank steps \
                 its 1/ws slice with the fused-AdamW kernel, and Muon's \
                 Newton-Schulz needs whole matrices. Drop \
                 --zero-elementwise, or switch the optimizer",
            )));
        }
        // D3 v1 (review): --zero-stage x grad_clip is unsafe and unlowered.
        // The all-reduce is emitted BEFORE the Deferred dispatch, but
        // two-phase clip folds the final micro-batch gradient into
        // m_partial AFTER it and derives the clip factor from a
        // RANK-LOCAL norm — so with real (non-replicated) data each rank
        // would clip differently and the owner's update would diverge.
        // It only looks bit-exact today because the loader is rank-blind.
        // Refuse, mirroring the zero x mode-table / zero x layerwise
        // refusals, rather than ship a latent miscompile.
        if s >= 1 && grad_clip != f64::MAX {
            return Err(CodegenError::new(
                "--zero-stage is incompatible with grad_clip: the gradient \
                 all-reduce precedes the two-phase clip, whose norm is \
                 computed rank-locally, so clipped multi-rank training \
                 would be silently wrong. Drop grad_clip or --zero-stage",
            ));
        }
    }

        Ok(csla_active)
    }
}

impl Compiler<'_> {
    /// AdamW parameter groups: refuse the compositions that hoist ONE
    /// weight-decay scalar out of the per-parameter loop, so `no_decay=[...]`
    /// is never silently ignored. Moved out of `compile_train_block_inner`
    /// byte-for-byte (roadmap A1); the three messages are pinned by
    /// `feature_rules.rs` against this file.
    pub(crate) fn no_decay_composition_admission(
        &self,
        no_decay_scope: &NoDecayScope,
        csla_active: bool,
    ) -> Result<(), CodegenError> {
        // weight-decay scalar out of the per-parameter loop. Each of these
        // would compile and train — decaying the parameters the user asked to
        // exempt — so they must refuse rather than silently ignore no_decay.
        if !no_decay_scope.is_empty() {
            if self.compile_options.muon_batch_ns {
                return Err(CodegenError::new(
                    "no_decay=[...] is not supported with --muon-batch-ns: the \
                     batched Newton-Schulz pre-loop takes one weight_decay \
                     scalar for the whole batch, so per-parameter exemption \
                     cannot be expressed there. Drop one",
                ));
            }
            if csla_active {
                return Err(CodegenError::new(
                    "no_decay=[...] is not supported with --layerwise-accum: the \
                     window-buffered group update hoists weight_decay out of \
                     the per-parameter loop. Drop one",
                ));
            }
            if self.compile_options.optim_state_offload {
                return Err(CodegenError::new(
                    "no_decay=[...] is not supported with --optim-state-offload \
                     yet (the staged host/device envelope is untested against \
                     per-parameter decay). Drop one",
                ));
            }
        }
        Ok(())
    }

    /// Item 7 (`--fuse-wgrad-accum`) admission: record whether this training
    /// block can reach the FASE-Deferred `on_param_grad` hook — the fusion's
    /// only firing site — and, when it cannot, the reason and remedy that the
    /// compile-scoped refusal in `Compiler::finish_wgrad_admission` reports.
    ///
    /// Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1):
    /// 119 lines that read configuration, emit no IR, and write exactly two
    /// `self` fields (`wgrad_hook_blocks`, `wgrad_declines`). The five
    /// inputs named here were five of the ~200 locals live at the call site.
    pub(crate) fn wgrad_fusion_admission(
        &mut self,
        grad_accumulation_steps: i64,
        grad_accumulation_decl: GradAccumulationDecl,
        optimizer_name: &str,
        fase_deferred: bool,
        fase_plan_mode: crate::fase::FaseMode,
    ) {
        // The fusion has exactly ONE way to fire: the FASE-Deferred
        // `on_param_grad` hook. `wengert_lower` plans only when that hook is
        // present, the sole `nsl_tensor_wgrad_accum` emission site sits inside
        // it, and — the part that made this invisible — the
        // `[wgrad-fusion] N chain(s) fused` counter is gated on it too. So a
        // build with the flag on and no hook produced no count, no warning and
        // no error: the user was told nothing at all.
        //
        // `grad_accumulation` defaults to 1 when the train block omits it, and
        // `fase::plan` returns `Passthrough` unconditionally at accumulation 1
        // for EVERY optimizer. That is the state `models/coder50m/pretrain.nsl`
        // and `pretrain_cert.nsl` are in, and `--pretrain-optimized` sets the
        // flag with no accumulation precondition — so the shipped pretraining
        // configuration fused zero chains while reporting success.
        //
        // NOT fixed by making the fusion work at accumulation 1: there is no
        // `m_partial` for the beta=1 GEMM to write into, so that needs a
        // beta=0 dW-output variant and a non-hook lowering path. What is fixed
        // here is the SILENCE.
        //
        // SCOPE: this block only RECORDS. The refusal is compile-scoped, in
        // `Compiler::finish_wgrad_admission` — see its header for why. The
        // note below stays per block because it is the part that names WHICH
        // block is inert, which a compile-scoped message cannot.
        let fase_hook_reachable = fase_deferred && self.features.source_ad_enabled;
        if self.compile_options.fuse_wgrad_accum {
            if fase_hook_reachable {
                self.wgrad_hook_blocks += 1;
            } else {
                // The remedy travels WITH the reason. It used to be one fixed
                // list appended to every refusal, which made two thirds of it
                // wrong in the common case: "switch to an Adam-family or SGD
                // optimizer" is dead advice at accumulation 1, because
                // `fase::plan` returns Passthrough there BEFORE it looks at
                // the optimizer at all — including on the shipped scripts,
                // which are AdamW already.
                let (reason, remedy): (String, &'static str) = if !self
                    .features
                    .source_ad_enabled
                {
                    (
                        "source-AD is off, so there is no compile-time adjoint \
                         tape to fuse and no FASE accumulate hook to fold into"
                            .to_string(),
                        "Add --source-ad, or drop --fuse-wgrad-accum.",
                    )
                } else if grad_accumulation_steps <= 1 {
                    // WHY the window is 1 is the whole content of this
                    // message, and asserting "the default when the train block
                    // omits it" unconditionally made it false for a block that
                    // plainly declares one: the train path CLAMPS a
                    // non-literal `grad_accumulation` to 1 with no diagnostic
                    // (spec/05-training-loop.nsl.md documents the asymmetry
                    // with `distill`, which refuses one instead). Sending that
                    // user to look for a missing `grad_accumulation=` that is
                    // right there in the source is the wrong-reason failure
                    // this admission exists to avoid.
                    //
                    // The noun is chosen, not hard-coded: a distill block
                    // lowers through this same function, and telling its
                    // author to edit "a train block" names a construct their
                    // program does not contain while leaving `distill`'s own
                    // `grad_accumulation` key unmentioned.
                    let noun = self.training_block_noun();
                    let how = match grad_accumulation_decl {
                        GradAccumulationDecl::Omitted => {
                            format!("the {noun} omits grad_accumulation, so it defaults to 1")
                        }
                        // A non-literal window is refused by the Training
                        // Configuration Contract before this point (train and
                        // distill alike), so Literal is exact.
                        GradAccumulationDecl::Literal => {
                            format!("the {noun} sets grad_accumulation to 1")
                        }
                    };
                    (
                        format!(
                            "grad_accumulation is 1 ({how}), so FASE is \
                             Passthrough and there is no m_partial for the \
                             beta=1 GEMM to accumulate into"
                        ),
                        if self.active_distill_context.is_some() {
                            "Set grad_accumulation to an integer literal >= 2 on \
                             the distill block, or drop --fuse-wgrad-accum."
                        } else {
                            "Set grad_accumulation to an integer literal >= 2 in a \
                             train block, or drop --fuse-wgrad-accum."
                        },
                    )
                } else {
                    (
                        format!(
                            "optimizer `{optimizer_name}` resolved to FASE \
                             {:?}, not Deferred — only a Deferred plan owns \
                             the per-parameter accumulate this fusion folds \
                             into",
                            fase_plan_mode
                        ),
                        "Use an Adam-family optimizer or SGD (Lion and unknown \
                         optimizers resolve to FullBuffer), or drop \
                         --fuse-wgrad-accum.",
                    )
                };
                // The instrument. Emitted on BOTH provenances, because the
                // whole defect was that the inert case said nothing; the
                // refusal is only reachable on one of them. Ordinal included
                // because the compile-scoped refusal deliberately does NOT
                // fire when a sibling block fused — so in a multi-block
                // program this line is the only thing that says which block
                // is inert.
                eprintln!(
                    "[wgrad-fusion] declined: {} #{} — {reason}",
                    self.training_block_noun(),
                    self.wgrad_block_ordinal()
                );
                self.wgrad_declines.push((reason, remedy));
            }
        }
    }
}
