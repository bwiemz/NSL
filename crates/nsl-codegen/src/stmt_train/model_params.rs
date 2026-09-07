//! Section 3 of the train block: resolve the model's type and layout,
//! enumerate its tensor parameters, build the runtime parameter list (and
//! the checkpoint name list), resolve the CPDT moment-precision plan into
//! per-parameter dtype-code lists, and build Muon's per-parameter mode
//! table.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1):
//! 612 lines, twelve escaping bindings ([`ModelParams`]), no `self` write.
//! The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`) pin the
//! parameter-list build on every fixture and the CPDT dtype lists on the
//! `*_cpdt` variants; the refusal text is found by the CLI composition
//! gate's wholesale sweep of `crates/nsl-codegen/src` and by the pins in
//! `feature_rules.rs` (`STMT_MODEL_PARAMS`); the `[cpdt]` / `[muon]`
//! notes are stderr pinned by the CPDT and Muon gates in `nsl-cli`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use crate::compiler::Compiler;
use crate::context::{FuncState, StructLayout};
use crate::error::CodegenError;
use crate::stmt::SURFACE_WEIGHTS;

/// The bindings section 3 produces; names are the driver's.
pub(crate) struct ModelParams {
    pub(crate) model_type_name: String,
    pub(crate) model_var_name: String,
    pub(crate) layout: StructLayout,
    pub(crate) model_ptr: Value,
    /// The caller's allocation surface, restored after each bracket.
    pub(crate) surface_prev: Value,
    /// Dotted tensor-parameter paths under the model, in list order.
    pub(crate) param_paths: Vec<String>,
    /// The runtime parameter list (`nsl_list_new` + one push per tensor).
    pub(crate) param_list: Value,
    /// `param_paths.len()` as an `iconst`.
    pub(crate) num_params_val: Value,
    /// The parameter-name list the checkpoint sidecar records.
    pub(crate) checkpoint_names_list: Option<Value>,
    /// The CPDT moment-precision plan as consumed (m codes, v codes).
    pub(crate) cpdt_moment_lists_consumed: Option<(Vec<u16>, Vec<u16>)>,
    /// The per-parameter dtype-code lists (m, v) the state allocation reads.
    pub(crate) cpdt_precision_dtypes: Option<(Value, Value)>,
    /// Muon: the per-parameter mode table's base pointer.
    pub(crate) mode_table_base: Option<Value>,
}

impl Compiler<'_> {
    /// Resolve the model and build the parameter lists (see the module header).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_model_params(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        model_sym: nsl_ast::Symbol,
        optimizer_name: &str,
        checkpoint_save_path: &Option<String>,
        csla_active: bool,
        fase_deferred: bool,
        fase_plan: &crate::fase::FasePlan,
    ) -> Result<ModelParams, CodegenError> {
        // Get the model pointer from state
        let (model_var, _) = *state.variables.get(&model_sym).ok_or_else(|| {
            CodegenError::new(format!(
                "undefined model variable '{}' in train block",
                self.resolve_sym(model_sym)
            ))
        })?;
        let model_ptr = builder.use_var(model_var);

        // Resolve model type name from the variable's semantic type.
        // First try model_var_types (set for for-loop model vars), then
        // state.variable_types (set for let-bound vars), then fall back to
        // scanning the type_map (unreliable — picks the first model type).
        let model_var_name = self.resolve_sym(model_sym).to_string();
        let model_type_name = self
            .models
            .model_var_types
            .get(&model_sym)
            .cloned()
            .or_else(|| {
                // Check semantic type from variable_types
                state
                    .variable_types
                    .get(&model_sym)
                    .and_then(|ty| match ty {
                        nsl_semantic::types::Type::Model { name, .. } => {
                            Some(self.resolve_sym(*name).to_string())
                        }
                        nsl_semantic::types::Type::Struct { name, .. } => {
                            Some(self.resolve_sym(*name).to_string())
                        }
                        _ => None,
                    })
            })
            .unwrap_or_else(|| model_var_name.clone());

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "no struct layout found for model '{}' in train block",
                    model_type_name
                ))
            })?;

        // Build param_list directly from the compiler's struct layouts instead
        // of the runtime pointer-probing collector. This keeps nested models
        // and fixed arrays aligned with the paths source AD already resolves.
        // Persistent pool for param_list and optimizer state allocation
        self.compile_call_by_name(builder, "nsl_gpu_set_persistent_pool", &[])?;

        // P0.1 per-surface VRAM accounting (the `SURFACE_*` tags at the top
        // of this module): each bracket below sets a surface for its
        // allocation region and restores the caller's surface afterwards
        // (get/set — nesting-safe).
        let surface_prev = self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
        let surface_weights = builder.ins().iconst(cl_types::I8, SURFACE_WEIGHTS);
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_weights])?;

        let param_paths = self.enumerate_model_tensor_paths(&model_var_name, &model_type_name);
        let param_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for path in &param_paths {
            let param_ptr = self
                .load_nested_field(builder, model_ptr, &layout, &model_type_name, path)
                .ok_or_else(|| {
                    CodegenError::new(format!(
                        "could not resolve model parameter '{}' in train block",
                        path,
                    ))
                })?;
            self.compile_call_by_name(builder, "nsl_list_push", &[param_list, param_ptr])?;
        }
        // End of the Weights bracket — restore the caller's surface.
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;

        // Milestone B: the .nslm header names for periodic checkpoints,
        // built ONCE at setup (host allocations — no surface bracket needed).
        // Same paths param_list was built from, so save order == list order.
        let checkpoint_names_list: Option<Value> = if checkpoint_save_path.is_some() {
            let l = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            for path in &param_paths {
                let display = path.strip_prefix("$model.").unwrap_or(path);
                let name_data_id = self.intern_string(display)?;
                let gv = self.module.declare_data_in_func(name_data_id, builder.func);
                let name_ptr = builder.ins().symbol_value(cl_types::I64, gv);
                self.compile_call_by_name(builder, "nsl_list_push", &[l, name_ptr])?;
            }
            Some(l)
        } else {
            None
        };

        // D3 (ZeRO-1): initialize the real sharding context ONCE at train
        // setup — the missing M43b emitter. The runtime builds the CPU-shm
        // SimulatedBackend from the `--devices N` spawner's env protocol
        // (rank + shm path); world_size is the compile-time value baked
        // here, and round-robin ownership (idx % ws) is established before
        // any optimizer machinery runs. world_size == 1 degenerates to
        // no-op collectives and rank-0 owning everything — identical to
        // the unsharded baseline by construction.
        if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
            let stage_val = builder
                .ins()
                .iconst(cl_types::I64, self.features.zero_stage.unwrap_or(1) as i64);
            let ws_val = builder
                .ins()
                .iconst(cl_types::I64, self.features.world_size.max(1) as i64);
            // D3 (review): ZeRO init/partition return codes were discarded —
            // a refused init (-2 NSL_SIMULATED_TP=0, -3 missing shm path)
            // left ZERO_CTX None, and every owner gate then read
            // owns_param==-1 and skipped ALL updates: the run trained a
            // frozen model and exited 0. Assert the rc so a refusal aborts
            // loudly instead of silently training nothing. (The runtime
            // FFIs still RETURN the code — unit tests exercise the refusal
            // paths directly; only the emitted call is made fatal.)
            let init_rc =
                self.compile_call_by_name(builder, "nsl_zero_init", &[stage_val, ws_val])?;
            let zero_i = builder.ins().iconst(cl_types::I64, 0);
            let init_ok = builder.ins().icmp(IntCC::Equal, init_rc, zero_i);
            let init_msg = "nsl: --zero-stage init failed (see message above) — \
                            aborting instead of training an unsynchronized model";
            self.intern_string(init_msg)?;
            let init_msg_ptr = self.compile_string_literal(builder, init_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[init_ok, init_msg_ptr])?;

            let np_val = builder
                .ins()
                .iconst(cl_types::I64, param_paths.len() as i64);
            // P4 item 13: BYTE-balanced ownership — the runtime reads each
            // param's byte size from param_list (identical on every rank) and
            // partitions by greedy LPT, so per-rank optimizer work and moment
            // memory track ~1/N in bytes rather than tensor count. Replaces
            // the index round-robin `nsl_zero_partition`.
            let part_rc = self.compile_call_by_name(
                builder,
                "nsl_zero_partition_bytes",
                &[param_list, np_val],
            )?;
            let part_ok = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, part_rc, zero_i);
            let part_msg = "nsl: --zero-stage partition failed — aborting";
            self.intern_string(part_msg)?;
            let part_msg_ptr = self.compile_string_literal(builder, part_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[part_ok, part_msg_ptr])?;

            // P3 ZeRO-3 (item 12): activate the tensor-granular residency
            // table and map every param pointer to its index (owners come
            // from the byte-balanced partition above). The weight-stream
            // registration/upload/evict sites the layerwise schedule emits
            // then redirect to the zero3 broadcast-fill backend at runtime.
            if self.features.zero_stage == Some(3) {
                self.compile_call_by_name(builder, "nsl_zero3_enable", &[])?;
                let z3_i_var = builder.declare_var(cl_types::I64);
                let z3_zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(z3_i_var, z3_zero);
                let z3_hdr = builder.create_block();
                let z3_body = builder.create_block();
                let z3_exit = builder.create_block();
                builder.ins().jump(z3_hdr, &[]);
                builder.switch_to_block(z3_hdr);
                state.current_block = Some(z3_hdr);
                let z3_i = builder.use_var(z3_i_var);
                let z3_n = builder
                    .ins()
                    .iconst(cl_types::I64, param_paths.len() as i64);
                let z3_c = builder.ins().icmp(IntCC::SignedLessThan, z3_i, z3_n);
                builder.ins().brif(z3_c, z3_body, &[], z3_exit, &[]);
                builder.switch_to_block(z3_body);
                builder.seal_block(z3_body);
                state.current_block = Some(z3_body);
                let z3_p =
                    self.compile_call_by_name(builder, "nsl_list_get", &[param_list, z3_i])?;
                self.compile_call_by_name(builder, "nsl_zero3_note_param", &[z3_p, z3_i])?;
                let z3_one = builder.ins().iconst(cl_types::I64, 1);
                let z3_next = builder.ins().iadd(z3_i, z3_one);
                builder.def_var(z3_i_var, z3_next);
                builder.ins().jump(z3_hdr, &[]);
                builder.seal_block(z3_hdr);
                builder.switch_to_block(z3_exit);
                builder.seal_block(z3_exit);
                state.current_block = Some(z3_exit);
            }
        }
        let num_params_val = builder
            .ins()
            .iconst(cl_types::I64, param_paths.len() as i64);

        // P0.3: arm the gradient-integrity exit report once, at train setup
        // (before the epoch/step loop), so --grad-integrity works with no env
        // var. The per-step check/note calls below feed the accumulator.
        if self.compile_options.grad_integrity {
            self.compile_call_by_name(builder, "nsl_grad_integrity_arm", &[])?;
            // All three backward paths feed the accumulator now: FullBuffer
            // (whole-list scan), FASE-interleaved (per-micro brackets), and
            // the CSLA windowed replay (window-scoped bracket — begin before
            // the range loop, note in the replay hook, end after the
            // epilogue update).
        }

        // CPDT precision-adaptive optimizer execution (v1): build per-param
        // storage dtype lists aligned with param_list, when active. Inactive ->
        // None (the existing FP32 path runs verbatim, zero behavior change).
        //
        // Borrow discipline: extract the owned dtype Vecs (and the activation
        // decision) into a local FIRST, which ends the `bus.cpdt_plan()` borrow.
        // Only then do the `compile_call_by_name` loop (which borrows `self`
        // mutably) run. No `unsafe`, no tensor clones.
        //
        // The ARBITRATED dtype lists — what the moments were actually
        // allocated from, not what CPDT offered — are also kept for the
        // in-place planning site: if the graph fingerprint later rejects the
        // pre-plan this consult consumed, the moments cannot be re-typed, so
        // the site re-arbitrates against the fresh plan and refuses on
        // divergence. Recording the raw CPDT offer instead would refuse a
        // correct compile with a wrong message whenever arbitration had
        // dropped (NotLoweredNoOptIn) or merged the offer.
        let cpdt_moment_lists_consumed: Option<(Vec<u16>, Vec<u16>)>;
        let cpdt_precision_dtypes: Option<(Value, Value)> = {
            let dtype_data: Option<(Vec<u16>, Vec<u16>)> = {
                // Pre-S2: the FASE cast wrapping was emitted ONLY on the
                // non-unified-dispatch Deferred branch (which runs iff WGGO is
                // inactive); the unified-dispatch arm hardcoded
                // `wrap_precision=false`. Allocating FP16 m/v on the WGGO path
                // would have fed FP16 buffers to an unwrapped FP32 update →
                // silent corruption.
                //
                // - S2 threaded the wrap through `emit_unified_optim_step_dispatch`'s
                //   Deferred sub-arm.
                // - S4 relaxed the gate from `wggo_overrides.is_none()` toward
                //   `true`, with a review-fix mode-table FullBuffer guard kept
                //   as the structural correctness backstop.
                // - S5 threads the wrap through `emit_stdlib_optim_call` so
                //   the unified-dispatch FullBuffer sub-arm ALSO wraps. With
                //   both sub-arms wrapping, the silent-corruption hazard is
                //   closed structurally and the FullBuffer guard can be
                //   lifted — `wrapped_path_active = true` unconditionally.
                //   The parameter is retained on `precision_active` as
                //   defense-in-depth for any future refactor that
                //   reintroduces a non-wrapping optimizer arm.
                let wrapped_path_active = true;

                // WGGO-plan moment bits take precedence when the plan
                // actually decided sub-32 storage for any layer: per the
                // paper, the Level-2 ILP chooses p_m/p_v (gated by an
                // informed sensitivity signal via `prec_allowed` — a plan
                // with no weight/calibration evidence carries 32/32 and
                // lands in the None arm here). Params are joined to layers
                // with the same `layer_prefix` the graph builder used;
                // unmatched params stay F32. Requires the Deferred plan
                // (both dispatch arms wrap m/v in the dequant→step→quant
                // envelope). 8-bit clamps to FP16 storage in v1, the same
                // ladder step as `clamp_int8_to_fp16`.
                let wggo_bits: Option<(Vec<u16>, Vec<u16>)> = if fase_deferred {
                    self.bus.wggo_overrides().and_then(|o| {
                        crate::cpdt_precision_exec::build_dtype_lists_from_overrides(
                            o,
                            &param_paths,
                        )
                    })
                } else {
                    None
                };

                let plan = self.bus.cpdt_plan();
                let active = plan
                    .map(|p| {
                        crate::cpdt_precision_exec::precision_active(
                            matches!(p.mode, crate::cpdt::CpdtMode::Full),
                            !p.precision.params.is_empty(),
                            true, // weights_present is implied by a non-empty precision plan
                            fase_deferred,
                            wrapped_path_active,
                        )
                    })
                    .unwrap_or(false);
                let cpdt_lists = if active {
                    let plan = plan.unwrap();
                    // Wrong-checkpoint backstop (review finding on the
                    // weights-only path): `cpdt_sensitivity::validate` needs
                    // an AppliedPlan and so cannot run when the offer was
                    // weights-only — a checkpoint naming a DIFFERENT model
                    // would otherwise reach here and report "active: 0
                    // moment buffer(s)", activation with no effect, the
                    // exact shape the join defect had. A non-empty
                    // precision plan whose names join ZERO of this block's
                    // params is certainly wrong; refuse it. (Partial joins
                    // are the join's documented residual gaps and are not
                    // judged here; the WGGO-path validate, when it runs,
                    // still checks the stronger every-layer property.)
                    if crate::cpdt_precision_exec::joined_param_count(
                        &plan.precision,
                        &param_paths,
                    ) == 0
                    {
                        return Err(CodegenError::new(
                            "--cpdt full derived a per-param precision plan \
                             from the weight map, but none of its parameter \
                             names join this train block's parameters — \
                             either the weight file describes a different \
                             model, or the parameter paths nest deeper than \
                             the join's one-segment strip accommodates (a \
                             recorded join limitation). Pass the checkpoint \
                             for THIS model, or drop --cpdt.",
                        ));
                    }
                    Some(crate::cpdt_precision_exec::build_dtype_lists(
                        &plan.precision,
                        &param_paths,
                    ))
                } else {
                    None
                };

                // Arbitration:
                // - WGGO's plan bits lower ONLY behind the explicit opt-in
                //   (--wggo-moment-precision): reduced-precision moments
                //   change training numerics. The dequant->step->quant cast
                //   envelope runs on BOTH devices (deferral-closure
                //   2026-07-14: nsl_tensor_zeros_like_dtype / nsl_tensor_cast
                //   / nsl_tensor_cast_into dispatch to the CFTP-v7 PTX cast
                //   kernels for GPU-resident params). Without the opt-in the
                //   decision stays advisory with a not-lowered notice.
                // - When both sources are live, merge CONSERVATIVELY per
                //   param: F32 wins. CPDT's PrecisionPlan carries per-param
                //   tiers (calibrated critical params pinned to F32) that a
                //   layer-uniform WGGO decision must not override; and a
                //   CPDT FP16 tier the WGGO layer kept at 32 bits is
                //   likewise deferred to the more conservative choice.
                // - When WGGO made NO moment-bit decision for this block at
                //   all (it may still be active for unrelated reasons —
                //   structural pruning, CSHA fusion, packing), an
                //   independent CPDT PrecisionPlan is not gated on the
                //   opt-in flag: there is nothing for it to arbitrate
                //   against. See `arbitrate_moment_precision`.
                use crate::cpdt_precision_exec::{
                    arbitrate_moment_precision, MomentPrecisionArbitration as MPA, DTYPE_F32,
                };
                match arbitrate_moment_precision(
                    wggo_bits,
                    cpdt_lists,
                    self.compile_options.wggo.moment_precision,
                ) {
                    MPA::NotLoweredNoOptIn => {
                        eprintln!(
                            "[cpdt] optimizer-moment precision NOT lowered: WGGO's \
                             plan carries reduced-precision m/v decisions but \
                             --wggo-moment-precision was not passed (opt-in: \
                             changes training numerics). Moments stay FP32."
                        );
                        None
                    }
                    MPA::Merged(m, v) => {
                        let sub32 = m.iter().chain(v.iter()).filter(|&&c| c != DTYPE_F32).count();
                        eprintln!(
                            "[cpdt] WGGO optimizer-moment precision active \
                             (merged with the CPDT per-param plan, F32 wins): \
                             {sub32} moment buffer(s) in FP16 storage \
                             (device-resident; GPU runs use the CFTP-v7 PTX \
                             cast kernels for the dequant->step->quant \
                             envelope)."
                        );
                        Some((m, v))
                    }
                    MPA::WggoOnly(m, v) => {
                        let sub32 = m.iter().chain(v.iter()).filter(|&&c| c != DTYPE_F32).count();
                        eprintln!(
                            "[cpdt] WGGO optimizer-moment precision active: {sub32} \
                             moment buffer(s) in FP16 storage (8-bit clamps to FP16 \
                             in v1; device-resident — GPU runs use the CFTP-v7 PTX \
                             cast kernels for the dequant->step->quant envelope)."
                        );
                        Some((m, v))
                    }
                    MPA::CpdtOnly(m, v) => {
                        // Every other active arm announces itself; a silent
                        // arm is invisible — this one WAS silent, which is
                        // part of how its dead consult went unnoticed.
                        let sub32 = m.iter().chain(v.iter()).filter(|&&c| c != DTYPE_F32).count();
                        eprintln!(
                            "[cpdt] optimizer-moment precision active (CPDT \
                             per-param plan): {sub32} moment buffer(s) in FP16 \
                             storage (device-resident; GPU runs use the \
                             CFTP-v7 PTX cast kernels for the \
                             dequant->step->quant envelope)."
                        );
                        Some((m, v))
                    }
                    MPA::Inactive => None,
                }
            };
            cpdt_moment_lists_consumed = dtype_data.clone();
            // P1 Muon item 11: muon's group update runs the stdlib muon_step
            // (no dequant->step->quant envelope), so reduced-precision
            // moments would feed FP16 buffers to an unwrapped update —
            // refuse instead of corrupting. (Reachable only via muon x
            // --layerwise-accum, where the plan is Deferred-shaped; off
            // CSLA, muon's FullBuffer plan already suppressed this.)
            if dtype_data.is_some() && optimizer_name == "muon" {
                return Err(CodegenError::new(
                    "muon does not support reduced-precision optimizer moments: \
                     the mixed Muon/AdamW step is the stdlib muon_step, which \
                     has no dequant->step->quant cast envelope. Drop \
                     --wggo-moment-precision / the CPDT precision plan, or use \
                     AdamW",
                ));
            }
            // P3 ZeRO-3: the owner-gated group updates don't thread the
            // precision envelope — refuse rather than feed FP16 moments to
            // an unwrapped update (deferral-must-refuse).
            if dtype_data.is_some() && self.features.zero_stage == Some(3) {
                return Err(CodegenError::new(
                    "--zero-stage 3 does not support reduced-precision \
                     optimizer moments yet. Drop --wggo-moment-precision / \
                     the CPDT precision plan, or use --zero-stage 2",
                ));
            }
            // P4 item 17: SR-BF16 authoritative weights. The fused SR step
            // is the ONLY sanctioned theta writer — refuse every composition
            // that could route any parameter's update through a non-SR path
            // (stdlib dispatch, interpreted envelope, owner-gated collective
            // update), or that assumes f32 authoritative storage.
            if self.features.param_dtype_bf16sr {
                if optimizer_name != "adamw" && optimizer_name != "adam" {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr supports only AdamW/Adam in v1 \
                         (the fused SR step; Muon SR-state is the item-18 \
                         ladder). Drop the flag or switch optimizers",
                    ));
                }
                if !self.compile_options.weight_stream.enabled {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr requires --weight-stream: the \
                         bf16 authoritative mirrors ride the streaming \
                         residency schedule (transient f32 working views)",
                    ));
                }
                // Item 16×11: the ONE supported ZeRO composition is
                // elementwise stage 3 — every rank steps its own bf16
                // slice with the SR kernel, so the sanctioned-θ-writer
                // invariant holds on every rank. Tensor-granular stage 3
                // owner-gates the update (non-owner slices would go stale),
                // and stages 1/2 route θ through the owner partition; both
                // stay refused. The stage-1/2 arm is defense-in-depth:
                // bf16-sr requires --weight-stream, which clap-requires
                // --layerwise-accum, which already refuses stages 1/2.
                match self.features.zero_stage {
                    Some(3) if self.features.zero_elementwise => {}
                    Some(3) => {
                        return Err(CodegenError::new(
                            "--param-dtype bf16-sr composes with --zero-stage 3 \
                             only under --zero-elementwise (the tensor-granular \
                             owner-gated update would leave non-owner slices \
                             un-stepped). Add --zero-elementwise or drop a flag",
                        ));
                    }
                    Some(_) => {
                        return Err(CodegenError::new(
                            "--param-dtype bf16-sr does not compose with \
                             --zero-stage 1/2 (the owner-partitioned optimizer \
                             would route θ updates around the SR step). Drop \
                             one of the flags",
                        ));
                    }
                    None => {}
                }
                if self.compile_options.optim_state_offload {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr does not compose with \
                         --optim-state-offload (m/v must be plain device f32 \
                         for the fused SR step)",
                    ));
                }
                if dtype_data.is_some() {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr does not compose with \
                         reduced-precision optimizer moments (drop \
                         --wggo-moment-precision / the CPDT precision plan)",
                    ));
                }
                if self.compile_options.training_reference {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr requires the fused optimizer \
                         step, which --training-reference disables",
                    ));
                }
                if !fase_deferred {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr requires the FASE-Deferred \
                         plan (the FullBuffer stdlib dispatch would update \
                         theta without stochastic rounding). Train with \
                         gradient accumulation / --source-ad",
                    ));
                }
                if self.bus.has_wggo_overrides() {
                    return Err(CodegenError::new(
                        "--param-dtype bf16-sr does not compose with WGGO \
                         per-layer FASE overrides yet (a FullBuffer-routed \
                         param would bypass the SR step). Drop the WGGO plan",
                    ));
                }
            }
            // P4 item 18 rung 2: BF16 Muon momentum (f32 working buffer +
            // counter-based SR store). The envelope lives in the CSLA muon
            // group update — refuse every path that would read/write the
            // bf16 m buffer without it.
            if self.features.muon_state_bf16 {
                if optimizer_name != "muon" {
                    return Err(CodegenError::new(
                        "--muon-state-dtype bf16 applies to the Muon \
                         optimizer only (AdamW reduced-precision moments are \
                         --wggo-moment-precision / the CPDT plan). Drop the \
                         flag or switch to Muon",
                    ));
                }
                if !self.compile_options.layerwise_accum {
                    return Err(CodegenError::new(
                        "--muon-state-dtype bf16 requires --layerwise-accum: \
                         the dequant->step->SR-quant envelope lives in the \
                         CSLA group update (the FullBuffer stdlib dispatch \
                         would touch the bf16 momentum unwrapped)",
                    ));
                }
                if self.features.zero_stage.is_some() {
                    return Err(CodegenError::new(
                        "--muon-state-dtype bf16 does not compose with \
                         --zero-stage yet (shard-allocated moments bypass \
                         the envelope). Drop one of the flags",
                    ));
                }
                if self.compile_options.optim_state_offload {
                    return Err(CodegenError::new(
                        "--muon-state-dtype bf16 does not compose with \
                         --optim-state-offload (host-resident momentum would \
                         bypass the device SR store). Drop one of the flags",
                    ));
                }
            }
            if let Some((m_codes, v_codes)) = dtype_data {
                let m_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                let v_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                for &code in &m_codes {
                    let c = builder.ins().iconst(cl_types::I64, code as i64);
                    self.compile_call_by_name(builder, "nsl_list_push", &[m_list, c])?;
                }
                for &code in &v_codes {
                    let c = builder.ins().iconst(cl_types::I64, code as i64);
                    self.compile_call_by_name(builder, "nsl_list_push", &[v_list, c])?;
                }
                Some((m_list, v_list))
            } else {
                None
            }
        };

        // FASE Codegen Phase 2: build per-parameter mode table from WGGO's
        // per-layer decisions and emit it as a .rodata byte array. The
        // backward loop below loads `modes[gai]` to choose Deferred vs
        // FullBuffer per param. When None (no WGGO active), the loops use
        // today's monolithic `fase_deferred` branch (byte-identical to
        // pre-Phase-2 codegen).
        let mode_table_base: Option<cranelift_codegen::ir::Value> = if optimizer_name == "muon" {
            // P5 Muon (review M5): FaseOptimizer::Unknown plans FullBuffer-
            // global, but plan_with_overrides still materializes an
            // all-FullBuffer per-layer table when WGGO overrides exist —
            // which would route Muon through the unified dispatch, where
            // the mixed-step routing flags are not threaded. The table is
            // semantically empty for Muon (every byte FullBuffer), so skip
            // it and keep Muon on the monolithic loop. Say so loudly.
            if self.bus.has_wggo_overrides() {
                eprintln!(
                    "[muon] note: WGGO per-layer FASE overrides do not apply to \
                     the mixed Muon/AdamW optimizer (it has no Deferred mode) — \
                     ignoring the mode table; training uses the standard \
                     per-param step."
                );
            }
            None
        } else {
            let modes = crate::fase_codegen_table::build_param_mode_table(
                &param_paths,
                &model_var_name,
                fase_plan,
                self.bus.wggo_overrides(),
            );
            match modes {
                Some(bytes) => {
                    let suffix = self.fase_table_counter;
                    self.fase_table_counter += 1;
                    let func_suffix = format!("t{suffix}");
                    let data_id = self.emit_param_mode_table_rodata(&bytes, &func_suffix)?;
                    let global = self.module.declare_data_in_func(data_id, builder.func);
                    Some(
                        builder
                            .ins()
                            .symbol_value(cranelift_codegen::ir::types::I64, global),
                    )
                }
                None => None,
            }
        };
        if csla_active && mode_table_base.is_some() {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with a WGGO per-parameter FASE \
                 mode table (mixed Deferred/FullBuffer accumulation): the \
                 window-buffered backward assumes the uniform Deferred hook. \
                 Drop --wggo overrides or --layerwise-accum",
            ));
        }

        Ok(ModelParams {
            model_type_name,
            model_var_name,
            layout,
            model_ptr,
            surface_prev,
            param_paths,
            param_list,
            num_params_val,
            checkpoint_names_list,
            cpdt_moment_lists_consumed,
            cpdt_precision_dtypes,
            mode_table_base,
        })
    }
}
