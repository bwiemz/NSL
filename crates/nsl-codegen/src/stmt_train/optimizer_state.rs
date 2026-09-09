//! Section 4 of the train block: the optimizer-state buffers. Decide how
//! many moment lists the optimizer needs, print the offload notes, refuse
//! the checkpoint compositions the `.optim` sidecar cannot serialize, and
//! emit the runtime loop that allocates one (or two) moment tensors per
//! parameter — device or host-resident, owner-gated under ZeRO, null
//! under stage 3 (filled later), route-conditional under Muon.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1):
//! 344 lines, five escaping bindings ([`OptimizerState`]), no `self`
//! write. The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`)
//! pin the allocation loop on every fixture (one and two state buffers,
//! the CPDT precision and offload variants); the refusal text is found by
//! the CLI composition gate's wholesale sweep of `crates/nsl-codegen/src`;
//! the `[offload]` / `[muon]` notes are stderr pinned by the offload and
//! Muon gates in `nsl-cli`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{BlockArg, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::{SURFACE_OPTIM_M, SURFACE_OPTIM_V};

/// The bindings section 4 produces; names are the driver's.
pub(crate) struct OptimizerState {
    /// 1 (SGD/Lion) or 2 (Adam-family, SOAP, Muon).
    pub(crate) num_state_buffers: usize,
    /// P4 item 18 rung 2: the per-param dtype-code list forcing every
    /// first-moment buffer to bf16 storage, when `--muon-state-dtype bf16`.
    pub(crate) muon_state_m_codes: Option<Value>,
    /// The first-moment list (momentum / first moment).
    pub(crate) state_list_1: Value,
    /// The second-moment list, or `iconst 0` when `num_state_buffers < 2`.
    pub(crate) state_list_2: Value,
    /// Item C: the ZeRO-3 deferred-moment-fill latch (1 slot).
    pub(crate) moment_fill_latch: Option<Value>,
}

impl Compiler<'_> {
    /// Emit the optimizer-state buffers (see the module header).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_optimizer_state_buffers(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        optimizer_name: &str,
        param_paths: &[String],
        muon_route_list: Option<Value>,
        cpdt_precision_dtypes: Option<(Value, Value)>,
        checkpoint_save_path: &Option<String>,
        checkpoint_load_path: &Option<String>,
        checkpoint_every: i64,
        num_params_val: Value,
        param_list: Value,
        surface_prev: Value,
    ) -> Result<OptimizerState, CodegenError> {
        // Number of state buffers per param depends on optimizer:
        //   SGD/Lion: 1 (velocity/momentum)
        //   Adam/AdamW/SOAP: 2 (first moment m, second moment v)
        //   Muon (mixed Muon/AdamW): 2 — m is the Muon momentum buffer on
        //   rank-2 hidden weights and the AdamW first moment on routed
        //   params; v is the AdamW second moment, allocated ONLY where the
        //   AdamW arm reads it (item 9; Muon-routed params carry a null).
        //
        // State buffers are now NslLists (sized at runtime from param count)
        // instead of compile-time Vec<Value>, because the number of actual
        // tensor parameters is only known at runtime after recursive collection.
        let num_state_buffers = match optimizer_name {
            "adam" | "adamw" | "soap" | "muon" => 2,
            _ => 1,
        };

        // Optimizer-state offload (scaling campaign item 4): m/v allocate
        // HOST-resident (pinned when a GPU is live — P0.2); every optimizer
        // step stages them to the device, runs the unchanged F32 update, and
        // copies back on the transfer stream. P0.3: COMPOSES with
        // reduced-precision moments on this (non-pipelined) path — host m/v
        // are stored at the planned dtype and staged through the combined
        // cross-device cast envelope (nsl_tensor_cast_from_host /
        // nsl_tensor_cast_to_host_into), which replaces the co-resident
        // nsl_tensor_cast_into requant that used to force a hard refusal
        // here. The pipelined train path still refuses offload outright
        // (see compile_train_block_pipelined).
        if self.compile_options.train.optim_state_offload {
            if cpdt_precision_dtypes.is_some() {
                nsl_log::nsl_log!(INFO, "offload", 
                    "[offload] optimizer state (m/v) is HOST-resident at the \
                     planned reduced-precision dtypes (offload x \
                     --wggo-moment-precision/CPDT composition): each optimizer \
                     step dequants host state to device F32, updates, and \
                     quant-casts back (halved PCIe staging traffic vs f32 \
                     offload). VRAM saved: {num_state_buffers}x parameter bytes."
                );
            } else {
                nsl_log::nsl_log!(INFO, "offload", 
                    "[offload] optimizer state (m/v) is HOST-resident: each optimizer \
                     step stages state to the device and copies it back (2 PCIe \
                     round-trips of total state per step). VRAM saved: \
                     {num_state_buffers}x parameter bytes."
                );
            }
            if optimizer_name == "muon" {
                nsl_log::nsl_log!(INFO, "muon", 
                    "[muon] note: v is fully allocated (host-resident) under \
                     --optim-state-offload — the offload stage-in envelope \
                     touches both moments unconditionally, so the \
                     Muon-route v skip (item 9) applies only to the \
                     resident path."
                );
            }
        }

        // P4 item 18 rung 2: per-param dtype-code list forcing every
        // first-moment buffer to BF16 storage (code 3). Reuses the CPDT
        // precision alloc plumbing; v stays f32 (null-sloted per param on
        // the Muon route, and the AdamW arm's v is untouched by rung 2).
        let muon_state_m_codes: Option<Value> = if self.features.muon_state_bf16 {
            let list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            let bf16_code = builder.ins().iconst(cl_types::I64, 3);
            for _ in 0..param_paths.len() {
                self.compile_call_by_name(builder, "nsl_list_push", &[list, bf16_code])?;
            }
            Some(list)
        } else {
            None
        };

        // Milestone B: full-state checkpointing composes only where the m/v
        // lists are complete plain-f32 tensors the runtime can serialize
        // positionally. Every excluded composition is a loud compile error,
        // not a runtime surprise (the runtime aborts too, as the belt).
        if checkpoint_save_path.is_some() || checkpoint_load_path.is_some() {
            let opt = optimizer_name.to_lowercase();
            if opt != "adamw" && opt != "adam" {
                return Err(CodegenError::new(format!(
                    "checkpoint_save/checkpoint_load support AdamW/Adam only \
                     (optimizer here is '{optimizer_name}'): the .optim sidecar \
                     serializes exactly the two f32 moment lists"
                )));
            }
            if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
                return Err(CodegenError::new(
                    "checkpoint_save/checkpoint_load do not compose with \
                     --zero-stage: sharded/owner-gated moment lists hold null \
                     placeholder slots that a positional serializer cannot \
                     represent",
                ));
            }
            if cpdt_precision_dtypes.is_some() {
                return Err(CodegenError::new(
                    "checkpoint_save/checkpoint_load do not compose with CPDT \
                     moment precision: the .optim sidecar stores plain f32 \
                     moments only",
                ));
            }
            if checkpoint_save_path.is_some() && checkpoint_every <= 0 {
                return Err(CodegenError::new(
                    "checkpoint_save requires checkpoint_every=<N optimizer \
                     steps> (a positive integer literal)",
                ));
            }
            if self.features.world_size > 1 {
                return Err(CodegenError::new(format!(
                    "checkpoint_save/checkpoint_load do not compose with \
                     --devices {} : every rank would rename onto the same \
                     path and the survivor is whichever rank finished last",
                    self.features.world_size
                )));
            }
        }
        if checkpoint_every > 0 && checkpoint_save_path.is_none() {
            return Err(CodegenError::new(
                "checkpoint_every without checkpoint_save is inert — add \
                 checkpoint_save=\"<path>\" or remove checkpoint_every",
            ));
        }

        let state_list_1 = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        let state_list_2 = if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_new", &[])?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };

        // Item C: under stage 3 the loop below pushes NULLS and
        // `emit_deferred_moment_fill` allocates at the first window register
        // belt. This 1-slot list is that fill's one-shot latch. A per-slot
        // `state_list_N[idx] == 0` test was the obvious alternative and is
        // WRONG on cost: a non-owner's tensor-granular slot is legitimately
        // null forever, so its guard never latches and the owner gate would
        // re-run — ~(ws-1)/ws of the sharded set, every optimizer step,
        // inside the hot window region. The fill covers every slot in one
        // pass, so one flag is also exactly as correct.
        let zero3_defer_moments = self.features.zero_stage == Some(3);
        let moment_fill_latch: Option<Value> = if zero3_defer_moments {
            let l = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            self.compile_call_by_name(builder, "nsl_list_push", &[l, z])?;
            Some(l)
        } else {
            None
        };

        // Loop: for i in 0..num_params, create zeros_like(param_list[i])
        {
            let init_counter_var = builder.declare_var(cl_types::I64);
            let init_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(init_counter_var, init_zero);

            let init_header = builder.create_block();
            let init_body = builder.create_block();
            let init_exit = builder.create_block();

            builder.ins().jump(init_header, &[]);
            builder.switch_to_block(init_header);
            // Do NOT seal init_header here — the back-edge from init_body hasn't been added yet
            state.current_block = Some(init_header);

            let idx = builder.use_var(init_counter_var);
            let cond = builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, num_params_val);
            builder.ins().brif(cond, init_body, &[], init_exit, &[]);

            builder.switch_to_block(init_body);
            builder.seal_block(init_body);
            state.current_block = Some(init_body);

            let param_i = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;

            // D3 v2 (ZeRO-1): shard the optimizer-state ALLOCATION. v1
            // owner-gated only the UPDATE, so every rank still allocated FULL
            // m/v (no memory win). Now, when zero is enabled, a non-owner of
            // param `idx` (owner = idx % world_size) allocates NOTHING for its
            // moment buffers and pushes a null (0) placeholder — the lists stay
            // length==num_params and global-index-addressable, and the
            // owner-gated update branch is the ONLY reader of m/v, so nulls are
            // never dereferenced (the end-of-train free loop is null-safe:
            // nsl_tensor_free(0) is a no-op). The per-step collectives iterate
            // the grad/param lists, NEVER m/v, so allocation gating cannot
            // perturb the identical-collective-sequence spin-barrier invariant.
            // `zero_enabled` is recomputed locally: the outer binding is
            // introduced far below (near the optimizer loop), out of scope here.
            // P3 ZeRO-3 (item C): stage 3 allocates NOTHING here. Its
            // per-parameter decision needs the ParameterPlan (who is
            // elementwise, who is tensor-granular sharded, who is a
            // replicated epilogue/tied param) and the runtime carve that
            // sizes an elementwise slice — neither exists this early, so
            // section 4 pushes nulls and `emit_deferred_moment_fill` at the
            // window register belt fills every slot ONCE, after
            // registration. A "allocate full here, replace at the belt"
            // shape would save nothing: the transient peak IS the cost.
            //
            // `zero_enabled` (stages 1/2) keeps its own arm below and is
            // recomputed locally: the outer binding is introduced far below
            // (near the optimizer loop), out of scope here.
            let zero3_defer = zero3_defer_moments;
            let zero_enabled = self
                .features
                .zero_stage
                .filter(|&s| (1..=2).contains(&s))
                .is_some();
            let offload = self.compile_options.train.optim_state_offload;
            // `cpdt_precision_dtypes` is `Option<(Value, Value)>` (Value: Copy),
            // so projecting each moment's dtype-code list by value is fine.
            let m_list = cpdt_precision_dtypes.map(|(m, _)| m).or(muon_state_m_codes);

            // P0.1: first-moment buffers under the OptimM surface. The offload
            // variant allocates HOST tensors — inert GPU tag (host state is not
            // VRAM).
            let surface_optim_m = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_M);
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_optim_m])?;
            // Muon perf campaign (`--muon-resident-momentum`): under offload,
            // Muon-routed rank-2 params keep their momentum DEVICE-resident
            // (the stdlib-call envelope skips its staging on the exact same
            // (route==0 && rank==2) condition — the two sites must agree or
            // a host m would be updated as if device / vice versa). Refused
            // combos (zero/bf16/non-muon/no-offload) were rejected at parse.
            let muon_resident_m = optimizer_name == "muon"
                && offload
                && self.compile_options.muon.resident_momentum;
            let buf1 = if zero3_defer {
                builder.ins().iconst(cl_types::I64, 0)
            } else if muon_resident_m {
                let route_list = muon_route_list
                    .expect("muon route list is built before state buffers (4a)");
                let resident =
                    self.emit_muon_route_predicate(builder, route_list, idx, param_i)?;
                let dev_b = builder.create_block();
                let host_b = builder.create_block();
                let merge_b = builder.create_block();
                builder.append_block_param(merge_b, cl_types::I64);
                builder.ins().brif(resident, dev_b, &[], host_b, &[]);

                builder.switch_to_block(dev_b);
                builder.seal_block(dev_b);
                state.current_block = Some(dev_b);
                let dev_m =
                    self.emit_moment_zeros_like(builder, param_i, idx, m_list, false)?;
                builder.ins().jump(merge_b, &[BlockArg::Value(dev_m)]);

                builder.switch_to_block(host_b);
                builder.seal_block(host_b);
                state.current_block = Some(host_b);
                let host_m =
                    self.emit_moment_zeros_like(builder, param_i, idx, m_list, true)?;
                builder.ins().jump(merge_b, &[BlockArg::Value(host_m)]);

                builder.switch_to_block(merge_b);
                builder.seal_block(merge_b);
                state.current_block = Some(merge_b);
                builder.block_params(merge_b)[0]
            } else if zero_enabled {
                self.emit_owner_gated_moment(builder, state, param_i, idx, m_list, offload)?
            } else {
                self.emit_moment_zeros_like(builder, param_i, idx, m_list, offload)?
            };
            self.compile_call_by_name(builder, "nsl_list_push", &[state_list_1, buf1])?;
            if num_state_buffers >= 2 {
                // P0.1: second-moment buffers under the OptimV surface.
                let surface_optim_v = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_V);
                self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_optim_v])?;
                let v_list = cpdt_precision_dtypes.map(|(_, v)| v);
                // P1 Muon item 9: v (the AdamW second moment) is UNREAD on
                // the Muon route — allocate it only where the AdamW arm can
                // execute (role flag set OR runtime rank != 2, the EXACT
                // condition muon_step routes on) and push a null placeholder
                // otherwise. The state list stays length==num_params; the
                // AdamW arm is the only v reader and the end-of-train free
                // loop is null-safe (nsl_tensor_free(0) is a no-op) — the
                // same discipline as ZeRO's non-owner nulls. DISABLED under
                // offload / CPDT precision plans: their stage-in envelopes
                // touch s2 unconditionally, and offloaded v is host-resident
                // (no VRAM to win back) — a loud note records the choice.
                let muon_cond_v = optimizer_name == "muon"
                    && !offload
                    && cpdt_precision_dtypes.is_none();
                let buf2 = if zero3_defer {
                    builder.ins().iconst(cl_types::I64, 0)
                } else if muon_cond_v {
                    let route_list = muon_route_list
                        .expect("muon route list is built before state buffers (4a)");
                    // needs_v == NOT(muon-routed): inverted from the shared
                    // predicate instead of hand-spelling the De Morgan form
                    // (`flag != 0 || rank != 2`) a fourth time.
                    let routed =
                        self.emit_muon_route_predicate(builder, route_list, idx, param_i)?;
                    let needs_v = builder.ins().icmp_imm_s(IntCC::Equal, routed, 0);
                    let alloc_b = builder.create_block();
                    let skip_b = builder.create_block();
                    let merge_b = builder.create_block();
                    builder.append_block_param(merge_b, cl_types::I64);
                    builder.ins().brif(needs_v, alloc_b, &[], skip_b, &[]);

                    builder.switch_to_block(alloc_b);
                    builder.seal_block(alloc_b);
                    state.current_block = Some(alloc_b);
                    let real = if zero_enabled {
                        self.emit_owner_gated_moment(
                            builder, state, param_i, idx, v_list, offload,
                        )?
                    } else {
                        self.emit_moment_zeros_like(builder, param_i, idx, v_list, offload)?
                    };
                    builder.ins().jump(merge_b, &[BlockArg::Value(real)]);

                    builder.switch_to_block(skip_b);
                    builder.seal_block(skip_b);
                    state.current_block = Some(skip_b);
                    let null_v = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(merge_b, &[BlockArg::Value(null_v)]);

                    builder.switch_to_block(merge_b);
                    builder.seal_block(merge_b);
                    state.current_block = Some(merge_b);
                    builder.block_params(merge_b)[0]
                } else if zero_enabled {
                    self.emit_owner_gated_moment(builder, state, param_i, idx, v_list, offload)?
                } else {
                    self.emit_moment_zeros_like(builder, param_i, idx, v_list, offload)?
                };
                self.compile_call_by_name(builder, "nsl_list_push", &[state_list_2, buf2])?;
            }

            let one_init = builder.ins().iconst(cl_types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one_init);
            builder.def_var(init_counter_var, next_idx);
            builder.ins().jump(init_header, &[]);
            // Now seal init_header — both predecessors (entry jump + back-edge) are connected
            builder.seal_block(init_header);

            builder.switch_to_block(init_exit);
            builder.seal_block(init_exit);
            state.current_block = Some(init_exit);

            // End of the OptimM/OptimV bracket — restore the caller's surface.
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;
        }

        Ok(OptimizerState {
            num_state_buffers,
            muon_state_m_codes,
            state_list_1,
            state_list_2,
            moment_fill_latch,
        })
    }
}
