//! FASE Deferred-mode emission for the train-block backward pass.
//!
//! Wired in from `stmt.rs` when `fase::plan()` returns `FaseMode::Deferred`.
//! Left deliberately thin — `stmt.rs` owns the outer micro-batch loop, the
//! parameter-list traversal, and the existing `accum_list` allocation; this
//! module only replaces:
//!
//!   1. the per-micro-batch accumulator update (was `accum += g`; now
//!      `m_partial += (1/N) * g` via the recipe from `fase_optimizer.rs`),
//!   2. the post-loop optimizer step (was "divide accum by N, run optimizer";
//!      now "run the fused per-parameter recipe, then zero m_partial").
//!
//! The buffer slot is the same allocation `stmt.rs` already makes — this
//! module does not allocate.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::fase::FasePlan;

/// Emit the Deferred-mode per-micro-batch accumulator update.
///
/// Free-function marker hook — the actual Cranelift IR emission is
/// `Compiler::fase_emit_accumulate`.
///
/// Returns `Ok(())` unconditionally; callers that only need to check whether
/// the stub is wired can call this.
pub fn emit_deferred_accumulate(_plan: &FasePlan) -> Result<(), String> {
    Ok(())
}

/// Emit the Deferred-mode fused final step (runs after the last micro-batch
/// backward): per-parameter optimizer update sourced from `m_partial`, then
/// `m_partial = 0`.
///
/// The real work lives in `Compiler::fase_emit_final_step`.  This free-function
/// is kept as a marker / integration test hook.
pub fn emit_deferred_final_step(_plan: &FasePlan) -> Result<(), String> {
    Ok(())
}

impl Compiler<'_> {
    /// Emit the per-parameter fused optimizer step for FASE Deferred mode.
    ///
    /// Runs on the final micro-batch of an accumulation window.  Reads
    /// `m_partial` (the accumulated mean gradient), updates the optimizer
    /// state (`m`, `v`), applies the parameter update, and zeroes
    /// `m_partial` for the next window.
    ///
    /// Caller contract: `theta_ptr`, `m_ptr`, `m_partial_ptr`, `v_ptr` are
    /// runtime pointers to the parameter's tensor and its optimizer-state
    /// buffers.  All four are borrowed (not owned) — this method does not
    /// free them.
    ///
    /// The `Tmp` register is allocated lazily (first write) and freed at the
    /// end of the method.  All other registers map directly to the four
    /// pointer arguments above.
    pub(crate) fn fase_emit_final_step(
        &mut self,
        builder: &mut FunctionBuilder,
        theta_ptr: Value,
        m_ptr: Value,
        m_partial_ptr: Value,
        v_ptr: Value,
        recipe: &crate::fase::UpdateRecipe,
        bc_params: Option<(Value, Value)>,  // (bc1_inv, bc2_inv) — None for non-AdamW
        wrap_precision: bool,  // CPDT precision-adaptive: wrap m/v in F32 cast/uncast
    ) -> Result<(), crate::error::CodegenError> {
        use crate::fase_optimizer::{emit_final_step, Register, UpdateOp};

        let program = emit_final_step(recipe);

        // CPDT precision wrapping: dequant m/v (FP16->F32) to working tensors,
        // run the unchanged FP32 update on those, then quant-store back into the
        // original (possibly FP16) buffers. F32 storage casts are identity copies,
        // so FP32-tier params are value-bit-identical. theta/m_partial unwrapped.
        //
        // Guard: only wrap when m_ptr and v_ptr are DISTINCT buffers. The SGD
        // path passes v == m as a placeholder (no v state); wrapping it would
        // double-cast the same pointer. This runtime check is the actual SGD
        // discriminator — the `precision_active` gate does NOT inspect optimizer
        // arity, so this guard (not the gate) is what makes single-state
        // optimizers safe if a caller ever passes wrap_precision=true for one.
        let wrap_precision = wrap_precision && m_ptr != v_ptr;
        let orig_m_ptr = m_ptr;
        let orig_v_ptr = v_ptr;
        // Note: emit the f32_code iconst and the cast calls ONLY inside the
        // wrap branch so the wrap_precision==false path emits zero new IR
        // (byte-for-byte identical to before this change).
        let (work_m, work_v) = if wrap_precision {
            let f32_code = builder.ins().iconst(cl_types::I64, 1); // DTYPE_F32
            let wm = self.compile_call_by_name(builder, "nsl_tensor_cast", &[m_ptr, f32_code])?;
            let wv = self.compile_call_by_name(builder, "nsl_tensor_cast", &[v_ptr, f32_code])?;
            (wm, wv)
        } else {
            (m_ptr, v_ptr)
        };
        let m_ptr = work_m;
        let v_ptr = work_v;

        // Lazily-allocated scratch tensor for the `Tmp` register.  Allocated
        // on first write, freed at end.
        let mut tmp_val: Option<Value> = None;

        // Lazy MHat / VHat scratch slots for bias-corrected moment views.
        // Allocated on first write by `ScalarMulByBc`; freed before returning.
        let mut m_hat_ptr: Option<Value> = None;
        let mut v_hat_ptr: Option<Value> = None;

        // Resolve a non-scratch symbolic register to a Cranelift Value.
        // Tmp, MHat, and VHat are handled inline in each op arm because they
        // require mutable access to `tmp_val`, `m_hat_ptr`, `v_hat_ptr`.
        #[inline(always)]
        fn resolve_reg(
            r: Register,
            theta_ptr: Value,
            m_ptr: Value,
            m_partial_ptr: Value,
            v_ptr: Value,
            m_hat_ptr: Option<Value>,
            v_hat_ptr: Option<Value>,
        ) -> Result<Value, crate::error::CodegenError> {
            match r {
                Register::Theta    => Ok(theta_ptr),
                Register::M        => Ok(m_ptr),
                Register::MPartial => Ok(m_partial_ptr),
                Register::V        => Ok(v_ptr),
                Register::MHat     => m_hat_ptr.ok_or_else(|| crate::error::CodegenError::new(
                    "Register::MHat read before ScalarMulByBc wrote it"
                )),
                Register::VHat     => v_hat_ptr.ok_or_else(|| crate::error::CodegenError::new(
                    "Register::VHat read before ScalarMulByBc wrote it"
                )),
                Register::G => Err(crate::error::CodegenError::new(
                    "final-step recipe must not reference Register::G"
                )),
                Register::Tmp => Err(crate::error::CodegenError::new(
                    "Tmp register must be resolved via tmp_val, not resolve_reg"
                )),
            }
        }

        // Constant used when calling helpers that borrow their argument (flags=0).
        let flags_zero = builder.ins().iconst(cl_types::I8, 0);

        for op in &program.ops {
            match op {
                // ── Zero ────────────────────────────────────────────────────
                UpdateOp::Zero(r) => {
                    let ptr = resolve_reg(*r, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                    self.compile_call_by_name(builder, "nsl_tensor_zero_inplace", &[ptr])?;
                }

                // ── ScalarMulAdd: dst = a*src + b_scale*b_src ───────────────
                // This is always `dst = a*src + b_scale*b_src` where dst may
                // alias src (e.g. M = β₁·M + (1-β₁)·MPartial).
                // Strategy: compute each scaled term into an owned temp, copy
                // the first term into dst, then add_inplace the second.
                UpdateOp::ScalarMulAdd { dst, src, a, b_src, b_scale } => {
                    let src_ptr = resolve_reg(*src, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                    let dst_ptr = if *dst == Register::Tmp {
                        // If dst is Tmp, we'll handle the pointer below.
                        None
                    } else {
                        Some(resolve_reg(*dst, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?)
                    };

                    // tmp1 = a * src  (owned)
                    let a_val = builder.ins().f64const(*a);
                    let tmp1 = self.compile_call_by_name(
                        builder, "nsl_tensor_mul_scalar", &[src_ptr, a_val, flags_zero]
                    )?;

                    let result = if let Some(b_reg) = b_src {
                        // tmp2 = b_scale * b_src  (owned)
                        let b_ptr = if *b_reg == Register::Tmp {
                            tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                                "ScalarMulAdd: Tmp register read before first write"
                            ))?
                        } else {
                            resolve_reg(*b_reg, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                        };
                        let b_scale_val = builder.ins().f64const(*b_scale);
                        let tmp2 = self.compile_call_by_name(
                            builder, "nsl_tensor_mul_scalar", &[b_ptr, b_scale_val, flags_zero]
                        )?;
                        // Merge: tmp1 += tmp2  (tmp1 is owned so we can mutate it)
                        self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[tmp1, tmp2])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[tmp2])?;
                        tmp1
                    } else {
                        tmp1
                    };

                    if *dst == Register::Tmp {
                        // Free old Tmp value if any, then store result as new Tmp.
                        if let Some(old) = tmp_val.replace(result) {
                            self.compile_call_by_name(builder, "nsl_tensor_free", &[old])?;
                        }
                    } else {
                        // Copy result into dst (which is a borrowed buffer), then free temp.
                        let dst_ptr = dst_ptr.unwrap();
                        self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dst_ptr, result])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[result])?;
                    }
                }

                // ── Square: dst = src * src ──────────────────────────────────
                UpdateOp::Square { dst, src } => {
                    let src_ptr = resolve_reg(*src, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                    // flags=0: do not consume src (borrowed)
                    let sq = self.compile_call_by_name(
                        builder, "nsl_tensor_mul", &[src_ptr, src_ptr, flags_zero]
                    )?;
                    if *dst == Register::Tmp {
                        if let Some(old) = tmp_val.replace(sq) {
                            self.compile_call_by_name(builder, "nsl_tensor_free", &[old])?;
                        }
                    } else {
                        let dst_ptr = resolve_reg(*dst, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                        self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dst_ptr, sq])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[sq])?;
                    }
                }

                // ── SquaredAccumulate: dst = src_scale·src + scale·operand² ──
                // dst and src are always the same register (e.g. V = β₂·V + …).
                // Strategy: compute sq=operand², scaled_sq=scale*sq, decay dst
                // in place by src_scale, then add_inplace(dst, scaled_sq).
                // The decay step is the β₂ EMA damping of the second moment —
                // omitting it (the historical behavior) computed
                // v += (1-β₂)·g², an un-damped running sum.
                UpdateOp::SquaredAccumulate { dst, src, src_scale, operand, scale } => {
                    let operand_ptr = if *operand == Register::Tmp {
                        tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                            "SquaredAccumulate: Tmp operand read before first write"
                        ))?
                    } else {
                        resolve_reg(*operand, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                    };

                    // sq = operand * operand  (owned)
                    let sq = self.compile_call_by_name(
                        builder, "nsl_tensor_mul", &[operand_ptr, operand_ptr, flags_zero]
                    )?;
                    // scaled_sq = scale * sq  (owned; relinquish sq)
                    let scale_val = builder.ins().f64const(*scale);
                    let flags_relinq = builder.ins().iconst(cl_types::I8, 0b0000_0001); // relinquish_a
                    let scaled_sq = self.compile_call_by_name(
                        builder, "nsl_tensor_mul_scalar", &[sq, scale_val, flags_relinq]
                    )?;

                    // dst += scaled_sq.
                    // If dst == src (always the case here), just add_inplace.
                    // If dst != src, copy src into dst first (shouldn't happen in
                    // the current optimizer programs, but handle for safety).
                    if dst != src {
                        let src_ptr = resolve_reg(*src, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                        let dst_ptr = resolve_reg(*dst, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                        self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dst_ptr, src_ptr])?;
                    }
                    let dst_ptr = if *dst == Register::Tmp {
                        tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                            "SquaredAccumulate: Tmp dst not yet allocated"
                        ))?
                    } else {
                        resolve_reg(*dst, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                    };
                    // Decay the accumulator (dst *= src_scale) before adding
                    // the new squared term. Skip emission entirely for the
                    // neutral factor so non-EMA programs stay byte-identical.
                    if *src_scale != 1.0 {
                        let decay_val = builder.ins().f64const(*src_scale);
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_mul_scalar_inplace",
                            &[dst_ptr, decay_val],
                        )?;
                    }
                    self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[dst_ptr, scaled_sq])?;
                    // FBIP relinq contract: when mul_scalar reuses the input
                    // buffer in-place, it bumps refcount so the input ptr is
                    // still valid post-call. The caller must free BOTH the
                    // input (sq) and the output (scaled_sq) — under FBIP they
                    // are the same pointer with rc=2; the two frees drop it
                    // to 0. Without the second free, every SquaredAccumulate
                    // leaks one operand-sized tensor per call.
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[scaled_sq])?;
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[sq])?;
                }

                // ── SqrtPlusEps: dst = sqrt(src) + eps ─────────────────────
                // IMPORTANT: nsl_tensor_sqrt uses FBIP (mutates in-place when
                // refcount==1).  Persistent state registers (V, M, MPartial,
                // Theta) have refcount=1 when fetched via nsl_list_get, so a
                // direct call would corrupt the stored optimizer state.  We
                // therefore make an explicit copy of the source tensor first
                // (add_scalar with +0 and flags=0 forces allocation of a fresh
                // owned tensor), then call sqrt on that copy.  The copy is
                // transient (owned by this scope) and is freed after the Div step
                // when tmp_val is replaced.
                UpdateOp::SqrtPlusEps { dst, src, eps } => {
                    let src_ptr = if *src == Register::Tmp {
                        tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                            "SqrtPlusEps: Tmp src read before first write"
                        ))?
                    } else {
                        resolve_reg(*src, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                    };

                    // Force a non-FBIP copy of src so the persistent V buffer is not
                    // mutated by the subsequent nsl_tensor_sqrt call.
                    // add_scalar with flags=0 (no relinquish) always allocates a fresh tensor.
                    let zero_val = builder.ins().f64const(0.0);
                    let src_copy = self.compile_call_by_name(
                        builder, "nsl_tensor_add_scalar", &[src_ptr, zero_val, flags_zero]
                    )?;

                    // sqrt_val = sqrt(src_copy).  nsl_tensor_sqrt may FBIP if refcount==1.
                    // src_copy.refcount is 1 (freshly allocated above), so FBIP fires:
                    //   sqrt_val == src_copy, src_copy.refcount bumped to 2.
                    // We then free our own hold (src_copy) to drop refcount back to 1,
                    // leaving sqrt_val alive with refcount=1.
                    // If FBIP does NOT fire (future-proofing), sqrt_val is a fresh tensor
                    // and freeing src_copy correctly drops it to 0.
                    let sqrt_val = self.compile_call_by_name(
                        builder, "nsl_tensor_sqrt", &[src_copy]
                    )?;
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[src_copy])?;
                    // eps_result = sqrt_val + eps  (relinquish sqrt_val → FBIP may mutate in-place)
                    let eps_val = builder.ins().f64const(*eps);
                    let flags_relinq = builder.ins().iconst(cl_types::I8, 0b0000_0001);
                    let eps_result = self.compile_call_by_name(
                        builder, "nsl_tensor_add_scalar", &[sqrt_val, eps_val, flags_relinq]
                    )?;

                    if *dst == Register::Tmp {
                        if let Some(old) = tmp_val.replace(eps_result) {
                            self.compile_call_by_name(builder, "nsl_tensor_free", &[old])?;
                        }
                    } else {
                        let dst_ptr = resolve_reg(*dst, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                        self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dst_ptr, eps_result])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[eps_result])?;
                    }
                    // FBIP relinq contract: caller must also free the input we
                    // relinquished to add_scalar. Under FBIP, sqrt_val and
                    // eps_result share the same buffer with rc=2; the eps_result
                    // free above drops to 1, this one drops to 0.
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[sqrt_val])?;
                }

                // ── Div: dst = src / divisor ─────────────────────────────────
                UpdateOp::Div { dst, src, divisor } => {
                    let src_ptr = if *src == Register::Tmp {
                        tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                            "Div: Tmp src read before first write"
                        ))?
                    } else {
                        resolve_reg(*src, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                    };
                    let divisor_ptr = if *divisor == Register::Tmp {
                        tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                            "Div: Tmp divisor read before first write"
                        ))?
                    } else {
                        resolve_reg(*divisor, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                    };

                    // result = src / divisor  (owned; flags=0 — borrow both)
                    let result = self.compile_call_by_name(
                        builder, "nsl_tensor_div", &[src_ptr, divisor_ptr, flags_zero]
                    )?;

                    if *dst == Register::Tmp {
                        // Free the old Tmp (the divisor we just read) and store result.
                        if let Some(old) = tmp_val.replace(result) {
                            self.compile_call_by_name(builder, "nsl_tensor_free", &[old])?;
                        }
                    } else {
                        let dst_ptr = resolve_reg(*dst, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?;
                        self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dst_ptr, result])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[result])?;
                    }
                }

                // ── Update: θ -= lr * (scaled_m + wd·θ) ────────────────────
                // Decomposed as:
                //   adj = -lr * scaled_m                 (owned)
                //   if wd != 0: wd_term = -lr*wd * θ    (owned)
                //               adj += wd_term; free wd_term
                //   θ += adj   (add_inplace modifies θ in-place)
                //   free adj
                UpdateOp::Update { lr, wd, scaled_m } => {
                    let scaled_m_ptr = if *scaled_m == Register::Tmp {
                        tmp_val.ok_or_else(|| crate::error::CodegenError::new(
                            "Update: Tmp scaled_m read before first write"
                        ))?
                    } else {
                        resolve_reg(*scaled_m, theta_ptr, m_ptr, m_partial_ptr, v_ptr, m_hat_ptr, v_hat_ptr)?
                    };

                    // adj = -lr * scaled_m  (owned)
                    let neg_lr = builder.ins().f64const(-(*lr));
                    let adj = self.compile_call_by_name(
                        builder, "nsl_tensor_mul_scalar", &[scaled_m_ptr, neg_lr, flags_zero]
                    )?;

                    if *wd != 0.0 {
                        // wd_term = (-lr * wd) * θ  (owned)
                        let neg_lr_wd = builder.ins().f64const(-(*lr) * (*wd));
                        let wd_term = self.compile_call_by_name(
                            builder, "nsl_tensor_mul_scalar", &[theta_ptr, neg_lr_wd, flags_zero]
                        )?;
                        self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[adj, wd_term])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[wd_term])?;
                    }

                    // θ += adj  (modifies θ buffer in-place)
                    self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[theta_ptr, adj])?;
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[adj])?;
                }

                // ── SgdUpdate: θ -= lr * m_partial ──────────────────────────
                UpdateOp::SgdUpdate { lr } => {
                    // adj = -lr * m_partial  (owned)
                    let neg_lr = builder.ins().f64const(-(*lr));
                    let adj = self.compile_call_by_name(
                        builder, "nsl_tensor_mul_scalar", &[m_partial_ptr, neg_lr, flags_zero]
                    )?;
                    // θ += adj  (in-place)
                    self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[theta_ptr, adj])?;
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[adj])?;
                }

                // ── Sign ─────────────────────────────────────────────────────
                // Lion uses Sign, but Lion is FullBuffer mode in FASE Deferred.
                // If this is reached, the plan is malformed.
                UpdateOp::Sign { .. } => {
                    return Err(crate::error::CodegenError::new(
                        "Sign op reached in Deferred path — Lion should be FullBuffer"
                    ));
                }

                // ── ScalarMulByBc ─────────────────────────────────────────────
                // Bias-corrected moment scratch: dst = src * bc_val.
                // bc_params must be Some when this op is emitted (AdamW only).
                // Allocates an owned tensor stored in m_hat_ptr / v_hat_ptr;
                // freed at the end of this function.
                UpdateOp::ScalarMulByBc { dst, src, kind } => {
                    let (bc1, bc2) = bc_params.ok_or_else(|| crate::error::CodegenError::new(
                        "ScalarMulByBc emitted but bc_params is None — dispatcher must supply bias-correction scalars"
                    ))?;
                    let bc_val = match kind {
                        crate::fase_optimizer::BcKind::Beta1 => bc1,
                        crate::fase_optimizer::BcKind::Beta2 => bc2,
                    };
                    let src_ptr = match src {
                        Register::M => m_ptr,
                        Register::V => v_ptr,
                        other => return Err(crate::error::CodegenError::new(
                            format!("ScalarMulByBc src must be M or V, got {:?}", other)
                        )),
                    };

                    // Allocate owned tensor: out = src * bc_val.
                    // flags_zero forces a new allocation without relinquishing src —
                    // we must NOT mutate the persistent m/v buffers.
                    let out = self.compile_call_by_name(
                        builder,
                        "nsl_tensor_mul_scalar",
                        &[src_ptr, bc_val, flags_zero],
                    )?;

                    match dst {
                        Register::MHat => {
                            if let Some(prev) = m_hat_ptr {
                                self.compile_call_by_name(builder, "nsl_tensor_free", &[prev])?;
                            }
                            m_hat_ptr = Some(out);
                        }
                        Register::VHat => {
                            if let Some(prev) = v_hat_ptr {
                                self.compile_call_by_name(builder, "nsl_tensor_free", &[prev])?;
                            }
                            v_hat_ptr = Some(out);
                        }
                        other => return Err(crate::error::CodegenError::new(
                            format!("ScalarMulByBc dst must be MHat or VHat, got {:?}", other)
                        )),
                    }
                }
            }
        }

        // Zero m_partial for the next accumulation window.
        self.compile_call_by_name(builder, "nsl_tensor_zero_inplace", &[m_partial_ptr])?;

        // Free any live Tmp tensor.
        if let Some(t) = tmp_val {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[t])?;
        }

        // Free bias-corrected moment scratches.
        if let Some(p) = m_hat_ptr {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[p])?;
        }
        if let Some(p) = v_hat_ptr {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[p])?;
        }

        // CPDT precision wrapping (quant-store): copy the updated working
        // tensors back into the original (possibly FP16) m/v buffers, then free
        // the F32 working tensors. nsl_tensor_cast_into returns void — match the
        // existing void-call form (return value ignored). When wrap_precision is
        // false, m_ptr/v_ptr == orig_m_ptr/orig_v_ptr and nothing is emitted.
        if wrap_precision {
            self.compile_call_by_name(builder, "nsl_tensor_cast_into", &[orig_m_ptr, m_ptr])?;
            self.compile_call_by_name(builder, "nsl_tensor_cast_into", &[orig_v_ptr, v_ptr])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[m_ptr])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[v_ptr])?;
        }

        Ok(())
    }

    /// Emit `m_partial += accum_scale * grad` for a single parameter.
    ///
    /// - `m_partial_ptr`: runtime pointer (i64) to the accumulator slot
    ///   (produced by `nsl_list_get(accum_list, i)`).
    /// - `grad_ptr`: runtime pointer (i64) to the just-computed gradient.
    /// - `accum_scale`: the recipe's `accum_scale` field (1.0/N).
    ///
    /// After this call, the caller should free `grad_ptr`.
    ///
    /// Case B (no axpy helper in nsl-runtime): scale via
    /// `nsl_tensor_mul_scalar`, then add in-place via
    /// `nsl_tensor_add_inplace`, then free the temporary.
    pub(crate) fn fase_emit_accumulate(
        &mut self,
        builder: &mut FunctionBuilder,
        m_partial_ptr: Value,
        grad_ptr: Value,
        accum_scale: f64,
    ) -> Result<(), crate::error::CodegenError> {
        // Step 0: migrate grad to m_partial's (device, dtype). Tape-AD produces
        // CPU f64 gradients while m_partial is GPU f32 (it was allocated via
        // zeros_like(param), inheriting the parameter's placement). Without
        // this, `nsl_tensor_add_inplace` below hits a device/dtype mismatch
        // (CPU f64 src into GPU f32 dst) and panics inside the runtime.
        // `to_device_like` is a no-op (refcount++) when placements already match.
        let grad_migrated =
            self.compile_call_by_name(builder, "nsl_tensor_to_device_like", &[grad_ptr, m_partial_ptr])?;

        // Step 1: scaled_grad = grad_migrated * accum_scale  (owned new tensor)
        let scale_val = builder.ins().f64const(accum_scale);
        // flags=1: relinquish `grad_migrated` — we own the refcount bump from
        // to_device_like and mul_scalar can reuse the buffer when unique.
        let flags_relinq = builder.ins().iconst(cl_types::I8, 1);
        let scaled_grad =
            self.compile_call_by_name(builder, "nsl_tensor_mul_scalar", &[grad_migrated, scale_val, flags_relinq])?;

        // Step 2: m_partial += scaled_grad  (in-place, void)
        self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[m_partial_ptr, scaled_grad])?;

        // Step 3: free both refs to the buffer mul_scalar relinquished into.
        // Under FBIP, scaled_grad and grad_migrated alias the same tensor
        // with rc=2; both frees together drop it to 0. Without the second
        // free this leaks one parameter-sized tensor per accumulate call —
        // ~217/step at 500M with grad_accumulation=8.
        self.compile_call_by_name(builder, "nsl_tensor_free", &[scaled_grad])?;
        self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_migrated])?;

        Ok(())
    }

    /// FASE Codegen Phase 2: emit a runtime branch on the per-parameter
    /// FASE mode. Loads `modes[gai]` (a u8 from the .rodata table), tests
    /// for `Deferred = 1`, and conditionally jumps to one of the two
    /// caller-provided blocks.
    ///
    /// Caller is responsible for:
    /// - creating both destination blocks AND a join block,
    /// - switching to and sealing each destination block before emitting
    ///   its body,
    /// - jumping from each destination to the join block,
    /// - switching to and sealing the join block.
    pub(crate) fn emit_fase_mode_branch(
        &mut self,
        builder: &mut cranelift_frontend::FunctionBuilder,
        mode_table_base: cranelift_codegen::ir::Value,
        gai: cranelift_codegen::ir::Value,
        deferred_block: cranelift_codegen::ir::Block,
        fullbuffer_block: cranelift_codegen::ir::Block,
    ) {
        use cranelift_codegen::ir::{condcodes::IntCC, types as cl_types, InstBuilder, MemFlags};
        let byte_addr = builder.ins().iadd(mode_table_base, gai);
        let mode = builder
            .ins()
            .load(cl_types::I8, MemFlags::trusted(), byte_addr, 0);
        let one_i8 = builder.ins().iconst(cl_types::I8, 1);
        let is_deferred = builder.ins().icmp(IntCC::Equal, mode, one_i8);
        builder
            .ins()
            .brif(is_deferred, deferred_block, &[], fullbuffer_block, &[]);
    }

    /// Emit a per-parameter stdlib optimizer step call. Factored from the
    /// monolithic FullBuffer optimizer-step loop in `compile_train_block`
    /// so both the original loop and the unified per-param dispatch loop
    /// (FASE Codegen Phase 3) can share the arg-shape logic.
    ///
    /// Each optimizer has a distinct arg shape; the `match` arms preserve
    /// the original ordering verbatim.
    ///
    /// `s2` is always passed; SGD/Lion/Muon arms ignore it. Caller must
    /// load `s2` from `state_list_2` for num_state_buffers >= 2, or pass
    /// `s1` as a placeholder otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_stdlib_optim_call(
        &mut self,
        builder: &mut cranelift_frontend::FunctionBuilder,
        optimizer_name: &str,
        opt_fn: &str,
        param_val: cranelift_codegen::ir::Value,
        grad_val: cranelift_codegen::ir::Value,
        s1: cranelift_codegen::ir::Value,
        s2: cranelift_codegen::ir::Value,
        lr: cranelift_codegen::ir::Value,
        momentum_const: cranelift_codegen::ir::Value,
        dampening_const: cranelift_codegen::ir::Value,
        weight_decay_const: cranelift_codegen::ir::Value,
        nesterov_const: cranelift_codegen::ir::Value,
        beta1_const: cranelift_codegen::ir::Value,
        beta2_const: cranelift_codegen::ir::Value,
        eps_const: cranelift_codegen::ir::Value,
        step_count_var: cranelift_frontend::Variable,
        // CPDT precision-adaptive optimizer execution (FullBuffer sub-arm,
        // S5 follow-on): when true, dequant s1 (and s2 when distinct) from
        // FP16/INT8 → F32 into owned working tensors, run the unchanged F32
        // stdlib FFI on those, then quant the results back into the original
        // FP16/INT8 buffers via nsl_tensor_cast_into and free the working
        // tensors. When false, the wrap branch emits zero new IR — the
        // call-pattern is byte-identical to pre-S5. The `s1 != s2` guard
        // mirrors `fase_emit_final_step`: single-state optimizers
        // (SGD/Lion/Muon) alias s2 to s1, and double-casting that alias
        // would corrupt the buffer on cast_into-back.
        wrap_precision: bool,
        // Bias-correction step index override (f64). The legacy monolithic
        // FullBuffer path passes `None`: t = step_count+1, i.e. the
        // MICRO-batch counter — under grad accumulation it jumps by N
        // between optimizer steps (historical semantic, preserved).
        // The unified per-param dispatch passes `Some(window_t)` — the
        // window counter (step_count+1)/N — so a FullBuffer-mode param's
        // Adam/AdamW/SOAP bias correction advances at the same rate as the
        // Deferred arm's `nsl_bias_correction_inv(β, opt_step)`. Without
        // this, a mixed mode table would step its two arms with different
        // effective bias corrections even given identical gradients.
        t_override: Option<cranelift_codegen::ir::Value>,
    ) -> Result<(), crate::error::CodegenError> {
        // CPDT FP16/INT8 wrap envelope (S5). theta and grad are NOT wrapped
        // — params stay in their natural dtype (FP32 by default); CPDT §3.2
        // v1 targets optimizer-state memory only. Only s1/s2 are wrapped.
        let orig_s1 = s1;
        let orig_s2 = s2;
        let wrap_s2 = wrap_precision && s1 != s2;
        let (s1, s2) = if wrap_precision {
            let f32_code = builder.ins().iconst(cl_types::I64, 1); // DTYPE_F32
            let ws1 = self.compile_call_by_name(builder, "nsl_tensor_cast", &[s1, f32_code])?;
            let ws2 = if wrap_s2 {
                self.compile_call_by_name(builder, "nsl_tensor_cast", &[s2, f32_code])?
            } else {
                // SGD/Lion/Muon: s2 is an alias for s1. Aliasing ws2 to ws1
                // preserves the invariant that the FFI sees the same
                // pointer in both slots (the helper ignores s2 anyway for
                // these optimizers).
                ws1
            };
            (ws1, ws2)
        } else {
            (s1, s2)
        };
        match optimizer_name {
            "sgd" => {
                self.compile_call_by_name(
                    builder,
                    opt_fn,
                    &[
                        param_val, grad_val, s1,
                        lr, momentum_const, dampening_const, weight_decay_const, nesterov_const,
                    ],
                )?;
            }
            "adam" | "adamw" => {
                let t_float = if let Some(t) = t_override {
                    t
                } else {
                    let t_val = builder.use_var(step_count_var);
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_one = builder.ins().iadd(t_val, one);
                    builder.ins().fcvt_from_sint(cl_types::F64, t_plus_one)
                };
                self.compile_call_by_name(
                    builder,
                    opt_fn,
                    &[
                        param_val, grad_val, s1, s2,
                        lr, beta1_const, beta2_const, eps_const, weight_decay_const, t_float,
                    ],
                )?;
            }
            "lion" => {
                self.compile_call_by_name(
                    builder,
                    opt_fn,
                    &[
                        param_val, grad_val, s1,
                        lr, beta1_const, beta2_const, weight_decay_const,
                    ],
                )?;
            }
            "muon" => {
                self.compile_call_by_name(
                    builder,
                    opt_fn,
                    &[
                        param_val, grad_val, s1,
                        lr, momentum_const, weight_decay_const, nesterov_const,
                    ],
                )?;
            }
            "soap" => {
                let t_float_s = if let Some(t) = t_override {
                    t
                } else {
                    let t_val_s = builder.use_var(step_count_var);
                    let one_s = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_s = builder.ins().iadd(t_val_s, one_s);
                    builder.ins().fcvt_from_sint(cl_types::F64, t_plus_s)
                };
                self.compile_call_by_name(
                    builder,
                    opt_fn,
                    &[
                        param_val, grad_val, s1, s2,
                        lr, beta1_const, beta2_const, eps_const, t_float_s,
                    ],
                )?;
            }
            _ => {
                return Err(crate::error::CodegenError::new(format!(
                    "unsupported optimizer '{}' in train block",
                    optimizer_name
                )));
            }
        }
        // CPDT FP16/INT8 wrap envelope (S5) — close. Quant-cast the working
        // F32 results back into the originally-typed storage buffers, then
        // free the working tensors. Mirrors fase_emit_final_step's exit.
        //
        // Asymmetry note vs fase_emit_final_step: that helper folds the
        // m_ptr==v_ptr alias into a single combined `wrap_precision` flag
        // at entry, so the entire cast branch is gated by a single
        // condition. Here we instead use `wrap_precision` for the s1
        // cast/cast_into (always paired) and a separate `wrap_s2` for
        // the s2 pair — when SGD aliases s2==s1 we still cast s1 once
        // and cast it back once, but skip the s2 work to avoid
        // double-cast-into / double-free of the same buffer. Net effect
        // for SGD: 1 cast in, 1 cast_into out, 1 free. For multi-state
        // optimizers with distinct s1/s2: 2 cast in, 2 cast_into out,
        // 2 free. Both shapes leak nothing and preserve the original
        // buffer identities.
        if wrap_precision {
            self.compile_call_by_name(builder, "nsl_tensor_cast_into", &[orig_s1, s1])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[s1])?;
            if wrap_s2 {
                self.compile_call_by_name(builder, "nsl_tensor_cast_into", &[orig_s2, s2])?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[s2])?;
            }
        }
        Ok(())
    }

    /// Emit a single unified optimizer-step loop with per-param runtime
    /// dispatch on the FASE mode table. Used when `mode_table_base.is_some()`.
    ///
    /// Mixed mode tables are honored under `two_phase_clip`: the
    /// accumulation loop upstream uses the scaled window-mean convention
    /// for BOTH modes when clipping is active (stmt.rs `ga_fullbuf` arm),
    /// so Phase A's global ||accum||² norm is uniform, and both dispatch
    /// arms apply the shared clip factor to their param's accumulated
    /// gradient before stepping.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn emit_unified_optim_step_dispatch(
        &mut self,
        builder: &mut cranelift_frontend::FunctionBuilder,
        state: &mut crate::context::FuncState,
        mode_table_base: cranelift_codegen::ir::Value,
        num_params_val: cranelift_codegen::ir::Value,
        param_list: cranelift_codegen::ir::Value,
        state_list_1: cranelift_codegen::ir::Value,
        state_list_2: cranelift_codegen::ir::Value,
        num_state_buffers: usize,
        accum_list: Option<cranelift_codegen::ir::Value>,
        opt_grads: cranelift_codegen::ir::Value,
        // Per-micro-batch gradient list. Only read when `two_phase_clip`
        // is active AND `fase_hook_active` is false: on that path the
        // standard accumulation loop was skipped for the stepping
        // micro-batch (stmt.rs 7e3), so Phase A must fuse the final
        // accumulate here (mirroring the monolithic Deferred Phase A) and
        // free the per-batch grads + list wrapper it consumed.
        grads_list: Option<cranelift_codegen::ir::Value>,
        // True when the source-AD FASE hook already accumulated every
        // param's gradient during adjoint lowering (grads_list is a null
        // sentinel in that case).
        fase_hook_active: bool,
        step_count_var: cranelift_frontend::Variable,
        fase_plan: &crate::fase::FasePlan,
        optimizer_name: &str,
        opt_fn: &str,
        lr: cranelift_codegen::ir::Value,
        momentum_const: cranelift_codegen::ir::Value,
        dampening_const: cranelift_codegen::ir::Value,
        weight_decay_const: cranelift_codegen::ir::Value,
        nesterov_const: cranelift_codegen::ir::Value,
        beta1_const: cranelift_codegen::ir::Value,
        beta2_const: cranelift_codegen::ir::Value,
        eps_const: cranelift_codegen::ir::Value,
        grad_accumulation_steps: i64,
        grad_clip_threshold: f64,
        // CPDT precision-adaptive optimizer execution: when Some, the
        // Deferred sub-arm wraps `fase_emit_final_step` in nsl_tensor_cast
        // (dequant FP16/INT8 → F32) and nsl_tensor_cast_into (quant back).
        // None preserves the prior FP32-only behavior bit-identically.
        // Threaded from stmt.rs's `cpdt_precision_dtypes` binding.
        cpdt_precision_dtypes: Option<(cranelift_codegen::ir::Value, cranelift_codegen::ir::Value)>,
    ) -> Result<(), crate::error::CodegenError> {
        use cranelift_codegen::ir::{condcodes::IntCC, InstBuilder};

        // ── 1. Bias-correction scalars (computed once, used by Deferred branch) ──
        let sc_val = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
        let grad_accum_const =
            builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
        let opt_step = builder.ins().sdiv(sc_plus_one, grad_accum_const);
        // Window-step index as f64 for the FullBuffer arm's stdlib
        // Adam/AdamW/SOAP bias correction — the same counter the Deferred
        // arm feeds nsl_bias_correction_inv, so a mixed table's two arms
        // advance their bias corrections in lockstep.
        let opt_step_f = builder.ins().fcvt_from_sint(cl_types::F64, opt_step);
        let beta1_for_bc = builder.ins().f64const(fase_plan.recipe.beta1);
        let beta2_for_bc = builder.ins().f64const(fase_plan.recipe.beta2);
        let bc1_inv = self.compile_call_by_name(
            builder,
            "nsl_bias_correction_inv",
            &[beta1_for_bc, opt_step],
        )?;
        let bc2_inv = self.compile_call_by_name(
            builder,
            "nsl_bias_correction_inv",
            &[beta2_for_bc, opt_step],
        )?;

        // ── 2. Phase A sum_sq loop + clip factor (two_phase_clip only) ──
        let clip_factor = if fase_plan.two_phase_clip {
            let Some(accum) = accum_list else {
                return Err(crate::error::CodegenError::new(
                    "two_phase_clip requires accum_list to be Some".to_string(),
                ));
            };

            let pa_tot_var = state.new_variable();
            builder.declare_var(pa_tot_var, cl_types::F64);
            let pa_zero = builder.ins().f64const(0.0);
            builder.def_var(pa_tot_var, pa_zero);
            let pa_i_var = state.new_variable();
            builder.declare_var(pa_i_var, cl_types::I64);
            let pa_i_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(pa_i_var, pa_i_zero);

            let pa_hdr = builder.create_block();
            let pa_body = builder.create_block();
            let pa_exit = builder.create_block();
            builder.ins().jump(pa_hdr, &[]);
            builder.switch_to_block(pa_hdr);
            let pa_i = builder.use_var(pa_i_var);
            let pa_cont =
                builder.ins().icmp(IntCC::SignedLessThan, pa_i, num_params_val);
            builder.ins().brif(pa_cont, pa_body, &[], pa_exit, &[]);
            builder.switch_to_block(pa_body);
            builder.seal_block(pa_body);

            let pa_mpart =
                self.compile_call_by_name(builder, "nsl_list_get", &[accum, pa_i])?;
            // Non-hook path: the standard accumulation loop was skipped for
            // this (stepping) micro-batch, so fuse its accumulation here —
            // scaled window-mean for EVERY param regardless of mode (the
            // FullBuffer arm switched to the scaled convention under
            // two-phase clip in stmt.rs's ga_fullbuf). When the source-AD
            // hook is active, accumulation already happened during adjoint
            // lowering and grads_list is a null sentinel — skip.
            if !fase_hook_active {
                let Some(grads) = grads_list else {
                    return Err(crate::error::CodegenError::new(
                        "two_phase_clip without FASE hook requires grads_list".to_string(),
                    ));
                };
                let pa_grad =
                    self.compile_call_by_name(builder, "nsl_list_get", &[grads, pa_i])?;
                self.fase_emit_accumulate(
                    builder,
                    pa_mpart,
                    pa_grad,
                    fase_plan.recipe.accum_scale,
                )?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[pa_grad])?;
            }
            let pa_sq =
                self.compile_call_by_name(builder, "nsl_tensor_sum_sq", &[pa_mpart])?;
            let pa_tot_cur = builder.use_var(pa_tot_var);
            let pa_tot_new = builder.ins().fadd(pa_tot_cur, pa_sq);
            builder.def_var(pa_tot_var, pa_tot_new);
            let pa_i_next = builder.ins().iadd_imm(pa_i, 1);
            builder.def_var(pa_i_var, pa_i_next);
            builder.ins().jump(pa_hdr, &[]);
            builder.switch_to_block(pa_exit);
            builder.seal_block(pa_hdr);
            builder.seal_block(pa_exit);

            // Free the grads_list wrapper consumed above (non-hook only).
            if !fase_hook_active {
                if let Some(grads) = grads_list {
                    self.compile_call_by_name(builder, "nsl_list_free", &[grads])?;
                }
            }

            let total_sq = builder.use_var(pa_tot_var);
            let norm = builder.ins().sqrt(total_sq);
            let eps_v = builder.ins().f64const(1e-6_f64);
            let denom = builder.ins().fadd(norm, eps_v);
            let tau_v = builder.ins().f64const(grad_clip_threshold);
            let ratio = builder.ins().fdiv(tau_v, denom);
            let one_f = builder.ins().f64const(1.0_f64);
            let cf = builder.ins().fmin(one_f, ratio);
            Some(cf)
        } else {
            None
        };

        // ── 3. Unified per-param loop with mode dispatch ──
        let opt_i_var = state.new_variable();
        builder.declare_var(opt_i_var, cl_types::I64);
        let opt_zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(opt_i_var, opt_zero);

        let hdr = builder.create_block();
        let body = builder.create_block();
        let exit = builder.create_block();
        builder.ins().jump(hdr, &[]);
        builder.switch_to_block(hdr);
        let opt_i = builder.use_var(opt_i_var);
        let cont = builder.ins().icmp(IntCC::SignedLessThan, opt_i, num_params_val);
        builder.ins().brif(cont, body, &[], exit, &[]);
        builder.switch_to_block(body);
        builder.seal_block(body);

        // Common per-param loads (theta, s1, s2) — shared by both paths.
        let theta =
            self.compile_call_by_name(builder, "nsl_list_get", &[param_list, opt_i])?;
        let s1 =
            self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, opt_i])?;
        let s2 = if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, opt_i])?
        } else {
            s1
        };

        // Per-iteration mode dispatch.
        let deferred_blk = builder.create_block();
        let fullbuf_blk = builder.create_block();
        let iter_join = builder.create_block();
        self.emit_fase_mode_branch(
            builder,
            mode_table_base,
            opt_i,
            deferred_blk,
            fullbuf_blk,
        );

        // ── Deferred path: m_partial-driven fused step ──
        // Only emit the fused-recipe body when the GLOBAL plan is Deferred:
        // a FullBuffer-global plan (Lion/Unknown/allow_v_approx=false) has a
        // G-referencing update recipe that `fase_emit_final_step` rejects at
        // compile time, and its mode table can never contain a Deferred
        // byte (`plan_with_overrides` clamps Deferred requests to FullBuffer
        // with FaseModeInfeasible) — the arm is structurally dead at
        // runtime, so emit it empty.
        builder.switch_to_block(deferred_blk);
        builder.seal_block(deferred_blk);
        if let (Some(accum), crate::fase::FaseMode::Deferred) = (accum_list, fase_plan.mode) {
            let m_partial =
                self.compile_call_by_name(builder, "nsl_list_get", &[accum, opt_i])?;
            if let Some(cf) = clip_factor {
                self.compile_call_by_name(
                    builder,
                    "nsl_tensor_mul_scalar_inplace",
                    &[m_partial, cf],
                )?;
            }
            // CPDT precision wrapping is now THREADED through this unified-dispatch
            // path. `cpdt_precision_dtypes.is_some()` ⇔ the caller built FP16/INT8
            // dtype lists for m/v; `fase_emit_final_step` wraps the F32 update with
            // nsl_tensor_cast/cast_into (and its own `m_ptr != v_ptr` SGD guard
            // prevents double-cast on single-state optimizers). When None, the
            // wrap branch is skipped and the emitted IR is byte-identical to the
            // pre-threading hardcoded `false`. The matching `wrapped_path_active`
            // gate at stmt.rs:3533 is what actually decides whether the caller
            // passes Some vs None — relaxing that gate is the S4 step that
            // activates the wrap end-to-end on the WGGO path.
            self.fase_emit_final_step(
                builder,
                theta,
                s1,
                m_partial,
                s2,
                &fase_plan.recipe,
                Some((bc1_inv, bc2_inv)),
                cpdt_precision_dtypes.is_some(),
            )?;
        }
        builder.ins().jump(iter_join, &[]);

        // ── FullBuffer path: stdlib optimizer dispatch ──
        builder.switch_to_block(fullbuf_blk);
        builder.seal_block(fullbuf_blk);
        let grad =
            self.compile_call_by_name(builder, "nsl_list_get", &[opt_grads, opt_i])?;
        // Two-phase clip: apply the shared clip factor to this param's
        // accumulated (window-mean) gradient before the stdlib step — the
        // exact mirror of the Deferred arm's m_partial scaling. Without
        // this, a FullBuffer-mode param under a mixed table would step
        // with the UNCLIPPED gradient.
        if let Some(cf) = clip_factor {
            self.compile_call_by_name(
                builder,
                "nsl_tensor_mul_scalar_inplace",
                &[grad, cf],
            )?;
        }
        // S5: CPDT FP16/INT8 wrap envelope is THREADED through the FullBuffer
        // sub-arm. Pre-S5 this arm hardcoded `wrap_precision=false`, so a
        // mixed mode table that routed any FP16-tier param through here
        // would have fed FP16 buffers to the F32 stdlib FFI (silent
        // corruption). The structural mode-table guard at stmt.rs (added in
        // the S4 review-fix) refused FP16 allocation whenever the table
        // contained ANY FullBuffer byte; with this arm now wrapping, that
        // guard can be lifted.
        self.emit_stdlib_optim_call(
            builder,
            optimizer_name,
            opt_fn,
            theta,
            grad,
            s1,
            s2,
            lr,
            momentum_const,
            dampening_const,
            weight_decay_const,
            nesterov_const,
            beta1_const,
            beta2_const,
            eps_const,
            step_count_var,
            cpdt_precision_dtypes.is_some(),
            Some(opt_step_f),
        )?;
        // The Deferred arm's fase_emit_final_step zeroes m_partial via its
        // recipe epilogue; the stdlib call does not. Zero this param's
        // accumulation buffer so the next window starts fresh — without
        // this, FullBuffer-mode params under a mode table accumulated
        // gradients ACROSS windows (the monolithic path's post-optimizer
        // cleanup loop never runs on the unified-dispatch path).
        if accum_list.is_some() {
            self.compile_call_by_name(builder, "nsl_tensor_zero_inplace", &[grad])?;
        }
        builder.ins().jump(iter_join, &[]);

        // ── Join + loop tail ──
        builder.switch_to_block(iter_join);
        builder.seal_block(iter_join);
        let one_tail = builder.ins().iconst(cl_types::I64, 1);
        let next = builder.ins().iadd(opt_i, one_tail);
        builder.def_var(opt_i_var, next);
        builder.ins().jump(hdr, &[]);
        builder.seal_block(hdr);

        builder.switch_to_block(exit);
        builder.seal_block(exit);
        state.current_block = Some(exit);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{plan, FaseConfig, FaseOptimizer};

    #[test]
    fn accumulate_stub_now_returns_ok() {
        let p = plan(&FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        });
        assert!(emit_deferred_accumulate(&p).is_ok());
    }

    #[test]
    fn final_step_free_function_is_a_marker() {
        // The real emission lives on Compiler::fase_emit_final_step; this free
        // function is kept only for future non-Compiler callers.
        let p = plan(&FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        });
        // Task 8: free-function now returns Ok (stub promoted to wired).
        assert!(emit_deferred_final_step(&p).is_ok());
    }
}
