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
    ) -> Result<(), crate::error::CodegenError> {
        use crate::fase_optimizer::{emit_final_step, Register, UpdateOp};

        let program = emit_final_step(recipe);

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

                // ── SquaredAccumulate: dst = src + scale * operand² ─────────
                // dst and src are always the same register (e.g. V = V + …).
                // Strategy: compute sq=operand², scaled_sq=scale*sq, then
                // add_inplace(dst, scaled_sq).  No copy needed when dst==src.
                UpdateOp::SquaredAccumulate { dst, src, operand, scale } => {
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
                            &format!("ScalarMulByBc src must be M or V, got {:?}", other)
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
                            &format!("ScalarMulByBc dst must be MHat or VHat, got {:?}", other)
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
