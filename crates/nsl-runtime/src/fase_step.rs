//! p9: fused per-parameter FASE-Deferred AdamW/Adam optimizer step.
//!
//! `fase_emit_final_step` used to interpret the AdamW `UpdateProgram` op-by-op:
//! ~15 kernel launches + 3 DtoD copies + ~10 transient alloc/frees **per
//! parameter per optimizer step**. This FFI performs the whole update in ONE
//! kernel launch per parameter (GPU) or one fused loop (CPU), BIT-EXACT with
//! the interpreted program:
//!
//!   m  = rn(rn(β₁·m) + rn((1-β₁)·mp))          [ScalarMulAdd]
//!   v  = rn(rn(β₂·v) + rn((1-β₂)·rn(mp·mp)))   [SquaredAccumulate]
//!   m̂  = rn(m·bc1_inv);  v̂ = rn(v·bc2_inv)     [ScalarMulByBc ×2]
//!   t  = rn(sqrt(v̂) + ε)                       [SqrtPlusEps]
//!   u  = m̂ / t                                 [Div]
//!   θ  = rn(θ + rn(rn(-lr·u) + rn(-lr·wd·θ)))  [Update; wd term skipped when wd==0,
//!                                               exactly like the emitted program]
//!
//! GPU: `FASE_FUSED_ADAMW_STEP_F32_PTX` mirrors each decomposed kernel's
//! rounding (`.rn`, `sqrt.rn`, `div.approx` — the same instructions the
//! separate SQRT/DIV kernels use), with all scalars converted f64→f32 at this
//! boundary exactly as every `nsl_tensor_*_scalar` op does. CPU: the same
//! order in f64. `m_partial` is only READ — the caller's zero-for-next-window
//! emission is unchanged.
//!
//! Codegen admission (`stmt_fase.rs`) pattern-matches the exact AdamW/Adam
//! program shape and falls back to the interpreted path for anything else, so
//! this FFI can assert its preconditions loudly instead of guessing.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::tensor::NslTensor;

/// Count of fused optimizer-step launches — always live (cheap atomic), used
/// by the differential gate's anti-vacuity check via the
/// `NSL_FASE_FUSED_COUNTER=1` atexit report (see `args.rs`) and the
/// `nsl_fase_fused_step_count` getter.
pub static FASE_FUSED_STEP_COUNT: AtomicU64 = AtomicU64::new(0);

/// In-process numeric getter (same family as `nsl_gpu_peak_allocated_bytes`):
/// lets gates assert the fused path actually fired without stderr scraping.
#[no_mangle]
pub extern "C" fn nsl_fase_fused_step_count() -> i64 {
    FASE_FUSED_STEP_COUNT.load(Ordering::Relaxed) as i64
}

/// Global sum-of-squares over every tensor in an `NslList` — the FASE
/// two-phase-clip Phase A norm, batched.
///
/// The per-tensor emission called `nsl_tensor_sum_sq` once per parameter,
/// and each call is a full pipeline drain (the f64 result crosses back to
/// the host synchronously): 74 drains per optimizer step at Coder-50M.
/// This routes uniform device-f32 dense lists through
/// `gpu_sum_sq_many_f32` — per-tensor kernels, ONE drain — and everything
/// else through the same per-tensor `nsl_tensor_sum_sq` it replaces.
///
/// NUMERICS: the batched device kernel accumulates each tensor in f64
/// (`nsl_sum_sq_f64_acc_f32`), where `nsl_tensor_sum_sq`'s GPU fast path
/// accumulates in f32. This matches the main `nsl_clip_grad_norm` device
/// path's choice — a clip decision is a comparison, so accumulation drift
/// can flip it, and f64 is the accumulation every other GPU reduction here
/// uses — but it is NOT bit-identical to the per-tensor emission. Codegen
/// gates the substitution on `NSL_FASE_BATCH_SUMSQ != "0"` (compile-time)
/// so bisections can restore the old path.
#[no_mangle]
pub extern "C" fn nsl_fase_sum_sq_list(mp_list: i64) -> f64 {
    let mps = crate::list::NslList::from_ptr(mp_list);
    let count = mps.len as usize;
    if count == 0 {
        return 0.0;
    }
    let ptrs: Vec<i64> = (0..count).map(|i| unsafe { *mps.data.add(i) }).collect();
    #[cfg(feature = "cuda")]
    {
        let uniform_gpu_f32 = ptrs.iter().all(|&p| {
            p != 0 && {
                let t = unsafe { &*(p as *const NslTensor) };
                t.device > 0 && t.dtype == 1 && t.is_contiguous()
            }
        });
        if uniform_gpu_f32 {
            return crate::cuda::gpu_sum_sq_many_f32(&ptrs);
        }
    }
    ptrs.iter()
        .map(|&p| crate::tensor::nsl_tensor_sum_sq(p))
        .sum()
}

/// Fused FASE-Deferred AdamW/Adam final step for one parameter.
///
/// `theta/m/v/mp` must be four same-length contiguous tensors on the same
/// device: all GPU-f32 or all CPU-f64 (the only configurations the FASE
/// working tensors take after the precision/offload envelopes resolve).
/// Anything else is a loud precondition failure — codegen admission should
/// have routed it to the interpreted path.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fase_fused_adamw_step(
    theta_ptr: i64,
    m_ptr: i64,
    v_ptr: i64,
    mp_ptr: i64,
    lr: f64,
    beta1: f64,
    one_minus_beta1: f64,
    beta2: f64,
    one_minus_beta2: f64,
    eps: f64,
    wd: f64,
    bc1_inv: f64,
    bc2_inv: f64,
) {
    let th = unsafe { &*(theta_ptr as *const NslTensor) };
    let m = unsafe { &*(m_ptr as *const NslTensor) };
    let v = unsafe { &*(v_ptr as *const NslTensor) };
    let mp = unsafe { &*(mp_ptr as *const NslTensor) };

    let n = th.len;
    assert!(
        m.len == n && v.len == n && mp.len == n,
        "fase_fused_step: length mismatch (theta={}, m={}, v={}, mp={})",
        n, m.len, v.len, mp.len
    );
    assert!(
        th.is_contiguous() && m.is_contiguous() && v.is_contiguous() && mp.is_contiguous(),
        "fase_fused_step: all tensors must be contiguous"
    );
    assert!(
        m.device == th.device && v.device == th.device && mp.device == th.device,
        "fase_fused_step: device mismatch (theta={}, m={}, v={}, mp={})",
        th.device, m.device, v.device, mp.device
    );
    let has_wd = wd != 0.0;

    if th.device > 0 {
        #[cfg(feature = "cuda")]
        {
            assert!(
                th.dtype == 1 && m.dtype == 1 && v.dtype == 1 && mp.dtype == 1,
                "fase_fused_step: GPU path requires f32 tensors (theta={}, m={}, v={}, mp={})",
                th.dtype, m.dtype, v.dtype, mp.dtype
            );
            // f64→f32 conversions mirror the decomposed FFI boundary exactly:
            // each scalar op received an f64 and did `as f32`; neg_lr and
            // neg_lr_wd were computed in f64 by codegen (`-(lr)`, `-(lr)*wd`)
            // before that conversion.
            crate::cuda::gpu_fase_fused_adamw_step(
                theta_ptr,
                m_ptr,
                v_ptr,
                mp_ptr,
                n as usize,
                beta1 as f32,
                one_minus_beta1 as f32,
                beta2 as f32,
                one_minus_beta2 as f32,
                eps as f32,
                (-lr) as f32,
                ((-lr) * wd) as f32,
                bc1_inv as f32,
                bc2_inv as f32,
                has_wd,
            );
            FASE_FUSED_STEP_COUNT.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("CUDA support not compiled");
        }
    }

    // CPU: same op order as the interpreted program (Rust emits no FMA for
    // separate mul/add, so each op rounds independently like the decomposed
    // per-op loops). The generic tensor ops compute NATIVELY in the tensor's
    // dtype on CPU (f64 tensors → f64 math; f32 tensors → f32 math with each
    // scalar converted `as f32` — e.g. `nsl_tensor_mul_scalar`'s `sf = s as
    // f32` loop), so both uniform-dtype configurations are mirrored here. The
    // SqrtPlusEps `+0.0` defensive copy is an identity for v̂ ≥ +0 (v is a
    // non-negative EMA of squares) and is skipped.
    if th.dtype == 0 && m.dtype == 0 && v.dtype == 0 && mp.dtype == 0 {
        let neg_lr = -lr;
        let neg_lr_wd = (-lr) * wd;
        let (td, md, vd, mpd) = (
            th.data as *mut f64,
            m.data as *mut f64,
            v.data as *mut f64,
            mp.data as *const f64,
        );
        for i in 0..n as usize {
            unsafe {
                let mp_i = *mpd.add(i);
                let t1 = *md.add(i) * beta1;
                let t2 = mp_i * one_minus_beta1;
                let m_new = t1 + t2;
                *md.add(i) = m_new;
                let sq = mp_i * mp_i;
                let ssq = sq * one_minus_beta2;
                let vdec = *vd.add(i) * beta2;
                let v_new = vdec + ssq;
                *vd.add(i) = v_new;
                let mh = m_new * bc1_inv;
                let vh = v_new * bc2_inv;
                let t = vh.sqrt() + eps;
                let u = mh / t;
                let mut adj = u * neg_lr;
                if has_wd {
                    let wdt = *td.add(i) * neg_lr_wd;
                    adj += wdt;
                }
                *td.add(i) += adj;
            }
        }
    } else if th.dtype == 1 && m.dtype == 1 && v.dtype == 1 && mp.dtype == 1 {
        // Uniform CPU f32 (e.g. an f32 model trained on CPU): the decomposed
        // CPU ops run f32-native loops, so mirror them in f32 with the same
        // per-scalar `as f32` conversions the FFI boundary performs.
        let b1 = beta1 as f32;
        let omb1 = one_minus_beta1 as f32;
        let b2 = beta2 as f32;
        let omb2 = one_minus_beta2 as f32;
        let epsf = eps as f32;
        let bc1 = bc1_inv as f32;
        let bc2 = bc2_inv as f32;
        let neg_lr = (-lr) as f32;
        let neg_lr_wd = ((-lr) * wd) as f32;
        let (td, md, vd, mpd) = (
            th.data as *mut f32,
            m.data as *mut f32,
            v.data as *mut f32,
            mp.data as *const f32,
        );
        for i in 0..n as usize {
            unsafe {
                let mp_i = *mpd.add(i);
                let t1 = *md.add(i) * b1;
                let t2 = mp_i * omb1;
                let m_new = t1 + t2;
                *md.add(i) = m_new;
                let sq = mp_i * mp_i;
                let ssq = sq * omb2;
                let vdec = *vd.add(i) * b2;
                let v_new = vdec + ssq;
                *vd.add(i) = v_new;
                let mh = m_new * bc1;
                let vh = v_new * bc2;
                let t = vh.sqrt() + epsf;
                let u = mh / t;
                let mut adj = u * neg_lr;
                if has_wd {
                    let wdt = *td.add(i) * neg_lr_wd;
                    adj += wdt;
                }
                *td.add(i) += adj;
            }
        }
    } else {
        panic!(
            "fase_fused_step: CPU path requires uniform f64 or f32 tensors \
             (theta={}, m={}, v={}, mp={})",
            th.dtype, m.dtype, v.dtype, mp.dtype
        );
    }
    FASE_FUSED_STEP_COUNT.fetch_add(1, Ordering::Relaxed);
}


/// Fusion-queue item 1: MULTI-TENSOR fused AdamW/Adam final step — the
/// whole non-clip Deferred optimizer loop in ONE call. Takes the train
/// block's param/m/v/m_partial NslLists and performs, per parameter,
/// exactly what the per-param loop emitted: the fused step (bit-identical
/// arithmetic) plus the shared tail's m_partial zeroing.
///
/// GPU device-f32 quadruples batch into ONE pointer-table kernel launch
/// (per-element math byte-for-byte the single-step kernel's, so results are
/// BIT-IDENTICAL to N launches). Anything else — CPU tensors, mixed
/// placement — falls back to the single-step FFI + zero, preserving every
/// existing numeric contract. Emitted only for configurations where the
/// per-param loop would have taken the fused path for EVERY param
/// (admission mirrored in codegen; see stmt.rs). Kill-switch:
/// NSL_FASE_MULTI_STEP=0 at compile time keeps the legacy loop.
/// `mp_scale` scales every m_partial read (`g = rn(mp[i] * mp_scale)`),
/// folding the two-phase-clip Phase B pre-scale into the same launch;
/// pass 1.0 for the unclipped path (branched around in-kernel, so 1.0 is
/// exactly the pre-mp_scale behaviour, bit for bit).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fase_fused_adamw_step_multi(
    params_list: i64,
    m_list: i64,
    v_list: i64,
    mp_list: i64,
    lr: f64,
    beta1: f64,
    one_minus_beta1: f64,
    beta2: f64,
    one_minus_beta2: f64,
    eps: f64,
    wd: f64,
    bc1_inv: f64,
    bc2_inv: f64,
    // AdamW parameter groups.
    //   wd_exempt_list       — NslList of per-param decay-exempt flags,
    //                          parallel to `params_list`; 0 when `no_decay`
    //                          was not configured, in which case every param
    //                          takes `wd` and this is byte-identical to
    //                          before the feature.
    //   wd_exempt_non_rank2  — 1 when `no_decay` included "vector" → exempt
    //                          anything not rank-2. Resolved per param by
    //                          `nsl_optim_param_wd`, the single rule.
    wd_exempt_list: i64,
    wd_exempt_non_rank2: i64,
    // Clip pre-scale, folded into the m_partial read. Independent of the two
    // above: parameter groups decide each param's λ, this scales the gradient
    // every param sees. Both must survive — dropping either compiles fine and
    // silently disables a shipped feature.
    mp_scale: f64,
) {
    let params = crate::list::NslList::from_ptr(params_list);
    let ms = crate::list::NslList::from_ptr(m_list);
    let vs = crate::list::NslList::from_ptr(v_list);
    let mps = crate::list::NslList::from_ptr(mp_list);
    let count = params.len as usize;
    assert!(
        ms.len as usize == count && vs.len as usize == count && mps.len as usize == count,
        "fase_fused_step_multi: list length mismatch"
    );
    // AdamW parameter groups: the exempt flags are indexed positionally
    // against `params_list`, so a length mismatch would silently shift every
    // exemption onto the wrong parameter.
    if wd_exempt_list != 0 {
        let ex = crate::list::NslList::from_ptr(wd_exempt_list);
        assert!(
            ex.len as usize == count,
            "fase_fused_step_multi: wd_exempt_list has {} entries for {} params — \
             the decay-exempt flags must be parallel to the parameter list",
            ex.len,
            count
        );
    }
    // Per-param λ, resolved by the one rule in optim_groups.rs. Empty list =>
    // every param gets `wd`, so this collapses to the previous scalar.
    let param_wd = |i: usize, tp: i64| -> f64 {
        if wd_exempt_list == 0 {
            return wd;
        }
        let ex = crate::list::NslList::from_ptr(wd_exempt_list);
        let flag = unsafe { *ex.data.add(i) };
        crate::optim_groups::nsl_optim_param_wd(tp, flag, wd_exempt_non_rank2, wd)
    };

    // Item 8: collect eligible parameters as `UpdateDesc`s and bucket by
    // `UpdateKey` before launching, instead of accumulating one flat pointer
    // table. See `UpdateKey` for why `device` has to be part of the key.
    #[cfg(feature = "cuda")]
    let mut descs: Vec<UpdateDesc> = Vec::new();
    #[cfg(feature = "cuda")]
    let mut batched_thetas: Vec<u64> = Vec::new();

    for i in 0..count {
        let (tp, mp_, vp, ap) = unsafe {
            (
                *params.data.add(i),
                *ms.data.add(i),
                *vs.data.add(i),
                *mps.data.add(i),
            )
        };
        assert!(
            tp != 0 && mp_ != 0 && vp != 0 && ap != 0,
            "fase_fused_step_multi: null tensor at param {i} (ZeRO/conditional-v \
             configurations must use the per-param loop)"
        );
        #[cfg(feature = "cuda")]
        {
            let th = unsafe { &*(tp as *const NslTensor) };
            let m = unsafe { &*(mp_ as *const NslTensor) };
            let v = unsafe { &*(vp as *const NslTensor) };
            let a = unsafe { &*(ap as *const NslTensor) };
            let uniform_gpu_f32 = th.device > 0
                && [th, m, v, a].iter().all(|t| {
                    t.device == th.device && t.dtype == 1 && t.is_contiguous() && t.len == th.len
                })
                && th.len <= u32::MAX as i64;
            // Review A1: two param entries can alias ONE theta storage
            // (tied weights). The legacy loop updated them sequentially;
            // concurrent grid slices would race last-writer-wins. Route
            // aliases to the sequential fallback arm (k is small — linear
            // scan is fine).
            //
            // Kept GLOBAL across all buckets rather than per-bucket, which
            // would batch slightly more: two aliasing thetas can only land in
            // different buckets by differing in device or dtype, and an alias
            // across devices is not a thing this can reason about safely.
            let theta_aliased = batched_thetas.contains(&(th.data as u64));
            if uniform_gpu_f32 && !theta_aliased {
                batched_thetas.push(th.data as u64);
                descs.push(UpdateDesc {
                    theta: th.data as u64,
                    m: m.data as u64,
                    v: v.data as u64,
                    mp: a.data as u64,
                    len: th.len as u32,
                    key: UpdateKey {
                        device: th.device,
                        dtype: th.dtype,
                        wd_bits: param_wd(i, tp).to_bits(),
                    },
                });
                continue;
            }
        }
        // Fallback arm: single fused step (its own preconditions assert
        // loudly) + the shared tail's m_partial zero. The clip pre-scale is
        // applied the way the legacy per-param loop did it — an in-place
        // scale of m_partial before the step (same rounding as the kernel
        // fold: one rn multiply).
        if mp_scale != 1.0 {
            crate::tensor::nsl_tensor_mul_scalar_inplace(ap, mp_scale);
        }
        nsl_fase_fused_adamw_step(
            tp, mp_, vp, ap, lr, beta1, one_minus_beta1, beta2, one_minus_beta2, eps,
            param_wd(i, tp), bc1_inv, bc2_inv,
        );
        crate::tensor::nsl_tensor_zero_inplace(ap);
    }

    #[cfg(feature = "cuda")]
    {
        // One launch per bucket. With a single CUDA device — every
        // configuration this repo currently builds — there is exactly one
        // bucket and this is byte-for-byte the previous behaviour.
        let buckets = bucket_updates(&descs);
        let n_buckets = buckets.len();
        for (key, members) in buckets {
            let k = members.len();
            if k == 0 {
                continue;
            }
            // Each bucket carries its OWN λ (AdamW parameter groups); with no
            // `no_decay` every param shares one λ and there is one bucket, so
            // this is the previous scalar exactly.
            let bucket_wd = f64::from_bits(key.wd_bits);
            let thetas: Vec<u64> = members.iter().map(|d| d.theta).collect();
            let ms_p: Vec<u64> = members.iter().map(|d| d.m).collect();
            let vs_p: Vec<u64> = members.iter().map(|d| d.v).collect();
            let mps_p: Vec<u64> = members.iter().map(|d| d.mp).collect();
            let lens: Vec<u32> = members.iter().map(|d| d.len).collect();
            crate::cuda::gpu_fase_fused_adamw_step_multi(
                &thetas,
                &ms_p,
                &vs_p,
                &mps_p,
                &lens,
                beta1 as f32,
                one_minus_beta1 as f32,
                beta2 as f32,
                one_minus_beta2 as f32,
                eps as f32,
                (-lr) as f32,
                ((-lr) * bucket_wd) as f32,
                bc1_inv as f32,
                bc2_inv as f32,
                // `bucket_wd`, not the flat `wd`: each bucket carries its own λ
                // from the parameter groups, so testing the scalar would tell a
                // decay-exempt bucket it still has weight decay.
                bucket_wd != 0.0,
                mp_scale as f32,
            );
            // Keep the fused-step counter's per-param semantics (gates read it).
            FASE_FUSED_STEP_COUNT.fetch_add(k as u64, Ordering::Relaxed);
            // One-time anti-vacuity marker for the e2e gates.
            static MARKED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !MARKED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[fase-multi] batched {k} params into one fused AdamW launch \
                     ({n_buckets} bucket(s))"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        nsl_tensor_add_inplace, nsl_tensor_add_scalar, nsl_tensor_copy_data, nsl_tensor_div,
        nsl_tensor_free, nsl_tensor_mul, nsl_tensor_mul_scalar, nsl_tensor_mul_scalar_inplace,
        nsl_tensor_sqrt,
    };
    #[cfg(feature = "cuda")]
    use crate::tensor::nsl_tensor_to_device;

    fn make_f64(data: &[f64]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = *v };
        }
        ptr
    }
    #[cfg(feature = "cuda")]
    fn make_f32(data: &[f32]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 1);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() {
            unsafe { *t.data_f32().add(i) = *v };
        }
        ptr
    }

    /// The interpreted `UpdateProgram`, reproduced through the SAME FFIs the
    /// emitted IR calls, with the same flag/alias choreography
    /// (`stmt_fase.rs::fase_emit_final_step`). This is what the fused step
    /// must match bit-for-bit.
    #[allow(clippy::too_many_arguments)]
    fn decomposed_reference(
        theta: i64, m: i64, v: i64, mp: i64,
        lr: f64, b1: f64, omb1: f64, b2: f64, omb2: f64, eps: f64, wd: f64,
        bc1: f64, bc2: f64,
    ) {
        // ScalarMulAdd: m = β₁·m + (1-β₁)·mp
        let t1 = nsl_tensor_mul_scalar(m, b1, 0);
        let t2 = nsl_tensor_mul_scalar(mp, omb1, 0);
        nsl_tensor_add_inplace(t1, t2);
        nsl_tensor_free(t2);
        nsl_tensor_copy_data(m, t1);
        nsl_tensor_free(t1);
        // SquaredAccumulate: v = β₂·v + (1-β₂)·mp²
        let sq = nsl_tensor_mul(mp, mp, 0);
        // Relinquish transfers our sq ref into the callee; ssq carries the
        // single live ref (same choreography as the emitted IR).
        let ssq = nsl_tensor_mul_scalar(sq, omb2, 0b0000_0001);
        nsl_tensor_mul_scalar_inplace(v, b2);
        nsl_tensor_add_inplace(v, ssq);
        nsl_tensor_free(ssq);
        // ScalarMulByBc ×2
        let mh = nsl_tensor_mul_scalar(m, bc1, 0);
        let vh = nsl_tensor_mul_scalar(v, bc2, 0);
        // SqrtPlusEps: tmp = sqrt(v̂ + 0.0-copy) + ε
        let vh_copy = nsl_tensor_add_scalar(vh, 0.0, 0);
        let sqrt_val = nsl_tensor_sqrt(vh_copy);
        nsl_tensor_free(vh_copy);
        let tmp = nsl_tensor_add_scalar(sqrt_val, eps, 0b0000_0001); // consumes sqrt_val
        // Div: u = m̂ / tmp
        let u = nsl_tensor_div(mh, tmp, 0);
        nsl_tensor_free(tmp);
        // Update: θ += -lr·u [+ -lr·wd·θ]
        let adj = nsl_tensor_mul_scalar(u, -lr, 0);
        if wd != 0.0 {
            let wdt = nsl_tensor_mul_scalar(theta, -lr * wd, 0);
            nsl_tensor_add_inplace(adj, wdt);
            nsl_tensor_free(wdt);
        }
        nsl_tensor_add_inplace(theta, adj);
        nsl_tensor_free(adj);
        for p in [u, mh, vh] {
            nsl_tensor_free(p);
        }
    }

    fn inputs_f64(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let th: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.13).sin() * 2.0).collect();
        let m: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.07).cos() * 0.01).collect();
        let v: Vec<f64> = (0..n).map(|i| (((i as f64) * 0.05).sin() * 0.001).abs()).collect();
        let mp: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.11).sin() * 0.5).collect();
        (th, m, v, mp)
    }

    const SCALARS: (f64, f64, f64, f64, f64, f64, f64, f64, f64) = (
        0.001,          // lr
        0.9,            // beta1
        0.1,            // 1-beta1 (as codegen computes it)
        0.999,          // beta2
        0.001,          // 1-beta2 approximation of codegen value
        1e-8,           // eps
        0.01,           // wd
        1.0526315789473684, // bc1_inv-ish (runtime f64)
        1.000500250125,     // bc2_inv-ish
    );

    #[test]
    fn fused_step_cpu_f64_matches_decomposed() {
        let n = 257;
        let (th, m, v, mp) = inputs_f64(n);
        let (lr, b1, omb1, b2, omb2, eps, wd, bc1, bc2) = SCALARS;
        for wd_case in [wd, 0.0] {
            let (rt, rm, rv, rmp) = (make_f64(&th), make_f64(&m), make_f64(&v), make_f64(&mp));
            decomposed_reference(rt, rm, rv, rmp, lr, b1, omb1, b2, omb2, eps, wd_case, bc1, bc2);
            let (ft, fm, fv, fmp) = (make_f64(&th), make_f64(&m), make_f64(&v), make_f64(&mp));
            nsl_fase_fused_adamw_step(ft, fm, fv, fmp, lr, b1, omb1, b2, omb2, eps, wd_case, bc1, bc2);
            for (name, r, f) in [("theta", rt, ft), ("m", rm, fm), ("v", rv, fv), ("mp", rmp, fmp)] {
                let (rr, ff) = (NslTensor::from_ptr(r), NslTensor::from_ptr(f));
                for i in 0..n {
                    let (a, b) = unsafe { (*rr.data_f64().add(i), *ff.data_f64().add(i)) };
                    assert_eq!(
                        a.to_bits(), b.to_bits(),
                        "cpu {name} mismatch (wd={wd_case}) at {i}: {a} vs {b}"
                    );
                }
            }
            for p in [rt, rm, rv, rmp, ft, fm, fv, fmp] {
                nsl_tensor_free(p);
            }
        }
    }

    /// Uniform CPU-f32 tensors (an f32 model trained on CPU — the FASE
    /// equivalence fixtures hit exactly this) must match the f32-native
    /// decomposed loops bit-for-bit.
    #[test]
    fn fused_step_cpu_f32_matches_decomposed() {
        fn make_f32_cpu(data: &[f32]) -> i64 {
            let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 1);
            let t = NslTensor::from_ptr(ptr);
            for (i, v) in data.iter().enumerate() {
                unsafe { *t.data_f32().add(i) = *v };
            }
            ptr
        }
        let n = 257;
        let (th, m, v, mp) = inputs_f64(n);
        let f32s = |xs: &[f64]| xs.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let (thf, mf, vf, mpf) = (f32s(&th), f32s(&m), f32s(&v), f32s(&mp));
        let (lr, b1, omb1, b2, omb2, eps, wd, bc1, bc2) = SCALARS;
        for wd_case in [wd, 0.0] {
            let (rt, rm, rv, rmp) =
                (make_f32_cpu(&thf), make_f32_cpu(&mf), make_f32_cpu(&vf), make_f32_cpu(&mpf));
            decomposed_reference(rt, rm, rv, rmp, lr, b1, omb1, b2, omb2, eps, wd_case, bc1, bc2);
            let (ft, fm, fv, fmp) =
                (make_f32_cpu(&thf), make_f32_cpu(&mf), make_f32_cpu(&vf), make_f32_cpu(&mpf));
            nsl_fase_fused_adamw_step(ft, fm, fv, fmp, lr, b1, omb1, b2, omb2, eps, wd_case, bc1, bc2);
            for (name, r, f) in [("theta", rt, ft), ("m", rm, fm), ("v", rv, fv), ("mp", rmp, fmp)] {
                let (rr, ff) = (NslTensor::from_ptr(r), NslTensor::from_ptr(f));
                for i in 0..n {
                    let (a, b) = unsafe { (*rr.data_f32().add(i), *ff.data_f32().add(i)) };
                    assert_eq!(
                        a.to_bits(), b.to_bits(),
                        "cpu-f32 {name} mismatch (wd={wd_case}) at {i}: {a} vs {b}"
                    );
                }
            }
            for p in [rt, rm, rv, rmp, ft, fm, fv, fmp] {
                nsl_tensor_free(p);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_step_gpu_f32_matches_decomposed() {
        let n = 257; // 256-block tail guard
        let (th, m, v, mp) = inputs_f64(n);
        let f32s = |xs: &[f64]| xs.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let (lr, b1, omb1, b2, omb2, eps, wd, bc1, bc2) = SCALARS;
        let to_gpu = |d: &[f32]| nsl_tensor_to_device(make_f32(d), 1);
        for wd_case in [wd, 0.0] {
            let (thf, mf, vf, mpf) = (f32s(&th), f32s(&m), f32s(&v), f32s(&mp));
            let (rt, rm, rv, rmp) = (to_gpu(&thf), to_gpu(&mf), to_gpu(&vf), to_gpu(&mpf));
            decomposed_reference(rt, rm, rv, rmp, lr, b1, omb1, b2, omb2, eps, wd_case, bc1, bc2);
            let (ft, fm, fv, fmp) = (to_gpu(&thf), to_gpu(&mf), to_gpu(&vf), to_gpu(&mpf));
            nsl_fase_fused_adamw_step(ft, fm, fv, fmp, lr, b1, omb1, b2, omb2, eps, wd_case, bc1, bc2);
            // GPU f32 -> CPU f64 upcast is lossless: equal upcast == equal bits.
            let up = |p: i64| -> Vec<f64> {
                let c = nsl_tensor_to_device(p, 0);
                let t = NslTensor::from_ptr(c);
                let out = (0..n).map(|i| unsafe { *t.data_f64().add(i) }).collect();
                nsl_tensor_free(c);
                out
            };
            for (name, r, f) in [("theta", rt, ft), ("m", rm, fm), ("v", rv, fv), ("mp", rmp, fmp)] {
                let (a, b) = (up(r), up(f));
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(), b[i].to_bits(),
                        "gpu {name} mismatch (wd={wd_case}) at {i}: {} vs {}",
                        a[i], b[i]
                    );
                }
            }
            for p in [rt, rm, rv, rmp, ft, fm, fv, fmp] {
                nsl_tensor_free(p);
            }
        }
    }
}

// ─── Item 8: the multi-tensor update descriptor table ──────────────────────
//
// The three items below are used by the CUDA launch path and by the CPU unit
// tests. The tests deliberately do NOT require the `cuda` feature — the whole
// point of extracting `build_block_tables` and `bucket_updates` as pure
// functions is that the batching logic is checked on every build, GPU or not.
// So they are compiled unconditionally, and dead-code is silenced only for a
// non-cuda, non-test build where nothing calls them.

/// One parameter's slot in a batched optimizer launch.
///
/// The roadmap called this `UpdateDesc`. It is the unit the batching key is
/// computed FROM and the unit the device tables are built from, so a rule
/// about which parameters may share a launch has exactly one place to live.
///
/// Deliberately NOT `#[repr(C)]`: it never crosses the FFI. The device side
/// receives parallel arrays — four pointer tables and a length table indexed
/// by PARAMETER, plus two tables indexed by BLOCK (see
/// [`build_block_tables`]) — because that is what the kernel indexes. Packing
/// a struct-of-arrays here and scattering it there would put the layout
/// contract in two places.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UpdateDesc {
    pub theta: u64,
    pub m: u64,
    pub v: u64,
    pub mp: u64,
    pub len: u32,
    pub key: UpdateKey,
}

/// What must match for two parameters to share one kernel launch.
///
/// `device` is the field that matters and the one the pre-item-8 code did not
/// consider: `NslTensor::device` is `0 = CPU, 1+ = CUDA device ID`, and the
/// old eligibility test only checked each parameter's four tensors against
/// EACH OTHER (`t.device == th.device`). Two parameters resident on different
/// CUDA devices both passed and were pushed into the same pointer table, so a
/// single launch would have dereferenced pointers from another device's
/// address space.
///
/// Bucketing is a PRECONDITION for handling that, not a fix for it. Nothing
/// in `gpu_fase_fused_adamw_step_multi` selects a device: `ensure_context()`
/// sets the one process-wide context chosen at init, so two buckets would both
/// launch on that context and the second would still dereference the other
/// device's pointers. What bucketing buys is that the parameters are already
/// partitioned by device, so making the launch device-aware becomes a change
/// to one loop rather than a rework of the eligibility test. Whether any of
/// this is reachable today depends on whether anything sets a device ID above
/// 1; the single-GPU paths do not.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct UpdateKey {
    /// CUDA device ID. Parameters on different devices cannot share a launch.
    pub device: u8,
    /// Tensor dtype tag. The batched kernel is f32-only; other dtypes never
    /// reach a bucket today, but the key carries the field so that adding a
    /// bf16 kernel is a new bucket rather than a new eligibility branch.
    pub dtype: u16,
    /// Weight decay, as raw f64 bits so the key stays `Eq`/`Ord`.
    ///
    /// AdamW parameter groups (`no_decay=[...]`) give exempt parameters
    /// λ = 0 while the rest keep the configured λ, and the flat-grid kernel
    /// takes `neg_lr_wd`/`has_wd` as LAUNCH parameters — one value for the
    /// whole grid. So λ has to partition the launches exactly as device and
    /// dtype do. It takes at most two distinct values, so this adds at most
    /// one extra launch, and each parameter still gets bit-identical
    /// arithmetic to a scalar-λ launch at its own λ.
    ///
    /// Bits, not f64: `UpdateKey` derives `Eq`/`Ord` for sort/dedup in
    /// `bucket_updates`, and f64 has neither. λ is only ever compared for
    /// exact equality here, and it is copied from one source value rather
    /// than computed per parameter, so bitwise identity is the right test.
    pub wd_bits: u64,
}

/// Map CUDA blocks to (parameter, element offset) for a FLAT launch grid.
///
/// The batched AdamW kernel used a rectangular grid `(ceil(max_n/block), k)`,
/// giving every parameter enough blocks for the LARGEST one. Measured against
/// Coder-50M's real parameter list (74 params, 38.5M elements, largest 16.4M):
/// 4,736,000 blocks launched to do 150,562 blocks of work — **96.8% exited
/// immediately on the bounds guard**, a 31.5x overshoot. It also capped the
/// parameter count at 65535 (the `grid.y` limit).
///
/// MEASURED (RTX 5070 Ti, CUDA 13.3, Coder-50M shapes, 200 iterations, n=3
/// runs each, spread under 0.2%): rectangular grid 6777.0 us/step, flat grid
/// 1641.8 us/step — a **4.13x speedup** on the batched AdamW step. The A/B
/// drove the SAME kernel and changed only the launch shape, so the difference
/// is block-launch overhead and nothing else.
///
/// `item8_gpu_bench` reproduces the FLAT arm only; the rectangular arm no
/// longer exists in the tree, so the comparison is not re-runnable from the
/// committed code. To reproduce it, make this function emit
/// `max(lens).div_ceil(block)` blocks for every parameter (and widen
/// `total_blocks` to match, or its `debug_assert` fires) and re-run the bench.
///
/// Returns `(block_param, block_base)`, both indexed by block id:
/// `block_param[b]` is the parameter block `b` works on and `block_base[b]`
/// is the element offset of its first thread within that parameter. The
/// kernel then needs two loads and no search.
///
/// Zero-length parameters get no blocks — they have no work, and emitting a
/// block for them would reintroduce exactly the wasted launch this removes.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn build_block_tables(lens: &[u32], block: u32) -> (Vec<u32>, Vec<u32>) {
    assert!(block > 0, "block size must be positive");
    let total_blocks: usize = lens
        .iter()
        .map(|&n| n.div_ceil(block) as usize)
        .sum();
    let mut bparam = Vec::with_capacity(total_blocks);
    let mut bbase = Vec::with_capacity(total_blocks);
    for (j, &n) in lens.iter().enumerate() {
        let nb = n.div_ceil(block);
        for b in 0..nb {
            bparam.push(j as u32);
            bbase.push(b * block);
        }
    }
    debug_assert_eq!(bparam.len(), total_blocks);
    (bparam, bbase)
}

/// Partition parameters into launch buckets by [`UpdateKey`].
///
/// Returns buckets in a deterministic order (sorted by key) so a run is
/// reproducible and a test can assert on the result. Order across buckets does
/// not affect numerics — each parameter is touched by exactly one bucket.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn bucket_updates(descs: &[UpdateDesc]) -> Vec<(UpdateKey, Vec<UpdateDesc>)> {
    let mut keys: Vec<UpdateKey> = descs.iter().map(|d| d.key).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .map(|k| {
            let members: Vec<UpdateDesc> =
                descs.iter().copied().filter(|d| d.key == k).collect();
            (k, members)
        })
        .collect()
}

#[cfg(test)]
mod item8_tests {
    //! Item 8 — CPU-side unit tests for the batching machinery.
    //!
    //! The triage noted that the pre-existing module tests cover only the
    //! single-parameter FFI, so `nsl_fase_fused_adamw_step_multi`'s routing
    //! had no test at all. These run everywhere (no `#[ignore]`, no CUDA)
    //! because `build_block_tables` and `bucket_updates` are pure functions —
    //! which is why they were written as free functions rather than inlined
    //! into the launch path.

    use super::*;

    fn desc(theta: u64, len: u32, device: u8, dtype: u16) -> UpdateDesc {
        desc_wd(theta, len, device, dtype, 0.01)
    }

    fn desc_wd(theta: u64, len: u32, device: u8, dtype: u16, wd: f64) -> UpdateDesc {
        UpdateDesc {
            theta,
            m: theta + 1,
            v: theta + 2,
            mp: theta + 3,
            len,
            key: UpdateKey {
                device,
                dtype,
                wd_bits: wd.to_bits(),
            },
        }
    }

    // ─── bucket_updates: weight decay is part of the key ───────────────────

    /// AdamW parameter groups hand exempt params λ = 0 and the rest the
    /// configured λ, but the flat-grid kernel takes `neg_lr_wd`/`has_wd` as
    /// LAUNCH parameters. So λ must partition launches exactly as device and
    /// dtype do — otherwise one bucket's λ would silently be applied to the
    /// other's parameters.
    #[test]
    fn distinct_weight_decay_splits_into_separate_buckets() {
        let descs = [
            desc_wd(0x1000, 16, 0, 1, 0.1),
            desc_wd(0x2000, 16, 0, 1, 0.0), // exempt (a norm gain)
            desc_wd(0x3000, 16, 0, 1, 0.1),
        ];
        let buckets = bucket_updates(&descs);
        assert_eq!(buckets.len(), 2, "λ must split the launch: {buckets:?}");
        for (key, members) in &buckets {
            let wd = f64::from_bits(key.wd_bits);
            for m in members {
                assert_eq!(
                    f64::from_bits(m.key.wd_bits),
                    wd,
                    "a bucket member carries a different λ than its bucket key"
                );
            }
        }
        let total: usize = buckets.iter().map(|(_, m)| m.len()).sum();
        assert_eq!(total, descs.len(), "bucketing dropped a parameter");
    }

    /// The common case must stay ONE launch: no `no_decay` means one λ.
    #[test]
    fn uniform_weight_decay_stays_a_single_bucket() {
        let descs = [
            desc_wd(0x1000, 16, 0, 1, 0.1),
            desc_wd(0x2000, 32, 0, 1, 0.1),
            desc_wd(0x3000, 64, 0, 1, 0.1),
        ];
        assert_eq!(bucket_updates(&descs).len(), 1);
    }

    // ─── build_block_tables ────────────────────────────────────────────────

    #[test]
    fn block_tables_cover_every_element_exactly_once() {
        let lens = [1u32, 255, 256, 257, 1000, 16_384_000];
        let block = 256u32;
        let (bparam, bbase) = build_block_tables(&lens, block);
        assert_eq!(bparam.len(), bbase.len());

        // Reconstruct the element set each block covers and check it is a
        // partition of every parameter's index range. This is the property the
        // kernel depends on: miss an element and a weight silently stops
        // updating; double-cover one and it updates twice.
        let mut covered: Vec<Vec<bool>> =
            lens.iter().map(|&n| vec![false; n as usize]).collect();
        for (b, (&j, &base)) in bparam.iter().zip(bbase.iter()).enumerate() {
            let j = j as usize;
            assert!(j < lens.len(), "block {b} names parameter {j}, out of range");
            for t in 0..block {
                let idx = (base + t) as usize;
                if idx < lens[j] as usize {
                    assert!(
                        !covered[j][idx],
                        "element {idx} of param {j} covered twice (block {b})"
                    );
                    covered[j][idx] = true;
                }
            }
        }
        for (j, c) in covered.iter().enumerate() {
            let missed = c.iter().filter(|x| !**x).count();
            assert_eq!(missed, 0, "param {j} has {missed} uncovered elements");
        }
    }

    /// Architecture constants read FROM `models/coder50m/model.nsl` at test
    /// time, so the fixture cannot quietly stop describing the model.
    ///
    /// The first version of this hardcoded VOCAB_SIZE=32000, N_KV_HEADS=2 and
    /// D_FF=1376 while claiming to be "Coder-50M's real parameter shapes".
    /// The real values are 49152 / 4 / 1408 — three of five constants wrong,
    /// giving a 38.5M-parameter list labelled as a ~50M model. It also carried
    /// an `assert_eq!(lens.len(), 74, "parameter count changed")` that could
    /// not fire, because the list was built from consts six lines above it:
    /// the assertion promised drift detection against a file it never read.
    /// Parsing the file is what makes the promise true.
    pub(super) fn coder50m_consts() -> std::collections::HashMap<String, i64> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crate is two levels below the repo root")
            .join("models/coder50m/model.nsl");
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut out = std::collections::HashMap::new();
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("const ") else {
                continue;
            };
            let Some((name, val)) = rest.split_once('=') else {
                continue;
            };
            let val = val.split('#').next().unwrap_or("").trim();
            if let Ok(v) = val.parse::<i64>() {
                out.insert(name.trim().to_string(), v);
            }
        }
        for k in ["VOCAB_SIZE", "D_MODEL", "N_LAYERS", "N_HEADS", "N_KV_HEADS", "D_FF"] {
            assert!(
                out.contains_key(k),
                "{k} not found in models/coder50m/model.nsl — the parser or the \
                 model changed, and every number pinned below is unfounded"
            );
        }
        out
    }

    /// Coder-50M's weight-tensor lengths, derived from those constants.
    ///
    /// Covers the tensors AdamW actually batches: the embedding, the final
    /// norm, and per block a norm + `wq/wk/wv/wo` + a norm + `w_gate/w_up/
    /// w_down`. It EXCLUDES the six `full([1])` metadata scalars each
    /// `GroupedQueryAttention` carries (`_d_model`, `_n_heads`, ...): they are
    /// one element each, contribute one block each under either grid, and so
    /// cannot change the conclusion — including them moves the overshoot from
    /// 38.2x to 62.9x, i.e. it would only flatter the result.
    pub(super) fn coder50m_param_lens() -> Vec<u32> {
        let c = coder50m_consts();
        let d = c["D_MODEL"] as u32;
        let ff = c["D_FF"] as u32;
        let heads = c["N_HEADS"] as u32;
        let kv_heads = c["N_KV_HEADS"] as u32;
        let head_dim = d / heads;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let mut lens = vec![c["VOCAB_SIZE"] as u32 * d, d];
        for _ in 0..c["N_LAYERS"] {
            lens.extend_from_slice(&[
                d,            // attn_norm
                d * q_dim,    // wq
                d * kv_dim,   // wk
                d * kv_dim,   // wv
                q_dim * d,    // wo
                d,            // ffn_norm
                d * ff,       // w_gate
                d * ff,       // w_up
                ff * d,       // w_down
            ]);
        }
        lens
    }

    #[test]
    fn block_count_matches_the_useful_minimum() {
        // The whole point: no block that exits immediately.
        let lens = coder50m_param_lens();
        let block = 256u32;
        let (bparam, _) = build_block_tables(&lens, block);
        let useful: usize = lens.iter().map(|&n| n.div_ceil(block) as usize).sum();
        assert_eq!(
            bparam.len(),
            useful,
            "the flat grid must launch exactly the blocks that have work"
        );
    }

    #[test]
    fn flat_grid_is_a_large_reduction_at_coder50m_shapes() {
        // Pins the measurement the change was justified by. The reduction is
        // bounded above by the parameter count (when one parameter dominates,
        // rectangular ~= useful * k), so a small synthetic fixture cannot show
        // it — which is why this derives the real shape list.
        let lens = coder50m_param_lens();
        let block = 256u32;
        let (bparam, _) = build_block_tables(&lens, block);
        let rectangular =
            (lens.iter().max().unwrap().div_ceil(block) as usize) * lens.len();
        let total: u64 = lens.iter().map(|&n| n as u64).sum();

        // Sanity-check the derivation against the model's own header, which
        // says "~50M parameters". A fixture that silently stopped describing
        // the model would land far from this.
        assert!(
            (45_000_000..55_000_000).contains(&total),
            "derived {total} weight elements, which is not a ~50M model — the \
             constants or the field list have drifted"
        );
        assert_eq!(lens.len(), 74, "expected 2 + 8*9 weight tensors");
        assert_eq!(rectangular, 7_274_496);
        assert_eq!(bparam.len(), 190_498);
        let factor = rectangular as f64 / bparam.len() as f64;
        assert!(
            (38.0..38.5).contains(&factor),
            "expected ~38.2x fewer blocks at Coder-50M shapes, got {factor:.2}x"
        );
    }

    #[test]
    fn zero_length_parameters_get_no_blocks() {
        // A zero-length parameter has no work. Emitting a block for it would
        // reintroduce exactly the wasted launch this change removes, and the
        // kernel's bounds guard would discard it anyway.
        let (bparam, _) = build_block_tables(&[0, 256, 0], 256);
        assert_eq!(bparam, vec![1]);
    }

    #[test]
    fn empty_shape_list_yields_no_blocks() {
        let (bparam, bbase) = build_block_tables(&[], 256);
        assert!(bparam.is_empty() && bbase.is_empty());
    }

    // ─── bucket_updates ────────────────────────────────────────────────────

    #[test]
    fn one_device_and_dtype_means_one_bucket() {
        // The single-GPU case, i.e. every configuration this repo builds
        // today. Behaviour must be identical to the pre-item-8 single table.
        let descs: Vec<UpdateDesc> = (0..8).map(|i| desc(0x1000 + i * 0x100, 64, 1, 1)).collect();
        let buckets = bucket_updates(&descs);
        assert_eq!(buckets.len(), 1, "single device+dtype must not split");
        assert_eq!(buckets[0].1.len(), 8);
        // Order within a bucket must be preserved: the pointer tables and the
        // length table are parallel arrays, so a reordering that touched one
        // and not the other would pair a parameter with another's length.
        let order: Vec<u64> = buckets[0].1.iter().map(|d| d.theta).collect();
        let expected: Vec<u64> = descs.iter().map(|d| d.theta).collect();
        assert_eq!(order, expected);
    }

    #[test]
    fn parameters_on_different_devices_never_share_a_launch() {
        // The defect the key exists to prevent. `NslTensor::device` is a CUDA
        // device ID, and the pre-item-8 eligibility test compared each
        // parameter's four tensors only against EACH OTHER — so two params on
        // devices 1 and 2 both passed and were pushed into the same pointer
        // table. One launch would then dereference another device's pointers.
        let descs = vec![
            desc(0x1000, 64, 1, 1),
            desc(0x2000, 64, 2, 1),
            desc(0x3000, 64, 1, 1),
        ];
        let buckets = bucket_updates(&descs);
        assert_eq!(buckets.len(), 2, "two devices must give two buckets");
        for (key, members) in &buckets {
            for d in members {
                assert_eq!(
                    d.key.device, key.device,
                    "a bucket contains a parameter from another device"
                );
            }
        }
        let total: usize = buckets.iter().map(|(_, m)| m.len()).sum();
        assert_eq!(total, 3, "bucketing must not drop or duplicate a parameter");
    }

    #[test]
    fn differing_dtype_splits_buckets() {
        let descs = vec![desc(0x1000, 64, 1, 1), desc(0x2000, 64, 1, 2)];
        assert_eq!(bucket_updates(&descs).len(), 2);
    }

    #[test]
    fn bucketing_is_a_partition() {
        // Every parameter appears in exactly one bucket. Losing one silently
        // stops a weight updating; duplicating one applies AdamW twice.
        let descs: Vec<UpdateDesc> = (0..20)
            .map(|i| desc(0x1000 + i * 0x100, 32 + i as u32, (i % 3) as u8, (i % 2) as u16))
            .collect();
        let buckets = bucket_updates(&descs);
        let mut seen: Vec<u64> = buckets
            .iter()
            .flat_map(|(_, m)| m.iter().map(|d| d.theta))
            .collect();
        seen.sort_unstable();
        let mut expected: Vec<u64> = descs.iter().map(|d| d.theta).collect();
        expected.sort_unstable();
        assert_eq!(seen, expected);
    }

    #[test]
    fn empty_input_yields_no_buckets() {
        assert!(bucket_updates(&[]).is_empty());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod item8_gpu_bench {
    //! Item 8 — measure the flat-grid launch at real shapes.
    //!
    //! In-crate rather than an integration test because
    //! `crate::cuda::gpu_fase_fused_adamw_step_multi` is `pub(crate)` and a
    //! benchmark is not a reason to widen the public surface.
    //!
    //! This is a MEASUREMENT, not a correctness gate. Numerical parity is held
    //! by `crates/nsl-cli/tests/multi_adamw_gate.rs`
    //! (`multi_adamw_gpu_bit_identical_and_fires`), which was verified to fail
    //! on a one-element offset in the block table — so it genuinely exercises
    //! this path rather than passing on a fallback.
    //!
    //! ```bash
    //! cargo test -p nsl-runtime --features cuda --lib item8_gpu_bench \
    //!     -- --ignored --nocapture --test-threads=1
    //! ```

    use super::super::cuda::{gpu_fase_fused_adamw_step_multi, nsl_cuda_init};
    use crate::{nsl_test_cuda_alloc, nsl_test_cuda_free, nsl_test_cuda_h2d};

    /// Reuses the shape list the CPU tests derive from
    /// `models/coder50m/model.nsl`, rather than a second copy of the
    /// constants — a duplicated fixture is exactly how the first version of
    /// this work ended up describing a model that does not exist.
    use super::item8_tests::coder50m_param_lens;

    /// Numerical correctness on UNALIGNED parameter lengths.
    ///
    /// The existing parity gate (`crates/nsl-cli/tests/multi_adamw_gate.rs`)
    /// drives `cuda_graph_gate.nsl`, whose six parameters are 2048/8192/8192/
    /// 8192/8192/2048 elements — every one an exact multiple of the 256-thread
    /// block. The kernel's `i >= n` bounds guard is therefore never taken on
    /// any path that gate covers, so a regression in partial-block handling
    /// would be caught by the CPU table test and by nothing that runs the
    /// actual kernel. Review found that gap; this closes it.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn unaligned_parameter_lengths_are_numerically_correct() {
        if unsafe { nsl_cuda_init() } != 0 {
            eprintln!("skipping: no CUDA device");
            return;
        }
        // Deliberately awkward: 1 element, one short of a block, one over a
        // block boundary, a prime, and a large non-multiple.
        let lens: Vec<u32> = vec![1, 255, 257, 1021, 70_001];
        let k = lens.len();

        // Distinct per-element values so a cross-parameter write or a skipped
        // tail shows up as a wrong number rather than a coincidence.
        let host: Vec<Vec<f32>> = lens
            .iter()
            .enumerate()
            .map(|(j, &n)| {
                (0..n)
                    .map(|i| ((j as f32 + 1.0) * 0.5) + (i as f32) * 1e-4)
                    .collect()
            })
            .collect();

        let mut bufs: Vec<[i64; 4]> = Vec::with_capacity(k);
        for (j, &n) in lens.iter().enumerate() {
            let bytes = n as i64 * 4;
            let mut quad = [0i64; 4];
            for (slot_idx, slot) in quad.iter_mut().enumerate() {
                let d = unsafe { nsl_test_cuda_alloc(bytes) };
                assert!(d != 0, "device alloc failed");
                // theta/m/v/mp all seeded from the same distinct pattern.
                let src: Vec<f32> = if slot_idx == 3 {
                    host[j].iter().map(|x| x * 0.1).collect()
                } else {
                    host[j].clone()
                };
                unsafe { nsl_test_cuda_h2d(d, src.as_ptr() as i64, bytes) };
                *slot = d;
            }
            bufs.push(quad);
        }
        let thetas: Vec<u64> = bufs.iter().map(|q| q[0] as u64).collect();
        let ms: Vec<u64> = bufs.iter().map(|q| q[1] as u64).collect();
        let vs: Vec<u64> = bufs.iter().map(|q| q[2] as u64).collect();
        let mps: Vec<u64> = bufs.iter().map(|q| q[3] as u64).collect();

        let (b1, omb1, b2, omb2) = (0.9f32, 0.1f32, 0.95f32, 0.05f32);
        let (eps, neg_lr, neg_lr_wd, bc1, bc2) =
            (1e-8f32, -1e-3f32, -1e-5f32, 1.0f32, 1.0f32);
        gpu_fase_fused_adamw_step_multi(
            &thetas, &ms, &vs, &mps, &lens, b1, omb1, b2, omb2, eps, neg_lr,
            neg_lr_wd, bc1, bc2, true, 1.0,
        );

        // Same arithmetic as the kernel, in f32, on the host.
        for (j, &n) in lens.iter().enumerate() {
            let mut got = vec![0f32; n as usize];
            unsafe {
                crate::nsl_test_cuda_d2h(
                    got.as_mut_ptr() as i64,
                    bufs[j][0],
                    n as i64 * 4,
                )
            };
            for i in 0..n as usize {
                let theta = host[j][i];
                let m0 = host[j][i];
                let v0 = host[j][i];
                let g = host[j][i] * 0.1;
                let m = m0 * b1 + g * omb1;
                let v = v0 * b2 + (g * g) * omb2;
                let mut upd = (m * bc1) / ((v * bc2).sqrt() + eps) * neg_lr;
                upd += theta * neg_lr_wd;
                let want = theta + upd;
                let err = (got[i] - want).abs() / want.abs().max(1e-3);
                assert!(
                    err < 1e-4,
                    "param {j} (len {n}) element {i}: got {}, want {want} \
                     (rel err {err:.2e}). A wrong value in the LAST partial \
                     block means the bounds guard or the block base is off; a \
                     wrong value elsewhere means cross-parameter indexing.",
                    got[i]
                );
            }
        }

        for q in bufs {
            for d in q {
                unsafe { nsl_test_cuda_free(d) };
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn flat_grid_multi_adamw_step_at_coder50m_shapes() {
        if unsafe { nsl_cuda_init() } != 0 {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let lens = coder50m_param_lens();
        let k = lens.len();
        let total: u64 = lens.iter().map(|&n| n as u64).sum();
        assert_eq!(k, 74, "parameter count changed; re-derive the numbers below");

        let mut bufs: Vec<[i64; 4]> = Vec::with_capacity(k);
        for &n in &lens {
            let bytes = n as i64 * 4;
            let host = vec![0.01f32; n as usize];
            let mut quad = [0i64; 4];
            for slot in quad.iter_mut() {
                let d = unsafe { nsl_test_cuda_alloc(bytes) };
                assert!(d != 0, "device alloc failed");
                unsafe { nsl_test_cuda_h2d(d, host.as_ptr() as i64, bytes) };
                *slot = d;
            }
            bufs.push(quad);
        }
        let thetas: Vec<u64> = bufs.iter().map(|q| q[0] as u64).collect();
        let ms: Vec<u64> = bufs.iter().map(|q| q[1] as u64).collect();
        let vs: Vec<u64> = bufs.iter().map(|q| q[2] as u64).collect();
        let mps: Vec<u64> = bufs.iter().map(|q| q[3] as u64).collect();

        let step = || {
            gpu_fase_fused_adamw_step_multi(
                &thetas, &ms, &vs, &mps, &lens, 0.9, 0.1, 0.95, 0.05, 1e-8, -1e-3, -1e-5,
                1.0, 1.0, true, 1.0,
            );
        };

        // Warm up: the first call builds and uploads the block tables and
        // loads the module. Timing that would measure setup, not the launch.
        for _ in 0..5 {
            step();
        }

        let iters = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            step();
        }
        let per_step_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let block = 256u32;
        let useful: u64 = lens.iter().map(|&n| n.div_ceil(block) as u64).sum();
        let rectangular = (*lens.iter().max().unwrap()).div_ceil(block) as u64 * k as u64;
        eprintln!(
            "multi AdamW @ Coder-50M: k={k}, {total} elems\n  \
             flat grid  : {useful} blocks\n  \
             rectangular: {rectangular} blocks ({:.1}x more)\n  \
             measured   : {per_step_us:.1} us/step over {iters} iters",
            rectangular as f64 / useful as f64
        );

        // Anti-vacuity. Four f32 arrays are read (theta, m, v, m_partial) and
        // FOUR written (m, v, theta, and m_partial's zeroing) = 8 * 4 bytes
        // per element. At 900 GB/s — just above the 5070 Ti's 896 GB/s spec,
        // so an unachievable ceiling and therefore a true lower bound on time
        // — that is a floor no correct kernel can beat. A launch that silently
        // did nothing would land far below it.
        let floor_us = (total * 4 * 8) as f64 / 900e9 * 1e6;
        assert!(
            per_step_us > floor_us * 0.5,
            "measured {per_step_us:.1} us is below half the bandwidth floor \
             ({floor_us:.1} us) — the kernel cannot have touched the data"
        );

        for q in bufs {
            for d in q {
                unsafe { nsl_test_cuda_free(d) };
            }
        }
    }
}
