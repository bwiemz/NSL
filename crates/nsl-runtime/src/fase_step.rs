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
