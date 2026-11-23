//! Fused SwiGLU for source AD: `swiglu(g, u) = silu(g) * u` and its adjoint.
//!
//! The compiler rewrites a forward `silu(g) * u` into one `swiglu` op
//! (`nsl_codegen::source_ad::fuse_swiglu_forward`), whose adjoint is the gate
//! branch (`nsl_tensor_swiglu_gate_backward*`, unchanged) and the up branch
//! `p̄ * silu(g)` (`nsl_tensor_swiglu_up_backward*`), or both from one launch
//! (`nsl_tensor_swiglu_backward*`). None of these is a user builtin.
//!
//! On the GPU, with every operand on one device, contiguous, one shape and
//! f32 or bf16, one KIR kernel (`nsl_kir::kernels::swiglu`) does the work:
//! f32 arithmetic in the unfused chain's order, every output rounded once to
//! its storage. Everywhere else (the host, a broadcast, another dtype, and a
//! recording tape) it is the decomposition itself, each op recording itself:
//! the same values in f32, more launches. A bf16 operand on that path is
//! widened and the result narrowed (the `storage_dt` slow path's contract),
//! and counted by `nsl_storage_dt_fallback_count`.
//!
//! With f32 storage the fused kernels are bit-identical to the chain they
//! replace. With bf16 storage the unfused chain rounds `silu(g)` to bf16 when
//! it stores it and the kernels do not; `NSL_SWIGLU_ROUND_SILU=1` (a
//! diagnostic) makes them round it, which reproduces the unfused bf16 chain
//! bit for bit.

use crate::tensor::{DTYPE_BF16, DTYPE_F32, NslTensor};

/// `NSL_SWIGLU_ROUND_SILU=1`: round `silu(g)` to bf16 inside the fused
/// kernels when `g` is bf16, as the unfused chain's store does. Read once.
#[cfg(feature = "cuda")]
pub(crate) fn round_silu() -> bool {
    static ROUND: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ROUND.get_or_init(|| std::env::var("NSL_SWIGLU_ROUND_SILU").ok().as_deref() == Some("1"))
}

/// Whether `ts` (and an output of `out`) take the fused kernel.
#[cfg(feature = "cuda")]
fn fused(ts: &[&NslTensor], out: u16) -> bool {
    let first = ts[0];
    !crate::autodiff::is_recording()
        && (out == DTYPE_F32 || out == DTYPE_BF16)
        && first.device > 0
        && first.len > 0
        && first.len < (1i64 << 31)
        && ts.iter().all(|t| {
            t.device == first.device
                && (t.dtype == DTYPE_F32 || t.dtype == DTYPE_BF16)
                && t.is_contiguous()
                && t.shape_eq(first)
        })
}

#[cfg(feature = "cuda")]
fn storage(tag: u16) -> nsl_kir::kernels::storage::Storage {
    use nsl_kir::kernels::storage::Storage;
    if tag == DTYPE_BF16 {
        Storage::Bf16
    } else {
        Storage::F32
    }
}

/// An owned f32 handle on `t`'s values: `t` (a new reference) or a widened copy.
fn widened(t: i64) -> i64 {
    let tt = NslTensor::from_ptr_ref(t);
    if tt.dtype == DTYPE_BF16 {
        return crate::tensor::precision_cast::convert_untaped(t, DTYPE_F32);
    }
    tt.refcount
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    t
}

/// `r` (owned) stored as `out`, freeing `r` if it had to be converted.
fn narrowed(r: i64, out: u16) -> i64 {
    if out == DTYPE_BF16 && NslTensor::from_ptr_ref(r).dtype != DTYPE_BF16 {
        let n = crate::tensor::precision_cast::convert_untaped(r, DTYPE_BF16);
        crate::tensor::nsl_tensor_free(r);
        return n;
    }
    r
}

/// `silu(g)` through the ordinary entry point (it records on a recording tape).
fn silu(g: i64) -> i64 {
    crate::tensor::activation::nsl_tensor_silu(g)
}

/// The decomposed forward `silu(g) * u`, broadcasting as the multiply does.
fn forward_decomposed(g: i64, u: i64) -> i64 {
    let s = silu(g);
    let p = crate::tensor::arithmetic::nsl_tensor_mul(s, u, 0);
    crate::tensor::nsl_tensor_free(s);
    p
}

/// The decomposed up adjoint `grad * silu(g)` (not reduced: the compiler
/// reduces it to `u`'s shape, as it reduced the unfused product).
fn up_decomposed(grad: i64, g: i64) -> i64 {
    let s = silu(g);
    let r = crate::tensor::arithmetic::nsl_tensor_mul(grad, s, 0);
    crate::tensor::nsl_tensor_free(s);
    r
}

/// Is a bf16 operand or result involved (then the slow path widens and
/// narrows, and is counted)?
fn any_bf16(ts: &[i64], out: u16) -> bool {
    out == DTYPE_BF16
        || ts
            .iter()
            .any(|&t| NslTensor::from_ptr_ref(t).dtype == DTYPE_BF16)
}

/// Run `f` on f32 views of `ts` and store its result as `out`.
fn slow(name: &str, ts: &[i64], out: u16, f: impl FnOnce(&[i64]) -> i64) -> i64 {
    if any_bf16(ts, out) {
        crate::storage_dt::announce_fallback(name);
    }
    let w: Vec<i64> = ts.iter().map(|&t| widened(t)).collect();
    let r = f(&w);
    for t in w {
        crate::tensor::nsl_tensor_free(t);
    }
    narrowed(r, out)
}

fn forward(g: i64, u: i64, out: Option<u16>) -> i64 {
    const NAME: &str = "nsl_tensor_swiglu";
    let (tg, tu) = (NslTensor::from_ptr_ref(g), NslTensor::from_ptr_ref(u));
    let out = out.unwrap_or(tg.dtype);
    #[cfg(feature = "cuda")]
    if fused(&[tg, tu], out) {
        return crate::cuda::storage_kernels::swiglu_forward(g, u, storage(out), round_silu());
    }
    let _ = (tg, tu);
    slow(NAME, &[g, u], out, |w| forward_decomposed(w[0], w[1]))
}

fn up_backward(grad: i64, g: i64, out: Option<u16>) -> i64 {
    const NAME: &str = "nsl_tensor_swiglu_up_backward";
    let (tgr, tg) = (NslTensor::from_ptr_ref(grad), NslTensor::from_ptr_ref(g));
    let out = out.unwrap_or(tgr.dtype);
    #[cfg(feature = "cuda")]
    if fused(&[tgr, tg], out) {
        return crate::cuda::storage_kernels::swiglu_up_backward(
            grad,
            g,
            storage(out),
            round_silu(),
        );
    }
    let _ = (tgr, tg);
    slow(NAME, &[grad, g], out, |w| up_decomposed(w[0], w[1]))
}

/// `[dgate, dup]` as an `NslList` (the caller takes both and frees the list).
fn backward(grad: i64, u: i64, g: i64, out: Option<u16>) -> i64 {
    let (tgr, tu, tg) = (
        NslTensor::from_ptr_ref(grad),
        NslTensor::from_ptr_ref(u),
        NslTensor::from_ptr_ref(g),
    );
    #[cfg(feature = "cuda")]
    let out_tag = out.unwrap_or(tgr.dtype);
    let list = crate::list::nsl_list_new();
    #[cfg(feature = "cuda")]
    if fused(&[tgr, tu, tg], out_tag) {
        let (dg, du) = crate::cuda::storage_kernels::swiglu_backward(
            grad,
            u,
            g,
            storage(out_tag),
            round_silu(),
        );
        crate::list::nsl_list_push(list, dg);
        crate::list::nsl_list_push(list, du);
        return list;
    }
    let _ = (tgr, tu, tg);
    let dg = match out {
        Some(t) => crate::storage_dt::nsl_tensor_swiglu_gate_backward_dt(grad, u, g, i64::from(t)),
        None => crate::tensor::activation::nsl_tensor_swiglu_gate_backward(grad, u, g),
    };
    let du = up_backward(grad, g, out);
    crate::list::nsl_list_push(list, dg);
    crate::list::nsl_list_push(list, du);
    list
}

/// `[dgate, dup, p]` as an `NslList`: [`backward`] and the forward value
/// `p = silu(gate) * up` (`forward`'s value), from one launch on the fast path.
fn backward_recompute(grad: i64, u: i64, g: i64, out: Option<u16>) -> i64 {
    let (tgr, tu, tg) = (
        NslTensor::from_ptr_ref(grad),
        NslTensor::from_ptr_ref(u),
        NslTensor::from_ptr_ref(g),
    );
    #[cfg(feature = "cuda")]
    let out_tag = out.unwrap_or(tgr.dtype);
    // One storage for all three outputs: without a requested dtype the
    // adjoints are `grad`'s and the forward value is `gate`'s (as the ops
    // they stand for store them), so the kernel serves only when they agree.
    #[cfg(feature = "cuda")]
    if (out.is_some() || tg.dtype == tgr.dtype) && fused(&[tgr, tu, tg], out_tag) {
        let (dg, du, p) = crate::cuda::storage_kernels::swiglu_backward_recompute(
            grad,
            u,
            g,
            storage(out_tag),
            round_silu(),
        );
        let list = crate::list::nsl_list_new();
        for t in [dg, du, p] {
            crate::list::nsl_list_push(list, t);
        }
        return list;
    }
    let _ = (tgr, tu, tg);
    let list = backward(grad, u, g, out);
    // The forward value in the forward op's own storage: the gate's dtype
    // (or the requested one), as the `swiglu` op it stands in for.
    crate::list::nsl_list_push(list, forward(g, u, out));
    list
}

fn out_tag(name: &str, out_dtype: i64) -> u16 {
    match u16::try_from(out_dtype) {
        Ok(DTYPE_F32) => DTYPE_F32,
        Ok(DTYPE_BF16) => DTYPE_BF16,
        _ => crate::fatal::die(
            crate::fatal::Fatal::UnsupportedDtype,
            &format!("{name}: output dtype {out_dtype} is not a storage dtype (f32 = 1, bf16 = 3)"),
        ),
    }
}

/// `silu(gate) * up`, stored as `gate` is.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu(gate: i64, up: i64) -> i64 {
    forward(gate, up, None)
}

/// [`nsl_tensor_swiglu`] with the result stored as `out_dtype` (bf16
/// activation storage: f32 = 1, bf16 = 3).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_dt(gate: i64, up: i64, out_dtype: i64) -> i64 {
    crate::storage_dt::refuse_tape("nsl_tensor_swiglu_dt");
    forward(gate, up, Some(out_tag("nsl_tensor_swiglu_dt", out_dtype)))
}

/// The up branch's adjoint `grad * silu(gate)`, stored as `grad` is.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_up_backward(grad: i64, gate: i64) -> i64 {
    up_backward(grad, gate, None)
}

/// [`nsl_tensor_swiglu_up_backward`] with the result stored as `out_dtype`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_up_backward_dt(grad: i64, gate: i64, out_dtype: i64) -> i64 {
    crate::storage_dt::refuse_tape("nsl_tensor_swiglu_up_backward_dt");
    up_backward(
        grad,
        gate,
        Some(out_tag("nsl_tensor_swiglu_up_backward_dt", out_dtype)),
    )
}

/// Both adjoints from one read of `grad`, `up` and `gate`: an `NslList`
/// `[dgate, dup]` (`dgate` is `nsl_tensor_swiglu_gate_backward`'s value,
/// `dup` `nsl_tensor_swiglu_up_backward`'s), each stored as `grad` is.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_backward(grad: i64, up: i64, gate: i64) -> i64 {
    backward(grad, up, gate, None)
}

/// [`nsl_tensor_swiglu_backward`] with both results stored as `out_dtype`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_backward_dt(
    grad: i64,
    up: i64,
    gate: i64,
    out_dtype: i64,
) -> i64 {
    crate::storage_dt::refuse_tape("nsl_tensor_swiglu_backward_dt");
    backward(
        grad,
        up,
        gate,
        Some(out_tag("nsl_tensor_swiglu_backward_dt", out_dtype)),
    )
}

/// Both adjoints and the forward value `p = silu(gate) * up` from one read
/// of `grad`, `up` and `gate`: an `NslList` `[dgate, dup, p]`. The lowering
/// emits it where a checkpointed step would otherwise recompute `p` for the
/// down projection's weight gradient.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_backward_recompute(grad: i64, up: i64, gate: i64) -> i64 {
    backward_recompute(grad, up, gate, None)
}

/// [`nsl_tensor_swiglu_backward_recompute`] with every result stored as `out_dtype`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_backward_recompute_dt(
    grad: i64,
    up: i64,
    gate: i64,
    out_dtype: i64,
) -> i64 {
    crate::storage_dt::refuse_tape("nsl_tensor_swiglu_backward_recompute_dt");
    backward_recompute(
        grad,
        up,
        gate,
        Some(out_tag(
            "nsl_tensor_swiglu_backward_recompute_dt",
            out_dtype,
        )),
    )
}
