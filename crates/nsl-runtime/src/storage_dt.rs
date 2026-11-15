//! dtype-explicit elementwise ops for bf16 activation storage.
//!
//! Under a train block's `precision="bf16"` the compiler decides each value's
//! storage dtype (f32 or bf16) and lowers the ops that read or write a bf16
//! value to these entry points, naming the output dtype explicitly. Each
//! operand may be f32 or bf16 independently; arithmetic is f32 and a bf16
//! result is rounded once, to nearest even, when stored.
//!
//! This is not implicit promotion. The language still refuses an op on two
//! known, different dtypes, and the ordinary entry points (`nsl_tensor_mul`,
//! ...) still refuse mixed operands (`require_same_dtype`). These exist for
//! the compiler's own storage plan, which states every output dtype.
//!
//! **Fast path:** every operand GPU-resident on one device, contiguous, and
//! (for two or more operands) one shape; then one storage-generic KIR kernel
//! (`crate::cuda::storage_kernels`). **Anything else** — a host tensor, a
//! strided view, a broadcast — widens the bf16 operands to f32, runs the f32
//! op, and narrows the result if a bf16 output was asked for. That is the
//! same value the fast path computes for an elementwise op (the f32 op rounds
//! to f32, the narrowing rounds once more to bf16, as the kernel's store
//! does), so it is a slow path, not a different arithmetic; it is counted
//! ([`nsl_storage_dt_fallback_count`]) and announced once.
//!
//! None of these records on the autodiff tape. A source-AD train step does
//! not record, and a call while the tape is recording is refused rather than
//! left out of the gradient silently.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use nsl_abi::wire::dt_op;

use crate::tensor::{NslTensor, DTYPE_BF16, DTYPE_F32};

static FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// How many dtype-explicit calls took the widen/compute/narrow path instead
/// of a storage-generic kernel.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_storage_dt_fallback_count() -> i64 {
    FALLBACKS.load(Ordering::Relaxed) as i64
}

/// The program's activation storage, as `main` declared it (a train block's
/// `precision="bf16"`); f32 unless declared.
static ACTIVATION_STORAGE: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(DTYPE_F32);

/// Declare the program's activation storage. Compiled `main` calls it, beside
/// the matmul configuration, only for a `precision="bf16"` program, and the
/// banner it prints is what a harness checks to know the arm it measured
/// really stored activations bf16.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_set_activation_storage(dtype: i64) {
    let tag = out_tag("nsl_set_activation_storage", dtype);
    ACTIVATION_STORAGE.store(tag, Ordering::Relaxed);
    if tag == DTYPE_BF16 {
        crate::nsl_log!(
            INFO,
            "nsl-matmul",
            "[nsl-matmul] activation storage: bf16 (train precision=\"bf16\": the planned activations \
             and activation gradients are stored bf16, each rounded once from f32 arithmetic; \
             parameters, their gradients, the optimizer state and the residual stream stay f32)"
        );
    }
}

/// The activation storage `main` declared (`DTYPE_F32` or `DTYPE_BF16`).
pub fn activation_storage() -> u16 {
    ACTIVATION_STORAGE.load(Ordering::Relaxed)
}

fn refuse_tape(op: &str) {
    if crate::autodiff::is_recording() {
        crate::fatal::die(
            crate::fatal::Fatal::Unsupported,
            &format!(
                "{op}: a bf16-storage op was called while the autodiff tape is recording. \
                 These ops are emitted only for source-AD train steps (precision=\"bf16\" \
                 requires --source-ad) and do not record themselves; under the tape the \
                 gradient through this op would be lost"
            ),
        );
    }
}

/// The dtype tag an `out_dtype` argument names. Only f32 and bf16.
fn out_tag(op: &str, out_dtype: i64) -> u16 {
    match u16::try_from(out_dtype) {
        Ok(DTYPE_F32) => DTYPE_F32,
        Ok(DTYPE_BF16) => DTYPE_BF16,
        _ => crate::fatal::die(
            crate::fatal::Fatal::UnsupportedDtype,
            &format!("{op}: output dtype {out_dtype} is not a storage dtype (f32 = 1, bf16 = 3)"),
        ),
    }
}

fn check_operand(op: &str, ptr: i64) -> &'static NslTensor {
    let t = NslTensor::from_ptr_ref(ptr);
    if t.dtype != DTYPE_F32 && t.dtype != DTYPE_BF16 {
        crate::fatal::die(
            crate::fatal::Fatal::UnsupportedDtype,
            &format!("{op}: operand dtype {} is not a storage dtype (f32 = 1, bf16 = 3)", t.dtype),
        );
    }
    t
}

/// Whether `ts` can take the storage-generic kernel: one device, all on it,
/// contiguous, non-empty, one shape.
#[cfg(feature = "cuda")]
fn fast_path(ts: &[&NslTensor]) -> bool {
    let first = ts[0];
    first.device > 0
        && first.len > 0
        && first.len < (1i64 << 31)
        && ts.iter().all(|t| t.device == first.device && t.is_contiguous() && t.shape_eq(first))
}

fn announce_fallback(op: &str) {
    FALLBACKS.fetch_add(1, Ordering::Relaxed);
    if !ANNOUNCED.swap(true, Ordering::Relaxed) {
        crate::nsl_log!(
            WARN,
            "bf16-storage",
            "{op}: an operand is on the host, strided or broadcast, so this bf16-storage op \
             widens to f32, computes and narrows (same values, slower). Further fallbacks are \
             counted, not printed (nsl_storage_dt_fallback_count)"
        );
    }
}

/// An owned f32 reference to `ptr`'s values: `ptr` itself (refcount bumped)
/// when it is f32, else a widened copy. Free it with `nsl_tensor_free`.
fn widened(ptr: i64) -> i64 {
    let t = NslTensor::from_ptr_ref(ptr);
    if t.dtype == DTYPE_F32 {
        t.refcount.fetch_add(1, Ordering::SeqCst);
        return ptr;
    }
    #[cfg(feature = "cuda")]
    if t.device > 0 && !t.is_contiguous() && let Some(w) = widened_dense_view(t) {
        return w;
    }
    crate::tensor::precision_cast::convert_untaped(ptr, DTYPE_F32)
}

/// A strided bf16 GPU view whose elements are exactly the `len` contiguous
/// elements from its data pointer, in some order (a transpose or other
/// permutation of a contiguous tensor; the weight-gradient GEMM's `x^T`):
/// widen that block and give the f32 copy the same shape and strides. The
/// generic path would make the view contiguous first, which the GPU's f32
/// strided copy refuses for a 2-byte element.
#[cfg(feature = "cuda")]
fn widened_dense_view(t: &NslTensor) -> Option<i64> {
    let nd = t.ndim as usize;
    if nd == 0 || t.len <= 0 {
        return None;
    }
    let shape: Vec<i64> = (0..nd).map(|i| unsafe { *t.shape.add(i) }).collect();
    let strides: Vec<i64> = (0..nd).map(|i| unsafe { *t.strides.add(i) }).collect();
    // Dense: ordered by stride (size-1 dims carry no stride information),
    // each stride is the product of the sizes of the dims inside it.
    let mut dims: Vec<usize> = (0..nd).filter(|&i| shape[i] != 1).collect();
    dims.sort_by_key(|&i| strides[i]);
    let mut expect = 1i64;
    for &i in &dims {
        if strides[i] != expect {
            return None;
        }
        expect *= shape[i];
    }
    if expect != t.len {
        return None;
    }
    let n = t.len as usize;
    crate::cuda::inner::set_oom_context("storage_dt widen");
    let buf = crate::cuda::inner::alloc_managed(n * 4);
    crate::cuda::gpu_cast_raw_bf16_to_f32(t.data as u64, buf as u64, n);
    let sh = crate::memory::checked_alloc(nd * std::mem::size_of::<i64>()) as *mut i64;
    let st = crate::memory::checked_alloc(nd * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..nd {
        unsafe {
            *sh.add(i) = shape[i];
            *st.add(i) = strides[i];
        }
    }
    let out = Box::new(NslTensor::new(buf, sh, st, nd as i64, t.len, t.device, DTYPE_F32, 1, 0));
    Some(Box::into_raw(out) as i64)
}

/// `r` (owned, f32) in `tag`: `r` itself, or a narrowed copy (and `r` freed).
fn narrowed(r: i64, tag: u16) -> i64 {
    if tag == DTYPE_F32 {
        return r;
    }
    let n = crate::tensor::precision_cast::convert_untaped(r, tag);
    crate::tensor::nsl_tensor_free(r);
    n
}

/// `a op b` with the output in `out_dtype`; `op` is an `nsl_abi::wire::dt_op`
/// binary code (`ADD`, `SUB`, `MUL`, `DIV`).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_binary_dt(op: i64, a: i64, b: i64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_binary_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    let (ta, tb) = (check_operand(NAME, a), check_operand(NAME, b));
    let kop = match op {
        dt_op::ADD => nsl_kir::kernels::elementwise::BinaryOp::Add,
        dt_op::SUB => nsl_kir::kernels::elementwise::BinaryOp::Sub,
        dt_op::MUL => nsl_kir::kernels::elementwise::BinaryOp::Mul,
        dt_op::DIV => nsl_kir::kernels::elementwise::BinaryOp::Div,
        other => crate::fatal::die(crate::fatal::Fatal::Unsupported, &format!("{NAME}: unknown binary op code {other}")),
    };
    #[cfg(feature = "cuda")]
    if fast_path(&[ta, tb]) {
        return crate::cuda::storage_kernels::binary(kop, a, b, storage(tag));
    }
    let _ = (ta, tb, kop);
    announce_fallback(NAME);
    let (wa, wb) = (widened(a), widened(b));
    let r = match op {
        dt_op::ADD => crate::tensor::arithmetic::nsl_tensor_add(wa, wb, 0),
        dt_op::SUB => crate::tensor::arithmetic::nsl_tensor_sub(wa, wb, 0),
        dt_op::MUL => crate::tensor::arithmetic::nsl_tensor_mul(wa, wb, 0),
        _ => crate::tensor::arithmetic::nsl_tensor_div(wa, wb, 0),
    };
    crate::tensor::nsl_tensor_free(wa);
    crate::tensor::nsl_tensor_free(wb);
    narrowed(r, tag)
}

/// `f(a)` with the output in `out_dtype`; `op` is an `nsl_abi::wire::dt_op`
/// unary code.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_unary_dt(op: i64, a: i64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_unary_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    let ta = check_operand(NAME, a);
    use nsl_kir::kernels::elementwise::UnaryOp as U;
    let (kop, f32_op): (U, extern "C" fn(i64) -> i64) = match op {
        dt_op::NEG => (U::Neg, crate::tensor::arithmetic::nsl_tensor_neg),
        dt_op::RELU => (U::Relu, crate::tensor::activation::nsl_tensor_relu),
        dt_op::EXP => (U::Exp, crate::tensor::activation::nsl_tensor_exp),
        dt_op::LOG => (U::Log, crate::tensor::activation::nsl_tensor_log),
        dt_op::SQRT => (U::Sqrt, crate::tensor::activation::nsl_tensor_sqrt),
        dt_op::ABS => (U::Abs, crate::tensor::activation::nsl_tensor_abs),
        dt_op::SIGMOID => (U::Sigmoid, crate::tensor::activation::nsl_tensor_sigmoid),
        dt_op::SILU => (U::Silu, crate::tensor::activation::nsl_tensor_silu),
        dt_op::SIN => (U::Sin, crate::tensor::trig::nsl_tensor_sin),
        dt_op::COS => (U::Cos, crate::tensor::trig::nsl_tensor_cos),
        other => crate::fatal::die(crate::fatal::Fatal::Unsupported, &format!("{NAME}: unknown unary op code {other}")),
    };
    #[cfg(feature = "cuda")]
    if fast_path(&[ta]) {
        return crate::cuda::storage_kernels::unary(kop, a, storage(tag));
    }
    let _ = (ta, kop);
    announce_fallback(NAME);
    let wa = widened(a);
    let r = f32_op(wa);
    crate::tensor::nsl_tensor_free(wa);
    narrowed(r, tag)
}

/// The SwiGLU gate adjoint `grad * up * silu'(input)` (source AD's
/// `nsl_tensor_swiglu_gate_backward`) with the output in `out_dtype`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_swiglu_gate_backward_dt(grad: i64, up: i64, input: i64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_swiglu_gate_backward_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    let ts = [check_operand(NAME, grad), check_operand(NAME, up), check_operand(NAME, input)];
    #[cfg(feature = "cuda")]
    if fast_path(&ts) {
        return crate::cuda::storage_kernels::swiglu_gate(grad, up, input, storage(tag));
    }
    let _ = ts;
    announce_fallback(NAME);
    let (wg, wu, wx) = (widened(grad), widened(up), widened(input));
    let r = crate::tensor::activation::nsl_tensor_swiglu_gate_backward(wg, wu, wx);
    for w in [wg, wu, wx] {
        crate::tensor::nsl_tensor_free(w);
    }
    narrowed(r, tag)
}

/// `rotate_half(a)` (`neg != 0`: its negation, the RoPE backward's fused
/// form) with the output in `out_dtype`. Moves and negates values, so a
/// bf16-to-bf16 call is exact.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rotate_half_dt(a: i64, neg: i64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_rotate_half_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    let ta = check_operand(NAME, a);
    let last = if ta.ndim == 0 { 0 } else { unsafe { *ta.shape.add(ta.ndim as usize - 1) } };
    #[cfg(feature = "cuda")]
    if fast_path(&[ta]) && last > 0 && last % 2 == 0 {
        let op = if neg != 0 {
            nsl_kir::kernels::elementwise::RotateHalfOp::Neg
        } else {
            nsl_kir::kernels::elementwise::RotateHalfOp::Plain
        };
        return crate::cuda::storage_kernels::rotate_half(op, a, storage(tag));
    }
    let _ = last;
    announce_fallback(NAME);
    let wa = widened(a);
    let r = if neg != 0 {
        crate::tensor::shape_ops::nsl_tensor_rotate_half_neg(wa)
    } else {
        crate::tensor::shape_ops::nsl_tensor_rotate_half(wa)
    };
    crate::tensor::nsl_tensor_free(wa);
    narrowed(r, tag)
}

/// RMSNorm over the last dimension with the output in `out_dtype`: `x` f32 or
/// bf16, `gamma` f32. Statistics are f32 and a bf16 output is rounded once:
/// the norm output that bf16 activation storage hands to the GEMMs.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rmsnorm_dt(x: i64, gamma: i64, eps: f64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_rmsnorm_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    let tx = check_operand(NAME, x);
    let tg = NslTensor::from_ptr_ref(gamma);
    let cols = if tx.ndim == 0 { 0 } else { unsafe { *tx.shape.add(tx.ndim as usize - 1) } };
    #[cfg(feature = "cuda")]
    if fast_path(&[tx])
        && tg.dtype == DTYPE_F32
        && tg.device == tx.device
        && tg.is_contiguous()
        && cols > 0
        && tg.len == cols
    {
        return crate::cuda::storage_kernels::rmsnorm(x, gamma, eps as f32, storage(tag));
    }
    let _ = (tg, cols);
    announce_fallback(NAME);
    let wx = widened(x);
    let r = crate::tensor::nsl_tensor_rmsnorm(wx, gamma, eps);
    crate::tensor::nsl_tensor_free(wx);
    narrowed(r, tag)
}

/// `a @ b` with the result stored as `out_dtype`: bf16 operands (an f32
/// operand is rounded to bf16 once: a parameter from the weight-cast cache),
/// f32 accumulation, one rounding of the output when it is bf16.
///
/// Fast path: the shapes one 2-D GEMM expresses (`b` 2-D, contiguous or a
/// transposed view; `a` contiguous with its leading dims collapsed into rows,
/// or a 2-D transposed view), on one device. Anything else widens, runs the
/// f32 `nsl_tensor_matmul` (under the BF16 math mode: bf16 operands, f32
/// output) and narrows, counted.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_matmul_dt(a: i64, b: i64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_matmul_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    check_operand(NAME, a);
    check_operand(NAME, b);
    #[cfg(feature = "cuda")]
    if let Some(c) = crate::cuda::storage_gemm::matmul(a, b, tag) {
        return c;
    }
    announce_fallback(NAME);
    let (wa, wb) = (widened(a), widened(b));
    let r = crate::tensor::arithmetic::nsl_tensor_matmul(wa, wb, 0);
    crate::tensor::nsl_tensor_free(wa);
    crate::tensor::nsl_tensor_free(wb);
    narrowed(r, tag)
}

#[cfg(feature = "cuda")]
fn storage(tag: u16) -> nsl_kir::kernels::storage::Storage {
    if tag == DTYPE_BF16 { nsl_kir::kernels::storage::Storage::Bf16 } else { nsl_kir::kernels::storage::Storage::F32 }
}

/// `reduce_to_shape(g, like)` with the result stored as `out_dtype`: the
/// gradient of a bf16 activation summed back to its shape (grouped-query
/// attention's dK/dV over the repeated heads). The identity (same shape, same
/// dtype) returns `g` itself with a reference, as `nsl_tensor_reduce_to_shape`
/// does; otherwise `g` is widened, reduced in f32 (the deterministic f32 sum)
/// and narrowed once.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_reduce_to_shape_dt(g: i64, like: i64, out_dtype: i64) -> i64 {
    const NAME: &str = "nsl_tensor_reduce_to_shape_dt";
    refuse_tape(NAME);
    let tag = out_tag(NAME, out_dtype);
    let tg = check_operand(NAME, g);
    let tl = NslTensor::from_ptr_ref(like);
    if tg.dtype == tag && tg.shape_eq(tl) {
        tg.refcount.fetch_add(1, Ordering::SeqCst);
        return g;
    }
    let w = widened(g);
    let r = crate::tensor::ad_ops::nsl_tensor_reduce_to_shape(w, like);
    crate::tensor::nsl_tensor_free(w);
    narrowed(r, tag)
}
