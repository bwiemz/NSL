//! The pre-norm block's residual add and the RMSNorm of its sum in one launch
//! (round-2 perf item 6): `h = x + o`, `n = rmsnorm(h, gamma, eps)`.
//!
//! The compiler groups an `Add` with the `RMSNorm` that reads its result at lowering time
//! (`nsl_codegen::wengert_lower`), and emits this entry point where it would have emitted
//! the two calls. Each `*_route` names the call the unfused lowering makes for that op:
//!
//! - `0`: the plain entry point (`nsl_tensor_add(x, o, 0)` / `nsl_tensor_rmsnorm`);
//! - `1` / `3`: the bf16 storage plan's dtype-explicit call with that output dtype
//!   (`nsl_tensor_binary_dt(ADD, ..)` / `nsl_tensor_rmsnorm_dt`, f32 = 1, bf16 = 3).
//!
//! On the GPU, when the two calls would run the KIR add kernel and the KIR RMSNorm kernel
//! (`x`, `o` contiguous, one shape, f32 or bf16, on one device; `gamma` a contiguous f32
//! vector of the row length there; the plain routes only for f32), one kernel
//! (`nsl_kir::kernels::norm::build_add_rmsnorm`) writes both outputs with those kernels'
//! arithmetic: bit-identical to the pair. Everywhere else it IS the pair of calls, so
//! the fallback is the unfused lowering itself.
//!
//! Returns an `NslList` `[h, n]`; the caller takes both and frees the list.

#[cfg(feature = "cuda")]
use crate::tensor::NslTensor;
use crate::tensor::{DTYPE_BF16, DTYPE_F32};

/// The fused launches taken and the calls that fell back to the pair (a parity gate
/// asserts the fused count, so a run that silently fell back cannot pass as fused).
static FUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PAIRS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(fused launches, fallbacks to the add + norm pair)` so far in this process.
pub fn counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (FUSED.load(Relaxed), PAIRS.load(Relaxed))
}

/// The output dtype a route stores, or `None` for the plain entry point.
fn route_dtype(name: &str, route: i64) -> Option<u16> {
    match route {
        0 => None,
        r if r == i64::from(DTYPE_F32) => Some(DTYPE_F32),
        r if r == i64::from(DTYPE_BF16) => Some(DTYPE_BF16),
        other => crate::fatal::die(
            crate::fatal::Fatal::UnsupportedDtype,
            &format!("{name}: route {other} is not 0 (plain), 1 (f32) or 3 (bf16)"),
        ),
    }
}

/// The storages of `(h, n)` when the fused kernel reproduces the two calls, else `None`.
#[cfg(feature = "cuda")]
fn fused_storages(
    x: &NslTensor,
    o: &NslTensor,
    g: &NslTensor,
    h_dt: Option<u16>,
    n_dt: Option<u16>,
) -> Option<(
    nsl_kir::kernels::storage::Storage,
    nsl_kir::kernels::storage::Storage,
)> {
    use nsl_kir::kernels::storage::Storage;
    let storage_dtype = |d: u16| d == DTYPE_F32 || d == DTYPE_BF16;
    let ok = !crate::autodiff::is_recording()
        && x.device > 0
        && x.ndim >= 1
        && x.len > 0
        && x.len < (1i64 << 31)
        && o.device == x.device
        && storage_dtype(x.dtype)
        && storage_dtype(o.dtype)
        && x.is_contiguous()
        && o.is_contiguous()
        && x.shape_eq(o);
    if !ok {
        return None;
    }
    let cols = unsafe { *x.shape.add(x.ndim as usize - 1) };
    if cols <= 0
        || g.dtype != DTYPE_F32
        || g.device != x.device
        || !g.is_contiguous()
        || g.len != cols
    {
        return None;
    }
    let st = |d: u16| {
        if d == DTYPE_BF16 {
            Storage::Bf16
        } else {
            Storage::F32
        }
    };
    // The plain add runs the f32 KIR kernel only on two f32 operands.
    let h = match h_dt {
        Some(d) => st(d),
        None if x.dtype == DTYPE_F32 && o.dtype == DTYPE_F32 => Storage::F32,
        None => return None,
    };
    // The plain RMSNorm reads an f32 input and writes f32.
    let n = match n_dt {
        Some(d) => st(d),
        None if h == Storage::F32 => Storage::F32,
        None => return None,
    };
    Some((h, n))
}

/// `[h, n]` (module docs). `eps` is passed as the unfused norm call receives it.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_add_rmsnorm(
    x: i64,
    o: i64,
    gamma: i64,
    eps: f64,
    h_route: i64,
    n_route: i64,
) -> i64 {
    const NAME: &str = "nsl_tensor_add_rmsnorm";
    let h_dt = route_dtype(NAME, h_route);
    let n_dt = route_dtype(NAME, n_route);
    let list = crate::list::nsl_list_new();
    #[cfg(feature = "cuda")]
    {
        let (tx, to, tg) = (
            NslTensor::from_ptr_ref(x),
            NslTensor::from_ptr_ref(o),
            NslTensor::from_ptr_ref(gamma),
        );
        if let Some((h_st, n_st)) = fused_storages(tx, to, tg, h_dt, n_dt) {
            // The two calls' own eps conversion (`eps as f32`).
            let (h, n) =
                crate::cuda::storage_kernels::add_rmsnorm(x, o, gamma, eps as f32, h_st, n_st);
            FUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::list::nsl_list_push(list, h);
            crate::list::nsl_list_push(list, n);
            return list;
        }
    }
    PAIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let h = match h_dt {
        None => crate::tensor::arithmetic::nsl_tensor_add(x, o, 0),
        Some(d) => {
            crate::storage_dt::nsl_tensor_binary_dt(nsl_abi::wire::dt_op::ADD, x, o, i64::from(d))
        }
    };
    let n = match n_dt {
        None => crate::tensor::nsl_tensor_rmsnorm(h, gamma, eps),
        Some(d) => crate::storage_dt::nsl_tensor_rmsnorm_dt(h, gamma, eps, i64::from(d)),
    };
    crate::list::nsl_list_push(list, h);
    crate::list::nsl_list_push(list, n);
    list
}
