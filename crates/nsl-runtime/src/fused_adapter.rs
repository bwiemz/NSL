//! WRGA B.3 Task 4: fused LoRA/IA³ adapter FFIs.
//!
//! These entry points are the *single-call* replacement for the unfused
//! three-FFI triple (`x @ W + ((x @ A) @ B) * scale`) emitted by B.2.1.
//! The AST rewrite in `wrga_adapter_rewrite.rs` emits a Call to one of
//! these FFIs when the site's `FusionTarget` is `EpilogueFusedLora` /
//! `ActivationFusedIa3` AND the compile target is sm_80+.
//!
//! # Status — Task 4 (CPU-fallback stubs)
//!
//! The real fused CUDA launcher (which invokes the PTX produced by
//! `wrga_fused_ptx::synthesize_fused_lora_ptx`) is the responsibility of
//! Task 5.  For Task 4 we only need the FFI contract to be reachable and
//! numerically correct so the AST rewrite can route sites through the
//! single-call path.  Each function delegates to the existing
//! `nsl_tensor_matmul` / `nsl_tensor_add` / `nsl_tensor_mul_scalar`
//! primitives — identical math to B.2.1's unfused triple, just reached
//! through a different FFI.  When Task 5 wires the real CUDA launch, the
//! body of these functions is replaced with a PTX dispatch keyed on
//! `kernel_handle`; the FFI signature does not change.
//!
//! # FFI contract
//!
//! All tensor pointers arrive as `i64` handles on the *NslTensor* heap,
//! following the rest of the runtime's calling convention.  Operand
//! ownership stays with the caller; the returned tensor is freshly
//! allocated and the caller must free it.

use crate::tensor::arithmetic::{
    nsl_tensor_add, nsl_tensor_matmul, nsl_tensor_mul, nsl_tensor_mul_scalar,
};
use crate::tensor::fbip_flags::{RELINQUISH_A, RELINQUISH_B};
use std::sync::atomic::{AtomicU64, Ordering};

/// B.3 Task 5: side-channel counter for fused-adapter kernel launches.
/// Increments once per call to a fused FFI (per call site, per invocation).
/// When `NSL_KERNEL_LAUNCH_COUNTER=1` is set in the environment, a value
/// is printed to stderr at process exit via the atexit hook registered in
/// `args.rs::nsl_args_init`.
pub(crate) static FUSED_ADAPTER_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Bump the fused-adapter launch counter.  Only effective (i.e. observed
/// by the atexit reporter) when `NSL_KERNEL_LAUNCH_COUNTER=1`; the
/// increment itself is unconditional and thread-safe.
#[inline]
fn record_fused_launch() {
    FUSED_ADAPTER_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Atexit hook: print `[nsl-kernel-count] <N>` to stderr.  Registered
/// from `args.rs::nsl_args_init` only when `NSL_KERNEL_LAUNCH_COUNTER=1`.
pub extern "C" fn nsl_fused_adapter_launch_count_atexit() {
    let n = FUSED_ADAPTER_LAUNCH_COUNT.load(Ordering::Relaxed);
    eprintln!("[nsl-kernel-count] {n}");
}

/// Fused LoRA matmul: `y = x @ W + ((x @ A) @ B) * scale` in a single
/// FFI call.
///
/// # Arguments
///
/// * `x`          — activation, shape `[..., k_in]`
/// * `w`          — base weight, shape `[k_in, d_out]`
/// * `lora_a`     — LoRA-A, shape `[k_in, rank]`
/// * `lora_b`     — LoRA-B, shape `[rank, d_out]`
/// * `scale`      — `alpha / rank`, passed as f32
/// * `kernel_handle` — opaque index into the compiler's fused-PTX registry;
///                     reserved for Task 5 (CUDA launch dispatch).  The
///                     CPU-fallback stub ignores it.
///
/// # Task 5 handoff
///
/// Replace the CPU body below with:
/// `crate::cuda::launch_fused_lora(kernel_handle, x, w, lora_a, lora_b, scale)`
/// once the launcher ships.  The FFI signature is frozen — do not change.
///
/// # Safety
/// All tensor pointers must be valid `*NslTensor` handles or `0`.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_lora_matmul(
    x: i64,
    w: i64,
    lora_a: i64,
    lora_b: i64,
    scale: f64,
    kernel_handle: i64,
) -> i64 {
    // kernel_handle is Task-5 territory; silence unused-var warning here.
    let _ = kernel_handle;

    if x == 0 || w == 0 || lora_a == 0 || lora_b == 0 {
        return 0;
    }
    record_fused_launch();

    // y_main = x @ W
    let y_main = nsl_tensor_matmul(x, w, 0);
    if y_main == 0 {
        return 0;
    }
    // x_a = x @ A
    let x_a = nsl_tensor_matmul(x, lora_a, 0);
    if x_a == 0 {
        return 0;
    }
    // x_ab = x_a @ B   (consume x_a)
    let x_ab = nsl_tensor_matmul(x_a, lora_b, RELINQUISH_A);
    if x_ab == 0 {
        return 0;
    }
    // scaled = x_ab * scale   (consume x_ab)
    let scaled = nsl_tensor_mul_scalar(x_ab, scale, RELINQUISH_A);
    if scaled == 0 {
        return 0;
    }
    // y = y_main + scaled   (consume both)
    nsl_tensor_add(y_main, scaled, RELINQUISH_A | RELINQUISH_B)
}

/// Fused IA³ matmul: `y = (x @ W) * gamma` in a single FFI call.
///
/// # Arguments
///
/// * `x`             — activation, shape `[..., k_in]`
/// * `w`             — base weight, shape `[k_in, d_out]`
/// * `ia3_scale`     — per-output-channel scale vector, shape `[d_out]`
/// * `kernel_handle` — reserved for Task 5; ignored by the CPU stub.
///
/// # Safety
/// All tensor pointers must be valid `*NslTensor` handles or `0`.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_ia3_matmul(
    x: i64,
    w: i64,
    ia3_scale: i64,
    kernel_handle: i64,
) -> i64 {
    let _ = kernel_handle;

    if x == 0 || w == 0 || ia3_scale == 0 {
        return 0;
    }
    record_fused_launch();

    // y_main = x @ W
    let y_main = nsl_tensor_matmul(x, w, 0);
    if y_main == 0 {
        return 0;
    }
    // y = y_main * ia3_scale (broadcast over last dim)  (consume y_main)
    nsl_tensor_mul(y_main, ia3_scale, RELINQUISH_A)
}
