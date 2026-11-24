//! Launchers for the storage-generic elementwise kernels (bf16 activation
//! storage): each operand f32 or bf16 as its tensor's dtype says, the result
//! in the dtype the caller names, f32 arithmetic and one round-to-nearest-even
//! on a bf16 store. The kernels are `nsl_kir::kernels::elementwise`'s `_st`
//! builders; this module lowers each (op, storages) instantiation once and
//! launches it the way `gpu_elementwise_binary` launches the f32 kernels (one
//! thread per element, 256-thread blocks).
//!
//! Callers (`crate::storage_dt`) have already checked the fast-path
//! preconditions: every operand on the same device, contiguous, one length
//! (and, for two or more operands, one shape), and a dtype that is f32 or
//! bf16. The launchers re-assert the dtype/storage agreement, because a
//! mismatch would read a buffer at the wrong element width.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use nsl_kir::kernels::elementwise::{BackwardOp, BinaryOp, RotateHalfOp, UnaryOp};
use nsl_kir::kernels::norm::NormOp;
use nsl_kir::kernels::storage::Storage;

use crate::tensor::{NslTensor, DTYPE_BF16, DTYPE_F32};

/// One instantiation: the family and op, the operand storages, the output's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kernel {
    Binary(BinaryOp, [Storage; 2], Storage),
    Unary(UnaryOp, Storage, Storage),
    SwigluGate([Storage; 3], Storage),
    RotateHalf(RotateHalfOp, Storage, Storage),
    Norm(NormOp, Storage, Storage),
    Rope(nsl_kir::kernels::rope::RopeDir, Storage, Storage),
    RopeRows(nsl_kir::kernels::rope::RopeDir, Storage, Storage),
    Swiglu(nsl_kir::kernels::swiglu::SwigluKernel),
    /// The residual add + RMSNorm: storages of `x`, `o`, the sum `h`, the norm output.
    AddRmsNorm([Storage; 4]),
}

/// The NUL-terminated PTX module and entry name of `k`, lowered once and kept
/// for the life of the process (`kernel_launch` keys its module cache on the
/// PTX buffer's address, so the buffer must not move or be freed).
fn module(k: Kernel) -> (&'static str, &'static str) {
    static MODULES: Mutex<Option<HashMap<Kernel, (&'static str, &'static str)>>> = Mutex::new(None);
    let mut guard = MODULES.lock().unwrap_or_else(|p| p.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    *map.entry(k).or_insert_with(|| {
        use nsl_kir::kernels::elementwise as ew;
        let ir = match k {
            Kernel::Binary(op, ins, out) => ew::build_binary_st(op, ins, out),
            Kernel::Unary(op, i, out) => ew::build_unary_st(op, i, out),
            Kernel::SwigluGate(ins, out) => ew::build_backward_st(BackwardOp::SwigluGate, &ins, out),
            Kernel::RotateHalf(op, i, out) => ew::build_rotate_half_st(op, i, out),
            Kernel::Norm(op, i, out) => nsl_kir::kernels::norm::build_st(op, i, out),
            Kernel::Rope(dir, i, out) => nsl_kir::kernels::rope::build(dir, i, out),
            Kernel::RopeRows(dir, i, out) => nsl_kir::kernels::rope::build_rows(dir, i, out),
            Kernel::Swiglu(k) => k.build(),
            Kernel::AddRmsNorm([x, o, h, n]) => {
                nsl_kir::kernels::norm::build_add_rmsnorm(x, o, h, n)
            }
        };
        let name: &'static str = Box::leak(format!("{}\0", ir.name).into_boxed_str());
        if let Err(errors) = nsl_kir::kir_verify::verify(&ir) {
            panic!("storage kernel `{}` failed KIR verification: {errors:?}", ir.name);
        }
        let ptx = String::from_utf8(nsl_kir::backend_ptx::lower_kir_to_ptx(&ir)).expect("PTX must be ASCII");
        let ptx: &'static str = Box::leak(ptx.into_boxed_str());
        (ptx, name)
    })
}

/// The storage a tensor's dtype tag denotes. Only f32 and bf16 reach here.
pub(crate) fn storage_of(t: &NslTensor) -> Storage {
    match t.dtype {
        DTYPE_F32 => Storage::F32,
        DTYPE_BF16 => Storage::Bf16,
        other => panic!("storage kernel operand has dtype {other}; only f32 (1) and bf16 (3) are storages"),
    }
}

fn dtype_of(s: Storage) -> u16 {
    match s {
        Storage::F32 => DTYPE_F32,
        Storage::Bf16 => DTYPE_BF16,
    }
}

/// A fresh contiguous device tensor shaped like `like`, `out` storage.
fn alloc_like(like: &NslTensor, out: Storage) -> i64 {
    let n = like.len as usize;
    let data = super::inner::alloc_managed(n * out.bytes());
    let shape = NslTensor::copy_shape(like.shape, like.ndim);
    let strides = NslTensor::compute_strides(shape, like.ndim);
    let t = Box::new(NslTensor::new(data, shape, strides, like.ndim, like.len, like.device, dtype_of(out), 1, 0));
    Box::into_raw(t) as i64
}

fn data(ptr: i64) -> u64 {
    NslTensor::from_ptr_ref(ptr).data as u64
}

/// Launch `k` one thread per element over `n` elements with `args` (each a
/// `u64` or `f32` stored in a `u64` slot as the kernel's parameter list
/// says). Fatal on a launch error: the output would be garbage.
fn launch(k: Kernel, n: usize, args: &mut [u64]) {
    let block = 256i64;
    launch_grid(k, (n as i64 + block - 1) / block, 0, args);
}

/// `launch` with an explicit grid and shared-memory size (256-thread blocks).
fn launch_grid(k: Kernel, grid: i64, shared: u32, args: &mut [u64]) {
    let (ptx, name) = module(k);
    let ptrs: Vec<*mut c_void> = args.iter_mut().map(|a| a as *mut u64 as *mut c_void).collect();
    let block = 256i64;
    let rc = super::inner::kernel_launch(ptx.as_ptr(), name.as_ptr(), [grid, 1, 1], [block, 1, 1], &ptrs, shared);
    if rc as u32 != 0 {
        crate::fatal::die(
            crate::fatal::Fatal::Unsupported,
            &format!("storage kernel `{}` failed to launch: CUresult {}", name.trim_end_matches('\0'), rc as u32),
        );
    }
    super::inner::sync_after_kernel();
}

/// `c = a op b`, `a` and `b` one shape, contiguous, on one device.
pub(crate) fn binary(op: BinaryOp, a: i64, b: i64, out: Storage) -> i64 {
    super::inner::set_oom_context("binary_dt");
    let (ta, tb) = (NslTensor::from_ptr_ref(a), NslTensor::from_ptr_ref(b));
    let k = Kernel::Binary(op, [storage_of(ta), storage_of(tb)], out);
    let c = alloc_like(ta, out);
    let n = ta.len as usize;
    launch(k, n, &mut [data(a), data(b), data(c), n as u64]);
    c
}

/// `c = f(a)`, `a` contiguous.
pub(crate) fn unary(op: UnaryOp, a: i64, out: Storage) -> i64 {
    super::inner::set_oom_context("unary_dt");
    let ta = NslTensor::from_ptr_ref(a);
    let k = Kernel::Unary(op, storage_of(ta), out);
    let c = alloc_like(ta, out);
    let n = ta.len as usize;
    launch(k, n, &mut [data(a), data(c), n as u64]);
    c
}

/// The SwiGLU gate adjoint `grad * up * silu'(input)`, every operand one
/// shape and contiguous.
pub(crate) fn swiglu_gate(grad: i64, up: i64, input: i64, out: Storage) -> i64 {
    super::inner::set_oom_context("swiglu_gate_backward_dt");
    let tg = NslTensor::from_ptr_ref(grad);
    let ins = [storage_of(tg), storage_of(NslTensor::from_ptr_ref(up)), storage_of(NslTensor::from_ptr_ref(input))];
    let k = Kernel::SwigluGate(ins, out);
    let c = alloc_like(tg, out);
    let n = tg.len as usize;
    launch(k, n, &mut [data(grad), data(up), data(input), data(c), n as u64]);
    c
}

/// `rotate_half` (or its negation) over the last dimension, `a` contiguous
/// with an even last dimension.
pub(crate) fn rotate_half(op: RotateHalfOp, a: i64, out: Storage) -> i64 {
    super::inner::set_oom_context("rotate_half_dt");
    let ta = NslTensor::from_ptr_ref(a);
    let last = if ta.ndim == 0 { 1 } else { unsafe { *ta.shape.add(ta.ndim as usize - 1) } };
    let k = Kernel::RotateHalf(op, storage_of(ta), out);
    let c = alloc_like(ta, out);
    let n = ta.len as usize;
    launch(k, n, &mut [data(a), data(c), n as u64, last as u64, (last / 2) as u64]);
    c
}

/// RMSNorm over the last dimension, one 256-thread block per row, as
/// `gpu_rmsnorm_f32` launches the f32 kernel. `x` contiguous (f32 or bf16),
/// `gamma` a contiguous f32 vector of the last dimension's length.
pub(crate) fn rmsnorm(x: i64, gamma: i64, eps: f32, out: Storage) -> i64 {
    super::inner::set_oom_context("rmsnorm_dt");
    let tx = NslTensor::from_ptr_ref(x);
    let cols = unsafe { *tx.shape.add(tx.ndim as usize - 1) } as u64;
    let rows = tx.len as u64 / cols;
    let k = Kernel::Norm(NormOp::RmsNorm, storage_of(tx), out);
    let c = alloc_like(tx, out);
    // The kernel's eps is an `.f32` parameter: its bits in the slot's low half.
    launch_grid(k, rows as i64, 256 * 4, &mut [data(x), data(c), data(gamma), rows, cols, u64::from(eps.to_bits())]);
    c
}

/// The residual add and the RMSNorm of its sum in one launch
/// (`nsl_kir::kernels::norm::build_add_rmsnorm`): `h = x + o` stored as `h_st`, `n =
/// rmsnorm(h, gamma, eps)` stored as `n_st`, one 256-thread block per row as [`rmsnorm`].
/// `x` and `o` contiguous, one shape, on one device; `gamma` a contiguous f32 vector of
/// the last dimension's length. Returns `(h, n)`.
pub(crate) fn add_rmsnorm(
    x: i64,
    o: i64,
    gamma: i64,
    eps: f32,
    h_st: Storage,
    n_st: Storage,
) -> (i64, i64) {
    super::inner::set_oom_context("add_rmsnorm");
    let (tx, to) = (NslTensor::from_ptr_ref(x), NslTensor::from_ptr_ref(o));
    let cols = unsafe { *tx.shape.add(tx.ndim as usize - 1) } as u64;
    let rows = tx.len as u64 / cols;
    let k = Kernel::AddRmsNorm([storage_of(tx), storage_of(to), h_st, n_st]);
    let h = alloc_like(tx, h_st);
    let n = alloc_like(tx, n_st);
    launch_grid(
        k,
        rows as i64,
        256 * 4,
        &mut [
            data(x),
            data(o),
            data(h),
            data(n),
            data(gamma),
            rows,
            cols,
            u64::from(eps.to_bits()),
        ],
    );
    (h, n)
}

/// The fused RoPE (`nsl_kir::kernels::rope`): `x` `[b, h, s, d]` f32 or bf16
/// read through its own strides, `cos`/`sin` contiguous f32 `[s, d]`, the
/// result contiguous `[b, h, s, d]` stored as `out`.
pub(crate) fn rope(dir: nsl_kir::kernels::rope::RopeDir, x: i64, cos: i64, sin: i64, out: Storage) -> i64 {
    super::inner::set_oom_context("rope");
    let tx = NslTensor::from_ptr_ref(x);
    let dim = |i: usize| unsafe { *tx.shape.add(i) } as u64;
    let st = |i: usize| unsafe { *tx.strides.add(i) } as u64;
    let k = Kernel::Rope(dir, storage_of(tx), out);
    let c = alloc_like(tx, out);
    let n = tx.len as usize;
    launch(
        k,
        n,
        &mut [data(x), data(cos), data(sin), data(c), n as u64, dim(3), dim(2), dim(1), st(0), st(1), st(2), st(3)],
    );
    c
}

/// The dense-rows RoPE (`nsl_kir::kernels::rope::build_rows`): `x` `[b, h,
/// s, d]` f32 or bf16 densely stored in `layout` (`crate::attn_layout`),
/// `cos`/`sin` contiguous f32 `[s, d]`, the result `[b, h, s, d]` in the same
/// layout, stored as `out`. The caller has checked `d % 16 == 0`, fewer than
/// 2^31 elements and 16-byte aligned operands ([`rope_rows_fits`]).
pub(crate) fn rope_rows(
    dir: nsl_kir::kernels::rope::RopeDir,
    x: i64,
    cos: i64,
    sin: i64,
    out: Storage,
    layout: crate::attn_layout::HeadLayout,
) -> i64 {
    super::inner::set_oom_context("rope");
    let tx = NslTensor::from_ptr_ref(x);
    let dims: [i64; 4] = std::array::from_fn(|i| unsafe { *tx.shape.add(i) });
    let n = tx.len as usize;
    let data_out = super::inner::alloc_managed(n * out.bytes());
    let c = crate::attn_layout::wrap_head_tensor(data_out, dims, layout, dtype_of(out), tx.device);
    let pos_div = match layout {
        crate::attn_layout::HeadLayout::Bhsd => 1,
        crate::attn_layout::HeadLayout::Bshd => dims[1] as u64,
    };
    let pairs = u64::from(nsl_kir::kernels::rope::ROWS_PAIRS);
    let chunks = n as u64 / (2 * pairs);
    let k = Kernel::RopeRows(dir, storage_of(tx), out);
    launch(k, chunks as usize, &mut [data(x), data(cos), data(sin), data(c), chunks, dims[3] as u64, dims[2] as u64, pos_div]);
    c
}

/// Whether [`rope_rows`] serves `x` (dense in `layout`, already checked by
/// the caller) with these tables: each half of a row splits into runs of
/// `ROWS_PAIRS` elements (`d` a multiple of 16) and every operand is 16-byte
/// aligned (the kernel's vector accesses; a run starts a multiple of 16
/// bytes into its row for either storage).
pub(crate) fn rope_rows_fits(x: &NslTensor, cos: &NslTensor, sin: &NslTensor) -> bool {
    let d = unsafe { *x.shape.add(3) } as u64;
    let pairs = 2 * u64::from(nsl_kir::kernels::rope::ROWS_PAIRS);
    d.is_multiple_of(pairs) && [x, cos, sin].iter().all(|t| (t.data as u64).is_multiple_of(16))
}

/// The fused SwiGLU kernels' elements per thread (`nsl_kir::kernels::swiglu`):
/// the widest of the family's widths, up to `SWIGLU_MAX_WIDTH`, that divides
/// `n` and to whose four-element vector access every operand is aligned;
/// one element per thread otherwise.
const SWIGLU_MAX_WIDTH: u32 = 8;

fn swiglu_width(n: usize, operands: &[(i64, Storage)]) -> u32 {
    let aligned = operands.iter().all(|&(t, s)| data(t).is_multiple_of(4 * s.bytes() as u64));
    nsl_kir::kernels::swiglu::WIDTHS
        .iter()
        .rev()
        .copied()
        .find(|&w| w <= SWIGLU_MAX_WIDTH && (w == 1 || (aligned && n.is_multiple_of(w as usize))))
        .unwrap_or(1)
}

/// Launch a fused SwiGLU kernel over `n` elements: `tensors` are its tensor
/// parameters in order (inputs, then outputs), all `n` long.
fn launch_swiglu(op: nsl_kir::kernels::swiglu::SwigluOp, ins: &[i64], outs: &[i64], out: Storage, round_silu: bool) {
    launch_swiglu_width(op, ins, outs, out, round_silu, None);
}

/// [`launch_swiglu`] with the width forced (`Some`, which the caller must
/// have checked is legal for these operands) or chosen (`None`).
fn launch_swiglu_width(
    op: nsl_kir::kernels::swiglu::SwigluOp,
    ins: &[i64],
    outs: &[i64],
    out: Storage,
    round_silu: bool,
    width: Option<u32>,
) {
    let n = NslTensor::from_ptr_ref(ins[0]).len as usize;
    let in_st: Vec<Storage> = ins.iter().map(|&t| storage_of(NslTensor::from_ptr_ref(t))).collect();
    let operands: Vec<(i64, Storage)> =
        ins.iter().copied().zip(in_st.iter().copied()).chain(outs.iter().map(|&t| (t, out))).collect();
    let width = width.unwrap_or_else(|| swiglu_width(n, &operands));
    let k = Kernel::Swiglu(nsl_kir::kernels::swiglu::SwigluKernel::new(op, &in_st, out, width, round_silu));
    let mut args: Vec<u64> = ins.iter().chain(outs).map(|&t| data(t)).collect();
    args.push(n as u64);
    let threads = (n / width as usize) as i64;
    launch_grid(k, (threads + 255) / 256, 0, &mut args);
}

/// `silu(gate) * up`, both one shape and contiguous.
pub(crate) fn swiglu_forward(gate: i64, up: i64, out: Storage, round_silu: bool) -> i64 {
    super::inner::set_oom_context("swiglu");
    let c = alloc_like(NslTensor::from_ptr_ref(gate), out);
    launch_swiglu(nsl_kir::kernels::swiglu::SwigluOp::Forward, &[gate, up], &[c], out, round_silu);
    c
}

/// `grad * silu(gate)`, both one shape and contiguous.
pub(crate) fn swiglu_up_backward(grad: i64, gate: i64, out: Storage, round_silu: bool) -> i64 {
    super::inner::set_oom_context("swiglu_up_backward");
    let c = alloc_like(NslTensor::from_ptr_ref(grad), out);
    launch_swiglu(nsl_kir::kernels::swiglu::SwigluOp::UpBackward, &[grad, gate], &[c], out, round_silu);
    c
}

/// `(dgate, dup, p)` from one launch (`p = silu(gate) * up`, the forward
/// value), every operand one shape and contiguous.
pub(crate) fn swiglu_backward_recompute(grad: i64, up: i64, gate: i64, out: Storage, round_silu: bool) -> (i64, i64, i64) {
    super::inner::set_oom_context("swiglu_backward_recompute");
    let like = NslTensor::from_ptr_ref(grad);
    let dg = alloc_like(like, out);
    let du = alloc_like(like, out);
    let p = alloc_like(like, out);
    launch_swiglu(
        nsl_kir::kernels::swiglu::SwigluOp::BackwardRecompute,
        &[grad, up, gate],
        &[dg, du, p],
        out,
        round_silu,
    );
    (dg, du, p)
}

/// `(dgate, dup)` from one launch, every operand one shape and contiguous.
pub(crate) fn swiglu_backward(grad: i64, up: i64, gate: i64, out: Storage, round_silu: bool) -> (i64, i64) {
    super::inner::set_oom_context("swiglu_backward");
    let like = NslTensor::from_ptr_ref(grad);
    let dg = alloc_like(like, out);
    let du = alloc_like(like, out);
    launch_swiglu(nsl_kir::kernels::swiglu::SwigluOp::Backward, &[grad, up, gate], &[dg, du], out, round_silu);
    (dg, du)
}

#[cfg(test)]
mod swiglu_timing {
    //! Diagnostic, not a gate: GPU time of the fused SwiGLU kernels at each
    //! width against the unfused launches they replace, on the Coder-1B MLP
    //! activation (`[4096, 8192]` bf16). Run with
    //! `cargo test -p nsl-runtime --lib --features cuda swiglu_timing -- --ignored --nocapture`.
    //! The printed times are host wall time over 20 calls (a loaded CPU inflates them);
    //! the kernel profiler is on, so `kernel_profile.json` in the working directory has
    //! each launch's GPU time from CUDA events.
    use super::*;
    use nsl_kir::kernels::elementwise::{BinaryOp, UnaryOp};
    use nsl_kir::kernels::swiglu::SwigluOp;

    fn randn_bf16(rows: i64, cols: i64) -> i64 {
        let l = crate::list::nsl_list_new();
        crate::list::nsl_list_push(l, rows);
        crate::list::nsl_list_push(l, cols);
        let host = crate::tensor::creation::nsl_tensor_randn(l);
        crate::list::nsl_list_free(l);
        let dev = crate::tensor::nsl_tensor_to_device(host, 1);
        crate::tensor::nsl_tensor_free(host);
        let b = crate::tensor::precision_cast::convert_untaped(dev, DTYPE_BF16);
        crate::tensor::nsl_tensor_free(dev);
        b
    }

    /// Mean microseconds per call of `f` over `reps` calls (after 3 warm-up
    /// calls), each call's outputs freed at once (the caching allocator hands
    /// the same blocks back, so no call pays for a driver allocation).
    fn time(reps: usize, mut f: impl FnMut() -> Vec<i64>) -> f64 {
        let mut call = || {
            for t in f() {
                crate::tensor::nsl_tensor_free(t);
            }
        };
        for _ in 0..3 {
            call();
        }
        crate::math::nsl_cuda_device_synchronize();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            call();
        }
        crate::math::nsl_cuda_device_synchronize();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    }

    #[test]
    #[ignore = "diagnostic: times the fused SwiGLU kernels on the GPU"]
    fn swiglu_kernel_timing() {
        assert_eq!(crate::cuda::nsl_cuda_init(), 0);
        crate::kernel_profiler::nsl_kernel_profiler_start();
        let (rows, cols) = (4096i64, 8192i64);
        let (g, u, dy) = (randn_bf16(rows, cols), randn_bf16(rows, cols), randn_bf16(rows, cols));
        let pass_mib = (rows * cols * 2) as f64 / (1024.0 * 1024.0);
        let report = |name: &str, passes: f64, us: f64| {
            let gbs = passes * pass_mib * 1024.0 * 1024.0 / (us * 1e-6) / 1e9;
            eprintln!("{name:<44} {us:>8.1} us  {passes:>2} passes  {gbs:>6.0} GB/s");
        };
        let b = Storage::Bf16;
        let reps = 20;
        // Twice each, in turn, so a first-use cost or a clock change shows up
        // as a disagreement between the two passes.
        for pass in 0..2 {
            eprintln!("-- pass {pass}");
            report("unfused silu (nsl_silu_b_b)", 2.0, time(reps, || vec![unary(UnaryOp::Silu, g, b)]));
            report("unfused mul (nsl_mul_bb_b)", 3.0, time(reps, || vec![binary(BinaryOp::Mul, g, u, b)]));
            report("unfused gate backward (swiglu_gate_bbb_b)", 4.0, time(reps, || vec![swiglu_gate(dy, u, g, b)]));
            report("copy-like: unary neg (nsl_neg_b_b)", 2.0, time(reps, || vec![unary(UnaryOp::Neg, g, b)]));
        for w in nsl_kir::kernels::swiglu::WIDTHS {
            let fwd = time(reps, || {
                let c = alloc_like(NslTensor::from_ptr_ref(g), b);
                launch_swiglu_width(SwigluOp::Forward, &[g, u], &[c], b, false, Some(w));
                vec![c]
            });
            report(&format!("fused forward width {w}"), 3.0, fwd);
            let up = time(reps, || {
                let c = alloc_like(NslTensor::from_ptr_ref(g), b);
                launch_swiglu_width(SwigluOp::UpBackward, &[dy, g], &[c], b, false, Some(w));
                vec![c]
            });
            report(&format!("fused up backward width {w}"), 3.0, up);
            let bwd = time(reps, || {
                let dg = alloc_like(NslTensor::from_ptr_ref(g), b);
                let du = alloc_like(NslTensor::from_ptr_ref(g), b);
                launch_swiglu_width(SwigluOp::Backward, &[dy, u, g], &[dg, du], b, false, Some(w));
                vec![dg, du]
            });
            report(&format!("fused backward (dg, du) width {w}"), 5.0, bwd);
            let bwdh = time(reps, || {
                let dg = alloc_like(NslTensor::from_ptr_ref(g), b);
                let du = alloc_like(NslTensor::from_ptr_ref(g), b);
                let p = alloc_like(NslTensor::from_ptr_ref(g), b);
                launch_swiglu_width(SwigluOp::BackwardRecompute, &[dy, u, g], &[dg, du, p], b, false, Some(w));
                vec![dg, du, p]
            });
            report(&format!("fused backward (dg, du, h) width {w}"), 6.0, bwdh);
        }
        }
        for t in [g, u, dy] {
            crate::tensor::nsl_tensor_free(t);
        }
    }
}
