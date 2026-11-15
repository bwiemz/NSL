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
use nsl_kir::kernels::storage::Storage;

use crate::tensor::{NslTensor, DTYPE_BF16, DTYPE_F32};

/// One instantiation: the family and op, the operand storages, the output's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kernel {
    Binary(BinaryOp, [Storage; 2], Storage),
    Unary(UnaryOp, Storage, Storage),
    SwigluGate([Storage; 3], Storage),
    RotateHalf(RotateHalfOp, Storage, Storage),
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
    let (ptx, name) = module(k);
    let ptrs: Vec<*mut c_void> = args.iter_mut().map(|a| a as *mut u64 as *mut c_void).collect();
    let block = 256i64;
    let grid = (n as i64 + block - 1) / block;
    let rc = super::inner::kernel_launch(ptx.as_ptr(), name.as_ptr(), [grid, 1, 1], [block, 1, 1], &ptrs, 0);
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
