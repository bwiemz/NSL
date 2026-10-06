//! C5 step 4: a tensor op whose operands have different dtypes is refused
//! (`docs/superpowers/specs/2026-09-26-dtype-semantics-design.md`).
//!
//! Before step 4 the CPU converted silently: either operand f32 gave an f32
//! result with the other side narrowed per element ("f32 wins"), in the
//! elementwise ops, matmul, conv2d and bias_add; compare and where read each
//! operand with its own accessor; cat converted later inputs into the first
//! one's dtype; and add_inplace cast its source into the destination's dtype.
//! Now every one of them dies with `Fatal::UnsupportedDtype`, naming the op,
//! both dtypes and the explicit fix (`.to(dtype)`).
//!
//! # Why the refusals run in a subprocess
//!
//! `fatal::die` ends the process (`std::process::exit`), so libtest cannot
//! observe it in-process. Each refusal gate re-execs THIS test binary with
//! `NSL_MIXED_DTYPE_SCENARIO` set -- the `zz_*_child` pattern
//! `gpu_dtype_refusal.rs` uses -- and asserts on the child's exit code and
//! stderr: the op name and both dtype names together, so a gate cannot pass
//! because some other check fired first. That is also what makes the gate
//! catch a re-added promotion branch: the elementwise dispatcher refuses a
//! mismatch too (for direct callers), but under the name
//! `tensor_elementwise_op`, so deleting an op's entry check fails its gate.
//!
//! The same-dtype tests below run in-process: they pin that refusing a
//! MISMATCH did not take a homogeneous f64 path with it.

use std::ffi::c_void;

use nsl_runtime::fatal::NSL_EXIT_UNSUPPORTED_DTYPE;

unsafe extern "C" {
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_tensor_full_dtype(shape_list: i64, value: f64, dtype: i64) -> i64;
    fn nsl_tensor_scalar(val: f64, dtype: i64) -> i64;
    fn nsl_tensor_add(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_sub(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_mul(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_div(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_matmul(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_conv2d(
        input: i64,
        weight: i64,
        bias: i64,
        stride_h: i64,
        stride_w: i64,
        pad_h: i64,
        pad_w: i64,
    ) -> i64;
    fn nsl_tensor_bias_add(tensor: i64, bias: i64) -> i64;
    fn nsl_tensor_compare(a: i64, b: i64, cmp_kind: i64) -> i64;
    fn nsl_tensor_where(cond: i64, true_val: i64, false_val: i64) -> i64;
    fn nsl_tensor_cat(tensor_list: i64, dim: i64) -> i64;
    fn nsl_fused_elementwise_2(a: i64, b: i64, ops_ptr: i64, num_ops: i64) -> i64;
    fn nsl_tensor_add_inplace(dst: i64, src: i64);
    fn nsl_tensor_scalar_mul_add_inplace(m: i64, g: i64, s: f64);
    fn nsl_tensor_layernorm(input: i64, weight: i64, bias: i64, eps: f64) -> i64;
    fn nsl_tensor_rmsnorm(input: i64, weight: i64, eps: f64) -> i64;
    fn nsl_tensor_scalar_rhs(x: i64, s: f64, opcode: i64) -> i64;
    fn nsl_sparse_from_dense(dense: i64, format: i64, threshold_bits: i64) -> i64;
    fn nsl_sparse_spmm(sparse: i64, dense: i64) -> i64;
    fn nsl_sparse_spmv(sparse: i64, vec: i64) -> i64;
    fn nsl_qtensor_quantize(t: i64, dtype: i64, granularity: i64, axis: i64, group: i64) -> i64;
    fn nsl_qtensor_dequantize(qt: i64) -> i64;
    fn nsl_qtensor_matmul_mixed(x: i64, qw: i64) -> i64;
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
}

const F64: i64 = 0;
const F32: i64 = 1;

/// Mirror of the runtime's `#[repr(C)]` `NslTensor` header, for reading a
/// result's dtype and elements (the same mirror `gpu_dtype_refusal.rs` keeps).
#[repr(C)]
struct TensorView {
    _magic: u32,
    data: *mut c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    _ndim: i64,
    len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    _device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

fn view(ptr: i64) -> &'static TensorView {
    assert_ne!(ptr, 0, "null tensor");
    unsafe { &*(ptr as *const TensorView) }
}

/// A tensor's elements, read in its own dtype and widened for comparison.
fn values(ptr: i64) -> Vec<f64> {
    let v = view(ptr);
    (0..v.len as usize)
        .map(|i| unsafe {
            match v.dtype {
                0 => *(v.data as *const f64).add(i),
                1 => f64::from(*(v.data as *const f32).add(i)),
                d => panic!("unexpected dtype {d}"),
            }
        })
        .collect()
}

fn list(items: &[i64]) -> i64 {
    let l = unsafe { nsl_list_new() };
    for &x in items {
        unsafe { nsl_list_push(l, x) };
    }
    l
}

fn full(dims: &[i64], value: f64, dtype: i64) -> i64 {
    let t = unsafe { nsl_tensor_full_dtype(list(dims), value, dtype) };
    assert_eq!(view(t).dtype as i64, dtype, "the producer must make the dtype asked for");
    t
}

// --- child ---------------------------------------------------------------

const SCENARIO: &str = "NSL_MIXED_DTYPE_SCENARIO";

/// One mixed-dtype call per scenario. Returns immediately unless the parent
/// set the scenario; is expected to EXIT inside the call.
#[test]
fn zz_mixed_dtype_child() {
    let Ok(scenario) = std::env::var(SCENARIO) else {
        return;
    };
    let a32 = || full(&[2, 3], 1.5, F32);
    let a64 = || full(&[2, 3], 1.5, F64);
    let _ = match scenario.as_str() {
        "add" => unsafe { nsl_tensor_add(a32(), a64(), 0) },
        "sub" => unsafe { nsl_tensor_sub(a64(), a32(), 0) },
        "mul" => unsafe { nsl_tensor_mul(a32(), a64(), 0) },
        "div" => unsafe { nsl_tensor_div(a64(), a32(), 0) },
        // The Wengert constant's shape: an f32 rank-0 tensor beside f64.
        "add_scalar_tensor" => unsafe { nsl_tensor_add(a64(), nsl_tensor_scalar(2.0, F32), 0) },
        "matmul" => unsafe { nsl_tensor_matmul(a32(), full(&[3, 2], 0.5, F64), 0) },
        "conv2d" => unsafe {
            nsl_tensor_conv2d(full(&[1, 1, 3, 3], 1.0, F32), full(&[1, 1, 2, 2], 1.0, F64), 0, 1, 1, 0, 0)
        },
        "conv2d_bias" => unsafe {
            let (x, w) = (full(&[1, 1, 3, 3], 1.0, F64), full(&[1, 1, 2, 2], 1.0, F64));
            nsl_tensor_conv2d(x, w, full(&[1], 0.5, F32), 1, 1, 0, 0)
        },
        "bias_add" => unsafe { nsl_tensor_bias_add(a32(), full(&[3], 0.5, F64)) },
        "compare" => unsafe { nsl_tensor_compare(a64(), nsl_tensor_scalar(0.0, F32), 0) },
        "where" => unsafe { nsl_tensor_where(a32(), a64(), nsl_tensor_scalar(0.0, F32)) },
        "cat" => unsafe { nsl_tensor_cat(list(&[a32(), a64()]), 0) },
        "fused_elementwise_2" => unsafe { nsl_fused_elementwise_2(a32(), a64(), list(&[0]), 1) },
        "add_inplace" => {
            unsafe { nsl_tensor_add_inplace(a32(), a64()) };
            0
        }
        "scalar_mul_add_inplace" => {
            unsafe { nsl_tensor_scalar_mul_add_inplace(a32(), a64(), 0.5) };
            0
        }
        "layernorm" => unsafe {
            nsl_tensor_layernorm(a32(), full(&[3], 1.0, F64), full(&[3], 0.0, F32), 1e-5)
        },
        "layernorm_bias" => unsafe {
            nsl_tensor_layernorm(a32(), full(&[3], 1.0, F32), full(&[3], 0.0, F64), 1e-5)
        },
        "rmsnorm" => unsafe { nsl_tensor_rmsnorm(a64(), full(&[3], 1.0, F32), 1e-5) },
        // Replays the baseline: x OP an f32 scalar tensor, refused by the op.
        "scalar_rhs" => unsafe { nsl_tensor_scalar_rhs(a64(), 2.0, 0) },
        "spmm" => unsafe {
            let sp = nsl_sparse_from_dense(full(&[2, 2], 1.0, F64), 1, 0f64.to_bits() as i64);
            nsl_sparse_spmm(sp, full(&[2, 3], 1.0, F32))
        },
        "spmv" => unsafe {
            let sp = nsl_sparse_from_dense(full(&[2, 2], 1.0, F32), 1, 0f64.to_bits() as i64);
            nsl_sparse_spmv(sp, full(&[2], 1.0, F64))
        },
        other => panic!("unknown {SCENARIO} '{other}'"),
    };
    // Only reached if nothing refused.
    println!("MIXED-DTYPES-NOT-REFUSED scenario={scenario}");
}

// --- parent --------------------------------------------------------------

/// Run `scenario` in a fresh process; require exit code 17
/// (`Fatal::UnsupportedDtype`) and a message naming `op` and the dtype pair.
fn assert_refused(scenario: &str, op: &str, dtypes: (&str, &str)) {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["zz_mixed_dtype_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(SCENARIO, scenario)
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("re-exec the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let ctx = format!("--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}");
    assert!(
        !stdout.contains("MIXED-DTYPES-NOT-REFUSED"),
        "scenario '{scenario}': mixed dtypes went through -- a promotion rule is back.\n{ctx}"
    );
    assert_eq!(
        out.status.code(),
        Some(NSL_EXIT_UNSUPPORTED_DTYPE),
        "scenario '{scenario}': expected the unsupported-dtype exit code.\n{ctx}"
    );
    let (a, b) = dtypes;
    let want = format!("{op}: operands have different dtypes, {a} and {b}");
    assert!(stderr.contains(&want), "scenario '{scenario}': expected `{want}`.\n{ctx}");
    assert!(
        stderr.contains(&format!("`.to({a})`")) && stderr.contains(&format!("`.to({b})`")),
        "scenario '{scenario}': the message must name the explicit conversion.\n{ctx}"
    );
}

#[test]
fn elementwise_ops_refuse_mixed_dtypes() {
    assert_refused("add", "nsl_tensor_add", ("f32", "f64"));
    assert_refused("sub", "nsl_tensor_sub", ("f64", "f32"));
    assert_refused("mul", "nsl_tensor_mul", ("f32", "f64"));
    assert_refused("div", "nsl_tensor_div", ("f64", "f32"));
    assert_refused("add_scalar_tensor", "nsl_tensor_add", ("f64", "f32"));
    assert_refused("fused_elementwise_2", "nsl_fused_elementwise_2", ("f32", "f64"));
}

#[test]
fn contractions_refuse_mixed_dtypes() {
    assert_refused("matmul", "nsl_tensor_matmul", ("f32", "f64"));
    assert_refused("conv2d", "nsl_tensor_conv2d", ("f32", "f64"));
    assert_refused("conv2d_bias", "nsl_tensor_conv2d", ("f64", "f32"));
    assert_refused("bias_add", "nsl_tensor_bias_add", ("f32", "f64"));
    assert_refused("spmm", "nsl_sparse_spmm", ("f64", "f32"));
    assert_refused("spmv", "nsl_sparse_spmv", ("f32", "f64"));
}

#[test]
fn select_compare_and_cat_refuse_mixed_dtypes() {
    assert_refused("compare", "nsl_tensor_compare", ("f64", "f32"));
    assert_refused("where", "nsl_tensor_where", ("f64", "f32"));
    assert_refused("cat", "nsl_tensor_cat", ("f32", "f64"));
}

#[test]
fn accumulates_and_norms_refuse_mixed_dtypes() {
    assert_refused("add_inplace", "nsl_tensor_add_inplace", ("f32", "f64"));
    assert_refused("scalar_mul_add_inplace", "nsl_tensor_scalar_mul_add_inplace", ("f32", "f64"));
    assert_refused("layernorm", "nsl_tensor_layernorm", ("f32", "f64"));
    assert_refused("layernorm_bias", "nsl_tensor_layernorm", ("f32", "f64"));
    assert_refused("rmsnorm", "nsl_tensor_rmsnorm", ("f64", "f32"));
    assert_refused("scalar_rhs", "nsl_tensor_add", ("f64", "f32"));
}

// --- same dtype ----------------------------------------------------------

/// The f64 paths compute in f64: 0.1 + 0.2 is 0.30000000000000004 only there.
#[test]
fn homogeneous_f64_ops_stay_f64() {
    let a = full(&[2, 2], 0.1, F64);
    let b = full(&[2, 2], 0.2, F64);
    let sum = unsafe { nsl_tensor_add(a, b, 0) };
    assert_eq!(view(sum).dtype as i64, F64);
    assert_eq!(values(sum), vec![0.1 + 0.2; 4]);

    // 0.1 * 3 + 0.1 * 3 is 0.6000000000000001 in f64 (with or without an
    // FMA); f32 gives 0.6000000238.
    let mm = unsafe { nsl_tensor_matmul(a, full(&[2, 2], 3.0, F64), 0) };
    assert_eq!(view(mm).dtype as i64, F64);
    for v in values(mm) {
        assert!((v - 0.6000000000000001).abs() < 1e-15, "f64 matmul gave {v}");
    }

    let gt = unsafe { nsl_tensor_compare(b, a, 0) };
    assert_eq!((view(gt).dtype as i64, values(gt)), (F64, vec![1.0; 4]));

    let sel = unsafe { nsl_tensor_where(gt, a, b) };
    assert_eq!((view(sel).dtype as i64, values(sel)), (F64, vec![0.1; 4]));

    let cat = unsafe { nsl_tensor_cat(list(&[a, b]), 0) };
    assert_eq!(view(cat).dtype as i64, F64);
    assert_eq!(values(cat), [vec![0.1; 4], vec![0.2; 4]].concat());

    let fused = unsafe { nsl_fused_elementwise_2(a, b, list(&[0]), 1) };
    assert_eq!((view(fused).dtype as i64, values(fused)), (F64, vec![0.1 + 0.2; 4]));
}

/// The 16-bit elementwise path reads both operands in their shared format
/// (it used to widen each operand by its own tag): f16 + f16 and bf16 + bf16
/// stay 16-bit, with the exact sums 1 + 0.5 and 2 + 0.25.
#[test]
fn homogeneous_16_bit_add_stays_16_bit() {
    const FP16: i64 = 2;
    const BF16: i64 = 3;
    for (dtype, a_bits, b_bits, want) in [
        (FP16, [0x3C00u16, 0x4000], [0x3800u16, 0x3400], [0x3E00u16, 0x4080]),
        (BF16, [0x3F80u16, 0x4000], [0x3F00u16, 0x3E80], [0x3FC0u16, 0x4010]),
    ] {
        let make = |bits: [u16; 2]| {
            let leaked: &'static [u16] = Box::leak(Box::new(bits));
            unsafe { nsl_tensor_from_static(leaked.as_ptr() as i64, list(&[2]), dtype) }
        };
        let sum = unsafe { nsl_tensor_add(make(a_bits), make(b_bits), 0) };
        let v = view(sum);
        assert_eq!(v.dtype as i64, dtype);
        let got = unsafe { std::slice::from_raw_parts(v.data as *const u16, 2) };
        assert_eq!(got, want, "dtype {dtype}");
    }
}

/// Quantization takes a default f32 tensor (it read f64 only and aborted on
/// one), dequantizes to f32 -- the dtype of the activations it meets -- and
/// the mixed-precision matmul dequantizes in the activation's dtype.
#[test]
fn quantized_weights_meet_activations_in_their_dtype() {
    let w = full(&[4, 3], 0.75, F32);
    let qw = unsafe { nsl_qtensor_quantize(w, 0, 0, 0, 0) };
    let deq = unsafe { nsl_qtensor_dequantize(qw) };
    assert_eq!(view(deq).dtype as i64, F32);
    for v in values(deq) {
        assert!((v - 0.75).abs() < 1e-2, "dequantized {v}");
    }
    for dtype in [F32, F64] {
        let x = full(&[2, 4], 1.0, dtype);
        let y = unsafe { nsl_qtensor_matmul_mixed(x, qw) };
        assert_eq!(view(y).dtype as i64, dtype, "the result is in x's dtype");
        for v in values(y) {
            assert!((v - 3.0).abs() < 4e-2, "x @ dequant(w) = {v}");
        }
    }
}
