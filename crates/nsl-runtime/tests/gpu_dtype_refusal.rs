//! The f32-only GPU kernel paths must REFUSE a non-f32 device tensor, not
//! read it at four bytes per element and return plausible wrong numbers.
//!
//! Background. Almost every kernel wrapper in `cuda/mod.rs` sizes its output
//! `alloc_managed(n * 4)`, hardcodes dtype 1 on the tensor it publishes, and
//! casts `t.data as *const f32` — but the dispatchers reaching them branched on
//! `device > 0` alone. `nsl_tensor_contiguous` and `nsl_tensor_slice` are the
//! clearest cases: their CPU arms have been dtype-correct for years
//! (`assert_elementwise_byte_copy` + `alloc_preserved_dtype_buffer`) and were
//! simply unreachable for a device tensor. Feed an fp16/bf16 GPU tensor in and
//! the kernel reads twice its allocation: not a fault, a wrong answer, and the
//! result is then published as f32 and believed.
//!
//! Device-resident non-f32 tensors are not hypothetical. `nsl_tensor_to_device`
//! carries any dtype outside {f64, f32} to the GPU verbatim
//! (`tensor/mod.rs:3870-3891`), `nsl_tensor_zeros_like_dtype` allocates 2-byte
//! device buffers (`tensor/precision_cast.rs:395-420`), `nsl_tensor_zeros_f16_on`
//! is emitted six times per CSHA fused backward, and CCR's
//! `--checkpoint-compress bf16` publishes bf16 halves. Each is kept away from
//! the f32 kernels today by an explicit widen the compiler emits; this file pins
//! what happens the day one is not.
//!
//! # Why every gate here is a subprocess, and not `#[should_panic]`
//!
//! **An `assert!` inside a `pub extern "C"` function does not panic — it
//! ABORTS.** Since Rust 1.81 an unwind that reaches an `extern "C"` frame is
//! turned into `panic_cannot_unwind` → `SIGABRT`. Every entry point in this
//! runtime is `extern "C"`, so `#[should_panic(expected = "f32-only")]` cannot
//! observe any of these refusals: the process is gone before libtest regains
//! control. The first draft of this file used `should_panic` and died with
//! `thread caused non-unwinding panic. aborting.` / `signal: 6, SIGABRT`.
//!
//! So each gate re-execs THIS test binary with `NSL_DTYPE_REFUSAL_SCENARIO`
//! set, and asserts on the child's exit status and stderr. That is strictly
//! stronger than `should_panic` anyway: `expected = "..."` only substring-matches
//! a panic payload, whereas this checks the operand name, the offending dtype
//! tag, and the `f32-only` phrase together — so a gate cannot pass because some
//! unrelated earlier shape or device assert fired first.
//!
//! The child pattern (a `zz_*_child` test that returns immediately unless its
//! env var is set) is the same one `matmul_transposed_operand.rs:493-527` uses.
//!
//! The `#[ignore]` reasons deliberately say "re-execs", not "spawns": `spawns`
//! is the classifier keyword for the `multiproc` tier
//! (`scripts/gpu-gate-inventory.awk:95`), which the nightly does not run
//! (`gpu-cert.yml:87` defaults to the `gpu` tier). These are ordinary
//! sub-second GPU gates that happen to fork once — they belong in the `gpu`
//! tier, not alongside the three-process ZeRO training runs.
//!
//! Running locally:
//!
//! ```bash
//! cargo test --package nsl-runtime --features cuda \
//!     --test gpu_dtype_refusal -- --ignored --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_runtime::nsl_cuda_init;

// Entry points under test. All are `pub extern "C"` in the runtime.
extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_zeros_like_dtype(template_ptr: i64, dtype: i64) -> i64;
    fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64;
    fn nsl_tensor_contiguous(tensor_ptr: i64) -> i64;
    fn nsl_tensor_slice(tensor_ptr: i64, dim: i64, start: i64, end: i64) -> i64;
    fn nsl_tensor_relu(tensor_ptr: i64) -> i64;
    fn nsl_tensor_softmax(tensor_ptr: i64, dim: i64) -> i64;
    fn nsl_tensor_matmul(a_ptr: i64, b_ptr: i64, flags: u8) -> i64;
    fn nsl_rmsnorm_dx_backward(dy_ptr: i64, x_ptr: i64, gamma_ptr: i64, eps: f64) -> i64;
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
}

/// Minimal mirror of the runtime's `NslTensor` header. The real type is not in
/// the public API; only the fields these tests read are reproduced. Field order
/// and widths follow `NslTensor`'s `#[repr(C)]` layout
/// (`crates/nsl-runtime/src/tensor/mod.rs`); this is the same mirror
/// `precision_cast_gpu_dispatch.rs` carries.
#[repr(C)]
struct TensorView {
    _magic: u32,
    _data: *mut std::ffi::c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    ndim: i64,
    len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

const DTYPE_F32: i64 = 1;
const DTYPE_BF16: i64 = 3;

/// Env var naming which poisoned call the child should make.
const SCENARIO: &str = "NSL_DTYPE_REFUSAL_SCENARIO";

// --- shared helpers ------------------------------------------------------

fn shape_list(dims: &[i64]) -> i64 {
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    list
}

fn view(ptr: i64) -> &'static TensorView {
    unsafe { &*(ptr as *const TensorView) }
}

/// NOTE the wording of the message below: it must NOT contain the literal
/// text of an ignore attribute. `scripts/gpu-gate-inventory.awk:158` matches
/// `#[ignore` anywhere on a non-comment line, and this is a multi-line string
/// whose lines end in `\` — the scanner's continuation loop would swallow the
/// following lines and attach the gate to whatever function came next. The
/// first draft of this file did exactly that: the manifest gained a phantom
/// `assert_is_bf16_on_device` row classified `gpu-inferred`.
fn require_cuda() {
    let rc = nsl_cuda_init();
    assert_eq!(
        rc, 0,
        "nsl_cuda_init returned {rc}: these gates are ignore-gated and only run \
         where a CUDA device is a precondition, so this is a lane \
         misconfiguration, not a skip"
    );
}

/// Assert the constructed input really is the thing under test.
///
/// Called immediately before every poisoned call. Without it, a regression in
/// the PRODUCER — a `nsl_tensor_to_device` that quietly widened bf16 to f32,
/// say — would leave every gate here green while exercising nothing. That is
/// the "IDENTICAL-from-empty-output" trap that faked two A/B comparisons on
/// 2026-07-30. It runs inside the child, so its failure message reaches the
/// parent's stderr comparison and fails the gate rather than satisfying it.
fn assert_is_bf16_on_device(ptr: i64, what: &str) {
    let v = view(ptr);
    assert_eq!(
        v.device, 1,
        "PRODUCER-REGRESSION {what}: expected a device-resident tensor, got \
         device={}",
        v.device
    );
    assert_eq!(
        v.dtype,
        DTYPE_BF16 as u16,
        "PRODUCER-REGRESSION {what}: expected DTYPE_BF16 ({DTYPE_BF16}), got \
         dtype={} — the producer widened the tensor and this gate would have \
         been vacuous",
        v.dtype
    );
}

/// Producer (a): a host bf16 buffer promoted to the device.
///
/// `nsl_tensor_from_static` writes the dtype tag through unvalidated, and
/// `nsl_tensor_to_device` carries any dtype outside {f64, f32} to the device
/// byte-for-byte — so this needs no feature flag and no compiler involvement.
/// The bit pattern is bf16 1.0 (`0x3F80`), which is also a perfectly plausible
/// pair of f32 bytes: misreading it does not crash, which is the whole problem.
fn bf16_gpu_from_host(dims: &[i64]) -> i64 {
    let n: i64 = dims.iter().product();
    let bits: Box<[u16]> = vec![0x3F80u16; n as usize].into_boxed_slice();
    let leaked: &'static [u16] = Box::leak(bits);
    let list = shape_list(dims);
    let cpu = unsafe { nsl_tensor_from_static(leaked.as_ptr() as i64, list, DTYPE_BF16) };
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };
    unsafe { nsl_list_free(list) };
    assert_is_bf16_on_device(gpu, "bf16_gpu_from_host");
    gpu
}

/// Producer (b): the runtime's own device bf16 allocator, reached from NSL via
/// `nsl_tensor_zeros_like_dtype`. This is the path the reduced-precision
/// optimizer-moment work (`--wggo-moment-precision`) takes, so it is the more
/// production-like of the two constructions.
fn bf16_gpu_zeros_like_f32(dims: &[i64]) -> i64 {
    let n: i64 = dims.iter().product();
    let vals: Box<[f32]> = vec![1.0f32; n as usize].into_boxed_slice();
    let leaked: &'static [f32] = Box::leak(vals);
    let list = shape_list(dims);
    let cpu = unsafe { nsl_tensor_from_static(leaked.as_ptr() as i64, list, DTYPE_F32) };
    let template = unsafe { nsl_tensor_to_device(cpu, 1) };
    unsafe { nsl_list_free(list) };
    assert_eq!(
        view(template).dtype,
        DTYPE_F32 as u16,
        "template must be f32 on the device, else zeros_like_dtype is not the \
         thing producing the bf16"
    );
    let out = unsafe { nsl_tensor_zeros_like_dtype(template, DTYPE_BF16) };
    assert_is_bf16_on_device(out, "bf16_gpu_zeros_like_f32");
    out
}

// --- child ---------------------------------------------------------------

/// The poisoned calls, one per scenario name.
///
/// Cargo runs every `#[test]` in the binary, so this has to be a test; it
/// returns immediately unless the parent asked for a scenario. It is expected
/// to ABORT — `assert_gpu_f32` panics inside a `pub extern "C"` frame, which
/// Rust turns into `panic_cannot_unwind`.
#[test]
fn zz_dtype_refusal_child() {
    let Ok(scenario) = std::env::var(SCENARIO) else {
        return;
    };
    require_cuda();

    match scenario.as_str() {
        // A NON-contiguous input: a contiguous tensor takes the refcount-bump
        // fast path and never reaches the kernel, so only a real view
        // exercises the guard. Highest-reach site in the change —
        // `nsl_tensor_contiguous` is called by essentially every GPU kernel
        // wrapper, the tensor tracer, and matmul operand preparation.
        "contiguous" => {
            let t = bf16_gpu_from_host(&[4, 3]);
            let transposed = unsafe { nsl_tensor_transpose(t, 0, 1) };
            assert_is_bf16_on_device(transposed, "transposed view");
            let _ = unsafe { nsl_tensor_contiguous(transposed) };
        }
        // `nsl_tensor_slice`'s GPU arm, which dispatched on `device > 0` alone
        // while its own CPU arm was dtype-aware.
        "slice" => {
            let t = bf16_gpu_from_host(&[4, 3]);
            let _ = unsafe { nsl_tensor_slice(t, 0, 1, 3) };
        }
        // Reaches `gpu_elementwise_unary_inplace`, because a freshly produced
        // tensor is uniquely owned and `nsl_tensor_relu` takes the FBIP arm.
        // That arm was missed by the first draft of the static drift gate: it
        // takes the kernel NAME as a parameter, so the `_f32` literal lives at
        // its ~25 call sites and not in the function reading the tensor.
        "relu" => {
            let t = bf16_gpu_zeros_like_f32(&[4, 3]);
            let _ = unsafe { nsl_tensor_relu(t) };
        }
        // Last-dim softmax: `nsl_tensor_contiguous` is a no-op here (the
        // tensor IS contiguous), so the refusal must come from
        // `gpu_softmax_f32`'s own guard, not from the one two frames up.
        "softmax" => {
            let t = bf16_gpu_from_host(&[4, 3]);
            let _ = unsafe { nsl_tensor_softmax(t, -1) };
        }
        // A training-hot backward with three tensor operands. Everything
        // upstream of the kernel (`to_device`, `contiguous`) is a no-op for an
        // already-contiguous device tensor.
        "rmsnorm_dx" => {
            let dy = bf16_gpu_from_host(&[2, 3]);
            let x = bf16_gpu_from_host(&[2, 3]);
            let gamma = bf16_gpu_from_host(&[3]);
            let _ = unsafe { nsl_rmsnorm_dx_backward(dy, x, gamma, 1e-6) };
        }
        // Retro-coverage for `gpu_matmul_f32`'s pre-existing assert, added
        // with item 9 and carrying ZERO coverage tree-wide until now: nothing
        // anywhere asserted a non-f32 matmul is refused, so the check could
        // have been deleted in a refactor with every gate still green.
        "matmul" => {
            let a = bf16_gpu_from_host(&[2, 3]);
            let b = bf16_gpu_from_host(&[3, 2]);
            let _ = unsafe { nsl_tensor_matmul(a, b, 0) };
        }
        other => panic!("unknown {SCENARIO} '{other}'"),
    }

    // Only reached if the guard did NOT fire. Distinctive text so the parent
    // can tell "refused as designed" from "sailed straight through".
    println!("GUARD-DID-NOT-FIRE scenario={scenario}");
}

// --- parent --------------------------------------------------------------

/// Run one scenario in a fresh process and assert it was refused.
///
/// Requires all three of: a non-zero exit, the `f32-only` phrase, and the
/// offending dtype tag reported as 3. The three together are what stop a gate
/// passing because an unrelated earlier assert fired — the failure mode
/// `should_panic` cannot distinguish.
fn assert_refuses(scenario: &str, expected_operand: &str) {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_dtype_refusal_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
            // The child is expected to abort; without this libtest's own
            // ignore filter would still have run it, but being explicit keeps
            // the child selection independent of how the parent was invoked.
            "--include-ignored",
        ])
        .env(SCENARIO, scenario)
        // The abort path prints a ~50-frame backtrace that buries the message
        // we are matching on and slows the gate down for nothing.
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("failed to re-exec the test binary");

    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        !stdout.contains("GUARD-DID-NOT-FIRE"),
        "scenario '{scenario}': the bf16 operand went straight through the f32 \
         kernel — the guard is missing or was removed.\n--- child stdout ---\n\
         {stdout}\n--- child stderr ---\n{stderr}"
    );
    assert!(
        !out.status.success(),
        "scenario '{scenario}': child exited 0; a refusal must terminate the \
         process.\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
    );
    assert!(
        !stderr.contains("PRODUCER-REGRESSION"),
        "scenario '{scenario}': the child died building its INPUT, not on the \
         guard — this gate would otherwise have passed vacuously.\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("f32-only"),
        "scenario '{scenario}': child failed, but not with the dtype refusal.\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("got dtype 3") || stderr.contains("dtype 3 @"),
        "scenario '{scenario}': refusal fired but did not name bf16 (dtype 3) as \
         the offending dtype, so it may be refusing something else.\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains(expected_operand),
        "scenario '{scenario}': refusal did not name the expected operand \
         '{expected_operand}'.\n--- child stderr ---\n{stderr}"
    );
}

/// Anti-vacuity, and the only gate here that runs in-process.
///
/// Both poisoned-input constructions must actually produce a device-resident
/// bf16 tensor of the right shape. If either producer regresses, this fails
/// with a plain assertion instead of leaving six refusal gates to pass for the
/// wrong reason.
#[test]
#[ignore = "requires CUDA GPU"]
fn both_producers_yield_a_real_device_resident_bf16_tensor() {
    require_cuda();

    let a = bf16_gpu_from_host(&[4, 3]);
    assert_eq!(view(a).ndim, 2);
    assert_eq!(view(a).len, 12);

    let b = bf16_gpu_zeros_like_f32(&[4, 3]);
    assert_eq!(view(b).ndim, 2);
    assert_eq!(view(b).len, 12);
}

#[test]
#[ignore = "requires CUDA GPU; re-execs this test binary to observe the abort"]
fn contiguous_refuses_a_transposed_bf16_device_view() {
    assert_refuses("contiguous", "nsl_tensor_contiguous");
}

#[test]
#[ignore = "requires CUDA GPU; re-execs this test binary to observe the abort"]
fn slice_refuses_a_bf16_device_tensor() {
    assert_refuses("slice", "nsl_tensor_slice");
}

#[test]
#[ignore = "requires CUDA GPU; re-execs this test binary to observe the abort"]
fn relu_refuses_a_bf16_device_tensor() {
    assert_refuses("relu", "nsl_relu_f32");
}

#[test]
#[ignore = "requires CUDA GPU; re-execs this test binary to observe the abort"]
fn softmax_refuses_a_bf16_device_tensor() {
    assert_refuses("softmax", "softmax_f32");
}

#[test]
#[ignore = "requires CUDA GPU; re-execs this test binary to observe the abort"]
fn rmsnorm_dx_backward_refuses_bf16_device_operands() {
    assert_refuses("rmsnorm_dx", "rmsnorm_dx_backward_f32");
}

#[test]
#[ignore = "requires CUDA GPU; re-execs this test binary to observe the abort"]
fn matmul_refuses_bf16_device_operands() {
    assert_refuses("matmul", "GPU matmul is f32-only");
}
