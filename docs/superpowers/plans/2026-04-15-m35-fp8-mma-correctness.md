# M35 FP8 MMA Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the correctness floor for FP8 MMA — six integration test files under `crates/nsl-runtime/tests/`, validating E4M3 forward + E5M2 backward + cast + scale against a CPU scalar FP8 reference.

**Architecture:** Each test file is an independent integration-test binary. A shared `tests/common/fp8_reference.rs` module provides the ground-truth scalar FP8 reference (quantize → dequantize → sequential FMA). Tests exercise the runtime FFI (`nsl_fp8_cast`, `nsl_fp8_matmul`, `nsl_fp8_compute_scale`, `fp8_matmul_e5m2_backward`) and assert outputs within tolerances published in the spec (2% E4M3, 10% E5M2, 1e-5 MMA-vs-fallback).

**Tech Stack:** Rust 2021, `cargo test`, existing `nsl-runtime` FP8 module (`crates/nsl-runtime/src/fp8.rs`), `rand` for deterministic seeding (already a workspace dep), `#[cfg(feature = "cuda")]` for MMA-path gating.

**Spec:** [`docs/superpowers/specs/2026-04-15-m35-fp8-mma-correctness-design.md`](../specs/2026-04-15-m35-fp8-mma-correctness-design.md)

---

## File Structure

| Path | Responsibility |
|---|---|
| `crates/nsl-runtime/tests/common/mod.rs` | Cargo integration-test convention — declares `pub mod fp8_reference;` so each test file can `mod common;` then `use common::fp8_reference::*;`. |
| `crates/nsl-runtime/tests/common/fp8_reference.rs` | Shared reference: format constants, tolerances, quantize/dequantize helpers, scalar matmul reference, deterministic input generator, tensor-builder helper. |
| `crates/nsl-runtime/tests/fp8_cast.rs` | f32 → FP8 → f32 round-trip (E4M3, E5M2). |
| `crates/nsl-runtime/tests/fp8_scale.rs` | `nsl_fp8_compute_scale` correctness across formats + edge cases. |
| `crates/nsl-runtime/tests/fp8_matmul_forward.rs` | E4M3 forward matmul vs reference, PerTensor, shapes 16/32/64/128. |
| `crates/nsl-runtime/tests/fp8_matmul_backward.rs` | E5M2 backward matmul vs reference, same shape set. |
| `crates/nsl-runtime/tests/fp8_dispatcher.rs` | MMA path == fallback path (feature-gated `cuda` + sm_89+). |

Each test file declares `mod common;` and pulls helpers via `use common::fp8_reference::*;`. Cargo's integration-test convention treats `tests/common/mod.rs` as a shared module (not its own test binary).

---

## Task 1: Shared reference harness

**Files:**
- Create: `crates/nsl-runtime/tests/common/mod.rs`
- Create: `crates/nsl-runtime/tests/common/fp8_reference.rs`

This is the ground-truth scalar FP8 reference every other test depends on. Write it first; ship without dependent tests so subsequent tasks can begin.

- [ ] **Step 1: Create `tests/common/mod.rs`**

```rust
// Shared helpers for FP8 integration tests. Declared as a module by every
// test file via `mod common;`. Cargo treats `tests/common/mod.rs` as a
// shared submodule, not its own test binary.
#![allow(dead_code)]

pub mod fp8_reference;
```

- [ ] **Step 2: Create `tests/common/fp8_reference.rs` with constants and quantize helpers**

```rust
//! Ground-truth scalar FP8 reference for integration tests.
//!
//! Every helper here is independent of the runtime's FP8 implementation.
//! If a test fails because the reference and the runtime disagree, triage
//! by inspecting the reference first (it must be bit-correct against the
//! published FP8 format spec) before assuming the runtime is at fault.

use nsl_runtime::fp8::{
    compute_scale, dequantize_fp8, fp8_matmul_cpu, quantize_fp8, FP8E4M3_MAX, FP8E5M2_MAX,
    FP8_FORMAT_E4M3, FP8_FORMAT_E5M2,
};

// ---------------------------------------------------------------------------
// Format identification + tolerance constants
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

impl Fp8Format {
    pub fn dtype_code(self) -> i64 {
        match self {
            Fp8Format::E4M3 => FP8_FORMAT_E4M3,
            Fp8Format::E5M2 => FP8_FORMAT_E5M2,
        }
    }

    pub fn max_repr(self) -> f32 {
        match self {
            Fp8Format::E4M3 => FP8E4M3_MAX,
            Fp8Format::E5M2 => FP8E5M2_MAX,
        }
    }
}

/// Max-abs relative tolerance for E4M3 matmul vs reference.
/// Source: published FP8 quantization-noise bounds for E4M3 (~1-2% rel err).
pub const E4M3_REL_TOL: f32 = 0.02;

/// Max-abs relative tolerance for E5M2 matmul vs reference.
/// Source: published FP8 quantization-noise bounds for E5M2 (~5-10% rel err).
pub const E5M2_REL_TOL: f32 = 0.10;

/// Max-abs absolute tolerance for MMA path vs CPU fallback.
/// 1e-5 covers tensor-core tiled-reduction vs scalar-FMA accumulation
/// ordering at shapes up to 128x128. Divergence beyond 1e-5 indicates an
/// algorithmic difference, not floating-point physics.
pub const DISPATCH_ABS_TOL: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Format-aware quantize / dequantize that match the runtime's scalar helpers.
// We delegate to the runtime's `quantize_fp8`/`dequantize_fp8` because those
// are the same scalar primitives the FFI uses internally — testing against
// them proves the FFI plumbing, not the scalar arithmetic.
// ---------------------------------------------------------------------------

pub fn quantize(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    quantize_fp8(x as f64, scale as f64, fmt.dtype_code()) as f32
}

pub fn dequantize(q: f32, scale: f32) -> f32 {
    dequantize_fp8(q as f64, scale as f64) as f32
}

/// Round-trip f32 through FP8 and back. Convenience for cast tests.
pub fn round_trip(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    dequantize(quantize(x, scale, fmt), scale)
}

/// Compute the format's quantization-step size at the magnitude of `x`.
/// FP8 has non-uniform spacing, so this is a per-value bound — not a single
/// uniform epsilon.
pub fn quantization_step(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    let precision = match fmt {
        Fp8Format::E4M3 => 0.125,
        Fp8Format::E5M2 => 0.5,
    };
    // The runtime quantizes `value/scale` rounded to `precision`. The worst-
    // case absolute error in the dequantized output is `precision * scale / 2`
    // for any value within the representable range, plus the clamp error
    // when |x| > scale * max_repr (handled separately by the caller).
    (precision as f32) * scale / 2.0
}
```

- [ ] **Step 3: Append the scalar-matmul reference to `fp8_reference.rs`**

```rust
// ---------------------------------------------------------------------------
// Scalar FP8 matmul reference: quantize → dequantize → sequential FMA.
// ---------------------------------------------------------------------------

pub struct Fp8ReferenceMatmul {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub format: Fp8Format,
    /// Scale used for both A and B (PerTensor — single scale).
    pub scale: f32,
}

impl Fp8ReferenceMatmul {
    /// Run the reference: dequantize each input element via the
    /// quantize→dequantize round-trip (simulating FP8 precision loss), then
    /// sequential f64-accumulated matmul, then return f32. Matches the
    /// runtime's `fp8_matmul_cpu` semantics exactly.
    pub fn compute_f32(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), self.m * self.k, "A shape mismatch");
        assert_eq!(b.len(), self.k * self.n, "B shape mismatch");

        let dq_a: Vec<f64> = a
            .iter()
            .map(|&v| round_trip(v, self.scale, self.format) as f64)
            .collect();
        let dq_b: Vec<f64> = b
            .iter()
            .map(|&v| round_trip(v, self.scale, self.format) as f64)
            .collect();

        fp8_matmul_cpu(&dq_a, &dq_b, self.m, self.k, self.n)
            .into_iter()
            .map(|v| v as f32)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Deterministic input generation. Always seed from a fixed u64 — bit-
// identical across runs.
// ---------------------------------------------------------------------------

pub fn seeded_input(len: usize, seed: u64) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

/// Standard shape set covered by matmul tests.
pub const FIXTURE_SHAPES: &[(usize, usize, usize)] = &[
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
];

/// Compute scale via the runtime's helper from the union of two arrays.
/// PerTensor convention: one scale covers both matmul inputs.
pub fn compute_pertensor_scale(a: &[f32], b: &[f32], fmt: Fp8Format) -> f32 {
    let combined: Vec<f64> = a.iter().chain(b.iter()).map(|&v| v as f64).collect();
    compute_scale(&combined, fmt.dtype_code()) as f32
}
```

- [ ] **Step 4: Add public tensor-builder helpers to the runtime under the `test-hooks` feature**

`NslTensor` fields are `pub(crate)`, so integration tests in `tests/` can't construct or read tensors directly. Add two `pub` helpers gated on the existing `test-hooks` feature (used elsewhere in the runtime — see `crates/nsl-runtime/src/memory.rs:73`).

Open `crates/nsl-runtime/src/tensor/mod.rs` and append (after the existing helpers near the bottom of the file, before `#[cfg(test)]`):

```rust
// ---------------------------------------------------------------------------
// Test-only helpers — build/read tensors from f32 slices for integration
// tests in `tests/*.rs`. Gated on the `test-hooks` feature so production
// builds don't expose this surface.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-hooks")]
pub fn test_build_tensor_2d_f32(rows: usize, cols: usize, data: &[f32]) -> i64 {
    assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
    let shape = crate::list::nsl_list_new();
    crate::list::nsl_list_push(shape, rows as i64);
    crate::list::nsl_list_push(shape, cols as i64);
    let ptr = nsl_tensor_zeros(shape);

    let t = unsafe { &*(ptr as *const NslTensor) };
    let buf = t.data as *mut f32;
    for (i, &v) in data.iter().enumerate() {
        unsafe { *buf.add(i) = v };
    }
    crate::list::nsl_list_free(shape);
    ptr
}

#[cfg(feature = "test-hooks")]
pub fn test_read_tensor_f32(ptr: i64) -> Vec<f32> {
    let t = unsafe { &*(ptr as *const NslTensor) };
    assert_eq!(t.dtype, 1, "expected f32 tensor (dtype=1), got dtype={}", t.dtype);
    let len = t.len as usize;
    let buf = t.data as *const f32;
    (0..len).map(|i| unsafe { *buf.add(i) }).collect()
}
```

Then append the harness re-export to `tests/common/fp8_reference.rs`:

```rust
// ---------------------------------------------------------------------------
// Tensor builder/reader: re-exports of the runtime's test-hooks helpers.
// All FP8 integration tests must be run with --features test-hooks.
// ---------------------------------------------------------------------------

pub use nsl_runtime::tensor::{test_build_tensor_2d_f32 as make_tensor_2d_f32,
                               test_read_tensor_f32 as read_tensor_f32};
```

- [ ] **Step 5: Append the assertion helper to `fp8_reference.rs`**

```rust
// ---------------------------------------------------------------------------
// Assertion helpers — print informative messages so failures are debuggable
// without rerunning the test.
// ---------------------------------------------------------------------------

/// Assert relative error is within tolerance. Guards against zero references.
pub fn assert_rel_err_le(test: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(test.len(), reference.len(), "{label}: length mismatch");
    let max_ref_abs = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_ref_abs < 1e-12 {
        // Reference is effectively zero — fall back to absolute tolerance.
        let max_abs_err = test
            .iter()
            .zip(reference)
            .map(|(t, r)| (t - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_err < tol,
            "{label}: reference is ~zero; max abs err {max_abs_err} > tol {tol}"
        );
        return;
    }
    let max_rel_err = test
        .iter()
        .zip(reference)
        .map(|(t, r)| (t - r).abs() / max_ref_abs)
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel_err <= tol,
        "{label}: max rel err {max_rel_err} > tol {tol} (max_ref_abs = {max_ref_abs})"
    );
}

/// Assert absolute error is within tolerance.
pub fn assert_abs_err_le(test: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(test.len(), reference.len(), "{label}: length mismatch");
    let max_abs_err = test
        .iter()
        .zip(reference)
        .map(|(t, r)| (t - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err <= tol,
        "{label}: max abs err {max_abs_err} > tol {tol}"
    );
}
```

- [ ] **Step 6: Verify the harness builds**

Run: `cargo test -p nsl-runtime --no-run --tests --features test-hooks`
Expected: compiles (no test runs since no `#[test]` attributes — this just verifies syntax + imports).

If a referenced symbol (`nsl_runtime::fp8::compute_scale`, `nsl_runtime::list::nsl_list_new`, etc.) is private, make it `pub` in the runtime crate in this same commit and note the change in the commit message.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/tests/common/
git commit -m "test(fp8): shared reference harness for M35 correctness suite

Common scalar FP8 reference (quantize-dequantize-sequential FMA),
tolerance constants (2% E4M3, 10% E5M2, 1e-5 dispatcher), seeded
input generation, NslTensor builder/reader helpers, rel/abs assertion
helpers. No tests yet; this is the foundation the next 5 task files
build on."
```

---

## Task 2: f32 ↔ FP8 cast round-trip (`fp8_cast.rs`)

**Files:**
- Create: `crates/nsl-runtime/tests/fp8_cast.rs`

Validates `nsl_fp8_cast` round-trip preserves values within the FP8 quantization step bound for both formats. Uses the FFI path end-to-end (build tensor → call FFI → read result → compare).

- [ ] **Step 1: Write the file with E4M3 round-trip test**

```rust
//! Integration tests for nsl_fp8_cast — f32 → FP8 → f32 round-trip
//! correctness across E4M3 and E5M2 formats.

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{nsl_fp8_cast, FP8_FORMAT_E4M3, FP8_FORMAT_E5M2};
use nsl_runtime::tensor::nsl_tensor_free;

/// E4M3: each element of a small fixture must round-trip within the
/// format's quantization-step bound at that value's magnitude.
#[test]
fn e4m3_cast_round_trip_within_step() {
    let data: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 100.0];
    let scale = compute_pertensor_scale(&data, &[], Fp8Format::E4M3);

    let input_ptr = make_tensor_2d_f32(1, data.len(), &data);
    let output_ptr = nsl_fp8_cast(input_ptr, FP8_FORMAT_E4M3, scale as f64);
    let output = read_tensor_f32(output_ptr);

    for (i, (&orig, &deq)) in data.iter().zip(&output).enumerate() {
        let bound = quantization_step(orig, scale, Fp8Format::E4M3);
        let err = (orig - deq).abs();
        assert!(
            err <= bound + 1e-6, // 1e-6 absolute slack for fp32 representation
            "E4M3 cast at index {i}: orig={orig} deq={deq} err={err} bound={bound}"
        );
    }

    nsl_tensor_free(input_ptr);
    nsl_tensor_free(output_ptr);
}
```

- [ ] **Step 2: Run the test, expect a clean pass or surface a bug**

Run: `cargo test -p nsl-runtime --test fp8_cast --features test-hooks -- --nocapture`
Expected: PASS. If FAIL, follow the spec's bug-triage order:
1. Reference correct? Inspect `quantization_step` against the FP8 spec.
2. Tolerance reasonable? `bound + 1e-6` covers fp32 representation slack.
3. Format/scale path? Print `scale` and the failing element's quantized value.
4. Fix in the same commit; comment in the code explaining the bug and the fix.

- [ ] **Step 3: Append the E5M2 test**

```rust
/// E5M2: same round-trip with a wider-range fixture (E5M2 sacrifices
/// precision for range — values up to ~57344 are representable).
#[test]
fn e5m2_cast_round_trip_within_step() {
    let data: Vec<f32> = vec![-1000.0, -10.0, 0.0, 0.5, 1.0, 100.0, 5000.0, 50000.0];
    let scale = compute_pertensor_scale(&data, &[], Fp8Format::E5M2);

    let input_ptr = make_tensor_2d_f32(1, data.len(), &data);
    let output_ptr = nsl_fp8_cast(input_ptr, FP8_FORMAT_E5M2, scale as f64);
    let output = read_tensor_f32(output_ptr);

    for (i, (&orig, &deq)) in data.iter().zip(&output).enumerate() {
        let bound = quantization_step(orig, scale, Fp8Format::E5M2);
        let err = (orig - deq).abs();
        assert!(
            err <= bound + 1e-6,
            "E5M2 cast at index {i}: orig={orig} deq={deq} err={err} bound={bound}"
        );
    }

    nsl_tensor_free(input_ptr);
    nsl_tensor_free(output_ptr);
}
```

- [ ] **Step 4: Append the auto-scale test**

```rust
/// Cast with `scale=0.0` triggers auto-scale — the cast computes scale
/// internally from the input. Verify the auto-computed scale matches what
/// `nsl_fp8_compute_scale` would have produced.
#[test]
fn cast_auto_scale_matches_explicit_scale() {
    let data: Vec<f32> = seeded_input(64, 42);
    let explicit_scale = compute_pertensor_scale(&data, &[], Fp8Format::E4M3);

    let input_a = make_tensor_2d_f32(1, data.len(), &data);
    let out_explicit = nsl_fp8_cast(input_a, FP8_FORMAT_E4M3, explicit_scale as f64);

    let input_b = make_tensor_2d_f32(1, data.len(), &data);
    let out_auto = nsl_fp8_cast(input_b, FP8_FORMAT_E4M3, 0.0);

    let v_explicit = read_tensor_f32(out_explicit);
    let v_auto = read_tensor_f32(out_auto);

    assert_abs_err_le(&v_auto, &v_explicit, 1e-6,
        "auto-scale must match explicit scale exactly");

    nsl_tensor_free(input_a);
    nsl_tensor_free(input_b);
    nsl_tensor_free(out_explicit);
    nsl_tensor_free(out_auto);
}
```

- [ ] **Step 5: Run all `fp8_cast` tests**

Run: `cargo test -p nsl-runtime --test fp8_cast --features test-hooks`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/tests/fp8_cast.rs
git commit -m "test(fp8): cast round-trip integration tests for E4M3 + E5M2

Validates nsl_fp8_cast round-trip preserves values within the FP8
quantization-step bound at each element's magnitude. Covers explicit
scale + auto-scale (scale=0.0) paths. Uses the FFI end-to-end with
the shared reference harness from Task 1."
```

If any bug surfaced, append `[fixes-bug]` and a one-line description to the commit body.

---

## Task 3: `compute_scale` correctness (`fp8_scale.rs`)

**Files:**
- Create: `crates/nsl-runtime/tests/fp8_scale.rs`

Validates `nsl_fp8_compute_scale` against a hand-computed expected scale, with edge cases for all-zero, single-element, and large-magnitude inputs.

- [ ] **Step 1: Write the file with E4M3 + E5M2 baseline tests**

```rust
//! Integration tests for nsl_fp8_compute_scale — verifies scale =
//! max_abs / FP8_MAX for both formats, plus edge cases.

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{nsl_fp8_compute_scale, FP8_FORMAT_E4M3, FP8_FORMAT_E5M2,
                       FP8E4M3_MAX, FP8E5M2_MAX};
use nsl_runtime::tensor::nsl_tensor_free;

/// Known max=4.0 → E4M3 scale = 4.0 / 448.0.
#[test]
fn e4m3_scale_matches_max_div_format_max() {
    let data: Vec<f32> = vec![1.0, -4.0, 2.0, 0.5];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 4.0_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON,
        "E4M3 scale {scale} != expected {expected}");

    nsl_tensor_free(ptr);
}

/// Known max=10000.0 → E5M2 scale = 10000.0 / 57344.0.
#[test]
fn e5m2_scale_matches_max_div_format_max() {
    let data: Vec<f32> = vec![1.0, -10000.0, 2.0, 50.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E5M2) as f32;
    let expected = 10000.0_f32 / FP8E5M2_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON,
        "E5M2 scale {scale} != expected {expected}");

    nsl_tensor_free(ptr);
}
```

- [ ] **Step 2: Append edge-case tests**

```rust
/// All-zero input: scale must be the sentinel 1.0 (not 0.0, which
/// would cause divide-by-zero in downstream cast).
#[test]
fn all_zero_input_returns_sentinel_one() {
    let data: Vec<f32> = vec![0.0; 16];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3);
    assert_eq!(scale, 1.0, "all-zero input must return sentinel 1.0");

    nsl_tensor_free(ptr);
}

/// Single-element tensor: scale = |element| / FP8_MAX.
#[test]
fn single_element_e4m3() {
    let data: Vec<f32> = vec![3.5];
    let ptr = make_tensor_2d_f32(1, 1, &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 3.5_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON);

    nsl_tensor_free(ptr);
}

/// Negative-only input: max_abs takes |.|, so scale uses the magnitude.
#[test]
fn negative_only_input_uses_magnitude() {
    let data: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 4.0_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON);

    nsl_tensor_free(ptr);
}

/// Input that exceeds E4M3's representable range: scale must scale the
/// largest element to land at exactly FP8E4M3_MAX after dividing by scale.
#[test]
fn large_magnitude_e4m3_clamps_to_max() {
    let data: Vec<f32> = vec![10000.0, -5000.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    // The largest element 10000.0, divided by scale, should equal FP8E4M3_MAX.
    let scaled_max = 10000.0_f32 / scale;
    assert!((scaled_max - FP8E4M3_MAX).abs() <= 1e-3,
        "scaled max {scaled_max} should equal {FP8E4M3_MAX}");

    nsl_tensor_free(ptr);
}

/// f64 tensor input (dtype=0) takes the f64 code path; same scale arithmetic.
/// Note: nsl_tensor_zeros produces f32 by default, so we exercise the f32
/// branch here. The f64 branch is exercised by the inline unit tests already.
#[test]
fn f32_tensor_dtype_path() {
    let data: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 2.0_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON);

    nsl_tensor_free(ptr);
}
```

- [ ] **Step 3: Run all `fp8_scale` tests**

Run: `cargo test -p nsl-runtime --test fp8_scale --features test-hooks`
Expected: 6 PASS.

If FAIL: follow triage. The most likely failure mode is the all-zero sentinel: if it returns 0.0 instead of 1.0, fix `compute_scale_f32` in `crates/nsl-runtime/src/fp8.rs:67-79` (it already has the `if max_abs == 0.0` guard for the f64 path; the f32 path has the same guard at line 74 — verify it's reached). Bundle the fix in the same commit.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/tests/fp8_scale.rs
git commit -m "test(fp8): compute_scale correctness across formats + edges

Validates nsl_fp8_compute_scale matches max_abs / FP8_MAX (1 ULP
tolerance) for E4M3 and E5M2, including all-zero sentinel, single
element, negative-only, large-magnitude clamping, and f32 dtype
dispatch."
```

---

## Task 4: E4M3 forward matmul vs reference (`fp8_matmul_forward.rs`)

**Files:**
- Create: `crates/nsl-runtime/tests/fp8_matmul_forward.rs`

The headline correctness test: FP8 matmul output must agree with the scalar reference within 2% relative error for E4M3.

- [ ] **Step 1: Write the file with the parameterized shape sweep**

```rust
//! Integration tests for E4M3 forward matmul — output of nsl_fp8_matmul
//! agrees with the scalar reference within E4M3_REL_TOL (2%).

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{nsl_fp8_cast, nsl_fp8_matmul, FP8_FORMAT_E4M3};
use nsl_runtime::tensor::nsl_tensor_free;

/// Run E4M3 matmul at one shape, assert against the scalar reference.
fn run_e4m3_at_shape(m: usize, k: usize, n: usize) {
    let a_data = seeded_input(m * k, 1);
    let b_data = seeded_input(k * n, 2);
    let scale = compute_pertensor_scale(&a_data, &b_data, Fp8Format::E4M3);

    // Reference: scalar quantize-dequantize-FMA.
    let reference = Fp8ReferenceMatmul {
        m, n, k,
        format: Fp8Format::E4M3,
        scale,
    };
    let ref_out = reference.compute_f32(&a_data, &b_data);

    // Under test: FFI cast + FFI matmul.
    let a_f32 = make_tensor_2d_f32(m, k, &a_data);
    let b_f32 = make_tensor_2d_f32(k, n, &b_data);
    let a_fp8 = nsl_fp8_cast(a_f32, FP8_FORMAT_E4M3, scale as f64);
    let b_fp8 = nsl_fp8_cast(b_f32, FP8_FORMAT_E4M3, scale as f64);

    let out_ptr = nsl_fp8_matmul(a_fp8, b_fp8, scale as f64, scale as f64);
    let test_out = read_tensor_f32(out_ptr);

    assert_rel_err_le(&test_out, &ref_out, E4M3_REL_TOL,
        &format!("E4M3 matmul {m}x{k}x{n}"));

    nsl_tensor_free(a_f32);
    nsl_tensor_free(b_f32);
    nsl_tensor_free(a_fp8);
    nsl_tensor_free(b_fp8);
    nsl_tensor_free(out_ptr);
}

#[test]
fn e4m3_matmul_16x16x16() { run_e4m3_at_shape(16, 16, 16); }

#[test]
fn e4m3_matmul_32x32x32() { run_e4m3_at_shape(32, 32, 32); }

#[test]
fn e4m3_matmul_64x64x64() { run_e4m3_at_shape(64, 64, 64); }

#[test]
fn e4m3_matmul_128x128x128() { run_e4m3_at_shape(128, 128, 128); }

/// Non-square shape — verifies dispatch handles M ≠ K ≠ N.
#[test]
fn e4m3_matmul_non_square() { run_e4m3_at_shape(64, 32, 16); }
```

- [ ] **Step 2: Run the matmul tests**

Run: `cargo test -p nsl-runtime --test fp8_matmul_forward --features test-hooks`
Expected: 5 PASS.

If FAIL: typical failure modes for FP8 matmul:
- Scale mismatch (output systematically scaled): print `scale` and verify `nsl_fp8_matmul` doesn't apply scale a second time after `nsl_fp8_cast` already did.
- Wrong dtype reaching matmul (output zeros or NaN): inspect `dtype` of `a_fp8` post-cast — should be 1 (f32) since cast dequantizes back to f32.
- Per-element systematic error within tolerance ceiling (e.g. exactly 2.0% rel err): suspect rounding mode in `quantize_fp8`. Compare reference vs runtime quantization step-by-step.

Fix in the same commit, with a comment explaining the bug.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/tests/fp8_matmul_forward.rs
git commit -m "test(fp8): E4M3 forward matmul vs scalar reference

Five shape variants (16/32/64/128/non-square). Each compares the FFI
cast + FFI matmul output against the scalar quantize-dequantize-FMA
reference within 2% relative error (E4M3 quantization noise bound)."
```

---

## Task 5: E5M2 backward matmul vs reference (`fp8_matmul_backward.rs`)

**Files:**
- Create: `crates/nsl-runtime/tests/fp8_matmul_backward.rs`

Same structure as Task 4, but tests the E5M2 backward path — `fp8_matmul_e5m2_backward` — within the wider 10% E5M2 tolerance.

- [ ] **Step 1: Write the file**

```rust
//! Integration tests for E5M2 backward matmul — output of
//! fp8_matmul_e5m2_backward agrees with the scalar reference within
//! E5M2_REL_TOL (10%).

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::fp8_matmul_e5m2_backward;
use nsl_runtime::tensor::nsl_tensor_free;

fn run_e5m2_at_shape(m: usize, k: usize, n: usize) {
    let a_data = seeded_input(m * k, 11);
    let b_data = seeded_input(k * n, 12);
    let scale = compute_pertensor_scale(&a_data, &b_data, Fp8Format::E5M2);

    // Reference (E5M2 quantization, then matmul).
    let reference = Fp8ReferenceMatmul {
        m, n, k,
        format: Fp8Format::E5M2,
        scale,
    };
    let ref_out = reference.compute_f32(&a_data, &b_data);

    // Under test: build f32 tensors, call the backward kernel which does
    // its own quantization to E5M2 internally.
    let a_ptr = make_tensor_2d_f32(m, k, &a_data);
    let b_ptr = make_tensor_2d_f32(k, n, &b_data);

    let out_ptr = fp8_matmul_e5m2_backward(a_ptr, b_ptr, scale, scale);
    let test_out = read_tensor_f32(out_ptr);

    assert_rel_err_le(&test_out, &ref_out, E5M2_REL_TOL,
        &format!("E5M2 backward matmul {m}x{k}x{n}"));

    nsl_tensor_free(a_ptr);
    nsl_tensor_free(b_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn e5m2_backward_16x16x16() { run_e5m2_at_shape(16, 16, 16); }

#[test]
fn e5m2_backward_32x32x32() { run_e5m2_at_shape(32, 32, 32); }

#[test]
fn e5m2_backward_64x64x64() { run_e5m2_at_shape(64, 64, 64); }

#[test]
fn e5m2_backward_128x128x128() { run_e5m2_at_shape(128, 128, 128); }
```

- [ ] **Step 2: Run the backward tests**

Run: `cargo test -p nsl-runtime --test fp8_matmul_backward --features test-hooks`
Expected: 4 PASS.

Triage on failure: same as Task 4. E5M2 has wider tolerance, so failures usually indicate either a real algorithm bug (not noise) or a scale-arithmetic error specific to the backward path.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/tests/fp8_matmul_backward.rs
git commit -m "test(fp8): E5M2 backward matmul vs scalar reference

Four shape variants (16/32/64/128). Compares fp8_matmul_e5m2_backward
output against the scalar E5M2 reference within 10% relative error."
```

---

## Task 6: MMA path == fallback path (`fp8_dispatcher.rs`)

**Files:**
- Create: `crates/nsl-runtime/tests/fp8_dispatcher.rs`

Compares the MMA-kernel matmul output against the CPU fallback at the same shapes. Feature-gated `cuda` + sm_89+ runtime check; falls back to `#[ignore]` on hardware that can't run it.

- [ ] **Step 1: Write the file with the gate + comparison**

```rust
//! Integration tests for FP8 matmul dispatcher — MMA kernel output and
//! CPU fallback output must agree within DISPATCH_ABS_TOL (1e-5) at all
//! tested shapes. Divergence > 1e-5 indicates an algorithmic difference,
//! not f32 accumulation-ordering physics.

mod common;
use common::fp8_reference::*;

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use nsl_runtime::fp8::{nsl_fp8_cast, nsl_fp8_matmul, FP8_FORMAT_E4M3};
    use nsl_runtime::tensor::nsl_tensor_free;

    /// Returns true if the current device supports FP8 mma.sync (sm_89+).
    /// Ada Lovelace (RTX 4000 series) is sm_89; Hopper (H100) is sm_90;
    /// Blackwell is sm_100/sm_120. All support FP8 mma.sync.
    /// Uses the runtime's `test_detect_sm_version` helper added in Step 2.
    fn is_sm_89_plus() -> bool {
        nsl_runtime::test_detect_sm_version() >= 89
    }

    fn run_dispatch_at_shape(m: usize, k: usize, n: usize) {
        if !is_sm_89_plus() {
            eprintln!("skipping {m}x{k}x{n}: device < sm_89");
            return;
        }
        let a_data = seeded_input(m * k, 21);
        let b_data = seeded_input(k * n, 22);
        let scale = compute_pertensor_scale(&a_data, &b_data, Fp8Format::E4M3);

        // MMA path: nsl_fp8_matmul.
        let a_f32 = make_tensor_2d_f32(m, k, &a_data);
        let b_f32 = make_tensor_2d_f32(k, n, &b_data);
        let a_fp8_mma = nsl_fp8_cast(a_f32, FP8_FORMAT_E4M3, scale as f64);
        let b_fp8_mma = nsl_fp8_cast(b_f32, FP8_FORMAT_E4M3, scale as f64);
        let out_mma = nsl_fp8_matmul(a_fp8_mma, b_fp8_mma, scale as f64, scale as f64);
        let v_mma = read_tensor_f32(out_mma);

        // Fallback path: scalar fp8_matmul_cpu via the reference (which uses
        // the same dequantize → sequential FMA pipeline as the runtime
        // fallback).
        let reference = Fp8ReferenceMatmul {
            m, n, k,
            format: Fp8Format::E4M3,
            scale,
        };
        let v_fallback = reference.compute_f32(&a_data, &b_data);

        assert_abs_err_le(&v_mma, &v_fallback, DISPATCH_ABS_TOL,
            &format!("MMA vs fallback {m}x{k}x{n}"));

        nsl_tensor_free(a_f32);
        nsl_tensor_free(b_f32);
        nsl_tensor_free(a_fp8_mma);
        nsl_tensor_free(b_fp8_mma);
        nsl_tensor_free(out_mma);
    }

    #[test]
    fn dispatch_16x16x16() { run_dispatch_at_shape(16, 16, 16); }

    #[test]
    fn dispatch_32x32x32() { run_dispatch_at_shape(32, 32, 32); }

    #[test]
    fn dispatch_64x64x64() { run_dispatch_at_shape(64, 64, 64); }

    #[test]
    fn dispatch_128x128x128() { run_dispatch_at_shape(128, 128, 128); }
}

#[cfg(not(feature = "cuda"))]
#[test]
#[ignore = "FP8 dispatcher tests require the cuda feature + sm_89+"]
fn cuda_feature_required() {
    eprintln!("compile with --features cuda to run dispatcher tests");
}
```

- [ ] **Step 2: Add a public `test_detect_sm_version` wrapper to the runtime**

The runtime has `pub(crate) fn detect_sm_version() -> u32` at `crates/nsl-runtime/src/cuda/mod.rs:133` (returns the combined SM version, e.g. 89 for Ada, 90 for Hopper). Expose it under the `test-hooks` feature so the dispatcher test can reach it.

Open `crates/nsl-runtime/src/cuda/mod.rs` and append at the bottom (after the `inner` module):

```rust
/// Test-only re-export: the current device's SM version (e.g. 89, 90).
/// Returns 0 when the cuda feature is off or no device is initialized.
#[cfg(feature = "test-hooks")]
#[no_mangle]
pub extern "C" fn test_detect_sm_version() -> u32 {
    inner::detect_sm_version()
}
```

If the runtime's lib.rs doesn't already re-export `pub use cuda::test_detect_sm_version`, add it next to the existing `pub use cuda::{...}` block at `crates/nsl-runtime/src/lib.rs:30`.

- [ ] **Step 3: Run on cuda-capable hardware**

Run: `cargo test -p nsl-runtime --test fp8_dispatcher --features "cuda test-hooks"`
Expected on sm_89+: 4 PASS.
Expected on <sm_89: 4 tests print "skipping … device < sm_89" but still PASS (early return).

If FAIL: this is the most informative failure of the suite. Divergence > 1e-5 between MMA and fallback at the same shape means the MMA kernel computes something different from sequential FMA (not just reorders). Common causes:
- Wrong tile shape in the MMA kernel (m16n8k32 expects specific layout).
- f16 intermediate accumulation in the MMA path that the fallback uses f32 for.
- Scale not flowing into the MMA kernel correctly.

Fix in the same commit; document the cause and the fix.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/tests/fp8_dispatcher.rs
git commit -m "test(fp8): dispatcher MMA == fallback within 1e-5

Compares nsl_fp8_matmul (MMA path on sm_89+) against the scalar
fallback at four square shapes. Feature-gated cuda + runtime
sm_89+ check. Divergence > 1e-5 = algorithmic bug, not accumulation
ordering."
```

---

## Task 7: CI integration verification

**Files:**
- Modify (if needed): `crates/nsl-runtime/Cargo.toml`

Cargo automatically discovers `tests/*.rs` files; no `[[test]]` entries should be required. This task verifies the assumption and adds explicit entries only if needed.

- [ ] **Step 1: Run the full integration suite without cuda**

Run: `cargo test -p nsl-runtime --tests --features test-hooks`
Expected: tests from `fp8_cast`, `fp8_scale`, `fp8_matmul_forward`, `fp8_matmul_backward` all PASS. `fp8_dispatcher` reports `cuda_feature_required` as ignored.

- [ ] **Step 2: Run the full integration suite with cuda (on cuda-capable hardware)**

Run: `cargo test -p nsl-runtime --tests --features "cuda test-hooks"`
Expected: all CPU tests PASS plus dispatcher tests PASS (or skip with reason on <sm_89 hardware).

- [ ] **Step 3: If cargo's auto-discovery missed any file, add explicit entries**

If `cargo test -p nsl-runtime --tests --features test-hooks` does not list one of the new files in its test output, add the explicit entry to `crates/nsl-runtime/Cargo.toml`:

```toml
[[test]]
name = "fp8_cast"
path = "tests/fp8_cast.rs"

[[test]]
name = "fp8_scale"
path = "tests/fp8_scale.rs"

[[test]]
name = "fp8_matmul_forward"
path = "tests/fp8_matmul_forward.rs"

[[test]]
name = "fp8_matmul_backward"
path = "tests/fp8_matmul_backward.rs"

[[test]]
name = "fp8_dispatcher"
path = "tests/fp8_dispatcher.rs"
```

- [ ] **Step 4: Commit (only if Cargo.toml needed updates)**

```bash
git add crates/nsl-runtime/Cargo.toml
git commit -m "build(fp8): explicit [[test]] entries for FP8 integration tests"
```

If no Cargo.toml change was required, skip this commit.

---

## Final verification

- [ ] **Run the full M35 correctness suite**

Run: `cargo test -p nsl-runtime fp8 --features "cuda test-hooks" 2>&1 | tail -30`

Expected output: a summary line for each of the 5 test binaries (`fp8_cast`, `fp8_scale`, `fp8_matmul_forward`, `fp8_matmul_backward`, `fp8_dispatcher`) reporting all tests passing. Approximately 22 tests total (3 cast + 6 scale + 5 forward + 4 backward + 4 dispatcher).

- [ ] **Confirm zero `#[ignore]` on correctness tests**

Run: `grep -rn "#\[ignore\]" crates/nsl-runtime/tests/fp8_*.rs`

Expected output: only the `cuda_feature_required` placeholder (which is the no-cuda fallback, not a skipped correctness test). Any other `#[ignore]` on a correctness test is a plan failure — it indicates an unfixed bug. Loop back to the relevant task and fix.
