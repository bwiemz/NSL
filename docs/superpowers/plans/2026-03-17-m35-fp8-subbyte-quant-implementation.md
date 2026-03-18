# M35: FP8 Compute & Sub-Byte Quantization (AWQ/GPTQ) — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FP8 Tensor Core compute with automatic scale management, AWQ 4-bit and GPTQ 4-bit/8-bit quantization with in-register dequantize-in-GEMM, making NSL competitive with production inference stacks.

**Architecture:** Three subsystems built bottom-up: (1) FP8 runtime with thread-local scale table, cast/matmul FFI, `@fp8_compute` decorator semantic + codegen; (2) AWQ runtime with packed 4-bit format, per-group scale/zero, dequant-GEMM FFI; (3) GPTQ runtime reusing AWQ packed format with Hessian-based quantization. Each adds PTX kernel stubs, codegen wiring, and E2E tests. Epilogue fusion extended with `MatmulKind` metadata last.

**Tech Stack:** Rust (runtime + codegen), Cranelift (codegen), PTX ISA 7.0+ (GPU kernels)

**Spec:** `docs/superpowers/specs/2026-03-15-m35-fp8-subbyte-quant-design.md`

---

## Scope Note

The spec covers 8 deliverables across 3 subsystems. This plan orders them so each produces independently testable code:

1. **Tasks 1-3**: FP8 runtime (scale table, cast, compute_scale — pure Rust, zero codegen dependency)
2. **Tasks 4-5**: FP8 semantic validation (`@fp8_compute` decorator, implicit conversion rules)
3. **Tasks 6-7**: FP8 codegen wiring (builtins, compiler fields, decorator extraction)
4. **Task 8**: FP8 E2E tests
5. **Tasks 9-11**: AWQ runtime (packed format, quantize, dequant-matmul)
6. **Tasks 12-13**: AWQ codegen wiring (QuantDtype extension, quant block codegen fields)
7. **Task 14**: AWQ E2E tests
8. **Tasks 15-16**: GPTQ runtime (Hessian-based quantization, reuses AWQ dequant kernel)
9. **Task 17**: GPTQ E2E tests
10. **Task 18**: Epilogue fusion `MatmulKind` extension
11. **Task 19**: Full verification + clippy

**Deferred to M35b:** FP8 Tensor Core PTX (`mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32` — requires SM90 hardware), AWQ/GPTQ dequant-in-GEMM PTX (requires PTX nibble extraction in MMA inner loop), `nsl quantize` CLI subcommand, calibration dataset management, FP8 dynamic calibration EMA.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/fp8.rs` | FP8 scale table, cast, compute_scale, matmul FFI | 250 |
| `crates/nsl-runtime/src/awq.rs` | AWQ packed format, quantize, dequant-matmul, free FFI | 300 |
| `crates/nsl-runtime/src/gptq.rs` | GPTQ quantize (Hessian-based), dequant-matmul, free FFI | 280 |
| `crates/nsl-semantic/src/fp8.rs` | `@fp8_compute` decorator validation | 80 |
| `crates/nsl-codegen/src/fp8.rs` | `@fp8_compute` decorator extraction, `Fp8ComputeInfo` | 80 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod fp8; pub mod awq; pub mod gptq;` |
| `crates/nsl-runtime/src/tensor.rs` | Add `DTYPE_FP8E4M3: u16 = 5`, `DTYPE_FP8E5M2: u16 = 6` constants |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod fp8;` |
| `crates/nsl-semantic/src/checker.rs` | Add `@fp8_compute` decorator validation dispatch, FP8 implicit conversion warning |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod fp8;` |
| `crates/nsl-codegen/src/builtins.rs` | Register 10 FFI functions (4 FP8 + 3 AWQ + 3 GPTQ) |
| `crates/nsl-codegen/src/compiler.rs` | Add `fp8_compute_fns: HashSet<String>`, `quant_configs: HashMap<String, QuantConfig>` |
| `crates/nsl-codegen/src/epilogue_fusion.rs` | Add `MatmulKind` enum, extend `EpilogueChain` |
| `crates/nsl-ast/src/block.rs` | Add `Awq4`, `Gptq4`, `Gptq8` to `QuantDtype` enum |
| `crates/nsl-parser/src/block.rs` | Parse `awq4`, `gptq4`, `gptq8` in quant block default dtype |
| `crates/nsl-cli/tests/e2e.rs` | Add M35 E2E tests |

---

## Phase 1: FP8 Runtime

### Task 1: FP8 Dtype Constants + Scale Table

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Create: `crates/nsl-runtime/src/fp8.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Add FP8 dtype constants to tensor.rs**

After the existing `DTYPE_F32` constant:

```rust
pub const DTYPE_FP16: u16 = 2;
pub const DTYPE_BF16: u16 = 3;
pub const DTYPE_INT8: u16 = 4;
pub const DTYPE_FP8E4M3: u16 = 5;
pub const DTYPE_FP8E5M2: u16 = 6;
```

NOTE: Check if DTYPE_FP16/BF16/INT8 are already defined. If so, only add the FP8 ones.

- [ ] **Step 2: Create `fp8.rs` with scale table and constants**

```rust
//! M35: FP8 scale management and cast operations.

use std::cell::RefCell;
use std::collections::HashMap;

/// Maximum representable value for E4M3 format.
pub const FP8E4M3_MAX: f32 = 448.0;
/// Maximum representable value for E5M2 format.
pub const FP8E5M2_MAX: f32 = 57344.0;

/// FP8 format identifier for FFI.
pub const FP8_FORMAT_E4M3: i64 = 0;
pub const FP8_FORMAT_E5M2: i64 = 1;

thread_local! {
    /// Per-tensor scale factors, keyed by tensor pointer (as i64).
    static FP8_SCALES: RefCell<HashMap<i64, f32>> = RefCell::new(HashMap::new());
}

/// Register scale for an FP8 tensor.
pub fn set_fp8_scale(tensor_ptr: i64, scale: f32) {
    FP8_SCALES.with(|s| s.borrow_mut().insert(tensor_ptr, scale));
}

/// Retrieve scale (returns 1.0 if unregistered — safe default).
pub fn get_fp8_scale(tensor_ptr: i64) -> f32 {
    FP8_SCALES.with(|s| *s.borrow().get(&tensor_ptr).unwrap_or(&1.0))
}

/// Remove scale entry (on tensor free).
pub fn remove_fp8_scale(tensor_ptr: i64) {
    FP8_SCALES.with(|s| s.borrow_mut().remove(&tensor_ptr));
}

/// Compute optimal scale factor: max(abs(tensor)) / fp8_max.
pub fn compute_scale(data: &[f64], fp8_format: i64) -> f64 {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX as f64,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX as f64,
        _ => FP8E4M3_MAX as f64,
    };
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / fp8_max
    }
}

/// Quantize a single f64 value to FP8 (simulated as clamped+scaled f64).
pub fn quantize_fp8(value: f64, scale: f64, fp8_format: i64) -> f64 {
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX as f64,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX as f64,
        _ => FP8E4M3_MAX as f64,
    };
    let scaled = value / scale;
    let clamped = scaled.clamp(-fp8_max, fp8_max);
    // Round to representable FP8 precision (simulate by rounding to nearest 0.125 for E4M3)
    let precision = match fp8_format {
        FP8_FORMAT_E4M3 => 0.125,
        FP8_FORMAT_E5M2 => 0.5,
        _ => 0.125,
    };
    (clamped / precision).round() * precision
}

/// Dequantize a simulated FP8 value back to f64.
pub fn dequantize_fp8(fp8_value: f64, scale: f64) -> f64 {
    fp8_value * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_table_set_get() {
        set_fp8_scale(42, 0.5);
        assert_eq!(get_fp8_scale(42), 0.5);
        assert_eq!(get_fp8_scale(999), 1.0); // default
        remove_fp8_scale(42);
        assert_eq!(get_fp8_scale(42), 1.0); // removed
    }

    #[test]
    fn test_compute_scale_e4m3() {
        let data = vec![100.0, -200.0, 50.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);
        // max_abs = 200, fp8_max = 448, scale = 200/448
        assert!((scale - 200.0 / 448.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_zero() {
        let data = vec![0.0, 0.0, 0.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);
        assert_eq!(scale, 1.0); // zero tensor -> scale 1.0
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let value = 1.5;
        let scale = 0.01; // small scale -> value fits in FP8
        let fp8 = quantize_fp8(value, scale, FP8_FORMAT_E4M3);
        let recovered = dequantize_fp8(fp8, scale);
        // E4M3 precision is 0.125, so after scale/round/unscale, error bounded
        assert!((recovered - value).abs() < scale * 0.125 + 1e-10);
    }

    #[test]
    fn test_clamping() {
        let value = 1000.0;
        let scale = 1.0; // scale=1 means value=1000 exceeds FP8E4M3 max (448)
        let fp8 = quantize_fp8(value, scale, FP8_FORMAT_E4M3);
        assert_eq!(fp8, 448.0); // clamped to max
    }
}
```

- [ ] **Step 3: Add `pub mod fp8;` to runtime lib.rs**

- [ ] **Step 4: Verify compilation, run tests, commit**

```bash
cargo test -p nsl-runtime fp8 -- --nocapture
git commit -m "feat(m35): add FP8 scale table, cast helpers, and dtype constants"
```

---

### Task 2: FP8 Cast FFI

**Files:**
- Modify: `crates/nsl-runtime/src/fp8.rs`

- [ ] **Step 1: Write test for FP8 cast FFI**

```rust
#[test]
fn test_fp8_cast_ffi_auto_scale() {
    // Create a test tensor-like f64 array
    let data = vec![1.0f64, 2.0, -3.0, 4.0];
    let scale = compute_scale(&data, FP8_FORMAT_E4M3);

    // Quantize each element
    let quantized: Vec<f64> = data.iter()
        .map(|&v| quantize_fp8(v, scale, FP8_FORMAT_E4M3))
        .collect();

    // Dequantize and check error
    for (orig, &quant) in data.iter().zip(quantized.iter()) {
        let recovered = dequantize_fp8(quant, scale);
        let rel_error = if orig.abs() > 1e-10 { (recovered - orig).abs() / orig.abs() } else { 0.0 };
        assert!(rel_error < 0.01, "FP8 E4M3 relative error {} too high for value {}", rel_error, orig);
    }
}
```

- [ ] **Step 2: Implement FFI functions**

```rust
use crate::tensor::NslTensor;
use std::ffi::c_void;

/// Cast a tensor to FP8 with given scale. If scale=0.0, auto-compute.
/// Returns a new tensor pointer (dtype remains f64 on CPU, values are FP8-simulated).
#[no_mangle]
pub extern "C" fn nsl_fp8_cast(tensor_ptr: i64, target_dtype: i64, scale: f64) -> i64 {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let len = t.len as usize;
    let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };

    let actual_scale = if scale == 0.0 {
        compute_scale(data, target_dtype)
    } else {
        scale
    };

    // Allocate new tensor with FP8-quantized values (stored as f64 on CPU)
    let mut result = Vec::with_capacity(len);
    for &v in data {
        result.push(dequantize_fp8(quantize_fp8(v, actual_scale, target_dtype), actual_scale));
    }

    // Create output tensor
    let out_data = result.as_mut_ptr();
    std::mem::forget(result);

    let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
    let out = crate::tensor::nsl_tensor_create_from_data(
        out_data as *mut c_void,
        shape.as_ptr() as *mut i64,
        t.ndim,
    );

    // Register scale for this tensor
    set_fp8_scale(out, actual_scale as f32);

    out
}

/// Compute optimal scale factor for FP8 conversion.
#[no_mangle]
pub extern "C" fn nsl_fp8_compute_scale(tensor_ptr: i64, fp8_dtype: i64) -> f64 {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let len = t.len as usize;
    let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
    compute_scale(data, fp8_dtype)
}
```

NOTE: `nsl_tensor_create_from_data` may not exist with that exact signature. Check `tensor.rs` for the actual tensor creation helper. Adapt accordingly — the pattern is to allocate shape/strides, fill the NslTensor struct, and return it as i64.

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime fp8 -- --nocapture
git commit -m "feat(m35): add FP8 cast and compute_scale FFI functions"
```

---

### Task 3: FP8 Matmul FFI (CPU Fallback)

**Files:**
- Modify: `crates/nsl-runtime/src/fp8.rs`

- [ ] **Step 1: Write test**

```rust
#[test]
fn test_fp8_matmul_cpu() {
    // 2x2 matmul: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let scale_a = compute_scale(&a, FP8_FORMAT_E4M3);
    let scale_b = compute_scale(&b, FP8_FORMAT_E4M3);

    // Quantize to FP8 and back (simulate what cast does)
    let a_fp8: Vec<f64> = a.iter().map(|&v| dequantize_fp8(quantize_fp8(v, scale_a, FP8_FORMAT_E4M3), scale_a)).collect();
    let b_fp8: Vec<f64> = b.iter().map(|&v| dequantize_fp8(quantize_fp8(v, scale_b, FP8_FORMAT_E4M3), scale_b)).collect();

    // Manual matmul
    let result = fp8_matmul_cpu(&a_fp8, &b_fp8, 2, 2, 2);

    // Check within FP8 tolerance
    assert!((result[0] - 19.0).abs() < 1.0); // ~5% tolerance for FP8
    assert!((result[1] - 22.0).abs() < 1.0);
    assert!((result[2] - 43.0).abs() < 2.0);
    assert!((result[3] - 50.0).abs() < 2.0);
}
```

- [ ] **Step 2: Implement CPU matmul path**

```rust
/// CPU fallback for FP8 matmul: A[M,K] @ B[K,N] -> C[M,N]
/// Accumulates in f64 (simulating f32 accumulation on GPU).
pub fn fp8_matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// FP8 matmul FFI: both inputs are FP8-cast tensors. Output is FP16-equivalent (f64 on CPU).
/// scale_a and scale_b are applied post-accumulation.
#[no_mangle]
pub extern "C" fn nsl_fp8_matmul(
    a_ptr: i64,
    b_ptr: i64,
    _scale_a: f64,
    _scale_b: f64,
) -> i64 {
    // On CPU, tensors are already dequantized f64. Just do standard matmul.
    // Scale application happens at cast time on CPU (GPU path applies post-MMA).
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };

    // Delegate to standard tensor matmul
    crate::tensor::nsl_tensor_matmul(a_ptr, b_ptr)
}

/// Update calibration running max (EMA). Returns updated running_max as f64.
#[no_mangle]
pub extern "C" fn nsl_fp8_update_calibration(
    _tensor_ptr: i64,
    _running_max_ptr: i64,
    _momentum: f64,
) -> f64 {
    // Stub for dynamic calibration — deferred to M35b
    0.0
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime fp8 -- --nocapture
git commit -m "feat(m35): add FP8 matmul CPU fallback and calibration stub"
```

---

## Phase 2: FP8 Semantic + Codegen

### Task 4: `@fp8_compute` Semantic Validation

**Files:**
- Create: `crates/nsl-semantic/src/fp8.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Create semantic validator** (same pattern as `moe.rs`, `context_parallel.rs`)

```rust
use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @fp8_compute decorator arguments.
/// Returns calibrate flag (default false).
pub fn validate_fp8_compute_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    let mut calibrate = false;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "calibrate" => {
                        if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                            calibrate = *b;
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@fp8_compute: calibrate must be a boolean".to_string(),
                                )
                                .with_label(arg.span, "expected true or false"),
                            );
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@fp8_compute: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    calibrate
}
```

- [ ] **Step 2: Add `pub mod fp8;` to semantic lib.rs**

- [ ] **Step 3: Wire into checker.rs** (after the `@context_parallel` block)

```rust
// M35: @fp8_compute decorator validation
if dname == "fp8_compute" {
    let resolve = |s: nsl_ast::Symbol| -> String {
        self.interner
            .resolve(s.0)
            .unwrap_or("")
            .to_string()
    };
    crate::fp8::validate_fp8_compute_decorator(
        deco,
        &resolve,
        &mut self.diagnostics,
    );
}
```

- [ ] **Step 4: Verify compilation, commit**

```bash
cargo check -p nsl-semantic
git commit -m "feat(m35): add @fp8_compute semantic validation"
```

---

### Task 5: FP8 Implicit Conversion Warning (Optional Enhancement)

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

This task adds a compile-time warning when FP8 narrowing conversion is attempted outside `.to()`. Since the type system already has `Fp8E4m3`/`Fp8E5m2` as types, this integrates with existing widening checks.

- [ ] **Step 1: Locate the implicit conversion/widening check in checker.rs**

Search for `widening_rank` or the binary op type resolution to find where implicit casts are checked.

- [ ] **Step 2: Add FP8 narrowing guard**

At the point where binary op or assignment type compatibility is checked, add:

```rust
// M35: Warn on implicit FP8 narrowing (fp16 -> fp8 requires explicit .to())
if target_dtype.is_fp8() && !source_dtype.is_fp8() {
    self.diagnostics.push(
        Diagnostic::warning(
            "implicit narrowing to FP8 — use .to(fp8e4m3) or .to(fp8e5m2) for explicit cast".to_string()
        )
        .with_label(span, "implicit fp8 narrowing"),
    );
}
```

NOTE: This may require adding `is_fp8()` helper to `DType` if it doesn't exist:
```rust
impl DType {
    pub fn is_fp8(&self) -> bool {
        matches!(self, DType::Fp8E4m3 | DType::Fp8E5m2)
    }
}
```

- [ ] **Step 3: Verify, commit**

```bash
cargo check -p nsl-semantic
git commit -m "feat(m35): add FP8 implicit narrowing warning"
```

---

### Task 6: FP8 Codegen Module + Builtins

**Files:**
- Create: `crates/nsl-codegen/src/fp8.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 1: Create codegen extraction module**

```rust
//! M35: FP8 compute codegen — @fp8_compute extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

#[derive(Debug, Clone)]
pub struct Fp8ComputeInfo {
    pub calibrate: bool,
}

pub fn extract_fp8_compute_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<Fp8ComputeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "fp8_compute" {
            let mut calibrate = false;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        if resolve_sym(name_sym) == "calibrate" {
                            if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                calibrate = *b;
                            }
                        }
                    }
                }
            }
            return Some(Fp8ComputeInfo { calibrate });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_fp8_compute_decorator(&[], &|_| "").is_none());
    }
}
```

- [ ] **Step 2: Add `pub mod fp8;` to codegen lib.rs**

- [ ] **Step 3: Register FP8 FFI functions in builtins.rs**

After the M34 context parallelism block:

```rust
    // --- M35: FP8 compute ---
    ("nsl_fp8_cast", &[types::I64, types::I64, types::F64], Some(types::I64)),
    ("nsl_fp8_matmul", &[types::I64, types::I64, types::F64, types::F64], Some(types::I64)),
    ("nsl_fp8_compute_scale", &[types::I64, types::I64], Some(types::F64)),
    ("nsl_fp8_update_calibration", &[types::I64, types::I64, types::F64], Some(types::F64)),
```

- [ ] **Step 4: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m35): add FP8 codegen module and builtin registrations"
```

---

### Task 7: FP8 Compiler Fields + Decorator Wiring

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Add compiler fields**

After `cp_ring_size`:

```rust
    /// M35: Functions with @fp8_compute decorator
    pub fp8_compute_fns: HashSet<String>,
```

Initialize as `fp8_compute_fns: HashSet::new()` in `Compiler::new()`.

NOTE: Also need `use std::collections::HashSet;` if not already imported (it's already used for `no_grad_fns`).

- [ ] **Step 2: Wire `@fp8_compute` extraction in model/fn compilation**

Find where `@context_parallel` extraction happens and add after it:

```rust
// M35: @fp8_compute decorator extraction
if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "fp8_compute" {
    if let Some(_info) = crate::fp8::extract_fp8_compute_decorator(
        std::slice::from_ref(deco),
        &|sym| self.resolve_sym(sym),
    ) {
        // Track this function as FP8 compute
        // fn_name will be set during function compilation
    }
}
```

NOTE: The exact wiring depends on where function decorators are processed. Check the pattern for `no_grad_fns` — that's the closest analogue (function-level decorator → HashSet).

- [ ] **Step 3: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m35): wire @fp8_compute compiler fields and decorator extraction"
```

---

### Task 8: FP8 E2E Tests

**Files:**
- Create: `examples/m35_fp8_basic.nsl`
- Create: `examples/m35_fp8_validation_error.nsl`
- Create: `tests/expected/m35_fp8_basic.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create basic FP8 example** (model with @fp8_compute decorator compiles)

```nsl
# M35: FP8 compute — decorator validation test

model FP8Model:
    @fp8_compute
    weight: int = 0

    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = FP8Model()
let x = ones([2, 4])
let y = m.forward(x)
print(y)
```

- [ ] **Step 2: Create validation error example** (unknown @fp8_compute arg)

```nsl
# M35: @fp8_compute validation error — unknown argument

model Bad:
    @fp8_compute(unknown_arg=42)
    weight: int = 0

    fn forward(self, x: Tensor) -> Tensor:
        return x
```

- [ ] **Step 3: Add E2E tests to e2e.rs**

```rust
// ---------------------------------------------------------------------------
// M35: FP8 Compute & Sub-Byte Quantization
// ---------------------------------------------------------------------------

#[test]
fn e2e_m35_fp8_basic() {
    assert_output_matches("m35_fp8_basic");
}

#[test]
fn e2e_m35_fp8_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m35_fp8_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m35_fp8_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("fp8_compute") || stderr.contains("unknown argument"),
        "Expected fp8_compute validation error in stderr, got: {}",
        stderr
    );
}
```

- [ ] **Step 4: Run E2E tests, commit**

```bash
cargo test -p nsl-cli e2e_m35 -- --nocapture
git commit -m "test(m35): add FP8 E2E tests for basic compilation and validation error"
```

---

## Phase 3: AWQ Runtime

### Task 9: AWQ Packed Format + Types

**Files:**
- Create: `crates/nsl-runtime/src/awq.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Create AWQ module with packed format and tests**

```rust
//! M35: AWQ (Activation-Aware Weight Quantization) 4-bit runtime.

use std::ffi::c_void;

/// AWQ packed weight representation.
/// Two 4-bit values packed per byte, low nibble first.
#[repr(C)]
pub struct AwqPackedWeight {
    /// Packed 4-bit data [K/2 * N bytes for K*N weights]
    pub data: *mut u8,
    /// Per-group FP16 scale factors (stored as f64 on CPU)
    pub scales: *mut f64,
    /// Per-group zero points (stored as f64 on CPU)
    pub zeros: *mut f64,
    /// Original weight dimensions [K, N]
    pub k: i64,
    pub n: i64,
    /// Group size (typically 128)
    pub group_size: i64,
    /// Number of groups per column: ceil(K / group_size)
    pub num_groups: i64,
    pub refcount: i64,
}

/// Pack two 4-bit values into one byte (low nibble first).
pub fn pack_int4(low: u8, high: u8) -> u8 {
    (low & 0x0F) | ((high & 0x0F) << 4)
}

/// Unpack a byte into two 4-bit values (low, high).
pub fn unpack_int4(packed: u8) -> (u8, u8) {
    (packed & 0x0F, (packed >> 4) & 0x0F)
}

/// Compute per-group scale and zero for asymmetric quantization.
/// Returns (scale, zero_point) where: quantized = round((value - zero) / scale)
pub fn compute_group_params(group: &[f64]) -> (f64, f64) {
    let min_val = group.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = group.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return (1.0, min_val);
    }

    let scale = (max_val - min_val) / 15.0; // 4-bit: 0..15
    let zero = min_val;
    (scale, zero)
}

/// Quantize a single value to 4-bit given scale and zero.
pub fn quantize_int4(value: f64, scale: f64, zero: f64) -> u8 {
    let q = ((value - zero) / scale).round() as i32;
    q.clamp(0, 15) as u8
}

/// Dequantize a 4-bit value to f64.
pub fn dequantize_int4(quantized: u8, scale: f64, zero: f64) -> f64 {
    (quantized as f64) * scale + zero
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_int4() {
        let packed = pack_int4(5, 12);
        let (low, high) = unpack_int4(packed);
        assert_eq!(low, 5);
        assert_eq!(high, 12);
    }

    #[test]
    fn test_pack_unpack_boundary() {
        let packed = pack_int4(0, 15);
        let (low, high) = unpack_int4(packed);
        assert_eq!(low, 0);
        assert_eq!(high, 15);
    }

    #[test]
    fn test_group_params() {
        let group = vec![0.0, 1.0, 2.0, 3.0];
        let (scale, zero) = compute_group_params(&group);
        assert!((scale - 0.2).abs() < 1e-10); // (3-0)/15
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn test_quantize_dequantize_int4() {
        let value = 1.5;
        let (scale, zero) = (0.2, 0.0);
        let q = quantize_int4(value, scale, zero);
        let recovered = dequantize_int4(q, scale, zero);
        assert!((recovered - value).abs() < scale); // within one quant step
    }

    #[test]
    fn test_quantize_clamp() {
        let q = quantize_int4(100.0, 1.0, 0.0); // way above max
        assert_eq!(q, 15);
    }
}
```

- [ ] **Step 2: Add `pub mod awq;` to runtime lib.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime awq -- --nocapture
git commit -m "feat(m35): add AWQ packed format types and int4 pack/unpack"
```

---

### Task 10: AWQ Quantize + Dequant-Matmul

**Files:**
- Modify: `crates/nsl-runtime/src/awq.rs`

- [ ] **Step 1: Write quantize test**

```rust
#[test]
fn test_awq_quantize_small_matrix() {
    // 4x4 weight matrix, group_size=4
    let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
    let packed = awq_quantize_cpu(&weights, 4, 4, 4);

    assert_eq!(packed.k, 4);
    assert_eq!(packed.n, 4);
    assert_eq!(packed.group_size, 4);
    assert_eq!(packed.num_groups, 1); // 4/4 = 1 group per column

    // Dequantize and check error
    let recovered = awq_dequantize_cpu(&packed);
    for (i, (&orig, &rec)) in weights.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < 0.15, // within ~1 quant step for this range
            "AWQ error too high at index {}: orig={}, recovered={}", i, orig, rec
        );
    }

    // Cleanup
    unsafe { awq_free_packed(&packed); }
}
```

- [ ] **Step 2: Implement quantize, dequantize, free**

```rust
/// Quantize a weight matrix [K, N] to AWQ 4-bit packed format.
pub fn awq_quantize_cpu(weights: &[f64], k: usize, n: usize, group_size: usize) -> AwqPackedWeight {
    let num_groups = (k + group_size - 1) / group_size;
    let packed_bytes = (k * n + 1) / 2; // 2 values per byte

    let data = vec![0u8; packed_bytes];
    let scales = vec![0.0f64; num_groups * n];
    let zeros = vec![0.0f64; num_groups * n];

    let mut data = data.into_boxed_slice();
    let mut scales = scales.into_boxed_slice();
    let mut zeros = zeros.into_boxed_slice();

    // Quantize column by column, group by group
    for col in 0..n {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(k);

            // Extract group values
            let group: Vec<f64> = (start..end).map(|row| weights[row * n + col]).collect();
            let (scale, zero) = compute_group_params(&group);

            scales[g * n + col] = scale;
            zeros[g * n + col] = zero;

            // Quantize and pack
            for (i, row) in (start..end).enumerate() {
                let q = quantize_int4(weights[row * n + col], scale, zero);
                let flat_idx = row * n + col;
                let byte_idx = flat_idx / 2;
                if flat_idx % 2 == 0 {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | (q & 0x0F);
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
                }
            }
        }
    }

    let data_ptr = Box::into_raw(data) as *mut u8;
    let scales_ptr = Box::into_raw(scales) as *mut f64;
    let zeros_ptr = Box::into_raw(zeros) as *mut f64;

    AwqPackedWeight {
        data: data_ptr,
        scales: scales_ptr,
        zeros: zeros_ptr,
        k: k as i64,
        n: n as i64,
        group_size: group_size as i64,
        num_groups: num_groups as i64,
        refcount: 1,
    }
}

/// Dequantize AWQ packed weights back to f64 for verification.
pub fn awq_dequantize_cpu(packed: &AwqPackedWeight) -> Vec<f64> {
    let k = packed.k as usize;
    let n = packed.n as usize;
    let group_size = packed.group_size as usize;
    let mut result = vec![0.0f64; k * n];

    let data = unsafe { std::slice::from_raw_parts(packed.data, (k * n + 1) / 2) };
    let scales = unsafe { std::slice::from_raw_parts(packed.scales, packed.num_groups as usize * n) };
    let zeros = unsafe { std::slice::from_raw_parts(packed.zeros, packed.num_groups as usize * n) };

    for col in 0..n {
        for row in 0..k {
            let g = row / group_size;
            let scale = scales[g * n + col];
            let zero = zeros[g * n + col];

            let flat_idx = row * n + col;
            let byte_idx = flat_idx / 2;
            let q = if flat_idx % 2 == 0 {
                data[byte_idx] & 0x0F
            } else {
                (data[byte_idx] >> 4) & 0x0F
            };

            result[flat_idx] = dequantize_int4(q, scale, zero);
        }
    }
    result
}

/// Free an AWQ packed weight.
pub unsafe fn awq_free_packed(packed: &AwqPackedWeight) {
    let k = packed.k as usize;
    let n = packed.n as usize;
    let num_groups = packed.num_groups as usize;
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(packed.data, (k * n + 1) / 2));
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(packed.scales, num_groups * n));
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(packed.zeros, num_groups * n));
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime awq -- --nocapture
git commit -m "feat(m35): implement AWQ quantize, dequantize, and free"
```

---

### Task 11: AWQ Matmul + FFI

**Files:**
- Modify: `crates/nsl-runtime/src/awq.rs`

- [ ] **Step 1: Write matmul test**

```rust
#[test]
fn test_awq_matmul_cpu() {
    // Input [2,4], Weight [4,3], Output [2,3]
    let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weights = vec![
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2,
    ];

    // Standard matmul reference
    let reference = matmul_cpu(&input, &weights, 2, 4, 3);

    // AWQ path: quantize weights, then matmul
    let packed = awq_quantize_cpu(&weights, 4, 3, 4);
    let awq_result = awq_matmul_cpu(&input, &packed, 2);

    // Check within 10% tolerance (4-bit quantization)
    for (i, (&r, &a)) in reference.iter().zip(awq_result.iter()).enumerate() {
        let tol = r.abs() * 0.1 + 0.1; // relative + absolute tolerance
        assert!(
            (r - a).abs() < tol,
            "AWQ matmul error at {}: ref={}, awq={}", i, r, a
        );
    }

    unsafe { awq_free_packed(&packed); }
}
```

- [ ] **Step 2: Implement AWQ matmul CPU path**

```rust
/// Simple CPU matmul: A[M,K] @ B[K,N] -> C[M,N]
fn matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// AWQ dequantize-matmul: input[M,K] @ packed_weight[K,N] -> output[M,N]
/// On CPU, dequantizes on-the-fly during matmul (simulating in-register dequant).
pub fn awq_matmul_cpu(input: &[f64], packed: &AwqPackedWeight, m: usize) -> Vec<f64> {
    let k = packed.k as usize;
    let n = packed.n as usize;
    let group_size = packed.group_size as usize;

    let data = unsafe { std::slice::from_raw_parts(packed.data, (k * n + 1) / 2) };
    let scales = unsafe { std::slice::from_raw_parts(packed.scales, packed.num_groups as usize * n) };
    let zeros = unsafe { std::slice::from_raw_parts(packed.zeros, packed.num_groups as usize * n) };

    let mut output = vec![0.0f64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                let g = p / group_size;
                let scale = scales[g * n + j];
                let zero = zeros[g * n + j];

                let flat_idx = p * n + j;
                let byte_idx = flat_idx / 2;
                let q = if flat_idx % 2 == 0 {
                    data[byte_idx] & 0x0F
                } else {
                    (data[byte_idx] >> 4) & 0x0F
                };

                let w = dequantize_int4(q, scale, zero);
                acc += input[i * k + p] * w;
            }
            output[i * n + j] = acc;
        }
    }
    output
}
```

- [ ] **Step 3: Add FFI wrappers**

```rust
/// Quantize FP16/f64 weight to AWQ4 packed format.
#[no_mangle]
pub extern "C" fn nsl_awq_quantize(
    weight_ptr: i64,
    group_size: i64,
    _calibration_ptr: i64,
) -> i64 {
    let t = unsafe { &*(weight_ptr as *const crate::tensor::NslTensor) };
    let len = t.len as usize;
    let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };

    // Assume 2D weight [K, N]
    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;

    let packed = awq_quantize_cpu(data, k, n, group_size as usize);
    Box::into_raw(Box::new(packed)) as i64
}

/// AWQ dequantize-in-GEMM matmul.
#[no_mangle]
pub extern "C" fn nsl_awq_matmul(
    input_ptr: i64,
    packed_ptr: i64,
    _group_size: i64,
) -> i64 {
    let input_t = unsafe { &*(input_ptr as *const crate::tensor::NslTensor) };
    let packed = unsafe { &*(packed_ptr as *const AwqPackedWeight) };

    let m = unsafe { *input_t.shape } as usize;
    let k = unsafe { *input_t.shape.add(1) } as usize;
    let input = unsafe { std::slice::from_raw_parts(input_t.data as *const f64, input_t.len as usize) };

    let result = awq_matmul_cpu(input, packed, m);

    // Create output tensor [M, N]
    let n = packed.n as usize;
    let shape = vec![m as i64, n as i64];
    let out_data = result.as_ptr();
    std::mem::forget(result);

    crate::tensor::nsl_tensor_create_from_data(
        out_data as *mut c_void,
        shape.as_ptr() as *mut i64,
        2,
    )
}

/// Free an AWQ packed weight.
#[no_mangle]
pub extern "C" fn nsl_awq_free(packed_ptr: i64) {
    if packed_ptr == 0 { return; }
    let packed = unsafe { Box::from_raw(packed_ptr as *mut AwqPackedWeight) };
    unsafe { awq_free_packed(&packed); }
}
```

NOTE: `nsl_tensor_create_from_data` may not exist. Check `tensor.rs` for the actual creation function and adapt. The key pattern is: allocate shape/strides, build NslTensor, return as i64.

- [ ] **Step 4: Run tests, commit**

```bash
cargo test -p nsl-runtime awq -- --nocapture
git commit -m "feat(m35): implement AWQ dequant-matmul CPU path and FFI"
```

---

### Task 12: Extend QuantDtype AST + Parser

**Files:**
- Modify: `crates/nsl-ast/src/block.rs`
- Modify: `crates/nsl-parser/src/block.rs`

- [ ] **Step 1: Add AWQ/GPTQ variants to QuantDtype enum**

```rust
#[derive(Debug, Clone, Serialize)]
pub enum QuantDtype {
    Int8,
    Int4,
    Awq4,
    Gptq4,
    Gptq8,
}
```

- [ ] **Step 2: Add parsing for new dtypes in parse_quant_block_stmt**

Find where `"int4"` and `"int8"` are matched and add:

```rust
"awq4" => QuantDtype::Awq4,
"gptq4" => QuantDtype::Gptq4,
"gptq8" => QuantDtype::Gptq8,
```

- [ ] **Step 3: Verify compilation, commit**

```bash
cargo check --workspace
git commit -m "feat(m35): extend QuantDtype with Awq4, Gptq4, Gptq8 variants"
```

---

### Task 13: AWQ Codegen Builtins + Compiler Fields

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Register AWQ FFI functions in builtins.rs**

```rust
    // --- M35: AWQ 4-bit quantization ---
    ("nsl_awq_quantize", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_awq_matmul", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_awq_free", &[types::I64], None),
```

- [ ] **Step 2: Add quant_configs to compiler**

```rust
    /// M35: Model quantization configs — "ModelName" → QuantConfig
    pub quant_configs: HashMap<String, QuantConfig>,
```

With supporting struct (in compiler.rs or a separate file):

```rust
/// Quantization configuration for a model.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub dtype: String,     // "awq4", "gptq4", "gptq8"
    pub group_size: i64,
}
```

Initialize as `quant_configs: HashMap::new()` in `Compiler::new()`.

- [ ] **Step 3: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m35): register AWQ builtins and add quant_configs compiler field"
```

---

### Task 14: AWQ E2E Tests

**Files:**
- Create: `examples/m35_awq_basic.nsl`
- Create: `tests/expected/m35_awq_basic.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create AWQ example** (model with quant block, compiles and runs forward)

```nsl
# M35: AWQ basic — model forward pass test (no actual quantization yet)

model QuantModel:
    weight: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

let m = QuantModel()
let x = ones([2, 4])
let y = m.forward(x)
print(y)
```

NOTE: This tests that the model compiles. Actual AWQ quantization wiring (quant block → codegen) is deferred until the `quant` block codegen path is fully implemented in a future task.

- [ ] **Step 2: Add E2E test, verify, commit**

```bash
cargo test -p nsl-cli e2e_m35 -- --nocapture
git commit -m "test(m35): add AWQ E2E test for basic model compilation"
```

---

## Phase 4: GPTQ Runtime

### Task 15: GPTQ Module + Quantize

**Files:**
- Create: `crates/nsl-runtime/src/gptq.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Create GPTQ module with Hessian-based quantization**

The GPTQ packed format is identical to AWQ — the difference is in how scales/zeros are computed.

```rust
//! M35: GPTQ (Generalized Post-Training Quantization) runtime.
//!
//! Reuses AWQ packed format. The difference is quantization:
//! GPTQ uses Hessian information to minimize reconstruction error.

use crate::awq::{AwqPackedWeight, pack_int4, unpack_int4, dequantize_int4, compute_group_params, quantize_int4};

/// GPTQ quantize: uses Hessian diagonal to weight error compensation.
/// For now, implements simple RTN (Round-To-Nearest) as baseline.
/// Full OBQ (Optimal Brain Quantizer) is deferred to M35b.
pub fn gptq_quantize_cpu(
    weights: &[f64],
    _hessian_diag: &[f64],
    k: usize,
    n: usize,
    group_size: usize,
    _bits: usize,
) -> AwqPackedWeight {
    // Baseline: RTN quantization (same as AWQ without salient channel detection).
    // Full GPTQ with Hessian error compensation is M35b.
    crate::awq::awq_quantize_cpu(weights, k, n, group_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::awq::awq_dequantize_cpu;

    #[test]
    fn test_gptq_quantize_rtn_baseline() {
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let hessian = vec![1.0f64; 4]; // identity Hessian
        let packed = gptq_quantize_cpu(&weights, &hessian, 4, 4, 4, 4);

        let recovered = awq_dequantize_cpu(&packed);
        for (i, (&orig, &rec)) in weights.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.15,
                "GPTQ RTN error too high at {}: orig={}, recovered={}", i, orig, rec
            );
        }

        unsafe { crate::awq::awq_free_packed(&packed); }
    }
}
```

- [ ] **Step 2: Add `pub mod gptq;` to runtime lib.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime gptq -- --nocapture
git commit -m "feat(m35): add GPTQ module with RTN baseline quantization"
```

---

### Task 16: GPTQ FFI + Builtins

**Files:**
- Modify: `crates/nsl-runtime/src/gptq.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 1: Add GPTQ FFI functions**

```rust
use std::ffi::c_void;

/// GPTQ quantize: weight + Hessian → packed weight.
#[no_mangle]
pub extern "C" fn nsl_gptq_quantize(
    weight_ptr: i64,
    hessian_ptr: i64,
    group_size: i64,
    bits: i64,
) -> i64 {
    let t = unsafe { &*(weight_ptr as *const crate::tensor::NslTensor) };
    let len = t.len as usize;
    let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };

    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;

    // Hessian diagonal (or identity if null)
    let hessian = if hessian_ptr == 0 {
        vec![1.0f64; k]
    } else {
        let h = unsafe { &*(hessian_ptr as *const crate::tensor::NslTensor) };
        unsafe { std::slice::from_raw_parts(h.data as *const f64, h.len as usize) }.to_vec()
    };

    let packed = gptq_quantize_cpu(data, &hessian, k, n, group_size as usize, bits as usize);
    Box::into_raw(Box::new(packed)) as i64
}

/// GPTQ dequant-matmul (same kernel as AWQ).
#[no_mangle]
pub extern "C" fn nsl_gptq_matmul(
    input_ptr: i64,
    packed_ptr: i64,
    _group_size: i64,
    _bits: i64,
) -> i64 {
    // Reuse AWQ matmul — packed format is identical
    crate::awq::nsl_awq_matmul(input_ptr, packed_ptr, _group_size)
}

/// Free a GPTQ packed weight.
#[no_mangle]
pub extern "C" fn nsl_gptq_free(packed_ptr: i64) {
    crate::awq::nsl_awq_free(packed_ptr);
}
```

- [ ] **Step 2: Register GPTQ builtins**

```rust
    // --- M35: GPTQ quantization ---
    ("nsl_gptq_quantize", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_gptq_matmul", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_gptq_free", &[types::I64], None),
```

- [ ] **Step 3: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m35): add GPTQ FFI and register builtins"
```

---

### Task 17: GPTQ E2E Test

Same pattern as Task 14 — create a model that compiles with GPTQ-style quant block.

- [ ] **Step 1: Add E2E test to e2e.rs, commit**

---

## Phase 5: Epilogue Fusion + Final Verification

### Task 18: Epilogue Fusion MatmulKind Extension

**Files:**
- Modify: `crates/nsl-codegen/src/epilogue_fusion.rs`

- [ ] **Step 1: Add `MatmulKind` enum and extend `EpilogueChain`**

```rust
/// Matmul variant for epilogue fusion dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum MatmulKind {
    Standard,
    Fp8 { a_scale: f32, b_scale: f32 },
    Awq4 { group_size: usize },
    Gptq { group_size: usize, bits: usize },
}
```

Add `pub matmul_kind: MatmulKind` field to `EpilogueChain`, default to `MatmulKind::Standard` in existing construction.

- [ ] **Step 2: Add tests for new variants**

```rust
#[test]
fn test_matmul_kind_default() {
    assert_eq!(MatmulKind::Standard, MatmulKind::Standard);
    assert_ne!(MatmulKind::Standard, MatmulKind::Fp8 { a_scale: 1.0, b_scale: 1.0 });
}

#[test]
fn test_matmul_kind_fp8() {
    let kind = MatmulKind::Fp8 { a_scale: 0.5, b_scale: 0.25 };
    if let MatmulKind::Fp8 { a_scale, b_scale } = kind {
        assert_eq!(a_scale, 0.5);
        assert_eq!(b_scale, 0.25);
    } else {
        panic!("expected Fp8");
    }
}
```

- [ ] **Step 3: Verify all existing epilogue tests still pass, commit**

```bash
cargo test -p nsl-codegen epilogue -- --nocapture
git commit -m "feat(m35): add MatmulKind to epilogue fusion for FP8/AWQ/GPTQ dispatch"
```

---

### Task 19: Full Test Suite + Clippy

- [ ] **Step 1: Run `cargo test --workspace --lib`** — all tests pass
- [ ] **Step 2: Run `cargo clippy --workspace -- -D warnings`** — clean
- [ ] **Step 3: Run M35 E2E tests** — pass
- [ ] **Step 4: Fix any issues, commit**

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | FP8 scale table + dtype constants | 5 unit |
| 2 | FP8 cast FFI | 1 unit |
| 3 | FP8 matmul CPU path | 1 unit |
| 4 | @fp8_compute semantic validation | compile check |
| 5 | FP8 implicit conversion warning | compile check |
| 6 | FP8 codegen module + builtins | 1 unit |
| 7 | FP8 compiler fields + wiring | compile check |
| 8 | FP8 E2E tests | 2 E2E |
| 9 | AWQ packed format + types | 5 unit |
| 10 | AWQ quantize + dequant-matmul | 2 unit |
| 11 | AWQ matmul FFI | compile check |
| 12 | QuantDtype AST extension | compile check |
| 13 | AWQ codegen builtins + compiler fields | compile check |
| 14 | AWQ E2E test | 1 E2E |
| 15 | GPTQ module + quantize | 1 unit |
| 16 | GPTQ FFI + builtins | compile check |
| 17 | GPTQ E2E test | 1 E2E |
| 18 | Epilogue fusion MatmulKind | 2 unit |
| 19 | Full verification | all tests |

**Total: 19 tasks, ~19 unit tests + 4 E2E tests**

### Deferred to M35b
- FP8 Tensor Core PTX (`mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32` — requires SM90 hardware)
- AWQ/GPTQ dequant-in-GEMM PTX (in-register nibble extraction in MMA inner loop)
- FP8 dynamic calibration (EMA running_max, calibration_interval)
- `nsl quantize` CLI subcommand for offline GPTQ quantization
- Calibration dataset management
- Full OBQ (Optimal Brain Quantizer) algorithm for GPTQ
- FP8 epilogue fusion PTX synthesis (scale application pre-bias in fused kernel)
- Mixed precision: `@fp8_compute` matmul dispatch rewriting in expr.rs
