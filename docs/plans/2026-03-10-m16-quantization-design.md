# M16: Quantization Foundations — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `quant static` block syntax, `QuantizedTensor` runtime type, and weight-only RTN quantization with INT8/INT4 (packed) support, per-tensor/per-channel/per-group granularity, and mixed-precision matmul.

**Architecture:** Three-layer implementation — Rust runtime (packed storage, quantized math), compiler (parser, semantic monomorphization, codegen), NSL stdlib (convenience wrappers).

**Tech Stack:** Rust (Cranelift codegen), NSL standard library

---

## Scope

**In scope (Tier 1):**
- `QuantizedTensor` runtime type with packed INT4/INT8 storage
- Per-tensor, per-channel, and per-group(N) granularity
- Weight-only Round-To-Nearest (RTN) quantization
- `quant static` block syntax with `default`, `exclude` (glob matching), optional `calibration`
- Mixed-precision matmul (`NslTensor @ QuantizedTensor`)
- Model monomorphization (compiler synthesizes quantized model type)
- Compile-time rejection of `.save()` on quantized models

**Out of scope (deferred):**
- FP8 dtype, activation quantization, mixed-precision assignment, sensitivity analysis → M21
- QAT, GPTQ, AWQ, SmoothQuant, hardware-targeted profiles → M22

---

## Layer 1: Rust Runtime

### QuantizedTensor struct

```rust
#[repr(C)]
pub struct QuantizedTensor {
    data: *mut u8,           // packed storage (INT4: 2 per byte, INT8: 1 per byte)
    scale: *mut f32,         // f32 scales (compact — not f64)
    zero_point: *mut u8,     // u8 zero points (0-15 for INT4, 0-255 for INT8)
    shape: *mut i64,         // logical shape (original dimensions)
    ndim: i64,
    dtype: i64,              // 0=INT8, 1=INT4
    granularity: i64,        // 0=PerTensor, 1=PerChannel, 2=PerGroup
    gran_axis: i64,          // axis for PerChannel/PerGroup
    group_size: i64,         // group size for PerGroup (0 otherwise)
    num_scales: i64,         // number of scale/zp entries
    refcount: i64,           // for compiler drop semantics
}
```

### Memory design decisions

- **Scales as f32, not f64:** Even though NslTensor uses f64, quantization metadata doesn't need
  double precision. For per-group(64) INT4: 32 bytes data + 4 bytes scale + 1 byte zp = ~14% overhead.
  With f64+i64 it would be 33% overhead, destroying compression benefits.
- **Zero points as u8:** INT4 zp range is [0, 15], INT8 is [0, 255]. Both fit in u8.
- **Refcount:** Required for compiler memory management (drop semantics on scope exit).

### Quantization math (asymmetric affine)

```
// Scale/zero-point computation:
range = max(max_val - min_val, 1e-7)   // epsilon prevents division by zero
scale = range / (qmax - qmin)
zero_point = clamp(round(-min_val / scale), qmin, qmax)

// Quantize:
q = clamp(round(x / scale) + zero_point, qmin, qmax)

// Dequantize:
x_approx = (q - zero_point) * scale
```

The epsilon clamp on range is critical: groups with all-identical values (common in sparse
models) would otherwise produce scale=0 and NaN propagation.

### INT4 bit packing

```rust
// Buffer MUST be zero-initialized (calloc/checked_alloc_zeroed) before packing.
// Using |= on uninitialized memory blends quantized bits with garbage.

// Pack: two values per byte, low nibble first
packed[i / 2] |= (val & 0x0F) << ((i % 2) * 4)

// Unpack:
val = (packed[i / 2] >> ((i % 2) * 4)) & 0x0F
```

### FFI functions

| Function | Signature | Purpose |
|---|---|---|
| `nsl_qtensor_quantize` | `(tensor, dtype, gran, axis, group_size) -> qt` | Weight-only RTN: reads min/max from tensor directly |
| `nsl_qtensor_quantize_calibrated` | `(tensor, min_t, max_t, dtype, gran, axis, group_size) -> qt` | Quantize with pre-collected calibration stats (future use) |
| `nsl_qtensor_dequantize` | `(qt) -> tensor` | Full dequantize back to f64 NslTensor |
| `nsl_qtensor_matmul_mixed` | `(x_tensor, qw) -> tensor` | Mixed-precision matmul (dequant-on-the-fly) |
| `nsl_qtensor_calibrate_minmax` | `(tensor, running_min, running_max) -> (min, max)` | Update running stats (future use) |
| `nsl_qtensor_free` | `(qt)` | Free all QuantizedTensor memory |
| `nsl_qtensor_dtype` | `(qt) -> int` | Query dtype enum |
| `nsl_qtensor_shape` | `(qt) -> tensor` | Query shape as 1D tensor |

---

## Layer 2: Compiler

### Syntax

```nsl
quant static QuantizedModel from trained_model:
    default: int4, per_group(128)
    calibration:                    # optional — reserved for future activation quantization
        data: cal_loader
        samples: 1000
    exclude: ["*.b", "embedding"]   # glob matching against flattened param names
```

### AST

```rust
Stmt::QuantBlock {
    kind: QuantKind,         // Static (only option in M16)
    name: String,            // "QuantizedModel" — becomes the instance variable name
    source: String,          // "trained_model" — existing model instance
    config: QuantConfig,
}

struct QuantConfig {
    default_dtype: QuantDtype,        // Int8 or Int4
    default_granularity: Granularity, // PerTensor, PerChannel(axis), PerGroup(axis, size)
    calibration: Option<CalibrationConfig>,
    exclude: Vec<String>,             // glob patterns
}
```

### Compiler pipeline

**Parser:** New `quant` keyword triggers `parse_quant_block()`. Indentation-scoped block
with key-value config lines.

**Semantic pass (monomorphization):**
1. Validate source is a known model variable of a model type
2. Deep-clone the source model's AST
3. Resolve `exclude` globs against flattened parameter names (fnmatch-style)
4. Rewrite non-excluded weight field types from `Tensor` to `QuantizedTensor`
5. Re-run semantic analysis on cloned `forward()` — detects `x @ w` where `w: QuantizedTensor`
   and rewrites to `qmatmul_mixed(x, w)` call
6. Register synthesized hidden type (e.g. `__TinyModel_Quantized`)
7. Reject `.save()` calls on `QuantizedTensor` types with compile error

**Codegen desugaring:** The `quant static` block emits:
1. Allocate instance of `__TinyModel_Quantized`
2. For each non-excluded weight field: call `nsl_qtensor_quantize(field, dtype, gran, axis, group_size)`
3. For each excluded field: copy the NslTensor pointer (bump refcount)
4. Bind the instance to the variable name (`QuantizedModel`)

### Type system integration

- `QuantizedTensor` is a new type in the type system, distinct from `Tensor`
- Binary `@` operator checks operand types: if right operand is `QuantizedTensor`, emits
  `qmatmul_mixed` instead of standard `matmul`
- `.save()` on any type containing `QuantizedTensor` fields produces a compile error:
  `"Cannot save quantized model: .nslm format does not yet support QuantizedTensor. Dequantize first or wait for mixed-dtype checkpoint support."`

---

## Layer 3: NSL Standard Library

### `stdlib/nsl/quant/ops.nsl`

```nsl
fn quantize(t: Tensor, dtype: str, granularity: str, axis: int, group_size: int) -> QuantizedTensor:
    let dtype_id = 0
    if dtype == "int4":
        dtype_id = 1

    let gran_id = 0
    if granularity == "per_channel":
        gran_id = 1
    if granularity == "per_group":
        gran_id = 2

    return nsl_qtensor_quantize(t, dtype_id, gran_id, axis, group_size)

fn dequantize(qt: QuantizedTensor) -> Tensor:
    return nsl_qtensor_dequantize(qt)
```

String-to-enum mapping happens in NSL before crossing the FFI boundary.

---

## End-to-End Example

```nsl
import nsl.nn.layers: Linear

model TinyModel(dim: int):
    fc1: Linear = Linear(dim, 128)
    fc2: Linear = Linear(128, dim)

    fn forward(self, x: Tensor) -> Tensor:
        let h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)

let m = TinyModel(64)
# ... training ...

# Quantize weights to INT4, per-group(128), keep biases in f64
quant static qm from m:
    default: int4, per_group(128)
    exclude: ["*.b"]

# Use quantized model — forward() just works
let x = randn([4, 64])
let out = qm.forward(x)
print(out)
```

Under the hood, `qm` is an instance of `__TinyModel_Quantized` where `fc1.w` and `fc2.w`
are `QuantizedTensor` and `fc1.b`/`fc2.b` remain `NslTensor`. The `forward()` method was
monomorphized to use `qmatmul_mixed` for `x @ self.w`.

---

## Test Plan

| # | Test | What it validates |
|---|---|---|
| 1 | INT8 per-tensor roundtrip | quantize → dequantize → values within tolerance |
| 2 | INT8 per-channel roundtrip | per-channel accuracy > per-tensor |
| 3 | INT4 per-group roundtrip | packed storage, dequantize accuracy |
| 4 | INT4 bit-packing exact | pack → unpack → exact value match |
| 5 | Zero-variance group | all-identical values don't produce NaN |
| 6 | Mixed matmul correctness | NslTensor @ QuantizedTensor ≈ NslTensor @ NslTensor |
| 7 | Monomorphized forward | quantized model output ≈ original model output |
| 8 | Exclude glob matching | excluded fields remain NslTensor |
| 9 | Compression ratio | INT4 per-group(64) uses ~50% less memory than INT8 |
| 10 | Reject .save() | quantized model .save() produces compile error |

---

## Roadmap Changes

**M16 updated to:**
> M16: Quantization foundations — `quant static` block, `QuantizedTensor` type, INT8/INT4 weight-only RTN,
> per-tensor/per-channel/per-group, packed INT4, mixed-precision matmul, monomorphized model codegen

**New milestones added:**
> M21: Advanced quantization — FP8 dtype, activation quantization (W8A8) with calibration,
> mixed-precision assignment, sensitivity analysis
>
> M22: Algorithmic quantization — QAT, GPTQ, AWQ, SmoothQuant, hardware-targeted quantization profiles
