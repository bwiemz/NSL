# M16: Quantization Foundations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `quant static` block syntax, `QuantizedTensor` runtime type with packed INT4/INT8 storage, weight-only RTN quantization, and mixed-precision matmul — enabling users to quantize trained model weights with a declarative block.

**Architecture:** Three layers — (1) Rust runtime: `QuantizedTensor` struct, quantize/dequantize/qmatmul FFI functions; (2) Compiler: extend `quant` block parser for `quant static Name from source:` syntax, semantic pass with model monomorphization, codegen emitting quantization calls; (3) NSL stdlib: convenience wrappers in `nsl.quant`.

**Tech Stack:** Rust (Cranelift 0.116 codegen), NSL standard library

**Design doc:** `docs/plans/2026-03-10-m16-quantization-design.md`

---

## Task 1: QuantizedTensor Struct + Allocation/Free

**Files:**
- Create: `crates/nsl-runtime/src/quantize.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` (add `pub mod quantize;`)

**Step 1: Create the QuantizedTensor struct and allocation helpers**

Create `crates/nsl-runtime/src/quantize.rs`:

```rust
use crate::memory::{checked_alloc, checked_alloc_zeroed, checked_free};
use crate::tensor::NslTensor;
use std::ptr;

/// Quantization dtype
const QDTYPE_INT8: i64 = 0;
const QDTYPE_INT4: i64 = 1;

/// Quantization granularity
const QGRAN_PER_TENSOR: i64 = 0;
const QGRAN_PER_CHANNEL: i64 = 1;
const QGRAN_PER_GROUP: i64 = 2;

#[repr(C)]
pub struct QuantizedTensor {
    pub data: *mut u8,         // packed storage (INT4: 2 per byte, INT8: 1 per byte)
    pub scale: *mut f32,       // f32 scales
    pub zero_point: *mut u8,   // u8 zero points
    pub shape: *mut i64,       // logical shape (original dimensions)
    pub ndim: i64,
    pub dtype: i64,            // QDTYPE_INT8=0, QDTYPE_INT4=1
    pub granularity: i64,      // QGRAN_PER_TENSOR=0, PER_CHANNEL=1, PER_GROUP=2
    pub gran_axis: i64,        // axis for per-channel/per-group
    pub group_size: i64,       // group size for per-group (0 otherwise)
    pub num_scales: i64,       // number of scale/zp entries
    pub refcount: i64,
}

impl QuantizedTensor {
    pub unsafe fn from_ptr(ptr: i64) -> &'static mut Self {
        &mut *(ptr as *mut Self)
    }
}

/// Allocate a QuantizedTensor on the heap. Returns pointer as i64.
fn alloc_qtensor(
    shape: &[i64],
    dtype: i64,
    granularity: i64,
    gran_axis: i64,
    group_size: i64,
) -> *mut QuantizedTensor {
    let ndim = shape.len() as i64;
    let total_elements: i64 = shape.iter().product();

    // Compute data buffer size
    let data_bytes = match dtype {
        QDTYPE_INT4 => ((total_elements as usize) + 1) / 2, // 2 values per byte
        QDTYPE_INT8 => total_elements as usize,
        _ => {
            eprintln!("nsl: invalid quantization dtype {}", dtype);
            std::process::abort();
        }
    };

    // Compute number of scale/zp entries
    let num_scales = compute_num_scales(shape, granularity, gran_axis, group_size);

    // Allocate all buffers
    // CRITICAL: data must be zeroed for INT4 packing (|= on uninitialized memory = garbage)
    let data = checked_alloc_zeroed(data_bytes);
    let scale = checked_alloc(num_scales * std::mem::size_of::<f32>()) as *mut f32;
    let zero_point = checked_alloc_zeroed(num_scales); // u8, zeroed
    let shape_buf = checked_alloc(shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        ptr::copy_nonoverlapping(shape.as_ptr(), shape_buf, shape.len());
    }

    let qt = checked_alloc(std::mem::size_of::<QuantizedTensor>()) as *mut QuantizedTensor;
    unsafe {
        (*qt).data = data;
        (*qt).scale = scale;
        (*qt).zero_point = zero_point;
        (*qt).shape = shape_buf;
        (*qt).ndim = ndim;
        (*qt).dtype = dtype;
        (*qt).granularity = granularity;
        (*qt).gran_axis = gran_axis;
        (*qt).group_size = group_size;
        (*qt).num_scales = num_scales as i64;
        (*qt).refcount = 1;
    }
    qt
}

/// Compute number of scale/zero-point entries based on granularity.
fn compute_num_scales(
    shape: &[i64],
    granularity: i64,
    gran_axis: i64,
    group_size: i64,
) -> usize {
    match granularity {
        QGRAN_PER_TENSOR => 1,
        QGRAN_PER_CHANNEL => {
            if gran_axis < 0 || gran_axis >= shape.len() as i64 {
                eprintln!("nsl: per_channel axis {} out of bounds for ndim {}", gran_axis, shape.len());
                std::process::abort();
            }
            shape[gran_axis as usize] as usize
        }
        QGRAN_PER_GROUP => {
            if gran_axis < 0 || gran_axis >= shape.len() as i64 {
                eprintln!("nsl: per_group axis {} out of bounds for ndim {}", gran_axis, shape.len());
                std::process::abort();
            }
            if group_size <= 0 {
                eprintln!("nsl: per_group group_size must be > 0, got {}", group_size);
                std::process::abort();
            }
            let axis_size = shape[gran_axis as usize] as usize;
            let num_groups = (axis_size + group_size as usize - 1) / group_size as usize;
            // Total = num_groups * product of all other dims along that axis
            let outer: usize = shape[..gran_axis as usize].iter().map(|&d| d as usize).product::<usize>().max(1);
            outer * num_groups
        }
        _ => {
            eprintln!("nsl: invalid granularity {}", granularity);
            std::process::abort();
        }
    }
}

/// Free all memory owned by a QuantizedTensor.
#[no_mangle]
pub extern "C" fn nsl_qtensor_free(ptr: i64) {
    if ptr == 0 { return; }
    let qt = unsafe { QuantizedTensor::from_ptr(ptr) };
    let total_elements: i64 = unsafe {
        (0..qt.ndim as usize).map(|i| *qt.shape.add(i)).product()
    };
    let data_bytes = match qt.dtype {
        QDTYPE_INT4 => ((total_elements as usize) + 1) / 2,
        _ => total_elements as usize,
    };
    unsafe {
        checked_free(qt.data, data_bytes);
        checked_free(qt.scale as *mut u8, qt.num_scales as usize * std::mem::size_of::<f32>());
        checked_free(qt.zero_point, qt.num_scales as usize);
        checked_free(qt.shape as *mut u8, qt.ndim as usize * std::mem::size_of::<i64>());
        checked_free(ptr as *mut u8, std::mem::size_of::<QuantizedTensor>());
    }
}

/// Increment refcount.
#[no_mangle]
pub extern "C" fn nsl_qtensor_addref(ptr: i64) {
    if ptr == 0 { return; }
    let qt = unsafe { QuantizedTensor::from_ptr(ptr) };
    qt.refcount += 1;
}

/// Decrement refcount, free if zero.
#[no_mangle]
pub extern "C" fn nsl_qtensor_release(ptr: i64) {
    if ptr == 0 { return; }
    let qt = unsafe { QuantizedTensor::from_ptr(ptr) };
    qt.refcount -= 1;
    if qt.refcount <= 0 {
        nsl_qtensor_free(ptr);
    }
}
```

**Step 2: Register the module**

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod quantize;
```

**Step 3: Build and verify**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-runtime/src/quantize.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(m16): add QuantizedTensor struct with allocation and refcounting"
```

---

## Task 2: INT4/INT8 Quantize + Dequantize Runtime Functions

**Files:**
- Modify: `crates/nsl-runtime/src/quantize.rs`

**Step 1: Add INT4 pack/unpack helpers and quantize function**

Append to `crates/nsl-runtime/src/quantize.rs`:

```rust
/// Pack a u8 value (0-15) into the INT4 packed buffer at logical index i.
/// PRECONDITION: buffer must be zero-initialized.
#[inline]
fn int4_pack(buf: *mut u8, i: usize, val: u8) {
    unsafe {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        *buf.add(byte_idx) |= (val & 0x0F) << shift;
    }
}

/// Unpack a u8 value (0-15) from the INT4 packed buffer at logical index i.
#[inline]
fn int4_unpack(buf: *const u8, i: usize) -> u8 {
    unsafe {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        (*buf.add(byte_idx) >> shift) & 0x0F
    }
}

/// Quantize an NslTensor to a QuantizedTensor using weight-only RTN.
/// Computes scale/zero-point from the tensor's own min/max values.
#[no_mangle]
pub extern "C" fn nsl_qtensor_quantize(
    tensor_ptr: i64,
    dtype: i64,
    granularity: i64,
    gran_axis: i64,
    group_size: i64,
) -> i64 {
    let tensor = unsafe { NslTensor::from_ptr(tensor_ptr) };
    let shape: Vec<i64> = unsafe {
        (0..tensor.ndim as usize).map(|i| *tensor.shape.add(i)).collect()
    };
    let total = tensor.len as usize;

    let (qmin, qmax): (f64, f64) = match dtype {
        QDTYPE_INT8 => (0.0, 255.0),
        QDTYPE_INT4 => (0.0, 15.0),
        _ => {
            eprintln!("nsl: unsupported quantization dtype {}", dtype);
            std::process::abort();
        }
    };

    let qt = alloc_qtensor(&shape, dtype, granularity, gran_axis, group_size);

    // Iterate over groups and quantize
    match granularity {
        QGRAN_PER_TENSOR => {
            // One scale/zp for entire tensor
            let (min_val, max_val) = find_minmax(tensor.data, total);
            let (scale, zp) = compute_scale_zp(min_val, max_val, qmin, qmax);
            unsafe {
                *(*qt).scale = scale;
                *(*qt).zero_point = zp;
            }
            for i in 0..total {
                let val = unsafe { *tensor.data.add(i) };
                let q = quantize_val(val, scale, zp, qmin, qmax);
                write_quantized(unsafe { &*qt }, i, q);
            }
        }
        QGRAN_PER_CHANNEL => {
            quantize_per_channel(tensor, unsafe { &mut *qt }, &shape, gran_axis, qmin, qmax);
        }
        QGRAN_PER_GROUP => {
            quantize_per_group(tensor, unsafe { &mut *qt }, &shape, gran_axis, group_size, qmin, qmax);
        }
        _ => {
            eprintln!("nsl: invalid granularity {}", granularity);
            std::process::abort();
        }
    }

    qt as i64
}

fn find_minmax(data: *const f64, len: usize) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for i in 0..len {
        let v = unsafe { *data.add(i) };
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }
    (min_val, max_val)
}

fn compute_scale_zp(min_val: f64, max_val: f64, qmin: f64, qmax: f64) -> (f32, u8) {
    let range = f64::max(max_val - min_val, 1e-7); // epsilon prevents div-by-zero
    let scale = (range / (qmax - qmin)) as f32;
    let zp = ((-min_val / scale as f64) + qmin).round().clamp(qmin, qmax) as u8;
    (scale, zp)
}

#[inline]
fn quantize_val(val: f64, scale: f32, zp: u8, qmin: f64, qmax: f64) -> u8 {
    let q = (val / scale as f64 + zp as f64).round().clamp(qmin, qmax);
    q as u8
}

fn write_quantized(qt: &QuantizedTensor, idx: usize, val: u8) {
    match qt.dtype {
        QDTYPE_INT4 => int4_pack(qt.data, idx, val),
        QDTYPE_INT8 => unsafe { *qt.data.add(idx) = val; },
        _ => unreachable!(),
    }
}

fn read_quantized(qt: &QuantizedTensor, idx: usize) -> u8 {
    match qt.dtype {
        QDTYPE_INT4 => int4_unpack(qt.data, idx),
        QDTYPE_INT8 => unsafe { *qt.data.add(idx) },
        _ => unreachable!(),
    }
}

fn quantize_per_channel(
    tensor: &NslTensor,
    qt: &mut QuantizedTensor,
    shape: &[i64],
    axis: i64,
    qmin: f64,
    qmax: f64,
) {
    let axis = axis as usize;
    let num_channels = shape[axis] as usize;
    let inner_size: usize = shape[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let channel_size: usize = inner_size;
    let outer_size: usize = shape[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);

    for c in 0..num_channels {
        // Find min/max across all elements in this channel
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for o in 0..outer_size {
            let base = o * num_channels * inner_size + c * inner_size;
            for i in 0..channel_size {
                let v = unsafe { *tensor.data.add(base + i) };
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }
        }
        let (scale, zp) = compute_scale_zp(min_val, max_val, qmin, qmax);
        unsafe {
            *qt.scale.add(c) = scale;
            *qt.zero_point.add(c) = zp;
        }
        // Quantize elements in this channel
        for o in 0..outer_size {
            let base = o * num_channels * inner_size + c * inner_size;
            for i in 0..channel_size {
                let val = unsafe { *tensor.data.add(base + i) };
                let q = quantize_val(val, scale, zp, qmin, qmax);
                write_quantized(qt, base + i, q);
            }
        }
    }
}

fn quantize_per_group(
    tensor: &NslTensor,
    qt: &mut QuantizedTensor,
    shape: &[i64],
    axis: i64,
    group_size: i64,
    qmin: f64,
    qmax: f64,
) {
    let axis = axis as usize;
    let axis_size = shape[axis] as usize;
    let gs = group_size as usize;
    let num_groups = (axis_size + gs - 1) / gs;
    let inner_size: usize = shape[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let outer_size: usize = shape[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);

    let mut scale_idx = 0usize;
    for o in 0..outer_size {
        for g in 0..num_groups {
            let start = g * gs;
            let end = std::cmp::min(start + gs, axis_size);

            // Find min/max for this group
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for c in start..end {
                let base = o * axis_size * inner_size + c * inner_size;
                for i in 0..inner_size {
                    let v = unsafe { *tensor.data.add(base + i) };
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
            }
            let (scale, zp) = compute_scale_zp(min_val, max_val, qmin, qmax);
            unsafe {
                *qt.scale.add(scale_idx) = scale;
                *qt.zero_point.add(scale_idx) = zp;
            }
            // Quantize group elements
            for c in start..end {
                let base = o * axis_size * inner_size + c * inner_size;
                for i in 0..inner_size {
                    let val = unsafe { *tensor.data.add(base + i) };
                    let q = quantize_val(val, scale, zp, qmin, qmax);
                    write_quantized(qt, base + i, q);
                }
            }
            scale_idx += 1;
        }
    }
}
```

**Step 2: Add dequantize function**

Append to `crates/nsl-runtime/src/quantize.rs`:

```rust
/// Dequantize a QuantizedTensor back to a full-precision NslTensor.
#[no_mangle]
pub extern "C" fn nsl_qtensor_dequantize(qt_ptr: i64) -> i64 {
    let qt = unsafe { QuantizedTensor::from_ptr(qt_ptr) };
    let shape: Vec<i64> = unsafe {
        (0..qt.ndim as usize).map(|i| *qt.shape.add(i)).collect()
    };
    let total: usize = shape.iter().map(|&d| d as usize).product();

    // Allocate output NslTensor
    let out_data = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;

    match qt.granularity {
        QGRAN_PER_TENSOR => {
            let scale = unsafe { *qt.scale } as f64;
            let zp = unsafe { *qt.zero_point } as f64;
            for i in 0..total {
                let q = read_quantized(qt, i) as f64;
                unsafe { *out_data.add(i) = (q - zp) * scale; }
            }
        }
        QGRAN_PER_CHANNEL => {
            dequantize_per_channel(qt, out_data, &shape);
        }
        QGRAN_PER_GROUP => {
            dequantize_per_group(qt, out_data, &shape);
        }
        _ => {
            eprintln!("nsl: invalid granularity in dequantize");
            std::process::abort();
        }
    }

    // Build NslTensor
    let shape_buf = checked_alloc(shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    let strides_buf = checked_alloc(shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        ptr::copy_nonoverlapping(shape.as_ptr(), shape_buf, shape.len());
        // Row-major strides
        let mut stride = 1i64;
        for i in (0..shape.len()).rev() {
            *strides_buf.add(i) = stride;
            stride *= shape[i];
        }
    }

    let tensor = checked_alloc(std::mem::size_of::<NslTensor>()) as *mut NslTensor;
    unsafe {
        (*tensor).data = out_data;
        (*tensor).shape = shape_buf;
        (*tensor).strides = strides_buf;
        (*tensor).ndim = shape.len() as i64;
        (*tensor).len = total as i64;
        (*tensor).refcount = 1;
    }
    tensor as i64
}

fn dequantize_per_channel(qt: &QuantizedTensor, out: *mut f64, shape: &[i64]) {
    let axis = qt.gran_axis as usize;
    let num_channels = shape[axis] as usize;
    let inner_size: usize = shape[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let outer_size: usize = shape[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);

    for c in 0..num_channels {
        let scale = unsafe { *qt.scale.add(c) } as f64;
        let zp = unsafe { *qt.zero_point.add(c) } as f64;
        for o in 0..outer_size {
            let base = o * num_channels * inner_size + c * inner_size;
            for i in 0..inner_size {
                let q = read_quantized(qt, base + i) as f64;
                unsafe { *out.add(base + i) = (q - zp) * scale; }
            }
        }
    }
}

fn dequantize_per_group(qt: &QuantizedTensor, out: *mut f64, shape: &[i64]) {
    let axis = qt.gran_axis as usize;
    let axis_size = shape[axis] as usize;
    let gs = qt.group_size as usize;
    let num_groups = (axis_size + gs - 1) / gs;
    let inner_size: usize = shape[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let outer_size: usize = shape[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);

    let mut scale_idx = 0usize;
    for o in 0..outer_size {
        for g in 0..num_groups {
            let start = g * gs;
            let end = std::cmp::min(start + gs, axis_size);
            let scale = unsafe { *qt.scale.add(scale_idx) } as f64;
            let zp = unsafe { *qt.zero_point.add(scale_idx) } as f64;
            for c in start..end {
                let base = o * axis_size * inner_size + c * inner_size;
                for i in 0..inner_size {
                    let q = read_quantized(qt, base + i) as f64;
                    unsafe { *out.add(base + i) = (q - zp) * scale; }
                }
            }
            scale_idx += 1;
        }
    }
}
```

**Step 3: Add query helpers**

Append to `crates/nsl-runtime/src/quantize.rs`:

```rust
/// Query dtype of a QuantizedTensor.
#[no_mangle]
pub extern "C" fn nsl_qtensor_dtype(qt_ptr: i64) -> i64 {
    let qt = unsafe { QuantizedTensor::from_ptr(qt_ptr) };
    qt.dtype
}

/// Query shape of a QuantizedTensor as a 1D NslTensor.
#[no_mangle]
pub extern "C" fn nsl_qtensor_shape(qt_ptr: i64) -> i64 {
    let qt = unsafe { QuantizedTensor::from_ptr(qt_ptr) };
    let ndim = qt.ndim as usize;

    let out_data = checked_alloc(ndim * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..ndim {
        unsafe { *out_data.add(i) = *qt.shape.add(i) as f64; }
    }

    let shape_buf = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    let strides_buf = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape_buf = ndim as i64;
        *strides_buf = 1;
    }

    let tensor = checked_alloc(std::mem::size_of::<NslTensor>()) as *mut NslTensor;
    unsafe {
        (*tensor).data = out_data;
        (*tensor).shape = shape_buf;
        (*tensor).strides = strides_buf;
        (*tensor).ndim = 1;
        (*tensor).len = ndim as i64;
        (*tensor).refcount = 1;
    }
    tensor as i64
}
```

**Step 4: Build and verify**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 5: Commit**

```bash
git add crates/nsl-runtime/src/quantize.rs
git commit -m "feat(m16): add quantize, dequantize, and query runtime functions"
```

---

## Task 3: Mixed-Precision Matmul

**Files:**
- Modify: `crates/nsl-runtime/src/quantize.rs`

**Step 1: Add mixed-precision matmul (dequantize-on-the-fly)**

Append to `crates/nsl-runtime/src/quantize.rs`:

```rust
/// Mixed-precision matmul: NslTensor (f64) @ QuantizedTensor -> NslTensor (f64).
/// Dequantizes the quantized weight on-the-fly during matmul.
/// Supports 2D: [M, K] @ [K, N] -> [M, N]
/// Supports batched: [..., M, K] @ [K, N] -> [..., M, N]
#[no_mangle]
pub extern "C" fn nsl_qtensor_matmul_mixed(x_ptr: i64, qw_ptr: i64) -> i64 {
    let x = unsafe { NslTensor::from_ptr(x_ptr) };
    let qw = unsafe { QuantizedTensor::from_ptr(qw_ptr) };

    // qw must be 2D [K, N]
    if qw.ndim != 2 {
        eprintln!("nsl: qmatmul_mixed requires 2D quantized weight, got ndim={}", qw.ndim);
        std::process::abort();
    }
    let k_w = unsafe { *qw.shape } as usize;
    let n = unsafe { *qw.shape.add(1) } as usize;

    // x must have last dim = K
    let x_shape: Vec<i64> = unsafe {
        (0..x.ndim as usize).map(|i| *x.shape.add(i)).collect()
    };
    let k_x = *x_shape.last().unwrap() as usize;
    if k_x != k_w {
        eprintln!("nsl: qmatmul_mixed dimension mismatch: x last dim {} != weight dim 0 {}", k_x, k_w);
        std::process::abort();
    }

    // Dequantize the weight matrix fully (simple approach for M16)
    let w_deq_ptr = nsl_qtensor_dequantize(qw_ptr);
    let w_deq = unsafe { NslTensor::from_ptr(w_deq_ptr) };

    // Use existing tensor matmul
    let result = crate::tensor::nsl_tensor_matmul(x_ptr, w_deq_ptr);

    // Release the temporary dequantized tensor
    crate::tensor::nsl_tensor_release(w_deq_ptr);

    result
}
```

**Step 2: Build and verify**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 3: Commit**

```bash
git add crates/nsl-runtime/src/quantize.rs
git commit -m "feat(m16): add mixed-precision matmul (NslTensor @ QuantizedTensor)"
```

---

## Task 4: Register Runtime Functions in Codegen

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

**Step 1: Register FFI functions in builtins.rs**

Add to the `RUNTIME_FUNCTIONS` array in `crates/nsl-codegen/src/builtins.rs`:

```rust
// Quantization (M16)
("nsl_qtensor_quantize", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
("nsl_qtensor_dequantize", &[types::I64], Some(types::I64)),
("nsl_qtensor_matmul_mixed", &[types::I64, types::I64], Some(types::I64)),
("nsl_qtensor_free", &[types::I64], None),
("nsl_qtensor_addref", &[types::I64], None),
("nsl_qtensor_release", &[types::I64], None),
("nsl_qtensor_dtype", &[types::I64], Some(types::I64)),
("nsl_qtensor_shape", &[types::I64], Some(types::I64)),
```

**Step 2: Add builtin dispatch in expr.rs**

Add dispatch cases in `crates/nsl-codegen/src/expr.rs` (near other builtin dispatches):

```rust
// nsl_qtensor_quantize(tensor, dtype, gran, axis, group_size) -> qtensor
if func_name == "nsl_qtensor_quantize" && !self.functions.contains_key(&func_name) {
    if args.len() != 5 {
        return Err(CodegenError::new("nsl_qtensor_quantize() takes 5 arguments"));
    }
    let a0 = self.compile_expr(builder, state, &args[0].value)?;
    let a1 = self.compile_expr(builder, state, &args[1].value)?;
    let a2 = self.compile_expr(builder, state, &args[2].value)?;
    let a3 = self.compile_expr(builder, state, &args[3].value)?;
    let a4 = self.compile_expr(builder, state, &args[4].value)?;
    return self.compile_call_by_name(builder, "nsl_qtensor_quantize", &[a0, a1, a2, a3, a4]);
}

// nsl_qtensor_dequantize(qtensor) -> tensor
if func_name == "nsl_qtensor_dequantize" && !self.functions.contains_key(&func_name) {
    if args.len() != 1 {
        return Err(CodegenError::new("nsl_qtensor_dequantize() takes 1 argument"));
    }
    let a0 = self.compile_expr(builder, state, &args[0].value)?;
    return self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[a0]);
}

// nsl_qtensor_matmul_mixed(tensor, qtensor) -> tensor
if func_name == "nsl_qtensor_matmul_mixed" && !self.functions.contains_key(&func_name) {
    if args.len() != 2 {
        return Err(CodegenError::new("nsl_qtensor_matmul_mixed() takes 2 arguments"));
    }
    let a0 = self.compile_expr(builder, state, &args[0].value)?;
    let a1 = self.compile_expr(builder, state, &args[1].value)?;
    return self.compile_call_by_name(builder, "nsl_qtensor_matmul_mixed", &[a0, a1]);
}

// nsl_qtensor_dtype(qtensor) -> int
if func_name == "nsl_qtensor_dtype" && !self.functions.contains_key(&func_name) {
    if args.len() != 1 {
        return Err(CodegenError::new("nsl_qtensor_dtype() takes 1 argument"));
    }
    let a0 = self.compile_expr(builder, state, &args[0].value)?;
    return self.compile_call_by_name(builder, "nsl_qtensor_dtype", &[a0]);
}

// nsl_qtensor_shape(qtensor) -> tensor
if func_name == "nsl_qtensor_shape" && !self.functions.contains_key(&func_name) {
    if args.len() != 1 {
        return Err(CodegenError::new("nsl_qtensor_shape() takes 1 argument"));
    }
    let a0 = self.compile_expr(builder, state, &args[0].value)?;
    return self.compile_call_by_name(builder, "nsl_qtensor_shape", &[a0]);
}
```

**Step 3: Build and verify**

Run: `cargo build -p nsl-codegen`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m16): register quantization FFI functions in codegen"
```

---

## Task 5: Register Quantization Builtins in Semantic Checker

**Files:**
- Modify: `crates/nsl-semantic/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/types.rs`

**Step 1: Add QuantizedTensor type variant**

In `crates/nsl-semantic/src/types.rs`, add a new variant to the `Type` enum:

```rust
QuantizedTensor,
```

This is a simple opaque type — quantized tensors are always passed as i64 pointers at the IR level, same as `Tensor`. The semantic difference enables the `@` operator rewriting.

**Step 2: Register quantization builtins in semantic checker**

In `crates/nsl-semantic/src/builtins.rs`, add within `register_builtins()`:

```rust
// Quantization builtins (M16)
def("nsl_qtensor_quantize", Type::Function {
    params: vec![Type::Tensor { shape: Shape::unknown(), dtype: DType::F64, device: Device::Cpu },
                 Type::Int, Type::Int, Type::Int, Type::Int],
    ret: Box::new(Type::QuantizedTensor),
});
def("nsl_qtensor_dequantize", Type::Function {
    params: vec![Type::QuantizedTensor],
    ret: Box::new(Type::Tensor { shape: Shape::unknown(), dtype: DType::F64, device: Device::Cpu }),
});
def("nsl_qtensor_matmul_mixed", Type::Function {
    params: vec![Type::Tensor { shape: Shape::unknown(), dtype: DType::F64, device: Device::Cpu },
                 Type::QuantizedTensor],
    ret: Box::new(Type::Tensor { shape: Shape::unknown(), dtype: DType::F64, device: Device::Cpu }),
});
def("nsl_qtensor_dtype", Type::Function {
    params: vec![Type::QuantizedTensor],
    ret: Box::new(Type::Int),
});
def("nsl_qtensor_shape", Type::Function {
    params: vec![Type::QuantizedTensor],
    ret: Box::new(Type::Tensor { shape: Shape::unknown(), dtype: DType::F64, device: Device::Cpu }),
});
```

**Step 3: Build and verify**

Run: `cargo build -p nsl-semantic`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-semantic/src/builtins.rs crates/nsl-semantic/src/types.rs
git commit -m "feat(m16): add QuantizedTensor type and register quantization builtins"
```

---

## Task 6: Extend Quant Block Parser for `quant static Name from source:`

**Files:**
- Modify: `crates/nsl-ast/src/block.rs` (extend `QuantBlock` struct)
- Modify: `crates/nsl-parser/src/block.rs` (rewrite `parse_quant_block_stmt`)

**Step 1: Extend the QuantBlock AST node**

Replace the existing `QuantBlock` struct in `crates/nsl-ast/src/block.rs`:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct QuantBlock {
    pub kind: QuantKind,
    pub name: Symbol,           // output variable name (e.g. "QuantizedModel")
    pub source: Symbol,         // source model variable (e.g. "trained_model")
    pub default_dtype: Option<QuantDtype>,
    pub default_granularity: Option<QuantGranularity>,
    pub exclude: Vec<String>,   // glob patterns for excluded fields
    pub calibration: Option<CalibrationConfig>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum QuantKind {
    Static,
}

#[derive(Debug, Clone, Serialize)]
pub enum QuantDtype {
    Int8,
    Int4,
}

#[derive(Debug, Clone, Serialize)]
pub enum QuantGranularity {
    PerTensor,
    PerChannel(i64),       // axis
    PerGroup(i64, i64),    // (axis, group_size)
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationConfig {
    pub data: Symbol,
    pub samples: i64,
}
```

**Step 2: Rewrite the quant block parser**

Replace `parse_quant_block_stmt` in `crates/nsl-parser/src/block.rs`:

```rust
/// Parse: quant static Name from source:
///     default: int4, per_group(128)
///     exclude: ["*.b", "embedding"]
///     calibration:
///         data: cal_loader
///         samples: 1000
pub fn parse_quant_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'quant'

    // Expect 'static' keyword (only kind supported in M16)
    let kind = if let TokenKind::Ident(sym) = p.peek().clone() {
        let name = p.interner.resolve(sym).unwrap_or("").to_string();
        if name == "static" {
            p.advance();
            QuantKind::Static
        } else {
            p.error_at(p.current_span(), &format!("expected 'static' after 'quant', got '{}'", name));
            QuantKind::Static
        }
    } else {
        p.error_at(p.current_span(), "expected 'static' after 'quant'");
        QuantKind::Static
    };

    // Name of the output variable
    let (name, _) = p.expect_ident();

    // Expect 'from' keyword
    if let TokenKind::Ident(sym) = p.peek().clone() {
        let kw = p.interner.resolve(sym).unwrap_or("").to_string();
        if kw == "from" {
            p.advance();
        } else {
            p.error_at(p.current_span(), &format!("expected 'from' keyword, got '{}'", kw));
        }
    } else {
        p.error_at(p.current_span(), "expected 'from' keyword");
    }

    // Source model variable
    let (source, _) = p.expect_ident();

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut default_dtype = None;
    let mut default_granularity = None;
    let mut exclude = Vec::new();
    let mut calibration = None;

    // Parse config lines
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) { break; }

        if let TokenKind::Ident(sym) = p.peek().clone() {
            let key = p.interner.resolve(sym).unwrap_or("").to_string();
            match key.as_str() {
                "default" => {
                    p.advance(); // consume 'default'
                    p.expect(&TokenKind::Colon);
                    // Parse dtype: int4 or int8
                    if let TokenKind::Ident(dt_sym) = p.peek().clone() {
                        let dt = p.interner.resolve(dt_sym).unwrap_or("").to_string();
                        match dt.as_str() {
                            "int4" => { default_dtype = Some(QuantDtype::Int4); p.advance(); }
                            "int8" => { default_dtype = Some(QuantDtype::Int8); p.advance(); }
                            _ => p.error_at(p.current_span(), &format!("unknown quant dtype '{}'", dt)),
                        }
                    }
                    // Optional comma + granularity
                    if p.eat(&TokenKind::Comma) {
                        default_granularity = Some(parse_quant_granularity(p));
                    }
                    p.expect_end_of_stmt();
                }
                "exclude" => {
                    p.advance(); // consume 'exclude'
                    p.expect(&TokenKind::Colon);
                    // Parse list of string literals: ["*.b", "embedding"]
                    p.expect(&TokenKind::LeftBracket);
                    while !p.at(&TokenKind::RightBracket) && !p.at(&TokenKind::Eof) {
                        if let TokenKind::StringLiteral(s) = p.peek().clone() {
                            let pattern = p.interner.resolve(s).unwrap_or("").to_string();
                            exclude.push(pattern);
                            p.advance();
                        }
                        if !p.eat(&TokenKind::Comma) { break; }
                    }
                    p.expect(&TokenKind::RightBracket);
                    p.expect_end_of_stmt();
                }
                "calibration" => {
                    p.advance(); // consume 'calibration'
                    p.expect(&TokenKind::Colon);
                    p.skip_newlines();
                    p.expect(&TokenKind::Indent);
                    p.skip_newlines();
                    let mut cal_data = None;
                    let mut cal_samples = 1000i64;
                    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
                        p.skip_newlines();
                        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) { break; }
                        if let TokenKind::Ident(csym) = p.peek().clone() {
                            let ckey = p.interner.resolve(csym).unwrap_or("").to_string();
                            match ckey.as_str() {
                                "data" => {
                                    p.advance(); p.expect(&TokenKind::Colon);
                                    let (dsym, _) = p.expect_ident();
                                    cal_data = Some(dsym);
                                    p.expect_end_of_stmt();
                                }
                                "samples" => {
                                    p.advance(); p.expect(&TokenKind::Colon);
                                    if let TokenKind::IntLiteral(n) = p.peek().clone() {
                                        cal_samples = n;
                                        p.advance();
                                    }
                                    p.expect_end_of_stmt();
                                }
                                _ => {
                                    p.error_at(p.current_span(), &format!("unknown calibration key '{}'", ckey));
                                    p.advance();
                                }
                            }
                        } else {
                            p.advance();
                        }
                    }
                    p.eat(&TokenKind::Dedent);
                    if let Some(data_sym) = cal_data {
                        calibration = Some(CalibrationConfig { data: data_sym, samples: cal_samples });
                    }
                }
                _ => {
                    p.error_at(p.current_span(), &format!("unknown quant config key '{}'", key));
                    p.advance();
                    p.expect_end_of_stmt();
                }
            }
        } else {
            p.advance(); // skip unexpected token
        }
    }
    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::QuantBlock(QuantBlock {
            kind, name, source,
            default_dtype, default_granularity,
            exclude, calibration, span,
        }),
        span,
        id: p.next_node_id(),
    }
}

fn parse_quant_granularity(p: &mut Parser) -> QuantGranularity {
    if let TokenKind::Ident(sym) = p.peek().clone() {
        let name = p.interner.resolve(sym).unwrap_or("").to_string();
        match name.as_str() {
            "per_tensor" => { p.advance(); QuantGranularity::PerTensor }
            "per_channel" => {
                p.advance();
                // Optional (axis) — default axis=0
                let axis = if p.eat(&TokenKind::LeftParen) {
                    let a = if let TokenKind::IntLiteral(n) = p.peek().clone() {
                        p.advance(); n
                    } else { 0 };
                    p.expect(&TokenKind::RightParen);
                    a
                } else { 0 };
                QuantGranularity::PerChannel(axis)
            }
            "per_group" => {
                p.advance();
                p.expect(&TokenKind::LeftParen);
                let gs = if let TokenKind::IntLiteral(n) = p.peek().clone() {
                    p.advance(); n
                } else {
                    p.error_at(p.current_span(), "per_group requires a group size");
                    128
                };
                p.expect(&TokenKind::RightParen);
                // Default axis=0 for per-group
                QuantGranularity::PerGroup(0, gs)
            }
            _ => {
                p.error_at(p.current_span(), &format!("unknown granularity '{}'", name));
                p.advance();
                QuantGranularity::PerTensor
            }
        }
    } else {
        p.error_at(p.current_span(), "expected granularity (per_tensor, per_channel, per_group)");
        QuantGranularity::PerTensor
    }
}
```

**Step 3: Update any AST imports/re-exports if needed**

Check `crates/nsl-ast/src/lib.rs` to ensure `QuantKind`, `QuantDtype`, `QuantGranularity`, `CalibrationConfig` are accessible from the block module.

**Step 4: Build and verify**

Run: `cargo build`
Expected: Compiles. The old `QuantBlock { config, body, span }` references in semantic/codegen will break — fix them in the next tasks.

**Step 5: Commit**

```bash
git add crates/nsl-ast/src/block.rs crates/nsl-parser/src/block.rs
git commit -m "feat(m16): extend quant block parser for 'quant static Name from source:' syntax"
```

---

## Task 7: Semantic Pass — Quant Block Validation

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

**Step 1: Add quant block semantic checking**

Find the existing `StmtKind::QuantBlock` handler in `checker.rs` and replace with:

```rust
StmtKind::QuantBlock(ref quant) => {
    // 1. Validate source is a known model variable
    let source_ty = self.lookup_type(quant.source);
    let source_model = match &source_ty {
        Type::Model { name, fields, methods } => {
            (name.clone(), fields.clone(), methods.clone())
        }
        _ => {
            self.error(quant.span, format!(
                "quant block source '{}' must be a model, got {:?}",
                self.resolve_sym(quant.source), source_ty
            ));
            return;
        }
    };

    // 2. Resolve exclude globs against flattened field names
    let excluded_fields: Vec<Symbol> = source_model.1.iter()
        .filter(|(field_name, _)| {
            let name_str = self.resolve_sym(*field_name).to_string();
            quant.exclude.iter().any(|pattern| glob_match(pattern, &name_str))
        })
        .map(|(sym, _)| *sym)
        .collect();

    // 3. Validate calibration data variable if present
    if let Some(ref cal) = quant.calibration {
        let cal_ty = self.lookup_type(cal.data);
        // Just verify the variable exists — calibration is optional in M16
        if matches!(cal_ty, Type::Error | Type::Unknown) {
            self.error(quant.span, format!(
                "calibration data '{}' not found",
                self.resolve_sym(cal.data)
            ));
        }
    }

    // 4. Build the quantized model type
    // Clone fields, replacing Tensor with QuantizedTensor for non-excluded fields
    let quant_fields: Vec<(Symbol, Type)> = source_model.1.iter()
        .map(|(name, ty)| {
            if excluded_fields.contains(name) {
                (*name, ty.clone())
            } else if is_tensor_type(ty) {
                (*name, Type::QuantizedTensor)
            } else {
                (*name, ty.clone())
            }
        })
        .collect();

    // 5. Register the output variable with a Model type containing QuantizedTensor fields
    let quant_model_ty = Type::Model {
        name: quant.name,
        fields: quant_fields,
        methods: source_model.2.clone(), // methods cloned — codegen handles monomorphization
    };
    self.declare_symbol(quant.name, quant_model_ty, quant.span, false, false);
}
```

**Step 2: Add glob matching helper**

Add a simple glob matcher function (supports `*` wildcard):

```rust
/// Simple glob match supporting '*' wildcards.
fn glob_match(pattern: &str, text: &str) -> bool {
    let mut pi = 0usize;
    let mut ti = 0usize;
    let pb = pattern.as_bytes();
    let tb = text.as_bytes();
    let mut star_pi = usize::MAX;
    let mut star_ti = 0usize;

    while ti < tb.len() {
        if pi < pb.len() && (pb[pi] == b'?' || pb[pi] == tb[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pb.len() && pb[pi] == b'*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < pb.len() && pb[pi] == b'*' {
        pi += 1;
    }
    pi == pb.len()
}
```

**Step 3: Add `.save()` rejection for QuantizedTensor**

In the method call checker (where `.save()` is handled), add:

```rust
// Reject .save() on models containing QuantizedTensor fields
if method_name == "save" {
    if let Type::Model { fields, .. } = &receiver_ty {
        if fields.iter().any(|(_, ty)| matches!(ty, Type::QuantizedTensor)) {
            self.error(span,
                "Cannot save quantized model: .nslm format does not yet support QuantizedTensor. \
                 Dequantize first or wait for mixed-dtype checkpoint support.".to_string()
            );
            return;
        }
    }
}
```

**Step 4: Build and verify**

Run: `cargo build -p nsl-semantic`
Expected: Compiles with no errors.

**Step 5: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs
git commit -m "feat(m16): add quant block semantic validation with glob matching and .save() rejection"
```

---

## Task 8: Codegen — Quant Block Compilation with Monomorphization

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`
- Modify: `crates/nsl-codegen/src/expr.rs` (@ operator rewriting)

**Step 1: Add compile_quant_block in stmt.rs**

Find the existing `StmtKind::QuantBlock` handler and replace with:

```rust
StmtKind::QuantBlock(ref quant) => {
    self.compile_quant_block(builder, state, quant)?;
}
```

Then add the method:

```rust
fn compile_quant_block(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    quant: &nsl_ast::block::QuantBlock,
) -> Result<(), CodegenError> {
    // 1. Get the source model instance pointer
    let source_name = self.resolve_sym(quant.source).to_string();
    let source_ptr = state.variables.get(&quant.source)
        .ok_or_else(|| CodegenError::new(format!("quant source '{}' not found", source_name)))?;
    let source_val = builder.use_var(source_ptr.0);

    // 2. Determine dtype and granularity integer codes
    let dtype_code = match quant.default_dtype {
        Some(nsl_ast::block::QuantDtype::Int8) => 0i64,
        Some(nsl_ast::block::QuantDtype::Int4) => 1i64,
        None => 0i64, // default INT8
    };
    let (gran_code, axis_val, gs_val) = match &quant.default_granularity {
        Some(nsl_ast::block::QuantGranularity::PerTensor) => (0i64, 0i64, 0i64),
        Some(nsl_ast::block::QuantGranularity::PerChannel(axis)) => (1i64, *axis, 0i64),
        Some(nsl_ast::block::QuantGranularity::PerGroup(axis, gs)) => (2i64, *axis, *gs),
        None => (0i64, 0i64, 0i64), // default per-tensor
    };

    let dtype_v = builder.ins().iconst(cl_types::I64, dtype_code);
    let gran_v = builder.ins().iconst(cl_types::I64, gran_code);
    let axis_v = builder.ins().iconst(cl_types::I64, axis_val);
    let gs_v = builder.ins().iconst(cl_types::I64, gs_val);

    // 3. Resolve exclude patterns
    let excluded: Vec<String> = quant.exclude.clone();

    // 4. Allocate new model struct (same layout as source)
    // Get model type info from semantic pass
    let model_ty = self.node_types.get(&quant.name);
    let source_ty = self.node_types.get(&quant.source);

    // For each field in the source model:
    // - If excluded: copy pointer, bump refcount
    // - If tensor: call nsl_qtensor_quantize, store QuantizedTensor ptr
    // Use the model's field layout to get/set fields by offset

    // Allocate output model struct
    let out_model = self.compile_model_alloc(builder, state, quant.name)?;

    // Iterate over fields
    let fields = self.get_model_fields(quant.source)?;
    for (i, (field_name, _field_ty)) in fields.iter().enumerate() {
        let fname = self.resolve_sym(*field_name).to_string();
        let is_excluded = excluded.iter().any(|pat| glob_match_codegen(pat, &fname));

        // Load field from source model
        let field_val = self.compile_model_field_load(builder, source_val, i)?;

        if is_excluded {
            // Copy as-is, bump refcount
            self.compile_call_by_name(builder, "nsl_tensor_addref", &[field_val])?;
            self.compile_model_field_store(builder, out_model, i, field_val)?;
        } else {
            // Quantize the tensor
            let qt_val = self.compile_call_by_name(
                builder,
                "nsl_qtensor_quantize",
                &[field_val, dtype_v, gran_v, axis_v, gs_v],
            )?;
            self.compile_model_field_store(builder, out_model, i, qt_val)?;
        }
    }

    // 5. Bind output variable
    let out_name = quant.name;
    let var = state.new_variable();
    builder.declare_var(var, cl_types::I64);
    builder.def_var(var, out_model);
    state.variables.insert(out_name, (var, cl_types::I64));

    Ok(())
}
```

Note: The exact model field access pattern (load/store by offset) depends on how models are laid out in memory. Check the existing `compile_model_def` and field access patterns to match. The field load/store helpers may need to be adapted based on the actual struct layout used by the compiler.

**Step 2: Rewrite `@` operator for QuantizedTensor in expr.rs**

In the matmul/`@` operator codegen path in `expr.rs`, add a check:

```rust
// In the BinOp::MatMul handler:
// Check if right operand is QuantizedTensor (from semantic type info)
let rhs_type = self.node_type(rhs.id).clone();
if matches!(rhs_type, Type::QuantizedTensor) {
    // Mixed-precision matmul
    return self.compile_call_by_name(builder, "nsl_qtensor_matmul_mixed", &[lhs_val, rhs_val]);
}
// Otherwise fall through to normal matmul
```

**Step 3: Build and verify**

Run: `cargo build`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m16): add quant block codegen with model monomorphization and @ rewriting"
```

---

## Task 9: NSL Standard Library Wrappers

**Files:**
- Create: `stdlib/nsl/quant/ops.nsl`

**Step 1: Create stdlib quantization wrappers**

Create `stdlib/nsl/quant/ops.nsl`:

```nsl
# NSL Quantization stdlib wrappers
# String-to-enum mapping before crossing FFI boundary

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

**Step 2: Commit**

```bash
git add stdlib/nsl/quant/ops.nsl
git commit -m "feat(m16): add nsl.quant stdlib wrappers with string-to-enum mapping"
```

---

## Task 10: Test Suite

**Files:**
- Create: `tests/m16_test.nsl`

**Step 1: Write quantization tests**

Create `tests/m16_test.nsl`:

```nsl
# M16 Quantization Tests

@test
fn test_int8_per_tensor_roundtrip():
    let t = randn([4, 8])
    let qt = nsl_qtensor_quantize(t, 0, 0, 0, 0)  # int8, per_tensor
    let deq = nsl_qtensor_dequantize(qt)
    # Dequantized should be close to original (INT8 has 256 levels)
    assert_close(t, deq, 0.1, 0.1)

@test
fn test_int8_per_channel_roundtrip():
    let t = randn([4, 8])
    let qt = nsl_qtensor_quantize(t, 0, 1, 0, 0)  # int8, per_channel, axis=0
    let deq = nsl_qtensor_dequantize(qt)
    # Per-channel should be more accurate than per-tensor
    assert_close(t, deq, 0.05, 0.05)

@test
fn test_int4_per_group_roundtrip():
    let t = randn([8, 16])
    let qt = nsl_qtensor_quantize(t, 1, 2, 0, 4)  # int4, per_group, axis=0, gs=4
    let deq = nsl_qtensor_dequantize(qt)
    # INT4 has only 16 levels, so tolerance is wider
    assert_close(t, deq, 0.5, 0.5)

@test
fn test_int4_packing_exact():
    # Create a small tensor with known values to test bit packing
    let t = full([4], 0.5)
    let qt = nsl_qtensor_quantize(t, 1, 0, 0, 0)  # int4, per_tensor
    let deq = nsl_qtensor_dequantize(qt)
    # All values are the same, so quantize->dequantize should be very close
    assert_close(t, deq, 0.2, 0.2)

@test
fn test_zero_variance_no_nan():
    # All identical values — tests the epsilon clamp on scale computation
    let t = full([8, 8], 3.0)
    let qt = nsl_qtensor_quantize(t, 0, 0, 0, 0)  # int8, per_tensor
    let deq = nsl_qtensor_dequantize(qt)
    # Should not produce NaN
    assert_close(t, deq, 0.01, 0.01)

@test
fn test_mixed_matmul_correctness():
    let x = randn([4, 8])
    let w = randn([8, 16])
    let expected = x @ w
    let qw = nsl_qtensor_quantize(w, 0, 1, 0, 0)  # int8, per_channel
    let result = nsl_qtensor_matmul_mixed(x, qw)
    # Mixed matmul should approximate normal matmul
    assert_close(expected, result, 1.0, 0.1)

@test
fn test_qtensor_dtype_query():
    let t = randn([4, 4])
    let qt8 = nsl_qtensor_quantize(t, 0, 0, 0, 0)  # int8
    let qt4 = nsl_qtensor_quantize(t, 1, 0, 0, 0)  # int4
    assert_eq(nsl_qtensor_dtype(qt8), 0)
    assert_eq(nsl_qtensor_dtype(qt4), 1)

@test
fn test_qtensor_shape_query():
    let t = randn([3, 5])
    let qt = nsl_qtensor_quantize(t, 0, 0, 0, 0)
    let s = nsl_qtensor_shape(qt)
    assert_close(s, tensor([3.0, 5.0]), 0.001, 0.001)
```

**Step 2: Run tests**

Run: `cargo run -p nsl-cli -- test tests/m16_test.nsl`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/m16_test.nsl
git commit -m "feat(m16): add quantization test suite (8 tests)"
```

---

## Task 11: Integration Test — Quantize a Trained Model

**Files:**
- Create: `examples/m16_quantize.nsl`

**Step 1: Write end-to-end quantization example**

Create `examples/m16_quantize.nsl`:

```nsl
import nsl.nn.layers: Linear

model TinyModel(dim: int):
    fc1: Linear = Linear(dim, 64)
    fc2: Linear = Linear(64, dim)

    fn forward(self, x: Tensor) -> Tensor:
        let h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)

# Create and "train" a model (just use random weights for demo)
let m = TinyModel(32)

# Test forward pass with full precision
let x = randn([4, 32])
let y_full = m.forward(x)
print(y_full)

# Quantize to INT4 with per-group(32)
quant static qm from m:
    default: int4, per_group(32)
    exclude: ["*.b"]

# Test forward pass with quantized model
let y_quant = qm.forward(x)
print(y_quant)

# Compare outputs
print("Full precision and quantized outputs should be similar")
```

**Step 2: Run the example**

Run: `cargo run -p nsl-cli -- run examples/m16_quantize.nsl`
Expected: Prints both tensors, quantized output should be close to full precision.

**Step 3: Commit**

```bash
git add examples/m16_quantize.nsl
git commit -m "feat(m16): add end-to-end quantization integration example"
```

---

## Task 12: Update README and Finalize

**Files:**
- Modify: `README.md`

**Step 1: Update README with M16 features**

Add a Quantization section to the README:

```markdown
### Quantization (M16)

NeuralScript supports weight quantization with the `quant` block:

```nsl
quant static qm from trained_model:
    default: int4, per_group(128)
    exclude: ["*.bias"]
```

Features:
- **QuantizedTensor type** with packed INT4 (2 values/byte) and INT8 storage
- **Per-tensor, per-channel, and per-group** granularity
- **Weight-only RTN** (Round-To-Nearest) quantization
- **Mixed-precision matmul** — `NslTensor @ QuantizedTensor` just works
- **Model monomorphization** — compiler synthesizes quantized model types automatically
- **Glob-based exclude** — keep specific layers in full precision
```

Update the milestone status table to show M16 as complete.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for M16 quantization completion"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | QuantizedTensor struct + alloc/free | `quantize.rs`, `lib.rs` |
| 2 | Quantize/dequantize runtime (INT4/INT8, per-tensor/channel/group) | `quantize.rs` |
| 3 | Mixed-precision matmul | `quantize.rs` |
| 4 | Register FFI functions in codegen | `builtins.rs`, `expr.rs` |
| 5 | Register builtins in semantic + QuantizedTensor type | `builtins.rs`, `types.rs` |
| 6 | Extend quant block parser | `block.rs` (ast + parser) |
| 7 | Semantic pass — quant block validation | `checker.rs` |
| 8 | Codegen — quant block + monomorphization + @ rewriting | `stmt.rs`, `expr.rs` |
| 9 | NSL stdlib wrappers | `stdlib/nsl/quant/ops.nsl` |
| 10 | Test suite (8 tests) | `tests/m16_test.nsl` |
| 11 | Integration example | `examples/m16_quantize.nsl` |
| 12 | README update | `README.md` |
