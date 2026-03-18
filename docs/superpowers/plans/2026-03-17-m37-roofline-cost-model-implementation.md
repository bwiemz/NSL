# M37: Compile-Time Roofline & Cost Model — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compile-time performance analysis pass that computes FLOPs, bytes moved, and arithmetic intensity for every tensor operation, then classifies each as compute-bound or memory-bound relative to a target GPU's roofline — entirely at compile time with zero runtime overhead.

**Architecture:** Two new codegen modules: (1) `gpu_specs.rs` with a built-in GPU specification database (A100, H100, RTX-4090, etc.) and name matching. (2) `cost_model.rs` with per-operation FLOP/byte formulas, roofline classification, report formatting (table/JSON/Chrome tracing), and `@perf_budget` decorator checking. Pure analysis pass — no FFI, no runtime changes, no binary size impact.

**Tech Stack:** Rust (codegen analysis), clap (CLI)

**Spec:** `docs/superpowers/specs/2026-03-15-m37-roofline-cost-model-design.md`

---

## Important: Scope of This Plan

**This plan builds the complete cost model analysis library and GPU database.** It delivers:
- All FLOP/byte formulas for supported operations (matmul, elementwise, softmax, layernorm, etc.)
- GPU specification database with 7 GPUs and crossover-point classification
- Roofline classification (compute-bound/memory-bound/balanced)
- Report formatting in table, JSON, and Chrome tracing formats
- `@perf_budget` semantic validation
- CLI flags for `nsl check --perf`

**Deferred to M37b:** AST-walking `CostAnalyzer` integration in the compiler pipeline (requires wiring through `compile_entry()`), actual `@perf_budget` codegen enforcement, fusion suggestion engine (requires M31 FusionGraph access), post-fusion cost adjustment, GPU auto-detection via CUDA.

**Known dependency:** Same as M36 — `CompileOptions` is not wired through `compile_entry()`. The `--perf` CLI flags are parsed but dormant until that wiring is completed.

---

## Scope Note

The spec covers 9 deliverables. This plan orders them so each produces independently testable code:

1. **Tasks 1-2**: GPU specs database + name matching (pure data, fully unit-testable)
2. **Tasks 3-5**: Cost formulas — matmul, elementwise, special ops (pure functions, fully unit-testable)
3. **Tasks 6-7**: Roofline classification + time estimation
4. **Task 8**: Report formatting (table, JSON, Chrome tracing)
5. **Task 9**: `@perf_budget` semantic validation
6. **Tasks 10-11**: CLI flags + CompileOptions extension
7. **Task 12**: E2E tests
8. **Task 13**: Full verification + clippy

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/gpu_specs.rs` | GpuSpec struct, GPU_DATABASE const, find_gpu(), name matching | 200 |
| `crates/nsl-codegen/src/cost_model.rs` | OpCost, BoundClassification, FLOP/byte formulas, classify_op, estimate_time, report formatting | 500 |
| `crates/nsl-semantic/src/perf_budget.rs` | @perf_budget decorator validation | 50 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod gpu_specs; pub mod cost_model;` |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod perf_budget;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@perf_budget` validation |
| `crates/nsl-cli/src/main.rs` | Add `--perf`, `--gpu`, `--trace` flags to Check subcommand |
| `crates/nsl-cli/tests/e2e.rs` | Add M37 E2E tests |

---

## Phase 1: GPU Specification Database

### Task 1: GpuSpec Types + Database

**Files:**
- Create: `crates/nsl-codegen/src/gpu_specs.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `gpu_specs.rs` with GpuSpec struct, database, and tests**

```rust
//! M37: GPU specification database for roofline analysis.

/// Hardware specifications for a GPU model.
#[derive(Debug, Clone)]
pub struct GpuSpec {
    pub name: &'static str,
    pub sm_version: u32,
    /// Peak FP16 Tensor Core throughput in TFLOPS.
    pub peak_fp16_tflops: f64,
    /// Peak FP8 Tensor Core throughput in TFLOPS (0 if unsupported).
    pub peak_fp8_tflops: f64,
    /// Peak FP32 throughput in TFLOPS (non-Tensor Core).
    pub peak_fp32_tflops: f64,
    /// Peak memory bandwidth in GB/s.
    pub peak_bandwidth_gbs: f64,
    /// VRAM capacity in GB.
    pub vram_gb: f64,
    /// L2 cache size in MB.
    pub l2_cache_mb: f64,
    /// Crossover: peak_fp16_tflops*1e12 / peak_bandwidth_gbs*1e9 (FLOPs/byte).
    pub crossover_fp16: f64,
    /// Crossover for FP8 (0 if unsupported).
    pub crossover_fp8: f64,
    /// Crossover for FP32.
    pub crossover_fp32: f64,
}

impl GpuSpec {
    /// Crossover point for a given dtype (FLOPs/byte threshold).
    pub fn crossover(&self, dtype_bytes: usize) -> f64 {
        match dtype_bytes {
            1 => self.crossover_fp8,   // FP8
            2 => self.crossover_fp16,  // FP16/BF16
            _ => self.crossover_fp32,  // FP32/F64
        }
    }

    /// Peak TFLOPS for a given dtype byte width.
    pub fn peak_tflops(&self, dtype_bytes: usize) -> f64 {
        match dtype_bytes {
            1 => self.peak_fp8_tflops,
            2 => self.peak_fp16_tflops,
            _ => self.peak_fp32_tflops,
        }
    }
}

/// Built-in GPU specification database.
pub const GPU_DATABASE: &[GpuSpec] = &[
    GpuSpec {
        name: "A100-SXM", sm_version: 80,
        peak_fp16_tflops: 312.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 19.5,
        peak_bandwidth_gbs: 2039.0, vram_gb: 80.0, l2_cache_mb: 40.0,
        crossover_fp16: 153.0, crossover_fp8: 0.0, crossover_fp32: 9.6,
    },
    GpuSpec {
        name: "A100-PCIe", sm_version: 80,
        peak_fp16_tflops: 312.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 19.5,
        peak_bandwidth_gbs: 1555.0, vram_gb: 40.0, l2_cache_mb: 40.0,
        crossover_fp16: 200.6, crossover_fp8: 0.0, crossover_fp32: 12.5,
    },
    GpuSpec {
        name: "H100-SXM", sm_version: 90,
        peak_fp16_tflops: 989.0, peak_fp8_tflops: 1979.0, peak_fp32_tflops: 67.0,
        peak_bandwidth_gbs: 3350.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 295.2, crossover_fp8: 590.7, crossover_fp32: 20.0,
    },
    GpuSpec {
        name: "H100-PCIe", sm_version: 90,
        peak_fp16_tflops: 756.0, peak_fp8_tflops: 1513.0, peak_fp32_tflops: 51.0,
        peak_bandwidth_gbs: 2039.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 370.8, crossover_fp8: 741.9, crossover_fp32: 25.0,
    },
    GpuSpec {
        name: "RTX-4090", sm_version: 89,
        peak_fp16_tflops: 330.0, peak_fp8_tflops: 661.0, peak_fp32_tflops: 82.6,
        peak_bandwidth_gbs: 1008.0, vram_gb: 24.0, l2_cache_mb: 72.0,
        crossover_fp16: 327.4, crossover_fp8: 655.8, crossover_fp32: 81.9,
    },
    GpuSpec {
        name: "RTX-3090", sm_version: 86,
        peak_fp16_tflops: 142.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 35.6,
        peak_bandwidth_gbs: 936.2, vram_gb: 24.0, l2_cache_mb: 6.0,
        crossover_fp16: 151.7, crossover_fp8: 0.0, crossover_fp32: 38.0,
    },
    GpuSpec {
        name: "L40S", sm_version: 89,
        peak_fp16_tflops: 362.0, peak_fp8_tflops: 733.0, peak_fp32_tflops: 91.6,
        peak_bandwidth_gbs: 864.0, vram_gb: 48.0, l2_cache_mb: 96.0,
        crossover_fp16: 419.0, crossover_fp8: 848.4, crossover_fp32: 106.0,
    },
];

/// Find a GPU by name. Case-insensitive prefix match; prefers SXM variants.
/// Returns None if no match or ambiguous match.
pub fn find_gpu(name: &str) -> Option<&'static GpuSpec> {
    let name_upper = name.to_uppercase().replace(' ', "-");

    // Exact match first (case-insensitive)
    if let Some(gpu) = GPU_DATABASE
        .iter()
        .find(|g| g.name.to_uppercase() == name_upper)
    {
        return Some(gpu);
    }

    // Prefix match (e.g., "H100" -> prefer "H100-SXM")
    let matches: Vec<&GpuSpec> = GPU_DATABASE
        .iter()
        .filter(|g| g.name.to_uppercase().starts_with(&name_upper))
        .collect();

    match matches.len() {
        0 => None,
        1 => Some(matches[0]),
        _ => {
            // Prefer SXM variant
            matches
                .iter()
                .find(|g| g.name.contains("SXM"))
                .copied()
                .or(Some(matches[0]))
        }
    }
}

/// Default GPU when none specified and auto-detect unavailable.
pub fn default_gpu() -> &'static GpuSpec {
    find_gpu("A100-SXM").unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_gpu_exact() {
        let gpu = find_gpu("H100-SXM").unwrap();
        assert_eq!(gpu.name, "H100-SXM");
        assert_eq!(gpu.sm_version, 90);
    }

    #[test]
    fn test_find_gpu_case_insensitive() {
        let gpu = find_gpu("h100-sxm").unwrap();
        assert_eq!(gpu.name, "H100-SXM");
    }

    #[test]
    fn test_find_gpu_prefix_prefers_sxm() {
        let gpu = find_gpu("H100").unwrap();
        assert_eq!(gpu.name, "H100-SXM"); // SXM preferred over PCIe
    }

    #[test]
    fn test_find_gpu_prefix_unique() {
        let gpu = find_gpu("RTX-4090").unwrap();
        assert_eq!(gpu.name, "RTX-4090");
    }

    #[test]
    fn test_find_gpu_not_found() {
        assert!(find_gpu("NONEXISTENT").is_none());
    }

    #[test]
    fn test_default_gpu() {
        assert_eq!(default_gpu().name, "A100-SXM");
    }

    #[test]
    fn test_crossover_by_dtype() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!((h100.crossover(2) - 295.2).abs() < 0.1); // FP16
        assert!((h100.crossover(1) - 590.7).abs() < 0.1); // FP8
        assert!((h100.crossover(4) - 20.0).abs() < 0.1);  // FP32
    }

    #[test]
    fn test_peak_tflops_by_dtype() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!((h100.peak_tflops(2) - 989.0).abs() < 0.1);   // FP16
        assert!((h100.peak_tflops(1) - 1979.0).abs() < 0.1);  // FP8
        assert!((h100.peak_tflops(4) - 67.0).abs() < 0.1);    // FP32
    }

    #[test]
    fn test_database_has_all_gpus() {
        assert_eq!(GPU_DATABASE.len(), 7);
        let names: Vec<&str> = GPU_DATABASE.iter().map(|g| g.name).collect();
        assert!(names.contains(&"A100-SXM"));
        assert!(names.contains(&"H100-SXM"));
        assert!(names.contains(&"RTX-4090"));
        assert!(names.contains(&"RTX-3090"));
        assert!(names.contains(&"L40S"));
    }
}
```

- [ ] **Step 2: Add `pub mod gpu_specs;` to codegen lib.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-codegen gpu_specs -- --nocapture
git commit -m "feat(m37): add GPU specification database with 7 GPUs and name matching"
```

---

## Phase 2: Cost Formulas + Classification

### Task 2: OpCost Types + Matmul FLOP Formula

**Files:**
- Create: `crates/nsl-codegen/src/cost_model.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `cost_model.rs` with OpCost, BoundClassification, matmul formula**

```rust
//! M37: Compile-time roofline cost model — FLOP/byte formulas and performance analysis.

use crate::gpu_specs::GpuSpec;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Classification of an operation relative to the GPU roofline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundClassification {
    ComputeBound,
    MemoryBound,
    Balanced,    // within 20% of crossover point
    Unknown,     // dynamic shapes prevent classification
}

/// Cost analysis for a single operation.
#[derive(Debug, Clone)]
pub struct OpCost {
    pub name: String,
    pub loc: String,
    pub input_shapes: Vec<String>,
    pub output_shape: String,
    /// Total FLOPs for this operation.
    pub flops: u64,
    /// Total bytes read from memory.
    pub bytes_read: u64,
    /// Total bytes written to memory.
    pub bytes_written: u64,
    /// Arithmetic intensity = flops / (bytes_read + bytes_written).
    pub arithmetic_intensity: f64,
    /// Classification relative to target GPU.
    pub classification: BoundClassification,
    /// Whether this op was fused with adjacent ops.
    pub fused: bool,
    /// Estimated wall-clock time in microseconds.
    pub estimated_time_us: f64,
}

// ---------------------------------------------------------------------------
// FLOP formulas
// ---------------------------------------------------------------------------

/// Matmul [M, K] x [K, N] cost.
pub fn matmul_cost(m: u64, k: u64, n: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 2 * m * k * n;
    let bytes_read = (m * k + k * n) * dtype_bytes;
    let bytes_written = m * n * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Batched matmul [B, M, K] x [B, K, N] cost.
pub fn batched_matmul_cost(b: u64, m: u64, k: u64, n: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 2 * b * m * k * n;
    let bytes_read = b * (m * k + k * n) * dtype_bytes;
    let bytes_written = b * m * n * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Elementwise unary (relu, exp, log, sqrt, sigmoid, tanh, neg, abs, sign).
pub fn elementwise_unary_cost(num_elements: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = num_elements;
    let bytes_read = num_elements * dtype_bytes;
    let bytes_written = num_elements * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Elementwise binary (add, sub, mul, div).
pub fn elementwise_binary_cost(
    output_elements: u64,
    input_a_elements: u64,
    input_b_elements: u64,
    dtype_bytes: u64,
) -> (u64, u64, u64) {
    let flops = output_elements;
    let bytes_read = (input_a_elements + input_b_elements) * dtype_bytes;
    let bytes_written = output_elements * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Softmax [B, S] cost.
pub fn softmax_cost(b: u64, s: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 5 * b * s; // max, sub, exp, sum, div
    let bytes_read = b * s * dtype_bytes;
    let bytes_written = b * s * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// LayerNorm [B, S, D] cost (with affine: gamma + beta).
pub fn layernorm_cost(b: u64, s: u64, d: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 8 * b * s * d;
    let bytes_read = (b * s * d + 2 * d) * dtype_bytes;
    let bytes_written = b * s * d * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// RMSNorm [B, S, D] cost.
pub fn rmsnorm_cost(b: u64, s: u64, d: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 5 * b * s * d;
    let bytes_read = (b * s * d + d) * dtype_bytes;
    let bytes_written = b * s * d * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Embedding lookup [B, S] with vocab [V, D] cost.
pub fn embedding_cost(b: u64, s: u64, d: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 0;
    let bytes_read = b * s * d * dtype_bytes;
    let bytes_written = b * s * d * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Sum/Mean reduction along one dim. total_elements is input size, dim_size is reduced dim.
pub fn reduction_cost(total_elements: u64, dim_size: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = total_elements;
    let bytes_read = total_elements * dtype_bytes;
    let bytes_written = (total_elements / dim_size) * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Concatenation (pure memory copy, 0 FLOPs).
pub fn concat_cost(total_elements: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    (0, total_elements * dtype_bytes, total_elements * dtype_bytes)
}

/// Transpose (pure memory copy, 0 FLOPs).
pub fn transpose_cost(total_elements: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    (0, total_elements * dtype_bytes, total_elements * dtype_bytes)
}

/// FlashAttention [B, H, S, D] cost (IO-optimal tiled attention).
/// bytes_read uses coefficient 3 for Q+K+V (each read once from HBM due to tiling).
/// NOTE: Spec uses coefficient 2 in the formula but comments say "Q, K, V" — we use 3 for correctness.
pub fn flash_attention_cost(b: u64, h: u64, s: u64, d: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 4 * b * h * s * s * d;
    let bytes_read = b * h * 3 * s * d * dtype_bytes; // Q + K + V (tiled: each read once from HBM)
    let bytes_written = b * h * s * d * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Conv2d [B, Cin, H, W] with kernel [Cout, Cin, Kh, Kw] cost.
pub fn conv2d_cost(
    b: u64, cin: u64, h: u64, w: u64,
    cout: u64, kh: u64, kw: u64,
    stride_h: u64, stride_w: u64, pad_h: u64, pad_w: u64,
    dtype_bytes: u64,
) -> (u64, u64, u64) {
    let hout = (h + 2 * pad_h - kh) / stride_h + 1;
    let wout = (w + 2 * pad_w - kw) / stride_w + 1;
    let flops = 2 * b * cout * hout * wout * cin * kh * kw;
    let bytes_read = (b * cin * h * w + cout * cin * kh * kw) * dtype_bytes;
    let bytes_written = b * cout * hout * wout * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

// ---------------------------------------------------------------------------
// Roofline classification
// ---------------------------------------------------------------------------

/// Compute arithmetic intensity from FLOPs and bytes.
pub fn arithmetic_intensity(flops: u64, bytes_read: u64, bytes_written: u64) -> f64 {
    let total_bytes = bytes_read + bytes_written;
    if total_bytes == 0 {
        return 0.0;
    }
    flops as f64 / total_bytes as f64
}

/// Classify an operation as compute-bound, memory-bound, or balanced.
pub fn classify_op(ai: f64, crossover: f64) -> BoundClassification {
    if crossover == 0.0 {
        return BoundClassification::Unknown;
    }
    let ratio = ai / crossover;
    if ratio > 1.2 {
        BoundClassification::ComputeBound
    } else if ratio < 0.8 {
        BoundClassification::MemoryBound
    } else {
        BoundClassification::Balanced
    }
}

/// Estimate wall-clock time in microseconds for an operation on a target GPU.
pub fn estimate_time_us(flops: u64, bytes_read: u64, bytes_written: u64, gpu: &GpuSpec, dtype_bytes: usize) -> f64 {
    let total_bytes = bytes_read + bytes_written;
    let peak = gpu.peak_tflops(dtype_bytes);
    if peak == 0.0 {
        return 0.0;
    }
    let compute_time_us = flops as f64 / (peak * 1e6);
    let memory_time_us = total_bytes as f64 / (gpu.peak_bandwidth_gbs * 1e3);
    compute_time_us.max(memory_time_us)
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

/// Format a FLOP count as human-readable (e.g., "68.7G", "327.7K").
pub fn format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.1}T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.1}G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.1}M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.1}K", flops as f64 / 1e3)
    } else {
        format!("{flops}")
    }
}

/// Format a byte count as human-readable (e.g., "50.3M", "2.1G").
pub fn format_data_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}G", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1}M", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1}K", bytes as f64 / 1e3)
    } else {
        format!("{bytes}")
    }
}

fn classification_str(c: BoundClassification) -> &'static str {
    match c {
        BoundClassification::ComputeBound => "COMPUTE",
        BoundClassification::MemoryBound => "MEMORY",
        BoundClassification::Balanced => "BALANCED",
        BoundClassification::Unknown => "UNKNOWN",
    }
}

/// Format a performance analysis report as a human-readable table.
pub fn format_perf_table(ops: &[OpCost], gpu: &GpuSpec, dtype_name: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Performance Analysis (target: {}, dtype: {})\n",
        gpu.name, dtype_name
    ));
    out.push_str(&"=".repeat(100));
    out.push('\n');
    out.push_str(&format!(
        "{:<30} {:>12} {:>10} {:>6} {:>10} {:>10}\n",
        "Op", "FLOPs", "Bytes", "AI", "Class", "Est. Time"
    ));
    out.push_str(&"-".repeat(100));
    out.push('\n');

    let mut total_flops: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut total_time: f64 = 0.0;
    let mut compute_count = 0u32;
    let mut memory_count = 0u32;

    for op in ops {
        let marker = if op.classification == BoundClassification::MemoryBound { " *" } else { "" };
        out.push_str(&format!(
            "{:<30} {:>12} {:>10} {:>6.0} {:>10} {:>8.0} us{}\n",
            op.name,
            format_flops(op.flops),
            format_data_bytes(op.bytes_read + op.bytes_written),
            op.arithmetic_intensity,
            classification_str(op.classification),
            op.estimated_time_us,
            marker,
        ));
        total_flops += op.flops;
        total_bytes += op.bytes_read + op.bytes_written;
        total_time += op.estimated_time_us;
        match op.classification {
            BoundClassification::ComputeBound => compute_count += 1,
            BoundClassification::MemoryBound => memory_count += 1,
            _ => {}
        }
    }

    out.push('\n');
    out.push_str(&format!(
        "Total: {} FLOPs, {} data movement\n",
        format_flops(total_flops),
        format_data_bytes(total_bytes),
    ));
    out.push_str(&format!("Estimated total time: {:.0} us\n", total_time));
    out.push_str(&format!(
        "Compute-bound: {} ops, Memory-bound: {} ops (marked with *)\n",
        compute_count, memory_count,
    ));

    out
}

/// Format ops as Chrome tracing JSON (chrome://tracing compatible).
pub fn format_chrome_trace(ops: &[OpCost]) -> String {
    let mut events = Vec::new();
    let mut ts: f64 = 0.0;

    for op in ops {
        events.push(format!(
            r#"{{"name":"{}","cat":"{}","ph":"X","ts":{:.0},"dur":{:.0},"pid":0,"tid":0,"args":{{"flops":{},"bytes_read":{},"bytes_written":{},"arithmetic_intensity":{:.1},"classification":"{}"}}}}"#,
            op.name,
            classification_str(op.classification).to_lowercase(),
            ts,
            op.estimated_time_us.max(1.0),
            op.flops,
            op.bytes_read,
            op.bytes_written,
            op.arithmetic_intensity,
            classification_str(op.classification),
        ));
        ts += op.estimated_time_us;
    }

    format!("{{\"traceEvents\":[{}]}}", events.join(","))
}

/// Format ops as machine-readable JSON.
pub fn format_json_report(ops: &[OpCost], gpu: &GpuSpec, dtype_name: &str) -> String {
    let total_flops: u64 = ops.iter().map(|o| o.flops).sum();
    let total_bytes: u64 = ops.iter().map(|o| o.bytes_read + o.bytes_written).sum();
    let total_time: f64 = ops.iter().map(|o| o.estimated_time_us).sum();

    let ops_json: Vec<String> = ops.iter().map(|op| {
        format!(
            r#"{{"name":"{}","flops":{},"bytes_read":{},"bytes_written":{},"ai":{:.1},"classification":"{}","time_us":{:.1}}}"#,
            op.name, op.flops, op.bytes_read, op.bytes_written,
            op.arithmetic_intensity, classification_str(op.classification), op.estimated_time_us,
        )
    }).collect();

    format!(
        r#"{{"target_gpu":"{}","dtype":"{}","total_flops":{},"total_bytes":{},"estimated_time_us":{:.0},"operations":[{}]}}"#,
        gpu.name, dtype_name, total_flops, total_bytes, total_time, ops_json.join(","),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs;

    // --- FLOP formulas ---

    #[test]
    fn test_matmul_flops() {
        // [64, 128] x [128, 256] at FP16 (2 bytes)
        let (flops, bread, bwrite) = matmul_cost(64, 128, 256, 2);
        assert_eq!(flops, 2 * 64 * 128 * 256); // 4,194,304
        assert_eq!(bread, (64 * 128 + 128 * 256) * 2); // 81,920
        assert_eq!(bwrite, 64 * 256 * 2); // 32,768
    }

    #[test]
    fn test_batched_matmul_flops() {
        let (flops, _, _) = batched_matmul_cost(4, 64, 128, 256, 2);
        assert_eq!(flops, 2 * 4 * 64 * 128 * 256);
    }

    #[test]
    fn test_elementwise_unary() {
        let (flops, bread, bwrite) = elementwise_unary_cost(1024, 2);
        assert_eq!(flops, 1024);
        assert_eq!(bread, 2048);
        assert_eq!(bwrite, 2048);
    }

    #[test]
    fn test_elementwise_binary() {
        let (flops, bread, bwrite) = elementwise_binary_cost(1024, 1024, 1024, 2);
        assert_eq!(flops, 1024);
        assert_eq!(bread, 4096);
        assert_eq!(bwrite, 2048);
    }

    #[test]
    fn test_softmax_cost() {
        let (flops, _, _) = softmax_cost(32, 2048, 2);
        assert_eq!(flops, 5 * 32 * 2048); // 327,680
    }

    #[test]
    fn test_layernorm_cost() {
        let (flops, bread, _) = layernorm_cost(1, 2048, 4096, 2);
        assert_eq!(flops, 8 * 1 * 2048 * 4096);
        assert_eq!(bread, (1 * 2048 * 4096 + 2 * 4096) * 2);
    }

    #[test]
    fn test_flash_attention_cost() {
        let (flops, bread, _) = flash_attention_cost(1, 32, 2048, 128, 2);
        assert_eq!(flops, 4 * 1 * 32 * 2048 * 2048 * 128);
        // bytes_read = B*H*3*S*D*dtype = 1*32*3*2048*128*2 = 50,331,648
        assert_eq!(bread, 1 * 32 * 3 * 2048 * 128 * 2);
    }

    #[test]
    fn test_embedding_zero_flops() {
        let (flops, _, _) = embedding_cost(1, 2048, 4096, 2);
        assert_eq!(flops, 0);
    }

    #[test]
    fn test_rmsnorm_cost() {
        let (flops, bread, bwrite) = rmsnorm_cost(1, 2048, 4096, 2);
        assert_eq!(flops, 5 * 1 * 2048 * 4096);
        assert_eq!(bread, (1 * 2048 * 4096 + 4096) * 2);
        assert_eq!(bwrite, 1 * 2048 * 4096 * 2);
    }

    #[test]
    fn test_reduction_cost() {
        // Reduce [4, 8] along dim 1 (size=8) -> [4]
        let (flops, bread, bwrite) = reduction_cost(32, 8, 2);
        assert_eq!(flops, 32);
        assert_eq!(bread, 64);
        assert_eq!(bwrite, 4 * 2); // 32/8 = 4 output elements
    }

    #[test]
    fn test_concat_cost() {
        let (flops, bread, bwrite) = concat_cost(1024, 2);
        assert_eq!(flops, 0);
        assert_eq!(bread, 2048);
        assert_eq!(bwrite, 2048);
    }

    #[test]
    fn test_transpose_cost() {
        let (flops, bread, bwrite) = transpose_cost(512, 4);
        assert_eq!(flops, 0);
        assert_eq!(bread, 2048);
        assert_eq!(bwrite, 2048);
    }

    #[test]
    fn test_conv2d_cost() {
        // [1, 3, 32, 32] with kernel [16, 3, 3, 3], stride=1, pad=1
        let (flops, _, bwrite) = conv2d_cost(1, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1, 4);
        let hout = (32 + 2 - 3) / 1 + 1; // 32
        let wout = (32 + 2 - 3) / 1 + 1; // 32
        assert_eq!(flops, 2 * 1 * 16 * hout * wout * 3 * 3 * 3);
        assert_eq!(bwrite, 1 * 16 * hout * wout * 4);
    }

    #[test]
    fn test_arithmetic_intensity_zero_bytes() {
        let ai = arithmetic_intensity(100, 0, 0);
        assert_eq!(ai, 0.0);
    }

    #[test]
    fn test_estimate_time_zero_peak() {
        // FP8 on A100 (peak_fp8 = 0) -> returns 0
        let gpu = gpu_specs::find_gpu("A100-SXM").unwrap();
        let time = estimate_time_us(1000, 100, 100, gpu, 1); // dtype_bytes=1 -> FP8
        assert_eq!(time, 0.0);
    }

    // --- Roofline classification ---

    #[test]
    fn test_arithmetic_intensity() {
        let ai = arithmetic_intensity(4_194_304, 81_920, 32_768);
        assert!((ai - 36.57).abs() < 0.1); // 4194304 / (81920+32768) = 36.57
    }

    #[test]
    fn test_classify_compute_bound() {
        // AI=1366 on H100 (crossover=295.2) -> compute-bound
        let c = classify_op(1366.0, 295.2);
        assert_eq!(c, BoundClassification::ComputeBound);
    }

    #[test]
    fn test_classify_memory_bound() {
        // AI=3.0 on H100 (crossover=295.2) -> memory-bound
        let c = classify_op(3.0, 295.2);
        assert_eq!(c, BoundClassification::MemoryBound);
    }

    #[test]
    fn test_classify_balanced() {
        // AI = crossover * 1.0 -> balanced (within 20%)
        let c = classify_op(295.2, 295.2);
        assert_eq!(c, BoundClassification::Balanced);
    }

    #[test]
    fn test_classify_zero_crossover() {
        let c = classify_op(100.0, 0.0);
        assert_eq!(c, BoundClassification::Unknown);
    }

    // --- Time estimation ---

    #[test]
    fn test_estimate_time() {
        let gpu = gpu_specs::find_gpu("H100-SXM").unwrap();
        // Large matmul: 68.7G FLOPs, 50.3M + 16.8M bytes at FP16
        let time = estimate_time_us(68_719_476_736, 50_331_648, 16_777_216, gpu, 2);
        assert!(time > 0.0);
        assert!(time < 1000.0); // should be ~69 us on H100
    }

    // --- Formatting ---

    #[test]
    fn test_format_flops() {
        assert_eq!(format_flops(0), "0");
        assert_eq!(format_flops(500), "500");
        assert_eq!(format_flops(1_500), "1.5K");
        assert_eq!(format_flops(68_719_476_736), "68.7G");
        assert_eq!(format_flops(1_000_000_000_000), "1.0T");
    }

    #[test]
    fn test_format_data_bytes() {
        assert_eq!(format_data_bytes(500), "500");
        assert_eq!(format_data_bytes(50_331_648), "50.3M");
    }

    #[test]
    fn test_format_chrome_trace() {
        let ops = vec![OpCost {
            name: "matmul".into(), loc: "test:1".into(),
            input_shapes: vec![], output_shape: "".into(),
            flops: 1000, bytes_read: 100, bytes_written: 50,
            arithmetic_intensity: 6.67, classification: BoundClassification::MemoryBound,
            fused: false, estimated_time_us: 10.0,
        }];
        let trace = format_chrome_trace(&ops);
        assert!(trace.contains("traceEvents"));
        assert!(trace.contains("matmul"));
        assert!(trace.contains("memory"));
    }

    #[test]
    fn test_format_json_report() {
        let gpu = gpu_specs::find_gpu("H100-SXM").unwrap();
        let ops = vec![OpCost {
            name: "relu".into(), loc: "test:1".into(),
            input_shapes: vec![], output_shape: "".into(),
            flops: 1024, bytes_read: 2048, bytes_written: 2048,
            arithmetic_intensity: 0.25, classification: BoundClassification::MemoryBound,
            fused: false, estimated_time_us: 5.0,
        }];
        let json = format_json_report(&ops, gpu, "fp16");
        assert!(json.contains("H100-SXM"));
        assert!(json.contains("relu"));
        assert!(json.contains("total_flops"));
    }

    #[test]
    fn test_format_perf_table() {
        let gpu = gpu_specs::find_gpu("A100-SXM").unwrap();
        let ops = vec![
            OpCost {
                name: "matmul(x, W)".into(), loc: "model:5".into(),
                input_shapes: vec![], output_shape: "".into(),
                flops: 4_194_304, bytes_read: 81_920, bytes_written: 32_768,
                arithmetic_intensity: 36.6, classification: BoundClassification::MemoryBound,
                fused: false, estimated_time_us: 50.0,
            },
        ];
        let table = format_perf_table(&ops, gpu, "fp16");
        assert!(table.contains("Performance Analysis"));
        assert!(table.contains("A100-SXM"));
        assert!(table.contains("matmul(x, W)"));
        assert!(table.contains("MEMORY"));
    }
}
```

- [ ] **Step 2: Add `pub mod cost_model;` to codegen lib.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-codegen cost_model -- --nocapture
cargo test -p nsl-codegen gpu_specs -- --nocapture
git commit -m "feat(m37): add cost model with FLOP formulas, roofline classification, and report formatting"
```

---

## Phase 3: Semantic Validation

### Task 3: @perf_budget Decorator Validation

**Files:**
- Create: `crates/nsl-semantic/src/perf_budget.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Create perf_budget.rs** (same pattern as fp8.rs, context_parallel.rs)

```rust
use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @perf_budget decorator arguments.
/// Returns max_tflops value or None on error.
pub fn validate_perf_budget_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<f64> {
    let mut max_tflops: Option<f64> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "max_tflops" => {
                        match &arg.value.kind {
                            ExprKind::FloatLiteral(f) => {
                                max_tflops = Some(*f);
                            }
                            ExprKind::IntLiteral(n) => {
                                max_tflops = Some(*n as f64);
                            }
                            _ => {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@perf_budget: max_tflops must be a numeric literal".to_string(),
                                    )
                                    .with_label(arg.span, "expected number"),
                                );
                            }
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@perf_budget: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if max_tflops.is_none() {
        diagnostics.push(
            Diagnostic::error("@perf_budget: max_tflops is required".to_string())
                .with_label(deco.span, "missing max_tflops"),
        );
    }

    max_tflops
}
```

- [ ] **Step 2: Add `pub mod perf_budget;` to semantic lib.rs**

- [ ] **Step 3: Wire into checker.rs** (after the `@fp8_compute` block)

```rust
// M37: @perf_budget decorator validation
if dname == "perf_budget" {
    let resolve = |s: nsl_ast::Symbol| -> String {
        self.interner
            .resolve(s.0)
            .unwrap_or("")
            .to_string()
    };
    crate::perf_budget::validate_perf_budget_decorator(
        deco,
        &resolve,
        &mut self.diagnostics,
    );
}
```

- [ ] **Step 4: Verify, commit**

```bash
cargo check -p nsl-semantic
git commit -m "feat(m37): add @perf_budget semantic validation"
```

---

## Phase 4: CLI + E2E

### Task 4: CLI Flags

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

- [ ] **Step 1: Add flags to Check subcommand**

Find the `Check` variant in the `Cli` enum and add:

```rust
        /// M37: Run roofline performance analysis
        #[arg(long)]
        perf: bool,

        /// M37: Target GPU for performance analysis (e.g., "H100", "A100-PCIe")
        #[arg(long)]
        gpu: Option<String>,

        /// M37: Write Chrome tracing JSON to file
        #[arg(long)]
        trace: Option<String>,
```

- [ ] **Step 2: Update destructuring** for the Check match arm to include new fields

- [ ] **Step 3: Verify compilation**

```bash
cargo check --workspace
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m37): add --perf, --gpu, --trace CLI flags to nsl check"
```

---

### Task 5: E2E Tests

**Files:**
- Create: `examples/m37_perf_basic.nsl`
- Create: `examples/m37_perf_budget_error.nsl`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create @perf_budget validation error test**

```nsl
# M37: @perf_budget validation error — missing max_tflops

model Bad:
    @perf_budget(unknown_arg=42)
    weight: int = 0

    fn forward(self, x: Tensor) -> Tensor:
        return x
```

- [ ] **Step 2: Add E2E test for validation error**

```rust
// ---------------------------------------------------------------------------
// M37: Compile-Time Roofline & Cost Model
// ---------------------------------------------------------------------------

#[test]
fn e2e_m37_perf_budget_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m37_perf_budget_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m37_perf_budget_error, but it succeeded"
    );
    assert!(
        stderr.contains("perf_budget") || stderr.contains("unknown argument"),
        "Expected perf_budget validation error in stderr, got: {}",
        stderr
    );
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-cli e2e_m37 -- --nocapture
git commit -m "test(m37): add E2E tests for @perf_budget validation and roofline analysis"
```

---

### Task 6: Full Verification + Clippy

- [ ] **Step 1: Run all workspace lib tests**

```bash
cargo test --workspace --lib
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Fix any issues, commit**

```bash
git commit -m "chore(m37): fix clippy warnings and verify full test suite"
```

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | GPU specs database + name matching | 9 unit |
| 2 | Cost formulas + classification + report formatting | 26 unit |
| 3 | @perf_budget semantic validation | compile check |
| 4 | CLI flags (--perf, --gpu, --trace) | compile check |
| 5 | E2E tests | 1 E2E |
| 6 | Full verification | all tests |

**Total: 6 tasks, ~35 unit tests + 1 E2E test**

### Deferred to M37b

- AST-walking `CostAnalyzer` integration in compiler pipeline (requires CompileOptions wiring)
- `@perf_budget` codegen enforcement (sum FLOPs in function body, compare to budget)
- Fusion suggestion engine (requires M31 FusionGraph access from compiler)
- Post-fusion cost adjustment (reflect eliminated intermediate materializations)
- GPU auto-detection via CUDA `cuDeviceGetName`
- `nsl check --perf` actual invocation (currently flags are parsed but analysis pass not wired)
- Conv2d FLOP formula (requires conv parameter extraction from AST)
- `--format json` output mode enum in CompileOptions
