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
    Balanced,
    Unknown,
}

/// Cost analysis for a single operation.
#[derive(Debug, Clone)]
pub struct OpCost {
    pub name: String,
    pub loc: String,
    pub input_shapes: Vec<String>,
    pub output_shape: String,
    pub flops: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub arithmetic_intensity: f64,
    pub classification: BoundClassification,
    pub fused: bool,
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
    (num_elements, num_elements * dtype_bytes, num_elements * dtype_bytes)
}

/// Elementwise binary (add, sub, mul, div).
pub fn elementwise_binary_cost(
    output_elements: u64,
    input_a_elements: u64,
    input_b_elements: u64,
    dtype_bytes: u64,
) -> (u64, u64, u64) {
    (
        output_elements,
        (input_a_elements + input_b_elements) * dtype_bytes,
        output_elements * dtype_bytes,
    )
}

/// Softmax [B, S] cost.
pub fn softmax_cost(b: u64, s: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 5 * b * s;
    let bytes = b * s * dtype_bytes;
    (flops, bytes, bytes)
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

/// Embedding lookup [B, S] with vocab [V, D] cost (0 FLOPs, pure memory).
pub fn embedding_cost(b: u64, s: u64, d: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let bytes = b * s * d * dtype_bytes;
    (0, bytes, bytes)
}

/// Sum/Mean reduction along one dim.
pub fn reduction_cost(total_elements: u64, dim_size: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    (
        total_elements,
        total_elements * dtype_bytes,
        (total_elements / dim_size) * dtype_bytes,
    )
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
/// NOTE: Spec uses coefficient 2, but Q+K+V = 3 tensors. We use 3 for correctness.
pub fn flash_attention_cost(b: u64, h: u64, s: u64, d: u64, dtype_bytes: u64) -> (u64, u64, u64) {
    let flops = 4 * b * h * s * s * d;
    let bytes_read = b * h * 3 * s * d * dtype_bytes;
    let bytes_written = b * h * s * d * dtype_bytes;
    (flops, bytes_read, bytes_written)
}

/// Conv2d [B, Cin, H, W] with kernel [Cout, Cin, Kh, Kw] cost.
#[allow(clippy::too_many_arguments)]
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
pub fn estimate_time_us(
    flops: u64,
    bytes_read: u64,
    bytes_written: u64,
    gpu: &GpuSpec,
    dtype_bytes: usize,
) -> f64 {
    let total_bytes = bytes_read + bytes_written;
    let peak = gpu.peak_tflops(dtype_bytes);
    // Fall back to FP32 throughput if this dtype is unsupported on the GPU
    let effective_peak = if peak == 0.0 { gpu.peak_fp32_tflops } else { peak };
    if effective_peak == 0.0 {
        return 0.0;
    }
    let compute_time_us = flops as f64 / (effective_peak * 1e6);
    let memory_time_us = total_bytes as f64 / (gpu.peak_bandwidth_gbs * 1e3);
    compute_time_us.max(memory_time_us)
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

/// Format a FLOP count as human-readable (e.g., "68.7G", "327.7K").
/// Uses SI units (1e9 for G, 1e6 for M) consistent with roofline convention.
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
/// Uses SI units (not binary) — consistent with bandwidth measurements (GB/s).
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
        let marker = if op.classification == BoundClassification::MemoryBound {
            " *"
        } else {
            ""
        };
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

    let ops_json: Vec<String> = ops
        .iter()
        .map(|op| {
            format!(
                r#"{{"name":"{}","flops":{},"bytes_read":{},"bytes_written":{},"ai":{:.1},"classification":"{}","time_us":{:.1}}}"#,
                op.name, op.flops, op.bytes_read, op.bytes_written,
                op.arithmetic_intensity, classification_str(op.classification), op.estimated_time_us,
            )
        })
        .collect();

    format!(
        r#"{{"target_gpu":"{}","dtype":"{}","total_flops":{},"total_bytes":{},"estimated_time_us":{:.0},"operations":[{}]}}"#,
        gpu.name, dtype_name, total_flops, total_bytes, total_time, ops_json.join(","),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs;

    #[test]
    fn test_matmul_flops() {
        let (flops, bread, bwrite) = matmul_cost(64, 128, 256, 2);
        assert_eq!(flops, 2 * 64 * 128 * 256);
        assert_eq!(bread, (64 * 128 + 128 * 256) * 2);
        assert_eq!(bwrite, 64 * 256 * 2);
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
        assert_eq!(flops, 5 * 32 * 2048);
    }

    #[test]
    fn test_layernorm_cost() {
        let (flops, bread, _) = layernorm_cost(1, 2048, 4096, 2);
        assert_eq!(flops, 8 * 2048 * 4096);
        assert_eq!(bread, (2048 * 4096 + 2 * 4096) * 2);
    }

    #[test]
    fn test_flash_attention_cost() {
        let (flops, bread, _) = flash_attention_cost(1, 32, 2048, 128, 2);
        assert_eq!(flops, 4 * 32 * 2048 * 2048 * 128);
        assert_eq!(bread, 32 * 3 * 2048 * 128 * 2);
    }

    #[test]
    fn test_embedding_zero_flops() {
        let (flops, _, _) = embedding_cost(1, 2048, 4096, 2);
        assert_eq!(flops, 0);
    }

    #[test]
    fn test_rmsnorm_cost() {
        let (flops, bread, bwrite) = rmsnorm_cost(1, 2048, 4096, 2);
        assert_eq!(flops, 5 * 2048 * 4096);
        assert_eq!(bread, (2048 * 4096 + 4096) * 2);
        assert_eq!(bwrite, 2048 * 4096 * 2);
    }

    #[test]
    fn test_reduction_cost() {
        let (flops, bread, bwrite) = reduction_cost(32, 8, 2);
        assert_eq!(flops, 32);
        assert_eq!(bread, 64);
        assert_eq!(bwrite, 4 * 2);
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
        let (flops, _, bwrite) = conv2d_cost(1, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1, 4);
        let hout = (32 + 2 - 3) + 1;
        let wout = (32 + 2 - 3) + 1;
        assert_eq!(flops, 2 * 16 * hout * wout * 3 * 3 * 3);
        assert_eq!(bwrite, 16 * hout * wout * 4);
    }

    #[test]
    fn test_arithmetic_intensity_zero_bytes() {
        assert_eq!(arithmetic_intensity(100, 0, 0), 0.0);
    }

    #[test]
    fn test_estimate_time_fp8_fallback_to_fp32() {
        // FP8 on A100 (peak_fp8 = 0) falls back to FP32 throughput (19.5 TFLOPS)
        let gpu = gpu_specs::find_gpu("A100-SXM").unwrap();
        let time = estimate_time_us(1000, 100, 100, gpu, 1);
        assert!(time > 0.0); // not zero — uses FP32 fallback
    }

    #[test]
    fn test_arithmetic_intensity() {
        let ai = arithmetic_intensity(4_194_304, 81_920, 32_768);
        // 4194304 / (81920+32768) = 36.57 (NOTE: spec Section 9 test claims 51.2
        // but that incorrectly omits bytes_written — our formula is correct per OpCost definition)
        assert!((ai - 36.57).abs() < 0.1);
    }

    #[test]
    fn test_classify_compute_bound() {
        assert_eq!(classify_op(1366.0, 295.2), BoundClassification::ComputeBound);
    }

    #[test]
    fn test_classify_memory_bound() {
        assert_eq!(classify_op(3.0, 295.2), BoundClassification::MemoryBound);
    }

    #[test]
    fn test_classify_balanced() {
        assert_eq!(classify_op(295.2, 295.2), BoundClassification::Balanced);
    }

    #[test]
    fn test_classify_zero_crossover() {
        assert_eq!(classify_op(100.0, 0.0), BoundClassification::Unknown);
    }

    #[test]
    fn test_estimate_time() {
        let gpu = gpu_specs::find_gpu("H100-SXM").unwrap();
        let time = estimate_time_us(68_719_476_736, 50_331_648, 16_777_216, gpu, 2);
        assert!(time > 0.0);
        assert!(time < 1000.0);
    }

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
        let ops = vec![OpCost {
            name: "matmul(x, W)".into(), loc: "model:5".into(),
            input_shapes: vec![], output_shape: "".into(),
            flops: 4_194_304, bytes_read: 81_920, bytes_written: 32_768,
            arithmetic_intensity: 36.6, classification: BoundClassification::MemoryBound,
            fused: false, estimated_time_us: 50.0,
        }];
        let table = format_perf_table(&ops, gpu, "fp16");
        assert!(table.contains("Performance Analysis"));
        assert!(table.contains("A100-SXM"));
        assert!(table.contains("matmul(x, W)"));
        assert!(table.contains("MEMORY"));
    }
}
