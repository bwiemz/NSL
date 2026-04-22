//! WRGA B.3.2 per-op profiling scope — decide whether the fused-backward
//! kernel milestone should proceed or be reframed as an allocator/tape
//! optimisation milestone.
//!
//! Post-cuBLAS-swap re-measurement (see `project_wrga_b32_measurement.md`
//! addendum 2026-04-21) showed backward wall-time went 188s→216s despite
//! cuBLAS matmul delivering ~400× speedup on the primitive. That rules
//! out "matmul-dominated"; but the actual decomposition (how much of
//! backward is matmul vs elementwise vs allocator/tape overhead) is
//! unknown. This bench answers that question by:
//!
//!   1. Running the prescribed-shape GatedLoRA train step under
//!      `NSL_PROFILE_KERNELS=1` so every kernel launch gets cuEvent-pair
//!      timed.
//!   2. Measuring end-to-end wall-clock via `Instant::elapsed()` around
//!      the `nsl run` subprocess.
//!   3. Parsing `kernel_profile.json` + bucketing each launch into a
//!      semantic category (matmul / attention_fused / adapter_fused /
//!      elementwise_arith / elementwise_activation / reduction /
//!      normalization / indexing / copy_layout / dropout_dequant /
//!      sparse / moe / other).
//!   4. Partitioning by iteration via the fused-GatedLoRA marker (same
//!      pattern as `wrga_gatedlora_backward_trigger.rs`) and reporting
//!      median per-category GPU time across the timed iters.
//!   5. Computing `host_overhead = wall_clock - gpu_total`. That is the
//!      time spent outside kernel launches — allocator, tape traversal,
//!      source-AD machinery, CPU/GPU memcpy, CUDA API overhead.
//!
//! The categorizer is a pure function (`categorize_kernel`). Unit tests
//! exercise it against synthetic fixtures so CI validates the
//! classification even on machines without CUDA. The actual bench is
//! `#[ignore]`-gated and `#[cfg(feature = "cuda")]`.
//!
//! Invoke with:
//!   cargo test --features cuda --test wrga_b32_per_op_breakdown \
//!       -- --ignored --nocapture

use serde_json::Value;
use std::path::Path;

// ─────────────────────────── Categorizer ───────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// cuBLAS sgemm + naive matmul PTX. The category fused-backward
    /// would nominally target — if this is <30% of backward GPU time,
    /// B.3.2's ceiling speedup is bounded and the milestone should be
    /// reframed.
    Matmul,
    /// Fused FA-2 forward / backward kernels.
    AttentionFused,
    /// Fused WRGA LoRA / IA³ / GatedLoRA adapter kernels.
    AdapterFused,
    /// Pointwise arithmetic: add, sub, mul, div, neg, abs, sign +
    /// scalar variants. Candidate for epilogue fusion but small unless
    /// very numerous.
    ElementwiseArith,
    /// Pointwise nonlinearity: relu, sigmoid, tanh, gelu, silu, exp,
    /// log, sqrt, cos, sin, rotate_half, clamp, + *_backward_f32.
    ElementwiseActivation,
    /// Reductions: global_sum, sum_dim, max_dim, tensor_stats.
    Reduction,
    /// Norms + softmaxes: rmsnorm, layernorm, softmax, log_softmax.
    Normalization,
    /// Indexing ops: gather, scatter_add, embedding, slice.
    Indexing,
    /// Memory-shuffling: strided_copy, bias_add (shape-broadcast), transpose-like.
    CopyLayout,
    /// Stochastic / numeric-type ops: dropout, dequant_*.
    DropoutDequant,
    /// Sparse kernels: csr/coo/bsr spmm + spmv.
    Sparse,
    /// MoE: expert_batched_gemm, moe_*.
    Moe,
    /// Anything unclassified — an alert that the categorizer needs
    /// extending if its share is non-trivial.
    Other,
}

impl OpCategory {
    pub fn all() -> &'static [OpCategory] {
        use OpCategory::*;
        &[
            Matmul, AttentionFused, AdapterFused,
            ElementwiseArith, ElementwiseActivation,
            Reduction, Normalization, Indexing, CopyLayout,
            DropoutDequant, Sparse, Moe, Other,
        ]
    }

    pub fn label(self) -> &'static str {
        use OpCategory::*;
        match self {
            Matmul                => "matmul",
            AttentionFused        => "attention_fused",
            AdapterFused          => "adapter_fused",
            ElementwiseArith      => "elementwise_arith",
            ElementwiseActivation => "elementwise_activation",
            Reduction             => "reduction",
            Normalization         => "normalization",
            Indexing              => "indexing",
            CopyLayout            => "copy_layout",
            DropoutDequant        => "dropout_dequant",
            Sparse                => "sparse",
            Moe                   => "moe",
            Other                 => "other",
        }
    }
}

/// Classify a kernel by name. Order of checks matters — fused-adapter
/// and fused-attention prefixes must be checked BEFORE generic
/// elementwise matches (e.g. the adapter name contains "_mul_"-ish
/// infixes).
pub fn categorize_kernel(name: &str) -> OpCategory {
    // Fused adapter (most specific, highest-level fusion target).
    if name.starts_with("nsl_wrga_fused_") {
        return OpCategory::AdapterFused;
    }
    // Fused attention.
    if name.starts_with("flash_attn_") {
        return OpCategory::AttentionFused;
    }
    // MoE (checked BEFORE matmul so `expert_batched_gemm` routes to
    // MoE rather than being swallowed by the generic `_gemm` probe).
    if name.starts_with("moe_") || name == "expert_batched_gemm" {
        return OpCategory::Moe;
    }
    // Matmul: cuBLAS synthetic marker + naive PTX matmul + batched
    // matmul + fp8 matmul kernels.
    if name == "sgemm_cublas"
        || name.contains("matmul")
        || name == "nsl_bmm_f32"
        || name.contains("_gemm")
        || name == "nsl_fp8_matmul_wgmma_kernel"
    {
        return OpCategory::Matmul;
    }
    // Sparse.
    if name.contains("_spmm_") || name.contains("_spmv_") {
        return OpCategory::Sparse;
    }
    // Reductions.
    if name == "nsl_global_sum_f32"
        || name == "nsl_det_global_sum_f32"
        || name == "nsl_sum_dim_f32"
        || name == "nsl_det_sum_dim_f32"
        || name == "nsl_max_dim_f32"
        || name == "nsl_tensor_stats_f32"
    {
        return OpCategory::Reduction;
    }
    // Normalization + softmaxes.
    if name == "nsl_rmsnorm_f32"
        || name == "nsl_layernorm_f32"
        || name == "nsl_softmax_f32"
        || name == "nsl_log_softmax_f32"
    {
        return OpCategory::Normalization;
    }
    // Indexing / gather-scatter / embedding.
    if name.starts_with("nsl_gather")
        || name.starts_with("nsl_scatter")
        || name.starts_with("nsl_det_scatter")
        || name.starts_with("nsl_embedding")
        || name == "nsl_slice_f32"
    {
        return OpCategory::Indexing;
    }
    // Copy/layout.
    if name == "nsl_strided_copy_f32"
        || name == "nsl_bias_add_f32"
        || name == "nsl_rope_cache_write"
    {
        return OpCategory::CopyLayout;
    }
    // Dropout + dequantization (training-path stochastic + mixed-precision loading).
    if name == "nsl_dropout_f32" || name.starts_with("nsl_dequant_") {
        return OpCategory::DropoutDequant;
    }
    // Pointwise activations (before elementwise-arith so backward
    // variants like `nsl_sigmoid_backward_f32` don't route to arith).
    if matches!(
        name,
        "nsl_relu_f32"    | "nsl_relu_backward_f32"    |
        "nsl_sigmoid_f32" | "nsl_sigmoid_backward_f32" |
        "nsl_tanh_f32"    | "nsl_tanh_backward_f32"    |
        "nsl_gelu_f32"    | "nsl_gelu_backward_f32"    |
        "nsl_silu_f32"    | "nsl_silu_backward_f32"    |
        "nsl_exp_f32"     | "nsl_log_f32"              |
        "nsl_sqrt_f32"    | "nsl_cos_f32"              | "nsl_sin_f32" |
        "nsl_rotate_half_f32" |
        "nsl_clamp_f32"   | "nsl_clamp_backward_f32"
    ) {
        return OpCategory::ElementwiseActivation;
    }
    // Pointwise arithmetic.
    if matches!(
        name,
        "nsl_add_f32" | "nsl_sub_f32" | "nsl_mul_f32" | "nsl_div_f32" |
        "nsl_neg_f32" | "nsl_abs_f32" | "nsl_sign_f32" |
        "nsl_add_scalar_f32" | "nsl_mul_scalar_f32"
    ) {
        return OpCategory::ElementwiseArith;
    }
    OpCategory::Other
}

// ─────────────────────────── Trace parsing ───────────────────────────

/// One recorded kernel launch: (name, duration_us, timestamp_us).
#[derive(Clone, Debug)]
pub struct TraceEvent {
    pub name: String,
    pub dur_us: f64,
    pub ts_us: f64,
}

#[allow(dead_code)] // used by ignored bench below + unit tests of partition_by_marker
pub fn parse_trace_from_json(text: &str) -> Result<Vec<TraceEvent>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let events = v
        .get("traceEvents")
        .and_then(|e| e.as_array())
        .ok_or_else(|| "missing traceEvents".to_string())?;
    let mut out: Vec<TraceEvent> = Vec::with_capacity(events.len());
    for ev in events {
        let name = ev.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
        let dur = ev.get("dur").and_then(|d| d.as_f64()).unwrap_or(0.0);
        let ts  = ev.get("ts" ).and_then(|t| t.as_f64()).unwrap_or(0.0);
        out.push(TraceEvent { name, dur_us: dur, ts_us: ts });
    }
    out.sort_by(|a, b| a.ts_us.partial_cmp(&b.ts_us).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

#[allow(dead_code)]
pub fn parse_trace(path: &Path) -> Result<Vec<TraceEvent>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_trace_from_json(&text)
}

/// Partition events into per-iteration groups by identifying launches
/// whose name starts with `marker_prefix` (one-per-iter marker). Events
/// BEFORE the first marker are treated as init/setup and dropped.
///
/// Returns a `Vec<Vec<TraceEvent>>` with one inner vec per marker found.
#[allow(dead_code)]
pub fn partition_by_marker(
    events: &[TraceEvent],
    marker_prefix: &str,
) -> Vec<Vec<TraceEvent>> {
    let marker_idxs: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| if e.name.starts_with(marker_prefix) { Some(i) } else { None })
        .collect();
    if marker_idxs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(marker_idxs.len());
    for (k, &start) in marker_idxs.iter().enumerate() {
        let end = marker_idxs.get(k + 1).copied().unwrap_or(events.len());
        out.push(events[start..end].to_vec());
    }
    out
}

// ─────────────────────────── Breakdown ───────────────────────────

#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct CategoryBreakdown {
    pub by_category_us: std::collections::BTreeMap<&'static str, f64>,
    pub total_gpu_us: f64,
    pub launch_count: usize,
}

/// Compute per-category GPU time for a single iteration.
#[allow(dead_code)]
pub fn breakdown_iter(iter_events: &[TraceEvent]) -> CategoryBreakdown {
    let mut map = std::collections::BTreeMap::new();
    for cat in OpCategory::all() {
        map.insert(cat.label(), 0.0);
    }
    let mut total = 0.0;
    for ev in iter_events {
        let cat = categorize_kernel(&ev.name);
        *map.entry(cat.label()).or_insert(0.0) += ev.dur_us;
        total += ev.dur_us;
    }
    CategoryBreakdown {
        by_category_us: map,
        total_gpu_us: total,
        launch_count: iter_events.len(),
    }
}

// ─────────────────────────── Unit tests (CPU-only) ───────────────────────────

#[cfg(test)]
mod categorizer_tests {
    use super::*;

    #[test]
    fn cublas_sgemm_is_matmul() {
        assert_eq!(categorize_kernel("sgemm_cublas"), OpCategory::Matmul);
    }

    #[test]
    fn fused_adapter_names_route_to_adapter_fused() {
        for name in [
            "nsl_wrga_fused_lora_m1n4096k4096r16",
            "nsl_wrga_fused_ia3_m1n2048k2048",
            "nsl_wrga_fused_gatedlora_m1n4096k4096r16",
        ] {
            assert_eq!(
                categorize_kernel(name),
                OpCategory::AdapterFused,
                "adapter name {name} should be AdapterFused"
            );
        }
    }

    #[test]
    fn flash_attention_prefix_routes_to_attention_fused() {
        for name in [
            "flash_attn_v2_causal_q32_kv32_d32",
            "flash_attn_backward_v2_causal_q32_kv32_d32",
        ] {
            assert_eq!(
                categorize_kernel(name),
                OpCategory::AttentionFused,
                "attention kernel {name} should be AttentionFused"
            );
        }
    }

    #[test]
    fn elementwise_arith_and_activation_are_separated() {
        assert_eq!(categorize_kernel("nsl_add_f32"),         OpCategory::ElementwiseArith);
        assert_eq!(categorize_kernel("nsl_mul_scalar_f32"),  OpCategory::ElementwiseArith);
        assert_eq!(categorize_kernel("nsl_sigmoid_f32"),     OpCategory::ElementwiseActivation);
        assert_eq!(categorize_kernel("nsl_gelu_backward_f32"), OpCategory::ElementwiseActivation);
        assert_eq!(categorize_kernel("nsl_rotate_half_f32"), OpCategory::ElementwiseActivation);
    }

    #[test]
    fn reductions_and_norms_are_separated() {
        assert_eq!(categorize_kernel("nsl_global_sum_f32"),   OpCategory::Reduction);
        assert_eq!(categorize_kernel("nsl_sum_dim_f32"),      OpCategory::Reduction);
        assert_eq!(categorize_kernel("nsl_max_dim_f32"),      OpCategory::Reduction);
        assert_eq!(categorize_kernel("nsl_rmsnorm_f32"),      OpCategory::Normalization);
        assert_eq!(categorize_kernel("nsl_layernorm_f32"),    OpCategory::Normalization);
        assert_eq!(categorize_kernel("nsl_softmax_f32"),      OpCategory::Normalization);
        assert_eq!(categorize_kernel("nsl_log_softmax_f32"),  OpCategory::Normalization);
    }

    #[test]
    fn indexing_copy_dropout_are_routed() {
        assert_eq!(categorize_kernel("nsl_gather_f32"),       OpCategory::Indexing);
        assert_eq!(categorize_kernel("nsl_scatter_add_f32"),  OpCategory::Indexing);
        assert_eq!(categorize_kernel("nsl_embedding_i32idx"), OpCategory::Indexing);
        assert_eq!(categorize_kernel("nsl_slice_f32"),        OpCategory::Indexing);
        assert_eq!(categorize_kernel("nsl_strided_copy_f32"), OpCategory::CopyLayout);
        assert_eq!(categorize_kernel("nsl_bias_add_f32"),     OpCategory::CopyLayout);
        assert_eq!(categorize_kernel("nsl_rope_cache_write"), OpCategory::CopyLayout);
        assert_eq!(categorize_kernel("nsl_dropout_f32"),      OpCategory::DropoutDequant);
        assert_eq!(
            categorize_kernel("nsl_dequant_fp8_e4m3_f32"),
            OpCategory::DropoutDequant
        );
    }

    #[test]
    fn sparse_and_moe_route_specifically() {
        assert_eq!(categorize_kernel("nsl_csr_spmm_f32"),     OpCategory::Sparse);
        assert_eq!(categorize_kernel("nsl_coo_spmv_f32"),     OpCategory::Sparse);
        assert_eq!(categorize_kernel("nsl_bsr_spmm_f32"),     OpCategory::Sparse);
        assert_eq!(categorize_kernel("moe_gather"),           OpCategory::Moe);
        assert_eq!(categorize_kernel("moe_scatter"),          OpCategory::Moe);
        assert_eq!(categorize_kernel("expert_batched_gemm"),  OpCategory::Moe);
    }

    #[test]
    fn unknown_names_fall_through_to_other() {
        assert_eq!(categorize_kernel(""), OpCategory::Other);
        assert_eq!(categorize_kernel("nsl_experimental_voodoo"), OpCategory::Other);
        assert_eq!(categorize_kernel("unknown_kernel"), OpCategory::Other);
    }

    #[test]
    fn fused_prefix_beats_generic_matmul_substring() {
        // `nsl_wrga_fused_gatedlora_*` could contain inner tokens that
        // would otherwise match the matmul probes. The categorizer
        // checks the fused prefix first; assert the ordering holds.
        let n = "nsl_wrga_fused_gatedlora_m1n4096k4096r16_gemm_backward";
        assert_eq!(categorize_kernel(n), OpCategory::AdapterFused);
    }
}

#[cfg(test)]
mod trace_partition_tests {
    use super::*;

    fn ev(name: &str, ts: f64, dur: f64) -> TraceEvent {
        TraceEvent { name: name.into(), ts_us: ts, dur_us: dur }
    }

    #[test]
    fn parse_trace_from_json_sorts_by_timestamp() {
        // Events emitted out of order on ts.
        let json = r#"{
            "traceEvents": [
                {"name": "b", "dur": 10, "ts": 200},
                {"name": "a", "dur": 5,  "ts": 100}
            ]
        }"#;
        let events = parse_trace_from_json(json).expect("parse");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "a");
        assert_eq!(events[1].name, "b");
    }

    #[test]
    fn partition_by_marker_groups_between_markers() {
        let events = vec![
            ev("init_noise",                                  10.0, 1.0),
            ev("nsl_wrga_fused_gatedlora_m1n4096k4096r16",    20.0, 2.0),
            ev("sgemm_cublas",                                30.0, 3.0),
            ev("nsl_add_f32",                                 40.0, 0.5),
            ev("nsl_wrga_fused_gatedlora_m1n4096k4096r16",    50.0, 2.0),
            ev("sgemm_cublas",                                60.0, 3.0),
        ];
        let iters = partition_by_marker(&events, "nsl_wrga_fused_gatedlora_");
        assert_eq!(iters.len(), 2, "two markers → two iter groups");
        assert_eq!(iters[0].len(), 3, "iter 0: marker + sgemm + add");
        assert_eq!(iters[1].len(), 2, "iter 1: marker + sgemm");
        // Event BEFORE first marker (init_noise) must be dropped.
        assert!(
            !iters.iter().any(|g| g.iter().any(|e| e.name == "init_noise")),
            "init events before first marker must be dropped"
        );
    }

    #[test]
    fn partition_by_marker_empty_when_no_marker() {
        let events = vec![ev("nsl_add_f32", 10.0, 1.0)];
        let iters = partition_by_marker(&events, "no_such_marker_");
        assert!(iters.is_empty());
    }

    #[test]
    fn breakdown_iter_sums_by_category() {
        let iter = vec![
            ev("nsl_wrga_fused_gatedlora_m1n4096k4096r16", 100.0, 3_000.0),
            ev("sgemm_cublas",                             103.0, 8_750.0),
            ev("sgemm_cublas",                             112.0, 0.0), // double-counting guard
            ev("nsl_sigmoid_f32",                          112.0, 320.0),
            ev("nsl_add_f32",                              113.0, 890.0),
            ev("nsl_sum_dim_f32",                          114.0, 480.0),
        ];
        let bd = breakdown_iter(&iter);
        assert_eq!(bd.launch_count, 6);
        let m = bd.by_category_us;
        assert_eq!(m["matmul"],                 8_750.0);
        assert_eq!(m["adapter_fused"],          3_000.0);
        assert_eq!(m["elementwise_activation"], 320.0);
        assert_eq!(m["elementwise_arith"],      890.0);
        assert_eq!(m["reduction"],              480.0);
        assert_eq!(bd.total_gpu_us, 8_750.0 + 3_000.0 + 320.0 + 890.0 + 480.0);
    }
}

// ─────────────────────────── GPU-only integration bench ───────────────────────────

#[cfg(feature = "cuda")]
mod bench {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    fn workspace_root() -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir).parent().unwrap().parent().unwrap().to_path_buf()
    }

    const WARMUP_ITERS: usize = 3;
    const TIMED_ITERS: usize = 10;
    const FUSED_MARKER_PREFIX: &str = "nsl_wrga_fused_gatedlora_";

    fn gen_fwd_bwd_prescribed() -> String {
        // Prescribed shape: b=32, seq=2048, dim=4096, rank=16, alpha=32.
        let batch: u64 = 32;
        let seq: u64 = 2048;
        let dim: u64 = 4096;
        let rank: u64 = 16;
        let alpha: u64 = 32;
        let tokens = batch * seq;
        let n = WARMUP_ITERS + TIMED_ITERS;
        format!(
            r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

train(model = m, epochs = {n}):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#,
            dim = dim, rank = rank, alpha = alpha, tokens = tokens, n = n
        )
    }

    /// Median of a sorted slice — returns 0.0 for empty.
    fn median(v: &mut [f64]) -> f64 {
        if v.is_empty() { return 0.0; }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 0 { 0.5 * (v[n/2 - 1] + v[n/2]) } else { v[n/2] }
    }

    #[test]
    #[ignore]
    fn b32_per_op_breakdown_prescribed_shape() {
        use assert_cmd::prelude::*;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tmpdir");
        let src_path = tmp.path().join("wrga_b32_profile.nsl");
        fs::write(&src_path, gen_fwd_bwd_prescribed()).expect("write nsl source");
        let stdlib = workspace_root().join("stdlib");

        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.current_dir(tmp.path())
            .env("NSL_STDLIB_PATH", &stdlib)
            .env("NSL_PROFILE_KERNELS", "1")
            .env("NSL_WRGA_FUSED_CUDA", "1")
            .arg("run")
            .args(["--source-ad", "--target", "cuda_sm80"])
            .arg(&src_path);

        let t0 = Instant::now();
        let out = cmd.output().expect("spawn nsl");
        let wall_us = t0.elapsed().as_micros() as f64;
        if !out.status.success() {
            panic!(
                "nsl run failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let json_path = tmp.path().join("kernel_profile.json");
        assert!(json_path.exists(), "kernel_profile.json missing");
        let events = parse_trace(&json_path).expect("parse trace");

        let iters = partition_by_marker(&events, FUSED_MARKER_PREFIX);
        let total_markers = iters.len();
        assert!(
            total_markers >= WARMUP_ITERS + TIMED_ITERS,
            "expected >= {} iters, saw {}",
            WARMUP_ITERS + TIMED_ITERS,
            total_markers
        );

        // Drop warmup iters, keep the last TIMED_ITERS.
        let timed_start = total_markers.saturating_sub(TIMED_ITERS);
        let timed: Vec<&Vec<TraceEvent>> = iters[timed_start..].iter().collect();

        // Per-iter breakdown → per-category medians.
        let mut per_cat_samples: std::collections::BTreeMap<&'static str, Vec<f64>> =
            std::collections::BTreeMap::new();
        let mut total_gpu_samples: Vec<f64> = Vec::with_capacity(timed.len());
        for it in &timed {
            let bd = breakdown_iter(it);
            for (k, v) in bd.by_category_us {
                per_cat_samples.entry(k).or_default().push(v);
            }
            total_gpu_samples.push(bd.total_gpu_us);
        }

        let median_gpu_us = median(&mut total_gpu_samples.clone());

        // Wall-clock is across the ENTIRE run (init + warmup + timed +
        // teardown); derive per-iter wall-time by dividing by the total
        // iteration count the train block executed.
        let wall_per_iter_us = wall_us / (total_markers as f64);

        // host_overhead estimated as wall_per_iter - median_gpu
        let host_overhead_us = (wall_per_iter_us - median_gpu_us).max(0.0);

        // Emit the decision-ready table. Stdout only — the `#[ignore]`
        // flag means CI never sees this; the output is for the engineer
        // running the scoping bench.
        println!("\n=== WRGA B.3.2 per-op breakdown — prescribed shape (b=32, seq=2048, dim=4096, r=16) ===");
        println!("wall_clock_total:  {:>12.1} ms  (across {} iters including warmup)", wall_us / 1000.0, total_markers);
        println!("wall_per_iter:     {:>12.1} ms", wall_per_iter_us / 1000.0);
        println!("median_gpu/iter:   {:>12.1} ms  ({:.1}% of wall)", median_gpu_us / 1000.0, 100.0 * median_gpu_us / wall_per_iter_us);
        println!("host_overhead:     {:>12.1} ms  ({:.1}% of wall)  [allocator / tape / CUDA API / source-AD]",
            host_overhead_us / 1000.0, 100.0 * host_overhead_us / wall_per_iter_us);
        println!();
        println!("GPU breakdown (median across {} timed iters):", TIMED_ITERS);
        println!("{:<28} {:>12} {:>8}", "category", "median (ms)", "% gpu");
        println!("{:<28} {:>12} {:>8}", "────────", "──────────", "─────");
        let mut rows: Vec<(&'static str, f64)> = per_cat_samples
            .iter_mut()
            .map(|(k, v)| (*k, median(v)))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (cat, ms) in &rows {
            let pct = if median_gpu_us > 0.0 { 100.0 * ms / median_gpu_us } else { 0.0 };
            println!("{:<28} {:>12.2} {:>7.1}%", cat, ms / 1000.0, pct);
        }
        println!();
        println!("=== B.3.2 scoping signal ===");
        let matmul_pct = rows
            .iter()
            .find(|(c, _)| *c == "matmul")
            .map(|(_, ms)| 100.0 * ms / median_gpu_us.max(1.0))
            .unwrap_or(0.0);
        let gpu_share = 100.0 * median_gpu_us / wall_per_iter_us.max(1.0);
        println!("matmul % of GPU time:       {:>6.1}%", matmul_pct);
        println!("GPU % of wall-clock:        {:>6.1}%", gpu_share);
        println!("matmul % of wall-clock:     {:>6.1}%  ← B.3.2 fused-backward's max theoretical share", 0.01 * matmul_pct * gpu_share);
        if gpu_share < 30.0 {
            println!("  → host-dominated: fused backward's ceiling is bounded by the {:.0}% GPU share.", gpu_share);
            println!("    Allocator / tape optimisation likely outranks B.3.2 in expected impact.");
        } else if matmul_pct > 60.0 {
            println!("  → matmul-dominated within GPU time AND GPU dominates wall: B.3.2 proceed with confidence.");
        } else {
            println!("  → mixed: inspect top-3 categories above; B.3.2's win depends on whether");
            println!("    the fused kernel eliminates the elementwise + reduction HBM roundtrips.");
        }
    }
}
