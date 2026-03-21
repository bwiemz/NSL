# WCET Two-Tier Upgrade (GPU Statistical + FPGA Certified) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the M53 WCET implementation which currently overclaims GPU certification. The code emits DO-178C compliance reports naming GPU as the target — this is a false claim (GPU warp scheduling is undocumented/proprietary; no static WCET tool supports it). Split into two tiers: GPU (statistical bounds, advisory only) and FPGA (certified, deterministic cycle counts via Xilinx XIR/DPU).

**Architecture:** Refactor `wcet.rs` to be target-aware. GPU path renamed to `estimate_*_gpu_statistical()` with confidence intervals. FPGA path added with deterministic cycle counting. DO-178C report generation guarded to FPGA-only. New CLI flags `--wcet-target` and `--fpga-device`.

**Tech Stack:** Rust, existing `nsl-codegen/src/wcet.rs`, `nsl-codegen/src/gpu_specs.rs`, `nsl-cli/src/main.rs`

**Research Basis:** Frontier Features notebook confirms GPU WCET is intractable. Static analysis tools (aiT, Chronos) restricted to CPUs with known cache policies. FPGA with OCM instruction isolation (DICTAT) is the SOTA for certifiable inference.

---

## Critical Issues in Current Code

| Location | Issue | Severity |
|----------|-------|----------|
| `wcet.rs` module doc (line 1-3) | Claims DO-178C certificate emission without target qualification | CRITICAL |
| `wcet.rs:466-470` `prove_static_cf()` | Trivially returns `proven=true` — false proof | CRITICAL |
| `wcet.rs:549,567` `emit_do178c_report()` | Emits DO-178C report for GPU targets | CRITICAL |
| `wcet.rs:257` `occupancy_worst_case` | Single value, no variance modeling | HIGH |
| `compiler/mod.rs:486` | Default GPU "A100-SXM" without target validation | MEDIUM |

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/nsl-codegen/src/wcet.rs` | Split `WcetDevice` → `WcetTarget`, rename GPU functions, guard DO-178C, add FPGA path |
| `crates/nsl-codegen/src/gpu_specs.rs` | Add `FpgaSpec`, `empirical_p95_ratio` field to `GpuSpec` |
| `crates/nsl-codegen/src/compiler/mod.rs` | Split `run_wcet_analysis()` by target |
| `crates/nsl-cli/src/main.rs` | Add `--wcet-target`, `--fpga-device` flags |
| Tests | Update all GPU tests, add FPGA tests |

---

## Tasks

### Phase 1: Fix Overclaims (Hotfix, 1-2 days)

- [ ] **1.1** Update `wcet.rs` module doc to clarify: "GPU = statistical bounds only (advisory), FPGA = certified bounds (DO-178C)".

- [ ] **1.2** Add `certifiable: bool` field to `WcetCertificate`. Set to `false` for GPU, `true` for FPGA only.

- [ ] **1.3** Guard `emit_do178c_report()` — return error if target is not FPGA:
```rust
if !cert.certifiable {
    return Err("DO-178C report requires FPGA target (GPU WCET is statistical only)".into());
}
```

- [ ] **1.4** Add `gpu_statistical_note` to GPU certificates:
```
"WARNING: This WCET bound is a statistical estimate based on roofline modeling.
 GPU warp scheduling is non-deterministic. Actual latency may exceed this bound.
 For safety-critical applications, use --wcet-target fpga."
```

- [ ] **1.5** Fix `prove_static_cf()` — return `proven=false` for GPU with warning message.

- [ ] **1.6** Update all tests that assert GPU DO-178C reports succeed → they should now fail.

### Phase 2: Target-Aware Data Structures (1 day)

- [ ] **2.1** Replace `WcetDevice` enum with `WcetTarget`:
```rust
pub enum WcetTarget {
    Gpu { device_name: String },
    Fpga { device_name: String, ocm_size_kb: u32 },
    GroqLpu,  // future, blocked on ISA docs
}
```

- [ ] **2.2** Add confidence and heuristic source to `OpWcet`:
```rust
pub struct OpWcet {
    // ... existing fields ...
    pub confidence: Option<f64>,          // 0.95 for GPU, 1.0 for FPGA
    pub heuristic_source: Option<String>, // "roofline_with_empirical_variance"
}
```

- [ ] **2.3** Add `FpgaSpec` struct to `gpu_specs.rs`:
```rust
pub struct FpgaSpec {
    pub device_name: &'static str,
    pub vendor: &'static str,
    pub clock_mhz: u32,
    pub ocm_size_kb: u32,
    pub pe_array_dims: (u32, u32),
    pub ocm_latency_cycles: u32,
    pub ddr_latency_cycles: u32,
}
```

- [ ] **2.4** Add `empirical_p95_ratio: f64` to `GpuSpec` (e.g., A100=1.3, H100=1.25).

### Phase 3: GPU Statistical Path (1 day)

- [ ] **3.1** Rename GPU WCET functions:
  - `wcet_matmul_gpu()` → `estimate_matmul_gpu_statistical()`
  - `wcet_elementwise_gpu()` → `estimate_elementwise_gpu_statistical()`
  - `wcet_softmax_gpu()` → `estimate_softmax_gpu_statistical()`

- [ ] **3.2** Add variance modeling to each GPU function:
```rust
let optimistic_ns = /* existing roofline formula */;
let p95_ns = (optimistic_ns as f64 * gpu.empirical_p95_ratio).ceil() as u64;
OpWcet {
    worst_case_ns: p95_ns,
    confidence: Some(0.95),
    heuristic_source: Some("roofline_model_with_empirical_p95_variance"),
    // ...
}
```

- [ ] **3.3** Update `run_wcet_analysis()` in `compiler/mod.rs` for GPU path:
  - Use statistical estimates
  - Emit advisory certificate (no DO-178C)
  - Print warning: "GPU WCET is an estimate, not a proof"

### Phase 4: FPGA Certified Path (2-3 days)

- [ ] **4.1** Add FPGA WCET functions:
```rust
pub fn wcet_matmul_fpga_certified(
    m: u64, k: u64, n: u64, dtype: &str, fpga: &FpgaSpec
) -> OpWcet {
    // Deterministic: PE array cycles + OCM latency (no variance)
    let pe_cycles = (m * k * n) / (fpga.pe_array_dims.0 * fpga.pe_array_dims.1) as u64;
    let total_cycles = pe_cycles + fpga.ocm_latency_cycles as u64;
    let ns = (total_cycles as f64 / fpga.clock_mhz as f64 * 1000.0).ceil() as u64;
    OpWcet { worst_case_ns: ns, confidence: Some(1.0), /* certified */ }
}
```

- [ ] **4.2** Add elementwise and softmax FPGA variants similarly.

- [ ] **4.3** Add FPGA-specific proof: `prove_no_ddr_access()` — verify no external memory operations in the certified path (all data from OCM).

- [ ] **4.4** Update `prove_static_cf()` for FPGA: walk CFG, verify all branches are shape-dependent (statically deterministic). Return `proven=true` only if verified.

- [ ] **4.5** Update `run_wcet_analysis()` for FPGA path:
  - Use certified FPGA estimates
  - Run `prove_no_ddr_access()` and `prove_static_cf(FPGA)`
  - Emit DO-178C report (allowed for FPGA)

### Phase 5: CLI Integration (0.5 days)

- [ ] **5.1** Add `--wcet-target` flag (enum: `gpu`, `fpga`, `groq`; default: `gpu`).

- [ ] **5.2** Add `--fpga-device` flag (string: e.g., `xcvu440`).

- [ ] **5.3** Update help text: clarify `gpu` = statistical advisory, `fpga` = certified DO-178C.

- [ ] **5.4** Wire flags through `CompileOptions` to `run_wcet_analysis()`.

- [ ] **5.5** When `--wcet-target groq`: emit error "Groq LPU WCET not yet supported (ISA not public)".

### Phase 6: Testing (1 day)

- [ ] **6.1** GPU path: verify certificates have `certifiable=false`, no DO-178C report, confidence=0.95.

- [ ] **6.2** FPGA path: verify certificates have `certifiable=true`, DO-178C report emitted, confidence=1.0.

- [ ] **6.3** Guard test: `--wcet --do178c-report` with GPU target → error.

- [ ] **6.4** Guard test: `--wcet --do178c-report --wcet-target fpga` → success.

- [ ] **6.5** Statistical bounds: verify GPU p95 estimate > optimistic estimate.

---

## Effort Estimate

- Phase 1 (fix overclaims): 1-2 days
- Phase 2 (data structures): 1 day
- Phase 3 (GPU statistical): 1 day
- Phase 4 (FPGA certified): 2-3 days
- Phase 5 (CLI): 0.5 days
- Phase 6 (testing): 1 day
- Total: **6-8 days**
