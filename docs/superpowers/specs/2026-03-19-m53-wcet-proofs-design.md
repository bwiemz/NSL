# M53: Hard Real-Time AI — WCET Proofs — Design Specification

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M53
**Prerequisites:** M36 (Compile-Time Memory Planning), M37 (Roofline Cost Model), M52 (Weight-Aware Compilation)
**Dependencies:** M54 (Unikernels) benefits from WCET-certified models; M55 (ZK Circuits) shares the static DAG analysis

## Overview

M53 adds a compiler pass that computes a mathematically rigorous Worst-Case Execution Time (WCET) bound for any NSL model forward pass, producing a JSON certificate with per-operation timing breakdowns and a proof chain that the total latency never exceeds a user-specified bound. When a developer writes `@real_time(max_latency_ms=15.0)`, the compiler either proves the bound holds or emits a structured error explaining exactly which operation causes the violation.

This is **structurally impossible in Python/PyTorch.** Python's garbage collector introduces unpredictable pauses (10-100ms). Dynamic dispatch through `__getattr__`, `nn.Module.forward`, and autograd hooks means the actual code path cannot be determined statically. Even `torch.compile()` with `mode="reduce-overhead"` cannot eliminate CUDA driver jitter, cudaMalloc latency, or Python interpreter overhead. NSL can prove WCET bounds because of three prior milestones:

1. **M36 (Memory Planning):** All tensor memory is slab-allocated at program start — zero dynamic allocation during inference. No `cuMemAlloc` calls, no fragmentation, no OOM recovery paths.
2. **M37 (Roofline Cost Model):** Every tensor operation has a known FLOP count and bytes-moved count, mapped to hardware-specific cycle estimates via a GPU specification database.
3. **M52 (Weight-Aware Compilation):** All weight values are compile-time constants. The computation graph is fully static — no data-dependent branches, no dynamic shapes.

The practical market is safety-critical edge AI: autonomous vehicles (ISO 26262), robotics (IEC 61508), aerospace (DO-178C), and medical devices (IEC 62304). These standards require *proven* worst-case timing bounds, not just statistical latency measurements. NSL with WCET proofs is the first language that can provide these guarantees for neural network inference.

---

## Section 1: Language Surface

### CLI Interface

```bash
# Enable WCET analysis pass
nsl build model.nsl --weights model.safetensors --wcet

# Generate WCET certificate JSON
nsl build model.nsl --weights model.safetensors --wcet --wcet-cert wcet_report.json

# Specify target hardware for cycle counting
nsl build model.nsl --weights model.safetensors --wcet --gpu H100 --cpu cortex-a78

# Check WCET without building (analysis only)
nsl check --wcet model.nsl --weights model.safetensors --gpu H100

# Generate DO-178C documentation stub
nsl build model.nsl --weights model.safetensors --wcet --do178c-report do178c/
```

### NSL Source Annotations

```python
model SafetyController:
    perception: VisionEncoder
    planner: PathPlanner
    actuator: ActuatorNet

    @real_time(max_latency_ms=15.0)
    fn forward(self, camera_frame: Tensor<[1, 3, 224, 224], fp16>) -> Tensor<[1, 6], fp32>:
        features = self.perception(camera_frame)
        plan = self.planner(features)
        return self.actuator(plan)

    @real_time(max_latency_ms=5.0, device="cpu")
    fn emergency_stop(self, sensor: Tensor<[1, 16], fp32>) -> Tensor<[1, 2], fp32>:
        # Must complete on CPU in 5ms even without GPU
        return self.safety_net(sensor)

model VisionEncoder:
    @wcet_budget(max_cycles=2_000_000)  # Fine-grained per-function budget
    fn forward(self, x: Tensor<[1, 3, 224, 224], fp16>) -> Tensor<[1, 512], fp16>:
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        return self.pool(x)
```

### Compiler Output on WCET Violation

```
error[E0900]: WCET bound exceeded
 --> model.nsl:8:5
  |
8 |     @real_time(max_latency_ms=15.0)
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ declared bound: 15.000 ms
  |
  WCET analysis for SafetyController.forward():
    Total WCET: 18.742 ms (124.9% of budget)

    Per-operation breakdown:
      perception.conv1    matmul [1,3,224,224]x[64,3,7,7]   3.214 ms  (17.1%)
      perception.relu     elementwise [1,64,112,112]         0.082 ms  ( 0.4%)
      perception.conv2    matmul [1,64,112,112]x[128,64,3,3] 6.891 ms  (36.8%)
      perception.pool     reduce [1,128,56,56] -> [1,512]   0.143 ms  ( 0.8%)
      planner.attn        matmul [1,512]x[512,512] * 3      1.892 ms  (10.1%)
      planner.ffn         matmul [1,512]x[2048] + relu      2.104 ms  (11.2%)
      actuator.linear1    matmul [1,512]x[256]               0.341 ms  ( 1.8%)
      actuator.linear2    matmul [1,256]x[6]                 0.012 ms  ( 0.1%)
      memory overhead     slab init + data transfer          0.089 ms  ( 0.5%)
      GPU launch overhead 12 kernel launches @ 3.5us each    0.042 ms  ( 0.2%)
      WCET padding        5% safety margin                   3.932 ms  (21.0%)

    Suggestions:
      1. Use FP8 for perception.conv2: saves ~3.4 ms (WCET drops to 15.3 ms)
      2. Fuse perception.conv1 + relu: saves ~0.08 ms (eliminates 1 kernel launch)
      3. Reduce conv2 channels from 128 to 96: saves ~1.7 ms (WCET drops to 17.0 ms)
      4. Apply suggestions 1 + 3 together: WCET drops to 13.6 ms (within budget)
```

---

## Section 2: Architecture

### WCET Analysis Pipeline

The WCET analyzer runs after the weight-aware pass (M52) and memory planner (M36), consuming their outputs:

```
Parse → Semantic → WeightAwarePass (M52) → MemoryPlanner (M36) → WCETAnalyzer → Codegen
                                                                       ↓
                                                              WCETCertificate (JSON)
                                                              DO-178C report (optional)
```

### Core Data Structures

```rust
// crates/nsl-semantic/src/wcet.rs

use std::collections::HashMap;

/// WCET analysis configuration.
pub struct WCETConfig {
    /// Target GPU for cycle counting (e.g., "H100", "A100", "RTX4090")
    pub gpu_target: Option<String>,
    /// Target CPU for cycle counting (e.g., "cortex-a78", "x86-64-v4")
    pub cpu_target: Option<String>,
    /// Safety margin multiplier (default: 1.05 = 5% padding)
    pub safety_margin: f64,
    /// Whether to generate DO-178C documentation
    pub do178c: bool,
    /// Output path for WCET certificate JSON
    pub cert_output: Option<std::path::PathBuf>,
    /// Whether to fail compilation on WCET violation (true) or warn (false)
    pub strict: bool,
}

impl Default for WCETConfig {
    fn default() -> Self {
        Self {
            gpu_target: None,
            cpu_target: None,
            safety_margin: 1.05,
            do178c: false,
            cert_output: None,
            strict: true,
        }
    }
}

/// The main WCET analysis pass.
pub struct WCETAnalyzer {
    /// Hardware cost database (extends M37 roofline cost model)
    cost_db: HardwareCostDatabase,
    /// Memory plan from M36 (to verify no dynamic allocation)
    memory_plan: MemoryPlanSummary,
    /// Weight-aware static graph from M52
    static_graph: StaticComputationGraph,
    /// Configuration
    config: WCETConfig,
    /// Per-operation WCET results
    op_timings: Vec<OpWCET>,
    /// Proof chain entries
    proof_chain: Vec<ProofEntry>,
}

/// WCET timing for a single operation.
#[derive(Debug, Clone)]
pub struct OpWCET {
    /// Operation name (human-readable)
    pub name: String,
    /// Operation kind
    pub kind: OpKind,
    /// Source location in NSL file
    pub source_loc: String,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Device this op runs on
    pub device: WCETDevice,
    /// Worst-case cycle count on target hardware
    pub worst_case_cycles: u64,
    /// Worst-case wall-clock time in nanoseconds
    pub worst_case_ns: u64,
    /// Breakdown: compute cycles
    pub compute_cycles: u64,
    /// Breakdown: memory access cycles (including worst-case cache misses)
    pub memory_cycles: u64,
    /// Breakdown: kernel launch overhead cycles
    pub launch_overhead_cycles: u64,
    /// Breakdown: synchronization overhead cycles
    pub sync_overhead_cycles: u64,
    /// Whether this op was constant-folded (zero runtime cost)
    pub folded: bool,
    /// Proof method used for this timing estimate
    pub proof_method: ProofMethod,
}

/// Categories of operations for WCET analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpKind {
    Matmul,
    Conv2d,
    Elementwise,
    Reduce,
    Softmax,
    Attention,
    DataTransfer,   // host <-> device copy
    KernelLaunch,   // GPU kernel dispatch overhead
    Synchronize,    // cuCtxSynchronize
    SlabInit,       // One-time slab allocation
    WeightLoad,     // Loading weights from .rodata
}

/// Execution device for WCET calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WCETDevice {
    GPU,
    CPU,
}

/// Method used to establish the WCET bound for an operation.
#[derive(Debug, Clone)]
pub enum ProofMethod {
    /// Analytical: computed from FLOP count, bandwidth, and hardware specs
    Analytical {
        flops: u64,
        bytes_moved: u64,
        peak_flops_per_sec: u64,
        peak_bandwidth_bytes_per_sec: u64,
    },
    /// Cache analysis: proved cache miss bounds using access pattern analysis
    CacheAnalysis {
        total_accesses: u64,
        worst_case_misses: u64,
        cache_line_size: u32,
        cache_size_bytes: u64,
    },
    /// Measured: from hardware-specific benchmarks in the cost database
    Measured {
        benchmark_name: String,
        measured_cycles: u64,
        confidence: f64,
    },
    /// Constant: operation was folded away at compile time
    ConstantFolded,
    /// Composite: sum of sub-operations
    Composite {
        sub_operations: Vec<String>,
    },
}

/// An entry in the proof chain, establishing a logical derivation for the WCET bound.
#[derive(Debug, Clone)]
pub struct ProofEntry {
    /// Unique ID for this proof step
    pub id: u32,
    /// Human-readable description of what is being proved
    pub statement: String,
    /// References to prior proof entries this step depends on
    pub depends_on: Vec<u32>,
    /// The specific proof technique used
    pub technique: ProofTechnique,
    /// Whether this proof step passes (bound holds)
    pub verified: bool,
}

/// Proof techniques used in WCET analysis.
#[derive(Debug, Clone)]
pub enum ProofTechnique {
    /// "No dynamic allocation" — verified by M36 memory plan
    StaticMemoryProof {
        planned_tensors: usize,
        unplanned_tensors: usize,
    },
    /// "No data-dependent branches" — verified by M52 static graph
    StaticControlFlowProof {
        total_branches: usize,
        data_dependent_branches: usize,
    },
    /// "Operation X takes at most Y cycles" — from hardware cost model
    OpTimingBound {
        op_name: String,
        bound_cycles: u64,
    },
    /// "Total WCET = sum of all operation WCETs" — additive composition
    AdditiveComposition {
        component_count: usize,
        total_ns: u64,
    },
    /// "No heap allocation in execution path" — verified by slab plan
    NoHeapProof {
        functions_checked: usize,
        heap_calls_found: usize,
    },
}
```

---

## Section 3: Hardware Cost Database

### GPU Specification Database

The WCET analyzer extends M37's cost model with worst-case (not average-case) cycle counts:

```rust
// crates/nsl-semantic/src/wcet.rs (continued)

/// Hardware specification database for WCET cycle counting.
pub struct HardwareCostDatabase {
    /// GPU specs indexed by model name
    gpus: HashMap<String, GpuSpec>,
    /// CPU specs indexed by model name
    cpus: HashMap<String, CpuSpec>,
}

/// GPU specifications for WCET analysis.
/// All values represent WORST-CASE (not typical) performance.
#[derive(Debug, Clone)]
pub struct GpuSpec {
    pub name: String,
    /// Peak FP32 TFLOPS (theoretical max, used as upper bound)
    pub peak_fp32_tflops: f64,
    /// Peak FP16 TFLOPS (with tensor cores)
    pub peak_fp16_tflops: f64,
    /// Peak FP8 TFLOPS (with tensor cores, if supported)
    pub peak_fp8_tflops: Option<f64>,
    /// Peak INT8 TOPS
    pub peak_int8_tops: Option<f64>,
    /// HBM bandwidth in GB/s (theoretical peak)
    pub hbm_bandwidth_gbps: f64,
    /// L2 cache size in bytes
    pub l2_cache_bytes: u64,
    /// L2 cache bandwidth in GB/s
    pub l2_bandwidth_gbps: f64,
    /// Shared memory per SM in bytes
    pub shared_mem_per_sm: u32,
    /// Number of SMs
    pub num_sms: u32,
    /// GPU clock speed in MHz (boost clock, worst-case = base clock)
    pub base_clock_mhz: u32,
    /// Kernel launch overhead in nanoseconds (worst case)
    pub kernel_launch_overhead_ns: u64,
    /// cuCtxSynchronize overhead in nanoseconds (worst case)
    pub sync_overhead_ns: u64,
    /// PCIe bandwidth for host-device transfers in GB/s
    pub pcie_bandwidth_gbps: f64,
    /// Worst-case SM occupancy degradation factor (0.0-1.0)
    /// Accounts for register pressure, shared memory limits, etc.
    pub occupancy_worst_case: f64,
}

/// CPU specifications for WCET analysis.
#[derive(Debug, Clone)]
pub struct CpuSpec {
    pub name: String,
    /// Clock speed in MHz (base, not boost — worst case)
    pub base_clock_mhz: u32,
    /// FP32 FLOPS per cycle per core (from SIMD width)
    pub fp32_flops_per_cycle: u32,
    /// FP16 FLOPS per cycle per core (if supported)
    pub fp16_flops_per_cycle: Option<u32>,
    /// L1 data cache size in bytes
    pub l1d_cache_bytes: u32,
    /// L2 cache size in bytes
    pub l2_cache_bytes: u64,
    /// L3 cache size in bytes (if applicable)
    pub l3_cache_bytes: Option<u64>,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Cache line size in bytes
    pub cache_line_bytes: u32,
    /// Number of cores
    pub num_cores: u32,
    /// Context switch overhead in nanoseconds (for interrupt-free WCET)
    pub context_switch_overhead_ns: u64,
}

impl HardwareCostDatabase {
    /// Create the built-in database with known GPU and CPU specifications.
    pub fn builtin() -> Self {
        let mut gpus = HashMap::new();
        let mut cpus = HashMap::new();

        // NVIDIA H100 SXM5
        gpus.insert("H100".to_string(), GpuSpec {
            name: "NVIDIA H100 SXM5".to_string(),
            peak_fp32_tflops: 67.0,
            peak_fp16_tflops: 989.0,    // with tensor cores, sparse
            peak_fp8_tflops: Some(1979.0),
            peak_int8_tops: Some(1979.0),
            hbm_bandwidth_gbps: 3350.0,
            l2_cache_bytes: 50 * 1024 * 1024,  // 50 MB
            l2_bandwidth_gbps: 12000.0,
            shared_mem_per_sm: 228 * 1024,
            num_sms: 132,
            base_clock_mhz: 1095,               // base clock, not boost
            kernel_launch_overhead_ns: 5000,     // 5us worst case
            sync_overhead_ns: 2000,              // 2us worst case
            pcie_bandwidth_gbps: 64.0,           // PCIe Gen5 x16
            occupancy_worst_case: 0.5,
        });

        // NVIDIA A100 SXM4
        gpus.insert("A100".to_string(), GpuSpec {
            name: "NVIDIA A100 SXM4".to_string(),
            peak_fp32_tflops: 19.5,
            peak_fp16_tflops: 312.0,
            peak_fp8_tflops: None,
            peak_int8_tops: Some(624.0),
            hbm_bandwidth_gbps: 2039.0,
            l2_cache_bytes: 40 * 1024 * 1024,
            l2_bandwidth_gbps: 6000.0,
            shared_mem_per_sm: 164 * 1024,
            num_sms: 108,
            base_clock_mhz: 765,
            kernel_launch_overhead_ns: 6000,
            sync_overhead_ns: 3000,
            pcie_bandwidth_gbps: 32.0,
            occupancy_worst_case: 0.5,
        });

        // NVIDIA Jetson Orin (edge deployment)
        gpus.insert("Orin".to_string(), GpuSpec {
            name: "NVIDIA Jetson AGX Orin".to_string(),
            peak_fp32_tflops: 5.3,
            peak_fp16_tflops: 170.0,    // with sparse
            peak_fp8_tflops: None,
            peak_int8_tops: Some(275.0),
            hbm_bandwidth_gbps: 204.8,  // LPDDR5
            l2_cache_bytes: 4 * 1024 * 1024,
            l2_bandwidth_gbps: 800.0,
            shared_mem_per_sm: 128 * 1024,
            num_sms: 16,
            base_clock_mhz: 624,
            kernel_launch_overhead_ns: 8000,
            sync_overhead_ns: 4000,
            pcie_bandwidth_gbps: 0.0,   // unified memory
            occupancy_worst_case: 0.4,
        });

        // ARM Cortex-A78 (autonomous vehicle SoC CPU)
        cpus.insert("cortex-a78".to_string(), CpuSpec {
            name: "ARM Cortex-A78".to_string(),
            base_clock_mhz: 2000,
            fp32_flops_per_cycle: 8,   // 128-bit NEON = 4 FP32 * 2 (FMA)
            fp16_flops_per_cycle: Some(16),
            l1d_cache_bytes: 64 * 1024,
            l2_cache_bytes: 512 * 1024,
            l3_cache_bytes: Some(4 * 1024 * 1024),
            memory_bandwidth_gbps: 51.2,
            cache_line_bytes: 64,
            num_cores: 4,
            context_switch_overhead_ns: 500,
        });

        // x86-64-v4 (server CPU, AVX-512)
        cpus.insert("x86-64-v4".to_string(), CpuSpec {
            name: "x86-64-v4 (AVX-512)".to_string(),
            base_clock_mhz: 2100,
            fp32_flops_per_cycle: 32,  // 512-bit * 2 (FMA)
            fp16_flops_per_cycle: Some(64),
            l1d_cache_bytes: 48 * 1024,
            l2_cache_bytes: 2 * 1024 * 1024,
            l3_cache_bytes: Some(36 * 1024 * 1024),
            memory_bandwidth_gbps: 102.4,
            cache_line_bytes: 64,
            num_cores: 16,
            context_switch_overhead_ns: 200,
        });

        Self { gpus, cpus }
    }

    pub fn get_gpu(&self, name: &str) -> Option<&GpuSpec> {
        self.gpus.get(name)
    }

    pub fn get_cpu(&self, name: &str) -> Option<&CpuSpec> {
        self.cpus.get(name)
    }
}
```

---

## Section 4: Per-Operation Cycle Counting

### WCET Formulas

Each operation's worst-case cycle count is derived from M37's cost formulas but uses worst-case (not typical) hardware parameters. The key difference: M37 uses peak throughput for classification; WCET uses base clock, worst-case occupancy, and cache miss bounds.

```rust
// crates/nsl-semantic/src/wcet.rs (continued)

impl WCETAnalyzer {
    /// Compute worst-case nanoseconds for a matmul operation on GPU.
    pub fn wcet_matmul_gpu(
        &self,
        m: u64, k: u64, n: u64,
        dtype: &str,
        gpu: &GpuSpec,
    ) -> OpWCET {
        let flops = 2 * m * k * n;
        let bytes_read = (m * k + k * n) * dtype_bytes(dtype) as u64;
        let bytes_written = m * n * dtype_bytes(dtype) as u64;
        let total_bytes = bytes_read + bytes_written;

        // Worst-case compute time: FLOPs / (peak_rate * occupancy_worst_case)
        let peak_flops_per_ns = match dtype {
            "fp16" | "bf16" => gpu.peak_fp16_tflops * 1e3, // TFLOPS -> GFLOPS -> FLOPS/ns
            "fp8" => gpu.peak_fp8_tflops.unwrap_or(gpu.peak_fp16_tflops) * 1e3,
            "int8" => gpu.peak_int8_tops.unwrap_or(gpu.peak_fp16_tflops) * 1e3,
            _ => gpu.peak_fp32_tflops * 1e3,
        };
        let effective_flops_per_ns = peak_flops_per_ns * gpu.occupancy_worst_case;
        let compute_ns = (flops as f64 / effective_flops_per_ns).ceil() as u64;

        // Worst-case memory time: assume all data comes from HBM (no L2 cache hits)
        let memory_ns = (total_bytes as f64 / (gpu.hbm_bandwidth_gbps * 1e-9)).ceil() as u64;

        // WCET = max(compute, memory) + launch overhead + sync overhead
        let base_ns = compute_ns.max(memory_ns);
        let total_ns = base_ns + gpu.kernel_launch_overhead_ns + gpu.sync_overhead_ns;

        OpWCET {
            name: format!("matmul [{m},{k}]x[{k},{n}]"),
            kind: OpKind::Matmul,
            source_loc: String::new(),
            input_shapes: vec![vec![m as usize, k as usize], vec![k as usize, n as usize]],
            output_shape: vec![m as usize, n as usize],
            dtype: dtype.to_string(),
            device: WCETDevice::GPU,
            worst_case_cycles: total_ns * gpu.base_clock_mhz as u64 / 1000, // ns -> cycles
            worst_case_ns: total_ns,
            compute_cycles: compute_ns * gpu.base_clock_mhz as u64 / 1000,
            memory_cycles: memory_ns * gpu.base_clock_mhz as u64 / 1000,
            launch_overhead_cycles: gpu.kernel_launch_overhead_ns * gpu.base_clock_mhz as u64 / 1000,
            sync_overhead_cycles: gpu.sync_overhead_ns * gpu.base_clock_mhz as u64 / 1000,
            folded: false,
            proof_method: ProofMethod::Analytical {
                flops,
                bytes_moved: total_bytes,
                peak_flops_per_sec: (effective_flops_per_ns * 1e9) as u64,
                peak_bandwidth_bytes_per_sec: (gpu.hbm_bandwidth_gbps * 1e9) as u64,
            },
        }
    }

    /// Compute worst-case nanoseconds for an elementwise operation on GPU.
    pub fn wcet_elementwise_gpu(
        &self,
        num_elements: u64,
        dtype: &str,
        op_name: &str,
        gpu: &GpuSpec,
    ) -> OpWCET {
        let bw = dtype_bytes(dtype) as u64;
        let bytes_read = num_elements * bw;
        let bytes_written = num_elements * bw;
        let total_bytes = bytes_read + bytes_written;

        // Elementwise ops are always memory-bound
        let memory_ns = (total_bytes as f64 / (gpu.hbm_bandwidth_gbps * 1e-9)).ceil() as u64;
        let total_ns = memory_ns + gpu.kernel_launch_overhead_ns + gpu.sync_overhead_ns;

        OpWCET {
            name: format!("{op_name} [{num_elements}]"),
            kind: OpKind::Elementwise,
            source_loc: String::new(),
            input_shapes: vec![vec![num_elements as usize]],
            output_shape: vec![num_elements as usize],
            dtype: dtype.to_string(),
            device: WCETDevice::GPU,
            worst_case_cycles: total_ns * gpu.base_clock_mhz as u64 / 1000,
            worst_case_ns: total_ns,
            compute_cycles: 0,  // memory-bound
            memory_cycles: memory_ns * gpu.base_clock_mhz as u64 / 1000,
            launch_overhead_cycles: gpu.kernel_launch_overhead_ns * gpu.base_clock_mhz as u64 / 1000,
            sync_overhead_cycles: gpu.sync_overhead_ns * gpu.base_clock_mhz as u64 / 1000,
            folded: false,
            proof_method: ProofMethod::Analytical {
                flops: num_elements,
                bytes_moved: total_bytes,
                peak_flops_per_sec: 0,
                peak_bandwidth_bytes_per_sec: (gpu.hbm_bandwidth_gbps * 1e9) as u64,
            },
        }
    }

    /// Compute worst-case nanoseconds for a matmul on CPU.
    pub fn wcet_matmul_cpu(
        &self,
        m: u64, k: u64, n: u64,
        dtype: &str,
        cpu: &CpuSpec,
    ) -> OpWCET {
        let flops = 2 * m * k * n;
        let bw = dtype_bytes(dtype) as u64;
        let total_bytes = (m * k + k * n + m * n) * bw;

        let flops_per_cycle = match dtype {
            "fp16" | "bf16" => cpu.fp16_flops_per_cycle.unwrap_or(cpu.fp32_flops_per_cycle),
            _ => cpu.fp32_flops_per_cycle,
        } as u64;

        // Worst-case: single core, no SIMD benefit for small matrices
        let effective_flops_per_cycle = if m * n < 64 {
            1  // scalar fallback for tiny matmuls
        } else {
            flops_per_cycle
        };

        let compute_cycles = flops / effective_flops_per_cycle;

        // Worst-case cache misses: assume no data in cache
        let cache_lines = (total_bytes + cpu.cache_line_bytes as u64 - 1) / cpu.cache_line_bytes as u64;
        let memory_ns = (total_bytes as f64 / (cpu.memory_bandwidth_gbps * 1e-9)).ceil() as u64;
        let memory_cycles = memory_ns * cpu.base_clock_mhz as u64 / 1000;

        let total_cycles = compute_cycles.max(memory_cycles);
        let total_ns = total_cycles * 1000 / cpu.base_clock_mhz as u64;

        OpWCET {
            name: format!("cpu_matmul [{m},{k}]x[{k},{n}]"),
            kind: OpKind::Matmul,
            source_loc: String::new(),
            input_shapes: vec![vec![m as usize, k as usize], vec![k as usize, n as usize]],
            output_shape: vec![m as usize, n as usize],
            dtype: dtype.to_string(),
            device: WCETDevice::CPU,
            worst_case_cycles: total_cycles,
            worst_case_ns: total_ns,
            compute_cycles,
            memory_cycles,
            launch_overhead_cycles: 0,
            sync_overhead_cycles: 0,
            folded: false,
            proof_method: ProofMethod::CacheAnalysis {
                total_accesses: flops + total_bytes / bw,
                worst_case_misses: cache_lines,
                cache_line_size: cpu.cache_line_bytes,
                cache_size_bytes: cpu.l2_cache_bytes,
            },
        }
    }
}

/// Helper: bytes per element for a dtype string.
fn dtype_bytes(dtype: &str) -> usize {
    match dtype {
        "fp8" | "int8" | "i8" => 1,
        "fp16" | "bf16" | "f16" => 2,
        "fp32" | "f32" | "int32" | "i32" => 4,
        "fp64" | "f64" => 8,
        _ => 4, // default to FP32
    }
}
```

---

## Section 5: Memory Access Pattern Analysis

### Cache Miss Bound Proofs

The WCET analyzer proves an upper bound on cache misses for each operation by analyzing the tensor access patterns:

```rust
// crates/nsl-semantic/src/wcet.rs (continued)

/// Cache miss analysis for a tensor access pattern.
pub struct CacheMissAnalysis {
    /// Total memory accesses
    pub total_accesses: u64,
    /// Proven upper bound on L2 cache misses
    pub worst_case_l2_misses: u64,
    /// Access pattern classification
    pub pattern: AccessPattern,
    /// Working set size in bytes
    pub working_set_bytes: u64,
    /// Whether the working set fits in L2 cache
    pub fits_in_l2: bool,
}

/// Classification of tensor access patterns for cache analysis.
#[derive(Debug, Clone)]
pub enum AccessPattern {
    /// Sequential scan: each element accessed once in order
    /// Cache misses = ceil(data_bytes / cache_line_size)
    Sequential,
    /// Strided access: elements accessed with a fixed stride
    /// Cache misses depend on stride vs. cache line size
    Strided { stride_bytes: u64 },
    /// Row-major matmul: one matrix sequential, other strided
    /// Inner dimension sequential -> good locality; outer dimension strided
    MatmulRowMajor { inner_dim: u64, outer_dim: u64 },
    /// Tiled access: blocked iteration for cache reuse
    /// Working set = tile_size^2; misses = total_bytes / cache_line if tile fits in cache
    Tiled { tile_m: u64, tile_n: u64, tile_k: u64 },
    /// Reduction: multiple passes over same data
    /// First pass: cold misses; subsequent passes: all hits if data fits in cache
    Reduction { passes: u32, data_bytes: u64 },
}

impl WCETAnalyzer {
    /// Analyze cache behavior for a matmul operation.
    /// Uses the tiled access pattern model: the matmul is decomposed into tiles
    /// that fit in L2 cache, minimizing cache misses.
    pub fn analyze_matmul_cache(
        &self,
        m: u64, k: u64, n: u64,
        dtype_bytes: u64,
        cache: &CacheSpec,
    ) -> CacheMissAnalysis {
        let a_bytes = m * k * dtype_bytes;
        let b_bytes = k * n * dtype_bytes;
        let c_bytes = m * n * dtype_bytes;
        let total_bytes = a_bytes + b_bytes + c_bytes;

        // For WCET, assume worst case: no tiling benefit, all accesses miss
        // This is a safe upper bound
        let cache_line_size = cache.line_size as u64;
        let worst_case_misses = (total_bytes + cache_line_size - 1) / cache_line_size;

        let fits_in_l2 = total_bytes <= cache.l2_bytes;

        CacheMissAnalysis {
            total_accesses: 2 * m * k * n + m * n,  // reads + writes
            worst_case_l2_misses: worst_case_misses,
            pattern: AccessPattern::MatmulRowMajor {
                inner_dim: k,
                outer_dim: m * n,
            },
            working_set_bytes: total_bytes,
            fits_in_l2,
        }
    }

    /// Analyze cache behavior for an elementwise operation.
    /// Streaming pattern: each element read once, written once.
    pub fn analyze_elementwise_cache(
        &self,
        num_elements: u64,
        dtype_bytes: u64,
        cache: &CacheSpec,
    ) -> CacheMissAnalysis {
        let total_bytes = num_elements * dtype_bytes * 2; // read + write
        let cache_line_size = cache.line_size as u64;
        let worst_case_misses = (total_bytes + cache_line_size - 1) / cache_line_size;

        CacheMissAnalysis {
            total_accesses: num_elements * 2,
            worst_case_l2_misses: worst_case_misses,
            pattern: AccessPattern::Sequential,
            working_set_bytes: total_bytes,
            fits_in_l2: total_bytes <= cache.l2_bytes,
        }
    }
}

pub struct CacheSpec {
    pub line_size: u32,
    pub l2_bytes: u64,
}
```

---

## Section 6: No-Heap-Allocation Proof

### Static Memory Verification

The WCET analyzer verifies that the entire certified execution path uses only slab-allocated memory (from M36), with zero dynamic allocation calls:

```rust
// crates/nsl-semantic/src/wcet.rs (continued)

/// Proof that no heap allocation occurs during the certified execution path.
pub struct NoHeapProof {
    /// Functions in the certified path
    pub functions_checked: Vec<String>,
    /// Number of tensor allocation sites in the path
    pub total_alloc_sites: usize,
    /// Number of sites covered by slab planning (M36)
    pub slab_planned_sites: usize,
    /// Number of sites that fall through to runtime allocation (VIOLATIONS)
    pub runtime_alloc_sites: usize,
    /// Specific violation locations
    pub violations: Vec<HeapViolation>,
    /// Whether the proof holds
    pub proven: bool,
}

#[derive(Debug, Clone)]
pub struct HeapViolation {
    /// Source location of the dynamic allocation
    pub source_loc: String,
    /// Function containing the allocation
    pub function: String,
    /// Reason this allocation is dynamic
    pub reason: String,
    /// Suggested fix
    pub suggestion: String,
}

impl WCETAnalyzer {
    /// Walk the function call graph from the @real_time entry point,
    /// verifying that every tensor allocation is covered by the M36 slab plan.
    pub fn prove_no_heap(
        &self,
        entry_function: &str,
        call_graph: &CallGraph,
        memory_plan: &MemoryPlanSummary,
    ) -> NoHeapProof {
        let mut functions_checked = Vec::new();
        let mut total_alloc_sites = 0usize;
        let mut slab_planned = 0usize;
        let mut violations = Vec::new();

        // BFS through call graph from entry point
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(entry_function.to_string());

        while let Some(func) = queue.pop_front() {
            if !visited.insert(func.clone()) { continue; }
            functions_checked.push(func.clone());

            // Check each allocation site in this function
            if let Some(allocs) = memory_plan.allocs_in_function(&func) {
                for alloc in allocs {
                    total_alloc_sites += 1;
                    if alloc.is_slab_planned {
                        slab_planned += 1;
                    } else {
                        violations.push(HeapViolation {
                            source_loc: alloc.source_loc.clone(),
                            function: func.clone(),
                            reason: format!("Tensor '{}' has dynamic size: {}", alloc.name, alloc.reason),
                            suggestion: format!(
                                "Add a bounded dimension: Tensor<[..., Dim < {}], ...>",
                                alloc.suggested_bound.unwrap_or(4096)
                            ),
                        });
                    }
                }
            }

            // Enqueue callees
            if let Some(callees) = call_graph.callees(&func) {
                for callee in callees {
                    queue.push_back(callee.clone());
                }
            }
        }

        let runtime_alloc_sites = violations.len();
        NoHeapProof {
            functions_checked,
            total_alloc_sites,
            slab_planned_sites: slab_planned,
            runtime_alloc_sites,
            violations,
            proven: runtime_alloc_sites == 0,
        }
    }
}

/// Summary of M36 memory plan, consumed by WCET analyzer.
pub struct MemoryPlanSummary {
    pub functions: HashMap<String, Vec<AllocSiteSummary>>,
}

pub struct AllocSiteSummary {
    pub name: String,
    pub source_loc: String,
    pub is_slab_planned: bool,
    pub reason: String,
    pub suggested_bound: Option<u64>,
}

impl MemoryPlanSummary {
    pub fn allocs_in_function(&self, func: &str) -> Option<&Vec<AllocSiteSummary>> {
        self.functions.get(func)
    }
}

/// Simplified call graph for WCET analysis.
pub struct CallGraph {
    edges: HashMap<String, Vec<String>>,
}

impl CallGraph {
    pub fn callees(&self, func: &str) -> Option<&Vec<String>> {
        self.edges.get(func)
    }
}
```

---

## Section 7: WCET Certificate Generation

### JSON Certificate Format

```rust
// crates/nsl-codegen/src/wcet_report.rs

use serde::{Serialize, Deserialize};

/// The complete WCET certificate, written as JSON.
#[derive(Debug, Serialize, Deserialize)]
pub struct WCETCertificate {
    /// Certificate version
    pub version: String,
    /// Date of generation (ISO 8601)
    pub generated_at: String,
    /// NSL compiler version
    pub compiler_version: String,
    /// Source file that was analyzed
    pub source_file: String,
    /// Weight file used (SHA-256 hash)
    pub weight_hash: String,
    /// Target hardware
    pub target: CertTarget,
    /// Declared WCET bound (from @real_time decorator)
    pub declared_bound_ms: f64,
    /// Computed WCET (before safety margin)
    pub computed_wcet_ms: f64,
    /// Final WCET with safety margin
    pub final_wcet_ms: f64,
    /// Safety margin multiplier used
    pub safety_margin: f64,
    /// Whether the bound holds
    pub bound_satisfied: bool,
    /// Per-operation timing breakdown
    pub operations: Vec<CertOpEntry>,
    /// Proof chain
    pub proof_chain: Vec<CertProofEntry>,
    /// No-heap allocation proof
    pub no_heap_proof: CertNoHeapProof,
    /// Static control flow proof
    pub static_control_flow: CertStaticCFProof,
    /// Summary statistics
    pub summary: CertSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertTarget {
    pub gpu: Option<String>,
    pub cpu: Option<String>,
    pub gpu_spec: Option<CertGpuSpec>,
    pub cpu_spec: Option<CertCpuSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertGpuSpec {
    pub name: String,
    pub peak_fp16_tflops: f64,
    pub hbm_bandwidth_gbps: f64,
    pub base_clock_mhz: u32,
    pub occupancy_worst_case: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertCpuSpec {
    pub name: String,
    pub base_clock_mhz: u32,
    pub fp32_flops_per_cycle: u32,
    pub memory_bandwidth_gbps: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertOpEntry {
    pub name: String,
    pub kind: String,
    pub source_loc: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub wcet_ns: u64,
    pub wcet_ms: f64,
    pub fraction_of_total: f64,
    pub compute_ns: u64,
    pub memory_ns: u64,
    pub launch_overhead_ns: u64,
    pub folded: bool,
    pub proof_method: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertProofEntry {
    pub id: u32,
    pub statement: String,
    pub depends_on: Vec<u32>,
    pub technique: String,
    pub verified: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertNoHeapProof {
    pub functions_checked: usize,
    pub alloc_sites_total: usize,
    pub alloc_sites_planned: usize,
    pub violations: usize,
    pub proven: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertStaticCFProof {
    pub total_branches: usize,
    pub data_dependent_branches: usize,
    pub proven: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CertSummary {
    pub total_ops: usize,
    pub total_folded_ops: usize,
    pub total_kernel_launches: usize,
    pub slowest_op: String,
    pub slowest_op_ms: f64,
    pub compute_bound_ops: usize,
    pub memory_bound_ops: usize,
}

/// Generate the WCET certificate JSON and write to file.
pub fn emit_wcet_certificate(
    cert: &WCETCertificate,
    output_path: &std::path::Path,
) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(cert)?;
    std::fs::write(output_path, json)?;
    Ok(())
}
```

### Example Certificate Output

```json
{
  "version": "1.0",
  "generated_at": "2026-03-19T14:30:00Z",
  "compiler_version": "nsl 0.9.0",
  "source_file": "safety_controller.nsl",
  "weight_hash": "a3f8c2...d91e",
  "target": {
    "gpu": "Orin",
    "gpu_spec": {
      "name": "NVIDIA Jetson AGX Orin",
      "peak_fp16_tflops": 170.0,
      "hbm_bandwidth_gbps": 204.8,
      "base_clock_mhz": 624,
      "occupancy_worst_case": 0.4
    }
  },
  "declared_bound_ms": 15.0,
  "computed_wcet_ms": 12.847,
  "final_wcet_ms": 13.489,
  "safety_margin": 1.05,
  "bound_satisfied": true,
  "operations": [
    {
      "name": "perception.conv1 matmul",
      "kind": "Matmul",
      "wcet_ms": 4.213,
      "fraction_of_total": 0.312,
      "proof_method": "Analytical"
    }
  ],
  "proof_chain": [
    {
      "id": 0,
      "statement": "All tensor memory is slab-allocated (0 dynamic allocations)",
      "depends_on": [],
      "technique": "StaticMemoryProof",
      "verified": true
    },
    {
      "id": 1,
      "statement": "No data-dependent branches in forward pass (0 of 12 branches are data-dependent)",
      "depends_on": [],
      "technique": "StaticControlFlowProof",
      "verified": true
    },
    {
      "id": 2,
      "statement": "Total WCET = sum of 24 operation WCETs = 12.847 ms",
      "depends_on": [0, 1],
      "technique": "AdditiveComposition",
      "verified": true
    },
    {
      "id": 3,
      "statement": "Final WCET (12.847 ms * 1.05 margin = 13.489 ms) <= declared bound (15.0 ms)",
      "depends_on": [2],
      "technique": "BoundCheck",
      "verified": true
    }
  ],
  "no_heap_proof": {
    "functions_checked": 8,
    "alloc_sites_total": 47,
    "alloc_sites_planned": 47,
    "violations": 0,
    "proven": true
  },
  "summary": {
    "total_ops": 24,
    "total_folded_ops": 3,
    "total_kernel_launches": 18,
    "slowest_op": "perception.conv2",
    "slowest_op_ms": 5.891,
    "compute_bound_ops": 6,
    "memory_bound_ops": 18
  }
}
```

---

## Section 8: DO-178C Documentation Stub

### Safety Standards Compliance

```rust
// crates/nsl-codegen/src/wcet_report.rs (continued)

/// DO-178C documentation generation stub.
/// Produces a directory structure compatible with aerospace certification requirements.
pub struct DO178CReport {
    /// Output directory
    pub output_dir: std::path::PathBuf,
}

impl DO178CReport {
    /// Generate DO-178C compliant documentation from WCET analysis.
    pub fn generate(
        &self,
        cert: &WCETCertificate,
    ) -> Result<(), std::io::Error> {
        std::fs::create_dir_all(&self.output_dir)?;

        // Section 1: Software Requirements Specification (SRS) stub
        self.write_srs(cert)?;

        // Section 2: Software Design Document (SDD) stub
        self.write_sdd(cert)?;

        // Section 3: Verification Results
        self.write_verification(cert)?;

        Ok(())
    }

    fn write_srs(&self, cert: &WCETCertificate) -> Result<(), std::io::Error> {
        let content = format!(
            "DO-178C Software Requirements Specification\n\
             ============================================\n\n\
             REQ-RT-001: The neural network forward pass SHALL complete within {:.3} ms\n\
             on target hardware: {}\n\n\
             REQ-MEM-001: The neural network forward pass SHALL NOT perform dynamic\n\
             memory allocation during execution.\n\n\
             REQ-DET-001: The neural network forward pass SHALL have deterministic\n\
             control flow independent of input data values.\n",
            cert.declared_bound_ms,
            cert.target.gpu.as_deref().unwrap_or("unspecified"),
        );
        std::fs::write(self.output_dir.join("srs.txt"), content)
    }

    fn write_sdd(&self, cert: &WCETCertificate) -> Result<(), std::io::Error> {
        let content = format!(
            "DO-178C Software Design Document\n\
             =================================\n\n\
             Model architecture: {} operations\n\
             Memory model: Static slab allocation ({} tensor sites)\n\
             Weight file: SHA-256 {}\n\
             Compilation: Weight-aware AOT with constant folding\n",
            cert.summary.total_ops,
            cert.no_heap_proof.alloc_sites_total,
            cert.weight_hash,
        );
        std::fs::write(self.output_dir.join("sdd.txt"), content)
    }

    fn write_verification(&self, cert: &WCETCertificate) -> Result<(), std::io::Error> {
        let content = format!(
            "DO-178C Verification Results\n\
             ============================\n\n\
             WCET Analysis Result: {}\n\
             Declared bound: {:.3} ms\n\
             Computed WCET: {:.3} ms (with {:.0}% safety margin: {:.3} ms)\n\n\
             No-heap proof: {}\n\
             Static control flow proof: {}\n\n\
             Proof chain: {} steps, all verified: {}\n",
            if cert.bound_satisfied { "PASS" } else { "FAIL" },
            cert.declared_bound_ms,
            cert.computed_wcet_ms,
            (cert.safety_margin - 1.0) * 100.0,
            cert.final_wcet_ms,
            if cert.no_heap_proof.proven { "PROVEN" } else { "FAILED" },
            if cert.static_control_flow.proven { "PROVEN" } else { "FAILED" },
            cert.proof_chain.len(),
            cert.proof_chain.iter().all(|p| p.verified),
        );
        std::fs::write(self.output_dir.join("verification.txt"), content)
    }
}
```

---

## Section 9: Codegen Changes

### `@real_time` Decorator Parsing

```rust
// Changes to crates/nsl-parser/src/parser.rs

/// Parse @real_time(max_latency_ms=15.0) decorator.
/// Also supports @real_time(max_latency_ms=15.0, device="cpu").
fn parse_real_time_decorator(&mut self) -> Result<Decorator, ParseError> {
    self.expect(Token::At)?;
    self.expect_ident("real_time")?;
    self.expect(Token::LParen)?;

    let mut max_latency_ms: Option<f64> = None;
    let mut device: Option<String> = None;

    loop {
        let key = self.expect_ident_any()?;
        self.expect(Token::Eq)?;
        match key.as_str() {
            "max_latency_ms" => {
                max_latency_ms = Some(self.parse_float_literal()?);
            }
            "device" => {
                device = Some(self.parse_string_literal()?);
            }
            other => {
                return Err(ParseError::UnexpectedArg(
                    format!("@real_time: unknown argument '{other}'"),
                ));
            }
        }
        if !self.check(Token::Comma) { break; }
        self.advance(); // consume comma
    }

    self.expect(Token::RParen)?;

    Ok(Decorator::RealTime {
        max_latency_ms: max_latency_ms.ok_or(ParseError::MissingArg("max_latency_ms"))?,
        device,
    })
}

/// Parse @wcet_budget(max_cycles=2_000_000) decorator.
fn parse_wcet_budget_decorator(&mut self) -> Result<Decorator, ParseError> {
    self.expect(Token::At)?;
    self.expect_ident("wcet_budget")?;
    self.expect(Token::LParen)?;
    self.expect_ident("max_cycles")?;
    self.expect(Token::Eq)?;
    let max_cycles = self.parse_int_literal()? as u64;
    self.expect(Token::RParen)?;
    Ok(Decorator::WCETBudget { max_cycles })
}
```

### Semantic Checker Integration

```rust
// Changes to crates/nsl-semantic/src/checker.rs

impl SemanticChecker {
    /// Check a function with @real_time decorator.
    /// Verifies:
    /// 1. Function has no dynamic branches (if conditions based on tensor values)
    /// 2. All tensor shapes in the function are statically known or bounded
    /// 3. Function does not use dynamic data structures (Vec, HashMap, etc.)
    /// 4. If --wcet is enabled, runs the WCET analysis pass
    pub fn check_real_time_function(
        &mut self,
        func: &FunctionDef,
        max_latency_ms: f64,
        device: Option<&str>,
    ) -> Result<(), SemanticError> {
        // 1. Verify no data-dependent branches
        let dynamic_branches = self.find_data_dependent_branches(func);
        if !dynamic_branches.is_empty() {
            return Err(SemanticError::WCETDynamicBranch {
                function: func.name.clone(),
                branches: dynamic_branches,
            });
        }

        // 2. Verify all shapes are static or bounded
        let dynamic_shapes = self.find_dynamic_shapes(func);
        if !dynamic_shapes.is_empty() {
            return Err(SemanticError::WCETDynamicShape {
                function: func.name.clone(),
                shapes: dynamic_shapes,
            });
        }

        // 3. Verify no dynamic data structures
        // (NSL doesn't have general-purpose collections, but check for
        //  variable-length operations like string concatenation)

        Ok(())
    }

    /// Find branches whose condition depends on tensor values (not shapes).
    fn find_data_dependent_branches(&self, func: &FunctionDef) -> Vec<String> {
        let mut violations = Vec::new();
        // Walk AST, find if/while/for nodes whose condition involves
        // tensor element access or runtime-computed values
        // Shape-based branches (e.g., if x.shape[0] > 512) are allowed
        // because shapes are compile-time known
        violations
    }

    /// Find tensor allocations with truly dynamic (unbounded) shapes.
    fn find_dynamic_shapes(&self, func: &FunctionDef) -> Vec<String> {
        let mut violations = Vec::new();
        // Walk AST, find tensor creation expressions where any dimension
        // is Symbolic (not Concrete or Bounded)
        violations
    }
}
```

### Static Control Flow Proof

```rust
// crates/nsl-semantic/src/wcet.rs (continued)

/// Proof that the computation graph has no data-dependent branches.
/// This is required for WCET analysis because data-dependent branches
/// could cause different execution paths with different timing.
pub struct StaticControlFlowProof {
    /// Total branch points in the certified function graph
    pub total_branches: usize,
    /// Branches that depend on tensor values (VIOLATIONS)
    pub data_dependent_branches: usize,
    /// Details of each branch
    pub branches: Vec<BranchAnalysis>,
    /// Whether the proof holds
    pub proven: bool,
}

#[derive(Debug, Clone)]
pub struct BranchAnalysis {
    /// Source location
    pub source_loc: String,
    /// The condition expression
    pub condition: String,
    /// Whether this branch is data-dependent
    pub is_data_dependent: bool,
    /// Why this branch is considered static or dynamic
    pub rationale: String,
}
```

---

## Section 10: Type System

### WCET-Aware Type Extensions

The type system enforces WCET constraints by rejecting operations that cannot have bounded timing:

```rust
// Extension to crates/nsl-semantic/src/types.rs

/// WCET annotation on a function type.
#[derive(Debug, Clone)]
pub enum WCETAnnotation {
    /// No WCET constraint
    None,
    /// @real_time(max_latency_ms=N)
    RealTime {
        max_latency_ms: f64,
        device: Option<String>,
    },
    /// @wcet_budget(max_cycles=N)
    CycleBudget {
        max_cycles: u64,
    },
}
```

Operations banned inside `@real_time` functions:
- Dynamic tensor creation (shapes unknown at compile time)
- While loops with data-dependent termination conditions
- Recursive function calls (unbounded depth)
- String operations (variable-length)
- `print()` / `log()` (I/O has unbounded latency)
- `py.call()` (Python FFI has unbounded latency)

---

## Section 11: Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_wcet_matmul_gpu_h100` | Matmul [1, 4096] x [4096, 4096] on H100. Verify WCET is within expected range (analytical formula). |
| `test_wcet_matmul_gpu_orin` | Same matmul on Jetson Orin. Verify WCET is ~30x higher than H100 (proportional to FLOPS ratio). |
| `test_wcet_elementwise_gpu` | ReLU on [1, 1048576] elements. Verify WCET is memory-bound (compute < memory time). |
| `test_wcet_matmul_cpu` | Matmul [1, 512] x [512, 512] on Cortex-A78. Verify WCET uses single-core worst case. |
| `test_cache_miss_sequential` | Sequential access of 1MB buffer with 64B cache lines. Verify misses = 16384. |
| `test_cache_miss_matmul` | MatmulRowMajor with 512x512 FP32. Verify working set = 3MB, cache miss bound correct. |
| `test_no_heap_proof_pass` | Function with all slab-planned tensors. Verify `proven = true`. |
| `test_no_heap_proof_fail` | Function with one dynamic-size tensor. Verify `proven = false`, violation reported. |
| `test_static_cf_proof_pass` | Function with only shape-based branches. Verify `proven = true`. |
| `test_static_cf_proof_fail` | Function with `if tensor[0] > 0.5`. Verify `proven = false`. |
| `test_safety_margin` | WCET of 10.0ms with 5% margin = 10.5ms. Bound of 11.0ms passes; bound of 10.2ms fails. |
| `test_certificate_json` | Generate certificate, parse JSON, verify all fields present and types correct. |

### E2E Tests

| Test | NSL Source | Description |
|------|-----------|-------------|
| `examples/m53_wcet_pass.nsl` | Small CNN with `@real_time(max_latency_ms=100.0)`. | Compilation succeeds, WCET certificate generated, bound satisfied. |
| `examples/m53_wcet_fail.nsl` | Large model with `@real_time(max_latency_ms=0.1)` (impossible bound). | Compilation fails with structured WCET violation error. |
| `examples/m53_wcet_cpu.nsl` | Tiny MLP with `@real_time(max_latency_ms=5.0, device="cpu")`. | WCET computed for CPU, certificate shows CPU timing. |
| `examples/m53_no_heap.nsl` | Model with all static shapes. | No-heap proof passes. |
| `examples/m53_dynamic_branch.nsl` | Model with `if relu(x).sum() > 0`. | Compile error: data-dependent branch in @real_time function. |
| `examples/m53_do178c.nsl` | Model with `--do178c-report`. | DO-178C directory created with SRS, SDD, and verification files. |
| `examples/m53_budget_per_fn.nsl` | Model with `@wcet_budget(max_cycles=1000000)` on subfunctions. | Per-function budgets checked and reported. |

### Validation Strategy

WCET proofs are conservative by design (overestimate). To validate that they are not *too* conservative:

1. For each E2E test model, run actual inference 10,000 times on the target hardware.
2. Record the maximum observed latency.
3. Verify that computed WCET > max observed latency (proof is sound).
4. Verify that computed WCET < 5x max observed latency (proof is not excessively loose).

---

## Section 12: Modified Files

### New Files

| File | Responsibility |
|------|----------------|
| `crates/nsl-semantic/src/wcet.rs` | WCETAnalyzer pass, hardware cost database, per-op cycle counting, no-heap proof, static control flow proof, cache miss analysis |
| `crates/nsl-codegen/src/wcet_report.rs` | WCETCertificate JSON generation, DO-178C report stub |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-codegen/src/compiler.rs` | `wcet_config: Option<WCETConfig>` field; invoke `WCETAnalyzer` after `WeightAwarePass` and `MemoryPlanner`; emit certificate |
| `crates/nsl-codegen/src/lib.rs` | `mod wcet_report;` declaration; `wcet`, `wcet_cert`, `do178c_report` in `CompileOptions` |
| `crates/nsl-semantic/src/checker.rs` | `check_real_time_function()`, `check_wcet_budget()` validation; reject banned operations inside `@real_time` |
| `crates/nsl-semantic/src/lib.rs` | `mod wcet;` declaration |
| `crates/nsl-parser/src/parser.rs` | Parse `@real_time(max_latency_ms=N)`, `@wcet_budget(max_cycles=N)` decorators |
| `crates/nsl-cli/src/main.rs` | `--wcet`, `--wcet-cert`, `--gpu`, `--cpu`, `--do178c-report`, `nsl check --wcet` subcommand |

---

## Section 13: Deliverables

1. `WCETAnalyzer` pass that computes worst-case execution time for every tensor operation in a certified function
2. Hardware cost database with worst-case cycle counts for H100, A100, Jetson Orin, Cortex-A78, x86-64-v4
3. `@real_time(max_latency_ms=N)` decorator with compile-time enforcement
4. `@wcet_budget(max_cycles=N)` per-function cycle budget decorator
5. No-heap-allocation proof that verifies all tensor memory is slab-planned
6. Static control flow proof that verifies no data-dependent branches
7. Cache miss bound analysis for matmul and elementwise access patterns
8. WCET certificate JSON with per-operation timing breakdown and proof chain
9. DO-178C documentation generation stub (SRS, SDD, verification files)
10. `nsl build --wcet` and `nsl check --wcet` CLI commands
11. Safety margin configuration (default 5%)
12. Structured error output with per-operation timing and optimization suggestions on WCET violation

## Out of Scope

- Runtime WCET monitoring or enforcement (the proof is compile-time only; no runtime instrumentation)
- Multi-GPU WCET (WCET is for single-device execution; multi-GPU adds NCCL communication variance)
- Training WCET (only inference forward pass is certifiable; backward pass has variable tape length)
- WCET for dynamic batching (batch size must be fixed for WCET; continuous batching from M29 is excluded)
- Formal verification of the WCET proof itself (the certificate is evidence for human auditors, not a machine-checked proof)
- WCET for Python FFI calls (`py.call()` is banned in `@real_time` functions)
- Thermal throttling modeling (base clock is used as worst case, but sustained thermal throttling is not modeled)
- NUMA topology effects on CPU WCET (single-socket assumption)
- Interrupt handling latency (assumes interrupt-free execution; unikernel mode from M54 provides this guarantee)

## Success Criteria

1. A small CNN model with `@real_time(max_latency_ms=100.0)` on H100 compiles successfully with a WCET certificate proving the bound holds.
2. A model whose WCET exceeds the declared bound produces a structured compile error with per-operation timing breakdown and actionable suggestions.
3. The no-heap-allocation proof correctly identifies functions with dynamic tensor allocations.
4. The static control flow proof correctly rejects functions with data-dependent branches.
5. WCET certificate JSON is valid, parseable, and contains all required fields.
6. Computed WCET values for known operations match analytical expectations within 10%.
7. DO-178C documentation stub is generated with correct content.
8. The WCET for a model on Jetson Orin is correctly higher than on H100 (proportional to hardware specs).
