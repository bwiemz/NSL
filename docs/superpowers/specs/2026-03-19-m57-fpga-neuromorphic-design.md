# M57: FPGA & Neuromorphic Backend — Design Specification

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M57 (Phase 13, v1.2.0)
**Prerequisites:** M47 (Multi-Backend Targeting — KIR infrastructure), M36 (Memory Planning — static allocation for FPGA BRAM mapping), M37 (Roofline Cost Model — resource budgeting)
**Dependencies:** M47 provides the Kernel IR abstraction that M57 extends to Verilog/Groq/SpiNNaker; M36's static memory plan maps directly to FPGA BRAM allocation; M37's cost model extends to FPGA resource budgets (LUT/BRAM/DSP counts); M53 (WCET Proofs) benefits from FPGA's deterministic execution

## Overview

M57 extends NSL's Kernel IR (KIR, from M47) to emit hardware description language (Verilog) for FPGA synthesis, specialized assembly for Groq's Linear Processing Unit (LPU/TSP), and spike-timing instructions for neuromorphic processors (SpiNNaker/Loihi). This transforms NSL from a GPU-focused ML compiler into a universal AI hardware frontend — any hardware startup can write a KIR backend and immediately get access to NSL's entire model ecosystem.

The key design insight is that KIR is already a flat, SSA-form representation with explicit memory access patterns and statically-known computation graphs. This maps remarkably well to hardware description: each KIR operation becomes a hardware functional unit, KIR basic blocks become pipeline stages, and KIR's `AddressSpace::Shared` maps directly to FPGA Block RAM. The static memory plan from M36 provides exact BRAM allocation — the compiler can prove at build time that the design fits within the FPGA's resource limits.

For FPGAs, the compiler partitions matmul operations into systolic array tiles that map to the FPGA fabric's DSP blocks and LUTs. The `nsl build --target fpga --device xilinx-u280` command produces a Verilog design ready for synthesis with Xilinx Vivado or Intel Quartus. Resource budgeting is a compile-time check: if the design exceeds the target FPGA's LUT, BRAM, or DSP count, the compiler reports an error before synthesis begins.

For Groq's TSP, the compiler exploits KIR's linear execution model — TSP executes instructions in a deterministic pipeline without branches, which is exactly what KIR basic blocks represent when branch-free (common in inference). The Groq backend maps KIR operations to TSP assembly instructions with statically-scheduled memory accesses.

For neuromorphic processors (SpiNNaker, Intel Loihi), the compiler emits spike-timing format — converting rate-coded activations to spike trains. This is a stub implementation targeting future collaboration with neuromorphic hardware teams.

**Why this is unique to NSL:** Python cannot target FPGAs because FPGA synthesis requires a static hardware description — no dynamic dispatch, no garbage collection, no interpreter. Existing HLS tools (Vitis HLS, Intel HLS) operate on C/C++ code and require manual optimization for efficient FPGA utilization. NSL's static IR, combined with compile-time shape checking and memory planning, provides the exact information FPGA synthesis needs without manual annotation.

---

## Section 1: Language Surface

### 1.1 CLI Interface

```bash
# FPGA synthesis
nsl build --target fpga --device xilinx-u280 model.nsl
nsl build --target fpga --device xilinx-u55c model.nsl
nsl build --target fpga --device intel-agilex model.nsl

# Groq TSP compilation
nsl build --target groq model.nsl

# Neuromorphic (stub)
nsl build --target neuromorphic --device spinnaker model.nsl
nsl build --target neuromorphic --device loihi model.nsl

# Resource budget check (no synthesis — just validate fitting)
nsl build --target fpga --device xilinx-u280 --resource-check model.nsl

# Simulation (generate Verilog testbench)
nsl build --target fpga --device xilinx-u280 --testbench model.nsl

# Multi-target (FPGA + GPU fallback)
nsl build --target fpga,cuda --device xilinx-u280 model.nsl
```

### 1.2 Target-Specific Annotations

```
# FPGA-specific: control systolic array tiling
@fpga_tile(rows=16, cols=16, depth=8)
kernel matmul_tile(
    A: Tensor<[16, 8], int8>,
    B: Tensor<[8, 16], int8>,
) -> Tensor<[16, 16], int32>:
    # Each element: C[i,j] = sum_k A[i,k] * B[k,j]
    for i in range(16):
        for j in range(16):
            let acc = 0
            for k in range(8):
                acc += A[i, k] * B[k, j]
            out[i, j] = acc

# FPGA-specific: clock frequency target
@fpga_clock(mhz=200)
model TinyClassifier:
    W1: Tensor<[784, 128], int8>
    b1: Tensor<[128], int8>
    W2: Tensor<[128, 10], int8>
    b2: Tensor<[10], int8>

    fn forward(self, x: Tensor<[1, 784], int8>) -> Tensor<[1, 10], int32>:
        let h = relu(matmul(x, self.W1) + self.b1)
        return matmul(h, self.W2) + self.b2

# Groq-specific: specify TSP instruction scheduling
@groq_stream(lanes=320)
kernel vector_add(a: Tensor<[N], f32>, b: Tensor<[N], f32>) -> Tensor<[N], f32>:
    let tid = thread_id()
    out[tid] = a[tid] + b[tid]
```

### 1.3 FPGA Resource Constraints

```
# Declare resource budget — compile error if exceeded
@fpga_resources(
    luts=1200000,      # Xilinx U280: 1,304,000 LUTs available
    bram_kb=4096,      # 4 MB of Block RAM
    dsps=9024,         # 9,024 DSP48 slices
    uram_kb=0,         # no UltraRAM used
)
model TinyClassifier:
    ...
```

```
# Automatic resource detection from device database
@fpga_device("xilinx-u280")    # compiler looks up resource limits
model TinyClassifier:
    ...
```

### 1.4 FPGA Pipeline Pragma

```
# Control pipeline initiation interval (II)
@fpga_pipeline(ii=1)    # fully pipelined — new input every clock cycle
fn systolic_mac(a: int8, b: int8, acc: int32) -> int32:
    return acc + a * b

@fpga_pipeline(ii=4)    # new input every 4 cycles (saves resources)
fn matmul_row(row: Tensor<[K], int8>, col: Tensor<[K], int8>) -> int32:
    let acc = 0
    for k in range(K):
        acc = systolic_mac(row[k], col[k], acc)
    return acc
```

### 1.5 Latency-Deterministic Execution

FPGA execution has zero variance — every inference takes exactly the same number of clock cycles. This feeds directly into M53's WCET proofs:

```
@real_time(max_latency_ms=0.5)    # M53 WCET annotation
@fpga_clock(mhz=200)
@fpga_device("xilinx-u280")
model SafetyClassifier:
    ...

    fn forward(self, x: Tensor<[1, 784], int8>) -> Tensor<[1, 10], int32>:
        ...
        # Compiler proves: forward pass takes exactly 4,500 cycles @ 200 MHz = 22.5 us
        # 22.5 us < 500 us (0.5 ms) -> WCET check passes
```

---

## Section 2: Architecture

### 2.1 Compilation Pipeline

```
NSL Source → Lexer → Parser → Type Checker → Shape Checker
                                                   ↓
                                         Memory Planner (M36)
                                                   ↓
                                         KIR Lowering (M47)
                                                   ↓
                                    ┌──────────────┼──────────────┐
                                    ↓              ↓              ↓
                              Verilog         Groq TSP       SpiNNaker
                              Backend         Backend         Backend
                                    ↓              ↓              ↓
                              .v files       .tsp files      .spinn files
                                    ↓
                            Resource Budget Check
                                    ↓
                         Vivado/Quartus synthesis (external tool)
```

### 2.2 FPGA Device Database

```rust
/// FPGA device specifications for resource budgeting.
#[derive(Debug, Clone)]
pub struct FpgaDevice {
    /// Device name (e.g., "xilinx-u280", "intel-agilex").
    pub name: String,

    /// Device family.
    pub family: FpgaFamily,

    /// Available resources.
    pub resources: FpgaResources,

    /// Maximum clock frequency (MHz).
    pub max_clock_mhz: u32,

    /// Memory interface (DDR4 bandwidth, HBM channels, etc.).
    pub memory: FpgaMemory,
}

#[derive(Debug, Clone, Copy)]
pub enum FpgaFamily {
    XilinxUltrascalePlus,
    XilinxVersal,
    IntelAgilex,
    IntelStratix,
    LatticeCrossLink,
}

/// FPGA resource counts.
#[derive(Debug, Clone)]
pub struct FpgaResources {
    /// Lookup Tables (LUTs) — the fundamental logic building block.
    pub luts: u64,

    /// Flip-Flops (FFs) — register storage.
    pub flip_flops: u64,

    /// Block RAM (BRAM) in kilobytes — on-chip SRAM.
    pub bram_kb: u64,

    /// UltraRAM (URAM) in kilobytes — larger on-chip memory (Xilinx only).
    pub uram_kb: u64,

    /// DSP slices — fixed-function multiply-accumulate units.
    pub dsps: u64,

    /// High-Bandwidth Memory (HBM) channels (0 if no HBM).
    pub hbm_channels: u32,
}

/// FPGA external memory specification.
#[derive(Debug, Clone)]
pub struct FpgaMemory {
    /// DDR4 bandwidth (GB/s).
    pub ddr4_bandwidth_gbps: f64,

    /// HBM bandwidth (GB/s), 0 if no HBM.
    pub hbm_bandwidth_gbps: f64,

    /// Total external memory (GB).
    pub total_memory_gb: f64,
}

/// Built-in device database.
pub fn get_device(name: &str) -> Option<FpgaDevice> {
    match name {
        "xilinx-u280" => Some(FpgaDevice {
            name: "xilinx-u280".to_string(),
            family: FpgaFamily::XilinxUltrascalePlus,
            resources: FpgaResources {
                luts: 1_304_000,
                flip_flops: 2_607_360,
                bram_kb: 4_032,    // 2,016 BRAM18K blocks = ~4 MB
                uram_kb: 36_000,   // 960 URAM blocks = ~36 MB
                dsps: 9_024,
                hbm_channels: 32,
            },
            max_clock_mhz: 300,
            memory: FpgaMemory {
                ddr4_bandwidth_gbps: 77.0,
                hbm_bandwidth_gbps: 460.0,
                total_memory_gb: 8.0,  // 8 GB HBM2
            },
        }),
        "xilinx-u55c" => Some(FpgaDevice {
            name: "xilinx-u55c".to_string(),
            family: FpgaFamily::XilinxUltrascalePlus,
            resources: FpgaResources {
                luts: 1_304_000,
                flip_flops: 2_607_360,
                bram_kb: 4_032,
                uram_kb: 36_000,
                dsps: 9_024,
                hbm_channels: 32,
            },
            max_clock_mhz: 300,
            memory: FpgaMemory {
                ddr4_bandwidth_gbps: 0.0,
                hbm_bandwidth_gbps: 460.0,
                total_memory_gb: 16.0,  // 16 GB HBM2
            },
        }),
        "intel-agilex" => Some(FpgaDevice {
            name: "intel-agilex".to_string(),
            family: FpgaFamily::IntelAgilex,
            resources: FpgaResources {
                luts: 2_505_000,
                flip_flops: 5_010_000,
                bram_kb: 45_000,   // M20K blocks
                uram_kb: 0,
                dsps: 5_760,       // Intel AI Tensor Blocks
                hbm_channels: 0,
            },
            max_clock_mhz: 400,
            memory: FpgaMemory {
                ddr4_bandwidth_gbps: 102.0,
                hbm_bandwidth_gbps: 0.0,
                total_memory_gb: 32.0,
            },
        }),
        _ => None,
    }
}
```

### 2.3 Core Data Structures

New module: `crates/nsl-codegen/src/backend_verilog.rs`

```rust
/// A Verilog module generated from KIR.
#[derive(Debug)]
pub struct VerilogModule {
    /// Module name.
    pub name: String,

    /// Input ports.
    pub inputs: Vec<VerilogPort>,

    /// Output ports.
    pub outputs: Vec<VerilogPort>,

    /// Internal wires and registers.
    pub signals: Vec<VerilogSignal>,

    /// Combinational and sequential logic blocks.
    pub logic_blocks: Vec<VerilogLogic>,

    /// Submodule instantiations (for systolic array PEs, etc.).
    pub instances: Vec<VerilogInstance>,

    /// Resource usage estimate.
    pub estimated_resources: ResourceEstimate,

    /// Pipeline latency in clock cycles.
    pub latency_cycles: u64,

    /// Clock frequency target (MHz).
    pub target_clock_mhz: u32,
}

#[derive(Debug, Clone)]
pub struct VerilogPort {
    pub name: String,
    pub direction: PortDirection,
    pub width: u32,         // bit width
    pub is_signed: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PortDirection {
    Input,
    Output,
    InOut,
}

#[derive(Debug, Clone)]
pub struct VerilogSignal {
    pub name: String,
    pub signal_type: SignalType,
    pub width: u32,
    pub is_signed: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum SignalType {
    Wire,
    Reg,
    Integer,
}

#[derive(Debug, Clone)]
pub enum VerilogLogic {
    /// Combinational: assign out = a + b;
    Assign {
        target: String,
        expr: VerilogExpr,
    },
    /// Sequential: always @(posedge clk) begin ... end
    AlwaysFF {
        clock: String,
        reset: Option<String>,
        body: Vec<VerilogStatement>,
    },
    /// Combinational block: always @(*) begin ... end
    AlwaysComb {
        body: Vec<VerilogStatement>,
    },
}

#[derive(Debug, Clone)]
pub enum VerilogExpr {
    Ident(String),
    Literal(i64, u32),   // value, bit width
    BinOp(Box<VerilogExpr>, VerilogBinOp, Box<VerilogExpr>),
    UnaryOp(VerilogUnaryOp, Box<VerilogExpr>),
    Index(Box<VerilogExpr>, Box<VerilogExpr>),
    Slice(Box<VerilogExpr>, u32, u32),  // signal[high:low]
    Concat(Vec<VerilogExpr>),            // {a, b, c}
    Ternary(Box<VerilogExpr>, Box<VerilogExpr>, Box<VerilogExpr>),  // cond ? a : b
}

#[derive(Debug, Clone, Copy)]
pub enum VerilogBinOp {
    Add, Sub, Mul, Div,
    And, Or, Xor,
    Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
}

#[derive(Debug, Clone, Copy)]
pub enum VerilogUnaryOp {
    Not, Neg,
}

#[derive(Debug, Clone)]
pub enum VerilogStatement {
    Assign { target: String, expr: VerilogExpr },
    If { cond: VerilogExpr, then_body: Vec<VerilogStatement>, else_body: Option<Vec<VerilogStatement>> },
    Case { selector: VerilogExpr, arms: Vec<(VerilogExpr, Vec<VerilogStatement>)> },
    ForLoop { var: String, start: i64, end: i64, body: Vec<VerilogStatement> },
}

#[derive(Debug, Clone)]
pub struct VerilogInstance {
    pub module_name: String,
    pub instance_name: String,
    pub port_connections: Vec<(String, VerilogExpr)>,
}

/// Resource usage estimate for the generated design.
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    pub luts: u64,
    pub flip_flops: u64,
    pub bram_kb: u64,
    pub dsps: u64,
    pub uram_kb: u64,
}
```

### 2.4 Component Map

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `VerilogBackend` | `crates/nsl-codegen/src/backend_verilog.rs` | KIR to Verilog translation, systolic array generation |
| `GroqBackend` | `crates/nsl-codegen/src/backend_groq.rs` | KIR to Groq TSP assembly |
| `NeuromorphicBackend` | `crates/nsl-codegen/src/backend_neuromorphic.rs` | KIR to SpiNNaker spike-timing format (stub) |
| `FpgaResourceChecker` | `crates/nsl-codegen/src/fpga_resource.rs` | Resource budgeting: prove design fits target FPGA |
| `SystolicMapper` | `crates/nsl-codegen/src/fpga_systolic.rs` | Partition matmul into systolic array tiles |
| `FpgaDevice` database | `crates/nsl-codegen/src/fpga_devices.rs` | Built-in FPGA device specifications |
| `FpgaRuntime` | `crates/nsl-runtime/src/fpga_runtime.rs` | Host-side FPGA communication (XDMA, OpenCL) |
| CLI integration | `crates/nsl-cli/src/main.rs` | `--target fpga/groq/neuromorphic`, `--device`, `--resource-check`, `--testbench` |

---

## Section 3: KIR-to-Verilog Lowering

### 3.1 Lowering Strategy

Each KIR operation maps to a Verilog hardware unit:

| KIR Operation | Verilog Implementation | Resource Cost |
|---------------|----------------------|---------------|
| `Add(dst, a, b)` | `assign dst = a + b;` | ~1 LUT per bit |
| `Sub(dst, a, b)` | `assign dst = a - b;` | ~1 LUT per bit |
| `Mul(dst, a, b)` | DSP48 slice instantiation | 1 DSP per 8x8 multiply |
| `Fma(dst, a, b, c)` | DSP48 MACC mode | 1 DSP (multiply-accumulate) |
| `Load(dst, ptr, addr)` | BRAM read port | 0 LUTs, 1 BRAM port |
| `Store(ptr, val, addr)` | BRAM write port | 0 LUTs, 1 BRAM port |
| `Barrier` | Pipeline register stage | 1 FF per signal bit |
| `ThreadId(dst, dim)` | Counter module | ~32 LUTs |
| `Cmp(dst, a, b, op)` | Comparator | ~1 LUT per bit |
| `Select(dst, cond, a, b)` | Mux: `assign dst = cond ? a : b;` | ~1 LUT per bit |
| `Const(dst, val)` | Wire tied to constant | 0 LUTs |

### 3.2 Verilog Backend Implementation

```rust
/// Translates KIR to Verilog hardware description.
pub struct VerilogBackend {
    /// Target FPGA device (for resource limits and feature set).
    device: FpgaDevice,

    /// Target clock frequency.
    target_clock_mhz: u32,

    /// Pipeline initiation interval.
    pipeline_ii: u32,

    /// Generated Verilog modules.
    modules: Vec<VerilogModule>,

    /// Next unique signal name index.
    signal_counter: u64,
}

impl VerilogBackend {
    pub fn new(device: FpgaDevice, target_clock_mhz: u32) -> Self {
        Self {
            device,
            target_clock_mhz,
            pipeline_ii: 1,
            modules: Vec::new(),
            signal_counter: 0,
        }
    }

    /// Lower a KIR kernel to a Verilog module.
    pub fn lower_kernel(&mut self, ir: &KernelIR) -> Result<VerilogModule, FpgaError> {
        let mut module = VerilogModule {
            name: ir.name.clone(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            signals: Vec::new(),
            logic_blocks: Vec::new(),
            instances: Vec::new(),
            estimated_resources: ResourceEstimate::default(),
            latency_cycles: 0,
            target_clock_mhz: self.target_clock_mhz,
        };

        // Add standard ports: clk, rst_n, start, done, valid
        module.inputs.push(VerilogPort {
            name: "clk".to_string(),
            direction: PortDirection::Input,
            width: 1,
            is_signed: false,
        });
        module.inputs.push(VerilogPort {
            name: "rst_n".to_string(),
            direction: PortDirection::Input,
            width: 1,
            is_signed: false,
        });
        module.inputs.push(VerilogPort {
            name: "start".to_string(),
            direction: PortDirection::Input,
            width: 1,
            is_signed: false,
        });
        module.outputs.push(VerilogPort {
            name: "done".to_string(),
            direction: PortDirection::Output,
            width: 1,
            is_signed: false,
        });
        module.outputs.push(VerilogPort {
            name: "valid".to_string(),
            direction: PortDirection::Output,
            width: 1,
            is_signed: false,
        });

        // Map KIR parameters to Verilog ports
        for param in &ir.params {
            let port = self.lower_param(param);
            module.inputs.push(port);
        }

        // Lower each KIR block to Verilog logic
        for block in &ir.blocks {
            self.lower_block(block, &mut module)?;
        }

        // Estimate resource usage
        module.estimated_resources = self.estimate_resources(&module);

        // Compute pipeline latency
        module.latency_cycles = self.compute_latency(&module);

        Ok(module)
    }

    /// Lower a KIR basic block to Verilog logic.
    fn lower_block(
        &mut self,
        block: &KirBlock,
        module: &mut VerilogModule,
    ) -> Result<(), FpgaError> {
        for op in &block.ops {
            self.lower_op(op, module)?;
        }

        // Lower terminator
        match &block.terminator {
            KirTerminator::Return => {
                // Set done signal high
                module.logic_blocks.push(VerilogLogic::Assign {
                    target: "done".to_string(),
                    expr: VerilogExpr::Literal(1, 1),
                });
            }
            KirTerminator::Branch(target) => {
                // Pipeline register: insert FF stage between blocks
                module.signals.push(VerilogSignal {
                    name: format!("stage_{}_valid", target),
                    signal_type: SignalType::Reg,
                    width: 1,
                    is_signed: false,
                });
            }
            KirTerminator::CondBranch(cond, then_block, else_block) => {
                // Mux between two pipeline paths
                let cond_name = self.wire_name(*cond);
                module.signals.push(VerilogSignal {
                    name: format!("branch_sel_{}", cond_name),
                    signal_type: SignalType::Wire,
                    width: 1,
                    is_signed: false,
                });
            }
        }

        Ok(())
    }

    /// Lower a single KIR operation to Verilog.
    fn lower_op(
        &mut self,
        op: &KirOp,
        module: &mut VerilogModule,
    ) -> Result<(), FpgaError> {
        match op {
            KirOp::Add(dst, a, b) => {
                let dst_name = self.wire_name(*dst);
                let a_name = self.wire_name(*a);
                let b_name = self.wire_name(*b);

                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width: 32, // determined by type inference
                    is_signed: true,
                });

                module.logic_blocks.push(VerilogLogic::Assign {
                    target: dst_name,
                    expr: VerilogExpr::BinOp(
                        Box::new(VerilogExpr::Ident(a_name)),
                        VerilogBinOp::Add,
                        Box::new(VerilogExpr::Ident(b_name)),
                    ),
                });
            }

            KirOp::Mul(dst, a, b) => {
                let dst_name = self.wire_name(*dst);
                let a_name = self.wire_name(*a);
                let b_name = self.wire_name(*b);

                // For 8-bit multiply: use DSP48 slice
                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width: 32,
                    is_signed: true,
                });

                // Instantiate DSP48 for the multiply
                module.instances.push(VerilogInstance {
                    module_name: "dsp48_multiply".to_string(),
                    instance_name: format!("mul_{}", dst_name),
                    port_connections: vec![
                        ("clk".to_string(), VerilogExpr::Ident("clk".to_string())),
                        ("a".to_string(), VerilogExpr::Ident(a_name)),
                        ("b".to_string(), VerilogExpr::Ident(b_name)),
                        ("p".to_string(), VerilogExpr::Ident(dst_name)),
                    ],
                });
            }

            KirOp::Fma(dst, a, b, c) => {
                let dst_name = self.wire_name(*dst);
                let a_name = self.wire_name(*a);
                let b_name = self.wire_name(*b);
                let c_name = self.wire_name(*c);

                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width: 32,
                    is_signed: true,
                });

                // DSP48 in MACC mode: P = A * B + C
                module.instances.push(VerilogInstance {
                    module_name: "dsp48_macc".to_string(),
                    instance_name: format!("fma_{}", dst_name),
                    port_connections: vec![
                        ("clk".to_string(), VerilogExpr::Ident("clk".to_string())),
                        ("a".to_string(), VerilogExpr::Ident(a_name)),
                        ("b".to_string(), VerilogExpr::Ident(b_name)),
                        ("c".to_string(), VerilogExpr::Ident(c_name)),
                        ("p".to_string(), VerilogExpr::Ident(dst_name)),
                    ],
                });
            }

            KirOp::Load(dst, addr, address_space) => {
                let dst_name = self.wire_name(*dst);
                let addr_name = self.wire_name(*addr);

                let mem_type = match address_space {
                    AddressSpace::Shared => "bram",
                    AddressSpace::Global => "ddr",
                    AddressSpace::Constant => "rom",
                    AddressSpace::Local => "reg_file",
                };

                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width: 32,
                    is_signed: true,
                });

                // Memory read: connect to BRAM/DDR read port
                module.logic_blocks.push(VerilogLogic::Assign {
                    target: dst_name,
                    expr: VerilogExpr::Index(
                        Box::new(VerilogExpr::Ident(format!("{}_mem", mem_type))),
                        Box::new(VerilogExpr::Ident(addr_name)),
                    ),
                });
            }

            KirOp::Store(addr, val, address_space) => {
                let addr_name = self.wire_name(*addr);
                let val_name = self.wire_name(*val);

                // Memory write: connect to BRAM/DDR write port
                module.logic_blocks.push(VerilogLogic::AlwaysFF {
                    clock: "clk".to_string(),
                    reset: Some("rst_n".to_string()),
                    body: vec![VerilogStatement::Assign {
                        target: format!("mem[{}]", addr_name),
                        expr: VerilogExpr::Ident(val_name),
                    }],
                });
            }

            KirOp::Cmp(dst, a, b, cmp_op) => {
                let dst_name = self.wire_name(*dst);
                let a_name = self.wire_name(*a);
                let b_name = self.wire_name(*b);

                let verilog_op = match cmp_op {
                    CmpOp::Eq => VerilogBinOp::Eq,
                    CmpOp::Ne => VerilogBinOp::Ne,
                    CmpOp::Lt => VerilogBinOp::Lt,
                    CmpOp::Le => VerilogBinOp::Le,
                    CmpOp::Gt => VerilogBinOp::Gt,
                    CmpOp::Ge => VerilogBinOp::Ge,
                };

                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width: 1,
                    is_signed: false,
                });

                module.logic_blocks.push(VerilogLogic::Assign {
                    target: dst_name,
                    expr: VerilogExpr::BinOp(
                        Box::new(VerilogExpr::Ident(a_name)),
                        verilog_op,
                        Box::new(VerilogExpr::Ident(b_name)),
                    ),
                });
            }

            KirOp::Select(dst, cond, a, b) => {
                let dst_name = self.wire_name(*dst);
                let cond_name = self.wire_name(*cond);
                let a_name = self.wire_name(*a);
                let b_name = self.wire_name(*b);

                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width: 32,
                    is_signed: true,
                });

                module.logic_blocks.push(VerilogLogic::Assign {
                    target: dst_name,
                    expr: VerilogExpr::Ternary(
                        Box::new(VerilogExpr::Ident(cond_name)),
                        Box::new(VerilogExpr::Ident(a_name)),
                        Box::new(VerilogExpr::Ident(b_name)),
                    ),
                });
            }

            KirOp::Const(dst, konst) => {
                let dst_name = self.wire_name(*dst);
                let (value, width) = match &konst.value {
                    ConstValue::I32(v) => (*v as i64, 32),
                    ConstValue::U32(v) => (*v as i64, 32),
                    ConstValue::F32(_) => {
                        return Err(FpgaError::UnsupportedType("f32 constants require fixed-point conversion".to_string()));
                    }
                    ConstValue::F64(_) => {
                        return Err(FpgaError::UnsupportedType("f64 not supported on FPGA".to_string()));
                    }
                };

                module.signals.push(VerilogSignal {
                    name: dst_name.clone(),
                    signal_type: SignalType::Wire,
                    width,
                    is_signed: true,
                });

                module.logic_blocks.push(VerilogLogic::Assign {
                    target: dst_name,
                    expr: VerilogExpr::Literal(value, width),
                });
            }

            KirOp::Barrier => {
                // Insert pipeline register stage
                // All signals from previous stage are registered (flip-flop)
                module.logic_blocks.push(VerilogLogic::AlwaysFF {
                    clock: "clk".to_string(),
                    reset: Some("rst_n".to_string()),
                    body: vec![], // filled by pipeline scheduler
                });
            }

            _ => {
                return Err(FpgaError::UnsupportedOp(format!("{:?}", op)));
            }
        }

        Ok(())
    }

    /// Generate unique wire name for a KIR variable.
    fn wire_name(&mut self, var: VarId) -> String {
        format!("w_{}", var)
    }

    /// Lower a KIR parameter to a Verilog port.
    fn lower_param(&self, param: &KirParam) -> VerilogPort {
        let width = match &param.ty {
            KirType::I32 | KirType::U32 | KirType::F32 => 32,
            KirType::I64 | KirType::U64 | KirType::F64 => 64,
            KirType::F16 | KirType::Bf16 => 16,
            KirType::Ptr(_, _) => 64,  // address width
            KirType::Vec(inner, count) => self.type_width(inner) * count,
        };

        VerilogPort {
            name: param.name.clone(),
            direction: PortDirection::Input,
            width,
            is_signed: matches!(param.ty, KirType::I32 | KirType::I64),
        }
    }

    fn type_width(&self, ty: &KirType) -> u32 {
        match ty {
            KirType::U32 | KirType::I32 | KirType::F32 => 32,
            KirType::U64 | KirType::I64 | KirType::F64 => 64,
            KirType::F16 | KirType::Bf16 => 16,
            KirType::Ptr(_, _) => 64,
            KirType::Vec(inner, count) => self.type_width(inner) * count,
        }
    }
}
```

### 3.3 Verilog Serialization

```rust
/// Serialize a VerilogModule to Verilog source text.
pub fn serialize_verilog(module: &VerilogModule) -> String {
    let mut out = String::new();

    // Header comment
    out.push_str(&format!(
        "// Generated by NeuralScript M57 FPGA Backend\n\
         // Module: {}\n\
         // Target clock: {} MHz\n\
         // Estimated latency: {} cycles ({:.3} us)\n\
         // Estimated resources: {} LUTs, {} DSPs, {} KB BRAM\n\n",
        module.name,
        module.target_clock_mhz,
        module.latency_cycles,
        module.latency_cycles as f64 / module.target_clock_mhz as f64,
        module.estimated_resources.luts,
        module.estimated_resources.dsps,
        module.estimated_resources.bram_kb,
    ));

    // Module declaration
    out.push_str(&format!("`timescale 1ns / 1ps\n\n"));
    out.push_str(&format!("module {} (\n", module.name));

    // Port list
    let all_ports: Vec<&VerilogPort> = module.inputs.iter()
        .chain(module.outputs.iter())
        .collect();

    for (i, port) in all_ports.iter().enumerate() {
        let dir = match port.direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
            PortDirection::InOut => "inout",
        };
        let signed = if port.is_signed { " signed" } else { "" };
        let width = if port.width > 1 {
            format!(" [{}:0]", port.width - 1)
        } else {
            String::new()
        };
        let comma = if i < all_ports.len() - 1 { "," } else { "" };
        out.push_str(&format!("    {}{}{} {}{}\n", dir, signed, width, port.name, comma));
    }

    out.push_str(");\n\n");

    // Signal declarations
    for signal in &module.signals {
        let sig_type = match signal.signal_type {
            SignalType::Wire => "wire",
            SignalType::Reg => "reg",
            SignalType::Integer => "integer",
        };
        let signed = if signal.is_signed { " signed" } else { "" };
        let width = if signal.width > 1 {
            format!(" [{}:0]", signal.width - 1)
        } else {
            String::new()
        };
        out.push_str(&format!("    {}{}{} {};\n", sig_type, signed, width, signal.name));
    }

    out.push_str("\n");

    // Logic blocks
    for logic in &module.logic_blocks {
        out.push_str(&serialize_logic(logic, 1));
    }

    // Submodule instances
    for inst in &module.instances {
        out.push_str(&format!("    {} {} (\n", inst.module_name, inst.instance_name));
        for (i, (port, expr)) in inst.port_connections.iter().enumerate() {
            let comma = if i < inst.port_connections.len() - 1 { "," } else { "" };
            out.push_str(&format!("        .{}({}){}\n", port, serialize_expr(expr), comma));
        }
        out.push_str("    );\n\n");
    }

    out.push_str("endmodule\n");
    out
}

fn serialize_logic(logic: &VerilogLogic, indent: usize) -> String {
    let pad = "    ".repeat(indent);
    match logic {
        VerilogLogic::Assign { target, expr } => {
            format!("{}assign {} = {};\n", pad, target, serialize_expr(expr))
        }
        VerilogLogic::AlwaysFF { clock, reset, body } => {
            let mut s = format!("{}always @(posedge {}", pad, clock);
            if let Some(rst) = reset {
                s.push_str(&format!(" or negedge {}", rst));
            }
            s.push_str(") begin\n");
            if let Some(rst) = reset {
                s.push_str(&format!("{}    if (!{}) begin\n", pad, rst));
                s.push_str(&format!("{}        // reset logic\n", pad));
                s.push_str(&format!("{}    end else begin\n", pad));
                for stmt in body {
                    s.push_str(&serialize_statement(stmt, indent + 2));
                }
                s.push_str(&format!("{}    end\n", pad));
            } else {
                for stmt in body {
                    s.push_str(&serialize_statement(stmt, indent + 1));
                }
            }
            s.push_str(&format!("{}end\n\n", pad));
            s
        }
        VerilogLogic::AlwaysComb { body } => {
            let mut s = format!("{}always @(*) begin\n", pad);
            for stmt in body {
                s.push_str(&serialize_statement(stmt, indent + 1));
            }
            s.push_str(&format!("{}end\n\n", pad));
            s
        }
    }
}

fn serialize_expr(expr: &VerilogExpr) -> String {
    match expr {
        VerilogExpr::Ident(name) => name.clone(),
        VerilogExpr::Literal(val, width) => format!("{}'d{}", width, val),
        VerilogExpr::BinOp(a, op, b) => {
            let op_str = match op {
                VerilogBinOp::Add => "+",
                VerilogBinOp::Sub => "-",
                VerilogBinOp::Mul => "*",
                VerilogBinOp::Div => "/",
                VerilogBinOp::And => "&",
                VerilogBinOp::Or => "|",
                VerilogBinOp::Xor => "^",
                VerilogBinOp::Shl => "<<",
                VerilogBinOp::Shr => ">>",
                VerilogBinOp::Eq => "==",
                VerilogBinOp::Ne => "!=",
                VerilogBinOp::Lt => "<",
                VerilogBinOp::Le => "<=",
                VerilogBinOp::Gt => ">",
                VerilogBinOp::Ge => ">=",
            };
            format!("({} {} {})", serialize_expr(a), op_str, serialize_expr(b))
        }
        VerilogExpr::Ternary(cond, a, b) => {
            format!("({} ? {} : {})", serialize_expr(cond), serialize_expr(a), serialize_expr(b))
        }
        VerilogExpr::Index(base, idx) => {
            format!("{}[{}]", serialize_expr(base), serialize_expr(idx))
        }
        VerilogExpr::Slice(base, high, low) => {
            format!("{}[{}:{}]", serialize_expr(base), high, low)
        }
        VerilogExpr::Concat(parts) => {
            let inner: Vec<String> = parts.iter().map(serialize_expr).collect();
            format!("{{{}}}", inner.join(", "))
        }
        VerilogExpr::UnaryOp(op, a) => {
            let op_str = match op {
                VerilogUnaryOp::Not => "~",
                VerilogUnaryOp::Neg => "-",
            };
            format!("({}{})", op_str, serialize_expr(a))
        }
    }
}

fn serialize_statement(stmt: &VerilogStatement, indent: usize) -> String {
    let pad = "    ".repeat(indent);
    match stmt {
        VerilogStatement::Assign { target, expr } => {
            format!("{}{} <= {};\n", pad, target, serialize_expr(expr))
        }
        VerilogStatement::If { cond, then_body, else_body } => {
            let mut s = format!("{}if ({}) begin\n", pad, serialize_expr(cond));
            for stmt in then_body {
                s.push_str(&serialize_statement(stmt, indent + 1));
            }
            if let Some(else_stmts) = else_body {
                s.push_str(&format!("{}end else begin\n", pad));
                for stmt in else_stmts {
                    s.push_str(&serialize_statement(stmt, indent + 1));
                }
            }
            s.push_str(&format!("{}end\n", pad));
            s
        }
        _ => format!("{}// unimplemented statement\n", pad),
    }
}
```

---

## Section 4: Systolic Array Mapping

### 4.1 Matmul Tiling for FPGA

```rust
/// Maps a matmul operation to a systolic array on FPGA fabric.
pub struct SystolicMapper {
    /// Systolic array dimensions.
    pub rows: u32,
    pub cols: u32,
    pub depth: u32,

    /// Data type (determines DSP usage).
    pub dtype: KirType,
}

/// Configuration for a systolic array processing element.
#[derive(Debug, Clone)]
pub struct ProcessingElement {
    /// Position in the array (row, col).
    pub row: u32,
    pub col: u32,

    /// Accumulator width (bits).
    pub acc_width: u32,

    /// Input data width (bits).
    pub data_width: u32,
}

/// A systolic array module for matrix multiplication.
#[derive(Debug)]
pub struct SystolicArray {
    /// Array dimensions.
    pub rows: u32,
    pub cols: u32,

    /// Processing elements.
    pub pes: Vec<ProcessingElement>,

    /// Input streaming interface.
    pub a_stream_width: u32,   // bits per clock for A matrix row
    pub b_stream_width: u32,   // bits per clock for B matrix column

    /// Output draining interface.
    pub c_drain_width: u32,    // bits per clock for C matrix result

    /// Pipeline depth (clock cycles from input to output).
    pub pipeline_depth: u32,

    /// Resource estimate for the entire array.
    pub resources: ResourceEstimate,
}

impl SystolicMapper {
    /// Generate a systolic array Verilog module for the given matmul dimensions.
    pub fn generate_systolic_array(
        &self,
        m: u32, n: u32, k: u32,
    ) -> Result<SystolicArray, FpgaError> {
        // Tile the matmul into systolic array-sized chunks
        let num_tiles_m = (m + self.rows - 1) / self.rows;
        let num_tiles_n = (n + self.cols - 1) / self.cols;
        let num_tiles_k = (k + self.depth - 1) / self.depth;

        let data_width = match self.dtype {
            KirType::I32 => 32,
            KirType::F16 => 16,
            _ => 8, // INT8 default
        };

        let acc_width = data_width * 2 + 8; // enough bits for k accumulations

        // Create processing elements
        let mut pes = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                pes.push(ProcessingElement {
                    row: r,
                    col: c,
                    acc_width,
                    data_width,
                });
            }
        }

        // Resource estimation
        let dsps = (self.rows * self.cols) as u64;  // 1 DSP per PE
        let luts = dsps * 50;  // ~50 LUTs per PE for control logic
        let flip_flops = dsps * (acc_width as u64 + data_width as u64 * 2);
        let bram_kb = (self.rows * data_width * self.depth / 8 / 1024
            + self.cols * data_width * self.depth / 8 / 1024) as u64;

        let pipeline_depth = self.rows + self.cols + self.depth - 2;

        Ok(SystolicArray {
            rows: self.rows,
            cols: self.cols,
            pes,
            a_stream_width: self.rows * data_width,
            b_stream_width: self.cols * data_width,
            c_drain_width: self.cols * acc_width,
            pipeline_depth,
            resources: ResourceEstimate {
                luts,
                flip_flops,
                bram_kb,
                dsps,
                uram_kb: 0,
            },
        })
    }

    /// Emit Verilog for a single processing element.
    pub fn emit_pe_verilog(pe: &ProcessingElement) -> String {
        format!(
r#"module pe_{r}_{c} (
    input  wire clk,
    input  wire rst_n,
    input  wire signed [{dw}:0] a_in,
    input  wire signed [{dw}:0] b_in,
    input  wire signed [{aw}:0] c_in,
    output reg  signed [{dw}:0] a_out,
    output reg  signed [{dw}:0] b_out,
    output reg  signed [{aw}:0] c_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= 0;
            b_out <= 0;
            c_out <= 0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            c_out <= c_in + a_in * b_in;
        end
    end
endmodule
"#,
            r = pe.row,
            c = pe.col,
            dw = pe.data_width - 1,
            aw = pe.acc_width - 1,
        )
    }
}
```

---

## Section 5: Resource Budgeting

### 5.1 FpgaResourceChecker

```rust
/// Validates that a generated design fits within the target FPGA's resources.
pub struct FpgaResourceChecker {
    device: FpgaDevice,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
pub struct ResourceReport {
    pub fits: bool,
    pub used: ResourceEstimate,
    pub available: FpgaResources,
    pub utilization: ResourceUtilization,
    pub bottleneck: Option<ResourceBottleneck>,
}

#[derive(Debug)]
pub struct ResourceUtilization {
    pub lut_pct: f64,
    pub ff_pct: f64,
    pub bram_pct: f64,
    pub dsp_pct: f64,
    pub uram_pct: f64,
}

#[derive(Debug)]
pub enum ResourceBottleneck {
    Luts { used: u64, available: u64 },
    BramKb { used: u64, available: u64 },
    Dsps { used: u64, available: u64 },
    UramKb { used: u64, available: u64 },
}

impl FpgaResourceChecker {
    pub fn check(&self, estimate: &ResourceEstimate) -> ResourceReport {
        let avail = &self.device.resources;

        let lut_pct = estimate.luts as f64 / avail.luts as f64 * 100.0;
        let ff_pct = estimate.flip_flops as f64 / avail.flip_flops as f64 * 100.0;
        let bram_pct = estimate.bram_kb as f64 / avail.bram_kb as f64 * 100.0;
        let dsp_pct = estimate.dsps as f64 / avail.dsps as f64 * 100.0;
        let uram_pct = if avail.uram_kb > 0 {
            estimate.uram_kb as f64 / avail.uram_kb as f64 * 100.0
        } else {
            0.0
        };

        let fits = lut_pct <= 100.0 && ff_pct <= 100.0 && bram_pct <= 100.0
            && dsp_pct <= 100.0 && uram_pct <= 100.0;

        let bottleneck = if !fits {
            if lut_pct > 100.0 {
                Some(ResourceBottleneck::Luts { used: estimate.luts, available: avail.luts })
            } else if bram_pct > 100.0 {
                Some(ResourceBottleneck::BramKb { used: estimate.bram_kb, available: avail.bram_kb })
            } else if dsp_pct > 100.0 {
                Some(ResourceBottleneck::Dsps { used: estimate.dsps, available: avail.dsps })
            } else {
                Some(ResourceBottleneck::UramKb { used: estimate.uram_kb, available: avail.uram_kb })
            }
        } else {
            None
        };

        ResourceReport {
            fits,
            used: estimate.clone(),
            available: avail.clone(),
            utilization: ResourceUtilization { lut_pct, ff_pct, bram_pct, dsp_pct, uram_pct },
            bottleneck,
        }
    }
}
```

---

## Section 6: Groq TSP Backend

### 6.1 Groq Architecture Model

```rust
/// Groq TSP (Tensor Streaming Processor) backend.
/// The TSP executes instructions in a deterministic pipeline without branches.
/// KIR basic blocks (when branch-free) map directly to TSP instruction streams.
#[derive(Debug)]
pub struct GroqBackend {
    /// Number of TSP lanes (320 per chip).
    pub lanes: u32,

    /// SRAM capacity per chip (230 MB).
    pub sram_mb: u32,

    /// Generated TSP instructions.
    pub instructions: Vec<TspInstruction>,
}

/// A single TSP instruction.
#[derive(Debug, Clone)]
pub struct TspInstruction {
    /// Instruction opcode.
    pub op: TspOp,

    /// Source operand addresses (SRAM addresses).
    pub src: Vec<u32>,

    /// Destination operand address (SRAM address).
    pub dst: u32,

    /// Vector length (number of lanes to activate).
    pub lanes: u32,

    /// Static schedule: clock cycle when this instruction executes.
    pub cycle: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum TspOp {
    /// Vector add: dst[i] = src0[i] + src1[i]
    VAdd,
    /// Vector multiply: dst[i] = src0[i] * src1[i]
    VMul,
    /// Vector FMA: dst[i] = src0[i] * src1[i] + src2[i]
    VFma,
    /// Matrix multiply (MXU): uses the matrix execution unit
    MxuMatmul,
    /// Vector ReLU: dst[i] = max(src0[i], 0)
    VRelu,
    /// Load from SRAM to vector register
    Load,
    /// Store from vector register to SRAM
    Store,
    /// No-op (pipeline bubble)
    Nop,
}

impl GroqBackend {
    pub fn new() -> Self {
        Self {
            lanes: 320,
            sram_mb: 230,
            instructions: Vec::new(),
        }
    }

    /// Lower a KIR kernel to TSP instructions.
    pub fn lower_kernel(&mut self, ir: &KernelIR) -> Result<Vec<TspInstruction>, FpgaError> {
        let mut instructions = Vec::new();
        let mut cycle = 0u64;

        for block in &ir.blocks {
            // TSP requires branch-free execution — verify no conditional terminators
            if matches!(block.terminator, KirTerminator::CondBranch(..)) {
                return Err(FpgaError::UnsupportedOp(
                    "Groq TSP does not support conditional branches — \
                     use select() for branchless computation".to_string()
                ));
            }

            for op in &block.ops {
                let tsp_inst = self.lower_op_to_tsp(op, cycle)?;
                cycle += 1; // TSP: one instruction per cycle
                instructions.push(tsp_inst);
            }
        }

        Ok(instructions)
    }

    fn lower_op_to_tsp(
        &self,
        op: &KirOp,
        cycle: u64,
    ) -> Result<TspInstruction, FpgaError> {
        match op {
            KirOp::Add(dst, a, b) => Ok(TspInstruction {
                op: TspOp::VAdd,
                src: vec![*a, *b],
                dst: *dst,
                lanes: self.lanes,
                cycle,
            }),

            KirOp::Mul(dst, a, b) => Ok(TspInstruction {
                op: TspOp::VMul,
                src: vec![*a, *b],
                dst: *dst,
                lanes: self.lanes,
                cycle,
            }),

            KirOp::Fma(dst, a, b, c) => Ok(TspInstruction {
                op: TspOp::VFma,
                src: vec![*a, *b, *c],
                dst: *dst,
                lanes: self.lanes,
                cycle,
            }),

            KirOp::Select(dst, cond, a, b) => {
                // TSP implements select as: dst = cond * a + (1 - cond) * b
                // using two FMAs
                Ok(TspInstruction {
                    op: TspOp::VFma,
                    src: vec![*cond, *a, *b],
                    dst: *dst,
                    lanes: self.lanes,
                    cycle,
                })
            }

            KirOp::Load(dst, addr, _) => Ok(TspInstruction {
                op: TspOp::Load,
                src: vec![*addr],
                dst: *dst,
                lanes: self.lanes,
                cycle,
            }),

            KirOp::Store(addr, val, _) => Ok(TspInstruction {
                op: TspOp::Store,
                src: vec![*val],
                dst: *addr,
                lanes: self.lanes,
                cycle,
            }),

            _ => Err(FpgaError::UnsupportedOp(format!(
                "KIR operation {:?} not yet supported on Groq TSP", op
            ))),
        }
    }

    /// Serialize TSP instructions to Groq assembly text.
    pub fn serialize_tsp(&self, instructions: &[TspInstruction]) -> String {
        let mut asm = String::new();
        asm.push_str("# Generated by NeuralScript M57 Groq Backend\n");
        asm.push_str(&format!("# Lanes: {}\n\n", self.lanes));

        for inst in instructions {
            let op_str = match inst.op {
                TspOp::VAdd => "VADD",
                TspOp::VMul => "VMUL",
                TspOp::VFma => "VFMA",
                TspOp::MxuMatmul => "MXU_MATMUL",
                TspOp::VRelu => "VRELU",
                TspOp::Load => "LD",
                TspOp::Store => "ST",
                TspOp::Nop => "NOP",
            };

            let src_str: Vec<String> = inst.src.iter().map(|s| format!("r{}", s)).collect();
            asm.push_str(&format!(
                "  {:6} r{}, {} ; cycle={}\n",
                op_str,
                inst.dst,
                src_str.join(", "),
                inst.cycle,
            ));
        }

        asm
    }
}
```

---

## Section 7: Neuromorphic Backend (Stub)

### 7.1 SpiNNaker Spike-Timing Format

```rust
/// Neuromorphic backend: converts rate-coded activations to spike trains.
/// This is a stub implementation targeting future collaboration.
pub struct NeuromorphicBackend {
    /// Target platform.
    pub target: NeuromorphicTarget,

    /// Timestep duration (ms).
    pub timestep_ms: f64,

    /// Number of simulation timesteps.
    pub num_timesteps: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum NeuromorphicTarget {
    /// University of Manchester SpiNNaker.
    SpiNNaker,
    /// Intel Loihi 2.
    Loihi,
}

/// A spike event in the neuromorphic simulation.
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Source neuron ID.
    pub neuron_id: u32,
    /// Timestep when the spike occurs.
    pub timestep: u32,
}

/// A neuromorphic network layer.
#[derive(Debug, Clone)]
pub struct NeuromorphicLayer {
    /// Number of neurons in this layer.
    pub num_neurons: u32,
    /// Synaptic weight matrix (source x target).
    pub weights: Vec<Vec<i8>>,
    /// Neuron threshold.
    pub threshold: i32,
    /// Leak rate.
    pub leak: i32,
}

impl NeuromorphicBackend {
    /// Convert a KIR kernel to neuromorphic network description.
    /// This is a stub — full implementation requires collaboration with
    /// neuromorphic hardware teams.
    pub fn lower_kernel(&self, ir: &KernelIR) -> Result<Vec<NeuromorphicLayer>, FpgaError> {
        // Stub: identify matmul + relu patterns and convert to LIF neuron layers
        let mut layers = Vec::new();

        // Walk KIR looking for matmul patterns
        for block in &ir.blocks {
            for op in &block.ops {
                match op {
                    KirOp::Fma(_, _, _, _) | KirOp::Mul(_, _, _) => {
                        // Each matmul becomes a fully-connected layer
                        // with integrate-and-fire neurons
                        layers.push(NeuromorphicLayer {
                            num_neurons: 128, // placeholder
                            weights: vec![],  // populated from weight data
                            threshold: 64,    // firing threshold
                            leak: 1,          // membrane potential leak
                        });
                    }
                    _ => {}
                }
            }
        }

        if layers.is_empty() {
            return Err(FpgaError::UnsupportedOp(
                "no matmul/relu patterns found for neuromorphic conversion".to_string()
            ));
        }

        Ok(layers)
    }

    /// Serialize to SpiNNaker PyNN format (stub).
    pub fn serialize_spinnaker(&self, layers: &[NeuromorphicLayer]) -> String {
        let mut out = String::new();
        out.push_str("# Generated by NeuralScript M57 Neuromorphic Backend (stub)\n");
        out.push_str("# Target: SpiNNaker\n\n");
        out.push_str("import pyNN.spiNNaker as sim\n\n");
        out.push_str(&format!("sim.setup(timestep={})\n\n", self.timestep_ms));

        for (i, layer) in layers.iter().enumerate() {
            out.push_str(&format!(
                "pop_{} = sim.Population({}, sim.IF_curr_exp(\n",
                i, layer.num_neurons
            ));
            out.push_str(&format!(
                "    v_thresh={}, tau_m=20.0\n", layer.threshold
            ));
            out.push_str("))\n\n");
        }

        out.push_str(&format!("sim.run({})\n", self.num_timesteps as f64 * self.timestep_ms));
        out
    }
}
```

---

## Section 8: Error Messages

### 8.1 Resource Budget Exceeded

```
error[E0701]: FPGA resource budget exceeded on xilinx-u280
  --> model.nsl:5
   |
 5 | @fpga_device("xilinx-u280")
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = Resource usage:
     LUTs:   1,450,000 / 1,304,000  (111.2%) -- EXCEEDED
     DSPs:       4,096 /     9,024  ( 45.4%)
     BRAM:       2,048 /     4,032  ( 50.8%)
   |
   = note: the matmul systolic array at line 15 requires 1,200,000 LUTs
   = help: reduce systolic array size with @fpga_tile(rows=8, cols=8)
   = help: or use a larger device: xilinx-u55c has 1,304,000 LUTs
```

### 8.2 Unsupported Operation on FPGA

```
error[E0702]: operation 'softmax' requires floating-point division — not supported on FPGA target
  --> model.nsl:12:16
   |
12 |     let probs = softmax(h, dim=-1)
   |                 ^^^^^^^ requires exp() and division
   |
   = note: FPGA targets support integer arithmetic only (int8, int16, int32)
   = help: use argmax() for classification, or implement a fixed-point approximation
   = help: or use --target fpga,cuda to fall back to GPU for unsupported operations
```

### 8.3 Conditional Branch on Groq TSP

```
error[E0703]: Groq TSP does not support conditional branches
  --> kernel.nsl:8:5
   |
 8 |     if x[tid] > threshold:
   |     ^^^^^^^^^^^^^^^^^^^^^^ data-dependent branch
   |
   = note: TSP executes instructions in a deterministic pipeline — no branches
   = help: use select(x[tid] > threshold, a, b) for branchless computation
```

### 8.4 Device Not Found

```
error[E0704]: unknown FPGA device 'xilinx-v7'
  --> command line
   |
   = available devices: xilinx-u280, xilinx-u55c, intel-agilex
   = help: use --device <name> with one of the supported devices
   = help: use nsl fpga list-devices to see all available devices
```

### 8.5 Resource Check Output

```
$ nsl build --target fpga --device xilinx-u280 --resource-check model.nsl

FPGA Resource Budget for 'TinyClassifier' on xilinx-u280
=========================================================
Resource        Used        Available    Utilization
LUTs            245,000     1,304,000    18.8%
Flip-Flops      180,000     2,607,360     6.9%
BRAM              1,024 KB    4,032 KB   25.4%
DSPs              2,048         9,024    22.7%
UltraRAM              0 KB   36,000 KB    0.0%

Design FITS within xilinx-u280 resources.

Pipeline latency: 4,500 cycles @ 200 MHz = 22.5 us
Throughput: 44.4M inferences/second (II=1)
```

---

## Section 9: Testing Strategy

### 9.1 Unit Tests

| Test | What it verifies |
|------|-----------------|
| `test_kir_add_to_verilog` | KIR Add operation produces correct `assign` statement |
| `test_kir_mul_to_verilog` | KIR Mul produces DSP48 instantiation |
| `test_kir_fma_to_verilog` | KIR Fma produces DSP48 MACC instantiation |
| `test_kir_load_to_verilog` | KIR Load produces BRAM read port |
| `test_kir_store_to_verilog` | KIR Store produces BRAM write port in always_ff block |
| `test_kir_cmp_to_verilog` | KIR Cmp produces comparator |
| `test_kir_select_to_verilog` | KIR Select produces ternary mux |
| `test_kir_const_to_verilog` | KIR Const produces wire tied to literal |
| `test_verilog_module_header` | Module ports (clk, rst_n, start, done) generated correctly |
| `test_verilog_valid_syntax` | Generated Verilog passes Icarus Verilog syntax check (iverilog -t null) |
| `test_systolic_pe_generation` | Single PE module with correct ports and MAC logic |
| `test_systolic_array_tiling` | 16x16 matmul tiled into 4x4 systolic array (4 tiles) |
| `test_resource_estimate_luts` | LUT estimate for add operations: ~1 per bit |
| `test_resource_estimate_dsps` | DSP estimate for multiply: 1 per multiply |
| `test_resource_estimate_bram` | BRAM estimate for weight storage: correct KB |
| `test_resource_check_fits` | Small design fits in xilinx-u280 |
| `test_resource_check_exceeds` | Oversized design reports budget exceeded with bottleneck |
| `test_groq_tsp_add` | KIR Add produces VADD instruction |
| `test_groq_tsp_fma` | KIR Fma produces VFMA instruction |
| `test_groq_branch_rejected` | Conditional branch in KIR produces error for Groq target |
| `test_neuromorphic_stub` | Matmul pattern produces neuromorphic layer description |
| `test_device_database_u280` | xilinx-u280 device lookup returns correct resource counts |
| `test_device_database_unknown` | Unknown device returns None |

### 9.2 Integration Tests

| Test | What it verifies |
|------|-----------------|
| `test_tiny_mlp_to_verilog` | 2-layer MLP (784->128->10) compiles to Verilog, passes syntax check |
| `test_tiny_mlp_resource_budget` | MLP design fits within xilinx-u280 resources |
| `test_matmul_systolic_end_to_end` | matmul(A, B) compiles to systolic array Verilog |
| `test_relu_to_verilog` | relu operation compiles to comparison + mux |
| `test_groq_tiny_kernel` | Simple vector add compiles to Groq TSP assembly |
| `test_pipeline_latency` | Pipeline latency calculation matches expected cycle count |

### 9.3 E2E Tests

| Test File | Description |
|-----------|-------------|
| `examples/m57_fpga_vector_add.nsl` | Simple vector add kernel -> Verilog. Verify valid syntax. |
| `examples/m57_fpga_matmul.nsl` | Matmul kernel -> systolic array Verilog. Verify valid syntax. |
| `examples/m57_fpga_tiny_mlp.nsl` | Tiny MLP -> full Verilog design. Resource budget check. |
| `examples/m57_fpga_resource_error.nsl` | Oversized design. Expect resource budget error. |
| `examples/m57_groq_vector_add.nsl` | Vector add -> Groq TSP assembly. Verify instruction format. |
| `examples/m57_groq_branch_error.nsl` | Kernel with branch on Groq target. Expect error. |
| `examples/m57_neuromorphic_stub.nsl` | Simple model -> neuromorphic format. Verify stub output. |

### 9.4 Simulation Tests (if Icarus Verilog available)

| Test | What it verifies |
|------|-----------------|
| `test_verilog_add_simulation` | Simulate vector add: output matches expected values |
| `test_verilog_matmul_simulation` | Simulate 4x4 matmul: output matches CPU reference |
| `test_verilog_relu_simulation` | Simulate relu: negative values zeroed |

---

## Section 10: File Changes Summary

**New files:**

| File | Lines (est.) | Responsibility |
|------|-------------|----------------|
| `crates/nsl-codegen/src/backend_verilog.rs` | 600 | KIR to Verilog translation, module serialization |
| `crates/nsl-codegen/src/backend_groq.rs` | 250 | KIR to Groq TSP assembly |
| `crates/nsl-codegen/src/backend_neuromorphic.rs` | 150 | KIR to SpiNNaker/Loihi spike format (stub) |
| `crates/nsl-codegen/src/fpga_resource.rs` | 200 | Resource budgeting: LUT/BRAM/DSP checking |
| `crates/nsl-codegen/src/fpga_systolic.rs` | 300 | Systolic array tiling and PE generation |
| `crates/nsl-codegen/src/fpga_devices.rs` | 150 | FPGA device database (xilinx-u280, u55c, intel-agilex) |
| `crates/nsl-runtime/src/fpga_runtime.rs` | 200 | Host-side FPGA communication (XDMA/OpenCL stub) |
| `examples/m57_fpga_vector_add.nsl` | 15 | E2E test: vector add to Verilog |
| `examples/m57_fpga_matmul.nsl` | 25 | E2E test: matmul to systolic array |
| `examples/m57_fpga_tiny_mlp.nsl` | 30 | E2E test: full MLP to Verilog |
| `examples/m57_fpga_resource_error.nsl` | 20 | E2E test: resource budget exceeded |
| `examples/m57_groq_vector_add.nsl` | 15 | E2E test: Groq TSP assembly |
| `examples/m57_groq_branch_error.nsl` | 15 | E2E test: branch error on Groq |
| `examples/m57_neuromorphic_stub.nsl` | 15 | E2E test: neuromorphic stub |

**Modified files:**

| File | Change |
|------|--------|
| `crates/nsl-codegen/src/lib.rs` | `pub mod backend_verilog; pub mod backend_groq; pub mod backend_neuromorphic; pub mod fpga_resource; pub mod fpga_systolic; pub mod fpga_devices;` |
| `crates/nsl-codegen/src/compiler.rs` | Route `--target fpga/groq/neuromorphic` to new backends in `compile_single_kernel()` |
| `crates/nsl-codegen/src/kernel_ir.rs` | Add `FpgaTile` and `GroqStream` annotation types to KernelIR metadata |
| `crates/nsl-runtime/src/lib.rs` | `pub mod fpga_runtime;` |
| `crates/nsl-cli/src/main.rs` | `--target fpga/groq/neuromorphic`, `--device`, `--resource-check`, `--testbench` flags |
| `crates/nsl-ast/src/decorators.rs` | `FpgaTile`, `FpgaClock`, `FpgaDevice`, `FpgaPipeline`, `GroqStream` decorator variants |
| `crates/nsl-parser/src/decorators.rs` | Parse `@fpga_tile(...)`, `@fpga_clock(...)`, `@fpga_device(...)`, `@fpga_pipeline(...)`, `@groq_stream(...)` |
| `crates/nsl-semantic/src/checker.rs` | Validate FPGA/Groq decorators: integer types only, no dynamic control flow |
| `crates/nsl-cli/tests/e2e.rs` | 7 new E2E tests |

---

## Section 11: Deliverables

1. `VerilogBackend` — KIR to Verilog translation for all portable operations (add, mul, fma, load, store, cmp, select)
2. `SystolicMapper` — partitions matmul into systolic array tiles, generates PE Verilog modules
3. `FpgaResourceChecker` — compile-time resource budgeting (LUT/BRAM/DSP) against target device
4. `FpgaDevice` database — built-in specs for Xilinx U280, U55C, Intel Agilex
5. `GroqBackend` — KIR to Groq TSP assembly (branch-free instruction streams)
6. `NeuromorphicBackend` — stub implementation for SpiNNaker/Loihi spike-timing format
7. `nsl build --target fpga --device <name>` CLI with `--resource-check` and `--testbench` options
8. `@fpga_tile`, `@fpga_clock`, `@fpga_device`, `@fpga_pipeline` decorators for FPGA-specific control
9. Latency-deterministic execution: pipeline latency computed at compile time (feeds M53 WCET)
10. Verilog syntax validation (Icarus Verilog) for all generated designs

---

## Out of Scope

- Full FPGA synthesis (Vivado/Quartus invocation — the compiler produces Verilog; synthesis is external)
- FPGA runtime data transfer (DMA engine, PCIe communication — stub only in M57; full runtime in future milestone)
- Multi-FPGA partitioning (spreading a model across multiple FPGA cards)
- FPGA-specific memory hierarchy optimization (HBM channel mapping, DDR bank interleaving)
- Floating-point FPGA cores (only integer/fixed-point arithmetic in M57)
- FPGA bitstream generation (requires vendor-specific toolchain)
- Groq hardware access (the backend produces assembly; running on Groq TSP requires their SDK)
- Neuromorphic full implementation (M57 provides a stub; full SNN conversion is a research project)
- Intel OneAPI / SYCL integration (FPGA programming via Intel's HLS flow)
- Formal equivalence checking (proving Verilog output matches NSL source semantics)
- FPGA power consumption estimation
- Clock domain crossing for multi-clock FPGA designs
- AXI bus protocol generation (for SoC integration)

## Success Criteria

- Simple vector add kernel: produces valid Verilog (passes `iverilog -t null` syntax check)
- Matmul kernel: produces systolic array Verilog with correct PE instantiations
- Tiny MLP (784->128->10): produces complete Verilog design with resource budget report
- Resource budget: oversized design produces clear error with bottleneck identification
- Resource budget: correctly-sized design reports "FITS" with utilization percentages
- Pipeline latency: computed cycle count matches hand-calculated value
- Groq backend: vector add produces valid TSP assembly with correct instruction sequence
- Groq backend: kernel with branch produces clear error ("TSP does not support branches")
- Neuromorphic stub: matmul pattern produces PyNN-compatible output
- All existing tests pass unchanged — FPGA/Groq/neuromorphic are additive backends
- Generated Verilog resource estimates are within 2x of actual Vivado synthesis results (validated on reference designs)
