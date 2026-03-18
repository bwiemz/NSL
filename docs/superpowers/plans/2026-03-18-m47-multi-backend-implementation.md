# M47: Multi-Backend Targeting — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a backend-agnostic Kernel IR between the NSL kernel AST and target-specific code emission, enabling the same `kernel` block to compile for CUDA (PTX), ROCm, Metal, or WebGPU — for the portable subset of operations. Refactor the existing direct AST→PTX path through the KIR so the PTX backend validates against existing hand-written kernels.

**Architecture:** Four new codegen modules: `kernel_ir.rs` (KIR types + builder), `kernel_lower.rs` (AST→KIR), `backend_ptx.rs` (KIR→PTX), + `gpu_target.rs` (GpuTarget enum, FeatureSet, capabilities). One new runtime module: `gpu_backend.rs` (GpuBackend trait). Semantic validation for `@target` decorator. CLI `--target` flag. The existing `kernel.rs` is preserved but the new pipeline is wired alongside it in `compile_single_kernel`.

**Tech Stack:** Rust (codegen IR + backend lowering + runtime trait + semantic + CLI)

**Spec:** `docs/superpowers/specs/2026-03-15-m47-multi-backend-design.md`

**Prerequisites:** None (standalone)

---

## Important: Scope of This Plan

**This plan builds the KIR foundation + PTX backend + infrastructure.** It delivers:
- `KernelIR` SSA-form intermediate representation (40+ instruction types)
- `KirBuilder` for programmatic KIR construction
- `KirOp`, `KirTerminator`, `KirType`, `AddressSpace` enums
- `FeatureSet` bitflags with backend capability tables
- `GpuTarget` enum (Cuda, Rocm, Metal, WebGpu)
- AST→KIR lowering for portable operations (arithmetic, thread indexing, memory, control flow, barrier)
- KIR→PTX backend lowering (validates against existing hand-written kernel patterns)
- `GpuBackend` trait definition (alloc, free, copy, load_module, launch_kernel, sync)
- `@target` decorator semantic validation
- `--target` CLI flag on `nsl build` and `nsl run`
- `CompileOptions.target` field
- 25+ unit tests covering KIR construction, PTX emission, feature gating, target parsing

**Deferred to M47b:** ROCm backend (`backend_amdgpu.rs` + `nsl-runtime-rocm` crate), Metal backend (`backend_metal.rs` + `nsl-runtime-metal` crate), WebGPU backend (`backend_wgsl.rs` + `nsl-runtime-webgpu` crate), migration of 15 hand-written PTX kernels from static strings to programmatic KIR, wiring `compile_single_kernel` to use KIR pipeline by default (M47a adds it alongside existing path), device-target compatibility checking in semantic layer, numerical parity tests across backends, `KirBuilder.build_tensor_add_kir()` and friends for all standard ops.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/kernel_ir.rs` | KernelIR, KirOp, KirType, KirBuilder, FeatureSet | 400 |
| `crates/nsl-codegen/src/kernel_lower.rs` | AST KernelDef → KIR lowering | 250 |
| `crates/nsl-codegen/src/backend_ptx.rs` | KIR → PTX text emission | 300 |
| `crates/nsl-codegen/src/gpu_target.rs` | GpuTarget enum, capabilities, feature validation | 120 |
| `crates/nsl-runtime/src/gpu_backend.rs` | GpuBackend trait, DeviceCapabilities, KernelArg | 100 |
| `crates/nsl-semantic/src/target.rs` | `@target` decorator validation | 80 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Add 4 new module declarations |
| `crates/nsl-codegen/src/compiler.rs` | Add `target` to CompileOptions, `compile_single_kernel_kir` method |
| `crates/nsl-codegen/src/builtins.rs` | No changes (existing kernel launch FFI sufficient) |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod gpu_backend;` |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod target;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@target` validation |
| `crates/nsl-cli/src/main.rs` | Add `--target` flag to Build and Run commands |

---

## Phase 1: Kernel IR + GpuTarget

### Task 1: GpuTarget + FeatureSet

**Files:**
- Create: `crates/nsl-codegen/src/gpu_target.rs`

- [ ] **Step 1: Create `gpu_target.rs` with GpuTarget enum, FeatureSet, and capability tables**

```rust
// crates/nsl-codegen/src/gpu_target.rs
//! M47: GPU target selection and feature capability detection.

/// GPU compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    Cuda,
    Rocm,
    Metal,
    WebGpu,
}

impl GpuTarget {
    /// Parse from CLI string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cuda" => Some(GpuTarget::Cuda),
            "rocm" | "amd" => Some(GpuTarget::Rocm),
            "metal" | "apple" => Some(GpuTarget::Metal),
            "webgpu" | "wgsl" => Some(GpuTarget::WebGpu),
            _ => None,
        }
    }

    /// Display name for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            GpuTarget::Cuda => "cuda",
            GpuTarget::Rocm => "rocm",
            GpuTarget::Metal => "metal",
            GpuTarget::WebGpu => "webgpu",
        }
    }

    /// Supported features for this target.
    pub fn features(&self) -> FeatureSet {
        match self {
            GpuTarget::Cuda => FeatureSet::SHARED_MEMORY
                | FeatureSet::WARP_SHUFFLE
                | FeatureSet::TENSOR_CORES
                | FeatureSet::ATOMIC_FLOAT
                | FeatureSet::F16_ARITHMETIC
                | FeatureSet::BF16_ARITHMETIC,
            GpuTarget::Rocm => FeatureSet::SHARED_MEMORY
                | FeatureSet::WARP_SHUFFLE
                | FeatureSet::TENSOR_CORES
                | FeatureSet::ATOMIC_FLOAT
                | FeatureSet::F16_ARITHMETIC
                | FeatureSet::BF16_ARITHMETIC,
            GpuTarget::Metal => FeatureSet::SHARED_MEMORY
                | FeatureSet::WARP_SHUFFLE
                | FeatureSet::ATOMIC_FLOAT
                | FeatureSet::F16_ARITHMETIC,
            GpuTarget::WebGpu => FeatureSet::SHARED_MEMORY
                | FeatureSet::F16_ARITHMETIC,
        }
    }

    /// Default warp/wavefront/SIMD width.
    pub fn warp_size(&self) -> u32 {
        match self {
            GpuTarget::Cuda => 32,
            GpuTarget::Rocm => 64,
            GpuTarget::Metal => 32,
            GpuTarget::WebGpu => 0, // no subgroup guarantees
        }
    }
}

/// Feature flags for GPU capabilities.
///
/// Used to validate that a kernel's required features are supported by the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSet(u32);

impl FeatureSet {
    pub const NONE: Self = FeatureSet(0);
    pub const SHARED_MEMORY: Self = FeatureSet(0x01);
    pub const WARP_SHUFFLE: Self = FeatureSet(0x02);
    pub const TENSOR_CORES: Self = FeatureSet(0x04);
    pub const ATOMIC_FLOAT: Self = FeatureSet(0x08);
    pub const SUBGROUP_OPS: Self = FeatureSet(0x10);
    pub const F16_ARITHMETIC: Self = FeatureSet(0x20);
    pub const BF16_ARITHMETIC: Self = FeatureSet(0x40);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Features required by kernel but not supported by target.
    pub fn missing(self, required: Self) -> Self {
        FeatureSet(required.0 & !self.0)
    }

    /// Human-readable list of feature names.
    pub fn names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.0 & 0x01 != 0 { names.push("shared_memory"); }
        if self.0 & 0x02 != 0 { names.push("warp_shuffle"); }
        if self.0 & 0x04 != 0 { names.push("tensor_cores"); }
        if self.0 & 0x08 != 0 { names.push("atomic_float"); }
        if self.0 & 0x10 != 0 { names.push("subgroup_ops"); }
        if self.0 & 0x20 != 0 { names.push("f16_arithmetic"); }
        if self.0 & 0x40 != 0 { names.push("bf16_arithmetic"); }
        names
    }
}

impl std::ops::BitOr for FeatureSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { FeatureSet(self.0 | rhs.0) }
}

impl std::ops::BitOrAssign for FeatureSet {
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

impl std::ops::BitAnd for FeatureSet {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self { FeatureSet(self.0 & rhs.0) }
}

impl std::ops::Not for FeatureSet {
    type Output = Self;
    fn not(self) -> Self { FeatureSet(!self.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_parse() {
        assert_eq!(GpuTarget::from_str("cuda"), Some(GpuTarget::Cuda));
        assert_eq!(GpuTarget::from_str("ROCM"), Some(GpuTarget::Rocm));
        assert_eq!(GpuTarget::from_str("metal"), Some(GpuTarget::Metal));
        assert_eq!(GpuTarget::from_str("webgpu"), Some(GpuTarget::WebGpu));
        assert_eq!(GpuTarget::from_str("vulkan"), None);
    }

    #[test]
    fn cuda_has_all_features() {
        let f = GpuTarget::Cuda.features();
        assert!(f.contains(FeatureSet::SHARED_MEMORY));
        assert!(f.contains(FeatureSet::WARP_SHUFFLE));
        assert!(f.contains(FeatureSet::TENSOR_CORES));
        assert!(f.contains(FeatureSet::BF16_ARITHMETIC));
    }

    #[test]
    fn webgpu_lacks_shuffle_and_tensor_cores() {
        let f = GpuTarget::WebGpu.features();
        assert!(f.contains(FeatureSet::SHARED_MEMORY));
        assert!(!f.contains(FeatureSet::WARP_SHUFFLE));
        assert!(!f.contains(FeatureSet::TENSOR_CORES));
    }

    #[test]
    fn feature_missing_detection() {
        let target = GpuTarget::WebGpu.features();
        let required = FeatureSet::SHARED_MEMORY | FeatureSet::TENSOR_CORES;
        let missing = target.missing(required);
        assert!(missing.contains(FeatureSet::TENSOR_CORES));
        assert!(!missing.contains(FeatureSet::SHARED_MEMORY));
    }

    #[test]
    fn feature_names() {
        let f = FeatureSet::WARP_SHUFFLE | FeatureSet::TENSOR_CORES;
        let names = f.names();
        assert!(names.contains(&"warp_shuffle"));
        assert!(names.contains(&"tensor_cores"));
        assert_eq!(names.len(), 2);
    }
}
```

### Task 2: Kernel IR Types + Builder

**Files:**
- Create: `crates/nsl-codegen/src/kernel_ir.rs`

- [ ] **Step 2: Create `kernel_ir.rs` with KernelIR, KirOp, KirType, AddressSpace, KirBuilder, and tests**

```rust
// crates/nsl-codegen/src/kernel_ir.rs
//! M47: Backend-agnostic Kernel IR — SSA-form intermediate representation
//! for GPU compute kernels.

use crate::gpu_target::FeatureSet;

pub type VarId = u32;
pub type BlockId = u32;

// ---------------------------------------------------------------------------
// KIR top-level types
// ---------------------------------------------------------------------------

/// A complete kernel in IR form.
#[derive(Debug, Clone)]
pub struct KernelIR {
    pub name: String,
    pub params: Vec<KirParam>,
    pub blocks: Vec<KirBlock>,
    /// Type of each VarId — populated by KirBuilder as operations are emitted.
    /// Required by backends to emit typed instructions (e.g., `add.f32` vs `add.u32`).
    pub var_types: std::collections::HashMap<VarId, KirType>,
    pub shared_mem_bytes: u32,
    pub workgroup_size: [u32; 3],
    pub required_features: FeatureSet,
    next_var: VarId,
}

#[derive(Debug, Clone)]
pub struct KirParam {
    pub id: VarId,
    pub name: String,
    pub ty: KirType,
    pub address_space: AddressSpace,
}

#[derive(Debug, Clone)]
pub struct KirBlock {
    pub id: BlockId,
    pub ops: Vec<KirOp>,
    pub terminator: Option<KirTerminator>,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum KirType {
    U32, I32, U64, I64,
    F16, Bf16, F32, F64,
    Bool,
    Ptr(Box<KirType>, AddressSpace),
    Vec(Box<KirType>, u32),
}

impl KirType {
    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            KirType::Bool => 1,
            KirType::U32 | KirType::I32 | KirType::F32 => 4,
            KirType::U64 | KirType::I64 | KirType::F64 => 8,
            KirType::F16 | KirType::Bf16 => 2,
            KirType::Ptr(_, _) => 8,
            KirType::Vec(inner, n) => inner.size_bytes() * *n as usize,
        }
    }

    /// PTX register prefix.
    pub fn ptx_reg_prefix(&self) -> &'static str {
        match self {
            KirType::U32 | KirType::I32 | KirType::Bool => "%r",
            KirType::U64 | KirType::I64 | KirType::Ptr(_, _) => "%rd",
            KirType::F32 => "%f",
            KirType::F64 => "%fd",
            KirType::F16 | KirType::Bf16 => "%h",
            KirType::Vec(_, _) => "%v",
        }
    }

    /// PTX type suffix.
    pub fn ptx_type(&self) -> &'static str {
        match self {
            KirType::U32 => "u32",
            KirType::I32 => "i32",
            KirType::U64 => "u64",
            KirType::I64 => "i64",
            KirType::F16 => "f16",
            KirType::Bf16 => "bf16",
            KirType::F32 => "f32",
            KirType::F64 => "f64",
            KirType::Bool => "pred",
            KirType::Ptr(_, _) => "u64",
            KirType::Vec(_, _) => "b32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    Constant,
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum KirOp {
    // Arithmetic (dst, a, b)
    Add(VarId, VarId, VarId),
    Sub(VarId, VarId, VarId),
    Mul(VarId, VarId, VarId),
    Div(VarId, VarId, VarId),
    Fma(VarId, VarId, VarId, VarId),  // dst = a * b + c
    Neg(VarId, VarId),
    Abs(VarId, VarId),

    // Math functions (dst, src)
    Sqrt(VarId, VarId),
    Exp(VarId, VarId),
    Log(VarId, VarId),
    Sin(VarId, VarId),
    Cos(VarId, VarId),
    Tanh(VarId, VarId),
    Pow(VarId, VarId, VarId),  // dst = base^exp

    // Type conversion
    Cast(VarId, VarId, KirType),

    // Memory
    Load(VarId, VarId, AddressSpace),
    Store(VarId, VarId, AddressSpace),  // *ptr = val
    AtomicAdd(VarId, VarId, AddressSpace),

    // Thread indexing (dst, dim: 0=x, 1=y, 2=z)
    ThreadId(VarId, u8),
    BlockIdx(VarId, u8),
    BlockDim(VarId, u8),
    GridDim(VarId, u8),
    GlobalId(VarId, u8),  // blockIdx*blockDim + threadIdx

    // Synchronization
    Barrier,
    WarpShuffle(VarId, VarId, VarId),  // dst = shuffle_down(val, offset)

    // Comparison (dst, a, b, op)
    Cmp(VarId, VarId, VarId, CmpOp),
    // Select (dst, cond, true_val, false_val)
    Select(VarId, VarId, VarId, VarId),

    // Constants
    Const(VarId, KirConst),

    // Pointer arithmetic
    PtrOffset(VarId, VarId, VarId),  // dst = base + offset * sizeof(pointee)

    // Shared memory fence
    SharedMemFence,
}

#[derive(Debug, Clone)]
pub enum KirTerminator {
    Branch(BlockId),
    CondBranch(VarId, BlockId, BlockId),
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

#[derive(Debug, Clone)]
pub struct KirConst {
    pub ty: KirType,
    pub value: ConstValue,
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// KirBuilder — programmatic IR construction
// ---------------------------------------------------------------------------

/// Builder for constructing KernelIR programmatically.
pub struct KirBuilder {
    name: String,
    params: Vec<KirParam>,
    blocks: Vec<KirBlock>,
    current_block: Option<BlockId>,
    next_var: VarId,
    var_types: std::collections::HashMap<VarId, KirType>,
    shared_mem_bytes: u32,
    workgroup_size: [u32; 3],
    required_features: FeatureSet,
}

impl KirBuilder {
    pub fn new(name: &str) -> Self {
        KirBuilder {
            name: name.to_string(),
            params: Vec::new(),
            blocks: Vec::new(),
            current_block: None,
            next_var: 0,
            var_types: std::collections::HashMap::new(),
            shared_mem_bytes: 0,
            workgroup_size: [256, 1, 1],
            required_features: FeatureSet::NONE,
        }
    }

    pub fn new_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    /// Allocate a new typed variable. The type is recorded for backend use.
    pub fn new_typed_var(&mut self, ty: KirType) -> VarId {
        let id = self.new_var();
        self.var_types.insert(id, ty);
        id
    }

    pub fn add_param(&mut self, name: &str, ty: KirType, address_space: AddressSpace) -> VarId {
        let id = self.new_var();
        self.var_types.insert(id, ty.clone());
        self.params.push(KirParam { id, name: name.to_string(), ty, address_space });
        id
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = self.blocks.len() as BlockId;
        self.blocks.push(KirBlock { id, ops: Vec::new(), terminator: None });
        id
    }

    pub fn set_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    pub fn emit(&mut self, op: KirOp) {
        let block_id = self.current_block.expect("no current block set");
        // Track required features
        match &op {
            KirOp::Barrier => self.required_features |= FeatureSet::SHARED_MEMORY,
            KirOp::WarpShuffle(_, _, _) => self.required_features |= FeatureSet::WARP_SHUFFLE,
            KirOp::SharedMemFence => self.required_features |= FeatureSet::SHARED_MEMORY,
            KirOp::AtomicAdd(_, _, AddressSpace::Global) => {
                // Float atomics need ATOMIC_FLOAT; integer atomics are universal
                self.required_features |= FeatureSet::ATOMIC_FLOAT;
            }
            _ => {}
        }
        self.blocks[block_id as usize].ops.push(op);
    }

    pub fn terminate(&mut self, term: KirTerminator) {
        let block_id = self.current_block.expect("no current block set");
        self.blocks[block_id as usize].terminator = Some(term);
    }

    pub fn set_workgroup_size(&mut self, size: [u32; 3]) {
        self.workgroup_size = size;
    }

    pub fn set_shared_mem(&mut self, bytes: u32) {
        self.shared_mem_bytes = bytes;
        if bytes > 0 {
            self.required_features |= FeatureSet::SHARED_MEMORY;
        }
    }

    pub fn finalize(self) -> KernelIR {
        KernelIR {
            name: self.name,
            params: self.params,
            blocks: self.blocks,
            var_types: self.var_types,
            shared_mem_bytes: self.shared_mem_bytes,
            workgroup_size: self.workgroup_size,
            required_features: self.required_features,
            next_var: self.next_var,
        }
    }
}

// ---------------------------------------------------------------------------
// KIR helpers
// ---------------------------------------------------------------------------

impl KernelIR {
    /// Count total operations across all blocks.
    pub fn op_count(&self) -> usize {
        self.blocks.iter().map(|b| b.ops.len()).sum()
    }

    /// Validate that all blocks have terminators.
    pub fn is_well_formed(&self) -> bool {
        self.blocks.iter().all(|b| b.terminator.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_add_kernel() -> KernelIR {
        let mut b = KirBuilder::new("test_add");
        let a_ptr = b.add_param("a", KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global), AddressSpace::Global);
        let b_ptr = b.add_param("b", KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global), AddressSpace::Global);
        let out_ptr = b.add_param("out", KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global), AddressSpace::Global);
        let len = b.add_param("len", KirType::U32, AddressSpace::Local);

        let entry = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();

        b.set_block(entry);
        let tid = b.new_var();
        b.emit(KirOp::GlobalId(tid, 0));
        let in_bounds = b.new_var();
        b.emit(KirOp::Cmp(in_bounds, tid, len, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(in_bounds, body, exit));

        b.set_block(body);
        // Compute a[tid], b[tid] addresses via PtrOffset
        let a_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        b.emit(KirOp::PtrOffset(a_addr, a_ptr, tid));
        let a_val = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(a_val, a_addr, AddressSpace::Global));
        let b_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        b.emit(KirOp::PtrOffset(b_addr, b_ptr, tid));
        let b_val = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(b_val, b_addr, AddressSpace::Global));
        let sum = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Add(sum, a_val, b_val));
        let out_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        b.emit(KirOp::PtrOffset(out_addr, out_ptr, tid));
        b.emit(KirOp::Store(out_addr, sum, AddressSpace::Global));
        b.terminate(KirTerminator::Branch(exit));

        b.set_block(exit);
        b.terminate(KirTerminator::Return);

        b.set_workgroup_size([256, 1, 1]);
        b.finalize()
    }

    #[test]
    fn builder_creates_valid_ir() {
        let ir = build_simple_add_kernel();
        assert_eq!(ir.name, "test_add");
        assert_eq!(ir.params.len(), 4);
        assert_eq!(ir.blocks.len(), 3);
        assert!(ir.is_well_formed());
        assert_eq!(ir.workgroup_size, [256, 1, 1]);
    }

    #[test]
    fn op_count() {
        let ir = build_simple_add_kernel();
        // entry: GlobalId + Cmp = 2, body: PtrOffset+Load+PtrOffset+Load+Add+PtrOffset+Store = 7, exit: 0
        assert_eq!(ir.op_count(), 9);
    }

    #[test]
    fn no_features_for_simple_kernel() {
        let ir = build_simple_add_kernel();
        assert!(ir.required_features.is_empty());
    }

    #[test]
    fn barrier_requires_shared_memory_feature() {
        let mut b = KirBuilder::new("test");
        let entry = b.new_block();
        b.set_block(entry);
        b.emit(KirOp::Barrier);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert!(ir.required_features.contains(FeatureSet::SHARED_MEMORY));
    }

    #[test]
    fn shuffle_requires_warp_shuffle_feature() {
        let mut b = KirBuilder::new("test");
        let entry = b.new_block();
        b.set_block(entry);
        let v0 = b.new_var();
        let v1 = b.new_var();
        let dst = b.new_var();
        b.emit(KirOp::WarpShuffle(dst, v0, v1));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert!(ir.required_features.contains(FeatureSet::WARP_SHUFFLE));
    }

    #[test]
    fn kir_type_sizes() {
        assert_eq!(KirType::F32.size_bytes(), 4);
        assert_eq!(KirType::F64.size_bytes(), 8);
        assert_eq!(KirType::F16.size_bytes(), 2);
        assert_eq!(KirType::U32.size_bytes(), 4);
        assert_eq!(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global).size_bytes(), 8);
    }

    #[test]
    fn kir_type_ptx_mapping() {
        assert_eq!(KirType::F32.ptx_type(), "f32");
        assert_eq!(KirType::U32.ptx_type(), "u32");
        assert_eq!(KirType::F32.ptx_reg_prefix(), "%f");
        assert_eq!(KirType::U32.ptx_reg_prefix(), "%r");
        assert_eq!(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global).ptx_reg_prefix(), "%rd");
    }
}
```

---

## Phase 2: KIR→PTX Backend + AST→KIR Lowering

### Task 3: PTX Backend

**Files:**
- Create: `crates/nsl-codegen/src/backend_ptx.rs`

- [ ] **Step 3: Create `backend_ptx.rs` with KIR→PTX lowering**

This module lowers a `KernelIR` to PTX text bytes. It replaces direct PTX emission for kernels that go through the KIR pipeline.

Key functions:
- `lower_kir_to_ptx(ir: &KernelIR) -> Vec<u8>` — main entry point
- Emits `.version 7.0`, `.target sm_70`, `.address_size 64`
- Declares shared memory if `shared_mem_bytes > 0`
- Emits `.visible .entry` with parameter declarations
- Emits register declarations (count vars by type)
- Emits each block as a label, each op as PTX instruction, each terminator as branch/ret
- Maps KirOp to PTX instructions per the spec table (ThreadId → `mov.u32 %r{dst}, %tid.x`, etc.)

Tests (5):
- `test_simple_add_ptx` — build tensor_add KIR, emit PTX, verify it contains expected instructions
- `test_ptx_thread_indexing` — GlobalId emits `mul.lo.u32` + `add.u32` (NOT `mad.lo.u32` — see MEMORY.md: causes INVALID_PTX on PTX ISA 7.0)
- `test_ptx_shared_memory_declaration` — shared_mem_bytes > 0 emits `.shared` directive
- `test_ptx_null_terminated` — output ends with 0 byte
- `test_ptx_barrier` — Barrier emits `bar.sync 0`

### Task 4: AST→KIR Lowering

**Files:**
- Create: `crates/nsl-codegen/src/kernel_lower.rs`

- [ ] **Step 4: Create `kernel_lower.rs` with AST KernelDef → KIR lowering**

This module translates a parsed `KernelDef` AST node into the `KernelIR`. It handles:
- Parameter mapping (NSL params → KirParams with inferred types)
- Body statement lowering (let bindings, assignments, if/else, expressions)
- Builtin recognition: `thread_id()` → `KirOp::ThreadId`, `sync_threads()` → `KirOp::Barrier`
- Arithmetic expression lowering: `a + b` → `KirOp::Add`, etc.
- Memory access: `a[tid]` → `KirOp::PtrOffset` + `KirOp::Load`

Key function:
- `lower_kernel_to_ir(kernel: &KernelDef, interner: &Interner, target: GpuTarget) -> KernelIR`

Tests (4):
- `test_lower_empty_kernel` — kernel with empty body produces valid IR
- `test_lower_params` — kernel params mapped to KirParams
- `test_lower_basic_ops` — arithmetic expressions lowered to correct KirOps
- `test_feature_tracking` — sync_threads() in body sets SHARED_MEMORY feature

**Note:** The lowering is a best-effort translation. Complex kernel bodies that use NSL features beyond the portable subset will fall back to the existing direct AST→PTX path. The KIR lowering handles the common cases (element-wise ops, thread indexing, loads, stores, branches).

---

## Phase 3: Runtime Trait + Semantic + CLI

### Task 5: GpuBackend Trait

**Files:**
- Create: `crates/nsl-runtime/src/gpu_backend.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 5: Create `gpu_backend.rs` with GpuBackend trait definition**

```rust
// crates/nsl-runtime/src/gpu_backend.rs
//! M47: Backend-agnostic GPU runtime trait.
//!
//! All GPU backends (CUDA, ROCm, Metal, WebGPU) implement this trait.
//! The compiled binary links against exactly one backend — no runtime dispatch.

use std::ffi::c_void;

/// Opaque device memory pointer.
pub type DevicePtr = *mut c_void;
/// Opaque handle to a loaded kernel module.
pub type ModuleHandle = u64;
/// Opaque handle to a GPU stream/command queue.
pub type StreamHandle = u64;

/// Errors from GPU operations.
#[derive(Debug)]
pub enum GpuError {
    OutOfMemory { requested: usize },
    InvalidPointer,
    KernelLaunchFailed { name: String, code: i32 },
    ModuleLoadFailed { reason: String },
    Unsupported { feature: String },
    DriverError { code: i32, message: String },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::OutOfMemory { requested } => write!(f, "GPU out of memory ({requested} bytes requested)"),
            GpuError::InvalidPointer => write!(f, "invalid device pointer"),
            GpuError::KernelLaunchFailed { name, code } => write!(f, "kernel '{name}' launch failed (error {code})"),
            GpuError::ModuleLoadFailed { reason } => write!(f, "module load failed: {reason}"),
            GpuError::Unsupported { feature } => write!(f, "unsupported feature: {feature}"),
            GpuError::DriverError { code, message } => write!(f, "GPU driver error {code}: {message}"),
        }
    }
}

/// Kernel launch argument.
#[derive(Debug, Clone)]
pub enum KernelArg {
    Ptr(DevicePtr),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
}

/// Device capability information.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub name: String,
    pub max_threads_per_block: u32,
    pub max_shared_memory: u32,
    pub warp_size: u32,
    pub has_tensor_cores: bool,
    pub has_f16_compute: bool,
    pub has_bf16_compute: bool,
    pub has_atomic_float: bool,
    pub total_memory_bytes: u64,
}

/// Backend-agnostic GPU runtime interface.
///
/// Each backend (CUDA, ROCm, Metal, WebGPU) provides an implementation.
/// The compiled binary links against exactly one backend at build time.
pub trait GpuBackend: Send + Sync {
    /// Allocate device memory. Returns a device pointer.
    fn alloc(&self, bytes: usize) -> Result<DevicePtr, GpuError>;

    /// Free device memory.
    fn free(&self, ptr: DevicePtr) -> Result<(), GpuError>;

    /// Copy bytes from host to device.
    fn copy_h2d(&self, host: *const u8, device: DevicePtr, bytes: usize) -> Result<(), GpuError>;

    /// Copy bytes from device to host.
    fn copy_d2h(&self, device: DevicePtr, host: *mut u8, bytes: usize) -> Result<(), GpuError>;

    /// Copy bytes from device to device (no host round-trip).
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<(), GpuError>;

    /// Load a compiled kernel module (PTX, AMDGPU, Metal library, WGSL).
    fn load_module(&self, code: &[u8]) -> Result<ModuleHandle, GpuError>;

    /// Launch a kernel by name from a loaded module.
    fn launch_kernel(
        &self,
        module: ModuleHandle,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        args: &[KernelArg],
    ) -> Result<(), GpuError>;

    /// Wait for all outstanding operations to complete.
    fn synchronize(&self) -> Result<(), GpuError>;

    /// Query device capabilities.
    fn capabilities(&self) -> DeviceCapabilities;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_error_display() {
        let e = GpuError::OutOfMemory { requested: 1024 };
        assert!(format!("{e}").contains("1024"));

        let e = GpuError::KernelLaunchFailed { name: "add".into(), code: -1 };
        assert!(format!("{e}").contains("add"));
    }

    #[test]
    fn kernel_arg_variants() {
        let args = vec![
            KernelArg::Ptr(std::ptr::null_mut()),
            KernelArg::U32(256),
            KernelArg::F32(1.0),
        ];
        assert_eq!(args.len(), 3);
    }
}
```

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod gpu_backend;
```

### Task 6: Semantic Validation + CLI

**Files:**
- Create: `crates/nsl-semantic/src/target.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`
- Modify: `crates/nsl-cli/src/main.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 6: Create `target.rs` validation, wire into checker, add CLI flag, extend CompileOptions**

`target.rs` — validate `@target(backend)` decorator:
```rust
// crates/nsl-semantic/src/target.rs
//! M47: @target(backend) decorator validation.

use nsl_ast::decl::Decorator;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

pub fn validate_target_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<String> {
    let mut targets = Vec::new();
    if let Some(ref args) = deco.args {
        for arg in args {
            // Positional args: target names
            if arg.name.is_none() {
                if let nsl_ast::expr::ExprKind::Ident(sym) = &arg.value.kind {
                    let name = resolve_sym(*sym);
                    let valid = ["cuda", "rocm", "metal", "webgpu"];
                    if !valid.contains(&name.as_str()) {
                        diagnostics.push(
                            Diagnostic::error(format!("unknown target '{name}', expected: cuda, rocm, metal, webgpu"))
                                .with_label(arg.value.span, "here"),
                        );
                    } else {
                        targets.push(name);
                    }
                }
            }
        }
    }
    if targets.is_empty() {
        diagnostics.push(
            Diagnostic::error("@target requires at least one backend name")
                .with_label(deco.span, "here"),
        );
    }
    targets
}
```

Wire into `checker.rs`:
```rust
if dname == "target" {
    let resolve = |s: nsl_ast::Symbol| -> String { self.interner.resolve(s.0).unwrap_or("").to_string() };
    crate::target::validate_target_decorator(deco, &resolve, &mut self.diagnostics);
}
```

Wire into `lib.rs`:
```rust
pub mod target;
```

Add `--target` flag to CLI `Build` and `Run` commands:
```rust
/// M47: GPU target backend (cuda, rocm, metal, webgpu)
#[arg(long, default_value = "cuda")]
target: String,
```

Add to `CompileOptions`:
```rust
/// M47: GPU compilation target.
pub target: crate::gpu_target::GpuTarget,
```
Initialize: `target: crate::gpu_target::GpuTarget::Cuda`.

Parse in CLI before passing to compiler:
```rust
let target = nsl_codegen::gpu_target::GpuTarget::from_str(&target_str)
    .unwrap_or_else(|| {
        eprintln!("unknown target '{}', expected: cuda, rocm, metal, webgpu", target_str);
        std::process::exit(1);
    });
```

---

## Phase 4: Wire Into Codegen + Build Verification

### Task 7: Wire Modules

**Files:**
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 7: Add module declarations to codegen lib.rs**

```rust
pub mod kernel_ir;
pub mod kernel_lower;
pub mod backend_ptx;
pub mod gpu_target;
```

### Task 8: Build Verification

- [ ] **Step 8: `cargo build` — verify no compile errors**

- [ ] **Step 9: `cargo test` — run all tests, expect 25+ new tests passing**

Expected new tests:
- `gpu_target::tests::*` (5 tests: target parse, cuda features, webgpu features, missing detection, feature names)
- `kernel_ir::tests::*` (7 tests: builder, op_count, no features, barrier feature, shuffle feature, type sizes, PTX mapping)
- `backend_ptx::tests::*` (5 tests: simple add, thread indexing, shared memory, null terminated, barrier)
- `kernel_lower::tests::*` (4 tests: empty kernel, params, basic ops, feature tracking)
- `gpu_backend::tests::*` (2 tests: error display, kernel arg variants)
- `target::tests::*` (wired through checker — no standalone tests, validated via semantic)

- [ ] **Step 10: `cargo clippy` — no warnings**

---

## Verification Checklist

After implementation, verify:

1. **KIR construction**: KirBuilder produces well-formed IR with correct params, blocks, ops, terminators
2. **Feature detection**: Barrier sets SHARED_MEMORY, WarpShuffle sets WARP_SHUFFLE, tracked automatically
3. **Feature gating**: GpuTarget.features() correctly reports capabilities per backend
4. **PTX emission**: KIR→PTX produces valid PTX text with correct instruction mapping
5. **GpuTarget parsing**: CLI strings parse to correct enum variants
6. **GpuBackend trait**: Trait compiles with correct associated types
7. **@target validation**: Invalid backend names produce clear diagnostics
8. **CompileOptions**: target field flows from CLI to compiler
9. **No regressions**: All 556+ existing tests pass (existing kernel.rs path unchanged)
