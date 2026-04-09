// crates/nsl-codegen/src/kernel_ir.rs
//! M47: Backend-agnostic Kernel IR -- SSA-form intermediate representation
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
    /// Type of each VarId -- populated by KirBuilder as operations are emitted.
    /// Required by backends to emit typed instructions (e.g., `add.f32` vs `add.u32`).
    pub var_types: std::collections::HashMap<VarId, KirType>,
    pub shared_mem_bytes: u32,
    pub workgroup_size: [u32; 3],
    pub required_features: FeatureSet,
    _next_var: VarId,
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
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
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
    Fma(VarId, VarId, VarId, VarId), // dst = a * b + c
    Neg(VarId, VarId),
    Abs(VarId, VarId),

    // Math functions (dst, src)
    Sqrt(VarId, VarId),
    Exp(VarId, VarId),
    Log(VarId, VarId),
    Sin(VarId, VarId),
    Cos(VarId, VarId),
    Tanh(VarId, VarId),
    Pow(VarId, VarId, VarId), // dst = base^exp

    // Type conversion
    Cast(VarId, VarId, KirType),

    // Memory
    Load(VarId, VarId, AddressSpace),
    Store(VarId, VarId, AddressSpace), // *ptr = val
    AtomicAdd(VarId, VarId, AddressSpace),

    // Thread indexing (dst, dim: 0=x, 1=y, 2=z)
    ThreadId(VarId, u8),
    BlockIdx(VarId, u8),
    BlockDim(VarId, u8),
    GridDim(VarId, u8),
    GlobalId(VarId, u8), // blockIdx*blockDim + threadIdx

    // Synchronization
    Barrier,
    WarpShuffle(VarId, VarId, VarId), // dst = shuffle_down(val, offset)

    // Comparison (dst, a, b, op)
    Cmp(VarId, VarId, VarId, CmpOp),
    // Select (dst, cond, true_val, false_val)
    Select(VarId, VarId, VarId, VarId),

    // Constants
    Const(VarId, KirConst),

    // Pointer arithmetic
    PtrOffset(VarId, VarId, VarId), // dst = base + offset * sizeof(pointee)

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
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
pub struct KirConst {
    pub ty: KirType,
    pub value: ConstValue,
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// KirBuilder -- programmatic IR construction
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

    /// Look up the type of a VarId, if recorded.
    pub fn var_type(&self, id: VarId) -> Option<KirType> {
        self.var_types.get(&id).cloned()
    }

    pub fn add_param(&mut self, name: &str, ty: KirType, address_space: AddressSpace) -> VarId {
        let id = self.new_var();
        self.var_types.insert(id, ty.clone());
        self.params.push(KirParam {
            id,
            name: name.to_string(),
            ty,
            address_space,
        });
        id
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = self.blocks.len() as BlockId;
        self.blocks.push(KirBlock {
            id,
            ops: Vec::new(),
            terminator: None,
        });
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
            _next_var: self.next_var,
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
        let a_ptr = b.add_param(
            "a",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        let b_ptr = b.add_param(
            "b",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        let out_ptr = b.add_param(
            "out",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
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
        assert_eq!(
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global).size_bytes(),
            8
        );
    }

    #[test]
    fn kir_type_ptx_mapping() {
        assert_eq!(KirType::F32.ptx_type(), "f32");
        assert_eq!(KirType::U32.ptx_type(), "u32");
        assert_eq!(KirType::F32.ptx_reg_prefix(), "%f");
        assert_eq!(KirType::U32.ptx_reg_prefix(), "%r");
        assert_eq!(
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global).ptx_reg_prefix(),
            "%rd"
        );
    }
}
