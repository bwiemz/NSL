// crates/nsl-kir/src/kernel_ir.rs
//! M47: Backend-agnostic Kernel IR -- SSA-form intermediate representation
//! for GPU compute kernels.

use crate::FeatureSet;

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
    /// Roadmap A2 step 5: `.maxntid` / `.minnctapersm` for the entry.
    pub launch_bounds: Option<LaunchBounds>,
    /// Roadmap A2 step 5: `.maxnreg` for the entry — the per-thread
    /// register cap `ptxas` allocates under.
    pub max_registers: Option<u32>,
    _next_var: VarId,
}

/// Launch bounds printed on the entry (roadmap A2 step 5): the largest
/// block the kernel is launched with, and optionally the number of blocks
/// per SM it must fit — the hand estate's `.maxntid N, 1, 1` /
/// `.minnctapersm M` pair that sets the register budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchBounds {
    pub max_threads: u32,
    pub min_blocks_per_sm: Option<u32>,
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
    /// Roadmap A2 step 2: the block's parameters — SSA values defined at
    /// block entry and given a value by every edge into the block (the
    /// `args` of the predecessor's `KirEdge`, in order). This is how a
    /// loop-carried value is written in SSA without phi nodes: the loop
    /// header takes the induction variable as a parameter, the entry edge
    /// passes its initial value and the back edge passes the next one.
    /// Block 0 (the entry) has none.
    pub params: Vec<KirBlockParam>,
    pub ops: Vec<KirOp>,
    pub terminator: Option<KirTerminator>,
}

/// One block parameter (roadmap A2 step 2): `id` is defined at the entry of
/// the block that lists it, typed `ty`, and `KirBuilder::add_block_param`
/// records the type in `var_types` like any other typed variable.
#[derive(Debug, Clone, PartialEq)]
pub struct KirBlockParam {
    pub id: VarId,
    pub ty: KirType,
}

/// A control-flow edge (roadmap A2 step 2): the target block and the
/// arguments for its parameters, in order. `KirEdge::to(b)` is an edge
/// with no arguments — `b.into()` spells the same — and `KirEdge::with(b,
/// args)` passes values. The verifier holds the argument count and types
/// to the target's parameter list (rule 7); the printers implement the
/// edge as a parallel copy into the parameter registers before the jump.
#[derive(Debug, Clone, PartialEq)]
pub struct KirEdge {
    pub target: BlockId,
    pub args: Vec<VarId>,
}

impl KirEdge {
    /// An edge to `target` that passes no arguments.
    pub fn to(target: BlockId) -> Self {
        KirEdge { target, args: Vec::new() }
    }

    /// An edge to `target` passing `args` for its parameters, in order.
    pub fn with(target: BlockId, args: Vec<VarId>) -> Self {
        KirEdge { target, args }
    }
}

impl From<BlockId> for KirEdge {
    fn from(target: BlockId) -> Self {
        KirEdge::to(target)
    }
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
    // M57 v1: narrow signed integer types for FPGA INT8 quantized inference.
    // I8  → layer-1 weight/activation dtype (i8×i8→i32, spec §4.6 layer 1)
    // I16 → intermediate headroom dtype (spec §4.6 headroom math)
    I8,
    I16,
    F16,
    Bf16,
    F32,
    F64,
    Bool,
    Ptr(Box<KirType>, AddressSpace),
    Vec(Box<KirType>, u32),
    // BitNet M35.1: packed ternary representation (4 trits per byte, 2 bits per trit).
    // At-rest format in HBM; unpacked into `TernaryUnpacked` for compute.
    // Layout per docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §2.1
    // (high-bits-first ordering verified against bitnet.cpp).
    //
    // Naming rationale: KIR keeps the encoding density (`Tq2`) in the variant
    // name because spec §7.1 reserves a future `Tq5Packed` variant (5 trits per
    // byte, the information-theoretic optimum — ~20% bandwidth efficiency
    // improvement). The semantic-layer counterpart is `Type::TernaryPacked`
    // (single variant, no density suffix) because user source code thinks at
    // dtype-level and shouldn't expose packing choices. The codegen bridge in
    // `crates/nsl-codegen/src/types.rs` maps both `Type::TernaryPacked` and
    // `Type::TernaryUnpacked` to `types::I8` (1 byte per storage atom).
    Tq2Packed,
    // BitNet M35.1: one trit per i8 (or register slot). Compute-time format.
    // Conversions to/from Tq2Packed are explicit ops via `bitnet::pack`/`unpack`.
    // Semantic-layer counterpart: `Type::TernaryUnpacked` (same name).
    TernaryUnpacked,
}

impl KirType {
    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            KirType::Bool | KirType::I8 | KirType::Tq2Packed | KirType::TernaryUnpacked => 1,
            KirType::I16 | KirType::F16 | KirType::Bf16 => 2,
            KirType::U32 | KirType::I32 | KirType::F32 => 4,
            KirType::U64 | KirType::I64 | KirType::F64 => 8,
            KirType::Ptr(_, _) => 8,
            KirType::Vec(inner, n) => inner.size_bytes() * *n as usize,
        }
    }

    /// PTX register prefix.
    pub fn ptx_reg_prefix(&self) -> &'static str {
        match self {
            KirType::U32 | KirType::I32
            | KirType::I8 | KirType::I16
            | KirType::Tq2Packed | KirType::TernaryUnpacked => "%r",
            // `setp` writes and `@%p` reads the predicate class (roadmap A2
            // step 5: the prefix said `%r` while every printer used `%p`).
            KirType::Bool => "%p",
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
            // PTX spells the signed integers `.s32` / `.s64`; `.i32` is not
            // a type (fixed with roadmap A2 step 4, when `Cast` to `I32`
            // first went through `ptxas`).
            KirType::I32 => "s32",
            KirType::U64 => "u64",
            KirType::I64 => "s64",
            KirType::I8 => "s8",
            KirType::I16 => "s16",
            KirType::F16 => "f16",
            KirType::Bf16 => "bf16",
            KirType::F32 => "f32",
            KirType::F64 => "f64",
            KirType::Bool => "pred",
            KirType::Ptr(_, _) => "u64",
            KirType::Vec(_, _) => "b32",
            KirType::Tq2Packed => "b8",
            KirType::TernaryUnpacked => "s8",
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
    /// Warp shuffle: `dst = val` from the lane selected by `mode` and
    /// `lane` (a `U32`: the delta for `Down`/`Up`, the XOR mask for `Xor`,
    /// the source lane for `Idx`) within a segment of `width` lanes
    /// (32 = the whole warp). Roadmap A2 step 4; the pre-step form was the
    /// `Down`, width-32 case.
    WarpShuffle { dst: VarId, val: VarId, lane: VarId, mode: ShuffleMode, width: u8 },
    /// Warp vote over `pred` (a `Bool`): `Any` / `All` produce a `Bool`,
    /// `Ballot` a `U32` bit per lane. Roadmap A2 step 4.
    Vote { dst: VarId, pred: VarId, mode: VoteMode },
    /// The lane index within the warp (`%laneid`, `U32`). Roadmap A2 step 4.
    LaneId(VarId),
    /// The warp index within the block (`%warpid`, `U32`). Not stable across
    /// context switches on every part; the estate mostly derives it from
    /// the thread index. Roadmap A2 step 4.
    WarpId(VarId),

    // Roadmap A2 step 4: the integer / bitwise ISA the hand estate's index
    // math and reductions are made of. All homogeneous (dst, a, b):
    // `And`/`Or`/`Xor` on integers or `Bool`s, `Not` on the same, `Rem` on
    // integers, `Min`/`Max` on any numeric type; `Shl`/`Shr` take a `U32`
    // shift amount and `Shr` is arithmetic for signed types.
    And(VarId, VarId, VarId),
    Or(VarId, VarId, VarId),
    Xor(VarId, VarId, VarId),
    Not(VarId, VarId),
    Shl(VarId, VarId, VarId),
    Shr(VarId, VarId, VarId),
    Rem(VarId, VarId, VarId),
    Min(VarId, VarId, VarId),
    Max(VarId, VarId, VarId),
    /// `dst = 1 / src` (`F32`: `rcp.approx`, `F64`: `rcp.rn`).
    Rcp(VarId, VarId),
    /// `dst = 1 / sqrt(src)` (`rsqrt.approx`).
    Rsqrt(VarId, VarId),
    /// A conversion with an explicit rounding mode; `Cast` picks the
    /// default (`Rn` for anything that can round, `Rzi` for float → int).
    CastRounded { dst: VarId, src: VarId, ty: KirType, mode: RoundMode },
    /// Vector load: `dsts.len()` (2 or 4) consecutive pointee-typed values
    /// from `ptr` (`ld.{space}.v{n}`). Each destination is its own scalar.
    LoadVec { dsts: Vec<VarId>, ptr: VarId, space: AddressSpace },
    /// Vector store of `vals.len()` (2 or 4) values to `ptr`.
    StoreVec { ptr: VarId, vals: Vec<VarId>, space: AddressSpace },
    /// Roadmap A2 step 4: a predicated side effect — `op` runs only when
    /// `pred` (a `Bool`) is true (`negate`: false). Only ops with no
    /// destination may be predicated (`Store`, `StoreVec`, `AtomicAdd`,
    /// `CpAsync`, the barriers); a predicated definition would be partial,
    /// which SSA forbids — use `Select` or a branch for a value.
    Predicated { pred: VarId, negate: bool, op: Box<KirOp> },

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

    // Roadmap A2 step 2: the async-copy group as first-class ops, so the
    // verifier can check the commit/wait discipline the hand-PTX had to
    // get right by inspection. `cp.async` on sm_80+ (FeatureSet::ASYNC_COPY).
    /// dst = the address of the kernel's shared-memory block (`shared_mem`),
    /// typed `Ptr(_, Shared)` by the producer.
    SharedBase(VarId),
    /// Copy `bytes` (4, 8 or 16) from a global address into a shared one
    /// without staging through registers. The copy is complete only after
    /// a `CpAsyncWait` that covers the group it is committed into.
    CpAsync { dst: VarId, src: VarId, bytes: u8 },
    /// Close the current async-copy group (every `CpAsync` since the last
    /// commit, or since the block began).
    CpAsyncCommit,
    /// Block until at most `pending` committed groups are still in flight
    /// (`pending == 0`: everything has landed).
    CpAsyncWait { pending: u8 },

    // Roadmap A2 step 2: tensor-core ops with shape-checked fragment types.
    // Fragments are `Vec(F16, 2)` (one packed f16x2 per .b32 register) for
    // the A/B operands and `F32` for the accumulator; the verifier holds
    // every listed register to that type (FeatureSet::TENSOR_CORES, sm_80).
    /// Warp-collective load of four 8x8 b16 matrices from shared memory:
    /// `ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16 {dst}, [addr]`.
    /// `addr` is a `Ptr(_, Shared)`; each `dst` is a `Vec(F16, 2)`.
    LdMatrixX4 { dst: [VarId; 4], addr: VarId, trans: bool },
    /// Warp-collective `d = a * b + c` on one m16n8k16 tile:
    /// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {d}, {a}, {b}, {c}`.
    /// `a` (4) and `b` (2) are `Vec(F16, 2)` fragments; `c` and `d` (4 each)
    /// are `F32`.
    MmaF16M16N8K16 { d: [VarId; 4], a: [VarId; 4], b: [VarId; 2], c: [VarId; 4] },

    // M57 v1: structured ops for FPGA target. GPU/CPU codegen ignores these
    // (existing AST → templated PTX path for GPU; Cranelift for CPU);
    // FPGA codegen consumes them in the KIR → HIR pass.
    Matmul {
        a: VarId, b: VarId, out: VarId,
        a_dtype: KirType, b_dtype: KirType, out_dtype: KirType,
        a_shape: [usize; 2], b_shape: [usize; 2],   // rank-2 hardcoded for v1
    },
    ElementwiseAdd {
        a: VarId, b: VarId, out: VarId,
        dtype: KirType, shape: [usize; 1],          // rank-1 hardcoded for v1
    },
    Relu {
        a: VarId, out: VarId,
        dtype: KirType, shape: [usize; 1],          // rank-1 hardcoded for v1
    },
}

#[derive(Debug, Clone)]
pub enum KirTerminator {
    /// Unconditional jump along one edge.
    Branch(KirEdge),
    /// `if cond { taken } else { fallthrough }`; `cond` is a `Bool`.
    CondBranch(VarId, KirEdge, KirEdge),
    Return,
}

impl KirTerminator {
    /// The edges this terminator leaves along, in (taken, not-taken) order.
    pub fn edges(&self) -> Vec<&KirEdge> {
        match self {
            KirTerminator::Branch(e) => vec![e],
            KirTerminator::CondBranch(_, t, f) => vec![t, f],
            KirTerminator::Return => vec![],
        }
    }

    /// Whether any edge passes block arguments.
    pub fn has_args(&self) -> bool {
        self.edges().iter().any(|e| !e.args.is_empty())
    }
}

/// Rounding for `CastRounded` (roadmap A2 step 4): the PTX `.rn` / `.rz` /
/// `.rm` / `.rp` modifiers for float results, and their `.rni` / `.rzi` /
/// `.rmi` / `.rpi` forms for a float → integer conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    /// To nearest even.
    Rn,
    /// Toward zero.
    Rz,
    /// Toward negative infinity.
    Rm,
    /// Toward positive infinity.
    Rp,
}

/// Lane selection for `WarpShuffle` (roadmap A2 step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleMode {
    /// From `lane_id + delta` (`shfl.sync.down`).
    Down,
    /// From `lane_id - delta` (`shfl.sync.up`).
    Up,
    /// From `lane_id ^ mask` (`shfl.sync.bfly`).
    Xor,
    /// From the given lane (`shfl.sync.idx`).
    Idx,
}

/// The warp vote a `Vote` takes (roadmap A2 step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteMode {
    Any,
    All,
    Ballot,
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
    launch_bounds: Option<LaunchBounds>,
    max_registers: Option<u32>,
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
            launch_bounds: None,
            max_registers: None,
        }
    }

    /// Roadmap A2 step 5: print `.maxntid max_threads, 1, 1` (and
    /// `.minnctapersm` when given) on the entry.
    pub fn set_launch_bounds(&mut self, max_threads: u32, min_blocks_per_sm: Option<u32>) {
        self.launch_bounds = Some(LaunchBounds { max_threads, min_blocks_per_sm });
    }

    /// Roadmap A2 step 5: print `.maxnreg n` on the entry.
    pub fn set_max_registers(&mut self, n: u32) {
        self.max_registers = Some(n);
    }

    pub fn new_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    /// Allocate a new typed variable. The type is recorded for backend use.
    pub fn new_typed_var(&mut self, ty: KirType) -> VarId {
        let id = self.new_var();
        if ty == KirType::Bf16 {
            self.required_features |= FeatureSet::BF16_ARITHMETIC;
        }
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
            params: Vec::new(),
            ops: Vec::new(),
            terminator: None,
        });
        id
    }

    /// Roadmap A2 step 2: add a typed parameter to `block` and return the
    /// `VarId` it defines at the block's entry. Every edge into `block`
    /// must then pass one argument per parameter, in order (verifier
    /// rule 7). The type is recorded in `var_types`.
    pub fn add_block_param(&mut self, block: BlockId, ty: KirType) -> VarId {
        let id = self.new_typed_var(ty.clone());
        self.blocks[block as usize].params.push(KirBlockParam { id, ty });
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
            KirOp::WarpShuffle { .. } | KirOp::Vote { .. } => {
                self.required_features |= FeatureSet::WARP_SHUFFLE
            }
            // Roadmap A2 step 4: a bf16 conversion is a PTX 7.8 / sm_80
            // instruction; the printer bumps the header when the kernel
            // requires the feature.
            KirOp::Cast(_, src, ty) | KirOp::CastRounded { src, ty, .. }
                if *ty == KirType::Bf16 || self.var_types.get(src) == Some(&KirType::Bf16) =>
            {
                self.required_features |= FeatureSet::BF16_ARITHMETIC;
            }
            KirOp::SharedMemFence => self.required_features |= FeatureSet::SHARED_MEMORY,
            KirOp::SharedBase(_) => self.required_features |= FeatureSet::SHARED_MEMORY,
            KirOp::CpAsync { .. } | KirOp::CpAsyncCommit | KirOp::CpAsyncWait { .. } => {
                self.required_features |= FeatureSet::SHARED_MEMORY | FeatureSet::ASYNC_COPY
            }
            KirOp::LdMatrixX4 { .. } => {
                self.required_features |= FeatureSet::SHARED_MEMORY | FeatureSet::TENSOR_CORES
            }
            KirOp::MmaF16M16N8K16 { .. } => self.required_features |= FeatureSet::TENSOR_CORES,
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
            launch_bounds: self.launch_bounds,
            max_registers: self.max_registers,
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

    /// Iterate all ops across all blocks in order.
    /// M57: consumed by the KIR → HIR pass (v1 assumes single-block KIR;
    /// multi-block KIR is out of scope until a future milestone).
    pub fn ops(&self) -> impl Iterator<Item = &KirOp> {
        self.blocks.iter().flat_map(|b| b.ops.iter())
    }

    /// Run the KIR verifier (`crate::kir_verify`): shape, SSA,
    /// def-before-use under dominance, and operand typing. `Ok(())` or
    /// every violation found.
    pub fn verify(&self) -> Result<(), Vec<crate::kir_verify::KirVerifyError>> {
        crate::kir_verify::verify(self)
    }

    /// Roadmap A2 step 5: how many registers of each class the kernel holds
    /// live at once, from the allocator (`crate::regalloc`). This is the
    /// number the PTX printer declares per class and the figure a register
    /// budget is checked against.
    pub fn register_pressure(&self) -> crate::regalloc::RegisterPressure {
        crate::regalloc::allocate(self).pressure()
    }

    /// `verify().is_ok()`. Until roadmap A2 step 2 this checked only that
    /// every block had a terminator; that is now rule 1 of the verifier.
    pub fn is_well_formed(&self) -> bool {
        self.verify().is_ok()
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
        b.terminate(KirTerminator::CondBranch(in_bounds, body.into(), exit.into()));

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
        b.terminate(KirTerminator::Branch(exit.into()));

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
        b.emit(KirOp::WarpShuffle { dst, val: v0, lane: v1, mode: ShuffleMode::Down, width: 32 });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert!(ir.required_features.contains(FeatureSet::WARP_SHUFFLE));
    }

    /// Roadmap A2 step 2: a grid-stride loop is a header block with the
    /// induction variable as a parameter; the entry edge passes the first
    /// index and the back edge passes the next one.
    #[test]
    fn block_params_carry_a_loop_variable() {
        let mut b = KirBuilder::new("loop");
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let idx = b.add_block_param(header, KirType::U32);
        b.set_block(entry);
        let start = b.new_typed_var(KirType::U32);
        b.emit(KirOp::GlobalId(start, 0));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![start])));
        b.set_block(header);
        let more = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(more, idx, n, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(more, body.into(), exit.into()));
        b.set_block(body);
        let stride = b.new_typed_var(KirType::U32);
        b.emit(KirOp::BlockDim(stride, 0));
        let next = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Add(next, idx, stride));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![next])));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.blocks[header as usize].params, vec![KirBlockParam { id: idx, ty: KirType::U32 }]);
        assert_eq!(ir.var_types.get(&idx), Some(&KirType::U32));
        assert!(ir.blocks[body as usize].terminator.as_ref().unwrap().has_args());
        assert!(!ir.blocks[header as usize].terminator.as_ref().unwrap().has_args());
        assert!(ir.is_well_formed(), "{:?}", ir.verify());
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

#[cfg(test)]
mod m57_v1_tests {
    use super::*;

    #[test]
    fn matmul_variant_carries_shape_and_dtype() {
        let op = KirOp::Matmul {
            a: 1, b: 2, out: 3,
            a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
            a_shape: [1, 784], b_shape: [784, 128],
        };
        match op {
            KirOp::Matmul { a_shape, b_shape, .. } => {
                assert_eq!(a_shape[1], 784);
                assert_eq!(b_shape[1], 128);
            }
            _ => panic!("expected Matmul variant"),
        }
    }

    #[test]
    fn elementwise_add_variant_is_rank_1() {
        let op = KirOp::ElementwiseAdd {
            a: 1, b: 2, out: 3,
            dtype: KirType::I32, shape: [128],
        };
        match op {
            KirOp::ElementwiseAdd { shape, .. } => assert_eq!(shape[0], 128),
            _ => panic!("expected ElementwiseAdd"),
        }
    }

    #[test]
    fn relu_variant_is_rank_1() {
        let op = KirOp::Relu {
            a: 1, out: 2,
            dtype: KirType::I32, shape: [128],
        };
        match op {
            KirOp::Relu { shape, .. } => assert_eq!(shape[0], 128),
            _ => panic!("expected Relu"),
        }
    }

    #[test]
    fn i8_type_has_correct_size_and_ptx() {
        assert_eq!(KirType::I8.size_bytes(), 1);
        assert_eq!(KirType::I8.ptx_reg_prefix(), "%r");
        assert_eq!(KirType::I8.ptx_type(), "s8");
    }

    #[test]
    fn i16_type_has_correct_size_and_ptx() {
        assert_eq!(KirType::I16.size_bytes(), 2);
        assert_eq!(KirType::I16.ptx_reg_prefix(), "%r");
        assert_eq!(KirType::I16.ptx_type(), "s16");
    }
}
