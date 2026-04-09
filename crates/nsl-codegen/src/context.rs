use cranelift_codegen::ir::{self, types as cl_types};
use cranelift_frontend::Variable;
use nsl_ast::Symbol;
use std::collections::{HashMap, HashSet};

use crate::ownership_expr::Ownership;

/// Per-function loop tracking for break/continue.
pub struct LoopContext {
    /// Block to jump to on `continue`. For while loops this is the header (condition check).
    /// For for-loops this is the increment block (so counter gets incremented before rechecking).
    pub continue_block: ir::Block,
    pub exit_block: ir::Block,
    /// Active DataLoader batch dict for this loop, when applicable.
    pub batch_var: Option<Variable>,
}

/// Struct field layout info.
#[derive(Clone)]
pub struct StructField {
    pub name: String,
    pub cl_type: cl_types::Type,
    pub offset: usize,
}

/// Struct memory layout.
#[derive(Clone)]
pub struct StructLayout {
    pub name: String,
    pub fields: Vec<StructField>,
    pub total_size: usize,
}

/// Tensor temporary tracking and cleanup state.
/// Manages intermediate tensor results that need freeing after statements,
/// DataLoader pointers, and scoped cleanup for loops.
pub struct TensorCleanupState {
    /// Cranelift Values of intermediate tensor results that need freeing after the current statement.
    /// Prevents memory leaks from compound expressions like `a + b + c`.
    pub tensor_temporaries: Vec<ir::Value>,
    /// Cranelift Values of DataLoader pointers that need stop+free at scope exit.
    pub dataloader_vars: Vec<ir::Value>,
    /// Stack of scope markers into tensor_temporaries, pushed on loop entry, popped on exit.
    /// Used to emit cleanup at break/continue without corrupting compiler state.
    pub temp_scope_stack: Vec<usize>,
    /// Non-loop batch dict variables that must be freed on explicit return.
    pub active_batch_vars: Vec<Variable>,
    /// True when inside a DataLoader for-loop with runtime scope tracking.
    /// Suppresses codegen-level tensor_temporaries cleanup to avoid double-free
    /// with scope_end (which handles all cleanup at end of each iteration).
    pub in_scoped_loop: bool,

    /// ELTLS (spec §4.2): per-Value ownership state. Populated by producer
    /// sites in compile_*. Missing entries are treated as Ownership::Unknown
    /// (strict fallback).
    pub expr_ownership: HashMap<ir::Value, Ownership>,

    /// ELTLS: deterministic list of Values currently in the Owned state.
    /// Iterated at statement boundaries in insertion order for stable,
    /// test-friendly cleanup.
    ///
    /// Invariant: for every v in owned_temporaries,
    ///   expr_ownership[v] == Ownership::Owned.
    /// Promotion to TapeHeld removes from this Vec.
    /// Consumption by a linear-last-use FFI removes from this Vec.
    pub owned_temporaries: Vec<ir::Value>,

    /// ELTLS: Values in TapeHeld state, freed at tape-region exit.
    pub tape_held: Vec<ir::Value>,

    /// ELTLS: per-function counter of consumer sites that hit Unknown.
    /// Goal: zero for hot-path functions after full rollout.
    pub unknown_ownership_count: usize,
}

impl TensorCleanupState {
    pub fn new() -> Self {
        TensorCleanupState {
            tensor_temporaries: Vec::new(),
            dataloader_vars: Vec::new(),
            temp_scope_stack: Vec::new(),
            active_batch_vars: Vec::new(),
            in_scoped_loop: false,
            expr_ownership: HashMap::new(),
            owned_temporaries: Vec::new(),
            tape_held: Vec::new(),
            unknown_ownership_count: 0,
        }
    }
}

impl Default for TensorCleanupState {
    fn default() -> Self {
        Self::new()
    }
}

/// Ownership and linear-type lowering state.
/// Tracks linear tensor consumption and sparse variable classification.
pub struct OwnershipState {
    /// M38b: Ownership lowering state for linear types free-at-consumption.
    /// `Some` when `--linear-types` is active and the function has ownership metadata.
    pub lowering: Option<crate::ownership::OwnershipLowering>,
    /// M38b: Cranelift Values of linear tensors consumed in the current statement.
    /// After statement-level cleanup, these are freed via `nsl_tensor_free` instead
    /// of waiting for scope exit. Only populated when `lowering.is_some()`.
    pub linear_consume_pending: Vec<ir::Value>,
    /// M50: Set of Symbol names that are known sparse tensor variables.
    /// Populated when codegen emits sparse_from_dense, sparse_coo, or format conversion calls
    /// assigned to a variable. Checked during binary op dispatch to route to sparse ops.
    pub sparse_vars: HashSet<Symbol>,
}

impl OwnershipState {
    pub fn new() -> Self {
        OwnershipState {
            lowering: None,
            linear_consume_pending: Vec::new(),
            sparse_vars: HashSet::new(),
        }
    }
}

impl Default for OwnershipState {
    fn default() -> Self {
        Self::new()
    }
}

/// Boolean flags that control codegen behavior for the current function.
/// Groups all simple on/off switches that modify compilation strategy.
pub struct CodegenFlags {
    /// True when compiling a @no_grad function body — emit tape_resume before returns.
    pub is_no_grad: bool,
    /// True when compiling a custom datatype method body (pack/unpack).
    /// Disables the "both indeterminate types → tensor" heuristic for binary ops.
    pub in_dtype_method: bool,
    /// True when compiling an element-wise unpack method body.
    /// Return statements should bitcast f64→i64 before returning.
    pub dtype_unpack_ret_bitcast: bool,
    /// True when compiling an expression via the unfused (training) path inside try_auto_fuse.
    /// Prevents infinite recursion when compile_expr re-enters try_auto_fuse.
    pub in_fuse_bypass: bool,
    /// True when compiling a train block step body (inside tape recording).
    /// Suppresses freeing tensor temporaries since backward needs them alive.
    pub in_tape_region: bool,
    /// Nesting depth inside branch-only control flow bodies such as if/elif/else and match arms.
    pub conditional_depth: usize,
    /// True while compiling code that executes once per yielded DataLoader batch inside train blocks.
    pub in_dataloader_batch_scope: bool,
    /// M35: True when compiling a function with @fp8_compute decorator.
    /// MatMul ops use nsl_fp8_matmul_training for E5M2 backward tape recording.
    pub is_fp8_compute: bool,
}

impl CodegenFlags {
    pub fn new() -> Self {
        CodegenFlags {
            is_no_grad: false,
            in_dtype_method: false,
            dtype_unpack_ret_bitcast: false,
            in_fuse_bypass: false,
            in_tape_region: false,
            conditional_depth: 0,
            in_dataloader_batch_scope: false,
            is_fp8_compute: false,
        }
    }
}

impl Default for CodegenFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-function compilation state (variables, loops).
pub struct FuncState {
    pub variables: HashMap<Symbol, (Variable, cl_types::Type)>,
    pub param_symbols: HashSet<Symbol>,
    pub non_owning_symbols: HashSet<Symbol>,
    pub dataloader_symbols: HashSet<Symbol>,
    pub borrowed_batch_symbols: HashSet<Symbol>,
    pub var_counter: usize,
    pub loop_stack: Vec<LoopContext>,
    pub current_block: Option<ir::Block>,
    /// Tensor temporary and cleanup tracking.
    pub cleanup: TensorCleanupState,
    /// Ownership and linear-type lowering state.
    pub ownership: OwnershipState,
    /// Boolean flags controlling codegen behavior.
    pub flags: CodegenFlags,
    /// Tracks resolved symbolic dimensions for M28 dynamic shapes assertions.
    pub symbolic_dims: crate::dynamic_shapes::SymbolicDimTracker,
    /// FBIP Phase 2: Per-binding use counts for single-use optimization.
    /// When a binding is referenced exactly once, the codegen can emit in-place
    /// op variants and skip clones.
    pub use_counts: Option<crate::use_count::UseCountMap>,
    /// M36: Cranelift Variable holding the GPU slab base pointer (from nsl_gpu_slab_init).
    /// When Some, slab-planned tensors use offsets into this slab instead of alloc_managed.
    pub slab_ptr_var: Option<Variable>,
    /// M52b: Maps Cranelift Value → weight name in WeightMap.
    /// Set when compile_member_access loads a model weight field that exists in the WeightMap.
    pub weight_values: HashMap<cranelift_codegen::ir::Value, String>,
    /// M44: Name of the function currently being compiled.
    /// Used by generate() to look up @grammar decorator configs.
    pub current_function_name: Option<String>,
    /// Semantic types for variables, used by step-variable cleanup to identify tensors.
    pub variable_types: HashMap<Symbol, nsl_semantic::types::Type>,
}

impl Default for FuncState {
    fn default() -> Self {
        Self::new()
    }
}

impl FuncState {
    pub fn new() -> Self {
        FuncState {
            variables: HashMap::new(),
            param_symbols: HashSet::new(),
            non_owning_symbols: HashSet::new(),
            dataloader_symbols: HashSet::new(),
            borrowed_batch_symbols: HashSet::new(),
            var_counter: 0,
            loop_stack: Vec::new(),
            current_block: None,
            cleanup: TensorCleanupState::new(),
            ownership: OwnershipState::new(),
            flags: CodegenFlags::new(),
            symbolic_dims: crate::dynamic_shapes::SymbolicDimTracker::new(),
            use_counts: None,
            slab_ptr_var: None,
            weight_values: HashMap::new(),
            current_function_name: None,
            variable_types: HashMap::new(),
        }
    }

    /// Allocate a new Cranelift Variable index.
    pub fn new_variable(&mut self) -> Variable {
        let var = Variable::from_u32(self.var_counter as u32);
        self.var_counter += 1;
        var
    }
}
