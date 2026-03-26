use std::collections::HashMap;
use cranelift_codegen::ir::{self, types as cl_types};
use cranelift_frontend::Variable;
use nsl_ast::Symbol;

/// Per-function loop tracking for break/continue.
pub struct LoopContext {
    /// Block to jump to on `continue`. For while loops this is the header (condition check).
    /// For for-loops this is the increment block (so counter gets incremented before rechecking).
    pub continue_block: ir::Block,
    pub exit_block: ir::Block,
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

/// Per-function compilation state (variables, loops).
pub struct FuncState {
    pub variables: HashMap<Symbol, (Variable, cl_types::Type)>,
    pub var_counter: usize,
    pub loop_stack: Vec<LoopContext>,
    pub current_block: Option<ir::Block>,
    /// True when compiling a @no_grad function body — emit tape_resume before returns.
    pub is_no_grad: bool,
    /// Cranelift Values of intermediate tensor results that need freeing after the current statement.
    /// Prevents memory leaks from compound expressions like `a + b + c`.
    pub tensor_temporaries: Vec<ir::Value>,
    /// Cranelift Values of DataLoader pointers that need stop+free at scope exit.
    pub dataloader_vars: Vec<ir::Value>,
    /// Stack of scope markers into tensor_temporaries, pushed on loop entry, popped on exit.
    /// Used to emit cleanup at break/continue without corrupting compiler state.
    pub temp_scope_stack: Vec<usize>,
    /// True when compiling a custom datatype method body (pack/unpack).
    /// Disables the "both indeterminate types → tensor" heuristic for binary ops.
    pub in_dtype_method: bool,
    /// True when compiling an element-wise unpack method body.
    /// Return statements should bitcast f64→i64 before returning.
    pub dtype_unpack_ret_bitcast: bool,
    /// True when compiling an expression via the unfused (training) path inside try_auto_fuse.
    /// Prevents infinite recursion when compile_expr re-enters try_auto_fuse.
    pub in_fuse_bypass: bool,
    /// Tracks resolved symbolic dimensions for M28 dynamic shapes assertions.
    pub symbolic_dims: crate::dynamic_shapes::SymbolicDimTracker,
    /// True when compiling a train block step body (inside tape recording).
    /// Suppresses freeing tensor temporaries since backward needs them alive.
    pub in_tape_region: bool,
    /// True when inside a DataLoader for-loop with runtime scope tracking.
    /// Suppresses codegen-level tensor_temporaries cleanup to avoid double-free
    /// with scope_end (which handles all cleanup at end of each iteration).
    pub in_scoped_loop: bool,
    /// M38b: Ownership lowering state for linear types free-at-consumption.
    /// `Some` when `--linear-types` is active and the function has ownership metadata.
    pub ownership_lowering: Option<crate::ownership::OwnershipLowering>,
    /// M35: True when compiling a function with @fp8_compute decorator.
    /// MatMul ops use nsl_fp8_matmul_training for E5M2 backward tape recording.
    pub is_fp8_compute: bool,
    /// FBIP Phase 2: Per-binding use counts for single-use optimization.
    /// When a binding is referenced exactly once, the codegen can emit in-place
    /// op variants and skip clones.
    pub use_counts: Option<crate::use_count::UseCountMap>,
    /// M36: Cranelift Variable holding the GPU slab base pointer (from nsl_gpu_slab_init).
    /// When Some, slab-planned tensors use offsets into this slab instead of alloc_managed.
    pub slab_ptr_var: Option<Variable>,
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
            var_counter: 0,
            loop_stack: Vec::new(),
            current_block: None,
            is_no_grad: false,
            tensor_temporaries: Vec::new(),
            dataloader_vars: Vec::new(),
            temp_scope_stack: Vec::new(),
            in_dtype_method: false,
            dtype_unpack_ret_bitcast: false,
            in_fuse_bypass: false,
            symbolic_dims: crate::dynamic_shapes::SymbolicDimTracker::new(),
            in_tape_region: false,
            in_scoped_loop: false,
            ownership_lowering: None,
            is_fp8_compute: false,
            use_counts: None,
            slab_ptr_var: None,
        }
    }

    /// Allocate a new Cranelift Variable index.
    pub fn new_variable(&mut self) -> Variable {
        let var = Variable::from_u32(self.var_counter as u32);
        self.var_counter += 1;
        var
    }
}
