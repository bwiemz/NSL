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
    /// B.2.1: byte offset of the single adapter-sidetable pointer slot.
    /// `None` when no `@adapter` decorator is active in this compile. When
    /// `Some(o)`, the slot at byte offset `o` holds an `i64` pointer to a
    /// heap-allocated table of adapter-tensor pointers, indexed by the linear
    /// position of each synthesized field across all active adapter sites
    /// targeting this model. Field access for synthesized names (`lora_A_*`,
    /// `lora_B_*`, `ia3_scale_*`, `gate_*`) routes through this slot.
    pub adapter_sidetable_offset: Option<usize>,
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

    /// ELTLS v2a (spec §6.2): Values born from a table-classified
    /// OwnedNewResult FFI during the CURRENT statement, registered at the
    /// dispatch boundary (`register_dispatch_result_ownership`). The
    /// nested-expression tracker accepts membership here as proof the
    /// value is an anonymous fresh temporary of this statement — WITHOUT
    /// a per-name allowlist entry. Cleared at every `compile_stmt` entry:
    /// the per-statement lifetime is what makes the check sound. A value
    /// bound to a variable in an earlier statement can be the SAME
    /// Cranelift Value a later identity-shaped dispatch returns (SSA
    /// use_var in one block), and tracking it there would free the live
    /// binding — clearing per statement makes that shape unmatchable.
    pub dispatch_fresh: std::collections::HashSet<ir::Value>,
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
            dispatch_fresh: std::collections::HashSet::new(),
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

/// Controls how `self` and `self.<field>` are lowered inside model methods.
#[derive(Debug, Clone)]
#[derive(Default)]
pub enum SelfResolution {
    /// Normal model method: `self` is bound to block_params[0] (struct pointer)
    /// and field accesses go through `struct_layouts`. Existing behavior.
    #[default]
    StructPointer,
    /// @export model method: `self` is phantom. `self.<field>` lowers to
    /// `load(weight_ptrs + field_index*8)` where `weight_ptrs_var` holds the
    /// Cranelift variable carrying the leading arg.
    WeightPtrsArray {
        weight_ptrs_var: cranelift_frontend::Variable,
    },
}


/// Per-function compilation state (variables, loops).
pub struct FuncState {
    pub variables: HashMap<Symbol, (Variable, cl_types::Type)>,
    /// Scope-aware liveness for FUNCTION-VALUED local bindings (nested
    /// `fn`s, lambda / fn-typed `let`s, params). `variables` is
    /// function-flat by Cranelift-frontend design and NEVER unbinds, but
    /// the CHECKER is block-scoped — a nested `fn sum` declared inside
    /// an if-arm is out of scope after the arm, and a later `sum(t)`
    /// resolves to the BUILTIN. The shadow-dispatch guard in
    /// `compile_call` must therefore consult THIS map, not `variables`:
    /// trusting the flat map rerouted post-scope builtin/module-fn
    /// calls into a dead (undefined-on-path or stale) function pointer
    /// — a runtime crash, review HIGH on 371ee02f. Keyed by symbol with
    /// a liveness count so shadowing across nested scopes balances.
    pub live_fn_bindings: HashMap<Symbol, u32>,
    /// One entry per OPEN binding scope (if/elif/else arms, loop bodies,
    /// match arms, block expressions), listing the symbols registered in
    /// that scope. `pop_fn_binding_scope` decrements their liveness.
    /// Registration with NO open scope is a function-root binding
    /// (params, top-level nested fns) and stays live for the whole
    /// function — which is exactly the checker's scoping for those.
    pub fn_binding_scopes: Vec<Vec<Symbol>>,
    /// Per-FUNCTION closure capture counts, moved off the compiler-global
    /// Registry (review HIGHs on 44c011c1/cb1bd16f: global entries leaked
    /// across nested-fn compiles and, with the block-scope-blind clear,
    /// across if-arm/loop-body scopes — a dead if-arm's non-capturing
    /// rebind deleted the outer closure's entry and the post-arm call
    /// executed the closure struct as code). Mutate ONLY through
    /// `set_closure_info`, which records an undo entry in the innermost
    /// open fn-binding scope; `pop_fn_binding_scope` rolls the scope's
    /// mutations back, mirroring the checker's block scoping exactly like
    /// `live_fn_bindings` above.
    pub closure_info: HashMap<Symbol, usize>,
    /// Undo log per open fn-binding scope: symbol -> the closure_info
    /// state BEFORE this scope's first mutation of it (None = absent).
    closure_scope_undo: Vec<HashMap<Symbol, Option<usize>>>,
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
    /// CSHA A.2.1a: Forward inverse of `weight_values` — weight name → most-recent
    /// Cranelift Value that loaded it. The FA call site uses this to resolve
    /// bridge `FusionMark.param_name` strings (e.g. `"TransformerBlock.wq"`) to
    /// the already-loaded weight pointer without re-emitting the load.
    ///
    /// If the same weight is loaded more than once in a function, the last load
    /// wins — this is safe because weight tensors are immutable and both Values
    /// point to the same runtime tensor.
    pub weights_by_name: HashMap<String, cranelift_codegen::ir::Value>,
    /// M44: Name of the function currently being compiled.
    /// Used by generate() to look up @grammar decorator configs.
    pub current_function_name: Option<String>,
    /// Semantic types for variables, used by step-variable cleanup to identify tensors.
    pub variable_types: HashMap<Symbol, nsl_semantic::types::Type>,
    /// ELTLS Task 16.1: Symbols that were pre-declared by the loop-body
    /// let-tensor predeclare pass. Used by eltls_clear_old_slot to unlock
    /// the slot-free path even when variable_types has no recorded entry
    /// (e.g., the first iteration of the loop before the let statement
    /// has been compiled). The slot is initialized to zero by the
    /// predeclare pass, and nsl_tensor_free_if_valid tolerates the null
    /// pointer on the first iteration.
    pub eltls_loop_predeclared: HashSet<Symbol>,
    /// Dict-local tensor lifetime: `Dict<_, Tensor>` locals that
    /// `dict_lifetime::collect_sweepable_dict_locals` proved safe to free
    /// with `nsl_dict_free_tensor_values` in the return-local sweep
    /// (single top-level call binding, subscript reads only, never
    /// returned/passed/stored-into). Empty when the body was never
    /// scanned — the sweep then adds nothing, which is the pre-existing
    /// leak-not-crash behavior.
    pub sweepable_dict_locals: HashSet<Symbol>,
    /// The loop-rebind subset of the dict sweep plan: dict locals bound in
    /// the DIRECT body of a top-level loop, used only inside it. The loop
    /// lowering zero-predeclares their slot; `compile_var_decl` frees the
    /// previous iteration's dict before each rebind; the return sweep
    /// (they are also in `sweepable_dict_locals`) frees the last one.
    pub dict_loop_rebind: HashSet<Symbol>,
    /// The syms from `dict_loop_rebind` whose slot the loop lowering
    /// ACTUALLY zero-predeclared. The rebind clear keys off THIS set, not
    /// `dict_loop_rebind`: a loop-body decl can shadow a function
    /// PARAMETER (checker-legal — the body is a child scope), which the
    /// scan cannot see; the predeclare skips such syms because the slot
    /// already exists, and a clear keyed on the plan alone would hand the
    /// caller's tensor to `nsl_dict_free_tensor_values` — memory
    /// corruption, review HIGH-1 on d114b5d7, reproduced. Keying on the
    /// inserted set inherits the predeclare's exact skip conditions.
    pub dict_loop_predeclared: HashSet<Symbol>,
    /// M62 Task 4: controls how `self` and `self.<field>` are lowered
    /// inside model methods. Default is `StructPointer` (existing behavior).
    /// Set to `WeightPtrsArray` by the @export method compiler (Task 6).
    pub self_resolution: SelfResolution,
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
            live_fn_bindings: HashMap::new(),
            fn_binding_scopes: Vec::new(),
            closure_info: HashMap::new(),
            closure_scope_undo: Vec::new(),
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
            weights_by_name: HashMap::new(),
            current_function_name: None,
            variable_types: HashMap::new(),
            eltls_loop_predeclared: HashSet::new(),
            sweepable_dict_locals: HashSet::new(),
            dict_loop_rebind: HashSet::new(),
            dict_loop_predeclared: HashSet::new(),
            self_resolution: SelfResolution::StructPointer,
        }
    }

    /// Allocate a new Cranelift Variable index.
    pub fn new_variable(&mut self) -> Variable {
        let var = Variable::from_u32(self.var_counter as u32);
        self.var_counter += 1;
        var
    }

    /// Open a binding scope for `live_fn_bindings`. Call at the start of
    /// every lowering that compiles a checker-scoped statement list
    /// (if/elif/else arms, loop bodies, match arms, block expressions);
    /// pair with `pop_fn_binding_scope` at its end.
    pub fn push_fn_binding_scope(&mut self) {
        self.fn_binding_scopes.push(Vec::new());
        self.closure_scope_undo.push(HashMap::new());
    }

    /// Close the innermost binding scope, retiring the fn-value bindings
    /// it registered. Direction of failure, stated carefully (the first
    /// draft of this comment had it BACKWARDS — review LOW on 682641ca):
    /// a missed POP *widens* liveness, and because the flat `variables`
    /// entry also persists, widened liveness is exactly the STALE-REROUTE
    /// direction — the dangerous one. An EXTRA pop is the safe direction
    /// (under-liveness falls back to builtin precedence, the pre-guard
    /// behavior). Today every skipped-pop path is an `Err` propagation
    /// that aborts the whole function compile, so no live window exists;
    /// any future error-recovery path must keep the pairs balanced.
    pub fn pop_fn_binding_scope(&mut self) {
        if let Some(scope) = self.fn_binding_scopes.pop() {
            for sym in scope {
                if let Some(n) = self.live_fn_bindings.get_mut(&sym) {
                    *n -= 1;
                    if *n == 0 {
                        self.live_fn_bindings.remove(&sym);
                    }
                }
            }
        }
        // Roll back this scope's closure_info mutations to their
        // pre-scope state — same safety direction as above: a missed
        // pop widens the metadata's lifetime (stale-reroute), an extra
        // pop merely restores earlier state.
        if let Some(undo) = self.closure_scope_undo.pop() {
            for (sym, prev) in undo {
                match prev {
                    Some(count) => {
                        self.closure_info.insert(sym, count);
                    }
                    None => {
                        self.closure_info.remove(&sym);
                    }
                }
            }
        }
    }

    /// Record a function-valued binding (nested `fn`, lambda / fn-typed
    /// `let`, fn-typed param) as live in the innermost open scope — or
    /// for the whole function when no scope is open (params, top-level
    /// nested fns), matching the checker's scoping.
    /// Set (Some) or clear (None) a symbol's closure capture count,
    /// recording the prior state in the innermost open fn-binding scope
    /// so `pop_fn_binding_scope` can restore it.
    pub fn set_closure_info(&mut self, sym: Symbol, count: Option<usize>) {
        let prev = match count {
            Some(c) => self.closure_info.insert(sym, c),
            None => self.closure_info.remove(&sym),
        };
        if let Some(undo) = self.closure_scope_undo.last_mut() {
            undo.entry(sym).or_insert(prev);
        }
    }

    pub fn register_fn_binding(&mut self, sym: Symbol) {
        *self.live_fn_bindings.entry(sym).or_insert(0) += 1;
        if let Some(top) = self.fn_binding_scopes.last_mut() {
            top.push(sym);
        }
    }
}
