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
        }
    }

    /// Allocate a new Cranelift Variable index.
    pub fn new_variable(&mut self) -> Variable {
        let var = Variable::from_u32(self.var_counter as u32);
        self.var_counter += 1;
        var
    }
}
