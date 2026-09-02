//! In-process capture of the CLIF every function of a compile is defined
//! from — what `--dump-ir` prints, kept as data instead of written to
//! stderr.
//!
//! One `Compiler::record_ir` call sits where each function's IR is final
//! (just before `define_function`); it prints when `dump_ir` is set and
//! pushes an [`IrDump`] when a caller installed a slot. The snapshot tests
//! over train-block lowering (`tests/train_clif_snapshots.rs`) are the
//! consumer: a refactor of the train-block compiler is behaviour-preserving
//! exactly when every dump it produces is byte-identical.

use std::cell::RefCell;
use std::rc::Rc;

use cranelift_codegen::ir::Function;
use cranelift_module::{DataId, FuncId, Module, ModuleDeclarations};

/// The CLIF of one defined function, as `--dump-ir` would print it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrDump {
    /// What `--dump-ir` writes between `--- IR: ` and ` ---`: the kind of
    /// function and its name, e.g. `main`, `fn 'f'`, `model method 'M__forward'`.
    pub label: String,
    /// `Function::display()` of the finished function.
    pub clif: String,
    /// The linkage name behind every external-name token the text refers
    /// to — each function (namespace 0) or data object (namespace 1) it
    /// names — so a reader can render `fn3 = u0:17 sig3` as
    /// `fn3 = nsl_tensor_matmul sig3`, and a snapshot does not move when an
    /// unrelated declaration shifts the indices. CLIF spells a reference
    /// two ways: `u<namespace>:<index>` where the writer has the function's
    /// parameters (`fn` declarations) and `userextname<ref>` where it does
    /// not (`gv … = symbol …`); both spellings of each name are listed.
    ///
    /// The function's own name in the `function u0:N(…)` header is NOT in
    /// this table: the compiler names each function it defines by a
    /// per-compile counter (`Compiler::next_func_index`), not by its
    /// `FuncId`, so that `N` resolves to nothing — `label` is what names
    /// the function.
    pub symbols: Vec<(String, String)>,
}

impl IrDump {
    /// Capture `func` as `label`, resolving its user-named references
    /// against the module that declared them.
    pub fn capture(label: String, func: &Function, module: &impl Module) -> Self {
        let decls = module.declarations();
        let mut symbols = Vec::new();
        for (name_ref, name) in func.params.user_named_funcs().iter() {
            let linkage = linkage_name(decls, name.namespace, name.index);
            symbols.push((name.to_string(), linkage.clone()));
            symbols.push((format!("{name_ref}"), linkage));
        }
        symbols.sort();
        symbols.dedup();
        Self {
            label,
            clif: func.display().to_string(),
            symbols,
        }
    }
}

fn linkage_name(decls: &ModuleDeclarations, namespace: u32, index: u32) -> String {
    match namespace {
        0 => {
            let id = FuncId::from_u32(index);
            decls.get_function_decl(id).linkage_name(id).into_owned()
        }
        1 => {
            let id = DataId::from_u32(index);
            decls.get_data_decl(id).linkage_name(id).into_owned()
        }
        // cranelift-module uses only the two; anything else stays as printed.
        _ => format!("u{namespace}:{index}"),
    }
}

/// Shared out-slot: the entry point keeps one `Rc` clone and reads it once
/// the compile finishes or errors; `Compiler::record_ir` appends to it in
/// definition order.
pub type IrCaptureSlot = Rc<RefCell<Vec<IrDump>>>;
