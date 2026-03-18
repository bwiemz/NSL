pub mod builtins;
pub mod checker;
pub mod context_parallel;
pub mod determinism;
pub mod fp8;
pub mod grammar;
pub mod kv_compress;
pub mod moe;
pub mod multimodal;
pub mod nan_analysis;
pub mod ownership;
pub mod pipeline;
pub mod ownership_autodiff;
pub mod perf_budget;
pub mod resolve;
pub mod scope;
pub mod shapes;
pub mod speculative;
pub mod target;
pub mod types;
pub mod vmap;

use std::collections::HashMap;

use nsl_ast::{Module, Symbol};
use nsl_errors::Diagnostic;
use nsl_lexer::Interner;

use crate::checker::{TypeChecker, TypeMap};
use crate::scope::ScopeMap;
use crate::types::Type;

/// Maps imported symbol names to their resolved types from other modules.
pub type ImportTypes = HashMap<Symbol, Type>;

/// Result of semantic analysis.
pub struct AnalysisResult {
    pub diagnostics: Vec<Diagnostic>,
    pub type_map: TypeMap,
    pub scopes: ScopeMap,
}

/// Run semantic analysis on a parsed module (single-file, backward compatible).
pub fn analyze(module: &Module, interner: &mut Interner) -> AnalysisResult {
    analyze_with_imports(module, interner, &HashMap::new())
}

/// Run semantic analysis with pre-resolved import types from other modules.
pub fn analyze_with_imports(
    module: &Module,
    interner: &mut Interner,
    import_types: &ImportTypes,
) -> AnalysisResult {
    let mut scopes = ScopeMap::new();

    // Phase 1: Register built-in types and functions
    builtins::register_builtins(&mut scopes, interner);

    // Phase 2: Run the type checker with import context
    let mut checker = TypeChecker::new(interner, &mut scopes);
    checker.set_import_types(import_types);
    checker.check_module(module);

    AnalysisResult {
        diagnostics: checker.diagnostics,
        type_map: checker.type_map,
        scopes,
    }
}
