pub mod builtins;
pub mod checker;
pub mod context_parallel;
pub mod determinism;
pub mod effects;
pub mod fp8;
pub mod grammar;
pub mod kv_compress;
pub mod moe;
pub mod multimodal;
pub mod nan_analysis;
pub mod ownership;
pub mod pipeline;
pub mod ownership_autodiff;
pub mod ownership_walker;
pub mod perf_budget;
pub mod resolve;
pub mod scope;
pub mod shape_algebra;
pub mod shapes;
pub mod sparse;
pub mod sparse_layout;
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

/// Per-function ownership metadata extracted during M38a analysis.
#[derive(Debug, Clone, Default)]
pub struct FunctionOwnershipInfo {
    /// Parameters that are consumed (moved) by this function.
    pub linear_params: Vec<Symbol>,
    /// Parameters that are @shared (refcounted, multiple uses OK).
    pub shared_params: Vec<Symbol>,
}

/// Result of semantic analysis.
pub struct AnalysisResult {
    pub diagnostics: Vec<Diagnostic>,
    pub type_map: TypeMap,
    pub scopes: ScopeMap,
    /// M38a: Per-function ownership metadata (function name → info).
    /// Only populated when linear_types analysis is enabled.
    pub ownership_info: HashMap<String, FunctionOwnershipInfo>,
}

/// Run semantic analysis on a parsed module (single-file, backward compatible).
pub fn analyze(module: &Module, interner: &mut Interner) -> AnalysisResult {
    analyze_with_imports(module, interner, &HashMap::new(), false)
}

/// Run semantic analysis with pre-resolved import types from other modules.
pub fn analyze_with_imports(
    module: &Module,
    interner: &mut Interner,
    import_types: &ImportTypes,
    linear_types: bool,
) -> AnalysisResult {
    let mut scopes = ScopeMap::new();

    // Phase 1: Register built-in types and functions
    builtins::register_builtins(&mut scopes, interner);

    // Phase 2: Run the type checker with import context
    let mut checker = TypeChecker::new(interner, &mut scopes);
    checker.set_import_types(import_types);
    checker.check_module(module);

    // M51: Effect analysis is now integrated into TypeChecker — propagation
    // and validation run at the end of check_module(). Diagnostics are merged.

    // Extract results from type checker, releasing its borrows on interner/scopes
    let mut diagnostics = checker.diagnostics;
    let type_map = checker.type_map;

    // M38a: Run ownership analysis when --linear-types is active.
    // Runs after type checking so we have the TypeMap for tensor type detection.
    let mut ownership_info = HashMap::new();
    if linear_types {
        let (ownership_diags, info) = ownership_walker::analyze_ownership(
            module, interner, &type_map, &scopes,
        );
        diagnostics.extend(ownership_diags);
        ownership_info = info;
    }

    AnalysisResult {
        diagnostics,
        type_map,
        scopes,
        ownership_info,
    }
}
