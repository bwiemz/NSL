mod block;
mod decl;
mod expr;
mod model;
mod ops;
mod stmt;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use nsl_ast::block::*;
use nsl_ast::decl::*;
use nsl_ast::expr::*;
use nsl_ast::operator::*;
use nsl_ast::pattern::{Pattern, PatternKind};
use nsl_ast::stmt::*;
use nsl_ast::types::TypeExpr;
use nsl_ast::{Module, NodeId, Symbol};
use nsl_errors::{Diagnostic, Span};
use nsl_lexer::Interner;

/// Custom dtype IDs start at 256 to avoid collision with built-in dtype codes.
const DTYPE_CUSTOM_START: u16 = 256;

/// Semantic metadata for a user-defined `datatype` block.
#[allow(dead_code)]
struct CustomDtypeSemanticInfo {
    dtype_id: u16,
    bit_width: u8,
    block_size: Option<u32>,
    has_pack: bool,
    has_unpack: bool,
    has_pack_ptx: bool,
    has_unpack_ptx: bool,
}

use crate::resolve::TypeResolver;
use crate::scope::*;
use crate::shapes;
use crate::types::*;

/// Maps each expression NodeId to its resolved Type.
pub type TypeMap = HashMap<NodeId, Type>;

pub struct TypeChecker<'a> {
    pub interner: &'a mut Interner,
    pub scopes: &'a mut ScopeMap,
    pub diagnostics: Vec<Diagnostic>,
    pub type_map: TypeMap,
    pub effect_checker: crate::effects::EffectChecker,
    current_scope: ScopeId,
    current_return_type: Option<Type>,
    /// M51: Tracks callee function names during function body checking
    /// for effect call graph construction.
    current_callees: Vec<String>,
    /// Pre-resolved types for imported symbols (from other modules).
    import_types: HashMap<Symbol, Type>,
    /// Registry of user-defined `datatype` blocks validated in this module.
    custom_datatypes: HashMap<String, CustomDtypeSemanticInfo>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(interner: &'a mut Interner, scopes: &'a mut ScopeMap) -> Self {
        Self {
            interner,
            scopes,
            diagnostics: Vec::new(),
            type_map: HashMap::new(),
            effect_checker: crate::effects::EffectChecker::new(),
            current_scope: ScopeId::ROOT,
            current_return_type: None,
            current_callees: Vec::new(),
            import_types: HashMap::new(),
            custom_datatypes: HashMap::new(),
        }
    }

    /// Set pre-resolved import types from other modules.
    pub fn set_import_types(&mut self, import_types: &HashMap<Symbol, Type>) {
        self.import_types = import_types.clone();
    }

    pub fn check_module(&mut self, module: &Module) {
        // Two-pass: first collect top-level declarations, then check bodies
        self.collect_top_level_decls(&module.stmts);
        for stmt in &module.stmts {
            self.check_stmt(stmt);
        }

        // M51: Propagate effects through call graph and validate assertions.
        self.effect_checker.propagate();
        self.effect_checker.validate();
        self.diagnostics.append(&mut self.effect_checker.diagnostics);
    }

    /// Pre-declare top-level names so forward references work.
    /// Two sub-passes: first imports (so types are available), then declarations.
    fn collect_top_level_decls(&mut self, stmts: &[Stmt]) {
        // Sub-pass 1: Process imports first so imported types are available
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::Import(import) => self.check_import(import),
                StmtKind::FromImport(import) => self.check_from_import(import),
                _ => {}
            }
        }
        // Sub-pass 2: Pre-declare type/function names
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::FnDef(fn_def) => {
                    let ty = self.build_fn_type(fn_def);
                    self.declare_symbol(fn_def.name, ty, stmt.span, true, false);
                }
                StmtKind::ModelDef(model_def) => {
                    // Declare model name as Unknown initially, refined during check
                    self.declare_symbol(model_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::StructDef(struct_def) => {
                    self.declare_symbol(struct_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::EnumDef(enum_def) => {
                    self.declare_symbol(enum_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::TraitDef(trait_def) => {
                    self.declare_symbol(trait_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::DatatypeDef(def) => {
                    self.declare_symbol(def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::ServeBlock(_) => {
                    // No top-level pre-declaration needed for serve blocks
                }
                StmtKind::Decorated { stmt, .. } => {
                    // Recurse into the inner stmt for pre-declaration
                    self.collect_top_level_decls(std::slice::from_ref(stmt));
                }
                _ => {}
            }
        }
    }

    // ===== Helpers =====

    fn check_block(&mut self, block: &Block, kind: ScopeKind) {
        let scope = self.scopes.push_scope(self.current_scope, kind);
        let prev = self.current_scope;
        self.current_scope = scope;
        for s in &block.stmts {
            self.check_stmt(s);
        }
        self.current_scope = prev;
    }

    fn build_fn_type(&mut self, fn_def: &FnDef) -> Type {
        let params: Vec<Type> = fn_def
            .params
            .iter()
            .filter_map(|p| {
                let name = self.resolve_name(p.name);
                if name == "self" {
                    return None;
                }
                Some(
                    p.type_ann
                        .as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or(Type::Unknown),
                )
            })
            .collect();
        let ret = fn_def
            .return_type
            .as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(Type::Void);
        let effect = if let Some(eff_expr) = &fn_def.return_effect {
            let resolver = TypeResolver {
                interner: self.interner,
                scopes: self.scopes,
                diagnostics: &mut self.diagnostics,
            };
            resolver.resolve_effect_expr(eff_expr, self.current_scope)
        } else {
            Effect::Inferred
        };
        Type::Function {
            params,
            ret: Box::new(ret),
            effect,
        }
    }

    fn resolve_type(&mut self, type_expr: &TypeExpr) -> Type {
        let mut resolver = TypeResolver {
            interner: self.interner,
            scopes: self.scopes,
            diagnostics: &mut self.diagnostics,
        };
        resolver.resolve(type_expr, self.current_scope)
    }

    fn declare_symbol(
        &mut self,
        name: Symbol,
        ty: Type,
        span: Span,
        is_const: bool,
        is_param: bool,
    ) {
        let info = SymbolInfo {
            ty,
            def_span: span,
            is_const,
            is_param,
            is_used: false,
        };
        if self.scopes.declare(self.current_scope, name, info.clone()).is_err() {
            // NSL allows rebinding (Python-like semantics), so update the existing binding
            if let Some(existing) = self.scopes.lookup_mut(self.current_scope, name) {
                existing.ty = info.ty;
                existing.is_const = info.is_const;
                existing.def_span = info.def_span;
            }
        }
    }

    fn declare_pattern(&mut self, pattern: &Pattern, ty: &Type) {
        self.declare_pattern_with_const(pattern, ty, false);
    }

    fn declare_pattern_with_const(&mut self, pattern: &Pattern, ty: &Type, is_const: bool) {
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                self.declare_symbol(*sym, ty.clone(), pattern.span, is_const, false);
            }
            PatternKind::Tuple(pats) => {
                if let Type::Tuple(types) = ty {
                    for (p, t) in pats.iter().zip(types.iter()) {
                        self.declare_pattern_with_const(p, t, is_const);
                    }
                } else {
                    for p in pats {
                        self.declare_pattern_with_const(p, &Type::Unknown, is_const);
                    }
                }
            }
            PatternKind::Typed { pattern, type_ann } => {
                let ann_ty = self.resolve_type(type_ann);
                self.declare_pattern_with_const(pattern, &ann_ty, is_const);
            }
            PatternKind::Wildcard => {} // Don't bind anything
            PatternKind::Rest(Some(sym)) => {
                self.declare_symbol(*sym, Type::Unknown, pattern.span, is_const, false);
            }
            _ => {} // Literal, Constructor, etc. — no bindings in M2
        }
    }

    fn resolve_name(&self, sym: Symbol) -> String {
        self.interner
            .resolve(sym.0)
            .unwrap_or("<unknown>")
            .to_string()
    }
}

/// Simple glob matching supporting `*` as a wildcard that matches any number
/// of characters (including zero). Uses a two-pointer / DP-lite approach.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (plen, tlen) = (pat.len(), txt.len());

    let mut pi = 0;
    let mut ti = 0;
    let mut star_pi: Option<usize> = None;
    let mut star_ti = 0;

    while ti < tlen {
        if pi < plen && (pat[pi] == txt[ti] || pat[pi] == '?') {
            pi += 1;
            ti += 1;
        } else if pi < plen && pat[pi] == '*' {
            star_pi = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }

    while pi < plen && pat[pi] == '*' {
        pi += 1;
    }

    pi == plen
}
