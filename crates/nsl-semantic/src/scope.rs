use std::collections::HashMap;

use nsl_ast::Symbol;
use nsl_errors::Span;

use crate::types::Type;

/// Identifies a scope in the scope arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

impl ScopeId {
    pub const ROOT: ScopeId = ScopeId(0);
}

/// Information about a single declared name.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub ty: Type,
    pub def_span: Span,
    pub is_const: bool,
    pub is_param: bool,
    pub is_used: bool,
}

/// A single lexical scope.
#[derive(Debug)]
pub struct Scope {
    pub parent: Option<ScopeId>,
    pub symbols: HashMap<Symbol, SymbolInfo>,
    pub kind: ScopeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Module,
    Function,
    Method,
    Model,
    Block,
    Loop,
    Lambda,
}

/// The scope arena: all scopes for a compilation unit.
pub struct ScopeMap {
    scopes: Vec<Scope>,
}

impl ScopeMap {
    pub fn new() -> Self {
        let root = Scope {
            parent: None,
            symbols: HashMap::new(),
            kind: ScopeKind::Module,
        };
        ScopeMap {
            scopes: vec![root],
        }
    }

    /// Create a new child scope, returning its ID.
    pub fn push_scope(&mut self, parent: ScopeId, kind: ScopeKind) -> ScopeId {
        let id = ScopeId(self.scopes.len() as u32);
        self.scopes.push(Scope {
            parent: Some(parent),
            symbols: HashMap::new(),
            kind,
        });
        id
    }

    /// Declare a symbol in the given scope. Returns Err with the existing
    /// SymbolInfo if the name is already declared in THIS scope (not parent).
    pub fn declare(
        &mut self,
        scope: ScopeId,
        name: Symbol,
        info: SymbolInfo,
    ) -> Result<(), SymbolInfo> {
        let s = &mut self.scopes[scope.0 as usize];
        if let Some(existing) = s.symbols.get(&name) {
            Err(existing.clone())
        } else {
            s.symbols.insert(name, info);
            Ok(())
        }
    }

    /// Look up a symbol, walking up the scope chain.
    pub fn lookup(&self, scope: ScopeId, name: Symbol) -> Option<(ScopeId, &SymbolInfo)> {
        let mut current = Some(scope);
        while let Some(sid) = current {
            let s = &self.scopes[sid.0 as usize];
            if let Some(info) = s.symbols.get(&name) {
                return Some((sid, info));
            }
            current = s.parent;
        }
        None
    }

    /// Mutable lookup for marking used, etc.
    pub fn lookup_mut(&mut self, scope: ScopeId, name: Symbol) -> Option<&mut SymbolInfo> {
        let mut current = Some(scope);
        while let Some(sid) = current {
            let parent = self.scopes[sid.0 as usize].parent;
            if self.scopes[sid.0 as usize].symbols.contains_key(&name) {
                return self.scopes[sid.0 as usize].symbols.get_mut(&name);
            }
            current = parent;
        }
        None
    }

    /// Get the ScopeKind.
    pub fn kind(&self, scope: ScopeId) -> ScopeKind {
        self.scopes[scope.0 as usize].kind
    }

    /// Walk up scopes to find the nearest enclosing function/method scope.
    pub fn enclosing_function(&self, scope: ScopeId) -> Option<ScopeId> {
        let mut current = Some(scope);
        while let Some(sid) = current {
            match self.scopes[sid.0 as usize].kind {
                ScopeKind::Function | ScopeKind::Method => return Some(sid),
                _ => current = self.scopes[sid.0 as usize].parent,
            }
        }
        None
    }

    /// Check if we are inside a loop scope.
    pub fn is_in_loop(&self, scope: ScopeId) -> bool {
        let mut current = Some(scope);
        while let Some(sid) = current {
            match self.scopes[sid.0 as usize].kind {
                ScopeKind::Loop => return true,
                ScopeKind::Function | ScopeKind::Method | ScopeKind::Lambda => return false,
                _ => current = self.scopes[sid.0 as usize].parent,
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_info(ty: Type) -> SymbolInfo {
        SymbolInfo {
            ty,
            def_span: Span::DUMMY,
            is_const: false,
            is_param: false,
            is_used: false,
        }
    }

    fn make_sym(n: u32) -> Symbol {
        // Create a synthetic symbol for testing
        use string_interner::Symbol as SI;
        Symbol(SI::try_from_usize(n as usize).unwrap())
    }

    #[test]
    fn declare_and_lookup() {
        let mut scopes = ScopeMap::new();
        let sym = make_sym(0);
        scopes
            .declare(ScopeId::ROOT, sym, dummy_info(Type::Int))
            .unwrap();
        let (sid, info) = scopes.lookup(ScopeId::ROOT, sym).unwrap();
        assert_eq!(sid, ScopeId::ROOT);
        assert_eq!(info.ty, Type::Int);
    }

    #[test]
    fn lookup_walks_parent() {
        let mut scopes = ScopeMap::new();
        let sym = make_sym(0);
        scopes
            .declare(ScopeId::ROOT, sym, dummy_info(Type::Float))
            .unwrap();
        let child = scopes.push_scope(ScopeId::ROOT, ScopeKind::Block);
        let (sid, info) = scopes.lookup(child, sym).unwrap();
        assert_eq!(sid, ScopeId::ROOT);
        assert_eq!(info.ty, Type::Float);
    }

    #[test]
    fn shadowing_in_child() {
        let mut scopes = ScopeMap::new();
        let sym = make_sym(0);
        scopes
            .declare(ScopeId::ROOT, sym, dummy_info(Type::Int))
            .unwrap();
        let child = scopes.push_scope(ScopeId::ROOT, ScopeKind::Block);
        scopes
            .declare(child, sym, dummy_info(Type::Str))
            .unwrap();
        let (sid, info) = scopes.lookup(child, sym).unwrap();
        assert_eq!(sid, child);
        assert_eq!(info.ty, Type::Str);
    }

    #[test]
    fn duplicate_in_same_scope() {
        let mut scopes = ScopeMap::new();
        let sym = make_sym(0);
        scopes
            .declare(ScopeId::ROOT, sym, dummy_info(Type::Int))
            .unwrap();
        let result = scopes.declare(ScopeId::ROOT, sym, dummy_info(Type::Float));
        assert!(result.is_err());
    }

    #[test]
    fn lookup_not_found() {
        let scopes = ScopeMap::new();
        let sym = make_sym(0);
        assert!(scopes.lookup(ScopeId::ROOT, sym).is_none());
    }

    #[test]
    fn is_in_loop_detection() {
        let mut scopes = ScopeMap::new();
        let func = scopes.push_scope(ScopeId::ROOT, ScopeKind::Function);
        let loop_scope = scopes.push_scope(func, ScopeKind::Loop);
        let inner = scopes.push_scope(loop_scope, ScopeKind::Block);
        assert!(scopes.is_in_loop(inner));
        assert!(scopes.is_in_loop(loop_scope));
        assert!(!scopes.is_in_loop(func));
    }
}
