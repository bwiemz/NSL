//! M56: Agent semantic analysis. Flag gate + agent registry + APG
//! extraction + cross-agent access, device, and fan-out rules.
//!
//! Spec: docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md

use std::collections::HashMap;
use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::{Module, Symbol};
use nsl_errors::{Diagnostic, Span};
use nsl_lexer::Interner;

// ---------------------------------------------------------------------------
// AgentRegistry — records every `agent` declaration for later APG extraction
// and cross-agent rule enforcement.
// ---------------------------------------------------------------------------

/// M56 agent registry — records every `agent` declaration in the module
/// for later APG extraction and cross-agent rule enforcement.
#[derive(Debug, Default)]
pub struct AgentRegistry {
    agents: HashMap<Symbol, RegisteredAgent>,
    /// Parallel lookup by resolved-name-string so test code can query
    /// without a live interner handle.
    by_name: HashMap<String, Symbol>,
}

#[derive(Debug)]
pub struct RegisteredAgent {
    pub def_symbol: Symbol,
    pub fields: Vec<FieldInfo>,
    pub methods: Vec<MethodInfo>,
    pub def_span: nsl_errors::Span,
}

#[derive(Debug)]
pub struct FieldInfo {
    pub name: Symbol,
    pub name_str: String,
    pub is_shared: bool,
    pub span: nsl_errors::Span,
}

#[derive(Debug)]
pub struct MethodInfo {
    pub name: Symbol,
    pub name_str: String,
    pub has_auto_device_transfer: bool,
    pub param_names: Vec<Symbol>,
    pub span: nsl_errors::Span,
}

impl RegisteredAgent {
    pub fn field_count(&self) -> usize { self.fields.len() }
    pub fn method_count(&self) -> usize { self.methods.len() }
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.iter().any(|m| m.name_str == name)
    }
}

impl AgentRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register_module(&mut self, module: &Module, interner: &Interner) {
        for stmt in &module.stmts {
            if let nsl_ast::stmt::StmtKind::AgentDef(def) = &stmt.kind {
                self.register_agent(def, interner);
            }
        }
    }

    fn register_agent(&mut self, def: &AgentDef, interner: &Interner) {
        let name_str = interner.resolve(def.name.0).unwrap_or("<unknown>").to_string();

        let fields: Vec<FieldInfo> = def.members.iter().filter_map(|m| match m {
            AgentMember::FieldDecl { name, decorators, span, .. } => Some(FieldInfo {
                name: *name,
                name_str: interner.resolve(name.0).unwrap_or("?").to_string(),
                is_shared: decorators.iter().any(|d| {
                    d.name.len() == 1
                        && interner.resolve(d.name[0].0).map_or(false, |s| s == "shared")
                }),
                span: *span,
            }),
            _ => None,
        }).collect();

        let methods: Vec<MethodInfo> = def.members.iter().filter_map(|m| match m {
            AgentMember::Method(fn_def, decorators) => Some(MethodInfo {
                name: fn_def.name,
                name_str: interner.resolve(fn_def.name.0).unwrap_or("?").to_string(),
                has_auto_device_transfer: decorators.iter().any(|d| {
                    d.name.len() == 1
                        && interner.resolve(d.name[0].0).map_or(false, |s| s == "auto_device_transfer")
                }),
                param_names: fn_def.params.iter().map(|p| p.name).collect(),
                span: fn_def.span,
            }),
            _ => None,
        }).collect();

        let registered = RegisteredAgent {
            def_symbol: def.name,
            fields,
            methods,
            def_span: def.span,
        };
        self.by_name.insert(name_str.clone(), def.name);
        self.agents.insert(def.name, registered);
    }

    pub fn get_by_name(&self, name: &str) -> Option<&RegisteredAgent> {
        self.by_name.get(name).and_then(|sym| self.agents.get(sym))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &RegisteredAgent)> {
        self.agents.iter()
    }
}

// ---------------------------------------------------------------------------
// Flag-gate check
// ---------------------------------------------------------------------------


/// Emit E0610 if agent declarations are present and `--linear-types` is off.
/// Spec §7 (flag gating) + §6.7 (error format).
///
/// `agent_decl_spans`: spans of all `agent Foo:` declarations found in the
/// module; typically produced during initial AST walk in
/// `crates/nsl-semantic/src/checker/mod.rs::collect_top_level_decls`.
pub fn check_linear_types_flag(
    agent_decl_spans: &[Span],
    linear_types_enabled: bool,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if linear_types_enabled {
        return;
    }
    let Some(&first_span) = agent_decl_spans.first() else {
        return;
    };
    diagnostics.push(
        Diagnostic::error(
            "E0610: M56 agent declarations require --linear-types\n\
             \n\
             requested: compile a program containing an agent declaration\n\
             expected:  the linear ownership checker (--linear-types) active\n\
             found:     --linear-types not passed to the compiler\n\
             \n\
             fix: add --linear-types to `nsl check` or `nsl build`.\n\
             \n\
             `nsl run` does not currently expose --linear-types;\n\
             for run, use `nsl build` and execute the produced binary.\n\
             (Tracked: Task 20 of this plan closes that gap.)",
        )
        .with_label(first_span, "agent declared here"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::{BytePos, FileId, Span};

    fn make_span() -> Span {
        Span::new(FileId(0), BytePos(10), BytePos(15))
    }

    #[test]
    fn flag_gate_errors_when_agent_present_without_linear_types() {
        let mut diags = Vec::new();
        let dummy = make_span();
        check_linear_types_flag(
            /* agent_decl_spans = */ &[dummy],
            /* linear_types_enabled = */ false,
            &mut diags,
        );
        assert_eq!(diags.len(), 1);
        assert!(
            diags[0].message.contains("E0610"),
            "expected E0610, got {}",
            diags[0].message
        );
        // Span is carried via labels[0].span (Diagnostic has no primary_span field)
        assert_eq!(diags[0].labels[0].span, dummy);
    }

    #[test]
    fn flag_gate_silent_when_linear_types_enabled() {
        let mut diags = Vec::new();
        let dummy = make_span();
        check_linear_types_flag(&[dummy], true, &mut diags);
        assert!(diags.is_empty());
    }

    #[test]
    fn flag_gate_silent_when_no_agents() {
        let mut diags = Vec::new();
        check_linear_types_flag(&[], false, &mut diags);
        assert!(diags.is_empty());
    }

    #[test]
    fn registry_records_agents_and_member_kinds() {
        let mut interner = nsl_lexer::Interner::new();
        let src = "agent Drafter:\n    kv_cache: KvCache = empty()\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n";
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex diagnostics: {:?}", lex_diags);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(parse_result.diagnostics.is_empty(), "parse diagnostics: {:?}", parse_result.diagnostics);

        let mut registry = AgentRegistry::new();
        registry.register_module(&parse_result.module, &interner);
        let drafter = registry.get_by_name("Drafter").expect("Drafter not registered");
        assert_eq!(drafter.field_count(), 1);
        assert_eq!(drafter.method_count(), 1);
        assert!(drafter.has_method("draft"));
    }
}
