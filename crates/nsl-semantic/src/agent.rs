//! M56: Agent semantic analysis. Flag gate + agent registry + APG
//! extraction + cross-agent access, device, and fan-out rules.
//!
//! Spec: docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md

use std::collections::{HashMap, HashSet};
use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::decl::{Decorator, FnDef};
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Block, StmtKind};
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
                        && interner.resolve(d.name[0].0) == Some("shared")
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
                        && interner.resolve(d.name[0].0) == Some("auto_device_transfer")
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

    pub fn get_by_name_symbol(&self, sym: Symbol) -> Option<&RegisteredAgent> {
        self.agents.get(&sym)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &RegisteredAgent)> {
        self.agents.iter()
    }
}

// ---------------------------------------------------------------------------
// APG — Action-Port Graph (M56 spec §2.2)
// ---------------------------------------------------------------------------

/// M56 Action-Port Graph extracted from a `@pipeline_agent` function body.
/// Spec §2.2.
#[derive(Debug)]
pub struct ActionPortGraph {
    /// Symbol of the pipeline function (`fn pipeline` in user code).
    pub pipeline_fn: Symbol,
    /// Participating agents (from the decorator's `agents=[...]` argument).
    pub agents: Vec<Symbol>,
    /// Edges — see `ApgEdge` variants.
    pub edges: Vec<ApgEdge>,
    /// Span of the `@pipeline_agent` decorator for diagnostics.
    pub decorator_span: Span,
}

#[derive(Debug, Clone)]
pub enum ApgEdge {
    /// A pipeline-function parameter flows into an agent method call.
    PipelineInputToAgent {
        pipeline_param: Symbol,
        target_agent: Symbol,
        target_method: Symbol,
        target_param: Symbol,
        span: Span,
    },
    /// A `let`-bound value flows into an agent method call.
    BindingToAgent {
        binding: Symbol,
        source_agent: Option<Symbol>,
        source_method: Option<Symbol>,
        target_agent: Symbol,
        target_method: Symbol,
        target_param: Symbol,
        span: Span,
    },
}

/// Walk the module's top-level statements and extract one `ActionPortGraph`
/// per `@pipeline_agent`-decorated `fn`.
pub fn extract_apgs(
    module: &Module,
    registry: &AgentRegistry,
    interner: &Interner,
    out_apgs: &mut Vec<ActionPortGraph>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for stmt in &module.stmts {
        // Decorators on top-level fns are wrapped in StmtKind::Decorated.
        let (fn_def, decorators): (&FnDef, &Vec<Decorator>) = match &stmt.kind {
            StmtKind::Decorated { decorators, stmt: inner_stmt } => {
                match &inner_stmt.kind {
                    StmtKind::FnDef(fn_def) => (fn_def, decorators),
                    _ => continue,
                }
            }
            _ => continue,
        };

        let (decorator, agents) = match find_pipeline_agent_decorator(decorators, interner) {
            Some(x) => x,
            None => continue,
        };

        for &agent_sym in &agents {
            if registry.get_by_name_symbol(agent_sym).is_none() {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "@pipeline_agent references unknown agent `{}`",
                        interner.resolve(agent_sym.0).unwrap_or("?")
                    ))
                    .with_label(decorator.span, "declared in this @pipeline_agent"),
                );
            }
        }

        let pipeline_params: Vec<Symbol> = fn_def.params.iter().map(|p| p.name).collect();
        let mut source_by_binding: HashMap<Symbol, (Symbol, Symbol)> = HashMap::new();
        let mut edges = Vec::new();
        walk_block_for_edges(
            &fn_def.body,
            registry,
            interner,
            &pipeline_params,
            &mut source_by_binding,
            &mut edges,
            diagnostics,
        );

        out_apgs.push(ActionPortGraph {
            pipeline_fn: fn_def.name,
            agents,
            edges,
            decorator_span: decorator.span,
        });
    }
}

fn find_pipeline_agent_decorator<'a>(
    decorators: &'a [Decorator],
    interner: &Interner,
) -> Option<(&'a Decorator, Vec<Symbol>)> {
    for d in decorators {
        if d.name.len() != 1 {
            continue;
        }
        if interner.resolve(d.name[0].0) != Some("pipeline_agent") {
            continue;
        }
        let agents = extract_agent_list(d, interner);
        return Some((d, agents));
    }
    None
}

fn extract_agent_list(decorator: &Decorator, interner: &Interner) -> Vec<Symbol> {
    let Some(args) = &decorator.args else {
        return Vec::new();
    };
    for arg in args {
        let is_agents = arg
            .name
            .and_then(|s| interner.resolve(s.0))
            == Some("agents");
        if !is_agents {
            continue;
        }
        if let ExprKind::ListLiteral(items) = &arg.value.kind {
            return items
                .iter()
                .filter_map(|item| match &item.kind {
                    ExprKind::Ident(s) => Some(*s),
                    _ => None,
                })
                .collect();
        }
    }
    Vec::new()
}

// `diags` is threaded through for Tasks 8-11, which will emit diagnostics
// inside the walker (cross-agent access, mutation, device, fan-out rules).
// Task 6 does not yet write to it, hence the suppression.
#[allow(clippy::only_used_in_recursion)]
fn walk_block_for_edges(
    block: &Block,
    registry: &AgentRegistry,
    interner: &Interner,
    pipeline_params: &[Symbol],
    source_by_binding: &mut HashMap<Symbol, (Symbol, Symbol)>,
    edges: &mut Vec<ApgEdge>,
    diags: &mut Vec<Diagnostic>,
) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value: Some(val), .. } => {
                if let Some(call) = as_agent_method_call(val, registry, interner) {
                    if let PatternKind::Ident(binding_sym) = &pattern.kind {
                        source_by_binding.insert(*binding_sym, (call.agent, call.method));
                    }
                    push_call_edges(&call, pipeline_params, source_by_binding, edges, val.span);
                }
            }
            StmtKind::Return(Some(expr)) => {
                if let Some(call) = as_agent_method_call(expr, registry, interner) {
                    push_call_edges(&call, pipeline_params, source_by_binding, edges, expr.span);
                }
            }
            // NOTE: source_by_binding is shared across branches (then/elif/else).
            // In v1, conditional-branch binding scope is not modeled; for branching
            // pipelines, binding sources from any branch are visible to statements
            // that follow the if. Acceptable because v1 pipelines are expected to be
            // linear (test: extracts_linear_pipeline_apg_from_method_calls).
            // TODO(future): per-branch SSA or join-node modeling for sound data-flow
            // in conditional pipelines.
            StmtKind::If { then_block, elif_clauses, else_block, .. } => {
                walk_block_for_edges(
                    then_block, registry, interner, pipeline_params, source_by_binding, edges, diags,
                );
                for (_, blk) in elif_clauses {
                    walk_block_for_edges(
                        blk, registry, interner, pipeline_params, source_by_binding, edges, diags,
                    );
                }
                if let Some(b) = else_block {
                    walk_block_for_edges(
                        b, registry, interner, pipeline_params, source_by_binding, edges, diags,
                    );
                }
            }
            // TODO(v2): For/While/WhileLet loop bodies are not walked; agent method
            // calls inside loops produce zero APG edges in v1. v1 pipelines are
            // expected to be linear (no loops over agent calls). If a user writes
            // agent calls inside a loop, the scheduler (Task 14) will not see them —
            // a future task should either walk these bodies or emit a diagnostic.
            _ => {}
        }
    }
}

struct AgentMethodCall {
    agent: Symbol,
    method: Symbol,
    args: Vec<Expr>,
    /// Parameter names of the called method (including `self` at index 0).
    target_param_names: Vec<Symbol>,
}

/// Try to interpret `expr` as `receiver_ident.method(args...)` where
/// `receiver_ident` (title-cased) names a registered agent.
fn as_agent_method_call(
    expr: &Expr,
    registry: &AgentRegistry,
    interner: &Interner,
) -> Option<AgentMethodCall> {
    // Method calls are represented as:
    //   ExprKind::Call { callee: MemberAccess { object: Ident(recv), member }, args }
    let ExprKind::Call { callee, args } = &expr.kind else {
        return None;
    };
    let ExprKind::MemberAccess { object, member } = &callee.kind else {
        return None;
    };
    let ExprKind::Ident(receiver_sym) = &object.kind else {
        return None;
    };
    let name = interner.resolve(receiver_sym.0)?;
    // v1 heuristic: convert the receiver variable name (lowercase) to title-case
    // and look up the registered agent by that name. This mirrors the NSL
    // convention that `agent Drafter:` is bound at the pipeline site as
    // `drafter`. Tasks 8–10 use the same heuristic via the same lookup shape.
    // TODO(task 12): replace heuristic with type-checker-resolved binding→agent
    // type map when semantic-phase integration lands.
    let title = {
        let mut chars = name.chars();
        match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        }
    };
    let agent = registry.get_by_name(&title)?;
    let method_str = interner.resolve(member.0)?;
    let method_info = agent.methods.iter().find(|m| m.name_str == method_str)?;
    Some(AgentMethodCall {
        agent: agent.def_symbol,
        method: *member,
        args: args.iter().map(|a| a.value.clone()).collect(),
        target_param_names: method_info.param_names.clone(),
    })
}

fn push_call_edges(
    call: &AgentMethodCall,
    pipeline_params: &[Symbol],
    source_by_binding: &HashMap<Symbol, (Symbol, Symbol)>,
    edges: &mut Vec<ApgEdge>,
    span: Span,
) {
    // Skip index 0 (implicit `self`); zip remaining params 1:1 with positional args.
    let params = call.target_param_names.iter().skip(1);
    for (arg_expr, target_param) in call.args.iter().zip(params) {
        let ExprKind::Ident(arg_sym) = &arg_expr.kind else {
            continue;
        };
        if pipeline_params.iter().any(|p| p == arg_sym) {
            edges.push(ApgEdge::PipelineInputToAgent {
                pipeline_param: *arg_sym,
                target_agent: call.agent,
                target_method: call.method,
                target_param: *target_param,
                span,
            });
        } else {
            let (source_agent, source_method) = source_by_binding
                .get(arg_sym)
                .copied()
                .map_or((None, None), |(a, m)| (Some(a), Some(m)));
            edges.push(ApgEdge::BindingToAgent {
                binding: *arg_sym,
                source_agent,
                source_method,
                target_agent: call.agent,
                target_method: call.method,
                target_param: *target_param,
                span,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Cycle detection — E0603
// ---------------------------------------------------------------------------

/// Spec §6.3. DFS-based cycle detection over agent-to-agent edges.
///
/// An agent-to-agent edge exists iff an `ApgEdge::BindingToAgent` has a known
/// `source_agent` — i.e., data produced by one agent is consumed by another.
/// `PipelineInputToAgent` edges never contribute because they have no source
/// agent.
pub fn detect_cycles(
    apg: &ActionPortGraph,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut adj: HashMap<Symbol, HashSet<Symbol>> = HashMap::new();
    for edge in &apg.edges {
        if let ApgEdge::BindingToAgent { source_agent: Some(src), target_agent: dst, .. } = edge {
            adj.entry(*src).or_default().insert(*dst);
        }
    }

    let mut visited = HashSet::new();
    let mut in_stack: Vec<Symbol> = Vec::new();
    let mut cycle_path: Option<Vec<Symbol>> = None;

    for &start in &apg.agents {
        if visited.contains(&start) {
            continue;
        }
        if dfs_cycle(start, &adj, &mut visited, &mut in_stack, &mut cycle_path) {
            break;
        }
    }

    if let Some(path) = cycle_path {
        let names: Vec<String> = path
            .iter()
            .map(|s| interner.resolve(s.0).unwrap_or("?").to_string())
            .collect();
        diagnostics.push(
            Diagnostic::error(format!(
                "E0603: circular port topology rejected — APG contains a cycle\n\
                 \n\
                 requested: acyclic APG\n\
                 expected:  no cycle in port-to-port edges\n\
                 found:     cycle {}\n\
                 \n\
                 fix: restructure the pipeline so ownership flows in one direction, \
                 or use @shared for data that must flow bidirectionally.",
                names.join(" -> ")
            ))
            .with_label(apg.decorator_span, "in this @pipeline_agent"),
        );
    }
}

fn dfs_cycle(
    node: Symbol,
    adj: &HashMap<Symbol, HashSet<Symbol>>,
    visited: &mut HashSet<Symbol>,
    in_stack: &mut Vec<Symbol>,
    cycle_path: &mut Option<Vec<Symbol>>,
) -> bool {
    if in_stack.contains(&node) {
        // Extract the cycle: from first occurrence of `node` through end of
        // in_stack, plus `node` repeated to close the visible cycle.
        let start_idx = in_stack.iter().position(|n| *n == node).unwrap();
        let mut path = in_stack[start_idx..].to_vec();
        path.push(node);
        *cycle_path = Some(path);
        return true;
    }
    if visited.contains(&node) {
        return false;
    }
    visited.insert(node);
    in_stack.push(node);

    if let Some(neighbors) = adj.get(&node) {
        // Collect into a sorted vec for deterministic output order.
        let mut sorted: Vec<Symbol> = neighbors.iter().copied().collect();
        sorted.sort_by_key(|s| s.0);
        for next in sorted {
            if dfs_cycle(next, adj, visited, in_stack, cycle_path) {
                return true;
            }
        }
    }
    in_stack.pop();
    false
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

// ---------------------------------------------------------------------------
// E0601 — cross-agent exclusive field access (M56 spec §1.1, §6.1)
// ---------------------------------------------------------------------------

/// M56 spec §1.1, §6.1. Walk every agent's method bodies; emit E0601 when
/// a method reads `<other_agent>.<field>` and the field is not `@shared`.
///
/// v1 heuristic: receiver-name → agent-type mapping is title-cased
/// (`drafter` → `Drafter`). See the same TODO(task 12) note on
/// `as_agent_method_call` — Task 12 will replace this with the
/// type-checker-resolved binding type map.
pub fn check_cross_agent_field_access(
    module: &nsl_ast::Module,
    registry: &AgentRegistry,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (_, agent) in registry.iter() {
        for method in &agent.methods {
            if let Some(fn_def) = find_agent_method_fn_def(module, agent.def_symbol, method.name) {
                walk_block_for_cross_field_access(
                    &fn_def.body, agent.def_symbol, registry, interner, diagnostics,
                );
            }
        }
    }
}

fn find_agent_method_fn_def(
    module: &nsl_ast::Module,
    agent_sym: Symbol,
    method_sym: Symbol,
) -> Option<&nsl_ast::decl::FnDef> {
    for stmt in &module.stmts {
        // AgentDef stmts may be wrapped in StmtKind::Decorated for any agent
        // that carries decorators on its declaration; find the inner.
        let agent_def = match &stmt.kind {
            StmtKind::AgentDef(def) => def,
            StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                StmtKind::AgentDef(def) => def,
                _ => continue,
            },
            _ => continue,
        };
        if agent_def.name != agent_sym {
            continue;
        }
        for member in &agent_def.members {
            if let AgentMember::Method(fn_def, _) = member {
                if fn_def.name == method_sym {
                    return Some(fn_def);
                }
            }
        }
    }
    None
}

fn walk_block_for_cross_field_access(
    block: &Block,
    current_agent: Symbol,
    registry: &AgentRegistry,
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::VarDecl { value: Some(val), .. } => {
                walk_expr_for_cross_field(val, current_agent, registry, interner, diags);
            }
            StmtKind::Return(Some(expr)) => {
                walk_expr_for_cross_field(expr, current_agent, registry, interner, diags);
            }
            StmtKind::Expr(expr) => {
                walk_expr_for_cross_field(expr, current_agent, registry, interner, diags);
            }
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                walk_expr_for_cross_field(condition, current_agent, registry, interner, diags);
                walk_block_for_cross_field_access(then_block, current_agent, registry, interner, diags);
                for (cond, b) in elif_clauses {
                    walk_expr_for_cross_field(cond, current_agent, registry, interner, diags);
                    walk_block_for_cross_field_access(b, current_agent, registry, interner, diags);
                }
                if let Some(e) = else_block {
                    walk_block_for_cross_field_access(e, current_agent, registry, interner, diags);
                }
            }
            // TODO(v2): For/While/WhileLet bodies not walked in v1; same
            // rationale as walk_block_for_edges (linear-pipeline-only scope).
            _ => {}
        }
    }
}

fn walk_expr_for_cross_field(
    expr: &Expr,
    current_agent: Symbol,
    registry: &AgentRegistry,
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) {
    match &expr.kind {
        ExprKind::MemberAccess { object, member } => {
            // First recurse into the receiver — it may itself be a nested
            // expression we need to walk (e.g. `(foo()).bar.baz`).
            walk_expr_for_cross_field(object, current_agent, registry, interner, diags);
            if let ExprKind::Ident(obj_sym) = &object.kind {
                let Some(obj_name) = interner.resolve(obj_sym.0) else { return };
                // Skip `self` — same-agent access is always allowed.
                if obj_name == "self" {
                    return;
                }
                // v1 heuristic: title-case the receiver name to find the
                // agent type. See TODO(task 12) on as_agent_method_call.
                let title = uppercase_first(obj_name);
                let Some(other_agent) = registry.get_by_name(&title) else { return };
                if other_agent.def_symbol == current_agent {
                    return;
                }
                let Some(field_name) = interner.resolve(member.0) else { return };
                let is_shared = other_agent
                    .fields
                    .iter()
                    .find(|f| f.name_str == field_name)
                    .map(|f| f.is_shared)
                    .unwrap_or(false);
                if !is_shared {
                    let current_name = interner.resolve(current_agent.0).unwrap_or("?");
                    diags.push(
                        Diagnostic::error(format!(
                            "E0601: agent '{}' cannot access exclusive field '{}' of agent '{}'\n\
                             \n\
                             requested: cross-agent field read\n\
                             expected:  field marked @shared, or self-access inside the\n\
                                        owning agent\n\
                             found:     {} accessing {}.{} (Exclusive)\n\
                             \n\
                             fix: use method-call syntax to move/borrow via a port,\n\
                                  or annotate the field as @shared for read-only access.",
                            current_name, field_name, title,
                            current_name, title, field_name,
                        ))
                        .with_label(expr.span, "exclusive field — owned by another agent"),
                    );
                }
            }
        }
        ExprKind::Call { callee, args } => {
            walk_expr_for_cross_field(callee, current_agent, registry, interner, diags);
            for arg in args {
                walk_expr_for_cross_field(&arg.value, current_agent, registry, interner, diags);
            }
        }
        ExprKind::ListLiteral(items) => {
            for item in items {
                walk_expr_for_cross_field(item, current_agent, registry, interner, diags);
            }
        }
        // Other ExprKind variants: recurse into sub-expressions where the AST
        // exposes them. For Task 8 v1 scope, the above cover all paths the
        // tests exercise. Add more arms if Task 21's E2E examples surface
        // unhandled cases.
        _ => {}
    }
}

fn uppercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
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

    #[test]
    fn extracts_linear_pipeline_apg_from_method_calls() {
        let src = "\
agent Drafter:\n    fn draft(self, prompt: Tensor) -> Tensor:\n        return prompt\n\
agent Reviewer:\n    fn review(self, draft: Tensor) -> Tensor:\n        return draft\n\
@pipeline_agent(agents=[Drafter, Reviewer])\n\
fn pipeline(prompt: Tensor) -> Tensor:\n    let draft = drafter.draft(prompt)\n    return reviewer.review(draft)\n";
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex diagnostics: {:?}", lex_diags);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(parse_result.diagnostics.is_empty(), "parse diagnostics: {:?}", parse_result.diagnostics);

        let mut registry = AgentRegistry::new();
        registry.register_module(&parse_result.module, &interner);

        let mut apgs = Vec::new();
        let mut diags = Vec::new();
        extract_apgs(&parse_result.module, &registry, &interner, &mut apgs, &mut diags);
        assert!(diags.is_empty(), "unexpected diagnostics during extraction: {:?}", diags);

        assert_eq!(apgs.len(), 1, "expected one @pipeline_agent function; got {}", apgs.len());
        let apg = &apgs[0];
        assert_eq!(
            apg.edges.len(),
            2,
            "expected 2 edges: prompt→drafter.in_prompt, draft→reviewer.in_draft; got {:?}",
            apg.edges
        );
    }

    #[test]
    fn cycle_detection_flags_bidirectional_send() {
        // A calls B; then calls A again - produces an A→B→A cycle.
        let src = "\
agent A:\n    fn a_fn(self, x: Tensor) -> Tensor:\n        return x\n\
agent B:\n    fn b_fn(self, y: Tensor) -> Tensor:\n        return y\n\
@pipeline_agent(agents=[A, B])\n\
fn loop_pipe(x: Tensor) -> Tensor:\n    let y = a.a_fn(x)\n    let z = b.b_fn(y)\n    return a.a_fn(z)\n";
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex diagnostics: {:?}", lex_diags);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(parse_result.diagnostics.is_empty(), "parse diagnostics: {:?}", parse_result.diagnostics);

        let mut registry = AgentRegistry::new();
        registry.register_module(&parse_result.module, &interner);

        let mut apgs = Vec::new();
        let mut diags = Vec::new();
        extract_apgs(&parse_result.module, &registry, &interner, &mut apgs, &mut diags);
        assert_eq!(apgs.len(), 1);
        for apg in &apgs {
            detect_cycles(apg, &interner, &mut diags);
        }
        assert!(diags.iter().any(|d| d.message.contains("E0603")),
            "expected E0603 cycle error, got: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>());
    }

    #[test]
    fn no_cycle_flagged_for_linear_pipeline() {
        // Reuse the linear-pipeline fixture from Task 6; assert 0 E0603 diagnostics.
        let src = "\
agent Drafter:\n    fn draft(self, prompt: Tensor) -> Tensor:\n        return prompt\n\
agent Reviewer:\n    fn review(self, draft: Tensor) -> Tensor:\n        return draft\n\
@pipeline_agent(agents=[Drafter, Reviewer])\n\
fn pipeline(prompt: Tensor) -> Tensor:\n    let draft = drafter.draft(prompt)\n    return reviewer.review(draft)\n";
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);

        let mut registry = AgentRegistry::new();
        registry.register_module(&parse_result.module, &interner);

        let mut apgs = Vec::new();
        let mut diags = Vec::new();
        extract_apgs(&parse_result.module, &registry, &interner, &mut apgs, &mut diags);
        for apg in &apgs {
            detect_cycles(apg, &interner, &mut diags);
        }
        assert!(!diags.iter().any(|d| d.message.contains("E0603")),
            "linear pipeline should not produce E0603, got: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>());
    }

    // -------------------------------------------------------------------------
    // Task 8: E0601 cross-agent exclusive field access
    // -------------------------------------------------------------------------

    #[test]
    fn cross_agent_exclusive_field_access_errors() {
        // Reviewer.review receives `drafter: Drafter` and reads `drafter.kv_cache`.
        // Title-casing "drafter" -> "Drafter" matches the registered agent.
        // Drafter.draft reads `self.kv_cache` (same-agent, exempt).
        let src = "\
agent Drafter:\n    kv_cache: Tensor = empty()\n    fn draft(self) -> Tensor:\n        return self.kv_cache\n\
agent Reviewer:\n    fn review(self, drafter: Drafter) -> Tensor:\n        return drafter.kv_cache\n";
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(parse_result.diagnostics.is_empty(), "parse: {:?}", parse_result.diagnostics);

        let mut registry = AgentRegistry::new();
        registry.register_module(&parse_result.module, &interner);
        let mut diags = Vec::new();
        check_cross_agent_field_access(&parse_result.module, &registry, &interner, &mut diags);

        assert!(diags.iter().any(|d| d.message.contains("E0601")),
            "expected E0601, got: {:?}", diags.iter().map(|d| &d.message).collect::<Vec<_>>());
        // Self-access (Drafter.draft reading self.kv_cache) MUST NOT produce E0601.
        let e0601_count = diags.iter().filter(|d| d.message.contains("E0601")).count();
        assert_eq!(e0601_count, 1, "exactly one E0601 expected (Reviewer.review's drafter.kv_cache); got {}", e0601_count);
    }

    #[test]
    fn cross_agent_shared_field_access_allowed() {
        let src = "\
agent Drafter:\n    @shared\n    cache: Tensor = empty()\n    fn draft(self) -> Tensor:\n        return self.cache\n\
agent Reviewer:\n    fn review(self, drafter: Drafter) -> Tensor:\n        return drafter.cache\n";
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        let mut registry = AgentRegistry::new();
        registry.register_module(&parse_result.module, &interner);
        let mut diags = Vec::new();
        check_cross_agent_field_access(&parse_result.module, &registry, &interner, &mut diags);
        assert!(!diags.iter().any(|d| d.message.contains("E0601")),
            "expected no E0601 for @shared field, got: {:?}", diags);
    }
}
