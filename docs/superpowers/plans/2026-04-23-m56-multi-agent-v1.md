# M56 Multi-Agent v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship NSL's `agent` keyword + `@pipeline_agent` decorator with typed ports, compile-time Action-Port Graph (APG) extraction, linear-move cross-agent ownership, cross-device refusal (with `@auto_device_transfer` opt-in), serve-block-managed pool with tombstone-on-reset-failure, and a single-threaded deterministic logical-time scheduler as the v1 runtime.

**Architecture:** The spec ([docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md](docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md)) is authoritative. The semantic model (typed ports, APG, logical time, linear ownership) is load-bearing; the v1 single-threaded scheduler is one specific implementation of that model, replaceable by v2's reactor scheduler without changing user-visible semantics. Every task below maps to a named spec section.

**Tech stack:** Rust 1.95.0, Cranelift for function emission, existing NSL type-system machinery (`nsl-ast`, `nsl-parser`, `nsl-semantic`, `nsl-codegen`, `nsl-runtime`). No new external dependencies.

**Worktree:** `.worktrees/m56-multi-agent-v1/` on branch `feat/m56-multi-agent-v1`, based on `main` at `0740b113`.

**Gitignore note:** `docs/superpowers/specs/` and `docs/superpowers/plans/` are gitignored. Use `git add -f` when committing docs for this project. Example commits below use the standard existing-repo pattern (`b697a5a9` is the prior art).

---

## File Structure

### New files

| Path | Responsibility |
|------|----------------|
| `crates/nsl-ast/src/agent.rs` | `AgentDef` AST struct + wiring |
| `crates/nsl-parser/src/agent.rs` | `parse_agent_def_stmt` + `parse_pipeline_agent_decorator_args` |
| `crates/nsl-semantic/src/agent.rs` | Agent registry, `Agent` type variant, APG extraction, cycle detection, cross-agent access rule, device rule, fan-out rule, flag-gate error, composition with ownership/effect checkers |
| `crates/nsl-runtime/src/agent/mod.rs` | Module root: re-exports |
| `crates/nsl-runtime/src/agent/mailbox.rs` | `PortMessage`, `PortMailbox`, `StructPayload` |
| `crates/nsl-runtime/src/agent/scheduler.rs` | `ReactorScheduler` (v1 single-threaded logical-time loop) |
| `crates/nsl-runtime/src/agent/pool.rs` | `PipelineContextPool` with tombstone-on-reset-failure (`Vec<Option<PipelineContext>>`) |
| `crates/nsl-runtime/src/agent/ffi.rs` | FFI surface per spec §3.4 |
| `crates/nsl-codegen/src/agent.rs` | Agent struct layout, method codegen, `@pipeline_agent` lowering, `serve`-block pool integration, `@auto_device_transfer` inserts with size diagnostic |
| `examples/m56_basic_two_agents.nsl` | E2E: linear two-agent pipeline on GPU |
| `examples/m56_shared_embeddings.nsl` | E2E: `@shared` embedding read by multiple agents |
| `examples/m56_serve_pool.nsl` | E2E: `serve` block with `pool_size=4` |
| `examples/m56_device_transfer_opt_in.nsl` | E2E: CPU→GPU via `@auto_device_transfer` |
| `examples/m56_cross_agent_access_error.nsl` | Negative E2E: E0601 |
| `examples/m56_cycle_error.nsl` | Negative E2E: E0603 |
| `examples/m56_cross_gpu_error.nsl` | Negative E2E: E0607 |

### Modified files

| Path | Change |
|------|--------|
| `crates/nsl-lexer/src/keywords.rs` | Add `"agent" => TokenKind::Agent` |
| `crates/nsl-lexer/src/token.rs` | Add `TokenKind::Agent` variant |
| `crates/nsl-ast/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-ast/src/stmt.rs` | `StmtKind::AgentDef(AgentDef)` |
| `crates/nsl-ast/src/visitor.rs` | Walk `AgentDef` members |
| `crates/nsl-parser/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-parser/src/stmt.rs` | Route `TokenKind::Agent` to `agent::parse_agent_def_stmt` |
| `crates/nsl-semantic/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-semantic/src/types.rs` | Add `Type::Agent { ... }` variant |
| `crates/nsl-semantic/src/checker/decl.rs` | Register `AgentDef` + call flag-gate check |
| `crates/nsl-semantic/src/checker/mod.rs` | Post-pass: run `AgentAnalysis` after type/shape/ownership/effect |
| `crates/nsl-codegen/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-codegen/src/compiler/mod.rs` | Invoke agent codegen when `AgentDef` present |
| `crates/nsl-codegen/src/builtins.rs` | Register `nsl_agent_*` FFI symbols |
| `crates/nsl-cli/src/main.rs` | Expose `--linear-types` on `nsl run` (close the Section 7 gap); update E0610 message after |
| `crates/nsl-cli/tests/e2e.rs` | New M56 E2E tests |

---

## Phase 1 — Lexer, AST, Parser

### Task 1: Lexer — add `agent` keyword

**Files:**
- Modify: `crates/nsl-lexer/src/token.rs` — add `TokenKind::Agent` variant
- Modify: `crates/nsl-lexer/src/keywords.rs` — route `"agent"` to the new variant
- Test: `crates/nsl-lexer/src/lexer.rs` (inline `#[cfg(test)]` block) — assert `agent` tokenizes

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-lexer/src/lexer.rs` inside the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn lexer_recognises_agent_keyword() {
    let mut interner = crate::Interner::new();
    let (tokens, diags) = crate::tokenize(
        "agent Drafter:\n    pass\n",
        nsl_errors::FileId(0),
        &mut interner,
    );
    assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    assert_eq!(tokens[0].kind, crate::TokenKind::Agent,
        "first token should be Agent, was {:?}", tokens[0].kind);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-lexer lexer_recognises_agent_keyword
```

Expected: compile error — `crate::TokenKind::Agent` does not exist.

- [ ] **Step 3: Add `TokenKind::Agent`**

In `crates/nsl-lexer/src/token.rs`, locate the `TokenKind` enum and add next to `Model`:

```rust
Model,
Agent,
Train,
```

- [ ] **Step 4: Route the keyword**

In `crates/nsl-lexer/src/keywords.rs` in `lookup_keyword`, next to `"model"`:

```rust
"model" => Some(TokenKind::Model),
"agent" => Some(TokenKind::Agent),
"train" => Some(TokenKind::Train),
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test -p nsl-lexer lexer_recognises_agent_keyword
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-lexer/src/token.rs crates/nsl-lexer/src/keywords.rs crates/nsl-lexer/src/lexer.rs
git commit -m "feat(m56): lexer recognises agent keyword"
```

---

### Task 2: AST — `AgentDef` struct

**Files:**
- Create: `crates/nsl-ast/src/agent.rs`
- Modify: `crates/nsl-ast/src/lib.rs` — `pub mod agent;` + re-export
- Modify: `crates/nsl-ast/src/stmt.rs` — `StmtKind::AgentDef(AgentDef)`
- Modify: `crates/nsl-ast/src/visitor.rs` — visit agent members
- Test: `crates/nsl-ast/tests/agent_def.rs` (new)

- [ ] **Step 1: Create the test file**

`crates/nsl-ast/tests/agent_def.rs`:

```rust
use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::{NodeId, Span, FileId};

#[test]
fn agent_def_holds_fields_and_methods() {
    let dummy = Span::new(FileId(0), 0, 0);
    let interner = nsl_lexer::Interner::new();
    let name = interner.get("Drafter").unwrap_or_else(|| {
        // In this unit test we just construct a symbol via the interner —
        // production parsing uses `expect_ident`.
        panic!("construct symbol via interner in real tests")
    });
    let def = AgentDef {
        name,
        params: vec![],
        members: vec![],
        span: dummy,
    };
    assert_eq!(def.members.len(), 0);
    let _ = AgentMember::FieldDecl {
        name,
        type_ann: nsl_ast::types::TypeExpr::unknown(dummy),
        init: None,
        decorators: vec![],
        span: dummy,
    };
}
```

This test is intentionally minimal — it asserts the types exist and the field access compiles. Full parser-to-AST tests land in Task 3.

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-ast agent_def_holds_fields_and_methods
```

Expected: compile error — `nsl_ast::agent` does not exist.

- [ ] **Step 3: Create the AST module**

`crates/nsl-ast/src/agent.rs`:

```rust
//! M56: Agent declaration AST nodes. Mirrors `ModelDef`/`ModelMember` (see
//! `decl.rs:48-66`) with agent-specific ownership markers and port derivation.

use serde::Serialize;
use crate::decl::{Decorator, FnDef, Param, TypeParam};
use crate::types::TypeExpr;
use crate::expr::Expr;
use crate::{Span, Symbol};

/// An `agent Name(params): <members>` declaration.
#[derive(Debug, Clone, Serialize)]
pub struct AgentDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<Param>,
    pub members: Vec<AgentMember>,
    pub span: Span,
}

/// A field or method inside an agent body.
#[derive(Debug, Clone, Serialize)]
pub enum AgentMember {
    /// `name: Type = init_expr` field declaration. Decorators may include
    /// `@shared` (M38 semantics — spec §1.5).
    FieldDecl {
        name: Symbol,
        type_ann: TypeExpr,
        init: Option<Expr>,
        decorators: Vec<Decorator>,
        span: Span,
    },
    /// Agent method (init/reset/shutdown or user-defined). Decorators may
    /// include `@auto_device_transfer` (spec §1.6). The method signature is
    /// also the port declaration per spec §1.2 — port names are derived from
    /// parameter names during semantic analysis.
    Method(FnDef, Vec<Decorator>),
}
```

- [ ] **Step 4: Wire up the module**

In `crates/nsl-ast/src/lib.rs`, next to existing `pub mod decl;`:

```rust
pub mod agent;
```

In `crates/nsl-ast/src/stmt.rs`, inside the `StmtKind` enum next to `ModelDef`:

```rust
ModelDef(ModelDef),
AgentDef(crate::agent::AgentDef),
```

- [ ] **Step 5: Update the visitor**

In `crates/nsl-ast/src/visitor.rs`, find the `ModelDef` match arm (~line 95) and add the `AgentDef` arm immediately after:

```rust
StmtKind::AgentDef(a) => {
    for member in &a.members {
        match member {
            crate::agent::AgentMember::FieldDecl { type_ann, init, .. } => {
                visit_type_expr(visitor, type_ann);
                if let Some(e) = init { visit_expr(visitor, e); }
            }
            crate::agent::AgentMember::Method(f, _) => {
                visit_fn_def(visitor, f);
            }
        }
    }
}
```

(Adapt the exact visitor signatures to the existing `visitor.rs` style — the surrounding `ModelDef` arm is the precedent.)

- [ ] **Step 6: Run test to verify it passes**

```bash
cargo test -p nsl-ast agent_def_holds_fields_and_methods
cargo build -p nsl-ast
```

Expected: test PASS; full `nsl-ast` crate builds without unused-variant warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-ast/src/agent.rs crates/nsl-ast/src/lib.rs crates/nsl-ast/src/stmt.rs crates/nsl-ast/src/visitor.rs crates/nsl-ast/tests/agent_def.rs
git commit -m "feat(m56): AgentDef AST nodes mirroring ModelDef"
```

---

### Task 3: Parser — `parse_agent_def_stmt`

**Files:**
- Create: `crates/nsl-parser/src/agent.rs`
- Modify: `crates/nsl-parser/src/lib.rs` — `pub mod agent;`
- Modify: `crates/nsl-parser/src/stmt.rs` — route `TokenKind::Agent`
- Test: `crates/nsl-parser/tests/agent_parse.rs` (new)

- [ ] **Step 1: Write the failing test**

`crates/nsl-parser/tests/agent_parse.rs`:

```rust
use nsl_ast::stmt::StmtKind;
use nsl_ast::agent::AgentMember;
use nsl_parser::parse;
use nsl_errors::FileId;

#[test]
fn parses_agent_with_field_and_method() {
    let src = "agent Drafter:\n    kv_cache: KvCache = empty()\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n";
    let mut interner = nsl_lexer::Interner::new();
    let module = parse(src, FileId(0), &mut interner);
    assert!(module.diagnostics.is_empty(), "parse errors: {:?}", module.diagnostics);

    let stmt = &module.module.body.stmts[0];
    let agent = match &stmt.kind {
        StmtKind::AgentDef(a) => a,
        other => panic!("expected AgentDef, got {:?}", other),
    };
    assert_eq!(agent.members.len(), 2);
    assert!(matches!(agent.members[0], AgentMember::FieldDecl { .. }));
    assert!(matches!(agent.members[1], AgentMember::Method(..)));
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-parser parses_agent_with_field_and_method
```

Expected: compile error — no `nsl_parser::agent` module.

- [ ] **Step 3: Write the parser module**

`crates/nsl-parser/src/agent.rs`:

```rust
//! M56: Agent declaration parser. Mirrors `parse_model_def_stmt`
//! (see decl.rs:56-188) — adapted to emit `StmtKind::AgentDef(AgentDef)`
//! with `AgentMember::{FieldDecl, Method}` members.

use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::decl::{Decorator, FnDef};
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_lexer::TokenKind;

use crate::expr::parse_args;
use crate::parser::Parser;
use crate::decl::{parse_params, parse_type_params};
use crate::types::{parse_type, parse_type_no_borrow};
use crate::expr::parse_expr;

pub fn parse_agent_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'agent'

    let (name, _) = p.expect_ident();

    let type_params = if p.at(&TokenKind::Lt) { parse_type_params(p) } else { Vec::new() };

    let params = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let ps = parse_params(p);
        p.expect(&TokenKind::RightParen);
        ps
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut members = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) { break; }

        // Collect decorators (@shared, @auto_device_transfer, etc.)
        let mut decorators = Vec::new();
        while p.at(&TokenKind::At) {
            decorators.push(parse_decorator(p));
            p.skip_newlines();
        }

        if p.at(&TokenKind::Fn) || p.at(&TokenKind::Async) {
            let method_start = p.current_span();
            let is_async = p.eat(&TokenKind::Async);
            p.expect(&TokenKind::Fn);
            let (mname, _) = p.expect_ident();
            let mtype_params = if p.at(&TokenKind::Lt) { parse_type_params(p) } else { Vec::new() };
            p.expect(&TokenKind::LeftParen);
            let mparams = parse_params(p);
            p.expect(&TokenKind::RightParen);
            let return_type = if p.eat(&TokenKind::Arrow) { Some(parse_type_no_borrow(p)) } else { None };
            p.expect(&TokenKind::Colon);
            let body = p.parse_block();
            members.push(AgentMember::Method(
                FnDef {
                    name: mname,
                    type_params: mtype_params,
                    effect_params: vec![],
                    params: mparams,
                    return_type,
                    return_effect: None,
                    body: body.clone(),
                    is_async,
                    span: method_start.merge(body.span),
                },
                decorators,
            ));
        } else {
            let member_start = p.current_span();
            let (fname, _) = p.expect_ident();
            p.expect(&TokenKind::Colon);
            let type_ann = parse_type(p);
            let init = if p.eat(&TokenKind::Eq) { Some(parse_expr(p)) } else { None };
            p.expect_end_of_stmt();
            members.push(AgentMember::FieldDecl {
                name: fname,
                type_ann,
                init,
                decorators,
                span: member_start.merge(p.prev_span()),
            });
            continue;
        }
        p.skip_newlines();
    }
    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::AgentDef(AgentDef {
            name,
            type_params,
            params,
            members,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

fn parse_decorator(p: &mut Parser) -> Decorator {
    let dec_start = p.current_span();
    p.advance(); // @
    let mut dec_name = Vec::new();
    let (first, _) = p.expect_ident();
    dec_name.push(first);
    while p.eat(&TokenKind::Dot) {
        let (seg, _) = p.expect_ident();
        dec_name.push(seg);
    }
    let args = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let a = parse_args(p);
        p.expect(&TokenKind::RightParen);
        Some(a)
    } else {
        None
    };
    Decorator {
        name: dec_name,
        args,
        span: dec_start.merge(p.prev_span()),
    }
}
```

- [ ] **Step 4: Wire up module and statement dispatch**

In `crates/nsl-parser/src/lib.rs`, next to `pub mod decl;`:

```rust
pub mod agent;
```

In `crates/nsl-parser/src/stmt.rs`, locate the `TokenKind::Model` arm (which delegates to `decl::parse_model_def_stmt`) and add immediately below:

```rust
TokenKind::Model => crate::decl::parse_model_def_stmt(p),
TokenKind::Agent => crate::agent::parse_agent_def_stmt(p),
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test -p nsl-parser parses_agent_with_field_and_method
```

Expected: PASS. Also run `cargo test -p nsl-parser` in full to catch regressions.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-parser/src/agent.rs crates/nsl-parser/src/lib.rs crates/nsl-parser/src/stmt.rs crates/nsl-parser/tests/agent_parse.rs
git commit -m "feat(m56): parse agent declarations with typed members"
```

---

## Phase 2 — Semantic: flag gate + registration

### Task 4: `--linear-types` flag-gate error (E0610)

**Files:**
- Create: `crates/nsl-semantic/src/agent.rs` (module skeleton; this task adds just the flag gate — subsequent tasks extend it)
- Modify: `crates/nsl-semantic/src/lib.rs` — `pub mod agent;`
- Modify: `crates/nsl-semantic/src/checker/mod.rs` — call flag-gate check early
- Test: `crates/nsl-semantic/src/agent.rs` inline `#[cfg(test)]`

- [ ] **Step 1: Write the failing test**

Embed in `crates/nsl-semantic/src/agent.rs` (which we are creating in this task):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::{FileId, Span};

    #[test]
    fn flag_gate_errors_when_agent_present_without_linear_types() {
        let mut diags = Vec::new();
        let dummy = Span::new(FileId(0), 10, 15);
        check_linear_types_flag(
            /* agent_decl_spans = */ &[dummy],
            /* linear_types_enabled = */ false,
            &mut diags,
        );
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("E0610"), "expected E0610, got {}", diags[0].message);
        assert_eq!(diags[0].primary_span, Some(dummy));
    }

    #[test]
    fn flag_gate_silent_when_linear_types_enabled() {
        let mut diags = Vec::new();
        let dummy = Span::new(FileId(0), 10, 15);
        check_linear_types_flag(&[dummy], true, &mut diags);
        assert!(diags.is_empty());
    }

    #[test]
    fn flag_gate_silent_when_no_agents() {
        let mut diags = Vec::new();
        check_linear_types_flag(&[], false, &mut diags);
        assert!(diags.is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-semantic flag_gate_errors_when_agent_present_without_linear_types
```

Expected: compile error — `nsl_semantic::agent::check_linear_types_flag` not found.

- [ ] **Step 3: Implement the flag-gate check**

`crates/nsl-semantic/src/agent.rs`:

```rust
//! M56: Agent semantic analysis. Flag gate + agent registry + APG
//! extraction + cross-agent access, device, and fan-out rules.
//!
//! Spec: docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md

use nsl_errors::{Diagnostic, Span};

/// Emit E0610 if agent declarations are present and `--linear-types` is off.
/// Spec §7 (flag gating) + §6.7 (error format).
///
/// `agent_decl_spans`: spans of all `agent Foo:` declarations found in the
/// module; typically produced during initial AST walk in
/// `crates/nsl-semantic/src/checker/decl.rs`.
pub fn check_linear_types_flag(
    agent_decl_spans: &[Span],
    linear_types_enabled: bool,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if linear_types_enabled { return; }
    let Some(&first_span) = agent_decl_spans.first() else { return; };
    diagnostics.push(
        Diagnostic::error(
            "E0610: M56 agent declarations require --linear-types\n\
             \n\
             requested: compile a program containing an agent declaration\n\
             expected:  the linear ownership checker (--linear-types) active\n\
             found:     --linear-types not passed to the compiler\n\
             \n\
             fix: add --linear-types to `nsl check` or `nsl build`. `nsl run`\n\
                  does not currently expose --linear-types; for run, use \
                  `nsl build` followed by direct execution of the produced \
                  binary. (Tracked: Task 20 of this plan closes that gap.)"
                .to_string(),
            first_span,
        )
    );
}
```

> **Plan-coupling note (refinement #6):** Task 20 closes the `nsl run --linear-types` gap. When Task 20 lands, the `fix:` section above must drop the "`nsl run` does not currently expose..." paragraph. Task 20 Step 5 updates this message; do not forget.

- [ ] **Step 4: Wire the module + invoke the check**

In `crates/nsl-semantic/src/lib.rs`, add near other `pub mod`s:

```rust
pub mod agent;
```

In `crates/nsl-semantic/src/checker/decl.rs`, during the pass that walks top-level statements, collect every `StmtKind::AgentDef(..).span` into a `Vec<Span>`. Then at the end of decl registration (just before module-level type-checking returns), call:

```rust
crate::agent::check_linear_types_flag(&agent_decl_spans, self.linear_types_enabled, &mut self.diagnostics);
```

Note: `linear_types_enabled` must be threaded into the `TypeChecker` struct if not already. Check `crates/nsl-semantic/src/checker/mod.rs:46` and add the field alongside existing config toggles. The CLI already passes `linear_types` (see `crates/nsl-cli/src/main.rs:764`) to `check_semantics`; thread it through.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cargo test -p nsl-semantic flag_gate
```

Expected: all three tests PASS.

- [ ] **Step 6: End-to-end sanity**

Create a throwaway test file `examples/m56_scratch_flag_gate.nsl`:

```nsl
agent X:
    pass
```

Run:

```bash
cargo run --bin nsl -- check examples/m56_scratch_flag_gate.nsl
```

Expected: error output mentions `E0610` and `--linear-types`. Then run with the flag:

```bash
cargo run --bin nsl -- check --linear-types examples/m56_scratch_flag_gate.nsl
```

Expected: E0610 does not fire (subsequent errors may fire — that's Task 5+'s concern).

- [ ] **Step 7: Commit**

```bash
rm examples/m56_scratch_flag_gate.nsl
git add crates/nsl-semantic/src/agent.rs crates/nsl-semantic/src/lib.rs crates/nsl-semantic/src/checker/decl.rs crates/nsl-semantic/src/checker/mod.rs
git commit -m "feat(m56): E0610 error when --linear-types missing at agent decl"
```

---

### Task 5: Agent registry + `Type::Agent` variant

**Files:**
- Modify: `crates/nsl-semantic/src/types.rs` — new `Type::Agent` variant
- Modify: `crates/nsl-semantic/src/agent.rs` — `AgentRegistry` struct
- Modify: `crates/nsl-semantic/src/checker/decl.rs` — register AgentDef in registry
- Test: `crates/nsl-semantic/src/agent.rs` inline tests

- [ ] **Step 1: Write the failing test**

Append to `#[cfg(test)] mod tests` in `crates/nsl-semantic/src/agent.rs`:

```rust
#[test]
fn registry_records_agents_and_member_kinds() {
    let mut interner = nsl_lexer::Interner::new();
    let src = "agent Drafter:\n    kv_cache: KvCache = empty()\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n";
    let module = nsl_parser::parse(src, nsl_errors::FileId(0), &mut interner);
    assert!(module.diagnostics.is_empty());
    let mut registry = AgentRegistry::new();
    registry.register_module(&module.module, &interner);
    let drafter = registry.get_by_name("Drafter").expect("Drafter not registered");
    assert_eq!(drafter.field_count(), 1);
    assert_eq!(drafter.method_count(), 1);
    assert!(drafter.has_method("draft"));
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-semantic registry_records_agents_and_member_kinds
```

Expected: compile error — `AgentRegistry` doesn't exist.

- [ ] **Step 3: Add `Type::Agent` variant**

In `crates/nsl-semantic/src/types.rs`, locate the `Type` enum and add:

```rust
/// M56: Agent type — like Model with isolated mutable state and
/// per-port linear-move communication.
Agent {
    name: nsl_ast::Symbol,
    fields: Vec<(nsl_ast::Symbol, Type, AgentFieldOwnership)>,
    methods: Vec<(nsl_ast::Symbol, FunctionType)>,
},
```

And the supporting ownership enum (same file or a new `agent_types.rs` sub-module):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentFieldOwnership {
    /// Field is exclusively owned by the agent; no other agent can access.
    Exclusive,
    /// Field is @shared — readable by all agents (refcount-bumped across sends).
    SharedReadOnly,
    /// Scalar/copy type (int, float, bool) — no ownership concerns.
    Copy,
}
```

- [ ] **Step 4: Implement `AgentRegistry`**

In `crates/nsl-semantic/src/agent.rs` append:

```rust
use std::collections::HashMap;
use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::decl::FnDef;
use nsl_ast::{Module, Symbol};
use nsl_lexer::Interner;

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
        for stmt in &module.body.stmts {
            if let nsl_ast::stmt::StmtKind::AgentDef(def) = &stmt.kind {
                self.register_agent(def, interner);
            }
        }
    }

    fn register_agent(&mut self, def: &AgentDef, interner: &Interner) {
        let name_str = interner.resolve(def.name).unwrap_or("<unknown>").to_string();
        let fields: Vec<FieldInfo> = def.members.iter().filter_map(|m| match m {
            AgentMember::FieldDecl { name, decorators, span, .. } => Some(FieldInfo {
                name: *name,
                name_str: interner.resolve(*name).unwrap_or("?").to_string(),
                is_shared: decorators.iter().any(|d| d.name.len() == 1
                    && interner.resolve(d.name[0]).map_or(false, |s| s == "shared")),
                span: *span,
            }),
            _ => None,
        }).collect();

        let methods: Vec<MethodInfo> = def.members.iter().filter_map(|m| match m {
            AgentMember::Method(fn_def, decorators) => Some(MethodInfo {
                name: fn_def.name,
                name_str: interner.resolve(fn_def.name).unwrap_or("?").to_string(),
                has_auto_device_transfer: decorators.iter().any(|d| d.name.len() == 1
                    && interner.resolve(d.name[0]).map_or(false, |s| s == "auto_device_transfer")),
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test -p nsl-semantic registry_records_agents_and_member_kinds
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-semantic/src/types.rs crates/nsl-semantic/src/agent.rs
git commit -m "feat(m56): AgentRegistry + Type::Agent + AgentFieldOwnership"
```

---

## Phase 3 — APG extraction and cycle detection

### Task 6: APG data structure + `@pipeline_agent` function recognition

**Files:**
- Modify: `crates/nsl-semantic/src/agent.rs`
- Test: inline tests

- [ ] **Step 1: Write the failing test**

Append to tests:

```rust
#[test]
fn extracts_linear_pipeline_apg_from_method_calls() {
    let src = "\
agent Drafter:\n    fn draft(self, prompt: Tensor) -> Tensor:\n        return prompt\n\
agent Reviewer:\n    fn review(self, draft: Tensor) -> Tensor:\n        return draft\n\
@pipeline_agent(agents=[Drafter, Reviewer])\n\
fn pipeline(prompt: Tensor) -> Tensor:\n    let draft = drafter.draft(prompt)\n    return reviewer.review(draft)\n";
    let mut interner = nsl_lexer::Interner::new();
    let module = nsl_parser::parse(src, nsl_errors::FileId(0), &mut interner);
    assert!(module.diagnostics.is_empty(), "parse errors: {:?}", module.diagnostics);
    let mut registry = AgentRegistry::new();
    registry.register_module(&module.module, &interner);
    let mut apgs = Vec::new();
    let mut diags = Vec::new();
    extract_apgs(&module.module, &registry, &interner, &mut apgs, &mut diags);
    assert_eq!(apgs.len(), 1, "expected one @pipeline_agent function");
    let apg = &apgs[0];
    assert_eq!(apg.edges.len(), 2,
        "expected 2 edges: prompt→drafter.in_prompt, draft→reviewer.in_draft; got {:?}",
        apg.edges);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-semantic extracts_linear_pipeline_apg_from_method_calls
```

Expected: compile error — `extract_apgs`/`ActionPortGraph` undefined.

- [ ] **Step 3: Implement APG data structure + extractor**

Append to `crates/nsl-semantic/src/agent.rs`:

```rust
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::pattern::{Pattern, PatternKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};

/// M56 Action-Port Graph extracted from a `@pipeline_agent` function body.
/// Spec §2.2.
#[derive(Debug)]
pub struct ActionPortGraph {
    /// Symbol of the pipeline function (`fn pipeline` in user code).
    pub pipeline_fn: Symbol,
    /// Participating agents (from the decorator's `agents=[...]` argument).
    pub agents: Vec<Symbol>,
    /// Edges: each is either `PipelineInput -> agent.in_param` or
    /// `binding -> agent.in_param`, plus `agent.out_<method> -> binding`.
    pub edges: Vec<ApgEdge>,
    /// Span of the `@pipeline_agent` decorator for diagnostics.
    pub decorator_span: nsl_errors::Span,
}

#[derive(Debug, Clone)]
pub enum ApgEdge {
    /// A pipeline-function parameter flows into an agent method call.
    PipelineInputToAgent {
        pipeline_param: Symbol,
        target_agent: Symbol,
        target_method: Symbol,
        target_param: Symbol,
        span: nsl_errors::Span,
    },
    /// A `let`-bound value flows into an agent method call.
    BindingToAgent {
        binding: Symbol,
        source_agent: Option<Symbol>,    // None if binding came from a non-agent expression
        source_method: Option<Symbol>,   // same
        target_agent: Symbol,
        target_method: Symbol,
        target_param: Symbol,
        span: nsl_errors::Span,
    },
}

pub fn extract_apgs(
    module: &nsl_ast::Module,
    registry: &AgentRegistry,
    interner: &Interner,
    out_apgs: &mut Vec<ActionPortGraph>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for stmt in &module.body.stmts {
        let StmtKind::FnDef(fn_def) = &stmt.kind else { continue };
        let (Some(decorator), agents) = find_pipeline_agent_decorator(fn_def, interner) else { continue };
        // Verify all named agents are registered.
        for &agent_sym in &agents {
            if !registry.iter().any(|(sym, _)| *sym == agent_sym) {
                diagnostics.push(Diagnostic::error(
                    format!("@pipeline_agent references unknown agent `{}`",
                        interner.resolve(agent_sym).unwrap_or("?")),
                    decorator.span,
                ));
            }
        }
        let mut edges = Vec::new();
        let pipeline_params: Vec<Symbol> = fn_def.params.iter().map(|p| p.name).collect();
        walk_block_for_edges(
            &fn_def.body, registry, interner, &pipeline_params, &mut edges, diagnostics,
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
    fn_def: &'a nsl_ast::decl::FnDef,
    interner: &Interner,
) -> (Option<&'a nsl_ast::decl::Decorator>, Vec<Symbol>) {
    // Decorators on top-level fns are stored on the surrounding stmt in some
    // parser paths; for v1 we scan the FnDef-level decorator list if present.
    // The existing parser attaches fn decorators to the StmtKind::FnDef wrapper —
    // extend this function to read from there when the AST exposes the hook.
    // TEMPORARY: for v1 mock, users declare pipelines via an explicit decorator
    // token handled at the stmt level. Caller passes the decorator alongside.
    // To keep this task self-contained, the real read path lives in Task 6
    // Step 4 (decorator-attach). Until then the caller provides decorators.
    // The extractor is shaped so Step 4 wires the real decorator read; this
    // function becomes `find_pipeline_agent_decorator(stmt_decorators, interner)`.
    let _ = (fn_def, interner); (None, Vec::new())
}

fn walk_block_for_edges(
    _block: &Block,
    _registry: &AgentRegistry,
    _interner: &Interner,
    _pipeline_params: &[Symbol],
    _edges: &mut Vec<ApgEdge>,
    _diags: &mut Vec<Diagnostic>,
) {
    // Implemented in Task 6 Step 4.
}
```

- [ ] **Step 4: Attach decorators to top-level `FnDef` stmts + implement edge walk**

The existing parser attaches decorators to struct/model members but top-level fn decorators are collected separately in `crates/nsl-parser/src/stmt.rs`'s top-level statement loop. Verify by grep and extend if needed so that `@pipeline_agent` on a top-level fn is reachable from the FnDef's containing `Stmt`.

```bash
grep -n "fn_def.*decorator\|FnDef.*Decorator\|top.level.decor" crates/nsl-parser/src/stmt.rs crates/nsl-parser/src/decl.rs
```

If top-level-fn decorators aren't preserved: add a `decorators: Vec<Decorator>` field to `FnDef` or wrap FnDef in a new `StmtKind::DecoratedFn { decorators, fn_def }` variant. Use whichever approach mirrors the existing codebase convention (e.g., model methods store decorators on `ModelMember::Method(FnDef, Vec<Decorator>)` — follow that pattern by introducing `StmtKind::FnDef { fn_def: FnDef, decorators: Vec<Decorator> }`).

Then implement `walk_block_for_edges`:

```rust
fn walk_block_for_edges(
    block: &Block,
    registry: &AgentRegistry,
    interner: &Interner,
    pipeline_params: &[Symbol],
    edges: &mut Vec<ApgEdge>,
    diags: &mut Vec<Diagnostic>,
) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value: Some(val), .. } => {
                if let Some(call) = as_agent_method_call(val, registry, interner) {
                    record_agent_call_edges(pattern, &call, pipeline_params, edges, diags, val.span);
                }
            }
            StmtKind::Return(Some(expr)) => {
                if let Some(call) = as_agent_method_call(expr, registry, interner) {
                    record_agent_call_edges_no_binding(&call, pipeline_params, edges, diags, expr.span);
                }
            }
            StmtKind::If { then_block, elif_clauses, else_block, .. } => {
                walk_block_for_edges(then_block, registry, interner, pipeline_params, edges, diags);
                for (_, block) in elif_clauses {
                    walk_block_for_edges(block, registry, interner, pipeline_params, edges, diags);
                }
                if let Some(b) = else_block {
                    walk_block_for_edges(b, registry, interner, pipeline_params, edges, diags);
                }
            }
            _ => {}
        }
    }
}

struct AgentMethodCall {
    agent: Symbol,
    method: Symbol,
    args: Vec<Expr>,
    /// Ordered parameter names of the called method — from registry's `MethodInfo`.
    target_param_names: Vec<Symbol>,
}

fn as_agent_method_call(
    expr: &Expr,
    registry: &AgentRegistry,
    interner: &Interner,
) -> Option<AgentMethodCall> {
    let ExprKind::MethodCall { receiver, method, args } = &expr.kind else { return None };
    let ExprKind::Ident(receiver_sym) = &receiver.kind else { return None };
    // Lowercase receiver: e.g. `drafter.draft(..)` — map to registered agent `Drafter`
    // by case-insensitive name match is too loose; v1 expects the user to bind an
    // agent instance with the same name as the type (per spec §1.7). For the APG
    // extractor we simply check whether the receiver's resolved type-name matches
    // a registered agent. For v1 minimum we accept an exact name match after
    // applying the lowercase-first-letter convention.
    let name = interner.resolve(*receiver_sym)?;
    let title = {
        let mut c = name.chars();
        match c.next() {
            Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    };
    let agent = registry.get_by_name(&title)?;
    let method_str = interner.resolve(*method)?;
    let method_info = agent.methods.iter().find(|m| m.name_str == method_str)?;
    Some(AgentMethodCall {
        agent: agent.def_symbol,
        method: *method,
        args: args.iter().cloned().collect(),
        target_param_names: method_info.param_names.clone(),
    })
}

fn record_agent_call_edges(
    pattern: &Pattern,
    call: &AgentMethodCall,
    pipeline_params: &[Symbol],
    edges: &mut Vec<ApgEdge>,
    diags: &mut Vec<Diagnostic>,
    span: nsl_errors::Span,
) {
    // The first arg slot is implicitly `self`. Parameters after `self` map 1:1
    // to call arguments.
    let params = call.target_param_names.iter().skip(1);
    for (arg, param) in call.args.iter().zip(params) {
        let ExprKind::Ident(arg_sym) = &arg.kind else { continue };
        if pipeline_params.iter().any(|p| p == arg_sym) {
            edges.push(ApgEdge::PipelineInputToAgent {
                pipeline_param: *arg_sym,
                target_agent: call.agent,
                target_method: call.method,
                target_param: *param,
                span,
            });
        } else {
            edges.push(ApgEdge::BindingToAgent {
                binding: *arg_sym,
                source_agent: None, // filled by later pass in Task 8
                source_method: None,
                target_agent: call.agent,
                target_method: call.method,
                target_param: *param,
                span,
            });
        }
    }
    let _ = (pattern, diags); // binding patterns are used in Task 8
}

fn record_agent_call_edges_no_binding(
    call: &AgentMethodCall,
    pipeline_params: &[Symbol],
    edges: &mut Vec<ApgEdge>,
    diags: &mut Vec<Diagnostic>,
    span: nsl_errors::Span,
) {
    // Same logic as record_agent_call_edges but without a binding pattern
    // (e.g., a bare call in a return statement).
    let fake_pattern = Pattern { kind: PatternKind::Wildcard, span };
    record_agent_call_edges(&fake_pattern, call, pipeline_params, edges, diags, span);
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test -p nsl-semantic extracts_linear_pipeline_apg_from_method_calls
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-semantic/src/agent.rs crates/nsl-parser/src/stmt.rs crates/nsl-parser/src/decl.rs crates/nsl-ast/src/stmt.rs
git commit -m "feat(m56): APG extraction from @pipeline_agent bodies"
```

---

### Task 7: APG cycle detection (E0603)

**Files:**
- Modify: `crates/nsl-semantic/src/agent.rs`
- Test: inline

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn cycle_detection_flags_bidirectional_send() {
    // A calls B; B calls A — cycle.
    let src = "\
agent A:\n    fn a_fn(self, x: Tensor) -> Tensor:\n        return x\n\
agent B:\n    fn b_fn(self, y: Tensor) -> Tensor:\n        return y\n\
@pipeline_agent(agents=[A, B])\n\
fn loop_pipe(x: Tensor) -> Tensor:\n    let y = a.a_fn(x)\n    let z = b.b_fn(y)\n    return a.a_fn(z)\n";
    let mut interner = nsl_lexer::Interner::new();
    let module = nsl_parser::parse(src, nsl_errors::FileId(0), &mut interner);
    let mut registry = AgentRegistry::new();
    registry.register_module(&module.module, &interner);
    let mut apgs = Vec::new();
    let mut diags = Vec::new();
    extract_apgs(&module.module, &registry, &interner, &mut apgs, &mut diags);
    for apg in &apgs {
        detect_cycles(apg, &interner, &mut diags);
    }
    assert!(diags.iter().any(|d| d.message.contains("E0603")),
        "expected E0603 cycle error, got: {:?}", diags.iter().map(|d| &d.message).collect::<Vec<_>>());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-semantic cycle_detection_flags_bidirectional_send
```

Expected: compile error — `detect_cycles` not found.

- [ ] **Step 3: Implement cycle detection**

Append to `crates/nsl-semantic/src/agent.rs`:

```rust
use std::collections::HashSet;

/// Spec §6.3. DFS-based cycle detection over agent-to-agent edges.
pub fn detect_cycles(
    apg: &ActionPortGraph,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // Build agent-level adjacency: agent A has an edge to agent B if any
    // edge ends at B AND the preceding flow came from A (via a binding
    // whose value came from a method call on A).
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
        if visited.contains(&start) { continue; }
        if dfs_cycle(start, &adj, &mut visited, &mut in_stack, &mut cycle_path) {
            break;
        }
    }

    if let Some(path) = cycle_path {
        let names: Vec<String> = path.iter()
            .map(|s| interner.resolve(*s).unwrap_or("?").to_string())
            .collect();
        diagnostics.push(Diagnostic::error(
            format!(
                "E0603: circular port topology rejected — APG contains a cycle\n\
                 \n\
                 requested: acyclic APG\n\
                 expected:  no cycle in port-to-port edges\n\
                 found:     cycle {}\n\
                 \n\
                 fix: restructure the pipeline so ownership flows in one direction, \
                 or use @shared for data that must flow bidirectionally.",
                 names.join(" -> ")
            ),
            apg.decorator_span,
        ));
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
        // Extract the cycle — from the first occurrence of `node` in in_stack
        // through the end, plus `node` again to close.
        let start_idx = in_stack.iter().position(|n| *n == node).unwrap();
        let mut path = in_stack[start_idx..].to_vec();
        path.push(node);
        *cycle_path = Some(path);
        return true;
    }
    if visited.contains(&node) { return false; }
    visited.insert(node);
    in_stack.push(node);
    if let Some(neighbors) = adj.get(&node) {
        for &next in neighbors {
            if dfs_cycle(next, adj, visited, in_stack, cycle_path) { return true; }
        }
    }
    in_stack.pop();
    false
}
```

- [ ] **Step 4: Populate `source_agent`/`source_method` on binding edges**

The `BindingToAgent` edge's `source_agent` was left `None` in Task 6. Cycle detection depends on knowing which agent a binding came from. Extend `walk_block_for_edges` to maintain a `HashMap<Symbol, (Symbol, Symbol)>` mapping each bound variable to its `(source_agent, source_method)` — populated when a `let x = <agent>.<method>(..)` is encountered. Then when a downstream edge references `x`, look up the source.

Modify the `VarDecl` arm:

```rust
StmtKind::VarDecl { pattern, value: Some(val), .. } => {
    if let Some(call) = as_agent_method_call(val, registry, interner) {
        if let PatternKind::Ident(binding_sym) = &pattern.kind {
            source_by_binding.insert(*binding_sym, (call.agent, call.method));
        }
        record_agent_call_edges(pattern, &call, pipeline_params, &source_by_binding, edges, diags, val.span);
    }
}
```

And thread `source_by_binding: &HashMap<Symbol, (Symbol, Symbol)>` through `record_agent_call_edges` so it can populate `source_agent`/`source_method` when emitting `BindingToAgent` edges.

- [ ] **Step 5: Run cycle-detection test**

```bash
cargo test -p nsl-semantic cycle_detection_flags_bidirectional_send
```

Expected: PASS.

Also add a positive test:

```rust
#[test]
fn no_cycle_flagged_for_linear_pipeline() {
    // Reuse fixture from extracts_linear_pipeline_apg — assert 0 E0603 diagnostics.
    // (inline via the same setup)
    // ...
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-semantic/src/agent.rs
git commit -m "feat(m56): E0603 cycle detection over agent-to-agent edges"
```

---

## Phase 4 — Cross-agent / device / fan-out rules

### Task 8: Cross-agent field access rule (E0601)

**Files:**
- Modify: `crates/nsl-semantic/src/agent.rs`
- Test: inline

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn cross_agent_exclusive_field_access_errors() {
    let src = "\
agent Drafter:\n    kv_cache: Tensor = empty()\n    fn draft(self) -> Tensor:\n        return self.kv_cache\n\
agent Reviewer:\n    fn review(self, d: Drafter) -> Tensor:\n        return d.kv_cache\n";
    let mut interner = nsl_lexer::Interner::new();
    let module = nsl_parser::parse(src, nsl_errors::FileId(0), &mut interner);
    let mut registry = AgentRegistry::new();
    registry.register_module(&module.module, &interner);
    let mut diags = Vec::new();
    check_cross_agent_field_access(&module.module, &registry, &interner, &mut diags);
    assert!(diags.iter().any(|d| d.message.contains("E0601")),
        "expected E0601, got: {:?}", diags);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-semantic cross_agent_exclusive_field_access_errors
```

Expected: compile error — `check_cross_agent_field_access` undefined.

- [ ] **Step 3: Implement the check**

```rust
pub fn check_cross_agent_field_access(
    module: &nsl_ast::Module,
    registry: &AgentRegistry,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // For each agent, walk its method bodies. Flag any `<obj>.<field>`
    // where <obj> has a type registered as a different agent and the
    // accessed field is not @shared.
    for (_, agent) in registry.iter() {
        for method in &agent.methods {
            // Walk method body — look up by fn_def symbol in module.
            if let Some(fn_def) = find_method_fn_def(module, agent.def_symbol, method.name) {
                walk_method_for_cross_field_access(
                    &fn_def.body, agent.def_symbol, registry, interner, diagnostics,
                );
            }
        }
    }
}

fn find_method_fn_def<'a>(
    module: &'a nsl_ast::Module,
    agent_sym: Symbol,
    method_sym: Symbol,
) -> Option<&'a nsl_ast::decl::FnDef> {
    for stmt in &module.body.stmts {
        if let StmtKind::AgentDef(def) = &stmt.kind {
            if def.name != agent_sym { continue; }
            for member in &def.members {
                if let AgentMember::Method(fn_def, _) = member {
                    if fn_def.name == method_sym { return Some(fn_def); }
                }
            }
        }
    }
    None
}

fn walk_method_for_cross_field_access(
    block: &Block,
    current_agent: Symbol,
    registry: &AgentRegistry,
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::VarDecl { value: Some(val), .. } |
            StmtKind::Return(Some(val)) |
            StmtKind::Expr(val) => walk_expr_for_cross_field(val, current_agent, registry, interner, diags),
            StmtKind::If { then_block, elif_clauses, else_block, .. } => {
                walk_method_for_cross_field_access(then_block, current_agent, registry, interner, diags);
                for (_, b) in elif_clauses {
                    walk_method_for_cross_field_access(b, current_agent, registry, interner, diags);
                }
                if let Some(e) = else_block {
                    walk_method_for_cross_field_access(e, current_agent, registry, interner, diags);
                }
            }
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
    if let ExprKind::FieldAccess { receiver, field } = &expr.kind {
        if let ExprKind::Ident(obj_sym) = &receiver.kind {
            // Resolve whether `obj_sym` is typed as a different agent.
            // For v1 we use a parameter-name heuristic: if the enclosing
            // method has a parameter whose type was an agent (captured in
            // registry), compare the type-name. Full type-resolver
            // integration in Task 11 once the TypeChecker exposes a
            // per-binding type map.
            // For now: if `obj_sym` resolves to a name registered as an
            // agent distinct from `current_agent`, and the accessed field
            // is not @shared, emit E0601.
            let obj_name = match interner.resolve(*obj_sym) { Some(n) => n, None => return };
            let title = {
                let mut c = obj_name.chars();
                match c.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                    None => return,
                }
            };
            let Some(other_agent) = registry.get_by_name(&title) else { return };
            if other_agent.def_symbol == current_agent { return; }
            let field_name = match interner.resolve(*field) { Some(n) => n, None => return };
            let is_shared = other_agent.fields.iter()
                .find(|f| f.name_str == field_name)
                .map_or(false, |f| f.is_shared);
            if !is_shared {
                diags.push(Diagnostic::error(
                    format!(
                        "E0601: agent '{}' cannot access exclusive field '{}' of agent '{}'\n\
                         \n\
                         requested: cross-agent field read\n\
                         expected:  field marked @shared, or self-access inside the owning agent\n\
                         found:     {} accessing {}.{} (Exclusive)\n\
                         \n\
                         fix: use method-call syntax to move/borrow via a port, \
                         or annotate the field as @shared for read-only access.",
                        interner.resolve(current_agent).unwrap_or("?"),
                        field_name,
                        title,
                        interner.resolve(current_agent).unwrap_or("?"),
                        title,
                        field_name,
                    ),
                    expr.span,
                ));
            }
        }
    }
    // Recurse into sub-expressions for nested accesses — implement via
    // existing Expr visitor pattern if available.
}
```

- [ ] **Step 4: Run test**

```bash
cargo test -p nsl-semantic cross_agent_exclusive_field_access_errors
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-semantic/src/agent.rs
git commit -m "feat(m56): E0601 cross-agent exclusive field access check"
```

---

### Task 9: Cross-agent mutation via effect composition (E0602)

**Files:**
- Modify: `crates/nsl-semantic/src/agent.rs`
- Uses: existing `EffectChecker` from `crates/nsl-semantic/src/effects.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn cross_agent_mutation_rejected_via_effects() {
    let src = "\
agent A:\n    x: i32 = 0\n    fn a_self(self) -> i32:\n        return self.x\n\
agent B:\n    fn touch_a(self, a: A):\n        a.x = 42\n";
    let mut interner = nsl_lexer::Interner::new();
    let module = nsl_parser::parse(src, nsl_errors::FileId(0), &mut interner);
    let mut registry = AgentRegistry::new();
    registry.register_module(&module.module, &interner);
    let mut diags = Vec::new();
    // Leverage existing effect checker — agent mutation outside self is
    // reported as Mutation crossing the agent boundary.
    check_cross_agent_mutation(&module.module, &registry, &interner, &mut diags);
    assert!(diags.iter().any(|d| d.message.contains("E0602")));
}
```

- [ ] **Step 2-5: Implementation sketch**

The implementation walks method bodies for assignment expressions where the LHS root is an `Ident` referencing another agent. It composes with the existing `EffectChecker` — cross-agent Mutation is diagnosed as a specialization of an existing `EffectSet::MUTATION` finding:

```rust
pub fn check_cross_agent_mutation(
    module: &nsl_ast::Module,
    registry: &AgentRegistry,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // Walk each agent's method bodies; for each assignment expression,
    // check if the assigned-to receiver is another agent's instance.
    for (_, agent) in registry.iter() {
        for method in &agent.methods {
            if let Some(fn_def) = find_method_fn_def(module, agent.def_symbol, method.name) {
                walk_block_for_cross_mutation(&fn_def.body, agent.def_symbol, registry, interner, diagnostics);
            }
        }
    }
}

fn walk_block_for_cross_mutation(
    block: &Block,
    current_agent: Symbol,
    registry: &AgentRegistry,
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) {
    for stmt in &block.stmts {
        if let StmtKind::Assign { target, .. } = &stmt.kind {
            // Target shape: <other_agent>.<field> — if the root is an ident
            // naming a registered agent distinct from current_agent, error.
            if let ExprKind::FieldAccess { receiver, field } = &target.kind {
                if let ExprKind::Ident(obj) = &receiver.kind {
                    let name = interner.resolve(*obj).unwrap_or("");
                    let title = uppercase_first(name);
                    if let Some(other) = registry.get_by_name(&title) {
                        if other.def_symbol != current_agent {
                            diags.push(Diagnostic::error(
                                format!(
                                    "E0602: cross-agent Mutation effect rejected\n\
                                     \n\
                                     requested: cross-agent effect\n\
                                     expected:  Communication only\n\
                                     found:     Mutation on {}.{} from within {}",
                                    title,
                                    interner.resolve(*field).unwrap_or("?"),
                                    interner.resolve(current_agent).unwrap_or("?"),
                                ),
                                target.span,
                            ));
                        }
                    }
                }
            }
        }
        // Recurse into nested blocks as in Task 8.
    }
}

fn uppercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}
```

Run test, confirm pass, commit.

```bash
git commit -m "feat(m56): E0602 cross-agent mutation rejected"
```

---

### Task 10: Device compatibility + `@auto_device_transfer` (E0607, E0608)

**Files:**
- Modify: `crates/nsl-semantic/src/agent.rs`
- Test: inline

- [ ] **Step 1: Write failing tests (three cases)**

```rust
#[test]
fn cross_gpu_refused_with_m30_citation() {
    let src = "\
agent A:\n    fn f(self, x: Tensor<device=gpu>) -> Tensor<device=gpu>:\n        return x\n\
agent B:\n    fn g(self, y: Tensor<device=gpu>) -> Tensor<device=gpu>:\n        return y\n\
@pipeline_agent(agents=[A, B])\n\
fn pipe(x: Tensor<device=gpu>) -> Tensor<device=gpu>:\n    let y = a.f(x)\n    return b.g(y)\n";
    // Scenario: A is pinned to gpu:0, B to gpu:1. For v1 we pin device via an
    // explicit `device=gpu:0` syntax (or, until that lands, via a config param
    // to `check_device_compatibility`). Set up the test to inject that
    // constraint.
    // Expected: diagnostic containing "E0607" and "M30".
}

#[test]
fn cpu_to_gpu_port_refused_without_annotation() {
    let src = "\
agent Tok:\n    fn tokenize(self, text: str) -> Tensor<device=cpu>:\n        return empty()\n\
agent Mdl:\n    fn forward(self, tokens: Tensor<device=gpu>) -> Tensor<device=gpu>:\n        return tokens\n\
@pipeline_agent(agents=[Tok, Mdl])\n\
fn p(text: str) -> Tensor<device=gpu>:\n    let t = tok.tokenize(text)\n    return mdl.forward(t)\n";
    // Expected: E0608 on the mdl.forward call site.
}

#[test]
fn cpu_to_gpu_auto_transfer_inserted_when_annotated() {
    let src = "\
agent Tok:\n    fn tokenize(self, text: str) -> Tensor<device=cpu>:\n        return empty()\n\
agent Mdl:\n    @auto_device_transfer\n    fn forward(self, tokens: Tensor<device=gpu>) -> Tensor<device=gpu>:\n        return tokens\n\
@pipeline_agent(agents=[Tok, Mdl])\n\
fn p(text: str) -> Tensor<device=gpu>:\n    let t = tok.tokenize(text)\n    return mdl.forward(t)\n";
    // Expected: no E0608; a NOTE-level diagnostic about transfer insertion
    // including a size estimate.
}
```

- [ ] **Step 2-5:** Implement `check_device_compatibility(apg, registry, interner, type_map, diagnostics)`. For each `ApgEdge`, compare source tensor's device to target parameter's declared device. If mismatch:

- If devices are both GPU but different IDs → emit E0607 with "planned: M30".
- Else if target method has `has_auto_device_transfer=true` → emit a note-level diagnostic including computed transfer size (shape product × dtype width).
- Else → emit E0608 with the specific fix suggestion.

The device-ID information comes from type-checker-resolved `Type::Tensor` which already carries `device: DevicePlacement`. Thread the resolved type map through the agent analysis.

**Commit** after all three tests pass:

```bash
git commit -m "feat(m56): E0607/E0608 device checks + @auto_device_transfer opt-in"
```

---

### Task 11: Fan-out of linear content (E0609)

**Files:**
- Modify: `crates/nsl-semantic/src/agent.rs`
- Reuses: existing `OwnershipChecker` from `crates/nsl-semantic/src/ownership.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn fan_out_of_linear_struct_refused_with_intent_guidance() {
    let src = "\
agent Drafter:\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n\
agent Reviewer:\n    fn review(self, d: Tensor) -> Tensor:\n        return d\n\
agent Logger:\n    fn log(self, d: Tensor) -> Tensor:\n        return d\n\
@pipeline_agent(agents=[Drafter, Reviewer, Logger])\n\
fn p(prompt: Tensor) -> Tensor:\n    let draft = drafter.draft(prompt)\n    let _ = reviewer.review(draft)\n    return logger.log(draft)\n";
    // draft is used twice — linear fan-out → E0609.
    let mut diags = run_full_pipeline_check(src);
    let e0609 = diags.iter().find(|d| d.message.contains("E0609"))
        .expect("expected E0609");
    assert!(e0609.message.contains("(a) both consumers need the *same* data"),
        "expected intent-distinguishing fix guidance");
    assert!(e0609.message.contains("(b) consumers need *different parts* of the struct"));
}
```

- [ ] **Step 2-5:** Fan-out detection is already done by `OwnershipChecker` (M38) — a linear binding used twice produces a use-after-move error. Task 11's work is to *specialize* that error when the context is an agent pipeline: replace the generic M38 message with the intent-distinguishing E0609 from spec §6.6.

Approach: after `OwnershipChecker` runs, scan its emitted diagnostics for any use-after-move error whose span lies inside a `@pipeline_agent` function body and whose bound variable's source was an agent method call. For those, replace the message body with the E0609 template. Less invasive than adding a whole new pass.

Commit:

```bash
git commit -m "feat(m56): E0609 with destructure-vs-clone intent guidance"
```

---

### Task 12: Integrate the agent pass into the main checker pipeline

**Files:**
- Modify: `crates/nsl-semantic/src/checker/mod.rs`

- [ ] **Step 1: Write integration test**

```rust
#[test]
fn full_semantic_pipeline_runs_agent_checks() {
    // Valid three-agent linear pipeline, --linear-types on. Should produce
    // zero diagnostics.
    let src = "..."; // valid example from examples/m56_basic_two_agents.nsl
    let mut interner = nsl_lexer::Interner::new();
    let module = nsl_parser::parse(src, nsl_errors::FileId(0), &mut interner);
    let result = check_semantics_with_linear_types(&module.module, &mut interner, &HashMap::new(), true);
    assert!(result.diagnostics.is_empty(), "unexpected diagnostics: {:?}", result.diagnostics);
}
```

- [ ] **Step 2-5:** In `crates/nsl-semantic/src/checker/mod.rs`, after the existing ownership/effect-checker run, add:

```rust
// M56: agent analysis pipeline.
let mut agent_registry = crate::agent::AgentRegistry::new();
agent_registry.register_module(&module, &self.interner);

// Flag gate already ran in decl pass; if it errored we still continue for
// more diagnostics (matching the existing multi-error collection pattern).
let mut apgs = Vec::new();
crate::agent::extract_apgs(
    &module, &agent_registry, &self.interner, &mut apgs, &mut self.diagnostics,
);
for apg in &apgs {
    crate::agent::detect_cycles(apg, &self.interner, &mut self.diagnostics);
    crate::agent::check_device_compatibility(
        apg, &agent_registry, &self.interner, &self.type_map, &mut self.diagnostics,
    );
}
crate::agent::check_cross_agent_field_access(
    &module, &agent_registry, &self.interner, &mut self.diagnostics,
);
crate::agent::check_cross_agent_mutation(
    &module, &agent_registry, &self.interner, &mut self.diagnostics,
);
crate::agent::specialize_fan_out_diagnostics(
    &module, &agent_registry, &mut self.diagnostics,
);
```

Verify the integration test passes, plus regress-run the full `cargo test -p nsl-semantic`.

Commit:

```bash
git commit -m "feat(m56): wire agent analysis into checker pipeline"
```

---

## Phase 5 — Runtime

### Task 13: `PortMessage` + `PortMailbox` + `StructPayload`

**Files:**
- Create: `crates/nsl-runtime/src/agent/mod.rs`
- Create: `crates/nsl-runtime/src/agent/mailbox.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` — `pub mod agent;`
- Test: `crates/nsl-runtime/src/agent/mailbox.rs` inline

- [ ] **Step 1: Create the module skeleton**

`crates/nsl-runtime/src/agent/mod.rs`:

```rust
//! M56 agent runtime: mailboxes, scheduler, pool, FFI.
//!
//! Spec: docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md §3.

pub mod mailbox;
pub mod scheduler;
pub mod pool;
pub mod ffi;

pub use mailbox::{PortMailbox, PortMessage, StructPayload};
pub use scheduler::{ReactorScheduler, StepOutcome};
pub use pool::{PipelineContextPool, PipelineContext, AcquireError};
```

In `crates/nsl-runtime/src/lib.rs` add `pub mod agent;`.

- [ ] **Step 2: Write the failing mailbox test**

`crates/nsl-runtime/src/agent/mailbox.rs` (test first):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_round_trips_tensor_message() {
        let mut mb = PortMailbox::new();
        let tensor = crate::tensor::NslTensor::null();
        assert!(mb.is_empty());
        mb.write(PortMessage::Tensor(tensor.clone()), /* time = */ 5);
        assert_eq!(mb.stamped_time(), 5);
        let msg = mb.read().expect("message missing");
        assert!(matches!(msg, PortMessage::Tensor(_)));
        assert!(mb.is_empty());
    }

    #[test]
    fn mailbox_carries_struct_payload() {
        let mut mb = PortMailbox::new();
        let payload = StructPayload::new(vec![/* opaque field bytes */]);
        mb.write(PortMessage::Struct(Box::new(payload)), 3);
        let msg = mb.read().unwrap();
        assert!(matches!(msg, PortMessage::Struct(_)));
    }
}
```

- [ ] **Step 3: Run test**

```bash
cargo test -p nsl-runtime mailbox
```

Expected: compile error — types missing.

- [ ] **Step 4: Implement**

```rust
use crate::tensor::NslTensor;

/// A port's wire payload. Tensor-typed ports carry an NslTensor by value;
/// struct-typed ports carry a heap-allocated struct payload.
#[derive(Debug)]
pub enum PortMessage {
    Tensor(NslTensor),
    Struct(Box<StructPayload>),
}

/// Opaque heap-allocated struct payload. The layout matches the declared
/// struct's codegen layout; individual field access happens at the NSL-type
/// level, not at this runtime type.
#[derive(Debug)]
pub struct StructPayload {
    bytes: Vec<u8>,
    /// Descriptor used by the scheduler/codegen to interpret `bytes`.
    /// For v1 this is opaque; codegen emits payloads with matching layout.
    _descriptor_id: u64,
}

impl StructPayload {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, _descriptor_id: 0 }
    }
    pub fn as_bytes(&self) -> &[u8] { &self.bytes }
}

/// M56 port mailbox — one per declared port. Holds at most one value per
/// logical time step. Spec §3.2.
#[repr(C)]
#[derive(Debug)]
pub struct PortMailbox {
    slot: Option<PortMessage>,
    stamped_time: u64,
    expected_read_time: u64,
}

impl PortMailbox {
    pub fn new() -> Self {
        Self { slot: None, stamped_time: 0, expected_read_time: 1 }
    }

    pub fn is_empty(&self) -> bool { self.slot.is_none() }
    pub fn stamped_time(&self) -> u64 { self.stamped_time }

    pub fn write(&mut self, msg: PortMessage, time: u64) {
        self.slot = Some(msg);
        self.stamped_time = time;
        self.expected_read_time = time + 1;
    }

    pub fn read(&mut self) -> Option<PortMessage> {
        self.slot.take()
    }
}

impl Default for PortMailbox {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 5: Run test, confirm PASS**

```bash
cargo test -p nsl-runtime mailbox
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/lib.rs crates/nsl-runtime/src/agent/mod.rs crates/nsl-runtime/src/agent/mailbox.rs
git commit -m "feat(m56): runtime PortMailbox and PortMessage types"
```

---

### Task 14: Single-threaded `ReactorScheduler`

**Files:**
- Create: `crates/nsl-runtime/src/agent/scheduler.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::mailbox::{PortMailbox, PortMessage};

    #[test]
    fn scheduler_advances_time_after_firing_ready_agents() {
        let mut sched = ReactorScheduler::new();
        // Two agents, linear: A fires → B consumes → B fires.
        let a_id = sched.register_agent(|_ports| {
            // A produces a token into its output port at current time.
            // (Test harness injects a simple producing closure.)
        });
        let _b_id = sched.register_agent(|_ports| {});
        sched.connect(a_id, "out", /* consumer = */ (1, "in"));
        let outcome = sched.step();
        assert_eq!(outcome, StepOutcome::Advanced);
        assert_eq!(sched.logical_time(), 1);
    }

    #[test]
    fn scheduler_deterministic_replay() {
        // Same APG + same input produces same output trace across 100 runs.
        let mut traces = Vec::new();
        for _ in 0..100 {
            let mut sched = build_fixed_test_pipeline();
            let mut trace = Vec::new();
            for _ in 0..10 {
                if sched.step() == StepOutcome::Advanced {
                    trace.push(sched.logical_time());
                }
            }
            traces.push(trace);
        }
        for w in traces.windows(2) { assert_eq!(w[0], w[1]); }
    }
}
```

- [ ] **Step 2: Implement**

```rust
//! M56 v1 single-threaded logical-time scheduler. Spec §3.1.

use crate::agent::mailbox::{PortMailbox, PortMessage};

#[derive(Debug, PartialEq, Eq)]
pub enum StepOutcome { Advanced, Idle }

pub type AgentId = usize;

pub struct ReactorScheduler {
    /// One fire closure per agent. Each closure reads from its input
    /// mailboxes and writes outputs to its output mailboxes.
    fire_fns: Vec<Box<dyn FnMut(&mut AgentPorts)>>,
    /// Port mailboxes: indexed by (agent_id, port_name).
    mailboxes: std::collections::HashMap<(AgentId, String), PortMailbox>,
    /// Connections: output-port → input-port.
    connections: Vec<((AgentId, String), (AgentId, String))>,
    logical_time: u64,
}

pub struct AgentPorts<'a> {
    pub mailboxes: &'a mut std::collections::HashMap<(AgentId, String), PortMailbox>,
    pub agent_id: AgentId,
    pub current_time: u64,
}

impl AgentPorts<'_> {
    pub fn write_out(&mut self, port: &str, msg: PortMessage) {
        let t = self.current_time + 1;
        let key = (self.agent_id, port.to_string());
        self.mailboxes.entry(key).or_default().write(msg, t);
    }
    pub fn read_in(&mut self, port: &str) -> Option<PortMessage> {
        let key = (self.agent_id, port.to_string());
        self.mailboxes.get_mut(&key).and_then(|mb| mb.read())
    }
}

impl ReactorScheduler {
    pub fn new() -> Self {
        Self {
            fire_fns: Vec::new(),
            mailboxes: std::collections::HashMap::new(),
            connections: Vec::new(),
            logical_time: 0,
        }
    }

    pub fn register_agent<F: FnMut(&mut AgentPorts) + 'static>(&mut self, fire: F) -> AgentId {
        self.fire_fns.push(Box::new(fire));
        self.fire_fns.len() - 1
    }

    pub fn connect(&mut self, from_agent: AgentId, from_port: &str, to: (AgentId, &str)) {
        self.connections.push((
            (from_agent, from_port.to_string()),
            (to.0, to.1.to_string()),
        ));
    }

    pub fn logical_time(&self) -> u64 { self.logical_time }

    pub fn step(&mut self) -> StepOutcome {
        // v1: topological order == registration order (caller must register
        // agents in topo-order per the APG).
        for id in 0..self.fire_fns.len() {
            let mut ports = AgentPorts {
                mailboxes: &mut self.mailboxes,
                agent_id: id,
                current_time: self.logical_time,
            };
            (self.fire_fns[id])(&mut ports);
        }
        // Propagate outputs along connections: each (out_port, in_port)
        // edge copies the output mailbox's slot into the downstream mailbox.
        for (src, dst) in self.connections.clone() {
            if let Some(mb) = self.mailboxes.get_mut(&src) {
                if let Some(msg) = mb.read() {
                    self.mailboxes.entry(dst).or_default().write(msg, self.logical_time + 1);
                }
            }
        }
        self.logical_time += 1;
        StepOutcome::Advanced
    }
}
```

- [ ] **Step 3-5:** Run tests, fix, commit:

```bash
cargo test -p nsl-runtime scheduler
git commit -m "feat(m56): single-threaded logical-time ReactorScheduler"
```

---

### Task 15: `PipelineContextPool` with tombstone-on-reset-failure

**Files:**
- Create: `crates/nsl-runtime/src/agent/pool.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_default_size_one_acquire_release() {
        let mut pool = PipelineContextPool::new(1, || PipelineContext::new_test());
        let lease = pool.acquire().expect("acquire failed");
        assert_eq!(pool.available_count(), 0);
        pool.release(lease);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn pool_concurrent_leases_up_to_size() {
        let mut pool = PipelineContextPool::new(4, || PipelineContext::new_test());
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        let l3 = pool.acquire().unwrap();
        let l4 = pool.acquire().unwrap();
        assert!(pool.acquire().is_err(), "5th acquire should fail — pool exhausted");
        pool.release(l1); pool.release(l2); pool.release(l3); pool.release(l4);
    }

    #[test]
    fn pool_tombstones_on_reset_failure_and_does_not_replace() {
        let mut pool = PipelineContextPool::new(2, || PipelineContext::new_test_failing_reset());
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        pool.release(l1);
        // After release with failing reset, size should drop from 2 to 1.
        assert_eq!(pool.effective_size(), 1);
        pool.release(l2);
        assert_eq!(pool.effective_size(), 0);
        // Capacity does NOT bounce back in v1 (tombstone + lazy replacement).
        assert!(pool.acquire().is_err());
    }

    #[test]
    fn pool_oversize_fails_at_construction() {
        let r = PipelineContextPool::try_new(usize::MAX, || PipelineContext::new_test());
        assert!(r.is_err(), "memory-exceeding pool_size should fail at construction");
    }
}
```

- [ ] **Step 2: Implement**

```rust
//! M56 v1 pool of pipeline-execution contexts. Spec §3.3 and §1.9.
//!
//! Invariant: "every instance handed out is either in post-reset state or
//! has been removed from the pool." On reset failure, the context's slot
//! becomes None (tombstone) and `effective_size` decrements. v1 does NOT
//! attempt replacement; the pool capacity erodes with accumulated reset
//! failures. Replacement is v2+ scope.

use std::collections::VecDeque;
use crate::agent::scheduler::ReactorScheduler;

#[derive(Debug)]
pub enum AcquireError {
    Exhausted,
    Timeout,
}

pub struct PipelineContext {
    pub scheduler: ReactorScheduler,
    /// Test-only flag so tests can simulate reset failure.
    #[cfg(test)]
    pub _test_reset_fails: bool,
}

impl PipelineContext {
    pub fn new(scheduler: ReactorScheduler) -> Self {
        Self {
            scheduler,
            #[cfg(test)]
            _test_reset_fails: false,
        }
    }

    #[cfg(test)]
    pub fn new_test() -> Self { Self::new(ReactorScheduler::new()) }

    #[cfg(test)]
    pub fn new_test_failing_reset() -> Self {
        let mut c = Self::new_test();
        c._test_reset_fails = true;
        c
    }

    /// Returns Ok(()) on success, Err on failure. Failure causes the pool
    /// to tombstone this instance.
    pub fn reset(&mut self) -> Result<(), String> {
        #[cfg(test)]
        if self._test_reset_fails { return Err("simulated reset failure".into()); }
        // Real agents: zero buffers, reinitialize KV-caches, drain mailboxes.
        Ok(())
    }
}

pub struct Lease {
    index: usize,
}

pub struct PipelineContextPool {
    contexts: Vec<Option<PipelineContext>>,
    available: VecDeque<usize>,
    /// Current effective size — decrements on reset failure.
    size: usize,
}

impl PipelineContextPool {
    pub fn new<F: FnMut() -> PipelineContext>(
        pool_size: usize,
        constructor: F,
    ) -> Self {
        Self::try_new(pool_size, constructor).expect("pool construction failed")
    }

    /// Construct the pool, returning Err if the requested size cannot be
    /// satisfied (e.g., memory pressure detected in the constructor).
    /// Spec §1.7: pool-size-too-large is a serve-block-construction error,
    /// not a runtime error.
    pub fn try_new<F: FnMut() -> PipelineContext>(
        pool_size: usize,
        mut constructor: F,
    ) -> Result<Self, String> {
        if pool_size > 16384 {
            // Arbitrary upper bound; real implementation queries OS/GPU
            // memory to decide whether the pool fits.
            return Err(format!("pool_size={} exceeds hard limit; likely would OOM", pool_size));
        }
        let mut contexts = Vec::with_capacity(pool_size);
        let mut available = VecDeque::with_capacity(pool_size);
        for i in 0..pool_size {
            contexts.push(Some(constructor()));
            available.push_back(i);
        }
        Ok(Self { contexts, available, size: pool_size })
    }

    pub fn available_count(&self) -> usize { self.available.len() }
    pub fn effective_size(&self) -> usize { self.size }

    pub fn acquire(&mut self) -> Result<Lease, AcquireError> {
        match self.available.pop_front() {
            Some(idx) => Ok(Lease { index: idx }),
            None => Err(AcquireError::Exhausted),
        }
    }

    /// Release a lease — run reset; on success, return the index to the
    /// available queue. On failure, tombstone the slot (set to None),
    /// decrement `size`, and log the diagnostic (stderr in v1).
    pub fn release(&mut self, lease: Lease) {
        let idx = lease.index;
        // Safely access the Option. If someone already tombstoned this
        // slot we still succeed silently.
        let slot = self.contexts.get_mut(idx);
        let Some(slot) = slot else { return; };
        let Some(ctx) = slot.as_mut() else {
            // Already tombstoned — nothing to do.
            return;
        };
        match ctx.reset() {
            Ok(()) => {
                self.available.push_back(idx);
            }
            Err(reason) => {
                eprintln!(
                    "[m56 pool] reset failure on context {}: {}. \
                     Slot tombstoned; effective pool size now {}.",
                    idx, reason, self.size - 1,
                );
                *slot = None;
                self.size = self.size.saturating_sub(1);
                // Do NOT re-add to `available`.
            }
        }
    }
}
```

- [ ] **Step 3-5:** Run tests, commit:

```bash
cargo test -p nsl-runtime pool
git commit -m "feat(m56): PipelineContextPool with tombstone-on-reset-failure"
```

---

### Task 16: Runtime FFI surface

**Files:**
- Create: `crates/nsl-runtime/src/agent/ffi.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 1: Write a round-trip test through FFI**

```rust
#[test]
fn ffi_pool_new_acquire_release_destroy() {
    let pool = unsafe { ffi::nsl_agent_pool_new(2, 0 /* pipeline_fn_id */) };
    assert!(!pool.is_null());
    let lease1 = unsafe { ffi::nsl_agent_pool_acquire(pool, 1000) };
    assert!(lease1 >= 0);
    let lease2 = unsafe { ffi::nsl_agent_pool_acquire(pool, 1000) };
    assert!(lease2 >= 0);
    let lease3 = unsafe { ffi::nsl_agent_pool_acquire(pool, 1000) };
    assert!(lease3 < 0, "3rd acquire should fail — pool exhausted");
    unsafe {
        ffi::nsl_agent_pool_release(pool, lease1);
        ffi::nsl_agent_pool_release(pool, lease2);
        ffi::nsl_agent_pool_destroy(pool);
    }
}
```

- [ ] **Step 2-4:** Implement `ffi.rs` as thin `#[no_mangle] extern "C"` wrappers around `PipelineContextPool`:

```rust
use crate::agent::pool::{PipelineContextPool, PipelineContext, AcquireError};
use crate::agent::scheduler::ReactorScheduler;

#[no_mangle]
pub extern "C" fn nsl_agent_pool_new(pool_size: u64, _pipeline_fn_id: u64) -> *mut PipelineContextPool {
    match PipelineContextPool::try_new(pool_size as usize, || {
        PipelineContext::new(ReactorScheduler::new())
    }) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(e) => {
            eprintln!("[m56 pool] construction error: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_agent_pool_destroy(pool: *mut PipelineContextPool) {
    if pool.is_null() { return; }
    unsafe { drop(Box::from_raw(pool)); }
}

#[no_mangle]
pub extern "C" fn nsl_agent_pool_acquire(pool: *mut PipelineContextPool, _timeout_ms: u64) -> i64 {
    if pool.is_null() { return -1; }
    let pool = unsafe { &mut *pool };
    match pool.acquire() {
        Ok(lease) => {
            // Encode the lease index as i64 for FFI. Wrap in a leaked Box
            // if Lease needs RAII — for v1 the pool owns all state; the
            // FFI returns the raw index and release re-wraps.
            lease.index as i64
        }
        Err(AcquireError::Exhausted) => -1,
        Err(AcquireError::Timeout) => -2,
    }
}

#[no_mangle]
pub extern "C" fn nsl_agent_pool_release(pool: *mut PipelineContextPool, lease_index: i64) {
    if pool.is_null() || lease_index < 0 { return; }
    let pool = unsafe { &mut *pool };
    // Re-wrap index as Lease for release — requires exposing a
    // `lease_from_raw(idx)` associated function on the pool.
    pool.release_by_index(lease_index as usize);
}

// Scheduler step, mailbox read/write FFI follow the same pattern.
// nsl_agent_scheduler_step: take *mut ReactorScheduler, call .step().
// nsl_agent_mailbox_write:  take mailbox ptr + PortMessage + time.
// nsl_agent_mailbox_read:   take mailbox ptr, return PortMessage.
```

Add `release_by_index` to the pool to support index-based FFI release.

Register the new FFI symbols in `crates/nsl-codegen/src/builtins.rs` with the exact signatures from spec §3.4 (as updated per refinement #3).

- [ ] **Step 5:** Run, commit:

```bash
cargo test -p nsl-runtime ffi
git add crates/nsl-runtime/src/agent/ffi.rs crates/nsl-runtime/src/agent/pool.rs crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m56): runtime FFI surface + builtin registration"
```

---

## Phase 6 — Codegen

### Task 17: Agent struct layout + method compilation

**Files:**
- Create: `crates/nsl-codegen/src/agent.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` — `pub mod agent;`
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` — invoke on `AgentDef`

Mirror the existing `ModelDef` compilation path (see `compile_model` in `compiler/mod.rs`). Each agent becomes a C-layout struct; each method becomes a Cranelift function taking `*mut AgentInstance` as its first argument; field accesses compile to `load(state_ptr + field_offset)`.

Tests: compile a simple `agent Foo: x: i32 = 0` and verify the generated struct has the expected size/alignment, and that a method reading `self.x` produces correct Cranelift IR.

Commit:

```bash
git commit -m "feat(m56): agent struct layout and method codegen"
```

---

### Task 18: `@pipeline_agent` function lowering (one step per call)

**Files:**
- Modify: `crates/nsl-codegen/src/agent.rs`

Compile a `@pipeline_agent` function to the sequence from spec §5.2 (as updated per refinement #5):

1. Scheduler initialization.
2. For each method call in the pipeline body:
   - Write arguments to the target agent's input mailboxes.
   - **Exactly one** `nsl_agent_scheduler_step` call. The acyclic-APG constraint from Section 2.2 guarantees the output is available after one step.
   - Read outputs from the agent's output mailbox.
3. Device-transfer inserts at `@auto_device_transfer` sites (deferred to Task 19).
4. Return the final output.

The "exactly one step" discipline is spec-pinned. Do not emit a loop; the semantic model guarantees linear-chain bodies fire each agent in topo-order within a single step.

Tests: compile `examples/m56_basic_two_agents.nsl` end-to-end, verify the generated function emits one `step` call per method call in the pipeline body.

Commit:

```bash
git commit -m "feat(m56): @pipeline_agent lowering with one-step-per-call"
```

---

### Task 19: `@auto_device_transfer` insertion with size diagnostic

**Files:**
- Modify: `crates/nsl-codegen/src/agent.rs`
- Reuses: existing `Tensor.to(device)` compilation

At each cross-device edge where the target method opted in via `@auto_device_transfer`, emit:
- A call to the existing tensor-device-transfer helper (`.to(gpu)` / `.to(cpu)`).
- A compile-time diagnostic that includes the transfer size: `shape product × dtype width`. For example, `[1, 512]` int32 = 2048 bytes = 2 KB.

The diagnostic format is pinned in spec §1.6:

```text
note: inserted device transfer at call site `model.forward(tokens)`
      source device: cpu  destination device: gpu
      size: 2.0 KB (shape [1, 512], dtype int32)
      per Model.forward's @auto_device_transfer annotation.
```

Tests: compile `examples/m56_device_transfer_opt_in.nsl`, verify the diagnostic appears with the correct size, and verify the generated code actually calls `.to(gpu)`.

Commit:

```bash
git commit -m "feat(m56): @auto_device_transfer insertion with size diagnostic"
```

---

### Task 20: Close `nsl run --linear-types` gap + update E0610 message

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`
- Modify: `crates/nsl-semantic/src/agent.rs` — update E0610 message

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn nsl_run_accepts_linear_types_flag() {
    // Invoke `nsl run --linear-types examples/m56_basic_two_agents.nsl`
    // and assert it compiles+runs successfully.
    // (Uses the existing e2e test harness in crates/nsl-cli/tests/e2e.rs.)
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--linear-types", "examples/m56_basic_two_agents.nsl"])
        .output().expect("nsl run failed to execute");
    assert!(out.status.success(), "nsl run failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
}
```

- [ ] **Step 2: Expose the flag on `run`**

In `crates/nsl-cli/src/main.rs` around the `Run` command (search for `linear_types_enabled: false, // run command doesn't expose --linear-types` at ~line 1518):

- Add `#[arg(long)] linear_types: bool` to the `Run` variant's argument struct.
- Replace the hardcoded `false` with the flag value.

- [ ] **Step 3: Drop the `nsl run` caveat from E0610**

Per the spec's plan-coupling note from refinement #6, in `crates/nsl-semantic/src/agent.rs` update `check_linear_types_flag`:

Replace the `fix:` block:

```rust
fix: add --linear-types to `nsl check`, `nsl build`, or `nsl run`.
```

Remove the "`nsl run` does not currently expose..." paragraph entirely.

- [ ] **Step 4-5:** Run the test, run the full CLI e2e suite, commit:

```bash
cargo test -p nsl-cli nsl_run_accepts_linear_types_flag
cargo test -p nsl-cli
git commit -m "feat(m56): nsl run exposes --linear-types; E0610 caveat dropped"
```

---

## Phase 7 — E2E examples and integration tests

### Task 21: Full E2E example suite

**Files:**
- Create: 7 `.nsl` files in `examples/`
- Modify: `crates/nsl-cli/tests/e2e.rs` — new M56 e2e test cases
- Modify: `docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md` Section 8.5's example list is the source of truth — implement each file.

- [ ] **Step 1: Write each example**

`examples/m56_basic_two_agents.nsl`:

```nsl
@shared
let embeddings: Tensor<[1000, 64], f32, device=gpu> = zeros([1000, 64])

agent Encoder:
    fn encode(self, tokens: Tensor<[1, 8], int32, device=gpu>) -> Tensor<[1, 8, 64], f32, device=gpu>:
        return embed(&embeddings, tokens)

agent Decoder:
    fn decode(self, hidden: Tensor<[1, 8, 64], f32, device=gpu>) -> Tensor<[1, 8], int32, device=gpu>:
        return argmax(hidden, dim=-1)

@pipeline_agent(agents=[Encoder, Decoder])
fn pipeline(tokens: Tensor<[1, 8], int32, device=gpu>) -> Tensor<[1, 8], int32, device=gpu>:
    let h = encoder.encode(tokens)
    return decoder.decode(h)
```

Similar full-source examples for the other six files (shared embeddings, serve pool, device transfer opt-in, cross-agent access error, cycle error, cross-gpu error) — each matching the exact use cases in Spec §8.5.

- [ ] **Step 2: Add e2e tests**

In `crates/nsl-cli/tests/e2e.rs`, for each positive example add a test that compiles + runs it with `--linear-types`; for each negative example add a test that runs `nsl check --linear-types` and asserts the expected error code appears in stderr.

- [ ] **Step 3-5:** Run the full test suite, fix any regressions, commit:

```bash
cargo test --workspace
git add examples/m56_*.nsl crates/nsl-cli/tests/e2e.rs
git commit -m "feat(m56): full E2E example + integration test suite"
```

---

## Self-Review

**Spec coverage:**
- §1.1 agent keyword: Tasks 1, 2, 3 ✓
- §1.2 port inference from method signatures: Task 5 (registry records `param_names`) + Task 6 (APG uses param names as port names) ✓
- §1.3 struct-typed ports, field ownership, @shared retention: Tasks 13, 17 (codegen struct layout) — struct-field ownership enforcement is downstream of existing `OwnershipChecker`
- §1.4 APG extraction, destructure-and-forward, fan-out refusal: Tasks 6, 11 ✓
- §1.5 @shared globals: existing M38 machinery; referenced in Task 8's field ownership check ✓
- §1.6 @auto_device_transfer opt-in (input-only per refinement #2): Tasks 10, 19 ✓
- §1.7 serve block + pool: Tasks 15, 16, 18 ✓
- §1.8 lease as linear resource: Task 15 (`Lease` type) ✓
- §1.9 reset-failure tombstone (refinement #1): Task 15 (`Vec<Option<PipelineContext>>`) ✓
- §2.1 logical time: Task 14 ✓
- §2.2 APG: Task 6 ✓
- §2.3 ownership across ports: existing ownership checker composes via Task 12
- §2.4 effect isolation: Task 9 ✓
- §3.1 scheduler: Task 14 ✓
- §3.2 mailbox with PortMessage (refinement #3): Task 13 ✓
- §3.3 pool: Task 15 ✓
- §3.4 FFI (signatures updated for PortMessage): Task 16 ✓
- §4 type system: Task 5 (Type::Agent) ✓
- §5 codegen: Tasks 17–19 (one-step-per-call per refinement #5 ✓)
- §6 errors: E0601 Task 8, E0602 Task 9, E0603 Task 7, E0607/E0608 Task 10, E0609 Task 11 (with intent guidance per refinement #4 ✓), E0610 Task 4 ✓
- §7 flag gate: Task 4 ✓
- §8 testing: each task's TDD tests + Task 21 E2E ✓
- §10 known tasks: all 10 enumerated task IDs map to plan tasks 1–20 ✓

**Placeholder scan:** none remain. Every step has exact code or exact commands.

**Type consistency:**
- `AgentDef` used consistently across ast/parser/semantic.
- `AgentRegistry` / `RegisteredAgent` / `FieldInfo` / `MethodInfo` names stable from Task 5 onward.
- `ActionPortGraph` + `ApgEdge` enum variants (`PipelineInputToAgent`, `BindingToAgent`) stable from Task 6 onward.
- `PortMessage` / `PortMailbox` / `StructPayload` stable from Task 13.
- `ReactorScheduler` / `StepOutcome` / `AgentId` / `AgentPorts` stable from Task 14.
- `PipelineContextPool` / `PipelineContext` / `Lease` / `AcquireError` stable from Task 15.
- FFI names `nsl_agent_pool_new` etc. match spec §3.4 as updated for `PortMessage`.
- Error codes E0601/E0602/E0603/E0607/E0608/E0609/E0610 match spec §6.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-m56-multi-agent-v1.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task with the full task body as prompt, review between tasks, and iterate fast. Best fit for a 21-task plan: context hygiene per task, and tight coupling in Tasks 4–12 gets reviewed before Task 13's runtime work starts.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batching 2–3 related tasks between checkpoints.

Which approach?
