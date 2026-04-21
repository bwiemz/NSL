# CPDT AST Auto-Detect + Four-Case Decision Table — Design

> **Framing:** Final Phase 1 Tier A follow-up. Adds AST auto-detection of `load_safetensors(...)` calls plus the four-case decision-table logic from the Phase 1 spec §2.1 to the `nsl build` command. Complements PR #90 (validate-body) and PR #91 (weight_aware opt-out) to complete the Phase 1 user-facing surface.

**Parent specs:**
- [2026-04-18-cpdt-weight-aware-phase1-design.md](2026-04-18-cpdt-weight-aware-phase1-design.md) §2.1 (four-case decision table)
- [2026-04-20-cpdt-validate-body-design.md](2026-04-20-cpdt-validate-body-design.md) (PR #90)
- [2026-04-20-cpdt-weight-aware-opt-out-design.md](2026-04-20-cpdt-weight-aware-opt-out-design.md) (PR #91)

**Worktree:** `.worktrees/cpdt-ast-autodetect` (branch `feat/cpdt-ast-autodetect`, based on `main` at 9b356ec9)

---

## 1. Motivation

Today users running `nsl build --cpdt full --cpdt-num-gpus 4 fixture.nsl` must also pass `--weights <path>` explicitly — even when their NSL source already declares `let w = load_safetensors("weights.safetensors")`. The information is duplicated across the source file and the CLI invocation; the CLI ignores the source-declared path.

Worse, when users forget `--weights` entirely, the build succeeds silently with a CPDT plan that has zero weight-derived tier assignments (because `cpdt::run`'s `weights: None` branch returns `PrecisionPlan::default()`). The tier-agreement diagnostic doesn't fire (weights_present is false). From the user's perspective, `--cpdt full` ran and produced nothing — no error, no warning, just a silent no-op. That's the exact failure mode the Phase 1 spec §2.1 four-case decision table was designed to eliminate.

## 2. Scope & Non-Goals

**In scope:**

- **AST walker** for `load_safetensors(<string_literal>)` calls in the module. Returns the first-found string literal as a `PathBuf`. The search skips nested contexts that could never execute (dead code inside an unreached branch); the walker is structural.
- **AST walker** for `@cpdt(..., weight_aware=<bool_literal>)` decorator. Returns the bool if found. Needed for Phase 1's four-case decision table to know whether to error on absent weights.
- **Four-case decision table** in `nsl-cli/src/main.rs` Build handler:
  1. AST-detected weights + no `--weights` flag: use AST-detected.
  2. No AST + `--weights` flag: use flag (current behavior).
  3. Both AST-detected and `--weights` flag: use flag + emit warning.
  4. Neither present + CPDT-enabled + `weight_aware=true` (default): emit **error** with three-option resolution message; exit 1.
- 5 CLI integration tests covering the four cases plus the `weight_aware=false` exemption.

**Out of scope:**

- Walking for `load_safetensors(<non-string-literal>)` calls (e.g. `load_safetensors(PATH_CONSTANT)`). Phase 1 treats only string-literal paths as "AST-detectable." Non-literal paths are unhandled; the user passes `--weights` explicitly for those.
- Multiple `load_safetensors(...)` calls in the same source. Phase 1 uses the first found. Multi-model scenarios are a separate concern.
- Changes to the non-CPDT paths (`--weight-analysis`, `--standalone`, etc.) that also read `--weights`. Their behavior is unchanged; the AST auto-detect flows only through the CPDT wiring.

## 3. Design

### 3.1 AST walkers

Both walkers live as free functions in a new module `crates/nsl-cli/src/ast_scan.rs` rather than in nsl-ast or nsl-codegen — they're CLI-specific resolution logic, not compiler state.

```rust
//! AST-scan helpers for the nsl-cli `build` command. Walks a parsed
//! `Module` to detect `load_safetensors(...)` calls and the
//! `@cpdt(weight_aware=...)` decorator so the CLI can apply the Phase 1
//! four-case decision table without compiling the full program.

use std::path::PathBuf;
use nsl_ast::{Module, Symbol};
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::expr::{Expr, ExprKind, Arg};
use nsl_lexer::Interner;

/// Find the first `load_safetensors(<string_literal>, ...)` call in
/// `module` and return its path argument. Returns `None` when no such
/// call exists, when the first arg isn't a string literal, or when
/// `load_safetensors` is shadowed (not resolvable as a builtin).
pub fn find_ast_weight_ref(module: &Module, interner: &Interner) -> Option<PathBuf> { ... }

/// Find `@cpdt(..., weight_aware=<bool>)` and return the bool. Returns
/// `None` when no `@cpdt` decorator exists or when the kwarg is absent.
/// The default value of `weight_aware` per the semantic pass is `true`,
/// so the CLI treats `None` as "weight_aware is true" for the four-case
/// decision table.
pub fn find_ast_cpdt_weight_aware(module: &Module, interner: &Interner) -> Option<bool> { ... }
```

Implementation: recursive walk over `module.stmts` via a private helper. Each walker short-circuits on first match (we don't need to enumerate all matches). The walkers do not need to resolve types, scopes, or imports — they operate on the raw AST.

### 3.2 Four-case decision table in the CLI

In `nsl-cli/src/main.rs`'s `Cli::Build` handler, after the source is parsed (around line 893 where `training_report` already parses) and before `compile_opts` is constructed (around line 1122):

```rust
// Parse once for AST scanning (the main compile path re-parses internally;
// this extra parse is cheap — a few MB/sec on the size fixtures Phase 1 ships).
let ast_source = std::fs::read_to_string(&file).unwrap_or_default();
let mut ast_interner = Interner::new();
let ast_file_id = FileId(0);
let (ast_tokens, _) = nsl_lexer::tokenize(&ast_source, ast_file_id, &mut ast_interner);
let ast_parse = nsl_parser::parse(&ast_tokens, &mut ast_interner);

let ast_weight_ref = ast_scan::find_ast_weight_ref(&ast_parse.module, &ast_interner);
let ast_weight_aware = ast_scan::find_ast_cpdt_weight_aware(&ast_parse.module, &ast_interner);

let resolved_weight_file: Option<PathBuf> = match (&weights, &ast_weight_ref) {
    (Some(flag_path), Some(ast_path)) => {
        eprintln!(
            "warning: --weights {} overrides AST-declared load_safetensors({:?}).",
            flag_path.display(), ast_path.display(),
        );
        Some(flag_path.clone())
    }
    (Some(flag_path), None) => Some(flag_path.clone()),
    (None, Some(ast_path)) => Some(ast_path.clone()),
    (None, None) => {
        let cpdt_enabled = cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off;
        let weight_aware = ast_weight_aware.unwrap_or(true);
        if cpdt_enabled && weight_aware && !standalone {
            eprintln!(
                "error: --cpdt {} requires weights. Resolve by ONE of:\n\
                 \n\
                 1. Add --weights <path.safetensors> to this invocation.\n\
                 2. Add `let w = load_safetensors(\"<path>\")` to your NSL source.\n\
                 3. Add `@cpdt(weight_aware=false)` to opt out of the weight-aware path\n\
                    (produces a CPDT plan without weight-derived tier assignments).",
                cpdt_mode.as_str(),
            );
            process::exit(1);
        }
        None
    }
};

// Use resolved_weight_file downstream instead of `weights`.
```

The `weight_file: if standalone { None } else { weights.clone() }` line in the `compile_opts` construction becomes `weight_file: if standalone { None } else { resolved_weight_file.clone() }`.

### 3.3 Error scope — CPDT only

The decision-table error fires only when CPDT is enabled. Other consumers of `--weights` (`--weight-analysis`, `--standalone`) already have their own error paths (see existing `--weight-analysis requires --weights` check at line 885). The new error is scoped to the CPDT-specific "silent no-op" mode.

### 3.4 The `--standalone` exception

`--standalone` currently forces `weight_file: None` regardless of `--weights` (the standalone pipeline bundles weights differently). AST auto-detect respects this: when `--standalone` is set, the decision table is bypassed for the error branch specifically. Flag-wins behavior still applies.

## 4. Tests

File: `crates/nsl-cli/tests/cpdt_weights_cli.rs`. Five tests using the existing `assert_cmd` + `tempfile` pattern from `cpdt_cli.rs`. Each test writes a minimal NSL source + (when needed) a tiny 4-byte safetensors file.

1. **`ast_autodetect_only_uses_source_path`**: Source has `load_safetensors("weights.safetensors")`; no `--weights` flag. Compile succeeds; stderr does NOT contain the four-case error. (We don't assert the tier-agreement diagnostic fires because that requires a full compile with AppliedPlan generation; limit the assertion to "no error.")

2. **`flag_only_uses_flag_path`**: Source has no `load_safetensors`; `--weights <path>` set. Compile succeeds; stderr does NOT contain the override warning.

3. **`both_present_flag_wins_with_warning`**: Source has `load_safetensors("ast.safetensors")`; `--weights flag.safetensors` set. Compile succeeds; stderr contains "overrides AST-declared" warning.

4. **`neither_present_cpdt_full_errors_with_three_options`**: Source has no `load_safetensors` and no `@cpdt(weight_aware=false)`; no `--weights` flag; `--cpdt full --cpdt-num-gpus 4` set. Compile fails (non-zero exit); stderr contains "requires weights" AND the three numbered options.

5. **`neither_present_cpdt_full_with_opt_out_succeeds`**: Same as test 4 but source has `@cpdt(weight_aware=false)`. Compile succeeds (no four-case error); stderr does NOT contain "requires weights".

Test fixtures: each test uses `tempfile::TempDir` + inline NSL source via `std::fs::write`. For tests that need a real weights file (tests 1 + 2 + 3), a tiny in-memory safetensors is generated via the existing `safetensors::tensor::serialize` crate (same pattern as `cpdt_validate_body.rs::wm_with_names`).

## 5. Commit Sequencing

Single PR, three commits:

1. `docs(cpdt): AST auto-detect + four-case decision table design` (this spec).
2. `feat(cpdt): AST walkers for load_safetensors + @cpdt(weight_aware)` (`ast_scan.rs` + unit tests for the walkers).
3. `feat(cpdt): four-case decision table in nsl-cli build` (wiring + 5 CLI integration tests).

## 6. Close-Out Criteria

- `ast_scan::find_ast_weight_ref` returns `Some` when the source has a string-literal `load_safetensors(...)` call; unit tests pass.
- `ast_scan::find_ast_cpdt_weight_aware` returns `Some(false)` when the decorator has `weight_aware=false`; unit tests pass.
- The four-case decision table in `nsl-cli/src/main.rs` routes to the correct `weight_file` and emits the correct warning/error in each case; 5 CLI integration tests pass.
- All existing tests still pass.
- Single PR.

## 7. Phase 2 Extension Path

When `@cpdt(weight_aware=...)` becomes per-decorator-site (PR #91's Phase 2 extension path), `find_ast_cpdt_weight_aware` changes signature to `(module, interner, site_span) -> Option<bool>` and the CLI resolves per-site. The four-case decision table stays; only the `weight_aware` lookup becomes scoped.

Similarly, multi-`load_safetensors` support (more than one safetensors file per model) becomes a keyed map from layer-name-prefix → weights-path. That's a separate Phase 2 design; the Phase 1 single-path walker remains the fast-path.
