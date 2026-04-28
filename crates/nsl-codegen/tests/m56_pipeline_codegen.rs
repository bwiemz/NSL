//! M56 Task 18 tests — verify `@pipeline_agent` functions compile to
//! Cranelift orchestration code (Option A: direct dispatch to per-agent
//! method symbols, no runtime scheduler call).
//!
//! Spec §5.2 scope decision (Task 18 v1):
//! - Option A: `@pipeline_agent` fns call `__nsl_agent_{Name}_{method}` directly.
//! - No `nsl_agent_scheduler_step` / `nsl_agent_pool_*` calls.
//! - No `@auto_device_transfer` (Task 19), no serve-block integration (v2).
//!
//! Both tests compile a source fixture to an object file and assert:
//! 1. Compilation succeeds without errors.
//! 2. The compiled object contains the pipeline fn symbol.
//! 3. The compiled object contains the per-agent method symbols, confirming
//!    Option A dispatch wiring.

// ── Helper: compile source → object bytes ────────────────────────────────────

fn compile_src(src: &str) -> Result<Vec<u8>, String> {
    use nsl_errors::Level;
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, Level::Error)) {
        return Err(format!("lex errors: {:?}", lex_diags));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed.diagnostics.iter().any(|d| matches!(d.level, Level::Error)) {
        return Err(format!("parse errors: {:?}", parsed.diagnostics));
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    let opts = nsl_codegen::CompileOptions::default();
    nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        &opts,
    )
    .map_err(|e| format!("codegen error: {}", e.message))
}

// ── Test 1: two-agent pipeline compiles with both method symbols in the object ─

/// Verify that a two-agent `@pipeline_agent` function compiles without errors
/// and that the compiled object contains:
/// - The pipeline fn symbol (`pipeline`).
/// - The Drafter method symbol (`__nsl_agent_Drafter_draft`).
/// - The Reviewer method symbol (`__nsl_agent_Reviewer_review`).
///
/// Spec §5.2 Option A: pipeline body dispatches directly to per-agent methods.
#[test]
fn pipeline_fn_compiles_with_agent_method_calls() {
    let src = "\
agent Drafter:\n    fn draft(self, p: i32) -> i32:\n        return p\n\
agent Reviewer:\n    fn review(self, draft: i32) -> i32:\n        return draft\n\
@pipeline_agent(agents=[Drafter, Reviewer])\n\
fn pipeline(prompt: i32) -> i32:\n    let draft = drafter.draft(prompt)\n    return reviewer.review(draft)\n";

    let bytes = compile_src(src).expect("@pipeline_agent fn must compile without errors");

    let obj_str = String::from_utf8_lossy(&bytes);

    // The pipeline fn symbol must be present.
    assert!(
        obj_str.contains("pipeline"),
        "compiled object must contain the pipeline fn symbol 'pipeline'"
    );

    // Both per-agent method symbols must be present (Option A: direct dispatch).
    assert!(
        obj_str.contains("__nsl_agent_Drafter_draft"),
        "compiled object must contain '__nsl_agent_Drafter_draft' \
         (Option A: @pipeline_agent dispatches directly to agent method symbols)"
    );
    assert!(
        obj_str.contains("__nsl_agent_Reviewer_review"),
        "compiled object must contain '__nsl_agent_Reviewer_review'"
    );
}

// ── Test 2: single-agent pipeline routes to pipeline lowering path ────────────

/// Verify that a single-agent `@pipeline_agent`-decorated fn:
/// 1. Compiles without errors (confirms the decorator-routing path fires).
/// 2. Produces a symbol for the pipeline fn in the compiled object.
/// 3. Produces a symbol for the agent method in the compiled object.
///
/// This distinguishes the @pipeline_agent path from the plain-fn path: a plain
/// `fn solo` compiled via `compile_fn_def` would not know about `a.a_fn(x)` and
/// would emit a warning + fall through to tensor dispatch.  The pipeline path
/// pre-binds `a` as a stack-allocated `AgentA` state pointer so the dispatch
/// succeeds.
#[test]
fn pipeline_agent_decorator_routes_to_pipeline_lowering() {
    let src = "\
agent A:\n    fn a_fn(self, x: i32) -> i32:\n        return x\n\
@pipeline_agent(agents=[A])\n\
fn solo(x: i32) -> i32:\n    return a.a_fn(x)\n";

    let bytes = compile_src(src)
        .expect("single-agent @pipeline_agent fn must compile without errors");

    let obj_str = String::from_utf8_lossy(&bytes);

    // Pipeline fn symbol must appear in the object.
    assert!(
        obj_str.contains("solo"),
        "compiled object must contain the pipeline fn symbol 'solo'"
    );

    // Per-agent method symbol must appear (confirms Option A dispatch wired up).
    assert!(
        obj_str.contains("__nsl_agent_A_a_fn"),
        "compiled object must contain '__nsl_agent_A_a_fn' \
         (confirms @pipeline_agent routing to pipeline lowering path)"
    );
}
