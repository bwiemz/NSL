//! `@target(backend)` after the Phase 0.6 scope freeze: `cuda` is the only
//! backend. The ROCm, Metal and WebGPU backends were removed (preserved at
//! tag `attic/scope-freeze-2026-10`). Naming one is an error that says so
//! and names the tag. Before the freeze, `rocm`, `metal` and `webgpu` were
//! accepted and nothing generated code for them.

use nsl_errors::FileId;
use nsl_lexer::Interner;

fn analyze_src(src: &str) -> Vec<String> {
    let mut interner = Interner::new();
    let file_id = FileId(0);
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, file_id, &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    let mut msgs: Vec<String> = lex_diags.iter().map(|d| d.message.clone()).collect();
    msgs.extend(parse_result.diagnostics.iter().map(|d| d.message.clone()));
    msgs.extend(
        analysis
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .map(|d| d.message.clone()),
    );
    msgs
}

fn model_with_target(args: &str) -> String {
    format!(
        "model M:\n    @target({args})\n    layer: int = 0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x\n"
    )
}

fn target_errs(args: &str) -> Vec<String> {
    analyze_src(&model_with_target(args))
        .into_iter()
        .filter(|m| m.contains("target"))
        .collect()
}

#[test]
fn cuda_is_accepted() {
    let errs = target_errs("cuda");
    assert!(errs.is_empty(), "@target(cuda) must be clean, got {errs:?}");
}

#[test]
fn a_removed_backend_is_refused_with_the_attic_tag() {
    for removed in ["rocm", "metal", "webgpu"] {
        let errs = target_errs(removed);
        assert_eq!(errs.len(), 1, "@target({removed}): exactly one error, got {errs:?}");
        let e = &errs[0];
        assert!(e.contains(&format!("'{removed}'")), "{e}");
        assert!(e.contains("removed"), "{e}");
        assert!(e.contains("attic/scope-freeze-2026-10"), "{e}");
    }
}

#[test]
fn an_unknown_backend_is_refused_naming_cuda() {
    let errs = target_errs("vulkan");
    assert_eq!(errs.len(), 1, "got {errs:?}");
    assert!(errs[0].contains("unknown target 'vulkan', expected: cuda"), "{}", errs[0]);
}

#[test]
fn a_removed_backend_beside_cuda_is_still_refused() {
    let errs = target_errs("cuda, metal");
    assert_eq!(errs.len(), 1, "got {errs:?}");
    assert!(errs[0].contains("'metal' was removed"), "{}", errs[0]);
}

#[test]
fn no_backend_at_all_is_refused() {
    let errs = target_errs("");
    assert!(
        errs.iter().any(|e| e.contains("@target requires at least one backend name")),
        "got {errs:?}"
    );
}
