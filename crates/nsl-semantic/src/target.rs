//! M47: @target(backend) decorator validation.
//!
//! `cuda` is the only backend. The ROCm, Metal and WebGPU backends were
//! removed in the Phase 0.6 scope freeze (preserved at tag
//! `attic/scope-freeze-2026-10`), so naming one is an error that says so.
//! The decorator itself stays: no codegen reads it, and refusing a removed
//! backend by name tells the user more than an unknown-decorator error would.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// The backend names `@target` accepted before the scope freeze.
const REMOVED_BACKENDS: &[&str] = &["rocm", "metal", "webgpu"];

/// Where the removed backends are preserved.
const ATTIC_TAG: &str = "attic/scope-freeze-2026-10";

/// Validate `@target(backend)` decorator arguments.
///
/// Returns the list of valid target names found. Emits diagnostics for
/// removed or unknown targets, and for a decorator that names no backend at
/// all. A rejected name already has its own error, so it does not also get
/// the "requires at least one" one.
pub fn validate_target_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<String> {
    let mut targets = Vec::new();
    let mut named_any = false;
    if let Some(ref args) = deco.args {
        for arg in args {
            // Positional args: target names
            if arg.name.is_none()
                && let ExprKind::Ident(sym) = &arg.value.kind
            {
                let name = resolve_sym(*sym);
                named_any = true;
                if name == "cuda" {
                    targets.push(name);
                } else if REMOVED_BACKENDS.contains(&name.as_str()) {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "target '{name}' was removed: the ROCm, Metal and WebGPU \
                             backends were cut in the Phase 0.6 scope freeze (preserved \
                             at tag `{ATTIC_TAG}`); the only target is cuda"
                        ))
                        .with_label(arg.value.span, "here"),
                    );
                } else {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "unknown target '{name}', expected: cuda (the ROCm, Metal and \
                             WebGPU backends were removed; see tag `{ATTIC_TAG}`)"
                        ))
                        .with_label(arg.value.span, "here"),
                    );
                }
            }
        }
    }
    if !named_any {
        diagnostics.push(
            Diagnostic::error("@target requires at least one backend name")
                .with_label(deco.span, "here"),
        );
    }
    targets
}
