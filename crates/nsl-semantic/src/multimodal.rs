//! M48: @multimodal decorator validation.

use nsl_ast::decl::Decorator;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @multimodal decorator.
/// Applied to ModelDef only. Informational -- warns if unexpected arguments provided.
pub fn validate_multimodal_decorator(
    deco: &Decorator,
    _resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    // @multimodal takes no arguments -- just validate it exists
    if let Some(ref args) = deco.args {
        if !args.is_empty() {
            diagnostics.push(
                Diagnostic::warning("@multimodal takes no arguments")
                    .with_label(deco.span, "unexpected arguments"),
            );
        }
    }
    true
}
