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
}
