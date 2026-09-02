use std::fmt;

use nsl_errors::{Diagnostic, Span};

/// A compile failure raised by codegen.
///
/// `message` is the user-facing text and the only field most sites set
/// (`CodegenError::new`). `span`, when present, is the innermost statement or
/// expression that was being compiled when the error was raised: the
/// `compile_stmt` / `compile_expr` dispatchers attach it on the way out
/// ([`Self::with_span_if_unset`]), so an error raised anywhere inside a
/// function body carries one without its raise site knowing about spans.
/// Errors raised outside statement compilation — kernel synthesis, model
/// collection, the WGGO pre-pass, `main` assembly — have none unless their
/// site attaches one. `notes` are rendered after the excerpt, one per line.
///
/// `Display` is the bare message; the CLI adds its `codegen error:` prefix
/// (or renders the excerpt through the source map, see
/// [`Self::to_diagnostic`]).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CodegenError {
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CodegenError {}

impl CodegenError {
    pub fn new(msg: impl Into<String>) -> Self {
        CodegenError {
            message: msg.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    /// The error's location, overriding any span already attached.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Attach `span` as the error's location unless a narrower one is already
    /// known. This is what the statement and expression dispatchers call as
    /// an error propagates outward: the innermost node that was being
    /// compiled is the one the user wants pointed at, and it is the first to
    /// run this, so later (wider) callers change nothing. `Span::DUMMY` —
    /// the span of a node the compiler synthesized — is not a location and
    /// is ignored.
    pub fn with_span_if_unset(mut self, span: Span) -> Self {
        if self.span.is_none() && span != Span::DUMMY {
            self.span = Some(span);
        }
        self
    }

    /// Add a note, rendered after the message (and the excerpt, when there is
    /// a span).
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// The error as an `nsl-errors` diagnostic: an error-level message with
    /// the span (if any) as its primary label and the notes carried over.
    /// Rendering it through a `SourceMap` gives the excerpt-and-caret form
    /// the frontend's own diagnostics use.
    pub fn to_diagnostic(&self) -> Diagnostic {
        let mut diag = Diagnostic::error(self.message.clone());
        if let Some(span) = self.span {
            diag = diag.with_label(span, "");
        }
        for note in &self.notes {
            diag = diag.with_note(note.clone());
        }
        diag
    }

    /// Construct a `MissingScales` error for the given projection path.
    ///
    /// Emitted during final-compile AWQ lowering when a calibration sidecar
    /// is present but does not contain scales for the named projection.
    /// Silent fallback to uncalibrated is a correctness trap, so this is
    /// always a hard error.
    pub fn missing_scales(projection_path: impl Into<String>) -> Self {
        let path = projection_path.into();
        CodegenError::new(format!(
            "MissingScales: AWQ calibration sidecar is present but missing \
             scales for projection '{path}'"
        ))
    }
}

impl From<crate::calibration::DiscoveryError> for CodegenError {
    fn from(e: crate::calibration::DiscoveryError) -> Self {
        CodegenError::new(format!("calibration discovery: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::{BytePos, FileId, LabelStyle};

    fn span(start: u32, end: u32) -> Span {
        Span::new(FileId(0), BytePos(start), BytePos(end))
    }

    #[test]
    fn display_is_the_bare_message() {
        // The CLI prefixes; a Display that also prefixed printed
        // "codegen error: codegen error: …" at every CLI site.
        let e = CodegenError::new("unsupported tensor-scalar op: Eq");
        assert_eq!(e.to_string(), "unsupported tensor-scalar op: Eq");
        assert_eq!(e.message, "unsupported tensor-scalar op: Eq");
        assert!(e.span.is_none());
        assert!(e.notes.is_empty());
    }

    #[test]
    fn innermost_span_wins_as_the_error_propagates_outward() {
        // Raised with no span, passed through an expression, then its
        // statement: the expression's (narrower) span is what remains.
        let e = CodegenError::new("x")
            .with_span_if_unset(span(10, 14))
            .with_span_if_unset(span(0, 40));
        assert_eq!(e.span, Some(span(10, 14)));
        // A site that knows better may still override.
        assert_eq!(e.with_span(span(2, 3)).span, Some(span(2, 3)));
    }

    #[test]
    fn a_dummy_span_is_not_a_location() {
        // Synthesized nodes carry Span::DUMMY; pointing the caret at byte 0
        // of file 0 would be a fabricated location.
        let e = CodegenError::new("x").with_span_if_unset(Span::DUMMY);
        assert!(e.span.is_none());
        let e = CodegenError::new("x")
            .with_span_if_unset(Span::DUMMY)
            .with_span_if_unset(span(5, 6));
        assert_eq!(e.span, Some(span(5, 6)));
    }

    #[test]
    fn diagnostic_carries_span_as_primary_label_and_notes() {
        let e = CodegenError::new("kernel 'k': `==` is not supported")
            .with_span(span(3, 9))
            .with_note("supported comparisons: < <= > >=");
        let d = e.to_diagnostic();
        assert_eq!(d.level, nsl_errors::Level::Error);
        assert_eq!(d.message, "kernel 'k': `==` is not supported");
        assert_eq!(d.labels.len(), 1);
        assert_eq!(d.labels[0].span, span(3, 9));
        assert_eq!(d.labels[0].style, LabelStyle::Primary);
        assert_eq!(d.notes, vec!["supported comparisons: < <= > >=".to_string()]);

        let bare = CodegenError::new("no span").to_diagnostic();
        assert!(bare.labels.is_empty());
    }
}
