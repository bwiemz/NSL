use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

use crate::diagnostic::{Diagnostic, Label, LabelStyle, Level};
use crate::span::{FileId, Span};

pub struct SourceMap {
    files: SimpleFiles<String, String>,
    /// When set, [`emit_diagnostic`](Self::emit_diagnostic) renders nothing.
    /// Diagnostics still reach the caller through the frontend's return
    /// values (the module loader returns `Err` on an error-level one), so
    /// this drops only the stderr rendering — for callers that drive the
    /// frontend many times over the same sources, such as the benches.
    quiet: bool,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            files: SimpleFiles::new(),
            quiet: false,
        }
    }

    /// A map whose [`emit_diagnostic`](Self::emit_diagnostic) is a no-op.
    pub fn silent() -> Self {
        Self {
            quiet: true,
            ..Self::new()
        }
    }

    pub fn add_file(&mut self, name: String, source: String) -> FileId {
        let id = self.files.add(name, source);
        FileId(id)
    }

    /// Whether `span` names a file in this map and lies within its source.
    /// `emit_diagnostic` prints nothing for a label it cannot locate (the
    /// renderer fails after the header, and its error is swallowed), so a
    /// caller with a span from elsewhere — a source the map never saw, or a
    /// synthesized node — checks this first and falls back to a plain line.
    /// [`Span::DUMMY`] is never contained: it is the synthesized-node marker,
    /// and it happens to be a valid empty range at the start of file 0, so
    /// without this check it would render a caret at `<first file>:1:1`.
    pub fn contains_span(&self, span: Span) -> bool {
        if span == Span::DUMMY {
            return false;
        }
        self.files
            .get(span.file_id.0)
            .map(|f| span.start <= span.end && (span.end.0 as usize) <= f.source().len())
            .unwrap_or(false)
    }

    pub fn emit_diagnostic(&self, diag: &Diagnostic) {
        if self.quiet {
            return;
        }
        let severity = match diag.level {
            Level::Error => codespan_reporting::diagnostic::Severity::Error,
            Level::Warning => codespan_reporting::diagnostic::Severity::Warning,
            Level::Info => codespan_reporting::diagnostic::Severity::Note,
        };

        let mut cs_diag = codespan_reporting::diagnostic::Diagnostic::new(severity)
            .with_message(&diag.message);

        let labels: Vec<_> = diag
            .labels
            .iter()
            .map(convert_label)
            .collect();
        cs_diag = cs_diag.with_labels(labels);

        if !diag.notes.is_empty() {
            cs_diag = cs_diag.with_notes(diag.notes.clone());
        }

        let writer = StandardStream::stderr(ColorChoice::Auto);
        let config = term::Config::default();
        let _ = term::emit(&mut writer.lock(), &config, &self.files, &cs_diag);
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

fn convert_label(label: &Label) -> codespan_reporting::diagnostic::Label<usize> {
    let style = match label.style {
        LabelStyle::Primary => codespan_reporting::diagnostic::LabelStyle::Primary,
        LabelStyle::Secondary => codespan_reporting::diagnostic::LabelStyle::Secondary,
    };
    codespan_reporting::diagnostic::Label::new(
        style,
        label.span.file_id.0,
        (label.span.start.0 as usize)..(label.span.end.0 as usize),
    )
    .with_message(&label.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::BytePos;

    #[test]
    fn contains_span_is_file_and_range_membership() {
        let mut map = SourceMap::new();
        let f = map.add_file("a.nsl".into(), "let x = 1\n".into()); // 10 bytes
        let sp = |file: FileId, s: u32, e: u32| Span::new(file, BytePos(s), BytePos(e));
        assert!(map.contains_span(sp(f, 0, 10)));
        assert!(map.contains_span(sp(f, 4, 5)));
        assert!(!map.contains_span(sp(f, 4, 11)), "past the end of the source");
        assert!(!map.contains_span(sp(f, 6, 5)), "inverted range");
        assert!(!map.contains_span(sp(FileId(7), 0, 1)), "file the map never saw");
        assert!(!SourceMap::new().contains_span(Span::DUMMY), "empty map");
        // DUMMY is (file 0, 0..0): a valid empty range in a non-empty map,
        // which must still be refused or a synthesized node renders at 1:1.
        assert!(!map.contains_span(Span::DUMMY), "DUMMY in a map that has file 0");
        assert!(map.contains_span(sp(f, 1, 1)), "a real empty span elsewhere is fine");
    }
}
