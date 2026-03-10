use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

use crate::diagnostic::{Diagnostic, Label, LabelStyle, Level};
use crate::span::FileId;

pub struct SourceMap {
    files: SimpleFiles<String, String>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            files: SimpleFiles::new(),
        }
    }

    pub fn add_file(&mut self, name: String, source: String) -> FileId {
        let id = self.files.add(name, source);
        FileId(id)
    }

    pub fn emit_diagnostic(&self, diag: &Diagnostic) {
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
            .map(|label| convert_label(label))
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
