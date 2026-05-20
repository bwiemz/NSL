//! Phase 2.5 Task 4: pre-pass reads source from disk when profile_source_text
//! is None but profile_source_file_name is set.
#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use std::io::Write;

#[test]
fn pre_pass_reads_source_from_disk_when_text_is_none() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp, "fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:").unwrap();
    writeln!(tmp, "    return x").unwrap();
    tmp.flush().unwrap();
    let path = tmp.path().to_path_buf();

    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    opts.profile_source_text = None; // no explicit text
    opts.profile_source_file_name = Some(path.display().to_string()); // only path

    // Pass empty src to run_pre_pass_only to force reliance on the disk fallback.
    let result = nsl_codegen::test_helpers::run_pre_pass_only("", &opts).unwrap();
    assert!(
        result.source_text.contains("fn forward"),
        "expected disk-read source in PrePassResult, got {:?}",
        result.source_text
    );
}

#[test]
fn pre_pass_prefers_explicit_text_over_disk_read() {
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    opts.profile_source_text = Some("# explicit override".to_string());
    opts.profile_source_file_name = Some("/nonexistent/bogus.nsl".to_string());

    let result = nsl_codegen::test_helpers::run_pre_pass_only("", &opts).unwrap();
    assert_eq!(
        result.source_text, "# explicit override",
        "explicit profile_source_text should win over disk read"
    );
}

#[test]
fn pre_pass_tolerates_missing_file_silently() {
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    opts.profile_source_text = None;
    opts.profile_source_file_name = Some("/definitely/not/a/real/path.nsl".to_string());

    // Should not error — fallback is silent; source_text stays empty.
    let result = nsl_codegen::test_helpers::run_pre_pass_only("", &opts).unwrap();
    assert_eq!(
        result.source_text, "",
        "missing file should degrade silently to empty source_text"
    );
}
