//! A packed `DataLoader` must be told which token ends a document.
//!
//! `packing = true` cuts the token stream into rows and, at every separator
//! token, starts a new document: attention is cut there, positions restart,
//! and the next-token target is dropped. The separator defaulted to id 0,
//! and id 0 is an ordinary token -- "!" in both shipped tokenizers. A
//! loader built without `pack_separator` therefore split documents at every
//! "!" and printed nothing. Rows dense in the separator then tripped the
//! 256-documents-per-row limit inside a loader worker thread, whose panic
//! killed only that thread: with one worker the epoch ended early with exit
//! status 0, with several the consumer waited forever for the batch the
//! dead worker had claimed.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn nsl(dir: &std::path::Path, source: &str) -> Command {
    let path = dir.join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir)
        .arg("run")
        .arg(&path)
        .arg("--target")
        .arg("cpu");
    cmd
}

fn loader_program(loader_args: &str) -> String {
    format!(
        r#"let tokens = tensor_cat([full([3], 7.0), zeros([1]), full([3], 9.0), zeros([1])], 0)
let labels = zeros([8])
let loader = DataLoader(tokens, {loader_args})
for batch in loader:
    print(batch.segment_ids)
"#
    )
}

#[test]
fn packing_without_a_separator_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let out = nsl(dir.path(), &loader_program("batch_size=1, seq_len=8, packing=true"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the program compiled and ran");
    assert!(
        stderr.contains("`packing = true` needs `pack_separator = <token id>`"),
        "{stderr}"
    );
}

#[test]
fn packing_with_a_separator_splits_at_that_token_only() {
    let dir = tempfile::tempdir().unwrap();
    let out = nsl(
        dir.path(),
        &loader_program("batch_size=1, seq_len=8, shuffle=false, packing=true, pack_separator=0"),
    )
    .output()
    .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    // 7 7 7 0 | 9 9 9 0: the separator belongs to the document it ends.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]"),
        "segment ids were not [0 0 0 0 1 1 1 1]:\n{stdout}"
    );
}

/// A packed batch derives its labels from the token stream, so a `labels`
/// tensor would be accepted and read by nothing.
#[test]
fn labels_with_packing_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let out = nsl(
        dir.path(),
        &loader_program("labels, batch_size=1, seq_len=8, packing=true, pack_separator=0"),
    )
    .output()
    .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the program compiled and ran");
    assert!(
        stderr.contains("`labels` cannot be combined with `packing = true`"),
        "{stderr}"
    );
}

/// `labels = None` is the spelled-out default, not a labels tensor.
#[test]
fn an_explicit_none_for_labels_is_accepted_with_packing() {
    let dir = tempfile::tempdir().unwrap();
    let out = nsl(
        dir.path(),
        &loader_program(
            "labels=None, batch_size=1, seq_len=8, shuffle=false, packing=true, pack_separator=0",
        ),
    )
    .output()
    .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
}

/// More than 256 documents in one row, with one worker and with two. Both
/// must stop the process with a failure status; neither may run on to the
/// statement after the loop, and neither may hang.
#[test]
fn a_row_with_too_many_documents_stops_the_run() {
    // 1024 tokens alternating 1, 0: 512 two-token documents in one row.
    let source = |workers: usize| {
        format!(
            r#"let pair = tensor_cat([ones([1]), zeros([1])], 0)
let tokens = pair.reshape([1, 2]).expand([1024, 2]).contiguous().reshape([2048])
let loader = DataLoader(tokens, batch_size=1, seq_len=1024, shuffle=false, num_workers={workers}, packing=true, pack_separator=0)
for batch in loader:
    print(batch.segment_ids.sum())
print("LOOP FINISHED")
"#
        )
    };
    for workers in [1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let mut child = nsl(dir.path(), &source(workers))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if std::time::Instant::now() > deadline {
                child.kill().unwrap();
                panic!("{workers} worker(s): the run hung after a loader worker died");
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        let out = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !status.success(),
            "{workers} worker(s): exit status 0 after a loader worker died\n{stderr}"
        );
        assert!(
            !stdout.contains("LOOP FINISHED"),
            "{workers} worker(s): the loop ended as if the data had run out"
        );
        assert!(
            stderr.contains("holds more than 256 documents")
                && stderr.contains("DataLoader worker thread panicked"),
            "{workers} worker(s): {stderr}"
        );
    }
}
