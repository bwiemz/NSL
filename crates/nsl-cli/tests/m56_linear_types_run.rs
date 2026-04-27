//! M56 Task 20 — verify `nsl run --linear-types` accepts the flag and
//! semantic-checks an agent declaration without firing E0610.

#[test]
fn nsl_run_accepts_linear_types_flag() {
    // The minimal source: an agent declaration that requires --linear-types.
    // We don't need the full pipeline to run successfully — just that the
    // flag is accepted and E0610 doesn't fire.
    use std::io::Write;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m56_t20.nsl");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(b"agent Foo:\n    fn noop(self):\n        return\n")
        .expect("write");
    drop(f);

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--linear-types", path.to_str().unwrap()])
        .output()
        .expect("nsl run failed to spawn");

    let stderr = String::from_utf8_lossy(&out.stderr);

    // The pipeline may fail to RUN (no entry point, no main, etc.) — we
    // don't care about success/failure of execution. We care that the
    // flag was accepted (no "unexpected argument" error from clap) and
    // E0610 was NOT emitted.
    assert!(
        !stderr.contains("E0610"),
        "E0610 should not fire when --linear-types is passed; stderr:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("unrecognized") && !stderr.contains("unexpected argument"),
        "--linear-types should be a valid flag for nsl run; stderr:\n{}",
        stderr
    );
}

#[test]
fn nsl_run_without_linear_types_still_errors_e0610_for_agents() {
    // Sanity: without the flag, E0610 still fires (Task 4 invariant preserved).
    use std::io::Write;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m56_t20_no_flag.nsl");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(b"agent Foo:\n    fn noop(self):\n        return\n")
        .expect("write");
    drop(f);

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", path.to_str().unwrap()])
        .output()
        .expect("nsl run failed to spawn");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("E0610"),
        "E0610 should fire when --linear-types is absent; stderr:\n{}",
        stderr
    );
}
