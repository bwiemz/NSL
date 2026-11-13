//! `DataLoader(...)` keyword arguments are a closed, typed set.
//!
//! Codegen copied every named literal into the loader's JSON config and the
//! runtime read the keys it knew with `as_u64()` / `as_bool()` and a default.
//! A misspelt key or a literal of the wrong type was therefore not an error —
//! it was the DEFAULT, silently: `seq_len = 2048.0` trained at seq_len 128,
//! `batch_size = 2.0` at batch 1, `shuffle = 1` did not shuffle, and
//! `seqlen = 2048` was ignored. Each of those changes what a run trains on
//! without a line of output, which is the failure the Training Configuration
//! Contract closed for `train(...)` and left open here.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(tag: &str, loader_args: &str) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!("nsl_dlkwargs_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    let src = format!(
        "let corpus = arange(64)\n\
         let loader = DataLoader(corpus, {loader_args})\n\
         let n = 0\n\
         for batch in loader:\n\
         \x20   let s = batch.input_ids.shape\n\
         \x20   if n == 0:\n\
         \x20       print(s[0])\n\
         \x20       print(s[1])\n\
         \x20   n = n + 1\n\
         print(n)\n\
         print(\"DL_KWARGS_DONE\")\n"
    );
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cpu"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// The accepted form still runs, with the geometry it asked for: 64 tokens at
/// batch 2 x seq 4 is 8 batches of shape [2, 4].
#[test]
fn well_typed_arguments_are_honoured() {
    let (ok, out, err) = run("ok", "batch_size = 2, seq_len = 4, shuffle = false, drop_last = true");
    assert!(ok && out.contains("DL_KWARGS_DONE"), "the control failed:\n{out}\n{err}");
    let lines: Vec<&str> = out.lines().map(str::trim).collect();
    assert_eq!(&lines[..3], ["2", "4", "8"], "batch geometry:\n{out}");
}

#[test]
fn a_mistyped_or_unknown_argument_is_refused() {
    for (tag, args, needle) in [
        ("float_seq", "batch_size = 2, seq_len = 4.0", "'seq_len' takes an integer literal"),
        ("float_batch", "batch_size = 2.0, seq_len = 4", "'batch_size' takes an integer literal"),
        ("int_shuffle", "batch_size = 2, seq_len = 4, shuffle = 1", "'shuffle' takes a `true`/`false` literal"),
        ("bool_seq", "batch_size = 2, seq_len = true", "'seq_len' takes an integer literal"),
        ("misspelt", "batch_size = 2, seqlen = 4", "unknown keyword argument 'seqlen'"),
        ("twice", "batch_size = 2, seq_len = 4, seq_len = 8", "'seq_len' given more than once"),
    ] {
        let (ok, out, err) = run(tag, args);
        assert!(
            !ok && !out.contains("DL_KWARGS_DONE"),
            "[{tag}] `DataLoader(corpus, {args})` ran — it would have trained on a default:\n{out}"
        );
        assert!(err.contains(needle), "[{tag}] the refusal must name the argument ({needle:?}):\n{err}");
    }
}
