//! Item 10 e2e: the offline measured path and the frozen tuning DB.
//!
//! The load-bearing property is the KEY ROUNDTRIP: `nsl autotune` must
//! build the identical cache key the compile path builds (same AST bytes,
//! same device identity), or it writes Measured records no compile ever
//! reads — and the feature silently reverts to the cost model. Asserting a
//! VERBOSE "cache hit" on a compile immediately after tuning is the proof;
//! asserting it again in a FRESH cache dir supplied only by `--autotune-db`
//! proves the frozen overlay end-to-end.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const PROG: &str = r#"@autotune(block_size=[64, 128, 256])
kernel scale_by(data: Tensor, scale: f32) -> Tensor:
    let tid = thread_id()
    data[tid] = data[tid] * scale

print("autotune_gate_ok")
"#;

fn nsl(cwd: &Path, args: &[&str], envs: &[(&str, &str)]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .current_dir(cwd)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .envs(envs.iter().copied())
        .output()
        .expect("spawn nsl");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn autotune_command_refuses_a_file_without_autotune_kernels() {
    let tmp = std::env::temp_dir().join(format!("nsl_at_none_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("plain.nsl"), "print(\"hi\")\n").unwrap();
    let (ok, _, err) = nsl(&tmp, &["autotune", "plain.nsl"], &[]);
    assert!(!ok, "autotune of a kernel-less file must refuse");
    assert!(
        err.contains("no @autotune kernels"),
        "wrong refusal:\n{err}"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn autotune_measures_and_compiles_consume_it_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_at_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("prog.nsl"), PROG).unwrap();

    // 1. Offline tune, freezing a DB. This is the measured path's one
    //    production trigger.
    let (ok, out, err) = nsl(
        &tmp,
        &["autotune", "prog.nsl", "--freeze", "db.json"],
        &[],
    );
    assert!(ok, "nsl autotune failed:\nstdout:\n{out}\nstderr:\n{err}");
    assert!(
        out.contains("measured winner"),
        "no measured winner reported:\n{out}"
    );
    let sha = out
        .lines()
        .find_map(|l| l.split("--autotune-db-sha256 ").nth(1))
        .expect("freeze must print the pin hash")
        .trim()
        .to_string();
    assert!(tmp.join("db.json").exists());

    // 2. A compile in the SAME cache dir now takes the Measured record —
    //    the key roundtrip. If the offline key drifts from the compile key,
    //    this is the assert that goes red.
    let (ok, _, err) = nsl(
        &tmp,
        &["build", "prog.nsl", "-o", "prog_a.bin"],
        &[("NSL_AUTOTUNE_VERBOSE", "1")],
    );
    assert!(ok, "build after tune failed:\n{err}");
    assert!(
        err.contains("cache hit for scale_by"),
        "the compile did not consume the measured record — key drift between \
         `nsl autotune` and the compile path:\n{err}"
    );

    // 3. Wrong pin refuses the build outright.
    let (ok, _, err) = nsl(
        &tmp,
        &[
            "build", "prog.nsl", "-o", "prog_b.bin",
            "--autotune-db", "db.json",
            "--autotune-db-sha256", "deadbeef",
        ],
        &[],
    );
    assert!(!ok, "a bad pin must refuse the build");
    assert!(err.contains("sha256 mismatch"), "wrong refusal:\n{err}");

    // 4. Fresh cache dir, records supplied ONLY by the pinned DB: the
    //    overlay alone must produce the cache hit.
    let tmp2 = std::env::temp_dir().join(format!("nsl_at_e2e2_{}", std::process::id()));
    std::fs::create_dir_all(&tmp2).unwrap();
    std::fs::write(tmp2.join("prog.nsl"), PROG).unwrap();
    let db_abs = tmp.join("db.json");
    let (ok, _, err) = nsl(
        &tmp2,
        &[
            "build", "prog.nsl", "-o", "prog_c.bin",
            "--autotune-db", db_abs.to_str().unwrap(),
            "--autotune-db-sha256", &sha,
        ],
        &[("NSL_AUTOTUNE_VERBOSE", "1")],
    );
    assert!(ok, "pinned build failed:\n{err}");
    assert!(
        err.contains("frozen db: 1 record(s)"),
        "frozen DB was not loaded:\n{err}"
    );
    assert!(
        err.contains("cache hit for scale_by"),
        "the frozen overlay did not supply the winner in a fresh cache dir:\n{err}"
    );
}
