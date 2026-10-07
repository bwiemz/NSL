//! The agents subsystem (M56) was removed in the Phase 0.6 scope freeze; the
//! code is preserved at tag `attic/scope-freeze-2026-10`. A program that
//! still uses it must fail loudly, naming the removal, rather than compile
//! into something else. `nsl run --linear-types`, which M56 added, outlived
//! it: the flag turns on the M38a ownership walker, as it does on `nsl check`
//! and `nsl build`.

use std::path::PathBuf;
use std::process::{Command, Output};

const ATTIC_TAG: &str = "attic/scope-freeze-2026-10";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/nsl-cli sits two levels below the workspace root")
        .to_path_buf()
}

/// Write `source` to a scratch file and run `nsl <args...> <file>` on it.
fn nsl(args: &[&str], source: &str) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).expect("write program");
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .arg(&path)
        .current_dir(dir.path())
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl")
}

#[test]
fn an_agent_block_is_refused_with_the_attic_tag() {
    let src = "agent Drafter:\n    steps: int = 0\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n";
    // `--linear-types` used to be what made an agent block legal (E0610).
    for args in [&["check"][..], &["check", "--linear-types"][..]] {
        let out = nsl(args, src);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "nsl {args:?} accepted an agent block:\n{stderr}");
        assert!(
            stderr.contains("agents subsystem (M56)") && stderr.contains(ATTIC_TAG),
            "nsl {args:?} must name the removal and the attic tag:\n{stderr}"
        );
    }
}

#[test]
fn removed_agent_decorators_are_refused_even_when_unknown_names_are_allowed() {
    for name in ["pipeline_agent", "auto_device_transfer"] {
        let src = format!("@{name}\nfn f() -> int:\n    return 1\n");
        let out = nsl(&["check", "--allow-unknown-decorators"], &src);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "@{name} must stay an error under --allow-unknown-decorators:\n{stderr}"
        );
        assert!(
            stderr.contains(&format!("@{name} was removed")) && stderr.contains(ATTIC_TAG),
            "@{name} must get its typed refusal:\n{stderr}"
        );
    }
}

#[test]
fn nsl_run_accepts_linear_types_flag() {
    let out = nsl(&["run", "--linear-types"], "let x = 41\nprint(x + 1)\n");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "nsl run --linear-types failed:\n{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "42", "stderr:\n{stderr}");
}
