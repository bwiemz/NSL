//! e2e regression: `.clone()` on a binding that ALIASES a model field (or
//! another live variable) must produce an independent copy — mutating the
//! clone must never mutate the aliased original.
//!
//! FBIP Phase 2's clone-elision (`crates/nsl-codegen/src/expr/advanced.rs`)
//! used to skip the `nsl_tensor_clone` FFI call whenever the source Ident
//! was "single-use" per a purely textual reference count
//! (`crate::use_count::UseCountMap`). That count has no notion of aliasing:
//! `let x = m.w` hands out the model's own tensor pointer with no retain,
//! and `x` being referenced exactly once (in `x.clone()`) does not mean the
//! object `x` is bound to is exclusively owned — `m.w` is still live and
//! reachable through `m`. Eliding the clone returned `m.w`'s own pointer,
//! so writing through the "clone" silently corrupted the model weight —
//! the identical corruption class ad59b929 (2026-07-23) fixed for the
//! runtime refcount==1 heuristic, reached here through static codegen
//! analysis instead. Same class of fix: the unsound heuristic is removed
//! and `.clone()` now always emits a real copy unless `--linear-types`
//! ownership lowering has actually proven no borrows/sharing exist.

use std::process::Command;

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const FIXTURE: &str = r#"model M:
    w: Tensor = zeros([2, 2]) - 1.0

let m = M()
let x = m.w
let c = x.clone()
c[0, 0] = 99.0
print(sum(m.w))
"#;

#[test]
fn e2e_clone_of_model_field_alias_does_not_corrupt_weight() {
    let root = workspace_root();
    let dir = std::env::temp_dir().join(format!("nsl_clone_alias_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, FIXTURE).expect("write fixture");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--features",
            if cfg!(feature = "cuda") { "cuda" } else { "" },
            "--",
            "run",
        ])
        .arg(&file)
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to execute nsl run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        output.status.success(),
        "run failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );

    let printed = stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no numeric line in stdout:\n{stdout}"));

    assert!(
        (printed - (-4.0)).abs() < 1e-6,
        "model weight `m.w` was corrupted by mutating a clone of an alias \
         to it: expected sum(m.w) == -4.0 (unmutated), got {printed}. A \
         clone-elision regression returned m.w's own pointer as the \
         'clone', so `c[0, 0] = 99.0` wrote straight through into the \
         weight.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
