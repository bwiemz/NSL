//! Milestone A, @cpdt fix: the train-block decorator must configure CPDT on
//! EVERY entry path, with the per-entry-point matrix PR #502's audit asked
//! for and never got.
//!
//! History being pinned: `apply_cpdt_decorator_from_ast` lived only in
//! `compile_returning_plan_impl` (single-file). Every program with an import
//! — i.e. every real program — compiled through `compile_entry_returning_plan`,
//! which never called it: `@cpdt(mode = off)` built byte-identical to the
//! no-decorator control, measured in PR #502 and re-measured 2026-08-15.
//! Additionally BOTH paths' `invoke_cpdt_if_enabled` early-returned silently
//! on mode-off / missing-cluster, so the pass-trace showed `did not run:
//! CPDT` even where the decorator HAD been applied.
//!
//! The witness through this file is the `[cpdt]` decorator marker plus the
//! `[pass-trace] CPDT:` disposition line — build-stderr differential against
//! a no-decorator control, the method that survived the binary-hash trap
//! (nsl is not byte-reproducible; hashing "differs" for everything).

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// No imports: exercises `compile_returning_plan_impl`.
const SINGLE_FILE: &str = r#"
model Tiny:
    w: Tensor = randn([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = randn([2, 4])
__DECORATOR__train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let y = m.forward(x)
        let loss = (y * y).sum()
"#;

/// One stdlib import: forces the multi-file loader and
/// `compile_entry_returning_plan` — the path that historically never applied
/// the decorator.
const MULTI_FILE: &str = r#"
from nsl.nn.losses import cross_entropy

model Tiny:
    w: Tensor = randn([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = randn([2, 4])
__DECORATOR__train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let y = m.forward(x)
        let loss = (y * y).sum()
"#;

fn build_stderr(tag: &str, template: &str, decorator: &str) -> String {
    let tmp = std::env::temp_dir().join(format!("nsl_cpdt_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    let src = template.replace(
        "__DECORATOR__",
        if decorator.is_empty() { String::new() } else { format!("{decorator}\n") }.as_str(),
    );
    std::fs::write(&prog, src).unwrap();
    let exe = tmp.join("out.exe");
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build", "-o", exe.to_str().unwrap()])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .env("NSL_PASS_TRACE", "1")
        .output()
        .expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success(),
        "[{tag}] build failed:\n{stderr}"
    );
    std::fs::remove_dir_all(&tmp).ok();
    stderr
}

const APPLIED_MARKER: &str = "[cpdt] @cpdt decorator applied: mode=off";

/// The control first: WITHOUT the decorator, the applied-marker must be
/// absent on both paths, or the positive assertions below prove nothing.
#[test]
fn control_builds_carry_no_decorator_marker() {
    for (tag, template) in [("single", SINGLE_FILE), ("multi", MULTI_FILE)] {
        let stderr = build_stderr(&format!("ctl_{tag}"), template, "");
        assert!(
            !stderr.contains("[cpdt] @cpdt decorator applied"),
            "[{tag}] control unexpectedly shows the decorator marker:\n{stderr}"
        );
    }
}

/// Single-file path: decorator applied, CPDT disposition recorded.
#[test]
fn single_file_applies_the_decorator_and_records_cpdt() {
    let stderr = build_stderr("single", SINGLE_FILE, "@cpdt(mode = off)");
    assert!(
        stderr.contains(APPLIED_MARKER),
        "single-file path lost the decorator application:\n{stderr}"
    );
    assert!(
        stderr.contains("CPDT: declined, mode off"),
        "CPDT must record a typed decline for mode=off:\n{stderr}"
    );
}

/// Multi-file path — THE regression this phase fixes. If this assertion
/// fails, `compile_entry_returning_plan` lost the decorator call again and
/// every imported-module program has an inert @cpdt.
#[test]
fn multi_file_applies_the_decorator_and_records_cpdt() {
    let stderr = build_stderr("multi", MULTI_FILE, "@cpdt(mode = off)");
    assert!(
        stderr.contains(APPLIED_MARKER),
        "MULTI-FILE path lost the decorator application (the PR #502 class):\n{stderr}"
    );
    assert!(
        stderr.contains("CPDT: declined, mode off"),
        "CPDT must record a typed decline for mode=off on the entry path:\n{stderr}"
    );
}

/// `@cpdt(mode = full)` with no cluster anywhere: previously a silent early
/// return; now a typed precondition decline that names the missing input.
#[test]
fn mode_full_without_cluster_declines_with_the_reason() {
    let stderr = build_stderr("nocluster", MULTI_FILE, "@cpdt(mode = full)");
    assert!(
        stderr.contains("CPDT: declined, precondition violated - no cluster specification"),
        "missing-cluster must be a typed decline, not silence:\n{stderr}"
    );
}
