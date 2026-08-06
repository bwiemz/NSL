//! Repo-hygiene gate: nothing under `target` may be tracked.
//!
//! This has now happened TWICE: d9c5fbb0 removed two generated reports that
//! were committed under `target/` despite the gitignore, and 7678dbf4
//! committed `target` ITSELF — as a symlink whose absolute path pointed into
//! one development box's `~/.claude/cargo-targets/<other-worktree>` build
//! dir. The second is the nastier failure: a fresh checkout materializes the
//! link, and every test that resolves the compiled binary via the
//! conventional `<repo>/target/debug/nsl` path either dies with
//! `NotADirectory`/`NotFound` (CI, other machines) or — worse — silently
//! runs ANOTHER WORKTREE's stale binary (the box the path exists on). A
//! whole class of e2e binaries reported instant all-fail sweeps that
//! read as environmental noise for a day.
//!
//! `.gitignore` cannot prevent this: an explicit `git add`/`ln -s` into a
//! tracked path bypasses it, which is exactly how both incidents happened.
//! Only a gate that asks git what is TRACKED can.

use std::process::Command;

#[test]
fn nothing_under_target_is_tracked() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let out = Command::new("git")
        .args(["ls-files", "--", "target", "target/"])
        .current_dir(&repo_root)
        .output();
    let out = match out {
        Ok(o) if o.status.success() => o,
        // No git on PATH or not a work tree (a source tarball): nothing to
        // check, and failing would gate on the environment, not the repo.
        _ => return,
    };
    let tracked = String::from_utf8_lossy(&out.stdout);
    assert!(
        tracked.trim().is_empty(),
        "tracked entries under `target`: {} — `git rm --cached` them. A \
         tracked `target` symlink points every conventional \
         <repo>/target/debug/nsl lookup at another machine's (or another \
         worktree's) build products.",
        tracked.trim().replace('\n', ", ")
    );
}
