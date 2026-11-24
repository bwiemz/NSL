//! A failed `nsl build` / `nsl run` removes its scratch directory.
//!
//! The compiler stages objects (and, for `nsl run`, the executable) in
//! `<temp>/nsl_build_<pid>` and `<temp>/nsl_run_<pid>`. Only the success paths
//! removed them: every failure leaves through `process::exit` from inside the
//! build, so a type error, a refusal or a failed link left the directory behind.
//! Test suites take those paths thousands of times, and on 2026-11-23 the
//! leftovers (with the suites' own) filled a 31 GB `/tmp` tmpfs: every link on
//! the machine then failed with "No space left on device". The directories are
//! now removed by an exit hook unless `NSL_KEEP_TEMP=1`.
//!
//! Each test points the temp dir at a directory of its own and lists it after
//! `nsl` exits.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A fresh directory to serve as `nsl`'s temp dir, and one for the program.
fn dirs(tag: &str) -> (PathBuf, PathBuf) {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("scratch_cleanup_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let tmp = base.join("tmp");
    let work = base.join("work");
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    (tmp, work)
}

fn nsl(tmp: &Path, work: &Path, args: &[&str], keep: bool) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(args)
        .current_dir(work)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        // `std::env::temp_dir()` reads TMPDIR on Unix and TMP/TEMP on Windows.
        .env("TMPDIR", tmp)
        .env("TMP", tmp)
        .env("TEMP", tmp)
        .env_remove("NSL_KEEP_TEMP");
    if keep {
        cmd.env("NSL_KEEP_TEMP", "1");
    }
    cmd.output().expect("spawn nsl")
}

/// The compiler's scratch directories left in `tmp`.
fn leftovers(tmp: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(tmp)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("nsl_build_") || n.starts_with("nsl_run_"))
        .collect();
    names.sort();
    names
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

const SINGLE: &str = "print(1)\n";
const MULTI: &str = "from nsl.math import clamp\n\nprint(clamp(2.0, 0.0, 1.0))\n";

/// The output path's directory does not exist, so the link fails after the
/// objects are written.
fn failed_link(src: &str, tag: &str, keep: bool) -> (PathBuf, Vec<String>) {
    let (tmp, work) = dirs(tag);
    std::fs::write(work.join("prog.nsl"), src).unwrap();
    let out = nsl(
        &tmp,
        &work,
        &["build", "prog.nsl", "-o", "no/such/dir/prog"],
        keep,
    );
    assert!(
        !out.status.success(),
        "the link was meant to fail:\n{}",
        stderr(&out)
    );
    assert!(
        stderr(&out).contains("link error"),
        "expected a link error:\n{}",
        stderr(&out)
    );
    let left = leftovers(&tmp);
    (tmp, left)
}

#[test]
fn a_failed_single_file_link_removes_its_scratch() {
    let (tmp, left) = failed_link(SINGLE, "single", false);
    assert!(left.is_empty(), "scratch left behind: {left:?}");
    let _ = std::fs::remove_dir_all(tmp.parent().unwrap());
}

#[test]
fn a_failed_multi_module_link_removes_its_scratch() {
    let (tmp, left) = failed_link(MULTI, "multi", false);
    assert!(left.is_empty(), "scratch left behind: {left:?}");
    let _ = std::fs::remove_dir_all(tmp.parent().unwrap());
}

#[test]
fn nsl_keep_temp_keeps_the_scratch_of_a_failed_link() {
    let (tmp, left) = failed_link(MULTI, "keep", true);
    assert_eq!(left.len(), 1, "expected the one build dir kept: {left:?}");
    let kept = tmp.join(&left[0]);
    let objects = std::fs::read_dir(&kept)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "o"))
        .count();
    assert!(
        objects > 0,
        "the kept dir {} holds no objects",
        kept.display()
    );
    let _ = std::fs::remove_dir_all(tmp.parent().unwrap());
}

#[test]
fn a_run_that_fails_to_compile_removes_its_scratch() {
    let (tmp, work) = dirs("run_type_error");
    // `nsl run` makes its directory before the build, which then exits on the
    // semantic error.
    std::fs::write(work.join("prog.nsl"), "print(no_such_name)\n").unwrap();
    let out = nsl(&tmp, &work, &["run", "prog.nsl"], false);
    assert!(
        !out.status.success(),
        "the build was meant to fail:\n{}",
        stderr(&out)
    );
    assert!(
        stderr(&out).contains("undefined variable"),
        "expected the semantic error:\n{}",
        stderr(&out)
    );
    let left = leftovers(&tmp);
    assert!(left.is_empty(), "scratch left behind: {left:?}");
    let _ = std::fs::remove_dir_all(tmp.parent().unwrap());
}

#[test]
fn emit_obj_of_a_multi_module_program_keeps_the_objects_it_reports() {
    let (tmp, work) = dirs("emit_obj");
    std::fs::write(work.join("prog.nsl"), MULTI).unwrap();
    let out = nsl(&tmp, &work, &["build", "--emit-obj", "prog.nsl"], false);
    assert!(
        out.status.success(),
        "nsl build --emit-obj failed:\n{}",
        stderr(&out)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let written: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("Wrote "))
        .collect();
    assert!(
        written.len() > 1,
        "expected one object per module:\n{stdout}"
    );
    for path in written {
        assert!(Path::new(path).is_file(), "{path} was reported but is gone");
    }
    let _ = std::fs::remove_dir_all(tmp.parent().unwrap());
}
