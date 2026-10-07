//! Where `nsl build` puts object files.
//!
//! - `--emit-obj -o X` writes the object to X. It used to be ignored: the
//!   object went to `<stem>.o` beside the source whatever `-o` said.
//! - `--emit-obj` alone still writes `<stem>.o` beside the source.
//! - A linked build's object is an intermediate. It used to be written
//!   beside the source too, where a failed link left it and two builds of the
//!   same file raced on one path; it now lives in the build's scratch dir.
//! - A program that imports modules emits one object per module, so `-o`
//!   cannot apply. It is no longer ignored silently: the build says so.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn nsl_build(dir: &Path, args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .arg("build")
        .args(args)
        .output()
        .expect("nsl runs")
}

fn assert_ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what}: nsl build failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// A program with no imports: the single-module build path.
fn single_module(dir: &Path) -> PathBuf {
    let src = dir.join("prog.nsl");
    std::fs::write(&src, "let x = 40 + 2\nprint(x)\n").unwrap();
    src
}

fn objects_beside(src: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(src.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "o" || x == "obj"))
        .collect()
}

#[test]
fn emit_obj_writes_the_object_where_dash_o_says() {
    let tmp = tempfile::tempdir().unwrap();
    let src = single_module(tmp.path());
    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let target = out_dir.join("custom_name.o");

    let out = nsl_build(tmp.path(), &[src.as_os_str(), "--emit-obj".as_ref(), "-o".as_ref(), target.as_os_str()]);
    assert_ok(&out, "--emit-obj -o");
    assert!(target.is_file(), "the object must be at -o's path {}", target.display());
    assert!(
        objects_beside(&src).is_empty(),
        "nothing may be written beside the source: {:?}",
        objects_beside(&src)
    );
}

#[test]
fn emit_obj_without_dash_o_writes_stem_dot_o_beside_the_source() {
    let tmp = tempfile::tempdir().unwrap();
    let src = single_module(tmp.path());
    let out = nsl_build(tmp.path(), &[src.as_os_str(), "--emit-obj".as_ref()]);
    assert_ok(&out, "--emit-obj");
    assert!(tmp.path().join("prog.o").is_file(), "documented default: <stem>.o beside the source");
}

#[test]
fn a_failed_link_leaves_no_object_beside_the_source() {
    // A successful link deletes the intermediate wherever it is, so only a
    // failed one shows where it was written: point -o into a directory that
    // does not exist.
    let tmp = tempfile::tempdir().unwrap();
    let src = single_module(tmp.path());
    let exe = tmp.path().join("no_such_dir").join("prog_exe");
    let out = nsl_build(tmp.path(), &[src.as_os_str(), "-o".as_ref(), exe.as_os_str()]);
    assert!(!out.status.success(), "linking into a missing directory must fail");
    assert!(
        objects_beside(&src).is_empty(),
        "the link intermediate must not be left in the source tree: {:?}",
        objects_beside(&src)
    );
}

#[test]
fn a_linked_build_writes_the_executable_where_dash_o_says() {
    let tmp = tempfile::tempdir().unwrap();
    let src = single_module(tmp.path());
    let exe = tmp.path().join(if cfg!(windows) { "prog_exe.exe" } else { "prog_exe" });
    let out = nsl_build(tmp.path(), &[src.as_os_str(), "-o".as_ref(), exe.as_os_str()]);
    assert_ok(&out, "linked build");
    assert!(exe.is_file(), "the executable must be at -o's path");
    assert!(objects_beside(&src).is_empty(), "{:?}", objects_beside(&src));
}

#[test]
fn multi_module_emit_obj_says_dash_o_does_not_apply() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("helper.nsl"), "fn add_two(x: int) -> int:\n    return x + 2\n").unwrap();
    let src = tmp.path().join("main.nsl");
    std::fs::write(&src, "from helper import add_two\nprint(add_two(40))\n").unwrap();
    let target = tmp.path().join("one.o");

    let out = nsl_build(tmp.path(), &[src.as_os_str(), "--emit-obj".as_ref(), "-o".as_ref(), target.as_os_str()]);
    assert_ok(&out, "multi-module --emit-obj -o");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("is not applied with --emit-obj"), "stderr:\n{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let wrote: Vec<&str> = stdout.lines().filter_map(|l| l.strip_prefix("Wrote ")).collect();
    assert!(wrote.len() >= 2, "one object per module, each printed: {stdout}");
    for path in &wrote {
        assert!(Path::new(path).is_file(), "printed object {path} must exist");
    }
}
