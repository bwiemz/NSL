//! `nsl build` / `nsl run` and the WRGA analysis commands.
//!
//! This is the tightly-coupled build cluster: the build paths (single/multi,
//! shared-lib, standalone), the `nsl run` execution path, and the WRGA
//! report/analysis helpers (which the build paths invoke). They share enough state that they live together in one
//! module. Shared frontend/config helpers remain in `main.rs` and are reached
//! via `crate::`.
//!
//! Previously a single ~2.3k-line `build.rs`; split into focused submodules
//! with no behavior change. The WRGA check-mode report helpers live in
//! [`reports`]; each build *flavor* gets its own file. The `--wrga-analyze /
//! --wrga-compare` overrides + plan-capture slot are carried explicitly on
//! `CompileOptions::wrga_check` (see `nsl_codegen::WrgaCheckContext`), not via
//! thread-locals. The `pub(crate)` re-exports below keep every existing
//! `crate::commands::build::<fn>` call site working unchanged.

mod normal;
mod options;
mod reports;
mod run;
mod shared_lib;
mod standalone;
mod wrga_check;

/// Remove a compile-scratch directory and everything in it.
///
/// Every temp-dir site used to end with `std::fs::remove_dir(&temp_dir)`, which
/// only succeeds on an EMPTY directory — and a build leaves `.o` files, PTX and
/// intermediate artifacts behind. The call is `let _ = ...`, so the failure was
/// silent and the directory survived with all its contents.
///
/// On this developer's machine that had accumulated **8781 leftover
/// `/tmp/nsl_*` directories totalling 29 GB**. Because `/tmp` is tmpfs there,
/// a full `cargo test --workspace` eventually exhausted it and `ld` began
/// failing with "final link failed: No space left on device" — surfacing as
/// 15 test failures that read exactly like a codegen regression.
///
/// Set `NSL_KEEP_TEMP=1` to retain the directory for debugging.
pub(crate) fn cleanup_temp_dir(dir: &std::path::Path) {
    // This path has dealt with the directory, so the exit hook need not.
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        dirs.retain(|d| d != dir);
    }
    remove_unless_kept(dir);
}

fn remove_unless_kept(dir: &std::path::Path) {
    if std::env::var("NSL_KEEP_TEMP").as_deref() == Ok("1") {
        nsl_log::nsl_log!(INFO, "nsl", "[nsl] NSL_KEEP_TEMP=1 — retaining {}", dir.display());
        return;
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Scratch directories this process made and no path has cleaned up yet.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());

/// Create the compile-scratch directory `<temp dir>/<name>`, and remove it when
/// the process exits unless a path has already called [`cleanup_temp_dir`] on it.
///
/// Only the success paths reach `cleanup_temp_dir`. A type error, a refused
/// flag, a codegen error or a failed link leaves through `process::exit` from
/// deep inside the build, past it, and the directory stayed behind with its
/// objects (and, for `nsl run`, a 100-270 MB unstripped executable). Test suites
/// exercise those paths by the thousand: on 2026-11-23 4,798 `nsl_build_<pid>`
/// directories (5.4 GB) sat in a 31 GB `/tmp` tmpfs that test leftovers had
/// filled, and every link on the machine failed. `process::exit` runs the C
/// `atexit` handlers, so registering the directory once covers every exit.
/// A process killed by a signal still leaves it.
pub(crate) fn create_scratch_dir(name: &str) -> std::io::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir)?;
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        unsafe extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        // SAFETY: `atexit` (C, available on glibc and MSVCRT/UCRT, as in
        // nsl-runtime's args.rs) takes a plain `extern "C" fn()`, and
        // `remove_scratch_dirs_at_exit` never unwinds.
        unsafe {
            atexit(remove_scratch_dirs_at_exit);
        }
    });
    if let Ok(mut dirs) = SCRATCH_DIRS.lock()
        && !dirs.contains(&dir)
    {
        dirs.push(dir.clone());
    }
    Ok(dir)
}

/// Exempt a scratch directory from removal at exit: it holds the command's
/// output (`nsl build --emit-obj` of a multi-module program reports objects there).
pub(crate) fn keep_scratch_dir(dir: &std::path::Path) {
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        dirs.retain(|d| d != dir);
    }
}

extern "C" fn remove_scratch_dirs_at_exit() {
    // An unwind out of an `extern "C"` fn aborts; a failure to clean up must not.
    let _ = std::panic::catch_unwind(|| {
        // `try_lock`: a thread holding the lock while the process exits must not hang the exit.
        let Ok(dirs) = SCRATCH_DIRS.try_lock() else {
            return;
        };
        for dir in dirs.iter() {
            remove_unless_kept(dir);
        }
    });
}

pub(crate) use normal::{run_build, run_build_inner};
pub(crate) use options::dispatch;
pub(crate) use run::{build_to_temp, execute_temp_build, run_run};
pub(crate) use shared_lib::run_build_shared;
pub(crate) use standalone::run_build_standalone;
pub(crate) use wrga_check::{run_check_wrga_analyze, run_check_wrga_compare};

/// Item 2: print the pass execution trace, once, at a point where compilation
/// has finished.
///
/// Idempotent — the first call prints, later ones are silent — so it can be
/// placed at EVERY terminal point without a build that passes through two of
/// them reporting twice. That matters: there are **13 call sites across 6
/// files** (`check.rs` x3, `test.rs`, `build/normal.rs` x4,
/// `build/options.rs` x2, `build/shared_lib.rs` x2,
/// `build/standalone.rs`), because "compilation finished" is not one place —
/// the link flavours, `--emit-obj` (which returns before any link), and
/// `check`/`test` (which never link at all) are all separate exits. An
/// emitter wired to only some of them reports nothing on the others, which is
/// precisely the "silently inert" failure the trace exists to detect.
///
/// The count was written as "seven" here from the first commit and was wrong
/// on the day it was written; it is spelled out per file above so the next
/// reader can check it in one `grep` rather than trusting the number.
pub fn emit_pass_trace() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    if !nsl_codegen::pass_trace::enabled() {
        return;
    }
    if DONE.swap(true, Ordering::SeqCst) {
        return;
    }
    eprint!("{}", nsl_codegen::pass_trace::report());
    // Step 4: the bus report goes out with the trace, from the same emitter
    // and under the same env var, because its `SILENT DEFAULT` finding is
    // computed by cross-referencing the two — a channel read while empty is
    // only a finding once you know its producing pass ran. Two emitters could
    // reach different terminal paths and print half the picture.
    eprint!("{}", nsl_codegen::pass_bus::report());
}
