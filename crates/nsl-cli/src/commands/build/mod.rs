//! `nsl build` / `nsl run` / `nsl zk` and the WRGA analysis commands.
//!
//! This is the tightly-coupled build cluster: the build paths (single/multi,
//! shared-lib, standalone, ZK), the `nsl run` execution path, the `nsl zk`
//! verify/stats subcommands, and the WRGA report/analysis helpers (which the
//! build paths invoke). They share enough state that they live together in one
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
mod zk;

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
    if std::env::var("NSL_KEEP_TEMP").as_deref() == Ok("1") {
        eprintln!("[nsl] NSL_KEEP_TEMP=1 — retaining {}", dir.display());
        return;
    }
    let _ = std::fs::remove_dir_all(dir);
}

pub(crate) use normal::{run_build, run_build_inner};
pub(crate) use options::dispatch;
pub(crate) use run::{build_to_temp, execute_temp_build, run_run};
pub(crate) use shared_lib::run_build_shared;
pub(crate) use standalone::run_build_standalone;
pub(crate) use wrga_check::{run_check_wrga_analyze, run_check_wrga_compare};
pub(crate) use zk::{run_build_zk, run_zk_cmd};
