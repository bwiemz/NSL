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
//! with no behavior change. The cross-cutting glue (WRGA check-mode
//! thread-locals, report helpers) lives in [`wrga_state`] / [`reports`]; each
//! build *flavor* gets its own file. The `pub(crate)` re-exports below keep
//! every existing `crate::commands::build::<fn>` call site working unchanged.

mod normal;
mod options;
mod reports;
mod run;
mod shared_lib;
mod standalone;
mod wrga_check;
mod wrga_state;
mod zk;

pub(crate) use normal::{run_build, run_build_inner};
pub(crate) use options::dispatch;
pub(crate) use run::run_run;
pub(crate) use shared_lib::run_build_shared;
pub(crate) use standalone::run_build_standalone;
pub(crate) use wrga_check::{run_check_wrga_analyze, run_check_wrga_compare};
pub(crate) use wrga_state::apply_wrga_check_overrides;
pub(crate) use zk::{run_build_zk, run_zk_cmd};
