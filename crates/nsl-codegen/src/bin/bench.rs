//! `nsl-codegen-bench` — measurement harness for PCA Tier B.
//!
//! Invocation contract pinned by §5.2 of
//! `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`:
//!
//! ```text
//! nsl-codegen-bench --fixture <name> --tier-b={on,off} \
//!     [--emit-time-only] [--seed <u64>] [--iterations <n>]
//! ```
//!
//! Exit codes:
//! - `0` measurement succeeded; `tier_b_bench_result:` line emitted to stdout.
//! - `1` CLI error / unknown fixture. Message to stderr, no output line.
//! - `2` CUDA / kernel launch error. Message + diagnostic to stderr.
//! - `3` framework error (timer setup failed, fixture loading not yet
//!   implemented, etc.).
//!
//! Task B1.5-1 scaffolds the binary (CLI + output emitter + launch-harness
//! skeleton). Fixture loading and the real kernel launch land in B1.5-2.

// The bench binary is gated on both `cuda` (for cudarc + PTX synthesis) and
// `debug_kernel_instrumentation` (for the M3 skip-decision HBM writeback the
// measurement protocol relies on). When either is missing, the binary still
// compiles but exits 3 with an explanatory message — Cargo's
// `required-features` already prevents building it without the flags, but
// the cfg-gated fallback keeps `cargo check` clean on subsetted feature
// configurations.

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
use nsl_codegen::bin::bench::cli;

// `launch` and `output` are reached via fully-qualified paths in B1.5-2's
// main-body code; importing them here would currently flag unused. The lib
// re-export keeps them callable as `nsl_codegen::bin::bench::{launch,output}`.

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
fn main() {
    let args = match cli::Args::parse() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(1);
        }
    };

    // Task B1.5-2 fills in fixture loading + PTX module load + launch +
    // skip-ratio readback + result-line emission here. For now the binary
    // exits 3 (framework not ready) so shell scripts that wire this up
    // early get a clear, deterministic signal.
    eprintln!("bench scaffolding complete; fixture loading not yet implemented");
    eprintln!(
        "invoked: fixture={} tier_b={:?} seed={} iterations={} emit_time_only={}",
        args.fixture, args.tier_b, args.seed, args.iterations, args.emit_time_only
    );
    std::process::exit(3);
}

#[cfg(not(all(feature = "cuda", feature = "debug_kernel_instrumentation")))]
fn main() {
    eprintln!(
        "nsl-codegen-bench requires both --features cuda and --features debug_kernel_instrumentation"
    );
    std::process::exit(3);
}
