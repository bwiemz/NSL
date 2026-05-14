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
//! - `3` framework error (timer setup failed, etc.).
//!
//! Task B1.5-1 scaffolded the binary (CLI + output emitter + launch-harness
//! skeleton). Task B1.5-2 (this file's current state) fills in the gate
//! fixture registry, the real launch + CUDA-event timing, and wires
//! main() end-to-end.

// The bench binary is gated on both `cuda` (for cudarc + PTX synthesis) and
// `debug_kernel_instrumentation` (for the M3 skip-decision HBM writeback the
// measurement protocol relies on). When either is missing, the binary still
// compiles but exits 3 with an explanatory message — Cargo's
// `required-features` already prevents building it without the flags, but
// the cfg-gated fallback keeps `cargo check` clean on subsetted feature
// configurations.

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
use nsl_codegen::bin::bench::{cli, fixtures, launch, output};

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
fn main() {
    let args = match cli::Args::parse() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(1);
        }
    };

    // -- Step 1: Look up the fixture. --
    let fixture = match fixtures::lookup(&args.fixture) {
        Some(f) => f,
        None => {
            eprintln!("error: unknown fixture: {}", args.fixture);
            std::process::exit(1);
        }
    };

    // -- Step 2: Run end-to-end (PTX synth → module load → 100×launch with
    // CUDA events → skip-ratio readback). --
    let tier_b_on = args.tier_b == cli::TierB::On;
    let result = unsafe {
        launch::run_fixture(
            &fixture,
            tier_b_on,
            args.seed,
            args.iterations,
            args.dump_output.as_deref(),
        )
    };
    let result = match result {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("launch error: {msg}");
            std::process::exit(2);
        }
    };

    // -- Step 3: Emit the result line per §5.2. --
    let line = output::emit_result(&output::ResultLine {
        fixture: fixture.name.to_string(),
        tier_b_on,
        median_us: result.median_us,
        n: args.iterations,
        skip_ratio: result.skip_ratio,
        seed: args.seed,
    });
    println!("{line}");
    std::process::exit(0);
}

#[cfg(not(all(feature = "cuda", feature = "debug_kernel_instrumentation")))]
fn main() {
    eprintln!(
        "nsl-codegen-bench requires both --features cuda and --features debug_kernel_instrumentation"
    );
    std::process::exit(3);
}
