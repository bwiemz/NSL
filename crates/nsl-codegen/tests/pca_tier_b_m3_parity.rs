//! M3 estimator/runtime skip-mask parity test.
//!
//! Spec §6.1 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`:
//! For each of the six existing `PackingFixture` entries, invoke the bench
//! binary twice (once with `--tier-b on`, once with `--tier-b off`, same seed)
//! and assert the captured forward `O` outputs are **byte-identical**.
//!
//! Bit-identical (not tolerance-bounded) is the correctness-preserving
//! guarantee: skipped tiles contribute exactly zero to softmax / PV
//! accumulation, so Tier-B-on and Tier-B-off must agree to the last bit.
//! Any divergence is a real bug, not a tolerance issue — stop and root
//! cause; do not relax to `assert_relative_eq!`.
//!
//! The bench binary's `--dump-output <path>` flag captures the `O` tensor
//! to disk after a single timed iteration. The test driver below invokes
//! the bench as a subprocess (via `env!("CARGO_BIN_EXE_bench")`) twice per
//! fixture and diffs the two files at the byte level.
//!
//! Build prerequisite: the bench binary is gated on `cuda` and
//! `debug_kernel_instrumentation` features. Run via:
//!
//! ```text
//! cargo test -p nsl-codegen --features "cuda debug_kernel_instrumentation" \
//!     --test pca_tier_b_m3_parity --release
//! ```

mod fixtures {
    include!("fixtures/mod.rs");
}
use fixtures::{fixture_matrix, segment_ids_from_fixture};

#[test]
fn fixture_matrix_constructs_six_fixtures() {
    let m = fixture_matrix();
    assert_eq!(m.len(), 6);
    let names: Vec<&str> = m.iter().map(|f| f.name).collect();
    assert!(names.contains(&"standard_3doc"));
    assert!(names.contains(&"long_seq_5doc"));
    assert!(names.contains(&"skewed_packing"));
    assert!(names.contains(&"boundary_dense"));
    assert!(names.contains(&"single_doc"));
    assert!(names.contains(&"tail_padding"));
}

#[test]
fn standard_3doc_segment_ids_have_three_documents() {
    let f = &fixture_matrix()[0];
    let ids = segment_ids_from_fixture(f);
    assert_eq!(ids.len(), 4096);
    assert_eq!(ids[0], 0);
    assert_eq!(ids[1365], 0);
    assert_eq!(ids[1366], 1);
    assert_eq!(ids[2731], 1);
    assert_eq!(ids[2732], 2);
    assert_eq!(ids[4095], 2);
}

#[test]
fn tail_padding_has_padding_sentinel() {
    let f = &fixture_matrix()[5];
    assert_eq!(f.name, "tail_padding");
    let ids = segment_ids_from_fixture(f);
    assert_eq!(ids[0], 0); // doc 0 starts at 0
    assert_eq!(ids[1024], 1); // doc 1 starts at 1024
    assert_eq!(ids[2048], u16::MAX); // padding starts at 2048
    assert_eq!(ids[4095], u16::MAX); // padding ends at 4095
}

#[test]
fn single_doc_all_same_segment() {
    let f = &fixture_matrix()[4];
    let ids = segment_ids_from_fixture(f);
    assert!(ids.iter().all(|&id| id == 0), "single_doc should be entirely doc 0");
}

// ── Bit-identical assertion harness (spec §6.1) ──────────────────────────
//
// Each test below invokes the bench binary as a subprocess with `--dump-output
// <path>` twice — once with `--tier-b on`, once with `--tier-b off` — using
// the same seed. The captured O tensor bytes MUST match exactly. The driver
// is `#[cfg]`-gated on `debug_kernel_instrumentation` because the bench
// binary requires that feature (see Cargo.toml `required-features`).

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
fn run_bench_capture_output_tensor(fixture: &str, tier_b: &str, seed: u64) -> Vec<u8> {
    let output_path = std::env::temp_dir()
        .join(format!("nsl_bench_out_{}_{}_{}.bin", fixture, tier_b, seed));
    let bench_bin = env!("CARGO_BIN_EXE_bench");
    let status = std::process::Command::new(bench_bin)
        .args([
            "--fixture",
            fixture,
            "--tier-b",
            tier_b,
            "--seed",
            &seed.to_string(),
            "--iterations",
            "1",
            "--dump-output",
            &output_path.to_string_lossy(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn bench binary {bench_bin:?}: {e}"));
    assert!(
        status.success(),
        "bench failed for fixture={fixture} tier_b={tier_b} (exit code = {:?})",
        status.code()
    );
    std::fs::read(&output_path).unwrap_or_else(|e| {
        panic!("failed to read --dump-output file {:?}: {e}", output_path)
    })
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
fn assert_parity_for_fixture(fixture: &str) {
    let seed: u64 = 42;
    let on = run_bench_capture_output_tensor(fixture, "on", seed);
    let off = run_bench_capture_output_tensor(fixture, "off", seed);
    assert_eq!(
        on.len(),
        off.len(),
        "Tier-B-on and Tier-B-off output sizes differ for fixture={fixture}: \
         on={} bytes, off={} bytes",
        on.len(),
        off.len()
    );
    if on != off {
        let mismatches: Vec<usize> = on
            .iter()
            .zip(off.iter())
            .enumerate()
            .filter_map(|(i, (a, b))| if a != b { Some(i) } else { None })
            .take(8)
            .collect();
        panic!(
            "Tier B output bit-differs from Tier-B-off on fixture={fixture}: \
             {} byte(s) mismatch; first offsets={:?} \
             (skip logic is no longer correctness-preserving — see spec §6.1)",
            on.iter().zip(off.iter()).filter(|(a, b)| a != b).count(),
            mismatches
        );
    }
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
#[test]
fn m3_parity_standard_3doc() {
    assert_parity_for_fixture("parity_1");
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
#[test]
fn m3_parity_long_seq_5doc() {
    assert_parity_for_fixture("parity_2");
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
#[test]
fn m3_parity_skewed_packing() {
    assert_parity_for_fixture("parity_3");
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
#[test]
fn m3_parity_boundary_dense() {
    assert_parity_for_fixture("parity_4");
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
#[test]
fn m3_parity_single_doc() {
    assert_parity_for_fixture("parity_5");
}

#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
#[test]
fn m3_parity_tail_padding() {
    assert_parity_for_fixture("parity_6");
}
