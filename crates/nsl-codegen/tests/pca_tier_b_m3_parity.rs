//! M3 estimator/runtime skip-mask parity test.
//!
//! For each fixture in spec §4.4: build the reference skip mask via
//! pca_tileskip::build, launch the Tier B kernel with instrumentation
//! enabled, read back the kernel's per-tile decisions, assert
//! bit-equality. Spec §4.3.
//!
//! The kernel-launch + readback harness is deferred to a Tier B.1.5
//! follow-up; the per-fixture tests below are #[ignore]'d until that
//! lands. The fixture matrix + reference-mask construction are testable
//! today and run via `cargo test ... -- fixture_matrix_constructs`.

mod fixtures {
    include!("fixtures/mod.rs");
}
use fixtures::{fixture_matrix, segment_ids_from_fixture, PackingFixture};

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

fn run_parity_for_fixture(_block_q: u32, _block_kv: u32, _fixture: &PackingFixture) {
    // Future Tier B.1.5: implement the launch + readback harness:
    //   1. Synthesize PTX via synthesize_flash_attention_ptx_v2_with_tier_b
    //      with the debug_kernel_instrumentation feature enabled.
    //   2. Allocate decisions buffer [batch=1, head=1, num_q_tiles, num_kv_tiles]:u8.
    //   3. Launch via cudarc with skip_decisions_ptr kernel param.
    //   4. Sync + memcpy_dtov the decisions buffer.
    //   5. Build reference mask via pca_tileskip::build(...).
    //   6. Assert bit-equality, with failure diagnostic naming
    //      the diverging (qt, kvt) coordinates.
    //
    // Reuses the launch helper shape from pca_tier_a_forward_correctness.rs
    // with the addition of the decisions buffer + arg.
    unimplemented!("M3 parity launch harness deferred to Tier B.1.5 follow-up — \
                   see pca_tier_a_forward_correctness::launch_forward for the harness shape; \
                   add skip_decisions buffer + arg + wire skip_decisions_ptr kernel param");
}

#[test]
#[ignore = "needs Tier B.1.5 launch harness + skip_decisions_ptr kernel param wiring"]
fn m3_parity_standard_3doc() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[0]);
}

#[test]
#[ignore = "needs Tier B.1.5 launch harness"]
fn m3_parity_long_seq_5doc() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[1]);
}

#[test]
#[ignore = "needs Tier B.1.5 launch harness"]
fn m3_parity_skewed_packing() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[2]);
}

#[test]
#[ignore = "needs Tier B.1.5 launch harness"]
fn m3_parity_boundary_dense() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[3]);
}

#[test]
#[ignore = "needs Tier B.1.5 launch harness"]
fn m3_parity_single_doc() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[4]);
}

#[test]
#[ignore = "needs Tier B.1.5 launch harness"]
fn m3_parity_tail_padding() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[5]);
}
