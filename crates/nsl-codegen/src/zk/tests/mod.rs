//! M55: End-to-end roundtrip tests for the ZK inference pipeline.
//!
//! These tests exercise the full pipeline: ZkDag -> lower_model_for_folding
//! -> WitnessGenerator -> FoldingProver.fold_layer -> finalize -> verify.

mod roundtrip_tests;
