//! M55: End-to-end roundtrip tests for the ZK inference pipeline.
//!
//! These tests exercise the full pipeline: ZkDag -> lower_dag_to_zkir -> WitnessGenerator
//! -> Halo2Backend.compile -> setup -> prove -> verify.

mod roundtrip_tests;
