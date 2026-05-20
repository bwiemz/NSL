//! B1.1 cost-model baseline snapshot.
//!
//! This file locks down the output of `pipeline_smem_bytes` and
//! `fused_hbm_bytes` at a canonical small_shape. Commit 1 establishes the
//! **pre-correction** baseline; commit 2 (the actual B1.1 fix) re-baselines
//! these snapshots and the diff in `cargo insta review` becomes the visible
//! footprint of the V3 audit findings.
//!
//! See `docs/superpowers/specs/2026-05-11-tier-b1-v3-cost-model-audit.md`
//! for the per-bug derivation table.

use insta::assert_snapshot;
use nsl_codegen::csha_pipeline::{fused_hbm_bytes, pipeline_smem_bytes, FusionLevel, TileConfig};
use nsl_codegen::wggo_cost::LayerShape;

fn small_shape() -> LayerShape {
    LayerShape {
        batch: 1,
        seq: 1024,
        d_model: 512,
        head_dim: 64,
        n_kv_heads: 4,
        dtype_bytes: 2,
    }
}

fn canonical_tiles() -> TileConfig {
    TileConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
    }
}

#[test]
fn pipeline_smem_bytes_canonical() {
    let bytes = pipeline_smem_bytes(small_shape(), canonical_tiles());
    assert_snapshot!(bytes.to_string());
}

#[test]
fn fused_hbm_pipeline_canonical() {
    let bytes = fused_hbm_bytes(small_shape(), FusionLevel::Pipeline);
    assert_snapshot!(bytes.to_string());
}

#[test]
fn fused_hbm_block_canonical() {
    let bytes = fused_hbm_bytes(small_shape(), FusionLevel::Block);
    assert_snapshot!(bytes.to_string());
}
