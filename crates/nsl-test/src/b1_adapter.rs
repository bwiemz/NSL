//! CPU-side B.1 -> row-major adapter (Phase 2.6 T6+T7).
//!
//! Per V-Phase-2.6-A (audit) + the B.1 save-activation retrofit
//! (commits 7c00ae1c / cda47e40 / 9e2148f1) + T2.7 GPU validation: B.1 writes
//! ALL save-activation tensors in row-major layout, byte-identical to the
//! dQ-kernel's read convention. So this adapter is IDENTITY passthrough --
//! it validates lengths and moves the buffers into a ForwardOutputs.
//!
//! (The original Phase 2.6 plan anticipated a chunked B.1 layout requiring
//! an index remap; the retrofit targets row-major directly, so no remap is
//! needed. If a future B.1 layout change reintroduces chunking, the per-tensor
//! remap would live here.)

use half::f16;
use crate::cpu_naive_forward::ForwardOutputs;

/// Raw d2h bytes from B.1's `_with_saves` slots, row-major (post-retrofit).
pub struct B1Saves {
    pub q_proj:  Vec<f16>,
    pub k_proj:  Vec<f16>,
    pub v_proj:  Vec<f16>,
    pub row_max: Vec<f32>,
    pub row_sum: Vec<f32>,
    pub o:       Vec<f16>,
}

/// Move B.1's row-major saves into a ForwardOutputs (identity passthrough,
/// length-validated). `batch/heads/seq/hd` give the expected lengths.
pub fn reshape_b1_saves_to_row_major(
    saves: B1Saves,
    batch: usize, heads: usize, seq: usize, hd: usize,
) -> ForwardOutputs {
    let proj_len = batch * heads * seq * hd;
    let stat_len = batch * heads * seq;
    assert_eq!(saves.q_proj.len(), proj_len, "q_proj length: got {} expected {}", saves.q_proj.len(), proj_len);
    assert_eq!(saves.k_proj.len(), proj_len, "k_proj length: got {} expected {}", saves.k_proj.len(), proj_len);
    assert_eq!(saves.v_proj.len(), proj_len, "v_proj length: got {} expected {}", saves.v_proj.len(), proj_len);
    assert_eq!(saves.row_max.len(), stat_len, "row_max length: got {} expected {}", saves.row_max.len(), stat_len);
    assert_eq!(saves.row_sum.len(), stat_len, "row_sum length: got {} expected {}", saves.row_sum.len(), stat_len);
    assert_eq!(saves.o.len(), proj_len, "o length: got {} expected {}", saves.o.len(), proj_len);
    ForwardOutputs {
        q_saved: saves.q_proj,
        k_saved: saves.k_proj,
        v_saved: saves.v_proj,
        row_max: saves.row_max,
        row_sum: saves.row_sum,
        o:       saves.o,
    }
}
