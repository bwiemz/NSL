//! d_model_chunk selection per variant + supported-matrix validation.
//! Per spec section 3.4 budget formula. Real implementation in B1.3.

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug)]
pub struct DowngradeReason(pub String);

/// Select d_model_chunk for the given config; returns Err with downgrade
/// reason if no chunk fits the SMEM budget.
///
/// Stub for B1.2: always returns Err so the dispatch safety-net fallthrough
/// is exercised in tests. B1.3 implements the real descending search.
///
/// TODO(B1.3): the descending search MUST bottom out at
/// `crate::csha_pipeline::CHUNK_PLANNER_FLOOR` (currently `16`). The planner
/// in `csha_pipeline::pipeline_smem_bytes` sizes chunk staging against that
/// floor; if `select` ever returns a smaller chunk, the planner over-admits
/// configs whose actual emission won't fit. Concretely, the real search
/// will iterate over `[128, 64, 32, csha_pipeline::CHUNK_PLANNER_FLOOR as u32]`
/// and call `smem_layout::validate_tier_b1_config` for each candidate.
pub fn select(config: &FlashAttentionConfig) -> Result<u32, DowngradeReason> {
    let _ = config;
    Err(DowngradeReason(
        "B1.2 stub: chunk_config::select not yet implemented; B1.3 lands the real search".to_string()
    ))
}
