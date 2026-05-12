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
pub fn select(config: &FlashAttentionConfig) -> Result<u32, DowngradeReason> {
    let _ = config;
    Err(DowngradeReason(
        "B1.2 stub: chunk_config::select not yet implemented; B1.3 lands the real search".to_string()
    ))
}
