//! Register accounting for the v2 scalar emitter. Used at compile time
//! to catch configs that would exceed sm_75's 255-register cap.

use crate::flash_attention::FlashAttentionConfig;

/// Maximum registers per thread on sm_75 (the oldest SM v2 targets).
pub const SM75_REGISTER_CAP: u32 = 255;

/// Counts registers per thread for `config`. Populated in Task 3.
pub fn count_registers(_config: &FlashAttentionConfig) -> u32 {
    0
}
