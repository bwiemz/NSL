//! Register accounting for the v2 scalar emitter. Used at compile time
//! to catch configs that would exceed sm_75's 255-register cap.

use crate::flash_attention::FlashAttentionConfig;

/// Maximum registers per thread on sm_75 (the oldest SM v2 targets).
pub const SM75_REGISTER_CAP: u32 = 255;

/// Counts registers per thread for `config`.
///
/// Matches spec Section 1's register table:
///   Q row slice:           head_dim / 32 regs (one f32 per d slice)
///   S dot-product scratch: 1 reg (current-k accumulator)
///   O_acc:                 head_dim / 32 regs (one f32 per d slice)
///   Softmax state:         5 regs (row_max, row_sum, correction, old_max, new_max)
///   Loop counters/scratch: ~10 regs
///   RoPE extras (if rope_q): 4 regs (q_a, q_b, cos, sin)
pub fn count_registers(config: &FlashAttentionConfig) -> u32 {
    let q_row      = (config.head_dim / 32) as u32;
    let s_scratch  = 1;
    let o_acc      = (config.head_dim / 32) as u32;
    let softmax    = 5;
    let scratch    = 10;
    let rope_extra = if config.rope_q { 4 } else { 0 };
    q_row + s_scratch + o_acc + softmax + scratch + rope_extra
}
