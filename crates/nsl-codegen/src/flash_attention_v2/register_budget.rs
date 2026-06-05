//! Register accounting for the v2 scalar emitter. Used at compile time
//! to catch configs that would exceed sm_75's 255-register cap.

use crate::flash_attention::FlashAttentionConfig;
#[cfg(test)]
use crate::flash_attention::RopeStyle;

/// Maximum registers per thread on sm_75 (the oldest SM v2 targets).
pub const SM75_REGISTER_CAP: u32 = 255;

/// Counts registers per thread for `config`.
///
/// Matches spec Section 1's register table:
///   Q row slice:             head_dim / 32 regs (one f32 per d slice)
///   S dot-product scratch:   1 reg (current-k accumulator)
///   O_acc:                   head_dim / 32 regs (one f32 per d slice)
///   Softmax state:           5 regs (row_max, row_sum, correction, old_max, new_max)
///   Loop counters/scratch:   ~10 regs
///   RoPE extras (if rope_q): 4 regs (q_a, q_b, cos, sin)
///   PCA segment (if segment_masked): 3 GPR scratch regs + 1 pred
///                                    reg (ptxas-tallied separately)
pub fn count_registers(config: &FlashAttentionConfig) -> u32 {
    let q_row         = (config.head_dim / 32) as u32;
    let s_scratch     = 1;
    let o_acc         = (config.head_dim / 32) as u32;
    let softmax       = 5;
    let scratch       = 10;
    let rope_extra    = if config.rope_q { 4 } else { 0 };
    // PCA Tier A (spec §6.4): helper adds 3 scratch GPRs + 1 pred
    // chain. Predicate regs are tallied separately by ptxas; here
    // we only count GPR pressure.
    let segment_extra = if config.segment_masked { 3 } else { 0 };
    q_row + s_scratch + o_acc + softmax + scratch + rope_extra + segment_extra
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_masked_budget_has_headroom_at_max_config() {
        let cfg = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: true,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: true,
            csha: None,
        };
        let used = count_registers(&cfg);
        assert!(
            used < SM75_REGISTER_CAP,
            "segment_masked at max config uses {used} regs, exceeds SM75 cap {SM75_REGISTER_CAP}"
        );
        // Spec §6.4 pinned headroom: ≥224 regs free at max config.
        // (count_registers sums: 4+1+4+5+10+4+3 = 31; 255-31 = 224 exactly.)
        assert!(
            SM75_REGISTER_CAP - used >= 224,
            "segment_masked max-config headroom below pinned value (spec §6.4): \
             used={used}, headroom={}",
            SM75_REGISTER_CAP - used
        );
    }
}
