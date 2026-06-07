//! §4.3 attention sinks — Sprint 1a cycle-7 precursor: effective_block_kv
//! indirection so multiple consumer sites in the v2 forward emitter agree
//! on the same row count.
//!
//! Today (`num_sink_tokens == 0` is the only flowing value post-cycle-5
//! refusal): `effective_block_kv(config) == config.block_kv`. The indirection
//! is byte-identical in emitted PTX vs pre-Sprint-1a.
//!
//! Sprint 1b lifts the cycle-5 refusal for the narrow Tier A forward
//! single-tile config and `effective_block_kv` then returns
//! `block_kv + num_sink_tokens`. Subsequent sprints (multi-tile + causal,
//! GQA, RoPE) build on this foundation.

use crate::flash_attention::FlashAttentionConfig;

/// The effective number of KV rows present in the SMEM KV slab. Equal to
/// `config.block_kv` when sinks are disabled (`num_sink_tokens == 0` — the
/// only currently-flowing value post-cycle-5 refusal). When sinks are
/// enabled in Sprint 1b: `block_kv + num_sink_tokens` for the v1 single-tile
/// case. Consumers MUST read this helper rather than the raw `block_kv`
/// to keep the load bound, S-store stride, softmax chunks, PV loop bound,
/// and SP scratch bytes in lockstep.
#[inline]
pub fn effective_block_kv(config: &FlashAttentionConfig) -> i64 {
    config.block_kv + (config.num_sink_tokens as i64)
}

/// Byte cost of the persistent sink slab at the front of the KV region.
/// Returns 0 when sinks are disabled.
#[inline]
pub fn sink_slab_bytes(config: &FlashAttentionConfig) -> usize {
    (config.num_sink_tokens as usize) * (config.head_dim as usize) * 2 /* f16 bytes */
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

    fn cfg(block_kv: i64, num_sink_tokens: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv, head_dim: 64,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::Adjacent, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens,
            gpu_sm: 80, segment_masked: false, csha: None,
        }
    }

    #[test]
    fn effective_block_kv_byte_identity_at_zero_sinks() {
        // Sprint 1a invariant: at num_sink_tokens=0 (the only flowing value
        // post-cycle-5 refusal), effective_block_kv MUST equal block_kv.
        for bkv in [16, 32, 64, 128, 256_i64] {
            assert_eq!(effective_block_kv(&cfg(bkv, 0)), bkv);
        }
    }

    #[test]
    fn effective_block_kv_extends_with_sinks() {
        // Sprint 1b will lift the cycle-5 refusal for narrow configs and
        // this is the value those sites will see.
        assert_eq!(effective_block_kv(&cfg(64, 4)), 68);
        assert_eq!(effective_block_kv(&cfg(128, 8)), 136);
    }

    #[test]
    fn sink_slab_bytes_zero_when_disabled() {
        assert_eq!(sink_slab_bytes(&cfg(64, 0)), 0);
    }

    #[test]
    fn sink_slab_bytes_is_f16_per_row_per_dim() {
        // num_sink × head_dim × 2 (f16)
        assert_eq!(sink_slab_bytes(&cfg(64, 4)), 4 * 64 * 2);
        assert_eq!(sink_slab_bytes(&cfg(64, 8)), 8 * 64 * 2);
    }
}
