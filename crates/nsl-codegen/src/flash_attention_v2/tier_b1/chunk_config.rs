//! d_model_chunk selection per variant + supported-matrix validation.
//! Per spec section 3.4 budget formula.

use crate::csha_pipeline::CHUNK_PLANNER_FLOOR;
use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::validate_tier_b1_config;

#[derive(Debug)]
pub struct DowngradeReason(pub String);

/// Candidate chunk sizes for the descending search. The FLOOR end MUST
/// match `csha_pipeline::CHUNK_PLANNER_FLOOR` — the planner's
/// `pipeline_smem_bytes` sizes chunk staging against that constant; if
/// `select` ever returned a smaller chunk, the planner would over-admit
/// configs whose actual emission wouldn't fit.
const CANDIDATE_CHUNKS: &[u32] = &[128, 64, 32, CHUNK_PLANNER_FLOOR as u32];

/// Select d_model_chunk for the given config. Returns Err with downgrade
/// reason if no chunk in CANDIDATE_CHUNKS fits the SMEM budget for this
/// (bq, bkv, hd, dm) tuple.
///
/// Per spec §3.4: descending search over `{128, 64, 32, FLOOR}`. Each
/// candidate is validated by `smem_layout::validate_tier_b1_config`,
/// which computes `q_tile + 4*(bkv*hd*2) + sp_scratch + chunk_staging`
/// and compares against `SMEM_DYNAMIC_BUDGET_BYTES` (99 KB on Blackwell).
///
/// After SMEM passes, `register_budget::analyze` (spec §5.4) runs once
/// per (bq, bkv, hd) — register pressure is chunk-independent, so its
/// failure short-circuits the search.
pub fn select(config: &FlashAttentionConfig) -> Result<u32, DowngradeReason> {
    // Reject configs that aren't Tier B.1-eligible (no CSHA extras or
    // d_model is zero/unset).
    let d_model = config.csha.as_ref().map(|c| c.d_model).unwrap_or(0);
    if d_model == 0 {
        return Err(DowngradeReason(
            "csha.d_model is zero or unset; cannot size chunk staging".to_string(),
        ));
    }

    // Walk the candidate list. First chunk that fits is the selected one.
    let mut last_err: Option<String> = None;
    for &chunk in CANDIDATE_CHUNKS {
        // Skip candidates larger than d_model — would over-load the
        // chunk-streaming inner loop with non-existent W rows.
        if chunk > d_model {
            continue;
        }
        // 1. SMEM budget check.
        if let Err(e) = validate_tier_b1_config(config, chunk) {
            last_err = Some(e.0);
            continue;
        }
        // 2. Register budget check (independent of chunk — depends only
        //    on bq/bkv/hd. Run AFTER SMEM passes to avoid the surface
        //    where SMEM was the proximate issue.)
        if let Err(spill) = crate::flash_attention_v2::tier_b1::register_budget::analyze(config) {
            last_err = Some(format!("chunk={} fits SMEM but {}", chunk, spill.0));
            // Register pressure is chunk-independent. Once it fails for
            // one chunk it'll fail for all — short-circuit to error.
            break;
        }
        return Ok(chunk);
    }

    Err(DowngradeReason(format!(
        "no candidate in {:?} passes Tier B.1 validators for (bq={}, bkv={}, hd={}, dm={}); last error: {}",
        CANDIDATE_CHUNKS,
        config.block_q,
        config.block_kv,
        config.head_dim,
        d_model,
        last_err.unwrap_or_else(|| "(no candidates tested; d_model smaller than all chunk sizes)".to_string()),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64, dm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: dm,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn select_admits_canonical_small_config() {
        // bq=32, bkv=32, hd=64, dm=2048 — verified admitted in V3 CSV
        // (row `32,32,64,2048,true,128,-`).
        let cfg = make_config(32, 32, 64, 2048);
        let chunk = select(&cfg).expect("small config should pass");
        assert!(CANDIDATE_CHUNKS.contains(&chunk));
    }

    #[test]
    fn select_rejects_fattest_canonical_config() {
        // bq=128, bkv=128, hd=128 — fixed_bytes alone is
        // q_tile (32768) + 4*kv_tile (131072) = 163840 bytes, which
        // already exceeds the 99 KB (101376 B) dynamic SMEM budget
        // before any chunk_staging is added. The V3 supported-matrix
        // CSV lists every `128,128,128,*` row as rejected with reason
        // `fixed_bytes=163840 exceeds budget alone`.
        //
        // Spec deviation from the original prompt: the prompt used
        // (64,64,128,4096) as the fattest example, but the V3 CSV
        // admits that tuple at chunk=16 (fits with 81920 + 12288 =
        // 94208 B < 101376 B budget). Using (128,128,128) instead
        // keeps the test name's intent — "no chunk fits this tuple"
        // — while matching CSV ground truth.
        let cfg = make_config(128, 128, 128, 4096);
        let result = select(&cfg);
        assert!(result.is_err(), "fattest canonical should be rejected");
    }

    #[test]
    fn select_rejects_zero_d_model() {
        let cfg = FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: 0,
                ..CshaExtras::default()
            }),
        };
        let result = select(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("d_model"));
    }

    #[test]
    fn select_descends_to_floor_when_larger_chunks_dont_fit() {
        // A config where chunk=64 doesn't fit but chunk=32 does.
        // bq=64, bkv=64, hd=96, dm=2048: try the descending walk.
        //
        // Hand-compute against `validate_tier_b1_config`:
        //   q_tile = 64*96*2 = 12288
        //   kv_ping_pong = 4*(64*96*2) = 49152
        //   fixed = 61440. Budget = 99*1024 = 101376.
        //   Available for chunk_staging = 39936.
        //   chunk_staging(chunk) = 2*chunk*96*2 + (64+64)*chunk*2
        //                        = 384*chunk + 256*chunk = 640*chunk
        //   chunk=64:  640*64 = 40960 > 39936 → reject
        //   chunk=32:  640*32 = 20480 < 39936 → admit
        //
        // V3 CSV row `64,64,96,2048,true,32,-` confirms admit-at-32.
        let cfg = make_config(64, 64, 96, 2048);
        let chunk = select(&cfg).expect("config should pass at chunk=32 or smaller");
        assert!(chunk <= 32, "expected chunk <= 32, got {}", chunk);
    }
}
