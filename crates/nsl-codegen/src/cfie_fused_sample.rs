//! CFIE — fused LM_head + softmax + top-k + top-p + sample kernel.
//!
//! Paper §4: the final-token-selection path in vLLM / TRT-LLM launches
//! 4-6 separate kernels and round-trips the `[1, vocab_size]` logits
//! tensor through HBM 6 times.  CFIE fuses the whole pipeline into a
//! single tiled kernel where the logits never leave SMEM / registers,
//! and only the final token ID is written back to HBM.
//!
//! This module produces the **structured op program** the PTX / CPU
//! backend consumes.  Writing the PTX itself is a backend concern —
//! what we do here is audit the algorithmic flow and give downstream
//! emitters a canonical, testable recipe.

use serde::Serialize;

/// Sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SamplingStrategy {
    /// Argmax (temperature = 0).
    Greedy,
    /// Top-K then multinomial.
    TopK,
    /// Top-K then Top-P (nucleus) then multinomial.
    TopKTopP,
    /// Pure multinomial (no top-k, no top-p).
    Multinomial,
}

impl SamplingStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            SamplingStrategy::Greedy => "greedy",
            SamplingStrategy::TopK => "top_k",
            SamplingStrategy::TopKTopP => "top_k_top_p",
            SamplingStrategy::Multinomial => "multinomial",
        }
    }
}

/// Hyper-parameters for the sampler.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub strategy: SamplingStrategy,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    /// Whether grammar masking is active — if so, every vocab tile is
    /// first checked against the compiled DFA's valid-token set.
    pub grammar_masked: bool,
    /// Whether a logits bias vector applies post-matmul (e.g. EOS
    /// penalty).  Causes an extra `ApplyLogitsBias` op.
    pub logits_bias: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::TopKTopP,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            grammar_masked: false,
            logits_bias: false,
        }
    }
}

/// Shape of the LM head matmul.
#[derive(Debug, Clone, Copy)]
pub struct LmHeadShape {
    pub d_model: u32,
    pub vocab_size: u32,
    /// Vocab tile size (e.g. 256 or 512) — the matmul is chunked along
    /// the output dimension to keep per-tile logits in SMEM.
    pub vocab_tile: u32,
    pub dtype_bytes: u32,
}

impl LmHeadShape {
    pub fn num_vocab_tiles(&self) -> u32 {
        if self.vocab_tile == 0 {
            return 0;
        }
        self.vocab_size.div_ceil(self.vocab_tile)
    }

    /// Bytes of one vocab tile's logits (FP32 accumulator).
    pub fn logits_tile_bytes(&self) -> u64 {
        (self.vocab_tile as u64) * 4
    }
}

/// One op in the fused decode-sample program.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum FusedSampleOp {
    /// Load the hidden state `[1, d_model]` into SMEM.
    LoadHidden,
    /// For the current vocab tile, compute `logits_tile = hidden @
    /// W_tile`, scale by `1 / temperature`, and leave the tile in
    /// SMEM / registers.
    MatmulTile {
        tile_index: u32,
        temperature_recip: f32,
    },
    /// Grammar DFA check — skip vocab entries not in the current
    /// state's valid-token set.
    GrammarMaskTile { tile_index: u32 },
    /// Apply an additive logits bias vector in-tile.
    ApplyLogitsBias { tile_index: u32 },
    /// Update the running top-K min-heap in registers.
    UpdateTopK { tile_index: u32 },
    /// Compute the running max for stable softmax in the same pass.
    UpdateRunningMax,
    /// Softmax over the retained top-k candidates only.
    SoftmaxTopK,
    /// Top-P (nucleus) filter on the top-k candidates.
    NucleusFilter { top_p: f32 },
    /// Greedy argmax — bypasses softmax/top-p entirely.
    Argmax,
    /// Multinomial sample from filtered candidates.
    MultinomialSample,
    /// Store the final token ID (the *only* HBM write of the kernel).
    StoreTokenId,
}

/// Complete structured program for one fused sample call.
#[derive(Debug, Clone)]
pub struct FusedSampleProgram {
    pub params: SamplingParams,
    pub shape: LmHeadShape,
    pub ops: Vec<FusedSampleOp>,
    /// HBM bytes saved versus an unfused pipeline (for reporting).
    pub hbm_bytes_saved: u64,
}

impl FusedSampleProgram {
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    /// Whether this program actually fuses work that would normally be
    /// split across kernels.
    pub fn is_fused(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, FusedSampleOp::MatmulTile { .. }))
            && self.ops.iter().any(|op| {
                matches!(
                    op,
                    FusedSampleOp::MultinomialSample
                        | FusedSampleOp::Argmax
                        | FusedSampleOp::SoftmaxTopK
                )
            })
    }
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

fn bytes_saved_vs_unfused(shape: &LmHeadShape) -> u64 {
    // Unfused pipeline materialises the full logits tensor 6 times
    // (matmul write, softmax R/W, top-k R/W, top-p R/W, sample R).
    // CFIE materialises it zero times.
    6 * (shape.vocab_size as u64) * 4 // FP32 logits
}

/// Build the fused program for given sampler parameters + LM-head shape.
pub fn emit_program(params: SamplingParams, shape: LmHeadShape) -> FusedSampleProgram {
    let num_tiles = shape.num_vocab_tiles();
    let mut ops = Vec::with_capacity((num_tiles as usize) * 4 + 6);
    ops.push(FusedSampleOp::LoadHidden);

    let inv_temp = if params.temperature > 0.0 {
        1.0 / params.temperature
    } else {
        1.0 // greedy — temperature irrelevant
    };

    for t in 0..num_tiles {
        ops.push(FusedSampleOp::MatmulTile {
            tile_index: t,
            temperature_recip: inv_temp,
        });
        if params.logits_bias {
            ops.push(FusedSampleOp::ApplyLogitsBias { tile_index: t });
        }
        if params.grammar_masked {
            ops.push(FusedSampleOp::GrammarMaskTile { tile_index: t });
        }
        ops.push(FusedSampleOp::UpdateTopK { tile_index: t });
        ops.push(FusedSampleOp::UpdateRunningMax);
    }

    match params.strategy {
        SamplingStrategy::Greedy => {
            ops.push(FusedSampleOp::Argmax);
        }
        SamplingStrategy::TopK => {
            ops.push(FusedSampleOp::SoftmaxTopK);
            ops.push(FusedSampleOp::MultinomialSample);
        }
        SamplingStrategy::TopKTopP => {
            ops.push(FusedSampleOp::SoftmaxTopK);
            ops.push(FusedSampleOp::NucleusFilter {
                top_p: params.top_p,
            });
            ops.push(FusedSampleOp::MultinomialSample);
        }
        SamplingStrategy::Multinomial => {
            ops.push(FusedSampleOp::SoftmaxTopK);
            ops.push(FusedSampleOp::MultinomialSample);
        }
    }
    ops.push(FusedSampleOp::StoreTokenId);

    FusedSampleProgram {
        params,
        shape,
        ops,
        hbm_bytes_saved: bytes_saved_vs_unfused(&shape),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> LmHeadShape {
        LmHeadShape {
            d_model: 512,
            vocab_size: 49_152,
            vocab_tile: 256,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn num_vocab_tiles_divides_cleanly() {
        let s = shape();
        assert_eq!(s.num_vocab_tiles(), 192); // 49152 / 256
    }

    #[test]
    fn program_emits_one_matmul_tile_per_vocab_chunk() {
        let prog = emit_program(SamplingParams::default(), shape());
        let matmul_count = prog
            .ops
            .iter()
            .filter(|op| matches!(op, FusedSampleOp::MatmulTile { .. }))
            .count();
        assert_eq!(matmul_count as u32, shape().num_vocab_tiles());
    }

    #[test]
    fn greedy_skips_softmax() {
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            ..Default::default()
        };
        let prog = emit_program(params, shape());
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, FusedSampleOp::Argmax)));
        assert!(!prog
            .ops
            .iter()
            .any(|op| matches!(op, FusedSampleOp::SoftmaxTopK)));
    }

    #[test]
    fn top_k_top_p_includes_nucleus_filter() {
        let prog = emit_program(SamplingParams::default(), shape());
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, FusedSampleOp::NucleusFilter { .. })));
    }

    #[test]
    fn grammar_mask_tiles_only_when_requested() {
        let off = emit_program(
            SamplingParams {
                grammar_masked: false,
                ..Default::default()
            },
            shape(),
        );
        let on = emit_program(
            SamplingParams {
                grammar_masked: true,
                ..Default::default()
            },
            shape(),
        );
        let off_count = off
            .ops
            .iter()
            .filter(|op| matches!(op, FusedSampleOp::GrammarMaskTile { .. }))
            .count();
        let on_count = on
            .ops
            .iter()
            .filter(|op| matches!(op, FusedSampleOp::GrammarMaskTile { .. }))
            .count();
        assert_eq!(off_count, 0);
        assert_eq!(on_count as u32, shape().num_vocab_tiles());
    }

    #[test]
    fn program_ends_with_store_token_id() {
        let prog = emit_program(SamplingParams::default(), shape());
        assert!(matches!(prog.ops.last(), Some(FusedSampleOp::StoreTokenId)));
    }

    #[test]
    fn program_is_marked_fused() {
        let prog = emit_program(SamplingParams::default(), shape());
        assert!(prog.is_fused());
    }

    #[test]
    fn hbm_savings_nonzero_for_typical_vocab() {
        let prog = emit_program(SamplingParams::default(), shape());
        assert!(prog.hbm_bytes_saved > 0);
    }

    #[test]
    fn temperature_zero_does_not_explode() {
        let params = SamplingParams {
            temperature: 0.0,
            strategy: SamplingStrategy::Greedy,
            ..Default::default()
        };
        let prog = emit_program(params, shape());
        for op in &prog.ops {
            if let FusedSampleOp::MatmulTile {
                temperature_recip, ..
            } = op
            {
                assert!(temperature_recip.is_finite());
            }
        }
    }

    #[test]
    fn logits_bias_inserts_extra_op_per_tile() {
        let params = SamplingParams {
            logits_bias: true,
            ..Default::default()
        };
        let prog = emit_program(params, shape());
        let bias_ops = prog
            .ops
            .iter()
            .filter(|op| matches!(op, FusedSampleOp::ApplyLogitsBias { .. }))
            .count();
        assert_eq!(bias_ops as u32, shape().num_vocab_tiles());
    }

    #[test]
    fn strategy_names_are_stable() {
        assert_eq!(SamplingStrategy::Greedy.as_str(), "greedy");
        assert_eq!(SamplingStrategy::TopK.as_str(), "top_k");
        assert_eq!(SamplingStrategy::TopKTopP.as_str(), "top_k_top_p");
    }
}
