//! CSHA Cycle 19 T3 — q_load.rs RoPE partner-shuffle regression witnesses.
//!
//! Bug: the inline RoPE emitter in `q_load.rs` loads the shuffle partner
//! into `%f2` via `shfl.sync.bfly.b32`, but the follow-up FMA on that
//! branch reads only self (`%f{reg}`), cos (`%f0`), and sin (`%f1`) —
//! `%f2` is never consumed. That means the rotation degenerates to
//! `new = self*cos + sin` (sin as an additive bias) instead of the
//! reference math from `csha_hooks.rs:1584-1595`:
//!     new_x0 = x0*cos − x1*sin
//!     new_x1 = x0*sin + x1*cos
//!
//! These tests are structural (PTX-string level): after the shfl that
//! writes `%f2`, at least one subsequent `fma.rn.f32` in the same RoPE
//! block MUST list `%f2` as a source operand. The HalfSplit variant is
//! partner-split by `%p0`, so both `@%p0` and `@!%p0` bodies must
//! consume `%f2` — the current tree emits byte-identical @/@! bodies
//! that both ignore the partner.
//!
//! Both tests are expected to RED on the current tree and GREEN after
//! the q_load.rs fix.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

fn config_inline_rope(style: RopeStyle) -> FlashAttentionConfig {
    // Non-CSHA config with rope_q=true so the inline RoPE branch in
    // q_load.rs fires (csha=None disables the fused_projections path).
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: true,
        rope_style: style,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: None,
        checkpoint: None,
    }
}

fn ptx_string(cfg: &FlashAttentionConfig) -> String {
    String::from_utf8(synthesize_flash_attention_ptx_v2(cfg))
        .expect("PTX must be valid UTF-8")
}

/// Slice the PTX starting from the first RoPE block marker up to a
/// reasonable stopping point (next Phase or block boundary).
fn rope_block_slice<'a>(ptx: &'a str, marker: &str) -> &'a str {
    let start = ptx
        .find(marker)
        .unwrap_or_else(|| panic!("expected RoPE marker '{}' in PTX", marker));
    // Stop at the next SMEM store (RoPE block always ends with
    // `cvt.rn.f16.f32 %h0, %f{reg}` before the shared-memory store).
    let tail = &ptx[start..];
    let end = tail.find("cvt.rn.f16.f32").unwrap_or(tail.len());
    &tail[..end]
}

/// Return true if the line reads `%f2` or `%f3` (partner or -partner) as
/// a source operand in a partner-consuming instruction (fma/neg/mul/add/mov).
/// `%f3` counts because the EVEN branch synthesizes `%f3 = neg %f2` and
/// then feeds `%f3` into the follow-up FMA — dropping `%f3` propagation
/// would still orphan the partner value.
fn line_consumes_partner(line: &str) -> bool {
    let is_partner_op = line.contains("fma.rn.f32")
        || line.contains("neg.f32")
        || line.contains("mov.f32")
        || line.contains("mul.rn.f32")
        || line.contains("add.rn.f32");
    if !is_partner_op {
        return false;
    }
    // Sources begin after the first comma (operand[0] is destination).
    let after_first_comma = match line.find(',') {
        Some(i) => &line[i + 1..],
        None => return false,
    };
    after_first_comma
        .split(|c: char| c == ',' || c.is_whitespace() || c == ';')
        .any(|tok| tok == "%f2" || tok == "%f3")
}

/// Extract only the lines under a given predicate prefix (e.g. `@%p1` vs
/// `@!%p1`). Prefix matching is exact on the trimmed-left line to avoid
/// `@!%p1` false-matching a `@%p1` filter.
fn lines_under_pred<'a>(slice: &'a str, pred: &str) -> String {
    let mut out = String::new();
    for l in slice.lines() {
        let t = l.trim_start();
        // Only accept an EXACT prefix match at the start of the line.
        // For `@%p1`, we must additionally reject `@!%p1`.
        let ok = if pred == "@%p1" {
            t.starts_with("@%p1") && !t.starts_with("@!")
        } else if pred == "@%p0" {
            t.starts_with("@%p0") && !t.starts_with("@!")
        } else {
            t.starts_with(pred)
        };
        if ok {
            out.push_str(l);
            out.push('\n');
        }
    }
    out
}

#[test]
fn qload_rope_adjacent_uses_partner_in_fma() {
    let cfg = config_inline_rope(RopeStyle::Adjacent);
    let ptx = ptx_string(&cfg);
    let slice = rope_block_slice(&ptx, "// rope adjacent slice 0");

    // Adjacent splits on `%p1` (p1 == even lane). Both the @%p1 (even, x0)
    // and @!%p1 (odd, x1) branches must consume the shuffled partner.
    // Symmetric assertion: a regression in EITHER branch must fail.
    let p1_lines = lines_under_pred(slice, "@%p1");
    let np1_lines = lines_under_pred(slice, "@!%p1");

    let p1_uses = p1_lines.lines().any(line_consumes_partner);
    let np1_uses = np1_lines.lines().any(line_consumes_partner);

    assert!(
        p1_uses,
        "Adjacent RoPE @%p1 (even, x0) branch must consume the shuffled \
         partner (%f2 or %f3=neg(%f2)) in a partner-consuming instruction \
         (fma/neg/mul/add).\n\nBlock:\n{}",
        slice
    );
    assert!(
        np1_uses,
        "Adjacent RoPE @!%p1 (odd, x1) branch must consume the shuffled \
         partner (%f2) in a partner-consuming instruction \
         (fma/neg/mul/add).\n\nBlock:\n{}",
        slice
    );
}

#[test]
fn qload_rope_halfsplit_uses_partner_in_fma() {
    let cfg = config_inline_rope(RopeStyle::HalfSplit);
    let ptx = ptx_string(&cfg);
    let slice = rope_block_slice(&ptx, "// rope halfsplit slice 0");

    // HalfSplit splits on `%p0` (lane < 16). Both the @%p0 (lane<16, x0)
    // and @!%p0 (lane>=16, x1) branches must consume the shuffled partner
    // via a partner-consuming instruction (fma/neg/mul/add/mov). The
    // pre-fix state emitted byte-identical @/@! bodies that both ignored
    // %f2 — symmetric assertions here catch a regression in either branch.
    let p0_lines = lines_under_pred(slice, "@%p0");
    let np0_lines = lines_under_pred(slice, "@!%p0");

    let has_p0_uses = p0_lines.lines().any(line_consumes_partner);
    let has_np0_uses = np0_lines.lines().any(line_consumes_partner);

    assert!(
        has_p0_uses,
        "HalfSplit @%p0 (lane<16, x0) branch must consume shuffled partner \
         (%f2 or %f3=neg(%f2)) in a partner-consuming instruction \
         (fma/neg/mul/add).\n\nBlock:\n{}",
        slice
    );
    assert!(
        has_np0_uses,
        "HalfSplit @!%p0 (lane>=16, x1) branch must consume shuffled \
         partner (%f2) in a partner-consuming instruction \
         (fma/neg/mul/add).\n\nBlock:\n{}",
        slice
    );
}
