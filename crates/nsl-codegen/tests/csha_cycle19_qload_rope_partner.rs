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

/// Count `fma.rn.f32` instructions in a PTX slice whose source-operand
/// list contains `%f2` as one of its operands (destination excluded).
fn fma_uses_f2_partner(slice: &str) -> usize {
    slice
        .lines()
        .filter(|line| line.contains("fma.rn.f32"))
        .filter(|line| {
            // Split at commas; operand[0] is destination, rest are sources.
            let after = match line.find("fma.rn.f32") {
                Some(pos) => &line[pos..],
                None => return false,
            };
            let parts: Vec<&str> = after.splitn(2, ',').collect();
            if parts.len() < 2 { return false; }
            let sources = parts[1];
            // Match `%f2` as a whole token (not `%f20`, `%f22`, ...).
            sources
                .split(|c: char| c == ',' || c.is_whitespace() || c == ';')
                .any(|tok| tok == "%f2")
        })
        .count()
}

#[test]
fn qload_rope_adjacent_uses_partner_in_fma() {
    let cfg = config_inline_rope(RopeStyle::Adjacent);
    let ptx = ptx_string(&cfg);
    let slice = rope_block_slice(&ptx, "// rope adjacent slice 0");
    let count = fma_uses_f2_partner(slice);
    assert!(
        count >= 1,
        "Adjacent RoPE block must consume the shuffled partner (%f2) in \
         at least one fma.rn.f32; found 0.\n\nBlock:\n{}",
        slice
    );
}

#[test]
fn qload_rope_halfsplit_uses_partner_in_fma() {
    let cfg = config_inline_rope(RopeStyle::HalfSplit);
    let ptx = ptx_string(&cfg);
    let slice = rope_block_slice(&ptx, "// rope halfsplit slice 0");

    // A branch "uses %f2" if any predicated instruction (fma or neg) in
    // that branch reads %f2 as a source operand. `neg.f32 %f3, %f2`
    // followed by fma using %f3 counts as consumption of the partner —
    // %f2 is not orphaned. What we're guarding against is the pre-fix
    // state where NO instruction after the shfl ever mentioned %f2 in a
    // source position under that predicate.
    fn branch_uses_f2(slice: &str, pred: &str) -> bool {
        slice
            .lines()
            .filter(|l| l.contains(pred))
            // Only fma / neg (and mov) can propagate %f2's value.
            .filter(|l| {
                l.contains("fma.rn.f32")
                    || l.contains("neg.f32")
                    || l.contains("mov.f32")
                    || l.contains("mul.rn.f32")
                    || l.contains("add.rn.f32")
            })
            .any(|l| {
                // Skip the destination operand (first after opcode).
                // A `%f2` in any source position proves consumption.
                let opcode_end = l.find(|c: char| c.is_whitespace() && l[..].contains("f32")).unwrap_or(0);
                // Take everything after the first comma (i.e. sources).
                let after_first_comma = match l.find(',') {
                    Some(i) => &l[i + 1..],
                    None => return false,
                };
                let _ = opcode_end;
                after_first_comma
                    .split(|c: char| c == ',' || c.is_whitespace() || c == ';')
                    .any(|tok| tok == "%f2")
            })
    }

    // `@!%p0` starts with `@!%p0` while `@%p0` starts with `@%p0` (no `!`).
    // We must distinguish them; a naive `.contains("@%p0")` matches both
    // because `@!%p0` contains `%p0`. Filter on the exact prefix.
    fn line_pred(line: &str) -> Option<&str> {
        let t = line.trim_start();
        if t.starts_with("@!%p0") { Some("@!%p0") }
        else if t.starts_with("@%p0") { Some("@%p0") }
        else { None }
    }

    let mut p0_lines = String::new();
    let mut np0_lines = String::new();
    for l in slice.lines() {
        match line_pred(l) {
            Some("@%p0") => { p0_lines.push_str(l); p0_lines.push('\n'); }
            Some("@!%p0") => { np0_lines.push_str(l); np0_lines.push('\n'); }
            _ => {}
        }
    }
    let has_p0_uses_f2 = branch_uses_f2(&p0_lines, "@%p0");
    let has_np0_uses_f2 = branch_uses_f2(&np0_lines, "@!%p0");

    assert!(
        has_p0_uses_f2,
        "HalfSplit @%p0 branch must consume shuffled partner (%f2) in an \
         fma.rn.f32.\n\nBlock:\n{}",
        slice
    );
    assert!(
        has_np0_uses_f2,
        "HalfSplit @!%p0 branch must consume shuffled partner (%f2) in \
         an fma.rn.f32.\n\nBlock:\n{}",
        slice
    );
}
