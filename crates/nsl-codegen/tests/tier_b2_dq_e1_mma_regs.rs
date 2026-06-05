//! E1: MMA lane setup and `%` prefix correctness tests.
//!
//! Two issues resolved in Phase 2.5 E1:
//!
//! ## Issue 1 — MMA `%` prefix bug
//!
//! `emit_mma_instruction` embeds register names verbatim in operand lists like
//! `{d_regs[0], d_regs[1], ...}` — it does NOT prepend `%`.  The fragment-load
//! helpers (`emit_load_a_fragment_smem`, `emit_load_b_fragment_smem`) take
//! names WITHOUT `%` and prepend it themselves in `ld.shared` instructions.
//!
//! Prior to E1, `emit_inner_loop_body` in dq.rs built register arrays like
//! `["s_a0", "s_a1", ...]` (no `%`) and passed them to both the load helpers
//! (correct) and to `emit_mma_instruction` (wrong — ptxas rejects bare names
//! in MMA operand lists).
//!
//! Fix (Option A): added `pct4`/`pct2` closures in `emit_inner_loop_body` that
//! build `%`-prefixed copies of each array right before each `emit_mma_instruction`
//! call.  The load helpers still receive the unprefixed arrays.  This matches the
//! existing convention in every other caller:
//!   - `wrga_fused_ptx.rs` constructs `"%main_a_frag0".into()` etc.
//!   - `attention_mma.rs` uses `format!("%s_acc_{}_0", t)` etc.
//!   - `projection_mma.rs` uses `format!("%q_acc_{}_0", t)` etc.
//!   - `flash_attention.rs` applies a `pct` closure before each MMA call
//!     (with a comment: "emit_mma_instruction embeds names verbatim").
//!
//! ## Issue 2 — `%mma_a_row` / `%mma_b_row` lane setup
//!
//! Both helpers compute lane-mapping INTERNALLY from `%tid.x`.  They use
//! `%mma_a_row` and `%mma_b_row` as SCRATCH u32 registers — NOT as
//! caller-supplied inputs.  The declarations in `emit_register_decls`:
//!   `.reg .u32 %mma_addr, %mma_a_row, %mma_b_row;`
//! are therefore correctly kept; they provide the scratch storage the helpers
//! need.  No caller setup of per-lane formulas is required or emitted.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn canonical_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            ..Default::default()
        }),
    }
}

// ── Issue 1: MMA operand % prefix ─────────────────────────────────────────────

/// Every MMA instruction (S's hd/16 tiled k-tiles, dP's hd/16 tiled k-tiles, dQ-update)
/// must emit operand lists with `%`-prefixed register names.  ptxas rejects bare names
/// like `{s_d0, ...}`.  Regression test: assert each MMA operand list uses `%`
/// on every register.
///
/// Post-Task-8: S=Q@K^T and dP=dO@V^T are both codegen-unrolled to hd/16 MMAs
/// each (inside the runtime DQ_NTILE_LOOP); the dQ-update is now ALSO fully
/// codegen-unrolled over (hd/8) output n-tiles x (bkv/16) k-tiles.
/// So emitted MMA headers = hd/16 (S) + hd/16 (dP) + (hd/8)*(bkv/16) (dQ).
/// At hd=128 (bkv_eff=32): 8 + 8 + 16*2 = 48.
#[test]
fn mma_operand_lists_contain_percent_prefix() {
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_effective_bkv;
    let cfg = canonical_cfg();
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    // Collect every line that contains "mma.sync.aligned.m16n8k16" (the MMA header)
    // and the following four lines (D, A, B, C operand lists).
    let lines: Vec<&str> = ptx.lines().collect();
    let mma_header_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains("mma.sync.aligned.m16n8k16"))
        .map(|(i, _)| i)
        .collect();
    let hd = cfg.head_dim as usize;
    let bkv = tier_b2_effective_bkv(&cfg) as usize;
    let expected = hd / 16 + hd / 16 + (hd / 8) * (bkv / 16);
    assert_eq!(
        mma_header_indices.len(), expected,
        "expected S(hd/16) + dP(hd/16) + dQ((hd/8)*(bkv/16)) = {expected} MMA instructions; got {}: full PTX:\n{ptx}",
        mma_header_indices.len()
    );
    for &idx in &mma_header_indices {
        // The 4 operand-list lines immediately follow the header.
        for operand_line_offset in 1..=4 {
            let line_idx = idx + operand_line_offset;
            if line_idx >= lines.len() { break; }
            let line = lines[line_idx];
            // Each line should look like `        {%reg0, %reg1, ...},`
            // Skip if it doesn't contain a `{` (shouldn't happen, but guard).
            if !line.contains('{') { continue; }
            // Extract the content inside the braces.
            let inner = line
                .split('{').nth(1)
                .and_then(|s| s.split('}').next())
                .unwrap_or("");
            for token in inner.split(',') {
                let trimmed = token.trim().trim_end_matches(';');
                if trimmed.is_empty() { continue; }
                assert!(
                    trimmed.starts_with('%'),
                    "MMA operand `{trimmed}` on line {line_idx} is missing `%` prefix.\n\
                     Full operand line: {line}\n\
                     Fix: emit_inner_loop_body must use pct4/pct2 wrappers before emit_mma_instruction."
                );
            }
        }
    }
}

/// The S = Q @ K^T MMA must reference `%s_d0` (with `%`) in its D-operand list,
/// not bare `s_d0`.
#[test]
fn s_mma_d_operand_has_percent_prefix() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // Positive assertion: prefixed form is present.
    assert!(
        ptx.contains("{%s_d0, %s_d1, %s_d2, %s_d3}"),
        "expected {{%s_d0, %s_d1, %s_d2, %s_d3}} in PTX (with %% prefix).\n\
         Got PTX excerpt around 'mma.sync':\n{}",
        ptx.lines()
            .filter(|l| l.contains("mma.sync") || l.contains("s_d"))
            .take(20)
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Negative assertion: un-prefixed form must NOT appear in an operand list context.
    // (The register _declarations_ like `.reg .f32 %s_d0` are fine and contain "s_d0"
    // after the `%` — but the operand form `{s_d0,` without `%` must be absent.)
    assert!(
        !ptx.contains("{s_d0,"),
        "found bare `{{s_d0,` (missing %% prefix) in PTX — the % prefix bug was not fixed."
    );
}

/// The dQ-update MMA must reference `%dq_acc_0_0` (with `%`) in its operand list.
#[test]
fn dq_update_mma_uses_prefixed_dq_acc_regs() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("{%dq_acc_0_0, %dq_acc_0_1, %dq_acc_0_2, %dq_acc_0_3}"),
        "expected {{%dq_acc_0_0, ...}} with %% prefix in dQ-update MMA.\n\
         PTX lines containing 'dq_acc':\n{}",
        ptx.lines()
            .filter(|l| l.contains("dq_acc"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Bare form must not appear in operand-list context.
    assert!(
        !ptx.contains("{dq_acc_0_0,"),
        "found bare `{{dq_acc_0_0,` (missing %% prefix) — dQ-update MMA operand bug not fixed."
    );
}

// ── Issue 2: %mma_a_row / %mma_b_row scratch registers ───────────────────────

/// Both helpers use `%mma_a_row` and `%mma_b_row` as SCRATCH registers for
/// computing per-lane addresses internally.  They MUST be declared in the
/// register decl block so ptxas can allocate them.
#[test]
fn mma_scratch_registers_are_declared() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // The combined decl is `.reg .u32 %mma_addr, %mma_a_row, %mma_b_row;`
    assert!(
        ptx.contains("%mma_a_row"),
        "expected `%mma_a_row` scratch register to be declared — it is required by \
         emit_load_a_fragment_smem as an address accumulator scratch."
    );
    assert!(
        ptx.contains("%mma_b_row"),
        "expected `%mma_b_row` scratch register to be declared — it is required by \
         emit_load_b_fragment_smem as an address accumulator scratch."
    );
    assert!(
        ptx.contains("%mma_addr"),
        "expected `%mma_addr` scratch register to be declared — it is required by \
         both fragment-load helpers for the lane-id derivation."
    );
}

/// The helpers self-derive per-lane row/col from `%tid.x` — no caller-emitted
/// per-lane formula is needed.  Assert that `%mma_a_row` is NOT set via the
/// old manual formula `lane / 4` in a caller-visible way (the helpers do this
/// internally, so only the helpers' own comment should reference it).
/// This is a weak structural test: we can't distinguish caller vs. helper code
/// in the flat PTX string, but we verify that the helper's self-derivation
/// comment IS present, confirming the modern rewrite is in use.
#[test]
fn fragment_helpers_self_derive_lane_mapping() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // The A-fragment helper emits this comment when it runs.
    assert!(
        ptx.contains("// Load A-fragment (m16xk16 f16 row-major) per PTX m16n8k16 spec"),
        "expected A-fragment self-derivation comment — confirms the N4-rewrite \
         helper is in use (not the old externally-setup version)."
    );
    // The B-fragment helper emits this comment when it runs.
    assert!(
        ptx.contains("// Load B-fragment (k16xn8 col-major f16) per PTX m16n8k16 spec"),
        "expected B-fragment self-derivation comment."
    );
}

// ── Regression: ptxas smoke test (active, not ignored) ───────────────────────

/// Full ptxas smoke test for the dQ-kernel PTX.
///
/// This test was previously `#[ignore]`'d in C1/C2/C3/C4 due to the MMA `%`
/// prefix bug.  E1 fixes the bug; this integration-level test confirms the fix
/// produces ptxas-clean PTX.
///
/// Gracefully skipped when ptxas is not on PATH (CI without CUDA toolkit).
#[test]
fn e1_dq_kernel_ptxas_clean() {
    use std::process::Command;
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let tmp = std::env::temp_dir().join("e1_dq_kernel_test.ptx");
    std::fs::write(&tmp, &ptx).unwrap();
    let cubin = std::env::temp_dir().join("e1_dq_kernel_test.cubin");
    let result = Command::new("ptxas")
        .args([
            "-arch=sm_80",
            tmp.to_str().unwrap(),
            "-o",
            cubin.to_str().unwrap(),
        ])
        .output();
    match result {
        Ok(out) if out.status.success() => {
            // Clean — ptxas accepted the PTX.
        }
        Ok(out) => {
            let ptx_excerpt: String = ptx.lines().take(100).collect::<Vec<_>>().join("\n");
            panic!(
                "ptxas rejected dQ-kernel PTX after E1 fix:\n\
                 stdout: {}\nstderr: {}\n\
                 PTX (first 100 lines):\n{}\n\
                 Full PTX written to: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
                ptx_excerpt,
                tmp.display(),
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // ptxas not on PATH — skip gracefully.
            eprintln!("ptxas not found on PATH, skipping E1 ptxas smoke test.");
        }
        Err(e) => panic!("Failed to spawn ptxas: {e}"),
    }
}
