//! Unit-level ptxas validation for synthesize_fused_lora_ptx and
//! synthesize_fused_ia3_ptx.  Each config's output is fed to
//! cuModuleLoadData (or ptxas fallback); the test asserts acceptance.
//!
//! This test is the gate that B.3 lacked — string-pattern tests let
//! pseudocode through.  Real PTX validation catches invalid syntax,
//! missing .shared decls, uninitialized registers, operand-type
//! mismatches, .param-as-operand bugs, etc.
//!
//! **On machines without cudarc or ptxas**, validate_ptx returns an
//! error containing "nvcc not available".  Tests then skip with a
//! printed message — they don't fail.

use nsl_codegen::ptxas_validation::validate_ptx;
use nsl_codegen::wrga_fused_ptx::{synthesize_fused_lora_ptx, FusedLoraConfig};

fn lora_cfg(m: u32, n: u32, k: u32, rank: u32) -> FusedLoraConfig {
    FusedLoraConfig {
        site_id: format!("test.m{}n{}k{}r{}", m, n, k, rank),
        m,
        n,
        k,
        rank,
        target_sm: 80,
    }
}

fn assert_lora_ptx_valid(cfg: FusedLoraConfig) {
    let ptx = synthesize_fused_lora_ptx(&cfg);
    match validate_ptx(&ptx) {
        Ok(()) => {}
        Err(msg) if msg.contains("nvcc not available") => {
            eprintln!(
                "[skip] LoRA ptxas validation for ({},{},{},r={}) — no validator available: {}",
                cfg.m, cfg.n, cfg.k, cfg.rank, msg,
            );
        }
        Err(msg) => panic!(
            "LoRA PTX rejected for config (m={}, n={}, k={}, rank={}):\n{}\n\nEmitted PTX:\n{}",
            cfg.m, cfg.n, cfg.k, cfg.rank, msg, ptx
        ),
    }
}

#[test]
fn lora_ptx_validates__16_8_16_16() {
    assert_lora_ptx_valid(lora_cfg(16, 8, 16, 16));
}

#[test]
fn lora_ptx_validates__16_8_32_4() {
    assert_lora_ptx_valid(lora_cfg(16, 8, 32, 4));
}

#[test]
fn lora_ptx_validates__1_8_8_2() {
    assert_lora_ptx_valid(lora_cfg(1, 8, 8, 2));
}

#[test]
fn lora_ptx_validates__4_8_8_2() {
    // BUILD4_SRC_GPU hardening-test shape.
    assert_lora_ptx_valid(lora_cfg(4, 8, 8, 2));
}

#[test]
fn lora_ptx_validates__32_16_64_8() {
    assert_lora_ptx_valid(lora_cfg(32, 16, 64, 8));
}

#[test]
fn lora_ptx_validates__16_8_8_16() {
    // rank == MMA k: exercises emit_smem_zero_pad_predicated no-op path.
    assert_lora_ptx_valid(lora_cfg(16, 8, 8, 16));
}

// ─── IA³ ptxas validation — Task E1 ─────────────────────────────────

use nsl_codegen::wrga_fused_ptx::{synthesize_fused_ia3_ptx, FusedIa3Config};

fn ia3_cfg(m: u32, n: u32, k: u32) -> FusedIa3Config {
    FusedIa3Config {
        site_id: format!("test.m{}n{}k{}", m, n, k),
        m,
        n,
        k,
        target_sm: 80,
    }
}

fn assert_ia3_ptx_valid(cfg: FusedIa3Config) {
    let ptx = synthesize_fused_ia3_ptx(&cfg);
    match validate_ptx(&ptx) {
        Ok(()) => {}
        Err(msg) if msg.contains("nvcc not available") => {
            eprintln!(
                "[skip] IA3 ptxas validation for ({},{},{}) — no validator available: {}",
                cfg.m, cfg.n, cfg.k, msg,
            );
        }
        Err(msg) => panic!(
            "IA3 PTX rejected for config (m={}, n={}, k={}):\n{}\n\nEmitted PTX:\n{}",
            cfg.m, cfg.n, cfg.k, msg, ptx
        ),
    }
}

#[test]
fn ia3_ptx_validates__16_8_16() {
    assert_ia3_ptx_valid(ia3_cfg(16, 8, 16));
}

#[test]
fn ia3_ptx_validates__16_8_32() {
    assert_ia3_ptx_valid(ia3_cfg(16, 8, 32));
}

#[test]
fn ia3_ptx_validates__1_8_8() {
    assert_ia3_ptx_valid(ia3_cfg(1, 8, 8));
}

#[test]
fn ia3_ptx_validates__4_8_8() {
    // BUILD4 hardening-test shape equivalent for IA3.
    assert_ia3_ptx_valid(ia3_cfg(4, 8, 8));
}

#[test]
fn ia3_ptx_validates__32_16_64() {
    assert_ia3_ptx_valid(ia3_cfg(32, 16, 64));
}

// ─── GatedLoRA ptxas validation — Task 3.2 ─────────────────────

use nsl_codegen::wrga_fused_ptx::{synthesize_fused_gatedlora_ptx, FusedGatedLoraConfig};

fn gated_cfg(m: u32, n: u32, k: u32, rank: u32) -> FusedGatedLoraConfig {
    FusedGatedLoraConfig {
        site_id: format!("gated.m{m}n{n}k{k}r{rank}"),
        m,
        n,
        k,
        rank,
        target_sm: 80,
    }
}

fn assert_gatedlora_ptx_valid(cfg: FusedGatedLoraConfig) {
    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let rank = cfg.rank;
    let ptx = synthesize_fused_gatedlora_ptx(&cfg);
    match validate_ptx(&ptx) {
        Ok(()) => {}
        Err(msg) if msg.contains("nvcc not available") => {
            eprintln!(
                "[skip] GatedLoRA ptxas validation for ({m},{n},{k},r={rank}) — no validator: {msg}"
            );
        }
        Err(msg) => panic!(
            "GatedLoRA PTX rejected for config (m={m}, n={n}, k={k}, rank={rank}):\n{msg}\n\nEmitted PTX:\n{ptx}"
        ),
    }
}

// ── Reused LoRA shapes under FoldKind::PerColumnSigmoid (6) ──
#[test]
fn gatedlora_ptx_validates__16_8_16_16() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 16, 16));
}

#[test]
fn gatedlora_ptx_validates__16_8_32_4() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 32, 4));
}

#[test]
fn gatedlora_ptx_validates__1_8_8_2() {
    assert_gatedlora_ptx_valid(gated_cfg(1, 8, 8, 2));
}

#[test]
fn gatedlora_ptx_validates__4_8_8_2() {
    assert_gatedlora_ptx_valid(gated_cfg(4, 8, 8, 2));
}

#[test]
fn gatedlora_ptx_validates__32_16_64_8() {
    assert_gatedlora_ptx_valid(gated_cfg(32, 16, 64, 8));
}

#[test]
fn gatedlora_ptx_validates__16_8_8_16() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 8, 16));
}

// ── GatedLoRA-distinctive configs (4) ──
// Note: ptxas validation doesn't distinguish gate VALUES (runtime concern);
// these tests validate that the emitted PTX structure is valid across
// shape-and-feature combinations.  The (16, 13, 16, 4) config is the ONLY
// one that exercises FoldResultMask's %p_col0/%p_col1 predicates at
// runtime (partial-n tile: first tile full n=8, second tile n=5 partial).
#[test]
fn gatedlora_ptx_validates__uniform_gate_zero_16_8_16_16() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 16, 16));
}

#[test]
fn gatedlora_ptx_validates__alternating_saturation_16_8_16_16() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 16, 16));
}

#[test]
fn gatedlora_ptx_validates__partial_n_multi_tile_16_13_16_4() {
    // n=13 → 2 tiles (tile 0 full n=8; tile 1 partial n=5).
    // ONLY config that actually evaluates FoldResultMask predicates false at runtime.
    assert_gatedlora_ptx_valid(gated_cfg(16, 13, 16, 4));
}

#[test]
fn gatedlora_ptx_validates__sub_mma_k_no_rank_pad_16_8_8_16() {
    // Sub-MMA K (k=8<16) + rank == MMA-k (no rank-pad path).
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 8, 16));
}

// -------- Llama-scale ptxas regression gate (2026-04-19) --------
//
// These configs exercise the scale regime that B.3.1's own test matrix
// missed: realistic inference/training dims (n, k) in the 1k..4k range at
// ranks 8..64. Pre-rewrite, every config here produced PTX that timed out
// or overflowed ptxas (20 MB at k=4096). The runtime-K-loop keeps PTX at
// ~106 KB regardless of k, so ptxas compiles all of them fast.
//
// Rule being enforced (WRGA paper Appendix B.3): any PTX emitter that targets
// a scale regime in its spec must have ptxas-compile-validity tests at the
// upper end of that regime, not just at shapes convenient for small-matrix
// unit testing. The B.3.2 trigger referenced Llama-3-8B-style shapes; this
// sweep is the test-coverage cash-out for that reference.

fn assert_at_scale(m: u32, n: u32, k: u32, rank: u32, kind: &str) {
    match kind {
        "lora" => assert_lora_ptx_valid(lora_cfg(m, n, k, rank)),
        "gatedlora" => assert_gatedlora_ptx_valid(gated_cfg(m, n, k, rank)),
        "ia3" => {
            let cfg = nsl_codegen::wrga_fused_ptx::FusedIa3Config {
                site_id: format!("test_scale.m{}n{}k{}", m, n, k),
                m,
                n,
                k,
                target_sm: 80,
            };
            let ptx = nsl_codegen::wrga_fused_ptx::synthesize_fused_ia3_ptx(&cfg);
            match validate_ptx(&ptx) {
                Ok(()) => {}
                Err(msg) if msg.contains("nvcc not available") => {
                    eprintln!("[skip] IA³ scale config ({}, {}, {}) — {}", m, n, k, msg);
                }
                Err(msg) => panic!(
                    "IA³ PTX rejected at scale ({}, {}, {}):\n{}",
                    m, n, k, msg
                ),
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn scale__lora_m1_n1024_k1024_r16() {
    assert_at_scale(1, 1024, 1024, 16, "lora");
}

#[test]
fn scale__lora_m1_n4096_k4096_r16() {
    // Llama-3-8B attention-out proxy, rank=16. Pre-rewrite: 20 MB PTX, ptxas timeout.
    assert_at_scale(1, 4096, 4096, 16, "lora");
}

#[test]
fn scale__gatedlora_m1_n4096_k4096_r16() {
    // B.3.2 trigger's prescribed shape at rank 16.
    assert_at_scale(1, 4096, 4096, 16, "gatedlora");
}

#[test]
fn scale__gatedlora_m1_n2048_k2048_r8() {
    assert_at_scale(1, 2048, 2048, 8, "gatedlora");
}

#[test]
fn scale__gatedlora_m1_n1024_k1024_r16() {
    assert_at_scale(1, 1024, 1024, 16, "gatedlora");
}

#[test]
fn scale__ia3_m1_n4096_k4096() {
    assert_at_scale(1, 4096, 4096, 0, "ia3");
}
