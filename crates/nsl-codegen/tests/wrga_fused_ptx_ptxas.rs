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

