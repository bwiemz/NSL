//! Verify emit_q_projection's PTX assembles cleanly on sm_120 and sm_80
//! when wrapped in a minimal .entry kernel.
//!
//! These tests are `#[ignore]`-gated because they require `ptxas` from a
//! CUDA toolkit. Run with:
//!     cargo test --package nsl-codegen --test tier_b1_q_projection_ptxas \
//!         -- --ignored --nocapture

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::projection_mma::emit_q_projection;
use std::fmt::Write;
use std::process::Command;

fn canonical_config(bq: i64, bkv: i64, hd: i64, dm: u32) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bkv,
        head_dim: hd,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: dm,
            ..CshaExtras::default()
        }),
    }
}

/// Emit a minimal `.entry` kernel wrapper. The wrapper declares all
/// registers and SMEM that `emit_q_projection` touches:
///   * The two params it null-guards on: csha_x_ptr + csha_wq_ptr.
///   * `%shmem_base` (u64) — used as SMEM base in the emitted address
///     arithmetic.
///   * An `extern .shared` allocation big enough for the Q tile +
///     chunk staging (we just size it to a generous 99 KB which is the
///     dynamic-SMEM cap on Blackwell).
fn write_minimal_kernel_wrapper(ptx: &mut String, sm: u32) {
    writeln!(ptx, ".version 8.7").unwrap();
    writeln!(ptx, ".target sm_{}", sm).unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();
    writeln!(ptx, ".extern .shared .align 16 .b8 shmem[];").unwrap();
    writeln!(ptx).unwrap();
    writeln!(ptx, ".visible .entry tier_b1_q_proj_test(").unwrap();
    writeln!(ptx, "    .param .u64 csha_x_ptr,").unwrap();
    writeln!(ptx, "    .param .u64 csha_wq_ptr").unwrap();
    writeln!(ptx, ") {{").unwrap();
    writeln!(ptx, "    .reg .u64 %shmem_base;").unwrap();
    // Lane-derived address registers consumed by the matmul_mma fragment
    // loaders (which the projection sweep calls). Per the post-N4
    // helper rewrite, the helpers derive lane from `%tid.x` and write
    // intermediate values into `%mma_addr` (u32 scratch) and `%mma_a_row`
    // / `%mma_b_row` (u32 row+col accumulators). The full Tier B.1
    // kernel declares these via `register_budget::declare_registers`;
    // the standalone wrapper has to declare them itself.
    writeln!(ptx, "    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;").unwrap();
    writeln!(ptx, "    cvta.shared.u64 %shmem_base, shmem;").unwrap();
}

fn run_ptxas_for_sm(sm: u32) {
    let cfg = canonical_config(32, 32, 32, 2048);
    let mut ptx = String::new();
    write_minimal_kernel_wrapper(&mut ptx, sm);
    emit_q_projection(&mut ptx, &cfg, 128);
    ptx.push_str("    ret;\n}\n");

    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "tier_b1_q_proj_sm{}_{}.ptx",
        sm,
        std::process::id()
    ));
    std::fs::write(&tmp, &ptx).expect("write ptx temp");

    let out = Command::new("ptxas")
        .arg("--gpu-name")
        .arg(format!("sm_{}", sm))
        .arg("-c")
        .arg(&tmp)
        .output()
        .expect("ptxas not on PATH; install CUDA toolkit");

    assert!(
        out.status.success(),
        "ptxas sm_{} failed:\nPTX file: {}\nSTDOUT:\n{}\nSTDERR:\n{}",
        sm,
        tmp.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_q_projection_ptxas_sm_120() {
    run_ptxas_for_sm(120);
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_q_projection_ptxas_sm_80() {
    run_ptxas_for_sm(80);
}
