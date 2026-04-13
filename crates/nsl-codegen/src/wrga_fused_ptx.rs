//! WRGA B.3 Task 2: epilogue-fused LoRA/IA³ MMA PTX generator.
//!
//! LoRA kernel structure (CRITICAL — interleaved epilogue):
//! The naive "main loop then epilogue" is WRONG because SMEM is
//! overwritten each K-iteration; post-loop access would only see
//! the last x tile.  Instead: each main K-iteration also performs
//! x_tile @ A_tile accumulating into an `epilogue_intermediate`
//! register, so `epilogue_intermediate == x @ A` after the K-loop.
//! The final (x@A) @ B * scale is then folded into main_accum
//! before storing to y.
//!
//! IA³ is simpler: no interleaving, just a post-loop γ-broadcast-mul.
//!
//! Scale is a `.param .f32` — see the `scale` parameter below.  This
//! enables kernel dedup across sites with different alpha values.

use crate::matmul_mma;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedLoraConfig {
    pub site_id: String,
    pub m: u32,         // batch
    pub n: u32,         // d_out
    pub k: u32,         // k_in (shared dim of x@W)
    pub rank: u32,      // ≤ 16
    pub target_sm: u32, // 80, 86, ...
                        // scale is intentionally NOT a field — passed at launch time as
                        // .param .f32.  See dedup notes in B.3 spec Risk #5.
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedIa3Config {
    pub site_id: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub target_sm: u32,
}

/// Kernel cache key for dedup.  Sites with matching key share one PTX.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoraKernelKey {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rank: u32,
    pub target_sm: u32,
}

impl FusedLoraConfig {
    pub fn kernel_key(&self) -> LoraKernelKey {
        LoraKernelKey {
            m: self.m,
            n: self.n,
            k: self.k,
            rank: self.rank,
            target_sm: self.target_sm,
        }
    }
}

const MMA_K_U32: u32 = 16; // m16n8k16

/// Synthesize the full PTX for an epilogue-fused LoRA matmul kernel.
///
/// **Rank ceiling:** `config.rank <= 16` (single-pass epilogue).  Caller
/// must validate before invoking; this fn panics on violation (caller
/// error — decorator inject pass enforces a proper compile error).
pub fn synthesize_fused_lora_ptx(config: &FusedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    let mut ptx = String::new();

    // Header.
    ptx.push_str(".version 7.0\n");
    ptx.push_str(&format!(".target sm_{}\n", config.target_sm));
    ptx.push_str(".address_size 64\n\n");

    ptx.push_str(&format!(
        ".visible .entry nsl_wrga_fused_lora_m{}n{}k{}r{}(\n",
        config.m, config.n, config.k, config.rank,
    ));
    ptx.push_str("    .param .u64 x_ptr,\n");
    ptx.push_str("    .param .u64 w_ptr,\n");
    ptx.push_str("    .param .u64 a_ptr,\n");
    ptx.push_str("    .param .u64 b_ptr,\n");
    ptx.push_str("    .param .f32 scale,       // alpha/rank — launch-time value for dedup\n");
    ptx.push_str("    .param .u64 y_ptr\n");
    ptx.push_str(")\n{\n");

    // Register decls.
    ptx.push_str("    // === Register declarations ===\n");
    ptx.push_str("    .reg .f32 %main_accum<8>;        // x @ W f32 accumulator\n");
    ptx.push_str("    .reg .b32 %main_a_frag<4>;       // current x tile fragment\n");
    ptx.push_str("    .reg .b32 %main_b_frag<2>;       // current W tile fragment\n");
    ptx.push_str("    .reg .b32 %epi_a_frag<4>;        // A-matrix tile (NOT x@A!)\n");
    ptx.push_str("    .reg .f32 %epi_interm<4>;        // incremental x @ A accumulator\n");
    ptx.push_str("    .reg .b32 %epi_b_frag<2>;        // B-matrix tile\n");
    ptx.push_str("    .reg .f32 %epi_final<8>;         // (x@A) @ B accumulator\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w, %smem_base_a, %smem_base_b;\n");
    ptx.push_str("    .reg .u32 %k_idx, %mma_addr, %mma_a_row, %mma_b_row;\n");
    ptx.push_str("    .reg .pred %k_pred;\n");
    ptx.push_str("    .reg .f32 %scale_reg;\n\n");

    // Load scale once before the K-loop.
    ptx.push_str("    // Load scale (alpha/rank) from parameter — dedup-friendly\n");
    ptx.push_str("    ld.param.f32 %scale_reg, [scale];\n\n");

    // Init main_accum + epi_interm to zero.
    ptx.push_str("    // Init accumulators\n");
    for i in 0..8 {
        ptx.push_str(&format!("    mov.f32 %main_accum{}, 0f00000000;\n", i));
    }
    for i in 0..4 {
        ptx.push_str(&format!("    mov.f32 %epi_interm{}, 0f00000000;\n", i));
    }
    ptx.push_str("\n");

    // Interleaved K-tile loop.
    let k_iters = config.k / MMA_K_U32;
    ptx.push_str(&format!(
        "    // === Interleaved main K-loop: {} tiles ===\n",
        k_iters,
    ));
    for k_tile in 0..k_iters {
        ptx.push_str(&format!("    // --- K-tile {} ---\n", k_tile));
        ptx.push_str(&format!("    // Load x tile [k={}]\n", k_tile * MMA_K_U32));
        emit_load_frag_a_main(&mut ptx, k_tile);
        ptx.push_str(&format!("    // Load W tile [k={}]\n", k_tile * MMA_K_U32));
        emit_load_frag_b_main(&mut ptx, k_tile);
        // (a) Main MMA: main_accum += x_tile @ W_tile
        emit_main_mma(&mut ptx);
        // (c) Load A tile for this K-chunk.
        ptx.push_str(&format!("    // Load A tile [k={}]\n", k_tile * MMA_K_U32));
        emit_load_frag_a_epi(&mut ptx, k_tile);
        // (b) Epilogue MMA: epi_interm += x_tile @ A_tile
        emit_epi_interm_mma(&mut ptx);
    }
    ptx.push_str("\n");

    // Post-loop epilogue.
    ptx.push_str("    // === Post-loop epilogue: (x@A) @ B * scale, fold into main_accum ===\n");
    ptx.push_str("    // Load B tile\n");
    emit_load_frag_b_epi(&mut ptx);
    emit_epi_final_mma(&mut ptx);
    // epi_final *= scale
    for i in 0..8 {
        ptx.push_str(&format!(
            "    mul.f32 %epi_final{}, %epi_final{}, %scale_reg;\n",
            i, i,
        ));
    }
    // main_accum += epi_final (the "epilogue fusion" step)
    for i in 0..8 {
        ptx.push_str(&format!(
            "    add.f32 %main_accum{}, %main_accum{}, %epi_final{};\n",
            i, i, i,
        ));
    }
    ptx.push_str("\n");

    // Store main_accum to y.
    ptx.push_str("    // === Store y ===\n");
    emit_store_y(&mut ptx);

    ptx.push_str("}\n");
    ptx
}

fn emit_load_frag_a_main(ptx: &mut String, k_tile: u32) {
    let regs: [String; 4] = [
        "main_a_frag0".into(),
        "main_a_frag1".into(),
        "main_a_frag2".into(),
        "main_a_frag3".into(),
    ];
    matmul_mma::emit_load_a_fragment_smem(
        ptx,
        &regs,
        &format!("%smem_base_x + {}", k_tile * 32),
        32,
    );
}
fn emit_load_frag_b_main(ptx: &mut String, k_tile: u32) {
    let regs: [String; 2] = ["main_b_frag0".into(), "main_b_frag1".into()];
    matmul_mma::emit_load_b_fragment_smem(
        ptx,
        &regs,
        &format!("%smem_base_w + {}", k_tile * 16),
        16,
    );
}
fn emit_load_frag_a_epi(ptx: &mut String, k_tile: u32) {
    let regs: [String; 4] = [
        "epi_a_frag0".into(),
        "epi_a_frag1".into(),
        "epi_a_frag2".into(),
        "epi_a_frag3".into(),
    ];
    matmul_mma::emit_load_a_fragment_smem(
        ptx,
        &regs,
        &format!("%smem_base_a + {}", k_tile * 32),
        32,
    );
}
fn emit_load_frag_b_epi(ptx: &mut String) {
    let regs: [String; 2] = ["epi_b_frag0".into(), "epi_b_frag1".into()];
    matmul_mma::emit_load_b_fragment_smem(ptx, &regs, "%smem_base_b", 16);
}
fn emit_main_mma(ptx: &mut String) {
    let d: [String; 4] = [
        "main_accum0".into(),
        "main_accum1".into(),
        "main_accum2".into(),
        "main_accum3".into(),
    ];
    let a: [String; 4] = [
        "main_a_frag0".into(),
        "main_a_frag1".into(),
        "main_a_frag2".into(),
        "main_a_frag3".into(),
    ];
    let b: [String; 2] = ["main_b_frag0".into(), "main_b_frag1".into()];
    let c: [String; 4] = [
        "main_accum0".into(),
        "main_accum1".into(),
        "main_accum2".into(),
        "main_accum3".into(),
    ];
    matmul_mma::emit_mma_instruction(ptx, &d, &a, &b, &c);
}
fn emit_epi_interm_mma(ptx: &mut String) {
    let d: [String; 4] = [
        "epi_interm0".into(),
        "epi_interm1".into(),
        "epi_interm2".into(),
        "epi_interm3".into(),
    ];
    let a: [String; 4] = [
        "main_a_frag0".into(),
        "main_a_frag1".into(),
        "main_a_frag2".into(),
        "main_a_frag3".into(),
    ];
    let b: [String; 2] = ["epi_a_frag0".into(), "epi_a_frag1".into()];
    let c: [String; 4] = [
        "epi_interm0".into(),
        "epi_interm1".into(),
        "epi_interm2".into(),
        "epi_interm3".into(),
    ];
    matmul_mma::emit_mma_instruction(ptx, &d, &a, &b, &c);
}
fn emit_epi_final_mma(ptx: &mut String) {
    let d: [String; 4] = [
        "epi_final0".into(),
        "epi_final1".into(),
        "epi_final2".into(),
        "epi_final3".into(),
    ];
    let a: [String; 4] = [
        "epi_interm0".into(),
        "epi_interm1".into(),
        "epi_interm2".into(),
        "epi_interm3".into(),
    ];
    let b: [String; 2] = ["epi_b_frag0".into(), "epi_b_frag1".into()];
    let c: [String; 4] = [
        "epi_final0".into(),
        "epi_final1".into(),
        "epi_final2".into(),
        "epi_final3".into(),
    ];
    matmul_mma::emit_mma_instruction(ptx, &d, &a, &b, &c);
}
fn emit_store_y(ptx: &mut String) {
    ptx.push_str("    // (store sequence — per-thread writes of main_accum to global y)\n");
    for i in 0..8 {
        ptx.push_str(&format!(
            "    st.global.f32 [%y_ptr + {}], %main_accum{};\n",
            i * 4,
            i,
        ));
    }
}

/// Synthesize the full PTX for an epilogue-fused IA³ matmul kernel.
/// IA³'s epilogue is `y = (x @ W) * γ` — one broadcast-mul after the
/// main matmul.  No epilogue interleaving needed.
pub fn synthesize_fused_ia3_ptx(config: &FusedIa3Config) -> String {
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");
    let mut ptx = String::new();
    ptx.push_str(".version 7.0\n");
    ptx.push_str(&format!(".target sm_{}\n", config.target_sm));
    ptx.push_str(".address_size 64\n\n");
    ptx.push_str(&format!(
        ".visible .entry nsl_wrga_fused_ia3_m{}n{}k{}(\n",
        config.m, config.n, config.k,
    ));
    ptx.push_str("    .param .u64 x_ptr,\n");
    ptx.push_str("    .param .u64 w_ptr,\n");
    ptx.push_str("    .param .u64 gamma_ptr,\n");
    ptx.push_str("    .param .u64 y_ptr\n");
    ptx.push_str(")\n{\n");
    ptx.push_str("    // === Register declarations (IA³ — simpler than LoRA) ===\n");
    ptx.push_str("    .reg .f32 %main_accum<8>;\n");
    ptx.push_str("    .reg .b32 %main_a_frag<4>;\n");
    ptx.push_str("    .reg .b32 %main_b_frag<2>;\n");
    ptx.push_str("    .reg .f32 %gamma<8>;\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w;\n");
    ptx.push_str("    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;\n\n");
    for i in 0..8 {
        ptx.push_str(&format!("    mov.f32 %main_accum{}, 0f00000000;\n", i));
    }
    let k_iters = config.k / MMA_K_U32;
    for k_tile in 0..k_iters {
        emit_load_frag_a_main(&mut ptx, k_tile);
        emit_load_frag_b_main(&mut ptx, k_tile);
        emit_main_mma(&mut ptx);
    }
    ptx.push_str("    // === IA³ epilogue: main_accum *= gamma (broadcast) ===\n");
    for i in 0..8 {
        ptx.push_str(&format!(
            "    ld.global.f32 %gamma{}, [%gamma_ptr + {}];\n",
            i,
            i * 4,
        ));
        ptx.push_str(&format!(
            "    mul.f32 %main_accum{}, %main_accum{}, %gamma{};\n",
            i, i, i,
        ));
    }
    emit_store_y(&mut ptx);
    ptx.push_str("}\n");
    ptx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_lora_config() -> FusedLoraConfig {
        FusedLoraConfig {
            site_id: "test.w".into(),
            m: 16,
            n: 8,
            k: 16,
            rank: 2,
            target_sm: 80,
        }
    }

    #[test]
    fn lora_ptx_uses_scale_as_param_not_literal() {
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        assert!(
            ptx.contains(".param .f32 scale"),
            "scale must be .param for kernel dedup; got PTX:\n{ptx}"
        );
        assert!(
            ptx.contains("ld.param.f32 %scale_reg, [scale]"),
            "kernel must load scale from param at entry"
        );
        assert!(
            ptx.contains("mul.f32 %epi_final0, %epi_final0, %scale_reg"),
            "epilogue mul must use %scale_reg"
        );
    }

    #[test]
    fn lora_ptx_emits_main_and_epilogue_mmas_per_k_tile() {
        // k=16 → 1 K-tile iteration; expected 1 main + 1 epi_interm + 1 epi_final = 3.
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 3,
            "LoRA k=16 expects 3 MMA (1 main + 1 epi_interm + 1 epi_final); got {mma_count}\nPTX:\n{ptx}"
        );
    }

    #[test]
    fn lora_ptx_folds_epilogue_into_main_accum() {
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        for i in 0..8 {
            let expected =
                format!("add.f32 %main_accum{}, %main_accum{}, %epi_final{}", i, i, i);
            assert!(
                ptx.contains(&expected),
                "missing fold step: {expected}\nPTX:\n{ptx}"
            );
        }
    }

    #[test]
    fn lora_ptx_rank_above_16_panics() {
        let cfg = FusedLoraConfig {
            site_id: "test.w".into(),
            m: 16,
            n: 8,
            k: 16,
            rank: 17,
            target_sm: 80,
        };
        let res = std::panic::catch_unwind(|| synthesize_fused_lora_ptx(&cfg));
        assert!(res.is_err(), "rank > 16 must panic — caller must enforce beforehand");
    }

    #[test]
    fn ia3_ptx_emits_single_mma_and_gamma_broadcast() {
        let cfg = FusedIa3Config {
            site_id: "test.w".into(),
            m: 16,
            n: 8,
            k: 16,
            target_sm: 80,
        };
        let ptx = synthesize_fused_ia3_ptx(&cfg);
        assert_eq!(
            ptx.matches("mma.sync.aligned.m16n8k16").count(),
            1,
            "IA³ must emit exactly 1 MMA (no epilogue matmul); got PTX:\n{ptx}"
        );
        assert!(
            ptx.contains("mul.f32 %main_accum0, %main_accum0, %gamma0"),
            "IA³ must broadcast-mul main_accum by gamma"
        );
        assert!(
            !ptx.contains(".param .f32 scale"),
            "IA³ has no scale parameter"
        );
    }

    #[test]
    fn dedup_key_ignores_site_id() {
        let a = FusedLoraConfig {
            site_id: "blocks.0.wq".into(),
            m: 16,
            n: 8,
            k: 16,
            rank: 4,
            target_sm: 80,
        };
        let b = FusedLoraConfig {
            site_id: "blocks.1.wq".into(),
            m: 16,
            n: 8,
            k: 16,
            rank: 4,
            target_sm: 80,
        };
        assert_eq!(
            a.kernel_key(),
            b.kernel_key(),
            "sites with same dims+rank+sm must share kernel — key must exclude site_id"
        );
    }
}
