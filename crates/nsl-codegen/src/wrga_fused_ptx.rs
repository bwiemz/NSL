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

/// Config for `synthesize_fused_gatedlora_ptx`.  Mirrors `FusedLoraConfig`
/// field-for-field; the produced kernel emits the `PerColumnSigmoid` fold
/// variant via the shared `emit_fused_adapter_kernel_body`.
///
/// Separate type so kernel-dedup and dispatch don't confuse GatedLoRA and
/// LoRA kernel instances (different kernel_handle ⇒ different PTX registry
/// entry).
///
/// scale is NOT a field — it flows at launch time via `.param .f32 scale`
/// (B.3 dedup invariant; see spec Risk #5).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedGatedLoraConfig {
    pub site_id: String,
    pub m: u32,         // batch
    pub n: u32,         // d_out
    pub k: u32,         // k_in (shared dim of x@W)
    pub rank: u32,      // ≤ 16
    pub target_sm: u32, // 80, 86, ...
                        // scale is intentionally NOT a field — passed at launch time as
                        // .param .f32.  See dedup notes in B.3 spec Risk #5.
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

/// Variation point for `emit_fused_adapter_kernel_body`'s fold step.
/// LoRA uses `Scalar`; GatedLoRA uses `PerColumnSigmoid`.
///
/// Note: scale is NOT carried here — it's a `.param .f32 scale` loaded
/// into `%scale_reg` at the kernel prolog, same for both adapter types.
/// B.3 spec Risk #5: scale-as-param enables kernel dedup across sites
/// with different alpha values.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FoldKind {
    /// LoRA: `main_accum += epi_final * %scale_reg`  (uniform scalar scale from param)
    Scalar,
    /// GatedLoRA: `main_accum += epi_final * sigmoid(gate[col]) * %scale_reg`  (per-column)
    PerColumnSigmoid {
        gate_ptr_param_name: &'static str,
        gate_load_phase: GateLoadPhase,
        partial_tile_mask: PartialTileMask,
    },
}

/// Scheduling hint for when to emit gate-load + sigmoid computation in the
/// PerColumnSigmoid fused body.
///
/// B.3.1 ships with `LastKIter` unconditionally; all shipped configs have
/// `k_iters >= 1` for the epilogue path.  `PostLoop` is reserved for a
/// potential future milestone that benchmarks load-phase alternatives.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateLoadPhase {
    /// Emit gate load + sigmoid at the end of the final main K-iteration,
    /// before its bar.sync.  Overlaps HBM load latency with the final MMA.
    LastKIter,
    /// Emit gate load + sigmoid after the main K-loop completes, before
    /// the post-loop (x@A)@B MMA.  Has NO emission site in B.3.1; reserved.
    PostLoop,
}

/// Partial-tile handling strategy for sub-MMA output tiles (n < 8).
///
/// B.3.1 ships `FoldResultMask` unconditionally; `SentinelGate` is reserved
/// for a future variant that stages gate through SMEM with a sentinel value.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartialTileMask {
    /// Mask fold contribution with @%p_colN predicate; OOB gate reads
    /// are harmless because the fold output is discarded.
    FoldResultMask,
    /// Pre-populate OOB gate SMEM slots with -20.0 such that sigmoid(-20) ≈ 0
    /// naturally zeros the contribution.  Not emitted by B.3.1.
    SentinelGate,
}

const MMA_K_U32: u32 = 16; // m16n8k16

/// Shared kernel body for fused adapter matmul kernels.
///
/// Emits the full kernel entry + body for either LoRA (FoldKind::Scalar)
/// or GatedLoRA (FoldKind::PerColumnSigmoid).  Variation points:
///   - Param block: PerColumnSigmoid adds `.param .u64 gate_ptr`
///   - Fold step: Scalar uses %scale_reg uniform multiply; PerColumnSigmoid
///     uses per-thread sigmoid(gate) multiply (implemented in Task 4.1;
///     this task leaves the PerColumnSigmoid fold as a placeholder that
///     emits the same Scalar math so unit tests compile and run plausibly).
///
/// All other phases (header, SMEM, register pool, indexing, param loads,
/// output coords, accumulator init, main K-loop, post-loop (x@A)@B MMA,
/// f32→packed-f16 conversion for final MMA A-operand, store, ret) are
/// identical between the two FoldKind variants.
#[allow(dead_code)]
fn emit_fused_adapter_kernel_body(
    ptx: &mut String,
    entry_name: &str,
    m: u32,
    n: u32,
    k: u32,
    rank: u32,
    fold: FoldKind,
) {
    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    use crate::kernel_skeleton::indexing::emit_thread_lane_warp_register_init;
    use crate::kernel_skeleton::params::{
        emit_ld_param_f32, emit_ld_param_u32, emit_ld_param_u64, emit_param_block, Param, ParamTy,
    };
    use crate::kernel_skeleton::smem::{emit_shmem_base_cvta, emit_static_smem_decl};
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::wrga_kernel_helpers::{
        emit_lora_output_tile_coords_dynamic, emit_lora_register_pool, emit_lora_stage_a_tile,
        emit_lora_stage_b_tile, emit_lora_stage_w_tile, emit_lora_stage_x_tile,
        emit_lora_store_output, emit_lora_tile_bases, emit_matmul_mma_lane_init,
        emit_zero_accumulators, wrga_lora_register_budget,
    };

    // Build a FusedLoraConfig just for register budget computation.
    // target_sm=80 matches the Sm80 header below; this function always targets sm_80+.
    let budget_cfg = crate::wrga_fused_ptx::FusedLoraConfig {
        site_id: String::new(),
        m,
        n,
        k,
        rank,
        target_sm: 80,
    };
    let budget = wrga_lora_register_budget(&budget_cfg);

    // 1. File header.
    emit_ptx_header(ptx, PtxVersion::V7_0, TargetSm::Sm80);

    // 2. Param block (opens entry + `{` body).
    // Variation point A: PerColumnSigmoid appends gate_ptr.
    let mut params = vec![
        Param { ty: ParamTy::U64, name: "x_ptr" },
        Param { ty: ParamTy::U64, name: "w_ptr" },
        Param { ty: ParamTy::U64, name: "a_ptr" },
        Param { ty: ParamTy::U64, name: "b_ptr" },
        Param { ty: ParamTy::F32, name: "scale" },
        Param { ty: ParamTy::U64, name: "y_ptr" },
        // m_rows and n_cols are passed as runtime u32 parameters so the same
        // kernel binary handles any batch size.  The compile-time m/n
        // are only used for staging (k-iteration count, rank padding) and
        // kernel naming; predication uses the runtime values.
        Param { ty: ParamTy::U32, name: "m_rows" },
        Param { ty: ParamTy::U32, name: "n_cols" },
    ];
    if matches!(fold, FoldKind::PerColumnSigmoid { .. }) {
        params.push(Param { ty: ParamTy::U64, name: "gate_ptr" });
    }
    emit_param_block(ptx, entry_name, &params);

    // 3. SMEM decl + register pool + shmem base + thread/lane/warp init.
    emit_static_smem_decl(ptx, 1536);
    emit_lora_register_pool(ptx, &budget);
    // u32 tile-base shadow regs for SMEM addressing.
    ptx.push_str("    .reg .u32 %smem_base_x_u32, %smem_base_w_u32, %smem_base_a_u32, %smem_base_b_u32;\n");
    // f32 and b16 scratch for tile-staging f32→f16 conversion.
    // NSL GPU tensors are f32 (dtype=1); the MMA kernel requires f16 operands in SMEM.
    // Staging loads each f32 element, converts to f16 with cvt.rn.f16.f32, and packs
    // two f16 values into a b32 for SMEM storage.
    ptx.push_str("    .reg .f32 %stg_f0, %stg_f1;\n");
    ptx.push_str("    .reg .b16 %stg_h0, %stg_h1;\n");
    // Lane-adjusted B-fragment bases: smem_base_[w|b]_u32 + (tid/4)*2 bytes.
    // Per m16n8k16 B-fragment layout, thread t addresses column (t/4) of the
    // B tile; the column byte offset must be baked into the SMEM base so that
    // emit_load_b_fragment_smem (which only walks k-rows) lands on the right
    // column.  Without this, all threads address column 0, causing SMEM OOB
    // for threads 4..31 (mma_b_row*(row_stride=16) would exceed tile bounds).
    ptx.push_str("    .reg .u32 %smem_base_w_lane_u32, %smem_base_a_lane_u32, %smem_base_b_lane_u32;\n");
    // packed f16x2 regs for converting epi_interm f32 -> f16x2 b32 before final MMA.
    ptx.push_str("    .reg .b32 %epi_packed0, %epi_packed1, %epi_packed2, %epi_packed3;\n");
    // runtime m/n for tile predication (avoids ILLEGAL_ADDRESS on tail rows/cols).
    ptx.push_str("    .reg .u32 %m_param, %n_param;\n");
    emit_shmem_base_cvta(ptx);
    emit_thread_lane_warp_register_init(ptx);
    emit_lora_tile_bases(ptx);
    // Convert u64 tile bases to u32 shared addresses for matmul_mma helpers.
    ptx.push_str("    cvt.u32.u64 %smem_base_x_u32, %x_tile_base;\n");
    ptx.push_str("    cvt.u32.u64 %smem_base_w_u32, %w_tile_base;\n");
    ptx.push_str("    cvt.u32.u64 %smem_base_a_u32, %a_tile_base;\n");
    ptx.push_str("    cvt.u32.u64 %smem_base_b_u32, %b_tile_base;\n");
    emit_matmul_mma_lane_init(ptx);
    // Per-lane B-fragment column base offset in col-major SMEM layout.
    // Col-major B-tile: column c starts at byte c*32 (= c * 16 k-rows * 2 bytes/f16).
    // Thread t accesses column (t/4), so lane column base = (tid/4)*32.
    // mma_b_row (computed in emit_matmul_mma_lane_init) = (tid%4)*4 (byte offset
    // within the column for the k-pair).  Together:
    //   b-frag-addr = smem_base_b_lane_u32 + mma_b_row * 1 + {0, 16}
    //              = smem_base_b_u32 + (tid/4)*32 + (tid%4)*4 + {0, 16}
    // All terms are multiples of 4, so ld.shared.b32 is always 4-byte aligned.
    ptx.push_str("    // Per-lane B-frag column base: (tid/4)*32 bytes (col-major stride=32)\n");
    ptx.push_str("    shr.u32 %r9, %tid_x, 2;\n");
    ptx.push_str("    shl.b32 %r9, %r9, 5;\n");  // * 32 bytes (col stride in col-major)
    ptx.push_str("    add.u32 %smem_base_w_lane_u32, %smem_base_w_u32, %r9;\n");
    // a_tile is also col-major (used as B-operand in epilogue MMA: x @ a_tile).
    ptx.push_str("    add.u32 %smem_base_a_lane_u32, %smem_base_a_u32, %r9;\n");
    ptx.push_str("    add.u32 %smem_base_b_lane_u32, %smem_base_b_u32, %r9;\n");

    // 4. Load params into named registers.
    emit_ld_param_u64(ptx, "%rd_x", "x_ptr");
    emit_ld_param_u64(ptx, "%rd_w", "w_ptr");
    emit_ld_param_u64(ptx, "%rd_a", "a_ptr");
    emit_ld_param_u64(ptx, "%rd_b", "b_ptr");
    emit_ld_param_u64(ptx, "%rd_y", "y_ptr");
    emit_ld_param_f32(ptx, "%scale_reg", "scale");
    emit_ld_param_u32(ptx, "%m_param", "m_rows");
    emit_ld_param_u32(ptx, "%n_param", "n_cols");

    // 5. Output-tile coords + zero accumulators.
    // Use runtime m_param / n_param for predication — see the m_rows/n_cols
    // params added in step 2.  This fixes ILLEGAL_ADDRESS when the prescan
    // config.m (used only for staging) differs from the actual runtime batch.
    emit_lora_output_tile_coords_dynamic(ptx);
    emit_zero_accumulators(ptx, "main_accum", 8);
    emit_zero_accumulators(ptx, "epi_interm", 4);

    // 6. Main K-loop with interleaved epilogue.
    //
    // NOTE: Each K-iteration stages x, w, and a into SMEM, then loads MMA
    // fragments.  The epi_interm MMA reuses %main_a_frag (x tile fragment)
    // as its A-operand — this is correct per spec §3 invariant (1): the
    // m16n8k16 A-fragment encodes only the m×k tile, independent of B.
    let k_iters = (k + MMA_K_U32 - 1) / MMA_K_U32;

    // Fragment register names for matmul_mma load helpers (without % — helpers add it).
    let main_a_frag_names: [String; 4] = [
        "main_a_frag0".into(), "main_a_frag1".into(),
        "main_a_frag2".into(), "main_a_frag3".into(),
    ];
    let main_b_frag_names: [String; 2] = [
        "main_b_frag0".into(), "main_b_frag1".into(),
    ];
    // epi_a_frag is used as B-operand (a_tile is col-major, k=16 × rank columns).
    // m16n8k16 B-fragment requires only 2 b32 registers per thread.
    let epi_a_frag_names: [String; 2] = [
        "epi_a_frag0".into(), "epi_a_frag1".into(),
    ];
    let epi_b_frag_names: [String; 2] = [
        "epi_b_frag0".into(), "epi_b_frag1".into(),
    ];

    // MMA operand arrays (with % — emit_mma_instruction uses names as-is).
    let main_a_frag: [String; 4] = [
        "%main_a_frag0".into(), "%main_a_frag1".into(),
        "%main_a_frag2".into(), "%main_a_frag3".into(),
    ];
    let main_b_frag: [String; 2] = [
        "%main_b_frag0".into(), "%main_b_frag1".into(),
    ];
    let epi_a_frag: [String; 2] = [
        "%epi_a_frag0".into(), "%epi_a_frag1".into(),
    ];
    let epi_b_frag: [String; 2] = [
        "%epi_b_frag0".into(), "%epi_b_frag1".into(),
    ];
    let main_accum: [String; 4] = [
        "%main_accum0".into(), "%main_accum1".into(),
        "%main_accum2".into(), "%main_accum3".into(),
    ];
    let epi_interm: [String; 4] = [
        "%epi_interm0".into(), "%epi_interm1".into(),
        "%epi_interm2".into(), "%epi_interm3".into(),
    ];
    let epi_final: [String; 4] = [
        "%epi_final0".into(), "%epi_final1".into(),
        "%epi_final2".into(), "%epi_final3".into(),
    ];

    for k_tile in 0..k_iters {
        ptx.push_str(&format!("    // ===== K-iteration {} =====\n", k_tile));

        emit_lora_stage_x_tile(ptx, k_tile, m, k);
        emit_lora_stage_w_tile(ptx, k_tile, n, k);
        emit_lora_stage_a_tile(ptx, k_tile, rank, k);

        ptx.push_str("    bar.sync 0;\n");

        // Load A and B fragments from SMEM using u32 tile base addresses.
        // k_tile offset within the tile is handled at stage time (fresh per-iter).
        emit_load_a_fragment_smem(ptx, &main_a_frag_names, "%smem_base_x_u32", 32);
        // row_stride_bytes=1: mma_b_row already holds the full byte offset within the
        // col-major column (= (tid%4)*4).  The lane base (smem_base_w_lane_u32) adds
        // the column base ((tid/4)*32).  k-pair register offset = {0, 16} bytes comes
        // from emit_load_b_fragment_smem's byte_col_offset = k_base_pair*2 = {0, 16}.
        emit_load_b_fragment_smem(ptx, &main_b_frag_names, "%smem_base_w_lane_u32", 1);
        // a_tile is col-major (B-operand in epilogue MMA: x @ a_tile).
        // Same col-major B-fragment addressing as w_tile above.
        emit_load_b_fragment_smem(ptx, &epi_a_frag_names, "%smem_base_a_lane_u32", 1);

        // Main MMA: main_accum += x_tile @ w_tile
        emit_mma_instruction(ptx, &main_accum, &main_a_frag, &main_b_frag, &main_accum);

        // Epilogue interleaved MMA: epi_interm += x_tile @ a_tile
        // Reuses %main_a_frag (from x_tile, row-major) as A-operand; uses
        // epi_a_frag (from a_tile, col-major B-fragment, 2 b32 regs) as B-operand.
        emit_mma_instruction(ptx, &epi_interm, &main_a_frag, &epi_a_frag, &epi_interm);

        ptx.push_str("    bar.sync 0;\n");
    }

    // 7. Post-loop: stage b once, compute (x@A) @ B.
    emit_lora_stage_b_tile(ptx, rank, n);
    ptx.push_str("    bar.sync 0;\n");
    // Same col-major byte-offset addressing as main W-fragment load.
    emit_load_b_fragment_smem(ptx, &epi_b_frag_names, "%smem_base_b_lane_u32", 1);

    // 8. Convert epi_interm (f32) → packed f16x2 (b32) for final MMA A-operand.
    //
    // MMA requires A-operand registers to be .b32 (packed f16 pairs).  epi_interm
    // holds f32 accumulators from the K-loop.  We convert per PTX invariant:
    //   cvt.rn.f16x2.f32 %dst, %f32_hi, %f32_lo
    // Packs: lower f16 = f32_lo, upper f16 = f32_hi.
    // Per m16n8k16 A-fragment layout, each thread holds 8 f16 values packed into
    // 4 b32 registers.  epi_interm[0..3] holds 4 f32 values → 2 packed b32.
    // The upper 2 b32 (for the k=8..15 half) are zero-padded.
    ptx.push_str("    // Pack epi_interm f32 -> f16x2 b32 for final MMA A-operand\n");
    ptx.push_str("    cvt.rn.f16x2.f32 %epi_packed0, %epi_interm1, %epi_interm0;\n");
    ptx.push_str("    cvt.rn.f16x2.f32 %epi_packed1, %epi_interm3, %epi_interm2;\n");
    ptx.push_str("    mov.b32 %epi_packed2, 0;\n");
    ptx.push_str("    mov.b32 %epi_packed3, 0;\n");
    let epi_packed: [String; 4] = [
        "%epi_packed0".into(), "%epi_packed1".into(),
        "%epi_packed2".into(), "%epi_packed3".into(),
    ];

    // Zero-init epi_final and compute (x@A) @ B using packed A-fragments.
    emit_zero_accumulators(ptx, "epi_final", 4);
    emit_mma_instruction(ptx, &epi_final, &epi_packed, &epi_b_frag, &epi_final);

    // 9. Scale epi_final, fold into main_accum.
    // Variation point B: Scalar uses %scale_reg uniform multiply;
    // PerColumnSigmoid placeholder emits the same Scalar math for now —
    // Task 4.1 replaces it with emit_gatedlora_fold() after the gate-load
    // helpers from Task Group 2 are in place.
    match fold {
        FoldKind::Scalar => {
            // LoRA fold: main_accum += epi_final * scale  (uniform scalar from param)
            for i in 0..4u32 {
                ptx.push_str(&format!(
                    "    mul.f32 %epi_final{i}, %epi_final{i}, %scale_reg;\n"
                ));
                ptx.push_str(&format!(
                    "    add.f32 %main_accum{i}, %main_accum{i}, %epi_final{i};\n"
                ));
            }
        }
        FoldKind::PerColumnSigmoid { .. } => {
            // GatedLoRA fold — implemented in Task 4.1.  For now, placeholder
            // emits a bar.sync so the kernel body still parses if reached
            // (though Task 3.1's stub synthesizer won't call us yet).
            ptx.push_str("    // GatedLoRA fold placeholder; Task 4.1 replaces this\n");
            ptx.push_str("    // with emit_gatedlora_fold() from wrga_kernel_helpers.rs\n");
            for i in 0..4u32 {
                ptx.push_str(&format!(
                    "    mul.f32 %epi_final{i}, %epi_final{i}, %scale_reg;\n"
                ));
                ptx.push_str(&format!(
                    "    add.f32 %main_accum{i}, %main_accum{i}, %epi_final{i};\n"
                ));
            }
        }
    }

    // 10. Store to y with predication.
    emit_lora_store_output(ptx, n);

    // 11. Close the entry body.
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

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

    let entry_name = format!(
        "nsl_wrga_fused_lora_m{}n{}k{}r{}",
        config.m, config.n, config.k, config.rank,
    );
    let mut ptx = String::new();
    emit_fused_adapter_kernel_body(
        &mut ptx,
        &entry_name,
        config.m,
        config.n,
        config.k,
        config.rank,
        FoldKind::Scalar,
    );
    ptx
}

/// Synthesize the full PTX for an epilogue-fused IA³ matmul kernel.
///
/// IA³ epilogue: `y = (x @ W) * γ` — one broadcast-mul after the main
/// matmul.  No epilogue interleaving is needed (unlike LoRA).
///
/// Kernel structure mirrors the LoRA synthesizer but:
///   - No A/B adapter matrices or epilogue MMA
///   - No scale parameter
///   - Per-thread γ load: each thread loads 2 f32 values for its 2 owned
///     output columns, avoiding PTX's lack of dynamic register indexing
pub fn synthesize_fused_ia3_ptx(config: &FusedIa3Config) -> String {
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    use crate::kernel_skeleton::indexing::emit_thread_lane_warp_register_init;
    use crate::kernel_skeleton::params::{
        emit_ld_param_u32, emit_ld_param_u64, emit_param_block, Param, ParamTy,
    };
    use crate::kernel_skeleton::smem::{emit_shmem_base_cvta, emit_static_smem_decl};
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::wrga_kernel_helpers::{
        emit_ia3_gamma_multiply, emit_ia3_load_gamma, emit_ia3_register_pool,
        emit_ia3_store_output, emit_ia3_tile_bases, emit_lora_output_tile_coords_dynamic,
        emit_lora_stage_w_tile, emit_lora_stage_x_tile, emit_matmul_mma_lane_init,
        emit_zero_accumulators, wrga_ia3_register_budget,
    };

    let mut ptx = String::new();
    let budget = wrga_ia3_register_budget(config);

    // 1. File header.
    emit_ptx_header(&mut ptx, PtxVersion::V7_0, TargetSm::Sm80);

    // 2. Param block.
    let entry_name = format!(
        "nsl_wrga_fused_ia3_m{}n{}k{}",
        config.m, config.n, config.k,
    );
    let params = [
        Param { ty: ParamTy::U64, name: "x_ptr" },
        Param { ty: ParamTy::U64, name: "w_ptr" },
        Param { ty: ParamTy::U64, name: "gamma_ptr" },
        Param { ty: ParamTy::U64, name: "y_ptr" },
        // Runtime m/n for tail predication (same pattern as LoRA).
        Param { ty: ParamTy::U32, name: "m_rows" },
        Param { ty: ParamTy::U32, name: "n_cols" },
    ];
    emit_param_block(&mut ptx, &entry_name, &params);

    // 3. SMEM decl + register pool + shmem base + thread/lane/warp init.
    //    IA³ only needs x_tile + w_tile: 512 + 256 = 768 bytes.
    emit_static_smem_decl(&mut ptx, 768);
    emit_ia3_register_pool(&mut ptx, &budget);
    // u32 tile-base shadow for matmul_mma helpers (require u32 addresses).
    ptx.push_str("    .reg .u32 %smem_base_x_u32, %smem_base_w_u32;\n");
    // f32 and b16 scratch for tile-staging f32→f16 conversion.
    ptx.push_str("    .reg .f32 %stg_f0, %stg_f1;\n");
    ptx.push_str("    .reg .b16 %stg_h0, %stg_h1;\n");
    // Per-lane B-fragment column base (same col-major geometry as LoRA W-tile).
    ptx.push_str("    .reg .u32 %smem_base_w_lane_u32;\n");
    // Runtime m/n registers for tile predication.
    ptx.push_str("    .reg .u32 %m_param, %n_param;\n");

    emit_shmem_base_cvta(&mut ptx);
    emit_thread_lane_warp_register_init(&mut ptx);
    emit_ia3_tile_bases(&mut ptx);

    // Convert u64 tile bases to u32 shared addresses for matmul_mma helpers.
    ptx.push_str("    cvt.u32.u64 %smem_base_x_u32, %x_tile_base;\n");
    ptx.push_str("    cvt.u32.u64 %smem_base_w_u32, %w_tile_base;\n");

    emit_matmul_mma_lane_init(&mut ptx);

    // Per-lane B-fragment column base: (tid/4)*32 bytes (col-major stride=32).
    // Identical derivation to LoRA's %smem_base_w_lane_u32.
    ptx.push_str("    // Per-lane B-frag column base: (tid/4)*32 bytes\n");
    ptx.push_str("    shr.u32 %r9, %tid_x, 2;\n");
    ptx.push_str("    shl.b32 %r9, %r9, 5;\n");
    ptx.push_str("    add.u32 %smem_base_w_lane_u32, %smem_base_w_u32, %r9;\n");

    // 4. Load params into named registers.
    emit_ld_param_u64(&mut ptx, "%rd_x", "x_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_w", "w_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_gamma", "gamma_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_y", "y_ptr");
    emit_ld_param_u32(&mut ptx, "%m_param", "m_rows");
    emit_ld_param_u32(&mut ptx, "%n_param", "n_cols");

    // 5. Output-tile coords + zero accumulators.
    emit_lora_output_tile_coords_dynamic(&mut ptx);
    emit_zero_accumulators(&mut ptx, "main_accum", 8);

    // 6. Main K-loop.
    let k_iters = (config.k + MMA_K_U32 - 1) / MMA_K_U32;

    let main_a_frag_names: [String; 4] = [
        "main_a_frag0".into(), "main_a_frag1".into(),
        "main_a_frag2".into(), "main_a_frag3".into(),
    ];
    let main_b_frag_names: [String; 2] = [
        "main_b_frag0".into(), "main_b_frag1".into(),
    ];
    let main_a_frag: [String; 4] = [
        "%main_a_frag0".into(), "%main_a_frag1".into(),
        "%main_a_frag2".into(), "%main_a_frag3".into(),
    ];
    let main_b_frag: [String; 2] = [
        "%main_b_frag0".into(), "%main_b_frag1".into(),
    ];
    let main_accum: [String; 4] = [
        "%main_accum0".into(), "%main_accum1".into(),
        "%main_accum2".into(), "%main_accum3".into(),
    ];

    for k_tile in 0..k_iters {
        ptx.push_str(&format!("    // ===== K-iteration {} =====\n", k_tile));

        emit_lora_stage_x_tile(&mut ptx, k_tile, config.m, config.k);
        emit_lora_stage_w_tile(&mut ptx, k_tile, config.n, config.k);

        ptx.push_str("    bar.sync 0;\n");

        emit_load_a_fragment_smem(&mut ptx, &main_a_frag_names, "%smem_base_x_u32", 32);
        emit_load_b_fragment_smem(&mut ptx, &main_b_frag_names, "%smem_base_w_lane_u32", 1);

        // Main MMA: main_accum += x_tile @ w_tile
        emit_mma_instruction(&mut ptx, &main_accum, &main_a_frag, &main_b_frag, &main_accum);

        ptx.push_str("    bar.sync 0;\n");
    }

    // 7. IA³ epilogue: load γ, broadcast-multiply, store.
    emit_ia3_load_gamma(&mut ptx);
    emit_ia3_gamma_multiply(&mut ptx);
    emit_ia3_store_output(&mut ptx, config.n);

    // 8. Close the entry body.
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    ptx
}

/// Synthesize PTX for a fused GatedLoRA forward adapter kernel.
///
/// Computes `y[i,j] = (x @ W)[i,j] + sigmoid(gate[j]) * ((x @ A) @ B)[i,j] * scale`
/// where `gate` is a per-output-column f32 vector.
///
/// Uses the shared `emit_fused_adapter_kernel_body` with
/// `FoldKind::PerColumnSigmoid` to reuse LoRA's proven kernel body plus
/// gate-load + sigmoid + per-column fold logic from `wrga_kernel_helpers`.
///
/// STUB — current body returns an invalid PTX header-only stub for
/// Task 3.2's red-state test setup.  Task 4.1 replaces this with the real
/// emission via `emit_fused_adapter_kernel_body(.., FoldKind::PerColumnSigmoid { .. })`.
pub fn synthesize_fused_gatedlora_ptx(config: &FusedGatedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    // STUB: intentionally invalid PTX to establish red test state.
    // Task 4.1 replaces this with emit_fused_adapter_kernel_body(..,
    // FoldKind::PerColumnSigmoid { .. }).
    //
    // The stub emits a .entry with a deliberately invalid instruction
    // (`this_is_not_a_real_ptx_opcode`) so ptxas rejects it.  An empty module
    // (header + comment only) is valid PTX and would silently pass ptxas.
    let sm = config.target_sm;
    format!(
        ".version 7.0\n\
         .target sm_{sm}\n\
         .address_size 64\n\
         \n\
         // STUB — Task 4.1 replaces this body\n\
         .visible .entry gatedlora_stub ()\n\
         {{\n\
             this_is_not_a_real_ptx_opcode;\n\
         }}\n"
    )
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
        // m16n8k16 epi_final accumulator is 4 f32 values (one D-fragment),
        // so the fold loop runs 0..4 (not 0..8).
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        for i in 0..4 {
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
