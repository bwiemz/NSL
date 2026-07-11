//! CPKD Innovation 2 — fused KL-CE distillation loss kernel synthesis.
//!
//! Computes the standard knowledge-distillation loss
//!
//! ```text
//! loss = alpha * CE(student_logits, labels)
//!      + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
//! ```
//!
//! WITHOUT materializing either `[rows, vocab]` logit tensor in HBM. One
//! CTA per token row streams the vocabulary in SMEM tiles; each tile holds
//! BOTH the student and the teacher logits for that vocab slice, and thread
//! 0 maintains three online-softmax accumulator families:
//!
//! - `(m_s1, S_s1)` — student logits at temperature 1 (for the CE term);
//! - `(m_sT, S_sT)` — student logits / T (for `LSE_T(s)`);
//! - `(m_tT, S_tT, C)` — teacher logits / T plus the KL cross-term
//!   `C = sum_v exp(t_v/T - m_tT) * (t_v - s_v)`, rescaled alongside `S_tT`
//!   whenever the running max advances (same rescale trick as the sum).
//!
//! Per valid row the kernel emits:
//!
//! ```text
//! CE  = lse_s1 - s_target                     (lse = m + ln S)
//! KL  = (C / S_tT) / T - lse_tT + lse_sT
//! loss_row = alpha * CE + (1 - alpha) * T^2 * KL
//! ```
//!
//! and saves the three per-row LSEs (`lse_s1`, `lse_sT`, `lse_tT`) to HBM —
//! `3 * rows * 4` bytes, the only forward state the backward needs.
//!
//! The backward re-computes both logits per (row, vocab) pair — mirroring
//! the fused linear-CE backward's "recompute, never save" design — and
//! forms
//!
//! ```text
//! dloss/ds_v = alpha * (p_s1(v) - 1{v==target})
//!            + (1 - alpha) * T * (p_sT(v) - p_tT(v))
//! ```
//!
//! scattering `dx_s`, `dW_s`, `dbias_s` via `red.global.add.f32`.
//! **The teacher receives no gradients by construction** — there are no
//! teacher-gradient outputs in the kernel ABI at all (composition-paper
//! invariant I-11 enforced at the ABI level, not by a runtime check).
//!
//! ## v1 scope (deferrals refuse loudly in `validate`)
//!
//! - f32 only (the fp16/bf16 mixed-precision path of `fused_linear_ce.rs`
//!   is a mechanical extension, deferred);
//! - `vocab_size <= 8192` (single-CTA path; the two-kernel large-vocab
//!   split is deferred);
//! - spectral top-k logit compression is ADVISORY in v1 (`cpkd_spectral`
//!   reports the effective rank; this kernel always runs the full vocab).
//!
//! PTX conventions follow `fused_linear_ce.rs`: null-terminated module
//! bytes (`cuModuleLoadData` contract), `.version 7.0` / `.target
//! sm_max(80, gpu_sm)`, ASCII-only comments (ptxas 13.x rejects Unicode),
//! `ex2.approx/lg2.approx` with log2(e)/ln(2) conversion constants.

use serde::Serialize;

/// Hard ceiling for the v1 single-CTA path (mirrors
/// `fused_linear_ce::LARGE_VOCAB_THRESHOLD`; the large-vocab two-kernel
/// variant is a deferred extension).
pub const KL_CE_MAX_VOCAB_V1: u32 = 8192;

/// Compile-time configuration for one fused KL-CE kernel pair.
///
/// Shapes are decorator-supplied compile-time constants (same static-shape
/// contract as `@fused_lm_ce`); alpha/temperature are RUNTIME kernel
/// parameters, so one PTX pair serves any (alpha, T) without respecialization.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FusedKlCeConfig {
    pub vocab_size: u32,
    /// Student hidden dim (must be % 32 == 0).
    pub student_hidden: u32,
    /// Teacher hidden dim (must be % 32 == 0; may differ from the student's).
    pub teacher_hidden: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    /// Vocab tile held in SMEM (% 128 == 0, <= vocab_size).
    pub vocab_tile: u32,
    pub gpu_sm: u32,
    pub ignore_index: i64,
}

impl Default for FusedKlCeConfig {
    fn default() -> Self {
        FusedKlCeConfig {
            vocab_size: 0,
            student_hidden: 0,
            teacher_hidden: 0,
            batch_size: 0,
            seq_len: 0,
            vocab_tile: 1024,
            gpu_sm: 80,
            ignore_index: -100,
        }
    }
}

impl FusedKlCeConfig {
    /// Validate the v1 preconditions. Every violated precondition names the
    /// deferral it corresponds to (repo doctrine: refuse loudly).
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 || self.student_hidden == 0 || self.teacher_hidden == 0 {
            return Err("fused_kl_ce: vocab_size/student_hidden/teacher_hidden must be non-zero".into());
        }
        if self.vocab_size > KL_CE_MAX_VOCAB_V1 {
            return Err(format!(
                "fused_kl_ce v1 supports vocab_size <= {} (single-CTA path); got {}. \
                 The large-vocab two-kernel variant is a deferred extension",
                KL_CE_MAX_VOCAB_V1, self.vocab_size
            ));
        }
        if !self.student_hidden.is_multiple_of(32) {
            return Err(format!(
                "fused_kl_ce: student_hidden must be a multiple of 32; got {}",
                self.student_hidden
            ));
        }
        if !self.teacher_hidden.is_multiple_of(32) {
            return Err(format!(
                "fused_kl_ce: teacher_hidden must be a multiple of 32; got {}",
                self.teacher_hidden
            ));
        }
        if self.vocab_tile == 0 || !self.vocab_tile.is_multiple_of(128) {
            return Err(format!(
                "fused_kl_ce: vocab_tile must be a non-zero multiple of 128; got {}",
                self.vocab_tile
            ));
        }
        if self.vocab_tile > self.vocab_size.next_multiple_of(128) {
            return Err(format!(
                "fused_kl_ce: vocab_tile ({}) exceeds padded vocab_size ({})",
                self.vocab_tile,
                self.vocab_size.next_multiple_of(128)
            ));
        }
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err("fused_kl_ce: batch_size and seq_len must be non-zero".into());
        }
        Ok(())
    }

    pub fn rows(&self) -> u32 {
        self.batch_size * self.seq_len
    }

    pub fn num_vocab_tiles(&self) -> u32 {
        self.vocab_size.div_ceil(self.vocab_tile)
    }

    /// SMEM layout: `[0 .. vtile*4)` student logit tile, `[vtile*4 ..
    /// 2*vtile*4)` teacher logit tile, one f32 scratch slot for the
    /// student's logit-at-target, padded to 32 bytes.
    pub fn shared_mem_bytes(&self) -> u32 {
        self.vocab_tile * 4 * 2 + 32
    }

    pub fn kernel_name(&self) -> String {
        format!(
            "nsl_fused_kl_ce_f32_v{}_hs{}_ht{}",
            self.vocab_size, self.student_hidden, self.teacher_hidden
        )
    }

    pub fn bwd_kernel_name(&self) -> String {
        format!(
            "nsl_fused_kl_ce_backward_f32_v{}_hs{}_ht{}",
            self.vocab_size, self.student_hidden, self.teacher_hidden
        )
    }

    fn sm_tag(&self) -> u32 {
        self.gpu_sm.max(80)
    }

    fn ptx_header(&self) -> String {
        format!(
            ".version 7.0\n.target sm_{}\n.address_size 64\n",
            self.sm_tag()
        )
    }
}

/// Synthesise the forward PTX. Returned bytes are null-terminated
/// (`cuModuleLoadData` reads a C string; see `fused_linear_ce.rs`).
pub fn synthesize_fused_kl_ce_ptx(cfg: &FusedKlCeConfig) -> Vec<u8> {
    let mut bytes = emit_fwd_kernel(cfg).into_bytes();
    bytes.push(0);
    bytes
}

/// Synthesise the backward PTX (null-terminated, same contract).
pub fn synthesize_fused_kl_ce_backward_ptx(cfg: &FusedKlCeConfig) -> Vec<u8> {
    let mut bytes = emit_bwd_kernel(cfg).into_bytes();
    bytes.push(0);
    bytes
}

// ─── Forward kernel ─────────────────────────────────────────────────────────
//
// Grid (rows, 1, 1), block (128, 1, 1). Per CTA:
//   1. targets[row] == ignore_index  ->  write four zeros, ret.
//   2. Tile loop: 128 threads stride-fill the STUDENT and TEACHER logit
//      tiles in SMEM (each thread computes both dot products for its vocab
//      slot); the thread whose v == target stashes the student logit in the
//      scratch slot.
//   3. Thread 0: one max-scan per model over the tile, then the online
//      rescale + accumulate pass updating (m_s1,S_s1), (m_sT,S_sT),
//      (m_tT,S_tT,C).
//   4. After the loop, thread 0 assembles CE + KL and stores loss + 3 LSEs.
fn emit_fwd_kernel(cfg: &FusedKlCeConfig) -> String {
    let name = cfg.kernel_name();
    let vocab = cfg.vocab_size;
    let hs = cfg.student_hidden;
    let ht = cfg.teacher_hidden;
    let vtile = cfg.vocab_tile;
    let n_tiles = cfg.num_vocab_tiles();
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();
    // SMEM offsets.
    let teacher_tile_offset = vtile * 4;
    let lat_offset = vtile * 4 * 2;

    let mut s = String::new();
    s.push_str(&cfg.ptx_header());
    s.push_str(&format!(
        ".extern .shared .align 4 .b8 smem_scratch[{smem_bytes}];\n\n"
    ));
    s.push_str(&format!(
        ".visible .entry {name}(\n\
         \t.param .u64 param_xs,\n\
         \t.param .u64 param_ws,\n\
         \t.param .u64 param_bs,\n\
         \t.param .u64 param_xt,\n\
         \t.param .u64 param_wt,\n\
         \t.param .u64 param_bt,\n\
         \t.param .u64 param_targets,\n\
         \t.param .u64 param_loss_out,\n\
         \t.param .u64 param_lse_s1_out,\n\
         \t.param .u64 param_lse_st_out,\n\
         \t.param .u64 param_lse_tt_out,\n\
         \t.param .u32 param_rows,\n\
         \t.param .u32 param_V,\n\
         \t.param .u32 param_HS,\n\
         \t.param .u32 param_HT,\n\
         \t.param .f32 param_alpha,\n\
         \t.param .f32 param_temp\n\
         ) {{\n"
    ));

    // Registers: numbered families sized generously (declared count must
    // exceed the highest index used — see bugs.md BIAS_ADD_F32_PTX lesson);
    // f32 working set uses NAMED registers to avoid the off-by-one class
    // entirely.
    s.push_str(
        "\t.reg .u64 %rd<32>;\n\
         \t.reg .u32 %r<20>;\n\
         \t.reg .s64 %tgt64;\n\
         \t.reg .f32 %facc_s, %facc_t, %fa, %fb, %ftmp, %ftmp2;\n\
         \t.reg .f32 %fmax_s1, %fsum_s1, %fmax_st, %fsum_st, %fmax_tt, %fsum_tt, %fcross;\n\
         \t.reg .f32 %ftmax_s, %ftmax_t, %fnewm, %fscale;\n\
         \t.reg .f32 %flog2e, %fln2, %falpha, %ftemp, %ftinv;\n\
         \t.reg .f32 %flse_s1, %flse_st, %flse_tt, %fce, %fkl, %floss, %fsv, %ftv;\n\
         \t.reg .pred %pskip, %pv, %pth0, %ptgt;\n\n",
    );

    // Params.
    s.push_str(
        "\tld.param.u64 %rd0, [param_xs];\n\
         \tld.param.u64 %rd1, [param_ws];\n\
         \tld.param.u64 %rd2, [param_bs];\n\
         \tld.param.u64 %rd3, [param_xt];\n\
         \tld.param.u64 %rd4, [param_wt];\n\
         \tld.param.u64 %rd5, [param_bt];\n\
         \tld.param.u64 %rd6, [param_targets];\n\
         \tld.param.u64 %rd7, [param_loss_out];\n\
         \tld.param.u64 %rd8, [param_lse_s1_out];\n\
         \tld.param.u64 %rd9, [param_lse_st_out];\n\
         \tld.param.u64 %rd10, [param_lse_tt_out];\n\
         \tld.param.f32 %falpha, [param_alpha];\n\
         \tld.param.f32 %ftemp, [param_temp];\n\
         \tmov.u32 %r0, %ctaid.x;   // row_idx\n\
         \tmov.u32 %r1, %tid.x;     // tid\n\
         \tmov.f32 %flog2e, 0f3FB8AA3B; // log2(e)\n\
         \tmov.f32 %fln2,   0f3F317218; // ln(2)\n\
         \trcp.approx.f32 %ftinv, %ftemp; // 1/T\n\n",
    );

    // Thread 0 inits the logit-at-target scratch slot.
    s.push_str(&format!(
        "\tsetp.eq.u32 %pth0, %r1, 0;\n\
         \t@!%pth0 bra INIT_DONE;\n\
         \tmov.u64 %rd11, smem_scratch;\n\
         \tadd.u64 %rd11, %rd11, {lat_offset};\n\
         \tst.shared.f32 [%rd11], 0fFF800000; // -INF sentinel\n\
         INIT_DONE:\n\
         \tbar.sync 0;\n\n"
    ));

    // Load target[row].
    s.push_str(
        "\tcvt.u64.u32 %rd12, %r0;\n\
         \tmul.lo.u64 %rd12, %rd12, 8;\n\
         \tadd.u64 %rd12, %rd6, %rd12;\n\
         \tld.global.s64 %tgt64, [%rd12];\n\n",
    );
    s.push_str(&format!(
        "\tsetp.eq.s64 %pskip, %tgt64, {ignore};\n\
         \t@%pskip bra SKIP_LABEL;\n\n"
    ));

    // Row bases: xs + row*HS*4, xt + row*HT*4.
    s.push_str(&format!(
        "\t// xs_row_base / xt_row_base\n\
         \tcvt.u64.u32 %rd13, %r0;\n\
         \tmov.u32 %r2, {hs};\n\
         \tcvt.u64.u32 %rd14, %r2;\n\
         \tmul.lo.u64 %rd13, %rd13, %rd14;\n\
         \tshl.b64 %rd13, %rd13, 2;\n\
         \tadd.u64 %rd13, %rd0, %rd13; // xs_row_base\n\
         \tcvt.u64.u32 %rd15, %r0;\n\
         \tmov.u32 %r2, {ht};\n\
         \tcvt.u64.u32 %rd16, %r2;\n\
         \tmul.lo.u64 %rd15, %rd15, %rd16;\n\
         \tshl.b64 %rd15, %rd15, 2;\n\
         \tadd.u64 %rd15, %rd3, %rd15; // xt_row_base\n\n"
    ));

    // Init accumulators.
    s.push_str(
        "\tmov.f32 %fmax_s1, 0fFF800000;\n\
         \tmov.f32 %fsum_s1, 0f00000000;\n\
         \tmov.f32 %fmax_st, 0fFF800000;\n\
         \tmov.f32 %fsum_st, 0f00000000;\n\
         \tmov.f32 %fmax_tt, 0fFF800000;\n\
         \tmov.f32 %fsum_tt, 0f00000000;\n\
         \tmov.f32 %fcross, 0f00000000;\n\n",
    );

    // Tile loop.
    s.push_str("\tmov.u32 %r3, 0; // tile_idx\nTILE_LOOP:\n");
    s.push_str(&format!("\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base\n\n"));

    // Inner fill loop: each thread computes student AND teacher logits for
    // its vocab slots.
    s.push_str(
        "\t\tmov.u32 %r5, 0; // sub-tile counter\n\
         \t\tINNER_LOOP:\n\
         \t\t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\t\tadd.u32 %r6, %r6, %r1;\n\
         \t\t\tadd.u32 %r6, %r6, %r4; // v_idx\n",
    );
    s.push_str(&format!(
        "\t\t\tsetp.lt.u32 %pv, %r6, {vocab};\n\
         \t\t\t@!%pv bra INNER_SKIP;\n\n"
    ));

    // Student dot: ws_row_base = ws + v*HS*4.
    s.push_str(&format!(
        "\t\t\t// student logit\n\
         \t\t\tcvt.u64.u32 %rd17, %r6;\n\
         \t\t\tmov.u32 %r7, {hs};\n\
         \t\t\tcvt.u64.u32 %rd18, %r7;\n\
         \t\t\tmul.lo.u64 %rd17, %rd17, %rd18;\n\
         \t\t\tshl.b64 %rd17, %rd17, 2;\n\
         \t\t\tadd.u64 %rd17, %rd1, %rd17; // ws_row_base\n\
         \t\t\tmov.f32 %facc_s, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         \t\t\tDOT_S_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd19, %r8;\n\
         \t\t\t\tshl.b64 %rd19, %rd19, 2;\n\
         \t\t\t\tadd.u64 %rd20, %rd13, %rd19;\n\
         \t\t\t\tld.global.f32 %fa, [%rd20];\n\
         \t\t\t\tadd.u64 %rd20, %rd17, %rd19;\n\
         \t\t\t\tld.global.f32 %fb, [%rd20];\n\
         \t\t\t\tfma.rn.f32 %facc_s, %fa, %fb, %facc_s;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n\
         \t\t\t\tsetp.lt.u32 %pv, %r8, {hs};\n\
         \t\t\t\t@%pv bra DOT_S_LOOP;\n\
         \t\t\t// + bias_s[v]\n\
         \t\t\tcvt.u64.u32 %rd21, %r6;\n\
         \t\t\tshl.b64 %rd21, %rd21, 2;\n\
         \t\t\tadd.u64 %rd21, %rd2, %rd21;\n\
         \t\t\tld.global.f32 %ftmp, [%rd21];\n\
         \t\t\tadd.f32 %facc_s, %facc_s, %ftmp;\n\n"
    ));

    // Teacher dot: wt_row_base = wt + v*HT*4.
    s.push_str(&format!(
        "\t\t\t// teacher logit\n\
         \t\t\tcvt.u64.u32 %rd22, %r6;\n\
         \t\t\tmov.u32 %r7, {ht};\n\
         \t\t\tcvt.u64.u32 %rd23, %r7;\n\
         \t\t\tmul.lo.u64 %rd22, %rd22, %rd23;\n\
         \t\t\tshl.b64 %rd22, %rd22, 2;\n\
         \t\t\tadd.u64 %rd22, %rd4, %rd22; // wt_row_base\n\
         \t\t\tmov.f32 %facc_t, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         \t\t\tDOT_T_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd19, %r8;\n\
         \t\t\t\tshl.b64 %rd19, %rd19, 2;\n\
         \t\t\t\tadd.u64 %rd20, %rd15, %rd19;\n\
         \t\t\t\tld.global.f32 %fa, [%rd20];\n\
         \t\t\t\tadd.u64 %rd20, %rd22, %rd19;\n\
         \t\t\t\tld.global.f32 %fb, [%rd20];\n\
         \t\t\t\tfma.rn.f32 %facc_t, %fa, %fb, %facc_t;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n\
         \t\t\t\tsetp.lt.u32 %pv, %r8, {ht};\n\
         \t\t\t\t@%pv bra DOT_T_LOOP;\n\
         \t\t\t// + bias_t[v]\n\
         \t\t\tcvt.u64.u32 %rd21, %r6;\n\
         \t\t\tshl.b64 %rd21, %rd21, 2;\n\
         \t\t\tadd.u64 %rd21, %rd5, %rd21;\n\
         \t\t\tld.global.f32 %ftmp, [%rd21];\n\
         \t\t\tadd.f32 %facc_t, %facc_t, %ftmp;\n\n"
    ));

    // Store both logits into SMEM (student tile at 0, teacher tile at
    // teacher_tile_offset).
    s.push_str(&format!(
        "\t\t\t// smem slot = (r5*128 + tid) * 4\n\
         \t\t\tmul.lo.u32 %r9, %r5, 128;\n\
         \t\t\tadd.u32 %r9, %r9, %r1;\n\
         \t\t\tshl.b32 %r9, %r9, 2;\n\
         \t\t\tmov.u64 %rd24, smem_scratch;\n\
         \t\t\tcvt.u64.u32 %rd25, %r9;\n\
         \t\t\tadd.u64 %rd24, %rd24, %rd25;\n\
         \t\t\tst.shared.f32 [%rd24], %facc_s;\n\
         \t\t\tadd.u64 %rd24, %rd24, {teacher_tile_offset};\n\
         \t\t\tst.shared.f32 [%rd24], %facc_t;\n\n"
    ));

    // Target capture (student logit at temperature 1).
    s.push_str(&format!(
        "\t\t\tcvt.s64.u32 %rd26, %r6;\n\
         \t\t\tsetp.eq.s64 %ptgt, %rd26, %tgt64;\n\
         \t\t\t@!%ptgt bra NOT_TARGET;\n\
         \t\t\tmov.u64 %rd27, smem_scratch;\n\
         \t\t\tadd.u64 %rd27, %rd27, {lat_offset};\n\
         \t\t\tst.shared.f32 [%rd27], %facc_s;\n\
         \t\t\tNOT_TARGET:\n\n"
    ));

    s.push_str(&format!(
        "\t\t\tINNER_SKIP:\n\
         \t\t\tadd.u32 %r5, %r5, 1;\n\
         \t\t\tsetp.lt.u32 %pv, %r5, {vtile_per_thread};\n\
         \t\t\t@%pv bra INNER_LOOP;\n\n\
         \t\tbar.sync 0;\n\n"
    ));

    // Thread 0 reduction.
    s.push_str(
        "\t\tsetp.eq.u32 %pth0, %r1, 0;\n\
         \t\t@!%pth0 bra TILE_REDUCE_DONE;\n\n",
    );

    // Max scan over both tiles (raw logits).
    s.push_str(&format!(
        "\t\t// tile max scan (student + teacher, raw logits)\n\
         \t\tmov.f32 %ftmax_s, 0fFF800000;\n\
         \t\tmov.f32 %ftmax_t, 0fFF800000;\n\
         \t\tmov.u32 %r10, 0;\n\
         \t\tSMEM_MAX_LOOP:\n\
         \t\t\tadd.u32 %r11, %r4, %r10;\n\
         \t\t\tsetp.lt.u32 %pv, %r11, {vocab};\n\
         \t\t\t@!%pv bra SMEM_MAX_DONE;\n\
         \t\t\tshl.b32 %r12, %r10, 2;\n\
         \t\t\tmov.u64 %rd28, smem_scratch;\n\
         \t\t\tcvt.u64.u32 %rd29, %r12;\n\
         \t\t\tadd.u64 %rd28, %rd28, %rd29;\n\
         \t\t\tld.shared.f32 %ftmp, [%rd28];\n\
         \t\t\tmax.f32 %ftmax_s, %ftmax_s, %ftmp;\n\
         \t\t\tadd.u64 %rd28, %rd28, {teacher_tile_offset};\n\
         \t\t\tld.shared.f32 %ftmp, [%rd28];\n\
         \t\t\tmax.f32 %ftmax_t, %ftmax_t, %ftmp;\n\
         \t\t\tadd.u32 %r10, %r10, 1;\n\
         \t\t\tsetp.lt.u32 %pv, %r10, {vtile};\n\
         \t\t\t@%pv bra SMEM_MAX_LOOP;\n\
         \t\tSMEM_MAX_DONE:\n\n"
    ));

    // Online rescale of the three accumulator families.
    //
    // (m_s1, S_s1): candidate = ftmax_s.
    // (m_sT, S_sT): candidate = ftmax_s / T.
    // (m_tT, S_tT, C): candidate = ftmax_t / T; C rescales with S_tT.
    s.push_str(
        "\t\t// online rescale: student T=1\n\
         \t\tmax.f32 %fnewm, %fmax_s1, %ftmax_s;\n\
         \t\tsub.f32 %fscale, %fmax_s1, %fnewm;\n\
         \t\tmul.f32 %fscale, %fscale, %flog2e;\n\
         \t\tex2.approx.f32 %fscale, %fscale;\n\
         \t\tmul.f32 %fsum_s1, %fsum_s1, %fscale;\n\
         \t\tmov.f32 %fmax_s1, %fnewm;\n\
         \t\t// online rescale: student /T\n\
         \t\tmul.f32 %ftmp, %ftmax_s, %ftinv;\n\
         \t\tmax.f32 %fnewm, %fmax_st, %ftmp;\n\
         \t\tsub.f32 %fscale, %fmax_st, %fnewm;\n\
         \t\tmul.f32 %fscale, %fscale, %flog2e;\n\
         \t\tex2.approx.f32 %fscale, %fscale;\n\
         \t\tmul.f32 %fsum_st, %fsum_st, %fscale;\n\
         \t\tmov.f32 %fmax_st, %fnewm;\n\
         \t\t// online rescale: teacher /T (sum AND cross-term)\n\
         \t\tmul.f32 %ftmp, %ftmax_t, %ftinv;\n\
         \t\tmax.f32 %fnewm, %fmax_tt, %ftmp;\n\
         \t\tsub.f32 %fscale, %fmax_tt, %fnewm;\n\
         \t\tmul.f32 %fscale, %fscale, %flog2e;\n\
         \t\tex2.approx.f32 %fscale, %fscale;\n\
         \t\tmul.f32 %fsum_tt, %fsum_tt, %fscale;\n\
         \t\tmul.f32 %fcross, %fcross, %fscale;\n\
         \t\tmov.f32 %fmax_tt, %fnewm;\n\n",
    );

    // Accumulate pass: per tile slot load s_v and t_v once, update all
    // three families.
    s.push_str(&format!(
        "\t\tmov.u32 %r10, 0;\n\
         \t\tSMEM_ACC_LOOP:\n\
         \t\t\tadd.u32 %r11, %r4, %r10;\n\
         \t\t\tsetp.lt.u32 %pv, %r11, {vocab};\n\
         \t\t\t@!%pv bra SMEM_ACC_DONE;\n\
         \t\t\tshl.b32 %r12, %r10, 2;\n\
         \t\t\tmov.u64 %rd28, smem_scratch;\n\
         \t\t\tcvt.u64.u32 %rd29, %r12;\n\
         \t\t\tadd.u64 %rd28, %rd28, %rd29;\n\
         \t\t\tld.shared.f32 %fsv, [%rd28];\n\
         \t\t\tadd.u64 %rd28, %rd28, {teacher_tile_offset};\n\
         \t\t\tld.shared.f32 %ftv, [%rd28];\n\
         \t\t\t// S_s1 += exp(s - m_s1)\n\
         \t\t\tsub.f32 %ftmp, %fsv, %fmax_s1;\n\
         \t\t\tmul.f32 %ftmp, %ftmp, %flog2e;\n\
         \t\t\tex2.approx.f32 %ftmp, %ftmp;\n\
         \t\t\tadd.f32 %fsum_s1, %fsum_s1, %ftmp;\n\
         \t\t\t// S_sT += exp(s/T - m_sT)\n\
         \t\t\tmul.f32 %ftmp, %fsv, %ftinv;\n\
         \t\t\tsub.f32 %ftmp, %ftmp, %fmax_st;\n\
         \t\t\tmul.f32 %ftmp, %ftmp, %flog2e;\n\
         \t\t\tex2.approx.f32 %ftmp, %ftmp;\n\
         \t\t\tadd.f32 %fsum_st, %fsum_st, %ftmp;\n\
         \t\t\t// e_t = exp(t/T - m_tT); S_tT += e_t; C += e_t * (t - s)\n\
         \t\t\tmul.f32 %ftmp, %ftv, %ftinv;\n\
         \t\t\tsub.f32 %ftmp, %ftmp, %fmax_tt;\n\
         \t\t\tmul.f32 %ftmp, %ftmp, %flog2e;\n\
         \t\t\tex2.approx.f32 %ftmp, %ftmp;\n\
         \t\t\tadd.f32 %fsum_tt, %fsum_tt, %ftmp;\n\
         \t\t\tsub.f32 %ftmp2, %ftv, %fsv;\n\
         \t\t\tfma.rn.f32 %fcross, %ftmp, %ftmp2, %fcross;\n\
         \t\t\tadd.u32 %r10, %r10, 1;\n\
         \t\t\tsetp.lt.u32 %pv, %r10, {vtile};\n\
         \t\t\t@%pv bra SMEM_ACC_LOOP;\n\
         \t\tSMEM_ACC_DONE:\n\n\
         \t\tTILE_REDUCE_DONE:\n\
         \t\tbar.sync 0;\n\n\
         \t\tadd.u32 %r3, %r3, 1;\n\
         \t\tsetp.lt.u32 %pv, %r3, {n_tiles};\n\
         \t\t@%pv bra TILE_LOOP;\n\n"
    ));

    // Final assembly (thread 0).
    s.push_str(&format!(
        "\tsetp.eq.u32 %pth0, %r1, 0;\n\
         \t@!%pth0 bra WRITE_DONE;\n\n\
         \t// lse_s1 = ln(S_s1) + m_s1  (and the /T pair analogously)\n\
         \tlg2.approx.f32 %flse_s1, %fsum_s1;\n\
         \tmul.f32 %flse_s1, %flse_s1, %fln2;\n\
         \tadd.f32 %flse_s1, %flse_s1, %fmax_s1;\n\
         \tlg2.approx.f32 %flse_st, %fsum_st;\n\
         \tmul.f32 %flse_st, %flse_st, %fln2;\n\
         \tadd.f32 %flse_st, %flse_st, %fmax_st;\n\
         \tlg2.approx.f32 %flse_tt, %fsum_tt;\n\
         \tmul.f32 %flse_tt, %flse_tt, %fln2;\n\
         \tadd.f32 %flse_tt, %flse_tt, %fmax_tt;\n\n\
         \t// CE = lse_s1 - s_target\n\
         \tmov.u64 %rd30, smem_scratch;\n\
         \tadd.u64 %rd30, %rd30, {lat_offset};\n\
         \tld.shared.f32 %ftmp, [%rd30];\n\
         \tsub.f32 %fce, %flse_s1, %ftmp;\n\n\
         \t// KL = (C / S_tT) / T - lse_tT + lse_sT\n\
         \tdiv.rn.f32 %fkl, %fcross, %fsum_tt;\n\
         \tmul.f32 %fkl, %fkl, %ftinv;\n\
         \tsub.f32 %fkl, %fkl, %flse_tt;\n\
         \tadd.f32 %fkl, %fkl, %flse_st;\n\n\
         \t// loss = alpha*CE + (1-alpha)*T^2*KL\n\
         \tmul.f32 %floss, %falpha, %fce;\n\
         \tmov.f32 %ftmp, 0f3F800000; // 1.0\n\
         \tsub.f32 %ftmp, %ftmp, %falpha;\n\
         \tmul.f32 %ftmp2, %ftemp, %ftemp;\n\
         \tmul.f32 %ftmp, %ftmp, %ftmp2;\n\
         \tfma.rn.f32 %floss, %ftmp, %fkl, %floss;\n\n\
         \t// stores\n\
         \tcvt.u64.u32 %rd31, %r0;\n\
         \tshl.b64 %rd31, %rd31, 2;\n\
         \tadd.u64 %rd30, %rd7, %rd31;\n\
         \tst.global.f32 [%rd30], %floss;\n\
         \tadd.u64 %rd30, %rd8, %rd31;\n\
         \tst.global.f32 [%rd30], %flse_s1;\n\
         \tadd.u64 %rd30, %rd9, %rd31;\n\
         \tst.global.f32 [%rd30], %flse_st;\n\
         \tadd.u64 %rd30, %rd10, %rd31;\n\
         \tst.global.f32 [%rd30], %flse_tt;\n\
         \tbra WRITE_DONE;\n\n\
         SKIP_LABEL:\n\
         \tsetp.eq.u32 %pth0, %r1, 0;\n\
         \t@!%pth0 bra WRITE_DONE;\n\
         \tcvt.u64.u32 %rd31, %r0;\n\
         \tshl.b64 %rd31, %rd31, 2;\n\
         \tadd.u64 %rd30, %rd7, %rd31;\n\
         \tst.global.f32 [%rd30], 0f00000000;\n\
         \tadd.u64 %rd30, %rd8, %rd31;\n\
         \tst.global.f32 [%rd30], 0f00000000;\n\
         \tadd.u64 %rd30, %rd9, %rd31;\n\
         \tst.global.f32 [%rd30], 0f00000000;\n\
         \tadd.u64 %rd30, %rd10, %rd31;\n\
         \tst.global.f32 [%rd30], 0f00000000;\n\n\
         WRITE_DONE:\n\
         \tret;\n\
         }}\n"
    ));

    s
}

// ─── Backward kernel ────────────────────────────────────────────────────────
//
// Grid (rows, 1, 1), block (128, 1, 1). No SMEM. Per (row, v) recomputes
// both logits, forms
//   dlogit_v = alpha*(p_s1 - 1{v==tgt}) + (1-alpha)*T*(p_sT - p_tT)
// scaled by grad/num_valid, then scatters dx_s / dW_s / dbias_s via
// red.global.add.f32 (dx_s/dW_s/dbias_s must be caller-zero-filled).
// NO teacher gradient outputs exist in the ABI (I-11).
fn emit_bwd_kernel(cfg: &FusedKlCeConfig) -> String {
    let name = cfg.bwd_kernel_name();
    let vocab = cfg.vocab_size;
    let hs = cfg.student_hidden;
    let ht = cfg.teacher_hidden;
    let vtile = cfg.vocab_tile;
    let n_tiles = cfg.num_vocab_tiles();
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;

    let mut s = String::new();
    s.push_str(&cfg.ptx_header());
    s.push('\n');

    s.push_str(&format!(
        ".visible .entry {name}(\n\
         \t.param .f32 param_grad_output,\n\
         \t.param .u64 param_xs,\n\
         \t.param .u64 param_ws,\n\
         \t.param .u64 param_bs,\n\
         \t.param .u64 param_xt,\n\
         \t.param .u64 param_wt,\n\
         \t.param .u64 param_bt,\n\
         \t.param .u64 param_targets,\n\
         \t.param .u64 param_lse_s1,\n\
         \t.param .u64 param_lse_st,\n\
         \t.param .u64 param_lse_tt,\n\
         \t.param .u64 param_dxs_out,\n\
         \t.param .u64 param_dws_out,\n\
         \t.param .u64 param_dbs_out,\n\
         \t.param .u32 param_rows,\n\
         \t.param .u32 param_V,\n\
         \t.param .u32 param_HS,\n\
         \t.param .u32 param_HT,\n\
         \t.param .f32 param_alpha,\n\
         \t.param .f32 param_temp,\n\
         \t.param .u32 param_num_valid\n\
         ) {{\n"
    ));

    s.push_str(
        "\t.reg .u64 %rd<34>;\n\
         \t.reg .u32 %r<20>;\n\
         \t.reg .s64 %tgt64;\n\
         \t.reg .f32 %facc_s, %facc_t, %fa, %fb, %ftmp;\n\
         \t.reg .f32 %fgrad, %fscale, %fnvf, %falpha, %ftemp, %ftinv, %flog2e;\n\
         \t.reg .f32 %flse_s1, %flse_st, %flse_tt;\n\
         \t.reg .f32 %fps1, %fpst, %fptt, %fdl, %fsc;\n\
         \t.reg .pred %pskip, %pv, %ptgt;\n\
         \t.reg .u32 %nvalid;\n\n",
    );

    s.push_str(
        "\tld.param.f32 %fgrad, [param_grad_output];\n\
         \tld.param.u64 %rd0, [param_xs];\n\
         \tld.param.u64 %rd1, [param_ws];\n\
         \tld.param.u64 %rd2, [param_bs];\n\
         \tld.param.u64 %rd3, [param_xt];\n\
         \tld.param.u64 %rd4, [param_wt];\n\
         \tld.param.u64 %rd5, [param_bt];\n\
         \tld.param.u64 %rd6, [param_targets];\n\
         \tld.param.u64 %rd7, [param_lse_s1];\n\
         \tld.param.u64 %rd8, [param_lse_st];\n\
         \tld.param.u64 %rd9, [param_lse_tt];\n\
         \tld.param.u64 %rd10, [param_dxs_out];\n\
         \tld.param.u64 %rd11, [param_dws_out];\n\
         \tld.param.u64 %rd12, [param_dbs_out];\n\
         \tld.param.f32 %falpha, [param_alpha];\n\
         \tld.param.f32 %ftemp, [param_temp];\n\
         \tld.param.u32 %nvalid, [param_num_valid];\n\
         \tcvt.rn.f32.u32 %fnvf, %nvalid;\n\
         \tmov.u32 %r0, %ctaid.x;\n\
         \tmov.u32 %r1, %tid.x;\n\
         \tmov.f32 %flog2e, 0f3FB8AA3B;\n\
         \trcp.approx.f32 %ftinv, %ftemp;\n\n",
    );

    // Target + skip.
    s.push_str(&format!(
        "\tcvt.u64.u32 %rd13, %r0;\n\
         \tmul.lo.u64 %rd13, %rd13, 8;\n\
         \tadd.u64 %rd13, %rd6, %rd13;\n\
         \tld.global.s64 %tgt64, [%rd13];\n\
         \tsetp.eq.s64 %pskip, %tgt64, {ignore};\n\
         \t@%pskip bra BWD_SKIP_LABEL;\n\n"
    ));

    // Saved LSEs.
    s.push_str(
        "\tcvt.u64.u32 %rd14, %r0;\n\
         \tshl.b64 %rd14, %rd14, 2;\n\
         \tadd.u64 %rd15, %rd7, %rd14;\n\
         \tld.global.f32 %flse_s1, [%rd15];\n\
         \tadd.u64 %rd15, %rd8, %rd14;\n\
         \tld.global.f32 %flse_st, [%rd15];\n\
         \tadd.u64 %rd15, %rd9, %rd14;\n\
         \tld.global.f32 %flse_tt, [%rd15];\n\n",
    );

    // Row bases.
    s.push_str(&format!(
        "\tcvt.u64.u32 %rd16, %r0;\n\
         \tmov.u32 %r2, {hs};\n\
         \tcvt.u64.u32 %rd17, %r2;\n\
         \tmul.lo.u64 %rd16, %rd16, %rd17;\n\
         \tshl.b64 %rd16, %rd16, 2;\n\
         \tadd.u64 %rd18, %rd0, %rd16; // xs_row_base\n\
         \tadd.u64 %rd19, %rd10, %rd16; // dxs_row_base\n\
         \tcvt.u64.u32 %rd20, %r0;\n\
         \tmov.u32 %r2, {ht};\n\
         \tcvt.u64.u32 %rd21, %r2;\n\
         \tmul.lo.u64 %rd20, %rd20, %rd21;\n\
         \tshl.b64 %rd20, %rd20, 2;\n\
         \tadd.u64 %rd20, %rd3, %rd20; // xt_row_base\n\n"
    ));

    // scale = grad / num_valid.
    s.push_str("\tdiv.rn.f32 %fscale, %fgrad, %fnvf;\n\n");

    // Tile loop.
    s.push_str(&format!(
        "\tmov.u32 %r3, 0;\n\
         BWD_TILE_LOOP:\n\
         \t\tmul.lo.u32 %r4, %r3, {vtile};\n\
         \t\tmov.u32 %r5, 0;\n\
         BWD_INNER_LOOP:\n\
         \t\t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\t\tadd.u32 %r6, %r6, %r1;\n\
         \t\t\tadd.u32 %r6, %r6, %r4; // v_idx\n\
         \t\t\tsetp.lt.u32 %pv, %r6, {vocab};\n\
         \t\t\t@!%pv bra BWD_INNER_SKIP;\n\n"
    ));

    // Recompute student logit.
    s.push_str(&format!(
        "\t\t\t// student logit recompute\n\
         \t\t\tcvt.u64.u32 %rd22, %r6;\n\
         \t\t\tmov.u32 %r7, {hs};\n\
         \t\t\tcvt.u64.u32 %rd23, %r7;\n\
         \t\t\tmul.lo.u64 %rd22, %rd22, %rd23;\n\
         \t\t\tshl.b64 %rd22, %rd22, 2;\n\
         \t\t\tadd.u64 %rd22, %rd1, %rd22; // ws_row_base\n\
         \t\t\tmov.f32 %facc_s, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         BWD_DOT_S_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd24, %r8;\n\
         \t\t\t\tshl.b64 %rd24, %rd24, 2;\n\
         \t\t\t\tadd.u64 %rd25, %rd18, %rd24;\n\
         \t\t\t\tld.global.f32 %fa, [%rd25];\n\
         \t\t\t\tadd.u64 %rd25, %rd22, %rd24;\n\
         \t\t\t\tld.global.f32 %fb, [%rd25];\n\
         \t\t\t\tfma.rn.f32 %facc_s, %fa, %fb, %facc_s;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n\
         \t\t\t\tsetp.lt.u32 %pv, %r8, {hs};\n\
         \t\t\t\t@%pv bra BWD_DOT_S_LOOP;\n\
         \t\t\tcvt.u64.u32 %rd26, %r6;\n\
         \t\t\tshl.b64 %rd26, %rd26, 2;\n\
         \t\t\tadd.u64 %rd27, %rd2, %rd26;\n\
         \t\t\tld.global.f32 %ftmp, [%rd27];\n\
         \t\t\tadd.f32 %facc_s, %facc_s, %ftmp;\n\n"
    ));

    // Recompute teacher logit.
    s.push_str(&format!(
        "\t\t\t// teacher logit recompute\n\
         \t\t\tcvt.u64.u32 %rd28, %r6;\n\
         \t\t\tmov.u32 %r7, {ht};\n\
         \t\t\tcvt.u64.u32 %rd29, %r7;\n\
         \t\t\tmul.lo.u64 %rd28, %rd28, %rd29;\n\
         \t\t\tshl.b64 %rd28, %rd28, 2;\n\
         \t\t\tadd.u64 %rd28, %rd4, %rd28; // wt_row_base\n\
         \t\t\tmov.f32 %facc_t, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         BWD_DOT_T_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd24, %r8;\n\
         \t\t\t\tshl.b64 %rd24, %rd24, 2;\n\
         \t\t\t\tadd.u64 %rd25, %rd20, %rd24;\n\
         \t\t\t\tld.global.f32 %fa, [%rd25];\n\
         \t\t\t\tadd.u64 %rd25, %rd28, %rd24;\n\
         \t\t\t\tld.global.f32 %fb, [%rd25];\n\
         \t\t\t\tfma.rn.f32 %facc_t, %fa, %fb, %facc_t;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n\
         \t\t\t\tsetp.lt.u32 %pv, %r8, {ht};\n\
         \t\t\t\t@%pv bra BWD_DOT_T_LOOP;\n\
         \t\t\tadd.u64 %rd27, %rd5, %rd26;\n\
         \t\t\tld.global.f32 %ftmp, [%rd27];\n\
         \t\t\tadd.f32 %facc_t, %facc_t, %ftmp;\n\n"
    ));

    // Softmax probabilities from saved LSEs.
    s.push_str(
        "\t\t\t// p_s1 = exp(s - lse_s1)\n\
         \t\t\tsub.f32 %ftmp, %facc_s, %flse_s1;\n\
         \t\t\tmul.f32 %ftmp, %ftmp, %flog2e;\n\
         \t\t\tex2.approx.f32 %fps1, %ftmp;\n\
         \t\t\t// p_sT = exp(s/T - lse_sT)\n\
         \t\t\tmul.f32 %ftmp, %facc_s, %ftinv;\n\
         \t\t\tsub.f32 %ftmp, %ftmp, %flse_st;\n\
         \t\t\tmul.f32 %ftmp, %ftmp, %flog2e;\n\
         \t\t\tex2.approx.f32 %fpst, %ftmp;\n\
         \t\t\t// p_tT = exp(t/T - lse_tT)\n\
         \t\t\tmul.f32 %ftmp, %facc_t, %ftinv;\n\
         \t\t\tsub.f32 %ftmp, %ftmp, %flse_tt;\n\
         \t\t\tmul.f32 %ftmp, %ftmp, %flog2e;\n\
         \t\t\tex2.approx.f32 %fptt, %ftmp;\n\n",
    );

    // dlogit assembly.
    s.push_str(
        "\t\t\t// dl = alpha*(p_s1 - is_target) + (1-alpha)*T*(p_sT - p_tT)\n\
         \t\t\tcvt.s64.u32 %rd30, %r6;\n\
         \t\t\tsetp.eq.s64 %ptgt, %rd30, %tgt64;\n\
         \t\t\t@%ptgt sub.f32 %fps1, %fps1, 0f3F800000;\n\
         \t\t\tmul.f32 %fdl, %falpha, %fps1;\n\
         \t\t\tsub.f32 %ftmp, %fpst, %fptt;\n\
         \t\t\tmov.f32 %fa, 0f3F800000;\n\
         \t\t\tsub.f32 %fa, %fa, %falpha;\n\
         \t\t\tmul.f32 %fa, %fa, %ftemp;\n\
         \t\t\tfma.rn.f32 %fdl, %fa, %ftmp, %fdl;\n\
         \t\t\tmul.f32 %fsc, %fdl, %fscale;\n\n",
    );

    // Scatter loops (student only).
    s.push_str(&format!(
        "\t\t\t// dW_s row base\n\
         \t\t\tcvt.u64.u32 %rd31, %r6;\n\
         \t\t\tmov.u32 %r9, {hs};\n\
         \t\t\tcvt.u64.u32 %rd32, %r9;\n\
         \t\t\tmul.lo.u64 %rd31, %rd31, %rd32;\n\
         \t\t\tshl.b64 %rd31, %rd31, 2;\n\
         \t\t\tadd.u64 %rd31, %rd11, %rd31;\n\
         \t\t\tmov.u32 %r9, 0;\n\
         BWD_H_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd33, %r9;\n\
         \t\t\t\tshl.b64 %rd33, %rd33, 2;\n\
         \t\t\t\t// dx_s[row,h] += sc * W_s[v,h]\n\
         \t\t\t\tadd.u64 %rd25, %rd22, %rd33;\n\
         \t\t\t\tld.global.f32 %fa, [%rd25];\n\
         \t\t\t\tmul.f32 %fb, %fsc, %fa;\n\
         \t\t\t\tadd.u64 %rd25, %rd19, %rd33;\n\
         \t\t\t\tred.global.add.f32 [%rd25], %fb;\n\
         \t\t\t\t// dW_s[v,h] += sc * x_s[row,h]\n\
         \t\t\t\tadd.u64 %rd25, %rd18, %rd33;\n\
         \t\t\t\tld.global.f32 %fa, [%rd25];\n\
         \t\t\t\tmul.f32 %fb, %fsc, %fa;\n\
         \t\t\t\tadd.u64 %rd25, %rd31, %rd33;\n\
         \t\t\t\tred.global.add.f32 [%rd25], %fb;\n\
         \t\t\t\tadd.u32 %r9, %r9, 1;\n\
         \t\t\t\tsetp.lt.u32 %pv, %r9, {hs};\n\
         \t\t\t\t@%pv bra BWD_H_LOOP;\n\
         \t\t\t// dbias_s[v] += sc\n\
         \t\t\tadd.u64 %rd25, %rd12, %rd26;\n\
         \t\t\tred.global.add.f32 [%rd25], %fsc;\n\n\
         BWD_INNER_SKIP:\n\
         \t\t\tadd.u32 %r5, %r5, 1;\n\
         \t\t\tsetp.lt.u32 %pv, %r5, {vtile_per_thread};\n\
         \t\t\t@%pv bra BWD_INNER_LOOP;\n\
         \t\tadd.u32 %r3, %r3, 1;\n\
         \t\tsetp.lt.u32 %pv, %r3, {n_tiles};\n\
         \t\t@%pv bra BWD_TILE_LOOP;\n\
         \tbra BWD_DONE;\n\n"
    ));

    // Skip path: zero dx_s row (dW/dbias untouched — caller zero-fills).
    s.push_str(&format!(
        "BWD_SKIP_LABEL:\n\
         \tcvt.u64.u32 %rd16, %r0;\n\
         \tmov.u32 %r2, {hs};\n\
         \tcvt.u64.u32 %rd17, %r2;\n\
         \tmul.lo.u64 %rd16, %rd16, %rd17;\n\
         \tshl.b64 %rd16, %rd16, 2;\n\
         \tadd.u64 %rd16, %rd10, %rd16; // dxs_row_base\n\
         \tmov.u32 %r5, 0;\n\
         BWD_ZERO_LOOP:\n\
         \t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\tadd.u32 %r6, %r6, %r1;\n\
         \t\tsetp.lt.u32 %pv, %r6, {hs};\n\
         \t\t@!%pv bra BWD_ZERO_DONE;\n\
         \t\tshl.b32 %r6, %r6, 2;\n\
         \t\tcvt.u64.u32 %rd17, %r6;\n\
         \t\tadd.u64 %rd17, %rd16, %rd17;\n\
         \t\tst.global.f32 [%rd17], 0f00000000;\n\
         \t\tadd.u32 %r5, %r5, 1;\n\
         \t\tbra BWD_ZERO_LOOP;\n\
         BWD_ZERO_DONE:\n\n\
         BWD_DONE:\n\
         \tret;\n\
         }}\n"
    ));

    s
}

// ─── Reference implementation (test oracle) ─────────────────────────────────

/// Pure-Rust f64 reference for the fused KL-CE forward. Returns
/// `(per_row_loss, lse_s1, lse_sT, lse_tT, num_valid)`.
///
/// Used by unit tests and the GPU parity suite; NOT a production fallback
/// (the production CPU path is the stdlib composite via tape AD).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn reference_forward_f64(
    xs: &[f64],
    ws: &[f64],
    bs: &[f64],
    xt: &[f64],
    wt: &[f64],
    bt: &[f64],
    targets: &[i64],
    rows: usize,
    vocab: usize,
    hs: usize,
    ht: usize,
    alpha: f64,
    temp: f64,
    ignore_index: i64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let mut loss = vec![0.0; rows];
    let mut lse_s1 = vec![0.0; rows];
    let mut lse_st = vec![0.0; rows];
    let mut lse_tt = vec![0.0; rows];
    let mut num_valid = 0usize;

    for r in 0..rows {
        let tgt = targets[r];
        if tgt == ignore_index {
            continue;
        }
        num_valid += 1;
        let s_logits: Vec<f64> = (0..vocab)
            .map(|v| {
                let mut acc = bs[v];
                for h in 0..hs {
                    acc += xs[r * hs + h] * ws[v * hs + h];
                }
                acc
            })
            .collect();
        let t_logits: Vec<f64> = (0..vocab)
            .map(|v| {
                let mut acc = bt[v];
                for h in 0..ht {
                    acc += xt[r * ht + h] * wt[v * ht + h];
                }
                acc
            })
            .collect();

        let lse = |xs: &[f64]| {
            let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
        };
        let l_s1 = lse(&s_logits);
        let s_t: Vec<f64> = s_logits.iter().map(|&x| x / temp).collect();
        let t_t: Vec<f64> = t_logits.iter().map(|&x| x / temp).collect();
        let l_st = lse(&s_t);
        let l_tt = lse(&t_t);

        let ce = l_s1 - s_logits[tgt as usize];
        // KL(p_t || p_s) at temperature T.
        let kl: f64 = (0..vocab)
            .map(|v| {
                let log_pt = t_t[v] - l_tt;
                let log_ps = s_t[v] - l_st;
                log_pt.exp() * (log_pt - log_ps)
            })
            .sum();

        loss[r] = alpha * ce + (1.0 - alpha) * temp * temp * kl;
        lse_s1[r] = l_s1;
        lse_st[r] = l_st;
        lse_tt[r] = l_tt;
    }

    (loss, lse_s1, lse_st, lse_tt, num_valid)
}

/// Pure-Rust f64 reference backward: returns `(dxs, dws, dbs)` — the
/// STUDENT gradients only (the teacher has none by construction).
#[allow(clippy::too_many_arguments)]
pub fn reference_backward_f64(
    xs: &[f64],
    ws: &[f64],
    bs: &[f64],
    xt: &[f64],
    wt: &[f64],
    bt: &[f64],
    targets: &[i64],
    rows: usize,
    vocab: usize,
    hs: usize,
    ht: usize,
    alpha: f64,
    temp: f64,
    ignore_index: i64,
    grad_output: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (_, lse_s1, lse_st, lse_tt, num_valid) = reference_forward_f64(
        xs, ws, bs, xt, wt, bt, targets, rows, vocab, hs, ht, alpha, temp, ignore_index,
    );
    let mut dxs = vec![0.0; rows * hs];
    let mut dws = vec![0.0; vocab * hs];
    let mut dbs = vec![0.0; vocab];
    if num_valid == 0 {
        return (dxs, dws, dbs);
    }
    let scale = grad_output / num_valid as f64;

    for r in 0..rows {
        let tgt = targets[r];
        if tgt == ignore_index {
            continue;
        }
        for v in 0..vocab {
            let mut s_v = bs[v];
            for h in 0..hs {
                s_v += xs[r * hs + h] * ws[v * hs + h];
            }
            let mut t_v = bt[v];
            for h in 0..ht {
                t_v += xt[r * ht + h] * wt[v * ht + h];
            }
            let mut p_s1 = (s_v - lse_s1[r]).exp();
            let p_st = (s_v / temp - lse_st[r]).exp();
            let p_tt = (t_v / temp - lse_tt[r]).exp();
            if v as i64 == tgt {
                p_s1 -= 1.0;
            }
            let dl = alpha * p_s1 + (1.0 - alpha) * temp * (p_st - p_tt);
            let sc = dl * scale;
            for h in 0..hs {
                dxs[r * hs + h] += sc * ws[v * hs + h];
                dws[v * hs + h] += sc * xs[r * hs + h];
            }
            dbs[v] += sc;
        }
    }
    (dxs, dws, dbs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> FusedKlCeConfig {
        FusedKlCeConfig {
            vocab_size: 256,
            student_hidden: 32,
            teacher_hidden: 64,
            batch_size: 2,
            seq_len: 4,
            vocab_tile: 128,
            gpu_sm: 80,
            ignore_index: -100,
        }
    }

    #[test]
    fn validate_accepts_small_cfg() {
        assert!(small_cfg().validate().is_ok());
    }

    #[test]
    fn validate_rejects_large_vocab() {
        let mut cfg = small_cfg();
        cfg.vocab_size = 16384;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("8192"), "err must name the v1 ceiling: {err}");
        assert!(err.contains("deferred"), "err must name the deferral: {err}");
    }

    #[test]
    fn validate_rejects_bad_hidden() {
        let mut cfg = small_cfg();
        cfg.student_hidden = 33;
        assert!(cfg.validate().is_err());
        let mut cfg = small_cfg();
        cfg.teacher_hidden = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_vocab_tile() {
        let mut cfg = small_cfg();
        cfg.vocab_tile = 100;
        assert!(cfg.validate().is_err());
        cfg.vocab_tile = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn ptx_is_null_terminated_and_ascii() {
        let cfg = small_cfg();
        for bytes in [
            synthesize_fused_kl_ce_ptx(&cfg),
            synthesize_fused_kl_ce_backward_ptx(&cfg),
        ] {
            assert_eq!(*bytes.last().unwrap(), 0, "must be null-terminated");
            // ptxas 13.x rejects non-ASCII anywhere in the module
            // (feedback_ptx_comment_ascii_only).
            assert!(
                bytes[..bytes.len() - 1].iter().all(|&b| b.is_ascii()),
                "PTX must be pure ASCII"
            );
        }
    }

    #[test]
    fn ptx_register_declarations_cover_uses() {
        // Guard against the `.reg .f32 %f<3>` class of bug (declares
        // %f0..%f2 while the body uses %f3 — see bugs.md 2026-07-10):
        // scan every numbered register use and assert it is below the
        // declared count for its family.
        let cfg = small_cfg();
        for ptx in [
            String::from_utf8(synthesize_fused_kl_ce_ptx(&cfg)[..].to_vec())
                .unwrap()
                .trim_end_matches('\0')
                .to_string(),
            String::from_utf8(synthesize_fused_kl_ce_backward_ptx(&cfg)[..].to_vec())
                .unwrap()
                .trim_end_matches('\0')
                .to_string(),
        ] {
            for family in ["rd", "r"] {
                let decl_marker = format!(".reg .u64 %{family}<");
                let decl_marker32 = format!(".reg .u32 %{family}<");
                let declared: u32 = ptx
                    .lines()
                    .find_map(|l| {
                        let l = l.trim();
                        let rest = l
                            .strip_prefix(&decl_marker)
                            .or_else(|| l.strip_prefix(&decl_marker32))?;
                        rest.split('>').next()?.parse().ok()
                    })
                    .unwrap_or_else(|| panic!("no decl for %{family} family"));
                let re = regex::Regex::new(&format!(r"%{family}(\d+)")).unwrap();
                let max_used = re
                    .captures_iter(&ptx)
                    .map(|c| c[1].parse::<u32>().unwrap())
                    .max()
                    .unwrap_or(0);
                assert!(
                    max_used < declared,
                    "%{family}{max_used} used but only %{family}<{declared}> declared"
                );
            }
        }
    }

    #[test]
    fn kernel_names_encode_shape() {
        let cfg = small_cfg();
        assert_eq!(cfg.kernel_name(), "nsl_fused_kl_ce_f32_v256_hs32_ht64");
        assert_eq!(
            cfg.bwd_kernel_name(),
            "nsl_fused_kl_ce_backward_f32_v256_hs32_ht64"
        );
        let fwd = String::from_utf8(synthesize_fused_kl_ce_ptx(&cfg)).unwrap();
        assert!(fwd.contains(&cfg.kernel_name()));
        let bwd = String::from_utf8(synthesize_fused_kl_ce_backward_ptx(&cfg)).unwrap();
        assert!(bwd.contains(&cfg.bwd_kernel_name()));
    }

    #[test]
    fn reference_forward_matches_composite_math() {
        // Cross-check the reference against a direct (non-online)
        // computation on a tiny problem.
        let rows = 3;
        let vocab = 5;
        let hs = 2;
        let ht = 3;
        let xs: Vec<f64> = (0..rows * hs).map(|i| (i as f64 * 0.37).sin()).collect();
        let ws: Vec<f64> = (0..vocab * hs).map(|i| (i as f64 * 0.11).cos()).collect();
        let bs: Vec<f64> = (0..vocab).map(|i| i as f64 * 0.01).collect();
        let xt: Vec<f64> = (0..rows * ht).map(|i| (i as f64 * 0.29).sin()).collect();
        let wt: Vec<f64> = (0..vocab * ht).map(|i| (i as f64 * 0.13).cos()).collect();
        let bt: Vec<f64> = (0..vocab).map(|i| -(i as f64) * 0.02).collect();
        let targets = vec![1i64, -100, 4];
        let alpha = 0.3;
        let temp = 2.0;

        let (loss, lse_s1, _lse_st, _lse_tt, num_valid) = reference_forward_f64(
            &xs, &ws, &bs, &xt, &wt, &bt, &targets, rows, vocab, hs, ht, alpha, temp, -100,
        );
        assert_eq!(num_valid, 2);
        assert_eq!(loss[1], 0.0, "ignored row contributes zero");

        // Direct softmax computation for row 0.
        let s0: Vec<f64> = (0..vocab)
            .map(|v| bs[v] + (0..hs).map(|h| xs[h] * ws[v * hs + h]).sum::<f64>())
            .collect();
        let t0: Vec<f64> = (0..vocab)
            .map(|v| bt[v] + (0..ht).map(|h| xt[h] * wt[v * ht + h]).sum::<f64>())
            .collect();
        let softmax = |xs: &[f64]| {
            let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = xs.iter().map(|&x| (x - m).exp()).collect();
            let s: f64 = e.iter().sum();
            e.into_iter().map(|x| x / s).collect::<Vec<f64>>()
        };
        let ps1 = softmax(&s0);
        let ce = -(ps1[1]).ln();
        assert!((lse_s1[0] - (s0[1] - ps1[1].ln())).abs() < 1e-9);
        let pst = softmax(&s0.iter().map(|&x| x / temp).collect::<Vec<_>>());
        let ptt = softmax(&t0.iter().map(|&x| x / temp).collect::<Vec<_>>());
        let kl: f64 = (0..vocab)
            .map(|v| ptt[v] * (ptt[v].ln() - pst[v].ln()))
            .sum();
        let expected = alpha * ce + (1.0 - alpha) * temp * temp * kl;
        assert!(
            (loss[0] - expected).abs() < 1e-9,
            "loss {} != expected {}",
            loss[0],
            expected
        );
    }

    #[test]
    fn reference_backward_matches_finite_differences() {
        let rows = 2;
        let vocab = 4;
        let hs = 2;
        let ht = 2;
        let xs: Vec<f64> = (0..rows * hs).map(|i| (i as f64 * 0.41).sin()).collect();
        let ws: Vec<f64> = (0..vocab * hs).map(|i| (i as f64 * 0.17).cos()).collect();
        let bs: Vec<f64> = (0..vocab).map(|i| i as f64 * 0.03).collect();
        let xt: Vec<f64> = (0..rows * ht).map(|i| (i as f64 * 0.23).sin()).collect();
        let wt: Vec<f64> = (0..vocab * ht).map(|i| (i as f64 * 0.19).cos()).collect();
        let bt: Vec<f64> = (0..vocab).map(|i| -(i as f64) * 0.01).collect();
        let targets = vec![2i64, 0];
        let alpha = 0.6;
        let temp = 3.0;

        let mean_loss = |ws_: &[f64], bs_: &[f64], xs_: &[f64]| -> f64 {
            let (loss, _, _, _, nv) = reference_forward_f64(
                xs_, ws_, bs_, &xt, &wt, &bt, &targets, rows, vocab, hs, ht, alpha, temp, -100,
            );
            loss.iter().sum::<f64>() / nv as f64
        };

        let (dxs, dws, dbs) = reference_backward_f64(
            &xs, &ws, &bs, &xt, &wt, &bt, &targets, rows, vocab, hs, ht, alpha, temp, -100, 1.0,
        );

        let eps = 1e-6;
        // dW_s finite differences.
        for i in [0usize, 3, 7] {
            let mut wp = ws.clone();
            wp[i] += eps;
            let mut wm = ws.clone();
            wm[i] -= eps;
            let fd = (mean_loss(&wp, &bs, &xs) - mean_loss(&wm, &bs, &xs)) / (2.0 * eps);
            assert!(
                (fd - dws[i]).abs() < 1e-5,
                "dW[{i}]: fd {fd} vs analytic {}",
                dws[i]
            );
        }
        // dbias_s.
        for i in [0usize, 2] {
            let mut bp = bs.clone();
            bp[i] += eps;
            let mut bm = bs.clone();
            bm[i] -= eps;
            let fd = (mean_loss(&ws, &bp, &xs) - mean_loss(&ws, &bm, &xs)) / (2.0 * eps);
            assert!(
                (fd - dbs[i]).abs() < 1e-5,
                "dbias[{i}]: fd {fd} vs analytic {}",
                dbs[i]
            );
        }
        // dx_s.
        for i in [0usize, 3] {
            let mut xp = xs.clone();
            xp[i] += eps;
            let mut xm = xs.clone();
            xm[i] -= eps;
            let fd = (mean_loss(&ws, &bs, &xp) - mean_loss(&ws, &bs, &xm)) / (2.0 * eps);
            assert!(
                (fd - dxs[i]).abs() < 1e-5,
                "dx[{i}]: fd {fd} vs analytic {}",
                dxs[i]
            );
        }
    }
}
