//! G3 — Fused linear-CE with separator-skip (v1, scalar F32).
//!
//! Emits two PTX kernels that implement:
//!   loss = cross_entropy(x @ W^T + bias, targets)
//! in a single pass per (batch, sequence) row, with a fast-path branch that
//! skips the entire matmul + softmax for positions labelled `-100`.
//!
//! ## Scope
//!
//! v1 is research-grade: scalar `fma.rn.f32`, F32 only, vocab ≤ 8192.
//! MMA tiling, fp16/bf16 storage, vocab > 8192, and quantised-W paths are
//! all deferred to v2.  See `known_limits` in the design doc.
//!
//! ## API
//!
//! ```rust,no_run
//! use nsl_codegen::fused_linear_ce::{FusedLinearCEConfig, Dtype};
//! let cfg = FusedLinearCEConfig {
//!     vocab_size: 4096, hidden_size: 128, seq_len: 64, batch_size: 2,
//!     vocab_tile: 1024, gpu_sm: 80, dtype: Dtype::F32,
//!     ignore_index: -100, max_vocab_v1: 8192,
//! };
//! cfg.validate().unwrap();
//! let fwd_ptx = nsl_codegen::fused_linear_ce::synthesize_fused_linear_ce_ptx(&cfg);
//! let bwd_ptx = nsl_codegen::fused_linear_ce::synthesize_fused_linear_ce_backward_ptx(&cfg);
//! ```

// ─── Config ─────────────────────────────────────────────────────────────────

/// Dtype selector for FusedLinearCE.  v1 supports F32 only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
}

impl Dtype {
    /// Human-readable tag used in kernel names and diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
        }
    }
}

/// All tunable parameters for the fused linear-CE kernel pair.
///
/// Construct directly; call [`FusedLinearCEConfig::validate`] before
/// synthesising PTX.
#[derive(Debug, Clone)]
pub struct FusedLinearCEConfig {
    /// Number of vocabulary entries (classes in the CE loss).
    pub vocab_size: u32,
    /// Hidden dimension (must be divisible by 32).
    pub hidden_size: u32,
    /// Sequence length.
    pub seq_len: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Vocabulary tile size for the streaming online-softmax loop.
    /// Default: 1024.
    pub vocab_tile: u32,
    /// Target SM version (e.g. 80 for sm_80).
    pub gpu_sm: u32,
    /// Element dtype — F32 only in v1.
    pub dtype: Dtype,
    /// Label value that means "ignore this token" (default -100).
    pub ignore_index: i64,
    /// Hard cap on `vocab_size` for v1.  Synthesiser aborts if exceeded.
    pub max_vocab_v1: u32,
}

impl Default for FusedLinearCEConfig {
    fn default() -> Self {
        FusedLinearCEConfig {
            vocab_size: 4096,
            hidden_size: 128,
            seq_len: 64,
            batch_size: 2,
            vocab_tile: 1024,
            gpu_sm: 80,
            dtype: Dtype::F32,
            ignore_index: -100,
            max_vocab_v1: 8192,
        }
    }
}

impl FusedLinearCEConfig {
    /// Validate configuration; returns `Err` with a descriptive message on
    /// any unsupported combination.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size > self.max_vocab_v1 {
            return Err(format!(
                "fused_linear_ce v1: vocab_size {} exceeds hard cap {} — \
                 use standard CrossEntropyLoss for large-vocab models (v2 scope)",
                self.vocab_size, self.max_vocab_v1
            ));
        }
        if !self.hidden_size.is_multiple_of(32) {
            return Err(format!(
                "fused_linear_ce: hidden_size {} must be divisible by 32",
                self.hidden_size
            ));
        }
        if self.dtype != Dtype::F32 {
            return Err("fused_linear_ce v1: only Dtype::F32 is supported".into());
        }
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err("fused_linear_ce: batch_size * seq_len must be > 0".into());
        }
        if self.vocab_tile == 0 || self.vocab_tile > self.vocab_size {
            return Err(format!(
                "fused_linear_ce: vocab_tile {} must be in [1, vocab_size={}]",
                self.vocab_tile, self.vocab_size
            ));
        }
        // v1 emits the inner vocab-tile fill as `vtile_per_thread = vocab_tile / 128`
        // sub-iterations per 128-thread CTA. If vocab_tile is not a multiple of
        // 128, integer-divide floors and the upper (vocab_tile % 128) entries
        // are NEVER written to smem; the subsequent online-softmax max/sum
        // reductions on thread 0 then read UNINITIALIZED smem for those slots,
        // silently corrupting the loss + gradient. Reject at validation time so
        // the silent-corruption path is unreachable from any caller (FFI or
        // @fused_lm_ce decorator).
        if !self.vocab_tile.is_multiple_of(128) {
            return Err(format!(
                "fused_linear_ce: vocab_tile {} must be a multiple of 128 (the \
                 inner-loop tile is filled by 128 threads in lockstep; non-128- \
                 aligned tiles leave the tail uninitialised in smem and corrupt \
                 the online-softmax reduction)",
                self.vocab_tile
            ));
        }
        Ok(())
    }

    /// Unique kernel name that encodes the vocab + hidden specialisation so
    /// the cudarc module cache differentiates specialisations.
    pub fn kernel_name(&self) -> String {
        format!(
            "nsl_fused_linear_ce_{}_v{}_h{}",
            self.dtype.tag(),
            self.vocab_size,
            self.hidden_size
        )
    }

    /// Name of the backward kernel.
    pub fn bwd_kernel_name(&self) -> String {
        format!(
            "nsl_fused_linear_ce_backward_{}_v{}_h{}",
            self.dtype.tag(),
            self.vocab_size,
            self.hidden_size
        )
    }

    /// Shared-memory budget per CTA: logits tile (vocab_tile * 4 bytes) +
    /// warp-shuffle scratch / LSE-max scalars (32 bytes pad).
    pub fn shared_mem_bytes(&self) -> u32 {
        self.vocab_tile * 4 + 32
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn sm_tag(&self) -> u32 {
        // v1 targets sm_80+; fall back gracefully if caller passes sm_75.
        self.gpu_sm.max(80)
    }

    fn ptx_header(&self) -> String {
        format!(
            ".version 7.0\n.target sm_{}\n.address_size 64\n",
            self.sm_tag()
        )
    }
}

// ─── Forward kernel synthesis ────────────────────────────────────────────────

/// Synthesise the forward PTX for the fused linear-CE kernel (null-terminated).
///
/// Returns null-terminated PTX bytes suitable for `cuModuleLoadData` — matching
/// the same convention as `backend_ptx::lower_kir_to_ptx`.
///
/// Grid: `(batch_size * seq_len, 1, 1)` — one CTA per token row.
/// Block: `(128, 1, 1)`.
///
/// Each CTA computes:
///   1. Load `target = targets[row_idx]`.
///   2. If `target == ignore_index`, write `loss=0, lse=0` and `ret`.
///   3. Otherwise: stream through vocabulary tiles, accumulating
///      online-max + log-sum-exp over `logit_v = dot(x[row], W[v]) + bias[v]`.
///      After the tile loop, compute `loss = -(logit_at_target - lse)` and
///      write to `loss_out` and `lse_out` (for backward reuse).
pub fn synthesize_fused_linear_ce_ptx(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    let mut bytes = emit_fwd_kernel(cfg).into_bytes();
    bytes.push(0); // null-terminate for cuModuleLoadData
    bytes
}

/// Synthesise the backward PTX for the fused linear-CE kernel (null-terminated).
///
/// Returns null-terminated PTX bytes suitable for `cuModuleLoadData`.
///
/// Grid/block: same as forward.
/// Recomputes logits (no logits buffer saved), computes
/// `dlogits_v = (softmax_v - 1{v==target}) * grad_output / num_valid`,
/// then scatters `dx += dlogits_v * W[v, :]` and
/// `dW[v, :] += dlogits_v * x[row, :]` via atomic adds.
pub fn synthesize_fused_linear_ce_backward_ptx(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    let mut bytes = emit_bwd_kernel(cfg).into_bytes();
    bytes.push(0); // null-terminate for cuModuleLoadData
    bytes
}

// ─── PTX emission — forward ───────────────────────────────────────────────────
//
// Shared-memory layout (4 bytes per float, all at 4-byte aligned offsets):
//   [0 .. vtile*4)       : logits tile — f32 per vocab entry
//   [vtile*4 .. vtile*4+4) : logit_at_target scratch (written by whichever
//                            thread computes the target vocab position)
//   [vtile*4+4 .. smem)  : alignment / future use


/// Clean reimplementation of the forward kernel with correct addressing.
fn emit_fwd_kernel(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = vocab.div_ceil(vtile);
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();
    // Smem layout: [0 .. vtile*4) = logits tile,
    //              [vtile*4 .. vtile*4+4) = logit_at_target scratch.
    let lat_offset = vtile * 4;

    // Build PTX as a single format-string vector.
    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    lines.push(cfg.ptx_header());
    lines.push(format!(".extern .shared .align 4 .b8 smem_scratch[{smem_bytes}];"));
    lines.push(String::new());
    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_loss_out,"));
    lines.push(p("\t.param .u64 param_lse_out,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H"));
    lines.push(p(") {"));
    // Registers.
    lines.push(p("\t.reg .u64 %rd<30>;"));
    lines.push(p("\t.reg .u32 %r<20>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .f32 %flog, %facc, %fmax, %fsum, %ftmax, %flse, %floss;"));
    lines.push(p("\t.reg .f32 %fa, %fb, %flog2e, %fln2, %ftmp;"));
    lines.push(p("\t.reg .pred %pskip, %pv, %pth0, %ptgt;"));
    lines.push(String::new());

    // Load params.
    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_loss_out];"));
    lines.push(p("\tld.param.u64 %rd5, [param_lse_out];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.x;   // row_idx"));
    lines.push(p("\tmov.u32 %r1, %tid.x;     // tid"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B; // log2(e)"));
    lines.push(p("\tmov.f32 %fln2,   0f3F317218; // ln(2)"));
    lines.push(String::new());

    // Thread 0 initialises logit_at_target smem slot to -INF.
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra INIT_DONE;"));
    lines.push(p("\tmov.u64 %rd6, smem_scratch;"));
    lines.push(format!("\tadd.u64 %rd6, %rd6, {lat_offset};"));
    lines.push(p("\tst.shared.f32 [%rd6], 0f80800000; // -INF sentinel"));
    lines.push(p("INIT_DONE:"));
    lines.push(p("\tbar.sync 0;"));
    lines.push(String::new());

    // Load target[row_idx].
    lines.push(p("\t// Load targets[row_idx] (i64)"));
    lines.push(p("\tcvt.u64.u32 %rd7, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd7, %rd7, 8;"));
    lines.push(p("\tadd.u64 %rd7, %rd3, %rd7;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd7];"));
    lines.push(String::new());

    // Skip branch.
    lines.push(format!("\t// setp.eq.s64: if target == {ignore} skip"));
    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@%pskip bra SKIP_LABEL;"));
    lines.push(String::new());

    // x_row_base = x + row*H*4.
    lines.push(format!("\t// x_row_base = x + row_idx * {hidden} * 4"));
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(format!("\tmov.u32 %r2, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd9, %r2;"));
    lines.push(p("\tmul.lo.u64 %rd8, %rd8, %rd9;"));
    lines.push(p("\tshl.b64 %rd8, %rd8, 2;"));
    lines.push(p("\tadd.u64 %rd8, %rd0, %rd8; // %rd8 = x_row_base"));
    lines.push(String::new());

    // Init online-softmax accumulators.
    lines.push(p("\tmov.f32 %fmax, 0f80800000; // -INF"));
    lines.push(p("\tmov.f32 %fsum, 0f00000000; // 0.0"));
    lines.push(String::new());

    // Outer tile loop.
    lines.push(p("\tmov.u32 %r3, 0; // tile_idx"));
    lines.push(p("TILE_LOOP:"));
    lines.push(format!("\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base"));
    lines.push(String::new());

    // Inner loop: stride-128 through tile.
    lines.push(p("\t\tmov.u32 %r5, 0; // sub-tile counter"));
    lines.push(p("\t\tINNER_LOOP:"));
    lines.push(p("\t\t\tmul.lo.u32 %r6, %r5, 128;"));
    lines.push(p("\t\t\tadd.u32 %r6, %r6, %r1;"));
    lines.push(p("\t\t\tadd.u32 %r6, %r6, %r4; // v_idx"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r6, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra INNER_SKIP;"));
    lines.push(String::new());

    // W_row_base = W + v_idx * H * 4.
    lines.push(format!("\t\t\t// W_row_base = W + v_idx * {hidden} * 4"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd10, %r6;"));
    lines.push(format!("\t\t\tmov.u32 %r7, {hidden};"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd11, %r7;"));
    lines.push(p("\t\t\tmul.lo.u64 %rd10, %rd10, %rd11;"));
    lines.push(p("\t\t\tshl.b64 %rd10, %rd10, 2;"));
    lines.push(p("\t\t\tadd.u64 %rd10, %rd1, %rd10; // %rd10 = W_row_base"));
    lines.push(String::new());

    // Dot-product loop.
    lines.push(p("\t\t\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\t\t\tmov.u32 %r8, 0; // h"));
    lines.push(p("\t\t\tDOT_LOOP:"));
    lines.push(p("\t\t\t\tcvt.u64.u32 %rd12, %r8;"));
    lines.push(p("\t\t\t\tshl.b64 %rd12, %rd12, 2; // h * 4"));
    lines.push(p("\t\t\t\tadd.u64 %rd13, %rd8, %rd12; // x_row_base + h*4"));
    lines.push(p("\t\t\t\tld.global.f32 %fa, [%rd13];"));
    lines.push(p("\t\t\t\tadd.u64 %rd14, %rd10, %rd12; // W_row_base + h*4"));
    lines.push(p("\t\t\t\tld.global.f32 %fb, [%rd14];"));
    lines.push(p("\t\t\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\t\t\tadd.u32 %r8, %r8, 1;"));
    lines.push(format!("\t\t\t\tsetp.lt.u32 %pv, %r8, {hidden};"));
    lines.push(p("\t\t\t\t@%pv bra DOT_LOOP;"));
    lines.push(String::new());

    // Add bias[v_idx].
    lines.push(p("\t\t\t// facc += bias[v_idx]"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd15, %r6;"));
    lines.push(p("\t\t\tshl.b64 %rd15, %rd15, 2;"));
    lines.push(p("\t\t\tadd.u64 %rd15, %rd2, %rd15;"));
    lines.push(p("\t\t\tld.global.f32 %ftmp, [%rd15];"));
    lines.push(p("\t\t\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(String::new());

    // Store logit to smem tile.
    // thread_local_slot = r5*128 + tid  (unique within vtile since vtile = vtile_per_thread * 128)
    lines.push(p("\t\t\t// Store logit to smem_scratch[(r5*128+tid)*4]"));
    lines.push(p("\t\t\tmul.lo.u32 %r9, %r5, 128;"));
    lines.push(p("\t\t\tadd.u32 %r9, %r9, %r1;"));
    lines.push(p("\t\t\tshl.b32 %r9, %r9, 2;"));
    lines.push(p("\t\t\tmov.u64 %rd16, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd17, %r9;"));
    lines.push(p("\t\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\t\tst.shared.f32 [%rd16], %facc;"));
    lines.push(String::new());

    // If v_idx == target, store logit_at_target to smem scratch slot.
    // Any thread may write here; only one thread will have v_idx == target.
    lines.push(p("\t\t\t// If v_idx == target, record logit_at_target in smem"));
    lines.push(p("\t\t\tcvt.s64.u32 %rd18, %r6;"));
    lines.push(p("\t\t\tsetp.eq.s64 %ptgt, %rd18, %tgt64;"));
    lines.push(p("\t\t\t@!%ptgt bra NOT_TARGET;"));
    lines.push(p("\t\t\tmov.u64 %rd19, smem_scratch;"));
    lines.push(format!("\t\t\tadd.u64 %rd19, %rd19, {lat_offset};"));
    lines.push(p("\t\t\tst.shared.f32 [%rd19], %facc;"));
    lines.push(p("\t\t\tNOT_TARGET:"));
    lines.push(String::new());

    lines.push(p("\t\t\tINNER_SKIP:"));
    lines.push(p("\t\t\tadd.u32 %r5, %r5, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r5, {vtile_per_thread};"));
    lines.push(p("\t\t\t@%pv bra INNER_LOOP;"));
    lines.push(String::new());

    // Sync: all logits now in smem.
    lines.push(p("\t\tbar.sync 0;"));
    lines.push(String::new());

    // Thread 0: scan smem tile, update running max+sum via online softmax.
    lines.push(p("\t\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t\t@!%pth0 bra TILE_REDUCE_DONE;"));
    lines.push(String::new());

    // Step 1: find tile_max.
    lines.push(p("\t\tmov.f32 %ftmax, 0f80800000;"));
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tSMEM_MAX_LOOP:"));
    lines.push(p("\t\t\tadd.u32 %r11, %r4, %r10; // v_base + i"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r11, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra SMEM_MAX_DONE;"));
    lines.push(p("\t\t\tshl.b32 %r12, %r10, 2;"));
    lines.push(p("\t\t\tmov.u64 %rd20, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd21, %r12;"));
    lines.push(p("\t\t\tadd.u64 %rd20, %rd20, %rd21;"));
    lines.push(p("\t\t\tld.shared.f32 %ftmp, [%rd20];"));
    lines.push(p("\t\t\tmax.f32 %ftmax, %ftmax, %ftmp;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {vtile};"));
    lines.push(p("\t\t\t@%pv bra SMEM_MAX_LOOP;"));
    lines.push(p("\t\tSMEM_MAX_DONE:"));
    lines.push(String::new());

    // Online softmax update: new_max = max(running_max, tile_max).
    // rescale: fsum *= exp(fmax - new_max)
    // accumulate: fsum += sum_v exp(logit_v - new_max)
    lines.push(p("\t\t// Online-softmax update"));
    lines.push(p("\t\tmax.f32 %flog, %fmax, %ftmax; // new_max (reuse %flog temporarily)"));
    lines.push(p("\t\tsub.f32 %fmax, %fmax, %flog; // old_max - new_max (negative or 0)"));
    lines.push(p("\t\tmul.f32 %fmax, %fmax, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %fmax, %fmax; // exp(old_max - new_max)"));
    lines.push(p("\t\tmul.f32 %fsum, %fsum, %fmax; // rescale"));
    lines.push(p("\t\tmov.f32 %fmax, %flog; // fmax = new_max"));
    lines.push(String::new());

    // Accumulate tile sum.
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tSMEM_SUM_LOOP:"));
    lines.push(p("\t\t\tadd.u32 %r11, %r4, %r10;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r11, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra SMEM_SUM_DONE;"));
    lines.push(p("\t\t\tshl.b32 %r12, %r10, 2;"));
    lines.push(p("\t\t\tmov.u64 %rd20, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd21, %r12;"));
    lines.push(p("\t\t\tadd.u64 %rd20, %rd20, %rd21;"));
    lines.push(p("\t\t\tld.shared.f32 %ftmp, [%rd20];"));
    lines.push(p("\t\t\tsub.f32 %ftmp, %ftmp, %fmax; // logit - new_max"));
    lines.push(p("\t\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\t\tadd.f32 %fsum, %fsum, %ftmp;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {vtile};"));
    lines.push(p("\t\t\t@%pv bra SMEM_SUM_LOOP;"));
    lines.push(p("\t\tSMEM_SUM_DONE:"));
    lines.push(String::new());

    lines.push(p("\t\tTILE_REDUCE_DONE:"));
    lines.push(p("\t\tbar.sync 0;"));
    lines.push(String::new());

    // Advance tile.
    lines.push(p("\t\tadd.u32 %r3, %r3, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r3, {n_tiles};"));
    lines.push(p("\t\t@%pv bra TILE_LOOP;"));
    lines.push(String::new());

    // After tile loop: thread 0 computes lse and loss.
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra WRITE_DONE;"));
    lines.push(String::new());

    // lse = log(fsum) + fmax.
    lines.push(p("\tlg2.approx.f32 %flse, %fsum; // log2(sum_exp)"));
    lines.push(p("\tmul.f32 %flse, %flse, %fln2; // ln(sum_exp)"));
    lines.push(p("\tadd.f32 %flse, %flse, %fmax; // lse = ln(sum_exp) + max"));
    lines.push(String::new());

    // logit_at_target from smem.
    lines.push(p("\tmov.u64 %rd22, smem_scratch;"));
    lines.push(format!("\tadd.u64 %rd22, %rd22, {lat_offset};"));
    lines.push(p("\tld.shared.f32 %flog, [%rd22]; // logit_at_target"));
    lines.push(String::new());

    // loss = lse - logit_at_target.
    lines.push(p("\tsub.f32 %floss, %flse, %flog;"));
    lines.push(String::new());

    // Write to HBM.
    lines.push(p("\tcvt.u64.u32 %rd23, %r0;"));
    lines.push(p("\tshl.b64 %rd23, %rd23, 2;"));
    lines.push(p("\tadd.u64 %rd24, %rd4, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], %floss;"));
    lines.push(p("\tadd.u64 %rd24, %rd5, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], %flse;"));
    lines.push(p("\tbra WRITE_DONE;"));
    lines.push(String::new());

    // Skip label: write zeros.
    lines.push(p("SKIP_LABEL:"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra WRITE_DONE;"));
    lines.push(p("\tcvt.u64.u32 %rd23, %r0;"));
    lines.push(p("\tshl.b64 %rd23, %rd23, 2;"));
    lines.push(p("\tadd.u64 %rd24, %rd4, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], 0f00000000;"));
    lines.push(p("\tadd.u64 %rd24, %rd5, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], 0f00000000;"));
    lines.push(String::new());

    lines.push(p("WRITE_DONE:"));
    lines.push(p("\tret;"));
    lines.push(p("}"));

    lines.join("\n")
}

// ─── PTX emission — backward ──────────────────────────────────────────────────

fn emit_bwd_kernel(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.bwd_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = vocab.div_ceil(vtile);
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();

    let mut s = String::new();

    s.push_str(&cfg.ptx_header());
    s.push('\n');

    s.push_str(&format!(
        ".extern .shared .align 4 .b8 smem_scratch[{}];\n\n",
        smem_bytes
    ));

    // Kernel signature: adds grad_output scalar, saved lse, dx_out, dW_out, dbias_out.
    // num_valid is passed as a u32 launch param (host-computed).
    s.push_str(&format!(
        ".visible .entry {name}(\n\
         \t.param .f32 param_grad_output,\n\
         \t.param .u64 param_x,\n\
         \t.param .u64 param_w,\n\
         \t.param .u64 param_bias,\n\
         \t.param .u64 param_targets,\n\
         \t.param .u64 param_lse,\n\
         \t.param .u64 param_dx_out,\n\
         \t.param .u64 param_dw_out,\n\
         \t.param .u64 param_dbias_out,\n\
         \t.param .u32 param_B,\n\
         \t.param .u32 param_S,\n\
         \t.param .u32 param_V,\n\
         \t.param .u32 param_H,\n\
         \t.param .u32 param_num_valid\n\
         ) {{\n"
    ));

    // Register declarations.
    s.push_str(
        "\t.reg .u64 %rd<24>;\n\
         \t.reg .u32 %r<20>;\n\
         \t.reg .s64 %target_val;\n\
         \t.reg .f32 %f<20>;\n\
         \t.reg .f32 %logit_acc;\n\
         \t.reg .f32 %grad_output;\n\
         \t.reg .f32 %lse_val;\n\
         \t.reg .f32 %scale;\n\
         \t.reg .pred %p_skip;\n\
         \t.reg .pred %p_valid;\n\
         \t.reg .pred %p_intile;\n\
         \t.reg .pred %p_is_target;\n\
         \t.reg .u32 %num_valid;\n\
         \t.reg .f32 %num_valid_f;\n\
    \n",
    );

    // Load parameters.
    s.push_str(
        "\tld.param.f32 %grad_output, [param_grad_output];\n\
         \tld.param.u64 %rd0, [param_x];\n\
         \tld.param.u64 %rd1, [param_w];\n\
         \tld.param.u64 %rd2, [param_bias];\n\
         \tld.param.u64 %rd3, [param_targets];\n\
         \tld.param.u64 %rd4, [param_lse];\n\
         \tld.param.u64 %rd5, [param_dx_out];\n\
         \tld.param.u64 %rd6, [param_dw_out];\n\
         \tld.param.u64 %rd7, [param_dbias_out];\n\
         \tld.param.u32 %num_valid, [param_num_valid];\n\
         \tcvt.rn.f32.u32 %num_valid_f, %num_valid;\n\
    \n",
    );

    // row_idx = ctaid.x.
    s.push_str(
        "\tmov.u32 %r0, %ctaid.x;\n\
         \tmov.u32 %r1, %tid.x;\n\
    \n",
    );

    // Load target.
    s.push_str(
        "\tcvt.u64.u32 %rd8, %r0;\n\
         \tmul.lo.u64 %rd8, %rd8, 8;\n\
         \tadd.u64 %rd8, %rd3, %rd8;\n\
         \tld.global.s64 %target_val, [%rd8];\n\
    \n",
    );

    // Skip branch.
    s.push_str(&format!(
        "\tsetp.eq.s64 %p_skip, %target_val, {ignore};\n\
         \t@%p_skip bra BWD_SKIP_LABEL;\n\
    \n"
    ));

    // Live path: load saved lse.
    s.push_str(
        "\tcvt.u64.u32 %rd9, %r0;\n\
         \tshl.b64 %rd9, %rd9, 2;\n\
         \tadd.u64 %rd9, %rd4, %rd9;\n\
         \tld.global.f32 %lse_val, [%rd9];\n\
    \n",
    );

    // x_row_base = x + row_idx * H * 4.
    s.push_str(&format!(
        "\tcvt.u64.u32 %rd10, %r0;\n\
         \tmov.u32 %r2, {hidden};\n\
         \tcvt.u64.u32 %rd11, %r2;\n\
         \tmul.lo.u64 %rd10, %rd10, %rd11;\n\
         \tshl.b64 %rd10, %rd10, 2;\n\
         \tadd.u64 %rd10, %rd0, %rd10;\n\
         \t// dx_row_base = dx_out + row_idx * H * 4\n\
         \tcvt.u64.u32 %rd20, %r0;\n\
         \tmul.lo.u64 %rd20, %rd20, %rd11;\n\
         \tshl.b64 %rd20, %rd20, 2;\n\
         \tadd.u64 %rd20, %rd5, %rd20;\n\
    \n"
    ));

    // scale = grad_output / num_valid_f.
    s.push_str(
        "\tdiv.rn.f32 %scale, %grad_output, %num_valid_f;\n\
    \n",
    );

    // log2e constant for exp.
    s.push_str("\tmov.f32 %f15, 0f3FB8AA3B; // log2(e)\n\n");

    // Outer tile loop.
    s.push_str(
        "\tmov.u32 %r3, 0; // tile_idx\n\
         BWD_TILE_LOOP:\n",
    );

    s.push_str(&format!(
        "\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base\n\
    \n"
    ));

    // Inner loop: each thread handles its vocab slice.
    s.push_str(
        "\t\tmov.u32 %r5, 0; // inner counter\n\
         BWD_INNER_LOOP:\n",
    );

    s.push_str(
        "\t\t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\t\tadd.u32 %r6, %r6, %r1;\n\
         \t\t\tadd.u32 %r6, %r6, %r4;\n\
         \t\t\t// v_idx = %r6\n",
    );

    s.push_str(&format!(
        "\t\t\tsetp.lt.u32 %p_valid, %r6, {vocab};\n\
         \t\t\t@!%p_valid bra BWD_INNER_SKIP;\n\
    \n"
    ));

    // Recompute logit_v = dot(x_row, W[v]) + bias[v].
    s.push_str(&format!(
        "\t\t\t// W_row_base for v_idx\n\
         \t\t\tcvt.u64.u32 %rd12, %r6;\n\
         \t\t\tmov.u32 %r7, {hidden};\n\
         \t\t\tcvt.u64.u32 %rd13, %r7;\n\
         \t\t\tmul.lo.u64 %rd12, %rd12, %rd13;\n\
         \t\t\tshl.b64 %rd12, %rd12, 2;\n\
         \t\t\tadd.u64 %rd12, %rd1, %rd12;\n\
    \n"
    ));

    s.push_str(
        "\t\t\tmov.f32 %logit_acc, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         BWD_DOT_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r8;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 2;\n\
         \t\t\t\tadd.u64 %rd15, %rd10, %rd14;\n\
         \t\t\t\tld.global.f32 %f0, [%rd15];\n\
         \t\t\t\tadd.u64 %rd16, %rd12, %rd14;\n\
         \t\t\t\tld.global.f32 %f1, [%rd16];\n\
         \t\t\t\tfma.rn.f32 %logit_acc, %f0, %f1, %logit_acc;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\t\tsetp.lt.u32 %p_valid, %r8, {hidden};\n\
         \t\t\t\t@%p_valid bra BWD_DOT_LOOP;\n\
    \n"
    ));

    // Add bias.
    s.push_str(
        "\t\t\t// bias\n\
         \t\t\tcvt.u64.u32 %rd17, %r6;\n\
         \t\t\tshl.b64 %rd17, %rd17, 2;\n\
         \t\t\tadd.u64 %rd17, %rd2, %rd17;\n\
         \t\t\tld.global.f32 %f2, [%rd17];\n\
         \t\t\tadd.f32 %logit_acc, %logit_acc, %f2;\n\
    \n",
    );

    // Compute softmax: p_v = exp(logit_v - lse).
    s.push_str(
        "\t\t\t// p_v = exp(logit_v - lse_val)\n\
         \t\t\tsub.f32 %f3, %logit_acc, %lse_val;\n\
         \t\t\tmul.f32 %f3, %f3, %f15; // * log2(e)\n\
         \t\t\tex2.approx.f32 %f3, %f3;  // p_v\n\
    \n",
    );

    // Subtract 1 if v == target.
    s.push_str(
        "\t\t\t// dlogit_v = p_v - (v == target ? 1 : 0)\n\
         \t\t\tcvt.s64.u32 %rd18, %r6;\n\
         \t\t\tsetp.eq.s64 %p_is_target, %rd18, %target_val;\n\
         \t\t\t@%p_is_target sub.f32 %f3, %f3, 0f3F800000; // -= 1.0\n\
         \t\t\t// scaled = dlogit_v * scale\n\
         \t\t\tmul.f32 %f4, %f3, %scale;\n\
    \n",
    );

    // Scatter to dx_out[row, h] += scaled * W[v, h].
    // And dW_out[v, h] += scaled * x[row, h].
    // And dbias_out[v] += scaled.
    s.push_str(&format!(
        "\t\t\t// Scatter: dx and dW (loop over H)\n\
         \t\t\t// dW_row_base = dW_out + v_idx * H * 4\n\
         \t\t\tcvt.u64.u32 %rd21, %r6;\n\
         \t\t\tmov.u32 %r9, {hidden};\n\
         \t\t\tcvt.u64.u32 %rd22, %r9;\n\
         \t\t\tmul.lo.u64 %rd21, %rd21, %rd22;\n\
         \t\t\tshl.b64 %rd21, %rd21, 2;\n\
         \t\t\tadd.u64 %rd21, %rd6, %rd21; // dW_row_base\n\
    \n"
    ));

    s.push_str(
        "\t\t\tmov.u32 %r9, 0; // h counter\n\
         BWD_H_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd23, %r9;\n\
         \t\t\t\tshl.b64 %rd23, %rd23, 2;\n\
         \t\t\t\t// W[v, h]\n\
         \t\t\t\tadd.u64 %rd14, %rd12, %rd23;\n\
         \t\t\t\tld.global.f32 %f5, [%rd14];\n\
         \t\t\t\t// dx_out[row, h] += scaled * W[v, h]\n\
         \t\t\t\tmul.f32 %f6, %f4, %f5;\n\
         \t\t\t\tadd.u64 %rd14, %rd20, %rd23;\n\
         \t\t\t\tred.global.add.f32 [%rd14], %f6;\n\
         \t\t\t\t// x[row, h]\n\
         \t\t\t\tadd.u64 %rd14, %rd10, %rd23;\n\
         \t\t\t\tld.global.f32 %f7, [%rd14];\n\
         \t\t\t\t// dW_out[v, h] += scaled * x[row, h]\n\
         \t\t\t\tmul.f32 %f8, %f4, %f7;\n\
         \t\t\t\tadd.u64 %rd14, %rd21, %rd23;\n\
         \t\t\t\tred.global.add.f32 [%rd14], %f8;\n\
         \t\t\t\tadd.u32 %r9, %r9, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\t\tsetp.lt.u32 %p_valid, %r9, {hidden};\n\
         \t\t\t\t@%p_valid bra BWD_H_LOOP;\n\
    \n"
    ));

    // dbias_out[v] += scaled.
    s.push_str(
        "\t\t\t// dbias_out[v] += scaled\n\
         \t\t\tcvt.u64.u32 %rd14, %r6;\n\
         \t\t\tshl.b64 %rd14, %rd14, 2;\n\
         \t\t\tadd.u64 %rd14, %rd7, %rd14;\n\
         \t\t\tred.global.add.f32 [%rd14], %f4;\n\
    \n",
    );

    s.push_str(
        "BWD_INNER_SKIP:\n\
         \t\t\tadd.u32 %r5, %r5, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\tsetp.lt.u32 %p_valid, %r5, {vtile_per_thread};\n\
         \t\t\t@%p_valid bra BWD_INNER_LOOP;\n\
    \n"
    ));

    // Advance tile counter.
    s.push_str("\t\tadd.u32 %r3, %r3, 1;\n");

    s.push_str(&format!(
        "\t\tsetp.lt.u32 %p_valid, %r3, {n_tiles};\n\
         \t\t@%p_valid bra BWD_TILE_LOOP;\n\
    \n"
    ));

    s.push_str("\tbra BWD_DONE;\n\n");

    // Skip path: zero out dx_out[row, :].
    s.push_str(
        "BWD_SKIP_LABEL:\n\
         \t// Zero dx_out[row, :] for skipped token\n",
    );

    s.push_str(&format!(
        "\tcvt.u64.u32 %rd10, %r0;\n\
         \tmov.u32 %r2, {hidden};\n\
         \tcvt.u64.u32 %rd11, %r2;\n\
         \tmul.lo.u64 %rd10, %rd10, %rd11;\n\
         \tshl.b64 %rd10, %rd10, 2;\n\
         \tadd.u64 %rd10, %rd5, %rd10; // dx_row_base\n\
    \n"
    ));

    // Each thread zeros its slice of H.
    s.push_str(&format!(
        "\t// Thread r1 zeros H/128 elements (stride 128)\n\
         \tmov.u32 %r5, 0;\n\
         BWD_ZERO_LOOP:\n\
         \t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\tadd.u32 %r6, %r6, %r1;\n\
         \t\tsetp.lt.u32 %p_valid, %r6, {hidden};\n\
         \t\t@!%p_valid bra BWD_ZERO_DONE;\n\
         \t\tshl.b32 %r6, %r6, 2;\n\
         \t\tcvt.u64.u32 %rd12, %r6;\n\
         \t\tadd.u64 %rd12, %rd10, %rd12;\n\
         \t\tst.global.f32 [%rd12], 0f00000000;\n\
         \t\tadd.u32 %r5, %r5, 1;\n\
         \t\tbra BWD_ZERO_LOOP;\n\
         BWD_ZERO_DONE:\n\
    \n"
    ));

    s.push_str("BWD_DONE:\n\tret;\n}\n");

    s
}

// ─── Inline tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> FusedLinearCEConfig {
        FusedLinearCEConfig::default()
    }

    // ── Config validation ────────────────────────────────────────────────

    #[test]
    fn test_validate_rejects_oversized_vocab() {
        let mut cfg = default_cfg();
        cfg.vocab_size = 9000;
        assert!(cfg.validate().is_err());
        assert!(cfg.validate().unwrap_err().contains("vocab_size"));
    }

    #[test]
    fn test_validate_rejects_unaligned_hidden() {
        let mut cfg = default_cfg();
        cfg.hidden_size = 33;
        assert!(cfg.validate().is_err());
        assert!(cfg.validate().unwrap_err().contains("hidden_size"));
    }

    #[test]
    fn test_validate_rejects_zero_seq() {
        let mut cfg = default_cfg();
        cfg.seq_len = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_unaligned_vocab_tile() {
        // The v1 inner fill is 128-thread-wide; a non-128-aligned tile leaves
        // the tail uninitialised in smem and silently corrupts the online-
        // softmax reduction (caught in adversarial review). Verify validate()
        // rejects all such configs.
        for bad in [1u32, 100, 127, 200, 256 + 1, 1023, 1025, 7777] {
            let mut cfg = default_cfg();
            cfg.vocab_tile = bad;
            // Ensure vocab_size is large enough to not hit the [1,vocab_size] gate first.
            cfg.vocab_size = cfg.vocab_size.max(bad);
            let err = cfg.validate().expect_err(&format!(
                "vocab_tile {bad} (not a multiple of 128) MUST be rejected"
            ));
            assert!(
                err.contains("multiple of 128"),
                "rejection of vocab_tile {bad} should mention 128-alignment; got: {err}"
            );
        }
        // Sanity: 128-aligned values still pass (the existing [1,vocab_size] / 128-divisible gate).
        for good in [128u32, 256, 512, 1024, 2048] {
            let mut cfg = default_cfg();
            cfg.vocab_tile = good;
            cfg.vocab_size = cfg.vocab_size.max(good);
            assert!(
                cfg.validate().is_ok(),
                "vocab_tile {good} (multiple of 128) MUST be accepted"
            );
        }
    }

    #[test]
    fn test_validate_accepts_default() {
        assert!(default_cfg().validate().is_ok());
    }

    // ── Kernel name ──────────────────────────────────────────────────────

    #[test]
    fn test_kernel_name_encodes_shape() {
        let cfg = default_cfg();
        let name = cfg.kernel_name();
        assert!(name.contains("f32"), "name should include dtype: {name}");
        assert!(name.contains("v4096"), "name should include vocab: {name}");
        assert!(name.contains("h128"), "name should include hidden: {name}");
    }

    // ── shared_mem_bytes monotonicity ────────────────────────────────────

    #[test]
    fn test_shared_mem_bytes_monotone_with_vocab_tile() {
        let mut cfg = default_cfg();
        cfg.vocab_tile = 512;
        let sm512 = cfg.shared_mem_bytes();
        cfg.vocab_tile = 1024;
        let sm1024 = cfg.shared_mem_bytes();
        assert!(sm1024 > sm512, "larger vocab_tile must need more smem");
    }

    // ── PTX round-trip snapshot ──────────────────────────────────────────

    #[test]
    fn test_fwd_ptx_starts_with_version_target() {
        let cfg = default_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).expect("PTX must be valid UTF-8");
        assert!(
            ptx.starts_with(".version 7.0\n.target sm_80"),
            "PTX must start with .version 7.0 + .target sm_80, got: {}",
            &ptx[..50.min(ptx.len())]
        );
    }

    #[test]
    fn test_fwd_ptx_contains_kernel_name() {
        let cfg = default_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        assert!(
            ptx.contains(&cfg.kernel_name()),
            "PTX must contain the kernel name {}",
            cfg.kernel_name()
        );
    }

    #[test]
    fn test_fwd_ptx_contains_extern_shared() {
        let cfg = default_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        assert!(
            ptx.contains(".extern .shared"),
            "PTX must have .extern .shared scratch declaration"
        );
    }

    #[test]
    fn test_fwd_ptx_contains_skip_predicate() {
        let cfg = default_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        // The -100 skip predicate is emitted as setp.eq.s64 ... -100
        assert!(
            ptx.contains("setp.eq.s64") && ptx.contains("-100"),
            "PTX must contain setp.eq.s64 ... -100 skip predicate"
        );
    }

    #[test]
    fn test_bwd_ptx_contains_bwd_kernel_name() {
        let cfg = default_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        assert!(ptx.contains(&cfg.bwd_kernel_name()));
    }

    #[test]
    fn test_ptx_ascii_only() {
        // Unicode in PTX triggers CUDA_ERROR_INVALID_PTX under cudarc JIT.
        let cfg = default_cfg();
        let fwd = synthesize_fused_linear_ce_ptx(&cfg);
        let bwd = synthesize_fused_linear_ce_backward_ptx(&cfg);
        for byte in fwd.iter().chain(bwd.iter()) {
            assert!(*byte < 128, "non-ASCII byte 0x{byte:02x} found in PTX");
        }
    }
}
