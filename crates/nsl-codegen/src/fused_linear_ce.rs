//! G3 — Fused linear-CE with separator-skip (v1 single-CTA + v2 two-kernel large-vocab).
//!
//! Emits PTX kernels that implement:
//!   loss = cross_entropy(x @ W^T + bias, targets)
//! with a fast-path branch that skips the entire matmul + softmax for
//! positions labelled `-100`.
//!
//! ## Design — single-CTA vs large-vocab routing
//!
//! The v1 path runs one CTA per `(batch, sequence)` token row and serially
//! scans ALL vocabulary tiles inside that CTA (online log-sum-exp). For
//! vocab > 8192 this becomes the bottleneck (e.g. NSL production vocab=49152
//! with vocab_tile=128 → 384 serial tiles per CTA).
//!
//! Sprint 3 adds a **two-kernel cross-CTA reduction** path that activates
//! when `vocab_size > LARGE_VOCAB_THRESHOLD` (= 8192).  Routing is decided
//! by [`FusedLinearCEConfig::is_large_vocab`]; below the threshold the v1
//! single-CTA kernel is emitted **byte-identical** to pre-Sprint-3 (this
//! invariant is locked down by `tests/fused_linear_ce_large_vocab_numerical.rs::ptx_byte_identity_at_v4096`).
//!
//! ### Large-vocab path — Kernel A + Kernel B (Option 1)
//!
//! Picked over (Opt 2) atomic-CAS LSE (non-deterministic; CAS-loop blast
//! radius) and (Opt 3) cooperative-groups grid sync (requires
//! cuLaunchCooperativeKernel which NSL's launcher does not currently call).
//!
//! * **Kernel A — per-tile partials** (`*_fwd_large_partials_*`)
//!   * Grid: `(num_tiles, B*S, 1)` — one CTA per (vocab_tile, token_row).
//!   * Block: `(128, 1, 1)`.
//!   * Each CTA computes `x @ W^T[tile_rows] + bias[tile_rows]` for its row
//!     into shared memory (same dot-product loop as v1) then thread 0
//!     reduces to a `(tile_max, tile_sum_unscaled)` pair where
//!     `tile_sum_unscaled = sum_v exp(logit_v - tile_max)`.
//!   * Writes 2 floats to `partials[row*num_tiles + tile_id]` HBM.
//!   * `ignore_index` rows write `(0, 0)` to all their tiles' partials
//!     and Kernel B's same-row guard short-circuits to a zero loss.
//!
//! * **Kernel B — per-row finalize** (`*_fwd_large_finalize_*`)
//!   * Grid: `(B*S, 1, 1)`, block `(128, 1, 1)`.
//!   * Each CTA's thread 0 reads `partials[row, 0..num_tiles]`, runs the
//!     online-LSE rescaling formula across tiles, recomputes the single
//!     `logit_at_target = x[row] @ W[tgt] + bias[tgt]` (one dot-product —
//!     cheap relative to the per-row finalize cost), writes
//!     `loss_out[row] = lse - logit_at_target` and
//!     `lse_out[row] = global_max + log(global_sum)`.
//!
//! Math identity vs v1: the online-LSE reduction in Kernel B is exactly
//! the same per-tile rescaling formula v1 runs inside one CTA, just split
//! across CTA boundaries. Numerical equivalence at f32 holds to within
//! one ULP per tile per row.
//!
//! ### Synchronisation
//!
//! The host MUST issue an implicit or explicit barrier (`cuCtxSynchronize`
//! or stream sync) between Kernel A and Kernel B because Kernel B reads
//! the partials Kernel A writes. The runtime launcher does this via the
//! same-default-stream serialisation guarantee (both kernels launched on
//! `current_stream()` = NULL stream; CUDA serialises NULL-stream work).
//! `nsl_fused_linear_ce_forward_large` also calls `cuCtxSynchronize` at
//! the end so callers can read the results.
//!
//! ### Backward
//!
//! The v1 backward kernel does NOT need cross-CTA LSE — it reads the saved
//! `lse_out` from forward and recomputes softmax per `(tile, row)` chunk.
//! It is reused unchanged for both paths. The per-tile scatter via
//! `red.global.add.f32` to dW already handles large vocab correctly — only
//! the outer tile loop scales, and bigger vocabs just mean more iterations.
//!
//! ## Scope
//!
//! Both paths are research-grade scalar `fma.rn.f32`, F32 only. MMA tiling,
//! fp16/bf16 storage, and quantised-W paths are deferred.
//!
//! ## API
//!
//! ```rust,no_run
//! use nsl_codegen::fused_linear_ce::{FusedLinearCEConfig, Dtype};
//! let cfg = FusedLinearCEConfig {
//!     vocab_size: 4096, hidden_size: 128, seq_len: 64, batch_size: 2,
//!     vocab_tile: 1024, gpu_sm: 80, dtype: Dtype::F32,
//!     ignore_index: -100, max_vocab_v1: 262144,
//! };
//! cfg.validate().unwrap();
//! let fwd_ptx = nsl_codegen::fused_linear_ce::synthesize_fused_linear_ce_ptx(&cfg);
//! let bwd_ptx = nsl_codegen::fused_linear_ce::synthesize_fused_linear_ce_backward_ptx(&cfg);
//! ```

// ─── Config ─────────────────────────────────────────────────────────────────

/// Vocab sizes at or below this use the v1 single-CTA path **byte-identical**
/// to pre-Sprint-3.  Above this, the Sprint-3 two-kernel large-vocab path
/// activates (per-tile partials + per-row finalize).
///
/// Picked = 8192 because that was the v1 hard cap; using it as the routing
/// threshold means the legacy path stays bit-for-bit unchanged for every
/// shape that v1 used to accept, and the new path only lights up for shapes
/// v1 used to reject.
pub const LARGE_VOCAB_THRESHOLD: u32 = 8192;

/// Absolute hard ceiling on `vocab_size`. Above this, both paths refuse to
/// emit. Sized to cover all real-world LM vocabularies (NSL=49152,
/// GPT-3=50257, LLaMA-3=128256) with headroom.
pub const MAX_VOCAB_HARD_CEILING: u32 = 262_144;

/// Dtype selector for FusedLinearCE.
///
/// Both v1 single-CTA and Sprint-3 large-vocab paths support **F32**,
/// **F16** (Sprint v3-2), and **Bf16** (Sprint v4-1). F16 and Bf16 both use
/// the standard mixed-precision convention: loads/stores in 16-bit (`.f16`
/// or `.bf16`), accumulators (online-LSE max + sum, dot products) in
/// `.f32`. The large-vocab partials buffer stays f32 for numerical
/// robustness regardless of input dtype. The backward gradient outputs
/// (dx/dW/dbias) are also written in f32 — this matches PyTorch's
/// mixed-precision convention where master weights and the optimizer state
/// stay at f32 while only forward activations + weights halve their
/// footprint.
///
/// **Bf16** uses a `.b16` register family identical to F16 (storage class
/// is the same width), but the cvt instructions differ — `cvt.f32.bf16`
/// and `cvt.rn.bf16.f32` instead of the F16 variants. The `-INF` sentinel
/// for the tail-zero path is also different: bf16 `-INF = 0xFF80` (sign
/// bit, exponent all-ones, zero mantissa — bf16 has an 8-bit exponent
/// like f32) versus f16 `0xFC00`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    /// Human-readable tag used in kernel names and diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }

    /// Storage size in bytes of one element in HBM / SMEM. Used to size the
    /// online-softmax SMEM tile and to compute byte strides for HBM
    /// addressing.
    #[inline]
    pub fn bytes_per_elem(self) -> u32 {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::Bf16 => 2,
        }
    }

    /// PTX type-suffix for `ld.global.*` / `ld.shared.*` loads.
    ///
    /// For F16/Bf16 this matches `tag()` — the cvt mnemonics use the
    /// dtype-specific suffix (`cvt.f32.f16` vs `cvt.f32.bf16`); the bare
    /// `ld.global.b16` itself is dtype-agnostic and used by both 16-bit
    /// paths.
    #[inline]
    pub fn ptx_load_suffix(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }

    /// PTX type-suffix for `st.global.*` / `st.shared.*` stores. Same family
    /// as `ptx_load_suffix` today; kept distinct for future asymmetric paths.
    #[inline]
    pub fn ptx_store_suffix(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }

    /// The PTX register family used to *hold* one element in registers
    /// during HBM↔SMEM staging at this dtype.
    ///
    /// For `F32` this is `"f32"` (registers used directly for math). For
    /// `F16` and `Bf16` we move bytes through `.b16` registers and explicitly
    /// `cvt.f32.{f16,bf16}`/`cvt.rn.{f16,bf16}.f32` to/from `.f32` math
    /// registers — this matches the standard mixed-precision convention and
    /// keeps the algorithm in f32 precision end-to-end.
    #[inline]
    pub fn ptx_reg_family(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "b16",
            Dtype::Bf16 => "b16",
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
    /// Per-config hard cap on `vocab_size`. Defaults to [`LARGE_VOCAB_THRESHOLD`]
    /// (= 8192) to preserve pre-Sprint-3 rejection behaviour at the default
    /// config; callers that want the Sprint-3 large-vocab path raise this
    /// up to (and not above) [`MAX_VOCAB_HARD_CEILING`]. The field name is
    /// retained for ABI stability with v1 callers.
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
    /// Returns `true` when the Sprint-3 two-kernel large-vocab path will
    /// be selected by [`synthesize_fused_linear_ce_ptx`]. False = legacy
    /// v1 single-CTA path (byte-identical to pre-Sprint-3).
    #[inline]
    pub fn is_large_vocab(&self) -> bool {
        self.vocab_size > LARGE_VOCAB_THRESHOLD
    }

    /// Number of vocab tiles per token row (used to size the partials
    /// buffer in the large-vocab path).
    #[inline]
    pub fn num_vocab_tiles(&self) -> u32 {
        self.vocab_size.div_ceil(self.vocab_tile)
    }

    /// Validate configuration; returns `Err` with a descriptive message on
    /// any unsupported combination.
    pub fn validate(&self) -> Result<(), String> {
        // Absolute ceiling — neither path emits above this.
        if self.vocab_size > MAX_VOCAB_HARD_CEILING {
            return Err(format!(
                "fused_linear_ce: vocab_size {} exceeds hard ceiling {} — \
                 LM vocabularies above 256K are out of scope (NSL=49152, \
                 GPT-3=50257, LLaMA-3=128256 all fit comfortably below)",
                self.vocab_size, MAX_VOCAB_HARD_CEILING
            ));
        }
        if self.vocab_size > self.max_vocab_v1 {
            return Err(format!(
                "fused_linear_ce: vocab_size {} exceeds per-config cap \
                 max_vocab_v1={} — raise max_vocab_v1 up to {} to opt into the \
                 large-vocab (cross-CTA) path",
                self.vocab_size, self.max_vocab_v1, MAX_VOCAB_HARD_CEILING
            ));
        }
        if !self.hidden_size.is_multiple_of(32) {
            return Err(format!(
                "fused_linear_ce: hidden_size {} must be divisible by 32",
                self.hidden_size
            ));
        }
        // F32 + F16 (Sprint v3-2) + Bf16 (Sprint v4-1) all accepted.
        match self.dtype {
            Dtype::F32 | Dtype::F16 | Dtype::Bf16 => {}
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
    ///
    /// For `is_large_vocab() == false` this names the v1 single-CTA forward
    /// kernel. For the large-vocab path callers should use
    /// [`large_partials_kernel_name`](Self::large_partials_kernel_name) and
    /// [`large_finalize_kernel_name`](Self::large_finalize_kernel_name)
    /// instead.
    pub fn kernel_name(&self) -> String {
        format!(
            "nsl_fused_linear_ce_{}_v{}_h{}",
            self.dtype.tag(),
            self.vocab_size,
            self.hidden_size
        )
    }

    /// Name of the backward kernel (shared between v1 and large-vocab paths
    /// — per-tile gradient scatter scales naturally with vocab).
    pub fn bwd_kernel_name(&self) -> String {
        format!(
            "nsl_fused_linear_ce_backward_{}_v{}_h{}",
            self.dtype.tag(),
            self.vocab_size,
            self.hidden_size
        )
    }

    /// Name of Kernel A (per-tile partials) — large-vocab path only.
    /// Grid: `(num_tiles, B*S, 1)`.
    pub fn large_partials_kernel_name(&self) -> String {
        format!(
            "nsl_fused_linear_ce_fwd_large_partials_{}_v{}_h{}_t{}",
            self.dtype.tag(),
            self.vocab_size,
            self.hidden_size,
            self.vocab_tile,
        )
    }

    /// Name of Kernel B (per-row finalize) — large-vocab path only.
    /// Grid: `(B*S, 1, 1)`.
    pub fn large_finalize_kernel_name(&self) -> String {
        format!(
            "nsl_fused_linear_ce_fwd_large_finalize_{}_v{}_h{}_t{}",
            self.dtype.tag(),
            self.vocab_size,
            self.hidden_size,
            self.vocab_tile,
        )
    }

    /// Size in bytes of the cross-kernel partials buffer used by the
    /// large-vocab path: `(B*S) * num_tiles * 2 * sizeof(f32)`.
    ///
    /// Each `(row, tile)` slot stores two floats:
    /// `[tile_max, tile_sum_unscaled = sum_v exp(logit_v - tile_max)]`.
    pub fn large_partials_bytes(&self) -> u64 {
        let rows = self.batch_size as u64 * self.seq_len as u64;
        let n_tiles = self.num_vocab_tiles() as u64;
        rows * n_tiles * 2 * 4
    }

    /// Shared-memory budget per CTA: logits tile
    /// (`vocab_tile * dtype.bytes_per_elem()` bytes) + warp-shuffle scratch /
    /// LSE-max scalars (32 bytes pad).
    ///
    /// At `Dtype::F32` this is `vocab_tile * 4 + 32` (unchanged from
    /// pre-Sprint-v3-2; the byte-identity snapshot pins this). At
    /// `Dtype::F16` the per-element size halves, so the SMEM tile halves
    /// too — useful headroom on smem-constrained SMs but not yet exploited
    /// to raise the vocab_tile cap (see `validate()` deferred follow-on).
    pub fn shared_mem_bytes(&self) -> u32 {
        self.vocab_tile * self.dtype.bytes_per_elem() + 32
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn sm_tag(&self) -> u32 {
        // v1 targets sm_80+; fall back gracefully if caller passes sm_75.
        self.gpu_sm.max(80)
    }

    fn ptx_header(&self) -> String {
        // bf16 cvt mnemonics require PTX ISA 7.8+; bump to 8.0 for the Bf16
        // path. F32 and F16 stay at 7.0 to preserve byte-identity with the
        // Sprint v1 and Sprint v3-2 snapshots.
        let version = match self.dtype {
            Dtype::F32 | Dtype::F16 => "7.0",
            Dtype::Bf16 => "8.0",
        };
        format!(
            ".version {version}\n.target sm_{}\n.address_size 64\n",
            self.sm_tag()
        )
    }
}

// ─── Forward kernel synthesis ────────────────────────────────────────────────

/// Synthesise the forward PTX for the fused linear-CE kernel.
///
/// **Routing**: when `cfg.is_large_vocab()` is `false` (vocab_size ≤ 8192),
/// this returns the v1 single-CTA kernel **byte-identical** to pre-Sprint-3.
/// When `true`, it returns a single PTX module containing *both* Kernel A
/// (per-tile partials, name = `cfg.large_partials_kernel_name()`) and
/// Kernel B (per-row finalize, name = `cfg.large_finalize_kernel_name()`).
/// The caller (`nsl_fused_linear_ce_forward_large` host launcher) loads
/// the single module and launches the two kernels back-to-back.
///
/// ### v1 single-CTA path (vocab ≤ 8192)
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
    if cfg.is_large_vocab() {
        synthesize_large_vocab_forward_ptx(cfg)
    } else {
        match cfg.dtype {
            // F32 path is BYTE-IDENTICAL to pre-Sprint-v3-2 — calls the
            // untouched emitter. Sprint 3's v1 byte-identity snapshot pins
            // this contract.
            Dtype::F32 => emit_fwd_kernel(cfg).into_bytes(),
            Dtype::F16 => emit_fwd_kernel_f16(cfg).into_bytes(),
            Dtype::Bf16 => emit_fwd_kernel_bf16(cfg).into_bytes(),
        }
    }
}

/// Synthesise a single PTX module containing both Kernel A (per-tile
/// partials) and Kernel B (per-row finalize) of the Sprint-3 large-vocab
/// two-kernel forward path.
///
/// Both kernels share the same `.version` / `.target` header and use
/// separately-named `.extern .shared` allocations so the module compiles
/// as one ptxas TU.
pub fn synthesize_large_vocab_forward_ptx(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    let mut out = String::new();
    out.push_str(&cfg.ptx_header());
    out.push('\n');
    match cfg.dtype {
        // F32 path is BYTE-IDENTICAL to pre-Sprint-v3-2 large-vocab — gated by
        // `tests/fused_linear_ce_large_vocab_numerical.rs::ptx_byte_identity_at_v4096`
        // (uses the small-vocab routing) and the structural large-vocab ptxas
        // test at v=49152. Calls the untouched emitters.
        Dtype::F32 => {
            out.push_str(&emit_large_partials_kernel(cfg, /*emit_header=*/ false));
            out.push('\n');
            out.push_str(&emit_large_finalize_kernel(cfg, /*emit_header=*/ false));
        }
        Dtype::F16 => {
            out.push_str(&emit_large_partials_kernel_f16(cfg));
            out.push('\n');
            out.push_str(&emit_large_finalize_kernel_f16(cfg));
        }
        Dtype::Bf16 => {
            out.push_str(&emit_large_partials_kernel_bf16(cfg));
            out.push('\n');
            out.push_str(&emit_large_finalize_kernel_bf16(cfg));
        }
    }
    out.into_bytes()
}

/// Synthesise the backward PTX for the fused linear-CE kernel.
///
/// **Same kernel for both v1 and large-vocab forward paths.** The backward
/// algorithm reads the saved per-row `lse_out` from the forward pass and
/// per-(tile, row) recomputes `p_v = exp(logit_v - lse_row)`; it does NOT
/// depend on whether the forward used the single-CTA or two-kernel reduce
/// — it only needs the correct global LSE, which Kernel B writes. Grid
/// stays `(B*S, 1, 1)`; each CTA serially scans `num_tiles` vocab tiles.
///
/// Trade-off: at large vocab the backward becomes the bottleneck (linear in
/// num_tiles per row). Parallelising it to a `(num_tiles, B*S, 1)` grid is
/// a future optimisation — correctness is unaffected today because the
/// `red.global.add.f32` scatters into dW and dx are commutative-associative.
///
/// Grid/block: same as v1 forward.
/// Recomputes logits (no logits buffer saved), computes
/// `dlogits_v = (softmax_v - 1{v==target}) * grad_output / num_valid`,
/// then scatters `dx += dlogits_v * W[v, :]` and
/// `dW[v, :] += dlogits_v * x[row, :]` via `red.global.add.f32`.
pub fn synthesize_fused_linear_ce_backward_ptx(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    match cfg.dtype {
        // F32 path is BYTE-IDENTICAL to pre-Sprint-v3-2 — pinned by
        // `tests/fused_linear_ce_v1_byte_identity.rs::v1_backward_*`.
        Dtype::F32 => emit_bwd_kernel(cfg).into_bytes(),
        Dtype::F16 => emit_bwd_kernel_f16(cfg).into_bytes(),
        Dtype::Bf16 => emit_bwd_kernel_bf16(cfg).into_bytes(),
    }
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

// ── F16 forward kernel ───────────────────────────────────────────────────────
//
// Mixed-precision convention (Sprint v3-2):
//   * HBM loads of x, W, bias are `ld.global.b16` → `cvt.f32.f16` into f32
//     math registers (the dot-product accumulators).
//   * SMEM logit tile is stored as `.b16` (2 bytes/elem) — the per-tile
//     online-LSE reduction loads `.b16`, converts to f32, then folds into
//     the f32 max + sum registers. This halves the SMEM footprint vs F32.
//   * +INF / -INF sentinels and the log2e + ln2 constants live in f32
//     registers; the online-LSE algorithm is bit-for-bit the same as F32.
//   * HBM outputs `loss_out` and `lse_out` are written as `.f32` (per
//     convention; downstream backward consumes the same f32 lse).
//
// The kernel layout, loop structure, sync points, and SMEM partitioning
// mirror `emit_fwd_kernel` exactly — only the dtype of HBM/SMEM staging
// changes. Keeping the structure parallel preserves the option of merging
// the two emitters in a future refactor (deferred for byte-identity safety).
fn emit_fwd_kernel_f16(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = vocab.div_ceil(vtile);
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();
    // F16 SMEM: 2 bytes/elem instead of 4. logit_at_target scratch slot lives
    // immediately after the tile and ALSO uses 2 bytes; the +32 padding in
    // shared_mem_bytes() leaves ample room.
    let elem_bytes = cfg.dtype.bytes_per_elem(); // = 2 for F16
    let lat_offset = vtile * elem_bytes;

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    lines.push(cfg.ptx_header());
    // `.align 2` because the SMEM is half-precision; ptxas auto-aligns to 4
    // for the +INF sentinel store anyway — but the declared align matches
    // the dtype.
    lines.push(format!(".extern .shared .align 2 .b8 smem_scratch[{smem_bytes}];"));
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
    lines.push(p("\t.reg .b16 %h0, %h1, %h2;"));
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

    // Thread 0 initialises logit_at_target smem slot to -INF (as fp16
    // bit-pattern 0xFBFF = -65504, the most negative finite fp16; the
    // logit_at_target slot is only consumed if the target was found, so a
    // strict -INF bit-pattern is not required — but using fp16 -INF
    // (0xFC00) keeps the semantics matched).
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra INIT_DONE;"));
    lines.push(p("\tmov.u64 %rd6, smem_scratch;"));
    lines.push(format!("\tadd.u64 %rd6, %rd6, {lat_offset};"));
    lines.push(p("\tmov.b16 %h0, 0xFC00; // fp16 -INF sentinel"));
    lines.push(p("\tst.shared.b16 [%rd6], %h0;"));
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

    // x_row_base = x + row * H * 2 (fp16: 2 bytes/elem).
    lines.push(format!(
        "\t// x_row_base = x + row_idx * {hidden} * {elem_bytes}"
    ));
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(format!("\tmov.u32 %r2, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd9, %r2;"));
    lines.push(p("\tmul.lo.u64 %rd8, %rd8, %rd9;"));
    lines.push(p("\tshl.b64 %rd8, %rd8, 1; // *2 for fp16"));
    lines.push(p("\tadd.u64 %rd8, %rd0, %rd8; // %rd8 = x_row_base"));
    lines.push(String::new());

    // Init online-softmax accumulators (f32).
    lines.push(p("\tmov.f32 %fmax, 0f80800000; // -INF (f32)"));
    lines.push(p("\tmov.f32 %fsum, 0f00000000;"));
    lines.push(String::new());

    // Outer tile loop.
    lines.push(p("\tmov.u32 %r3, 0; // tile_idx"));
    lines.push(p("TILE_LOOP:"));
    lines.push(format!("\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base"));
    lines.push(String::new());

    lines.push(p("\t\tmov.u32 %r5, 0; // sub-tile counter"));
    lines.push(p("\t\tINNER_LOOP:"));
    lines.push(p("\t\t\tmul.lo.u32 %r6, %r5, 128;"));
    lines.push(p("\t\t\tadd.u32 %r6, %r6, %r1;"));
    lines.push(p("\t\t\tadd.u32 %r6, %r6, %r4; // v_idx"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r6, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra INNER_SKIP;"));
    lines.push(String::new());

    // W_row_base = W + v_idx * H * 2.
    lines.push(format!(
        "\t\t\t// W_row_base = W + v_idx * {hidden} * {elem_bytes}"
    ));
    lines.push(p("\t\t\tcvt.u64.u32 %rd10, %r6;"));
    lines.push(format!("\t\t\tmov.u32 %r7, {hidden};"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd11, %r7;"));
    lines.push(p("\t\t\tmul.lo.u64 %rd10, %rd10, %rd11;"));
    lines.push(p("\t\t\tshl.b64 %rd10, %rd10, 1; // *2 for fp16"));
    lines.push(p("\t\t\tadd.u64 %rd10, %rd1, %rd10; // %rd10 = W_row_base"));
    lines.push(String::new());

    // Dot-product loop — load .b16, cvt to .f32, fma in f32.
    lines.push(p("\t\t\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\t\t\tmov.u32 %r8, 0; // h"));
    lines.push(p("\t\t\tDOT_LOOP:"));
    lines.push(p("\t\t\t\tcvt.u64.u32 %rd12, %r8;"));
    lines.push(p("\t\t\t\tshl.b64 %rd12, %rd12, 1; // h * 2"));
    lines.push(p("\t\t\t\tadd.u64 %rd13, %rd8, %rd12;"));
    lines.push(p("\t\t\t\tld.global.b16 %h1, [%rd13];"));
    lines.push(p("\t\t\t\tcvt.f32.f16 %fa, %h1;"));
    lines.push(p("\t\t\t\tadd.u64 %rd14, %rd10, %rd12;"));
    lines.push(p("\t\t\t\tld.global.b16 %h1, [%rd14];"));
    lines.push(p("\t\t\t\tcvt.f32.f16 %fb, %h1;"));
    lines.push(p("\t\t\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\t\t\tadd.u32 %r8, %r8, 1;"));
    lines.push(format!("\t\t\t\tsetp.lt.u32 %pv, %r8, {hidden};"));
    lines.push(p("\t\t\t\t@%pv bra DOT_LOOP;"));
    lines.push(String::new());

    // Add bias[v_idx].
    lines.push(p("\t\t\t// facc += bias[v_idx] (fp16 HBM)"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd15, %r6;"));
    lines.push(p("\t\t\tshl.b64 %rd15, %rd15, 1; // *2"));
    lines.push(p("\t\t\tadd.u64 %rd15, %rd2, %rd15;"));
    lines.push(p("\t\t\tld.global.b16 %h1, [%rd15];"));
    lines.push(p("\t\t\tcvt.f32.f16 %ftmp, %h1;"));
    lines.push(p("\t\t\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(String::new());

    // Store logit to smem tile (cvt back to fp16; tile is .b16 stride 2).
    lines.push(p("\t\t\t// Store logit to smem_scratch[(r5*128+tid)*2] as fp16"));
    lines.push(p("\t\t\tmul.lo.u32 %r9, %r5, 128;"));
    lines.push(p("\t\t\tadd.u32 %r9, %r9, %r1;"));
    lines.push(p("\t\t\tshl.b32 %r9, %r9, 1; // *2"));
    lines.push(p("\t\t\tmov.u64 %rd16, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd17, %r9;"));
    lines.push(p("\t\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\t\tcvt.rn.f16.f32 %h2, %facc;"));
    lines.push(p("\t\t\tst.shared.b16 [%rd16], %h2;"));
    lines.push(String::new());

    // If v_idx == target, record logit_at_target in smem (also fp16).
    lines.push(p("\t\t\t// If v_idx == target, record logit_at_target (fp16)"));
    lines.push(p("\t\t\tcvt.s64.u32 %rd18, %r6;"));
    lines.push(p("\t\t\tsetp.eq.s64 %ptgt, %rd18, %tgt64;"));
    lines.push(p("\t\t\t@!%ptgt bra NOT_TARGET;"));
    lines.push(p("\t\t\tmov.u64 %rd19, smem_scratch;"));
    lines.push(format!("\t\t\tadd.u64 %rd19, %rd19, {lat_offset};"));
    lines.push(p("\t\t\tst.shared.b16 [%rd19], %h2;"));
    lines.push(p("\t\t\tNOT_TARGET:"));
    lines.push(String::new());

    lines.push(p("\t\t\tINNER_SKIP:"));
    lines.push(p("\t\t\tadd.u32 %r5, %r5, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r5, {vtile_per_thread};"));
    lines.push(p("\t\t\t@%pv bra INNER_LOOP;"));
    lines.push(String::new());

    // Sync: all logits in smem.
    lines.push(p("\t\tbar.sync 0;"));
    lines.push(String::new());

    // Thread 0 reduces smem tile.
    lines.push(p("\t\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t\t@!%pth0 bra TILE_REDUCE_DONE;"));
    lines.push(String::new());

    // Step 1: find tile_max (read fp16 → cvt to f32, fold into f32 max).
    lines.push(p("\t\tmov.f32 %ftmax, 0f80800000;"));
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tSMEM_MAX_LOOP:"));
    lines.push(p("\t\t\tadd.u32 %r11, %r4, %r10; // v_base + i"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r11, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra SMEM_MAX_DONE;"));
    lines.push(p("\t\t\tshl.b32 %r12, %r10, 1; // *2"));
    lines.push(p("\t\t\tmov.u64 %rd20, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd21, %r12;"));
    lines.push(p("\t\t\tadd.u64 %rd20, %rd20, %rd21;"));
    lines.push(p("\t\t\tld.shared.b16 %h1, [%rd20];"));
    lines.push(p("\t\t\tcvt.f32.f16 %ftmp, %h1;"));
    lines.push(p("\t\t\tmax.f32 %ftmax, %ftmax, %ftmp;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {vtile};"));
    lines.push(p("\t\t\t@%pv bra SMEM_MAX_LOOP;"));
    lines.push(p("\t\tSMEM_MAX_DONE:"));
    lines.push(String::new());

    // Online-softmax rescale.
    lines.push(p("\t\t// Online-softmax update"));
    lines.push(p("\t\tmax.f32 %flog, %fmax, %ftmax; // new_max"));
    lines.push(p("\t\tsub.f32 %fmax, %fmax, %flog;"));
    lines.push(p("\t\tmul.f32 %fmax, %fmax, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %fmax, %fmax;"));
    lines.push(p("\t\tmul.f32 %fsum, %fsum, %fmax;"));
    lines.push(p("\t\tmov.f32 %fmax, %flog;"));
    lines.push(String::new());

    // Tile sum (fp16 load → cvt to f32 → fold into f32 sum).
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tSMEM_SUM_LOOP:"));
    lines.push(p("\t\t\tadd.u32 %r11, %r4, %r10;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r11, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra SMEM_SUM_DONE;"));
    lines.push(p("\t\t\tshl.b32 %r12, %r10, 1; // *2"));
    lines.push(p("\t\t\tmov.u64 %rd20, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd21, %r12;"));
    lines.push(p("\t\t\tadd.u64 %rd20, %rd20, %rd21;"));
    lines.push(p("\t\t\tld.shared.b16 %h1, [%rd20];"));
    lines.push(p("\t\t\tcvt.f32.f16 %ftmp, %h1;"));
    lines.push(p("\t\t\tsub.f32 %ftmp, %ftmp, %fmax;"));
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

    lines.push(p("\t\tadd.u32 %r3, %r3, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r3, {n_tiles};"));
    lines.push(p("\t\t@%pv bra TILE_LOOP;"));
    lines.push(String::new());

    // Finalize: thread 0 writes loss + lse as f32.
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra WRITE_DONE;"));
    lines.push(String::new());

    lines.push(p("\tlg2.approx.f32 %flse, %fsum;"));
    lines.push(p("\tmul.f32 %flse, %flse, %fln2;"));
    lines.push(p("\tadd.f32 %flse, %flse, %fmax;"));
    lines.push(String::new());

    // logit_at_target from smem (read fp16, cvt to f32).
    lines.push(p("\tmov.u64 %rd22, smem_scratch;"));
    lines.push(format!("\tadd.u64 %rd22, %rd22, {lat_offset};"));
    lines.push(p("\tld.shared.b16 %h1, [%rd22];"));
    lines.push(p("\tcvt.f32.f16 %flog, %h1;"));
    lines.push(String::new());

    lines.push(p("\tsub.f32 %floss, %flse, %flog;"));
    lines.push(String::new());

    // Write outputs as f32 (loss + lse stay f32 regardless of dtype).
    lines.push(p("\tcvt.u64.u32 %rd23, %r0;"));
    lines.push(p("\tshl.b64 %rd23, %rd23, 2;"));
    lines.push(p("\tadd.u64 %rd24, %rd4, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], %floss;"));
    lines.push(p("\tadd.u64 %rd24, %rd5, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], %flse;"));
    lines.push(p("\tbra WRITE_DONE;"));
    lines.push(String::new());

    // Skip label: write zeros (f32).
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

// ─── PTX emission — large-vocab path (Sprint 3) ──────────────────────────────

/// Per-tile partials kernel (Kernel A) — see module-level Design block.
///
/// Grid: `(num_tiles, B*S, 1)`. Block: `(128, 1, 1)`.
///
/// Each CTA:
///   1. row_idx = ctaid.y, tile_idx = ctaid.x.
///   2. Loads target = targets[row_idx]; if == ignore_index, writes (0, 0)
///      to its partials slot and returns (kernel B's same-row guard zeros
///      the row's loss).
///   3. Fills smem[0..vocab_tile*4] with logits = x[row] @ W[tile_rows]^T
///      + bias[tile_rows] using the same 128-thread inner fill v1 uses.
///   4. bar.sync; then thread 0 reduces smem to (tile_max, tile_sum_unscaled)
///      where tile_sum_unscaled = sum_v exp(logit_v - tile_max), and stores
///      both floats to partials[row*num_tiles + tile_idx, 0..2].
fn emit_large_partials_kernel(cfg: &FusedLinearCEConfig, emit_header: bool) -> String {
    let name = cfg.large_partials_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = cfg.num_vocab_tiles();
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    if emit_header {
        lines.push(cfg.ptx_header());
    }
    // Each kernel declares its own .extern .shared. Driver allocates per-launch.
    lines.push(format!(
        ".extern .shared .align 4 .b8 smem_partials_{vocab}[{smem_bytes}];"
    ));
    lines.push(String::new());

    // Signature: x, W, bias, targets, partials, B, S, V, H, num_tiles.
    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_partials,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H,"));
    lines.push(p("\t.param .u32 param_num_tiles"));
    lines.push(p(") {"));

    lines.push(p("\t.reg .u64 %rd<30>;"));
    lines.push(p("\t.reg .u32 %r<24>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .f32 %facc, %fa, %fb, %ftmp;"));
    lines.push(p("\t.reg .f32 %ftmax, %ftsum, %flog2e;"));
    lines.push(p("\t.reg .pred %pskip, %pv, %pth0;"));
    lines.push(String::new());

    // Load params.
    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_partials];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.y;   // row_idx"));
    lines.push(p("\tmov.u32 %r1, %tid.x;     // tid"));
    lines.push(p("\tmov.u32 %r2, %ctaid.x;   // tile_idx"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B; // log2(e)"));
    lines.push(String::new());

    // partials_slot_ptr = partials + (row*num_tiles + tile_idx) * 2 * 4
    lines.push(format!("\t// partials_slot = partials + (row*{n_tiles}+tile) * 8"));
    lines.push(p("\tcvt.u64.u32 %rd5, %r0;"));
    lines.push(format!("\tmov.u32 %r3, {n_tiles};"));
    lines.push(p("\tcvt.u64.u32 %rd6, %r3;"));
    lines.push(p("\tmul.lo.u64 %rd5, %rd5, %rd6;"));
    lines.push(p("\tcvt.u64.u32 %rd7, %r2;"));
    lines.push(p("\tadd.u64 %rd5, %rd5, %rd7;"));
    lines.push(p("\tshl.b64 %rd5, %rd5, 3; // *8"));
    lines.push(p("\tadd.u64 %rd5, %rd4, %rd5; // %rd5 = partials_slot_ptr"));
    lines.push(String::new());

    // Load target.
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd8, %rd8, 8;"));
    lines.push(p("\tadd.u64 %rd8, %rd3, %rd8;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd8];"));
    lines.push(String::new());

    // Skip path: thread 0 writes (0, 0) to partials slot, return.
    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@!%pskip bra LP_NOTSKIP;"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LP_DONE;"));
    lines.push(p("\tst.global.f32 [%rd5], 0f00000000;"));
    lines.push(p("\tst.global.f32 [%rd5+4], 0f00000000;"));
    lines.push(p("\tbra LP_DONE;"));
    lines.push(p("LP_NOTSKIP:"));
    lines.push(String::new());

    // x_row_base = x + row * H * 4.
    lines.push(p("\tcvt.u64.u32 %rd9, %r0;"));
    lines.push(format!("\tmov.u32 %r4, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd10, %r4;"));
    lines.push(p("\tmul.lo.u64 %rd9, %rd9, %rd10;"));
    lines.push(p("\tshl.b64 %rd9, %rd9, 2;"));
    lines.push(p("\tadd.u64 %rd9, %rd0, %rd9; // %rd9 = x_row_base"));
    lines.push(String::new());

    // v_base = tile_idx * vtile.
    lines.push(format!("\tmul.lo.u32 %r5, %r2, {vtile}; // v_base"));
    lines.push(String::new());

    // Inner fill loop: each thread writes vtile_per_thread entries in stride-128.
    lines.push(p("\tmov.u32 %r6, 0; // sub-iter counter"));
    lines.push(p("LP_INNER:"));
    lines.push(p("\t\tmul.lo.u32 %r7, %r6, 128;"));
    lines.push(p("\t\tadd.u32 %r7, %r7, %r1;       // intra-tile slot"));
    lines.push(p("\t\tadd.u32 %r8, %r7, %r5;       // v_idx = v_base + slot"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r8, {vocab};"));
    lines.push(p("\t\t@!%pv bra LP_INNER_TAIL_ZERO;"));
    lines.push(String::new());

    // W_row_base = W + v_idx * H * 4.
    lines.push(p("\t\tcvt.u64.u32 %rd11, %r8;"));
    lines.push(format!("\t\tmov.u32 %r9, {hidden};"));
    lines.push(p("\t\tcvt.u64.u32 %rd12, %r9;"));
    lines.push(p("\t\tmul.lo.u64 %rd11, %rd11, %rd12;"));
    lines.push(p("\t\tshl.b64 %rd11, %rd11, 2;"));
    lines.push(p("\t\tadd.u64 %rd11, %rd1, %rd11; // %rd11 = W_row_base"));
    lines.push(String::new());

    // Dot product over H.
    lines.push(p("\t\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\t\tmov.u32 %r10, 0; // h"));
    lines.push(p("\t\tLP_DOT:"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd13, %r10;"));
    lines.push(p("\t\t\tshl.b64 %rd13, %rd13, 2;"));
    lines.push(p("\t\t\tadd.u64 %rd14, %rd9, %rd13;"));
    lines.push(p("\t\t\tld.global.f32 %fa, [%rd14];"));
    lines.push(p("\t\t\tadd.u64 %rd14, %rd11, %rd13;"));
    lines.push(p("\t\t\tld.global.f32 %fb, [%rd14];"));
    lines.push(p("\t\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {hidden};"));
    lines.push(p("\t\t\t@%pv bra LP_DOT;"));
    lines.push(String::new());

    // facc += bias[v_idx].
    lines.push(p("\t\tcvt.u64.u32 %rd15, %r8;"));
    lines.push(p("\t\tshl.b64 %rd15, %rd15, 2;"));
    lines.push(p("\t\tadd.u64 %rd15, %rd2, %rd15;"));
    lines.push(p("\t\tld.global.f32 %ftmp, [%rd15];"));
    lines.push(p("\t\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(p("\t\tbra LP_INNER_STORE;"));
    lines.push(String::new());

    // Tail-zero (v_idx out of bounds): store -INF so it doesn't perturb max/sum.
    lines.push(p("LP_INNER_TAIL_ZERO:"));
    lines.push(p("\t\tmov.f32 %facc, 0f80800000; // -INF"));
    lines.push(String::new());

    // Store logit to smem[slot*4].
    lines.push(p("LP_INNER_STORE:"));
    lines.push(p("\t\tshl.b32 %r11, %r7, 2;"));
    lines.push(format!("\t\tmov.u64 %rd16, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r11;"));
    lines.push(p("\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\tst.shared.f32 [%rd16], %facc;"));
    lines.push(String::new());

    lines.push(p("\t\tadd.u32 %r6, %r6, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r6, {vtile_per_thread};"));
    lines.push(p("\t\t@%pv bra LP_INNER;"));
    lines.push(String::new());

    // bar.sync, then thread 0 reduces smem to (tile_max, tile_sum_unscaled).
    lines.push(p("\tbar.sync 0;"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LP_DONE;"));
    lines.push(String::new());

    // Pass 1: find tile_max over smem[0..vtile].
    lines.push(p("\tmov.f32 %ftmax, 0f80800000;"));
    lines.push(p("\tmov.u32 %r12, 0;"));
    lines.push(p("LP_RED_MAX:"));
    lines.push(p("\t\tshl.b32 %r13, %r12, 2;"));
    lines.push(format!("\t\tmov.u64 %rd18, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd19, %r13;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd18, %rd19;"));
    lines.push(p("\t\tld.shared.f32 %ftmp, [%rd18];"));
    lines.push(p("\t\tmax.f32 %ftmax, %ftmax, %ftmp;"));
    lines.push(p("\t\tadd.u32 %r12, %r12, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r12, {vtile};"));
    lines.push(p("\t\t@%pv bra LP_RED_MAX;"));
    lines.push(String::new());

    // Pass 2: compute tile_sum_unscaled = sum_v exp(logit_v - tile_max).
    lines.push(p("\tmov.f32 %ftsum, 0f00000000;"));
    lines.push(p("\tmov.u32 %r12, 0;"));
    lines.push(p("LP_RED_SUM:"));
    lines.push(p("\t\tshl.b32 %r13, %r12, 2;"));
    lines.push(format!("\t\tmov.u64 %rd18, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd19, %r13;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd18, %rd19;"));
    lines.push(p("\t\tld.shared.f32 %ftmp, [%rd18];"));
    lines.push(p("\t\tsub.f32 %ftmp, %ftmp, %ftmax;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tadd.f32 %ftsum, %ftsum, %ftmp;"));
    lines.push(p("\t\tadd.u32 %r12, %r12, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r12, {vtile};"));
    lines.push(p("\t\t@%pv bra LP_RED_SUM;"));
    lines.push(String::new());

    // Store (tile_max, tile_sum_unscaled) to partials.
    lines.push(p("\tst.global.f32 [%rd5],   %ftmax;"));
    lines.push(p("\tst.global.f32 [%rd5+4], %ftsum;"));
    lines.push(String::new());

    lines.push(p("LP_DONE:"));
    lines.push(p("\tret;"));
    lines.push(p("}"));

    lines.join("\n")
}

/// Per-row finalize kernel (Kernel B) — see module-level Design block.
///
/// Grid: `(B*S, 1, 1)`. Block: `(128, 1, 1)`. Uses ONE thread (tid 0) to do
/// the cross-tile reduce — `num_tiles` is at most a few hundred so the
/// serial reduce per row is cheap; the gain is parallelism across rows
/// (the loop that was inside one CTA is now ONE iteration per row, with
/// `num_tiles` independent partials already computed by Kernel A).
///
/// Each CTA:
///   1. row_idx = ctaid.x.
///   2. Loads targets[row]; if ignore, writes loss=0/lse=0 and returns.
///   3. Thread 0 reads partials[row, 0..num_tiles], runs online-LSE rescaling
///      to compute (global_max, global_sum).
///   4. Thread 0 recomputes logit_at_target = x[row] @ W[tgt] + bias[tgt]
///      (one dot product of length H — cheap).
///   5. Writes loss_out[row] = -(logit_at_target - global_max - log(global_sum))
///      and lse_out[row] = global_max + log(global_sum).
fn emit_large_finalize_kernel(cfg: &FusedLinearCEConfig, emit_header: bool) -> String {
    let name = cfg.large_finalize_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let n_tiles = cfg.num_vocab_tiles();
    let ignore = cfg.ignore_index;

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    if emit_header {
        lines.push(cfg.ptx_header());
    }
    lines.push(String::new());

    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_partials,"));
    lines.push(p("\t.param .u64 param_loss_out,"));
    lines.push(p("\t.param .u64 param_lse_out,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H,"));
    lines.push(p("\t.param .u32 param_num_tiles"));
    lines.push(p(") {"));

    lines.push(p("\t.reg .u64 %rd<24>;"));
    lines.push(p("\t.reg .u32 %r<16>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .f32 %fmax, %fsum, %ftmax, %ftsum, %fnew_max;"));
    lines.push(p("\t.reg .f32 %ftmp, %fa, %fb, %facc, %flog, %flse, %floss;"));
    lines.push(p("\t.reg .f32 %flog2e, %fln2;"));
    lines.push(p("\t.reg .pred %pskip, %pth0, %pv;"));
    lines.push(String::new());

    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_partials];"));
    lines.push(p("\tld.param.u64 %rd5, [param_loss_out];"));
    lines.push(p("\tld.param.u64 %rd6, [param_lse_out];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.x;   // row_idx"));
    lines.push(p("\tmov.u32 %r1, %tid.x;"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B;"));
    lines.push(p("\tmov.f32 %fln2,   0f3F317218;"));
    lines.push(String::new());

    // Only thread 0 does the work; other threads idle then exit.
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LF_DONE;"));
    lines.push(String::new());

    // Load target.
    lines.push(p("\tcvt.u64.u32 %rd7, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd7, %rd7, 8;"));
    lines.push(p("\tadd.u64 %rd7, %rd3, %rd7;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd7];"));
    lines.push(String::new());

    // loss_out_ptr / lse_out_ptr for this row.
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(p("\tshl.b64 %rd8, %rd8, 2;"));
    lines.push(p("\tadd.u64 %rd9, %rd5, %rd8;  // loss_out[row]"));
    lines.push(p("\tadd.u64 %rd10, %rd6, %rd8; // lse_out[row]"));
    lines.push(String::new());

    // Skip path: write 0, 0.
    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@!%pskip bra LF_REDUCE;"));
    lines.push(p("\tst.global.f32 [%rd9],  0f00000000;"));
    lines.push(p("\tst.global.f32 [%rd10], 0f00000000;"));
    lines.push(p("\tbra LF_DONE;"));
    lines.push(p("LF_REDUCE:"));
    lines.push(String::new());

    // partials_row_base = partials + row * num_tiles * 8.
    lines.push(p("\tcvt.u64.u32 %rd11, %r0;"));
    lines.push(format!("\tmov.u32 %r2, {n_tiles};"));
    lines.push(p("\tcvt.u64.u32 %rd12, %r2;"));
    lines.push(p("\tmul.lo.u64 %rd11, %rd11, %rd12;"));
    lines.push(p("\tshl.b64 %rd11, %rd11, 3; // *8"));
    lines.push(p("\tadd.u64 %rd11, %rd4, %rd11; // partials_row_base"));
    lines.push(String::new());

    // Online-LSE reduce across tiles.
    lines.push(p("\tmov.f32 %fmax, 0f80800000; // -INF"));
    lines.push(p("\tmov.f32 %fsum, 0f00000000;"));
    lines.push(p("\tmov.u32 %r3, 0; // tile_idx"));
    lines.push(p("LF_LOOP:"));
    lines.push(p("\t\tcvt.u64.u32 %rd13, %r3;"));
    lines.push(p("\t\tshl.b64 %rd13, %rd13, 3;"));
    lines.push(p("\t\tadd.u64 %rd13, %rd11, %rd13;"));
    lines.push(p("\t\tld.global.f32 %ftmax, [%rd13];"));
    lines.push(p("\t\tld.global.f32 %ftsum, [%rd13+4];"));
    lines.push(String::new());

    // new_max = max(fmax, ftmax).
    lines.push(p("\t\tmax.f32 %fnew_max, %fmax, %ftmax;"));
    // Rescale running sum: fsum *= exp(fmax - new_max).
    lines.push(p("\t\tsub.f32 %ftmp, %fmax, %fnew_max;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tmul.f32 %fsum, %fsum, %ftmp;"));
    // Rescale tile sum: ftsum *= exp(ftmax - new_max).
    lines.push(p("\t\tsub.f32 %ftmp, %ftmax, %fnew_max;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tmul.f32 %ftsum, %ftsum, %ftmp;"));
    // Accumulate.
    lines.push(p("\t\tadd.f32 %fsum, %fsum, %ftsum;"));
    lines.push(p("\t\tmov.f32 %fmax, %fnew_max;"));
    lines.push(String::new());

    lines.push(p("\t\tadd.u32 %r3, %r3, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r3, {n_tiles};"));
    lines.push(p("\t\t@%pv bra LF_LOOP;"));
    lines.push(String::new());

    // lse = log(sum) + max.
    lines.push(p("\tlg2.approx.f32 %flse, %fsum;"));
    lines.push(p("\tmul.f32 %flse, %flse, %fln2;"));
    lines.push(p("\tadd.f32 %flse, %flse, %fmax;"));
    lines.push(String::new());

    // logit_at_target: recompute single dot product x[row] @ W[tgt] + bias[tgt].
    // x_row_base = x + row*H*4.
    lines.push(p("\tcvt.u64.u32 %rd14, %r0;"));
    lines.push(format!("\tmov.u32 %r4, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd15, %r4;"));
    lines.push(p("\tmul.lo.u64 %rd14, %rd14, %rd15;"));
    lines.push(p("\tshl.b64 %rd14, %rd14, 2;"));
    lines.push(p("\tadd.u64 %rd14, %rd0, %rd14; // x_row_base"));
    // W_tgt_base = W + tgt * H * 4 (tgt is in %tgt64 as s64; mul is u64 since vocab>0).
    lines.push(p("\tmul.lo.s64 %rd16, %tgt64, %rd15;"));
    lines.push(p("\tshl.b64 %rd16, %rd16, 2;"));
    lines.push(p("\tadd.u64 %rd16, %rd1, %rd16; // W_tgt_base"));
    lines.push(String::new());

    // Dot.
    lines.push(p("\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\tmov.u32 %r5, 0;"));
    lines.push(p("LF_DOT:"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r5;"));
    lines.push(p("\t\tshl.b64 %rd17, %rd17, 2;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd14, %rd17;"));
    lines.push(p("\t\tld.global.f32 %fa, [%rd18];"));
    lines.push(p("\t\tadd.u64 %rd18, %rd16, %rd17;"));
    lines.push(p("\t\tld.global.f32 %fb, [%rd18];"));
    lines.push(p("\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\tadd.u32 %r5, %r5, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r5, {hidden};"));
    lines.push(p("\t\t@%pv bra LF_DOT;"));
    lines.push(String::new());

    // Add bias[tgt].
    lines.push(p("\tmul.lo.s64 %rd19, %tgt64, 4;"));
    lines.push(p("\tadd.u64 %rd19, %rd2, %rd19;"));
    lines.push(p("\tld.global.f32 %ftmp, [%rd19];"));
    lines.push(p("\tadd.f32 %facc, %facc, %ftmp; // logit_at_target"));
    lines.push(String::new());

    // loss = lse - logit_at_target.
    lines.push(p("\tsub.f32 %floss, %flse, %facc;"));
    lines.push(p("\tst.global.f32 [%rd9],  %floss;"));
    lines.push(p("\tst.global.f32 [%rd10], %flse;"));
    lines.push(String::new());

    lines.push(p("LF_DONE:"));
    // Suppress unused-V warning: keep V param consumed.
    let _ = vocab;
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

// ── F16 large-vocab kernels (Sprint v3-2) ────────────────────────────────────
//
// Mixed-precision convention for the two-kernel large-vocab path:
//   * Kernel A (per-tile partials) loads x / W / bias as `.b16` → `cvt.f32.f16`,
//     accumulates dot products in f32, stages SMEM logits as `.b16`, and
//     writes the cross-CTA partials buffer as f32 — the partials buffer
//     stays f32 for cross-CTA numerical robustness (the online-LSE rescale
//     in Kernel B compounds across tiles; fp16 partials would visibly
//     degrade lse accuracy at vocab=49152+).
//   * Kernel B (per-row finalize) reads f32 partials AND ALSO recomputes
//     `logit_at_target = x[row] @ W[tgt] + bias[tgt]` — at dtype=F16 that
//     recompute uses the same fp16 HBM staging convention as Kernel A.
//     The final `loss_out` / `lse_out` writes stay f32 (same as v1).
fn emit_large_partials_kernel_f16(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.large_partials_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = cfg.num_vocab_tiles();
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    // `.align 2` because SMEM is fp16.
    lines.push(format!(
        ".extern .shared .align 2 .b8 smem_partials_{vocab}[{smem_bytes}];"
    ));
    lines.push(String::new());

    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_partials,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H,"));
    lines.push(p("\t.param .u32 param_num_tiles"));
    lines.push(p(") {"));

    lines.push(p("\t.reg .u64 %rd<30>;"));
    lines.push(p("\t.reg .u32 %r<24>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .b16 %h0, %h1, %h2;"));
    lines.push(p("\t.reg .f32 %facc, %fa, %fb, %ftmp;"));
    lines.push(p("\t.reg .f32 %ftmax, %ftsum, %flog2e;"));
    lines.push(p("\t.reg .pred %pskip, %pv, %pth0;"));
    lines.push(String::new());

    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_partials];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.y;   // row_idx"));
    lines.push(p("\tmov.u32 %r1, %tid.x;     // tid"));
    lines.push(p("\tmov.u32 %r2, %ctaid.x;   // tile_idx"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B;"));
    lines.push(String::new());

    // partials_slot = partials + (row*num_tiles + tile) * 2 * 4 (f32 partials).
    lines.push(format!("\t// partials_slot = partials + (row*{n_tiles}+tile) * 8 (f32 partials)"));
    lines.push(p("\tcvt.u64.u32 %rd5, %r0;"));
    lines.push(format!("\tmov.u32 %r3, {n_tiles};"));
    lines.push(p("\tcvt.u64.u32 %rd6, %r3;"));
    lines.push(p("\tmul.lo.u64 %rd5, %rd5, %rd6;"));
    lines.push(p("\tcvt.u64.u32 %rd7, %r2;"));
    lines.push(p("\tadd.u64 %rd5, %rd5, %rd7;"));
    lines.push(p("\tshl.b64 %rd5, %rd5, 3; // *8"));
    lines.push(p("\tadd.u64 %rd5, %rd4, %rd5;"));
    lines.push(String::new());

    // Load target.
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd8, %rd8, 8;"));
    lines.push(p("\tadd.u64 %rd8, %rd3, %rd8;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd8];"));
    lines.push(String::new());

    // Skip path: write (0, 0) f32 partials.
    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@!%pskip bra LP_NOTSKIP;"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LP_DONE;"));
    lines.push(p("\tst.global.f32 [%rd5], 0f00000000;"));
    lines.push(p("\tst.global.f32 [%rd5+4], 0f00000000;"));
    lines.push(p("\tbra LP_DONE;"));
    lines.push(p("LP_NOTSKIP:"));
    lines.push(String::new());

    // x_row_base = x + row * H * 2 (fp16).
    lines.push(p("\tcvt.u64.u32 %rd9, %r0;"));
    lines.push(format!("\tmov.u32 %r4, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd10, %r4;"));
    lines.push(p("\tmul.lo.u64 %rd9, %rd9, %rd10;"));
    lines.push(p("\tshl.b64 %rd9, %rd9, 1; // *2 fp16"));
    lines.push(p("\tadd.u64 %rd9, %rd0, %rd9;"));
    lines.push(String::new());

    lines.push(format!("\tmul.lo.u32 %r5, %r2, {vtile}; // v_base"));
    lines.push(String::new());

    lines.push(p("\tmov.u32 %r6, 0;"));
    lines.push(p("LP_INNER:"));
    lines.push(p("\t\tmul.lo.u32 %r7, %r6, 128;"));
    lines.push(p("\t\tadd.u32 %r7, %r7, %r1;"));
    lines.push(p("\t\tadd.u32 %r8, %r7, %r5;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r8, {vocab};"));
    lines.push(p("\t\t@!%pv bra LP_INNER_TAIL_ZERO;"));
    lines.push(String::new());

    // W_row_base — fp16 stride 2.
    lines.push(p("\t\tcvt.u64.u32 %rd11, %r8;"));
    lines.push(format!("\t\tmov.u32 %r9, {hidden};"));
    lines.push(p("\t\tcvt.u64.u32 %rd12, %r9;"));
    lines.push(p("\t\tmul.lo.u64 %rd11, %rd11, %rd12;"));
    lines.push(p("\t\tshl.b64 %rd11, %rd11, 1; // *2 fp16"));
    lines.push(p("\t\tadd.u64 %rd11, %rd1, %rd11;"));
    lines.push(String::new());

    lines.push(p("\t\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tLP_DOT:"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd13, %r10;"));
    lines.push(p("\t\t\tshl.b64 %rd13, %rd13, 1; // *2 fp16"));
    lines.push(p("\t\t\tadd.u64 %rd14, %rd9, %rd13;"));
    lines.push(p("\t\t\tld.global.b16 %h0, [%rd14];"));
    lines.push(p("\t\t\tcvt.f32.f16 %fa, %h0;"));
    lines.push(p("\t\t\tadd.u64 %rd14, %rd11, %rd13;"));
    lines.push(p("\t\t\tld.global.b16 %h1, [%rd14];"));
    lines.push(p("\t\t\tcvt.f32.f16 %fb, %h1;"));
    lines.push(p("\t\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {hidden};"));
    lines.push(p("\t\t\t@%pv bra LP_DOT;"));
    lines.push(String::new());

    // Bias fp16.
    lines.push(p("\t\tcvt.u64.u32 %rd15, %r8;"));
    lines.push(p("\t\tshl.b64 %rd15, %rd15, 1; // *2"));
    lines.push(p("\t\tadd.u64 %rd15, %rd2, %rd15;"));
    lines.push(p("\t\tld.global.b16 %h2, [%rd15];"));
    lines.push(p("\t\tcvt.f32.f16 %ftmp, %h2;"));
    lines.push(p("\t\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(p("\t\tbra LP_INNER_STORE;"));
    lines.push(String::new());

    // Tail-zero: store fp16 -INF directly to smem so it doesn't perturb
    // max/sum.  Review Finding 2: previously this path mov'd the f32
    // bit-pattern 0f80800000 (which is -1.175e-38 — the smallest normal
    // negative f32, NOT -INF) into %facc and fell through to the shared
    // store-via-cvt path.  cvt.rn.f16.f32 then mapped -1.175e-38 to fp16
    // 0x0000 (below fp16 subnormal range), and the downstream
    // LP_RED_MAX/LP_RED_SUM reads max'd with 0.0 — corrupting the
    // per-tile LSE whenever all real logits in the tile were negative.
    // We now branch around the cvt and write 0xFC00 (fp16 -INF) directly.
    lines.push(p("LP_INNER_TAIL_ZERO:"));
    lines.push(p("\t\tshl.b32 %r11, %r7, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd16, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r11;"));
    lines.push(p("\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\tmov.b16 %h2, 0xFC00; // fp16 -INF (direct, no f32 cvt)"));
    lines.push(p("\t\tst.shared.b16 [%rd16], %h2;"));
    lines.push(p("\t\tbra LP_INNER_AFTER_STORE;"));
    lines.push(String::new());

    // Store logit to smem as fp16 — real-tile path goes through the
    // f32 → fp16 cvt below.
    lines.push(p("LP_INNER_STORE:"));
    lines.push(p("\t\tshl.b32 %r11, %r7, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd16, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r11;"));
    lines.push(p("\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\tcvt.rn.f16.f32 %h2, %facc;"));
    lines.push(p("\t\tst.shared.b16 [%rd16], %h2;"));
    lines.push(p("LP_INNER_AFTER_STORE:"));
    lines.push(String::new());

    lines.push(p("\t\tadd.u32 %r6, %r6, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r6, {vtile_per_thread};"));
    lines.push(p("\t\t@%pv bra LP_INNER;"));
    lines.push(String::new());

    lines.push(p("\tbar.sync 0;"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LP_DONE;"));
    lines.push(String::new());

    // Reduce smem fp16 tile to (tile_max, tile_sum_unscaled) in f32.
    lines.push(p("\tmov.f32 %ftmax, 0f80800000;"));
    lines.push(p("\tmov.u32 %r12, 0;"));
    lines.push(p("LP_RED_MAX:"));
    lines.push(p("\t\tshl.b32 %r13, %r12, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd18, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd19, %r13;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd18, %rd19;"));
    lines.push(p("\t\tld.shared.b16 %h0, [%rd18];"));
    lines.push(p("\t\tcvt.f32.f16 %ftmp, %h0;"));
    lines.push(p("\t\tmax.f32 %ftmax, %ftmax, %ftmp;"));
    lines.push(p("\t\tadd.u32 %r12, %r12, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r12, {vtile};"));
    lines.push(p("\t\t@%pv bra LP_RED_MAX;"));
    lines.push(String::new());

    lines.push(p("\tmov.f32 %ftsum, 0f00000000;"));
    lines.push(p("\tmov.u32 %r12, 0;"));
    lines.push(p("LP_RED_SUM:"));
    lines.push(p("\t\tshl.b32 %r13, %r12, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd18, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd19, %r13;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd18, %rd19;"));
    lines.push(p("\t\tld.shared.b16 %h0, [%rd18];"));
    lines.push(p("\t\tcvt.f32.f16 %ftmp, %h0;"));
    lines.push(p("\t\tsub.f32 %ftmp, %ftmp, %ftmax;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tadd.f32 %ftsum, %ftsum, %ftmp;"));
    lines.push(p("\t\tadd.u32 %r12, %r12, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r12, {vtile};"));
    lines.push(p("\t\t@%pv bra LP_RED_SUM;"));
    lines.push(String::new());

    // Store (tile_max, tile_sum_unscaled) as f32 partials.
    lines.push(p("\tst.global.f32 [%rd5],   %ftmax;"));
    lines.push(p("\tst.global.f32 [%rd5+4], %ftsum;"));
    lines.push(String::new());

    lines.push(p("LP_DONE:"));
    lines.push(p("\tret;"));
    lines.push(p("}"));

    lines.join("\n")
}

/// F16 finalize kernel.
///
/// Reads f32 partials (Kernel A writes f32 regardless of activation dtype),
/// runs online-LSE rescale in f32, recomputes `logit_at_target = x[row] @
/// W[tgt] + bias[tgt]` with fp16 staging + f32 fma accumulator, writes
/// `loss_out` and `lse_out` as f32.
fn emit_large_finalize_kernel_f16(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.large_finalize_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let n_tiles = cfg.num_vocab_tiles();
    let ignore = cfg.ignore_index;

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();
    lines.push(String::new());

    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_partials,"));
    lines.push(p("\t.param .u64 param_loss_out,"));
    lines.push(p("\t.param .u64 param_lse_out,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H,"));
    lines.push(p("\t.param .u32 param_num_tiles"));
    lines.push(p(") {"));

    lines.push(p("\t.reg .u64 %rd<24>;"));
    lines.push(p("\t.reg .u32 %r<16>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .b16 %h0, %h1, %h2;"));
    lines.push(p("\t.reg .f32 %fmax, %fsum, %ftmax, %ftsum, %fnew_max;"));
    lines.push(p("\t.reg .f32 %ftmp, %fa, %fb, %facc, %flog, %flse, %floss;"));
    lines.push(p("\t.reg .f32 %flog2e, %fln2;"));
    lines.push(p("\t.reg .pred %pskip, %pth0, %pv;"));
    lines.push(String::new());

    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_partials];"));
    lines.push(p("\tld.param.u64 %rd5, [param_loss_out];"));
    lines.push(p("\tld.param.u64 %rd6, [param_lse_out];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.x;"));
    lines.push(p("\tmov.u32 %r1, %tid.x;"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B;"));
    lines.push(p("\tmov.f32 %fln2,   0f3F317218;"));
    lines.push(String::new());

    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LF_DONE;"));
    lines.push(String::new());

    lines.push(p("\tcvt.u64.u32 %rd7, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd7, %rd7, 8;"));
    lines.push(p("\tadd.u64 %rd7, %rd3, %rd7;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd7];"));
    lines.push(String::new());

    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(p("\tshl.b64 %rd8, %rd8, 2;"));
    lines.push(p("\tadd.u64 %rd9, %rd5, %rd8;"));
    lines.push(p("\tadd.u64 %rd10, %rd6, %rd8;"));
    lines.push(String::new());

    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@!%pskip bra LF_REDUCE;"));
    lines.push(p("\tst.global.f32 [%rd9],  0f00000000;"));
    lines.push(p("\tst.global.f32 [%rd10], 0f00000000;"));
    lines.push(p("\tbra LF_DONE;"));
    lines.push(p("LF_REDUCE:"));
    lines.push(String::new());

    // partials_row_base = partials + row * num_tiles * 8 (f32 partials).
    lines.push(p("\tcvt.u64.u32 %rd11, %r0;"));
    lines.push(format!("\tmov.u32 %r2, {n_tiles};"));
    lines.push(p("\tcvt.u64.u32 %rd12, %r2;"));
    lines.push(p("\tmul.lo.u64 %rd11, %rd11, %rd12;"));
    lines.push(p("\tshl.b64 %rd11, %rd11, 3;"));
    lines.push(p("\tadd.u64 %rd11, %rd4, %rd11;"));
    lines.push(String::new());

    // Online-LSE reduce — identical to F32 path (partials are f32).
    lines.push(p("\tmov.f32 %fmax, 0f80800000;"));
    lines.push(p("\tmov.f32 %fsum, 0f00000000;"));
    lines.push(p("\tmov.u32 %r3, 0;"));
    lines.push(p("LF_LOOP:"));
    lines.push(p("\t\tcvt.u64.u32 %rd13, %r3;"));
    lines.push(p("\t\tshl.b64 %rd13, %rd13, 3;"));
    lines.push(p("\t\tadd.u64 %rd13, %rd11, %rd13;"));
    lines.push(p("\t\tld.global.f32 %ftmax, [%rd13];"));
    lines.push(p("\t\tld.global.f32 %ftsum, [%rd13+4];"));
    lines.push(String::new());

    lines.push(p("\t\tmax.f32 %fnew_max, %fmax, %ftmax;"));
    lines.push(p("\t\tsub.f32 %ftmp, %fmax, %fnew_max;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tmul.f32 %fsum, %fsum, %ftmp;"));
    lines.push(p("\t\tsub.f32 %ftmp, %ftmax, %fnew_max;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tmul.f32 %ftsum, %ftsum, %ftmp;"));
    lines.push(p("\t\tadd.f32 %fsum, %fsum, %ftsum;"));
    lines.push(p("\t\tmov.f32 %fmax, %fnew_max;"));
    lines.push(String::new());

    lines.push(p("\t\tadd.u32 %r3, %r3, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r3, {n_tiles};"));
    lines.push(p("\t\t@%pv bra LF_LOOP;"));
    lines.push(String::new());

    lines.push(p("\tlg2.approx.f32 %flse, %fsum;"));
    lines.push(p("\tmul.f32 %flse, %flse, %fln2;"));
    lines.push(p("\tadd.f32 %flse, %flse, %fmax;"));
    lines.push(String::new());

    // x_row_base — fp16 stride 2.
    lines.push(p("\tcvt.u64.u32 %rd14, %r0;"));
    lines.push(format!("\tmov.u32 %r4, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd15, %r4;"));
    lines.push(p("\tmul.lo.u64 %rd14, %rd14, %rd15;"));
    lines.push(p("\tshl.b64 %rd14, %rd14, 1; // *2 fp16"));
    lines.push(p("\tadd.u64 %rd14, %rd0, %rd14;"));
    // W_tgt_base — fp16 stride 2.
    lines.push(p("\tmul.lo.s64 %rd16, %tgt64, %rd15;"));
    lines.push(p("\tshl.b64 %rd16, %rd16, 1; // *2 fp16"));
    lines.push(p("\tadd.u64 %rd16, %rd1, %rd16;"));
    lines.push(String::new());

    // Dot loop — fp16 → f32 → fma.
    lines.push(p("\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\tmov.u32 %r5, 0;"));
    lines.push(p("LF_DOT:"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r5;"));
    lines.push(p("\t\tshl.b64 %rd17, %rd17, 1; // *2"));
    lines.push(p("\t\tadd.u64 %rd18, %rd14, %rd17;"));
    lines.push(p("\t\tld.global.b16 %h0, [%rd18];"));
    lines.push(p("\t\tcvt.f32.f16 %fa, %h0;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd16, %rd17;"));
    lines.push(p("\t\tld.global.b16 %h1, [%rd18];"));
    lines.push(p("\t\tcvt.f32.f16 %fb, %h1;"));
    lines.push(p("\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\tadd.u32 %r5, %r5, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r5, {hidden};"));
    lines.push(p("\t\t@%pv bra LF_DOT;"));
    lines.push(String::new());

    // Bias[tgt] fp16.
    lines.push(p("\tmul.lo.s64 %rd19, %tgt64, 2; // fp16 stride"));
    lines.push(p("\tadd.u64 %rd19, %rd2, %rd19;"));
    lines.push(p("\tld.global.b16 %h2, [%rd19];"));
    lines.push(p("\tcvt.f32.f16 %ftmp, %h2;"));
    lines.push(p("\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(String::new());

    lines.push(p("\tsub.f32 %floss, %flse, %facc;"));
    lines.push(p("\tst.global.f32 [%rd9],  %floss;"));
    lines.push(p("\tst.global.f32 [%rd10], %flse;"));
    lines.push(String::new());

    lines.push(p("LF_DONE:"));
    let _ = vocab;
    lines.push(p("\tret;"));
    lines.push(p("}"));

    lines.join("\n")
}

// ── F16 backward kernel ──────────────────────────────────────────────────────
//
// Mixed-precision convention (Sprint v3-2):
//   * x / W / bias HBM loads are `ld.global.b16` + `cvt.f32.f16` into f32
//     math registers. Backward recomputes logits from forward inputs and
//     the saved f32 lse, so the dtype of `x`/`W`/`bias` is the same as in
//     the forward kernel.
//   * The saved `lse` buffer stays `.f32` (written by the forward kernel
//     as f32 regardless of activation dtype) — `ld.global.f32 %lse_val`.
//   * The `grad_output` parameter is still `.param .f32` (a scalar; no
//     reason to halve a single value).
//   * Gradient outputs `dx`, `dW`, `dbias` stay `.f32` and the cross-CTA
//     accumulator uses `red.global.add.f32`. Rationale:
//       - `red.global.add.f16` is not portable across SMs (some pre-sm_70
//         lack it; sm_80+ supports it but adds a numerical-determinism
//         risk via non-deterministic accumulation order in fp16).
//       - PyTorch's standard mixed-precision convention writes master
//         gradients in f32; downstream optimizer state stays f32.
//       - Per the Sprint v3-2 spec: "current backward signature returns
//         f32 dW even when dtype=F16; this matches PyTorch's
//         mixed-precision convention".
//     The optional fp16 down-cast in an epilogue kernel is deferred.
//
// Output buffers dx/dW/dbias MUST be allocated by the caller as f32 even
// when `dtype = F16` — the runtime FFI layer threads this convention.
fn emit_bwd_kernel_f16(cfg: &FusedLinearCEConfig) -> String {
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

    // SMEM not used by backward (forward stored everything it needs in HBM),
    // but the declaration is kept for ABI parity with the F32 path's launcher.
    s.push_str(&format!(
        ".extern .shared .align 2 .b8 smem_scratch[{smem_bytes}];\n\n"
    ));

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

    s.push_str(
        "\t.reg .u64 %rd<24>;\n\
         \t.reg .u32 %r<20>;\n\
         \t.reg .s64 %target_val;\n\
         \t.reg .b16 %h0, %h1, %h2;\n\
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

    s.push_str(
        "\tmov.u32 %r0, %ctaid.x;\n\
         \tmov.u32 %r1, %tid.x;\n\
    \n",
    );

    s.push_str(
        "\tcvt.u64.u32 %rd8, %r0;\n\
         \tmul.lo.u64 %rd8, %rd8, 8;\n\
         \tadd.u64 %rd8, %rd3, %rd8;\n\
         \tld.global.s64 %target_val, [%rd8];\n\
    \n",
    );

    s.push_str(&format!(
        "\tsetp.eq.s64 %p_skip, %target_val, {ignore};\n\
         \t@%p_skip bra BWD_SKIP_LABEL;\n\
    \n"
    ));

    // Saved lse stays f32 even at dtype=F16 (matches forward's f32 write).
    s.push_str(
        "\tcvt.u64.u32 %rd9, %r0;\n\
         \tshl.b64 %rd9, %rd9, 2;\n\
         \tadd.u64 %rd9, %rd4, %rd9;\n\
         \tld.global.f32 %lse_val, [%rd9];\n\
    \n",
    );

    // x_row_base + dx_row_base. x stride 2 (fp16); dx stride 4 (f32 output).
    s.push_str(&format!(
        "\tcvt.u64.u32 %rd10, %r0;\n\
         \tmov.u32 %r2, {hidden};\n\
         \tcvt.u64.u32 %rd11, %r2;\n\
         \tmul.lo.u64 %rd10, %rd10, %rd11;\n\
         \tshl.b64 %rd10, %rd10, 1; // x: *2 (fp16)\n\
         \tadd.u64 %rd10, %rd0, %rd10;\n\
         \t// dx_row_base = dx_out + row_idx * H * 4 (f32 grad)\n\
         \tcvt.u64.u32 %rd20, %r0;\n\
         \tmul.lo.u64 %rd20, %rd20, %rd11;\n\
         \tshl.b64 %rd20, %rd20, 2; // dx: *4 (f32)\n\
         \tadd.u64 %rd20, %rd5, %rd20;\n\
    \n"
    ));

    s.push_str("\tdiv.rn.f32 %scale, %grad_output, %num_valid_f;\n\n");
    s.push_str("\tmov.f32 %f15, 0f3FB8AA3B; // log2(e)\n\n");

    s.push_str(
        "\tmov.u32 %r3, 0; // tile_idx\n\
         BWD_TILE_LOOP:\n",
    );

    s.push_str(&format!(
        "\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base\n\
    \n"
    ));

    s.push_str(
        "\t\tmov.u32 %r5, 0; // inner counter\n\
         BWD_INNER_LOOP:\n",
    );

    s.push_str(
        "\t\t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\t\tadd.u32 %r6, %r6, %r1;\n\
         \t\t\tadd.u32 %r6, %r6, %r4;\n",
    );

    s.push_str(&format!(
        "\t\t\tsetp.lt.u32 %p_valid, %r6, {vocab};\n\
         \t\t\t@!%p_valid bra BWD_INNER_SKIP;\n\
    \n"
    ));

    // W_row_base — fp16 stride 2.
    s.push_str(&format!(
        "\t\t\t// W_row_base for v_idx (fp16)\n\
         \t\t\tcvt.u64.u32 %rd12, %r6;\n\
         \t\t\tmov.u32 %r7, {hidden};\n\
         \t\t\tcvt.u64.u32 %rd13, %r7;\n\
         \t\t\tmul.lo.u64 %rd12, %rd12, %rd13;\n\
         \t\t\tshl.b64 %rd12, %rd12, 1; // *2 fp16\n\
         \t\t\tadd.u64 %rd12, %rd1, %rd12;\n\
    \n"
    ));

    // Dot product — fp16 loads → cvt.f32.f16 → fma.f32.
    s.push_str(
        "\t\t\tmov.f32 %logit_acc, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         BWD_DOT_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r8;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 1; // *2 fp16\n\
         \t\t\t\tadd.u64 %rd15, %rd10, %rd14;\n\
         \t\t\t\tld.global.b16 %h0, [%rd15];\n\
         \t\t\t\tcvt.f32.f16 %f0, %h0;\n\
         \t\t\t\tadd.u64 %rd16, %rd12, %rd14;\n\
         \t\t\t\tld.global.b16 %h1, [%rd16];\n\
         \t\t\t\tcvt.f32.f16 %f1, %h1;\n\
         \t\t\t\tfma.rn.f32 %logit_acc, %f0, %f1, %logit_acc;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\t\tsetp.lt.u32 %p_valid, %r8, {hidden};\n\
         \t\t\t\t@%p_valid bra BWD_DOT_LOOP;\n\
    \n"
    ));

    // Add bias (fp16).
    s.push_str(
        "\t\t\t// bias (fp16)\n\
         \t\t\tcvt.u64.u32 %rd17, %r6;\n\
         \t\t\tshl.b64 %rd17, %rd17, 1; // *2\n\
         \t\t\tadd.u64 %rd17, %rd2, %rd17;\n\
         \t\t\tld.global.b16 %h2, [%rd17];\n\
         \t\t\tcvt.f32.f16 %f2, %h2;\n\
         \t\t\tadd.f32 %logit_acc, %logit_acc, %f2;\n\
    \n",
    );

    // p_v = exp(logit_v - lse).
    s.push_str(
        "\t\t\tsub.f32 %f3, %logit_acc, %lse_val;\n\
         \t\t\tmul.f32 %f3, %f3, %f15;\n\
         \t\t\tex2.approx.f32 %f3, %f3;\n\
    \n",
    );

    s.push_str(
        "\t\t\tcvt.s64.u32 %rd18, %r6;\n\
         \t\t\tsetp.eq.s64 %p_is_target, %rd18, %target_val;\n\
         \t\t\t@%p_is_target sub.f32 %f3, %f3, 0f3F800000; // -= 1.0\n\
         \t\t\tmul.f32 %f4, %f3, %scale;\n\
    \n",
    );

    // dW_row_base — f32 stride 4 (dW is master-precision f32 regardless of activation dtype).
    s.push_str(&format!(
        "\t\t\t// dW_row_base = dW_out + v_idx * H * 4 (f32 master grad)\n\
         \t\t\tcvt.u64.u32 %rd21, %r6;\n\
         \t\t\tmov.u32 %r9, {hidden};\n\
         \t\t\tcvt.u64.u32 %rd22, %r9;\n\
         \t\t\tmul.lo.u64 %rd21, %rd21, %rd22;\n\
         \t\t\tshl.b64 %rd21, %rd21, 2; // *4 (f32)\n\
         \t\t\tadd.u64 %rd21, %rd6, %rd21;\n\
    \n"
    ));

    // H-loop scatter.
    //   W[v, h]  is fp16 (stride 2 from %rd12)
    //   x[row,h] is fp16 (stride 2 from %rd10)
    //   dx_out + dW_out are f32 (stride 4 from %rd20 / %rd21)
    s.push_str(
        "\t\t\tmov.u32 %r9, 0; // h counter\n\
         BWD_H_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd23, %r9;\n\
         \t\t\t\tshl.b64 %rd23, %rd23, 1; // *2 fp16 (for W, x loads)\n\
         \t\t\t\t// W[v, h] fp16\n\
         \t\t\t\tadd.u64 %rd14, %rd12, %rd23;\n\
         \t\t\t\tld.global.b16 %h0, [%rd14];\n\
         \t\t\t\tcvt.f32.f16 %f5, %h0;\n\
         \t\t\t\t// f6 = scaled * W[v, h]  (f32 grad slice)\n\
         \t\t\t\tmul.f32 %f6, %f4, %f5;\n\
         \t\t\t\t// dx_out[row, h] += f6  (f32 destination - stride 4)\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r9;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 2;\n\
         \t\t\t\tadd.u64 %rd14, %rd20, %rd14;\n\
         \t\t\t\tred.global.add.f32 [%rd14], %f6;\n\
         \t\t\t\t// x[row, h] fp16\n\
         \t\t\t\tadd.u64 %rd14, %rd10, %rd23;\n\
         \t\t\t\tld.global.b16 %h1, [%rd14];\n\
         \t\t\t\tcvt.f32.f16 %f7, %h1;\n\
         \t\t\t\t// f8 = scaled * x[row, h]\n\
         \t\t\t\tmul.f32 %f8, %f4, %f7;\n\
         \t\t\t\t// dW_out[v, h] += f8  (f32 destination - stride 4)\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r9;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 2;\n\
         \t\t\t\tadd.u64 %rd14, %rd21, %rd14;\n\
         \t\t\t\tred.global.add.f32 [%rd14], %f8;\n\
         \t\t\t\tadd.u32 %r9, %r9, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\t\tsetp.lt.u32 %p_valid, %r9, {hidden};\n\
         \t\t\t\t@%p_valid bra BWD_H_LOOP;\n\
    \n"
    ));

    // dbias[v] += scaled (f32 output stride 4).
    s.push_str(
        "\t\t\t// dbias_out[v] += scaled (f32 output)\n\
         \t\t\tcvt.u64.u32 %rd14, %r6;\n\
         \t\t\tshl.b64 %rd14, %rd14, 2; // *4 f32\n\
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

    s.push_str("\t\tadd.u32 %r3, %r3, 1;\n");

    s.push_str(&format!(
        "\t\tsetp.lt.u32 %p_valid, %r3, {n_tiles};\n\
         \t\t@%p_valid bra BWD_TILE_LOOP;\n\
    \n"
    ));

    s.push_str("\tbra BWD_DONE;\n\n");

    // Skip path: zero dx_out[row, :] as f32 (stride 4).
    s.push_str(
        "BWD_SKIP_LABEL:\n\
         \t// Zero dx_out[row, :] for skipped token (f32 output)\n",
    );

    s.push_str(&format!(
        "\tcvt.u64.u32 %rd10, %r0;\n\
         \tmov.u32 %r2, {hidden};\n\
         \tcvt.u64.u32 %rd11, %r2;\n\
         \tmul.lo.u64 %rd10, %rd10, %rd11;\n\
         \tshl.b64 %rd10, %rd10, 2; // *4 (f32 dx)\n\
         \tadd.u64 %rd10, %rd5, %rd10;\n\
    \n"
    ));

    s.push_str(&format!(
        "\tmov.u32 %r5, 0;\n\
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


// ─── PTX emission — Bf16 path (Sprint v4-1) ──────────────────────────────────
//
// Bf16 emitters mirror the F16 emitters structurally — same kernel layout,
// loop nesting, sync points, register allocation, and SMEM partitioning.
// The only differences are:
//   * `cvt.f32.f16`      -> `cvt.f32.bf16`     (HBM/SMEM bf16 -> f32 math)
//   * `cvt.rn.f16.f32`   -> `cvt.rn.bf16.f32`  (f32 -> bf16 store)
//   * `0xFC00`           -> `0xFF80`           (bf16 -INF sentinel —
//                                              bf16 has an 8-bit exponent
//                                              like f32, so -INF is sign-bit
//                                              + all-ones exponent + zero
//                                              mantissa = 0xFF80)
//   * `.version 7.0`     -> `.version 8.0`     (bf16 cvt mnemonics require
//                                              PTX ISA 7.8+; bumped via
//                                              `ptx_header()` dtype dispatch)
//
// SMEM declaration stays `.align 2 .b8` (bf16 is 16-bit storage like f16);
// the bare `ld.global.b16` / `st.shared.b16` storage instructions are
// dtype-agnostic — only the surrounding cvt mnemonics differ.
//
// Output buffers `loss_out` / `lse_out` / `dx` / `dW` / `dbias` stay f32 —
// same master-grad convention as the F16 path.

// ── Bf16 forward kernel ───────────────────────────────────────────────────────
//
// Mixed-precision convention (Sprint v3-2):
//   * HBM loads of x, W, bias are `ld.global.b16` → `cvt.f32.bf16` into f32
//     math registers (the dot-product accumulators).
//   * SMEM logit tile is stored as `.b16` (2 bytes/elem) — the per-tile
//     online-LSE reduction loads `.b16`, converts to f32, then folds into
//     the f32 max + sum registers. This halves the SMEM footprint vs F32.
//   * +INF / -INF sentinels and the log2e + ln2 constants live in f32
//     registers; the online-LSE algorithm is bit-for-bit the same as F32.
//   * HBM outputs `loss_out` and `lse_out` are written as `.f32` (per
//     convention; downstream backward consumes the same f32 lse).
//
// The kernel layout, loop structure, sync points, and SMEM partitioning
// mirror `emit_fwd_kernel` exactly — only the dtype of HBM/SMEM staging
// changes. Keeping the structure parallel preserves the option of merging
// the two emitters in a future refactor (deferred for byte-identity safety).
fn emit_fwd_kernel_bf16(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = vocab.div_ceil(vtile);
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();
    // Bf16 SMEM: 2 bytes/elem instead of 4. logit_at_target scratch slot lives
    // immediately after the tile and ALSO uses 2 bytes; the +32 padding in
    // shared_mem_bytes() leaves ample room.
    let elem_bytes = cfg.dtype.bytes_per_elem(); // = 2 for Bf16
    let lat_offset = vtile * elem_bytes;

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    lines.push(cfg.ptx_header());
    // `.align 2` because the SMEM is half-precision; ptxas auto-aligns to 4
    // for the +INF sentinel store anyway — but the declared align matches
    // the dtype.
    lines.push(format!(".extern .shared .align 2 .b8 smem_scratch[{smem_bytes}];"));
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
    lines.push(p("\t.reg .b16 %h0, %h1, %h2;"));
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

    // Thread 0 initialises logit_at_target smem slot to -INF (as bf16
    // bit-pattern 0xFBFF = -65504, the most negative finite bf16; the
    // logit_at_target slot is only consumed if the target was found, so a
    // strict -INF bit-pattern is not required — but using bf16 -INF
    // (0xFF80) keeps the semantics matched).
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra INIT_DONE;"));
    lines.push(p("\tmov.u64 %rd6, smem_scratch;"));
    lines.push(format!("\tadd.u64 %rd6, %rd6, {lat_offset};"));
    lines.push(p("\tmov.b16 %h0, 0xFF80; // bf16 -INF sentinel"));
    lines.push(p("\tst.shared.b16 [%rd6], %h0;"));
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

    // x_row_base = x + row * H * 2 (bf16: 2 bytes/elem).
    lines.push(format!(
        "\t// x_row_base = x + row_idx * {hidden} * {elem_bytes}"
    ));
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(format!("\tmov.u32 %r2, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd9, %r2;"));
    lines.push(p("\tmul.lo.u64 %rd8, %rd8, %rd9;"));
    lines.push(p("\tshl.b64 %rd8, %rd8, 1; // *2 for bf16"));
    lines.push(p("\tadd.u64 %rd8, %rd0, %rd8; // %rd8 = x_row_base"));
    lines.push(String::new());

    // Init online-softmax accumulators (f32).
    lines.push(p("\tmov.f32 %fmax, 0f80800000; // -INF (f32)"));
    lines.push(p("\tmov.f32 %fsum, 0f00000000;"));
    lines.push(String::new());

    // Outer tile loop.
    lines.push(p("\tmov.u32 %r3, 0; // tile_idx"));
    lines.push(p("TILE_LOOP:"));
    lines.push(format!("\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base"));
    lines.push(String::new());

    lines.push(p("\t\tmov.u32 %r5, 0; // sub-tile counter"));
    lines.push(p("\t\tINNER_LOOP:"));
    lines.push(p("\t\t\tmul.lo.u32 %r6, %r5, 128;"));
    lines.push(p("\t\t\tadd.u32 %r6, %r6, %r1;"));
    lines.push(p("\t\t\tadd.u32 %r6, %r6, %r4; // v_idx"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r6, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra INNER_SKIP;"));
    lines.push(String::new());

    // W_row_base = W + v_idx * H * 2.
    lines.push(format!(
        "\t\t\t// W_row_base = W + v_idx * {hidden} * {elem_bytes}"
    ));
    lines.push(p("\t\t\tcvt.u64.u32 %rd10, %r6;"));
    lines.push(format!("\t\t\tmov.u32 %r7, {hidden};"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd11, %r7;"));
    lines.push(p("\t\t\tmul.lo.u64 %rd10, %rd10, %rd11;"));
    lines.push(p("\t\t\tshl.b64 %rd10, %rd10, 1; // *2 for bf16"));
    lines.push(p("\t\t\tadd.u64 %rd10, %rd1, %rd10; // %rd10 = W_row_base"));
    lines.push(String::new());

    // Dot-product loop — load .b16, cvt to .f32, fma in f32.
    lines.push(p("\t\t\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\t\t\tmov.u32 %r8, 0; // h"));
    lines.push(p("\t\t\tDOT_LOOP:"));
    lines.push(p("\t\t\t\tcvt.u64.u32 %rd12, %r8;"));
    lines.push(p("\t\t\t\tshl.b64 %rd12, %rd12, 1; // h * 2"));
    lines.push(p("\t\t\t\tadd.u64 %rd13, %rd8, %rd12;"));
    lines.push(p("\t\t\t\tld.global.b16 %h1, [%rd13];"));
    lines.push(p("\t\t\t\tcvt.f32.bf16 %fa, %h1;"));
    lines.push(p("\t\t\t\tadd.u64 %rd14, %rd10, %rd12;"));
    lines.push(p("\t\t\t\tld.global.b16 %h1, [%rd14];"));
    lines.push(p("\t\t\t\tcvt.f32.bf16 %fb, %h1;"));
    lines.push(p("\t\t\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\t\t\tadd.u32 %r8, %r8, 1;"));
    lines.push(format!("\t\t\t\tsetp.lt.u32 %pv, %r8, {hidden};"));
    lines.push(p("\t\t\t\t@%pv bra DOT_LOOP;"));
    lines.push(String::new());

    // Add bias[v_idx].
    lines.push(p("\t\t\t// facc += bias[v_idx] (bf16 HBM)"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd15, %r6;"));
    lines.push(p("\t\t\tshl.b64 %rd15, %rd15, 1; // *2"));
    lines.push(p("\t\t\tadd.u64 %rd15, %rd2, %rd15;"));
    lines.push(p("\t\t\tld.global.b16 %h1, [%rd15];"));
    lines.push(p("\t\t\tcvt.f32.bf16 %ftmp, %h1;"));
    lines.push(p("\t\t\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(String::new());

    // Store logit to smem tile (cvt back to bf16; tile is .b16 stride 2).
    lines.push(p("\t\t\t// Store logit to smem_scratch[(r5*128+tid)*2] as bf16"));
    lines.push(p("\t\t\tmul.lo.u32 %r9, %r5, 128;"));
    lines.push(p("\t\t\tadd.u32 %r9, %r9, %r1;"));
    lines.push(p("\t\t\tshl.b32 %r9, %r9, 1; // *2"));
    lines.push(p("\t\t\tmov.u64 %rd16, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd17, %r9;"));
    lines.push(p("\t\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\t\tcvt.rn.bf16.f32 %h2, %facc;"));
    lines.push(p("\t\t\tst.shared.b16 [%rd16], %h2;"));
    lines.push(String::new());

    // If v_idx == target, record logit_at_target in smem (also bf16).
    lines.push(p("\t\t\t// If v_idx == target, record logit_at_target (bf16)"));
    lines.push(p("\t\t\tcvt.s64.u32 %rd18, %r6;"));
    lines.push(p("\t\t\tsetp.eq.s64 %ptgt, %rd18, %tgt64;"));
    lines.push(p("\t\t\t@!%ptgt bra NOT_TARGET;"));
    lines.push(p("\t\t\tmov.u64 %rd19, smem_scratch;"));
    lines.push(format!("\t\t\tadd.u64 %rd19, %rd19, {lat_offset};"));
    lines.push(p("\t\t\tst.shared.b16 [%rd19], %h2;"));
    lines.push(p("\t\t\tNOT_TARGET:"));
    lines.push(String::new());

    lines.push(p("\t\t\tINNER_SKIP:"));
    lines.push(p("\t\t\tadd.u32 %r5, %r5, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r5, {vtile_per_thread};"));
    lines.push(p("\t\t\t@%pv bra INNER_LOOP;"));
    lines.push(String::new());

    // Sync: all logits in smem.
    lines.push(p("\t\tbar.sync 0;"));
    lines.push(String::new());

    // Thread 0 reduces smem tile.
    lines.push(p("\t\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t\t@!%pth0 bra TILE_REDUCE_DONE;"));
    lines.push(String::new());

    // Step 1: find tile_max (read bf16 → cvt to f32, fold into f32 max).
    lines.push(p("\t\tmov.f32 %ftmax, 0f80800000;"));
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tSMEM_MAX_LOOP:"));
    lines.push(p("\t\t\tadd.u32 %r11, %r4, %r10; // v_base + i"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r11, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra SMEM_MAX_DONE;"));
    lines.push(p("\t\t\tshl.b32 %r12, %r10, 1; // *2"));
    lines.push(p("\t\t\tmov.u64 %rd20, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd21, %r12;"));
    lines.push(p("\t\t\tadd.u64 %rd20, %rd20, %rd21;"));
    lines.push(p("\t\t\tld.shared.b16 %h1, [%rd20];"));
    lines.push(p("\t\t\tcvt.f32.bf16 %ftmp, %h1;"));
    lines.push(p("\t\t\tmax.f32 %ftmax, %ftmax, %ftmp;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {vtile};"));
    lines.push(p("\t\t\t@%pv bra SMEM_MAX_LOOP;"));
    lines.push(p("\t\tSMEM_MAX_DONE:"));
    lines.push(String::new());

    // Online-softmax rescale.
    lines.push(p("\t\t// Online-softmax update"));
    lines.push(p("\t\tmax.f32 %flog, %fmax, %ftmax; // new_max"));
    lines.push(p("\t\tsub.f32 %fmax, %fmax, %flog;"));
    lines.push(p("\t\tmul.f32 %fmax, %fmax, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %fmax, %fmax;"));
    lines.push(p("\t\tmul.f32 %fsum, %fsum, %fmax;"));
    lines.push(p("\t\tmov.f32 %fmax, %flog;"));
    lines.push(String::new());

    // Tile sum (bf16 load → cvt to f32 → fold into f32 sum).
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tSMEM_SUM_LOOP:"));
    lines.push(p("\t\t\tadd.u32 %r11, %r4, %r10;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r11, {vocab};"));
    lines.push(p("\t\t\t@!%pv bra SMEM_SUM_DONE;"));
    lines.push(p("\t\t\tshl.b32 %r12, %r10, 1; // *2"));
    lines.push(p("\t\t\tmov.u64 %rd20, smem_scratch;"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd21, %r12;"));
    lines.push(p("\t\t\tadd.u64 %rd20, %rd20, %rd21;"));
    lines.push(p("\t\t\tld.shared.b16 %h1, [%rd20];"));
    lines.push(p("\t\t\tcvt.f32.bf16 %ftmp, %h1;"));
    lines.push(p("\t\t\tsub.f32 %ftmp, %ftmp, %fmax;"));
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

    lines.push(p("\t\tadd.u32 %r3, %r3, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r3, {n_tiles};"));
    lines.push(p("\t\t@%pv bra TILE_LOOP;"));
    lines.push(String::new());

    // Finalize: thread 0 writes loss + lse as f32.
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra WRITE_DONE;"));
    lines.push(String::new());

    lines.push(p("\tlg2.approx.f32 %flse, %fsum;"));
    lines.push(p("\tmul.f32 %flse, %flse, %fln2;"));
    lines.push(p("\tadd.f32 %flse, %flse, %fmax;"));
    lines.push(String::new());

    // logit_at_target from smem (read bf16, cvt to f32).
    lines.push(p("\tmov.u64 %rd22, smem_scratch;"));
    lines.push(format!("\tadd.u64 %rd22, %rd22, {lat_offset};"));
    lines.push(p("\tld.shared.b16 %h1, [%rd22];"));
    lines.push(p("\tcvt.f32.bf16 %flog, %h1;"));
    lines.push(String::new());

    lines.push(p("\tsub.f32 %floss, %flse, %flog;"));
    lines.push(String::new());

    // Write outputs as f32 (loss + lse stay f32 regardless of dtype).
    lines.push(p("\tcvt.u64.u32 %rd23, %r0;"));
    lines.push(p("\tshl.b64 %rd23, %rd23, 2;"));
    lines.push(p("\tadd.u64 %rd24, %rd4, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], %floss;"));
    lines.push(p("\tadd.u64 %rd24, %rd5, %rd23;"));
    lines.push(p("\tst.global.f32 [%rd24], %flse;"));
    lines.push(p("\tbra WRITE_DONE;"));
    lines.push(String::new());

    // Skip label: write zeros (f32).
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

// ── Bf16 large-vocab kernels (Sprint v3-2) ────────────────────────────────────
//
// Mixed-precision convention for the two-kernel large-vocab path:
//   * Kernel A (per-tile partials) loads x / W / bias as `.b16` → `cvt.f32.bf16`,
//     accumulates dot products in f32, stages SMEM logits as `.b16`, and
//     writes the cross-CTA partials buffer as f32 — the partials buffer
//     stays f32 for cross-CTA numerical robustness (the online-LSE rescale
//     in Kernel B compounds across tiles; bf16 partials would visibly
//     degrade lse accuracy at vocab=49152+).
//   * Kernel B (per-row finalize) reads f32 partials AND ALSO recomputes
//     `logit_at_target = x[row] @ W[tgt] + bias[tgt]` — at dtype=Bf16 that
//     recompute uses the same bf16 HBM staging convention as Kernel A.
//     The final `loss_out` / `lse_out` writes stay f32 (same as v1).
fn emit_large_partials_kernel_bf16(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.large_partials_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let vtile = cfg.vocab_tile;
    let n_tiles = cfg.num_vocab_tiles();
    let vtile_per_thread = vtile / 128;
    let ignore = cfg.ignore_index;
    let smem_bytes = cfg.shared_mem_bytes();

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();

    // `.align 2` because SMEM is bf16.
    lines.push(format!(
        ".extern .shared .align 2 .b8 smem_partials_{vocab}[{smem_bytes}];"
    ));
    lines.push(String::new());

    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_partials,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H,"));
    lines.push(p("\t.param .u32 param_num_tiles"));
    lines.push(p(") {"));

    lines.push(p("\t.reg .u64 %rd<30>;"));
    lines.push(p("\t.reg .u32 %r<24>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .b16 %h0, %h1, %h2;"));
    lines.push(p("\t.reg .f32 %facc, %fa, %fb, %ftmp;"));
    lines.push(p("\t.reg .f32 %ftmax, %ftsum, %flog2e;"));
    lines.push(p("\t.reg .pred %pskip, %pv, %pth0;"));
    lines.push(String::new());

    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_partials];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.y;   // row_idx"));
    lines.push(p("\tmov.u32 %r1, %tid.x;     // tid"));
    lines.push(p("\tmov.u32 %r2, %ctaid.x;   // tile_idx"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B;"));
    lines.push(String::new());

    // partials_slot = partials + (row*num_tiles + tile) * 2 * 4 (f32 partials).
    lines.push(format!("\t// partials_slot = partials + (row*{n_tiles}+tile) * 8 (f32 partials)"));
    lines.push(p("\tcvt.u64.u32 %rd5, %r0;"));
    lines.push(format!("\tmov.u32 %r3, {n_tiles};"));
    lines.push(p("\tcvt.u64.u32 %rd6, %r3;"));
    lines.push(p("\tmul.lo.u64 %rd5, %rd5, %rd6;"));
    lines.push(p("\tcvt.u64.u32 %rd7, %r2;"));
    lines.push(p("\tadd.u64 %rd5, %rd5, %rd7;"));
    lines.push(p("\tshl.b64 %rd5, %rd5, 3; // *8"));
    lines.push(p("\tadd.u64 %rd5, %rd4, %rd5;"));
    lines.push(String::new());

    // Load target.
    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd8, %rd8, 8;"));
    lines.push(p("\tadd.u64 %rd8, %rd3, %rd8;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd8];"));
    lines.push(String::new());

    // Skip path: write (0, 0) f32 partials.
    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@!%pskip bra LP_NOTSKIP;"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LP_DONE;"));
    lines.push(p("\tst.global.f32 [%rd5], 0f00000000;"));
    lines.push(p("\tst.global.f32 [%rd5+4], 0f00000000;"));
    lines.push(p("\tbra LP_DONE;"));
    lines.push(p("LP_NOTSKIP:"));
    lines.push(String::new());

    // x_row_base = x + row * H * 2 (bf16).
    lines.push(p("\tcvt.u64.u32 %rd9, %r0;"));
    lines.push(format!("\tmov.u32 %r4, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd10, %r4;"));
    lines.push(p("\tmul.lo.u64 %rd9, %rd9, %rd10;"));
    lines.push(p("\tshl.b64 %rd9, %rd9, 1; // *2 bf16"));
    lines.push(p("\tadd.u64 %rd9, %rd0, %rd9;"));
    lines.push(String::new());

    lines.push(format!("\tmul.lo.u32 %r5, %r2, {vtile}; // v_base"));
    lines.push(String::new());

    lines.push(p("\tmov.u32 %r6, 0;"));
    lines.push(p("LP_INNER:"));
    lines.push(p("\t\tmul.lo.u32 %r7, %r6, 128;"));
    lines.push(p("\t\tadd.u32 %r7, %r7, %r1;"));
    lines.push(p("\t\tadd.u32 %r8, %r7, %r5;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r8, {vocab};"));
    lines.push(p("\t\t@!%pv bra LP_INNER_TAIL_ZERO;"));
    lines.push(String::new());

    // W_row_base — bf16 stride 2.
    lines.push(p("\t\tcvt.u64.u32 %rd11, %r8;"));
    lines.push(format!("\t\tmov.u32 %r9, {hidden};"));
    lines.push(p("\t\tcvt.u64.u32 %rd12, %r9;"));
    lines.push(p("\t\tmul.lo.u64 %rd11, %rd11, %rd12;"));
    lines.push(p("\t\tshl.b64 %rd11, %rd11, 1; // *2 bf16"));
    lines.push(p("\t\tadd.u64 %rd11, %rd1, %rd11;"));
    lines.push(String::new());

    lines.push(p("\t\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\t\tmov.u32 %r10, 0;"));
    lines.push(p("\t\tLP_DOT:"));
    lines.push(p("\t\t\tcvt.u64.u32 %rd13, %r10;"));
    lines.push(p("\t\t\tshl.b64 %rd13, %rd13, 1; // *2 bf16"));
    lines.push(p("\t\t\tadd.u64 %rd14, %rd9, %rd13;"));
    lines.push(p("\t\t\tld.global.b16 %h0, [%rd14];"));
    lines.push(p("\t\t\tcvt.f32.bf16 %fa, %h0;"));
    lines.push(p("\t\t\tadd.u64 %rd14, %rd11, %rd13;"));
    lines.push(p("\t\t\tld.global.b16 %h1, [%rd14];"));
    lines.push(p("\t\t\tcvt.f32.bf16 %fb, %h1;"));
    lines.push(p("\t\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\t\tadd.u32 %r10, %r10, 1;"));
    lines.push(format!("\t\t\tsetp.lt.u32 %pv, %r10, {hidden};"));
    lines.push(p("\t\t\t@%pv bra LP_DOT;"));
    lines.push(String::new());

    // Bias bf16.
    lines.push(p("\t\tcvt.u64.u32 %rd15, %r8;"));
    lines.push(p("\t\tshl.b64 %rd15, %rd15, 1; // *2"));
    lines.push(p("\t\tadd.u64 %rd15, %rd2, %rd15;"));
    lines.push(p("\t\tld.global.b16 %h2, [%rd15];"));
    lines.push(p("\t\tcvt.f32.bf16 %ftmp, %h2;"));
    lines.push(p("\t\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(p("\t\tbra LP_INNER_STORE;"));
    lines.push(String::new());

    // Tail-zero: store bf16 -INF directly to smem so it doesn't perturb
    // max/sum.  Review Finding 2: previously this path mov'd the f32
    // bit-pattern 0f80800000 (which is -1.175e-38 — the smallest normal
    // negative f32, NOT -INF) into %facc and fell through to the shared
    // store-via-cvt path.  cvt.rn.bf16.f32 then mapped -1.175e-38 to bf16
    // 0x0000 (below bf16 subnormal range), and the downstream
    // LP_RED_MAX/LP_RED_SUM reads max'd with 0.0 — corrupting the
    // per-tile LSE whenever all real logits in the tile were negative.
    // We now branch around the cvt and write 0xFF80 (bf16 -INF) directly.
    lines.push(p("LP_INNER_TAIL_ZERO:"));
    lines.push(p("\t\tshl.b32 %r11, %r7, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd16, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r11;"));
    lines.push(p("\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\tmov.b16 %h2, 0xFF80; // bf16 -INF (direct, no f32 cvt)"));
    lines.push(p("\t\tst.shared.b16 [%rd16], %h2;"));
    lines.push(p("\t\tbra LP_INNER_AFTER_STORE;"));
    lines.push(String::new());

    // Store logit to smem as bf16 — real-tile path goes through the
    // f32 → bf16 cvt below.
    lines.push(p("LP_INNER_STORE:"));
    lines.push(p("\t\tshl.b32 %r11, %r7, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd16, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r11;"));
    lines.push(p("\t\tadd.u64 %rd16, %rd16, %rd17;"));
    lines.push(p("\t\tcvt.rn.bf16.f32 %h2, %facc;"));
    lines.push(p("\t\tst.shared.b16 [%rd16], %h2;"));
    lines.push(p("LP_INNER_AFTER_STORE:"));
    lines.push(String::new());

    lines.push(p("\t\tadd.u32 %r6, %r6, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r6, {vtile_per_thread};"));
    lines.push(p("\t\t@%pv bra LP_INNER;"));
    lines.push(String::new());

    lines.push(p("\tbar.sync 0;"));
    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LP_DONE;"));
    lines.push(String::new());

    // Reduce smem bf16 tile to (tile_max, tile_sum_unscaled) in f32.
    lines.push(p("\tmov.f32 %ftmax, 0f80800000;"));
    lines.push(p("\tmov.u32 %r12, 0;"));
    lines.push(p("LP_RED_MAX:"));
    lines.push(p("\t\tshl.b32 %r13, %r12, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd18, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd19, %r13;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd18, %rd19;"));
    lines.push(p("\t\tld.shared.b16 %h0, [%rd18];"));
    lines.push(p("\t\tcvt.f32.bf16 %ftmp, %h0;"));
    lines.push(p("\t\tmax.f32 %ftmax, %ftmax, %ftmp;"));
    lines.push(p("\t\tadd.u32 %r12, %r12, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r12, {vtile};"));
    lines.push(p("\t\t@%pv bra LP_RED_MAX;"));
    lines.push(String::new());

    lines.push(p("\tmov.f32 %ftsum, 0f00000000;"));
    lines.push(p("\tmov.u32 %r12, 0;"));
    lines.push(p("LP_RED_SUM:"));
    lines.push(p("\t\tshl.b32 %r13, %r12, 1; // *2"));
    lines.push(format!("\t\tmov.u64 %rd18, smem_partials_{vocab};"));
    lines.push(p("\t\tcvt.u64.u32 %rd19, %r13;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd18, %rd19;"));
    lines.push(p("\t\tld.shared.b16 %h0, [%rd18];"));
    lines.push(p("\t\tcvt.f32.bf16 %ftmp, %h0;"));
    lines.push(p("\t\tsub.f32 %ftmp, %ftmp, %ftmax;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tadd.f32 %ftsum, %ftsum, %ftmp;"));
    lines.push(p("\t\tadd.u32 %r12, %r12, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r12, {vtile};"));
    lines.push(p("\t\t@%pv bra LP_RED_SUM;"));
    lines.push(String::new());

    // Store (tile_max, tile_sum_unscaled) as f32 partials.
    lines.push(p("\tst.global.f32 [%rd5],   %ftmax;"));
    lines.push(p("\tst.global.f32 [%rd5+4], %ftsum;"));
    lines.push(String::new());

    lines.push(p("LP_DONE:"));
    lines.push(p("\tret;"));
    lines.push(p("}"));

    lines.join("\n")
}

/// Bf16 finalize kernel.
///
/// Reads f32 partials (Kernel A writes f32 regardless of activation dtype),
/// runs online-LSE rescale in f32, recomputes `logit_at_target = x[row] @
/// W[tgt] + bias[tgt]` with bf16 staging + f32 fma accumulator, writes
/// `loss_out` and `lse_out` as f32.
fn emit_large_finalize_kernel_bf16(cfg: &FusedLinearCEConfig) -> String {
    let name = cfg.large_finalize_kernel_name();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let n_tiles = cfg.num_vocab_tiles();
    let ignore = cfg.ignore_index;

    let mut lines: Vec<String> = Vec::new();
    let p = |l: &str| l.to_owned();
    lines.push(String::new());

    lines.push(format!(".visible .entry {name}("));
    lines.push(p("\t.param .u64 param_x,"));
    lines.push(p("\t.param .u64 param_w,"));
    lines.push(p("\t.param .u64 param_bias,"));
    lines.push(p("\t.param .u64 param_targets,"));
    lines.push(p("\t.param .u64 param_partials,"));
    lines.push(p("\t.param .u64 param_loss_out,"));
    lines.push(p("\t.param .u64 param_lse_out,"));
    lines.push(p("\t.param .u32 param_B, .param .u32 param_S,"));
    lines.push(p("\t.param .u32 param_V, .param .u32 param_H,"));
    lines.push(p("\t.param .u32 param_num_tiles"));
    lines.push(p(") {"));

    lines.push(p("\t.reg .u64 %rd<24>;"));
    lines.push(p("\t.reg .u32 %r<16>;"));
    lines.push(p("\t.reg .s64 %tgt64;"));
    lines.push(p("\t.reg .b16 %h0, %h1, %h2;"));
    lines.push(p("\t.reg .f32 %fmax, %fsum, %ftmax, %ftsum, %fnew_max;"));
    lines.push(p("\t.reg .f32 %ftmp, %fa, %fb, %facc, %flog, %flse, %floss;"));
    lines.push(p("\t.reg .f32 %flog2e, %fln2;"));
    lines.push(p("\t.reg .pred %pskip, %pth0, %pv;"));
    lines.push(String::new());

    lines.push(p("\tld.param.u64 %rd0, [param_x];"));
    lines.push(p("\tld.param.u64 %rd1, [param_w];"));
    lines.push(p("\tld.param.u64 %rd2, [param_bias];"));
    lines.push(p("\tld.param.u64 %rd3, [param_targets];"));
    lines.push(p("\tld.param.u64 %rd4, [param_partials];"));
    lines.push(p("\tld.param.u64 %rd5, [param_loss_out];"));
    lines.push(p("\tld.param.u64 %rd6, [param_lse_out];"));
    lines.push(p("\tmov.u32 %r0, %ctaid.x;"));
    lines.push(p("\tmov.u32 %r1, %tid.x;"));
    lines.push(p("\tmov.f32 %flog2e, 0f3FB8AA3B;"));
    lines.push(p("\tmov.f32 %fln2,   0f3F317218;"));
    lines.push(String::new());

    lines.push(p("\tsetp.eq.u32 %pth0, %r1, 0;"));
    lines.push(p("\t@!%pth0 bra LF_DONE;"));
    lines.push(String::new());

    lines.push(p("\tcvt.u64.u32 %rd7, %r0;"));
    lines.push(p("\tmul.lo.u64 %rd7, %rd7, 8;"));
    lines.push(p("\tadd.u64 %rd7, %rd3, %rd7;"));
    lines.push(p("\tld.global.s64 %tgt64, [%rd7];"));
    lines.push(String::new());

    lines.push(p("\tcvt.u64.u32 %rd8, %r0;"));
    lines.push(p("\tshl.b64 %rd8, %rd8, 2;"));
    lines.push(p("\tadd.u64 %rd9, %rd5, %rd8;"));
    lines.push(p("\tadd.u64 %rd10, %rd6, %rd8;"));
    lines.push(String::new());

    lines.push(format!("\tsetp.eq.s64 %pskip, %tgt64, {ignore};"));
    lines.push(p("\t@!%pskip bra LF_REDUCE;"));
    lines.push(p("\tst.global.f32 [%rd9],  0f00000000;"));
    lines.push(p("\tst.global.f32 [%rd10], 0f00000000;"));
    lines.push(p("\tbra LF_DONE;"));
    lines.push(p("LF_REDUCE:"));
    lines.push(String::new());

    // partials_row_base = partials + row * num_tiles * 8 (f32 partials).
    lines.push(p("\tcvt.u64.u32 %rd11, %r0;"));
    lines.push(format!("\tmov.u32 %r2, {n_tiles};"));
    lines.push(p("\tcvt.u64.u32 %rd12, %r2;"));
    lines.push(p("\tmul.lo.u64 %rd11, %rd11, %rd12;"));
    lines.push(p("\tshl.b64 %rd11, %rd11, 3;"));
    lines.push(p("\tadd.u64 %rd11, %rd4, %rd11;"));
    lines.push(String::new());

    // Online-LSE reduce — identical to F32 path (partials are f32).
    lines.push(p("\tmov.f32 %fmax, 0f80800000;"));
    lines.push(p("\tmov.f32 %fsum, 0f00000000;"));
    lines.push(p("\tmov.u32 %r3, 0;"));
    lines.push(p("LF_LOOP:"));
    lines.push(p("\t\tcvt.u64.u32 %rd13, %r3;"));
    lines.push(p("\t\tshl.b64 %rd13, %rd13, 3;"));
    lines.push(p("\t\tadd.u64 %rd13, %rd11, %rd13;"));
    lines.push(p("\t\tld.global.f32 %ftmax, [%rd13];"));
    lines.push(p("\t\tld.global.f32 %ftsum, [%rd13+4];"));
    lines.push(String::new());

    lines.push(p("\t\tmax.f32 %fnew_max, %fmax, %ftmax;"));
    lines.push(p("\t\tsub.f32 %ftmp, %fmax, %fnew_max;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tmul.f32 %fsum, %fsum, %ftmp;"));
    lines.push(p("\t\tsub.f32 %ftmp, %ftmax, %fnew_max;"));
    lines.push(p("\t\tmul.f32 %ftmp, %ftmp, %flog2e;"));
    lines.push(p("\t\tex2.approx.f32 %ftmp, %ftmp;"));
    lines.push(p("\t\tmul.f32 %ftsum, %ftsum, %ftmp;"));
    lines.push(p("\t\tadd.f32 %fsum, %fsum, %ftsum;"));
    lines.push(p("\t\tmov.f32 %fmax, %fnew_max;"));
    lines.push(String::new());

    lines.push(p("\t\tadd.u32 %r3, %r3, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r3, {n_tiles};"));
    lines.push(p("\t\t@%pv bra LF_LOOP;"));
    lines.push(String::new());

    lines.push(p("\tlg2.approx.f32 %flse, %fsum;"));
    lines.push(p("\tmul.f32 %flse, %flse, %fln2;"));
    lines.push(p("\tadd.f32 %flse, %flse, %fmax;"));
    lines.push(String::new());

    // x_row_base — bf16 stride 2.
    lines.push(p("\tcvt.u64.u32 %rd14, %r0;"));
    lines.push(format!("\tmov.u32 %r4, {hidden};"));
    lines.push(p("\tcvt.u64.u32 %rd15, %r4;"));
    lines.push(p("\tmul.lo.u64 %rd14, %rd14, %rd15;"));
    lines.push(p("\tshl.b64 %rd14, %rd14, 1; // *2 bf16"));
    lines.push(p("\tadd.u64 %rd14, %rd0, %rd14;"));
    // W_tgt_base — bf16 stride 2.
    lines.push(p("\tmul.lo.s64 %rd16, %tgt64, %rd15;"));
    lines.push(p("\tshl.b64 %rd16, %rd16, 1; // *2 bf16"));
    lines.push(p("\tadd.u64 %rd16, %rd1, %rd16;"));
    lines.push(String::new());

    // Dot loop — bf16 → f32 → fma.
    lines.push(p("\tmov.f32 %facc, 0f00000000;"));
    lines.push(p("\tmov.u32 %r5, 0;"));
    lines.push(p("LF_DOT:"));
    lines.push(p("\t\tcvt.u64.u32 %rd17, %r5;"));
    lines.push(p("\t\tshl.b64 %rd17, %rd17, 1; // *2"));
    lines.push(p("\t\tadd.u64 %rd18, %rd14, %rd17;"));
    lines.push(p("\t\tld.global.b16 %h0, [%rd18];"));
    lines.push(p("\t\tcvt.f32.bf16 %fa, %h0;"));
    lines.push(p("\t\tadd.u64 %rd18, %rd16, %rd17;"));
    lines.push(p("\t\tld.global.b16 %h1, [%rd18];"));
    lines.push(p("\t\tcvt.f32.bf16 %fb, %h1;"));
    lines.push(p("\t\tfma.rn.f32 %facc, %fa, %fb, %facc;"));
    lines.push(p("\t\tadd.u32 %r5, %r5, 1;"));
    lines.push(format!("\t\tsetp.lt.u32 %pv, %r5, {hidden};"));
    lines.push(p("\t\t@%pv bra LF_DOT;"));
    lines.push(String::new());

    // Bias[tgt] bf16.
    lines.push(p("\tmul.lo.s64 %rd19, %tgt64, 2; // bf16 stride"));
    lines.push(p("\tadd.u64 %rd19, %rd2, %rd19;"));
    lines.push(p("\tld.global.b16 %h2, [%rd19];"));
    lines.push(p("\tcvt.f32.bf16 %ftmp, %h2;"));
    lines.push(p("\tadd.f32 %facc, %facc, %ftmp;"));
    lines.push(String::new());

    lines.push(p("\tsub.f32 %floss, %flse, %facc;"));
    lines.push(p("\tst.global.f32 [%rd9],  %floss;"));
    lines.push(p("\tst.global.f32 [%rd10], %flse;"));
    lines.push(String::new());

    lines.push(p("LF_DONE:"));
    let _ = vocab;
    lines.push(p("\tret;"));
    lines.push(p("}"));

    lines.join("\n")
}

// ── Bf16 backward kernel ──────────────────────────────────────────────────────
//
// Mixed-precision convention (Sprint v3-2):
//   * x / W / bias HBM loads are `ld.global.b16` + `cvt.f32.bf16` into f32
//     math registers. Backward recomputes logits from forward inputs and
//     the saved f32 lse, so the dtype of `x`/`W`/`bias` is the same as in
//     the forward kernel.
//   * The saved `lse` buffer stays `.f32` (written by the forward kernel
//     as f32 regardless of activation dtype) — `ld.global.f32 %lse_val`.
//   * The `grad_output` parameter is still `.param .f32` (a scalar; no
//     reason to halve a single value).
//   * Gradient outputs `dx`, `dW`, `dbias` stay `.f32` and the cross-CTA
//     accumulator uses `red.global.add.f32`. Rationale:
//       - `red.global.add.f16` is not portable across SMs (some pre-sm_70
//         lack it; sm_80+ supports it but adds a numerical-determinism
//         risk via non-deterministic accumulation order in bf16).
//       - PyTorch's standard mixed-precision convention writes master
//         gradients in f32; downstream optimizer state stays f32.
//       - Per the Sprint v3-2 spec: "current backward signature returns
//         f32 dW even when dtype=Bf16; this matches PyTorch's
//         mixed-precision convention".
//     The optional bf16 down-cast in an epilogue kernel is deferred.
//
// Output buffers dx/dW/dbias MUST be allocated by the caller as f32 even
// when `dtype = Bf16` — the runtime FFI layer threads this convention.
fn emit_bwd_kernel_bf16(cfg: &FusedLinearCEConfig) -> String {
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

    // SMEM not used by backward (forward stored everything it needs in HBM),
    // but the declaration is kept for ABI parity with the F32 path's launcher.
    s.push_str(&format!(
        ".extern .shared .align 2 .b8 smem_scratch[{smem_bytes}];\n\n"
    ));

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

    s.push_str(
        "\t.reg .u64 %rd<24>;\n\
         \t.reg .u32 %r<20>;\n\
         \t.reg .s64 %target_val;\n\
         \t.reg .b16 %h0, %h1, %h2;\n\
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

    s.push_str(
        "\tmov.u32 %r0, %ctaid.x;\n\
         \tmov.u32 %r1, %tid.x;\n\
    \n",
    );

    s.push_str(
        "\tcvt.u64.u32 %rd8, %r0;\n\
         \tmul.lo.u64 %rd8, %rd8, 8;\n\
         \tadd.u64 %rd8, %rd3, %rd8;\n\
         \tld.global.s64 %target_val, [%rd8];\n\
    \n",
    );

    s.push_str(&format!(
        "\tsetp.eq.s64 %p_skip, %target_val, {ignore};\n\
         \t@%p_skip bra BWD_SKIP_LABEL;\n\
    \n"
    ));

    // Saved lse stays f32 even at dtype=Bf16 (matches forward's f32 write).
    s.push_str(
        "\tcvt.u64.u32 %rd9, %r0;\n\
         \tshl.b64 %rd9, %rd9, 2;\n\
         \tadd.u64 %rd9, %rd4, %rd9;\n\
         \tld.global.f32 %lse_val, [%rd9];\n\
    \n",
    );

    // x_row_base + dx_row_base. x stride 2 (bf16); dx stride 4 (f32 output).
    s.push_str(&format!(
        "\tcvt.u64.u32 %rd10, %r0;\n\
         \tmov.u32 %r2, {hidden};\n\
         \tcvt.u64.u32 %rd11, %r2;\n\
         \tmul.lo.u64 %rd10, %rd10, %rd11;\n\
         \tshl.b64 %rd10, %rd10, 1; // x: *2 (bf16)\n\
         \tadd.u64 %rd10, %rd0, %rd10;\n\
         \t// dx_row_base = dx_out + row_idx * H * 4 (f32 grad)\n\
         \tcvt.u64.u32 %rd20, %r0;\n\
         \tmul.lo.u64 %rd20, %rd20, %rd11;\n\
         \tshl.b64 %rd20, %rd20, 2; // dx: *4 (f32)\n\
         \tadd.u64 %rd20, %rd5, %rd20;\n\
    \n"
    ));

    s.push_str("\tdiv.rn.f32 %scale, %grad_output, %num_valid_f;\n\n");
    s.push_str("\tmov.f32 %f15, 0f3FB8AA3B; // log2(e)\n\n");

    s.push_str(
        "\tmov.u32 %r3, 0; // tile_idx\n\
         BWD_TILE_LOOP:\n",
    );

    s.push_str(&format!(
        "\t\tmul.lo.u32 %r4, %r3, {vtile}; // v_base\n\
    \n"
    ));

    s.push_str(
        "\t\tmov.u32 %r5, 0; // inner counter\n\
         BWD_INNER_LOOP:\n",
    );

    s.push_str(
        "\t\t\tmul.lo.u32 %r6, %r5, 128;\n\
         \t\t\tadd.u32 %r6, %r6, %r1;\n\
         \t\t\tadd.u32 %r6, %r6, %r4;\n",
    );

    s.push_str(&format!(
        "\t\t\tsetp.lt.u32 %p_valid, %r6, {vocab};\n\
         \t\t\t@!%p_valid bra BWD_INNER_SKIP;\n\
    \n"
    ));

    // W_row_base — bf16 stride 2.
    s.push_str(&format!(
        "\t\t\t// W_row_base for v_idx (bf16)\n\
         \t\t\tcvt.u64.u32 %rd12, %r6;\n\
         \t\t\tmov.u32 %r7, {hidden};\n\
         \t\t\tcvt.u64.u32 %rd13, %r7;\n\
         \t\t\tmul.lo.u64 %rd12, %rd12, %rd13;\n\
         \t\t\tshl.b64 %rd12, %rd12, 1; // *2 bf16\n\
         \t\t\tadd.u64 %rd12, %rd1, %rd12;\n\
    \n"
    ));

    // Dot product — bf16 loads → cvt.f32.bf16 → fma.f32.
    s.push_str(
        "\t\t\tmov.f32 %logit_acc, 0f00000000;\n\
         \t\t\tmov.u32 %r8, 0;\n\
         BWD_DOT_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r8;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 1; // *2 bf16\n\
         \t\t\t\tadd.u64 %rd15, %rd10, %rd14;\n\
         \t\t\t\tld.global.b16 %h0, [%rd15];\n\
         \t\t\t\tcvt.f32.bf16 %f0, %h0;\n\
         \t\t\t\tadd.u64 %rd16, %rd12, %rd14;\n\
         \t\t\t\tld.global.b16 %h1, [%rd16];\n\
         \t\t\t\tcvt.f32.bf16 %f1, %h1;\n\
         \t\t\t\tfma.rn.f32 %logit_acc, %f0, %f1, %logit_acc;\n\
         \t\t\t\tadd.u32 %r8, %r8, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\t\tsetp.lt.u32 %p_valid, %r8, {hidden};\n\
         \t\t\t\t@%p_valid bra BWD_DOT_LOOP;\n\
    \n"
    ));

    // Add bias (bf16).
    s.push_str(
        "\t\t\t// bias (bf16)\n\
         \t\t\tcvt.u64.u32 %rd17, %r6;\n\
         \t\t\tshl.b64 %rd17, %rd17, 1; // *2\n\
         \t\t\tadd.u64 %rd17, %rd2, %rd17;\n\
         \t\t\tld.global.b16 %h2, [%rd17];\n\
         \t\t\tcvt.f32.bf16 %f2, %h2;\n\
         \t\t\tadd.f32 %logit_acc, %logit_acc, %f2;\n\
    \n",
    );

    // p_v = exp(logit_v - lse).
    s.push_str(
        "\t\t\tsub.f32 %f3, %logit_acc, %lse_val;\n\
         \t\t\tmul.f32 %f3, %f3, %f15;\n\
         \t\t\tex2.approx.f32 %f3, %f3;\n\
    \n",
    );

    s.push_str(
        "\t\t\tcvt.s64.u32 %rd18, %r6;\n\
         \t\t\tsetp.eq.s64 %p_is_target, %rd18, %target_val;\n\
         \t\t\t@%p_is_target sub.f32 %f3, %f3, 0f3F800000; // -= 1.0\n\
         \t\t\tmul.f32 %f4, %f3, %scale;\n\
    \n",
    );

    // dW_row_base — f32 stride 4 (dW is master-precision f32 regardless of activation dtype).
    s.push_str(&format!(
        "\t\t\t// dW_row_base = dW_out + v_idx * H * 4 (f32 master grad)\n\
         \t\t\tcvt.u64.u32 %rd21, %r6;\n\
         \t\t\tmov.u32 %r9, {hidden};\n\
         \t\t\tcvt.u64.u32 %rd22, %r9;\n\
         \t\t\tmul.lo.u64 %rd21, %rd21, %rd22;\n\
         \t\t\tshl.b64 %rd21, %rd21, 2; // *4 (f32)\n\
         \t\t\tadd.u64 %rd21, %rd6, %rd21;\n\
    \n"
    ));

    // H-loop scatter.
    //   W[v, h]  is bf16 (stride 2 from %rd12)
    //   x[row,h] is bf16 (stride 2 from %rd10)
    //   dx_out + dW_out are f32 (stride 4 from %rd20 / %rd21)
    s.push_str(
        "\t\t\tmov.u32 %r9, 0; // h counter\n\
         BWD_H_LOOP:\n\
         \t\t\t\tcvt.u64.u32 %rd23, %r9;\n\
         \t\t\t\tshl.b64 %rd23, %rd23, 1; // *2 bf16 (for W, x loads)\n\
         \t\t\t\t// W[v, h] bf16\n\
         \t\t\t\tadd.u64 %rd14, %rd12, %rd23;\n\
         \t\t\t\tld.global.b16 %h0, [%rd14];\n\
         \t\t\t\tcvt.f32.bf16 %f5, %h0;\n\
         \t\t\t\t// f6 = scaled * W[v, h]  (f32 grad slice)\n\
         \t\t\t\tmul.f32 %f6, %f4, %f5;\n\
         \t\t\t\t// dx_out[row, h] += f6  (f32 destination - stride 4)\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r9;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 2;\n\
         \t\t\t\tadd.u64 %rd14, %rd20, %rd14;\n\
         \t\t\t\tred.global.add.f32 [%rd14], %f6;\n\
         \t\t\t\t// x[row, h] bf16\n\
         \t\t\t\tadd.u64 %rd14, %rd10, %rd23;\n\
         \t\t\t\tld.global.b16 %h1, [%rd14];\n\
         \t\t\t\tcvt.f32.bf16 %f7, %h1;\n\
         \t\t\t\t// f8 = scaled * x[row, h]\n\
         \t\t\t\tmul.f32 %f8, %f4, %f7;\n\
         \t\t\t\t// dW_out[v, h] += f8  (f32 destination - stride 4)\n\
         \t\t\t\tcvt.u64.u32 %rd14, %r9;\n\
         \t\t\t\tshl.b64 %rd14, %rd14, 2;\n\
         \t\t\t\tadd.u64 %rd14, %rd21, %rd14;\n\
         \t\t\t\tred.global.add.f32 [%rd14], %f8;\n\
         \t\t\t\tadd.u32 %r9, %r9, 1;\n",
    );

    s.push_str(&format!(
        "\t\t\t\tsetp.lt.u32 %p_valid, %r9, {hidden};\n\
         \t\t\t\t@%p_valid bra BWD_H_LOOP;\n\
    \n"
    ));

    // dbias[v] += scaled (f32 output stride 4).
    s.push_str(
        "\t\t\t// dbias_out[v] += scaled (f32 output)\n\
         \t\t\tcvt.u64.u32 %rd14, %r6;\n\
         \t\t\tshl.b64 %rd14, %rd14, 2; // *4 f32\n\
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

    s.push_str("\t\tadd.u32 %r3, %r3, 1;\n");

    s.push_str(&format!(
        "\t\tsetp.lt.u32 %p_valid, %r3, {n_tiles};\n\
         \t\t@%p_valid bra BWD_TILE_LOOP;\n\
    \n"
    ));

    s.push_str("\tbra BWD_DONE;\n\n");

    // Skip path: zero dx_out[row, :] as f32 (stride 4).
    s.push_str(
        "BWD_SKIP_LABEL:\n\
         \t// Zero dx_out[row, :] for skipped token (f32 output)\n",
    );

    s.push_str(&format!(
        "\tcvt.u64.u32 %rd10, %r0;\n\
         \tmov.u32 %r2, {hidden};\n\
         \tcvt.u64.u32 %rd11, %r2;\n\
         \tmul.lo.u64 %rd10, %rd10, %rd11;\n\
         \tshl.b64 %rd10, %rd10, 2; // *4 (f32 dx)\n\
         \tadd.u64 %rd10, %rd5, %rd10;\n\
    \n"
    ));

    s.push_str(&format!(
        "\tmov.u32 %r5, 0;\n\
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

    #[test]
    fn test_validate_rejects_above_hard_ceiling() {
        let mut cfg = default_cfg();
        // Raise per-config cap so we don't hit the legacy gate first.
        cfg.max_vocab_v1 = u32::MAX;
        cfg.vocab_size = MAX_VOCAB_HARD_CEILING + 1;
        cfg.vocab_tile = 128;
        let err = cfg.validate().expect_err("above hard ceiling must reject");
        assert!(err.contains("hard ceiling"), "{err}");
    }

    #[test]
    fn test_is_large_vocab_predicate_matches_threshold() {
        let mut cfg = default_cfg();
        cfg.max_vocab_v1 = MAX_VOCAB_HARD_CEILING;

        cfg.vocab_size = LARGE_VOCAB_THRESHOLD;
        cfg.vocab_tile = 128;
        assert!(!cfg.is_large_vocab(), "AT threshold must be small-vocab (v1 path)");

        cfg.vocab_size = LARGE_VOCAB_THRESHOLD + 128;
        assert!(cfg.is_large_vocab(), "above threshold must route to large-vocab path");
    }

    #[test]
    fn test_validate_accepts_large_vocab_when_cap_raised() {
        let cfg = FusedLinearCEConfig {
            vocab_size: 49152,
            hidden_size: 128,
            seq_len: 64,
            batch_size: 2,
            vocab_tile: 128,
            gpu_sm: 80,
            dtype: Dtype::F32,
            ignore_index: -100,
            max_vocab_v1: MAX_VOCAB_HARD_CEILING,
        };
        cfg.validate().expect("vocab=49152 with raised cap MUST validate");
        assert!(cfg.is_large_vocab());
    }

    #[test]
    fn test_num_vocab_tiles_div_ceil() {
        let mut cfg = default_cfg();
        cfg.vocab_size = 49152;
        cfg.vocab_tile = 128;
        cfg.max_vocab_v1 = MAX_VOCAB_HARD_CEILING;
        assert_eq!(cfg.num_vocab_tiles(), 49152 / 128);

        // Non-divisible: rounds up.
        cfg.vocab_size = 49153;
        assert_eq!(cfg.num_vocab_tiles(), (49153 + 127) / 128);
    }

    #[test]
    fn test_large_partials_bytes_matches_formula() {
        let cfg = FusedLinearCEConfig {
            vocab_size: 49152,
            hidden_size: 128,
            seq_len: 64,
            batch_size: 2,
            vocab_tile: 128,
            gpu_sm: 80,
            dtype: Dtype::F32,
            ignore_index: -100,
            max_vocab_v1: MAX_VOCAB_HARD_CEILING,
        };
        // (B*S) * num_tiles * 2 floats * 4 bytes = 128 * 384 * 8 = 393_216
        assert_eq!(cfg.large_partials_bytes(), 128 * 384 * 8);
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

    // ── Large-vocab two-kernel PTX synthesis (structural) ────────────────

    fn large_vocab_cfg() -> FusedLinearCEConfig {
        FusedLinearCEConfig {
            vocab_size: 49152,
            hidden_size: 128,
            seq_len: 64,
            batch_size: 2,
            vocab_tile: 128,
            gpu_sm: 80,
            dtype: Dtype::F32,
            ignore_index: -100,
            max_vocab_v1: MAX_VOCAB_HARD_CEILING,
        }
    }

    #[test]
    fn test_large_ptx_contains_both_kernel_entries() {
        let cfg = large_vocab_cfg();
        cfg.validate().unwrap();
        assert!(cfg.is_large_vocab());

        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();

        // Kernel A entry.
        let kname_a = cfg.large_partials_kernel_name();
        assert!(
            ptx.contains(&format!(".visible .entry {kname_a}(")),
            "missing Kernel A entry: {kname_a}"
        );
        // Kernel B entry.
        let kname_b = cfg.large_finalize_kernel_name();
        assert!(
            ptx.contains(&format!(".visible .entry {kname_b}(")),
            "missing Kernel B entry: {kname_b}"
        );
    }

    #[test]
    fn test_large_ptx_header_emitted_once() {
        let cfg = large_vocab_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        // Exactly one .version line + one .target line at module scope.
        assert_eq!(
            ptx.matches(".version 7.0").count(),
            1,
            ".version must appear exactly once at module scope"
        );
        assert_eq!(
            ptx.matches(".target sm_80").count(),
            1,
            ".target must appear exactly once at module scope"
        );
    }

    #[test]
    fn test_large_ptx_contains_skip_predicate() {
        let cfg = large_vocab_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        assert!(ptx.contains("setp.eq.s64"));
        assert!(ptx.contains("-100"));
    }

    #[test]
    fn test_large_ptx_partials_pointer_arithmetic_is_present() {
        let cfg = large_vocab_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).unwrap();
        // Both kernels must dereference partials with an 8-byte stride
        // (one f32 pair per (row, tile) slot). The store in Kernel A
        // writes ftmax then ftsum at +0 and +4.
        assert!(ptx.contains("st.global.f32 [%rd5],   %ftmax;"));
        assert!(ptx.contains("st.global.f32 [%rd5+4], %ftsum;"));
        // The load in Kernel B reads ftmax then ftsum at +0 and +4.
        assert!(ptx.contains("ld.global.f32 %ftmax, [%rd13];"));
        assert!(ptx.contains("ld.global.f32 %ftsum, [%rd13+4];"));
    }

    #[test]
    fn test_large_ptx_is_ascii_only() {
        let cfg = large_vocab_cfg();
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        for b in bytes.iter() {
            assert!(*b < 128, "non-ASCII byte 0x{b:02x} in large-vocab PTX");
        }
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
