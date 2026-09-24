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
//! single-CTA kernel is emitted, computing exactly what it did before
//! Sprint 3 (see the KIR section below for how that is proved now).
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
//! Both paths are research-grade scalar `fma.rn.f32`, accumulating in f32
//! whatever the storage dtype. MMA tiling and quantised-W paths are
//! deferred.
//!
//! ## KIR (roadmap A2 step 10)
//!
//! The kernels were hand-assembled PTX text and moved onto [`KernelIR`],
//! one role at a time with all three dtypes from one builder; the file no
//! longer writes PTX text. The v1 forward: [`build_forward`] builds it, keeping the hand
//! kernels' control flow, barriers and floating-point order, and
//! `tests/fused_linear_ce_fwd_kir_equivalence.rs` runs the frozen hand
//! emitters (`tests/fixtures/fused_linear_ce_hand.rs`) against it on the PTX
//! interpreter and requires the same output bits, and an f64 reference's
//! answer. It targets the KIR floor (`sm_70`; bf16 raises it to `sm_80` and
//! ISA 7.8), so `gpu_sm` no longer reaches it. The large-vocab pair has
//! moved as well: [`build_large_partials`] and [`build_large_finalize`],
//! lowered into one module under one header, proved by
//! `tests/fused_linear_ce_large_kir_equivalence.rs` over Kernel A's
//! two-dimensional grid then Kernel B. The backward, [`build_backward`],
//! scatters with `red.global.add.f32` and is proved by
//! `tests/fused_linear_ce_bwd_kir_equivalence.rs`, which runs the
//! reductions in the interpreter's schedule order.
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

use crate::backend_ptx::{lower_kir_module_to_ptx, lower_kir_to_ptx};
use crate::cfie_decode_attention::{at, cmp, konst, load, op2, ptr, widen};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, SmemLayout, SmemRegion, VarId,
};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Vocab sizes at or below this use the v1 single-CTA path. Above this, the
/// Sprint-3 two-kernel large-vocab path activates (per-tile partials +
/// per-row finalize).
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
    /// v1 single-CTA path.
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
}

// ─── Forward kernel synthesis ────────────────────────────────────────────────

/// Synthesise the forward PTX for the fused linear-CE kernel.
///
/// **Routing**: when `cfg.is_large_vocab()` is `false` (vocab_size ≤ 8192),
/// this returns the v1 single-CTA kernel ([`build_forward`], lowered).
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
    // Contract: returned bytes are null-terminated, matching the convention
    // established by `backend_ptx::lower_kir_to_ptx`. The CUDA driver entry
    // point `cuModuleLoadData` reads the PTX module bytes as a C string until
    // it hits a `0` byte — passing a non-terminated buffer is UB (the driver
    // reads past the buffer end). The large-vocab branch null-terminates
    // itself; only the small-vocab branch needs the explicit push here.
    if cfg.is_large_vocab() {
        synthesize_large_vocab_forward_ptx(cfg)
    } else {
        // Roadmap A2 step 10: one KIR builder for all three dtypes; the
        // printer's output is already null-terminated.
        emit_forward(cfg)
    }
}

/// Synthesise a single PTX module containing both Kernel A (per-tile
/// partials) and Kernel B (per-row finalize) of the Sprint-3 large-vocab
/// two-kernel forward path.
///
/// Both kernels ([`build_large_partials`], [`build_large_finalize`]) share
/// one `.version` / `.target` header; Kernel A's tile is the module's one
/// dynamic shared block (Kernel B uses none, and is launched with 0), so
/// the module compiles as one ptxas TU.
pub fn synthesize_large_vocab_forward_ptx(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    // Roadmap A2 step 10: one KIR builder per kernel for all three dtypes;
    // the module printer's output is already null-terminated.
    emit_large_forward(cfg)
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
    // Roadmap A2 step 10: one KIR builder for all three dtypes; the
    // printer's output is already null-terminated.
    emit_backward(cfg)
}

// ─── KIR — v1 forward ────────────────────────────────────────────────────────
//
// Roadmap A2 step 10: the v1 single-CTA forward kernel is built as KIR, one
// builder for all three dtypes. It keeps the hand kernels' control flow,
// barriers and floating-point order (`tests/fused_linear_ce_fwd_kir_equivalence.rs`
// runs the frozen hand emitters against it on the PTX interpreter and
// requires the same output bits). Dynamic shared memory is a two-region
// `SmemLayout` at the hand kernels' offsets: the logits tile in the storage
// dtype (so a 16-bit logit is rounded to it before the reduction, as the hand
// kernels did) and the logit-at-target slot after it.

/// Threads per CTA, and the stride of the v1 tile fill.
const V1_BLOCK: u32 = 128;
/// Index of each shared region in [`v1_forward_smem`].
const R_LOGITS: u32 = 0;
const R_TARGET: u32 = 1;

/// The storage type of `x`, `W`, `bias` and the shared logits tile.
fn elem_type(dtype: Dtype) -> KirType {
    match dtype {
        Dtype::F32 => KirType::F32,
        Dtype::F16 => KirType::F16,
        Dtype::Bf16 => KirType::Bf16,
    }
}

/// `[logits: vocab_tile][logit_at_target: 1]`, both in the storage dtype,
/// dynamic (the launcher passes [`FusedLinearCEConfig::shared_mem_bytes`],
/// which covers it).
fn v1_forward_smem(cfg: &FusedLinearCEConfig) -> SmemLayout {
    let elem = elem_type(cfg.dtype);
    let bytes = cfg.dtype.bytes_per_elem();
    let region = |name: &str, elems: u32| SmemRegion {
        name: name.to_string(),
        bytes: elems * bytes,
        align: bytes,
        elem: elem.clone(),
    };
    SmemLayout { regions: vec![region("logits", cfg.vocab_tile), region("logit_at_target", 1)], dynamic: true }
}

fn i64_const(b: &mut KirBuilder, v: i64) -> VarId {
    let dst = b.new_typed_var(KirType::I64);
    b.emit(KirOp::Const(dst, KirConst { ty: KirType::I64, value: ConstValue::I64(v) }));
    dst
}

/// A storage-dtype element as f32: loaded, and widened unless it is f32.
fn load_elem(b: &mut KirBuilder, dtype: Dtype, addr: VarId, space: AddressSpace) -> VarId {
    let raw = load(b, elem_type(dtype), addr, space);
    if dtype == Dtype::F32 {
        return raw;
    }
    let wide = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(wide, raw, KirType::F32));
    wide
}

/// An f32 value in the storage dtype: rounded to nearest unless it is f32.
fn to_elem(b: &mut KirBuilder, dtype: Dtype, v: VarId) -> VarId {
    if dtype == Dtype::F32 {
        return v;
    }
    let ty = elem_type(dtype);
    let narrow = b.new_typed_var(ty.clone());
    b.emit(KirOp::Cast(narrow, v, ty));
    narrow
}

/// `for (i = start; i < end; i += step) body(i)`, entered from the current
/// block; the builder is left in the loop's exit block.
fn counted_loop(
    b: &mut KirBuilder,
    (start, end, step): (VarId, VarId, VarId),
    body: impl FnOnce(&mut KirBuilder, VarId),
) {
    let head = b.new_block();
    let body_block = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, KirType::U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![start])));

    b.set_block(head);
    let finished = cmp(b, i, end, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body_block)));

    b.set_block(body_block);
    body(b, i);
    let next = op2(b, KirType::U32, KirOp::Add, i, step);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![next])));

    b.set_block(done);
}

/// Build the v1 single-CTA forward kernel (vocab ≤ 8192) as KIR.
///
/// One CTA per token row, 128 threads. The CFG, in the hand kernels' order:
///
/// ```text
/// entry        thread 0 stores -inf to the logit-at-target slot   bar
///              target = targets[row]; target == ignore_index ? skip : body
/// body         per vocab tile (tile, max, sum):
///                per sub-tile j: v = tile*vtile + j*128 + tid; v < V ?
///                  logit = fma-dot(x[row], W[v]) + bias[v]
///                  logits[j*128 + tid] = elem(logit)
///                  v == target ? logit_at_target = elem(logit)
///                bar
///                thread 0: tile max (first -inf), rescale, tile sum
///                bar
///              thread 0: lse = log(sum) + max; loss = lse - logit_at_target
///              loss_out[row] = loss; lse_out[row] = lse
/// skip         thread 0: loss_out[row] = lse_out[row] = 0
/// ```
///
/// Only thread 0's running max and sum are meaningful; the other threads
/// carry their initial values through the tile loop, as the hand kernels'
/// untouched registers did.
pub fn build_forward(cfg: &FusedLinearCEConfig) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, I64, U32, U64};

    assert!(!cfg.is_large_vocab(), "the v1 forward serves vocab <= {LARGE_VOCAB_THRESHOLD}");
    let dtype = cfg.dtype;
    let elem = elem_type(dtype);
    let n_tiles = cfg.vocab_size.div_ceil(cfg.vocab_tile);

    let mut b = KirBuilder::new(&cfg.kernel_name());
    // The params, in FFI order (the launcher marshals them positionally).
    // B, S, V and H are baked; the params stay for the launch ABI.
    let x = b.add_param("x", ptr(elem.clone(), Global), Global);
    let w = b.add_param("w", ptr(elem.clone(), Global), Global);
    let bias = b.add_param("bias", ptr(elem.clone(), Global), Global);
    let targets = b.add_param("targets", ptr(I64, Global), Global);
    let loss_out = b.add_param("loss_out", ptr(F32, Global), Global);
    let lse_out = b.add_param("lse_out", ptr(F32, Global), Global);
    for name in ["B", "S", "V", "H"] {
        b.add_param(name, U32, Global);
    }
    b.set_smem_layout(v1_forward_smem(cfg));
    b.set_workgroup_size([V1_BLOCK, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let row = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(row, 0));
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let region = |b: &mut KirBuilder, r: u32| {
        let dst = b.new_typed_var(ptr(elem.clone(), Shared));
        b.emit(KirOp::SharedRegion { dst, region: r });
        dst
    };
    let logits = region(&mut b, R_LOGITS);
    let target_slot = region(&mut b, R_TARGET);
    let f_neg_inf = konst(&mut b, ConstValue::F32(f32::NEG_INFINITY));
    let f_zero = konst(&mut b, ConstValue::F32(0.0));

    // Thread 0 initialises the logit-at-target slot to -inf.
    let init = b.new_block();
    let init_done = b.new_block();
    let is_thread0 = cmp(&mut b, tid, zero, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(is_thread0, KirEdge::to(init), KirEdge::to(init_done)));
    b.set_block(init);
    let sentinel = to_elem(&mut b, dtype, f_neg_inf);
    b.emit(KirOp::Store(target_slot, sentinel, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(init_done)));

    b.set_block(init_done);
    b.emit(KirOp::Barrier);
    let target_addr = at(&mut b, I64, Global, targets, row);
    let target = load(&mut b, I64, target_addr, Global);
    let ignore = i64_const(&mut b, cfg.ignore_index);
    let skips = cmp(&mut b, target, ignore, CmpOp::Eq);
    let body = b.new_block();
    let skip = b.new_block();
    let exit = b.new_block();
    b.terminate(KirTerminator::CondBranch(skips, KirEdge::to(skip), KirEdge::to(body)));

    // ── the vocab tile loop ─────────────────────────────────────────────
    b.set_block(body);
    let hidden = konst(&mut b, ConstValue::U32(cfg.hidden_size));
    let hidden_wide = konst(&mut b, ConstValue::U64(cfg.hidden_size as u64));
    let vocab = konst(&mut b, ConstValue::U32(cfg.vocab_size));
    let vtile = konst(&mut b, ConstValue::U32(cfg.vocab_tile));
    let block = konst(&mut b, ConstValue::U32(V1_BLOCK));
    let row_wide = widen(&mut b, row);
    let x_row = op2(&mut b, U64, KirOp::Mul, row_wide, hidden_wide);

    let tile_head = b.new_block();
    let tile_body = b.new_block();
    let tiles_done = b.new_block();
    let tile = b.add_block_param(tile_head, U32);
    let run_max = b.add_block_param(tile_head, F32);
    let run_sum = b.add_block_param(tile_head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, vec![zero, f_neg_inf, f_zero])));

    b.set_block(tile_head);
    let n_tiles_c = konst(&mut b, ConstValue::U32(n_tiles));
    let tiles_finished = cmp(&mut b, tile, n_tiles_c, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(tiles_finished, KirEdge::to(tiles_done), KirEdge::to(tile_body)));

    b.set_block(tile_body);
    let v_base = op2(&mut b, U32, KirOp::Mul, tile, vtile);
    let per_thread = konst(&mut b, ConstValue::U32(cfg.vocab_tile / V1_BLOCK));
    counted_loop(&mut b, (zero, per_thread, one), |b, j| {
        let lane_base = op2(b, U32, KirOp::Mul, j, block);
        let slot = op2(b, U32, KirOp::Add, lane_base, tid);
        let v = op2(b, U32, KirOp::Add, slot, v_base);
        let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
        let fill = b.new_block();
        let fill_done = b.new_block();
        b.terminate(KirTerminator::CondBranch(in_vocab, KirEdge::to(fill), KirEdge::to(fill_done)));

        b.set_block(fill);
        let v_wide = widen(b, v);
        let w_row = op2(b, U64, KirOp::Mul, v_wide, hidden_wide);
        // logit = sum_h fma(x[row, h], W[v, h], acc)
        let dot_head = b.new_block();
        let dot_body = b.new_block();
        let dot_done = b.new_block();
        let h = b.add_block_param(dot_head, U32);
        let acc = b.add_block_param(dot_head, F32);
        b.terminate(KirTerminator::Branch(KirEdge::with(dot_head, vec![zero, f_zero])));
        b.set_block(dot_head);
        let dot_finished = cmp(b, h, hidden, CmpOp::Ge);
        b.terminate(KirTerminator::CondBranch(dot_finished, KirEdge::to(dot_done), KirEdge::to(dot_body)));
        b.set_block(dot_body);
        let h_wide = widen(b, h);
        let x_index = op2(b, U64, KirOp::Add, x_row, h_wide);
        let x_addr = at(b, elem.clone(), Global, x, x_index);
        let xv = load_elem(b, dtype, x_addr, Global);
        let w_index = op2(b, U64, KirOp::Add, w_row, h_wide);
        let w_addr = at(b, elem.clone(), Global, w, w_index);
        let wv = load_elem(b, dtype, w_addr, Global);
        let acc_next = b.new_typed_var(F32);
        b.emit(KirOp::Fma(acc_next, xv, wv, acc));
        let h_next = op2(b, U32, KirOp::Add, h, one);
        b.terminate(KirTerminator::Branch(KirEdge::with(dot_head, vec![h_next, acc_next])));

        b.set_block(dot_done);
        let bias_addr = at(b, elem.clone(), Global, bias, v);
        let bias_v = load_elem(b, dtype, bias_addr, Global);
        let logit = op2(b, F32, KirOp::Add, acc, bias_v);
        let stored = to_elem(b, dtype, logit);
        let logit_addr = at(b, elem.clone(), Shared, logits, slot);
        b.emit(KirOp::Store(logit_addr, stored, Shared));
        // Only the thread holding the target's column writes the slot.
        let v_signed = b.new_typed_var(I64);
        b.emit(KirOp::Cast(v_signed, v, I64));
        let is_target = cmp(b, v_signed, target, CmpOp::Eq);
        let record = b.new_block();
        b.terminate(KirTerminator::CondBranch(is_target, KirEdge::to(record), KirEdge::to(fill_done)));
        b.set_block(record);
        b.emit(KirOp::Store(target_slot, stored, Shared));
        b.terminate(KirTerminator::Branch(KirEdge::to(fill_done)));

        b.set_block(fill_done);
    });
    // Every logit of the tile is in shared memory.
    b.emit(KirOp::Barrier);

    // Thread 0: the tile's max, the rescale, the tile's sum.
    let reduce = b.new_block();
    let reduce_done = b.new_block();
    let max_next = b.add_block_param(reduce_done, F32);
    let sum_next = b.add_block_param(reduce_done, F32);
    let not_thread0 = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        not_thread0,
        KirEdge::with(reduce_done, vec![run_max, run_sum]),
        KirEdge::to(reduce),
    ));

    b.set_block(reduce);
    // A scan over the tile's first `vtile` entries that stops at the vocab.
    let scan = |b: &mut KirBuilder, acc0: VarId, step: &dyn Fn(&mut KirBuilder, VarId, VarId) -> VarId| -> VarId {
        let head = b.new_block();
        let body = b.new_block();
        let done = b.new_block();
        let i = b.add_block_param(head, U32);
        let acc = b.add_block_param(head, F32);
        let out = b.add_block_param(done, F32);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, acc0])));
        b.set_block(head);
        let v = op2(b, U32, KirOp::Add, v_base, i);
        let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
        let in_tile = b.new_block();
        b.terminate(KirTerminator::CondBranch(in_vocab, KirEdge::to(in_tile), KirEdge::with(done, vec![acc])));
        b.set_block(in_tile);
        let tile_open = cmp(b, i, vtile, CmpOp::Lt);
        b.terminate(KirTerminator::CondBranch(tile_open, KirEdge::to(body), KirEdge::with(done, vec![acc])));
        b.set_block(body);
        let addr = at(b, elem.clone(), Shared, logits, i);
        let s = load_elem(b, dtype, addr, Shared);
        let acc_next = step(b, acc, s);
        let i_next = op2(b, U32, KirOp::Add, i, one);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, acc_next])));
        b.set_block(done);
        out
    };
    let tile_max = scan(&mut b, f_neg_inf, &|b, m, s| op2(b, F32, KirOp::Max, m, s));
    let new_max = op2(&mut b, F32, KirOp::Max, run_max, tile_max);
    let shift = op2(&mut b, F32, KirOp::Sub, run_max, new_max);
    let rescale = b.new_typed_var(F32);
    b.emit(KirOp::Exp(rescale, shift));
    let rescaled = op2(&mut b, F32, KirOp::Mul, run_sum, rescale);
    let tile_sum = scan(&mut b, rescaled, &|b, acc, s| {
        let d = op2(b, F32, KirOp::Sub, s, new_max);
        let e = b.new_typed_var(F32);
        b.emit(KirOp::Exp(e, d));
        op2(b, F32, KirOp::Add, acc, e)
    });
    b.terminate(KirTerminator::Branch(KirEdge::with(reduce_done, vec![new_max, tile_sum])));

    // The logits tile is refilled next tile.
    b.set_block(reduce_done);
    b.emit(KirOp::Barrier);
    let tile_next = op2(&mut b, U32, KirOp::Add, tile, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, vec![tile_next, max_next, sum_next])));

    // ── thread 0 writes loss and lse ────────────────────────────────────
    b.set_block(tiles_done);
    let write = b.new_block();
    let not_writer = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_writer, KirEdge::to(exit), KirEdge::to(write)));
    b.set_block(write);
    let log_sum = b.new_typed_var(F32);
    b.emit(KirOp::Log(log_sum, run_sum));
    let lse = op2(&mut b, F32, KirOp::Add, log_sum, run_max);
    let logit_at_target = load_elem(&mut b, dtype, target_slot, Shared);
    let loss = op2(&mut b, F32, KirOp::Sub, lse, logit_at_target);
    let loss_addr = at(&mut b, F32, Global, loss_out, row);
    b.emit(KirOp::Store(loss_addr, loss, Global));
    let lse_addr = at(&mut b, F32, Global, lse_out, row);
    b.emit(KirOp::Store(lse_addr, lse, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // ── an ignored row: thread 0 writes zeros ───────────────────────────
    b.set_block(skip);
    let zero_write = b.new_block();
    let not_zeroer = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zeroer, KirEdge::to(exit), KirEdge::to(zero_write)));
    b.set_block(zero_write);
    let loss_addr = at(&mut b, F32, Global, loss_out, row);
    b.emit(KirOp::Store(loss_addr, f_zero, Global));
    let lse_addr = at(&mut b, F32, Global, lse_out, row);
    b.emit(KirOp::Store(lse_addr, f_zero, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// The v1 forward module: [`build_forward`], verified and lowered
/// (null-terminated for `cuModuleLoadData`).
///
/// # Panics
///
/// If the built kernel fails KIR verification — a bug in this module, not a
/// condition a caller can provoke.
fn emit_forward(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    let ir = build_forward(cfg);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    lower_kir_to_ptx(&ir)
}

// ─── KIR — large-vocab forward ───────────────────────────────────────────────
//
// Roadmap A2 step 10, second slice: the two large-vocab kernels are built as
// KIR too, each from one builder for all three dtypes, and lowered into one
// module under one header (`lower_kir_module_to_ptx`).
// `tests/fused_linear_ce_large_kir_equivalence.rs` runs Kernel A over every
// (tile, row) and Kernel B over every row, hand and KIR, on the PTX
// interpreter and requires the same output bits.

/// The per-tile logits of Kernel A, in the storage dtype, dynamic (the
/// launcher passes [`FusedLinearCEConfig::shared_mem_bytes`], which covers
/// it).
fn partials_smem(cfg: &FusedLinearCEConfig) -> SmemLayout {
    let bytes = cfg.dtype.bytes_per_elem();
    let logits = SmemRegion {
        name: "logits".to_string(),
        bytes: cfg.vocab_tile * bytes,
        align: bytes,
        elem: elem_type(cfg.dtype),
    };
    SmemLayout { regions: vec![logits], dynamic: true }
}

/// `sum_h fma(a[a_row + h], b[b_row + h], acc)` over `h < hidden`, from
/// zero, in index order — the hand kernels' dot. `a_row` and `b_row` are
/// 64-bit element offsets, each `U64` or `I64`.
fn fma_dot(
    b: &mut KirBuilder,
    dtype: Dtype,
    (a, a_row): (VarId, VarId),
    (bm, b_row): (VarId, VarId),
    hidden: u32,
) -> VarId {
    use AddressSpace::Global;
    use KirType::{F32, U32};
    let elem = elem_type(dtype);
    let zero = konst(b, ConstValue::U32(0));
    let one = konst(b, ConstValue::U32(1));
    let f_zero = konst(b, ConstValue::F32(0.0));
    let hidden = konst(b, ConstValue::U32(hidden));
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let h = b.add_block_param(head, U32);
    let acc = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, f_zero])));
    b.set_block(head);
    let finished = cmp(b, h, hidden, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));
    b.set_block(body);
    let index = |b: &mut KirBuilder, row: VarId| {
        let ty = b.var_type(row).expect("a typed row offset");
        let h_wide = b.new_typed_var(ty.clone());
        b.emit(KirOp::Cast(h_wide, h, ty.clone()));
        op2(b, ty, KirOp::Add, row, h_wide)
    };
    let a_index = index(b, a_row);
    let a_addr = at(b, elem.clone(), Global, a, a_index);
    let av = load_elem(b, dtype, a_addr, Global);
    let b_index = index(b, b_row);
    let b_addr = at(b, elem, Global, bm, b_index);
    let bv = load_elem(b, dtype, b_addr, Global);
    let acc_next = b.new_typed_var(F32);
    b.emit(KirOp::Fma(acc_next, av, bv, acc));
    let h_next = op2(b, U32, KirOp::Add, h, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![h_next, acc_next])));
    b.set_block(done);
    acc
}

/// Build Kernel A, the per-tile partials, as KIR.
///
/// Grid `(num_tiles, B*S)`, 128 threads: CTA `(tile, row)` writes the
/// tile's `(max, sum_v exp(logit_v - max))` to `partials[row][tile]`.
///
/// ```text
/// entry     target = targets[row]; target == ignore_index ? skip : body
/// body      per sub-tile j: slot = j*128 + tid; v = tile*vtile + slot
///             logits[slot] = elem(v < V ? fma-dot(x[row], W[v]) + bias[v] : -inf)
///           bar
///           thread 0: max over the tile's lanes (from -inf), then the
///             sum of exp(lane - max) (from 0); partials[row][tile] = (max, sum)
/// skip      thread 0: partials[row][tile] = (0, 0)
/// ```
///
/// A lane past the vocab holds `-inf`, which neither the max nor the sum
/// sees; so unlike the v1 kernel's, the scans run over the whole tile.
pub fn build_large_partials(cfg: &FusedLinearCEConfig) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, I64, U32, U64};

    let dtype = cfg.dtype;
    let elem = elem_type(dtype);
    let mut b = KirBuilder::new(&cfg.large_partials_kernel_name());
    // The params, in FFI order. B, S, V, H and num_tiles are baked.
    let x = b.add_param("x", ptr(elem.clone(), Global), Global);
    let w = b.add_param("w", ptr(elem.clone(), Global), Global);
    let bias = b.add_param("bias", ptr(elem.clone(), Global), Global);
    let targets = b.add_param("targets", ptr(I64, Global), Global);
    let partials = b.add_param("partials", ptr(F32, Global), Global);
    for name in ["B", "S", "V", "H", "num_tiles"] {
        b.add_param(name, U32, Global);
    }
    b.set_smem_layout(partials_smem(cfg));
    b.set_workgroup_size([V1_BLOCK, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let row = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(row, 1));
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let tile = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(tile, 0));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let f_zero = konst(&mut b, ConstValue::F32(0.0));
    let f_neg_inf = konst(&mut b, ConstValue::F32(f32::NEG_INFINITY));
    // This CTA's (max, sum) pair: partials[(row * num_tiles + tile) * 2].
    let n_tiles = konst(&mut b, ConstValue::U64(cfg.num_vocab_tiles() as u64));
    let row_wide = widen(&mut b, row);
    let row_first = op2(&mut b, U64, KirOp::Mul, row_wide, n_tiles);
    let tile_wide = widen(&mut b, tile);
    let pair_index = op2(&mut b, U64, KirOp::Add, row_first, tile_wide);
    let two = konst(&mut b, ConstValue::U64(2));
    let first = op2(&mut b, U64, KirOp::Mul, pair_index, two);
    let max_slot = at(&mut b, F32, Global, partials, first);
    let sum_slot = at(&mut b, F32, Global, max_slot, one);

    let target_addr = at(&mut b, I64, Global, targets, row);
    let target = load(&mut b, I64, target_addr, Global);
    let ignore = i64_const(&mut b, cfg.ignore_index);
    let skips = cmp(&mut b, target, ignore, CmpOp::Eq);
    let body = b.new_block();
    let skip = b.new_block();
    let exit = b.new_block();
    b.terminate(KirTerminator::CondBranch(skips, KirEdge::to(skip), KirEdge::to(body)));

    // ── the tile fill ───────────────────────────────────────────────────
    b.set_block(body);
    let logits = b.new_typed_var(ptr(elem.clone(), Shared));
    b.emit(KirOp::SharedRegion { dst: logits, region: 0 });
    let hidden_wide = konst(&mut b, ConstValue::U64(cfg.hidden_size as u64));
    let x_row = op2(&mut b, U64, KirOp::Mul, row_wide, hidden_wide);
    let vtile = konst(&mut b, ConstValue::U32(cfg.vocab_tile));
    let v_base = op2(&mut b, U32, KirOp::Mul, tile, vtile);
    let vocab = konst(&mut b, ConstValue::U32(cfg.vocab_size));
    let block = konst(&mut b, ConstValue::U32(V1_BLOCK));
    let per_thread = konst(&mut b, ConstValue::U32(cfg.vocab_tile / V1_BLOCK));
    counted_loop(&mut b, (zero, per_thread, one), |b, j| {
        let lane_base = op2(b, U32, KirOp::Mul, j, block);
        let slot = op2(b, U32, KirOp::Add, lane_base, tid);
        let v = op2(b, U32, KirOp::Add, slot, v_base);
        let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
        let fill = b.new_block();
        let store = b.new_block();
        let logit = b.add_block_param(store, F32);
        b.terminate(KirTerminator::CondBranch(
            in_vocab,
            KirEdge::to(fill),
            KirEdge::with(store, vec![f_neg_inf]),
        ));

        b.set_block(fill);
        let v_wide = widen(b, v);
        let w_row = op2(b, U64, KirOp::Mul, v_wide, hidden_wide);
        let dot = fma_dot(b, dtype, (x, x_row), (w, w_row), cfg.hidden_size);
        let bias_addr = at(b, elem.clone(), Global, bias, v);
        let bias_v = load_elem(b, dtype, bias_addr, Global);
        let with_bias = op2(b, F32, KirOp::Add, dot, bias_v);
        b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![with_bias])));

        // A 16-bit `-inf` rounds to that dtype's `-inf`.
        b.set_block(store);
        let stored = to_elem(b, dtype, logit);
        let addr = at(b, elem.clone(), Shared, logits, slot);
        b.emit(KirOp::Store(addr, stored, Shared));
    });
    // Every logit of the tile is in shared memory.
    b.emit(KirOp::Barrier);

    // ── thread 0 reduces the tile ───────────────────────────────────────
    let reduce = b.new_block();
    let not_thread0 = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_thread0, KirEdge::to(exit), KirEdge::to(reduce)));
    b.set_block(reduce);
    // Bottom-tested, as the hand kernels' scans were (the tile is never
    // empty): with the test at the top, ptxas unrolled the f32 scans whole
    // on sm_90 and sm_120 (161 and 130 registers against the hand
    // kernels' 32 and 40).
    let scan = |b: &mut KirBuilder, acc0: VarId, step: &dyn Fn(&mut KirBuilder, VarId, VarId) -> VarId| -> VarId {
        let body = b.new_block();
        let done = b.new_block();
        let i = b.add_block_param(body, U32);
        let acc = b.add_block_param(body, F32);
        let out = b.add_block_param(done, F32);
        b.terminate(KirTerminator::Branch(KirEdge::with(body, vec![zero, acc0])));
        b.set_block(body);
        let addr = at(b, elem.clone(), Shared, logits, i);
        let s = load_elem(b, dtype, addr, Shared);
        let acc_next = step(b, acc, s);
        let i_next = op2(b, U32, KirOp::Add, i, one);
        let more = cmp(b, i_next, vtile, CmpOp::Lt);
        b.terminate(KirTerminator::CondBranch(
            more,
            KirEdge::with(body, vec![i_next, acc_next]),
            KirEdge::with(done, vec![acc_next]),
        ));
        b.set_block(done);
        out
    };
    let tile_max = scan(&mut b, f_neg_inf, &|b, m, s| op2(b, F32, KirOp::Max, m, s));
    let tile_sum = scan(&mut b, f_zero, &|b, acc, s| {
        let d = op2(b, F32, KirOp::Sub, s, tile_max);
        let e = b.new_typed_var(F32);
        b.emit(KirOp::Exp(e, d));
        op2(b, F32, KirOp::Add, acc, e)
    });
    b.emit(KirOp::Store(max_slot, tile_max, Global));
    b.emit(KirOp::Store(sum_slot, tile_sum, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // ── an ignored row: thread 0 writes (0, 0) ──────────────────────────
    b.set_block(skip);
    let zero_write = b.new_block();
    let not_zeroer = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zeroer, KirEdge::to(exit), KirEdge::to(zero_write)));
    b.set_block(zero_write);
    b.emit(KirOp::Store(max_slot, f_zero, Global));
    b.emit(KirOp::Store(sum_slot, f_zero, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// Build Kernel B, the per-row finalize, as KIR.
///
/// Grid `(B*S)`, 128 threads, of which thread 0 does the work: an
/// online-LSE fold of the row's `num_tiles` partials, and one dot for the
/// target's logit.
///
/// ```text
/// entry     thread 0 ? work : exit
/// work      target = targets[row]; target == ignore_index ? skip : fold
/// fold      per tile t (max, sum), from (-inf, 0):
///             m = max(max, tmax)
///             sum = sum * exp(max - m) + tsum * exp(tmax - m); max = m
///           lse = log(sum) + max
///           loss = lse - (fma-dot(x[row], W[target]) + bias[target])
///           loss_out[row] = loss; lse_out[row] = lse
/// skip      loss_out[row] = lse_out[row] = 0
/// ```
///
/// The target is not range-checked (Kernel A's partials hold every other
/// column): a target outside `[0, V)` other than the ignore index reads
/// outside `W`, as it did in the hand kernels.
pub fn build_large_finalize(cfg: &FusedLinearCEConfig) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I64, U32, U64};

    let dtype = cfg.dtype;
    let elem = elem_type(dtype);
    let mut b = KirBuilder::new(&cfg.large_finalize_kernel_name());
    let x = b.add_param("x", ptr(elem.clone(), Global), Global);
    let w = b.add_param("w", ptr(elem.clone(), Global), Global);
    let bias = b.add_param("bias", ptr(elem.clone(), Global), Global);
    let targets = b.add_param("targets", ptr(I64, Global), Global);
    let partials = b.add_param("partials", ptr(F32, Global), Global);
    let loss_out = b.add_param("loss_out", ptr(F32, Global), Global);
    let lse_out = b.add_param("lse_out", ptr(F32, Global), Global);
    for name in ["B", "S", "V", "H", "num_tiles"] {
        b.add_param(name, U32, Global);
    }
    b.set_workgroup_size([V1_BLOCK, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let row = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(row, 0));
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let zero = konst(&mut b, ConstValue::U32(0));
    let work = b.new_block();
    let exit = b.new_block();
    let not_thread0 = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_thread0, KirEdge::to(exit), KirEdge::to(work)));

    b.set_block(work);
    let target_addr = at(&mut b, I64, Global, targets, row);
    let target = load(&mut b, I64, target_addr, Global);
    let loss_addr = at(&mut b, F32, Global, loss_out, row);
    let lse_addr = at(&mut b, F32, Global, lse_out, row);
    let ignore = i64_const(&mut b, cfg.ignore_index);
    let skips = cmp(&mut b, target, ignore, CmpOp::Eq);
    let fold = b.new_block();
    let skip = b.new_block();
    b.terminate(KirTerminator::CondBranch(skips, KirEdge::to(skip), KirEdge::to(fold)));

    // ── the online-LSE fold over the row's partials ─────────────────────
    b.set_block(fold);
    let one = konst(&mut b, ConstValue::U32(1));
    let f_zero = konst(&mut b, ConstValue::F32(0.0));
    let f_neg_inf = konst(&mut b, ConstValue::F32(f32::NEG_INFINITY));
    let n_tiles = cfg.num_vocab_tiles();
    let n_tiles_wide = konst(&mut b, ConstValue::U64(n_tiles as u64));
    let row_wide = widen(&mut b, row);
    let row_first = op2(&mut b, U64, KirOp::Mul, row_wide, n_tiles_wide);
    let two = konst(&mut b, ConstValue::U64(2));
    let row_pairs = op2(&mut b, U64, KirOp::Mul, row_first, two);
    let row_partials = at(&mut b, F32, Global, partials, row_pairs);
    let head = b.new_block();
    let body = b.new_block();
    let folded = b.new_block();
    let t = b.add_block_param(head, U32);
    let run_max = b.add_block_param(head, F32);
    let run_sum = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, f_neg_inf, f_zero])));
    b.set_block(head);
    let n_tiles_c = konst(&mut b, ConstValue::U32(n_tiles));
    let finished = cmp(&mut b, t, n_tiles_c, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(folded), KirEdge::to(body)));
    b.set_block(body);
    let two32 = konst(&mut b, ConstValue::U32(2));
    let pair = op2(&mut b, U32, KirOp::Mul, t, two32);
    let tmax_addr = at(&mut b, F32, Global, row_partials, pair);
    let tile_max = load(&mut b, F32, tmax_addr, Global);
    let tsum_addr = at(&mut b, F32, Global, tmax_addr, one);
    let tile_sum = load(&mut b, F32, tsum_addr, Global);
    let new_max = op2(&mut b, F32, KirOp::Max, run_max, tile_max);
    let rescale = |b: &mut KirBuilder, from: VarId, sum: VarId| {
        let shift = op2(b, F32, KirOp::Sub, from, new_max);
        let e = b.new_typed_var(F32);
        b.emit(KirOp::Exp(e, shift));
        op2(b, F32, KirOp::Mul, sum, e)
    };
    let run_scaled = rescale(&mut b, run_max, run_sum);
    let tile_scaled = rescale(&mut b, tile_max, tile_sum);
    let sum_next = op2(&mut b, F32, KirOp::Add, run_scaled, tile_scaled);
    let t_next = op2(&mut b, U32, KirOp::Add, t, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![t_next, new_max, sum_next])));

    b.set_block(folded);
    let log_sum = b.new_typed_var(F32);
    b.emit(KirOp::Log(log_sum, run_sum));
    let lse = op2(&mut b, F32, KirOp::Add, log_sum, run_max);
    // The target's logit, recomputed: one dot of length H.
    let hidden_wide = konst(&mut b, ConstValue::U64(cfg.hidden_size as u64));
    let x_row = op2(&mut b, U64, KirOp::Mul, row_wide, hidden_wide);
    let hidden_signed = i64_const(&mut b, cfg.hidden_size as i64);
    let w_row = op2(&mut b, I64, KirOp::Mul, target, hidden_signed);
    let dot = fma_dot(&mut b, dtype, (x, x_row), (w, w_row), cfg.hidden_size);
    let bias_addr = at(&mut b, elem.clone(), Global, bias, target);
    let bias_v = load_elem(&mut b, dtype, bias_addr, Global);
    let logit_at_target = op2(&mut b, F32, KirOp::Add, dot, bias_v);
    let loss = op2(&mut b, F32, KirOp::Sub, lse, logit_at_target);
    b.emit(KirOp::Store(loss_addr, loss, Global));
    b.emit(KirOp::Store(lse_addr, lse, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // ── an ignored row ──────────────────────────────────────────────────
    b.set_block(skip);
    let f_zero = konst(&mut b, ConstValue::F32(0.0));
    b.emit(KirOp::Store(loss_addr, f_zero, Global));
    b.emit(KirOp::Store(lse_addr, f_zero, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// The large-vocab forward module: [`build_large_partials`] then
/// [`build_large_finalize`], verified and lowered under one header
/// (null-terminated for `cuModuleLoadData`).
///
/// # Panics
///
/// If either kernel fails KIR verification — a bug in this module, not a
/// condition a caller can provoke.
fn emit_large_forward(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    let kernels = [build_large_partials(cfg), build_large_finalize(cfg)];
    for ir in &kernels {
        if let Err(errors) = crate::kir_verify::verify(ir) {
            panic!("{} failed KIR verification: {errors:?}", ir.name);
        }
    }
    lower_kir_module_to_ptx(&[&kernels[0], &kernels[1]])
}

// ─── KIR — backward ─────────────────────────────────────────────────────────
//
// Roadmap A2 step 10, third slice: the backward is KIR too, one builder for
// all three dtypes. Its loops are bottom-tested, as the hand kernels' were
// (every trip count is at least one), and its scatters are `AtomicAdd`s,
// which lower to `red.global.add.f32`. `tests/fused_linear_ce_bwd_kir_equivalence.rs`
// runs it against the frozen hand kernels on the PTX interpreter.

/// `for (i = start; ; ) { carried = body(i, carried); i += step; if !(i < end) break; }`
/// — a loop tested at the bottom, for a trip count known to be at least one.
/// The builder is left in the loop's exit block; the carried values after
/// the last trip are returned.
fn bottom_tested_loop(
    b: &mut KirBuilder,
    (start, end, step): (VarId, VarId, VarId),
    init: &[VarId],
    body: impl FnOnce(&mut KirBuilder, VarId, &[VarId]) -> Vec<VarId>,
) -> Vec<VarId> {
    let tys: Vec<KirType> = init.iter().map(|v| b.var_type(*v).expect("a typed loop value")).collect();
    let body_block = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(body_block, KirType::U32);
    let carried: Vec<VarId> = tys.iter().map(|t| b.add_block_param(body_block, t.clone())).collect();
    let out: Vec<VarId> = tys.iter().map(|t| b.add_block_param(done, t.clone())).collect();
    let mut entry_args = vec![start];
    entry_args.extend_from_slice(init);
    b.terminate(KirTerminator::Branch(KirEdge::with(body_block, entry_args)));

    b.set_block(body_block);
    let next_carried = body(b, i, &carried);
    assert_eq!(next_carried.len(), carried.len(), "a loop body returns one value per carried value");
    let i_next = op2(b, KirType::U32, KirOp::Add, i, step);
    let more = cmp(b, i_next, end, CmpOp::Lt);
    let mut back = vec![i_next];
    back.extend_from_slice(&next_carried);
    b.terminate(KirTerminator::CondBranch(
        more,
        KirEdge::with(body_block, back),
        KirEdge::with(done, next_carried),
    ));
    b.set_block(done);
    out
}

/// Build the backward kernel as KIR.
///
/// One CTA per token row, 128 threads, no shared memory; the same kernel
/// serves both forward paths (it reads the saved per-row lse).
///
/// ```text
/// entry     target = targets[row]; target == ignore_index ? skip : live
/// live      scale = grad_output / f32(num_valid)
///           per tile t, per sub-tile j (bottom-tested):
///             v = j*128 + tid + t*vtile; v < V ?
///               logit = fma-dot(x[row], W[v]) + bias[v]
///               p = exp(logit - lse[row]); v == target ? p -= 1
///               g = p * scale
///               per h (bottom-tested):
///                 red dx[row, h] += g * W[v, h]
///                 red dW[v, h]   += g * x[row, h]
///               red dbias[v] += g
/// skip      every thread zeroes dx[row, tid + 128k] for tid + 128k < H
/// ```
///
/// `x`, `W` and `bias` are in the storage dtype and widened to f32; `lse`
/// and the three gradients are f32 whatever the dtype.
pub fn build_backward(cfg: &FusedLinearCEConfig) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I64, U32, U64};

    let dtype = cfg.dtype;
    let elem = elem_type(dtype);
    let mut b = KirBuilder::new(&cfg.bwd_kernel_name());
    // The params, in FFI order. B, S, V and H are baked; num_valid is read.
    let grad_output = b.add_param("grad_output", F32, Global);
    let x = b.add_param("x", ptr(elem.clone(), Global), Global);
    let w = b.add_param("w", ptr(elem.clone(), Global), Global);
    let bias = b.add_param("bias", ptr(elem.clone(), Global), Global);
    let targets = b.add_param("targets", ptr(I64, Global), Global);
    let lse = b.add_param("lse", ptr(F32, Global), Global);
    let dx_out = b.add_param("dx_out", ptr(F32, Global), Global);
    let dw_out = b.add_param("dw_out", ptr(F32, Global), Global);
    let dbias_out = b.add_param("dbias_out", ptr(F32, Global), Global);
    for name in ["B", "S", "V", "H"] {
        b.add_param(name, U32, Global);
    }
    let num_valid = b.add_param("num_valid", U32, Global);
    b.set_workgroup_size([V1_BLOCK, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let num_valid_f = b.new_typed_var(F32);
    b.emit(KirOp::Cast(num_valid_f, num_valid, F32));
    let row = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(row, 0));
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let block = konst(&mut b, ConstValue::U32(V1_BLOCK));
    let hidden = konst(&mut b, ConstValue::U32(cfg.hidden_size));
    let hidden_wide = konst(&mut b, ConstValue::U64(cfg.hidden_size as u64));
    let row_wide = widen(&mut b, row);
    let x_row = op2(&mut b, U64, KirOp::Mul, row_wide, hidden_wide);
    let target_addr = at(&mut b, I64, Global, targets, row);
    let target = load(&mut b, I64, target_addr, Global);
    let ignore = i64_const(&mut b, cfg.ignore_index);
    let skips = cmp(&mut b, target, ignore, CmpOp::Eq);
    let live = b.new_block();
    let skip = b.new_block();
    let exit = b.new_block();
    b.terminate(KirTerminator::CondBranch(skips, KirEdge::to(skip), KirEdge::to(live)));

    // ── the live row ────────────────────────────────────────────────────
    b.set_block(live);
    let lse_addr = at(&mut b, F32, Global, lse, row);
    let lse_row = load(&mut b, F32, lse_addr, Global);
    let scale = op2(&mut b, F32, KirOp::Div, grad_output, num_valid_f);
    let f_one = konst(&mut b, ConstValue::F32(1.0));
    let vocab = konst(&mut b, ConstValue::U32(cfg.vocab_size));
    let vtile = konst(&mut b, ConstValue::U32(cfg.vocab_tile));
    let n_tiles = konst(&mut b, ConstValue::U32(cfg.num_vocab_tiles()));
    let per_thread = konst(&mut b, ConstValue::U32(cfg.vocab_tile / V1_BLOCK));
    let elem_at = |b: &mut KirBuilder, base: VarId, index: VarId| {
        let addr = at(b, elem.clone(), Global, base, index);
        load_elem(b, dtype, addr, Global)
    };
    bottom_tested_loop(&mut b, (zero, n_tiles, one), &[], |b, tile, _| {
        let v_base = op2(b, U32, KirOp::Mul, tile, vtile);
        bottom_tested_loop(b, (zero, per_thread, one), &[], |b, j, _| {
            let lane_base = op2(b, U32, KirOp::Mul, j, block);
            let slot = op2(b, U32, KirOp::Add, lane_base, tid);
            let v = op2(b, U32, KirOp::Add, slot, v_base);
            let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
            let column = b.new_block();
            let column_done = b.new_block();
            b.terminate(KirTerminator::CondBranch(in_vocab, KirEdge::to(column), KirEdge::to(column_done)));

            b.set_block(column);
            let v_wide = widen(b, v);
            let w_row = op2(b, U64, KirOp::Mul, v_wide, hidden_wide);
            let f_zero = konst(b, ConstValue::F32(0.0));
            let dot = bottom_tested_loop(b, (zero, hidden, one), &[f_zero], |b, h, acc| {
                let h_wide = widen(b, h);
                let x_index = op2(b, U64, KirOp::Add, x_row, h_wide);
                let xv = elem_at(b, x, x_index);
                let w_index = op2(b, U64, KirOp::Add, w_row, h_wide);
                let wv = elem_at(b, w, w_index);
                let acc_next = b.new_typed_var(F32);
                b.emit(KirOp::Fma(acc_next, xv, wv, acc[0]));
                vec![acc_next]
            })[0];
            let bias_v = elem_at(b, bias, v);
            let logit = op2(b, F32, KirOp::Add, dot, bias_v);
            let shifted = op2(b, F32, KirOp::Sub, logit, lse_row);
            let p = b.new_typed_var(F32);
            b.emit(KirOp::Exp(p, shifted));
            // dlogit = p - [v == target]
            let v_signed = b.new_typed_var(I64);
            b.emit(KirOp::Cast(v_signed, v, I64));
            let is_target = cmp(b, v_signed, target, CmpOp::Eq);
            let p_minus_one = op2(b, F32, KirOp::Sub, p, f_one);
            let dlogit = b.new_typed_var(F32);
            b.emit(KirOp::Select(dlogit, is_target, p_minus_one, p));
            let g = op2(b, F32, KirOp::Mul, dlogit, scale);
            bottom_tested_loop(b, (zero, hidden, one), &[], |b, h, _| {
                let h_wide = widen(b, h);
                let w_index = op2(b, U64, KirOp::Add, w_row, h_wide);
                let wv = elem_at(b, w, w_index);
                let dx_part = op2(b, F32, KirOp::Mul, g, wv);
                let x_index = op2(b, U64, KirOp::Add, x_row, h_wide);
                let dx_addr = at(b, F32, Global, dx_out, x_index);
                b.emit(KirOp::AtomicAdd(dx_addr, dx_part, Global));
                let xv = elem_at(b, x, x_index);
                let dw_part = op2(b, F32, KirOp::Mul, g, xv);
                let dw_addr = at(b, F32, Global, dw_out, w_index);
                b.emit(KirOp::AtomicAdd(dw_addr, dw_part, Global));
                vec![]
            });
            let dbias_addr = at(b, F32, Global, dbias_out, v);
            b.emit(KirOp::AtomicAdd(dbias_addr, g, Global));
            b.terminate(KirTerminator::Branch(KirEdge::to(column_done)));

            b.set_block(column_done);
            vec![]
        });
        vec![]
    });
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // ── an ignored row: its dx row is zeroed ────────────────────────────
    b.set_block(skip);
    let head = b.new_block();
    let body = b.new_block();
    let k = b.add_block_param(head, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero])));
    b.set_block(head);
    let lane_base = op2(&mut b, U32, KirOp::Mul, k, block);
    let slot = op2(&mut b, U32, KirOp::Add, lane_base, tid);
    let in_row = cmp(&mut b, slot, hidden, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(in_row, KirEdge::to(body), KirEdge::to(exit)));
    b.set_block(body);
    let slot_wide = widen(&mut b, slot);
    let dx_index = op2(&mut b, U64, KirOp::Add, x_row, slot_wide);
    let dx_addr = at(&mut b, F32, Global, dx_out, dx_index);
    let f_zero = konst(&mut b, ConstValue::F32(0.0));
    b.emit(KirOp::Store(dx_addr, f_zero, Global));
    let k_next = op2(&mut b, U32, KirOp::Add, k, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next])));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// The backward module: [`build_backward`], verified and lowered
/// (null-terminated for `cuModuleLoadData`).
///
/// # Panics
///
/// If the built kernel fails KIR verification — a bug in this module, not a
/// condition a caller can provoke.
fn emit_backward(cfg: &FusedLinearCEConfig) -> Vec<u8> {
    let ir = build_backward(cfg);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    lower_kir_to_ptx(&ir)
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
        // The v1 forward is KIR (roadmap A2 step 10) and targets the KIR
        // floor, sm_70, whatever `gpu_sm` says (the driver JIT-compiles it
        // forward); the bf16 conversions raise the ISA to 7.8.
        let cfg = default_cfg();
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes).expect("PTX must be valid UTF-8");
        assert!(
            ptx.starts_with(".version 7.0\n.target sm_70"),
            "PTX must start with .version 7.0 + .target sm_70, got: {}",
            &ptx[..50.min(ptx.len())]
        );
        let bf16 = FusedLinearCEConfig { dtype: Dtype::Bf16, ..default_cfg() };
        let ptx_bytes = synthesize_fused_linear_ce_ptx(&bf16);
        let ptx = std::str::from_utf8(&ptx_bytes).expect("PTX must be valid UTF-8");
        assert!(ptx.starts_with(".version 7.8\n"), "{}", &ptx[..50.min(ptx.len())]);
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
        // Exactly one .version line + one .target line at module scope (the
        // KIR floor, as for every KIR module without bf16).
        assert_eq!(
            ptx.matches(".version 7.0").count(),
            1,
            ".version must appear exactly once at module scope"
        );
        assert_eq!(
            ptx.matches(".target sm_70").count(),
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
    fn test_large_ptx_partials_are_f32_pairs() {
        // Both kernels address partials as f32 pairs, one per (row, tile):
        // Kernel A stores its tile's (max, sum), or (0, 0) for an ignored
        // row; Kernel B loads each tile's pair. The addressing itself is
        // pinned by `tests/fused_linear_ce_large_kir_equivalence.rs`.
        let cfg = large_vocab_cfg();
        let a = build_large_partials(&cfg);
        let b = build_large_finalize(&cfg);
        let partials = |ir: &KernelIR| ir.params.iter().find(|p| p.name == "partials").unwrap().ty.clone();
        let pair = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        assert_eq!(partials(&a), pair);
        assert_eq!(partials(&b), pair);
        let ptx = String::from_utf8(synthesize_fused_linear_ce_ptx(&cfg)).unwrap();
        let (kernel_a, kernel_b) = ptx.split_at(ptx.rfind(".visible .entry ").unwrap());
        assert_eq!(kernel_a.matches("st.global.f32 ").count(), 4);
        assert_eq!(kernel_b.matches("ld.global.f32 ").count(), 5, "a pair per tile, x, W and bias");
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

    // ── Null-termination contract ─────────────────────────────────────────
    //
    // Every public synthesizer in this module returns null-terminated PTX
    // bytes so they can be passed straight to `cuModuleLoadData` without
    // the caller having to remember to append `0u8`. This matches the
    // convention established by `backend_ptx::lower_kir_to_ptx`. Forgetting
    // the null is silent UB (CUDA driver reads past the buffer) — pin every
    // (synthesizer × dtype × routing) combination so any future emitter
    // that bypasses the synthesizer wrapper trips this test.

    fn null_term_cfg(dtype: Dtype, large: bool) -> FusedLinearCEConfig {
        FusedLinearCEConfig {
            // 4096 (small-vocab path), 49152 (large-vocab path).
            vocab_size: if large { 49152 } else { 4096 },
            hidden_size: 128,
            seq_len: 32,
            batch_size: 1,
            vocab_tile: if large { 128 } else { 1024 },
            gpu_sm: 80,
            dtype,
            ignore_index: -100,
            max_vocab_v1: if large { MAX_VOCAB_HARD_CEILING } else { 8192 },
        }
    }

    #[test]
    fn fwd_ptx_is_null_terminated_small_vocab_f32() {
        let cfg = null_term_cfg(Dtype::F32, false);
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "small-vocab F32 forward must end with NUL");
    }

    #[test]
    fn fwd_ptx_is_null_terminated_small_vocab_f16() {
        let cfg = null_term_cfg(Dtype::F16, false);
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "small-vocab F16 forward must end with NUL");
    }

    #[test]
    fn fwd_ptx_is_null_terminated_small_vocab_bf16() {
        let cfg = null_term_cfg(Dtype::Bf16, false);
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "small-vocab Bf16 forward must end with NUL");
    }

    #[test]
    fn fwd_ptx_is_null_terminated_large_vocab_f32() {
        let cfg = null_term_cfg(Dtype::F32, true);
        assert!(cfg.is_large_vocab());
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "large-vocab F32 forward must end with NUL");
        // Pin no double-null — dispatcher delegates without re-pushing.
        let trailing_nulls = bytes.iter().rev().take_while(|&&b| b == 0).count();
        assert_eq!(trailing_nulls, 1, "dispatcher must not double-null the large-vocab path");
    }

    #[test]
    fn fwd_ptx_is_null_terminated_large_vocab_f16() {
        let cfg = null_term_cfg(Dtype::F16, true);
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "large-vocab F16 forward must end with NUL");
    }

    #[test]
    fn fwd_ptx_is_null_terminated_large_vocab_bf16() {
        let cfg = null_term_cfg(Dtype::Bf16, true);
        let bytes = synthesize_fused_linear_ce_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "large-vocab Bf16 forward must end with NUL");
    }

    #[test]
    fn large_vocab_synth_is_null_terminated_direct_call() {
        // Direct call to the large-vocab synthesizer (not via dispatcher).
        let cfg = null_term_cfg(Dtype::F32, true);
        let bytes = synthesize_large_vocab_forward_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "synthesize_large_vocab_forward_ptx must end with NUL");
    }

    #[test]
    fn bwd_ptx_is_null_terminated_f32() {
        let cfg = null_term_cfg(Dtype::F32, false);
        let bytes = synthesize_fused_linear_ce_backward_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "F32 backward must end with NUL");
    }

    #[test]
    fn bwd_ptx_is_null_terminated_f16() {
        let cfg = null_term_cfg(Dtype::F16, false);
        let bytes = synthesize_fused_linear_ce_backward_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "F16 backward must end with NUL");
    }

    #[test]
    fn bwd_ptx_is_null_terminated_bf16() {
        let cfg = null_term_cfg(Dtype::Bf16, false);
        let bytes = synthesize_fused_linear_ce_backward_ptx(&cfg);
        assert_eq!(bytes.last(), Some(&0u8), "Bf16 backward must end with NUL");
    }

    #[test]
    fn ptx_has_no_interior_nuls() {
        // CUDA driver stops at the first NUL — if the kernel text itself
        // contains an embedded NUL, the module load truncates the bytes
        // silently. Pin that interior NULs never appear (the trailing NUL
        // is the ONLY 0 byte in a well-formed PTX module).
        let cfg = default_cfg();
        for synth_name in ["fwd", "bwd"] {
            let bytes = if synth_name == "fwd" {
                synthesize_fused_linear_ce_ptx(&cfg)
            } else {
                synthesize_fused_linear_ce_backward_ptx(&cfg)
            };
            let n = bytes.len();
            for (i, b) in bytes[..n - 1].iter().enumerate() {
                assert_ne!(
                    *b, 0u8,
                    "interior NUL at byte {i} in {synth_name} PTX would truncate cuModuleLoadData"
                );
            }
        }
    }
}
