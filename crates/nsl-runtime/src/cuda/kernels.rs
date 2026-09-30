//! PTX modules for the GPU elementwise and fused-optimizer kernels. Every
//! one is built from KIR by `nsl_kir::kernels` (roadmap A2 step 11,
//! new-roadmap item 5): this file holds no hand-written PTX, and
//! `scripts/hand-ptx-freeze.sh` keeps it that way. Each module is lowered
//! once, on first use, and is NUL-terminated for the CUDA driver API.

// --- Binary ops ---

// `nsl_add_f32`, `nsl_sub_f32`, `nsl_mul_f32` and `nsl_div_f32` are built from KIR by
// `nsl_kir::kernels::elementwise` (roadmap A2 step 11); `nsl-codegen`'s
// `elementwise_binary_kir_equivalence` gate holds them to the hand-written
// modules they replace. Each module is built once, on first use, and kept:
// `kernel_launch` keys its module cache on the buffer's address.

use nsl_kir::kernels::elementwise::BinaryOp;

fn binary_module(op: BinaryOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 4]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        // The printer emits ASCII only, so this cannot fail for a kernel
        // `nsl-kir` builds.
        BinaryOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::binary_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = BinaryOp::ALL.iter().position(|o| *o == op).expect("every BinaryOp is in ALL");
    &modules[slot]
}

/// `nsl_add_f32`: `c[i] = a[i] + b[i]`, NUL-terminated.
pub(crate) fn add_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Add)
}

/// `nsl_sub_f32`: `c[i] = a[i] - b[i]`, NUL-terminated.
pub(crate) fn sub_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Sub)
}

/// `nsl_mul_f32`: `c[i] = a[i] * b[i]`, NUL-terminated.
pub(crate) fn mul_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Mul)
}

/// `nsl_div_f32`: `c[i] = a[i] / b[i]` with `div.approx.f32`, NUL-terminated.
/// The approximate division is load-bearing: the fused kernels that are held
/// bit-exact to a decomposed chain (the FASE AdamW steps among them) divide
/// the way this kernel does.
pub(crate) fn div_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Div)
}

// The scalar-operand family, `c[i] = a[i] op s`, likewise built by
// `nsl_kir::kernels::elementwise` (its `elementwise_scalar_kir_equivalence`
// gate), `nsl_div_scalar_f32` included (`div.approx.f32`, like `nsl_div_f32`).

use nsl_kir::kernels::elementwise::ScalarOp;

pub(crate) fn scalar_module(op: ScalarOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 4]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        ScalarOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::scalar_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = ScalarOp::ALL.iter().position(|o| *o == op).expect("every ScalarOp is in ALL");
    &modules[slot]
}

/// `nsl_mul_scalar_f32`: `c[i] = a[i] * s`, NUL-terminated.
pub(crate) fn mul_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Mul)
}

/// `nsl_add_scalar_f32`: `c[i] = a[i] + s`, NUL-terminated.
pub(crate) fn add_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Add)
}

/// `nsl_sub_scalar_f32`: `c[i] = a[i] - s`, NUL-terminated.
pub(crate) fn sub_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Sub)
}

/// `nsl_div_scalar_f32`: `c[i] = a[i] / s` with `div.approx.f32`, the
/// division `nsl_div_f32` performs, so the scalar sweep's `Div(x, Constant)`
/// fold stays bit-exact with the broadcast divide it replaces. NUL-terminated.
pub(crate) fn div_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Div)
}

// Two kernels that must match a decomposed computation bit for bit, built by
// `nsl_kir::kernels::elementwise` with explicitly rounded arithmetic
// (`KirOp::{AddRn, MulRn}` print `.rn`, which ptxas never contracts into an
// `fma`); `elementwise_rn_kir_equivalence` holds them to the hand-written
// modules they replace.

/// `nsl_scalar_mul_add_inplace_f32(m, g, s, n)`: `m[i] = m[i] + g[i] * s`,
/// the FASE accumulate epilogue, bit-exact with `nsl_mul_scalar_f32` then
/// `nsl_add_f32`. NUL-terminated.
pub(crate) fn scalar_mul_add_inplace_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::scalar_mul_add_inplace_ptx()).expect("PTX must be ASCII")
    })
}

/// `nsl_muon_scale_inv_frob_f32(x, c, stats, n)`:
/// `c[i] = x[i] * (1 / (sqrt(stats[3]) + 1e-7))`, with `stats` the output of
/// `nsl_tensor_stats_f32` read on the device. NUL-terminated.
pub(crate) fn muon_scale_inv_frob_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::muon_scale_inv_frob_ptx()).expect("PTX must be ASCII")
    })
}

// The unary family, `c[i] = f(a[i])`, likewise built by
// `nsl_kir::kernels::elementwise` (its `elementwise_unary_kir_equivalence`
// gate). `nsl_tanh_f32` is built there too, as its own function
// ([`tanh_f32_ptx`]).

use nsl_kir::kernels::elementwise::UnaryOp;

pub(crate) fn unary_module(op: UnaryOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 13]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        UnaryOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::unary_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = UnaryOp::ALL.iter().position(|o| *o == op).expect("every UnaryOp is in ALL");
    &modules[slot]
}

/// `nsl_neg_f32`: `c[i] = -a[i]`, NUL-terminated.
pub(crate) fn neg_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Neg)
}

/// `nsl_relu_f32`: `c[i] = max(a[i], 0)`, NUL-terminated.
pub(crate) fn relu_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Relu)
}

/// `nsl_exp_f32`: `c[i] = e^a[i] (ex2.approx)`, NUL-terminated.
pub(crate) fn exp_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Exp)
}

/// `nsl_log_f32`: `c[i] = ln a[i] (lg2.approx)`, NUL-terminated.
pub(crate) fn log_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Log)
}

/// `nsl_sqrt_f32`: `c[i] = sqrt(a[i]) (IEEE)`, NUL-terminated.
pub(crate) fn sqrt_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sqrt)
}

/// `nsl_abs_f32`: `c[i] = |a[i]|`, NUL-terminated.
pub(crate) fn abs_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Abs)
}

/// `nsl_sign_f32`: `c[i] = sign(a[i]) (0 for 0 and NaN)`, NUL-terminated.
pub(crate) fn sign_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sign)
}

/// `nsl_sigmoid_f32`: `c[i] = 1 / (1 + e^-a[i])`, NUL-terminated.
pub(crate) fn sigmoid_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sigmoid)
}

/// `nsl_sin_f32`: `c[i] = sin a[i] (sin.approx)`, NUL-terminated.
pub(crate) fn sin_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sin)
}

/// `nsl_cos_f32`: `c[i] = cos a[i] (cos.approx)`, NUL-terminated.
pub(crate) fn cos_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Cos)
}

/// `nsl_silu_f32`: `c[i] = a[i] * sigmoid(a[i])`, NUL-terminated.
pub(crate) fn silu_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Silu)
}

/// `nsl_gelu_f32`: `c[i] = a[i] * sigmoid(1.702 * a[i])`, NUL-terminated. The
/// slope is `nsl_kir::kernels::elementwise::GELU_SLOPE`, the one
/// `nsl_gelu_backward_srcad_f32` differentiates with (`BackwardOp::GeluSrcad`).
pub(crate) fn gelu_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Gelu)
}

/// `nsl_clamp_f32`: `c[i] = min(max(a[i], lo), hi)`, NUL-terminated.
pub(crate) fn clamp_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Clamp)
}



/// `nsl_rotate_half_f32` / `nsl_rotate_half_neg_f32` (`a, c, n, last_dim,
/// half`), built by `nsl_kir::kernels::elementwise::build_rotate_half` (its
/// `rotate_half_kir_equivalence` gate). The `_neg` form is the fused RoPE
/// backward `neg(rotate_half(x))` (mfu-fusion C2): the negation moves to the
/// second half, bit-identical to the two-launch composition because an f32
/// negation is a sign-bit flip. NUL-terminated.
pub(crate) fn rotate_half_module(op: nsl_kir::kernels::elementwise::RotateHalfOp) -> &'static str {
    use nsl_kir::kernels::elementwise::RotateHalfOp;
    static MODULES: std::sync::OnceLock<[String; 2]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        RotateHalfOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::rotate_half_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = RotateHalfOp::ALL.iter().position(|o| *o == op).expect("every RotateHalfOp is in ALL");
    &modules[slot]
}


// --- Matrix multiplication ---
//
// The f32 single-matmul PTX kernel (`nsl_matmul_f32`) was deleted 2026-04-21
// as part of the cuBLAS swap (spec docs/superpowers/specs/2026-04-21-matmul-
// cublas-swap-design.md). f32 single matmul now dispatches to cuBLAS sgemm
// via `cuda::cublas_inner::sgemm_row_major` — see `cuda::gpu_matmul_f32`.
// The batched f32 path (`nsl_bmm_f32`) is out of scope per spec §6 and
// stays a PTX kernel (since built by `nsl_kir::kernels::bmm`).




// --- Backward kernels for activation functions ---
//
// The tape-AD `nsl_{relu,sigmoid,tanh,silu}_backward_f32`, the source-AD
// `nsl_{sigmoid,tanh,silu,gelu}_backward_srcad_f32` and the fused SwiGLU gate
// adjoint `nsl_swiglu_gate_backward_f32` are built by
// `nsl_kir::kernels::elementwise` (its `elementwise_backward_kir_equivalence`
// gate). The source-AD kernels match a chain of separate launches bit for
// bit, so their derivative arithmetic is explicitly rounded (`.rn`, which
// ptxas never contracts into an `fma`). So is `nsl_clamp_backward_f32`
// (`clamp_backward_kir_equivalence`). So is `nsl_gelu_backward_f32`
// ([`gelu_backward_f32_ptx`]).

use nsl_kir::kernels::elementwise::BackwardOp;

pub(crate) fn backward_module(op: BackwardOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 9]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        BackwardOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::backward_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = BackwardOp::ALL.iter().position(|o| *o == op).expect("every BackwardOp is in ALL");
    &modules[slot]
}

/// `nsl_gelu_backward_f32(grad, input, out, n)`: the tape-AD adjoint of the
/// GELU tanh approximation, built by
/// `nsl_kir::kernels::elementwise::build_gelu_backward` (its
/// `tanh_gelu_backward_kir_equivalence` gate). Past `|k| = 43.5` the
/// derivative is its limit, 1 above and 0 below; the hand kernel returned
/// garbage and then NaN there (`div.approx.f32`'s flush range).
/// NUL-terminated.
pub(crate) fn gelu_backward_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::gelu_backward_ptx()).expect("PTX must be ASCII")
    })
}


// Fused per-parameter FASE-Deferred AdamW/Adam optimizer step (Milestone C ·
// p9): ONE launch replacing the ~15-launch interpreted `UpdateProgram`
// (`fase_emit_final_step`), BIT-EXACT with it. Built by
// `nsl_kir::kernels::optim::build_fase_adamw_step` (its doc has the per-op
// rounding order: every op `.rn`, `sqrt.rn`, and the decomposed `Div`'s
// `div.approx.f32`); `fase_adamw_step_kir_equivalence` holds it to the
// hand-written module it replaced. NUL-terminated.
pub(crate) fn fase_fused_adamw_step_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::optim::fase_adamw_step_ptx()).expect("PTX must be ASCII")
    })
}

// Fusion-queue item 1: MULTI-TENSOR fused AdamW step. One launch updates
// every parameter, with base pointers read from device pointer tables and
// per-parameter length from ntab. The arithmetic body is byte-for-byte the
// `nsl_fase_fused_adamw_step_f32` sequence (same roundings, same div.approx),
// so per-element results are BIT-IDENTICAL to the per-param launches. The
// shared tail's m_partial zeroing is folded in (store 0 after the read) —
// value-identical to the separate nsl_tensor_zero_inplace pass it replaces.
//
// (Roadmap item 8 replaced the original `grid.y = parameter index` mapping;
// the doc comment below describes what it does now. This paragraph used to
// end "blocks past a shorter param's end exit immediately", which was the
// waste item 8 removed.)
/// Multi-parameter fused AdamW, FLAT grid (roadmap item 8).
///
/// Was a rectangular grid: `(ceil(max_n/256), k, 1)`, i.e. every parameter got
/// enough blocks for the LARGEST parameter and the surplus exited on the
/// bounds guard. Measured on Coder-50M's real parameter list (74 params,
/// 38.5M elements, largest 16.4M): 4,736,000 blocks launched for 150,562
/// blocks of work — 96.8% exited immediately, a 31.5x overshoot.
///
/// Now the host precomputes two u32 tables indexed by BLOCK id — `bptab[b]`
/// is the parameter that block `b` works on, `bbtab[b]` is the element offset
/// of that block's first thread within the parameter — so the grid is
/// `(sum(ceil(n_i/256)), 1, 1)` with no wasted blocks and no binary search in
/// the kernel. The tables depend only on the shape list, so they are built
/// once and cached until it changes.
///
/// Dropping grid.y also removes the 65535-parameter cap.
///
/// CONTRACT: the kernel no longer reads `%ntid.x`, so `blockDim.x` at launch
/// MUST equal the `block` argument `build_block_tables` was called with. Both
/// come from the single `let block = 256i64` in
/// `gpu_fase_fused_adamw_step_multi`, so they cannot currently desync — but if
/// they ever did, a smaller `blockDim` silently skips elements (weights stop
/// updating) and a larger one applies AdamW twice to the overlap. Neither is
/// caught by any test, because no test can vary the two independently.
///
/// `mp_scale` folds the two-phase-clip Phase B pre-scale
/// (`nsl_tensor_mul_scalar_inplace(m_partial, clip_factor)`) into the read:
/// `g = rn(mp[i] * mp_scale)` — the same single f32 rounding the in-place
/// scale-then-read performs, so the clip path stays bit-identical to the
/// per-param loop it replaces. `mp_scale == 1.0` branches AROUND the
/// multiply so the non-clip path keeps exact bit-identity (a `mul.rn` by
/// 1.0 would canonicalize NaN payloads, which "bit-identical" must not).
///
/// Built by `nsl_kir::kernels::optim::build_fase_adamw_multi` (new-roadmap
/// item 5); `fase_adamw_multi_kir_equivalence` holds it to the hand-written
/// module it replaced. NUL-terminated.
pub(crate) fn fase_fused_adamw_multi_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::optim::fase_adamw_multi_ptx()).expect("PTX must be ASCII")
    })
}

/// Roadmap item 8, bf16-SR arm: MULTI-parameter fused SR-BF16 AdamW step on
/// the SAME flat grid as `nsl_fase_fused_adamw_multi_f32` (bptab/bbtab block
/// tables, per-param base pointers from device tables), with the SR-BF16
/// arithmetic body and rounding tail of `nsl_fase_fused_adamw_step_bf16sr`.
///
/// BIT-IDENTITY to the per-param SR loop it replaces holds because the SR
/// dither is a pure function of (sr_key, ctrtab[p] + element) — the same
/// counters the per-param launches use (`param_idx << SR_PARAM_SHIFT`, set
/// at registration) — so the draw for every (param, element, step) is
/// independent of launch shape. The arithmetic body is byte-for-byte the
/// per-param kernel's sequence.
///
/// Contract deltas vs the f32 multi kernel, both deliberate:
///   - NO `mp_scale`: the per-param SR entry has no clip fold (SR composes
///     with FASE-Deferred accumulation; the clip path is refused upstream),
///     and adding one here would create an arm the per-param path cannot
///     mirror.
///   - NO in-kernel m_partial zero: the per-param SR kernel leaves mp to the
///     FASE-Deferred lifecycle; mirroring that keeps the replacement
///     observationally identical.
///
/// `ntab` is u32 — enforced at the batching entry itself
/// (`bf16sr_multi_impl`'s `u32::try_from(len)` aborts on overflow;
/// registration only bounds len below 2^40 for the counter scheme). The f32
/// multi demotes oversize params to its sequential arm instead; both are
/// unreachable on hardware where a >4G-element param's moments alone
/// exceed VRAM. `ctrtab` is u64 per-param SR counter bases. Same
/// `blockDim.x == build_block_tables block` contract as the f32 multi
/// kernel — one constant feeds both.
///
/// Built by `nsl_kir::kernels::optim::build_fase_adamw_multi_bf16sr`
/// (new-roadmap item 5); `fase_adamw_multi_bf16sr_kir_equivalence` holds it
/// to the hand-written module it replaced. NUL-terminated.
pub(crate) fn fase_fused_adamw_multi_bf16sr_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::optim::fase_adamw_multi_bf16sr_ptx()).expect("PTX must be ASCII")
    })
}

// P4 item 17: fused FASE-Deferred AdamW step with a BF16 AUTHORITATIVE theta
// and counter-based stochastic rounding (SR-BF16, no FP32 master copy),
// built by `nsl_kir::kernels::optim::build_fase_adamw_step_bf16sr` (its
// `sr_bf16_kir_equivalence` gate holds it to the hand-written module it
// replaced). The update is `nsl_fase_fused_adamw_step_f32`'s, rounding for
// rounding; theta loads as bf16 (bits << 16, exact) and stores through the
// SR tail: a splitmix64 hash of (sr_key, sr_ctr_base + element) gives a
// 16-bit dither added to the f32 bits before the low 16 are dropped.
// `sr_key` arrives precomputed as `sr_step_key(seed, step)`, so the kernel
// is a pure function of its counters, bit-identical to the CPU reference
// `sr_bf16::sr_bf16_round_counter`. Edge policy (mirrors sr_bf16.rs): ±Inf
// propagates, NaN becomes quiet 0x7FC0|sign, a rounding carry into the
// all-ones exponent saturates to ±max-normal 0x7F7F|sign, and subnormals
// round through the same bit arithmetic. NUL-terminated.
pub(crate) fn fase_fused_adamw_step_bf16sr_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::optim::fase_adamw_step_bf16sr_ptx()).expect("PTX must be ASCII")
    })
}

// P4 item 17: the standalone SR-BF16 rounding probe, the SAME dither hash and
// rounding tail as the step above applied to an arbitrary f32 buffer, built
// by `nsl_kir::kernels::optim::build_sr_bf16_round_probe`. It exists so the
// parity gate can hold the GPU tail bit-identical to the CPU reference on
// adversarial inputs (max-normal, subnormal, Inf, NaN) without the
// div.approx-dependent optimizer arithmetic in the way. NUL-terminated.
pub(crate) fn sr_bf16_round_probe_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::optim::sr_bf16_round_probe_ptx()).expect("PTX must be ASCII")
    })
}

/// `nsl_clamp_backward_f32(grad, input, out, min_val, max_val, n)`:
/// `out[i] = (input[i] >= min_val && input[i] <= max_val) ? grad[i] : 0`,
/// built by `nsl_kir::kernels::elementwise` (its
/// `clamp_backward_kir_equivalence` gate). NUL-terminated.
pub(crate) fn clamp_backward_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::clamp_backward_ptx()).expect("PTX must be ASCII")
    })
}

/// `nsl_tanh_f32(a, c, n)`: `c[i] = tanh(a[i])`, built by
/// `nsl_kir::kernels::elementwise::build_tanh` (its
/// `tanh_gelu_backward_kir_equivalence` gate). It saturates at `|x| = 43.5`
/// and returns a NaN as it is; the hand kernel returned 0 for every `x`
/// from about 43.67 up, NaN included (`div.approx.f32`'s flush range).
/// NUL-terminated.
pub(crate) fn tanh_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::tanh_ptx()).expect("PTX must be ASCII")
    })
}

