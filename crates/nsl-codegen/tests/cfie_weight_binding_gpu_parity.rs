//! CFIE Cycle 9: GPU numeric parity for the PRODUCTION weight-upload
//! FFIs (`nsl_cfie_upload_weight_f16` / `_f32` / `nsl_cfie_weights_reset`).
//!
//! Cycles 1-8 shipped the full compile-to-GPU CFIE pipeline; the ONE
//! remaining runtime gap before endpoint-driven generation was that
//! `nsl_cfie_decode_step`'s `layer_weights_ptr` had no PRODUCTION
//! producer — model weights load to f32 on CPU but the decode kernels
//! need f16 [out][in] row-major DEVICE pointers, and nothing
//! uploaded/cast/assembled them.  Cycle 9 closes that with the three
//! upload FFIs, proven on hardware.
//!
//! The decode-block kernel + the f16 [out][in] row-major layout are
//! ALREADY GPU-proven by the manual-upload test
//! (`cfie_decode_block_gpu_parity::decode_step_token_sequence_matches_cpu_chain`,
//! which uploads via `nsl_test_cuda_h2d`).  THIS file productionizes the
//! upload as real FFIs and proves the FFI path reproduces that result:
//! every weight is uploaded through `nsl_cfie_upload_weight_f16` /
//! `_f32` (NOT the test helpers), the `n_layers * 9` device-pointer
//! table is assembled from what those FFIs return, and the four-token
//! decode sequence must EXACTLY equal the CPU reference chain (decode
//! blocks per `cfie_persistent_ptx::cpu_reference` fed f16-ROUNDED
//! weights, then the sampler per `cfie_sample_ptx::cpu_reference`).
//!
//! Also proven: `nsl_cfie_weights_reset` frees cleanly (a second bind
//! after reset still drives the same token sequence), and the negative
//! contract (`n_elems == 0` => -1).
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_weight_binding_gpu_parity \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_persistent_ptx::{cpu_reference, emit, DecodeBlockConfig};
use nsl_codegen::cfie_sample_ptx::{
    cpu_reference as cpu_reference_sample, emit as emit_sampler, FusedSampleKernelConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_decode_step, nsl_cfie_engine_destroy, nsl_cfie_engine_finalize,
    nsl_cfie_kv_pool_alloc, nsl_cfie_register_kernel, nsl_cfie_upload_weight_f16,
    nsl_cfie_upload_weight_f32, nsl_cfie_weights_reset,
};
use nsl_runtime::cfie::ffi::{nsl_cfie_kv_slot_acquire, nsl_cfie_kv_slots_init};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
    nsl_test_cuda_jit_log,
};

// ---------------------------------------------------------------------------
// CUDA availability guard (house convention: verbatim per-file copy)
// ---------------------------------------------------------------------------

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// f16 <-> f32 bit converters (per-file copy, house convention).  Used
// ONLY to derive the f16-rounded f32 masters the CPU reference consumes;
// the DEVICE cast is done by the production upload FFI (half::f16 RNE).
// ---------------------------------------------------------------------------

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let e = (127 + e - 14) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let b = x.to_bits();
    let sign = (b >> 31) & 1;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 255 {
        return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16;
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

fn f16_round(x: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(x))
}

// ---------------------------------------------------------------------------
// Deterministic PRNG — the CORRECTED zero-mean mapping (the pca_tier_a
// precedent's `(seed >> 33) as u32 / u32::MAX` only spans [-0.5, 0); the
// mean -0.25 bias saturates every SwiGLU gate in a decode block).  Copied
// verbatim from cfie_decode_block_gpu_parity so the CPU chain is
// bit-identical to that proven fixture.
// ---------------------------------------------------------------------------

fn fill_seeded(dst: &mut [f32], mut seed: u64) {
    for x in dst.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (seed >> 32) as u32;
        *x = ((u as f64) / 4294967296.0) as f32 - 0.5;
    }
}

// ---------------------------------------------------------------------------
// Device-buffer RAII for the TEST-owned scratch buffers (x_a/x_b/tok/
// lm_head/gamma-less path).  The WEIGHT buffers are engine-owned — they
// come back as raw device ptrs from the upload FFIs and are freed by
// nsl_cfie_weights_reset / nsl_cfie_engine_destroy, NOT wrapped here (a
// DevBuf around them would double-free).
// ---------------------------------------------------------------------------

struct DevBuf(i64);

impl DevBuf {
    fn alloc(bytes: usize) -> Self {
        let p = nsl_test_cuda_alloc(bytes as i64);
        assert!(p != 0, "device alloc of {} bytes returned null", bytes);
        DevBuf(p)
    }
    fn ptr(&self) -> i64 {
        self.0
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.0 != 0 {
            nsl_test_cuda_free(self.0);
        }
    }
}

fn h2d_f32(dst: i64, src: &[f32]) {
    nsl_test_cuda_h2d(dst, src.as_ptr() as i64, (src.len() * 4) as i64);
}

fn d2h_u32_one(src: i64) -> u32 {
    let mut out = [0u32; 1];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, src, 4);
    out[0]
}

// ---------------------------------------------------------------------------
// Failure diagnostics: on rc != 0 print the driver JIT log, then panic.
// ---------------------------------------------------------------------------

fn panic_with_jit_log(what: &str, rc: i64, ptx: &str) -> ! {
    let mut ptx_nul = ptx.as_bytes().to_vec();
    ptx_nul.push(0);
    let log_ptr = nsl_test_cuda_jit_log(ptx_nul.as_ptr() as i64);
    let log = if log_ptr == 0 {
        "<no log>".to_string()
    } else {
        unsafe { std::ffi::CStr::from_ptr(log_ptr as *const std::os::raw::c_char) }
            .to_string_lossy()
            .into_owned()
    };
    panic!("{} failed with rc {}; driver JIT log:\n{}", what, rc, log);
}

// ---------------------------------------------------------------------------
// Fixture: 2-layer toy block; per-slot 4 tokens (mirrors the proven
// decode_step test config exactly so its cpu_reference chain is reused).
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const HEAD_DIM: usize = 32;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const D_FF: usize = 128;
const PER_SLOT: usize = 4;
const MAX_SLOTS: usize = 2;
const N_LAYERS: usize = 2;
const VOCAB: usize = 128;
const TOP_K: u32 = 8;

// Uniform pool: [n_layers][2][max_tokens][n_kv_heads][head_dim] f16.
const POOL_BYTES: usize = N_LAYERS * 2 * (MAX_SLOTS * PER_SLOT) * N_KV_HEADS * HEAD_DIM * 2;

fn block_cfg() -> DecodeBlockConfig {
    DecodeBlockConfig {
        d_model: D_MODEL as u32,
        head_dim: HEAD_DIM as u32,
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        d_ff: D_FF as u32,
        per_slot_max_tokens: PER_SLOT as u32,
        max_slots: MAX_SLOTS as u32,
        n_layers: N_LAYERS as u32,
        rope_theta: 10000.0,
        eps: 1e-5,
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
    }
}

/// One layer's host-side weights (f32 masters, seeded), [out][in]
/// row-major (exactly the layout the upload FFI copies verbatim).
struct LayerWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
}

fn seeded_layer_weights(seed: u64) -> LayerWeights {
    let mut w = LayerWeights {
        wq: vec![0f32; N_HEADS * HEAD_DIM * D_MODEL],
        wk: vec![0f32; N_KV_HEADS * HEAD_DIM * D_MODEL],
        wv: vec![0f32; N_KV_HEADS * HEAD_DIM * D_MODEL],
        wo: vec![0f32; D_MODEL * N_HEADS * HEAD_DIM],
        w_gate: vec![0f32; D_FF * D_MODEL],
        w_up: vec![0f32; D_FF * D_MODEL],
        w_down: vec![0f32; D_MODEL * D_FF],
        norm1: vec![0f32; D_MODEL],
        norm2: vec![0f32; D_MODEL],
    };
    fill_seeded(&mut w.wq, seed);
    fill_seeded(&mut w.wk, seed.wrapping_add(1));
    fill_seeded(&mut w.wv, seed.wrapping_add(2));
    fill_seeded(&mut w.wo, seed.wrapping_add(3));
    fill_seeded(&mut w.w_gate, seed.wrapping_add(4));
    fill_seeded(&mut w.w_up, seed.wrapping_add(5));
    fill_seeded(&mut w.w_down, seed.wrapping_add(6));
    fill_seeded(&mut w.norm1, seed.wrapping_add(7));
    fill_seeded(&mut w.norm2, seed.wrapping_add(8));
    // Xavier-ish scale (~1/sqrt(fan_in) for d_model 64) keeps the
    // residual stream O(1) so the block chain stays well-conditioned.
    for m in [
        &mut w.wq, &mut w.wk, &mut w.wv, &mut w.wo, &mut w.w_gate, &mut w.w_up, &mut w.w_down,
    ] {
        for v in m.iter_mut() {
            *v *= 0.35;
        }
    }
    // Norm gammas near 1.0 keep the residual stream well-conditioned.
    for g in w.norm1.iter_mut().chain(w.norm2.iter_mut()) {
        *g = 1.0 + 0.1 * *g;
    }
    w
}

/// Engine-owned device weights: the 9 device pointers the upload FFIs
/// returned, in decode_step order (wq wk wv wo w_gate w_up w_down norm1
/// norm2), plus the f16-rounded f32 masters the CPU reference consumes
/// (norm gammas pass through un-rounded — the kernel loads them as f32).
struct BoundLayerWeights {
    ptrs: [u64; 9],
    rounded: LayerWeights,
}

/// Upload one layer through the PRODUCTION FFIs: 7 matmul matrices as
/// f16 via `nsl_cfie_upload_weight_f16`, the two norm gammas as f32 via
/// `nsl_cfie_upload_weight_f32`.  Every returned ptr must be > 0.
fn bind_layer(w: &LayerWeights) -> BoundLayerWeights {
    let mut ptrs = [0u64; 9];
    let upload16 = |m: &[f32]| -> u64 {
        let p = nsl_cfie_upload_weight_f16(m.as_ptr() as i64, m.len() as i64);
        assert!(p > 0, "nsl_cfie_upload_weight_f16 returned {}", p);
        p as u64
    };
    ptrs[0] = upload16(&w.wq);
    ptrs[1] = upload16(&w.wk);
    ptrs[2] = upload16(&w.wv);
    ptrs[3] = upload16(&w.wo);
    ptrs[4] = upload16(&w.w_gate);
    ptrs[5] = upload16(&w.w_up);
    ptrs[6] = upload16(&w.w_down);
    let n1 = nsl_cfie_upload_weight_f32(w.norm1.as_ptr() as i64, w.norm1.len() as i64);
    assert!(n1 > 0, "nsl_cfie_upload_weight_f32(norm1) returned {}", n1);
    let n2 = nsl_cfie_upload_weight_f32(w.norm2.as_ptr() as i64, w.norm2.len() as i64);
    assert!(n2 > 0, "nsl_cfie_upload_weight_f32(norm2) returned {}", n2);
    ptrs[7] = n1 as u64;
    ptrs[8] = n2 as u64;

    // f16-rounded masters for the CPU reference (device sees these exact
    // values after the RNE cast); norm gammas stay exact (f32 on device).
    let round = |m: &[f32]| -> Vec<f32> { m.iter().map(|&x| f16_round(x)).collect() };
    BoundLayerWeights {
        ptrs,
        rounded: LayerWeights {
            wq: round(&w.wq),
            wk: round(&w.wk),
            wv: round(&w.wv),
            wo: round(&w.wo),
            w_gate: round(&w.w_gate),
            w_up: round(&w.w_up),
            w_down: round(&w.w_down),
            norm1: w.norm1.clone(),
            norm2: w.norm2.clone(),
        },
    }
}

fn cpu_block(
    cfg: &DecodeBlockConfig,
    x: &[f32],
    w: &LayerWeights,
    kv_k: &mut Vec<f32>,
    kv_v: &mut Vec<f32>,
    pos: u32,
) -> Vec<f32> {
    cpu_reference(
        cfg, x, &w.wq, &w.wk, &w.wv, &w.wo, &w.w_gate, &w.w_up, &w.w_down, &w.norm1, &w.norm2,
        kv_k, kv_v, pos,
    )
}

/// Full engine bring-up: register kind 2 (decode_block) + kind 1
/// (fused_sample), init slots, allocate the KV pool, finalize.  Returns
/// the decode-block PTX (for JIT-log diagnostics).  Does NOT upload any
/// weights — that is done through the production FFIs by the caller.
fn setup_engine(sampler: (&str, &str, u32)) -> String {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let (ptx, meta) = emit(&block_cfg());
    let name = meta.kernel_name.as_str();
    let rc = nsl_cfie_register_kernel(
        2, // kind 2 = decode_block
        0,
        ptx.as_ptr() as i64,
        ptx.len() as i64,
        name.as_ptr() as i64,
        name.len() as i64,
        1, // grid = 1 CTA
        meta.block_dim as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 2) refused");
    let (s_ptx, s_name, s_block) = sampler;
    let rc = nsl_cfie_register_kernel(
        1, // kind 1 = fused_sample
        0,
        s_ptx.as_ptr() as i64,
        s_ptx.len() as i64,
        s_name.as_ptr() as i64,
        s_name.len() as i64,
        1,
        s_block as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 1) refused");
    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, PER_SLOT as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(POOL_BYTES as i64), 0);
    let n = nsl_cfie_engine_finalize();
    if n != 2 {
        panic_with_jit_log("nsl_cfie_engine_finalize", n, &ptx);
    }
    ptx
}

/// Build the sampler program + PTX (Greedy top-k) mirroring the proven
/// decode_step fixture.
fn build_sampler() -> (
    nsl_codegen::cfie_fused_sample::FusedSampleProgram,
    String,
    String,
    u32,
) {
    let sampler_params = SamplingParams {
        strategy: SamplingStrategy::Greedy,
        temperature: 0.0,
        top_k: TOP_K,
        ..Default::default()
    };
    let program = emit_program(
        sampler_params,
        LmHeadShape {
            d_model: D_MODEL as u32,
            vocab_size: VOCAB as u32,
            vocab_tile: 128,
            dtype_bytes: 2,
        },
    );
    let (s_ptx, s_meta) = emit_sampler(
        &program,
        &FusedSampleKernelConfig {
            d_model: D_MODEL as u32,
            vocab_size: VOCAB as u32,
            vocab_tile: 128,
            top_k: TOP_K,
            sm_version: 80,
            grammar_states: 0,
        },
    );
    (program, s_ptx, s_meta.kernel_name, s_meta.block_dim)
}

// ---------------------------------------------------------------------------
// Test-owned final-norm gamma + lm_head fixtures (shared across the two
// bind phases so the CPU reference sees the exact same masters).
// ---------------------------------------------------------------------------

struct SampleFixtures {
    gamma: Vec<f32>,
    lm_head: Vec<f32>,
    lm_rounded: Vec<f32>,
}

fn sample_fixtures() -> SampleFixtures {
    let mut gamma = vec![0f32; D_MODEL];
    fill_seeded(&mut gamma, 0xF1);
    for g in gamma.iter_mut() {
        *g = 1.0 + 0.1 * *g;
    }
    let mut lm_head = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut lm_head, 0xF2);
    let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
    SampleFixtures {
        gamma,
        lm_head,
        lm_rounded,
    }
}

/// Drive four greedy tokens through nsl_cfie_decode_step from a COLD slot
/// (pos 0..3) and return (gpu_tokens, cpu_tokens).  `l0`/`l1` are the
/// bound (uploaded) layer weights; `fx` the final-norm gamma + lm_head.
/// The final-norm gamma and lm_head are uploaded through the production
/// FFIs too (f32 and f16 respectively) so the WHOLE weight table is
/// FFI-produced.
#[allow(clippy::too_many_arguments)]
fn run_four_tokens(
    ptx: &str,
    program: &nsl_codegen::cfie_fused_sample::FusedSampleProgram,
    l0: &BoundLayerWeights,
    l1: &BoundLayerWeights,
    fx: &SampleFixtures,
    slot: i64,
    x_a: &DevBuf,
    x_b: &DevBuf,
    tok_dev: &DevBuf,
) -> (Vec<u32>, Vec<u32>) {
    let cfg = block_cfg();

    // HOST weights table: n_layers records x 9 u64 device pointers, in
    // decode_step order — assembled from the upload FFI return values.
    let mut table = [0u64; N_LAYERS * 9];
    table[..9].copy_from_slice(&l0.ptrs);
    table[9..].copy_from_slice(&l1.ptrs);

    // Final-norm gamma (f32) and lm_head (f16 [vocab][d_model]) through
    // the production FFIs.
    let gamma_ptr = nsl_cfie_upload_weight_f32(fx.gamma.as_ptr() as i64, fx.gamma.len() as i64);
    assert!(gamma_ptr > 0, "upload final-norm gamma returned {}", gamma_ptr);
    let lm_ptr = nsl_cfie_upload_weight_f16(fx.lm_head.as_ptr() as i64, fx.lm_head.len() as i64);
    assert!(lm_ptr > 0, "upload lm_head returned {}", lm_ptr);

    let (mut kv0_k, mut kv0_v) = (Vec::<f32>::new(), Vec::<f32>::new());
    let (mut kv1_k, mut kv1_v) = (Vec::<f32>::new(), Vec::<f32>::new());
    let mut gpu_tokens = Vec::new();
    let mut cpu_tokens = Vec::new();
    for pos in 0..PER_SLOT as u32 {
        let mut x = vec![0f32; D_MODEL];
        fill_seeded(&mut x, 0xD00D + pos as u64);
        h2d_f32(x_a.ptr(), &x);

        let rng_seed = 0x5EED_0000 + pos as i64;
        let rc = nsl_cfie_decode_step(
            x_a.ptr(),
            x_b.ptr(),
            table.as_ptr() as i64,
            N_LAYERS as i64,
            gamma_ptr,
            lm_ptr,
            slot,
            pos as i64,
            rng_seed,
            0, // grammar_state (no grammar registered)
            tok_dev.ptr(),
        );
        if rc != 0 {
            panic_with_jit_log("nsl_cfie_decode_step", rc, ptx);
        }
        gpu_tokens.push(d2h_u32_one(tok_dev.ptr()));

        // CPU mirror: per-layer decode_block chain (f16-rounded weights),
        // then the sampler on f16-rounded lm_head + exact gamma.
        let x1 = cpu_block(&cfg, &x, &l0.rounded, &mut kv0_k, &mut kv0_v, pos);
        let x2 = cpu_block(&cfg, &x1, &l1.rounded, &mut kv1_k, &mut kv1_v, pos);
        cpu_tokens.push(cpu_reference_sample(
            program,
            &x2,
            &fx.gamma,
            &fx.lm_rounded,
            None,
            rng_seed as u64,
        ));
    }
    (gpu_tokens, cpu_tokens)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// THE flagship proof: every weight uploaded through the PRODUCTION
/// FFIs, then the four-token decode sequence must EXACTLY equal the CPU
/// reference chain — proving FFI-uploaded weights drive decode_step
/// correctly.  Then reset frees cleanly and a fresh bind reproduces the
/// same sequence.
#[test]
#[ignore = "requires CUDA GPU"]
fn ffi_uploaded_weights_drive_decode_step_matches_cpu_chain() {
    if !cuda_available() {
        return;
    }
    let (program, s_ptx, s_name, s_block) = build_sampler();
    let ptx = setup_engine((&s_ptx, &s_name, s_block));

    // Fresh, uploaded-weight state: reset any leftover weight allocs.
    assert_eq!(nsl_cfie_weights_reset(), 0);

    let slot = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot, 0);

    let l0_master = seeded_layer_weights(0xA0);
    let l1_master = seeded_layer_weights(0xA1);
    let fx = sample_fixtures();

    // --- Phase 1: bind via the production FFIs and drive 4 tokens. ---
    let l0 = bind_layer(&l0_master);
    let l1 = bind_layer(&l1_master);

    let x_a = DevBuf::alloc(D_MODEL * 4);
    let x_b = DevBuf::alloc(D_MODEL * 4);
    let tok_dev = DevBuf::alloc(4);

    let (gpu_tokens, cpu_tokens) =
        run_four_tokens(&ptx, &program, &l0, &l1, &fx, slot, &x_a, &x_b, &tok_dev);
    assert_eq!(
        gpu_tokens, cpu_tokens,
        "FFI-uploaded weights: decode-loop token sequence must match the CPU chain"
    );
    eprintln!(
        "phase 1 (FFI upload) decode_step token sequence: {:?}",
        gpu_tokens
    );

    // --- Phase 2: reset frees the weights, then a fresh bind on a fresh
    // cold slot must reproduce the identical token sequence (proves reset
    // released cleanly and the FFI path is re-runnable). ---
    assert_eq!(nsl_cfie_weights_reset(), 0);
    // A second reset is idempotent (nothing left to free).
    assert_eq!(nsl_cfie_weights_reset(), 0);

    let slot2 = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot2, 1, "second cold slot");

    let l0b = bind_layer(&l0_master);
    let l1b = bind_layer(&l1_master);
    let (gpu_tokens2, cpu_tokens2) =
        run_four_tokens(&ptx, &program, &l0b, &l1b, &fx, slot2, &x_a, &x_b, &tok_dev);
    assert_eq!(
        gpu_tokens2, cpu_tokens2,
        "after reset+re-bind: token sequence must match the CPU chain"
    );
    assert_eq!(
        gpu_tokens2, gpu_tokens,
        "re-bind must reproduce the identical token sequence"
    );
    eprintln!(
        "phase 2 (reset + re-bind) decode_step token sequence: {:?}",
        gpu_tokens2
    );

    // engine_destroy frees the KV pool AND every remaining weight alloc.
    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

/// Negative contract: an upload with `n_elems == 0` refuses with -1 and
/// records no allocation.  (The full bad-arg matrix — null ptr, negative
/// n_elems, byte-count overflow — is unit-tested CPU-side in engine.rs;
/// this pins the contract on the GPU build too.)
#[test]
#[ignore = "requires CUDA GPU"]
fn upload_zero_elems_refuses_minus1() {
    if !cuda_available() {
        return;
    }
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);

    let host = [1.0f32, 2.0, 3.0, 4.0];
    let hp = host.as_ptr() as i64;
    assert_eq!(
        nsl_cfie_upload_weight_f16(hp, 0),
        -1,
        "f16 upload of 0 elems must refuse with -1"
    );
    assert_eq!(
        nsl_cfie_upload_weight_f32(hp, 0),
        -1,
        "f32 upload of 0 elems must refuse with -1"
    );
    assert_eq!(
        nsl_cfie_upload_weight_f16(0, 4),
        -1,
        "f16 upload with null host ptr must refuse with -1"
    );
    // Nothing recorded => reset is a clean no-op.
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_engine_destroy(), 0);
}
