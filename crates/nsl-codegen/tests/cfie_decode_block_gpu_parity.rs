//! CFIE Cycle 7: GPU numeric parity for the persistent decode-block
//! kernel (kind 2) and the nsl_cfie_decode_step host loop through the
//! REAL engine path: register -> kv_slots_init -> kv_pool_alloc ->
//! finalize -> launch.
//!
//! First hardware execution of the kernel family.  Two proofs:
//!
//!   1. chained direct launches at pos = 0..3 against
//!      cfie_persistent_ptx::cpu_reference's KV-append chain
//!      (f16-rounded weights fed to the CPU side);
//!   2. THE flagship decode-loop proof: nsl_cfie_decode_step drives
//!      kinds 2 + 1 for four greedy tokens and the TOKEN SEQUENCE must
//!      equal the CPU chain (cpu decode_block per layer + cpu sampler
//!      reference per token), plus slot bookkeeping (+4 advance) and
//!      the -2 capacity refusal at per-slot exhaustion.
//!
//! Tolerance: 5e-3 max-abs at pos 0-1 (f16 weights + ex2/rsqrt/sin/cos
//! .approx, PCA/CSHA hd<=32 budget) widening to 1e-2 by pos 2-3 —
//! the kernel's KV cache carries ITS OWN approx-rounded K/V rows for
//! all earlier positions while the CPU chain appends exact-math rows,
//! so the divergence compounds with sequence position.
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_decode_block_gpu_parity \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_persistent_ptx::{cpu_reference, emit, DecodeBlockConfig};
use nsl_codegen::cfie_sample_ptx::{
    cpu_reference as cpu_reference_sample, emit as emit_sampler, FusedSampleKernelConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_decode_step, nsl_cfie_engine_destroy, nsl_cfie_engine_finalize,
    nsl_cfie_kv_pool_alloc, nsl_cfie_launch_decode_block, nsl_cfie_register_kernel,
};
use nsl_runtime::cfie::ffi::{
    nsl_cfie_kv_slot_acquire, nsl_cfie_kv_slot_advance, nsl_cfie_kv_slots_init,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
    nsl_test_cuda_jit_log,
};

// ---------------------------------------------------------------------------
// CUDA availability guard (house convention: per-file copy)
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
// f16 <-> f32 bit converters (per-file copy, house convention)
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
// Deterministic PRNG (same LCG as pca_tier_a_forward_correctness.rs,
// with a corrected ZERO-MEAN mapping: the precedent's `(seed >> 33) as
// u32 / u32::MAX` only spans [-0.5, 0) — the mean -0.25 bias is
// harmless for attention fixtures but saturates every SwiGLU gate in a
// decode block (gate ~= +14, down-proj ~= -8000), pushing outputs far
// past where an absolute 5e-3 budget is meaningful).
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
// Device-buffer RAII: every buffer freed on every path.
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

fn h2d_u16(dst: i64, src: &[u16]) {
    nsl_test_cuda_h2d(dst, src.as_ptr() as i64, (src.len() * 2) as i64);
}

fn d2h_f32(src: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, src, (len * 4) as i64);
    out
}

fn d2h_u32_one(src: i64) -> u32 {
    let mut out = [0u32; 1];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, src, 4);
    out[0]
}

/// Upload an f32 weight matrix as f16 bits; returns (device buf,
/// f16-rounded f32 copy for the CPU reference).
fn upload_f16(w: &[f32]) -> (DevBuf, Vec<f32>) {
    let bits: Vec<u16> = w.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let dev = DevBuf::alloc(bits.len() * 2);
    h2d_u16(dev.ptr(), &bits);
    let rounded = w.iter().map(|&x| f16_round(x)).collect();
    (dev, rounded)
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
// Fixture: 2-layer toy block; per-slot 4 tokens so the decode_step test
// can hit the capacity refusal after its 4 chained tokens.
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

/// One layer's host-side weights (f32 masters, seeded).
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
    // residual stream O(1) so the absolute parity budget is meaningful.
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

/// Device copies of one layer's weights: 7 matmul matrices as f16, the
/// two norm gammas as f32.  `rounded` holds the f16-rounded f32 masters
/// the CPU reference consumes (norm gammas pass through un-rounded —
/// the kernel loads them as f32).
struct DevLayerWeights {
    bufs: Vec<DevBuf>, // wq wk wv wo w_gate w_up w_down norm1 norm2
    rounded: LayerWeights,
}

impl DevLayerWeights {
    fn upload(w: &LayerWeights) -> Self {
        let mut bufs = Vec::with_capacity(9);
        let mut round = |m: &[f32]| -> Vec<f32> {
            let (dev, r) = upload_f16(m);
            bufs.push(dev);
            r
        };
        let wq = round(&w.wq);
        let wk = round(&w.wk);
        let wv = round(&w.wv);
        let wo = round(&w.wo);
        let w_gate = round(&w.w_gate);
        let w_up = round(&w.w_up);
        let w_down = round(&w.w_down);
        // Norm gammas: f32 on device, exact on CPU.
        let n1 = DevBuf::alloc(w.norm1.len() * 4);
        h2d_f32(n1.ptr(), &w.norm1);
        bufs.push(n1);
        let n2 = DevBuf::alloc(w.norm2.len() * 4);
        h2d_f32(n2.ptr(), &w.norm2);
        bufs.push(n2);
        DevLayerWeights {
            bufs,
            rounded: LayerWeights {
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                norm1: w.norm1.clone(),
                norm2: w.norm2.clone(),
            },
        }
    }

    fn ptrs(&self) -> [u64; 9] {
        let mut p = [0u64; 9];
        for (i, b) in self.bufs.iter().enumerate() {
            p[i] = b.ptr() as u64;
        }
        p
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

/// Full engine bring-up: register kind 2 (and optionally kind 1), init
/// slots, allocate the pool, finalize.  Returns the decode-block PTX.
fn setup_engine(sampler: Option<(&str, &str, u32)>) -> String {
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
    let mut expected = 1i64;
    if let Some((s_ptx, s_name, s_block)) = sampler {
        let rc = nsl_cfie_register_kernel(
            1,
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
        expected = 2;
    }
    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, PER_SLOT as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(POOL_BYTES as i64), 0);
    let n = nsl_cfie_engine_finalize();
    if n != expected {
        panic_with_jit_log("nsl_cfie_engine_finalize (decode_block)", n, &ptx);
    }
    ptx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_block_chained_positions_match_cpu_kv_append_chain() {
    if !cuda_available() {
        return;
    }
    let ptx = setup_engine(None);
    // Slot 1 + layer 1: mid-pool offsets on both axes.
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);
    let slot = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot, 1);
    let layer = 1i64;

    let cfg = block_cfg();
    let weights = seeded_layer_weights(0xB10C);
    let dev = DevLayerWeights::upload(&weights);
    let p = dev.ptrs();

    let x_in = DevBuf::alloc(D_MODEL * 4);
    let x_out = DevBuf::alloc(D_MODEL * 4);

    let mut kv_k: Vec<f32> = Vec::new();
    let mut kv_v: Vec<f32> = Vec::new();
    for pos in 0..PER_SLOT as u32 {
        let mut x = vec![0f32; D_MODEL];
        fill_seeded(&mut x, 0x9000 + pos as u64);
        h2d_f32(x_in.ptr(), &x);

        let rc = nsl_cfie_launch_decode_block(
            x_in.ptr(),
            x_out.ptr(),
            p[0] as i64,
            p[1] as i64,
            p[2] as i64,
            p[3] as i64,
            p[4] as i64,
            p[5] as i64,
            p[6] as i64,
            p[7] as i64,
            p[8] as i64,
            layer,
            slot,
            pos as i64,
        );
        if rc != 0 {
            panic_with_jit_log("nsl_cfie_launch_decode_block", rc, &ptx);
        }
        let gpu = d2h_f32(x_out.ptr(), D_MODEL);
        let cpu = cpu_block(&cfg, &x, &dev.rounded, &mut kv_k, &mut kv_v, pos);

        // 5e-3 at pos 0-1; 1e-2 by pos 2-3: the kernel's KV rows for
        // earlier positions carry its own f16+approx rounding while the
        // CPU chain appends exact-math rows, so drift compounds.
        let budget = if pos < 2 { 5e-3 } else { 1e-2 };
        let mut max_abs = 0f32;
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let d = (g - c).abs();
            if d > max_abs {
                max_abs = d;
            }
            assert!(
                d <= budget,
                "decode_block pos {} out[{}]: gpu {} vs cpu {} (diff {}, budget {})",
                pos,
                i,
                g,
                c,
                d,
                budget
            );
        }
        eprintln!(
            "decode_block chain parity (layer {}, slot {}, pos {}): max_abs = {:.3e}",
            layer, slot, pos, max_abs
        );
    }

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_step_token_sequence_matches_cpu_chain() {
    if !cuda_available() {
        return;
    }
    // Flagship decode-loop proof: kinds 2 + 1 through the host loop.
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
    let ptx = setup_engine(Some((&s_ptx, &s_meta.kernel_name, s_meta.block_dim)));

    let slot = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot, 0);

    let cfg = block_cfg();
    let l0 = seeded_layer_weights(0xA0);
    let l1 = seeded_layer_weights(0xA1);
    let dev0 = DevLayerWeights::upload(&l0);
    let dev1 = DevLayerWeights::upload(&l1);
    // HOST weights table: n_layers records x 9 u64 device pointers.
    let mut table = [0u64; N_LAYERS * 9];
    table[..9].copy_from_slice(&dev0.ptrs());
    table[9..].copy_from_slice(&dev1.ptrs());

    let mut gamma = vec![0f32; D_MODEL];
    fill_seeded(&mut gamma, 0xF1);
    for g in gamma.iter_mut() {
        *g = 1.0 + 0.1 * *g;
    }
    let mut lm_head = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut lm_head, 0xF2);

    let x_a = DevBuf::alloc(D_MODEL * 4);
    let x_b = DevBuf::alloc(D_MODEL * 4);
    let gamma_dev = DevBuf::alloc(D_MODEL * 4);
    h2d_f32(gamma_dev.ptr(), &gamma);
    let lm_bits: Vec<u16> = lm_head.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let lm_dev = DevBuf::alloc(lm_bits.len() * 2);
    h2d_u16(lm_dev.ptr(), &lm_bits);
    let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
    let tok_dev = DevBuf::alloc(4);

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
            gamma_dev.ptr(),
            lm_dev.ptr(),
            slot,
            pos as i64,
            rng_seed,
            0, // grammar_state (no grammar registered)
            tok_dev.ptr(),
        );
        if rc != 0 {
            panic_with_jit_log("nsl_cfie_decode_step", rc, &ptx);
        }
        gpu_tokens.push(d2h_u32_one(tok_dev.ptr()));

        // CPU mirror: per-layer decode_block chain, then the sampler.
        let x1 = cpu_block(&cfg, &x, &dev0.rounded, &mut kv0_k, &mut kv0_v, pos);
        let x2 = cpu_block(&cfg, &x1, &dev1.rounded, &mut kv1_k, &mut kv1_v, pos);
        cpu_tokens.push(cpu_reference_sample(
            &program,
            &x2,
            &gamma,
            &lm_rounded,
            None,
            rng_seed as u64,
        ));
    }
    assert_eq!(
        gpu_tokens, cpu_tokens,
        "decode-loop token sequence must match the CPU chain"
    );
    eprintln!("decode_step token sequence: {:?}", gpu_tokens);

    // Slot bookkeeping advanced by exactly 4 (advance-by-0 probes len).
    assert_eq!(
        nsl_cfie_kv_slot_advance(slot, 0),
        PER_SLOT as i64,
        "slot seq_len must have advanced by 4"
    );
    // Capacity refusal: the slot is full, the 5th token refuses with -2
    // before any launch.
    let rc = nsl_cfie_decode_step(
        x_a.ptr(),
        x_b.ptr(),
        table.as_ptr() as i64,
        N_LAYERS as i64,
        gamma_dev.ptr(),
        lm_dev.ptr(),
        slot,
        PER_SLOT as i64,
        7,
        0,
        tok_dev.ptr(),
    );
    assert_eq!(rc, -2, "per-slot exhaustion must refuse with -2");
    assert_eq!(
        nsl_cfie_kv_slot_advance(slot, 0),
        PER_SLOT as i64,
        "refusal must not disturb the booked length"
    );

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_step_pos_mismatch_refuses_minus3_and_rolls_back() {
    if !cuda_available() {
        return;
    }
    // G21 edge: nsl_cfie_decode_step books the token via
    // kv_slot_advance(slot, 1) BEFORE any launch; if the resulting length
    // disagrees with the caller's pos (`new_len != pos + 1`), it MUST
    // un-book (rollback) and return -3 without corrupting slot state.
    // Proof: a good pos-0 step, a bad pos-5 step that must -3 + roll back,
    // then a good pos-1 step whose token still matches the CPU chain.
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
    let ptx = setup_engine(Some((&s_ptx, &s_meta.kernel_name, s_meta.block_dim)));

    let slot = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot, 0);

    let cfg = block_cfg();
    let l0 = seeded_layer_weights(0xA0);
    let l1 = seeded_layer_weights(0xA1);
    let dev0 = DevLayerWeights::upload(&l0);
    let dev1 = DevLayerWeights::upload(&l1);
    let mut table = [0u64; N_LAYERS * 9];
    table[..9].copy_from_slice(&dev0.ptrs());
    table[9..].copy_from_slice(&dev1.ptrs());

    let mut gamma = vec![0f32; D_MODEL];
    fill_seeded(&mut gamma, 0xF1);
    for g in gamma.iter_mut() {
        *g = 1.0 + 0.1 * *g;
    }
    let mut lm_head = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut lm_head, 0xF2);

    let x_a = DevBuf::alloc(D_MODEL * 4);
    let x_b = DevBuf::alloc(D_MODEL * 4);
    let gamma_dev = DevBuf::alloc(D_MODEL * 4);
    h2d_f32(gamma_dev.ptr(), &gamma);
    let lm_bits: Vec<u16> = lm_head.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let lm_dev = DevBuf::alloc(lm_bits.len() * 2);
    h2d_u16(lm_dev.ptr(), &lm_bits);
    let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
    let tok_dev = DevBuf::alloc(4);

    let (mut kv0_k, mut kv0_v) = (Vec::<f32>::new(), Vec::<f32>::new());
    let (mut kv1_k, mut kv1_v) = (Vec::<f32>::new(), Vec::<f32>::new());

    // Helper: one GPU decode_step at `pos` with a per-pos seeded x.
    let step = |pos: i64, x_host: &[f32]| -> i64 {
        h2d_f32(x_a.ptr(), x_host);
        nsl_cfie_decode_step(
            x_a.ptr(),
            x_b.ptr(),
            table.as_ptr() as i64,
            N_LAYERS as i64,
            gamma_dev.ptr(),
            lm_dev.ptr(),
            slot,
            pos,
            0x5EED_0000 + pos,
            0,
            tok_dev.ptr(),
        )
    };
    // Helper: the CPU token for `pos`, advancing the per-layer KV chains.
    let cpu_token = |pos: u32,
                         x_host: &[f32],
                         kv0_k: &mut Vec<f32>,
                         kv0_v: &mut Vec<f32>,
                         kv1_k: &mut Vec<f32>,
                         kv1_v: &mut Vec<f32>|
     -> u32 {
        let x1 = cpu_block(&cfg, x_host, &dev0.rounded, kv0_k, kv0_v, pos);
        let x2 = cpu_block(&cfg, &x1, &dev1.rounded, kv1_k, kv1_v, pos);
        cpu_reference_sample(
            &program,
            &x2,
            &gamma,
            &lm_rounded,
            None,
            (0x5EED_0000 + pos as i64) as u64,
        )
    };

    // (1) pos 0: a clean step. Slot len advances 0 -> 1.
    let mut x0 = vec![0f32; D_MODEL];
    fill_seeded(&mut x0, 0xD00D);
    let rc0 = step(0, &x0);
    if rc0 != 0 {
        panic_with_jit_log("nsl_cfie_decode_step (pos 0)", rc0, &ptx);
    }
    let gpu0 = d2h_u32_one(tok_dev.ptr());
    let cpu0 = cpu_token(0, &x0, &mut kv0_k, &mut kv0_v, &mut kv1_k, &mut kv1_v);
    assert_eq!(gpu0, cpu0, "pos 0 token must match the CPU chain");
    assert_eq!(
        nsl_cfie_kv_slot_advance(slot, 0),
        1,
        "slot len must be 1 after the pos-0 step"
    );

    // (2) pos 5 with current len 1: advance books to 2, 2 != 5 + 1, so
    // the step MUST roll back and return -3. The CPU KV chains are NOT
    // touched (the GPU rolled its book-keeping AND wrote no KV rows).
    let mut x_bad = vec![0f32; D_MODEL];
    fill_seeded(&mut x_bad, 0xBAD5);
    let rc_bad = step(5, &x_bad);
    assert_eq!(rc_bad, -3, "pos/len mismatch must refuse with -3");
    assert_eq!(
        nsl_cfie_kv_slot_advance(slot, 0),
        1,
        "the rejected step's advance must have been rolled back (len still 1)"
    );

    // (3) pos 1: the correct next position now succeeds and its token
    // still matches the CPU chain — proving the failed step corrupted
    // neither the slot book-keeping nor the device KV cache.
    let mut x1 = vec![0f32; D_MODEL];
    fill_seeded(&mut x1, 0xD00E);
    let rc1 = step(1, &x1);
    if rc1 != 0 {
        panic_with_jit_log("nsl_cfie_decode_step (pos 1 after refusal)", rc1, &ptx);
    }
    let gpu1 = d2h_u32_one(tok_dev.ptr());
    let cpu1 = cpu_token(1, &x1, &mut kv0_k, &mut kv0_v, &mut kv1_k, &mut kv1_v);
    assert_eq!(
        gpu1, cpu1,
        "pos 1 token must match the CPU chain — no state corruption from the -3 refusal"
    );
    assert_eq!(
        nsl_cfie_kv_slot_advance(slot, 0),
        2,
        "slot len must be 2 after the recovered pos-1 step"
    );
    eprintln!(
        "decode_step -3 recovery: tokens [{}, {}] (pos-5 refused with -3, rolled back)",
        gpu0, gpu1
    );

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}
