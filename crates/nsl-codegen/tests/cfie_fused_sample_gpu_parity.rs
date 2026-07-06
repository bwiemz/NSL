//! CFIE Cycle 7: GPU numeric parity for the fused decode-sample kernel
//! (kind 1) through the REAL engine path: register -> finalize ->
//! nsl_cfie_launch_fused_sample.
//!
//! First hardware execution of the kernel family.  Token outputs are
//! asserted EXACTLY equal to cfie_sample_ptx::cpu_reference (fed
//! f16-ROUNDED lm-head weights): the greedy argmax is deterministic and
//! the stochastic paths share a bit-for-bit xorshift64* PRNG with the
//! CPU mirror.  Softmax internals use ex2/rsqrt.approx so probabilities
//! are never asserted — only the sampled/argmax token.
//!
//! The grammar-masked test splices the DFA mask .global into the
//! sampler module and finalizes — the repo's first real
//! cuModuleGetGlobal binding.
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_fused_sample_gpu_parity \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_grammar::{compile, GrammarSpec};
use nsl_codegen::cfie_grammar_ptx::{emit_mask_global, mask_bytes, splice_mask_into_module};
use nsl_codegen::cfie_sample_ptx::{cpu_reference, emit, FusedSampleKernelConfig};
use nsl_runtime::cfie::engine::{
    nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_launch_fused_sample,
    nsl_cfie_register_kernel,
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
// u32 / u32::MAX` only spans [-0.5, 0), whose mean -0.25 bias blows up
// matmul-chain fixtures — kept zero-mean across all five CFIE parity
// suites).
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
// Fixture
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const VOCAB: usize = 256;
const TOP_K: u32 = 8;

fn shape(vocab: u32) -> LmHeadShape {
    LmHeadShape {
        d_model: D_MODEL as u32,
        vocab_size: vocab,
        vocab_tile: 128,
        dtype_bytes: 2,
    }
}

fn kernel_cfg(vocab: u32, grammar_states: u32) -> FusedSampleKernelConfig {
    FusedSampleKernelConfig {
        d_model: D_MODEL as u32,
        vocab_size: vocab,
        vocab_tile: 128,
        top_k: TOP_K,
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
        grammar_states,
    }
}

/// Destroy any prior engine state, register the given sampler module as
/// the kind-1 kernel, finalize.  Returns nothing — the PTX stays with
/// the caller for JIT-log diagnostics.
fn setup_engine(ptx: &str, kernel_name: &str, block_dim: u32) {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let rc = nsl_cfie_register_kernel(
        1, // kind 1 = fused_sample
        0,
        ptx.as_ptr() as i64,
        ptx.len() as i64,
        kernel_name.as_ptr() as i64,
        kernel_name.len() as i64,
        1, // single-CTA latency path
        block_dim as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 1) refused");
    let n = nsl_cfie_engine_finalize();
    if n != 1 {
        panic_with_jit_log("nsl_cfie_engine_finalize (fused_sample)", n, ptx);
    }
}

/// Seeded fixture data: hidden [d_model], gamma [d_model] near 1.0,
/// lm_head f32 [vocab][d_model] in [-0.5, 0.5).
fn fixture_data(vocab: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut hidden = vec![0f32; D_MODEL];
    let mut gamma_raw = vec![0f32; D_MODEL];
    let mut lm_head = vec![0f32; vocab * D_MODEL];
    fill_seeded(&mut hidden, seed);
    fill_seeded(&mut gamma_raw, seed.wrapping_add(1));
    fill_seeded(&mut lm_head, seed.wrapping_add(2));
    // Gamma near 1.0 keeps the RMSNorm well-conditioned.
    let gamma: Vec<f32> = gamma_raw.iter().map(|&g| 1.0 + 0.1 * g).collect();
    (hidden, gamma, lm_head)
}

/// Upload the fixture, launch through the engine, download the token.
fn launch_once(
    ptx: &str,
    hidden: &[f32],
    gamma: &[f32],
    lm_head_f32: &[f32],
    rng_seed: u64,
    grammar_state: u32,
) -> u32 {
    let lm_bits: Vec<u16> = lm_head_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();

    let hidden_dev = DevBuf::alloc(hidden.len() * 4);
    let gamma_dev = DevBuf::alloc(gamma.len() * 4);
    let lm_dev = DevBuf::alloc(lm_bits.len() * 2);
    let tok_dev = DevBuf::alloc(4);
    h2d_f32(hidden_dev.ptr(), hidden);
    h2d_f32(gamma_dev.ptr(), gamma);
    h2d_u16(lm_dev.ptr(), &lm_bits);

    let rc = nsl_cfie_launch_fused_sample(
        hidden_dev.ptr(),
        gamma_dev.ptr(),
        lm_dev.ptr(),
        tok_dev.ptr(),
        rng_seed as i64,
        grammar_state as i64,
    );
    if rc != 0 {
        panic_with_jit_log("nsl_cfie_launch_fused_sample", rc, ptx);
    }
    d2h_u32_one(tok_dev.ptr())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA GPU"]
fn fused_sample_greedy_argmax_matches_cpu_exactly() {
    if !cuda_available() {
        return;
    }
    let params = SamplingParams {
        strategy: SamplingStrategy::Greedy,
        temperature: 0.0,
        top_k: TOP_K,
        ..Default::default()
    };
    let program = emit_program(params, shape(VOCAB as u32));
    let (ptx, meta) = emit(&program, &kernel_cfg(VOCAB as u32, 0));
    setup_engine(&ptx, &meta.kernel_name, meta.block_dim);

    for seed in [0x11u64, 0x22, 0x33, 0x44, 0x55] {
        let (hidden, gamma, lm_head) = fixture_data(VOCAB, seed);
        let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
        // Greedy must ignore the rng seed — pass a varying one anyway.
        let gpu = launch_once(&ptx, &hidden, &gamma, &lm_head, seed ^ 0xF00D, 0);
        let cpu = cpu_reference(&program, &hidden, &gamma, &lm_rounded, None, seed ^ 0xF00D);
        assert_eq!(
            gpu, cpu,
            "greedy token mismatch for fixture seed {:#x}: gpu {} vs cpu {}",
            seed, gpu, cpu
        );
        assert!((gpu as usize) < VOCAB);
        eprintln!("fused_sample greedy (fixture {:#x}): token {}", seed, gpu);
    }

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn fused_sample_top_k_stochastic_matches_cpu_exactly() {
    if !cuda_available() {
        return;
    }
    let params = SamplingParams {
        strategy: SamplingStrategy::TopK,
        temperature: 0.8,
        top_k: TOP_K,
        ..Default::default()
    };
    let program = emit_program(params, shape(VOCAB as u32));
    let (ptx, meta) = emit(&program, &kernel_cfg(VOCAB as u32, 0));
    setup_engine(&ptx, &meta.kernel_name, meta.block_dim);

    let (hidden, gamma, lm_head) = fixture_data(VOCAB, 0xABCD);
    let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
    let mut seen = std::collections::BTreeSet::new();
    for rng_seed in [1u64, 7, 42, 0xDEAD_BEEF, 0x1234_5678_9ABC_DEF0] {
        let gpu = launch_once(&ptx, &hidden, &gamma, &lm_head, rng_seed, 0);
        let cpu = cpu_reference(&program, &hidden, &gamma, &lm_rounded, None, rng_seed);
        assert_eq!(
            gpu, cpu,
            "top-k token mismatch for rng seed {:#x}: gpu {} vs cpu {} (bit-for-bit PRNG)",
            rng_seed, gpu, cpu
        );
        seen.insert(gpu);
        eprintln!("fused_sample top-k (rng {:#x}): token {}", rng_seed, gpu);
    }
    // Sanity: the stochastic path is actually stochastic across seeds.
    assert!(
        seen.len() > 1,
        "5 seeds all sampled token {:?} — suspicious for a stochastic path",
        seen
    );

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn fused_sample_top_k_top_p_nucleus_matches_cpu_exactly() {
    if !cuda_available() {
        return;
    }
    // Exercises the insertion-sort + nucleus-cutoff kernel sections.
    let params = SamplingParams {
        strategy: SamplingStrategy::TopKTopP,
        temperature: 0.7,
        top_k: TOP_K,
        top_p: 0.9,
        ..Default::default()
    };
    let program = emit_program(params, shape(VOCAB as u32));
    let (ptx, meta) = emit(&program, &kernel_cfg(VOCAB as u32, 0));
    setup_engine(&ptx, &meta.kernel_name, meta.block_dim);

    let (hidden, gamma, lm_head) = fixture_data(VOCAB, 0x5EED);
    let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
    for rng_seed in [3u64, 99, 0xCAFE, 0xFFFF_FFFF_FFFF_FFFF] {
        let gpu = launch_once(&ptx, &hidden, &gamma, &lm_head, rng_seed, 0);
        let cpu = cpu_reference(&program, &hidden, &gamma, &lm_rounded, None, rng_seed);
        assert_eq!(
            gpu, cpu,
            "top-k/top-p token mismatch for rng seed {:#x}: gpu {} vs cpu {}",
            rng_seed, gpu, cpu
        );
        eprintln!("fused_sample top-k/top-p (rng {:#x}): token {}", rng_seed, gpu);
    }

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn fused_sample_grammar_mask_binds_global_and_constrains_tokens() {
    if !cuda_available() {
        return;
    }
    // 128-token vocab (one tile, e2e precedent); sequence DFA 42 -> 7:
    // state 0 allows ONLY token 42, state 1 allows ONLY token 7.
    const GVOCAB: usize = 128;
    let dfa = compile(&GrammarSpec::sequence(&[42, 7], GVOCAB as u32)).expect("DFA compiles");
    let mask = mask_bytes(&dfa);

    let params = SamplingParams {
        strategy: SamplingStrategy::Greedy,
        temperature: 0.0,
        top_k: TOP_K,
        grammar_masked: true,
        ..Default::default()
    };
    let program = emit_program(params, shape(GVOCAB as u32));
    let (sampler_ptx, meta) = emit(&program, &kernel_cfg(GVOCAB as u32, dfa.num_states));
    let spliced = splice_mask_into_module(&sampler_ptx, &emit_mask_global(&dfa));
    assert!(spliced.contains("nsl_cfie_grammar_mask["));
    // Finalize resolves the mask .global via cuModuleGetGlobal — the
    // repo's first hardware execution of that binding.
    setup_engine(&spliced, &meta.kernel_name, meta.block_dim);

    let (hidden, gamma, lm_head_raw) = fixture_data(GVOCAB, 0x6AAA);
    // Force the UNMASKED argmax onto token 100 (grammar-illegal in every
    // DFA state) so the mask observably changes the outcome.
    let mut lm_head = lm_head_raw;
    for d in 0..D_MODEL {
        lm_head[100 * D_MODEL + d] = if hidden[d] >= 0.0 { 0.5 } else { -0.5 };
    }
    let lm_rounded: Vec<f32> = lm_head.iter().map(|&x| f16_round(x)).collect();
    let unmasked_argmax = cpu_reference(&program, &hidden, &gamma, &lm_rounded, None, 1);
    assert_eq!(unmasked_argmax, 100, "fixture must argmax on token 100 unmasked");

    let row_bytes = GVOCAB / 8;
    for (state, expect) in [(0u32, 42u32), (1, 7)] {
        let gpu = launch_once(&spliced, &hidden, &gamma, &lm_head, 1, state);
        let cpu = cpu_reference(&program, &hidden, &gamma, &lm_rounded, Some((&mask, state)), 1);
        assert_eq!(
            gpu, cpu,
            "grammar-masked token mismatch in state {}: gpu {} vs cpu {}",
            state, gpu, cpu
        );
        assert_eq!(
            gpu, expect,
            "sequence DFA state {} legalizes exactly token {}",
            state, expect
        );
        assert_ne!(gpu, unmasked_argmax, "mask must veto the unmasked argmax");
        // Grammar-legal per the mask bytes (LSB-first bit order).
        let bit = (mask[state as usize * row_bytes + gpu as usize / 8] >> (gpu & 7)) & 1;
        assert_eq!(bit, 1, "sampled token {} must be legal in state {}", gpu, state);
        eprintln!("fused_sample grammar (state {}): token {}", state, gpu);
    }

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}
