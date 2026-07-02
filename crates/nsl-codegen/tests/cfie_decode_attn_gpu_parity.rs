//! CFIE Cycle 7: GPU numeric parity for the direct-indexing
//! decode-attention kernel (kind 0) through the REAL engine path:
//! register -> kv_slots_init -> kv_pool_alloc -> finalize ->
//! nsl_cfie_launch_decode_attn.
//!
//! First hardware execution of the kernel family: K/V rows are seeded
//! into the engine-owned pool at hand-computed byte offsets of the
//! uniform layout [n_layers][2][max_tokens][n_kv_heads][head_dim] f16
//! and the output is compared against
//! cfie_decode_attention::cpu_reference fed f16-ROUNDED K/V.
//!
//! Tolerance: 5e-3 max-abs (f16 mantissa + ex2.approx softmax, same
//! budget as PCA/CSHA head_dim<=32 forward parity).
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_decode_attn_gpu_parity \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::cfie_decode_attention::{cpu_reference, emit, DecodeAttentionConfig};
use nsl_runtime::cfie::engine::{
    nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_kv_pool_alloc,
    nsl_cfie_kv_pool_base, nsl_cfie_launch_decode_attn, nsl_cfie_register_kernel,
};
use nsl_runtime::cfie::ffi::{nsl_cfie_kv_slot_acquire, nsl_cfie_kv_slots_init};
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
// Device-buffer RAII so every buffer is freed on every path (the
// 2026-05-27 PCA flakiness was root-caused to leaked test buffers).
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

// ---------------------------------------------------------------------------
// Failure diagnostics: on rc != 0 print the driver JIT log, then panic
// (hard-fail — cuda_available already passed, soft-skips mask breakage).
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
// Fixture: small 2-layer config; strides hand-derived below and used
// for the host-side pool uploads.
// ---------------------------------------------------------------------------

const N_LAYERS: usize = 2;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const PER_SLOT: usize = 16;
const MAX_SLOTS: usize = 2;

// Byte strides of [n_layers][2][max_tokens][n_kv_heads][head_dim] f16.
const TOKEN_STRIDE_BYTES: usize = N_KV_HEADS * HEAD_DIM * 2; // 128
const KV_HALF_BYTES: usize = MAX_SLOTS * PER_SLOT * N_KV_HEADS * HEAD_DIM * 2; // 4096
const LAYER_BYTES: usize = 2 * KV_HALF_BYTES; // 8192
const POOL_BYTES: usize = N_LAYERS * LAYER_BYTES; // 16384

fn fixture_cfg() -> DecodeAttentionConfig {
    DecodeAttentionConfig {
        n_layers: N_LAYERS as u32,
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        per_slot_max_tokens: PER_SLOT as u32,
        max_slots: MAX_SLOTS as u32,
        kv_dtype_bytes: 2,
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
    }
}

/// Full engine bring-up for one test: destroy any prior state, emit +
/// register the kernel, init slots, allocate + zero the pool, finalize.
/// Returns the PTX (for JIT-log diagnostics on launch failure).
fn setup_engine() -> String {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let (ptx, meta) = emit(&fixture_cfg());
    let name = meta.kernel_name.as_str();
    let rc = nsl_cfie_register_kernel(
        0,
        0,
        ptx.as_ptr() as i64,
        ptx.len() as i64,
        name.as_ptr() as i64,
        name.len() as i64,
        N_HEADS as i64, // grid_dim_is_n_heads
        meta.block_dim as i64,
        0, // all CFIE kernels use static .shared
    );
    assert_eq!(rc, 0, "register_kernel(kind 0) refused");
    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, PER_SLOT as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(POOL_BYTES as i64), 0);
    let n = nsl_cfie_engine_finalize();
    if n != 1 {
        panic_with_jit_log("nsl_cfie_engine_finalize (decode_attn)", n, &ptx);
    }
    ptx
}

/// Upload one slot's K/V token rows (f32 -> f16 bits) into the pool at
/// the hand-computed (layer, slot) byte offsets.
fn upload_kv(layer: usize, slot: usize, k_f32: &[f32], v_f32: &[f32]) {
    let base = nsl_cfie_kv_pool_base();
    assert!(base != 0, "pool base accessor returned 0 after pool_alloc");
    let k_bits: Vec<u16> = k_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let v_bits: Vec<u16> = v_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let slot_off = slot * PER_SLOT * TOKEN_STRIDE_BYTES;
    let k_off = base + (layer * LAYER_BYTES + slot_off) as i64;
    let v_off = base + (layer * LAYER_BYTES + KV_HALF_BYTES + slot_off) as i64;
    h2d_u16(k_off, &k_bits);
    h2d_u16(v_off, &v_bits);
}

/// Launch through the engine FFI and compare against cpu_reference fed
/// f16-rounded K/V.  `k_f32`/`v_f32` are one slot's contiguous
/// [seq_len][n_kv_heads][head_dim] rows.
fn run_and_compare(
    ptx: &str,
    layer: usize,
    slot: usize,
    seq_len: usize,
    k_f32: &[f32],
    v_f32: &[f32],
    q: &[f32],
) {
    upload_kv(layer, slot, k_f32, v_f32);

    let q_dev = DevBuf::alloc(q.len() * 4);
    let out_dev = DevBuf::alloc(N_HEADS * HEAD_DIM * 4);
    h2d_f32(q_dev.ptr(), q);

    let rc = nsl_cfie_launch_decode_attn(
        q_dev.ptr(),
        out_dev.ptr(),
        layer as i64,
        slot as i64,
        seq_len as i64,
    );
    if rc != 0 {
        panic_with_jit_log("nsl_cfie_launch_decode_attn", rc, ptx);
    }
    let gpu = d2h_f32(out_dev.ptr(), N_HEADS * HEAD_DIM);

    let k_rounded: Vec<f32> = k_f32.iter().map(|&x| f16_round(x)).collect();
    let v_rounded: Vec<f32> = v_f32.iter().map(|&x| f16_round(x)).collect();
    let cpu = cpu_reference(
        q,
        &k_rounded,
        &v_rounded,
        N_HEADS as u32,
        N_KV_HEADS as u32,
        HEAD_DIM as u32,
        seq_len as u32,
    );

    let mut max_abs = 0f32;
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let d = (g - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= 5e-3,
            "decode_attn (layer {}, slot {}, seq {}) out[{}]: gpu {} vs cpu {} (diff {})",
            layer,
            slot,
            seq_len,
            i,
            g,
            c,
            d
        );
    }
    eprintln!(
        "decode_attn parity (layer {}, slot {}, seq {}): max_abs = {:.3e}",
        layer, slot, seq_len, max_abs
    );
}

fn seeded_kv_q(seq_len: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut k = vec![0f32; seq_len * N_KV_HEADS * HEAD_DIM];
    let mut v = vec![0f32; seq_len * N_KV_HEADS * HEAD_DIM];
    let mut q = vec![0f32; N_HEADS * HEAD_DIM];
    fill_seeded(&mut k, seed);
    fill_seeded(&mut v, seed.wrapping_add(1));
    fill_seeded(&mut q, seed.wrapping_add(2));
    (k, v, q)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_attn_layer0_slot0_partial_tile_matches_cpu() {
    if !cuda_available() {
        return;
    }
    let ptx = setup_engine();
    let slot = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot, 0);

    let seq_len = 5;
    let (k, v, q) = seeded_kv_q(seq_len, 0xC0FFEE);
    run_and_compare(&ptx, 0, 0, seq_len, &k, &v, &q);

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_attn_mid_pool_layer1_slot1_ignores_decoy_regions() {
    if !cuda_available() {
        return;
    }
    let ptx = setup_engine();
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);
    let slot = nsl_cfie_kv_slot_acquire();
    assert_eq!(slot, 1);

    // Fill every OTHER (layer, slot) region with full-slot decoy noise:
    // a stride/base-offset bug in the kernel or the engine's kv_base
    // injection would drag decoy values into the output.
    for (l, s, seed) in [(0usize, 0usize, 7u64), (0, 1, 8), (1, 0, 9)] {
        let mut dk = vec![0f32; PER_SLOT * N_KV_HEADS * HEAD_DIM];
        let mut dv = vec![0f32; PER_SLOT * N_KV_HEADS * HEAD_DIM];
        fill_seeded(&mut dk, seed);
        fill_seeded(&mut dv, seed.wrapping_add(100));
        upload_kv(l, s, &dk, &dv);
    }

    let seq_len = 7;
    let (k, v, q) = seeded_kv_q(seq_len, 0xDECAF);
    run_and_compare(&ptx, 1, 1, seq_len, &k, &v, &q);

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_attn_seq_len_zero_writes_zeros() {
    if !cuda_available() {
        return;
    }
    let ptx = setup_engine();
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);

    let mut q = vec![0f32; N_HEADS * HEAD_DIM];
    fill_seeded(&mut q, 0xBEEF);
    let q_dev = DevBuf::alloc(q.len() * 4);
    let out_dev = DevBuf::alloc(N_HEADS * HEAD_DIM * 4);
    h2d_f32(q_dev.ptr(), &q);
    // Prefill the output with a sentinel so untouched bytes are visible.
    let sentinel = vec![7.5f32; N_HEADS * HEAD_DIM];
    h2d_f32(out_dev.ptr(), &sentinel);

    let rc = nsl_cfie_launch_decode_attn(q_dev.ptr(), out_dev.ptr(), 0, 0, 0);
    if rc != 0 {
        panic_with_jit_log("nsl_cfie_launch_decode_attn (seq 0)", rc, &ptx);
    }
    let gpu = d2h_f32(out_dev.ptr(), N_HEADS * HEAD_DIM);
    for (i, g) in gpu.iter().enumerate() {
        assert_eq!(*g, 0.0, "seq_len 0 must write exact zeros, out[{}] = {}", i, g);
    }

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn decode_attn_full_slot_seq_len_equals_per_slot() {
    if !cuda_available() {
        return;
    }
    let ptx = setup_engine();
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);

    let seq_len = PER_SLOT; // 16: the full-slot boundary case
    let (k, v, q) = seeded_kv_q(seq_len, 0xFEED);
    run_and_compare(&ptx, 0, 0, seq_len, &k, &v, &q);

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}
