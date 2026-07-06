//! CFIE Cycle 7: GPU numeric parity for the per-layer KV-quant
//! decode-attention kernels (kind 5) through the REAL engine path:
//! one registration PER LAYER -> kv_slots_init -> kv_pool_alloc
//! (total_pool_bytes of the MIXED layout) -> finalize ->
//! nsl_cfie_launch_quant_attn with the f32-bit-punned scale params.
//!
//! First hardware execution of the kernel family.  2-layer mixed
//! config: layer 0 Int8/Int8 (register dequant with runtime scales),
//! layer 1 Fp16/Fp16 (scale params declared but unused).  K/V halves
//! are uploaded at the cfie_kv_quant_ptx::pool_layout byte offsets;
//! Int8 halves are quantized CPU-side with quantize_symmetric_i8 and
//! the CPU reference dequantizes the SAME i8 bytes, so parity is
//! limited only by kernel arithmetic (5e-3 max-abs budget).
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_kv_quant_gpu_parity \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::cfie_kv_quant::KvPrecision;
use nsl_codegen::cfie_kv_quant_ptx::{
    cpu_reference_layer, emit_all, pool_layout, quantize_symmetric_i8, total_pool_bytes,
    CpuKvHalf, QuantDecodeAttentionConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_kv_pool_alloc,
    nsl_cfie_kv_pool_base, nsl_cfie_launch_quant_attn, nsl_cfie_register_kernel,
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
// Deterministic PRNG (LCG from pca_tier_a_forward_correctness.rs with
// the corrected zero-mean mapping used across the CFIE parity suites).
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

fn h2d_i8(dst: i64, src: &[i8]) {
    nsl_test_cuda_h2d(dst, src.as_ptr() as i64, src.len() as i64);
}

fn d2h_f32(src: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, src, (len * 4) as i64);
    out
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
// Fixture: 2-layer mixed precision (L0 Int8/Int8, L1 Fp16/Fp16).
// ---------------------------------------------------------------------------

const N_LAYERS: usize = 2;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const PER_SLOT: usize = 16;
const MAX_SLOTS: usize = 2;
const TOKEN_STRIDE_ELEMS: usize = N_KV_HEADS * HEAD_DIM; // per-token record

fn quant_cfg() -> QuantDecodeAttentionConfig {
    QuantDecodeAttentionConfig {
        n_layers: N_LAYERS as u32,
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        per_slot_max_tokens: PER_SLOT as u32,
        max_slots: MAX_SLOTS as u32,
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
        layer_precisions: vec![
            (KvPrecision::Int8, KvPrecision::Int8),
            (KvPrecision::Fp16, KvPrecision::Fp16),
        ],
    }
}

/// Engine bring-up: register ONE kernel per layer under kind 5, init
/// slots, allocate the mixed-layout pool, finalize.  Returns the
/// per-layer PTX strings (for JIT-log diagnostics).
fn setup_engine(cfg: &QuantDecodeAttentionConfig) -> Vec<String> {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let mut ptxs = Vec::new();
    for (layer, (ptx, meta)) in emit_all(cfg).into_iter().enumerate() {
        assert_eq!(meta.layer_idx as usize, layer);
        let name = meta.kernel_name.as_str();
        let rc = nsl_cfie_register_kernel(
            5, // kind 5 = quant_attn: one registration per layer
            layer as i64,
            ptx.as_ptr() as i64,
            ptx.len() as i64,
            name.as_ptr() as i64,
            name.len() as i64,
            N_HEADS as i64, // grid_dim_is_n_heads
            meta.block_dim as i64,
            0,
        );
        assert_eq!(rc, 0, "register_kernel(kind 5, layer {}) refused", layer);
        ptxs.push(ptx);
    }
    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, PER_SLOT as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(total_pool_bytes(cfg) as i64), 0);
    let n = nsl_cfie_engine_finalize();
    if n != N_LAYERS as i64 {
        panic_with_jit_log("nsl_cfie_engine_finalize (quant_attn)", n, &ptxs[0]);
    }
    ptxs
}

/// Byte offset of `slot`'s first token inside a half whose element
/// width is `elem_bytes`.
fn slot_byte_off(slot: usize, elem_bytes: usize) -> usize {
    slot * PER_SLOT * TOKEN_STRIDE_ELEMS * elem_bytes
}

/// Upload an Int8 half's rows for (offset, slot).
fn upload_i8_half(half_offset_bytes: u64, slot: usize, data: &[i8]) {
    let base = nsl_cfie_kv_pool_base();
    assert!(base != 0, "pool base accessor returned 0 after pool_alloc");
    h2d_i8(base + half_offset_bytes as i64 + slot_byte_off(slot, 1) as i64, data);
}

/// Upload an Fp16 half's rows (f32 -> f16 bits) for (offset, slot).
fn upload_f16_half(half_offset_bytes: u64, slot: usize, data_f32: &[f32]) {
    let base = nsl_cfie_kv_pool_base();
    assert!(base != 0, "pool base accessor returned 0 after pool_alloc");
    let bits: Vec<u16> = data_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    h2d_u16(base + half_offset_bytes as i64 + slot_byte_off(slot, 2) as i64, &bits);
}

/// Launch layer `layer` through the engine FFI with bit-punned scales
/// and return the output.
fn launch_layer(
    ptx: &str,
    layer: usize,
    q: &[f32],
    slot: usize,
    seq_len: usize,
    k_scale: f32,
    v_scale: f32,
) -> Vec<f32> {
    let q_dev = DevBuf::alloc(q.len() * 4);
    let out_dev = DevBuf::alloc(N_HEADS * HEAD_DIM * 4);
    h2d_f32(q_dev.ptr(), q);

    let rc = nsl_cfie_launch_quant_attn(
        layer as i64,
        q_dev.ptr(),
        out_dev.ptr(),
        slot as i64,
        seq_len as i64,
        k_scale.to_bits() as i64, // f32::to_bits in the i64 low 32 (ABI)
        v_scale.to_bits() as i64,
    );
    if rc != 0 {
        panic_with_jit_log("nsl_cfie_launch_quant_attn", rc, ptx);
    }
    d2h_f32(out_dev.ptr(), N_HEADS * HEAD_DIM)
}

fn assert_close(gpu: &[f32], cpu: &[f32], label: &str) {
    let mut max_abs = 0f32;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let d = (g - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= 5e-3,
            "quant_attn {} out[{}]: gpu {} vs cpu {} (diff {})",
            label,
            i,
            g,
            c,
            d
        );
    }
    eprintln!("quant_attn parity ({}): max_abs = {:.3e}", label, max_abs);
}

fn seeded_kv_q(seq_len: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut k = vec![0f32; seq_len * TOKEN_STRIDE_ELEMS];
    let mut v = vec![0f32; seq_len * TOKEN_STRIDE_ELEMS];
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
fn quant_attn_layer0_int8_register_dequant_matches_cpu() {
    if !cuda_available() {
        return;
    }
    let cfg = quant_cfg();
    let layout = pool_layout(&cfg);
    let ptxs = setup_engine(&cfg);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);

    let seq_len = 7;
    let (k, v, q) = seeded_kv_q(seq_len, 0x1888);
    let (qk, k_scale) = quantize_symmetric_i8(&k);
    let (qv, v_scale) = quantize_symmetric_i8(&v);
    upload_i8_half(layout[0].k_offset_bytes, 0, &qk);
    upload_i8_half(layout[0].v_offset_bytes, 0, &qv);

    let gpu = launch_layer(&ptxs[0], 0, &q, 0, seq_len, k_scale, v_scale);
    let cpu = cpu_reference_layer(
        &q,
        CpuKvHalf::Int8 { data: &qk, scale: k_scale },
        CpuKvHalf::Int8 { data: &qv, scale: v_scale },
        N_HEADS as u32,
        N_KV_HEADS as u32,
        HEAD_DIM as u32,
        seq_len as u32,
    );
    assert_close(&gpu, &cpu, "layer 0 int8/int8, slot 0, seq 7");

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn quant_attn_layer1_fp16_mid_pool_ignores_decoy_regions() {
    if !cuda_available() {
        return;
    }
    let cfg = quant_cfg();
    let layout = pool_layout(&cfg);
    let ptxs = setup_engine(&cfg);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 1);

    // Decoy noise in the OTHER regions: layer 0's int8 halves (both
    // slots) and layer 1's slot 0 — a baked-offset bug would drag decoy
    // bytes into the output.
    let mut decoy_f = vec![0f32; PER_SLOT * TOKEN_STRIDE_ELEMS];
    for (s, seed) in [(0usize, 51u64), (1, 52)] {
        fill_seeded(&mut decoy_f, seed);
        let (dq, _) = quantize_symmetric_i8(&decoy_f);
        upload_i8_half(layout[0].k_offset_bytes, s, &dq);
        fill_seeded(&mut decoy_f, seed + 100);
        let (dq, _) = quantize_symmetric_i8(&decoy_f);
        upload_i8_half(layout[0].v_offset_bytes, s, &dq);
    }
    fill_seeded(&mut decoy_f, 53);
    upload_f16_half(layout[1].k_offset_bytes, 0, &decoy_f);
    fill_seeded(&mut decoy_f, 54);
    upload_f16_half(layout[1].v_offset_bytes, 0, &decoy_f);

    // Full-slot boundary case on the mid-pool (layer 1, slot 1) target.
    let seq_len = PER_SLOT;
    let (k, v, q) = seeded_kv_q(seq_len, 0x2999);
    upload_f16_half(layout[1].k_offset_bytes, 1, &k);
    upload_f16_half(layout[1].v_offset_bytes, 1, &v);

    // Fp16 layer: scale params declared but unused — pass sentinels
    // that would corrupt the output if the kernel consumed them.
    let gpu = launch_layer(&ptxs[1], 1, &q, 1, seq_len, 123.5, -777.25);
    let k_rounded: Vec<f32> = k.iter().map(|&x| f16_round(x)).collect();
    let v_rounded: Vec<f32> = v.iter().map(|&x| f16_round(x)).collect();
    let cpu = cpu_reference_layer(
        &q,
        CpuKvHalf::Fp16(&k_rounded),
        CpuKvHalf::Fp16(&v_rounded),
        N_HEADS as u32,
        N_KV_HEADS as u32,
        HEAD_DIM as u32,
        seq_len as u32,
    );
    assert_close(&gpu, &cpu, "layer 1 fp16/fp16, slot 1, seq 16");

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn quant_attn_both_layers_back_to_back_use_their_own_registration() {
    if !cuda_available() {
        return;
    }
    // One engine bring-up, both layers launched in sequence: proves the
    // (kind 5, layer_idx) registry dispatch picks each layer's
    // specialized kernel (their load paths and pool offsets differ).
    let cfg = quant_cfg();
    let layout = pool_layout(&cfg);
    let ptxs = setup_engine(&cfg);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);

    let seq_len = 11;
    let (k0, v0, q0) = seeded_kv_q(seq_len, 0x3AAA);
    let (qk0, k_scale0) = quantize_symmetric_i8(&k0);
    let (qv0, v_scale0) = quantize_symmetric_i8(&v0);
    upload_i8_half(layout[0].k_offset_bytes, 0, &qk0);
    upload_i8_half(layout[0].v_offset_bytes, 0, &qv0);

    let (k1, v1, q1) = seeded_kv_q(seq_len, 0x4BBB);
    upload_f16_half(layout[1].k_offset_bytes, 0, &k1);
    upload_f16_half(layout[1].v_offset_bytes, 0, &v1);

    let gpu0 = launch_layer(&ptxs[0], 0, &q0, 0, seq_len, k_scale0, v_scale0);
    let gpu1 = launch_layer(&ptxs[1], 1, &q1, 0, seq_len, 1.0, 1.0);

    let cpu0 = cpu_reference_layer(
        &q0,
        CpuKvHalf::Int8 { data: &qk0, scale: k_scale0 },
        CpuKvHalf::Int8 { data: &qv0, scale: v_scale0 },
        N_HEADS as u32,
        N_KV_HEADS as u32,
        HEAD_DIM as u32,
        seq_len as u32,
    );
    let k1_rounded: Vec<f32> = k1.iter().map(|&x| f16_round(x)).collect();
    let v1_rounded: Vec<f32> = v1.iter().map(|&x| f16_round(x)).collect();
    let cpu1 = cpu_reference_layer(
        &q1,
        CpuKvHalf::Fp16(&k1_rounded),
        CpuKvHalf::Fp16(&v1_rounded),
        N_HEADS as u32,
        N_KV_HEADS as u32,
        HEAD_DIM as u32,
        seq_len as u32,
    );
    assert_close(&gpu0, &cpu0, "back-to-back layer 0 int8");
    assert_close(&gpu1, &cpu1, "back-to-back layer 1 fp16");

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}
