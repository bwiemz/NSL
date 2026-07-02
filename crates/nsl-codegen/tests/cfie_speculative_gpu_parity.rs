//! CFIE Cycle 7: GPU numeric parity for the speculative kernels —
//! tree-mask verification attention (kind 3) and the rejection-sampling
//! epilogue (kind 4) — through the REAL engine path: register ->
//! kv_slots_init -> kv_pool_alloc -> finalize -> typed launch FFIs.
//!
//! First hardware execution of the kernel family.
//!
//!   * verify: per the host contract, the committed prefix rows AND the
//!     num_nodes draft rows are uploaded to positions
//!     [0, seq_len + num_nodes) of (layer, slot) before launch; the
//!     tree(2,2) ancestor mask comes from the real
//!     cfie_speculative::build_tree_mask helper.  f32 compare at 5e-3
//!     max-abs (f16 KV + ex2.approx softmax, PCA/CSHA hd<=32 budget).
//!   * reject: (accepted, correction_token) asserted EXACTLY equal to
//!     cpu_reference_reject — the kernel PRNG is bit-for-bit
//!     xorshift64* including the golden-gamma zero-seed guard.
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_speculative_gpu_parity \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::cfie_speculative::build_tree_mask;
use nsl_codegen::cfie_speculative_ptx::{
    cpu_reference_reject, cpu_reference_verify, emit_rejection_kernel, emit_verify_attention,
    mask_bits_from_tree, RejectionConfig, VerifyAttentionConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_kv_pool_alloc,
    nsl_cfie_kv_pool_base, nsl_cfie_launch_spec_reject, nsl_cfie_launch_spec_verify,
    nsl_cfie_register_kernel,
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

fn h2d_u32(dst: i64, src: &[u32]) {
    nsl_test_cuda_h2d(dst, src.as_ptr() as i64, (src.len() * 4) as i64);
}

fn d2h_f32(src: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, src, (len * 4) as i64);
    out
}

fn d2h_i32_one(src: i64) -> i32 {
    let mut out = [0i32; 1];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, src, 4);
    out[0]
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
// Verify fixture: KV dims shared with the decode-attn suite; the
// tree(2, 2) mask (7 BFS nodes) comes from the real helper.
// ---------------------------------------------------------------------------

const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const PER_SLOT: usize = 16;
const MAX_SLOTS: usize = 2;
const N_LAYERS: usize = 2; // pool depth; layer_idx is a launch param

// Byte strides of the uniform [n_layers][2][max_tokens][nkv][hd] f16 pool.
const TOKEN_STRIDE_BYTES: usize = N_KV_HEADS * HEAD_DIM * 2;
const KV_HALF_BYTES: usize = MAX_SLOTS * PER_SLOT * N_KV_HEADS * HEAD_DIM * 2;
const LAYER_BYTES: usize = 2 * KV_HALF_BYTES;
const POOL_BYTES: usize = N_LAYERS * LAYER_BYTES;

fn verify_cfg() -> VerifyAttentionConfig {
    // 3 levels x branching 2 = 1 + 2 + 4 = 7 BFS nodes — the paper's
    // tree(2,2) shape (root, two drafts, four grandchild drafts).
    let tree = build_tree_mask(3, 2);
    VerifyAttentionConfig {
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        per_slot_max_tokens: PER_SLOT as u32,
        max_slots: MAX_SLOTS as u32,
        num_nodes: tree.num_nodes,
        mask_bits: mask_bits_from_tree(&tree),
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
    }
}

/// Engine bring-up for the verify kernel: register kind 3, init slots,
/// allocate + zero the pool, finalize.  Returns the PTX.
fn setup_verify_engine(cfg: &VerifyAttentionConfig) -> String {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let (ptx, meta) = emit_verify_attention(cfg);
    let name = meta.kernel_name.as_str();
    let rc = nsl_cfie_register_kernel(
        3, // kind 3 = spec_verify
        0,
        ptx.as_ptr() as i64,
        ptx.len() as i64,
        name.as_ptr() as i64,
        name.len() as i64,
        N_HEADS as i64, // grid_dim_is_n_heads
        meta.block_dim as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 3) refused");
    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, PER_SLOT as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(POOL_BYTES as i64), 0);
    let n = nsl_cfie_engine_finalize();
    if n != 1 {
        panic_with_jit_log("nsl_cfie_engine_finalize (spec_verify)", n, &ptx);
    }
    ptx
}

/// Upload one slot's K/V token rows (f32 -> f16 bits) at the uniform
/// (layer, slot) byte offsets — prefix AND draft rows in one block.
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

/// Launch verify through the engine and compare vs cpu_reference_verify
/// (fed f16-rounded K/V) at 5e-3.
fn run_verify_and_compare(
    cfg: &VerifyAttentionConfig,
    ptx: &str,
    layer: usize,
    slot: usize,
    seq_len: usize,
    seed: u64,
) {
    let nn = cfg.num_nodes as usize;
    let rows = seq_len + nn;
    let mut k = vec![0f32; rows * N_KV_HEADS * HEAD_DIM];
    let mut v = vec![0f32; rows * N_KV_HEADS * HEAD_DIM];
    let mut q = vec![0f32; nn * N_HEADS * HEAD_DIM];
    fill_seeded(&mut k, seed);
    fill_seeded(&mut v, seed.wrapping_add(1));
    fill_seeded(&mut q, seed.wrapping_add(2));

    // Host contract: prefix rows [0, seq_len) plus the nn draft rows at
    // [seq_len, seq_len + nn) of (layer, slot), uploaded BEFORE launch.
    upload_kv(layer, slot, &k, &v);

    let q_dev = DevBuf::alloc(q.len() * 4);
    let out_dev = DevBuf::alloc(q.len() * 4);
    h2d_f32(q_dev.ptr(), &q);

    let rc = nsl_cfie_launch_spec_verify(
        q_dev.ptr(),
        out_dev.ptr(),
        layer as i64,
        slot as i64,
        seq_len as i64,
    );
    if rc != 0 {
        panic_with_jit_log("nsl_cfie_launch_spec_verify", rc, ptx);
    }
    let gpu = d2h_f32(out_dev.ptr(), q.len());

    let k_rounded: Vec<f32> = k.iter().map(|&x| f16_round(x)).collect();
    let v_rounded: Vec<f32> = v.iter().map(|&x| f16_round(x)).collect();
    let cpu = cpu_reference_verify(cfg, &q, &k_rounded, &v_rounded, seq_len as u32);

    let mut max_abs = 0f32;
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let d = (g - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= 5e-3,
            "spec_verify (layer {}, slot {}, seq {}) out[{}]: gpu {} vs cpu {} (diff {})",
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
        "spec_verify parity (layer {}, slot {}, seq {}, nodes {}): max_abs = {:.3e}",
        layer, slot, seq_len, nn, max_abs
    );
}

// ---------------------------------------------------------------------------
// Reject fixture
// ---------------------------------------------------------------------------

const K_TOKENS: usize = 4;
const R_VOCAB: usize = 64;

fn reject_cfg() -> RejectionConfig {
    RejectionConfig {
        k_tokens: K_TOKENS as u32,
        vocab_size: R_VOCAB as u32,
        sm_version: 80,
    }
}

/// Engine bring-up for the reject kernel (no pool needed — the kernel
/// takes explicit device buffers).  Returns the PTX.
fn setup_reject_engine(cfg: &RejectionConfig) -> String {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let (ptx, meta) = emit_rejection_kernel(cfg);
    let name = meta.kernel_name.as_str();
    let rc = nsl_cfie_register_kernel(
        4, // kind 4 = spec_reject
        0,
        ptx.as_ptr() as i64,
        ptx.len() as i64,
        name.as_ptr() as i64,
        name.len() as i64,
        1, // grid = 1, serial thread-0 walk
        meta.block_dim as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 4) refused");
    let n = nsl_cfie_engine_finalize();
    if n != 1 {
        panic_with_jit_log("nsl_cfie_engine_finalize (spec_reject)", n, &ptx);
    }
    ptx
}

/// Launch the reject kernel and assert EXACT equality with
/// cpu_reference_reject.
fn run_reject_and_compare(
    cfg: &RejectionConfig,
    ptx: &str,
    target_probs: &[f32],
    draft_probs: &[f32],
    draft_tokens: &[u32],
    seed: u64,
    label: &str,
) -> (i32, u32) {
    let tp_dev = DevBuf::alloc(target_probs.len() * 4);
    let dp_dev = DevBuf::alloc(draft_probs.len() * 4);
    let dt_dev = DevBuf::alloc(draft_tokens.len() * 4);
    let acc_dev = DevBuf::alloc(4);
    let corr_dev = DevBuf::alloc(4);
    h2d_f32(tp_dev.ptr(), target_probs);
    h2d_f32(dp_dev.ptr(), draft_probs);
    h2d_u32(dt_dev.ptr(), draft_tokens);

    let rc = nsl_cfie_launch_spec_reject(
        tp_dev.ptr(),
        dp_dev.ptr(),
        dt_dev.ptr(),
        seed as i64,
        acc_dev.ptr(),
        corr_dev.ptr(),
    );
    if rc != 0 {
        panic_with_jit_log("nsl_cfie_launch_spec_reject", rc, ptx);
    }
    let gpu = (d2h_i32_one(acc_dev.ptr()), d2h_u32_one(corr_dev.ptr()));
    let cpu = cpu_reference_reject(cfg, target_probs, draft_probs, draft_tokens, seed);
    assert_eq!(
        gpu, cpu,
        "spec_reject ({label}, seed {:#x}): gpu {:?} vs cpu {:?} — PRNG must be bit-for-bit",
        seed, gpu, cpu
    );
    eprintln!(
        "spec_reject ({label}, seed {:#x}): accepted {}, correction {:#x}",
        seed, gpu.0, gpu.1
    );
    gpu
}

/// Seeded row-stochastic target matrix + drafted tokens/probs.
fn seeded_reject_fixture(seed: u64) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let mut raw = vec![0f32; K_TOKENS * R_VOCAB];
    fill_seeded(&mut raw, seed);
    let mut target = vec![0f32; K_TOKENS * R_VOCAB];
    for j in 0..K_TOKENS {
        let row = &raw[j * R_VOCAB..(j + 1) * R_VOCAB];
        let shifted: Vec<f32> = row.iter().map(|&x| x + 0.5).collect(); // [0, 1)
        let sum: f32 = shifted.iter().sum();
        for (t, s) in target[j * R_VOCAB..(j + 1) * R_VOCAB].iter_mut().zip(&shifted) {
            *t = s / sum;
        }
    }
    let mut dp_raw = vec![0f32; K_TOKENS];
    fill_seeded(&mut dp_raw, seed.wrapping_add(1));
    // d in [0.5/vocab, 1/vocab): p_target/d spans ~[0, 4) -> mixed
    // accept/reject outcomes across seeds.
    let draft_probs: Vec<f32> = dp_raw
        .iter()
        .map(|&x| (0.75 + 0.5 * x) / R_VOCAB as f32)
        .collect();
    let mut tok_raw = vec![0f32; K_TOKENS];
    fill_seeded(&mut tok_raw, seed.wrapping_add(2));
    let draft_tokens: Vec<u32> = tok_raw
        .iter()
        .map(|&x| (((x + 0.5) * R_VOCAB as f32) as u32).min(R_VOCAB as u32 - 1))
        .collect();
    (target, draft_probs, draft_tokens)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA GPU"]
fn spec_verify_tree_mask_layer0_slot0_matches_cpu() {
    if !cuda_available() {
        return;
    }
    let cfg = verify_cfg();
    let ptx = setup_verify_engine(&cfg);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);

    run_verify_and_compare(&cfg, &ptx, 0, 0, 5, 0x7EE7);

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn spec_verify_mid_pool_layer1_slot1_ignores_decoy_regions() {
    if !cuda_available() {
        return;
    }
    let cfg = verify_cfg();
    let ptx = setup_verify_engine(&cfg);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 0);
    assert_eq!(nsl_cfie_kv_slot_acquire(), 1);

    // Decoy noise in every OTHER (layer, slot) region: a stride or
    // base-offset bug would drag decoy values into the output.
    for (l, s, seed) in [(0usize, 0usize, 31u64), (0, 1, 32), (1, 0, 33)] {
        let mut dk = vec![0f32; PER_SLOT * N_KV_HEADS * HEAD_DIM];
        let mut dv = vec![0f32; PER_SLOT * N_KV_HEADS * HEAD_DIM];
        fill_seeded(&mut dk, seed);
        fill_seeded(&mut dv, seed.wrapping_add(100));
        upload_kv(l, s, &dk, &dv);
    }

    // seq 9 + 7 nodes = 16 rows: exactly the full slot.
    let seq = PER_SLOT - cfg.num_nodes as usize;
    run_verify_and_compare(&cfg, &ptx, 1, 1, seq, 0xACE5);

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn spec_reject_profiles_match_cpu_exactly() {
    if !cuda_available() {
        return;
    }
    let cfg = reject_cfg();
    let ptx = setup_reject_engine(&cfg);

    // (a) all-accept: full target mass on each drafted token ->
    // p_target/d >= 1 > r for every draw.
    let draft_tokens: Vec<u32> = vec![3, 17, 42, 63];
    let mut target = vec![0f32; K_TOKENS * R_VOCAB];
    for (j, &t) in draft_tokens.iter().enumerate() {
        target[j * R_VOCAB + t as usize] = 1.0;
    }
    let draft_probs = vec![0.5f32; K_TOKENS];
    let (acc, corr) =
        run_reject_and_compare(&cfg, &ptx, &target, &draft_probs, &draft_tokens, 0xAA, "all-accept");
    assert_eq!(acc, K_TOKENS as i32, "every draft must accept");
    assert_eq!(corr, u32::MAX, "all-accept publishes the 0xFFFFFFFF sentinel");

    // (b) first-reject: row 0 has ZERO target mass on its drafted token
    // -> ratio 0 rejects immediately; the correction samples row 0's
    // residual (= the target row itself, clamped subtraction of d at a
    // zero entry).
    let (mut target_fr, draft_probs_fr, draft_tokens_fr) = seeded_reject_fixture(0xB0);
    let tok0 = draft_tokens_fr[0] as usize;
    // Move row 0's tok0 mass onto another token, keeping the row a
    // distribution.
    let moved = target_fr[tok0];
    target_fr[tok0] = 0.0;
    target_fr[(tok0 + 1) % R_VOCAB] += moved;
    let (acc, corr) = run_reject_and_compare(
        &cfg,
        &ptx,
        &target_fr,
        &draft_probs_fr,
        &draft_tokens_fr,
        0xB1,
        "first-reject",
    );
    assert_eq!(acc, 0, "zero target mass on the draft must reject at j = 0");
    assert!(corr != u32::MAX && (corr as usize) < R_VOCAB);

    // (c) p_draft = 0 row: the division guard rejects row 1 regardless
    // of the draw.
    let (target_z, mut draft_probs_z, draft_tokens_z) = seeded_reject_fixture(0xC0);
    // Row 0 always accepts (target mass 1.0 on its token would break the
    // distribution; instead make ratio >= 1 by shrinking d... simplest:
    // give row 0 full mass on its drafted token).
    let mut target_z = target_z;
    let tok0z = draft_tokens_z[0] as usize;
    for v in target_z[..R_VOCAB].iter_mut() {
        *v = 0.0;
    }
    target_z[tok0z] = 1.0;
    draft_probs_z[0] = 0.5;
    draft_probs_z[1] = 0.0;
    let (acc, _corr) = run_reject_and_compare(
        &cfg,
        &ptx,
        &target_z,
        &draft_probs_z,
        &draft_tokens_z,
        0xC1,
        "p-draft-zero",
    );
    assert_eq!(acc, 1, "p_draft 0 at row 1 must reject after row 0 accepts");

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn spec_reject_seeded_profiles_and_zero_seed_guard_match_cpu_exactly() {
    if !cuda_available() {
        return;
    }
    let cfg = reject_cfg();
    let ptx = setup_reject_engine(&cfg);

    let mut outcomes = std::collections::BTreeSet::new();
    for (i, seed) in [0u64, 1, 7, 42, 0xDEAD_BEEF, u64::MAX].into_iter().enumerate() {
        // seed 0 exercises the golden-gamma zero-seed guard on hardware.
        let (target, draft_probs, draft_tokens) = seeded_reject_fixture(0xD0 + i as u64);
        let (acc, corr) = run_reject_and_compare(
            &cfg,
            &ptx,
            &target,
            &draft_probs,
            &draft_tokens,
            seed,
            "seeded",
        );
        assert!(acc >= 0 && acc <= K_TOKENS as i32);
        if acc < K_TOKENS as i32 {
            assert!((corr as usize) < R_VOCAB);
        } else {
            assert_eq!(corr, u32::MAX);
        }
        outcomes.insert(acc);
    }
    // Sanity: the fixture family actually produces mixed accept counts.
    assert!(
        outcomes.len() > 1,
        "6 seeded profiles all accepted {:?} tokens — fixture too degenerate",
        outcomes
    );

    assert_eq!(nsl_cfie_engine_destroy(), 0);
}
