#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! The bf16 weight-cast cache must be INVISIBLE in results and visible only
//! in launch counts.
//!
//! The cache's one honest failure mode is serving a stale bf16 image of a
//! parameter that has moved — silently wrong forwards forever after. Every
//! check here therefore ends in a BIT-comparison against the fresh-scratch
//! path (RNE casting is deterministic, so cached-vs-fresh must be
//! byte-identical, not merely close), and the staleness checks drive theta
//! through the REAL mutation funnels: the fused AdamW extern, the copy_data
//! extern, and the allocator free hook.
//!
//! Registration goes through the production path too — a zero-lr
//! `nsl_fase_fused_adamw_step` call, which registers and stales without
//! moving a single theta bit (AdamW's update and its decoupled decay are
//! both scaled by lr). A parallel test-only registration hook would let the
//! fase_step wiring rot while this gate stayed green.
//!
//! Process isolation: the math mode resolves once per process, so the
//! parent test re-runs this binary as a child with `NSL_MATMUL_BF16=1` (the
//! `matmul_bf16_mode.rs` pattern).

use nsl_runtime::fase_step::nsl_fase_fused_adamw_step;
use nsl_runtime::fused_linear_ce::{nsl_fused_linear_ce_backward_gemm, nsl_fused_linear_ce_forward_gemm};
use nsl_runtime::tensor::{
    nsl_tensor_copy_data, nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device,
    test_build_tensor_2d_f32, test_read_tensor_f32,
};
use nsl_runtime::{test_bf16_cast_cache_reset, test_bf16_cast_cache_stats, test_lce_logits_probe};

/// 1024^3 puts mnk/(a+b) exactly at the 512 arithmetic-intensity gate, so
/// the bf16-storage path (and with it the cache) engages while each GEMM
/// stays ~2 GFLOP — this gate may run beside a multi-day training tenant.
const N: usize = 1024;

const PROBE: &str = "NSL_BF16_CACHE_PROBE";

fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn gpu_2d(rows: usize, cols: usize, data: &[f32]) -> i64 {
    let t = test_build_tensor_2d_f32(rows, cols, data);
    let g = nsl_tensor_to_device(t, 1);
    if g != t {
        nsl_tensor_free(t);
    }
    g
}

fn matmul_read(a: i64, b: i64) -> Vec<f64> {
    let c = nsl_tensor_matmul(a, b, 0);
    // The download keeps the device's f32 (C5 step 2a); widen for the f64 checks.
    let c_cpu = nsl_tensor_to_device(c, 0);
    let out: Vec<f64> = test_read_tensor_f32(c_cpu).into_iter().map(f64::from).collect();
    if c_cpu != c {
        nsl_tensor_free(c_cpu);
    }
    nsl_tensor_free(c);
    out
}

/// Register `w` with the cast cache through the production funnel: a fused
/// AdamW step at lr=0 (theta bits provably unchanged — the update term and
/// the decoupled decay are both multiplied by lr).
fn register_via_zero_lr_step(w: i64, m: i64, v: i64, mp: i64) {
    nsl_fase_fused_adamw_step(
        w, m, v, mp, /*lr*/ 0.0, /*beta1*/ 0.9, /*omb1*/ 0.1, /*beta2*/ 0.999,
        /*omb2*/ 0.001, /*eps*/ 1e-8, /*wd*/ 0.0, /*bc1_inv*/ 1.0, /*bc2_inv*/ 1.0,
    );
}

fn zeros_gpu() -> i64 {
    gpu_2d(N, N, &vec![0.0f32; N * N])
}

#[test]
#[ignore = "spawned by cast_cache_gates_gpu"]
fn zz_probe_child() {
    if std::env::var(PROBE).is_err() {
        return; // parent process: the launcher test below drives this.
    }

    let a_data = deterministic(N * N, 211);
    let w_data = deterministic(N * N, 223);
    let a = gpu_2d(N, N, &a_data);
    let w = gpu_2d(N, N, &w_data);
    let (m, v, mp) = (zeros_gpu(), zeros_gpu(), zeros_gpu());

    // -- check 1: unregistered operands never touch the cache ------------
    test_bf16_cast_cache_reset();
    let c_fresh = matmul_read(a, w);
    let c_fresh2 = matmul_read(a, w);
    let (h, r, e) = test_bf16_cast_cache_stats();
    assert_eq!(
        (h, r, e),
        (0, 0, 0),
        "no parameter was registered, yet the cache moved: hits={h} recasts={r} evictions={e}"
    );
    assert_eq!(c_fresh, c_fresh2, "fresh-scratch RNE casting must be deterministic");

    // -- check 2: a registered param casts once, hits thereafter, and the
    //    cached bits ARE the fresh bits --------------------------------
    register_via_zero_lr_step(w, m, v, mp);
    let c_cached = matmul_read(a, w);
    let c_cached2 = matmul_read(a, w);
    let (h, r, _e) = test_bf16_cast_cache_stats();
    assert_eq!(r, 1, "one registered operand should cast exactly once, got {r} recasts");
    assert!(h >= 1, "the second GEMM should have hit the cached image, hits={h}");
    assert_eq!(
        c_fresh, c_cached,
        "cached weight image must be BIT-identical to the fresh-scratch cast"
    );
    assert_eq!(c_cached, c_cached2, "hits must serve identical bits");

    // -- check 3: the optimizer moving theta stales the image ------------
    // A real step (lr>0) with a nonzero accumulated gradient: theta moves,
    // and the very next GEMM must see the moved theta, not the image.
    let g_data = deterministic(N * N, 227);
    let mp_real = gpu_2d(N, N, &g_data);
    nsl_fase_fused_adamw_step(
        w, m, v, mp_real, /*lr*/ 0.01, 0.9, 0.1, 0.999, 0.001, 1e-8, 0.0, 1.0, 1.0,
    );
    let c_after_step = matmul_read(a, w);
    assert_ne!(
        c_after_step, c_cached,
        "theta moved by a real optimizer step but the GEMM still returned \
         the pre-step product — the cache served a STALE image"
    );
    // ...and what it returned is exactly the fresh cast of the moved theta.
    test_bf16_cast_cache_reset();
    let c_after_step_fresh = matmul_read(a, w);
    assert_eq!(
        c_after_step, c_after_step_fresh,
        "post-step cached product differs from the fresh-scratch product of \
         the same theta — the recast happened but from wrong bytes"
    );

    // -- check 4: copy_data into a registered param evicts ---------------
    register_via_zero_lr_step(w, m, v, mp);
    let _ = matmul_read(a, w); // warm the image
    let w2_data = deterministic(N * N, 229);
    let w2 = gpu_2d(N, N, &w2_data);
    nsl_tensor_copy_data(w, w2);
    let (_h, _r, e_before) = test_bf16_cast_cache_stats();
    assert!(
        e_before >= 1,
        "copy_data into a registered parameter must evict its image, evictions={e_before}"
    );
    let c_overwritten = matmul_read(a, w);
    test_bf16_cast_cache_reset();
    let c_overwritten_fresh = matmul_read(a, w);
    assert_eq!(
        c_overwritten, c_overwritten_fresh,
        "product after copy_data does not match the fresh cast of the new bytes"
    );
    assert_ne!(
        c_overwritten, c_fresh,
        "sanity: the overwrite should have changed the product (w2 != w)"
    );

    // -- check 5: freeing a registered param evicts, so a recycled address
    //    cannot inherit its image ---------------------------------------
    register_via_zero_lr_step(w, m, v, mp);
    let _ = matmul_read(a, w); // warm the image
    let (_h, _r, e0) = test_bf16_cast_cache_stats();
    nsl_tensor_free(w); // -> free_managed -> evict at its top
    let (_h2, _r2, e1) = test_bf16_cast_cache_stats();
    assert!(
        e1 > e0,
        "freeing a registered parameter must evict via the free hook \
         (evictions {e0} -> {e1})"
    );
    // The allocator will likely hand the same block back for a same-size
    // request. Whether or not it does, an UNREGISTERED tensor there must
    // compute from its own bytes.
    let w3_data = deterministic(N * N, 233);
    let w3 = gpu_2d(N, N, &w3_data);
    let c_recycled = matmul_read(a, w3);
    test_bf16_cast_cache_reset();
    let c_recycled_fresh = matmul_read(a, w3);
    assert_eq!(
        c_recycled, c_recycled_fresh,
        "recycled-address tensor did not compute from its own bytes"
    );

    sub_view_checks();
    head_route_checks();

    println!("CACHE-PROBE-OK");
}

/// The tensor's device data pointer (`NslTensor` is `repr(C)` and its `data`
/// follows the `u32` magic; the fields are crate-private).
fn data_ptr(t: i64) -> i64 {
    #[repr(C)]
    struct Head {
        _magic: u32,
        data: *mut std::ffi::c_void,
    }
    unsafe { (*(t as *const Head)).data as i64 }
}

/// One fused LM-head forward through the production extern (raw device
/// pointers, i64 targets), returning (loss, lse) per row as f32 bits.
fn lm_head_forward(x: i64, w: i64, targets: i64, rows: usize, v: usize, h: usize) -> Vec<u32> {
    let loss = gpu_2d(rows, 1, &vec![0.0f32; rows]);
    let lse = gpu_2d(rows, 1, &vec![0.0f32; rows]);
    let rc = nsl_fused_linear_ce_forward_gemm(
        data_ptr(x),
        data_ptr(w),
        0,
        data_ptr(targets),
        data_ptr(loss),
        data_ptr(lse),
        1,
        rows as i64,
        v as i64,
        h as i64,
        0,
    );
    assert_eq!(rc, 0, "fused LM-head forward failed");
    let mut out = Vec::with_capacity(2 * rows);
    for t in [loss, lse] {
        let host = nsl_tensor_to_device(t, 0);
        out.extend(test_read_tensor_f32(host).into_iter().map(f32::to_bits));
        nsl_tensor_free(host);
        nsl_tensor_free(t);
    }
    out
}

/// -- check 6: a SUB-VIEW of a registered parameter is served from the
///    parent's image. The fused LM head's chunk GEMMs pass `W + chunk_start *
///    h` with the chunk's element count, which the cache's (base, length) key
///    misses: before `acquire_sub` every chunk was cast into scratch on every
///    call. The fused LM head now takes the WHOLE parent's image once per
///    call (one recast cold, one hit warm) and serves its three chunks (4096
///    + 4096 + a 1024 tail, every chunk at or above the 512 intensity gate)
///    from it; the results are bit-identical to the scratch path's.
fn sub_view_checks() {
    let (rows, h, v) = (1024usize, 1024usize, 2 * 4096 + 1024);
    let x = gpu_2d(rows, h, &deterministic(rows * h, 239));
    let w = gpu_2d(v, h, &deterministic(v * h, 241));
    let zeros = || gpu_2d(v, h, &vec![0.0f32; v * h]);
    let (m, vv, mp) = (zeros(), zeros(), zeros());
    // i64 targets in [0, v): each one is an (low, high) pair of 32-bit words.
    let mut words = Vec::with_capacity(2 * rows);
    for r in 0..rows {
        words.push(f32::from_bits(((r * 37 + 11) % v) as u32));
        words.push(f32::from_bits(0));
    }
    let targets = gpu_2d(rows, 2, &words);

    test_bf16_cast_cache_reset();
    let fresh = lm_head_forward(x, w, targets, rows, v, h);
    assert_eq!(test_bf16_cast_cache_stats(), (0, 0, 0), "nothing registered, yet the cache moved");

    register_via_zero_lr_step(w, m, vv, mp);
    let (h0, r0, _) = test_bf16_cast_cache_stats();
    let cached = lm_head_forward(x, w, targets, rows, v, h);
    let (h1, r1, _) = test_bf16_cast_cache_stats();
    assert_eq!(
        (h1 - h0, r1 - r0),
        (0, 1),
        "a registered head: the call takes the whole parent's image once (one recast, served \
         to all three vocabulary chunks) (hits, recasts) = (0, 1)"
    );
    assert_eq!(cached, fresh, "the sub-view image must serve the bits the scratch cast computes");
    let cached2 = lm_head_forward(x, w, targets, rows, v, h);
    let (h2, r2, _) = test_bf16_cast_cache_stats();
    assert_eq!((h2 - h1, r2 - r1), (1, 0), "a warm parent image serves the call");
    assert_eq!(cached2, fresh);

    // A real optimizer step stales the parent: the next call recasts it whole
    // and computes from the moved theta.
    let g = gpu_2d(v, h, &deterministic(v * h, 251));
    nsl_fase_fused_adamw_step(w, m, vv, g, 0.01, 0.9, 0.1, 0.999, 0.001, 1e-8, 0.0, 1.0, 1.0);
    let after = lm_head_forward(x, w, targets, rows, v, h);
    assert_ne!(after, fresh, "theta moved but the head returned the pre-step loss: a stale image");
    test_bf16_cast_cache_reset();
    let after_fresh = lm_head_forward(x, w, targets, rows, v, h);
    assert_eq!(after, after_fresh, "after a step, the recast parent image differs from the scratch cast");

    for t in [x, w, m, vv, mp, targets, g] {
        nsl_tensor_free(t);
    }
}

/// -- check 7: at hidden 512 (Coder-50M's) the head's logits GEMM passes the
///    intensity gate and its dx and dW GEMMs do not. The backward must
///    recompute the logits on the forward's route, the bf16 head, so they
///    are the forward's logits bit for bit (before, any failing backward GEMM
///    sent all three to the per-GEMM path, whose recompute accumulates in
///    another order). Two chunks: 4096 columns and a 1024-column tail.
fn head_route_checks() {
    let (rows, h, v) = (1024usize, 512usize, 4096 + 1024);
    let x = gpu_2d(rows, h, &deterministic(rows * h, 257));
    let w = gpu_2d(v, h, &deterministic(v * h, 263));
    let mut words = Vec::with_capacity(2 * rows);
    for r in 0..rows {
        words.push(f32::from_bits(((r * 53 + 7) % v) as u32));
        words.push(f32::from_bits(0));
    }
    let targets = gpu_2d(rows, 2, &words);
    let loss = gpu_2d(rows, 1, &vec![0.0f32; rows]);
    let lse = gpu_2d(rows, 1, &vec![0.0f32; rows]);
    let dx = gpu_2d(rows, h, &vec![0.0f32; rows * h]);
    let dw = gpu_2d(v, h, &vec![0.0f32; v * h]);

    let _ = test_lce_logits_probe(true);
    let rc = nsl_fused_linear_ce_forward_gemm(
        data_ptr(x), data_ptr(w), 0, data_ptr(targets), data_ptr(loss), data_ptr(lse),
        1, rows as i64, v as i64, h as i64, 0,
    );
    assert_eq!(rc, 0, "fused LM-head forward failed");
    let rc = nsl_fused_linear_ce_backward_gemm(
        i64::from(1.0f32.to_bits()), data_ptr(x), data_ptr(w), 0, data_ptr(targets), data_ptr(lse),
        data_ptr(dx), data_ptr(dw), 0, 1, rows as i64, v as i64, h as i64, rows as i64, 0,
    );
    assert_eq!(rc, 0, "fused LM-head backward failed");
    let chunks = test_lce_logits_probe(false);

    let (fwd, bwd): (Vec<_>, Vec<_>) = chunks.iter().partition(|c| !c.0);
    assert_eq!((fwd.len(), bwd.len()), (2, 2), "two vocabulary chunks each way");
    assert!(fwd.iter().all(|c| c.1), "at hidden 512 the forward's logits GEMM passes the gate: the bf16 head");
    assert!(
        bwd.iter().all(|c| c.1),
        "the backward must recompute the logits on the forward's route (the bf16 head), \
         whatever its dx and dW GEMMs do"
    );
    for (i, (f, b)) in fwd.iter().zip(&bwd).enumerate() {
        let differ = f.2.iter().zip(&b.2).filter(|(p, q)| p != q).count();
        assert_eq!(differ, 0, "chunk {i}: {differ} of {} recomputed logits differ from the forward's", f.2.len());
    }
    for t in [x, w, targets, loss, lse, dx, dw] {
        nsl_tensor_free(t);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_cache_gates_gpu() {
    if std::env::var(PROBE).is_ok() {
        return; // child process: only the probe runs there.
    }
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args(["--test-threads=1", "--nocapture", "--include-ignored", "zz_probe_child"])
        .env(PROBE, "1")
        .env("NSL_MATMUL_BF16", "1")
        // Scrub EVERY mode var the resolvers consult, not just the two this
        // test owns: a leaked NSL_MATMUL_PEDANTIC=1 outranks BF16=1 and a
        // leaked MIN_RATIO>512 declines this test's ratio-exactly-512 GEMMs
        // — both would fail check 2 with a message blaming the fase wiring.
        .env_remove("NSL_MATMUL_BF16_CAST_CACHE")
        .env_remove("NSL_MATMUL_BF16_ROUND")
        .env_remove("NSL_MATMUL_BF16_MIN_RATIO")
        .env_remove("NSL_MATMUL_PEDANTIC")
        .env_remove("NSL_MATMUL_TF32")
        .env_remove("NSL_ASYNC_ALLOC")
        .output()
        .expect("spawn probe child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains("CACHE-PROBE-OK"),
        "cast-cache probe child failed (status {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status
    );
}
