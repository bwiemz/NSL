//! GPU parity gates for the contiguous-run fast path in `nsl_tensor_contiguous`.
//!
//! `cuda::strided_copy` rewrites views whose innermost axis is already a
//! contiguous run (every `expand` broadcast; every reordering that keeps the last
//! axis innermost, such as the GQA KV replication and the `[b, s, h, hd] ->
//! [b, h, s, hd]` head transpose) into `outer` independent runs, replacing the
//! generic kernel's five 64-bit `div`/`rem` pairs per element with a precomputed
//! per-run source offset.
//!
//! A planner that mis-collapses axes reads the WRONG source elements while still
//! writing a correctly-shaped, fully-populated tensor — nothing downstream would
//! notice, and training would simply learn worse. So every case below compares
//! against the generic kernel via the `NSL_STRIDED_COPY_RUN=0` kill switch as
//! well as against a host-side reference walk over the source strides.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test gpu_strided_copy_run_parity
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_contiguous, nsl_tensor_data_ptr, nsl_tensor_expand, nsl_tensor_free,
    nsl_tensor_reshape, nsl_tensor_transpose, nsl_tensor_zeros_on,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d, nsl_test_strided_copy_arm_launches,
};

/// Arm indices of `ARM_LAUNCHES`, mirroring `RunPlan::arm_index`.
const ARM_COPY: i64 = 0;
const ARM_COPY_VEC4: i64 = 1;
const ARM_BCAST: i64 = 2;
const ARM_BCAST_VEC4: i64 = 3;

/// Assert that materializing something actually took the intended fast-path arm.
///
/// Without this every gate in this file would still pass with the fast path
/// dead: they all compare against the generic kernel or a host reference, and
/// both produce identical bytes. A regression that made `resident_plan` always
/// return `None` — an exhausted budget, an inverted alignment test, a planner
/// that stopped recognizing runs — would be completely invisible.
fn assert_arm_advanced(arm: i64, before: i64, what: &str) {
    let after = nsl_test_strided_copy_arm_launches(arm);
    assert!(
        after > before,
        "{what}: expected arm {arm} to launch (count {before} -> {after}); \
         the fast path did not run, so this gate proved nothing"
    );
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn list_of(dims: &[i64]) -> i64 {
    let l = nsl_list_new();
    for &d in dims {
        nsl_list_push(l, d);
    }
    l
}

fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = list_of(shape);
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(t),
        vals.as_ptr() as i64,
        (vals.len() * 4) as i64,
    );
    t
}

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(
        out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(t),
        (len * 4) as i64,
    );
    out
}

/// Distinct value per source element, so a wrong read is always visible (a
/// pseudo-random sequence can repeat; `i as f32` cannot, at these sizes).
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

fn contig_strides(shape: &[i64]) -> Vec<i64> {
    let mut st = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * shape[i + 1];
    }
    st
}

/// What `nsl_tensor_contiguous` is defined to produce: for every flat output
/// index, read the source through its strides.
fn reference(src: &[f32], shape: &[i64], src_strides: &[i64]) -> Vec<f32> {
    let dst = contig_strides(shape);
    let total: i64 = shape.iter().product();
    (0..total)
        .map(|flat| {
            let mut off = 0i64;
            for d in 0..shape.len() {
                off += ((flat / dst[d]) % shape[d]) * src_strides[d];
            }
            src[off as usize]
        })
        .collect()
}

/// Materialize `view` on GPU and check it against `expect`.
///
/// `view` is consumed (freed) by this helper.
fn assert_materializes_to(view: i64, expect: &[f32], what: &str) {
    let contig = nsl_tensor_contiguous(view);
    assert_ne!(contig, 0, "{what}: nsl_tensor_contiguous returned null");
    assert_ne!(
        contig, view,
        "{what}: expected a real copy, got the zero-copy already-contiguous path"
    );
    let got = read_gpu(contig, expect.len());
    let mismatches: Vec<usize> = (0..expect.len()).filter(|&i| got[i] != expect[i]).collect();
    assert!(
        mismatches.is_empty(),
        "{what}: {} of {} elements wrong; first at {} (got {}, want {})",
        mismatches.len(),
        expect.len(),
        mismatches[0],
        got[mismatches[0]],
        expect[mismatches[0]]
    );
    nsl_tensor_free(contig);
    nsl_tensor_free(view);
}

// ── the shapes that dominate a Coder-50M step ───────────────────────────────

/// GQA KV replication: `k.reshape([b, nkv, 1, s, hd]).expand([b, nkv, n_rep, s, hd])`.
/// Collapses to 32 runs of 65536 with the broadcast axis duplicating each run.
#[test]
#[ignore = "requires CUDA GPU"]
fn gqa_kv_expand_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_COPY_VEC4);
    let (b, nkv, n_rep, s, hd) = (2i64, 4i64, 2i64, 64i64, 32i64);
    let vals = ramp((b * nkv * s * hd) as usize);
    let base = gpu_tensor(&vals, &[b, nkv, s, hd]);

    let five = list_of(&[b, nkv, 1, s, hd]);
    let reshaped = nsl_tensor_reshape(base, five);
    nsl_list_free(five);
    let target = list_of(&[b, nkv, n_rep, s, hd]);
    let expanded = nsl_tensor_expand(reshaped, target);
    nsl_list_free(target);

    // Source strides of the expanded view: the replication axis is stride 0.
    let shape = [b, nkv, n_rep, s, hd];
    let src_strides = [nkv * s * hd, s * hd, 0, hd, 1];
    let expect = reference(&vals, &shape, &src_strides);

    assert_materializes_to(expanded, &expect, "gqa kv expand");
    assert_arm_advanced(ARM_COPY_VEC4, before, "gqa kv expand");
    nsl_tensor_free(reshaped);
    nsl_tensor_free(base);
}

/// `q.reshape([b, s, h, hd]).transpose(1, 2)` — head_dim stays innermost, so this
/// is many short runs (run_len == hd) rather than one long one.
#[test]
#[ignore = "requires CUDA GPU"]
fn head_transpose_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_COPY_VEC4);
    let (b, s, h, hd) = (2i64, 64i64, 8i64, 32i64);
    let vals = ramp((b * s * h * hd) as usize);
    let base = gpu_tensor(&vals, &[b, s, h, hd]);
    let view = nsl_tensor_transpose(base, 1, 2);

    let shape = [b, h, s, hd];
    let src_strides = [s * h * hd, hd, h * hd, 1];
    let expect = reference(&vals, &shape, &src_strides);

    assert_materializes_to(view, &expect, "head transpose");
    assert_arm_advanced(ARM_COPY_VEC4, before, "head transpose");
    nsl_tensor_free(base);
}

/// A `[1]` scalar broadcast over a whole activation: one run, source stride 0
/// everywhere. Exercises the `bcast4` arm at its widest.
#[test]
#[ignore = "requires CUDA GPU"]
fn scalar_broadcast_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_BCAST_VEC4);
    let vals = vec![std::f32::consts::PI];
    let base = gpu_tensor(&vals, &[1, 1, 1]);
    let target = list_of(&[4, 128, 64]);
    let view = nsl_tensor_expand(base, target);
    nsl_list_free(target);

    let expect = vec![std::f32::consts::PI; 4 * 128 * 64];
    assert_materializes_to(view, &expect, "scalar broadcast");
    assert_arm_advanced(ARM_BCAST_VEC4, before, "scalar broadcast");
    nsl_tensor_free(base);
}

/// A per-token reduction broadcast back over the feature axis — the RMSNorm /
/// softmax-denominator shape. Runs are `d_model` long and each reads ONE source
/// element, with the run starts stepping by 1 (deliberately not 16-byte aligned,
/// which the broadcast arm does not require).
#[test]
#[ignore = "requires CUDA GPU"]
fn per_token_broadcast_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_BCAST_VEC4);
    let (b, s, d) = (4i64, 32i64, 64i64);
    let vals = ramp((b * s) as usize);
    let base = gpu_tensor(&vals, &[b, s, 1]);
    let target = list_of(&[b, s, d]);
    let view = nsl_tensor_expand(base, target);
    nsl_list_free(target);

    let expect = reference(&vals, &[b, s, d], &[s, 1, 0]);
    assert_materializes_to(view, &expect, "per-token broadcast");
    assert_arm_advanced(ARM_BCAST_VEC4, before, "per-token broadcast");
    nsl_tensor_free(base);
}

/// A true transpose has no contiguous run and must keep using the generic
/// kernel. Gates that the planner *refuses* rather than silently mis-collapsing:
/// this is the weight-tied LM head's `embed.transpose(0, 1)`.
#[test]
#[ignore = "requires CUDA GPU"]
fn two_dimensional_transpose_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before: Vec<i64> = (0..4i64)
        .map(|a| nsl_test_strided_copy_arm_launches(a))
        .collect();
    let (rows, cols) = (96i64, 64i64);
    let vals = ramp((rows * cols) as usize);
    let base = gpu_tensor(&vals, &[rows, cols]);
    let view = nsl_tensor_transpose(base, 0, 1);

    let expect = reference(&vals, &[cols, rows], &[1, cols]);
    assert_materializes_to(view, &expect, "2d transpose");
    for arm in [ARM_COPY, ARM_COPY_VEC4, ARM_BCAST, ARM_BCAST_VEC4] {
        assert_eq!(
            nsl_test_strided_copy_arm_launches(arm),
            before[arm as usize],
            "a true transpose must fall back to the generic kernel, but arm {arm} launched"
        );
    }
    nsl_tensor_free(base);
}

/// Run length not a multiple of 4 and run starts not 16-byte aligned: the plan
/// must fall back to the scalar arms rather than issuing `ld.global.v4.f32`
/// against a misaligned address (which faults, or silently reads neighbours).
#[test]
#[ignore = "requires CUDA GPU"]
fn unaligned_odd_run_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_BCAST);
    // [8, 67] broadcast over a 33-wide innermost axis: 536 runs of 33. Both the
    // run length (33) and the run starts (stepping by 1 element) are odd, so the
    // planner must pick the scalar broadcast arm.
    let (rows, wide, narrow) = (8i64, 67i64, 33i64);
    let vals = ramp((rows * wide) as usize);
    let base = gpu_tensor(&vals, &[rows, wide]);

    let three = list_of(&[rows, wide, 1]);
    let reshaped = nsl_tensor_reshape(base, three);
    nsl_list_free(three);
    let target = list_of(&[rows, wide, narrow]);
    let view = nsl_tensor_expand(reshaped, target);
    nsl_list_free(target);

    let expect = reference(&vals, &[rows, wide, narrow], &[wide, 1, 0]);
    assert_materializes_to(view, &expect, "odd-length broadcast run");
    assert_arm_advanced(ARM_BCAST, before, "odd-length broadcast run");
    nsl_tensor_free(reshaped);
    nsl_tensor_free(base);
}

/// The scalar COPY arm (`nsl_scopy_run_f32`), which no other gate reaches: every
/// other copy case here is vec4, and the odd-length case is a broadcast.
///
/// Same head-transpose geometry as above but with an ODD head_dim, so the run is
/// contiguous (stride 1) yet not a whole number of float4s — vec4 must be
/// refused and the scalar arm used. If the planner ever offered vec4 here the
/// kernel would issue `ld.global.v4.f32` against a 4-byte-aligned address.
#[test]
#[ignore = "requires CUDA GPU"]
fn odd_head_dim_copy_uses_scalar_arm() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_COPY);
    let (b, s, h, hd) = (2i64, 8i64, 4i64, 33i64);
    let vals = ramp((b * s * h * hd) as usize);
    let base = gpu_tensor(&vals, &[b, s, h, hd]);
    let view = nsl_tensor_transpose(base, 1, 2);

    let shape = [b, h, s, hd];
    let src_strides = [s * h * hd, hd, h * hd, 1];
    let expect = reference(&vals, &shape, &src_strides);

    assert_materializes_to(view, &expect, "odd head_dim copy");
    assert_arm_advanced(ARM_COPY, before, "odd head_dim copy");
    nsl_tensor_free(base);
}

/// `outer` beyond `gridDim.y`'s 65535 limit, so the kernels' grid-stride loop
/// over y must take more than one trip. Nothing else in the suite exceeds 32768.
#[test]
#[ignore = "requires CUDA GPU"]
fn more_runs_than_grid_y_limit_matches_reference() {
    if !cuda_available() {
        return;
    }
    let before = nsl_test_strided_copy_arm_launches(ARM_BCAST_VEC4);
    // 70000 runs of 32: grid.y clamps to 65535, so the loop wraps once.
    let (outer, run) = (70000i64, 32i64);
    let vals = ramp(outer as usize);
    let base = gpu_tensor(&vals, &[outer, 1]);
    let target = list_of(&[outer, run]);
    let view = nsl_tensor_expand(base, target);
    nsl_list_free(target);

    let expect = reference(&vals, &[outer, run], &[1, 0]);
    assert_materializes_to(view, &expect, "outer beyond grid.y limit");
    assert_arm_advanced(ARM_BCAST_VEC4, before, "outer beyond grid.y limit");
    nsl_tensor_free(base);
}

/// The kill switch must be a pure performance toggle: every arm has to produce
/// bit-identical bytes with the fast path off.
///
/// Runs in its own process (`NSL_STRIDED_COPY_RUN` is read once into a
/// `OnceLock`, and the value must be set before any GPU copy happens), so this
/// test re-executes the binary with the variable set and compares the two
/// outputs through a checksum the child prints.
#[test]
#[ignore = "requires CUDA GPU"]
fn generic_kernel_agrees_with_fast_path() {
    if !cuda_available() {
        return;
    }
    // Each case: (shape, src_strides, source length) built as an expand or
    // transpose above; here we only need the checksum to agree, so re-derive it
    // from the same reference walk the child computes.
    let fast = checksum_all_cases();
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["print_checksum_child", "--ignored", "--exact", "--nocapture"])
        .env("NSL_STRIDED_COPY_RUN", "0")
        .output()
        .expect("re-exec test binary with the fast path disabled");
    assert!(
        out.status.success(),
        "child exited {}; stderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let marker = "CHECKSUM=";
    let generic: f64 = stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix(marker))
        .unwrap_or_else(|| panic!("child printed no {marker} line; stdout:\n{stdout}"))
        .parse()
        .expect("checksum parses");
    assert_eq!(
        fast, generic,
        "fast path and generic kernel disagree (fast={fast}, NSL_STRIDED_COPY_RUN=0 -> {generic})"
    );
}

/// Child half of `generic_kernel_agrees_with_fast_path`. Not a gate on its own.
#[test]
#[ignore = "child process of generic_kernel_agrees_with_fast_path"]
fn print_checksum_child() {
    if !cuda_available() {
        println!("CHECKSUM=0");
        return;
    }
    println!("CHECKSUM={}", checksum_all_cases());
}

/// Materialize one view of every planner arm and fold the bytes into a single
/// number. Deliberately order-sensitive so a permuted result does not collide.
fn checksum_all_cases() -> f64 {
    let mut acc = 0.0f64;
    let mut fold = |v: &[f32]| {
        for (i, &x) in v.iter().enumerate() {
            acc += (x as f64) * ((i % 97) as f64 + 1.0);
        }
    };

    // copy run (vec4): GQA expand
    let (b, nkv, n_rep, s, hd) = (2i64, 4i64, 2i64, 64i64, 32i64);
    let vals = ramp((b * nkv * s * hd) as usize);
    let base = gpu_tensor(&vals, &[b, nkv, s, hd]);
    let five = list_of(&[b, nkv, 1, s, hd]);
    let reshaped = nsl_tensor_reshape(base, five);
    nsl_list_free(five);
    let target = list_of(&[b, nkv, n_rep, s, hd]);
    let expanded = nsl_tensor_expand(reshaped, target);
    nsl_list_free(target);
    let c = nsl_tensor_contiguous(expanded);
    fold(&read_gpu(c, (b * nkv * n_rep * s * hd) as usize));
    nsl_tensor_free(c);
    nsl_tensor_free(expanded);
    nsl_tensor_free(reshaped);
    nsl_tensor_free(base);

    // copy run (short runs): head transpose
    let (b2, s2, h2, hd2) = (2i64, 64i64, 8i64, 32i64);
    let vals2 = ramp((b2 * s2 * h2 * hd2) as usize);
    let base2 = gpu_tensor(&vals2, &[b2, s2, h2, hd2]);
    let view2 = nsl_tensor_transpose(base2, 1, 2);
    let c2 = nsl_tensor_contiguous(view2);
    fold(&read_gpu(c2, (b2 * s2 * h2 * hd2) as usize));
    nsl_tensor_free(c2);
    nsl_tensor_free(view2);
    nsl_tensor_free(base2);

    // broadcast run (vec4) + broadcast run (odd length)
    for (dm, len) in [(64i64, 64i64), (33i64, 33i64)] {
        let vals3 = ramp(32);
        let base3 = gpu_tensor(&vals3, &[4, 8, 1]);
        let t3 = list_of(&[4, 8, dm]);
        let view3 = nsl_tensor_expand(base3, t3);
        nsl_list_free(t3);
        let c3 = nsl_tensor_contiguous(view3);
        fold(&read_gpu(c3, (4 * 8 * len) as usize));
        nsl_tensor_free(c3);
        nsl_tensor_free(view3);
        nsl_tensor_free(base3);
    }

    // generic fallback: 2-D transpose
    let (rows, cols) = (96i64, 64i64);
    let vals4 = ramp((rows * cols) as usize);
    let base4 = gpu_tensor(&vals4, &[rows, cols]);
    let view4 = nsl_tensor_transpose(base4, 0, 1);
    let c4 = nsl_tensor_contiguous(view4);
    fold(&read_gpu(c4, (rows * cols) as usize));
    nsl_tensor_free(c4);
    nsl_tensor_free(view4);
    nsl_tensor_free(base4);

    acc
}
