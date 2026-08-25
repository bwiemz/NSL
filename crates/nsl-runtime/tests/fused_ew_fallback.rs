//! mfu-fusion C3: `nsl_fused_ew_chain`'s decomposed-replay fallback vs
//! hand-sequenced public-FFI calls.
//!
//! The replay IS the CPU arm (and the shape-mismatch / non-contiguous /
//! mixed-device arm on GPU): it interprets the descriptor by calling the
//! ordinary FFIs in original tape order, so it must be BYTE-EQUAL to
//! sequencing those FFIs by hand — that is the whole bit-exactness contract.
//! Every test here runs on CPU tensors with null ptx/kname, so the file
//! needs neither the cuda feature nor a GPU.
//!
//! Leak checking: the CPU allocator exposes no counters reachable from an
//! integration test (`NslTensor.refcount` is pub(crate)), so the
//! refcount-balance assertion lives in the crate-internal unit test
//! `fused_chain::tests::replay_balances_input_refcounts_and_frees_interiors`.
//! Here we pin the observable half: inputs remain alive and VALUE-INTACT
//! after the call, and freeing them afterwards does not double-free.
//!
//! Run: cargo test -p nsl-runtime --test fused_ew_fallback

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::arithmetic::{
    nsl_tensor_add, nsl_tensor_mul, nsl_tensor_neg,
};
use nsl_runtime::tensor::fused_chain::{
    nsl_fused_ew_chain, nsl_fused_ew_fallback_count, nsl_fused_ew_launch_count,
};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get, nsl_tensor_ndim,
    nsl_tensor_reduce_to_shape, nsl_tensor_scalar, nsl_tensor_shape_dim,
};

const DTYPE_F32: i64 = 1;

// --- Descriptor v1 byte layout, duplicated from the runtime on purpose ---
// (a layout drift then breaks these tests instead of the pinned contract).
const OP_ADD: u8 = 0;
#[allow(dead_code)] // part of the duplicated contract table
const OP_SUB: u8 = 1;
const OP_MUL: u8 = 2;
#[allow(dead_code)] // part of the duplicated contract table
const OP_DIV: u8 = 3;
const OP_NEG: u8 = 4;
const OP_RTS: u8 = 5;
const K_INPUT: u8 = 0;
const K_PREV: u8 = 1;
const K_IMM: u8 = 2;
const K_ABSENT: u8 = 255;

fn build_desc(n_inputs: u8, steps: &[(u8, u8, u8, u8, u8, u32)]) -> Vec<u8> {
    let mut d = vec![1u8, steps.len() as u8, n_inputs, 0u8];
    for &(op, lk, li, rk, ri, imm) in steps {
        d.push(op);
        d.push(lk);
        d.push(li);
        d.push(rk);
        d.push(ri);
        d.extend_from_slice(&imm.to_le_bytes());
    }
    d
}

/// Build a CPU f32 tensor over a leaked buffer (pattern from
/// wgrad_accum_fused.rs).
fn tensor(vals: &[f32], shape_dims: &[i64]) -> i64 {
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let shape = nsl_list_new();
    for d in shape_dims {
        nsl_list_push(shape, *d);
    }
    let t = nsl_tensor_from_static(leaked.as_ptr() as i64, shape, DTYPE_F32);
    nsl_list_free(shape);
    t
}

/// Deterministic values in [-1, 1) (pattern from wgrad_accum_fused.rs).
fn det(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Flat read of a rank-1/2 CPU tensor as f64 (`f32 -> f64` is lossless and
/// injective, so `to_bits` equality of the reads IS byte equality of the
/// underlying f32 data).
fn read_all(t: i64) -> Vec<f64> {
    let ndim = nsl_tensor_ndim(t);
    assert!((1..=2).contains(&ndim), "test reader handles rank 1-2, got {ndim}");
    let mut out = Vec::new();
    if ndim == 1 {
        let n = nsl_tensor_shape_dim(t, 0);
        for i in 0..n {
            let idx = nsl_list_new();
            nsl_list_push(idx, i);
            out.push(nsl_tensor_get(t, idx));
            nsl_list_free(idx);
        }
    } else {
        let r = nsl_tensor_shape_dim(t, 0);
        let c = nsl_tensor_shape_dim(t, 1);
        for i in 0..r {
            for j in 0..c {
                let idx = nsl_list_new();
                nsl_list_push(idx, i);
                nsl_list_push(idx, j);
                out.push(nsl_tensor_get(t, idx));
                nsl_list_free(idx);
            }
        }
    }
    out
}

fn assert_byte_equal(chain: i64, reference: i64, what: &str) {
    let cv = read_all(chain);
    let rv = read_all(reference);
    assert_eq!(cv.len(), rv.len(), "{what}: element count differs");
    for (i, (c, r)) in cv.iter().zip(rv.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            r.to_bits(),
            "{what}: byte mismatch at flat index {i} (chain={c}, reference={r})"
        );
    }
}

fn run_chain(desc: &[u8], inputs: &[i64]) -> i64 {
    let mut h = [0i64; 6];
    h[..inputs.len()].copy_from_slice(inputs);
    nsl_fused_ew_chain(
        std::ptr::null(),
        std::ptr::null(),
        desc.as_ptr(),
        desc.len() as u64,
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        inputs.len() as i64,
    )
}

/// (a) 3-op chain mul -> add -> neg, all same shape.
#[test]
fn chain_mul_add_neg_matches_hand_sequenced() {
    let v0 = det(0x11, 8);
    let v1 = det(0x12, 8);
    let v2 = det(0x13, 8);
    let (i0, i1, i2) = (tensor(&v0, &[8]), tensor(&v1, &[8]), tensor(&v2, &[8]));

    let desc = build_desc(
        3,
        &[
            (OP_MUL, K_INPUT, 0, K_INPUT, 1, 0),
            (OP_ADD, K_PREV, 0, K_INPUT, 2, 0),
            (OP_NEG, K_PREV, 1, K_ABSENT, 0, 0),
        ],
    );
    let fb_before = nsl_fused_ew_fallback_count();
    let chain = run_chain(&desc, &[i0, i1, i2]);
    assert_ne!(chain, 0, "chain returned null");
    assert!(
        nsl_fused_ew_fallback_count() > fb_before,
        "CPU call must take (and count) the decomposed replay"
    );

    // Hand-sequenced: literally the ops the compiler removed, same order.
    let t0 = nsl_tensor_mul(i0, i1, 0);
    let t1 = nsl_tensor_add(t0, i2, 0);
    let reference = nsl_tensor_neg(t1);
    assert_byte_equal(chain, reference, "mul->add->neg");

    // Inputs must be alive and value-intact (the FFI never frees or
    // mutates its inputs on the flags=0 replay).
    for (t, v) in [(i0, &v0), (i1, &v1), (i2, &v2)] {
        let vals = read_all(t);
        for (i, (got, want)) in vals.iter().zip(v.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                (*want as f64).to_bits(),
                "input value corrupted at {i}"
            );
        }
    }

    for p in [t0, t1, reference, chain, i0, i1, i2] {
        nsl_tensor_free(p);
    }
}

/// (b) chain with an Imm: mul(i0, imm) -> add(p0, i1). The replay must
/// materialize the imm exactly as baseline Constant lowering does
/// (`nsl_tensor_scalar(v as f64, 1)`).
#[test]
fn chain_with_imm_matches_hand_sequenced_scalar_tensor() {
    let v0 = det(0x21, 6);
    let v1 = det(0x22, 6);
    let (i0, i1) = (tensor(&v0, &[6]), tensor(&v1, &[6]));
    // A value with no short binary expansion, narrowed to f32 exactly as the
    // compile side would bake it (the literal parses to the nearest f32; the
    // descriptor carries those exact bits).
    let imm = 0.517_f32;

    let desc = build_desc(
        2,
        &[
            (OP_MUL, K_INPUT, 0, K_IMM, 0, imm.to_bits()),
            (OP_ADD, K_PREV, 0, K_INPUT, 1, 0),
        ],
    );
    let chain = run_chain(&desc, &[i0, i1]);
    assert_ne!(chain, 0);

    let s = nsl_tensor_scalar(imm as f64, 1);
    let t0 = nsl_tensor_mul(i0, s, 0);
    let reference = nsl_tensor_add(t0, i1, 0);
    assert_byte_equal(chain, reference, "mul(imm)->add");

    for p in [s, t0, reference, chain, i0, i1] {
        nsl_tensor_free(p);
    }
}

/// (c) same-shape RtsCheck: an identity on the tape (reduce_to_shape with a
/// like-ref of the uniform shape). The replay runs the REAL reduce, whose
/// same-shape arm is retain-and-return — output must still byte-match the
/// hand-sequenced version.
#[test]
fn chain_with_identity_rts_matches_hand_sequenced() {
    let v0 = det(0x31, 6);
    let v1 = det(0x32, 6);
    let v2 = det(0x33, 6);
    let shape = [2i64, 3];
    let (i0, i1, i2) = (tensor(&v0, &shape), tensor(&v1, &shape), tensor(&v2, &shape));

    let desc = build_desc(
        3,
        &[
            (OP_MUL, K_INPUT, 0, K_INPUT, 1, 0),
            (OP_RTS, K_PREV, 0, K_INPUT, 2, 0),
            (OP_ADD, K_PREV, 1, K_INPUT, 2, 0),
        ],
    );
    let chain = run_chain(&desc, &[i0, i1, i2]);
    assert_ne!(chain, 0);

    let t0 = nsl_tensor_mul(i0, i1, 0);
    let t1 = nsl_tensor_reduce_to_shape(t0, i2);
    let reference = nsl_tensor_add(t1, i2, 0);
    assert_byte_equal(chain, reference, "mul->rts(identity)->add");

    for p in [t0, t1, reference, chain, i0, i1, i2] {
        nsl_tensor_free(p);
    }
}

/// (d) genuinely-reducing RtsCheck: like-ref [3] against a [2,3] flow — on
/// the fast path this would be a gate miss; the replay must execute the real
/// reduce and keep going.
#[test]
fn chain_with_reducing_rts_matches_hand_sequenced() {
    let v0 = det(0x41, 6);
    let v1 = det(0x42, 6);
    let vlike = det(0x43, 3);
    let (i0, i1) = (tensor(&v0, &[2, 3]), tensor(&v1, &[2, 3]));
    let like = tensor(&vlike, &[3]);

    let desc = build_desc(
        3,
        &[
            (OP_MUL, K_INPUT, 0, K_INPUT, 1, 0),
            (OP_RTS, K_PREV, 0, K_INPUT, 2, 0),
            (OP_ADD, K_PREV, 1, K_INPUT, 2, 0),
        ],
    );
    let chain = run_chain(&desc, &[i0, i1, like]);
    assert_ne!(chain, 0);
    assert_eq!(nsl_tensor_ndim(chain), 1, "reduced result must be rank-1");
    assert_eq!(nsl_tensor_shape_dim(chain, 0), 3);

    let t0 = nsl_tensor_mul(i0, i1, 0);
    let t1 = nsl_tensor_reduce_to_shape(t0, like);
    let reference = nsl_tensor_add(t1, like, 0);
    assert_byte_equal(chain, reference, "mul->rts(reduce)->add");

    for p in [t0, t1, reference, chain, i0, i1, like] {
        nsl_tensor_free(p);
    }
}

/// The fallback counter must move on every replayed call (anti-vacuity for
/// the GPU gates that assert fallbacks == 0). `>=` because counters are
/// process-global and other tests in this binary run concurrently.
#[test]
fn fallback_counter_counts_replays() {
    let v = det(0x51, 4);
    let i0 = tensor(&v, &[4]);
    let desc = build_desc(1, &[(OP_NEG, K_INPUT, 0, K_ABSENT, 0, 0)]);
    let fb = nsl_fused_ew_fallback_count();
    let la = nsl_fused_ew_launch_count();
    let out1 = run_chain(&desc, &[i0]);
    let out2 = run_chain(&desc, &[i0]);
    assert!(nsl_fused_ew_fallback_count() >= fb + 2);
    // No GPU launch can have happened from a CPU tensor with null PTX; the
    // launch counter may still move if unrelated GPU tests share the binary,
    // so only pin it when it cannot (default-features build has no GPU path).
    #[cfg(not(feature = "cuda"))]
    assert_eq!(nsl_fused_ew_launch_count(), la);
    #[cfg(feature = "cuda")]
    let _ = la;
    for p in [out1, out2, i0] {
        nsl_tensor_free(p);
    }
}

// --- (e) malformed descriptor = loud abort, observed from a child process.
// The suite's precedent for observing process-fatal behavior is the
// re-exec-self pattern (matmul_transposed_operand.rs): the parent spawns the
// test binary with a hidden env-gated child test and asserts on its exit.

const ABORT_PROBE_ENV: &str = "NSL_FUSED_EW_ABORT_PROBE";

/// Child half: only does anything when spawned by the parent below.
#[test]
fn zz_fused_ew_abort_probe_child() {
    if std::env::var(ABORT_PROBE_ENV).ok().as_deref() != Some("1") {
        return;
    }
    let v = det(0x61, 4);
    let i0 = tensor(&v, &[4]);
    // Bad version byte: must abort, not degrade.
    let desc = build_desc(1, &[(OP_NEG, K_INPUT, 0, K_ABSENT, 0, 0)]);
    let mut bad = desc.clone();
    bad[0] = 9;
    let _ = run_chain(&bad, &[i0]);
    // If we get here the refusal did not fire; exit 0 would fool the parent.
    println!("ABORTPROBE survived");
}

#[test]
fn malformed_descriptor_aborts_loudly() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_fused_ew_abort_probe_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ABORT_PROBE_ENV, "1")
        .output()
        .expect("spawn abort probe child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "child survived a malformed descriptor (v=9)\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("ABORTPROBE survived"),
        "chain call returned instead of aborting"
    );
    assert!(
        stderr.contains("malformed descriptor"),
        "abort must name the malformed descriptor; stderr was:\n{stderr}"
    );
}

// --- GPU fast-path test: COMPILE-CHECKED under --features cuda, but not run
// (gpu-tier lanes run it with --include-ignored; the card is busy during
// this campaign). Uses a hand-written chain kernel following the emitter
// convention (out first, then inputs, then n; .rn on every arithmetic op).

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use nsl_runtime::tensor::nsl_tensor_to_device;

    /// out[i] = (in0[i] * in1[i]) + in2[i], emitter conventions:
    /// signature (out, in0.., n), `.rn` so ptxas cannot contract the
    /// mul+add into an FMA (the decomposed pair double-rounds).
    const TEST_CHAIN_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_test_fused_mul_add(\n\
    .param .u64 o, .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<10>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [o];\n\
    ld.param.u64 %rd2, [a];\n\
    ld.param.u64 %rd3, [b];\n\
    ld.param.u64 %rd4, [c];\n\
    ld.param.u64 %rd5, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs1, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs2, [%rd8];\n\
    mul.rn.f32 %fs4, %fs1, %fs2;\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs3, [%rd8];\n\
    add.rn.f32 %fs4, %fs4, %fs3;\n\
    add.u64 %rd9, %rd1, %rd7;\n\
    st.global.f32 [%rd9], %fs4;\n\
DONE: ret;\n\
}\0";

    /// Named const (not an inline literal) so `.as_ptr()` matches the
    /// PTX-const convention used everywhere in the crate.
    const TEST_CHAIN_KNAME: &str = "nsl_test_fused_mul_add\0";

    fn cuda_available() -> bool {
        if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
            return false;
        }
        nsl_runtime::nsl_cuda_init() == 0
    }

    /// Fast path vs hand-sequenced GPU ops: byte-equal, launches counter
    /// moves, fallback counter does not.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn gpu_fast_path_matches_hand_sequenced_and_counts() {
        if !cuda_available() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        let n = 257usize; // exercises the tail guard
        let v0 = det(0x71, n);
        let v1 = det(0x72, n);
        let v2 = det(0x73, n);
        let g0 = nsl_tensor_to_device(tensor(&v0, &[n as i64]), 1);
        let g1 = nsl_tensor_to_device(tensor(&v1, &[n as i64]), 1);
        let g2 = nsl_tensor_to_device(tensor(&v2, &[n as i64]), 1);

        let desc = build_desc(
            3,
            &[
                (OP_MUL, K_INPUT, 0, K_INPUT, 1, 0),
                (OP_ADD, K_PREV, 0, K_INPUT, 2, 0),
            ],
        );
        let la = nsl_fused_ew_launch_count();
        let fb = nsl_fused_ew_fallback_count();
        let chain = nsl_fused_ew_chain(
            TEST_CHAIN_PTX.as_ptr(),
            TEST_CHAIN_KNAME.as_ptr(),
            desc.as_ptr(),
            desc.len() as u64,
            g0,
            g1,
            g2,
            0,
            0,
            0,
            3,
        );
        assert_ne!(chain, 0);
        assert!(nsl_fused_ew_launch_count() > la, "fast path did not launch");
        assert_eq!(
            nsl_fused_ew_fallback_count(),
            fb,
            "uniform GPU f32 inputs must not fall back"
        );

        let t0 = nsl_tensor_mul(g0, g1, 0);
        let reference = nsl_tensor_add(t0, g2, 0);
        // Stage both to CPU for the byte compare.
        let c_chain = nsl_tensor_to_device(chain, 0);
        let c_ref = nsl_tensor_to_device(reference, 0);
        assert_byte_equal(c_chain, c_ref, "gpu mul->add");

        for p in [c_chain, c_ref, t0, reference, chain, g0, g1, g2] {
            nsl_tensor_free(p);
        }
    }
}
