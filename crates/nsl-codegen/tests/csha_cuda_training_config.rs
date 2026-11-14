//! GPU gate for the kernel pair a `--csha` training build actually emits.
//!
//! `maybe_synthesize_csha_training_ptx` builds one configuration: fusion
//! level 1 with every fusion flag off — a FlashAttention forward that also
//! saves Q, K, V and the softmax statistics, and the scalar fused backward
//! that reads them. Q, K and V arrive already projected. This file launches
//! the attention part of that pair (dQ, dK, dV; no projection weights) over
//! more than one batch element and more than one head, which
//! the fused-projection gates in `csha_cuda_backward.rs` (one batch element,
//! one head) cannot, and compares with the f64 reference in
//! `csha_f64_reference.rs` (pinned to PyTorch).
//!
//! It also covers the launch grid under head pruning. The kernels derive
//! `batch = block_y / heads` and `head = block_y % heads` from the FULL head
//! count and return early for a pruned head, so the grid must cover
//! `batch * heads` blocks. The launchers used to launch `batch * active_heads`
//! blocks: with batch 2, 8 heads and 4 active that is 8 blocks, all of which
//! decode to batch element 0, and batch element 1 kept its zero-initialised
//! output under a success return code.
//!
//! ```text
//! cargo test -p nsl-codegen --release --features cuda \
//!     --test csha_cuda_training_config -- --ignored --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "csha_f64_reference.rs"]
mod csha_f64_reference;
use csha_f64_reference::{attention_backward_f64_from_f32, d_o_seq, det_seq, rel_l2, AttnGrads64};

use std::ffi::CString;

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2_backward,
    smem_layout::{self, needs_dynamic_smem},
    synthesize_backward, synthesize_forward_multi_tile_combined,
};

use nsl_runtime::flash_attention::{
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha, nsl_flash_attention_csha_backward,
    nsl_flash_attention_csha_with_saves,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
    nsl_test_cuda_jit_log,
};

/// Tile size the training build pins for both the forward-with-saves and the
/// backward (`backward_block_q` / `backward_block_kv` in
/// `compiler/kernel.rs`).
const BLOCK: usize = 32;

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    causal: bool,
}

impl Shape {
    fn elems(&self) -> usize {
        self.batch * self.heads * self.seq * self.head_dim
    }
    fn slice(&self) -> usize {
        self.seq * self.head_dim
    }
}

/// The training preset of `maybe_synthesize_csha_training_ptx`, with
/// `d_model` 0: Q, K and V are given, there are no projection weights, and a
/// zero `d_model` is how the backward is told to emit no projection or
/// RMSNorm phases. (With a nonzero `d_model` the backward reads Wq, Wk and Wv
/// unconditionally, so null weights are a GPU fault, not a skipped phase.)
fn training_config(shape: &Shape) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: BLOCK as i64,
        block_kv: BLOCK as i64,
        head_dim: shape.head_dim as i64,
        causal: shape.causal,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 1,
            fused_rmsnorm: false,
            fused_projections: false,
            fused_output_proj: false,
            active_heads: 0,
            rmsnorm_eps: 1e-5,
            d_model: 0,
            save_activations_for_backward: true,
            skip_rmsnorm_prologue: false,
            static_seq_len: None,
        }),
        checkpoint: None,
    }
}

struct Run {
    /// `[batch, heads, seq, head_dim]` each.
    o: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

struct Qkv {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Qkv {
    fn new(shape: &Shape) -> Self {
        // Order-1 activations, a different stream per tensor.
        let draw = |seed: u32| -> Vec<f32> {
            det_seq(seed, shape.elems()).into_iter().map(|u| 2.0 * u).collect()
        };
        Qkv { q: draw(11), k: draw(12), v: draw(13) }
    }

    /// The f64 reference for one (batch, head) slice.
    fn reference(&self, shape: &Shape, b: usize, h: usize, d_o: &[f32]) -> AttnGrads64 {
        let n = shape.slice();
        let at = (b * shape.heads + h) * n;
        attention_backward_f64_from_f32(
            &self.q[at..at + n],
            &self.k[at..at + n],
            &self.v[at..at + n],
            &d_o[at..at + n],
            shape.seq,
            shape.head_dim,
            shape.causal,
        )
    }
}

fn jit_log(ptx: &[u8]) -> String {
    let p = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
    if p != 0 {
        unsafe { std::ffi::CStr::from_ptr(p as *const i8) }.to_string_lossy().into_owned()
    } else {
        "<no log>".into()
    }
}

fn upload_f32(host: &[f32]) -> i64 {
    let bytes = (host.len() * 4) as i64;
    let dev = nsl_test_cuda_alloc(bytes);
    assert_ne!(dev, 0, "device allocation failed");
    nsl_test_cuda_h2d(dev, host.as_ptr() as i64, bytes);
    dev
}

fn zeroed_device(bytes: usize) -> i64 {
    let dev = nsl_test_cuda_alloc(bytes as i64);
    assert_ne!(dev, 0, "device allocation failed");
    let zeros = vec![0u8; bytes];
    nsl_test_cuda_h2d(dev, zeros.as_ptr() as i64, bytes as i64);
    dev
}

fn read_f32(dev: i64, elems: usize) -> Vec<f32> {
    let mut out = vec![0f32; elems];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, (elems * 4) as i64);
    out
}

fn read_f16_as_f32(dev: i64, elems: usize) -> Vec<f32> {
    let mut raw = vec![0u16; elems];
    nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, dev, (elems * 2) as i64);
    raw.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
}

/// Forward with saves, then the backward, as a training step launches them.
/// `active_heads` is passed to both launchers (0 = all heads).
fn forward_and_backward(
    shape: &Shape,
    qkv: &Qkv,
    d_o: &[f32],
    active_heads: usize,
) -> Result<Run, String> {
    assert_eq!(shape.seq % BLOCK, 0, "the training kernels need seq to be a multiple of {BLOCK}");
    let config = training_config(shape);
    let scale = 1.0f32 / (shape.head_dim as f32).sqrt();
    let n = shape.elems();
    let (b, h, s, hd) =
        (shape.batch as i64, shape.heads as i64, shape.seq as i64, shape.head_dim as i64);

    nsl_cuda_init();
    let q_dev = upload_f32(&qkv.q);
    let k_dev = upload_f32(&qkv.k);
    let v_dev = upload_f32(&qkv.v);
    let do_dev = upload_f32(d_o);
    // Raw-pointer contract: the forward stores its output as f16.
    let out_dev = zeroed_device(n * 2);
    let lse_dev = zeroed_device(shape.batch * shape.heads * shape.seq * 4);
    let dq_dev = zeroed_device(n * 4);
    let dk_dev = zeroed_device(n * 4);
    let dv_dev = zeroed_device(n * 4);
    let all = [q_dev, k_dev, v_dev, do_dev, out_dev, lse_dev, dq_dev, dk_dev, dv_dev];
    let saves = unsafe { nsl_csha_alloc_backward_activations(b, h, s, hd) };
    let cleanup = || {
        unsafe { nsl_csha_free_backward_activations(saves) };
        for p in all {
            nsl_test_cuda_free(p);
        }
    };

    let fwd_ptx = synthesize_forward_multi_tile_combined(&config);
    let fwd_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let fwd_smem =
        if needs_dynamic_smem(&config) { smem_layout::total_bytes(&config) as i64 } else { 0 };
    let rc = nsl_flash_attention_csha_with_saves(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        b, h, s, hd,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        fwd_smem,
        fwd_ptx.as_ptr() as i64, fwd_name.as_ptr() as i64,
        BLOCK as i64, BLOCK as i64,
        i64::from(shape.causal),
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        active_heads as i64, 0,
        saves.q_proj, saves.k_proj, saves.v_proj,
        saves.row_max, saves.row_sum,
        saves.x_raw,
        0, 0, 0, 0, 0,
    );
    if rc != 0 {
        let log = jit_log(&fwd_ptx);
        cleanup();
        return Err(format!("forward rc={rc}\nJIT log:\n{log}"));
    }

    let mut bwd_ptx = synthesize_backward(&config).map_err(|e| format!("synth backward: {e}"))?;
    if !bwd_ptx.ends_with('\0') {
        bwd_ptx.push('\0');
    }
    let bwd_ptx = bwd_ptx.into_bytes();
    let fw = flash_attention_kernel_name_v2(&config);
    let bwd_name = CString::new(match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{rest}"),
        None => format!("flash_attn_backward_{fw}"),
    })
    .unwrap();
    let bwd_smem = shared_mem_bytes_v2_backward(&config);
    let bwd_smem_dyn =
        if bwd_smem > smem_layout::SMEM_BUDGET_BYTES { bwd_smem as i64 } else { 0 };
    let rc = nsl_flash_attention_csha_backward(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        b, h, s, hd,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        bwd_smem_dyn,
        bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
        BLOCK as i64, BLOCK as i64,
        i64::from(shape.causal),
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        active_heads as i64, 0,
        saves.q_proj, saves.k_proj, saves.v_proj,
        saves.row_max, saves.row_sum,
        saves.x_raw,
        do_dev, dq_dev, dk_dev, dv_dev,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    );
    if rc != 0 {
        let log = jit_log(&bwd_ptx);
        cleanup();
        return Err(format!("backward rc={rc}\nJIT log:\n{log}"));
    }

    let run = Run {
        o: read_f16_as_f32(out_dev, n),
        dq: read_f32(dq_dev, n),
        dk: read_f32(dk_dev, n),
        dv: read_f32(dv_dev, n),
    };
    cleanup();
    Ok(run)
}

/// Relative L2 tolerance per (batch, head) slice. The gradients are f32; the
/// operands Q, K and V are saved as f16 (2^-11 relative rounding each) and
/// the kernels use approximate exp and divide. Measured worst case on an
/// RTX PRO 4500: 4e-4.
const REL_TOL: f64 = 4e-3;

/// Compare every (batch, head) slice with its reference. A slice whose head
/// is pruned (`h >= active_heads > 0`) must be exactly zero instead.
fn compare(shape: &Shape, qkv: &Qkv, d_o: &[f32], run: &Run, active_heads: usize) -> Vec<String> {
    let mut failures = Vec::new();
    let n = shape.slice();
    for b in 0..shape.batch {
        for h in 0..shape.heads {
            let at = (b * shape.heads + h) * n;
            let tensors: [(&str, &[f32]); 4] = [
                ("o", &run.o[at..at + n]),
                ("dq", &run.dq[at..at + n]),
                ("dk", &run.dk[at..at + n]),
                ("dv", &run.dv[at..at + n]),
            ];
            if active_heads > 0 && h >= active_heads {
                for (name, got) in tensors {
                    let nonzero = got.iter().filter(|v| **v != 0.0).count();
                    if nonzero != 0 {
                        failures.push(format!(
                            "batch {b} head {h} is pruned but {name} has {nonzero} nonzero entries"
                        ));
                    }
                }
                continue;
            }
            let want = qkv.reference(shape, b, h, d_o);
            let wants: [&[f64]; 4] = [&want.o, &want.dq, &want.dk, &want.dv];
            let mut line = format!("[batch {b} head {h}]");
            for ((name, got), reference) in tensors.into_iter().zip(wants) {
                let err = rel_l2(name, got, reference);
                line.push_str(&format!(" {name}={err:.2e}"));
                if err.is_nan() || err >= REL_TOL {
                    let zeros = got.iter().filter(|v| **v == 0.0).count();
                    failures.push(format!(
                        "batch {b} head {h}: {name} off by {err:.3e} relative (tol {REL_TOL:.0e}), \
                         {zeros} of {} entries exactly zero",
                        got.len()
                    ));
                }
            }
            eprintln!("{line}");
        }
    }
    failures
}

/// The emitted training pair over a batch of two and four heads at the
/// gradient magnitudes of a real pretraining step.
///
/// One tile (`seq == BLOCK`). At more than one tile this pair's backward is
/// wrong, whatever the batch and head counts: at two tiles the forward
/// output is right (3e-4) and dQ, dK and dV are 130%, 45% and 43% off, at
/// batch 1 with one head as at batch 2 with four. That is a separate defect
/// and is not gated here.
#[test]
#[ignore = "requires CUDA GPU"]
fn training_kernels_match_reference_over_batch_and_heads() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let mut failures = Vec::new();
    for causal in [true, false] {
        let shape = Shape { batch: 2, heads: 4, seq: BLOCK, head_dim: 32, causal };
        let qkv = Qkv::new(&shape);
        for magnitude in [1e-6f32, 1e-7] {
            let d_o = d_o_seq(shape.elems(), magnitude);
            match forward_and_backward(&shape, &qkv, &d_o, 0) {
                Ok(run) => {
                    eprintln!("-- causal={causal} dO~{magnitude:e}");
                    failures.extend(
                        compare(&shape, &qkv, &d_o, &run, 0)
                            .into_iter()
                            .map(|f| format!("causal={causal} dO~{magnitude:e}: {f}")),
                    );
                }
                Err(e) => failures.push(format!("causal={causal} dO~{magnitude:e}: {e}")),
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Head pruning with a batch above one: every batch element must still be
/// computed for the active heads, in the forward and in the backward.
#[test]
#[ignore = "requires CUDA GPU"]
fn pruned_heads_do_not_drop_batch_elements() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let shape = Shape { batch: 2, heads: 8, seq: 32, head_dim: 32, causal: true };
    let qkv = Qkv::new(&shape);
    let d_o = d_o_seq(shape.elems(), 1e-6);
    let run = forward_and_backward(&shape, &qkv, &d_o, 4).expect("launch");
    let failures = compare(&shape, &qkv, &d_o, &run, 4);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The same grid through the forward entry point that takes no save
/// buffers (`nsl_flash_attention_csha`, the inference call).
#[test]
#[ignore = "requires CUDA GPU"]
fn pruned_heads_do_not_drop_batch_elements_in_the_inference_forward() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let shape = Shape { batch: 2, heads: 8, seq: 32, head_dim: 32, causal: true };
    let active_heads = 4usize;
    let qkv = Qkv::new(&shape);
    let config = training_config(&shape);
    let scale = 1.0f32 / (shape.head_dim as f32).sqrt();
    let n = shape.elems();

    let q_dev = upload_f32(&qkv.q);
    let k_dev = upload_f32(&qkv.k);
    let v_dev = upload_f32(&qkv.v);
    let out_dev = zeroed_device(n * 2);
    let lse_dev = zeroed_device(shape.batch * shape.heads * shape.seq * 4);
    let ptx = synthesize_forward_multi_tile_combined(&config);
    let name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let smem =
        if needs_dynamic_smem(&config) { smem_layout::total_bytes(&config) as i64 } else { 0 };
    let rc = nsl_flash_attention_csha(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        shape.batch as i64, shape.heads as i64, shape.seq as i64, shape.head_dim as i64,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        smem,
        ptx.as_ptr() as i64, name.as_ptr() as i64,
        BLOCK as i64, BLOCK as i64,
        1,
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        active_heads as i64, 0,
        0, 0, 0, 0, 0,
    );
    let o = read_f16_as_f32(out_dev, n);
    for p in [q_dev, k_dev, v_dev, out_dev, lse_dev] {
        nsl_test_cuda_free(p);
    }
    assert_eq!(rc, 0, "launch rc={rc}\n{}", jit_log(&ptx));

    let zeros = vec![0f32; n];
    let slice = shape.slice();
    let mut failures = Vec::new();
    for b in 0..shape.batch {
        for h in 0..shape.heads {
            let at = (b * shape.heads + h) * slice;
            let got = &o[at..at + slice];
            if h >= active_heads {
                if got.iter().any(|v| *v != 0.0) {
                    failures.push(format!("batch {b} head {h} is pruned but its output is nonzero"));
                }
                continue;
            }
            let want = qkv.reference(&shape, b, h, &zeros);
            let err = rel_l2("o", got, &want.o);
            eprintln!("[batch {b} head {h}] o={err:.2e}");
            if err.is_nan() || err >= REL_TOL {
                failures.push(format!("batch {b} head {h}: output off by {err:.3e} relative"));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
