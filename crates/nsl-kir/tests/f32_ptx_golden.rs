//! The f32 kernels of every family that bf16 activation storage makes
//! dtype-generic, held byte for byte to the PTX they printed before that work.
//!
//! The storage-generic builders (bf16 or f32 per operand, f32 compute) replace
//! the f32-only ones, and the f32 instantiation must not move: every
//! `*_kir_equivalence` gate, every measured number and every checkpoint
//! fingerprint was taken on these exact bytes. `tests/golden_f32/<kernel>.ptx`
//! was written from the builders as they stood before the refactor
//! (`NSL_BLESS_F32_GOLDEN=1 cargo test -p nsl-kir --test f32_ptx_golden`
//! re-blesses, which is only right when an f32 kernel is changed on purpose).

use std::path::PathBuf;

use nsl_kir::kernels::flash_attn::{bwd, fwd};
use nsl_kir::kernels::{det_sum, elementwise, lce_chunk, norm, rmsnorm_dx, strided_copy};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden_f32")
}

/// Every f32 kernel the bf16-storage families touch, by entry name.
fn kernels() -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    for op in elementwise::BinaryOp::ALL {
        out.push((op.kernel_name().to_string(), elementwise::binary_ptx(op)));
    }
    for op in elementwise::ScalarOp::ALL {
        out.push((op.kernel_name().to_string(), elementwise::scalar_ptx(op)));
    }
    for op in elementwise::UnaryOp::ALL {
        out.push((op.kernel_name().to_string(), elementwise::unary_ptx(op)));
    }
    for op in elementwise::BackwardOp::ALL {
        out.push((op.kernel_name().to_string(), elementwise::backward_ptx(op)));
    }
    for (op, name) in [
        (elementwise::RotateHalfOp::Plain, "nsl_rotate_half_f32"),
        (elementwise::RotateHalfOp::Neg, "nsl_rotate_half_neg_f32"),
    ] {
        out.push((name.to_string(), elementwise::rotate_half_ptx(op)));
    }
    for op in norm::NormOp::ALL {
        out.push((format!("norm_{op:?}"), norm::ptx(op)));
    }
    for op in rmsnorm_dx::RmsNormDxOp::ALL {
        out.push((format!("rmsnorm_dx_{op:?}"), rmsnorm_dx::ptx(op)));
    }
    out.push(("scopy_module".to_string(), strided_copy::ptx()));
    for op in det_sum::DetSumOp::ALL {
        out.push((format!("det_sum_{op:?}"), det_sum::ptx(op)));
    }
    for op in lce_chunk::LceChunkOp::ALL {
        out.push((format!("lce_chunk_{op:?}"), lce_chunk::ptx(op)));
    }
    // The tensor-core attention, every head_dim and mask, at the launched
    // tilings (bf16 I/O, option (a), must leave these bytes alone).
    for hd in fwd::FwdConfig::HEAD_DIMS {
        for causal in [false, true] {
            let f = fwd::FwdConfig::new(hd, causal);
            out.push((f.kernel_name(), fwd::ptx(f)));
            let b = bwd::BwdConfig::new(hd, causal);
            out.push((b.dq_kernel_name(), bwd::dq_ptx(b)));
            out.push((b.dkdv_kernel_name(), bwd::dkdv_ptx(b)));
        }
        let pre = bwd::BwdConfig::pre_only(hd);
        out.push((pre.pre_kernel_name(), bwd::pre_ptx(pre)));
    }
    // The fused optimizer steps, pinned 2026-11-20 before bf16 parameter
    // storage (`param_dtype=`) rebuilt the multi-tensor kernel as a family
    // over the parameter's storage: today's kernels are the family's f32 (and
    // CSLA SR) members and must print these exact bytes.
    use nsl_kir::kernels::optim;
    for (name, ptx) in [
        (optim::FASE_ADAMW_STEP_NAME, optim::fase_adamw_step_ptx()),
        (
            optim::FASE_ADAMW_STEP_MBF16_NAME,
            optim::fase_adamw_step_mbf16_ptx(),
        ),
        (optim::FASE_ADAMW_MULTI_NAME, optim::fase_adamw_multi_ptx()),
        (
            optim::FASE_ADAMW_MULTI_MBF16_NAME,
            optim::fase_adamw_multi_mbf16_ptx(),
        ),
        (
            optim::FASE_ADAMW_STEP_BF16SR_NAME,
            optim::fase_adamw_step_bf16sr_ptx(),
        ),
        (
            optim::FASE_ADAMW_MULTI_BF16SR_NAME,
            optim::fase_adamw_multi_bf16sr_ptx(),
        ),
        (
            optim::SR_BF16_ROUND_PROBE_NAME,
            optim::sr_bf16_round_probe_ptx(),
        ),
    ] {
        out.push((name.to_string(), ptx));
    }
    out
}

#[test]
fn every_f32_kernel_prints_the_bytes_it_printed_before_bf16_storage() {
    let dir = golden_dir();
    let bless = std::env::var("NSL_BLESS_F32_GOLDEN").is_ok_and(|v| v == "1");
    if bless {
        std::fs::create_dir_all(&dir).expect("mkdir golden_f32");
    }
    let mut checked = 0;
    let mut moved = Vec::new();
    for (name, ptx) in kernels() {
        let path = dir.join(format!("{name}.ptx"));
        if bless {
            std::fs::write(&path, &ptx).expect("write golden");
            continue;
        }
        let want = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (bless with NSL_BLESS_F32_GOLDEN=1)", path.display()));
        if want != ptx {
            moved.push(name);
        }
        checked += 1;
    }
    assert!(moved.is_empty(), "f32 PTX changed for {moved:?}");
    if !bless {
        assert!(checked >= 40, "only {checked} kernels checked");
    }
}
