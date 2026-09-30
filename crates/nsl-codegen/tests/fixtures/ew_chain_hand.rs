// Frozen fixture (new-roadmap item 5): the hand-written PTX emitter for the
// fused adjoint elementwise-chain kernel (MFU campaign C3), exactly as
// `crates/nsl-codegen/src/fusion.rs` held it at 1d3c5498, before the kernel
// moved onto KIR (`crate::ew_chain_ptx`). Lines below this comment are that
// file's lines 393 to 542, unedited; the `ew_chain_kir_equivalence` gate
// includes it with `#[path]` (supplying the `crate::ew_chain_fusion` path it
// names) and runs its output against the KIR kernel's. Do not change it: it
// is the pre-migration behaviour the equivalence claim is about.
//
// Not scanned by the hand-PTX freeze: nothing under `tests/` is.

/// Synthesize PTX for one fused adjoint elementwise chain (MFU campaign C3).
///
/// Descriptor-driven variant of [`synthesize_fused_ptx_sm`]: operands are
/// addressed explicitly (Input slot / previous step / f32 immediate) instead
/// of consumed positionally, and — load-bearing — add/sub/mul carry an
/// explicit `.rn`. The chain replaces standalone `nsl_add_f32`-class kernels
/// whose single bare ops cannot contract (no in-kernel partner) and so are
/// correctly-rounded f32; a bare `add.f32`/`mul.f32` inside a multi-op
/// kernel IS FMA-contraction-eligible under ptxas, which would break the
/// bit-exactness contract (the `TANH_BACKWARD_SRCAD` lesson). Div is the
/// exception: the standalone `nsl_div_f32` kernel uses `div.approx.f32`, so
/// a Div step must emit `div.approx.f32` — `div.rn` would be a DIFFERENT
/// operation and diverge from the baseline chain (follow-the-code, runtime
/// package finding). `neg.f32` is exact. `RtsCheck` steps emit
/// nothing (identity on the fast path — the runtime gate has already proven
/// the like-ref's shape equals the uniform shape). Immediates render inline
/// as `0fXXXXXXXX` operands: no register, no load, and the f64->f32
/// narrowing happened once at compile time exactly where the baseline's
/// `nsl_tensor_scalar(v, 1)` narrowed it.
///
/// Signature: `(param_out, param_in0..N-1, param_n: u64)`, flat contiguous
/// f32, 1-D grid, block decided by the launcher. `.target sm_{gpu_sm}`
/// (sm_80 floor at the call site); tid arithmetic uses `mul.lo.u32` +
/// `add.u32` (never `mad.lo.u32` at ISA 7.0); ASCII only; NUL-terminated.
pub fn synthesize_fused_chain_ptx(
    sig: &crate::ew_chain_fusion::ChainSig,
    kname: &str,
    gpu_sm: u32,
) -> Vec<u8> {
    use crate::ew_chain_fusion::{EwOpcode, Operand};

    let n_inputs = sig.n_inputs as usize;
    let n_steps = sig.steps.len();
    let mut ptx = String::new();

    ptx.push_str(".version 7.0\n");
    ptx.push_str(&format!(".target sm_{}\n", gpu_sm.max(80)));
    ptx.push_str(".address_size 64\n\n");

    ptx.push_str(&format!(".visible .entry {}(\n", kname));
    ptx.push_str("    .param .u64 param_out,\n");
    for i in 0..n_inputs {
        ptx.push_str(&format!("    .param .u64 param_in{},\n", i));
    }
    ptx.push_str("    .param .u64 param_n\n");
    ptx.push_str(") {\n");

    let f_regs = (n_inputs + n_steps + 4).max(16);
    let rd_regs = (4 + n_inputs + 2).max(16);
    ptx.push_str("    .reg .u32 %r<8>;\n");
    ptx.push_str(&format!("    .reg .u64 %rd<{}>;\n", rd_regs));
    ptx.push_str(&format!("    .reg .f32 %f<{}>;\n", f_regs));
    ptx.push_str("    .reg .pred %p<4>;\n\n");

    // Thread ID = blockIdx.x * blockDim.x + threadIdx.x (no mad.lo at 7.0)
    ptx.push_str("    mov.u32 %r0, %tid.x;\n");
    ptx.push_str("    mov.u32 %r1, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %r2, %ntid.x;\n");
    ptx.push_str("    mul.lo.u32 %r3, %r1, %r2;\n");
    ptx.push_str("    add.u32 %r0, %r0, %r3;\n");
    ptx.push_str("    cvt.u64.u32 %rd0, %r0;\n\n");

    ptx.push_str("    ld.param.u64 %rd1, [param_n];\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd0, %rd1;\n");
    ptx.push_str("    @%p0 bra $L_end;\n\n");

    ptx.push_str("    shl.b64 %rd2, %rd0, 2;\n\n");

    for i in 0..n_inputs {
        ptx.push_str(&format!(
            "    ld.param.u64 %rd{}, [param_in{}];\n",
            3 + i,
            i
        ));
        ptx.push_str(&format!("    add.u64 %rd{}, %rd{}, %rd2;\n", 3 + i, 3 + i));
        ptx.push_str(&format!("    ld.global.f32 %f{}, [%rd{}];\n", i, 3 + i));
    }
    ptx.push('\n');

    // Per-step result register (or aliased operand string for RtsCheck).
    let mut step_regs: Vec<String> = Vec::with_capacity(n_steps);
    let opnd = |o: Operand, step_regs: &[String]| -> String {
        match o {
            Operand::Input(i) => format!("%f{}", i),
            Operand::Prev(k) => step_regs[k as usize].clone(),
            Operand::Imm(bits) => format!("0f{:08X}", bits),
        }
    };
    for (k, step) in sig.steps.iter().enumerate() {
        let out = format!("%f{}", n_inputs + k);
        match step.op {
            EwOpcode::RtsCheck => {
                // Identity on the fast path: alias the flowing operand.
                let src = opnd(step.lhs, &step_regs);
                step_regs.push(src);
                continue;
            }
            EwOpcode::Neg => {
                let src = opnd(step.lhs, &step_regs);
                ptx.push_str(&format!("    neg.f32 {}, {};\n", out, src));
            }
            EwOpcode::Add | EwOpcode::Sub | EwOpcode::Mul | EwOpcode::Div => {
                let lhs = opnd(step.lhs, &step_regs);
                let rhs = opnd(
                    step.rhs.expect("binary chain step carries a rhs"),
                    &step_regs,
                );
                let ins = match step.op {
                    EwOpcode::Add => "add.rn.f32",
                    EwOpcode::Sub => "sub.rn.f32",
                    EwOpcode::Mul => "mul.rn.f32",
                    // Matches nsl_div_f32's div.approx.f32 — NOT div.rn (see
                    // the fn docs; bit-exactness is against the baseline
                    // kernel, and the baseline is approx).
                    EwOpcode::Div => "div.approx.f32",
                    _ => unreachable!(),
                };
                ptx.push_str(&format!("    {} {}, {}, {};\n", ins, out, lhs, rhs));
            }
        }
        step_regs.push(out);
    }

    let result = step_regs
        .last()
        .expect("chain has at least one step")
        .clone();
    ptx.push_str(&format!(
        "\n    ld.param.u64 %rd{}, [param_out];\n",
        3 + n_inputs
    ));
    ptx.push_str(&format!(
        "    add.u64 %rd{}, %rd{}, %rd2;\n",
        3 + n_inputs,
        3 + n_inputs
    ));
    ptx.push_str(&format!(
        "    st.global.f32 [%rd{}], {};\n",
        3 + n_inputs,
        result
    ));

    ptx.push_str("\n$L_end:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    bytes.push(0); // cuModuleLoadData C-string contract
    bytes
}
