//! Pin the fused elementwise-chain PTX emitter's load-bearing properties:
//! `.rn` on every arithmetic instruction (forbids ptxas FMA contraction —
//! the bit-exactness contract vs the standalone kernels), no `mad.lo.u32`
//! at ISA 7.0, sm_80 floor, ASCII-only, NUL-terminated, and the
//! `(out, in0.., n)` param order the runtime launcher marshals against.

use nsl_codegen::ew_chain_fusion::{ChainSig, ChainStep, EwOpcode, Operand};
use nsl_codegen::fusion::synthesize_fused_chain_ptx;

fn canonical_sig() -> ChainSig {
    // mul(i0,i1); rts(p0,i2); add(p1,x3f000000); neg(p2); div(p3,i3)
    ChainSig {
        n_inputs: 4,
        steps: vec![
            ChainStep {
                op: EwOpcode::Mul,
                lhs: Operand::Input(0),
                rhs: Some(Operand::Input(1)),
            },
            ChainStep {
                op: EwOpcode::RtsCheck,
                lhs: Operand::Prev(0),
                rhs: Some(Operand::Input(2)),
            },
            ChainStep {
                op: EwOpcode::Add,
                lhs: Operand::Prev(1),
                rhs: Some(Operand::Imm(0x3f00_0000)),
            },
            ChainStep {
                op: EwOpcode::Neg,
                lhs: Operand::Prev(2),
                rhs: None,
            },
            ChainStep {
                op: EwOpcode::Div,
                lhs: Operand::Prev(3),
                rhs: Some(Operand::Input(3)),
            },
        ],
    }
}

fn emit(sig: &ChainSig) -> (Vec<u8>, String) {
    let kname = sig.kernel_name();
    let bytes = synthesize_fused_chain_ptx(sig, &kname, 80);
    let text = std::str::from_utf8(&bytes[..bytes.len() - 1])
        .expect("PTX body is valid UTF-8")
        .to_string();
    (bytes, text)
}

#[test]
fn arith_opcodes_match_the_baseline_kernels() {
    // add/sub/mul MUST carry an explicit .rn: the standalone kernels'
    // single bare ops cannot contract, and .rn is what forbids ptxas from
    // FMA-contracting the multi-op chain into different bits. Div is the
    // deliberate exception: nsl_div_f32 is div.approx.f32, so a Div step
    // must be div.approx.f32 — div.rn would DIVERGE from the baseline.
    let (_, text) = emit(&canonical_sig());
    for line in text.lines() {
        let t = line.trim();
        for op in ["add.", "sub.", "mul."] {
            if t.starts_with(op) && t.contains(".f32") {
                assert!(
                    t.starts_with(&format!("{op}rn.f32")),
                    "f32 arithmetic without .rn is FMA-contraction-eligible: {t}"
                );
            }
        }
        if t.starts_with("div.") {
            assert!(
                t.starts_with("div.approx.f32"),
                "Div must match nsl_div_f32's div.approx.f32: {t}"
            );
        }
    }
    // Anti-vacuity: the canonical sig exercises add, mul, and div.
    assert!(text.contains("mul.rn.f32"));
    assert!(text.contains("add.rn.f32"));
    assert!(text.contains("div.approx.f32"));
    assert!(text.contains("neg.f32"));
}

#[test]
fn no_mad_lo_at_isa_7_0() {
    let (_, text) = emit(&canonical_sig());
    assert!(
        !text.contains("mad.lo"),
        "mad.lo.u32 is invalid at PTX ISA 7.0 — emit mul.lo.u32 + add.u32"
    );
    assert!(text.contains("mul.lo.u32"));
    assert!(text.starts_with(".version 7.0\n"));
}

#[test]
fn sm_80_floor_holds() {
    let sig = canonical_sig();
    let bytes = synthesize_fused_chain_ptx(&sig, "k", 52);
    let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
    assert!(
        text.contains(".target sm_80"),
        "requested sm below the floor must clamp to sm_80"
    );
}

#[test]
fn ascii_only_and_nul_terminated() {
    let (bytes, text) = emit(&canonical_sig());
    assert_eq!(*bytes.last().unwrap(), 0, "cuModuleLoadData C-string contract");
    assert!(
        text.bytes().all(|b| b.is_ascii()),
        "non-ASCII anywhere in PTX = CUDA_ERROR_INVALID_PTX"
    );
}

#[test]
fn param_order_is_out_inputs_n() {
    let sig = canonical_sig();
    let (_, text) = emit(&sig);
    let out_pos = text.find("param_out").expect("param_out");
    let n_pos = text.find("param_n").expect("param_n");
    let mut last = out_pos;
    for i in 0..sig.n_inputs {
        let p = text
            .find(&format!("param_in{i}"))
            .unwrap_or_else(|| panic!("param_in{i} missing"));
        assert!(p > last, "param_in{i} out of order");
        last = p;
    }
    assert!(n_pos > last, "param_n must come last");
}

#[test]
fn rts_step_emits_no_instruction() {
    // Same chain with and without the RtsCheck step must differ only in
    // register numbering, never in instruction count.
    let with_rts = canonical_sig();
    // Same chain minus the rts step: every later Prev shifts down by one and
    // the like-ref input slot (i2) disappears, renumbering i3 -> i2.
    let shift = |o: Operand| match o {
        Operand::Prev(k) if k >= 1 => Operand::Prev(k - 1),
        Operand::Input(3) => Operand::Input(2),
        other => other,
    };
    let mut steps = with_rts.steps.clone();
    steps.remove(1);
    let without = ChainSig {
        n_inputs: 3,
        steps: steps
            .into_iter()
            .map(|mut s| {
                s.lhs = shift(s.lhs);
                s.rhs = s.rhs.map(shift);
                s
            })
            .collect(),
    };
    let count = |sig: &ChainSig| {
        let (_, text) = emit(sig);
        text.lines()
            .filter(|l| {
                let t = l.trim();
                ["add.", "sub.", "mul.", "div.", "neg."]
                    .iter()
                    .any(|p| t.starts_with(p) && t.contains(".f32"))
            })
            .count()
    };
    assert_eq!(count(&with_rts), count(&without));
}

#[test]
fn imm_renders_inline_not_loaded() {
    let (_, text) = emit(&canonical_sig());
    assert!(
        text.contains("0f3F000000"),
        "immediate must render as an inline 0fHEX operand"
    );
    // The imm must not consume an input slot: exactly n_inputs loads.
    let loads = text.matches("ld.global.f32").count();
    assert_eq!(loads, 4);
}
