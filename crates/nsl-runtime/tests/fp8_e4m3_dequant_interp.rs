#![cfg(all(feature = "cuda", feature = "test-hooks"))]

//! The GPU E4M3 dequant kernel (`nsl_dequant_fp8_e4m3_f32`, the KV-cache
//! FP8 read path) on the cooperative-CTA interpreter — no GPU needed.
//!
//! 1. All 256 codes decode to the OCP E4M3 value bit for bit: signed zeros,
//!    the subnormals m * 2^-9 (which the kernel used to decode as
//!    (1 + m/8) * 2^-7), the normals, and NaN for S.1111.111 (which it used to
//!    decode as ±480).
//! 2. It agrees with the CPU decoder on bytes the CPU encoder wrote, so a KV
//!    block quantized on the host reads back the same on either device.

mod common;
use common::fp8_reference::Fp8Format;

#[allow(dead_code)]
#[path = "../../nsl-codegen/tests/support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

use nsl_runtime::kv_compress::quantize::{dequantize_fp8, quantize_fp8};
use nsl_runtime::DEQUANT_FP8_E4M3_F32_PTX;
use std::collections::HashMap;

const INP: u64 = 0x10_0000;
const OUT: u64 = 0x20_0000;
const BLOCK: u32 = 256;
const POISON: u32 = 0xDEAD_BEEF;
const TAIL: usize = 5;

/// Runs the kernel over `codes` (launched as `gpu_dequant_fp8_e4m3_f32` does:
/// `ceil(n / 256)` blocks of 256, plus one spare block that must write
/// nothing) and returns the output words, including a poisoned tail.
fn run(codes: &[u8], order: Order) -> Vec<u32> {
    let ptx = DEQUANT_FP8_E4M3_F32_PTX.trim_end_matches('\0');
    let prog = parse(ptx);
    let n = codes.len();
    let mut global = vec![
        Segment { base: INP, bytes: codes.iter().copied().chain([0x38u8; TAIL]).collect() },
        Segment {
            base: OUT,
            bytes: std::iter::repeat_n(POISON.to_le_bytes(), n + TAIL).flatten().collect(),
        },
    ];
    let args: HashMap<String, u64> = [("inp", INP), ("out", OUT), ("n", n as u64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let grid = n.div_ceil(BLOCK as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_y: 1,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global[1].bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn reference_bits(code: u8) -> u32 {
    match Fp8Format::E4M3.decode(code) {
        Some(v) => (v as f32).to_bits(),
        None => f32::NAN.to_bits(),
    }
}

#[test]
fn every_e4m3_code_decodes_to_its_ocp_value() {
    let codes: Vec<u8> = (0..=255).collect();
    for order in [Order::Ascending, Order::Descending] {
        let out = run(&codes, order);
        for &code in &codes {
            let got = out[code as usize];
            let want = reference_bits(code);
            assert_eq!(
                got,
                want,
                "code {code:#04x}: kernel {} ({got:#010x}), OCP E4M3 {} ({want:#010x})",
                f32::from_bits(got),
                f32::from_bits(want)
            );
        }
        assert!(out[256..].iter().all(|&w| w == POISON), "wrote past n");
    }
}

#[test]
fn gpu_decode_matches_the_cpu_decoder_on_cpu_encoded_bytes() {
    // Magnitudes from 2^-12 (below the smallest subnormal) to 2^10 (past the
    // 448 saturation point), both signs.
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let values: Vec<f32> = (0..1000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let e = (s >> 40) as f32 / (1u64 << 24) as f32 * 22.0 - 12.0;
            let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
            sign * e.exp2()
        })
        .collect();
    let mut codes = vec![0u8; values.len()];
    quantize_fp8(&values, &mut codes);
    let mut cpu = vec![0.0f32; values.len()];
    dequantize_fp8(&codes, &mut cpu);
    let gpu = run(&codes, Order::Ascending);
    for (i, (&c, &g)) in cpu.iter().zip(&gpu).enumerate() {
        assert_eq!(g, c.to_bits(), "value {} (code {:#04x}): cpu {c}, gpu {}", values[i], codes[i], f32::from_bits(g));
    }
    assert!(codes.iter().any(|&c| c & 0x78 == 0 && c & 7 != 0), "no subnormal codes exercised");
}
