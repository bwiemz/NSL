//! P4 item 17: GPU/CPU bit-parity for the SR-BF16 rounding tail.
//!
//! The PTX kernels (`FASE_FUSED_ADAMW_STEP_BF16SR_PTX`, probe) and the CPU
//! reference (`sr_bf16::sr_bf16_round_counter`) must agree BIT-FOR-BIT: the
//! dither hash is pure integer arithmetic and the rounding rule is pure bit
//! manipulation, so any divergence is a defect, not numerics. The probe
//! kernel duplicates the fused kernel's exact tail so this gate covers the
//! production rounding path without needing to reproduce div.approx-based
//! optimizer arithmetic on the host.

#![cfg(feature = "cuda")]

use nsl_runtime::sr_bf16::{sr_bf16_gpu_probe_host, sr_bf16_round_counter, sr_mix64, SR_PARAM_SHIFT};

/// Deterministic value stream for adversarial coverage — uses the module's
/// own mixer so the test needs no OS RNG (repo determinism doctrine).
fn value_stream(n: usize) -> Vec<f32> {
    let mut vals = Vec::with_capacity(n);
    // Hand-picked edge cases first.
    vals.extend_from_slice(&[
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.000_001,          // just above an exact bf16 value
        -3.999_9,
        f32::MAX,           // rounding-overflow saturation candidate
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xFF80_0001), // negative NaN payload
        f32::MIN_POSITIVE,           // smallest normal
        f32::from_bits(0x0000_0001), // smallest subnormal
        f32::from_bits(0x0000_5000), // deep subnormal
        1.5e-40,                     // bf16-subnormal range
        -2.7e38,                     // near the top binade
        3.389e38,                    // just under bf16 max normal
    ]);
    // Deterministic pseudo-random fill across magnitudes.
    let mut i = 0u64;
    while vals.len() < n {
        let bits = sr_mix64(0xC0FFEE, i) as u32;
        let v = f32::from_bits(bits);
        vals.push(v);
        i += 1;
    }
    vals.truncate(n);
    vals
}

/// GPU probe output == CPU reference, element for element, over an
/// adversarial value set and multiple (seed, step, param) counter tuples.
#[test]
#[ignore = "requires CUDA GPU"]
fn sr_bf16_gpu_tail_is_bit_identical_to_cpu_reference() {
    let vals = value_stream(65536);
    for (seed, step, pidx) in [(42u64, 1u64, 0u64), (42, 7, 3), (0xDEADBEEF, 123, 17)] {
        let ctr_base = pidx << SR_PARAM_SHIFT;
        let gpu = sr_bf16_gpu_probe_host(&vals, seed, step, ctr_base);
        assert_eq!(gpu.len(), vals.len());
        for (i, (&x, &g)) in vals.iter().zip(gpu.iter()).enumerate() {
            let cpu = sr_bf16_round_counter(x, seed, step, ctr_base + i as u64);
            assert_eq!(
                g, cpu,
                "SR divergence at elem {i} (x={x:?} bits={:#010x}): gpu={g:#06x} cpu={cpu:#06x} \
                 (seed={seed} step={step} pidx={pidx})",
                x.to_bits(),
            );
        }
    }
}

/// Changing any counter component changes the dither stream (the GPU side
/// must not accidentally ignore a kernel parameter).
#[test]
#[ignore = "requires CUDA GPU"]
fn sr_bf16_gpu_streams_vary_with_counters() {
    // A value with a large truncation remainder so dither differences show.
    let vals = vec![1.000_001_f32; 4096];
    let base = sr_bf16_gpu_probe_host(&vals, 42, 1, 0);
    let d_seed = sr_bf16_gpu_probe_host(&vals, 43, 1, 0);
    let d_step = sr_bf16_gpu_probe_host(&vals, 42, 2, 0);
    let d_parm = sr_bf16_gpu_probe_host(&vals, 42, 1, 1u64 << SR_PARAM_SHIFT);
    assert_ne!(base, d_seed, "seed must vary the SR stream");
    assert_ne!(base, d_step, "step must vary the SR stream");
    assert_ne!(base, d_parm, "param base must vary the SR stream");
    // And determinism: identical counters → identical output.
    let again = sr_bf16_gpu_probe_host(&vals, 42, 1, 0);
    assert_eq!(base, again, "identical counters must reproduce bit-identically");
}
