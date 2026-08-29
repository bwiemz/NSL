//! The BF16 matmul operand cast under `NSL_MATMUL_BF16_ROUND=sr`.
//!
//! ## What this gate is actually about
//!
//! The bf16 matmul mode's error is usually reported as an ENSEMBLE statistic —
//! "mean operand error 0.0003 ULP, unbiased" — and by that measure
//! round-to-nearest is fine. That measure is the wrong one for training, and
//! this gate exists to pin the property it misses.
//!
//! `round(W)` is a pure function of `W`. At 1B with lr=3e-5, 98.7% of weight
//! elements move LESS THAN HALF A BF16 ULP per optimizer step (measured by
//! differencing chain checkpoints 4000 steps apart), so the rounding error on
//! the weight operand is the SAME error step after step — 0.98 self-correlated
//! after one step. That is a standing bias in the function the forward and the
//! dgrad differentiate, and Adam's moment windows cannot average away a term
//! that does not change. It carries half the GEMM's squared error.
//!
//! So the property that matters is not accuracy per cast, it is DECORRELATION
//! ACROSS CASTS, plus unbiasedness so nothing is traded for it. Both are
//! asserted below against the RNE control, which must show the opposite.

#![cfg(feature = "cuda")]

use nsl_runtime::sr_bf16::{
    bf16_operand_cast_probe_host, bf16_operand_cast_rne_host, sr_step_key,
};

/// bf16 storage bits -> f32. Widening is exact (bf16 is f32's top 16 bits), so
/// this is a reinterpret, not a conversion. Defined here rather than widening
/// `tensor::bf16_bits_to_f32`'s crate-private visibility for a test's sake.
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Deterministic value stream in the magnitude band real 1B weights occupy
/// (median |w| ~ 1e-2), using the module's own mixer — no OS RNG.
fn weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = nsl_runtime::sr_bf16::sr_mix64(0x5EED_1234_ABCD_0001, i as u64);
            // Uniform in [-0.03, 0.03), the observed weight band.
            ((h >> 11) as f64 / (1u64 << 53) as f64 * 0.06 - 0.03) as f32
        })
        .collect()
}

/// Round-to-nearest repeats itself exactly; stochastic rounding does not.
///
/// This IS the defect and the fix, side by side. If the RNE arm ever stops
/// being bit-identical the mode has become non-deterministic; if the SR arm
/// ever becomes identical its dither has stopped advancing and it has
/// silently degraded into the standing error it exists to break up.
#[test]
fn rne_repeats_itself_and_sr_does_not() {
    let vals = weights(64 * 1024);
    let key = sr_step_key(777, 0);

    let rne_a = bf16_operand_cast_rne_host(&vals);
    let rne_b = bf16_operand_cast_rne_host(&vals);
    assert_eq!(
        rne_a, rne_b,
        "round-to-nearest must be a pure function of the input"
    );

    // Two casts of the SAME buffer at different counter windows — exactly what
    // consecutive optimizer steps do to a weight that has not moved.
    let sr_a = bf16_operand_cast_probe_host(&vals, key, 0);
    let sr_b = bf16_operand_cast_probe_host(&vals, key, vals.len() as u64);
    assert_ne!(sr_a, sr_b, "SR must draw a fresh dither per counter window");

    // Not just "not equal" — a large fraction must actually differ, or the
    // dither is advancing but barely biting. Only elements that sit strictly
    // between two bf16 grid points can differ at all, so measure against
    // those rather than the whole buffer.
    let inexact = vals
        .iter()
        .filter(|v| v.to_bits() & 0x0000_FFFF != 0)
        .count();
    let differing = sr_a
        .iter()
        .zip(&sr_b)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        inexact > vals.len() / 2,
        "test stream is degenerate: only {inexact} of {} values are inexact \
         in bf16, so this gate could pass vacuously",
        vals.len()
    );
    let frac = differing as f64 / inexact as f64;
    assert!(
        (0.2..0.8).contains(&frac),
        "SR re-rounded {:.1}% of the inexact values between two windows; \
         expected a substantial but not universal fraction (each element \
         flips only when its two dithers straddle its residual)",
        100.0 * frac
    );
}

/// SR is unbiased: averaging repeated casts of one buffer converges on the
/// original value, while RNE converges on its own fixed rounding.
///
/// This is what makes the decorrelation above safe to want. A dither that
/// decorrelated but skewed would trade a standing bias for a standing drift.
#[test]
fn averaging_sr_casts_converges_to_the_input_and_rne_does_not() {
    let n = 32 * 1024;
    let vals = weights(n);
    let key = sr_step_key(20260828, 0);

    let rounds = 64u64;
    let mut acc = vec![0.0f64; n];
    for r in 0..rounds {
        let bits = bf16_operand_cast_probe_host(&vals, key, r * n as u64);
        for (a, b) in acc.iter_mut().zip(&bits) {
            *a += bf16_bits_to_f32(*b) as f64;
        }
    }

    let rne = bf16_operand_cast_rne_host(&vals);
    let ulp = |v: f32| (v.abs().log2().floor() - 7.0).exp2() as f64;

    // Two different statistics, and the difference is the whole point.
    //
    // The SIGNED MEAN across elements answers "is this cast biased?" and is
    // small for BOTH modes -- round-to-nearest is unbiased over an ENSEMBLE,
    // which is exactly the measurement this module's doc comment calls the
    // wrong one. Asserting `rne_bias > sr_bias` would therefore be asserting
    // something false.
    //
    // The PER-ELEMENT RMS answers the question that matters: after averaging
    // `rounds` casts, how much error is LEFT ON EACH WEIGHT? SR's shrinks as
    // 1/sqrt(rounds) because each cast draws a fresh dither; RNE's does not
    // shrink at all, because every cast returns the identical value. That
    // standing per-element error is the gradient bias the mode exists to
    // break up, and it is invisible to the signed mean, which cancels it
    // across elements.
    let mut sr_bias = 0.0f64;
    let mut rne_bias = 0.0f64;
    let mut sr_sq = 0.0f64;
    let mut rne_sq = 0.0f64;
    for i in 0..n {
        let u = ulp(vals[i]).max(f64::MIN_POSITIVE);
        let sr_err = (acc[i] / rounds as f64 - vals[i] as f64) / u;
        let rne_err = (bf16_bits_to_f32(rne[i]) as f64 - vals[i] as f64) / u;
        sr_bias += sr_err;
        rne_bias += rne_err;
        sr_sq += sr_err * sr_err;
        rne_sq += rne_err * rne_err;
    }
    let sr_bias = (sr_bias / n as f64).abs();
    let rne_bias = (rne_bias / n as f64).abs();
    let sr_rms = (sr_sq / n as f64).sqrt();
    let rne_rms = (rne_sq / n as f64).sqrt();

    // With `rounds` independent dithers the SR mean sits within ~0.5/sqrt(64)
    // ULP of the input. RNE has no averaging to do -- its error is whatever
    // its single fixed rounding chose, and it is the same every step forever.
    assert!(
        sr_bias < 0.02,
        "mean of {rounds} SR casts is {sr_bias:.4} ULP off the input; SR must \
         be unbiased or it trades a standing bias for a standing drift"
    );

    // Anti-vacuity: the comparison is only meaningful if the per-cast error is
    // much larger than the averaged one -- i.e. averaging did real work.
    let per_cast: f64 = {
        let bits = bf16_operand_cast_probe_host(&vals, key, 999 * n as u64);
        let s: f64 = (0..n)
            .map(|i| {
                let u = ulp(vals[i]).max(f64::MIN_POSITIVE);
                ((bf16_bits_to_f32(bits[i]) as f64 - vals[i] as f64) / u).powi(2)
            })
            .sum();
        (s / n as f64).sqrt()
    };
    assert!(
        per_cast > 10.0 * sr_bias,
        "per-cast SR error ({per_cast:.4} ULP rms) is not meaningfully larger \
         than the {rounds}-cast mean ({sr_bias:.4} ULP) -- this gate would \
         pass even if averaging did nothing"
    );
    // The "and rne does not" half of this test's name, which until now was
    // computed and thrown away.
    //
    // For a residual uniform in [0,1) ULP: RNE's error is uniform in
    // [-0.5, 0.5] ULP, so its RMS is sqrt(1/12) = 0.289 and averaging cannot
    // move it. One SR cast has variance r(1-r), averaging to
    // sqrt(E[r(1-r)]/rounds) = sqrt(1/(6*64)) = 0.051. The expected ratio is
    // about 5.7; the threshold is 3 so ordinary sampling spread cannot trip
    // it, while a ratio near 1 -- which is what "SR stopped decorrelating"
    // or "the dither is a pure function of the value" would produce -- fails.
    assert!(
        rne_rms > 3.0 * sr_rms,
        "after {rounds} casts the per-element residual is {sr_rms:.4} ULP rms \
         under SR and {rne_rms:.4} ULP rms under RNE (ratio {:.2}, expected \
         ~5.7). RNE's error must SURVIVE averaging -- that standing \
         per-element term is the gradient bias SR exists to break up. A ratio \
         near 1 means the SR dither stopped advancing across casts.",
        rne_rms / sr_rms.max(f64::MIN_POSITIVE)
    );

    // The signed means are reported, not compared: both modes are unbiased
    // across the ensemble, so a comparison between them would be noise.
    assert!(
        rne_bias.is_finite(),
        "rne signed mean bias is not finite ({rne_bias})"
    );
}

/// The production cast must reach the LAST element of a large operand.
///
/// `gpu_cast_raw` caps its grid at 4096 blocks because its kernel carries a
/// grid-stride loop. The SR kernel does NOT: it is one element per thread, so
/// the same cap would leave everything past 4096*256 = 1,048,576 elements
/// holding whatever the recycled scratch block last contained. Real operands
/// are far larger than that -- a 2048x8192 FFN weight is 16.7M elements -- so
/// this gate runs above the cap on purpose.
#[test]
fn sr_cast_covers_operands_larger_than_the_grid_cap() {
    let n = (4096 * 256) + 4097; // one full cap, plus a partial block past it
    let vals = weights(n);
    let bits = bf16_operand_cast_probe_host(&vals, sr_step_key(5, 0), 0);
    assert_eq!(bits.len(), n);

    // Every element must land within one ULP of its input. An uncast tail
    // reads recycled scratch, which is overwhelmingly not that.
    let bad = (0..n)
        .filter(|&i| {
            let want = vals[i];
            let got = bf16_bits_to_f32(bits[i]);
            let ulp = (want.abs().log2().floor() - 7.0).exp2();
            !((got - want).abs() <= ulp)
        })
        .count();
    assert_eq!(
        bad, 0,
        "{bad} of {n} elements were not correctly rounded -- the tail past \
         the 1,048,576-element grid cap is the suspect"
    );
}
