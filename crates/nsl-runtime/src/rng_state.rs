//! Capturable RNG state for resumable training (item 8).
//!
//! Training draws randomness from two independent streams, and a resume that
//! restores θ/m/v/step but not these two computes a *different* trajectory
//! from the run it claims to continue:
//!
//! * the thread-local sampling RNG ([`crate::sampling`]) — CPU dropout masks,
//!   `randn`/`rand` initialization, multinomial sampling;
//! * the GPU dropout counter below — the per-call seed for the
//!   `dropout_f32` PTX kernel's hash PRNG.
//!
//! Both are snapshot as *small scalars* so the checkpoint sidecar can carry
//! them: the sampling RNG through ChaCha's word position (O(1), no replay),
//! the GPU counter directly.
//!
//! The GPU counter lives here rather than as a function-local `static` inside
//! `cuda::gpu_dropout_f32` for two reasons: a CPU-only build must still be
//! able to round-trip the field (a checkpoint written on the GPU box is
//! readable on a CPU box), and `nsl_rng_seed` must be able to seed it —
//! before this it started at a hard-coded 42 and `--seed` never reached GPU
//! dropout at all.

use std::sync::atomic::{AtomicU64, Ordering};

/// Historical starting value of the GPU dropout counter. Kept as the default
/// so a run without `--seed` behaves exactly as before.
pub const GPU_DROPOUT_SEED_DEFAULT: u64 = 42;

static GPU_DROPOUT_COUNTER: AtomicU64 = AtomicU64::new(GPU_DROPOUT_SEED_DEFAULT);

/// Claim `len` counter values for one dropout launch and return the base.
/// Each element of the launch derives its own hash from `base + i`, so the
/// stride must be the element count — overlapping windows would correlate
/// masks across calls.
pub fn gpu_dropout_next_seed(len: u64) -> u64 {
    GPU_DROPOUT_COUNTER.fetch_add(len, Ordering::SeqCst)
}

/// Current GPU dropout counter (the value the next launch would claim).
pub fn gpu_dropout_counter() -> u64 {
    GPU_DROPOUT_COUNTER.load(Ordering::SeqCst)
}

/// Restore the GPU dropout counter (checkpoint resume, or `--seed`).
pub fn set_gpu_dropout_counter(v: u64) {
    GPU_DROPOUT_COUNTER.store(v, Ordering::SeqCst);
}

/// Counter for the BF16 matmul operand cast's stochastic-rounding dither
/// (`NSL_MATMUL_BF16_ROUND=sr`).
///
/// Deliberately its OWN stream rather than a share of the dropout counter:
/// the two are claimed at different rates (one per dropout call vs two per
/// high-intensity GEMM), so folding them together would make either one's
/// dither depend on whether the model happens to use dropout.
static BF16_SR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Claim `len` dither values for one operand cast and return the base.
///
/// The BASE IS THE ONLY THING THAT DECORRELATES THE ROUNDING ACROSS STEPS,
/// and that is the entire point of the SR mode. Under round-to-nearest the
/// operand error is a standing perturbation — measured 0.98 self-correlation
/// after one optimizer step, because 98.7% of 1B weight elements move less
/// than half a bf16 ULP per step — so it biases the gradient instead of
/// averaging out of it. A monotonically claimed window gives every launch a
/// fresh dither, which is what turns that bias back into noise.
///
/// Claiming per launch (not per step) also means a GEMM that runs twice in
/// one step — the same weight in the forward and again in the dgrad — draws
/// two independent dithers. That is intentional: they are separate roundings
/// of the same value, and correlating them would reinstate exactly the
/// standing error this mode exists to break up.
pub fn bf16_sr_next_counter(len: u64) -> u64 {
    BF16_SR_COUNTER.fetch_add(len, Ordering::SeqCst)
}

/// Current BF16-SR dither counter (the value the next cast would claim).
pub fn bf16_sr_counter() -> u64 {
    BF16_SR_COUNTER.load(Ordering::SeqCst)
}

/// Restore the BF16-SR dither counter.
///
/// NOT yet carried in the checkpoint sidecar: unlike the dropout counter,
/// this one selects a ROUNDING, not a mask, so a resume that restarts it at
/// zero computes a statistically equivalent trajectory rather than a wrong
/// one. Persisting it is a follow-up, and until then a resumed SR run is
/// reproducible only from its own start.
pub fn set_bf16_sr_counter(v: u64) {
    BF16_SR_COUNTER.store(v, Ordering::SeqCst);
}

/// Serializes the tests that assert on the PROCESS-GLOBAL RNG state.
///
/// Before item 8 the GPU dropout counter was a function-local `static` inside
/// a cuda-gated fn, unreachable from any test. Now three tests in the same
/// lib-test binary write it — this module's counter test, `RngSnapshot::
/// restore`, and `nsl_rng_seed` — and cargo runs them on parallel threads.
/// Without this lock the absolute-value assertions would fail a few times a
/// year as an unexplained CI red, which is worse than a consistent failure.
#[cfg(test)]
pub(crate) static GLOBAL_RNG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A point-in-time snapshot of every training RNG stream.
///
/// `sampling_seed` is the ChaCha seed **as bytes** rather than the `u64` that
/// was passed to `--seed`: restoring must reproduce the exact stream, and
/// `seed_from_u64` is only one of the ways the generator can be keyed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RngSnapshot {
    pub sampling_seed: [u8; 32],
    /// ChaCha word position — 68 bits, so it does not fit a u64. Carried as
    /// a `u128` here and as a hi/lo pair in the checkpoint header.
    pub sampling_pos: u128,
    pub gpu_dropout_ctr: u64,
}

impl RngSnapshot {
    /// Capture every stream. Must run on the training thread: the sampling
    /// RNG is thread-local, and the value that matters is the one the step
    /// body has been drawing from.
    pub fn capture() -> Self {
        let (sampling_seed, sampling_pos) = crate::sampling::rng_snapshot();
        RngSnapshot {
            sampling_seed,
            sampling_pos,
            gpu_dropout_ctr: gpu_dropout_counter(),
        }
    }

    /// Restore every stream. Same thread requirement as [`Self::capture`].
    pub fn restore(&self) {
        crate::sampling::rng_restore(self.sampling_seed, self.sampling_pos);
        set_gpu_dropout_counter(self.gpu_dropout_ctr);
    }

    /// Lowercase hex of the ChaCha seed, for the checkpoint header.
    pub fn seed_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.sampling_seed {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Inverse of [`Self::seed_hex`]. `None` on any malformed input — the
    /// caller refuses rather than resuming with a half-parsed seed.
    pub fn seed_from_hex(hex: &[u8]) -> Option<[u8; 32]> {
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, chunk) in hex.chunks_exact(2).enumerate() {
            let s = std::str::from_utf8(chunk).ok()?;
            out[i] = u8::from_str_radix(s, 16).ok()?;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_hex_round_trips() {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        let snap = RngSnapshot {
            sampling_seed: seed,
            sampling_pos: 0,
            gpu_dropout_ctr: 0,
        };
        let hex = snap.seed_hex();
        assert_eq!(hex.len(), 64);
        assert_eq!(RngSnapshot::seed_from_hex(hex.as_bytes()), Some(seed));
    }

    #[test]
    fn malformed_seed_hex_is_refused_not_guessed() {
        // Short, long, and non-hex all return None: a partially-parsed seed
        // would restore a DIFFERENT stream while looking successful.
        assert_eq!(RngSnapshot::seed_from_hex(b"00"), None);
        assert_eq!(RngSnapshot::seed_from_hex(&[b'0'; 65]), None);
        assert_eq!(RngSnapshot::seed_from_hex(&[b'z'; 64]), None);
    }

    /// The contract the checkpoint depends on: capture mid-stream, keep
    /// drawing, restore, and the SAME values come back. A snapshot that
    /// recorded only the seed would restart the stream — the resumed run
    /// would re-use dropout masks the interrupted run already consumed.
    #[test]
    fn rng_snapshot_restores_the_stream() {
        let _guard = GLOBAL_RNG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::sampling::nsl_manual_seed(1234);
        // Advance to a non-trivial position (and across a ChaCha block
        // boundary — 16 words per block — so the word-position restore is
        // actually exercised rather than trivially landing at 0).
        for _ in 0..37 {
            let _ = crate::sampling::rng_f64();
        }

        let snap = RngSnapshot::capture();
        let expected: Vec<f64> = (0..20).map(|_| crate::sampling::rng_f64()).collect();

        // Divergence: reseed to something else entirely, then restore.
        crate::sampling::nsl_manual_seed(9999);
        for _ in 0..5 {
            let _ = crate::sampling::rng_f64();
        }
        snap.restore();

        let replayed: Vec<f64> = (0..20).map(|_| crate::sampling::rng_f64()).collect();
        assert_eq!(
            expected, replayed,
            "restoring a snapshot must reproduce the exact continuation"
        );
        assert_eq!(RngSnapshot::capture().sampling_pos, snap.sampling_pos + 20 * 2,
            "20 f64 draws consume 40 32-bit words — if this drifts, the \
             position arithmetic no longer describes the stream");
    }

    /// The seed alone is NOT the state. Pinned because "just store the seed"
    /// is the natural-looking shortcut, and it silently rewinds the stream.
    #[test]
    fn seed_alone_does_not_reproduce_a_mid_stream_position() {
        let _guard = GLOBAL_RNG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::sampling::nsl_manual_seed(77);
        for _ in 0..10 {
            let _ = crate::sampling::rng_f64();
        }
        let mid = crate::sampling::rng_f64();
        crate::sampling::nsl_manual_seed(77);
        assert_ne!(
            mid,
            crate::sampling::rng_f64(),
            "if these matched, the stream would not be advancing at all"
        );
    }

    #[test]
    fn gpu_dropout_counter_advances_by_launch_length() {
        let _guard = GLOBAL_RNG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_gpu_dropout_counter(1000);
        assert_eq!(gpu_dropout_next_seed(64), 1000);
        assert_eq!(gpu_dropout_counter(), 1064);
        // Restore the default so a later test in this process is not keyed
        // off this one's leftovers.
        set_gpu_dropout_counter(GPU_DROPOUT_SEED_DEFAULT);
    }
}
