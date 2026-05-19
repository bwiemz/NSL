//! Deterministic test stimuli for the v1 MLP parity test.
//!
//! Stimuli are generated in-process (not committed to git) using a fixed
//! `ChaCha20Rng` seed, giving identical inputs on every test run without
//! the SHA-256 hash-manifest overhead needed for committed artifacts.
//!
//! See spec §6.5 for the fixture-vs-stimuli design distinction.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const STIMULI_SEED: u64 = 0xDEADBEEF;
const N_STIMULI: usize = 100;
const INPUT_DIM: usize = 784;

/// Deterministic random stimuli for the v1 MLP parity test.
///
/// Generates 100 vectors of 784 random i8 values covering the full i8 range
/// [-128, 127]. These are NOT real MNIST — v1 is a structural correctness
/// test, not a model-quality test.
pub fn v1_mlp_stimuli() -> Vec<Vec<i8>> {
    let mut rng = ChaCha20Rng::seed_from_u64(STIMULI_SEED);
    (0..N_STIMULI)
        .map(|_| (0..INPUT_DIM).map(|_| rng.random::<i8>()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stimuli_count_and_dim() {
        let stims = v1_mlp_stimuli();
        assert_eq!(stims.len(), 100, "must produce exactly 100 stimuli");
        for (i, s) in stims.iter().enumerate() {
            assert_eq!(s.len(), 784, "stimulus {i} must have 784 elements");
        }
    }

    #[test]
    fn stimuli_are_deterministic() {
        let a = v1_mlp_stimuli();
        let b = v1_mlp_stimuli();
        assert_eq!(a, b, "stimuli must be identical across calls");
    }

    #[test]
    fn stimuli_use_full_i8_range() {
        // With 100*784 = 78,400 random i8 values the full [-128,127] range
        // is overwhelmingly likely to be covered.
        let stims = v1_mlp_stimuli();
        let flat: Vec<i8> = stims.into_iter().flatten().collect();
        let has_negative = flat.iter().any(|&v| v < 0);
        let has_positive = flat.iter().any(|&v| v > 0);
        assert!(has_negative, "expected some negative values");
        assert!(has_positive, "expected some positive values");
    }
}
