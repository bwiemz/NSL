use super::types::VerifyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Greedy rejection sampling (temperature=0).
/// Accepts draft token at position i iff verifier's argmax equals draft token.
/// On rejection, returns verifier's argmax as replacement.
/// If all K accepted, samples bonus from position K.
///
/// `verifier_logits`: flat [K+1, vocab_size] row-major
/// `draft_tokens`: [K] draft token IDs
pub fn rejection_sample_greedy(
    verifier_logits: &[f32],
    draft_tokens: &[i64],
    vocab_size: usize,
) -> VerifyResult {
    let k = draft_tokens.len();
    assert_eq!(verifier_logits.len(), (k + 1) * vocab_size);
    let mut accepted_tokens = Vec::with_capacity(k + 1);

    for i in 0..k {
        let row = &verifier_logits[i * vocab_size..(i + 1) * vocab_size];
        let verifier_argmax = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;

        if verifier_argmax as i64 != draft_tokens[i] {
            accepted_tokens.push(verifier_argmax as i64);
            return VerifyResult {
                num_accepted: i,
                accepted_tokens,
                has_bonus: false,
            };
        }
        accepted_tokens.push(draft_tokens[i]);
    }

    // All K accepted — bonus from position K
    let bonus_row = &verifier_logits[k * vocab_size..(k + 1) * vocab_size];
    let bonus = bonus_row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    accepted_tokens.push(bonus as i64);
    VerifyResult {
        num_accepted: k,
        accepted_tokens,
        has_bonus: true,
    }
}

/// Stochastic rejection sampling (temperature > 0).
/// Accepts with probability min(1, p_verifier / p_draft).
/// On rejection, samples from adjusted distribution max(0, p_v - p_d) / Z.
pub fn rejection_sample_stochastic(
    verifier_logits: &[f32],
    draft_logits: &[f32],
    draft_tokens: &[i64],
    vocab_size: usize,
    temperature: f32,
    seed: u64,
) -> VerifyResult {
    let k = draft_tokens.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut accepted_tokens = Vec::with_capacity(k + 1);

    for i in 0..k {
        let t = draft_tokens[i] as usize;
        let v_row = &verifier_logits[i * vocab_size..(i + 1) * vocab_size];
        let d_row = &draft_logits[i * vocab_size..(i + 1) * vocab_size];

        let v_probs = softmax_with_temperature(v_row, temperature);
        let d_probs = softmax_with_temperature(d_row, temperature);

        let p_v = v_probs[t];
        let p_d = d_probs[t].max(1e-10);
        let accept_prob = (p_v / p_d).min(1.0);

        if rng.gen::<f32>() >= accept_prob {
            let token = sample_adjusted_distribution(&v_probs, &d_probs, &mut rng);
            accepted_tokens.push(token as i64);
            return VerifyResult {
                num_accepted: i,
                accepted_tokens,
                has_bonus: false,
            };
        }
        accepted_tokens.push(draft_tokens[i]);
    }

    let bonus_row = &verifier_logits[k * vocab_size..(k + 1) * vocab_size];
    let bonus_probs = softmax_with_temperature(bonus_row, temperature);
    let bonus_token = sample_from_probs(&bonus_probs, &mut rng);
    accepted_tokens.push(bonus_token as i64);
    VerifyResult {
        num_accepted: k,
        accepted_tokens,
        has_bonus: true,
    }
}

/// Main entry point — dispatches by temperature.
pub fn rejection_sample(
    verifier_logits: &[f32],
    draft_logits: &[f32],
    draft_tokens: &[i64],
    vocab_size: usize,
    temperature: f32,
    seed: u64,
) -> VerifyResult {
    if temperature == 0.0 {
        rejection_sample_greedy(verifier_logits, draft_tokens, vocab_size)
    } else {
        rejection_sample_stochastic(
            verifier_logits,
            draft_logits,
            draft_tokens,
            vocab_size,
            temperature,
            seed,
        )
    }
}

fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // CRITICAL-3 fix: guard against all-masked (all -inf) or NaN logits
    if !max_val.is_finite() {
        let n = logits.len();
        return if n > 0 { vec![1.0 / n as f32; n] } else { vec![] };
    }
    let exps: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    // Guard against sum=0 (all exps underflowed)
    if sum == 0.0 || !sum.is_finite() {
        let n = logits.len();
        return if n > 0 { vec![1.0 / n as f32; n] } else { vec![] };
    }
    exps.iter().map(|&e| e / sum).collect()
}

fn sample_adjusted_distribution(v_probs: &[f32], d_probs: &[f32], rng: &mut StdRng) -> usize {
    let adjusted: Vec<f32> = v_probs
        .iter()
        .zip(d_probs.iter())
        .map(|(&v, &d)| (v - d).max(0.0))
        .collect();
    let sum: f32 = adjusted.iter().sum();
    if sum < 1e-10 {
        return sample_from_probs(v_probs, rng);
    }
    let normalized: Vec<f32> = adjusted.iter().map(|&a| a / sum).collect();
    sample_from_probs(&normalized, rng)
}

fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_all_accepted() {
        let vocab_size = 4;
        let draft_tokens = vec![2i64, 0, 3];
        let verifier_logits: Vec<f32> = vec![
            0.1, 0.2, 5.0, 0.3, // argmax=2
            5.0, 0.1, 0.2, 0.3, // argmax=0
            0.1, 0.2, 0.3, 5.0, // argmax=3
            0.1, 5.0, 0.3, 0.2, // bonus: argmax=1
        ];
        let result = rejection_sample_greedy(&verifier_logits, &draft_tokens, vocab_size);
        assert_eq!(result.num_accepted, 3);
        assert!(result.has_bonus);
        assert_eq!(result.accepted_tokens, vec![2, 0, 3, 1]);
    }

    #[test]
    fn test_greedy_first_rejected() {
        let vocab_size = 4;
        let draft_tokens = vec![2i64, 0, 3];
        let verifier_logits: Vec<f32> = vec![
            0.1, 5.0, 0.2, 0.3, // argmax=1, draft=2 -> REJECT
            5.0, 0.1, 0.2, 0.3, //
            0.1, 0.2, 0.3, 5.0, //
            0.1, 5.0, 0.3, 0.2, //
        ];
        let result = rejection_sample_greedy(&verifier_logits, &draft_tokens, vocab_size);
        assert_eq!(result.num_accepted, 0);
        assert!(!result.has_bonus);
        assert_eq!(result.accepted_tokens, vec![1]);
    }

    #[test]
    fn test_greedy_middle_rejected() {
        let vocab_size = 4;
        let draft_tokens = vec![2i64, 0, 3];
        let verifier_logits: Vec<f32> = vec![
            0.1, 0.2, 5.0, 0.3, // argmax=2
            0.1, 5.0, 0.2, 0.3, // argmax=1, draft=0 -> REJECT
            0.1, 0.2, 0.3, 5.0, //
            0.1, 5.0, 0.3, 0.2, //
        ];
        let result = rejection_sample_greedy(&verifier_logits, &draft_tokens, vocab_size);
        assert_eq!(result.num_accepted, 1);
        assert!(!result.has_bonus);
        assert_eq!(result.accepted_tokens, vec![2, 1]);
    }

    #[test]
    fn test_stochastic_high_acceptance() {
        let vocab_size = 4;
        let draft_tokens = vec![2i64];
        let draft_logits: Vec<f32> = vec![-0.1, -5.0, -0.01, -5.0];
        let verifier_logits: Vec<f32> = vec![
            -0.1, -5.0, -0.01, -5.0, //
            -0.1, -5.0, -0.01, -5.0, //
        ];
        let mut accept_count = 0;
        for seed in 0..100 {
            let result = rejection_sample_stochastic(
                &verifier_logits,
                &draft_logits,
                &draft_tokens,
                vocab_size,
                1.0,
                seed,
            );
            if result.num_accepted == 1 {
                accept_count += 1;
            }
        }
        assert!(
            accept_count > 80,
            "Expected >80% acceptance, got {}/100",
            accept_count
        );
    }

    #[test]
    fn test_stochastic_low_acceptance() {
        let vocab_size = 4;
        let draft_tokens = vec![0i64];
        let draft_logits: Vec<f32> = vec![-0.01, -5.0, -5.0, -5.0];
        let verifier_logits: Vec<f32> = vec![
            -5.0, -5.0, -0.01, -5.0, //
            -5.0, -5.0, -0.01, -5.0, //
        ];
        let mut accept_count = 0;
        for seed in 0..100 {
            let result = rejection_sample_stochastic(
                &verifier_logits,
                &draft_logits,
                &draft_tokens,
                vocab_size,
                1.0,
                seed,
            );
            if result.num_accepted == 1 {
                accept_count += 1;
            }
        }
        assert!(
            accept_count < 20,
            "Expected <20% acceptance, got {}/100",
            accept_count
        );
    }

    #[test]
    fn test_unified_dispatch_greedy() {
        let vocab_size = 3;
        let draft_tokens = vec![1i64];
        let verifier_logits: Vec<f32> = vec![0.1, 5.0, 0.2, 0.1, 0.2, 5.0];
        let draft_logits: Vec<f32> = vec![0.0; 3];
        let result = rejection_sample(
            &verifier_logits,
            &draft_logits,
            &draft_tokens,
            vocab_size,
            0.0,
            0,
        );
        assert_eq!(result.num_accepted, 1);
        assert!(result.has_bonus);
    }
}
