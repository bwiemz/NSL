/// Compute partial softmax for a chunk: returns (max, sum_exp, weighted_values).
/// weighted_values[i] = exp(values[i] - max)
pub fn partial_softmax(values: &[f32]) -> (f32, f32, Vec<f32>) {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    (max_val, sum, exps)
}

/// Merge two partial softmax results using online correction.
///
/// Returns (new_max, new_sum, corrected_weights) where:
/// - corrected_weights contains both old (rescaled) and new weights
/// - new_sum = old_sum * correction + new_sum_local
pub fn merge_partial_softmax(
    old_max: f32,
    old_sum: f32,
    old_weights: &[f32],
    new_max: f32,
    new_sum: f32,
    new_weights: &[f32],
) -> (f32, f32, Vec<f32>) {
    let global_max = old_max.max(new_max);
    let old_correction = (old_max - global_max).exp();
    let new_correction = (new_max - global_max).exp();

    let corrected_old_sum = old_sum * old_correction;
    let corrected_new_sum = new_sum * new_correction;
    let total_sum = corrected_old_sum + corrected_new_sum;

    let mut corrected = Vec::with_capacity(old_weights.len() + new_weights.len());
    for &w in old_weights {
        corrected.push(w * old_correction);
    }
    for &w in new_weights {
        corrected.push(w * new_correction);
    }

    (global_max, total_sum, corrected)
}

/// Standard softmax for reference testing.
pub fn softmax(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_pass_matches_single_pass() {
        // Single-pass softmax over [1.0, 2.0, 3.0, 4.0]
        let full = vec![1.0f32, 2.0, 3.0, 4.0];
        let full_softmax = softmax(&full);

        // Two-pass: first [1.0, 2.0], then correct with [3.0, 4.0]
        let (max1, sum1, weighted1) = partial_softmax(&[1.0, 2.0]);
        let (max2, sum2, weighted2) = partial_softmax(&[3.0, 4.0]);

        let (_final_max, final_sum, corrected) =
            merge_partial_softmax(max1, sum1, &weighted1, max2, sum2, &weighted2);

        // Normalize
        let result: Vec<f32> = corrected.iter().map(|&w| w / final_sum).collect();

        for (a, b) in full_softmax.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_correction_with_larger_new_max() {
        // Old pass had small values, new pass has large values
        let (max1, sum1, w1) = partial_softmax(&[0.1, 0.2]);
        let (max2, _sum2, _w2) = partial_softmax(&[10.0, 20.0]);

        let (final_max, final_sum, _) =
            merge_partial_softmax(max1, sum1, &w1, max2, _sum2, &_w2);

        assert!((final_max - 20.0).abs() < 1e-5);
        assert!(final_sum > 0.0);
    }
}
