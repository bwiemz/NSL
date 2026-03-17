/// Compute importance and load CV-squared losses for MoE load balancing.
pub fn compute_aux_losses(
    expert_weights: &[f32],
    expert_indices: &[i32],
    total_tokens: usize,
    num_experts: usize,
    top_k: usize,
) -> (f32, f32) {
    let mut importance = vec![0.0f32; num_experts];
    let mut load = vec![0.0f32; num_experts];
    for i in 0..(total_tokens * top_k) {
        let expert = expert_indices[i] as usize;
        if expert < num_experts {
            importance[expert] += expert_weights[i];
            load[expert] += 1.0;
        }
    }
    let importance_loss = cv_squared(&importance);
    let load_loss = cv_squared(&load);
    (importance_loss, load_loss)
}

/// Coefficient of variation squared: (std / mean)^2
fn cv_squared(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / n;
    if mean.abs() < 1e-10 {
        return 0.0;
    }
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    variance / (mean * mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_routing_zero_loss() {
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let (imp, load) = compute_aux_losses(&weights, &indices, 4, 4, 1);
        assert!(imp < 1e-6, "got {}", imp);
        assert!(load < 1e-6, "got {}", load);
    }

    #[test]
    fn test_imbalanced_routing_positive_loss() {
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 0, 0, 0];
        let (imp, load) = compute_aux_losses(&weights, &indices, 4, 4, 1);
        assert!(imp > 0.1, "got {}", imp);
        assert!(load > 0.1, "got {}", load);
    }

    #[test]
    fn test_cv_squared_uniform() {
        let vals = vec![2.0, 2.0, 2.0, 2.0];
        assert!(cv_squared(&vals) < 1e-10);
    }

    #[test]
    fn test_cv_squared_varied() {
        let vals = vec![1.0, 3.0, 1.0, 3.0];
        let cv2 = cv_squared(&vals);
        assert!(
            (cv2 - 0.25).abs() < 1e-5,
            "cv_squared = {}, expected 0.25",
            cv2
        );
    }
}
