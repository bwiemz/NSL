/// Result of CPU-path top-k routing (owned Vecs, not raw pointers).
pub struct CpuRoutingResult {
    pub expert_indices: Vec<i32>,
    pub expert_weights: Vec<f32>,
    pub sorted_token_indices: Vec<i32>,
    pub expert_boundaries: Vec<i32>,
    pub total_assigned: i64,
    pub importance_loss: f32,
    pub load_loss: f32,
}

/// Route tokens to experts via top-k gating with softmax.
/// `logits`: flat [total_tokens, num_experts] row-major
pub fn route_topk(
    logits: &[f32],
    total_tokens: usize,
    num_experts: usize,
    top_k: usize,
    capacity_factor: f64,
) -> CpuRoutingResult {
    assert!(top_k == 1 || top_k == 2, "top_k must be 1 or 2");
    assert_eq!(logits.len(), total_tokens * num_experts);

    let capacity = ((total_tokens as f64 / num_experts as f64) * capacity_factor).ceil() as usize;

    let mut expert_indices = Vec::with_capacity(total_tokens * top_k);
    let mut expert_weights = Vec::with_capacity(total_tokens * top_k);

    for t in 0..total_tokens {
        let row = &logits[t * num_experts..(t + 1) * num_experts];
        // Stable softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected = &indexed[..top_k];
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        for &(expert_id, weight) in selected {
            expert_indices.push(expert_id as i32);
            expert_weights.push(weight / weight_sum);
        }
    }

    // Token sorting: group by expert
    let total_assignments = total_tokens * top_k;
    let mut expert_counts = vec![0usize; num_experts];
    let mut expert_boundaries = vec![0i32; num_experts + 1];

    let mut assignment_to_expert = Vec::with_capacity(total_assignments);
    for &idx in expert_indices.iter().take(total_assignments) {
        let expert = idx as usize;
        if expert_counts[expert] < capacity {
            expert_counts[expert] += 1;
            assignment_to_expert.push(Some(expert));
        } else {
            assignment_to_expert.push(None);
        }
    }

    let mut offset = 0i32;
    for e in 0..num_experts {
        expert_boundaries[e] = offset;
        offset += expert_counts[e] as i32;
    }
    expert_boundaries[num_experts] = offset;
    let total_assigned = offset as i64;

    let mut sorted_token_indices = vec![0i32; total_assigned as usize];
    let mut write_pos = vec![0usize; num_experts];
    for (i, assignment) in assignment_to_expert.iter().enumerate() {
        if let Some(expert) = *assignment {
            let pos = expert_boundaries[expert] as usize + write_pos[expert];
            sorted_token_indices[pos] = (i / top_k) as i32;
            write_pos[expert] += 1;
        }
    }

    let (importance_loss, load_loss) = super::aux_loss::compute_aux_losses(
        &expert_weights,
        &expert_indices,
        total_tokens,
        num_experts,
        top_k,
    );

    CpuRoutingResult {
        expert_indices,
        expert_weights,
        sorted_token_indices,
        expert_boundaries,
        total_assigned,
        importance_loss,
        load_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk1_routes_to_max() {
        let logits: Vec<f32> = vec![
            1.0, 2.0, 0.5, 3.0, 0.1, 0.2, 0.1, 0.2, 4.0, 0.5, 5.0, 0.3,
        ];
        let result = route_topk(&logits, 4, 3, 1, 2.0);
        assert_eq!(result.expert_indices, vec![1, 0, 2, 1]);
        assert_eq!(result.expert_weights.len(), 4);
        for w in &result.expert_weights {
            assert!(*w > 0.0 && *w <= 1.0);
        }
    }

    #[test]
    fn test_topk2_routes_to_top_two() {
        let logits: Vec<f32> = vec![
            1.0, 5.0, 3.0, 0.1, 4.0, 0.1, 0.2, 3.0, 0.1, 0.2, 0.3, 6.0,
        ];
        let result = route_topk(&logits, 3, 4, 2, 2.0);
        assert_eq!(result.expert_indices.len(), 6);
        assert_eq!(result.expert_indices[0], 1);
        assert_eq!(result.expert_indices[1], 2);
        let w0 = result.expert_weights[0] + result.expert_weights[1];
        assert!((w0 - 1.0).abs() < 1e-5);
    }
}
