/// Result of CPU-path top-k routing (owned Vecs, not raw pointers).
pub struct CpuRoutingResult {
    pub expert_indices: Vec<i32>,
    pub expert_weights: Vec<f32>,
    pub sorted_token_indices: Vec<i32>,
    /// Per-sorted-position routing weight. Length == `total_assigned`,
    /// aligned with `sorted_token_indices`. For each surviving assignment
    /// `i`, this is the (re-normalized) softmax probability the routed
    /// token gave the expert it was placed under. Downstream consumers
    /// pass this directly to `dispatch::gather_tokens` for the
    /// gating-weight broadcast (sum-of-weights == 1.0 per token when no
    /// capacity drop occurs).
    pub sorted_assignment_weights: Vec<f32>,
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
        // IMPORTANT-4 fix: guard against sum=0 (all exps underflowed)
        let probs: Vec<f32> = if sum > 0.0 && sum.is_finite() {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / num_experts as f32; num_experts] // uniform fallback
        };
        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected = &indexed[..top_k];
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        let safe_weight_sum = if weight_sum > 1e-8 { weight_sum } else { 1.0 };
        for &(expert_id, weight) in selected {
            expert_indices.push(expert_id as i32);
            expert_weights.push(weight / safe_weight_sum);
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
    let mut sorted_assignment_weights = vec![0.0f32; total_assigned as usize];
    let mut write_pos = vec![0usize; num_experts];
    for (i, assignment) in assignment_to_expert.iter().enumerate() {
        if let Some(expert) = *assignment {
            let pos = expert_boundaries[expert] as usize + write_pos[expert];
            sorted_token_indices[pos] = (i / top_k) as i32;
            // expert_weights is indexed by the original assignment index `i`
            // (length total_tokens * top_k). Re-aligning by sorted position
            // here lets the gather pipeline broadcast gating weights
            // without re-deriving (token, k-slot) -> assignment-index.
            sorted_assignment_weights[pos] = expert_weights[i];
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
        sorted_assignment_weights,
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

    #[test]
    fn test_aux_loss_gradient_direction() {
        // Imbalanced routing should have higher aux_loss than balanced
        let logits_imbalanced: Vec<f32> = vec![
            10.0, 0.0,
            9.0, 0.0,
            8.0, 0.1,
            7.0, 0.1,
        ];
        let result_imb = route_topk(&logits_imbalanced, 4, 2, 1, 2.0);
        let coeff = 0.01f32;
        let aux_loss_imb = coeff * (result_imb.importance_loss + result_imb.load_loss);

        let logits_balanced: Vec<f32> = vec![
            5.0, 0.0,
            0.0, 5.0,
            5.0, 0.0,
            0.0, 5.0,
        ];
        let result_bal = route_topk(&logits_balanced, 4, 2, 1, 2.0);
        let aux_loss_bal = coeff * (result_bal.importance_loss + result_bal.load_loss);

        assert!(aux_loss_imb > aux_loss_bal,
            "imbalanced loss ({}) should be greater than balanced loss ({})",
            aux_loss_imb, aux_loss_bal);
    }

    #[test]
    fn sorted_assignment_weights_topk1_equals_per_token_softmax() {
        // top_k=1: each token contributes exactly one assignment with weight 1.0
        // (single-element softmax always normalizes to 1.0 inside route_topk).
        let logits: Vec<f32> = vec![
            1.0, 2.0, 0.5,
            3.0, 0.1, 0.2,
            0.1, 0.2, 4.0,
            0.5, 5.0, 0.3,
        ];
        let result = route_topk(&logits, 4, 3, 1, 2.0);
        assert_eq!(result.sorted_assignment_weights.len(), result.total_assigned as usize);
        for &w in &result.sorted_assignment_weights {
            assert!((w - 1.0).abs() < 1e-5, "top_k=1 weight should re-normalize to 1.0, got {}", w);
        }
    }

    #[test]
    fn sorted_assignment_weights_topk2_sum_per_token_equals_one() {
        // top_k=2: route_topk re-normalizes so the surviving pair sums to 1.0
        // per token. After sorting by expert, summing the weights for both
        // assignments of the same original token must still equal 1.0.
        let logits: Vec<f32> = vec![
            1.0, 5.0, 3.0, 0.1,
            4.0, 0.1, 0.2, 3.0,
            0.1, 0.2, 0.3, 6.0,
        ];
        let total_tokens = 3;
        let result = route_topk(&logits, total_tokens, 4, 2, 2.0);
        // For each token, accumulate the routing weight from every sorted
        // position that came from that token. With no capacity drop the sum
        // must be 1.0 (within softmax precision).
        let mut per_token_sums = vec![0.0f32; total_tokens];
        for (pos, &tok) in result.sorted_token_indices.iter().enumerate() {
            per_token_sums[tok as usize] += result.sorted_assignment_weights[pos];
        }
        for (t, &s) in per_token_sums.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-5, "token {} weights should sum to 1.0, got {}", t, s);
        }
    }

    #[test]
    fn sorted_assignment_weights_aligned_with_sorted_token_indices() {
        // Cross-check: sorted_assignment_weights[pos] must equal
        // expert_weights[original_assignment_index] where the original index
        // is t * top_k + k (k being which top-k slot landed token t in
        // sorted_token_indices[pos]'s expert). Verifies the i/top_k mapping
        // is consistent.
        let logits: Vec<f32> = vec![
            1.0, 5.0, 3.0, 0.1,
            4.0, 0.1, 0.2, 3.0,
            0.1, 0.2, 0.3, 6.0,
        ];
        let result = route_topk(&logits, 3, 4, 2, 2.0);
        for (pos, &tok) in result.sorted_token_indices.iter().enumerate() {
            let t = tok as usize;
            let sorted_w = result.sorted_assignment_weights[pos];
            // The two candidate weights for this token were stored at
            // expert_weights[t * 2 + 0..2]. Sorted-weight must match one of
            // them (we don't know which slot survived if both went to
            // different experts that may overflow capacity differently).
            let w0 = result.expert_weights[t * 2];
            let w1 = result.expert_weights[t * 2 + 1];
            assert!(
                (sorted_w - w0).abs() < 1e-6 || (sorted_w - w1).abs() < 1e-6,
                "sorted weight {} at pos {} (tok {}) must match one of original weights ({}, {})",
                sorted_w, pos, t, w0, w1
            );
        }
    }

    #[test]
    fn test_capacity_overflow_drops_tokens() {
        let logits: Vec<f32> = vec![
            5.0, 0.1,
            4.0, 0.1,
            3.0, 0.1,
            2.0, 0.1,
            1.0, 0.1,
            0.5, 0.1,
        ];
        let result = route_topk(&logits, 6, 2, 1, 1.0);
        let exp0_count = result.expert_boundaries[1] - result.expert_boundaries[0];
        assert_eq!(exp0_count, 3, "expert 0 should have capacity-limited 3 tokens, got {}", exp0_count);
        assert_eq!(result.total_assigned, 3, "total_assigned should be 3 (dropped 3)");
    }
}
