/// Result of CPU-path top-k routing (owned Vecs, not raw pointers).
pub struct CpuRoutingResult {
    pub expert_indices: Vec<i32>,
    pub expert_weights: Vec<f32>,
    pub sorted_token_indices: Vec<i32>,
    /// Per-sorted-position routing weight. Length == `total_assigned`,
    /// aligned with `sorted_token_indices`. For each surviving assignment
    /// `i`, this is the softmax probability the routed token gave the
    /// expert it was placed under, normalized over the token's SURVIVING
    /// assignments. Downstream consumers pass this directly to
    /// `dispatch::gather_tokens` for the gating-weight broadcast: the
    /// weights of every token that keeps at least one assignment sum to
    /// 1.0, whether or not a capacity drop took one of them.
    pub sorted_assignment_weights: Vec<f32>,
    pub expert_boundaries: Vec<i32>,
    pub total_assigned: i64,
    pub importance_loss: f32,
    pub load_loss: f32,
}

/// Route tokens to experts via top-k gating with softmax.
/// `logits`: flat [total_tokens, num_experts] row-major
///
/// An expert holds at most `ceil(total_tokens * top_k / num_experts *
/// capacity_factor)` assignments. When more tokens select it, it keeps the
/// ones that gave it the highest softmax probability (ties to the earlier
/// token) and the rest of those assignments are dropped; a token whose
/// other assignment survives has that one's weight renormalized to 1.0.
///
/// Which assignments are dropped used to be decided by token order: the
/// first `capacity` tokens to select an expert kept it, so the end of every
/// sequence lost its experts first whatever the router thought of it, and a
/// token left with one of its two experts kept that expert's partial weight,
/// scaling its output down by the weight of the dropped one.
pub fn route_topk(
    logits: &[f32],
    total_tokens: usize,
    num_experts: usize,
    top_k: usize,
    capacity_factor: f64,
) -> CpuRoutingResult {
    assert!(top_k == 1 || top_k == 2, "top_k must be 1 or 2");
    assert_eq!(logits.len(), total_tokens * num_experts);

    // Each token makes `top_k` assignments, so an expert's fair share is
    // `total_tokens * top_k / num_experts`; `capacity_factor` is the slack
    // on top of that. Without the `top_k` term a perfectly balanced top-2
    // router at factor 1.0 had room for half its assignments: every token
    // past the midpoint was dropped by both of its experts and came out
    // of the layer as zeros.
    let capacity = ((total_tokens as f64 * top_k as f64 / num_experts as f64) * capacity_factor)
        .ceil() as usize;

    let mut expert_indices = Vec::with_capacity(total_tokens * top_k);
    let mut expert_weights = Vec::with_capacity(total_tokens * top_k);
    // Softmax probability behind each assignment, before the top-k
    // renormalization: the router's confidence, used to rank the
    // assignments competing for an over-subscribed expert. (The
    // renormalized weight cannot rank them: under top-1 it is 1.0 for
    // every token.)
    let mut assignment_probs: Vec<f32> = Vec::with_capacity(total_tokens * top_k);

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
            assignment_probs.push(weight);
        }
    }

    // Token sorting: group by expert
    let total_assignments = total_tokens * top_k;
    let mut expert_counts = vec![0usize; num_experts];
    let mut expert_boundaries = vec![0i32; num_experts + 1];

    // Each expert keeps its `capacity` most confident assignments.
    let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
    for (i, &idx) in expert_indices.iter().enumerate().take(total_assignments) {
        by_expert[idx as usize].push(i);
    }
    let mut kept = vec![false; total_assignments];
    for (expert, assignments) in by_expert.iter_mut().enumerate() {
        if assignments.len() > capacity {
            assignments.sort_by(|&a, &b| {
                assignment_probs[b]
                    .total_cmp(&assignment_probs[a])
                    .then(a.cmp(&b))
            });
            assignments.truncate(capacity);
        }
        expert_counts[expert] = assignments.len();
        for &i in assignments.iter() {
            kept[i] = true;
        }
    }

    let mut offset = 0i32;
    for e in 0..num_experts {
        expert_boundaries[e] = offset;
        offset += expert_counts[e] as i32;
    }
    expert_boundaries[num_experts] = offset;
    let total_assigned = offset as i64;

    // What is left of each token's routing weight after the drops.
    let mut surviving_weight = vec![0.0f32; total_tokens];
    for (i, &k) in kept.iter().enumerate() {
        if k {
            surviving_weight[i / top_k] += expert_weights[i];
        }
    }

    let mut sorted_token_indices = vec![0i32; total_assigned as usize];
    let mut sorted_assignment_weights = vec![0.0f32; total_assigned as usize];
    let mut write_pos = vec![0usize; num_experts];
    // Token order within each expert's group.
    for (i, &k) in kept.iter().enumerate() {
        if !k {
            continue;
        }
        let expert = expert_indices[i] as usize;
        let token = i / top_k;
        let pos = expert_boundaries[expert] as usize + write_pos[expert];
        sorted_token_indices[pos] = token as i32;
        // expert_weights is indexed by the original assignment index `i`
        // (length total_tokens * top_k). Re-aligning by sorted position
        // here lets the gather pipeline broadcast gating weights
        // without re-deriving (token, k-slot) -> assignment-index.
        let survivors = surviving_weight[token];
        sorted_assignment_weights[pos] = if survivors > 1e-8 {
            expert_weights[i] / survivors
        } else {
            expert_weights[i]
        };
        write_pos[expert] += 1;
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

    /// Over capacity, an expert keeps the tokens that selected it most
    /// confidently, wherever they sit in the sequence.
    ///
    /// Four tokens all pick expert 0 under top-1; two experts at factor 1.0
    /// give it room for ceil(4 * 1 / 2) = 2. The probability each token
    /// gives expert 0 is the logistic of its logit gap (the other logit is
    /// 0): 1/(1+e^-1) = 0.7311, 1/(1+e^-3) = 0.9526, 1/(1+e^-0.5) = 0.6225,
    /// 1/(1+e^-2) = 0.8808. The two largest belong to tokens 1 and 3.
    /// Dropping in token order kept tokens 0 and 1.
    #[test]
    fn over_capacity_keeps_the_most_confident_tokens_not_the_first() {
        let logits: Vec<f32> = vec![
            1.0, 0.0,
            3.0, 0.0,
            0.5, 0.0,
            2.0, 0.0,
        ];
        let result = route_topk(&logits, 4, 2, 1, 1.0);
        assert_eq!(result.total_assigned, 2);
        assert_eq!(result.expert_boundaries, vec![0, 2, 2]);
        assert_eq!(result.sorted_token_indices, vec![1, 3]);
        assert_eq!(result.sorted_assignment_weights, vec![1.0, 1.0]);
    }

    /// A token that loses one of its two experts to capacity carries its
    /// whole weight on the one it keeps.
    ///
    /// Two tokens, two experts, top-2, factor 0.5: capacity is
    /// ceil(2 * 2 / 2 * 0.5) = 1 per expert, and both tokens select both
    /// experts. Token 0 (logits 2, 0) gives expert 0 probability 0.8808 and
    /// expert 1 0.1192; token 1 (logits 0, 1) gives them 0.2689 and 0.7311.
    /// Expert 0 keeps token 0 and expert 1 keeps token 1, so each token has
    /// one surviving assignment, at weight 1.0. Before, expert 1 also kept
    /// token 0 (first come), token 1 was dropped by both, and the survivors
    /// kept their partial weights 0.8808 and 0.1192.
    #[test]
    fn a_token_left_with_one_expert_gets_its_full_weight() {
        let logits: Vec<f32> = vec![
            2.0, 0.0,
            0.0, 1.0,
        ];
        let result = route_topk(&logits, 2, 2, 2, 0.5);
        assert_eq!(result.total_assigned, 2);
        assert_eq!(result.expert_boundaries, vec![0, 1, 2]);
        assert_eq!(result.sorted_token_indices, vec![0, 1]);
        for &w in &result.sorted_assignment_weights {
            assert!((w - 1.0).abs() < 1e-6, "surviving weight should be 1.0, got {w}");
        }
    }

    /// Two experts, top-2: every token is assigned to both, so each expert
    /// receives `total_tokens` assignments -- its exact fair share. At
    /// factor 1.0 nothing may be dropped. The capacity used to be
    /// `tokens / experts` regardless of `top_k`, i.e. 2 of the 4 here, and
    /// tokens 2 and 3 lost both of their assignments.
    #[test]
    fn balanced_top2_routing_drops_nothing_at_factor_one() {
        let logits: Vec<f32> = vec![0.3, 0.1, 0.2, 0.4, 0.5, 0.5, 0.1, 0.9];
        let result = route_topk(&logits, 4, 2, 2, 1.0);
        assert_eq!(result.total_assigned, 8, "4 tokens x top-2 = 8 assignments");
        for expert in 0..2 {
            let count = result.expert_boundaries[expert + 1] - result.expert_boundaries[expert];
            assert_eq!(count, 4, "expert {expert} holds every token");
        }
        let mut seen = [0usize; 4];
        for &token in &result.sorted_token_indices {
            seen[token as usize] += 1;
        }
        assert_eq!(seen, [2, 2, 2, 2], "each token keeps both assignments");
    }
}
