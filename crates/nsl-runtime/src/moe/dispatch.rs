/// Reorder input tokens according to sorted_token_indices.
/// Returns a new Vec with tokens in expert-grouped order.
pub fn scatter_tokens(
    tokens: &[f32],
    sorted_indices: &[i32],
    hidden_dim: usize,
) -> Vec<f32> {
    let num_assigned = sorted_indices.len();
    let mut sorted = vec![0.0f32; num_assigned * hidden_dim];
    for (dst_row, &src_row) in sorted_indices.iter().enumerate() {
        let src_off = src_row as usize * hidden_dim;
        let dst_off = dst_row * hidden_dim;
        sorted[dst_off..dst_off + hidden_dim]
            .copy_from_slice(&tokens[src_off..src_off + hidden_dim]);
    }
    sorted
}

/// Combine expert outputs back to original token order with weighted sum.
pub fn gather_tokens(
    expert_outputs: &[f32],
    reverse_indices: &[i32],
    expert_weights: &[f32],
    total_tokens: usize,
    _top_k: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; total_tokens * hidden_dim];
    let num_assigned = reverse_indices.len();
    for i in 0..num_assigned {
        let original_token = reverse_indices[i] as usize;
        let weight = expert_weights[i];
        let src_off = i * hidden_dim;
        let dst_off = original_token * hidden_dim;
        for d in 0..hidden_dim {
            output[dst_off + d] += weight * expert_outputs[src_off + d];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_gather_roundtrip() {
        let tokens: Vec<f32> = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let sorted_indices = vec![2i32, 0, 1];
        let hidden_dim = 2;

        let scattered = scatter_tokens(&tokens, &sorted_indices, hidden_dim);
        assert_eq!(scattered, vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);

        let expert_outputs = scattered.clone();
        let expert_weights = vec![1.0f32, 1.0, 1.0];
        let reverse_indices = vec![2i32, 0, 1];

        let gathered = gather_tokens(
            &expert_outputs, &reverse_indices, &expert_weights,
            3, 1, hidden_dim,
        );
        assert_eq!(gathered, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_moe_full_pipeline_matches_naive() {
        // 4 tokens, 2 experts, top_k=1, hidden_dim=3, intermediate_dim=2
        let tokens: Vec<f32> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let hidden_dim = 3;
        let intermediate_dim = 2;
        let num_experts = 2;

        // expert_weights[expert][hidden][intermediate] row-major
        let expert_weights: Vec<f32> = vec![
            // expert 0: [3x2]
            1.0, 0.0,
            0.0, 1.0,
            0.5, 0.5,
            // expert 1: [3x2]
            0.0, 1.0,
            1.0, 0.0,
            0.5, 0.5,
        ];

        // Force routing: tokens 0,3 -> expert 0, tokens 1,2 -> expert 1
        let logits: Vec<f32> = vec![
            5.0, 0.0,
            0.0, 5.0,
            0.0, 5.0,
            5.0, 0.0,
        ];

        let routing = crate::moe::router::route_topk(&logits, 4, num_experts, 1, 2.0);

        let scattered = scatter_tokens(&tokens, &routing.sorted_token_indices, hidden_dim);

        // Naive expert matmul
        let mut expert_outputs = vec![0.0f32; routing.total_assigned as usize * intermediate_dim];
        for e in 0..num_experts {
            let start = routing.expert_boundaries[e] as usize;
            let end = routing.expert_boundaries[e + 1] as usize;
            for t in start..end {
                for j in 0..intermediate_dim {
                    let mut sum = 0.0f32;
                    for k in 0..hidden_dim {
                        sum += scattered[t * hidden_dim + k]
                            * expert_weights[e * hidden_dim * intermediate_dim + k * intermediate_dim + j];
                    }
                    expert_outputs[t * intermediate_dim + j] = sum;
                }
            }
        }

        // Gather with all weights=1.0 for top_k=1
        let gather_weights: Vec<f32> = vec![1.0; routing.total_assigned as usize];
        let output = gather_tokens(
            &expert_outputs,
            &routing.sorted_token_indices,
            &gather_weights,
            4,
            1,
            intermediate_dim,
        );

        assert_eq!(output.len(), 4 * intermediate_dim);
        // Token 0 ([1,0,0]) through expert 0: [1*1+0*0+0*0.5, 1*0+0*1+0*0.5] = [1.0, 0.0]
        assert!((output[0] - 1.0).abs() < 1e-5, "got {}", output[0]);
        assert!((output[1] - 0.0).abs() < 1e-5, "got {}", output[1]);
    }

    #[test]
    fn test_gather_topk2_weighted_sum() {
        let expert_outputs: Vec<f32> = vec![
            10.0, 20.0,
            30.0, 40.0,
            50.0, 60.0,
            70.0, 80.0,
        ];
        let reverse_indices = vec![0i32, 0, 1, 1];
        let expert_weights = vec![0.6f32, 0.4, 0.7, 0.3];

        let gathered = gather_tokens(
            &expert_outputs, &reverse_indices, &expert_weights,
            2, 2, 2,
        );
        assert!((gathered[0] - 18.0).abs() < 1e-5);
        assert!((gathered[1] - 28.0).abs() < 1e-5);
        assert!((gathered[2] - 56.0).abs() < 1e-5);
        assert!((gathered[3] - 66.0).abs() < 1e-5);
    }
}
