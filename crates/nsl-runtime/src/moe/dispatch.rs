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
