/// Configuration for an MoE layer, extracted from @moe decorator at compile time.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub num_experts: i32,
    pub top_k: i32,
    pub capacity_factor: f32,
    pub aux_loss_coeff: f32,
}

/// Output of the routing decision. All pointer fields are device-resident (GPU) memory.
#[repr(C)]
pub struct MoeRoutingResult {
    /// Flat array of expert assignments: [total_tokens * top_k]
    pub expert_indices: *mut i32,
    /// Gating weights after softmax: [total_tokens * top_k]
    pub expert_weights: *mut f32,
    /// Sorted token indices: [total_assigned]
    pub sorted_token_indices: *mut i32,
    /// Expert boundaries: [num_experts + 1]
    pub expert_boundaries: *mut i32,
    /// Number of tokens actually assigned (after capacity truncation)
    pub total_assigned: i64,
    /// Importance loss scalar (CV-squared of per-expert weight sums)
    pub importance_loss: f32,
    /// Load loss scalar (CV-squared of per-expert token counts)
    pub load_loss: f32,
}

impl MoeRoutingResult {
    pub fn new_zeroed() -> Self {
        Self {
            expert_indices: std::ptr::null_mut(),
            expert_weights: std::ptr::null_mut(),
            sorted_token_indices: std::ptr::null_mut(),
            expert_boundaries: std::ptr::null_mut(),
            total_assigned: 0,
            importance_loss: 0.0,
            load_loss: 0.0,
        }
    }
}
