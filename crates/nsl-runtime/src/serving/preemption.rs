//! PreemptionManager: swap/recompute strategies for memory pressure.

use crate::serving::request::{InferenceRequest, RequestState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreemptionPolicy {
    Swap,
    Recompute,
}

pub struct PreemptionManager {
    pub default_policy: PreemptionPolicy,
    pub pcie_bandwidth: f64,
    pub prefill_throughput: f64,
}

impl PreemptionManager {
    pub fn new() -> Self {
        PreemptionManager {
            default_policy: PreemptionPolicy::Recompute,
            pcie_bandwidth: 25e9,
            prefill_throughput: 10_000.0,
        }
    }

    pub fn choose_policy(&self, total_tokens: usize, kv_bytes_per_token: usize) -> PreemptionPolicy {
        let swap_time = (total_tokens * kv_bytes_per_token) as f64 / self.pcie_bandwidth;
        let recompute_time = total_tokens as f64 / self.prefill_throughput;
        if swap_time < recompute_time {
            PreemptionPolicy::Swap
        } else {
            PreemptionPolicy::Recompute
        }
    }

    pub fn preempt_recompute(request: &mut InferenceRequest) {
        let generated = request.generated_tokens.clone();
        request.state = RequestState::Preempted { generated_so_far: generated };
        request.kv_seq_id = None;
    }

    pub fn resume_recompute(request: &mut InferenceRequest) {
        if let RequestState::Preempted { generated_so_far } = &request.state {
            let mut full_tokens = request.prompt_tokens.clone();
            full_tokens.extend(generated_so_far);
            request.prompt_tokens = full_tokens;
            request.state = RequestState::Prefilling { tokens_processed: 0 };
        }
    }
}

impl Default for PreemptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_swap_for_small_kv() {
        let pm = PreemptionManager::new();
        let policy = pm.choose_policy(100, 256);
        assert_eq!(policy, PreemptionPolicy::Swap);
    }

    #[test]
    fn choose_swap_for_long_sequences() {
        let pm = PreemptionManager::new();
        let policy = pm.choose_policy(4000, 8192);
        assert_eq!(policy, PreemptionPolicy::Swap);
    }

    #[test]
    fn preempt_and_resume_recompute() {
        let mut req = InferenceRequest::new(0, vec![1, 2, 3], 10, 0.7, 0.9);
        req.state = RequestState::Decoding;
        req.generated_tokens = vec![10, 11, 12];

        PreemptionManager::preempt_recompute(&mut req);
        assert!(matches!(req.state, RequestState::Preempted { .. }));
        assert!(req.kv_seq_id.is_none());

        PreemptionManager::resume_recompute(&mut req);
        assert!(matches!(req.state, RequestState::Prefilling { tokens_processed: 0 }));
        assert_eq!(req.prompt_tokens, vec![1, 2, 3, 10, 11, 12]);
    }
}
