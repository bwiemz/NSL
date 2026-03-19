//! InferenceRequest: per-request state machine for continuous batching.

pub type RequestId = u64;

#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    Waiting,
    Prefilling { tokens_processed: usize },
    Decoding,
    Preempted { generated_so_far: Vec<i64> },
    Complete,
}

pub struct InferenceRequest {
    pub id: RequestId,
    pub state: RequestState,
    pub prompt_tokens: Vec<i64>,
    pub generated_tokens: Vec<i64>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub kv_seq_id: Option<u64>,
    pub priority: i64,
    pub total_tokens: usize,
    /// M44b: Per-request grammar FSM state for constrained decoding.
    pub grammar_state: Option<crate::grammar::GrammarRequestState>,
}

impl InferenceRequest {
    pub fn new(
        id: RequestId,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Self {
        let total = prompt_tokens.len();
        InferenceRequest {
            id,
            state: RequestState::Waiting,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            temperature,
            top_p,
            kv_seq_id: None,
            priority: 0,
            total_tokens: total,
            grammar_state: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.state == RequestState::Complete
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state, RequestState::Prefilling { .. } | RequestState::Decoding)
    }

    pub fn remaining_prefill(&self) -> usize {
        match self.state {
            RequestState::Prefilling { tokens_processed } => {
                self.prompt_tokens.len().saturating_sub(tokens_processed)
            }
            _ => 0,
        }
    }

    /// Mark a generated token. Returns true if generation is now complete.
    pub fn push_token(&mut self, token_id: i64) -> bool {
        self.generated_tokens.push(token_id);
        self.total_tokens += 1;
        if self.generated_tokens.len() >= self.max_tokens || token_id == 2 {
            self.state = RequestState::Complete;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_lifecycle() {
        let mut req = InferenceRequest::new(0, vec![1, 2, 3], 5, 0.7, 0.9);
        assert_eq!(req.state, RequestState::Waiting);
        assert!(!req.is_active());

        req.state = RequestState::Prefilling { tokens_processed: 0 };
        assert!(req.is_active());
        assert_eq!(req.remaining_prefill(), 3);

        req.state = RequestState::Decoding;
        assert!(req.is_active());

        assert!(!req.push_token(10));
        assert!(!req.push_token(11));
        assert_eq!(req.generated_tokens.len(), 2);

        assert!(req.push_token(2)); // EOS
        assert!(req.is_complete());
    }

    #[test]
    fn request_max_tokens() {
        let mut req = InferenceRequest::new(0, vec![1], 3, 0.7, 0.9);
        req.state = RequestState::Decoding;
        assert!(!req.push_token(10));
        assert!(!req.push_token(11));
        assert!(req.push_token(12)); // 3rd token
        assert!(req.is_complete());
    }
}
