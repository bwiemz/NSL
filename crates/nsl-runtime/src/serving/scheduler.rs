//! BatchScheduler: continuous batching with chunked prefill.

use std::collections::VecDeque;
use crate::serving::request::{InferenceRequest, RequestId, RequestState};

pub struct SchedulerConfig {
    pub max_batch: usize,
    pub max_seq_len: usize,
    pub kv_blocks: usize,
    pub prefill_chunk: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_batch: 32,
            max_seq_len: 4096,
            kv_blocks: 2048,
            prefill_chunk: 512,
        }
    }
}

pub struct SchedulerStep {
    pub prefill_chunks: Vec<(RequestId, usize, usize)>,
    pub decode_ids: Vec<RequestId>,
}

pub struct BatchScheduler {
    pub config: SchedulerConfig,
    pub waiting: VecDeque<InferenceRequest>,
    pub active: Vec<InferenceRequest>,
    pub completed: Vec<InferenceRequest>,
    next_id: RequestId,
}

impl BatchScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        BatchScheduler {
            config,
            waiting: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_id: 0,
        }
    }

    pub fn enqueue(
        &mut self,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        let req = InferenceRequest::new(id, prompt_tokens, max_tokens, temperature, top_p);
        self.waiting.push_back(req);
        id
    }

    pub fn step(&mut self) -> SchedulerStep {
        // Admit waiting requests if batch has room
        while self.active.len() < self.config.max_batch {
            if let Some(mut req) = self.waiting.pop_front() {
                req.state = RequestState::Prefilling { tokens_processed: 0 };
                self.active.push(req);
            } else {
                break;
            }
        }

        let mut prefill_chunks = Vec::new();
        let mut decode_ids = Vec::new();

        for req in &mut self.active {
            match &req.state {
                RequestState::Prefilling { tokens_processed } => {
                    let start = *tokens_processed;
                    let end = (start + self.config.prefill_chunk).min(req.prompt_tokens.len());
                    prefill_chunks.push((req.id, start, end));
                    if end >= req.prompt_tokens.len() {
                        req.state = RequestState::Decoding;
                    } else {
                        req.state = RequestState::Prefilling { tokens_processed: end };
                    }
                }
                RequestState::Decoding => {
                    decode_ids.push(req.id);
                }
                _ => {}
            }
        }

        SchedulerStep { prefill_chunks, decode_ids }
    }

    pub fn record_token(&mut self, request_id: RequestId, token_id: i64) -> bool {
        if let Some(req) = self.active.iter_mut().find(|r| r.id == request_id) {
            req.push_token(token_id)
        } else {
            false
        }
    }

    pub fn drain_completed(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].is_complete() {
                let req = self.active.remove(i);
                self.completed.push(req);
            } else {
                i += 1;
            }
        }
    }

    pub fn take_completed(&mut self, request_id: RequestId) -> Option<InferenceRequest> {
        if let Some(pos) = self.completed.iter().position(|r| r.id == request_id) {
            Some(self.completed.remove(pos))
        } else {
            None
        }
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_basic_lifecycle() {
        let config = SchedulerConfig {
            max_batch: 2, max_seq_len: 100, kv_blocks: 64, prefill_chunk: 512,
        };
        let mut sched = BatchScheduler::new(config);

        let id0 = sched.enqueue(vec![1, 2, 3], 2, 0.7, 0.9);
        let id1 = sched.enqueue(vec![4, 5], 2, 0.7, 0.9);
        let _id2 = sched.enqueue(vec![6], 2, 0.7, 0.9);

        assert_eq!(sched.waiting.len(), 3);

        // Step 1: admits first 2, both do single-chunk prefill
        let step = sched.step();
        assert_eq!(sched.active.len(), 2);
        assert_eq!(sched.waiting.len(), 1);
        assert_eq!(step.prefill_chunks.len(), 2);

        // Step 2: both now decoding
        let step = sched.step();
        assert_eq!(step.decode_ids.len(), 2);

        assert!(!sched.record_token(id0, 10));
        assert!(sched.record_token(id0, 11)); // done
        assert!(!sched.record_token(id1, 20));

        sched.drain_completed();
        assert_eq!(sched.active.len(), 1);
        assert_eq!(sched.completed.len(), 1);

        // id2 admitted
        let step = sched.step();
        assert_eq!(sched.active.len(), 2);
        assert_eq!(step.prefill_chunks.len(), 1);
        assert_eq!(step.decode_ids.len(), 1);
    }

    #[test]
    fn scheduler_chunked_prefill() {
        let config = SchedulerConfig {
            max_batch: 4, max_seq_len: 4096, kv_blocks: 256, prefill_chunk: 3,
        };
        let mut sched = BatchScheduler::new(config);

        let id = sched.enqueue(vec![1, 2, 3, 4, 5, 6, 7], 2, 0.7, 0.9);

        let step = sched.step();
        assert_eq!(step.prefill_chunks, vec![(id, 0, 3)]);

        let step = sched.step();
        assert_eq!(step.prefill_chunks, vec![(id, 3, 6)]);

        let step = sched.step();
        assert_eq!(step.prefill_chunks, vec![(id, 6, 7)]);

        let step = sched.step();
        assert_eq!(step.decode_ids, vec![id]);
    }
}
