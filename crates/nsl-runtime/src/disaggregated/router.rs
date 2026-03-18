//! M41: Disaggregated inference router.
//!
//! The router process accepts requests, dispatches prefill to prefill workers,
//! routes KV-transfer completion to decode workers, and streams tokens back.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::messages::{DisaggRequestState, WorkerHandle, WorkerRole};

// ---------------------------------------------------------------------------
// Scheduling policies
// ---------------------------------------------------------------------------

/// How to select a prefill worker for a new request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrefillPolicy {
    /// Route to the prefill worker with fewest active requests.
    LeastLoaded,
    /// Simple round-robin rotation.
    RoundRobin,
}

/// How to select a decode worker after prefill completes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodePolicy {
    /// Route to the decode worker with fewest active sequences.
    LeastLoaded,
    /// Route to the decode worker with the most free KV-cache blocks.
    MemoryAware,
}

// ---------------------------------------------------------------------------
// Router configuration
// ---------------------------------------------------------------------------

pub struct DisaggregatedConfig {
    pub num_prefill_workers: usize,
    pub num_decode_workers: usize,
    pub prefill_policy: PrefillPolicy,
    pub decode_policy: DecodePolicy,
    pub max_batch_per_decode: usize,
    pub max_seq_len: usize,
    pub kv_blocks_per_worker: u32,
    pub drain_timeout_ms: u64,
}

impl Default for DisaggregatedConfig {
    fn default() -> Self {
        DisaggregatedConfig {
            num_prefill_workers: 1,
            num_decode_workers: 1,
            prefill_policy: PrefillPolicy::LeastLoaded,
            decode_policy: DecodePolicy::LeastLoaded,
            max_batch_per_decode: 32,
            max_seq_len: 4096,
            kv_blocks_per_worker: 2048,
            drain_timeout_ms: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub struct DisaggregatedRouter {
    config: DisaggregatedConfig,
    prefill_pool: Vec<WorkerHandle>,
    decode_pool: Vec<WorkerHandle>,
    requests: HashMap<u64, DisaggRequestState>,
    next_request_id: AtomicU64,
    round_robin_prefill: usize,
}

impl DisaggregatedRouter {
    pub fn new(config: DisaggregatedConfig) -> Self {
        let mut prefill_pool = Vec::new();
        for i in 0..config.num_prefill_workers {
            prefill_pool.push(WorkerHandle::new(i as i32, WorkerRole::Prefill, 0));
        }

        let mut decode_pool = Vec::new();
        for i in 0..config.num_decode_workers {
            decode_pool.push(WorkerHandle::new(
                i as i32,
                WorkerRole::Decode,
                config.kv_blocks_per_worker,
            ));
        }

        DisaggregatedRouter {
            config,
            prefill_pool,
            decode_pool,
            requests: HashMap::new(),
            next_request_id: AtomicU64::new(0),
            round_robin_prefill: 0,
        }
    }

    /// Enqueue a new request. Returns the request ID.
    pub fn enqueue_request(&mut self) -> u64 {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.requests.insert(id, DisaggRequestState::Queued);
        id
    }

    /// Select a prefill worker for a queued request.
    pub fn select_prefill_worker(&mut self) -> Option<i32> {
        if self.prefill_pool.is_empty() {
            return None;
        }
        match self.config.prefill_policy {
            PrefillPolicy::LeastLoaded => {
                self.prefill_pool
                    .iter()
                    .min_by_key(|w| w.active_requests)
                    .map(|w| w.rank)
            }
            PrefillPolicy::RoundRobin => {
                let idx = self.round_robin_prefill % self.prefill_pool.len();
                self.round_robin_prefill += 1;
                Some(self.prefill_pool[idx].rank)
            }
        }
    }

    /// Select a decode worker for a completed prefill.
    pub fn select_decode_worker(&self, _kv_blocks_needed: u32) -> Option<i32> {
        if self.decode_pool.is_empty() {
            return None;
        }
        match self.config.decode_policy {
            DecodePolicy::LeastLoaded => {
                self.decode_pool
                    .iter()
                    .min_by_key(|w| w.active_requests)
                    .map(|w| w.rank)
            }
            DecodePolicy::MemoryAware => {
                self.decode_pool
                    .iter()
                    .max_by_key(|w| w.free_kv_blocks)
                    .map(|w| w.rank)
            }
        }
    }

    /// Mark a request as dispatched to a prefill worker.
    pub fn mark_prefilling(&mut self, request_id: u64, prefill_rank: i32) {
        if let Some(worker) = self.prefill_pool.iter_mut().find(|w| w.rank == prefill_rank) {
            worker.active_requests += 1;
        }
        self.requests.insert(request_id, DisaggRequestState::Prefilling { prefill_rank });
    }

    /// Mark a request as transferring KV from prefill to decode.
    pub fn mark_transferring(&mut self, request_id: u64, from_rank: i32, to_rank: i32) {
        if let Some(worker) = self.prefill_pool.iter_mut().find(|w| w.rank == from_rank) {
            worker.active_requests = worker.active_requests.saturating_sub(1);
        }
        self.requests.insert(request_id, DisaggRequestState::Transferring { from_rank, to_rank });
    }

    /// Mark a request as decoding on a decode worker, reserving KV blocks.
    pub fn mark_decoding(&mut self, request_id: u64, decode_rank: i32, kv_blocks_used: u32) {
        if let Some(worker) = self.decode_pool.iter_mut().find(|w| w.rank == decode_rank) {
            worker.active_requests += 1;
            worker.free_kv_blocks = worker.free_kv_blocks.saturating_sub(kv_blocks_used);
        }
        self.requests.insert(request_id, DisaggRequestState::Decoding {
            decode_rank,
            tokens_generated: 0,
        });
    }

    /// Record a generated token for a request.
    pub fn record_token(&mut self, request_id: u64) {
        if let Some(DisaggRequestState::Decoding { tokens_generated, .. }) = self.requests.get_mut(&request_id) {
            *tokens_generated += 1;
        }
    }

    /// Mark a request as complete.
    pub fn mark_complete(&mut self, request_id: u64, total_tokens: u32) {
        if let Some(DisaggRequestState::Decoding { decode_rank, .. }) = self.requests.get(&request_id) {
            let decode_rank = *decode_rank;
            if let Some(worker) = self.decode_pool.iter_mut().find(|w| w.rank == decode_rank) {
                worker.active_requests = worker.active_requests.saturating_sub(1);
            }
        }
        self.requests.insert(request_id, DisaggRequestState::Complete { total_tokens });
    }

    /// Get the current state of a request.
    pub fn request_state(&self, request_id: u64) -> Option<&DisaggRequestState> {
        self.requests.get(&request_id)
    }

    /// Number of active (non-complete, non-queued) requests.
    pub fn active_request_count(&self) -> usize {
        self.requests.values().filter(|s| {
            matches!(s,
                DisaggRequestState::Prefilling { .. }
                | DisaggRequestState::Transferring { .. }
                | DisaggRequestState::Decoding { .. }
            )
        }).count()
    }

    /// Number of queued requests waiting for prefill.
    pub fn queued_count(&self) -> usize {
        self.requests.values().filter(|s| matches!(s, DisaggRequestState::Queued)).count()
    }

    /// Number of completed requests.
    pub fn completed_count(&self) -> usize {
        self.requests.values().filter(|s| matches!(s, DisaggRequestState::Complete { .. })).count()
    }

    /// Drain completed requests, returning their IDs.
    pub fn drain_completed(&mut self) -> Vec<u64> {
        let completed: Vec<u64> = self.requests.iter()
            .filter(|(_, s)| matches!(s, DisaggRequestState::Complete { .. }))
            .map(|(id, _)| *id)
            .collect();
        for id in &completed {
            self.requests.remove(id);
        }
        completed
    }
}

// ---------------------------------------------------------------------------
// Global context + FFI
// ---------------------------------------------------------------------------

static DISAGG_CTX: Mutex<Option<DisaggregatedRouter>> = Mutex::new(None);

/// Read NSL_ROLE env var and return a role code.
/// Returns: 0 = router, 1 = prefill, 2 = decode.
/// Used by codegen for role dispatch branching (avoids string ops in Cranelift IR).
#[no_mangle]
pub extern "C" fn nsl_disagg_get_role() -> i64 {
    match std::env::var("NSL_ROLE").as_deref() {
        Ok("router") => 0,
        Ok("prefill") => 1,
        Ok("decode") => 2,
        _ => 0, // default: router (monolithic fallback)
    }
}

/// Read NSL_LOCAL_RANK env var and return the rank.
/// Returns the integer rank (default 0).
#[no_mangle]
pub extern "C" fn nsl_disagg_get_rank() -> i64 {
    std::env::var("NSL_LOCAL_RANK")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(0)
}

/// Initialize the disaggregated inference router.
///
/// Parameters (all i64 for Cranelift):
/// - num_prefill: number of prefill workers
/// - num_decode: number of decode workers
/// - max_batch: max batch size per decode worker
/// - kv_blocks: KV blocks per decode worker
///
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_disagg_init(
    num_prefill: i64,
    num_decode: i64,
    max_batch: i64,
    kv_blocks: i64,
) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let config = DisaggregatedConfig {
        num_prefill_workers: num_prefill as usize,
        num_decode_workers: num_decode as usize,
        max_batch_per_decode: max_batch as usize,
        kv_blocks_per_worker: kv_blocks as u32,
        ..Default::default()
    };
    *guard = Some(DisaggregatedRouter::new(config));
    0
}

/// Enqueue a request to the router. Returns the request ID.
#[no_mangle]
pub extern "C" fn nsl_disagg_enqueue() -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.enqueue_request() as i64
}

/// Select a prefill worker for the next request. Returns the worker rank, or -1 if none.
#[no_mangle]
pub extern "C" fn nsl_disagg_select_prefill() -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.select_prefill_worker().map(|r| r as i64).unwrap_or(-1)
}

/// Select a decode worker. Returns the worker rank, or -1 if none.
#[no_mangle]
pub extern "C" fn nsl_disagg_select_decode(kv_blocks_needed: i64) -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.select_decode_worker(kv_blocks_needed as u32).map(|r| r as i64).unwrap_or(-1)
}

/// Mark a request as prefilling on the given worker.
#[no_mangle]
pub extern "C" fn nsl_disagg_mark_prefilling(request_id: i64, prefill_rank: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.mark_prefilling(request_id as u64, prefill_rank as i32);
    0
}

/// Mark a request as decoding on the given worker, reserving kv_blocks_used blocks.
#[no_mangle]
pub extern "C" fn nsl_disagg_mark_decoding(request_id: i64, decode_rank: i64, kv_blocks_used: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.mark_decoding(request_id as u64, decode_rank as i32, kv_blocks_used as u32);
    0
}

/// Record a generated token for a request.
#[no_mangle]
pub extern "C" fn nsl_disagg_record_token(request_id: i64, _token_id: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.record_token(request_id as u64);
    0
}

/// Mark a request as complete.
#[no_mangle]
pub extern "C" fn nsl_disagg_mark_complete(request_id: i64, total_tokens: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.mark_complete(request_id as u64, total_tokens as u32);
    0
}

/// Returns the number of queued requests.
#[no_mangle]
pub extern "C" fn nsl_disagg_queued_count() -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.queued_count() as i64
}

/// Returns the number of active (in-flight) requests.
#[no_mangle]
pub extern "C" fn nsl_disagg_active_count() -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.active_request_count() as i64
}

/// Returns the number of completed requests.
#[no_mangle]
pub extern "C" fn nsl_disagg_completed_count() -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.completed_count() as i64
}

/// Destroy the disaggregated router context.
#[no_mangle]
pub extern "C" fn nsl_disagg_destroy() -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_disagg_destroy();
        guard
    }

    #[test]
    fn router_basic_lifecycle() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 2,
            num_decode_workers: 3,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        assert_eq!(router.prefill_pool.len(), 2);
        assert_eq!(router.decode_pool.len(), 3);

        // Enqueue and dispatch
        let req0 = router.enqueue_request();
        let req1 = router.enqueue_request();
        assert_eq!(router.queued_count(), 2);

        // Select prefill workers
        let pw0 = router.select_prefill_worker().unwrap();
        router.mark_prefilling(req0, pw0);
        let pw1 = router.select_prefill_worker().unwrap();
        router.mark_prefilling(req1, pw1);
        assert_eq!(router.active_request_count(), 2);
        assert_eq!(router.queued_count(), 0);

        // Complete prefill, start decode
        let dw0 = router.select_decode_worker(4).unwrap();
        router.mark_transferring(req0, pw0, dw0);
        router.mark_decoding(req0, dw0, 4);

        // Generate tokens
        router.record_token(req0);
        router.record_token(req0);
        if let Some(DisaggRequestState::Decoding { tokens_generated, .. }) = router.request_state(req0) {
            assert_eq!(*tokens_generated, 2);
        }

        // Complete
        router.mark_complete(req0, 2);
        assert_eq!(router.completed_count(), 1);

        let drained = router.drain_completed();
        assert_eq!(drained, vec![req0]);
        assert_eq!(router.completed_count(), 0);
    }

    #[test]
    fn least_loaded_prefill_policy() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 2,
            num_decode_workers: 1,
            prefill_policy: PrefillPolicy::LeastLoaded,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        // Both idle — should pick rank 0 (first min)
        let r = router.select_prefill_worker().unwrap();
        assert_eq!(r, 0);

        // Load rank 0 — next should pick rank 1
        let req = router.enqueue_request();
        router.mark_prefilling(req, 0);
        let r = router.select_prefill_worker().unwrap();
        assert_eq!(r, 1);
    }

    #[test]
    fn round_robin_prefill_policy() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 3,
            num_decode_workers: 1,
            prefill_policy: PrefillPolicy::RoundRobin,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        assert_eq!(router.select_prefill_worker().unwrap(), 0);
        assert_eq!(router.select_prefill_worker().unwrap(), 1);
        assert_eq!(router.select_prefill_worker().unwrap(), 2);
        assert_eq!(router.select_prefill_worker().unwrap(), 0); // wraps
    }

    #[test]
    fn memory_aware_decode_policy() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 1,
            num_decode_workers: 3,
            decode_policy: DecodePolicy::MemoryAware,
            kv_blocks_per_worker: 100,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        // Reduce blocks on worker 0
        router.decode_pool[0].free_kv_blocks = 50;
        // Worker 1 has most blocks (100) — should be selected
        let r = router.select_decode_worker(10).unwrap();
        // Workers 1 and 2 both have 100; first max is rank 1
        assert!(r == 1 || r == 2);
    }

    #[test]
    fn ffi_init_destroy() {
        let _lock = setup();

        assert_eq!(nsl_disagg_init(2, 4, 32, 2048), 0);
        assert_eq!(nsl_disagg_init(1, 1, 32, 2048), -1); // double init

        assert_eq!(nsl_disagg_enqueue(), 0); // first request
        assert_eq!(nsl_disagg_enqueue(), 1); // second request
        assert_eq!(nsl_disagg_queued_count(), 2);

        let pw = nsl_disagg_select_prefill();
        assert!(pw >= 0);
        nsl_disagg_mark_prefilling(0, pw);
        assert_eq!(nsl_disagg_active_count(), 1);

        nsl_disagg_mark_complete(0, 10);
        assert_eq!(nsl_disagg_completed_count(), 1);

        assert_eq!(nsl_disagg_destroy(), 0);
    }
}
