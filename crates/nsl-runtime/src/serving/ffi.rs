//! FFI exports for the M29 serving engine.

use std::sync::Mutex;

use crate::serving::preemption::PreemptionManager;
use crate::serving::request::RequestId;
use crate::serving::scheduler::{BatchScheduler, SchedulerConfig};

static SERVE_CTX: Mutex<Option<ServeContext>> = Mutex::new(None);

struct ServeContext {
    scheduler: BatchScheduler,
    _preemption: PreemptionManager,
}

#[no_mangle]
pub extern "C" fn nsl_serve_init(
    max_batch: i64,
    max_seq_len: i64,
    kv_blocks: i64,
    prefill_chunk: i64,
) -> i64 {
    let config = SchedulerConfig {
        max_batch: max_batch as usize,
        max_seq_len: max_seq_len as usize,
        kv_blocks: kv_blocks as usize,
        prefill_chunk: prefill_chunk as usize,
        speculative_tokens: 0,
    };
    let mut guard = SERVE_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(ServeContext {
        scheduler: BatchScheduler::new(config),
        _preemption: PreemptionManager::new(),
    });
    0
}

#[no_mangle]
pub extern "C" fn nsl_serve_enqueue(
    prompt_ptr: i64,
    prompt_len: i64,
    max_tokens: i64,
    temperature: f64,
    top_p: f64,
) -> i64 {
    let tokens = if prompt_ptr != 0 && prompt_len > 0 {
        let ptr = prompt_ptr as *const i64;
        let len = prompt_len as usize;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    } else {
        Vec::new()
    };
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    ctx.scheduler.enqueue(tokens, max_tokens as usize, temperature, top_p) as i64
}

#[no_mangle]
pub extern "C" fn nsl_serve_step() -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    let _step = ctx.scheduler.step();
    ctx.scheduler.active_count() as i64
}

#[no_mangle]
pub extern "C" fn nsl_serve_record_token(request_id: i64, token_id: i64) -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    if ctx.scheduler.record_token(request_id as RequestId, token_id) { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn nsl_serve_drain_completed() -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    let before = ctx.scheduler.completed.len();
    ctx.scheduler.drain_completed();
    (ctx.scheduler.completed.len() - before) as i64
}

#[no_mangle]
pub extern "C" fn nsl_serve_has_work() -> i64 {
    let guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_serve_init not called");
    if ctx.scheduler.has_work() { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn nsl_serve_completed_count() -> i64 {
    let guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_serve_init not called");
    ctx.scheduler.completed.len() as i64
}

#[no_mangle]
pub extern "C" fn nsl_serve_preempt(request_id: i64) -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    if let Some(req) = ctx.scheduler.active.iter_mut().find(|r| r.id == request_id as u64) {
        PreemptionManager::preempt_recompute(req);
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn nsl_serve_destroy() -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// M44b: Constrained decoding — grammar-aware serving FFI
// ---------------------------------------------------------------------------

/// Apply grammar constraint to logits for a request.
/// Called between model forward pass and sampling.
/// `request_id`: which request's grammar state to use.
/// `logits_ptr`: pointer to mutable f32 array of vocab_size.
/// Returns 0 on success/no-op, negative on error.
#[no_mangle]
pub extern "C" fn nsl_serve_apply_grammar(request_id: i64, logits_ptr: i64) -> i64 {
    let guard = match SERVE_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let ctx = match guard.as_ref() {
        Some(c) => c,
        None => return -2,
    };

    // Find the request and apply its grammar mask
    if let Some(req) = ctx.scheduler.active.iter().find(|r| r.id == request_id as u64) {
        if let Some(ref grammar_state) = req.grammar_state {
            if grammar_state.active {
                // Lock GRAMMAR_CTX directly — avoid FFI wrapper's redundant lock in hot path
                let grammar_guard = match crate::grammar::GRAMMAR_CTX.lock() {
                    Ok(g) => g,
                    Err(_) => return -1,
                };
                if let Some(ref grammar_ctx) = *grammar_guard {
                    if logits_ptr != 0 {
                        let logits = unsafe {
                            std::slice::from_raw_parts_mut(
                                logits_ptr as *mut f32,
                                grammar_ctx.fsm.vocab_size,
                            )
                        };
                        grammar_ctx.fsm.apply_logit_mask(logits, grammar_state.current_state);
                    }
                }
            }
        }
    }
    0
}

/// Advance grammar state after a token is generated.
/// Called after sampling to update the FSM state.
/// Returns 0 on success/no-op, negative on error.
#[no_mangle]
pub extern "C" fn nsl_serve_advance_grammar(request_id: i64, token_id: i64) -> i64 {
    let mut guard = match SERVE_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let ctx = match guard.as_mut() {
        Some(c) => c,
        None => return -2,
    };

    if let Some(req) = ctx.scheduler.active.iter_mut().find(|r| r.id == request_id as u64) {
        if let Some(ref mut grammar_state) = req.grammar_state {
            if grammar_state.active {
                // Lock GRAMMAR_CTX directly — avoid FFI wrapper's redundant lock
                let grammar_guard = match crate::grammar::GRAMMAR_CTX.lock() {
                    Ok(g) => g,
                    Err(_) => return -1,
                };
                if let Some(ref grammar_ctx) = *grammar_guard {
                    if let Some(next) = grammar_ctx.fsm.step(grammar_state.current_state, token_id as u32) {
                        grammar_state.current_state = next;
                    } else {
                        grammar_state.active = false;
                    }
                }
            }
        }
    }
    0
}

/// Set grammar constraint on a request (called after enqueue).
/// `start_state`: the FSM start state for this grammar.
/// Returns 0 on success, -1 if request not found.
#[no_mangle]
pub extern "C" fn nsl_serve_set_grammar(request_id: i64, start_state: i64) -> i64 {
    let mut guard = match SERVE_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let ctx = match guard.as_mut() {
        Some(c) => c,
        None => return -2,
    };

    // Set grammar on waiting or active request
    for req in ctx.scheduler.waiting.iter_mut().chain(ctx.scheduler.active.iter_mut()) {
        if req.id == request_id as u64 {
            req.grammar_state = Some(crate::grammar::GrammarRequestState::new(start_state as u32));
            return 0;
        }
    }
    -1 // request not found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared test lock to prevent parallel tests from conflicting on the
    /// global SERVE_CTX and GRAMMAR_CTX mutexes.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        // Clean up any previous state
        nsl_serve_destroy();
        crate::grammar::nsl_grammar_destroy();
        guard
    }

    #[test]
    fn test_grammar_set_and_query() {
        let _lock = setup();

        // Initialize serving engine
        assert_eq!(nsl_serve_init(8, 4096, 256, 512), 0);
        // Initialize grammar FSM (3 states, 4 vocab, start=0)
        assert_eq!(crate::grammar::nsl_grammar_init(3, 4, 0), 0);

        // Enqueue a request
        let req_id = nsl_serve_enqueue(0, 0, 10, 0.7, 0.9);
        assert!(req_id >= 0);

        // Set grammar on the request
        assert_eq!(nsl_serve_set_grammar(req_id, 0), 0);

        // Verify grammar_state is set by checking the request
        let guard = SERVE_CTX.lock().unwrap();
        let ctx = guard.as_ref().unwrap();
        let req = ctx.scheduler.waiting.iter()
            .chain(ctx.scheduler.active.iter())
            .find(|r| r.id == req_id as u64)
            .unwrap();
        assert!(req.grammar_state.is_some());
        let gs = req.grammar_state.as_ref().unwrap();
        assert_eq!(gs.current_state, 0);
        assert!(gs.active);
        drop(guard);

        nsl_serve_destroy();
        crate::grammar::nsl_grammar_destroy();
    }

    #[test]
    fn test_apply_grammar_no_grammar() {
        let _lock = setup();

        assert_eq!(nsl_serve_init(8, 4096, 256, 512), 0);

        // Enqueue request without grammar
        let req_id = nsl_serve_enqueue(0, 0, 10, 0.7, 0.9);
        // Step to move request into active list
        nsl_serve_step();

        // Apply grammar to request with no grammar_state → returns 0 (no-op)
        assert_eq!(nsl_serve_apply_grammar(req_id, 0), 0);

        nsl_serve_destroy();
    }

    #[test]
    fn test_advance_grammar_no_grammar() {
        let _lock = setup();

        assert_eq!(nsl_serve_init(8, 4096, 256, 512), 0);

        // Enqueue request without grammar
        let req_id = nsl_serve_enqueue(0, 0, 10, 0.7, 0.9);
        nsl_serve_step();

        // Advance grammar on request with no grammar_state → returns 0 (no-op)
        assert_eq!(nsl_serve_advance_grammar(req_id, 42), 0);

        nsl_serve_destroy();
    }
}
