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
