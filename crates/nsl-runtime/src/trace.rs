// M18b: Trace-based op recording for ONNX export

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────────────
// Global tracing state
// ─────────────────────────────────────────────────────────────────────────────

pub static TRACING: AtomicBool = AtomicBool::new(false);
static TRACE_GRAPH: Mutex<Option<TraceGraph>> = Mutex::new(None);

// Test-only sentinel: set to `true` only while a `trace::tests` test body
// is actively recording.  `record_op` checks this in `#[cfg(test)]` builds
// so that tensor-op tests running in parallel (arithmetic, activation, etc.)
// cannot inject spurious ops into the shared TRACE_GRAPH and cause
// "expected N ops, got N+1" failures.
//
// **THREAD-LOCAL** — `nsl_trace_start()` sets the global `TRACING` flag to
// true, which means every test thread sees `is_tracing() == true` for the
// duration of any trace test's recording window. A global `AtomicBool` here
// was insufficient: the trace-test thread set it true, then parallel
// tensor-op test threads also saw it true and pushed stray ops into
// `TRACE_GRAPH`. Per-thread state means parallel threads always observe
// their own `false` initial value regardless of what the trace-test thread
// has set on its own thread.
#[cfg(test)]
thread_local! {
    pub(crate) static TRACE_TEST_RECORDING: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

pub type TraceNodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Transpose,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    LayerNorm,
    Reshape,
    Unsqueeze,
    Expand,
    Concat,
    Gather,
    Neg,
    Exp,
    Log,
    Sqrt,
}

#[derive(Debug, Clone)]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Ints(Vec<i64>),
}

#[derive(Debug, Clone)]
pub struct TraceOp {
    pub op_type: OpType,
    pub input_ptrs: Vec<i64>,
    pub output_ptr: i64,
    pub output_shape: Vec<i64>,
    pub output_dtype: u16,
    pub attributes: Vec<(String, AttrValue)>,
}

pub struct TraceGraph {
    pub ops: Vec<TraceOp>,
    pub inputs: Vec<(i64, String)>,
    pub outputs: Vec<(i64, String)>,
    pub ptr_to_node: HashMap<i64, TraceNodeId>,
}

impl TraceGraph {
    fn new() -> Self {
        TraceGraph {
            ops: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            ptr_to_node: HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public helpers (called by other runtime modules)
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether tracing is currently active.
#[inline]
pub fn is_tracing() -> bool {
    TRACING.load(Ordering::Relaxed)
}

/// Record a tensor operation into the active trace graph.
/// No-op if tracing is not active.
pub fn record_op(
    op_type: OpType,
    input_ptrs: Vec<i64>,
    output_ptr: i64,
    output_shape: Vec<i64>,
    output_dtype: u16,
    attributes: Vec<(String, AttrValue)>,
) {
    if !is_tracing() {
        return;
    }
    // In test builds, only record when a trace::tests body explicitly opted in.
    // This blocks stray calls from parallel tensor-op tests (arithmetic,
    // activation, etc.) that happen to run while TRACING=true. The flag is
    // thread-local — only the thread that called `TRACE_TEST_RECORDING.set(true)`
    // sees `true`; every other test thread sees its own initial `false`.
    #[cfg(test)]
    if !TRACE_TEST_RECORDING.with(|c| c.get()) {
        return;
    }
    let mut guard = TRACE_GRAPH.lock().expect("TRACE_GRAPH mutex poisoned");
    if let Some(graph) = guard.as_mut() {
        let node_id = graph.ops.len();
        graph.ops.push(TraceOp {
            op_type,
            input_ptrs,
            output_ptr,
            output_shape,
            output_dtype,
            attributes,
        });
        graph.ptr_to_node.insert(output_ptr, node_id);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFI functions
// ─────────────────────────────────────────────────────────────────────────────

/// Start tracing: set TRACING=true and initialise a fresh TraceGraph.
#[no_mangle]
pub extern "C" fn nsl_trace_start() {
    let mut guard = TRACE_GRAPH.lock().expect("TRACE_GRAPH mutex poisoned");
    *guard = Some(TraceGraph::new());
    // Enable tracing after the graph is initialised so that any concurrent
    // callers that race on the flag will see a valid graph.
    TRACING.store(true, Ordering::SeqCst);
}

/// Register a tensor pointer + name as a graph input.
///
/// `name_ptr` must point to a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub extern "C" fn nsl_trace_register_input(tensor_ptr: i64, name_ptr: i64) {
    let name = unsafe {
        let raw = name_ptr as *const u8;
        let mut len = 0usize;
        while *raw.add(len) != 0 {
            len += 1;
        }
        std::str::from_utf8(std::slice::from_raw_parts(raw, len))
            .unwrap_or("<invalid utf8>")
            .to_owned()
    };
    let mut guard = TRACE_GRAPH.lock().expect("TRACE_GRAPH mutex poisoned");
    if let Some(graph) = guard.as_mut() {
        graph.inputs.push((tensor_ptr, name));
    }
}

/// Register a tensor pointer + name as a graph output.
///
/// `name_ptr` must point to a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub extern "C" fn nsl_trace_register_output(tensor_ptr: i64, name_ptr: i64) {
    let name = unsafe {
        let raw = name_ptr as *const u8;
        let mut len = 0usize;
        while *raw.add(len) != 0 {
            len += 1;
        }
        std::str::from_utf8(std::slice::from_raw_parts(raw, len))
            .unwrap_or("<invalid utf8>")
            .to_owned()
    };
    let mut guard = TRACE_GRAPH.lock().expect("TRACE_GRAPH mutex poisoned");
    if let Some(graph) = guard.as_mut() {
        graph.outputs.push((tensor_ptr, name));
    }
}

/// Stop tracing: set TRACING=false, take the graph out of the Mutex, box it,
/// and return its raw pointer as an i64.  Returns 0 if no trace was active.
#[no_mangle]
pub extern "C" fn nsl_trace_stop() -> i64 {
    // Disable tracing first so concurrent calls stop recording.
    TRACING.store(false, Ordering::SeqCst);
    let mut guard = TRACE_GRAPH.lock().expect("TRACE_GRAPH mutex poisoned");
    match guard.take() {
        Some(graph) => Box::into_raw(Box::new(graph)) as i64,
        None => 0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialise all trace tests because they share global mutable state.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    /// Helper: reset global state between tests so they don't interfere.
    /// Recovers from a poisoned `TRACE_GRAPH` mutex so that a prior test
    /// panicking while holding the lock doesn't cascade into the next test.
    fn reset_trace() {
        TRACING.store(false, Ordering::SeqCst);
        let mut guard = match TRACE_GRAPH.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = None;
    }

    #[test]
    fn test_trace_basic_ops() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_trace();
        TRACE_TEST_RECORDING.with(|c| c.set(true));

        // Start trace
        nsl_trace_start();
        assert!(is_tracing(), "tracing should be active after nsl_trace_start");

        // Register input: fake tensor pointer 100, name "input"
        let input_name = b"input\0";
        nsl_trace_register_input(100, input_name.as_ptr() as i64);

        // Record op 1: Add(100, 200) -> 300
        record_op(
            OpType::Add,
            vec![100, 200],
            300,
            vec![2, 3],
            0, // f64
            vec![],
        );

        // Record op 2: MatMul(300, 400) -> 500
        record_op(
            OpType::MatMul,
            vec![300, 400],
            500,
            vec![2, 4],
            0,
            vec![("transpose_b".to_string(), AttrValue::Int(1))],
        );

        // Register output: tensor 500, name "output"
        let output_name = b"output\0";
        nsl_trace_register_output(500, output_name.as_ptr() as i64);

        // Stop trace and recover the graph
        let raw = nsl_trace_stop();
        assert!(!is_tracing(), "tracing should be inactive after nsl_trace_stop");
        assert_ne!(raw, 0, "nsl_trace_stop should return a non-null pointer");

        TRACE_TEST_RECORDING.with(|c| c.set(false));
        let graph = unsafe { Box::from_raw(raw as *mut TraceGraph) };

        // Verify ops
        assert_eq!(graph.ops.len(), 2, "expected 2 recorded ops");
        assert_eq!(graph.ops[0].op_type, OpType::Add);
        assert_eq!(graph.ops[0].input_ptrs, vec![100, 200]);
        assert_eq!(graph.ops[0].output_ptr, 300);
        assert_eq!(graph.ops[0].output_shape, vec![2, 3]);
        assert_eq!(graph.ops[1].op_type, OpType::MatMul);
        assert_eq!(graph.ops[1].output_shape, vec![2, 4]);
        assert_eq!(graph.ops[1].attributes.len(), 1);

        // Verify inputs
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].0, 100);
        assert_eq!(graph.inputs[0].1, "input");

        // Verify outputs
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].0, 500);
        assert_eq!(graph.outputs[0].1, "output");

        // Verify pointer→node mapping
        assert_eq!(graph.ptr_to_node.get(&300), Some(&0));
        assert_eq!(graph.ptr_to_node.get(&500), Some(&1));
    }

    #[test]
    fn test_trace_pointer_resolution() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_trace();
        TRACE_TEST_RECORDING.with(|c| c.set(true));

        nsl_trace_start();

        // Input tensor at ptr 10
        let input_name = b"x\0";
        nsl_trace_register_input(10, input_name.as_ptr() as i64);

        // op0: add(10, 20) -> 30  (ptr 20 is a weight / initializer, not a computed node)
        record_op(OpType::Add, vec![10, 20], 30, vec![4], 0, vec![]);

        // op1: relu(30) -> 40
        record_op(OpType::Relu, vec![30], 40, vec![4], 0, vec![]);

        // Output is ptr 40
        let output_name = b"y\0";
        nsl_trace_register_output(40, output_name.as_ptr() as i64);

        let raw = nsl_trace_stop();
        TRACE_TEST_RECORDING.with(|c| c.set(false));
        let graph = unsafe { Box::from_raw(raw as *mut TraceGraph) };

        // ptr 10 is in inputs
        let input_ptrs: Vec<i64> = graph.inputs.iter().map(|(p, _)| *p).collect();
        assert!(input_ptrs.contains(&10), "ptr 10 should be registered as input");

        // ptr 20 is NOT in ptr_to_node — it's an initializer/weight, not a computed output
        assert!(
            !graph.ptr_to_node.contains_key(&20),
            "ptr 20 (weight) should NOT be in ptr_to_node"
        );

        // ptr 30 maps to node 0 (the Add op)
        assert_eq!(
            graph.ptr_to_node.get(&30),
            Some(&0),
            "ptr 30 should map to node 0 (Add)"
        );

        // ptr 40 maps to node 1 (the Relu op)
        assert_eq!(
            graph.ptr_to_node.get(&40),
            Some(&1),
            "ptr 40 should map to node 1 (Relu)"
        );
    }

    // Regression: pins that `TRACE_TEST_RECORDING` is genuinely thread-local.
    // A worker thread spawned while the main thread has set its own cell to
    // `true` MUST observe its own per-thread initial `false`. If this ever
    // regresses to a global (e.g. someone "consolidates" the state back to
    // an `AtomicBool`), parallel tensor-op tests will silently push stray
    // entries into the shared `TRACE_GRAPH` during a trace test's recording
    // window — the same bug this fix closes.
    #[test]
    fn trace_test_recording_is_thread_local() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_trace();
        TRACE_TEST_RECORDING.with(|c| c.set(true));
        assert!(TRACE_TEST_RECORDING.with(|c| c.get()), "main thread set it true");

        let observed_in_worker = std::thread::spawn(|| {
            TRACE_TEST_RECORDING.with(|c| c.get())
        })
        .join()
        .expect("worker thread must not panic");

        assert!(
            !observed_in_worker,
            "TRACE_TEST_RECORDING must be thread-local — a parallel thread MUST observe its own initial false"
        );
        TRACE_TEST_RECORDING.with(|c| c.set(false));
    }

    // End-to-end regression: while a trace test holds the recording window
    // open on the main thread (TRACING=true globally + TRACE_TEST_RECORDING
    // true thread-locally), a parallel thread calling `record_op` must NOT
    // pollute `TRACE_GRAPH`. This is the exact contamination path the
    // pre-fix global AtomicBool let through.
    #[test]
    fn parallel_thread_record_op_does_not_pollute_trace_graph() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_trace();
        TRACE_TEST_RECORDING.with(|c| c.set(true));
        nsl_trace_start();

        // Worker thread simulates a parallel tensor-op test. `TRACING` is
        // true globally (set by `nsl_trace_start()` above), but the worker's
        // own `TRACE_TEST_RECORDING` is its initial false — so `record_op`
        // must early-return without touching TRACE_GRAPH.
        let worker_saw_recording = std::thread::spawn(|| {
            record_op(OpType::Add, vec![1, 2], 3, vec![4], 0, vec![]);
            TRACE_TEST_RECORDING.with(|c| c.get())
        })
        .join()
        .expect("worker thread must not panic");
        assert!(
            !worker_saw_recording,
            "worker thread must observe its own thread-local default false (was true → would re-introduce the parallel contamination bug)"
        );

        // Main thread DOES record its own op — its thread-local is true.
        record_op(OpType::Mul, vec![10, 20], 30, vec![4], 0, vec![]);

        let raw = nsl_trace_stop();
        TRACE_TEST_RECORDING.with(|c| c.set(false));
        let graph = unsafe { Box::from_raw(raw as *mut TraceGraph) };

        // Exactly one op — only the main thread's Mul. The worker's Add was
        // filtered by the thread-local check.
        assert_eq!(
            graph.ops.len(),
            1,
            "worker-thread record_op must NOT have polluted TRACE_GRAPH (got {} ops, expected 1)",
            graph.ops.len()
        );
        assert_eq!(graph.ops[0].op_type, OpType::Mul);
    }
}
