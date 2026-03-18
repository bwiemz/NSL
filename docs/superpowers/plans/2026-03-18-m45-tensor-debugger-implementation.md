# M45: Time-Travel Tensor Debugger — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trace-record-and-replay debugger for tensor computations. `--trace` instruments every tensor op to capture shapes, dtypes, and per-tensor statistics (min/max/mean/std) in a compact binary trace. The trace enables post-mortem NaN/Inf root-cause detection and trace-vs-trace diffing. Also add a compile-time NaN risk analysis pass.

**Architecture:** Create `crates/nsl-runtime/src/tensor_trace.rs` (new module, alongside the existing `trace.rs` which handles ONNX export tracing) with `TraceRecorder`, `TraceEntry` binary format, and FFI functions. All FFI functions use `i64` parameters matching the Cranelift calling convention — NOT the spec's `u16`/`u32` types. Create `crates/nsl-semantic/src/nan_analysis.rs` for compile-time NaN risk detection. Add `@no_trace`/`@trace_breakpoint` decorator validation. Add trace diffing and Chrome export utilities.

**Tech Stack:** Rust (runtime trace recording + semantic analysis + CLI)

**Spec:** `docs/superpowers/specs/2026-03-15-m45-tensor-debugger-design.md`

**Prerequisites:** None (standalone)

---

## Important: Scope of This Plan

**This plan builds the trace recording infrastructure + NaN analysis + trace utilities.** It delivers:
- `TraceEntry` — 124-byte fixed-size binary format with per-op stats
- `TraceRecorder` — thread-local recorder with NaN/Inf sentinel detection
- `TraceHeader` — binary trace file header (magic "NSLT", version, timestamp, op count)
- FFI functions: `nsl_trace_init`, `nsl_trace_record_op`, `nsl_trace_suppress/unsuppress`, `nsl_trace_breakpoint`, `nsl_trace_flush`
- `NanAnalyzer` — compile-time NaN risk detection (log, div, sqrt, pow patterns)
- `ValueConstraint` tracking (Unconstrained, NonNegative, StrictlyPositive)
- Trace diffing: `find_first_divergence` for two traces
- Chrome tracing export: trace → JSON for `chrome://tracing`
- `@no_trace` and `@trace_breakpoint` semantic validation
- `--trace`, `--nan-analysis` CLI flags and CompileOptions fields
- 7 builtin FFI registrations (all i64 params per Cranelift convention)
- 17+ unit tests

**Deferred to M45b:** Interactive TUI debugger (`nsl debug` with ratatui), GPU stats reduction kernel (PTX for device tensor stats), codegen instrumentation pass (wrapping FFI calls with `nsl_trace_record_op`), backward chain reconstruction for NaN root-cause in TUI, `--trace-output` custom path flag, high-rank shape extension (ndim > 4).

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/tensor_trace.rs` | `TraceRecorder`, `TraceEntry`, `TraceHeader`, recording FFI, binary format I/O | 300 |
| `crates/nsl-runtime/src/trace_diff.rs` | Trace diffing + Chrome tracing export | 120 |
| `crates/nsl-semantic/src/nan_analysis.rs` | Compile-time NaN risk detection with `ValueConstraint` | 150 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod tensor_trace; pub mod trace_diff;` |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod nan_analysis;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@no_trace`, `@trace_breakpoint` validation, NaN analysis |
| `crates/nsl-codegen/src/builtins.rs` | Register 6 trace FFI functions |
| `crates/nsl-codegen/src/lib.rs` | Add `trace` and `nan_analysis` to CompileOptions |
| `crates/nsl-cli/src/main.rs` | Add `--trace`, `--nan-analysis` flags |

---

## Phase 1: Binary Trace Format + Recorder

### Task 1: TraceEntry + TraceRecorder + FFI

**Files:**
- Create: `crates/nsl-runtime/src/tensor_trace.rs`

- [ ] **Step 1: Create tensor_trace.rs with trace format, recorder, and FFI**

The TraceEntry is a 124-byte fixed-size struct for O(1) random access. The TraceRecorder is a thread-local singleton that records ops and detects NaN/Inf.

Key types:
- `TraceEntry` — #[repr(C, packed)] with op_id, op_type, flags, timestamp, 3x tensor stats (in0, in1, out)
- `TensorStats` — ndim, dtype, device, shape[4], min, max, mean, std
- `TraceHeader` — magic "NSLT", version, timestamp, num_ops
- `TraceRecorder` — entries vec, start_time, active flag, suppress_depth (for @no_trace nesting)

FFI functions:
- `nsl_trace_init()` → initialize recorder
- `nsl_trace_record_op(op_type, in0_ptr, in1_ptr, out_ptr)` → record one op, return 0 or 1 (NaN break)
- `nsl_trace_suppress()` / `nsl_trace_unsuppress()` → @no_trace scope management
- `nsl_trace_breakpoint()` → mark breakpoint in trace
- `nsl_trace_flush()` → write trace to disk
- `nsl_trace_destroy()` → cleanup

Tests (8):
- `trace_entry_size` — assert `size_of::<TraceEntry>()` is correct
- `trace_header_roundtrip` — write header, read back, verify
- `trace_record_basic` — record 3 ops, verify entry count
- `trace_nan_detection` — record op with NaN output, verify flag set
- `trace_suppress_unsuppress` — suppress blocks recording
- `trace_suppress_nesting` — nested suppress/unsuppress counts correctly
- `trace_breakpoint_flag` — breakpoint sets flag bit
- `ffi_lifecycle` — init/flush/destroy

### Task 2: Trace Diffing + Chrome Export

**Files:**
- Create: `crates/nsl-runtime/src/trace_diff.rs`

- [ ] **Step 2: Create trace_diff.rs with diffing algorithm and Chrome export**

```rust
//! M45: Trace diffing and Chrome tracing export.

use crate::tensor_trace::TraceEntry;

/// Result of comparing two traces.
#[derive(Debug)]
pub enum DiffResult {
    /// Traces are identical within threshold.
    Identical,
    /// Op type sequence diverges at the given positions.
    OpMismatch { pos_a: usize, pos_b: usize },
    /// Stats diverge beyond threshold at the given op.
    StatsDiverge { op_id: usize, delta_mean: f32, delta_max: f32 },
    /// Traces have different lengths.
    LengthMismatch { len_a: usize, len_b: usize },
}

/// Find the first point of divergence between two traces.
pub fn find_first_divergence(
    a: &[TraceEntry],
    b: &[TraceEntry],
    threshold: f32,
) -> DiffResult {
    if a.len() != b.len() {
        return DiffResult::LengthMismatch { len_a: a.len(), len_b: b.len() };
    }
    for i in 0..a.len() {
        if a[i].op_type != b[i].op_type {
            return DiffResult::OpMismatch { pos_a: i, pos_b: i };
        }
        let delta_mean = (a[i].out_mean - b[i].out_mean).abs();
        let delta_max = (a[i].out_max - b[i].out_max).abs();
        if delta_mean > threshold || delta_max > threshold {
            return DiffResult::StatsDiverge { op_id: i, delta_mean, delta_max };
        }
    }
    DiffResult::Identical
}

/// Export a trace to Chrome Trace Event Format JSON.
pub fn export_chrome_json(entries: &[TraceEntry], op_names: &[&str]) -> String {
    let mut events = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let name = if (entry.op_type as usize) < op_names.len() {
            op_names[entry.op_type as usize]
        } else {
            "unknown"
        };
        let dur = if i + 1 < entries.len() {
            entries[i + 1].timestamp_ns - entry.timestamp_ns
        } else {
            0
        };
        // Guard NaN/Inf → 0.0 for valid JSON (NaN is not valid JSON)
        let safe = |v: f32| -> f32 { if v.is_finite() { v } else { 0.0 } };
        events.push(format!(
            r#"{{"name":"{}","cat":"tensor_op","ph":"X","ts":{},"dur":{},"pid":0,"tid":0,"args":{{"op_id":{},"out_min":{},"out_max":{},"out_mean":{},"out_std":{},"has_nan":{}}}}}"#,
            name, entry.timestamp_ns / 1000, dur / 1000,
            entry.op_id, safe(entry.out_min), safe(entry.out_max),
            safe(entry.out_mean), safe(entry.out_std),
            entry.flags & 0x01 != 0,
        ));
    }
    format!(r#"{{"traceEvents":[{}]}}"#, events.join(","))
}
```

Tests (4):
- `diff_identical_traces`
- `diff_op_mismatch`
- `diff_stats_diverge`
- `chrome_export_format`

---

## Phase 2: NaN Analysis + Semantic + CLI

### Task 3: Compile-Time NaN Risk Analysis

**Files:**
- Create: `crates/nsl-semantic/src/nan_analysis.rs`

- [ ] **Step 3: Create nan_analysis.rs with ValueConstraint tracking and risk pattern detection**

```rust
//! M45: Compile-time NaN risk analysis.

use std::collections::HashMap;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Known value constraint for a tensor binding.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueConstraint {
    Unconstrained,
    NonNegative,        // >= 0 (from relu, abs, x*x)
    StrictlyPositive,   // > 0  (from relu(x) + eps, softplus, exp)
}

/// Analyzes function bodies for NaN/Inf risk patterns.
pub struct NanAnalyzer {
    constraints: HashMap<Symbol, ValueConstraint>,
    pub diagnostics: Vec<Diagnostic>,
}

impl NanAnalyzer {
    pub fn new() -> Self {
        NanAnalyzer { constraints: HashMap::new(), diagnostics: Vec::new() }
    }

    /// Mark a binding as having a known constraint.
    pub fn set_constraint(&mut self, sym: Symbol, constraint: ValueConstraint) {
        self.constraints.insert(sym, constraint);
    }

    /// Get the constraint for a binding.
    pub fn get_constraint(&self, sym: &Symbol) -> ValueConstraint {
        self.constraints.get(sym).cloned().unwrap_or(ValueConstraint::Unconstrained)
    }

    /// Check if a function call is a NaN risk given its argument constraints.
    ///
    /// Returns a warning diagnostic if a risk is detected, None otherwise.
    pub fn check_call_risk(
        &self,
        func_name: &str,
        arg_syms: &[Symbol],
        span: nsl_errors::Span,
    ) -> Option<Diagnostic> {
        match func_name {
            "log" => {
                if let Some(arg) = arg_syms.first() {
                    let c = self.get_constraint(arg);
                    if c != ValueConstraint::StrictlyPositive {
                        return Some(
                            Diagnostic::warning("log() argument may be zero or negative — NaN risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            "sqrt" => {
                if let Some(arg) = arg_syms.first() {
                    let c = self.get_constraint(arg);
                    if c == ValueConstraint::Unconstrained {
                        return Some(
                            Diagnostic::warning("sqrt() argument may be negative — NaN risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            // Division: second argument could be zero → Inf
            "div" => {
                if arg_syms.len() >= 2 {
                    let c = self.get_constraint(&arg_syms[1]);
                    if c != ValueConstraint::StrictlyPositive {
                        return Some(
                            Diagnostic::warning("division by tensor that may contain zeros — Inf risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            // pow and log(softmax(...)) detection deferred to M45b
            _ => None,
        }
    }

    /// Infer constraints from known function results.
    pub fn infer_constraint(&self, func_name: &str) -> ValueConstraint {
        match func_name {
            "relu" | "abs" => ValueConstraint::NonNegative,
            "exp" | "softplus" => ValueConstraint::StrictlyPositive,
            "sigmoid" => ValueConstraint::StrictlyPositive, // (0, 1)
            _ => ValueConstraint::Unconstrained,
        }
    }
}
```

Tests (5):
- `log_unconstrained_warns`
- `log_strictly_positive_no_warn`
- `sqrt_unconstrained_warns`
- `sqrt_non_negative_no_warn`
- `relu_produces_non_negative`

### Task 4: Semantic Validation + CLI + Builtins

**Files:**
- Modify: semantic checker, CLI, CompileOptions, builtins

- [ ] **Step 4: Wire @no_trace/@trace_breakpoint validation, add CLI flags, register FFI**

Semantic: validate `@no_trace` and `@trace_breakpoint` target FnDef only.

CLI flags on `Run` and `Check`:
- `--trace` (Run only)
- `--nan-analysis` (Check only, or standalone)

CompileOptions: add `trace: bool` and `nan_analysis: bool`.

Builtins: register 7 FFI functions (init, record_op, suppress, unsuppress, breakpoint, flush, destroy). All use `i64` params per Cranelift convention — NOT the spec's u16/u32 types.

---

## Phase 3: Build Verification

- [ ] **Step 5: `cargo build`**
- [ ] **Step 6: `cargo test` — expect 17+ new tests**
- [ ] **Step 7: `cargo clippy`**

---

## Verification Checklist

1. **TraceEntry size**: Fixed at expected byte count for O(1) random access
2. **NaN detection**: Output with NaN sets flag and returns Break action
3. **Suppress nesting**: Nested @no_trace scopes count correctly
4. **Trace diff**: Identical traces → Identical, divergent stats → StatsDiverge
5. **Chrome export**: Valid JSON with traceEvents array
6. **NaN analysis**: `log(unconstrained)` warns, `log(exp(x))` doesn't
7. **Constraint inference**: relu→NonNegative, exp→StrictlyPositive
8. **FFI lifecycle**: init/record/flush/destroy
9. **No regressions**: All 633+ existing tests pass
