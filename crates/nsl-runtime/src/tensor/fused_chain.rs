//! Fused elementwise-chain FFI (mfu-fusion C3).
//!
//! `nsl_fused_ew_chain` executes a short adjoint elementwise chain that the
//! compiler's `ew_chain_fusion` peephole collapsed into one Passthrough op.
//! The compile side ships three `.rodata` blobs per unique chain signature —
//! a synthesized PTX module, its kernel entry name, and a byte DESCRIPTOR —
//! and this FFI decides per call which of two paths runs:
//!
//!  * **Fast path** (GPU): every input is a non-null, GPU-resident,
//!    contiguous f32 tensor of one uniform shape (including every RtsCheck
//!    like-reference — the shape a `reduce_to_shape` member would target, so
//!    the reduce is an identity and the kernel can skip it). One launch of
//!    the synthesized kernel replaces the whole chain.
//!  * **Decomposed replay** (any gate miss — this is also the CPU arm): the
//!    descriptor is interpreted by calling the ordinary public FFIs
//!    (`nsl_tensor_add/sub/mul/div/neg`, `nsl_tensor_scalar`,
//!    `nsl_tensor_reduce_to_shape`) in original tape order. This executes
//!    literally the ops the compiler removed, through the same
//!    reconcile/broadcast machinery, so it is BIT-EXACT to the decomposed
//!    baseline by construction.
//!
//! # Descriptor v1 (PINNED CONTRACT with `nsl-codegen`'s `ew_chain_fusion`)
//!
//! ```text
//! [u8 version=1][u8 n_steps][u8 n_inputs][u8 flags=0]
//! then per step, 9 bytes:
//! [u8 opcode][u8 lhs_kind][u8 lhs_idx][u8 rhs_kind][u8 rhs_idx][u32 imm_bits LE]
//! ```
//!
//! * opcode: Add=0, Sub=1, Mul=2, Div=3, Neg=4, RtsCheck=5.
//!   Neg has no rhs (`rhs_kind = 255`). RtsCheck: lhs = the value flowing
//!   through (Prev of the previous step, or an Input); rhs = the like-ref
//!   tensor (Input) whose SHAPE the reduce would target. On the fast path an
//!   RtsCheck is an identity (the kernel emits nothing for it).
//! * operand kind: Input=0 (idx into the <=6 tensor handles), Prev=1
//!   (idx = the step index whose result it consumes), Imm=2 (an f32 from
//!   `imm_bits`; idx unused). kind 255 = absent. `imm_bits` is meaningful
//!   only when a kind is Imm.
//! * The final step's result is the chain output.
//!
//! A malformed descriptor is a compiler bug, never a data condition — it
//! aborts loudly (deferral-must-refuse), it does not degrade.
//!
//! # Ownership
//!
//! The FFI allocates only its result and NEVER frees its input handles. The
//! replay frees exactly the interiors it created, as it goes. No DtoH, no
//! ctx-sync, and no transfer-stream interaction happen on the fast path.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::autodiff;

use super::ad_ops::{nsl_tensor_reduce_to_shape, nsl_tensor_scalar};
use super::arithmetic::{
    nsl_tensor_add, nsl_tensor_div, nsl_tensor_mul, nsl_tensor_neg, nsl_tensor_sub,
};
use super::nsl_tensor_free;
// Only the cuda fast-path gate and the crate-internal tests touch the
// struct itself; the replay works purely through i64-handle FFIs.
#[cfg(any(feature = "cuda", test))]
use super::NslTensor;

// --- Descriptor v1 wire constants (pinned; see module doc) ---
const DESC_VERSION: u8 = 1;
const OP_ADD: u8 = 0;
const OP_SUB: u8 = 1;
const OP_MUL: u8 = 2;
const OP_DIV: u8 = 3;
const OP_NEG: u8 = 4;
const OP_RTS_CHECK: u8 = 5;
const KIND_INPUT: u8 = 0;
const KIND_PREV: u8 = 1;
const KIND_IMM: u8 = 2;
const KIND_ABSENT: u8 = 255;
/// Header bytes before the per-step records.
const DESC_HEADER_LEN: usize = 4;
/// Bytes per step record.
const DESC_STEP_LEN: usize = 9;
/// FFI arity cap — doubles as the compile-side profitability cap.
const MAX_INPUTS: usize = 6;

/// Anti-vacuity counters, same family as `WGRAD_FUSED_COUNT` /
/// `WGRAD_FALLBACK_COUNT` in `arithmetic.rs`: the replay fallback is
/// warn-once (quiet after the first), so without counters a green parity
/// gate could not distinguish "the fused kernel agrees with the chain" from
/// "the fused kernel never ran". Always live (one relaxed atomic per chain
/// call); `NSL_FUSED_EW_COUNTER=1` / `NSL_EVENTS` only gate the report.
pub static FUSED_EW_LAUNCHES: AtomicU64 = AtomicU64::new(0);
pub static FUSED_EW_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// In-process getters (same family as `nsl_wgrad_fused_count`).
#[no_mangle]
pub extern "C" fn nsl_fused_ew_launch_count() -> i64 {
    FUSED_EW_LAUNCHES.load(Ordering::Relaxed) as i64
}

#[no_mangle]
pub extern "C" fn nsl_fused_ew_fallback_count() -> i64 {
    FUSED_EW_FALLBACKS.load(Ordering::Relaxed) as i64
}

/// Teardown counter report: NSL_EVENTS machine twin + gated human stderr
/// line, mirroring `nsl_wgrad_count_atexit` in `args.rs` exactly.
/// Registered as an atexit hook from `nsl_args_init` (see args.rs); also
/// exported so codegen-emitted teardown code can call it directly.
#[no_mangle]
pub extern "C" fn nsl_fused_ew_counters_report() {
    let launches = nsl_fused_ew_launch_count();
    let fallbacks = nsl_fused_ew_fallback_count();
    crate::events::emit(
        "fused_ew_counters",
        None,
        &[
            ("launches", crate::events::i(launches)),
            ("fallbacks", crate::events::i(fallbacks)),
        ],
    );
    if std::env::var("NSL_FUSED_EW_COUNTER").ok().as_deref() == Some("1") {
        eprintln!("[fused-ew] fused launches: {launches}, decomposed fallbacks: {fallbacks}");
    }
}

/// One decoded descriptor step (wire layout in the module doc).
#[derive(Clone, Copy)]
struct ChainStep {
    opcode: u8,
    lhs_kind: u8,
    lhs_idx: u8,
    rhs_kind: u8,
    rhs_idx: u8,
    imm_bits: u32,
}

fn desc_abort(msg: &str) -> ! {
    eprintln!(
        "nsl_fused_ew_chain: malformed descriptor: {msg} \
         (descriptor v1 is a pinned compiler contract — this is a codegen \
         bug, not a data condition)"
    );
    std::process::abort();
}

/// Parse + validate a v1 descriptor. Returns the steps and `n_inputs`.
/// Every malformation aborts loudly — the descriptor is compiler-emitted,
/// so degrading here would hide a codegen bug behind wrong numerics.
fn parse_descriptor(desc: *const u8, desc_len: u64) -> (Vec<ChainStep>, usize) {
    if desc.is_null() {
        desc_abort("null descriptor pointer");
    }
    let desc_len = desc_len as usize;
    if desc_len < DESC_HEADER_LEN {
        desc_abort("shorter than the 4-byte header");
    }
    // Bound BEFORE building the slice: n_steps is a u8, so a valid v1
    // descriptor can never exceed 4 + 9*255 bytes — a larger desc_len is a
    // corrupt argument, and materializing a slice of that length first
    // would be unsound.
    if desc_len > DESC_HEADER_LEN + 255 * DESC_STEP_LEN {
        desc_abort("desc_len exceeds the v1 maximum (4 + 9*255)");
    }
    let bytes = unsafe { std::slice::from_raw_parts(desc, desc_len) };
    if bytes[0] != DESC_VERSION {
        desc_abort("unknown version (expected 1)");
    }
    let n_steps = bytes[1] as usize;
    let n_inputs = bytes[2] as usize;
    if bytes[3] != 0 {
        desc_abort("nonzero flags byte (v1 defines none)");
    }
    if n_steps == 0 {
        desc_abort("zero steps");
    }
    if n_inputs == 0 || n_inputs > MAX_INPUTS {
        desc_abort("n_inputs outside 1..=6");
    }
    if desc_len != DESC_HEADER_LEN + n_steps * DESC_STEP_LEN {
        desc_abort("desc_len does not match 4 + 9*n_steps");
    }

    let validate_operand = |kind: u8, idx: u8, step: usize, role: &str| match kind {
        KIND_INPUT => {
            if (idx as usize) >= n_inputs {
                desc_abort(&format!("step {step}: {role} Input idx out of range"));
            }
        }
        KIND_PREV => {
            if (idx as usize) >= step {
                desc_abort(&format!("step {step}: {role} Prev idx not an earlier step"));
            }
        }
        KIND_IMM => {}
        _ => desc_abort(&format!("step {step}: {role} has invalid operand kind")),
    };

    let mut steps = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let off = DESC_HEADER_LEN + i * DESC_STEP_LEN;
        let st = ChainStep {
            opcode: bytes[off],
            lhs_kind: bytes[off + 1],
            lhs_idx: bytes[off + 2],
            rhs_kind: bytes[off + 3],
            rhs_idx: bytes[off + 4],
            imm_bits: u32::from_le_bytes([
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
                bytes[off + 8],
            ]),
        };
        match st.opcode {
            OP_ADD | OP_SUB | OP_MUL | OP_DIV => {
                validate_operand(st.lhs_kind, st.lhs_idx, i, "lhs");
                if st.rhs_kind == KIND_ABSENT {
                    desc_abort(&format!("step {i}: binary op with absent rhs"));
                }
                validate_operand(st.rhs_kind, st.rhs_idx, i, "rhs");
                if st.lhs_kind == KIND_IMM && st.rhs_kind == KIND_IMM {
                    // One imm_bits field per step — two Imm operands cannot
                    // be encoded, so seeing both is a corrupt record.
                    desc_abort(&format!("step {i}: both operands Imm"));
                }
            }
            OP_NEG => {
                validate_operand(st.lhs_kind, st.lhs_idx, i, "lhs");
                if st.rhs_kind != KIND_ABSENT {
                    desc_abort(&format!("step {i}: Neg with a present rhs"));
                }
            }
            OP_RTS_CHECK => {
                // lhs = the flowing value (Prev or Input); rhs = the
                // like-reference, which is always a real tensor Input.
                if st.lhs_kind != KIND_PREV && st.lhs_kind != KIND_INPUT {
                    desc_abort(&format!("step {i}: RtsCheck lhs must be Prev or Input"));
                }
                validate_operand(st.lhs_kind, st.lhs_idx, i, "lhs");
                if st.rhs_kind != KIND_INPUT {
                    desc_abort(&format!("step {i}: RtsCheck like-ref must be an Input"));
                }
                validate_operand(st.rhs_kind, st.rhs_idx, i, "rhs");
            }
            _ => desc_abort(&format!("step {i}: unknown opcode")),
        }
        steps.push(st);
    }
    (steps, n_inputs)
}

/// Warn-once when a chain call takes the decomposed replay. The replay is
/// bit-exact, so this is a performance note, not a correctness event —
/// per-call printing would flood a CPU run where EVERY chain replays.
fn warn_fallback_once(reason: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("[fused-ew] falling back to decomposed replay ({reason})");
    }
}

/// Fast-path admission gate. `None` = admitted; `Some(reason)` = replay.
/// Uniform shape across ALL inputs also covers every RtsCheck like-ref
/// (like-refs are ordinary input slots), which is what makes the kernel's
/// treat-RtsCheck-as-identity legal.
#[cfg(feature = "cuda")]
fn fast_path_reject_reason(
    inputs: &[&NslTensor],
    ptx: *const u8,
    kname: *const u8,
) -> Option<&'static str> {
    if ptx.is_null() || kname.is_null() {
        return Some("no embedded PTX/kernel name");
    }
    let first = inputs[0];
    if first.device == 0 {
        return Some("CPU-resident input");
    }
    for t in inputs {
        if t.device == 0 {
            return Some("CPU-resident input");
        }
        if t.device != first.device {
            return Some("inputs on different GPU devices");
        }
        if t.dtype != crate::tensor::DTYPE_F32 {
            return Some("non-f32 input");
        }
        if !t.is_contiguous() {
            return Some("non-contiguous input");
        }
        if !t.shape_eq(first) {
            return Some("non-uniform input shapes");
        }
    }
    None
}

/// Decomposed replay: interpret the descriptor through the ordinary public
/// FFIs in original tape order. Frees every interior it creates as soon as
/// its last use is behind us; never frees an input handle.
fn replay_decomposed(steps: &[ChainStep], inputs: &[i64]) -> i64 {
    // Remaining-use counts per step result, from the descriptor itself, so
    // interiors are released as the replay walks forward (bounding live
    // intermediates the same way the baseline's FreeTensor schedule did).
    let mut uses_left = vec![0u32; steps.len()];
    for st in steps {
        if st.lhs_kind == KIND_PREV {
            uses_left[st.lhs_idx as usize] += 1;
        }
        if st.rhs_kind == KIND_PREV {
            uses_left[st.rhs_idx as usize] += 1;
        }
    }

    let mut results: Vec<i64> = Vec::with_capacity(steps.len());
    let mut alive: Vec<bool> = Vec::with_capacity(steps.len());

    for st in steps {
        // Materialize an Imm exactly as baseline Constant lowering does:
        // `nsl_tensor_scalar(v as f64, 1)` — the f64 the compiler saw was
        // already narrowed to these f32 bits once, and `(bits as f64) as f32`
        // round-trips them, so the scalar tensor holds identical bytes.
        let mut imm_temp: i64 = 0;
        let mut resolve = |kind: u8, idx: u8| -> i64 {
            match kind {
                KIND_INPUT => inputs[idx as usize],
                KIND_PREV => results[idx as usize],
                KIND_IMM => {
                    let v = f32::from_bits(st.imm_bits);
                    imm_temp = nsl_tensor_scalar(v as f64, 1);
                    imm_temp
                }
                _ => desc_abort("operand kind resolved after validation changed"),
            }
        };

        let lhs = resolve(st.lhs_kind, st.lhs_idx);
        let out = match st.opcode {
            OP_NEG => nsl_tensor_neg(lhs),
            OP_RTS_CHECK => {
                let like = resolve(st.rhs_kind, st.rhs_idx);
                // The real reduce the baseline would have run. Same-shape
                // like-refs make this a retain-and-return identity — the
                // returned handle is still a reference we own and free.
                nsl_tensor_reduce_to_shape(lhs, like)
            }
            OP_ADD => nsl_tensor_add(lhs, resolve(st.rhs_kind, st.rhs_idx), 0),
            OP_SUB => nsl_tensor_sub(lhs, resolve(st.rhs_kind, st.rhs_idx), 0),
            OP_MUL => nsl_tensor_mul(lhs, resolve(st.rhs_kind, st.rhs_idx), 0),
            OP_DIV => nsl_tensor_div(lhs, resolve(st.rhs_kind, st.rhs_idx), 0),
            _ => desc_abort("opcode resolved after validation changed"),
        };
        if imm_temp != 0 {
            nsl_tensor_free(imm_temp);
        }
        // Release interiors whose last consumer just ran (flags=0 calls
        // never consume their operands, so the replay owns every interior
        // reference until it frees it here).
        for (kind, idx) in [(st.lhs_kind, st.lhs_idx), (st.rhs_kind, st.rhs_idx)] {
            if kind == KIND_PREV {
                let j = idx as usize;
                uses_left[j] -= 1;
                if uses_left[j] == 0 && alive[j] {
                    nsl_tensor_free(results[j]);
                    alive[j] = false;
                }
            }
        }
        results.push(out);
        alive.push(true);
    }

    // The final step's result is the chain output; any other still-alive
    // result (a never-consumed interior — descriptor-legal, if odd) is ours
    // to release.
    let last = results.len() - 1;
    for j in 0..last {
        if alive[j] {
            nsl_tensor_free(results[j]);
        }
    }
    results[last]
}

/// Execute one fused elementwise chain. See the module doc for the
/// descriptor contract and the fast-path/replay split.
///
/// Tensor handles ride as `i64` (`*mut NslTensor as i64`), the convention of
/// every tensor FFI in this crate; unused input slots are 0-padded by the
/// compile side and ignored here. The result is a fresh owned handle —
/// never null on success (all failure modes either abort or degrade to the
/// replay, which itself aborts rather than returning null).
#[no_mangle]
#[allow(clippy::too_many_arguments)] // pinned FFI contract: fixed 6-slot handle arity
pub extern "C" fn nsl_fused_ew_chain(
    ptx: *const u8,
    kname: *const u8,
    desc: *const u8,
    desc_len: u64,
    in0: i64,
    in1: i64,
    in2: i64,
    in3: i64,
    in4: i64,
    in5: i64,
    n_inputs: i64,
) -> i64 {
    // Adjoint-only op: the compiler emits it exclusively inside the source-AD
    // backward, which never runs under an armed tape. The replay path calls
    // recording-aware FFIs, so running it while recording would silently put
    // half a chain on the tape — refuse instead of guessing.
    assert!(
        !autodiff::is_recording(),
        "nsl_fused_ew_chain: called while the autodiff tape is recording; \
         this op is emitted only in the source-AD adjoint and has no tape rule"
    );

    let (steps, desc_n_inputs) = parse_descriptor(desc, desc_len);
    if n_inputs < 1 || n_inputs as usize > MAX_INPUTS {
        desc_abort("n_inputs argument outside 1..=6");
    }
    if n_inputs as usize != desc_n_inputs {
        desc_abort("n_inputs argument disagrees with the descriptor header");
    }
    let handles = [in0, in1, in2, in3, in4, in5];
    let inputs = &handles[..n_inputs as usize];
    // A null handle is a lowering bug (the compile side 0-pads only UNUSED
    // slots): neither path could execute it, so refuse loudly.
    if inputs.contains(&0) {
        desc_abort("null tensor handle within n_inputs");
    }

    #[cfg(feature = "cuda")]
    {
        let first: &NslTensor = unsafe { &*(inputs[0] as *const NslTensor) };
        let mut refs: [&NslTensor; MAX_INPUTS] = [first; MAX_INPUTS];
        for (k, &h) in inputs.iter().enumerate().skip(1) {
            refs[k] = unsafe { &*(h as *const NslTensor) };
        }
        let refs = &refs[..inputs.len()];
        match fast_path_reject_reason(refs, ptx, kname) {
            None => {
                let out = crate::cuda::gpu_fused_ew_launch(refs, ptx, kname);
                if out != 0 {
                    FUSED_EW_LAUNCHES.fetch_add(1, Ordering::Relaxed);
                    return out;
                }
                // Allocation/launch failure degrades to the bit-exact replay
                // rather than aborting mid-backward.
                warn_fallback_once("GPU alloc or launch failed");
            }
            Some(reason) => warn_fallback_once(reason),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx, kname);
        warn_fallback_once("runtime built without the cuda feature");
    }

    FUSED_EW_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    replay_decomposed(&steps, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a v1 descriptor from (opcode, lhs_kind, lhs_idx, rhs_kind,
    /// rhs_idx, imm_bits) tuples — the same byte layout the compile side
    /// emits (kept in duplicate here on purpose: a layout drift breaks the
    /// tests instead of the contract).
    fn build_desc(n_inputs: u8, steps: &[(u8, u8, u8, u8, u8, u32)]) -> Vec<u8> {
        let mut d = vec![DESC_VERSION, steps.len() as u8, n_inputs, 0u8];
        for &(op, lk, li, rk, ri, imm) in steps {
            d.push(op);
            d.push(lk);
            d.push(li);
            d.push(rk);
            d.push(ri);
            d.extend_from_slice(&imm.to_le_bytes());
        }
        d
    }

    fn f32_tensor(vals: &[f32]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[vals.len() as i64], 1);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in vals.iter().enumerate() {
            unsafe { *t.data_f32().add(i) = *v };
        }
        ptr
    }

    fn refcount(ptr: i64) -> i64 {
        NslTensor::from_ptr(ptr).refcount.load(Ordering::SeqCst)
    }

    /// The replay must leave every input handle's refcount exactly where it
    /// found it (crate-internal test: integration tests cannot reach the
    /// refcount field, so the leak-shape assertion lives here).
    #[test]
    fn replay_balances_input_refcounts_and_frees_interiors() {
        let a = f32_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let b = f32_tensor(&[0.5, 0.25, -1.0, 2.0]);
        let c = f32_tensor(&[10.0, 20.0, 30.0, 40.0]);
        let rc = (refcount(a), refcount(b), refcount(c));

        // mul(i0,i1) -> rts(p0, like=i2) -> add(p1, i2): the identity-rts
        // replay retains-and-returns, which is exactly the aliasing case the
        // interior bookkeeping must balance.
        let desc = build_desc(
            3,
            &[
                (OP_MUL, KIND_INPUT, 0, KIND_INPUT, 1, 0),
                (OP_RTS_CHECK, KIND_PREV, 0, KIND_INPUT, 2, 0),
                (OP_ADD, KIND_PREV, 1, KIND_INPUT, 2, 0),
            ],
        );
        let out = nsl_fused_ew_chain(
            std::ptr::null(),
            std::ptr::null(),
            desc.as_ptr(),
            desc.len() as u64,
            a,
            b,
            c,
            0,
            0,
            0,
            3,
        );
        assert_ne!(out, 0);
        assert_eq!((refcount(a), refcount(b), refcount(c)), rc, "input refcount leaked");
        let ot = NslTensor::from_ptr(out);
        assert_eq!(ot.refcount.load(Ordering::SeqCst), 1, "result must be a fresh sole reference");
        nsl_tensor_free(out);
        for p in [a, b, c] {
            nsl_tensor_free(p);
        }
    }

    #[test]
    fn fallback_counter_increments_per_call() {
        let a = f32_tensor(&[1.0, -1.0]);
        let desc = build_desc(1, &[(OP_NEG, KIND_INPUT, 0, KIND_ABSENT, 0, 0)]);
        let before = nsl_fused_ew_fallback_count();
        let out = nsl_fused_ew_chain(
            std::ptr::null(),
            std::ptr::null(),
            desc.as_ptr(),
            desc.len() as u64,
            a,
            0,
            0,
            0,
            0,
            0,
            1,
        );
        // `>`: the counter is process-global and sibling tests replay
        // chains concurrently — an exact delta would flake on ordering.
        assert!(nsl_fused_ew_fallback_count() > before);
        nsl_tensor_free(out);
        nsl_tensor_free(a);
    }
}
