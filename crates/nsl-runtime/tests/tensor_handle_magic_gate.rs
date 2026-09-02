//! Every `extern "C"` tensor op must reject a handle that is not a live
//! tensor — in release builds, not only under `debug_assert!`.
//!
//! Background. `NslTensor::from_ptr` used to be a bare
//! `unsafe { &mut *(ptr as *mut NslTensor) }`, and 321 more call sites
//! wrote the cast out by hand. The only validation anywhere was a
//! handful of `debug_assert!(t.is_valid())`, and **`debug_assert!` compiles
//! to nothing in the release profile the production `nsl build` and every
//! training run use**. So a stale handle — a tensor freed by an earlier
//! scope teardown, an i64 that was never a tensor, a field offset read out
//! of a struct that has moved — was silent undefined behaviour exactly in
//! the builds that matter, and typically surfaced as wrong numbers rather
//! than a crash.
//!
//! The `magic` field has been the first field of the `#[repr(C)]` struct
//! since the beginning (`0x4E534C54`, "NSLT"), and `nsl_tensor_free` poisons
//! it with `TENSOR_FREED` (`0x0000DEAD`) before releasing the box. This file
//! pins that those two facts are actually *used*: a plausibility test and
//! one load-and-compare at every entry point, and a fatal diagnostic naming
//! the handle and which of the four failures it was.
//!
//! # Why every gate here is a subprocess, and not `#[should_panic]`
//!
//! The check aborts; it does not panic. `bad_handle` calls
//! `std::process::abort()` — deliberately, because the callers are
//! `extern "C"` frames that a panic cannot unwind through (since Rust 1.81
//! that is `panic_cannot_unwind` → `SIGABRT` anyway, after a second
//! backtrace), and because the neighbouring fatal paths in this runtime
//! (`assert.rs`, `nsl_tensor_clone`'s null check) abort too. libtest never
//! regains control, so `#[should_panic]` cannot observe any of this.
//!
//! Each gate therefore re-execs THIS binary with `NSL_HANDLE_GATE_SCENARIO`
//! set and asserts on the child's exit status and stderr — the same pattern
//! as `gpu_dtype_refusal.rs` and `matmul_transposed_operand.rs`, and
//! strictly stronger than a substring match on a panic payload: it checks
//! the phrase, the reason, and the absence of a "sailed through" marker
//! together, so a gate cannot pass because some unrelated assert fired
//! first.
//!
//! The `valid` scenario is the control. Without it every assertion here
//! would still hold if `from_ptr` unconditionally aborted, and the gate
//! would be measuring nothing.
//!
//! CPU-only: no `#[ignore]`, no CUDA, no GPU-cert manifest entry.

use nsl_runtime::tensor::{
    TENSOR_FREED, nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_ndim,
};

const SCENARIO: &str = "NSL_HANDLE_GATE_SCENARIO";
const DTYPE_F32: i64 = 1;

/// A live rank-1 f32 tensor over leaked static data (`owns_data = 0`), so
/// nothing here depends on the allocator recycling anything.
fn live_tensor() -> i64 {
    let data: &'static mut [f32] = Box::leak(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice());
    let shape = nsl_runtime::list::nsl_list_new();
    nsl_runtime::list::nsl_list_push(shape, 4);
    let t = nsl_tensor_from_static(data.as_mut_ptr() as i64, shape, DTYPE_F32);
    nsl_runtime::list::nsl_list_free(shape);
    assert_ne!(t, 0, "PRODUCER-REGRESSION: nsl_tensor_from_static returned 0");
    t
}

// --- child ---------------------------------------------------------------

/// Runs one scenario in its own process. Cargo runs every `#[test]` in the
/// binary, so this has to be a test; it returns immediately unless the
/// parent asked for a scenario. Every scenario but `valid` is expected to
/// ABORT inside the runtime.
#[test]
fn zz_handle_gate_child() {
    let Ok(scenario) = std::env::var(SCENARIO) else {
        return;
    };

    match scenario.as_str() {
        // A null handle: the compiler emits 0 for an unbound tensor slot, and
        // an op that reads it used to fault at some later field offset.
        "null" => {
            let _ = nsl_tensor_ndim(0);
        }
        // The poison `nsl_tensor_free` writes. Reproduced directly rather
        // than by freeing, so the assertion on the *reason* is deterministic:
        // after a real free, glibc's tcache overwrites the first bytes of the
        // chunk with its own metadata, which reads back as "not a tensor"
        // rather than as the poison. Both are refusals; only this one is
        // stable enough to assert the wording of.
        "poisoned" => {
            let t = live_tensor();
            // SAFETY: `magic` is the first field of a `#[repr(C)]` struct, so
            // it is at offset 0 of the live allocation `t` names.
            unsafe { *(t as *mut u32) = TENSOR_FREED };
            let _ = nsl_tensor_ndim(t);
        }
        // An i64 that was never a tensor — the shape a field offset read out
        // of a moved struct, or a list handle passed to a tensor op, takes.
        "garbage" => {
            let buf: &'static mut [u64] = Box::leak(vec![0u64; 32].into_boxed_slice());
            let _ = nsl_tensor_ndim(buf.as_mut_ptr() as i64);
        }
        // A small integer used as a handle — a scalar field read out of the
        // wrong slot. Reading the magic there would fault; the plausibility
        // test has to catch it FIRST or the diagnostic becomes a SIGSEGV.
        "low" => {
            let _ = nsl_tensor_ndim(0x40);
        }
        // A misaligned handle. An `NslTensor` starts with a `u32` and a
        // pointer, so its address is always 8-aligned; anything else cannot
        // be one, and reading a `u32` there is what would be undefined.
        "misaligned" => {
            let buf: &'static mut [u64] = Box::leak(vec![0u64; 32].into_boxed_slice());
            let _ = nsl_tensor_ndim(buf.as_mut_ptr() as i64 + 3);
        }
        // The real use-after-free. Asserted only as "refused", because which
        // of the two reasons it reports depends on what the allocator did
        // with the chunk: glibc's tcache writes its own metadata over the
        // first bytes, so the poison usually reads back as "not a tensor".
        // This is the one scenario that depends on the allocator keeping the
        // freed chunk mapped and readable — true of glibc, of musl, and of
        // the system allocators on macOS and Windows for a chunk this size,
        // but NOT of a hardened allocator that decommits on free, nor under
        // ASan's quarantine. If this file ever runs under one of those, this
        // scenario is the one to reach for first.
        "freed" => {
            let t = live_tensor();
            nsl_tensor_free(t);
            let _ = nsl_tensor_ndim(t);
        }
        // The control: a live handle must sail straight through.
        "valid" => {
            let t = live_tensor();
            assert_eq!(nsl_tensor_ndim(t), 1, "PRODUCER-REGRESSION: live tensor is not rank-1");
        }
        other => panic!("unknown {SCENARIO} '{other}'"),
    }

    // Only reached if the check did NOT fire. Distinctive text so the parent
    // can tell "refused as designed" from "sailed straight through".
    println!("GUARD-DID-NOT-FIRE scenario={scenario}");
}

// --- parent --------------------------------------------------------------

fn run_child(scenario: &str) -> (std::process::Output, String, String) {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_handle_gate_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
            "--include-ignored",
        ])
        .env(SCENARIO, scenario)
        // The abort path prints a backtrace when RUST_BACKTRACE is set, which
        // buries the line being matched on.
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("failed to re-exec the test binary");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    (out, stdout, stderr)
}

/// Assert one scenario was refused, and (optionally) refused for the stated
/// reason. `reason = None` means only that it was refused.
fn assert_refuses(scenario: &str, reason: Option<&str>) {
    let (out, stdout, stderr) = run_child(scenario);

    assert!(
        !stdout.contains("GUARD-DID-NOT-FIRE"),
        "scenario '{scenario}': the bad handle went straight into the op — the \
         magic check is missing or was removed.\n--- child stdout ---\n{stdout}\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        !stderr.contains("PRODUCER-REGRESSION"),
        "scenario '{scenario}': the child died building its INPUT, not on the \
         check — this gate would otherwise have passed vacuously.\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        !out.status.success(),
        "scenario '{scenario}': child exited 0; an invalid handle must \
         terminate the process.\n--- child stdout ---\n{stdout}\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("invalid tensor handle"),
        "scenario '{scenario}': child failed, but not on the handle check — so \
         it may be failing for an unrelated reason.\n--- child stderr ---\n{stderr}"
    );
    if let Some(reason) = reason {
        assert!(
            stderr.contains(reason),
            "scenario '{scenario}': refused, but did not report '{reason}', so \
             the diagnostic does not say what was wrong with the handle.\n\
             --- child stderr ---\n{stderr}"
        );
    }
}

#[test]
fn null_handle_is_refused() {
    assert_refuses("null", Some("null handle"));
}

#[test]
fn freed_tensor_magic_is_refused_as_use_after_free() {
    assert_refuses("poisoned", Some("use-after-free"));
}

/// The reason line must name the magic it found and the one it expected —
/// without them the diagnostic cannot distinguish "wrong kind of handle" from
/// "right handle, corrupted memory".
#[test]
fn non_tensor_handle_is_refused_and_names_the_magic() {
    assert_refuses("garbage", Some("not a tensor (magic 0x00000000"));
    let (_, _, stderr) = run_child("garbage");
    assert!(
        stderr.contains("expected 0x4E534C54"),
        "the 'not a tensor' line must name the expected magic.\n--- child stderr ---\n{stderr}"
    );
}

/// The case the whole check exists for: a handle used after `nsl_tensor_free`.
/// Which reason it reports depends on whether the allocator has already
/// overwritten the poisoned field, so only the refusal itself is asserted.
#[test]
fn use_after_free_is_refused() {
    assert_refuses("freed", None);
}

/// The plausibility test must fire BEFORE the magic is read, or these two
/// become a bare SIGSEGV (a small integer) or undefined behaviour (an
/// unaligned read) instead of the diagnostic the check exists to print.
#[test]
fn implausible_addresses_are_refused_before_the_magic_is_read() {
    assert_refuses("low", Some("not an address a tensor can live at"));
    assert_refuses("misaligned", Some("not an address a tensor can live at"));
}

/// The control. A live handle must reach the op untouched — otherwise every
/// assertion above would still pass with `from_ptr` aborting unconditionally,
/// and this file would be measuring nothing.
#[test]
fn live_handle_is_accepted() {
    let (out, stdout, stderr) = run_child("valid");
    assert!(
        out.status.success(),
        "a live tensor handle was refused.\n--- child stdout ---\n{stdout}\n\
         --- child stderr ---\n{stderr}"
    );
    assert!(
        !stderr.contains("invalid tensor handle"),
        "a live tensor handle tripped the check.\n--- child stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("GUARD-DID-NOT-FIRE"),
        "the control scenario did not run to completion.\n--- child stdout ---\n{stdout}"
    );
}
