#![cfg(feature = "interop")]
//! M62 bridge: an unsupported tensor dtype must be REFUSED, not fatal.
//!
//! Three host-controlled inputs used to take the embedding process down
//! instead of returning an error code. All three are reachable from
//! `python/nslpy/_bridge.py` with no filtering in between, so "the host" is a
//! Python interpreter:
//!
//!   1. `nsl_model_call_dlpack` fed a DLPack tensor whose dtype has no NSL
//!      mapping. `dlpack_to_nsl_tensor` returns 0 for those, and the caller
//!      collected the 0 like any other pointer and handed it to
//!      `nsl_tensor_to_desc`, which dereferenced it. **torch's default integer
//!      dtype is int64**, so `NslModel.forward(torch.tensor([1, 2, 3]))`
//!      SEGFAULTED. Confirmed on the unmodified branch: the child died with
//!      `unix_wait_status(139)`.
//!   2. `nsl_desc_to_tensor` handed an `NslTensorDesc.dtype` outside `0..=9`.
//!      `capi_dtype_to_nsl` called `std::process::abort()` — confirmed on the
//!      unmodified branch as `unix_wait_status(134)`.
//!   3. The same bad tag arriving through a compiled `@export` wrapper, which
//!      now has to branch on `nsl_desc_to_tensor` returning 0.
//!
//! Every scenario runs in a CHILD process (`std::env::current_exe()` re-exec,
//! the pattern from `gpu_dtype_refusal.rs:288`) so the parent can tell
//! "refused with our message" from "died on a signal" — an in-process
//! `#[should_panic]` cannot observe either a SIGSEGV or a SIGABRT.
//!
//! ## Anti-vacuity
//!
//! A `-1` proves nothing on its own: the dispatch wrapper's arity check
//! (`c_wrapper.rs:480`) also returns `-1`. Three separate positive observables
//! pin that these gates fail for the reason claimed:
//!
//!   * `well_formed_f32_dlpack_imports_and_int64_does_not` — the two
//!     hand-built `DLManagedTensor`s differ ONLY in dtype, and the f32 one
//!     really does import (`nsl_dlpack_import` returns a live dtype-1, len-4
//!     tensor). Without this a malformed struct would refuse for the wrong
//!     reason.
//!   * The `ok_f32_dlpack` scenario runs the SAME export with the SAME arity
//!     under `NSL_CAPI_TRACE=1` and must reach the dispatcher
//!     (`model_call name=alpha` on stderr). The `bad_int64_dlpack` scenario
//!     must NOT reach it — which is what rules the arity check out as the
//!     source of its `-1`.
//!   * `call_ok_f32` drives the compiled `@export` through `nsl_model_call`
//!     with a preallocated output buffer and asserts `rc == 0` AND the
//!     computed values (`x * 2.0`), so the `call_desc42` refusal on the same
//!     export cannot be a call that was broken to begin with.
//!
//! ## Known gap (pre-existing, NOT introduced here)
//!
//! `ok_f32_dlpack` asserts the call reaches the dispatcher, not `rc == 0`:
//! `nsl_model_call_dlpack` builds its output descs with
//! `NslTensorDesc::default()`, so `dst.data` is null, and
//! `nsl_dispatch_apply_result` refuses a null caller buffer
//! (`c_api/mod.rs:1081`). The DLPack path therefore cannot currently succeed
//! for a tensor-returning export. That is a separate defect — the output desc
//! needs an allocation strategy the C ABI does not currently have — and is
//! tracked as a follow-up. It does not weaken these gates, because the
//! trace-marker pair above discriminates on WHERE the call stopped, not on
//! whether it finished.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

use nsl_runtime::c_api::NslTensorDesc;
use nsl_runtime::dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};

const SCENARIO: &str = "NSL_DLPACK_REFUSAL_SCENARIO";
const LIB_ENV: &str = "NSL_DLPACK_REFUSAL_LIB";
const WEIGHTS_ENV: &str = "NSL_DLPACK_REFUSAL_WEIGHTS";

/// The export driven by every scenario. The body MUST touch `x` — a bare
/// `return x` would hand a null straight back to `nsl_tensor_to_desc_ffi`,
/// which null-checks, so the missing-guard mutation would not crash and the
/// `call_desc42` gate would pass vacuously.
const EXPORT_SRC: &str = "\n@export\nfn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 2.0\n";

// ---------------------------------------------------------------------------
// DLPack struct construction (by hand, as dlpack.rs's own unit tests do)
// ---------------------------------------------------------------------------

fn managed(shape: &mut [i64], data: *mut c_void, code: u8, bits: u8) -> DLManagedTensor {
    DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device: DLDevice { device_type: DLDeviceType::KDLCpu, device_id: 0 },
            ndim: shape.len() as std::os::raw::c_int,
            dtype: DLDataType { code, bits, lanes: 1 },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    }
}

fn managed_f32(shape: &mut [i64], data: *mut c_void) -> DLManagedTensor {
    managed(shape, data, DLDataTypeCode::KDLFloat as u8, 32)
}

/// `code = KDLInt, bits = 64` — exactly what `torch.tensor([1, 2, 3])`
/// produces, and the dtype `dl_dtype_to_nsl` has no arm for.
fn managed_int64(shape: &mut [i64], data: *mut c_void) -> DLManagedTensor {
    managed(shape, data, DLDataTypeCode::KDLInt as u8, 64)
}

fn last_error() -> String {
    let p = nsl_runtime::c_api::nsl_get_last_error() as *const c_char;
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

// ---------------------------------------------------------------------------
// Anti-vacuity: the two structs differ ONLY in dtype
// ---------------------------------------------------------------------------

/// In-process, no shared library: proves the hand-built f32 `DLManagedTensor`
/// is genuinely importable and the int64 one is rejected at exactly the dtype
/// step. Without this, a typo in the struct (a null shape, a wrong ndim) would
/// make every refusal gate below pass for the wrong reason.
#[test]
fn well_formed_f32_dlpack_imports_and_int64_does_not() {
    let mut fdata: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut idata: Vec<i64> = vec![1, 2, 3, 4];
    let mut shape_a: Vec<i64> = vec![4];
    let mut shape_b: Vec<i64> = vec![4];

    let ok = managed_f32(&mut shape_a, fdata.as_mut_ptr() as *mut c_void);
    let bad = managed_int64(&mut shape_b, idata.as_mut_ptr() as *mut c_void);

    let ok_ptr = nsl_runtime::dlpack::nsl_dlpack_import(&ok as *const _ as i64);
    assert_ne!(
        ok_ptr, 0,
        "the f32 DLManagedTensor must import — if this fails the refusal gates \
         below are testing a malformed struct, not a dtype"
    );
    // Positive observable: a real tensor, not merely a non-zero integer.
    let desc = describe(ok_ptr);
    assert_eq!(desc, (1_i32, 1_i32, 4_i64), "expected (dtype=f32, ndim=1, len=4), got {desc:?}");
    nsl_runtime::tensor::nsl_tensor_free(ok_ptr);

    let bad_ptr = nsl_runtime::dlpack::nsl_dlpack_import(&bad as *const _ as i64);
    assert_eq!(
        bad_ptr, 0,
        "int64 (code=0, bits=64) has no NSL mapping and must import as 0 — \
         this is the 0 that used to be dereferenced"
    );
}

/// Read back (dtype, ndim, len) through the public desc surface.
fn describe(tensor_ptr: i64) -> (i32, i32, i64) {
    let mut d = NslTensorDesc::default();
    nsl_runtime::c_api::nsl_tensor_to_desc_ffi(tensor_ptr, &mut d as *mut _ as i64);
    let len: i64 = if d.ndim <= 0 {
        1
    } else {
        (0..d.ndim as usize)
            .map(|i| unsafe { std::ptr::read_unaligned(d.shape.add(i)) })
            .product()
    };
    (d.dtype, d.ndim, len)
}

// ---------------------------------------------------------------------------
// Child scenarios
// ---------------------------------------------------------------------------

#[test]
#[ignore = "child process — driven by the gates below via re-exec"]
fn zz_dlpack_refusal_child() {
    let scenario = match std::env::var(SCENARIO) {
        Ok(s) => s,
        Err(_) => return,
    };
    match scenario.as_str() {
        "desc42" => {
            // `nsl_desc_to_tensor` with a dtype tag outside 0..=9. Used to
            // `abort()`; must now return 0 with an error naming the tag.
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let mut shape: Vec<i64> = vec![4];
            let desc = NslTensorDesc {
                data: data.as_ptr() as *mut c_void,
                shape: shape.as_mut_ptr(),
                strides: std::ptr::null_mut(),
                ndim: 1,
                dtype: 42,
                device_type: 0,
                device_id: 0,
                tape_id: 0,
            };
            nsl_runtime::c_api::nsl_clear_error();
            let t = nsl_runtime::c_api::nsl_desc_to_tensor(&desc as *const _ as i64);
            assert_eq!(t, 0, "dtype 42 must not produce a tensor");
            println!("CHILD-RESULT rc=0 err={}", last_error());
        }
        _ => run_model_scenario(&scenario),
    }
    println!("CHILD-COMPLETED scenario={scenario}");
}

fn run_model_scenario(scenario: &str) {
    let lib = std::env::var(LIB_ENV).expect("child needs the built shared library path");
    let weights = std::env::var(WEIGHTS_ENV).expect("child needs the weights path");
    let w = CString::new(weights).unwrap();
    let l = CString::new(lib).unwrap();
    let model =
        nsl_runtime::c_api::nsl_model_create_with_lib(w.as_ptr() as i64, l.as_ptr() as i64);
    // Distinctive text so the parent can tell "the harness broke" from "the
    // guard did not fire".
    if model == 0 {
        eprintln!("PRODUCER-REGRESSION could not create the model: {}", last_error());
        std::process::exit(3);
    }
    let name = CString::new("alpha").unwrap();

    let mut fdata: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut idata: Vec<i64> = vec![1, 2, 3, 4];
    let mut shape: Vec<i64> = vec![4];

    match scenario {
        "ok_f32_dlpack" | "bad_int64_dlpack" => {
            let mut dl = if scenario == "ok_f32_dlpack" {
                managed_f32(&mut shape, fdata.as_mut_ptr() as *mut c_void)
            } else {
                managed_int64(&mut shape, idata.as_mut_ptr() as *mut c_void)
            };
            let mut ins: Vec<*mut DLManagedTensor> = vec![&mut dl];
            let mut outs: Vec<*mut DLManagedTensor> = vec![std::ptr::null_mut()];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call_dlpack(
                model,
                name.as_ptr() as i64,
                ins.as_mut_ptr() as i64,
                1,
                outs.as_mut_ptr() as i64,
                1,
            );
            println!("CHILD-RESULT rc={rc} err={}", last_error());
        }
        "null_dlpack" => {
            let mut ins: Vec<*mut DLManagedTensor> = vec![std::ptr::null_mut()];
            let mut outs: Vec<*mut DLManagedTensor> = vec![std::ptr::null_mut()];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call_dlpack(
                model,
                name.as_ptr() as i64,
                ins.as_mut_ptr() as i64,
                1,
                outs.as_mut_ptr() as i64,
                1,
            );
            println!("CHILD-RESULT rc={rc} err={}", last_error());
        }
        "call_ok_f32" | "call_desc42" => {
            // Drive the COMPILED @export wrapper through the desc ABI. Unlike
            // the DLPack path this one preallocates the output buffer, so the
            // f32 case genuinely returns 0 and the computed values are
            // readable — the anti-vacuity anchor for `call_desc42`.
            let mut out_buf: Vec<f32> = vec![0.0; 4];
            let mut out_shape: Vec<i64> = vec![4];
            let dtype = if scenario == "call_ok_f32" { 1 } else { 42 };
            let mut input = NslTensorDesc {
                data: fdata.as_mut_ptr() as *mut c_void,
                shape: shape.as_mut_ptr(),
                strides: std::ptr::null_mut(),
                ndim: 1,
                dtype,
                device_type: 0,
                device_id: 0,
                tape_id: 0,
            };
            let mut output = NslTensorDesc {
                data: out_buf.as_mut_ptr() as *mut c_void,
                shape: out_shape.as_mut_ptr(),
                strides: std::ptr::null_mut(),
                ndim: 1,
                dtype: 1,
                device_type: 0,
                device_id: 0,
                tape_id: 0,
            };
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call(
                model,
                name.as_ptr() as i64,
                &mut input as *mut _ as i64,
                1,
                &mut output as *mut _ as i64,
                1,
            );
            println!("CHILD-RESULT rc={rc} err={} out={out_buf:?}", last_error());
        }
        other => panic!("unknown scenario '{other}'"),
    }
    nsl_runtime::c_api::nsl_model_destroy(model);
}

// ---------------------------------------------------------------------------
// Parent: build once, re-exec per scenario
// ---------------------------------------------------------------------------

struct Child {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

impl Child {
    /// The `rc=` field of the child's `CHILD-RESULT` line. Panics with the
    /// full child output if the child never got that far, so a signal death
    /// is reported as a signal death rather than as a missing substring.
    fn rc(&self) -> i64 {
        let line = self
            .stdout
            .lines()
            .find(|l| l.contains("CHILD-RESULT"))
            .unwrap_or_else(|| {
                panic!(
                    "child never reached CHILD-RESULT (status {:?}) — it died before \
                     the call returned.\n--- stdout ---\n{}\n--- stderr ---\n{}",
                    self.status, self.stdout, self.stderr
                )
            });
        let tok = line
            .split_whitespace()
            .find_map(|t| t.strip_prefix("rc="))
            .expect("CHILD-RESULT line always carries rc=");
        tok.parse().expect("rc= is an integer")
    }

    fn err(&self) -> String {
        let line = self
            .stdout
            .lines()
            .find(|l| l.contains("CHILD-RESULT"))
            .expect("CHILD-RESULT present");
        line.split_once("err=").map(|(_, e)| e.to_string()).unwrap_or_default()
    }

    fn assert_survived(&self, scenario: &str) {
        assert!(
            !self.stderr.contains("PRODUCER-REGRESSION"),
            "scenario '{scenario}': the child died building its INPUT, not on the \
             guard — this gate would otherwise pass vacuously.\n--- stderr ---\n{}",
            self.stderr
        );
        assert!(
            self.status.success(),
            "scenario '{scenario}': child exited {:?}. A refusal must RETURN, not \
             kill the process — a non-zero status here (139 = SIGSEGV, 134 = \
             SIGABRT) is the exact regression this gate exists for.\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            self.status, self.stdout, self.stderr
        );
        assert!(
            self.stdout.contains("CHILD-COMPLETED"),
            "scenario '{scenario}': child exited 0 but never reached the end of \
             the scenario.\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.stdout, self.stderr
        );
    }
}

fn run_child(scenario: &str, ctx: &Option<(String, String)>, trace: bool) -> Child {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "zz_dlpack_refusal_child",
        "--exact",
        "--nocapture",
        "--test-threads=1",
        "--include-ignored",
    ])
    .env(SCENARIO, scenario)
    // A SIGABRT/SIGSEGV path prints a long backtrace that buries the line
    // being matched on and slows the gate for nothing.
    .env("RUST_BACKTRACE", "0");
    if trace {
        cmd.env("NSL_CAPI_TRACE", "1");
    }
    if let Some((lib, weights)) = ctx {
        cmd.env(LIB_ENV, lib).env(WEIGHTS_ENV, weights);
    }
    let out = cmd.output().expect("failed to re-exec the test binary");
    Child {
        status: out.status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// `nsl_desc_to_tensor` with `dtype = 42`. No shared library needed.
#[test]
fn desc_to_tensor_refuses_an_out_of_range_dtype_tag_instead_of_aborting() {
    let c = run_child("desc42", &None, false);
    c.assert_survived("desc42");
    assert_eq!(c.rc(), 0, "the returned tensor pointer must be 0");
    let err = c.err();
    assert!(
        err.contains("42"),
        "the error must name the offending tag 42, got: {err}"
    );
    assert!(
        err.contains("dtype"),
        "the error must say the problem is a dtype, got: {err}"
    );
}

/// The four scenarios that need a compiled `@export`. One build (~138 MB),
/// four children, one cleanup — `nsl build --shared-lib` is far and away the
/// dominant cost here.
#[test]
fn dlpack_and_export_wrappers_refuse_unsupported_dtypes_without_killing_the_host() {
    let tmp = scratch_dir();
    let (lib, weights) = build_lib(&tmp);
    let ctx = Some((
        lib.to_string_lossy().into_owned(),
        weights.to_string_lossy().into_owned(),
    ));

    // ── ANTI-VACUITY 1: the compiled export really works ─────────────────
    // Same export, same arity, through the desc ABI with a preallocated
    // output buffer. `rc == 0` AND the doubled values.
    let ok = run_child("call_ok_f32", &ctx, false);
    ok.assert_survived("call_ok_f32");
    assert_eq!(
        ok.rc(),
        0,
        "a well-formed f32 desc must succeed — if this is non-zero the \
         refusal gates below cannot be attributed to the dtype.\n{}",
        ok.stdout
    );
    assert!(
        ok.stdout.contains("out=[2.0, 4.0, 6.0, 8.0]"),
        "the export must have actually computed x * 2.0 (proves the call ran \
         rather than short-circuiting), got: {}",
        ok.stdout
    );

    // ── GATE: bad dtype through the compiled @export wrapper ─────────────
    // Without the emitted null branch, `nsl_desc_to_tensor` returns 0 and the
    // impl dereferences it => SIGSEGV.
    let bad_call = run_child("call_desc42", &ctx, false);
    bad_call.assert_survived("call_desc42");
    assert_eq!(
        bad_call.rc(),
        -1,
        "an unrecognized desc dtype must come back as -1 from nsl_model_call"
    );

    // ── ANTI-VACUITY 2: the DLPack path reaches the dispatcher on f32 ─────
    // `model_call name=alpha` is `capi_trace`'s line from inside
    // `nsl_model_call`. Its presence here and its ABSENCE below is what
    // proves the int64 -1 is our pre-dispatch refusal and not the dispatch
    // wrapper's arity check.
    let ok_dl = run_child("ok_f32_dlpack", &ctx, true);
    ok_dl.assert_survived("ok_f32_dlpack");
    assert!(
        ok_dl.stderr.contains("model_call name=alpha"),
        "the f32 DLPack call must reach the dispatcher.\n--- stderr ---\n{}",
        ok_dl.stderr
    );
    assert!(
        !ok_dl.err().contains("unsupported DLPack dtype"),
        "a supported dtype must not trip the dtype refusal, got: {}",
        ok_dl.err()
    );

    // ── GATE (a): int64 DLPack input ─────────────────────────────────────
    let bad_dl = run_child("bad_int64_dlpack", &ctx, true);
    bad_dl.assert_survived("bad_int64_dlpack");
    assert_eq!(bad_dl.rc(), -1, "an unsupported DLPack dtype must return -1");
    let err = bad_dl.err();
    assert!(
        err.contains("input 0"),
        "the error must name the offending input INDEX so a caller with several \
         tensors knows which one to fix, got: {err}"
    );
    assert!(
        err.contains("code=0") && err.contains("bits=64"),
        "the error must name the offending DLPack dtype (code=0 int, bits=64), \
         got: {err}"
    );
    assert!(
        err.contains("bfloat16") && err.contains("int32"),
        "the error must list what IS supported, got: {err}"
    );
    assert!(
        !bad_dl.stderr.contains("model_call name=alpha"),
        "the refusal must happen BEFORE dispatch — if the dispatcher ran, this \
         -1 could be its arity check rather than the dtype guard.\n\
         --- stderr ---\n{}",
        bad_dl.stderr
    );

    // ── GATE: a null DLManagedTensor* slot ───────────────────────────────
    let null_dl = run_child("null_dlpack", &ctx, false);
    null_dl.assert_survived("null_dlpack");
    assert_eq!(null_dl.rc(), -1, "a null DLManagedTensor* must return -1");
    assert!(
        null_dl.err().contains("input 0") && null_dl.err().contains("null DLManagedTensor"),
        "the error must name the null input, got: {}",
        null_dl.err()
    );

    cleanup_scratch(&tmp);
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn scratch_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("nsl_dlpack_refusal_{}", std::process::id()))
}

fn build_lib(tmp: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    std::fs::create_dir_all(tmp).unwrap();
    let src = tmp.join("m.nsl");
    std::fs::write(&src, EXPORT_SRC).unwrap();
    let weights = tmp.join("w.safetensors");
    std::fs::write(&weights, b"\x02\x00\x00\x00\x00\x00\x00\x00{}").unwrap();
    let lib_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let lib = tmp.join(format!("libm.{lib_ext}"));
    let manifest_dir: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let stdlib = workspace_root.join("stdlib");
    let out = std::process::Command::new(nsl_bin())
        .env("NSL_STDLIB_PATH", &stdlib)
        .args([
            "build",
            "--shared-lib",
            src.to_str().unwrap(),
            // Without -o, `nsl build` derives the output path from the SOURCE
            // stem — which has twice put a ~128 MB ELF into the repo.
            "-o",
            lib.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "nsl build --shared-lib failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    (lib, weights)
}

/// Remove the compile scratch. Each `nsl build --shared-lib` costs ~138 MB
/// (the library statically links the runtime); on a tmpfs /tmp, accumulating
/// them exhausts the filesystem and the LINKER starts failing, which surfaces
/// as unrelated-looking failures across the workspace.
fn cleanup_scratch(tmp: &std::path::Path) {
    if std::env::var("NSL_KEEP_TEMP").as_deref() == Ok("1") {
        return;
    }
    let _ = std::fs::remove_dir_all(tmp);
}

/// Path to the `nsl` binary built alongside this test. `nsl` lives in the
/// sibling `nsl-cli` crate, so Cargo does not set `CARGO_BIN_EXE_nsl` here.
fn nsl_bin() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("locate test executable");
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    dir.join(format!("nsl{}", std::env::consts::EXE_SUFFIX))
}
