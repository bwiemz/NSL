#![cfg(feature = "interop")]
//! M62 bridge: an unsupported tensor dtype must be REFUSED, not fatal.
//!
//! Four host-controlled inputs used to take the embedding process down
//! instead of returning an error code. All four are reachable from
//! `python/nslpy/_bridge.py` or from a plain C host with no filtering in
//! between, so "the host" is a Python interpreter:
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
//!   4. `nsl_model_call_dlpack` called with `num_inputs > 0` and a NULL input
//!      ARRAY (distinct from 5's null ELEMENT). The import loop was skipped
//!      entirely, `input_descs` stayed EMPTY, and the caller's `num_inputs` was
//!      still forwarded alongside `Vec::as_mut_ptr()` of an empty `Vec` —
//!      which is `NonNull::dangling()`, i.e. `0x8`. `nsl_desc_to_tensor(8)`
//!      sails past its `desc_ptr == 0` check and reads `ndim` off address 8:
//!      SIGSEGV. This is the one path that turned a caller's honest NULL into a
//!      dangling pointer BEFORE the guard in 3 could see it.
//!   5. A NULL `DLManagedTensor*` element inside a non-null input array.
//!
//! Every scenario runs in a CHILD process (`std::env::current_exe()` re-exec,
//! the pattern from `gpu_dtype_refusal.rs:288`) so the parent can tell
//! "refused with our message" from "died on a signal" — an in-process
//! `#[should_panic]` cannot observe either a SIGSEGV or a SIGABRT.
//!
//! ## NOT closed here: a bad dtype that IS mapped still aborts the host
//!
//! The list above is not the complete set of host-controlled dtype deaths, and
//! must not be read as one. `dl_dtype_to_nsl` maps DLPack int8 to tag 4 and
//! int32 to tag 9, both INSIDE `capi_dtype_to_nsl`'s accepted `0..=9`. So
//! `torch.int8`/`torch.int32` inputs — and desc tags 4/7/9 on the
//! `nsl_model_call` ABI — clear every boundary guard above, reach the compiled
//! impl, and die on `NslTensor::data_f64`'s PLAIN `assert_eq!`
//! (`tensor/mod.rs:531`) inside a `pub extern "C"` fn: a non-unwinding panic,
//! i.e. SIGABRT / exit 134. Closing that means `emit_c_abi_wrapper` must check
//! the declared dtype it currently discards at `c_wrapper.rs:178`, which also
//! changes what tags 0 and 3 do (today they are accepted and silently compute
//! wrong values), so it is a behaviour change needing its own gate and is
//! deferred. This is pre-existing on both sides of the change.
//!
//! ## Harness: the child has TWO error thread-locals
//!
//! This test binary links `nsl-runtime` as an rlib; `libm.so` (built by
//! `nsl build --shared-lib`) STATICALLY links its own copy. `nm -D` on the test
//! binary shows zero `nsl_*` dynamic symbols, so nothing interposes: they have
//! separate `LAST_ERROR` thread-locals. A refusal raised on the exe side (every
//! `nsl_model_call_dlpack` message) is readable via [`last_error`]; a refusal
//! raised INSIDE the emitted `@export` wrapper (every `emit_null_tensor_guard`
//! message) is not, and shows up as an empty `err=`. The `call_*` scenarios
//! therefore also `dlsym` the model library's own `nsl_get_last_error` and
//! print it as `CHILD-LIBERR` — without that, `call_desc42` could only assert a
//! return code, and deleting `emit_set_error` from the emitted guard kept the
//! gate green.
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
//! ## Item 7: the DLPack output path now WORKS, and this file gates it
//!
//! Until item 7, `nsl_model_call_dlpack` could not return a tensor result
//! (its output descs were zero-initialised and `nsl_dispatch_apply_result`
//! refused the null buffer); `ok_f32_dlpack` asserted the appended
//! "cannot allocate DLPack outputs" note. That note's own comment mandated
//! deleting it in the change that added an output allocator — this is that
//! change. `ok_f32_dlpack` now asserts `rc == 0`, the computed values read
//! THROUGH the returned `DLManagedTensor`, and the ownership contract
//! (deleter releases the NSL tensor exactly once — observed via refcount in
//! `dlpack.rs`'s unit tests and via the RSS-bounded leak loop here).
//!
//! The item-7 scenarios also gate:
//!   * `call_into_*` — the capacity contract: exact-fit succeeds with the
//!     result shape deep-copied into caller-owned arrays; an undersized
//!     buffer REFUSES and reports the required byte count (before item 7
//!     the same call memcpy'd unchecked into the caller's guess — a silent
//!     heap overrun).
//!   * `call_scalar_*` — the scalar-return wild read. The typed wrapper
//!     stores the scalar's raw bits where the scratch desc's data POINTER
//!     lives; the old flow handed that desc to `nsl_dispatch_apply_result`,
//!     which memcpy'd FROM the value-reinterpreted-as-address. Any
//!     scalar-returning export driven through `nsl_model_call` hit it.
//!   * `alloc_leak_loop` / `into_leak_loop` — the new entry points must not
//!     inherit the legacy path's impl-result leak (256 KiB per call on the
//!     `delta` export; 2000 iterations would grow RSS by ~500 MiB if either
//!     path leaked).
//!   * `sig_json` — `nsl_model_get_export_signature` returns the compiler's
//!     `ExportInfo` JSON (shape/dtype/arity), and refuses unknown names.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

use nsl_runtime::c_api::NslTensorDesc;
use nsl_runtime::dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLManagedTensor, DLTensor, KDL_CPU};

const SCENARIO: &str = "NSL_DLPACK_REFUSAL_SCENARIO";
const LIB_ENV: &str = "NSL_DLPACK_REFUSAL_LIB";
const WEIGHTS_ENV: &str = "NSL_DLPACK_REFUSAL_WEIGHTS";

/// The exports driven by every scenario. ONE `nsl build --shared-lib` (~138 MB
/// and the dominant cost of this file) produces all three.
///
/// Each body MUST touch its tensor parameters — a bare `return x` would hand a
/// null straight back to `nsl_tensor_to_desc_ffi`, which null-checks, so the
/// missing-guard mutation would not crash and the `call_desc42` gate would pass
/// vacuously.
///
///   * `alpha` — one tensor param. The single-param guard.
///   * `beta` — TWO tensor params, so a refusal on the SECOND one runs
///     `emit_null_tensor_guard`'s "free what we already imported" loop with a
///     non-empty list. With `alpha` alone that loop had zero iterations in
///     every scenario, and the commit's leak-free-refusal claim was unverified.
///   * `forward` — the fixed name `nsl_model_forward_grad` dispatches to, so
///     the grad path's own `desc_to_nsl_tensor` refusal can be driven.
///   * `gamma` — a SCALAR-f64 return. Before item 7 this shape wild-read on
///     every `nsl_model_call` (the scalar bits were dereferenced as the
///     memcpy source address); now it routes through
///     `nsl_dispatch_apply_scalar_result` and must produce the exact value.
///   * `delta` — 64Ki elements (256 KiB), the leak-loop payload: big enough
///     that a per-call impl-result leak moves RSS by ~500 MiB over the loop
///     while a leak-free path stays flat.
///   * `memcpy` — the NAME is load-bearing: the reverse-interposition gate.
///     The artifact's statically-linked runtime references libc `memcpy` on
///     every dispatch copy. When item 7's first interposition fix
///     (-Bsymbolic-functions) was in place, every intra-image reference
///     bound at link time — so this export's wrapper CAPTURED the runtime's
///     own memcpy calls (pointer-args ABI meets memcpy's semantics: crash or
///     heap corruption on model create / first dispatch). The codegen fix
///     (dispatch calls a Linkage::Local sibling; no linker flag) must keep
///     libc references resolving to libc. `assert_survived` plus the value
///     check on THIS export is the gate; do not rename it.
const EXPORT_SRC: &str = concat!(
    "\n@export\nfn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 2.0\n",
    "\n@export\nfn beta(x: Tensor<[4], f32>, y: Tensor<[4], f32>) -> Tensor<[4], f32>:\n",
    "    return x * 2.0 + y\n",
    "\n@export\nfn forward(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 2.0\n",
    "\n@export\nfn gamma(x: Tensor<[4], f32>) -> f64:\n    return 7.5\n",
    "\n@export\nfn delta(x: Tensor<[65536], f32>) -> Tensor<[65536], f32>:\n    return x * 2.0\n",
    "\n@export\nfn memcpy(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 3.0\n",
    "\n@export\nfn ident(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x\n",
);

// ---------------------------------------------------------------------------
// DLPack struct construction (by hand, as dlpack.rs's own unit tests do)
// ---------------------------------------------------------------------------

fn managed(shape: &mut [i64], data: *mut c_void, code: u8, bits: u8) -> DLManagedTensor {
    DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device: DLDevice { device_type: KDL_CPU, device_id: 0 },
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

/// This TEST BINARY's copy of the runtime error slot (the rlib's thread-local).
/// See "the child has TWO error thread-locals" in the module docs.
fn last_error() -> String {
    let p = nsl_runtime::c_api::nsl_get_last_error() as *const c_char;
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// The MODEL LIBRARY's copy of the runtime error slot.
///
/// `libm.so` statically links its own `nsl-runtime`, so every message set by
/// the Cranelift-emitted `@export` wrapper (`emit_null_tensor_guard`'s
/// `nsl_set_error_cstr` call) lands in the `.so`'s `LAST_ERROR`, which
/// [`last_error`] above cannot read. `dlopen` on an already-loaded path returns
/// the SAME handle the runtime's own `libloading` load produced, and the
/// scenario runs on one thread, so this reads exactly the slot the wrapper
/// wrote. Production reads it this way too: `python/nslpy/_core.py` does
/// `ctypes.CDLL(model.so)` and pulls `nsl_get_last_error` off that handle.
#[cfg(unix)]
fn lib_last_error(lib: &str) -> String {
    use std::os::raw::c_int;
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    let path = match CString::new(lib) {
        Ok(p) => p,
        Err(_) => return String::new(),
    };
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return String::new();
    }
    let sym = CString::new("nsl_get_last_error").unwrap();
    let f = unsafe { dlsym(handle, sym.as_ptr()) };
    if f.is_null() {
        return String::new();
    }
    let f = unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> i64>(f) };
    let p = f() as *const c_char;
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// No `dlsym` equivalent is wired up here; the `CHILD-LIBERR` assertions in the
/// parent are `cfg!(unix)`-guarded to match, rather than silently falling back
/// to the exe-side slot (which would make them pass for the wrong reason).
#[cfg(not(unix))]
fn lib_last_error(_lib: &str) -> String {
    String::new()
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
    let lib_path = std::env::var(LIB_ENV).expect("child needs the built shared library path");
    let weights = std::env::var(WEIGHTS_ENV).expect("child needs the weights path");
    let w = CString::new(weights).unwrap();
    let l = CString::new(lib_path.clone()).unwrap();
    let model =
        nsl_runtime::c_api::nsl_model_create_with_lib(w.as_ptr() as i64, l.as_ptr() as i64);
    // Distinctive text so the parent can tell "the harness broke" from "the
    // guard did not fire".
    if model == 0 {
        eprintln!("PRODUCER-REGRESSION could not create the model: {}", last_error());
        std::process::exit(3);
    }
    let name = CString::new("alpha").unwrap();
    let beta_name = CString::new("beta").unwrap();

    let mut fdata: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut ydata: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let mut idata: Vec<i64> = vec![1, 2, 3, 4];
    let mut shape: Vec<i64> = vec![4];
    let mut shape_y: Vec<i64> = vec![4];
    let mut shape_o: Vec<i64> = vec![4];

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
            // Item 7: on success, read the values THROUGH the returned
            // ownership-transferring DLManagedTensor, then release it via its
            // deleter — the full consumer contract in one scenario.
            let mut out_vals: Vec<f32> = Vec::new();
            let mut out_meta = String::new();
            if rc == 0 && !outs[0].is_null() {
                let m = unsafe { &*outs[0] };
                let t = &m.dl_tensor;
                out_meta = format!(
                    "ndim={} bits={} code={} dev={}",
                    t.ndim, t.dtype.bits, t.dtype.code, t.device.device_type
                );
                if t.ndim == 1 && !t.shape.is_null() && !t.data.is_null() {
                    let n = unsafe { *t.shape } as usize;
                    out_vals =
                        unsafe { std::slice::from_raw_parts(t.data as *const f32, n) }.to_vec();
                }
                nsl_runtime::dlpack::nsl_dlpack_free(outs[0] as i64);
            }
            println!(
                "CHILD-RESULT rc={rc} out={out_vals:?} meta=[{out_meta}] err={}",
                last_error()
            );
        }
        // Item 7 — ownership model A: capacity-checked caller-alloc.
        "call_into_ok" | "call_into_undersized" => {
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            // Caller-owned output buffers per the call_into contract: data
            // capacity in `caps`, shape/strides arrays with slot capacity
            // declared via ndim ON ENTRY.
            let cap: u64 = if scenario == "call_into_ok" { 16 } else { 8 };
            let mut out_buf: Vec<f32> = vec![0.0; 4];
            let mut out_shape: Vec<i64> = vec![0; 8];
            let mut out_strides: Vec<i64> = vec![0; 8];
            let mut output = NslTensorDesc {
                data: out_buf.as_mut_ptr() as *mut c_void,
                shape: out_shape.as_mut_ptr(),
                strides: out_strides.as_mut_ptr(),
                ndim: 8,
                dtype: 1,
                device_type: 0,
                device_id: 0,
                tape_id: 0,
            };
            let caps: Vec<u64> = vec![cap];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call_into(
                model,
                name.as_ptr() as i64,
                &mut input as *mut _ as i64,
                1,
                &mut output as *mut _ as i64,
                1,
                caps.as_ptr() as i64,
            );
            println!(
                "CHILD-RESULT rc={rc} out={out_buf:?} ndim={} shape0={} dtype={} \
                 stride0={} err={}",
                output.ndim, out_shape[0], output.dtype, out_strides[0], last_error()
            );
            println!("CHILD-LIBERR {}", lib_last_error(&lib_path));
        }
        // Item 7 — the scalar-return wild read (gamma returns f64 7.5).
        "call_scalar_model_call" | "call_scalar_into" => {
            let gamma_name = CString::new("gamma").unwrap();
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            let mut scalar_out: f64 = 0.0;
            let mut out_shape: Vec<i64> = vec![0; 8];
            let mut output = NslTensorDesc {
                data: &mut scalar_out as *mut f64 as *mut c_void,
                shape: out_shape.as_mut_ptr(),
                strides: std::ptr::null_mut(),
                ndim: 8,
                dtype: 1,
                device_type: 0,
                device_id: 0,
                tape_id: 0,
            };
            nsl_runtime::c_api::nsl_clear_error();
            let rc = if scenario == "call_scalar_model_call" {
                nsl_runtime::c_api::nsl_model_call(
                    model,
                    gamma_name.as_ptr() as i64,
                    &mut input as *mut _ as i64,
                    1,
                    &mut output as *mut _ as i64,
                    1,
                )
            } else {
                let caps: Vec<u64> = vec![8];
                nsl_runtime::c_api::nsl_model_call_into(
                    model,
                    gamma_name.as_ptr() as i64,
                    &mut input as *mut _ as i64,
                    1,
                    &mut output as *mut _ as i64,
                    1,
                    caps.as_ptr() as i64,
                )
            };
            println!(
                "CHILD-RESULT rc={rc} scalar={scalar_out} ndim={} dtype={} err={}",
                output.ndim, output.dtype, last_error()
            );
        }
        // Item 7 — ownership model B: NSL allocates, DLPack transfers.
        "call_alloc_ok" => {
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            let mut outs: Vec<*mut DLManagedTensor> = vec![std::ptr::null_mut()];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call_alloc(
                model,
                name.as_ptr() as i64,
                &mut input as *mut _ as i64,
                1,
                outs.as_mut_ptr() as i64,
                1,
            );
            let mut out_vals: Vec<f32> = Vec::new();
            if rc == 0 && !outs[0].is_null() {
                let m = unsafe { &*outs[0] };
                let t = &m.dl_tensor;
                if t.ndim == 1 && !t.shape.is_null() && !t.data.is_null() {
                    let n = unsafe { *t.shape } as usize;
                    out_vals =
                        unsafe { std::slice::from_raw_parts(t.data as *const f32, n) }.to_vec();
                }
                nsl_runtime::dlpack::nsl_dlpack_free(outs[0] as i64);
            }
            println!("CHILD-RESULT rc={rc} out={out_vals:?} err={}", last_error());
        }
        // Item 7 — scalar exports must REFUSE the alloc path (no DLPack
        // scalar), not wild-read or crash.
        "call_alloc_scalar_refused" => {
            let gamma_name = CString::new("gamma").unwrap();
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            // POISON, not null: a null-initialised slot cannot distinguish
            // "the callee nulled every slot" (the documented contract) from
            // "the callee never touched them".
            let mut outs: Vec<*mut DLManagedTensor> = vec![0xDEAD_BEEF_usize as *mut _];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call_alloc(
                model,
                gamma_name.as_ptr() as i64,
                &mut input as *mut _ as i64,
                1,
                outs.as_mut_ptr() as i64,
                1,
            );
            println!(
                "CHILD-RESULT rc={rc} slot_null={} err={}",
                outs[0].is_null(),
                last_error()
            );
            println!("CHILD-LIBERR {}", lib_last_error(&lib_path));
        }
        // Item 7 — leak gates. 2000 × 256 KiB outputs: a per-call impl-result
        // leak (the legacy path's contract) would grow RSS ~500 MiB; the new
        // entry points must stay flat. RSS is read from /proc/self/statm.
        "alloc_leak_loop" | "into_leak_loop" => {
            let delta_name = CString::new("delta").unwrap();
            const N_ELEM: usize = 65536;
            let mut big: Vec<f32> = (0..N_ELEM).map(|i| i as f32).collect();
            let mut big_shape: Vec<i64> = vec![N_ELEM as i64];
            // Warm up allocator pools before the baseline read.
            const WARMUP: usize = 50;
            const ITERS: usize = 2000;
            let mut rss_before = 0u64;
            for i in 0..(WARMUP + ITERS) {
                if i == WARMUP {
                    rss_before = rss_kb();
                }
                let mut input =
                    desc(big.as_mut_ptr() as *mut c_void, big_shape.as_mut_ptr(), 1);
                if scenario == "alloc_leak_loop" {
                    let mut outs: Vec<*mut DLManagedTensor> = vec![std::ptr::null_mut()];
                    let rc = nsl_runtime::c_api::nsl_model_call_alloc(
                        model,
                        delta_name.as_ptr() as i64,
                        &mut input as *mut _ as i64,
                        1,
                        outs.as_mut_ptr() as i64,
                        1,
                    );
                    assert_eq!(rc, 0, "alloc iteration {i} failed: {}", last_error());
                    nsl_runtime::dlpack::nsl_dlpack_free(outs[0] as i64);
                } else {
                    let mut out_buf: Vec<f32> = vec![0.0; N_ELEM];
                    let mut out_shape: Vec<i64> = vec![0; 8];
                    let mut output = NslTensorDesc {
                        data: out_buf.as_mut_ptr() as *mut c_void,
                        shape: out_shape.as_mut_ptr(),
                        strides: std::ptr::null_mut(),
                        ndim: 8,
                        dtype: 1,
                        device_type: 0,
                        device_id: 0,
                        tape_id: 0,
                    };
                    let caps: Vec<u64> = vec![(N_ELEM * 4) as u64];
                    let rc = nsl_runtime::c_api::nsl_model_call_into(
                        model,
                        delta_name.as_ptr() as i64,
                        &mut input as *mut _ as i64,
                        1,
                        &mut output as *mut _ as i64,
                        1,
                        caps.as_ptr() as i64,
                    );
                    assert_eq!(rc, 0, "into iteration {i} failed: {}", last_error());
                }
            }
            let rss_after = rss_kb();
            let grown_kb = rss_after.saturating_sub(rss_before);
            println!(
                "CHILD-RESULT rc=0 rss_before={rss_before} rss_after={rss_after} \
                 grown_kb={grown_kb} err="
            );
        }
        // Item 7 — reverse-interposition gate: an export named `memcpy`
        // must not capture the runtime's own libc memcpy calls (see the
        // EXPORT_SRC docs). Reaching CHILD-RESULT at all is half the gate —
        // under -Bsymbolic-functions this crashed before or during dispatch.
        "reverse_interpose" => {
            let memcpy_name = CString::new("memcpy").unwrap();
            let mut out_buf: Vec<f32> = vec![0.0; 4];
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            let mut output = desc(out_buf.as_mut_ptr() as *mut c_void, shape_o.as_mut_ptr(), 1);
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call(
                model,
                memcpy_name.as_ptr() as i64,
                &mut input as *mut _ as i64,
                1,
                &mut output as *mut _ as i64,
                1,
            );
            println!("CHILD-RESULT rc={rc} out={out_buf:?} err={}", last_error());
        }
        // Item 7 — an export returning its INPUT parameter. Three properties
        // in one child, all previously uncovered:
        //   (a) call_alloc must REFUSE it — the result aliases the caller's
        //       buffer, so ownership cannot be transferred (a consumer would
        //       outlive the input and the deleter would free foreign memory);
        //   (b) call_into must handle it repeatedly without crashing or
        //       leaking — the captured result and the typed wrapper's input
        //       wrapper are the SAME pointer, so the release accounting has
        //       to be exactly one decref each (the return path increfs);
        //   (c) a REFUSED armed dispatch must leave the thread-local mode
        //       disarmed: a later call in the same thread must still work.
        "ident_alias" => {
            let ident_name = CString::new("ident").unwrap();
            let mut outs: Vec<*mut DLManagedTensor> = vec![0xDEAD_BEEF_usize as *mut _];
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            nsl_runtime::c_api::nsl_clear_error();
            let rc_alloc = nsl_runtime::c_api::nsl_model_call_alloc(
                model,
                ident_name.as_ptr() as i64,
                &mut input as *mut _ as i64,
                1,
                outs.as_mut_ptr() as i64,
                1,
            );
            // The aliasing refusal is raised by finish_alloc INSIDE the model
            // image (the ownership FFIs are dlsym'd from it), so the exe-side
            // slot is empty by construction — read the library's slot, the
            // same way the emitted-wrapper refusals are read.
            println!(
                "CHILD-ALLOC rc={rc_alloc} slot_null={} err={}",
                outs[0].is_null(),
                last_error()
            );
            println!("CHILD-LIBERR {}", lib_last_error(&lib_path));

            // (b) repeated call_into on the aliasing export.
            let mut ok_all = true;
            for _ in 0..500 {
                let mut inp = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
                let mut ob: Vec<f32> = vec![0.0; 4];
                let mut osh: Vec<i64> = vec![0; 8];
                let mut ost: Vec<i64> = vec![0; 8];
                let mut out = NslTensorDesc {
                    data: ob.as_mut_ptr() as *mut c_void,
                    shape: osh.as_mut_ptr(),
                    strides: ost.as_mut_ptr(),
                    ndim: 8,
                    dtype: 1,
                    device_type: 0,
                    device_id: 0,
                    tape_id: 0,
                };
                let caps: Vec<u64> = vec![16];
                let rc = nsl_runtime::c_api::nsl_model_call_into(
                    model,
                    ident_name.as_ptr() as i64,
                    &mut inp as *mut _ as i64,
                    1,
                    &mut out as *mut _ as i64,
                    1,
                    caps.as_ptr() as i64,
                );
                if rc != 0 || ob != vec![1.0f32, 2.0, 3.0, 4.0] {
                    ok_all = false;
                    break;
                }
            }

            // (c) refusal, then a successful call on the SAME thread.
            let mut inp = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            let mut ob: Vec<f32> = vec![0.0; 4];
            let mut osh: Vec<i64> = vec![0; 8];
            let mut ost: Vec<i64> = vec![0; 8];
            let mut out = NslTensorDesc {
                data: ob.as_mut_ptr() as *mut c_void,
                shape: osh.as_mut_ptr(),
                strides: ost.as_mut_ptr(),
                ndim: 8,
                dtype: 1,
                device_type: 0,
                device_id: 0,
                tape_id: 0,
            };
            let small: Vec<u64> = vec![8];
            let rc_refused = nsl_runtime::c_api::nsl_model_call_into(
                model,
                ident_name.as_ptr() as i64,
                &mut inp as *mut _ as i64,
                1,
                &mut out as *mut _ as i64,
                1,
                small.as_ptr() as i64,
            );
            let mut inp2 = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1);
            let caps: Vec<u64> = vec![16];
            let rc_after = nsl_runtime::c_api::nsl_model_call_into(
                model,
                ident_name.as_ptr() as i64,
                &mut inp2 as *mut _ as i64,
                1,
                &mut out as *mut _ as i64,
                1,
                caps.as_ptr() as i64,
            );
            println!(
                "CHILD-RESULT rc=0 loop_ok={ok_all} rc_refused={rc_refused} \
                 rc_after={rc_after} out={ob:?} err="
            );
        }
        // Item 7 — signature introspection.
        "sig_json" => {
            let gamma_name = CString::new("gamma").unwrap();
            let missing = CString::new("no_such_export").unwrap();
            nsl_runtime::c_api::nsl_clear_error();
            let alpha_ptr = nsl_runtime::c_api::nsl_model_get_export_signature(
                model,
                name.as_ptr() as i64,
            );
            let gamma_ptr = nsl_runtime::c_api::nsl_model_get_export_signature(
                model,
                gamma_name.as_ptr() as i64,
            );
            let alpha_json = if alpha_ptr != 0 {
                unsafe { CStr::from_ptr(alpha_ptr as *const c_char) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };
            let gamma_json = if gamma_ptr != 0 {
                unsafe { CStr::from_ptr(gamma_ptr as *const c_char) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };
            nsl_runtime::c_api::nsl_clear_error();
            let missing_ptr = nsl_runtime::c_api::nsl_model_get_export_signature(
                model,
                missing.as_ptr() as i64,
            );
            println!("CHILD-SIG-ALPHA {alpha_json}");
            println!("CHILD-SIG-GAMMA {gamma_json}");
            println!(
                "CHILD-RESULT rc={} err={}",
                if missing_ptr == 0 { 0 } else { 99 },
                last_error()
            );
        }
        // A null ELEMENT inside a real array.
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
        // A null ARRAY with a declared arity — a DIFFERENT site from
        // `null_dlpack` above, which the per-element `dl.is_null()` check does
        // not reach. This is the one that used to hand `0x8` (an empty `Vec`'s
        // dangling pointer) to the dispatch wrapper and SIGSEGV.
        "null_dlpack_array" => {
            let mut outs: Vec<*mut DLManagedTensor> = vec![std::ptr::null_mut()];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call_dlpack(
                model,
                name.as_ptr() as i64,
                0, // NULL input array …
                1, // … while still declaring one input
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
            // The refusal above was raised INSIDE libm.so, whose runtime copy
            // has its own LAST_ERROR — see the module docs.
            println!("CHILD-LIBERR {}", lib_last_error(&lib_path));
        }
        // TWO tensor params, so the guard on the SECOND one runs
        // `emit_null_tensor_guard`'s free loop with one already-imported tensor
        // in it. With the single-param export that loop had zero iterations.
        "call_beta_ok" | "call_beta_desc42_arg1" => {
            let mut out_buf: Vec<f32> = vec![0.0; 4];
            let y_dtype = if scenario == "call_beta_ok" { 1 } else { 42 };
            let mut output = desc(out_buf.as_mut_ptr() as *mut c_void, shape_o.as_mut_ptr(), 1);
            // The C ABI takes ONE contiguous NslTensorDesc array.
            let mut ins = [
                desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), 1),
                desc(ydata.as_mut_ptr() as *mut c_void, shape_y.as_mut_ptr(), y_dtype),
            ];
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::c_api::nsl_model_call(
                model,
                beta_name.as_ptr() as i64,
                ins.as_mut_ptr() as i64,
                2,
                &mut output as *mut _ as i64,
                1,
            );
            println!("CHILD-RESULT rc={rc} err={} out={out_buf:?}", last_error());
            println!("CHILD-LIBERR {}", lib_last_error(&lib_path));
        }
        // The grad path's own `desc_to_nsl_tensor` refusal. Unlike the OUTPUT
        // snapshot guard (unreachable — `nsl_model_call` validates the tag
        // before mirroring it), the INPUT snapshot reads CALLER-supplied descs
        // and really is reachable. A 0 left there is dereferenced by
        // `run_backward_core`'s tape_id lookup.
        "fwd_grad_ok" | "fwd_grad_desc42" => {
            let mut out_buf: Vec<f32> = vec![0.0; 4];
            let dtype = if scenario == "fwd_grad_ok" { 1 } else { 42 };
            let mut input = desc(fdata.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), dtype);
            let mut output = desc(out_buf.as_mut_ptr() as *mut c_void, shape_o.as_mut_ptr(), 1);
            let mut ctx: i64 = 0;
            nsl_runtime::c_api::nsl_clear_error();
            let rc = nsl_runtime::grad_context::nsl_model_forward_grad(
                model,
                &mut input as *mut _ as i64,
                1,
                &mut output as *mut _ as i64,
                1,
                &mut ctx as *mut _ as i64,
            );
            println!("CHILD-RESULT rc={rc} err={} out={out_buf:?}", last_error());
            println!("CHILD-LIBERR {}", lib_last_error(&lib_path));
            if ctx != 0 {
                nsl_runtime::grad_context::nsl_grad_context_destroy(ctx);
            }
        }
        other => panic!("unknown scenario '{other}'"),
    }
    nsl_runtime::c_api::nsl_model_destroy(model);
}

/// Resident set size in KiB, from /proc/self/statm (Linux). Returns 0
/// elsewhere; the leak-loop assertions are cfg-gated to match.
#[cfg(target_os = "linux")]
fn rss_kb() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    // No unwrap_or(0) fallback: a broken read would make the leak gate
    // compare 0 against 0 and pass vacuously forever.
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|t| t.parse().ok())
        .expect("/proc/self/statm must yield a resident-pages field");
    assert!(pages > 0, "resident pages must be non-zero");
    pages * 4 // 4 KiB pages
}

#[cfg(not(target_os = "linux"))]
fn rss_kb() -> u64 {
    0
}

/// A CPU, contiguous, 1-D `NslTensorDesc` over a caller-owned buffer.
fn desc(data: *mut c_void, shape: *mut i64, dtype: i32) -> NslTensorDesc {
    NslTensorDesc {
        data,
        shape,
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype,
        device_type: 0,
        device_id: 0,
        tape_id: 0,
    }
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

    /// The error read out of the MODEL LIBRARY's runtime copy, on its own line
    /// so it cannot collide with `err=`'s substring split. Empty for scenarios
    /// that do not emit it (and on non-unix, where no `dlsym` reader is wired
    /// up) — every assertion on it is `cfg!(unix)`-guarded accordingly.
    fn lib_err(&self) -> String {
        self.stdout
            .lines()
            .find_map(|l| l.strip_prefix("CHILD-LIBERR "))
            .unwrap_or("")
            .to_string()
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
    // The -1 alone leaves the emitted guard's `nsl_set_error_cstr` call
    // completely uncovered — deleting it keeps the return code. Read the
    // message out of the LIBRARY's error slot (the exe-side `err=` is empty for
    // this scenario by construction; see the module docs).
    if cfg!(unix) {
        let lerr = bad_call.lib_err();
        assert!(
            lerr.contains("parameter 'x'") && lerr.contains("'alpha'"),
            "the emitted wrapper's refusal must name the offending PARAMETER and \
             the export, so a host with several arguments knows which to fix. \
             got: {lerr}\n--- stdout ---\n{}",
            bad_call.stdout
        );
        assert!(
            lerr.contains("dtype tag"),
            "the emitted wrapper's refusal must say the problem is a dtype tag, \
             got: {lerr}"
        );
    }

    // ── ANTI-VACUITY 1b: the TWO-param export works ──────────────────────
    // Anchors `call_beta_desc42_arg1` below the same way `call_ok_f32` anchors
    // `call_desc42`: same export, same arity, only the dtype differs.
    let beta_ok = run_child("call_beta_ok", &ctx, false);
    beta_ok.assert_survived("call_beta_ok");
    assert_eq!(beta_ok.rc(), 0, "beta(x, y) with two f32 descs must succeed\n{}", beta_ok.stdout);
    assert!(
        beta_ok.stdout.contains("out=[12.0, 24.0, 36.0, 48.0]"),
        "beta must have computed x * 2 + y, got: {}",
        beta_ok.stdout
    );

    // ── GATE: bad dtype on the SECOND parameter ──────────────────────────
    // The guard fires at param index 1, so `emit_null_tensor_guard`'s
    // "free what we already imported" loop runs with ONE tensor in it. Every
    // other scenario refuses at index 0, where that loop is empty — so without
    // this the commit's "a refusal must not also be a leak" claim, and the
    // `lists_live` clone added for the tuple path, execute in no test at all.
    let beta_bad = run_child("call_beta_desc42_arg1", &ctx, false);
    beta_bad.assert_survived("call_beta_desc42_arg1");
    assert_eq!(
        beta_bad.rc(),
        -1,
        "a bad tag on the second parameter must also refuse\n{}",
        beta_bad.stdout
    );
    if cfg!(unix) {
        let lerr = beta_bad.lib_err();
        assert!(
            lerr.contains("parameter 'y'") && lerr.contains("'beta'"),
            "the refusal must name the SECOND parameter (not the first, which \
             was fine), got: {lerr}\n--- stdout ---\n{}",
            beta_bad.stdout
        );
    }
    assert!(
        beta_bad.stdout.contains("out=[0.0, 0.0, 0.0, 0.0]"),
        "a refused call must not have written the caller's output buffer, got: {}",
        beta_bad.stdout
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
    // The f32 desc really was reconstructed into a LIVE dtype-1 tensor inside
    // the dispatch wrapper — `desc_to_nsl_tensor` traces only AFTER
    // `capi_dtype_to_nsl` accepts the tag, so this line cannot be reached by an
    // input the dtype guard would refuse.
    //
    // (This replaces `assert!(!ok_dl.err().contains("unsupported DLPack
    // dtype"))`, which was structurally always true: `err=` is the exe-side
    // slot and the refusal here is raised inside libm.so, so it held for every
    // possible implementation — including one with the whole dtype guard
    // deleted. Line 461's trace marker was already doing all the work.)
    assert!(
        ok_dl.stderr.contains("desc_to_tensor") && ok_dl.stderr.contains("dtype=1"),
        "the f32 input must have been imported as a live dtype-1 tensor by the \
         dispatch wrapper.\n--- stderr ---\n{}",
        ok_dl.stderr
    );
    // …and (item 7) the call now SUCCEEDS end-to-end: the values are read
    // through the returned ownership-transferring DLManagedTensor, whose
    // metadata must describe a 1-D f32 CPU tensor, and the child releases it
    // via its deleter without dying.
    assert_eq!(
        ok_dl.rc(),
        0,
        "a well-formed f32 DLPack call must now return 0 — the output \
         allocator landed with item 7.\n{}",
        ok_dl.stdout
    );
    assert!(
        ok_dl.stdout.contains("out=[2.0, 4.0, 6.0, 8.0]"),
        "the DLPack output must carry the computed x * 2.0 values, got: {}",
        ok_dl.stdout
    );
    assert!(
        ok_dl.stdout.contains("meta=[ndim=1 bits=32 code=2 dev=1]"),
        "the DLManagedTensor metadata must describe a 1-D f32 CPU tensor \
         (code 2 = kDLFloat, dev 1 = kDLCPU), got: {}",
        ok_dl.stdout
    );

    // ── ITEM-7 GATE: call_into capacity contract ─────────────────────────
    let into_ok = run_child("call_into_ok", &ctx, false);
    into_ok.assert_survived("call_into_ok");
    assert_eq!(into_ok.rc(), 0, "exact-capacity call_into must succeed\n{}", into_ok.stdout);
    assert!(
        into_ok.stdout.contains("out=[2.0, 4.0, 6.0, 8.0]"),
        "call_into must deep-copy the computed values, got: {}",
        into_ok.stdout
    );
    assert!(
        into_ok.stdout.contains("ndim=1 shape0=4"),
        "call_into must deep-copy the result shape into the CALLER'S arrays \
         and set ndim to the result rank, got: {}",
        into_ok.stdout
    );
    assert!(
        into_ok.stdout.contains("dtype=1"),
        "call_into must report the RESULT's dtype tag (1 = f32) — a regression \
         to f64 would have the caller read the buffer at the wrong width, got: {}",
        into_ok.stdout
    );
    assert!(
        into_ok.stdout.contains("stride0=1"),
        "call_into must deep-copy STRIDES into the caller's array too (element \
         strides; 1 for a contiguous 1-D f32 result). Without this the \
         documented strides half of the contract is unenforced, got: {}",
        into_ok.stdout
    );

    let into_small = run_child("call_into_undersized", &ctx, false);
    into_small.assert_survived("call_into_undersized");
    assert_eq!(
        into_small.rc(),
        -1,
        "an undersized output buffer must REFUSE (the legacy path memcpy'd \
         into it unchecked — a silent heap overrun)\n{}",
        into_small.stdout
    );
    if cfg!(unix) {
        let lerr = into_small.lib_err();
        assert!(
            lerr.contains("requires 16 bytes") && lerr.contains("8 bytes"),
            "the capacity refusal must report the REQUIRED byte count so the \
             caller can reallocate and retry, got: {lerr}\n{}",
            into_small.stdout
        );
    }
    assert!(
        into_small.stdout.contains("out=[0.0, 0.0, 0.0, 0.0]"),
        "a capacity refusal must not have written the caller's buffer, got: {}",
        into_small.stdout
    );

    // ── ITEM-7 GATE: the scalar-return wild read is fixed ────────────────
    // gamma returns f64 7.5. Before the fix the dispatch flow memcpy'd FROM
    // the bit pattern of 7.5 reinterpreted as an address — at best garbage,
    // at worst SIGSEGV (assert_survived is load-bearing here).
    let sc_call = run_child("call_scalar_model_call", &ctx, false);
    sc_call.assert_survived("call_scalar_model_call");
    assert_eq!(sc_call.rc(), 0, "scalar export via nsl_model_call must succeed\n{}", sc_call.stdout);
    assert!(
        sc_call.stdout.contains("scalar=7.5 ndim=0 dtype=0"),
        "the scalar value must arrive AS A VALUE (7.5, rank 0, f64 tag) — not \
         be dereferenced as an address, got: {}",
        sc_call.stdout
    );

    let sc_into = run_child("call_scalar_into", &ctx, false);
    sc_into.assert_survived("call_scalar_into");
    assert_eq!(sc_into.rc(), 0, "scalar export via call_into must succeed\n{}", sc_into.stdout);
    assert!(
        sc_into.stdout.contains("scalar=7.5 ndim=0 dtype=0"),
        "call_into's scalar path must produce the same value, got: {}",
        sc_into.stdout
    );

    // ── ITEM-7 GATE: call_alloc ownership transfer ───────────────────────
    let alloc_ok = run_child("call_alloc_ok", &ctx, false);
    alloc_ok.assert_survived("call_alloc_ok");
    assert_eq!(alloc_ok.rc(), 0, "call_alloc must succeed\n{}", alloc_ok.stdout);
    assert!(
        alloc_ok.stdout.contains("out=[2.0, 4.0, 6.0, 8.0]"),
        "call_alloc's DLManagedTensor must carry the computed values, got: {}",
        alloc_ok.stdout
    );

    let alloc_scalar = run_child("call_alloc_scalar_refused", &ctx, false);
    alloc_scalar.assert_survived("call_alloc_scalar_refused");
    assert_eq!(
        alloc_scalar.rc(),
        -1,
        "a scalar-returning export must REFUSE the alloc path (no DLPack \
         scalar representation)\n{}",
        alloc_scalar.stdout
    );
    assert!(
        alloc_scalar.stdout.contains("slot_null=true"),
        "a refused alloc call must NULL every output slot — the child poisons \
         the slot with 0xDEADBEEF first, so this proves the callee wrote it \
         rather than merely never touching it (a host that frees non-NULL \
         slots after rc=-1 would otherwise chase garbage), got: {}",
        alloc_scalar.stdout
    );
    if cfg!(unix) {
        assert!(
            alloc_scalar.lib_err().contains("scalar"),
            "the refusal must say the problem is the scalar return, got: {}",
            alloc_scalar.lib_err()
        );
    }

    // ── ITEM-7 GATE: the new paths do not inherit the legacy leak ────────
    // 2000 × 256 KiB. Leaking the impl result (struct + owns_data buffer)
    // would grow RSS by ~500 MiB; 64 MiB of headroom absorbs allocator
    // fragmentation while still failing a real leak by an order of magnitude.
    if cfg!(target_os = "linux") {
        for scen in ["alloc_leak_loop", "into_leak_loop"] {
            let leak = run_child(scen, &ctx, false);
            leak.assert_survived(scen);
            let grown_kb: u64 = leak
                .stdout
                .lines()
                .find(|l| l.contains("CHILD-RESULT"))
                .and_then(|l| l.split_whitespace().find_map(|t| t.strip_prefix("grown_kb=")))
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| panic!("{scen}: no grown_kb in\n{}", leak.stdout));
            eprintln!("[item7-leak-gate] {scen}: RSS grew {grown_kb} KiB over 2000 calls");
            assert!(
                grown_kb < 64 * 1024,
                "{scen}: RSS grew {grown_kb} KiB over 2000 calls with 256 KiB \
                 outputs — the ownership path is leaking the impl result \
                 (~500 MiB expected from the legacy leak).\n{}",
                leak.stdout
            );
        }
    }

    // ── ITEM-7 GATE: reverse interposition ───────────────────────────────
    // An export deliberately named `memcpy` — the runtime's dispatch copy
    // path calls libc memcpy, and a regression toward link-time
    // self-binding (-Bsymbolic-functions or equivalent) makes that call hit
    // the export wrapper instead: crash before CHILD-RESULT.
    let rev = run_child("reverse_interpose", &ctx, false);
    rev.assert_survived("reverse_interpose");
    assert_eq!(
        rev.rc(),
        0,
        "the memcpy-named export must dispatch cleanly (its presence must \
         not disturb the runtime's own libc calls)\n{}",
        rev.stdout
    );
    assert!(
        rev.stdout.contains("out=[3.0, 6.0, 9.0, 12.0]"),
        "the memcpy-named export must compute ITS OWN body (x * 3.0), got: {}",
        rev.stdout
    );

    // ── ITEM-7 GATE: input-aliasing, refcount hygiene, re-arm ────────────
    let alias = run_child("ident_alias", &ctx, false);
    alias.assert_survived("ident_alias");
    let alloc_line = alias
        .stdout
        .lines()
        .find(|l| l.contains("CHILD-ALLOC"))
        .unwrap_or_else(|| panic!("no CHILD-ALLOC in\n{}", alias.stdout));
    assert!(
        alloc_line.contains("rc=-1"),
        "call_alloc must REFUSE an export that returns its input: the result \
         aliases the caller's buffer, so a DLPack consumer could outlive it \
         and the deleter would free memory NSL never allocated. got: {alloc_line}"
    );
    assert!(
        alloc_line.contains("slot_null=true"),
        "the aliasing refusal must still NULL the poisoned slot: {alloc_line}"
    );
    if cfg!(unix) {
        let lerr = alias.lib_err();
        assert!(
            lerr.contains("aliases memory NSL does not own"),
            "the refusal must explain WHY and point at the alternative; note it \
             is raised in the MODEL IMAGE's runtime copy, so it is only \
             readable through that library's nsl_get_last_error. got: {lerr}"
        );
    }
    assert!(
        alias.stdout.contains("loop_ok=true"),
        "500 call_into dispatches of an input-returning export must all \
         succeed with correct values — the captured result and the typed \
         wrapper's input wrapper are the same pointer, so a mis-accounted \
         release shows up as a double-free crash or wrong data.\n{}",
        alias.stdout
    );
    assert!(
        alias.stdout.contains("rc_refused=-1") && alias.stdout.contains("rc_after=0"),
        "a REFUSED armed dispatch must leave the thread-local ownership mode \
         disarmed, so the next call on the same thread still succeeds \
         (rc_refused=-1 then rc_after=0).\n{}",
        alias.stdout
    );

    // ── ITEM-7 GATE: export-signature introspection ──────────────────────
    let sig = run_child("sig_json", &ctx, false);
    sig.assert_survived("sig_json");
    assert_eq!(
        sig.rc(),
        0,
        "an unknown export name must return NULL from \
         nsl_model_get_export_signature\n{}",
        sig.stdout
    );
    // `find` + slice, not `strip_prefix`: the child's FIRST stdout line is
    // glued to libtest's "test zz_... ..." prefix, so the marker is not at
    // column 0 there.
    let alpha_sig = sig
        .stdout
        .lines()
        .find_map(|l| l.find("CHILD-SIG-ALPHA ").map(|i| &l[i + "CHILD-SIG-ALPHA ".len()..]))
        .unwrap_or_else(|| panic!("no CHILD-SIG-ALPHA in\n{}", sig.stdout));
    let alpha_json: serde_json::Value =
        serde_json::from_str(alpha_sig).expect("alpha signature must be valid JSON");
    assert_eq!(alpha_json["symbol_name"], "alpha");
    assert_eq!(
        alpha_json["return_type"]["Tensor"]["shape"],
        serde_json::json!(["4"]),
        "alpha's return shape must be the declared [4]: {alpha_json}"
    );
    assert_eq!(
        alpha_json["return_type"]["Tensor"]["dtype"], "F32",
        "alpha's return dtype must be F32: {alpha_json}"
    );
    assert_eq!(
        alpha_json["params"].as_array().map(Vec::len),
        Some(1),
        "alpha takes one parameter: {alpha_json}"
    );
    let gamma_sig = sig
        .stdout
        .lines()
        .find_map(|l| l.find("CHILD-SIG-GAMMA ").map(|i| &l[i + "CHILD-SIG-GAMMA ".len()..]))
        .unwrap_or_else(|| panic!("no CHILD-SIG-GAMMA in\n{}", sig.stdout));
    let gamma_json: serde_json::Value =
        serde_json::from_str(gamma_sig).expect("gamma signature must be valid JSON");
    assert_eq!(
        gamma_json["return_type"]["Scalar"], "F64",
        "gamma's return type must be Scalar F64: {gamma_json}"
    );
    assert!(
        sig.err().contains("no_such_export"),
        "the unknown-name refusal must name the missing export, got: {}",
        sig.err()
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

    // ── GATE (b): a null DLManagedTensor* ELEMENT ────────────────────────
    let null_dl = run_child("null_dlpack", &ctx, false);
    null_dl.assert_survived("null_dlpack");
    assert_eq!(null_dl.rc(), -1, "a null DLManagedTensor* must return -1");
    assert!(
        null_dl.err().contains("input 0") && null_dl.err().contains("null DLManagedTensor"),
        "the error must name the null input, got: {}",
        null_dl.err()
    );

    // ── GATE (c): a null input ARRAY with a declared arity ───────────────
    // A DIFFERENT site from (b): the per-element `dl.is_null()` check never
    // runs, because the whole import block was skipped. `input_descs` stayed
    // empty while `num_inputs` was still forwarded, so the dispatch wrapper
    // received an empty `Vec`'s dangling pointer (0x8) — which
    // `nsl_desc_to_tensor`'s `== 0` check does not catch — and `desc.ndim` was
    // read off address 8. `assert_survived` is the load-bearing half of this
    // gate: before the guard the child exited 139.
    let null_arr = run_child("null_dlpack_array", &ctx, false);
    null_arr.assert_survived("null_dlpack_array");
    assert_eq!(
        null_arr.rc(),
        -1,
        "num_inputs > 0 with a null input array must return -1, not dispatch\n{}",
        null_arr.stdout
    );
    assert!(
        null_arr.err().contains("num_inputs=1") && null_arr.err().contains("null"),
        "the error must name the count/array disagreement rather than the \
         generic model-pointer refusal, got: {}",
        null_arr.err()
    );

    // ── ANTI-VACUITY 3 + GATE: the grad path's INPUT snapshot ────────────
    // `nsl_model_forward_grad` maps caller-supplied descs through
    // `desc_to_nsl_tensor` and would leave a 0 in `ctx.input_ptrs`, which
    // `run_backward_core`'s tape_id lookup dereferences. The PR added that
    // refusal and nothing drove it. (Its OUTPUT-side twin is deliberately NOT
    // gated: those descs are written by `nsl_model_call`, which validates the
    // tag before mirroring it, so the guard is a documented trip-wire.)
    let grad_ok = run_child("fwd_grad_ok", &ctx, false);
    grad_ok.assert_survived("fwd_grad_ok");
    assert_eq!(
        grad_ok.rc(),
        0,
        "a well-formed f32 desc must record a tape — otherwise the refusal \
         below cannot be attributed to the dtype\n{}",
        grad_ok.stdout
    );
    assert!(
        grad_ok.stdout.contains("out=[2.0, 4.0, 6.0, 8.0]"),
        "forward_grad must have actually computed x * 2.0, got: {}",
        grad_ok.stdout
    );

    let grad_bad = run_child("fwd_grad_desc42", &ctx, false);
    grad_bad.assert_survived("fwd_grad_desc42");
    assert_eq!(
        grad_bad.rc(),
        -1,
        "an unrecognized input desc dtype must refuse before the tape records\n{}",
        grad_bad.stdout
    );
    assert!(
        grad_bad.err().contains("nsl_model_forward_grad")
            && grad_bad.err().contains("input 0")
            && grad_bad.err().contains("unrecognized dtype tag"),
        "the grad refusal must name the function and the offending input index, \
         got: {}",
        grad_bad.err()
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
