//! Functional e2e for @export tuple marshaling in the typed C-ABI wrapper.
//!
//! Previously both directions were refused at compile time ("@export tuple
//! input parameter/return ... not yet supported") even though the semantic
//! validator accepts tuples and the C header already declared the tuple ABI
//! (`const NslTensorDesc* items, int32_t count` for params;
//! `NslTensorDesc* __rets, int32_t* __num_rets` for returns).
//!
//! This builds a real shared library with a tuple-returning and a
//! tuple-taking export, dlsyms the TYPED symbols, calls them with
//! hand-constructed descriptors, and verifies values end to end.

use std::os::raw::c_void;

/// Must match `NslTensorDesc` in nsl-runtime/src/c_api/mod.rs (48 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Desc {
    data: *mut c_void,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
    /// C-API dtype convention: 0 = f32, 1 = f64 (inverted from NslTensor.dtype).
    dtype: i32,
    device_type: i32,
    device_id: i32,
    tape_id: i64,
}

impl Desc {
    fn zeroed() -> Self {
        Desc {
            data: std::ptr::null_mut(),
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            ndim: 0,
            dtype: 0,
            device_type: 0,
            device_id: 0,
            tape_id: 0,
        }
    }
}

/// Read element `i` of a returned desc as f64, honoring its dtype tag.
unsafe fn desc_elem(desc: &Desc, i: usize) -> f64 {
    match desc.dtype {
        0 => *(desc.data as *const f32).add(i) as f64,
        1 => *(desc.data as *const f64).add(i),
        other => panic!("unexpected desc dtype {other}"),
    }
}

fn nsl_bin() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("locate test executable");
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    dir.join(format!("nsl{}", std::env::consts::EXE_SUFFIX))
}

fn build_test_lib(nsl_src: &str) -> std::path::PathBuf {
    use std::process::Command;
    let tmp = std::env::temp_dir().join(format!("nsl_tuple_export_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = tmp.join("test.nsl");
    std::fs::write(&src, nsl_src).unwrap();
    let lib_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = tmp.join(format!("libtupletest.{lib_ext}"));

    let manifest_dir: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let stdlib = workspace_root.join("stdlib");

    let output = Command::new(nsl_bin())
        .env("NSL_STDLIB_PATH", &stdlib)
        .args([
            "build",
            "--shared-lib",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "nsl build --shared-lib failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    out
}

#[test]
fn tuple_return_and_param_round_trip() {
    let lib_path = build_test_lib(
        r#"
@export
fn split2(x: Tensor<[4], f32>) -> (Tensor<[4], f32>, Tensor<[4], f32>):
    return (x * 2.0, x * 3.0)

@export
fn addpair(pair: (Tensor<[4], f32>, Tensor<[4], f32>)) -> Tensor<[4], f32>:
    return pair[0] + pair[1]
"#,
    );

    let lib = unsafe { libloading::Library::new(&lib_path) }.expect("load shared lib");

    // Input: [1, 2, 3, 4] f32, shape [4], strides [1].
    let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut shape: Vec<i64> = vec![4];
    let mut strides: Vec<i64> = vec![1];
    let x_desc = Desc {
        data: data.as_mut_ptr() as *mut c_void,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        ndim: 1,
        dtype: 0, // f32 (C-API convention)
        device_type: 0,
        device_id: 0,
        tape_id: 0,
    };
    // The wrapper only null-checks the model pointer for plain fns.
    let dummy_model: i64 = 0x1;

    // ── Tuple RETURN: split2(x) -> (x*2, x*3) ─────────────────────────────
    type Split2Fn =
        unsafe extern "C" fn(i64, *const Desc, *mut Desc, *mut i32) -> i32;
    let split2: libloading::Symbol<Split2Fn> =
        unsafe { lib.get(b"split2") }.expect("dlsym split2");

    let mut rets = [Desc::zeroed(), Desc::zeroed()];
    let mut num_rets: i32 = 0;
    let rc = unsafe { split2(dummy_model, &x_desc, rets.as_mut_ptr(), &mut num_rets) };
    assert_eq!(rc, 0, "split2 returned error");
    assert_eq!(num_rets, 2, "split2 must report 2 outputs");
    for (k, factor) in [(0usize, 2.0f64), (1, 3.0)] {
        assert_eq!(rets[k].ndim, 1, "ret[{k}] ndim");
        let n = unsafe { *rets[k].shape } as usize;
        assert_eq!(n, 4, "ret[{k}] length");
        for i in 0..4 {
            let got = unsafe { desc_elem(&rets[k], i) };
            let want = (i as f64 + 1.0) * factor;
            assert!(
                (got - want).abs() < 1e-5,
                "split2 ret[{k}][{i}] = {got}, want {want}"
            );
        }
    }

    // ── Tuple PARAM: addpair((a, b)) -> a + b ─────────────────────────────
    type AddPairFn =
        unsafe extern "C" fn(i64, *const Desc, i32, *mut Desc) -> i32;
    let addpair: libloading::Symbol<AddPairFn> =
        unsafe { lib.get(b"addpair") }.expect("dlsym addpair");

    let mut data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let mut shape_b: Vec<i64> = vec![4];
    let mut strides_b: Vec<i64> = vec![1];
    let b_desc = Desc {
        data: data_b.as_mut_ptr() as *mut c_void,
        shape: shape_b.as_mut_ptr(),
        strides: strides_b.as_mut_ptr(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
        tape_id: 0,
    };
    let items = [x_desc, b_desc];
    let mut ret = Desc::zeroed();
    let rc = unsafe { addpair(dummy_model, items.as_ptr(), 2, &mut ret) };
    assert_eq!(rc, 0, "addpair returned error");
    for i in 0..4 {
        let got = unsafe { desc_elem(&ret, i) };
        let want = (i as f64 + 1.0) + (i as f64 + 1.0) * 10.0;
        assert!(
            (got - want).abs() < 1e-5,
            "addpair ret[{i}] = {got}, want {want}"
        );
    }

    // ── Count-mismatch guard: wrong count must fail, not overrun ──────────
    let mut ret2 = Desc::zeroed();
    let rc = unsafe { addpair(dummy_model, items.as_ptr(), 1, &mut ret2) };
    assert_eq!(rc, -1, "addpair with wrong tuple count must return -1");
}
