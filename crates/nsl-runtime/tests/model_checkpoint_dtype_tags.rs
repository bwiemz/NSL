//! `model_load` must refuse a checkpoint whose tensor dtype differs from the
//! model's, for every tag, not just f32 against f64 (C5 step 2a,
//! `docs/superpowers/specs/2026-09-26-dtype-semantics-design.md`).
//!
//! Until step 2a the `.nslm` header recorded "f32" for f32 and "f64" for every
//! other tag, and `model_load` compared those strings. An fp16 checkpoint
//! therefore loaded into a bf16 model without complaint and its bytes were
//! reinterpreted. The header now names each tag, and the gates below pin both
//! halves: a matching tag loads byte for byte, and a mismatched 2-byte tag is
//! refused before any byte is copied.
//!
//! The refusal is `std::process::abort()`, which a test cannot catch, so the
//! refusal gate re-execs this binary with `NSL_CKPT_DTYPE_SCENARIO` set and
//! checks the child's exit status and stderr.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use nsl_abi::wire::dtype::{DTYPE_BF16, DTYPE_F32, DTYPE_FP16};
// Naming the entry points through the crate is what links the runtime in; an
// `extern "C"` block alone leaves every symbol undefined.
use nsl_runtime::checkpoint::{nsl_model_load, nsl_model_save};

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
}

const SCENARIO: &str = "NSL_CKPT_DTYPE_SCENARIO";
const CKPT_PATH: &str = "NSL_CKPT_DTYPE_PATH";

fn list_of(values: &[i64]) -> i64 {
    let list = unsafe { nsl_list_new() };
    for &v in values {
        unsafe { nsl_list_push(list, v) };
    }
    list
}

/// A CPU tensor over a leaked copy of `bytes`, tagged `dtype`. Leaked because
/// the tensor does not own its buffer and `model_load` writes into it.
fn cpu_tensor(bytes: &[u8], dims: &[i64], dtype: u16) -> (i64, *const u8) {
    let data: &'static mut [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
    let ptr = data.as_mut_ptr();
    let shape = list_of(dims);
    let t = unsafe { nsl_tensor_from_static(ptr as i64, shape, i64::from(dtype)) };
    (t, ptr)
}

fn save(path: &Path, name: &str, tensor: i64) {
    let p = path.to_str().expect("utf-8 path");
    let name = CString::new(name).unwrap().into_raw() as i64;
    let names = list_of(&[name]);
    let tensors = list_of(&[tensor]);
    nsl_model_save(p.as_ptr() as i64, p.len() as i64, names, tensors);
}

fn load(path: &Path, tensor: i64) {
    let p = path.to_str().expect("utf-8 path");
    let tensors = list_of(&[tensor]);
    nsl_model_load(p.as_ptr() as i64, p.len() as i64, tensors);
}

/// A checkpoint path in a directory of its own: tests run on parallel threads
/// and each removes its directory when done.
fn scratch(file: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nsl_ckpt_dtype_{}_{file}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir.join(file)
}

fn header_names(path: &Path, dtype: &str) -> bool {
    let needle = format!("\"dtype\":\"{dtype}\"");
    let bytes = std::fs::read(path).expect("read checkpoint");
    bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
}

/// 2-byte payload with no zero bytes, so a load that copied nothing is visible.
fn payload(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11) | 1).collect()
}

#[test]
fn a_bf16_checkpoint_loads_into_a_bf16_model_byte_for_byte() {
    let path = scratch("bf16.nslm");
    let bytes = payload(2 * 6);
    let (src, _) = cpu_tensor(&bytes, &[2, 3], DTYPE_BF16);
    save(&path, "w", src);

    assert!(
        header_names(&path, "bf16"),
        "the header must name the bf16 tag, not collapse it to f64"
    );

    let (dst, dst_data) = cpu_tensor(&vec![0u8; bytes.len()], &[2, 3], DTYPE_BF16);
    load(&path, dst);
    let got = unsafe { std::slice::from_raw_parts(dst_data, bytes.len()) };
    assert_eq!(got, bytes.as_slice(), "bf16 payload changed through save/load");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn f32_checkpoints_keep_their_legacy_spelling() {
    // Files written before the header named every tag must still load, so the
    // two spellings they used are unchanged.
    let path = scratch("f32.nslm");
    let bytes: Vec<u8> = [1.5_f32, -0.0, 3.25].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (src, _) = cpu_tensor(&bytes, &[3], DTYPE_F32);
    save(&path, "w", src);
    assert!(header_names(&path, "f32"), "f32 must keep its legacy header spelling");

    let (dst, dst_data) = cpu_tensor(&[0u8; 12], &[3], DTYPE_F32);
    load(&path, dst);
    assert_eq!(unsafe { std::slice::from_raw_parts(dst_data, 12) }, bytes.as_slice());
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn an_fp16_checkpoint_is_refused_by_a_bf16_model() {
    let path = scratch("fp16.nslm");
    let (src, _) = cpu_tensor(&payload(2 * 4), &[4], DTYPE_FP16);
    save(&path, "w", src);

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["zz_load_into_bf16_child", "--exact", "--nocapture"])
        .env(SCENARIO, "load_into_bf16")
        .env(CKPT_PATH, &path)
        .output()
        .expect("re-exec test binary");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "loading an fp16 checkpoint into a bf16 model must abort; child exited {} with stderr:\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains("dtype mismatch for tensor #0: file has fp16, model expects bf16"),
        "the refusal must name both tags; stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("CHILD_LOADED"),
        "the child got past model_load; stderr:\n{stderr}"
    );
}

/// Child half of `an_fp16_checkpoint_is_refused_by_a_bf16_model`. A no-op
/// unless the parent set the scenario variable.
#[test]
fn zz_load_into_bf16_child() {
    if std::env::var(SCENARIO).as_deref() != Ok("load_into_bf16") {
        return;
    }
    let path = PathBuf::from(std::env::var(CKPT_PATH).expect("parent sets the path"));
    let (dst, _) = cpu_tensor(&[0u8; 8], &[4], DTYPE_BF16);
    load(&path, dst);
    eprintln!("CHILD_LOADED");
}
