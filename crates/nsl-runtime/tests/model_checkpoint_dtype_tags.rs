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
//!
//! The same harness pins the structural refusals added after the external
//! review of 2026-10-06: a count mismatch (it only warned, then loaded
//! positionally), a truncated data section (found mid-copy, after earlier
//! tensors were overwritten), a header size past the end of the file (an
//! unchecked slice), and a reorder of same-shaped parameters, which only
//! the named load can see.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use nsl_abi::wire::dtype::{DTYPE_BF16, DTYPE_F32, DTYPE_FP16};
// Naming the entry points through the crate is what links the runtime in; an
// `extern "C"` block alone leaves every symbol undefined.
use nsl_runtime::checkpoint::{nsl_model_load, nsl_model_load_named, nsl_model_save};

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

/// `n` f32 values starting at `base`, as bytes.
fn f32_bytes(base: f32, n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| (base + i as f32).to_le_bytes()).collect()
}

fn save_many(path: &Path, named: &[(&str, i64)]) {
    let p = path.to_str().expect("utf-8 path");
    let names: Vec<i64> = named.iter().map(|(n, _)| CString::new(*n).unwrap().into_raw() as i64).collect();
    let tensors: Vec<i64> = named.iter().map(|&(_, t)| t).collect();
    nsl_model_save(p.as_ptr() as i64, p.len() as i64, list_of(&names), list_of(&tensors));
}

fn load_named(path: &Path, named: &[(&str, i64)]) {
    let p = path.to_str().expect("utf-8 path");
    let names: Vec<i64> = named.iter().map(|(n, _)| CString::new(*n).unwrap().into_raw() as i64).collect();
    let tensors: Vec<i64> = named.iter().map(|&(_, t)| t).collect();
    nsl_model_load_named(p.as_ptr() as i64, p.len() as i64, list_of(&names), list_of(&tensors));
}

/// Two f32 [2, 2] tensors named `a` and `b`, saved to a fresh file.
fn two_tensor_checkpoint(file: &str) -> PathBuf {
    let path = scratch(file);
    let (a, _) = cpu_tensor(&f32_bytes(1.0, 4), &[2, 2], DTYPE_F32);
    let (b, _) = cpu_tensor(&f32_bytes(10.0, 4), &[2, 2], DTYPE_F32);
    save_many(&path, &[("a", a), ("b", b)]);
    path
}

/// Re-exec the child for `scenario` against `path`; returns its stderr after
/// asserting it aborted before reaching the end of the load.
fn refused(scenario: &str, path: &Path) -> String {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["zz_structural_child", "--exact", "--nocapture"])
        .env(SCENARIO, scenario)
        .env(CKPT_PATH, path)
        .output()
        .expect("re-exec test binary");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(!out.status.success(), "{scenario}: the load must abort; stderr:\n{stderr}");
    assert!(!stderr.contains("CHILD_LOADED"), "{scenario}: the child got past the load:\n{stderr}");
    stderr
}

#[test]
fn a_checkpoint_with_more_tensors_than_the_model_is_refused() {
    let path = two_tensor_checkpoint("count.nslm");
    let stderr = refused("count", &path);
    assert!(stderr.contains("the file has 2 tensors and the model has 1"), "{stderr}");
}

#[test]
fn a_truncated_checkpoint_is_refused_before_any_tensor_is_copied() {
    let path = two_tensor_checkpoint("truncated.nslm");
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
    let stderr = refused("truncated", &path);
    // The whole layout is checked up front; the old loader found the short
    // file at tensor #1, after tensor #0 was already overwritten.
    assert!(stderr.contains("the header describes 32 data bytes but the file holds 28"), "{stderr}");
}

#[test]
fn a_header_size_past_the_end_of_the_file_is_refused() {
    let path = two_tensor_checkpoint("header.nslm");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let stderr = refused("header", &path);
    assert!(stderr.contains(&format!("the header claims {} bytes", u64::MAX)), "{stderr}");
}

#[test]
fn a_reordered_checkpoint_is_refused_by_the_named_load() {
    let path = two_tensor_checkpoint("reordered.nslm");
    let stderr = refused("reordered", &path);
    assert!(stderr.contains("tensor #0 is 'a' in the file but 'b' in the model"), "{stderr}");
}

/// A train-block checkpoint names entries through the model variable with
/// dotted indices (`m.blocks.0.w`); `model_load` names fields with brackets
/// (`blocks[0].w`). Loading one into the other is the documented warm start.
#[test]
fn a_train_checkpoint_name_matches_the_model_field_path() {
    let path = scratch("warm.nslm");
    let bytes = f32_bytes(3.0, 4);
    let (src, _) = cpu_tensor(&bytes, &[2, 2], DTYPE_F32);
    save_many(&path, &[("m.blocks.0.w", src)]);
    let (dst, dst_data) = cpu_tensor(&[0u8; 16], &[2, 2], DTYPE_F32);
    load_named(&path, &[("blocks[0].w", dst)]);
    assert_eq!(unsafe { std::slice::from_raw_parts(dst_data, 16) }, bytes.as_slice());
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// Child half of the structural refusals above. A no-op unless the parent
/// set the scenario variable.
#[test]
fn zz_structural_child() {
    let Ok(scenario) = std::env::var(SCENARIO) else { return };
    let path = PathBuf::from(std::env::var(CKPT_PATH).expect("parent sets the path"));
    let (a, _) = cpu_tensor(&[0u8; 16], &[2, 2], DTYPE_F32);
    let (b, _) = cpu_tensor(&[0u8; 16], &[2, 2], DTYPE_F32);
    match scenario.as_str() {
        "count" => load(&path, a),
        "truncated" | "header" => {
            let p = path.to_str().unwrap();
            nsl_model_load(p.as_ptr() as i64, p.len() as i64, list_of(&[a, b]));
        }
        "reordered" => load_named(&path, &[("b", b), ("a", a)]),
        _ => return,
    }
    eprintln!("CHILD_LOADED");
}
