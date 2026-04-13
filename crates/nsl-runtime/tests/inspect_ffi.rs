use nsl_runtime::inspect::ffi::{nsl_inspect_record_stats, nsl_inspect_set_dir};

fn set_tmp_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let bytes = path.as_bytes();
    unsafe {
        nsl_inspect_set_dir(bytes.as_ptr(), bytes.len());
    }
    tmp
}

#[test]
fn record_stats_writes_expected_file_layout() {
    let dir = set_tmp_dir();
    let stats: [f64; 6] = [1.5, 0.3, -2.0, 4.5, 0.0, 0.0];
    let name = "h0";
    let nb = name.as_bytes();
    let rc = unsafe { nsl_inspect_record_stats(stats.as_ptr(), 100, nb.as_ptr(), nb.len()) };
    assert_eq!(rc, 0, "expected 0, got {}", rc);

    let path = dir.path().join("step_100_h0.stats.bin");
    assert!(path.exists(), "expected stats file at {}", path.display());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"NSLI");
}

#[test]
fn record_stats_null_pointer_returns_error() {
    let _dir = set_tmp_dir();
    let rc = unsafe { nsl_inspect_record_stats(std::ptr::null(), 1, std::ptr::null(), 0) };
    assert_ne!(rc, 0);
}

#[test]
fn record_stats_creates_dir_if_missing() {
    let outer = tempfile::tempdir().unwrap();
    let nested = outer.path().join("a").join("b");
    let pstr = nested.to_str().unwrap();
    let pb = pstr.as_bytes();
    unsafe {
        nsl_inspect_set_dir(pb.as_ptr(), pb.len());
    }
    let stats: [f64; 6] = [0.0; 6];
    let name = "x";
    let nb = name.as_bytes();
    let rc = unsafe { nsl_inspect_record_stats(stats.as_ptr(), 0, nb.as_ptr(), nb.len()) };
    assert_eq!(rc, 0);
    assert!(nested.join("step_0_x.stats.bin").exists());
}

#[test]
fn nsl_tensor_stats_symbol_exists() {
    let _f: extern "C" fn(i64, *mut f64) -> i32 = nsl_runtime::inspect::stats_kernel::nsl_tensor_stats;
}

#[test]
fn nsl_inspect_dump_full_symbol_exists() {
    let _f: unsafe extern "C" fn(i64, u64, *const u8, usize) -> i32 =
        nsl_runtime::inspect::ffi::nsl_inspect_dump_full;
}
