//! Unit test for the .nslm reader — uses in-memory bytes only.

mod common {
    include!("common/mod.rs");
}

use common::nslm_reader::read_nslm;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_nslm_bytes(
    params: &[(&str, Vec<i64>, &str, Vec<u8>)],
) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("tempfile");

    let mut header_entries = Vec::new();
    let mut data_offset: u64 = 0;
    for (name, shape, dtype, bytes) in params {
        let nbytes = bytes.len() as u64;
        header_entries.push(format!(
            r#"{{"name":"{}","shape":{:?},"dtype":"{}","offset":{},"nbytes":{}}}"#,
            name, shape, dtype, data_offset, nbytes
        ));
        data_offset += nbytes;
    }
    let header = format!(r#"{{"params":[{}]}}"#, header_entries.join(","));
    let header_bytes = header.as_bytes();

    file.write_all(b"NSLM").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header_bytes).unwrap();
    // 64-byte alignment padding from file start, matching checkpoint.rs
    let header_total = 4 + 4 + 8 + header_bytes.len();
    let pad = (64 - (header_total % 64)) % 64;
    file.write_all(&vec![0u8; pad]).unwrap();
    for (_, _, _, bytes) in params {
        file.write_all(bytes).unwrap();
    }
    file.flush().unwrap();
    file
}

#[test]
fn reads_single_f32_tensor() {
    let values: Vec<f32> = vec![0.5, -0.3, 1.25, 0.0];
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
    let file = write_nslm_bytes(&[("w", vec![4], "f32", bytes)]);

    let got = read_nslm(file.path()).expect("read ok");
    let w = got.get("w").expect("w present");
    assert_eq!(w.len(), 4);
    for (g, e) in w.iter().zip(values.iter()) {
        assert!((g - e).abs() < 1e-9_f32, "got {} want {}", g, e);
    }
}

#[test]
fn reads_f64_tensor_and_downcasts_to_f32() {
    let values: Vec<f64> = vec![0.5, -0.25, 1.0];
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
    let file = write_nslm_bytes(&[("b", vec![3], "f64", bytes)]);

    let got = read_nslm(file.path()).expect("read ok");
    let b = got.get("b").expect("b present");
    assert_eq!(b.len(), 3);
    assert!((b[0] - 0.5).abs() < 1e-6);
    assert!((b[1] + 0.25).abs() < 1e-6);
    assert!((b[2] - 1.0).abs() < 1e-6);
}

#[test]
fn rejects_bad_magic() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"XXXX").unwrap();
    file.flush().unwrap();
    let result = read_nslm(file.path());
    assert!(result.is_err());
}
