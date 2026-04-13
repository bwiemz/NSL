use nsl_runtime::inspect::format::{StatsHeader, FullHeader, write_stats, write_full};

#[test]
fn write_stats_round_trips_header() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let stats: [f64; 6] = [1.5, 0.3, -2.0, 4.5, 0.0, 0.0];
    write_stats(tmp.path(), 100, "h0", &stats).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();

    assert_eq!(&bytes[0..4], b"NSLI", "magic");
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 1);
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let json_bytes = &bytes[16..16 + header_len];
    let header: StatsHeader = serde_json::from_slice(json_bytes).unwrap();
    assert_eq!(header.step, 100);
    assert_eq!(header.tensor_name, "h0");
    assert!((header.mean - 1.5).abs() < 1e-9);
    assert!((header.std - 0.3).abs() < 1e-9);
    assert_eq!(header.nan_count, 0);
}

#[test]
fn write_full_aligns_data_to_64_bytes() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let header = FullHeader {
        step: 5,
        tensor_name: "qkv".into(),
        kind: "full".into(),
        dtype: "bf16".into(),
        shape: vec![1, 8, 16],
        stats: StatsHeader {
            step: 5, tensor_name: "qkv".into(), kind: "full".into(),
            mean: 0.0, std: 1.0, min: -1.0, max: 1.0,
            nan_count: 0, inf_count: 0,
        },
    };
    let data: Vec<u8> = (0..256u16).flat_map(|n| n.to_le_bytes()).collect();
    write_full(tmp.path(), &header, &data).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();

    assert_eq!(&bytes[0..4], b"NSLI");
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let expected_data_start = ((16 + header_len + 63) / 64) * 64;
    assert_eq!(expected_data_start % 64, 0);
    assert_eq!(&bytes[expected_data_start..expected_data_start + data.len()], &data[..]);
}

#[test]
fn write_stats_fails_on_missing_dir() {
    let path = std::path::Path::new("/definitely/nonexistent/path/foo.bin");
    assert!(write_stats(path, 1, "x", &[0.0; 6]).is_err());
}
