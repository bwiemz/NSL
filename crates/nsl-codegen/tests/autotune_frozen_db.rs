//! Item 10: frozen tuning database — freeze determinism, the sha256 pin,
//! per-record validation at lookup, and the one-DB-per-process rule.
//!
//! ONE test function on purpose: `load_frozen_db` succeeds at most once per
//! process (the overlay is a `OnceLock`), so the failure-path probes must
//! run BEFORE the successful load, in one deterministic sequence. Parallel
//! `#[test]`s would race the one-shot state — the same process-global
//! lesson as the pass-trace unit tests.

use nsl_codegen::autotune::{
    freeze_records, load_cache_record, load_frozen_db, AutotuneCacheRecord, CacheReject,
    DeviceIdentity, SelectionMethod, CACHE_SCHEMA_VERSION,
};

fn record(kernel: &str, key: &str, device: &DeviceIdentity) -> AutotuneCacheRecord {
    AutotuneCacheRecord {
        schema_version: CACHE_SCHEMA_VERSION,
        kernel: kernel.to_string(),
        cache_key: key.to_string(),
        device: device.clone(),
        selection: SelectionMethod::Measured,
        winner: vec![("block_size".to_string(), 128)],
        winner_ms: 0.5,
        variants_tested: 3,
        all_timings: vec![],
        timestamp_secs: 0,
    }
}

fn test_device() -> DeviceIdentity {
    DeviceIdentity {
        device_name: "FrozenDB Test GPU".to_string(),
        sm_version: 120,
        sm_count: 40,
        driver_version: 13030,
    }
}

#[test]
fn frozen_db_end_to_end() {
    let dir = std::env::temp_dir().join(format!("nsl_frozen_db_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let device = test_device();

    // Freeze is deterministic and order-insensitive: same records in any
    // order produce byte-identical JSON and the same pin hash.
    let mut a = vec![record("k2", "hb", &device), record("k1", "ha", &device)];
    let mut b = vec![record("k1", "ha", &device), record("k2", "hb", &device)];
    let (json_a, sha_a) = freeze_records(&mut a);
    let (json_b, sha_b) = freeze_records(&mut b);
    assert_eq!(json_a, json_b, "freeze must sort — input order leaked into the bytes");
    assert_eq!(sha_a, sha_b);

    let db_path = dir.join("db.json");
    std::fs::write(&db_path, &json_a).unwrap();

    // Pin mismatch refuses BEFORE the one-shot overlay is consumed.
    let err = load_frozen_db(&db_path, Some("deadbeef")).unwrap_err();
    assert!(
        err.contains("sha256 mismatch"),
        "wrong refusal for a bad pin: {err}"
    );

    // Unparseable refuses too (also before the one-shot).
    let junk = dir.join("junk.json");
    std::fs::write(&junk, b"not json").unwrap();
    let err = load_frozen_db(&junk, None).unwrap_err();
    assert!(err.contains("unparseable"), "wrong refusal for junk: {err}");

    // Correct pin loads (case-insensitive hex).
    let n = load_frozen_db(&db_path, Some(&sha_a.to_uppercase())).unwrap();
    assert_eq!(n, 2);

    // Second load refuses — one DB per process, no silent shadowing.
    let err = load_frozen_db(&db_path, None).unwrap_err();
    assert!(err.contains("twice"), "wrong refusal for a second load: {err}");

    // Overlay lookup: the record for the matching device comes back.
    let rec = load_cache_record("k1", "ha", &device, None)
        .expect("no reject")
        .expect("overlay record found");
    assert_eq!(rec.winner, vec![("block_size".to_string(), 128)]);
    assert!(matches!(rec.selection, SelectionMethod::Measured));

    // A foreign device is REJECTED at lookup — a DB tuned elsewhere must
    // never supply winners here.
    let other = DeviceIdentity { device_name: "Other GPU".to_string(), ..test_device() };
    match load_cache_record("k1", "ha", &other, None) {
        Err(CacheReject::DeviceMismatch { .. }) => {}
        other => panic!("foreign device must reject with DeviceMismatch, got {other:?}"),
    }

    // A kernel the DB does not carry falls through to the disk cache path
    // (no entry there either in this tmp-less lookup) — Ok(None), not a
    // reject.
    assert!(matches!(
        load_cache_record("k_absent", "hx", &device, None),
        Ok(None)
    ));
}
