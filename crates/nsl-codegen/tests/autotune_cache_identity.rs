//! Roadmap item 10 (mechanical half): the `@autotune` cache key must describe
//! the machine it was produced on, and an entry that describes some *other*
//! machine must be refused rather than reused.
//!
//! ## What was wrong
//!
//! The cache key was built from `gpu_specs::default_gpu()`, whose documented
//! contract is "the default when auto-detect is unavailable" — it returns
//! `A100-SXM` unconditionally. Every machine therefore hashed the string
//! "A100-SXM / sm_80 / 108 SMs" into its key, so a `.nsl-cache/autotune` entry
//! produced on one GPU was indistinguishable from an entry for any other, and
//! would be read back and used verbatim.
//!
//! `old_key_scheme_collided_across_machines` reproduces that collision, so the
//! gates below are anchored to a defect that is demonstrated rather than
//! asserted.

use nsl_codegen::autotune::{
    build_cache_record, cache_dir, check_cache, hash_kernel_ast, load_cache_record, AutotuneResult,
    CacheReject, DeviceIdentity, SelectionMethod, CACHE_SCHEMA_VERSION, NO_DEVICE_NAME,
};
use std::path::{Path, PathBuf};

fn dev(name: &str, sm_version: u32, sm_count: u32, driver_version: u32) -> DeviceIdentity {
    DeviceIdentity {
        device_name: name.to_string(),
        sm_version,
        sm_count,
        driver_version,
    }
}

/// This box.
fn blackwell() -> DeviceIdentity {
    dev("RTX 5070 Ti", 120, 70, 13_030)
}

/// A machine whose tuning results must never be confused with this box's.
fn hopper() -> DeviceIdentity {
    dev("H100-SXM", 90, 132, 12_040)
}

fn key(device: &DeviceIdentity) -> String {
    hash_kernel_ast(
        "k",
        b"body",
        &vec![("block".to_string(), vec![64, 128])],
        &[],
        device,
    )
}

fn sample_result(device: &DeviceIdentity) -> AutotuneResult {
    AutotuneResult {
        winner: vec![("block".to_string(), 128)],
        all_timings: vec![
            (vec![("block".to_string(), 64)], 1.5),
            (vec![("block".to_string(), 128)], 0.8),
        ],
        device: device.clone(),
        selection: SelectionMethod::Measured,
    }
}

/// Repo root, derived from this package's manifest dir.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/nsl-codegen has two ancestors")
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// The defect, reproduced
// ---------------------------------------------------------------------------

#[test]
fn old_key_scheme_collided_across_machines() {
    // NEGATIVE CONTROL. Rebuild the identity the way the pre-item-10 caller did
    // — from `default_gpu()`, ignoring whatever hardware is actually present —
    // and show that two unmistakably different machines produced the SAME key.
    // If this ever stops collapsing, the control has gone slack and the gates
    // below no longer prove anything.
    let a100 = nsl_codegen::gpu_specs::default_gpu();
    let as_if_v1 = |_actual_machine: &DeviceIdentity| DeviceIdentity {
        device_name: a100.name.to_string(),
        sm_version: a100.sm_version,
        sm_count: a100.num_sms,
        driver_version: 0,
    };

    assert_eq!(
        key(&as_if_v1(&blackwell())),
        key(&as_if_v1(&hopper())),
        "the v1 scheme is supposed to collide here — that is the bug item 10 fixes"
    );

    // And the fix: real identities are distinguishable.
    assert_ne!(
        key(&blackwell()),
        key(&hopper()),
        "a Blackwell key must not equal a Hopper key"
    );
}

#[test]
fn local_identity_is_not_the_database_default() {
    // `default_gpu()` is a fallback for cost modelling, never an identity.
    // Whatever this test host is, the identity must not silently claim A100.
    let local = DeviceIdentity::local();
    let default_spec = nsl_codegen::gpu_specs::default_gpu();
    if local.device_name == NO_DEVICE_NAME {
        // GPU-less machine: the sentinel, not a real card's spec.
        assert_eq!(local, DeviceIdentity::no_device());
        assert_eq!(local.sm_version, 0);
        assert_eq!(local.sm_count, 0);
    } else {
        // A real device was probed. It may legitimately BE an A100, but then
        // the SM count has to come from the driver rather than the database
        // default's constant.
        assert!(
            local.sm_version > 0 && local.sm_count > 0,
            "a probed device must report a real compute capability and SM count, got {}",
            local.describe()
        );
        assert!(
            local.driver_version > 0,
            "a probed device must report a driver version, got {}",
            local.describe()
        );
    }
    // Either way: the sentinel must never be mistakable for a database entry.
    assert!(
        nsl_codegen::gpu_specs::find_gpu(NO_DEVICE_NAME).is_none(),
        "the no-device sentinel must not resolve to a GpuSpec"
    );
    let _ = default_spec;
}

// ---------------------------------------------------------------------------
// Record round-trip and validation
// ---------------------------------------------------------------------------

#[test]
fn a_present_device_must_actually_be_probed() {
    // `local_identity_is_not_the_database_default` passes on a GPU-less machine
    // by taking its sentinel branch, so on its own it cannot tell me whether the
    // driver probe works — it would stay green if `cuda_device_identity` always
    // returned None. This gate is the one that fails in that case.
    //
    // The oracle is the PRE-EXISTING probe (`cuda_device_name`, used by the CSHA
    // planner since 2026-07-14), so this is not the new code checking itself:
    // if that one names a device and the new one cannot identify it, the new one
    // is broken.
    let Some(name) = nsl_runtime::cuda_device_name() else {
        eprintln!("[skip] no CUDA device visible (cuda_device_name returned None)");
        return;
    };
    let local = DeviceIdentity::local();
    eprintln!("probed identity: {}", local.describe());
    assert_ne!(
        local,
        DeviceIdentity::no_device(),
        "cuda_device_name() reports \"{name}\" but cuda_device_identity() gave up — \
         the driver probe is broken"
    );
    assert_eq!(local.device_name, name, "both probes must name the same card");
    assert!(
        local.sm_version >= 50 && local.sm_version < 300,
        "implausible compute capability {} for \"{name}\"",
        local.sm_version
    );
    assert!(
        local.sm_count > 0 && local.sm_count < 1024,
        "implausible SM count {} for \"{name}\"",
        local.sm_count
    );
    assert!(
        local.driver_version >= 10_000,
        "implausible driver version {} — CUDA 10.0 is 10000",
        local.driver_version
    );
}

#[test]
fn record_round_trips_through_serde() {
    let result = sample_result(&blackwell());
    let record = build_cache_record("abc123", "some_kernel", &result);
    let json = serde_json::to_string_pretty(&record).expect("serialize");
    let back: nsl_codegen::autotune::AutotuneCacheRecord =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(record, back);
    assert_eq!(back.schema_version, CACHE_SCHEMA_VERSION);
    assert_eq!(back.device, blackwell());
    assert_eq!(back.winner, vec![("block".to_string(), 128)]);
    assert_eq!(back.winner_ms, 0.8, "winner_ms must be the winner's timing");
    assert_eq!(back.variants_tested, 2);
}

#[test]
fn cost_model_selection_round_trips_and_names_its_spec() {
    let result = AutotuneResult {
        selection: SelectionMethod::CostModel {
            spec: "RTX-5070-Ti".to_string(),
        },
        ..sample_result(&blackwell())
    };
    let record = build_cache_record("k1", "kern", &result);
    let json = serde_json::to_string(&record).expect("serialize");
    // The distinction has to survive to disk: a reader must be able to tell an
    // estimate from a measurement without re-running anything.
    assert!(
        json.contains("cost_model"),
        "selection method must be recorded, got {json}"
    );
    assert!(
        json.contains("RTX-5070-Ti"),
        "the priced-against spec must be recorded, got {json}"
    );
    let back: nsl_codegen::autotune::AutotuneCacheRecord =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        back.selection,
        SelectionMethod::CostModel {
            spec: "RTX-5070-Ti".to_string()
        }
    );
}

/// Write a record to the real cache dir, run `f`, then always remove the file.
fn with_cached_record<T>(
    kernel: &str,
    hash: &str,
    record_json: &str,
    f: impl FnOnce() -> T,
) -> std::io::Result<T> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{kernel}_{hash}.json"));
    std::fs::write(&path, record_json)?;
    let out = f();
    std::fs::remove_file(&path).ok();
    Ok(out)
}

#[test]
fn a_record_from_another_machine_is_refused() {
    let hash = "item10_devmismatch_hash";
    let kernel = "item10_devmismatch_kernel";
    let record = build_cache_record(hash, kernel, &sample_result(&hopper()));
    let json = serde_json::to_string(&record).unwrap();

    let (rejected, missed) = with_cached_record(kernel, hash, &json, || {
        (
            load_cache_record(kernel, hash, &blackwell()),
            check_cache(kernel, hash, &blackwell()),
        )
    })
    .expect("cache dir writable");

    match rejected {
        Err(CacheReject::DeviceMismatch { found, expected }) => {
            assert_eq!(*found, hopper());
            assert_eq!(*expected, blackwell());
        }
        other => panic!("expected DeviceMismatch, got {other:?}"),
    }
    assert!(
        missed.is_none(),
        "a foreign record must read as a cache MISS, not a hit"
    );
}

#[test]
fn the_same_machine_still_hits() {
    // Anti-vacuity for the test above: the rejection must be caused by the
    // device differing, not by the harness never producing a hit at all.
    let hash = "item10_samemachine_hash";
    let kernel = "item10_samemachine_kernel";
    let record = build_cache_record(hash, kernel, &sample_result(&blackwell()));
    let json = serde_json::to_string(&record).unwrap();

    let hit = with_cached_record(kernel, hash, &json, || {
        check_cache(kernel, hash, &blackwell())
    })
    .expect("cache dir writable");

    assert_eq!(
        hit,
        Some(vec![("block".to_string(), 128)]),
        "an entry for this exact machine must be reused"
    );
}

#[test]
fn a_renamed_or_copied_file_is_refused() {
    let kernel = "item10_keymismatch_kernel";
    // Record says it belongs to key A; it is filed under key B.
    let record = build_cache_record("key_A", kernel, &sample_result(&blackwell()));
    let json = serde_json::to_string(&record).unwrap();

    let rejected = with_cached_record(kernel, "key_B", &json, || {
        load_cache_record(kernel, "key_B", &blackwell())
    })
    .expect("cache dir writable");

    match rejected {
        Err(CacheReject::KeyMismatch { found, expected }) => {
            assert_eq!(found, "key_A");
            assert_eq!(expected, "key_B");
        }
        other => panic!("expected KeyMismatch, got {other:?}"),
    }
}

#[test]
fn an_older_schema_is_refused() {
    let kernel = "item10_schema_kernel";
    let hash = "item10_schema_hash";
    let mut record = build_cache_record(hash, kernel, &sample_result(&blackwell()));
    record.schema_version = CACHE_SCHEMA_VERSION - 1;
    let json = serde_json::to_string(&record).unwrap();

    let rejected = with_cached_record(kernel, hash, &json, || {
        load_cache_record(kernel, hash, &blackwell())
    })
    .expect("cache dir writable");

    match rejected {
        Err(CacheReject::SchemaMismatch { found, expected }) => {
            assert_eq!(found, CACHE_SCHEMA_VERSION - 1);
            assert_eq!(expected, CACHE_SCHEMA_VERSION);
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[test]
fn a_pre_item10_cache_file_is_refused_not_misread() {
    // The exact bytes the old `format!`-based writer produced. The old reader
    // recovered a winner from this by locating `"winner":`, then the next `{`,
    // then the next `}` — no schema check, no device check. The danger is not
    // that it fails to parse; it is that it parses into a plausible winner for
    // hardware it never describes. It has to be REFUSED.
    let legacy = r#"{
  "kernel": "legacy_kernel",
  "device": "cost_model",
  "compute_capability": "",
  "sm_count": 0,
  "variants_tested": 2,
  "winner": {"block": 128},
  "median_time_ms": 0.8000,
  "all_timings": [
    {"params": {"block": 64}, "median_ms": 1.5000},
    {"params": {"block": 128}, "median_ms": 0.8000}
  ],
  "timestamp": "1750000000",
  "cache_key": "item10_legacy_hash"
}"#;
    let kernel = "legacy_kernel";
    let hash = "item10_legacy_hash";

    let (rejected, missed) = with_cached_record(kernel, hash, legacy, || {
        (
            load_cache_record(kernel, hash, &blackwell()),
            check_cache(kernel, hash, &blackwell()),
        )
    })
    .expect("cache dir writable");

    assert!(
        matches!(rejected, Err(CacheReject::Unparseable(_))),
        "a v1 file must be refused, got {rejected:?}"
    );
    assert!(
        missed.is_none(),
        "a v1 file must not yield a winner — it records no device identity at all"
    );
}

#[test]
fn a_missing_entry_is_a_miss_not_a_rejection() {
    let got = load_cache_record("item10_absent_kernel", "item10_absent_hash", &blackwell());
    assert_eq!(got, Ok(None), "no file on disk is Ok(None), not an error");
}

#[test]
fn only_device_and_key_mismatches_warn() {
    // Rationale check for `is_noteworthy`. A schema bump is the expected state
    // of a cache directory right after an upgrade; warning per kernel there
    // trains users to ignore the channel. A foreign entry is the thing nobody
    // would otherwise notice.
    assert!(!CacheReject::Unparseable("x".into()).is_noteworthy());
    assert!(!CacheReject::SchemaMismatch {
        found: 1,
        expected: 2
    }
    .is_noteworthy());
    assert!(CacheReject::KeyMismatch {
        found: "a".repeat(64),
        expected: "b".repeat(64)
    }
    .is_noteworthy());
    assert!(CacheReject::DeviceMismatch {
        found: Box::new(hopper()),
        expected: Box::new(blackwell())
    }
    .is_noteworthy());
    assert!(CacheReject::EmptyWinner.is_noteworthy());
}

#[test]
fn reject_descriptions_do_not_panic_on_short_keys() {
    // `describe` truncates keys to 16 chars for readability; a shorter key must
    // not index out of bounds.
    let msg = CacheReject::KeyMismatch {
        found: "ab".to_string(),
        expected: "cd".to_string(),
    }
    .describe();
    assert!(msg.contains("ab") && msg.contains("cd"), "got {msg}");

    let msg = CacheReject::DeviceMismatch {
        found: Box::new(hopper()),
        expected: Box::new(blackwell()),
    }
    .describe();
    assert!(
        msg.contains("H100-SXM") && msg.contains("RTX 5070 Ti"),
        "a device mismatch must name both machines, got {msg}"
    );
}

// ---------------------------------------------------------------------------
// Drift gates
// ---------------------------------------------------------------------------

#[test]
fn the_autotune_key_is_not_built_from_the_database_default() {
    // Tree -> invariant, which is the direction that actually catches
    // regressions: someone reintroducing `default_gpu()` as the cache identity
    // would not think to update a registry.
    let src = std::fs::read_to_string(repo_root().join("crates/nsl-codegen/src/compiler/kernel.rs"))
        .expect("kernel.rs readable");
    let start = src
        .find("fn autotune_select_best")
        .expect("autotune_select_best still exists");
    let end = src[start..]
        .find("\n    // ── FlashAttention")
        .map(|o| start + o)
        .unwrap_or(src.len());
    let body = &src[start..end];

    assert!(
        body.contains("DeviceIdentity::local()"),
        "the cache key must be built from the driver-probed local identity"
    );
    let key_call = body
        .find("hash_kernel_ast")
        .expect("the key is still built by hash_kernel_ast");
    let key_args_end = body[key_call..]
        .find(");")
        .map(|o| key_call + o)
        .expect("hash_kernel_ast call terminates");
    assert!(
        !body[key_call..key_args_end].contains("default_gpu"),
        "default_gpu() must not feed the cache key — it returns A100-SXM on every machine"
    );
}

#[test]
fn autotune_selection_method_is_honest() {
    // The module doc states that the measured path is not wired and that cache
    // entries therefore record CostModel. Both halves are gated here so the doc
    // cannot rot into a false claim about what NSL measures.
    let kernel_rs =
        std::fs::read_to_string(repo_root().join("crates/nsl-codegen/src/compiler/kernel.rs"))
            .expect("kernel.rs readable");
    assert!(
        kernel_rs.contains("find_best_variant_cost_model("),
        "the production path is the cost-model path"
    );

    // Nothing outside autotune.rs may call the measured path. When someone
    // wires it, this fails and they must update the module doc that says it is
    // unwired — which is the point.
    let mut callers = Vec::new();
    let mut dirs = vec![repo_root().join("crates")];
    let mut files_scanned = 0usize;
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                if path.ends_with("nsl-codegen/src/autotune.rs") || path == manifest_of_this_test() {
                    continue;
                }
                files_scanned += 1;
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                // `find_best_variant_cost_model(` also contains
                // `find_best_variant`, so match on the exact call opener.
                if text.contains("find_best_variant(") {
                    callers.push(path.display().to_string());
                }
            }
        }
    }
    assert!(
        files_scanned > 500,
        "the walk must actually reach the tree; only scanned {files_scanned} files"
    );
    assert!(
        callers.is_empty(),
        "find_best_variant (the measured path) gained callers: {callers:?}. \
         Update autotune.rs's module doc — it currently states the measured path is unwired."
    );
}

#[test]
fn flash_attention_block_sizes_are_never_tuned() {
    // The flash-attention path printed "--no-autotune, using middle values for
    // flash_attention", which reads as though the flag caused that choice. It
    // did not: `select_middle_values` is the primary config unconditionally on
    // that path, and no cost model or benchmark runs there at all. If real
    // selection is ever added, this gate fails and the message must be revisited
    // along with it.
    let src = std::fs::read_to_string(repo_root().join("crates/nsl-codegen/src/compiler/kernel.rs"))
        .expect("kernel.rs readable");
    let anchor = src
        .find("Select the middle values as the primary config")
        .expect("the flash-attention primary-config selection still exists");
    // Look at the whole flash-attention synthesis region leading up to it.
    let region_start = src[..anchor]
        .rfind("let tune_params: crate::autotune::TuningParams")
        .expect("the flash-attention tuning-param block still exists");
    let region = &src[region_start..anchor];
    for tuner in [
        "find_best_variant",
        "find_best_variant_cost_model",
        "hash_kernel_ast",
        "check_cache",
    ] {
        assert!(
            !region.contains(tuner),
            "the flash-attention path now calls {tuner} — it does real selection, \
             so the '--no-autotune changes nothing here' message is no longer true"
        );
    }
    assert!(
        !region.contains("--no-autotune, using middle values"),
        "the misleading wording came back"
    );
}

fn manifest_of_this_test() -> PathBuf {
    repo_root().join("crates/nsl-codegen/tests/autotune_cache_identity.rs")
}
