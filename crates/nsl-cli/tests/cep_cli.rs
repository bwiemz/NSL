//! Task 9: end-to-end CEP CLI integration tests.
//!
//! Exercises the now-complete CEP frontend pipeline through the `nsl` binary:
//!
//!   1. `build --cep-prune` without `--weights` fails fast (the weights gate
//!      fires before any analysis, so the fixture need only be syntactically
//!      valid — but we use the canonical analyzable fixture anyway).
//!   2. `build --cep-prune --weights <w>` on the canonical analyzable fixture
//!      runs the full pipeline (frontend analysis → recognizer → cross-check →
//!      search), prints the Pruning Report, and writes a versioned delta
//!      JSON (`cep_version == 2`, `mode == "prune"`). A plan that misses its
//!      target exits non-zero and writes nothing.
//!   3. `build --cep-prune` on `examples/gpt2.nsl` fails (gpt2 carries no
//!      `@cep_prune` decorator and its fused-QKV attention is unrecognized).
//!   4. `check --cep-search` on the searchable fixture runs the search oracle,
//!      prints the Architecture Search Report, and writes a delta JSON
//!      (`mode == "search"`).
//!
//! These spawn the real `nsl` binary, so they set `NSL_STDLIB_PATH` exactly as
//! `cpdt_cli.rs` does — the full frontend (`frontend_with_flags`) analyzes the
//! source and `process::exit(1)`s on any error-level diagnostic, so the happy
//! paths require fixtures that actually type-check.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

fn canonical_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_canonical_small.nsl")
}

fn searchable_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_searchable.nsl")
}

/// Write a safetensors file with the seven weight tensors of each layer of
/// the canonical fixture (d_model=64, 2 layers, 4 heads, 2 kv heads,
/// head_dim=16, d_ff=128), all zeros, minus any tensor named in `omit`.
/// `bracket` selects `blocks[0].attn.wq` (as `model_save` and `nsl export`
/// write it) over `blocks.0.attn.wq`.
fn write_weights(path: &Path, bracket: bool, omit: &[&str]) {
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};

    let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
    for l in 0..2 {
        for (rest, shape) in [
            ("attn.wq", vec![64, 64]),
            ("attn.wk", vec![64, 32]),
            ("attn.wv", vec![64, 32]),
            ("attn.wo", vec![64, 64]),
            ("ffn.w_gate", vec![64, 128]),
            ("ffn.w_up", vec![64, 128]),
            ("ffn.w_down", vec![128, 64]),
        ] {
            let name = if bracket {
                format!("blocks[{l}].{rest}")
            } else {
                format!("blocks.{l}.{rest}")
            };
            if !omit.contains(&name.as_str()) {
                specs.push((name, shape));
            }
        }
    }
    let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = specs
        .into_iter()
        .map(|(name, shape)| {
            let elems: usize = shape.iter().product();
            (name, shape, vec![0u8; elems * 4])
        })
        .collect();
    let views: HashMap<String, TensorView<'_>> = buffers
        .iter()
        .map(|(name, shape, data)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), data.as_slice()).unwrap(),
            )
        })
        .collect();
    let bytes = serialize(&views, None).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_matching_safetensors(path: &Path) {
    write_weights(path, false, &[]);
}

fn nsl() -> Command {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd
}

/// The canonical fixture with its `@cep_prune(...)` line replaced.
fn canonical_with_decorator(dir: &Path, decorator: &str) -> PathBuf {
    let src = fs::read_to_string(canonical_fixture()).unwrap();
    let original = r#"@cep_prune(target = h100, sparsity = 0.3, granularity = "head+ffn")"#;
    assert!(
        src.contains(original),
        "the canonical fixture's decorator changed"
    );
    let path = dir.join("model.nsl");
    fs::write(&path, src.replace(original, decorator)).unwrap();
    path
}

#[test]
fn cep_prune_without_weights_errors() {
    let mut cmd = nsl();
    cmd.arg("build").arg(canonical_fixture()).arg("--cep-prune");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--weights"));
}

/// The canonical fixture has 90,432 parameters (embedding 256·64, two blocks
/// of 36,992, final norm 64). The 30% the decorator asks for is within what
/// head and FFN pruning can remove (47.6%), so the plan succeeds.
#[test]
fn cep_prune_end_to_end_writes_delta() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    let out_path = tmp.path().join("delta.cep.json");
    write_matching_safetensors(&weights_path);

    let mut cmd = nsl();
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CEP Pruning Report ==="))
        .stdout(predicate::str::contains(
            "Sparsity target and constraints met.",
        ))
        .stdout(predicate::str::contains("(target 30.0%)"))
        .stdout(predicate::str::contains("CEP delta written"));

    let delta = fs::read_to_string(&out_path).expect("delta JSON written");
    let v: serde_json::Value = serde_json::from_str(&delta).expect("delta is valid JSON");
    assert_eq!(v["cep_version"], 2);
    assert_eq!(v["mode"], "prune");
    assert_eq!(v["feasible"], true);
    assert_eq!(v["baseline_profile"]["param_bytes"], 90_432 * 4);
    let achieved = v["sparsity"]["achieved"].as_f64().unwrap();
    // At least the target, and no more than one step (a key/value group,
    // 6,144 parameters) beyond it.
    assert!(
        (0.3..0.3 + 6_144.0 / 90_432.0).contains(&achieved),
        "achieved {achieved}"
    );
}

/// `model_save` and `nsl export` name a layer's tensors `blocks[0].…`.
#[test]
fn cep_prune_accepts_model_save_tensor_names() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    write_weights(&weights_path, true, &[]);
    let mut cmd = nsl();
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(tmp.path().join("d.json"));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CEP delta written"));
}

/// A layer without its output projection cannot be pruned consistently.
#[test]
fn cep_prune_refuses_weights_missing_a_projection() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    let out = tmp.path().join("d.json");
    write_weights(&weights_path, false, &["blocks.1.attn.wo"]);
    let mut cmd = nsl();
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("blocks[1].attn.wo"))
        .stderr(predicate::str::contains("no such tensor"));
    assert!(!out.exists());
}

/// Head and FFN pruning can remove at most 47.6% of the canonical fixture, so
/// a 90% request cannot be met. The command must fail and leave no file that
/// could be mistaken for the result.
#[test]
fn cep_prune_that_misses_its_target_fails_and_writes_nothing() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    let out = tmp.path().join("d.json");
    let weights_out = tmp.path().join("pruned.safetensors");
    write_matching_safetensors(&weights_path);
    let mut cmd = nsl();
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--cep-sparsity")
        .arg("0.9")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out)
        .arg("--cep-emit-weights")
        .arg(&weights_out);
    cmd.assert()
        .failure()
        .stdout(predicate::str::contains("NOT MET: sparsity target missed"))
        .stdout(predicate::str::contains("47.6% removed"))
        .stdout(predicate::str::contains("constraints met.").not())
        .stderr(predicate::str::contains("no file was written"));
    assert!(!out.exists(), "no delta for a plan that failed");
    assert!(!weights_out.exists(), "no weights for a plan that failed");
}

/// A latency limit below the baseline is something to prune towards: with no
/// sparsity asked for, the plan still removes components until the limit
/// holds. One layer costs 8 kernel launches at 5 µs, so no head or FFN step
/// can get a two-layer model under 60 µs; that limit fails.
#[test]
fn cep_prune_treats_a_latency_limit_as_a_target() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    write_matching_safetensors(&weights_path);

    // Read the baseline latency off the delta of a run that changes nothing.
    let plain = canonical_with_decorator(
        tmp.path(),
        r#"@cep_prune(target = h100, sparsity = 0.0, granularity = "head+ffn")"#,
    );
    let out = tmp.path().join("plain.json");
    nsl()
        .arg("build")
        .arg(&plain)
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Prune log (0 steps):"));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
    let baseline = v["baseline_profile"]["estimated_latency_us"]
        .as_f64()
        .unwrap();
    assert!(
        baseline > 90.0,
        "18 launches at 5us alone are 90us; got {baseline}"
    );

    // A limit just under the baseline: reachable, and steps are needed.
    let limit = baseline - 0.01;
    let limited = canonical_with_decorator(
        tmp.path(),
        &format!(
            r#"@cep_prune(target = h100, sparsity = 0.0, granularity = "head+ffn", constraints = {{latency_us: {limit}}})"#
        ),
    );
    let out = tmp.path().join("limited.json");
    nsl()
        .arg("build")
        .arg(&limited)
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Sparsity target and constraints met.",
        ));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
    let chosen = v["chosen"]["profile"]["estimated_latency_us"]
        .as_f64()
        .unwrap();
    assert!(
        chosen <= limit,
        "chosen {chosen} must be within the limit {limit}"
    );
    assert!(
        v["sparsity"]["achieved"].as_f64().unwrap() > 0.0,
        "steps were needed to get there"
    );

    // A limit no head or FFN step can reach.
    let impossible = canonical_with_decorator(
        tmp.path(),
        r#"@cep_prune(target = h100, sparsity = 0.0, granularity = "head+ffn", constraints = {latency_us: 60.0})"#,
    );
    let out = tmp.path().join("impossible.json");
    nsl()
        .arg("build")
        .arg(&impossible)
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out)
        .assert()
        .failure()
        .stdout(predicate::str::contains("NOT MET: latency limit missed"))
        .stderr(predicate::str::contains("latency limit missed"));
    assert!(!out.exists());
}

#[test]
fn cep_prune_refuses_gpt2() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    write_matching_safetensors(&weights_path);

    let mut cmd = nsl();
    cmd.arg("build")
        .arg(workspace_root().join("examples/gpt2.nsl"))
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path);
    // gpt2 has no @cep_prune decorator and its fused-QKV attention is not the
    // canonical GQA structure — either way the prune path must refuse.
    cmd.assert().failure();
}

/// 90,432 parameters; 2 blocks of 8 kernel launches plus embedding and head.
#[test]
fn cep_profile_prints_the_estimate_and_what_it_assumes() {
    let mut cmd = nsl();
    cmd.arg("check")
        .arg(canonical_fixture())
        .arg("--cep-profile")
        .arg("--cep-target")
        .arg("H100-SXM");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CEP Compilation Profile"))
        .stdout(predicate::str::contains("nothing is compiled or measured"))
        .stdout(predicate::str::contains("Target: H100-SXM"))
        .stdout(predicate::str::contains("Params: 0.1M (90432)"))
        .stdout(predicate::str::contains(
            "Estimates are for one forward pass of batch 1 x 2048 tokens, 4-byte values, tied output head.",
        ))
        .stdout(predicate::str::contains("Kernel launches per forward (fusion template): 18"))
        .stdout(predicate::str::contains("us per forward pass ("))
        .stdout(predicate::str::contains("us/token over 2048 tokens)"));
}

/// The shipped 50M model ties its output head to the embedding: 48,767,488
/// parameters, not the 73.9M that counting the head again gives.
#[test]
fn cep_profile_counts_a_tied_head_once() {
    let mut cmd = nsl();
    cmd.arg("check")
        .arg(workspace_root().join("models/coder50m/model.nsl"))
        .arg("--cep-profile")
        .arg("--cep-target")
        .arg("H100-SXM");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Params: 48.8M (48767488)"))
        .stdout(predicate::str::contains("batch 1 x 1024 tokens"))
        .stdout(predicate::str::contains(
            "Kernel launches per forward (fusion template): 66",
        ));
}

// --cep-profile with unknown target must fail with a clear error message.
#[test]
fn cep_profile_rejects_unknown_target() {
    let mut cmd = nsl();
    cmd.arg("check")
        .arg(canonical_fixture())
        .arg("--cep-profile")
        .arg("--cep-target")
        .arg("NoSuchGPU");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unknown CEP target"));
}

// --cep-search and --cep-profile are mutually exclusive.
#[test]
fn cep_search_and_profile_are_mutually_exclusive() {
    let mut cmd = nsl();
    cmd.arg("check")
        .arg(canonical_fixture())
        .arg("--cep-search")
        .arg("--cep-profile");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("mutually exclusive"));
}

/// The fixture spans d_model {256, 384, 512} x n_heads {4, 6, 8} and asks for
/// the lowest latency. The estimate grows with both axes, so the order of the
/// corners is known without running it: d=256/H=4 first, d=512/H=8 last.
#[test]
fn cep_search_ranks_the_fixture() {
    let tmp = TempDir::new().unwrap();
    let out_path = tmp.path().join("search.cep.json");

    let mut cmd = nsl();
    cmd.arg("check")
        .arg(searchable_fixture())
        .arg("--cep-search")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--cep-out")
        .arg(&out_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "=== CEP Architecture Search Report ===",
        ))
        .stdout(predicate::str::contains(
            "Search space: 9 candidates, 9 distinct",
        ))
        .stdout(predicate::str::contains(
            "Feasible (constraints met): 9 candidates",
        ))
        .stdout(predicate::str::contains("Objective: min_latency"))
        .stdout(predicate::str::contains(
            "  1. d=256, L=6, H=4, KV=2, FFN=1024",
        ))
        .stdout(predicate::str::contains("CEP delta written"));

    let delta = fs::read_to_string(&out_path).expect("delta JSON written");
    let v: serde_json::Value = serde_json::from_str(&delta).expect("delta is valid JSON");
    assert_eq!(v["mode"], "search");
    let ranked = v["ranked_candidates"].as_array().unwrap();
    assert_eq!(ranked.len(), 9);
    let dims = |i: usize| {
        (
            ranked[i]["spec"]["d_model"].as_u64().unwrap(),
            ranked[i]["spec"]["n_heads"][0].as_u64().unwrap(),
        )
    };
    assert_eq!(dims(0), (256, 4));
    assert_eq!(dims(8), (512, 8));
    let latency = |i: usize| {
        ranked[i]["profile"]["estimated_latency_us"]
            .as_f64()
            .unwrap()
    };
    for i in 1..9 {
        assert!(
            latency(i - 1) < latency(i),
            "rank {i} is not slower than rank {}",
            i - 1
        );
    }
}

/// With a limit no candidate meets there is nothing to recommend: the command
/// fails, says so, marks what it lists, and writes no file.
#[test]
fn cep_search_with_no_feasible_candidate_fails() {
    let tmp = TempDir::new().unwrap();
    let src = fs::read_to_string(searchable_fixture()).unwrap();
    let original = "@cep_search(target = h100, objective = min_latency)";
    assert!(src.contains(original));
    let file = tmp.path().join("search.nsl");
    fs::write(
        &file,
        src.replace(
            original,
            "@cep_search(target = h100, objective = min_latency, constraints = {peak_memory_bytes: 1})",
        ),
    )
    .unwrap();
    let out_path = tmp.path().join("search.cep.json");
    let mut cmd = nsl();
    cmd.arg("check")
        .arg(&file)
        .arg("--cep-search")
        .arg("--cep-out")
        .arg(&out_path);
    cmd.assert()
        .failure()
        .stdout(predicate::str::contains(
            "Feasible (constraints met): 0 candidates",
        ))
        .stdout(predicate::str::contains(
            "NOT MET: no candidate meets the constraints.",
        ))
        .stdout(predicate::str::contains("(infeasible)"))
        .stdout(predicate::str::contains("Top 3 architectures").not())
        .stderr(predicate::str::contains("no architecture to recommend"));
    assert!(!out_path.exists());
}

/// `@cep_search` has no default objective: the one the proposal recommends
/// ranks the smallest block stack first, which nobody asked for.
#[test]
fn cep_search_requires_an_objective() {
    let tmp = TempDir::new().unwrap();
    let src = fs::read_to_string(searchable_fixture()).unwrap();
    let file = tmp.path().join("search.nsl");
    fs::write(
        &file,
        src.replace(
            "@cep_search(target = h100, objective = min_latency)",
            "@cep_search(target = h100)",
        ),
    )
    .unwrap();
    let mut cmd = nsl();
    cmd.arg("check").arg(&file).arg("--cep-search");
    cmd.assert().failure().stderr(predicate::str::contains(
        "@cep_search: an objective is required",
    ));
}
