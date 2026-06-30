//! CEP G18 — end-to-end binary claim.
//!
//! Paper §6.3 promises: the artifacts CEP emits when it selects a chosen
//! design (SP1's sliced safetensors + SP2's rewritten source) self-describe
//! to the SAME compilation profile that CEP showed the user. If they drift
//! — even on one hardware metric — the user's binary claim is broken
//! (e.g. the report says "Binary size: 4MB" but the actual artifact would
//! compile to a 6MB binary).
//!
//! ## What this test asserts
//!
//! Two complementary checks:
//!
//! 1. **Spec round-trip via the oracle.** `cep_oracle::evaluate` is a pure
//!    function of `ModelSpec`. Re-parsing the SP2-emitted source and
//!    extracting a `ModelSpec` from it should yield a spec that evaluates
//!    to exactly `chosen.profile` — bit-exact on every integer hardware
//!    metric (param_bytes, peak_memory_bytes, binary_size_bytes,
//!    kernel_launches, total_flops, total_hbm_bytes) and within ε on the
//!    f64 latency. This catches any drift between what CEP reports and
//!    what the emitted source DESCRIBES.
//!
//! 2. **Weight artifact shape claim.** The oracle does not consume the
//!    weight file directly, so step 1 alone would let a regression in
//!    `cep_slice` (wrong tensor shapes in the sliced safetensors) slip
//!    through. `cross_check_dims` covers wq / wk / wv / w_gate / w_up,
//!    but DOESN'T cover `wo` and `w_down` — the two axis-0-reduction
//!    tensors that are the trickiest under pruning. We add explicit
//!    shape claims for both against `reextracted_spec`.
//!
//! Together with [`cep_emit_source`] / [`cep_slice`]'s unit tests
//! (mechanics) and [`cep_slice_cli`] / [`cep_emit_source_cli`] (CLI
//! plumbing), this closes G18 from the PR #227 deferral list.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nsl_codegen::cep::{plan_to_prune_delta, run_prune, CepPruneInput};
use nsl_codegen::cep_emit_source::apply_prune_delta_to_source;
use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};
use nsl_codegen::cep_importance::RooflineSlackTable;
use nsl_codegen::cep_oracle::evaluate;
use nsl_codegen::cep_search::{Constraints, Granularity};
use nsl_codegen::cep_slice::{apply_prune_delta_to_weights, write_sliced_weights};
use nsl_codegen::gpu_specs::find_gpu;
use nsl_codegen::weight_aware::WeightMap;
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;
use tempfile::TempDir;

const TARGET: &str = "H100";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/nsl-codegen")
        .to_path_buf()
}

fn canonical_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_canonical_small.nsl")
}

fn ensure_stdlib_path() {
    if std::env::var("NSL_STDLIB_PATH").is_err() {
        std::env::set_var("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    }
}

/// Parse the canonical fixture and return its raw AST module plus a name
/// resolver. CEP's recognizer walks the AST directly, so semantic analysis
/// is not required for `extract_model_spec` / `apply_prune_delta_to_source`.
fn parse_fixture(source: &str) -> (nsl_ast::Module, impl Fn(nsl_ast::Symbol) -> String) {
    let mut interner = nsl_lexer::Interner::new();
    let file_id = nsl_errors::FileId(0);
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, file_id, &mut interner);
    assert!(
        lex_diags.iter().all(|d| d.level != nsl_errors::Level::Error),
        "lex must succeed on canonical fixture: {lex_diags:?}"
    );
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parse_result
            .diagnostics
            .iter()
            .all(|d| d.level != nsl_errors::Level::Error),
        "parse must succeed on canonical fixture: {:?}",
        parse_result.diagnostics
    );
    let resolve = move |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
    (parse_result.module, resolve)
}

/// Write a safetensors file matching `cep_canonical_small.nsl` shapes:
/// d_model=64, 2 layers, 4 heads, 2 kv heads, head_dim=16, d_ff=128.
fn write_canonical_safetensors(path: &Path) {
    let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
    for l in 0..2 {
        specs.push((format!("blocks.{l}.attn.wq"), vec![64, 64]));
        specs.push((format!("blocks.{l}.attn.wk"), vec![64, 32]));
        specs.push((format!("blocks.{l}.attn.wv"), vec![64, 32]));
        specs.push((format!("blocks.{l}.attn.wo"), vec![64, 64]));
        specs.push((format!("blocks.{l}.ffn.w_gate"), vec![64, 128]));
        specs.push((format!("blocks.{l}.ffn.w_up"), vec![64, 128]));
        specs.push((format!("blocks.{l}.ffn.w_down"), vec![128, 64]));
    }
    // The fixture's embed slot is intentionally not in the WeightMap —
    // CEP's per-layer recognizer only consumes block-keyed entries; the
    // embed is handled by passthrough at slice time (and the resulting
    // sliced WeightMap simply has no `embed` entry, mirroring the input).

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
    let bytes = serialize(&views, &None).expect("serialize canonical safetensors");
    std::fs::write(path, bytes).expect("write canonical safetensors");
}

/// G18 closure: SP1's sliced safetensors + SP2's rewritten source feed back
/// through the oracle and produce a CompilationProfile that AGREES with
/// `chosen.profile` on every integer hardware metric.
///
/// If this assertion ever fires, the user's CEP report and the artifacts
/// the report describes have diverged — the binary claim is broken.
#[test]
fn cep_artifacts_self_describe_to_chosen_profile() {
    ensure_stdlib_path();
    let tmp = TempDir::new().unwrap();

    let original_source =
        std::fs::read_to_string(canonical_fixture()).expect("read canonical fixture source");
    let (module, resolve) = parse_fixture(&original_source);

    let baseline_spec = extract_model_spec(&module, &resolve).expect("baseline recognized");

    let weights_path = tmp.path().join("baseline.safetensors");
    write_canonical_safetensors(&weights_path);
    let baseline_weights = WeightMap::load(&weights_path).expect("load baseline weights");
    cross_check_dims(&baseline_spec, &baseline_weights, &resolve)
        .expect("baseline spec and baseline weights agree on shapes");

    let plan = run_prune(CepPruneInput {
        spec: baseline_spec.clone(),
        weights: Some(&baseline_weights),
        target: TARGET,
        constraints: Constraints {
            // Pick a sparsity target large enough to force BOTH a head
            // edit AND an FFN-width edit on this fixture so the test
            // exercises both axes. The non-trivial-delta guard below
            // catches no-op deltas; the FFN-fired guard further catches
            // the "all reduction came from head prune" edge case (low-
            // sparsity targets satisfy on the first head group and leave
            // SP1's FFN row-slicing on `w_down` completely untested).
            target_sparsity: 0.35,
            ..Default::default()
        },
        granularity: Granularity::HeadAndFfn,
        roofline_slack: RooflineSlackTable::default(),
    });

    let chosen = plan
        .outcome
        .chosen
        .clone()
        .expect("greedy prune always reports a chosen candidate");
    let baseline_profile = plan
        .outcome
        .baseline
        .clone()
        .expect("baseline profile always recorded");

    // The non-trivial-delta guard: chosen MUST be a strict reduction over
    // baseline on params, otherwise the round-trip below is trivially
    // satisfied and any regression hides.
    assert!(
        chosen.profile.param_bytes < baseline_profile.param_bytes,
        "test fixture must force at least one prune edit \
         (chosen={} bytes, baseline={} bytes); raise target_sparsity if this regresses",
        chosen.profile.param_bytes,
        baseline_profile.param_bytes,
    );

    let delta = plan_to_prune_delta(&plan, &baseline_spec);

    // FFN-fired guard: an all-heads-only delta would never exercise SP1's
    // FFN row-slicing on `w_down` (the trickiest tensor — axis-0 reduction).
    // Without this guard the test could pass with target_sparsity tuned just
    // low enough that head pruning alone satisfies it.
    assert!(
        delta.per_layer.iter().any(|ld| ld.new_d_ff.is_some()),
        "delta must include at least one FFN-width edit so SP1's w_down slicing is exercised; \
         raise target_sparsity if this regresses"
    );
    assert!(
        delta
            .per_layer
            .iter()
            .any(|ld| !ld.pruned_heads.is_empty()),
        "delta must include at least one head edit so SP1's wo slicing is exercised; \
         raise target_sparsity if this regresses"
    );

    // --- SP2: emit the rewritten source.
    let emitted_source =
        apply_prune_delta_to_source(&original_source, &module, &resolve, &baseline_spec, &delta)
            .expect("SP2 emits rewritten source");

    // --- SP1: slice the weights and write to a fresh safetensors file.
    let sliced_map = apply_prune_delta_to_weights(&baseline_weights, &baseline_spec, &delta)
        .expect("SP1 slices weights");
    let sliced_path = tmp.path().join("pruned.safetensors");
    write_sliced_weights(&sliced_map, &sliced_path).expect("write sliced safetensors");

    // --- Re-load the artifacts as if a downstream consumer received them.
    let (emitted_module, emitted_resolve) = parse_fixture(&emitted_source);
    let reextracted_spec =
        extract_model_spec(&emitted_module, &emitted_resolve).expect("emitted source recognized");
    let sliced_weights = WeightMap::load(&sliced_path).expect("re-load sliced safetensors");

    // The artifact pair must be self-consistent before we can trust any
    // claim against it.
    cross_check_dims(&reextracted_spec, &sliced_weights, &emitted_resolve)
        .expect("emitted source and sliced weights agree on shapes");

    // `cross_check_dims` covers wq / wk / wv / w_gate / w_up against
    // `n_heads·head_dim` / `n_kv_heads·head_dim` / `d_ff`. It deliberately
    // skips `wo` (attention output projection, `[n_heads·head_dim, d_model]`)
    // and `w_down` (FFN down projection, `[d_ff, d_model]`) — yet those
    // two tensors are EXACTLY the ones whose axis-0 dimension shrinks
    // under pruning. A regression in SP1's row-slicing on either would
    // slip through `cross_check_dims`. Cover them explicitly here against
    // `reextracted_spec` (which equals `chosen.spec` if the rest of the
    // round-trip is correct).
    for l in 0..reextracted_spec.n_layers as usize {
        let q_proj = (reextracted_spec.n_heads[l] * reextracted_spec.head_dim[l]) as usize;
        let d_ff = reextracted_spec.d_ff[l] as usize;
        let d_model = reextracted_spec.d_model as usize;

        let wo_key = format!("blocks.{l}.attn.wo");
        let wo_entry = sliced_weights
            .get(&wo_key)
            .unwrap_or_else(|| panic!("{wo_key} present in sliced weights"));
        let wo_ok = wo_entry.shape == [q_proj, d_model] || wo_entry.shape == [d_model, q_proj];
        assert!(
            wo_ok,
            "{wo_key} row-slice drift: sliced shape {:?} disagrees with chosen \
             [n_heads·head_dim, d_model] = [{q_proj}, {d_model}]",
            wo_entry.shape
        );

        let wd_key = format!("blocks.{l}.ffn.w_down");
        let wd_entry = sliced_weights
            .get(&wd_key)
            .unwrap_or_else(|| panic!("{wd_key} present in sliced weights"));
        let wd_ok = wd_entry.shape == [d_ff, d_model] || wd_entry.shape == [d_model, d_ff];
        assert!(
            wd_ok,
            "{wd_key} row-slice drift: sliced shape {:?} disagrees with chosen \
             [d_ff, d_model] = [{d_ff}, {d_model}]",
            wd_entry.shape
        );
    }

    // --- Re-evaluate the oracle on the pruned artifact pair.
    let gpu = find_gpu(TARGET).expect("H100 GPU spec available");
    let reeval_profile = evaluate(&reextracted_spec, gpu).expect("re-evaluate pruned spec");

    // --- THE BINARY CLAIM.
    assert_eq!(
        reeval_profile.param_bytes, chosen.profile.param_bytes,
        "param_bytes drift: re-evaluated artifact reports {} bytes; \
         CEP told the user {} bytes",
        reeval_profile.param_bytes, chosen.profile.param_bytes
    );
    assert_eq!(
        reeval_profile.peak_memory_bytes, chosen.profile.peak_memory_bytes,
        "peak_memory_bytes drift: re-evaluated={}, chosen={}",
        reeval_profile.peak_memory_bytes, chosen.profile.peak_memory_bytes
    );
    assert_eq!(
        reeval_profile.binary_size_bytes, chosen.profile.binary_size_bytes,
        "binary_size_bytes drift: re-evaluated={}, chosen={}",
        reeval_profile.binary_size_bytes, chosen.profile.binary_size_bytes
    );
    assert_eq!(
        reeval_profile.kernel_launches, chosen.profile.kernel_launches,
        "kernel_launches drift: re-evaluated={}, chosen={}",
        reeval_profile.kernel_launches, chosen.profile.kernel_launches
    );
    assert_eq!(
        reeval_profile.total_flops, chosen.profile.total_flops,
        "total_flops drift: re-evaluated={}, chosen={}",
        reeval_profile.total_flops, chosen.profile.total_flops
    );
    assert_eq!(
        reeval_profile.total_hbm_bytes, chosen.profile.total_hbm_bytes,
        "total_hbm_bytes drift: re-evaluated={}, chosen={}",
        reeval_profile.total_hbm_bytes, chosen.profile.total_hbm_bytes
    );

    // Latency is f64 — assert near-equality rather than bit-exact so a
    // legitimate refactor that reorders FLOPs/bytes accumulation by a
    // single ULP doesn't flake. A 1e-6 relative tolerance is well below
    // the per-layer rounding the report itself prints.
    let lat_a = reeval_profile.estimated_latency_us;
    let lat_b = chosen.profile.estimated_latency_us;
    let lat_tol = lat_b.abs() * 1e-6 + 1e-9;
    assert!(
        (lat_a - lat_b).abs() <= lat_tol,
        "estimated_latency_us drift beyond ε: re-evaluated={lat_a}, chosen={lat_b}, tol={lat_tol}"
    );
}
