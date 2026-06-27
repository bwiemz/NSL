//! §5 FASE + PCA composition regression tests (CFTP v8).
//!
//! Pins that a single train block with BOTH triggers active simultaneously:
//!   * `grad_accumulation = 4`            -> FASE Deferred mode (§2.1-2.4)
//!   * dataset with `packing = true`      -> PCA segment-masked attention (§4.2)
//! produces a `TrainBlockReport` with BOTH sections populated, AND that
//! end-to-end compilation does not crash. This is the gap closure for paper
//! §5 (composition) — prior tests cover each optimization in isolation; this
//! is the only test that verifies they co-activate on the same train block.

use std::path::{Path, PathBuf};
use std::process::Command;

use nsl_codegen::fase::FaseMode;
use nsl_codegen::training_report::build_report;
use nsl_codegen::CompileOptions;
use nsl_errors::FileId;
use nsl_lexer::{tokenize, Interner};

// ─── Fixture helpers ────────────────────────────────────────────────────────

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> String {
    std::fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"))
}

/// Canonical §5 composition source — single source of truth for the
/// positive-path tests (report-level pin, codegen pin, subprocess CLI).
/// Lives at `tests/fixtures/training_report_fase_pca_composition.nsl`.
const COMPOSED_FIXTURE: &str = "training_report_fase_pca_composition.nsl";

/// Minimal model declaration so inline sources semantic-check. Train block
/// configs reference `model = m`; semantic analysis demands `m` be in scope.
/// (The canonical fixture has its own model decl.)
const MODEL_PREAMBLE: &str = concat!(
    "model Linear:\n",
    "    w: Tensor = ones([2, 1])\n",
    "    fn forward(self, x: Tensor) -> Tensor:\n",
    "        return x @ self.w\n",
    "\n",
    "let m = Linear()\n",
    "\n",
);

// ─── Parse + semantic ───────────────────────────────────────────────────────

/// Parse then run the semantic checker; returns (ast, interner, diagnostics).
/// This anchors the v8 semantic-checker fix in every bare-AST test — if the
/// `data:` section bypass or dataset-field whitelist regresses, the relevant
/// test's `assert_no_semantic_errors` panic will surface it.
fn parse_and_check(src: &str) -> (nsl_ast::Module, Interner, Vec<nsl_errors::Diagnostic>) {
    let mut interner = Interner::new();
    let (tokens, _lex_diags) = tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    (parsed.module, interner, analysis.diagnostics)
}

fn assert_no_semantic_errors(diags: &[nsl_errors::Diagnostic]) {
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(
        errors.is_empty(),
        "expected zero semantic errors; got {}: {:?}",
        errors.len(),
        errors
    );
}

// ─── Positive composition (§5 paper-canonical) ──────────────────────────────

/// §5 paper-canonical composition: dataset packing + grad_accumulation
/// co-activate on the same train block. Pins via the canonical fixture +
/// semantic-checker pass, so the test exercises every layer the
/// composition has to flow through.
#[test]
fn fase_deferred_and_pca_co_activate_on_same_train_block() {
    let src = read_fixture(COMPOSED_FIXTURE);
    let (ast, interner, diags) = parse_and_check(&src);
    assert_no_semantic_errors(&diags);
    let report = build_report(&ast, &interner, &fixture_path(COMPOSED_FIXTURE));

    assert_eq!(
        report.train_blocks.len(),
        1,
        "expected exactly one train block"
    );
    let block = &report.train_blocks[0];

    // FASE: grad_accumulation=4 + AdamW -> Deferred mode (paper §2.2).
    assert_eq!(
        block.fase.plan.mode,
        FaseMode::Deferred,
        "expected FASE Deferred (grad_accumulation=4 + AdamW); got {:?}",
        block.fase.plan.mode
    );
    assert_eq!(block.fase.plan.accumulation, 4, "expected accumulation=4");

    // PCA: packing=true on referenced dataset -> Some(PcaSection).
    let pca = block
        .pca
        .as_ref()
        .expect("expected PCA section to be populated when packing=true");
    assert!(
        pca.packing_config.enabled,
        "expected packing_config.enabled=true"
    );
    // PCA detection should pick a real strategy (NOT NoPacking) given
    // 5:1 max_seq_len / mean_doc_length ratio.
    use nsl_codegen::pca_detect::PcaStrategy;
    assert_ne!(
        pca.detection.strategy,
        PcaStrategy::NoPacking,
        "PCA detection should elect a real strategy for ratio=5:1; got NoPacking"
    );
}

// ─── Negative compositions ─────────────────────────────────────────────────

/// Negative: packing=true but grad_accumulation=1 -> only PCA active.
/// Confirms PCA fires INDEPENDENTLY of FASE's accumulation trigger.
/// Pins the planner's exact contract: `accumulation==1` short-circuits to
/// `FaseMode::Passthrough` BEFORE the optimizer is inspected (fase.rs:189-200).
#[test]
fn pca_active_without_fase_when_grad_accumulation_one() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "    packing = true\n",
            "    max_sequence_length = 2048\n",
            "    mean_doc_length = 400\n",
            "\n",
            "train(model = m, grad_accumulation = 1):\n",
            "    data:\n",
            "        source = PretrainCorpus\n",
            "    optimizer: AdamW\n",
        )
    );
    let (ast, interner, diags) = parse_and_check(&src);
    assert_no_semantic_errors(&diags);
    let report = build_report(&ast, &interner, Path::new("pca_only.nsl"));
    let block = &report.train_blocks[0];

    // accumulation==1 -> Passthrough (planner's documented short-circuit).
    assert_eq!(
        block.fase.plan.mode,
        FaseMode::Passthrough,
        "accumulation=1 must map to Passthrough; got {:?}",
        block.fase.plan.mode
    );
    assert_eq!(block.fase.plan.accumulation, 1);
    // PCA still active.
    assert!(
        block.pca.is_some(),
        "PCA should be active independently of FASE"
    );
    assert!(block.pca.as_ref().unwrap().packing_config.enabled);
}

/// Negative: grad_accumulation=4 but no packing -> only FASE active.
/// Confirms FASE fires INDEPENDENTLY of PCA's packing trigger.
#[test]
fn fase_active_without_pca_when_dataset_has_no_packing() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "    max_sequence_length = 2048\n",
            "\n",
            "train(model = m, grad_accumulation = 4):\n",
            "    data:\n",
            "        source = PretrainCorpus\n",
            "    optimizer: AdamW\n",
        )
    );
    let (ast, interner, diags) = parse_and_check(&src);
    assert_no_semantic_errors(&diags);
    let report = build_report(&ast, &interner, Path::new("fase_only.nsl"));
    let block = &report.train_blocks[0];

    // FASE Deferred active.
    assert_eq!(block.fase.plan.mode, FaseMode::Deferred);
    assert_eq!(block.fase.plan.accumulation, 4);
    // PCA inactive (dataset has no packing key).
    assert!(
        block.pca.is_none(),
        "PCA should be None when dataset has no packing key; got {:?}",
        block.pca
    );
}

/// §2.5 + §4.2: gradient clipping co-activates with PCA without disabling
/// either Deferred mode or PCA detection.
#[test]
fn fase_grad_clip_composes_with_pca_packing() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "    packing = true\n",
            "    max_sequence_length = 2048\n",
            "    mean_doc_length = 400\n",
            "\n",
            "train(model = m, grad_accumulation = 4, grad_clip = 1.0):\n",
            "    data:\n",
            "        source = PretrainCorpus\n",
            "    optimizer: AdamW\n",
        )
    );
    let (ast, interner, diags) = parse_and_check(&src);
    assert_no_semantic_errors(&diags);
    let report = build_report(&ast, &interner, Path::new("composition_clip.nsl"));
    let block = &report.train_blocks[0];

    assert_eq!(block.fase.plan.mode, FaseMode::Deferred);
    assert_eq!(block.fase.plan.accumulation, 4);
    assert!(
        block.pca.is_some(),
        "PCA should remain active with grad_clip + packing"
    );
}

// ─── End-to-end codegen (production path) ──────────────────────────────────

/// Composition regression: real Cranelift codegen runs on the §5 canonical
/// fixture AND the resulting compile preserves both activations. Uses the
/// SAME fixture as the report-level test so a single canonical source
/// drives all paths.
///
/// Pins both:
///   - The compile completes without error (H1 production-path closure —
///     the `data: source = <ident>` config pair is skipped at codegen),
///   - The same parsed AST surfaces FASE Deferred + PCA Some via the
///     report builder (catches a regression that compiles cleanly but
///     silently strips dataset metadata before lowering).
#[test]
fn composed_source_compiles_and_preserves_both_activations() {
    let src = read_fixture(COMPOSED_FIXTURE);

    // 1) End-to-end Cranelift compile (mirrors fase_deferred_smoke).
    let opts = CompileOptions {
        source_ad: true,
        ..Default::default()
    };
    nsl_codegen::debug_compile_and_return_plan(&src, &opts).expect(
        "FASE Deferred + PCA-packing composition must compile without error",
    );

    // 2) Same source -> semantic-check + report-builder pin so a
    //    regression that compiles cleanly but strips dataset metadata
    //    before lowering still fails this test.
    let (ast, interner, diags) = parse_and_check(&src);
    assert_no_semantic_errors(&diags);
    let report = build_report(&ast, &interner, &fixture_path(COMPOSED_FIXTURE));
    let block = report
        .train_blocks
        .first()
        .expect("expected one train block in canonical fixture");
    assert_eq!(
        block.fase.plan.mode,
        FaseMode::Deferred,
        "composition must keep FASE Deferred; got {:?}",
        block.fase.plan.mode
    );
    assert!(
        block.pca.is_some(),
        "composition must keep PCA section populated"
    );
}

// ─── Semantic-checker negative regressions (M1) ────────────────────────────

/// M1 regression: unknown `data:` section keys must be refused. Before v8
/// the bypass accepted ANY ident-target Assign; v8 narrows it to an
/// allowlist (currently `{source}`) and surfaces a typo'd key.
#[test]
fn data_section_unknown_key_is_refused() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "\n",
            "train(model = m, grad_accumulation = 4):\n",
            "    data:\n",
            "        surce = PretrainCorpus\n", // typo for `source`
            "    optimizer: AdamW\n",
        )
    );
    let (_ast, _interner, diags) = parse_and_check(&src);
    let msgs: Vec<String> = diags.iter().map(|d| d.message.to_string()).collect();
    assert!(
        msgs.iter()
            .any(|m| m.contains("unknown data-section key") && m.contains("surce")),
        "expected 'unknown data-section key surce'; got: {msgs:?}",
    );
}

/// H2 regression: paper-canonical `separator_token_id` must pass semantic
/// check AND populate the PCA packing config (the v8 first-pass omitted
/// it — adversarial review caught it).
#[test]
fn separator_token_id_accepted_and_threaded_through() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "    packing = true\n",
            "    max_sequence_length = 2048\n",
            "    mean_doc_length = 400\n",
            "    separator_token_id = 2\n",
            "\n",
            "train(model = m, grad_accumulation = 4):\n",
            "    data:\n",
            "        source = PretrainCorpus\n",
            "    optimizer: AdamW\n",
        )
    );
    let (ast, interner, diags) = parse_and_check(&src);
    assert_no_semantic_errors(&diags);
    let report = build_report(&ast, &interner, Path::new("sep.nsl"));
    let pca = report.train_blocks[0]
        .pca
        .as_ref()
        .expect("expected PCA section");
    assert_eq!(
        pca.packing_config.separator_token_id,
        Some(2),
        "separator_token_id must flow into PCA packing config"
    );
}

// ─── Subprocess regression: `nsl check --training-report` ──────────────────

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn run_check(format: Option<&str>) -> (i32, String, String) {
    let root = workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let stdlib_path = root.join("stdlib");

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(&cargo_toml)
        .args(["-p", "nsl-cli", "--", "check"]);
    match format {
        Some(f) => {
            cmd.arg(format!("--training-report={}", f));
        }
        None => {
            cmd.arg("--training-report");
        }
    }
    cmd.arg(fixture_path(COMPOSED_FIXTURE));
    cmd.env("NSL_STDLIB_PATH", &stdlib_path);

    let out = cmd.output().expect("spawn nsl check");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// `nsl check --training-report` text output pins BOTH FASE Deferred and
/// PCA packing sections on the same train block.
#[test]
fn training_report_text_pins_both_fase_and_pca() {
    let (code, stdout, stderr) = run_check(None);
    assert_eq!(code, 0, "exit non-zero; stderr:\n{}", stderr);
    assert!(
        stdout.contains("Deferred"),
        "missing FASE 'Deferred' mode; stdout:\n{}",
        stdout
    );
    assert!(
        stdout.contains("grad_accumulation: 4"),
        "missing grad_accumulation=4 line; stdout:\n{}",
        stdout
    );
    assert!(
        stdout.contains("packing:              enabled"),
        "missing PCA 'packing: enabled' line; stdout:\n{}",
        stdout
    );
}

/// `nsl check --training-report=json` exposes BOTH `fase` and `pca` JSON
/// pointer paths so downstream tooling can pin composition behavior.
#[test]
fn training_report_json_pins_both_fase_and_pca() {
    let (code, stdout, stderr) = run_check(Some("json"));
    assert_eq!(code, 0, "exit non-zero; stderr:\n{}", stderr);
    let json_start = stdout
        .find('{')
        .unwrap_or_else(|| panic!("no JSON object in stdout:\n{}", stdout));
    let json: serde_json::Value = serde_json::from_str(&stdout[json_start..])
        .unwrap_or_else(|e| panic!("invalid JSON: {}\nstdout:\n{}", e, stdout));

    let mode = json
        .pointer("/train_blocks/0/fase/plan/mode")
        .expect("/train_blocks/0/fase/plan/mode missing");
    assert_eq!(mode, "Deferred", "fase mode mismatch; json:\n{}", stdout);

    let accum = json
        .pointer("/train_blocks/0/fase/plan/accumulation")
        .expect("/train_blocks/0/fase/plan/accumulation missing");
    assert_eq!(accum, 4, "accumulation mismatch; json:\n{}", stdout);

    let pca_enabled = json
        .pointer("/train_blocks/0/pca/packing_config/enabled")
        .expect("/train_blocks/0/pca/packing_config/enabled missing");
    assert_eq!(
        pca_enabled, true,
        "pca packing enabled mismatch; json:\n{}",
        stdout
    );
}

// ─── M4/L4 semantic-checker regressions (v9 follow-on) ────────────────────

/// M4 regression: calibration tools emit `mean_doc_length` / `doc_length_stddev`
/// as floats (e.g. 384.7). Before the v9 fix the checker hardcoded `Type::Int`
/// for those fields and would reject float literals with "must be Int, got Float".
#[test]
fn mean_doc_length_as_float_literal_accepted() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "    packing = true\n",
            "    max_sequence_length = 2048\n",
            "    mean_doc_length = 384.7\n",
            "    doc_length_stddev = 128.3\n",
            "\n",
            "train(model = m, grad_accumulation = 4):\n",
            "    data:\n",
            "        source = PretrainCorpus\n",
            "    optimizer: AdamW\n",
        )
    );
    let (_ast, _interner, diags) = parse_and_check(&src);
    let type_errors: Vec<_> = diags
        .iter()
        .filter(|d| {
            matches!(d.level, nsl_errors::Level::Error)
                && (d.message.contains("mean_doc_length") || d.message.contains("doc_length_stddev"))
        })
        .collect();
    assert!(
        type_errors.is_empty(),
        "mean_doc_length and doc_length_stddev must accept Float literals; got errors: {:?}",
        type_errors
    );
}

/// L4 regression: negative literal for mean_doc_length / doc_length_stddev must
/// be rejected at semantic-check time — a negative average document length is
/// physically impossible and indicates a miscalibrated stats file.
#[test]
fn mean_doc_length_negative_literal_rejected() {
    let src = format!(
        "{MODEL_PREAMBLE}{}",
        concat!(
            "dataset PretrainCorpus(\"/data/pile\"):\n",
            "    source = \"/data/pile.bin\"\n",
            "    packing = true\n",
            "    max_sequence_length = 2048\n",
            "    mean_doc_length = -1\n",
            "\n",
            "train(model = m, grad_accumulation = 4):\n",
            "    data:\n",
            "        source = PretrainCorpus\n",
            "    optimizer: AdamW\n",
        )
    );
    let (_ast, _interner, diags) = parse_and_check(&src);
    let msgs: Vec<String> = diags.iter().map(|d| d.message.to_string()).collect();
    assert!(
        msgs.iter().any(|m| m.contains("mean_doc_length") && m.contains("non-negative")),
        "expected 'mean_doc_length must be non-negative'; got: {msgs:?}",
    );
}
