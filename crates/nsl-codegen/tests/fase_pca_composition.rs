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

fn parse(src: &str) -> (nsl_ast::Module, Interner) {
    let mut interner = Interner::new();
    let (tokens, _diags) = tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    (parsed.module, interner)
}

const COMPOSED_SRC: &str = concat!(
    "dataset PretrainCorpus(\"/data/pile\"):\n",
    "    source = \"/data/pile.bin\"\n",
    "    packing = true\n",
    "    max_sequence_length = 2048\n",
    "    mean_doc_length = 400\n",
    "\n",
    "train(model = m, grad_accumulation = 4):\n",
    "    data:\n",
    "        source = PretrainCorpus\n",
    "    optimizer: AdamW\n",
);

/// §5 paper-canonical composition: dataset packing + grad_accumulation
/// co-activate on the same train block.
#[test]
fn fase_deferred_and_pca_co_activate_on_same_train_block() {
    let (ast, interner) = parse(COMPOSED_SRC);
    let report = build_report(&ast, &interner, Path::new("composition.nsl"));

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

/// Negative: packing=true but grad_accumulation=1 -> only PCA active.
/// Confirms PCA fires INDEPENDENTLY of FASE's accumulation trigger.
#[test]
fn pca_active_without_fase_when_grad_accumulation_one() {
    let src = concat!(
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
    );
    let (ast, interner) = parse(src);
    let report = build_report(&ast, &interner, Path::new("pca_only.nsl"));
    let block = &report.train_blocks[0];

    // FASE Deferred requires accumulation > 1.
    assert_ne!(
        block.fase.plan.mode,
        FaseMode::Deferred,
        "Deferred mode requires accumulation > 1; got {:?}",
        block.fase.plan.mode
    );
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
    let src = concat!(
        "dataset PretrainCorpus(\"/data/pile\"):\n",
        "    max_sequence_length = 2048\n",
        "\n",
        "train(model = m, grad_accumulation = 4):\n",
        "    data:\n",
        "        source = PretrainCorpus\n",
        "    optimizer: AdamW\n",
    );
    let (ast, interner) = parse(src);
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
    let src = concat!(
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
    );
    let (ast, interner) = parse(src);
    let report = build_report(&ast, &interner, Path::new("composition_clip.nsl"));
    let block = &report.train_blocks[0];

    assert_eq!(block.fase.plan.mode, FaseMode::Deferred);
    assert_eq!(block.fase.plan.accumulation, 4);
    assert!(
        block.pca.is_some(),
        "PCA should remain active with grad_clip + packing"
    );
}

/// Composition regression: real Cranelift codegen run on the composed
/// source must not crash. Mirrors the smoke-style assertion of
/// `fase_deferred_smoke::fase_deferred_compiles_adamw_grad_accum_4`,
/// but with the PCA trigger also active.
#[test]
fn composed_source_compiles_without_error() {
    // Standalone code path needs `from nsl.nn.losses import mse_loss` plus
    // a model + step body so the parser produces a complete program for the
    // compiler. (The report-level tests above use the bare-AST shortcut.)
    let src = concat!(
        "from nsl.nn.losses import mse_loss\n",
        "\n",
        "dataset PretrainCorpus(\"/data/pile\"):\n",
        "    source = \"/data/pile.bin\"\n",
        "    packing = true\n",
        "    max_sequence_length = 2048\n",
        "    mean_doc_length = 400\n",
        "\n",
        "model Tiny:\n",
        "    w: Tensor = ones([4, 4])\n",
        "\n",
        "    fn forward(self, x: Tensor) -> Tensor:\n",
        "        return x @ self.w\n",
        "\n",
        "let m = Tiny()\n",
        "let x = ones([2, 4])\n",
        "let y = zeros([2, 4])\n",
        "\n",
        "train(model = m, epochs = 1, grad_accumulation = 4):\n",
        "    data:\n",
        "        source = PretrainCorpus\n",
        "    optimizer: AdamW(lr = 0.001, weight_decay = 0.01)\n",
        "    step(batch):\n",
        "        let pred = m.forward(x)\n",
        "        let loss = mse_loss(pred, y)\n",
    );
    let opts = CompileOptions {
        source_ad: true,
        ..Default::default()
    };
    nsl_codegen::debug_compile_and_return_plan(src, &opts).expect(
        "FASE Deferred + PCA-packing composition must compile without error",
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

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
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
    cmd.arg(fixture("training_report_fase_pca_composition.nsl"));
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
