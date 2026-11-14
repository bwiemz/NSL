//! CPKD Innovation 5: end-to-end `nsl check --cpkd-design-student` CLI tests.
//!
//! Modeled on `cep_cli.rs`: spawns the real `nsl` binary with
//! `NSL_STDLIB_PATH` set so the frontend can resolve stdlib imports.
//!
//!   1. The repo searchable fixture (`cep_searchable.nsl`) prints the Student
//!      Design Report. Its `@search` axes span nine candidates (three widths
//!      by three head counts); a budget that cuts the widest ones is
//!      exercised here, and a sub-teacher selection on a one-axis fixture.
//!   2. A budget below every candidate refuses loudly, naming the smallest
//!      candidate's parameter count.
//!   3. A model WITHOUT `@search` decorators (`cep_canonical_small.nsl`)
//!      refuses loudly, telling the user to add `@search` axes.
//!   4. Bad budget strings and unknown GPUs refuse with clear errors.

use assert_cmd::prelude::*;
use predicates::prelude::*;
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

fn searchable_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_searchable.nsl")
}

fn no_axes_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_canonical_small.nsl")
}

/// Variant of `cep_searchable.nsl` with one axis: baseline d_model=384 with
/// n_heads=4 / n_kv_heads=2, and a d_model axis [128, 256, 384]. All three
/// candidates are valid (4 % 2 == 0, d mult-64). With the output head tied to
/// the embedding a candidate has d·(4096 + 6·(2·4·64 + 2·2·64 + 3·1024 + 2) + 1)
/// = 27,149·d parameters: {3,475,072; 6,950,144; 10,425,216} — a 9M budget
/// forces a strictly smaller student than the teacher.
fn write_sub_teacher_fixture(dir: &Path) -> PathBuf {
    let src = r#"from nsl.nn.norms import RMSNorm

@cep_search(target = h100, objective = param_efficiency)
@search(d_model, [128, 256, 384])
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model SearchNet:
    embed: Tensor = randn([4096, 384]) * full([1], 0.02)
    blocks: [TransformerBlock; 6] = TransformerBlock(384, 4, 2, 64, 1024, 0.1)
"#;
    let path = dir.join("cpkd_sub_teacher.nsl");
    fs::write(&path, src).unwrap();
    path
}

fn nsl_check(file: &Path) -> Command {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(file);
    cmd
}

#[test]
fn design_report_prints_on_searchable_fixture() {
    // cep_searchable.nsl spans d_model {256, 384, 512} x n_heads {4, 6, 8}
    // with 2 key/value heads: d·(24,077 + 768·n_heads) parameters. The three
    // d_model=512 candidates (13.9M, 14.7M, 15.5M) exceed a 13M budget; the
    // other six (6.95M .. 11.6M) fit.
    let mut cmd = nsl_check(&searchable_fixture());
    cmd.arg("--cpkd-design-student")
        .arg("13M")
        .arg("--cpkd-target")
        .arg("H100-SXM");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CPKD Student Design Report ==="))
        .stdout(predicate::str::contains("Target: H100-SXM"))
        .stdout(predicate::str::contains("Candidates: 9 enumerated, 6 within budget"))
        .stdout(predicate::str::contains("Teacher -> student layer mapping"))
        .stdout(predicate::str::contains("student  0 <- teacher  0"));
}

#[test]
fn budget_selects_sub_teacher_candidate() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_sub_teacher_fixture(tmp.path());
    // Teacher = 10,425,216 params; 9M budget leaves the d_model=256 (6.95M)
    // and d_model=128 (3.48M) candidates.
    let mut cmd = nsl_check(&fixture);
    cmd.arg("--cpkd-design-student").arg("9M");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CPKD Student Design Report ==="))
        .stdout(predicate::str::contains("Candidates: 3 enumerated, 2 within budget"))
        .stdout(predicate::str::contains("% reduction"))
        .stdout(predicate::str::contains("(0.0% reduction)").not())
        .stdout(predicate::str::contains("student  0 <- teacher  0"));
}

#[test]
fn budget_below_all_candidates_refuses_loudly() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_sub_teacher_fixture(tmp.path());
    // Smallest candidate (d_model=128) is 3,475,072 params — 1M cannot fit.
    let mut cmd = nsl_check(&fixture);
    cmd.arg("--cpkd-design-student").arg("1M");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no student candidate fits"))
        .stderr(predicate::str::contains("3475072"))
        .stderr(predicate::str::contains("widen the @search axes"));
}

#[test]
fn model_without_search_axes_refuses_loudly() {
    let mut cmd = nsl_check(&no_axes_fixture());
    cmd.arg("--cpkd-design-student").arg("1M");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("@search"))
        .stderr(predicate::str::contains("requires @search(axis, [values]) decorators"));
}

#[test]
fn bad_budget_string_refuses() {
    let mut cmd = nsl_check(&searchable_fixture());
    cmd.arg("--cpkd-design-student").arg("12X");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("invalid parameter budget"));
}

#[test]
fn unknown_gpu_refuses_with_supported_list() {
    let mut cmd = nsl_check(&searchable_fixture());
    cmd.arg("--cpkd-design-student")
        .arg("13M")
        .arg("--cpkd-target")
        .arg("NoSuchGPU-9000");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unknown student-design target GPU"))
        .stderr(predicate::str::contains("H100-SXM"));
}

#[test]
fn cpkd_target_without_design_flag_refuses() {
    let mut cmd = nsl_check(&searchable_fixture());
    cmd.arg("--cpkd-target").arg("H100-SXM");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--cpkd-target requires --cpkd-design-student"));
}
