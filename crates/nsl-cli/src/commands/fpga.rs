//! `nsl fpga-compile` — NSL -> KIR -> HIR -> Verilog pipeline (experimental).
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;

use nsl_lexer::Interner;

// ──────────────────────────────────────────────────────────────────────────────
// M57.1 §3.6 (Task 4.1): `nsl fpga-compile` end-to-end pipeline dispatch.
//
// Pipeline stages:
//   1. Read NSL source
//   2. Lex + parse + semantic resolve (Phase 1's i8 alias lives in nsl-lexer)
//   3. AST → KIR via nsl_codegen::kernel_lower_fpga::lower
//   4. KIR → HIR via nsl_codegen::hir::KirToHirPass
//   5. Bake fixture weight/bias values into LocalParams by name
//   6. HIR → Verilog via nsl_codegen::backend_verilog::VerilogEmitter
//   7. Write <output-dir>/tiny_mlp.v
//
// The fixture sidecar (`<stem>_weights.bin`) is auto-discovered if `--fixture`
// is omitted; the optional `<stem>.toml` manifest's `sha256` is verified when
// present. Spec §6.1.
// ──────────────────────────────────────────────────────────────────────────────

/// The fixed top-level module name emitted by the v1 KIR→HIR pass.
/// Mirrors `KirToHirPass::lower(&kir, "tiny_mlp")` below.
const FPGA_V1_MODULE_NAME: &str = "tiny_mlp";

/// Sequential variant module name — distinct from the combinational `tiny_mlp`
/// so a single Verilator testbench can instantiate BOTH for the v1↔v2
/// cross-check.  Matches `Vtiny_mlp_seq.h` (clocked testbench) and the
/// `sequential_v1_mlp_skeleton` snapshot.
const FPGA_V1_SEQ_MODULE_NAME: &str = "tiny_mlp_seq";

/// Derive `<dir>/<stem>_weights.bin` from the .nsl path (sidecar convention,
/// spec §6.1). Returns `None` when the source path has no stem.
fn derive_default_fixture_path(source: &std::path::Path) -> Option<PathBuf> {
    let stem = source.file_stem()?.to_owned();
    let mut name = stem;
    name.push("_weights.bin");
    Some(source.with_file_name(name))
}

/// Derive `<dir>/<stem>.toml` from a fixture .bin path. Returns `None` when
/// the fixture path has no stem.
fn derive_toml_path(fixture: &std::path::Path) -> Option<PathBuf> {
    Some(fixture.with_extension("toml"))
}

/// Walks the fixture's blocks and writes element values into the matching
/// `W<i>` (weight) and `b<i>` (bias) `LocalParamArray`s declared by the
/// KIR→HIR pass (spec §3.2 / Task W6).
///
/// Pre-W5 (commit `94a05f8e`) the KIR→HIR pass emitted per-element scalar
/// `LocalParam`s named `W<i>_<k>_<o>` and `b<i>_<o>`; W5 collapsed those into
/// `LocalParamArray { name: "W<i>", dims: [k_dim, n_outputs], values: <flat> }`
/// and `LocalParamArray { name: "b<i>", dims: [n_outputs], values: <flat> }`
/// to feed the genvar-indexed ripple body (`SignalRef::IndexedLocalParam`).
/// The bake helper now writes directly into `LocalParamArray.values`, which is
/// row-major flat: for a 2-D W of shape `[K, N]`, `values[k * N + o]` is the
/// `[k][o]` cell (matching the fixture's on-disk layout — see
/// `nsl_test::fixture::FixtureBlock::shape_used`).
///
/// Missing `LocalParamArray`s are not silently tolerated: a mismatch between
/// the fixture's declared block shape and the HIR's declared arrays indicates
/// a divergence between the .nsl source (which the KIR→HIR pass walks) and the
/// fixture file the CLI was pointed at. We surface this as an error string so
/// the caller's `eprintln!("error: {e}")` reaches the user.
fn bake_fixture_into_localparams(
    module: &mut nsl_codegen::hir::HirModule,
    fixture: &nsl_test::fixture::FixtureFile,
) -> Result<(), String> {
    use nsl_test::fixture::{BlockDtype, BlockKind};

    // Decode block values to a uniform i128 representation up front. (The
    // `LocalParamArray.values` field is `Vec<i128>` so any v1 dtype fits.)
    let decode_values = |block: &nsl_test::fixture::FixtureBlock| -> Result<Vec<i128>, String> {
        Ok(match block.dtype {
            BlockDtype::I8 => block.as_i8().into_iter().map(|v| v as i128).collect(),
            BlockDtype::I16 => {
                return Err(format!(
                    "fixture block (layer {}) dtype i16 is not supported in v1",
                    block.layer
                ));
            }
            BlockDtype::I32 => block.as_i32().into_iter().map(|v| v as i128).collect(),
            BlockDtype::I64 => block.as_i64().into_iter().map(|v| v as i128).collect(),
        })
    };

    // Helper: locate a `LocalParamArray` by name and return its index in
    // `module.local_param_arrays_mut()`. Errors fail-loud with a structural
    // message — the KIR→HIR pass either declared it (under exactly the
    // `W<i>` / `b<i>` convention) or the .nsl source is structurally
    // incompatible with this fixture.
    let find_lpa_idx = |module: &nsl_codegen::hir::HirModule,
                        name: &str|
     -> Result<usize, String> {
        module
            .local_param_arrays()
            .iter()
            .position(|lpa| lpa.name == name)
            .ok_or_else(|| {
                format!(
                    "fixture references LocalParamArray `{name}` not declared by the \
                     KIR→HIR pass (source/.nsl and fixture/.bin out of sync — \
                     regenerate the fixture or update the .nsl source)"
                )
            })
    };

    for block in &fixture.blocks {
        match block.kind {
            BlockKind::Weight => {
                let used = block.shape_used();
                if used.len() != 2 {
                    return Err(format!(
                        "fixture weight block (layer {}) has rank {}; expected 2",
                        block.layer,
                        used.len()
                    ));
                }
                let k_dim = used[0] as usize;
                let n_outputs = used[1] as usize;
                let values = decode_values(block)?;
                if values.len() != k_dim * n_outputs {
                    return Err(format!(
                        "fixture weight block (layer {}): element count {} != K·N = {}·{} = {}",
                        block.layer,
                        values.len(),
                        k_dim,
                        n_outputs,
                        k_dim * n_outputs
                    ));
                }
                let arr_name = format!("W{}", block.layer);
                let idx = find_lpa_idx(module, &arr_name)?;
                {
                    // Shape-check against the HIR's declared dims. The
                    // KIR→HIR pass derived these from the .nsl source's
                    // `const W<i>: Tensor<[K, N], i8>` declaration; the
                    // fixture must match exactly (row-major [K, N]).
                    let lpa = &module.local_param_arrays()[idx];
                    if lpa.dims != vec![k_dim, n_outputs] {
                        return Err(format!(
                            "LocalParamArray `{}` dims mismatch: HIR declares {:?} \
                             but fixture provides [{}, {}]",
                            arr_name, lpa.dims, k_dim, n_outputs
                        ));
                    }
                }
                // Row-major [K, N]: index = k * n_outputs + o — same layout
                // as `LocalParamArray.values` (`values[r * dims[1] + c]`
                // per nodes.rs:77-79).
                module.local_param_arrays_mut()[idx].values = values;
            }
            BlockKind::Bias => {
                let used = block.shape_used();
                if used.len() != 1 {
                    return Err(format!(
                        "fixture bias block (layer {}) has rank {}; expected 1",
                        block.layer,
                        used.len()
                    ));
                }
                let n_outputs = used[0] as usize;
                let values = decode_values(block)?;
                if values.len() != n_outputs {
                    return Err(format!(
                        "fixture bias block (layer {}): element count {} != N = {}",
                        block.layer,
                        values.len(),
                        n_outputs
                    ));
                }
                let arr_name = format!("b{}", block.layer);
                let idx = find_lpa_idx(module, &arr_name)?;
                {
                    let lpa = &module.local_param_arrays()[idx];
                    if lpa.dims != vec![n_outputs] {
                        return Err(format!(
                            "LocalParamArray `{}` dims mismatch: HIR declares {:?} \
                             but fixture provides [{}]",
                            arr_name, lpa.dims, n_outputs
                        ));
                    }
                }
                module.local_param_arrays_mut()[idx].values = values;
            }
        }
    }

    Ok(())
}

/// End-to-end pipeline. Returns a flat String error for the simple
/// "eprintln!('error: {e}'); exit(1)" pattern used by the rest of the CLI.
pub(crate) fn run_fpga_compile(
    source_path: &std::path::Path,
    fixture_arg: Option<&PathBuf>,
    output_dir_arg: Option<&PathBuf>,
    test_taps: bool,
    seq: bool,
) -> Result<(), String> {
    // ── 1. Read NSL source ──────────────────────────────────────────────────
    let source_text = std::fs::read_to_string(source_path)
        .map_err(|e| format!("read NSL source `{}`: {e}", source_path.display()))?;

    // ── 2. Lex + parse ──────────────────────────────────────────────────────
    //
    // NOTE: semantic analysis (`nsl_semantic::analyze_*`) is INTENTIONALLY
    // skipped on the FPGA path. The v1 KIR→HIR recognizer in
    // `kernel_lower_fpga::lower` walks the AST directly and matches on
    // structural shape (`model TinyMlp:` + `relu(matmul(...) + bias)`); it
    // does not require `matmul`/`relu` to be resolvable as standard-library
    // symbols. Running the full semantic analyzer here would emit
    // "undefined variable `matmul`" errors that are spurious in the FPGA
    // dispatch context. The recognizer's own surface (any structural
    // mismatch) maps to `UnsupportedV1Shape` with a clear "expected …, found …"
    // explanation, which is the appropriate error surface for v1's
    // recognizer-only frontend.
    use nsl_errors::{Level, SourceMap};
    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file(source_path.display().to_string(), source_text.clone());
    let mut interner = Interner::new();

    let (tokens, lex_errors) = nsl_lexer::tokenize(&source_text, file_id, &mut interner);
    for diag in &lex_errors {
        source_map.emit_diagnostic(diag);
    }

    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    for diag in &parse_result.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    let total_errors = lex_errors
        .iter()
        .chain(parse_result.diagnostics.iter())
        .filter(|d| d.level == Level::Error)
        .count();
    if total_errors > 0 {
        return Err(format!(
            "{total_errors} lex/parse error(s) in NSL source `{}` — see diagnostics above",
            source_path.display()
        ));
    }

    // ── 3. AST → KIR (Fpga target dispatch) ─────────────────────────────────
    let kir = nsl_codegen::kernel_lower_fpga::lower(&parse_result.module, &interner)
        .map_err(|e| format!("AST→KIR (FPGA target): {e}"))?;

    // ── 4. KIR → HIR ────────────────────────────────────────────────────────
    // M57.2: `seq` selects the sequential clocked FSM lowering; combinational
    // is the default (`sequential: false`).  Use the struct literal directly so
    // we don't need to touch `KirToHirPass::new`'s other callers.
    // The sequential module uses a distinct name (`tiny_mlp_seq`) so that a
    // single Verilator testbench can instantiate both variants simultaneously
    // for the v1↔v2 cross-check (Vtiny_mlp_seq.h / Vtiny_mlp.h).
    let module_name = if seq { FPGA_V1_SEQ_MODULE_NAME } else { FPGA_V1_MODULE_NAME };
    let mut hir = nsl_codegen::hir::KirToHirPass { test_taps, sequential: seq }
        .lower(&kir, module_name)
        .map_err(|e| format!("KIR→HIR: {e}"))?;

    // Stash cycle_count before hir is borrowed by bake_fixture_into_localparams
    // and emit_module (both take &/&mut HirModule); reading it here lets us print
    // after emit without keeping an extra borrow alive.
    let cycle_count = hir.cycle_count;

    // ── 5. Bake fixture into LocalParams ────────────────────────────────────
    let fixture_path: PathBuf = match fixture_arg {
        Some(p) => p.clone(),
        None => derive_default_fixture_path(source_path).ok_or_else(|| {
            format!(
                "no --fixture provided and could not derive default sidecar from `{}`",
                source_path.display()
            )
        })?,
    };
    let fixture_bytes = std::fs::read(&fixture_path)
        .map_err(|e| format!("read fixture `{}`: {e}", fixture_path.display()))?;

    // Optional manifest hash check (spec §6.1). Missing .toml is not an
    // error — the bin is still parsed; we just skip the integrity check.
    if let Some(toml_path) = derive_toml_path(&fixture_path) {
        if toml_path.exists() {
            let toml_text = std::fs::read_to_string(&toml_path).map_err(|e| {
                format!("read manifest `{}`: {e}", toml_path.display())
            })?;
            let manifest: toml::Value = toml::from_str(&toml_text).map_err(|e| {
                format!("parse manifest `{}`: {e}", toml_path.display())
            })?;
            let expected_hash = manifest
                .get("sha256")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    format!(
                        "manifest `{}` missing `sha256` top-level field",
                        toml_path.display()
                    )
                })?;
            nsl_test::fixture::verify_hash(&fixture_bytes, expected_hash).map_err(|e| {
                format!(
                    "fixture `{}` hash mismatch against manifest `{}`: {e}",
                    fixture_path.display(),
                    toml_path.display()
                )
            })?;
        }
    }

    let fixture_file = nsl_test::fixture::parse(&fixture_bytes)
        .map_err(|e| format!("parse fixture `{}`: {e}", fixture_path.display()))?;
    bake_fixture_into_localparams(&mut hir, &fixture_file)?;

    // ── 6. HIR → Verilog ────────────────────────────────────────────────────
    let verilog = nsl_codegen::backend_verilog::VerilogEmitter::emit_module(&hir);

    // ── 7. Write <output-dir>/<module_name>.v ───────────────────────────────
    // Combinational → tiny_mlp.v; sequential → tiny_mlp_seq.v.
    let out_dir = output_dir_arg
        .cloned()
        .unwrap_or_else(|| PathBuf::from("target/fpga"));
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create output dir `{}`: {e}", out_dir.display()))?;
    let out_path = out_dir.join(format!("{module_name}.v"));
    std::fs::write(&out_path, &verilog)
        .map_err(|e| format!("write `{}`: {e}", out_path.display()))?;
    println!("Wrote {}", out_path.display());

    // M57.2: print the deterministic cycle count when --seq was requested.
    if seq {
        if let Some(cycles) = cycle_count {
            println!("total_cycles={cycles}");
        }
    }

    Ok(())
}
