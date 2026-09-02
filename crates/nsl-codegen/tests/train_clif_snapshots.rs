//! CLIF snapshots of train-block lowering: every function the compiler
//! defines for each program under `tests/train_clif/`, as `--dump-ir` would
//! print it, pinned with `insta`.
//!
//! Roadmap item A1 splits `compile_train_block_inner` into passes. A
//! refactor of it is behaviour-preserving exactly when these snapshots do
//! not move, so they are the gate every A1 PR runs; a snapshot that moves
//! under a change that meant to change nothing is the finding.
//!
//! Each program is compiled twice and the two dumps compared before the
//! snapshot is checked: a difference between them is codegen
//! nondeterminism (a `HashMap` walked in emission order), which is a bug
//! in its own right and is reported as such rather than as a flaky
//! snapshot.
//!
//! To review and accept a moved snapshot:
//!   cargo insta review -p nsl-codegen
//!
//! The calling-convention token in every signature is the host's
//! (`system_v`, `windows_fastcall`, `apple_aarch64`) and is normalised to
//! `<callconv>` so one snapshot serves every CI lane.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use nsl_codegen::ir_capture::IrDump;
use nsl_codegen::CompileOptions;
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root above crates/nsl-codegen")
        .to_path_buf()
}

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/train_clif")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Pin the process environment the compiler reads, once, before any test
/// thread compiles.
///
/// The codegen crate reads some forty `NSL_*` knobs at compile time
/// (`NSL_PHASE_TIMING` injects clock calls into `main`, `NSL_FASE_FUSED_STEP`,
/// `NSL_FUSE_*`, `NSL_CCR_SEGMENT_FREE`, … change what is emitted). A shell
/// with one exported would fail every snapshot here — or, worse, accept a
/// variant into the snapshot files that a clean environment then rejects.
/// So every `NSL_*` variable is cleared, and `NSL_STDLIB_PATH` is set to
/// this tree's `stdlib/`: the optimizer modules the fixtures import are
/// part of what the snapshots pin.
fn pin_environment() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let nsl_vars: Vec<_> = std::env::vars_os()
            .map(|(key, _)| key)
            .filter(|key| key.to_string_lossy().starts_with("NSL_"))
            .collect();
        for key in nsl_vars {
            std::env::remove_var(key);
        }
        std::env::set_var("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    });
}

fn errors(diags: &[nsl_errors::Diagnostic]) -> Vec<&str> {
    diags
        .iter()
        .filter(|d| matches!(d.level, Level::Error))
        .map(|d| d.message.as_str())
        .collect()
}

/// Lex, parse, analyse and compile `src` in-process, returning every
/// function's CLIF in definition order. Any failure is a test failure:
/// a program that stops compiling is not a snapshot to skip.
fn compile_capturing_ir(name: &str, src: &str, options: &CompileOptions) -> Vec<IrDump> {
    pin_environment();
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(errors(&lex_diags).is_empty(), "{name}: lex errors: {:?}", errors(&lex_diags));
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        errors(&parsed.diagnostics).is_empty(),
        "{name}: parse errors: {:?}",
        errors(&parsed.diagnostics)
    );
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        errors(&analysis.diagnostics).is_empty(),
        "{name}: semantic errors: {:?}",
        errors(&analysis.diagnostics)
    );
    let imported_fns = nsl_codegen::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        options,
    )
    .unwrap_or_else(|e| panic!("{name}: stdlib loader: {}", e.message));
    let (dumps, result) = nsl_codegen::compile_entry_capturing_ir(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &imported_fns,
        HashMap::new(),
        HashSet::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        options,
    );
    if let Err(e) = result {
        panic!("{name}: compile failed: {}", e.message);
    }
    assert!(!dumps.is_empty(), "{name}: the compile defined no function");
    dumps
}

/// The dumps as one text, in `--dump-ir`'s framing, with every function
/// named by its label, every `u<namespace>:<index>` reference replaced by
/// the symbol it names and the host calling convention normalised.
fn render(dumps: &[IrDump]) -> String {
    let mut out = String::new();
    for d in dumps {
        out.push_str("--- IR: ");
        out.push_str(&d.label);
        out.push_str(" ---\n");
        out.push_str(&name_references(&label_header(d), &d.symbols));
        out.push('\n');
    }
    for cc in ["system_v", "windows_fastcall", "apple_aarch64"] {
        // CLIF prints the calling convention after a signature's return
        // list, in `sigN = ...` lines and the `function` header.
        out = out.replace(&format!(" {cc}"), " <callconv>");
    }
    out
}

/// `function u0:N(…)` with `u0:N` replaced by `<label>`. The compiler names
/// each function it defines by a per-compile counter, not its `FuncId`
/// (see `IrDump::symbols`), so the header's `N` is the function's position
/// in definition order: it moves whenever a function is added before it,
/// and names nothing. The label is what identifies the function.
fn label_header(d: &IrDump) -> String {
    let (name, rest) = d
        .clif
        .strip_prefix("function ")
        .and_then(|header| header.split_once('('))
        .unwrap_or_else(|| panic!("{}: CLIF does not open with `function <name>(`", d.label));
    assert!(
        is_user_reference(name),
        "{}: the function is named `{name}`, not `u<namespace>:<index>`",
        d.label
    );
    format!("function <{}>({rest}", d.label)
}

/// Replace each external-name token (`u<namespace>:<index>` in `fn`
/// declarations, `userextname<ref>` in `gv … = symbol …` lines) in `clif`
/// by its symbol. Token by token, so `u0:1` never matches inside `u0:12`.
/// A reference the table cannot name is a hole in the capture, not
/// something to pass through: the snapshot would then move with the index.
fn name_references(clif: &str, symbols: &[(String, String)]) -> String {
    let mut out = String::with_capacity(clif.len());
    for line in clif.lines() {
        let mut first = true;
        for word in line.split(' ') {
            if !first {
                out.push(' ');
            }
            first = false;
            let named = symbols
                .iter()
                .find(|(token, _)| token == word)
                .map(|(_, name)| name.as_str());
            match named {
                Some(name) => out.push_str(name),
                None => {
                    // A reference may carry punctuation (`u0:3(i64)`), which
                    // the table's bare tokens would not match.
                    let bare: String = word
                        .trim_start_matches(|c: char| !c.is_ascii_alphanumeric())
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == ':')
                        .collect();
                    assert!(
                        !is_user_reference(&bare),
                        "{word} has no symbol in the dump's table: {symbols:?}"
                    );
                    out.push_str(word);
                }
            }
        }
        out.push('\n');
    }
    out
}

fn is_user_reference(word: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = word.strip_prefix("userextname") {
        return digits(rest);
    }
    word.strip_prefix('u')
        .and_then(|rest| rest.split_once(':'))
        .is_some_and(|(ns, idx)| digits(ns) && digits(idx))
}

/// Compile `src` twice, require the two runs to agree function for
/// function, and pin the result under `snapshot`.
fn check(snapshot: &str, src: &str, options: &CompileOptions) {
    let first = compile_capturing_ir(snapshot, src, options);
    let second = compile_capturing_ir(snapshot, src, options);
    assert_eq!(
        first.len(),
        second.len(),
        "{snapshot}: two compiles of the same program defined a different number of functions"
    );
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(
            a.label, b.label,
            "{snapshot}: two compiles of the same program defined functions in a different order"
        );
        if a.clif != b.clif {
            let dir = std::env::temp_dir();
            let (pa, pb) = (
                dir.join(format!("{snapshot}.first.clif")),
                dir.join(format!("{snapshot}.second.clif")),
            );
            std::fs::write(&pa, render(&first)).expect("write first dump");
            std::fs::write(&pb, render(&second)).expect("write second dump");
            panic!(
                "{snapshot}: two compiles of the same program emitted different CLIF for {} — \
                 codegen nondeterminism, not a snapshot to accept. First difference:\n{}\n\
                 Both dumps: {} and {}",
                a.label,
                first_difference(&a.clif, &b.clif),
                pa.display(),
                pb.display()
            );
        }
    }
    let labels: Vec<&str> = first.iter().map(|d| d.label.as_str()).collect();
    assert!(
        labels.contains(&"main"),
        "{snapshot}: no `main` was defined; the train block lowers into it. Defined: {labels:?}"
    );
    insta::assert_snapshot!(snapshot, render(&first));
}

/// The first line on which `a` and `b` disagree, with context, for the
/// nondeterminism message.
fn first_difference(a: &str, b: &str) -> String {
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    let n = la.len().max(lb.len());
    let at = (0..n)
        .find(|&i| la.get(i) != lb.get(i))
        .expect("the texts differ but every line agrees");
    let mut out = String::new();
    for i in at.saturating_sub(3)..(at + 4).min(n) {
        let mark = if i == at { ">" } else { " " };
        out.push_str(&format!(
            "{mark} {:>5} | {:<70} | {}\n",
            i + 1,
            la.get(i).copied().unwrap_or("<end>"),
            lb.get(i).copied().unwrap_or("<end>")
        ));
    }
    out
}

/// The tape (runtime autograd) lowering, every flag at its default.
fn tape() -> CompileOptions {
    CompileOptions::default()
}

/// The source-AD lowering (`--source-ad`, the production path), every
/// other flag at its default.
fn source_ad() -> CompileOptions {
    CompileOptions {
        source_ad: true,
        ..CompileOptions::default()
    }
}

/// One snapshot per (program, options) pair. The program name is the
/// fixture file under `tests/train_clif/`; the snapshot is named
/// `<program>_<variant>`. Also defines `matrix()`, the whole table, for
/// the vacuity check below.
macro_rules! snapshots {
    ($($test:ident: $program:literal, $options:expr;)*) => {
        $(
            #[test]
            fn $test() {
                check(stringify!($test), &fixture(concat!($program, ".nsl")), &$options);
            }
        )*

        fn matrix() -> Vec<(&'static str, &'static str, CompileOptions)> {
            vec![$((stringify!($test), concat!($program, ".nsl"), $options),)*]
        }
    };
}

/// A flag variant whose CLIF equals its base's is not testing the flag —
/// the flag is inert on that program (or on that lowering: a
/// `debug_training` variant over the source-AD path was identical to its
/// base, because the FASE hook owns the gradients there and the checksum
/// call it gates is skipped). Every entry in the table must produce a
/// distinct text.
#[test]
fn no_two_entries_produce_the_same_clif() {
    let renders: Vec<(&str, String)> = matrix()
        .iter()
        .map(|(name, program, options)| {
            (*name, render(&compile_capturing_ir(name, &fixture(program), options)))
        })
        .collect();
    for (i, (name, text)) in renders.iter().enumerate() {
        for (other, other_text) in &renders[i + 1..] {
            assert!(
                text != other_text,
                "{name} and {other} compile to the same CLIF: one of them tests nothing"
            );
        }
    }
}

/// Struct constructors are declared and defined by walking
/// `TypeRegistry::struct_layouts`, so its iteration order is their
/// declaration order (`FuncId`, hence every `fn0 = u0:N` reference and the
/// object's symbol order) and their definition order (the `function u0:N`
/// header). While that map was a `HashMap`, six builds of
/// `examples/m5_generic_instantiation_check.nsl` (four structs) gave five
/// distinct object files (bugs.md 2026-09-02); it is a `BTreeMap` now, and
/// the order is the name order.
///
/// Six structs declared out of name order: a walk in source order or in
/// hash order (719 permutations of 720) fails on the first compile, so the
/// two-compile agreement check above does not have to get lucky here.
#[test]
fn struct_ctors_are_defined_in_name_order() {
    const SRC: &str = "\
struct Zeta:
    v: int

struct Alpha:
    v: int

struct Mu:
    v: int

struct Beta:
    v: int

struct Omega:
    v: int

struct Gamma:
    v: int

fn main():
    let z = Zeta(v = 1)
    let a = Alpha(v = z.v)
    let m = Mu(v = a.v)
    let b = Beta(v = m.v)
    let o = Omega(v = b.v)
    let g = Gamma(v = o.v)
    print(g.v)
";
    let expected: Vec<String> = ["Alpha", "Beta", "Gamma", "Mu", "Omega", "Zeta"]
        .iter()
        .map(|name| format!("struct ctor '{name}'"))
        .collect();
    let mut renders = Vec::new();
    for run in 1..=3 {
        let dumps = compile_capturing_ir("struct_ctor_order", SRC, &tape());
        let ctors: Vec<&str> = dumps
            .iter()
            .map(|d| d.label.as_str())
            .filter(|label| label.starts_with("struct ctor "))
            .collect();
        assert_eq!(
            ctors, expected,
            "compile {run}: struct constructors were not defined in name order"
        );
        renders.push(render(&dumps));
    }
    assert!(
        renders.iter().all(|r| r == &renders[0]),
        "three compiles of the same six-struct program emitted different CLIF"
    );
}

snapshots! {
    // Both lowerings of each program.
    sgd_tape: "sgd", tape();
    sgd_source_ad: "sgd", source_ad();
    mlp_adamw_tape: "mlp_adamw", tape();
    mlp_adamw_source_ad: "mlp_adamw", source_ad();
    lion_tape: "lion", tape();
    lion_source_ad: "lion", source_ad();
    muon_tape: "muon", tape();
    muon_source_ad: "muon", source_ad();
    dataloader_checkpoint_tape: "dataloader_checkpoint", tape();
    dataloader_checkpoint_source_ad: "dataloader_checkpoint", source_ad();
    csla_ffn_source_ad: "csla_ffn", source_ad();
    csla_ffn_muon_source_ad: "csla_ffn_muon", source_ad();

    // The flags that take the train block down a different path, each on
    // its own (plus the flags it refuses to run without) over the
    // source-AD lowering of the program that exercises it.
    mlp_adamw_deterministic: "mlp_adamw", CompileOptions { deterministic: true, ..source_ad() };
    mlp_adamw_cuda_graphs: "mlp_adamw", CompileOptions { cuda_graphs: true, ..source_ad() };
    mlp_adamw_transient_arena: "mlp_adamw", CompileOptions { transient_arena: true, ..source_ad() };
    mlp_adamw_fuse_wgrad_accum: "mlp_adamw", CompileOptions { fuse_wgrad_accum: true, ..source_ad() };
    mlp_adamw_grad_integrity: "mlp_adamw", CompileOptions { grad_integrity: true, ..source_ad() };
    // Over the tape lowering: on the source-AD path the FASE hook owns the
    // gradients and the checksum this flag adds is skipped (inert variant).
    mlp_adamw_debug_training: "mlp_adamw", CompileOptions { debug_training: true, ..tape() };
    mlp_adamw_optim_state_offload: "mlp_adamw", CompileOptions { optim_state_offload: true, ..source_ad() };
    muon_optim_state_offload: "muon", CompileOptions { optim_state_offload: true, ..source_ad() };
    muon_resident_momentum: "muon", CompileOptions {
        optim_state_offload: true,
        muon_resident_momentum: true,
        ..source_ad()
    };
    dataloader_checkpoint_cuda_graphs: "dataloader_checkpoint", CompileOptions { cuda_graphs: true, ..source_ad() };
    csla_ffn_checkpoint_blocks: "csla_ffn", CompileOptions { checkpoint_blocks: true, ..source_ad() };
    csla_ffn_layerwise_accum: "csla_ffn", CompileOptions {
        checkpoint_blocks: true,
        layerwise_accum: true,
        ..source_ad()
    };
    csla_ffn_muon_layerwise_accum: "csla_ffn_muon", CompileOptions {
        checkpoint_blocks: true,
        layerwise_accum: true,
        ..source_ad()
    };
    csla_ffn_muon_state_bf16: "csla_ffn_muon", CompileOptions {
        checkpoint_blocks: true,
        layerwise_accum: true,
        muon_state_bf16: true,
        ..source_ad()
    };
}
