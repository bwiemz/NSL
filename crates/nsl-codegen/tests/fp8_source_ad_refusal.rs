//! Item 9 phase 2 — `@fp8_compute` must REFUSE under source-to-source AD.
//!
//! # The bug
//!
//! `@fp8_compute` routes matmuls in the decorated body to
//! `nsl_fp8_matmul_training`, which records a `TapeOp::Fp8MatMul` so the
//! backward applies the E5M2 round-trip to both gradients.  That routing
//! lives in `expr/advanced.rs` (`state.flags.is_fp8_compute`).
//!
//! Source AD never executes that code.  It inlines the method body into a
//! `WengertList` and lowers it through `wengert_lower.rs`, where
//! `PrimalOp::Matmul` emits an unconditional `nsl_tensor_matmul`.  Neither
//! `source_ad.rs` nor `wengert_lower.rs` contains a single occurrence of
//! "fp8".  So a user who writes the decorator and trains with `--source-ad`
//! — the default, and the path every production model in this repo uses —
//! gets plain f32 with no error and no warning.
//!
//! # What these tests pin
//!
//! * `tape_ad_actually_routes_the_matmul_through_fp8` — the NEGATIVE
//!   CONTROL.  Without it, the refusal below could be "passing" against a
//!   decorator that never worked on any path, and the whole diagnostic
//!   would be a lie.  Under tape AD the SAME program emits
//!   `nsl_fp8_matmul_training` and NO `nsl_tensor_matmul`, proving the
//!   decorator has a real effect that source AD is dropping.
//!
//! * `source_ad_refuses_a_decorated_method` — the refusal fires, and the
//!   message names the offending `Model::method` so the user knows which
//!   decorator to remove.
//!
//! * `undecorated_model_still_compiles_under_source_ad` — the refusal is
//!   keyed on the decorator, not on source AD being enabled.
//!
//! * `an_unrelated_models_same_named_method_does_not_trip_the_refusal` —
//!   the record is keyed `Model::method`, not by bare method name.  Two
//!   models both called `forward`, one decorated and NOT trained, must not
//!   poison the other.

#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;
use object::{Object, ObjectSection, ObjectSymbol};
use std::collections::{HashMap, HashSet};

/// One decorated model, trained. The single matmul in the program is the
/// one inside the decorated `forward`.
const DECORATED: &str = r#"
model FP8Model:
    w: Tensor = ones([8, 8])

    @fp8_compute
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = FP8Model()
let x = ones([8, 8])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

/// Byte-identical to `DECORATED` minus the decorator.
const UNDECORATED: &str = r#"
model FP8Model:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = FP8Model()
let x = ones([8, 8])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

/// Two models, both with a method named `forward`. Only the one that is
/// NOT trained carries the decorator.
const DECORATED_BYSTANDER: &str = r#"
model Helper:
    hw: Tensor = ones([8, 8])

    @fp8_compute
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.hw

model FP8Model:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = FP8Model()
let x = ones([8, 8])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

/// The production shape: the decorated method sits on a SUB-model reached
/// through `self.field`, not on the model the train block names. This
/// exercises the nested-model inline site, which resolves the callee's
/// model type through `model_field_types` rather than
/// `model_instance_types`.
const DECORATED_SUBMODEL: &str = r#"
model Inner:
    iw: Tensor = ones([8, 8])

    @fp8_compute
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.iw

model Outer:
    inner: Inner = Inner()

    fn forward(self, x: Tensor) -> Tensor:
        return self.inner.forward(x)

let m = Outer()
let x = ones([8, 8])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

fn ensure_stdlib_path() {
    if std::env::var("NSL_STDLIB_PATH").is_err() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root above crates/nsl-codegen");
        std::env::set_var("NSL_STDLIB_PATH", workspace_root.join("stdlib"));
    }
}

/// Compile `src`, returning the object bytes or the codegen error message.
fn compile(src: &str, source_ad: bool) -> Result<Vec<u8>, String> {
    ensure_stdlib_path();
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(
        !lex_diags.iter().any(|d| matches!(d.level, Level::Error)),
        "lex errors"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        !parsed
            .diagnostics
            .iter()
            .any(|d| matches!(d.level, Level::Error)),
        "parse errors: {:?}",
        parsed.diagnostics
    );
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        !analysis
            .diagnostics
            .iter()
            .any(|d| matches!(d.level, Level::Error)),
        "semantic errors: {:?}",
        analysis.diagnostics
    );

    let opts = CompileOptions {
        source_ad,
        target: "sm_75".to_string(),
        ..Default::default()
    };
    let imported_fns = nsl_codegen::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        &opts,
    )
    .map_err(|e| format!("stdlib_loader: {}", e.message))?;

    nsl_codegen::compile_entry(
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
        false,
        &opts,
    )
    .map_err(|e| e.message)
}

/// Symbols referenced by a relocation. Symbol-table membership alone does
/// not prove a call was emitted — `declare_runtime_functions` declares
/// every FFI unconditionally.
fn relocation_symbol_set(obj_bytes: &[u8]) -> HashSet<String> {
    let file = object::File::parse(obj_bytes).expect("object::File::parse");
    let name_by_index: HashMap<_, _> = file
        .symbols()
        .filter_map(|s| {
            let n = s.name().ok()?;
            Some((
                s.index(),
                nsl_codegen::linker::strip_host_symbol_prefix(n).to_string(),
            ))
        })
        .collect();

    let mut out = HashSet::new();
    for section in file.sections() {
        for (_offset, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                if let Some(name) = name_by_index.get(&idx) {
                    out.insert(name.clone());
                }
            }
        }
    }
    out
}

/// NEGATIVE CONTROL for the refusal below.
///
/// Under tape AD the decorated program's ONLY matmul goes through
/// `nsl_fp8_matmul_training`, and `nsl_tensor_matmul` is absent from the
/// relocation set entirely. That absence is the point: it proves the
/// decorator changes generated code on at least one path, so "source AD
/// drops it" is a real regression rather than a decorator that never did
/// anything anywhere.
#[test]
fn tape_ad_actually_routes_the_matmul_through_fp8() {
    let obj = compile(DECORATED, false).expect("tape-AD compile should succeed");
    let relocs = relocation_symbol_set(&obj);
    assert!(
        relocs.contains("nsl_fp8_matmul_training"),
        "tape AD should route the decorated matmul through nsl_fp8_matmul_training"
    );
    assert!(
        !relocs.contains("nsl_tensor_matmul"),
        "the program's only matmul is the decorated one, so a plain \
         nsl_tensor_matmul reloc under tape AD means the decorator was \
         bypassed and this control proves nothing"
    );
}

/// The refusal. Before this change the same compile returned Ok and the
/// step trained in f32.
#[test]
fn source_ad_refuses_a_decorated_method() {
    // Not `expect_err`: the Ok payload is a whole object file, and dumping
    // ~130 KB of `Vec<u8>` Debug output buries the actual failure.
    let err = match compile(DECORATED, true) {
        Err(e) => e,
        Ok(obj) => panic!(
            "source AD must refuse a @fp8_compute method, not silently drop \
             it — but the compile succeeded, emitting {} bytes of object",
            obj.len()
        ),
    };
    assert!(
        err.contains("@fp8_compute"),
        "the diagnostic must name the decorator; got: {err}"
    );
    assert!(
        err.contains("FP8Model::forward"),
        "the diagnostic must name the offending Model::method so the user \
         knows which decorator to remove; got: {err}"
    );
    assert!(
        err.contains("--tape-ad"),
        "the diagnostic must name the path on which the decorator DOES \
         work; got: {err}"
    );
}

/// The refusal keys on the decorator, not on source AD being on.
#[test]
fn undecorated_model_still_compiles_under_source_ad() {
    let obj = compile(UNDECORATED, true)
        .expect("an undecorated model must still compile under source AD");
    let relocs = relocation_symbol_set(&obj);
    assert!(
        relocs.contains("nsl_tensor_matmul"),
        "sanity: the undecorated step should emit a plain matmul"
    );
}

/// The nested-model inline site must record too. `m.forward` is
/// undecorated; the decorator is on `Inner::forward`, reached via
/// `self.inner.forward(x)` — which is how every real model in this repo
/// is structured. A refusal that only covered the top-level call would
/// pass `source_ad_refuses_a_decorated_method` and still miss every
/// production program.
#[test]
fn source_ad_refuses_a_decorated_submodel_method() {
    let err = match compile(DECORATED_SUBMODEL, true) {
        Err(e) => e,
        Ok(obj) => panic!(
            "source AD must refuse a @fp8_compute method reached through a \
             sub-model field — but the compile succeeded, emitting {} bytes \
             of object",
            obj.len()
        ),
    };
    assert!(
        err.contains("Inner::forward"),
        "the diagnostic must name the SUB-model's method, not the outer \
         one; got: {err}"
    );
}

/// The record is keyed `Model::method`. A same-named method on a model
/// that is never trained must not fail an unrelated train block — that
/// would be a false refusal on a valid program.
#[test]
fn an_unrelated_models_same_named_method_does_not_trip_the_refusal() {
    compile(DECORATED_BYSTANDER, true).expect(
        "only FP8Model::forward is inlined by this train block; \
         Helper::forward carries the decorator but is never inlined, so \
         the refusal must not fire",
    );
}
