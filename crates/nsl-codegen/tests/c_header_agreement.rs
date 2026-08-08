//! The generated C header must describe the ABI the runtime actually
//! implements — machine-checked, not by eye.
//!
//! ## Why this gate did not exist and needed to
//!
//! `nsl-abi`'s `cross_check` validates the codegen's `RUNTIME_FUNCTIONS` table
//! against the runtime's `extern "C"` bodies, and it is thorough (650
//! signatures). But it iterates the DECLARED TABLE, so a surface with no table
//! entry is invisible to it — and the emitted header is exactly that: literal
//! C text assembled in `c_header.rs`, describing runtime symbols the compiler
//! never declares to Cranelift because host code, not emitted code, calls them.
//!
//! Two existing tests touch the header and neither could see a type:
//! `c_header_compiles.rs` takes the address of one prototype and otherwise runs
//! `-fsyntax-only` (a wrong-but-consistent prototype compiles fine), and
//! `c_header_snapshot.rs` asserts `header.contains("nsl_model_destroy")`, which
//! a wrong return type passes.
//!
//! When first written, this gate was RED on the shipped header:
//!
//! * `typedef int32_t (*NslExportFn)(..., int32_t n_inputs, ..., int32_t
//!   n_outputs)` against the five-I64-params-returning-I64 signature that
//!   `build_dispatch_wrapper_signature` actually emits. A host calling an
//!   export through this typedef passes 32-bit counts into a callee reading
//!   64-bit registers — the upper halves are caller-undefined — and truncates
//!   the returned status.
//! * `void nsl_model_destroy(NslModel*)` against `-> i64`.

use nsl_abi::{parse_c_prototypes, parse_externs_in_file, AbiScalar, ParsedType};
use nsl_codegen::c_header::emit;

fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// The header for a module with no `@export` functions — i.e. exactly the
/// fixed runtime surface every generated header carries.
fn fixed_surface_header() -> String {
    emit(&[], "agreement_gate")
}

fn render(t: &ParsedType) -> String {
    match t {
        ParsedType::Known(AbiScalar::Int(w)) => format!("i{w}"),
        ParsedType::Known(AbiScalar::Float(w)) => format!("f{w}"),
        ParsedType::Unknown(s) => format!("?{s}"),
    }
}

fn render_all(ts: &[ParsedType]) -> String {
    ts.iter().map(render).collect::<Vec<_>>().join(", ")
}

#[test]
fn generated_header_prototypes_agree_with_runtime_externs() {
    let header = fixed_surface_header();
    let protos = parse_c_prototypes(&header);

    // Non-vacuity, part 1: the parser must actually find the surface. A
    // silently-degrading parser is the classic way a drift gate stops
    // guarding anything while staying green.
    assert!(
        protos.len() >= 6,
        "parsed only {} prototype(s) from the generated header — the emitter's \
         format changed and this gate is checking almost nothing:\n{}",
        protos.len(),
        header
    );

    let capi = workspace_root().join("crates/nsl-runtime/src/c_api/mod.rs");
    let src = std::fs::read_to_string(&capi).expect("read c_api/mod.rs");
    let externs = parse_externs_in_file(&src, "nsl-runtime/src/c_api/mod.rs");
    assert!(
        externs.len() >= 20,
        "parsed only {} extern fn(s) from c_api/mod.rs",
        externs.len()
    );

    let mut checked = 0usize;
    let mut problems = Vec::new();
    for p in &protos {
        let Some(imp) = externs.iter().find(|e| e.name == p.name) else {
            // Not every header name is an `extern "C"` in this one file (the
            // `NslExportFn` typedef is checked by the test below). Names that
            // match nothing are reported by the coverage assert, not here.
            continue;
        };
        checked += 1;
        if p.params.len() != imp.params.len() {
            problems.push(format!(
                "  {}: header declares {} param(s) [{}], runtime takes {} [{}]",
                p.name,
                p.params.len(),
                render_all(&p.params),
                imp.params.len(),
                render_all(&imp.params)
            ));
            continue;
        }
        for (i, (h, r)) in p.params.iter().zip(imp.params.iter()).enumerate() {
            // An Unknown on either side means "cannot verify", not "differs":
            // `NslModel*` is a header-only opaque tag. Pointers on both sides
            // already collapse to Int(64), so this only skips genuinely
            // unmodelled spellings.
            if matches!(h, ParsedType::Unknown(_)) || matches!(r, ParsedType::Unknown(_)) {
                continue;
            }
            if h != r {
                problems.push(format!(
                    "  {}: param {} is {} in the header, {} in the runtime",
                    p.name,
                    i,
                    render(h),
                    render(r)
                ));
            }
        }
        match (&p.ret, &imp.ret) {
            (None, Some(r)) => problems.push(format!(
                "  {}: header returns void, runtime returns {}",
                p.name,
                render(r)
            )),
            (Some(h), None) => problems.push(format!(
                "  {}: header returns {}, runtime returns nothing",
                p.name,
                render(h)
            )),
            (Some(h), Some(r)) => {
                if !matches!(h, ParsedType::Unknown(_))
                    && !matches!(r, ParsedType::Unknown(_))
                    && h != r
                {
                    problems.push(format!(
                        "  {}: header returns {}, runtime returns {}",
                        p.name,
                        render(h),
                        render(r)
                    ));
                }
            }
            (None, None) => {}
        }
    }

    // Non-vacuity, part 2: matching zero names would make the loop above a
    // no-op. Every lifecycle/error prototype in the fixed surface is an
    // `extern "C"` in c_api/mod.rs, so the floor is the size of that set.
    assert!(
        checked >= 5,
        "only {checked} header prototype(s) matched a runtime extern — the \
         names diverged, so this gate compared almost nothing"
    );

    assert!(
        problems.is_empty(),
        "{} ABI disagreement(s) between the generated C header and the runtime \
         it describes — a host that trusts the header calls with the wrong \
         register widths:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

#[test]
fn the_export_fn_typedef_matches_the_signature_codegen_emits() {
    use cranelift_codegen::ir::types;
    use cranelift_codegen::isa::CallConv;

    let header = fixed_surface_header();
    let typedef = parse_c_prototypes(&header)
        .into_iter()
        .find(|p| p.name == "NslExportFn")
        .expect("the header must declare the NslExportFn typedef");

    // Compared against the emitter itself, not against a transcription of it:
    // this is the signature every `@export` dispatch wrapper is built with.
    let sig = nsl_codegen::c_wrapper::build_dispatch_wrapper_signature(CallConv::SystemV);

    assert_eq!(
        typedef.params.len(),
        sig.params.len(),
        "NslExportFn declares {} param(s) [{}]; codegen emits {}",
        typedef.params.len(),
        render_all(&typedef.params),
        sig.params.len()
    );

    for (i, (h, emitted)) in typedef.params.iter().zip(sig.params.iter()).enumerate() {
        if matches!(h, ParsedType::Unknown(_)) {
            continue; // `NslModel*` / `NslTensorDesc*` are header-only tags
        }
        let want = if emitted.value_type == types::I64 {
            ParsedType::Known(AbiScalar::Int(64))
        } else {
            panic!(
                "the dispatch wrapper signature grew a non-I64 param {i} \
                 ({:?}) — teach this gate the new type rather than deleting \
                 the check",
                emitted.value_type
            );
        };
        assert_eq!(
            *h,
            want,
            "NslExportFn param {i} is {} in the header but I64 in the emitted \
             signature — a host passing a 32-bit count leaves the callee's \
             upper half caller-undefined",
            render(h)
        );
    }

    let ret = typedef
        .ret
        .as_ref()
        .expect("NslExportFn must return a status, not void");
    assert_eq!(
        sig.returns.len(),
        1,
        "dispatch wrapper stopped returning exactly one value"
    );
    assert_eq!(
        *ret,
        ParsedType::Known(AbiScalar::Int(64)),
        "NslExportFn returns {} in the header but I64 in the emitted \
         signature — the status is truncated on the way back to the host",
        render(ret)
    );
}

#[test]
fn the_c_type_mapping_does_not_silently_accept_nonsense() {
    use nsl_abi::abi_from_c;
    // Widths must be distinguished, or the gate above cannot fail.
    assert_eq!(abi_from_c("int64_t"), Some(AbiScalar::Int(64)));
    assert_eq!(abi_from_c("int32_t"), Some(AbiScalar::Int(32)));
    assert_ne!(abi_from_c("int32_t"), abi_from_c("int64_t"));
    // Pointers are machine words regardless of pointee, matching abi_from_rust.
    assert_eq!(abi_from_c("const NslTensorDesc*"), Some(AbiScalar::Int(64)));
    assert_eq!(abi_from_c("void*"), Some(AbiScalar::Int(64)));
    // Float and int of the same width are different register classes.
    assert_ne!(abi_from_c("double"), abi_from_c("int64_t"));
    // Unmodelled spellings must be None (-> Unknown -> "cannot verify"),
    // never a guess.
    assert_eq!(abi_from_c("NslModel"), None);
}
