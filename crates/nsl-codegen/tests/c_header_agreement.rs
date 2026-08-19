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
use nsl_codegen::c_header::{
    emit, ExportDevice, ExportDtype, ExportInfo, ExportParamInfo, ExportTypeInfo,
};

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

    // The header is OURS: every spelling in it is one `c_header.rs` chose, so
    // an unmodelled one means `abi_from_c` cannot see a type we emit — a hole
    // in the gate, not a "cannot verify". (The runtime side keeps
    // cannot-verify semantics: it legitimately contains types this crate does
    // not model.)
    let unmodelled: Vec<String> = protos
        .iter()
        .flat_map(|p| {
            p.params
                .iter()
                .chain(p.ret.iter())
                .filter_map(move |t| match t {
                    ParsedType::Unknown(s) if !s.contains('*') => {
                        Some(format!("  {}: `{s}`", p.name))
                    }
                    _ => None,
                })
        })
        .collect();
    assert!(
        unmodelled.is_empty(),
        "the generated header uses {} C type spelling(s) `abi_from_c` does not \
         model, so nothing checks them — teach the mapping or change the \
         emitter:\n{}",
        unmodelled.len(),
        unmodelled.join("\n")
    );

    let mut checked = 0usize;
    let mut problems = Vec::new();
    for p in &protos {
        // ALL same-name externs, not the first: `#[cfg]` variants of one
        // symbol must agree with the header individually (nsl-abi learned this
        // the same way — see its `every_same_name_variant_must_agree_not_just_one`).
        let impls: Vec<_> = externs.iter().filter(|e| e.name == p.name).collect();
        if impls.is_empty() {
            // Not every header name is an `extern "C"` in this one file (the
            // `NslExportFn` typedef is checked by the test below). Names that
            // match nothing are reported by the coverage assert, not here.
            continue;
        }
        checked += 1;
        for imp in impls {
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
    }

    // Non-vacuity, part 2 — by NAME, not by count. A floor set just under the
    // current value (the first draft said `>= 5`) lets a symbol silently drop
    // out of the header, which is the other half of the drift this gate is
    // for: the runtime keeps the function, the header stops describing it,
    // and a host writing its own extern declaration is back to guessing.
    for required in [
        "nsl_abi_version",
        "nsl_model_create",
        "nsl_model_destroy",
        "nsl_model_call",
        // Item-7 ownership models + introspection. Without these entries a
        // dropped prototype would pass silently — the pairwise comparison
        // above only checks names present in BOTH lists.
        "nsl_model_call_into",
        "nsl_model_call_alloc",
        "nsl_model_get_export_signature",
        "nsl_get_last_error",
        "nsl_clear_error",
    ] {
        assert!(
            protos.iter().any(|p| p.name == required),
            "the generated header no longer declares `{required}` — either it was \
             dropped from the emitter, or the parser stopped seeing it"
        );
        assert!(
            externs.iter().any(|e| e.name == required),
            "`{required}` is declared in the header but is no longer an extern \"C\" \
             in c_api/mod.rs"
        );
    }
    assert!(
        checked >= 6,
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

/// One `@export` per scalar dtype, so every arm of the slot mapping is
/// rendered into a real prototype.
fn every_scalar_dtype_export() -> ExportInfo {
    let dtypes = [
        ExportDtype::I8,
        ExportDtype::I16,
        ExportDtype::I32,
        ExportDtype::I64,
        ExportDtype::U8,
        ExportDtype::U16,
        ExportDtype::U32,
        ExportDtype::U64,
        ExportDtype::F32,
        ExportDtype::F64,
        ExportDtype::F16,
        ExportDtype::BF16,
        ExportDtype::Bool,
    ];
    let mut params = vec![ExportParamInfo {
        name: "x".into(),
        ty: ExportTypeInfo::Tensor {
            shape: vec!["B".into(), "8".into()],
            dtype: ExportDtype::F32,
            device: ExportDevice::Any,
        },
    }];
    for (i, dt) in dtypes.iter().enumerate() {
        params.push(ExportParamInfo {
            name: format!("s{i}"),
            ty: ExportTypeInfo::Scalar(*dt),
        });
    }
    ExportInfo {
        symbol_name: "forward".into(),
        raw_name: "forward".into(),
        params,
        return_type: ExportTypeInfo::Tensor {
            shape: vec!["B".into(), "4".into()],
            dtype: ExportDtype::F32,
            device: ExportDevice::Any,
        },
    }
}

/// The `@export` prototypes must describe the slots their wrapper reads.
///
/// This is the half the first version of this gate could not see: it built the
/// header with an EMPTY export list, so `emit_prototype`/`emit_param` never
/// ran. A review found a live drift hiding there — `c_type_for_scalar` mapped
/// `ExportDtype` independently of `c_wrapper::cranelift_type_for_scalar`,
/// which collapses everything but I32/I64/F32/F64 into a 64-bit slot, so
/// `@export fn f(..., k: u8)` was declared `uint8_t k` against a callee
/// reading a full register. Same defect as the `NslExportFn` typedef, one
/// function away from it.
#[test]
fn export_prototypes_agree_with_the_wrapper_signature_codegen_builds() {
    use cranelift_codegen::ir::types;
    use cranelift_codegen::isa::CallConv;

    let info = every_scalar_dtype_export();
    let header = emit(std::slice::from_ref(&info), "exports");
    let protos = parse_c_prototypes(&header);
    let proto = protos
        .iter()
        .find(|p| p.name == info.symbol_name)
        .unwrap_or_else(|| {
            panic!(
                "no prototype for the `{}` export in:\n{header}",
                info.symbol_name
            )
        });

    let sig = nsl_codegen::c_wrapper::build_c_abi_wrapper_signature(&info, CallConv::SystemV);

    assert_eq!(
        proto.params.len(),
        sig.params.len(),
        "prototype declares {} param(s) [{}]; the wrapper takes {}",
        proto.params.len(),
        render_all(&proto.params),
        sig.params.len()
    );

    let mut problems = Vec::new();
    for (i, (h, emitted)) in proto.params.iter().zip(sig.params.iter()).enumerate() {
        let want = match emitted.value_type {
            types::I8 => AbiScalar::Int(8),
            types::I16 => AbiScalar::Int(16),
            types::I32 => AbiScalar::Int(32),
            types::I64 => AbiScalar::Int(64),
            types::F32 => AbiScalar::Float(32),
            types::F64 => AbiScalar::Float(64),
            other => panic!("teach this gate the new slot type {other:?} at param {i}"),
        };
        // Unknown here would mean the parser could not read a spelling the
        // emitter produced — a gate hole, not a pass.
        match h {
            ParsedType::Known(k) if *k == want => {}
            ParsedType::Known(k) => problems.push(format!(
                "  param {i}: header says {}, wrapper reads {}",
                render(&ParsedType::Known(*k)),
                render(&ParsedType::Known(want))
            )),
            ParsedType::Unknown(s) => problems.push(format!(
                "  param {i}: header spelling `{s}` is not modelled by abi_from_c, so \
                 nothing checks it"
            )),
        }
    }
    assert!(
        problems.is_empty(),
        "{} @export prototype disagreement(s) with the wrapper signature — a C host \
         passes the declared width into a callee reading the slot width:\n{}",
        problems.len(),
        problems.join("\n")
    );

    let ret = proto.ret.as_ref().expect("prototype must return a status");
    assert_eq!(sig.returns.len(), 1);
    assert_eq!(
        *ret,
        ParsedType::Known(AbiScalar::Int(32)),
        "@export status is {} in the header, I32 in the wrapper",
        render(ret)
    );
}

/// The header's convenience inlines must not shadow runtime symbols.
///
/// They were emitted as `static inline int32_t nsl_model_forward(...)` /
/// `nsl_model_backward(...)`, both of which the runtime exports with different
/// signatures — `nsl_model_backward` takes a `GradContext*` first argument.
/// A host that included the header and called either name silently bound to
/// the file-local inline instead of the library symbol.
#[test]
fn header_inlines_do_not_shadow_runtime_symbols() {
    let mut info = every_scalar_dtype_export();
    info.symbol_name = "forward".into();
    let mut backward = info.clone();
    backward.symbol_name = "backward".into();
    let header = emit(&[info, backward], "shadow");

    let capi = workspace_root().join("crates/nsl-runtime/src/c_api/mod.rs");
    let grad = workspace_root().join("crates/nsl-runtime/src/grad_context.rs");
    let mut runtime_names: Vec<String> = Vec::new();
    for f in [&capi, &grad] {
        let src = std::fs::read_to_string(f).unwrap_or_default();
        runtime_names.extend(
            parse_externs_in_file(&src, "runtime")
                .into_iter()
                .map(|s| s.name),
        );
    }
    assert!(
        runtime_names.iter().any(|n| n == "nsl_model_forward")
            && runtime_names.iter().any(|n| n == "nsl_model_backward"),
        "the two symbols this test exists to protect are no longer exported — \
         re-target it rather than deleting it"
    );

    for name in &runtime_names {
        let defined_as_inline = header.contains(&format!("static inline int64_t {name}("))
            || header.contains(&format!("static inline int32_t {name}("));
        assert!(
            !defined_as_inline,
            "the generated header DEFINES `{name}` as a static inline, shadowing the \
             runtime symbol of the same name for every host that includes it"
        );
    }

    // And the convenience inlines are still emitted, under their own names —
    // otherwise this test passes by the header having no inlines at all.
    assert!(
        header.contains("static inline int64_t nsl_export_forward(")
            && header.contains("static inline int64_t nsl_export_backward("),
        "the convenience inlines vanished; this test would then be vacuous:\n{header}"
    );
}
