//! Gap G — FFI scale argument type alignment for `compile_flash_attention_call`.
//!
//! Before this fix, `compile_flash_attention_call` pushed `scale_val`
//! directly onto the call arg list for `nsl_flash_attention` /
//! `nsl_flash_attention_csha` / `nsl_flash_attention_csha_with_saves`.
//! All three FFI signatures declare arg 5 (`scale_bits`) as `I64`
//! carrying the f32 bit pattern (matching the test-harness convention
//! `scale.to_bits() as i64` and the canonical codegen pattern in
//! `wengert_lower.rs::emit_fused_attention_call`).
//!
//! However `scale_val` came from `compile_expr` on the source-level
//! 4th argument to `scaled_dot_product_attention`.  For a scalar
//! expression like `1.0 / sqrt(head_dim)` this lowers to a Cranelift
//! F64 Value.  Feeding F64 into an I64 argument slot triggers a
//! Cranelift verifier error (`inst.ins_types[n] != arg.value_type`),
//! and the outer `Module::define_function` fails with a generic
//! "Verifier errors" message that skinned the Gap F toy-pretrain
//! smoke into a skip-with-diagnostic.
//!
//! Gap G inserts an F64/F32 → I32-bits → uextend-to-I64 conversion
//! at the single site `scale_val` enters `compile_flash_attention_call`,
//! so all three outgoing FFI calls receive a well-typed I64
//! `scale_bits` regardless of what source-level type the expression
//! compiled to.
//!
//! ## What this test pins
//!
//!   1. A source program that calls `scaled_dot_product_attention`
//!      with a Cranelift-F64 `scale` expression inside an
//!      `@flash_attention(head_dim=32)`-decorated method **compiles
//!      end-to-end without a Cranelift verifier error**.
//!   2. The emitted object references `nsl_flash_attention_csha` (the
//!      inference path exercising the same scale-bits call arg the
//!      Gap F smoke previously couldn't reach).
//!
//! ## What it does NOT cover
//!
//!   - Real GPU execution.
//!   - The `@train` block path — that still requires Gap F.2 decorator
//!     arg threading + the optimizer-symbol-mangling fix to land
//!     before `nsl_flash_attention_csha_with_saves` is reachable from
//!     source.  Those are out of Gap G's "fix only the type mismatch"
//!     scope; see `csha_gap_f_toy_pretrain_smoke.rs`.

#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;
use object::{Object, ObjectSection, ObjectSymbol};
use std::collections::{HashMap, HashSet};

/// Tiny inference program: one `@flash_attention(head_dim=32)` method
/// with a scalar `scale = 1.0 / sqrt(32.0)` expression (F64 in
/// Cranelift IR) passed as the 4th arg to
/// `scaled_dot_product_attention`.  No `@train` block — we only need
/// to exercise the FFI verifier surface, not AD.
const FA_INFERENCE_SRC: &str = r#"
model TinyAttn:
    wq: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wq
        let v = x @ self.wq
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([32, 32])
let out = m.forward(x)
"#;

fn compile_program(src: &str) -> Result<Vec<u8>, String> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, Level::Error)) {
        return Err(format!(
            "lex errors: {:?}",
            lex_diags
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        ));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        return Err(format!(
            "parse errors: {:?}",
            parsed
                .diagnostics
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        ));
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        return Err(format!(
            "semantic errors: {:?}",
            analysis
                .diagnostics
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        ));
    }
    let opts = CompileOptions {
        csha: nsl_codegen::CshaOptions {
            mode: Some("auto".into()),
            ..Default::default()
        },
        target: "sm_75".to_string(),
        ..Default::default()
    };
    let imported_fns = nsl_codegen::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        &opts,
    )
    .map_err(|e| format!("stdlib_loader failed: {}", e.message))?;
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
    .map_err(|e| format!("compile_entry failed: {}", e.message))
}

fn relocation_symbol_set(obj_bytes: &[u8]) -> HashSet<String> {
    let file = object::File::parse(obj_bytes).expect("object::File::parse");
    let name_by_index: HashMap<_, _> = file
        .symbols()
        .filter_map(|s| s.name().ok().map(|n| (s.index(), n.to_string())))
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

/// Load-bearing assertion: compiling a source-level
/// `scaled_dot_product_attention(q, k, v, scale)` call with a Cranelift
/// F64 `scale` under an `@flash_attention`-decorated method must NOT
/// produce a verifier error.
///
/// Pre-Gap-G this failed with a generic "Verifier errors" from
/// `Module::define_function` because `scale_val` (F64) was pushed
/// directly into the I64 `scale_bits` argument slot of
/// `nsl_flash_attention_csha`.  Post-Gap-G the compile succeeds and
/// the object references the forward FFI.
#[test]
fn inference_scale_expr_compiles_and_references_csha_forward_ffi() {
    let obj = match compile_program(FA_INFERENCE_SRC) {
        Ok(b) => b,
        Err(e) => panic!(
            "[gap-g] compile failed — Gap G regression suspected. \
             If this message contains 'Verifier errors', the scale-bits \
             conversion in compile_flash_attention_call regressed.  \
             If it mentions an unrelated symbol (e.g. an optim FFI name), \
             the fix is still intact — just expand this test's program \
             to isolate the FFI verifier path.  error: {e}"
        ),
    };

    let called = relocation_symbol_set(&obj);
    let fired_forward_csha = called.contains("nsl_flash_attention_csha")
        || called.contains("nsl_flash_attention_csha_with_saves");
    let fired_forward_plain = called.contains("nsl_flash_attention");

    // Diagnostic — surface everything FA-related.
    let fa_related: Vec<&String> = called
        .iter()
        .filter(|n| n.contains("flash_attention"))
        .collect();
    eprintln!("[gap-g] flash_attention relocations: {fa_related:?}");

    assert!(
        fired_forward_csha || fired_forward_plain,
        "[gap-g] compile succeeded (no verifier error — good) but neither \
         `nsl_flash_attention_csha` nor `nsl_flash_attention` is \
         referenced.  Either the `@flash_attention` decorator scan failed \
         to build a FlashAttentionContext (unit tests in \
         csha_gap_f_decorator.rs pin that separately), or the SDPA call \
         was routed through the naive (non-flash) lowering.  \
         relocations={called:?}"
    );
}
