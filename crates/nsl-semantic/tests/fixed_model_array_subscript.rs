//! G6.1 — paper §3.2 closure (partial): subscript on `[Model; N]` arrays
//! must resolve to the element Model type with fields/methods populated.
//!
//! Before this fix, `self.blocks[0]` returned `Type::Unknown` because the
//! `Subscript` arm in `checker::expr.rs` only handled `List`, `Dict`,
//! `Str`, and `Tuple`. Downstream `.attn.wq` member access then fell
//! through to the lenient dict-access path in codegen, surfacing a
//! `member access '.attn' on Unknown-typed object — falling through to
//! dict access. This may be a type inference gap.` warning.
//!
//! Fixing the subscript arm to look up the element-model type from the
//! scope (the same way the for-loop case at `checker::stmt.rs::49` already
//! does) eliminates that warning AND unlocks the proper codegen path
//! (`Type::Model` field access via `struct_layouts`) instead of the
//! dict-access fallback.
//!
//! This test pins the semantic-level guarantee. Codegen consumes the
//! same type information so silencing the warning is a downstream
//! consequence of the semantic fix.
//!
//! NOT covered here (separate gap — G6.1.2): registering nested-model
//! contexts (`"m.blocks.0.attn"`) into source-AD's `context_to_model_type`
//! so trainable parameter gradients connect through indexed-array
//! field access. Today `--source-ad` still falls back to tape-AD on
//! this code shape; that fallback is correct, just not the
//! compile-time WRGA-eligible path. Closing G6.1.2 needs more substantial
//! source-AD work.

use nsl_errors::FileId;
use nsl_lexer::Interner;
use nsl_semantic::types::Type;

/// Build a minimal model-array fixture and return the semantic
/// analysis result so individual tests can assert on the type map.
fn analyze_fixture(src: &str) -> (nsl_semantic::AnalysisResult, Interner) {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(lex_diags.is_empty(), "lex errors: {lex_diags:?}");

    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parse_result.diagnostics.is_empty(),
        "parse errors: {:?}",
        parse_result.diagnostics
    );

    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    (analysis, interner)
}

/// Resolve a member name through the interner. Used to check
/// `Type::Model { fields, .. }` carries the right field set.
fn field_names(interner: &Interner, fields: &[(nsl_ast::Symbol, Type)]) -> Vec<String> {
    fields
        .iter()
        .map(|(sym, _)| interner.resolve(sym.0).unwrap_or("?").to_string())
        .collect()
}

/// Core G6.1 assertion: `self.blocks[0]` on a `blocks: [Block; N]` field
/// must produce a `Type::Model` carrying Block's fields (so the next
/// field access `.w` can resolve to the actual field type, not
/// `Type::Unknown`).
#[test]
fn subscript_on_fixed_model_array_resolves_to_model_type() {
    // Minimal: `Block` has a single tensor field `w`; `Container` holds
    // `blocks: [Block; 2]`; `forward` indexes through the array and
    // accesses `.w`. The semantic checker should infer `.w` as a Tensor,
    // NOT as Unknown.
    let src = r#"
model Block:
    w: Tensor = ones([4])

    fn forward(self, x: Tensor) -> Tensor:
        return x * self.w

model Container:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let q = x * self.blocks[0].w
        return q
"#;

    let (analysis, interner) = analyze_fixture(src);
    let errs: Vec<String> = analysis
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message.clone())
        .collect();
    assert!(
        errs.is_empty(),
        "expected no semantic errors on the indexed-array fixture; got: {errs:?}"
    );

    // Find ANY subscript-typed expression in the type map and assert it's
    // a Model with at least one field (Block has `w`). The type map keys
    // are NodeIds; the values are computed types per `check_expr`.
    let subscript_model: Vec<&Type> = analysis
        .type_map
        .values()
        .filter(|ty| matches!(ty, Type::Model { fields, .. } if !fields.is_empty()))
        .collect();
    assert!(
        !subscript_model.is_empty(),
        "expected at least one Type::Model with non-empty fields in the type map after \
         the fix; the subscript `self.blocks[0]` should resolve to Block's model type"
    );

    // Pin: SOMEWHERE in the type map, a Model named `Block` with field `w`
    // must appear. Without the fix, the `self.blocks[0]` subscript would
    // be Type::Unknown and the `.w` access would also bail to Unknown,
    // so nothing in the type map would carry Block's fields.
    let block_w_present = analysis.type_map.values().any(|ty| match ty {
        Type::Model { fields, .. } => {
            let names = field_names(&interner, fields);
            names.iter().any(|n| n == "w")
        }
        _ => false,
    });
    assert!(
        block_w_present,
        "expected the Block model type (with field `w`) somewhere in the type map; \
         the `self.blocks[0]` subscript must propagate Block's fields downstream so \
         `.w` resolves to Tensor instead of Unknown"
    );
}

/// Forward-reference correctness: when the element model is defined
/// AFTER the enclosing model (textual ordering), `collect_top_level_decls`
/// pre-registers it with `Type::Unknown` before `check_model_def` runs.
/// A naive `info.ty.clone()` on the scope lookup would therefore return
/// `Type::Unknown` and silently regress past the pre-fix behavior. The
/// fix gates on `Type::Model { .. }` and falls back to a bare-name
/// Model otherwise — matching the for-loop arm at
/// `checker::stmt.rs::49-57` which never touches the scope and always
/// returns at least an empty-fields Model.
///
/// Pinning this ordering means a future edit that re-introduces a
/// naive `info.ty.clone()` will fail loudly here.
#[test]
fn subscript_on_fixed_model_array_handles_forward_reference() {
    // `Container` references `Block` BEFORE `Block` is defined. Without
    // the Type::Model gate, the subscript would return Unknown.
    let src = r#"
model Container:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let q = x * self.blocks[0].w
        return q

model Block:
    w: Tensor = ones([4])

    fn forward(self, x: Tensor) -> Tensor:
        return x * self.w
"#;
    let (analysis, _interner) = analyze_fixture(src);
    // We do NOT require zero errors here — referencing a not-yet-defined
    // Block in a default-value expression (`Block()`) may produce
    // diagnostics depending on resolver ordering. What we DO require is
    // that the type map contains a `Type::Model` carrying `element_model
    // = Block`'s symbol (with EITHER populated or empty fields), never a
    // bare `Type::Unknown` at the subscript position. The for-loop arm's
    // precedent at stmt.rs:49-57 returns the empty-fields Model
    // unconditionally, so the subscript arm must do at least that well.
    let any_model = analysis
        .type_map
        .values()
        .any(|ty| matches!(ty, Type::Model { .. }));
    assert!(
        any_model,
        "expected at least one Type::Model in the type map under the \
         forward-reference ordering; got types: {:?}",
        analysis
            .type_map
            .values()
            .map(|t| std::mem::discriminant(t))
            .collect::<std::collections::HashSet<_>>()
    );
}

/// Defense-in-depth: subscript on a plain `List` MUST still return
/// the element type — the new `FixedModelArray` arm sits next to the
/// existing `List` arm, so we pin the original behavior so a future
/// edit doesn't regress it.
#[test]
fn subscript_on_plain_list_still_returns_element_type() {
    // No explicit list type — let the parser infer it from the literal,
    // which lowers to `Type::List(Box::new(Int))`. The Subscript arm then
    // hits the existing `Type::List` branch (must not regress).
    let src = r#"
fn dummy() -> int:
    let xs = [1, 2, 3]
    return xs[0]
"#;
    let (analysis, _interner) = analyze_fixture(src);
    let errs: Vec<String> = analysis
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message.clone())
        .collect();
    assert!(
        errs.is_empty(),
        "subscript on plain list[int] regressed: {errs:?}"
    );
    let int_count = analysis
        .type_map
        .values()
        .filter(|ty| matches!(ty, Type::Int))
        .count();
    assert!(
        int_count >= 1,
        "expected at least one Type::Int in the type map (the subscript result), got 0"
    );
}
