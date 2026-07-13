//! `nsl-abi` — a machine-checkable source of truth for the runtime C-ABI.
//!
//! # The problem this closes
//!
//! The codegen declares every runtime function it emits calls to in
//! `nsl-codegen/src/builtins.rs::RUNTIME_FUNCTIONS` — a table of *Cranelift*
//! signatures `(name, &[param types], Option<ret type>)`. The runtime
//! *implements* those functions as `#[no_mangle] extern "C" fn`s in
//! `nsl-runtime`. **The two are linked by symbol name only** — the Rust
//! compiler never checks that the declared arity and types match the
//! implementation. So a drift — a parameter added on one side but not the
//! other, an `f64` where the table says `I64` (passed in the wrong register
//! class), or an implementation that was renamed/removed — compiles cleanly and
//! only manifests as a stack-corrupting runtime crash or silent miscompile.
//!
//! This crate parses both surfaces and cross-checks them, so that drift is
//! caught by a test rather than at runtime. It is the *validator* first
//! increment of the broader "single ABI source of truth" effort; the same
//! normalized model ([`FnSig`]/[`AbiScalar`]) can later be extended to the
//! generated C headers and to *generating* the declarations instead of
//! validating them.
//!
//! It is deliberately dependency-free and parses source *text* (it does not
//! link against the codegen or runtime), so it stays a cheap standalone gate.

use std::collections::BTreeMap;
use std::path::Path;

/// A normalized ABI scalar: how a value is actually passed at the C-ABI level.
///
/// Both the Cranelift declaration and the Rust `extern "C"` implementation are
/// lowered to this so they can be compared exactly. Pointers and integer
/// handles collapse to `Int(64)` (they share the general-purpose register
/// class), but floats stay distinct by width: swapping `f64` for `i64` is a
/// real calling-convention bug (xmm vs gp register), so `Float(64)` and
/// `Int(64)` must NOT compare equal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AbiScalar {
    /// Integer / pointer passed in a general-purpose register. Width in bits.
    Int(u16),
    /// Floating point passed in a vector register. Width in bits.
    Float(u16),
}

/// A parsed parameter or return type: either a recognized [`AbiScalar`] or an
/// unrecognized type token we preserve verbatim (so the checker can treat it as
/// "cannot verify" rather than silently guessing, and surface it).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ParsedType {
    Known(AbiScalar),
    Unknown(String),
}

/// A normalized function signature parsed from either surface.
#[derive(Clone, Debug)]
pub struct FnSig {
    pub name: String,
    pub params: Vec<ParsedType>,
    /// `None` = no return value; `Some(ty)` = returns `ty`.
    pub ret: Option<ParsedType>,
    /// Where this signature was parsed from (for diagnostics).
    pub source: String,
}

/// Map a Cranelift `types::XXX` identifier (as written in the table) to an
/// [`AbiScalar`]. Returns `None` for identifiers we do not model (e.g. vector
/// or reference types), which the caller records as `Unknown`.
pub fn abi_from_cranelift(ident: &str) -> Option<AbiScalar> {
    Some(match ident {
        "I8" => AbiScalar::Int(8),
        "I16" => AbiScalar::Int(16),
        "I32" => AbiScalar::Int(32),
        "I64" => AbiScalar::Int(64),
        "F32" => AbiScalar::Float(32),
        "F64" => AbiScalar::Float(64),
        _ => return None,
    })
}

/// Map a Rust type as written in an `extern "C"` signature to an [`AbiScalar`].
/// Raw pointers of any kind become `Int(64)` (a machine word). Returns `None`
/// for types we do not model.
pub fn abi_from_rust(ty: &str) -> Option<AbiScalar> {
    let t = ty.trim();
    // Any raw pointer (`*mut T`, `*const T`, incl. `*mut c_void`) is a 64-bit
    // machine word == Cranelift I64 on the targets NSL supports.
    if t.starts_with('*') {
        return Some(AbiScalar::Int(64));
    }
    Some(match t {
        "i64" | "u64" | "usize" | "isize" => AbiScalar::Int(64),
        "i32" | "u32" => AbiScalar::Int(32),
        "i16" | "u16" => AbiScalar::Int(16),
        "i8" | "u8" | "bool" => AbiScalar::Int(8),
        "f64" => AbiScalar::Float(64),
        "f32" => AbiScalar::Float(32),
        _ => return None,
    })
}

fn classify_cranelift(ident: &str) -> ParsedType {
    match abi_from_cranelift(ident) {
        Some(s) => ParsedType::Known(s),
        None => ParsedType::Unknown(ident.to_string()),
    }
}

fn classify_rust(ty: &str) -> ParsedType {
    match abi_from_rust(ty) {
        Some(s) => ParsedType::Known(s),
        None => ParsedType::Unknown(ty.trim().to_string()),
    }
}

/// Strip `// line comments` from a single line (outside string literals). The
/// runtime/codegen sources do not embed `//` inside the string literals that
/// appear within signatures, so a naive strip is safe here and keeps the
/// signature parsers from tripping over trailing comments.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Remove `/* ... */` block comments, replacing each with a single space to
/// preserve token separation. UTF-8 safe (slices only at `find` boundaries).
/// Used only on the `RUNTIME_FUNCTIONS` table text, which contains no string
/// literals embedding `/*`, so this cannot corrupt a real signature.
fn strip_block_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        out.push(' ');
        match rest[start + 2..].find("*/") {
            Some(end) => rest = &rest[start + 2 + end + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Return the substring enclosed by the balanced delimiter pair starting at the
/// `open` character found at or after `from`. `open`/`close` are e.g. `('(',
/// ')')` or `('[', ']')`. Returns `(inner, index_after_close)`.
fn balanced(src: &str, from: usize, open: char, close: char) -> Option<(&str, usize)> {
    let bytes = src.as_bytes();
    let start = src[from..].find(open)? + from;
    let mut depth = 0usize;
    let mut i = start;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some((&src[start + 1..i], i + 1));
            }
        }
        i += 1;
    }
    None
}

/// Split a delimited list on top-level commas (commas not nested inside `()`,
/// `[]`, `<>`). Used for both the Cranelift `&[...]` type list and Rust param
/// lists.
fn split_top_level(list: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut prev = '\0';
    for c in list.chars() {
        match c {
            '(' | '[' | '<' => {
                depth += 1;
                cur.push(c);
            }
            // The `>` in a `->` return arrow (e.g. a `fn(i64) -> i64` param
            // type) is NOT a generic/bracket close — do not let it drive depth
            // negative, or every following top-level comma would be missed and
            // arity would collapse.
            '>' if prev == '-' => {
                cur.push(c);
            }
            ')' | ']' | '>' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
        prev = c;
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Parse the `RUNTIME_FUNCTIONS` table (the codegen's *declared* signatures)
/// out of the text of `builtins.rs`.
pub fn parse_runtime_functions_table(src_raw: &str) -> Vec<FnSig> {
    let mut out = Vec::new();
    // Strip comments first: the table's `&[...]` blocks carry inline comments
    // like `// q, k, v, out` whose COMMAS would otherwise be counted as extra
    // parameters by `split_top_level`. Block comments are stripped too as
    // defense in depth — a stray `]`/`)` inside one could otherwise truncate
    // `balanced` and silently drop the tail of the table.
    let no_block = strip_block_comments(src_raw);
    let src: String = no_block
        .lines()
        .map(strip_line_comment)
        .collect::<Vec<_>>()
        .join("\n");
    let src = src.as_str();
    let Some(const_pos) = src.find("RUNTIME_FUNCTIONS") else {
        return out;
    };
    // Anchor on the assignment `=` first: the `[` between `RUNTIME_FUNCTIONS`
    // and `=` belongs to the TYPE annotation (`&[(&str, &[types::Type], ...)]`),
    // not the value. The table body is the balanced `[...]` after `= &`.
    let Some(eq_rel) = src[const_pos..].find('=') else {
        return out;
    };
    let Some((body, _)) = balanced(src, const_pos + eq_rel, '[', ']') else {
        return out;
    };

    // Each entry is a top-level `( "name", &[..types..], <ret> )`.
    let mut i = 0;
    while let Some((entry, next)) = balanced(body, i, '(', ')') {
        i = next;
        // name: first double-quoted string literal in the entry.
        let name = entry
            .find('"')
            .and_then(|q| entry[q + 1..].find('"').map(|e| entry[q + 1..q + 1 + e].to_string()));
        let Some(name) = name else { continue };

        // params: the `&[ ... ]` type list. Extract balanced `[...]`.
        let params = match balanced(entry, 0, '[', ']') {
            Some((inner, after)) => {
                let params = split_top_level(inner)
                    .iter()
                    .filter_map(|tok| {
                        tok.rsplit("::")
                            .next()
                            .map(|id| classify_cranelift(id.trim()))
                    })
                    .collect::<Vec<_>>();
                // ret is after the params bracket.
                let tail = &entry[after..];
                let ret = parse_table_ret(tail);
                out.push(FnSig {
                    name,
                    params,
                    ret,
                    source: "builtins.rs::RUNTIME_FUNCTIONS".to_string(),
                });
                continue;
            }
            None => Vec::new(),
        };
        // Entry with no `&[...]` (shouldn't happen) — record with empty params.
        out.push(FnSig {
            name,
            params,
            ret: None,
            source: "builtins.rs::RUNTIME_FUNCTIONS".to_string(),
        });
    }
    out
}

/// Parse the return slot of a table entry tail (`, Some(types::I64) ),` or
/// `, None ),`).
fn parse_table_ret(tail: &str) -> Option<ParsedType> {
    if let Some(some_pos) = tail.find("Some(") {
        // Extract inside Some( ... ) and take the last `::`-segment.
        if let Some((inner, _)) = balanced(tail, some_pos, '(', ')') {
            let id = inner.rsplit("::").next().unwrap_or(inner).trim();
            return Some(classify_cranelift(id));
        }
    }
    // `None` (or nothing recognizable) => no return value.
    None
}

/// Parse every `extern "C" fn nsl_*` implementation out of the text of a single
/// source file. A name may appear more than once (distinct `#[cfg(...)]`
/// variants); the caller collects all of them.
pub fn parse_externs_in_file(src: &str, file_label: &str) -> Vec<FnSig> {
    let cleaned: String = src
        .lines()
        .map(strip_line_comment)
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = Vec::new();
    let needle = "extern \"C\" fn ";
    let mut search_from = 0;
    while let Some(rel) = cleaned[search_from..].find(needle) {
        let at = search_from + rel + needle.len();
        search_from = at;
        // Read the fn name identifier.
        let rest = &cleaned[at..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after_name = at + name.len();
        // Params: the balanced `( ... )` immediately following the name.
        let Some((param_str, after_params)) = balanced(&cleaned, after_name, '(', ')') else {
            continue;
        };
        let params = split_top_level(param_str)
            .iter()
            .map(|p| classify_rust(param_type(p)))
            .collect::<Vec<_>>();
        // Return: optional `-> Type` up to the first `{`, `where`, or `;`.
        let ret = parse_extern_ret(&cleaned[after_params..]);
        out.push(FnSig {
            name,
            params,
            ret,
            source: file_label.to_string(),
        });
    }
    out
}

/// Given a Rust parameter `pattern: type`, return just the `type` slice. Splits
/// on the first top-level `:` that is not part of a `::` path separator.
fn param_type(param: &str) -> &str {
    let bytes = param.as_bytes();
    let mut i = 0;
    let mut depth = 0i32;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            ':' if depth == 0 => {
                // Skip `::` path separators.
                if bytes.get(i + 1) == Some(&b':') {
                    i += 2;
                    continue;
                }
                return param[i + 1..].trim();
            }
            _ => {}
        }
        i += 1;
    }
    // No `:` (e.g. `self`, or a bare type) — return as-is.
    param.trim()
}

/// Parse the `-> Type` return slot after an extern fn's parameter list.
fn parse_extern_ret(after_params: &str) -> Option<ParsedType> {
    // Find the signature terminator so we don't wander into the body.
    let end = ["{", ";", "\n}"]
        .iter()
        .filter_map(|t| after_params.find(t))
        .min()
        .unwrap_or(after_params.len());
    let sig_tail = &after_params[..end];
    let arrow = sig_tail.find("->")?;
    let mut ret = sig_tail[arrow + 2..].trim();
    // Trim a trailing `where` clause if present.
    if let Some(w) = ret.find("where") {
        ret = ret[..w].trim();
    }
    if ret.is_empty() {
        return None;
    }
    Some(classify_rust(ret))
}

/// Parse the names generated by `define_inplace_unary!(<name>, ...)` in the
/// runtime. Each expands to `pub extern "C" fn <name>(ptr: i64) -> i64`, so we
/// register that fixed signature — otherwise these 9 ops look "missing" to a
/// text scan for `extern "C" fn`.
pub fn parse_inplace_unary_macro(src: &str) -> Vec<FnSig> {
    let mut out = Vec::new();
    let needle = "define_inplace_unary!";
    let mut from = 0;
    while let Some(rel) = src[from..].find(needle) {
        let at = from + rel + needle.len();
        from = at;
        let Some((inner, _)) = balanced(src, at, '(', ')') else {
            continue;
        };
        let name = split_top_level(inner)
            .first()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        out.push(FnSig {
            name,
            params: vec![ParsedType::Known(AbiScalar::Int(64))],
            ret: Some(ParsedType::Known(AbiScalar::Int(64))),
            source: "activation.rs::define_inplace_unary!".to_string(),
        });
    }
    out
}

/// A single detected disagreement between a declared signature and its
/// implementation(s).
#[derive(Clone, Debug)]
pub struct Mismatch {
    pub name: String,
    pub kind: MismatchKind,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MismatchKind {
    /// Declared in `RUNTIME_FUNCTIONS` but no `extern "C"` (or macro) impl found.
    MissingImpl,
    /// Impl found, but parameter count differs.
    ArityMismatch,
    /// Impl found with matching arity, but a param or return type differs.
    TypeMismatch,
}

/// Compare one declared signature against all implementation variants sharing
/// its name. Returns `None` if any variant matches; otherwise the best-effort
/// mismatch against the first variant. Unknown (unmodeled) types on either side
/// are treated as wildcards for that one position so an unparsed exotic type
/// does not produce a false positive — arity is still enforced.
fn compare(declared: &FnSig, impls: &[FnSig]) -> Option<Mismatch> {
    if impls.is_empty() {
        return Some(Mismatch {
            name: declared.name.clone(),
            kind: MismatchKind::MissingImpl,
            detail: format!(
                "declared in {} with {} param(s) but no runtime `extern \"C\" fn {}` found",
                declared.source,
                declared.params.len(),
                declared.name
            ),
        });
    }
    // EVERY variant must agree, not just one. The checker cannot see
    // `#[cfg(...)]`, so a symbol with several textual `extern "C" fn`
    // definitions — e.g. an interop stub gated `#[cfg(not(feature="interop"))]`
    // plus the real impl gated `#[cfg(feature="interop")]` — has exactly ONE of
    // them linked per build config, while the codegen emits a single call with
    // the table's signature. For that call to be correct in EVERY config, all
    // variants must match the table; flag if ANY variant disagrees.
    let bad: Vec<&FnSig> = impls
        .iter()
        .filter(|im| !sigs_compatible(declared, im))
        .collect();
    if bad.is_empty() {
        return None;
    }
    let im = bad[0];
    let variant_note = if impls.len() > 1 {
        format!(
            " ({} of {} same-name (cfg?) variants disagree)",
            bad.len(),
            impls.len()
        )
    } else {
        String::new()
    };
    if im.params.len() != declared.params.len() {
        return Some(Mismatch {
            name: declared.name.clone(),
            kind: MismatchKind::ArityMismatch,
            detail: format!(
                "declared {} param(s) in {} but impl in {} has {}{}",
                declared.params.len(),
                declared.source,
                im.source,
                im.params.len(),
                variant_note,
            ),
        });
    }
    Some(Mismatch {
        name: declared.name.clone(),
        kind: MismatchKind::TypeMismatch,
        detail: format!(
            "type disagreement (impl {}): declared {:?} ret {:?} vs impl {:?} ret {:?}{}",
            im.source, declared.params, declared.ret, im.params, im.ret, variant_note,
        ),
    })
}

/// Two positions are compatible if:
/// - both are `Known` and equal;
/// - both are `Unknown` and spelled identically; or
/// - exactly one is `Unknown` — treated as a wildcard so we never false-positive
///   on a type this validator does not model yet.
///
/// The single-`Unknown` wildcard is a deliberate blind spot: an unmodeled
/// runtime type (e.g. a struct passed/returned *by value*, which is an sret ABI
/// hazard) matched against a modeled `types::X` is accepted. The table side is
/// fully modeled today (only `I64/F64/I8/I32`), so this only relaxes checking
/// when the RUNTIME side uses an exotic type; extend [`abi_from_rust`] to
/// tighten it as the surface grows.
fn types_compatible(a: &ParsedType, b: &ParsedType) -> bool {
    match (a, b) {
        (ParsedType::Known(x), ParsedType::Known(y)) => x == y,
        (ParsedType::Unknown(x), ParsedType::Unknown(y)) => x == y,
        _ => true,
    }
}

fn sigs_compatible(declared: &FnSig, im: &FnSig) -> bool {
    if declared.params.len() != im.params.len() {
        return false;
    }
    if !declared
        .params
        .iter()
        .zip(&im.params)
        .all(|(a, b)| types_compatible(a, b))
    {
        return false;
    }
    match (&declared.ret, &im.ret) {
        (None, None) => true,
        (Some(a), Some(b)) => types_compatible(a, b),
        _ => false,
    }
}

/// The outcome of a full cross-check.
#[derive(Debug, Default)]
pub struct Report {
    /// Declared entries whose implementation matched exactly.
    pub verified: usize,
    /// Declared entries whose implementation was found only via the inplace
    /// macro (rather than a textual `extern "C" fn`).
    pub via_macro: usize,
    /// All detected disagreements.
    pub mismatches: Vec<Mismatch>,
}

/// Cross-check the declared table against the parsed implementations (textual
/// externs plus macro-generated ones).
pub fn cross_check(declared: &[FnSig], impls: &[FnSig], macro_impls: &[FnSig]) -> Report {
    let mut by_name: BTreeMap<&str, Vec<&FnSig>> = BTreeMap::new();
    for s in impls {
        by_name.entry(s.name.as_str()).or_default().push(s);
    }
    let mut macro_names: BTreeMap<&str, &FnSig> = BTreeMap::new();
    for s in macro_impls {
        macro_names.insert(s.name.as_str(), s);
    }

    let mut report = Report::default();
    for d in declared {
        let textual: Vec<FnSig> = by_name
            .get(d.name.as_str())
            .map(|v| v.iter().map(|&s| s.clone()).collect())
            .unwrap_or_default();
        if !textual.is_empty() {
            match compare(d, &textual) {
                None => report.verified += 1,
                Some(m) => report.mismatches.push(m),
            }
        } else if let Some(mac) = macro_names.get(d.name.as_str()) {
            match compare(d, std::slice::from_ref(mac)) {
                None => report.via_macro += 1,
                Some(m) => report.mismatches.push(m),
            }
        } else {
            report.mismatches.push(Mismatch {
                name: d.name.clone(),
                kind: MismatchKind::MissingImpl,
                detail: format!(
                    "declared in {} ({} param(s)) but no runtime impl found",
                    d.source,
                    d.params.len()
                ),
            });
        }
    }
    report
}

/// Convenience: run the whole check against a workspace root directory. Reads
/// `builtins.rs`, every `.rs` under `nsl-runtime/src`, and the inplace macro.
pub fn check_workspace(workspace_root: &Path) -> std::io::Result<Report> {
    let builtins = std::fs::read_to_string(
        workspace_root.join("crates/nsl-codegen/src/builtins.rs"),
    )?;
    let declared = parse_runtime_functions_table(&builtins);

    let runtime_src = workspace_root.join("crates/nsl-runtime/src");
    let mut impls = Vec::new();
    let mut macro_impls = Vec::new();
    for path in rust_files(&runtime_src)? {
        let text = std::fs::read_to_string(&path)?;
        let label = path
            .strip_prefix(workspace_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        impls.extend(parse_externs_in_file(&text, &label));
        if text.contains("define_inplace_unary!") {
            macro_impls.extend(parse_inplace_unary_macro(&text));
        }
    }
    Ok(cross_check(&declared, &impls, &macro_impls))
}

/// Recursively collect `.rs` files under a directory.
fn rust_files(dir: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_table_entry_with_mixed_types() {
        let src = r#"
        const RUNTIME_FUNCTIONS: &[(&str, &[types::Type], Option<types::Type>)] = &[
            (
                "nsl_tensor_mul_scalar",
                &[types::I64, types::F64, types::I8],
                Some(types::I64),
            ),
            (
                "nsl_free",
                &[types::I64],
                None,
            ),
        ];
        "#;
        let sigs = parse_runtime_functions_table(src);
        assert_eq!(sigs.len(), 2);
        assert_eq!(sigs[0].name, "nsl_tensor_mul_scalar");
        assert_eq!(
            sigs[0].params,
            vec![
                ParsedType::Known(AbiScalar::Int(64)),
                ParsedType::Known(AbiScalar::Float(64)),
                ParsedType::Known(AbiScalar::Int(8)),
            ]
        );
        assert_eq!(sigs[0].ret, Some(ParsedType::Known(AbiScalar::Int(64))));
        assert_eq!(sigs[1].name, "nsl_free");
        assert_eq!(sigs[1].ret, None);
    }

    #[test]
    fn parses_extern_fn_including_pointers_and_multiline() {
        let src = r#"
        #[no_mangle]
        pub extern "C" fn nsl_tensor_mul_scalar(
            tensor: i64,
            scalar: f64,
            in_place: u8,
        ) -> i64 {
            0
        }

        #[no_mangle]
        pub unsafe extern "C" fn nsl_write(ptr: *mut c_void, len: usize) {
        }
        "#;
        let sigs = parse_externs_in_file(src, "test.rs");
        assert_eq!(sigs.len(), 2);
        assert_eq!(sigs[0].name, "nsl_tensor_mul_scalar");
        assert_eq!(
            sigs[0].params,
            vec![
                ParsedType::Known(AbiScalar::Int(64)),
                ParsedType::Known(AbiScalar::Float(64)),
                ParsedType::Known(AbiScalar::Int(8)),
            ]
        );
        assert_eq!(sigs[0].ret, Some(ParsedType::Known(AbiScalar::Int(64))));
        // pointer -> Int(64); no return
        assert_eq!(sigs[1].name, "nsl_write");
        assert_eq!(
            sigs[1].params,
            vec![
                ParsedType::Known(AbiScalar::Int(64)),
                ParsedType::Known(AbiScalar::Int(64)),
            ]
        );
        assert_eq!(sigs[1].ret, None);
    }

    #[test]
    fn cross_check_flags_arity_and_type_and_missing() {
        let declared = vec![
            FnSig {
                name: "ok".into(),
                params: vec![ParsedType::Known(AbiScalar::Int(64))],
                ret: Some(ParsedType::Known(AbiScalar::Int(64))),
                source: "table".into(),
            },
            FnSig {
                name: "arity".into(),
                params: vec![ParsedType::Known(AbiScalar::Int(64))],
                ret: None,
                source: "table".into(),
            },
            FnSig {
                name: "typ".into(),
                params: vec![ParsedType::Known(AbiScalar::Int(64))],
                ret: None,
                source: "table".into(),
            },
            FnSig {
                name: "gone".into(),
                params: vec![],
                ret: None,
                source: "table".into(),
            },
        ];
        let impls = vec![
            FnSig {
                name: "ok".into(),
                params: vec![ParsedType::Known(AbiScalar::Int(64))],
                ret: Some(ParsedType::Known(AbiScalar::Int(64))),
                source: "rt".into(),
            },
            FnSig {
                name: "arity".into(),
                params: vec![
                    ParsedType::Known(AbiScalar::Int(64)),
                    ParsedType::Known(AbiScalar::Int(64)),
                ],
                ret: None,
                source: "rt".into(),
            },
            FnSig {
                name: "typ".into(),
                params: vec![ParsedType::Known(AbiScalar::Float(64))],
                ret: None,
                source: "rt".into(),
            },
        ];
        let report = cross_check(&declared, &impls, &[]);
        assert_eq!(report.verified, 1);
        let kinds: BTreeMap<&str, &MismatchKind> =
            report.mismatches.iter().map(|m| (m.name.as_str(), &m.kind)).collect();
        assert_eq!(kinds.get("arity"), Some(&&MismatchKind::ArityMismatch));
        assert_eq!(kinds.get("typ"), Some(&&MismatchKind::TypeMismatch));
        assert_eq!(kinds.get("gone"), Some(&&MismatchKind::MissingImpl));
    }

    #[test]
    fn every_same_name_variant_must_agree_not_just_one() {
        // Models a cfg-split symbol: an interop stub (3 params) plus the real
        // impl (4 params), table declares 4. Accepting because ONE variant
        // matches would miss that the default-build stub drifted (Finding 1).
        let declared = vec![FnSig {
            name: "nsl_x".into(),
            params: vec![ParsedType::Known(AbiScalar::Int(64)); 4],
            ret: Some(ParsedType::Known(AbiScalar::Int(64))),
            source: "table".into(),
        }];
        let impls = vec![
            FnSig {
                name: "nsl_x".into(),
                params: vec![ParsedType::Known(AbiScalar::Int(64)); 4],
                ret: Some(ParsedType::Known(AbiScalar::Int(64))),
                source: "real.rs".into(),
            },
            FnSig {
                name: "nsl_x".into(),
                params: vec![ParsedType::Known(AbiScalar::Int(64)); 3],
                ret: Some(ParsedType::Known(AbiScalar::Int(64))),
                source: "stub.rs".into(),
            },
        ];
        let report = cross_check(&declared, &impls, &[]);
        assert_eq!(report.verified, 0, "must NOT pass when a variant disagrees");
        assert_eq!(report.mismatches.len(), 1);
        assert_eq!(report.mismatches[0].kind, MismatchKind::ArityMismatch);
    }

    #[test]
    fn fn_pointer_param_does_not_collapse_arity() {
        // A `-> i64` return arrow inside a fn-pointer param must not drive the
        // top-level comma splitter's depth negative (Finding 2).
        let src = r#"
        pub extern "C" fn nsl_cb(cb: extern "C" fn(i64) -> i64, tensor: i64, n: i64) -> i64 {
            0
        }
        "#;
        let sigs = parse_externs_in_file(src, "t.rs");
        assert_eq!(sigs.len(), 1, "only the outer fn is an extern def");
        assert_eq!(
            sigs[0].params.len(),
            3,
            "fn-pointer param must not collapse the comma split, got {:?}",
            sigs[0].params
        );
    }

    #[test]
    fn unknown_types_are_wildcards_but_arity_still_checked() {
        let declared = vec![FnSig {
            name: "x".into(),
            params: vec![ParsedType::Unknown("V128".into())],
            ret: None,
            source: "table".into(),
        }];
        let impls = vec![FnSig {
            name: "x".into(),
            params: vec![ParsedType::Known(AbiScalar::Int(64))],
            ret: None,
            source: "rt".into(),
        }];
        // Unknown declared param is a wildcard -> compatible.
        assert!(cross_check(&declared, &impls, &[]).mismatches.is_empty());
    }

    #[test]
    fn inplace_macro_names_are_registered_with_fixed_sig() {
        let src = r#"
        define_inplace_unary!(nsl_tensor_relu_inplace, |v: f32| v, |v: f64| v, PTX, "k\0");
        define_inplace_unary!(nsl_tensor_exp_inplace, |v: f32| v, |v: f64| v, PTX, "k\0");
        "#;
        let macros = parse_inplace_unary_macro(src);
        assert_eq!(macros.len(), 2);
        assert_eq!(macros[0].name, "nsl_tensor_relu_inplace");
        assert_eq!(macros[0].params, vec![ParsedType::Known(AbiScalar::Int(64))]);
        assert_eq!(macros[0].ret, Some(ParsedType::Known(AbiScalar::Int(64))));
    }
}
