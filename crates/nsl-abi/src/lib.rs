//! `nsl-abi` — a machine-checkable source of truth for the runtime C-ABI.
//!
//! # The problem this closes
//!
//! The codegen declares every runtime function it emits calls to in a
//! `RUNTIME_FUNCTIONS*` table under `nsl-codegen/src` (today a single one in
//! `builtins.rs`) — a table of *Cranelift*
//! signatures `(name, &[param types], Option<ret type>)`. The runtime
//! *implements* those functions as `#[unsafe(no_mangle)] extern "C" fn`s in
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

/// Map a C type as written in a generated header to an [`AbiScalar`].
///
/// Pointers of any spelling become `Int(64)` — same rule as [`abi_from_rust`],
/// same reason. `void` used as a return type is *absence* of a value, not a
/// type, so it is handled by the caller and returns `None` here.
///
/// This exists because the emitted header is the one host-facing surface the
/// `RUNTIME_FUNCTIONS` cross-check structurally cannot see: that check iterates
/// the declared table, and the header is hand-written text in `c_header.rs`
/// with no table entry at all.
pub fn abi_from_c(ty: &str) -> Option<AbiScalar> {
    let mut t = ty.trim();
    // Qualifiers and tags carry no ABI meaning.
    for prefix in ["const ", "volatile ", "struct ", "enum ", "union "] {
        while let Some(rest) = t.strip_prefix(prefix) {
            t = rest.trim_start();
        }
    }
    let t = t.trim_end();
    if t.ends_with('*') {
        return Some(AbiScalar::Int(64));
    }
    Some(match t {
        "int64_t" | "uint64_t" | "size_t" | "ssize_t" | "ptrdiff_t" | "intptr_t"
        | "uintptr_t" | "long long" | "unsigned long long" => AbiScalar::Int(64),
        "int32_t" | "uint32_t" | "int" | "unsigned" | "unsigned int" => AbiScalar::Int(32),
        "int16_t" | "uint16_t" | "short" | "unsigned short" => AbiScalar::Int(16),
        "int8_t" | "uint8_t" | "char" | "signed char" | "unsigned char" | "_Bool" | "bool" => {
            AbiScalar::Int(8)
        }
        "double" => AbiScalar::Float(64),
        "float" => AbiScalar::Float(32),
        _ => return None,
    })
}

fn classify_c(ty: &str) -> ParsedType {
    match abi_from_c(ty) {
        Some(s) => ParsedType::Known(s),
        None => ParsedType::Unknown(ty.trim().to_string()),
    }
}

/// Replace every balanced `{ … }` region with `;`, so a declaration that
/// follows a function or struct body still lands in its own `;`-delimited
/// chunk.
///
/// A SEMICOLON, not a space: a function definition carries no trailing `;`, so
/// blanking its body would glue `static inline T f(…)` to whatever declaration
/// comes next, and the merged chunk parses as `f` alone — the following
/// declaration disappears exactly as it did before the elision was added.
///
/// The `extern "C" {` linkage block is dropped first, and that is not a
/// detail: it wraps EVERY declaration in the header, so counting its brace
/// elides the entire file and the gate silently checks nothing. Its now
/// unmatched closing brace falls out harmlessly — a `}` at depth 0 is simply
/// not copied.
fn elide_brace_regions(src: &str) -> String {
    let src = src.replace("extern \"C\" {", " ");
    let mut out = String::with_capacity(src.len());
    let mut depth = 0usize;
    for c in src.chars() {
        match c {
            '{' => {
                if depth == 0 {
                    out.push(';');
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
            }
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Strip `/* … */` and `// …` from C text, preserving token separation.
///
/// Both helpers it delegates to document preconditions written for the
/// `RUNTIME_FUNCTIONS` table ("contains no string literals embedding `/*`").
/// A generated C header satisfies them for a different reason: `c_header::emit`
/// writes no `//` comments at all, and its only string literals are the export
/// names inside `nsl_model_call(...)` bodies, which `elide_brace_regions`
/// removes before any of this is parsed.
fn strip_c_comments(src: &str) -> String {
    let no_block = strip_block_comments(src);
    no_block
        .lines()
        .map(strip_line_comment)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split off the trailing parameter NAME from a C parameter declaration,
/// leaving the type. `int64_t n_inputs` -> `int64_t`; `const NslTensorDesc*
/// inputs` -> `const NslTensorDesc*`; `int64_t` and `int64_t*` (unnamed) are
/// returned unchanged.
fn c_param_type(param: &str) -> String {
    let p = param.trim();
    if p.ends_with('*') {
        return p.to_string();
    }
    let last = match p.rsplit(|c: char| c.is_whitespace() || c == '*').next() {
        Some(l) if !l.is_empty() => l,
        _ => return p.to_string(),
    };
    // A single-token param IS the type (`void`, `int64_t`).
    if last == p {
        return p.to_string();
    }
    // Multi-word type spellings whose last token is still part of the type.
    if matches!(last, "int" | "char" | "long" | "short" | "double" | "float" | "unsigned") {
        return p.to_string();
    }
    p[..p.len() - last.len()].trim_end().to_string()
}

/// Parse the function declarations and function-pointer typedefs out of a
/// generated C header.
///
/// Deliberately narrow: it understands the two forms `c_header::emit` produces
/// (`RET name(params);` and `typedef RET (*Name)(params);`).
///
/// Brace-enclosed regions — struct bodies and `static inline` function bodies —
/// are ELIDED before splitting, not used as a skip condition. Skipping any
/// `;`-chunk containing a brace looked equivalent and was not: a declaration
/// following a function body shares that body's closing brace in its chunk, so
/// it was silently dropped. `emit_static_inline_wrappers` runs last, which
/// makes "after the inlines" exactly where a new lifecycle prototype would
/// naturally be added — the one place the parser could not see. (Eliding also
/// means a `static inline` DEFINITION now parses as a declaration, which is
/// correct: its declarator is well-formed and worth checking.)
///
/// Callers should assert on the names they expect rather than a count: a
/// parser that silently degrades makes its gate vacuous, and a floor set just
/// under the current value still lets one declaration disappear.
pub fn parse_c_prototypes(header: &str) -> Vec<FnSig> {
    let src = strip_c_comments(header);
    // Preprocessor lines are not declarations and do not end in `;`.
    let src: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let src = elide_brace_regions(&src);

    let mut out = Vec::new();
    for stmt in src.split(';') {
        let s = stmt.trim();
        if s.is_empty() || !s.contains('(') {
            continue;
        }
        let is_typedef = s.starts_with("typedef");
        let body = if is_typedef {
            s["typedef".len()..].trim_start()
        } else {
            s
        };

        let (name, ret_txt, params_txt) = if is_typedef {
            // `RET (*Name)(params)` — the first group names the pointer, the
            // second holds the parameters.
            let (ptr_group, after) = match balanced(body, 0, '(', ')') {
                Some(x) => x,
                None => continue,
            };
            let name = ptr_group.trim().trim_start_matches('*').trim().to_string();
            let (params, _) = match balanced(body, after, '(', ')') {
                Some(x) => x,
                None => continue,
            };
            let open = match body.find('(') {
                Some(i) => i,
                None => continue,
            };
            (name, body[..open].trim().to_string(), params.to_string())
        } else {
            let (params, _) = match balanced(body, 0, '(', ')') {
                Some(x) => x,
                None => continue,
            };
            let open = match body.find('(') {
                Some(i) => i,
                None => continue,
            };
            let head = body[..open].trim();
            let name = match head.rsplit(|c: char| c.is_whitespace() || c == '*').next() {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            let ret = head[..head.len() - name.len()].trim().to_string();
            (name, ret, params.to_string())
        };

        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }

        let params: Vec<ParsedType> = split_top_level(&params_txt)
            .iter()
            .map(|p| c_param_type(p))
            .filter(|t| t != "void")
            .map(|t| classify_c(&t))
            .collect();
        let ret = if ret_txt.trim() == "void" {
            None
        } else {
            Some(classify_c(&ret_txt))
        };

        out.push(FnSig {
            name,
            params,
            ret,
            source: "generated C header".to_string(),
        });
    }
    out
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

/// Locate every `const RUNTIME_FUNCTIONS*` declaration in a comment-stripped
/// source text, returning the byte offset and spelling of each name.
///
/// `const` ONLY. A registry spelled `static RUNTIME_FUNCTIONS`, or built by a
/// `LazyLock`, is invisible here and contributes nothing — silently. The
/// `tables_parsed >= 1` assertion in `signature_agreement` catches the total
/// case (a registry that moved wholesale), and the truncation floor catches a
/// large partial one, but a single table converted to `static` alongside
/// others would go unnoticed. Keep the registry `const`.
///
/// Anchoring on the `const` KEYWORD — rather than on the bare identifier — is
/// what makes it safe to look past the first hit. `builtins.rs` mentions
/// `RUNTIME_FUNCTIONS` several times that are not declarations (the `for &(..)
/// in RUNTIME_FUNCTIONS` loop, `use super::RUNTIME_FUNCTIONS`, and the tests
/// that iterate it); anchoring on the identifier would find those, take the
/// next `=` after each, and parse an unrelated balanced `[...]` as though it
/// were a table.
fn runtime_table_decl_offsets(src: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    for (i, _) in src.match_indices("const ") {
        // Reject `const` as a suffix of a longer identifier (`MY_const `).
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                continue;
            }
        }
        let name_start = i + "const ".len();
        let rest = &src[name_start..];
        let name_end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        if rest[..name_end].starts_with("RUNTIME_FUNCTIONS") {
            out.push((name_start, rest[..name_end].to_string()));
        }
    }
    out
}

/// Parse every `RUNTIME_FUNCTIONS*` table (the codegen's *declared*
/// signatures) out of the text of one source file.
///
/// More than one table per file — and more than one file — is supported so the
/// registry can be grouped by domain without the gate going quiet. A parser
/// that read only the first table would, after such a split, return a short
/// list; that is caught by the truncation floor in `signature_agreement`, but
/// only as a confusing failure rather than as correct behaviour.
pub fn parse_runtime_functions_table(src_raw: &str) -> Vec<FnSig> {
    parse_runtime_functions_table_in_file(src_raw, "builtins.rs").0
}

/// As [`parse_runtime_functions_table`], but labelling each signature with the
/// file it came from so a mismatch report names the file to edit.
///
/// Returns `(signatures, declarations_seen, declarations_parsed)`. The two
/// counts exist because every failure path in [`parse_one_table`] is SILENT:
/// a missing `=`, an initializer that is not a slice literal, an unbalanced
/// `[...]` all yield "no entries" rather than an error. A caller that only
/// looks at the signatures cannot distinguish "this file declares no table"
/// from "this file declares a table I failed to read", and the second is a
/// parser regression that should be loud. When the counts disagree, something
/// that looks like a table was not read.
///
/// NOTE: string literals are not stripped (only comments are), so a Rust
/// source file containing a `const RUNTIME_FUNCTIONS…` inside a string —
/// a test fixture for this very parser, say — is read as a real table. The
/// crates scanned by [`check_workspace`] contain no such fixture; this
/// crate's own tests are safe only because `nsl-abi/src` is not scanned.
pub fn parse_runtime_functions_table_in_file(
    src_raw: &str,
    label: &str,
) -> (Vec<FnSig>, usize, usize) {
    let mut out = Vec::new();
    // Cheap reject: `check_workspace` reads every `.rs` in two crates
    // (331 files, ~11 MB), and comment-stripping copies each one twice.
    if !src_raw.contains("RUNTIME_FUNCTIONS") {
        return (out, 0, 0);
    }
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
    let decls = runtime_table_decl_offsets(src);
    let seen = decls.len();
    let mut parsed = 0usize;
    for (const_pos, const_name) in decls {
        if parse_one_table(src, const_pos, label, &const_name, &mut out) {
            parsed += 1;
        }
    }
    (out, seen, parsed)
}

/// Parse the single table whose name begins at `const_pos`, appending its
/// entries to `out`.
fn parse_one_table(
    src: &str,
    const_pos: usize,
    label: &str,
    const_name: &str,
    out: &mut Vec<FnSig>,
) -> bool {
    // Anchor on the assignment `=` first: the `[` between `RUNTIME_FUNCTIONS`
    // and `=` belongs to the TYPE annotation (`&[(&str, &[types::Type], ...)]`),
    // not the value. The table body is the balanced `[...]` after `= &`.
    let Some(eq_rel) = src[const_pos..].find('=') else {
        return false;
    };
    let after_eq = const_pos + eq_rel + 1;

    // The initializer's `[` must come before this statement's `;`.
    //
    // Without that bound, a sibling const whose value is NOT a slice —
    // `const RUNTIME_FUNCTIONS_COUNT: usize = 682;`, an associated const with
    // no initializer, a `&str` — finds no `[` of its own, and `balanced`
    // happily scans past the semicolon to the NEXT table's body and parses
    // all of its entries a second time. Every one would then be cross-checked
    // twice, a real mismatch reported twice under the wrong const name, and
    // the parsed total inflated — which loosens the truncation floor instead
    // of tripping it. Adding a `_COUNT` beside a split registry is an
    // ordinary thing to do, so this is a live trap rather than a theoretical
    // one.
    let semi = src[after_eq..]
        .find(';')
        .map(|i| after_eq + i)
        .unwrap_or(src.len());
    let Some(open) = src[after_eq..].find('[').map(|i| after_eq + i) else {
        return false;
    };
    if open > semi {
        return false;
    }
    let Some((body, _)) = balanced(src, open, '[', ']') else {
        return false;
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
                    source: format!("{label}::{const_name}"),
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
            source: format!("{label}::{const_name}"),
        });
    }
    true
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
    /// The same name is declared by more than one table entry.
    ///
    /// Cranelift accepts a repeat `declare_function` when the signature
    /// matches, so a duplicate is invisible at build time; `builtins.rs`
    /// then does `fns.insert(name, ...)` into a `HashMap`, so two entries
    /// that DISAGREE resolve last-wins by declaration order. Detected here
    /// rather than only inside one table, because a name duplicated ACROSS
    /// two tables in two files is exactly what splitting the registry risks
    /// and is the case a per-table check structurally cannot see.
    DuplicateDecl,
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
    /// How many `const RUNTIME_FUNCTIONS*` declarations were spotted.
    pub tables_found: usize,
    /// How many of those were successfully parsed into entries. Fewer than
    /// `tables_found` means a declaration was recognised but not read — a
    /// parser regression, which is otherwise silent.
    pub tables_parsed: usize,
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

    // Duplicate declarations, across every table and file.
    let mut seen: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for d in declared {
        seen.entry(d.name.as_str()).or_default().push(d.source.as_str());
    }
    for (name, sources) in &seen {
        if sources.len() > 1 {
            report.mismatches.push(Mismatch {
                name: (*name).to_string(),
                kind: MismatchKind::DuplicateDecl,
                detail: format!("declared {} times: {}", sources.len(), sources.join(", ")),
            });
        }
    }

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
/// every `.rs` under `nsl-codegen/src` and `nsl-runtime/src`, plus the inplace
/// macro.
///
/// The declared side is found by SCANNING rather than by a hard-coded path.
/// It was `crates/nsl-codegen/src/builtins.rs` alone, which made the gate's
/// coverage depend on the registry staying in one file: moving or splitting it
/// would have left this reading a path that no longer held the table, and the
/// only thing standing between that and a vacuously-green ABI check was the
/// truncation floor in `signature_agreement`. Scanning costs one directory walk
/// of a crate whose sibling is already walked the same way.
pub fn check_workspace(workspace_root: &Path) -> std::io::Result<Report> {
    let codegen_src = workspace_root.join("crates/nsl-codegen/src");
    let mut declared = Vec::new();
    // Sorted so the report is stable across filesystems: `rust_files` walks in
    // `read_dir` order, which is not ordered on ext4/btrfs.
    let mut codegen_files = rust_files(&codegen_src)?;
    codegen_files.sort();
    let mut tables_found = 0usize;
    let mut tables_parsed = 0usize;
    for path in codegen_files {
        let text = std::fs::read_to_string(&path)?;
        let label = path
            .strip_prefix(workspace_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let (sigs, seen, parsed) = parse_runtime_functions_table_in_file(&text, &label);
        declared.extend(sigs);
        tables_found += seen;
        tables_parsed += parsed;
    }

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
    let mut report = cross_check(&declared, &impls, &macro_impls);
    report.tables_found = tables_found;
    report.tables_parsed = tables_parsed;
    Ok(report)
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

    /// The registry may be grouped into several tables — the point of parsing
    /// every `const RUNTIME_FUNCTIONS*` rather than only the first.
    #[test]
    fn parses_every_table_in_a_file_not_just_the_first() {
        let src = r#"
        const RUNTIME_FUNCTIONS_SCALAR: &[(&str, &[types::Type], Option<types::Type>)] = &[
            ("nsl_pow_int", &[types::I64, types::I64], Some(types::I64)),
        ];

        pub(crate) const RUNTIME_FUNCTIONS_TENSOR: &[(&str, &[types::Type], Option<types::Type>)] = &[
            ("nsl_tensor_ones", &[types::I64], Some(types::I64)),
            ("nsl_tensor_free", &[types::I64], None),
        ];
        "#;
        let sigs = parse_runtime_functions_table(src);
        let names: Vec<String> = sigs.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, ["nsl_pow_int", "nsl_tensor_ones", "nsl_tensor_free"]);
    }

    /// Only DECLARATIONS are tables. `builtins.rs` names `RUNTIME_FUNCTIONS`
    /// in a `for` loop, a `use`, and its own tests; anchoring on the bare
    /// identifier would take the next `=` after each of those and parse an
    /// unrelated balanced `[...]` as a table, inventing entries.
    #[test]
    fn ignores_mentions_that_are_not_declarations() {
        let src = r#"
        use super::RUNTIME_FUNCTIONS;

        const RUNTIME_FUNCTIONS: &[(&str, &[types::Type], Option<types::Type>)] = &[
            ("nsl_alloc", &[types::I64], Some(types::I64)),
        ];

        pub fn declare(module: &mut ObjectModule) {
            for &(name, params, ret) in RUNTIME_FUNCTIONS {
                let sig = ["not", "a", "table"];
                let _ = (name, params, ret, sig);
            }
        }
        "#;
        let sigs = parse_runtime_functions_table(src);
        assert_eq!(
            sigs.len(),
            1,
            "expected only the declaration to parse, got {:?}",
            sigs.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        assert_eq!(sigs[0].name, "nsl_alloc");
    }

    /// A sibling const whose value is not a slice must not consume the next
    /// table's body.
    ///
    /// Without bounding the initializer search at the statement's `;`, the
    /// `_COUNT` const below finds no `[` of its own and `balanced` runs on to
    /// the next table, parsing all of its entries a SECOND time — every one
    /// cross-checked twice, mismatches attributed to the wrong const, and the
    /// parsed total inflated so the truncation floor gets more slack rather
    /// than tripping.
    #[test]
    fn a_non_slice_sibling_const_does_not_steal_the_next_table() {
        let src = r#"
        const RUNTIME_FUNCTIONS_COUNT: usize = 682;

        const RUNTIME_FUNCTIONS_TENSOR: &[(&str, &[types::Type], Option<types::Type>)] = &[
            ("nsl_tensor_ones", &[types::I64], Some(types::I64)),
            ("nsl_tensor_free", &[types::I64], None),
        ];
        "#;
        let (sigs, seen, parsed) = parse_runtime_functions_table_in_file(src, "probe.rs");
        let names: Vec<String> = sigs.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, ["nsl_tensor_ones", "nsl_tensor_free"]);
        assert_eq!(seen, 2, "both consts look like declarations");
        assert_eq!(parsed, 1, "only one of them is a table");
    }

    /// A recognised declaration that cannot be read is counted, so the caller
    /// can tell "no table here" from "a table I failed to parse".
    #[test]
    fn an_unreadable_declaration_is_counted_but_not_parsed() {
        let src = "const RUNTIME_FUNCTIONS_DOC: &str = \"no table here\";";
        let (sigs, seen, parsed) = parse_runtime_functions_table_in_file(src, "probe.rs");
        assert!(sigs.is_empty());
        assert_eq!((seen, parsed), (1, 0));
    }

    /// `const` as the tail of a longer identifier is not the keyword.
    #[test]
    fn does_not_match_const_inside_an_identifier() {
        let src = r#"
        let MY_const RUNTIME_FUNCTIONS_FAKE = ["nope"];
        "#;
        assert!(parse_runtime_functions_table(src).is_empty());
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

    /// The two ways this parser silently degraded to checking NOTHING, both
    /// found the hard way.
    #[test]
    fn c_prototype_parsing_survives_the_shapes_a_real_header_has() {
        let header = r#"
#ifndef NSL_M_H
#define NSL_M_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NslModel NslModel;
typedef struct {
    void*    data;
    int32_t  ndim;
} NslTensorDesc;

typedef int64_t (*NslExportFn)(NslModel* model, const NslTensorDesc* inputs,
                               int64_t n_inputs);

int64_t   nsl_abi_version(void);
int64_t   nsl_model_call(NslModel* model, const char* name, int64_t n);

static inline int64_t nsl_export_forward(NslModel* model, int64_t n) {
    return nsl_model_call(model, "forward", n);
}

int64_t   nsl_added_after_the_inline(NslModel* model);

#ifdef __cplusplus
}
#endif
#endif
"#;
        let sigs = parse_c_prototypes(header);
        let names: Vec<&str> = sigs.iter().map(|s| s.name.as_str()).collect();

        // 1. `extern "C" {` wraps EVERY declaration. Counting its brace as a
        //    region elided the entire file — the gate stayed green while
        //    checking zero prototypes.
        assert!(
            names.contains(&"nsl_abi_version") && names.contains(&"nsl_model_call"),
            "the extern \"C\" linkage block swallowed the declarations: {names:?}"
        );

        // 2. A declaration AFTER a function body shares that body's closing
        //    brace in its `;`-chunk. Skipping brace-bearing chunks dropped it,
        //    and the inline wrappers are emitted LAST — so "after the inlines"
        //    is exactly where a new lifecycle prototype would be added.
        assert!(
            names.contains(&"nsl_added_after_the_inline"),
            "a declaration following a function body was dropped: {names:?}"
        );

        // 3. The struct body must not become a bogus signature.
        assert!(
            !names.contains(&"NslTensorDesc"),
            "a struct definition parsed as a function: {names:?}"
        );

        // 4. The function-pointer typedef keeps its own name and widths.
        let ef = sigs.iter().find(|s| s.name == "NslExportFn").expect("typedef");
        assert_eq!(ef.ret, Some(ParsedType::Known(AbiScalar::Int(64))));
        assert_eq!(
            ef.params,
            vec![
                ParsedType::Known(AbiScalar::Int(64)),
                ParsedType::Known(AbiScalar::Int(64)),
                ParsedType::Known(AbiScalar::Int(64)),
            ]
        );

        // 5. `void` is absence, not a parameter.
        let v = sigs.iter().find(|s| s.name == "nsl_abi_version").unwrap();
        assert!(v.params.is_empty(), "`(void)` produced a parameter: {:?}", v.params);
    }
}
