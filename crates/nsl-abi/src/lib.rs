//! `nsl-abi` — the source of truth for the runtime C-ABI.
//!
//! # The problem this closes
//!
//! The codegen declares every runtime function it emits calls to as a
//! *Cranelift* signature `(name, &[param types], Option<ret type>)`, and the
//! runtime *implements* those functions as `#[unsafe(no_mangle)] extern "C"
//! fn`s in `nsl-runtime`. Linked by symbol name only, the two would drift
//! silently — a parameter added on one side but not the other, an `f64`
//! where the declaration says `I64` (passed in the wrong register class), or
//! an implementation that was renamed/removed compiles cleanly and only
//! manifests as a stack-corrupting runtime crash or silent miscompile.
//!
//! Since roadmap A3 there is one copy of each signature, here, as data:
//!
//! * [`table`] — the runtime functions the codegen calls
//!   (`for_each_runtime_fn!`, [`RUNTIME_ABI`]), rendered into the codegen's
//!   Cranelift declarations and into compile-time assertions in the runtime
//!   (`abi_check.rs`, via [`typed`]) that `rustc` checks against each
//!   implementation. A row that disagrees with its implementation fails the
//!   runtime's build naming the function.
//! * [`capi`] — the host-facing C API (`for_each_capi_fn!`), rendered into
//!   the generated C header's prototypes, the same runtime assertions and
//!   the `ctypes` mirror `nsl abi python` writes for `nslpy`.
//! * [`wire`] — the *wire constants* both crates must agree on: dtype tags,
//!   the tensor header's data offset, the plan bits, the allocation-surface
//!   tags, the ABI version, and the shared record formats. Each crate reads
//!   them from here (instead of the compiler importing them from the
//!   runtime), which is what lets the compiler stop depending on the
//!   runtime's dependency tree.
//!
//! The crate is deliberately dependency-free so both sides can build it
//! cheaply. What remains of the original text validator is the small
//! C-prototype parser ([`parse_c_prototypes`]) and the normalized
//! signature model ([`FnSig`]/[`AbiScalar`]) the `c_header_agreement` gate
//! uses to read the generated header back; the Rust-side text parser and
//! its `cross_check` were deleted once the typed assertions covered every
//! row (A3 step 6).

pub mod wire;
pub mod capi;
pub mod table;
pub mod typed;
pub use table::{FnDecl, RUNTIME_ABI};


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
#[cfg(test)]
mod tests {
    use super::*;

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
