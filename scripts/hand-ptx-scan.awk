#!/usr/bin/env awk -f
#
# hand-ptx-scan.awk — count the lines of one Rust source file that put PTX
# text into a string literal.
#
# Roadmap item A2 froze the set of files that may emit PTX by hand (see
# scripts/hand-ptx-freeze.sh). This scanner decides membership. It does NOT
# grep the file: a grep sees `ld.global` in a doc comment, in a `//` note, in
# a `#[cfg(test)]` assertion on some other module's output, and reports each
# as a kernel. Instead it tokenizes just enough Rust to know when it is
# inside a string literal — `"..."` with escapes, `b"..."`, and multi-line
# `r"..."` / `r#"..."#` raw strings — and applies the PTX patterns to the
# string CONTENT only. `//` and nested `/* */` comments outside strings are
# dropped, char literals (`'"'`, `'\''`) are stepped over so they cannot open
# or close a string, and every item under a `#[cfg(test)]` / `#[test]`
# attribute is skipped — brace-for-brace for a `{ … }` item, up to the `;`
# for a brace-less one — so a test that asserts
# `ptx.contains("mma.sync.aligned")` on KIR output, or a `#[cfg(test)]`
# fixture constant, does not make its file a hand-PTX emitter.
#
# A line counts when the string text it carries looks like PTX:
#
#   D  a directive:      `.reg .f32 %f<8>;`  `.visible .entry k(`  `.shared .align 16 .b8`
#   I  an instruction:   `mul.lo.u32 %r1, %r2, 4;`  `st.global.f32 [%rd1], %f2;`
#                        mnemonic, at least one typed/modifier suffix from a
#                        fixed PTX vocabulary, then an operand (`%reg`, `[addr]`,
#                        a `{}` format hole, an immediate, or a label)
#   M  a bare mnemonic that takes no operand or is unmistakable on its own:
#                        `mma.sync.aligned`  `wgmma.`  `cp.async.commit_group`
#                        `ldmatrix.sync`  `bar.sync 0`  `ld.param.`  `@%p1 bra`
#   S  a whole-line statement that is also an English word: `ret;` `exit;`
#                        `bra LOOP;` — only with nothing but whitespace before it
#
# ...and when that text is an emitted LINE rather than a fragment: it carries
# a `;`, a `\n` escape, a `{}` format hole, or comes from inside a multi-line
# raw string. A PTX consumer's `ptx.contains(".target sm_90")` or
# `line.starts_with(".shared")` names a directive but emits nothing, and does
# not count.
#
# Output: the number of counted lines, on one line. With `-v show=1` each
# counted line is printed instead, as `<file>:<line>: <string text>`, so a
# new member can be explained.
#
# Written against POSIX awk (no interval expressions, no `^` inside groups):
# the CI runner's awk is mawk, the developer's is gawk.
#
# Usage: awk -f scripts/hand-ptx-scan.awk [-v show=1] <file.rs>

BEGIN {
    # Tokenizer state.
    in_str = 0        # inside "..." (or b"..."); Rust strings may span lines
    in_raw = 0        # inside r"..." / r#"..."#
    raw_hashes = 0    #   number of # the raw string was opened with
    in_block = 0      # nesting depth of /* */ comments
    # #[cfg(test)] skipping: `pending` is set by the attribute and resolved by
    # the next `{` (one more level of skip — depth 1 from production code,
    # deeper for a `#[test]` inside an already-skipped `mod tests`), `;`
    # (a brace-less item) or `}` (a `,`-terminated field, variant or arm
    # whose enclosing block closed); either way the attribute is spent.
    # String text is collected
    # only while neither is in force, so the brace-less item's own string
    # (`#[cfg(test)] const K: &str = "…";`) is not seen either.
    pending = 0
    skip_depth = 0
    count = 0

    typeset = "pred|[bsu](8|16|32|64)|f(16|32|64)|f16x2|bf16|bf16x2|tf32|e4m3|e5m2" \
              "|lo|hi|wide|rn|rz|rm|rp|rni|rzi|rmi|rpi|sat|ftz|approx|full|aligned" \
              "|shared|global|local|param|const|nc|sync|to|v2|v4|x2|x4|cta|gpu|sys|uni" \
              "|commit_group|wait_group|wait_all"

    # A directive is a dotted keyword followed by what PTX puts after it — a
    # second dotted keyword (`.reg .f32`, `.shared .align`), a number, an
    # arch, an entry name and its parameter list. Prose such as "the static
    # .shared footprint" has neither and does not count.
    pat_directive = "\\.(reg|shared|extern|visible|param|global|const|local)" \
                    " +\\.(align|pred|entry|shared|global|const|param|[bfsu][0-9]+|v[24])" \
                    "|\\.(align|version|address_size|maxntid|reqntid|minnctapersm) +[0-9{]" \
                    "|\\.target +sm_|\\.entry +[^ ]+ *\\(|\\.pragma +(\\\\)?\""
    # Matched against the text with a space prepended, so "not preceded by an
    # identifier char" needs no `^` alternative.
    pat_instr = "[^a-zA-Z0-9_.][a-z][a-z0-9]+(\\.[a-z0-9{}]+)*\\.(" typeset ")(\\.[a-z0-9{}]+)*" \
                " +(\\[?[%{]|[0-9]|[A-Z$_-])"
    # ...and the operand-less ones (`membar.gl;`, `fence.acq_rel.gpu;`),
    # which the I pattern cannot see.
    pat_bare = "mma\\.sync\\.aligned|wgmma\\.|cp\\.async\\.|ldmatrix\\.sync" \
               "|bar(rier)?\\.(sync|arrive)|(ld|st)\\.(volatile\\.)?(global|shared|param|local|const)\\." \
               "|\\.visible +\\.entry|\\.reg +\\.|@%p[0-9a-z_]* +[a-z]" \
               "|membar\\.(gl|cta|sys);|fence\\.(acq_rel|sc|proxy)"
    # `ret;`, `exit;` and `bra LABEL;` are English too ("cleared on exit;"),
    # so they count only as a whole statement: nothing but whitespace before
    # them on the line (the text is matched with a space prepended).
    pat_stmt = "^ +(ret;|exit;|bra +[A-Za-z_$%])"
}

function is_ptx(s) {
    s = " " s
    return s ~ pat_directive || s ~ pat_instr || s ~ pat_bare || s ~ pat_stmt
}

# A line of PTX, not a fragment: ends an instruction (`;`), ends a line
# (`\n` escape), is a line of a raw string, or is a format string with a
# hole (`.global .align 1 .b8 {}[{}] = {{` — a data initializer's opener,
# which a consumer would never write).
function emitted_line(s) {
    return s ~ /;/ || s ~ /\\n/ || s ~ /\n/ || s ~ /\{[^{}]*\}/
}

function ident_char(c) {
    return c ~ /[A-Za-z0-9_]/
}

function hashes(k,    s) {
    s = ""
    while (k-- > 0) s = s "#"
    return s
}

{
    line = $0
    n = length(line)
    text = ""          # string content seen on this line
    i = 1
    if (!in_str && !in_raw && !in_block && line ~ /^[ \t]*#\[(cfg\((all\()?test[,)]|test\])/) {
        pending = 1
        # Resume after the attribute: `#[cfg(test)] mod t;` on one line.
        i = index(line, "]") + 1
    }
    while (i <= n) {
        c = substr(line, i, 1)
        nx = substr(line, i + 1, 1)
        prev = (i > 1) ? substr(line, i - 1, 1) : ""
        live = (!pending && skip_depth == 0)
        if (in_raw) {
            # Close on `"` followed by exactly raw_hashes `#`.
            if (c == "\"" && substr(line, i + 1, raw_hashes) == hashes(raw_hashes)) {
                in_raw = 0
                i += 1 + raw_hashes
                if (live) text = text " "
                continue
            }
            if (live) text = text c
            i++
            continue
        }
        if (in_str) {
            if (c == "\\") { if (live) text = text c nx; i += 2; continue }
            if (c == "\"") { in_str = 0; if (live) text = text " "; i++; continue }
            if (live) text = text c
            i++
            continue
        }
        if (in_block) {
            if (c == "/" && nx == "*") { in_block++; i += 2; continue }
            if (c == "*" && nx == "/") { in_block--; i += 2; continue }
            i++
            continue
        }
        # Normal code.
        if (c == "/" && nx == "/") break
        if (c == "/" && nx == "*") { in_block = 1; i += 2; continue }
        if (c == "'") {
            # Char literal or lifetime. `'\...'` is always a char literal;
            # `'x'` is one; anything else is a lifetime/label — step over
            # the quote alone.
            if (nx == "\\") {
                # `'\''`: the escaped char sits at i+2, the closer at or
                # after i+3 — search from there or land on the closer.
                j = index(substr(line, i + 3), "'")
                i += (j > 0) ? j + 3 : 3
                continue
            }
            if (substr(line, i + 2, 1) == "'") { i += 3; continue }
            i++
            continue
        }
        if (c == "\"") {
            in_str = 1
            i++
            continue
        }
        if (c == "b" && nx == "r" && !ident_char(prev)) {
            # `br"..."` / `br#"..."#`: the `r` branch below sees `b` as a
            # preceding identifier char, so step onto the `r` with the byte
            # prefix consumed.
            i++
            prev = ""
            c = "r"
            nx = substr(line, i + 1, 1)
        }
        if (c == "r" && (nx == "\"" || nx == "#") && !ident_char(prev)) {
            # Raw string opener: r, then #*, then ". Count the hashes.
            h = 0
            while (substr(line, i + 1 + h, 1) == "#") h++
            if (substr(line, i + 1 + h, 1) == "\"") {
                in_raw = 1
                raw_hashes = h
                i += 2 + h
                continue
            }
        }
        if (pending) {
            if (c == "{") { pending = 0; skip_depth++; i++; continue }
            if (c == ";") { pending = 0; i++; continue }
            # A `}` while pending: the attribute sat on a `,`-terminated
            # item (a field, a variant, a match arm) and its enclosing
            # block just closed — spend it, or the next fn is swallowed.
            if (c == "}") pending = 0
        }
        if (skip_depth > 0) {
            if (c == "{") skip_depth++
            if (c == "}") skip_depth--
        }
        i++
    }
    # A raw string that runs past the end of the line contributed a whole
    # emitted line; mark it with a real newline (normal-string escapes stay
    # as the two characters `\n`).
    if (in_raw && !pending && skip_depth == 0) text = text "\n"
    if (text != "" && is_ptx(text) && emitted_line(text)) {
        count++
        if (show) {
            gsub(/\n/, " ", text)
            printf "%s:%d: %s\n", FILENAME, FNR, text
        }
    }
}

END {
    if (!show) print count
}
