#!/usr/bin/env awk -f
#
# gpu-gate-inventory.awk — classify every #[ignore] test in the tree.
#
# Emits one TSV row per ignored test:
#
#   file <TAB> fn <TAB> class <TAB> reason
#
# The CLASS decides whether scripts/gpu-cert.sh will execute the gate:
#
#   gpu           needs a CUDA device            -> RUN
#   toolchain     needs ptxas/cuobjdump only     -> RUN (toolchain, no device)
#   isolate       needs a device AND its own     -> RUN, one process, alone
#                 process (a faulting kernel
#                 poisons the CUDA context)
#   multiproc     spawns extra nsl processes     -> RUN in the slow tier
#   generator     REWRITES fixtures/baselines    -> NEVER auto-run
#   broken        documented-broken / deferred   -> NEVER auto-run
#   diagnostic    prints or sweeps; asserts      -> NEVER auto-run
#                 nothing meaningful
#   gpu-inferred  bare #[ignore] in a file that   -> RUN
#                 is `feature = "cuda"`-gated
#   cpu-stub      `#[cfg(not(feature = "cuda"))]` -> NEVER in the lane: the
#                 placeholder for non-cuda builds    cuda-featured build the
#                                                    lane uses compiles it OUT
#   unclassified  bare #[ignore], no reason, no   -> NOT run, but REPORTED
#                 cuda gating
#
# `unclassified` exists so that a gate can never be silently dropped from the
# lane: anything the ruleset does not recognise shows up in the report as
# not-run rather than vanishing. Ordering matters — DENY patterns are tested
# before ALLOW patterns, because several deny-worthy reasons also mention a
# GPU ("GPU required - c20 T1-followup lands the FFI plumbing" is a
# known-broken gate, not a runnable one).
#
# `gpu-inferred` covers the 82 bare `#[ignore]` attributes whose author wrote
# no reason. Every one of them lives in a file that either cfg-gates on the
# cuda feature or does not — and the four non-cuda files were checked by hand
# (a safetensors generator, a PTX dump, an FFI stub, a CPU diagnostic stub).
# Callers pass -v cuda_gated=1 when the file mentions `feature = "cuda"`.
#
# NAME/PATH DENY RULES RUN FIRST, BEFORE the empty-reason inference. Review
# finding F1: previously `r == ""` returned `gpu-inferred` immediately, so the
# ONLY thing keeping `gen_cpdt_precision_fp16_weights` (a bare `#[ignore]` that
# rewrites a committed .safetensors) out of the lane was that its file happens
# not to contain the text `feature = "cuda"`. The cuda_gated flag is a
# whole-file text grep, so a single doc comment mentioning the feature would
# have promoted a fixture writer into a RUN class. An unnamed generator must be
# refused on its NAME, not on a coincidence.

function classify(r, fname, path,    lower) {
    # A file whose whole purpose is generating a fixture. Narrow on purpose:
    # the suffix names the FILE's role, so it cannot catch a parity gate.
    if (path ~ /_gen\.rs$/)                            return "generator"

    # Name-based rules apply ONLY to a bare `#[ignore]` with no reason — the
    # F1 case. They must NOT override an explicit reason: `generate_` and
    # `regenerate_` also begin the names of real CFIE decode parity gates
    # (`generate_token_sequence_matches_cpu_reference`), and demoting those
    # out of the lane would silently delete coverage — the same defect as F1,
    # pointing the other way.
    if (r == "") {
        if (fname ~ /^(gen|generate|regenerate)_/)     return "generator"
        if (fname ~ /^dump_/)                          return "diagnostic"
        return cuda_gated ? "gpu-inferred" : "unclassified"
    }
    lower = tolower(r)

    # ---- DENY on reason (see note above) -------------------------------
    # Opt-in cargo features we do not build. Checked BEFORE the "manual
    # invocation" rule below, which would otherwise mislabel these as
    # fixture writers — a label that implies destructiveness they lack.
    if (lower ~ /requires --features/)                 return "broken"
    # Fixture/baseline writers. Running these under --ignored would silently
    # rewrite the very reference data other gates compare against.
    if (lower ~ /generator|regenerat/)                 return "generator"
    if (lower ~ /manual invocation/)                   return "generator"
    # Self-declared non-assertions.
    if (lower ~ /not a correctness gate/)              return "diagnostic"
    if (lower ~ /^diagnostic:/)                        return "diagnostic"
    if (lower ~ /debug helper/)                        return "diagnostic"
    # Documented-broken or blocked on unlanded work.
    if (lower ~ /not yet wired/)                       return "broken"
    if (lower ~ /pre-existing/)                        return "broken"
    if (lower ~ /prerequisite/)                        return "broken"
    if (lower ~ /honest partial/)                      return "broken"
    if (lower ~ /followup lands/)                      return "broken"
    # External inputs we will not fetch, and opt-in cargo features.
    if (lower ~ /fetched hf checkpoint/)               return "broken"
    if (lower ~ /requires --features/)                 return "broken"
    if (lower ~ /cfg\(test\) bss stub/)                return "broken"

    # ---- ALLOW --------------------------------------------------------
    if (lower ~ /run alone/)                           return "isolate"
    if (lower ~ /spawns/)                              return "multiproc"
    if (lower ~ /cuda gpu|cuda device|cuda-capable gpu|cuda toolchain \+ gpu/)
        return "gpu"
    if (lower ~ /sm_89/)                               return "gpu"
    # Needs nvcc/ptxas/cuobjdump or the CUDA link step, but no device.
    if (lower ~ /ptxas|cuobjdump|cuda toolchain/)      return "toolchain"

    return "unclassified"
}

# Track `#[cfg(not(feature = "cuda"))]` through the current attribute stack.
# The lane builds WITH the cuda feature, so a test under this cfg is compiled
# OUT of every binary the lane can run — classifying it into a RUN class
# guarantees a permanent NOTFOUND (the two placeholder stubs
# `wrga_b32_trigger_measurement_requires_cuda` and
# `fp8_dispatcher::cuda_feature_required` sat exactly there). The flag is
# position-independent within the stack (the cfg may come before or after
# `#[ignore]`) and is cleared by the first line that is not an attribute,
# comment, or blank — so a cfg-gated item elsewhere in the file cannot leak
# onto a later, genuinely runnable gate (that would silently DELETE coverage,
# the F1 defect pointing the other way).
# Block-comment tracker: lines inside a multi-line /* ... */ are neither
# item boundaries (they must not clear the cfg-not flag, whatever character
# they start with) nor gate text (a code sample mentioning `#[ignore]`
# inside a block comment is prose, not a gate — the `//`-comment guard in
# the ignore rule never covered this).

# CRLF proofing (review finding, 2026-07-30): on a checkout with CRLF
# working-tree files, every `$0` ends in \r, so the multi-line-attribute
# continuation regex (`\\[ \t]*$`) never matches — the reason string is
# silently dropped and a `broken`-quarantined gate reclassifies to
# `gpu-inferred`, un-quarantining a documented-segfaulting test in the
# regenerated manifest. Strip the CR before any rule sees the line.
{ sub(/\r$/, "") }

in_block_comment {
    if ($0 ~ /\*\//) in_block_comment = 0
    next
}
/^[ \t]*\/\*/ && !/\*\// {
    in_block_comment = 1
    next
}

# Liberal spacing on purpose: `#[cfg( not( feature = "cuda" ) )]` is legal
# and gpu-cert.sh's whole-file grep would still set cuda_gated=1 for the
# file, so a spacing variant this regex missed would classify a bare-ignore
# stub `gpu-inferred` — a NOTFOUND. No `next`: the same line may also carry
# the `#[ignore]` (one-line attribute stacks are inside the scanner's
# support envelope since review finding F4), and swallowing it here made
# the stub VANISH from the inventory — the exact silent drop the header
# forbids. A pure cfg line falls through harmlessly: it matches no other
# rule, and the boundary rule exempts attribute lines.
/^[ \t]*#[ \t]*\[[ \t]*cfg\([ \t]*not\([ \t]*feature[ \t]*=[ \t]*"cuda"[ \t]*\)[ \t]*\)[ \t]*\]/ {
    in_not_cuda_stack = 1
}

# Accumulate a (possibly line-continued) #[ignore ...] attribute.
#
# NOT anchored to start-of-line (review finding F4): `#[test] #[ignore = "..."]`
# on ONE line is valid Rust and was previously invisible to the scanner — a gate
# outside the lane AND outside the drift gate. Comment lines are excluded so the
# prose that discusses `#[ignore]` in doc comments is not mistaken for a gate.
/#\[ignore/ {
    trimmed = $0
    sub(/^[ \t]+/, "", trimmed)
    if (trimmed ~ /^\/\//) next

    attr = $0
    # Continue ONLY while the line ends in a backslash — the actual Rust
    # continuation marker (review finding F3). The previous loop continued
    # until a line ended in `]`, so an `#[ignore]` carrying a trailing
    # `// comment` ate the following `fn` line and kept eating; the real gate
    # VANISHED and its class landed on some later, non-ignored function.
    # `--write-manifest` would then bake that loss in and the drift gate would
    # defend the wrong inventory forever.
    while (attr ~ /\\[ \t]*$/ && (getline nextline) > 0) {
        # getline bypasses the main-rule CR strip — without this, a CRLF
        # checkout re-introduces \r here, the continuation regex stops
        # matching after ONE joined line, and the reason silently truncates
        # to empty (the same corruption the top-of-file strip fixes).
        sub(/\r$/, "", nextline)
        sub(/[ \t]*\\[ \t]*$/, "", attr)
        sub(/^[ \t]+/, " ", nextline)
        attr = attr nextline
    }

    # `#[ignore = "..."] #[cfg(not(feature = "cuda"))]` on ONE line: the
    # start-anchored cfg rule above cannot see a cfg attribute that follows
    # the ignore on the same line (or on a swallowed continuation line), so
    # the accumulated attribute text is checked here too. Without this the
    # one-line form of the fp8_dispatcher stub classified `gpu` — the
    # permanent NOTFOUND this class exists to kill (review finding).
    if (attr ~ /#[ \t]*\[[ \t]*cfg\([ \t]*not\([ \t]*feature[ \t]*=[ \t]*"cuda"[ \t]*\)[ \t]*\)[ \t]*\]/)
        in_not_cuda_stack = 1

    # The reason runs from the first `"` to the LAST `"]` — the attribute's
    # own terminator. Keying on the last bare quote instead would swallow a
    # trailing comment that happens to contain one.
    reason = ""
    fq = index(attr, "\"")
    if (fq > 0) {
        last = 0; start = 1
        while ((i = index(substr(attr, start), "\"]")) > 0) {
            last = start + i - 1
            start = last + 2
        }
        if (last > fq) reason = substr(attr, fq + 1, last - fq - 1)
    }
    gsub(/[ \t]+/, " ", reason)
    pending = 1
    next
}

# The first fn after the attribute is the test it gates. The cfg-not check
# overrides the reason-based classifier: `cuda_feature_required`'s reason
# ("...require the cuda feature + sm_89+") matches the `gpu` ALLOW rule, but
# no cuda-featured binary contains the symbol.
pending && /^[ \t]*(pub[ \t]+)?(async[ \t]+)?fn[ \t]+[A-Za-z0-9_]+/ {
    line = $0
    sub(/^[ \t]*(pub[ \t]+)?(async[ \t]+)?fn[ \t]+/, "", line)
    sub(/[^A-Za-z0-9_].*$/, "", line)
    cls = in_not_cuda_stack ? "cpu-stub" : classify(reason, line, FILENAME)
    printf "%s\t%s\t%s\t%s\n", FILENAME, line, cls, reason
    pending = 0
    reason = ""
}

# Attribute-stack boundary: any line that is not an attribute, comment
# (line or block), or blank ends the stack (the fn-emission rule above has
# already consumed the flag by the time this runs on the same line).
#
# KNOWN LIMIT — item-level cfg only. A `#[cfg(not(feature = "cuda"))]` on a
# `mod stubs { ... }` of ignored tests is defeated here: the `mod` opener
# clears the flag and every test inside classifies by its reason —
# a visible NOTFOUND, not silent coverage loss, which is the direction this
# rule deliberately errs in. No in-tree stub is mod-scoped today (all 7
# cfg-not sites are item-level, checked 2026-07-28); supporting the mod
# form needs brace tracking, which this scanner should grow only when such
# a stub actually appears rather than carrying half-a-parser on spec.
!/^[ \t]*#[ \t]*\[/ && !/^[ \t]*\/\// && !/^[ \t]*\/\*/ && !/^[ \t]*\*/ && !/^[ \t]*$/ {
    in_not_cuda_stack = 0
}
