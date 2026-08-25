#!/usr/bin/env bash
#
# check-doc-agreement.sh
#
# Turn documentation drift into a build failure (roadmap item 21).
#
# Companion to check-version-agreement.sh: that script pins version strings,
# this one pins claims about the CODE. Every check here was written against a
# drift that was actually found in the tree on 2026-07-24, so none of them is
# hypothetical:
#
#   (a) docs/wiki/Optimization-Passes.md stated "CCR — Common-kernel
#       Combination Rewriting ... No implementation file exists" while
#       crates/nsl-codegen/src/ccr.rs was 2027 lines, wired into stmt.rs, and
#       driven by --checkpoint-blocks. The doc had the pass's NAME wrong too
#       (it is Compiler-Chosen Recomputation). The wiki page was last touched
#       2026-07-02; ccr.rs landed 2026-07-15 — the doc simply predated the
#       implementation and nothing forced a revisit.  -> checks 1 and 2
#
#   (b) .github/workflows/ci.yml carried a `merge-gate-preview` job described
#       as existing "only during PR 3's review window", to be replaced by a
#       required `merge-gate` job. That was ~12 weeks stale, the replacement
#       never landed, and the job's stated premise (that the test it runs is
#       #[ignore]d) had become false.  -> check 4
#
#   (c) README.md said "All benchmarks run on CPU" while models/benchmarks/
#       held RTX 5070 Ti certification curves at 50M/500M/1B.  -> check 3
#
# Checks:
#   1. Every repo-relative markdown link in the checked docs resolves.
#   2. No doc claims a pass is unimplemented while its source file exists.
#   3. Docs that describe benchmark scope must acknowledge the GPU results
#      that exist on disk.
#   4. Every `cargo ...` command documented in the Testing-Strategy CI table is
#      actually run by ci.yml.
#   5. Every CI job id named in the docs exists in .github/workflows/.
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

fail=0
err() { printf '  \xE2\x9C\x97 %s\n' "$1" >&2; fail=1; }
ok() { printf '  \xE2\x9C\x93 %s\n' "$1"; }

# STATUS.md and spec/ were added by item 19 (2026-08-25): both had drifted for
# months — STATUS's review horizon predated seven shipped roadmap items, and
# spec/ documented an open train-config namespace, LLVM compilation, and five
# subcommands that do not exist — precisely because no gate ever read them.
DOCS=(README.md SPECIFICATION.md STATUS.md)
while IFS= read -r f; do DOCS+=("$f"); done < <(find docs/wiki spec -name '*.md' | sort)

# ---------------------------------------------------------------------------
# 1. Repo-relative markdown links must resolve.
#
# The CCR section literally ended with "Do not link to a source file that does
# not exist" — a rule with no enforcement. This is the enforcement.
# Skips external URLs, pure anchors, and mailto:.
# ---------------------------------------------------------------------------
echo "1. Repo-relative markdown links resolve"
broken=0
checked=0
for doc in "${DOCS[@]}"; do
  doc_dir="$(dirname "${doc}")"
  # Extract the target of every ](...) link.
  while IFS= read -r target; do
    [[ -z "${target}" ]] && continue
    # Markdown permits <angle-bracket> targets, which is how paths containing
    # spaces are written (e.g. `<../research/CPDT Research.pdf>`).
    target="${target#<}"
    target="${target%>}"
    # Skip anchors, mailto:, and ANY scheme:// URL (http, https, chrome, ...).
    case "${target}" in
      '#'*|mailto:*) continue ;;
    esac
    if [[ "${target}" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*:// ]]; then
      continue
    fi
    # Strip any #anchor.
    path="${target%%#*}"
    [[ -z "${path}" ]] && continue
    checked=$((checked + 1))
    if [[ ! -e "${doc_dir}/${path}" ]]; then
      err "${doc}: link target does not exist: ${target}"
      broken=$((broken + 1))
    fi
  done < <(grep -oE '\]\([^)]+\)' "${doc}" | sed 's/^](//; s/)$//' || true)
done
[[ "${broken}" -eq 0 ]] && ok "all ${checked} repo-relative links resolve"

# ---------------------------------------------------------------------------
# 2. No doc may call a pass unimplemented while its source exists.
#
# Table: <pass label> <one source file that proves it is implemented>
# Add a row when a pass gains an implementation file.
# ---------------------------------------------------------------------------
echo
echo "2. No doc claims an implemented pass is unimplemented"
PASS_SOURCES=(
  "CCR:crates/nsl-codegen/src/ccr.rs"
  "FASE:crates/nsl-codegen/src/fase.rs"
  "WGGO:crates/nsl-codegen/src/wggo.rs"
  "CSHA:crates/nsl-codegen/src/csha.rs"
  "WRGA:crates/nsl-codegen/src/wrga.rs"
  "CPDT:crates/nsl-codegen/src/cpdt.rs"
  "CEP:crates/nsl-codegen/src/cep.rs"
  "CPKD:crates/nsl-codegen/src/cpkd.rs"
  "CFIE:crates/nsl-codegen/src/cfie.rs"
  "PCA:crates/nsl-codegen/src/pca_detect.rs"
  "MemoryPlanner:crates/nsl-codegen/src/memory_planner.rs"
)
# Phrases that assert absence. Matched case-insensitively on the same line as
# the pass label, or on the 3 lines following it (docs put the claim under the
# heading).
ABSENCE_RE='no implementation|not implemented|unimplemented|no implementation file|planned future pass|does not exist yet'

for row in "${PASS_SOURCES[@]}"; do
  label="${row%%:*}"
  src="${row#*:}"
  if [[ ! -e "${src}" ]]; then
    # The pass genuinely has no implementation — the table row is the stale
    # thing. Say so loudly rather than silently skipping.
    err "table row '${label}' points at a missing source file: ${src} (update PASS_SOURCES in this script)"
    continue
  fi
  hit=0
  for doc in "${DOCS[@]}"; do
    if grep -n -A 3 -E "(^|[^A-Za-z])${label}([^A-Za-z]|$)" "${doc}" 2>/dev/null \
        | grep -qiE "${ABSENCE_RE}"; then
      err "${doc}: asserts ${label} is unimplemented, but ${src} exists"
      hit=1
    fi
  done
  [[ "${hit}" -eq 0 ]] || true
done
ok "checked ${#PASS_SOURCES[@]} passes against absence claims"

# ---------------------------------------------------------------------------
# 3. Benchmark-scope claims must acknowledge on-disk GPU results.
# ---------------------------------------------------------------------------
echo
echo "3. Benchmark-scope claims match what is on disk"
gpu_bench_count="$(find models/benchmarks -name '*.md' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${gpu_bench_count}" -gt 0 ]]; then
  # An unqualified "all benchmarks ... CPU" is the drift; a scoped claim
  # ("the tables on this page are CPU measurements") is fine.
  if grep -qiE '^[^|]*all benchmarks (run|are)[^.]*\bCPU\b' README.md; then
    err "README.md claims ALL benchmarks are CPU, but ${gpu_bench_count} GPU benchmark documents exist under models/benchmarks/"
  else
    ok "README.md does not make an unqualified all-CPU benchmark claim (${gpu_bench_count} GPU docs on disk)"
  fi
else
  ok "no GPU benchmark documents on disk — nothing to reconcile"
fi

# ---------------------------------------------------------------------------
# 4. The documented CI command table must match what ci.yml actually runs.
#
# docs/wiki/Testing-Strategy.md hand-mirrors ci.yml's steps as a table of
# `cargo ...` commands. A first attempt at this check scanned the docs for
# anything shaped like a job id and asserted it existed in the workflows —
# which matched prose (a design-doc FILENAME, the YAML key `continue-on-error`)
# and produced pure noise. Checking the COMMANDS is precise: each is a literal
# string that either appears in ci.yml or does not.
# ---------------------------------------------------------------------------
echo
echo "4. Documented CI commands match .github/workflows/ci.yml"
strategy_doc="docs/wiki/Testing-Strategy.md"
missing_cmds=0
if [[ -e "${strategy_doc}" && -e .github/workflows/ci.yml ]]; then
  ci_text="$(tr -s '[:space:]' ' ' < .github/workflows/ci.yml)"
  n_cmds=0
  while IFS= read -r cmd; do
    [[ -z "${cmd}" ]] && continue
    n_cmds=$((n_cmds + 1))
    # Normalise whitespace the same way so a line-wrapped YAML command matches.
    norm="$(tr -s '[:space:]' ' ' <<< "${cmd}")"
    norm="${norm#"${norm%%[![:space:]]*}"}"
    norm="${norm%"${norm##*[![:space:]]}"}"
    if [[ "${ci_text}" != *"${norm}"* ]]; then
      err "${strategy_doc} documents CI command '${norm}', which ci.yml does not run"
      missing_cmds=$((missing_cmds + 1))
    fi
    # Scope: ONLY the "### CI" section's table. The doc also documents
    # developer-local commands (`cargo insta review`, `cargo insta accept`)
    # that CI is not expected to run — scanning the whole file flagged those.
  done < <(awk '/^### CI$/{inci=1; next} inci && /^#{2,3} /{inci=0} inci' "${strategy_doc}" \
           | grep -oE '`cargo [^`]+`' | tr -d '`' | sort -u || true)
  [[ "${missing_cmds}" -eq 0 ]] && ok "all ${n_cmds} documented cargo commands appear in ci.yml"
else
  ok "no Testing-Strategy CI table to reconcile"
fi

# ---------------------------------------------------------------------------
# 5. CI job ids named in the docs must exist in .github/workflows/.
#
# Roadmap item 21 asked for this; check 4 above covers COMMANDS, which is a
# different surface. The docs name jobs directly too — CONTRIBUTING.md calls
# out `test-onnx-rt` and `fpga` as blocking, Testing-Strategy.md names
# `doc-agreement` and `gpu-gate-inventory`. A renamed or deleted job leaves
# those references pointing at nothing, which is exactly how the
# `merge-gate-preview` reference outlived the job it described.
#
# Scope: backticked kebab-case tokens on lines that also mention CI or a job,
# minus RUNNER LABELS (`ubuntu-latest`, `macos-14`) and crate names, which have
# the same shape as a job id and are what a first cut of this check flagged.
#
# KNOWN GAP, stated rather than papered over: only KEBAB-CASE tokens (>=1
# hyphen) are validated. A single-word job id such as `fpga` is
# indistinguishable from ordinary backticked prose, so renaming THAT job would
# not be caught here. Widening the pattern flags every inline code span in the
# docs, which is worse than the gap.
# checked against the job ids parsed from the workflows. Restricting
# to those lines is what keeps this precise — an earlier attempt at a job-id
# check scanned freely and matched prose and YAML keys, producing pure noise.
# ---------------------------------------------------------------------------
echo
echo "5. CI job ids named in the docs exist in .github/workflows/"
# Only keys under the top-level `jobs:` mapping are job ids. A blanket
# two-space-key match also picks up `push`, `pull_request`, `contents` and the
# other `on:`/`permissions:` keys — which is precisely the "matched YAML keys"
# noise the previous attempt at this check ran into.
job_ids="$(awk '/^jobs:/{inj=1; next} /^[a-z]/{inj=0} inj && /^  [a-z0-9_-]+:/{sub(/:.*/,""); sub(/^  /,""); print}' \
           .github/workflows/*.yml 2>/dev/null | sort -u)"
if [[ -z "${job_ids}" ]]; then
  err "no job ids parsed out of .github/workflows/*.yml — this check would be vacuous"
else
  n_jobs="$(wc -l <<< "${job_ids}" | tr -d ' ')"
  bad_jobs=0
  n_refs=0
  for doc in CONTRIBUTING.md "${DOCS[@]}"; do
    [[ -e "${doc}" ]] || continue
    while IFS= read -r tok; do
      [[ -z "${tok}" ]] && continue
      n_refs=$((n_refs + 1))
      if ! grep -qx -- "${tok}" <<< "${job_ids}"; then
        err "${doc} names CI job '${tok}', which no workflow defines"
        bad_jobs=$((bad_jobs + 1))
      fi
    done < <(grep -iE -A 1 '(\bCI\b|\bjobs?\b)' "${doc}" \
             | grep -oE '`[a-z0-9]+(-[a-z0-9]+)+`' | tr -d '`' \
             | grep -vE '^(ubuntu|macos|windows)-' \
             | grep -vE '^(cargo|continue-on-error|rust-toolchain|check-doc-agreement|check-version-agreement|gpu-cert|pull-request|merge-queue|nsl-cli|nsl-codegen|nsl-runtime|nsl-semantic|test-threads|include-ignored)' \
             | sort -u || true)
  done
  # Only tokens that ARE job ids should be asserted; anything else shaped like
  # one but absent is reported above. Guard against the filter eating
  # everything, which would make this pass while checking nothing.
  # The docs name at least four jobs today (test-onnx-rt, doc-agreement,
  # gpu-gate-inventory, ...). A threshold of 0 would still read green after the
  # filter silently ate all but one reference, so pin it near the real count.
  if [[ "${n_refs}" -lt 3 ]]; then
    err "check 5 matched only ${n_refs} doc job reference(s) — the filter is too aggressive and this check is near-vacuous"
  elif [[ "${bad_jobs}" -eq 0 ]]; then
    ok "all ${n_refs} documented CI job references resolve (${n_jobs} jobs defined)"
  fi
fi

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "Documentation agreement check FAILED — reconcile the drift above." >&2
  echo "Each failure is a doc that contradicts the tree; fix the doc, or fix" >&2
  echo "the table in scripts/check-doc-agreement.sh if the code moved." >&2
  exit 1
fi
echo "Documentation agreement check passed."
