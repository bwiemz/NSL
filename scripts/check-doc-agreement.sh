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
#   4. Every CI job name quoted in the docs exists in .github/workflows/.
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

fail=0
err() { printf '  \xE2\x9C\x97 %s\n' "$1" >&2; fail=1; }
ok() { printf '  \xE2\x9C\x93 %s\n' "$1"; }

DOCS=(README.md SPECIFICATION.md)
while IFS= read -r f; do DOCS+=("$f"); done < <(find docs/wiki -name '*.md' | sort)

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

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "Documentation agreement check FAILED — reconcile the drift above." >&2
  echo "Each failure is a doc that contradicts the tree; fix the doc, or fix" >&2
  echo "the table in scripts/check-doc-agreement.sh if the code moved." >&2
  exit 1
fi
echo "Documentation agreement check passed."
