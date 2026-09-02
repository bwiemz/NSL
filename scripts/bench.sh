#!/usr/bin/env bash
#
# bench.sh — the CPU benchmark lane (roadmap T3).
#
# Six criterion benches, one per stage of the compile pipeline plus the
# runtime's host-side hot paths. Each bench's own doc comment says what its
# corpus is and which figure to compare:
#
#   package        bench      measures
#   nsl-lexer      lex        bytes/s over stdlib/ + examples/ + models/
#   nsl-parser     parse      tokens/s over the same corpus, pre-tokenized
#   nsl-semantic   analyze    single-module analysis of the import-free examples
#   nsl-cli        frontend   load_all_modules over the three coder recipes
#   nsl-codegen    compile    compile_module over the clean import-free examples
#   nsl-runtime    tape       tape record+backward, CPU f32 matmul, alloc churn
#
# Usage:
#   scripts/bench.sh                        run all six, compare with the last run
#   scripts/bench.sh save NAME              run, store the results as baseline NAME
#   scripts/bench.sh compare NAME           run, compare against baseline NAME
#   scripts/bench.sh ... -- <criterion args>
#                                           e.g. `-- --quick`, `-- --noplot`,
#                                           `-- --noise-threshold 0.05`, or a
#                                           name filter such as `-- lex/corpus`
#   NSL_BENCH_ONLY="lex parse" scripts/bench.sh ...
#                                           run a subset (bench names, space-separated)
#
# The workflow this exists for: `save main` on the base commit, switch to the
# branch, `compare main`. Criterion prints the per-bench change with a p-value
# and a verdict; the verdict's threshold is `--noise-threshold` (default 1%).
# .github/workflows/bench.yml runs exactly this on a hosted runner and posts
# the verdicts to the job summary.
#
# Why the benches are named here rather than `cargo bench --workspace`: that
# form also runs every crate's libtest harness in bench mode, which compiles
# the test targets of all eleven crates to execute zero benchmarks.
#
# Baselines live under $CARGO_TARGET_DIR/criterion/ (default target/), so a
# target dir per worktree means a baseline per worktree: save and compare
# from the same one.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "bench.sh: $*" >&2
    exit 2
}

# bench:package, in pipeline order.
BENCHES="lex:nsl-lexer parse:nsl-parser analyze:nsl-semantic frontend:nsl-cli compile:nsl-codegen tape:nsl-runtime"

mode="run"
name=""
case "${1:-}" in
    save|compare)
        mode="$1"
        name="${2:-}"
        [[ -n "${name}" ]] || die "'${mode}' needs a baseline NAME"
        shift 2
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    ""|--)
        ;;
    *)
        die "unknown argument '$1' (try --help)"
        ;;
esac
if [[ "${1:-}" == "--" ]]; then
    shift
fi

# Arguments for the criterion binary (after cargo's own `--`).
args=()
case "${mode}" in
    save)    args+=(--save-baseline "${name}") ;;
    compare) args+=(--baseline "${name}") ;;
esac
args+=("$@")

only="${NSL_BENCH_ONLY:-}"
names=""
ran=0
for pair in ${BENCHES}; do
    bench="${pair%%:*}"
    pkg="${pair##*:}"
    names+="${bench} "
    if [[ -n "${only}" && " ${only} " != *" ${bench} "* ]]; then
        continue
    fi
    echo "bench.sh: ${pkg} --bench ${bench}" >&2
    cargo bench -p "${pkg}" --bench "${bench}" -- "${args[@]}"
    ran=$((ran + 1))
done

[[ "${ran}" -gt 0 ]] || die "NSL_BENCH_ONLY='${only}' matched no bench (have: ${names% })"
