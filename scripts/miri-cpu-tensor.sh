#!/usr/bin/env bash
# Run the CPU tensor unit tests of nsl-runtime under Miri (roadmap C2).
#
# The runtime hands compiled code `i64` handles and turns them back into
# `&'static mut NslTensor` at every `extern "C"` entry point
# (`NslTensor::from_ptr`). Miri's Stacked Borrows model is the only tool in
# the tree that can tell a sound use of that pattern from an unsound one:
# holding one of those references across a call that re-derives the same
# handle is undefined behaviour that runs "fine" natively. The CPU path is
# pure Rust, so the whole `tensor::tests` module interprets; the GPU path
# (`--features cuda`) cannot be interpreted and is not attempted.
#
# Needs a nightly toolchain with the `miri` and `rust-src` components:
#
#     rustup toolchain install nightly --profile minimal \
#         --component miri --component rust-src
#     cargo +nightly miri setup
#
# `-Zmiri-disable-isolation` lets the tests read the clock / env;
# `-Zmiri-permissive-provenance` silences the int-to-pointer warning that
# every handle conversion would otherwise print (the handles are integers by
# ABI design, which is exactly the C2 debt this run measures);
# `-Zmiri-ignore-leaks` turns off the exit-time leak check, because the
# question here is aliasing, and the tests leak on purpose (shape lists and
# tensors that are never freed — 110 allocations across the module). Drop
# that flag to audit test hygiene instead.
#
# Usage:  scripts/miri-cpu-tensor.sh --sweep          # every module of the crate,
#                                                     # one process each, one
#                                                     # summary line per module
#         scripts/miri-cpu-tensor.sh                  # tensor::tests, one process
#         scripts/miri-cpu-tensor.sh --each           # one process per test, so an
#                                                     # error in one test does not
#                                                     # hide the rest (slower)
#         scripts/miri-cpu-tensor.sh [--each] tensor:: # any test-name filter; the
#                                                     # whole tensor namespace is
#                                                     # what found the cpu.rs
#                                                     # aliasing case
#
# Caveat for wider filters: Miri deliberately perturbs the results of the
# transcendental shims (tanh, exp, ...) by a few ulps, and differently on
# each run, precisely so code cannot depend on one libm's rounding. A test
# that asserts bit-exact transcendental results therefore fails under Miri
# while passing natively — in `tensor::activation` that is
# `gelu_backward_cpu_f64_matches_tanh_deriv`,
# `silu_backward_cpu_f64_matches_6op` and
# `swiglu_gate_backward_cpu_f64_bit_exact_vs_pair`, in a varying subset.
# Those are normal test failures, not "Undefined Behavior", and the run
# continues past them: the line to read is the final `test result` plus
# the absence of any "Undefined Behavior" report.
set -euo pipefail
cd "$(dirname "$0")/.."
export MIRIFLAGS="${MIRIFLAGS:--Zmiri-disable-isolation -Zmiri-permissive-provenance -Zmiri-ignore-leaks}"
# Modules whose tests cannot run under Miri at all, with the operation that
# stops them (the first such test aborts the whole process, so they are
# skipped by `--sweep` rather than reported as findings):
#   c_api            re-exec tests spawn a child (`posix_spawnattr_init`)
#   data_source      file-backed memory mappings
#   dataloader       file-backed memory mappings
#   kernel_profiler  `atexit`
#   profiler         `atexit`
#   profiling        `atexit`
# `tensor` is the default filter's own module and `cuda`/`onnx*`/`huggingface`
# are feature-gated or network-bound; `fase_step`, `host_profile` and `fuzz`
# exceed the per-module cap and are skipped for time, not for a Miri
# limitation (`fuzz`'s two seeded loops interpret 15,000 and ~5,000 tensor
# ops each and do not finish in ten minutes apiece; its four small tests
# are clean under `--each fuzz::`).
SWEEP_SKIP="c_api data_source dataloader kernel_profiler profiler profiling cuda onnx onnx_rt_op huggingface fase_step host_profile fuzz"

if [[ "${1:-}" == "--sweep" ]]; then
  fail=0
  for f in crates/nsl-runtime/src/*.rs crates/nsl-runtime/src/*/mod.rs; do
    m=${f#crates/nsl-runtime/src/}; m=${m%/mod.rs}; m=${m%.rs}
    [[ "$m" == lib ]] && continue
    case " $SWEEP_SKIP " in *" $m "*) continue;; esac
    n=$(cargo +nightly miri test -p nsl-runtime --lib -- "${m}::" --list 2>/dev/null | grep -c ": test$" || true)
    [[ "$n" == 0 ]] && continue
    out=$(timeout "${MIRI_MODULE_TIMEOUT:-1200}" cargo +nightly miri test -p nsl-runtime --lib -- "${m}::" 2>&1 || true)
    res=$(printf '%s\n' "$out" | grep -E "^test result" | tail -1)
    ub=$(printf '%s\n' "$out" | grep -m1 -E "Undefined Behavior|unsupported operation" | cut -c1-160)
    if [[ -n "$ub" ]]; then fail=1; fi
    echo "$m ($n tests): ${res:-no summary}${ub:+ | $ub}"
  done
  exit $fail
fi

each=0
if [[ "${1:-}" == "--each" ]]; then each=1; shift; fi
filter="${1:-tensor::tests}"

if [[ "$each" == 1 ]]; then
  names=$(cargo +nightly miri test -p nsl-runtime --lib -- "$filter" --list 2>/dev/null \
    | sed -n 's/: test$//p')
  fail=0
  for n in $names; do
    if cargo +nightly miri test -p nsl-runtime --lib -- --exact "$n" >/tmp/miri-one.log 2>&1; then
      echo "ok   $n"
    else
      echo "FAIL $n"; grep -m1 -A3 "Undefined Behavior\|panicked" /tmp/miri-one.log || tail -5 /tmp/miri-one.log
      fail=1
    fi
  done
  exit $fail
fi

exec cargo +nightly miri test -p nsl-runtime --lib -- "$filter"
