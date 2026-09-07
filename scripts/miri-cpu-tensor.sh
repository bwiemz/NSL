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
# Usage:  scripts/miri-cpu-tensor.sh                  # tensor::tests, one process
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
