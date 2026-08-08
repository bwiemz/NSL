#!/usr/bin/env python3
"""Item 6 — the standardized throughput matrix across 50M / 500M / 1B.

# Why this exists rather than another `mfu_bench.py` invocation

Almost every throughput number this project has quoted came from Coder-50M.
That is a fine probe for launch overhead and elementwise bottlenecks, and it
says nothing about 1B, where the matmuls are 16x larger, the logits surface is
the same size, and the memory schedule is the binding constraint rather than
the kernel count.

`mfu_bench.py` also hard-codes `ModelShape` to the 50M architecture, so it
cannot be pointed at another scale even by hand.

# The measurement rule this harness exists to enforce

**Arms are interleaved, never run back to back.** PR #454 measured a "40%
sgemm win" that was entirely a cold-clock artifact: the first arm ran on a cool
GPU at low clocks, the second on a warm one, and the ranking inverted. The tell
was that kernels the flag could not possibly touch also got faster.

So: every arm in a cell is built first, then the arms are executed round-robin
for `--rounds` rounds, and each arm's reported figure is its BEST round. A
thermal or contention excursion then hits every arm in that round together
instead of landing on whichever one happened to run first. `--rounds 1` is
available for a smoke run and is labelled as not comparable in the output.

# Two passes, because instrumentation is not free

`NSL_HOST_PROFILE=1` wraps six runtime entry points in `Instant::now()` and
prints a line per large copy; `--profile-kernels` adds CUDA events. Measuring
throughput with either on measures the instrumentation.

* **Pass A (timing)** — no instrumentation. Yields tokens/s, MFU, allocator and
  driver peak memory, and the loss stream.
* **Pass B (attribution)** — one extra run per arm with `NSL_HOST_PROFILE=1`
  and `NSL_PASS_TRACE=1`. Yields time by region, PCIe bytes, pass dispositions,
  and fast-path admission counts. Its step times are deliberately discarded.

HBM traffic is NOT collected here. There is no counter for it in the runtime
and no NVML field that reports bytes; `nvidia-smi dmon`'s `mem%` is a
utilisation percentage, not traffic, and multiplying it by peak bandwidth would
produce a number that looks measured and is not. `--ncu` runs Nsight Compute
over a short replay for the real `dram__bytes` figure when it is installed;
without it the column reports `n/a` rather than an estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Dense TFLOPS per device, shared with mfu_bench.py. Measured with a cuBLAS
# GemmEx probe at N=8192, not whitepaper figures — see that file's header.
PEAK_TFLOPS_BY_DEVICE = {
    "NVIDIA GeForce RTX 5070 Ti": {"fp32": 43.9, "tf32": 43.9, "bf16": 87.9},
    "NVIDIA RTX PRO 4500 Blackwell": {"fp32": 36.5, "tf32": 87.6, "bf16": 200.3},
}


@dataclass(frozen=True)
class Scale:
    """One model scale, mirroring its `models/<dir>/config.nsl`."""

    key: str
    dir: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    max_seq_len: int

    def matmul_params(self) -> int:
        """Parameters in a matmul on every token.

        The input embedding is a gather, not a matmul, so it is excluded; the
        output projection is included even under weight tying, because the
        matmul still runs.
        """
        attn = (
            self.d_model * self.n_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_heads * self.head_dim * self.d_model
        )
        ffn = 3 * self.d_model * self.d_ff
        head = self.d_model * self.vocab_size
        return self.n_layers * (attn + ffn) + head

    def flops_per_token(self, seq_len: int) -> float:
        """Forward+backward model FLOPs per token (6N + attention terms)."""
        dense = 6.0 * self.matmul_params()
        attn = 6.0 * self.n_layers * self.n_heads * self.head_dim * seq_len
        return dense + attn


SCALES = {
    s.key: s
    for s in [
        Scale("50m", "coder50m", 8, 512, 8, 4, 64, 1408, 49152, 1024),
        Scale("500m", "coder500m", 24, 1280, 20, 10, 64, 3520, 49152, 1024),
        Scale("1b", "coder1b", 16, 2048, 32, 8, 64, 8192, 49152, 2048),
    ]
}


@dataclass(frozen=True)
class Arm:
    """One comparison arm: compiler flags plus the environment it runs under."""

    key: str
    flags: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    #: Precision the MFU roofline is taken against.
    precision: str = "tf32"
    what: str = ""
    #: `grad_accumulation` the generated program uses. The layerwise/srbf16
    #: family REQUIRES >= 2 (FASE-Deferred window schedule); with 1 the
    #: compile refuses and the 2026-08-02 matrix silently dropped the arm.
    accum: int = 1
    #: Whether the generated program passes grad_clip. `--layerwise-accum`
    #: refuses two-phase clipping (the global L2 norm is never materialized
    #: on the layerwise schedule), so those arms must drop it.
    grad_clip: bool = True


# The arms the matrix compares. `reference` is the independent baseline that
# `--training-reference` builds: it strips every field-controlled optimization
# AND the inferred fused LM head, so it is a baseline for the optimized stack
# rather than a baseline for itself.
ARMS = {
    a.key: a
    for a in [
        Arm("reference", ("--source-ad", "--training-reference"),
            what="composite everything — the independent baseline"),
        # accum STAYS 1 deliberately. Raising it to 2 would make the wgrad
        # fusion live, but it would also move this arm off the (accum=1,
        # grad_clip=True) group the loss-comparability check below keys on —
        # `optimized` would stop being comparable to `reference`/`fp32`/`bf16`,
        # which is the comparison this matrix exists for. The honest fix is to
        # say what the arm actually measures.
        Arm("optimized", ("--pretrain-optimized",),
            what="the shipped bundle: source-AD, WGGO greedy, CSHA auto, "
                 "--fuse-rmsnorm-backward, and the inferred fused LM head. "
                 "NOTE: the bundle also sets --fuse-wgrad-accum, but that "
                 "fusion is INERT here — it fires only through the "
                 "FASE-Deferred on_param_grad hook, which needs "
                 "grad_accumulation >= 2, and this arm runs at 1. The build "
                 "stderr carries a `[wgrad-fusion] declined:` line saying so. "
                 "Any tok/s delta this arm shows over `reference` therefore "
                 "belongs to the OTHER bundle members, not to the weight-"
                 "gradient fusion"),
        Arm("layerwise",
            ("--source-ad", "--checkpoint-blocks", "--layerwise-accum",
             "--weight-stream"),
            what="the CSLA window-buffered schedule with weight streaming — "
                 "the shipped memory stack that makes 1B fit. NOT built on "
                 "--pretrain-optimized: the bundle's WGGO per-parameter FASE "
                 "mode table is refused by --layerwise-accum (the "
                 "window-buffered backward assumes the uniform Deferred "
                 "hook), so this is the same composition the SR-BF16 "
                 "campaign certifies. Trains with grad_accumulation=2 and NO "
                 "grad_clip (both required by --layerwise-accum), so its "
                 "loss stream is not comparable to the other arms'",
            accum=2, grad_clip=False),
        Arm("fp32", ("--source-ad",), (("NSL_MATMUL_TF32", "0"),), "fp32",
            what="full-f32 matmul (TF32 tensor cores off)"),
        Arm("tf32", ("--source-ad",), (), "tf32",
            what="TF32 tensor cores — the compiler default"),
        Arm("bf16", ("--source-ad",), (("NSL_MATMUL_BF16", "1"),), "bf16",
            what="BF16 matmul inputs with f32 accumulate"),
        Arm("srbf16",
            ("--source-ad", "--checkpoint-blocks", "--layerwise-accum",
             "--weight-stream", "--param-dtype", "bf16-sr"),
            (), "bf16",
            what="BF16-authoritative parameters with stochastic-rounded AdamW. "
                 "Trains with grad_accumulation=2 and NO grad_clip (both "
                 "required by --layerwise-accum), so its loss stream is not "
                 "comparable to the other arms'",
            accum=2, grad_clip=False),
    ]
}

#: The matrix from item 6. `microbatches=None` means "probe for the largest
#: that fits" (see `largest_fitting`).
MATRIX = [
    ("50m", 1024, [1, 4, 8, 16], ["reference", "optimized"]),
    ("500m", 1024, None, ["fp32", "bf16", "srbf16"]),
    ("1b", 512, None, ["reference", "optimized", "layerwise"]),
    ("1b", 2048, None, ["reference", "optimized", "layerwise"]),
]


TEMPLATE = """\
# Generated by models/benchmarks/matrix_bench.py — do not edit.
from model import {model_class}
from nsl.nn.losses import cross_entropy

let m = {model_class}()
m.to(cuda)

let tokens = load_mmap("{tokens_path}", 3)
let loader = DataLoader(tokens, batch_size={batch_size}, seq_len={seq_len}, shuffle=false, drop_last=true)

print("MATRIX_BEGIN")

train(model=m, epochs=1{clip_arg}{accum_arg}):
    optimizer: AdamW(lr=0.0003, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let ls = logits.shape
        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])
        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(step)
            print(loss)

print("MATRIX_END")
"""
# The `logits.shape` read is the house idiom on purpose: it is what every
# shipped pretraining script does, so measuring anything else would measure a
# program nobody runs. Item 4's inference folds it to the proven `[B, S, V]`
# under `--fuse-lm-head`, and leaves it alone otherwise — which is exactly the
# difference the reference-vs-optimized comparison is meant to price.


STEP_RE = re.compile(r"^\s*(\d+)\s*$")
LOSS_RE = re.compile(r"^tensor\(\[([0-9.eE+-]+)\]\)\s*$")
GPU_MEM_RE = re.compile(
    r"\[gpu-mem\]\s+step=(\d+)\s+driver=(\d+)MB\s+alloc=(\d+)MB\s+reserved=(\d+)MB"
)
PCIE_RE = re.compile(
    r"pcie\s+h2d=([0-9.]+) MiB\s+d2h=([0-9.]+) MiB\s+total=([0-9.]+) MiB"
)
REGION_RE = re.compile(
    r"^\s{2}(\w+)\s+([0-9.]+) ms\s+(\d+) calls\s+([0-9.]+) us/call\s+([0-9.]+)%",
    re.M,
)
DISPOSITION_RE = re.compile(r"^\[pass-trace\] (\w+): (.+)$")
INFERRED_RE = re.compile(r"\[lm-head-fusion\] inferred: vocab=(\d+) hidden=(\d+) rows=\S+=(\d+)")


@dataclass
class RunResult:
    ok: bool
    step_times_ms: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    peak_alloc_mb: int = 0
    peak_driver_mb: int = 0
    stderr: str = ""
    oom: bool = False


def detect_gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return out.strip().splitlines()[0].strip() if out.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def build_program(work: Path, scale: Scale, tokens: Path, batch: int, seq: int,
                  arm: Arm) -> Path:
    """Write the generated program beside a copy of the model sources.

    The program does `from model import ...`, so it has to sit in a directory
    containing that model's `model.nsl` and `config.nsl`.
    """
    work.mkdir(parents=True, exist_ok=True)
    src_dir = REPO / "models" / scale.dir
    for name in ("model.nsl", "config.nsl"):
        (work / name).write_bytes((src_dir / name).read_bytes())
    model_class = _model_class_of(src_dir / "model.nsl")
    prog = work / "matrix_generated.nsl"
    prog.write_text(
        TEMPLATE.format(
            model_class=model_class,
            tokens_path=tokens.as_posix(),
            batch_size=batch,
            seq_len=seq,
            clip_arg=", grad_clip=1.0" if arm.grad_clip else "",
            accum_arg=f", grad_accumulation={arm.accum}" if arm.accum > 1 else "",
        ),
        encoding="utf-8",
    )
    return prog


def _model_class_of(model_nsl: Path) -> str:
    """The `model <Name>:` a scale's model.nsl declares.

    Read rather than assumed: the three scales do not share a class name, and
    a wrong guess is an import error 40 seconds into a compile.
    """
    for line in model_nsl.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^model\s+(\w+)\s*:", line)
        if m:
            return m.group(1)
    raise SystemExit(f"no `model <Name>:` declaration in {model_nsl}")


def build_binary(nsl: Path, prog: Path, arm: Arm, timeout: int) -> Path | None:
    """Compile one arm ONCE into a standalone binary.

    Every arm flag here is a COMPILE-time decision (`--pretrain-optimized`,
    `--training-reference`, `--param-dtype`), while the environment knobs
    (`NSL_MATMUL_TF32`, `NSL_MATMUL_BF16`) are read at run time — so one build
    per arm serves every round. `nsl run` would recompile on each of the
    ~3 rounds x 20 steps, which at 1B is minutes of wall clock per round buying
    nothing: the timer only ever measures the interval between step callbacks.
    """
    # Absolute: the compiler runs with `cwd=prog.parent`, so a relative `-o`
    # resolves against that directory a SECOND time and the linker fails with
    # "cannot open output file".
    out = (prog.parent / "prog").resolve()
    env = dict(os.environ)
    env.setdefault("NSL_STDLIB_PATH", str(REPO / "stdlib"))
    # [pass-trace] dispositions are COMPILE-time output; setting this on the
    # runtime (as the attribution pass once did) yields an empty table.
    env["NSL_PASS_TRACE"] = "1"
    r = subprocess.run(
        [str(nsl), "build", prog.name, *arm.flags, "-o", str(out)],
        cwd=prog.parent, capture_output=True, text=True, env=env, timeout=timeout,
    )
    # The pass markers this harness reports as "admissions" — [lm-head-fusion],
    # [csha], [wggo], [fuse], [pass-trace] — are COMPILE-time stderr. Scanning
    # only the runtime stderr for them reported every one as false/empty while
    # the runs were in fact optimized (caught on the 2026-08-02 matrix).
    (prog.parent / "build_stderr.txt").write_text(r.stderr)
    if r.returncode != 0 or not out.exists():
        print(f"    build FAILED for {arm.key}: {r.stderr.strip().splitlines()[-3:]}",
              flush=True)
        return None
    return out


def run_once(binary: Path, arm: Arm, max_steps: int, timeout: int,
             extra_env: dict[str, str] | None = None) -> RunResult:
    """Execute one prebuilt arm once, timing the interval between callbacks."""
    env = dict(os.environ)
    env.update(dict(arm.env))
    # nsl_debug_gpu_mem early-returns after step 5 without this, so the peak
    # columns would only describe the warmup.
    env["NSL_DEBUG_MEM_ALL"] = "1"
    env.update(extra_env or {})

    started = time.monotonic()
    proc = subprocess.Popen(
        [str(binary)],
        cwd=binary.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )

    stamps: list[float] = []
    losses: list[float] = []
    out_lines: list[str] = []

    # The reader runs on this thread but the deadline is enforced by polling
    # the child, because a run that hangs before its first callback — an OOM
    # retry loop, a capture deadlock — never reaches the loop body.
    import threading

    def pump_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out_lines.append(line)
            if STEP_RE.match(line):
                stamps.append(time.monotonic())
            else:
                m = LOSS_RE.match(line)
                if m:
                    losses.append(float(m.group(1)))
            if len(stamps) >= max_steps:
                break

    err_chunks: list[str] = []

    def pump_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_chunks.append(line)

    t_out = threading.Thread(target=pump_stdout, daemon=True)
    t_err = threading.Thread(target=pump_stderr, daemon=True)
    t_out.start()
    t_err.start()
    try:
        while t_out.is_alive():
            if time.monotonic() - started > timeout:
                break
            t_out.join(timeout=0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        t_out.join(timeout=10)
        t_err.join(timeout=10)

    stderr = "".join(err_chunks)
    peak_alloc = peak_driver = 0
    for m in GPU_MEM_RE.finditer(stderr):
        peak_driver = max(peak_driver, int(m.group(2)))
        peak_alloc = max(peak_alloc, int(m.group(3)))

    deltas = [1000.0 * (b - a) for a, b in zip(stamps, stamps[1:])]
    oom = "out of memory" in stderr.lower() or "OOM" in stderr
    return RunResult(
        ok=len(deltas) >= 2 and not oom,
        step_times_ms=deltas,
        losses=losses,
        peak_alloc_mb=peak_alloc,
        peak_driver_mb=peak_driver,
        stderr=stderr,
        oom=oom,
    )


def median_ms(r: RunResult, warmup: int) -> float:
    """Median step time after discarding `warmup` leading steps.

    Median rather than mean: a single 300 ms excursion (a cuBLAS autotune, a
    page fault) moves the mean of a 20-step run by 15 ms and the median by
    nothing.
    """
    body = r.step_times_ms[warmup:] or r.step_times_ms
    return statistics.median(body) if body else float("inf")


def interleaved(bins: dict[str, Path], arms: list[str], rounds: int,
                max_steps: int, timeout: int, warmup: int) -> dict[str, list[RunResult]]:
    """Run every arm once per round, round-robin.

    This ordering is the whole point (see the module docstring): arms must
    share their thermal and contention conditions, so a bad round degrades all
    of them rather than whichever ran first.
    """
    results: dict[str, list[RunResult]] = {a: [] for a in arms}
    for r in range(rounds):
        for a in arms:
            res = run_once(bins[a], ARMS[a], max_steps, timeout)
            results[a].append(res)
            tag = "ok" if res.ok else ("OOM" if res.oom else "FAILED")
            ms = median_ms(res, warmup) if res.ok else float("nan")
            print(f"    round {r + 1}/{rounds}  {a:<10} {tag:>6}  "
                  f"{ms:8.1f} ms/step", flush=True)
    return results


def largest_fitting(nsl: Path, work: Path, scale: Scale, tokens: Path, seq: int,
                    arms: list[str], candidates: list[int], timeout: int) -> int:
    """The largest microbatch that runs without OOM on EVERY arm in the cell.

    Every arm, not the most frugal one: a comparison run at different batch
    sizes per arm is not a comparison. Probed descending with two steps, which
    is enough to allocate the full backward.
    """
    for b in sorted(candidates, reverse=True):
        ok = True
        for a in arms:
            prog = build_program(work / f"probe_b{b}_{a}", scale, tokens, b, seq,
                                 ARMS[a])
            binary = build_binary(nsl, prog, ARMS[a], timeout)
            if binary is None:
                ok = False
                break
            res = run_once(binary, ARMS[a], max_steps=3, timeout=timeout)
            if not res.ok:
                ok = False
                print(f"    probe batch={b:<3} {a:<10} "
                      f"{'OOM' if res.oom else 'failed'}", flush=True)
                break
            print(f"    probe batch={b:<3} {a:<10} ok "
                  f"(peak alloc {res.peak_alloc_mb} MB)", flush=True)
        if ok:
            return b
    # No batch fits every arm. Fall back to 1 rather than dropping the cell:
    # at 1B the reference arm is EXPECTED to OOM — that is the finding, and a
    # skipped cell reports it as absence of data instead. The arms that do fit
    # still get a comparable number (they share batch 1), and the ones that do
    # not are recorded as OOM in the results.
    print("    no microbatch fits every arm; falling back to 1 and recording "
          "per-arm OOM", flush=True)
    return 1


def attribution_run(binary: Path, arm: Arm, timeout: int) -> dict:
    """Pass B: one instrumented run whose TIMES ARE DISCARDED.

    `NSL_HOST_PROFILE=1` wraps six runtime entry points in `Instant::now()`;
    including its step times in the throughput table would report the cost of
    measuring. What it is here for is the attribution: which regions, how many
    PCIe bytes, which passes reported what, and whether the fast paths were
    admitted.

    Compile-time decisions (fused LM head, CSHA/WGGO plans, peephole fusions,
    pass dispositions) come from the build stderr saved by `build_binary`;
    only the region timings, PCIe bytes, and runtime launches come from this
    run's stderr.
    """
    res = run_once(binary, arm, max_steps=8, timeout=timeout,
                   extra_env={"NSL_HOST_PROFILE": "1", "NSL_PASS_TRACE": "1"})
    build_log_path = binary.parent / "build_stderr.txt"
    build_log = build_log_path.read_text() if build_log_path.exists() else ""
    regions = {}
    for m in REGION_RE.finditer(res.stderr):
        regions[m.group(1)] = {
            "ms": float(m.group(2)),
            "calls": int(m.group(3)),
            "pct": float(m.group(5)),
        }
    pcie = None
    pms = PCIE_RE.findall(res.stderr)
    if pms:
        # LAST interval: the first one folds in the one-time model upload,
        # which is not steady-state traffic (and the region table already
        # reflects the last interval).
        h2d, d2h, total = pms[-1]
        pcie = {"h2d_mib": float(h2d), "d2h_mib": float(d2h),
                "total_mib": float(total)}
    dispositions = {}
    for line in build_log.splitlines():
        m = DISPOSITION_RE.match(line.strip())
        if m and m.group(1) not in ("did", "N"):
            dispositions[m.group(1)] = m.group(2)
    admissions = {
        "fused_lm_head": bool(INFERRED_RE.search(build_log)),
        "csha_plan": "[csha]" in build_log,
        "wggo_plan": "[wggo]" in build_log,
        "fuse_peephole": build_log.count("[fuse]"),
        "sdpa_fused_forward": "sdpa fused forward: launched" in res.stderr,
    }
    return {
        "ok": res.ok,
        "regions": regions,
        "pcie": pcie,
        "pass_dispositions": dispositions,
        "admissions": admissions,
    }


def loss_divergence(a: list[float], b: list[float]) -> dict | None:
    """Max relative divergence between two loss streams over their overlap."""
    n = min(len(a), len(b))
    if n == 0:
        return None
    worst, at = 0.0, -1
    for i in range(n):
        denom = max(abs(a[i]), 1e-12)
        rel = abs(a[i] - b[i]) / denom
        if rel > worst:
            worst, at = rel, i
    return {
        "steps_compared": n,
        "max_rel": worst,
        "at_step": at,
        "bit_identical": all(a[i] == b[i] for i in range(n)),
        "first": [a[0], b[0]],
        "last": [a[n - 1], b[n - 1]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsl", type=Path, default=REPO / "target/release/nsl")
    # Under `target/`, NOT under `models/`. Each cell writes a copy of the
    # scale's `model.nsl` + `config.nsl` beside its generated program (the
    # program does `from model import ...`), and `model_config_drift` scans
    # `models/` for exactly that shape — so an output directory inside
    # `models/` makes every work dir look like a new, uncovered model and turns
    # that gate red for as long as the results are on disk.
    ap.add_argument("--tokens", type=Path,
                    default=REPO / "target/matrix_logs/tokens.bin")
    ap.add_argument("--out", type=Path, default=REPO / "target/matrix_logs")
    ap.add_argument("--cells", default="all",
                    help="comma-separated <scale>@<seq> selectors, or 'all'")
    ap.add_argument("--rounds", type=int, default=3,
                    help="interleaved rounds per cell; 1 is a smoke run and is "
                         "labelled not-comparable in the output")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--gpu-name", default=None)
    ap.add_argument("--skip-attribution", action="store_true")
    args = ap.parse_args()

    if not args.nsl.exists():
        sys.exit(f"compiler not found at {args.nsl} — build with:\n"
                 f"  cargo build --release --features cuda --bin nsl")

    gpu = args.gpu_name or detect_gpu_name()
    if gpu not in PEAK_TFLOPS_BY_DEVICE:
        sys.exit(f"no measured peak-TFLOPS entry for {gpu!r}. An MFU% against "
                 f"the wrong roofline is worse than none — add the device or "
                 f"pass --gpu-name. Known: {sorted(PEAK_TFLOPS_BY_DEVICE)}")

    # Everything downstream runs subprocesses with a `cwd` of its own, so a
    # relative path from the invocation would mean something different in each.
    args.out = args.out.resolve()
    args.tokens = args.tokens.resolve()
    args.nsl = args.nsl.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.tokens.exists():
        print(f"generating token stream at {args.tokens}", flush=True)
        subprocess.run(
            [sys.executable, str(REPO / "models/benchmarks/gen_tokens.py"),
             str(args.tokens), "--tokens", "4000000"],
            check=True,
        )

    selected = MATRIX
    if args.cells != "all":
        want = set(args.cells.split(","))
        selected = [c for c in MATRIX if f"{c[0]}@{c[1]}" in want]
        if not selected:
            sys.exit(f"no cells matched {args.cells!r}; available: "
                     + ", ".join(f"{c[0]}@{c[1]}" for c in MATRIX))

    report = {
        "gpu": gpu,
        "rounds": args.rounds,
        "comparable": args.rounds >= 2,
        "max_steps": args.max_steps,
        "warmup": args.warmup,
        "cells": [],
    }

    for scale_key, seq, microbatches, arm_keys in selected:
        scale = SCALES[scale_key]
        print(f"\n=== {scale_key} @ seq {seq} — arms: {', '.join(arm_keys)} ===",
              flush=True)
        work = args.out / f"{scale_key}_s{seq}"
        batches = microbatches
        if batches is None:
            print("  probing for the largest microbatch that fits EVERY arm",
                  flush=True)
            b = largest_fitting(args.nsl, work, scale, args.tokens, seq,
                                arm_keys, [16, 8, 4, 2, 1], args.timeout)
            batches = [b]

        for batch in batches:
            print(f"  -- microbatch {batch} --", flush=True)
            bins: dict[str, Path] = {}
            for a in arm_keys:
                prog = build_program(work / f"b{batch}_{a}", scale, args.tokens,
                                     batch, seq, ARMS[a])
                print(f"    building {a} ...", flush=True)
                b = build_binary(args.nsl, prog, ARMS[a], args.timeout)
                if b is not None:
                    bins[a] = b
            live = [a for a in arm_keys if a in bins]
            if not live:
                print("    every arm failed to build; cell skipped", flush=True)
                continue
            results = interleaved(bins, live, args.rounds,
                                  args.max_steps, args.timeout, args.warmup)

            peak = PEAK_TFLOPS_BY_DEVICE[gpu]
            fpt = scale.flops_per_token(seq)
            cell = {
                "scale": scale_key,
                "seq": seq,
                "microbatch": batch,
                "matmul_params_m": scale.matmul_params() / 1e6,
                "mflop_per_token": fpt / 1e6,
                "arms": {},
            }
            # A refused BUILD is a result, not an absence: the 2026-08-02
            # matrix lost three arms exactly because a missing key reads as
            # "not requested".
            for a in arm_keys:
                if a in bins:
                    continue
                stderr_tail = ""
                log = work / f"b{batch}_{a}" / "build_stderr.txt"
                if log.exists():
                    stderr_tail = "\n".join(
                        log.read_text(errors="replace").strip().splitlines()[-3:])
                cell["arms"][a] = {"ok": False, "why": "build refused",
                                   "build_stderr_tail": stderr_tail}
            ref_losses: list[float] | None = None
            for a in live:
                runs = [r for r in results[a] if r.ok]
                if not runs:
                    cell["arms"][a] = {"ok": False,
                                       "why": "OOM" if any(r.oom for r in results[a])
                                              else "no step callbacks"}
                    continue
                # BEST round, not the mean of rounds: the interleaving already
                # made the rounds comparable, so the remaining spread is noise
                # from outside the process and the minimum is the closest to
                # the machine's actual capability.
                best = min(runs, key=lambda r: median_ms(r, args.warmup))
                ms = median_ms(best, args.warmup)
                # on_step fires per MICRO-batch (the step counter increments
                # and callbacks run once per DataLoader iteration; only the
                # optimizer update is gated on grad_accumulation) — so a
                # "step" is batch*seq tokens for EVERY arm, accumulating or
                # not. Multiplying by accum here inflated the accum arms 2x.
                toks = batch * seq
                tok_s = toks / (ms / 1000.0)
                tflops = tok_s * fpt / 1e12
                mfu = 100.0 * tflops / peak[ARMS[a].precision]
                if a == live[0]:
                    ref_losses = best.losses
                cell["arms"][a] = {
                    "ok": True,
                    "what": ARMS[a].what,
                    "flags": list(ARMS[a].flags),
                    "env": dict(ARMS[a].env),
                    "ms_per_step_median": ms,
                    "ms_per_step_all_rounds": [median_ms(r, args.warmup) for r in runs],
                    "useful_tokens_per_s": tok_s,
                    "tflops": tflops,
                    "mfu_pct": mfu,
                    "mfu_roofline": ARMS[a].precision,
                    "peak_alloc_mb": best.peak_alloc_mb,
                    "peak_driver_mb": best.peak_driver_mb,
                    "loss_divergence_vs_first_arm":
                        loss_divergence(ref_losses or [], best.losses)
                        if a != live[0]
                        and (ARMS[a].accum, ARMS[a].grad_clip)
                        == (ARMS[live[0]].accum, ARMS[live[0]].grad_clip)
                        else None,
                }
                if not args.skip_attribution:
                    cell["arms"][a]["attribution"] = attribution_run(
                        bins[a], ARMS[a], args.timeout)
            cell["hbm_traffic"] = "n/a — no runtime counter; run with --ncu for dram__bytes"
            report["cells"].append(cell)
            (args.out / "matrix_results.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            print(f"  wrote {args.out / 'matrix_results.json'}", flush=True)

    (args.out / "matrix_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'matrix_results.json'}")
    if not report["comparable"]:
        print("NOTE: --rounds 1 — arms did not share thermal conditions, so "
              "these numbers rank arms unreliably. See the module docstring.")


if __name__ == "__main__":
    main()
