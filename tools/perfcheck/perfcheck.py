#!/usr/bin/env python3
"""NSL against PyTorch on speed and memory: same model, same tokens per step, same card.

Every number this prints is measured on the machine it runs on. Subcommands, in the
order a campaign uses them (each GPU one is short enough to queue on its own):

    roofline  dense GEMM throughput per precision (the MFU denominator), via cuBLAS
              from PyTorch, at 8192^3 and at the model's own projection shapes.
    build     generate the step-limited NSL training program for a size and
              `nsl build` it once per flag set (and the NSL_PHASE_TIMING twin).
    run       one ROUND of timed runs over a list of arms at one size, each arm a
              fresh process, in an order rotated by the round number.
    phases    instrumented runs (device syncs around forward, backward, optimizer);
              their update times are discarded, only the split is kept.
    workspace whether PyTorch's allocator peak counts the cuBLAS workspace.
    nsys      one arm under Nsight Systems; `nsys_summary.py` reads the trace.
    diag      one NSL arm under extra diagnostic environment (NSL_HOST_PROFILE=1, ...).
    report    every result under --out as tables: median over rounds, min-max
              spread, within-round spread, MFU, allocator / reserved / driver peaks.
    pairs     paired tok/s ratios between two arms, taken within each round.

Before/after a compiler change: freeze each toolchain in a directory (`bin/nsl`,
`lib/libnsl_runtime.a`, `stdlib/`), register it with `--toolchain NAME=DIR`, and name
arms `nsl_tf32@NAME`; both builds then run in the same rounds and `pairs --pair
nsl_tf32@after/nsl_tf32@before` gives the paired ratio.

Arms. `nsl_tf32`: the 1B production posture (PROD_FLAGS, f32 weights, the runtime's
default TF32 cuBLAS math mode). `nsl_bf16`: the same flags built and run with
NSL_MATMUL_BF16=1 (read at build time; the runtime's banner is checked, and since
2026-11 a binary run with a disagreeing NSL_MATMUL_BF16 refuses to start).
`nsl_*_nodrain`: the same binary under NSL_SKIP_GPU_DRAIN=1. `nsl_tf32_blocks`: PROD_FLAGS without
`--checkpoint-selective` (full block recompute). `torch_<tf32|bf16>_<eager|compile>_
<none|blocks|sel>`: `torch_arm.py` with that precision, compilation and checkpointing.

Timing: a step-limited program per arm (the token file holds exactly `--updates`
optimizer updates), one stamp per optimizer update when its loss reaches the host.
The window opens at stamp `--warmup` (0-based), so `--updates - --warmup - 1` updates
are timed (15 of 24 by default): tok/s is those updates x tokens-per-update over the
wall time between the window's first and last stamps.

Memory, three numbers per arm, all whole-process peaks:
  alloc     the caching allocator's peak of bytes handed out to tensors
            (NSL `gpu_peak_bytes()`, PyTorch `max_memory_allocated()`);
  reserved  the caching allocator's peak of bytes it held from the driver
            (NSL `NSL_MEMSTATS=1` "Peak reserved", PyTorch `max_memory_reserved()`);
  driver    the process's used memory as nvidia-smi reports it, polled every 100 ms
            (includes the CUDA context, library workspaces and module images).

Synthetic tokens (a seeded uniform draw over the vocabulary): the step does the same
work whatever the ids are.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Self

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "models" / "benchmarks"))
import gpu_guard

PROD_FLAGS = [
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
]
BLOCKS_FLAGS = [f for f in PROD_FLAGS if f != "--checkpoint-selective"]
COMPRESS_FLAGS = [*PROD_FLAGS, "--checkpoint-compress", "bf16"]
BF16_BANNER = "cuBLAS math mode: BF16"
TF32_BANNER = "cuBLAS math mode: TF32"
OOM_MARKERS = ("CUDA_ERROR_OUT_OF_MEMORY", "VRAM free", "out of memory", "OutOfMemory")


@dataclasses.dataclass(frozen=True)
class Shape:
    """Architecture (from `models/<dir>/config.nsl`) and the recipe's step shape."""

    directory: str
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    micro_batch: int
    seq: int
    accum: int

    @property
    def tokens_per_update(self) -> int:
        return self.micro_batch * self.seq * self.accum

    def matmul_params(self) -> int:
        """Parameters in a matmul for every token (the tied LM head counts; the
        embedding gather does not). `models/benchmarks/mfu_bench.py`'s convention."""
        attn = (
            self.d_model * self.n_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_heads * self.head_dim * self.d_model
        )
        return self.n_layers * (attn + 3 * self.d_model * self.d_ff) + self.d_model * self.vocab

    def flops_per_token(self) -> float:
        """Model FLOPs per token, forward + backward: 6N plus causal attention
        (score and value matmuls, halved for the mask). Recompute is not counted:
        this is MFU, not hardware utilisation."""
        attn = 6.0 * self.n_layers * self.n_heads * self.head_dim * self.seq
        return 6.0 * self.matmul_params() + attn


SIZE_DIR = {"50m": "coder50m", "500m": "coder500m", "1b": "coder1b"}


def read_shape(size: str, micro_batch: int | None, seq: int | None, accum: int | None) -> Shape:
    directory = SIZE_DIR[size]
    src = (REPO / "models" / directory / "config.nsl").read_text()

    def const(name: str) -> int:
        m = re.search(rf"^const {name} = (\d+)", src, flags=re.MULTILINE)
        if m is None:
            raise ValueError(f"models/{directory}/config.nsl: no `const {name}`")
        return int(m.group(1))

    # The recipe's step: `config.nsl` must agree with `pretrain_prod.nsl`
    # (crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs), and every recipe trains
    # at MAX_SEQ_LEN.
    mb = const("PRETRAIN_BATCH_SIZE")
    sq = const("MAX_SEQ_LEN")
    ac = const("PRETRAIN_GRAD_ACCUMULATION")
    return Shape(
        directory=directory,
        vocab=const("VOCAB_SIZE"),
        d_model=const("D_MODEL"),
        n_layers=const("N_LAYERS"),
        n_heads=const("N_HEADS"),
        n_kv_heads=const("N_KV_HEADS"),
        head_dim=const("HEAD_DIM"),
        d_ff=const("D_FF"),
        micro_batch=micro_batch or mb,
        seq=seq or sq,
        accum=accum or ac,
    )


# ── arms ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "nsl" | "torch"
    precision: str  # "tf32" | "bf16": selects the roofline
    flags: tuple[str, ...] = ()
    # NSL: environment read by `nsl build` (the matmul mode is a COMPILE-time setting:
    # `MatmulConfig::with_env_fallback`; the runtime never switches mode on its own
    # environment, and refuses a run whose NSL_MATMUL_BF16 disagrees with the binary).
    build_env: tuple[tuple[str, str], ...] = ()
    env: tuple[tuple[str, str], ...] = ()  # NSL: environment of the run
    compile: bool = False
    ckpt: str = "selective"
    # NSL: which toolchain builds the arm (`<arm>@<variant>`, see `--toolchain`);
    # "" is `--nsl` with the repository's stdlib.
    variant: str = ""

    @property
    def build_key(self) -> BuildKey:
        return self.flags, self.build_env, self.variant


# (flags, build environment, toolchain variant): one built binary per key.
BuildKey = tuple[tuple[str, ...], tuple[tuple[str, str], ...], str]


BF16 = (("NSL_MATMUL_BF16", "1"),)
# The runtime's documented Perf knob: skip the per-micro-batch release of cached
# transient memory to the driver (`nsl_gpu_drain_cache`). Same binary.
NO_DRAIN = (("NSL_SKIP_GPU_DRAIN", "1"),)


def arm(name: str) -> Arm:
    """An arm by name. `<nsl arm>@<variant>` is the same arm built by the toolchain
    registered as `--toolchain <variant>=DIR`, so two compilers can be timed in the
    same rounds (before/after a change) and compared with `pairs`."""
    base, sep, variant = name.partition("@")
    if sep:
        a = arm(base)
        if a.kind != "nsl" or not re.fullmatch(r"[A-Za-z0-9_.-]+", variant):
            raise SystemExit(f"arm {name!r}: only NSL arms take an @variant (alphanumeric)")
        return dataclasses.replace(a, name=name, variant=variant)
    nsl = {
        "nsl_tf32": Arm(name, "nsl", "tf32", tuple(PROD_FLAGS)),
        "nsl_bf16": Arm(name, "nsl", "bf16", tuple(PROD_FLAGS), BF16),
        "nsl_tf32_nodrain": Arm(name, "nsl", "tf32", tuple(PROD_FLAGS), (), NO_DRAIN),
        "nsl_bf16_nodrain": Arm(name, "nsl", "bf16", tuple(PROD_FLAGS), BF16, NO_DRAIN),
        "nsl_tf32_blocks": Arm(name, "nsl", "tf32", tuple(BLOCKS_FLAGS)),
        "nsl_bf16_blocks": Arm(name, "nsl", "bf16", tuple(BLOCKS_FLAGS), BF16),
        "nsl_tf32_compress": Arm(name, "nsl", "tf32", tuple(COMPRESS_FLAGS)),
        "nsl_bf16_compress": Arm(name, "nsl", "bf16", tuple(COMPRESS_FLAGS), BF16),
    }
    if name in nsl:
        return nsl[name]
    m = re.fullmatch(r"torch_(tf32|bf16)_(eager|compile)_(none|blocks|sel)", name)
    if m is None:
        raise SystemExit(f"unknown arm {name!r}")
    ckpt = {"none": "none", "blocks": "blocks", "sel": "selective"}[m.group(3)]
    return Arm(name, "torch", m.group(1), compile=m.group(2) == "compile", ckpt=ckpt)


def flag_key(key: BuildKey) -> str:
    flags, build_env, variant = key
    tag = "_".join(f.removeprefix("--").replace("-", "") for f in flags)
    if build_env:
        tag += "__" + "_".join(f"{k.removeprefix('NSL_').lower()}{v}" for k, v in build_env)
    if variant:
        tag += "@" + variant
    return "nsl_" + tag


@dataclasses.dataclass(frozen=True)
class Toolchain:
    """What builds an NSL arm: the `nsl` binary and the stdlib it compiles against."""

    nsl: Path
    stdlib: Path


def toolchains(args: argparse.Namespace) -> dict[str, Toolchain]:
    """`--toolchain NAME=DIR` registrations. DIR is a frozen toolchain: `bin/nsl`,
    `lib/libnsl_runtime.a` (found by `nsl` next to its own `bin/`), `stdlib/`. The
    empty name is `--nsl` with the repository's stdlib."""
    out = {"": Toolchain(args.nsl.resolve(), REPO / "stdlib")}
    for spec in getattr(args, "toolchain", None) or []:
        name, sep, d = spec.partition("=")
        root = Path(d).resolve()
        if not sep or not (root / "bin" / "nsl").is_file():
            raise SystemExit(f"--toolchain {spec!r}: want NAME=DIR with DIR/bin/nsl")
        if not (root / "lib" / "libnsl_runtime.a").is_file():
            raise SystemExit(f"--toolchain {spec!r}: DIR/lib/libnsl_runtime.a missing")
        if not (root / "stdlib").is_dir():
            raise SystemExit(f"--toolchain {spec!r}: DIR/stdlib missing")
        out[name] = Toolchain(root / "bin" / "nsl", root / "stdlib")
    return out


# ── the NSL program ──────────────────────────────────────────────────────────

TEMPLATE = """\
# GENERATED by tools/perfcheck/perfcheck.py; not a recipe.
# The pretrain_prod.nsl step (AdamW, warmup_cosine, grad_clip 1.0, @fused_lm_ce,
# optimizer-step schedule unit) without checkpoints or validation. The token file
# holds exactly {updates} optimizer updates; one OPT_STEP line per update.
from model import NSLCoder
from nsl.nn.losses import cross_entropy

let m = NSLCoder()
m.to(cuda)

let tokens = load_mmap("{tokens}", 3)
let loader = DataLoader(tokens, batch_size={mb}, seq_len={seq}, shuffle=false, drop_last=true)

print("LOSS_STREAM_BEGIN")

@fused_lm_ce(enabled=true, vocab_size={vocab}, hidden_size={d_model}, batch_size={mb}, seq_len={seq})
train(model=m, epochs=1, grad_accumulation={accum}, schedule_unit="optimizer_step", grad_clip=1.0):
    optimizer: AdamW(lr=0.0001, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
    scheduler: warmup_cosine(warmup_steps=2, total_steps={updates}, min_lr=0.00003)
    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let flat_logits = logits.reshape([{rows}, {vocab}])
        let flat_labels = batch.labels.reshape([{rows}])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            if step % {accum} == 0:
                print("OPT_STEP")
                print(loss)

print("LOSS_STREAM_END")
print("PEAK_BYTES")
print(gpu_peak_bytes())
print("PEAK_WEIGHTS")
print(gpu_surface_at_peak_bytes(1))
print("PEAK_OPTIM_M")
print(gpu_surface_at_peak_bytes(2))
print("PEAK_OPTIM_V")
print(gpu_surface_at_peak_bytes(3))
print("PEAK_M_PARTIAL")
print(gpu_surface_at_peak_bytes(4))
print("PEAK_ACTIVATIONS")
print(gpu_surface_at_peak_bytes(6))
print("PERFCHECK_COMPLETE")
"""


def size_dir(out: Path, size: str, shape: Shape) -> Path:
    return out / f"{size}_b{shape.micro_batch}_s{shape.seq}_a{shape.accum}"


def tokens_file(work: Path, shape: Shape, updates: int) -> Path:
    """Exactly `updates` optimizer updates of tokens (plus slack below one micro-batch)."""
    path = work / f"tokens_u{updates}.bin"
    n = updates * shape.tokens_per_update + 8
    if not path.exists() or path.stat().st_size != 2 * n:
        rng = np.random.Generator(np.random.PCG64(1234))
        path.parent.mkdir(parents=True, exist_ok=True)
        rng.integers(0, shape.vocab, size=n, dtype=np.uint16).tofile(path)
    return path


def nsl_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("NSL_STDLIB_PATH", str(REPO / "stdlib"))
    # The arm decides the math mode; an inherited value must not decide it.
    for k in ("NSL_MATMUL_BF16", "NSL_MATMUL_BF16_ROUND", "NSL_PHASE_TIMING", "NSL_SKIP_GPU_DRAIN"):
        env.pop(k, None)
    return env


def build_nsl(
    tc: Toolchain,
    work: Path,
    shape: Shape,
    key: BuildKey,
    updates: int,
    phase: bool,
) -> Path:
    flags, build_env, variant = key
    nsl = tc.nsl
    d = work / flag_key(key)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("model.nsl", "config.nsl"):
        shutil.copyfile(REPO / "models" / shape.directory / f, d / f)
    tokens = tokens_file(work, shape, updates)
    src = d / "bench.nsl"
    src.write_text(
        TEMPLATE.format(
            updates=updates,
            tokens=tokens,
            mb=shape.micro_batch,
            seq=shape.seq,
            accum=shape.accum,
            vocab=shape.vocab,
            d_model=shape.d_model,
            rows=shape.micro_batch * shape.seq,
        )
    )
    binary = d / ("prog_phase" if phase else "prog")
    env = nsl_env()
    env["NSL_STDLIB_PATH"] = str(tc.stdlib)
    env.update(dict(build_env))
    if phase:
        env["NSL_PHASE_TIMING"] = "1"  # read at COMPILE time
    r = subprocess.run(
        [str(nsl), "build", *flags, str(src), "-o", str(binary)],
        cwd=d,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    (d / f"build_{binary.name}.stderr").write_text(r.stderr + r.stdout)
    if r.returncode != 0:
        raise SystemExit(f"{d.name}: nsl build failed:\n{(r.stderr + r.stdout)[-3000:]}")
    # The fused LM-CE route is decided at build time; an unfused binary would be a
    # different (and much larger) step. Refuse it rather than time it.
    if "[fused-lce] forward route=" not in r.stderr or "will NOT run" in r.stderr:
        raise SystemExit(f"{d.name}: fused LM-CE did not engage:\n{r.stderr[-3000:]}")
    (d / f"{binary.name}.meta.json").write_text(
        json.dumps(
            {
                "flags": list(flags),
                "build_env": dict(build_env),
                "updates": updates,
                "nsl": str(nsl),
                "variant": variant,
                "stdlib": str(tc.stdlib),
            },
            indent=1,
        )
    )
    return binary


# ── running ──────────────────────────────────────────────────────────────────


class GpuSampler:
    """nvidia-smi pollers: per-process used memory (100 ms) and the card's SM clock,
    temperature and power (200 ms), stamped on the parent's monotonic clock (the
    clock the update stamps use: Python's perf_counter and monotonic are both
    CLOCK_MONOTONIC on Linux)."""

    def __init__(self) -> None:
        self.peak: dict[int, int] = {}
        self.card: list[tuple[float, float, float, float]] = []  # t, MHz, degC, W
        self._procs: list[subprocess.Popen[str]] = []
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> Self:
        mem = subprocess.Popen(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )  # fmt: skip
        card = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )  # fmt: skip
        self._procs = [mem, card]

        def read_mem() -> None:
            assert mem.stdout is not None
            for line in mem.stdout:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    pid, mib = int(parts[0]), int(parts[1])
                    self.peak[pid] = max(self.peak.get(pid, 0), mib)

        def read_card() -> None:
            assert card.stdout is not None
            for line in card.stdout:
                try:
                    mhz, temp, watts = (float(x) for x in line.split(","))
                except ValueError:
                    continue
                self.card.append((time.monotonic(), mhz, temp, watts))

        self._threads = [threading.Thread(target=f, daemon=True) for f in (read_mem, read_card)]
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in self._procs:
            p.terminate()
            p.wait(timeout=10)
        for t in self._threads:
            t.join(timeout=5)

    def card_between(self, lo: float, hi: float) -> dict[str, float] | None:
        xs = [c for c in self.card if lo <= c[0] <= hi]
        if not xs:
            return None
        return {
            "sm_mhz_mean": statistics.fmean(c[1] for c in xs),
            "sm_mhz_min": min(c[1] for c in xs),
            "temp_c_max": max(c[2] for c in xs),
            "power_w_mean": statistics.fmean(c[3] for c in xs),
            "samples": len(xs),
        }


def run_stamped(
    cmd: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: float
) -> tuple[int, list[tuple[float, str]], int | None, GpuSampler]:
    """Run, stamping each stdout line's arrival; return (rc, lines, driver peak MiB,
    the sampler)."""
    lines: list[tuple[float, str]] = []
    with GpuSampler() as mem, log.open("w") as lf:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=lf, text=True)
        assert p.stdout is not None
        watchdog = threading.Timer(timeout, p.kill)
        watchdog.start()
        try:
            for line in p.stdout:
                lines.append((time.monotonic(), line.rstrip("\n")))
        finally:
            watchdog.cancel()
            p.wait()
        time.sleep(0.3)  # one more sample after exit cannot raise the peak, but
        # a sample in flight when the child exits is still read
        peak = mem.peak.get(p.pid)
    log.with_suffix(".stdout").write_text("".join(f"{t:.6f}\t{s}\n" for t, s in lines))
    return p.returncode, lines, peak, mem


def steady(stamps: list[float], warmup: int, tokens_per_update: int) -> dict[str, Any] | None:
    """Throughput and spread over the updates between `stamps[warmup]` and the last stamp."""
    window = stamps[warmup:]
    if len(window) < 3:
        return None
    gaps = [b - a for a, b in itertools.pairwise(window)]
    wall = window[-1] - window[0]
    return {
        "tok_s": len(gaps) * tokens_per_update / wall,
        "update_s_median": statistics.median(gaps),
        "update_s_mean": statistics.fmean(gaps),
        "update_s_min": min(gaps),
        "update_s_max": max(gaps),
        "update_s_cv": statistics.pstdev(gaps) / statistics.fmean(gaps),
        "updates_measured": len(gaps),
        "all_update_s": [b - a for a, b in itertools.pairwise(stamps)],
    }


def parse_loss(line: str) -> float | None:
    """The value of a printed scalar tensor (`tensor([11.2])`, `tensor([NaN])`) or a bare
    number; None when the line is neither. `float` reads nan/inf in any case."""
    m = re.fullmatch(r"\s*(?:tensor\(\[)?\s*([^\]\s]+)\s*(?:\]\))?\s*", line)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def loss_problem(losses: list[float], updates: int) -> str | None:
    """Why a run's per-update losses cannot be trusted, or None: one per update, all
    finite, and lower at the end than at the start (24 updates of AdamW at lr 1e-4 from
    a fresh init fall by 0.04-0.26 nats on both frameworks)."""
    if len(losses) != updates:
        return f"{len(losses)} losses for {updates} updates"
    if not all(math.isfinite(x) for x in losses):
        return f"non-finite loss: {losses}"
    if losses[-1] >= losses[0]:
        return f"loss did not fall: {losses[0]} -> {losses[-1]}"
    return None


def after(text: list[str], marker: str) -> int | None:
    try:
        return int(float(text[text.index(marker) + 1]))
    except (ValueError, IndexError):
        return None


def run_nsl_arm(
    a: Arm,
    work: Path,
    shape: Shape,
    updates: int,
    warmup: int,
    rnd: int,
    timeout: float,
    phase: bool = False,
) -> dict[str, Any]:
    d = work / flag_key(a.build_key)
    binary = d / ("prog_phase" if phase else "prog")
    if not binary.exists():
        raise SystemExit(f"{binary} missing: run `perfcheck.py build` first")
    meta = json.loads((d / f"{binary.name}.meta.json").read_text())
    if meta["updates"] != updates:
        raise SystemExit(f"{binary} was built for {meta['updates']} updates, not {updates}")
    env = nsl_env()
    env.update(dict(a.build_env))  # as `export NSL_MATMUL_BF16=1; nsl run` would have it
    env.update(dict(a.env))
    env["NSL_MEMSTATS"] = "1"  # printed at exit only; costs nothing during the run
    tag = "phase" if phase else f"r{rnd}"
    log = work / "logs" / f"{a.name}_{tag}.stderr"
    log.parent.mkdir(parents=True, exist_ok=True)
    rc, lines, driver, sampler = run_stamped([str(binary)], d, env, log, timeout)
    err = log.read_text()
    text = [s for _, s in lines]
    row: dict[str, Any] = {
        "arm": a.name,
        "kind": "nsl",
        "precision": a.precision,
        "flags": list(a.flags),
        "build_env": dict(a.build_env),
        "env": dict(a.env),
        "round": rnd,
        "updates": updates,
        "warmup_updates": warmup,
        "tokens_per_update": shape.tokens_per_update,
        "driver_peak_mib": driver,
        "status": "ok",
    }
    if a.precision == "bf16" and BF16_BANNER not in err:
        row["status"] = "error"
        row["error"] = "NSL_MATMUL_BF16=1 set but the runtime did not print its BF16 banner"
    elif a.precision == "tf32" and TF32_BANNER not in err:
        row["status"] = "error"
        row["error"] = "tf32 arm: the runtime did not print its TF32 math-mode banner"
    if rc != 0:
        row["status"] = "oom" if any(m in err for m in OOM_MARKERS) else "error"
        row["error"] = err[-2500:]
    if "PERFCHECK_COMPLETE" not in text and row["status"] == "ok":
        row["status"] = "error"
        row["error"] = "program did not reach PERFCHECK_COMPLETE\n" + err[-2500:]
    stamps = [t for t, s in lines if s == "OPT_STEP"]
    losses: list[float] = []
    for i, s in enumerate(text):
        if s == "OPT_STEP" and i + 1 < len(text):
            v = parse_loss(text[i + 1])
            if v is not None:
                losses.append(v)
    row["losses"] = losses
    problem = loss_problem(losses, updates)
    if problem is not None and row["status"] == "ok":
        row["status"] = "error"
        row["error"] = problem
    row["stamps"] = stamps
    row["steady"] = steady(stamps, warmup, shape.tokens_per_update)
    if len(stamps) > warmup:
        row["card_steady"] = sampler.card_between(stamps[warmup], stamps[-1])
    row["alloc_peak_bytes"] = after(text, "PEAK_BYTES")
    row["surfaces_at_peak"] = {
        k: after(text, mk)
        for k, mk in (
            ("weights", "PEAK_WEIGHTS"),
            ("optim_m", "PEAK_OPTIM_M"),
            ("optim_v", "PEAK_OPTIM_V"),
            ("m_partial", "PEAK_M_PARTIAL"),
            ("activations", "PEAK_ACTIVATIONS"),
        )
    }
    m = re.search(r"Peak reserved:\s+([0-9.]+) (GB|MB|KB|B)\b", err)
    if m:
        unit = {"GB": 2**30, "MB": 2**20, "KB": 2**10, "B": 1}[m.group(2)]
        row["reserved_peak_bytes"] = int(float(m.group(1)) * unit)
    if phase:
        fwd, bwd, opt = [], [], []
        for s in text:
            pm = re.match(r"\[phase\] fwd=([0-9.]+) bwd=([0-9.]+)", s)
            if pm:
                fwd.append(float(pm.group(1)))
                bwd.append(float(pm.group(2)))
            om = re.match(r"\[phase\] opt=([0-9.]+)", s)
            if om:
                opt.append(float(om.group(1)))
        row["phase_micro"] = {"fwd": fwd, "bwd": bwd}
        row["phase_opt"] = opt
    return row


def run_torch_arm(
    a: Arm,
    work: Path,
    size: str,
    shape: Shape,
    updates: int,
    warmup: int,
    rnd: int,
    timeout: float,
    python: str,
    phase: bool = False,
) -> dict[str, Any]:
    tag = "phase" if phase else f"r{rnd}"
    out = work / "torch_json" / f"{a.name}_{tag}.json"
    out.unlink(missing_ok=True)
    log = work / "logs" / f"{a.name}_{tag}.stderr"
    log.parent.mkdir(parents=True, exist_ok=True)
    cache = work.parent / "torch_cache"
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    cmd = [
        python, str(HERE / "torch_arm.py"),
        "--arm", a.name, "--size", size, "--precision", a.precision, "--ckpt", a.ckpt,
        "--micro-batch", str(shape.micro_batch), "--seq", str(shape.seq),
        "--accum", str(shape.accum), "--updates", str(updates),
        "--warmup-updates", str(warmup), "--tokens", str(tokens_file(work, shape, updates)),
        "--out", str(out), "--round", str(rnd),
    ]  # fmt: skip
    if a.compile:
        cmd.append("--compile")
    if phase:
        cmd.append("--phase-timing")
    rc, _, driver, sampler = run_stamped(cmd, HERE, env, log, timeout)
    if not out.exists():
        return {"arm": a.name, "kind": "torch", "precision": a.precision, "round": rnd,
                "status": "error", "error": f"rc={rc}, no JSON\n{log.read_text()[-2500:]}",
                "driver_peak_mib": driver}  # fmt: skip
    row: dict[str, Any] = json.loads(out.read_text())
    row["kind"] = "torch"
    row["driver_peak_mib"] = driver
    problem = loss_problem(row["losses"], updates)
    if problem is not None and row["status"] == "ok":
        row["status"] = "error"
        row["error"] = problem
    row["steady"] = steady(row["stamps"], warmup, shape.tokens_per_update)
    if len(row["stamps"]) > warmup:
        row["card_steady"] = sampler.card_between(row["stamps"][warmup], row["stamps"][-1])
    return row


# ── subcommands ──────────────────────────────────────────────────────────────


def cmd_roofline(args: argparse.Namespace) -> int:
    gpu_guard.acquire_or_refuse("perfcheck-roofline")
    import torch

    # PyTorch's defaults are kept and recorded: TF32 off except in the tf32 rows, and
    # reduced-precision (16-bit) split-K reductions allowed for bf16, as the bf16 arms
    # run with them too.
    out: dict[str, Any] = {
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "method": "torch.mm, CUDA events, 10 warm-up then 3 x 30 timed launches; "
        "median of the 3 batches",
        "square": {},
        "shapes": {},
    }

    def tflops(m: int, k: int, n: int, prec: str) -> float:
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        dt = torch.bfloat16 if prec == "bf16" else torch.float32
        x = torch.randn(m, k, device="cuda", dtype=dt)
        w = torch.randn(k, n, device="cuda", dtype=dt)
        for _ in range(10):
            torch.mm(x, w)
        torch.cuda.synchronize()
        rates = []
        for _ in range(3):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(30):
                torch.mm(x, w)
            e.record()
            e.synchronize()
            rates.append(2.0 * m * k * n * 30 / (s.elapsed_time(e) / 1e3) / 1e12)
        return statistics.median(rates)

    for prec in ("fp32", "tf32", "bf16"):
        out["square"][prec] = {str(n): tflops(n, n, n, prec) for n in (4096, 8192)}
        print(f"{prec}: {out['square'][prec]}", flush=True)
    # The models' own projections at their recipes' row counts (tokens per micro-batch).
    for size in ("500m", "1b"):
        s = read_shape(size, None, None, None)
        rows = s.micro_batch * s.seq
        q, kv = s.n_heads * s.head_dim, s.n_kv_heads * s.head_dim
        for label, (k, n) in {
            "qkv": (s.d_model, q + 2 * kv),
            "wo": (q, s.d_model),
            "gate_up": (s.d_model, 2 * s.d_ff),
            "down": (s.d_ff, s.d_model),
            "lm_head": (s.d_model, s.vocab),
        }.items():
            for prec in ("tf32", "bf16"):
                out["shapes"].setdefault(size, {}).setdefault(label, {})[prec] = tflops(
                    rows, k, n, prec
                )
        print(f"{size}: {out['shapes'][size]}", flush=True)
    torch.cuda.empty_cache()
    if args.nsl.exists():
        out["nsl_gemm"] = nsl_gemm_probe(args.nsl.resolve(), args.out.resolve() / "nsl_gemm")
        print(f"nsl: {out['nsl_gemm']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "roofline.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_workspace(args: argparse.Namespace) -> int:
    """Whether PyTorch's `max_memory_allocated` counts the cuBLAS workspace: the
    allocated-bytes delta across the process's first GEMM, whose output is freed."""
    gpu_guard.acquire_or_refuse("perfcheck-workspace")
    import torch

    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    c = a @ b
    del c
    torch.cuda.synchronize()
    out = {
        "first_gemm_retained_bytes": torch.cuda.memory_allocated() - before,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "torch": torch.__version__,
    }
    print(json.dumps(out), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "torch_workspace.json").write_text(json.dumps(out, indent=1))
    return 0


GEMM_PROBE = """\
# GENERATED by tools/perfcheck/perfcheck.py: NSL's own matmul path on one GEMM.
let a = randn([N, N]).to(cuda)
let b = randn([N, N]).to(cuda)
for i in range(10):
    let c = a @ b
print((a @ b).sum().item())
let t0 = clock()
for i in range(10):
    let c = a @ b
print((a @ b).sum().item())
let t1 = clock()
for i in range(40):
    let c = a @ b
print((a @ b).sum().item())
let t2 = clock()
print("GEMM_S")
print(t1 - t0)
print(t2 - t1)
"""


def nsl_gemm_probe(nsl: Path, work: Path, n: int = 8192) -> dict[str, Any]:
    """TFLOPS of `a @ b` (f32 tensors, n x n) through NSL's runtime, built in its default
    math mode and with NSL_MATMUL_BF16=1 (which casts both operands per call).
    Timed as (40 launches - 10 launches) so the closing sync and readback cancel."""
    work.mkdir(parents=True, exist_ok=True)
    src = work / "gemm.nsl"
    src.write_text(GEMM_PROBE.replace("N, N", f"{n}, {n}"))
    res: dict[str, Any] = {"n": n}
    for label, extra in (("default", {}), ("bf16", {"NSL_MATMUL_BF16": "1"})):
        env = nsl_env()
        env.update(extra)
        binary = work / f"gemm_{label}"
        r = subprocess.run(
            [str(nsl), "build", str(src), "-o", str(binary)],
            cwd=work, capture_output=True, text=True, env=env, check=False,
        )  # fmt: skip
        if r.returncode != 0:
            res[label + "_error"] = (r.stderr + r.stdout)[-1500:]
            continue
        p = subprocess.run(
            [str(binary)], cwd=work, capture_output=True, text=True, env=env, check=False
        )
        text = p.stdout.splitlines()
        try:
            i = text.index("GEMM_S")
            t10, t40 = float(text[i + 1]), float(text[i + 2])
            res[label] = 2.0 * n**3 * 30 / (t40 - t10) / 1e12
            res[label + "_banner"] = next(
                (ln for ln in p.stderr.splitlines() if "math mode" in ln), None
            )
        except (ValueError, IndexError):
            res[label] = None
            res[label + "_error"] = (p.stderr + p.stdout)[-1500:]
    return res


def cmd_build(args: argparse.Namespace) -> int:
    shape = read_shape(args.size, args.micro_batch, args.seq, args.accum)
    work = size_dir(args.out.resolve(), args.size, shape)
    keys = {arm(n).build_key for n in args.arms.split(",") if arm(n).kind == "nsl"}
    tcs = toolchains(args)
    for key in sorted(keys):
        if key[2] not in tcs:
            raise SystemExit(f"variant {key[2]!r}: no --toolchain {key[2]}=DIR given")
        for phase in (False, True) if args.phase else (False,):
            t = time.monotonic()
            b = build_nsl(tcs[key[2]], work, shape, key, args.updates, phase)
            print(f"[build] {b} ({time.monotonic() - t:.0f} s)", flush=True)
    return 0


def cmd_run(args: argparse.Namespace, phase: bool = False) -> int:
    gpu_guard.acquire_or_refuse("perfcheck-run")
    shape = read_shape(args.size, args.micro_batch, args.seq, args.accum)
    work = size_dir(args.out.resolve(), args.size, shape)
    arms = [arm(n) for n in args.arms.split(",")]
    k = args.round % len(arms)
    arms = arms[k:] + arms[:k]  # rotate so no arm always runs first (or last)
    for a in arms:
        t = time.monotonic()
        if a.kind == "nsl":
            row = run_nsl_arm(a, work, shape, args.updates, args.warmup, args.round,
                              args.timeout, phase)  # fmt: skip
        else:
            row = run_torch_arm(a, work, args.size, shape, args.updates, args.warmup,
                                args.round, args.timeout, args.python, phase)  # fmt: skip
        row["size"] = args.size
        row["shape"] = dataclasses.asdict(shape)
        row["flops_per_token"] = shape.flops_per_token()
        row["wall_s"] = time.monotonic() - t
        tag = "phase" if phase else f"r{args.round}"
        dest = work / ("phases" if phase else "results") / f"{a.name}_{tag}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(row, indent=1))
        st = row.get("steady") or {}
        print(
            f"[{tag}] {a.name:26s} {row['status']:5s} "
            f"{st.get('tok_s', 0):9,.0f} tok/s  cv {st.get('update_s_cv', 0):.3f}  "
            f"alloc {(row.get('alloc_peak_bytes') or 0) / 2**30:5.2f} GiB  "
            f"driver {(row.get('driver_peak_mib') or 0) / 1024:5.2f} GiB  "
            f"({row['wall_s']:.0f} s)",
            flush=True,
        )
    return 0


def cmd_diag(args: argparse.Namespace) -> int:
    """One NSL arm under extra diagnostic environment (e.g. NSL_HOST_PROFILE=1).
    An instrumented run: its timings are not the arm's throughput."""
    gpu_guard.acquire_or_refuse("perfcheck-diag")
    shape = read_shape(args.size, args.micro_batch, args.seq, args.accum)
    work = size_dir(args.out.resolve(), args.size, shape)
    a = arm(args.arm)
    if a.kind != "nsl":
        raise SystemExit("diag runs NSL arms only")
    d = work / flag_key(a.build_key)
    binary = d / ("prog_phase" if args.phase else "prog")
    env = nsl_env()
    env.update(dict(a.build_env))
    env.update(dict(a.env))
    env.update(dict(kv.split("=", 1) for kv in args.env))
    log = work / "logs" / f"{a.name}_diag_{args.tag}.stderr"
    log.parent.mkdir(parents=True, exist_ok=True)
    rc, _, _, _ = run_stamped([str(binary)], d, env, log, args.timeout)
    print(f"[diag] {a.name} {args.tag}: rc={rc}, {log}", flush=True)
    return rc


def cmd_nsys(args: argparse.Namespace) -> int:
    gpu_guard.acquire_or_refuse("perfcheck-nsys")
    shape = read_shape(args.size, args.micro_batch, args.seq, args.accum)
    work = size_dir(args.out.resolve(), args.size, shape)
    a = arm(args.arm)
    rep = work / "nsys" / f"{a.name}{'_phase' if args.phase else ''}"
    rep.parent.mkdir(parents=True, exist_ok=True)
    nsys = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
            "--force-overwrite=true", "--export=sqlite", "-o", str(rep)]  # fmt: skip
    if a.kind == "nsl":
        d = work / flag_key(a.build_key)
        binary = d / ("prog_phase" if args.phase else "prog")
        env = nsl_env()
        env.update(dict(a.build_env))
        env.update(dict(a.env))
        cmd, cwd = [*nsys, str(binary)], d
    else:
        env = dict(os.environ)
        cache = work.parent / "torch_cache"
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        env["TRITON_CACHE_DIR"] = str(cache / "triton")
        out = work / "torch_json" / f"{a.name}_nsys.json"
        cmd = [*nsys, args.python, str(HERE / "torch_arm.py"),
               "--arm", a.name, "--size", args.size, "--precision", a.precision,
               "--ckpt", a.ckpt, "--micro-batch", str(shape.micro_batch),
               "--seq", str(shape.seq), "--accum", str(shape.accum),
               "--updates", str(args.updates), "--warmup-updates", str(args.warmup),
               "--tokens", str(tokens_file(work, shape, args.updates)),
               "--out", str(out), "--no-kernel-probe"]  # fmt: skip
        if a.compile:
            cmd.append("--compile")
        if args.phase:
            cmd.append("--phase-timing")
        cwd = HERE
    log = work / "logs" / f"{rep.name}_nsys.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            timeout=args.timeout, check=False).returncode  # fmt: skip
    print(f"[nsys] {a.name}: rc={rc}, {rep}.sqlite", flush=True)
    return rc


def _fmt_range(xs: list[float], fmt: str) -> str:
    if not xs:
        return "-"
    med = statistics.median(xs)
    if len(xs) == 1:
        return format(med, fmt)
    return f"{format(med, fmt)} [{format(min(xs), fmt)}-{format(max(xs), fmt)}]"


def phase_split(r: dict[str, Any]) -> dict[str, float] | None:
    """Mean seconds per steady update in forward(+loss), backward(+recompute), optimizer
    (+clip), and the rest of the update's wall time, from an instrumented run."""
    w, acc = r["warmup_updates"], r["shape"]["accum"]
    stamps = r["stamps"]
    if len(stamps) <= w + 1:
        return None
    n = len(stamps) - 1 - w
    wall = (stamps[-1] - stamps[w]) / n
    if r["kind"] == "nsl":
        # One [phase] fwd/bwd line per micro-batch, one opt line per update; the update
        # whose stamp opens the window is excluded, as the stamps exclude it.
        fwd = r["phase_micro"]["fwd"][(w + 1) * acc : (w + 1 + n) * acc]
        bwd = r["phase_micro"]["bwd"][(w + 1) * acc : (w + 1 + n) * acc]
        opt = r["phase_opt"][w + 1 : w + 1 + n]
        f, b, o = sum(fwd) / n, sum(bwd) / n, sum(opt) / n
    else:
        ph = r["phases"][w + 1 : w + 1 + n]
        f = sum(x["fwd"] for x in ph) / n
        b = sum(x["bwd"] for x in ph) / n
        o = sum(x["opt"] for x in ph) / n
    return {"wall": wall, "fwd": f, "bwd": b, "opt": o, "other": wall - f - b - o}


REPORT_ORDER = [
    "nsl_tf32", "nsl_tf32_nodrain", "nsl_tf32_blocks", "nsl_tf32_compress",
    "nsl_bf16", "nsl_bf16_nodrain", "nsl_bf16_blocks", "nsl_bf16_compress",
    "torch_tf32_eager_sel", "torch_tf32_eager_blocks", "torch_tf32_eager_none",
    "torch_bf16_eager_sel", "torch_bf16_eager_blocks", "torch_bf16_eager_none",
    "torch_bf16_compile_sel", "torch_bf16_compile_blocks", "torch_bf16_compile_none",
]  # fmt: skip


def cmd_report(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    roof_path = out / "roofline.json"
    roof = json.loads(roof_path.read_text()) if roof_path.exists() else None
    for work in sorted(p for p in out.iterdir() if (p / "results").is_dir()):
        rows: dict[str, list[dict[str, Any]]] = {}
        for f in sorted((work / "results").glob("*.json")):
            r = json.loads(f.read_text())
            # Re-check results written before the loss check existed, and any since.
            problem = loss_problem(r.get("losses") or [], r.get("updates", 0))
            if r["status"] == "ok" and problem is not None:
                r["status"] = "error"
                r["error"] = problem
            rows.setdefault(r["arm"], []).append(r)
        order = [*REPORT_ORDER, *sorted(set(rows) - set(REPORT_ORDER))]
        rows = {k: rows[k] for k in order if k in rows}
        print(f"\n### {work.name}\n")
        print(
            "| arm | rounds | tok/s median [min-max] | s/update median | within-round CV | "
            "MFU | alloc peak GiB | reserved peak GiB | driver peak GiB | SM MHz | status |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, rs in rows.items():
            ok = [r for r in rs if r["status"] == "ok" and r.get("steady")]
            toks = [r["steady"]["tok_s"] for r in ok]
            upd = [r["steady"]["update_s_median"] for r in ok]
            cv = [r["steady"]["update_s_cv"] for r in ok]
            mfu = "-"
            if roof and toks:
                peak = max(roof["square"][rs[0]["precision"]].values()) * 1e12
                mfu = f"{100 * statistics.median(toks) * rs[0]['flops_per_token'] / peak:.1f}%"
            alloc = [r["alloc_peak_bytes"] / 2**30 for r in rs if r.get("alloc_peak_bytes")]
            resv = [r["reserved_peak_bytes"] / 2**30 for r in rs if r.get("reserved_peak_bytes")]
            drv = [r["driver_peak_mib"] / 1024 for r in rs if r.get("driver_peak_mib")]
            mhz = [r["card_steady"]["sm_mhz_mean"] for r in ok if r.get("card_steady")]
            status = ",".join(sorted({r["status"] for r in rs}))
            print(
                f"| {name} | {len(ok)}/{len(rs)} | {_fmt_range(toks, ',.0f')} | "
                f"{_fmt_range(upd, '.3f')} | {_fmt_range(cv, '.3f')} | {mfu} | "
                f"{_fmt_range(alloc, '.2f')} | {_fmt_range(resv, '.2f')} | "
                f"{_fmt_range(drv, '.2f')} | {_fmt_range(mhz, '.0f')} | {status} |"
            )
        if (work / "phases").is_dir():
            print(
                "\n| arm (instrumented) | wall s/update | vs timed median | forward | backward "
                "| optimizer | rest | alloc peak fwd / bwd / opt GiB |"
                "\n|---|---|---|---|---|---|---|---|"
            )
            for f in sorted((work / "phases").glob("*.json")):
                r = json.loads(f.read_text())
                sp = phase_split(r) if r["status"] == "ok" else None
                if sp is None:
                    print(f"| {r['arm']} | {r['status']} | | | | | | |")
                    continue
                timed = [x["steady"]["update_s_median"] for x in rows.get(r["arm"], [])
                         if x["status"] == "ok" and x.get("steady")]  # fmt: skip
                cost = (
                    f"{100 * (sp['wall'] / statistics.median(timed) - 1):+.1f}%" if timed else "-"
                )
                peaks = "-"
                if r.get("phase_peaks"):
                    last = r["phase_peaks"][-1]
                    peaks = " / ".join(f"{last[k] / 2**30:.2f}" for k in ("fwd", "bwd", "opt"))
                print(
                    f"| {r['arm']} | {sp['wall']:.3f} | {cost} | {sp['fwd']:.3f} | "
                    f"{sp['bwd']:.3f} | {sp['opt']:.3f} | {sp['other']:.3f} | {peaks} |"
                )
        report_rounds(rows)
        report_pairs(rows)
        report_nsl_memory(work, rows)
        report_host_profile(work)
    return 0


# Comparisons the write-up quotes, as (numerator arm, denominator arm).
PAIRS = [
    ("torch_tf32_eager_sel", "nsl_tf32"),
    ("torch_bf16_eager_sel", "nsl_bf16"),
    ("torch_bf16_compile_sel", "nsl_bf16"),
    ("torch_bf16_compile_sel", "nsl_tf32"),
    ("nsl_bf16", "nsl_tf32"),
    ("torch_bf16_eager_sel", "torch_tf32_eager_sel"),
    ("torch_bf16_compile_sel", "torch_bf16_eager_sel"),
    ("nsl_tf32", "nsl_tf32_blocks"),
    ("torch_tf32_eager_sel", "torch_tf32_eager_blocks"),
    ("torch_bf16_compile_sel", "torch_bf16_compile_blocks"),
    ("torch_bf16_compile_none", "torch_bf16_compile_sel"),
    ("nsl_tf32_nodrain", "nsl_tf32"),
    ("nsl_bf16_nodrain", "nsl_bf16"),
    ("nsl_tf32_compress", "nsl_tf32"),
    ("nsl_bf16_compress", "nsl_bf16"),
]


def tok_by_round(rs: list[dict[str, Any]]) -> dict[int, float]:
    return {r["round"]: r["steady"]["tok_s"] for r in rs if r["status"] == "ok" and r.get("steady")}


def paired_ratios(
    rows: dict[str, list[dict[str, Any]]], num: str, den: str
) -> list[tuple[int, float]]:
    """tok/s of `num` over `den` within each round both completed, in round order."""
    a, b = tok_by_round(rows.get(num, [])), tok_by_round(rows.get(den, []))
    return [(k, a[k] / b[k]) for k in sorted(set(a) & set(b))]


def report_pairs(rows: dict[str, list[dict[str, Any]]]) -> None:
    print("\n| paired ratio (tok/s, same round) | per round | median |\n|---|---|---|")
    for num, den in PAIRS:
        pr = paired_ratios(rows, num, den)
        if pr:
            per = " ".join(f"r{k}:{v:.3f}" for k, v in pr)
            print(f"| {num} / {den} | {per} | {statistics.median(v for _, v in pr):.3f} |")


def report_rounds(rows: dict[str, list[dict[str, Any]]]) -> None:
    """Each arm's tok/s per round and its deviation from the arm's median."""
    print("\n| arm | tok/s per round (deviation from the arm's median) |\n|---|---|")
    for name, rs in rows.items():
        by = tok_by_round(rs)
        if not by:
            continue
        med = statistics.median(by.values())
        cells = " ".join(
            f"r{k}:{v:,.0f} ({100 * (v / med - 1):+.1f}%)" for k, v in sorted(by.items())
        )
        print(f"| {name} | {cells} |")


def top_contexts_at_peak(stderr_text: str) -> list[tuple[str, str]]:
    """The `NSL_MEMSTATS=1` summary's "Top contexts at peak" rows: (context, size)."""
    block = stderr_text.split("Top contexts at peak:", 1)
    if len(block) < 2:
        return []
    out = []
    for line in block[1].splitlines()[1:]:
        m = re.match(r"\s+(.+?): ([0-9.]+ [KMG]?B) \(\d+ blocks\)", line)
        if m is None:
            break
        out.append((m.group(1), m.group(2)))
    return out


def report_nsl_memory(work: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
    """NSL's allocator surfaces and allocation contexts at its global peak (round 0)."""
    printed = False
    for name, rs in rows.items():
        r0 = next((r for r in rs if r.get("kind") == "nsl" and r["round"] == 0), None)
        log = work / "logs" / f"{name}_r0.stderr"
        if r0 is None or not log.exists():
            continue
        if not printed:
            print("\n| NSL arm, round 0 | surfaces at peak (GiB) | top contexts at peak |"
                  "\n|---|---|---|")  # fmt: skip
            printed = True
        surf = ", ".join(f"{k} {v / 2**30:.2f}" for k, v in r0["surfaces_at_peak"].items() if v)
        ctx = ", ".join(f"{c} {b}" for c, b in top_contexts_at_peak(log.read_text())[:5])
        print(f"| {name} | {surf} | {ctx} |")


def host_profile_summary(stderr_text: str) -> dict[str, float] | None:
    """Medians over the `NSL_HOST_PROFILE=1` per-step reports: attributed ms, and per probe
    the calls and ms per step."""
    steps = stderr_text.split("[host-profile] step:")[1:]
    if not steps:
        return None
    total: list[float] = []
    per: dict[str, list[tuple[float, float]]] = {}
    for st in steps:
        m = re.match(r"\s*([0-9.]+) ms attributed", st)
        if m is None:
            continue
        total.append(float(m.group(1)))
        for pm in re.finditer(r"^\s+(\w+)\s+([0-9.]+) ms\s+(\d+) calls", st, re.MULTILINE):
            per.setdefault(pm.group(1), []).append((float(pm.group(2)), float(pm.group(3))))
    out = {"attributed_ms": statistics.median(total), "steps": float(len(total))}
    for probe, xs in per.items():
        out[f"{probe}_ms"] = statistics.median(x[0] for x in xs)
        out[f"{probe}_calls"] = statistics.median(x[1] for x in xs)
    return out


def report_host_profile(work: Path) -> None:
    for log in sorted((work / "logs").glob("*_diag_hostprof.stderr")):
        hp = host_profile_summary(log.read_text())
        if hp is None:
            continue
        print(f"\n{log.name}: median per micro-batch over {hp['steps']:.0f} steps:")
        for k, v in hp.items():
            if k != "steps":
                print(f"  {k} {v:.1f}")


def cmd_pairs(args: argparse.Namespace) -> int:
    """Paired ratios NUM/DEN (tok/s) for each `--pair NUM/DEN`, per size directory."""
    out = args.out.resolve()
    for work in sorted(p for p in out.iterdir() if (p / "results").is_dir()):
        rows: dict[str, list[dict[str, Any]]] = {}
        for f in sorted((work / "results").glob("*.json")):
            r = json.loads(f.read_text())
            rows.setdefault(r["arm"], []).append(r)
        printed = False
        for spec in args.pair:
            num, sep, den = spec.partition("/")
            if not sep:
                raise SystemExit(f"--pair {spec!r}: want NUM/DEN")
            rs = paired_ratios(rows, num, den)
            if not rs:
                continue
            if not printed:
                print(f"\n### {work.name}\n\n| ratio | rounds | per round | median [min-max] |")
                print("|---|---|---|---|")
                printed = True
            vals = [v for _, v in rs]
            per = " ".join(f"{v:.3f}" for v in vals)
            print(f"| {num} / {den} | {len(vals)} | {per} | {_fmt_range(vals, '.3f')} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", type=Path, required=True, help="work dir on a real disk")
        p.add_argument("--size", choices=sorted(SIZE_DIR), default="1b")
        p.add_argument("--micro-batch", type=int)
        p.add_argument("--seq", type=int)
        p.add_argument("--accum", type=int)
        p.add_argument("--updates", type=int, default=24, help="optimizer updates per run")
        p.add_argument("--warmup", type=int, default=8, help="updates discarded per run")
        p.add_argument("--timeout", type=float, default=840.0)
        p.add_argument("--nsl", type=Path, default=REPO / "target" / "release" / "nsl")
        p.add_argument(
            "--toolchain",
            action="append",
            default=[],
            help="NAME=DIR: build arms named <arm>@NAME with DIR/bin/nsl and DIR/stdlib",
        )
        p.add_argument("--python", default=sys.executable)

    p = sub.add_parser("roofline")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nsl", type=Path, default=REPO / "target" / "release" / "nsl")
    p = sub.add_parser("workspace")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("build")
    common(p)
    p.add_argument("--arms", default="nsl_tf32")
    p.add_argument("--phase", action="store_true", help="also build the NSL_PHASE_TIMING twin")
    for name in ("run", "phases"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--arms", required=True)
        p.add_argument("--round", type=int, default=0)
    p = sub.add_parser("nsys")
    common(p)
    p.add_argument("--arm", required=True)
    p.add_argument("--phase", action="store_true")
    p = sub.add_parser("diag")
    common(p)
    p.add_argument("--arm", required=True)
    p.add_argument("--phase", action="store_true")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the run")
    p.add_argument("--tag", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair", action="append", required=True, help="NUM/DEN arm names")
    args = ap.parse_args()
    if args.cmd == "roofline":
        return cmd_roofline(args)
    if args.cmd == "workspace":
        return cmd_workspace(args)
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "phases":
        return cmd_run(args, phase=True)
    if args.cmd == "nsys":
        return cmd_nsys(args)
    if args.cmd == "diag":
        return cmd_diag(args)
    if args.cmd == "pairs":
        return cmd_pairs(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
