#!/usr/bin/env python3
"""Fusion-campaign measurement harness (mfu-fusion master plan, section 6).

Measures the campaign's six Pass-A throughput arms on the 500M prod posture,
the Pass-B busy-vs-span host-gap split for A0/A1/A3, and the free NSL_EVENTS
signals — with resident_bench.py's proven discipline: OPT_STEP wall-arrival
stamps, warmup discard, interleaved round-robin rounds with best-round
reporting, gpu_guard refuse-don't-warn, and instrumented passes whose
timings are DISCARDED from throughput.

BUILD MATRIX — six builds, not four
-----------------------------------
The campaign brief sketched four builds (B0..B3) with the fusion
kill-switches toggled per RUN. That is wrong for this system: the switches
are read at COMPILE time inside codegen — verified against source in this
worktree: NSL_FUSE_ROPE_NEG is read during adjoint generation
(crates/nsl-codegen/src/source_ad.rs:1493), exactly like the shipped
NSL_FUSE_NORM_RESIDUAL precedent (source_ad.rs:1602), and the master plan
(sections 2-3) specifies NSL_FUSE_ELEMENTWISE_BWD / NSL_FUSE_SCALAR_IMM as
first-line env checks of compile-time tape passes. An env set at run time
changes NOTHING. The six arms therefore need six builds, with the envs
applied at `nsl build` time, and the runs carry no fusion envs at all:

    build   flags                                     build-time env
    b0_off  prod (--source-ad --checkpoint-blocks     3 fusion switches = 0
            --fuse-rmsnorm-backward)
    b0_on   prod                                      switches UNSET (on)
    b1_off  prod + --fuse-wgrad-accum                 = 0
    b1_on   prod + --fuse-wgrad-accum                 unset
    b2_on   prod + --fuse-wgrad-accum --cuda-graphs   unset
    b3_off  prod + --cuda-graphs                      = 0

    arm  build   what
    A0   b0_off  baseline: prod posture, fusion OFF
    A1   b0_on   elementwise/scalar/rope fusion delta (one-axis A/B vs A0)
    A2   b1_off  wgrad accumulation fusion alone
    A3   b1_on   wgrad + fusion (the recommended-posture candidate)
    A4   b2_on   + cuda-graphs on the fused tape; NSL_CUDA_GRAPH_LOG=1 on
                 the first round (r0) only (backward-region taint evidence)
    A5   b3_off  cuda-graphs alone on the unfused tape (attribution arm)

Builds NEVER see NSL_PHASE_TIMING (compile-time knob — it bakes syncs into
the binary) and the ON builds actively UNSET the fusion switches so a stray
shell export cannot silently produce six identical binaries.

UPDATE LIMITING — a generated program, not the recipe binary
------------------------------------------------------------
models/coder500m/pretrain_prod.nsl has no step-limit argument: it trains
epochs=1 over the whole corpus slice (512 optimizer updates), writes
checkpoints every 100 updates, and ends with a held-out validation pass —
and its loss cadence is every 20 MICRO-steps, so it never prints a
per-update marker to stamp. This harness therefore mirrors how
resident_bench.py limits the 1B recipe: it generates bench.nsl in the
recipe's exact posture (geometry, fused-CE hints, optimizer, scheduler
shape, grad_clip — parsed OUT OF the recipe file, with tokens-per-update
asserted == 16384, refusing on mismatch) minus checkpointing and
validation, prints OPT_STEP once per optimizer update, and sizes the token
stream so epochs=1 ends after --updates optimizer updates.

Pass B runs 10 updates on the SAME binary by swapping the baked tokens file
(the path is compiled in; the CONTENTS are read at run time) for a
10-update stream — no extra build, and no wall-clock kill that would skip
the profiler's atexit flush of kernel_profile.json. Pass A always re-copies
the full stream first, so an interrupted campaign cannot leave a profile
stream under a timing arm.

GPU protocol
------------
gpu_guard.acquire_or_refuse gates every GPU run (refuse-don't-warn); it is
taken AFTER the builds (compile-only, no GPU) so binaries can be prepared
while the device is busy and reused later via --skip-build. Children run
under setsid (their own process group) and the watchdog kills the GROUP —
an orphaned child holding VRAM is the documented failure mode. Pass-A
rounds carry only NSL_EVENTS; profiler/host-profile envs are scrubbed from
the inherited environment so they cannot contaminate a timing round.

Usage (from models/benchmarks/):
    python3 fusion_bench.py --nsl-bin /path/to/nsl --out-dir /path/on/real/disk
        [--rounds 3] [--updates 30] [--warmup 10] [--arms A0,A1,...]
        [--skip-build] [--dry-run] [--timeout 1800]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_tokens
import gpu_guard
from matrix_bench import PEAK_TFLOPS_BY_DEVICE, SCALES

REPO = Path(__file__).resolve().parents[2]
CODER500M = REPO / "models/coder500m"
RECIPE = CODER500M / "pretrain_prod.nsl"
SCALE = SCALES["500m"]
STDLIB = str(REPO / "stdlib")

# 2 * 1024 * 8 — asserted against the recipe's own text, refused on mismatch.
EXPECTED_TOKENS_PER_UPDATE = 16384
# Pass-B length and event-pool size (>= 2 full micro-batches at 500M).
PROFILE_UPDATES = 10
PROFILE_POOL = "16384"

BASE_FLAGS: tuple[str, ...] = (
    "--source-ad",
    "--checkpoint-blocks",
    "--fuse-rmsnorm-backward",
)

FUSION_ENV_KEYS: tuple[str, ...] = (
    "NSL_FUSE_ELEMENTWISE_BWD",
    "NSL_FUSE_SCALAR_IMM",
    "NSL_FUSE_ROPE_NEG",
)
FUSION_OFF: tuple[tuple[str, str], ...] = tuple((k, "0") for k in FUSION_ENV_KEYS)

# Never inherited into a BUILD: the fusion switches (each build sets its own),
# NSL_PHASE_TIMING (compile-time — bakes syncs in), and the rmsnorm-fold kill
# (the fold is part of the fixed baseline posture, not an axis of this bench).
BUILD_SCRUB: tuple[str, ...] = (
    *FUSION_ENV_KEYS,
    "NSL_PHASE_TIMING",
    "NSL_FUSE_NORM_RESIDUAL",
    "NSL_PASS_TRACE",
    "NSL_PROFILE_ADJOINT",
)
# Never inherited into a RUN: instrumentation belongs only to the pass that
# asks for it (fusion envs are compile-time and inert here, scrubbed anyway).
RUN_SCRUB: tuple[str, ...] = (
    *FUSION_ENV_KEYS,
    "NSL_PHASE_TIMING",
    "NSL_FUSE_NORM_RESIDUAL",
    "NSL_PASS_TRACE",
    "NSL_PROFILE_ADJOINT",
    "NSL_PROFILE_KERNELS",
    "NSL_PROFILE_KERNELS_POOL",
    "NSL_HOST_PROFILE",
    "NSL_CUDA_GRAPH_LOG",
    "NSL_EVENTS",
    "NSL_MEMSTATS",
)

MARKER_RMSNORM = "[fuse] rmsnorm dx+residual folds:"
RE_RMSNORM = re.compile(r"\[fuse\] rmsnorm dx\+residual folds: (\d+)")
RE_WGRAD = re.compile(r"\[wgrad-fusion\] (\d+) chain\(s\) fused")
RE_EW_CHAINS = re.compile(r"\[fuse\] elementwise backward chains: (\d+)")
RE_SCALAR_IMM = re.compile(r"\[fuse\] scalar immediates: (\d+)")
RE_ROPE = re.compile(r"\[fuse\] rope backward folds: (\d+)")

OOM_MARKERS = ("CUDA_ERROR_OUT_OF_MEMORY", "VRAM free", "Requested:", "out of memory")

TEMPLATE = """\
# GENERATED by models/benchmarks/fusion_bench.py — do not commit.
# The models/coder500m/pretrain_prod.nsl posture minus checkpointing and
# held-out validation: pure steady-state training throughput. Fused-CE hints
# and reshape dims are compile-time literals (a runtime logits.shape read
# refuses the fusion). OPT_STEP prints once per optimizer update — its
# stdout ARRIVAL stamp is the tok/s clock.
from model import NSLCoder
from nsl.nn.losses import cross_entropy

let m = NSLCoder()
m.to(cuda)

let tokens = load_mmap("TOKENS_PATH", 3)
let loader = DataLoader(tokens, batch_size=MICRO_BATCH, seq_len=SEQ_LEN, shuffle=true, drop_last=true)

print("LOSS_STREAM_BEGIN")

@fused_lm_ce(enabled=true, vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, batch_size=MICRO_BATCH, seq_len=SEQ_LEN)
train(model=m, epochs=1, grad_accumulation=ACCUM, grad_clip=GRAD_CLIP):
    optimizer: OPTIMIZER_SPEC
    scheduler: warmup_cosine(warmup_steps=WARMUP_MICRO, total_steps=TOTAL_MICRO, min_lr=MIN_LR)
    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let flat_logits = logits.reshape([FLAT_ROWS, VOCAB_SIZE])
        let flat_labels = batch.labels.reshape([FLAT_ROWS])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            if step % ACCUM == 0:
                print("OPT_STEP")
                print(step)
                print(loss)

print("LOSS_STREAM_END")
print("FUSION_BENCH_COMPLETE")
"""


@dataclasses.dataclass(frozen=True)
class Build:
    """One `nsl build` invocation: flags plus COMPILE-time fusion env."""

    id: str
    flags: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    fusion_on: bool
    wgrad: bool
    graphs: bool


BUILDS = {
    b.id: b
    for b in [
        Build("b0_off", BASE_FLAGS, FUSION_OFF, False, False, False),
        Build("b0_on", BASE_FLAGS, (), True, False, False),
        Build(
            "b1_off",
            (*BASE_FLAGS, "--fuse-wgrad-accum"),
            FUSION_OFF,
            False,
            True,
            False,
        ),
        Build("b1_on", (*BASE_FLAGS, "--fuse-wgrad-accum"), (), True, True, False),
        Build(
            "b2_on",
            (*BASE_FLAGS, "--fuse-wgrad-accum", "--cuda-graphs"),
            (),
            True,
            True,
            True,
        ),
        Build("b3_off", (*BASE_FLAGS, "--cuda-graphs"), FUSION_OFF, False, False, True),
    ]
}


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    key: str
    build_id: str
    what: str
    #: Pass-B wall-split member (A0/A1/A3 only; graphs arms are excluded —
    #: the per-launch profiler refuses to compose with graph replay).
    profiled: bool = False

    @property
    def build(self) -> Build:
        return BUILDS[self.build_id]


ARMS = {
    a.key: a
    for a in [
        ArmSpec("A0", "b0_off", "baseline: prod posture, fusion OFF", profiled=True),
        ArmSpec("A1", "b0_on", "fusion delta (one-axis A/B vs A0)", profiled=True),
        ArmSpec("A2", "b1_off", "wgrad accumulation fusion alone"),
        ArmSpec(
            "A3",
            "b1_on",
            "wgrad + fusion (recommended-posture candidate)",
            profiled=True,
        ),
        ArmSpec(
            "A4", "b2_on", "+ cuda-graphs on the fused tape (graph log on round 1)"
        ),
        ArmSpec("A5", "b3_off", "cuda-graphs alone on the unfused tape (attribution)"),
    ]
}


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The recipe's training geometry, parsed out of pretrain_prod.nsl."""

    micro_batch: int
    seq_len: int
    accum: int
    vocab: int
    hidden: int
    grad_clip: str
    optimizer_spec: str
    min_lr: str

    @property
    def tokens_per_update(self) -> int:
        return self.micro_batch * self.seq_len * self.accum


@dataclasses.dataclass
class RoundResult:
    round: int
    tok_s: float
    wall_s: float
    opt_steps_measured: int
    kernel_launch_count: int | None = None
    gpu_launch_count: int | None = None
    fused_ew: dict[str, int] | None = None


@dataclasses.dataclass
class EventSignals:
    """The free NSL_EVENTS signals of one run (item 17 stream)."""

    kernel_launch_count: int | None = None
    gpu_launch_count: int | None = None
    fused_ew: dict[str, int] | None = None
    alloc_first_mid_last: list[int] | None = None


@dataclasses.dataclass
class KernelProfile:
    """Busy-vs-span split over one kernel_profile.json trace window."""

    host_gap_pct: float
    busy_ms: float
    span_ms: float
    n_events: int
    #: (name, launches, total_ms) sorted by total_ms descending.
    table: list[tuple[str, int, float]]


@dataclasses.dataclass
class ArmResult:
    spec: ArmSpec
    rounds: list[RoundResult] = dataclasses.field(default_factory=list)
    oom: bool = False
    error: str | None = None
    alloc_first_mid_last: list[int] | None = None
    host_gap_pct: float | None = None
    profile_busy_ms: float | None = None
    profile_span_ms: float | None = None
    profile_error: str | None = None
    kernel_table: list[tuple[str, int, float]] = dataclasses.field(default_factory=list)
    violations: list[str] = dataclasses.field(default_factory=list)

    @property
    def best(self) -> RoundResult | None:
        return max(self.rounds, key=lambda r: r.tok_s) if self.rounds else None

    @property
    def spread_pct(self) -> float | None:
        if len(self.rounds) < 2:
            return None
        tok = [r.tok_s for r in self.rounds]
        return 100.0 * (max(tok) - min(tok)) / max(tok)


def parse_recipe(recipe: Path) -> Geometry:
    """Parse the geometry OUT OF the recipe and refuse on any mismatch.

    tokens/update is not hard-coded on trust: batch*seq*accum is read from
    the recipe's own DataLoader/train lines and must equal 16384, and the
    fused-CE hints must agree with the DataLoader and with matrix_bench's
    500m Scale (the MFU flops model) — a drifted recipe would otherwise
    silently mis-normalize every headline number.
    """
    text = recipe.read_text()
    loader = re.search(
        r"let loader = DataLoader\([^)]*batch_size=(\d+), seq_len=(\d+)", text
    )
    train = re.search(
        r"^train\(model=[^)]*grad_accumulation=(\d+), grad_clip=([0-9.]+)",
        text,
        re.MULTILINE,
    )
    fused = re.search(
        r"@fused_lm_ce\(enabled=true, vocab_size=(\d+), hidden_size=(\d+), "
        r"batch_size=(\d+), seq_len=(\d+)\)",
        text,
    )
    optim = re.search(r"optimizer: (AdamW\([^)]*\))", text)
    min_lr = re.search(r"min_lr=([0-9.e+-]+)", text)
    if not (loader and train and fused and optim and min_lr):
        sys.exit(
            f"cannot parse {recipe} — the geometry/optimizer lines moved; refusing to guess"
        )
    batch, seq = int(loader.group(1)), int(loader.group(2))
    accum, grad_clip = int(train.group(1)), train.group(2)
    if (int(fused.group(3)), int(fused.group(4))) != (batch, seq):
        sys.exit(
            f"{recipe}: @fused_lm_ce hints disagree with the DataLoader — refusing"
        )
    geom = Geometry(
        micro_batch=batch,
        seq_len=seq,
        accum=accum,
        vocab=int(fused.group(1)),
        hidden=int(fused.group(2)),
        grad_clip=grad_clip,
        optimizer_spec=optim.group(1),
        min_lr=min_lr.group(1),
    )
    if geom.tokens_per_update != EXPECTED_TOKENS_PER_UPDATE:
        sys.exit(
            f"{recipe}: batch*seq*accum = {geom.tokens_per_update}, expected "
            f"{EXPECTED_TOKENS_PER_UPDATE} — the recipe geometry moved; refusing"
        )
    if geom.vocab != SCALE.vocab_size or geom.hidden != SCALE.d_model:
        sys.exit(
            f"{recipe}: vocab/hidden ({geom.vocab}/{geom.hidden}) disagree with "
            f"matrix_bench SCALES['500m'] ({SCALE.vocab_size}/{SCALE.d_model}) — "
            "the MFU flops model would be wrong; refusing"
        )
    return geom


def render_program(geom: Geometry, tokens_path: Path, updates: int) -> str:
    repl = {
        "TOKENS_PATH": str(tokens_path),
        "MICRO_BATCH": str(geom.micro_batch),
        "SEQ_LEN": str(geom.seq_len),
        "VOCAB_SIZE": str(geom.vocab),
        "HIDDEN_SIZE": str(geom.hidden),
        "ACCUM": str(geom.accum),
        "GRAD_CLIP": geom.grad_clip,
        "OPTIMIZER_SPEC": geom.optimizer_spec,
        "WARMUP_MICRO": str(2 * geom.accum),
        "TOTAL_MICRO": str(updates * geom.accum),
        "MIN_LR": geom.min_lr,
        "FLAT_ROWS": str(geom.micro_batch * geom.seq_len),
    }
    prog = TEMPLATE
    for key, value in repl.items():
        prog = prog.replace(key, value)
    return prog


def build_env(build: Build) -> dict[str, str]:
    env = dict(os.environ)
    for key in BUILD_SCRUB:
        env.pop(key, None)
    env.setdefault("NSL_STDLIB_PATH", STDLIB)
    env.update(dict(build.env))
    return env


def run_env(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for key in RUN_SCRUB:
        env.pop(key, None)
    env.setdefault("NSL_STDLIB_PATH", STDLIB)
    env.update(extra)
    return env


def stdlib_path() -> str:
    """The EFFECTIVE stdlib path: build_env/run_env use setdefault, so a
    pre-existing export wins — the dry-run rendering and the results
    provenance must say the same thing the real envs do."""
    return os.environ.get("NSL_STDLIB_PATH", STDLIB)


def render_cmd(
    argv: list[str],
    cwd: Path,
    set_env: dict[str, str],
    unset: tuple[str, ...] = (),
) -> str:
    """One copy-pasteable line for --dry-run, env scrubs included."""
    parts = ["env"]
    parts += [f"-u {k}" for k in unset if k not in set_env]
    parts += [f"{k}={v}" for k, v in set_env.items()]
    parts += argv
    return f"(cd {cwd} && {' '.join(parts)})"


def do_build(
    build: Build, out: Path, geom: Geometry, updates: int, nsl: Path, timeout: int
) -> Path:
    """Generate the program, copy model/config beside it, `nsl build` once.

    stderr is persisted to <out>/build_<id>.stderr — it carries the fusion
    markers the campaign asserts on. The tokens path is baked as the build
    dir's tokens.bin; its contents are swapped per pass.
    """
    d = out / f"build_{build.id}"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("model.nsl", "config.nsl"):
        shutil.copy2(CODER500M / name, d / name)
    src = d / "bench.nsl"
    src.write_text(render_program(geom, d / "tokens.bin", updates))
    binary = d / "prog"
    try:
        r = subprocess.run(
            [str(nsl), "build", *build.flags, str(src), "-o", str(binary)],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=build_env(build),
            check=False,  # returncode handled below with the persisted stderr
        )
    except subprocess.TimeoutExpired as e:
        partial = e.stderr if isinstance(e.stderr, str) else ""
        (out / f"build_{build.id}.stderr").write_text(partial)
        raise RuntimeError(f"{build.id}: build timed out after {timeout}s") from e
    (out / f"build_{build.id}.stderr").write_text(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{build.id}: build failed:\n{r.stderr[-2000:]}")
    return binary


def check_build_markers(
    build: Build, stderr: str
) -> tuple[dict[str, int | None], list[str]]:
    """Assert the expected compile-time markers; collect the counts.

    Hard failures (raise): rmsnorm fold marker absent anywhere; wgrad builds
    without a `[wgrad-fusion] N chain(s) fused` line with N > 0; fused
    LM-CE not engaging (resident_bench's refusal — an unfused binary runs at
    a 384 MiB/step handicap and benchmarks the wrong thing); a fusion
    marker with N > 0 on an OFF build (the kill-switch did not disarm — the
    A/B would attribute nothing). Soft (warn): the new elementwise/scalar/
    rope markers absent on an ON build — that side of the campaign may not
    have landed in the compiler yet.
    """
    warnings: list[str] = []
    counts: dict[str, int | None] = {}
    if MARKER_RMSNORM not in stderr:
        raise RuntimeError(
            f"{build.id}: '{MARKER_RMSNORM}' missing from build stderr — the "
            "--fuse-rmsnorm-backward posture did not engage"
        )
    rms = RE_RMSNORM.search(stderr)
    counts["rmsnorm_folds"] = int(rms.group(1)) if rms else None
    if build.wgrad:
        wg = RE_WGRAD.search(stderr)
        if not wg or int(wg.group(1)) <= 0:
            raise RuntimeError(
                f"{build.id}: expected '[wgrad-fusion] N chain(s) fused' with "
                f"N > 0 in build stderr (got: "
                f"{wg.group(0) if wg else 'no fused-count line'})"
            )
        counts["wgrad_fused"] = int(wg.group(1))
    else:
        counts["wgrad_fused"] = None
    if "[fused-lce] forward route=" not in stderr or "will NOT run" in stderr:
        raise RuntimeError(
            f"{build.id}: fused LM-CE did not engage at build time — refusing "
            "to benchmark an unfused binary"
        )
    for name, rx in (
        ("ew_chains", RE_EW_CHAINS),
        ("scalar_imms", RE_SCALAR_IMM),
        ("rope_folds", RE_ROPE),
    ):
        m = rx.search(stderr)
        counts[name] = int(m.group(1)) if m else None
        if build.fusion_on and m is None:
            warnings.append(
                f"{build.id}: '{name}' marker absent on an ON build — the "
                "codegen side may not have landed yet"
            )
        if not build.fusion_on and m is not None and int(m.group(1)) > 0:
            raise RuntimeError(
                f"{build.id}: fusion marker {name}={m.group(1)} on an OFF "
                "build — the compile-time kill-switch did not disarm"
            )
    return counts, warnings


def run_binary(
    binary: Path, env: dict[str, str], timeout: int, log: Path
) -> tuple[int, list[tuple[float, str]]]:
    """Run, stamping each stdout line's arrival time (the tok/s clock).

    The stamped lines are persisted beside the stderr log. The child gets
    its own session/process group (setsid semantics) and the watchdog kills
    the GROUP — killing only the visible pid leaves children holding VRAM.
    """
    lines: list[tuple[float, str]] = []
    with log.open("w") as lf:
        p = subprocess.Popen(
            [str(binary)],
            cwd=binary.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=lf,
            text=True,
            errors="replace",  # a stray non-UTF8 byte must not kill the reader
            start_new_session=True,
        )
        assert p.stdout is not None

        def kill_group() -> None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        watchdog = threading.Timer(timeout, kill_group)
        watchdog.start()
        try:
            for line in p.stdout:
                lines.append((time.monotonic(), line.rstrip("\n")))
        except BaseException:
            # A reader failure must not strand a child holding VRAM (and the
            # GPU lock) — kill the group before the final wait.
            kill_group()
            raise
        finally:
            watchdog.cancel()
            p.wait()
    log.with_suffix(".stdout").write_text("".join(f"{t:.6f}\t{s}\n" for t, s in lines))
    return p.returncode, lines


def parse_timing(
    lines: list[tuple[float, str]], warmup: int, tokens_per_update: int, rnd: int
) -> RoundResult | None:
    """tok/s over the steady-state window: OPT_STEP arrival stamps after
    warmup. Cadence-independent — tokens are counted, not inferred."""
    stamps = [t for t, s in lines if s == "OPT_STEP"]
    if len(stamps) <= warmup + 2:
        return None
    window = stamps[warmup:]
    wall = window[-1] - window[0]
    steps = len(window) - 1
    if wall <= 0 or steps == 0:
        return None
    return RoundResult(
        round=rnd,
        tok_s=steps * tokens_per_update / wall,
        wall_s=wall,
        opt_steps_measured=steps,
    )


def parse_events(path: Path) -> EventSignals:
    """Free signals from the NSL_EVENTS stream — tolerant parse, last event
    of a counter kind wins (they are teardown counters)."""
    sig = EventSignals()
    if not path.exists():
        return sig
    alloc: dict[int, int] = {}
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        kind = ev.get("kind")
        fields = ev.get("fields")
        if not isinstance(fields, dict):
            continue
        if kind == "kernel_launch_count" and isinstance(fields.get("count"), int):
            sig.kernel_launch_count = fields["count"]
        elif kind == "gpu_launch_count" and isinstance(fields.get("count"), int):
            sig.gpu_launch_count = fields["count"]
        elif kind == "fused_ew_counters":
            sig.fused_ew = {k: v for k, v in fields.items() if isinstance(v, int)}
        elif kind == "gpu_mem_step":
            step, v = ev.get("step"), fields.get("allocated_bytes")
            if isinstance(step, int) and isinstance(v, int):
                # First event carrying step k is the post-cleanup END event
                # of micro k-1 (interior steps emit twice).
                alloc.setdefault(step, v)
    if alloc:
        series = [alloc[k] for k in sorted(alloc)]
        sig.alloc_first_mid_last = [series[0], series[len(series) // 2], series[-1]]
    return sig


def parse_kernel_profile(path: Path) -> KernelProfile | str:
    """Busy-vs-span split from the Chrome trace, or a refusal string.

    metadata.timing_valid == false means every duration is an unresolved
    0.0 — the file is refused, not averaged.
    """
    if not path.exists():
        return f"{path.name} not written (run died before the atexit flush?)"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return f"unparseable trace: {e}"
    if not isinstance(data, dict):
        return "unexpected trace shape (not an object)"
    meta = data.get("metadata")
    if not isinstance(meta, dict) or meta.get("timing_valid") is not True:
        return "metadata.timing_valid is not true — durations unresolved; file refused"
    events = data.get("traceEvents")
    if not isinstance(events, list):
        return "no traceEvents array"
    busy_us = 0.0
    t0, t1 = math.inf, -math.inf
    agg: dict[str, tuple[int, float]] = {}
    n = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ts, dur = ev.get("ts"), ev.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        n += 1
        busy_us += dur
        t0 = min(t0, float(ts))
        t1 = max(t1, float(ts) + float(dur))
        name = str(ev.get("name", "?"))
        count, total = agg.get(name, (0, 0.0))
        agg[name] = (count + 1, total + float(dur))
    if n == 0 or t1 <= t0:
        return "empty trace window"
    span_us = t1 - t0
    table = sorted(
        ((name, count, total / 1e3) for name, (count, total) in agg.items()),
        key=lambda row: -row[2],
    )
    return KernelProfile(
        host_gap_pct=100.0 * (1.0 - busy_us / span_us),
        busy_ms=busy_us / 1e3,
        span_ms=span_us / 1e3,
        n_events=n,
        table=table,
    )


def ensure_tokens(path: Path, n_tokens: int, geom: Geometry) -> None:
    if not path.exists() or path.stat().st_size != n_tokens * 2:
        gen_tokens.generate(path, n_tokens, geom.vocab, geom.tokens_per_update)


def detect_device() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,  # a failure gets a friendly refusal below
    )
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit("nvidia-smi failed — cannot name the device, cannot pick a roofline")
    return r.stdout.strip().splitlines()[0]


def launch_counts_of(res: ArmResult) -> dict[str, object] | None:
    """The launch counters of the best round (falling back to the latest
    round that carried any)."""
    candidates = ([res.best] if res.best else []) + list(reversed(res.rounds))
    for r in candidates:
        if r and (
            r.kernel_launch_count is not None
            or r.gpu_launch_count is not None
            or r.fused_ew is not None
        ):
            return {
                "round": r.round,
                "kernel_launch_count": r.kernel_launch_count,
                "gpu_launch_count": r.gpu_launch_count,
                "fused_ew": r.fused_ew,
            }
    return None


def arm_table(
    arms: list[ArmSpec],
    results: dict[str, ArmResult],
    flops_per_token: float,
    roofline: float,
) -> list[str]:
    rows = [
        (
            "| arm | build | tok/s (best) | MFU % | spread % | gpu launches | "
            "kernel launches | fused_ew | host_gap % |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        res = results[a.key]
        if res.oom:
            rows.append(
                f"| {a.key} | {a.build_id} | OOM (a result, not a failure) "
                f"| — | — | — | — | — | — |"
            )
            continue
        best = res.best
        if best is None:
            why = "FAILED" if res.error else "no timing"
            rows.append(f"| {a.key} | {a.build_id} | {why} | — | — | — | — | — | — |")
            continue
        # An arm that timed some rounds and errored later still shows its
        # best round — the error is flagged, not allowed to hide the data.
        arm_cell = f"{a.key} (late-round error)" if res.error else a.key
        mfu = 100.0 * best.tok_s * flops_per_token / roofline
        spread = f"{res.spread_pct:.2f}" if res.spread_pct is not None else "—"
        lc = launch_counts_of(res) or {}
        gpu_n = lc.get("gpu_launch_count")
        ker_n = lc.get("kernel_launch_count")
        few = lc.get("fused_ew")
        few_s = (
            f"launches={few.get('launches', '?')} fallbacks={few.get('fallbacks', '?')}"
            if isinstance(few, dict)
            else "absent"
        )
        if res.host_gap_pct is not None:
            gap = f"{res.host_gap_pct:.1f}"
        elif res.profile_error:
            gap = "refused"
        else:
            gap = "—"
        rows.append(
            f"| {arm_cell} | {a.build_id} | {best.tok_s:,.0f} | {mfu:.2f} | {spread} "
            f"| {gpu_n if gpu_n is not None else '—'} "
            f"| {ker_n if ker_n is not None else '—'} | {few_s} | {gap} |"
        )
    return rows


def build_table(
    builds: list[Build], markers: dict[str, dict[str, int | None]]
) -> list[str]:
    rows = [
        (
            "| build | flags | build env | wgrad fused | rmsnorm folds | ew chains "
            "| scalar imms | rope folds |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in builds:
        m = markers.get(b.id, {})

        def cell(key: str, m: dict[str, int | None] = m) -> str:
            v = m.get(key)
            return str(v) if v is not None else "—"

        env_s = " ".join(f"{k}={v}" for k, v in b.env) or "(fusion switches unset)"
        rows.append(
            f"| {b.id} | {' '.join(b.flags)} | {env_s} | {cell('wgrad_fused')} "
            f"| {cell('rmsnorm_folds')} | {cell('ew_chains')} "
            f"| {cell('scalar_imms')} | {cell('rope_folds')} |"
        )
    return rows


def render_results_md(
    arms: list[ArmSpec],
    results: dict[str, ArmResult],
    builds: list[Build],
    markers: dict[str, dict[str, int | None]],
    warnings: list[str],
    device: str,
    flops_per_token: float,
    roofline: float,
    geom: Geometry,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = [
        "# Fusion campaign — Pass A/B measurement (fusion_bench.py)",
        "",
        (
            f"Device: {device} · roofline tf32 {roofline / 1e12:.1f} TFLOPS · "
            f"flops/token {flops_per_token:.3e} (500m @ seq {geom.seq_len}) · "
            f"{geom.tokens_per_update} tokens/update"
        ),
        (
            f"Protocol: {args.updates} requested updates, first {args.warmup} "
            f"discarded, {args.rounds} rounds interleaved round-robin, best round "
            f"reported. Pass B: {PROFILE_UPDATES} updates, "
            f"NSL_PROFILE_KERNELS_POOL={PROFILE_POOL}; profiler timings are NEVER "
            f"included in throughput."
        ),
        "",
        "## Arms",
        "",
        *arm_table(arms, results, flops_per_token, roofline),
        "",
        "## Builds (compile-time markers from build_<id>.stderr)",
        "",
        *build_table(builds, markers),
        "",
    ]
    profiled = [a for a in arms if a.profiled and results[a.key].kernel_table]
    if profiled:
        lines += ["## Pass B — per-kernel aggregate (top 12 by total time)", ""]
        for a in profiled:
            res = results[a.key]
            lines += [
                (
                    f"### {a.key} (busy {res.profile_busy_ms:.0f} ms / "
                    f"span {res.profile_span_ms:.0f} ms → "
                    f"host_gap {res.host_gap_pct:.1f}%)"
                ),
                "",
                "| kernel | launches | total ms |",
                "|---|---|---|",
            ]
            lines += [
                f"| {name} | {count} | {total:.2f} |"
                for name, count, total in res.kernel_table[:12]
            ]
            lines.append("")
    violations = [f"{k}: {v}" for k, r in results.items() for v in r.violations]
    if violations:
        lines += ["## VIOLATIONS", "", *[f"- {v}" for v in violations], ""]
    if warnings:
        lines += ["## Warnings", "", *[f"- {w}" for w in warnings], ""]
    if "A4" in results and not results["A4"].error:
        lines += [
            "## Notes",
            "",
            (
                "- A4's cuda-graph capture/taint log (`NSL_CUDA_GRAPH_LOG=1`, "
                "first round only) is in `A4_r0.stderr` — the backward-region "
                "taint excerpt is the follow-up PR's specification."
            ),
            "",
        ]
    return "\n".join(lines)


def dry_run(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    builds: list[Build],
    geom: Geometry,
    out: Path,
    nsl: Path,
    n_main: int,
    n_prof: int,
) -> int:
    """Print every command — builds included — without executing anything."""
    p = print
    p("[dry-run] fusion_bench: nothing below is executed")
    p(
        f"[dry-run] recipe geometry (parsed from {RECIPE}): "
        f"batch={geom.micro_batch} seq={geom.seq_len} accum={geom.accum} -> "
        f"{geom.tokens_per_update} tokens/update (asserted == "
        f"{EXPECTED_TOKENS_PER_UPDATE})"
    )
    p("[dry-run] gpu_guard.acquire_or_refuse('fusion_bench') would gate all runs")
    p(
        f"[dry-run] tokens: gen_tokens.generate({out / 'tokens_main.bin'}, "
        f"n_tokens={n_main}, vocab={geom.vocab}, block={geom.tokens_per_update})"
    )
    p(
        f"[dry-run] tokens: gen_tokens.generate({out / 'tokens_profile.bin'}, "
        f"n_tokens={n_prof}, vocab={geom.vocab}, block={geom.tokens_per_update})"
    )
    for b in builds:
        d = out / f"build_{b.id}"
        if args.skip_build:
            p(f"[dry-run] build {b.id}: skipped (--skip-build), expect {d / 'prog'}")
            continue
        p(
            f"[dry-run] write {d / 'bench.nsl'} (pretrain_prod posture, {args.updates} updates)"
        )
        p(
            "[dry-run] build "
            + b.id
            + ": "
            + render_cmd(
                [
                    str(nsl),
                    "build",
                    *b.flags,
                    str(d / "bench.nsl"),
                    "-o",
                    str(d / "prog"),
                ],
                d,
                {"NSL_STDLIB_PATH": stdlib_path(), **dict(b.env)},
                BUILD_SCRUB,
            )
            + f"  # stderr -> {out / f'build_{b.id}.stderr'}"
        )
    for b in builds:
        p(
            f"[dry-run] copy {out / 'tokens_main.bin'} -> {out / f'build_{b.id}' / 'tokens.bin'}"
        )
    for rnd in range(args.rounds):
        for a in arms:
            d = out / f"build_{a.build_id}"
            extra = {"NSL_EVENTS": str(out / f"{a.key}_r{rnd}.events")}
            if a.key == "A4" and rnd == 0:
                extra["NSL_CUDA_GRAPH_LOG"] = "1"
            p(
                f"[dry-run] pass A round {rnd} {a.key}: "
                + render_cmd(
                    [str(d / "prog")],
                    d,
                    {"NSL_STDLIB_PATH": stdlib_path(), **extra},
                    RUN_SCRUB,
                )
                + f"  # stderr -> {out / f'{a.key}_r{rnd}.stderr'}"
            )
    for a in arms:
        if not a.profiled:
            continue
        d = out / f"build_{a.build_id}"
        p(f"[dry-run] copy {out / 'tokens_profile.bin'} -> {d / 'tokens.bin'}")
        p(
            f"[dry-run] pass B {a.key} ({PROFILE_UPDATES} updates, timings discarded): "
            + render_cmd(
                [str(d / "prog")],
                d,
                {
                    "NSL_STDLIB_PATH": stdlib_path(),
                    "NSL_PROFILE_KERNELS": "1",
                    "NSL_PROFILE_KERNELS_POOL": PROFILE_POOL,
                },
                RUN_SCRUB,
            )
            + f"  # parses {d / 'kernel_profile.json'}"
        )
    p(f"[dry-run] would write {out / 'results.json'} and {out / 'RESULTS.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--nsl-bin", type=Path, required=True, help="path to the nsl compiler binary"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="bench output dir on REAL DISK (never /tmp — a 31G tmpfs wiped between sessions)",
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument(
        "--updates",
        type=int,
        default=30,
        help="requested optimizer updates per Pass-A run",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="leading updates discarded from the tok/s window",
    )
    ap.add_argument(
        "--arms",
        type=str,
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARMS)}",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse existing build_<id>/prog binaries (markers re-checked from saved stderr)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print every command (builds included) without executing anything",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-run watchdog seconds (kills the GROUP)",
    )
    args = ap.parse_args()

    out: Path = args.out_dir.resolve()
    nsl: Path = args.nsl_bin.resolve()
    if out == Path("/tmp") or Path("/tmp") in out.parents:
        sys.exit(
            "--out-dir is under /tmp — a 31G tmpfs wiped between sessions; "
            "evidence goes on real disk"
        )

    if args.updates <= args.warmup + 2:
        sys.exit(
            f"--updates {args.updates} <= --warmup {args.warmup} + 2 — no "
            "measurable steady-state window; every round would be vacuous"
        )

    want = [k.strip() for k in args.arms.split(",") if k.strip()]
    unknown = [k for k in want if k not in ARMS]
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}; known: {', '.join(ARMS)}")
    arms = [ARMS[k] for k in dict.fromkeys(want)]
    builds = [BUILDS[bid] for bid in dict.fromkeys(a.build_id for a in arms)]

    geom = parse_recipe(RECIPE)
    n_main = args.updates * geom.tokens_per_update + 8 * geom.seq_len
    n_prof = PROFILE_UPDATES * geom.tokens_per_update + 8 * geom.seq_len

    if args.dry_run:
        return dry_run(args, arms, builds, geom, out, nsl, n_main, n_prof)

    if not nsl.exists():
        sys.exit(f"compiler not found at {nsl}")
    out.mkdir(parents=True, exist_ok=True)
    ensure_tokens(out / "tokens_main.bin", n_main, geom)
    ensure_tokens(out / "tokens_profile.bin", n_prof, geom)

    # ── builds first (compile-only — no GPU, no guard needed yet) ─────────
    warnings: list[str] = []
    markers: dict[str, dict[str, int | None]] = {}
    binaries: dict[str, Path] = {}
    for b in builds:
        stderr_path = out / f"build_{b.id}.stderr"
        if args.skip_build:
            binary = out / f"build_{b.id}" / "prog"
            if not binary.exists():
                sys.exit(f"--skip-build, but {binary} is missing")
            # A reused binary must be the binary the report describes: the
            # saved bench.nsl must match the current recipe/--updates render
            # (the scheduler total_steps and geometry are baked into it).
            src = out / f"build_{b.id}" / "bench.nsl"
            expected = render_program(
                geom, out / f"build_{b.id}" / "tokens.bin", args.updates
            )
            if not src.exists() or src.read_text() != expected:
                sys.exit(
                    f"--skip-build, but {src} does not match the current "
                    "recipe/--updates rendering — the reused binary would "
                    "misdescribe the measurement; rebuild without --skip-build"
                )
            if stderr_path.exists():
                counts, warns = check_build_markers(b, stderr_path.read_text())
            else:
                counts = {}
                warns = [
                    (
                        f"{b.id}: --skip-build with no saved build stderr — "
                        f"marker assertions skipped"
                    )
                ]
        else:
            env_s = " ".join(f"{k}={v}" for k, v in b.env) or "(fusion switches unset)"
            print(f"[build] {b.id}: {' '.join(b.flags)}  {env_s}", flush=True)
            binary = do_build(b, out, geom, args.updates, nsl, args.timeout)
            counts, warns = check_build_markers(b, stderr_path.read_text())
        binaries[b.id] = binary
        markers[b.id] = counts
        warnings += warns
    for w in warnings:
        print(f"[warn] {w}", flush=True)

    # Pass A always starts from the full stream — a previous interrupted
    # campaign may have left the 10-update profile stream in a build dir.
    for b in builds:
        shutil.copy2(out / "tokens_main.bin", out / f"build_{b.id}" / "tokens.bin")

    # ── every GPU run from here on is behind the guard ────────────────────
    gpu_guard.acquire_or_refuse("fusion_bench")
    device = detect_device()
    if device not in PEAK_TFLOPS_BY_DEVICE:
        sys.exit(
            f"no measured peak-TFLOPS entry for {device!r} — an MFU against the "
            f"wrong roofline is worse than none. Known: {sorted(PEAK_TFLOPS_BY_DEVICE)}"
        )
    roofline = PEAK_TFLOPS_BY_DEVICE[device]["tf32"] * 1e12
    flops_per_token = SCALE.flops_per_token(geom.seq_len)

    # ── Pass A: interleaved timing rounds, events-only instrumentation ────
    results = {a.key: ArmResult(spec=a) for a in arms}
    for rnd in range(args.rounds):
        for a in arms:
            res = results[a.key]
            if res.oom or res.error:
                continue
            events_path = out / f"{a.key}_r{rnd}.events"
            events_path.unlink(missing_ok=True)
            extra = {"NSL_EVENTS": str(events_path)}
            if a.key == "A4" and rnd == 0:
                # First round only: the graph capture/taint log is the
                # follow-up PR's evidence; it also adds stderr volume, so it
                # stays out of the other rounds.
                extra["NSL_CUDA_GRAPH_LOG"] = "1"
            rc, lines = run_binary(
                binaries[a.build_id],
                run_env(extra),
                args.timeout,
                out / f"{a.key}_r{rnd}.stderr",
            )
            if rc != 0:
                stderr_text = (out / f"{a.key}_r{rnd}.stderr").read_text(
                    errors="replace"
                )
                # Keep the events tail: the LAST STEP-BOUNDARY sample, which
                # is NOT the allocated bytes at failure (the abort's own
                # NSL_MEMSTATS dump in the run stderr is) — labeled as such.
                if res.alloc_first_mid_last is None:
                    res.alloc_first_mid_last = parse_events(
                        events_path
                    ).alloc_first_mid_last
                if any(mk in stderr_text for mk in OOM_MARKERS):
                    res.oom = True
                    print(
                        f"[round {rnd}] {a.key}: OOM (a result, not a failure)",
                        flush=True,
                    )
                else:
                    res.error = stderr_text[-800:]
                    print(f"[round {rnd}] {a.key}: FAILED rc={rc}", flush=True)
                continue
            timing = parse_timing(lines, args.warmup, geom.tokens_per_update, rnd)
            sig = parse_events(events_path)
            if timing:
                timing.kernel_launch_count = sig.kernel_launch_count
                timing.gpu_launch_count = sig.gpu_launch_count
                timing.fused_ew = sig.fused_ew
                res.rounds.append(timing)
                print(f"[round {rnd}] {a.key}: {timing.tok_s:,.0f} tok/s", flush=True)
            else:
                print(
                    f"[round {rnd}] {a.key}: no timing "
                    f"(need > {args.warmup + 2} OPT_STEP arrivals)",
                    flush=True,
                )
            if res.alloc_first_mid_last is None:
                res.alloc_first_mid_last = sig.alloc_first_mid_last
            if a.build.fusion_on:
                if sig.fused_ew is None:
                    w = (
                        f"{a.key} round {rnd}: fused_ew_counters absent — the "
                        "runtime event kind may not have landed yet"
                    )
                    warnings.append(w)
                    print(f"[warn] {w}", flush=True)
                elif sig.fused_ew.get("fallbacks", 0) != 0:
                    v = (
                        f"round {rnd}: fused_ew_counters.fallbacks="
                        f"{sig.fused_ew.get('fallbacks')} != 0 — the fused fast "
                        "path is not engaging uniformly"
                    )
                    res.violations.append(v)
                    print(f"[VIOLATION] {a.key} {v}", flush=True)

    # ── Pass B: kernel-profile wall split (timings DISCARDED) ─────────────
    for a in arms:
        if not a.profiled:
            continue
        res = results[a.key]
        if res.oom or res.error:
            continue
        if a.build.graphs:
            # Structurally unreachable (A0/A1/A3 are graph-free), kept as a
            # belt: the per-launch profiler refuses to compose with replay.
            res.profile_error = "graphs build — profiler refuses; excluded"
            continue
        d = out / f"build_{a.build_id}"
        shutil.copy2(out / "tokens_profile.bin", d / "tokens.bin")
        prof = d / "kernel_profile.json"
        prof.unlink(missing_ok=True)
        rc, _ = run_binary(
            binaries[a.build_id],
            run_env(
                {"NSL_PROFILE_KERNELS": "1", "NSL_PROFILE_KERNELS_POOL": PROFILE_POOL}
            ),
            args.timeout,
            out / f"{a.key}_kernels.stderr",
        )
        parsed = parse_kernel_profile(prof)
        if isinstance(parsed, str):
            res.profile_error = f"rc={rc}; {parsed}"
            print(f"[pass B] {a.key}: REFUSED — {res.profile_error}", flush=True)
            continue
        res.host_gap_pct = parsed.host_gap_pct
        res.profile_busy_ms = parsed.busy_ms
        res.profile_span_ms = parsed.span_ms
        res.kernel_table = parsed.table
        shutil.copy2(prof, out / f"{a.key}_kernel_profile.json")
        print(
            f"[pass B] {a.key}: host_gap {parsed.host_gap_pct:.1f}% "
            f"(busy {parsed.busy_ms:.0f} ms / span {parsed.span_ms:.0f} ms over "
            f"{parsed.n_events} events; timings discarded from throughput)",
            flush=True,
        )

    # ── report ────────────────────────────────────────────────────────────
    git_rev = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,  # provenance is best-effort; an empty rev is visible
    ).stdout.strip()
    payload: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "roofline_tf32": roofline,
        "flops_per_token": flops_per_token,
        "tokens_per_update": geom.tokens_per_update,
        "geometry": dataclasses.asdict(geom),
        "nsl_bin": str(nsl),
        "nsl_stdlib_path": stdlib_path(),
        "worktree_commit": git_rev,
        "updates": args.updates,
        "warmup": args.warmup,
        "rounds_requested": args.rounds,
        "builds": {
            b.id: {
                "flags": list(b.flags),
                "env": dict(b.env),
                "markers": markers.get(b.id, {}),
                "stderr": f"build_{b.id}.stderr",
            }
            for b in builds
        },
        "warnings": warnings,
    }
    arms_out: dict[str, object] = {}
    for a in arms:
        res = results[a.key]
        best = res.best
        arms_out[a.key] = {
            "build_id": a.build_id,
            "env": {
                "build": dict(a.build.env),
                "run": {"NSL_CUDA_GRAPH_LOG": "1 (first round, r0, only)"}
                if a.key == "A4"
                else {},
            },
            "what": a.what,
            "oom": res.oom,
            "error": res.error,
            "rounds": [dataclasses.asdict(r) for r in res.rounds],
            "best_tok_s": best.tok_s if best else None,
            "mfu_pct": (100.0 * best.tok_s * flops_per_token / roofline)
            if best
            else None,
            "spread_pct": res.spread_pct,
            "launch_counts": launch_counts_of(res),
            "host_gap_pct": res.host_gap_pct,
            "profile_busy_ms": res.profile_busy_ms,
            "profile_span_ms": res.profile_span_ms,
            "profile_error": res.profile_error,
            "kernel_table": [
                {"name": name, "launches": count, "total_ms": total}
                for name, count, total in res.kernel_table[:30]
            ],
            "alloc_series_first_mid_last": res.alloc_first_mid_last,
            "violations": res.violations,
        }
    # A partial --arms rerun must not clobber prior arms'/builds' results
    # (resident_bench's rule): merge what an earlier invocation recorded,
    # letting this run's rows win.
    prior = out / "results.json"
    if prior.exists():
        try:
            prev = json.loads(prior.read_text())
        except json.JSONDecodeError:
            prev = None
        if isinstance(prev, dict):
            prev_arms = prev.get("arms")
            if isinstance(prev_arms, dict):
                arms_out = {**prev_arms, **arms_out}
            prev_builds = prev.get("builds")
            builds_out = payload["builds"]
            if isinstance(prev_builds, dict) and isinstance(builds_out, dict):
                payload["builds"] = {**prev_builds, **builds_out}
    payload["arms"] = arms_out
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    (out / "RESULTS.md").write_text(
        render_results_md(
            arms,
            results,
            builds,
            markers,
            warnings,
            device,
            flops_per_token,
            roofline,
            geom,
            args,
        )
    )

    # ── one-screen summary ────────────────────────────────────────────────
    print()
    for line in arm_table(arms, results, flops_per_token, roofline):
        print(line)
    print()
    for b in builds:
        m = markers.get(b.id, {})
        print(
            f"[markers] {b.id}: wgrad_fused={m.get('wgrad_fused')} "
            f"rmsnorm_folds={m.get('rmsnorm_folds')} ew_chains={m.get('ew_chains')} "
            f"scalar_imms={m.get('scalar_imms')} rope_folds={m.get('rope_folds')}"
        )
    all_violations = [f"{k}: {v}" for k, r in results.items() for v in r.violations]
    for v in all_violations:
        print(f"[VIOLATION] {v}")
    for w in warnings:
        print(f"[warn] {w}")
    if "A4" in results and results["A4"].rounds:
        print(f"[note] A4 graph capture/taint log: {out / 'A4_r0.stderr'}")
    # Zero timed rounds on a non-OOM arm is a vacuous measurement, not a
    # success — the exit code must say so.
    vacuous = [
        a.key
        for a in arms
        if not results[a.key].oom
        and not results[a.key].error
        and not results[a.key].rounds
    ]
    for key in vacuous:
        print(f"[VACUOUS] {key}: no timed rounds — this is not a green result")
    print(f"\nresults -> {out / 'results.json'}\nreport  -> {out / 'RESULTS.md'}")
    failed = (
        any(r.error for r in results.values()) or bool(all_violations) or bool(vacuous)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
