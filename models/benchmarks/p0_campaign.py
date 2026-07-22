"""P0 measurement campaigns (post-#403 certification + post-#405 pipeline bench).

Subcommands:
  certify  — item 2: 50M multi-seed + wd=0 integrity + reference-path arms,
             500M multi-seed, 1B streaming-stack run. Loss every step;
             grad/param-norm curves from --monitor --health-interval JSON;
             gradient coverage from --grad-integrity.
  bench    — item 3: 1B pipeline arms A..G (cumulative flag stack) at the
             demo config (batch=2, seq=512, accum=8, 4 windows), recording
             tokens/s, phase times, GPU util, PCIe throughput, allocator +
             nvidia-smi peaks, host RSS, compile time.

Every run's raw stdout/stderr is preserved under models/benchmarks/p0_logs/.
Summaries are printed as markdown tables for the results docs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NSL = REPO / "target" / "release" / "nsl"
LOGS = REPO / "models" / "benchmarks" / "p0_logs"
TOKENS_DIR = LOGS / "tokens"


@dataclass
class GpuSample:
    t: float
    mem_mib: float
    util_pct: float


@dataclass
class PcieSample:
    t: float
    rx_mb_s: float
    tx_mb_s: float


class GpuSampler:
    """Background nvidia-smi pollers: memory/util (query-gpu) + PCIe (dmon)."""

    def __init__(self) -> None:
        self.samples: list[GpuSample] = []
        self.pcie: list[PcieSample] = []
        self._procs: list[subprocess.Popen[str]] = []
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        q = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._procs.append(q)

        def read_q() -> None:
            assert q.stdout is not None
            for line in q.stdout:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    try:
                        self.samples.append(
                            GpuSample(time.time(), float(parts[0]), float(parts[1]))
                        )
                    except ValueError:
                        pass

        d = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "t"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._procs.append(d)

        def read_d() -> None:
            assert d.stdout is not None
            for line in d.stdout:
                if line.lstrip().startswith("#"):
                    continue
                cols = line.split()
                # dmon -s t: gpu rxpci txpci (MB/s)
                if len(cols) >= 3:
                    try:
                        self.pcie.append(
                            PcieSample(time.time(), float(cols[1]), float(cols[2]))
                        )
                    except ValueError:
                        pass

        for target in (read_q, read_d):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        for p in self._procs:
            p.terminate()
        for th in self._threads:
            th.join(timeout=2.0)


@dataclass
class RunResult:
    name: str
    ok: bool
    wall_s: float
    compile_s: float | None
    init_s: float | None = None
    stream_wall_s: float | None = None
    losses: list[tuple[int, float]] = field(default_factory=list)
    step_walls: list[float] = field(default_factory=list)
    fwd_s: float = 0.0
    bwd_s: float = 0.0
    opt_s: float = 0.0
    peak_alloc_bytes: int | None = None
    ws_counters: dict[str, int] = field(default_factory=dict)
    smi_peak_mib: float | None = None
    util_mean_pct: float | None = None
    pcie_rx_peak: float | None = None
    pcie_tx_peak: float | None = None
    vm_hwm_kb: int | None = None
    health: list[dict[str, object]] = field(default_factory=list)
    grad_integrity: str | None = None


LOSS_RE = re.compile(r"^-?\d+(\.\d+)?(e-?\d+)?$")
PHASE_RE = re.compile(r"\[phase\] (?:fwd=(\S+) bwd=(\S+)|opt=(\S+))")
WS_RE = re.compile(r"\[weight-stream\] uploads: (\d+) evicts: (\d+) writeback: (\d+) registered: (\d+) ptr_moves: (\d+) pack_uploads: (\d+) pack_evicts: (\d+) prefetches: (\d+) async_wb: (\d+)")


def run_arm(
    name: str,
    program: Path,
    tokens_path: Path,
    flags: list[str],
    env: dict[str, str],
    rewrites: list[tuple[str, str]] | None = None,
    timeout_s: int = 7200,
) -> RunResult:
    """Run one arm: rewrite markers, spawn nsl, timestamp output, sample GPU."""
    LOGS.mkdir(parents=True, exist_ok=True)
    src = program.read_text()
    src = src.replace("CERT_TOKENS_PATH", tokens_path.as_posix())
    for frm, to in rewrites or []:
        if frm not in src:
            raise SystemExit(f"{name}: rewrite marker '{frm}' missing in {program}")
        src = src.replace(frm, to)
    workdir = LOGS / name
    workdir.mkdir(parents=True, exist_ok=True)
    prog = workdir / "prog.nsl"
    prog.write_text(src)
    # `from model import ...` resolves relative to the program file — copy
    # the model-directory companions next to the rewritten program.
    for companion in ("model.nsl", "config.nsl"):
        cpath = program.parent / companion
        if cpath.exists():
            (workdir / companion).write_text(cpath.read_text())

    import os

    full_env = dict(os.environ)
    full_env.update(env)
    full_env["NSL_STDLIB_PATH"] = str(REPO / "stdlib")

    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    proc = subprocess.Popen(
        [str(NSL), "run", *flags, str(prog)],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=full_env,
    )

    out_lines: list[tuple[float, str]] = []
    err_lines: list[tuple[float, str]] = []

    def drain(pipe: object, sink: list[tuple[float, str]]) -> None:
        for line in pipe:  # type: ignore[attr-defined]
            sink.append((time.time(), line.rstrip("\n")))

    to_ = threading.Thread(target=drain, args=(proc.stdout, out_lines), daemon=True)
    te_ = threading.Thread(target=drain, args=(proc.stderr, err_lines), daemon=True)
    to_.start()
    te_.start()

    vm_hwm: int | None = None
    health_file = workdir / "prog.nsl.nsl-health.json"
    health_seen: dict[int, dict[str, object]] = {}
    try:
        while proc.poll() is None:
            if time.time() - t0 > timeout_s:
                proc.kill()
                break
            try:
                status = Path(f"/proc/{proc.pid}/status").read_text()
                m = re.search(r"VmHWM:\s+(\d+) kB", status)
                if m:
                    vm_hwm = int(m.group(1))
            except OSError:
                pass
            # The health monitor OVERWRITES its snapshot file each flush —
            # poll and keep every distinct step's snapshot for the curves.
            if health_file.exists():
                try:
                    snap = json.loads(health_file.read_text())
                    step = int(snap.get("step", -1))
                    if step >= 0:
                        health_seen[step] = snap
                except (json.JSONDecodeError, OSError, ValueError):
                    pass
            time.sleep(1.0)
    finally:
        to_.join(timeout=5)
        te_.join(timeout=5)
        sampler.stop()
    wall = time.time() - t0
    ok = proc.returncode == 0

    (workdir / "stdout.log").write_text(
        "\n".join(f"{t - t0:10.3f} {s}" for t, s in out_lines)
    )
    (workdir / "stderr.log").write_text(
        "\n".join(f"{t - t0:10.3f} {s}" for t, s in err_lines)
    )

    res = RunResult(name=name, ok=ok, wall_s=wall, compile_s=None, vm_hwm_kb=vm_hwm)

    # Compile time: the deterministic banner fires at compiled-program
    # start, so banner-t0 = compile+link. CERT_MODEL_READY - banner = model
    # init (randn on CPU + HtoD + streaming registration).
    prog_start: float | None = None
    for t, s in err_lines:
        if "[nsl] deterministic" in s or "deterministic RNG seed" in s:
            prog_start = t
            break
    if prog_start is None and out_lines:
        prog_start = out_lines[0][0]
    if prog_start is not None:
        res.compile_s = prog_start - t0
    for t, s in out_lines:
        if s.strip() == "CERT_MODEL_READY" and prog_start is not None:
            res.init_s = t - prog_start
            break
    # Loss-stream wall (steady training window, excludes compile/init/
    # teardown) for the throughput headline.
    t_begin = t_end = None
    for t, s in out_lines:
        if s.strip() == "LOSS_STREAM_BEGIN":
            t_begin = t
        elif s.strip() == "LOSS_STREAM_END":
            t_end = t
    if t_begin is not None and t_end is not None:
        res.stream_wall_s = t_end - t_begin

    # Losses + step walls: on_step prints step then loss on separate lines.
    in_stream = False
    pending_step: int | None = None
    last_step_t: float | None = None
    for t, s in out_lines:
        st = s.strip()
        if st == "LOSS_STREAM_BEGIN":
            in_stream = True
            last_step_t = t
            continue
        if st == "LOSS_STREAM_END":
            in_stream = False
            continue
        if not in_stream:
            continue
        inner = st
        if inner.startswith("tensor([") and inner.endswith("])"):
            inner = inner[len("tensor([") : -2]
        if pending_step is None:
            if LOSS_RE.match(inner) and "." not in inner and "e" not in inner:
                pending_step = int(inner)
        elif LOSS_RE.match(inner):
            res.losses.append((pending_step, float(inner)))
            if last_step_t is not None:
                res.step_walls.append(t - last_step_t)
            last_step_t = t
            pending_step = None
    # Phase lines.
    for _, s in out_lines:
        m = PHASE_RE.search(s)
        if m:
            if m.group(1) is not None:
                res.fwd_s += float(m.group(1))
                res.bwd_s += float(m.group(2))
            else:
                res.opt_s += float(m.group(3))
    # Peak allocator bytes from the program's epilogue print.
    for i, (_, s) in enumerate(out_lines):
        if s.strip() == "PEAK_TOTAL_BYTES" and i + 1 < len(out_lines):
            try:
                res.peak_alloc_bytes = int(float(out_lines[i + 1][1].strip()))
            except ValueError:
                pass
    # Weight-stream counters (stderr atexit line).
    for _, s in err_lines:
        m = WS_RE.search(s)
        if m:
            keys = [
                "uploads", "evicts", "writeback", "registered", "ptr_moves",
                "pack_uploads", "pack_evicts", "prefetches", "async_wb",
            ]
            res.ws_counters = {k: int(v) for k, v in zip(keys, m.groups())}
    # Health snapshots: polled from the per-flush overwritten file, plus a
    # final read after exit.
    if health_file.exists():
        try:
            snap = json.loads(health_file.read_text())
            step = int(snap.get("step", -1))
            if step >= 0:
                health_seen[step] = snap
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    res.health = [health_seen[k] for k in sorted(health_seen)]
    # Grad-integrity report: a "[grad-integrity]" marker followed by
    # checks=/expected_params=/gradient_params=/finite=/nonzero=/missing=
    # lines — join the block.
    for i, (_, s) in enumerate(err_lines):
        if s.strip() == "[grad-integrity]":
            block = [
                err_lines[j][1].strip()
                for j in range(i + 1, min(i + 7, len(err_lines)))
            ]
            res.grad_integrity = " ".join(block)
            break
    # GPU sampler stats.
    if sampler.samples:
        res.smi_peak_mib = max(x.mem_mib for x in sampler.samples)
        res.util_mean_pct = sum(x.util_pct for x in sampler.samples) / len(
            sampler.samples
        )
    if sampler.pcie:
        res.pcie_rx_peak = max(x.rx_mb_s for x in sampler.pcie)
        res.pcie_tx_peak = max(x.tx_mb_s for x in sampler.pcie)
    (workdir / "result.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in res.__dict__.items()
                if k not in ("losses", "step_walls", "health")
            }
            | {
                "n_losses": len(res.losses),
                "first_loss": res.losses[0][1] if res.losses else None,
                "last_loss": res.losses[-1][1] if res.losses else None,
                "losses": res.losses,
                "step_walls": res.step_walls,
                "health": res.health,
            },
            indent=1,
        )
    )
    return res


def tokens_per_step_for(batch: int, seq: int, accum: int) -> int:
    return batch * seq * accum


def summarize(res: RunResult, tokens_per_loss_line: int) -> str:
    """One markdown row. `tokens_per_loss_line` = batch*seq (on_step fires
    per MICRO-batch; under CSLA the window backward lands on accumulation-
    boundary micro-batches, so per-line walls are bimodal). The tok/s
    headline is trained-tokens / loss-stream wall — cadence-independent.
    """
    first = res.losses[0][1] if res.losses else float("nan")
    last = res.losses[-1][1] if res.losses else float("nan")
    total_tokens = len(res.losses) * tokens_per_loss_line
    tps = (
        total_tokens / res.stream_wall_s
        if res.stream_wall_s and res.stream_wall_s > 0
        else float("nan")
    )
    ws = res.ws_counters
    return (
        f"| {res.name} | {'ok' if res.ok else 'FAIL'} | {len(res.losses)} "
        f"| {first:.4f} | {last:.4f} | {res.stream_wall_s or 0:.0f} | {tps:.0f} "
        f"| {res.fwd_s:.1f}/{res.bwd_s:.1f}/{res.opt_s:.1f} "
        f"| {(res.peak_alloc_bytes or 0) / 1e9:.2f} "
        f"| {(res.smi_peak_mib or 0) / 1024:.2f} "
        f"| {res.util_mean_pct or 0:.0f} "
        f"| {res.pcie_rx_peak or 0:.0f}/{res.pcie_tx_peak or 0:.0f} "
        f"| {(res.vm_hwm_kb or 0) / 1e6:.1f} "
        f"| {res.compile_s or 0:.1f}+{res.init_s or 0:.0f} "
        f"| {ws.get('uploads', 0)}/{ws.get('evicts', 0)}"
        f"/{ws.get('pack_uploads', 0)}/{ws.get('pack_evicts', 0)}"
        f"/{ws.get('prefetches', 0)}/{ws.get('async_wb', 0)} |"
    )


HEADER = (
    "| arm | status | micro-batches | first loss | last loss | train wall s "
    "| tok/s | fwd/bwd/opt s | alloc peak GB | smi peak GB | util % "
    "| PCIe rx/tx MB/s | host HWM GB | compile+init s "
    "| up/ev/pup/pev/pf/awb |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def cmd_bench(steps: int) -> None:
    tokens_n = steps * tokens_per_step_for(2, 512, 8)
    tok = TOKENS_DIR / "bench_1b.bin"
    gen_tokens(tok, tokens_n)
    base = [
        "--source-ad", "--deterministic", "--checkpoint-blocks",
        "--layerwise-accum", "--optim-state-offload", "--weight-stream",
    ]
    arms: list[tuple[str, list[str]]] = [
        ("A_baseline_stream", base),
        ("B_arena", [*base, "--stream-arena"]),
        ("C_prefetch", [*base, "--stream-arena", "--stream-prefetch"]),
        (
            "D_async_wb",
            [*base, "--stream-arena", "--stream-prefetch", "--stream-async-writeback"],
        ),
        (
            "E_ckpt_stride_auto",
            [
                *base, "--stream-arena", "--stream-prefetch",
                "--stream-async-writeback", "--checkpoint-stride", "auto",
            ],
        ),
        (
            # NOTE: `auto` falls back to stride 1 at 1B (symbolic shapes —
            # nothing to price), and explicit stride >= 2 REFUSES under
            # --layerwise-accum (SDPA side-bands cannot cross replay
            # ranges), so the stride lever is measured as documented-inert
            # on this stack; E == D functionally.
            "F_fused_rmsnorm_all",
            [
                *base, "--stream-arena", "--stream-prefetch",
                "--stream-async-writeback", "--fuse-rmsnorm-backward",
            ],
        ),
    ]
    env = {"NSL_WS_COUNTER": "1", "NSL_PHASE_TIMING": "1"}
    rows = []
    for name, flags in arms:
        print(f"=== {name}: {' '.join(flags)}")
        res = run_arm(
            name,
            REPO / "models" / "coder1b" / "pretrain_cert.nsl",
            tok,
            flags,
            env,
        )
        row = summarize(res, 2 * 512)
        print(row)
        rows.append(row)
    print("\n## Pipeline bench (1B, demo config, RTX 5070 Ti)\n")
    print(HEADER)
    print("\n".join(rows))


def gen_tokens(path: Path, n: int) -> None:
    if path.exists() and path.stat().st_size == n * 2:
        return
    subprocess.run(
        [
            "python3",
            str(REPO / "models" / "benchmarks" / "gen_tokens.py"),
            str(path),
            "--tokens",
            str(n),
        ],
        check=True,
    )


def cmd_certify(scale: str, steps: int, seeds: list[int]) -> None:
    if scale == "50m":
        program = REPO / "models" / "coder50m" / "pretrain_cert.nsl"
        batch, seq, accum = 1, 1024, 1
        flags = ["--source-ad", "--deterministic"]
        tok = TOKENS_DIR / f"cert_50m_{steps}.bin"
    elif scale == "500m":
        program = REPO / "models" / "coder500m" / "pretrain_cert.nsl"
        batch, seq, accum = 2, 512, 8
        flags = [
            "--source-ad", "--deterministic", "--checkpoint-blocks",
            "--optim-state-offload",
        ]
        tok = TOKENS_DIR / f"cert_500m_{steps}.bin"
    elif scale == "1b":
        program = REPO / "models" / "coder1b" / "pretrain_cert.nsl"
        batch, seq, accum = 2, 512, 8
        flags = [
            "--source-ad", "--deterministic", "--checkpoint-blocks",
            "--layerwise-accum", "--optim-state-offload", "--weight-stream",
            "--stream-arena", "--stream-prefetch", "--stream-async-writeback",
        ]
        tok = TOKENS_DIR / f"cert_1b_{steps}.bin"
    else:
        raise SystemExit(f"unknown scale {scale}")
    gen_tokens(tok, steps * tokens_per_step_for(batch, seq, accum))

    # Timeout sized to measured per-micro-batch walls (50M ~1.2s at accum=1,
    # 500M ~6.2s, 1B ~10.4s at accum=8) with headroom.
    per_micro_s = {"50m": 1.2, "500m": 6.2, "1b": 10.4}[scale]
    timeout_s = int(steps * accum * per_micro_s * 1.6) + 900
    env = {"NSL_WS_COUNTER": "1"}
    # 500M/1B run FASE-Deferred (accum=8): per-param grad norms are skipped
    # by design there (see monitor_fase_gate.rs) — add --grad-integrity so
    # gradient coverage is still certified at those scales.
    monitor = ["--monitor", "--health-interval", "25"]
    if scale in ("500m", "1b"):
        monitor = [*monitor, "--grad-integrity"]
    rows = []
    for seed in seeds:
        name = f"cert_{scale}_s{seed}"
        print(f"=== {name}")
        res = run_arm(
            name, program, tok, [*flags, "--seed", str(seed), *monitor], env,
            timeout_s=timeout_s,
        )
        rows.append(summarize(res, batch * seq))
        print(rows[-1])
    if scale == "50m":
        # wd=0 integrity arm: full gradient coverage under --grad-integrity.
        name = "cert_50m_wd0_integrity"
        print(f"=== {name}")
        res = run_arm(
            name,
            program,
            tok,
            [*flags, "--seed", "1", "--grad-integrity", *monitor],
            env,
            rewrites=[("weight_decay=0.1", "weight_decay=0.0")],
        )
        rows.append(summarize(res, batch * seq))
        print(rows[-1])
        if res.grad_integrity:
            print(f"grad-integrity: {res.grad_integrity}")
        # Reference path at reduced scale: --training-reference (fusion/FBIP/
        # fused-step all off) vs the standard arm, same seed + data.
        name = "cert_50m_reference"
        print(f"=== {name}")
        res = run_arm(
            name,
            program,
            tok,
            [*flags, "--seed", "1", "--training-reference"],
            env,
        )
        rows.append(summarize(res, batch * seq))
        print(rows[-1])
    print(f"\n## Certification {scale}\n")
    print(HEADER)
    print("\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--steps", type=int, default=4)
    c = sub.add_parser("certify")
    c.add_argument("scale", choices=["50m", "500m", "1b"])
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    if args.cmd == "bench":
        cmd_bench(args.steps)
    else:
        cmd_certify(args.scale, args.steps, args.seeds)


if __name__ == "__main__":
    main()
