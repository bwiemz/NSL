#!/usr/bin/env python3
"""P1 Muon items 5 + 7: Muon-vs-AdamW experiment runner.

Subcommands:
  tune   3 lrs per optimizer x 400 steps (equal lr-tuning budget)
  main   best-lr head-to-head x N steps (default 3000), optional 2nd seed
  sweep  item 7: ns_steps {3,4,5}, nesterov on/off, momentum sweep (400 steps)

Every arm: same --seed (identical init), same token-file prefix (identical
data order), batch 1 x seq 1024, grad_clip 1.0, dropout as-configured, val
loss on held-out NSL + Rust byte streams every 200 steps.

Results: one JSON per arm under muon_logs/<arm>/result.json + a markdown
summary printed at the end of each subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NSL = REPO / "target" / "release" / "nsl"
DATA = HERE / "muon_data"
LOGS = HERE / "muon_logs"
PROGRAM = HERE / "muon50m" / "pretrain_exp.nsl"

SEQ = 1024


def make_token_prefix(steps: int) -> Path:
    """First steps*SEQ+1 u16 tokens of the master stream (same data order)."""
    need = steps * SEQ + 1
    out = DATA / f"train_{steps}.bin"
    master = DATA / "train_tokens.bin"
    if out.exists() and out.stat().st_size == need * 2:
        return out
    data = master.read_bytes()[: need * 2]
    assert len(data) == need * 2, f"master stream too small for {steps} steps"
    out.write_bytes(data)
    return out


def run_arm(
    name: str,
    optimizer_line: str,
    steps: int,
    seed: int,
    timeout_s: int | None = None,
) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    workdir = LOGS / name
    workdir.mkdir(parents=True, exist_ok=True)
    result_path = workdir / "result.json"
    if result_path.exists():
        r = json.loads(result_path.read_text())
        if r.get("status") == "ok":
            print(f"[skip] {name}: cached ok result")
            return r

    tokens = make_token_prefix(steps)
    src = PROGRAM.read_text()
    src = src.replace("OPTIMIZER_LINE", optimizer_line)
    src = src.replace("TOKENS_PATH", tokens.as_posix())
    src = src.replace("DATA_DIR", DATA.as_posix())
    src = src.replace("FINAL_STEP", str(steps))
    (workdir / "prog.nsl").write_text(src)
    (workdir / "model.nsl").write_text((PROGRAM.parent / "model.nsl").read_text())

    env = dict(os.environ)
    env["NSL_STDLIB_PATH"] = str(REPO / "stdlib")

    if timeout_s is None:
        timeout_s = int(steps * 1.5) + 600

    t0 = time.time()
    proc = subprocess.Popen(
        [str(NSL), "run", "--source-ad", "--deterministic", "--seed", str(seed),
         str(workdir / "prog.nsl")],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out_lines: list[tuple[float, str]] = []
    err_lines: list[tuple[float, str]] = []

    def drain(pipe, sink):
        for line in pipe:
            sink.append((time.time(), line.rstrip("\n")))

    to = threading.Thread(target=drain, args=(proc.stdout, out_lines))
    te = threading.Thread(target=drain, args=(proc.stderr, err_lines))
    to.start()
    te.start()
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = -9
    to.join()
    te.join()
    wall = time.time() - t0

    (workdir / "stdout.log").write_text(
        "\n".join(f"{t - t0:9.2f} {l}" for t, l in out_lines)
    )
    (workdir / "stderr.log").write_text(
        "\n".join(f"{t - t0:9.2f} {l}" for t, l in err_lines)
    )

    # Parse: numeric lines come in (step, loss) pairs; VAL_NSL / VAL_RS
    # sentinel lines are each followed by one loss line.
    def num(s: str) -> float | None:
        s = s.strip()
        if s.startswith("tensor([") and s.endswith("])"):
            s = s[len("tensor(["):-2]
        try:
            return float(s)
        except ValueError:
            return None

    train_curve: list[tuple[int, float, float]] = []  # step, loss, t_rel
    val_nsl: list[tuple[int, float]] = []
    val_rs: list[tuple[int, float]] = []
    valbig: dict[str, float] = {}
    stream = [(t, l) for t, l in out_lines]
    i = 0
    in_stream = False
    last_step = None
    t_begin = None
    t_last = None
    while i < len(stream):
        t, line = stream[i]
        ls = line.strip()
        if ls == "LOSS_STREAM_BEGIN":
            in_stream = True
            t_begin = t
        elif ls == "LOSS_STREAM_END":
            break
        elif in_stream:
            if ls == "VAL_NSL" and i + 1 < len(stream):
                v = num(stream[i + 1][1])
                if v is not None and last_step is not None:
                    val_nsl.append((last_step, v))
                i += 1
            elif ls == "VAL_RS" and i + 1 < len(stream):
                v = num(stream[i + 1][1])
                if v is not None and last_step is not None:
                    val_rs.append((last_step, v))
                i += 1
            elif ls == "VALBIG_NSL" and i + 1 < len(stream):
                v = num(stream[i + 1][1])
                if v is not None:
                    valbig["nsl"] = v
                i += 1
            elif ls == "VALBIG_RS" and i + 1 < len(stream):
                v = num(stream[i + 1][1])
                if v is not None:
                    valbig["rs"] = v
                i += 1
            else:
                v = num(ls)
                if v is not None:
                    if last_step is None or (v == int(v) and 0 <= v <= steps + 1
                                             and (not train_curve
                                                  or v == train_curve[-1][0] + 1
                                                  or v == 1)):
                        # step line (integers in sequence)
                        nxt = num(stream[i + 1][1]) if i + 1 < len(stream) else None
                        if nxt is not None:
                            train_curve.append((int(v), nxt, t - (t_begin or t)))
                            last_step = int(v)
                            t_last = stream[i + 1][0]
                            i += 1
        i += 1

    status = "ok" if rc == 0 and train_curve else f"FAIL(rc={rc})"
    stream_wall = (t_last - t_begin) if (t_begin and t_last) else wall
    result = {
        "name": name,
        "status": status,
        "optimizer": optimizer_line,
        "steps_requested": steps,
        "steps_done": len(train_curve),
        "seed": seed,
        "wall_s": round(wall, 1),
        "stream_wall_s": round(stream_wall, 1),
        "tok_per_s": round(len(train_curve) * SEQ / stream_wall, 1)
        if stream_wall > 0 else 0.0,
        "first_loss": train_curve[0][1] if train_curve else None,
        "last_loss": train_curve[-1][1] if train_curve else None,
        "train_curve_every25": [c for c in train_curve if c[0] % 25 == 0],
        "val_nsl": val_nsl,
        "val_rs": val_rs,
        "valbig": valbig,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(
        f"[{status}] {name}: {len(train_curve)} steps, "
        f"last={result['last_loss']}, val_nsl={val_nsl[-1] if val_nsl else None}, "
        f"wall={wall:.0f}s"
    )
    return result


def adamw_line(lr: float) -> str:
    return (
        f"AdamW(lr={lr}, weight_decay=0.1, beta1=0.9, beta2=0.95)"
    )


def muon_line(
    lr: float,
    adamw_lr: float = 0.0003,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
) -> str:
    nv = "true" if nesterov else "false"
    return (
        f"Muon(lr={lr}, adamw_lr={adamw_lr}, momentum={momentum}, "
        f"nesterov={nv}, ns_steps={ns_steps}, weight_decay=0.1, "
        f"beta1=0.9, beta2=0.95)"
    )


def cmd_tune(args) -> None:
    rows = []
    for lr in [0.0001, 0.0003, 0.0006]:
        rows.append(run_arm(f"tune{args.steps}_adamw_{lr}", adamw_line(lr), args.steps, args.seed))
    for lr in [0.01, 0.02, 0.04]:
        rows.append(run_arm(f"tune{args.steps}_muon_{lr}", muon_line(lr), args.steps, args.seed))
    print("\n| arm | last train | val_nsl | val_rs | tok/s |")
    print("|---|---|---|---|---|")
    for r in rows:
        vn = r["val_nsl"][-1][1] if r["val_nsl"] else float("nan")
        vr = r["val_rs"][-1][1] if r["val_rs"] else float("nan")
        print(f"| {r['name']} | {r['last_loss']:.4f} | {vn:.4f} | {vr:.4f} "
              f"| {r['tok_per_s']} |")


def cmd_main(args) -> None:
    rows = []
    for seed in args.seeds:
        rows.append(run_arm(
            f"main{args.steps}_adamw{args.adamw_lr}_s{seed}", adamw_line(args.adamw_lr), args.steps, seed))
        mtag = f"main{args.steps}_muon{args.muon_lr}"
        if args.muon_momentum != 0.95:
            mtag += f"_mom{args.muon_momentum}"
        rows.append(run_arm(
            f"{mtag}_s{seed}",
            muon_line(args.muon_lr, momentum=args.muon_momentum), args.steps, seed))
    print("\n| arm | last train | tiny val_nsl | tiny val_rs | BIG val_nsl | BIG val_rs | wall s | tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        vn = r["val_nsl"][-1][1] if r["val_nsl"] else float("nan")
        vr = r["val_rs"][-1][1] if r["val_rs"] else float("nan")
        vb = r.get("valbig", {})
        print(f"| {r['name']} | {r['last_loss']:.4f} | {vn:.4f} | {vr:.4f} "
              f"| {vb.get('nsl', float('nan')):.4f} | {vb.get('rs', float('nan')):.4f} "
              f"| {r['stream_wall_s']} | {r['tok_per_s']} |")


def run_arm_500m(name: str, optimizer_line: str, opt_steps: int, seed: int) -> dict:
    """500M confirmation arm: batch 2 x seq 512 x accum 8, CCR + offload.
    `opt_steps` counts OPTIMIZER steps; on_step fires per MICRO-batch."""
    global PROGRAM
    micro = opt_steps * 8
    tokens_needed_steps = micro  # DataLoader consumes 2*512 tokens per micro
    need = micro * 2 * 512 + 1
    out = DATA / f"train500m_{opt_steps}.bin"
    master = DATA / "train_tokens.bin"
    if not (out.exists() and out.stat().st_size == need * 2):
        data = master.read_bytes()[: need * 2]
        assert len(data) == need * 2, "master stream too small for 500m arm"
        out.write_bytes(data)

    LOGS.mkdir(parents=True, exist_ok=True)
    workdir = LOGS / name
    workdir.mkdir(parents=True, exist_ok=True)
    result_path = workdir / "result.json"
    if result_path.exists():
        r = json.loads(result_path.read_text())
        if r.get("status") == "ok":
            print(f"[skip] {name}: cached ok result")
            return r

    prog500 = HERE / "muon500m" / "pretrain_exp.nsl"
    src = prog500.read_text()
    src = src.replace("OPTIMIZER_LINE", optimizer_line)
    src = src.replace("TOKENS_PATH", out.as_posix())
    src = src.replace("DATA_DIR", DATA.as_posix())
    (workdir / "prog.nsl").write_text(src)
    (workdir / "model.nsl").write_text((HERE / "muon500m" / "model.nsl").read_text())

    env = dict(os.environ)
    env["NSL_STDLIB_PATH"] = str(REPO / "stdlib")
    t0 = time.time()
    proc = subprocess.Popen(
        [str(NSL), "run", "--source-ad", "--deterministic", "--seed", str(seed),
         "--checkpoint-blocks", "--optim-state-offload",
         str(workdir / "prog.nsl")],
        cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    out_lines: list[tuple[float, str]] = []
    err_lines: list[tuple[float, str]] = []

    def drain(pipe, sink):
        for line in pipe:
            sink.append((time.time(), line.rstrip("\n")))

    to = threading.Thread(target=drain, args=(proc.stdout, out_lines))
    te = threading.Thread(target=drain, args=(proc.stderr, err_lines))
    to.start()
    te.start()
    timeout_s = int(micro * 8.0) + 1200
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = -9
    to.join()
    te.join()
    wall = time.time() - t0
    (workdir / "stdout.log").write_text(
        "\n".join(f"{t - t0:9.2f} {l}" for t, l in out_lines))
    (workdir / "stderr.log").write_text(
        "\n".join(f"{t - t0:9.2f} {l}" for t, l in err_lines))

    def num(s: str) -> float | None:
        s = s.strip()
        if s.startswith("tensor([") and s.endswith("])"):
            s = s[len("tensor(["):-2]
        try:
            return float(s)
        except ValueError:
            return None

    curve: list[tuple[int, float]] = []
    val_nsl: list[tuple[int, float]] = []
    val_rs: list[tuple[int, float]] = []
    last_step = None
    i = 0
    in_stream = False
    while i < len(out_lines):
        _, line = out_lines[i]
        ls = line.strip()
        if ls == "LOSS_STREAM_BEGIN":
            in_stream = True
        elif ls == "LOSS_STREAM_END":
            break
        elif in_stream:
            if ls == "VAL_NSL" and i + 1 < len(out_lines):
                v = num(out_lines[i + 1][1])
                if v is not None and last_step is not None:
                    val_nsl.append((last_step, v))
                i += 1
            elif ls == "VAL_RS" and i + 1 < len(out_lines):
                v = num(out_lines[i + 1][1])
                if v is not None and last_step is not None:
                    val_rs.append((last_step, v))
                i += 1
            else:
                v = num(ls)
                if v is not None and v == int(v) and 0 <= v <= micro + 1:
                    nxt = num(out_lines[i + 1][1]) if i + 1 < len(out_lines) else None
                    if nxt is not None:
                        curve.append((int(v), nxt))
                        last_step = int(v)
                        i += 1
        i += 1

    status = "ok" if rc == 0 and curve else f"FAIL(rc={rc})"
    result = {
        "name": name, "status": status, "optimizer": optimizer_line,
        "opt_steps": opt_steps, "micro_batches_done": len(curve),
        "seed": seed, "wall_s": round(wall, 1),
        "first_loss": curve[0][1] if curve else None,
        "last_loss": curve[-1][1] if curve else None,
        "train_curve_every50": [c for c in curve if c[0] % 50 == 0],
        "val_nsl": val_nsl, "val_rs": val_rs,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[{status}] {name}: {len(curve)} micros, last={result['last_loss']}, "
          f"val_nsl={val_nsl[-1] if val_nsl else None}, wall={wall:.0f}s")
    return result


def cmd_confirm500m(args) -> None:
    rows = [
        run_arm_500m(f"c500m{args.steps}_adamw_s{args.seed}",
                     adamw_line(args.adamw_lr), args.steps, args.seed),
        run_arm_500m(f"c500m{args.steps}_muon_s{args.seed}",
                     muon_line(args.muon_lr), args.steps, args.seed),
    ]
    print("\n| arm | last train | final val_nsl | final val_rs | wall s |")
    print("|---|---|---|---|---|")
    for r in rows:
        vn = r["val_nsl"][-1][1] if r["val_nsl"] else float("nan")
        vr = r["val_rs"][-1][1] if r["val_rs"] else float("nan")
        print(f"| {r['name']} | {r['last_loss']:.4f} | {vn:.4f} | {vr:.4f} "
              f"| {r['wall_s']} |")


def cmd_sweep(args) -> None:
    lr = args.muon_lr
    rows = []
    for ns in [3, 4, 5]:
        rows.append(run_arm(
            f"sweep{args.steps}_ns{ns}", muon_line(lr, ns_steps=ns), args.steps, args.seed))
    rows.append(run_arm(
        f"sweep{args.steps}_nonesterov", muon_line(lr, nesterov=False), args.steps, args.seed))
    for mom in [0.8, 0.9, 0.99]:
        rows.append(run_arm(
            f"sweep{args.steps}_mom{mom}", muon_line(lr, momentum=mom), args.steps, args.seed))
    print("\n| arm | last train | val_nsl | val_rs | tok/s |")
    print("|---|---|---|---|---|")
    for r in rows:
        vn = r["val_nsl"][-1][1] if r["val_nsl"] else float("nan")
        vr = r["val_rs"][-1][1] if r["val_rs"] else float("nan")
        print(f"| {r['name']} | {r['last_loss']:.4f} | {vn:.4f} | {vr:.4f} "
              f"| {r['tok_per_s']} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tune")
    t.add_argument("--steps", type=int, default=400)
    t.add_argument("--seed", type=int, default=1)
    t.set_defaults(fn=cmd_tune)

    m = sub.add_parser("main")
    m.add_argument("--steps", type=int, default=3000)
    m.add_argument("--seeds", type=int, nargs="+", default=[1])
    m.add_argument("--adamw-lr", type=float, default=0.0003)
    m.add_argument("--muon-lr", type=float, default=0.02)
    m.add_argument("--muon-momentum", type=float, default=0.95)
    m.set_defaults(fn=cmd_main)

    s = sub.add_parser("sweep")
    s.add_argument("--steps", type=int, default=400)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--muon-lr", type=float, default=0.02)
    s.set_defaults(fn=cmd_sweep)

    c = sub.add_parser("confirm500m")
    c.add_argument("--steps", type=int, default=150,
                   help="OPTIMIZER steps (x8 micro-batches)")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--adamw-lr", type=float, default=0.0003)
    c.add_argument("--muon-lr", type=float, default=0.02)
    c.set_defaults(fn=cmd_confirm500m)

    args = ap.parse_args()
    assert NSL.exists(), f"release binary missing: {NSL}"
    assert (DATA / "train_tokens.bin").exists(), "run muon_data.py first"
    args.fn(args)


if __name__ == "__main__":
    main()
