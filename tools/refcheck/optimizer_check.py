#!/usr/bin/env python3
"""Compare NSL's optimizer stack with PyTorch's over a few real update steps.

Two measurements, both from the same weights and batches on each side:

* **update parity** — AdamW with any of gradient accumulation, global-norm
  clipping, decoupled weight decay and a `warmup_cosine` schedule, run for
  `--steps` optimizer steps; reported as the relative error of the total
  parameter update, per parameter.
* **applied learning rate** (`--applied-lr`) — SGD's update is linear in the
  per-micro-batch gradients, and over a dozen small steps the model barely
  moves, so `init - after = sum_k c_k * g_k(init)`. A least-squares fit against
  the reference gradients recovers `c_k`, the rate NSL actually applied to
  micro-batch `k`. This is the measurement that showed the schedule running one
  step late; a behavioural "does the schedule do anything" test cannot.

    tools/refcheck/optimizer_check.py --tokens corpus.bin --nsl target/release/nsl \\
        --accum 4 --clip 1.0 --wd 0.1 --schedule -- --source-ad --checkpoint-blocks
    tools/refcheck/optimizer_check.py --tokens corpus.bin --nsl target/release/nsl \\
        --applied-lr --accum 4 -- --source-ad

bf16 activation storage (`--precision bf16`, written into the train header) is
held to PyTorch autocast: with `--autocast bf16` the twin's update is also
computed under `torch.autocast(bf16)`, and the exit status is the criterion of
design 6.2: NSL's median and worst per-tensor update error against PyTorch f32
are no larger than autocast's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ref import (
    SIZES,
    TRAIN_PROGRAM,
    Cfg,
    RefModel,
    Spread,
    autocast,
    batch_from_stream,
    bf16_criterion,
    grad_of,
    read_nslm,
    relative_errors,
    run_nsl,
    warmup_cosine,
    write_model,
)

WARMUP, TOTAL, MIN_LR = 8.0, 100.0, 1e-5
FIT_PARAMS = ["blocks[3].ffn.w_up", "blocks[7].attn.wo", "blocks[0].ffn.w_down"]


def schedule_step(a: argparse.Namespace, k: int) -> int:
    """The schedule step of the update that micro-batch ``k``'s window ends in.

    Optimizer-step units: the index of that update. Micro-batch units (and
    "legacy", a toolchain from before the key, which counted micro-batches):
    the index of the window's last micro-batch.
    """
    window = k // a.accum
    if a.schedule_unit == "optimizer_step":
        return window
    return window * a.accum + a.accum - 1


def nsl_run(a: argparse.Namespace, cfg, seq: int, optimizer: str, micro: int) -> None:
    unit = (
        f', schedule_unit="{a.schedule_unit}"'
        if a.accum > 1 and (a.schedule or a.applied_lr) and a.schedule_unit != "legacy"
        else ""
    )
    train_args = (
        (f", grad_accumulation={a.accum}" if a.accum > 1 else "")
        + unit
        + (f", grad_clip={a.clip}" if a.clip > 0 else "")
        + (', precision="bf16"' if a.precision == "bf16" else "")
    )
    schedule = f"    scheduler: warmup_cosine(warmup_steps={WARMUP:g}, total_steps={TOTAL:g}, min_lr={MIN_LR})\n"
    program = TRAIN_PROGRAM.format(
        load="",
        save_init='model_save(m, "init.nslm")\n',
        batch=1,
        seq=seq,
        rows=seq,
        vocab=cfg.vocab,
        d_model=cfg.d_model,
        train_args=train_args,
        optimizer=optimizer,
        scheduler=schedule if a.schedule or a.applied_lr else "",
    )
    run_nsl(a.nsl, a.workdir, program, a.nsl_flags)


def applied_lr(
    a: argparse.Namespace, cfg, seq: int, tokens: np.ndarray, micro: int
) -> int:
    nsl_run(a, cfg, seq, f"SGD(lr={a.lr})", micro)
    init, after = (
        read_nslm(a.workdir / "init.nslm"),
        read_nslm(a.workdir / "after.nslm"),
    )
    model = RefModel(cfg, init)
    grads = []
    for k in range(micro):
        model.zero_grad(set_to_none=True)
        model.loss(*batch_from_stream(tokens, k, 1, seq)).backward()
        named = model.named()
        grads.append(np.concatenate([grad_of(named[n]).ravel() for n in FIT_PARAMS]))
    delta = np.concatenate(
        [
            (init[n].astype(np.float64) - after[n].astype(np.float64)).ravel()
            for n in FIT_PARAMS
        ]
    )
    basis = np.stack(grads, axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, delta, rcond=None)
    residual = float(np.linalg.norm(basis @ coeffs - delta) / np.linalg.norm(delta))
    expected = [
        warmup_cosine(a.lr, schedule_step(a, k), WARMUP, TOTAL, MIN_LR)
        for k in range(micro)
    ]
    print(f"{'micro':>5s} {'applied lr':>12s} {'schedule':>12s}")
    worst = 0.0
    for k, (got, want) in enumerate(zip(coeffs * a.accum, expected)):
        print(f"{k:5d} {got:12.4e} {want:12.4e}")
        worst = max(worst, abs(got - want) / a.lr)
    ok = worst < 0.02
    print(
        f"{'ok' if ok else 'FAIL'}: largest |applied - schedule| is {worst:.1%} of the base rate (fit residual {residual:.1e})"
    )
    return 0 if ok else 1


def update_parity(
    a: argparse.Namespace, cfg, seq: int, tokens: np.ndarray, micro: int
) -> int:
    nsl_run(
        a,
        cfg,
        seq,
        f"AdamW(lr={a.lr}, weight_decay={a.wd}, beta1=0.9, beta2=0.95, eps=1e-8)",
        micro,
    )
    init, after = (
        read_nslm(a.workdir / "init.nslm"),
        read_nslm(a.workdir / "after.nslm"),
    )
    d_ref = reference_update(a, cfg, seq, tokens, micro, init, None)
    d_nsl = {
        name: after[name].astype(np.float64) - init[name].astype(np.float64)
        for name in d_ref
    }
    rel_b = relative_errors(d_nsl, d_ref)
    rels = sorted(((r, k) for k, r in rel_b.items()), reverse=True)
    for rel, name in rels[:5]:
        print(f"{name:32s} update relative error {rel:.2e}")
    b = Spread.of(rel_b)
    if a.autocast:
        d_ac = reference_update(a, cfg, seq, tokens, micro, init, a.autocast)
        sa = Spread.of(relative_errors(d_ac, d_ref))
        sc = Spread.of(relative_errors(d_nsl, d_ac))
        for label, sp in (
            ("(a) autocast vs f32", sa),
            ("(b) NSL vs f32     ", b),
            ("(c) NSL vs autocast", sc),
        ):
            print(f"{label}  median {sp.median:.2e}  worst {sp.worst:.2e}")
        ok, line = bf16_criterion([b], [sa])
        print(line)
        return 0 if ok else 1
    ok = b.worst <= a.tol
    print(
        f"{'ok' if ok else 'FAIL'}: median {b.median:.2e}, worst {b.worst:.2e} over {len(rels)} parameters (tolerance {a.tol:.0e})"
    )
    return 0 if ok else 1


def reference_update(
    a: argparse.Namespace,
    cfg: Cfg,
    seq: int,
    tokens: np.ndarray,
    micro: int,
    init: dict[str, np.ndarray],
    mode: str | None,
) -> dict[str, np.ndarray]:
    """The twin's total update per parameter (f64) from `init`, under autocast
    when `mode` says so (the optimizer, clipping and schedule stay f32)."""
    model = RefModel(cfg, init)
    model.train()
    params = list(model.params.values())
    opt = torch.optim.AdamW(
        params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.wd
    )
    for k in range(micro):
        with autocast(mode):
            loss = model.loss(*batch_from_stream(tokens, k, 1, seq)) / a.accum
        loss.backward()
        if (k + 1) % a.accum == 0:
            if a.schedule:
                for group in opt.param_groups:
                    group["lr"] = warmup_cosine(
                        a.lr, schedule_step(a, k), WARMUP, TOTAL, MIN_LR
                    )
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
    out = {
        name: p.detach().cpu().numpy().astype(np.float64)
        - init[name].astype(np.float64)
        for name, p in model.named().items()
    }
    del model, opt
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", choices=sorted(SIZES), default="50m")
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--nsl", default="nsl")
    ap.add_argument("--workdir", type=Path, default=Path("refcheck-work/optim"))
    ap.add_argument("--steps", type=int, default=3, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument(
        "--schedule-unit",
        choices=["optimizer_step", "micro_batch", "legacy"],
        default="optimizer_step",
        help="what one schedule step is under --accum; 'legacy' omits the key "
        "for a toolchain that predates it (0.10-lts), which counts micro-batches",
    )
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--schedule", action="store_true", help="add warmup_cosine(warmup 8, total 100)"
    )
    ap.add_argument(
        "--applied-lr",
        action="store_true",
        help="recover the rate applied per micro-batch",
    )
    ap.add_argument("--tol", type=float, default=3e-2)
    ap.add_argument(
        "--precision",
        choices=["f32", "bf16"],
        default="f32",
        help="the NSL train header's precision= (bf16: activation storage)",
    )
    ap.add_argument(
        "--autocast",
        choices=["bf16"],
        help="also run the twin under torch.autocast and apply the bf16 criterion",
    )
    ap.add_argument("nsl_flags", nargs="*", default=["--source-ad"])
    a = ap.parse_args()
    if a.applied_lr and (a.clip > 0 or a.wd > 0):
        raise SystemExit("--applied-lr fits a plain SGD update; drop --clip/--wd")

    cfg = write_model(a.size, a.workdir)
    seq, micro = cfg.max_seq, a.steps * a.accum
    tokens = np.fromfile(a.tokens, dtype=np.uint16)[: micro * seq]
    if len(tokens) < micro * seq:
        raise SystemExit(f"{a.tokens}: need {micro * seq} tokens, found {len(tokens)}")
    tokens.tofile(a.workdir / "tokens.bin")
    return (
        applied_lr(a, cfg, seq, tokens, micro)
        if a.applied_lr
        else update_parity(a, cfg, seq, tokens, micro)
    )


if __name__ == "__main__":
    sys.exit(main())
