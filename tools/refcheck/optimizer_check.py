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
    RefModel,
    batch_from_stream,
    grad_of,
    read_nslm,
    run_nsl,
    warmup_cosine,
    write_model,
)

WARMUP, TOTAL, MIN_LR = 8.0, 100.0, 1e-5
FIT_PARAMS = ["blocks[3].ffn.w_up", "blocks[7].attn.wo", "blocks[0].ffn.w_down"]


def nsl_run(a: argparse.Namespace, cfg, seq: int, optimizer: str, micro: int) -> None:
    train_args = (f", grad_accumulation={a.accum}" if a.accum > 1 else "") + (
        f", grad_clip={a.clip}" if a.clip > 0 else ""
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
    # The update at the end of an accumulation window uses the rate of its last micro-batch.
    expected = [
        warmup_cosine(
            a.lr, (k // a.accum) * a.accum + a.accum - 1, WARMUP, TOTAL, MIN_LR
        )
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
    model = RefModel(cfg, init)
    model.train()
    params = list(model.params.values())
    opt = torch.optim.AdamW(
        params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.wd
    )
    for k in range(micro):
        (model.loss(*batch_from_stream(tokens, k, 1, seq)) / a.accum).backward()
        if (k + 1) % a.accum == 0:
            if a.schedule:
                for group in opt.param_groups:
                    group["lr"] = warmup_cosine(a.lr, k, WARMUP, TOTAL, MIN_LR)
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
    rels = []
    for name, p in model.named().items():
        d_ref = p.detach().cpu().numpy().astype(np.float64) - init[name].astype(
            np.float64
        )
        d_nsl = after[name].astype(np.float64) - init[name].astype(np.float64)
        rels.append(
            (
                float(np.linalg.norm(d_ref - d_nsl) / (np.linalg.norm(d_ref) + 1e-300)),
                name,
            )
        )
    rels.sort(reverse=True)
    for rel, name in rels[:5]:
        print(f"{name:32s} update relative error {rel:.2e}")
    median = float(np.median([r for r, _ in rels]))
    ok = rels[0][0] <= a.tol
    print(
        f"{'ok' if ok else 'FAIL'}: median {median:.2e}, worst {rels[0][0]:.2e} over {len(rels)} parameters (tolerance {a.tol:.0e})"
    )
    return 0 if ok else 1


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
