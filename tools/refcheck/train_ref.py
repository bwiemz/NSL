#!/usr/bin/env python3
"""A PyTorch training run that mirrors an NSL `train` block, for loss-curve comparison.

Same weights (an `.nslm` written by `model_save`), same batches in the same order
(`DataLoader(shuffle=false)`), gradient accumulation as a mean over the window,
global-norm clipping, AdamW decaying every parameter as NSL does without `no_decay`,
and `warmup_cosine` evaluated on the micro-batch index. Prints `micro <n> loss <x>`
with `n` 1-based, the number NSL's `on_step(step, loss)` reports.

    tools/refcheck/train_ref.py --size 500m --init init.nslm --tokens train.bin --val val.bin \\
        --batch 2 --accum 8 --micro-steps 4096 --lr 3e-4 --warmup 200 --total 4096 --min-lr 3e-5 \\
        --wd 0.1 --clip 1.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ref import SIZES, RefModel, batch_from_stream, read_nslm, warmup_cosine


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", choices=sorted(SIZES), default="50m")
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--val", type=Path)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--micro-steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument("--total", type=float, default=0.0, help="0 = constant rate")
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 matmuls, NSL's default arithmetic",
    )
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    cfg = SIZES[a.size][1]
    seq = a.seq or cfg.max_seq
    tokens = np.fromfile(a.tokens, dtype=np.uint16)
    model = RefModel(cfg, read_nslm(a.init))
    model.train()
    params = list(model.params.values())
    opt = torch.optim.AdamW(
        params, lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps, weight_decay=a.wd
    )
    start = time.time()
    for micro in range(a.micro_steps):
        loss = model.loss(*batch_from_stream(tokens, micro, a.batch, seq))
        (loss / a.accum).backward()
        if (micro + 1) % a.print_every == 0:
            print(f"micro {micro + 1} loss {loss.item():.4f}", flush=True)
        if (micro + 1) % a.accum == 0:
            if a.total > 0:
                for group in opt.param_groups:
                    group["lr"] = warmup_cosine(
                        a.lr, micro, a.warmup, a.total, a.min_lr
                    )
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
    print(f"elapsed {time.time() - start:.1f}s")
    if a.val:
        model.eval()
        held_out = np.fromfile(a.val, dtype=np.uint16)
        batches = len(held_out) // (a.batch * seq)
        with torch.no_grad():
            total = sum(
                model.loss(*batch_from_stream(held_out, i, a.batch, seq)).item()
                for i in range(batches)
            )
        print(f"VAL_LOSS {total / batches:.4f} over {batches} batches")


if __name__ == "__main__":
    main()
