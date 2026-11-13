#!/usr/bin/env python3
"""Compare NSL's gradients with PyTorch autograd, parameter by parameter.

One SGD step at lr 1 with no momentum, decay or clipping leaves `init - after`
equal to the gradient the optimizer was handed, so the comparison needs nothing
from NSL but two `model_save`s. The same weights and the same batch then go
through the PyTorch twin.

    tools/refcheck/grad_check.py --size 50m --tokens corpus.bin \\
        --nsl target/release/nsl -- --source-ad --checkpoint-blocks

Everything after `--` is passed to `nsl run`, so each flag combination a recipe
uses can be checked as it is run. Exit status 1 when a parameter's relative
error exceeds `--tol`.
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
    write_model,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", choices=sorted(SIZES), default="50m")
    ap.add_argument(
        "--tokens",
        required=True,
        type=Path,
        help="u16 token stream (load_mmap dtype 3)",
    )
    ap.add_argument("--nsl", default="nsl", help="the nsl binary under test")
    ap.add_argument("--workdir", type=Path, default=Path("refcheck-work/grad"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, help="defaults to the model's max_seq_len")
    ap.add_argument(
        "--tol", type=float, default=5e-3, help="largest acceptable relative error"
    )
    ap.add_argument(
        "--all", action="store_true", help="print every parameter, not the worst ten"
    )
    ap.add_argument(
        "nsl_flags",
        nargs="*",
        default=["--source-ad"],
        help="flags for `nsl run` (after --)",
    )
    a = ap.parse_args()

    cfg = write_model(a.size, a.workdir)
    seq = a.seq or cfg.max_seq
    tokens = np.fromfile(a.tokens, dtype=np.uint16)[: a.batch * seq]
    if len(tokens) < a.batch * seq:
        raise SystemExit(
            f"{a.tokens}: need {a.batch * seq} tokens, found {len(tokens)}"
        )
    tokens.tofile(a.workdir / "tokens.bin")

    program = TRAIN_PROGRAM.format(
        load="",
        save_init='model_save(m, "init.nslm")\n',
        batch=a.batch,
        seq=seq,
        rows=a.batch * seq,
        vocab=cfg.vocab,
        d_model=cfg.d_model,
        train_args="",
        optimizer="SGD(lr=1.0)",
        scheduler="",
    )
    nsl_loss = run_nsl(a.nsl, a.workdir, program, a.nsl_flags)[0]

    init, after = (
        read_nslm(a.workdir / "init.nslm"),
        read_nslm(a.workdir / "after.nslm"),
    )
    model = RefModel(cfg, init)
    model.train()
    loss = model.loss(*batch_from_stream(tokens, 0, a.batch, seq))
    loss.backward()
    print(f"loss: nsl {nsl_loss:.6f}  torch {loss.item():.6f}")

    rows: list[tuple[float, str, float, float, float]] = []
    for name, p in model.named().items():
        g_ref = grad_of(p)
        g_nsl = init[name].astype(np.float64) - after[name].astype(np.float64)
        n_ref, n_nsl = float(np.linalg.norm(g_ref)), float(np.linalg.norm(g_nsl))
        cos = float((g_ref * g_nsl).sum() / (n_ref * n_nsl + 1e-300))
        rows.append(
            (
                float(np.linalg.norm(g_ref - g_nsl)) / (n_ref + 1e-300),
                name,
                n_ref,
                n_nsl,
                cos,
            )
        )
    rows.sort(reverse=True)

    print(
        f"{'parameter':32s} {'|g| torch':>11s} {'|g| nsl':>11s} {'cosine':>9s} {'rel err':>9s}"
    )
    for rel, name, n_ref, n_nsl, cos in rows if a.all else rows[:10]:
        print(f"{name:32s} {n_ref:11.4e} {n_nsl:11.4e} {cos:9.6f} {rel:9.2e}")
    # A tensor the reference does not train must not have moved.
    trained = set(model.named())
    moved = [
        n for n in init if n not in trained and np.abs(init[n] - after[n]).max() > 0
    ]
    if moved:
        print(
            f"FAIL: NSL updated tensors that are not parameters of the reference: {moved[:8]}"
        )
    worst = rows[0][0]
    median = float(np.median([r[0] for r in rows]))
    verdict = "ok" if worst <= a.tol and not moved else "FAIL"
    print(
        f"{verdict}: {len(rows)} parameters, median relative error {median:.2e}, worst {worst:.2e} (tolerance {a.tol:.0e})"
    )
    del model
    torch.cuda.empty_cache()
    return 0 if verdict == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
