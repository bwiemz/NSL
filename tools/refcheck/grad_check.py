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

bf16 activation storage (`--precision bf16` writes `precision="bf16"` into the
train header) is held to PyTorch autocast, not to f32: with `--autocast bf16`
the twin also runs under `torch.autocast(bf16)` and, for each of `--batches`
batches, three per-tensor errors are recorded: (a) autocast vs f32, (b) NSL vs
f32, (c) NSL vs autocast. The exit status is then the criterion fixed before
any run (design 6.2): for every batch, NSL's median and worst error in (b) are
no larger than the largest corresponding value of (a) over the batches.

    tools/refcheck/grad_check.py --size 50m --tokens corpus.bin --nsl nsl \\
        --precision bf16 --autocast bf16 --batches 4 -- --source-ad
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
    Spread,
    autocast,
    batch_from_stream,
    bf16_criterion,
    grad_of,
    read_nslm,
    relative_errors,
    run_nsl,
    write_model,
)


def reference_grads(
    cfg,
    init: dict[str, np.ndarray],
    tokens: np.ndarray,
    batch: int,
    seq: int,
    mode: str | None,
) -> tuple[float, dict[str, np.ndarray]]:
    """The twin's loss and per-parameter gradient (f64) on batch 0 of `tokens`,
    from `init`, optionally under autocast."""
    model = RefModel(cfg, init)
    model.train()
    with autocast(mode):
        loss = model.loss(*batch_from_stream(tokens, 0, batch, seq))
    loss.backward()
    grads = {name: grad_of(p) for name, p in model.named().items()}
    value = loss.item()
    del model
    torch.cuda.empty_cache()
    return value, grads


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
    ap.add_argument(
        "--batches",
        type=int,
        default=1,
        help="consecutive batches of the stream, one NSL run each",
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
    n = a.batch * seq
    stream = np.fromfile(a.tokens, dtype=np.uint16)[: a.batches * n]
    if len(stream) < a.batches * n:
        raise SystemExit(
            f"{a.tokens}: need {a.batches * n} tokens, found {len(stream)}"
        )
    train_args = ', precision="bf16"' if a.precision == "bf16" else ""
    program = TRAIN_PROGRAM.format(
        load="",
        save_init='model_save(m, "init.nslm")\n',
        batch=a.batch,
        seq=seq,
        rows=a.batch * seq,
        vocab=cfg.vocab,
        d_model=cfg.d_model,
        train_args=train_args,
        optimizer="SGD(lr=1.0)",
        scheduler="",
    )

    failed = False
    spreads: dict[str, list[Spread]] = {"a": [], "b": [], "c": []}
    for i in range(a.batches):
        tokens = stream[i * n : (i + 1) * n]
        tokens.tofile(a.workdir / "tokens.bin")
        nsl_loss = run_nsl(a.nsl, a.workdir, program, a.nsl_flags)[0]
        init, after = (
            read_nslm(a.workdir / "init.nslm"),
            read_nslm(a.workdir / "after.nslm"),
        )
        loss, g_ref = reference_grads(cfg, init, tokens, a.batch, seq, None)
        g_nsl = {
            name: init[name].astype(np.float64) - after[name].astype(np.float64)
            for name in g_ref
        }
        print(f"batch {i}: loss nsl {nsl_loss:.6f}  torch {loss:.6f}")
        rel_b = relative_errors(g_nsl, g_ref)
        rows = sorted(((rel_b[k], k) for k in g_ref), reverse=True)
        print(
            f"{'parameter':32s} {'|g| torch':>11s} {'|g| nsl':>11s} {'cosine':>9s} {'rel err':>9s}"
        )
        for rel, name in rows if a.all else rows[:10]:
            r, x = g_ref[name], g_nsl[name]
            n_ref, n_nsl = float(np.linalg.norm(r)), float(np.linalg.norm(x))
            cos = float((r * x).sum() / (n_ref * n_nsl + 1e-300))
            print(f"{name:32s} {n_ref:11.4e} {n_nsl:11.4e} {cos:9.6f} {rel:9.2e}")
        # A tensor the reference does not train must not have moved.
        moved = [
            k for k in init if k not in g_ref and np.abs(init[k] - after[k]).max() > 0
        ]
        if moved:
            print(
                f"FAIL: NSL updated tensors that are not parameters of the reference: {moved[:8]}"
            )
            failed = True
        b = Spread.of(rel_b)
        spreads["b"].append(b)
        if a.autocast:
            loss_ac, g_ac = reference_grads(cfg, init, tokens, a.batch, seq, a.autocast)
            spreads["a"].append(Spread.of(relative_errors(g_ac, g_ref)))
            spreads["c"].append(Spread.of(relative_errors(g_nsl, g_ac)))
            print(f"batch {i}: torch autocast loss {loss_ac:.6f}")
            for key, label in (
                ("a", "(a) autocast vs f32"),
                ("b", "(b) NSL vs f32     "),
                ("c", "(c) NSL vs autocast"),
            ):
                sp = spreads[key][-1]
                print(
                    f"batch {i}: {label}  median {sp.median:.2e}  worst {sp.worst:.2e}"
                )
        else:
            verdict = "ok" if b.worst <= a.tol else "FAIL"
            failed |= verdict == "FAIL"
            print(
                f"{verdict}: {len(rel_b)} parameters, median relative error {b.median:.2e}, worst {b.worst:.2e} (tolerance {a.tol:.0e})"
            )
    if a.autocast:
        ok, line = bf16_criterion(spreads["b"], spreads["a"])
        print(line)
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
