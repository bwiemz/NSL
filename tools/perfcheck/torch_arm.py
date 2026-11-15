#!/usr/bin/env python3
"""One PyTorch arm of the NSL-vs-PyTorch benchmark: a timed training run of NSLCoder.

The model is `tools/refcheck/ref.py`'s twin (same parameters, same `[in, out]`
layout, same initialisation), run the way a PyTorch user would run it for speed:

* attention is `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`
  pinned to the flash backend (the run refuses to report if another kernel ran);
* RoPE tables are cast to the activations' dtype (the twin multiplies by f32 tables);
* the optimizer is `torch.optim.AdamW(fused=True)` over every parameter, global-norm
  clipping with `clip_grad_norm_(foreach=True)`, `warmup_cosine` set per update;
* the loss is `F.cross_entropy` on materialised `[rows, vocab]` logits. PyTorch has no
  fused linear cross-entropy in core; NSL's `@fused_lm_ce` never materialises them.

Precision (`--precision`):
  tf32  f32 weights and activations, TF32 matmuls. Flash attention needs 16-bit
        operands, so q/k/v are cast to bf16 around the attention call and the output
        back to f32: NSL's "f32" posture likewise runs attention on f16 operands.
  bf16  `torch.autocast(bfloat16)` over the forward, f32 master weights.

Checkpointing (`--ckpt`), per transformer block, non-reentrant:
  none       nothing recomputed.
  blocks     every block recomputed in the backward (NSL `--checkpoint-blocks`).
  selective  matmul and attention outputs saved, everything else recomputed
             (NSL `--checkpoint-selective`: "never recompute matmul-class ops").

Timing: one `loss.item()` per optimizer update (NSL prints the loss once per update,
which synchronises the same way), wall-clock stamp after it. Writes one JSON file.
Synthetic tokens are fine: the step does the same work whatever the ids are.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import re
import sys
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "refcheck"))
from ref import (
    SIZES,
    Cfg,
    RefModel,
    batch_from_stream,
    rmsnorm,
    rotate_half,
    warmup_cosine,
)

# Ops whose outputs the selective policy keeps (aten overloads seen in this model's
# forward under eager, autocast and compile). Everything else is recomputed.
_SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._flash_attention_forward.default,
}


def _selective_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    del ctx, args, kwargs
    return CheckpointPolicy.MUST_SAVE if op in _SAVE_OPS else CheckpointPolicy.PREFER_RECOMPUTE


def model_dropout(directory: str) -> float:
    """`const DROPOUT` of a model directory's `model.nsl`, the value NSL trains with."""
    src = (REPO / "models" / directory / "model.nsl").read_text()
    m = re.search(r"^const DROPOUT = ([0-9.]+)", src, flags=re.MULTILINE)
    if m is None:
        raise ValueError(f"models/{directory}/model.nsl: no `const DROPOUT`")
    return float(m.group(1))


class PerfModel(RefModel):
    """`RefModel` with a block function that can be checkpointed and SDPA-with-GQA."""

    def __init__(self, cfg: Cfg, ckpt: str, attn_cast: torch.dtype | None) -> None:
        super().__init__(cfg, None, "cuda")
        self.ckpt = ckpt
        self.attn_cast = attn_cast
        self.blocks: list[tuple[torch.Tensor, ...]] = [
            tuple(
                self.p(f"blocks[{i}].{n}")
                for n in (
                    "attn_norm.weight",
                    "attn.wq",
                    "attn.wk",
                    "attn.wv",
                    "attn.wo",
                    "ffn_norm.weight",
                    "ffn.w_gate",
                    "ffn.w_up",
                    "ffn.w_down",
                )
            )
            for i in range(cfg.n_layers)
        ]

    def block(self, i: int, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        an, wq, wk, wv, wo, fn, wg, wu, wd = self.blocks[i]
        batch, seq, _ = x.shape
        h = rmsnorm(x, an, c.eps)
        q = (h @ wq).view(batch, seq, c.n_heads, c.head_dim).transpose(1, 2)
        k = (h @ wk).view(batch, seq, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = (h @ wv).view(batch, seq, c.n_kv_heads, c.head_dim).transpose(1, 2)
        cos = self.cos[:seq][None, None].to(q.dtype)
        sin = self.sin[:seq][None, None].to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out_dtype = q.dtype
        if self.attn_cast is not None:
            q, k, v = q.to(self.attn_cast), k.to(self.attn_cast), v.to(self.attn_cast)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        a = a.to(out_dtype).transpose(1, 2).reshape(batch, seq, c.n_heads * c.head_dim)
        x = x + F.dropout(a @ wo, c.dropout, self.training)
        h = rmsnorm(x, fn, c.eps)
        f = (F.silu(h @ wg) * (h @ wu)) @ wd
        return x + F.dropout(f, c.dropout, self.training)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.p("embed")[ids]
        for i in range(self.cfg.n_layers):
            fn = partial(self.block, i)
            if self.ckpt == "none":
                x = fn(x)
            elif self.ckpt == "blocks":
                x = cast(torch.Tensor, checkpoint(fn, x, use_reentrant=False))
            else:
                x = cast(
                    torch.Tensor,
                    checkpoint(
                        fn,
                        x,
                        use_reentrant=False,
                        context_fn=partial(create_selective_checkpoint_contexts, _selective_policy),
                    ),
                )
        return rmsnorm(x, self.p("norm.weight"), self.cfg.eps)


@dataclasses.dataclass
class Result:
    arm: str
    size: str
    precision: str
    compile: bool
    ckpt: str
    micro_batch: int
    seq: int
    accum: int
    tokens_per_update: int
    updates: int
    warmup_updates: int
    round: int
    status: str = "ok"
    error: str | None = None
    stamps: list[float] = dataclasses.field(default_factory=list)
    losses: list[float] = dataclasses.field(default_factory=list)
    phases: list[dict[str, float]] = dataclasses.field(default_factory=list)
    # --phase-timing only: the allocator's peak inside each phase of each update
    # (reset before the phase), in bytes.
    phase_peaks: list[dict[str, int]] = dataclasses.field(default_factory=list)
    alloc_peak_bytes: int | None = None
    reserved_peak_bytes: int | None = None
    attn_kernels: list[str] = dataclasses.field(default_factory=list)
    gemm_kernel_sample: list[str] = dataclasses.field(default_factory=list)
    torch_version: str = torch.__version__
    cuda_version: str | None = torch.version.cuda
    device: str | None = None
    cublas_version: str | None = None


def _profile_kernels(step: Callable[[], None]) -> tuple[list[str], list[str]]:
    """Kernel names one micro-step launches: the attention ones, and a GEMM sample."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    names = {e.name for e in prof.events() or []}
    attn = sorted(n for n in names if re.search(r"flash|fmha|attention|sdpa", n, re.IGNORECASE))
    gemm = sorted(
        n for n in names if re.search(r"gemm|nvjet|cutlass|xmma|sm\d+_", n, re.IGNORECASE)
    )
    return attn, gemm[:8]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True)
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--precision", choices=["tf32", "bf16"], required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ckpt", choices=["none", "blocks", "selective"], required=True)
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--warmup-updates", type=int, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument(
        "--phase-timing",
        action="store_true",
        help="synchronise around forward / backward / optimizer and record each "
        "(an instrumented run: its update times are not the arm's throughput)",
    )
    a = ap.parse_args()

    directory, base_cfg = SIZES[a.size]
    cfg = dataclasses.replace(base_cfg, dropout=model_dropout(directory))
    res = Result(
        arm=a.arm,
        size=a.size,
        precision=a.precision,
        compile=a.compile,
        ckpt=a.ckpt,
        micro_batch=a.micro_batch,
        seq=a.seq,
        accum=a.accum,
        tokens_per_update=a.micro_batch * a.seq * a.accum,
        updates=a.updates,
        warmup_updates=a.warmup_updates,
        round=a.round,
    )

    def write() -> None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dataclasses.asdict(res), indent=1))

    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # NSL's default; bf16 arm unaffected
        torch.backends.cudnn.allow_tf32 = True
        res.device = torch.cuda.get_device_name(0)
        res.cublas_version = importlib.metadata.version("nvidia-cublas-cu12")
        bf16 = a.precision == "bf16"
        model = PerfModel(cfg, a.ckpt, None if bf16 else torch.bfloat16)
        model.train()
        params = list(model.parameters())
        opt = torch.optim.AdamW(
            params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True
        )
        tokens = np.fromfile(a.tokens, dtype=np.uint16)
        need = a.updates * res.tokens_per_update
        if tokens.size < need:
            raise ValueError(f"{a.tokens}: {tokens.size} tokens, need {need}")

        def loss_of(ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                return model.loss(ids, labels)

        loss_fn = torch.compile(loss_of) if a.compile else loss_of
        sync = torch.cuda.synchronize
        reset_peak = torch.cuda.reset_peak_memory_stats
        peak_now = torch.cuda.max_memory_allocated
        micro = 0
        loss = torch.zeros((), device="cuda")

        for update in range(a.updates):
            phase = {"fwd": 0.0, "bwd": 0.0, "opt": 0.0}
            peaks = {"fwd": 0, "bwd": 0, "opt": 0}
            for _ in range(a.accum):
                ids, labels = batch_from_stream(tokens, micro, a.micro_batch, a.seq)
                if a.phase_timing:
                    sync()
                    reset_peak()
                t0 = time.perf_counter()
                loss = loss_fn(ids, labels)
                if a.phase_timing:
                    sync()
                    peaks["fwd"] = max(peaks["fwd"], peak_now())
                    reset_peak()
                t1 = time.perf_counter()
                (loss / a.accum).backward()
                if a.phase_timing:
                    sync()
                    peaks["bwd"] = max(peaks["bwd"], peak_now())
                phase["fwd"] += t1 - t0
                phase["bwd"] += time.perf_counter() - t1
                micro += 1
            if a.phase_timing:
                reset_peak()
            t2 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
            lr = warmup_cosine(1e-4, update, 2, a.updates, 3e-5)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            if a.phase_timing:
                sync()
                phase["opt"] = time.perf_counter() - t2
                peaks["opt"] = peak_now()
                res.phases.append(phase)
                res.phase_peaks.append(peaks)
            res.losses.append(loss.item())  # the once-per-update sync NSL also pays
            res.stamps.append(time.perf_counter())
            if update == 1:
                # Identify the kernels after compilation and allocator warm-up, on a
                # micro-step whose gradient is thrown away before the next update.
                ids, labels = batch_from_stream(tokens, 0, a.micro_batch, a.seq)
                res.attn_kernels, res.gemm_kernel_sample = _profile_kernels(
                    lambda ids=ids, labels=labels: loss_fn(ids, labels).backward()
                )
                opt.zero_grad(set_to_none=True)
                if not any("flash" in k.lower() for k in res.attn_kernels):
                    raise RuntimeError(
                        f"flash attention did not run; attention kernels: {res.attn_kernels}"
                    )
                res.stamps[-1] = time.perf_counter()  # do not bill the probe to update 2
        # With --phase-timing the peak was reset per phase: the whole-run figure is
        # the largest phase peak, recorded below from phase_peaks.
        res.alloc_peak_bytes = (
            max(max(p.values()) for p in res.phase_peaks)
            if res.phase_peaks
            else torch.cuda.max_memory_allocated()
        )
        res.reserved_peak_bytes = torch.cuda.max_memory_reserved()
    except torch.OutOfMemoryError as e:
        res.status = "oom"
        res.error = str(e)[-1500:]
        res.alloc_peak_bytes = torch.cuda.max_memory_allocated()
        res.reserved_peak_bytes = torch.cuda.max_memory_reserved()
    except Exception:  # noqa: BLE001 - the JSON row records any failure; exit code says so
        res.status = "error"
        res.error = traceback.format_exc()[-3000:]
    if res.status == "ok" and not all(math.isfinite(x) for x in res.losses):
        res.status = "error"
        res.error = f"non-finite loss: {res.losses}"
    write()
    print(f"[torch_arm] {a.arm} {res.status}", flush=True)
    return 0 if res.status in ("ok", "oom") else 1


if __name__ == "__main__":
    raise SystemExit(main())
