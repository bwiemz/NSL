"""PyTorch twin of the NSLCoder decoder, plus the pieces every check here shares.

The model mirrors `models/coder*/model.nsl` and `stdlib/nsl/nn/{gqa,rope,norms}.nsl`
operation for operation. NSL writes a linear layer as `x @ W` with `W` shaped
`[in, out]`; the twin keeps that layout, so a checkpoint's tensors load without
transposes and a gradient or an update can be compared tensor against tensor.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
NSLM_DTYPES: dict[str, type[np.floating]] = {"f32": np.float32, "f64": np.float64}


@dataclass(frozen=True)
class Cfg:
    """Architecture constants; the defaults are Coder-50M's."""

    vocab: int = 49152
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 1408
    max_seq: int = 1024
    theta: float = 10000.0
    dropout: float = 0.0
    eps: float = 1e-5


SIZES: dict[str, tuple[str, Cfg]] = {
    "50m": ("coder50m", Cfg()),
    "500m": (
        "coder500m",
        Cfg(d_model=1280, n_layers=24, n_heads=20, n_kv_heads=10, d_ff=3520),
    ),
    "1b": (
        "coder1b",
        Cfg(
            d_model=2048,
            n_layers=16,
            n_heads=32,
            n_kv_heads=8,
            d_ff=8192,
            max_seq=2048,
            theta=500000.0,
        ),
    ),
}


def read_nslm(path: str | Path) -> dict[str, np.ndarray]:
    """Read an `.nslm` file (`model_save` or `checkpoint_save`) into name -> f32 array.

    Names are canonicalised to `model_save`'s form: `blocks[0].attn.wq`. A training
    checkpoint names tensors through the model variable (`m.blocks.0.attn.wq`).
    """
    data = Path(path).read_bytes()
    if data[:4] != b"NSLM":
        raise ValueError(f"{path}: not an NSLM file")
    (header_len,) = struct.unpack_from("<Q", data, 8)
    header = json.loads(data[16 : 16 + header_len])
    total = 16 + header_len
    data_start = total + (
        (64 - (total % 64)) % 64
    )  # the writer 64-aligns the data section
    out: dict[str, np.ndarray] = {}
    for p in header["params"]:
        dtype = NSLM_DTYPES[p["dtype"]]
        count = p["nbytes"] // np.dtype(dtype).itemsize
        arr = np.frombuffer(
            data, dtype=dtype, count=count, offset=data_start + p["offset"]
        )
        name = re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", p["name"])
        out[name.removeprefix("m.")] = arr.reshape(p["shape"]).astype(np.float32)
    return out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class RefModel(torch.nn.Module):
    """NSLCoder. With `weights`, every parameter is taken from an `.nslm` read."""

    def __init__(
        self,
        cfg: Cfg,
        weights: dict[str, np.ndarray] | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator().manual_seed(0)
        self.params = torch.nn.ParameterDict()
        c = cfg
        dq, dkv = c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim
        depth = 1.0 / math.sqrt(2 * c.n_layers)

        def add(name: str, shape: list[int], std: float | None) -> None:
            if weights is not None:
                t = torch.from_numpy(np.array(weights[name])).reshape(shape)
            elif std is None:
                t = torch.ones(shape)
            else:
                t = torch.randn(shape, generator=gen) * std
            self.params[name.replace(".", "/")] = torch.nn.Parameter(t.to(device))

        add("embed", [c.vocab, c.d_model], 0.02)
        for i in range(c.n_layers):
            b = f"blocks[{i}]"
            add(f"{b}.attn_norm.weight", [c.d_model], None)
            add(f"{b}.attn.wq", [c.d_model, dq], math.sqrt(2 / (c.d_model + dq)))
            add(f"{b}.attn.wk", [c.d_model, dkv], math.sqrt(2 / (c.d_model + dkv)))
            add(f"{b}.attn.wv", [c.d_model, dkv], math.sqrt(2 / (c.d_model + dkv)))
            add(
                f"{b}.attn.wo", [dq, c.d_model], math.sqrt(2 / (c.d_model + dq)) * depth
            )
            add(f"{b}.ffn_norm.weight", [c.d_model], None)
            add(
                f"{b}.ffn.w_gate",
                [c.d_model, c.d_ff],
                math.sqrt(2 / (c.d_model + c.d_ff)),
            )
            add(
                f"{b}.ffn.w_up",
                [c.d_model, c.d_ff],
                math.sqrt(2 / (c.d_model + c.d_ff)),
            )
            add(
                f"{b}.ffn.w_down",
                [c.d_ff, c.d_model],
                math.sqrt(2 / (c.d_model + c.d_ff)) * depth,
            )
        add("norm.weight", [c.d_model], None)

        inv_freq = torch.exp(
            torch.arange(0, c.head_dim, 2, dtype=torch.float32)
            * (-math.log(c.theta) / c.head_dim)
        )
        freqs = (
            torch.arange(c.max_seq, dtype=torch.float32)[:, None] * inv_freq[None, :]
        )
        table = torch.cat([freqs, freqs], dim=1)
        # Plain attributes rather than buffers: the model is built on its device
        # and never moved, and a checker can see the type of an assignment.
        self.cos: torch.Tensor = table.cos().to(device)
        self.sin: torch.Tensor = table.sin().to(device)

    def p(self, name: str) -> torch.Tensor:
        return self.params[name.replace(".", "/")]

    def named(self) -> dict[str, torch.Tensor]:
        """Parameters under their `.nslm` names."""
        return {k.replace("/", "."): v for k, v in self.params.items()}

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        batch, seq = ids.shape
        x = self.p("embed")[ids]
        cos, sin = self.cos[:seq][None, None], self.sin[:seq][None, None]
        groups = c.n_heads // c.n_kv_heads
        for i in range(c.n_layers):
            b = f"blocks[{i}]"
            h = rmsnorm(x, self.p(f"{b}.attn_norm.weight"), c.eps)
            q = (
                (h @ self.p(f"{b}.attn.wq"))
                .view(batch, seq, c.n_heads, c.head_dim)
                .transpose(1, 2)
            )
            k = (
                (h @ self.p(f"{b}.attn.wk"))
                .view(batch, seq, c.n_kv_heads, c.head_dim)
                .transpose(1, 2)
            )
            v = (
                (h @ self.p(f"{b}.attn.wv"))
                .view(batch, seq, c.n_kv_heads, c.head_dim)
                .transpose(1, 2)
            )
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(batch, seq, c.n_heads * c.head_dim) @ self.p(
                f"{b}.attn.wo"
            )
            x = x + F.dropout(a, c.dropout, self.training)
            h = rmsnorm(x, self.p(f"{b}.ffn_norm.weight"), c.eps)
            gate = F.silu(h @ self.p(f"{b}.ffn.w_gate"))
            f = (gate * (h @ self.p(f"{b}.ffn.w_up"))) @ self.p(f"{b}.ffn.w_down")
            x = x + F.dropout(f, c.dropout, self.training)
        return rmsnorm(x, self.p("norm.weight"), c.eps)

    def loss(self, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.hidden(ids) @ self.p("embed").t()  # tied LM head
        return F.cross_entropy(
            logits.view(-1, self.cfg.vocab), labels.view(-1), ignore_index=-100
        )


def grad_of(p: torch.Tensor) -> np.ndarray:
    """A parameter's gradient as f64, refusing a parameter backward never reached."""
    if p.grad is None:
        raise ValueError("parameter received no gradient")
    return p.grad.detach().cpu().numpy().astype(np.float64)


def batch_from_stream(
    tokens: np.ndarray, index: int, batch: int, seq: int, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch `index` as NSL's `DataLoader(shuffle=false, drop_last=true)` builds it.

    Rows are consecutive non-overlapping `seq`-token windows of the stream; the labels
    are the inputs shifted by one inside the window, with the last position ignored.
    """
    rows = [
        tokens[(index * batch + b) * seq : (index * batch + b + 1) * seq].astype(
            np.int64
        )
        for b in range(batch)
    ]
    ids = torch.from_numpy(np.stack(rows)).to(device)
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    return ids, labels


def warmup_cosine(
    base: float, step: float, warmup: float, total: float, min_lr: float
) -> float:
    """`stdlib/nsl/optim/schedulers.nsl::warmup_cosine`."""
    if step < warmup:
        return base * (step / warmup) if warmup > 0 else base
    if total <= warmup:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    if progress >= 1.0:
        return min_lr
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# --- driving the NSL side --------------------------------------------------------------

TRAIN_PROGRAM = """from model import NSLCoder
from nsl.nn.losses import cross_entropy

let m = NSLCoder()
{load}m.to(cuda)
{save_init}let tokens = load_mmap("tokens.bin", 3)
let loader = DataLoader(tokens, batch_size={batch}, seq_len={seq}, shuffle=false, drop_last=true)

@fused_lm_ce(enabled=true, vocab_size={vocab}, hidden_size={d_model}, batch_size={batch}, seq_len={seq})
train(model=m, epochs=1{train_args}):
    optimizer: {optimizer}
{scheduler}    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let flat_logits = logits.reshape([{rows}, {vocab}])
        let flat_labels = batch.labels.reshape([{rows}])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(loss)

model_save(m, "after.nslm")
print("REFCHECK_DONE")
"""


def write_model(size: str, workdir: Path) -> Cfg:
    """Copy the size's `model.nsl` into `workdir` with dropout forced to 0.

    The two frameworks cannot share a dropout mask, so every check here runs without
    one; the copy also keeps the run off the model directory's toolchain pin.
    """
    directory, cfg = SIZES[size]
    src = (REPO / "models" / directory / "model.nsl").read_text()
    src = re.sub(
        r"^const DROPOUT = [0-9.]+", "const DROPOUT = 0.0", src, flags=re.MULTILINE
    )
    block = re.search(r"TransformerBlock\((\d+(?:, [0-9.]+)*)\)", src)
    if block is None:
        raise ValueError(
            f"{directory}/model.nsl: no TransformerBlock(...) literal list"
        )
    args = block.group(1).split(", ")
    args[5] = "0.0"  # dropout_p
    src = src.replace(block.group(0), f"TransformerBlock({', '.join(args)})")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.nsl").write_text(src)
    return cfg


def run_nsl(nsl: str, workdir: Path, program: str, flags: list[str]) -> list[float]:
    """Run `program` from `workdir`; return the losses `on_step` printed."""
    (workdir / "prog.nsl").write_text(program)
    env = dict(os.environ)
    env.setdefault("NSL_STDLIB_PATH", str(REPO / "stdlib"))
    done = subprocess.run(
        [nsl, "run", *flags, "prog.nsl"],
        check=False,  # the output is inspected below: a failed run prints why
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
    )
    if "REFCHECK_DONE" not in done.stdout:
        raise RuntimeError(
            f"nsl run failed (exit {done.returncode}):\n{(done.stdout + done.stderr)[-3000:]}"
        )
    return [
        float(line[8:-2])
        for line in done.stdout.splitlines()
        if line.startswith("tensor([")
    ]
