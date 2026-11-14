"""PyTorch (float64 autograd) reference for examples/csha_toy_pretrain_hd32.nsl.

Prints the literals pinned in crates/nsl-cli/tests/csha_toy_pretrain_gate.rs:
the loss of the example's one training step and, for each of Wq, Wk, Wv and
the RMSNorm weight, `sum(d * d)` and `sum(d * probe)` of the SGD update `d`
the step applies, each with the scale its error is judged against. Inputs are rebuilt exactly as the example builds them
(`sin` of an f32 `arange`).

    python tools/refcheck/csha_toy_literals.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
f = np.float32


def s(n: int, c: float, ph: float = 0.0) -> np.ndarray:
    a = np.arange(n, dtype=np.float32) * f(c)
    if ph:
        a = a + f(ph)
    return np.sin(a).astype(np.float32)


def run(B: int, H: int, lr: float) -> list[tuple[float, float]]:
    n = B * H * 1024
    g0 = (f(1.0) + f(0.25) * s(32, 0.9)).astype(np.float32)
    w0 = [
        (f(0.3) * s(1024, c, ph)).astype(np.float32).reshape(32, 32)
        for c, ph in ((0.37, 0.0), (0.53, 1.0), (0.71, 2.0))
    ]
    x = torch.tensor(s(n, 0.11).reshape(B, H, 32, 32).astype(np.float64))
    t = torch.tensor(
        (f(1.0) + f(0.5) * s(n, 0.23, 0.5))
        .astype(np.float32)
        .reshape(B, H, 32, 32)
        .astype(np.float64)
    )
    probe = torch.tensor(s(1024, 0.19, 0.3).reshape(32, 32).astype(np.float64))
    gprobe = torch.tensor(s(32, 0.19, 0.3).astype(np.float64))
    g = torch.tensor(g0.astype(np.float64), requires_grad=True)
    wq, wk, wv = (torch.tensor(w.astype(np.float64), requires_grad=True) for w in w0)
    eps = float(f(0.00001))
    xn = x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * g
    q, k, v = xn @ wq, xn @ wk, xn @ wv
    sc = float(f(1.0) / np.sqrt(f(32.0)))
    a = (q @ k.transpose(-1, -2)) * sc
    a = a.masked_fill(
        torch.triu(torch.ones(32, 32, dtype=torch.bool), 1), float("-inf")
    )
    out = torch.softmax(a, -1) @ v
    loss = (out * out * t).sum() * float(f(0.000001))
    loss.backward()
    # Each statistic with the scale its error is judged against: the value
    # itself for the loss and the squared norms, and |d| * |probe| (the
    # largest the projection could be) for the projections, which are sums
    # with cancellation.
    stats = [(loss.item(), abs(loss.item()))]
    for w, p in ((wq, probe), (wk, probe), (wv, probe), (g, gprobe)):
        d = -lr * w.grad
        dd = (d * d).sum().item()
        stats += [(dd, dd), ((d * p).sum().item(), (dd * (p * p).sum().item()) ** 0.5)]
    return stats


NAMES = (
    "loss",
    "sum(dWq * dWq)",
    "sum(dWq * probe)",
    "sum(dWk * dWk)",
    "sum(dWk * probe)",
    "sum(dWv * dWv)",
    "sum(dWv * probe)",
    "sum(dgamma * dgamma)",
    "sum(dgamma * gprobe)",
)


def main() -> int:
    batch, heads, lr = 1, 1, 1e5
    vals = run(batch, heads, lr)
    print(f"// batch {batch}, {heads} head(s), lr {lr:g}")
    print("const EXPECTED: &Expected = &[")
    for name, (v, scale) in zip(NAMES, vals):
        print(f'    ("{name}", {v!r}, {scale!r}),')
    print("];")
    return 0


if __name__ == "__main__":
    sys.exit(main())
