"""PyTorch (float64 autograd) reference for the CSHA fused attention backward.

Prints the literals pinned in crates/nsl-codegen/tests/csha_f64_reference.rs.
Reproduces that file's deterministic inputs (`det_seq`, an LCG) and
differentiates

    x_norm = x / sqrt(mean(x^2) + eps) * gamma
    Q = x_norm @ Wq ; K = x_norm @ Wk ; V = x_norm @ Wv
    O = softmax(rope(Q) rope(K)^T * scale [+ causal mask]) V      (rope optional)
    L = sum(O * dO)

with respect to Q, K, V, Wq, Wk, Wv, x and x_norm. For each configuration it
prints the L2 norm and three sampled elements of every tensor as Rust
literals:

    python tools/refcheck/csha_backward_literals.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def det_seq(seed: int, n: int) -> np.ndarray:
    s = seed & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = np.float32(np.float32(s >> 16) / np.float32(65535.0)) - np.float32(0.5)
    return out


def gamma_seq(n: int) -> np.ndarray:
    # 0.75 + 0.5 * (u + 0.5), u = det_seq(46): values in [0.75, 1.25].
    u = det_seq(46, n)
    return (np.float32(0.75) + np.float32(0.5) * (u + np.float32(0.5))).astype(np.float32)


def rope_tables(seq: int, hd: int) -> tuple[np.ndarray, np.ndarray]:
    # A rotation per (position, pair) built from IEEE-exact f32 operations
    # only (multiply, subtract, sqrt), so NumPy and Rust produce the same
    # bits: c = 1.8 * det_seq(47) in [-0.9, 0.9], s = sqrt(1 - c*c).
    c = (det_seq(47, seq * hd // 2) * np.float32(1.8)).astype(np.float32)
    s = np.sqrt(np.float32(1.0) - c * c).astype(np.float32)
    return c.reshape(seq, hd // 2), s.reshape(seq, hd // 2)


def rope_adjacent(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Pairs (2i, 2i+1): y0 = x0 cos - x1 sin ; y1 = x0 sin + x1 cos.
    x0 = t[..., 0::2]
    x1 = t[..., 1::2]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack((y0, y1), dim=-1).reshape(t.shape)


def run(seq: int, hd: int, dm: int, causal: bool, do_mag: float, batch: int = 1, rope: bool = False) -> dict[str, np.ndarray]:
    eps = float(np.float32(1e-5))
    scale = float(np.float32(1.0) / np.sqrt(np.float32(hd)))
    x_np = det_seq(42, batch * seq * dm).reshape(batch, seq, dm)
    wq_np = det_seq(43, dm * hd).reshape(dm, hd)
    wk_np = det_seq(44, dm * hd).reshape(dm, hd)
    wv_np = det_seq(45, dm * hd).reshape(dm, hd)
    g_np = gamma_seq(dm)
    do_np = (det_seq(99, batch * seq * hd) * np.float32(do_mag)).astype(np.float32).reshape(batch, seq, hd)

    x = torch.tensor(x_np.astype(np.float64), requires_grad=True)
    wq = torch.tensor(wq_np.astype(np.float64), requires_grad=True)
    wk = torch.tensor(wk_np.astype(np.float64), requires_grad=True)
    wv = torch.tensor(wv_np.astype(np.float64), requires_grad=True)
    g = torch.tensor(g_np.astype(np.float64))
    d_o = torch.tensor(do_np.astype(np.float64))

    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    xn = x / rms * g
    xn.retain_grad()
    q = xn @ wq
    k = xn @ wk
    v = xn @ wv
    for t in (q, k, v):
        t.retain_grad()
    if rope:
        cos_np, sin_np = rope_tables(seq, hd)
        cos = torch.tensor(cos_np.astype(np.float64))
        sin = torch.tensor(sin_np.astype(np.float64))
        q_att = rope_adjacent(q, cos, sin)
        k_att = rope_adjacent(k, cos, sin)
    else:
        q_att, k_att = q, k
    s = (q_att @ k_att.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = p @ v
    loss = (o * d_o).sum()
    loss.backward()
    return {
        "o": o.detach().numpy().reshape(-1),
        "dq": q.grad.numpy().reshape(-1),
        "dk": k.grad.numpy().reshape(-1),
        "dv": v.grad.numpy().reshape(-1),
        "dwq": wq.grad.numpy().reshape(-1),
        "dwk": wk.grad.numpy().reshape(-1),
        "dwv": wv.grad.numpy().reshape(-1),
        "dx": x.grad.numpy().reshape(-1),
        "dx_norm": xn.grad.numpy().reshape(-1),
    }


def main() -> int:
    configs = [
        # (name, seq, hd, dm, causal, do_mag, batch, rope)
        ("S32_HD32_NONCAUSAL_1E6", 32, 32, 32, False, 1e-6, 1, False),
        ("S32_HD32_CAUSAL_1E7", 32, 32, 32, True, 1e-7, 1, False),
        ("S128_HD32_CAUSAL_1E6", 128, 32, 32, True, 1e-6, 1, False),
        ("S32_HD32_DM64_CAUSAL_1E6", 32, 32, 64, True, 1e-6, 1, False),
        ("S32_HD32_ROPE_CAUSAL_1E6", 32, 32, 32, True, 1e-6, 1, True),
        ("S32_HD32_BATCH2_CAUSAL_1E6", 32, 32, 32, True, 1e-6, 2, False),
    ]
    for name, seq, hd, dm, causal, mag, batch, rope in configs:
        r = run(seq, hd, dm, causal, mag, batch, rope)
        print(f"// {name}: seq={seq} hd={hd} d_model={dm} causal={causal} dO x {mag:g} batch={batch} rope={rope}")
        print(f"const {name}: &[(&str, f64, [(usize, f64); 3])] = &[")
        for key in ("o", "dq", "dk", "dv", "dwq", "dwk", "dwv", "dx", "dx_norm"):
            a = r[key]
            n = a.shape[0]
            idx = [1, n // 2 + 3, n - 2]
            norm = float(np.sqrt((a * a).sum()))
            samples = ", ".join(f"({i}, {float(a[i])!r})" for i in idx)
            print(f'    ("{key}", {norm!r}, [{samples}]),')
        print("];")
    return 0


if __name__ == "__main__":
    sys.exit(main())
