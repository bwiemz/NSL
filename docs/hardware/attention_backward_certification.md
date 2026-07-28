# Attention-backward numerical certification

Roadmap item 15. One row per (kernel, forward source, head_dim, block_q,
seq_len, causal, gradient) actually executed against a CPU reference, with the
observed error and the tolerance it was checked against — the format
[`README.md`](README.md) requires before a capability may be called Validated.

Nothing of this shape existed before. The gates had been *run* — the item-19
certification lane executed them, and one surfaced a real launcher bug (49 args
to a 50-param kernel, see `ci/gpu-cert-known-red.txt`) that was then fixed — but
the results were never **banked**. So `cuda_status.md` carried "numerics
pending" for kernels that were in fact correct, and nothing recorded WHICH
regimes had been checked.

## Reference hardware

| field | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (sm_120) |
| Driver | 610.43.03 |
| CUDA | 13.3 |
| Date | 2026-07-26 |
| Commit | see `git log` for the commit adding this file |

**The Tier B.2 emitters hardcode `.target sm_80`** (`flash_attention_v2/tier_b2/
backward/dq.rs`), so the PTX here is JIT-recompiled to sm_120. A genuine
sm_80 (A100) or sm_90 (H100) row is therefore **physically unobtainable on this
box** — see [What is NOT certified](#what-is-not-certified).

## Reproduce

```bash
cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
    --test tier_b2_dkdv_kernel_cpu_reference \
    --test tier_b2_full_backward_cpu_reference \
    -- --ignored --nocapture --test-threads=1
```

The canary subset runs via `tools/gpu-test.sh` (POSIX; the pre-existing
`tools/gpu-test.ps1` needs pwsh, which is not installed on this box, so the
manifest could not previously be executed here at all).

## Results — all 68 rows PASS

Worst observed margin is **52.8% of tolerance**
(`tier_b2_full_backward_sweep_b1_forward`, dWq at head_dim 128), followed by
38.1%, 28.2%, 17.1% and 14.5%; 54 of the 68 rows are under 10%. "margin"
is `rel err / rel tol`, so lower is more headroom.

| test | kernel | fwd source | head_dim | block_q | seq | causal | grad | max\|gpu\| | max\|ref\| | max_abs | rel err | rel tol | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tier_b2_dkdv_sweep_b1_forward` | validate_dkdv | B1Forward | 32 | 32 | 32 | no | dV | 6.136946e0 | 6.129289e0 | 1.013422e-2 | 1.6534e-3 | 3.0000e-2 | 5.5% |
| `tier_b2_dkdv_sweep_b1_forward` | validate_dkdv | B1Forward | 32 | 32 | 32 | no | dK | 2.603539e-3 | 2.605030e-3 | 3.543450e-6 | 1.3602e-3 | 3.0000e-2 | 4.5% |
| `tier_b2_dkdv_sweep_b1_forward` | validate_dkdv | B1Forward | 64 | 32 | 32 | no | dV | 4.189002e0 | 4.186460e0 | 8.784771e-3 | 2.0984e-3 | 5.0000e-2 | 4.2% |
| `tier_b2_dkdv_sweep_b1_forward` | validate_dkdv | B1Forward | 64 | 32 | 32 | no | dK | 1.773856e-3 | 1.774763e-3 | 1.779583e-6 | 1.0027e-3 | 5.0000e-2 | 2.0% |
| `tier_b2_dkdv_sweep_b1_forward` | validate_dkdv | B1Forward | 128 | 32 | 32 | no | dV | 1.445690e0 | 1.443081e0 | 9.134710e-3 | 6.3300e-3 | 8.0000e-2 | 7.9% |
| `tier_b2_dkdv_sweep_b1_forward` | validate_dkdv | B1Forward | 128 | 32 | 32 | no | dK | 4.373284e-4 | 4.376689e-4 | 1.262961e-6 | 2.8857e-3 | 8.0000e-2 | 3.6% |
| `tier_b2_dkdv_sweep_cpu_naive` | validate_dkdv | CpuNaive | 32 | 64 | 64 | no | dV | 3.820273e0 | 3.818316e0 | 7.799506e-3 | 2.0427e-3 | 3.0000e-2 | 6.8% |
| `tier_b2_dkdv_sweep_cpu_naive` | validate_dkdv | CpuNaive | 32 | 64 | 64 | no | dK | 1.954588e-2 | 1.953068e-2 | 6.571505e-5 | 3.3647e-3 | 3.0000e-2 | 11.2% |
| `tier_b2_dkdv_sweep_cpu_naive` | validate_dkdv | CpuNaive | 64 | 64 | 64 | no | dV | 9.749866e-1 | 9.701702e-1 | 6.052434e-3 | 6.2385e-3 | 5.0000e-2 | 12.5% |
| `tier_b2_dkdv_sweep_cpu_naive` | validate_dkdv | CpuNaive | 64 | 64 | 64 | no | dK | 6.130722e-3 | 6.130070e-3 | 9.316020e-6 | 1.5197e-3 | 5.0000e-2 | 3.0% |
| `tier_b2_dkdv_sweep_cpu_naive` | validate_dkdv | CpuNaive | 128 | 32 | 128 | no | dV | 7.907779e-1 | 7.882419e-1 | 4.390419e-3 | 5.5699e-3 | 8.0000e-2 | 7.0% |
| `tier_b2_dkdv_sweep_cpu_naive` | validate_dkdv | CpuNaive | 128 | 32 | 128 | no | dK | 1.554616e-3 | 1.555941e-3 | 8.043135e-6 | 5.1693e-3 | 8.0000e-2 | 6.5% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dQ | 5.459595e-2 | 5.460314e-2 | 1.823902e-5 | 3.3403e-4 | 5.0000e-2 | 0.7% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dK | 1.380157e-2 | 1.380299e-2 | 1.457892e-5 | 1.0562e-3 | 5.0000e-2 | 2.1% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dV | 4.320312e0 | 4.315096e0 | 8.820772e-3 | 2.0442e-3 | 5.0000e-2 | 4.1% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dWq | 5.627441e-2 | 5.626123e-2 | 1.357086e-4 | 2.4121e-3 | 2.0000e-2 | 12.1% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dWk | 1.089478e-2 | 1.089810e-2 | 3.151642e-5 | 2.8919e-3 | 2.0000e-2 | 14.5% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dWv | 4.909375e1 | 4.910469e1 | 2.845383e-2 | 5.7945e-4 | 2.0000e-2 | 2.9% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 64 | 32 | 32 | no | dx | 1.299741e1 | 1.300292e1 | 1.362586e-2 | 1.0479e-3 | 2.0000e-2 | 5.2% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dQ | 6.252289e-3 | 6.252356e-3 | 2.339017e-6 | 3.7410e-4 | 8.0000e-2 | 0.5% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dK | 4.372597e-4 | 4.376689e-4 | 1.375825e-6 | 3.1435e-3 | 8.0000e-2 | 3.9% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dV | 1.445312e0 | 1.443081e0 | 9.297371e-3 | 6.4427e-3 | 8.0000e-2 | 8.1% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dWq | 1.436234e-3 | 1.432262e-3 | 1.512445e-5 | 1.0560e-2 | 2.0000e-2 | 52.8% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dWk | 3.867149e-4 | 3.866363e-4 | 1.104272e-6 | 2.8561e-3 | 2.0000e-2 | 14.3% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dWv | 2.093750e1 | 2.093644e1 | 1.074219e-2 | 5.1309e-4 | 2.0000e-2 | 2.6% |
| `tier_b2_full_backward_sweep_b1_forward` | full_backward | B1Forward | 128 | 32 | 32 | no | dx | 4.514357e0 | 4.515089e0 | 3.067017e-3 | 6.7928e-4 | 3.0000e-2 | 2.3% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dQ | 5.499268e-2 | 5.499138e-2 | 2.151728e-5 | 3.9128e-4 | 5.0000e-2 | 0.8% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dK | 1.236572e-1 | 1.236650e-1 | 4.039705e-5 | 3.2667e-4 | 5.0000e-2 | 0.7% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dV | 8.781250e1 | 8.778739e1 | 3.139496e-2 | 3.5762e-4 | 5.0000e-2 | 0.7% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dWq | 6.237793e-2 | 6.240517e-2 | 1.372956e-4 | 2.2001e-3 | 2.0000e-2 | 11.0% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dWk | 2.109375e-1 | 2.109715e-1 | 1.500100e-4 | 7.1104e-4 | 2.0000e-2 | 3.6% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dWv | 1.268750e2 | 1.268720e2 | 7.826996e-2 | 6.1692e-4 | 2.0000e-2 | 3.1% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 64 | 32 | 32 | yes | dx | 1.458984e2 | 1.458295e2 | 7.572937e-2 | 5.1930e-4 | 2.0000e-2 | 2.6% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dQ | 6.214142e-3 | 6.214825e-3 | 2.846587e-6 | 4.5803e-4 | 8.0000e-2 | 0.6% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dK | 6.885529e-3 | 6.884537e-3 | 3.259163e-6 | 4.7340e-4 | 8.0000e-2 | 0.6% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dV | 7.012500e1 | 7.009580e1 | 4.175568e-2 | 5.9569e-4 | 8.0000e-2 | 0.7% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dWq | 3.089905e-3 | 3.088857e-3 | 1.742598e-5 | 5.6416e-3 | 2.0000e-2 | 28.2% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dWk | 1.299858e-3 | 1.301187e-3 | 9.922893e-6 | 7.6260e-3 | 2.0000e-2 | 38.1% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dWv | 8.587500e1 | 8.588984e1 | 8.929443e-2 | 1.0396e-3 | 2.0000e-2 | 5.2% |
| `tier_b2_full_backward_sweep_b1_forward_causal` | full_backward | B1Forward | 128 | 32 | 32 | yes | dx | 4.021903e1 | 4.012018e1 | 1.085396e-1 | 2.7054e-3 | 3.0000e-2 | 9.0% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dQ | 2.079010e-3 | 2.081912e-3 | 9.325799e-6 | 4.4794e-3 | 5.0000e-2 | 9.0% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dK | 6.130219e-3 | 6.130070e-3 | 1.110043e-5 | 1.8108e-3 | 5.0000e-2 | 3.6% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dV | 9.750977e-1 | 9.701702e-1 | 6.144941e-3 | 6.3339e-3 | 5.0000e-2 | 12.7% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dWq | 4.234314e-3 | 4.226455e-3 | 7.913448e-6 | 1.8724e-3 | 2.0000e-2 | 9.4% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dWk | 7.469177e-3 | 7.471053e-3 | 2.145115e-5 | 2.8712e-3 | 2.0000e-2 | 14.4% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dWv | 1.960938e1 | 1.961393e1 | 9.611130e-3 | 4.9002e-4 | 2.0000e-2 | 2.5% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 64 | 64 | 64 | no | dx | 9.821590e-1 | 9.826829e-1 | 1.402915e-3 | 1.4276e-3 | 2.0000e-2 | 7.1% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dQ | 4.974365e-3 | 4.980855e-3 | 1.065247e-5 | 2.1387e-3 | 8.0000e-2 | 2.7% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dK | 6.305695e-3 | 6.309053e-3 | 1.336448e-5 | 2.1183e-3 | 8.0000e-2 | 2.6% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dV | 1.432617e0 | 1.431973e0 | 8.382797e-3 | 5.8540e-3 | 8.0000e-2 | 7.3% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dWq | 7.133484e-3 | 7.142451e-3 | 1.688767e-5 | 2.3644e-3 | 2.0000e-2 | 11.8% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dWk | 4.756927e-3 | 4.760585e-3 | 1.624599e-5 | 3.4126e-3 | 2.0000e-2 | 17.1% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dWv | 2.093750e1 | 2.094216e1 | 1.170540e-2 | 5.5894e-4 | 2.0000e-2 | 2.8% |
| `tier_b2_full_backward_sweep_cpu_naive` | full_backward | CpuNaive | 128 | 32 | 32 | no | dx | 4.516253e0 | 4.514808e0 | 2.433300e-3 | 5.3896e-4 | 3.0000e-2 | 1.8% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dQ | 1.396484e-1 | 1.396940e-1 | 5.888939e-5 | 4.2156e-4 | 5.0000e-2 | 0.8% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dK | 6.045532e-2 | 6.044530e-2 | 3.251806e-5 | 5.3797e-4 | 5.0000e-2 | 1.1% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dV | 8.525000e1 | 8.522300e1 | 5.315399e-2 | 6.2370e-4 | 5.0000e-2 | 1.2% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dWq | 1.436768e-1 | 1.436429e-1 | 1.581907e-4 | 1.1013e-3 | 2.0000e-2 | 5.5% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dWk | 7.067871e-2 | 7.067695e-2 | 6.511062e-5 | 9.2124e-4 | 2.0000e-2 | 4.6% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dWv | 1.242500e2 | 1.242346e2 | 8.303070e-2 | 6.6834e-4 | 2.0000e-2 | 3.3% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 64 | 64 | 64 | yes | dx | 1.396256e2 | 1.396058e2 | 1.010243e-1 | 7.2364e-4 | 2.0000e-2 | 3.6% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dQ | 9.826660e-2 | 9.824289e-2 | 5.297363e-5 | 5.3921e-4 | 8.0000e-2 | 0.7% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dK | 4.403687e-2 | 4.403755e-2 | 4.732609e-5 | 1.0747e-3 | 8.0000e-2 | 1.3% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dV | 7.062500e1 | 7.064560e1 | 3.263092e-2 | 4.6190e-4 | 8.0000e-2 | 0.6% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dWq | 8.966064e-2 | 8.965194e-2 | 1.067147e-4 | 1.1903e-3 | 2.0000e-2 | 6.0% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dWk | 2.207947e-2 | 2.209130e-2 | 5.933736e-5 | 2.6860e-3 | 2.0000e-2 | 13.4% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dWv | 8.668750e1 | 8.676807e1 | 8.616638e-2 | 9.9307e-4 | 2.0000e-2 | 5.0% |
| `tier_b2_full_backward_sweep_cpu_naive_causal` | full_backward | CpuNaive | 128 | 32 | 32 | yes | dx | 3.992070e1 | 3.981777e1 | 1.030502e-1 | 2.5880e-3 | 3.0000e-2 | 8.6% |


## Coverage axes actually exercised

| axis | covered | not covered |
|---|---|---|
| head_dim | dK/dV: 32, 64, 128 — full backward: **64, 128 only** | 16, 256; hd=32 in the full backward |
| seq_len | 32, 64, 128 | anything not equal to a whole number of tiles |
| block_q | 32, 64 | **bq=64 at hd=128** — the dispatch ladder pins bq=64 for hd 32/64/128, but every hd=128 row here was measured at bq=32, so the tiling production would pick for hd=128 is NOT in this set |
| causal | yes and no | — |
| forward source | `B1Forward` (real Tier B.1 kernel) and `CpuNaive` | — |
| gradients | dQ, dK, dV, dWq, dWk, dWv, dx | — |
| GQA group size | **1 only** | 2, 4, 8 — now REFUSED, see below |
| heads | **1 only** | >1 for the Tier B.2 hybrid (already refused via `active_heads == 1`) |

## What is NOT certified

Recorded here rather than left implicit, because an uncertified regime that
still dispatches is worse than one that refuses.

1. **Grouped-query attention (`gqa_group_size > 1`).** The zero-copy stride
   pattern is implemented (`backward/hbm_addr.rs` divides the head register by
   the group size for K/V/dK/dV addressing), but every Tier B.2 numerical sweep
   fixes `gqa_group_size: 1`, so **the Tier B.2 grouped addressing** has never
   been compared against a reference. (A different path IS covered:
   `gqa_backward_matches_cpu_reference` exercises the runtime expand-KV
   envelope green across five GQA ratios, and is in the canary. It does not
   touch Tier B.2.)

   Item 15 added `DispatchReject::GqaUnvalidated` plus a loud
   `UnsupportedConfig` refusal in `synthesize_tier_b2_backward`. Note this does
   NOT make grouped-KV safe: no emitter under `phases/backward/` reads
   `gqa_group_size` either, so the scalar path is grouped-KV-blind as well.
   Both backward paths are unfit for GQA; the guards stop Tier B.2 from adding
   a second one and make the direct-call route say so. Lift point is on the
   `GqaUnvalidated` doc comment.

   In production this is defense-in-depth rather than a live fix: CSHA level is
   clamped to 1 (`compiler/kernel.rs:900-922`), so Tier B.2 never dispatches
   there at all.

2. **sm_80 / sm_90.** Not obtainable without A100/H100 hardware. The claim in
   `cuda_status.md` is scoped to sm_120 accordingly; do not read these rows as
   covering Ampere or Hopper.

3. **Partial tiles in the Tier B.2 full backward.** `dq.rs` documents a
   full-tiles-only precondition and the masking is unimplemented; the sweeps
   pin `seq == block_q` or a whole multiple. For Tier B.2 the guard is
   `seq_len == block_q` in `tier_b2_hybrid_backward_eligible`
   (`tier_b2/dispatch.rs`), so a ragged shape selects the scalar backward.
   (`ragged_seq_refuses_gpu_and_falls_back_correct` covers the *scalar* Phase-2
   FFI's own ragged refusal, not this one — it is a separate guard on a
   separate path.)

4. **Multi-tile CSHA fused backward.** Refused (Phase 1.1); the canary note on
   `t6_3_smoke_single_config` says so.
