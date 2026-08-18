# The 50M AD differential — tape-F32 vs source-F32 (roadmap item 5)

**Date:** 2026-08-17 · **Hardware:** RTX PRO 4500 Blackwell 32 GB, CUDA 13.x ·
**Harness:** `models/benchmarks/ad_differential_50m.py` over
`models/coder50m/ad_differential.nsl` (128 micro-steps of the production
corpus, TF32 off, dropout off, no fused LCE, `shuffle=false`, per-seed
`--seed` init; per seed: source×2 + tape×2, both same-mode pairs are the
control).

## Headline results

1. **The differential caught a real gradient bug before certifying anything.**
   The first 3-seed run diverged systematically (tape landed 0.14–0.24
   lower at the endpoint, same direction every seed). `--debug-training`
   per-parameter gradient checksums showed the tape returning **exactly
   zero for every K/V projection weight**: the stdlib GQA K/V path is
   `expand(...).contiguous().reshape(...)` and a materializing
   `contiguous` recorded nothing on the tape, silently freezing K/V while
   the loss curves looked healthy. Fixed (same-shape Reshape relabel);
   the CI gate gained a GQA leg so the class stays pinned.

2. **After the fix, the gated windows agree on all 3 seeds** (table below
   from the post-fix run): step-0 forward gaps 0.03–0.06 (deterministic
   f32 lowering difference between the two forwards, well under the 0.15
   structural floor) and endpoint means within the control-derived
   allowance.

3. **The tape is run-to-run deterministic end-to-end** (tape-A == tape-B
   to every printed digit through step 128, every seed). Source pairs
   reproduce step 0 exactly and then pick up backward/optimizer
   nondeterminism (~1e-3–8e-3 endpoint spread), which amplifies
   chaotically mid-run — a same-seed source pair reaches O(1) mid-run
   gaps by ~step 100. This is why the early window is REPORTED, not
   gated: cross-mode early gaps are dominated by Lyapunov amplification
   of the legitimate step-0 offset, and same-mode controls cannot
   envelope it (each pair shares its forward exactly).

Post-fix measured windows (gating run banked in the PR; endpoint allowance
= max(3×control, 0.10)):

| seed | step0 gap (gated) | early gap (reported) | endpoint gap | verdict |
|------|-------------------|----------------------|--------------|---------|
| 1000 | 5.388e-02 | 2.735e-01 | 5.696e-02 | AGREE |
| 1001 | 6.071e-02 | 2.522e-01 | 7.608e-02 | AGREE |
| 1002 | 3.478e-02 | 4.252e-01 | 1.536e-02 | AGREE |

Cross-scale gradient validation backing the verdict:

- CPU-f64 tiny loader fixture (TinyLM): tape == source to **every printed
  digit** over 32 steps (CI-gated).
- CPU-f64 GQA fixture (masked, packed): tape == source to **3e-7
  relative** over 64 steps (CI-gated).
- GPU-f32 GQA fixtures (masked AND causal): tape-GPU == tape-CPU to
  ~1e-3 on every parameter's checksum.
- 50M: tape-CPU == tape-GPU checksums **to every printed digit**.

## Open question (handoff to the next campaign)

Under `--debug-training` at 50M, the **source** arm's wq/wk gradient
checksums are 16–21× the tape's (wv/wo agree; the affected params are
exactly the RoPE'd branches). The tape cross-validates at every scale
reachable (above), so the anomaly points at the de-optimized
`--debug-training` source arm at 50M-scale configs (seq 1024, head_dim
64) — a path combination nothing else exercises. It does not reproduce
at tiny scale in any device/mask combination tried. Next diagnostic:
scale a single-block probe (seq 256→1024, head_dim 8→64) under
`--debug-training --source-ad` until the discrepancy turns on, then
bisect the source lowering of the RoPE'd q/k chain. Note AdamW's
per-parameter scale invariance means a constant grad-scale error is
invisible in loss trajectories — only checksum-level comparison catches
this class.

## Reproduction

```bash
# full campaign (needs the local corpus + a CUDA build)
python models/benchmarks/ad_differential_50m.py \
    --nsl <cuda-nsl-binary> --seeds 3

# gradient checksums, either mode
cd models/coder50m
nsl run --seed 1000 --debug-training [--source-ad] ad_differential.nsl
```

Tape reference speed at 50M: ~0.27 steps/s (softmax backward bounces
[8,1024,1024] per block through the CPU) vs source-AD's ~16 steps/s —
fine for a reference, unusable for production.
