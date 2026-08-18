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

| seed | step0 gap (gated, floor 2e-3) | early gap (reported) | endpoint gap | verdict |
|------|-------------------------------|----------------------|--------------|---------|
| 1000 | 4.768e-06 | 1.324e-03 | 5.670e-04 | AGREE |
| 1001 | 1.144e-05 | 6.695e-04 | 6.744e-03 | AGREE |
| 1002 | 1.907e-06 | 3.551e-03 | 1.025e-02 | AGREE |

(Pre-causal-fix, for contrast: step-0 3.5e-02..6.1e-02, endpoint 1.5e-02..7.6e-02
with the tape systematically LOWER. Those runs "passed" only because the
floors were calibrated against them — see the third finding below.)

Cross-scale gradient validation backing the verdict:

- CPU-f64 tiny loader fixture (TinyLM): tape == source to **every printed
  digit** over 32 steps (CI-gated).
- CPU-f64 GQA fixture (masked, packed): tape == source to **3e-7
  relative** over 64 steps (CI-gated).
- GPU-f32 GQA fixtures (masked AND causal): tape-GPU == tape-CPU to
  ~1e-3 on every parameter's checksum.
- 50M: tape-CPU == tape-GPU checksums **to every printed digit**.
- 50M post-causal-fix: per-parameter `sum|grad|` agrees between modes to
  2-4% on GPU f32 (was 16-21x on the RoPE'd wq/wk); tiny-scale CPU f64
  probes agree to **1.00 on every parameter**.

## The third finding: the differential's own residual was a real bug

The banked run above is the SECOND correction. The first 3-seed run
diverged systematically (dead K/V gradients, fixed in PR #511). What
remained after that fix still had the tape landing systematically LOWER
at the endpoint (0.015-0.076, same sign every seed) and a step-0 forward
gap of ~5e-2 that no f32 rounding argument supports at f64 CPU scale.
That residual was **`scaled_dot_product_attention` silently ignoring the
causal flag in the eager/tape lowering**:

- source AD read the causal flag from the POSITIONAL slot and defaulted
  to causal; the eager/tape path honoured ONLY the named `causal=`
  spelling and defaulted to NON-causal;
- the stdlib GQA passes it positionally
  (`scaled_dot_product_attention(q, k, v, scale, true)`), so **every
  tape-mode causal LM trained with bidirectional attention** — each
  position could read its own future tokens;
- label leakage lowers the loss once training starts, which is exactly
  the systematic sign of the residual. It reads as "the tape is doing
  slightly better", not as a bug.

Arbitration was a finite-difference probe: perturb one block's `wq` by
±eps and compare `(L(w+eps) - L(w-eps)) / 2eps` against the signed
gradient sum each mode reports (the `sum(grad)` column added to
`--debug-training` for this purpose). Tape's value matched FD in
magnitude; source's was ~100x larger with the opposite sign — which
pointed at a FORWARD mismatch rather than a backward bug, and the
four-cell (mode x causal-spelling) matrix then localized it exactly:
tape produced byte-identical losses for `causal=true` and `causal=false`.

Two lessons worth carrying:

1. **A systematic sign is a bug signature; a magnitude is not.** Both
   times this campaign found a real defect, the tell was that every seed
   diverged the SAME direction. A noise envelope only bounds magnitude.
2. **Floors calibrated against a broken baseline encode the breakage.**
   The original step-0 floor (0.15) and endpoint floor (0.10) were set
   from measurements taken while one arm ran non-causal attention; both
   sat just above the leakage signal, so the gate certified a run where
   the two arms computed different functions. Post-fix the same windows
   measure 2e-06..1e-05 and 6e-04..1e-02, and the floors are now 2e-3 /
   0.03 — set from the source control spread, not from the observed
   cross-mode gap.

Regression pin: `crates/nsl-cli/tests/sdpa_causal_contract_gate.rs` runs
all five spellings (positional true/false, named true/false, absent)
under both AD modes and requires exact cross-mode agreement AND that
causal differs from non-causal, so it cannot pass vacuously by both
modes ignoring the flag. Mutation-tested against the pre-fix lowering.

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
