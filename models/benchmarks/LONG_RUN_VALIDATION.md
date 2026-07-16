# Long-run training validation (A3 — Milestone A, "trust the new baseline")

Sustained validation that CCR checkpointing, the FASE window accumulator, and
the async-owned memory paths do not **drift** over many optimizer steps —
something the 2-step parity gates cannot see. Two layers of evidence:

1. **Automated drift gate** (`crates/nsl-cli/tests/long_run_drift_gpu_gate.rs`)
   — the fast, permanent CI proxy.
2. **500M several-step run** (this doc) — the real-scale confirmation.

The 1.037B configuration was validated end-to-end in the #374 campaign
(9.9 GiB peak, 19/19 micro-batches, offload + CCR); see
`MEMORY_REDUCTION_RESULTS.md`. This doc adds a multi-step 500M CCR + allocator
stability run on top of that.

## 1. Automated drift gate (permanent)

`long_run_ccr_bit_exact_and_no_allocator_drift` trains a small packed-GQA LM
for **48 optimizer steps (96 micro-batches)** under `--deterministic` and
asserts, across the *whole* run:

| Property | Result |
|---|---|
| `--checkpoint-blocks` loss == baseline, **every step** (bit-exact) | ✅ 96/96 |
| Reserved VRAM stable after warmup (no per-step leak) | ✅ 2 MB flat |
| Per-step driver-alloc churn bounded (no accumulating alloc) | ✅ |

Wall-clock ~14 s on an RTX 5070 Ti. Because `--deterministic` makes the whole
device path bit-reproducible (A2's deterministic embedding kernel), the gate
uses the strongest possible claim: bit-exactness at every step, not a
tolerance band.

## 2. 500M multi-step run

Model: `models/coder500m/` (≈505M params, d_model 1280, 24 layers, GQA 20/10,
head_dim 64, vocab 49152), batch 2, grad_accum 8, seq 256, synthetic stream
sized for 64 optimizer steps. Production GPU path (atomicAdd embedding — the
deterministic kernel is O(vocab·embed·seq), impractical at 49k vocab).

**Finding: the baseline (no CCR, no offload) does not fit.** At 16 GiB the
505M baseline OOMs even at seq 256 — optimizer `m`+`v`+`m_partial` (~6 GiB) +
weights (~2 GiB) + activations + the 49k-vocab LM-head logits/backward exceed
the ~15 GiB free. This is exactly the memory wall CCR + offload exist to
remove, so a baseline-vs-checkpointed loss comparison is impossible at 500M —
the baseline cannot be run. The validation therefore confirms the
**memory-reduction stack trains stably over many steps**:

```
nsl run --source-ad --checkpoint-blocks --optim-state-offload \
  models/coder500m/pretrain_fase.nsl      # seq 256, NSL_MEMSTATS/NSL_DEBUG_MEM_ALL
```

### Results (`--checkpoint-blocks --optim-state-offload`, seq 256)

| Metric | Value |
|---|---|
| Fits on 16 GiB | ✅ driver 3.66–3.81 GB, alloc 2.11 GB |
| Reserved VRAM across the run | **2262 MB, constant every step — no leak** |
| First loss | 11.22 |
| Loss after ~39 steps | 2.5e-5 (converges on the degenerate single-token stream) |
| Baseline (no CCR/offload) | **OOM at 16 GiB — cannot run** |

The `all-ones` synthetic stream is a trivial task, so the loss collapses
quickly; the point of this run is the **memory profile over many steps**, not
learning quality (learning quality is covered by the bit-exact automated gate
and the #374 real-token 500M matrix). The constant 2262 MB reserve across every
step is the decisive result: neither CCR recompute, the FASE `m_partial` window,
nor the async optimizer-state offload leaks memory over a sustained run.

### Conclusion

Milestone A's baseline is trustworthy: CCR is bit-exact across a whole run on
the deterministic path (automated gate), the production path stays within the
nondeterminism band (A2's tolerance gate), and the memory-reduction stack holds
a flat allocator profile over dozens of steps at 505M scale. The 505M/1B
absolute peaks and the exact-vs-checkpointed VRAM deltas are tabulated in
`MEMORY_REDUCTION_RESULTS.md` (#374); this doc adds the *sustained-stability*
dimension those point-in-time measurements did not cover.
