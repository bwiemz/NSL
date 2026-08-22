# M46 `--deterministic` — the flash-attention backward was never routed

**Status:** fixed (`b1dae1c2`, `0aa73856`). This record is the evidence.
**Hardware:** RTX PRO 4500 Blackwell (32 GB, sm_120), CUDA 13.x.

`--deterministic` advertises run-to-run device bit-reproducibility. For any
model with attention it did not deliver it, and the gate that certified the
flag could not have noticed.

## EC1 — the gate was vacuous twice over

`srbf16_e2e_deterministic_and_sane_gpu` was the cited proof. Two independent
reasons it passed with the bug fully present, either sufficient on its own:

1. **Its fixture never ran the op.** `csla_layerwise_ffn.nsl` states on its own
   line 4 that it is "deliberately attention-free". A determinism gate
   certifies exactly the ops its fixture executes.
2. **Grid size, not parameter count, decides whether the race expresses.** A
   first replacement fixture at 2 heads / seq 256 *did* dispatch the atomic
   kernel — confirmed by the `[flash-bwd] GPU backward dispatched` line — and
   was still bit-identical across 3 replicates.

So "add attention to the fixture" would have produced a second vacuous gate.

## EC2 — the mechanism is dQ, not dK/dV

Phase-2's grid is `(B*nh, ceil(S/block_kv), 1)`. Each CTA owns one
(batch-head, kv-tile) and loops the Q tiles, accumulating **dK/dV in its own
SMEM** — no cross-block reduction there. **dQ** is the cross-block reduction:
every kv-tile CTA adds into the same dQ rows via `atom.global.add.f32`, so the
summation order follows scheduling.

The quantity that keeps the race alive is therefore the number of contending
CTAs on dQ = `batch x heads x ceil(seq_len / block_kv)`. An earlier draft of
this analysis said dK/dV and sized the gate fixture from that; the geometry it
picked happened to work, for the wrong reason. Anyone resizing that fixture
must preserve the dQ contention count.

Divergence with the fix removed, 3 replicates per row:

| geometry | max spread |
| --- | --- |
| heads 2, seq 256 | 9.5e-07 (once exactly 0.0) |
| heads 4, seq 512 | 1.9e-06 |
| **heads 8, seq 512** | **2.1e-05**  <- gate fixture |
| heads 8, seq 1024 | 1.8e-05 |

## EC3 — the forward was never the problem

`models/coder50m/det_forward_check.nsl`, 3 replicates x 8 passes: spread
**0.000e+00**. Divergence begins at the step after the first optimizer update.
With `grad_accumulation=2` the first update lands at the end of step 1, so
step 2 is the first forward that can read a perturbed weight — that ordering
is the localization, and it points at the backward rather than at sampling.

## EC4 — the fix, measured at the geometry the cert lane actually runs

One release binary, one runtime archive, `models/coder50m/pretrain_cert.nsl`
built once and run repeatedly. The two arms differ ONLY by the documented kill
switch, so nothing about the compiler or the archive varies between them:

| arm | routing | per run | loss streams |
| --- | --- | --- | --- |
| `NSL_FLASH_BWD_CPU=0` | GPU atomic (pre-fix behaviour) | 5.11 s (n=2) | **6 of 6 distinct** |
| default under `--deterministic` | CPU reference | 91.79 s (n=6, spread 687 ms) | **6 of 6 identical** |

Measured 2026-08-21. **Cost: 17.95x** end-to-end on this fixture. That is a
whole-program figure and is NOT the same number as the ~36x *per-step* slope
measured on `det_train_check.nsl`: the program figure is diluted by compile,
model init and the validation pass. Quote whichever matches the question, and
say which one it is.

At this geometry the pre-fix path is *reliably* nondeterministic — 6/6
distinct, not a marginal flake — which is what makes the gate's negative arm a
real assertion rather than a coin toss.

## EC5 — `scripts/arena-parity.sh` needed the fix; it was not slowed by it

Worth stating because the opposite was assumed. That script builds a coder50m
training program twice and demands byte-identical loss streams **under
`--deterministic`** (line 57). With routing opted out it fails in **12
seconds** at step 1, `loss streams differ`. Pre-fix it was not passing quickly
— it was failing quickly. With the fix it reaches the CUDA-graph composition
stage ~477 s in, where it fails for an unrelated, pre-existing reason
(`regions=2 captured=0 taints=2`, "graphs arm captured nothing").

The one real lane concern is a profile artifact, not the fix: the Rust wrapper
`arena_parity_script_passes` invokes `env!("CARGO_BIN_EXE_nsl")`, and
`scripts/gpu-cert.sh` runs `cargo test` without `--release`, so the lane
executes a **debug** `nsl` through a CPU-reference backward.

## EC6 — the kill switch is tri-state on purpose

`NSL_FLASH_BWD_CPU` unset = follow `--deterministic`; `=1` = force the CPU
reference even without the flag; `=0` = opt out. The opt-out prints a one-shot
warning naming the mechanism, because silently keeping the flag's name while
dropping its guarantee is the failure this whole record is about.

## EC7 — what this does NOT establish

- It does not make the deterministic path fast. The proper fix is a
  deterministic-reduction phase-2 dQ kernel; until someone writes one, the
  flag costs what EC4 says it costs.
- The 17.95x and ~36x figures are Coder-50M numbers. They are not measured at
  500M or 1B.
- Bit-reproducibility is certified **run-to-run on one machine**, not across
  GPUs, driver versions, or block-size selections.
