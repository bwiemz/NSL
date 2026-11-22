# Dropout that keeps a key instead of its mask (2026-11-21, re-measured on main bc543f71)

Branch `perf/dropout-storage`. Design notes (outside the repo):
`nsl-audit/research/v3-reports/dropout-storage-design.md`; runs and scripts:
`nsl-audit/research/work/PERF-DROPOUT/`.

## What changed

Source-AD dropout used to keep, per call, an f32 mask the size of its input and its output
for the backward; CCR force-saved both because a random draw must not be replayed. The GPU
dropout draws from a counter hash (`nsl_kir::kernels::dropout`): element `i` is kept when
`H(base + i) < threshold`, so the mask is a pure function of the launch's counter base.

- The forward (`nsl_tensor_dropout_fwd_key`, kernel `nsl_dropout_key_f32`) computes the same
  output with the same instructions and claims the counter with the same call, and keeps a
  32-byte key (counter base, threshold, scale bits, length, magic) instead of the mask.
- The backward (`nsl_tensor_dropout_bwd`, `nsl_dropout_bwd_f32`) redraws each keep bit from
  the key and computes `(dy * m) * f32(1/(1-p))` with two `mul.rn.f32`, the stored-mask
  backward's operations in its order.
- CCR never replays the draw (the key stays saved). The output is freed at its block's end
  when the backward reads only its shape (a shape stub stands in; under `--layerwise-accum`
  the window buffers the stub, since #876). When its value is read it stays saved under
  `--checkpoint-selective` and is recomputed from its input and the key under full block
  recompute and `--layerwise-accum` (`nsl_tensor_dropout_apply`, `nsl_dropout_apply_f32`).
- `NSL_DROPOUT_REGEN=0` at build time restores the stored mask. The two modes are
  bit-identical, so the mode is not in the execution fingerprint and a checkpoint resumes
  in either. On the CPU the draw is the sampling ChaCha stream, not indexable by element,
  so there the key is the mask. Tape AD (`nsl run` without `--source-ad`) is unchanged.

The build log says what each dropout keeps, e.g. for the 500M recipe:
`[dropout] 48 dropout(s) keep a 32-byte key and redraw their mask in the backward; 24 of 48
output(s) freed after their block (read for shape only), 0 recomputed from the key`.

### Why 24 of 48 at 500M, and why that is all of dropout's cost

Each 500M block has two dropouts. The MLP's output reaches the backward only through the
residual add's `reduce_to_shape` (its shape), so it is freed. The attention's output is read
by the recompute clone of the block-interior residual sum `h = x + dropout(o)`, which the
selective policy recomputes, so it stays. Without dropout that clone reads `o` (the output
projection), which then stays instead; with dropout `o` is released. So on the production
line both masks and the MLP output go (30 of 40 MiB per 500M block) and the remaining 10
MiB is the residual-stream input the policy keeps with or without dropout.

Full block recompute and CSLA recompute the attention output from `o` and the key, which
costs recomputing `o` (one Wo GEMM per block in the backward). Before #876 (D1) CSLA freed no
interior per segment and could not stub the MLP output either, so it recomputed that too (the
W_down GEMM: 0.982x against main 04c0e01e); with #876 the MLP output takes a stub under CSLA
as on the other schedules.

## Numerics

| check | result |
|---|---|
| PTX interpreter (`dropout_regen_kir_equivalence.rs`): key kernel vs mask kernel, regenerated backward vs the compiler's fused chain over the stored mask, replay vs forward; specials, NaN payloads, counters across 2^32; bad keys | byte-identical; bad keys poison with NaN |
| Silicon (`hwcert_kir_numerics_gpu.rs`, RTX PRO 4500): same comparisons through the runtime's own ops, p 0.1 and 0.5 | bit-identical |
| Coder-50M (dropout 0.1), 6 updates x 8 micro-batches, `--deterministic`: main vs this branch vs `NSL_DROPOUT_REGEN=0`, plus a repeat (CSLA and the production flags on main bc543f71, with #876; all four also on 04c0e01e) | losses and saved model byte-identical for the production flags, full block recompute, `--layerwise-accum` and `precision="bf16"` (with `NSL_FLASH_BWD_CPU=0`: under `--deterministic` the attention backward otherwise runs on the host, which refuses bf16 operands, on main too) |
| CLI gates (`dropout_regen_gate.rs`) | CPU and GPU equivalence for every flag set; resume across the switch in both directions; a toy with 8 MiB dropout tensors saves 31 of 32 MiB at its peak; `--cuda-graphs` captured 12 regions and replayed 72 times against 4 and 24 with the stored mask |

Allocator peaks of those 50M runs (one dropout tensor is 2 MiB, 8 blocks):

| flags | main (bytes) | this branch (bytes) | saved | predicted |
|---|---|---|---|---|
| production (`--checkpoint-selective`), f32 storage | 1,488,989,696 | 1,438,670,336 | 48 MiB | 8 x (2 masks + MLP output) = 48 MiB |
| production, `precision="bf16"` | 1,543,679,488 | 1,493,360,128 | 48 MiB | 48 MiB |
| full block recompute | 1,338,259,968 | 1,271,163,392 | 64 MiB | 8 x 4 tensors = 64 MiB |
| `--layerwise-accum` (8-micro-batch window), main bc543f71 | 2,364,698,624 | 1,843,568,128 | 497 MiB | 8 x 64 MiB = 512 MiB |

## Memory and speed at 500M and 1B

`tools/perfcheck` on the RTX PRO 4500 (200 W), toolchains frozen from main and from this
branch (the 500M rows against main bc543f71 with #876 and #877, the 1B rows against main
04c0e01e: neither PR changes the 1B production line), both arms of a pair in the same round, 3 rounds, order alternating, 24 updates per
run (15 timed). Another session's GPU jobs started during some first attempts; each arm now
records every compute process on the device during its run, and every run that shared the
device was repeated (`PERF-DROPOUT/contam.py`). Allocator, reserved and driver peaks agreed to
0.01 GiB in every round.

| size, flags (precision) | allocator GiB | reserved GiB | driver GiB | tok/s main / branch | paired ratio, median [range] |
|---|---|---|---|---|---|
| 500M, production `P` (`precision="bf16"`, the recipe) | 12.88 → **12.18** (−0.70) | 13.84 → 13.16 | 14.22 → 13.54 | 22,700 / 22,869 | **1.011** [1.007–1.011] |
| 500M, full block recompute (TF32, no clip, no wgrad fusion) | 10.66 → **9.72** (−0.94) | 11.75 → 10.71 | 12.13 → 11.08 | 9,858 / 9,791 | 0.992 [0.992–0.993] |
| 500M, `--layerwise-accum` (TF32) | 18.19 → **10.71** (−7.48) | 19.85 → 11.71 | 20.22 → 12.08 | 9,607 / 9,755 | **1.008** [1.006–1.016] |
| 1B, production `P` (`precision="bf16"`, the recipe: dropout 0, elided) | 23.66 → 23.66 | 24.18 → 24.18 | 24.57 → 24.57 | 13,538 / 13,539 | 1.000 [0.997–1.001] |
| 1B, production `P`, model dropout 0.1 (`nsl_bf16s_drop01`) | 25.41 → **23.91** (−1.50) | 25.93 → 24.43 | 26.31 → 24.81 | 13,175 / 13,354 | **1.010** [0.996–1.014] |

`NSL_MEMSTATS=1` top contexts at the peak:

| arm | main | this branch |
|---|---|---|
| 500M `P` | `dropout_f32` 960.0 MB (96 blocks) | `dropout_key_f32` 240.0 MB (72 blocks: the 24 attention-dropout outputs the residual recompute reads, and 48 keys) |
| 500M full block recompute | `dropout_f32` 960.0 MB | not among the contexts at the peak |
| 500M `--layerwise-accum` | `dropout_f32` 7.50 GB (768 blocks) | not among the contexts at the peak |
| 1B `_drop01` | `dropout_f32` 2.00 GB | `dropout_key_f32` 512.0 MB (16 attention-dropout outputs and 32 keys) |

At 1B dropout 0.1 now costs 0.25 GiB over the dropout-free recipe instead of 1.75 GiB.

`--layerwise-accum` now peaks 0.99 GiB above full block recompute with the same flags
(10.71 against 9.72 GiB); before both changes it was 7.5 GiB above (D1's write-up).

Speed. On the production line and under `--layerwise-accum` the branch is faster (less
traffic: no mask is written or read; the regenerating backward is one kernel as the fused
chain was; CSLA moves 7.5 GiB less through its window). Full block recompute pays one Wo GEMM
per block in the backward to replay the attention dropout (0.992). Against main 04c0e01e,
before #876, CSLA also replayed the MLP output to measure its shape (no stubs there) and ran at
0.982x (21.56 → 14.53 GiB); with #876 that replay is gone, as the build log shows (24 outputs
read for shape only, 24 recomputed, against 0 and 48 before).

## Reproduce

```
# freeze each toolchain, build the arms (CPU), then one queued GPU job per round
PERF-DROPOUT/build_freeze.sh <worktree> regen
PERF-DROPOUT/build_arms.sh regen 500m nsl_bf16s nsl_tf32_lw nsl_tf32_blocks_nowgrad_noclip
PERF-DROPOUT/build_arms.sh regen 1b nsl_bf16s nsl_bf16s_drop01
PERF-DROPOUT/campaign.sh 500m PERF-DROPOUT/jobs_500m.txt 0 1 2
# the 50M exactness runs
PERF-DROPOUT/det_ab.py build|run|compare main:prod regen:prod regen:prod:off ...
```
