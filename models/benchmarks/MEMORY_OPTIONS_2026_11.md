# What NSL's memory options are worth (2026-11-15)

Every memory option NSL ships, applied one at a time on top of the production flags
(`P` = `--source-ad --checkpoint-blocks --checkpoint-selective --fuse-rmsnorm-backward
--fuse-wgrad-accum`), measured with `tools/perfcheck` at Coder-1B (2 x 2048, accumulation 4)
and Coder-500M (2 x 1024, accumulation 8) in both matmul postures, checked for numerics at
Coder-50M, and given a verdict. The measurements were taken with no change to the compiler
or runtime; the changes that followed from them, including a new option measured the same way
(`first_moment_dtype="bf16"`), are in [Changes made after the measurements](#changes-made-after-the-measurements).

Toolchain: one frozen build (compiler, CUDA runtime archive, stdlib) of `perf/memory-audit`
at 7687d576 (`perf/round-1`: `main` plus cublasLt GEMM selection and the tensor-core
attention), registered with `perfcheck --toolchain mem=DIR`. Card: RTX PRO 4500 Blackwell,
32 GB, 200 W cap, PCIe 5 x16; 60 GB host RAM. Raw data (one JSON per run, logs, the driver
scripts named below) is in `/mnt/extrastorage/nsl-audit/research/work/PERF-MEMORY/` on the
measuring machine.

## Summary

| option (on top of `P`) | 1B bf16 (recipe) alloc / driver change | 1B bf16 speed, paired | 500M TF32 (recipe) alloc change, speed | numerics | verdict |
|---|---|---|---|---|---|
| `--checkpoint-blocks` without `-selective` | **-5.38 / -5.32 GiB** (21.14 / 22.48) | 0.897 | -2.06 GiB, 0.884 | bit-exact | **keep** (the cheapest large saving) |
| `--checkpoint-budget-mib 3072` (with blocks only) | -2.75 / -2.74 | 0.947 | +1.02, 1.019 (3 GiB > 500M's selective set) | same CCR machinery, not separately checked | **keep** (the knob between the two) |
| `--checkpoint-compress bf16` | -3.40 / -3.37 | 0.924 | -1.30, 0.946 | held-out loss within seed noise | **keep** (small fix, D6) |
| `--no-bf16-cast-cache` (bf16 only) | -1.80 / -1.75 | 0.966 | (bf16 500M: -0.98, 0.953) | bit-exact | **keep** |
| no checkpointing | OOM at 1B | - | **+12.17 GiB (24.89), 1.115** (bf16: 1.161) | bit-exact | **keep** at 500M only |
| `--checkpoint-stride 2` | **+0.63** / +0.55 | 1.014 | -0.12, 1.019 | bit-exact | **retire** for selective recipes |
| `--optim-state-offload` (needs dropping `--fuse-wgrad-accum`) | **-11.84 / -11.90** (14.68 / 15.90) | 0.423 (0.596 vs the same drop without offload) | -5.97 (6.75 alloc, 8.00 driver), 0.477 | bit-exact | **keep** as the small-card escape hatch; **fix** its composition (m_partial) |
| `--cuda-graphs` | -1.80 (it switches the cast cache off) | 0.895 | 0, 0.842 | bit-exact | **retire** (or fix, see below) |
| `--wggo-moment-precision` | none: byte-identical binary | - | none | - | **fix** (inert; now a compile error where inert) |
| `--wggo-memory-budget` (the only way to get bf16 m) | refused at compile time on every recipe | - | refused | - | **fix** |
| `--wggo greedy` (what the two above need) | 0 | **0.824** | 0, 0.885 | - | **fix** (not memory: costs 12-18%) |
| `--layerwise-accum` (needs no clip, no `--fuse-wgrad-accum`) | OOM at 1B | - | **+9.97 GiB** vs its control (22.68), 0.965 vs control | bit-exact | **fix** (D1) |
| `--weight-stream` (default residency) | OOM at 1B (streams nothing) | - | +9.97 (identical to `--layerwise-accum`) | bit-exact | **fix** (D1, D2) |
| `--weight-stream` all layers (`NSL_WS_RESIDENT=0`) | +0.30 / +2.29 (fits where the above OOM) | 0.544 (0.831 vs control) | +8.17, 0.596 | bit-exact | **fix** (rides D1) |
| `--stream-arena` / `+ --stream-prefetch` / `+ --stream-async-writeback` | +0.36 to +1.05 | 0.45 (0.83 vs plain streaming) | +8.2 to +8.4, 0.50 | bit-exact | **retire** unless D5 is fixed |
| `--param-dtype bf16-sr` (needs `--weight-stream`) | +2.10 / +2.26 (weights 4.02 -> 2.43 GiB) | 0.611 (0.933 vs control) | +9.15, 0.687 | held-out loss = f32 params (mean diff +0.000, 3 seeds) | **fix** (decouple from streaming) |
| `--zero-stage 1` (one GPU, needs no clip) | 0 | 1.006 | 0, 1.003 | bit-exact | **retire** for one GPU |
| `NSL_ASYNC_ALLOC=1` | driver **+1.18** | 0.984 | driver +0.35, 0.974 | bit-exact | **retire** |
| `first_moment_dtype="bf16"` (AdamW argument, added after the audit) | **-2.00 / -2.00** (24.52 / 25.79) | 1.004 | -0.96, 1.001 | held-out loss: pre-registered test passed (5 seeds) | **keep**; candidate default (see the end) |

No option measured in the audit qualifies for **default**: every saving costs speed, and the
large ones cost 5-60%. The one added after it, `first_moment_dtype="bf16"`, costs no speed; it is
a default candidate once a longer convergence run than the 50M one supports it. The two most
useful facts for the recipes:

* Dropping `--checkpoint-selective` takes the 1B recipe from 27.79 to **22.48 GiB driver**,
  the same as PyTorch bf16 + `torch.compile` (22.25), for 10% speed, bit-exactly.
* Offload takes it to **15.90 GiB driver** (14.68 allocator) for 58% speed, bit-exactly. Most of
  that cost is not the optimizer state: offload forces dropping `--fuse-wgrad-accum`, which is
  worth 1.41x at 1B bf16 by itself, and it stages the gradient accumulator over PCIe on every
  micro-batch.

The whole layerwise family (`--layerwise-accum`, `--weight-stream` and its sub-options,
`--param-dtype bf16-sr`) is broken as a memory option on these recipes: those paths skip the
forward's per-segment early-free, so every block's interiors stay live to the end of each
micro-batch's forward. They were meant to remove the 4 GiB gradient accumulator and stream
the 3.6 GiB of layer weights; instead they add 8-12 GiB.

## Method

* Arms: `tools/perfcheck/perfcheck.py` `MEMORY_ARMS`. Each
  arm is `P` plus one option, or, where the compiler refuses a production flag next to the option,
  the nearest flag set it accepts plus a control arm with the same drop and no option:
  `nowgrad` (`P` without `--fuse-wgrad-accum`: offload's control), `nowgrad_noclip` and
  `blocks_nowgrad_noclip` (the layerwise family's controls: `--layerwise-accum` refuses
  `grad_clip=` and `--fuse-wgrad-accum`, and the family runs without `--checkpoint-selective`
  because with it `--layerwise-accum` OOMs at both sizes). `grad_clip` became part of the build key.
* Timing: perfcheck's step-limited program (24 updates, the first 9 untimed). Each queued GPU
  job is one line of a job file: the production arm first, then up to six
  options; perfcheck round = 10 x round + line, so every option is paired with the production
  arm inside the same job. Order reversed on odd rounds and rotated by perfcheck. Rounds: 1B
  bf16 0-2 (plus a 4th for the first job's arms), 1B TF32 5-7, 500M TF32 0-2, 500M bf16 5-7.
  Arms that OOMed, or that cannot differ by construction, were dropped after one round (the
  "1/1" rows). Ratios are tok/s within a job, median [min-max] over rounds; memory peaks were
  identical in every round (single values).
* Memory: allocator peak (`gpu_peak_bytes`), reserved peak (`NSL_MEMSTATS`), driver peak
  (nvidia-smi per process, 100 ms), and the allocator's surfaces and contexts at the global
  peak (`NSL_MEMSTATS` "Top contexts at peak").
* Numerics, options that claim exactness: `tools/refcheck/optimizer_check.py` at 50M (8 AdamW
  updates, accumulation 4, weight decay 0.1, warmup_cosine, clip 1.0 where the arm keeps it),
  run twice per option: production mode (TF32, `--no-bf16-lt-tune`) for agreement with PyTorch,
  and `--deterministic` for a bit comparison of `after.nslm` and every micro-batch loss against
  the control. Production mode is not
  run-to-run reproducible (D7), so the bit check needs `--deterministic`, where two runs of `P`
  are identical in all 49,816,392 parameters.
* Numerics, options that change results on purpose (`--checkpoint-compress`, `--param-dtype
  bf16-sr`): 2,000 AdamW updates at 50M (batch 1 x 1024, accumulation 4, lr 3e-4, warmup 100,
  cosine to 3e-5; the refcheck corpus, 8.2 M tokens, one pass, dropout 0), then the mean
  held-out loss over 256 x 1024 tokens of `val.bin`; same seed and data as the control, seeds 1-3.

## Where the 1B production peak goes

`P`, bf16 (the recipe), at the allocator's global peak (26.53 GiB; reserved 27.41, driver 27.79):

| item | GiB | PyTorch bf16 + compile (20.58 alloc / 21.91 reserved / 22.25 driver) |
|---|---|---|
| parameters, f32 (1,073,809,408 = 4.000 GiB, plus RoPE tables) | 4.02 | 4.0 |
| Adam m, f32 | 4.00 | 4.0 |
| Adam v, f32 | 4.00 | 4.0 |
| gradient accumulator (`m_partial`), f32 | 4.00 | 4.0 (`.grad`) |
| bf16 weight images (cast cache, 112 matrices; 1.812 from the shapes) | 1.81 | - |
| matmul outputs live at the peak (saved selective set, f32) | 5.49 | - |
| residual-stream adds (`nsl_add_f32`: boundaries + mid-block residuals) | 1.65 | - |
| backward transients and the rest (fused chains 0.47, div 0.42, mul 0.34, silu 0.13, rmsnorm 0.09, other 0.10) | 1.55 | - |
| all activations and transients | 10.51 | 4.58 |
| allocator cache (reserved - allocated) | 0.88 | 1.33 |
| context, cuBLAS/cublasLt workspaces, modules (driver - reserved) | 0.38 | 0.34 |

The 16 GiB of persistent state is the same in both frameworks. The whole 5.95 GiB allocator gap is
activations: 1.81 GiB of bf16 weight images that PyTorch does not keep, and about 4.1 GiB of
activations NSL keeps in f32 where autocast keeps bf16. At 500M two more items appear. The
allocator rounds every large block up to a multiple of 2 MiB (`caching_allocator.rs`
`LARGE_ALIGNMENT`). That costs 0.18 GiB per f32 copy of the 500M parameters (0.72 GiB over
parameters, m, v and m_partial) and 0.16 GiB of the 0.98 GiB cast cache (0.82 from the shapes).
1B's power-of-two shapes round exactly, so it pays nothing. The `NSL_ASYNC_ALLOC` arm, which
does not round, shows 1.88 GiB per copy instead of 2.06.

## Results

### Coder-1B, bf16 (the recipe's posture): `nsl_bf16` 9,626 [9,306-9,761] tok/s, 26.53 / 27.41 / 27.79 GiB

| arm | ok/runs | tok/s median [min-max] | / `P`, paired | / control, paired | alloc GiB (vs `P`) | reserved | driver (vs `P`) | at peak: weights / m / v / m_partial / activations |
|---|---|---|---|---|---|---|---|---|
| `P` (`nsl_bf16`) | 14/14 | 9,626 [9,306-9,761] | - | - | 26.53 | 27.41 | 27.79 | 4.02 / 4.00 / 4.00 / 4.00 / 10.51 |
| `blocks` | 4/4 | 8,671 [8,647-8,698] | 0.897 [0.887-0.904] | - | 21.14 (-5.38) | 22.09 | 22.48 (-5.32) | 4.02 / 4.00 / 4.00 / 4.00 / 5.13 |
| `budget3g` | 3/3 | 9,160 [9,114-9,170] | 0.947 [0.947-0.951] | - | 23.77 (-2.75) | 24.67 | 25.06 (-2.74) | 4.02 / 4.00 / 4.00 / 4.00 / 7.76 |
| `compress` | 4/4 | 8,929 [8,896-8,999] | 0.924 [0.917-0.930] | - | 23.13 (-3.40) | 24.04 | 24.43 (-3.37) | 4.02 / 4.00 / 4.00 / 4.00 / 7.11 |
| `nocastcache` | 4/4 | 9,332 [9,302-9,358] | 0.966 [0.955-0.971] | - | 24.73 (-1.80) | 25.66 | 26.04 (-1.75) | 4.02 / 4.00 / 4.00 / 4.00 / 8.71 |
| `graphs` | 4/4 | 8,652 [8,607-8,713] | 0.895 [0.886-0.904] | - | 24.73 (-1.80) | 25.66 | 26.05 (-1.75) | 4.02 / 4.00 / 4.00 / 4.00 / 8.71 |
| `stride2` | 3/3 | 9,770 [9,347-9,785] | 1.014 [0.975-1.017] | - | 27.16 (+0.63) | 27.96 | 28.34 (+0.55) | 4.02 / 4.00 / 4.00 / 4.00 / 11.14 |
| `nockpt` | 0/1 OOM | - | - | - | - | 29.82 | 30.20 | - |
| `nowgrad` (control) | 3/3 | 6,822 [6,748-6,848] | 0.708 [0.704-0.712] | - | 26.63 (+0.11) | 27.60 | 27.98 (+0.19) | 4.02 / 4.00 / 4.00 / 4.00 / 10.62 |
| `offload` | 3/3 | 4,075 [3,934-4,079] | 0.423 [0.410-0.424] | 0.596 [0.583-0.597] (`nowgrad`) | 14.68 (-11.84) | 15.51 | 15.90 (-11.90) | 4.02 / 0 / 0 / 0 / 10.67 |
| `wggo` | 3/3 | 7,936 [7,802-7,978] | 0.824 [0.814-0.829] | - | 26.49 (-0.04) | 27.41 | 27.79 (0) | 4.02 / 4.00 / 4.00 / 4.00 / 10.47 |
| `zero1` | 1/1 | 9,685 | 1.006 | - | 26.53 (0) | 27.41 | 27.79 (0) | same as `P` |
| `asyncalloc` | 1/1 | 9,465 | 0.984 | - | 26.52 | (not tracked) | 28.98 (+1.18) | same as `P` |
| `nowgrad_noclip` (control) | 1/1 | 6,787 | 0.705 | - | 26.63 (+0.11) | 27.60 | 27.98 | 4.02 / 4.00 / 4.00 / 4.00 / 10.62 |
| `lwsel` (layerwise, selective) | 0/1 OOM | - | - | - | - | 29.85 | 30.23 | - |
| `blocks_nowgrad_noclip` (control) | 3/3 | 6,293 [6,243-6,294] | 0.654 [0.654-0.671] | - | 21.25 (-5.27) | 22.16 | 22.54 (-5.26) | 4.02 / 4.00 / 4.00 / 4.00 / 5.23 |
| `lw` | 0/1 OOM | - | - | - | - | 29.85 | 30.23 | - |
| `ws` (default residency) | 0/1 OOM | - | - | - | - | 29.85 | 30.23 | - |
| `wsall` | 3/3 | 5,227 [5,227-5,230] | 0.544 [0.543-0.544] | 0.831 (control) | 26.82 (+0.30) | 29.70 | 30.08 (+2.29) | 0.62 / 4.00 / 4.00 / 0 / 18.20 |
| `wsarena` | 3/3 | 4,331 [4,246-4,343] | 0.451 [0.441-0.451] | 0.829 [0.812-0.830] (`wsall`) | 26.89 (+0.36) | 29.77 | 30.15 (+2.35) | 0.62 / 4.00 / 4.00 / 0 / 18.27 |
| `wsprefetch` | 3/3 | 4,300 [4,295-4,379] | 0.446 [0.445-0.456] | - | 27.11 (+0.59) | 29.62 | 29.75 (+1.96) | 0.85 / 4.00 / 4.00 / 0 / 18.27 |
| `wsasync` | 3/3 | 4,375 [4,332-4,447] | 0.454 [0.451-0.461] | - | 27.57 (+1.05) | 29.70 | 30.08 (+2.29) | 1.30 / 4.00 / 4.00 / 0 / 18.27 |
| `srbf16` | 3/3 | 5,868 [5,859-5,872] | 0.611 [0.610-0.630] | 0.933 [0.932-0.938] (control) | 28.63 (+2.10) | 29.64 | 30.05 (+2.26) | 2.43 / 4.00 / 4.00 / 0 / 18.20 |

### Coder-1B, TF32: `nsl_tf32` 7,237 [6,913-7,251] tok/s, 24.63 / 25.48 / 25.87 GiB

| arm | ok/runs | tok/s | / `P`, paired | / control | alloc (vs `P`) | reserved | driver (vs `P`) | weights / m / v / m_partial / act |
|---|---|---|---|---|---|---|---|---|
| `blocks` | 3/3 | 6,327 [6,308-6,327] | 0.873 [0.873-0.912] | - | 19.25 (-5.38) | 20.12 | 20.50 (-5.37) | 4.02 / 4 / 4 / 4 / 3.23 |
| `budget3g` | 3/3 | 6,761 [6,700-6,779] | 0.935 [0.933-0.969] | - | 21.88 (-2.75) | 22.80 | 23.18 (-2.68) | 4.02 / 4 / 4 / 4 / 5.86 |
| `compress` | 3/3 | 6,824 [6,797-6,826] | 0.942 [0.941-0.983] | - | 21.24 (-3.40) | 22.27 | 22.65 (-3.21) | 4.02 / 4 / 4 / 4 / 5.22 |
| `graphs` | 3/3 | 6,169 [6,147-6,171] | 0.852 [0.851-0.889] | - | 24.63 (0) | 25.48 | 25.87 (0) | same as `P` |
| `nowgrad` | 3/3 | 5,774 [5,768-5,774] | 0.797 [0.796-0.834] | - | 24.82 (+0.19) | 25.61 | 25.99 | |
| `offload` | 3/3 | 3,705 [3,605-3,710] | 0.512 [0.512-0.521] | 0.642 [0.625-0.642] | 12.87 (-11.76) | 13.98 | 14.37 (-11.50) | 4.02 / 0 / 0 / 0 / 8.85 |
| `blocks_nowgrad_noclip` | 3/3 | 5,191 [5,156-5,192] | 0.717 [0.715-0.717] | - | 19.44 (-5.20) | 20.30 | 20.69 | |
| `lw` | 0/3 OOM | - | - | - | - | 29.73 | 30.11 | |
| `wsall` | 3/3 | 4,491 [4,491-4,505] | 0.623 [0.621-0.623] | 0.868 [0.865-0.871] | 26.76 (+2.12) | 29.62 | 30.00 (+4.13) | 0.62 / 4 / 4 / 0 / 18.14 |
| `srbf16` | 3/3 | 4,964 [4,941-4,964] | 0.686 [0.685-0.686] | 0.956 [0.956-0.958] | 28.57 (+3.93) | 29.82 | 30.20 (+4.34) | 2.43 / 4 / 4 / 0 / 18.14 |

### Coder-500M, TF32 (the recipe's posture): `nsl_tf32` 12,764 [12,697-12,943] tok/s, 12.72 / 14.05 / 14.44 GiB

| arm | ok/runs | tok/s | / `P`, paired | / control | alloc (vs `P`) | reserved | driver (vs `P`) | weights / m / v / m_partial / act |
|---|---|---|---|---|---|---|---|---|
| `blocks` | 3/3 | 11,288 [11,266-11,430] | 0.884 [0.882-0.896] | - | 10.66 (-2.06) | 12.01 | 12.38 (-2.05) | 2.07 / 2.06 / 2.06 / 2.06 / 2.40 |
| `budget3g` | 3/3 | 13,004 [12,997-13,022] | 1.019 [1.018-1.021] | - | 13.74 (+1.02) | 14.87 | 15.25 (+0.81) | ... / 5.48 |
| `compress` | 3/3 | 12,069 [12,062-12,079] | 0.946 [0.945-0.946] | - | 11.42 (-1.30) | 12.65 | 13.03 (-1.40) | ... / 3.16 |
| `graphs` | 3/3 | 10,751 [10,750-10,799] | 0.842 [0.842-0.847] | - | 12.72 (0) | 14.05 | 14.46 | ... / 4.46 |
| `nockpt` | 3/3 | 14,238 [14,238-14,248] | 1.115 [1.115-1.117] | - | 24.89 (+12.17) | 26.15 | 26.53 (+12.10) | ... / 15.93 |
| `stride2` | 3/3 | 12,983 [12,900-12,999] | 1.019 [1.016-1.019] | - | 12.60 (-0.12) | 14.02 | 14.40 (-0.03) | ... / 4.34 |
| `asyncalloc` | 3/3 | 12,404 [12,364-12,421] | 0.974 [0.973-0.974] | - | 11.93 (no 2 MiB rounding in its accounting) | - | 14.79 (+0.35) | 1.89 / 1.88 / 1.88 / 1.88 / 4.38 |
| `nowgrad` | 3/3 | 9,908 [9,623-10,022] | 0.778 [0.758-0.785] | - | 12.72 | 14.06 | 14.44 | |
| `offload` | 3/3 | 6,069 [6,055-6,086] | 0.477 [0.476-0.477] | 0.613 [0.607-0.629] | 6.75 (-5.97) | 7.62 | 8.00 (-6.44) | 2.07 / 0 / 0 / 0 / 4.67 |
| `wggo` | 3/3 | 11,302 [11,299-11,332] | 0.885 [0.883-0.885] | - | 12.70 | 14.05 | 14.44 | |
| `zero1` | 3/3 | 12,802 [12,800-12,967] | 1.003 [1.001-1.012] | - | 12.72 | 14.05 | 14.44 | |
| `nowgrad_noclip` | 3/3 | 9,954 [9,927-9,969] | 0.778 [0.776-0.780] | - | 12.72 | 14.06 | 14.44 | |
| `lwsel` | 0/3 OOM | - | - | - | - | 29.85 | - | |
| `blocks_nowgrad_noclip` | 3/3 | 9,159 [9,148-9,161] | 0.718 [0.717-0.718] | - | 10.66 (-2.06) | 11.76 | 12.14 | 2.07 / 2.06 / 2.06 / 2.06 / 2.40 |
| `lw` | 3/3 | 8,839 [8,839-8,840] | 0.693 [0.692-0.694] | 0.965 [0.965-0.966] | 22.68 (+9.97) | 26.09 | 26.47 (+12.04) | 2.07 / 2.06 / 2.06 / 0 / 16.48 |
| `ws` (default residency) | 3/3 | 8,859 [8,832-8,951] | 0.696 [0.693-0.701] | 0.968 [0.964-0.977] | 22.68 (+9.97) | 26.09 | 26.47 | identical to `lw` |
| `wsall` | 3/3 | 7,624 [7,620-7,630] | 0.596 [0.589-0.597] | 0.832 (control) | 20.89 (+8.17) | 24.16 | 24.54 (+10.10) | 0.32 / 2.06 / 2.06 / 0 / 16.44 |
| `wsarena` | 3/3 | 6,341 [6,326-6,351] | 0.496 [0.490-0.496] | 0.832 [0.830-0.832] (`wsall`) | 20.93 | 24.19 | 24.57 | 0.32 / ... / 16.48 |
| `wsprefetch` | 3/3 | 6,389 [6,388-6,415] | 0.499 [0.496-0.501] | 0.838 [0.837-0.841] (`wsall`) | 21.00 | 24.26 | 24.64 | 0.39 / ... |
| `wsasync` | 3/3 | 6,450 [6,448-6,507] | 0.504 [0.503-0.505] | 0.846 [0.845-0.854] (`wsall`) | 21.14 | 24.40 | 24.78 | 0.53 / ... |
| `srbf16` | 3/3 | 8,751 [8,739-8,769] | 0.687 [0.685-0.687] | 0.957 [0.954-0.957] | 21.87 (+9.15) | 25.15 | 25.53 (+11.10) | 1.31 / 2.06 / 2.06 / 0 / 16.44 |

### Coder-500M, bf16: `nsl_bf16` 15,574 [15,571-15,577] tok/s, 13.70 / 14.79 / 15.17 GiB

| arm | ok/runs | tok/s | / `P`, paired | / control | alloc (vs `P`) | reserved | driver (vs `P`) |
|---|---|---|---|---|---|---|---|
| `blocks` | 3/3 | 14,382 [14,380-14,387] | 0.923 [0.923-0.924] | - | 11.64 (-2.06) | 12.80 | 13.18 (-1.99) |
| `budget3g` | 3/3 | 15,979 [15,973-15,980] | 1.026 | - | 14.73 (+1.02) | 15.67 | 16.05 |
| `compress` | 3/3 | 14,551 [14,547-14,553] | 0.934 | - | 12.42 (-1.28) | 13.54 | 13.93 |
| `nocastcache` | 3/3 | 14,844 [14,844-14,848] | 0.953 | - | 12.72 (-0.98) | 14.07 | 14.46 |
| `graphs` | 3/3 | 13,996 [13,984-13,999] | 0.899 [0.898-0.899] | - | 12.72 (-0.98) | 14.07 | 14.48 |
| `nockpt` | 3/3 | 18,074 [18,071-18,085] | 1.161 | - | 25.88 (+12.17) | 27.11 | 27.49 (+12.32) |
| `nowgrad` | 3/3 | 11,091 [11,081-11,095] | 0.712 | - | 13.70 | 14.88 | 15.26 |
| `offload` | 3/3 | 6,469 [6,464-6,480] | 0.415 [0.415-0.416] | 0.583 [0.583-0.585] | 7.73 (-5.97) | 8.43 | 8.81 (-6.36) |
| `blocks_nowgrad_noclip` | 3/3 | 10,481 [10,471-10,481] | 0.673 | - | 11.64 | 12.85 | 13.08 |
| `lw` | 3/3 | 10,076 [10,065-10,087] | 0.647 | 0.962 [0.960-0.962] | 23.68 (+9.98) | 26.95 | 27.33 |
| `wsall` | 3/3 | 8,213 [8,198-8,215] | 0.527 | 0.784 [0.782-0.784] | 20.89 (+7.19) | 24.16 | 24.54 |
| `srbf16` | 3/3 | 9,542 [9,542-9,549] | 0.613 | 0.911 [0.910-0.911] | 21.87 (+8.17) | 25.15 | 25.53 |

## Option by option

**`--optim-state-offload`.** What it does: m and v live in pinned host memory and are staged to
the device, in f32, for each optimizer step and copied back on a transfer stream. The gradient
accumulator `m_partial` is also host-resident. Every micro-batch stages it to the device,
accumulates into it, and copies it back synchronously (`stmt_fase.rs` `fase_emit_accumulate`).
At 1B that is about 34 GB of PCIe traffic per update for `m_partial` and 21 GB for the step. It
refuses `--fuse-wgrad-accum`, whose device GEMM writes `m_partial` directly.

Expected saving: m + v + m_partial = 12.00 GiB (1B), 5.65 GiB plus 0.54 GiB of 2 MiB rounding
(500M). Measured against its control: -11.95 GiB (1B) and -5.97 GiB (500M); the surfaces show m, v
and m_partial at 0, and activations grow by 0.05-0.21 GiB of staging copies. Speed: 0.42x of `P`
at 1B bf16. 0.71x of that is the lost `--fuse-wgrad-accum`; the offload itself costs 0.60x of
its control. Host cost: 12.9 GB pinned at 1B.

Bit-exact against its control. Verdict: **keep** as the escape hatch for cards under ~24 GB,
and **fix** its composition. If `m_partial` stayed on the device, the accumulation would make
no PCIe trips, `--fuse-wgrad-accum` would compose, and the saving would be 8 GiB at about a
third of the traffic.

**`--param-dtype bf16-sr`.** Streamed layer parameters are stored as device bf16 with a
stochastically rounded fused AdamW update. f32 working copies exist only inside a layer's
bracket (`sr_bf16.rs`, `srbf16_widen_view` at 0.23 GiB at the peak). It requires
`--weight-stream`, which requires `--layerwise-accum`, so it has no clip and no
`--fuse-wgrad-accum`. The embedding (0.375 GiB) stays f32.

Expected: layer weights 3.625 -> 1.813 GiB at 1B. Measured: the weights surface goes 4.02 ->
2.43 (1.81 bf16 + 0.375 embedding + 0.23 for one f32 view). At 500M it is 0.81 GiB below `lw`
(expected 0.82). The run as a whole is +2.1 GiB above `P` at 1B, because of D1. Speed: 0.93x of
its control at 1B, 0.91-0.96x at 500M.

Numerics: held-out loss after 2,000 updates at 50M, against the same flags with f32 parameters:
6.328 / 6.448 / 6.402 vs 6.342 / 6.413 / 6.422 (seeds 1-3). The differences are -0.014, +0.034,
-0.020 (mean +0.000), inside the seed-to-seed spread (sd 0.057). Verdict: **fix**. Decoupled
from streaming, so that it runs on `P`'s schedule, it would remove both the f32 master (1.81
GiB) and the bf16 image cache (1.81 GiB) at 1B with no PCIe cost.

**`--wggo-moment-precision` and `--wggo-memory-budget`.** The mechanism exists: bf16 first moment,
f32 second moment (`cpdt_precision_exec.rs`; v is never narrowed, per bugs.md 2026-11-13). A
plan can never select it on these recipes:

* `--wggo-moment-precision` alone, or with `--wggo greedy|full`, produces a binary
  byte-identical (`cmp`) to the one built without it. WGGO only chooses sub-32-bit moments with
  "informed" sensitivity (`--wggo-weights`) or under `--wggo-memory-budget`. The activation
  report still prints `--wggo-moment-precision: witnessed by construction`.
* `--wggo-memory-budget N` is refused at compile time at every size: "96 tensor operation(s)
  in this train block's step have no compile-time size" (1B; 144 at 500M, 48 at 50M, i.e. 6 per
  block). This happens although the step is fed by a `DataLoader` with literal `batch_size`,
  `seq_len` and `drop_last=true`. WGGO's sizing (`wggo_model_shape.rs`) seeds from parameter dims
  and the loader, but not from the type annotations CCR's `ccr_static_sizes` also uses.

Expected saving if reachable: 2.0 GiB at 1B (2 of the 8 moment bytes per parameter). Its speed
is unknown: the reduced-precision envelope also takes the step off the fused AdamW kernel
(`stmt_fase.rs`, `fused_scalars` requires `!wrap_precision`). Verdict: **fix**, with a direct
switch and a fused kernel for bf16 m, or the WGGO sizing fix.

**`--wggo greedy`** (the plan the two flags above need; part of `--pretrain-optimized` until
the bundle was redefined, see the end).
It changes no memory and costs 18% at 1B bf16 and 11.5% at 500M TF32. The plan takes the
optimizer step off the single batched multi-tensor AdamW launch. Diffing the call lists of the
two binaries: `nsl_fase_fused_adamw_step_multi` and `nsl_fase_sum_sq_list` are replaced by
per-parameter `nsl_fase_fused_adamw_step`, `nsl_tensor_sum_sq`, and the stdlib
`nsl_optim_adamw__adamw_step`. It also prints `mem=28790.5MB` for a run whose measured peak is
27,166 MiB allocated. Verdict: **fix**, not a memory option.

**`--layerwise-accum` (CSLA).** It runs the window's N forwards first, then replays the backward
layer by layer, with each layer's update right after its range. That should replace the
full-model `m_partial` (4 GiB at 1B) with one layer's accumulator plus the epilogue's (measured
0.6 GiB peak). Window buffering should add N-1 micro-batches of saved imports (about 1.5 GiB of
boundaries at 1B, N=4).

Measured: OOM at 1B, and at 500M +9.97 GiB over its control (22.68 vs 10.66), with activations
16.5 GiB against 2.4. The cause (D1) is that `ccr_frees_per_segment()`
(`forward_lowering.rs:378`) returns false under `layerwise_accum`, and the weight-stream forward
path is excluded at `forward_lowering.rs:101`. So the Item-11 per-segment early-free does not
run, and every block's interiors stay live until the end of each micro-batch's forward. The
contexts at the 1B peak show it: `nsl_silu_f32` holds 2.00 GB, all 16 blocks' SwiGLU activations
(16 x 128 MiB), against 128 MB (one block) in the control. Building `P` itself with
`NSL_CCR_SEGMENT_FREE=0` OOMs at 1B in the first forward with the same signature (silu 1.62 GB,
13 of 16 blocks).

With selective checkpointing it OOMs at both sizes (it buffers N x the selective set). It also
refuses `grad_clip`, which at 50M was worth about 0.2 nats of held-out loss after 2,000 updates
(no-clip runs: mean 6.393; `P`: mean 6.196; the schedule itself is bit-exact, so the
difference is the clip). Speed: 0.96-0.97x of its control. Bit-exact. Verdict: **fix** (D1);
until then it is a memory regression.

**`--weight-stream`.** Layer parameters live in pinned host mirrors and are uploaded per bracket
(forward segment, backward replay range) and written back after their update. By default a
residency policy decides at the first registration whether anything streams: it streams until
the free VRAM covers a reserve of half the card.

On this card at 1B the policy saw 18,508 MiB free against a 16,082 MiB reserve. It streamed
nothing (`pinned=144 of 144`) and the run then OOMed (D2). The decision also depends on what
other processes hold on the GPU at that moment. With `NSL_WS_RESIDENT=0` everything streams: the
weights surface drops 4.02 -> 0.62 GiB at 1B (expected: 3.625 GiB of layer weights minus a 0.23
GiB bracket). That turns `lw`'s OOM into a 30.08 GiB-driver fit at 0.83x of the non-CSLA
control. PCIe traffic is 19.6 GB HtoD and 4.05 GB DtoH per 1B update.

Bit-exact (per-parameter, arena, prefetch, async-writeback). Verdict: **fix**. Its value only
appears once D1 is fixed, and the residency policy needs the planned activation peak (or should
stream whenever the flag is given).

**`--stream-arena`, `--stream-prefetch`, `--stream-async-writeback`.** They are slower than plain
streaming: 0.83x at both sizes, and prefetch plus async writeback recover at most 1.5%. Memory
is the same or up to 0.75 GiB worse. The arena's `upload_pack_inner` (`weight_stream.rs`) gathers
every parameter's pinned mirror into a pinned staging buffer with a single-threaded host
`memcpy` before each pack's one HtoD. At 1B that is 19.6 GB of host copies per update (D5).
Bit-exact. Verdict: **retire**, unless the mirrors are made contiguous per layer so the pack can
upload from them without a gather.

**`--checkpoint-compress bf16`.** It stores the 128 saved matmul-class outputs per micro-batch
(8 per block: q, k, v, the attention output, o, gate, up, down) as bf16. Expected: the f32
selective set is 6.75 GiB at 1B and 2.70 GiB at 500M. `P` already releases 16 of these early
(one per block, never read for value), so the saving is 2.9-3.4 GiB at 1B and 1.1-1.35 GiB at
500M. Measured: -3.40 and -1.30 GiB, at 0.92-0.95x.

Compression skips the release-early analysis (`ccr.rs`: the release scan reads
`per_segment_kept`, which compression leaves empty). The 16 unread outputs are therefore cast
and kept in bf16, about 0.25 GiB at 1B (D6). Numerics, 2,000 updates at 50M: bf16 GEMMs
6.157 / 6.202 / 6.185 vs `P` 6.189 / 6.142 / 6.256 (differences -0.033, +0.060, -0.071, mean
-0.014); TF32 6.159 / 6.191 vs 6.150 / 6.200 (+0.009, -0.008). All inside the seed spread (sd
0.057; a repeat of one seed moved 0.002). Verdict: **keep** (opt-in).

**Checkpointing variants.**

* Full block recompute (drop `--checkpoint-selective`): -5.38 GiB at 1B for 0.87-0.90x, and
  -2.06 GiB at 500M for 0.88-0.92x. Expected about 5.4 at 1B (the 6.25 GiB kept selective set,
  minus one block's recompute working set). Bit-exact. It is the best GiB-per-percent option
  measured and the one that brings the 1B recipe to PyTorch's driver peak.
* `--checkpoint-budget-mib 3072` on full recompute: the knapsack flips 58 tensors back to
  saved. At 1B that is -2.75 GiB at 0.935-0.947x. At 500M 3 GiB is more than the whole selective
  set (2.46 GiB), so it saves more than `P` (+1.02 GiB) and runs 1.02-1.03x faster.
* No checkpointing: OOM at 1B; at 500M +12.2 GiB (24.9 / 26.5 GiB) for 1.12x (TF32) and 1.16x
  (bf16). Bit-exact.
* `--checkpoint-stride 2`: +0.63 GiB at 1B (the backward recomputes two blocks' interiors at
  once), -0.12 GiB at 500M (expected -0.117: 12 boundaries of 10 MiB). Speed 1.01-1.02x.
  Bit-exact.

Verdicts: blocks and budget **keep**; none at 500M is **keep** (fits 32 GB, +12-16%); stride
**retire** with selective.

**`--no-bf16-cast-cache`.** The cache keeps one persistent bf16 image per weight matrix,
registered by the fused AdamW step. Expected 1.812 GiB at 1B; at 500M 0.824 GiB, plus 2 MiB
rounding, for 0.98. Measured -1.80 and -0.98 GiB (the `matmul_f32` context at the peak drops
7.30 -> 5.50 GiB), for 0.966x and 0.953x. Bit-exact. Verdict: **keep** (opt-in). The better
answer is bf16 parameters (bf16-sr on the normal schedule), where the image is the parameter.

**`--cuda-graphs`.** It captured 140 regions with 2,803 replays and 0 taints at 1B. It changes
no allocations itself (identical to `nocastcache` in bf16, identical to `P` in TF32). In bf16
it switches the cast cache off (`bf16_cast_cache.rs` refuses the composition), and in both
postures it turns off the timed cublasLt GEMM selection (`lt_matmul.rs`: the tune
synchronises). The net is 0.85-0.90x. Bit-exact. Verdict: **retire**: NSL is GPU-bound, and the
recipe header's "+0.8%" predates the GEMM tune. A fix would need the tune to finish before
capture.

**`--zero-stage 1`.** One device: every collective returns early (`world_size <= 1`), so it
saves nothing (0 GiB, 1.00x), refuses `grad_clip`, and is bit-exact with `P` without clip.
Stage 3 requires the layerwise family. Verdict: **retire** for one GPU (refuse at
`--devices 1`).

**`NSL_ASYNC_ALLOC=1`.** `cuMemAllocAsync` instead of the caching allocator. The allocator's
reserved figure goes blind, the driver peak rises by 1.18 GiB (1B) and 0.35 GiB (500M), and
speed is 0.97-0.98x. Verdict: **retire**.

## Numerics

`--deterministic` (bit comparison) and production mode (agreement with PyTorch), Coder-50M, 8
AdamW updates, the whole model (49,816,392 parameters):

| A / B | deterministic: params differing, losses | A vs PyTorch, prod. mode (median / worst rel. error) | B vs PyTorch, prod. mode |
|---|---|---|---|
| `P` / `P` again | 0, identical | 1.56e-3 / 2.36e-3 | 1.56e-3 / 2.37e-3 |
| `P` / `--cuda-graphs` (72 captured, 279 replays) | 0, identical | 1.56e-3 / 2.36e-3 | 1.55e-3 / 2.37e-3 |
| `P` / `--checkpoint-stride 2` | 0, identical | | 1.55e-3 / 2.37e-3 |
| `P` / full recompute | 0, identical | | 1.55e-3 / 2.38e-3 |
| `P` / no checkpointing | 0, identical | | 1.56e-3 / 2.38e-3 |
| `P` / `NSL_ASYNC_ALLOC=1` | 0, identical | | 1.55e-3 / 2.39e-3 |
| `nowgrad` / `--optim-state-offload` | 0, identical | 1.55e-3 / 2.40e-3 | 1.55e-3 / 2.39e-3 |
| `nowgrad_noclip` / `--layerwise-accum` (selective) | 0, identical | 1.59e-3 / 2.45e-3 | 1.59e-3 / 2.48e-3 |
| `blocks_nowgrad_noclip` / `--layerwise-accum` | 0, identical | 1.59e-3 / 2.47e-3 | 1.59e-3 / 2.46e-3 |
| `lw` / `--weight-stream` (default: nothing streamed) | 0, identical | | 1.59e-3 / 2.45e-3 |
| `lw` / `--weight-stream`, all streamed | 0, identical | | 1.59e-3 / 2.47e-3 |
| all streamed / `+ --stream-arena` | 0, identical | | 1.59e-3 / 2.47e-3 |
| all streamed / `+ arena + --stream-prefetch` | 0, identical | | 1.59e-3 / 2.47e-3 |
| all streamed / `+ arena + prefetch + --stream-async-writeback` | 0, identical | | 1.59e-3 / 2.47e-3 |
| `P` without clip / `--zero-stage 1` | 0, identical | 1.60e-3 / 2.49e-3 | 1.60e-3 / 2.47e-3 |
| `P` bf16 / `--no-bf16-cast-cache` | 0, identical | 4.82e-3 / 5.95e-3 | 4.80e-3 / 5.90e-3 |
| `P` / `nowgrad` (reference: `--fuse-wgrad-accum` is not bit-exact) | 48.4 M differ, from micro-batch 8 | | |

The cast-cache pair is meaningful: the same deterministic bf16 run with
`NSL_BF16_CAST_CACHE_COUNTER=1` reports `hits=504 recasts=168` and an identical `after.nslm`.
In production mode every pair, `P` against itself included, differs in about 97% of parameters
after 8 updates (max |difference| 3-11% of the update; `exact_table.txt`), so production mode
can only show agreement with PyTorch. The bit claims hold only under `--deterministic`, and
every claim checked there holds.

Training comparisons (held-out loss, 2,000 updates at 50M):

| arm | seed 1 | seed 2 | seed 3 | difference to its control (mean) |
|---|---|---|---|---|
| `P`, bf16 GEMMs | 6.1895 (repeat 6.1872) | 6.1420 | 6.2556 | - |
| `P` + `--checkpoint-compress bf16` | 6.1569 | 6.2019 | 6.1848 | -0.033, +0.060, -0.071 (-0.014) |
| `P`, TF32 | 6.1503 | 6.1995 | | - |
| `P` + compress, TF32 | 6.1591 | 6.1913 | | +0.009, -0.008 (0.000) |
| layerwise + weight-stream, f32 params (no clip) | 6.3422 | 6.4134 | 6.4221 | vs `P`: +0.153, +0.271, +0.167 (+0.197) |
| the same + `--param-dtype bf16-sr` | 6.3282 | 6.4476 | 6.4021 | -0.014, +0.034, -0.020 (0.000) |

## The largest remaining items at 1B, and what to build next

After these options the 1B recipe's allocator peak is still 16 GiB of f32 persistent state (the
same as PyTorch's), 1.81 GiB of bf16 weight images, and about 8.7 GiB of f32 activations and
transients. In the order I would build them, with estimated savings at 1B:

1. **bf16 activation storage** (in progress on `perf/bf16-storage`): about -3 GiB, estimated
   there.
2. **bf16-sr parameters on the normal schedule** (decouple `--param-dtype bf16-sr` from
   `--weight-stream`/CSLA): -1.81 GiB of f32 master for the layer weights and -1.81 GiB of cast
   cache, because the parameter is the image. No PCIe cost, and fewer casts. Its numerics are
   already measured above (no difference at 50M).
3. **A reachable bf16 first moment** (a direct switch plus a fused AdamW kernel reading and
   writing bf16 m; or fix WGGO's sizing): -2.0 GiB. Done: `first_moment_dtype="bf16"`, measured
   at the end.
4. **D1**: the per-segment early-free in the CSLA and weight-stream forwards. Then CSLA can drop
   `m_partial` (-3.4 GiB, but no clip) and streaming can drop the layer weights (-3.4 GiB, PCIe).
5. **Offload with a device-resident `m_partial`**, so `--fuse-wgrad-accum` composes: -8 GiB for
   about 17 GB per update of PCIe, overlappable with the next window's forward.
6. **Allocator**: flat per-kind buffers for parameters and optimizer state (no 2 MiB rounding:
   -0.72 GiB at 500M; also one launch for the multi-tensor optimizer).

Items 1-3 together take the 1B allocator peak to about 26.53 - 3.0 - 3.6 - 2.0 = **17.9 GiB**
(est.), against PyTorch's 20.58, without giving up speed. Full recompute (-5.4 GiB, 0.90x)
and offload (8-12 GiB) then remain as stated trade-offs for smaller cards.

## Defects found (not fixed here)

* **D1** (fixed 2026-11-21, `perf/d1-early-free`; see "D1 fixed" below): `--layerwise-accum` and `--weight-stream` lose the forward per-segment early-free
  (`forward_lowering.rs:378` `ccr_frees_per_segment`, `:101` ws path). Reproducer: perfcheck arms
  `nsl_bf16_lw` / `nsl_bf16_ws` at 1B (OOM), and `nsl_tf32_lw` at 500M (+9.97 GiB vs
  `nsl_tf32_blocks_nowgrad_noclip`). Contrast: `P` built with `NSL_CCR_SEGMENT_FREE=0` OOMs at 1B
  the same way (command under Reproduce).
* **D2**: the weight-stream residency policy pins every parameter on a 32 GB card at 1B and the
  run OOMs. Its decision depends on other processes' VRAM. Reproducer: `nsl_bf16_ws` at 1B,
  stderr line `[weight-stream] residency: pinned=144 of 144 ... free_at_decision=18508 MiB
  reserve=16082 MiB must_free=0 MiB` (with `NSL_WS_COUNTER=1`).
* **D3**: `--wggo-moment-precision` is inert on every recipe (binary byte-identical, reported as
  "witnessed"). `--wggo-memory-budget` is refused for 6 unsized ops per block. Reproducer:
  `perfcheck.py build --size 1b --arms nsl_bf16_moment` (refused); `nsl build` of the
  production program with and without `--wggo-moment-precision`, then `cmp` of the binaries.
* **D4**: `--wggo greedy` costs 12-18% by unbatching the optimizer step (call-list diff above).
  `--pretrain-optimized` includes it. (The bundle no longer does; D3's inert case is now a
  compile error. See the end.)
* **D5**: `--stream-arena` gathers host mirrors through a single-threaded `memcpy` per upload,
  making it 0.83x of per-parameter streaming.
* **D6**: `--checkpoint-compress` keeps the 16 outputs per micro-batch the backward never reads,
  in bf16 (about 0.25 GiB at 1B), and casts them.
* **D7**: the production flags are not run-to-run reproducible at 50M even with the GEMM tune
  off: two runs of `P` differ in 97% of parameters after 8 updates, first in the loss of
  micro-batch 9. Under `--deterministic` they are identical.
* **D8**: the caching allocator rounds large blocks to 2 MiB (0.72 GiB of persistent waste at
  500M).
* Stale documentation: `--fuse-wgrad-accum` is worth 1.41x at 1B bf16, 1.25x at 1B TF32, 1.29x
  at 500M TF32 and 1.40x at 500M bf16 (paired `P` / `nowgrad`; doc and recipe header: "+11.5%"). The recipe header's `--cuda-graphs`
  "+0.8%" predates the GEMM tune. The CLI help says `--weight-stream` holds a parameter's device
  buffer only inside its brackets, but by default on this card it streams nothing.

## Changes made after the measurements

Three commits on `perf/memory-audit` act on the verdicts above:

1. **Corrected claims.** This document; the `--fuse-wgrad-accum`, `--cuda-graphs`, offload and
   checkpointing figures in the recipe headers and CLI help; the `--weight-stream` help (it
   holds parameters on the device whenever its residency policy decides they fit); measured
   costs in the help of `--cuda-graphs`, the streaming sub-options, `--optim-state-offload` and
   `NSL_ASYNC_ALLOC`.
2. **Flags that say what they do.** `--wggo-moment-precision` on a train block where WGGO
   decides no reduced-precision moment (every recipe: no sensitivity data, no budget) is a
   compile error that names `first_moment_dtype="bf16"`, so it can no longer be reported as
   witnessed (D3, first half). `--pretrain-optimized` is now `P` plus `--fuse-lm-head auto`;
   `--wggo greedy` and `--csha auto` left it, and `--wggo`'s help states its cost (D4).
   `--zero-stage` with one device warns at compile time that it shards nothing.
3. **A reachable bf16 first moment**, below.

### `first_moment_dtype="bf16"`

`optimizer: AdamW(..., first_moment_dtype="bf16")` (also `Adam`; the default is `"f32"`; every
other optimizer refuses the argument). The first moment `m` is stored bf16 between steps,
rounded to nearest even on store; the update reads it widened and does its arithmetic in f32,
and `θ'` is computed from the f32 `m'`, not the rounded one. `v`, the f32 master weights and the
gradient accumulator are unchanged. It lives in the fused AdamW step: two KIR kernels
(`nsl_fase_fused_adamw_multi_f32_mbf16` and the per-parameter
`nsl_fase_fused_adamw_step_f32_mbf16`, `crates/nsl-kir/src/kernels/optim.rs`) chosen by `m`'s
dtype, which is also part of the multi-tensor launch's bucket key. So it needs the
FASE-Deferred plan (`grad_accumulation >= 2`). Every composition that would hand the bf16
buffer to f32 arithmetic is a compile error: no deferred plan, `--training-reference`,
`NSL_FASE_FUSED_STEP=0`, `--optim-state-offload`, `--layerwise-accum`, `--zero-stage`,
`--param-dtype bf16-sr`, WGGO/CPDT moment precision or per-layer plans, `@pipeline`. A
checkpoint's sidecar names each moment's dtype and the execution fingerprint carries `m1=bf16`.
A resume across a change of the argument is refused in both directions rather than converted:
f32 -> bf16 would round the saved moments, and bf16 -> f32 would continue a different optimizer
from the one that wrote them.

Measured with a toolchain frozen at the commit that adds the perfcheck arm (b5a3cf65):
`nsl_<prec>_m1bf16` (`P` plus the argument) paired with `nsl_<prec>` in one job, 3 rounds,
order alternating:

| posture | `P` tok/s, median [range] | bf16 m, paired ratio | allocator GiB | reserved GiB | driver GiB | `m` at the peak |
|---|---|---|---|---|---|---|
| 1B, bf16 GEMMs (recipe) | 9,587 [9,576-9,778] | 1.004 [0.992-1.004] | 26.53 -> **24.52 (-2.00)** | 27.41 -> 25.41 | 27.79 -> **25.79 (-2.00)** | 4.00 -> 2.00 |
| 1B, TF32 | 7,222 [7,218-7,237] | 1.001 [0.994-1.004] | 24.63 -> 22.63 (-2.00) | 25.48 -> 23.48 | 25.87 -> 23.87 (-2.00) | 4.00 -> 2.00 |
| 500M, TF32 (recipe) | 12,710 [12,708-12,720] | 1.001 [0.987-1.004] | 12.72 -> 11.76 (-0.96) | 14.05 -> 12.93 | 14.44 -> 13.31 (-1.12) | 2.06 -> 1.10 |
| 500M, bf16 GEMMs | 15,519 [15,507-15,571] | 1.003 [1.001-1.003] | 13.70 -> 12.74 (-0.96) | 14.79 -> 13.61 | 15.17 -> 13.99 (-1.18) | 2.06 -> 1.10 |

The saving is the predicted half of `m` (at 500M, 1.10 rather than 1.03 GiB remain: the 2 MiB
block rounding, D8, is a larger share of the smaller buffers). Speed does not move: every ratio's
range spans 1.0. The production arms reproduce the audit's numbers (identical memory; 1B bf16
9,587 against 9,626 tok/s).

Convergence, Coder-50M, held-out loss after 2,000 updates (the setup of the training comparisons
above: `P`, bf16 GEMMs, clip 1.0), `first_moment_dtype` f32 against bf16 with the same seeds and
data. The pass criterion was written down before any bf16 run: the mean paired difference must
be within 2 standard errors of the paired differences AND within the f32 arm's range across
seeds. A run that failed would have counted against its arm.

| seed | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| f32 m | 6.1684 | 6.1637 | 6.1982 | 6.1140 | 6.1343 |
| bf16 m | 6.1398 | 6.2001 | 6.2032 | 6.1141 | 6.1618 |
| difference | -0.0285 | +0.0363 | +0.0050 | +0.0001 | +0.0275 |

Mean difference +0.0081, standard error 0.0114 (2 SE = 0.0228), f32 range 0.0842: **pass** on
both conditions. Its resolution is limited: the 95% interval of the mean difference is -0.023 to
+0.040 (t with 4 degrees of freedom), so an effect smaller than about 0.04 in held-out loss is
not excluded. Part of the spread is production mode's run-to-run nondeterminism (D7), which a
fixed seed does not remove. Both arms' 50M peaks confirm the argument took effect (1,489,061,888
-> 1,375,798,272 bytes, -108 MiB for 49.8 M parameters).

Verdict: **keep**. It is 2 GiB at 1B for no speed, and passed the 50M test. Making it the
default would change every existing recipe's arithmetic and its checkpoint fingerprint. That
wants a longer run at 500M or 1B first, because the 50M test cannot see an effect under 0.04.

### D1 fixed (2026-11-21, `perf/d1-early-free`)

The layerwise forward (`--layerwise-accum`) and the streamed forward (`--weight-stream`) now free
each checkpointed block's interiors at the block's end, as the other schedules do. The CSLA window
also buffers the shape stubs the free lists make (one `nsl_tensor_shape_stub` per residual add
whose operand was freed), and the streamed forward frees a segment's interiors before evicting
its weights. `NSL_CCR_SEGMENT_FREE=0` restores the old lifetimes in every mode. Same harness as
above (`tools/perfcheck`, MEMORY_ARMS), toolchains `tc_fix` (b6a9e200) and `tc_main` (7eea4517),
3 rounds per arm (1 for the two 1B arms at main); allocator / reserved / driver peaks were
identical in every round.

| arm | alloc GiB | reserved | driver | tok/s median | paired |
|---|---|---|---|---|---|
| 500M TF32 `blocks_nowgrad_noclip` (control) | 10.66 | 11.75 | 12.13 | 9,862 | - |
| 500M TF32 `lw`, main | 21.56 | 25.06 | 25.44 | 9,558 | - |
| 500M TF32 `lw`, fixed | 18.19 | 19.85 | 20.22 | 9,598 | 1.005 vs main, 0.972 vs control |
| 500M TF32 `ws` (default residency), fixed | 18.19 | 19.85 | 20.22 | 9,580 | 0.999 vs `lw` |
| 500M TF32 `wsall`, fixed | 16.44 | 17.82 | 18.20 | 8,109 | 0.823 vs control |
| 1B bf16 `blocks_nowgrad_noclip` (control) | 21.32 | 22.21 | 22.59 | 6,741 | - |
| 1B bf16 `lw`, main | OOM | 29.14 | 29.52 | - | - |
| 1B bf16 `lw`, fixed | 19.57 | 20.39 | 20.77 | 6,691 | 0.993 vs control |
| 1B bf16 `ws` (default residency), fixed (main: OOM in the audit) | 19.57 | 20.39 | 20.77 | 6,699 | - |
| 1B bf16 `wsall`, main | 25.13 | 28.11 | 28.49 | 5,539 (1 round) | - |
| 1B bf16 `wsall`, fixed | 14.35 | 15.18 | 15.57 | 5,669 | 0.846 vs `ws` |

At 1B the per-block signature is gone. On main the peak held `nsl_silu_f32` 2.00 GB and
`nsl_mul_f32` 1.88-2.00 GB (all 16 blocks) and `rope` 2.76 GB; after the fix none of them is
among the largest contexts at the peak. `lw`'s activations are 6.95 GiB against the control's 5.30, and its `m_partial` is 0.60 GiB against 4.00: 1.75 GiB under the
control. The forward frees 624 interiors over 16 segments per micro-batch (compile-time witness).
`ws` at default residency pins all 144 parameters on this card (D2, not changed here), so it
measures the same as `lw`.

At 500M the fix removes 3.37 GiB (activations at the peak 15.36 -> 11.68 GiB), but `lw` stays
7.5 GiB above its control, and the design's estimate (about 0.25 GiB of buffered boundaries per
micro-batch) was wrong. The window's activations grow by 1,220 MB per buffered micro-batch
(`[gpu-mem]` surfaces, steps 1-5). At the peak, `dropout_f32` holds 7.50 GB in 768 blocks: 8
micro-batches x 24 blocks x 2 dropouts x (output + f32 mask), 10 MiB each. CCR never recomputes
a random draw, so both are force-saved, and the window holds them for all 8 micro-batches where
the control holds one (960 MB). The residual-stream boundaries are another 1.81 GB
(`nsl_add_f32`). The 500M recipe has dropout 0.1; 1B has none.
Not done here, and unmeasured: the backward appears to read each dropout output only through
the residual add's `reduce_to_shape` (500M reports 0 shape-only reads of recompute victims, 1B
reports 16), so a stub for force-saved outputs could save up to 480 MiB per buffered
micro-batch; a 1-byte mask a further 360 MiB.

Bit-exactness at 50M (`--deterministic`, `PERF-D1/exact_det.sh`): `lw` against a repeat of
itself, `blocks_nowgrad_noclip`, `lw` with `NSL_CCR_SEGMENT_FREE=0`, `ws`, `wsall`; `wsall`
against `wsall` with the kill switch and against `--stream-arena`; `nowgrad_noclip` against
`lwsel`; `lwsel` against `lwsel` with the kill switch. Each has init equal, 0 of 49,816,392
parameters differing and identical losses, and every run agrees with PyTorch (median 1.69e-3,
worst 2.40e-3).

## Reproduce

`VENV` is a Python environment with CUDA PyTorch (the perfcheck harness, `tools/perfcheck/`);
`DIR` a frozen toolchain (`bin/nsl`, `lib/libnsl_runtime.a`, `stdlib/`); every GPU command was
one queued job of at most ~15 minutes on the measuring machine.

```bash
P="VENV/bin/python tools/perfcheck/perfcheck.py"; T="--toolchain mem=DIR"; OUT=/a/real/disk/mem
# build every arm (CPU only); an arm that fails to compile is a result (see `moment`)
$P build --size 1b --arms nsl_bf16@mem,nsl_bf16_offload@mem,... $T --out $OUT
# one job: the production arm first, then up to six options; --round = 10*ROUND + job
$P run --size 1b --arms nsl_bf16@mem,nsl_bf16_compress@mem,nsl_bf16_blocks@mem,nsl_bf16_budget3g@mem,nsl_bf16_nocastcache@mem,nsl_bf16_graphs@mem \
   --round 10 $T --out $OUT
$P run --size 1b --arms nsl_bf16@mem,nsl_bf16_nowgrad@mem,nsl_bf16_offload@mem,nsl_bf16_stride2@mem,nsl_bf16_wggo@mem --round 11 $T --out $OUT
$P run --size 1b --arms nsl_bf16@mem,nsl_bf16_blocks_nowgrad_noclip@mem,nsl_bf16_srbf16@mem,nsl_bf16_wsall@mem,nsl_bf16_wsarena@mem --round 12 $T --out $OUT
$P run --size 1b --arms nsl_bf16@mem,nsl_bf16_wsprefetch@mem,nsl_bf16_wsasync@mem --round 13 $T --out $OUT
$P pairs --out $OUT --pair nsl_bf16_offload@mem/nsl_bf16@mem --pair nsl_bf16_offload@mem/nsl_bf16_nowgrad@mem
$P report --out $OUT

# exactness at 50M: the option and its control, both --deterministic, then compare after.nslm
VENV/bin/python tools/refcheck/optimizer_check.py --size 50m --tokens train.bin --nsl DIR/bin/nsl \
   --workdir w/offload --steps 8 --accum 4 --wd 0.1 --schedule --clip 1.0 -- \
   --source-ad --checkpoint-blocks --checkpoint-selective --fuse-rmsnorm-backward \
   --optim-state-offload --matmul-mode tf32 --deterministic

# the early-free contrast for D1: the production flags without the per-segment free OOM at 1B
NSL_CCR_SEGMENT_FREE=0 NSL_MATMUL_BF16=1 DIR/bin/nsl build --source-ad --checkpoint-blocks \
   --checkpoint-selective --fuse-rmsnorm-backward --fuse-wgrad-accum bench.nsl -o prog
```

The weight-stream arms run with `NSL_WS_COUNTER=1`, which prints the residency decision and the
transfer counters at exit. The first-moment arms are `nsl_<tf32|bf16>_m1bf16` (perfcheck
`OPTIM_ARMS`: the argument is part of the generated program, so of the build key), run as
`--arms nsl_bf16@m1,nsl_bf16_m1bf16@m1 --toolchain m1=DIR`. The 50M training comparisons are `train(...)` blocks with the 50M
model (dropout 0), `DataLoader(batch_size=1, seq_len=1024, shuffle=false)` over the first
2,000 x 4,096 tokens of the refcheck corpus (`tools/refcheck/make_corpus.py`), AdamW(lr 3e-4,
weight decay 0.1, betas 0.9/0.95), `warmup_cosine(100, 2000, 3e-5)`, accumulation 4, then a
forward-only loop over `val.bin`, `nsl run --seed S`.
