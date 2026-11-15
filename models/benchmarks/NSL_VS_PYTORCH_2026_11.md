# NSL against PyTorch on one GPU: speed and memory (2026-11-14)

The first measurement of NSL against PyTorch on the same model, the same tokens per
step and the same card. Every number in this document was measured on this machine
by `tools/perfcheck/` unless it is marked as an estimate: **estimates are written in
italics with "(est.)"**; everything else is a measurement.

## Summary

At the production recipes' shapes, PyTorch runs the same training step
**1.44x (1B) and 1.53x (500M) faster than NSL at the same precision policy** (f32
weights and activations, TF32 matmuls, the same selective checkpointing), at about
the same memory. With bf16, the recipe PyTorch users actually run (autocast plus
`torch.compile`) is **2.23x (1B) and 2.14x (500M) faster than NSL's
`NSL_MATMUL_BF16=1` and uses 6 GiB less memory at 1B** (20.6 against 26.5 GiB
allocator peak). Ratios are paired within a round, medians over rounds.

| Coder-1B, 16,384 tokens per update | tok/s | MFU | allocator peak |
|---|---|---|---|
| NSL production posture (TF32) | 4,916 | 42.2% | 24.63 GiB |
| PyTorch eager, TF32, same checkpointing | 7,259 | 62.3% | 24.87 GiB |
| NSL, `NSL_MATMUL_BF16=1` | 6,747 | 27.5% | 26.53 GiB |
| PyTorch bf16 autocast + `torch.compile`, same checkpointing | 14,981 | 61.1% | 20.58 GiB |

(Medians of four rounds; MFU against the TF32 and bf16 rooflines measured below.)

NSL is not host-bound: the GPU is busy 96% of an update. The time goes to kernels
that are slower than PyTorch's for the same work, in three places that account for
the whole TF32 gap at 1B (per optimizer update, from Nsight Systems):

- **attention**: 673 ms against 74 ms. NSL's flash-attention backward takes 8.6 ms per
  layer and micro-batch where PyTorch's FlashAttention-2 takes 0.77 ms (11x), and its
  forward 1.76 ms against 0.30 ms (5.9x);
- **GEMMs**: 2,110 ms against 1,520 ms for the same FLOPs. cuBLAS 13.8's `sgemm`
  heuristic picks 64x64-tile sm80 kernels for most of NSL's backward GEMMs and five of
  the seven forward ones; PyTorch's cuBLAS 12.8 picks 128x128 and 256x128 tiles;
- **host synchronisation**: 131 ms of GPU idle against 9 ms, from 536 synchronous
  host-to-device copies per update and a per-micro-batch release of the allocator's
  cache to the driver.

Everything else (norms, RoPE, SwiGLU, copies, loss, optimizer) costs NSL about what
it costs eager PyTorch at TF32. It is where `torch.compile` pulls ahead at bf16.

The 1B production posture reproduces the August record (5,053 tok/s at 25.2 GiB,
`POSTURE_1B_THROUGHPUT_2026_08_27.md`): 4,678-5,068 tok/s over four rounds at
24.63 GiB on today's `main`.

## Setup

| | |
|---|---|
| card | NVIDIA RTX PRO 4500 Blackwell (GB203, sm_120), 32 GB GDDR7, 200 W board power cap (every timed run sat at the cap: 199.9 W mean) |
| driver | 620.70.06 |
| NSL | `main` at 565567c5, `cargo build --release --features cuda`; links the system cuBLAS 13.8.0.4 (CUDA 13.4) |
| PyTorch | 2.11.0+cu128 (cuBLAS 12.8.4.1, Triton 3.6.0), Python 3.12, in `/mnt/extrastorage/nsl-audit/venv-cuda` |
| harness | `tools/perfcheck/` (commits 8048a8ad, 78243bc1) |
| models | Coder-500M (d 1280, 24 layers, 20/10 heads, d_ff 3520, dropout 0.1) and Coder-1B (d 2048, 16 layers, 32/8 heads, d_ff 8192, dropout 0) from `models/coder*/config.nsl`; vocab 49,152, tied LM head |
| step shapes | the production recipes': 500M micro-batch 2 x 1024 tokens, accumulation 8; 1B 2 x 2048, accumulation 4. 16,384 tokens per optimizer update in both |
| data | synthetic: a seeded uniform draw over the vocabulary, the same file and the same windows for both frameworks. The step does the same work whatever the ids are |

### Method

- **NSL arms** run a program `perfcheck.py build` generates from the production recipe
  (`pretrain_prod.nsl`'s train block: AdamW, warmup_cosine, `grad_clip=1.0`,
  `@fused_lm_ce`, `schedule_unit="optimizer_step"`) without checkpoint writes or
  validation, built once with `nsl build` and run as a binary. The build is refused if
  the fused LM cross-entropy did not engage, and each run is refused unless the runtime's
  own `cuBLAS math mode:` banner names the arm's precision.
- **PyTorch arms** run `torch_arm.py`, one fresh process per run.
- **Timing.** The token file holds exactly 24 optimizer updates. Each update prints its
  loss (NSL) or calls `loss.item()` (PyTorch): one host synchronisation per update on
  both sides, timestamped when it arrives. The first 9 updates are not timed (cuBLAS
  autotuning, `torch.compile`, allocator growth): tok/s is 15 updates x 16,384 tokens
  over the wall time from the end of update 9 to the end of update 24.
- **Repetition.** Three rounds, and a fourth for the five core arms (it repeats the
  round in which one arm was disturbed, see Results). Each round runs every arm once,
  each arm in its own process, the order rotated by the round number so no arm always
  runs first on a cool card or last on a hot one. Tables give the median over rounds
  with [min-max] across rounds, and the median of the within-run coefficient of
  variation of the 15 timed update times.
- **Contention.** Every GPU job went through `nslq gpu` (one GPU job at a time, none
  longer than ~11 minutes) and ran at idle CPU priority (`chrt -i 0`). Another project's
  builds and other sessions' CPU work (tests, a mutation-testing run at nice 19, which
  still preempts an idle-class process) shared the 16-core CPU during the campaign
  (load average 3-12). Both frameworks' host threads are sensitive to that, which is
  one reason for the rotation, the spread columns and the paired ratios.
- **Power cap.** The card held 200 W in every steady window, so its SM clock is set by
  the workload: 2,170-2,250 MHz on the TF32 arms, up to 2,430 MHz on NSL's bf16 arm,
  whose kernels draw less. The SM clock column is the mean over the timed window. Clocks
  drift down ~1-3% over a run as the card warms; rotation spreads that over the arms.
- **MFU** = tok/s x model FLOPs per token / roofline. Model FLOPs per token are
  `models/benchmarks/mfu_bench.py`'s: 6 x (parameters in a matmul per token, the tied
  LM head included, the embedding gather not) plus causal attention
  (6 x layers x heads x head_dim x seq). Recompute is not counted. The roofline is the
  best dense GEMM rate measured on this card in the arm's precision (below).
- **Memory**, three whole-process peaks per run:
  *alloc*, the caching allocator's peak of bytes handed to tensors (NSL
  `gpu_peak_bytes()`, PyTorch `torch.cuda.max_memory_allocated()`); *reserved*, the
  allocator's peak of bytes held from the driver (NSL `NSL_MEMSTATS=1` "Peak reserved",
  PyTorch `max_memory_reserved()`); *driver*, the process's used memory from
  `nvidia-smi --query-compute-apps`, polled every 100 ms (CUDA context, library
  workspaces and module images included). The card-sizing number is *driver*.

## Roofline

Dense GEMM throughput measured on this card with `torch.mm` (cuBLAS 12.8), CUDA
events, three batches of 30 launches after 10 warm-up launches, median batch; two
measurements a few minutes apart, before the timed rounds (`perfcheck.py roofline`). The MFU denominator is the
best square result in the arm's precision: **80 TFLOPS TF32, 168 TFLOPS bf16**.

| precision | 4096^3 | 8192^3 |
|---|---|---|
| f32 (no TF32) | 29.6 / 30.0 | 30.8 / 30.9 |
| TF32 | 77.3 / 75.4 | 79.9 / 79.7 |
| bf16 | 166.2 / 167.9 | 151.4 / 149.6 |

The bf16 8192^3 GEMM is slower than 4096^3 because it holds the 200 W cap longer and
the clock drops. The models' own projection shapes (rows = tokens per micro-batch)
reach 67-85 TFLOPS TF32 and 141-162 TFLOPS bf16 (`roofline.json`, `shapes`).

Through NSL's own matmul path (`a @ b`, 8192^3, f32 tensors), the same probe gives
71.4 TFLOPS in the default TF32 mode (cuBLAS 13.8, 10% under PyTorch's 79.7 on the
identical shape) and 154.4 TFLOPS built with `NSL_MATMUL_BF16=1` (operand casts
served from the weight-cast cache).

## Results

Median over rounds [min-max across rounds]; "CV" is the median within-run coefficient
of variation of the 15 timed update times. MFU uses the roofline above. Arms:
`nsl_tf32` is the production posture, `nsl_bf16` the same flags with
`NSL_MATMUL_BF16=1`, `_nodrain` adds `NSL_SKIP_GPU_DRAIN=1`, `_blocks` drops
`--checkpoint-selective` (full block recompute), `_compress` adds
`--checkpoint-compress bf16`; `torch_<precision>_<eager|compile>_<sel|blocks|none>`
names the PyTorch arm's precision, compilation and checkpointing.

### Coder-1B (micro-batch 2 x 2048, accumulation 4)

| arm | rounds | tok/s median [min-max] | s/update median | within-round CV | MFU | alloc peak GiB | reserved peak GiB | driver peak GiB | SM MHz | status |
|---|---|---|---|---|---|---|---|---|---|---|
| nsl_tf32 | 4/4 | 4,916 [4,678-5,068] | 3.344 [3.233-3.497] | 0.007 [0.004-0.020] | 42.2% | 24.63 [24.63-24.63] | 25.51 [25.51-25.51] | 25.83 [25.83-25.83] | 2244 [2206-2290] | ok |
| nsl_tf32_nodrain | 3/3 | 5,147 [5,139-5,152] | 3.177 [3.167-3.180] | 0.006 [0.006-0.009] | 44.2% | 24.63 [24.63-24.63] | 25.47 [25.47-25.47] | 25.79 [25.79-25.79] | 2178 [2176-2183] | ok |
| nsl_tf32_blocks | 3/3 | 4,456 [4,448-4,458] | 3.666 [3.646-3.674] | 0.008 [0.007-0.029] | 38.2% | 19.25 [19.25-19.25] | 20.18 [20.18-20.18] | 20.49 [20.49-20.49] | 2183 [2182-2193] | ok |
| nsl_tf32_compress | 3/3 | 1,680 [1,604-1,685] | 9.639 [9.596-10.208] | 0.018 [0.009-0.021] | 14.4% | 21.24 [21.24-21.24] | 22.31 [22.31-22.31] | 22.63 [22.63-22.63] | 2612 [2612-2623] | ok |
| nsl_bf16 | 4/4 | 6,747 [6,190-6,761] | 2.429 [2.421-2.644] | 0.004 [0.001-0.006] | 27.5% | 26.53 [26.53-26.53] | 27.44 [27.44-27.44] | 27.75 [27.75-27.75] | 2430 [2417-2493] | ok |
| nsl_bf16_nodrain | 3/3 | 6,843 [6,775-6,931] | 2.371 [2.361-2.380] | 0.015 [0.007-0.029] | 27.9% | 26.53 [26.53-26.53] | 27.41 [27.41-27.41] | 27.72 [27.72-27.72] | 2402 [2389-2410] | ok |
| nsl_bf16_compress | 3/3 | 1,812 [1,708-1,840] | 8.965 [8.831-9.563] | 0.018 [0.007-0.022] | 7.4% | 23.13 [23.13-23.13] | 23.96 [23.96-23.96] | 24.28 [24.28-24.28] | 2688 [2687-2694] | ok |
| torch_tf32_eager_sel | 4/4 | 7,259 [6,719-7,278] | 2.254 [2.247-2.431] | 0.008 [0.005-0.012] | 62.3% | 24.87 [24.87-24.87] | 25.21 [25.21-25.21] | 25.55 [25.55-25.55] | 2210 [2200-2282] | ok |
| torch_tf32_eager_blocks | 3/3 | 6,368 [6,363-6,463] | 2.569 [2.536-2.572] | 0.006 [0.003-0.006] | 54.7% | 18.86 [18.86-18.86] | 19.79 [19.79-19.79] | 20.12 [20.12-20.12] | 2167 [2166-2195] | ok |
| torch_bf16_eager_sel | 4/4 | 11,291 [10,468-11,474] | 1.446 [1.427-1.558] | 0.008 [0.001-0.019] | 46.0% | 23.23 [23.23-23.23] | 24.09 [24.09-24.09] | 24.43 [24.43-24.43] | 2340 [2315-2398] | ok |
| torch_bf16_compile_sel | 4/4 | 14,981 [13,867-15,150] | 1.091 [1.082-1.176] | 0.007 [0.001-0.013] | 61.1% | 20.58 [20.58-20.58] | 21.91 [21.91-21.91] | 22.25 [22.25-22.25] | 2239 [2227-2321] | ok |
| torch_bf16_compile_blocks | 3/3 | 12,954 [12,938-13,107] | 1.266 [1.250-1.266] | 0.001 [0.001-0.002] | 52.8% | 20.58 [20.58-20.58] | 21.20 [21.20-21.20] | 21.54 [21.54-21.54] | 2222 [2219-2246] | ok |
| torch_bf16_compile_none | 3/3 | 15,637 [15,612-15,778] | 1.047 [1.038-1.049] | 0.001 [0.001-0.001] | 63.8% | 23.76 [23.76-23.76] | 24.16 [24.16-24.16] | 24.50 [24.50-24.50] | 2243 [2236-2255] | ok |

### Coder-500M (micro-batch 2 x 1024, accumulation 8)

| arm | rounds | tok/s median [min-max] | s/update median | within-round CV | MFU | alloc peak GiB | reserved peak GiB | driver peak GiB | SM MHz | status |
|---|---|---|---|---|---|---|---|---|---|---|
| nsl_tf32 | 4/4 | 8,902 [8,329-8,971] | 1.839 [1.827-1.960] | 0.005 [0.001-0.013] | 36.0% | 12.72 [12.72-12.72] | 13.84 [13.84-13.84] | 14.15 [14.15-14.15] | 2329 [2322-2388] | ok |
| nsl_tf32_nodrain | 3/3 | 9,401 [9,355-9,408] | 1.742 [1.739-1.748] | 0.005 [0.003-0.007] | 38.0% | 12.72 [12.72-12.72] | 14.05 [14.05-14.05] | 14.37 [14.37-14.37] | 2255 [2255-2255] | ok |
| nsl_tf32_blocks | 3/3 | 8,000 [7,756-8,026] | 2.043 [2.037-2.127] | 0.007 [0.007-0.031] | 32.3% | 10.66 [10.66-10.66] | 11.80 [11.80-11.80] | 12.11 [12.11-12.11] | 2259 [2258-2292] | ok |
| nsl_bf16 | 4/4 | 11,548 [10,749-11,578] | 1.418 [1.415-1.520] | 0.002 [0.001-0.013] | 22.2% | 13.70 [13.70-13.70] | 14.83 [14.83-14.83] | 15.14 [15.14-15.14] | 2460 [2454-2521] | ok |
| nsl_bf16_nodrain | 3/3 | 12,129 [12,066-12,160] | 1.348 [1.344-1.352] | 0.008 [0.007-0.010] | 23.3% | 13.70 [13.70-13.70] | 14.79 [14.79-14.79] | 15.10 [15.10-15.10] | 2408 [2405-2412] | ok |
| torch_tf32_eager_sel | 4/4 | 13,615 [12,537-13,750] | 1.195 [1.182-1.302] | 0.017 [0.013-0.022] | 55.0% | 11.45 [11.45-11.45] | 11.79 [11.79-11.79] | 12.12 [12.12-12.12] | 2250 [2241-2335] | ok |
| torch_tf32_eager_blocks | 3/3 | 11,675 [11,638-11,799] | 1.379 [1.378-1.394] | 0.023 [0.014-0.042] | 47.1% | 9.09 [9.09-9.09] | 9.59 [9.59-9.59] | 9.93 [9.93-9.93] | 2209 [2200-2213] | ok |
| torch_bf16_eager_sel | 4/4 | 17,784 [16,239-18,181] | 0.907 [0.891-1.006] | 0.028 [0.020-0.041] | 34.1% | 10.94 [10.94-10.94] | 11.85 [11.85-11.85] | 12.19 [12.19-12.19] | 2349 [2325-2418] | ok |
| torch_bf16_compile_sel | 4/4 | 24,010 [23,094-25,075] | 0.679 [0.654-0.708] | 0.015 [0.002-0.027] | 46.1% | 10.01 [10.01-10.01] | 11.04 [11.04-11.09] | 11.38 [11.38-11.43] | 2312 [2277-2354] | ok |
| torch_bf16_compile_blocks | 3/3 | 21,086 [20,964-21,207] | 0.777 [0.773-0.778] | 0.002 [0.002-0.014] | 40.5% | 9.99 [9.98-9.99] | 11.08 [11.01-11.08] | 11.42 [11.35-11.42] | 2280 [2280-2288] | ok |
| torch_bf16_compile_none | 3/3 | 26,161 [26,110-26,357] | 0.626 [0.622-0.627] | 0.002 [0.002-0.002] | 50.2% | 11.13 [11.13-11.13] | 12.23 [12.23-12.26] | 12.57 [12.57-12.60] | 2285 [2269-2285] | ok |

### Paired ratios

Each ratio is taken within one round (the two arms ran minutes apart under the same
conditions), then the median over rounds; the per-round values are given so the spread
is visible. Rounds 0-3 for the core arms, 0-2 for the others.

| comparison | 1B per round | 1B | 500M per round | 500M |
|---|---|---|---|---|
| PyTorch eager TF32 / NSL TF32 (same policy, same checkpointing) | 1.431 1.516 1.444 1.436 | **1.44** | 1.533 1.540 1.518 1.505 | **1.53** |
| PyTorch eager bf16 / NSL bf16 | 1.700 1.664 1.680 1.691 | **1.69** | 1.563 1.578 1.509 1.511 | **1.54** |
| PyTorch compiled bf16 / NSL bf16 | 2.245 2.222 2.215 2.240 | **2.23** | 2.167 2.025 2.132 2.148 | **2.14** |
| PyTorch compiled bf16 / NSL TF32 (each one's production posture) | 2.989 3.129 2.969 2.964 | **2.98** | 2.795 2.623 2.772 2.773 | **2.77** |
| NSL bf16 / NSL TF32 | 1.332 1.408 1.340 1.323 | 1.34 | 1.290 1.295 1.300 1.291 | 1.29 |
| PyTorch eager bf16 / eager TF32 | 1.582 1.546 1.560 1.558 | 1.56 | 1.316 1.327 1.292 1.295 | 1.31 |
| PyTorch compiled / eager (bf16) | 1.320 1.335 1.318 1.325 | 1.32 | 1.386 1.283 1.413 1.422 | 1.40 |
| NSL selective / full recompute (TF32) | 1.137 1.079 1.129 | 1.13 | 1.118 1.112 1.148 | 1.12 |
| PyTorch selective / full recompute (eager TF32) | 1.122 1.144 1.141 | 1.14 | 1.178 1.162 1.162 | 1.16 |
| PyTorch selective / full recompute (compiled bf16) | 1.156 1.160 1.155 | 1.16 | 1.182 1.107 1.178 | 1.18 |
| PyTorch no checkpointing / selective (compiled bf16) | 1.042 1.039 1.047 | 1.04 | 1.051 1.121 1.058 | 1.06 |
| NSL without the per-micro-batch drain, TF32 | 1.017 1.071 1.023 | 1.02 | 1.049 1.057 1.050 | 1.05 |
| NSL without the per-micro-batch drain, bf16 | 1.004 1.012 1.028 | 1.01 | 1.051 1.052 1.042 | 1.05 |
| NSL `--checkpoint-compress bf16`, TF32 | 0.332 0.351 0.319 | 0.33 | not run | |
| NSL `--checkpoint-compress bf16`, bf16 | 0.268 0.272 0.253 | 0.27 | not run | |

### Reading the tables

- **Same precision policy, same checkpointing** (`nsl_tf32` against
  `torch_tf32_eager_sel`): PyTorch is 1.44x faster at 1B and 1.53x at 500M; the
  allocator peaks are within 0.3 GiB at 1B and 1.3 GiB at 500M (NSL higher at 500M).
- **bf16.** NSL's bf16 mode buys 1.34x (1B) and 1.29x (500M) over its TF32; PyTorch's
  eager autocast buys 1.56x and 1.31x over its TF32, and `torch.compile` another 1.32x
  and 1.40x. NSL's bf16 mode *raises* its memory by 1.9 GiB (1B) through the bf16
  weight-cast cache; autocast lowers PyTorch's (bf16 activations).
- **Checkpointing.** Selective over full block recompute is worth 1.13x on NSL and
  1.14x / 1.16x on PyTorch at 1B, for +5.4 GiB (NSL) and +6.0 GiB (PyTorch eager TF32).
  Under `torch.compile` selective and full recompute peak at the same 20.58 GiB: their
  forward peaks differ (20.0 against 17.1 GiB) but the compiled backward's own working
  set sets the peak in both. No checkpointing at all fits at 23.8 GiB and is 1.04x
  faster than selective.
- **The per-micro-batch drain** (`NSL_SKIP_GPU_DRAIN=1`) is worth 1.02x (1B TF32),
  1.01x (1B bf16), 1.05x (500M TF32) and 1.05x (500M bf16), at the same allocator peak
  (the reserved peak rises by at most 0.2 GiB).
- **`--checkpoint-compress bf16`** (1B only) lowers NSL's allocator peak by 3.4 GiB
  (24.63 to 21.24 GiB at TF32, 26.53 to 23.13 at bf16) and runs at 0.33x and 0.27x the
  speed: 9.6 and 9.0 s per update.
- **Spread.** In rounds 0-2, 66 of the 72 arm-rounds lie within 2% of their arm's
  median. The six exceptions are single rounds 3-6% slow: 1B `nsl_tf32` round 1 (ran
  at load average 12, the card no hotter than in the other rounds), both 1B compress
  arms in round 2, and three 500M arms in rounds 1-2. The fourth round of the core
  arms (run to repeat the disturbed one, right after the last compress round) came out
  6.5-10% slower on every arm at both sizes at once, with SM clocks 2-4% higher, the
  card in P1 and no other compute process on it; the cause was not found. The medians
  include every round. The paired ratios cancel machine-wide shifts like that one:
  each lies within 5% of its median, most within 1-2%. The 500M PyTorch eager arms
  have the largest within-run CV (2-4%): their 15 timed updates last 13-18 s.
- **Sanity.** On these tokens both frameworks start at loss 11.20-11.23 (1B) and
  11.05-11.07 (500M) and fall by 0.20-0.26 and 0.04-0.09 over the 24 updates; NSL's
  TF32 and bf16 arms agree to 1e-4 at the last update.

## What is and is not like for like

Matched on both sides:

| | NSL | PyTorch |
|---|---|---|
| model | `models/coder*/model.nsl` (NSLCoder: tied embedding, RMSNorm eps 1e-5, RoPE, GQA, SwiGLU) | `tools/refcheck/ref.py`'s twin of the same model, same parameter shapes and `[in, out]` layout, same init distributions (not the same values) |
| tokens per optimizer update | 16,384 (1B: 2 x 2048 x 4; 500M: 2 x 1024 x 8) | same windows of the same token file |
| optimizer | AdamW b1 0.9, b2 0.95, eps 1e-8, wd 0.1 on every parameter, global-norm clip 1.0, warmup_cosine per update, accumulation as a mean | `torch.optim.AdamW(fused=True)`, `clip_grad_norm_(foreach=True)`, same schedule, `loss / accum` |
| dropout | 0.1 at 500M on the attention and FFN outputs, 0 at 1B (`const DROPOUT`) | same places, same rates |
| master weights | f32 | f32 |
| host syncs | one per update (the loss is printed) | one per update (`loss.item()`) |
| attention | its own flash kernel on f16 operands (gradients scaled by a power of two), K/V copied out to every query head first (4x at 1B, 2x at 500M) | FlashAttention-2 (`pytorch_flash`) on bf16 operands, GQA native (`enable_gqa=True`); the run refuses unless a flash kernel ran |
| checkpointing, "selective" | `--checkpoint-blocks --checkpoint-selective`: per block, matmul / SDPA outputs saved, norms, RoPE, elementwise recomputed; dropout results force-saved | per block, `checkpoint(use_reentrant=False)` with a selective policy: `mm`/`addmm`/`bmm`/flash-attention outputs MUST_SAVE, everything else PREFER_RECOMPUTE; dropout replayed from the saved RNG state |
| checkpointing, "blocks" | `--checkpoint-blocks`: every block interior recomputed | `checkpoint(use_reentrant=False)` per block |

Not matched, and which way it leans:

- **Cross-entropy.** NSL's `@fused_lm_ce` runs the LM head GEMM in chunks and never
  materialises the `[4096, 49152]` logits. PyTorch has no fused linear cross-entropy in
  core; these arms materialise the logits (f32 in the TF32 arm: 0.75 GiB plus its
  gradient; bf16 under autocast, upcast to f32 inside `cross_entropy`). This favours
  NSL on memory. Nothing here was done to give PyTorch a chunked loss (Liger-style
  kernels exist outside core).
- **What "bf16" means.** `NSL_MATMUL_BF16=1` casts GEMM operands to bf16 (weights
  through a cast cache) and keeps f32 accumulation, f32 GEMM outputs and f32 storage
  for every activation; low-intensity products stay TF32. `torch.autocast(bfloat16)`
  produces bf16 GEMM outputs, so most saved activations and most elementwise traffic
  are bf16 (norms, softmax and the loss upcast to f32). The two "bf16" arms therefore
  move different numbers of bytes; the PyTorch one is the recipe people actually use.
- **GEMM library.** NSL links the system cuBLAS 13.8.0.4 (CUDA 13.4); the PyTorch wheel
  ships cuBLAS 12.8.4.1. They pick different kernels for the same shape.
- **Fusion.** `torch.compile` (Inductor/Triton) fuses norms, RoPE, SwiGLU and the
  residual adds into generated kernels; NSL has its own fused elementwise chains, a
  fused RMSNorm backward and a fused weight-gradient accumulate. Eager PyTorch fuses
  nothing beyond its library kernels.
- **Gradient accumulation.** NSL (`--fuse-wgrad-accum`, FASE Deferred) writes each
  weight gradient straight into its accumulation buffer with one cuBLAS call and
  steps all 146 parameters in one fused AdamW launch. PyTorch autograd accumulates
  into `.grad` and the fused AdamW is a multi-tensor launch.
- **Allocators.** Both are caching allocators; NSL additionally releases its idle
  transient pool to the driver after every micro-batch (`nsl_gpu_drain_cache`,
  disabled by `NSL_SKIP_GPU_DRAIN=1`; its cost is in Results).

## Where NSL's step time goes (Coder-1B, production posture)

Three instruments, each in its own run so none of them is in the timed numbers:

1. `NSL_PHASE_TIMING=1` builds (device syncs around each phase, compiled in) against
   `torch_arm.py --phase-timing` (the same syncs): wall time per phase.
2. Nsight Systems (`nsys profile --trace=cuda`) of the plain production binary and of
   the PyTorch arms, cut into updates at the optimizer kernel and into phases on the
   GPU timeline (`nsys_summary.py`); 15 steady updates each. Tracing adds 7-10% of
   wall time to NSL, 2% to eager and 12% to compiled PyTorch (host side); kernel
   durations are GPU-timed and unaffected.
3. `NSL_HOST_PROFILE=1` (the runtime's own host-side timers).

### Phases (synchronised wall time per update, seconds)

Coder-1B:

| arm (instrumented) | wall s/update | forward | backward | optimizer | rest |
|---|---|---|---|---|---|
| nsl_bf16 | 2.606 | 0.588 | 1.879 | 0.059 | 0.079 |
| nsl_tf32 | 3.365 | 0.829 | 2.381 | 0.057 | 0.098 |
| torch_bf16_compile_blocks | 1.369 | 0.334 | 0.969 | 0.064 | 0.002 |
| torch_bf16_compile_none | 1.128 | 0.335 | 0.727 | 0.063 | 0.002 |
| torch_bf16_compile_sel | 1.167 | 0.331 | 0.773 | 0.063 | 0.001 |
| torch_bf16_eager_sel | 1.571 | 0.410 | 1.096 | 0.064 | 0.001 |
| torch_tf32_eager_blocks | 2.754 | 0.670 | 2.019 | 0.063 | 0.001 |
| torch_tf32_eager_sel | 2.419 | 0.671 | 1.684 | 0.063 | 0.001 |

Coder-500M:

| arm (instrumented) | wall s/update | forward | backward | optimizer | rest |
|---|---|---|---|---|---|
| nsl_bf16 | 1.498 | 0.344 | 1.041 | 0.028 | 0.085 |
| nsl_tf32 | 1.988 | 0.505 | 1.336 | 0.029 | 0.118 |
| torch_bf16_compile_sel | 0.727 | 0.190 | 0.504 | 0.031 | 0.002 |
| torch_bf16_eager_sel | 0.995 | 0.279 | 0.685 | 0.030 | 0.002 |
| torch_tf32_eager_sel | 1.298 | 0.362 | 0.903 | 0.030 | 0.002 |

"Forward" includes the loss; "backward" includes the recompute; "optimizer" includes
gradient clipping; "rest" is the update's wall time outside the three (NSL: the data
loader, the per-micro-batch cache drain, scope cleanup). The syncs cost NSL 1-3% and
PyTorch 7-9% against the untimed runs, so compare phases with phases, not with the
throughput tables. Per-phase allocator peaks for PyTorch (reset before each phase)
are in the phase JSONs: compiled bf16 peaks at 20.0 GiB in the forward and 20.58 GiB
in the backward with selective checkpointing, 17.1 and 20.58 GiB with full recompute.

The backward is where most of the difference is. At 1B and TF32 NSL spends 0.70 s
more than eager PyTorch in it per update, 0.16 s more in the forward, 0.10 s more
outside the three phases and the same in the optimizer. Against compiled bf16, NSL's
bf16 posture spends 1.11 s more in the backward, 0.26 s in the forward and 0.08 s
outside. At 500M and TF32: +0.43 s, +0.14 s and +0.12 s.

### Kernels (Nsight Systems, ms per update)

| | NSL TF32 | PyTorch TF32 eager | NSL bf16 | PyTorch bf16 eager | PyTorch bf16 compile |
|---|---|---|---|---|---|
| wall (traced) | 3,575 | 2,300 | 2,681 | 1,477 | 1,223 |
| GPU busy | 3,445 (96.3%) | 2,291 (99.6%) | 2,562 (95.6%) | 1,469 (99.5%) | 1,217 (99.5%) |
| GPU idle | 131 | 9 | 119 | 8 | 6 |
| GEMM | 2,110 | 1,520 | 1,013 | 725 | 792 |
| attention | 673 | 74 | 665 | 75 | 81 |
| optimizer (+clip) | 51 | 58 | 51 | 58 | 65 |
| everything else | 611 | 638 | 832 | 612 | 278 |
| kernel launches | 8,043 | 10,261 | 9,819 | 12,205 | 4,649 |

On the GPU timeline the NSL TF32 update is 24 ms before the first forward, 890 ms of
forward (four micro-batches), 2,609 ms of backward and 52 ms of optimizer; the
recompute inside the backward (norms, RoPE, SwiGLU activation, the cos/sin lookups)
is 72 ms. PyTorch eager TF32: 646 / 1,596 / 58 ms, recompute 55 ms.

Top NSL kernels at TF32 (ms per update, launches per update):

| kernel | ms | launches | what it is |
|---|---|---|---|
| `cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4` | 707 | 432 | backward GEMMs (dgrad / wgrad) |
| `flash_attn_bwd_main_c1_q32_kv32_w4` | 552 | 64 | attention backward, 8.6 ms per call |
| `cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4` | 500 | 384 | backward GEMMs |
| `cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4` | 411 | 432 | q, k, v, o, down projections and the LM-head chunks |
| `cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4` | 264 | 128 | gate and up projections |
| `nsl_mul_f32` | 140 | 964 | RoPE, SwiGLU, scaling |
| `nvjet_sm120_sss_tf32_mma_256x128x32...` | 122 | 64 | backward GEMM |
| `flash_attn_p0_r0_hs_g1_c1_t0_q64_kv64_v2_mma` | 113 | 128 | attention forward, one launch per batch row |
| `nvjet_sm120_sss_tf32_mma_64x176x32...` | 107 | 96 | LM-head GEMMs |
| `nsl_scopy_run4_f32` | 77 | 2,048 | strided copies (head transposes, the GQA K/V expansion) |
| `nsl_fase_fused_adamw_multi_f32` | 51 | 1 | AdamW, all 146 parameters in one launch |

The same layer, GEMM by GEMM (forward, µs per call, one micro-batch of 4,096 tokens):

| projection | NSL TF32 | PyTorch TF32 |
|---|---|---|
| q (2048 -> 2048) | 613 | 463 |
| k, v (2048 -> 512) | 177, 175 | 129, 130 |
| o (2048 -> 2048) | 801 | 489 |
| gate, up (2048 -> 8192) | 1,842, 1,846 | 1,841, 1,833 |
| down (8192 -> 2048) | 2,817 | 2,314 |
| attention forward | 881 + 874 (one launch per batch row) | 302 |

In the backward, one layer's FFN GEMMs take 16.6 ms in NSL (64x64 tiles, ~2.95 ms each)
against 11.8 ms in PyTorch (256x128 tiles, ~1.95 ms each). Where cuBLAS gives NSL a
large-tile kernel (gate and up: 128x128 against PyTorch's 256x128) the two take the
same 1.84 ms: the gap is the kernel the heuristic returns, not NSL's operands or
layouts being slower in themselves.

### Host side

`NSL_HOST_PROFILE=1` attributes ~800 ms of host time per micro-batch, 98% of it inside
134 synchronous `cuMemcpyHtoD` calls (1.1 MiB in all) and 2% in ~1,700 kernel
launches at 9 µs each. A synchronous host-to-device copy from pageable memory waits
for the device, so each of those 134 copies is a device synchronisation: the host
cannot run more than one copy ahead of the GPU, and the GPU idles while the host
issues the next kernels. The trace agrees: 536 `cuMemcpyHtoD_v2` per update and
131 ms of GPU idle. 128 of the 134 copies per micro-batch are 8,192 bytes, the size of
RoPE's position vector (2,048 f32): `RotaryEmbedding.forward` builds it with `arange`
on the host on every call (`stdlib/nsl/nn/rope.nsl`), and 128 = 16 layers x (q, k) x
(cos, sin lookup) x (forward, recompute). The per-micro-batch
`nsl_gpu_drain_cache` adds a `cuCtxSynchronize` and returns ~9.6 GB of transient
pool to the driver, which the next micro-batch allocates again (~95 `cuMemFree` and
~95 `cuMemAlloc` per micro-batch).

PyTorch eager keeps its GPU 99.6% busy with 10,261 launches per update: launch
overhead is not what separates the two.

## Five places to make NSL faster or smaller

Gains are for Coder-1B at the production shape unless stated; they do not add up
exactly, because removing one bottleneck changes the clock and the overlap of the rest.

1. **The attention kernels.** Measured: 673 ms per update at TF32 and 665 ms at bf16
   against PyTorch's 74-81 ms; the backward kernel is 8.6 ms per call against
   FlashAttention-2's 0.77 ms on the same card and shape, the forward 1.76 ms (two
   launches, one per batch row) against 0.30 ms. Matching FlashAttention-2 would take
   *~590 ms off a 3.3 s TF32 update (+21% tok/s, est.) and off a 2.43 s bf16 update
   (+32%, est.)*. Attending over the 8 KV heads directly instead of materialising
   them 4x (`k5.expand(...).contiguous()` in `stdlib/nsl/nn/gqa.nsl`) also removes
   the expansion copies and their saved copies (*~0.5-1 GiB at 1B, est.*). Confidence
   in the size of the prize: high (both kernels measured on this card). In reaching
   it: medium, it is kernel engineering (tile sizes, a sequence-parallel backward,
   16-bit inputs end to end).
2. **TF32 GEMM kernel selection.** Measured: 2,110 ms against 1,520 ms for identical
   GEMM FLOPs; NSL's 8192^3 probe reaches 71 TFLOPS against PyTorch's 80. cuBLAS 13.8's
   `sgemm` heuristic returns 64x64-tile kernels for most of the backward and for the
   q/k/v/o/down projections; where it returns 128x128 (gate, up) NSL matches PyTorch.
   Asking cublasLt for several algorithms per shape and timing them once (NSL already
   does this for bf16, `NSL_MATMUL_BF16_LT_TUNE`) is the obvious route. *Gain ~0.5-0.6 s
   per TF32 update (+18-22%, est.)*; confidence medium-high (that better kernels exist
   for these shapes is measured; that cublasLt's candidate list contains them on
   cuBLAS 13.8 is not). At bf16, NSL's GEMMs take 1,013 ms against 725-792 ms, plus
   196 ms of `nsl_cast_f32_to_bf16` on activations before each GEMM.
3. **Fusing the memory-bound work.** Measured: everything that is not a GEMM, attention
   or the optimizer costs NSL 611 ms (TF32) and 832 ms (bf16) per update, against
   278 ms under `torch.compile` (bf16). Visible pieces: RoPE as ~10 kernels per q and
   per k (copies, two table lookups, mul, rotate, mul, add) at ~0.7 ms per layer and
   micro-batch; SwiGLU's `silu` and `mul` unfused (1.1 ms per layer and micro-batch
   forward, again in the recompute); 2,048 strided-copy launches per update; in bf16
   the activation casts. *A fused RoPE, a fused SwiGLU, residual-add + RMSNorm fusion
   and dropping the K/V expansion would remove ~300 ms per update at TF32 (+10%, est.);
   at bf16 with 16-bit activations, up to ~550 ms (+29%, est.)*. Confidence medium.
4. **16-bit activations (memory first, then speed).** Measured: NSL's selective
   checkpointing keeps 5.4 GiB of f32 matmul outputs per micro-batch at 1B (the
   allocator's `matmul_f32` context at peak); `NSL_MATMUL_BF16=1` adds a 1.9 GiB weight
   cast cache and keeps every activation in f32, so NSL's bf16 posture peaks at
   26.5 GiB against PyTorch's 20.6 (compile) and 23.2 (eager). `--checkpoint-compress
   bf16` already stores the saved matmul outputs in bf16 and takes the peak to
   21.24 GiB, but at 1,680 tok/s (3.0x slower; item in the defects below). *A bf16
   activation path (bf16 GEMM outputs, bf16 storage, f32 norms and loss, as autocast
   does) would save ~3-5 GiB at 1B and the 196 ms of casts per update (est.)*;
   confidence medium.
5. **Host synchronisation.** Measured: 131 ms of GPU idle per TF32 update (3.7%) from
   536 synchronous host-to-device copies, and the drain worth 1-2% (1B) and 5% (500M)
   on its own. Keeping small host-built tensors (RoPE's positions above all) on
   the device or staging them through pinned memory with `cuMemcpyHtoDAsync`, and
   dropping the per-micro-batch drain (or doing it once per optimizer step), would
   recover *~4% at 1B and ~5-8% at 500M (est.)*; confidence high, and it is cheap.

## Defects and oddities found (not fixed here)

- **The flash-attention forward launches once per batch row**
  (`nsl_sdpa_fused_forward`, `crates/nsl-runtime/src/flash_attention.rs`): a
  workaround for the packed-sequence segment staging, applied to unpacked batches too.
  The total work is right; each launch fills half the grid (1,024 CTAs at 1B), and the
  pair takes 1.76 ms against FlashAttention-2's 0.30 ms for the whole batch.
- **`--checkpoint-compress bf16` triples the step time at 1B**: 9.6 s against 3.3 s
  per update (paired ratio 0.33 at TF32, 0.27 at bf16) for 3.4 GiB less allocator
  peak, i.e. 1.6 s more per micro-batch. Narrowing 5.4 GiB of saved activations to
  bf16 and widening them back is ~16 GiB of memory traffic, tens of milliseconds at
  this card's bandwidth. Not traced; the cause is open.
- **`NSL_MATMUL_BF16` is read at build time on `main`** (`MatmulConfig::
  with_env_fallback`; the runtime ignores the variable by design, see
  `resolve_math_mode_from`). `models/benchmarks/posture_bench_1b.py`'s P4 arm runs
  P3's binary with the variable set at run time and says the setting is "a RUNTIME
  env, same artifact, different run": on today's toolchain that arm silently measures
  TF32. `pretrain_prod.nsl`'s `export NSL_MATMUL_BF16=1; nsl run ...` still works,
  because `nsl run` compiles in the same process. This harness builds the bf16 arms
  with the variable set and refuses a run whose banner does not say BF16.
- **The per-micro-batch drain** (`nsl_gpu_drain_cache` after every micro-batch,
  `stmt_train/driver.rs`) returns ~9.6 GB to the driver and re-allocates it each
  micro-batch at 1B; its own comment calls it a workaround.

## Reproduce

Raw results (one JSON per run, logs, traces, trace summaries) are in
`/mnt/extrastorage/nsl-audit/research/work/PERF/runs/` on the measuring machine; this
document quotes `perfcheck.py report` on that directory. Commands, from the repository
root, each GPU one queued separately on this machine
(`/mnt/extrastorage/nsl-audit/bin/nslq gpu -- ...`):

```bash
cargo build --release --bin nsl --features cuda
uv venv --python 3.12 VENV
uv pip install --python VENV/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python VENV/bin/python numpy tokenizers
P="VENV/bin/python tools/perfcheck/perfcheck.py"; OUT=/a/real/disk/perf
CORE=nsl_tf32,nsl_bf16,torch_tf32_eager_sel,torch_bf16_eager_sel,torch_bf16_compile_sel
EXTRA=nsl_tf32_nodrain,nsl_bf16_nodrain,nsl_tf32_blocks,torch_tf32_eager_blocks,torch_bf16_compile_blocks,torch_bf16_compile_none

$P roofline --out $OUT
for s in 1b 500m; do
  $P build --size $s --arms nsl_tf32,nsl_bf16,nsl_tf32_blocks,nsl_tf32_compress,nsl_bf16_compress --phase --out $OUT
  for r in 0 1 2; do $P run --size $s --arms $CORE --round $r --out $OUT
                     $P run --size $s --arms $EXTRA --round $r --out $OUT; done
  $P phases --size $s --arms $CORE --out $OUT
done
$P run --size 1b --arms nsl_tf32_compress,nsl_bf16_compress --round 0 --out $OUT   # rounds 0-2
for a in nsl_tf32 nsl_bf16 torch_tf32_eager_sel torch_bf16_eager_sel torch_bf16_compile_sel; do
  $P nsys --size 1b --arm $a --out $OUT; done
$P diag --size 1b --arm nsl_tf32 --env NSL_HOST_PROFILE=1 --tag hostprof --out $OUT
$P report --out $OUT
python3 tools/perfcheck/test_perfcheck.py   # the harness's own arithmetic, no GPU
```

Trace summaries (`nsys_summary.py TRACE --boundary B --fwd-start F --bwd-start W
[--opt-start O]`): NSL `B=fase_fused_adamw F=nsl_embedding_i32idx
W=lce_chunk_dlogits`; PyTorch eager `B=FusedAdam O=LpNormFunctor
F=vectorized_gather_kernel W=nll_loss_backward`; PyTorch compiled `B=FusedAdam
O=LpNormFunctor F=fused__to_copy_add_div_index_mean W=log_softmax_backward_data`.
