# NSL against PyTorch on one GPU: speed and memory (2026-11-14)

The first measurement of NSL against PyTorch on the same model, the same tokens per
step and the same card. Every number in this document was measured on this machine
by `tools/perfcheck/` unless it is marked as an estimate: **estimates are written in
italics with "(est.)"**; everything else is a measurement.

*Follow-up, same day (`perf/host-gemm`): GEMM kernel selection, the step-boundary drain
and the RoPE uploads were fixed, and `--checkpoint-compress` no longer triples the step.
NSL is 1.17x faster at 1B TF32, 1.10x at 1B bf16 (the recipe) and 1.23x at 500M TF32
than the numbers below; see "Follow-up" near the end. The rest of this document is the
baseline as measured.*

## Summary

**Coder-1B's production recipe trains in bf16.** `models/coder1b/pretrain_prod.nsl`
runs `export NSL_MATMUL_BF16=1` with `--source-ad --checkpoint-blocks
--checkpoint-selective --fuse-rmsnorm-backward --fuse-wgrad-accum`; that is the
`nsl_bf16` arm here. The `nsl_tf32` arm (same flags, f32 weights and activations, TF32
matmuls) is not the 1B recipe: it is the f32 posture the recipe ran until August and
the arm that matches PyTorch's TF32 precision policy. Coder-500M's recipe sets no
`NSL_MATMUL_BF16` and runs TF32 (with only `--source-ad --checkpoint-blocks`; the 500M
NSL rows use the 1B flag set at the 500M shape).

- **Each framework's production posture at 1B** (NSL bf16 against PyTorch bf16 autocast
  plus `torch.compile`, both with selective checkpointing): PyTorch is **2.23x
  faster**; NSL runs at **27.5% MFU** against 61.1%, and peaks **5.5 GiB higher** in
  the driver-reported process memory (27.75 against 22.25 GiB; 5.9 GiB in allocator
  peak, 26.53 against 20.58).
- **Same precision policy** (f32 weights and activations, TF32 matmuls, the same
  selective checkpointing): PyTorch eager is **1.44x faster at 1B and 1.53x at 500M**,
  at the same memory (driver peak 25.83 against 25.55 GiB at 1B).
- **At 500M**, the production recipe's TF32 against PyTorch's compiled bf16 is 2.77x;
  NSL bf16 against compiled bf16 is 2.14x.

Ratios are paired within a round, then the median over rounds.

| Coder-1B, 16,384 tokens per update | tok/s | MFU | allocator peak | driver peak |
|---|---|---|---|---|
| **NSL production posture** (`NSL_MATMUL_BF16=1`) | 6,747 | 27.5% | 26.53 GiB | 27.75 GiB |
| PyTorch bf16 autocast + `torch.compile`, same checkpointing | 14,981 | 61.1% | 20.58 GiB | 22.25 GiB |
| NSL f32 / TF32 (not the 1B recipe) | 4,916 | 42.2% | 24.63 GiB | 25.83 GiB |
| PyTorch eager, TF32, same checkpointing | 7,259 | 62.3% | 24.87 GiB | 25.55 GiB |

(Medians of four rounds; MFU against the TF32 and bf16 rooflines measured below;
"allocator peak" and "driver peak" are defined under Method.)

NSL is not host-bound: the GPU is busy 96% of an update. The time goes to kernels
that are slower than PyTorch's for the same work. At TF32 three places account for
the whole gap at 1B (per optimizer update, from Nsight Systems):

- **attention**: 673 ms against 74 ms. NSL's flash-attention backward takes 8.6 ms per
  layer and micro-batch where PyTorch's FlashAttention-2 takes 0.77 ms (11x), and its
  forward 1.76 ms against 0.29 ms (6x);
- **GEMMs**: 2,110 ms against 1,520 ms for the same FLOPs. cuBLAS 13.8's `sgemm`
  heuristic picks 64x64-tile sm80 kernels for most of NSL's backward GEMMs and five of
  the seven forward projections; PyTorch's cuBLAS 12.8 picks 128x128 to 256x128 tiles;
  every projection is 15-63% slower in NSL;
- **host synchronisation**: 131 ms of GPU idle against 9 ms, from 536 synchronous
  host-to-device copies per update and a per-micro-batch release of the allocator's
  cache to the driver.

In the bf16 production posture the same three plus the memory-bound work separate
NSL from compiled PyTorch: attention +584 ms, everything that is not a GEMM, attention
or the optimizer +554 ms (`torch.compile` fuses it and keeps it in bf16), GEMMs +221
ms, idle +113 ms, out of a 1,458 ms difference per traced update.

The f32 posture reproduces the August record it comes from (5,053 tok/s at 25.2 GiB,
`POSTURE_1B_THROUGHPUT_2026_08_27.md`, arm P3): 4,678-5,068 tok/s over four rounds at
24.63 GiB on today's `main`. The bf16 posture that replaced it in the recipe was
6,568 tok/s there (arm P4); it is 6,190-6,761 here.

## Setup

| | |
|---|---|
| card | NVIDIA RTX PRO 4500 Blackwell (GB203, sm_120), 32 GB GDDR7, 200 W board power cap (every timed run sat at the cap: 199.9 W mean) |
| driver | 620.70.06 |
| NSL | `main` at 565567c5, `cargo build --release --features cuda`; links the system cuBLAS 13.8.0.4 (CUDA 13.4) |
| PyTorch | 2.11.0+cu128 (cuBLAS 12.8.4.1, Triton 3.6.0), Python 3.12, in `/mnt/extrastorage/nsl-audit/venv-cuda` |
| harness | `tools/perfcheck/` on branch `perf/baseline` |
| models | Coder-500M (d 1280, 24 layers, 20/10 heads, d_ff 3520, dropout 0.1) and Coder-1B (d 2048, 16 layers, 32/8 heads, d_ff 8192, dropout 0) from `models/coder*/config.nsl`; vocab 49,152, tied LM head |
| step shapes | the production recipes' (`PRETRAIN_BATCH_SIZE`, `MAX_SEQ_LEN`, `PRETRAIN_GRAD_ACCUMULATION` from `config.nsl`): 500M micro-batch 2 x 1024 tokens, accumulation 8; 1B 2 x 2048, accumulation 4. 16,384 tokens per optimizer update in both |
| data | synthetic: a seeded uniform draw over the vocabulary, the same file and the same windows for both frameworks. The step does the same work whatever the ids are |

### Method

- **NSL arms** run a program `perfcheck.py build` generates from the production recipe
  (`pretrain_prod.nsl`'s train block: AdamW, warmup_cosine, `grad_clip=1.0`,
  `@fused_lm_ce`, `schedule_unit="optimizer_step"`) without checkpoint writes or
  validation, built once with `nsl build` and run as a binary. The build is refused if
  the fused LM cross-entropy did not engage, and each run is refused unless the runtime's
  own `cuBLAS math mode:` banner names the arm's precision.
- **PyTorch arms** run `torch_arm.py`, one fresh process per run.
- **Timing.** The token file holds exactly 24 optimizer updates. Each update's loss
  reaches the host once (NSL prints it, PyTorch calls `loss.item()`) and is
  timestamped when it arrives. Neither side is otherwise free of synchronisation:
  PyTorch copies each micro-batch's ids from pageable host memory
  (`batch_from_stream`'s `.to("cuda")`), a stream synchronisation per micro-batch (4
  `cudaStreamSynchronize` per 1B update, plus the loss, in the trace); NSL makes 134
  synchronous host-to-device copies per micro-batch (see "Host side"). The first 9 updates are not timed (cuBLAS
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
  The two allocator peaks do not count quite the same bytes. NSL's `gpu_peak_bytes`
  counts its pooled blocks plus the direct and asynchronous device allocations it
  routes through the same accounting (`record_external_alloc`: attention workspace,
  slabs); it does not count the workspace cuBLAS allocates for itself (NSL never
  calls `cublasSetWorkspace`) nor the 64 MiB workspace of the opt-in cublasLt path
  (`NSL_MATMUL_BF16_LT=1`, a raw `cuMemAlloc` in `cuda/lt_matmul.rs`; not used here). PyTorch gives cuBLAS a workspace from
  its caching allocator, so `max_memory_allocated` includes it: the first GEMM of a
  process retains 8,519,680 bytes (the default `CUBLAS_WORKSPACE_CONFIG`, 2 x 4 MiB +
  8 x 16 KiB; `perfcheck.py workspace`). Both differences are under 0.1 GiB; the
  driver peak counts everything on both sides.
- **Untested caveats.** (a) Two `nvidia-smi` pollers run during every timed window
  (per-process memory every 100 ms, clocks every 200 ms). Whether their driver
  queries slow either framework was not measured; NSL, which makes ~380 `cuMemAlloc`
  and ~380 `cuMemFree` calls per 1B update through its per-micro-batch drain, is the
  more exposed. (b) The bf16 roofline ran with PyTorch's default
  `allow_bf16_reduced_precision_reduction=True` (split-K reductions may round to
  bf16); whether the bf16 rate changes with it off was not measured. The TF32 and
  f32 rows are unaffected.

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
`nsl_bf16` is the 1B production posture (the 1B recipe's flags, built and run with
`NSL_MATMUL_BF16=1`), `nsl_tf32` the same flags in f32 with TF32 matmuls (the 500M
recipe's precision, not the 1B recipe), `_nodrain` adds `NSL_SKIP_GPU_DRAIN=1`, `_blocks` drops
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
| PyTorch compiled bf16 / NSL bf16 (1B: each one's production posture) | 2.245 2.222 2.215 2.240 | **2.23** | 2.167 2.025 2.132 2.148 | **2.14** |
| PyTorch compiled bf16 / NSL TF32 (500M: each one's production precision) | 2.989 3.129 2.969 2.964 | 2.98 | 2.795 2.623 2.772 2.773 | **2.77** |
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

### Per round

tok/s of every run, with its deviation from the arm's median over all its rounds.

Coder-1B:

| arm | tok/s per round (deviation from the arm's median) |
|---|---|
| nsl_tf32 | r0:5,068 (+3.1%) r1:4,801 (-2.3%) r2:5,032 (+2.3%) r3:4,678 (-4.8%) |
| nsl_tf32_nodrain | r0:5,152 (+0.1%) r1:5,139 (-0.1%) r2:5,147 (+0.0%) |
| nsl_tf32_blocks | r0:4,458 (+0.0%) r1:4,448 (-0.2%) r2:4,456 (+0.0%) |
| nsl_tf32_compress | r0:1,680 (+0.0%) r1:1,685 (+0.3%) r2:1,604 (-4.5%) |
| nsl_bf16 | r0:6,749 (+0.0%) r1:6,761 (+0.2%) r2:6,744 (-0.0%) r3:6,190 (-8.3%) |
| nsl_bf16_nodrain | r0:6,775 (-1.0%) r1:6,843 (+0.0%) r2:6,931 (+1.3%) |
| nsl_bf16_compress | r0:1,812 (+0.0%) r1:1,840 (+1.5%) r2:1,708 (-5.7%) |
| torch_tf32_eager_sel | r0:7,253 (-0.1%) r1:7,278 (+0.3%) r2:7,265 (+0.1%) r3:6,719 (-7.4%) |
| torch_tf32_eager_blocks | r0:6,463 (+1.5%) r1:6,363 (-0.1%) r2:6,368 (+0.0%) |
| torch_bf16_eager_sel | r0:11,474 (+1.6%) r1:11,250 (-0.4%) r2:11,332 (+0.4%) r3:10,468 (-7.3%) |
| torch_bf16_compile_sel | r0:15,150 (+1.1%) r1:15,022 (+0.3%) r2:14,940 (-0.3%) r3:13,867 (-7.4%) |
| torch_bf16_compile_blocks | r0:13,107 (+1.2%) r1:12,954 (+0.0%) r2:12,938 (-0.1%) |
| torch_bf16_compile_none | r0:15,778 (+0.9%) r1:15,612 (-0.2%) r2:15,637 (+0.0%) |

Coder-500M:

| arm | tok/s per round (deviation from the arm's median) |
|---|---|
| nsl_tf32 | r0:8,971 (+0.8%) r1:8,897 (-0.1%) r2:8,906 (+0.1%) r3:8,329 (-6.4%) |
| nsl_tf32_nodrain | r0:9,408 (+0.1%) r1:9,401 (+0.0%) r2:9,355 (-0.5%) |
| nsl_tf32_blocks | r0:8,026 (+0.3%) r1:8,000 (+0.0%) r2:7,756 (-3.0%) |
| nsl_bf16 | r0:11,572 (+0.2%) r1:11,525 (-0.2%) r2:11,578 (+0.3%) r3:10,749 (-6.9%) |
| nsl_bf16_nodrain | r0:12,160 (+0.3%) r1:12,129 (+0.0%) r2:12,066 (-0.5%) |
| torch_tf32_eager_sel | r0:13,750 (+1.0%) r1:13,705 (+0.7%) r2:13,524 (-0.7%) r3:12,537 (-7.9%) |
| torch_tf32_eager_blocks | r0:11,675 (+0.0%) r1:11,799 (+1.1%) r2:11,638 (-0.3%) |
| torch_bf16_eager_sel | r0:18,092 (+1.7%) r1:18,181 (+2.2%) r2:17,476 (-1.7%) r3:16,239 (-8.7%) |
| torch_bf16_compile_sel | r0:25,075 (+4.4%) r1:23,334 (-2.8%) r2:24,686 (+2.8%) r3:23,094 (-3.8%) |
| torch_bf16_compile_blocks | r0:21,207 (+0.6%) r1:21,086 (+0.0%) r2:20,964 (-0.6%) |
| torch_bf16_compile_none | r0:26,357 (+0.7%) r1:26,161 (+0.0%) r2:26,110 (-0.2%) |

### Reading the tables

- **Production posture against production posture** at 1B (`nsl_bf16` against
  `torch_bf16_compile_sel`): PyTorch is 2.23x faster and 5.5 GiB smaller in driver
  peak. At 500M the NSL recipe is TF32: compiled bf16 PyTorch is 2.77x faster than it.
- **Same precision policy, same checkpointing** (`nsl_tf32` against
  `torch_tf32_eager_sel`): PyTorch is 1.44x faster at 1B and 1.53x at 500M; the
  allocator peaks are within 0.3 GiB at 1B and 1.3 GiB at 500M (NSL higher at 500M),
  the driver peaks within 0.3 and 2.0 GiB.
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
- **Spread** (the report's per-round table). Rounds 0-2 agree within about 2% for
  every arm but six single rounds that ran 3-6% slow: 1B `nsl_tf32` round 1 (at load
  average 12, the card no hotter than in the other rounds), both 1B compress arms in
  round 2, and three 500M arms in rounds 1-2. The fourth round of the core
  arms (run to repeat the disturbed one, right after the last compress round) came out
  6.5-10% slower on every arm at both sizes at once, with SM clocks 2-4% higher, the
  card in P1 and no other compute process on it; the cause was not found. The medians
  include every round. The paired ratios cancel machine-wide shifts like that one:
  each lies within 5% of its median, most within 1-2%. The 500M PyTorch eager arms
  have the largest within-run CV (2-4%): their 15 timed updates last 13-18 s.
- **Sanity.** On these tokens both frameworks start at loss 11.20-11.23 (1B) and
  11.05-11.07 (500M) and fall by 0.20-0.26 and 0.04-0.09 over the 24 updates; NSL's
  TF32 and bf16 arms agree to 1e-4 at the last update. A run is refused (and the
  report re-checks stored runs) unless it has one loss per update, all finite, and
  the last below the first; every run in these tables passes.

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

## Where NSL's step time goes (Coder-1B)

Both NSL precisions are traced: the bf16 columns are the 1B production posture, the
TF32 columns the matched-precision comparison.

Three instruments, each in its own run so none of them is in the timed numbers:

1. `NSL_PHASE_TIMING=1` builds (device syncs around each phase, compiled in) against
   `torch_arm.py --phase-timing` (the same syncs): wall time per phase.
2. Nsight Systems (`nsys profile --trace=cuda`) of the un-instrumented NSL binaries
   and of the PyTorch arms, cut into updates at the optimizer kernel and into phases on the
   GPU timeline (`nsys_summary.py`); 15 steady updates each. Tracing adds 7-10% of
   wall time to NSL TF32, 10% to NSL bf16, 2% to eager and 12% to compiled PyTorch
   (host side); kernel durations are GPU-timed and unaffected.
3. `NSL_HOST_PROFILE=1` (the runtime's own host-side timers).

### Phases (synchronised wall time per update, seconds)

Coder-1B:

| arm (instrumented) | wall s/update | vs timed median | forward | backward | optimizer | rest | alloc peak fwd / bwd / opt GiB |
|---|---|---|---|---|---|---|---|
| nsl_bf16 | 2.606 | +7.3% | 0.588 | 1.879 | 0.059 | 0.079 | - |
| nsl_tf32 | 3.365 | +0.6% | 0.829 | 2.381 | 0.057 | 0.098 | - |
| torch_bf16_compile_blocks | 1.369 | +8.1% | 0.334 | 0.969 | 0.064 | 0.002 | 17.10 / 20.58 / 16.02 |
| torch_bf16_compile_none | 1.128 | +7.7% | 0.335 | 0.727 | 0.063 | 0.002 | 23.57 / 23.76 / 16.02 |
| torch_bf16_compile_sel | 1.167 | +6.9% | 0.331 | 0.773 | 0.063 | 0.001 | 20.00 / 20.58 / 16.02 |
| torch_bf16_eager_sel | 1.571 | +8.7% | 0.410 | 1.096 | 0.064 | 0.001 | 23.23 / 21.79 / 16.02 |
| torch_tf32_eager_blocks | 2.754 | +7.2% | 0.670 | 2.019 | 0.063 | 0.001 | 18.11 / 18.86 / 16.02 |
| torch_tf32_eager_sel | 2.419 | +7.3% | 0.671 | 1.684 | 0.063 | 0.001 | 24.12 / 24.87 / 16.02 |

Coder-500M:

| arm (instrumented) | wall s/update | vs timed median | forward | backward | optimizer | rest | alloc peak fwd / bwd / opt GiB |
|---|---|---|---|---|---|---|---|
| nsl_bf16 | 1.498 | +5.6% | 0.344 | 1.041 | 0.028 | 0.085 | - |
| nsl_tf32 | 1.988 | +8.1% | 0.505 | 1.336 | 0.029 | 0.118 | - |
| torch_bf16_compile_sel | 0.727 | +7.1% | 0.190 | 0.504 | 0.031 | 0.002 | 9.42 / 10.01 / 7.75 |
| torch_bf16_eager_sel | 0.995 | +9.7% | 0.279 | 0.685 | 0.030 | 0.002 | 10.94 / 10.28 / 7.70 |
| torch_tf32_eager_sel | 1.298 | +8.6% | 0.362 | 0.903 | 0.030 | 0.002 | 11.07 / 11.45 / 7.66 |

"Forward" includes the loss; "backward" includes the recompute; "optimizer" includes
gradient clipping; "rest" is the update's wall time outside the three (NSL: the data
loader, the per-micro-batch cache drain, scope cleanup). "vs timed median" is the
instrumented wall time against the arm's un-instrumented median (all rounds): the
syncs cost NSL +0.6% to +8.1% and PyTorch +6.9% to +9.7% (NSL TF32 at 1B is +3.4%
against rounds 0-2 alone, which exclude the slow fourth round), so compare phases
with phases, not with the throughput tables. The allocator peaks per phase are
PyTorch's, reset before each phase of the last update; NSL has no equivalent hook.

The backward is where most of the difference is. At 1B and TF32 NSL spends 0.70 s
more than eager PyTorch in it per update, 0.16 s more in the forward, 0.10 s more
outside the three phases and the same in the optimizer. Against compiled bf16, NSL's
bf16 posture spends 1.11 s more in the backward, 0.26 s in the forward and 0.08 s
outside. At 500M and TF32: +0.43 s, +0.14 s and +0.12 s.

### Kernels (Nsight Systems, ms per update)

| | NSL TF32 | PyTorch TF32 eager | NSL bf16 (1B production) | PyTorch bf16 eager | PyTorch bf16 compile |
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

Layer by layer (`nsys_summary.py --layers 16`: medians over every layer and
micro-batch of the 15 traced updates, µs per call, one micro-batch of 4,096 tokens):

| | NSL TF32 | PyTorch TF32 eager | NSL bf16 (1B production) | PyTorch bf16 eager | PyTorch bf16 compile |
|---|---|---|---|---|---|
| q (2048 -> 2048) | 632 | 486 | 291 | 262 | 253 |
| k (2048 -> 512) | 172 | 131 | 175 | 69 | 64 |
| v (2048 -> 512) | 169 | 130 | 175 | 67 | 66 |
| o (2048 -> 2048) | 792 | 487 | 292 | 261 | 256 |
| gate (2048 -> 8192) | 2,121 | 1,844 | 870 | 875 | 879 |
| up (2048 -> 8192) | 2,125 | 1,832 | 1,080 | 872 | 876 |
| down (8192 -> 2048) | 2,821 | 2,117 | 988 | 993 | 982 |
| attention forward, per layer | 1,760 | 289 | 1,758 | 287 | 313 |
| backward, all GEMMs of one layer | 20,265 | 14,434 | 9,869 | 6,801 | 7,706 |
| backward, the six FFN GEMMs of one layer | 16,633 | 11,852 | 7,753 | 5,485 | 6,172 |

At TF32 every projection is slower in NSL, by 15% (gate, up: cuBLAS gives NSL a
128x128 kernel where PyTorch gets 256x128) to 63% (o: a 64x64 kernel); the backward
layer's GEMMs take 1.40x as long. In bf16 the large projections match PyTorch's
(gate, down) and the small ones do not (k, v 2.6x), and the backward layer's GEMMs
take 1.45x eager PyTorch's. NSL's per-layer attention forward is the same in both
precisions (it runs on f16 operands either way) and 6x FlashAttention-2's.

### Host side

`NSL_HOST_PROFILE=1` (medians over the 96 micro-batches of a TF32 run) attributes
768 ms of host time per micro-batch: 750 ms (98%) inside 134 synchronous
`cuMemcpyHtoD` calls (1.1 MiB in all), 15 ms in 1,590 kernel launches (9.5 µs each)
and 2 ms in 1,447 device allocations. A synchronous host-to-device copy from pageable memory waits
for the device, so each of those 134 copies is a device synchronisation: the host
cannot run more than one copy ahead of the GPU, and the GPU idles while the host
issues the next kernels. The trace agrees: 536 `cuMemcpyHtoD_v2` per update and
131 ms of GPU idle (`--copies`: 512 of them per update, 128 per micro-batch, are
8,192 bytes; PyTorch makes 4 host-to-device copies per update, one 32 KiB batch of ids
per micro-batch). The 8,192-byte copies are the size of
RoPE's position vector (2,048 f32): `RotaryEmbedding.forward` builds it with `arange`
on the host on every call (`stdlib/nsl/nn/rope.nsl`), and 128 = 16 layers x (q, k) x
(cos, sin lookup) x (forward, recompute). The per-micro-batch
`nsl_gpu_drain_cache` adds a `cuCtxSynchronize` and returns ~9.6 GB of transient
pool to the driver, which the next micro-batch allocates again (380 `cuMemFree` and
380 `cuMemAlloc` per update in the trace, ~95 of each per micro-batch).

PyTorch eager keeps its GPU 99.6% busy with 10,261 launches per update: launch
overhead is not what separates the two.

## Five places to make NSL faster or smaller

Gains are for Coder-1B at the production shape unless stated. The 1B recipe trains in
bf16, so the bf16 estimate is the one that matters for it; the TF32 one is for the
matched-precision comparison and the 500M recipe. They do not add up exactly, because
removing one bottleneck changes the clock and the overlap of the rest.

1. **The attention kernels.** Measured: 673 ms per update at TF32 and 665 ms at bf16
   against PyTorch's 74-81 ms; the backward kernel is 8.6 ms per call against
   FlashAttention-2's 0.77 ms on the same card and shape, the forward 1.76 ms per layer
   (two launches, one per batch row) against 0.29 ms. Matching FlashAttention-2 would
   take *~590 ms off the 2.43 s bf16 production update (+32% tok/s, est.) and off a
   3.3 s TF32 update (+21%, est.)*. Attending over the 8 KV heads directly instead of materialising
   them 4x (`k5.expand(...).contiguous()` in `stdlib/nsl/nn/gqa.nsl`) also removes
   the expansion copies and their saved copies (*~0.5-1 GiB at 1B, est.*). Confidence
   in the size of the prize: high (both kernels measured on this card). In reaching
   it: medium, it is kernel engineering (tile sizes, a sequence-parallel backward,
   16-bit inputs end to end).
2. **GEMM kernel selection.** Measured, bf16 (production): NSL's GEMMs take 1,013 ms
   per update against 725 (eager) and 792 (compiled) ms; the large projections match
   PyTorch's, k and v are 2.6x slower and the backward layer's GEMMs 1.45x; another
   196 ms go to `nsl_cast_f32_to_bf16` on activations before each GEMM. Measured, TF32:
   2,110 ms against 1,520 ms for identical GEMM FLOPs; NSL's 8192^3 probe reaches
   71 TFLOPS against PyTorch's 80. cuBLAS 13.8's `sgemm` heuristic returns 64x64-tile
   kernels for most of the backward and for the q/k/v/o/down projections, and even gate
   and up, where it returns 128x128, are 15% slower than PyTorch's 256x128. Asking
   cublasLt for several algorithms per shape and timing them once is the obvious route.
   NSL has that for bf16 behind `NSL_MATMUL_BF16_LT=1` (off by default because it
   changes bits; not measured here: every bf16 arm ran the default `cublasGemmEx`
   path), and nothing like it for TF32. *Gain ~0.2-0.3 s per bf16 production update
   (+10-14%, est.), ~0.5-0.6 s per TF32 update (+18-22%, est.)*; confidence medium-high (that faster kernels exist for
   these shapes on this card is measured; that cublasLt's candidate list on cuBLAS 13.8
   contains them is not).
3. **Fusing the memory-bound work.** Measured: everything that is not a GEMM, attention
   or the optimizer costs NSL 611 ms (TF32) and 832 ms (bf16) per update, against
   278 ms under `torch.compile` (bf16). Visible pieces: RoPE as ~10 kernels per q and
   per k (copies, two table lookups, mul, rotate, mul, add) at ~0.7 ms per layer and
   micro-batch; SwiGLU's `silu` and `mul` unfused (`nsl_silu_f32` alone 45 ms per update,
   0.35 ms per call, plus a ~0.5 ms `nsl_mul_f32` on the same tensor, forward and
   recompute); 2,048 strided-copy launches per update; in bf16
   the activation casts. *A fused RoPE, a fused SwiGLU, residual-add + RMSNorm fusion
   and dropping the K/V expansion would remove ~300 ms per update at TF32 (+10%, est.);
   at bf16 with 16-bit activations, up to ~550 ms (+29%, est.)*. Confidence medium.
4. **16-bit activations (memory first, then speed).** Measured: NSL's selective
   checkpointing keeps 5.4 GiB of f32 matmul outputs per micro-batch at 1B (the
   allocator's `matmul_f32` context at peak); `NSL_MATMUL_BF16=1` adds a 1.9 GiB weight
   cast cache and keeps every activation in f32, so NSL's bf16 production posture peaks
   at 26.5 GiB allocator / 27.8 GiB driver against PyTorch's 20.6 / 22.3 (compile) and
   23.2 / 24.4 (eager). `--checkpoint-compress
   bf16` already stores the saved matmul outputs in bf16 and takes the peak to
   21.24 GiB, but at 1,680 tok/s (3.0x slower; item in the defects below). *A bf16
   activation path (bf16 GEMM outputs, bf16 storage, f32 norms and loss, as autocast
   does) would save ~3-5 GiB at 1B and the 196 ms of casts per update (est.)*;
   confidence medium.
5. **Host synchronisation.** Measured: 119 ms of GPU idle per bf16 production update
   (4.4%) and 131 ms per TF32 update (3.7%), from 536 synchronous host-to-device copies, and the drain worth 1-2% (1B) and 5% (500M)
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
  this card's bandwidth. Not traced; the cause is open. *(Traced and fixed on
  `perf/host-gemm`: the backward lost the forward's saved logsumexp and recomputed it;
  see the follow-up section below.)*
- **`NSL_MATMUL_BF16` is read at build time on `main`** (`MatmulConfig::
  with_env_fallback`; the runtime ignores the variable by design, see
  `resolve_math_mode_from`). `models/benchmarks/posture_bench_1b.py`'s P4 arm runs
  P3's binary with the variable set at run time and says the setting is "a RUNTIME
  env, same artifact, different run": on today's toolchain that arm silently measures
  TF32. `pretrain_prod.nsl`'s `export NSL_MATMUL_BF16=1; nsl run ...` still works,
  because `nsl run` compiles in the same process. This harness builds the bf16 arms
  with the variable set and refuses a run whose banner does not say BF16. *(On
  `perf/host-gemm` a binary run with a disagreeing `NSL_MATMUL_BF16` refuses to start,
  and P4 builds its own `--matmul-mode bf16` artifact.)*
- **The per-micro-batch drain** (`nsl_gpu_drain_cache` after every micro-batch,
  `stmt_train/driver.rs`) returns ~9.6 GB to the driver and re-allocates it each
  micro-batch at 1B; its own comment calls it a workaround. *(On `perf/host-gemm` the
  boundary releases memory only when the pool holds markedly more than the step
  needed; see below.)*

## Follow-up: host stalls, GEMM selection, compressed saves (`perf/host-gemm`, 2026-11-14)

Changes against the third and fifth items above and the defects list, each measured
on its own. Every toolchain was frozen (compiler, runtime archive, stdlib) and timed in
the same interleaved rounds with `perfcheck --toolchain NAME=DIR` (arms
`nsl_tf32@NAME`, `nsl_bf16@NAME`, the production flag set; `nsl_bf16` is built with
`NSL_MATMUL_BF16=1`, how the 1B recipe trains); ratios are paired within a round, then
the median over rounds. `before` is this document's runtime (`main` at 565567c5); each
later toolchain adds one change to the one before it:

| toolchain | change | commit |
|---|---|---|
| `c1` | RoPE positions gathered from a device-resident iota, not uploaded per call | 4b827951 |
| `c2` | the step boundary releases idle GPU memory only from a bloated pool | fa10886e |
| `c3` | TF32 GEMMs through cublasLt with an enumerated, timed plan per shape | 52c07d7d |
| `c4` | `--checkpoint-compress` keeps the flash forward's saved logsumexp | 018fd86c |
| `c5` | bf16-storage GEMMs take the timed plan too; large-tile whitelist, cheaper tune | e6b126fe |

The run-time `NSL_MATMUL_BF16` refusal (263888b6) changes no arithmetic and is not
timed; 49b0c43b caps the timed selection at 128 shapes per process (no effect on these
programs, which have 15).

### Results

Three rounds; tok/s is the median [min-max] over rounds; each ratio is taken within a
round against the toolchain above it (per-round values, then the median), and the last
column against `before`. The rounds ran beside another worker's GPU jobs, which moved
every arm together by up to 6% between rounds; the paired ratios cancel that.

**Coder-1B** (2 x 2048 x 4):

| `nsl_tf32` | tok/s | vs the row above, per round | median | vs `before` |
|---|---|---|---|---|
| `before` | 5,035 [4,759-5,050] | | | |
| `c1` RoPE positions | 5,046 [4,742-5,057] | 0.997 1.002 1.002 | 1.002 | 1.002 |
| `c2` drain | 5,111 [4,874-5,187] | 1.028 1.013 1.026 | 1.026 | 1.024 |
| `c3` TF32 GEMMs | 5,886 [5,595-5,915] | 1.148 1.152 1.140 | 1.148 | **1.171** |
| `c5` bf16 GEMMs, cheaper tune | 5,900 [5,551-5,912] | 0.992 1.002 0.999 | 0.999 | **1.171** |

| `nsl_bf16` (the 1B recipe) | tok/s | vs the row above, per round | median | vs `before` |
|---|---|---|---|---|
| `before` | 6,721 [6,422-6,741] | | | |
| `c1` | 6,672 [6,361-6,759] | 0.991 1.003 0.993 | 0.993 | 0.993 |
| `c2` | 6,957 [6,501-6,980] | 1.022 1.029 1.046 | 1.029 | 1.032 |
| `c3` | 6,971 [6,531-7,014] | 1.005 1.002 1.005 | 1.005 | 1.034 |
| `c5` | 7,388 [6,820-7,390] | 1.044 1.060 1.053 | 1.053 | **1.096** |

**Coder-500M** (2 x 1024 x 8):

| `nsl_tf32` (the 500M recipe's precision) | tok/s | vs the row above, per round | median | vs `before` |
|---|---|---|---|---|
| `before` | 8,942 [8,349-9,030] | | | |
| `c1` | 9,033 [8,496-9,061] | 1.003 1.018 1.010 | 1.010 | 1.010 |
| `c2` | 9,480 [8,927-9,514] | 1.050 1.051 1.049 | 1.050 | 1.060 |
| `c3` | 11,039 [10,383-11,076] | 1.164 1.163 1.164 | 1.164 | **1.234** |
| `c5` | 11,078 [10,267-11,079] | 1.000 0.989 1.004 | 1.000 | **1.230** |

| `nsl_bf16` | tok/s | vs the row above, per round | median | vs `before` |
|---|---|---|---|---|
| `before` | 11,560 [10,719-11,605] | | | |
| `c1` | 11,008 [10,989-11,765] | 1.025 0.952 1.014 | 1.014 | 1.014 |
| `c2` | 12,337 [11,594-12,392] | 1.055 1.121 1.053 | 1.055 | 1.068 |
| `c3` | 12,458 [11,652-12,507] | 1.005 1.010 1.009 | 1.009 | 1.078 |
| `c5` | 13,073 [12,245-13,150] | 1.051 1.049 1.051 | 1.051 | **1.133** |

Memory: the allocator peaks do not move (24.63 / 26.53 GiB at 1B TF32 / bf16, 12.72 /
13.70 GiB at 500M, every toolchain). The driver-reported peak moves by -0.04 to +0.04
GiB at 1B and by +0.29 GiB at 500M TF32 (14.15 to 14.44: the pool now keeps its working
set, +0.21 GiB reserved, and TF32 runs now hold the 64 MiB cublasLt workspace); 500M
bf16 +0.03.

**The RoPE change on its own is at the noise floor.** Against the toolchain before it,
it is +1.0% at 500M TF32 (all three rounds positive) and between -0.7% and +1.4% on the
other arms. Measured again on top of everything else (`c5` against `c5` with the iota
path disabled, three more rounds): 1B 1.007 [0.987-1.023] TF32 and 0.999 [0.995-1.002]
bf16, 500M 0.998 [0.998-1.000] TF32 and 1.009 [1.008-1.065] bf16. It removes 128
device synchronisations per micro-batch, but the host was already ahead of the GPU
there, so they cost the GPU little; it is kept for the 500M gain and because the
copies it removes become a larger share as the attention kernels get faster.

**`--checkpoint-compress bf16`** at 1B (`c4` against `c3`, which still has the defect):

| arm | `c3` tok/s (s/update) | `c4` tok/s (s/update) | `c4` / `c3`, per round | allocator peak |
|---|---|---|---|---|
| TF32 + compress | 1,788 [1,684-1,790] (9.16) | 5,614 [5,288-5,619] (2.92) | 3.141 3.142 3.137 | 21.24 GiB |
| bf16 + compress | 1,869 [1,753-1,870] (8.77) | 6,583 [6,224-6,594] (2.49) | 3.551 3.527 3.522 | 23.13 GiB |

With the fix, compression costs 4.6% (TF32: 5,614 against 5,886 for `c3` without it)
and 5.6% (bf16: 6,583 against 6,971) for 3.39 GiB less allocator peak (24.63 to 21.24
and 26.53 to 23.13 GiB); before it cost 3x.

**The first update** pays the GEMM tune once per process: at 1B TF32 it took 14.4 s
with `c3` (11 s of tuning over 15 shapes) and 4.5 s with `c5` (about 1 s), against
3.4 s with `before`; at 500M 1.8 s with `c5` against 1.8 s.

Against PyTorch (*not paired: the PyTorch numbers are this document's, from earlier
rounds, est.*): eager TF32 / NSL TF32 goes from 1.44x to *~1.23x* at 1B and from 1.53x
to *~1.23x* at 500M; PyTorch bf16 + `torch.compile` / NSL bf16 at 1B from 2.23x to
*~2.0x*. Attention (673 ms of a 3.0 s update) is now the largest single difference.

### What each change does, and the evidence

**RoPE positions (`c1`).** `RotaryEmbedding.forward` gathers rows `0..seq_len` of its
cached cos/sin tables with `embedding_lookup(table, arange(0, seq_len))`. The positions
are built on the host and were uploaded with a synchronous `cuMemcpyHtoD_v2` on every
call: 128 per micro-batch at 1B, each waiting for the device to drain. The host-side
bounds check now also notes whether the indices are exactly `0.0 .. n-1`; if so the
kernel reads a device copy uploaded once per device. Same kernel, same index values,
same output. In the 1B TF32 trace the 8 KB copies go from 12,321 per run to 34.

**Step-boundary drain (`c2`).** The per-micro-batch `nsl_gpu_drain_cache` returned
every idle transient segment to the driver (95 `cuMemFree` per micro-batch at 1B, 24 ms
of GPU idle while the host sat in them) and the next micro-batch allocated them again.
It was added in April 2026 when the pool grew ~3 GB per step without it. That growth no
longer happens: with the drain off, 500M and 1B reach their working set in two or three
micro-batches and stay flat for 192 and 96 micro-batches, with no driver allocation
after the third (`NSL_DEBUG_MEM_ALL=1`, reserved 14,392 MB at 500M and 26,084 MB at 1B
throughout). The boundary now releases only when the reservation exceeds the window's
peak by more than max(1 GiB, half the window's transient demand), which never fires
here; an allocation that fails still synchronises, drains and retries (a gate allocates
a post-training tensor larger than any cached segment under a memory limit that fits it
only once the idle cache is released: 7 driver frees under the limit, 0 without). The
reserved peak moves by -0.04 GiB at 1B (25.51 to 25.47) and +0.21 GiB at 500M (13.84 to
14.05).

**TF32 GEMM selection (`c3`).** cuBLAS 13.8's `cublasSgemm` heuristic on sm_120 returns
the 64x64-tile `cutlass_80_tensorop_s1688gemm_64x64_16x6` kernel for most of the
projections and their gradients, and cublasLt's heuristic offers only one or two
candidates for the NN/NT forms, both that kernel. Enumerating cublasLt's algorithm
configurations (`cublasLtMatmulAlgoGetIds` x tile x stages, `cublasLtMatmulAlgoCheck`)
reaches the 128x128 / 128x256 / 256x128 kernels PyTorch's cuBLAS 12.8 picks. Per call on
random operands at 1B's shapes (ms; `gemmprobe/enum.cu` in the work directory; F/D/W
= forward, input gradient, weight gradient):

| GEMM (cuBLAS m x n x k) | sgemm | best enumerated | |
|---|---|---|---|
| F q, o (2048 x 4096 x 2048) | 0.62, 0.69 | 0.47, 0.47 | 1.32x, 1.47x |
| D q, o (2048 x 4096 x 2048, A^T) | 0.61, 0.68 | 0.47, 0.48 | 1.30x, 1.40x |
| W q, o (2048 x 2048 x 4096, B^T) | 0.67, 0.65 | 0.50, 0.50 | 1.35x, 1.28x |
| F k/v (512 x 4096 x 2048) | 0.17 | 0.12 | 1.41x |
| F gate/up (8192 x 4096 x 2048) | 2.02 | 1.82 | 1.11x |
| D gate/up (2048 x 4096 x 8192) | 2.62 | 2.04 | 1.29x |
| W gate/up (8192 x 2048 x 4096) | 2.74 | 1.95 | 1.41x |
| F down (2048 x 4096 x 8192) | 2.80 | 2.01 | 1.39x |
| D down (8192 x 4096 x 2048) | 1.79 | 1.79 | 1.00x |
| W down (2048 x 8192 x 4096) | 2.84 | 1.96 | 1.45x |

(Timed on random data: on zero-filled operands the same kernels run ~25% faster on this
power-capped card, which is how an early probe under-reported the gap.) Each GEMM of at
least 2^28 multiply-adds is now timed once per process on its live operands (the
heuristic's candidates plus enumerated configurations, CUDA events, two rounds) and the
winner cached; every later call of the shape replays the identical kernel. Not under
`--deterministic` (cuBLAS's heuristic is reproducible, a timed winner is not),
`--cuda-graphs` (the tune synchronises), or `--no-bf16-lt-tune`.

**bf16 GEMMs (`c5`).** The same enumeration on bf16 operands (`gemmprobe/enum_bf16.cu`)
against `cublasGemmEx`'s default: q/o 0.28 -> 0.24 ms, gate/up dgrad 1.26 -> 0.95, wgrad
1.27 -> 0.94, down wgrad 1.14 -> 0.95; gate/up forward and down dgrad unchanged (0.88,
0.90). bf16-storage products now take the timed plan by default when timed selection
applies (before, only under `--bf16-lt`, and over the heuristic's candidates only).
`c5` also restricts the enumeration to the five tiles every winner used and times
each candidate once before re-timing the four fastest, which cut the 1B tune from 11 s
of the first update (`c3`) to about 1 s with the same steady-state kernels.

**Compressed saves (`c4`).** The 3x was not the casts (55 + 52 ms per update) but
`nsl_flash_lse_f32`: 108 ms per call, 6.9 s of the 9.6 s update, a kernel the plain run
never launches. Compression stores the attention output as bf16 and the backward reads
a restored f32 copy; the codegen side channel that hands the backward the forward's
saved logsumexp was keyed by the original value, so the backward got none and the
runtime recomputed it, one thread per row over the whole key range. The side channel
now follows both casts. On Coder-50M with the production flags the previous toolchain
launches `nsl_flash_lse` 28 times in two updates, the fixed one never.

### After, on the GPU timeline (Nsight Systems, Coder-1B, ms per update)

| | TF32 before | TF32 `c5` | bf16 before | bf16 `c5` | PyTorch TF32 eager |
|---|---|---|---|---|---|
| wall (traced) | 3,575 | 3,031 | 2,681 | 2,434 | 2,300 |
| GPU idle | 131 | 10 | 119 | 11 | 9 |
| GEMM | 2,110 | 1,665 | 1,013 | 856 | 1,520 |
| attention | 673 | 672 | 665 | 677 | 74 |

The GEMMs now run `cutlass_80_tensorop_s1688gemm` 128x128 / 128x256 / 256x128 (TF32)
and `s16816gemm_bf16` 128x128 / 256x128 (bf16) kernels. The 64x64 kernel is left on one
batched product per layer and micro-batch (batch 2, `cublasGemmStridedBatchedEx`, which
this change does not route: 43 ms per update). The remaining
synchronous host-to-device copies (token ids for the embedding forward and again for
its backward, labels, two 32 KB LM-head tables, a few scalars: about 7 per micro-batch)
leave 10 ms of idle per update, under 1%, and were left alone.

### Correctness

`tools/refcheck` at 50M with the production flag set (gradient check without
`--fuse-wgrad-accum`, which needs accumulation; optimizer check: AdamW, accumulation 4,
clip 1.0, weight decay 0.1, schedule), relative error against the PyTorch twin, worst
over 74 parameters [median]:

| check | `before` | `c3` (TF32 GEMMs) | `c5` (bf16 GEMMs too) |
|---|---|---|---|
| gradient, TF32 | 2.56e-3 [1.40e-3] | 2.61e-3 [1.41e-3] | 2.62e-3 [1.41e-3] |
| gradient, bf16 | 6.31e-3 [4.46e-3] | 6.23e-3 [4.43e-3] | 6.30e-3 [4.41e-3] |
| 3 AdamW updates, TF32 | 4.03e-3 [2.37e-3] | 3.94e-3 [2.38e-3] | 3.95e-3 [2.37e-3] |
| 3 AdamW updates, bf16 | 9.45e-3 [7.20e-3] | 9.01e-3 [7.23e-3] | 9.43e-3 [7.23e-3] |

The bf16 gradient check exceeds the tool's 5e-3 default tolerance (a TF32 number)
before and after alike: bf16 operands keep 8 significant bits.

### Reproducing the follow-up

Raw results, logs, traces and the probes are in
`/mnt/extrastorage/nsl-audit/research/work/PERF-HOST-GEMM/` on the measuring machine
(`runs/`, `gemmprobe/`, `campaign2.sh`, `campaign3.sh`). A toolchain is frozen as a
directory holding `bin/nsl`, the CUDA runtime archive as `lib/libnsl_runtime.a` and
`stdlib/`; then, per size and round, each command one queued GPU job:

```bash
P="VENV/bin/python tools/perfcheck/perfcheck.py"; T="--toolchain before=DIR0 --toolchain c1=DIR1 ..."
$P build --size 1b --arms nsl_tf32@before,nsl_tf32@c1,... $T --out OUT
$P run --size 1b --arms nsl_tf32@before,nsl_tf32@c1,... --round 0 --out OUT
$P pairs --out OUT --pair nsl_tf32@c1/nsl_tf32@before --pair nsl_tf32@c2/nsl_tf32@c1
$P nsys --size 1b --arm nsl_tf32@c5 --out OUT
```

## Both rounds together, against `main` and PyTorch (`perf/round-1`, 2026-11-15)

`perf/host-gemm` (above) and the tensor-core attention
(`ATTENTION_TC_2026_11.md`) combined, timed against `main` at c92f9300 (`before`)
with the PyTorch arms in the same rounds: frozen toolchains, production flags, three
interleaved rounds, arm order reversed in the second round, ratios paired within a
round, then the median [min-max] over rounds.

| arm | Coder-1B tok/s | driver peak GiB | Coder-500M tok/s | driver peak GiB |
|---|---|---|---|---|
| `nsl_tf32@before` | 5,067 [5,066-5,068] | 25.83 | 8,952 [8,932-8,966] | 14.15 |
| `nsl_tf32@after` | 7,245 [7,121-7,269] | 25.87 | 12,758 [12,729-13,066] | 14.44 |
| `torch_tf32_eager_sel` | 7,335 [7,292-7,369] | 25.55 | 13,842 [13,809-14,009] | 12.12 |
| `nsl_bf16@before` | 6,778 [6,739-6,793] | 27.75 | 11,639 [11,627-11,705] | 15.14 |
| `nsl_bf16@after` | 9,619 [9,572-9,650] | 27.79 | 15,593 [15,568-15,632] | 15.17 |
| `torch_bf16_compile_sel` | 15,209 [15,050-15,229] | 22.25 | 25,121 [25,079-25,342] | 11.38 |

| paired ratio | Coder-1B | Coder-500M |
|---|---|---|
| `nsl_tf32@after / nsl_tf32@before` | **1.430** [1.405-1.434] | **1.423** [1.422-1.463] |
| `nsl_bf16@after / nsl_bf16@before` | **1.420** [1.416-1.424] | **1.338** [1.336-1.341] |
| `torch_tf32_eager_sel / nsl_tf32@after` | 1.014 [1.012-1.024] | 1.085 [1.072-1.085] |
| `torch_bf16_compile_sel / nsl_bf16@after` | 1.578 [1.572-1.581] | 1.614 [1.608-1.621] |
| `torch_bf16_compile_sel / nsl_bf16@before` | 2.239 [2.233-2.247] | 2.158 [2.157-2.165] |

At TF32 NSL now matches eager PyTorch at 1B (1.4% behind) and is 8.5% behind at 500M,
at the same memory. Against compiled PyTorch in bf16, the 1B recipe's posture, the
gap is 1.58x (it was 2.24x), and NSL holds 5.5 GiB more at its peak. What remains is
the work bf16 activation storage removes (NSL stores activations in f32 and casts
both operands of every GEMM: 322 ms of casts and strided copies per 1B update in the
baseline trace) and the elementwise and norm kernels torch.compile fuses.

Agreement with PyTorch at 50M (`tools/refcheck`, production flags), `before` / `after`:
`grad_check` TF32 worst relative error 2.55e-3 / 2.30e-3 (tolerance 5e-3), bf16
6.34e-3 / 6.16e-3 (both above the 5e-3 default, which is set for TF32 operands; see
`tools/refcheck/README.md`); `optimizer_check` TF32 4.02e-3 / 3.59e-3, bf16 9.33e-3 /
9.08e-3 (tolerance 3e-2).

Raw results: `/mnt/extrastorage/nsl-audit/research/work/PERF-ROUND1/` on the measuring
machine (`campaign.sh`, `refcheck.sh`, `runs/`).

## Reproduce

Raw results (one JSON per run, logs, traces, trace summaries) are in
`/mnt/extrastorage/nsl-audit/research/work/PERF/runs/` on the measuring machine. Every
number above comes from one of the commands below: the throughput, memory, per-round,
paired-ratio, phase, NSL memory-context and host-profile tables from `report`; the
roofline and NSL GEMM probe from `roofline` (`roofline.json`); the cuBLAS workspace
bytes from `workspace`; every per-update kernel, phase, copy and per-layer figure from
`nsys_summary.py` on the five traces. Each GPU command was queued on its own on this
machine (`/mnt/extrastorage/nsl-audit/bin/nslq gpu -- ...`).

```bash
cargo build --release --bin nsl --features cuda
uv venv --python 3.12 VENV
uv pip install --python VENV/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python VENV/bin/python numpy tokenizers
VENV/bin/python tools/perfcheck/test_perfcheck.py      # harness arithmetic; needs torch, no GPU
P="VENV/bin/python tools/perfcheck/perfcheck.py"; OUT=/a/real/disk/perf
CORE=nsl_tf32,nsl_bf16,torch_tf32_eager_sel,torch_bf16_eager_sel,torch_bf16_compile_sel
EXTRA=nsl_tf32_nodrain,nsl_bf16_nodrain,nsl_tf32_blocks,torch_tf32_eager_blocks,torch_bf16_compile_blocks,torch_bf16_compile_none

$P roofline --out $OUT
$P workspace --out $OUT
for s in 1b 500m; do
  $P build --size $s --arms nsl_tf32,nsl_bf16,nsl_tf32_blocks,nsl_tf32_compress,nsl_bf16_compress --phase --out $OUT
  for r in 0 1 2; do $P run --size $s --arms $CORE --round $r --out $OUT
                     $P run --size $s --arms $EXTRA --round $r --out $OUT; done
  $P run --size $s --arms $CORE --round 3 --out $OUT
  $P phases --size $s --arms $CORE --out $OUT
done
for r in 0 1 2; do $P run --size 1b --arms nsl_tf32_compress,nsl_bf16_compress --round $r --out $OUT; done
$P phases --size 1b --arms torch_bf16_compile_blocks,torch_bf16_compile_none,torch_tf32_eager_blocks --out $OUT
for a in nsl_tf32 nsl_bf16 torch_tf32_eager_sel torch_bf16_eager_sel torch_bf16_compile_sel; do
  $P nsys --size 1b --arm $a --out $OUT; done
$P diag --size 1b --arm nsl_tf32 --env NSL_HOST_PROFILE=1 --tag hostprof --out $OUT
$P report --out $OUT

N="VENV/bin/python tools/perfcheck/nsys_summary.py"; T=$OUT/1b_b2_s2048_a4/nsys
COMMON="--copies --layers 16 --top 40"
for a in nsl_tf32 nsl_bf16; do
  $N $T/$a.sqlite $COMMON --boundary fase_fused_adamw --fwd-start nsl_embedding_i32idx \
    --bwd-start lce_chunk_dlogits --layer-mark flash_attn_bwd_main \
    --recompute 'nsl_rmsnorm_f32|rmsnorm_rinv|rotate_half|nsl_silu_f32|embedding_f32'; done
for a in torch_tf32_eager_sel torch_bf16_eager_sel; do
  $N $T/$a.sqlite $COMMON --boundary FusedAdam --opt-start LpNormFunctor \
    --fwd-start vectorized_gather_kernel --bwd-start nll_loss_backward \
    --layer-mark flash_bwd_dq_dk_dv \
    --recompute 'pow_tensor_scalar|MeanOps|silu_kernel|neg_kernel|CatArrayBatchedCopy|rsqrt'; done
$N $T/torch_bf16_compile_sel.sqlite $COMMON --boundary FusedAdam --opt-start LpNormFunctor \
  --fwd-start fused__to_copy_add_div_index_mean --bwd-start log_softmax_backward_data \
  --layer-mark flash_bwd_dq_dk_dv
```

`nsys_summary.py --sequence 40` additionally prints one forward's first kernels and one
backward layer in order, the source of the kernel-by-kernel descriptions above (RoPE
as ~10 kernels, the per-row attention launches).
