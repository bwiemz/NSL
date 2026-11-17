# bf16 activation storage: convergence at Coder-500M (2026-11-16)

**Question.** Does `precision = "bf16"` (bf16 activation storage) train as well as the
arithmetic the production recipes used before it? Its gradients were already checked one
step at a time (`tools/refcheck`, within PyTorch autocast's distance of f32 at 50M and
500M). This record covers the accumulated effect over a training run.

**Answer.** Yes, at the resolution five seeds give. Over 5 seeds x 4,000 updates at
Coder-500M, the held-out loss with bf16 storage was no worse than with TF32 GEMMs or with
bf16 GEMMs over f32 storage, at every 1,000-update mark. At 4,000 updates the mean paired
difference was -0.0035 against TF32 and -0.0001 against bf16 GEMMs, with two standard
errors of 0.018. All eight means are negative, so bf16 storage was slightly lower than both
references on average at every mark. The test was one-sided and fixed before the first run.

This is what moved `models/coder500m` and `models/coder1b` to `precision = "bf16"` and to the
`0.11` toolchain channel.

## Design

The criterion was written and hashed (sha256 `daafd18f…5327`) on 2026-11-15, before the
pilot and before any training run, and was not edited afterwards. The number of seeds was
fixed at five on 2026-11-16, after seed 1 and before any run of seeds 2-5 (sha256
`e77e7451…16e3`). The test was evaluated once, at n = 5; there was no interim test and no
early stop.

- **Model.** Coder-500M as in `models/coder500m/model.nsl`: d 1280, 24 layers, GQA 20/10,
  d_ff 3520, vocab 49152, tied LM head, dropout 0.1.
- **Step.** The 500M recipe's: batch 2 x 1024, accumulation 8 (16,384 tokens per update),
  AdamW lr 1.5e-4 (betas 0.9/0.95, eps 1e-8, weight decay 0.1), global-norm clip 1.0,
  `@fused_lm_ce`, `schedule_unit = "optimizer_step"`, and
  `warmup_cosine(warmup 200, total 4,000, min_lr 3e-5)`. 4,000 updates is 65.5M tokens.
- **Flags.** Common to every arm: `--source-ad --checkpoint-blocks --checkpoint-selective
  --fuse-rmsnorm-backward --fuse-wgrad-accum --seed s`. This is the 1B recipe's flag line.
- **Arms.**
  - NS: `precision = "bf16"`.
  - NB: `--matmul-mode bf16`, which is bf16 GEMM operands over f32 storage, the 1B recipe's
    arithmetic before this change.
  - NT: `--matmul-mode tf32`, the 500M recipe's arithmetic before this change.
- **Seeds.** For each seed s, `manual_seed(s)` wrote one initial state that all three arms
  loaded. `--seed s` seeded the data shuffle and the dropout stream, so every arm of seed s
  saw the same batches with the same masks.
- **Corpus.** Built for this check from permissively licensed source on the measuring
  machine: Rust crates from the cargo registry and rust-src, and Python distributions
  (MPL and proprietary packages excluded). Deduplicated by file hash and split by package,
  so no package is in both splits. Tokenized with the v1 tokenizer the 50M and 500M recipes
  use.
  - train.bin: 66,163,391 tokens (sha256 `0a8c3f65…22f7`). One epoch is 32,306
    micro-batches, so no window repeats in the 32,000 a run reads.
  - val.bin: 1,994,186 tokens from 53 packages held out (sha256 `69e987f8…8035`).
  - The production corpora are not on the measuring machine.
- **Metric.** Held-out loss, the mean cross-entropy over all 1,946 consecutive 1,024-token
  windows of val.bin, computed after 1,000, 2,000, 3,000 and 4,000 updates. One f32
  evaluator scored every arm: the PyTorch twin of `tools/refcheck/ref.py`, reading the
  checkpoint's weights, with TF32 off, no autocast and no dropout.
- **Criterion.** For each seed s, D_s = NS - NT and E_s = NS - NB. SE is the sample
  standard deviation divided by sqrt(n). The criterion passes if, at every mark, all four
  of these hold:
  - mean D <= 2 SE(D)
  - mean D <= range(NT) across seeds
  - mean E <= 2 SE(E)
  - mean E <= range(NB) across seeds

  A failure at an early mark fails the whole test, even if the arms agree at 4,000 updates.
- **Mode.**
  - `--deterministic` (with `NSL_FLASH_BWD_CPU=0`) made two runs bit-identical, but at 19.7%
    of production throughput, below the 80% the criterion required to use it. Every arm
    therefore ran in production mode, which is not bit-reproducible: two production runs
    differed by up to 0.087 in per-batch loss after 200 updates.
  - Seed 1 also ran NB twice, to measure that noise in held-out terms.
- **Chunking.**
  - Each process trained 500 updates, saved the full training state (weights, moments,
    loader position and RNG streams) and exited. The next process resumed from that state.
  - The 96 chunks of seeds 2-5 had no failures and no reruns.

## Results

| update | D = NS - NT, mean | 2 SE | range(NT) | E = NS - NB, mean | 2 SE | range(NB) | result |
|---|---|---|---|---|---|---|---|
| 1,000 | -0.0102 | 0.0180 | 0.0294 | -0.0131 | 0.0269 | 0.0259 | pass |
| 2,000 | -0.0042 | 0.0319 | 0.0951 | -0.0015 | 0.0335 | 0.0661 | pass |
| 3,000 | -0.0055 | 0.0196 | 0.0543 | -0.0015 | 0.0215 | 0.0468 | pass |
| 4,000 | -0.0035 | 0.0183 | 0.0467 | -0.0001 | 0.0185 | 0.0327 | pass |

Held-out loss at 4,000 updates (the initial state scores 11.067):

| seed | NS | NB | NT | D | E |
|---|---|---|---|---|---|
| 1 | 2.96558 | 2.94694 | 2.96404 | +0.00154 | +0.01864 |
| 2 | 2.96622 | 2.96885 | 2.97254 | -0.00632 | -0.00262 |
| 3 | 2.98799 | 2.96595 | 2.97056 | +0.01742 | +0.02204 |
| 4 | 2.94874 | 2.95971 | 2.94225 | +0.00649 | -0.01098 |
| 5 | 2.95229 | 2.97961 | 2.98898 | -0.03669 | -0.02732 |

**Noise floor.** In seed 1, a second run of NB from the same initial state, data order and
seed differed from the first by +0.021, +0.052, +0.028 and +0.023 in held-out loss at the
four marks. That is larger than any of the eight mean differences above. A single seed
cannot separate these arms.

**Resolution.** The across-seed standard deviation of E at 3,000-4,000 updates was
0.021-0.024. Five seeds therefore bound a regression of about 0.02 nats at 4,000 updates;
a smaller effect is not excluded.

## Same code as shipped

The runs used a frozen build of the bf16-storage branch from before its review fix-ups. A
deterministic run of the shipped compiler (2e0f26aa, the code merged as #869) reproduced that
build's deterministic pilot bit for bit before seeds 2-5 ran: the same logged losses at
updates 10-50, and the same weight hash after 25 updates. #868's GemmEx rounding fix does
not reach these runs: every storage GEMM at 500M and 1B is at least 2^28 multiply-adds and
takes cublasLt.

## Limits

- **Size.** The test ran at 500M. The 1B recipe was switched on the same evidence: the same
  block structure and flag line, a 500M gradient check within autocast's distance, and the
  1B speed and memory measurements. It was not repeated at 1B.
- **Horizon.** 65.5M tokens is about 0.13 tokens per parameter, which is early training.
  This test says nothing about drift over a full run. The 1B recipe scores its held-out
  sets only after the epoch. During the run, its training loss, printed every 20
  micro-steps, is the only signal.
- **Corpus.** The corpus is code only. The 1B recipe's corpus is mixed code and prose.

## Cost and data

- **GPU time.** About 23 GPU-hours on one RTX PRO 4500 Blackwell:
  - Seed 1 took 8.2 GPU-hours, including the repeat NB arm. Its checkpoints went to a
    42 MB/s disk.
  - Seeds 2-5 took 14.1 GPU-hours, with checkpoints on NVMe.
  - The determinism pilot took 0.7 GPU-hours.
- **Raw data.** On the measuring machine, `/mnt/extrastorage/nsl-audit/research/work/`:
  - `PERF-BF16/s7/` has the criterion, the seed decision, the equivalence check, every
    chunk log, every held-out evaluation and the analysis scripts.
  - `CORPUS-500M/` has the corpus builder, the manifest with per-file hashes and licences,
    and the sums.
