# Compiler-Generated All-Reduce Compression (CGAC) — Errata

**Companion to** `CGAC.pdf`. **Date:** 2026-11-13.

The proposal was reviewed in November 2026 against its own arithmetic, the
papers and libraries it cites, and this repository. This file lists the
statements in the PDF that should not be relied on and says what is true
instead. Arithmetic was recomputed from the proposal's own figures. External
facts were checked against the linked source unless the entry says otherwise.

---

## Part 1 — Errors in the proposal

### E1 — QSGD levels from different ranks cannot be added as integers, and the ratio is 4.0×, not 6.4×

**Proposal.** §3.2: each rank quantizes against its own scale ("Compute
∥g∥∞"), and aggregation is an "associative sum (integer addition of
quantized levels)". "For b = 4: 6.4× compression." §2.2: "QSGD-with-EF and
signSGD-with-EF also use the same machinery."

**Correction.** Each rank has a different scale, so a level stands for a
different amount on each rank and the integer levels cannot be added. In a
numeric check with 8 ranks, one of them 30 times larger than the others,
adding the levels and dequantizing with any single scale gave a relative
error of about 0.9 to 2.5. With one scale shared by all ranks the error was
about 0.25, and unbiased. A shared scale needs a max-all-reduce before the
gradient all-reduce.

The summed levels also need room to grow: 1 + b + ⌈log₂N⌉ bits per element
on the wire, which is 8 bits for b = 4 and N = 8. That is 32/8 = 4.0×, not
the 32/(b+1) = 6.4× claimed.

Error feedback (EF) for QSGD is unnecessary, because QSGD is unbiased.

**Source.** Numeric check.

### E2 — CFIP passthrough is not a valid reduction, and NCCL has no operator for it

**Proposal.** §2.4: "element-wise sum in the CFIP format's representation
(integer sum for the mantissa; per-block scale renormalization)". §4.2: "NCCL
custom reduction op registered at init". §4.4: "The NCCL library supports
this pattern directly via hierarchical collective configurations."

**Correction.** The CFIP formats, as the proposal describes them, store small
integers with a scale per block. When ranks hold different block scales,
Σ sᵣqᵣ ≠ (Σ qᵣ)·s, so adding the integer parts does not add the values.
Renormalization on the receiving side cannot recover the sum.

NCCL (NVIDIA's collective communications library) has one user-defined
reduction operator, `ncclRedOpCreatePreMulSum`, which multiplies by a scalar
and then sums. No hierarchical-collective API was found in NCCL (the search
was not exhaustive); the pattern in §4.4 would be two communicators built by
hand.

**Source.** Numeric check for the sum.
<https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2205/user-guide/docs/api/ops.html>
for the operator.

### E3 — The PowerSGD+ steps are not the paper's: the SVD is of an all-reduced full update, and the error buffer is never cleared

**Proposal.** §3.1, periodic reprojection: "Compute SVD of the accumulated EF
buffer", "Replace Qt with the top-r right singular vectors", "Clear the
accumulated EF". §2.3: the reprojection "costs an SVD (a few milliseconds for
typical ranks)".

**Correction.** In the PowerSGD+ paper (Xie, Chen, Yuan, September 2025),
every τ steps the workers all-reduce the full, uncompressed update Δ and take
the singular value decomposition (SVD) of that average; the paper calls the
step `SVD-Comp`. Communication per iteration is nr + mr + mn/τ. The
proposal's version differs in three ways that matter:

- The error buffer is never cleared; e ← Δ − Δ̂ on every step. Clearing it
  discards gradient that has not yet been transmitted.
- An SVD taken on each worker's local buffer gives the workers inconsistent
  bases and breaks the all-reduce. The paper makes this point about ATOMO.
- The periodic step costs a full m×n SVD per matrix plus an uncompressed
  all-reduce. With τ = 100, the effective compression of a 4096×4096 matrix
  at r = 4 is capped near 84×: 4096² / (2·4·4096 + 4096²/100) = 83.7.

The statement in §2.3 that the original PowerSGD can fail "regardless of
learning rate" is supported: the paper's Theorem 1 covers any step schedule
and base optimizer, for a 2×2 high-noise example.

**Source.** <https://arxiv.org/abs/2509.11254> (PDF). The 84× figure is
recomputed from the paper's communication formula.

### E4 — The signSGD description is wrong on aggregation, the residual, the ratio and the timing

**Proposal.** §3.3: "sum of signs (not associative in general; handled via
all-gather + majority vote)"; "Compute ∣g∣ — residual for EF"; "1 bit per
element. 32× compression vs FP32"; "CGAC uses signSGD only in late-training
phases (last 10-20% of steps) where gradient magnitudes converge and the sign
approximation is dominant anyway". §4.2: "Majority vote: all-gather + bitwise
OR + popcount."

**Correction.**

- Integer sign counts are associative, and a majority vote is the sign of the
  summed counts. The real cost is width: a count needs ⌈log₂(N+1)⌉ bits.
- A bitwise OR does not give a majority. A per-position count is needed.
- The error-feedback residual is p − α·sign(p) with p = g + e, the gradient
  plus the carried error. It is not |g|.
- With an all-gather the ratio is not 32×. A ring all-reduce of FP32 moves
  2(N−1)/N·32 bits per element per rank, and a sign all-gather moves N−1
  bits. The ratio is 8× at N = 8, 4× at N = 16 and 1× at N = 64.
- The late-training argument is backwards. Late training has the lowest
  per-coordinate signal-to-noise ratio, which is where plain signSGD is
  weakest.
- signSGD without error feedback does not converge in general, and error
  feedback with a scaled sign repairs it (Karimireddy et al. 2019; not
  re-checked against the source). Majority vote combined with per-worker
  error feedback, as the proposal combines them, is covered by neither
  analysis.

**Source.** The ratios are recomputed from the all-reduce formula in §1.1.

### E5 — The headline gain contradicts the proposal's own cost model, and the utility study it cites points the other way

**Proposal.** Abstract: "20-40% training throughput improvement on multi-node
training at 10Gbps inter-node bandwidth; 5-15% improvement at faster 200Gbps
networks". §1.1 puts the all-reduce at about 40 seconds at 10 Gbps, against
"approximately 1-2 seconds" of backward compute.

**Correction.** By the model in §1.1, a step at 10 Gbps is 39.2 s of
all-reduce plus 1–2 s of compute. Removing 99% of the all-reduce leaves
0.39 s plus the compute, a speedup of 17× to 29× (about 21× at 1.5 s of
compute), not 20–40%. At 200 Gbps, where the all-reduce is 1.96 s, the same
calculation gives 2.0× to 2.9×, not 5–15%. The stated gains are an order of
magnitude away from the proposal's own model.

The study the proposal cites for utility, Agarwal et al. (MLSys 2022), found
that "only in 6 cases out of more than 200, gradient compression methods
provide speedup over optimized synchronous data-parallel training in the
typical data-center setting". The proposal does not engage with that result.

**Source.** Recomputed from the proposal's own figures;
<https://arxiv.org/abs/2103.00543>.

### E6 — PowerSGD at M = N = 4096, r = 4 compresses 512×, not 1024×

**Proposal.** §3.1: "2 × r ⋅ (M + N ) floats per all-reduce", and "For M = N
= 4096 and r = 4: compression ratio ≈ 1024×." §2.1 table: compression
"rank/(min_dim)".

**Correction.** The proposal's own formula M·N/(2r(M+N)) gives
4096² / (2·4·8192) = 256×. The correct count is r(M+N) floats, which gives
512×. The §2.1 table entry is the inverse of a compression ratio and is also
off by a factor of 2: for a square matrix the ratio is min_dim/(2·rank).

**Source.** Recomputed from the proposal's own figures.

### E7 — Error-feedback buffers for a 7B model are about 28 GB per rank, not 8 GB

**Proposal.** §4.3: "approximately 30% of gradient memory (the fraction of
parameters that are large linear layer weights)" and "~8 GB of EF buffers".

**Correction.** In a 7B-parameter transformer nearly all parameters are large
matrices, so the buffers are about 7e9 × 4 bytes = 28 GB per rank. PyTorch's
documentation says the same of its own hook: "extra memory of the same size
as the model's gradients". This also cancels the gradient-buffer saving that
§2.5 attributes to FASE Deferred, the NSL optimizer mode that keeps no
separate gradient buffer.

**Source.** Recomputed from the proposal's own figures;
<https://docs.pytorch.org/docs/2.8/ddp_comm_hooks.html>.

### E8 — The convergence gate would fail on the PowerSGD+ paper's own results

**Proposal.** §6.3: "training a small model (50M) for 5000 steps with CGAC
must produce final loss within 1% of uncompressed baseline".

**Correction.** Table 3 of the PowerSGD+ paper, LLaMA-60M on C4, reports a
perplexity of 29.88 at full rank, against 35.46 for PowerSGD+ at r = 4 and
32.63 at r = 8. In log-loss (the logarithm of perplexity) that is +5.0% and
+2.6%. For LLaMA-130M the figures are 27.36 against 23.06.

**Source.** <https://arxiv.org/abs/2509.11254> (PDF, Table 3). The percentages
are recomputed.

### E9 — PyTorch's PowerSGD hook manages its own error-feedback state

**Proposal.** Abstract: PyTorch's hook "applies one algorithm globally and
requires user-managed error-feedback buffers". §2.2: "In PyTorch, the user
manages this buffer".

**Correction.** PyTorch's `PowerSGDState` holds the buffers, defaults to
`use_error_feedback=True, warm_start=True`, and implements
`__getstate__`/`__setstate__` for checkpointing. "One algorithm globally" is
half right: the hook all-reduces biases, and tensors below
`min_compression_rate`, uncompressed. The hook also delays compression
(`start_powerSGD_iter=1000`) because "the beginning of training phase is
usually very sensitive to inaccurate gradients". The proposal has no warm-up.

**Source.** <https://docs.pytorch.org/docs/2.8/ddp_comm_hooks.html>.

### E10 — The design does not say how compression composes with AdamW

**Proposal.** §2.5: "FASE Deferred → compress MHat's gradient contribution
(per step), reduce, dequant, accumulate into MHat."

**Correction.** PowerSGD's analysis and experiments use SGD, with momentum
applied after compression. NSL trains with AdamW, and §2.5 does not say how
the second moment `v` is formed from a compressed gradient. 1-bit Adam exists
because naive compression into Adam's non-linear update misbehaves (not
re-checked against the source).

**Source.** <https://ar5iv.labs.arxiv.org/html/1905.13727>.

### E11 — Smaller slips and internal contradictions

- §1.1: "At NVLink/NVSwitch intra-node bandwidth (600 Gbps): the all-reduce
  shrinks to ~300 ms". By the section's own formula 600 Gbps gives 0.65 s,
  and NVLink 3's actual 600 GB/s gives 0.08 s. Neither is 300 ms. The
  10 Gbps (39.2 s) and 200 Gbps (1.96 s) figures do follow from the formula.
- §2.1 says "The selection is made at compile time". The same section relies
  on gradient statistics "measured during the first N training steps", and
  §2.6 on a bandwidth pass that "runs once at initialization".
- §5.1, invariant I-37, says "Runtime cannot change the algorithm". §3.3
  uses signSGD "only in late-training phases".
- Checked and correct as cited: "120× compression on LSTM" (rank 4, Table 3
  of the PowerSGD paper) and "54% communication time reduction on ResNet-18"
  (its introduction). End to end, that paper's time per batch fell 23%
  (ResNet-18, rank 2) and 55% (LSTM). LQ-SGD (arXiv 2506.17974) exists and
  is PowerSGD plus logarithmic quantization. Reference 8 (Aragani et al.)
  was not checked.

**Source.** Recomputed from the proposal's own figures. PowerSGD:
<https://ar5iv.labs.arxiv.org/html/1905.13727>.

---

## Part 2 — Superseded by later work

- **Infrequent synchronisation.** Low-bandwidth multi-node training has moved
  away from per-step gradient compression. Streaming DiLoCo
  (<https://arxiv.org/abs/2501.18512>) overlaps communication with compute and
  quantizes outer gradients to 4 bits. MuLoCo
  (<https://arxiv.org/abs/2505.23725>) uses Muon as the inner optimizer with
  error feedback, compresses the delta to 2 bits, and communicates 8× less
  than DiLoCo. NSL already has Muon, so this is the natural design if 10 GbE
  clusters are a real target.
- **ZeRO++** (<https://arxiv.org/abs/2306.10209>). This extension of ZeRO
  (Zero Redundancy Optimizer) sharding is the production precedent for
  compression inside sharded collectives, and the proposal does not cite it.
  It replaces reduce-scatter with a quantized gradient
  average built on all-to-all, precisely because quantized values cannot be
  summed in an all-reduce. It reports 4× less communication and up to 2.16×
  throughput at 384 GPUs.
- **Error-feedback theory.** The field has moved to EF21 (not re-checked
  against the source) and to "Momentum Provably Improves Error Feedback!"
  (Fatkhullin, Tyurin, Richtárik), which the PowerSGD+ paper cites
  (<https://arxiv.org/abs/2509.11254>). Classic error feedback with stochastic
  gradients can need large batches. An EF design here should be the momentum
  variant.
- **Interconnects.** 400–800 Gb/s per GPU is now standard for training
  clusters, which strengthens the conclusion of Agarwal et al.
  (<https://arxiv.org/abs/2103.00543>). The 10 Gbps regime is hobbyist or
  geo-distributed training, which is DiLoCo's territory.

---

## Part 3 — What the code implements

Nothing in this proposal is implemented — the repository has no
gradient-compression code and no flag or decorator for it — and the pieces it
would build on are also missing: the collectives planned by CPDT (the
companion Compile-Planned Distributed Training proposal) are not lowered into
compiled code, there is no hierarchical all-reduce, and the ZeRO runtime
reduces whole tensor lists synchronously.

---

## Part 4 — Recommended direction

1. Do not build CGAC as specified; mark the proposal superseded.
2. Measure first: time each collective in the ZeRO runtime's gradient
   reduction, and build nothing unless communication exceeds about 30% of
   step time after bucketing and overlap.
3. The first real lever is asynchronous, overlapped, bucketed collectives in
   the existing ZeRO runtime, which carry no accuracy risk.
4. If compression is still wanted, use shared-scale unbiased quantization
   (about 4× at 8 ranks, no error-feedback state, composes with AdamW) or a
   ZeRO++-style all-to-all; treat PowerSGD and MuLoCo-style sync as research.

---

## Part 5 — Implementation defects found in the same review

This proposal has no implementation, so the review had no CGAC code to run.
It found no implementation defects to list here.
