# Compiler-Synthesized Holistic Attention (CSHA) — Errata

**Companion to** `NSL-CSHA-Research.PDF`. **Date:** 2026-11-13.

This file lists the statements in the CSHA proposal (April 2026) that should
not be relied on, and what is true instead. The proposal was reviewed in
November 2026 against its own arithmetic, the literature it cites, and the
code in this repository that implements it. Arithmetic corrections were
recomputed from the proposal's own figures. External facts carry a link where
they were checked against a source, and say so where they were not. The
statements about the code come from reading it and from compile-only builds;
no kernel was run on a GPU for the review itself. The fixes recorded in
Part 5 came afterwards, and those marked as run were run on a GPU.

Notation used below: S is the sequence length, D the model width, d the head
dimension. HBM is the GPU's main (high-bandwidth) memory. A CTA is a
cooperative thread array, one CUDA thread block. FA2, FA3 and FA4 are
FlashAttention-2, -3 and -4. GQA is grouped-query attention, RoPE is rotary
position embedding, and SDPA is scaled dot-product attention.

---

## Part 1 — Errors in the proposal

### E1 — Level 2 pipelining repeats the K and V projections per query tile

**Proposal.** §2.2. The producer pseudo-code computes
`K_tile = RMSNorm(x[kv_tile]) @ Wk + RoPE` and `V_tile` inside
`for each q_tile`, and the text concludes: "They never touch HBM. The only
HBM accesses are reading `x` (once) and writing the final output (once)",
giving a "5× reduction in HBM traffic". The scheme is called "directly
analogous to FA3's producer-consumer warp specialization".

**Correction.** FlashAttention launches one CTA per (batch, head, query
tile), and every CTA walks all key/value tiles. K and V are shared by all
query tiles, and materialising them once is what keeps the projections at
O(S·D·d) work. Computing them inside the query-tile loop repeats that work in
every CTA. For the proposal's configuration (S=1024, D=512, d=64, 8 heads,
4 KV heads, block_q=64):

- Unfused, the K and V projections cost 2·1024·512·256 = 268 MFLOP each,
  537 MFLOP together.
- Fused, each CTA projects all 1024 rows for its own head:
  2·1024·512·64 = 67 MFLOP per projection. There are 16 query tiles × 8
  heads = 128 CTAs, each doing two projections: 128 × 2 × 67 MFLOP =
  17.2 GFLOP. That is 32 times the unfused cost, and about 8 times the
  attention matrix products themselves (QKᵀ and PV over 8 heads,
  2.15 GFLOP). A causal mask halves the fused figure.
- Each CTA reads `x` at width D=512 instead of K and V at width d=64. Summed
  over the 128 CTAs in fp16, that is 128 × 1024 × 512 × 2 bytes = 134 MB of
  input traffic against 128 × 1024 × 64 × 2 bytes × 2 tensors = 33.5 MB,
  four times as much.

The "never touch HBM" claim and the 5× traffic reduction therefore hold only
when the whole sequence fits in one tile. The FA3 analogy does not hold
either: FA3's producer warps issue loads; they do not compute matrix
products. The implementation materialises Q, K and V for this reason (see
Part 3).

**Source.** Recomputed from the proposal's configuration. FA3:
https://arxiv.org/abs/2407.08608

### E2 — The headline speedup does not follow from the proposal's numbers

**Proposal.** §10: "2-5× faster than FlashAttention-3 on the same hardware".

**Correction.**

- The baseline row in §7.1 is FlashAttention-2. FA3 is never modelled.
- The best figure that leaves the model's output unchanged is §7.1's
  Level 2 row: 6.9 μs / 2.8 μs = 2.5×. The 3.6× in §6.3 (6.9 / 1.9) needs
  head pruning and top-k sparsity, both of which change what the model
  computes.
- The "5×" is a ratio of HBM traffic (40 MB / 8 MB), not of time.
- §7.2 itself concedes that the benefit shrinks as D, S and batch size grow.
  FA3 already reaches 740 TFLOPS on an H100, 75% of the 989 TFLOPS peak the
  proposal uses, so in the compute-bound regime the attention kernel has at
  most 989 / 740 ≈ 1.34× left to gain.
- The 6.9 μs baseline is itself wrong (E3), and the best available
  calibration for fusing boundary operations is a published result of up to
  13.2%, not a multiple (Part 2).

**Source.** Recomputed from the proposal's own figures. FA3 throughput:
https://arxiv.org/abs/2407.08608

### E3 — Every "Compute" row of the §1 roofline table is memory-bound

**Proposal.** §1. The table labels the Q, K and V projections,
FlashAttention and the output projection "Compute", totals 6.9 μs, and the
text says "approximately 4.2μs (61%) is spent on memory-bound boundary
operations".

**Correction.** With the proposal's H100 figures, an operation is
compute-bound only above 989e12 / 3.35e12 ≈ 295 FLOP per byte. The table's
own arithmetic intensities for those five rows are 67 to 178, so all of them
are memory-bound. The table computes each time as FLOP / peak throughput for
the "Compute" rows and bytes / bandwidth for the "Memory" rows, and never
takes the larger of the two. Applying the roofline consistently, the layer
takes 40 MB / 3.35 TB/s = 11.9 μs. The rows the table marks "Memory"
(RMSNorm, RoPE, GQA reshape, residual add) move 14 of the 40 MB, which is
35% of the time, not 61%.

**Source.** Recomputed from the proposal's own figures.

### E4 — The premise does not hold at the size it is argued for

**Proposal.** §1: "these boundary round-trips are *the* bottleneck", for
NSLCoder-50M at batch=1, seq=1024.

**Correction.** Every tensor in the §1 table is 1–2 MB and the layer's total
traffic is 40 MB. The H100's L2 cache is 50 MB, so a tensor written by one
kernel and read by the next is mostly served from L2, not from HBM. Two
further points were not re-checked against a source: 989 TFLOPS is not
reachable on a problem of about 0.5 GFLOP, and each kernel launch costs
microseconds, so the absolute figures "6.9μs" and "2.8μs" are not
meaningful. Fusion at this size saves kernel launches, not bandwidth.

**Source.** H100 L2 cache size:
https://www.thundercompute.com/blog/nvidia-h100-specs-full-guide

### E5 — The fused backward of §5.1 has the same per-tile blow-up as Level 2

**Proposal.** §5.1: "accumulate weight gradients as attention backward
produces dQ/dK/dV tiles", in "1-2 kernel launches".

**Correction.** dK and dV for a row are complete only after every query tile
has contributed to them. Forming `x_normᵀ·dK` from the per-tile partial
gradients is valid, because the product is linear, but it performs the
weight-gradient matrix product once per query tile instead of once: 16 times
at the proposal's configuration. This is the blow-up described in E1. The
implementation computes these products in a separate pass after the
attention backward.

### E6 — Head pruning by the norms of Wq and Wk is wrong

**Proposal.** §3.1: "Heads with very small norms contribute little to
attention and can be pruned at compile time". §3.1 also estimates attention
entropy "Using the pre-trained Q/K weights", and §4.3 says of attention
sinks that "The compiler can detect this from weight analysis".

**Correction.** A small Wq or Wk norm drives that head's logits toward zero,
which makes its attention uniform. The head then outputs the mean of V
through Wo, which is not small. A head contributes little only when its
Wv/Wo path is small. Any pruning of a head with nonzero output changes the
function the model computes, and during training the weights move after the
compile-time decision was made. Attention entropy and attention sinks depend
on the inputs and cannot be derived from the weights alone.

### E7 — A single-tile causal sequence still needs the mask

**Proposal.** §4.1: "For seq_len ≤ block_kv: Skip the entire masking branch
(single tile, no mask needed)".

**Correction.** A single tile is the diagonal tile, which is exactly the
tile that needs the j > i mask. Only tiles strictly below the diagonal are
mask-free.

### E8 — The per-head precision rules of §3.3 rest on the wrong quantities

**Proposal.** §3.3: "Value projections → FP8 E5M2 (wider range for
accumulation)", with the other heads assigned a precision by weight
magnitude.

**Correction.** Accumulation happens in the f32 accumulator, not in the
operand format, so the operand's range is not an accumulation concern. The
operands of the tensor-core multiply are activations (x_norm·W), not
weights, so weight magnitude is not the criterion for choosing their
precision. Whether Hopper exposes FP8 operands through the `mma.sync`
instruction at all, rather than only through its warpgroup form (WGMMA),
could not be verified.

### E9 — Internal contradictions and arithmetic slips

- §2.2, "Wq is 512×512 = 1MB (fp16)": 512 × 512 × 2 bytes is 0.5 MB. 1 MB is
  the fp32 size. The §1 table uses fp32 sizes throughout, while §3.3 and
  §6.3 say Q, K and V are FP16.
- The unfused baseline is 40 MB in §1 and §2.2 but 42 MB in §6.3 and §7.1.
- §2.2, "~8MB (read x + write output only)": by the §1 table's own
  accounting of 2 MB per tensor, reading `x` and writing the output is 4 MB.
- §2.2 says Level 2's only HBM accesses are `x` and the final output. §2.3
  defines the output projection and residual add as what Level 3 adds. §6.3
  then prints "Level 3 NOT feasible" followed by "Using Level 2 with
  epilogue-fused output projection". The three sections disagree about which
  level contains the output projection.

**Source.** Recomputed from the proposal's own figures.

### E10 — Unverified statements and citation slips

- §3.2, "block_q=128, block_kv=128 for FA2; block_q=64, block_kv=64 for
  FA3": these tile sizes could not be verified. The FA4 announcement
  describes 128-query tiles on Blackwell
  (https://tridao.me/blog/2026/flash4/).
- §1, "linked via ctypes/pybind11": PyTorch's `scaled_dot_product_attention`
  carries its attention backends in-tree (not re-checked against PyTorch's
  source). They are still opaque to the Inductor compiler, so the section's
  conclusion stands, but the Abstract's "fundamentally impossible in
  PyTorch" is too strong; see the Deep Kernel Fusion item in Part 2.
- §8 lists "DeepFusionKernel" but the reference list has no entry for it.
  The work is "Deep Kernel Fusion for Transformers", arXiv 2602.11808,
  February 2026 (https://arxiv.org/html/2602.11808v1).
- Reference 3 (FA3, arXiv 2407.08608) was checked and says what the proposal
  attributes to it. References 1, 2 and 4–7 were not checked.

---

## Part 2 — Superseded by later work

- **FlashAttention-4** (arXiv 2603.05451, 5 March 2026;
  https://tridao.me/blog/2026/flash4/). It predates the proposal's April
  2026 date and is not mentioned. It reports 1605 TFLOPS in BF16 on a B200
  (71% utilisation), a forward 1.1–1.3× faster than cuDNN 9.13 and 2.1–2.7×
  faster than Triton, and that cuDNN 9.13/9.14 absorbed its techniques so
  that "the latest cuDNN offers similar perf to FA4". The baseline to beat is
  therefore cuDNN's fused SDPA, not FA2. It is written in CuTe-DSL using the
  `tcgen05.mma` instruction and tensor memory, and it finds that "On
  Blackwell, FLOPs are not the limiter for backward; shared memory bandwidth
  is."
- **FA2 was never at peak.** The FA3 paper measured FA2 at only 35%
  utilisation on an H100 (https://arxiv.org/abs/2407.08608), so the
  proposal's baseline of FA2 running at 989 TFLOPS was already unrealistic
  when it was written.
- **PyTorch exposes the cuDNN baseline** as `SDPBackend.CUDNN_ATTENTION`
  (https://pytorch.org/docs/main/generated/torch.nn.attention.SDPBackend.html).
  Any measured comparison should include it.
- **Deep Kernel Fusion for Transformers** (arXiv 2602.11808, February 2026;
  https://arxiv.org/html/2602.11808v1). It fuses the SwiGLU feed-forward
  network and reports up to 13.2% on an H100 over SGLang. This is the best
  available calibration for what fusing boundary operations buys, and it is
  far from 2–5×.
- **Target hardware.** The development machine is an RTX PRO 4500 Blackwell
  (compute capability sm_120), not an H100, so the proposal's 228 KB of
  shared memory and 3.35 TB/s do not describe the target. `tcgen05` is
  reported as unavailable on sm_120; this rests on unofficial documentation
  and an NVIDIA forum reply, and no official statement was found
  (https://grigoryevko.github.io/nvopen-tools/cicc/targets/sm120.html,
  https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598/18).
  If so, `mma.sync` is the right instruction here and FA4's route is closed.

---

## Part 3 — What the code implements

No shipped model runs a CSHA kernel, and since the second pass of fixes no
NSL program can train through one: a train or grad step on CSHA-claimed
attention is a compile error (Part 5). Paths are relative to the repository
root.

**What a stock model gets.** `GroupedQueryAttention`
(`stdlib/nsl/nn/gqa.nsl`) carries no decorator. Its attention runs as a
fused FlashAttention forward launched by `nsl_sdpa_fused_forward` (64×64
tiles, `mma.sync`) and the older "v1" backward launched by
`nsl_flash_attention_backward`, both in
`crates/nsl-runtime/src/flash_attention.rs`. RMSNorm, the three projections,
RoPE, the GQA `expand().contiguous()` copy, the output projection and the
residual add are all separate kernels. `--csha` changes none of this: the
object built from `models/coder50m/pretrain.nsl` was byte-identical with and
without `--csha auto`. The compiler now says so. When a train block's
attention is not inside an `@flash_attention` method,
`invoke_csha_if_enabled` (`crates/nsl-codegen/src/stmt_pass_bridges.rs`)
prints a single `[csha] not applied` line and records the pass as declined,
not applied.

**Switches.**

- Flags: `--csha [auto|boundary|pipeline|block|off]` and `--csha-report`.
  `--pretrain-optimized` includes `--csha auto`.
- Decorators: `@flash_attention(causal=, head_dim=)`,
  `@csha(level=, target=, disable=)`, `@checkpoint`,
  `@pca(strategy=per_document)`.
- The fused training path is claimed when all of these hold:
  `@flash_attention` on a method, a train block, source AD, `--csha`, and
  an operation chain RMSNorm → Matmul(wq|wk|wv) → [RoPE] → SDPA with nothing
  in between (`scan` in `crates/nsl-codegen/src/csha_boundary.rs`). A
  reshape or transpose into heads breaks the chain, so only single-head toy
  programs match. A program that does match no longer compiles: the fused
  backward gives wrong gradients (Part 5), and `lower_single_op` in
  `crates/nsl-codegen/src/wengert_lower.rs` refuses to emit the pair
  (`CSHA_FUSED_TRAINING_WITHHELD`). `examples/csha_toy_pretrain_hd32.nsl`
  is such a program; it trains without `--csha`. The launchers
  `nsl_flash_attention_csha_with_saves` and
  `nsl_flash_attention_csha_backward` are therefore reached only from tests
  and direct FFI calls. `examples/m27_gqa.nsl` carries `@flash_attention`
  but has no fused chain. Calling an `@flash_attention` method outside a
  train block is not affected by that refusal. Such a call runs the v2
  kernel, whose launchers refuse a sequence length that is not a multiple
  of the kernel's tiles (Part 5).
- With the decorator but without `--csha`, a train block's attention lowers
  to plain matmul, softmax, matmul (`lower_single_op` in
  `crates/nsl-codegen/src/wengert_lower.rs`).

**Component by component.**

- **§2.1 Level 1 (RMSNorm prologue, RoPE epilogue) — absent.**
  `crates/nsl-codegen/src/csha_boundary.rs` only scans for the pattern. The
  training configuration built in `maybe_synthesize_csha_training_ptx`
  (`crates/nsl-codegen/src/compiler/kernel.rs`) is level 1 with every fusion
  flag off: plain FlashAttention, plus saved activations, plus a fused
  backward. The configuration with RMSNorm fused but projections not
  (`CshaExtras::level1`) is refused by `synthesize_backward_with_tier_b`
  (`crates/nsl-codegen/src/flash_attention_v2/mod.rs`). Neither side is
  right: Level 1 as proposed is sound and cheap, and it is unbuilt.
- **§2.2 Level 2 (pipelining) — partial, reachable only from tests and the
  bench binary.** `synthesize` in
  `crates/nsl-codegen/src/flash_attention_v2/tier_b1/mod.rs` (the "Tier
  B.1" kernel) is a single-iteration scaffold that handles the first
  key/value tile only, and its RMSNorm is a separate pre-pass kernel that
  writes a re-laid-out copy of `x` to HBM. Tier B.1 is selected only at
  level 2 or above (`synthesize_flash_attention_ptx_v2_with_tier_b` in
  `crates/nsl-codegen/src/flash_attention_v2/mod.rs`), and
  `maybe_synthesize_csha_training_ptx` hard-codes level 1 and clamps any
  plan that asks for more, so no training configuration can select it. For
  more than one tile, `synthesize_forward_multi_tile_combined` (same file)
  replaces the fused kernel with two launches that save Q, K and V to HBM
  and then run classic attention. The implementation is right, for the
  reason in E1.
- **§2.3 Level 3 (whole block) — absent.** The planner can print
  "L3-block"; no kernel exists.
- **§3 per-layer specialisation — analysis only.** `analyze` in
  `crates/nsl-codegen/src/csha_specialize.rs` exists, but
  `invoke_csha_if_enabled` never passes it weights and fixes the head count
  at 8, and its "entropy" (`entropy_bucket`) is the coefficient of variation
  of the heads' norms. The proposal is wrong here (E6).
- **§4.2 zero-copy GQA — partial.** The forward does not have it: the
  kernel has no grouped form and the standard library copies. The v1
  backward has a native `_gqa` kernel variant. The proposal's idea is right.
- **§4.3 attention sinks — partial.** A narrow first version: single-tile,
  non-causal, not in training (`attention_sinks_v1_eligible` in
  `crates/nsl-codegen/src/flash_attention_v2/sinks.rs`).
- **§5.1 fused backward — partial, and withheld from training.** The
  kernel is scalar ("Tier C") and uses no tensor cores. With fused projections, more than one tile
  is refused unless batch and head count are both 1. The tensor-core
  variant ("Tier B.2") requires a single active head and
  `d_model == head_dim` (`tier_b2_hybrid_backward_compile_time_eligible` in
  `crates/nsl-codegen/src/flash_attention_v2/tier_b2/dispatch.rs`), so it is
  test-only.
- **§6.1 transparent use — absent**, as described above.
- **§6.3 report — implemented, for decorated attention only**
  (`CshaPlan::render_report` in `crates/nsl-codegen/src/csha.rs`, behind
  `--csha-report`). It opens with a note that its traffic and time figures
  are a bandwidth model on a nominal layer shape, not measurements of the
  model, and that training builds emit the level-1 configuration whatever
  level is planned. Each time is bytes divided by bandwidth
  (`est_latency_us` in `crates/nsl-codegen/src/csha_pipeline.rs`), so every
  printed speedup equals its traffic ratio.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Withdraw Levels 2 and 3 as specified, from the proposal and from what the
   planner advertises: K and V must be projected once, not per query tile.
2. Build Level 1 as §2.1 describes it: one kernel that reads `x` once and
   writes Q′, K′ and V, in the grouped-KV form that makes §4.2 possible.
3. State the expected gain as single-digit to low-teens percent per layer,
   and measure against the stock path and cuDNN SDPA through PyTorch.
4. Redo §1 with a consistent roofline for the actual sm_120 target, and base
   any head pruning on the Wv/Wo path instead of Wq/Wk (E3, E6).
5. Treat a tensor-core backward for the stock path on sm_120 as research; FA4
   found the backward limited by shared-memory bandwidth, not FLOPs.

---

## Part 5 — Implementation defects found in the same review

The review also built the code and emulated its f16 gradient path
numerically; no kernel was launched on a GPU for it. This part lists what it
found wrong in the implementation, as distinct from the proposal's own
errors in Part 1. Fixed items are described in `CHANGELOG.md` under
"Unreleased". The open list is as of 2026-11-13.

The fixes after the first two were made in a second pass, which did launch
the kernels, on an RTX PRO 4500. Its measurements are relative L2 errors
against an f64 evaluation of the layer from the unrounded inputs
(`crates/nsl-codegen/tests/csha_f64_reference.rs`), which CPU tests hold to
literals from PyTorch's float64 autograd. The GPU tests named below are
`#[ignore]` and need `--features cuda`; "the fused pair" means the forward
that saves activations and the scalar fused backward.

### Fixed

- `--csha` on a model whose attention is not inside an `@flash_attention`
  method printed a fusion plan ("CSHA L2-pipeline, 6.00x speedup, Backward:
  Tier B.2"), reported "CSHA fused backward rejected" four times per layer
  and recorded the pass as applied, while compiling a byte-identical
  object. It now prints one line saying CSHA was not applied, publishes no
  fused-backward claims and records a decline, and the full report opens
  with a note that its figures are a bandwidth model on a nominal layer
  shape (`invoke_csha_if_enabled` in
  `crates/nsl-codegen/src/stmt_pass_bridges.rs`, `CshaPlan::render_report`
  in `crates/nsl-codegen/src/csha.rs`). The test is
  `crates/nsl-cli/tests/csha_decorator_free_report_gate.rs`.
- Outside CSHA's own code, the stock flash-attention backward (not a CSHA
  kernel) converted its gradient operands to f16 unscaled and underflowed;
  that was fixed separately by per-tile power-of-two scaling and is
  described in `CHANGELOG.md`.
- The CSHA backward lost small gradients to f16: it narrowed the incoming
  gradient dO to f16 without scaling and stored dQ, dK, dV, dWq, dWk and
  dWv as f16, which is exactly zero below 2.98e-8. Run on the old code with
  dO drawn uniformly in ±5e-7, dQ was 6% to 11% off, dV 8% to 16% and dWq
  4% to 8%; in ±5e-8, dQ was 82% to 89% off and dV 86% to 100%; in ±5e-10
  every gradient was zero. The kernel now reads dO as f32 and stores every
  gradient as f32, the lowering allocates f32 outputs, and the launcher
  refuses an f16 tensor in a gradient slot (`csha_backward_impl` in
  `crates/nsl-runtime/src/flash_attention.rs`, `phases/backward/` under
  `crates/nsl-codegen/src/flash_attention_v2/`, `PrimalOp::FusedCshaBackward`
  in `crates/nsl-codegen/src/wengert_lower.rs`). The error is now 5e-4 to
  2.1e-3 at every magnitude of dO, which is the f16 precision of the
  operands Q, K, V and the weights. The test is
  `crates/nsl-codegen/tests/csha_cuda_backward.rs`.
- The backward test that should have caught that could not: it drew dO of
  order 1, rounded it to f16, gave the rounded values to its CPU oracle too,
  and compared under absolute tolerances of 5e-3 to 4e-2. It passed on the
  old code. It is replaced by the gate above, which draws dO at 1, 1e-6,
  1e-7 and 1e-9, compares all eight gradients and the forward output with
  the f64 reference under a relative tolerance of 4e-3, and requires a
  2^-20 rescale of dO to rescale every output bit for bit. Run against the
  old code, the new gate fails at every magnitude below 1.
- The tensor-core backward (Tier B.2, test-only) converted dS and the dO
  tile to f16 unscaled before `mma.sync`. Run on the old code at dO in
  ±5e-7, dQ was 80% to 92% off with 93% of it exactly zero; at ±5e-8 it was
  all zero. The launcher now multiplies each (batch, head) slice of dO by
  the power of two that puts its largest element in [64, 128) as it narrows
  it, and multiplies dQ, dK and dV by the reciprocal before the projection
  kernel reads them; the dQ and dK/dV kernels saturate dS at f16's largest
  finite value (`csha_tier_b2_backward_launch` in
  `crates/nsl-runtime/src/flash_attention.rs`, `emit_ds_clamp` in
  `tier_b2/backward/dq.rs` and `dkdv.rs`). The error is now 3e-4 to 6e-4 at
  every magnitude. The test is
  `crates/nsl-codegen/tests/tier_b2_backward_f64_gate.rs`.
- With head pruning and a batch above 1, whole batch elements were skipped:
  the launch grid was `batch × active_heads`, but the kernel derives batch
  and head from the full head count. Run with batch 2, 8 heads and 4
  active, the output and all three gradients of batch element 1 were
  exactly zero under a success code, in the forward and in the backward.
  All four CSHA launch sites now launch `batch × heads` blocks and rely on
  the kernel's own guard to skip pruned heads (`nsl_flash_attention_csha`,
  `nsl_flash_attention_csha_with_saves` and both backward launchers in
  `crates/nsl-runtime/src/flash_attention.rs`). The test is
  `crates/nsl-codegen/tests/csha_cuda_training_config.rs`.
- The training forward took its active head count from the planner's
  nominal 8-head layer while the backward passed 0, and so did the launches
  in a method body, so a model with more than 8 heads would have had heads
  8 and up left at zero. Every launch now passes the count of the
  configuration its kernel was synthesized from (`emit_fused_forward_under_claim`
  and `lower_single_op` in `crates/nsl-codegen/src/wengert_lower.rs`,
  `compile_flash_attention_call` in
  `crates/nsl-codegen/src/expr/advanced.rs`). For the training launches
  this is no longer reachable, because the fused pair is withheld from
  train steps (below); what the launchers do with a count below the head
  count is covered by `crates/nsl-codegen/tests/csha_cuda_training_config.rs`.
- The result of the forward launch was discarded for
  `nsl_flash_attention_csha` and `nsl_flash_attention`, so a refused launch
  let the zero-initialised output flow on. All three forward launches in a
  method body now trap on a non-zero return code
  (`trap_on_attention_launch_failure` in
  `crates/nsl-codegen/src/expr/advanced.rs`). The test is
  `a_refused_attention_launch_traps` in
  `crates/nsl-codegen/tests/train_clif_snapshots.rs`.
- The fused-projection kernels read `x`, and the backward its saved copy
  `x_raw`, as rows `head_dim` wide while indexing them by `d_model`, and
  the save buffer is sized by `head_dim`. Run at `d_model` 64 and
  `head_dim` 32, the forward output was 126% off and every gradient between
  125% and 215%, with a success code. Both forward launchers and the
  backward now refuse a fused-projection kernel whose `d_model` exceeds
  `head_dim`, and the tensor-core backward refuses any `d_model` other than
  `head_dim` (`csha_refuses_wide_fused_projection` in
  `crates/nsl-runtime/src/flash_attention.rs`). The test is
  `fused_kernels_refuse_a_d_model_above_head_dim` in
  `crates/nsl-codegen/tests/csha_cuda_backward.rs`.
- The Tier B.1 launcher kept its re-laid-out copy of each projection weight
  in a cache keyed on the weight's device address, never invalidated or
  freed, so an in-place weight update or a reused address left the kernel
  on old weights. The cache is removed; the weights are converted on every
  launch and the scratch is freed with it (`csha_tier_b1_prepass_substitute`
  in `crates/nsl-runtime/src/flash_attention.rs`,
  `crates/nsl-runtime/src/cuda/tier_b1_prepass.rs`). The test is
  `tier_b1_rereads_weights_updated_in_place` in
  `crates/nsl-codegen/tests/tier_b1_csha_ffi_e2e.rs`.
- The Tier B.1 pre-passes handle one batch element and one head, and a
  larger launch went ahead. The launcher now refuses a batch or head count
  above 1 for a Tier B.1 kernel, and refuses instead of launching the
  kernel on buffers the pre-pass did not prepare. The test is
  `tier_b1_refuses_batch_or_heads_above_one` in the same file.
- A train step could go through the fused pair although its gradients do
  not match a reference on the one configuration a training build emits
  (level 1, no fusion flag): at two tiles the forward output is right and
  dQ, dK and dV are 130%, 45% and 43% off; with more than one head the
  train step stopped on a tensor shape mismatch; and one head with one tile
  was never shown right. A train or grad step on CSHA-claimed attention is
  now a compile error that names `@flash_attention` and `--csha`, says the
  fused backward is withheld because its gradients do not match the
  reference, and says to compile without `--csha`
  (`CSHA_FUSED_TRAINING_WITHHELD` in
  `crates/nsl-codegen/src/wengert_lower.rs`). Without `--csha` the same
  program trains on the unfused attention; run on a GPU, every statistic
  it prints agrees with PyTorch to 2e-6. The tests are
  `crates/nsl-codegen/tests/csha_training_withheld.rs` (the refusal fires
  for a train block and not for a forward-only use) and
  `crates/nsl-cli/tests/csha_toy_pretrain_gate.rs`. The defects behind the
  refusal are still in the kernels and are the open items below.
- `examples/csha_toy_pretrain_hd32.nsl` could not run (its `x` was
  two-dimensional, which the fused forward aborts on) and was degenerate:
  `x` and every weight were ones and the loss was `sum(out)`, so dQ, dK,
  dWq and dWk were exactly zero. Its only test compiled it and looked at
  relocation symbols, and two GPU tests in `nsl-cli` compared the fused
  path with the unfused one under a loose tolerance. The example is
  replaced by a program with non-uniform inputs, a gradient of order 1e-6
  at the attention output and printed statistics of each weight's update;
  `crates/nsl-cli/tests/csha_toy_pretrain_gate.rs` runs it on a GPU without
  `--csha` and asserts every statistic against PyTorch.
- The scalar backward reads Wq, Wk and Wv without a null check whenever its
  configuration has a nonzero `d_model`, so a launch with a null weight was
  a GPU fault (CUDA_ERROR_ILLEGAL_ADDRESS, seen in a run) surfacing at a
  later driver call. The launcher now refuses such a launch
  (`csha_backward_impl` in `crates/nsl-runtime/src/flash_attention.rs`).
  The test is `backward_refuses_a_null_projection_weight` in
  `crates/nsl-codegen/tests/csha_cuda_backward.rs`.
- The v2 attention kernels have no tail guard on the sequence: a CTA
  stores all `block_q` of its query rows. An `@flash_attention` method
  called at a sequence of 32 under the default 64-row tile wrote 32 rows
  past the end of its output, into the 2 KB that followed, on every call.
  In `examples/csha_toy_pretrain_hd32.nsl` with a call of the method before
  the train block, the tensor overwritten was the program's copy of Wq,
  and the Wq update it printed had a squared norm of 17.2 against
  PyTorch's 0.0061 while the loss and every other statistic agreed. The
  weight and its gradient were right; dumping Wq around the call showed it
  unchanged, and a run with a learning rate of 0 reported exactly the
  squared norm of the first half of the copy. The same happened without
  `--csha` and in tape AD, where the train step itself runs the kernel. The
  plain, CSHA and CSHA-with-saves forward launchers and the CSHA backward
  launcher now refuse a sequence that is not a multiple of `block_q` and
  `block_kv` (`fa_refuses_ragged_seq` in
  `crates/nsl-runtime/src/flash_attention.rs`; the per-document kernels,
  which bound rows by the document, are not checked), and the generated
  code stops on the refusal. The Tier B.2 branch of the backward returns
  before the scalar path's check and has its own, against `block_q` and
  the tile its dq and dkdv kernels use; compiled NSL selects that branch
  only when the sequence is one tile, but a direct FFI call could pass a
  longer one. The packed source-AD channel already declined
  such shapes. The tests are
  `a_sequence_that_does_not_fill_whole_tiles_is_ragged` in that file,
  `tier_b2_backward_refuses_a_sequence_that_does_not_fill_whole_tiles` in
  `crates/nsl-codegen/tests/tier_b2_backward_f64_gate.rs` (refused at 96
  rows under a 64-row tile with no output touched, not refused at 128), and
  `crates/nsl-cli/tests/flash_attention_ragged_seq_gate.rs`, which checks
  the refusal at a sequence of 32 and the output against attention written
  out by hand at 64 and 128. The same example at a sequence of 64, one
  whole tile, trained in tape AD through the decorated kernel agrees with
  PyTorch to 1e-4 of each statistic, the f16 rounding of the kernel's
  output, and prints the same numbers with and without `--csha`: tape AD
  does not use the CSHA kernels.

### Open

The next three are the reasons the fused pair is withheld from train steps.
They can no longer be reached from an NSL program, only from tests and
direct FFI calls, and all three have to be fixed and gated from NSL source
before `CSHA_FUSED_TRAINING_WITHHELD` is lifted.

- On the configuration a training build emits (level 1, no fusion flag),
  the fused backward is wrong at more than one tile. At two tiles the
  forward output is right (3e-4) and dQ, dK and dV are 130%, 45% and 43%
  off, at batch 1 with one head as at batch 2 with four; at one tile the
  same shapes are right (4e-4) for dQ, dK and dV. With fused projections
  more than one tile is right at batch 1 with one head and refused
  otherwise. Where: the per-query-tile launch loop of `csha_backward_impl`
  in `crates/nsl-runtime/src/flash_attention.rs` and the kernel it
  launches.
- The fused pair does not handle more than one head from NSL source. With
  `x` of shape `[2, 16, 32, 32]` the train step stopped with "tensor shape
  mismatch in elementwise op" (`[1, 2, 32, 32]` against `[2, 16, 32, 32]`).
  Suspected, by reading: the gradient with respect to `x` that the fused
  backward returns is allocated as `[batch, seq, d_model]`, with no head
  dimension (`PrimalOp::FusedCshaBackward` in
  `crates/nsl-codegen/src/wengert_lower.rs`). In the same program a direct
  call of the method returned NaN; not investigated.
- The weight gradients of that configuration (dWq, dWk, dWv and the
  gradient with respect to `x`) have never been compared with a reference.
  The FFI-level gates cover dQ, dK and dV for it, and all eight gradients
  only for the fused-projection configuration. The one end-to-end run, at
  one head and one tile, had the squared norms of the Wk and Wv updates
  12% high against PyTorch, but that program also called the method at a
  sequence of 32 under a 64-row tile, which wrote past the attention
  output (Fixed above), so the figure cannot be attributed. By reading, that
  configuration is handed `x` and the weights although no fusion flag is
  set, runs the forward's RMSNorm prologue on an `x` that is already
  normalised, and runs `emit_dproj` and `emit_drmsnorm`
  (`crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs`)
  in the backward. Next step: launch that configuration at the FFI level
  as the lowering does and compare each of the eight outputs with the f64
  reference.
- `@flash_attention` applies to the whole module, not to the function it
  decorates. The first decorated function or method found sets one
  kernel context for the module (`compile_flash_attention_kernels` in
  `crates/nsl-codegen/src/compiler/kernel.rs`), and every
  `scaled_dot_product_attention` compiled outside source AD then checks
  only that the context exists (`crates/nsl-codegen/src/expr/calls.rs`,
  whose comment says it checks the enclosing function), so an undecorated
  function's attention runs the decorated kernel, with the decorator's
  head size and tiles. By reading the code only. Right: keep the context
  per decorated function, and lower other attention as if there were no
  decorator.
- `nsl_flash_attention_quantized` launches the kernel its caller passes
  with `grid_x = ceil(seq_len / block_q)` and no check of the sequence. Nothing
  calls it: it is exported in `crates/nsl-abi/src/table.rs` and no emitter,
  runtime function, stdlib module or test refers to it. It is left without
  the refusal because it serves a paged, quantized KV cache, whose query
  length in decoding need not be a whole tile; whoever wires it up has to
  decide what its tail does.
- For decorated attention the plan is still made on a nominal layer shape
  (batch 1, sequence 1024, `d_model` 512, 8 heads, 4 KV heads), the level-1
  saving is the constant 0.58, and the kernel named in the report is not
  one that is emitted. Where: `invoke_csha_if_enabled` in
  `crates/nsl-codegen/src/stmt_pass_bridges.rs`. By reading the code only.
  Right: plan from the model's dimensions; name only emitted kernels.
