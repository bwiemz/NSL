# Compiler-Synthesized Holistic Attention (CSHA) — Errata

**Companion to** `NSL-CSHA-Research.PDF`. **Date:** 2026-11-13.

This file lists the statements in the CSHA proposal (April 2026) that should
not be relied on, and what is true instead. The proposal was reviewed in
November 2026 against its own arithmetic, the literature it cites, and the
code in this repository that implements it. Arithmetic corrections were
recomputed from the proposal's own figures. External facts carry a link where
they were checked against a source, and say so where they were not. The
statements about the code come from reading it and from compile-only builds;
no kernel was run on a GPU for this review.

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

No shipped model runs a CSHA kernel. Paths are relative to the repository
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
- The fused path needs all of: `@flash_attention` on a method, a train
  block, source AD, `--csha`, and an operation chain RMSNorm →
  Matmul(wq|wk|wv) → [RoPE] → SDPA with nothing in between (`scan` in
  `crates/nsl-codegen/src/csha_boundary.rs`). A reshape or transpose into
  heads breaks the chain, so only single-head toy programs match.
  `examples/csha_toy_pretrain_hd32.nsl` is the only program in `models/` or
  `examples/` that reaches the CSHA launchers
  `nsl_flash_attention_csha_with_saves` and
  `nsl_flash_attention_csha_backward`. `examples/m27_gqa.nsl` carries
  `@flash_attention` but has no fused chain.
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
- **§5.1 fused backward — partial.** The kernel that runs is scalar ("Tier
  C") and uses no tensor cores. With fused projections, more than one tile
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
numerically; no kernel was launched on a GPU. This part lists what it found
wrong in the implementation, as distinct from the proposal's own errors in
Part 1. Fixed items are described in `CHANGELOG.md` under "Unreleased". The
open list is as of 2026-11-13.

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
  described in `CHANGELOG.md`, while the CSHA kernels' own f16 gradient
  handling is not fixed and is the first open item below.

### Open

- The CSHA backward loses small gradients to f16: dO is narrowed unscaled,
  and dQ, dK, dV, dWq, dWk and dWv are stored as f16, which is zero below
  2.98e-8. Emulated at rms(dO) = 1e-6, dQ has 57% relative error with 80%
  exact zeros, dV 12% and dWq 4.2%; at 1e-7 dQ is all zero. Where:
  `csha_backward_impl` in `crates/nsl-runtime/src/flash_attention.rs`.
  Numeric check; no GPU run. Right: f32 from end to end.
- The tensor-core backward (Tier B.2, test-only) converts dS and the dO
  tile to f16 unscaled before `mma.sync`, as the stock backward did before
  its fix. Where: `emit_ds_scatter_to_smem` in `tier_b2/backward/dq.rs` and
  `emit_dscol_scatter` in `tier_b2/backward/dkdv.rs`, both under
  `crates/nsl-codegen/src/flash_attention_v2/`. By reading the code only.
  Right: the same power-of-two scaling.
- With head pruning and a batch above 1, whole batch elements are suspected
  to be skipped: the launch grid is `batch × active_heads`, but the kernel
  derives batch and head from the full head count. Batch 2, 8 heads, 4
  active would leave batch 1 zero with a success code. Where:
  `nsl_flash_attention_csha` in `crates/nsl-runtime/src/flash_attention.rs`.
  By reading the code only. Right: launch the full `batch × heads` grid.
- The training forward takes its active head count from the planner's
  nominal 8 while the backward passes 0, so a claimed model with more than 8
  heads would compute only 8 in the forward. Where:
  `emit_fused_forward_under_claim` and `lower_single_op` in
  `crates/nsl-codegen/src/wengert_lower.rs`. By reading the code only.
- The result of the forward launch is discarded for
  `nsl_flash_attention_csha` and `nsl_flash_attention`. The launcher can
  refuse with -1, and the zero-initialised output then flows on. Where:
  `compile_flash_attention_call` in
  `crates/nsl-codegen/src/expr/advanced.rs`. By reading the code only.
  Right: trap, as the path that saves activations already does.
- The saved `x_raw` buffer is sized `batch·heads·seq·head_dim` f32 but is
  indexed as `[seq, d_model]` by the backward and its post-pass; it is
  suspected to be read out of bounds whenever `d_model > heads·head_dim`.
  Where: `nsl_csha_alloc_backward_activations` in
  `crates/nsl-runtime/src/flash_attention.rs`. By reading the code only.
- The Tier B.1 weight cache is keyed on the device address and is never
  invalidated or freed. It goes stale after an in-place weight write, or
  when a freed address is reused by another tensor of the same shape. It is
  reachable only from tests, the bench binary and direct FFI calls. Where:
  `w_chunkified_cached` in `crates/nsl-runtime/src/cuda/tier_b1_prepass.rs`.
  By reading the code only.
- The Tier B.1 pre-passes handle one batch element and one head's width
  (the `x` pre-pass has no batch dimension), while the planner advertises
  Tier B.1 for an 8-head, grouped-KV model. Where:
  `csha_tier_b1_prepass_substitute` in
  `crates/nsl-runtime/src/flash_attention.rs`. By reading the code only.
  Right: refuse batch or head counts above 1 in the launcher.
- For decorated attention the plan is still made on a nominal layer shape
  (batch 1, sequence 1024, `d_model` 512, 8 heads, 4 KV heads), the level-1
  saving is the constant 0.58, and the kernel named in the report is not
  one that is emitted. Where: `invoke_csha_if_enabled` in
  `crates/nsl-codegen/src/stmt_pass_bridges.rs`. By reading the code only.
  Right: plan from the model's dimensions; name only emitted kernels.
- The backward test that would catch the f16 loss cannot. It draws dO of
  order 1, rounds it to f16 and gives the rounded values to the CPU
  reference, so both sides round alike at a magnitude where f16 is
  harmless, under absolute tolerances of 1e-2. Where:
  `run_fused_backward_config_seq` in
  `crates/nsl-codegen/tests/csha_cuda_backward.rs`. By reading the code only.
- The toy pretraining test asserts relocation symbols only, and its example
  is degenerate: `x` and every weight are ones and the loss is `sum(out)`,
  so dQ, dK, dWq and dWk are exactly zero and any path for them passes.
  Where: `crates/nsl-codegen/tests/csha_gap_f_toy_pretrain_smoke.rs` and
  `examples/csha_toy_pretrain_hd32.nsl`. By reading the code only.
