# CFTP — Errata

**Companion to** `CFTP.pdf`. **Date:** 2026-11-13.

`CFTP.pdf` ("Compiler-Fused Training Pipeline", April 2026) proposes two
techniques. FASE (Fused Accumulation-Step Elimination) folds gradient
accumulation and the optimizer step into the backward pass. PCA
(Pack-Compiled Attention) builds the document boundaries of packed sequences
into the attention kernel. The PDF cannot be edited, so this file records
which of its statements should not be relied on and what is true instead. The
proposal was reviewed in November 2026 against its own arithmetic, the
literature, and the code in this repository. Every figure below was
recomputed and every quotation was checked against the PDF. Each Source line
says whether an external source was read directly, seen only in a
search-result summary, or not re-checked. Statements about the code come from
reading it and from small CPU runs; nothing was run on a GPU. Section numbers
are the PDF's.

---

## Part 1 — Errors in the proposal

### FASE

#### E1 — FASE-Deferred saves no peak memory: `m_partial` is the gradient buffer

**Proposal.** Abstract: "eliminating gradient buffers entirely". §2.4: "The
savings come from never holding ALL gradients simultaneously — only one
layer's gradient at a time during the fused step." §5 table: "FASE: No
gradient buffer (peak)", "~200 MB peak". §6.3: "Gradient buffer: ELIMINATED
(peak reduced by ~200MB)".

**Correction.** `m_partial` is a full parameter-sized buffer that holds the
running mean gradient for the whole accumulation window. It is the gradient
buffer under another name, and §2.3 concedes that the totals are equal:
"Wait — the total is the same? Yes, because m_partial is the same size as the
gradient buffer."

The peak difference that §2.4 then claims rests on a "Standard" timeline that
is not how PyTorch behaves. That timeline holds each micro-batch's gradients
side by side ("[grads₀ + grads₁ building]") and counts the last activations
into the optimizer-step peak ("Params + Full Grads + Optimizer + Last
Activations"). PyTorch adds each leaf gradient into `.grad` in place as the
backward pass proceeds, so the only transient is one layer's gradient, and
activations are freed before `optimizer.step()`. Both pipelines therefore
peak at parameters + one parameter-sized accumulator + m + v + one transient
layer gradient. The saving is zero, not ~200 MB.

A parameter-sized buffer can be saved in only two ways: step immediately with
no accumulation (N = 1), or accumulate straight into m and v as in Option B
of §2.2, which changes the numerics (E3). FASE-Deferred does neither.

**Source.** The proposal's own §2.3 table and §2.4 timelines. The description
of PyTorch's in-place `.grad` accumulation was not re-checked against
PyTorch's documentation.

#### E2 — PyTorch's hook can accumulate; the quotation is optimi's, truncated

**Proposal.** Abstract: "(something PyTorch's
register_post_accumulate_grad_hook explicitly cannot do)". §1: "as documented
by PyTorch itself and the optimī library: "features which rely on persistent
gradients like gradient accumulation won't work." The fundamental tension:
you can fuse backward+optimizer OR you can accumulate gradients, but not
both." §2.6: the hook "fires after every backward, with no way to know
whether this is micro-batch 0 or N-1". §2.2, Option A: "This is what
PyTorch's hook approach does and why it's incompatible."

**Correction.**

- The quoted sentence is from optimi's gradient-release page, not from
  PyTorch, and the words "AMP's GradScaler, gradient clipping, or" have been
  cut from the middle of it.
- PyTorch's tutorial says only that the technique applies "(if gradients take
  up a portion of your memory and you do not need to do gradient
  accumulation)". It does not say the hook cannot accumulate.
- The hook is a Python closure and can read a counter. optimi's "optimizer
  accumulation" does exactly that, with
  `opt.optimizer_accumulation = (idx+1) % accumulation_steps != 0`, fusing
  accumulation into the hook. It needs only PyTorch 2.1 or later. So "but not
  both" was already false when the proposal was written.
- FASE-Deferred itself could be written as a short hook: add `g/N` to a
  per-parameter accumulator, clear `p.grad`, and run AdamW on the Nth call.
- Option A misdescribes the tutorial's hook, which keeps no accumulator and
  steps immediately.

**Source.** Read directly:
https://optimi.benjaminwarner.dev/gradient_release/,
https://optimi.benjaminwarner.dev/optimizer_accumulation/ and
https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html.

#### E3 — The two second-moment formulas in §2.2 differ by a factor of N

**Proposal.** §2.2 defines the running accumulator as "v_partial_i =
v_partial_{i-1} + (1/N²) × gᵢ²". Option B then uses "v_t ≈ β₂ v_{t-1} +
(1-β₂) × (1/N) Σ gᵢ²" and says it "actually provides a richer signal than the
standard computation".

**Correction.** The accumulator ends the window at (1/N²)·Σgᵢ². Option B uses
(1/N)·Σgᵢ². These are different quantities, a factor of N apart. Neither
equals the standard (mean g)²:

- (1/N)·Σgᵢ² − ((1/N)·Σgᵢ)² = Var(gᵢ) ≥ 0, so Option B always inflates v and
  shrinks the step, by up to √N when micro-batch noise dominates the
  gradient.
- The 1/N² form deflates v by up to a factor of N when the micro-batch
  gradients agree.

With N = 4: for zero-mean noise, Option B's v is 4.0 times the standard value
and the step is halved; for four identical gradients g = 3, the 1/N² form
gives 2.25 where the standard value is 9.

The exact form in the §2.3 pseudo-code, "v_k = β₂ * v_k + (1-β₂) * g_acc_k²",
is the correct one, and it is what the code implements (Part 3).

**Source.** Algebra, plus a numeric check of both cases.

#### E4 — The stability claim for Option B is unsupported

**Proposal.** §2.2: "Research on gradient noise scale shows this is often
superior for training stability."

**Correction.** No citation is given. The nearest published work is AdamA
(Zhang et al., arXiv 2305.19982), which integrates gradients directly into
the optimizer state and reports up to 23% memory reduction; the proposal does
not cite it. The claim is unsupported; it has not been shown false.

**Source.** https://arxiv.org/abs/2305.19982 (title, authors and abstract
read directly).

#### E5 — The peak-VRAM and throughput figures in §6.3 are not derived

**Proposal.** §6.3: "Estimated peak VRAM: 4.2 GB (vs 5.1 GB without
FASE+PCA)" and "Estimated training throughput: +35% vs standard pipeline".

**Correction.** 5.1 − 4.2 = 0.9 GB. The memory entries of the §5 table sum to
512 MB + ~200 MB + 128 KB ≈ 712 MB, which is also the total that table
prints, so the 0.9 GB is not the sum of the savings the proposal lists. Both
large entries are themselves wrong: the ~200 MB is zero (E1) and the 512 MB
is at most 128 MB (E11). The +35% is derived nowhere in the proposal; the
compute saving that can be derived from its example is about 6% of layer
FLOPs (E10).

**Source.** Recomputed from the proposal's own figures.

#### E6 — "N-1 gradient HBM round-trips" saved is unsupported

**Proposal.** §5 table: "FASE: Per-layer fused optimizer" saves "N-1 gradient
HBM round-trips".

**Correction.** Every micro-batch still writes each gradient into
`m_partial`, so no per-micro-batch gradient write is removed, and the
proposal gives no count that would support the figure. Separately, the
proposal calls GPU memory HBM (high-bandwidth memory) throughout, but the
target card named in §6.3, the RTX 5070 Ti, has GDDR7 memory, not HBM (not
re-checked against the source).

#### E7 — Reference [1] carries the arXiv number of a different paper

**Proposal.** References: "Pudipeddi, B., et al. "Training Large Neural
Networks with Constant Memory using a New Execution Algorithm." arXiv
2104.00237, 2021."

**Correction.** That paper is arXiv 2002.05645 (February 2020). arXiv
2104.00237 is "Optimizer Fusion: Efficient Training with Better Locality and
Parallelism" (Jiang et al.). It is the actual prior art for fusing the
optimizer into the backward pass, and the proposal does not cite it under its
own name.

**Source.** https://arxiv.org/abs/2002.05645 and
https://arxiv.org/abs/2104.00237 (both read directly).

### PCA

#### E8 — A tile-skip map cannot be a compile-time constant

**Proposal.** §4.2, Strategy 2: "The compiler generates a tile-skip map as a
compile-time constant embedded in the kernel."

**Correction.** Which tiles are fully cross-document depends on where the
document boundaries fall, and that is per-batch data. No map fixed at compile
time can be right for the next batch. §4.6 concedes the point: the compiler
knows "not the specific documents, but the packing pattern". Skip decisions
have to be made at run time from each batch's segment ids, which is what the
code does (Part 3).

#### E9 — The example's tile-skip fraction is 50% (44% causal), not ~55%

**Proposal.** §4.2: "With tile-level skipping, this becomes ~55% of
tile-pairs skipped entirely (no matmul computed)", repeated in §5 and §6.3.
The tile map in the same section reads "Q-tile 3 (Doc A tokens 384-399 + Doc
B tokens 0-15): attend to KV-tiles 0-4".

**Correction.** With 1024 tokens and 128-token tiles there are 8 tiles and 64
tile pairs. For documents of 400, 300 and 324 tokens, and equally for three
documents of about 340, 32 pairs contain at least one same-document cell, so
32 of 64 are skipped: 50%.

That figure counts the upper triangle, which causal FlashAttention already
skips. Against that baseline there are 36 causal tile pairs, of which 16 are
skipped: 44%.

The tile map has two slips. Q-tile 3 covers positions 384–511, so it holds
Doc A tokens 384–399 and Doc B tokens 0–111, not 0–15. It must attend to
KV-tiles 0–3 under a causal mask, or 0–5 without one; "0-4" is neither.

**Source.** Recomputed by enumerating the tile pairs for both layouts.

#### E10 — Tile skipping saves about 6% of layer FLOPs, not ~50% or +35%

**Proposal.** §5 table: "~50% of attention FLOPs". §6.3: "Estimated training
throughput: +35% vs standard pipeline".

**Correction.** Skipping tile pairs shrinks only the score and value products
(QKᵀ and PV). The projections and the feed-forward network are untouched.
Take S = 1024 and a model width of d = 512; the width is an assumption, as
the proposal does not state one. Per token per layer:

- projections plus feed-forward network: about 12d² ≈ 3.1M multiply-adds;
- causal QKᵀ plus PV: about S·d ≈ 0.52M multiply-adds.

The skippable part is therefore about 14% of layer FLOPs, and skipping 44% of
it (E9) saves about 6%. Neither of the proposal's figures follows from its
own example.

**Source.** Recomputed from the proposal's example and the assumed width.

#### E11 — The 512 MB mask figure is for length 2048; the example uses 1024

**Proposal.** §3: "For seq_len=2048: 4M entries × 4 bytes = 16MB per batch
element", which gives 512 MB at batch size 32. §5 table: "PCA: No attention
mask tensor", "512 MB". §6.3: "Mask memory saved: 512 MB (no S×S mask
tensor)".

**Correction.** The configuration used in §4.1, §5 and §6.2 is
`sequence_length = 1024` with batch size 32. 1024² × 4 bytes × 32 = 128 MB as
32-bit floats, or 32 MB as booleans. §4.2 itself gives 4 MB per batch element
at this length.

The baseline is also one that nobody pays. NSL's own unpacked loader does not
build a per-batch S×S mask (see the note in `build_simple_batch`,
`crates/nsl-runtime/src/dataloader.rs`), and FlexAttention materialises none
(E12). The example mask in §3 is drawn as full blocks, although the model is
causal and its mask is lower-triangular within each block.

**Source.** Recomputed from the proposal's own figures.

#### E12 — FlexAttention already skips masked blocks and masks by document

**Proposal.** §4.6: FlexAttention "still evaluates the mask at runtime
per-tile. It cannot skip entire tile-pairs because it doesn't know the
document layout until the batch is constructed", and "PyTorch cannot generate
tile-skip maps because the mask is a dense tensor."

**Correction.** FlexAttention's `BlockMask` is a block-sparsity map
(`num_blocks`, `indices`, block size 128) computed per batch from the user's
`mask_mod` function. It skips fully masked blocks, applies the mask only on
partial blocks, and does not materialise an S×S tensor. Its announcement uses
document masking, `document_id[q_idx] == document_id[kv_idx]`, with a
per-document position transform, as a headline example, and cites a 71%
throughput gain from sample packing in torchtune. FlexAttention shipped in
PyTorch 2.5 (2024). It is prior art for Strategies 1 and 2 and predates the
proposal.

**Source.** https://pytorch.org/blog/flexattention/ (read directly).

#### E13 — RoPE does not need a per-document position reset

**Proposal.** §4.3: "Packed sequences require per-document position IDs
(positions reset at document boundaries)."

**Correction.** A RoPE (rotary position embedding) attention score depends
only on `pos_q − pos_k`. Once attention is restricted to same-document pairs,
subtracting the document's start from both positions leaves every score
unchanged.

Numerically, a 324-token document was scored at positions 0–323 and again at
700–1023 with head dimension 64. The largest score difference was 4.5e-4 in
32-bit floats, on scores of typical size 8, and 1e-12 in 64-bit floats. A
short CPU training run of the packed-attention fixture, once with
per-document positions and once with continuous positions, gave the same
losses to within 32-bit rounding.

The position fusion of §4.3, and the "PCA: No position_id tensor" and "PCA:
Fused RoPE position reset" rows of the §5 table, therefore buy nothing for a
RoPE model. A reset would matter only for a model with absolute position
embeddings.

**Source.** Numeric check in NumPy, and the training run described.

#### E14 — NSL's packing removes no padding; it cuts a token stream into rows

**Proposal.** §3: "Sequence packing concatenates multiple short documents
into one long sequence to eliminate padding waste."

**Correction.** That sentence describes bin-packing of documents. NSL's
`packing=true` cuts a contiguous token stream into fixed-length rows
(`pack_batch` in `crates/nsl-runtime/src/packing.rs`). There was never any
padding to remove. What packing adds in this system is only the restriction
that attention must not cross a separator, so the padding-waste motivation
does not apply to it.

#### E15 — Reference [6] is an arXiv preprint, not an ACL 2022 paper

**Proposal.** References: "Krell, M., et al. "Efficient Sequence Packing
without Cross-contamination." ACL 2022."

**Correction.** The paper is arXiv 2107.02027 (version 2, October 2022), and
its arXiv record lists no journal or conference. References [3], [4] and [5]
were not re-checked.

**Source.** https://arxiv.org/abs/2107.02027 (read directly).

---

## Part 2 — Superseded by later work

Some of this work predates the proposal and was missed by it. It is listed
here because it changes what the design should be compared with.

### FASE

- optimi, gradient release and optimizer accumulation:
  https://optimi.benjaminwarner.dev/gradient_release/ and
  https://optimi.benjaminwarner.dev/optimizer_accumulation/ (read directly).
  This is accumulation fused into the post-accumulate hook, with no gradient
  buffer, in stock PyTorch. It accepts an approximate second moment in
  exchange for dropping the buffer. FASE's exact Deferred mode cannot drop
  the buffer (E1).
- AdamA (Zhang et al.): https://arxiv.org/abs/2305.19982 (abstract read
  directly). It integrates gradients into the optimizer state and reports up
  to 23% memory reduction, also with an approximate second moment (that
  detail was not re-checked against the paper). If the goal of FASE is
  memory, an AdamA-style mode is the only route for N > 1.
- StableAdamW in optimi. optimi's suggested answer to clipping under gradient
  release is update clipping, which needs no global norm (not re-checked
  against the source). It would let a fused step with no accumulation work
  without the two-phase pass of §2.5.
- No hardware change was found that affects FASE.

### PCA

- `torch.nn.attention.varlen.varlen_attn`:
  https://docs.pytorch.org/docs/2.15/nn.attention.varlen.html and
  https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html
  (from search-result summaries; the pages were not read). PyTorch now ships
  packed variable-length attention driven by cumulative sequence offsets,
  with no mask. The summaries list PyTorch 2.10.0.dev or later for the
  tutorial, grouped-query and sliding-window support in the 2.12–2.13
  documentation, and an out-parameter variant and a cuDNN backend in 2.15.
  This is Strategy 3 as a stock API. flash-attn's `flash_attn_varlen_func`
  is older still (not re-checked against the source).
- Consequence. With FlexAttention (E12) and `varlen_attn`, §4.6 "Why PyTorch
  Cannot Do This" no longer holds for any of the three strategies. The
  baseline for any PCA speed claim should be FlexAttention with document
  masking and `varlen_attn`, not a dense S×S mask.

---

## Part 3 — What the code implements

### FASE

FASE is switched on by `grad_accumulation = N` with N > 1 in `train(...)`,
with no annotation, as §6.1 says. The decorator
`@fase(mode = auto|deferred|full_buffer|off, allow_v_approx = bool)`
overrides the planner; `allow_v_approx = false` selects FullBuffer instead of
Deferred. With N = 1 the plan is `Passthrough` and no fused step is
generated.

- Planner (§7.1, "FASE pass"): implemented, in
  `crates/nsl-codegen/src/fase.rs`. AdamW, Adam and plain SGD get Deferred.
  Lion, SOAP, Muon and unknown optimizers get the FullBuffer fallback, so
  §7.1's "per-layer AdamW/SGD/Lion step code" does not hold for Lion.
- Accumulation `m_partial += (1/N)·g` (§2.3): implemented, in
  `fase_emit_accumulate` (`crates/nsl-codegen/src/stmt_fase.rs`).
- "g_k is immediately freed" (§2.3): partial. It is true only under
  `--source-ad`. Tape AD is the default for `nsl run`, and under it the whole
  gradient list of a micro-batch is materialised and then added. The shipped
  pretrain scripts do pass `--source-ad`.
- Optimizer step fused into the final backward, layer by layer (§2.3, §2.4,
  §5): absent. Every micro-batch accumulates, and a separate optimizer pass
  runs after the last backward: one fused AdamW kernel per parameter, or one
  multi-tensor launch for all of them (`crates/nsl-runtime/src/fase_step.rs`,
  `crates/nsl-kir/src/kernels/optim.rs`). The difference costs no memory,
  because per-layer fusion saves none (E1). The `--layerwise-accum` flag
  (CSLA, compiler-scheduled layerwise accumulation) is closer to the
  proposal's shape but refuses clipping; it was not examined.
- Second moment: implemented as the exact form of §2.3,
  `v = β₂v + (1−β₂)(mean g)²`, not Option B. Bias correction counts optimizer
  steps, weight decay is algebraically PyTorch's decoupled form, and windows
  carry across epoch boundaries. On a small fixture (N = 4, two windows, with
  and without clipping, tape and source AD) the trained weights matched
  PyTorch's AdamW with standard accumulation to every printed digit.
  `crates/nsl-codegen/tests/fase_numerical_validation.rs` compares trained
  weights with independent references, including a fixture that separates
  (mean g)² from mean(g²).
- Two-phase clipping (§2.5): implemented, in
  `crates/nsl-codegen/src/stmt_train/optimizer_step.rs`, as a global-norm
  clip of the window mean, `min(1, τ/(‖mean g‖ + 1e-6))`, the same as
  PyTorch's `clip_grad_norm_`. The module §7.1 plans for it,
  `crates/nsl-codegen/src/fase_clip.rs` (`ClipPlan`, `ClipProgram`), has no
  consumer.
- Memory schedule and M36 planner integration (§2.4, §7.1): absent.
  `crates/nsl-codegen/src/fase_memory.rs` is read only by the training report
  (`crates/nsl-codegen/src/training_report.rs`), where `FaseSection::memory`
  is always `None`.
- In use: `models/coder500m/pretrain_prod.nsl`,
  `models/coder1b/pilot_finish.nsl`, `models/coder-rl/pretrain.nsl` and
  others, with `grad_accumulation` of 4 or 8, mostly with `grad_clip = 1.0`.

### PCA

Two separate mechanisms exist, and the one the proposal describes is not the
one that works.

- The proposal's surface (§4.1, §6.2), a `dataset` block with
  `packing = true`: partial. The block is planner metadata only; code
  generation skips it and it never becomes a loader. `packing = true` there
  makes the `@flash_attention` and CSHA kernels (the attention-fusion kernels
  of the companion proposal) segment-masked, fed from a thread-local registry
  (`crates/nsl-runtime/src/pca_rope_runtime.rs`). `pack_separator` in the
  block is accepted by the checker and read by nothing:
  `extract_dataset_packing` (`crates/nsl-codegen/src/pca_activation.rs`)
  reads only `separator_token_id`, which is itself never consumed. No program
  in the repository runs this route end to end with a packing loader.
- The working route: `DataLoader(..., packing=true, pack_separator=id)`, with
  the model calling `GroupedQueryAttention.forward_packed` on `x`,
  `batch.segment_ids`, `batch.position_ids` and the training flag
  (`stdlib/nsl/nn/gqa.nsl`). That lowers to
  `scaled_dot_product_attention_packed`. §6.2 says PCA activates
  automatically; this route does not. The user must call `forward_packed`
  and pass `--source-ad`.
- Strategy 1, segment-id kernel: implemented, forward and backward. The mask
  is `(k > q) OR seg[q] != seg[k]` on absolute positions, so documents that
  straddle tiles are handled per element. The fused kernels run only with all
  of: a GPU, `--source-ad`, head dimension 32, 64 or 128, a sequence length
  that is a multiple of the tile size, and contiguous Q, K and V
  (`nsl_sdpa_fused_forward` in `crates/nsl-runtime/src/flash_attention.rs`).
  Otherwise, and always under the default tape AD, attention falls back to a
  dense `[b,1,s,s]` float mask built in a host loop on every attention call
  (`nsl_packed_mask_from_segment_ids` in `crates/nsl-runtime/src/packing.rs`),
  which is the cost §3 set out to remove. Masking correctness was established
  by reading the kernels, not by a GPU run.
- Strategy 2, compile-time tile-skip map: absent, and the implementation is
  the side that is right (E8). `pca_tileskip::build` and
  `pca_segment::plan_kernel` have no production caller. In their place the
  kernels skip tiles at run time from a per-tile segment-range table held in
  shared memory (`crates/nsl-codegen/src/pca_tilerange.rs`). The skips are
  sound for non-decreasing segment ids, which are validated at launch.
- Strategy 3, one CTA (CUDA thread block) per document: partial. Kernels
  exist (`crates/nsl-codegen/src/flash_attention_v2/per_doc_cta.rs`,
  `crates/nsl-codegen/src/pca_per_doc.rs`) but are not reachable from a
  program. This was established by reading the call sites, not by a run.
- Position fusion (§4.3, "no position_id tensor ever exists"): absent on the
  working route, where `position_ids` is a `[B,S]` float tensor passed to
  `rope.forward_with_positions`. An in-kernel reset from document starts
  exists only on the CSHA route. The difference is moot (E13).
- Loss masking (§4.4): done with labels of −100 and the `cross_entropy`
  ignore value. No separator skipping was found in the fused cross-entropy
  code (searched, not read in full). This is fine as it is; §4.4 itself calls
  the saving "tiny".
- In use: none of the pretrain scripts under `models/coder*/` uses packing.
  Only `models/benchmarks/wggo_all6_pretrain.nsl` and test fixtures do, on
  synthetic token streams.
- Tests: `crates/nsl-cli/tests/stage_c_packed_parity.rs` (CPU, packed against
  masked) and `crates/nsl-codegen/tests/sdpa_fused_packed_gpu_parity.rs`
  (kernels against 64-bit references with documents straddling tiles; marked
  `#[ignore]`, needs the `cuda` and `test-hooks` features, and was not run
  for this review).

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

### FASE

1. Drop the peak-memory claim (E1). Present FASE-Deferred as exact
   accumulation with one fused step, and compare it with optimi (E2, E7).
2. For a real saving, add a fused in-hook step for N = 1 without clipping
   (about 2 GB at 500M 32-bit parameters) and an opt-in AdamA-style mode
   for N > 1.

### PCA

3. Redefine Strategy 2 as run-time tile-range skipping, drop the position
   reset for RoPE models (E13), and wire or delete Strategy 3.
4. Before any speed claim, benchmark against FlexAttention document masking
   and `varlen_attn`, and measure whether masking a chunked token stream
   helps validation loss at all.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code, on the CPU only. This part lists what it found
wrong in the implementation, as distinct from the proposal's own errors in
Part 1. The fixed items are described in `CHANGELOG.md` under "Unreleased".
The open list is as of 2026-11-13; each open item says whether it was
reproduced by running, shown by a numeric check, or found by reading the
code only.

### Fixed

#### FASE

- The full-buffer accumulation path computed a different optimizer in three
  ways. It is the path used by Lion, SOAP, Muon and any train block with
  `@fase(mode = off | full_buffer)` or `@fase(allow_v_approx = false)`. It
  accumulated the sum of the micro-batch gradients instead of their mean;
  Adam's bias correction counted micro-batches instead of optimizer steps;
  and `grad_clip` was applied to each micro-batch and the results summed.
  All three now follow the deferred path: window mean, optimizer-step
  count, and one global-norm clip of the window mean. On the review's
  fixture every path now trains to the same weights as standard AdamW
  accumulation. Test: `crates/nsl-cli/tests/accumulation_full_buffer_gate.rs`.
- Under `grad_clip`, the per-parameter fallback of the deferred step
  ignored `no_decay = [...]`. The fallback is reached with
  `--training-reference`, `NSL_FASE_MULTI_STEP=0`, optimizer-state offload
  or reduced-precision moments. It now goes through the grouped step, as
  the arm without clipping does. Test: the same file.
- The planner's rationale strings said the deferred path uses a
  "batch-variance v approximation". They now say it is exact.

#### PCA

- `DataLoader(..., packing = true)` without `pack_separator` used the
  default separator id 0, which is the token "!" in both shipped
  tokenizers. It is now a compile error, and so is passing `labels`
  together with `packing = true`. Test:
  `crates/nsl-cli/tests/packing_separator_gate.rs`.
- A packed row with more than 256 documents panicked inside a loader worker
  thread. With one worker the epoch ended early with exit status 0; with
  several the run hung. A worker panic now aborts the process with a
  message naming the separator.

### Open

#### FASE

- A trailing partial accumulation window is discarded without a message.
  With `grad_accumulation = 4`, a run of 10 micro-batches ends with the
  weights of a run of 8; the last two gradients are computed and dropped.
  `emit_train_teardown`, `crates/nsl-codegen/src/stmt_train/teardown.rs`.
  Reproduced by running. Right behaviour: flush the window, rescaled for
  the micro-batches it holds, or warn.
- Under accumulation the learning-rate schedule counts micro-batches, so
  `warmup_steps` and `total_steps` are in micro-batches, not optimizer
  steps. `emit_scheduler_lr`,
  `crates/nsl-codegen/src/stmt_train/scheduler_step.rs`; the module header
  states the unit. By reading the code only. Either offer optimizer-step
  units or refuse ambiguous use; the choice is open.
- The plan's recipe field `v_uses_approx` is still `true` for AdamW and
  Adam on the deferred path, whose second moment is exact (E3). `plan` and
  `make_recipe`, `crates/nsl-codegen/src/fase.rs`. By reading the code only.
- The tests `accumulate_stub_now_returns_ok` and
  `final_step_free_function_is_a_marker` assert that
  `emit_deferred_accumulate` and `emit_deferred_final_step` return
  `Ok(())`, which both do unconditionally.
  `crates/nsl-codegen/src/stmt_fase.rs`. By reading the code only. Right
  behaviour: delete the two functions and their tests.
- `fase_deferred_compiles_adamw_with_grad_clip`
  (`crates/nsl-codegen/tests/fase_deferred_smoke.rs`) returns early, and so
  passes, when its fixture file is missing. The file is present today. By
  reading the code only.
- `source_ad_peak_is_lower_than_tape_ad`
  (`crates/nsl-codegen/tests/fase_peak_memory_impl.rs`) compares NSL's
  source-AD path with NSL's own tape path, which materialises every
  gradient. It cannot show a saving against ordinary in-place
  accumulation, and there is none (E1). By reading the code only.
- The memory model `fase_breakdown`
  (`crates/nsl-codegen/src/fase_memory.rs`) charges the standard baseline
  with every layer's activations and FASE with one layer's, and returns
  the same numbers for Deferred and FullBuffer. Its test
  `deferred_cuts_peak_under_accumulation` passes because of the activation
  term. Nothing consumes the model (Part 3). By reading the code only.
- `emit_sgd` with `with_momentum = true`
  (`crates/nsl-codegen/src/fase_optimizer.rs`) adds `m_partial` to the
  momentum buffer with coefficient 1.0 and ignores the momentum value. It
  is unreachable: SGD with momentum under accumulation is refused at
  compile time. By reading the code only. Right behaviour: delete it.
- `crates/nsl-codegen/src/fase_clip.rs` (`ClipPlan`, `ClipProgram`) has no
  consumer; code generation writes the same clip formula out itself
  (Part 3). By reading the code only. Right behaviour: delete it.

#### PCA

- The `dataset`-block route hands the loader's `segment_ids` and
  `doc_starts`, stored as 32-bit floats, to kernels that read 16-bit and
  32-bit integers, and never moves `doc_starts` to the device. A sequence
  of at most one tile would run with reinterpreted ids; a longer one is
  refused and traps. `emit_packing_registry_stash`,
  `crates/nsl-codegen/src/stmt.rs`. By reading the code only.
- The segment path has no upper bound on sequence length. The forward
  kernel copies one 16-bit id per position into a 32768-byte shared-memory
  buffer, and `nsl_sdpa_fused_forward`
  (`crates/nsl-runtime/src/flash_attention.rs`) does not refuse a packed
  sequence longer than 16384, which would write past the buffer. Shipped
  lengths are 1024 to 4096. By reading the code only.
- `pack_separator` in a `dataset` block is accepted by `check_dataset_def`
  (`crates/nsl-semantic/src/checker/block.rs`) and read by nothing.
  `extract_dataset_packing` (`crates/nsl-codegen/src/pca_activation.rs`)
  reads only `separator_token_id`, which nothing consumes. By reading the
  code only. Right behaviour: remove the key or wire it through.
- A packed row with more than 256 documents still stops the run, now by
  aborting with a message. The limit, `MAX_NUM_DOCS` in `pack_batch`
  (`crates/nsl-runtime/src/packing.rs`), only sizes the position-reset
  table, which a RoPE model does not need (E13). Reproduced by running.
  Right behaviour: remove the limit together with the reset.
- `@pca(strategy = per_document)` cannot run. Every production call site
  passes a document count of 0, and `nsl_flash_attention_csha`
  (`crates/nsl-runtime/src/flash_attention.rs`) refuses a per-document
  kernel with that count and returns −1, which traps. `admit`
  (`crates/nsl-codegen/src/pca_per_doc.rs`) also trusts user-supplied
  length estimates. By reading the code only. Right: wire it or delete it.
- The module comment of `crates/nsl-codegen/src/pca_per_doc.rs` says the
  per-document path "is wired end to end" and "reachable from surface
  NSL"; it is not (previous item). Its only command-line test,
  `crates/nsl-cli/tests/run_pca_per_doc_gate.rs`, checks a stderr marker
  and not the exit status, because its fixture does not compile. By
  reading the code only.
- A comment in `stdlib/nsl/nn/gqa.nsl`, above `forward_packed`, calls the
  CPU fallback "numerically identical to the fused path". The fused path
  rounds its K and V tiles and its output to 16-bit floats, and the GPU
  parity test requires its gradients to differ from the exact ones by more
  than 1e-4 and less than 1e-2. By reading the code only.
- `crates/nsl-codegen/src/pca_tileskip.rs` says its skip fraction matches
  the proposal's 55% (see E9). Its test
  `packed_sequence_skips_cross_doc_tiles` asserts only `> 0.5` on a
  fraction that includes causal skips, 44/64 ≈ 0.69 for its layout.
  `build` there and `pca_segment::plan_kernel` have no production caller.
  By reading the code only. Right behaviour: delete both.
- For a kernel that is segment-masked but not causal, the forward omits
  the segment test (`emit`,
  `crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs`),
  so a bidirectional packed model would attend across documents with no
  error. Unreachable today: the packed op is always causal. By reading the
  code only. Right behaviour: refuse the combination.
