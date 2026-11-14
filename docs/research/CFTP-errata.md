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
reading it and from small CPU runs. The review itself ran nothing on a GPU;
the GPU tests named in Parts 3 and 5 were run later, when the defects in
Part 5 were fixed. Section numbers are the PDF's.

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
  PyTorch's `clip_grad_norm_`. The separate module §7.1 plans for it does
  not exist: code generation writes the formula out itself.
- Memory schedule and M36 planner integration (§2.4, §7.1): absent. The
  training report (`crates/nsl-codegen/src/training_report.rs`) prints no
  memory figures for FASE.
- A run that ends inside an accumulation window (not covered by the
  proposal): `emit_trailing_window_flush` in the same file steps once more,
  on the mean of the micro-batches the window holds, after the last
  epoch's `on_epoch` callback has run. Under `--layerwise-accum` it prints
  a warning instead and the pending gradients are not applied (Part 5).
- The learning-rate schedule under accumulation (not covered by the
  proposal): it advances once per optimizer step, so `warmup_steps` and
  `total_steps` are in optimizer steps. `schedule_unit = "micro_batch"` in
  `train(...)` makes it advance once per micro-batch, the unit before
  2026-11; `schedule_unit = "optimizer_step"` states the default. A block
  that combines a schedule with accumulation and states neither gets a
  compile-time warning (`schedule_unit_warning`,
  `crates/nsl-semantic/src/train_config.rs`). The 0.10-lts
  toolchain accepts the key too and keeps micro-batches as its default, so
  the production recipes pinned to it (`models/coder500m`,
  `models/coder1b`) state `schedule_unit = "optimizer_step"` and mean the
  same thing on both toolchains.
- In use: `models/coder500m/pretrain_prod.nsl`,
  `models/coder1b/pilot_finish.nsl`, `models/coder-rl/pretrain.nsl` and
  others, with `grad_accumulation` of 4 or 8, mostly with `grad_clip = 1.0`.

### PCA

Two separate mechanisms exist, and the one the proposal describes is not the
one that works.

- The proposal's surface (§4.1, §6.2), a `dataset` block with
  `packing = true`: partial. The block is planner metadata only; code
  generation skips it and it never becomes a loader. It accepts four
  fields, `packing`, `max_sequence_length`, `mean_doc_length` and
  `doc_length_stddev`, all literals, and refuses every other, including
  `pack_separator` (`check_dataset_def`,
  `crates/nsl-semantic/src/checker/block.rs`). `packing = true` there makes
  the `@flash_attention` and CSHA kernels (the attention-fusion kernels of
  the companion proposal) segment-masked, fed from a thread-local registry
  (`crates/nsl-runtime/src/pca_rope_runtime.rs`). Nothing can feed that
  registry correctly today: a `DataLoader(..., packing = true)` in the same
  module is a compile-time error, because the loader's tensors are not in
  the layout those kernels read. A kernel that is segment-masked and not
  causal is also a compile-time error.
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
  that is a multiple of the tile size and at most 16384, and contiguous Q,
  K and V (`nsl_sdpa_fused_forward` in
  `crates/nsl-runtime/src/flash_attention.rs`).
  Otherwise, and always under the default tape AD, attention falls back to a
  dense `[b,1,s,s]` float mask built in a host loop on every attention call
  (`nsl_packed_mask_from_segment_ids` in `crates/nsl-runtime/src/packing.rs`),
  which is the cost §3 set out to remove. Masking correctness was established
  by reading the kernels, and then by the GPU parity test named below.
- Strategy 2, compile-time tile-skip map: absent, and the implementation is
  the side that is right (E8). The kernels skip tiles at run time from a
  per-tile segment-range table held in shared memory
  (`crates/nsl-codegen/src/pca_tilerange.rs`). The skips are sound for
  non-decreasing segment ids, which are validated at launch.
- Strategy 3, one CTA (CUDA thread block) per document: partial. Kernels
  exist (`crates/nsl-codegen/src/flash_attention_v2/per_doc_cta.rs`,
  `crates/nsl-codegen/src/pca_per_doc.rs`) but are not reachable from a
  program: no attention call site passes them a document count, and
  `@pca(strategy = per_document)` is a compile-time error
  (`validate_pca_decorator`, `crates/nsl-semantic/src/cftp.rs`).
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
  (kernels against 64-bit references with documents straddling tiles, and
  the 16384-position bound; marked `#[ignore]`, needs the `cuda` and
  `test-hooks` features; it passes on an RTX PRO 4500).

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
- A trailing partial accumulation window was discarded without a message.
  With `grad_accumulation = 4`, a run of 10 micro-batches ended with the
  weights of a run of 8; the last two gradients were computed and dropped.
  The run now steps once more at the end, on the mean of the micro-batches
  the window holds, with the same clip and the next bias-correction count,
  as a PyTorch loop that steps on its last partial window does. The
  layer-major schedule is the exception and is listed under Open. Test:
  `crates/nsl-cli/tests/accumulation_full_buffer_gate.rs`, against an AdamW
  written out in the test and checked there against PyTorch's.
- Under accumulation the learning-rate schedule counted micro-batches, so
  `warmup_steps` and `total_steps` were reached N times sooner than the
  same number of optimizer steps, and nothing said so. The schedule now
  counts optimizer steps, and the unit can be stated:
  `schedule_unit = "optimizer_step"` in `train(...)` is the default,
  `schedule_unit = "micro_batch"` is the unit a block had before. A block
  that combines a schedule with accumulation and states neither is compiled
  with a warning, because a schedule written in micro-batches now runs N
  times slower; a checkpoint saved under the old unit is refused on resume
  by such a block. Test: `crates/nsl-cli/tests/schedule_unit_gate.rs`.
- The plan's recipe carried a field `v_uses_approx` that was `true` for
  AdamW and Adam on the deferred path, whose second moment is exact (E3).
  Nothing read it. It is removed.
- Two functions, `emit_deferred_accumulate` and `emit_deferred_final_step`,
  returned `Ok(())` unconditionally, and two tests asserted that they did.
  The functions and the tests are removed.
- `fase_deferred_compiles_adamw_with_grad_clip`
  (`crates/nsl-codegen/tests/fase_deferred_smoke.rs`) returned early, and so
  passed, when its fixture file was missing. It now fails in that case.
- The memory model `fase_breakdown` charged the standard baseline with
  every layer's activations and FASE with one layer's, returned the same
  numbers for Deferred and FullBuffer, and had no consumer; its test passed
  because of the activation term. The module
  (`crates/nsl-codegen/src/fase_memory.rs`) and the empty `memory` field of
  the training report are removed.
- `emit_sgd` had a momentum branch that added `m_partial` to the momentum
  buffer with coefficient 1.0 and never read the momentum value. It was
  unreachable (SGD with momentum under accumulation is refused at compile
  time) and is removed, with the `SgdMomentum` plan variant.
- `crates/nsl-codegen/src/fase_clip.rs` (`ClipPlan`, `ClipProgram`) had no
  consumer and is removed.
- `source_ad_peak_is_lower_than_tape_ad`
  (`crates/nsl-codegen/tests/fase_peak_memory_impl.rs`) was presented as
  FASE's peak-memory test. It compares NSL's source-AD path with NSL's own
  tape path, which materialises every gradient, so it cannot show a saving
  against ordinary in-place accumulation, and there is none (E1). The file
  now says what it compares and what it does not show. Its fixture said
  the test was blocked; run with the `test-hooks` feature it passes, at
  4.21 MB against 3.42 MB, and the fixture now says that.

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
- The `dataset`-block route handed the loader's `segment_ids` and
  `doc_starts`, stored as 32-bit floats, to kernels that read 16-bit and
  32-bit integers, and never moved `doc_starts` to the device. A sequence
  of at most one tile would have run with reinterpreted ids; a longer one
  was refused at launch and trapped. `DataLoader(..., packing = true)` in a
  module whose `@flash_attention` kernels are segment-masked is now a
  compile-time error that names `GroupedQueryAttention.forward_packed`,
  the route that converts the ids. A packed batch that reaches such a
  train block some other way stops the run with the same message; that
  second check was not exercised by a test. The route had a second defect
  in front of this one: such a module failed to compile, with an internal
  error, as soon as its train block read any `DataLoader`, because one
  call in the batch prologue passed an 8-bit value where a 64-bit one was
  declared. That call is fixed, so the case without a packing loader now
  compiles. Test:
  `crates/nsl-cli/tests/packed_flash_attention_refusal_gate.rs`.
- The segment path had no upper bound on sequence length. The forward
  kernel copies one 16-bit id per position into a 32768-byte shared-memory
  buffer, and `nsl_sdpa_fused_forward` launched it on a packed sequence
  longer than 16384, which writes past the buffer. It now declines such a
  sequence, which then takes the dense-mask path, and says so once. Tests:
  `segment_ids_fit_shared_memory` in
  `crates/nsl-runtime/src/pca_tier_b_runtime.rs`, and on a GPU
  `the_packed_forward_declines_a_sequence_longer_than_its_segment_buffer`
  in `crates/nsl-codegen/tests/sdpa_fused_packed_gpu_parity.rs`.
- `pack_separator` in a `dataset` block was accepted and read by nothing,
  and so was `separator_token_id`. Both are refused, with every other
  field of the block that nothing consumes; the message says the separator
  is given to the loader.
- A packed row with more than 256 documents stopped the run. The limit
  only sized the position-reset table, which the working route does not
  read. It is no longer a limit: such a row loads, with segment ids and
  positions for every document, and the table records the first 257
  document starts. What remains is the real bound, 65536 documents per row,
  because segment ids are 16-bit. The reset table itself is still built.
  Tests: `crates/nsl-runtime/src/packing.rs`,
  `crates/nsl-cli/tests/packing_separator_gate.rs`.
- `@pca(strategy = per_document)` could not run: every call site passed a
  document count of 0, the launcher refused it, and the program trapped. It
  is now a compile-time error, and so is `@pca(strategy = off)`, which
  nothing read. The kernels and the admission code stay, unreachable from a
  program. Test: `crates/nsl-cli/tests/run_pca_per_doc_gate.rs`, which used
  to check a stderr marker and not the exit status.
- The module comment of `crates/nsl-codegen/src/pca_per_doc.rs` said the
  per-document path "is wired end to end". It now says what is missing.
- A comment in `stdlib/nsl/nn/gqa.nsl` called the CPU fallback of
  `forward_packed` "numerically identical to the fused path". It now says
  that the two mask the same pairs and differ by 16-bit rounding.
- `crates/nsl-codegen/src/pca_tileskip.rs` claimed to match the proposal's
  55% (E9) and had no production caller, and neither had
  `pca_segment::plan_kernel` nor `pca_rope::plan`. All three are removed
  with their tests.
- A kernel that was segment-masked but not causal omitted the segment test,
  so it would have attended across documents. The combination is now a
  compile-time error.

### Open

#### FASE

- Under `--layerwise-accum` (and so under `--zero-stage 3`, which requires
  it) a trailing partial accumulation window is still not applied. That
  schedule keeps saved activations, not gradients, between micro-batches,
  and its window backward is built for exactly N of them. The run now
  prints a warning with the number of dropped micro-batches. A run that
  ends inside a window is therefore the one case in which
  `--layerwise-accum` is not bit-identical to the interleaved schedule;
  compiling with `NSL_FASE_FLUSH_TAIL=0` makes the interleaved schedule
  drop the tail as well, which is how the parity tests compare the two.
  `emit_trailing_window_flush`,
  `crates/nsl-codegen/src/stmt_train/optimizer_step.rs`, and
  `emit_csla_window_backward`,
  `crates/nsl-codegen/src/stmt_train/csla_window.rs`. Reproduced by
  running. Right behaviour: replay a window of fewer than N micro-batches.

#### PCA

- The position-reset table (`doc_starts`), the loader's `position_ids` and
  the in-kernel reset on the CSHA route are still built, although a RoPE
  model does not need them (E13). They are no longer a limit on anything.
  `pack_rows`, `crates/nsl-runtime/src/packing.rs`;
  `crates/nsl-codegen/src/pca_rope.rs`. By reading the code only.
- The per-document kernels and their admission code
  (`crates/nsl-codegen/src/flash_attention_v2/per_doc_cta.rs`,
  `crates/nsl-codegen/src/pca_per_doc.rs`) remain, unreachable from a
  program. Wiring them needs a per-row document count at the attention call
  sites and a guard for a document longer than one tile; deleting them is
  the alternative. The choice is open.
