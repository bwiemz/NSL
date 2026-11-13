# NSL-WRGA Research — Errata

**Companion to** `NSL-WRGA-Research.PDF`. **Date:** 2026-11-13.

This file lists the statements in the April 2026 proposal for WRGA
(Wengert-pruned roofline-guided adaptation) that should not be relied on, and
says what is true instead. It comes from a November 2026 review that
recomputed the proposal's arithmetic, compared its statements about other
work with the sources, and compared its design with the code in this
repository by reading the source and running small test programs. Where an
outside fact was not read from the primary source, the entry says so. The
proposal's two appendices in this folder, `NSL-WRGA-Research-Appendix-B.md`
and `NSL-WRGA-Research-Appendix-B2-addendum.md`, are cited where they record
measurements.

Abbreviations used below: LoRA is low-rank adaptation, which adds a trainable
product of two small matrices A and B beside a frozen weight W. IA³ is the
adapter that multiplies an activation by a learned vector. PEFT is
parameter-efficient fine-tuning. AD is automatic differentiation. SVD is the
singular value decomposition. GFLOP and TFLOP are 10⁹ and 10¹² floating-point
operations. FFN and MLP both mean a block's feed-forward layers. cuBLAS is
NVIDIA's matrix-multiply library, PTX is NVIDIA's GPU assembly language, and
TF32 is the reduced-precision tensor-core mode for 32-bit floats.

---

## Part 1 — Errors in the proposal

### E1 — The §2.5 comparison table is arithmetically impossible

**Proposal.** §2.5, "Comparison (NSLCoder-50M, batch=32, seq=1024)": full
fine-tuning "4.8 GB" and "234 GFLOP"; LoRA r=8 in PyTorch "3.2 GB" and "~220
GFLOP"; WRGA "1.9 GB" and "48 GFLOP". The sample report in §4 repeats them:
"Estimated training FLOP/step: 48.2 GFLOP (vs ~220 GFLOP for PyTorch LoRA)".

**Correction.** For NSLCoder-50M (8 blocks, width 512, feed-forward width
1408, vocabulary 49,152, tied output head) one forward pass over one
1024-token sequence costs about 117 GFLOP: 48.3 for the weight matmuls in
the blocks, 51.5 for the output head, and 17.2 for the two attention products
counted at width 512. A batch of 32 costs 32 × 117.04 = 3,745 GFLOP.

- 234 GFLOP is twice the forward pass of one sequence (234 / 117 = 2.0). It
  is not a training step at batch 32.
- 48 GFLOP is 41% of the forward pass of one sequence (48 / 117). A training
  step still runs the forward pass through every layer, so no pruning of the
  backward pass can bring a step below the forward cost.
- The LoRA row puts PyTorch LoRA at 220 / 234 = 94% of full fine-tuning. The
  published figure is about two thirds: "LoRA takes slightly more than ⅔ of
  the FLOPs that full fine-tuning does per pass".
- The logits tensor alone, [32, 1024, 49152] in 32-bit floats, is
  32 × 1024 × 49152 × 4 bytes = 6.44 GB. That is more than every entry in
  the "Peak Memory" column.

**Source.** Recomputed from the model's dimensions. The LoRA cost is quoted
from <https://thinkingmachines.ai/blog/lora/> (checked).

### E2 — Backward speedup is L / |L_a| only when the top layers are adapted

**Proposal.** §5.1: "Speedup factor (backward only): L / |L_a|. For a
32-layer model with 4 adapted layers: 8× backward speedup."

**Correction.** Reverse-mode differentiation must carry the activation
gradient down through every layer above the lowest adapted layer, whether or
not those layers are adapted. With adapters on layers 1 and 32 of 32, the
activation-gradient backward runs through all 32 layers, where the formula
predicts 16×. The formula holds only when the adapted layers are the topmost
ones, which is the case in the §2.1 example (layers 28–32, "~5/32 = 15.6%").
The saving depends on the position of the lowest adapted layer, not on how
many layers are adapted.

### E3 — PyTorch already skips a frozen prefix; "impossible in PyTorch" is false

**Proposal.** §1, Problem 1: the autograd engine must "traverse the full
computation graph during backward to discover which nodes need gradients".
Problem 4: "PyTorch saves activations for the backward pass of *all* layers
(frozen and trainable)". §2.1: "It cannot eliminate dead gradient paths
because it doesn't know they're dead until it traverses them." §2.4: "Even
with `torch.compile`, the fusion scope is limited to operations within a
single `nn.Module.forward()` call."

**Correction.** PyTorch's autograd notes say "an operation is only recorded
in the backward graph if at least one of its input tensors require grad".
Operations below the first trainable tensor are therefore never recorded,
never traversed, and their activations are never saved. Above the first
trainable tensor, PyTorch skips only the weight-gradient products of the
frozen weights, which is exactly what dead-gradient elimination does in NSL.
Problems 1 and 4 describe costs PyTorch does not pay.

The claims that the technique cannot be done elsewhere do not hold either:

- `torch.compile`, through AOTAutograd, traces a joint forward-and-backward
  graph ahead of time and runs dead-code elimination on it.
- `jax.grad` differentiates only with respect to its argument, and the XLA
  compiler removes dead code (not re-checked against the source).
- TorchDynamo inlines calls to submodules, so fusion is not confined to one
  `forward()` (not re-checked against the source).

**Source.** <https://docs.pytorch.org/docs/2.9/notes/autograd.html> (checked).
The `torch.compile` statement is from a search-result summary of
<https://dev-discuss.pytorch.org/t/how-does-torch-compile-work-with-autograd/1621>,
not from reading the page.

### E4 — The roofline labels do not follow from the stated hardware, and invert

**Proposal.** §2.2 table: Q/K/V projections "73-102", FFN gate/up matmuls
"137.4", LM head "169.5", all labelled "Compute". §4 sample report: "Target
hardware: NVIDIA H100-SXM (989 GFLOPS FP32, 3.35 TB/s HBM)".

**Correction.**

- 989 is the H100's dense FP16 tensor-core peak, and the unit is TFLOPS, not
  GFLOPS. Its FP32 peak is 67 TFLOPS.
- The arithmetic-intensity column reproduces exactly with a matmul of
  M = 1024 rows and 4-byte elements, AI = 2·M·K·N / (4·(M·K + K·N + M·N)):
  73.1 for a 512×256 weight, 102.4 for 512×512, 137.4 for 512×1408 and 169.5
  for 512×49152. The column therefore describes one 1024-token sequence, not
  the batch of 32 used in §2.5.
- The "Compute" labels follow only from the FP32 ridge without tensor cores,
  67×10¹² / 3.35×10¹² = 20 FLOP/byte. Against the ridge the compiler's own
  GPU table uses for half precision, 989×10¹² / 3.35×10¹² = 295 FLOP/byte,
  every matmul in the table is memory-bound (the largest is 169.5), and the
  placement advice inverts.

**Source.** Recomputed from the proposal's own figures. The H100 peaks are
from a search-result summary in which several specification listings agree,
not from NVIDIA's datasheet.

### E5 — Adapter compute is not "free": the fused kernel measured slower

**Proposal.** §2.2: "The adapter's compute is effectively *free* — it
executes during time that would otherwise be wasted." §5.3: "WRGA maximizes
total adapter capacity subject to zero wall-clock overhead".

**Correction.** Appendix B records the measurement. Section B.6 reports that
the "fused forward loses 15–67×" at every m tested, m ∈ {1, 16, 64, 256,
1024}, and section B.4 records the resulting decision: "unfused cuBLAS is
the default at every `m`". The same appendix records that the fused forward
first reported as shipped worked only for n ≤ 64, and that the 106×
backward-to-forward reading that prompted the fused-backward milestone was a
measurement artefact; the clean value was 0.37×.

**Source.** `NSL-WRGA-Research-Appendix-B.md`, sections B.3, B.4 and B.6.

### E6 — Training cannot peak below inference memory; parameter counts disagree

**Proposal.** §2.5: "The peak memory of WRGA fine-tuning is often *less
than* the peak memory of inference". §2.5 table: WRGA trainable parameters
"0.4M (0.8%)". §4 sample report: "Adapter parameters: 41,984 (0.086% of
model)". §2.5 table: "LoRA r=8 (PyTorch)" with "0.4M (0.8%)".

**Correction.**

- A training step contains the inference forward pass, so its peak memory
  cannot be lower than that of inference.
- The two WRGA figures differ by a factor of about ten: 0.4M is 0.82% of
  48.8M parameters and 41,984 is 0.086%.
- LoRA at r = 8 on all seven weight matrices of all eight blocks (per
  block: two of 512×512, two of 512×256 and three between widths 512 and
  1408) is 8 × 8 × (2·1024 + 2·768 + 3·1920) = 598,016 parameters, 1.2% of
  the model, not 0.4M.

**Source.** Recomputed from the proposal's own figures.

### E7 — Softmax and normalisation layers have no weight to adapt

**Proposal.** §2.2 table: Softmax, "High-rank adapter (r=16+)";
LayerNorm/RMSNorm, "IA³-style scaling (free)".

**Correction.** A softmax has no weight matrix, so it cannot carry a
rank-r adapter. §2.4 says as much in its own sample output:
"blocks.7.softmax: UNFUSIBLE with rank-adapter, using IA³ scaling instead".
A learned scaling vector on the output of a normalisation layer is the same
thing as unfreezing that layer's existing gain; it introduces no new
parameter and is not a separate adapter.

### E8 — The rank-allocation rule is unsupported and contradicts itself

**Proposal.** §2.3, step 3: "Matrices with slow decay (high effective rank)
need high-rank adapters." Step 4 defines adaptation headroom as "the gap
between the weight's effective rank and its full rank". The combined score
uses a `gradient_magnitude_estimate` that "comes from Innovation 1's Wengert
analysis".

**Correction.**

- Steps 3 and 4 point in opposite directions. Step 3 gives more rank to a
  weight with high effective rank; the headroom of step 4 shrinks as
  effective rank grows.
- No evidence is given that a weight's own spectrum shows where adaptation
  is needed. The LoRA paper's analysis (its §7.3) found that the learned
  update mostly amplifies directions the pretrained weight does not
  emphasise, so the top of the weight's spectrum is not evidence about the
  update.
- The Wengert analysis is a structural reachability analysis. It says which
  gradients are needed and yields no magnitudes, so it cannot supply the
  third factor of the score.

**Source.** The LoRA finding is from a summary of the fetched page
<https://arxiv.org/html/2106.09685v2>, not a verbatim reading of §7.3.

### E9 — AdaLoRA, GaLore and plain LoRA are misdescribed

**Proposal.** §6.1: AdaLoRA "adds significant per-step compute (one SVD per
adapted layer per step)". §6 table, "Runtime overhead": AdaLoRA "High
(SVD)", GaLore "Moderate (SVD)". §1, Problem 2: methods "apply the same rank
`r` across all layers because there is no principled, zero-cost way to
determine per-layer optimal ranks", using information "that PyTorch cannot
access at the framework level".

**Correction.**

- AdaLoRA does not compute an SVD per step. Its abstract says that its
  parameterisation of the update lets it "circumvent intensive exact SVD
  computations".
- GaLore recomputes its projector only every T steps, with T typically 50
  to 1000, and reports overhead under 10%. "Moderate (SVD)" overstates it.
- An SVD of each pretrained weight before training is a few lines of
  PyTorch. The information is accessible there; nothing about it requires a
  compiler.

**Source.** AdaLoRA: <https://arxiv.org/abs/2303.10512> (checked). The GaLore
figures are from a secondary source quoting Zhao et al.; the GaLore paper
itself was not opened.

### E10 — Citation and cross-reference slips

- Reference 2 gives AdaLoRA's venue as "ICML 2023". It is ICLR 2023
  (<https://arxiv.org/abs/2303.10512>, checked).
- Reference 4 attributes DoRA to "Liu, H., et al." DoRA is by Shih-Yang Liu
  et al., ICML 2024 (<https://arxiv.org/abs/2402.09353>, checked). H. Liu is
  the first author of IA³ (not re-checked against the source).
- IA³, LoftQ and ReFT are discussed in the text but do not appear in the
  reference list.
- Appendix B refers to "§6.X", "§6.Y" and "§6.Z of the main paper". The
  proposal has no such sections; its §6 has only §6.1. Appendix B is a set
  of testing-discipline rules rather than a research appendix, and the
  measurements it records (see E5) are its factual content.

---

## Part 2 — Superseded by later work

- **"LoRA Without Regret"** (Schulman, Thinking Machines, 29 September 2025;
  <https://thinkingmachines.ai/blog/lora/>, checked). It reports no additional
  benefit from adapting attention matrices beyond adapting the MLP matrices
  alone, and recommends adapting all layers, especially the MLPs. The
  proposal's placement policy (minimal rank on the feed-forward matrices,
  skip the head, prefer scaling at softmax and normalisation) is the
  opposite. It also finds the best LoRA learning rate is about 10× that of
  full fine-tuning, and that 1/r scaling makes it roughly independent of
  rank; NSL has one learning rate for all parameters.
- **EVA** (Paischer et al., NeurIPS 2025,
  <https://arxiv.org/abs/2410.07170>; from a search-result summary). This
  is the published form of deciding ranks before training: an incremental
  SVD of activations on a few minibatches, then ranks redistributed by
  explained variance. It is driven by data; the proposal's weight-spectrum
  rule has no comparable evidence.
- **"LoRA vs Full Fine-tuning: An Illusion of Equivalence"**
  (<https://arxiv.org/abs/2410.21228>; from a search-result summary). It
  reports that low-rank updates introduce "intruder dimensions", and that
  methods which modify singular vectors, such as PiSSA, have many. This is
  a caution against designing adapters from a weight SVD.
- **Plain LoRA holds up against its variants.** "A Unified Study of LoRA
  Variants" (<https://arxiv.org/abs/2601.22708>, January 2026; from a
  search-result summary) and "Which LoRA?"
  (<https://arxiv.org/abs/2606.10428>, June 2026; abstract read), whose
  abstract reports no significant advantage of four variants over basic
  LoRA. A correct and complete plain LoRA (save, merge, any weight shape)
  is worth more than placement machinery.
- **rsLoRA** (Kalajdzievski, <https://arxiv.org/abs/2312.03732>, checked).
  Dividing by the rank slows learning at higher rank; scaling by α/√r is
  proposed instead. It predates the proposal and appears neither there nor
  in the code.
- **GaLore 2** (<https://arxiv.org/abs/2504.20437>, April 2025; from a
  search-result summary) scales GaLore to pre-training, so the GaLore
  column of the §6 table is stale.
- **Hardware.** The RTX PRO 4500 Blackwell used for development is not in
  the compiler's GPU table. The runtime's default matmul mode is TF32 on
  tensor cores, which the roofline model does not represent: 4-byte tensors
  are classified against the ridge derived from the 67 TFLOPS FP32 peak.

---

## Part 3 — What the code implements

WRGA is switched on by decorators only: `@adapter(type=lora|ia3|gatedlora,
target=["Class.field"], rank=, alpha=)`, `@freeze(include=|exclude=)` and
`@wrga(mode=, budget=, target=, layers=, adapter=)`. `nsl run` has no WRGA
flag. `nsl build` has `--wrga-report` and `--wrga-fold-allocations`;
`nsl check` has `--wrga-analyze`, `--wrga-compare`, `--wrga-ablate` and
`--wrga-target`. No shipped model uses any of the three decorators; only
test fixtures do, all with 2-D inputs and square 4×4 or 8×8 weights.

- **Dead-gradient elimination (Innovation 1): implemented, but not by
  WRGA.** It is the generic source-AD pass `eliminate_dead_gradients` in
  `crates/nsl-codegen/src/source_ad.rs`, which runs under `--source-ad`
  with or without `@wrga`. WRGA's own second filter,
  `eliminate_by_backward_live`, driven by `select_trainable` in
  `crates/nsl-codegen/src/wrga.rs`, duplicated the generic pass and has
  been taken out of the train block (Part 5); the set `select_trainable`
  builds now feeds only the report. The generic pass is the right one.
- **`@freeze`: implemented and correct.** Frozen parameters are left out of
  the train block's parameter list, so they receive no update and no
  optimizer state. Checked with SGD and with AdamW weight decay, on CPU and
  GPU.
- **`@adapter` (LoRA, IA³, gated LoRA): partial.** The rewrite lives in
  `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`, with
  `wrga_adapter_inject.rs` and `wrga_adapter_init.rs` beside it. LoRA is
  `x @ W + ((x @ A) @ B) * (alpha / rank)` with A of shape [in, r], B of
  shape [r, out] initialised to zero, and alpha defaulting to the rank. IA³
  is `(x @ W) * gamma` with gamma initialised to one. On the default
  (unfused) path the gradients of LoRA, frozen-base LoRA, IA³ and gated
  LoRA agree with PyTorch autograd in 64-bit floats to within 7e-8 after
  two SGD steps, for 2-D and 3-D inputs, under both tape and source AD. The
  rewrite matches only the form `x @ self.w`, the rank is the declared
  `rank=` or 2, and there is no save, merge or unmerge for adapters.
- **Roofline placement (Innovation 2): report-only.**
  `crates/nsl-codegen/src/wrga_roofline.rs` is fed by
  `infer_sites_from_wengert` in `wrga.rs`, which supplies placeholder sites
  (512×512, 2-byte elements, batch 1, sequence 128) rather than the model's
  shapes. The result appears only in the `--wrga-report` output; the
  adapters that exist come solely from `@adapter`. Neither side is right
  here; see E4, E5 and E7.
- **Spectral rank allocation (Innovation 3, "requires --weights"):
  report-only.** A randomized SVD exists (`run_spectral` in `wrga.rs`,
  `compute_spectral_rank` in `crates/nsl-codegen/src/wrga_spectral.rs`), but
  its ranks never size an adapter; `crates/nsl-codegen/src/wrga_prescan.rs`
  uses the declared rank or 2. Only the top 16 singular values are kept, so
  the effective rank is at most 16 and the values in the proposal's sample
  report (48.2, 52.1, 89) cannot occur. Neither side is right; see E8.
- **Epilogue fusion (Innovation 4): partial, opt-in.** PTX kernels for LoRA
  and gated LoRA are emitted by `crates/nsl-codegen/src/wrga_fused_ptx.rs`
  and launched from `crates/nsl-runtime/src/fused_adapter.rs`. They need a
  pinned target (`--target sm_80` or later) and `NSL_WRGA_FUSED_CUDA=1`.
  The kernel computes in half precision, handles 2-D inputs only, and
  measured 15–67× slower than cuBLAS (E5). The IA³ "fused" path always
  runs the CPU-style arithmetic. The implementation's default, unfused, is
  the right side.
- **Memory-planned activation sharing (Innovation 5): report-only.**
  `crates/nsl-codegen/src/wrga_memory.rs` produces a slot count that is
  printed in the report. `apply_wrga_hints` in
  `crates/nsl-codegen/src/memory_planner.rs` does not change the allocator,
  and the sizes it is given are always zero. Neither side is right.
- **`@wrga(mode=auto)`, "compiler synthesizes optimal adapter config":
  absent.** No adapter is created without `@adapter`; the full model
  trains.
- **`mode=manual` and `mode=hybrid`: parsed, no effect on training.**
  Trainability is decided by `@freeze` alone, under tape AD and source AD
  alike, so neither mode changes what trains as the proposal describes.
  The modes should be given a meaning or refused.
- **`budget=0.5M`, `target=h100`: not honoured.** The budget must be an
  integer literal and does not bound the adapter parameter count; the
  `target=` written in the decorator does not reach the analysis.
- **Per-instance targets such as `"blocks.6.wq"`: absent.** Only
  `Class.field` is accepted, and it applies to every instance.
- **Custom adapter model (`adapter=GatedLoRA`, §8.2): absent.** It is
  validated and then reported as "(placement integration pending)". The
  built-in `type=gatedlora` uses `sigmoid(gate)` with the gate and B
  initialised to zero, unlike the proposal's `gate = ones` with no sigmoid.
  The implementation's initialisation is sound; the adapter starts as a
  no-op.
- **`nsl_wrga_epilogue_fused_lora` (§7.3): present, unused.** It is defined
  in `crates/nsl-runtime/src/peft/ffi.rs` and nothing calls it.
- **DoRA, rsLoRA scaling, adapter dropout, and an AdaLoRA-style rank
  schedule: absent.**

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Finish plain LoRA first: any weight shape, the transposed-weight form
   `x @ self.w.transpose(0, 1)`, adapters saved with the model, and merge.
2. Withdraw the §2.5 table, the L / |L_a| formula and the "impossible in
   PyTorch" framing (E1–E3).
3. Replace roofline placement and weight-spectrum ranks with something
   measured: adapt the MLP matrices by default; if per-layer ranks are
   wanted, calibrate on a few batches and compare with uniform rank at
   equal parameters on NSLCoder-50M.
4. Keep the unfused path as the default, and add a separate adapter
   learning rate and an rsLoRA scaling option.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code, on CPU and in a few short GPU runs. This part
lists what it found wrong in the implementation, as distinct from the
proposal's own errors in Part 1. Fixed items are described in `CHANGELOG.md`
under "Unreleased". The open list is as of 2026-11-13.

### Fixed

- `@wrga(mode = manual)` and `@wrga(mode = hybrid, layers = [...])` trained
  nothing under `--source-ad` and exited 0. The trainable set of the WRGA
  prune pass came out empty, because its patterns never match instance
  paths, and the source-AD driver then dropped every adjoint operation;
  tape AD ignored the mode and trained everything. The filter is removed
  (`optimize_adjoint_tape` in
  `crates/nsl-codegen/src/stmt_train/adjoint_tape_opt.rs`), and what trains
  is decided by `@freeze` alone in both AD modes. The test is
  `crates/nsl-cli/tests/wrga_mode_and_rank_gate.rs`.
- A pinned target (`--target sm_80` and later) alone selected the fused
  adapter call, whose adjoints assume 2-D activations. An `@adapter` on a
  `[batch, seq, d]` activation produced gradients of the wrong shape and
  aborted at the first optimizer step. The fused path now also requires
  `NSL_WRGA_FUSED_CUDA=1` at compile time (`fused_path_selected` in
  `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`); the same test file
  covers it. The adjoints themselves are unchanged; see the open list.
- The launcher of the synthesized fused kernel took `m` from the
  second-to-last dimension and, for a 3-D input, returned a 2-D result
  holding batch 0 only; it also read strided views in storage order. It
  now declines anything but contiguous 2-D operands, and the caller falls
  back to the ordinary operations (`kernel_can_read` in
  `crates/nsl-runtime/src/fused_adapter.rs`).

### Open

- With the fused path opted in (`NSL_WRGA_FUSED_CUDA=1` and a pinned
  target), the GPU kernel computes in half precision, including the base
  product `x @ W`, with no notice. On 2-D non-constant inputs the result
  was off by up to 1.0e-3 (relative 3.3e-4) against a 64-bit reference.
  Where: `try_cuda_launch_fused_lora` in
  `crates/nsl-runtime/src/fused_adapter.rs`. Reproduced by running.
- The class in an `@adapter` target is ignored. With models `Aaa` and `Zzz`
  both owning a field `w`, `target=["Zzz.w"]` attaches the adapter to `Aaa`
  with only an INFO line; the adapter on `Aaa` trains and `Zzz` gets none.
  Where: `run_with_compiler` in
  `crates/nsl-codegen/src/wrga_adapter_inject.rs`. Reproduced by running.
  Right: honour the class, and make an ambiguous field an error.
- Trained adapters are not saved. A frozen-base LoRA run took the loss from
  1.3207 to 0.3248; `model_save` followed by `model_load` into a fresh
  model gave 1.3207 again, with no message. There is no merge, so a LoRA
  fine-tune has no persistent output. Where: `compile_model_save` in
  `crates/nsl-codegen/src/expr/advanced.rs`. Reproduced by running.
  Right: save and load the adapter tensors, and add merge and unmerge.
- `print(m.lora_A_Toy_w__lora.sum())` prints a raw pointer value, a large
  integer, instead of the sum: a synthesized adapter field has no type
  unless it is first bound with `let a: Tensor = ...`. The cause was not
  located; these fields are resolved in `compile_member_access` in
  `crates/nsl-codegen/src/expr/access.rs`. Reproduced by running.
- The randomized SVD behind the spectral figures is biased on low-rank
  input. On an exactly rank-5 40×30 matrix the largest singular value came
  out 1.333 instead of 1.0, because numerically zero residual columns are
  normalised (max |QᵀQ − I| = 0.70); the unit test accepts this as a
  "known bias". Where: `qr_orthonormalize` in
  `crates/nsl-codegen/src/wrga_spectral.rs`. Numeric check.
- A checkpoint stored as 8-bit floats is decoded as `byte / 127.0`, not as
  floating point: the byte 0x80, which is −0.0, reads as 1.008. Spectral
  analysis of such a checkpoint is meaningless, and nothing says so.
  Where: `WeightDType::to_f64` in `crates/nsl-codegen/src/weight_aware.rs`.
  By reading the code only.
- `@adapter` on a weight used as `x @ self.w.transpose(0, 1)`, the form of
  the standard library's `Linear`, is accepted and does nothing: A and B
  are allocated and handed to the optimizer but never enter the forward, A
  is unchanged, B stays zero, and the run exits 0 with no message. Where:
  `match_adapter_site` in `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`.
  Reproduced by running. Right: match this form, or fail the build.
- `@adapter(target=["Linear.w"])` on a model imported from the standard
  library gives only a warning, "field 'w' not found in any known model".
  With `@freeze` beside it nothing trains and the loss does not change.
  Where: `run_with_compiler` in
  `crates/nsl-codegen/src/wrga_adapter_inject.rs`. Reproduced by running.
  Right: a target that resolves to no field is a compile error.
- `@wrga(mode=manual)` and `@wrga(mode=hybrid, layers=[...])` are accepted
  and change nothing about what trains (see Fixed). `select_trainable` in
  `crates/nsl-codegen/src/wrga.rs` still matches `Class.field` targets and
  bare layer prefixes against instance paths such as `m.b1.wq`, so the set
  it gives the report is empty in these modes. By reading the code only.
  Right: give the modes a meaning, or refuse them.
- `@wrga(budget=N)` does not bound the adapter parameter count. On six
  32×32 sites with weights supplied, `budget=1000` and `budget=100000` both
  gave 6,144 parameters, and `budget=0` (unbounded) gave 4,352. The
  parameter count is passed on as a total of ranks. Where: `run_spectral`
  in `crates/nsl-codegen/src/wrga.rs` and `allocate_ranks` in
  `crates/nsl-codegen/src/wrga_spectral.rs`. Reproduced by running.
- `@wrga(target=h100)` is reported as "Target hardware: A100-SXM": the
  decorator's target never reaches the analysis, and the build's fallback
  name "rtx5070ti" matches no entry in the GPU table, so the default GPU is
  used. Where: `analysis_to_wrga_inputs` in
  `crates/nsl-cli/src/analysis_bridges.rs` and `invoke_wrga_if_enabled` in
  `crates/nsl-codegen/src/stmt_pass_bridges.rs`. Reproduced by running.
- A target written as a string literal, `@wrga(target="h100")`, stores the
  argument's name instead of its value. Where: `validate_wrga_decorator`
  in `crates/nsl-semantic/src/wrga.rs`. By reading the code only.
- With a frozen base (`@freeze` plus `@adapter`) and `--weights`, the
  spectral allocator cannot run: sites are inferred from the trainable
  parameters, which are then the adapter matrices, so no checkpoint key
  matches (a warning says so) and A keeps the declared rank or 2. Where:
  `infer_sites_from_wengert` in `crates/nsl-codegen/src/wrga.rs`.
  Reproduced by running. Right: key the sites on the frozen weights.
- The documentation of `AdapterConfig.rank` says that without a declared
  rank "WRGA picks the rank via roofline + spectral analysis". The rank is
  then 2. Where: `AdapterConfig` in `crates/nsl-semantic/src/wrga.rs` and
  `prescan_adapter_sites_from_decorators` in
  `crates/nsl-codegen/src/wrga_prescan.rs`. Reproduced by running.
- `nsl check --wrga-analyze --weights w.safetensors` ignores the weights
  with no message; the report is the same with and without them. Where:
  `run_check_wrga_analyze` in
  `crates/nsl-cli/src/commands/build/wrga_check.rs`. Reproduced by running.
- With the fused path opted in, the synthesized IA³ kernel can never
  launch: the call is always emitted with kernel handle −1. Where:
  `synthesize_ia3_fused_call` in
  `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`. By reading the code
  only.
- With the fused path opted in, an `@adapter` on a 3-D activation aborts at
  the first optimizer step under `--source-ad`: the fused adjoints do not
  sum over leading dimensions, so for `x` of shape [2, 3, 4] the weight
  gradient has 32 elements, not 16. Right: flatten the leading dimensions.
  Where: `generate` in `crates/nsl-codegen/src/source_ad.rs`. Reproduced
  by running, when a pinned target alone selected this path.
- An adapter on a non-square weight aborts in every mode: for `w` of shape
  [4, 6] used as `x @ self.w`, A is created as [6, 2] and the run stops on
  "matmul inner dimension mismatch (3x4 @ 6x2)". The shape is read as
  [out, in]; the matched form needs [in, out]. Right: take the widths
  from the use site. Where: `resolve_dims_for_target` in
  `crates/nsl-codegen/src/wrga_adapter_inject.rs`. Reproduced by running.
- `--wrga-report` prints "Activation memory: 15 tensors → 18 tensors (0.0%
  eliminated)": the left figure counts operations and the right counts
  tensors, and `--wrga-compare` then shows WRGA at "120.0%". "Backward
  pruning: 27 ops → 19 ops (29.6% eliminated)", with nothing frozen, counts
  inputs and constants as eliminated. Where: `prune` in
  `crates/nsl-codegen/src/wrga_prune.rs`. Reproduced by running.
- The report prints "Fused: 6/6 (100.0%)" for a program with no adapters,
  in a build that runs unfused. Where: `render_report` in
  `crates/nsl-codegen/src/wrga.rs`. Reproduced by running.
- The adapter parameter count in the report is a placeholder. Without
  weights, or when no checkpoint key matches, it is the suggested rank
  times 1024 per site and the budget is ignored: 98,304 for a model whose
  figure would be 6,144, and "Adapter parameters: 65536 | Sites: 4" for a
  program with 256 adapter parameters on 2 weights. Where: `run_spectral`
  in `crates/nsl-codegen/src/wrga.rs`. Reproduced by running.
- The baselines of `--wrga-compare` are not computed for the program. The
  "AdaLoRA" row is WRGA's own parameter count, and every baseline's
  backward and activation columns are the constant 100% and its fusion
  column 0%. Where: `compute_compare_baselines` in
  `crates/nsl-codegen/src/wrga.rs`. Reproduced by running.
- "Memory plan: 18 live activations → 5 slots (reuse 72.2%)" is suspected
  to be unachievable. Each saved activation's lifetime ends at its last
  forward use, but these are the tensors the backward reads later, so all
  are live at the forward/backward boundary and shared slots would alias
  them. The plan is not applied today. Where: `compute_var_ranges` in
  `crates/nsl-codegen/src/wrga_memory.rs`. By reading the code only.
- The warning "alpha is only meaningful for LoRA; ignored for
  ia3/gatedlora" is false for gated LoRA: with `alpha=4, rank=2` the
  adapter trains exactly as with scale 2.0. Where:
  `validate_adapter_decorator` in `crates/nsl-semantic/src/wrga.rs`.
  Reproduced by running. Right: drop the warning for gated LoRA.
- The module documentation of `crates/nsl-codegen/src/wrga_roofline.rs`
  says a model compiled for an RTX 5070 Ti and for an H100 receives
  different placements. Every matmul site is the placeholder of
  `infer_sites_from_wengert`, whose arithmetic intensity of 85.3 is
  memory-bound on every GPU in the table (lowest half-precision ridge
  151.7), so every site gets LoRA at the maximum rank. Numeric check.
- The GPU tests of the fused kernels cannot detect a wrong kernel: the
  LoRA fixture has `w = zeros` and `x`, A and B all ones, and its expected
  16.0 ± 1e-4 is also what a kernel returns that drops the base product,
  transposes A or B or mis-indexes an operand. Where: `BUILD4_SRC_GPU` in
  `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`. By reading
  the code only. The IA³ and gated fixtures are constant fills as well.
- No test trains a model with an adapter on a non-square weight, or checks
  a `budget` that should bind, so neither defect above is guarded. Where:
  `crates/nsl-cli/tests/`. By reading the code only.
- The check that a fused site left no `x @ A` intermediate cannot fire. It
  searches for `lora_A_<site>` with the site written with a dot ("Toy.w"),
  while the field names use underscores ("lora_A_Toy_w__lora"). Where:
  `verify_fused_sites_have_no_intermediate` in
  `crates/nsl-codegen/src/wrga_fusion.rs`. By reading the code only.
- The fusion plan pairs placement i with the i-th rank, but the ranks come
  back in the iteration order of a hash map of checkpoint weights, which
  differs from run to run, and only for the keys that matched. The pairing
  is arbitrary; nothing consumes it today. Where: `build_fusion_plan` in
  `crates/nsl-codegen/src/wrga_fusion.rs`, called from `run` in
  `crates/nsl-codegen/src/wrga.rs`. By reading the code only.
- `nsl_wrga_epilogue_fused_lora` (the proposal's §7.3 runtime entry) and
  the module around it have no caller. Where:
  `crates/nsl-runtime/src/peft/ffi.rs`. By reading the code only.
