# Compiler-Exempt Sensitivity Handling (CESH) — Errata

**Companion to** `CESH.pdf`. **Date:** 2026-11-13.

This file lists the statements in the April 2026 CESH proposal that a reader
should not rely on, and says what is true instead. The proposal was checked
in November 2026 against its own arithmetic, the papers it cites, later
literature, and the NSL repository. Each entry says how it was checked; where
a point rests on memory of a paper rather than a fresh reading, the entry
says so. Section numbers are those printed in the PDF.

---

## Part 1 — Errors in the proposal

### E1 — The accuracy targets are better than published results for 4-bit weights and activations

**Proposal.** §5.3: "Naive W4A4 baseline: typically 30-60% perplexity
increase on wikitext-2", "CESH role-specific: <5% perplexity increase",
"CESH + DIGS: <2% perplexity increase", "CESH + DIGS + OSP: within 1% of FP16
baseline". The abstract calls the embedding and head case "the single-largest
class of quantization accuracy failures in LLM deployment".

**Correction.** W4A4 means 4-bit weights and 4-bit activations. QuaRot
reports the following perplexities for Llama-2-7B with 4-bit weights,
activations and key-value cache (RTN is round-to-nearest):

| Method                | Perplexity | Against FP16 |
|-----------------------|-----------:|-------------:|
| FP16 baseline         |       5.47 |            — |
| QuaRot + GPTQ         |       6.10 |       +11.5% |
| QuaRot + RTN          |       8.37 |         +53% |
| OmniQuant             |      14.26 |         2.6× |
| SmoothQuant + RTN     |      83.12 |          15× |

The best result in that table loses 11.5%, more than twice the proposal's
"<5%" target. Naive uniform W4A4 does not cost 30–60%; it
collapses by orders of magnitude. The collapse comes from 4-bit activations
in the transformer body, so exempting the embedding and head from
quantization cannot produce "<5%".

The abstract's premise is unsupported. The common quantization pipelines do
not quantize the embedding or the head at all (see E7; not re-checked against
their source code).

**Source.** https://arxiv.org/abs/2404.00456. The figures were taken from
search-result excerpts of the paper's Tables 1 and 3, not from a full read.
Percentages recomputed.

### E2 — Rotating only the LM head is not free at inference

**Proposal.** §2.3: "No runtime cost (the rotation of activations is fused
into the preceding RMSNorm); inference latency is unchanged."

**Correction.** The LM head is the language-model output projection. A
rotation `H` commutes with RMSNorm only when the norm has no per-channel
gain. With a gain vector `g`,

    (g ⊙ x / rms(x)) · H  ≠  g ⊙ (x·H) / rms(x·H)

so the rotation cannot be folded into the norm. NSL's RMSNorm has a gain
vector (`RMSNorm.weight`, initialised to ones), as the module comment in
`crates/nsl-runtime/src/optim_groups.rs` records.

QuaRot obtains a rotation with no runtime cost by first folding the norm
gains into the adjacent linear layers and then rotating the whole residual
stream: the embeddings and every block's read and write weights. Rotating
only the head leaves two options. One is an explicit Hadamard transform on
every token at inference, which is a runtime cost. The other is a
whole-model rotation, which contradicts the body default in §2.2, "No
rotation tricks by default".

§2.2 also sets "Activations at FP16 (never quantized at this boundary)" for
the head. Unquantized activations remove the main thing a rotation buys.

**Source.** Algebra, and a read of the named NSL file. QuaRot is
https://arxiv.org/abs/2404.00456; only its results tables were consulted (see
E1), so the description of its procedure is not re-checked against the paper.

### E3 — DIGS has no working mechanism as described, and depends on components that do not exist

**Proposal.** §2.4 and §3.3 describe DIGS (Decomposition-Informed Gradient
Sculpting): "confidence = 1.0 - normalize(abs(W_lo))", multiplied into the
head's gradient before the optimizer step. §1.3 says it is effective where
"tokenizer merge-table structure creates predictable decomposition-hostile
directions". §2.4: "CFTP's W_hi/W_lo decomposition is the direct analog to
TCFP's." The only reference is number 8, "Wiemer, B. (2025). DIGS:
Decomposition-Informed Gradient Sculpting for TCFP-12. CodeForge technical
report."

**Correction.**

- The reference is an unpublished report. It could not be verified.
- `W_lo` is the rounding residual of `W` on the FP8 grid, so `|W_lo|` is
  roughly `|W|` times a pseudo-random fraction. The mask is therefore an
  inverse-magnitude mask plus noise. Nothing in the definition connects it to
  tokenizer merge-table structure.
- If `normalize` divides by the maximum, the weight with the largest residual
  gets confidence 0 and is frozen.
- Under Adam or AdamW a slowly varying per-element gradient scale `c`
  cancels. The first moment becomes `c·m` and the second `c²·v`, so the
  update is `c·m / (c·√v + ε) = m / (√v + ε/c)`. The scale only changes the
  effective `ε`. NSL routes the `embedding` and `head` parameter roles to
  AdamW (`crates/nsl-codegen/src/param_roles.rs`), so on exactly the tensors
  DIGS targets it is close to a no-op.
- The decomposition DIGS reads does not exist in NSL. The CFTP proposal in
  this folder covers fused gradient accumulation (FASE) and sequence packing;
  it defines no `W_hi`/`W_lo` split. A search of the repository finds no
  `W_hi`/`W_lo`, CFIP, CTQS, GEBA or TCFP anywhere. Three of the five
  contributions listed in the abstract depend on these names.

**Source.** Algebra for the optimizer argument; a search of the repository
and of the CFTP proposal text for the missing components.

### E4 — The low-rank embedding claims contradict each other

**Proposal.** Abstract: "15-25% embedding memory reduction via optional
low-rank factorization". §2.5: "the compression ratio is (V · d)/(V · r +
r · d) ≈ 11× for V = 256000", and "This is equivalent to the original
lookup at the cost of an extra small matmul per token." §3.4: "the factored
matmul is 8192 × 256 × 4096 = 8.6 GFLOPs".

**Correction.**

- With V = 256,000, d = 3,072 and r = 256 the formula gives
  786,432,000 / 66,322,432 = 11.86×, which is a 91.6% reduction. That is an
  order of magnitude away from the abstract's 15–25%.
- The factored lookup is not equivalent to the original. A product of a
  V × 256 and a 256 × 3,072 matrix has rank at most 256; the original matrix
  can have rank up to 3,072. The two agree only if the original already has
  rank 256 or less.
- 8192 × 256 × 4096 = 8.59 × 10⁹ is a count of multiply-add pairs. Counted as
  floating-point operations it is 17.2 GFLOP.
- Two consequences are not addressed (both from memory of the literature, not
  re-checked against a source). When the head is tied to the embedding, as it
  is in this repository's coder models (`x @ self.embed.transpose(0,1)`),
  factorizing the embedding caps the rank of the logits at r, which is a
  softmax bottleneck. And trained embeddings are not low-rank in the sense
  the §4.2 mitigation assumes, a rank that "captures 99%+ of the spectral
  energy" at small r.

**Source.** Recomputed from the proposal's own figures; linear algebra for
the rank argument.

### E5 — Gemma-7B's embedding share is 9.2%, not 20%

**Proposal.** §1.1 table row: "Gemma-7B 256,000 3,072 1.57 B 20%", and
"Gemma-7B: 20% of weights are embedding+head". §2.5 builds on "Gemma-7B's
20% case", and §5.2 repeats it.

**Correction.** Gemma uses one matrix for both the input embedding and the
output head. The Gemma report lists 786,825,216 embedding parameters and
7,751,248,896 non-embedding parameters, so the share is

    786,825,216 / (786,825,216 + 7,751,248,896) = 9.2%

The proposal's 1.57 B is 2 × 256,000 × 3,072, which counts the shared matrix
twice. The correct figures are 0.787 B and 9.2%, so the proposal's main
motivating case is overstated by about a factor of two.

The Llama-3 and Phi-3 rows are arithmetically right provided those models
keep separate embedding and head matrices.

**Source.** https://arxiv.org/abs/2403.08295; share recomputed.

### E6 — The policy for a tied embedding and head is undefined

**Proposal.** §2.1: "Tied embedding/head (a common pattern) → both layers
get the shared role inferred from position."

**Correction.** The proposal never says which policy the shared role
carries. The §2.2 embedding default is "Keep weights at FP16"; the LM-head
default is "Weights at FP16 or FP8" with a rotation when quantized. For a
tied matrix these conflict, and tied is the case this repository's models
use.

### E7 — Prior work is misdescribed in §1.1 and §1.2

The first three items rest on memory of the papers and reference code and
were not re-checked against the sources.

- §1.1, "the embedding layer originates them". LLM.int8() does not say that
  the embedding layer originates outlier features. "Massive Activations"
  (the proposal's reference 9) reports them appearing abruptly after the
  early blocks, not at the embedding.
- §1.1, "Setting outlier features to zero drops performance by 20%+". The
  figure above 20% in LLM.int8() is the drop in top-1 attention softmax
  probability mass, not a drop in task performance.
- §1.2, AWQ "keeps first and last layers at FP16 by default" and for GPTQ
  "embedding and head layers often end up higher bit budget". The reference
  GPTQ and AWQ code quantizes only the linear layers inside the transformer
  blocks. The embedding and head are skipped, not given a higher bit budget
  by calibration.
- §1.1, the AutoQuantize score "consistently ranks these layers as 'keep in
  higher precision.'" This claim could not be verified.

The §2.6 description of Outlier-Safe Pre-Training (OSP) was checked and is
accurate: a 1.4B-parameter model trained on 1T tokens, excess kurtosis 0.04
against 1818.56, a 4-bit average score of 35.7 against 26.5, and about 2%
training overhead. Source: https://arxiv.org/abs/2506.19697.

---

## Part 2 — Superseded by later work

- **Rotation methods now target block-scaled FP4.** TORQ
  (https://arxiv.org/abs/2605.19561), DuQuant++ (April 2026), INFOQUANT
  (https://arxiv.org/abs/2605.26175), ReSpinQuant
  (https://arxiv.org/abs/2604.11080) and ConQuR
  (https://arxiv.org/abs/2605.10793) design their rotations around the
  scaling groups of the MXFP4 and NVFP4 formats. These were found by search
  and only their abstracts were read. For a compiler aimed at Blackwell
  hardware, a fixed Hadamard on one layer is no longer the baseline, and the
  policy should be stated in block-scaled formats rather than as "W4".
- **Role-aware defaults were already shipping practice.** The llama.cpp
  quantizer has `--output-tensor-type` and `--token-embedding-type` options
  (confirmed by a search of copies of its `tools/quantize/quantize.cpp`). Its
  K-quant presets also keep the output and token-embedding tensors at a
  higher type (from memory; not re-checked against the source). The §1.2
  sentence "the formalization in a compiler is missing" needed this citation
  even in April 2026.

---

## Part 3 — What the code implements

No part of the proposal is implemented: the repository has no `@role`,
`@quant_override`, `@digs`, `LowRank`, `outlier_safe`, Hadamard head
rotation, or WGGO quantization policy under `crates/`, `stdlib/` or
`models/`.

Older, adjacent code that a reader might mistake for it:

- **Parameter roles.** `@param_role("embedding"|"head"|"hidden")`, plus
  inference from `embedding_lookup(self.x, …)`, in `classify_param_roles`
  (`crates/nsl-codegen/src/param_roles.rs`). Reachable from NSL source. It
  drives Muon-versus-AdamW routing and `no_decay=[…]` only; quantization
  never reads it.
- **Part of the OSP recipe.** The Muon optimizer and that role-based routing
  exist. Single-Scale RMSNorm and the embedding projection do not.
- **The `quant` block.** `quant static q from m:` with `default: int8|int4`
  and `per_tensor|per_channel|per_group` is the only quantization path NSL
  source can reach (`crates/nsl-codegen/src/stmt_quant.rs`,
  `nsl_qtensor_quantize` in `crates/nsl-runtime/src/quantize.rs`). It
  quantizes and immediately dequantizes into a new f32 model; there is no
  packed inference. It treats every field alike unless the user writes
  `exclude:` globs.
- **AWQ and GPTQ.** `awq4`, `gptq4` and `gptq8` parse in a `quant` block but
  are refused by `crates/nsl-semantic/src/quant_requests.rs`. The runtime
  code behind them (`crates/nsl-runtime/src/awq.rs`, `gptq.rs`) is reachable
  only from Rust unit tests. Neither follows its published algorithm
  faithfully, so the refusal is the right behaviour.
- **`@quantize(dtype="awq4")`.** Reachable. It runs a calibration pass that
  writes per-channel activation scales to a sidecar file
  (`crates/nsl-codegen/src/calibration/awq_hook.rs`). The only consumer sits
  behind the refused `awq4` block, so the output is never used.
- **`@fp8_compute`.** Reachable. It sets one flag per function, so a
  decorated forward runs the LM-head matmul in FP8 like any other matmul.
- **NVFP4 quantization with an optional Hadamard transform.**
  `nsl_nvfp4_quantize` in `crates/nsl-runtime/src/fp8.rs`. Not reachable:
  `extract_fp4_compute_decorator` in `crates/nsl-codegen/src/fp8.rs` has no
  caller.
- **BitNet loader.** `crates/nsl-codegen/src/bitnet/loader.rs` ternarizes
  only attention and MLP projections, chosen by a name-suffix whitelist. This
  is the one other place where embedding and head are left alone.

No shipped model uses `quant`, `@quantize` or `@fp8_compute`; only
`examples/m16_quantize.nsl` and the `examples/m35_*` files do.

One document in the repository still misdescribes this area:
`docs/summaries/06-implementation-status.md` lists AWQ and GPTQ as
"Production"; `STATUS.md` is the accurate one. `docs/wiki/Glossary.md` used
to expand CESH as "Compiler-Evaluated Speculative Hypothesis" and describe
speculative decoding, which is not what `CESH.pdf` proposes; it now gives
the real title and says the proposal is not implemented.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Keep one piece: role-aware default exclusion. Feed `classify_param_roles`
   (plus a runtime check that the tensor is rank 2) into the `quant` block so
   embedding, head and non-matrix parameters are skipped unless named, print
   the per-field decisions, and measure logit error and perplexity change on
   `coder50m`. The `quant` block must first handle real models.
2. Drop DIGS, the head-only "free" rotation, and low-rank for tied embeddings.
3. If outlier-safe training is wanted, add Single-Scale RMSNorm and an
   embedding projection to the standard library and measure kurtosis.
4. Restate the policy for block-scaled FP4 and the targets against E1's table.

---

## Part 5 — Implementation defects found in the same review

The review that checked the proposal also ran the code. CESH itself is not
implemented, so this part lists what was found wrong in the adjacent code
that Part 3 describes, as distinct from the proposal's own errors. Fixed
items are described in `CHANGELOG.md` under "Unreleased". The open list is
as of 2026-11-13.

### Fixed

- **`quant` blocks destroyed any tensor whose values excluded zero.** The
  zero point was clamped into the code range and every value was then
  decoded against the wrong offset, so a normalisation gain of
  [0.90 .. 1.04] came back as 0.14 in every element at int8 and at int4.
  The quantized range now always includes zero (`compute_scale_zp` in
  `crates/nsl-runtime/src/quantize.rs`, with round-trip unit tests there).
- **The Glossary described a different idea under the name CESH.**
  `docs/wiki/Glossary.md` expanded it as "Compiler-Evaluated Speculative
  Hypothesis" and described speculative decoding. It now gives the real
  title and says the proposal is not implemented.

### Open, in the adjacent code

- **A `quant` block reads a strided view in storage order.** With a model
  field that is a transposed 3×2 view, `x @ w` should be `[-4.5, 4.5]`; the
  quantized model returns `[-1.49, 1.49]` and reports nothing. Where:
  `nsl_qtensor_quantize` in `crates/nsl-runtime/src/quantize.rs`.
  Reproduced by running. Right behaviour: materialize the view first.
- **The other quantizer entry points read tensors the same way, and none
  checks the device.** `nsl_awq_quantize` (`crates/nsl-runtime/src/awq.rs`)
  and `nsl_gptq_quantize` (`crates/nsl-runtime/src/gptq.rs`) also take
  elements in storage order; neither they nor `nsl_qtensor_quantize` look
  at which device holds the tensor. By reading the code only. Right
  behaviour: stage a GPU tensor to the host, or refuse it.
- **`gptq_quantize_obq` does worse than plain rounding on GPTQ's own
  objective.** For a 128×64 weight in groups of 32 with correlated inputs,
  the output error tr(ΔᵀHΔ) is 0.615 for round-to-nearest, 0.769 for this
  code and 0.254 for reference GPTQ. Where:
  `crates/nsl-runtime/src/gptq.rs`; NSL source cannot reach it. Numeric
  check.
- **`gptq_quantize_obq` moves its group scales and never updates H⁻¹.**
  Each group's scale and zero are recomputed from compensated weights at
  every column, so earlier codes decode with a later scale. Numeric check,
  on a second matrix of that shape: fixing this alone takes the error from
  0.665 to 0.268, and the upper Cholesky factor of H⁻¹ then gives 0.240,
  the reference value. Group parameters should come from the original W.
- **`nsl_nvfp4_quantize` with the Hadamard flag returns transformed
  values.** The result is in the Hadamard domain, in the original shape,
  with no inverse applied and no record that one is needed: relative error
  1.43, against 0.106 once the inverse is applied and 0.104 with the flag
  off. Where: `crates/nsl-runtime/src/fp8.rs`; NSL source cannot reach it.
  Numeric check.
- **AWQ pre-scaling and the AWQ matmul disagree on the input-channel
  axis.** `nsl_awq_pre_scale_weight` scales columns and documents them as
  input channels; `nsl_awq_matmul` treats rows as input channels. The
  inverse scale is applied nowhere, so a pre-scaled weight would change the
  model's function. Where: `crates/nsl-runtime/src/awq.rs`, behind the
  refused `awq4` block. By reading the code only.
- **`@quantize(dtype="awq4")` computes scales that nothing consumes.** The
  decorator is accepted and runs a calibration pass that writes a sidecar
  file. Its only reader is in `compile_quant_block`
  (`crates/nsl-codegen/src/stmt_quant.rs`), behind the refused `awq4`
  block. By reading the code only. Right behaviour: drop the calibration
  trigger or wire a consumer.
- **The code named AWQ does not use its calibration.** `nsl_awq_quantize`
  (`crates/nsl-runtime/src/awq.rs`) ignores its calibration argument and
  rounds to each group's minimum and maximum. The hook in
  `crates/nsl-codegen/src/calibration/awq_hook.rs` collects per-channel
  max |x| where AWQ uses the mean, and the exponent is fixed at 0.5 with
  no search over it and no clip search. By reading the code only.
- **GPTQ accepts a bit width and ignores it.** `gptq_quantize_cpu` passes
  `bits` to `gptq_quantize_obq`, which always calls `quantize_int4`, so a
  request for 8 bits would produce 4-bit codes. Where:
  `crates/nsl-runtime/src/gptq.rs`; NSL source cannot reach it. By reading
  the code only.
- **The NVFP4 Hadamard flag is silently skipped unless the tensor's total
  length is a power of two.** When it does apply, it runs over the whole
  flattened tensor. Blocks default to 256 elements with unconstrained f32
  scales, which is not the NVFP4 format. Where: `quantize_nvfp4` in
  `crates/nsl-runtime/src/fp8.rs`. By reading the code only.
- **A `quant` block cannot handle a model that has a sub-model, and what
  it does varies from run to run.** Of six runs on a model with one
  sub-model field, three aborted with "invalid tensor handle"; three ran
  the inner model's layout on the outer's data and died in a matmul.
  `compile_quant_block`, `crates/nsl-codegen/src/stmt_quant.rs`. Reproduced
  by running. It should use the source's type and recurse into sub-models.
- **`docs/summaries/06-implementation-status.md` calls AWQ and GPTQ
  "Production".** Its rows for M21-M22 and M35 and its feature list say
  so. NSL source can reach neither, and the items above show neither is
  correct. `STATUS.md` is accurate. By reading the code only.
- **The `gptq.rs` module comment names a step the code does not perform.**
  It says "Compute Cholesky factorization of H^{-1}"; the code factors H
  and inverts it once. Where: `crates/nsl-runtime/src/gptq.rs`. By reading
  the code only.
- **The GPTQ tests cannot detect the defects above.**
  `test_gptq_obq_reduces_error_vs_rtn` compares weight mean-squared error,
  which a correct GPTQ increases; the others assert only that an error is
  under a fixed bound. Where: `crates/nsl-runtime/src/gptq.rs`. By reading
  the code only. Right behaviour: compare output error with
  round-to-nearest and with a stored reference.
- **The AWQ "end-to-end" tests check only that bytes change.**
  `awq_quantize_with_scales` in `crates/nsl-runtime/src/awq.rs` is, by
  design, an int8 abs-max stand-in, and its tests assert only that the
  output bytes differ when the scales differ. By reading the code only.
- **The only test that calls `nsl_nvfp4_quantize` compares the GPU result
  with the CPU result of the same code**, so the missing inverse passes.
  Where: `crates/nsl-runtime/tests/fp8_gpu_device_guard.rs`. By reading
  the code only.
