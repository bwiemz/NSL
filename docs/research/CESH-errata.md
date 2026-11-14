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
  quantizes and immediately dequantizes into a new f32 copy of the model,
  descending into sub-models; there is no packed inference. It quantizes
  every tensor field alike, whatever its role, unless the user writes
  `exclude:` globs.
- **AWQ and GPTQ.** `awq4`, `gptq4` and `gptq8` parse in a `quant` block but
  are refused by `crates/nsl-semantic/src/quant_requests.rs`, and code
  generation refuses them too. The runtime code behind the names
  (`crates/nsl-runtime/src/awq.rs`, `gptq.rs`) is reachable only from Rust
  unit tests. `gptq.rs` follows the published algorithm and is tested
  against a NumPy reference. `awq.rs` is per-group round-to-nearest into a
  4-bit packed format; it is not AWQ, and its module comment says so. The
  refusal remains the right behaviour: nothing feeds either routine
  calibration data from NSL source.
- **`@quantize(dtype="awq4")`.** Refused, in every form of the decorator.
  It used to select the model for an activation-statistics pass and had
  activation-retention copies compiled into the model's forward for it;
  nothing read the statistics. The hook that collected them
  (`awq_hook.rs` under `crates/nsl-codegen/src/calibration/`) and the
  discovery that found the models are deleted.
- **`@fp8_compute`.** Reachable. It sets one flag per function, so a
  decorated forward runs the LM-head matmul in FP8 like any other matmul.
- **Block-scaled FP4 rounding with an optional Hadamard rotation.**
  `nsl_nvfp4_quantize` in `crates/nsl-runtime/src/fp8.rs`. Not reachable:
  `@fp4_compute` is a refused decorator. Despite the name it is not the
  NVFP4 data format: it takes any block size and keeps f32 scales, and the
  rotation runs over the whole flattened tensor.
- **BitNet loader.** `crates/nsl-codegen/src/bitnet/loader.rs` ternarizes
  only attention and MLP projections, chosen by a name-suffix whitelist. This
  is the one other place where embedding and head are left alone.

No shipped model uses `quant`, `@quantize` or `@fp8_compute`; only
`examples/m16_quantize.nsl` and the `examples/m35_*` files do.

`docs/summaries/06-implementation-status.md` and `SPECIFICATION.md` used to
list AWQ and GPTQ as available ("Production"); both now say that the
routines are runtime library code NSL source does not reach.
`docs/wiki/Glossary.md` used to expand CESH as "Compiler-Evaluated
Speculative Hypothesis" and describe speculative decoding, which is not what
`CESH.pdf` proposes; it now gives the real title and says the proposal is
not implemented.

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

- **A `quant` block read a strided view in storage order.** With a model
  field that is a transposed 3×2 view, `x @ w` is `[-4.5, 4.5]`; the
  quantized model returned `[-1.49, 1.49]` and reported nothing. The
  quantizer now reads every tensor in logical order (`logical_values_f64`
  in `crates/nsl-runtime/src/quantize.rs`); a test runs that model through
  a `quant` block.
- **The other quantizer entry points read tensors the same way, and none
  checked the device.** `nsl_awq_quantize`, `nsl_awq_matmul`
  (`crates/nsl-runtime/src/awq.rs`), `nsl_gptq_quantize`,
  `nsl_gptq_quantize_ext` and `nsl_gptq_hessian_add_batch`
  (`crates/nsl-runtime/src/gptq.rs`) took elements in storage order, and
  neither they nor `nsl_qtensor_quantize` looked at which device holds the
  tensor. All of them now read through the same routine, which follows
  strides and stages a GPU tensor to the host, and a `quant` block moves
  each quantized field back to the device its source is on. Tests cover a
  strided view on the host, a tensor and a strided view on a GPU at each
  entry point, and a `quant` block on a model that was moved to the GPU.
- **A `quant` block could not handle a model that has a sub-model, and
  what it did varied from run to run.** It took the model type from a scan
  of a hash map and passed every field, sub-model pointers and `int`
  fields included, to the tensor quantizer: of six runs on a model with
  one sub-model, three aborted with "invalid tensor handle" and three ran
  the inner model's layout on the outer's data. It now uses the source
  variable's type and copies the model field by field: tensors are
  quantized, sub-models and fixed-size arrays of sub-models are copied
  recursively, other fields are carried over. An `exclude:` pattern
  matches a field's dotted path (`inner.w`) or its own name
  (`compile_quant_block` in `crates/nsl-codegen/src/stmt_quant.rs`).
- **`gptq_quantize_obq` did worse than plain rounding on GPTQ's own
  objective**, because it moved its group scales and never updated H⁻¹.
  Each group's scale and zero were recomputed from compensated weights at
  every column, so earlier codes decoded with a later scale, and the
  inverse Hessian was used unchanged throughout. For a 128×64 weight in
  groups of 32 with correlated inputs the output error tr(ΔᵀHΔ) was 0.769
  against 0.615 for round-to-nearest. Group parameters now come from the
  original W and each rounding error is spread with the upper Cholesky
  factor of H⁻¹. On a problem of that shape the output error is 0.0108
  against 0.0474 for round-to-nearest, and it equals a NumPy
  implementation of the published algorithm to seven digits on three
  shapes; those numbers are in the unit tests.
- **GPTQ accepted a bit width and ignored it.** A request for 8 bits
  produced 4-bit codes. The packed format has one width, so any other is
  now an error (`GptqError::UnsupportedBits`), as is a Hessian that is
  neither K×K nor a K-element diagonal, which used to fall back to
  rounding without a message.
- **The `gptq.rs` module comment named a step the code did not perform.**
  It said "Compute Cholesky factorization of H^{-1}" where the code
  factored H and inverted it once. The comment now describes the
  algorithm, and the code performs that step.
- **The GPTQ tests could not detect any of this.**
  `test_gptq_obq_reduces_error_vs_rtn` compared weight mean-squared error,
  which a correct GPTQ increases; the others asserted only that an error
  was under a fixed bound. The tests now compare output error with
  round-to-nearest and with stored reference values, and one asserts that
  the weight error rises.
- **`nsl_nvfp4_quantize` with the Hadamard flag returned transformed
  values.** The result was in the Hadamard domain, in the original shape,
  with no inverse applied and no record that one was needed. It now
  returns the reconstruction in the tensor's own basis. On a 1,024-element
  test vector the returned tensor's relative error is 0.122 (0.065 with
  the flag off), where the transformed values scored 1.435; a unit test
  holds those numbers, computed separately in NumPy.
- **The NVFP4 Hadamard flag was silently skipped unless the tensor's
  total length is a power of two.** A request it cannot honour is now an
  error (`NvFp4Error::HadamardLength`; fatal with a message from the FFI
  entry point). The rotation still runs over the whole flattened tensor,
  and blocks still default to 256 elements with unconstrained f32 scales:
  the routine is block-scaled FP4 rounding, not the NVFP4 format, and its
  comments now say so (`crates/nsl-runtime/src/fp8.rs`).
- **The only test that called `nsl_nvfp4_quantize` compared the GPU result
  with the CPU result of the same code**, so the missing inverse passed.
  That device test stays for what it checks; the reference test above
  calls the same entry point on the host.
- **AWQ pre-scaling and the AWQ matmul disagreed on the input-channel
  axis.** `nsl_awq_pre_scale_weight` scaled columns and documented them as
  input channels; `nsl_awq_matmul` treats rows as input channels; and the
  inverse scale was applied nowhere, so a pre-scaled weight would have
  changed the model's function. `nsl_awq_pre_scale_weight` and the code
  in `compile_quant_block` that called it, behind the refused `awq4`
  block, are removed. `nsl_awq_matmul` now checks that its input is
  `[M, K]` for a `[K, N]` weight.
- **The code named AWQ did not use its calibration.** `nsl_awq_quantize`
  ignored its calibration argument and rounded to each group's minimum
  and maximum. It is not AWQ and no longer presents itself as AWQ: the
  module comment in `crates/nsl-runtime/src/awq.rs` and the status
  documents describe it as per-group round-to-nearest, and a non-null
  calibration argument is refused instead of ignored. The calibration
  hook that collected a per-channel max |x| (where AWQ uses the mean) has
  since been deleted; see the last item of this list.
- **The AWQ "end-to-end" tests checked only that bytes change.**
  `awq_quantize_with_scales`, an int8 abs-max stand-in, existed for tests
  that asserted its output differed when the scales differed. The
  stand-in and those tests are removed.
- **`docs/summaries/06-implementation-status.md` called AWQ and GPTQ
  "Production".** Its rows for M21-M22 and M35 and its feature list now
  say that NSL source reaches neither, and what each routine is.
- **`@quantize(dtype="awq4")` computed scales that nothing consumed.** The
  decorator was accepted and selected the model for a calibration pass
  that wrote per-channel activation maxima to a sidecar file; it also had
  activation-retention copies compiled into the model's forward in every
  build. Nothing read the result. The decorator is now a compile error in
  every form (`crates/nsl-semantic/src/quant_requests.rs`): `awq4`, and
  the bare decorator that meant the same, say that AWQ calibration is not
  implemented, any other dtype says that dtype is not implemented, and
  both point at the `quant` block. The calibration hook, the discovery of
  `@quantize` models and the tests that exercised only them are deleted.
  The calibration harness itself remains for the gradient-importance
  statistics; no `nsl` command runs it.

### Open, in the adjacent code

Nothing in this list is open.
