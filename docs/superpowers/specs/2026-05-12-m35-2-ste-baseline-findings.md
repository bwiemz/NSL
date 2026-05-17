# V-M35.2a-STE findings — BitNet b1.58 training STE convention

**Pre-implementation verification:** spec §2.6 of `2026-05-12-m35-2a-bitnet-backward-design.md`.
**Date:** 2026-05-12
**Status:** DISCREPANCY

## Summary

Spec §4.1 of `2026-05-12-m35-2a-bitnet-backward-design.md` mandates **clipped STE**
with threshold `|x_int8| == 127` for `absmax_quant_backward`, on the assumption that
this "matches the de-facto b1.58 reference behavior."

The survey below shows the opposite: every accessible canonical b1.58 source — the
PI.2-verified community port, the most-starred community implementation, and a
training-focused reproduction — uses **vanilla STE (identity passthrough via the
`x + (quant(x) - x).detach()` trick)**, with no clip-mask on the backward gradient.
Microsoft has never published training code; their public artifacts are inference
only (`microsoft/BitNet`) or a "Training Tips" PDF (`microsoft/unilm/bitnet/`)
that does not include executable STE code in version control.

Per spec §4.4 ("If findings show a different convention, the emitter adopts the
verified convention"), the M35.2a emitter should be amended to vanilla STE before
backward-kernel code is written. Details and decision in §"Spec impact" and
§"Outcome" below.

## Sources surveyed

### 1. Microsoft training code

- **Repo:** `microsoft/unilm` (subdirectory `bitnet/`)
- **Path / file:** N/A — only `README.md` and `The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf` are committed.
- **Commit SHA (pinned):** `66d1f839c86698fa2d59c2c4ee361c58af191abe` (latest commit touching `bitnet/`)
- **STE implementation excerpt:** N/A — no `.py` files in `microsoft/unilm/bitnet/`.

The companion repo `microsoft/BitNet` is the official inference framework
(`bitnet.cpp`); it explicitly advertises itself as inference-only and contains
no training code or STE backward.

The `Training_Tips_Code_FAQ.pdf` is referenced as the closest thing to a Microsoft
training reference, but it is a non-version-controllable PDF (not text/code) and
the contained snippets are not pinned to a commit-line we can anchor against.
Treated as informal documentation, not as a canonical anchor.

- **Conclusion:** Microsoft training code is not publicly available as version-controlled source. No canonical Microsoft anchor exists for the STE backward.

### 2. HuggingFace official examples

- **HuggingFace `microsoft/bitnet-b1.58-2B-4T` model card:** points users to
  `microsoft/BitNet` (inference framework) and the technical report
  (arXiv:2504.12285). No training code is linked from the model card.
- **HuggingFace `transformers` library `BitNetModel`:** present at
  `src/transformers/models/bitnet/modeling_bitnet.py` and
  `src/transformers/models/bitnet/modular_bitnet.py` (HF transformers main
  commit `7ee56fc257b1dec80a03b7e4808aafe671699ace`). The implementation is
  inference-only — it loads pre-quantized weights and reproduces the BitNet
  attention/MLP topology, but contains no `BitLinear` STE backward or
  `torch.autograd.Function` for activation quantization. The HF docs page for
  the model explicitly warns that the standard `transformers` execution path
  is not optimized and does not implement the BitNet kernels.
- **STE implementation excerpt:** N/A — HF official does not ship a training-mode
  BitLinear.
- **Conclusion:** HuggingFace ships no official training-mode STE for BitNet b1.58.
  The recommended path for fine-tuning is `microsoft/bitnet-b1.58-2B-4T-bf16`
  (the BF16 master weights) plus the user's own STE implementation; no STE
  template is provided.

### 3. Community reproductions

#### 3.1 `1bitLLM/bitnet_b1_58-3B` (PI.2-verified canonical port)

- **URL + commit:** `https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/af89e318d78a70802061246bf037199d2fb97020/utils_quant.py`
- **HF revision SHA (pinned):** `af89e318d78a70802061246bf037199d2fb97020` (lastModified 2024-03-29)
- **STE excerpt (verbatim):**

```python
def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.type(dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)


class BitLinear(nn.Linear):
    def __init__(self, *kargs, weight_bits=1, input_bits=8, **kwargs):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        quant_input  = input       + (activation_quant(input,  self.input_bits)  - input      ).detach()
        quant_weight = self.weight + (weight_quant   (self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
```

- **Notes:** This is the canonical community port previously verified for M35.1's
  forward-path logit match (per NSL project memory). No `torch.autograd.Function`
  is defined. The STE is the **detach trick**: `x + (quant(x) - x).detach()`. The
  forward result equals `quant(x)`; the backward gradient passes through as
  `dL/dx = dL/d(quant(x))` (identity), because only the un-detached `x` carries
  gradient. **No clip-mask is applied — vanilla STE, not clipped STE.**

#### 3.2 `kyegomez/BitNet` (most-starred community implementation)

- **URL + commit:** `https://github.com/kyegomez/BitNet/blob/b8d2708001324644eb968788fa50761b92e7c4e2/bitnet/bitlinear.py`
- **Commit SHA (pinned):** `b8d2708001324644eb968788fa50761b92e7c4e2`
- **STE excerpt (verbatim):**

```python
def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)

        # STE using detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w      + (weight_quant   (w     ) - w     ).detach()
        y = F.linear(x_quant, w_quant)
        return y
```

- **Notes:** Same detach-trick vanilla STE. Inline comment "STE using detach"
  states the convention explicitly. No clip mask, no
  `torch.autograd.Function`. Cites the BitNet papers in the repo README. Note:
  the weight quantizer here is "binary sign" not b1.58 ternary, but the
  **activation** STE — which is what spec §4.1 mandates clipping for — is
  identical vanilla form. The same `(activation_quant(x) - x).detach()` idiom
  applies in all BitNet flavors in this repo.

#### 3.3 `AlarioAI/bitnet` (training-focused reproduction)

- **URL + commit:** `https://github.com/AlarioAI/bitnet/blob/ab8e046d8db8e84de75763086f156e544630852f/bitnet/nn/quantization.py`
- **Commit SHA (pinned):** `ab8e046d8db8e84de75763086f156e544630852f`
- **STE excerpt (verbatim):**

```python
def ste_ternary(weight: Tensor) -> Tensor:
    """Quantize weights to {-1, 0, +1} using straight-through estimator.
    BitNet b1.58 formula: W_tilde = RoundClip(W / (gamma + eps), -1, 1)
    where gamma = mean(|W|)
    """
    gamma = weight.abs().mean()
    scaled = weight / (gamma + 1e-5)
    quantized = torch.clamp(torch.round(scaled), -1, 1)
    return quantized + (weight - weight.detach())  # STE trick


def quantize_activations(x: Tensor, num_bits: int = 8) -> tuple[Tensor, Tensor]:
    """Absmax quantization of activations to [-Q_b, Q_b].
    Returns quantized tensor and gamma (for dequantization).
    """
    q_b = 2 ** (num_bits - 1)
    gamma = x.abs().amax(dim=tuple(range(1, x.ndim)), keepdim=True).clamp(min=1e-5)
    quantized = torch.clamp(x * q_b / gamma, -q_b + 1e-5, q_b - 1e-5)
    return quantized, gamma
```

- **Notes:** Explicitly tagged "BitNet b1.58 formula" in the docstring; ternary
  weights {-1, 0, +1} matching the b1.58 spec. The STE is the
  `weight + (weight - weight.detach())` rearrangement of the detach trick
  (mathematically identical to `x + (quant(x) - x).detach()`). No
  `torch.autograd.Function` is defined; no clip mask. Repo positions itself as
  "Train and evaluate 1.58 bits Neural Networks" — i.e., training-mode is the
  primary use case, and the STE used in production training is vanilla. This
  is the highest-signal community source on the **training-time** STE
  convention specifically.

## Consensus analysis

Three independent community sources surveyed; **3/3 use vanilla STE via the detach
trick**, **0/3 use clipped STE**, **0/3 define an explicit `torch.autograd.Function`
with a saturation-zeroing backward**. Two of the three (1bitLLM and AlarioAI) explicitly
identify their implementations as the b1.58 formula (citing arXiv:2402.17764 directly).

Microsoft has published no training-mode STE code as version-controlled source.
Their `microsoft/BitNet` repo is inference-only, their `unilm/bitnet/` directory
contains only a README + PDF, and their HuggingFace integration via the HF
`transformers` library ships only inference. The closest thing to a "Microsoft
training reference" is the 2024-03 "Training Tips, Code and FAQ" PDF, which is not
a code anchor and is not citable as an immutable line-of-code reference.

The spec §4.2 rationale for choosing clipped STE — that vanilla STE at 1-2 bit weights
causes 15-30% training divergence per Hubara et al. 2017 — is a **theoretically
defensible position from QAT literature**, but it is at odds with the empirical
behavior of b1.58 training in the wild: the canonical community ports all use
vanilla STE and converge successfully (`1bitLLM/bitnet_b1_58-3B` produced a working
3B-parameter checkpoint, validated against the M35.1 logit match). The Hubara
finding is most directly relevant to weight quantization at extreme low-bit
widths; b1.58's design choice to keep activations at 8 bits (where saturation rate
is bounded and rare) shifts that risk-reward calculation.

This is **not a "vanilla vs. clipped is unsettled in the literature"** situation:
the question being verified by V-M35.2a-STE is specifically what the de-facto b1.58
reference implementations use, and on that question the survey is unanimous in
favor of vanilla STE.

## Spec impact

- **DISCREPANCY recovery path:** the spec already anticipates this outcome.
  Spec §4.4 states: "STE implementation in `absmax_quant_backward` waits for the
  findings doc to confirm clipped STE matches the b1.58 reference. If findings
  show a different convention (e.g., asymmetric clip at 126), the emitter adopts
  the verified convention." The findings here show the convention is **vanilla
  STE (no clip mask)**.
- **Recommended spec amendment** (to be made before Phase 4 backward-kernel emission):
  - **§4.1:** change `absmax_quant_backward` from clipped STE
    (`is_saturated = (|x_int8| == 127); dX_final = is_saturated ? 0 : dX_pre_STE`)
    to vanilla STE (`dX_final = dX_pre_STE` — identity passthrough; the kernel
    becomes a trivial copy/forward of `dX_pre_STE`, which may collapse into
    `ternary_gemm_backward`'s output write).
  - **§4.2:** retain the Hubara rationale as a "deferred Phase 2.1 option"
    note, not as the v1 mandate. The argument for clipped STE remains valid as
    a *training-stability optimization* worth A/B testing later; it is just
    not the b1.58-canonical baseline that V-M35.2a-STE was scoped to verify.
  - **§4.5 (reference impl):** change `BitLinearSTE(torch.autograd.Function)`
    backward to identity (`return grad_x_pre_ste, None, None, None`); drop the
    `clip_mask = (x_int8.abs() != 127).float()` step. The
    `torch.autograd.Function` form should still be preferred over the detach
    trick for the reference impl (per spec §4.5 idiom-α reasoning), since it
    keeps the fixture-vs-kernel data flow explicit even though the new backward
    is trivially identity.
  - **§5.3 fixtures:** `bf06_saturated_input` and `bf07_partial_saturation`
    remain valuable as **regression** fixtures (they confirm vanilla STE does
    NOT zero the gradient on saturation, which is itself a bit-checkable
    property), but their expected values change: `bf06` expects
    `dX_final = dX_pre_STE` (not 0) on every saturated element.
  - **§4.3 (no config-driven STE):** retained — vanilla being the verified
    baseline does not change the V-P1-A "no preferential modifications"
    argument against a `SteVariant` enum.
- **What does NOT change:** the two-kernel split
  (`ternary_gemm_backward → absmax_quant_backward`), the kernel signatures, the
  fixture file format, the V-P1-A/B/C-D verification framing, and the
  PyTorch-as-reference choice in §4.5. Only the *content* of the backward STE
  computation flips from clip-then-pass to pass.
- **A potential simplification opportunity:** if vanilla STE makes
  `absmax_quant_backward` a trivial identity, an implementer may choose to
  emit `ternary_gemm_backward` to write directly into `dX_final` (skipping the
  `dX_pre_STE` intermediate buffer). This is essentially a free version of the
  Phase 2.1 epilogue-fused `ternary_gemm_backward_with_ste` optimization
  (spec §3.6, §10.1). Worth considering as a Phase 1 simplification rather
  than a deferred Phase 2.1 — but the two-buffer form is also fine if the
  implementer prefers reference-decomposition parity.

## Outcome

**Status:** DISCREPANCY.

**Decision:** **Adopt vanilla STE (identity passthrough; no clip mask) in
`absmax_quant_backward`**, validated by 3-of-3 surveyed community b1.58 sources
(Microsoft training code not publicly available as version-controlled source).
Amend spec §4.1 + §4.2 + §4.5 + §5.3 per the "Recommended spec amendment" list
above before Phase 4 backward-kernel emission begins.

The clipped-STE design retained as a deferred **Phase 2.1 option** if production
training data later shows vanilla STE is causing instability at NSL's training
scale. Spec §4.2's Hubara reasoning is reframed accordingly.

**Canonical anchor reference (per IR-002 one-time anchor discipline):**
- **Primary source:** `1bitLLM/bitnet_b1_58-3B/utils_quant.py`
- **HF revision SHA (immutable pin):** `af89e318d78a70802061246bf037199d2fb97020`
- **File URL (pinned):** `https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/af89e318d78a70802061246bf037199d2fb97020/utils_quant.py`
- **STE function (lines):** `BitLinear.forward` — the two `x + (quant(x) - x).detach()`
  expressions for `quant_input` and `quant_weight`.
- **Rationale for choosing this as the primary anchor:** (a) already PI.2-verified
  for M35.1's forward-path logit match (NSL project memory cites it as the canonical
  community port), making it the most-stable cross-milestone reference;
  (b) explicitly cites and reproduces b1.58 (arXiv:2402.17764) rather than the
  earlier 1-bit BitNet; (c) is a producing-checkpoint reference (a working 3B
  model uses this code), not just a theoretical implementation; (d) HuggingFace
  revision SHAs are immutable, satisfying IR-002.

**Secondary corroborating anchors** (not used for fixture generation; cited for
consensus evidence):
- `kyegomez/BitNet` @ `b8d2708001324644eb968788fa50761b92e7c4e2`, file
  `bitnet/bitlinear.py` (most-starred community implementation).
- `AlarioAI/bitnet` @ `ab8e046d8db8e84de75763086f156e544630852f`, file
  `bitnet/nn/quantization.py` (training-focused reproduction; explicitly
  identified as b1.58 formula).

Future M35.2a kernel implementations validate against this anchor + the 10 JSON
fixtures (per M35.2a spec §5.3) regenerated under the vanilla-STE convention.
Live external sources are not consulted during kernel implementation.
Re-anchoring requires a fresh V-M35.2a-STE round.

## Surprising findings (call-outs for plan reviewers)

1. **Microsoft has not, and may not ever, publish version-controlled b1.58
   training code.** The 2024-03 "Training Tips" PDF is the closest artifact; it
   is not a usable anchor by IR-002 standards. Spec §2.6's research procedure
   ("check Microsoft's official BitNet training code if publicly available")
   should be treated as **expected to return 'not available'** in future
   verification rounds, not as a likely positive signal. M35.2a's reference
   anchor will live in the community-port branch indefinitely.
2. **HuggingFace `transformers`'s `BitNetModel` is inference-only.** Anyone
   looking for a "HF-blessed BitLinear with backward" via the standard
   `transformers` library will not find one; the HF docs explicitly steer
   users away from the standard library path for performance reasons. This is
   not a bug in HF's offering — it is by design (Microsoft's official path is
   bitnet.cpp).
3. **The kyegomez weight quantizer is binary sign, not b1.58 ternary.** kyegomez
   predates the b1.58 paper and uses an older 1-bit BitNet weight scheme.
   However, the **activation quantizer** there — which is what `absmax_quant_backward`
   targets — is identical absmax-int8 with detach STE. Citing kyegomez as a
   b1.58 source is mildly anachronistic on the weight side, but accurate on
   the activation side. For activation-STE consensus, kyegomez is fine.
4. **`AlarioAI/bitnet` is the strongest training-mode signal.** Among the three
   community sources, AlarioAI is the only repo explicitly positioned as a
   training reproduction (the other two ship checkpoints). It uses the
   `weight + (weight - weight.detach())` rearrangement of the detach trick;
   this is the same idiom and the same gradient values, just written
   differently. Anyone reviewing the spec amendment should treat AlarioAI as
   the single highest-signal "what does b1.58 training actually look like in
   the wild" data point.
