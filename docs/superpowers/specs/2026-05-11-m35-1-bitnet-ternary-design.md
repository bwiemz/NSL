# M35.1: BitNet b1.58 Ternary Quantization — Inference

**Milestone:** M35.1 (BitNet inference; Phase 1 of M35)
**Stacks on:** AWQ forward-pass calibration (#125, #127), WGGO Phase 2 backward-pass calibration (#134, #144), M52 weight-aware compilation
**Date:** 2026-05-11
**Status:** Design (Q1–Q6 + Warp Model brainstormed; ready for plan)

---

## 1. Background and Phase 1 scope

NSL has shipped AWQ (activation-aware weight quantization, forward-pass calibration via the AWQ end-to-end pipeline). AWQ stores weights as f16/bf16 with f32 scales; the "quant" is calibration-derived scales, not bit-packed storage. WGGO Phase 2 added backward-pass gradient-importance calibration (PR #146-149, merged 2026-05-11). GPTQ runtime kernels exist in `crates/nsl-runtime/src/gptq.rs` but lack compiler/semantic integration; treated here as in-flight infrastructure, not a shipped feature.

**BitNet b1.58 (Ma et al. 2024) is the first NSL dtype with genuinely sub-byte storage** — ~1.58 bits per weight, packed at 2 bits per trit. The paper reports near-FP16 quality with 1.4× inference speedup and 7.16× memory reduction at 3B scale.

### 1.1 Phase 1 (M35.1) deliverable

Inference-only ternary quantization with ternary as a first-class NSL dtype. The Phase 1/Phase 2 boundary is **"ternary as a first-class NSL dtype"** (Q1 refinement) — not "BitNet inference end-to-end." This boundary makes Phase 2 truly additive: operations on ternary tensors that require gradients.

Phase 1 ships:

- **KIR additions** (~50 lines in `kernel_lower.rs`): `KirType::Tq2Packed` and `KirType::TernaryUnpacked` variants. NSL's source-level type system needs these regardless of which subsystem produces PTX.
- **Parser + type-system integration** for `Tensor<[...], ternary>` in NSL source. Type checks, shape propagation, semantic analysis.
- **Codegen for ternary loads/stores** + the full BitNet forward pass.
- **Packed/unpacked representations as first-class** with explicit conversion ops (`nsl.quant.ternary.pack`, `nsl.quant.ternary.unpack`).
- **Full b1.58 forward pass** matching the paper's numerics: 8-bit absmean activation quantization + ternary GEMM + RMSNorm prologue (CSHA-style fusion).
- **HF checkpoint loader** for the Microsoft BitNet b1.58 3B release (revision-pinned, see §6).
- **End-to-end inference example** in `examples/bitnet_b158_inference.nsl`.
- **Validation:** FP16 ULP logit-match on 32 reference prompts against the pinned HF checkpoint.

### 1.2 Phase 2 (M35.2) — additive, deferred

Phase 2 ships training: STE backward through ternary forward, FP32 shadow weights with explicit `quantize_to_ternary` ops, activation-quant backward, optimizer integration, training loop.

**Phase 2 PRs only modify files in the Phase 2 file list** (see §7.2): `*_backward.rs`, `*_shadow.rs`, `orchestrator_train.rs`. Modifications to any Phase 1 file from a Phase 2 PR constitute a Phase 1 regression, not Phase 2 work — must be filed as a separate Phase 1 fix PR. This makes the additive claim auditable at PR-review time.

### 1.3 Phase 1 → Phase 2 escalation gate

After Phase 1 lands and is exercised on real workloads:

- **Implementation-quality gate:** ≥80% of paper's claimed inference speedup AND ≥80% of memory reduction on the 3B configuration. If not, Phase 2 is deferred pending kernel optimization (file: "BitNet kernel performance optimization").
- **Method-quality gate:** trained b1.58 checkpoints from the HF release achieve perplexity within 5% of equivalent-parameter FP16 baselines on a held-out evaluation set. If not, Phase 2 is deferred indefinitely; the method itself doesn't deliver, and shipping training for a sub-baseline method is wasted scope.

Both gates must pass for Phase 2 to proceed.

---

## 2. Ternary dtype and packed/unpacked representations

### 2.1 Packed representation (at-rest, HBM)

**2 bits per trit, 4 trits per byte.** Encoding: `{00 = -1, 01 = 0, 10 = +1, 11 = unused}`.

**Trit-within-byte ordering matches Microsoft's bitnet.cpp `unpack_weights` function.** Pinned via pre-implementation verification (see §10). The hand-constructed unit test (`tests/bitnet_packed_repr.rs::ordering_matches_bitnet_cpp`) asserts a 4-trit byte unpacks to the expected trit values, independent of the full HF checkpoint load — catches ordering drift in seconds, not via end-to-end logit divergence.

Information-theoretic optimum is 5 trits per byte (1.6 bits/trit; `3^5 = 243 < 256`). We trade ~20% bandwidth efficiency for simpler unpacking (single mask+shift vs division-by-3) and reference-implementation match. See §7 for the deferred Phase 1.5 path if bandwidth-constrained deployment becomes a workload.

### 2.2 Unpacked representation (compute, registers/SMEM)

One trit per i8 or register slot (consumer-dependent). Conversions are explicit ops in NSL source:

```nsl
nsl.quant.ternary.pack(x: Tensor<..., ternary_unpacked>) -> Tensor<..., ternary>
nsl.quant.ternary.unpack(x: Tensor<..., ternary>) -> Tensor<..., ternary_unpacked>
```

Both representations are first-class NSL dtypes with full type-system support. The packed/unpacked distinction is exposed to the user; conversion is never implicit.

### 2.3 SASS-level unpacking discipline

Per-trit unpack **must emit a single SASS instruction** (`BFE.U32` or equivalent bit-field extract) at sm_80 through sm_120. Phase 1's test matrix asserts this via SASS count check (`tests/bitnet_sass_discipline.rs`). Regression to multi-instruction unpacking (e.g., `SHR + AND` sequence) fails the check.

- **sm_80 (Ampere) through sm_120 (Blackwell):** one instruction via `BFE.U32`.
- **sm_75 and earlier:** two instructions (`SHR + AND`). **Out of v1 scope** per existing NSL architecture floor.

### 2.4 Bandwidth concession (documented for future revisit)

Option A's 2-bit packing costs ~20% additional HBM bandwidth vs the 5-per-byte information-theoretic optimum. At BitNet 3B inference scale, this translates to ~0.15 ms per token of additional HBM read time on typical 1 TB/s HBM.

For Phase 1 (correctness/reference-match focus), the simpler kernel path is worth the bandwidth cost. **A future "Phase 1.5: packed-bandwidth optimization" tracks this if bandwidth-constrained deployments become a real workload.** Same discipline as PCA Tier B's §10 deferred risk-management items.

---

## 3. Subsystem architecture: `crates/nsl-codegen/src/bitnet/`

Mirrors the FA-2 v2 / CSHA / WRGA precedent. Dedicated subsystem rather than extending `kernel_lower.rs` — BitNet is a small kernel family with multiple phase emitters (load, activation-quant, GEMM, finalize), and the subsystem pattern keeps phase emitters discoverable and CSHA-hook-callable.

### 3.1 Phase 1 file layout

```
crates/nsl-codegen/src/bitnet/
├── mod.rs                  # Orchestrator: synthesize_kernel(config) -> PtxModule
├── config.rs               # BitNetKernelConfig
├── reference.rs            # CPU reference impl, #[cfg(test)]
└── phases/
    ├── README.md                   # Documents Phase 1 + planned Phase 2 layout
    ├── packed_load.rs              # PUBLIC: HBM→SMEM ternary load + on-the-fly unpack
    ├── quantized_ternary_gemm.rs   # PUBLIC: absmean prologue + ternary GEMM (fused unit)
    ├── finalize.rs                 # PUBLIC: dequant + epilogue
    ├── absmean_quant.rs            # INTERNAL: callable from tests only
    └── ternary_gemm.rs             # INTERNAL: never publicly exposed
```

**Phase 2 stub files are NOT created in Phase 1.** The intended Phase 2 layout (`ternary_gemm_backward.rs`, `quantize_shadow.rs`, `absmean_quant_backward.rs`, `orchestrator_train.rs`) is documented in `phases/README.md` and §7, but the actual files are only added when Phase 2 ships. This avoids dead-code orphan-rot if Phase 2 is deferred indefinitely per §1.3's escalation criterion.

### 3.2 API shape (γ)

Phase emitters take a `KernelContext` and emit PTX fragments into it. They do **not** assume standalone-kernel framing. The orchestrator composes them into a Phase 1 standalone kernel; future CSHA-fused mode bypasses the orchestrator and calls public phase emitters directly from CSHA's hook system, the same way RoPE epilogue is currently invoked.

This matches FA-2 v2's pattern at the **API level**, not just the directory level. `bitnet/phases/quantized_ternary_gemm.rs` is the BitNet analog of `flash_attention_v2/phases/s_compute.rs`.

### 3.3 Phase-emitter visibility split (institutional rule IR-001)

**Public emitters** are the safe composition units (no preconditions a caller can violate):

- `packed_load.rs` — no input invariants beyond standard HBM-tile addressing.
- `quantized_ternary_gemm.rs` — absmean prologue + ternary GEMM body, fused. No partial-quantization state can be observed externally.
- `finalize.rs` — operates on the fused GEMM's output; no preconditions beyond standard dtypes.

**Internal emitters** have input invariants enforced by visibility:

- `absmean_quant.rs` — callable for unit testing but external callers should use `quantized_ternary_gemm`.
- `ternary_gemm.rs` — **never publicly exposed.** Has the "activations must be absmean-quantized" precondition; making it private makes that invariant impossible to violate from outside the subsystem.

This is an instance of institutional rule **IR-001: invariants on data flowing between phase emitters should be enforced by API shape, not docstrings** (see §8).

### 3.4 KIR integration

`kernel_lower.rs` adds two variants to `KirType`:

```rust
pub enum KirType {
    // ... existing variants ...
    Tq2Packed,         // 2-bit packed ternary; 4 trits per byte
    TernaryUnpacked,   // one trit per i8 register slot
}
```

Plus shape/stride logic and a dispatch call to `bitnet::synthesize_kernel(config)` when the op shape matches BitNet's expected pattern (ternary weight × FP16/BF16 activation matmul).

Total `kernel_lower.rs` changes: ~50 lines.

---

## 4. Phase emitter design (Phase 1)

### 4.1 `packed_load.rs` (public)

- Reads packed ternary weights from HBM into SMEM via `cp.async` (depth=2 ping-pong; see §4.5).
- Unpacks on-the-fly into register tiles: single `BFE.U32` per trit; validated at SASS level.
- Output: register tile of int8 ternary values (or directly into `mma.sync` input fragment registers).

### 4.2 `absmean_quant.rs` (subsystem-internal)

- **Per-row reduction:** compute `mean(|x[r, :]|)` over the row's `hidden_dim` elements. Warp-shuffle reduction (asserted parallel by SASS check; serialized per-thread loop fails the check).
- **Per-row quantization:** `q[r, k] = round(clip(x[r, k] / scale[r], -127, 127))` as i8.
- **Storage:** int8 quantized activations + FP32 scales (one per row).
- **Scale dtype is FP32 throughout** (prologue + dequant arithmetic). Matches bitnet.cpp; avoids small-magnitude precision loss on rows with absmean close to FP16 underflow. Per-row storage cost is negligible (4 bytes × batch × seq_len).

**Required test fixtures** in `tests/bitnet_absmean_quant.rs`:

| Fixture | Purpose |
|---|---|
| Zero-magnitude row | Div-by-zero guard. Expected output: all zeros; scale = epsilon-floored or zero. |
| Uniform-magnitude row | absmean = uniform value; quantized output is all ±127. |
| Mixed-sign mixed-magnitude row | Standard case; verifies round-and-clip arithmetic. |
| Single-outlier row | Outlier dominates absmean; verifies reasonable quantization of non-outliers. |

### 4.3 `quantized_ternary_gemm.rs` (public, fused)

The only external path to compute a ternary GEMM. Composes `absmean_quant` (internal) + `ternary_gemm` body (internal) as one PTX-emission sequence. The absmean prologue is structurally inseparable from the GEMM, enforcing the "activations must be quantized first" invariant via API shape (IR-001).

GEMM body:
```
Y[r, c] = scale[r] × Σ_k (X_q[r, k] × W_ternary[k, c])
```

Inner accumulation in i32 (int8 × ternary → up to i9 per product, sum over hidden_dim fits in i32). Dequant to FP16 via FP32-scale multiplication; output Y in FP16 (or BF16 — depends on surrounding model's dtype).

### 4.4 `finalize.rs` (public) — scope discipline

**Required in Phase 1:**
- Dequant FP32 accumulator → FP16/BF16 output.
- Write to HBM with standard tile addressing.

**Optional in Phase 1 (standard transformer block components, supported):**
- Bias add.
- Residual add.

**Deferred to Phase 1.5 or later:**
- **RMSNorm fold for the next BitNet layer's prologue (CSHA-style fusion across layer boundaries).** Adds cross-layer fusion complexity; ships after Phase 1 baseline is validated.

This distinction prevents Phase 1 scope creep — `finalize.rs` ships the required functionality + optional standard components, and explicitly defers cross-layer fusion.

### 4.5 Warp model: cp.async ping-pong, no role split

- **cp.async** (Ampere+) for asynchronous HBM → SMEM transfer. Standard pattern for HBM-bound kernels; BitNet's load-bound nature makes this the right baseline.
- **Ping-pong depth = 2.** Double-buffered: load tile N+1 while computing tile N.
- **No producer-consumer role split.** All warps load + compute uniformly. Simpler than FA-2-style warp specialization.

If BitNet 3B inference at Phase 1's measured speedup doesn't meet §1.3's 80%-of-paper-claims gate, a Phase 1.5 follow-up adds role-split warp specialization. Same precedent as CSHA Tier B (Level 2 pipelining) being a follow-on to Tier A's simpler warp model.

---

## 5. CPU reference implementation (`bitnet/reference.rs`)

Pure Rust, FP32 throughout (no simulated FP16 intermediate rounding — preserves "reference is independent ground truth"). `#[cfg(test)]`-gated; estimated ~150 lines.

### 5.1 Reference correctness, anchored to bitnet.cpp (one-time)

Institutional rule **IR-002: external references (bitnet.cpp, HF checkpoints, etc.) are one-time correctness anchors during initial implementation, not ongoing CI dependencies** (see §8).

**Procedure:**

1. Phase 1 implementer builds bitnet.cpp locally (or uses Microsoft's pre-built binary).
2. Runs bitnet.cpp on ~10 hand-chosen fixtures covering edge cases (zero rows, uniform rows, sign-mixed rows, outlier rows, multi-row matrices).
3. Captures bitnet.cpp's outputs as JSON in `tests/fixtures/bitnet_reference_outputs.json` (~100 KB total).
4. `bitnet/reference.rs` is asserted to match these fixtures **bit-exact** (CPU FP32 arithmetic, no GPU precision loss). This assertion is a Phase 1 unit test that runs in CI.
5. **After Phase 1 lands, the bitnet.cpp dependency disappears from CI.** Future kernel changes validate against the JSON fixtures; future reference changes (rare) re-anchor against bitnet.cpp.

**Fallback if bitnet.cpp isn't buildable on the implementer's platform:** hand-derive golden values for the same 10 fixtures with the derivation documented in code comments. Less rigorous but available.

### 5.2 Tolerance discipline (kernel-vs-reference)

| Output | Tolerance | Rationale |
|---|---|---|
| FP16 GEMM output | `1e-3` relative (cast to FP32, compare) | FP16 ULP; tighter would produce false positives from FP16's representation floor |
| int8 absmean_quant output | bit-exact | Same arithmetic in both implementations; no precision gap |
| FP32 absmean scales | `1e-6` relative | FP32 ULP |

### 5.3 Visibility migration path

Phase 1: `#[cfg(test)]` on the reference module. Internal to `nsl-codegen`'s own test suite.

Future: if another crate needs to consume the reference for its own testing, migrate to a `reference` Cargo feature flag (additive change; doesn't break Phase 1's setup).

---

## 6. Validation harness (HF BitNet b1.58 3B)

### 6.1 Model identity, dual-pinned (IR-002)

**Both pins required:**

- **HF model ID:** to-be-pinned (see §10 pre-implementation verification). Placeholder: `microsoft/BitNet-b1.58-3B`. Actual canonical ID determined at verification time.
- **Revision SHA:** pinned in `tests/fixtures/bitnet_b158_3b_revision.txt`. Anchors against the paper-version checkpoint. HF revisions are immutable.
- **File SHA-256:** pinned in `tests/fixtures/bitnet_b158_3b_sha256.txt`. Validates download integrity.

The revision SHA is the upstream anchor (against what the paper measured); the file SHA-256 is the integrity check (download not corrupted). Together they make validation non-circular: bumping the revision SHA in a future PR is a deliberate version bump (cache invalidates, fresh download, new fixtures); a silent HF Hub change can't drift the test.

### 6.2 Fetch script

`scripts/fetch_bitnet_b158_3b.sh`:
- Reads both pins from `tests/fixtures/bitnet_b158_3b_revision.txt` and `tests/fixtures/bitnet_b158_3b_sha256.txt`.
- Downloads from `<model_id>@<revision_sha>` via HF Hub.
- Validates against file SHA-256.
- Caches in `~/.cache/nsl-tests/bitnet_b158_3b/`.

Linux/macOS only — bash script, not PowerShell-compatible. Windows users use Phase 1 once Windows-compatible tooling is added as a follow-on (see §7).

### 6.3 CI cache key

`bitnet-b158-3b-weights-${revision_sha}`. Revision SHA bump → cache key changes → cache invalidates → fresh download. Prevents cache-poisoning on legitimate version bumps. Same shape as Cargo's lockfile-aware dependency cache.

### 6.4 Prompt set (32 prompts, pinned)

`tests/fixtures/bitnet_b158_phase1_prompts.txt`. Categorized for structural rationale:

- **8 short factual questions** ("What is the capital of France?"). Typical chat use case.
- **8 code-completion contexts.** Partial Python function definitions. BitNet b1.58's strong suit per paper.
- **8 long-context prompts (≥1K tokens).** Exercises full attention path.
- **8 edge-case prompts.** Very short (1 token), very long (close to context limit), unusual unicode.

Adding to the prompt set is an additive spec change. Modifying existing prompts invalidates the reference logits and requires regeneration.

### 6.5 Reference logits

`tests/fixtures/bitnet_b158_3b_reference_logits.bin`. Computed once via bitnet.cpp + HF tokenizer against the pinned prompt set; vendored in the repo (~32 × 32000 vocab × 2 bytes FP16 = ~2 MB). Same institutional pattern as the reference-implementation JSON fixtures (IR-002).

### 6.6 Merge gate

**Logit-level match within FP16 ULP** (`1e-3` relative) on all 32 prompts. The rigorous gate.

**Token-level greedy-sampling match** is a secondary check — sensitive to logit precision; not the primary gate (sampling can amplify small logit perturbations into different token choices, producing false-negative gates).

---

## 7. Out-of-scope (deferred or Phase 2)

### 7.1 Deferred to Phase 1.5 (single-issue follow-on if needed)

- **5-per-byte packed-bandwidth optimization.** Trades the 2-bit packing's simplicity for ~20% bandwidth efficiency at 3B scale (~0.15ms/token at 1 TB/s HBM). Revisit if bandwidth-constrained deployments become a real workload.
- **Producer-consumer warp role split.** Trades uniform-warp simplicity for better latency hiding via warp specialization. Revisit if Phase 1's measured speedup doesn't hit §1.3's 80%-of-paper-claims gate.
- **CSHA-style RMSNorm fold across layer boundaries.** Trades `finalize.rs` simplicity for cross-layer fusion. Revisit after Phase 1 baseline is validated.

### 7.2 Phase 2 (M35.2)

Phase 2 PRs add the following files (NOT created in Phase 1):

- `bitnet/phases/ternary_gemm_backward.rs` — STE backward through the forward GEMM.
- `bitnet/phases/quantize_shadow.rs` — FP32 shadow weights → ternary packed format (optimizer step).
- `bitnet/phases/absmean_quant_backward.rs` — backward through activation quantization (if separately quantizable; otherwise folded into STE).
- `bitnet/orchestrator_train.rs` — training-mode orchestrator composing forward + backward phases.

The intended layout is documented in `crates/nsl-codegen/src/bitnet/phases/README.md` (added in Phase 1).

### 7.3 Out-of-scope platform support

**Phase 1 targets Linux and macOS** (matches existing NSL calibration infrastructure platform guard). Windows support is a separate scope item; `scripts/fetch_bitnet_b158_3b.sh` is bash, not PowerShell-compatible. Windows users of NSL can use Phase 1 once Windows-compatible tooling is added as a follow-on; not in Phase 1's scope.

Same platform-scoping discipline as the WGGO Phase 2 merge-gate spec.

### 7.4 Out of M35 entirely

- **Binary BitNet (original 2023 paper, {-1, +1}).** M35 targets b1.58 (ternary). Original BitNet is a separate future milestone if a workload demands it.
- **CSHA-fused BitNet mode.** Phase 1's (γ) API shape preserves the option; actual fusion lands as a follow-on once Phase 1 is stable.

---

## 8. Institutional discipline principles (IR-001, IR-002)

This spec introduces a project institutional-rules registry at `docs/institutional-rules.md` (created as part of Phase 1's commit 1). Two rules are codified now; the registry is the long-term home for cross-cutting project disciplines surfaced across multiple specs.

### 8.1 IR-001 — API-shape-enforced invariants

**Rule:** Invariants on data flowing between phase emitters (or analogous internal interfaces) should be enforced by API shape, not docstrings.

**Pattern:** When a phase emitter has a precondition on its inputs that can't be type-checked statically, make the emitter subsystem-internal and expose a composed public emitter that establishes the precondition before invoking the internal one.

**Example (this spec):** `ternary_gemm.rs` requires activations to be absmean-quantized first. Rather than document this as a precondition in a docstring, `ternary_gemm.rs` is kept subsystem-internal; the public path is `quantized_ternary_gemm.rs`, which fuses absmean + GEMM. External callers cannot invoke `ternary_gemm.rs` directly.

**Other instances:** The calibration FFI's compile-time `.expect()` for `weight_index_map`'s pre-population is a related discipline. Type-safe builders (e.g., `BitNetKernelConfig` with required fields) are another instance.

### 8.2 IR-002 — External references as one-time anchors

**Rule:** External references (bitnet.cpp, HF checkpoints, third-party implementations) are one-time correctness anchors during initial implementation, not ongoing CI dependencies.

**Pattern:** During initial implementation, capture the external reference's relevant output as committed fixtures. Validate the NSL implementation against the fixtures, not against the live external source. Future regressions are caught by fixture comparisons; future legitimate updates re-anchor against the external source as a one-time step.

**Examples (this spec):**

- Trit-within-byte ordering (§2.1): bitnet.cpp's `unpack_weights` function is inspected once; the convention is pinned in the spec and asserted by a hand-constructed-byte unit test.
- CPU reference correctness (§5.1): bitnet.cpp's outputs on 10 fixtures are captured as JSON; the Rust reference asserts bit-exact match.
- HF model identity (§6.1): the canonical Microsoft revision SHA is pinned; CI fetches from `<model_id>@<revision_sha>` (HF revisions are immutable).

**Anti-pattern this prevents:** ongoing C++ build dependencies in CI; tests that silently drift when upstream pushes updates; "test failed because external service changed" failure modes.

### 8.3 Registry as a living document

`docs/institutional-rules.md` is created in Phase 1's commit 1 with IR-001 and IR-002. Future specs that surface cross-cutting disciplines (anti-treadmill ordering, pre-implementation verification, conservative-skip semantics, etc.) add their own IR-NNN entries. Each rule has a stable identifier so specs can cite by reference.

---

## 9. Verification matrix (per-commit)

Each commit's verification target is independently auditable. **Commit 3 is gating for commits 4-7** — if `bitnet/reference.rs` doesn't match the bitnet.cpp JSON fixtures bit-exactly, no kernel phase can be validated. Resolve commit 3 to green before commits 4-7 begin. If bitnet.cpp produces unexpected outputs (e.g., upstream bug), file discrepancy upstream and fall back to hand-derived golden values per §5.1.

| Commit | Subject | KIR types | Phase emitters | bitnet.cpp ref fixtures | SASS unpack | Reference unit tests | Logit match (3B) |
|---|---|---|---|---|---|---|---|
| 1 | KIR types + dtype scaffolding + `docs/institutional-rules.md` | ✓ added | — | — | — | — | — |
| 2 | Packed/unpacked conversion ops | ✓ used | — | — | — | — | — |
| 3 | Reference impl (`reference.rs`) + bitnet.cpp JSON fixtures | — | — | ✓ captured + matched | — | ✓ self-test | — |
| 4 | `packed_load.rs` + SASS discipline test | — | ✓ packed_load | — | ✓ `BFE.U32` | ✓ vs ref | — |
| 5 | `absmean_quant.rs` (internal) + 4 test fixtures | — | ✓ absmean_quant | — | ✓ warp-shuffle | ✓ vs ref | — |
| 6 | `quantized_ternary_gemm.rs` (public, fused) | — | ✓ fused GEMM | — | ✓ | ✓ vs ref | — |
| 7 | `finalize.rs` + orchestrator | — | ✓ finalize | — | ✓ | ✓ vs ref | — |
| 8 | Parser + type-system integration for `Tensor<..., ternary>` | — | — | — | — | ✓ vs ref | — |
| 9 | HF checkpoint loader + `scripts/fetch_bitnet_b158_3b.sh` + revision/SHA pins | — | — | — | — | — | — |
| 10 | End-to-end logit-match (merge gate) | — | — | — | — | — | **✓ PASS** |

---

## 10. Pre-implementation verification (before any commits)

Three load-bearing identifiers must be resolved before implementation begins. Each is ~15 minutes of work and defers ambiguity from implementation-time to spec-time. Same discipline as the WGGO Phase 2 merge-gate spec's pre-implementation verification steps.

### 10.1 Trit-within-byte ordering (before commit 4)

**Task:** inspect bitnet.cpp's `unpack_weights` function (or equivalent at the source-code level). Document the exact byte-to-trit mapping (which bit positions hold trit 0, trit 1, trit 2, trit 3). Pin the encoding convention in §2.1 of this spec (replace the "pinned via pre-implementation verification" placeholder with the verified mapping). The hand-constructed unit test asserts against this pinned mapping.

**Acceptance criterion:** §2.1 contains an explicit table of `bit position → trit index` for the canonical byte layout, with a citation to the specific bitnet.cpp source line.

### 10.2 Canonical HF model ID (before commit 9)

**Task:** verify the canonical Microsoft BitNet b1.58 3B model on HuggingFace Hub. Search for "BitNet-b1.58" under `microsoft/` namespace; identify the model whose card matches the paper-version checkpoint (released alongside the b1.58 paper). Pin the exact ID + revision SHA in `tests/fixtures/bitnet_b158_3b_revision.txt` before commit 9.

**Acceptance criterion:** The placeholder `microsoft/BitNet-b1.58-3B` in §6.1 is replaced with the verified ID; `tests/fixtures/bitnet_b158_3b_revision.txt` contains a valid revision SHA; the SHA-256 file is populated by downloading the actual artifact.

**Fallback:** If no canonical Microsoft release exists (paper may have been released via different channel), document the alternative source and rationale in §6.1.

### 10.3 GPTQ shipping status (for §1 accuracy)

**Task (already done):** GPTQ has runtime kernels (`crates/nsl-runtime/src/gptq.rs`, 1119 lines, FFI exports including `nsl_gptq_hessian_*`) but no compiler/semantic integration. The §1.0 background framing reflects this: AWQ is the shipped end-to-end pipeline; GPTQ is in-flight infrastructure. No spec changes needed.

---

## 11. Estimated scope

- **Pre-implementation verification:** ~0.5 day (three identifiers, ~15 min each plus context).
- **Spec design (this document):** ~0.5 day, completed.
- **Implementation:** ~3-4 weeks for Phase 1 (10 commits per §9 matrix), assuming bitnet.cpp builds cleanly and the canonical HF model ID resolves without ambiguity.
- **Total:** ~4 weeks from spec-write to PR-ready.

If pre-implementation verification surfaces non-trivial issues (e.g., bitnet.cpp isn't buildable, HF model ID is ambiguous, requires alternative source), scope extends to ~5 weeks.

---

## Appendix A: References

- **Issue:** to be filed when implementation begins.
- **Related specs:**
  - `docs/superpowers/specs/2026-04-22-awq-real-subprocess-completion-design.md` — AWQ pipeline that this stacks on.
  - `docs/superpowers/specs/2026-05-06-134-decouple-calibration-design.md` — calibration architecture (WGGO Phase 2 backward) shipped concurrently.
- **Research papers (NotebookLM notebook #8 "Ternary & Extreme Quantization"):**
  - Ma et al. 2024 — *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (BitNet b1.58).
  - Wang et al. 2023 — *BitNet: Scaling 1-bit Transformers for Large Language Models* (original binary BitNet).
  - Microsoft Research — bitnet.cpp (reference C++ implementation; one-time correctness anchor per IR-002).
  - Li et al. 2016 — *Ternary Weight Networks* (TWN; precedent for ternary quantization).
- **Code references (post-Phase-1 expected paths):**
  - `crates/nsl-codegen/src/bitnet/` — new subsystem (this spec's primary deliverable).
  - `crates/nsl-codegen/src/kernel_lower.rs` — KIR additions (~50 lines; line numbers pinned at commit 1).
  - `crates/nsl-runtime/src/gptq.rs` — GPTQ runtime kernels (related, not part of M35.1).
  - `crates/nsl-codegen/src/calibration/` — AWQ + WGGO calibration infrastructure that this stacks on.
- **Institutional rules registry:** `docs/institutional-rules.md` (created in commit 1).
