# M35.2a: BitNet b1.58 Backward Kernel Emission — Design

**Milestone:** M35.2a (BitNet b1.58 Phase 2, sub-milestone a — kernel-level backward)
**Stacks on:** M35.1 Phase 1 inference (PRs #154/#155/#156/#157/#159).
**Date:** 2026-05-12
**Status:** Design (Q1-Q5 + error-handling brainstormed; design-only PR until V-P1-D measured)

---

## 1. Background, M35.2 decomposition, and M35.2a scope

M35.1 shipped BitNet b1.58 inference end-to-end: KIR types, packed/unpacked conversion, CPU reference, four PTX phase emitters, `synthesize_kernel` orchestrator, parser+semantic integration for `Tensor<..., ternary>`, HF safetensors loader, and an end-to-end logit-match merge gate (`#[ignore]`'d pending two Linux follow-on items: reference logits binary + weight-scale wiring through `finalize.rs::emit`).

Phase 2 (training) was explicitly deferred per M35.1 spec §1.2 with a file-boundary discipline pinned in `crates/nsl-codegen/src/bitnet/phases/README.md`: Phase 2 PRs may only create `ternary_gemm_backward.rs`, `quantize_shadow.rs`, `absmean_quant_backward.rs`, `orchestrator_train.rs`.

### 1.1 M35.2 decomposition

Phase 2 spans five concerns (STE backward, FP32 shadow weights, activation-quant backward, optimizer integration, training loop). One spec covering all five would replicate M35.1's at-the-edge experience (16 commits, broad surface). M35.2 decomposes into three sub-milestones with explicit out-of-scope pointers:

**M35.2a — Kernel-level backward emission (THIS spec)**

- STE gradient computation as emitted PTX (forward → backward kernel composition).
- Activation-quantization backward path (gradient through absmax + round + clip).
- Backward-pass tile structure, register pressure, SMEM layout.
- New files: `bitnet/phases/ternary_gemm_backward.rs`, `bitnet/phases/absmax_quant_backward.rs`, `bitnet/orchestrator_train.rs` (training-mode wrapper).
- **Out of scope:** how autograd invokes the backward kernel (M35.2b); how shadow weights interact with parameter storage (M35.2b); optimizer-step semantics (M35.2c).

**M35.2b — Autograd integration + shadow weights**

- How NSL's autograd system invokes M35.2a's backward kernel.
- Shadow-weight tensor types and their relationship to ternary forward weights.
- The contract between FP32 shadow tensors and the ternary tensors the forward consumes.
- New file: `bitnet/phases/quantize_shadow.rs` + autograd bindings + shadow-weight types in the type system.
- **Out of scope:** kernel-level backward math (M35.2a); optimizer-step semantics (M35.2c).

**M35.2c — Optimizer integration + training loop**

- Optimizer step: FP32 shadow weight updates, requantization to ternary forward representation.
- Training-loop integration: how the end-to-end training step composes forward + backward + optimizer.
- Validation: end-to-end training of a small BitNet model, perplexity check.
- New artifacts: optimizer for ternary parameters; end-to-end training example.
- **Out of scope:** kernel-level math (M35.2a); autograd plumbing (M35.2b).

Each sub-spec validates its own scope; downstream sub-specs build on validated upstream work. The decomposition is concrete with "out of scope" pointers at every boundary, preventing scope creep.

### 1.2 Design-only posture (M35.2a spec lands now; implementation gated)

V-P1-D (Phase 1 → Phase 2 escalation criteria: ≥80% paper speedup, ≥80% memory reduction, ≤5% perplexity gap vs FP16 baselines) cannot currently be measured. M35.1's merge gate is `#[ignore]`'d on two Linux-only follow-on items (reference logits binary generation via bitnet.cpp, weight-scale wiring through `finalize.rs::emit`). bitnet.cpp doesn't build on Windows MSVC; the merge gate is unrunnable.

**M35.2a proceeds as design-only.** The spec captures all design decisions; implementation of `bitnet/phases/*backward*.rs`, `*shadow*.rs`, and `orchestrator_train.rs` waits for V-P1-D to be measured. This matches the AWQ #142 → #134 precedent: design PR landed before measurement; implementation cited the design PR by SHA; permanent design-artifact value regardless of gate outcome.

**Structural enforcement of design-only** (not just convention):

The M35.2a PR includes a file `BLOCKED_ON_V_P1_D.md` at the repo root containing the gate, measurement procedure, expected timeline (Linux access required), and unblocking criterion. A CI workflow rejects any PR that introduces non-stub content in `crates/nsl-codegen/src/bitnet/phases/*backward*.rs`, `*shadow*.rs`, or `orchestrator_train.rs` while `BLOCKED_ON_V_P1_D.md` exists at HEAD. The V-P1-D measurement PR (a separate Linux-CI PR shipping the M35.1 follow-on items + measuring §1.3 criteria) records the result by **updating** `BLOCKED_ON_V_P1_D.md` with `STATUS: PASSED` / `STATUS: FAILED` lines + supporting evidence. The file itself is **deleted (case-pass) by M35.2a's implementation PR commit 1** as the formal unblock event (see §11 verification matrix), or **deleted (case-fail) by an M35.2 deferral PR** that also retires the M35.2 spec docs.

Cost: ~30 lines of CI configuration + one file. Benefit: design-only is enforced by tooling, not memory. Discipline survives team turnover, intent drift, and "small implementation help" requests.

---

## 2. Pre-implementation verification gates

Six gates resolve load-bearing identifiers and platform questions before M35.2a implementation. Same discipline as M35.1's §10 (PI.1 trit ordering, PI.2 HF model identity, PI.3 GPTQ shipping). Some gates are verifiable now during the design phase; others gate implementation.

### 2.1 V-P1-A — Phase 2 file scaffolding documented (✅ verified during design)

**Status:** Already satisfied. `crates/nsl-codegen/src/bitnet/phases/README.md` (M35.1 Task 1) documents the four planned Phase 2 files as "additive, NOT created in Phase 1" with audit-discipline section.

### 2.2 V-P1-B — (γ) API shape preserved (✅ verified during design)

**Status:** Already satisfied. All four M35.1 forward phase emitters (`packed_load`, `quantized_ternary_gemm`, `finalize`, plus `absmean_quant` and `ternary_gemm` as `pub` + `#[doc(hidden)]` after PR #159) are publicly callable. M35.2a's `ternary_gemm_backward::emit` can compose against the existing forward emitters as building blocks.

### 2.3 V-P1-C — `tests/bitnet_reference_impl.rs` FP32 and extensible (✅ verified during design)

**Status:** Already satisfied. The four pub fns (`absmax_scale_row`, `quantize_row_int8`, `ternary_gemm_element`, `forward_reference`) are pure FP32 functions. Adding `*_backward` variants is additive — no Phase 1 refactor needed.

**Notational pin:** M35.1's spec called the file `bitnet/reference.rs`; actual location post-Task-3 is `tests/bitnet_reference_impl.rs` (for the cfg-isolation reason from M35.1 Task 3 step 3.8). M35.2a spec language uses the actual path.

### 2.4 V-P1-D — Phase 1 → Phase 2 escalation criteria (BLOCKS implementation, not design)

**Status:** Blocked on Linux follow-on. Cannot be measured until M35.1's two follow-on items land (reference_logits.bin + weight-scale wiring) and the merge gate is un-ignored on Linux CI.

**Criteria** (from M35.1 spec §1.3):

- ≥80% of paper's claimed inference speedup at the 3B configuration.
- ≥80% of paper's claimed memory reduction at the 3B configuration.
- Trained b1.58 checkpoints achieve perplexity within 5% of equivalent-parameter FP16 baselines on a held-out evaluation set.

Both implementation-quality and method-quality criteria must pass for M35.2 to proceed.

**Outcomes:**

- **Pass:** unblock M35.2a implementation; `BLOCKED_ON_V_P1_D.md` deleted.
- **Fail:** M35.2 deferred indefinitely; M35.2a design PR remains in tree as permanent artifact for future revival.
- **Partial:** revisit Phase 1 first to address the gap before M35.2 proceeds.

### 2.5 V-P1-D-prep — document stable Linux access mechanism (gates V-P1-D measurement)

**Task:** Before measuring V-P1-D, document the Linux access mechanism (dedicated machine, cloud GPU rental, dual-boot, remote contributor with stable access). Confirm it's stable enough for ongoing M35.2 development, not just a single measurement event.

V-P1-D is the *first* Linux-required step; M35.2 implementation requires Linux for backward CPU reference validation (if bitnet.cpp has backward; otherwise hand-derived per Phase 1 fallback), GPU kernel correctness tests, M35.2c training-loss equivalence, and end-to-end training validation. If Linux access is one-time/borrowed, M35.2's implementation distribution-of-responsibility differs from a stable-access scenario (e.g., remote Linux contributor doing GPU-required validation; primary developer doing Windows-side spec/parser work).

**Acceptance:** a one-paragraph note in `docs/superpowers/specs/2026-05-12-m35-2a-bitnet-backward-design.md` (this file) recording the access mechanism + cost + estimated reliability. Updated before V-P1-D measurement begins.

**Linux access mechanism (chosen 2026-05-12, per Task A.1 of the M35.2a plan):**

- **Mechanism:** WSL2 (Windows Subsystem for Linux 2) on the existing Windows 11 development machine, using CUDA passthrough to the RTX 5070 Ti (sm_120). Same physical hardware as the primary Windows MSVC environment; bitnet.cpp builds + GPU measurements run inside WSL2 via standard `cmake + nvcc` (`/usr/local/cuda` from WSL2's CUDA toolkit) rather than via separate Linux infrastructure.
- **Setup time before first V-P1-D measurement:** ~1-2 hours one-time (install Ubuntu 22.04 WSL2 distro + WSL2 CUDA toolkit + clone microsoft/BitNet + cmake build). The ~13 GB HF model download is a one-time ~30 min cost on a typical home connection and is cached at `~/.cache/nsl-tests/bitnet_b158_3b/` per spec §6.2 (the cache directory lives in WSL2's filesystem; the Linux-only bash script `scripts/fetch_bitnet_b158_3b.sh` runs natively in WSL2).
- **Ongoing reliability:** Stable for the duration of M35.2 (~3-6 months). Same machine as primary development; no rental expiry, no remote contributor coordination. Subject only to occasional Windows update events that may briefly affect WSL2 (rare; recoverable in <30 min). WSL2's CUDA support has been mature since Windows 11 21H2; the user's Windows 11 Home 10.0.26200 build supports it cleanly.
- **Estimated cost across M35.2 (a+b+c):** Effectively zero (machine + electricity already incurred). Compared to a cloud-rental alternative (~$200/week reserved A100; ~$2,400 over 12 weeks), WSL2 saves ~$2K and eliminates rental-coordination overhead.
- **WSL2-specific caveats:** WSL2 compute-bound perf is typically 95-99% of native Linux; HBM-bound workloads (BitNet is HBM-bound for ternary unpack) may show slightly larger deltas vs native, generally still within 5%. V-P1-D's ≥80% speedup criterion is measured *against WSL2's own FP16 baseline* (same harness, same WSL2 environment), not against a hypothetical native-Linux baseline — apples-to-apples within the chosen platform. If a future workload requires sub-5% match against native Linux, fall back to cloud GPU rental for that specific measurement.
- **Fallback if WSL2 becomes unavailable:** Cloud GPU rental (RunPod A100, ~$1/hour) used as one-time V-P1-D measurement venue. Plan would shift M35.2b/c ongoing development to a rented instance for the duration; cost ~$2K total. Fallback never needed unless WSL2 CUDA support breaks via an unrelated Windows event.

### 2.6 V-M35.2a-STE — confirm clipped STE matches b1.58 training reference (gates STE implementation)

**Task:** verify the b1.58 reference STE choice. Q2's clipped-STE-with-clip-threshold-`|x_int8| == 127` decision rests on "matches the de-facto b1.58 reference behavior." STE lives in training code, not inference code; bitnet.cpp may not have backward. Verification procedure:

1. Check whether Microsoft's BitNet training code is publicly available (search the b1.58 paper's references + GitHub orgs).
2. If unavailable, check HuggingFace's official BitNet training examples / notebooks.
3. If neither, survey 2-3 community reproductions for STE-variant consensus.
4. If all sources confirm clipped STE with `|x_int8| == 127` threshold: proceed with the spec as written.
5. If sources disagree OR the threshold differs (e.g., 126 for asymmetric range): document the discrepancy, pick the most-supported variant, note alternatives for M35.2a Phase 2.1 if real training shows the choice matters.

**Findings doc:** `docs/superpowers/specs/2026-05-12-m35-2-ste-baseline-findings.md`. Same one-time-anchor discipline as M35.1's PI.1 trit-ordering verification (IR-002).

**Budget:** ~30 minutes for the search + ~30 minutes for the findings doc. Cheap relative to the cost of training-instability bugs that surface during M35.2c gate verification.

### 2.7 V-M35.2a-determinism — byte-identical preflight for Tier 2 single-step descent (gates Tier 2 equivalence assertion)

**Task:** before Q5's Tier 2 single-step descent test asserts sign-matching, verify the single-step descent itself produces byte-identical parameter values across two runs with identical seed + identical input. Analog of M35.1's WGGO #134 commit-1a-style byte-identical preflight.

If the kernel is non-deterministic (e.g., `redux.sync` accidentally used instead of `shfl.sync`, or HBM atomic ordering enters via implementation back doors), the Tier 2 sign-matching assertion flakes — same `param_after - param_before` sign may differ across runs. The byte-identical preflight catches this before equivalence claims are made.

**Acceptance:** two runs of the Tier 2 fixture with identical seed produce byte-identical `param_after` values for every sampled parameter position. Asserted as a separate Tier 2.0 test that gates Tier 2.

---

## 3. Backward kernel architecture (Q1 + Q1.1)

### 3.1 Two chained backward kernels (Q1 decision)

Phase 1's forward path fuses absmax+ternary_gemm into one kernel and keeps finalize separate. M35.2a's backward follows the same kernel-count granularity with two chained kernels:

- **`ternary_gemm_backward`** — consumes `dY`, `X_int8` (saved from forward), `X_scale` (saved from forward), `W_ternary` (HBM-resident). Produces `dX_pre_STE` (FP32) and `dW` (FP32).
- **`absmax_quant_backward`** — consumes `dX_pre_STE` and `X_int8` (to detect clip saturation). Produces `dX_final` (FP32, post-STE).

### 3.2 Forward/backward fusion asymmetry (pinned honestly)

Phase 1's forward fusion was prologue-shaped: `absmax → ternary_gemm` with absmax's output feeding GEMM. Two reasons drove fusion:

- API-shape-enforced invariant (IR-001): "ternary GEMM requires absmax-quantized input" — fusing made the invariant structurally true.
- Data flow: absmax's output is GEMM's input; separation produces HBM round-trip.

Phase 2's backward fusion would be epilogue-shaped: `ternary_gemm_backward → absmax_quant_backward` (STE). Different structure; neither forward reason applies cleanly:

- API-shape invariant: "absmax_quant_backward requires dX input" is trivially true since dX is what it operates on, not really a domain invariant to enforce.
- Data flow: dX_pre_STE → STE → dX_final has the round-trip cost the two-kernel separation accepts.

"Two chained kernels mirrors Phase 1's granularity" is right at the kernel-count level (two kernels each direction). It is misleading if read as "same fusion strategy in both directions." The decision to NOT fuse backward is a real choice — the HBM round-trip cost of dX_pre_STE is accepted in exchange for testability and reference-decomposition matching.

**Phase 2.1 deferral (forward-looking):** if profiling shows the dX_pre_STE HBM round-trip is a measurable bottleneck (>5% of training-step time at production scale), epilogue-fused `ternary_gemm_backward_with_ste` becomes the M35.2a Phase 2.1 optimization. Cost: ~200 LOC of fused emission + a new public phase emitter. Trade-off: harder reference-impl debugging.

### 3.3 dW accumulator residency: SMEM tiled with separate backward tile config (Q1.1 decision)

At production scale (BitNet 3B, `hidden_dim=3200`, `d_model≈4096` in MLP), dW per linear layer is ~52 MB total. Per-CTA tile (block_k × block_n × 4 bytes FP32 accumulator) at Phase 1's defaults (block_k=block_n=128) is 64 KB — ~64% of typical Ampere/Ada/Blackwell SMEM (~100 KB/SM), leaving little for X/W tiles.

**Decision:** dW accumulates in SMEM with backward-specific tile-size fields independent from forward's. Three new fields added to `BitNetKernelConfig`:

```rust
pub struct BitNetKernelConfig {
    // ... existing forward fields ...
    pub block_m_backward: u32,
    pub block_n_backward: u32,
    pub block_k_backward: u32,
}
```

This is **the only Phase 1 file modification permitted in M35.2a** (see §9 V-P1-A exception list). Forward and backward tiles are independent (not constrained to divide). Backward tiles can be smaller than forward (e.g., 64×64 → 16 KB dW SMEM region, fits comfortably alongside X/W tiles).

**HBM atomic adds rejected:** float summation order is non-deterministic, violates §7's bit-equivalence claim.

**Separate dW kernel pass rejected:** equivalent to the three-kernels option from Q1; extra launch overhead per layer.

Matches FA-2 v2 Tier B.1's pattern of forward and backward having independent tile shapes.

### 3.4 Determinism discipline for SMEM-tiled dW accumulation

The "no HBM atomics" framing addresses cross-CTA non-determinism. Cross-warp reduction order, within-warp reduction primitive choice, and grid partitioning are additional non-determinism surfaces requiring explicit positive design choices:

1. **Cross-warp reduction order is fixed at PTX emission.** The reduction sequence (e.g., warp 0 reads, warp 1 reads, ...) is hardcoded in the emitter, not driven by scheduling. Specifically: `bar.sync 0` + sequential per-warp atomicAdd-to-SMEM, OR per-warp partial writes to disjoint SMEM regions followed by deterministic reduction. NOT runtime-arbitrated SMEM atomics.

2. **dW grid partitioning ensures CTA-disjoint dW regions.** Each CTA accumulates its dW share into a disjoint slice; no cross-CTA accumulation. Either each CTA writes to a non-overlapping output range, OR a final separate kernel reduces per-CTA dW into the layer's dW.

3. **Within-warp reductions use `shfl.sync` (defined order), NOT `redux.sync` (implementation-defined).** `shfl.sync.bfly` butterfly reduction is the canonical determinism-preserving pattern.

Determinism verification: V-M35.2a-determinism (§2.7) runs the Tier 2 fixture twice with identical seeds and asserts byte-identical parameter updates before the sign-matching equivalence test runs.

### 3.5 Backward kernel naming convention

Backward kernel PTX symbol names mirror forward:

- `nsl_bitnet_b158_ternary_gemm_backward_m{block_m_bwd}_n{block_n_bwd}_k{block_k_bwd}_{act_dtype}` — emitted by `bitnet/phases/ternary_gemm_backward.rs::emit`.
- `nsl_bitnet_b158_absmax_quant_backward_h{hidden_dim}` — emitted by `bitnet/phases/absmax_quant_backward.rs::emit`.

Greppable, per-kernel testable, dispatch-table-friendly. The `BitNetKernelConfig::kernel_name()` method extends with `backward_kernel_name()` for the backward emitter; both names participate in the same fail-fast `_ => panic!` discipline on invalid `activation_dtype` (per M35.1's Task 1 cleanup).

### 3.6 Backward kernel calling convention (interface for M35.2b autograd)

Backward kernels take an ABI-stable parameter list. M35.2b's autograd integration will dispatch through this signature; pinning it now prevents downstream re-coupling.

```c
// ternary_gemm_backward
void nsl_bitnet_b158_ternary_gemm_backward_<config>(
    const float* dY,             // [batch, out_dim] FP32 upstream gradient
    const int8_t* x_int8_saved,  // [batch, hidden_dim] saved from forward
    const float* x_scale_saved,  // [batch] FP32 per-row scale saved from forward
    const uint8_t* w_ternary,    // [hidden_dim, out_dim] packed 2-bit ternary (HBM-resident)
    int32_t batch,
    int32_t hidden_dim,
    int32_t out_dim,
    float* dX_pre_STE,           // [batch, hidden_dim] FP32 output (HBM)
    float* dW                    // [hidden_dim, out_dim] FP32 output (HBM; SMEM-accumulated per §3.3, flushed at kernel end)
);

// absmax_quant_backward
void nsl_bitnet_b158_absmax_quant_backward_<config>(
    const float* dX_pre_STE,     // [batch, hidden_dim] FP32 input
    const int8_t* x_int8_saved,  // [batch, hidden_dim] saved from forward (for clip detection)
    int32_t batch,
    int32_t hidden_dim,
    float* dX_final              // [batch, hidden_dim] FP32 output (post-STE)
);
```

This signature is the contract M35.2b builds on. M35.2b adds autograd hooks that allocate the input/output buffers and dispatch through this signature; M35.2a does not specify how the buffers are allocated or how the dispatch happens (those are M35.2b decisions).

### 3.7 Budget arithmetic scope

Backward SMEM budget formulation parallel to M35.1 spec §3.4:

- `backward_fixed_bytes()` — fixed cost of backward's SMEM allocations (X SMEM, W SMEM, dY SMEM, dX accumulator SMEM, dW tile SMEM). ~50 LOC.
- `backward_chunk_staging_bytes()` — analogous to forward's chunk staging with backward-specific dimensions. ~50 LOC.
- `chunk_config::select` extended (or new `backward_chunk_config::select` introduced) for backward selection. ~100 LOC.
- Supported-matrix CSV for `(forward_tile, backward_tile)` combinations. ~50 LOC of generation script + the CSV itself.
- Unit tests for the new formulas. ~100 LOC.

**Total: ~350 LOC for budget arithmetic alone**, not counting the kernel emitters themselves. Consistent with FA-2 v2 Tier B.1's §3 LOC contribution.

---

## 4. STE expression (Q2)

### 4.1 Clipped STE in `absmax_quant_backward`

`absmax_quant_backward::emit` produces PTX that, for each element:

1. Loads `dX_pre_STE[r, k]` (FP32) from HBM.
2. Loads `x_int8_saved[r, k]` (i8) from HBM.
3. Computes `is_saturated = (|x_int8| == 127)` — single PTX comparison.
4. Emits `dX_final[r, k] = is_saturated ? 0.0 : dX_pre_STE[r, k]` via PTX `selp.f32` or equivalent conditional move.
5. Stores `dX_final[r, k]` to HBM.

Cheap: one comparison + one conditional move per element. No FP16 activations need to be saved by forward — saturation detection uses `x_int8 == ±127` (forward's `quantize_row_int8` clamps `x * 127.0 / scale` to ±127 iff `|x| >= scale`).

### 4.2 Why not vanilla STE (calibrated rejection)

QAT literature documents two distinct cost modes from vanilla STE at low bit-widths:

- **Training divergence (α):** at 1-2 bit weights (BitNet's regime), vanilla STE causes ~15-30% of training runs to fail convergence outright, per Hubara et al. 2017 and follow-up work. Mechanism: gradients flow through saturated activations, producing wrong gradient signals that destabilize optimization.
- **Quality degradation (β):** even when training converges, vanilla STE produces 2-5% worse final perplexity than clipped STE at equivalent compute budgets.

The rejection is decisive on (α) alone: a 15-30% training-failure rate makes M35.2c's training-loss equivalence test flaky-or-failing across runs, blocking the gate's verification entirely. (β) is the additional cost even when (α) doesn't fire.

If a future configuration shows clipped STE is unnecessarily conservative (e.g., the saturation rate at production scale is so low that vanilla STE would be safe), revisit as M35.2a Phase 2.1. The default for v1 is clipped STE per the b1.58 reference's STE choice (per V-M35.2a-STE findings).

### 4.3 Why not config-driven STE (V-P1-A exception erosion)

Adding `BitNetKernelConfig::ste_variant: SteVariant { Vanilla, Clipped, Soft }` would expand the V-P1-A permitted-modification list beyond the three backward tile-size fields. The V-P1-A discipline requires structural (not preferential) justification for additional Phase 1 modifications; "research flexibility" doesn't meet the criterion. Revisit if/when production training shows the STE choice matters.

### 4.4 V-M35.2a-STE gates the STE implementation

§2.6 details. STE implementation in `absmax_quant_backward` waits for the findings doc to confirm clipped STE matches the b1.58 reference. If findings show a different convention (e.g., asymmetric clip at 126), the emitter adopts the verified convention.

---

## 5. CPU reference (Q3)

### 5.1 PyTorch autograd as canonical anchor

The backward CPU reference uses PyTorch's autograd through a custom `torch.autograd.Function`. PyTorch is cross-platform (no Linux blocker), and PyTorch autograd is the de-facto industry-standard ground truth for STE backward — every QAT library uses it as the reference. IR-002 anchor cleanly applies: capture PyTorch outputs once, commit as JSON fixtures, NSL reference + future NSL kernel implementations validate against the fixtures.

**Cross-platform benefits beyond "no Linux blocker":**

- Reference correctness verification is faster — implementer can run the reference, modify it, inspect outputs in their primary environment. No "wait for the Linux runner" friction. Reduces V-M35.2a-STE verification's wall-clock time from days to hours.
- Fixture regeneration is local — future updates to NSL kernel behavior regenerate fixtures on the implementer's machine.
- Reference is reviewable in PR diff — reference source (`tests/fixtures/bitnet_backward_reference.py`) is checked in alongside the fixture JSON; reviewers verify reference logic directly, not just fixture values.

The Linux dependency for M35.2 implementation continues to apply (V-P1-D, kernel correctness tests, M35.2c training-loss equivalence) but the reference itself does not gate on Linux access.

### 5.2 Pinned reference implementation shape (idiom α)

The reference uses `torch.autograd.Function` with explicit `forward` and `backward` methods (idiom α). This matches the data flow of NSL's custom kernel:

```python
class BitLinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_ternary, x_scale):
        # absmax quantize, ternary GEMM, dequant
        x_int8 = (x * (127.0 / x_scale)).clamp(-127, 127).round().to(torch.int8)
        ctx.save_for_backward(x_int8, x_scale, w_ternary)
        return x_int8.float() @ w_ternary.float() * (x_scale / 127.0)

    @staticmethod
    def backward(ctx, grad_output):
        x_int8, x_scale, w_ternary = ctx.saved_tensors
        # Clipped STE: zero gradient where |x_int8| == 127
        clip_mask = (x_int8.abs() != 127).float()
        grad_x_pre_ste = (grad_output @ w_ternary.t().float()) * (x_scale / 127.0)
        grad_x = grad_x_pre_ste * clip_mask
        # dW computation
        grad_w = x_int8.t().float() @ grad_output
        return grad_x, grad_w, None
```

Idiom α is chosen because: (i) it matches the data flow of NSL's kernel (explicit gradient computation, explicit clip mask), making fixture comparison meaningful; (ii) it's the most-used pattern in BitNet community reproductions (per V-M35.2a-STE's findings); (iii) the `forward`/`backward` methods are inspection-friendly, supporting reviewer verification of the reference's correctness.

Idioms β (detach trick) and γ (hooks) produce equivalent gradient values mathematically but obscure the relationship between reference and kernel. Avoided. If V-M35.2a-STE findings show the community uses a different idiom predominantly, switch to that.

### 5.3 Backward fixture set (10 fixtures)

5 inherited from M35.1 (backward-relevant) + 5 new backward-specific:

| Fixture | Inherited from M35.1 | Tests |
|---|---|---|
| `bf01_uniform` | Yes (analog of f02) | Basic backward through uniform-magnitude row |
| `bf02_sign_mixed` | Yes (analog of f04) | Mixed-sign gradient flow |
| `bf03_outlier` | Yes (analog of f05) | Outlier dominates absmax; backward through compressed non-outliers |
| `bf04_multi_row` | Yes (analog of f06) | Per-row scales in backward |
| `bf05_zero_weight_col` | Yes (analog of f07) | Backward through zero-output columns (dW non-zero, dX zero) |
| `bf06_saturated_input` | NEW | All `|x_int8| == 127`; expected `dX_final = 0` everywhere |
| `bf07_partial_saturation` | NEW | Mix of 127s and non-127s; exercises clip-mask boundary |
| `bf08_zero_grad_output` | NEW | `grad_output = 0`; expected `dX_final = 0`, `dW = 0` |
| `bf09_identity_grad_output` | NEW | `grad_output = 1` everywhere; tests un-saturated identity pathway |
| `bf10_random_seeded` | NEW | Fixed-seed random `grad_output` + inputs; deterministic cross-platform |

Each fixture records: input `x`, `w_ternary`, `x_scale`, `grad_output` → expected `dX_final` (post-STE), `dW`. JSON format compatible with M35.1's `bitnet_reference_outputs.json` structure.

Saturation-related fixtures (bf06, bf07) stress-test the clip mask directly; if the kernel emits the wrong clip mask, these fail.

**Fixture file:** `crates/nsl-codegen/tests/fixtures/bitnet_backward_reference_outputs.json`. **Reference source:** `crates/nsl-codegen/tests/fixtures/bitnet_backward_reference.py` (cross-platform; commit alongside fixtures).

---

## 6. Forward-saved state contract (Q4)

### 6.1 Training-mode wrapper kernel saves; forward kernel untouched

`orchestrator_train.rs` emits a training-mode wrapper that:

1. Calls the M35.1 forward kernel (`synthesize_kernel(config)`).
2. Appends a separate save-to-HBM kernel writing `x_int8` (saved by absmax_quant phase) + `x_scale` to per-layer buffers.

**Zero Phase 1 file modifications beyond Q1.1's three tile-size additions.** Forward kernel stays inference-only as designed.

### 6.2 Stream-parallelism discipline (load-bearing for performance)

Per-layer extra kernel launch overhead is bounded by stream parallelism: save kernels dispatch on a **separate stream** from the main forward path, so they don't serialize forward progress. This is the implementation discipline the wrapper MUST follow.

Cost components:

- **Kernel computation:** genuinely cheap — memory-bound copy of `x_int8` (1 byte/element) and `x_scale` (FP32, 1 value per row). For BitNet 3B at typical seq_len, ~MBs of data per layer.
- **Launch overhead:** ~5-10 microseconds per launch, fixed. Across 30+ transformer blocks, ~150-300 microseconds per training step's forward pass.

If the implementation submits save kernels on the main stream, launch overhead becomes serializing and adds ~5% to training-step time. The wrapper's *correctness* doesn't depend on stream choice (saved values are correct either way), but *performance* does. Spec pins separate-stream dispatch as a requirement, not a recommendation.

### 6.3 Saved-state buffer layout

Per-layer save buffers:

- `x_int8_saved[layer_idx]`: pointer to `[batch * seq_len * hidden_dim]` bytes in HBM. Allocated by training-mode wrapper at first forward; freed after backward consumes it.
- `x_scale_saved[layer_idx]`: pointer to `[batch * seq_len]` FP32 values in HBM. Same lifecycle.

For BitNet 3B (`hidden_dim=3200`, `num_layers=26`, `seq_len=2048`, `batch=1`): ~170 MB for x_int8 + ~210 KB for scales per training step. Manageable on consumer GPU; sub-1% of typical activation memory budget.

### 6.4 Why not extend `BitNetKernelConfig` with `save_for_backward: bool` (option 2 rejected)

Would add a fourth Phase 1 modification beyond Q1.1's three, expanding V-P1-A's permitted-modification list. The V-P1-A meta-rule keeps additions narrowly bounded; "saves one kernel launch per layer" is a preferential benefit, not a structural justification.

### 6.5 Why not recompute `x_int8` in backward (option 3 rejected — architectural)

At BitNet 3B scale, recomputing absmax+quant is ~8K FLOPs per row vs. ~4 KB HBM read per row — recomputation is faster in absolute terms (compute is cheaper than HBM at this scale). The actual load-bearing rejection is **architectural**:

- Phase 2's autograd integration becomes responsible for ensuring FP16 activations are available at backward time (couples M35.2b's autograd work to backward-pass design).
- The recomputation introduces a new code path (absmax+quant in backward) that must match forward's absmax+quant exactly — any drift produces incorrect gradients silently.
- Phase 1's forward path becomes implicitly dependent on autograd's activation-saving discipline, which contradicts Phase 1's "inference-only forward" architectural commitment.

Saving x_int8 (option 1) avoids all three concerns at the cost of one extra kernel launch per layer. The compute-vs-bandwidth tradeoff favors recomputation; the architectural cleanliness favors saving.

---

## 7. Validation tiers (Q5)

### 7.1 Two-tier kernel-correctness gate

**Tier 1: Fixture gradient bit-equivalence.** NSL backward asserts FP32-ULP match (1e-4 relative for dX/dW; bit-exact for clip mask + scales) against PyTorch reference JSON (§5.3 fixture set). Tolerance per CSHA Tier A's `O(sqrt(d)·ε_f16)` pattern, accounting for chain-rule accumulation across hidden_dim multiplications.

Bit-exact requirements:

- Clip mask (the `is_saturated` boolean field): bit-exact match against reference.
- `x_scale` (FP32): bit-exact (already validated in forward).

Relative-tolerance requirements:

- `dX_final` (FP32): `rel_diff <= 1e-4 || abs_diff < 1e-7`.
- `dW` (FP32): `rel_diff <= 1e-4 || abs_diff < 1e-7`.

**Tier 2: Single-step gradient-descent test.** Apply computed gradients with learning rate to a toy BitLinear; assert parameter values move in *expected direction* (sign-matching against hand-derivation, NOT against PyTorch output).

### 7.2 Tier 2 fixture (specific)

Three requirements for sign-matching to be load-bearing:

1. **Hand-derivable expected direction:** loss function's gradient sign for each parameter is computable without running the kernel. Simplest version: `loss = MSE(output, target)` with `target = constant vector`.
2. **Initial parameters bias output predictably away from target:** initial weight matrix + absmax scale chosen so `output[0] < target[0]` (or vice versa). Makes gradient direction non-ambiguous.
3. **Small learning rate (e.g., 1e-3):** update small enough that direction-of-movement is the load-bearing signal, not magnitude saturation.

Per-parameter assertion: `sign(param_after - param_before) == sign(expected_direction)`.

```python
# Fixture setup (committed at tests/fixtures/bitnet_backward_single_step/)
import torch
hidden_dim = 128
batch_size = 4
output_dim = 64

torch.manual_seed(42)
x = torch.randn(batch_size, hidden_dim) * 0.5  # small magnitudes
w_ternary_init = ternary_init(hidden_dim, output_dim, seed=42)
x_scale_init = (x.abs().max(dim=-1).values).clamp(min=1e-5)

target = torch.ones(batch_size, output_dim)  # all-ones target

# Sanity check: initial output is biased AWAY from target
assert output_initial.mean() < target.mean(), "fixture setup invariant"

# Expected gradient direction (hand-derived per linear layer):
# dL/dw[k, c] = -(target[*, c] - output[*, c]) @ x[*, k]
# Sample positions where |dL/dw| > sign_threshold so direction is unambiguous
sample_positions = pick_top_k_by_magnitude(expected_grad, k=output_dim // 4)

# NSL kernel computes gradient; assert sign-matching per-sampled-position
for (k, c) in sample_positions:
    assert sign(param_after[k, c] - param_before[k, c]) == sign(expected_direction[k, c]), \
        f"sign mismatch at ({k}, {c}): {param_after[k, c] - param_before[k, c]} vs expected {expected_direction[k, c]}"
```

Sign-matching compares against hand-derived expected directions, NOT against PyTorch's autograd output. PyTorch comparison alone IS bit-equivalence (Tier 1) applied at post-step level; it would NOT catch sign-flip bugs that affect both kernel and reference.

### 7.3 Validation tier boundaries

| Bug class | Caught by |
|---|---|
| Magnitude error in dX or dW | Tier 1 (fixture bit-equivalence) |
| Sign-flip in gradient | Tier 2 (single-step descent) |
| Transposed-matrix bug | Tier 2 (single-step descent) |
| Clip-mask zero/one inversion | Tier 1 (clip mask bit-exact comparison) |
| Cumulative gradient drift over many steps | M35.2c (training-loss equivalence) |
| Wrong learning-rate-scale interaction | M35.2c (training-loss equivalence) |
| Optimizer-momentum state drift | M35.2c (training-loss equivalence) |

M35.2a's tiers complete before M35.2b lands; M35.2c's test runs after M35.2b's autograd integration is in place. If M35.2a passes both tiers but M35.2c fails, the failure source is in M35.2b/M35.2c integration, not in M35.2a's kernel.

### 7.4 Tier 2.0 byte-identical preflight (V-M35.2a-determinism)

Before Tier 2's sign-matching assertion ships, V-M35.2a-determinism (§2.7) gates: two runs of the Tier 2 fixture with identical seed produce byte-identical `param_after` values. Non-determinism would cause Tier 2 to flake (same parameter could move in opposite directions across runs). Tier 2.0 catches the determinism violation before it manifests as flaky Tier 2 failures.

### 7.5 Discipline: "Tier 1 + Tier 2 pass but kernel suspected buggy" guidance

If the M35.2a implementer suspects a bug that Tier 1 + Tier 2 don't catch, the response is NOT to add a lightweight multi-step training test (which would pull M35.2c scope into M35.2a, violating decomposition boundary). Instead:

1. **Identify the specific bug class suspected.** "Vague unease about gradient correctness" isn't actionable; "I think the clip mask might be inverted for negative-zero edge cases" is.
2. **Add a targeted Tier 1 fixture covering that bug class.** If clip-mask-on-negative-zero is the concern, add a fixture with negative-zero in `x_int8` and assert the clip mask handles it.
3. **If the suspicion is about cumulative behavior** (gradient drift over many steps), defer to M35.2c's training-loss equivalence test. Don't replicate it in M35.2a.

Each tier has a specific bug-class scope; expanding a tier should expand the bug-class coverage, not duplicate validation that downstream tiers handle.

---

## 8. Error handling and unsupported-config behavior

### 8.1 Planner-level dispatch for unsupported backward configs

`backward_chunk_config::select(config)` returns either `Ok(BackwardChunkConfig)` or `Err(DowngradeReason)`. The `Err` variants enumerate specific reasons:

- `Err::DwAccumulatorTooLarge { required_bytes, available_bytes }` — dW SMEM accumulator doesn't fit at any chunk size.
- `Err::RegisterPressureExceeded { estimated_regs, cap }` — backward kernel's static register analysis exceeds 255/thread.
- `Err::UnsupportedDtype { dtype }` — backward kernel doesn't support this dtype (e.g., bf16 with no bf16 backward path).

Planner consumes the `Err` and dispatches the affected layer to Tier C scalar backward (a generic fallback path that doesn't depend on M35.2a-specific tiling). The planner's dispatch decision is recorded in the supported-matrix CSV (`crates/nsl-codegen/tests/fixtures/bitnet_backward_supported_matrix.csv`) so users can see what configs admit which backward path.

### 8.2 Saved-state existence assertion (training-mode invariant)

Backward kernel asserts the saved-state buffers (`x_int8_saved`, `x_scale_saved`) are non-null at entry. The wrapper invariant guarantees they're populated when `training_mode=true`; if backward is invoked outside training mode, the kernel panics with a clear message: `"backward invoked on inference-only forward state; missing x_int8_saved buffer"`. NOT a silent zero-gradient failure.

### 8.3 Saved-state shape verification

The wrapper stores not just the buffers but also their shape (`hidden_dim`, `batch_size`, `num_rows`). Backward kernel asserts the stored shape matches the layer's expected shape at backward entry. Mismatch panics with: `"saved state shape mismatch: forward saved {stored_shape}; backward expected {expected_shape}"`. Pre-empts the "read past saved buffer end" silent corruption mode.

### 8.4 Assertion placement

All three checks fire at backward-kernel-entry (host-side, before PTX dispatch), NOT at PTX-emission time. Cost: ~30 LOC of assertion code; assertions only fire on programming errors (mismatched invocation, invariant violation), not on normal training paths.

---

## 9. Institutional discipline (IR-001, IR-002, IR-003)

### 9.1 V-P1-A exception list (running)

Permitted Phase 1 file modifications during M35.2a:

1. `BitNetKernelConfig::block_m_backward`, `block_n_backward`, `block_k_backward` — three additive fields justified by Q1.1's SMEM-tiled-with-backward-config decision.
2. `docs/institutional-rules.md` — append IR-003 (registry is append-only by design; same shape as IR-001 + IR-002 codified in M35.1 Task 1).

Adding any further Phase 1 modifications requires explicit brainstorm-level decision with rationale. Convenience or research-flexibility motivations are insufficient; the modification must be either (a) load-bearing for a settled M35.2a decision, or (b) a regression fix to Phase 1 (which would be a Phase 1 PR, not an M35.2a PR).

### 9.2 IR-003: separate design-proceeding work from measurement-dependent implementation

**Rule.** Work that can proceed on the platform/state available now should be separated from work that depends on unmeasured criteria. Design proceeds on the platform-independent surface; measurement-dependent implementation waits for the gate.

**Pattern.** Identify the load-bearing measurement (V-P1-D, perplexity gate, hardware-behavior probe). Identify what can be designed without it (architectural decisions, API shape, fixture sets, reference implementations). Ship the design as a permanent artifact; gate the implementation on the measurement.

**Origin.** Surfaced repeatedly across NSL milestones: AWQ design PR #142 before #134 measurement, CSHA Tier B.1 V1/V2/V3 pre-implementation verification, PCA Tier B SMEM probe, M35.2a design-only with V-P1-D gate. Codified in this spec.

**Examples.**

- AWQ #142 (design) landed before #134 (implementation) measured calibration-PPL outcomes. Design PR was self-contained; implementation cited design PR by SHA; design remained a permanent artifact regardless of measurement outcome.
- M35.2a (this spec) ships design-only with `BLOCKED_ON_V_P1_D.md` + CI enforcement. Implementation waits for V-P1-D pass.
- CSHA Tier B.1 V1/V2/V3 verified load-bearing assumptions before kernel implementation began. Caught three substantive spec drifts pre-implementation.

**Anti-pattern this prevents.** Implementation work accumulates while waiting for measurement, then re-litigates design decisions when measurement results arrive. The institutional cost is wasted implementation effort if measurement fails; the institutional value is design-as-permanent-artifact if measurement succeeds OR fails.

**Discipline pattern: structural enforcement, not convention.** IR-003 isn't satisfied by intent — it requires CI checks, BLOCKED_ON files, or other tooling that prevents implementation-creep into design-only PRs. Convention-only enforcement degrades with team turnover.

### 9.3 Design-only CI enforcement (structural)

CI workflow `.github/workflows/design_only_enforcement.yml`:

```yaml
name: M35.2 design-only enforcement
on:
  pull_request:
    paths:
      - 'crates/nsl-codegen/src/bitnet/phases/**'
      - 'crates/nsl-codegen/src/bitnet/orchestrator_train.rs'
      - 'BLOCKED_ON_V_P1_D.md'

jobs:
  reject-impl-while-blocked:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check BLOCKED_ON_V_P1_D.md exists
        id: blocked
        run: |
          if [[ -f BLOCKED_ON_V_P1_D.md ]]; then
            echo "blocked=true" >> "$GITHUB_OUTPUT"
          else
            echo "blocked=false" >> "$GITHUB_OUTPUT"
          fi
      - name: Reject non-stub backward emitter content
        if: steps.blocked.outputs.blocked == 'true'
        run: |
          for f in crates/nsl-codegen/src/bitnet/phases/*backward*.rs \
                   crates/nsl-codegen/src/bitnet/phases/*shadow*.rs \
                   crates/nsl-codegen/src/bitnet/orchestrator_train.rs; do
            if [[ -f "$f" ]]; then
              lines=$(grep -v '^\s*//' "$f" | grep -v '^\s*$' | wc -l)
              if [[ "$lines" -gt 5 ]]; then
                echo "FAIL: $f has $lines non-comment non-blank lines while BLOCKED_ON_V_P1_D.md exists"
                exit 1
              fi
            fi
          done
          echo "PASS: backward emitter files are stub-only while V-P1-D is unmeasured"
```

The workflow runs on every PR touching the relevant paths. While `BLOCKED_ON_V_P1_D.md` exists at repo root, any PR introducing >5 non-comment non-blank lines in the listed Phase 2 backward files fails CI. The threshold of 5 lines accommodates module docstrings + `//! TODO(M35.2a impl gated on V-P1-D)` stub markers.

`BLOCKED_ON_V_P1_D.md` is removed in the V-P1-D-pass measurement PR (which unblocks implementation) or the M35.2-deferral PR (which retires the design).

---

## 10. Out of scope (M35.2b, M35.2c, Phase 2.1 deferrals)

### 10.1 M35.2b (autograd + shadow weights)

- How NSL's source-AD `WengertList` lowering encodes the backward call dispatch.
- FP32 shadow weight tensor type in `nsl-semantic` (parallel to Type::TernaryPacked + Type::TernaryUnpacked).
- The `quantize_to_ternary(FP32) -> TernaryPacked` op semantics + lowering.
- Activation-memory lifecycle management at the autograd level (the wrapper saves; autograd consumes; this spec doesn't pin autograd's consumption mechanism).

### 10.2 M35.2c (optimizer + training loop)

- AdamW-style optimizer state for ternary parameters (m, v moments are FP32 even though parameters are ternary).
- Optimizer step: FP32 shadow weight updates, requantization to ternary forward representation.
- Training-loop integration: end-to-end forward + backward + optimizer composition.
- Training-loss equivalence test (Layer 3 analog of FA-2 v2 Tier B.1's gradient bit-equivalence).
- End-to-end training of a small BitNet model from a checkpoint with perplexity gate (analog of M35.1's logit-match merge gate).

### 10.3 M35.2a Phase 2.1 deferrals (forward-looking)

- **Epilogue-fused `ternary_gemm_backward_with_ste`** — if dX_pre_STE HBM round-trip is measurable bottleneck (>5% of training-step time at production scale). Cost: ~200 LOC + new public emitter.
- **Config-driven STE variant** — if production training shows the STE choice matters and clipped STE is unnecessarily conservative for specific configurations. Cost: V-P1-A exception expansion + emitter switch logic.
- **Bf16 backward path** — if bf16 forward becomes common in BitNet workloads (M35.1's forward supports bf16; backward currently does too if `BitNetKernelConfig::activation_dtype == Bf16`, but a separate Phase 2.1 may optimize bf16-specific backward).

### 10.4 Out of M35.2 entirely

- Binary BitNet (original 2023 paper, {-1, +1}) — M35 targets b1.58 (ternary). Original BitNet is a separate future milestone if a workload demands it.
- CSHA-fused BitNet backward — Phase 1's (γ) API shape preserves the option (backward emitters can compose into CSHA's hook system); actual fusion lands as a follow-on once M35.2a-c stack is stable.

---

## 11. Verification matrix (per-commit, parallel to M35.1 §9)

M35.2a's implementation commits gate on the verification table below. Implementation order matches the commit table; pre-implementation verifications complete before commit 1.

| Commit | Subject | V-gates needed | Tier 1 | Tier 2 | Tier 2.0 (determinism) |
|---|---|---|---|---|---|
| pre-1 | V-P1-A/B/C/D-prep + V-M35.2a-STE findings doc | A/B/C ✅, D-prep, STE | — | — | — |
| pre-2 | V-P1-D measurement (Linux follow-on) | D | — | — | — |
| 1 | `BLOCKED_ON_V_P1_D.md` deletion + `.github/workflows/design_only_enforcement.yml` retirement | D ✅ | — | — | — |
| 2 | `backward_chunk_config::select` + supported-matrix CSV | — | — | — | — |
| 3 | `bitnet/phases/ternary_gemm_backward.rs` (dX path, dW SMEM-tiled) | — | ✓ vs PyTorch ref | — | — |
| 4 | `bitnet/phases/absmax_quant_backward.rs` (clipped STE) | STE ✅ | ✓ vs PyTorch ref | — | — |
| 5 | `bitnet/orchestrator_train.rs` (training-mode wrapper + save kernels on parallel stream) | — | ✓ end-to-end | — | — |
| 6 | Tier 2 single-step descent test + V-M35.2a-determinism preflight | determinism | ✓ | ✓ | ✓ |
| 7 | Backward fixture JSON committed (10 fixtures from §5.3) | — | ✓ all 10 | — | — |
| 8 | Phase 2.1 deferral notes in spec + supported-matrix CSV documentation | — | — | — | — |

**Commit 1 is the unblock commit** — its only purpose is to delete `BLOCKED_ON_V_P1_D.md` and the CI enforcement workflow after V-P1-D passes. Commits 2-8 are the actual implementation work; the CI enforcement ensures none of them can land while V-P1-D is unmeasured.

---

## 12. Estimated scope

- **Pre-implementation verification:** ~3-4 days total.
  - V-P1-A/B/C ✅ (already verified during design).
  - V-P1-D-prep ~1 day (document Linux access mechanism).
  - V-M35.2a-STE ~1 hour (search + findings doc).
  - V-M35.2a-determinism ~half day (byte-identical preflight implementation).
  - V-P1-D measurement: Linux-dependent; timing varies (1-4 weeks depending on Linux access stability).

- **Implementation (when V-P1-D passes):** ~3-4 weeks for M35.2a alone (8 commits per §11 verification matrix). Parallel to M35.1's 10-task structure.

- **M35.2 total (M35.2a + M35.2b + M35.2c):** ~10-14 weeks from V-P1-D pass.

- **Linux dependency throughout implementation:** kernel correctness tests, M35.2c training-loss equivalence, end-to-end training validation. Not just at V-P1-D gate.

If V-P1-D fails, M35.2a's spec remains as a permanent artifact (per IR-003); no implementation cost is sunk.

---

## Appendix A: References

- **Issue:** to be filed when V-P1-D measurement is scheduled.
- **Related specs:**
  - `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` — M35.1 Phase 1 design (PR #154, merged).
  - `docs/superpowers/specs/2026-05-12-m35-2-ste-baseline-findings.md` — V-M35.2a-STE findings (to be written during pre-implementation phase).
- **Related PRs:**
  - PR #156 (M35.1 implementation, merged) — establishes Phase 1 baseline.
  - PR #157 (M35.1 CI cleanup, merged) — post-merge clippy + workflow fixes.
  - PR #159 (M35.1 visibility cleanup, in flight) — `#[doc(hidden)]` on internal emitters + KirType naming rationale.
- **Research papers (NotebookLM notebook #8):**
  - Ma et al. 2024 — *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*.
  - Hubara et al. 2017 — *Quantized Neural Networks* (vanilla STE training divergence rates at low bit-widths).
  - Wang et al. 2023 — *BitNet: Scaling 1-bit Transformers for Large Language Models* (original binary BitNet).
- **Reference implementations:**
  - `1bitLLM/bitnet_b1_58-3B/utils_quant.py` — community port, PI.2-verified canonical b1.58 implementation.
  - `microsoft/BitNet` (bitnet.cpp) — inference reference (no backward).
- **Code references (post-M35.2a expected paths):**
  - `crates/nsl-codegen/src/bitnet/phases/ternary_gemm_backward.rs` — new in M35.2a.
  - `crates/nsl-codegen/src/bitnet/phases/absmax_quant_backward.rs` — new in M35.2a.
  - `crates/nsl-codegen/src/bitnet/orchestrator_train.rs` — new in M35.2a.
  - `crates/nsl-codegen/src/bitnet/config.rs` — extended with three backward tile-size fields (V-P1-A exception).
  - `docs/institutional-rules.md` — extended with IR-003 (registry append, V-P1-A exception by registry-design).
  - `tests/fixtures/bitnet_backward_reference.py` — PyTorch reference (idiom α).
  - `tests/fixtures/bitnet_backward_reference_outputs.json` — 10 backward fixtures.
  - `tests/fixtures/bitnet_backward_single_step/` — Tier 2 fixture data.
  - `crates/nsl-codegen/tests/fixtures/bitnet_backward_supported_matrix.csv` — `(forward_tile, backward_tile)` combinations supported by M35.2a.
- **Institutional rules registry:** `docs/institutional-rules.md` (IR-001 + IR-002 from M35.1; IR-003 added in M35.2a).
