# NSL-WRGA Research -- Appendix B addendum (HISTORICAL, superseded)

> **Status:** **SUPERSEDED** by `NSL-WRGA-Research-Appendix-B.md`. The
> integration-ready manuscript folds B.2 / B.3 / B.5 / B.6 from this file into a
> clean appendix and adds rules B.7 / B.8 / B.9. This file is retained as the
> discoveries-in-context historical record so future readers can trace how each
> rule emerged from its surfacing incident; treat the manuscript as
> authoritative for paper-integration purposes.

**Added:** 2026-04-18 as part of B.3.1 milestone close-out
**Target:** Appendix B (institutional-memory retrospectives), new subsection B.2

---

## B.2 -- Approximation-based activation helper testing rule

**Discovered during WRGA B.3.1 tolerance design (2026-04-18); generalizes beyond sigmoid and beyond WRGA.**

Any approximation-based activation helper (sigmoid, tanh, GELU, softmax, and related) emitted as a dedicated PTX helper MUST have at least one mid-range integration fixture whose tolerance is set such that a sign-based-stub implementation (one that returns a small set of constants based on sign/zero of the input) would FAIL the fixture.

Pure-saturation fixtures (testing only inputs that produce exact-constant outputs like 0.0, 0.5, 1.0) are necessary but not sufficient; they are unable to distinguish "the approximation is emitted correctly" from "the helper is silently stubbed."

### Concrete B.3.1 instantiation

The GatedLoRA `gate=1.0f` mid-range fixture at 1e-3 tolerance. A sign-based-stub sigmoid would return 0.5 at gate=1.0, producing y-level error of ~3.7 (>3700x the tolerance) -- fixture fails loudly.

The 1e-3 tolerance was chosen based on ULP derivation through the fold step:
- sigmoid ULP ~9e-8
- propagates to ~5e-6 per element
- main-path MMA noise floor ~2e-4 dominates
- 1e-3 provides ~5x headroom

### Generalization

The same rule applies to any future PTX helper that implements an approximation-based transcendental (exp, log, tanh, erf, GELU). Each such helper needs a mid-range fixture designed so the specific sign-based-stub attack vector for THAT function is above the chosen tolerance. For sigmoid this is gate=1.0; for tanh it would be input=0.5 (tanh(0.5) ~= 0.4621, stub returns 0); for GELU it would be input=1.0 (GELU(1.0) ~= 0.8413, stub returns 1.0 or input).

### Saturation fixtures still necessary

Saturation fixtures are still required:
- A fixture at gate=0 (sigmoid=0.5 exact) proves the helper is invoked
- Fixtures at gate=+/-inf (sigmoid=1/0 exact in fp32) prove the full numerical range routes correctly

But saturation alone is insufficient: a `match { Pos => 1.0, Neg => 0.0, Zero => 0.5 }` stub passes all three. The mid-range fixture is the only one that discriminates the curve from the piecewise constant.

---

## B.3 -- Scale-regime ptxas coverage rule

**Discovered during WRGA B.3.2 trigger measurement (2026-04-19); generalizes B.2's "coverage gap" insight from numerical to structural.**

Any PTX emitter whose spec references a specific scale regime (model dimensions, sequence length, parameter count, tile size, rank ceiling, or any target-workload parameter) MUST have ptxas-compile-validity tests at the upper end of that regime, not just at shapes convenient for small-matrix unit testing.

Pure small-shape ptxas coverage is necessary but not sufficient; it is unable to distinguish "the emitter produces valid PTX at any scale" from "the emitter happens to produce valid PTX at the shapes the unit tests exercise."

### Concrete B.3.1 failure mode

B.3.1's ptxas unit sweep covered shapes up to `(m=32, n=16, k=64, rank=8)`. The spec referenced "Llama-3-8B-style workloads" in the B.3.2 trigger condition (`seq=2048`, `rank=16`, implicit dim=4096). That gap was invisible at commit time because every B.3.1 ptxas test passed; it became visible only when the B.3.2 trigger measurement tried to exercise the prescribed workload.

At the prescribed shape `(m=1, n=4096, k=4096, rank=16)`, the fused GatedLoRA PTX was 20 MB / 478k lines (the emitter unrolled the K-loop at codegen time, producing 256 copies of the body). ptxas either ran out of memory or never returned, so the fused forward path silently fell back to CPU math. The B.3.2 trigger measurement was initially invalid as a result.

### Scale-factor diagnostic

The gap isn't "the emitter has a bug at large shapes." It's "the PTX-size-vs-shape relationship was never measured." A scale sweep at synthesis time (no ptxas, just `ptx.len()`) would have caught this instantly. The B.3.2 rewrite cycle added `ptx_scale_sweep.rs` which dumps emitted PTX size across {(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)}. Pre-rewrite output:

| n x k | PTX bytes | Lines |
|---|---|---|
| 64 x 64 | 335 KB | 8k |
| 256 x 256 | 1.3 MB | 30k |
| 1024 x 1024 | 5.0 MB | 120k |
| 2048 x 2048 | 10 MB | 239k |
| 4096 x 4096 | 20 MB | 478k |

**Linear-in-k scaling is the red flag.** Any kernel emitter whose PTX output grows with a spec-referenced dimension at rate faster than O(log n) at synthesis time is structurally vulnerable to this class of bug.

### Generalization

The rule applies to any future PTX-emitting milestone whose spec references a scale regime:

- **FlashAttention-2 variants** (CSHA, PCA, future): spec references sequence lengths up to 8k, 32k, 128k -- ptxas tests must include at least one config at seq=8k per intended upper bound.
- **CSHA / PCA** (future WRGA paper sections): spec may reference model sizes up to 70B; ptxas coverage at realistic layer dims (e.g., d_model=8192) is mandatory.
- **KV compression (M42)**: spec references cache sizes up to 512 MB; emitter tests must include at least one config exercising that upper bound.
- **Tensor parallelism (M30)**: spec references 8-way and 16-way shards; codegen ptxas tests must include both.

The rule is: **test the scale you're targeting, not the scale your unit tests happen to hit.** If the spec's target use case is the Llama-3-8B prefill path, the kernel's ptxas-validity tests must include at least one configuration at Llama-3-8B's actual projection dimensions.

### Regression gate cash-out

B.3.2's rewrite added 6 permanent ptxas regression tests in `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs` covering n in {1024, 2048, 4096} x rank in {8, 16, 32, 64} for LoRA / GatedLoRA, plus IA3 at n=k=4096. Any future regression at scale fails CI rather than silently reintroducing the CPU-fallback trap.

### Relationship to B.2 (numerical coverage)

B.2 said: "the tolerance must be tight enough that the degenerate implementation fails." B.3 extends this from numerical to structural: "the shape coverage must be wide enough that a shape-dependent degeneracy fails." Both rules apply to the same institutional failure mode - a kernel that passes all its tests for reasons unrelated to whether it actually works in production.

### Retrospective on B.3.1

B.3.1's close-out document (2026-04-18) claimed fused forward shipped end-to-end. This claim was structurally wrong. Fused forward worked at n <= 64; at Llama-scale shapes it silently CPU-fell-back. The B.3.1 close-out needs the following correction:

> **CORRECTION (2026-04-19):** Fused forward shipped for shape regime n <= 64. At spec-referenced shapes (n=4096 Llama-3-8B proxy), the emitter produced 20 MB PTX which ptxas could not compile, causing silent CPU fallback. Root cause: compile-time K-loop unrolling produced PTX with size O(k). Fixed 2026-04-19 by converting to a PTX runtime loop; post-fix PTX is ~106 KB regardless of k. See the B.3 addendum in this document.

The correction is documented in the memory file for future retrospective cycles.

---

## B.5 -- Verify the specific code path, not a proxy

**Discovered during WRGA B.3.2 Option 3 spec validation (2026-04-19); generalizes B.3's "test at the scale you target" rule from structural to procedural.**

Any spec claiming that a specific code path fires under specific conditions must include a probe that directly verifies the claim. Proxy measurements (launch-counter increments, memory profile changes, throughput deltas) can be satisfied by code paths other than the one the spec claims, and a proxy's success doesn't establish that the claimed path fires.

The probe must:

1. Exercise the specific conditions the spec's claim depends on.
2. Observe a signal that **only** the claimed path can produce.
3. Fail if the proxy is satisfied through an unclaimed path.

### Illustrative failures (both caught only after implementation began)

- **B.3.1's Llama-scale claim** was verified via ptxas tests at n <= 64. Proxy passed because ptxas accepts small PTX even when the spec's target scale is structurally broken. The direct probe -- "feed the emitter a config at the spec-referenced shape (n=4096), confirm the resulting PTX ptxas-compiles" -- would have caught the 20 MB / 478k-line failure at CI time. Without it, the failure was only discovered during B.3.2 trigger measurement.

- **B.3.2 Option 3's fused-handler claim** was verified by initial spec prep via a launch-counter increment in a top-level inference call (`let y = m.forward(x)` after the train block). Proxy passed because inference-time fusion is unrelated to the train-block path the spec claimed to fix. The direct probe -- "run `m.forward(x)` inside `train(...)` with `NSL_WRGA_GPU_LAUNCH_COUNTER=1`; assert `[nsl-gpu-launch-count] >= epochs`" -- would have shown `[nsl-gpu-launch-count] 0` and surfaced the structural gap (`model_method_bodies` populated from unrewritten AST) before any implementation work began.

### Generalization

The rule applies to any future spec where:

- An intervention targets a specific execution context (train-block path, inference-only path, sm_80+ path, source-AD walk, compile_main, etc.).
- The intervention claims to change behavior in that context.
- The verification method has any ambiguity about which context satisfied the assertion.

When in doubt, instrument the context with a signal that ONLY the claimed path can produce -- such as a param-count or adjoint-shape assertion whose value depends on the specific code path executing. Launch-counter increments are useful but insufficient if the same counter can be incremented by adjacent code paths.

### Composition with B.3

B.3 said: "test the scale you're targeting, not the scale your unit tests happen to hit." B.5 extends this from scale to code path: "verify the code path you're targeting, not a proxy satisfied by an adjacent path." Both rules target the same class of institutional failure -- specs that pass their tests for reasons unrelated to whether the intervention actually works in production.

### Application to Option 3 re-spec (2026-04-19)

Option 3's re-spec (3e + original option 3) lands with two direct probes:

- **Launch-counter assertion under train block:** `NSL_WRGA_GPU_LAUNCH_COUNTER=1` + `train(epochs=3)` + `@adapter(type=gatedlora)` asserts `[nsl-gpu-launch-count] >= 3`. Proxy-level (necessary but not sufficient by B.5).

- **Trainable-params connection-count assertion:** source-AD reports `5/5 trainable tensor params connected` (x, W, A, B, gate) rather than `1/5`. This signal can only reach 5 if source-AD's Wengert walk has seen the fused FFI call and extracted its 5 inputs -- exactly the code path 3e+option3 claims to enable. This is the B.5-compliant direct probe.

The combination catches both the proxy-pass and the proxy-fail misclassification failure modes.

### B.5 (clarification, 2026-04-19 later in day): the rule applies recursively

Any verification step whose result is cited by a spec as an established fact is itself a code-path claim subject to B.5. "Task 0 verified the precondition" is not a shield against B.5 unless Task 0's own verification satisfied B.5 -- meaning it exercised the specific conditions the precondition depends on, not proxy conditions that happen to have overlapping behavior.

Concrete failure observed: revised Option 3 spec §7 listed "No fix for adapter-field CPU placement (verified unnecessary by Task 0)" as a non-goal. Task 0 had probed gate placement via a **top-level inference call** (`let y = m.forward(x)` after the train block), not via an **in-train-block call** (`m.forward(x)` inside `train(step):`). These hit different code paths: inference mode lets the user manually reassign `m.gate_... = zeros(...).to(cuda)` between the train block and the probe; training mode has no such reassignment site because the adapter side-table is populated mid-train-block. Task 0's verification satisfied a claim about the inference path — it did not establish anything about the training path — but the spec's non-goal inherited the result as if it had. This is exactly the proxy-satisfies-the-wrong-claim failure the main B.5 rule targets, one meta-level up.

The clarification: **the rule is recursive.** A spec's "verified elsewhere" citation carries the full B.5 obligation of the original verification. A proxy verification cannot confer spec-level confidence even when the spec explicitly cites it as "precondition verified."

### Three-instances retrospective

As of 2026-04-19 this is the third occurrence of the same failure mode in two days:

1. **B.3.1's Llama-scale claim** verified via n≤64 ptxas tests.
2. **Original Option 3's fused-handler claim** verified via top-level launch-counter.
3. **Revised Option 3's precondition claim** (Task 0) verified via inference-path placement probe.

The recurrence suggests the rule is being applied in retrospect (when a blocker surfaces the violation), not proactively during spec/verification prep. The rule is correct; the *timing* of its application is the gap. Whether this warrants codifying a separate discipline ("always ask B.5's three questions during verification prep, not after") is an open question — two more instances would confirm; one occurrence doesn't yet.

---

## B.6 -- Re-validate a measurement-gated trigger against a clean baseline before scheduling the milestone

**Discovered during WRGA B.3.2 resolution (2026-05-23); composes B.3 ("test the scale you target") with a temporal corollary.**

A measurement-gated milestone trigger fires on a *number*. If that number was produced on a substrate the measurement depends on — and that substrate has known defects or has changed since — the trigger fired on an artifact, not on the phenomenon it was designed to detect. Before scheduling the milestone, **re-take the measurement on a clean substrate and re-evaluate the trigger.**

### Concrete B.3.2 instantiation

The B.3.2 trigger condition was `backward_time > 2.5x forward_time`. It "fired" at **106x** on 2026-04-19. But that reading was taken before three substrate fixes the measurement depended on:

1. **The matmul primitive** was the naive `nsl_matmul_f32` (~1-2 TFLOPs/s, ~15-30x below peak); the cuBLAS swap landed afterward.
2. **The backward path** was suspected of CPU-falling-back on some ops (the file's own follow-up #6: "50s/iter suggests CPU fallback or allocator thrash").
3. **The kernel profiler** reported zero durations (`cuEventCreate` before context init) — so the only available signal was wall-clock with overhead baked in, not per-kernel GPU time.

After all three were fixed, the clean per-op breakdown (2026-05-23) measured **backward = 0.37x forward** — the trigger does not fire. The milestone that the 106x reading "scheduled" would have optimized a 15%-of-GPU slice while the real bottleneck (the fused *forward* kernel, 73% of GPU) sat untouched. **The trigger fired on noise from a broken substrate.**

### The diagnostic

The 2026-04-21 addendum already sensed this — it qualified "schedule B.3.2" with "after per-op-profiling scoping," and noted twice that "optimizing the top-of-stack bottleneck surfaces the next layer." B.6 names the underlying rule: **a trigger's number inherits the validity of the substrate it was measured on.** When the substrate is known-defective or has changed, the number is provisional until re-measured. The "two-instances observation" in that addendum (each intervention surfacing a layer-below bottleneck) is the same phenomenon viewed from the measurement side.

### Generalization

The rule applies to any measurement-gated milestone trigger (CPDT Phase 2's 20%-disagreement gate, CSHA tier promotion thresholds, any "schedule X when metric Y crosses Z"):

- Enumerate the substrate the metric depends on (primitives, fallback paths, profiler/timer validity, shapes, device placement).
- If any substrate element is known-defective or has changed since the reading, the trigger is **provisional** — re-measure before acting.
- Bake the substrate dependencies into the trigger spec itself, so a future reader knows which fixes invalidate an old reading.

### Composition with B.3 and B.5

- B.3: test the *scale* you target.
- B.5: verify the *code path* you target, not a proxy.
- B.6: re-measure the *trigger* on a clean substrate before you act on it.

All three target the same institutional failure: **a number that passes its own test for reasons unrelated to the phenomenon the number was meant to capture.** B.3 catches it in shape coverage, B.5 in code-path attribution, B.6 in substrate validity over time.
