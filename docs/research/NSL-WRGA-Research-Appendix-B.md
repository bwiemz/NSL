# Appendix B — Institutional Rules

## Preface

Appendix B collects **institutional rules** — discipline that emerged from
episodes observed during WRGA's design and evaluation, and that
generalises beyond the immediate incident. Every rule in this appendix
targets one failure class:

> **A test, measurement, or verification that passes for reasons
> unrelated to whether the intervention actually works in production.**

We call this the *Anti-Goodhart* failure class, after Goodhart's Law:
once a proxy becomes the success criterion, it stops being a faithful
signal. The rules below catalogue distinct mechanisms by which the
WRGA compiler's checks could (and did) silently degenerate into
Goodhartised proxies.

Each rule is presented with:

- a **discovery context** — the episode that surfaced it;
- a **rule statement** — the generalised discipline;
- a **probe construction** — what concretely to add so the rule is
  enforceable rather than aspirational; and
- a **composition with sibling rules** — how this rule complements the
  others; readers landing on one rule should see where its blind spots
  are covered.

### Promotion criterion

A mechanism is promoted to an institutional rule after **three
independent observations**, or after a **single observation whose
generalisation across the system is structurally evident** (i.e., the
mechanism is rooted in a structural property of the compiler that
recurs by construction across the codebase). The two-clause
criterion lets a rule stand on three corroborating incidents *or* on
one incident plus a structural argument for generalisability. Where
the present appendix relies on the second clause, the discovery
context flags the single-instance basis explicitly.

### Milestone labels

Several rules cite WRGA development milestones by short labels —
B.3.1 (the original fused GatedLoRA forward path described in §6.X
of the main paper), B.3.2 (the fused backward kernel work explored
in §6.Y), and so on. These labels refer to artefacts described in
the body of the paper, not to project-management cycles; the
appendix's own rule numbering (B.2, B.3, B.5, …) is separate from
the milestone labels and is referenced as "Appendix B section B.N"
or simply "rule B.N" where the distinction matters.

### Numbering

Rule B.1 was published in an earlier revision of this paper (the
original WRGA experimental-setup retrospective) and is not reproduced
in the present manuscript; the appendix below collects rules B.2
through B.9. B.4 is intentionally reserved as a project-milestone
label (see §B.4 below) and carries no institutional rule; this is
the only gap in the numbering.

---

## B.2 — Approximation-based activation helper testing rule

**Discovered during** the tolerance design for WRGA's original fused
GatedLoRA forward path (milestone B.3.1); generalises beyond
sigmoid and beyond WRGA.

### Rule

Any approximation-based activation helper (sigmoid, tanh, GELU,
softmax, exp, log, erf, and related) emitted as a dedicated PTX
helper MUST have at least one **mid-range** integration fixture
whose tolerance is set such that a sign-based-stub implementation
— one that returns a small set of constants based on the sign or
zero of the input — would FAIL the fixture.

Pure-saturation fixtures (testing only inputs that produce
exact-constant outputs such as 0.0, 0.5, 1.0) are necessary but not
sufficient: they cannot distinguish *the approximation is emitted
correctly* from *the helper is silently stubbed*.

### Concrete instantiation

In the GatedLoRA forward path the discriminating fixture sets the
gate input to 1.0 with a tolerance of 1e-3. A sign-based stub for
sigmoid would return 0.5 at gate = 1.0, producing a y-level error
of approximately 3.7 — about 3700× the tolerance — so the fixture
fails loudly.

The 1e-3 tolerance is derived via a ULP analysis through the fold
step:

- sigmoid ULP ≈ 9e-8,
- propagates to ≈ 5e-6 per element,
- main-path MMA noise floor ≈ 2e-4 dominates,
- 1e-3 gives ≈ 5× headroom.

### Generalisation

For each future PTX helper that implements an approximation-based
transcendental, the mid-range fixture must be designed so the
sign-based-stub attack vector specific to *that function* is above
the chosen tolerance:

| Function | Mid-range probe | Stub answer | Real answer |
|----------|----------------|-------------|-------------|
| sigmoid  | gate = 1.0     | 0.5         | ≈ 0.731     |
| tanh     | x = 0.5        | 0           | ≈ 0.462     |
| GELU     | x = 1.0        | 1.0 (or x)  | ≈ 0.841     |

### Saturation fixtures remain necessary

Saturation fixtures are still required:

- a fixture at gate = 0 (sigmoid = 0.5 exact) proves the helper is
  invoked rather than skipped;
- fixtures at gate = ±∞ (sigmoid = 1 / 0 exact in fp32) prove the
  full numerical range routes correctly.

But saturation alone is insufficient: a
`match { Pos => 1.0, Neg => 0.0, Zero => 0.5 }` stub passes all
three. The mid-range fixture is the only one that discriminates
the curve from the piecewise constant.

### Composition

B.2 is the *numerical* instance of the Anti-Goodhart class. B.3
extends it from numerical to *structural* coverage; B.5 generalises
further from coverage to code-path attribution. B.7 (rule B.7
below) is the *test-design* instance of both, applied to the test's
own premise — a test that cannot fail under its premise is the same
failure mode one meta-level up.

---

## B.3 — Scale-regime ptxas coverage rule

**Discovered during** the trigger measurement for the WRGA fused
backward kernel milestone (B.3.2); generalises B.2's coverage gap
from numerical to structural.

### Rule

Any PTX emitter whose spec references a specific scale regime —
model dimensions, sequence length, parameter count, tile size,
rank ceiling, or any target-workload parameter — MUST have
ptxas-compile-validity tests at the upper end of that regime, not
just at shapes convenient for small-matrix unit testing.

Pure small-shape ptxas coverage is necessary but not sufficient:
it cannot distinguish *the emitter produces valid PTX at any scale*
from *the emitter happens to produce valid PTX at the shapes the
unit tests exercise*.

### Concrete failure mode

The B.3.1 ptxas unit sweep covered shapes up to
`(m = 32, n = 16, k = 64, rank = 8)`. The spec referenced
Llama-3-8B-style workloads in the B.3.2 trigger condition
(`seq = 2048`, `rank = 16`, implicit `dim = 4096`). The gap was
invisible at commit time because every B.3.1 ptxas test passed; it
became visible only when the B.3.2 trigger measurement tried to
exercise the prescribed workload.

At the prescribed shape `(m = 1, n = 4096, k = 4096, rank = 16)`,
the fused GatedLoRA PTX was **20 MB / 478 000 lines** — the emitter
unrolled the K-loop at codegen time, producing 256 copies of the
body. ptxas either ran out of memory or never returned, so the
fused forward path silently **fell back to CPU math**. The B.3.2
trigger measurement was initially invalid as a result.

### Scale-factor diagnostic

The gap is not *the emitter has a bug at large shapes*; it is *the
PTX-size-vs-shape relationship was never measured*. A scale sweep
at synthesis time — no ptxas, simply observing the emitted PTX
length — catches this instantly. Pre-rewrite output:

| n × k       | PTX bytes | Lines |
|-------------|-----------|-------|
| 64 × 64     | 335 KB    | 8 k   |
| 256 × 256   | 1.3 MB    | 30 k  |
| 1024 × 1024 | 5.0 MB    | 120 k |
| 2048 × 2048 | 10 MB     | 239 k |
| 4096 × 4096 | 20 MB     | 478 k |

**Linear-in-k scaling is the red flag.** Any kernel emitter whose
PTX output grows with a spec-referenced dimension at rate faster
than O(log n) at synthesis time is structurally vulnerable to this
class of bug.

### Generalisation

The rule applies to any future PTX-emitting milestone whose spec
references a scale regime:

- FlashAttention-2 variants (CSHA, PCA, future) — spec references
  sequence lengths up to 8 k, 32 k, 128 k; ptxas tests must
  include at least one config at each intended upper bound.
- CSHA / PCA future model-size targets — coverage at realistic
  layer dims (e.g. `d_model = 8192`) is mandatory.
- KV compression (M42) — spec references cache sizes up to 512 MB;
  emitter tests must include at least one config exercising that
  bound.
- Tensor parallelism (M30) — spec references 8-way and 16-way
  shards; codegen ptxas tests must cover both.

Stated tightly: **test the scale you are targeting, not the scale
your unit tests happen to hit.** If the spec's target use case is
the Llama-3-8B prefill path, the kernel's ptxas-validity tests
must include at least one configuration at Llama-3-8B's actual
projection dimensions.

### Permanent regression coverage

The corresponding rewrite added six permanent ptxas regression
tests covering `n ∈ {1024, 2048, 4096}` crossed with
`rank ∈ {8, 16, 32, 64}` for LoRA and GatedLoRA, plus IA-3 at
`n = k = 4096`. Any future regression at scale is caught by the
regression test suite rather than silently reintroducing the
CPU-fallback trap.

### Retrospective on B.3.1

An earlier WRGA milestone reported that the fused forward path
had shipped end-to-end. In fact it shipped only for `n ≤ 64`. At
spec-referenced shapes (`n = 4096`) the emitter produced 20 MB of
PTX that ptxas could not compile, causing silent CPU fallback. The
fix converted the K-loop into a PTX runtime loop, reducing emitted
PTX to approximately 106 KB independent of k.

### Composition

B.2 caught numerical degeneration; B.3 catches structural
degeneration. Both target a test that passes "for reasons
unrelated to whether the kernel actually works." B.5 generalises
the same shape-vs-direct-probe gap from *coverage* to *code-path
attribution* — the failure mode is identical, expressed against
execution context rather than tensor shape.

---

## B.4 — *reserved, intentionally unused*

B.4 is the project-milestone label for the fused-forward staging
rewrite decision documented in §6.Z of the main paper. The
decision — fused stays opt-in (controlled by an
environment-controlled switch documented in §6.Z), unfused cuBLAS
is the default at every `m`, the staging rewrite was not pursued —
does not generalise to an institutional rule. The slot is
preserved in this appendix so the numbering of the remaining
rules matches their citation history.

---

## B.5 — Verify the specific code path, not a proxy

**Discovered during** the validation of the train-block fused-handler
design (a revision of the B.3.2 follow-on work); generalises B.3
from structural coverage to procedural attribution.

### Rule

Any spec claiming that a specific code path fires under specific
conditions must include a probe that directly verifies the claim.
Proxy measurements (launch-counter increments, memory-profile
changes, throughput deltas) can be satisfied by code paths *other*
than the one the spec claims, and a proxy's success does not
establish that the claimed path fires.

The probe must:

1. Exercise the specific conditions the spec's claim depends on.
2. Observe a signal that **only** the claimed path can produce.
3. Fail if the proxy is satisfied through an unclaimed path.

### Illustrative failures

Both caught only after implementation began:

- **B.3.1's Llama-scale claim** was verified via ptxas tests at
  `n ≤ 64`. The proxy passed because ptxas accepts small PTX even
  when the spec's target scale is structurally broken. The direct
  probe — feed the emitter a config at the spec-referenced shape
  (`n = 4096`) and confirm the resulting PTX is ptxas-compilable —
  would have caught the 20 MB / 478 k-line failure at commit time.
  Without it, the failure was discovered only during the B.3.2
  trigger measurement.

- **The train-block fused-handler claim** was verified by an
  initial verification step via a launch-counter increment in a
  top-level inference call (the model's forward pass invoked
  *after* the training block had completed). The proxy passed
  because inference-time fusion is unrelated to the train-block
  path the spec claimed to fix. The direct probe — call the
  model's forward pass *inside* the training block and assert
  that the GPU launch counter increments at least once per epoch
  — would have shown the counter at zero and surfaced the
  structural gap (the method-body recompile cache populated from
  unrewritten AST) before any implementation work began.

### Generalisation

The rule applies to any future spec where:

- an intervention targets a specific execution context (train
  block, inference-only, sm_80+, source-AD walk, the main
  compilation entry point, and so on);
- the intervention claims to change behaviour in that context;
- the verification has any ambiguity about which context satisfied
  the assertion.

When in doubt, instrument the context with a signal that **only**
the claimed path can produce — a parameter-count or adjoint-shape
assertion whose value depends on the specific code path executing.
Launch-counter increments are useful but insufficient if the same
counter can be incremented by adjacent paths.

### Application to the revised design

The revised design for the train-block fused-handler intervention
adopts two direct probes:

- **Launch-counter assertion under the train block.** With the
  GPU launch counter enabled and a 3-epoch training run, assert
  that the per-epoch launch count reaches at least the epoch
  count. This is proxy-level — necessary but not sufficient by
  B.5.

- **Trainable-parameters connection-count assertion.** Source-AD
  reports five out of five trainable tensor parameters connected
  (`x`, `W`, `A`, `B`, `gate`) rather than one out of five. This
  signal can reach 5 only if source-AD's Wengert walk has seen
  the fused FFI call and extracted its five inputs — exactly the
  code path the revised design claims to enable. **This is the
  B.5-compliant direct probe.**

The combination catches both the proxy-pass and the proxy-fail
mis-classification failure modes.

### B.5 is recursive

Any verification step whose result is **cited** by a spec as an
established fact is itself a code-path claim subject to B.5. "An
earlier verification step established the precondition" is not a
shield against B.5 unless that step itself satisfied B.5 — meaning
it exercised the specific conditions the precondition depends on,
not proxy conditions with overlapping behaviour.

A concrete failure was observed: a revised version of the same
design listed adapter-field CPU placement as a verified-elsewhere
non-goal. The earlier verification step had probed gate placement
via a top-level inference call rather than via an in-train-block
call. These hit different code paths: inference mode lets the
user manually reassign the adapter field on the GPU between the
training block and the probe; training mode has no such
reassignment site because the side-table is populated mid-train-
block. The earlier step satisfied a claim about the inference
path; it did not establish anything about the training path — but
the spec inherited the result as if it had. The clarification:
**the rule is recursive**. A spec's *verified elsewhere* citation
carries the full B.5 obligation of the original verification.

### Recurrence

Within a single short period, three instances of the same failure
mode surfaced:

1. B.3.1's Llama-scale claim verified via `n ≤ 64` ptxas tests.
2. The original train-block fused-handler claim verified via a
   top-level launch-counter.
3. A revised precondition for the same design verified via an
   inference-path placement probe.

The recurrence suggests the rule tends to be applied in
retrospect — when a downstream blocker surfaces the violation —
rather than proactively during specification and verification
preparation. The rule is correct; the *timing* of its application
is the operational gap.

### Composition

B.3 catches *scale*-proxy failures; B.5 catches *code-path*-proxy
failures. B.6 closes the temporal axis: even a B.5-compliant probe
reads a *number*, and that number inherits the validity of the
substrate it was measured on.

---

## B.6 — Re-validate a measurement-gated trigger against a clean baseline before scheduling the milestone

**Discovered during** the resolution of the B.3.2 trigger
measurement; composes B.3 ("test the scale you target") and B.5
("verify the code path") with a temporal corollary.

### Rule

A measurement-gated milestone trigger fires on a *number*. If that
number was produced on a substrate the measurement depends on —
and that substrate has known defects or has changed since — the
trigger fired on an artefact, not on the phenomenon it was
designed to detect. **Before scheduling the milestone, re-take the
measurement on a clean substrate and re-evaluate the trigger.**

### Concrete instantiation

The B.3.2 trigger condition was
`backward_time > 2.5 × forward_time`. It "fired" at **106×** in an
early reading. But that reading was taken before three substrate
fixes the measurement depended on:

1. **The matmul primitive** was the project's naïve f32 matmul,
   later replaced by cuBLAS (the early matmul achieved roughly
   1–2 TFLOPs/s, 15–30× below peak).
2. **The backward path** was suspected of CPU-falling-back on
   some operations — a contemporaneous diagnostic noted that
   roughly 50 s/iteration was consistent with CPU fallback or
   allocator thrash.
3. **The kernel profiler** reported zero durations (`cuEventCreate`
   before CUDA context initialisation), so the only available
   signal was wall-clock with overhead baked in, not per-kernel
   GPU time.

After all three were fixed, the clean per-op breakdown measured
**backward = 0.37 × forward** — the trigger does **not** fire.
The milestone that the 106× reading would have scheduled would
have optimised a 15 % slice of GPU time while the real
bottleneck (the fused **forward** kernel, 73 % of GPU) sat
untouched. **The trigger fired on noise from a broken substrate.**

### Diagnostic and rule statement

Earlier work already sensed this — it qualified *schedule B.3.2*
with *after per-op-profiling scoping*, and noted twice that
*optimising the top-of-stack bottleneck surfaces the next layer*.
B.6 names the underlying rule: **a trigger's number inherits the
validity of the substrate it was measured on.** When the
substrate is known-defective or has changed, the number is
provisional until re-measured.

### Generalisation

The rule applies to any measurement-gated milestone trigger —
CPDT Phase 2's 20 %-disagreement gate, CSHA tier-promotion
thresholds, any "schedule X when metric Y crosses Z":

1. Enumerate the substrate the metric depends on — primitives,
   fallback paths, profiler/timer validity, shapes, device
   placement.
2. If any substrate element is known-defective or has changed
   since the reading, the trigger is **provisional** —
   re-measure before acting.
3. Bake the substrate dependencies into the trigger spec itself,
   so a future reader knows which fixes invalidate an old
   reading.

### Two-instances corollary: structural perf claims are also measurement-gated

The same cycle exposed a structural variant: a *structural* perf
argument is itself a measurement-gated claim. The small-`m`
crossover hypothesis for the fused-staging rewrite was a
confident structural argument about an unmeasured quantity (the
kernel's own cost). An `m`-sweep falsified it at every `m`:
fused forward loses 15–67× at
`m ∈ {1, 16, 64, 256, 1024}`. The lesson generalises B.6 once
more: **never act on a structural perf claim without measuring
the quantity it assumes is small.**

### Composition

B.3/B.5/B.6 are the first three rules in a longer family
targeting tests, measurements, and verifications:

- **B.3** — test the *scale* you target;
- **B.5** — verify the *code path* you target, not a proxy;
- **B.6** — re-measure the *trigger* on a clean substrate before
  acting on it.

See also B.7 (test design), B.8 (user-visible diagnostic output),
and B.9 (compiler-internal classification heuristics) for the
remaining members of the family. All target the same
institutional failure: **a number that passes its own test for
reasons unrelated to the phenomenon the number was meant to
capture.**

---

## B.7 — Vacuous-pass test sentinel rule

**Discovered during** three independent post-implementation
reviews of WRGA's regression and ablation test suites, which
surfaced the same vacuity mechanism in three distinct shapes
over a few days.

### Rule

Any test whose purpose is to *pin existing behaviour* — regression
tests, contract tests, ablation contracts — must be checked to
ensure it can **actually fail** under the conditions it claims to
enforce. A test that cannot fail under its own premise is
**vacuous**: it passes for reasons unrelated to whether the system
under test still has the property the test claims to pin.

The check has three forms, one per common vacuity mechanism:

1. **Adversarial strip-test.** For each regression test of a
   defensive guard, temporarily remove the guard and confirm the
   test fails. A brief adversarial edit — temporarily removing the
   guard, re-running the tests, then reinstating the guard —
   provides strong evidence that the test genuinely catches the
   regression.
2. **State-shape assertion, not filename-presence.** When the
   contract is "this command produces no side effects in
   directory D", assert on the full read-back of `D`, not on the
   absence of a specific filename. A specific-filename probe can
   pass if the side effect lands at a *different* filename
   (e.g., a different code-path dispatch that the test's premise
   did not predict).
3. **Loop-premise guard.** When an assertion lives inside a loop
   over a collection, verify the loop has a guaranteed non-zero
   iteration count under the conditions being tested, *or* assert
   on the collection's shape directly. An `assert!` body inside a
   loop over an empty collection is dead code.

### Concrete instantiations

| Test                                             | Vacuity mechanism                                            | Fix                                                       |
|--------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| Null-guard regression for forward-only adapters  | Could not be assumed non-vacuous without an adversarial strip; the strip-test confirmed the guard's removal triggers the regression | Adversarial strip retained as the validation step before commit |
| No-side-effect contract for the analyse path     | Checked for a specific output filename; the dispatcher wrote elsewhere | Full directory state check — assert the directory contents equal the input set |
| Skip-fusion ablation contract                    | Assertion lived inside a loop over a collection that the ablation contract made empty | Premise check that the baseline produces a non-empty collection, plus a direct assertion on the collection's shape under ablation |

In each case the test was green at commit time and would have
passed for reasons **unrelated to the property the test claimed
to pin** had a subsequent edit drifted the system under test.
Each was caught after the fact by subsequent review, or by
proactive instrumentation during the original landing. The rule
names the discipline so the catch is up-front.

### Generalisation

The rule extends to any test category where the test's *intent*
and its *actual constraint* may diverge:

- **Regression tests** — strip the fix, confirm failure.
- **Contract tests** — verify the contract is more than the
  test's measured side effect; if "no side effects" is the
  contract, observe state shape, not enumerated artefacts.
- **Ablation contracts** — when asserting *X does not happen
  under ablation A*, pin a baseline that proves *X does happen
  without ablation A*. The pair establishes that the test's null
  hypothesis is refutable.

### Composition

B.2 said: *the tolerance must be tight enough that the
degenerate implementation fails.* B.5 said: *verify the code
path you target, not a proxy.* B.7 is the test-design instance
of both, applied to the test's own premise: **a test that
cannot fail under its premise is Goodhart-proxy at the meta
level.** The fix is the same in spirit — add a discriminator
the failure mode would actually trip on — applied to the test's
design rather than to its subject. **B.8 is the user-output
sibling: where B.7 demands the test be able to fail under its
premise, B.8 demands the user-visible diagnostic refuse or warn
when its premise is degraded.**

---

## B.8 — Silently-wrong-output discipline

**Discovered during** the construction of WRGA's ablation
harness; promoted on a single observation under the second
clause of the promotion criterion (the failure mode is
structurally evident across any tool that emits aggregated
diagnostics).

### Rule

When a diagnostic produces user-visible output, an *intervention
that disables one of the diagnostic's inputs* must either:

1. refuse to emit the diagnostic (a hard error on the
   incompatible combination), or
2. emit the diagnostic with a **prominent warning** that names
   the degraded input and points at a no-cost alternative.

A clean-looking diagnostic produced from a degraded input is the
worst outcome: it presents structured numerical information that
the reader will treat as authoritative, when in fact the numbers
are artefacts of the degradation, not of the system under
measurement.

### Concrete instantiation

The ablation harness implements `--wrga-ablate=spectral` as a
no-op that replaces the SVD-derived rank allocation with a record
carrying the suggested rank but zero adapter parameters. This is
a *correct* no-op for the `--wrga-analyze` report — the analyse
path does not need shape data. (Both CLI flags are documented in
§X of the main paper.)

But `--wrga-compare` recovers the per-site
`(m + n)` quantity from the same field to baseline
LoRA / AdaLoRA / GaLore / ReFT against the WRGA plan. With
`adapter_params = 0`, the recovery underflows to zero, so the
PEFT comparison table prints **all zeros** for every method. The
reader sees a clean, structurally well-formed table — and
concludes WRGA wins on every dimension by a factor of infinity.
The output is consistent with its own headers but **lies about
the world**.

The fix is to emit a warning on stderr at the entry point of the
diagnostic, naming the degraded input (spectral allocation) and
the no-cost alternative (`--wrga-analyze`). Two corresponding
tests verify that the warning fires under `--wrga-compare` and
is absent under `--wrga-analyze`.

### Generalisation

The rule applies to any future diagnostic emitter — WRGA report,
CSHA tier-promotion summary, CPDT plan dump, source-AD
trainable-params breakdown — that aggregates upstream signals
into a user-visible artefact:

- for each upstream signal, identify which interventions disable
  it (ablations, error fallbacks, opt-outs, low-confidence
  regimes);
- for each (intervention, downstream-consumer) pair, decide:
  does the downstream still produce *meaningful* output? If yes,
  keep silent. If no, refuse or warn — never silently degrade.

The bar for "meaningful" is the **failure mode of the reader**,
not the implementer's correctness intuition. If a reader could
draw the wrong conclusion from the output's *shape* even when
the output's *individual values* are technically correct, the
output is silently wrong.

The same discipline applies to user-visible mentions of features
that are documented or labelled but not yet wired — a
parenthetical `(pending)`, `(no-op)`, or `(stub)` marker
prevents the reader inferring executed behaviour from an
artefact's shape, just as the warning prevents the reader
inferring a real result from a degenerate input.

### Composition

B.2 was about saturation fixtures that pass under a stub; B.3
was about shapes that compile under degenerate input; B.5 was
about probes satisfied through unclaimed paths. **B.8 is the
user-output corollary of the same family**: a diagnostic that
*looks* correct but reports the diagnostic's degradation, not
the system's behaviour. **B.7 is the test-design sibling: where
B.8 demands the diagnostic refuse or warn when its premise is
degraded, B.7 demands the test be able to fail under its own
premise.** The remediation shape across all of them is the same
— add a discriminator (warning, refusal, label) that
distinguishes "meaningful output" from "output emitted in a
degraded regime."

B.8 is promoted on a single concrete instance under the second
clause of the promotion criterion: the failure class (clean
diagnostic from degraded input) is structurally evident in any
tool that aggregates upstream signals into a user-visible
artefact, and the remediation (refuse or warn) is a
language-agnostic discipline. A subsequent independent
observation will reinforce the rule but is not required for its
promotion.

---

## B.9 — Heuristic safety: tighten until false positives fail

**Discovered during** review of the reduction-fused adapter
implementation (paper §2.4 pattern 3); generalises to any
compile-time classification heuristic that participates in
dispatch.

### Rule

Any compile-time heuristic that classifies a name, shape,
decorator, or other identifier *into a category the codegen
pipeline dispatches on* must be tight enough that it **rejects
the dominant false positives in its domain**. Substring or
keyword matching on identifiers from a domain where the keyword
has multiple roles is a landmine — it cannot distinguish "this
is a reduction op" from "this is a projection layer whose name
happens to contain `pool`."

The discipline:

1. Enumerate the high-likelihood false positives in the keyword's
   problem domain (ML naming conventions for "reduce", "pool",
   "norm", "attn", and so on).
2. Tighten the rule until each of them is rejected.
3. Pin the tightening with a regression test that asserts the
   would-be false positives do **not** trigger the dispatch.

### Concrete instantiation

The fusion classifier for the reduction-fused adapter pattern
(paper §2.4 pattern 3) recognises reduction sites by name as a
fallback when the authoritative classification carrier is
unavailable (legacy callers, test helpers, pre-scan). The
initial rule was a substring match shown in Listing B.9.1.

**Listing B.9.1: Initial substring-based reduction-name heuristic.**

```rust
fn is_reduction_name(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("reduce") || n.contains("pool")
        || n.contains("sum") || n.contains("mean")
}
```

Review pointed out that the keywords have **two** roles in ML
naming:

| Real reduction         | False positive (matmul-shaped projection) |
|------------------------|-------------------------------------------|
| `head.pool`            | `pool_q_proj`                             |
| `head.global.pool`     | `reduce_op_result_proj`                   |

The substring match flips every false positive on the right
column to the reduction-fused dispatch target, which subsequently
degrades to the unfused mul path — a silent performance
regression with no test signal.

The fix is a structural rule (Listing B.9.2) that requires a
**dot-delimited path component equal to** one of
`{reduce, pool, sum, mean}`. The new rule matches `head.pool`,
`head.global.pool`, and the bare token `pool`, but does not
match `pool_q_proj` because the latter has no dot-delimited
component equal to `pool`.

**Listing B.9.2: Structural rule requiring a dot-delimited path component to equal one of the reduction tokens.**

```rust
fn is_reduction_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.split('.').any(|c| matches!(c, "reduce" | "pool" | "sum" | "mean"))
}
```

A regression test pins five would-be false positives as
non-matches.

### Generalisation

The pattern recurs across the compiler:

- decorator-name classifiers (`@adapter`, `@fuse`, `@wrga`)
  must not flip arbitrary identifier-substring matches into
  pipeline dispatch;
- shape-pattern classifiers ("this looks like a Q/K/V
  projection") must enumerate the false positives in the model
  architectures the classifier sees;
- dtype-string classifiers must not match arbitrary substrings
  of user-supplied dtype aliases.

Stated tightly: **if your heuristic cannot distinguish a
plausible false positive in the wild, tighten until it does.**
Heuristics that fall back to "unstructured" matching on
overloaded keywords are silent-degradation vectors; structural
rules (dot-delimited tokens, trailing-suffix anchoring,
authoritative carriers) are the antidote.

### Architectural corollary

Wherever practical, the dispatch should not rely on the
heuristic at all — it should carry the **authoritative
classification** end-to-end. The structural fix adopted
alongside the tightened heuristic was not just hardening the
regex but adding an authoritative classification field (the
`SiteKind` enumeration) to the adapter-placement record, so
downstream passes consult the upstream-known truth and the
heuristic survives only as a fallback for legacy callers. The
heuristic-tightening rule is the local mitigation; the
authoritative-carrier pattern is the structural cure, and is
likely to be promoted in a future revision of this appendix to
its own rule.

### Composition

B.2/B.3/B.5/B.6 catch the proxy-vs-direct-probe failure in
*tests*. B.7 catches it in *test design*. B.8 catches it in
*diagnostic output*. B.9 catches it at the deepest layer: **the
compiler's own classification heuristics are themselves
Goodhart proxies for the structural facts they approximate.** A
heuristic that admits false positives is a proxy whose "success
criterion" (substring match) has decoupled from its underlying
purpose (categorising operations).

---

## Appendix B summary

| Rule | Domain                                | Discriminator                                            |
|------|---------------------------------------|----------------------------------------------------------|
| B.2  | Approximation-helper numerical tests  | Mid-range fixture above the stub-attack tolerance        |
| B.3  | PTX emitter shape coverage            | ptxas-valid at spec-referenced upper bound               |
| B.4  | *Reserved (see preface)*              | —                                                         |
| B.5  | Code-path attribution                 | Signal only the claimed path can produce                 |
| B.6  | Measurement-gated triggers            | Re-measure on a clean substrate                          |
| B.7  | Regression / contract test design     | The test must be able to fail under its own premise      |
| B.8  | User-visible diagnostic output        | Refuse or warn — never silently degrade                  |
| B.9  | Compile-time classification heuristics| Reject plausible false positives in the domain           |

Each rule names a specific mechanism by which a check could (and
did) silently degenerate into a Goodhartised proxy. None
subsumes the others; all compose. Future episodes that surface a
new mechanism in the same family will be added as B.10 and
onward, with the same shape: discovery context, rule statement,
probe construction, composition. Candidates already on the
horizon — most prominently the authoritative-carrier architectural
pattern named in B.9's corollary — will be promoted once their
generalisation has been observed under a second clause of the
promotion criterion. Mechanisms that do not fit the
Anti-Goodhart family — pure engineering lessons, architectural
preferences, ergonomic discoveries — belong in the contributor
documentation rather than in Appendix B.
