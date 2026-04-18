# NSL-WRGA Research -- Appendix B addendum (pending integration into next PDF regeneration)

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
