# CPDT Calibrate Audit — Phase 1 Step 0

**Audited file:** `crates/nsl-codegen/src/cpdt_precision.rs`
**File commit SHA:** `de740817cf796301d1dad9d1e1c57ef75e900128` (verify via: `git show de740817cf796301d1dad9d1e1c57ef75e900128:crates/nsl-codegen/src/cpdt_precision.rs`)
**Audit date:** 2026-04-18
**Audit method:** grep for field accesses against `WeightEntry`, `PrecisionConfig`, and call sites of name-inference helpers.

## Grep Invocation

```
grep -nE "entry\.|cfg\.|layer_of|is_embedding|is_norm|is_first_or_last|position_criticality|sensitivity|num_elements|\.data|\.dtype|\.name" /tmp/cpdt_precision_prerefactor.rs
```

## Grep Output

```
6://!   * **High sensitivity** → FP32 m + FP32 v  (8 bytes/param)
14://! spectral_condition × gradient_magnitude × position_criticality
59:    /// Provably-stable parameters (very low sensitivity).  INT8 m + v.
93:    pub sensitivity_score: f64,
168:    if entry.num_elements == 0 {
171:    let bw = entry.dtype.byte_width();
174:    for i in 0..entry.num_elements {
176:        if off + bw > entry.data.len() {
179:        let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
196:    if entry.num_elements == 0 {
199:    let bw = entry.dtype.byte_width();
201:    for i in 0..entry.num_elements {
203:        if off + bw > entry.data.len() {
206:        let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
209:    (sum_sq / entry.num_elements as f64).sqrt()
215:pub fn position_criticality(layer: u32, n_layers: u32) -> f64 {
223:fn layer_of(name: &str) -> Option<u32> {
234:fn is_embedding(name: &str) -> bool {
239:fn is_norm(name: &str) -> bool {
244:fn is_first_or_last_layer(layer: Option<u32>, n_layers: u32) -> bool {
251:fn sensitivity(
259:        .map(|l| position_criticality(l, n_layers))
261:    let elements = entry.num_elements.max(1) as f64;
272:    if is_norm(name) || is_embedding(name) || is_first_or_last_layer(layer, n_layers) {
275:    if score >= cfg.high_threshold {
277:    } else if score >= cfg.medium_threshold {
279:    } else if score >= cfg.low_threshold {
291:    let layer = layer_of(&entry.name);
292:    let score = sensitivity(entry, layer, cfg.n_layers);
293:    let tier = choose_tier(&entry.name, layer, cfg.n_layers, score, cfg);
295:    let param_bytes = (entry.num_elements as u64) * (entry.dtype.byte_width() as u64);
296:    let optim_bytes = (entry.num_elements as u64) * (m.bytes() as u64 + v.bytes() as u64);
297:    let stochastic = cfg.embedding_stochastic_rounding && is_embedding(&entry.name);
299:    name: entry.name.clone(),
305:    sensitivity_score: score,
320:        baseline_fp32 += (entry.num_elements as u64) * 8; // FP32 m + v
350:            num_elements: rows * cols,
364:    fn position_criticality_u_shape() {
365:        assert!((position_criticality(0, 8) - 1.3).abs() < 1e-9);
366:        assert!((position_criticality(7, 8) - 1.3).abs() < 1e-9);
367:        assert!(position_criticality(3, 8) < 1.1);
402:    fn middle_layer_low_sensitivity_gets_lower_tier() {
423:            baseline_fp32 += (entry.num_elements as u64) * 8;
460:    fn layer_of_recognises_common_patterns() {
461:        assert_eq!(layer_of("blocks.6.attn.wq"), Some(6));
462:        assert_eq!(layer_of("layers.12.norm"), Some(12));
463:        assert_eq!(layer_of("h.3.mlp.fc"), Some(3));
464:        assert_eq!(layer_of("embedding.weight"), None);
```

## Enumerated Inputs

The pre-refactor `cpdt_precision.rs::classify_param` reads:

**From `WeightEntry`:**
- `entry.data` (via `spectral_condition_proxy` on line 179, `gradient_magnitude_estimate` on lines 206, 209)
- `entry.dtype` (via `to_f64` dispatch on lines 179, 206; `byte_width()` on lines 171, 199, 295)
- `entry.num_elements` (denominator in `sensitivity` on line 261; parameter byte computation on lines 295-296, 320; bounds checks on lines 168, 196, 174, 201)
- `entry.name` (routed through `layer_of` on line 291, `is_norm` on line 272, `is_embedding` on lines 272, 297, 299)

**From `PrecisionConfig`:**
- `cfg.n_layers` (position criticality normalization on line 259; first/last detection on line 293)
- `cfg.high_threshold`, `cfg.medium_threshold`, `cfg.low_threshold` (tier selection on lines 275-279)
- `cfg.embedding_stochastic_rounding` (stochastic flag on line 297)

**Derived (pure functions of `entry.name` + `cfg.n_layers`):**
- `layer_of(name)` — defined line 223, parses `"blocks.N."`, `"layers.N."`, `"h.N."` prefixes
- `is_embedding(name)` — defined line 234, lowercase substring match for `embed`, `tok_embeddings.weight`, `wte.weight`
- `is_norm(name)` — defined line 239, lowercase substring match for `norm` (excluding `normalize`)
- `is_first_or_last_layer(layer, n_layers)` — defined line 244, logic: `layer == 0 || layer + 1 == n_layers`
- `position_criticality(layer, n_layers)` — defined line 215, U-shape curve returning f64 weight
- `sensitivity(entry, layer, n_layers)` — defined line 251, computes gradient_magnitude_estimate(entry) × position_criticality(layer, n_layers) / element_count

## Conclusion

**Input set:** `{entry.name, entry.data, entry.dtype, entry.num_elements, cfg.*}`.

**Closed-form inversion:** sound for calibration step 2. The weight-dependent factors (`entry.data` via `spectral_condition_proxy` + `gradient_magnitude_estimate`) are what make calibration-to-byte-identity meaningful; the non-weight factors decompose cleanly into the unified scorer's `position_criticality` + `element_count` + `layer_kind` override.

**Phase 1 decision:** since the pre-refactor already requires weights (it panics on missing data), Phase 1's byte-identity target is **weights-present path vs pre-refactor, byte-identical on the baseline corpus**. The no-weights path of the unified scorer is a NEW capability; its quality is measured by the 95% parameter-weighted agreement gate between no-weights and weights-present paths on the same corpus.

**Spectral deferred:** Phase 1 drops the `spectral_condition_proxy` factor entirely. Phase 2 re-introduces spectral as a fourth multiplicative factor with its own cache. The unified scorer's Phase 1 formula is:

```
sensitivity(W, l, kind) = gradient_magnitude_est(W) × position_criticality(l, L) / element_count(W)
```

No additional hidden inputs surfaced. Proceed to calibration Step 1.

---

## Joint Solver Determinism (Phase 1 Prep Check)

**File:** `crates/nsl-codegen/src/cpdt_joint.rs`
**Check:** grep for `rand|thread_rng|SystemTime|Instant::now|HashMap::iter`.
**Result:** zero matches. `cpdt_joint::solve` is deterministic given identical inputs.

**Consequence:** byte-identity of tier assignments → byte-identity of `total_optim_bytes` → byte-identity of joint-solver output. No explicit joint-solver snapshot test needed in Commit 1; the tier-assignment byte-identity regression gate is sufficient.
