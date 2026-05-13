# PCA Tier B.1.5 + B.2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `nsl-codegen-bench` measurement harness, run M2/M6 measurements against the gate fixture, apply the §10 outcomes-matrix decision from the design spec, and conditionally ship B.2 backward integration.

**Architecture:** New single-file binary at `crates/nsl-codegen/src/bin/bench.rs` invokes `synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b)` with both `None` (Tier B off) and `Some(...)` (Tier B on), times each via CUDA events, reads back the M3 skip-decision HBM buffer for ratio computation, and emits parseable output lines per the spec's §5.2 contract. Shell scripts orchestrate the 5×100 measurement protocol. After measurements, the outcome (keep / keep-with-sparsity-gate / revert) determines whether B.2 backward work proceeds.

**Tech Stack:** Rust 1.95.0, cudarc 0.19, existing `nsl-codegen` crate, shell (`bash`-compatible scripts via Git Bash on Windows), insta snapshots, existing `PackingFixture` matrix.

**Design spec:** `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` (commit `cab2ba58`).

**Pre-implementation reading required:**

- The design spec (read in full before starting Task 1).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs` for `synthesize_flash_attention_ptx_v2_with_tier_b` signature.
- `crates/nsl-codegen/src/pca_tilerange.rs` for `emit_skip_decision_writeback` and the existing instrumentation buffer shape.
- `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs` for the 6 currently-`#[ignore]`'d fixtures + the launch-harness gap.
- `crates/nsl-codegen/tests/fixtures/mod.rs` for the `PackingFixture` matrix.

---

## File Structure

### Created

- `crates/nsl-codegen/src/bin/bench.rs` — bench binary entry point (B1.5-1).
- `crates/nsl-codegen/src/bin/bench/cli.rs` — CLI parsing module (B1.5-1).
- `crates/nsl-codegen/src/bin/bench/fixtures.rs` — fixture registry (gate + sensitivity + parity dispatch) (B1.5-2, B1.5-4).
- `crates/nsl-codegen/src/bin/bench/launch.rs` — kernel launch + CUDA-event timing harness (B1.5-1, B1.5-3).
- `crates/nsl-codegen/src/bin/bench/output.rs` — `tier_b_bench_result:` line emitter (B1.5-1).
- `scripts/measure_tier_b_m2.sh` — M2 skip-ratio orchestration (B1.5-5).
- `scripts/measure_tier_b_m6.sh` — M6 wall-time orchestration (B1.5-5).
- `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` — measurement findings (B1.5-6).
- `docs/superpowers/specs/2026-05-XX-tier-b-b2-predicate-verification-findings.md` — V-B.2-predicate findings (B2-1; only if B1.5-6 outcome ∈ {keep, keep-with-sparsity-gate}).

### Modified

- `crates/nsl-codegen/Cargo.toml` — register the `bench` binary (B1.5-1).
- `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs` — un-ignore the 6 fixtures; wire bit-identical assertion (B1.5-3).
- `crates/nsl-codegen/tests/fixtures/mod.rs` — add the three sensitivity fixtures and the single gate fixture (B1.5-2, B1.5-4).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs` — `should_emit_tier_b(config) -> bool` helper added; bench binary toggles via this central point (B1.5-1, B1.5-6 conditional on revert outcome).
- `crates/nsl-codegen/src/pca_tilerange.rs` — `emit_skip_decision_writeback` PTX gains the round-robin owning_warp formula (B1.5-3 — must precede the parity tests because they read the HBM buffer).
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/ds_compute.rs` — gains skip predicate at tile-loop boundary (B2-2; conditional).
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs` — gains range-table preamble (B2-2; conditional).
- `crates/nsl-codegen/tests/fa_v2_snapshots.rs` — backward snapshot variants for Tier B-on (B2-2; conditional).
- `crates/nsl-codegen/tests/pca_sass_byte_identity.rs` — backward SASS BRA.U assertions (B2-2; conditional).

---

## Phase B.1.5 — Measurement infrastructure + outcome decision

### Task B1.5-1: Bench binary scaffolding (CLI + output + launch + Cargo wiring)

**Files:**
- Create: `crates/nsl-codegen/src/bin/bench.rs`
- Create: `crates/nsl-codegen/src/bin/bench/cli.rs`
- Create: `crates/nsl-codegen/src/bin/bench/launch.rs`
- Create: `crates/nsl-codegen/src/bin/bench/output.rs`
- Modify: `crates/nsl-codegen/Cargo.toml`

**Spec reference:** §5 of the design spec — invocation contract, output format, exit codes, --seed.

- [ ] **Step 1: Register the binary in Cargo.toml**

Add to `crates/nsl-codegen/Cargo.toml` under the `[[bin]]` or workspace section:

```toml
[[bin]]
name = "bench"
path = "src/bin/bench.rs"
required-features = ["cuda", "debug_kernel_instrumentation"]
```

Verify: `cargo metadata --format-version 1 | grep '"name":"bench"'` returns a hit.

- [ ] **Step 2: Write a failing test for the CLI parser**

Create `crates/nsl-codegen/tests/bench_cli_smoke.rs`:

```rust
//! Smoke test that the bench binary's CLI parser accepts the documented surface.

#[test]
fn cli_parses_required_args() {
    let args = vec!["bench", "--fixture", "gate_4096", "--tier-b", "on"];
    let parsed = nsl_codegen::bin::bench::cli::parse_from(&args).expect("parses");
    assert_eq!(parsed.fixture, "gate_4096");
    assert_eq!(parsed.tier_b, nsl_codegen::bin::bench::cli::TierB::On);
    assert_eq!(parsed.seed, 42);
    assert_eq!(parsed.iterations, 100);
}

#[test]
fn cli_rejects_missing_fixture() {
    let args = vec!["bench", "--tier-b", "on"];
    assert!(nsl_codegen::bin::bench::cli::parse_from(&args).is_err());
}
```

Run: `cargo test -p nsl-codegen --test bench_cli_smoke`
Expected: FAIL with `bench::cli` module not found.

- [ ] **Step 3: Implement the minimal CLI parser**

Create `crates/nsl-codegen/src/bin/bench/cli.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TierB {
    On,
    Off,
}

#[derive(Debug, Clone)]
pub struct Args {
    pub fixture: String,
    pub tier_b: TierB,
    pub seed: u64,
    pub iterations: u32,
    pub emit_time_only: bool,
}

impl Args {
    pub fn parse() -> Result<Self, String> {
        let argv: Vec<String> = std::env::args().collect();
        let argv_ref: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
        parse_from(&argv_ref)
    }
}

pub fn parse_from(argv: &[&str]) -> Result<Args, String> {
    let mut fixture: Option<String> = None;
    let mut tier_b: Option<TierB> = None;
    let mut seed: u64 = 42;
    let mut iterations: u32 = 100;
    let mut emit_time_only = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i] {
            "--fixture" => { fixture = Some(argv.get(i + 1).ok_or("missing fixture")?.to_string()); i += 2; }
            "--tier-b" => {
                let v = argv.get(i + 1).ok_or("missing tier-b value")?;
                tier_b = Some(match *v {
                    "on"  => TierB::On,
                    "off" => TierB::Off,
                    _ => return Err(format!("invalid tier-b: {v}")),
                });
                i += 2;
            }
            "--seed" => { seed = argv.get(i + 1).ok_or("missing seed")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?; i += 2; }
            "--iterations" => { iterations = argv.get(i + 1).ok_or("missing iterations")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?; i += 2; }
            "--emit-time-only" => { emit_time_only = true; i += 1; }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        fixture: fixture.ok_or("--fixture required")?,
        tier_b: tier_b.ok_or("--tier-b required")?,
        seed,
        iterations,
        emit_time_only,
    })
}
```

Also expose the module at `crates/nsl-codegen/src/bin/bench.rs`'s `mod cli;` and add a `pub mod bin { pub mod bench { pub mod cli; } }` in `src/lib.rs` so the test can reach it. (Alternative: re-export via `cfg(test)` — pick the simpler structure.)

Run: `cargo test -p nsl-codegen --test bench_cli_smoke`
Expected: PASS.

- [ ] **Step 4: Write the output line emitter test**

Create `crates/nsl-codegen/tests/bench_output_format.rs`:

```rust
use nsl_codegen::bin::bench::output::{emit_result, ResultLine};

#[test]
fn emit_result_format_matches_spec_section_5_2() {
    let line = emit_result(&ResultLine {
        fixture: "gate_4096".into(),
        tier_b_on: true,
        median_us: 234.567,
        n: 100,
        skip_ratio: 0.487,
        seed: 42,
    });
    assert_eq!(
        line,
        "tier_b_bench_result:fixture=gate_4096:tier_b=on:median_us=234.567:n=100:skip_ratio=0.487:seed=42"
    );
}

#[test]
fn emit_result_tier_b_off_label() {
    let line = emit_result(&ResultLine {
        fixture: "gate_4096".into(), tier_b_on: false, median_us: 257.0,
        n: 100, skip_ratio: 0.0, seed: 42,
    });
    assert!(line.contains(":tier_b=off:"));
    assert!(line.contains(":skip_ratio=0:") || line.contains(":skip_ratio=0.0"));
}
```

Run: `cargo test -p nsl-codegen --test bench_output_format`
Expected: FAIL — module not found.

- [ ] **Step 5: Implement the output emitter**

Create `crates/nsl-codegen/src/bin/bench/output.rs`:

```rust
#[derive(Debug, Clone)]
pub struct ResultLine {
    pub fixture: String,
    pub tier_b_on: bool,
    pub median_us: f64,
    pub n: u32,
    pub skip_ratio: f64,
    pub seed: u64,
}

pub fn emit_result(line: &ResultLine) -> String {
    format!(
        "tier_b_bench_result:fixture={}:tier_b={}:median_us={}:n={}:skip_ratio={}:seed={}",
        line.fixture,
        if line.tier_b_on { "on" } else { "off" },
        line.median_us,
        line.n,
        line.skip_ratio,
        line.seed,
    )
}
```

Run: `cargo test -p nsl-codegen --test bench_output_format`
Expected: PASS.

- [ ] **Step 6: Implement the launch harness skeleton (kernel-launch loop with CUDA events)**

Create `crates/nsl-codegen/src/bin/bench/launch.rs` with the structure documented in spec §8.2. Use cudarc 0.19's low-level driver API (matching the probe binary's style from PR #168's `tier_b_smem_probe.rs`):

```rust
use cudarc::driver::sys;
use std::os::raw::c_void;

pub struct LaunchResult {
    pub median_us: f64,
    pub skip_ratio: f64,
}

pub unsafe fn time_kernel_launches(
    func: sys::CUfunction,
    args: &mut [*mut c_void],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shmem_bytes: u32,
    iterations: u32,
    skip_decisions_buf: Option<(sys::CUdeviceptr, usize)>,
) -> Result<LaunchResult, String> {
    // 1. Create start + stop CUevents.
    // 2. Loop iterations times: cuEventRecord(start), cuLaunchKernel, cuEventRecord(stop),
    //    cuEventSynchronize, accumulate ms via cuEventElapsedTime.
    // 3. Median: collect all per-iter times, sort, take middle (or sum/n if simpler — spec
    //    §8.1 says median of 5 OUTER, sum of 100 INNER; this function is for ONE outer run.
    //    Return median across the 100 inner runs as a proxy for "characteristic time of this run").
    // 4. If skip_decisions_buf provided: cuMemcpyDtoH the buffer once at the end (decisions are
    //    deterministic per-fixture; one snapshot suffices), count set bits / total slots.
    todo!("implement per spec §8.2; smoke-test against null-launch placeholder")
}
```

For the initial scaffolding, return a stub that compiles. The real implementation depends on having a real PTX module loaded; defer to Task B1.5-2 once we have a fixture and PTX.

Add `mod launch;` in `bench.rs`.

Run: `cargo build -p nsl-codegen --features cuda --bin bench`
Expected: builds (warns about unused `todo!()`; that's fine).

- [ ] **Step 7: Wire bench main() that ties cli + output + launch**

`crates/nsl-codegen/src/bin/bench.rs`:

```rust
#[cfg(feature = "cuda")]
mod cli;
#[cfg(feature = "cuda")]
mod output;
#[cfg(feature = "cuda")]
mod launch;

#[cfg(feature = "cuda")]
fn main() {
    let args = match cli::Args::parse() {
        Ok(a) => a,
        Err(msg) => { eprintln!("error: {msg}"); std::process::exit(1); }
    };
    // Task B1.5-2 fills in fixture loading + launch + output here.
    eprintln!("bench scaffolding complete; fixture loading not yet implemented");
    eprintln!("invoked: fixture={} tier_b={:?} seed={} iter={}", args.fixture, args.tier_b, args.seed, args.iterations);
    std::process::exit(3); // framework not yet ready
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("bench binary requires --features cuda");
    std::process::exit(3);
}
```

Run: `cargo run -p nsl-codegen --features "cuda debug_kernel_instrumentation" --bin bench -- --fixture gate_4096 --tier-b on`
Expected: exits 3 with the scaffolding-complete message.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/Cargo.toml \
        crates/nsl-codegen/src/bin/bench.rs \
        crates/nsl-codegen/src/bin/bench/cli.rs \
        crates/nsl-codegen/src/bin/bench/launch.rs \
        crates/nsl-codegen/src/bin/bench/output.rs \
        crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/tests/bench_cli_smoke.rs \
        crates/nsl-codegen/tests/bench_output_format.rs
git commit -m "feat(pca-tier-b15-b2): bench binary scaffolding (B1.5-1)

CLI + output line format + launch-harness skeleton per design spec
§5. Invocation contract pinned: tier_b_bench_result: prefix, exit
codes 0/1/2/3, --seed default 42, --iterations default 100.

Gate: builds; CLI parses required and optional args; output line
emitter matches the §5.2 format string exactly."
```

---

### Task B1.5-2: Gate fixture wiring + launch harness completion

**Files:**
- Modify: `crates/nsl-codegen/src/bin/bench/fixtures.rs` (create new submodule)
- Modify: `crates/nsl-codegen/src/bin/bench.rs`
- Modify: `crates/nsl-codegen/src/bin/bench/launch.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/mod.rs` (add `should_emit_tier_b(config)`)

**Spec reference:** §4.1 (gate fixture dims) + §5.2 (CLI surface) + §8 (measurement protocol).

- [ ] **Step 1: Add `should_emit_tier_b(config)` helper**

In `crates/nsl-codegen/src/flash_attention_v2/mod.rs`, add:

```rust
/// Returns true when Tier B's PTX is to be emitted for this config.
///
/// Today returns false unconditionally — Tier B is dormant infrastructure
/// per PR #168. The bench binary inspects this flag via the
/// `synthesize_flash_attention_ptx_v2_with_tier_b` second argument
/// (passing `Some(...)` to override).
///
/// After M2/M6 measurement (B1.5-6), this is updated per the outcomes
/// matrix (§10 of the 2026-05-13 design spec).
pub fn should_emit_tier_b(_config: &FlashAttentionConfig) -> bool {
    false
}
```

Build: `cargo build -p nsl-codegen`. PASS.

- [ ] **Step 2: Write gate fixture registry test**

Create `crates/nsl-codegen/tests/bench_fixtures.rs`:

```rust
use nsl_codegen::bin::bench::fixtures;

#[test]
fn gate_fixture_dims_match_spec_section_4_1() {
    let f = fixtures::lookup("gate_4096").expect("gate fixture exists");
    assert_eq!(f.config.block_q, 64);
    assert_eq!(f.config.block_kv, 64);
    assert_eq!(f.config.head_dim, 64);
    assert!(f.config.causal);
    assert!(f.config.segment_masked);
    assert_eq!(f.seq_len, 4096);
    assert_eq!(f.batch, 4);
    assert!((f.target_sparsity - 0.5).abs() < 1e-6);
}

#[test]
fn unknown_fixture_returns_none() {
    assert!(fixtures::lookup("nonexistent").is_none());
}
```

Run: `cargo test -p nsl-codegen --test bench_fixtures`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the gate fixture**

Create `crates/nsl-codegen/src/bin/bench/fixtures.rs`:

```rust
use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: &'static str,
    pub config: FlashAttentionConfig,
    pub seq_len: u32,
    pub batch: u32,
    pub target_sparsity: f64,
}

pub fn lookup(name: &str) -> Option<Fixture> {
    REGISTRY.iter().find(|f| f.name == name).cloned()
}

static REGISTRY: &[Fixture] = &[
    Fixture {
        name: "gate_4096",
        config: FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 64,
            causal: true, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
        },
        seq_len: 4096,
        batch: 4,
        target_sparsity: 0.5,
    },
    // Sensitivity fixtures added in Task B1.5-4.
];
```

Run: `cargo test -p nsl-codegen --test bench_fixtures`
Expected: PASS.

- [ ] **Step 4: Implement the launch harness's real body**

Fill in `crates/nsl-codegen/src/bin/bench/launch.rs`'s `time_kernel_launches` per the §8.2 CUDA-events pattern. Critical pieces:

- Generate random Q/K/V from `--seed` via a deterministic PRNG (e.g., StepRng with linear sequence, or pull in `rand` if not already a transitive dep).
- Generate the segment mask from the same seed; target_sparsity controls the segment count.
- Build PTX via `synthesize_flash_attention_ptx_v2_with_tier_b(config, if tier_b_on { Some((seq_len, residency)) } else { None })`.
- Allocate HBM for Q, K, V, O (forward outputs), and the skip-decisions buffer per `emit_skip_decision_writeback` shape.
- Run 100 inner iterations; CUDA events bracket each; sum ms; convert to median_us.
- Read back skip-decisions buffer once at end; count set bits / total slots = skip_ratio.

This is the largest step in the plan (~80 LOC of unsafe cudarc code). Mirror `tier_b_smem_probe.rs`'s style.

- [ ] **Step 5: Wire main() to load fixture + launch + emit**

Update `crates/nsl-codegen/src/bin/bench.rs`:

```rust
fn main() {
    let args = ...;
    let fixture = match fixtures::lookup(&args.fixture) {
        Some(f) => f,
        None => { eprintln!("unknown fixture: {}", args.fixture); std::process::exit(1); }
    };
    let result = match unsafe { launch::run_fixture(&fixture, args.tier_b == cli::TierB::On, args.seed, args.iterations) } {
        Ok(r) => r,
        Err(msg) => { eprintln!("launch error: {msg}"); std::process::exit(2); }
    };
    let line = output::emit_result(&output::ResultLine {
        fixture: fixture.name.into(),
        tier_b_on: args.tier_b == cli::TierB::On,
        median_us: result.median_us,
        n: args.iterations,
        skip_ratio: result.skip_ratio,
        seed: args.seed,
    });
    println!("{line}");
    std::process::exit(0);
}
```

- [ ] **Step 6: Smoke test on hardware**

```bash
cargo run -p nsl-codegen --features "cuda debug_kernel_instrumentation" --bin bench -- \
    --fixture gate_4096 --tier-b on --seed 42 --iterations 100
```

Expected: emits a well-formed `tier_b_bench_result:` line on stdout; exits 0.

```bash
cargo run -p nsl-codegen --features "cuda debug_kernel_instrumentation" --bin bench -- \
    --fixture gate_4096 --tier-b off --seed 42 --iterations 100
```

Expected: emits a different `tier_b_bench_result:` line (same fixture, different `tier_b=` field, likely different `median_us` and `skip_ratio=0`); exits 0.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/bin/bench/fixtures.rs \
        crates/nsl-codegen/src/bin/bench.rs \
        crates/nsl-codegen/src/bin/bench/launch.rs \
        crates/nsl-codegen/src/flash_attention_v2/mod.rs \
        crates/nsl-codegen/tests/bench_fixtures.rs
git commit -m "feat(pca-tier-b15-b2): gate fixture + launch harness (B1.5-2)

Gate fixture pinned per design §4.1: segment-masked causal,
seq_len=4096, head_dim=64, batch=4, block 64x64, sparsity=50%,
sm_120. should_emit_tier_b(config) helper added in
flash_attention_v2/mod.rs as the central toggle point (currently
returns false unconditionally; bench binary overrides via
synthesize_flash_attention_ptx_v2_with_tier_b's second arg).

Smoke test: both --tier-b=on and --tier-b=off produce valid
output lines on RTX 5070 Ti."
```

---

### Task B1.5-3: Parity tier — un-ignore the 6 M3 fixtures + bit-identical assertion

**Files:**
- Modify: `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs`
- Modify: `crates/nsl-codegen/src/pca_tilerange.rs` (round-robin owning_warp formula in `emit_skip_decision_writeback`)

**Spec reference:** §6.1 (bit-identical assertion) + §6.2 (round-robin owning_warp formula) + §6.4 (edge cases).

- [ ] **Step 1: Read the current `emit_skip_decision_writeback` shape**

```bash
grep -n "emit_skip_decision_writeback\|fn emit_skip_decision_writeback" crates/nsl-codegen/src/pca_tilerange.rs
```

Confirm the current v1 hardcodes warp 0. The PTX-level body to change is the writeback-predicate construction.

- [ ] **Step 2: Write a regression snapshot test for the new round-robin PTX**

Create `crates/nsl-codegen/tests/pca_tier_b_owning_warp_snapshot.rs`:

```rust
//! Snapshot test for the round-robin owning_warp PTX emission.
//! Spec §6.2 of 2026-05-13-pca-tier-b15-and-b2-design.md.

use nsl_codegen::pca_tilerange;
use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};

fn fixture() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[test]
fn round_robin_writeback_emission() {
    let mut ptx = String::new();
    pca_tilerange::emit_skip_decision_writeback(
        &mut ptx, &fixture(), /* num_q_tiles = */ 64, /* num_kv_tiles = */ 64,
        /* num_warps = */ 4,
    );
    // Sentinels indicating the round-robin shape (not warp-0-always).
    assert!(ptx.contains("shr.u32") && ptx.contains("%warp_id"),
            "round-robin requires warp_id derivation:\n{ptx}");
    assert!(ptx.contains("rem.u32") || ptx.contains("and.b32"),
            "round-robin uses rem.u32 (or and.b32 if num_warps is power of 2):\n{ptx}");
    insta::assert_snapshot!("owning_warp_round_robin_gate_fixture", ptx);
}
```

Run: `cargo test -p nsl-codegen --test pca_tier_b_owning_warp_snapshot`
Expected: FAIL — current emit produces warp-0-always (no `shr.u32 %warp_id`).

- [ ] **Step 3: Modify `emit_skip_decision_writeback` to use round-robin**

Apply the PTX-level change from spec §6.2. Critical: wrap in PTX lexical scope `{ ... }` per IR-007 to avoid register collisions across call sites.

```ptx
{
    .reg .u32 %tid, %warp_id, %lane, %owning_warp;
    .reg .pred %p_warp_owner, %p_lane_owner, %p_writeback;
    mov.u32 %tid, %tid.x;
    shr.u32 %warp_id, %tid, 5;
    mul.lo.u32 %owning_warp, %qt, NUM_KV_TILES;
    add.u32 %owning_warp, %owning_warp, %kvt;
    rem.u32 %owning_warp, %owning_warp, NUM_WARPS;
    setp.eq.u32 %p_warp_owner, %warp_id, %owning_warp;
    and.b32 %lane, %tid, 0x1F;
    setp.eq.u32 %p_lane_owner, %lane, 0;
    and.pred %p_writeback, %p_warp_owner, %p_lane_owner;
    @%p_writeback st.global.u8 [...], %skip_decision;
}
```

Also add `debug_assert!(num_warps > 0)` Rust-side per §6.4.

Re-run snapshot test; review with `cargo insta review` and accept.

- [ ] **Step 4: Verify ptxas still compiles cleanly**

```bash
# Build a small test kernel that includes the new emission
cargo test -p nsl-codegen --features cuda --test pca_sass_byte_identity tier_b_sass 2>&1 | tail -20
```

Expected: All `tier_b_sass::*` tests pass (or only the pre-existing `tier_b_preamble_sm80_uniform_proxy_via_brau` fails per the prior baseline known-flake; new tests pass).

- [ ] **Step 5: Write the bit-identical assertion harness**

Modify `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs`:

Replace each `#[ignore]` test with a real implementation that calls the bench binary as a subprocess (via `std::process::Command`) twice — once with `--tier-b on`, once with `--tier-b off`, with the same `--seed` — and asserts the captured outputs are byte-identical at the `O` (forward output) tensor.

Skeleton:

```rust
fn run_bench_capture_output_tensor(fixture: &str, tier_b: &str, seed: u64) -> Vec<u8> {
    let output_path = format!("/tmp/bench_out_{}_{}.bin", fixture, tier_b);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_bench"))
        .args([
            "--fixture", fixture,
            "--tier-b", tier_b,
            "--seed", &seed.to_string(),
            "--iterations", "1",
            "--dump-output", &output_path,  // NEW flag added in B1.5-3's bench updates
        ])
        .status()
        .expect("bench runs");
    assert!(status.success(), "bench failed for {fixture}/{tier_b}");
    std::fs::read(&output_path).expect("output exists")
}

#[test]
fn parity_fixture_1_byte_identical() {
    let on = run_bench_capture_output_tensor("parity_1", "on", 42);
    let off = run_bench_capture_output_tensor("parity_1", "off", 42);
    assert_eq!(on, off, "Tier B output bit-differs from Tier-B-off on parity_1");
}

// ... five more tests for parity_2..parity_6 ...
```

Add `--dump-output <path>` flag to the bench binary (CLI + main()) that writes the O tensor to a file at end of run.

- [ ] **Step 6: Register the parity fixtures**

Add `parity_1` through `parity_6` to `bench/fixtures.rs::REGISTRY`, populated from the existing `PackingFixture` matrix in `crates/nsl-codegen/tests/fixtures/mod.rs`.

- [ ] **Step 7: Run all 6 parity tests on hardware**

```bash
cargo test -p nsl-codegen --features "cuda debug_kernel_instrumentation" --test pca_tier_b_m3_parity --release
```

Expected: 6 passed; 0 failed.

If any fails: investigate — bit-difference between Tier-B-on and Tier-B-off means the skip logic is no longer correctness-preserving (real bug, not a tolerance issue). Stop and root-cause.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/pca_tilerange.rs \
        crates/nsl-codegen/src/bin/bench/cli.rs \
        crates/nsl-codegen/src/bin/bench.rs \
        crates/nsl-codegen/src/bin/bench/fixtures.rs \
        crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs \
        crates/nsl-codegen/tests/pca_tier_b_owning_warp_snapshot.rs \
        crates/nsl-codegen/tests/snapshots/
git commit -m "feat(pca-tier-b15-b2): parity tier + round-robin owning_warp (B1.5-3)

Six pca_tier_b_m3_parity fixtures un-ignored; bit-identical
assertion implemented via bench-binary subprocess (--dump-output
captures the O tensor; main test diffs the two outputs).

emit_skip_decision_writeback now uses round-robin
owning_warp(qt, kvt) = (qt * num_kv_tiles + kvt) % num_warps per
design §6.2. PTX wrapped in lexical scope per IR-007.
debug_assert!(num_warps > 0) per §6.4. Snapshot for the new
shape committed; ptxas-clean on sm_120.

Bit-identical assertion: skip logic is correctness-preserving by
construction (skipped tiles contribute exactly zero); any
divergence would be a real bug, not a tolerance issue."
```

---

### Task B1.5-4: Sensitivity tier — three fixtures at sparsities {10%, 50%, 90%}

**Files:**
- Modify: `crates/nsl-codegen/src/bin/bench/fixtures.rs`
- Create: `crates/nsl-codegen/tests/bench_sensitivity_fixtures.rs`

**Spec reference:** §4.3 (sensitivity tier composition + curve-shape sub-question).

- [ ] **Step 1: Write sensitivity-fixture lookup test**

Create `crates/nsl-codegen/tests/bench_sensitivity_fixtures.rs`:

```rust
use nsl_codegen::bin::bench::fixtures;

#[test]
fn sensitivity_fixtures_share_gate_dims_differ_in_sparsity() {
    let gate = fixtures::lookup("gate_4096").unwrap();
    for (name, sparsity) in [("sensitivity_10", 0.1), ("sensitivity_50", 0.5), ("sensitivity_90", 0.9)] {
        let f = fixtures::lookup(name).expect(name);
        assert_eq!(f.config.block_q,   gate.config.block_q);
        assert_eq!(f.config.block_kv,  gate.config.block_kv);
        assert_eq!(f.config.head_dim,  gate.config.head_dim);
        assert_eq!(f.seq_len,          gate.seq_len);
        assert_eq!(f.batch,            gate.batch);
        assert!((f.target_sparsity - sparsity).abs() < 1e-6);
    }
}

#[test]
fn sensitivity_50_matches_gate_fixture_structurally() {
    let s50 = fixtures::lookup("sensitivity_50").unwrap();
    let gate = fixtures::lookup("gate_4096").unwrap();
    assert!((s50.target_sparsity - gate.target_sparsity).abs() < 1e-6,
            "sensitivity_50 is structurally identical to gate fixture per design §4.3.1");
}
```

Run: FAIL (sensitivity fixtures not registered).

- [ ] **Step 2: Add the three fixtures to the registry**

In `bench/fixtures.rs::REGISTRY`, append three entries with the gate fixture's dims but `target_sparsity` ∈ {0.1, 0.5, 0.9}.

Re-run test: PASS.

- [ ] **Step 3: Run sensitivity fixtures end-to-end**

```bash
for s in 10 50 90; do
    cargo run -p nsl-codegen --features "cuda debug_kernel_instrumentation" --bin bench -- \
        --fixture sensitivity_$s --tier-b on --seed 42 --iterations 100
done
```

Expected: three valid output lines; `skip_ratio` increases monotonically (10 → 50 → 90).

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/bin/bench/fixtures.rs \
        crates/nsl-codegen/tests/bench_sensitivity_fixtures.rs
git commit -m "feat(pca-tier-b15-b2): sensitivity tier fixtures (B1.5-4)

Three sensitivity fixtures at sparsities {10%, 50%, 90%} share
the gate fixture's other dims per design §4.3. sensitivity_50
is structurally identical to gate_4096 (cross-check redundancy
per §4.3.1). Smoke-tested on RTX 5070 Ti; skip_ratio
monotonically increases with sparsity as expected."
```

---

### Task B1.5-5: M2/M6 shell scripts

**Files:**
- Create: `scripts/measure_tier_b_m2.sh`
- Create: `scripts/measure_tier_b_m6.sh`

**Spec reference:** §8 (measurement protocol) + §5.2 (CLI / output contract).

- [ ] **Step 1: Write `measure_tier_b_m6.sh` (wall-time median-of-5)**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Spec ref: docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md §8

FIXTURE="${1:-gate_4096}"
OUTER_RUNS=5
INNER_ITERS=100
SEED=42
BENCH="cargo run -q -p nsl-codegen --features 'cuda debug_kernel_instrumentation' --bin bench --release --"

echo "fixture,outer_run,tier_b,median_us,skip_ratio,seed"
for run in $(seq 1 $OUTER_RUNS); do
    for tier_b in on off; do
        OUT=$($BENCH --fixture "$FIXTURE" --tier-b "$tier_b" --seed "$SEED" --iterations "$INNER_ITERS" --emit-time-only)
        # Parse: tier_b_bench_result:fixture=X:tier_b=Y:median_us=Z:n=N:skip_ratio=R:seed=S
        median=$(echo "$OUT" | sed -n 's/.*:median_us=\([0-9.]*\):.*/\1/p')
        skip=$(echo "$OUT"   | sed -n 's/.*:skip_ratio=\([0-9.]*\):.*/\1/p')
        echo "$FIXTURE,$run,$tier_b,$median,$skip,$SEED"
    done
done
```

Make executable: `chmod +x scripts/measure_tier_b_m6.sh`.

- [ ] **Step 2: Write `measure_tier_b_m2.sh` (skip-ratio across fixtures)**

```bash
#!/usr/bin/env bash
set -euo pipefail
SEED=42
INNER_ITERS=100
BENCH="cargo run -q -p nsl-codegen --features 'cuda debug_kernel_instrumentation' --bin bench --release --"

echo "fixture,tier_b,skip_ratio,median_us,seed"
for fx in gate_4096 sensitivity_10 sensitivity_50 sensitivity_90 parity_1 parity_2 parity_3 parity_4 parity_5 parity_6; do
    for tier_b in on off; do
        OUT=$($BENCH --fixture "$fx" --tier-b "$tier_b" --seed "$SEED" --iterations "$INNER_ITERS" --emit-time-only)
        median=$(echo "$OUT" | sed -n 's/.*:median_us=\([0-9.]*\):.*/\1/p')
        skip=$(echo "$OUT"   | sed -n 's/.*:skip_ratio=\([0-9.]*\):.*/\1/p')
        echo "$fx,$tier_b,$skip,$median,$SEED"
    done
done
```

Make executable.

- [ ] **Step 3: Smoke-test the scripts**

```bash
bash scripts/measure_tier_b_m6.sh gate_4096 > /tmp/m6_smoke.csv
bash scripts/measure_tier_b_m2.sh             > /tmp/m2_smoke.csv
wc -l /tmp/m6_smoke.csv /tmp/m2_smoke.csv
head -3 /tmp/m6_smoke.csv
```

Expected: m6_smoke.csv has 11 lines (1 header + 5 runs × 2 tier_b); m2_smoke.csv has 21 lines (1 header + 10 fixtures × 2 tier_b).

- [ ] **Step 4: Commit**

```bash
git add scripts/measure_tier_b_m2.sh scripts/measure_tier_b_m6.sh
git commit -m "feat(pca-tier-b15-b2): M2/M6 measurement scripts (B1.5-5)

Shell scripts orchestrate the 5x100 measurement protocol per
design §8. Parse bench binary's tier_b_bench_result: output
line; emit CSV for findings doc.

measure_tier_b_m6.sh: median-of-5 wall-time at one fixture
(default gate_4096); emits 1+(5*2)=11 lines.
measure_tier_b_m2.sh: skip-ratio sweep across gate + sensitivity
+ parity fixtures; emits 1+(10*2)=21 lines."
```

---

### Task B1.5-6: Run measurements + commit findings + apply outcome

**Files:**
- Create: `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md`
- Conditional (revert outcome only): `crates/nsl-codegen/src/flash_attention_v2/mod.rs` (`should_emit_tier_b` stays returning false; tests get `#[ignore]` markers)

**Spec reference:** §3 (acceptance bar) + §10 (outcomes matrix) + §3.4 (revert semantics).

- [ ] **Step 1: Run the full measurement protocol**

```bash
bash scripts/measure_tier_b_m6.sh gate_4096          > /c/tmp/tier_b_m6_gate.csv
bash scripts/measure_tier_b_m6.sh sensitivity_10     > /c/tmp/tier_b_m6_s10.csv
bash scripts/measure_tier_b_m6.sh sensitivity_50     > /c/tmp/tier_b_m6_s50.csv
bash scripts/measure_tier_b_m6.sh sensitivity_90     > /c/tmp/tier_b_m6_s90.csv
bash scripts/measure_tier_b_m2.sh                    > /c/tmp/tier_b_m2_sweep.csv
```

- [ ] **Step 2: Compute the gate-fixture decision**

From `/c/tmp/tier_b_m6_gate.csv`:

```
median_on  = median(rows where tier_b==on)
median_off = median(rows where tier_b==off)
walltime_win = (median_off - median_on) / median_off
```

From `/c/tmp/tier_b_m2_sweep.csv`, take the gate_4096 row with tier_b=on; that's the skip_ratio.

**Decision:**

- `walltime_win >= 0.10 AND skip_ratio >= 0.40` → outcome is **keep** (unconditional or with sparsity gate, per sensitivity curve).
- Otherwise → outcome is **revert**.

If keep: compute the sensitivity curve from sensitivity_10/50/90 medians. If 10% sparsity shows `walltime_win < 0` (Tier B hurts at low sparsity), outcome is **keep with sparsity gate**, threshold derived from the curve. Else **keep unconditionally**.

- [ ] **Step 3: Write findings doc**

`docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md`:

```markdown
# Tier B M2/M6 Measurement Findings

**Date:** 2026-05-XX
**Hardware:** NVIDIA RTX 5070 Ti (sm_120, Blackwell)
**Driver:** 591.86 / CUDA 13.2
**Protocol:** Spec §8 of `2026-05-13-pca-tier-b15-and-b2-design.md`
**Seed:** 42
**Inner iterations:** 100, outer runs: 5

## Gate fixture (gate_4096) — keep/revert decision

[paste from /c/tmp/tier_b_m6_gate.csv]

- median_on  = ... us
- median_off = ... us
- walltime_win = ...%
- skip_ratio   = ...

**Threshold check:**
- walltime_win >= 10%: [YES/NO]
- skip_ratio   >= 40%: [YES/NO]

**Outcome:** [keep unconditionally | keep with sparsity gate | revert]

## Sensitivity curve

| sparsity | walltime_win | skip_ratio | shape inference |
|---|---|---|---|
| 10% | ... | ... | ... |
| 50% | ... | ... | matches gate w/in +-10% ? |
| 90% | ... | ... | ... |

**Curve shape:** [monotonic-linear | thresholded | saturating]

## Parity tier results

All six parity_N fixtures: bit-identical (passed during B1.5-3); no
re-run needed here.

## Decision rationale

[justification per §10 outcomes matrix]
```

- [ ] **Step 4a: If outcome is "revert" — apply dead-code state**

Edit `crates/nsl-codegen/src/flash_attention_v2/mod.rs`:

`should_emit_tier_b(config)` stays returning false (no change; it already does).

Add `#[ignore = "Tier B disabled per M2/M6 results 2026-05-XX"]` to every test in `pca_tier_b_*.rs` files that exercises the `Some(...)` Tier B path. Document the 6-month decay timer start date in `should_emit_tier_b`'s docstring.

Update `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_pca_tier_b_v1_shipped.md` and `MEMORY.md` to record the revert outcome and the decay timer date.

- [ ] **Step 4b: If outcome is "keep" — leave should_emit_tier_b returning false (planner-dispatch is downstream)**

`should_emit_tier_b` continues returning false — the bench binary overrides via `Some(...)` directly. The planner-side dispatch decision (when to opt in for non-bench callers) is its own spec, downstream of this PR.

Update the memory file to record the keep outcome.

- [ ] **Step 4c: If outcome is "keep with sparsity gate" — record threshold value**

Same as keep, plus: findings doc records the sparsity threshold value derived from the curve, so the downstream planner-dispatch spec has a concrete number to use.

- [ ] **Step 5: Commit findings + any code state changes**

```bash
git add -f docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md
[git add other files conditional on outcome]
git commit -m "docs(pca-tier-b15-b2): M2/M6 measurement findings + outcome (B1.5-6)

Outcome: [outcome name].

[outcome-specific description]"
```

---

## Phase B.2 — Backward kernel integration (gated on B1.5-6 outcome ∈ {keep, keep-with-sparsity-gate})

**Skip this phase entirely if B1.5-6's outcome is revert.**

### Task B2-1: V-B.2-predicate pre-implementation verification

**Files:**
- Create: `docs/superpowers/specs/2026-05-XX-tier-b-b2-predicate-verification-findings.md`

**Spec reference:** §7.3 of the design spec.

- [ ] **Step 1: Read backward's iteration order**

```bash
grep -n "for.*qt.*kvt\|for.*kvt.*qt\|q_tile_iter\|kv_tile_iter" \
    crates/nsl-codegen/src/flash_attention_v2/phases/backward/*.rs
```

Inspect the discovered iteration order. Identify whether outer loop is Q (case α) or KV (case β).

- [ ] **Step 2: Read existing skip-predicate call site in forward**

```bash
grep -n "emit_skip_predicate" crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs
```

Note the operand order and register-class usage. Compare to what backward would need given its iteration order.

- [ ] **Step 3: Write findings doc**

Create `docs/superpowers/specs/2026-05-XX-tier-b-b2-predicate-verification-findings.md`:

```markdown
# V-B.2-predicate — Findings

**Spec reference:** §7.3 of `2026-05-13-pca-tier-b15-and-b2-design.md`
**Date:** 2026-05-XX
**Budget used:** [actual minutes]

## Backward iteration order

[Q-outer/KV-inner = case alpha | KV-outer/Q-inner = case beta]

Evidence: [grep results, file:line references showing the loop structure]

## Predicate-emission shape

[Same as forward (wholesale reuse) | Transposed (IterationOrder parameter needed)]

## Decision

B.2 proceeds with [wholesale reuse | parameterized emit_skip_predicate].
[If parameterized: estimate ~50 additional LOC split between pca_tilerange.rs and ds_compute.rs.]
```

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/specs/2026-05-XX-tier-b-b2-predicate-verification-findings.md
git commit -m "docs(pca-tier-b15-b2): V-B.2-predicate findings (B2-1)

[case alpha = wholesale reuse | case beta = IterationOrder param]"
```

---

### Task B2-2: Backward kernel integration

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/ds_compute.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/mod.rs` (`synthesize_flash_attention_ptx_v2_backward_with_tier_b` constructor)
- Modify: `crates/nsl-codegen/src/pca_tilerange.rs` (if case β: add `IterationOrder` parameter to `emit_skip_predicate`)
- Modify: `crates/nsl-codegen/tests/fa_v2_snapshots.rs` (backward snapshots)

**Spec reference:** §7.4 + §7.1 + §7.2.

- [ ] **Step 1: Write the backward-with-tier-b synthesizer test**

Mirror the forward's snapshot test pattern in `fa_v2_snapshots.rs`:

```rust
#[test]
fn backward_kernel_full__32x32x32_nocsha_tier_b_off() {
    let cfg = csha_canonical();  // existing helper
    let ptx_no_tier_b = synthesize_flash_attention_ptx_v2_backward(&cfg);
    let ptx_explicit_none = synthesize_flash_attention_ptx_v2_backward_with_tier_b(&cfg, None);
    assert_eq!(ptx_no_tier_b, ptx_explicit_none, "no-op guarantee: None == 1-arg form");
    insta::assert_snapshot!("backward_kernel_full__32x32x32_nocsha_tier_b_off", String::from_utf8(ptx_no_tier_b).unwrap());
}

#[test]
fn backward_kernel_full__32x32x32_nocsha_tier_b_on() {
    let cfg = csha_canonical();
    let ptx = synthesize_flash_attention_ptx_v2_backward_with_tier_b(&cfg, Some((4096, SegmentResidency::Tiled)));
    insta::assert_snapshot!("backward_kernel_full__32x32x32_nocsha_tier_b_on", String::from_utf8(ptx).unwrap());
}
```

Run: FAIL — `synthesize_flash_attention_ptx_v2_backward_with_tier_b` doesn't exist; nor does the snapshot.

- [ ] **Step 2: Mirror the forward's synthesizer signature for backward**

Add `synthesize_flash_attention_ptx_v2_backward_with_tier_b(config, tier_b: Option<(u32, SegmentResidency)>)`. The existing 1-arg form `synthesize_flash_attention_ptx_v2_backward(config)` becomes a wrapper that passes None.

Plumb `tier_b: Option<...>` through `backward/prelude.rs` and `backward/ds_compute.rs` (the latter being the analog of forward's `s_compute.rs`).

- [ ] **Step 3: Add range-table preamble + skip predicate to backward**

In `backward/prelude.rs::emit`, after the existing Tier A setup, conditional Tier B preamble call (mirroring `forward/prelude.rs`):

```rust
if let Some((seq_len, residency)) = tier_b {
    pca_tilerange::emit_range_table_preamble(out, config, seq_len, "seg_smem", /* offset for backward */);
}
```

In `backward/ds_compute.rs`, at the tile-loop boundary, call `pca_tilerange::emit_skip_predicate` (either wholesale per case α, or with `IterationOrder::KVOuter` per case β).

- [ ] **Step 4: Accept the new snapshots**

```bash
cargo test -p nsl-codegen --test fa_v2_snapshots backward_kernel 2>&1
INSTA_UPDATE=always cargo test -p nsl-codegen --test fa_v2_snapshots backward_kernel
cargo insta review  # or accept-all if confident
```

- [ ] **Step 5: Verify no-op guarantee for non-Tier-B configs**

```bash
cargo test -p nsl-codegen --test fa_v2_snapshots 2>&1 | grep -E "test result|FAIL|PASS"
```

Expected: existing backward snapshots (without Tier B) are byte-identical to today's tree.

- [ ] **Step 6: ptxas + SASS verification**

```bash
cargo test -p nsl-codegen --features cuda --test pca_sass_byte_identity tier_b_sass 2>&1
```

Expected: backward-kernel SASS BRA.U assertions pass on sm_120.

- [ ] **Step 7: Run backward parity (B2-3 preview)**

```bash
cargo test -p nsl-codegen --features "cuda debug_kernel_instrumentation" --test pca_tier_b_m3_parity --release
```

Expected: all six parity fixtures still pass (forward parity; backward parity in B2-3).

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/mod.rs \
        crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs \
        crates/nsl-codegen/src/flash_attention_v2/phases/backward/ds_compute.rs \
        crates/nsl-codegen/src/pca_tilerange.rs \
        crates/nsl-codegen/tests/fa_v2_snapshots.rs \
        crates/nsl-codegen/tests/snapshots/
git commit -m "feat(pca-tier-b15-b2): backward kernel integration (B2-2)

synthesize_flash_attention_ptx_v2_backward_with_tier_b plumbs the
skip predicate end-to-end through backward/prelude.rs and
ds_compute.rs. Reuses emit_range_table_preamble +
emit_skip_predicate from forward's pca_tilerange [wholesale per
case alpha | with IterationOrder::KVOuter per case beta].

No-op guarantee: synthesize_flash_attention_ptx_v2_backward
(1-arg form) produces byte-identical PTX as before. Verified via
fa_v2_snapshots test pinning.

ptxas-clean on sm_120; SASS BRA.U assertions pass per IR-007."
```

---

### Task B2-3: Backward parity tier extension

**Files:**
- Modify: `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs` (add backward variants for the 6 fixtures)

**Spec reference:** §7.4 (backward parity tier extension).

- [ ] **Step 1: Add backward parity tests for the 6 fixtures**

For each parity_N fixture, add a sibling test that runs backward with Tier B on/off and asserts bit-identical dQ, dK, dV outputs.

Skeleton:

```rust
#[test]
fn parity_fixture_1_backward_byte_identical() {
    let (on_dq, on_dk, on_dv) = run_bench_backward("parity_1", "on", 42);
    let (off_dq, off_dk, off_dv) = run_bench_backward("parity_1", "off", 42);
    assert_eq!(on_dq, off_dq, "dQ differs");
    assert_eq!(on_dk, off_dk, "dK differs");
    assert_eq!(on_dv, off_dv, "dV differs");
}
```

Add a `--dump-backward-outputs <path>` flag to the bench binary that writes dQ/dK/dV after a backward run; integrate with the parity-fixture launch.

- [ ] **Step 2: Run all 6 backward parity tests**

```bash
cargo test -p nsl-codegen --features "cuda debug_kernel_instrumentation" --test pca_tier_b_m3_parity backward --release
```

Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs \
        crates/nsl-codegen/src/bin/bench.rs \
        crates/nsl-codegen/src/bin/bench/cli.rs
git commit -m "feat(pca-tier-b15-b2): backward parity tier (B2-3)

Six backward-pass parity tests assert bit-identical dQ/dK/dV
between Tier-B-on and Tier-B-off across the existing
PackingFixture matrix. Symmetric-correctness property (§7.1)
verified on hardware."
```

---

## Push + PR

After all in-scope tasks complete (B.1.5-1..6, plus B.2 if outcome ∈ {keep, keep-with-sparsity-gate}):

- [ ] **Step 1: Push branch**

```bash
git push -u origin worktree-feat-pca-tier-b15-and-b2:feat/pca-tier-b15-and-b2
```

- [ ] **Step 2: Open PR with outcome-conditional title**

Title per §11.3:

- keep unconditionally → `feat(pca-tier-b): forward + backward (B.1.5 + B.2 measurement-validated)`
- keep with sparsity gate → `feat(pca-tier-b): forward + backward (B.1.5 + B.2; sparsity-gated dispatch follows)`
- revert → `feat(pca-tier-b): disable Tier B dispatch (B.1.5 measurements below thresholds)`

```bash
gh pr create --head feat/pca-tier-b15-and-b2 --title "<title>" --body "$(cat <<'EOF'
## Summary
- Tier B.1.5 measurement harness (bench binary + scripts + findings)
- [If keep] Tier B.2 backward kernel integration with symmetric-correctness reuse
- [If revert] Dead-code state per IR-009 with 6-month decay timer

## Measurements
- M6 wall-time win: X%
- M2 skip ratio: Y%
- Outcome: <name>

## Test plan
- [ ] All 6 parity fixtures pass (forward)
- [ ] [If B.2] All 6 backward parity fixtures pass
- [ ] [If B.2] Snapshots stable
- [ ] [If B.2] SASS BRA.U assertions pass on sm_120
- [ ] [If revert] Tier B tests #[ignore]'d; should_emit_tier_b returns false

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

After writing this plan, the spec-coverage check:

- §3 (acceptance bar) → driven by B1.5-6 decision logic.
- §4 (three-tier matrix) → B1.5-2 (gate), B1.5-3 (parity), B1.5-4 (sensitivity).
- §5 (bench binary) → B1.5-1 with §5.2 invocation contract pinned.
- §6 (M3 parity + round-robin) → B1.5-3.
- §7 (B.2 inheritance with V-B.2-predicate) → B2-1, B2-2, B2-3.
- §8 (measurement protocol) → B1.5-5 (scripts) + B1.5-6 (execution).
- §10 (outcomes matrix) → B1.5-6 decision tree.
- §11 (milestones table) → this plan's task structure mirrors it 1:1.

No placeholders. Type consistency: `synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b: Option<(u32, SegmentResidency)>)` used consistently; `should_emit_tier_b(config) -> bool` used consistently; `Fixture { config, seq_len, batch, target_sparsity }` used consistently.

Plan is complete; ready to execute via subagent-driven-development (recommended) or executing-plans.
