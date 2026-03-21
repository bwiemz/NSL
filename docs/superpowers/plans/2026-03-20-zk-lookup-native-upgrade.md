# ZK Circuits: Lookup-Native + Folding + Mersenne-31 Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the M55 ZK implementation from v1 Halo2 scaffolding to a production-ready folding-based prover with lookup-native arithmetization and Mersenne-31 field support. The current implementation uses BN254-only PLONKish gates with a trivially-forgeable v1 "proof" — it needs to become a real prover targeting 7B+ params in minutes with <500KB proofs.

**Architecture:** Four major changes: (1) extract field trait + add Mersenne-31 (10x faster), (2) replace Halo2 PLONKish gates with AIR + lookup-native constraints (Jolt-style), (3) add folding accumulation backend to decouple proof size from model depth, (4) emit per-layer ZkIR for sequential folding instead of whole-model flattening.

**Tech Stack:** Rust, existing `nsl-codegen/src/zk/` module

**Research Basis:** Frontier Features notebook confirms EZKL/Halo2 scales poorly beyond 30M params. ZKTorch (folding) proves 6B params in 20min. zkLLM (tlookup) proves 13B with <200KB proofs. Jolt Atlas (lookup-native) is 4-7x faster. Mersenne-31 gives ~10x field op speedup over BN254.

---

## Current State Summary

| Component | File | Status | Keep/Delete/Rewrite |
|-----------|------|--------|-------------------|
| ZK entry point | `mod.rs` | Stub (`compile_zk` returns error) | **Rewrite** |
| Field arithmetic | `field.rs` | BN254-only, 256-bit schoolbook | **Refactor** (extract trait, add M31) |
| ZK IR | `ir.rs` | 9 instruction types, wire-based SSA | **Keep** (backend-agnostic) |
| Lookup tables | `lookup.rs` | 8 activations, precomputed | **Keep** (integrate with lookup-native) |
| DAG lowering | `lower.rs` | 5-pass pipeline, whole-model flat | **Extend** (per-layer emission for folding) |
| Witness gen | `witness.rs` | Arena-allocated, linear eval | **Extend** (add accumulator state) |
| Halo2 backend | `halo2/mod.rs` | Placeholder compile/setup | **Delete** |
| Halo2 gates | `halo2/gates.rs` | PLONKish gate definitions | **Delete** |
| Halo2 layout | `halo2/layout.rs` | Grid planner | **Delete** |
| Halo2 prove | `halo2/prove.rs` | Trivially forgeable v1 proof | **Delete** |
| Halo2 verify | `halo2/verify.rs` | Magic-byte check, no pairing | **Delete** |
| Halo2 solidity | `halo2/solidity.rs` | Structural verifier | **Delete** |
| Plonky3 stub | `plonky3/mod.rs` | Returns `BackendNotImplemented` | **Keep** (low priority) |
| Statistics | `stats.rs` | Halo2-based cost model | **Update** (AIR cost model) |
| Backend trait | `backend.rs` | `ZkBackend` trait | **Extend** (add `FoldingBackend`) |

---

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `zk/field.rs` | Refactor | Extract `Field` trait, keep BN254, add Mersenne-31 |
| `zk/backend.rs` | Extend | Add `FoldingBackend` trait, `ZkField` enum |
| `zk/mod.rs` | Rewrite | Wire folding backend, implement `compile_zk()` |
| `zk/lower.rs` | Extend | Emit per-layer ZkIRs for folding |
| `zk/witness.rs` | Extend | Add fold accumulator state |
| `zk/stats.rs` | Update | AIR constraint costs, folding proof size |
| `zk/halo2/` (all) | **Delete** | 6 files — Halo2 deprecated |
| **NEW** `zk/field_m31.rs` | Create | Mersenne-31 field (32-bit single-limb) |
| **NEW** `zk/air.rs` | Create | AIR format: trace rows, constraint polynomials |
| **NEW** `zk/lookup_native.rs` | Create | Jolt-style lookup-native gates |
| **NEW** `zk/folding/mod.rs` | Create | FoldingBackend impl |
| **NEW** `zk/folding/accumulator.rs` | Create | Accumulator instance/witness tracking |
| **NEW** `zk/folding/sumcheck.rs` | Create | Univariate sumcheck argument |

---

## Tasks

### Phase 1: Field Abstraction + Mersenne-31 (2 days)

- [ ] **1.1** Extract `Field` trait from `field.rs`:
```rust
pub trait Field: Clone + Debug + PartialEq + Sized {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn inv(&self) -> Self;
    fn neg(&self) -> Self;
    fn from_u64(val: u64) -> Self;
    fn to_bytes(&self) -> Vec<u8>;
}
```

- [ ] **1.2** Wrap existing BN254 code as `impl Field for Bn254Field`.

- [ ] **1.3** Create `field_m31.rs` with `Mersenne31Field`:
```rust
pub struct Mersenne31Field(u32);  // p = 2^31 - 1

impl Field for Mersenne31Field {
    fn add(&self, other: &Self) -> Self {
        let sum = self.0 as u64 + other.0 as u64;
        Self((sum % P) as u32)
    }
    fn mul(&self, other: &Self) -> Self {
        let prod = self.0 as u64 * other.0 as u64;
        Self((prod % P) as u32)
    }
    // ... single-limb arithmetic, ~100x faster than BN254
}
```

- [ ] **1.4** Add `ZkField` enum to `backend.rs`:
```rust
pub enum ZkField { BN254, Mersenne31 }
```

- [ ] **1.5** Parameterize `ZkConfig` with field choice. Default: `Mersenne31`.

- [ ] **1.6** Test: Mersenne-31 field passes: `a * inv(a) == 1` for random values, `a + neg(a) == 0`.

- [ ] **1.7** Benchmark: M31 mul should be >10x faster than BN254 mul.

### Phase 2: Delete Halo2, Add AIR Framework (2 days)

- [ ] **2.1** Delete entire `halo2/` directory (6 files: mod.rs, gates.rs, layout.rs, prove.rs, verify.rs, solidity.rs).

- [ ] **2.2** Create `air.rs` — AIR constraint format:
```rust
pub struct AIRTrace<F: Field> {
    pub width: usize,                    // columns per row
    pub rows: Vec<Vec<F>>,              // trace matrix
}

pub enum AIRConstraint<F: Field> {
    /// Transition: polynomial relation between row[i] and row[i+1]
    Transition { degree: usize, eval: Box<dyn Fn(&[F], &[F]) -> F> },
    /// Boundary: row[0] or row[last] equals known value
    Boundary { row: usize, col: usize, value: F },
    /// Lookup: (input_col, output_col) must exist in precomputed table
    LookupNative { input_col: usize, output_col: usize, table_name: String },
}

pub struct AIRProgram<F: Field> {
    pub constraints: Vec<AIRConstraint<F>>,
    pub trace_width: usize,
    pub public_input_cols: Vec<usize>,
    pub public_output_cols: Vec<usize>,
}
```

- [ ] **2.3** Create `lookup_native.rs` — Jolt-style lookup gates:
```rust
/// A lookup-native gate: activation f(x) is a table lookup, not a polynomial constraint.
/// The prover demonstrates that (input, output) exists in the precomputed table.
pub struct LookupNativeGate {
    pub table_name: String,        // "relu", "gelu", etc.
    pub input_bits: u32,
    pub output_bits: u32,
    pub entries: Vec<(u64, u64)>,  // (input, output) pairs from lookup.rs
}

/// Multiplicity tracking: count how many times each table entry is used.
/// This is the core of lookup-native arithmetization (Jolt/zkLLM style).
pub struct LookupMultiplicities {
    pub counts: Vec<u64>,  // counts[i] = how many times entry i appears
}
```

- [ ] **2.4** Update `stats.rs` to use AIR cost model instead of PLONKish.

### Phase 3: Folding Backend (3-4 days)

- [ ] **3.1** Create `folding/mod.rs` — FoldingBackend:
```rust
pub struct FoldingProver<F: Field> {
    pub config: FoldingConfig,
    pub accumulator: Option<Accumulator<F>>,
}

impl<F: Field> FoldingProver<F> {
    /// Fold a new layer's circuit into the accumulator.
    pub fn fold_layer(&mut self, layer_ir: &ZkIR, witness: &[F]) -> Result<(), ZkError> {
        let new_instance = self.compile_layer(layer_ir, witness)?;
        match &self.accumulator {
            None => self.accumulator = Some(Accumulator::initial(new_instance)),
            Some(acc) => {
                let challenge = self.generate_challenge(acc, &new_instance);
                self.accumulator = Some(acc.fold(new_instance, challenge)?);
            }
        }
        Ok(())
    }

    /// Finalize: produce the proof from the final accumulator.
    pub fn finalize(&self) -> Result<ZkProof, ZkError> { ... }
}
```

- [ ] **3.2** Create `folding/accumulator.rs`:
```rust
pub struct Accumulator<F: Field> {
    pub instance: Vec<F>,      // committed instance values
    pub witness: Vec<F>,       // witness values (private)
    pub error_term: F,         // relaxation error (Nova-style)
    pub num_folds: u32,
}

impl<F: Field> Accumulator<F> {
    pub fn fold(&self, new: Self, challenge: F) -> Result<Self, ZkError> {
        // Nova-style: acc' = acc + challenge * new
        // Instance: linear combination
        // Error term: updated via cross-terms
        ...
    }
}
```

- [ ] **3.3** Create `folding/sumcheck.rs` — univariate sumcheck for fast fold verification:
```rust
pub fn sumcheck_prove<F: Field>(
    polynomial: &[F],    // multilinear polynomial evaluations
    claim: F,            // claimed sum
) -> (Vec<Vec<F>>, F)   // (round_polynomials, final_evaluation)
```

- [ ] **3.4** Update `ZkBackend` trait in `backend.rs`:
```rust
pub trait FoldingBackend<F: Field> {
    fn fold_layer(&mut self, ir: &ZkIR, witness: &[F]) -> Result<(), ZkError>;
    fn finalize(&self) -> Result<ZkProof, ZkError>;
    fn verify(proof: &ZkProof, public_io: &[F]) -> Result<bool, ZkError>;
}
```

### Phase 4: Per-Layer Lowering for Folding (1-2 days)

- [ ] **4.1** Extend `lower.rs` to emit per-layer ZkIRs:
```rust
pub fn lower_model_for_folding(dag: &[ZkOp]) -> Vec<ZkIR> {
    // Group operations by layer boundaries (matmul + activation = one fold unit)
    let layer_groups = partition_by_layer(dag);
    layer_groups.iter().map(|group| lower_layer(group)).collect()
}
```

- [ ] **4.2** Layer boundary detection: matmul → activation → (optional norm) = one foldable unit.

- [ ] **4.3** Each `ZkIR` in the output is independently foldable — the prover processes them sequentially.

- [ ] **4.4** Test: a 6-layer MLP produces 6 ZkIRs (one per layer).

### Phase 5: Wire compile_zk() (1 day)

- [ ] **5.1** Rewrite `compile_zk()` in `mod.rs`:
```rust
pub fn compile_zk(
    dag: &[ZkOp],
    config: &ZkConfig,
    weights: Option<&WeightMap>,
) -> Result<ZkProof, ZkError> {
    let field = match config.field {
        ZkField::Mersenne31 => compile_zk_typed::<Mersenne31Field>(dag, config, weights),
        ZkField::BN254 => compile_zk_typed::<Bn254Field>(dag, config, weights),
    };
    field
}

fn compile_zk_typed<F: Field>(
    dag: &[ZkOp], config: &ZkConfig, weights: Option<&WeightMap>,
) -> Result<ZkProof, ZkError> {
    // 1. Lower DAG to per-layer ZkIRs
    let layer_irs = lower_model_for_folding(dag);

    // 2. Create folding prover
    let mut prover = FoldingProver::<F>::new(config);

    // 3. Fold each layer
    for (i, ir) in layer_irs.iter().enumerate() {
        let witness = generate_witness::<F>(ir, weights)?;
        prover.fold_layer(ir, &witness)?;
    }

    // 4. Finalize proof
    prover.finalize()
}
```

- [ ] **5.2** Update `@zk_proof` decorator handling to pass field and config to `compile_zk()`.

- [ ] **5.3** Update CLI `--zk-circuit` to accept `--zk-field m31|bn254` flag.

### Phase 6: Testing & Benchmarks (1-2 days)

- [ ] **6.1** Correctness: simple 2-layer MLP, verify proof passes verification.

- [ ] **6.2** Soundness: tamper with witness, verify proof FAILS verification.

- [ ] **6.3** Lookup correctness: ReLU lookup table produces correct field elements for all input values.

- [ ] **6.4** Folding: verify 12-layer transformer produces constant-size proof (not linear in layers).

- [ ] **6.5** Field benchmark: M31 prove time vs BN254 prove time (expect ~10x speedup).

- [ ] **6.6** Backwards compatibility: `ZkField::BN254` still works (for EVM Solidity verifiers).

---

## What's Kept vs. Deleted

**Kept (backend-agnostic, solid foundation):**
- `ir.rs` — ZkInstruction format (9 instruction types work for any backend)
- `lookup.rs` — 8 activation lookup tables (integrate with lookup-native)
- `lower.rs` — 5-pass pipeline (extend for per-layer emission)
- `witness.rs` — Arena-allocated witness generator (extend for accumulator)
- `plonky3/mod.rs` — Stub (low priority, leave for future)

**Deleted (Halo2 v1 scaffolding):**
- `halo2/mod.rs`, `halo2/gates.rs`, `halo2/layout.rs` — PLONKish architecture
- `halo2/prove.rs`, `halo2/verify.rs` — Trivially forgeable v1 proof
- `halo2/solidity.rs` — Non-functional Solidity verifier

**Created (new folding + AIR architecture):**
- `field_m31.rs` — Mersenne-31 field (~100x faster than BN254)
- `air.rs` — AIR constraint format (replaces PLONKish)
- `lookup_native.rs` — Jolt-style lookup gates
- `folding/mod.rs` — Folding prover
- `folding/accumulator.rs` — Accumulator state
- `folding/sumcheck.rs` — Sumcheck argument

---

## Effort Estimate

- Phase 1 (field abstraction + M31): 2 days
- Phase 2 (delete Halo2, add AIR): 2 days
- Phase 3 (folding backend): 3-4 days
- Phase 4 (per-layer lowering): 1-2 days
- Phase 5 (wire compile_zk): 1 day
- Phase 6 (testing): 1-2 days
- Total: **10-13 days**
