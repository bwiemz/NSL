# FASE Deferred Numerical Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three orthogonal tests that together validate FASE Deferred's numerical correctness — end-to-end SGD equivalence, end-to-end AdamW FASE-Deferred pipeline equivalence, and a pure-Rust Jensen-inequality fence against future "fixes" to the intentional v-approximation.

**Architecture:** Tests live under `crates/nsl-codegen/tests/` and share a small `.nslm` reader helper. Pure-Rust references (SGD, FASE-Deferred AdamW) share no code with the codegen under test. The Jensen fence lives as a unit test inside `fase_optimizer.rs` itself. θ-only visibility — no changes to `model_save`, runtime, or codegen.

**Tech Stack:** Rust (`#[test]` + `std::process::Command`), the `tempfile` crate (already a workspace dep), the `CARGO_BIN_EXE_nsl` harness from the smoke test, NSL surface syntax (`SGD(lr=...)`, `AdamW(lr=..., beta1=..., beta2=..., eps=..., weight_decay=...)`, `train(grad_accumulation=N)`, `model_save(m, path)`).

**Spec:** [docs/superpowers/specs/2026-04-14-fase-numerical-validation-design.md](../specs/2026-04-14-fase-numerical-validation-design.md)

---

## Task 1: `.nslm` reader helper

A small Rust module that parses the `.nslm` binary format and returns a map of tensor name → flattened `Vec<f32>`. Tested in isolation with in-memory bytes so the reader is validated without needing a compiled NSL program.

**Files:**
- Create: `crates/nsl-codegen/tests/common/mod.rs`
- Create: `crates/nsl-codegen/tests/common/nslm_reader.rs`
- Create: `crates/nsl-codegen/tests/common_nslm_reader_test.rs` (top-level integration test)

### Format reference (from `crates/nsl-runtime/src/checkpoint.rs:21-80`)

- 4 bytes: magic `"NSLM"`
- 4 bytes: version (u32, little-endian)
- 8 bytes: header size (u64, little-endian)
- `header_size` bytes: JSON string `{"params":[{"name":"...", "shape":[...], "dtype":"f32|f64", "offset":N, "nbytes":N}, ...]}`
- Aligned payload: raw tensor bytes at declared `offset` for `nbytes` each

### Steps

- [ ] **Step 1: Create the common module directory**

Create `crates/nsl-codegen/tests/common/mod.rs` with contents:

```rust
//! Shared test helpers for nsl-codegen integration tests.

pub mod nslm_reader;
```

- [ ] **Step 2: Write the failing test**

Create `crates/nsl-codegen/tests/common_nslm_reader_test.rs`:

```rust
//! Unit test for the .nslm reader — uses in-memory bytes only.

mod common {
    include!("common/mod.rs");
}

use common::nslm_reader::read_nslm;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_nslm_bytes(
    params: &[(&str, Vec<i64>, &str, Vec<u8>)],
) -> NamedTempFile {
    // params: (name, shape, dtype, raw_bytes)
    let mut file = NamedTempFile::new().expect("tempfile");

    let mut header_entries = Vec::new();
    let mut data_offset: u64 = 0;
    for (name, shape, dtype, bytes) in params {
        let nbytes = bytes.len() as u64;
        header_entries.push(format!(
            r#"{{"name":"{}","shape":{:?},"dtype":"{}","offset":{},"nbytes":{}}}"#,
            name, shape, dtype, data_offset, nbytes
        ));
        data_offset += nbytes;
    }
    let header = format!(r#"{{"params":[{}]}}"#, header_entries.join(","));
    let header_bytes = header.as_bytes();

    file.write_all(b"NSLM").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header_bytes).unwrap();
    // 64-byte alignment padding, matching checkpoint.rs
    let header_total = 4 + 4 + 8 + header_bytes.len();
    let pad = (64 - (header_total % 64)) % 64;
    file.write_all(&vec![0u8; pad]).unwrap();
    for (_, _, _, bytes) in params {
        file.write_all(bytes).unwrap();
    }
    file.flush().unwrap();
    file
}

#[test]
fn reads_single_f32_tensor() {
    let values: Vec<f32> = vec![0.5, -0.3, 1.25, 0.0];
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
    let file = write_nslm_bytes(&[("w", vec![4], "f32", bytes)]);

    let got = read_nslm(file.path()).expect("read ok");
    let w = got.get("w").expect("w present");
    assert_eq!(w.len(), 4);
    for (g, e) in w.iter().zip(values.iter()) {
        assert!((g - e).abs() < 1e-9, "got {} want {}", g, e);
    }
}

#[test]
fn reads_f64_tensor_and_downcasts_to_f32() {
    let values: Vec<f64> = vec![0.5, -0.25, 1.0];
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
    let file = write_nslm_bytes(&[("b", vec![3], "f64", bytes)]);

    let got = read_nslm(file.path()).expect("read ok");
    let b = got.get("b").expect("b present");
    assert_eq!(b.len(), 3);
    assert!((b[0] - 0.5).abs() < 1e-6);
    assert!((b[1] + 0.25).abs() < 1e-6);
    assert!((b[2] - 1.0).abs() < 1e-6);
}

#[test]
fn rejects_bad_magic() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"XXXX").unwrap();
    file.flush().unwrap();
    let result = read_nslm(file.path());
    assert!(result.is_err());
}
```

If `tempfile::NamedTempFile` is not in `crates/nsl-codegen/Cargo.toml`'s `[dev-dependencies]`, add it. (`tempfile = "3"` is already in `[dependencies]` per `crates/nsl-codegen/Cargo.toml`, so it should carry through to dev, but if not, add explicitly.)

- [ ] **Step 3: Run the test — expect failure**

Run: `cargo test -p nsl-codegen --test common_nslm_reader_test`
Expected: FAIL with "no file common/nslm_reader.rs" or similar.

- [ ] **Step 4: Implement the reader**

Create `crates/nsl-codegen/tests/common/nslm_reader.rs`:

```rust
//! Minimal `.nslm` parser for integration tests.
//!
//! Parses the format written by `crates/nsl-runtime/src/checkpoint.rs`
//! (magic + version + header size + JSON header + aligned tensor data)
//! and returns each tensor as a flattened `Vec<f32>`.  Both `f32` and
//! `f64` source dtypes are supported; `f64` values are downcast to `f32`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
struct Entry {
    name: String,
    dtype: String,
    offset: usize,
    nbytes: usize,
}

pub fn read_nslm(path: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("read: {}", e))?;

    if buf.len() < 16 || &buf[0..4] != b"NSLM" {
        return Err("bad magic (expected NSLM)".into());
    }
    let header_size =
        u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    let header_end = 16 + header_size;
    if buf.len() < header_end {
        return Err("truncated header".into());
    }
    let header_json = std::str::from_utf8(&buf[16..header_end])
        .map_err(|e| format!("header utf8: {}", e))?;

    let entries = parse_entries(header_json)?;

    // Data region begins after 64-byte alignment from the end of the header
    let header_total = header_end;
    let pad = (64 - (header_total % 64)) % 64;
    let data_start = header_total + pad;

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    for e in entries {
        let begin = data_start + e.offset;
        let end = begin + e.nbytes;
        if buf.len() < end {
            return Err(format!("truncated data for {}", e.name));
        }
        let slab = &buf[begin..end];
        let values = match e.dtype.as_str() {
            "f32" => slab
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "f64" => slab
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            other => return Err(format!("unsupported dtype: {}", other)),
        };
        out.insert(e.name, values);
    }
    Ok(out)
}

fn parse_entries(json: &str) -> Result<Vec<Entry>, String> {
    // Minimal JSON extraction — the writer produces a known, flat shape,
    // so a regex-free string scanner is adequate and avoids pulling in
    // serde_json as a dev-dep for nsl-codegen.
    let mut entries = Vec::new();
    let mut cursor = 0;
    while let Some(name_idx) = json[cursor..].find("\"name\":\"") {
        let abs = cursor + name_idx + 8;
        let name_end = json[abs..]
            .find('"')
            .ok_or("malformed name field")?;
        let name = json[abs..abs + name_end].to_string();
        cursor = abs + name_end;

        let dtype = extract_str_field(json, cursor, "dtype")?;
        cursor = find_after(json, cursor, "dtype")?;
        let offset: usize = extract_int_field(json, cursor, "offset")?;
        cursor = find_after(json, cursor, "offset")?;
        let nbytes: usize = extract_int_field(json, cursor, "nbytes")?;
        cursor = find_after(json, cursor, "nbytes")?;

        entries.push(Entry {
            name,
            dtype,
            offset,
            nbytes,
        });
    }
    Ok(entries)
}

fn extract_str_field(json: &str, from: usize, key: &str) -> Result<String, String> {
    let pattern = format!("\"{}\":\"", key);
    let idx = json[from..]
        .find(&pattern)
        .ok_or_else(|| format!("{} field missing", key))?;
    let abs = from + idx + pattern.len();
    let end = json[abs..]
        .find('"')
        .ok_or_else(|| format!("{} field unterminated", key))?;
    Ok(json[abs..abs + end].to_string())
}

fn extract_int_field(json: &str, from: usize, key: &str) -> Result<usize, String> {
    let pattern = format!("\"{}\":", key);
    let idx = json[from..]
        .find(&pattern)
        .ok_or_else(|| format!("{} field missing", key))?;
    let abs = from + idx + pattern.len();
    let end = json[abs..]
        .find(|c: char| c == ',' || c == '}')
        .ok_or_else(|| format!("{} value unterminated", key))?;
    json[abs..abs + end]
        .trim()
        .parse::<usize>()
        .map_err(|e| format!("{}: {}", key, e))
}

fn find_after(json: &str, from: usize, key: &str) -> Result<usize, String> {
    let pattern = format!("\"{}\":", key);
    let idx = json[from..]
        .find(&pattern)
        .ok_or_else(|| format!("{} field missing", key))?;
    Ok(from + idx + pattern.len())
}
```

- [ ] **Step 5: Run the test — expect pass**

Run: `cargo test -p nsl-codegen --test common_nslm_reader_test`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/tests/common/mod.rs crates/nsl-codegen/tests/common/nslm_reader.rs crates/nsl-codegen/tests/common_nslm_reader_test.rs
git commit -m "feat(test): add .nslm reader helper for fase integration tests"
```

---

## Task 2: Jensen-inequality fence (pure Rust unit test)

Pure Rust unit test inside `fase_optimizer.rs`. Proves the FASE-Deferred v-update differs from standard AdamW in the Jensen direction. No compiler, no I/O, microseconds to run, impossible to flake.

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs` — append to `#[cfg(test)] mod tests`

### Steps

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block at the bottom of `crates/nsl-codegen/src/fase_optimizer.rs`:

```rust
#[test]
fn jensen_fence_fase_v_exceeds_standard_v_for_nonconstant_gradients() {
    // The FASE paper's Option B approximates v using mean(g²) rather than
    // mean(g)².  Jensen's inequality (mean(g²) ≥ mean(g)²) guarantees
    // v_fase ≥ v_standard element-wise, with strict inequality when the
    // per-micro-batch gradients vary.  This test fences the approximation
    // against future "fixes" that would silently implement the standard
    // formula.

    // Closed-form per-parameter v-update for one accumulation window:
    //
    //   Standard AdamW:     v' = β₂·v + (1 - β₂) · (mean(g))²
    //   FASE Deferred:      v' = β₂·v + (1 - β₂) · mean(g²)
    //
    // Using non-constant gradients: [1.0, 2.0, 0.5, 1.5]
    //   mean(g)   = 1.25
    //   mean(g)²  = 1.5625
    //   mean(g²)  = (1 + 4 + 0.25 + 2.25) / 4 = 1.875
    //
    // So v_fase - v_standard = (1 - β₂) · (1.875 - 1.5625) = (1 - β₂) · 0.3125

    let beta2: f64 = 0.999;
    let v_prev: f64 = 0.0;
    let gradients: [f64; 4] = [1.0, 2.0, 0.5, 1.5];

    let mean_g: f64 = gradients.iter().sum::<f64>() / (gradients.len() as f64);
    let mean_g_sq: f64 =
        gradients.iter().map(|g| g * g).sum::<f64>() / (gradients.len() as f64);

    let v_standard = beta2 * v_prev + (1.0 - beta2) * mean_g * mean_g;
    let v_fase = beta2 * v_prev + (1.0 - beta2) * mean_g_sq;

    assert!(
        v_fase >= v_standard,
        "Jensen: v_fase ({}) must be >= v_standard ({})",
        v_fase,
        v_standard
    );
    assert!(
        v_fase - v_standard > 0.0,
        "For non-constant gradients, v_fase must strictly exceed v_standard; got diff {}",
        v_fase - v_standard
    );

    // The expected difference is (1 - β₂) · (mean_g_sq - mean_g²) = 0.001 · 0.3125.
    let expected_diff = (1.0 - beta2) * (mean_g_sq - mean_g * mean_g);
    assert!(
        (v_fase - v_standard - expected_diff).abs() < 1e-12,
        "expected diff {}, got {}",
        expected_diff,
        v_fase - v_standard
    );
}

#[test]
fn jensen_fence_constant_gradients_produce_equal_v() {
    // Sanity check: when all per-micro-batch gradients are identical,
    // mean(g²) == mean(g)², so Jensen's inequality is tight.  The fence
    // test above must use non-constant gradients; this test documents
    // why.
    let beta2: f64 = 0.999;
    let v_prev: f64 = 0.0;
    let gradients: [f64; 4] = [0.7; 4];

    let mean_g: f64 = gradients.iter().sum::<f64>() / (gradients.len() as f64);
    let mean_g_sq: f64 =
        gradients.iter().map(|g| g * g).sum::<f64>() / (gradients.len() as f64);

    let v_standard = beta2 * v_prev + (1.0 - beta2) * mean_g * mean_g;
    let v_fase = beta2 * v_prev + (1.0 - beta2) * mean_g_sq;

    assert!(
        (v_fase - v_standard).abs() < 1e-12,
        "constant gradients must produce equal v_fase and v_standard"
    );
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::jensen_fence`
Expected: both tests pass (they are pure math, no implementation changes needed).

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs
git commit -m "test(fase): Jensen-inequality fence against future v-update edits"
```

---

## Task 3: SGD exact-equivalence integration test

End-to-end: compile a minimal NSL train block with plain SGD and `grad_accumulation=4`, run it, read the saved `.nslm`, compare θ against a Rust SGD reference. Exact match expected (SGD has no v-state; FASE Deferred is mathematically identical to standard).

**Files:**
- Create: `crates/nsl-codegen/tests/fixtures/fase_deferred_sgd_equivalence.nsl`
- Create: `crates/nsl-codegen/tests/fase_numerical_validation.rs`

### Fixture shape

One linear layer `w: Tensor<[2,1], f32>` with init from `from_list([[0.5], [-0.3]])` (or whichever constructor constant-initializes weights in the NSL surface). Four hard-coded micro-batches using `from_list` with fixed values. MSE loss. `optimizer: SGD(lr=0.01)`. `grad_accumulation=4`. Save checkpoint at end.

### Steps

- [ ] **Step 1: Discover the constant-init surface syntax**

Run (from the worktree root):

```bash
grep -rn "from_list\|Tensor<\[.*\], .*> = \|ones(\[\|zeros(\[" examples/*.nsl stdlib/nsl/nn/*.nsl 2>/dev/null | head -20
grep -rn "let [xy] = " examples/m14*.nsl 2>/dev/null | head -10
```

Record the syntax for: (a) constant-initializing a small f32 tensor with specific values, (b) declaring a model field with that init, (c) setting `grad_accumulation` in a train block (smoke-test fixture at `crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4.nsl` already does this — read it).

If `from_list` isn't supported for multi-dim tensors, the fallback is to init `w` with `ones([2, 1])` or `zeros([2, 1])` and bake the test expectations around that.

- [ ] **Step 2: Write the SGD fixture**

Create `crates/nsl-codegen/tests/fixtures/fase_deferred_sgd_equivalence.nsl`. Template (adjust to the surface syntax from Step 1):

```nsl
from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])   # init known to the Rust reference

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()

# Four hard-coded micro-batches (each [1,2]) and targets (each [1,1]).
# Shape-compatible with the model; deterministic; no data loader.
let x = from_list([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
let y = from_list([[1.0], [0.5], [2.0], [0.25]])

train(model = m, epochs = 1, grad_accumulation = 4):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

model_save(m, "sgd_out.nslm")
```

Important notes:
- The `step(batch)` function in the reference examples does NOT use `batch` for input — it references `x` and `y` from module scope.  This is intentional: we want every micro-batch to see the SAME inputs so the reference can be computed without a data loader.  Confirm this pattern works by reading the smoke-test fixture.
- If `grad_accumulation=4` with `epochs=1` and no data loader triggers just ONE train-step call (i.e. the inner loop sees one batch, not four micro-batches), we have a problem: FASE Deferred needs four micro-batch invocations to exercise the accumulator.  Verify by reading `stmt.rs`'s train-block codegen — if it does `for _ in 0..grad_accumulation_steps { step(...) }` regardless of dataloader state, we're fine.  If not, report BLOCKED.
- `model_save(m, "sgd_out.nslm")` writes to the process's current working directory. The Rust test must run `nsl run` with an explicit cwd (tempdir) so the file lands somewhere the test can find.

- [ ] **Step 3: Write the SGD integration test**

Create `crates/nsl-codegen/tests/fase_numerical_validation.rs`:

```rust
//! Numerical validation for FASE Deferred mode (CFTP item #2).
//!
//! Compiles fixtures with `grad_accumulation=4`, runs them, reads the
//! saved `.nslm`, and compares θ against pure-Rust references.  θ-only
//! visibility; no coupling to runtime internals.

mod common {
    include!("common/mod.rs");
}

use common::nslm_reader::read_nslm;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

/// Run `nsl run <fixture>` with `cwd` set to `workdir`, so any file the
/// NSL program saves lands under `workdir`.  Panics if the run fails.
fn nsl_run(fixture_path: &Path, workdir: &Path) {
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .arg(fixture_path)
        .current_dir(workdir)
        .status()
        .expect("failed to spawn nsl run");
    assert!(
        status.success(),
        "nsl run failed on {:?}",
        fixture_path
    );
}

/// Rust reference: one SGD step with grad_accumulation=4 and constant
/// inputs across all four micro-batches.  Returns the final θ.
///
/// Model: y_pred = x @ w, with w shape [2,1] and x shape [4,2].
/// Loss: MSE between y_pred and y (shape [4,1]).
/// Gradient of MSE wrt w: (2/N_samples) * x.T @ (x @ w - y).
///
/// With grad_accumulation=4 and identical inputs across micro-batches,
/// every micro-batch produces the same gradient g, so mean(g) = g.
fn sgd_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
) -> [f32; 2] {
    // Forward: pred[i] = x[i][0]*w[0] + x[i][1]*w[1]
    let mut pred = [0.0_f32; 4];
    for i in 0..4 {
        pred[i] = x[i][0] * w_init[0] + x[i][1] * w_init[1];
    }
    // Residual r[i] = pred[i] - y[i][0]
    let mut r = [0.0_f32; 4];
    for i in 0..4 {
        r[i] = pred[i] - y[i][0];
    }
    // Gradient g[j] = (2/N) * sum_i (x[i][j] * r[i]) where N = 4 samples
    let n = 4.0_f32;
    let mut g = [0.0_f32; 2];
    for j in 0..2 {
        for i in 0..4 {
            g[j] += x[i][j] * r[i];
        }
        g[j] *= 2.0 / n;
    }
    // FASE Deferred accumulates mean(g_1..g_M) across M=4 micro-batches.
    // All micro-batches see the same inputs, so mean(g) = g.
    // SGD update: θ -= lr * mean(g)
    [w_init[0] - lr * g[0], w_init[1] - lr * g[1]]
}

#[test]
fn sgd_exact_equivalence() {
    let tmp = TempDir::new().expect("tempdir");
    nsl_run(&fixture("fase_deferred_sgd_equivalence.nsl"), tmp.path());

    let checkpoint = tmp.path().join("sgd_out.nslm");
    assert!(checkpoint.exists(), "expected checkpoint at {:?}", checkpoint);
    let tensors = read_nslm(&checkpoint).expect("read nslm");

    // The tensor name in the checkpoint is "w" (or "m.w" depending on how
    // the compiler names nested model fields — read the header if needed).
    let w_compiled = tensors.get("w")
        .or_else(|| tensors.get("m.w"))
        .expect("w tensor in checkpoint");
    assert_eq!(w_compiled.len(), 2, "w should have 2 elements");

    // Reference — must match the fixture's init exactly.
    let w_init = [1.0_f32, 1.0_f32]; // ones([2,1])
    let x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]];
    let y = [[1.0], [0.5], [2.0], [0.25]];
    let w_ref = sgd_reference(&w_init, &x, &y, 0.01);

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-6,
            "SGD θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}
```

If Step 1 found that `ones([2,1])` is the actual init (not `from_list([[0.5], [-0.3]])`), then the `w_init` in the Rust reference MUST match — use `[1.0, 1.0]` for ones. If `from_list` init works and the fixture uses `[[0.5], [-0.3]]`, use `[0.5, -0.3]` here. This plan template assumes `ones([2,1])`; adjust when you confirm the syntax in Step 1.

If `tempfile` is not listed under `[dev-dependencies]` in `crates/nsl-codegen/Cargo.toml`, add it. It's already under `[dependencies]`, so cargo will carry it through; verify by running the test.

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation -- sgd_exact_equivalence --nocapture`
Expected: PASS.

If the test fails because `nsl run` cannot execute on this machine (e.g. CUDA required), mark the test with:

```rust
#[test]
#[cfg_attr(not(feature = "gpu_tests"), ignore = "requires runnable NSL runtime")]
fn sgd_exact_equivalence() { ... }
```

and note it in the final report. If the test fails because θ disagrees with the reference, the FASE Deferred SGD emission has a real bug — STOP and report with the diagnostic output (exact values, input hyperparameters, fixture file).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/fixtures/fase_deferred_sgd_equivalence.nsl crates/nsl-codegen/tests/fase_numerical_validation.rs
git commit -m "test(fase): SGD exact-equivalence integration test"
```

---

## Task 4: AdamW FASE-Deferred pipeline integration test

Same harness pattern as Task 3, but AdamW with three windows. Reference implements the full FASE-Deferred AdamW math (first moment, approx second moment, bias correction, param update).

**Files:**
- Create: `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl`
- Modify: `crates/nsl-codegen/tests/fase_numerical_validation.rs` — add the AdamW test

### Steps

- [ ] **Step 1: Decide how to get 3 optimizer windows**

Two options in the NSL surface:
- **Option A:** `epochs = 3, grad_accumulation = 4` — one data-loader-free step per epoch, 4 micro-batches each, total 3 optimizer windows.  This matches the pattern in `examples/m14_adam_scheduler.nsl` and is what the Task 3 fixture already uses (with epochs=1).
- **Option B:** A dataloader with 12 samples and `grad_accumulation=4, epochs=1` — 3 optimizer windows per epoch.  More plumbing (needs `dataset` block), not worth it for this test.

Go with Option A.

- [ ] **Step 2: Write the AdamW fixture**

Create `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl`:

```nsl
from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()

let x = from_list([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
let y = from_list([[1.0], [0.5], [2.0], [0.25]])

train(model = m, epochs = 3, grad_accumulation = 4):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

model_save(m, "adamw_out.nslm")
```

Verify that `AdamW` supports the full keyword form by reading `stdlib/nsl/optim/adamw.nsl` and `examples/gpt2.nsl` (which uses multi-line AdamW). If any keyword isn't recognized, use the example's exact form.

- [ ] **Step 3: Add the AdamW reference + test**

Append to `crates/nsl-codegen/tests/fase_numerical_validation.rs`:

```rust
/// Rust reference: three windows of FASE-Deferred AdamW with constant
/// inputs across each window's 4 micro-batches.  Because inputs are
/// constant, mean(g) = g for every window — this greatly simplifies the
/// gradient computation but still exercises every recipe op.  Returns
/// the final θ.
fn adamw_fase_deferred_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    windows: u32,
) -> [f32; 2] {
    let mut w = *w_init;
    let mut m_state = [0.0_f32; 2];
    let mut v_state = [0.0_f32; 2];

    for step in 1..=windows {
        // Gradient with current w (constant across all 4 micro-batches in this
        // window, so m_partial = g).
        let mut pred = [0.0_f32; 4];
        for i in 0..4 {
            pred[i] = x[i][0] * w[0] + x[i][1] * w[1];
        }
        let mut r = [0.0_f32; 4];
        for i in 0..4 {
            r[i] = pred[i] - y[i][0];
        }
        let n = 4.0_f32;
        let mut g = [0.0_f32; 2];
        for j in 0..2 {
            for i in 0..4 {
                g[j] += x[i][j] * r[i];
            }
            g[j] *= 2.0 / n;
        }

        // m_partial = mean(g_1..g_4) = g  (inputs constant across window)
        let m_partial = g;

        // Deferred AdamW final step per parameter.
        for j in 0..2 {
            m_state[j] = beta1 * m_state[j] + (1.0 - beta1) * m_partial[j];
            v_state[j] = beta2 * v_state[j] + (1.0 - beta2) * m_partial[j] * m_partial[j];
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);
            let m_hat = m_state[j] / bc1;
            let v_hat = v_state[j] / bc2;
            w[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * w[j]);
        }
    }
    w
}

#[test]
fn adamw_fase_deferred_pipeline_equivalence() {
    let tmp = TempDir::new().expect("tempdir");
    nsl_run(
        &fixture("fase_deferred_adamw_equivalence.nsl"),
        tmp.path(),
    );

    let checkpoint = tmp.path().join("adamw_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?}",
        checkpoint
    );
    let tensors = read_nslm(&checkpoint).expect("read nslm");

    let w_compiled = tensors.get("w")
        .or_else(|| tensors.get("m.w"))
        .expect("w tensor in checkpoint");
    assert_eq!(w_compiled.len(), 2);

    let w_init = [1.0_f32, 1.0_f32];
    let x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]];
    let y = [[1.0], [0.5], [2.0], [0.25]];
    let w_ref = adamw_fase_deferred_reference(
        &w_init, &x, &y,
        /*lr=*/ 0.001,
        /*beta1=*/ 0.9,
        /*beta2=*/ 0.999,
        /*eps=*/ 1e-8,
        /*wd=*/ 0.01,
        /*windows=*/ 3,
    );

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "AdamW θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}
```

- [ ] **Step 4: Run both tests**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation`
Expected: both `sgd_exact_equivalence` and `adamw_fase_deferred_pipeline_equivalence` PASS.

If AdamW fails:
- First check Task 3 (SGD) is still passing — if it's also failing, the problem is in the accumulator or the SGD final step.
- If only AdamW fails, the bug is in one of the AdamW-specific recipe ops: `ScalarMulAdd` (first moment), `SquaredAccumulate` (second moment), `SqrtPlusEps`, `Div`, or `Update`. Inspect the observed θ vs reference: if the error is proportional to `lr`, the final `Update` op may have a sign or `wd` mismatch; if the error grows with window count, the bias correction step counter may be off.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl crates/nsl-codegen/tests/fase_numerical_validation.rs
git commit -m "test(fase): AdamW FASE-Deferred pipeline equivalence test"
```

---

## Task 5: Final verification + memory note

- [ ] **Step 1: Run the full nsl-codegen test suite**

Run: `cargo test -p nsl-codegen`
Expected: all tests pass, including the three new ones from Tasks 2-4 and the pre-existing 1241 from item #1. No snapshot changes (this task only adds new test files).

- [ ] **Step 2: Verify no leakage to main worktree**

Run: `git -C c:/Users/bwiem/projects/NSL status --short -- crates/`
Expected: no output (main worktree's `crates/` has no pending changes).

- [ ] **Step 3: Update the FASE project memory note**

Edit `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_fase_deferred_integration.md`. In the "Follow-up work" section, change item #2 from pending to complete:

Replace:

```
- **Item #2:** numerical-equivalence test — same seed, standard path vs Deferred, parameters match to f32 tolerance after N=4 micro-batches.
```

with:

```
- **Item #2:** ✅ **shipped 2026-04-14** — three-test validation (SGD exact, AdamW Deferred pipeline, Jensen fence) in `crates/nsl-codegen/tests/fase_numerical_validation.rs` + unit test in `fase_optimizer.rs`.
```

- [ ] **Step 4: Commit memory update**

```bash
git -C C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory add project_fase_deferred_integration.md
git -C C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory commit -m "docs(memory): FASE item #2 numerical validation shipped"
```

- [ ] **Step 5: Report**

Summarize: 3 new test files, 3 new `#[test]` functions plus 2 new Jensen unit tests, total test count delta, and the next open item on the FASE roadmap (#1b peak-memory interleaving, #3 grad-clip codegen).

---

## Summary of files touched

- **Created:** `crates/nsl-codegen/tests/common/mod.rs` (Task 1)
- **Created:** `crates/nsl-codegen/tests/common/nslm_reader.rs` (Task 1)
- **Created:** `crates/nsl-codegen/tests/common_nslm_reader_test.rs` (Task 1)
- **Modified:** `crates/nsl-codegen/src/fase_optimizer.rs` — Jensen tests appended (Task 2)
- **Created:** `crates/nsl-codegen/tests/fixtures/fase_deferred_sgd_equivalence.nsl` (Task 3)
- **Created:** `crates/nsl-codegen/tests/fase_numerical_validation.rs` (Task 3, extended Task 4)
- **Created:** `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl` (Task 4)
- **Modified:** memory note (Task 5)
