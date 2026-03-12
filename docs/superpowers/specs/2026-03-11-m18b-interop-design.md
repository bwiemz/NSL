# M18b Interop Design Specification

**Goal:** Enable NeuralScript to load pretrained HuggingFace models, serialize weights in safetensors format, and export inference graphs to ONNX — all in pure Rust, no Python dependency.

**Scope:** Safetensors read/write, HuggingFace Hub weight loading, trace-based ONNX export, `nsl export` CLI command.

**Deferred to M18c:** py.call FFI (CPython embedding), DLPack zero-copy tensor exchange, from_torch/to_torch bidirectional bridging.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python dependency | None | Keeps NSL's "pure Rust+NSL" architecture. HF loading via `hf-hub` crate. |
| Weight mapping | Convention-based, explicit | User defines NSL model, loads weights via name mapping with optional overrides |
| ONNX export | Trace-based | Run model with dummy input, capture op trace, emit ONNX graph. Covers inference models. |
| Serialization format | Dual (.nslm + safetensors) | .nslm stays native, safetensors is the interop bridge format |
| CLI export | `export_model()` entry point | User writes dummy input construction in NSL, full type safety, multi-input support |
| Interop deps | Feature-gated | `interop` Cargo feature gates heavy deps (networking, protobuf). Core builds unaffected. |

---

## Section 1: Safetensors Read/Write

### Overview

Read and write the safetensors format — a simple, safe binary format for storing tensors. No arbitrary code execution risk. This is the foundation that HuggingFace loading builds on.

### Rust Implementation (`crates/nsl-runtime/src/safetensors.rs`)

**`nsl_safetensors_load(path_ptr: i64, device: i64) -> i64` (returns NslDict pointer)**

- Memory-map the `.safetensors` file via the `safetensors` crate (zero-copy view of disk data)
- Iterate over tensors in the file
- For each tensor:
  - If `device == 0` (CPU): allocate f32 buffer, convert f16/bf16 source data to f32 using the `half` crate
  - If `device > 0` (GPU): allocate f32 buffer via `cuMemAllocManaged`, convert and write directly to unified memory
- Construct an `NslTensor` (dtype=1/f32, device as specified) for each tensor
- Return an `NslDict<String, NslTensor>` with parameter names as keys

**Critical: No f64 inflation.** CPU loads as f32 (not f64). A 14GB f16 model inflates to 28GB f32, not 56GB f64. The f64 dtype is for NSL's compute tensors, not weight storage.

**`nsl_safetensors_save(dict_ptr: i64, path_ptr: i64)`**

- Iterate dict entries, collect tensor names and data
- GPU tensors (already f32) = direct byte copy to file buffer
- CPU f64 tensors = downcast to f32 on write
- Serialize via `safetensors` crate's writer

### NSL Stdlib API (`stdlib/nsl/compat.nsl`)

```
fn load_safetensors(path: str, device: str = "cpu") -> Dict[str, Tensor]
fn save_safetensors(weights: Dict[str, Tensor], path: str)
```

### Memory Flow (7B model, 14GB f16 on disk)

```
disk (f16, 14GB) --> mmap (zero-copy) --> f32 allocation (28GB, CPU or GPU)
                                          No intermediate f64 buffer
```

---

## Section 2: HuggingFace Weight Loading

### Overview

Download a model from HuggingFace Hub and load its safetensors weights into a user-defined NSL model using convention-based name mapping. The compiler generates a static offset table — no runtime reflection.

### The AOT Constraint

At runtime, `model_ptr` is a raw block of bytes. The runtime has no knowledge of field names, nesting, or boundaries. The compiler must generate the metadata table at compile time and pass it to the runtime FFI.

### Codegen (`crates/nsl-codegen/src/expr.rs`)

When the compiler sees `from_hf("gpt2", m, device="cuda")`:

1. Resolve `m`'s model type at compile time
2. Recursively walk the model's field tree, generating entries:
   ```
   ("layers[0].attention.q_proj.weight", byte_offset=256, shape=[768,768], transpose=true)
   ```
3. Emit the table as a static byte array in `.rodata`
4. Emit a call to `nsl_hf_load(model_ptr, metadata_ptr, metadata_len, repo_id_ptr, device)`

### Rust FFI (`crates/nsl-runtime/src/huggingface.rs`)

```rust
#[repr(C)]
struct ParamMeta {
    name: *const c_char,    // "layers[0].attention.q_proj.weight"
    offset: usize,          // byte offset into model struct
    shape: *const i64,      // expected shape
    ndim: i64,
    transpose: bool,        // explicit, not guessed from shape
}

// Downloads from HF Hub, loads safetensors, maps weights into model struct
nsl_hf_load(
    model_ptr: i64,
    metadata_ptr: i64,      // pointer to [ParamMeta; N]
    metadata_len: i64,
    repo_id_ptr: i64,
    device: i64,
) -> i64  // 0 = success, nonzero = error code
```

**Flow:**
1. `hf-hub` crate downloads safetensors + config.json to `~/.cache/huggingface/` (cached on repeat loads)
2. `load_safetensors` mmap's files into `Dict[str, Tensor]` in f32
3. For each `ParamMeta` entry:
   - Find matching HF weight by name mapping
   - Validate shape (post-transpose)
   - Transpose if flagged
   - `memcpy` into `model_ptr + offset`
4. Error if any NSL param has no matching weight
5. Warn if any HF weight is unused

### Transpose Rules (Compile-Time, Not Shape-Guessed)

The "Square Matrix Trap": auto-detecting transpose from shape mismatch fails for square matrices like `[768, 768]` attention projections. Instead, transpose is determined by layer type:

| Layer Type | Transpose? | Reason |
|-----------|-----------|--------|
| `nn.Linear.weight` | Yes | PyTorch `[out, in]` -> NSL `[in, out]` |
| `nn.Embedding.weight` | No | Same layout |
| `nn.LayerNorm.weight/bias` | No | 1D, no transpose needed |

### Name Mapping Conventions (Compile-Time)

| HuggingFace | NSL |
|-------------|-----|
| `self_attn` | `attention` |
| `mlp.gate_proj` | `ffn.gate` |
| `embed_tokens` | `embedding` |
| `lm_head` | `head` |

User can override with an explicit mapping dict:
```
let m = from_hf("gpt2", m, mapping={"model.embed_tokens.weight": "embedding.weight"})
```

### NSL Stdlib API

```
fn from_hf(repo_id: str, model: Model, device: str = "cpu") -> Model
fn from_hf(repo_id: str, model: Model, device: str = "cpu",
           mapping: Dict[str, str] = {}) -> Model
```

### No `hf_config()` Function

NeuralScript is strictly typed — there is no `any` type, and `NslDict` maps `String -> NslTensor`. Parsing heterogeneous JSON (strings, booleans, nested objects) into NSL types is not viable. Users inspect config.json on the HF Hub website and construct their NSL model with the correct hyperparameters.

---

## Section 3: Trace-Based ONNX Export

### Overview

Run an NSL model with dummy input, capture the sequence of tensor operations, and serialize the trace as an ONNX graph for deployment to inference engines (ONNX Runtime, TensorRT, etc.).

### Phase 1: Tracing Runtime (`crates/nsl-runtime/src/trace.rs`)

The tracer tracks tensor pointers through the computation:

```rust
struct TraceGraph {
    ops: Vec<TraceOp>,
    inputs: Vec<(i64, String)>,              // (tensor_ptr, name)
    outputs: Vec<(i64, String)>,             // (tensor_ptr, name)
    ptr_to_node: HashMap<i64, TraceNodeId>,  // maps tensor_ptr -> producing op
}

struct TraceOp {
    op_type: OpType,              // MatMul, Add, Relu, Softmax, etc.
    inputs: Vec<TraceNodeId>,     // references to prior ops or graph inputs
    output_ptr: i64,              // pointer to output tensor (for tracking)
    output_shape: Vec<i64>,
    output_dtype: u8,
    attributes: Vec<(String, AttrValue)>,  // e.g., axis=1, epsilon=1e-5
}
```

**FFI functions:**
- `nsl_trace_start()` — enter tracing mode, allocate trace buffer
- `nsl_trace_register_input(tensor_ptr, name_ptr)` — mark tensor as graph input
- `nsl_trace_register_output(tensor_ptr, name_ptr)` — mark tensor as graph output
- `nsl_trace_stop() -> i64` — return trace graph pointer

**Each runtime op** (when global `TRACING` flag is set):
1. Execute normally (real shapes, real data)
2. Record op type, input pointers, output pointer
3. Insert `output_ptr -> node_id` into `ptr_to_node`

### Phase 2: ONNX Graph Builder (`crates/nsl-runtime/src/onnx.rs`)

For each `TraceOp`, resolve each input pointer:
- **Matches a registered input** -> ONNX `ValueInfoProto` (graph input edge)
- **Matches a prior op's output** in `ptr_to_node` -> intermediate ONNX edge
- **Neither** -> static weight/constant -> read the memory behind that pointer, serialize as ONNX `TensorProto` initializer

This pointer-lifecycle approach completely decouples the tracer from NSL's model struct layout. Weights are auto-resolved without walking the model struct.

**Protobuf handling:** ONNX `.proto` definitions are pre-compiled to Rust structs once and checked into the repo as `onnx_proto.rs`. The runtime depends only on `prost` for encoding — no `prost-build`, no `protoc` binary required to build NSL.

### Phase 3: Serialization

- `nsl_onnx_export(trace_ptr, path_ptr)` — serialize the ONNX `ModelProto` to disk
- Large models (>2GB): weights stored as external data files (ONNX standard)
- Opset version: 17

### Codegen Emission for `to_onnx(model, input, "out.onnx")`

```
nsl_trace_start()
nsl_trace_register_input(input_ptr, "input_0")
let result = model.forward(input_ptr)
nsl_trace_register_output(result, "output_0")
let trace = nsl_trace_stop()
nsl_onnx_export(trace, "out.onnx")
```

### Op Coverage (M18b Minimum Viable)

| Category | ONNX Operators |
|----------|---------------|
| Arithmetic | Add, Sub, Mul, Div |
| Matrix | MatMul, Transpose |
| Activation | Relu, Sigmoid, Tanh, Softmax |
| Normalization | LayerNormalization |
| Shape | Reshape, Unsqueeze, Expand, Concat |
| Embedding | Gather |

This covers the full transformer forward pass. Causal masks use additive masking (`scores + mask`), so the mask tensor becomes a static initializer via the auto-resolution logic — no `Where` op needed. The exported ONNX file has a fixed sequence length, which is standard for trace-based inference exports.

### Limitations (Documented)

- Fixed sequence length (trace captures concrete shapes)
- No dynamic control flow (input-dependent if/loops become static)
- Inference graph only (no grad/optimizer ops)

---

## Section 4: `nsl export` CLI Command

### Overview

Adds an `nsl export` subcommand with two distinct modes: source-to-ONNX (compilation + tracing) and checkpoint-to-safetensors (pure data translation).

### CLI Interface

```bash
# ONNX: compile + run export_model() entry point
nsl export model.nsl

# Safetensors: pure data conversion, no compilation
nsl export checkpoint.nslm -o weights.safetensors
nsl export checkpoint.nslm --format safetensors --output weights.safetensors
```

### Two Modes

**1. Source -> ONNX** (`nsl export model.nsl`):
- Compile the `.nsl` file with `export_model` as entry point (instead of `main`)
- Execute — the function calls `to_onnx()` with proper dummy inputs
- Output path, dtype, shape all controlled in NSL code
- Error if `export_model()` function not found

**2. Checkpoint -> Safetensors** (`nsl export checkpoint.nslm -o out.safetensors`):
- Pure data translation, no compiler involved
- Read `.nslm` header + tensor data, write safetensors
- Millisecond execution

### Why `export_model()` Instead of CLI Flags

The "Embedding Layer Dtype Trap": a `--input-shape 1,128` flag would generate float tensors by default, but transformer models start with an `Embedding` layer that expects integer token IDs. Generating the wrong dtype causes a runtime panic or undefined memory reads.

By using an `export_model()` entry point, the user writes dummy input construction in NSL with the correct types:

```python
fn export_model():
    let m = GPT2()
    let m = from_hf("gpt2", m)
    let dummy_ids = ones([1, 128], dtype="int64")
    to_onnx(m, dummy_ids, "gpt2.onnx")
```

This also naturally supports multi-input models without CLI flag gymnastics.

### Flags

- `--format <onnx|safetensors>` — explicit format (overrides extension inference)
- `--output <path>` / `-o <path>` — output path (required for safetensors, optional for ONNX since path is in NSL code)

### Format Inference (when `--format` omitted)

- Input `.nsl` -> ONNX mode (look for `export_model()`)
- Input `.nslm` + output `.safetensors` -> safetensors data conversion
- Unknown -> error with helpful message listing supported formats

---

## Section 5: File Structure & Module Organization

### New Files

| File | Responsibility |
|------|---------------|
| `crates/nsl-runtime/src/safetensors.rs` | Safetensors read/write FFI (mmap, dtype conversion, NslDict return) |
| `crates/nsl-runtime/src/huggingface.rs` | HF Hub download via `hf-hub` crate, weight loading with offset table |
| `crates/nsl-runtime/src/weight_map.rs` | Name mapping conventions, transpose rules, shape validation |
| `crates/nsl-runtime/src/trace.rs` | Tracing runtime (op recording, pointer tracking, boundary registration) |
| `crates/nsl-runtime/src/onnx.rs` | TraceGraph -> ONNX protobuf conversion, initializer extraction |
| `crates/nsl-runtime/src/onnx_proto.rs` | Pre-generated prost structs from ONNX .proto (checked in, not built) |
| `stdlib/nsl/compat.nsl` | NSL-facing API: `from_hf()`, `load_safetensors()`, `save_safetensors()`, `to_onnx()` |

### Modified Files

| File | Changes |
|------|---------|
| `crates/nsl-runtime/src/lib.rs` | Add `mod safetensors, huggingface, weight_map, trace, onnx, onnx_proto` (behind `#[cfg(feature = "interop")]`) |
| `crates/nsl-runtime/Cargo.toml` | Add feature-gated deps |
| `crates/nsl-codegen/src/builtins.rs` | Register new FFI functions in `RUNTIME_FUNCTIONS` |
| `crates/nsl-codegen/src/expr.rs` | Compile `from_hf()` (emit offset table + FFI call), compile `to_onnx()` (emit trace wrapper) |
| `crates/nsl-cli/src/main.rs` | Add `export` subcommand |

### Cargo Feature Gate

```toml
# crates/nsl-runtime/Cargo.toml
[features]
default = []
interop = ["safetensors", "hf-hub", "prost", "half"]

[dependencies]
safetensors = { version = "0.4", optional = true }
hf-hub = { version = "0.3", optional = true }
prost = { version = "0.13", optional = true }
half = { version = "2.4", optional = true }
```

```rust
// crates/nsl-runtime/src/lib.rs
#[cfg(feature = "interop")]
pub mod safetensors;
#[cfg(feature = "interop")]
pub mod huggingface;
#[cfg(feature = "interop")]
pub mod weight_map;
#[cfg(feature = "interop")]
pub mod trace;
#[cfg(feature = "interop")]
pub mod onnx;
#[cfg(feature = "interop")]
pub mod onnx_proto;
```

### Build Impact

- `cargo build` (default) — same compile time as today, no extra deps
- `cargo build --features interop` — adds networking + serialization deps
- `nsl export` checks at startup for `interop` feature, clear error if missing

### Dependencies (All Pure Rust)

| Crate | Purpose |
|-------|---------|
| `safetensors 0.4` | Safetensors format parsing and writing |
| `hf-hub 0.3` | HuggingFace Hub API client (download, caching, auth) |
| `prost 0.13` | Protobuf encoding for ONNX serialization |
| `half 2.4` | f16/bf16 <-> f32 conversion for weight loading |

### What Stays Unchanged

- `.nslm` checkpoint format (still the native format)
- Existing runtime ops (just gain a tracing flag check when `interop` feature is active)
- Parser/lexer (no new syntax — `from_hf`, `to_onnx`, etc. are regular function calls recognized by codegen)
- Semantic checker (type signatures handled through existing function type resolution)

---

## Section 6: Testing Strategy

### Unit Tests (`crates/nsl-runtime/src/`)

**safetensors.rs:**
- Round-trip: create tensors -> `save` -> `load` -> verify data matches
- Test f16/bf16 input files (create synthetic safetensors with `half` crate)
- Edge cases: empty file, single tensor, many tensors, large tensor

**weight_map.rs:**
- Name mapping: verify `self_attn` -> `attention` conventions
- Transpose flag correctness: Linear (yes) vs Embedding (no) vs LayerNorm (no)
- Shape mismatch detection (hard error)
- Unmapped NSL param (error), unused HF weight (warning)
- **Offset table mock:** allocate a `#[repr(C)]` mock struct, create synthetic `[ParamMeta]` array, call `nsl_load_weights`, assert memory at exact byte offsets was correctly written

**trace.rs:**
- Trace a sequence of ops (add, matmul, relu), verify TraceGraph op count
- Input/output boundary registration via `register_input`/`register_output`
- Pointer resolution: registered input vs prior op output vs static weight (initializer)

**onnx.rs:**
- Convert TraceGraph to ONNX protobuf, deserialize back, verify:
  - Correct node count and op types
  - Initializer shapes match source tensors
  - Input/output names match registered names
  - Opset version = 17

### CLI Integration Tests (`crates/nsl-cli/tests/`)

- `nsl export test.nslm -o out.safetensors` — succeeds, output is valid safetensors
- `nsl export test.nsl` — finds `export_model()`, runs trace, writes `.onnx`
- `nsl export test.nsl` with no `export_model()` — clear error message
- `nsl export test.txt` — format inference error with helpful message
- `nsl export` with no args — prints usage help

### Integration Tests (`tests/`)

- **Safetensors round-trip:** NSL program creates tensors, saves as safetensors, loads back, verifies data
- **ONNX export:** 2-layer MLP with `export_model()`, verify output is valid ONNX protobuf
- **HF offline:** bundled fixture in `tests/fixtures/` (tiny safetensors + synthetic offset table), test mapping without network
- **HF network:** `#[ignore]` test using `hf-internal-testing/tiny-random-gpt2` (~2MB), run manually not in CI

### End-to-End Test (`examples/m18b_interop.nsl`)

```python
model SmallTransformer:
    embedding: nn.Embedding
    layers: [TransformerBlock; 2]
    norm: nn.LayerNorm
    head: nn.Linear

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.embedding.forward(x)
        for i in range(2):
            h = self.layers[i].forward(h)
        h = self.norm.forward(h)
        return self.head.forward(h)

fn export_model():
    let m = SmallTransformer(vocab=1000, dim=64, heads=4, layers=2)
    let m = from_hf("hf-internal-testing/tiny-random-gpt2", m)
    let dummy = ones([1, 16], dtype="int64")
    to_onnx(m, dummy, "small_transformer.onnx")

fn main():
    let m = SmallTransformer(vocab=1000, dim=64, heads=4, layers=2)

    # Test HF download + weight mapping into struct
    let m = from_hf("hf-internal-testing/tiny-random-gpt2", m, device="cpu")

    # Verify loaded weights produce output
    let dummy = ones([1, 16], dtype="int64")
    let out = m.forward(dummy)
    print("Forward pass shape:", out.shape)

    # Test safetensors round-trip
    save_safetensors(m.parameters(), "roundtrip.safetensors")
    let reloaded = load_safetensors("roundtrip.safetensors")
    print("Round-trip:", len(reloaded), "tensors")
    print("End-to-end interop complete")
```
