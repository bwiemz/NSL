# M15: Tokenization + Standard Library — Design Document

**Goal:** Full-spec tokenization system, neural network standard library, and test framework — culminating in an end-to-end "tokenize text → feed to model → train a language model" demo.

**Architecture:** Bottom-up, runtime-first. Rust runtime primitives → test framework → nn layers in NSL → tokenizer system → integration. Each layer validates the one below it.

**Key decisions:**
- Tokenizer keyword is hybrid: keyword for declaration/compile-time checking, desugars to runtime calls (like `train`)
- NN layers are `model` definitions in NSL, with targeted Rust runtime primitives for heavy ops
- BPE/WordPiece/Unigram/SentencePiece via HuggingFace `tokenizers` Rust crate (not hand-rolled)
- Encode/decode callable from NSL; training delegated to Rust runtime
- Full test framework with `@test`, assertions, orchestrator-pattern isolation

---

## Section 1: Runtime Primitives

### New Tensor Operations (nsl-runtime/src/tensor.rs)

**Creation & initialization:**
- `nsl_tensor_randn(shape) -> tensor` — normal distribution (for weight init)

**Activation functions:**
- `nsl_tensor_relu(tensor) -> tensor`
- `nsl_tensor_gelu(tensor) -> tensor`
- `nsl_tensor_silu(tensor) -> tensor`
- `nsl_tensor_sigmoid(tensor) -> tensor`
- `nsl_tensor_tanh_act(tensor) -> tensor`
- `nsl_tensor_softmax(tensor, dim) -> tensor` — numerically stable

**Fused layer operations:**
- `nsl_tensor_layernorm(input, weight, bias, eps) -> tensor`
  - Tape: saves `{input, mean, inv_std, weight, bias}` for backward
- `nsl_tensor_rmsnorm(input, weight, eps) -> tensor`
  - Tape: saves `{input, rms, weight}` for backward
- `nsl_tensor_embedding_lookup(weight, indices) -> tensor`
- `nsl_tensor_conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w) -> tensor`

**Stochastic/routing ops (require tape state):**
- `nsl_tensor_dropout(tensor, p, training) -> tensor`
  - Generates random mask, applies it, saves `TapeOp::Dropout{input, mask, p}` to tape
  - Backward: `grad_input = grad_output * mask / (1 - p)`
- `nsl_tensor_maxpool2d(input, kernel_h, kernel_w, stride, padding) -> tensor`
  - Computes argmax indices per window, saves `TapeOp::MaxPool2d{input_shape, indices, ...}`
  - Backward: scatters gradients to max positions

**Shape operations:**
- `nsl_tensor_cat(tensor_list, dim) -> tensor` — concatenation
- `nsl_tensor_slice(tensor, dim, start, end) -> tensor` — slicing along a dimension

**Training mode (global state):**
- `nsl_set_training_mode(mode: bool)` — called by `train` block, toggleable by user
- `nsl_is_training() -> bool` — read by Dropout and other training-sensitive ops

### Autodiff Backward Entries

All new ops that participate in gradient computation need backward implementations in `autodiff.rs`:
- Activations: relu, gelu, silu, sigmoid, tanh, softmax
- LayerNorm, RMSNorm (using saved mean/inv_std/rms)
- Embedding lookup (scatter gradients to selected rows)
- Conv2d (full convolution backward for weight and input)
- Dropout (multiply by saved mask)
- MaxPool2d (scatter to argmax indices)
- Cat, Slice (split/pad gradients)

### Assertion Runtime Functions

- `nsl_assert(condition: i8, msg_ptr: i64, msg_len: i64)` — abort with message on failure
- `nsl_assert_eq(a: i64, b: i64, msg_ptr: i64, msg_len: i64)` — equality check
- `nsl_assert_close(tensor_a: i64, tensor_b: i64, rtol: f64, atol: f64, msg_ptr: i64, msg_len: i64)`
  - Shape validation first: check ndim match, then each dimension
  - Then element-wise: `abs(a - b) <= atol + rtol * abs(b)`

### String Deallocation

- `nsl_string_free(ptr: i64)` — free dynamically allocated strings
- Codegen emits `nsl_string_free` for string temporaries at end of block scope
- Critical for `tokenizer.decode()` return values

---

## Section 2: Test Framework (`nsl test`)

### Language Additions
- `@test` decorator on functions — marks them as test cases
- Semantic checker validates: `@test` functions take no args, return nothing
- `assert(condition)`, `assert_eq(a, b)`, `assert_close(a, b, rtol, atol)` as builtins

### CLI (`nsl test` subcommand)
- Added to clap in `nsl-cli`
- Discovery: scan AST for `@test`-decorated functions
- Filtering: `nsl test file.nsl --filter test_name`

### Orchestrator Pattern (Process Isolation)
Critical: a failed assertion calls `abort()`, killing the process. If all tests run in one process, one failure kills the rest.

**Solution:**
1. `nsl test` compiles a binary with a hidden `main` that accepts `--run <test_name>`
2. CLI discovers `@test` functions at compile time
3. CLI spawns the binary as a child process once per test
4. Exit code 0 = pass, non-zero = fail (stderr captured for error details)
5. Full summary printed regardless of crashes

```
nsl test tests/nn_test.nsl
  ├─ Compile → test_bin.exe (accepts --run <name>)
  ├─ Discover: [test_relu_pos, test_relu_neg, test_ln]
  ├─ Spawn: test_bin.exe --run test_relu_pos  → exit 0 → PASS
  ├─ Spawn: test_bin.exe --run test_relu_neg  → exit 1 → FAIL
  ├─ Spawn: test_bin.exe --run test_ln        → exit 0 → PASS
  └─ Summary: 2 passed, 1 failed
```

### String Constants for Assert Messages
- Compiler emits string bytes into read-only data segment
- Runtime signature: `nsl_assert(condition: i8, msg_ptr: i64, msg_len: i64)`
- For `assert_eq`/`assert_close`: runtime formats actual vs expected in error output

---

## Section 3: Neural Network Standard Library

### Architecture
All layers are `model` definitions in NSL calling Rust runtime primitives for heavy operations. This validates the `model` system at scale and keeps layers inspectable, composable, and autodiff-compatible.

### Weight Initialization
- **Kaiming/He** (default for ReLU/GELU layers): `randn * sqrt(2.0 / fan_in)`
- **Embedding**: `randn * 0.02` (GPT-style)
- **LayerNorm/RMSNorm**: weights = ones, biases = zeros

### Linear Weight Shape (HF Compatibility)
- `Linear.w` stored as `[out_features, in_features]` (matches PyTorch/HuggingFace)
- Forward: `x @ transpose(self.w) + self.b`
- Avoids mandatory transpose on every `from_hf()` model load

### Training Mode
- Global runtime state: `nsl_set_training_mode(bool)` / `nsl_is_training() -> bool`
- `train` block sets training=true before step, restores after
- `Dropout.forward()` checks `nsl_is_training()` and passes to `nsl_tensor_dropout`

### Layers

| Layer | File | Rust Primitive |
|-------|------|----------------|
| Linear(in, out) | layers.nsl | matmul (existing) |
| Embedding(vocab, dim) | layers.nsl | nsl_tensor_embedding_lookup |
| Conv2d(in_ch, out_ch, k, s, p) | layers.nsl | nsl_tensor_conv2d |
| MaxPool2d(k, s, p) | layers.nsl | nsl_tensor_maxpool2d |
| LayerNorm(dim, eps) | norms.nsl | nsl_tensor_layernorm |
| RMSNorm(dim, eps) | norms.nsl | nsl_tensor_rmsnorm |
| Dropout(p) | dropout.nsl | nsl_tensor_dropout |
| MLP(in, hidden, out) | layers.nsl | Composition: Linear → act → Linear |
| Attention(dim, heads) | attention.nsl | Q/K/V Linear + softmax + output Linear |

### Activation Functions (standalone `fn`, not `model`)

| Function | File | Rust Primitive |
|----------|------|----------------|
| relu | activations.nsl | nsl_tensor_relu |
| gelu | activations.nsl | nsl_tensor_gelu |
| silu | activations.nsl | nsl_tensor_silu |
| sigmoid | activations.nsl | nsl_tensor_sigmoid |
| tanh | activations.nsl | nsl_tensor_tanh_act |
| softmax | activations.nsl | nsl_tensor_softmax |

### File Structure
```
stdlib/nsl/nn/
  layers.nsl        — Linear, Embedding, Conv2d, MaxPool2d, MLP
  norms.nsl         — LayerNorm, RMSNorm
  activations.nsl   — relu, gelu, silu, sigmoid, tanh, softmax
  attention.nsl     — Attention (multi-head)
  dropout.nsl       — Dropout
  losses.nsl        — (existing: mse, l1, cross_entropy, bce)
```

---

## Section 4: Tokenizer System

### Compiler Pipeline
The `tokenizer` keyword is already parsed into AST (`TokenizerDef`). Semantic checker and codegen currently treat it as a no-op.

**Semantic checker updates:**
- Validate config: `algorithm` ∈ {bpe, wordpiece, sentencepiece, unigram, char, byte}
- Validate `vocab_size` is positive integer
- Validate body sections: `special_tokens`, `normalize`, `pre_tokenize`, `padding`, `truncation`
- Compile-time vocab checking: if `model_file` provided, read vocab size from file and verify against `Embedding(vocab_size, ...)` usage

**Codegen:**
- `TokenizerDef` desugars to runtime calls
- Constructs config, passes to training/loading runtime function
- Stores tokenizer handle in local variable
- Methods (`encode`, `decode`, `encode_batch`, etc.) dispatch to runtime

### Rust Runtime (HuggingFace `tokenizers` crate)

**Dependency:** `tokenizers = "0.21"` in `nsl-runtime/Cargo.toml`

This is the first external dependency in `nsl-runtime`. Justified because:
- Pure Rust, no C/C++ dependencies
- Battle-tested (used by all HF Python tokenizers)
- Provides BPE, WordPiece, Unigram, SentencePiece algorithms
- Avoids months of hand-rolling complex algorithms

**Runtime functions** (`crates/nsl-runtime/src/tokenizer.rs`):

Training:
- `nsl_bpe_train(corpus_path, vocab_size, min_freq, special_tokens) -> handle`
- `nsl_wordpiece_train(corpus_path, vocab_size, min_freq) -> handle`
- `nsl_unigram_train(corpus_path, vocab_size) -> handle`
- `nsl_byte_tokenizer_new(vocab_size) -> handle` (no training needed)
- `nsl_char_tokenizer_new(vocab_size) -> handle` (no training needed)

I/O:
- `nsl_tokenizer_load(path_ptr, path_len) -> handle`
- `nsl_tokenizer_save(handle, path_ptr, path_len)`

Encoding/Decoding:
- `nsl_tokenizer_encode(handle, text_ptr, text_len) -> tensor` (1D int tensor)
- `nsl_tokenizer_decode(handle, tensor) -> str_ptr` (caller must free)
- `nsl_tokenizer_encode_batch(handle, texts_list, padding, truncation, max_len) -> list`
  - Returns `NslList` of `[input_ids_tensor, attention_mask_tensor]`
- `nsl_tokenizer_decode_batch(handle, tensor) -> list<str>`

Metadata:
- `nsl_tokenizer_vocab_size(handle) -> int`

Streaming:
- `nsl_streaming_tokenizer_create(handle, chunk_size, num_workers) -> stream_handle`
- `nsl_streaming_tokenizer_process(stream_handle, input_path, output_path)`

### NSL Stdlib Wrappers
```
stdlib/nsl/tokenize/
  tokenizer.nsl     — Tokenizer wrapper, encode/decode
  trainers.nsl      — BPE/WordPiece/SentencePiece/Unigram trainer wrappers
  streaming.nsl     — StreamingTokenizer for large corpora
```

---

## Section 5: Integration & End-to-End Deliverable

### Integration Example (`examples/m15_tiny_lm.nsl`)

```python
from nsl.nn.layers import Linear, Embedding
from nsl.nn.norms import LayerNorm
from nsl.nn.activations import gelu
from nsl.nn.losses import cross_entropy

tokenizer ByteTok(algorithm="byte", vocab_size=256):
    special_tokens:
        pad = "<pad>"

model TinyLM:
    embed: Embedding(256, 64)
    ln: LayerNorm(64)
    proj: Linear(64, 256)

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.embed.forward(x)
        h = self.ln.forward(h)
        h = gelu(h)
        return self.proj.forward(h)

let m = TinyLM()
let tok = ByteTok()
let encoded = tok.encode("hello world this is a test")

train(model=m, epochs=10):
    optimizer: Adam(lr=0.001)
    step(batch):
        let input = encoded[0..-1]
        let targets = encoded[1..]
        let logits = m.forward(input)
        let loss = cross_entropy(logits, targets)
    callbacks:
        on_epoch(epoch, loss):
            print(loss)
```

Key details:
- Uses `algorithm="byte"` (no training needed) for integration test
- Causal LM: input shifted by 1 position from targets
- Separate test trains BPE from corpus file

### Test Suite (`tests/m15_test.nsl`)
- `@test` functions for each nn layer (forward pass, known values)
- `@test` functions for activation functions
- `@test` functions for tokenizer encode/decode roundtrip
- `@test` functions for assert_close correctness
- Run via `nsl test tests/m15_test.nsl`

### Success Criteria
1. `nsl test` discovers and runs `@test` functions with pass/fail reporting and process isolation
2. All nn layers have passing tests (Linear, Embedding, LayerNorm, RMSNorm, Attention, Conv2d, Dropout, MLP, MaxPool2d)
3. All activation functions have passing tests (relu, gelu, silu, sigmoid, tanh, softmax)
4. Tokenizer can train BPE from corpus, encode text, decode back (roundtrip)
5. Byte tokenizer works without training
6. Batch encoding returns [input_ids, attention_mask]
7. Integration example compiles, trains, and loss decreases over epochs
8. Autodiff backward passes work for all new ops
