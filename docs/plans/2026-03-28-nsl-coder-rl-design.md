# NSL-Coder-RL: Compiler-Guided Reinforcement Learning for Code Generation

**Date:** 2026-03-28
**Status:** Design
**Goal:** Train a small (10-15M param) language model that generates correct NSL code,
understands English task descriptions, and can read/write Rust — using the NSL compiler
as the reward model instead of human annotation.

---

## 1. Why This Works

### The Compiler-as-Reward-Model Advantage

Traditional RL for code (CodeRL, AlphaCode, RLHF-tuned Codex) requires either:
- Human annotators rating code quality (expensive, slow, noisy)
- A separate reward model trained on human preferences (doubles compute)
- Unit test suites written by humans (limited coverage)

NSL eliminates all three. The compilation pipeline provides a **5-stage graduated reward
signal** in milliseconds:

| Stage | What it catches | Time | Reward |
|-------|----------------|------|--------|
| **Parse** | Syntax errors, indentation, keywords | <1ms | 0.0 |
| **Semantic** | Undefined variables, type mismatches | <5ms | 0.2 |
| **Shape check** | Dimension mismatches, broadcasting errors | <5ms | 0.4 |
| **Compile** | Codegen errors, unresolved symbols | <50ms | 0.6 |
| **Execute + test** | Runtime errors, assertion failures | <500ms | 0.8-1.0 |

This is 5x more signal density than Python (which only has parse + runtime), and
the shape checking stage is unique to NSL — it catches an entire category of ML bugs
that no other language detects at compile time.

### The Error String Advantage

The NSL compiler produces error messages with:
- Exact source location (file:line:col)
- Expected vs actual types/shapes
- Symbolic dimension names (`Tensor<[batch, 512]> vs Tensor<[512, 1408]>`)

These error strings are appended to the prompt for retry attempts, creating an
**implicit curriculum**: the model learns to read and fix compiler errors, developing
an internal model of NSL's type system.

---

## 2. Model Architecture

### 2.1 Size and Shape

**Target: 12M parameters** — sufficient for a domain-restricted language model.

| Component | Dimension | Parameters |
|-----------|-----------|------------|
| Embedding | 4096 vocab x 384 hidden | 1.6M |
| 6 transformer blocks | 384 hidden, 6 heads, 1024 FFN | 8.5M |
| LM head (weight-tied) | 384 x 4096 | (shared) |
| RMSNorm (7 instances) | 384 each | 2.7K |
| **Total** | | **~10.1M** |

**Design choices:**
- **384 hidden dim** — sweet spot for 10M-class models (GPT-2 small is 768, but we have 100x less data)
- **6 layers** — enough for multi-step reasoning about type/shape relationships
- **6 attention heads, 64 head dim** — standard ratio
- **1024 FFN dim** — 2.67x hidden (SwiGLU)
- **4096 vocab** — custom NSL-aware tokenizer (see Section 3)
- **2048 max sequence length** — enough for most NSL functions + error context
- **RoPE positional encoding** — no learned positional embeddings

### 2.2 Why Not Smaller/Larger

- **<5M params**: Can't learn the relationship between shape annotations and matmul
  dimensions. Shape reasoning requires multi-step attention across function boundaries.
- **>20M params**: Overfits on the ~140K tokens of NSL training data. The RL phase
  generates more data, but the base model needs to be small enough to iterate fast
  (target: 1000 GRPO steps/hour on a single RTX 5070 Ti).

### 2.3 Multi-Language Support

The model needs three "languages":
1. **English** — for understanding task descriptions and generating comments
2. **NSL** — primary code generation target
3. **Rust** — for reading/understanding the NSL runtime (optional, for advanced tasks)

The tokenizer handles all three. English tokens are standard BPE. NSL keywords are
single tokens. Rust keywords share tokens with NSL where possible (both have `fn`,
`let`, `struct`, `impl`).

---

## 3. Tokenizer Design

### 3.1 NSL-Aware Tokenizer

**4096 total tokens**, allocated:

| Category | Count | Examples |
|----------|-------|---------|
| NSL keywords | 50 | `fn`, `let`, `const`, `model`, `train`, `Tensor`, `grad`, `kernel`, `serve` |
| NSL builtins | 80 | `randn`, `zeros`, `matmul`, `softmax`, `cross_entropy`, `embedding_lookup` |
| NSL types | 20 | `int`, `float`, `bool`, `str`, `Tensor`, `list`, `dict` |
| NSL operators | 30 | `@` (matmul), `|>` (pipe), `**`, `+=`, `->`, `:`, `=` |
| NSL punctuation | 20 | `(`, `)`, `[`, `]`, `{`, `}`, `,`, `.`, `\n`, indent/dedent |
| Rust keywords | 30 | `pub`, `struct`, `impl`, `use`, `mut`, `unsafe`, `&`, `'a` |
| Numbers | 32 | `0`-`9`, `.`, `e`, `+`, `-`, common literals (0.0, 1.0, 0.01) |
| English BPE | 3000 | Standard BPE on English text (task descriptions, comments) |
| Special | 10 | `<|start|>`, `<|end|>`, `<|code|>`, `<|error|>`, `<|fix|>`, `<|pad|>` |
| Identifiers | 824 | Common variable names (`x`, `y`, `loss`, `logits`, `hidden`, `batch`, `self`) |

**Key design decisions:**
- Every NSL keyword is exactly 1 token — the model never has to assemble `re` + `turn`
- Shape syntax `[batch, seq, 512]` is tokenized as `[` + `batch` + `,` + `seq` + `,` + `512` + `]`
- Indentation is explicit tokens (INDENT/DEDENT) like Python's tokenizer
- Common patterns like `fn forward(self,` are frequent enough to memorize

### 3.2 Building the Tokenizer

```
Phase 1: Start with the 230 reserved tokens (keywords, builtins, operators, specials)
Phase 2: Run BPE on the combined English + NSL + Rust corpus to learn 3866 merge rules
Phase 3: Verify every NSL keyword resolves to exactly 1 token
Phase 4: Export as JSON (compatible with NSL's existing tokenizer_encode/decode)
```

---

## 4. Training Data

### 4.1 Phase 1: Supervised Fine-Tuning (SFT) Corpus

| Source | Files | Tokens (est.) | Purpose |
|--------|-------|---------------|---------|
| NSL stdlib | 25 .nsl files, 800 LOC | ~8K | Canonical NSL patterns |
| NSL examples | 89 .nsl files, 1500 LOC | ~15K | Diverse usage patterns |
| NSL models | 15 .nsl files, 500 LOC | ~5K | Real model architectures |
| NSL spec | 13 chapters, 4200 LOC | ~40K | Language reference (English + NSL) |
| Rust runtime | 282 .rs files, 133K LOC | ~500K | Runtime implementation |
| Synthetic pairs | generated | ~100K | "Task: ... Code: ..." pairs |
| **Total** | | **~668K tokens** |

The synthetic pairs are generated by:
1. Taking each stdlib function and writing an English description
2. Taking each example and writing a task prompt that would produce it
3. Generating variations (rename variables, change dimensions, swap optimizers)

### 4.2 Phase 2: Rejection Sampling Data

After SFT, generate 10K task→code pairs, filter by compiler:
- Keep only pairs where the generated code compiles AND passes shape checking
- This creates a high-quality dataset of ~1-3K correct examples
- Retrain on this filtered set (1-2 epochs)

### 4.3 Phase 3: GRPO Training Data

Generated on-the-fly during RL:
- Task generator produces prompts (see Section 6)
- Model generates K=8 candidates per prompt
- NSL compiler scores each candidate
- GRPO update uses the within-group reward normalization

---

## 5. GRPO Training Loop

### 5.1 Algorithm

```
for each training step:
    1. Sample a task prompt from the task generator
    2. Generate K=8 code completions from the policy model
    3. Score each completion through the NSL compilation pipeline:
       - Parse:     pass=0.2, fail=0.0
       - TypeCheck:  pass=0.4
       - ShapeCheck: pass=0.6
       - Compile:    pass=0.8
       - Execute+Test: pass=1.0, runtime_error=0.7
    4. Compute group-normalized advantages:
       A_i = (r_i - mean(r)) / (std(r) + eps)
    5. GRPO policy gradient update:
       L = -sum(A_i * log(pi(code_i | prompt)))
    6. KL penalty against reference model:
       L += beta * KL(pi || pi_ref)
```

### 5.2 Reward Function Details

```python
def compute_reward(nsl_code: str, task: Task) -> float:
    # Stage 1: Parse
    result = nsl_check(nsl_code, stage="parse")
    if not result.ok:
        return 0.0 + partial_credit(result)

    # Stage 2: Semantic analysis
    result = nsl_check(nsl_code, stage="semantic")
    if not result.ok:
        return 0.2 + 0.1 * error_specificity(result)

    # Stage 3: Shape checking
    result = nsl_check(nsl_code, stage="shapes")
    if not result.ok:
        return 0.4 + 0.1 * shape_error_quality(result)

    # Stage 4: Full compilation
    result = nsl_build(nsl_code)
    if not result.ok:
        return 0.6

    # Stage 5: Execution + tests
    if task.has_test:
        exec_result = nsl_run(nsl_code, test=task.test_code, timeout=5.0)
        if exec_result.passed:
            return 1.0
        elif exec_result.ran_without_crash:
            return 0.8
        else:
            return 0.7
    else:
        return 0.85  # compiles but no test available

def partial_credit(parse_error) -> float:
    """Give credit for how close to valid the parse got."""
    # Count valid statements before the error
    return min(0.15, parse_error.valid_stmt_count * 0.02)
```

### 5.3 Error-Augmented Retry

When a candidate scores < 1.0, create a retry prompt:

```
<|code|>
{original_code}
<|error|>
{compiler_error_message}
<|fix|>
```

The model learns to read compiler errors and produce fixes. This is trained as a
separate "fix" task alongside the "generate" task, alternating during GRPO.

### 5.4 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| K (group size) | 8 | Balance between diversity and compute |
| Learning rate | 1e-5 | Standard for RL fine-tuning |
| KL coefficient (beta) | 0.05 | Prevent catastrophic forgetting of SFT knowledge |
| Temperature | 0.8 | Encourage diversity in generation |
| Top-p | 0.95 | Standard nucleus sampling |
| Max sequence length | 2048 | Enough for function + error context |
| Batch size | 32 prompts x 8 candidates = 256 | Fits in 16GB VRAM |
| Steps | 10,000 GRPO steps | ~80K unique task-code pairs seen |
| Epochs over task set | ~20 | Tasks are recycled with different seeds |

---

## 6. Task Generator

### 6.1 Task Categories

| Category | Weight | Example |
|----------|--------|---------|
| **Function writing** | 30% | "Write an NSL function that computes the mean of a tensor along dim 1" |
| **Model definition** | 20% | "Define a 2-layer MLP model with ReLU activation and dropout" |
| **Bug fixing** | 20% | "Fix this code: [broken NSL code] Error: [compiler error]" |
| **Type annotation** | 10% | "Add type annotations to this function: fn foo(x, y) -> ?" |
| **Shape reasoning** | 10% | "What shape does `x @ w.transpose(0,1)` produce if x is [32, 512] and w is [512, 1024]?" |
| **Train block** | 5% | "Write a train block that trains model M with AdamW and cosine schedule" |
| **Rust reading** | 5% | "What does this Rust function do? [runtime code]. Explain in English." |

### 6.2 Template-Based Generation

Tasks are generated from templates with random parameters:

```
TEMPLATE: "Write an NSL function named {name} that takes a Tensor of shape
[{dim1}, {dim2}] and returns a Tensor of shape [{dim1}, {dim3}]. Use {activation}
activation and {norm} normalization."

PARAMS:
  name: random from [transform, project, encode, process, compute]
  dim1: random from [batch, 32, 64, 128]
  dim2: random from [64, 128, 256, 512]
  dim3: random from [64, 128, 256, 512]
  activation: random from [relu, gelu, silu, sigmoid]
  norm: random from [layernorm, rmsnorm, none]
```

### 6.3 Curriculum Learning

Start with easy tasks, increase difficulty:

| Phase | Steps | Task Difficulty |
|-------|-------|-----------------|
| 1 | 0-2000 | Single-line expressions, simple functions |
| 2 | 2000-5000 | Multi-line functions, model definitions |
| 3 | 5000-8000 | Bug fixing, shape reasoning, train blocks |
| 4 | 8000-10000 | Full programs, multi-file, Rust reading |

---

## 7. Infrastructure

### 7.1 Training Stack

Everything runs on NSL's own infrastructure:

```
NSL-Coder-RL Training Stack:
  Model:        12M param transformer (NSL model definition)
  Tokenizer:    Custom NSL-aware BPE (built with NSL tokenizer API)
  Training:     NSL train block with AdamW (once train block GPU bug is fixed)
  Reward:       nsl check + nsl run (subprocess calls)
  Data:         NSL stdlib + examples + spec + synthetic
  Hardware:     Single RTX 5070 Ti (16GB VRAM)
```

### 7.2 Reward Server

A lightweight process that:
1. Receives NSL code strings over a pipe/socket
2. Writes each to a temp .nsl file
3. Runs `nsl check` and `nsl run` with timeout
4. Returns the reward score + error string
5. Parallelized across 8 worker threads (one per K group member)

### 7.3 Self-Play Potential

After the model reaches >50% compilation rate:
1. Model generates a task description (English)
2. Model generates a solution (NSL code)
3. Compiler verifies the solution
4. If correct: add (task, solution) to the training set
5. If wrong: add (task, broken_code, error, fixed_code) as a fix-task

This creates an infinite training data flywheel with zero human involvement.

---

## 8. Evaluation

### 8.1 Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Parse rate | >95% | % of generated code that parses |
| Compile rate | >80% | % that fully compiles |
| Test pass rate | >60% | % that passes provided unit tests |
| Shape correctness | >90% | % with correct tensor dimensions |
| First-attempt success | >40% | % correct on first try (no retry) |
| Fix-after-error rate | >70% | % correct after seeing compiler error |

### 8.2 Benchmark Suite

50 hand-written tasks spanning all categories:
- 10 simple function tasks (1-5 lines)
- 10 model definition tasks
- 10 training configuration tasks
- 10 bug fix tasks (with known errors)
- 10 shape reasoning tasks

Each task has a reference solution and a test that validates correctness.

---

## 9. Implementation Plan

### Phase 1: Tokenizer + SFT (2-3 days)
1. Build NSL-aware tokenizer (token_nsl.py or in Rust)
2. Collect and format SFT corpus
3. Generate synthetic task-code pairs
4. Train base model via SFT for 5-10 epochs

### Phase 2: Rejection Sampling (1 day)
1. Generate 10K completions from SFT model
2. Score through compiler
3. Retrain on filtered correct completions

### Phase 3: GRPO (3-5 days)
1. Build reward server (parallel nsl check workers)
2. Implement GRPO update rule
3. Build task generator with curriculum
4. Run 10K GRPO steps
5. Evaluate on benchmark suite

### Phase 4: Self-Play (ongoing)
1. Enable task generation from the model itself
2. Continuous self-play loop
3. Periodic evaluation snapshots

**Total estimated time: 7-10 days to first usable model.**

---

## 10. The NSL Bootstrap

The ultimate goal: NSL-Coder-RL writes NSL code that improves NSL itself.

```
Human: "Add a LayerScale module to the NSL stdlib"
NSL-Coder-RL generates:
  model LayerScale(dim: int, init_val: float):
      gamma: Tensor = full([dim], init_val)

      fn forward(self, x: Tensor) -> Tensor:
          return x * self.gamma

Compiler: OK
Tests: PASS
→ Merged into stdlib
→ Model retrains on its own contribution
```

This is the compiler-guided AI development loop: the model writes code, the compiler
validates it, correct code becomes training data for the next generation.
