# NotebookLM Questions for NSL General Notebook

Upload the 6 summary docs from `docs/summaries/` as sources first, then ask these questions
and save each answer as a NotebookLM note.

---

## Architecture & Compilation (3 questions)

### Q1: Compilation Pipeline
How does NeuralScript's compilation pipeline work end-to-end? Walk through the 8 Rust crates (nsl-errors, nsl-lexer, nsl-ast, nsl-parser, nsl-semantic, nsl-codegen, nsl-runtime, nsl-cli) and explain how source code flows from .nsl file to native executable. What role does Cranelift play? How does the C ABI runtime get linked in?

### Q2: Tensor Runtime Representation
What is the NslTensor struct layout in memory? How does NSL handle the CPU (f64) vs GPU (f32) dtype split? How does device dispatch work — when a tensor operation is called, how does the runtime decide whether to use the CPU path or launch a CUDA kernel?

### Q3: Module System & Multi-File Compilation
How does NSL resolve imports and compile multi-file programs? What is the dependency resolution strategy? How does two-pass function compilation work (declaring signatures first, then compiling bodies)?

---

## Type System & Safety (3 questions)

### Q4: Tensor Shape System
What compile-time guarantees does NSL provide for tensor shapes? How do named dimensions (e.g., Tensor<[batch="B", heads="H", seq="S", dim=64], fp8>) work? What shape errors are caught at compile time vs runtime? How does broadcasting verification work?

### Q5: Shape Algebra
How does the shape algebra solver (M49) prove things like reshape validity with symbolic dimensions? What constraint types does it support (equality, divisibility, range bounds)? Give an example of a proof it can perform.

### Q6: Moat Features
What are NeuralScript's "moat features" — capabilities that are structurally impossible in Python/PyTorch? For each one (memory planning, roofline cost model, linear types, vmap, source-to-source AD, compiled FSM decoding, determinism proofs, shape algebra, effect system, weight-aware compilation, WCET proofs, ZK circuits), explain WHY Python can't do it and HOW NSL's static compilation enables it.

---

## GPU & Optimization (3 questions)

### Q7: Operator Fusion System
How does NSL's 4-level operator fusion system work? Explain each level: (1) elementwise fusion — how are chains of element-wise ops detected and compiled into single PTX kernels? (2) epilogue fusion — how are post-matmul operations fused into the matmul kernel? (3) reduction fusion — how are map-reduce patterns fused? (4) graph-level fusion — how does the DAG analysis identify global fusion opportunities?

### Q8: FlashAttention-2/3 Implementation
How does NSL's FlashAttention implementation work? Cover: tiled attention with online softmax, Hopper wgmma.mma_async with TMA loads and warp specialization (producer/consumer warps, pingpong scheduling), Ampere mma.sync fallback, paged KV-cache integration, RoPE fusion (in-register rotary embedding), GQA support, tree-structured causal masks for speculative decoding, logsumexp saving for backward pass, and the 21+ kernel variant parameterization.

### Q9: Weight-Aware Compilation
How does weight-aware compilation (M52) work? How does the compiler load .safetensors at compile time? What constant folding does it perform (matmul, add, relu with known operands)? How does dead weight elimination work? What is the sparsity analysis (CSR layout)? How does SHA-256 integrity checking protect against weight corruption?

---

## Inference Stack (3 questions)

### Q10: Production Inference Architecture
How do PagedAttention, continuous batching, and the serve block DSL work together for production inference? Explain: paged KV-cache with Copy-on-Write, the serve block's request scheduler, chunked prefill interleaved with decode, iteration-level batching, preemption on memory pressure, and ragged tensor handling for variable-length sequences.

### Q11: Scaling Inference
How does NSL scale inference across multiple GPUs? Cover: tensor parallelism (@shard decorator, weight splitting, all-reduce), ring attention (@context_parallel, sequence distribution across GPUs), pipeline parallelism (1F1B scheduling, stage staggering), and disaggregated inference (separate prefill/decode worker pools with KV transfer).

### Q12: Constrained Decoding
How does NSL compile grammars into minimized DFAs for structured generation? Walk through the pipeline: Grammar (BNF/regex) → NFA (Thompson construction) → DFA (subset construction with epsilon closure) → Minimized DFA (Hopcroft's algorithm) → Token-level FSM with per-state bitmasks for logit masking. How does this guarantee valid JSON/SQL/code output?

---

## Training & Autodiff (3 questions)

### Q13: Tape-Based Autodiff
How does NSL's tape-based automatic differentiation work? Explain: tape recording during forward pass, backward pass computation via grad() keyword, gradient accumulation via pointer-keyed hashmap. What are the safety rules around tape op raw pointers (pointer-as-key only, never dereference, shape stored separately)?

### Q14: Training DSL
How does the train block DSL compile into an optimized training loop? What does the compiler generate (epoch loop, implicit tape_start/backward/tape_stop, per-parameter optimizer dispatch, scheduler injection, auto-import of optimizer/scheduler modules)? What are the key implementation gotchas (grad is a keyword so use 'gradient' in stdlib, model is a keyword, use ** not pow())?

### Q15: Source-to-Source AD
How does source-to-source AD (M40) differ from tape-based AD? Explain Wengert extraction (converting forward AST to elementary operation sequence), adjoint generation (walking Wengert ops in reverse), and why this enables compiler optimizations on the backward pass that tape-based AD cannot.

---

## Frontier Features (3 questions)

### Q16: ZK Inference Circuits
How would ZK inference circuits work in NSL? Explain: compiling a forward pass to PLONKish constraints, the 4 privacy modes (weight_private, input_private, full_private, architecture_attestation), fixed-point arithmetic for field compatibility, lookup tables for non-linear activations (ReLU, GELU), and why NSL's static computation DAG is essential (Python's dynamic dispatch prevents this).

### Q17: Effect System
How does NSL's effect system track side effects? Explain: the 4 effect categories (IO, RANDOM, MUTATION, COMMUNICATION), effect inference via call graph propagation, the @pure and @deterministic decorators, and how effect tracking enables reproducibility verification, parallelism safety, and optimization. What is the known M51a limitation with explicit Rng seeds?

### Q18: Implementation Maturity (Updated 2026-03-21)

What is the current implementation maturity of NeuralScript across all milestones M9-M62? The codebase has shipped most major compiler/runtime plans, but several subsystems still retain deferred, fallback-only, or stubbed pieces. Key implemented areas include FlashAttention-3 Hopper wgmma, 50+ AD backward rules, EAGLE-2 dynamic draft trees + Lookahead decoding, Blackwell quantization scaffolding (MXFP8/NVFP4 helpers and runtime quantizers), FBIP in-place mutation with GPU kernels and static reuse analysis, effect polymorphism (Effect::Var + unification), format-agnostic @layout sparsity with TACO concrete index notation, cost-guided fusion profitability with register pressure + occupancy, two-tier WCET, Jolt lookup-native ZK gates + Nova folding + Mersenne-31 field, rematerialization, SharedMem pipeline comm, C API forward pass, and disaggregated worker loops. Also call out the major partial areas still visible in code, such as RDMA ibverbs wiring, GDS cuFile integration, some serve/AD/ZK backend paths, and expert-parallel all-to-all. Categorize each milestone as Production, Functional, Framework, or Not Started.
