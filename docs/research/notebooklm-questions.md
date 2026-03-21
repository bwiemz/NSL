# NotebookLM Research Questions

After uploading sources to each notebook, ask these questions and save each answer as a pinned Note.
These are designed to produce actionable research briefs for NSL development.

---

## 1. Competitive Landscape

### Architecture & Compilation
- What are the fundamental architectural differences between MLIR-based compilers (Mojo, Triton) and LLVM JIT-based approaches (Julia)? What tradeoffs does each make?
- How does XLA's StableHLO intermediate representation compare to TVM's Relay/Relax IR? What can each express that the other cannot?
- What caused Swift for TensorFlow to fail? Extract every concrete lesson about building a new ML language.
- How does Halide's algorithm/schedule separation influence modern ML compilers like TVM and Triton?

### GPU & Kernels
- Compare GPU kernel authoring across Mojo (structured kernels), Triton (block-level DSL), Julia (CUDA.jl), and raw CUDA. What does each abstract away and what control does each give up?
- How does Mojo achieve multi-vendor GPU portability without CUDA? What are the limitations compared to CUDA-native approaches?
- What performance claims does Mojo make for GPU kernels vs CUDA C++? Are they validated by independent benchmarks?

### Type Systems & Ownership
- Compare Mojo's ownership model (origins, lifetimes) with Rust's borrow checker. Where do they diverge and why?
- How does Julia's multiple dispatch approach compare to trait-based systems (Mojo, Rust) for ML workloads?
- What are the known limitations of JAX's tracing-based approach to JIT compilation? What programs can't be expressed?

### Autodiff
- Compare JAX's composable function transforms (grad/jit/vmap) with Julia's fragmented AD ecosystem (Zygote, Enzyme, ForwardDiff). What are the practical consequences of each design?
- Why does Julia have 5+ competing AD backends? What does this fragmentation tell us about the difficulty of language-level AD?
- How does Enzyme's LLVM-level AD approach differ from source-to-source AD (Zygote) and tracing-based AD (JAX)?

### Strategy & Positioning
- What gaps exist across all these frameworks that no existing tool fills well?
- Which features do Mojo and Julia promise but haven't delivered yet? Where are they behind their roadmaps?
- What does Triton's integration with PyTorch (torch.compile/Inductor) tell us about the viability of standalone ML languages?

---

## 2. ML Compiler Theory

### IR Design
- What are the key design principles behind MLIR's multi-level dialect approach? Why is a single-level IR insufficient for ML?
- How does Cranelift's IR design differ from LLVM IR? What are the implications for compilation speed vs optimization quality?
- Compare TVM's two-level IR (Relax + TensorIR) with XLA's single-level HLO. When does multi-level IR win?

### Optimization Passes
- What are the most impactful compiler optimization passes for ML workloads specifically (not general-purpose code)?
- How do polyhedral compilation techniques apply to tensor operations? What are the limits of the polyhedral model for ML?
- Compare auto-scheduling approaches: TVM's MetaSchedule, Triton's autotuning, and XLA's HLO pass pipeline. Which produces better code and why?

### Fusion & Memory
- What are the different operator fusion strategies used across ML compilers? Classify them by approach and effectiveness.
- How do ML compilers handle memory planning and buffer allocation? What techniques minimize peak memory usage?
- What is the relationship between tiling, fusion, and memory hierarchy optimization in tensor computations?

### Code Generation
- What techniques do ML compilers use to generate efficient code for GPU accelerators? How do they differ from CPU codegen?
- How does the BYOC (Bring Your Own Codegen) pattern work in TVM? What are the integration costs for new hardware backends?

---

## 3. Type Systems & Safety

### Linear Types
- What is the core theory behind linear types? How do they guarantee exactly-once use of resources?
- Compare Wadler's original linear types with Rust's ownership model. Where does Rust diverge from the theory and why?
- What are the practical challenges of adding linear types to a language that wasn't designed for them? What does Rust's experience teach us?

### Effect Systems
- What is the Koka programming language's approach to row-polymorphic effect types? How does it track side effects at the type level?
- Compare algebraic effect systems (Koka, Eff, Frank) with monadic effect encoding (Haskell). What are the ergonomic and performance tradeoffs?
- How are effect handlers compiled efficiently? What does the research say about the runtime overhead?

### Tensor Shape Safety
- What approaches exist for verifying tensor shapes at compile time using dependent types? Compare implementations in Idris, Haskell, and Scala.
- How does PyTorch's named tensor experiment relate to academic work on dimension types? Why hasn't it been widely adopted?
- What is the state of the art in shape algebra — symbolic reasoning about tensor dimensions at compile time?

### Refinement & Region Types
- How do refinement types (Liquid Haskell, Stainless) relate to compile-time tensor shape checking? Could NSL use refinement types for shape verification?
- What is region-based memory management and how does it compare to ownership types for GPU memory safety?
- How do session types enforce communication protocols? Could they be applied to multi-device tensor operations?

### Design Implications for NSL
- Given NSL's M38 linear types milestone, what is the minimal set of linear type features needed for tensor ownership safety without making the language too restrictive?
- For NSL's M51 effect system, which effect system design (Koka-style rows, algebraic effects, or simpler annotations) best fits a compiled ML language?
- For NSL's M49 shape algebra, what compile-time reasoning about tensor dimensions is practically achievable without full dependent types?

---

## 4. Inference Optimization

### Attention Mechanisms
- What are the key algorithmic innovations in FlashAttention 1, 2, and 3? How does each version improve upon the last?
- How does FlashAttention's tiling strategy exploit GPU memory hierarchy (SRAM vs HBM)? What is the IO complexity analysis?
- How does PagedAttention's virtual memory approach to KV-cache management reduce memory waste? Quantify the improvement.

### Speculative Decoding
- Compare speculative decoding approaches: draft models (Leviathan, Chen), Medusa (parallel heads), and EAGLE (feature uncertainty). What are the speedup characteristics of each?
- What determines the acceptance rate in speculative decoding? How does draft model quality affect throughput?
- How does speculative decoding interact with other optimizations like KV-cache sharing and continuous batching?

### Serving Architecture
- How does continuous batching (Orca) differ from static batching? What is the throughput improvement and why?
- Compare the serving architectures of vLLM, TGI, and TensorRT-LLM. What are each system's strengths and bottlenecks?
- How does SGLang's radix attention and prefix caching reduce redundant computation? Quantify the benefits.
- What is disaggregated inference and when does separating prefill from decode improve serving efficiency?

### Quantization
- Compare GPTQ, AWQ, and SmoothQuant for inference quantization. When should each be used?
- What is the current state of FP8 inference? How does it compare to INT8 and INT4 in practice?
- How do Mixture of Experts models change inference optimization requirements compared to dense models?

### Parallelism
- How does GSPMD automatically parallelize ML models across devices? What annotations does it require?
- Compare tensor parallelism, pipeline parallelism, and sequence parallelism for inference workloads specifically (not training).

---

## 5. Frontier Features

### WCET Analysis
- What are the main approaches to worst-case execution time analysis (static analysis, measurement-based, hybrid)? Compare their accuracy and applicability.
- What makes WCET analysis particularly challenging for modern hardware with caches, pipelines, and branch prediction?
- Has WCET analysis been applied to ML inference workloads? What are the unique challenges for neural network execution?
- What tools exist for WCET analysis (aiT, OTAWA, Chronos) and how do they work?

### Zero-Knowledge ML
- What is the current state of zkML? How practical is it for production inference today?
- Compare EZKL, zkTorch, and other zkML frameworks. What model sizes can they handle and at what cost?
- What are the fundamental performance bottlenecks in generating ZK proofs for neural network inference?
- How does the choice of proof system (Halo2, Plonky2, STARKs) affect zkML performance?

### Sparse Computation
- How does the TACO compiler generate efficient code for sparse tensor operations? What is its compilation approach?
- What sparse tensor formats exist beyond CSR/CSC? How does format choice affect performance for different operations?
- How do sparse tensors interact with GPU execution? What are the challenges of sparse computation on GPUs?

### Unikernels & Bare Metal
- What are the practical benefits of unikernel deployment for ML inference? Quantify boot time, memory footprint, and attack surface reduction.
- Compare MirageOS and Unikraft architectures. Which is more suitable as a deployment target for compiled ML models?
- What is the state of bare-metal ML deployment on embedded devices (Raspberry Pi, edge accelerators)?

### Neuromorphic & FPGA
- What is the current state of neuromorphic computing for ML inference (Intel Loihi 2, IBM TrueNorth)?
- How do FPGA-based ML inference approaches (Vitis AI, OpenVINO) compare to GPU inference in latency and power efficiency?
- What programming models exist for spiking neural networks? How mature is the tooling?

### Reproducibility
- What causes non-determinism in GPU-based ML inference? Catalog all sources of non-determinism.
- What approaches exist for ensuring bitwise reproducibility in distributed ML? What is the performance cost?

---

## 6. Performance Theory

### Roofline Analysis
- Explain the roofline model in detail. How do you construct a roofline plot for a specific GPU (e.g., A100, H100)?
- How do you determine whether a kernel is compute-bound or memory-bound? What metrics are needed?
- How does the roofline model extend to account for tensor cores, mixed precision, and hierarchical memory?

### GPU Performance Optimization
- What are the most important CUDA optimization techniques for ML kernels? Rank by typical performance impact.
- How does GPU occupancy affect performance? When is maximizing occupancy the wrong strategy?
- What is the role of shared memory, registers, and L1/L2 cache in GPU kernel optimization?
- How do tensor cores change the performance model? What conditions must be met to utilize them effectively?

### Memory Bandwidth
- What is the theoretical vs achievable memory bandwidth on modern GPUs (A100, H100, H200)? What limits utilization?
- How does memory coalescing affect bandwidth for tensor operations? What access patterns are optimal?
- What techniques exist for hiding memory latency (prefetching, double buffering, warp-level pipelining)?

### Mixed Precision
- What is the performance impact of FP16 vs BF16 vs FP8 on different GPU architectures? Quantify the throughput differences.
- When does mixed precision training/inference hurt model quality? What are the failure modes?
- How do you choose between FP16, BF16, and FP8 for a given workload?

### Profiling & Benchmarking
- How should NVIDIA Nsight Compute and Nsight Systems be used together for kernel optimization? What does each tool reveal?
- What are the MLPerf benchmark categories and what do they measure? How should results be interpreted?
- What common profiling mistakes lead to incorrect performance conclusions?

---

## 7. NSL General (Cross-Cutting)

- Based on the NSL spec, what are the most ambitious language features and which have the highest implementation risk?
- How does NSL's compile-time tensor shape checking compare to approaches in other languages?
- What is the rationale for NSL's `model` keyword vs using classes/structs? What does it enable?
- How does NSL's `grad` keyword approach to autodiff compare to library-based approaches?
- What are the key architectural decisions in NSL's Cranelift-based compilation pipeline?
