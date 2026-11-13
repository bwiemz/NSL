# Compiler-Scheduled Sparsity (CSS) — Errata

**Companion to** `CSS.pdf`. **Date:** 2026-11-13.

This file lists the statements in the April 2026 CSS proposal that a reader
should not rely on, and says what is true instead. The proposal was checked
in November 2026 against its own arithmetic, the sources it cites, later
literature and vendor documentation, and the NSL repository. Each entry says
how it was checked; where a point rests on search-result excerpts or on
memory of a paper rather than a full reading, the entry says so. Section
numbers are those printed in the PDF.

---

## Part 1 — Errors in the proposal

### E1 — The 1.7× Neural Magic figure is against dense BF16, not dense FP8

**Proposal.** §1.1: "Sparse FP8 kernels achieve 1.7× lower latency and 1.5×
higher throughput vs dense FP8 on Llama 3.1 8B (Red Hat Neural Magic,
2024)", and "Combined 4× FP8 compression + 2× sparsity = 2.5× vs BF16
dense." §2.5 sets parity with "1.7× latency / 1.5× throughput" as the aim,
and the §5.3 table has "Dense FP8 → CSS 2:4 FP8 on H100" at 1.7×.

**Correction.** The source measures sparse FP8 against dense BF16; that is
where 1.7× latency and 1.5× throughput come from. It attributes to sparsity
itself at most about 30% on latency and 20% on throughput. Against dense FP8
the supportable target is therefore at most about 1.3× latency and 1.2×
throughput, not 1.7×. (If the source's 30% is read as 30% lower latency, the
factor is 1 / 0.7 = 1.43×, still well short of 1.7×.)

"4× FP8 compression" is also wrong: FP8 is 8 bits against BF16's 16, which is
2×. The 2.5× is a kernel-level "potential" figure in the source.

**Source.**
https://developers.redhat.com/articles/2024/12/18/24-sparse-llama-fp8-sota-performance-nvidia-hopper-gpus

### E2 — The 2:4 storage layout gives three different metadata sizes, and the 16× claim is 8×

**Proposal.** §3.2 declares `indices: [u4; M * K / 2]` under a comment that
says "2 bits per group of 4 indicating which 2 of 4 are non-zero" and "so
M*K/16 bytes for indices". §2.6: "50% (sparsity) × 8× (IL2 2-bit) = 16×
density reduction."

**Correction.** The declaration, its comment and the hardware disagree:

| Description                         | Bits per dense element | Bytes   |
|-------------------------------------|-----------------------:|---------|
| Declared: M·K/2 entries of 4 bits   |                      2 | M·K/4   |
| Comment: 2 bits per group of 4      |                    0.5 | M·K/16  |
| Hardware: 2 bits per kept value     |                      1 | M·K/8   |

Hardware 2:4 metadata is 2 bits per kept value, which is 4 bits per group of
four, or 1 bit per dense element.

With that metadata, 2-bit values at 2:4 cost 2 × 0.5 + 1 = 2 bits per dense
element. Against dense FP16 that is 16 / 2 = 8×, not 16×. By the same count,
sparse 4-bit storage costs 4 × 0.5 + 1 = 3 bits per dense element, which is
75% of dense 4-bit, not 50%.

**Source.** Recomputed from the proposal's own figures, taking hardware 2:4
metadata as 2 bits per kept value.

### E3 — The PyTorch DINOv2 result does not show what §2.3 says it shows

**Proposal.** §2.3: "dynamically updating the mask during training produces
virtually no accuracy degradation (82.8 → 82.7 on ImageNet top-1) vs static
mask which often drops 1-2 points." §3.5 repeats the figure and says zeroed
positions stay zero because "the update is masked before being applied".
§2.3 sets the mask recomputation interval K between 50 and 500 steps.

**Correction.** The source is misread in three ways.

- The 82.7 model was sparse for 70% of training and was then trained dense.
  The final model is not 2:4. The model that stays 2:4 scores 82.3.
- The mask in the source is recomputed on every step, and a straight-through
  dense gradient updates all weights, including the masked ones. That is the
  opposite of "the update is masked before being applied" and of a K = 50 to
  500 schedule.
- The static-mask figure of "1-2 points" is not in the source.

The end-to-end training gain reported there is 6%. The same post says that
"2:4 sparsity degrades accuracy too much" for large language models.

Every language-model speedup the proposal quotes is for inference; that 6%
is the only training number among its sources.

**Source.** https://pytorch.org/blog/accelerating-neural-network-training/

### E4 — A top-2-of-4 mask cannot use the sparse path in the backward pass

**Proposal.** §3.4: "for each group of 4 elements, keep the top-2 by
absolute magnitude, zero the rest."

**Correction.** The hardware accepts only its first operand as sparse, and a
matrix that is 2:4 along its rows is not 2:4 when transposed. The backward
pass multiplies by the transpose, so a mask built this way gives a weight
whose transpose cannot use the sparse kernel. PyTorch's 2:4 training support
prunes 4×4 tiles so that the result is 2:4 in both directions. The proposal's
training recipe does not address this.

**Source.** Logic. The PyTorch detail was not separately sourced.

### E5 — 4:8 NVFP4 sparsity exists only on datacenter Blackwell, and CUTLASS cannot be used as §3.3 describes

**Proposal.** §3.3: "CSS uses CUTLASS 3.6+ which added NVFP4 + 4:8 sparsity
support", and "The CUTLASS library is linked as a static library in the NSL
binary; the emitted kernel calls are direct function pointers into CUTLASS's
compiled templates." §2.4 selects "4:8 sparse NVFP4 tensor cores" whenever
the target is Blackwell. Reference 6 is "NVIDIA. (2024). Blackwell
Architecture Whitepaper: 4:8 Structured Sparsity in NVFP4."

**Correction.** PTX is NVIDIA's GPU assembly language; `sm_NNN` names a GPU
architecture.

- What the proposal has right. 2:4 sparsity is available from Ampere onward
  through the PTX `mma.sp` instruction (2:4 for f16 and bf16, 1:2 for tf32),
  and FP8 sparse `mma.sp` was added in PTX 8.4. Pair-wise 4:8 is a real
  instruction-set feature: `tcgen05.mma.sp` with `.kind::mxf4` or
  `mxf4nvf4`, PTX 8.6, on `sm_100a`, documented as "matrix A is pair-wise
  structured sparse at a granularity of 4:8". Later PTX versions add a
  SparsityVersion bit where version 0 is 4:8 and version 1 is 2:4.
- The CUTLASS version is wrong. CUTLASS 3.6 added Hopper structured-sparse
  GEMM. Blackwell and NVFP4 support arrived in 3.8, and sparse GEMM for
  `sm_100` came later still.
- It does not apply to this project's GPU. `tcgen05` is a datacenter
  Blackwell instruction (`sm_100a`, `sm_103a`). The repository targets
  `sm_120` (noted in `crates/nsl-codegen/src/fp8.rs`), which has neither
  `tcgen05` nor `wgmma`. PTX 8.7 extended `mma.sp::ordered_metadata` to FP8,
  FP6 and FP4 element types on `sm_120`, but excerpts of the PTX 8.8 notes
  indicate that the block-scaled `mxf4` and `mxf4nvf4` kinds are excluded
  from `.sp` there. "4:8 NVFP4 on Blackwell" is therefore not available on
  the machine this project uses.
- CUTLASS cannot be linked as a static library. It is a header-only C++
  template library that needs a CUDA C++ compiler at build time. NSL is a
  Rust and Cranelift compiler that emits PTX directly.
- The whitepaper cited as reference 6 could not be found.

**Source.**
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html and
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html.
The PTX page could not be fetched past its early sections, so every PTX
statement above rests on search-result excerpts of NVIDIA's documentation,
not on a full read.

### E6 — SparseGPT and Wanda are one-shot pruners with no fine-tuning

**Proposal.** §1.2: "With sparse fine-tuning (SparseGPT, Wanda) it drops
below 1 point". §6.2: "current SparseGPT/Wanda already achieve <1 point
loss".

**Correction.** Both methods prune in one shot and involve no fine-tuning.
In the Wanda paper, 2:4 pruning of LLaMA-7B roughly doubles perplexity, from
5.68 to about 11 (from memory of the paper; not re-checked against the
source). Papers from 2026 report strict one-shot 2:4 collapsing reasoning
accuracy, for example Qwen3 falling from 54% to 15%.

**Source.** The 2026 figures come from search-result summaries, not from
reading the papers.

### E7 — Claims that could not be verified

- §2.2, "This is the Tencent progressive recipe" and "Tencent reports
  1.3-1.8× end-to-end acceleration". Neither could be verified.
- §1.2, "Chinese AI labs use 2:4 more actively than Western labs". This
  could not be verified.

The post behind reference 10 (HPC-AI, 1.27× inference speedup) does exist:
https://hpc-ai.com/blog/explore_Semi-structured_sparcity.

---

## Part 2 — Superseded by later work

- **cuSPARSELt 0.10.** https://docs.nvidia.com/cuda/cusparselt/index.html.
  It lists architectures SM 8.0 through 12.1 and the data types FP16, BF16,
  INT8, FP32, E4M3, E5M2 and E2M1, including block-scaled rows. An official
  library path for sparse FP8 and FP4 on Hopper and Blackwell, `sm_120`
  included, now exists. That supersedes both the "kernel maturity" premise
  of §1.2 and the plan to integrate CUTLASS by hand.
- **Accuracy recovery has moved to learned masks.**
  https://pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/
  (October 2025) uses MaskLLM as its 2:4 reference. There, 4-bit plus 2:4 on
  Llama-2-7B averages 60.3 against 63.8 dense, and recovers to 64.2 only
  with a low-rank adapter. The same post says 2:4 "often remains unsupported
  in most custom quantization kernels" and that Sparse Marlin is Ampere-only,
  so tooling is still the bottleneck. SparseForge
  (https://arxiv.org/abs/2605.06402; found by search, not read in full)
  reports dense-level 2:4 on LLaMA-2-7B after 5B retraining tokens. Magnitude
  masks with a fixed schedule, as in §2.2 and §2.3, are not the current
  recipe.
- **Milder patterns on 2:4 hardware.** Sliding-window decomposition
  (https://arxiv.org/abs/2603.05232; search-level only) maps 6:8-style
  patterns onto 2:4 kernels with a measured 1.33× speedup. That is a better
  accuracy-for-speed point than strict 2:4, and the type design in §2.1 has
  no place for it.
- **Plain 2:4 on datacenter Blackwell.** Through the SparsityVersion bit
  described in E5, `mxf4nvf4` now also supports 2:4, so 4:8 is not the only
  sparse NVFP4 pattern there.

---

## Part 3 — What the code implements

No part of the proposal is implemented: there is no `@sparsity` decorator,
pruning schedule, 2:4 mask computation, compressed 2:4 storage, `mma.sp`
emitter, format selection or composition with other features, and
`Sparse<[…], dtype, "2:4">` does not parse.

Older, adjacent code that a reader might mistake for it:

- **The `Sparse<[shape], dtype, coo|csr|csc|bsr>` type.** It parses and
  resolves (`crates/nsl-semantic/src/sparse.rs`), which is the "scaffolding"
  §1.3 refers to. NSL source cannot use it: there is no constructor, and the
  checker rejects `sparse @ dense`.
- **The runtime sparse library.** `crates/nsl-runtime/src/sparse.rs` (COO,
  CSR, CSC and BSR formats; sparse-times-dense matrix and vector products,
  add, multiply and conversions), the kernels in
  `crates/nsl-kir/src/kernels/spmm.rs` and `spmv.rs`, and the GPU wrappers
  in `crates/nsl-runtime/src/cuda/mod.rs`. None of it is reachable from NSL
  source, and none of these formats is 2:4.
- **Sparse codegen planning.** `crates/nsl-codegen/src/sparse.rs` holds
  kernel selection (including a `PtxMmaSp` variant), the `SparseCostModel`,
  format selection and merge lattices. It has no callers.
- **`@sparse(pattern="2:4"|"block")`.** Reachable, and validated by the
  checker, but it has no effect: the decorated weight stays fully dense and
  the matmul is dense.
- **Weight-aware sparse matmul (milestone M52c).** `try_sparse_matmul` in
  `crates/nsl-codegen/src/expr/advanced.rs`, with the analysis in
  `crates/nsl-codegen/src/weight_aware.rs`. This is the only sparse path NSL
  source can reach, and it is withheld: a run with `-w weights.safetensors`
  uses the dense matmul unless the environment variable
  `NSL_M52C_SPARSE_MATMUL=1` is set (and `--no-sparse-codegen` is not
  given). When reached, it emits a CSR matmul for a 2-D weight that is at
  least 50% near-zero; it is not 2:4.
- **`@layout`.** Listed in `docs/summaries/06-implementation-status.md`, but
  it is a refused decorator (`UNIMPLEMENTED_DECORATORS` in
  `crates/nsl-semantic/src/decorator_registry.rs`).

No shipped model uses any of this.
`docs/summaries/06-implementation-status.md` calls the M50 and M52 sparse
milestones "Production", which is not accurate. `docs/wiki/Glossary.md` used
to describe CSS as existing compiler behaviour; it now says the proposal is
not implemented.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Remove 4:8 NVFP4 and the CUTLASS linking plan from scope.
2. Measure before building. Run cuSPARSELt 0.10 dense against sparse on the
   project's `sm_120` card at the coder models' MLP shapes, forward and
   backward, including prune-and-compress time. Continue only if the
   linear-layer gain is clearly above 1.2×.
3. If it is, emit `mma.sp::ordered_metadata` from the existing MMA emitter,
   prune 4×4 tiles in both directions, and use cuSPARSELt as the test oracle.
4. Set expectations from E1 and E3: single-digit percent end to end for
   training, at most 1.3× for inference, with an accuracy cost that needs
   learned-mask recovery. This ranks below pretraining correctness work.

---

## Part 5 — Implementation defects found in the same review

The review that checked the proposal also ran the code. CSS itself is not
implemented, so this part lists what was found wrong in the adjacent code
that Part 3 describes, as distinct from the proposal's own errors. Fixed
items are described in `CHANGELOG.md` under "Unreleased". The open list is
as of 2026-11-13.

### Fixed

- **A run with `-w weights.safetensors` aborted if the model held a 2-D
  weight at least half zeros.** Such a weight reached the weight-aware
  sparse matmul, which aborted on CPU ("invalid tensor handle") and, by
  reading, was wrong on GPU in operand order, pointers and dtype. That path
  is now withheld unless `NSL_M52C_SPARSE_MATMUL=1` is set
  (`try_sparse_matmul` in `crates/nsl-codegen/src/expr/advanced.rs`), so
  the dense matmul runs. The path itself is not repaired; it is the first
  and seventh items of the open list.
- **A variable was treated as a sparse tensor because of a function's
  name.** Whenever the function that produced a value had "sparse" in its
  name, the value was dispatched as sparse, so a dense result of
  `make_sparse_like()` aborted at its first `+`. Only the type decides now
  (`crates/nsl-codegen/src/stmt.rs`).
- **The Glossary described CSS as existing compiler behaviour.** The entry
  in `docs/wiki/Glossary.md` now says the proposal is not implemented.

### Open, in the adjacent code

- **With `NSL_M52C_SPARSE_MATMUL=1`, the sparse matmul is wrong on GPU in
  three ways.** The call site is `x @ W` but the kernel computes
  `W_sparse @ x`; the CSR arrays are host addresses passed as device
  pointers; the values keep the weight's stored dtype while the kernel
  reads f32. Where: `try_sparse_matmul`, and `nsl_sparse_matmul` in
  `crates/nsl-runtime/src/tensor/arithmetic.rs`. By reading the code only.
- **Duplicate entries in a COO tensor give different results on different
  paths.** `nsl_sparse_to_dense` overwrites with the last duplicate; COO
  and CSR `nsl_sparse_spmm` and the GPU kernels sum them. Where:
  `crates/nsl-runtime/src/sparse.rs`; NSL source cannot reach it. By
  reading the code only.
- **`nsl_sparse_add` and `nsl_sparse_mul` treat a CSC or BSR operand as all
  zeros.** `sparse_to_triples` returns an empty list for those two formats.
  Where: `crates/nsl-runtime/src/sparse.rs`; NSL source cannot reach it. By
  reading the code only.
- **Sparse operations record no gradient.** `compile_sparse_binary_op` in
  `crates/nsl-codegen/src/expr/advanced.rs` emits `nsl_sparse_spmm`,
  `nsl_sparse_add` and `nsl_sparse_mul` as plain calls, not traced calls.
  Unreachable while no sparse value can be constructed. By reading the
  code only.
- **`@sparse(pattern="2:4")` and `@sparse(pattern="block")` are accepted
  and not honoured.** The decorated weight stays fully dense and the matmul
  is dense, with no message. Where: `validate_sparse_decorator`, called
  from `crates/nsl-semantic/src/checker/model.rs`; codegen never reads the
  decorator. Reproduced by running. Right behaviour: refuse it through
  `UNIMPLEMENTED_DECORATORS`, as `@layout` is.
- **An unknown format name in a `Sparse<...>` type is accepted.** Anything
  other than `coo`, `csr`, `csc` or `bsr` silently becomes the format `?`.
  Where: `resolve_sparse_format` in `crates/nsl-semantic/src/resolve.rs`.
  By reading the code only.
- **With `NSL_M52C_SPARSE_MATMUL=1`, the sparse matmul aborts on CPU.**
  `x @ self.w` with a 4×8 weight that is 87.5% zeros stops with "invalid
  tensor handle": the fallback in `nsl_sparse_matmul` passes a raw data
  pointer to the dense matmul as a tensor. Reproduced by running, before
  the path was withheld. No test covers the path. It needs the dense
  weight on the fallback and a test of `x @ W` on both targets.
- **`nsl_sparse_spmm` returns a null tensor for a CSC operand** on CPU.
  Where: `crates/nsl-runtime/src/sparse.rs`; NSL source cannot reach it. By
  reading the code only.
- **`gpu_bsr_spmm_f32` cannot launch for more than 64 output columns, and
  mis-sizes its output.** It asks for min(256, N) × 16 threads per block,
  over the limit of 1024 when N > 64. It allocates M × N while the kernel
  writes whole 16-row blocks, and reads rows of B past K, when sizes are
  not multiples of 16. Where: `crates/nsl-runtime/src/cuda/mod.rs`; not
  reachable from NSL source. By reading the code only.
- **A sparse-plus-sparse or sparse-times-sparse result is cleaned up as a
  dense tensor.** `compile_sparse_binary_op` registers the sparse pointer
  those calls return as an owned dense tensor. Where:
  `crates/nsl-codegen/src/expr/advanced.rs`; unreachable while no sparse
  value can be constructed. By reading the code only.
- **`docs/summaries/06-implementation-status.md` calls the M50 and M52
  sparse milestones "Production" and lists `@layout`.** NSL source cannot
  reach the M50 library, the M52 sparse path is withheld, and `@layout` is
  a refused decorator. By reading the code only.
- **`SparseCostModel` uses placeholder sizes.** It assumes 4-byte indices
  and values; the runtime stores 64-bit indices and f64 values. Where:
  `crates/nsl-codegen/src/sparse.rs`, which has no callers. By reading the
  code only. The CSR and COO kernels in `crates/nsl-kir/src/kernels/` are
  not part of this list: they are tested against an independent reference.
