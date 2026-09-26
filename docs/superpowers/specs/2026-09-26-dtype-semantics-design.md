# Tensor dtype semantics before 1.0 — one meaning on every device

**Roadmap criterion:** *Do the CPU/GPU dtype semantic redesign before 1.0.*

This is a design spec, not a plan. It covers four things:

- what a tensor's element type means today on each path;
- what it should mean;
- the decisions that need an owner's call, each with a recommendation;
- the order of steps that gets there, each shippable and gated.

## The one-sentence problem

A program's declared element type is not the type it computes in. It is
not the type the checker thinks it computes in either. And the answer
changes with the device, the allocation path, and whether the tensor has
been to the GPU and back.

## Where it stands

Everything below was read from the tree on 2026-09-25. Line numbers are
approximate.

### Three different answers to "what dtype is `zeros([4])`?"

| Layer | Answer | Where |
|---|---|---|
| Semantic checker | `f64` | `nsl-semantic/src/builtins.rs` ~207-240, `checker/ops.rs` ~230-251 type every creation builtin (`zeros`, `ones`, `rand`, `randn`, `empty`, `full`, `arange`) as `F64` on CPU |
| Runtime, ordinary path | `f32` | `tensor/creation.rs` `tensor_from_shape_list` passes dtype `1`; `nsl_tensor_{zeros,ones,full,rand,randn,arange}` and `nsl_tensor_zeros_on` all produce f32 |
| Runtime, slab path | `f64` | `stmt_assign.rs` `try_compile_slab_tensor` maps the checker's `F64` to runtime tag `0`, so a slab-planned `zeros` is f64 while the same call off the slab is f32 |

`docs/architecture/runtime.md`'s "CPU f64 or GPU f32 by default" is
therefore stale in its first half. `SPECIFICATION.md:51` ("f64 (CPU
default), f32 (GPU/training default)") and `:130` ("automatic f64/f32
conversion on device transfer") describe the design the code drifted
away from.

### What the checker accepts

- `let a: Tensor<[4], f32> = zeros([4])` is an **error**: "annotation is
  `Tensor<[4], f32>`, but value has type `Tensor<[4], f64, cpu>`". The
  same program with `f64` checks clean, and the runtime then builds an f32
  tensor.
- `fp16`/`bf16` annotations fail the same way. `f16` is not a recognised
  name at all (`resolve.rs` ~404 maps it to `Unknown`; only `fp16` works).
- Tensor assignability demands identical dtypes (`types.rs` ~474), **except**
  when either shape is rank-0 or unknown (`types.rs` ~458-461): then it
  returns `true` before comparing dtypes. `zeros(s)` with a non-literal `s`
  is assignable to a `bf16` annotation.
- Model-field initialisers are not dtype-checked.
- Nothing checks whether a (device, dtype) pair is legal: no rule says
  "f64 cannot live on the GPU".

### What the runtime does with mixed dtypes

- **Binary elementwise, CPU** (`cpu.rs` ~140-223): both inputs 16-bit
  gives 16-bit; **either input f32 gives f32 ("f32 wins")**; otherwise f64,
  widening f16/bf16. Mixed inputs are silently narrowed or widened, never
  refused.
- **Unary and reductions, CPU:** the pattern is `if dtype == 1 { f32 } else
  { data_f64() }`, so f16/bf16 hit the accessor's assertion (activation,
  trig, reduction, `ad_ops`, `rotate_half`). CPU matmul is f32 if either
  input is f32, else f64; f16 panics in `data_f64`.
- **Three memory-safety holes** follow from "not f32 means f64":
  - The FBIP in-place arm of `nsl_tensor_exp` (`activation.rs` ~43-53) and
    its sibling unary arms write 8-byte f64 elements into whatever buffer
    is not f32. An f16 buffer is 2 bytes per element, so this writes out
    of bounds.
  - `nsl_fused_elementwise_2` (`cpu.rs` ~426-482) reads `b` with `a`'s
    dtype.
  - `nsl_fused_matmul_epilogue` (`cpu.rs` ~557-640) reads raw memory as f32
    whatever the tag says.
- **Helpers that mint f64:**
  - `create_tensor_with_shape_rs` (`cpu.rs` ~741; sampling uses it and says
    "always f64");
  - `create_tensor_from_f64_data`;
  - `create_scalar_tensor_dtype` (f64 for every dtype except 1);
  - `nsl_tensor_from_custom_dtype`;
  - the Wengert lowering's `promote_to_tensor`, which makes f64 scalars
    while other scalar sites in the same file make f32.

### Device transfer rewrites the dtype

`nsl_tensor_to_device` (`tensor/mod.rs` ~4089):
- **Upload:** f64 becomes f32 through a staging buffer. f32 is copied as
  is. `U16_TOKEN` becomes `I32`. Every other dtype is byte-copied and keeps
  its tag.
- **Download:** f32 **always becomes f64**.

So an f32 tensor that goes CPU → GPU → CPU comes back as f64.
`CHANGELOG.md` records one real bug (`l1_backward`) caused by exactly this.

### Casts do not cast

- `.to(f32)` for a standard dtype compiles to `nsl_tensor_from_custom_dtype`,
  which returns its input unchanged for any tag below 256. Yet the GPU's
  refusal messages tell users to "cast with `.to(f32)`".
- `nsl_tensor_cast` (`precision_cast.rs`) implements f32/fp16/bf16 and
  refuses f64 both as source and as target.

### Weights and checkpoints

- safetensors loads everything as f32, whatever the file says, and saves
  everything as F32.
- `model_save` downcasts f64 staging buffers.
- `model_load` aborts on an f32/f64 mismatch, telling the user to "re-save
  with the same device convention (CPU=f64, GPU=f32)". The runtime enforces
  the convention the checker contradicts.

### GPU

- `assert_gpu_f32` guards roughly 70 kernels, and `dtype_guard_drift` pins
  that every f32-shaped `gpu_*` function has a guard.
- `gpu_matmul_f32` refuses anything else ("mixed-precision GEMM dispatch is
  not implemented").
- 16-bit and FP8 storage exist only behind explicit modes: `sr_bf16`,
  `fp8`, `precision_cast` and the CSHA f16 buffers.

### Interop

- `nsl_tensor_to_desc` reports the **runtime** tag.
- The `@export` wrapper checks only that a tag is recognised, not that it
  matches the declared `ExportDtype`. Meanwhile the generated C header
  advertises the declared dtype.
- A CPU tensor declared f32 is handed out as f64 data (correctly tagged 0)
  whenever it came from a GPU download, the slab, f64 arithmetic, sampling,
  or a Wengert scalar.

### What the tests pin

- `dtype_abi_lock` pins the tag table and `dtype_guard_drift` the GPU
  guards.
- `test_set_element` reads `zeros` as f32, and the safetensors tests assert
  tag 1.
- CPU-vs-GPU tests use 1e-3-class tolerances, and
  `docs/wiki/Testing-Strategy.md` ~115 says "don't use exact equality
  between CPU (f64) and GPU (f32)". Several CPU tests are bit-exact in f64.

## The contract this spec proposes

1. **The declared element type is the storage type, on every device.**
   - A tensor typed `Tensor<S, f32>` holds f32 on the CPU and on the GPU.
   - No layer widens or narrows it implicitly: not creation, not transfer,
     not arithmetic, not checkpoint I/O, not export.
2. **The default floating dtype is `f32`.** It is what the runtime already
   makes, what every GPU kernel computes, and what checkpoints hold.
   - `f64` stays available by annotation or `dtype=` argument, on the CPU.
   - An unannotated creation builtin is `f32` in the checker, the runtime
     and the slab path alike.
3. **Mixed dtypes are refused, and conversion is explicit.**
   - A binary op on `f32` and `f64` is a compile error when both dtypes
     are known, and a typed fatal (`Fatal::UnsupportedDtype`) at run time
     when one is not.
   - The fix the message offers is `.to(dtype)`, and `.to(dtype)` really
     converts for every supported pair. That includes f64, with
     round-to-nearest-even.
   - The CPU `f32 wins` rule is removed.
4. **(Device, dtype) legality is a checked table.**
   - The table says which dtypes each device stores and computes (GPU
     today: f32 compute, plus f16/bf16/fp8 storage behind the existing
     modes; CPU: f32 and f64 compute, plus f16/bf16 storage).
   - A program that moves an f64 tensor to the GPU is refused at compile
     time, with `.to(f32)` offered, rather than silently narrowed.
5. **Transfer preserves the tag.** Upload and download are byte copies of
   the same dtype. `U16_TOKEN → I32` stays, as a documented index-type
   widening, because tokens are not floating data.
6. **Anything reporting a dtype reports the storage type.** That covers
   checkpoints, exports and descriptors. `@export` refuses a tensor whose
   tag differs from its declared `ExportDtype` instead of handing out
   mislabelled memory.

## Decisions that need the owner

Each has a recommendation. The steps below assume the recommendation, and
each step says what changes if the call goes the other way.

1. **Default float dtype: f32 (recommended) or f64.**
   - f32 matches what the runtime, the GPU and checkpoints already do, so
     most programs change nothing.
   - f64 would mean rewriting the creation FFIs, and every GPU upload would
     become a refusal.
2. **Mixed arithmetic: refuse (recommended) or promote by rules.**
   - PyTorch promotes (`f32 + f64 → f64`). NSL's "ignored configuration is
     not a compatibility strategy" line argues for explicit casts, and
     refusing is reversible: a promotion table can be added later without
     breaking programs.
   - If promotion is chosen, step 4 lands a promotion table in the checker
     instead of an error, and the runtime computes in the promoted type.
3. **f64 on the GPU: refuse (recommended) or emulate.**
   - No kernel exists. Emulating by narrowing is what happens today, and
     it is the bug.
4. **CPU half precision: storage-only (recommended) or compute.**
   - Storage-only: f16/bf16 CPU tensors support creation, copy, cast and
     transfer. Every compute op refuses them with a typed fatal naming
     `.to(f32)`.
   - Compute would mean widen-compute-narrow in every op.

## Steps

Each step is one PR, with its own gate. The early steps are behaviour-
preserving or pure bug fixes, so they can land before the owner decides
the questions above.

**Step 0: memory safety (no semantic change).**
- The FBIP in-place arms check the dtype they write, and so do
  `nsl_fused_elementwise_2` (both operands) and `nsl_fused_matmul_epilogue`.
- A mismatch is `Fatal::UnsupportedDtype` instead of an out-of-bounds
  write.
- Gate: a Miri-run test per arm with an f16 input.

**Step 1: `.to(dtype)` converts.**
- Standard-dtype `.to()` lowers to `nsl_tensor_cast`.
- `nsl_tensor_cast` gains f64↔f32 (round-to-nearest-even), f64↔f16 and
  f64↔bf16 on the CPU. The GPU refusals' advice becomes true.
- Gate: a cast matrix test (every supported pair, IEEE corners, bit
  patterns).

**Step 2: transfer preserves the tag.**
- Download stops widening f32.
- Upload of f64 becomes a refusal naming `.to(f32)`. The checker's
  device table makes that a compile-time error in step 5; until then it is
  the runtime fatal.
- `model_load`'s device convention check becomes a tag equality check.
- Gate: a round-trip test that asserts the bytes and the tag.
  `l1_backward`'s regression test stays green.

**Step 3: one default across layers.**
- The checker types creation builtins as `f32`, and accepts `f16` as an
  alias of `fp16`.
- The slab path follows the checker, so both agree on f32.
- The f64-minting helpers (sampling, `create_scalar_tensor_dtype`,
  `from_custom_dtype`, the Wengert scalars) produce the operand's dtype.
- Programs annotated `f64` now get f64 storage. Creation FFIs gain a dtype
  argument, and codegen passes the checker's dtype.
- Gate: a checker-vs-runtime agreement test. It compiles creation calls
  under f32, f64 and bf16 annotations, runs them, and requires
  `runtime tag == checker dtype` for every one, on and off the slab.

**Step 4: mixed arithmetic refuses.**
- The checker rejects mismatched dtypes in binary ops, matmul and compare.
- The rank-0 / unknown-shape bypass in assignability no longer skips the
  dtype comparison: shape and dtype are independent questions.
- The runtime's `f32 wins` branch becomes `Fatal::UnsupportedDtype`.
- Gate: the refusal fixtures, plus a mutation run over the dispatch (a
  re-added promotion branch must fail a test).

**Step 5: device legality table.**
- `nsl-semantic` gains the (device, dtype) table and checks `.to(device)`,
  `@target` placements and model device moves against it.
- Model-field initialisers are dtype-checked like any other assignment.

**Step 6: honest interop.**
- `@export` compares the tag to the declared `ExportDtype`.
- `nslpy` accepts the tags the header can advertise, not only 0 and 1.
- Gate: an export round trip per dtype.

**Step 7: docs and tolerances.**
- Rewrite:
  - `runtime.md`'s "opaque default";
  - `SPECIFICATION.md` §dtypes;
  - the wiki's Runtime-Internals and Testing-Strategy pages.
- Retune the CPU-vs-GPU parity tests that assumed an f64 CPU reference.
  Where both sides are now f32, the right tolerance is the kernel's
  reduction-order bound, not "f64 vs f32". Each retune rides the
  tolerance-audit discipline: a named mutant that the new bound catches.

## Non-goals

- Mixed-precision GEMM and autocast. That is a compute feature, not a
  semantic fix, and it stays behind the existing explicit modes.
- New storage formats.
- A promotion lattice, unless decision 2 goes the other way.

## Risks

- **Step 3 changes programs that were annotated `f64` and relied on f32
  runtime storage.** It is correct but observable: loss curves shift by
  f32-vs-f64 rounding. The CHANGELOG entry must say so, and any golden
  numbers pinned under the old behaviour are re-derived, not widened.
- **Step 4 turns silently-promoted programs into compile errors.** The
  example corpus and stdlib must be swept in the same PR. Each fix is an
  explicit `.to()`.
