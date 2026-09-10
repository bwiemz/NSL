# Roadmap A2 — KIR v2: the kernel IR the hand-written PTX can move onto

**Roadmap criterion:** *"KIR v2", staged so it never blocks feature work:
1. Freeze: no new hand-PTX files; new kernels go through KIR (enforce with
a CI grep gate, exactly like `design_only_enforcement.yml`). 2. Grow KIR to
cover what the hand-PTX actually uses: typed registers with SSA virtual
regs + a linear-scan allocator; `mma.sync`/`ldmatrix`/`cp.async`/`wgmma` as
first-class ops with shape-checked fragment types; shared-memory layouts as
a typed `SmemLayout` (you already have `flash_attention_v2/smem_layout.rs`
— promote it); barriers and async-copy groups as ops with a verifier that
checks commit/wait pairing. 3. Migrate bottom-up: `precision_cast_ptx.rs`
(436) → `matmul_mma.rs` (332) → `moe_kernels.rs` (656) → `cfie_*_ptx.rs`
(≈6.5K) → `fused_linear_ce.rs` (4.3K) → FA v2 last. Gate each migration on
bit-identical PTX or SASS-baseline equivalence so the snapshot tests carry
the proof. 4. Delete FA v1 (`flash_attention.rs`, 8,495 lines) once v2
covers its variants — 45K lines of attention across three files is the
largest duplication in the tree.*

This is a design spec, not a plan: it records what of A2 has already
landed, what the frozen estate actually uses that KIR cannot yet say, the
shape of each addition and the verifier rule that comes with it, why the IR
moves into a leaf crate, what "equivalent" means for a migrated kernel, and
the order the estate migrates in. The roadmap's "bit-identical PTX" gate is
not achievable as written — a KIR-lowered kernel names its registers
differently from the hand text by construction — so the "Proof" section
says what replaces it. Everything below was read from the tree at the time
of writing; line counts are from `scripts/hand-ptx-freeze.sh --list` and
will drift.

## Where it stands

**Step 1 is done.** `ci/hand-ptx-manifest.txt` lists the 71 files that
write PTX into a string (2026-09-02); `scripts/hand-ptx-freeze.sh --check`
(membership by `scripts/hand-ptx-scan.awk`, a string-literal scanner that
ignores comments and `#[cfg(test)]` items) fails CI when a file joins the
set or a listed file stops emitting. It is a gate on the file set, not a
line count, so kernel bug fixes are unaffected.

**Step 2 is half done.** Three PRs landed on top of the freeze:
`kir_verify.rs` (rules 1–4: block shape, SSA, def-before-use under
dominance, operand typing), the async-copy group (`SharedBase`, `CpAsync`,
`CpAsyncCommit`, `CpAsyncWait`; rule 5, the commit/wait discipline per
block) and the tensor-core pair (`LdMatrixX4`, `MmaF16M16N8K16`; rule 6,
fragment typing), each assembled by `ptxas` in CI's cuda lane. Of the
roadmap's step-2 list that leaves: the register model, `SmemLayout`, and
the ISA coverage the estate needs beyond one MMA shape.

**KIR produces no CUDA kernel today.** `KirOp` has 48 variants;
`kernel_lower::lower_kernel_to_ir` (a user `kernel` block) emits 15 of them
and refuses loops and stores; `kernel_lower_fpga` emits the three FPGA
structured ops. `Compiler::compile_kernels` sends Metal and WebGPU through
KIR and sends CUDA to the AST→PTX `KernelCompiler` in `kernel.rs` — a
third path, itself a manifest member. `backend_ptx::lower_kir_to_ptx` is
called only from tests. So "grow KIR to cover the estate" is not an
extension of a working CUDA path; it is the first production use of the
PTX printer.

**The estate is 15.3K lines of PTX text in 71 files**, 9.3K in
`nsl-codegen` and 6.0K in `nsl-runtime`. The runtime share matters for the
design (see "A leaf crate"): `cuda/fused_kernels.rs` (3,538 PTX lines) and
`cuda/kernels.rs` (1,639) are the two largest emitters in the tree, ahead
of `fused_linear_ce.rs` (1,535) and FA v1 `flash_attention.rs` (1,353).
The roadmap-named targets, with what they use:

| File | PTX lines | Instruction families beyond scalar arithmetic |
|---|---:|---|
| `precision_cast_ptx.rs` | 46 | `ld/st.global.b16`, `cvt.rn.{bf16,f16}.f32`, `shl.b64`, a grid-stride loop |
| `matmul_mma.rs` | 25 | `mma.sync.m16n8k16` (predicated variant), per-lane `ld.shared` fragment loads — a fragment library, not a kernel |
| `moe_kernels.rs` | 248 | `bar.sync`, `mma.sync`, `atom`, dynamic shared memory sized by `num_experts`, six kernels |
| `cfie_*_ptx.rs` (7 files) | 1,453 | `bar.sync`, shared-memory reductions, `rem`/`and` lane math; no shuffles, no tensor cores, no `cp.async` |
| `fused_linear_ce.rs` + `cpkd_fused_loss.rs` | 1,998 | online-softmax loops, 16-bit loads, 95 + 19 `.reg` sites |
| `flash_attention_v2/` (25 members) | 2,941 | `mma.sync`, `ldmatrix`, `cp.async` (161 sites), `shfl.sync.bfly`, ping/pong pipelines; `smem_layout.rs` (1,762 lines, 74 `pub fn`) computes every offset |
| `flash_attention.rs` (FA v1) | 1,353 | everything above plus `wgmma` (sm_90), which otherwise appears only in the runtime's `cuda/kernels_hopper.rs` (FA-3: `wgmma`, TMA, `mbarrier`, `setmaxnreg`) |

Two facts shape the whole migration. Registers in the hand estate are
*named* (`%rd_src`, `%f_val`, `%p_done`), not numbered: the 220 interpolated
`%f{i}` sites are the minority. And there is no register helper anywhere —
`register_budget.rs` (FA v2, two copies) is a closed-form *count* checked
against the 255 cap, not an allocator.

## What KIR cannot say yet

Read against the four cast kernels — the smallest member, four straight
grid-stride loops — KIR today falls short in five places, and every larger
member needs all five plus more.

**1. A loop-carried value.** Rule 3 (def-before-use under dominance) makes a
loop unexpressible: the header's `idx` is defined in the entry block and
redefined in the body, which is a rule-2 violation, and no `phi` exists to
merge them (`kir_verify.rs`'s own loop test pins the *refusal*). KIR v2 adds
**block parameters**, not phi nodes: `KirBlock { params: Vec<(VarId,
KirType)> }`, `KirTerminator::Branch(BlockId, Vec<VarId>)` and
`CondBranch(cond, (BlockId, Vec<VarId>), (BlockId, Vec<VarId>))`. A block
parameter is defined at block entry, so rule 3 holds unchanged; a new rule 7
checks every edge passes exactly the target's parameter count with matching
types. Cranelift, which the CPU path already lowers to, uses the same form,
and it has no "phi must be first in the block" ordering invariant to
verify. The PTX printer implements an edge as a parallel copy (`mov`s into
the target's parameter registers), with a scratch register for the one
case where an argument is also a parameter of the same target (the swap
problem). A kernel-block `while`/`for` in `kernel_lower.rs` then lowers
instead of refusing, which is what lets `kernel.rs` retire (Steps, 3).

**2. Integer and bitwise arithmetic, and the rest of the scalar ISA.** No
`Shl`/`Shr`/`And`/`Or`/`Xor`/`Not`/`Rem`/`Min`/`Max`/`Rcp`/`Rsqrt`; `Select`
prints `selp.b32` whatever the type; `WarpShuffle` is hard-wired to
`shfl.sync.down` with width 32. The estate's index math is shifts and masks
(`shl.b64 %rd_off, %rd_idx, 2`), the attention reductions are
`shfl.sync.bfly` and `min`/`max`, and the CFIE samplers are `rem`/`and` on
lane ids in front of shared-memory reductions. These are
ordinary two-operand ops with the rule-4 homogeneity check; the shuffle
becomes `WarpShuffle { dst, val, lane: VarId, mode: Down | Up | Xor | Idx,
width: u8 }` and gains `Vote { mode: Any | All | Ballot }` and
`LaneId`/`WarpId` sources. `mul.lo` stays the only multiply (the ISA-7.0
`mad.lo.u32` ban the estate documents is preserved by never emitting it).

**3. Sixteen-bit memory and rounding.** `Load`/`Store` of an `F16`/`Bf16`
pointee print `ld.global.bf16`, which is not a PTX instruction (memory ops
on 16-bit floats are `.b16`), and `Cast` prints `cvt.bf16.f32` without the
rounding modifier a narrowing float conversion requires — `ptxas` rejects
both, which is why no kernel has exercised them. KIR v2 gives the two
16-bit types a `.b16` register class for memory traffic, and `Cast` carries
a rounding mode (`Rn` default for narrowing floats, exact for widening,
`Rz`/`Rm`/`Rp` on request), printed as `cvt.rn.bf16.f32`. The PTX
`.version` follows the feature set: `bf16` conversions need `7.8`, which
`FeatureSet::BF16_ARITHMETIC` already names but nothing consults.

**4. Vector memory.** `KirType::Vec(T, n)` exists for MMA fragments, but
there is no vector load or store; the runtime's `cuda/strided_copy.rs`
moves its rows with `ld.global.v4` / `st.global.v4`, and the fragment
stores in FA v2 pack two values per store. `Load`/`Store` accept a
`Vec` destination/source and print the `.v2`/`.v4` form with the brace
list; rule 4 requires the pointee to be the vector's element type and
`n ∈ {2, 4}`.

**5. Predicated side effects.** `matmul_mma.rs` has a predicated `mma`,
`kernel_skeleton/pad.rs` a predicated shared-memory store, and the CFIE
samplers guard stores with `@%p`. Predicating a value-producing op is a
partial definition, which breaks SSA; KIR v2 therefore allows `Predicated
{ pred, negate, op }` only around ops with no destination — `Store`,
`AtomicAdd`, `CpAsync`, the barrier family — and the verifier refuses the
rest. Conditional *values* use `Select` or a branch; the predicated
accumulate in `matmul_mma.rs` becomes an `Mma` followed by a `Select` on
each accumulator, which is what `ptxas` produces for either spelling.

Beyond the cast kernels, the estate needs three more things that are
design decisions rather than gaps:

**Register classes and the allocator.** The printer numbers registers by
`VarId`: `%f17` is variable 17 whatever its type, and it declares *every*
class at `max VarId + 1` (`.reg .u32 %r<N>; .reg .u64 %rd<N>; .reg .f32
%f<N>; .reg .pred %p<N>` …), so a kernel with N values declares four to
five N registers; `GlobalId` needs two scratch registers and takes them at
`dst + 1000` with a matching floor on the declaration count. `ptxas` does
the real allocation, so none of this is a correctness problem — it is why
"typed registers + a linear-scan allocator" is on the roadmap at all: the
declared count is the only register-pressure number a kernel exposes, the
`+1000` idiom cannot compose two lowered fragments, and a migrated kernel's
PTX is not comparable with its hand version while every register is
numbered by creation order. KIR v2's printer numbers registers per class,
densely, from a linear scan over the block-order linearization (block
parameters make live-in/live-out explicit, so live ranges are intervals
plus the edge copies), draws its own scratch from the same allocator, and
declares each class at its maximum simultaneously-live count.
`KernelIR::register_pressure() -> PerClass<u32>` exposes that number; the
two `register_budget.rs` closed forms become assertions against it, and
`.maxnreg` / `.maxntid` / `.reqntid` become `KernelIR` attributes the
printer emits.

**`SmemLayout`.** KIR has one flat `shared_mem[N]` and one `SharedBase`. The
estate has named regions at computed offsets (FA v2's `smem_layout.rs`
returns `q_offset(config)`, `kv_offset(config)`, … as bytes into one
`extern .shared` block; MoE sizes its histogram region by `num_experts`)
and one hard invariant recorded there: *never mix a static `.shared`
declaration with an `extern` one* (the sm_120 illegal-address finding).
KIR v2 adds `SmemLayout { regions: Vec<SmemRegion { name, bytes, align,
elem: KirType }>, dynamic: bool }` on the kernel, `KirOp::SharedRegion(dst,
region)` yielding `Ptr(elem, Shared)` at the region's offset, and rule 8:
regions do not overlap, the total fits the target budget (48 KiB static;
the 99 KiB opt-in when `dynamic`), an `LdMatrix` or 16-byte `CpAsync`
address comes from a region aligned to 16, and a kernel is static or
dynamic, never both. `flash_attention_v2/smem_layout.rs` keeps its 74
accessors and gains one `fn layout(&FlashAttentionConfig) -> SmemLayout`
that the accessors are then derived from — the promotion the roadmap asks
for, without touching a caller.

**Barriers and the tensor-core set.** `Barrier` is `bar.sync 0`, and
that is every barrier in the estate: all 248 sites are `bar.sync 0`, no
named barrier and no `bar.arrive`, so KIR keeps the one form and rule 9
is only that a `Barrier` is not predicated. The sm_90 synchronisation
(`mbarrier`, TMA) lives in the two Hopper files and stays out with them
(Non-goals). `Mma` becomes
shape- and dtype-parameterised (`Mma { shape: M16N8K16 | M16N8K8, a_ty:
F16 | Bf16, acc: F32, .. }` with the fragment arrays sized by the shape),
`LdMatrix { count: 1 | 2 | 4, trans }` covers the `x1`/`x2` forms, and the
verifier's rule 6 checks fragment counts and types per shape. `wgmma`
(sm_90) is used by FA v1 and by the runtime's FA-3 kernel and enters KIR
only if an FA v2 variant needs it before v1 is deleted (Steps, 12).
Warp-uniformity of barrier placement is not verified: it would need
divergence analysis, and the estate's barriers are all in uniform control
flow by construction of the emitters.

## A leaf crate

Forty percent of the estate lives in `nsl-runtime`, and after A3 step 5
the runtime cannot depend on `nsl-codegen` (the edge now points the other
way, optionally, behind `cuda`). The runtime's cast kernels show what that
costs today: `cuda/precision_cast_kernels.rs` embeds the four PTX strings
as statics, a "constant materialisation" of the codegen emitter, with a
byte-for-byte parity test in `nsl-codegen` to keep them in step.

So `kernel_ir.rs`, `kir_verify.rs`, `backend_ptx.rs` and `FeatureSet` move
to a new crate `crates/nsl-kir` that depends on nothing in the workspace,
the same shape as `nsl-abi` and `nsl-log`. `nsl-codegen` re-exports it
(`pub use nsl_kir as kernel_ir` keeps every path), and the runtime builds
its kernels with `KirBuilder` at first use — the PTX text is produced in
microseconds, and the runtime already JIT-loads text through
`cuModuleLoadData`. The embedded statics and their parity test go away
with the first migration. The other printers (AMDGPU, Metal, WGSL) stay in
`nsl-codegen`: they are compile-time targets and need nothing the runtime
has.

## Proof

"Bit-identical PTX" is the wrong gate: the hand kernels name registers and
the KIR printer numbers them, so no migrated kernel's text is identical to
its predecessor. What the roadmap wants from the gate — the migration
cannot change what the GPU runs — is established at three levels, and each
migrating PR states which it used:

1. **Normalised-text identity**, for straight-line kernels. `nsl_kir::
   normalize(ptx) -> String` renames every register to its class plus
   first-use ordinal (`%rd_src` → `%rd0`, `%f_val` → `%f0`), strips
   comments and blank lines, and canonicalises the `.reg` declarations.
   The hand text and the KIR text normalise to the same string when the
   instruction sequence is the same — which is the bar for the cast
   kernels, the MoE gather/scatter, and most of CFIE. The PR records the
   normalised diff (empty, or the reviewed instruction-level difference).
2. **SASS equivalence**, for the rest. The workspace-level
   `tests/sass_baselines/` harness (`sass_baseline_helpers.rs`: `ptxas` +
   `cuobjdump`, instruction count within `tolerance`, `spill_bytes` exact)
   already gates PCA tier B. Before a kernel migrates, its hand version's
   baseline is recorded; the KIR version must land within it. The harness
   soft-skips without a device, so this gate runs in the local GPU lane
   (`scripts/gpu-tier.sh certify`), and the PR quotes the numbers.
3. **The kernel's own device tests**, always: the `ci/gpu-cert-manifest.tsv`
   rows for the kernel are unchanged by the migration, and `ptxas`
   assembly of the KIR text runs in hosted CI's cuda lane through the
   existing `*_ptxas.rs` pattern.

The freeze then shrinks: a migrated file is deleted, `--write-manifest`
drops it, and `MIN_MEMBERS` in `scripts/hand-ptx-freeze.sh` is lowered in
the same PR. The three KIR snapshots in `tests/snapshot_tests.rs` are
re-blessed once, when the allocator lands (Steps, 5), with the diff
reviewed as the allocator's specification.

## Steps

Each is one PR, gated on the workspace's existing checks (the CLIF
snapshots, the codegen lib tests, the cuda lane's `ptxas` step) plus the
proof level named. Steps 1–6 build the IR; 7–13 spend it. The estate is
frozen throughout, so nothing here blocks a kernel fix.

1. **`nsl-kir`.** Move the four modules; `nsl-codegen` re-exports; a
   `cargo tree` gate proves the crate has no workspace dependency; the
   three KIR snapshots are byte-identical. *Landed:* `crates/nsl-kir`
   (`kernel_ir`, `kir_verify`, `backend_ptx`, `FeatureSet`), re-exported
   at the historical `nsl_codegen` paths; `crates/nsl-kir/tests/leaf.rs`
   pins the empty `[dependencies]` table; the snapshots did not change.
   three KIR snapshots are byte-identical.
2. **Block parameters.** `KirBlock::params`, terminator arguments, rule 7,
   edge copies in every printer (the non-PTX printers may refuse loops
   with their "unhandled" path until they need them); the verifier's loop
   test flips from pinning a refusal to pinning an accepted loop;
   `kernel_lower.rs` lowers `while`/`for`. *Landed (IR half):*
   `KirBlockParam`, `KirEdge`, `add_block_param`, rule 7, the PTX
   printer's parallel copy with per-class scratch, the AMDGPU / Metal /
   WGSL printers marking edge arguments unhandled, the
   `kir_block_params_ptxas` gate and the `kir_grid_stride_loop_ptx`
   snapshot. Lowering `while`/`for` in `kernel_lower.rs` is SSA
   construction over the block's locals and goes with step 3, where
   `kernel.rs` retires.
3. **Retire `kernel.rs`.** CUDA `kernel` blocks go through KIR like every
   other target; the AST→PTX `KernelCompiler` is deleted (43 PTX lines,
   one manifest member fewer). Proof: normalised-text identity on the
   kernel snapshot tests.
4. **The scalar ISA.** Bitwise, shifts, `Rem`, `Min`/`Max`, `Rcp`/`Rsqrt`,
   typed `Select`, the shuffle modes and votes, lane/warp ids, the `.b16`
   memory class, `Cast` rounding modes, vector loads/stores, predicated
   side effects, PTX version from the feature set. Each op has a rule-4
   case and a `ptxas` test. *Landed* (before step 3, which does not need
   it): every family above, `WarpShuffle` in its struct form with modes
   and width, `Cast` printing the modifier PTX requires with
   `CastRounded` for an explicit mode, `LoadVec`/`StoreVec` with one
   scalar per lane, `Predicated` refused around a definition (rule 8),
   the `.b16` class, `FeatureSet::BF16_ARITHMETIC` consulted for
   `.version 7.8` / `sm_80`, the `kir_scalar_isa_ptxas` gate.
5. **Register classes and the allocator.** Dense per-class numbering by
   linear scan, printer scratch from the allocator (the `+1000` idiom
   goes), `register_pressure()`, the `.maxnreg`/`.maxntid`/`.reqntid`
   attributes; the snapshots re-blessed with the reviewed diff.
   *Landed:* `nsl_kir::regalloc` (`RegClass`, `allocate`, `Allocation`,
   `RegisterPressure`), the printer's rename pass, named `%gid0`/`%gid1`
   scratch, `LaunchBounds` and `max_registers` on `KernelIR`,
   `KernelIR::register_pressure()`; the four KIR snapshots re-blessed
   (dense numbering, `.reg` counts at the live maximum). The two
   `register_budget.rs` closed forms are replaced when their kernels
   migrate (step 12).
6. **`SmemLayout` and the tensor-core set.** Rules 8 and 9;
   `flash_attention_v2/smem_layout.rs::layout()`; `Mma`/`LdMatrix`
   parameterised by shape and dtype. `matmul_mma.rs`'s lane-mapping tests
   move to `nsl-kir` as the specification of the fragment layouts.
7. **Precision casts.** The four kernels as KIR in `nsl-kir`
   (`nsl_kir::kernels::cast`), built by the runtime at first use;
   `precision_cast_ptx.rs`, the embedded statics and the parity test are
   deleted (two members fewer). Proof: normalised-text identity, quoted in
   the PR. This is the end-to-end proof of the pipeline.
8. **MoE.** Six kernels; dynamic `SmemLayout` sized by `num_experts`,
   predicated stores, one MMA tile. Proof: SASS equivalence for the GEMM,
   normalised identity for the rest.
9. **CFIE**, one file per PR in size order (`decode_attention`,
   `kv_quant`, `spec_sampler`, `speculative`, `sample`, `persistent`;
   `grammar` is one line and goes with the first). Proof: normalised
   identity.
10. **Fused loss heads.** `fused_linear_ce.rs`, then `cpkd_fused_loss.rs`.
    Proof: SASS equivalence (the online-softmax loops reorder under
    scheduling).
11. **The runtime kernels**, by family (`strided_copy`,
    `tier_b1_prepass`, then `kernels.rs` and `fused_kernels.rs` in
    slices), each slice one PR. Proof: normalised identity where
    straight-line, SASS equivalence otherwise. `cuda/kernels_hopper.rs`
    (FA-3, sm_90) stays a member, listed with its reason, until the sm_90
    set enters KIR.
12. **FA v2**, by phase directory, tier B.1 and B.2 last; the SASS
    baselines and the two no-spill gates already exist here and are the
    proof. `matmul_mma.rs` and `kernel_skeleton/` are deleted with their
    last caller.
13. **Delete FA v1** (`flash_attention.rs`, both crates) once the v2
    selector covers every variant it still routes to v1; with it,
    `kernels_hopper.rs` is the only `wgmma` left in the tree.

## Measured outcomes

- The freeze manifest goes from 71 members to two (`backend_ptx.rs`, the
  member by construction, and the Hopper kernel until sm_90 lands), and
  `MIN_MEMBERS` follows it down.
- Every CUDA kernel in the tree has a verifier run before `ptxas` sees it,
  and `register_pressure()` replaces the two hand-maintained budget
  formulas.
- One producer of PTX text serves the compiler and the runtime; the
  runtime embeds no kernel strings.
- The recurrence-unrolling and fused-top-k work the roadmap lists as
  blocked on A2 ("a KIR that can express gather/scatter with typed
  layouts") has its IR: loops, vector memory, `SmemLayout` and predicated
  scatter are steps 2, 4 and 6.

## Non-goals

- A general mid-level IR (A6): KIR v2 is an assembler with a verifier and
  an allocator, not an optimiser; no fusion, scheduling or CSE pass is
  proposed here.
- Changing any kernel's numerics: every migration is behaviour-preserving
  under the proof levels above, and a kernel whose hand version is wrong is
  fixed in place first, as a member.
- `wgmma`, TMA, `mbarrier`, `setmaxnreg` and the rest of sm_90 until an
  FA v2 variant needs them; `cuda/kernels_hopper.rs` and FA v1 are their
  only users.
- Warp-uniformity analysis for barriers, and named barriers: nothing in
  the estate uses either.
- The non-PTX printers reaching parity with the PTX printer; they gain an
  op when a target needs it, and refuse otherwise.
