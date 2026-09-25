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
cannot change what the GPU runs — is established at four levels, and each
migrating PR states which it used:

1. **Normalised-text identity**, for straight-line kernels whose
   instruction sequences genuinely coincide. `nsl_kir::normalize(ptx) ->
   String` renames every register to its class plus first-use ordinal
   (`%rd_src` → `%rd0`, `%f_val` → `%f0`), strips comments and blank
   lines, and canonicalises the `.reg` declarations. The PR records the
   normalised diff (empty, or the reviewed instruction-level difference).

   **Step 7 found the premise does not hold, and this level was not used
   there.** The KIR lowering of a construct need not pick the same
   instruction as the hand author did for the same semantics: `PtrOffset`
   scales an index with `mul.lo.u64 %rd, %rd, 4` where the cast kernels
   were hand-written with `shl.b64 %rd, %rd, 2`. Register renaming cannot
   reconcile those, and widening the normaliser until it can — teaching
   it that a shift by 2 is a multiply by 4 — would be laundering an
   algebraic identity rather than erasing a naming difference, which is
   exactly the class of change a migration gate exists to show. Use this
   level only after confirming the two sequences match; when they do not,
   use level 3 rather than growing the normaliser.
2. **SASS equivalence**, for the rest. The workspace-level
   `tests/sass_baselines/` harness (`sass_baseline_helpers.rs`: `ptxas` +
   `cuobjdump`, instruction count within `tolerance`, `spill_bytes` exact)
   already gates PCA tier B. Before a kernel migrates, its hand version's
   baseline is recorded; the KIR version must land within it. The harness
   soft-skips without a device, so this gate runs in the local GPU lane
   (`scripts/gpu-tier.sh certify`), and the PR quotes the numbers.
3. **Differential execution**, when the instruction sequences differ but
   the computed function must not. The hand module is frozen in the test
   as a fixture — which is what lets the claim outlive the emitter being
   deleted — and both it and the KIR module are executed on a small
   interpreter for the PTX subset they use, over a set of launch
   geometries, asserting the same output bytes *and*, separately, that
   those bytes match an independent reference implementation. Agreement
   alone would be satisfied by two kernels wrong in the same way.

   The interpreter must reject any mnemonic it does not model rather than
   skipping it: a silently-ignored instruction makes every assertion in
   the suite vacuous. Prove the gate bites by mutation before trusting
   it. `precision_cast_kir_equivalence` (step 7) is the worked example;
   its limitation is that modelling `cvt` in Rust does not prove the
   hardware rounds the same way, so level 4 still carries fidelity to the
   machine.

4. **The kernel's own device tests**, always: the `ci/gpu-cert-manifest.tsv`
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
   construction over the block's locals and went with step 3, where
   `kernel.rs` retired.
3. **Retire `kernel.rs`.** CUDA `kernel` blocks go through KIR like every
   other target; the AST→PTX `KernelCompiler` is deleted (43 PTX lines,
   one manifest member fewer). Proof: normalised-text identity on the
   kernel snapshot tests.
   *Landed:* `kernel_lower.rs` is the one front door — stores, `if`/
   `elif`/`else`, `for ... in range(...)`, `while`, `break`/`continue`, a
   bare `return`, compound assignment, and assignment to a `let`-declared
   local, which a join or a loop header carries as a block parameter
   (structured SSA construction: the locals assigned in an arm or a body
   are found by a pre-scan, bound to fresh parameters at the join/header,
   and every edge passes its values); `compiler/kernel.rs` lowers every
   target through it and prints CUDA with `backend_ptx`; `kernel.rs` is
   deleted and the manifest is 70 members. The proof is NOT text identity:
   the two emitters never shared an instruction sequence (`kernel.rs`
   did its index arithmetic in u64 and shifted, KIR widens a u32 offset at
   the pointer; `kernel.rs` wrote `div.approx.f32`, KIR writes the IEEE
   `div.rn.f32`), so the gate is the reviewed diff instead: seven
   `kernel_block_*` snapshots in `tests/snapshot_tests.rs` (the e2e
   fixtures' kernels and one of each control-flow shape) and
   `tests/kernel_block_ptxas.rs`, which assembles all of them in the cuda
   lane. Two printer bugs surfaced on the way and are fixed: `mul.lo.f32`
   and `div.f32` are not PTX (float multiply has no `.lo`; float division
   needs a rounding mode).
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
   deleted (two members fewer). Proof: differential execution (level 3) —
   normalised-text identity was specified here and turned out to be the
   wrong property, for the reason recorded under level 1. This is the
   end-to-end proof of the pipeline. **Done** (#696); the freeze went
   from 71 members to 69.
8. **MoE.** Six kernels; dynamic `SmemLayout` sized by `num_experts`,
   predicated stores, one MMA tile. Proof: SASS equivalence for the GEMM,
   differential execution or normalised identity for the rest, per level
   1's caveat.
9. **CFIE**, one file per PR in size order (`decode_attention`,
   `kv_quant`, `spec_sampler`, `speculative`, `sample`, `persistent`;
   `grammar` is one line and goes with the first). Proof: normalised
   identity where the sequences coincide, differential execution
   otherwise (level 1's caveat).

   **`decode_attention` and `grammar` done**; the freeze went from 68
   members to 66. `grammar`'s fragment is byte-identical (level 1 with
   nothing to normalise): `backend_ptx::global_byte_array` prints the
   initialized `.global` array. `decode_attention` is level 3, and this
   kernel needed more of the interpreter than step 7's: a cooperative CTA
   (threads run to a `bar.sync`, which releases only when every thread
   waits at it), shared memory, loops, and two thread schedules per launch,
   ascending and descending, so a missing barrier changes the answer under
   one of them. The hand *emitter* is the fixture
   (`tests/fixtures/cfie_decode_attn_hand.rs`, verbatim), since its strides
   are baked per configuration; the KIR module matches it bit for bit over
   four geometries and twelve calls, the shared answer matches
   `cpu_reference`, and deleting any one of the five barriers, nudging any
   baked stride or the softmax scale, or dropping the tail-tile clamp is
   caught. Loosening pass 1's `tok < seq_len` guard is *not* — an
   equivalent mutant: its extra scores land past `tcnt`, which every later
   loop bounds away, and its extra K rows stay inside the pool. Three
   things surfaced:

   - the printer declared no `shared_mem` for a region layout — the
     `SharedRegion` arm named a symbol nothing declared. It prints
     `.shared .align <max> .b8 shared_mem[<total>]` (or the `.extern` form
     for a dynamic layout) now;
   - the hand header paired `.target sm_{N}` with an ISA that cannot name
     every `N` in the GPU table: `ptxas` 13.2 refuses `.version 7.0` with
     sm_86, sm_87 and sm_89 and `.version 8.6` with sm_120. The KIR module targets
     the floor (`sm_70`) and the driver JIT-compiles it forward;
     `tests/cfie_decode_attn_ptxas.rs` assembles it for sm_75 through
     sm_120 and records the refusals. `DecodeAttentionConfig` lost its
     `sm_version`. The other five CFIE emitters shared the convention;
     they keep hand headers and take their ISA from
     `gpu_specs::ptx_isa_for_sm`, which names every target in the table
     (`tests/cfie_ptx_headers_ptxas.rs`);
   - a test that read the base kernel's hand register names
     (`cfie_kv_quant_ptx`'s stride check) reads `kv_strides` instead.

   **`kv_quant` done**; the freeze went from 66 members to 65. Its
   per-layer kernels run the same algorithm as `decode_attention` with a
   different pool, so the builder is shared rather than copied:
   `cfie_decode_attention::build_flash_decode` takes a `PoolLayout`, either
   the uniform f16 pool with a runtime `layer_idx`, or one layer's K and V
   halves at baked byte offsets (`Baked`), each f16 or int8. A baked half
   is `kv_base` (a byte pointer) plus its offset, `Cast` to a pointer of
   the half's element type — KIR's `Cast` is how one pointer type becomes
   another, printed `cvt.u64.u64` — and indexed in elements from there.
   An int8 element is `ld.global.s8`, `cvt.rn.f32.s8` and a multiply by
   the half's `.f32` scale param, the hand kernel's register dequant; KIR
   needed nothing new for it. The base kernel still matches its own gate
   through the refactor. Level 3 again, on the same interpreter, now shared
   (`tests/support/cta_ptx_interp.rs`) and taught `ld.param.f32`,
   `ld.global.s8` (sign-extending — a test pins that), `cvt.rn.f32.s8` and
   `cvt.u64.u64`. The KIR kernels match the frozen hand emitter bit for bit
   for every layer of four mixed-precision pools, and the gate catches the
   same mutations plus a nudged half offset and the K and V scales swapped.
   Two things surfaced:

   - an int8 half with an odd element count
     (`max_tokens * n_kv_heads * head_dim`) put the next f16 half on an odd
     byte, a misaligned 2-byte load the GPU faults on. `validate` refuses
     that layout now (the hand kernel emitted it);
   - the module targets the KIR floor like the base kernel, so
     `QuantDecodeAttentionConfig` lost its `sm_version` too.
     `tests/cfie_ptx_headers_ptxas.rs` still assembles the modules for every
     architecture in the table.

   **`spec_sampler` done**; the freeze went from 65 members to 64. Two
   kernels, the draft greedy sampler (kind 7) and the target prob-row
   writer (kind 8), each one CTA of 128 threads: a cooperative hidden load
   and RMSNorm with a seven-step shared-memory tree reduction, then a
   streaming online-softmax pass over 128-row vocab tiles whose merge runs
   in thread 0 alone, carrying (max, sum, argmax) through the tile loop as
   block parameters. The two kernels are built from the same builder
   sections. That matters beyond tidiness: the engine's self-speculation
   anchor needs the draft's `1 / sum` and the verify row's `ex2(0) / sum`
   to be the same bits, so the two kernels' shared arithmetic order is now
   one piece of code, and a test pins the equality on the interpreter.
   `KirOp::Exp`, `Rsqrt` and an f32 `Div` print exactly the hand kernels'
   `mul` by log2(e) + `ex2.approx`, `rsqrt.approx.f32` and `div.rn.f32`, so
   KIR needed nothing new; the interpreter learned `rsqrt.approx.f32` and
   32-bit integer stores. The KIR kernels match the frozen hand emitter bit
   for bit over five geometries (a single token and feature, one past a
   tile, exactly one tile, a d_model past the block, a ragged three-tile
   vocab) under both schedules, and the gate catches a deleted barrier,
   nudged vocab / d_model / `1 / d_model` / epsilon immediates, a relaxed
   first-max-wins comparison (a planted tie decides it) and each forced
   tail guard. What it does *not* catch is named in the test, with why:
   the barrier after the hidden load (each thread's sum of squares reads
   only what that thread stored), the one after the last reduction step
   (thread 0 alone is active there and reads its own write), and the
   tail-tile clamp (the guard stores -inf past the vocab, and merging a
   -inf leaves the state unchanged). Both redundant barriers stay: removing
   them would change the kernel, which a migration does not.
   `SpecSamplerConfig` lost its `sm_version`, as the other two did.

   **`speculative`, in two slices.** The file holds two unrelated kernels,
   so each gets its own PR, and the file leaves the freeze with the second.
   *The rejection epilogue is done* (the first slice). It is a serial
   thread-0 walk with no shared memory: the acceptance loop, then at the
   first rejection the Leviathan residual's total mass, the empty-residual
   fallback to the drafted token, and the CDF walk. Its xorshift64* PRNG is
   plain KIR — `Shr` / `Shl` / `Xor` / `Mul` on `U64`, a `Cast` to `U32`
   for the top bits and another to `F32` — so KIR needed nothing new. The
   interpreter learned 64-bit xor and shifts (PTX's rule: an amount at or
   past the width gives 0), `cvt.u32.u64`, `cvt.rn.f32.u32`, 32-bit
   integer loads and hex immediates. Every operation the kernel does is
   exact in the interpreter, so the gate compares against the CPU
   reference with bit equality rather than a tolerance, over cases that
   reach every branch (all-accept, a rejection that resamples, a zero and
   a negative draft probability, a dominated drafted token, an empty
   residual, random rows, K = 32, a one-entry vocab) and the zero seed. It
   catches a nudged xorshift amount, multiplier, golden gamma, 2^-24 scale,
   sentinel or vocab stride; a forced draft-probability guard; and a
   dropped residual subtraction or clamp. Two mutants are equivalent and
   named: forcing the residual-mass test (an empty residual walks a CDF of
   zeros, which never selects, so it stores the drafted token as the
   fallback does), and forcing the positive-entry test in the CDF walk
   (the running sum only grows at a positive entry, so the first index
   where it reaches the target is positive unless the draw is exactly 0).
   `RejectionConfig` lost its `sm_version`. *The verify attention kernel
   is done* (the second slice). It is the decode-attention kernel run once
   per tree node, so `build_flash_decode` was cut into the sections it is
   made of — the entry (`begin_flash_decode`), the Q row load, the prefix
   tile loop, one tile (`flash_tile`), and the output publish — and the
   verify kernel is those sections per node plus one more `flash_tile`
   over the appended draft rows, whose score hook turns a score to -inf
   where the node's baked mask bit is clear (`Shr`, `And`, `Select`). The
   decode and KV-quant gates pass unchanged over the refactor. The
   interpreter learned `and.b32/b64` and `selp`. The gate sweeps the
   paper's tree(2,2), scattered masks up to 33 nodes, a chain and a single
   node, prefixes across every tile edge, and GQA/MHA, under two
   schedules, and checks the answer against `cpu_reference_verify`. It
   catches a deleted barrier, a nudged stride, scale or node offset, one
   flipped mask bit, a dropped mask and a dropped prefix clamp. Two
   barriers per node are equivalent mutants and named: the tree tile's
   closing barrier (no pass 1 follows before the publish barrier, and the
   only write between them is to `l`, which pass 3 does not read) and the
   node's closing one (nothing reads `q` after the tree tile's scores
   barrier). `VerifyAttentionConfig` lost its `sm_version`, and the file
   left the freeze (64 → 63).

   **`sample` done.** The fused decode-sample kernel starts with the
   spec sampler's sections (the hidden load and the RMSNorm, now two
   functions so the norm can be optional, and the row dot, over a
   `Common` whose rstd region is a parameter) and draws with the
   rejection kernel's `build_prng_draw`. What is its own: the replace-min
   candidate merge with its min-scans, the greedy argmax, the softmax, the
   nucleus insertion sort and cutoff, the multinomial walk, and the
   grammar hook, which loads its mask byte as `I8` and casts it to `U32`
   (the low 8 bits are what is tested, and sign extension leaves them
   alone). The interpreter learned `ld.u8`, `cvt.u32.s8` and 64-bit
   `selp`. A sampler writes one token, so the gate had to be built to
   see its mutants: an RMSNorm fault or constant only rescales every
   logit, which a sharp softmax ignores and a flat one barely feels, so
   the mutation cases include a top-k program at temperature 2, a hidden
   row whose one large entry (element 127) reaches thread 0 only through
   the last lane of every tree-reduction level, 64-bit seeds (small ones
   leave the xorshift state too small for a low-bit constant nudge to
   reach the draw), and planted ties for the first-wins comparisons. It
   catches every barrier that orders something, nudged RMSNorm, PRNG,
   temperature, top_p and top_k constants, relaxed min-scan, merge and
   argmax comparisons, a dropped vocab guard and a skipped grammar mask.
   The equivalent mutants are named: four barriers (after the hidden
   load, the last tree level, the normalised row and the candidate init;
   the last two only each on its own), the walk's positive-entry test
   (the same argument as the rejection kernel's), and the merge's tail
   clamp (the tail lanes hold -inf, which never beats the list's minimum
   under the strict `>`). `FusedSampleKernelConfig` lost its
   `sm_version`, and the file left the freeze (63 → 62).

   **`persistent` done, and with it step 9.** The persistent decode block
   runs a whole layer's decode step in one CTA: RMSNorm, the Q/K/V
   projections with RoPE, the f16 KV append, attention over `pos + 1`
   tokens, W_o and the residual, RMSNorm again, the silu FFN in 128-row
   tiles, and the second residual. Its attention is the decode kernel's
   own sections: the head loop builds a `FlashCtx` per Q head (the head's
   Q row and its attention-output row live in this kernel's shared
   memory) and runs `prefix_pass` and `publish_output`, which now takes
   the output's address space. Its two RMSNorms start with the spec
   sampler's sum-of-squares tree, now a section of its own
   (`sum_of_squares_tree`); the finish differs (every thread divides by
   `sqrt(mean + eps)` and writes a separate row), so it is this kernel's.
   KIR gained one op, `Exp2`: RoPE's frequency is `ex2(i2 * c)` with no
   `log2(e)` pre-scale, which `Exp` cannot print. The interpreter learned
   `sqrt.rn`, `sin.approx` / `cos.approx` (as `sin` / `cos`),
   `cvt.rn.f16.f32`, `st.b16` and `rem`. The gate is output-rich — a
   residual row and a pool record rather than one token — so the mutants
   show without special inputs: over sequence lengths across every
   attention tile edge, `d_model` past the block, `d_ff` over two FFN
   tiles, `head_dim` equal to the block, 256 Q pairs, GQA and MHA and a
   non-default RoPE base and epsilon, the hand and KIR modules leave the
   same bytes, and the answer matches `cpu_reference` fed the same f16
   weights and history (within the f16 rounding of the appended record
   the kernel attends over). It catches every barrier that orders
   something, nudged pool strides, per-slot, `d_ff`, softmax scale,
   `1 / d_model`, epsilon, RoPE scale and silu constant, and both dropped
   tail clamps. Four barriers are named equivalent mutants: after the x
   load and after the W_o residual (each thread's sum of squares reads
   only the elements it wrote), after zeroing y (the down projection and
   the output read `y` the same way), and RMSNorm2's closing barrier
   (the y-zeroing barrier publishes `xn` just as well; the two are each
   redundant only while the other stays, and deleting both is caught).
   One behaviour changed at a caller: the KIR verifier refuses a static
   shared layout past 48 KB, which the hand emitter would print and
   `serve.rs` would then decline to use, so the module exposes
   `smem_bytes` and serve asks it before emitting. `DecodeBlockConfig`
   lost its `sm_version`, and the file left the freeze (62 → 61). Every
   CFIE kernel is KIR now.
10. **Fused loss heads.** `fused_linear_ce.rs`, then `cpkd_fused_loss.rs`.
    Proof: SASS equivalence (the online-softmax loops reorder under
    scheduling).

    **Proof, revisited: level 3.** Scheduling reorders machine
    instructions, not the PTX's floating-point operations, and the KIR
    builds keep those in the hand kernels' order, so the loss heads are
    proved the way step 9 proved CFIE: differential execution against the
    frozen hand emitters, bit for bit, plus an independent reference and
    mutation tests. That is a statement about every output bit, which an
    instruction count within a tolerance is not; the device suites and the
    `ptxas` gates carry fidelity to the machine, as in step 9.
    `fused_linear_ce.rs` holds twelve kernels (the v1 forward, the
    large-vocab partials and finalize, and the backward, each in f32, f16
    and bf16), so it moves in three slices, one per role with all three
    dtypes from one builder, and leaves the freeze with the last. The hand
    emitters are frozen whole in `tests/fixtures/fused_linear_ce_hand.rs`
    for all three.

    **The v1 forward done** (first slice). `build_forward` builds it for
    every dtype: the storage dtype picks the element type of `x`, `W`,
    `bias` and the shared logits tile, so a 16-bit logit is rounded into
    the tile before the reduction, as the hand kernels did. The tile is a
    dynamic `SmemLayout` (the launcher's `shared_mem_bytes()` covers it).
    KIR needed nothing new; the interpreter learned dynamic `.extern
    .shared` memory, `.s64` loads and compares with negative immediates,
    `lg2.approx`, the bf16 conversions and 16-bit `mov`. The gate sweeps
    the three dtypes over ragged and exact tiles, a tile wider than the
    block, targets in every tile and at lanes past 0, ignored rows and a
    target past the vocab, under two schedules, and checks each row's loss
    and log-sum-exp against an f64 reference. It catches every barrier
    deleted, a nudged vocab, hidden, tile width, tile count, lanes-per-thread
    or ignore index, and a relaxed fill guard or sum-scan bound. Two
    mutants are equivalent and named: relaxing either bound of thread 0's
    max scan, which then reads one lane holding nothing the running max
    does not already cover (a logit an earlier tile of the row wrote, or
    the logit-at-target slot), and `max` is idempotent. One more nudge is
    equivalent and is avoided rather than named: the tile width minus one,
    since it is both the tile's stride and the scans' bound, re-tiles the
    vocab with neither gap nor double count.

    **The large-vocab pair done** (second slice). `build_large_partials`
    (Kernel A, grid `(num_tiles, B*S)`) and `build_large_finalize` (Kernel
    B, grid `(B*S)`) build them for every dtype. The launcher loads both
    from one module, so the printer gained `lower_kir_module_to_ptx`: one
    header covering every kernel's features, the shared block declared
    once (two kernels declaring different blocks are refused, since both
    would be `shared_mem`), then the entries; for one kernel it prints
    exactly what `lower_kir_to_ptx` does. The hand kernels wrote the
    16-bit tail sentinel as a literal (`0xFC00`, `0xFF80`); the KIR fill
    carries the f32 `-inf` through the same `cvt.rn` as the real logits,
    which converts an infinity exactly. The interpreter learned `%ctaid.y`
    and `add`/`sub`/`mul.lo` on `.s64`. The gate launches Kernel A over its
    whole two-dimensional grid and then Kernel B, as the host does, and
    compares all of global memory, partials included, across the three
    dtypes, ragged, exact and one-column tiles, ignored rows and a vocab
    past the routing threshold; it checks loss and log-sum-exp against an
    f64 reference (over storage-rounded logits for the lse; the target's
    logit is recomputed in f32 and not rounded). It catches the barrier
    deleted, every loop of either kernel run one trip long, the relaxed
    vocab guard, every baked constant of either kernel nudged, and a finite
    tail. No mutant is equivalent: the shared tile's pad is poisoned with a
    finite f32, so even the max scan's extra trip shows. Loop shape reaches
    the machine: with Kernel A's two scans tested at the top, `ptxas`
    unrolled the f32 ones whole on sm_90 and sm_120 (161 and 130 registers
    against the hand kernels' 32 and 40), so they are bottom-tested, as the
    hand scans were. Kernel A then takes 31–39 registers on sm_80, sm_90
    and sm_120 in every dtype, where the hand kernels took 160–167 on sm_80;
    Kernel B takes the hand kernels' 32 and 40; nothing spills.

    **The backward done** (third slice); `fused_linear_ce.rs` has left the
    freeze (manifest 61 → 60). `build_backward` builds it for every dtype,
    its loops bottom-tested as the hand kernel's were. Its three scatters
    are `KirOp::AtomicAdd`, and the slice found that op's printer wrong:
    it printed `atom.global.add.f32 %v, [p], %v`, writing memory's old
    value into the value's register, a definition the IR does not have and
    the allocator could not see, so a later read of the value (or a value
    allocated the same register) saw the old memory. `AtomicAdd` returns
    nothing, so it now prints `red`, the reduction without a result. The
    interpreter learned `red.{global,shared}.add.f32` as a read-modify-write
    in the schedule's thread order, which makes f32 atomics differentially
    testable: both kernels add the same values in the same per-thread
    order, so under one schedule their sums agree bit for bit. The gate
    compares all of global memory (`dx`, pre-filled so an ignored row's
    zeroing shows, `dW` and `dbias`) across the three dtypes, a ragged and
    an exact tile, a hidden wider than the block, an ignored row and a
    target past the vocab, and checks the gradient against an f64
    reference. It catches every scatter deleted, every loop run one trip
    long but one, every guard relaxed and every baked constant nudged
    (including the `1` taken off the target's probability). The one
    equivalent mutant is named: one more trip of the tile loop starts past
    the vocab, where the guard turns every lane away. Registers match or
    beat the hand kernel's (28–40 against 32–40, no spills).

    **`cpkd_fused_loss.rs` done**; the step's second loss head has left the
    freeze too (manifest 60 → 59). The KL-CE distillation forward and
    backward are f32 only, so each is one builder with no dtype axis.
    `build_forward` holds the student and teacher logit tiles and the
    student's logit-at-target in one dynamic `SmemLayout` at the hand
    kernel's offsets and carries seven running values through the tile loop
    (three online-softmax families, the teacher's with its KL cross-term).
    Its scans stop at the vocab at the top and at the tile at the bottom, as
    the hand scans did. `build_backward` reuses the fused linear-CE
    backward's shape with a second dot and three probabilities, and
    scatters the student's gradients only, so the ABI still has no
    teacher-gradient output (invariant I-11). The interpreter learned
    `rcp.approx.f32` (the reciprocal of the temperature), modelled as an
    exact `1 / x`. The gate runs both kernels, hand and KIR, under two
    schedules and requires the same bits: the loss and three LSEs, then
    `dx_s` (pre-filled), `dW_s` and `dbias_s`. It checks them against the
    crate's own f64 references (`reference_forward_f64`,
    `reference_backward_f64`) and catches:
    - every barrier and every scatter deleted;
    - every guard relaxed and every loop run one trip long, but those
      named below;
    - every baked constant nudged (vocab, both hiddens, tile width, tile
      count, lanes per thread, the ignore index, the `1` in `1 - alpha`);
    - the `-inf` seeding the maxima replaced by 0. Shared inputs keep every
      tile's max positive, so a 0 seed never wins a `max`; that mutation is
      judged on inputs whose logits are all negative.

    Three mutants are equivalent and named. Relaxing either kernel's tile
    loop adds a trip that starts past the vocab. Relaxing the forward max
    scan's vocab guard reads a lane an earlier tile wrote, which the running
    maxima already cover. The max scan's tile bound is caught, since the
    lane past the student tile is the teacher tile's first. KIR uses 28–33
    registers where the hand kernels used 32–40 (sm_80/90/120), with no
    spills.
11. **The runtime kernels**, by family (`strided_copy`,
    `tier_b1_prepass`, then `kernels.rs` and `fused_kernels.rs` in
    slices), each slice one PR. Proof: normalised identity where
    straight-line, SASS equivalence otherwise. `cuda/kernels_hopper.rs`
    (FA-3, sm_90) stays a member, listed with its reason, until the sm_90
    set enters KIR.

    **Proof, revisited: level 3**, as for steps 9 and 10: the runtime
    kernels loop (a grid-stride walk at least), so they are not
    straight-line, and executing the frozen hand module against the KIR
    one bit for bit says more than a SASS instruction count. Each runtime
    family is described in `nsl_kir::kernels`, as step 7's casts are, and
    the runtime lowers it once, on first use, into a `OnceLock` (the module
    cache keys on the buffer's address).

    **`strided_copy` done** (first slice; manifest 59 → 58).
    `nsl_kir::kernels::strided_copy` builds the four run-copy arms (`run`,
    `run4`, `bcast`, `bcast4`) as one module with the hand module's entry
    names and `(src, dst, offsets, run_len, outer)` signature; the runtime's
    `strided_copy::run_module()` replaces `STRIDED_COPY_RUN_PTX`. KIR needed
    nothing new. The interpreter learned `%nctaid.y` (every launch now
    states it), `ld`/`st` `.v4.f32` with a braced register list, and a
    label on its instruction's line. The gate launches both modules over
    the whole grid `RunPlan::geometry` gives, under two schedules that
    reverse the CTA order as well as the thread order, on runs shorter than
    their block, runs spanning two blocks, fewer y blocks than runs, and
    overlapping, out-of-order source runs, with NaN payloads and `-0.0` in
    the source; it requires the same bytes in all of global memory and the
    exact copy. It catches either bound relaxed, the store dropped, the
    offsets' or f32's element size nudged, the vector arms' unit width
    nudged, and either block index read as 0; no mutant is equivalent. The
    kernels drop to the KIR floor (`sm_70`; the hand module declared
    `sm_80` for nothing it used) and take 14–24 registers where the hand
    ones took 16–24 (sm_80/90/120), with no spills.

    **`tier_b1_prepass` done** (second slice; manifest 58 → 57).
    `nsl_kir::kernels::tier_b1_prepass` builds the X pre-pass (one CTA per
    row: a strided sum of squares, thread 0 adding the other partials to
    its own in thread order through a static `SmemLayout`, then the
    normalised, gamma-scaled row narrowed to f16 in chunks-major order) and
    the W pre-pass (one thread per weight, narrowed to f16 at its
    col-major-within-chunk position), each its own module as before; the
    runtime's `csha_tier_b1_prepass_{x,w}_ptx()` replace the two constants.
    KIR needed nothing new. One instruction changes: the hand X kernel took
    the mean square with `div.approx.f32`, and KIR's f32 division is
    `div.rn.f32`, so the mean is now correctly rounded where it was within
    2 ulp, ahead of an `rsqrt.approx.f32`. The interpreter models every
    approximate form by its exact counterpart, so the gate sees the two as
    one; it learned element-typed `.shared` declarations addressed by name,
    `div.approx.f32` and `cvt.rn.f32.u64`. The gate runs the X pre-pass on
    fewer columns than threads, a ragged three-trip column loop and CTAs
    past the rows, and the W pre-pass on a half block and a ragged last
    block, under two schedules, requiring the same bytes in all of global
    memory, an f64 RMSNorm reference for X and the exact layout for W. It
    catches either barrier deleted, every bound relaxed, the column
    stride, element sizes, the reduction's first partial and the masks'
    `1` nudged, and the block index read as 0; no mutant is equivalent.
    Registers: X 20–24 against the hand kernel's 20–24, W 10–14 against
    10–11 (sm_80/90/120), no spills.

    **`kernels.rs`, the binary arithmetic family** (third slice; the file
    stays a member until its last kernel moves). `nsl_kir::kernels::
    elementwise` builds `nsl_add_f32`, `nsl_sub_f32` and `nsl_mul_f32`, one
    thread per element with the hand kernels' 32-bit index and 64-bit bound;
    the runtime's `{add,sub,mul}_f32_ptx()` replace the three constants.
    Nothing new was needed in KIR or the interpreter. The gate runs sizes
    below, at and past a block, out of place and in place (`c` aliasing `a`,
    as the in-place launcher binds it), over IEEE corners (NaN, infinities,
    signed zeros, subnormals, overflow), under two schedules and with one
    block past the grid, and requires the same bytes as the hand kernels and
    the IEEE result, with nothing past `n` written. It catches the relaxed
    bound, every address's element size nudged, the subtraction's operands
    swapped and the block index read as 0. Registers are the hand kernels'
    12 on sm_80/90/120. `nsl_div_f32` is left out on purpose: it divides
    with `div.approx.f32` (within 2 ulp, and 0 for a divisor whose magnitude
    is in `(2^126, 2^128)`), KIR's f32 division is `div.rn.f32`, and a
    migration does not change numerics, so its move is a separate decision.

    **`kernels.rs`, the unary family** (fourth slice). `nsl_kir::kernels::
    elementwise` builds `nsl_{neg,relu,exp,log,sqrt,abs,sign,sigmoid,sin,
    cos,silu}_f32` and `nsl_clamp_f32` in the hand kernels' instruction
    sequences: `exp` as `ex2.approx` of `x * log2(e)`, `log` as `lg2.approx`
    times `ln 2`, the sigmoid family through `ex2.approx` and `rcp.approx`,
    `sign` as two compares and two selects. The interpreter learned `neg`,
    `abs` and `min` on f32. The gate runs them out of place and in place over
    IEEE corners and requires the hand kernels' bytes and the kernels'
    formula (the approximate instructions modelled exactly), with a sanity
    check against the f64 functions. It catches the relaxed bound, both
    addresses' element size, every baked constant nudged one ulp, the
    clamp's bounds swapped and the block index read as 0. Registers are
    10–12 against 10–12. Two stay hand-written. `nsl_tanh_f32` divides with
    `div.approx.f32`. `nsl_gelu_f32` is the second: the gate's f64 check
    found that it multiplies by `0f3FD9999A`, which is 1.7, where its
    backward (`0f3FD9DB23`) and every description of it use 1.702. So the
    GPU gradient was not the derivative of the GPU forward. Per this spec's
    non-goals, that kernel is fixed in place first, as a member, and moves
    after.
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
