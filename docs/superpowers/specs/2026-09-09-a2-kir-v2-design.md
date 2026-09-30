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

    **`kernels.rs`, GELU** (fifth slice). With the slope fixed in place
    (`0f3FD9DB23`, 1.702), `nsl_gelu_f32` joins the unary family as
    `x * sigmoid(k * x)`, `k` = `elementwise::GELU_SLOPE`. The fixture
    freezes the fixed hand kernel, and the gate adds it to every unary test.
    Among the nudged constants it catches the slope put back to 1.7. The
    CPU-lane `gelu_slope_drift` gate in `nsl-runtime` now reads the forward
    slope from the KIR module and holds it to the hand-written source-AD
    backward's. Registers are 12/12/11 against the hand kernel's 11/11/10
    (sm_80/90/120). `nsl_tanh_f32` was the family's last hand kernel,
    waiting on the approximate-division decision; it moved with
    `div.approx` (item 11's last entry).

    **`kernels.rs`, the scalar-operand family** (sixth slice).
    `nsl_{mul,add,sub}_scalar_f32` (`c[i] = a[i] op s`, `s` an `.f32`
    parameter) are the binary kernels with the second load replaced by the
    parameter, built as `elementwise::ScalarOp`. The gate runs them over
    IEEE-corner inputs and eleven scalars, from signed zeros through a
    subnormal and the extremes to both infinities and a NaN, out of place
    and in place. It requires the hand kernels' bytes and the IEEE result,
    and catches the relaxed bound, both element sizes, the subtraction's
    operands swapped, a neighbour's operation and the block index read as
    0. Registers are the hand kernels' 10 on sm_80/90/120.
    `nsl_div_scalar_f32` waits with `nsl_div_f32`. Two kernels that look
    like this family do not move yet: `nsl_scalar_mul_add_inplace_f32` and
    `nsl_muon_scale_inv_frob_f32` spell their arithmetic `mul.rn` /
    `add.rn` precisely so ptxas cannot contract a multiply feeding an add
    into one `fma`, and KIR's `Add`/`Mul` print the bare, contractible
    form. They need an explicitly rounded arithmetic form in KIR first, as
    do the source-AD backward and FASE AdamW kernels that use the same
    spelling.

    **KIR's explicitly rounded arithmetic, and its first two users**
    (seventh slice). `KirOp::{AddRn, SubRn, MulRn}` are `Add`/`Sub`/`Mul`
    on `f32`/`f64` printed with `.rn`. They compute the same IEEE result,
    but ptxas never contracts a `.rn` multiply and the add that reads it
    into one `fma`, so each operation rounds on its own, as it would through
    memory. The verifier holds them to homogeneous float types (a new
    `RoundedArithNotFloat`). The metal/wgsl/amdgpu backends lower them as
    the bare forms, and the interpreter treats them as the bare forms,
    since it never contracts. With them `nsl_scalar_mul_add_inplace_f32`
    (`m[i] += g[i] * s`, bit-exact with a scalar multiply then an add) and
    `nsl_muon_scale_inv_frob_f32` (`c[i] = x[i] * (1 / (sqrt(stats[3]) +
    1e-7))`) move, in the hand kernels' instruction order. Their gate
    requires the hand kernels' bytes and the formula with every operation
    rounded. Because the interpreter cannot see a contraction, it pins
    the `.rn` spellings against the hand kernels, and it names the dropped
    `.rn` as the equivalent mutant that pin exists for. It also catches the
    bound, every element size, a neighbour's operation, the stats slot,
    the epsilon, the one and the block index. On the machine, ptxas turns
    the bare `mul.f32` + `add.f32` pair into one `FFMA`, and the `.rn` pair
    into `FMUL` + `FADD`. Assembled and disassembled with `nvdisasm`, both
    KIR kernels issue exactly the hand kernels' floating-point instruction
    sequence on sm_80/90/120 (2 and 44 instructions), and use the same
    registers (10, and 14/13/15). Only the address arithmetic differs.

    **The activation-backward kernels** (eighth slice): nine kernels built
    by `nsl_kir::kernels::elementwise::build_backward` (`BackwardOp`).
    - **Tape-AD**, `nsl_{relu,sigmoid,tanh,silu}_backward_f32`: bare
      arithmetic, so ptxas may contract, as before. Tape-AD tanh's
      `1 - y*y` becomes one `FFMA` on both sides.
    - **Source-AD**, `nsl_{sigmoid,tanh,silu,gelu}_backward_srcad_f32` and
      `nsl_swiglu_gate_backward_f32`: every derivative operation is
      `.rn`, and the sigmoid stays bare.
    - **The gate:** `elementwise_backward_kir_equivalence` requires the
      frozen hand kernels' bytes and the rounded formula, and pins every
      mnemonic count.
    - **SASS:** identical floating-point sequences on sm_75/80/90/120,
      except the SwiGLU gate on sm_80+. There the independent `grad * up`
      `FMUL` is scheduled elsewhere, with the same multiset and no `FFMA`.
      Registers are within two of the hand kernels.
    - **Clamp adjoint:** `nsl_clamp_backward_f32` followed once the
      interpreter gained `and.pred` (`build_clamp_backward`). Its gate,
      `clamp_backward_kir_equivalence`, runs bounds that are ordinary,
      equal, reversed, infinite and NaN. The SASS is the hand kernel's
      `FSETP.GTU`/`FSETP.GE`/`FSEL` core with 12 registers on sm_80/90/120;
      only the address arithmetic differs (`IMAD.WIDE` in place of shift
      and add), with the same instruction count.
    - **RoPE `rotate_half` pair:** `nsl_rotate_half_f32` and the fused
      backward `nsl_rotate_half_neg_f32` (`build_rotate_half`), branching
      on `i % last_dim < half` as the hand kernels do. The gate is
      `rotate_half_kir_equivalence`. The SASS keeps the registers and the
      64-bit remainder call on sm_80/90/120; the address arithmetic
      (`IMAD.WIDE` for `LEA`) adds 2 to 8 instructions.
    - **Tape-AD GELU adjoint:** `nsl_gelu_backward_f32` moved with
      `nsl_tanh_f32` once `div.approx` existed (see below).

    **`div.approx`, and the fused FASE AdamW step** (new-roadmap item 5).
    - **The op:** `KirOp::DivApprox` prints `div.approx.f32`. It is f32
      only; the verifier's `ApproxDivNotF32` rejects any other type. `Div`
      stays the IEEE `div.rn`. The metal/wgsl/amdgpu backends lower it as
      `Div`. This unblocks the kernels the earlier slices left hand-written
      for their `div.approx`: `nsl_div_f32`, `nsl_div_scalar_f32`,
      `nsl_tanh_f32`, `nsl_gelu_backward_f32`, and the three other FASE
      AdamW variants.
    - **The two divisions:** `nsl_div_f32` and `nsl_div_scalar_f32` join
      the binary and scalar-operand families as `BinaryOp::Div` /
      `ScalarOp::Div`, lowered to `DivApprox`. Their frozen hand modules
      join `elementwise_binary_hand.rs` / `elementwise_scalar_hand.rs`, and
      the two family gates now pin `div.approx` against them, name
      `div.approx` → `div.rn` as an equivalent mutant, and catch the
      swapped operands and a neighbour's operation. SASS: the same
      instruction count and floating-point multiset on sm_80/90/120; the
      scalar kernel's sm_80 schedule moves one `FMUL` of the reciprocal
      expansion, and registers are up to two above the hand kernels'.
    - **The kernel:** `nsl_fase_fused_adamw_step_f32` moves to
      `nsl_kir::kernels::optim`. It keeps the hand kernel's rounding (every
      arithmetic op `.rn`, then `sqrt.rn` and `div.approx`) and its
      instruction order. Decoupled weight decay is a branch that joins
      through a block parameter.
    - **The gate:** `fase_adamw_step_kir_equivalence` requires the frozen
      hand kernel's bytes across five hyperparameter sets with and without
      decay, and the AdamW formula rounded per operation. It pins every
      mnemonic count. The interpreter computes `div.approx` as the IEEE
      quotient, so `div.approx` → `div.rn` is a named equivalent mutant.
      The gate catches the bound, the block index, all seven element
      sizes, each hyperparameter's slot, the decay branch inverted, each
      add and multiply, the square root and the division.
    - **SASS:** the floating-point multiset is identical on sm_80/90/120,
      and the sequence is identical on sm_80. On sm_90/120 two independent
      `FMUL`s trade places. Registers are 16/16/18 against 18/18/18.
    - **The multi-tensor step:** `nsl_fase_fused_adamw_multi_f32`
      (`build_fase_adamw_multi`) follows. It reads its θ/m/v/mp bases from
      device pointer tables and its element from the host-built
      `bptab`/`bbtab` block tables (`bbtab[b] + tid`, never `%ntid`). It
      branches around the Phase-B clip pre-scale when `mp_scale == 1.0`
      and zeroes `mp` after the step. Its gate,
      `fase_adamw_multi_kir_equivalence`, runs the flat grid over five
      ragged parameters, one of them empty, and catches the bound, both
      indices, the `bbtab + tid` add, all fifteen element sizes, every
      scalar and table slot, both branches and the zeroing store. SASS:
      the floating-point multiset is identical on sm_80/90/120, the
      sequence identical on sm_80, and registers are 20/20/20 against
      24/24/21; the rest of the difference is address arithmetic
      (`IMAD.WIDE` for shift and add).
    - **tanh and the tape-AD GELU adjoint, with their saturation fixed:**
      `nsl_tanh_f32` (`build_tanh`) and `nsl_gelu_backward_f32`
      (`build_gelu_backward`) compute `tanh(v)` as `(e − 1) / (e + 1)`
      through `div.approx.f32`. That returns 0 for a divisor in
      `(2^126, 2^128)`.
      - **The interpreter** now models that documented range as `a * ±0`.
        Every earlier gate still passes: none divides by anything that
        large.
      - **The hand kernels' bug.** Under that model the gate shows the hand
        tanh returning 0 at 44, `+∞` and NaN (it clamped at 44), and the
        hand adjoint going wrong past `x ≈ 10`.
      - **The fix.** The KIR tanh clamps at `TANH_SATURATION` (43.5) and
        returns a NaN input through an ordered `setp.eq` and a `selp`. The
        adjoint keeps its arithmetic and selects the derivative's limits
        (1, 0) past `|k| = 43.5`.
      - **The gate,** `tanh_gelu_backward_kir_equivalence`, requires the
        hand kernels' bytes where they were right and pins the limits
        past it. It holds both kernels to f64 over the whole range and
        names `div.approx` → `div.rn` as an equivalent mutant: the
        saturation keeps the divisor out of the range.
      - **SASS:** tanh's floating-point sequence is the hand kernel's plus
        an `FSETP.NEU`, the `selp` becoming a predicated quotient. The
        adjoint's is the hand kernel's plus two `FSETP` and two `FSEL`,
        on sm_80/90/120.
    - **SR-BF16, and the ops it needed.** Two additions to KIR:
      - `KirType::U16`: raw 16-bit bits in a `%r` register, loaded and
        stored as `.u16` and widened with `cvt.u32.u16`. PTX lets
        `ld`/`st`/`cvt` take a wider register.
      - `KirOp::Bitcast`: `mov.b32` / `mov.b64` between same-width
        scalars. The verifier's `BitcastWidth` refuses anything else.
      - With them, `nsl_sr_bf16_round_probe` and
        `nsl_fase_fused_adamw_step_bf16sr` moved (`optim.rs`). They share
        one tail, `sr_bf16_bits`: the splitmix64 dither on `u64`, then the
        saturate, ∞ and quiet-NaN paths, joined by a block parameter. The
        f32 single and multi steps now share their AdamW body with it,
        with byte-identical PTX.
      - **The gate,** `sr_bf16_kir_equivalence`, requires the hand
        kernels' bytes and the host's `sr_bf16` reference bit for bit.
        Crafted inputs reach the saturating carry. It kills mutants of
        every hash constant, shift, mask and special value.
      - **SASS:** the step issues the hand kernel's floating-point
        instructions (16/16/18 registers against 18/18/18), and the probe
        keeps the hand kernel's registers.
      - **The multi-tensor SR step,** `nsl_fase_fused_adamw_multi_bf16sr`,
        followed (`build_fase_adamw_multi_bf16sr`). It is assembled from
        parts the other kernels already use: the f32 multi kernel's
        flat-grid header (`multi_header`), the SR step's bf16 load, AdamW
        body and tail, plus a `u64` counter table. Its gate,
        `fase_adamw_multi_bf16sr_kir_equivalence`, requires the hand
        kernel's bytes and the host reference at each parameter's own
        counter. It also requires that every parameter is byte-identical to
        the per-parameter kernel launched on it alone, which is the
        contract the batching entry relies on.
      - **SASS:** the same floating-point and memory instructions as the
        hand kernel, in 20/20/22 registers against 22/22/24.
      - Every FASE AdamW kernel is now KIR.
    - **`fused_kernels.rs`, data movement** (the first slice there).
      `nsl_bias_add_f32`, `nsl_gather_dim_f32`, `nsl_strided_copy_f32` and
      `nsl_slice_f32` are in `nsl_kir::kernels::data_movement`. The strided
      walk is a loop with `(dim, remaining, src_offset)` block parameters.
      The interpreter gained `cvt.rzi.u64.f32`.
      - **The gate,** `data_movement_kir_equivalence`, requires the hand
        kernels' bytes and formulas. It covers gather indices that are
        fractional, negative, NaN or out of range, and transposed,
        broadcast, zero-stride and 0-d views, and it kills the index
        arithmetic's mutants.
      - **SASS:** registers are equal or fewer.
    - **`fused_kernels.rs`, the 2-D-block row lookups.**
      `nsl_embedding_f32`, `nsl_embedding_i32idx`, `nsl_gather_f32` and
      `nsl_gather_i32idx` are in `nsl_kir::kernels::lookup`. The index is
      truncated from f32 or sign-extended from i32 (`cvt.u64.s32`), and the
      gather bounds it unsigned.
      - **The interpreter models two-dimensional blocks.** `run_cta_2d`
        numbers a block's threads `x + y · ntid.x`. The warps and the
        schedules follow that number, and each thread reads its `%tid.x`,
        `%tid.y` and the block's `%ntid.y`. `run_cta` is the one-row case,
        so no earlier gate changed. It also gained `ld.s32`, `cvt.s64.s32`, `cvt.u64.s32` and
        `cvt.rzi.s64.f32`.
      - **The gate,** `lookup_kir_equivalence`, requires the hand kernels'
        bytes and the lookup formula. It runs on a 16 × 16 block and on an
        8 × 4 one, so that `%ntid.x` and `%ntid.y` differ. It kills mutants
        of the bounds, the 2-D index, the element sizes, the adds and
        strides, and the f32 truncation. The i32 index read zero-extended
        is named as an equivalent mutant.
      - **SASS:** the same memory and compare instructions. Registers are
        up by at most two (16 against 14, 14 against 12). The difference is
        the `IMAD.WIDE` address arithmetic, where the hand kernels' `shl`
        gave `LEA`.
    - **`fused_kernels.rs`, the embedding backward family.**
      `nsl_embedding_bwd_{f32,i32idx}` (the `red.global.add` scatter) and
      `nsl_embedding_bwd_det_{f32,i32idx}` (the per-output loop over the
      positions) are in `nsl_kir::kernels::embedding_bwd`.
      - **Index and bounds.** The index is widened signed. The atomic
        kernels keep the hand kernels' `idx < 0` guard, then bound `vocab`
        on the index's bits, unsigned. Given the guard, that agrees with the
        hand kernels' signed bound.
      - **Loop shape.** The deterministic loop's conditional branches carry
        no block arguments: a hit and a skip block join at a latch. With a
        copy-carrying edge on the match test, ptxas neither if-converted
        nor unrolled the loop. Now it does both, as for the hand kernel.
      - **The gate,** `embedding_bwd_kir_equivalence`, requires the hand
        kernels' bytes on 16 × 16 and 8 × 4 blocks. The output starts
        non-zero, so an add is told from a store. The atomic sums follow the
        schedule's order and the deterministic ones follow the positions. It
        kills mutants of the bounds, the 2-D index, the address arithmetic,
        the negative guard's boundary, the f32 truncation, the atomic add,
        and the loop's match, step, start and accumulate. Two mutants are
        named as equivalent: the i32 index zero-extended, and the atomic
        guard removed outright (the unsigned bound rejects a negative index
        too).
      - **SASS:** the same memory, conversion and atomic instructions as the
        hand kernels. Registers are 14 against 12 for the atomic pair, and
        within ±4 of the hand kernels for the unrolled deterministic pair.
      - **Still hand-written:** the uncalled `nsl_scatter_add_f32`.
    - **`fused_kernels.rs`, the fused RMSNorm gamma backward.**
      `nsl_rmsnorm_rinv_rows_f32` (a thread per row, `1 / sqrt(Σ x² /
      cols + eps)`) and `nsl_rmsnorm_dgamma_f32` (a thread per column,
      `Σ_i (dy · x) · rinv[i]` in row order) are in
      `nsl_kir::kernels::rmsnorm_dgamma`. Every float operation is
      correctly rounded: `fma.rn`, `cvt.rn.f32.u64`, `div.rn`, `add.rn`,
      `sqrt.rn`, `mul.rn`. The dgamma loop carries the element offset and
      steps it by `cols`, as the hand kernel stepped its pointers. Forming
      `i · cols + j` each trip cost 69 more instructions on sm_80.
      - **The gate,** `rmsnorm_dgamma_kir_equivalence`, requires the hand
        kernels' bytes and the formulas, bit for bit, over single-row,
        single-column and ragged shapes with signed zeros, on 256- and
        32-thread blocks under two schedules. It kills mutants of each
        bound, the block index, every element size, add and stride, the
        step, the start, and every float operation.
      - **SASS:** the same loads, `FFMA`, `FMUL` and `FADD` as the hand
        kernels, each loop unrolled 8× as before. Registers: rinv 22/22/24
        against 20/18/22, dgamma 30/32/38 against 31/31/32 (sm_80/90/120).
        With 256-thread blocks, neither costs occupancy: sm_120's 1,536
        threads per SM hold six blocks either way. Instruction counts are
        15–23% higher, from per-trip address arithmetic.
    - **`fused_kernels.rs`, integer dequantization.**
      `nsl_dequant_int8_per_head_f32`, `nsl_dequant_int8_per_token_f32` and
      `nsl_dequant_int4_per_group_f32` are in `nsl_kir::kernels::dequant`.
      The int8 pair loads `s8` through `KirType::I8`. For int4's packed
      nibbles KIR gained `KirType::U8`, a byte in a `%r` register (`.u8`,
      `cvt.u32.u8`). The interpreter gained `cvt.rn.f32.s16` (the hand
      kernels' widening) and `cvt.u32.u8`.
      - **The gate,** `dequant_kir_equivalence`, requires the hand kernels'
        bytes over every byte value and IEEE-corner scales. It checks the
        formulas, and that int4's result is one `fma` rounding, not two.
        It kills mutants of the scale index, the signed widening and every
        part of the nibble unpack. `cvt.u32.u8` → `.s8` is named as an
        equivalent mutant, since the nibble masks remove the sign.
      - **SASS:** the same registers and floating-point instructions as
        the hand kernels on sm_80/90/120.
      - **The fp8 E4M3 decoder,** `nsl_dequant_fp8_e4m3_f32`, followed
        (`build_fp8_e4m3`). It builds normals by bit assembly, zero and
        subnormals as `m · 2^-9` with the sign OR-ed back in, and the
        quiet NaN for `S.1111.111`. It uses `U8` and `Bitcast`.
        - **Gates:** `fp8_e4m3_kir_equivalence` requires the hand kernel's
          bytes and the restated OCP value for all 256 codes, and kills
          the mutants of every field constant, both branches and each OR.
          Widening the sign mask from 1 to 3 is named as an equivalent
          mutant. `nsl-runtime`'s `fp8_e4m3_dequant_interp` now runs the
          KIR module the runtime launches, against the OCP reference and
          the CPU decoder.
        - **SASS:** registers 10/10/10 against 10/9/12.
    - **`fused_kernels.rs`, the sparse matrix-vector products.**
      `nsl_csr_spmv_f32` and `nsl_coo_spmv_f32` are in
      `nsl_kir::kernels::spmv`. CSR is a thread per row over `u32` row
      pointers and column indices, `fma.rn`-accumulated from `+0.0` in
      nonzero order. COO is a thread per nonzero over `i64` indices, adding
      an unfused `mul.rn` product into `y` with `red.global.add.f32` (the
      hand kernel's `atom`, whose result it never read). The CSR loop exits
      to a block its header dominates, so the exit edge carries no copies.
      - **The gate,** `spmv_kir_equivalence`, requires the hand kernels'
        bytes over empty, long and repeated rows, on 256- and 32-thread
        blocks under two schedules, and, on exact data, the fused CSR sum
        and the COO atomic update of a non-zero `y`. It kills mutants of
        each bound, the block index, every element size (4 against 8 bytes
        either way) and 64-bit add, the CSR `+ 1`, step, start and
        `fma`, the COO product and the atomic add read as a store.
      - **SASS** (sm_80/90/120): the same loads, `FFMA`/`FMUL` and atomics
        as the hand kernels, the CSR loop unrolled 8× as before. Registers
        are 0–4 more (CSR 28/26/28 against 24/24/26, COO 18/14/16 against
        16/14/14).
    - **`fused_kernels.rs`, dropout.** `nsl_dropout_f32` is in
      `nsl_kir::kernels::dropout`: the 32-bit multiply-xorshift hash of
      `seed + i`, a strict unsigned threshold compare, and two `selp`s
      (the scale factor and the mask).
      - **The gate,** `dropout_kir_equivalence`, requires the hand kernel's
        bytes and the restated hash's decisions. A pair of thresholds,
        `hash(e7)` and `hash(e7) + 1`, makes every hash mutant a
        deterministic kill: a changed hash lands below the first or at or
        above the second.
      - **SASS:** the same instruction mix. Registers are 12/16/14 against
        12/13/12, with no occupancy effect at 256 threads.
    - **`fused_kernels.rs`, the deterministic sums.**
      `nsl_det_global_sum_f32` and `nsl_det_sum_dim_f32` are in
      `nsl_kir::kernels::det_sum`: one thread per result (the launch's one
      thread, or `%ctaid.x`), adding in ascending index order from `+0.0`.
      Each add is `add.rn.f32`, where the hand kernels' was `add.f32`: the
      rounding is the same, and a multiply can no longer contract into it.
      The loop leaves its header for a block the header dominates, so the
      exit edge carries no copies.
      - **The gate,** `det_sum_kir_equivalence`, requires the hand kernels'
        bytes and the ascending sum, bit for bit, over empty and ragged
        extents with order-sensitive data, signed zeros and an infinity, in
        both block orders. It kills mutants of each bound, the block index,
        the quotient and remainder, every element size, add and stride, the
        step, the accumulator's start (`-0.0` and `1.0`) and the add.
      - **SASS:** the same 8× unrolled loop (8 `LDG`, 8 `FADD`) as the hand
        kernels on sm_80/90/120. Registers are equal on sm_90 and sm_120 and
        two more on sm_80 (16 against 14; 30 against 28), on a one-thread
        block.
      - **The short-axis per-dim sum.** `nsl_sum_dim_short_f32` is the same
        loop (`DetSumOp::DimShort`), with the thread's global index as the
        output on 256-thread blocks. The gate runs it on 256- and 4-thread
        blocks, which kills `%tid.x` and `%ntid.x` mutants. Its SASS is the
        hand kernel's 8× unrolled loop; registers are 32/28/30 against
        27/26/28, with no occupancy change at 256 threads.
    - **`fused_kernels.rs`, the batched Muon Newton-Schulz kernels.**
      `nsl_muon_batch_{mom,sumsq,pack,poly,update}_f32` are in
      `nsl_kir::kernels::muon_batch`. The matrices are reached through
      `u64` pointer tables. Every float operation is `.rn`, so none fuses.
      - **Loop shape.** The reduction hoists the `nest` test out of its
        loop into two straight-line loops. Each loop leaves through an exit
        block whose incoming edge carries no block arguments. When the exit
        edge's copies were printed inline in the loop header (`@!p bra
        else; mov ...; bra out; else: bra body`), ptxas did not unroll the
        loop. With the copies out of line, it unrolls it 4× as it did the
        hand kernel's loop. The `muon_batch` unit tests pin that shape.
        The printer now places such copies out of line for every
        conditional edge (see "The printer's conditional edges" below).
      - **The gate,** `muon_batch_kir_equivalence`, requires the hand
        kernels' bytes and formulas over batched, scattered matrices,
        square, wide and tall, with both flags. It kills mutants of every
        bound, block index, element size and rounded float operation, the
        flags, the transpose, the diagonal and the reduction's stride, tree
        and barriers. Poly's quotient read as a remainder is named as an
        equivalent mutant.
      - **SASS:** the same float and memory instructions as the hand
        kernels, with registers equal or fewer. The exception is the
        reduction: its two loops are each unrolled, and it uses fewer
        registers (25–28 against 30–38).
    - **`fused_kernels.rs`, the GPU cross-entropy backward pair.**
      `nsl_ce_bwd_count_f32` and `nsl_ce_bwd_finish_f32` are in
      `nsl_kir::kernels::ce_bwd`. The count is one 256-thread block. Each
      thread counts the valid targets (`t >= 0`) at a stride of 256, a
      `u32` shared-memory tree sums the counts, and thread 0 writes
      `max(count, 1)` as f32. The finish is a thread per element, with
      `div.u32` splitting the index into row and column. It writes 0 for
      an invalid row, and otherwise `(sm - [j == t]) · (go / denom)` with
      `sub.rn`, `div.rn` and `mul.rn`. `gop` is read only when the scale
      comes from the device (the host passes null otherwise). A target is
      loaded once as `u32` and read both ways, f32 truncated by
      `cvt.rzi.s32.f32` and s32 by `Bitcast`, and a `selp` picks one. The
      hand kernels branched between two loads. The interpreter gained
      `cvt.rzi.s32.f32`, `cvt.u32.s32`, `setp.*.s32` and `mov.s32`.
      - **The gate,** `ce_bwd_kir_equivalence`, requires the hand
        kernels' bytes and the restated formulas. It covers f32 targets
        with fractions, `-0.5`, NaN and out-of-range values, and s32
        targets down to `i32::MIN`. The count runs under all four
        schedules. The finish runs on 256- and 32-thread blocks with the
        device scale and the immediate one, where a null `gop` faults if it
        is read. It kills mutants of each bound, the stride, the tree's
        half and halving, both barriers, the valid test (strict or
        unsigned), the reading's choice and flag, the increment, the clamp,
        the row and column split, the one-hot test, its `1` and
        subtraction, the scale's source, the divide, the multiply, the
        invalid rows' `0`, every element size and every 64-bit add.
      - **SASS** (sm_80/90/120): the same `LDG`, `LDS` and `BAR` counts
        and float instructions as the hand kernels. Registers are equal or
        fewer (count 17/20/19 against 20/22/20, finish 17/18/16 against
        17/18/18).
    - **`fused_kernels.rs`, the shared-memory tree reductions.**
      `nsl_global_sum_f32`, `nsl_sum_dim_f32` and `nsl_max_dim_f32` are in
      `nsl_kir::kernels::block_reduce`: one 256-thread block per result
      (one block for the global sum, `%ctaid.x` per output for the per-dim
      pair). Thread `k` folds `k, k + 256, …` from the identity (`+0.0`,
      or `-inf` for the max). A 256-entry `f32` shared buffer then takes a
      tree (`s[k] ⊕= s[k + h]`, `h = 128 … 1`, a barrier per level), and
      thread 0 writes `s[0]`. The sums add with `add.rn.f32` (the hand
      kernels' `add.f32` rounds the same), and the max is `max.f32`. The
      per-dim loop carries its element offset and steps it by
      `256 · inner`, where the hand kernel formed `k · inner` on each trip.
      - **The gate,** `block_reduce_kir_equivalence`, requires the hand
        kernels' bytes and the restated tree order, bit for bit. It runs
        under all four thread schedules and both block orders. The data
        covers empty, single and ragged extents longer than two strides,
        order-sensitive values, signed zeros, NaNs for the max, an
        all-`-0.0` extent, and a spike only the second trip reaches. It
        kills mutants of every bound, the block index, the quotient and
        remainder, every element size, 64-bit add and multiply, the
        stride, the tree's half, halving and partner, both barriers, both
        combines and the identity. Two mutants are named as equivalent:
        every thread writing the result (each stores the same final
        `s[0]`), and the max's stride halved (each value is read twice,
        and `max.f32` is idempotent).
      - **SASS** (sm_80/90/120): the same 8× unrolled loop (8 `LDG`), 3
        `LDS`, 3 `BAR` and 9 `FADD`/`FMNMX` as the hand kernels. Registers
        are within two of them (global sum 17/22/20 against 16/20/20,
        per-dim 28/27/32 against 28/29/32 and 28/27/30 against 28/29/30),
        all at or under 32, so occupancy at 256 threads is unchanged.
    - **`fused_kernels.rs`, the f64 sum of squares.**
      `nsl_sum_sq_f64_acc_f32` (gradient clipping's norm) is in
      `nsl_kir::kernels::sum_sq`. It is grid-strided on 256-thread blocks,
      with the stride `%ntid.x · %nctaid.x` (KIR's `GridDim`). Each value is
      widened to f64 and square-accumulated with one `fma.rn.f64`. An f64
      shared-memory tree follows, and thread 0 writes the block's partial
      to `out[%ctaid.x]`. The interpreter gained `%nctaid.x` (a
      `Launch::nctaid_x` field; 0, the default for every other gate, makes a
      read panic) and f64 accumulation: `0d` immediates, `mov`/`ld`/`st` of
      `.f64`, `cvt.f64.f32`, `add(.rn).f64` and `fma.rn.f64`.
      - **The gate,** `sum_sq_kir_equivalence`, launches whole grids: the
        runtime's `clamp(ceil(n / 256), 1, 256)`, a single block, and three
        blocks that do not divide the work. They run under all four thread
        schedules and both block orders, over values from `2^-140`
        (subnormal) to `2^100`, whose squares leave f32's range at both
        ends. It requires the hand kernel's bytes and the restated per-block
        fold and tree, bit for bit. It kills mutants of the bound, the
        stride and its factors, both block indices, the widening, the fused
        square-accumulate, the tree's half, halving, partner and add, both
        barriers, the identity, every element size and every 64-bit add.
      - **SASS** (sm_80/90/120): the same one `LDG`, `DFMA`, `DADD`, `F2F`,
        three `LDS` and three `BAR` as the hand kernel. Registers are 16/16/18
        against 16/14/15, so there is no occupancy change at 256 threads.
    - **`fused_kernels.rs`, the tensor statistics.**
      `nsl_tensor_stats_f32` is in `nsl_kir::kernels::tensor_stats`: one
      256-thread block writes `out[0..4] = [min, max, Σx, Σx²]`. Thread `k`
      folds `k, k + 256, …` into four partials from `+inf`, `-inf`, `+0.0`
      and `+0.0`. Four shared regions of 256 `f32` then take one tree
      (`s[k] ⊕= s[k + h]`, `h = 128 … 1`, a barrier per level, the four
      statistics in order at each step), and thread 0 writes the four
      `s[0]`. The order of every combine is the hand kernel's. The sums use
      `add.rn.f32` and the square `mul.rn.f32`; the min and max are
      `min.f32` and `max.f32`, as before.
      - **One deliberate change on hardware.** ptxas contracted the hand
        kernel's `mul.f32` + `add.f32` into an `FFMA` (8 per unrolled loop
        on sm_80/90/120), so its `Σx²` rounded once per element where the
        PTX reads twice. `nsl_muon_batch_sumsq_f32` squares and adds with
        two roundings and is documented as bit-identical to this sum (the
        sequential Frobenius path), which held only in the PTX. With the
        rounding explicit, it holds on hardware, and the result no longer
        depends on ptxas.
      - **The gate,** `tensor_stats_kir_equivalence`, requires the hand
        kernel's bytes on the interpreter (which reads `mul.f32` and
        `add.f32` as two roundings) and the restated order, bit for bit.
        It runs under all four thread schedules. The data covers empty,
        single and ragged extents past two strides, full 24-bit
        significands, signed zeros, NaNs, infinities, a subnormal,
        squares out of range at both ends, all-positive and all-negative
        extents, and a two-value extent on which a fused square rounds
        differently. It kills mutants of the bound, stride, the tree's
        half, halving, idle test and partner, both barriers, every region
        offset, element size, 64-bit add and output slot, every combine,
        the square, each identity, and the square fused into its
        accumulate (`fma.rn.f32`). Three mutants are named as equivalent:
        every thread writing the result, and the first output slot's
        element size and address add (its offset is `0 · 4`).
      - **SASS** (sm_80/90/120): the same 8× unrolled loop (8 `LDG`), 8
        `STS`, 12 `LDS`, 3 `BAR`, 4 `STG` and 18 `FMNMX` as the hand
        kernel. The 8 `FFMA` become 8 `FMUL` + 8 `FADD`. Registers are
        22/26/34 against 20/20/24; it is a single-block kernel, so
        occupancy does not bear on it.
    - **`fused_kernels.rs`, the fused linear cross-entropy finalize.**
      `nsl_lce_finalize_f32` is in `nsl_kir::kernels::lce_finalize`. It is
      grid-strided over rows (stride `%ntid.x · %nctaid.x`) and writes
      `lse = m + ln s` and `loss = t >= 0 ? lse - tl : 0` per row. KIR
      gained one op for it, `Log2`, the bare `lg2.approx.f32`: `Log` is `ln`
      and multiplies by `ln 2` after, which would round twice where the hand
      kernel folds `ln 2` into the add as one `fma.rn.f32` (`lse = lg2(s) ·
      ln 2 + m`). The target is read as `s64`, `tl` is loaded only for a
      valid row, and the loss subtracts with `sub.rn.f32`.
      - **The gate,** `lce_finalize_kir_equivalence`, launches whole grids
        (the runtime's `ceil(rows / 256)`, one block, three) under all four
        thread schedules and both block orders. The sums have full
        significands and include zero, a subnormal, `inf` and NaN. The
        targets are valid, zero, negative and `i64::MIN`. It requires the
        hand kernel's bytes and the restated formulas, bit for bit. It
        kills mutants of the bound, the start's block index, a stride that
        skips rows or never advances, the `lg2`, `ln 2`, each `fma`
        operand, the `fma` rounded twice, the target's sign test, the
        invalid row's zero, the subtraction, every element size, 64-bit
        add and pointer. A stride that shrinks but stays nonzero is named as
        an equivalent mutant: every row is still covered, and a row
        computed twice is written with the same bytes.
      - **SASS** (sm_80/90/120): the same 4 `LDG`, 2 `STG`, one `MUFU`
        (lg2), one `FFMA`, and the same float and compare instructions as
        the hand kernel. Registers are 18/28/28 against 20/20/22, all at or
        under 32, so occupancy at 256 threads is unchanged.
    - **`fused_kernels.rs`, the deterministic scatter-add.**
      `nsl_det_scatter_add_f32` is in `nsl_kir::kernels::det_scatter`. On a
      `(vocab_size, embed_dim)` grid of 16 × 16 blocks, thread `(row, col)`
      starts from `input[row, col]`, walks every position in order, adds
      `src[i, col]` wherever the index names `row`, and writes the sum once
      (no atomics, so the result is the same on every run). The f32 index
      converts with the hand kernel's `cvt.rzi.u64.f32`, which saturates: a
      negative index and a NaN land on row 0. The kernel keeps that, and the
      gate pins it. The sum adds with `add.rn.f32`.
      - **The gate,** `det_scatter_kir_equivalence`, runs the 16 × 16 block
        and an 8 × 4 one (so `%ntid.x` and `%ntid.y` differ), with a spare
        block in each direction, under two schedules. The indices repeat
        rows, skip rows, and include fractions, `-0.5`, `-1.5`, NaN,
        `-inf`, `vocab`, `1e30` and `+inf`. It requires the hand kernel's
        bytes and the restated ordered scatter, bit for bit. It kills
        mutants of each bound, both block indices, `%tid.y`, `%ntid.y`,
        every element size, row stride and 64-bit add, the match test, the
        loop's step, the start from `input`, the accumulate, the conversion
        read as signed, and every pointer.
      - **SASS** (sm_80/90/120): the same 8× unrolled loop (17 `LDG`), 8
        `F2I`, 8 `FADD` and the same compares and branches as the hand
        kernel, at the same registers (32/32/38).
    - **`fused_kernels.rs`, the row softmax and log-softmax.**
      `nsl_softmax_f32` and `nsl_log_softmax_f32` are in
      `nsl_kir::kernels::softmax`, one 256-thread block per row. Each keeps
      the hand kernel's three passes and order of combines: a per-thread
      max over columns `k, k + 256, …`, folded by thread 0 in order through
      shared memory; a per-thread sum of `2^((x - max) · log2 e)` (the
      softmax stores each exponential), folded the same way and finished
      with `rcp.approx` or `lg2 · ln 2`; then `out *= 1 / sum` or `out = (x
      - max) - ln sum`. Every add, subtract and multiply is `.rn`. None of
      the hand kernels' multiply-add pairs could contract, and ptxas made
      no `FFMA` of either. The CTA interpreter learned to read several
      statements on one line, as the hand kernels write their shared
      accesses.
      - **The gate,** `softmax_kir_equivalence`, launches one block per
        row plus one past the last, under all four thread schedules. Rows
        are 0, 1, 255, 256, 257 and 700 columns wide, with order-sensitive
        sums, masked (`-inf`) columns, a fully masked row, a NaN and
        `+inf`. It requires the hand kernels' bytes and the restated
        passes, bit for bit. It kills mutants of the row bound, every
        column loop's bound and stride, each fold's start, bound and step,
        the thread-0 test, all four barriers, the row base, the sum
        region's offset, every element size and 64-bit add, every combine,
        `ex2` and its `log2 e` scale, the finish (`rcp`, or `lg2` and `ln
        2`), the softmax's store of the exponentials, and each identity.
        Named equivalent mutants: the max fold starting on thread 0's own
        partial and the max loop striding by 128 (the max is idempotent),
        the log-softmax's last loop striding by 128 (a pure map), the
        sum's identity as `-0` (every exponential is `+0` or more), and
        the element size of the fold's final `sm[0]` store (it has none).
      - **SASS** (sm_80/90/120): the same 10/17 `LDG`, 2/1 `STG`, 18
        `LDS`, 4 `STS`, `BAR`, `MUFU`, `FMUL`, `FADD` and 16 `FMNMX` as the
        hand kernels. Registers are 24/28/24 and 28/32/24 against 19/24/20
        and 24/27/24, all at or under 32, so occupancy at 256 threads is
        unchanged. Thread 0 reads `%ntid.x` once, before its fold: read in
        the fold's header, ptxas left both folds rolled for sm_90 and
        sm_120 (4 `LDS` against 18).
    - **`fused_kernels.rs`, the row LayerNorm and RMSNorm forwards.**
      `nsl_layernorm_f32` and `nsl_rmsnorm_f32` are in
      `nsl_kir::kernels::norm`, one 256-thread block per row. Each statistic
      is a per-thread partial sum over columns `k, k + 256, …`, folded by
      thread 0 in order through shared memory, divided by `cols` with
      `div.approx`; the LayerNorm's mean, then `rsqrt(Σ(x - mean)² / cols +
      eps)`, the RMSNorm's `rsqrt(Σx² / cols + eps)`. Every add, subtract and
      multiply is `.rn`: ptxas contracted the hand kernels' squares, `Σ /
      cols + eps` (`div.approx` is a multiply by the reciprocal) and the
      LayerNorm's `· gamma + beta` into `FFMA`, and the KIR kernels round
      twice, as the PTX and the CPU reference do. The hand LayerNorm reduced both statistics
      through one region, and thread 0 stored its variance partial to the
      mean's slot with no barrier after the other threads read the mean: a
      race. The KIR LayerNorm gives each statistic its own region.
      - **The gate,** `norm_kir_equivalence`, launches one block per row
        plus one past the last, with shared memory poisoned, under all four
        thread schedules. Rows are 0, 1, 255, 256, 257, 300, 700 and 40
        columns wide, with order-sensitive sums, a constant row, a NaN,
        `+inf`, squares that overflow and a zero `eps`. The KIR kernels
        match the restated passes bit for bit under every schedule. The
        hand RMSNorm agrees with them byte for byte under every schedule;
        the hand LayerNorm only under the one that runs thread 0 last, and
        the gate shows it wrong under the other three. It kills mutants of
        the row bound, every column loop's bound and stride, each fold's
        start, bound and step, the thread-0 test, every barrier, the row
        base, the LayerNorm's variance region (moved back onto the mean's),
        every element size and 64-bit add, every add, subtract and multiply
        (flipped and dropped), the `div.approx`, the `rsqrt` and each sum's
        identity. Named equivalent mutants: the last pass striding by 128
        (a pure map) and each sum's identity as `-0` (a sum of squares is
        never `-0`, and no row is all `-0`).
      - **SASS** (sm_80/90/120): the same 19/10 `LDG`, one `STG`, 18/9
        `LDS`, 4/2 `STS`, `BAR` and `MUFU` as the hand kernels, and the same
        unrolling. The hand kernels' 10/9 `FFMA` are as many `FMUL` and
        `FADD`. Registers are 28/30/24 and 24/26/20 against 22/26/24 and
        22/24/22, all at or under 32, so occupancy at 256 threads is
        unchanged.
    - **`fused_kernels.rs`, the fused RMSNorm input-gradient backward.**
      `nsl_rmsnorm_dx_bwd_f32` and its residual-folding twin
      `nsl_rmsnorm_dx_bwd_add_f32` are in `nsl_kir::kernels::rmsnorm_dx`,
      one 256-thread block per row. Each thread accumulates `S1 = Σ x²` and
      `S2 = Σ (dy · γ) · x` with the hand kernel's explicit `fma.rn`; thread
      0 folds both partials in one in-order loop and stores `rinv =
      min(rsqrt(S1 / cols + eps), 1e12)` (the CPU and tape-AD underflow
      guard) and `S2`; then `dx = (γ · dy) · rinv - x · ((rinv · rinv) ·
      rinv) · S2 / cols`, plus `res` for the twin. Every other add, subtract
      and multiply is `.rn`: ptxas fused the hand kernels' `S1 / cols + eps`
      and `(γ · dy) · rinv - …` (the multiply by `rinv` into the
      subtraction) into `FFMA`, and the KIR kernels round twice, as
      the PTX and the CPU reference do. The entry caps registers at 32
      (`.maxnreg`): left alone, ptxas gave the plain kernel 35–36 where the
      hand kernel had 32, which would drop a 256-thread block from 64 warps
      to 48 on sm_80 and sm_90. At 32 nothing spills.
      - **The gate,** `rmsnorm_dx_kir_equivalence`, launches one block per
        row plus one past the last, with shared memory poisoned, under all
        four thread schedules. Rows are 0, 1, 255, 256, 257, 300, 700 and
        40 columns wide, with order-sensitive sums, a constant row, a NaN,
        `+inf`, squares that overflow, and zero and tiny rows at `eps = 0`
        whose `rsqrt` the clamp holds. It requires the hand kernels' bytes
        and the restated passes, bit for bit. It kills mutants of the row
        bound, both column loops' bounds and strides, the fold's start,
        bound and step, the thread-0 test, both barriers, the row base, the
        second region's offset, every element size and 64-bit add, both
        `fma`s (dropped, or without their accumulate), every add, subtract
        and multiply (flipped and dropped), both `div.approx`, the `rsqrt`,
        the clamp (flipped and dropped) and each identity. Named equivalent
        mutants: the write pass striding by 128 (a pure map), and each
        sum's identity as `-0` (a `-0` partial survives only in a zero sum,
        which multiplies an `x` of zero).
      - **SASS** (sm_80/90/120): the same 27/28 `LDG`, one `STG`, 18 `LDS`,
        4 `STS`, `BAR` and `MUFU` as the hand kernels, and the same
        unrolling. The hand kernels' 18 `FFMA` are the explicit `fma`s' 16
        and 2 contractions; the KIR kernels have the 16, and the 2 are
        `FMUL` + `FADD`. Registers are 32/29/32 and 29/30/30 against
        32/32/34.
    - **`fused_kernels.rs`, the 2-D max pooling forward.**
      `nsl_maxpool2d_f32` is in `nsl_kir::kernels::maxpool`, a thread per
      output element. It splits the flat index with `rem`/`div` into `(n,
      c, oh, ow)`, walks the window in `ky`, `kx` order, skips taps in the
      padding (`oh · stride + ky < padding`, likewise across) or past the
      input, and keeps the running `(max, argmax)` from `(-inf, 0)` unless
      `x <= max`, as the hand kernel does: ties keep the first tap, a NaN
      tap wins and the next tap displaces it. The update is two `selp`s
      where the hand kernel branched around two moves. The argmax is stored
      as `u64`, which the CTA interpreter now models (`st` of `.u64`,
      `.b64`, `.s64`).
      - **The gate,** `maxpool_kir_equivalence`, launches whole grids with a
        spare block on 256- and 32-thread blocks under two schedules, over
        overlapping, tiling and gapped windows, padding of 0 to 2 with
        windows wholly in the padding, ties (`-0` against `+0` among them),
        NaNs and infinities. It requires the hand kernel's bytes and the
        restated walk, and kills mutants of every bound and padding test,
        every `div` and `rem`, every index multiply, add and subtract, the
        loop starts and steps, the tie test (`le` as `lt` or `ge`), both
        `selp`s, the max's start, every element size and the argmax's
        64-bit store.
      - **SASS** (sm_80/90/120): the same one load, two stores, one float
        compare, branches and six 64-bit `div`/`rem` calls as the hand
        kernel, in 28/28/26 registers against 30.
    - **The printer's conditional edges.** A `CondBranch` edge that
      carries block arguments branches to a trampoline printed after the
      last block, which makes the copies and jumps to the target; the
      branch is a plain `@p bra; bra` pair. With the copies inline around
      the branch, ptxas neither unrolled nor if-converted the loops those
      branches close.
      - **SASS across the runtime's KIR modules:** unchanged except
        `nsl_csha_tier_b1_prepass_x`, whose loops now unroll (sm_80 memory
        instructions 8 → 22).
      - **fused_linear_ce v1:** the forward is unchanged; the backward's
        tiled loop unrolls again (sm_80 memory instructions 17 → 340, the
        hand kernel 772), recovering the unroll #723's migration lost.
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
