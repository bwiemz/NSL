// crates/nsl-kir/src/kernels/strided_copy.rs
//! The four strided run-copy kernels, as KIR (roadmap A2 step 11).
//!
//! `nsl_tensor_contiguous` materialises most GPU views without the generic
//! strided kernel's per-element coordinate decomposition: the runtime's
//! planner (`nsl_runtime::cuda::strided_copy::plan_run`) rewrites a view as
//! `outer` contiguous runs of `run_len` elements and uploads one source
//! offset per run. These kernels then copy the runs. They replace the
//! runtime's `STRIDED_COPY_RUN_PTX`, a hand-written module of the same four
//! entries.
//!
//! ## The kernels
//!
//! | entry | reads | moves |
//! |---|---|---|
//! | `nsl_scopy_run_f32` | `src[offsets[o] + i]` | one f32 |
//! | `nsl_scopy_run4_f32` | `src[offsets[o] + 4i ..][..4]` | one `v4.f32` |
//! | `nsl_scopy_bcast_f32` | `src[offsets[o]]` | one f32 |
//! | `nsl_scopy_bcast4_f32` | `src[offsets[o]]` | one f32, stored as `v4` |
//!
//! Each writes run `o` at `dst[o * run_len ..]`. They share one module and
//! one signature, `(src, dst, offsets, run_len, outer)`, every parameter a
//! `.u64`: `src` and `dst` f32 pointers, `offsets` the `outer` source
//! element offsets.
//!
//! ## Geometry
//!
//! `grid.x` covers one run and `grid.y` the runs, walked grid-stride so
//! `outer` may exceed the 65535 `gridDim.y` limit. The thread's unit `i` is
//! `blockIdx.x * blockDim.x + threadIdx.x`; a thread past the run (`i >=
//! run_len`, or `run_len / 4` for the vector arms) exits before the loop.
//! The block is at most [`SCOPY_MAX_BLOCK`] threads.
//!
//! ## Widths
//!
//! `run_len` and `outer` are read as `u64` and narrowed to 32 bits, as the
//! hand kernels did; the planner refuses a run longer than `u32::MAX` and
//! caps `outer` at 2^20, so neither narrowing loses anything it admits.
//! The run index and the thread's unit are 32-bit; every element index is
//! 64-bit, and `o * run_len` is a full 64-bit product of the two narrowed
//! values (the hand kernels' `mul.wide.u32`).

use crate::backend_ptx::lower_kir_module_to_ptx;
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp,
    KirTerminator, KirType, VarId,
};

/// The largest block the launcher uses (`RunPlan::geometry` clamps to it),
/// so the kernels declare it as their `.maxntid`.
pub const SCOPY_MAX_BLOCK: u32 = 256;

/// Which arm of the run copy a kernel is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopyArm {
    /// `dst[o*run_len + i] = src[offsets[o] + i]`.
    Run,
    /// [`ScopyArm::Run`] four elements at a time.
    Run4,
    /// `dst[o*run_len + i] = src[offsets[o]]`: the innermost source stride
    /// is 0.
    Bcast,
    /// [`ScopyArm::Bcast`] four elements at a time: one scalar load, one
    /// vector store.
    Bcast4,
}

impl ScopyArm {
    /// Every arm, in module order, which is also the runtime's
    /// `RunPlan::arm_index` order.
    pub const ALL: [ScopyArm; 4] = [ScopyArm::Run, ScopyArm::Run4, ScopyArm::Bcast, ScopyArm::Bcast4];

    /// The `.visible .entry` name. Pinned: the runtime looks each arm up in
    /// the loaded module by exactly this string.
    pub fn kernel_name(self) -> &'static str {
        match self {
            ScopyArm::Run => "nsl_scopy_run_f32",
            ScopyArm::Run4 => "nsl_scopy_run4_f32",
            ScopyArm::Bcast => "nsl_scopy_bcast_f32",
            ScopyArm::Bcast4 => "nsl_scopy_bcast4_f32",
        }
    }

    /// Whether the arm moves four elements per thread and iteration.
    pub fn vec4(self) -> bool {
        matches!(self, ScopyArm::Run4 | ScopyArm::Bcast4)
    }

    /// Whether the arm reads one source element per run.
    pub fn broadcast(self) -> bool {
        matches!(self, ScopyArm::Bcast | ScopyArm::Bcast4)
    }
}

fn u32_const(b: &mut KirBuilder, v: u32) -> VarId {
    let dst = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(dst, KirConst { ty: KirType::U32, value: ConstValue::U32(v) }));
    dst
}

fn cast(b: &mut KirBuilder, src: VarId, ty: KirType) -> VarId {
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Cast(dst, src, ty));
    dst
}

fn f32_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
}

/// Build `arm` as KIR.
///
/// ```text
/// entry:          i = blockIdx.x*blockDim.x + threadIdx.x
///                 run_len, outer narrowed to u32
///                 units = run_len (>> 2 for a vector arm)
///                 if i >= units { br exit } else { br setup }
/// setup:          e = i (<< 2 for a vector arm), as u64
///                 br loop(blockIdx.y)
/// loop(o: u32):   if o >= outer { br exit } else { br body }
/// body:           off = offsets[o]
///                 v = src[off + e]      (src[off] for a broadcast arm)
///                 dst[o*run_len + e] = v
///                 br loop(o + gridDim.y)
/// exit:           ret
/// ```
pub fn build(arm: ScopyArm) -> KernelIR {
    let mut b = KirBuilder::new(arm.kernel_name());

    let src = b.add_param("src", f32_ptr(), AddressSpace::Global);
    let dst = b.add_param("dst", f32_ptr(), AddressSpace::Global);
    let offsets = b.add_param(
        "offsets",
        KirType::Ptr(Box::new(KirType::U64), AddressSpace::Global),
        AddressSpace::Global,
    );
    let run_len = b.add_param("run_len", KirType::U64, AddressSpace::Global);
    let outer = b.add_param("outer", KirType::U64, AddressSpace::Global);

    let entry = b.new_block();
    let setup = b.new_block();
    let loop_head = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    let o = b.add_block_param(loop_head, KirType::U32);

    // ── entry: the thread's unit, and the x bound ────────────────────
    b.set_block(entry);
    let i = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(i, 0));
    let run_len32 = cast(&mut b, run_len, KirType::U32);
    let outer32 = cast(&mut b, outer, KirType::U32);
    let units = if arm.vec4() {
        let two = u32_const(&mut b, 2);
        let units = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Shr(units, run_len32, two));
        units
    } else {
        run_len32
    };
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, i, units, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    // ── setup: the element index within a run, and the y walk ────────
    b.set_block(setup);
    let e32 = if arm.vec4() {
        let two = u32_const(&mut b, 2);
        let e = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Shl(e, i, two));
        e
    } else {
        i
    };
    let e = cast(&mut b, e32, KirType::U64);
    let run_len_w = cast(&mut b, run_len32, KirType::U64);
    let o0 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::BlockIdx(o0, 1));
    let step = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GridDim(step, 1));
    b.terminate(KirTerminator::Branch(KirEdge::with(loop_head, vec![o0])));

    // ── loop head: the run bound ─────────────────────────────────────
    b.set_block(loop_head);
    let done = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(done, o, outer32, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(done, KirEdge::to(exit), KirEdge::to(body)));

    // ── body: one unit of run `o` ────────────────────────────────────
    b.set_block(body);
    let o64 = cast(&mut b, o, KirType::U64);
    let off_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::U64), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(off_addr, offsets, o64));
    let off = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Load(off, off_addr, AddressSpace::Global));
    let src_idx = if arm.broadcast() {
        off
    } else {
        let s = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Add(s, off, e));
        s
    };
    let src_addr = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(src_addr, src, src_idx));
    let vals: Vec<VarId> = match arm {
        ScopyArm::Run4 => {
            let dsts: Vec<VarId> = (0..4).map(|_| b.new_typed_var(KirType::F32)).collect();
            b.emit(KirOp::LoadVec { dsts: dsts.clone(), ptr: src_addr, space: AddressSpace::Global });
            dsts
        }
        _ => {
            let v = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Load(v, src_addr, AddressSpace::Global));
            if arm.vec4() { vec![v; 4] } else { vec![v] }
        }
    };
    let row = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Mul(row, o64, run_len_w));
    let dst_idx = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Add(dst_idx, row, e));
    let dst_addr = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(dst_addr, dst, dst_idx));
    if vals.len() == 4 {
        b.emit(KirOp::StoreVec { ptr: dst_addr, vals, space: AddressSpace::Global });
    } else {
        b.emit(KirOp::Store(dst_addr, vals[0], AddressSpace::Global));
    }
    let next = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Add(next, o, step));
    b.terminate(KirTerminator::Branch(KirEdge::with(loop_head, vec![next])));

    // ── exit ─────────────────────────────────────────────────────────
    b.set_block(exit);
    b.terminate(KirTerminator::Return);

    b.set_workgroup_size([SCOPY_MAX_BLOCK, 1, 1]);
    b.set_launch_bounds(SCOPY_MAX_BLOCK, None);
    b.finalize()
}

/// Build the four arms and lower them to one NUL-terminated PTX module,
/// entries in [`ScopyArm::ALL`] order.
///
/// # Panics
///
/// If a built kernel fails verification: a bug in this module, not a
/// condition a caller can provoke, since the kernels take no input.
pub fn ptx() -> Vec<u8> {
    let irs: Vec<KernelIR> = ScopyArm::ALL.iter().map(|&arm| build(arm)).collect();
    for (arm, ir) in ScopyArm::ALL.iter().zip(&irs) {
        if let Err(errors) = crate::kir_verify::verify(ir) {
            panic!("strided-copy kernel `{}` failed KIR verification: {errors:?}", arm.kernel_name());
        }
    }
    let refs: Vec<&KernelIR> = irs.iter().collect();
    lower_kir_module_to_ptx(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kir_verify::verify;

    #[test]
    fn all_four_kernels_verify() {
        for arm in ScopyArm::ALL {
            if let Err(errors) = verify(&build(arm)) {
                panic!("{arm:?} failed verification: {errors:?}");
            }
        }
    }

    /// The runtime marshals `(src, dst, offsets, run_len, outer)`
    /// positionally, every one a `u64`-sized value.
    #[test]
    fn the_parameter_list_is_src_dst_offsets_run_len_outer() {
        for arm in ScopyArm::ALL {
            let ir = build(arm);
            let names: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(names, ["src", "dst", "offsets", "run_len", "outer"], "{arm:?}");
            assert_eq!(ir.params[3].ty, KirType::U64, "{arm:?}: run_len is u64");
            assert_eq!(ir.params[4].ty, KirType::U64, "{arm:?}: outer is u64");
        }
    }

    #[test]
    fn the_entry_names_are_pinned() {
        let names: Vec<&str> = ScopyArm::ALL.iter().map(|a| a.kernel_name()).collect();
        assert_eq!(
            names,
            ["nsl_scopy_run_f32", "nsl_scopy_run4_f32", "nsl_scopy_bcast_f32", "nsl_scopy_bcast4_f32"]
        );
        for arm in ScopyArm::ALL {
            assert_eq!(build(arm).name, arm.kernel_name(), "{arm:?}");
        }
    }

    /// One module, every entry in it, NUL-terminated ASCII (the driver reads
    /// it as a C string), and no `mad.lo.u32` (invalid at ISA 7.0).
    #[test]
    fn the_module_holds_every_entry_as_nul_terminated_ascii() {
        let bytes = ptx();
        assert_eq!(bytes.last(), Some(&0));
        assert!(!bytes[..bytes.len() - 1].contains(&0));
        assert!(bytes.is_ascii());
        let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
        assert!(!text.contains("mad.lo.u32"));
        assert_eq!(text.matches(".visible .entry ").count(), 4);
        for arm in ScopyArm::ALL {
            assert!(text.contains(&format!(".visible .entry {}(", arm.kernel_name())), "{arm:?}");
        }
        // Nothing here needs more than the ISA floor.
        assert!(text.starts_with(".version 7.0\n"), "{}", &text[..40]);
    }

    /// The vector arms move `v4.f32` and the scalar arms do not.
    #[test]
    fn only_the_vector_arms_use_vector_memory() {
        let bytes = ptx();
        let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
        for arm in ScopyArm::ALL {
            let start = text.find(&format!(".visible .entry {}(", arm.kernel_name())).unwrap();
            let end = text[start + 1..].find(".visible .entry ").map_or(text.len(), |e| start + 1 + e);
            let body = &text[start..end];
            assert_eq!(body.contains("st.global.v4.f32"), arm.vec4(), "{arm:?}");
            assert_eq!(body.contains("ld.global.v4.f32"), arm == ScopyArm::Run4, "{arm:?}");
        }
    }
}
