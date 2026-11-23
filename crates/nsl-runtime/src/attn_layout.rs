//! The memory layout of attention head tensors.
//!
//! `scaled_dot_product_attention` takes `[batch, heads, seq, head_dim]`
//! operands. The projections that produce them are `[batch, seq, heads ·
//! head_dim]` GEMM outputs, and the stdlib's head view
//! `q.reshape([b, s, h, d]).transpose(1, 2)` is that buffer itself, read
//! through strides: its memory order is `[b, s, h, d]` ([`HeadLayout::Bshd`]).
//! The attention's consumers go the other way (`out.transpose(1, 2)
//! .reshape([b, s, h · d])` into the output projection; the gradients back
//! into the projections' GEMMs), so a `[b, s, h, d]` buffer is what both
//! ends want.
//!
//! The tensor-core attention kernels read and write either layout
//! (`nsl_kir::kernels::flash_attn`, `bshd` variants), and the `rope` builtin
//! keeps its input's layout. So the head view flows through RoPE, the
//! attention forward and backward and back into the GEMMs without a
//! transpose copy: the transposes in NSL source stay metadata.
//!
//! `NSL_ATTN_BSHD=0` turns this off: RoPE writes the contiguous `[b, h, s,
//! d]` layout, and the attention copies any `[b, s, h, d]` operand to it
//! first and writes its outputs in it, as before this layout existed. The
//! values are the same either way: only where they are stored differs.

use crate::tensor::NslTensor;

/// Which dense memory order a `[b, h, s, d]` tensor has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum HeadLayout {
    /// Contiguous `[b, h, s, d]`: element `(b, h, s, d)` at
    /// `((b·H + h)·S + s)·D + d`.
    Bhsd,
    /// The attention layout `[b, s, h, d]` viewed as `[b, h, s, d]`:
    /// element `(b, h, s, d)` at `((b·S + s)·H + h)·D + d`.
    Bshd,
}

impl HeadLayout {
    /// The dims of `[b, h, s, d]` from outermost to innermost in memory.
    fn order(self) -> [usize; 4] {
        match self {
            HeadLayout::Bhsd => [0, 1, 2, 3],
            HeadLayout::Bshd => [0, 2, 1, 3],
        }
    }

    /// The element strides of a `[b, h, s, d]` tensor of `shape` stored this way.
    pub(crate) fn strides(self, shape: [i64; 4]) -> [i64; 4] {
        let mut st = [0i64; 4];
        let mut acc = 1i64;
        for &dim in self.order().iter().rev() {
            st[dim] = acc;
            acc *= shape[dim].max(1);
        }
        st
    }
}

fn dims4(t: &NslTensor) -> [i64; 4] {
    std::array::from_fn(|i| unsafe { *t.shape.add(i) })
}

/// Whether `t` (rank 4) is densely stored in `layout`. A dim of extent 1 may
/// carry any stride: it never addresses a second element.
pub(crate) fn is_in(t: &NslTensor, layout: HeadLayout) -> bool {
    if t.ndim != 4 {
        return false;
    }
    let shape = dims4(t);
    let want = layout.strides(shape);
    (0..4).all(|i| shape[i] <= 1 || unsafe { *t.strides.add(i) } == want[i])
}

/// The head layout `t` is stored in: contiguous first (a tensor with one
/// head or one position is both), or `None` for any other strides.
pub(crate) fn head_layout(t: &NslTensor) -> Option<HeadLayout> {
    if is_in(t, HeadLayout::Bhsd) {
        Some(HeadLayout::Bhsd)
    } else if is_in(t, HeadLayout::Bshd) {
        Some(HeadLayout::Bshd)
    } else {
        None
    }
}

/// Whether RoPE and the tensor-core attention keep the `[b, s, h, d]`
/// layout (`NSL_ATTN_BSHD=0` turns it off); read once.
pub(crate) fn bshd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NSL_ATTN_BSHD").ok().as_deref() != Some("0"))
}

/// The layout the tensor-core attention computes in for a query `q`:
/// `[b, s, h, d]` when `q` is stored that way, the layout is on and the
/// kernels can address it ([`crate::attention_tc::supports_bshd`]);
/// contiguous otherwise (also when `q` is both, with one head or one
/// position). The other operands are put into it ([`to_layout`]).
pub(crate) fn tc_layout(q: &NslTensor, b: i64, h: i64, kv_h: i64, s: i64, d: i64) -> HeadLayout {
    let fits = crate::attention_tc::supports_bshd(b as usize, d as usize, h as usize, kv_h as usize, s as usize);
    if bshd_enabled() && fits && head_layout(q) == Some(HeadLayout::Bshd) {
        HeadLayout::Bshd
    } else {
        HeadLayout::Bhsd
    }
}

/// Route indices of [`BSHD_ROUTE_COUNTS`].
pub(crate) const ROUTE_ROPE: usize = 0;
pub(crate) const ROUTE_FWD: usize = 1;
pub(crate) const ROUTE_BWD: usize = 2;

/// Process-wide counts of the head-layout routes, for gates that must prove
/// a route ran: `[RoPE kept [b, s, h, d], attention forward read it,
/// attention backward read it]`.
pub(crate) static BSHD_ROUTE_COUNTS: [std::sync::atomic::AtomicI64; 3] =
    [const { std::sync::atomic::AtomicI64::new(0) }; 3];

/// Count one run of route `which`; the first run of each prints its
/// `[attn-layout]` line.
pub(crate) fn count_route(which: usize) {
    if BSHD_ROUTE_COUNTS[which].fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        let what = match which {
            ROUTE_ROPE => "rope keeps the [b, s, h, d] head layout",
            ROUTE_FWD => "tensor-core attention forward reads and writes [b, s, h, d]",
            _ => "tensor-core attention backward reads and writes [b, s, h, d]",
        };
        crate::nsl_log!(INFO, "attn-layout", "[attn-layout] {what}");
    }
}

/// How many times route `which` ([`ROUTE_ROPE`], [`ROUTE_FWD`],
/// [`ROUTE_BWD`]) ran in this process.
#[cfg(test)]
pub(crate) fn route_count(which: usize) -> i64 {
    BSHD_ROUTE_COUNTS[which].load(std::sync::atomic::Ordering::Relaxed)
}

/// Wrap `data` (`len` elements of `dtype`, owned) as a `[b, h, s, d]` tensor
/// of `shape` stored in `layout`. For [`HeadLayout::Bshd`] the owner is the
/// contiguous `[b, s, h, d]` tensor and the result a view of it, exactly
/// what `x.transpose(1, 2)` of that tensor is (built without recording on
/// a tape).
pub(crate) fn wrap_head_tensor(data: *mut std::ffi::c_void, shape: [i64; 4], layout: HeadLayout, dtype: u16, device: u8) -> i64 {
    let len: i64 = shape.iter().product();
    let mem_shape: [i64; 4] = match layout {
        HeadLayout::Bhsd => shape,
        HeadLayout::Bshd => [shape[0], shape[2], shape[1], shape[3]],
    };
    let sh = NslTensor::copy_shape(mem_shape.as_ptr() as *mut i64, 4);
    let st = NslTensor::compute_strides(sh, 4);
    let owner = NslTensor::publish(Box::new(NslTensor::new(data, sh, st, 4, len, device, dtype, 1, 0)));
    if layout == HeadLayout::Bhsd {
        return owner;
    }
    let view = NslTensor::new_view_i64(owner, &shape, &layout.strides(shape), 4, len);
    crate::tensor::nsl_tensor_free(owner);
    view
}

/// `t` (a `[b, h, s, d]` device tensor) stored in `layout`: `t` itself with
/// its refcount raised when it already is, else a copy. The caller frees the
/// result. Records nothing on an autodiff tape.
pub(crate) fn to_layout(t: i64, layout: HeadLayout) -> i64 {
    let tt = NslTensor::from_ptr_ref(t);
    if is_in(tt, layout) {
        tt.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        return t;
    }
    let _pause = crate::autodiff::TapePause::new();
    match layout {
        HeadLayout::Bhsd => crate::tensor::nsl_tensor_contiguous(t),
        HeadLayout::Bshd => {
            // transpose(1, 2) -> contiguous [b, s, h, d] -> viewed back.
            let shape = dims4(tt);
            let len = tt.len;
            let tst: [i64; 4] = std::array::from_fn(|i| unsafe { *tt.strides.add(i) });
            let mem_view = NslTensor::new_view_i64(
                t,
                &[shape[0], shape[2], shape[1], shape[3]],
                &[tst[0], tst[2], tst[1], tst[3]],
                4,
                len,
            );
            let mem = crate::tensor::nsl_tensor_contiguous(mem_view);
            crate::tensor::nsl_tensor_free(mem_view);
            let out = NslTensor::new_view_i64(mem, &shape, &HeadLayout::Bshd.strides(shape), 4, len);
            crate::tensor::nsl_tensor_free(mem);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: [i64; 4], strides: [i64; 4]) -> NslTensor {
        let sh = NslTensor::copy_shape(shape.as_ptr() as *mut i64, 4);
        let st = NslTensor::copy_shape(strides.as_ptr() as *mut i64, 4);
        NslTensor::new(std::ptr::null_mut(), sh, st, 4, shape.iter().product(), 0, 1, 0, 0)
    }

    /// The stdlib's head view `x.reshape([b, s, h, d]).transpose(1, 2)` of a
    /// contiguous projection is [`HeadLayout::Bshd`]; a contiguous tensor is
    /// [`HeadLayout::Bhsd`]; one head (or one position) makes the two the
    /// same buffer, reported contiguous; anything else is neither.
    #[test]
    fn head_layouts_are_recognised_by_their_strides() {
        let (b, h, s, d) = (2i64, 4, 6, 8);
        assert_eq!(HeadLayout::Bhsd.strides([b, h, s, d]), [h * s * d, s * d, d, 1]);
        assert_eq!(HeadLayout::Bshd.strides([b, h, s, d]), [s * h * d, d, h * d, 1]);
        assert_eq!(head_layout(&tensor([b, h, s, d], [h * s * d, s * d, d, 1])), Some(HeadLayout::Bhsd));
        assert_eq!(head_layout(&tensor([b, h, s, d], [s * h * d, d, h * d, 1])), Some(HeadLayout::Bshd));
        let one_head = tensor([b, 1, s, d], [s * d, d, d, 1]);
        assert_eq!(head_layout(&one_head), Some(HeadLayout::Bhsd));
        assert!(is_in(&one_head, HeadLayout::Bshd));
        // A batch of one: its batch stride is never used.
        assert_eq!(head_layout(&tensor([1, h, s, d], [7, d, h * d, 1])), Some(HeadLayout::Bshd));
        // [b, h, d, s] (a key transpose) and a padded row are neither.
        assert_eq!(head_layout(&tensor([b, h, s, d], [h * s * d, s * d, 1, s])), None);
        assert_eq!(head_layout(&tensor([b, h, s, d], [s * h * (d + 8), d + 8, h * (d + 8), 1])), None);
    }
}
