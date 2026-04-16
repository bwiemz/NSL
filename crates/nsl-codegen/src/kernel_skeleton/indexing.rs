//! Thread/lane/warp/block-index register initialization.
//!
//! Emits the fixed 6-line tid/warp/lane/bid dance used by both FA and
//! WRGA.  Zero parameters — PTX convention fixes the register names.

/// Emit the `.reg .u32` declaration for tid/warp/lane/block-index registers.
///
/// Register names are fixed by PTX convention:
///   %tid_x, %warp_id, %lane, %bid_x, %bid_y
///
/// Callers needing different names alias locally with `mov`.
pub fn emit_thread_lane_warp_register_decl(ptx: &mut String) {
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
}

/// Emit the 5-line tid/warp/lane/block-index initialization block.
///
/// After this returns, the following registers hold useful values:
///   %tid_x   (u32) = threadIdx.x
///   %warp_id (u32) = tid_x / 32
///   %lane    (u32) = tid_x % 32
///   %bid_x   (u32) = blockIdx.x
///   %bid_y   (u32) = blockIdx.y
///
/// The declarations from `emit_thread_lane_warp_register_decl` must
/// already be in scope.  Some callers emit these two blocks far apart
/// in their kernel prolog (e.g. FA separates them by ~110 lines of
/// param loads); others emit back-to-back.  The split helpers preserve
/// that flexibility.
pub fn emit_thread_lane_warp_register_init(ptx: &mut String) {
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    shr.u32 %warp_id, %tid_x, 5;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");
}
