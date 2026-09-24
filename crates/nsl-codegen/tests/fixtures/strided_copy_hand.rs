//! The hand-written strided run-copy module, frozen (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::strided_copy::STRIDED_COPY_RUN_PTX` as it stood when
//! the kernels moved to `nsl_kir::kernels::strided_copy`, verbatim. It is
//! the reference side of `strided_copy_kir_equivalence.rs` and is not
//! compiled into anything else.

/// Four entry points sharing one module, selected by `RunPlan::kernel_name`.
///
/// Common contract:
///   `src`, `dst`     — f32 device pointers
///   `offsets`        — i64[outer], source element offset of each run
///   `run_len`        — elements per run
///   `outer`          — number of runs
///   grid.x spans one run, grid.y spans the runs (grid-stride, so `outer` may
///   exceed 65535); `dst` run `o` starts at element `o * run_len`.
///
/// ISA 7.0: no `mad.lo.u32` (see `feedback_ptx_comment_ascii_only` and the
/// PTX invariants in the wiki) -- index math is `mul.lo.u32` + `add.u32`.
/// ASCII only: a non-ASCII byte anywhere in the module makes `cuModuleLoadData`
/// return CUDA_ERROR_INVALID_PTX.
pub const STRIDED_COPY_RUN_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
// dst[o*run_len + i] = src[offsets[o] + i]\n\
.visible .entry nsl_scopy_run_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    setp.ge.u32 %p1, %r3, %r4;\n\
    @%p1 bra SC_RUN_DONE;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_RUN_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_RUN_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    add.u64 %rd9, %rd9, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.f32 [%rd10], %f1;\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_RUN_LOOP;\n\
SC_RUN_DONE: ret;\n\
}\n\
\n\
// Vector form of nsl_scopy_run_f32: i counts float4 units.\n\
.visible .entry nsl_scopy_run4_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<5>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    shr.u32 %r8, %r4, 2;\n\
    setp.ge.u32 %p1, %r3, %r8;\n\
    @%p1 bra SC_RUN4_DONE;\n\
    shl.b32 %r9, %r3, 2;\n\
    cvt.u64.u32 %rd6, %r9;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_RUN4_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_RUN4_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    add.u64 %rd9, %rd9, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.v4.f32 [%rd10], {%f1, %f2, %f3, %f4};\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_RUN4_LOOP;\n\
SC_RUN4_DONE: ret;\n\
}\n\
\n\
// dst[o*run_len + i] = src[offsets[o]]  (innermost source stride 0)\n\
.visible .entry nsl_scopy_bcast_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    setp.ge.u32 %p1, %r3, %r4;\n\
    @%p1 bra SC_BC_DONE;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_BC_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_BC_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.f32 [%rd10], %f1;\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_BC_LOOP;\n\
SC_BC_DONE: ret;\n\
}\n\
\n\
// Vector form of nsl_scopy_bcast_f32: one scalar load, one v4 store.\n\
.visible .entry nsl_scopy_bcast4_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    shr.u32 %r8, %r4, 2;\n\
    setp.ge.u32 %p1, %r3, %r8;\n\
    @%p1 bra SC_BC4_DONE;\n\
    shl.b32 %r9, %r3, 2;\n\
    cvt.u64.u32 %rd6, %r9;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_BC4_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_BC4_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.v4.f32 [%rd10], {%f1, %f1, %f1, %f1};\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_BC4_LOOP;\n\
SC_BC4_DONE: ret;\n\
}\0";
