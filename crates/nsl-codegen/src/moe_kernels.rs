//! M32: PTX kernels for MoE dispatch pipeline.
//!
//! Four kernels:
//! 1. moe_token_sort_histogram — count tokens per expert (shared memory histogram)
//! 2. moe_scatter — reorder tokens into expert-sorted order
//! 3. expert_batched_gemm — tiled matmul per expert using blockIdx.z for expert selection
//! 4. moe_gather — weighted combination back to original order (atomic add for top_k>1)

/// Generate PTX for the MoE token sorting histogram kernel.
///
/// Counts how many tokens are assigned to each expert per thread block.
/// Output: histogram[blockIdx * num_experts + expert_id] = count.
///
/// Uses extern (dynamic) shared memory so num_experts is not capped at any
/// compile-time limit. The caller must pass `num_experts * sizeof(u32)` as
/// `dynamicSmemBytes` when launching the kernel.
pub fn gen_moe_token_sort_histogram_ptx() -> Vec<u8> {
    let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry moe_token_sort_histogram(
    .param .u64 expert_indices_ptr,
    .param .u64 histogram_ptr,
    .param .u32 total_assignments,
    .param .u32 num_experts
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<10>;
    .reg .pred %p<4>;
    // Dynamic shared memory — size set at launch: num_experts * sizeof(u32)
    .extern .shared .align 4 .u32 local_hist[];

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mul.lo.u32 %r3, %r0, %r1;
    add.u32 %r3, %r3, %r2;

    // Load params
    ld.param.u64 %rd0, [expert_indices_ptr];
    ld.param.u64 %rd1, [histogram_ptr];
    ld.param.u32 %r4, [total_assignments];
    ld.param.u32 %r5, [num_experts];

    // Zero shared memory — cooperative: each thread zeros slots in a loop
    // (handles num_experts > blockDim.x)
    mov.u32 %r15, %r2;
HIST_ZERO_LOOP:
    setp.ge.u32 %p0, %r15, %r5;
    @%p0 bra HIST_ZERO_DONE;
    mul.lo.u32 %r16, %r15, 4;
    st.shared.u32 [local_hist + %r16], 0;
    add.u32 %r15, %r15, %r1;  // stride by blockDim.x
    bra HIST_ZERO_LOOP;
HIST_ZERO_DONE:
    bar.sync 0;

    // Bounds check
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra HIST_DONE;

    // Load expert index for this assignment
    cvt.u64.u32 %rd2, %r3;
    mul.lo.u64 %rd2, %rd2, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.u32 %r6, [%rd3];

    // AtomicAdd to shared histogram
    mul.lo.u32 %r7, %r6, 4;
    atom.shared.add.u32 %r8, [local_hist + %r7], 1;

    bar.sync 0;

    // Write block's histogram to global memory (cooperative loop)
    mov.u32 %r15, %r2;
HIST_WRITE_LOOP:
    setp.ge.u32 %p2, %r15, %r5;
    @%p2 bra HIST_DONE;
    mul.lo.u32 %r9, %r0, %r5;
    add.u32 %r9, %r9, %r15;
    cvt.u64.u32 %rd4, %r9;
    mul.lo.u64 %rd4, %rd4, 4;
    add.u64 %rd5, %rd1, %rd4;
    mul.lo.u32 %r10, %r15, 4;
    ld.shared.u32 %r11, [local_hist + %r10];
    st.global.u32 [%rd5], %r11;
    add.u32 %r15, %r15, %r1;  // stride by blockDim.x
    bra HIST_WRITE_LOOP;

HIST_DONE:
    ret;
}
"#;
    let mut bytes = ptx.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

/// Generate PTX for the MoE scatter kernel.
///
/// Reorders tokens according to sorted_token_indices.
/// Each thread copies one entire row (hidden_dim elements).
pub fn gen_moe_scatter_ptx() -> Vec<u8> {
    let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry moe_scatter(
    .param .u64 tokens_ptr,
    .param .u64 sorted_indices_ptr,
    .param .u64 sorted_tokens_ptr,
    .param .u32 num_assigned,
    .param .u32 hidden_dim
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<2>;
    .reg .pred %p<2>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mul.lo.u32 %r3, %r0, %r1;
    add.u32 %r3, %r3, %r2;

    ld.param.u64 %rd0, [tokens_ptr];
    ld.param.u64 %rd1, [sorted_indices_ptr];
    ld.param.u64 %rd2, [sorted_tokens_ptr];
    ld.param.u32 %r4, [num_assigned];
    ld.param.u32 %r5, [hidden_dim];

    // Each thread handles one row
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra SCATTER_DONE;

    // Load source row index
    cvt.u64.u32 %rd3, %r3;
    mul.lo.u64 %rd4, %rd3, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r6, [%rd5];

    // Copy row element by element
    mul.lo.u32 %r7, %r6, %r5;
    mul.lo.u32 %r8, %r3, %r5;
    mov.u32 %r9, 0;

SCATTER_LOOP:
    setp.ge.u32 %p1, %r9, %r5;
    @%p1 bra SCATTER_DONE;

    add.u32 %r10, %r7, %r9;
    cvt.u64.u32 %rd6, %r10;
    mul.lo.u64 %rd6, %rd6, 4;
    add.u64 %rd7, %rd0, %rd6;
    ld.global.f32 %f0, [%rd7];

    add.u32 %r11, %r8, %r9;
    cvt.u64.u32 %rd8, %r11;
    mul.lo.u64 %rd8, %rd8, 4;
    add.u64 %rd9, %rd2, %rd8;
    st.global.f32 [%rd9], %f0;

    add.u32 %r9, %r9, 1;
    bra SCATTER_LOOP;

SCATTER_DONE:
    ret;
}
"#;
    let mut bytes = ptx.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

/// Select tile size for expert batched GEMM based on problem size and GPU.
///
/// Returns `(tile_m, tile_n)`. Tensor-core path was deleted (see PR
/// notes — the inline MMA fragment loader was structurally wrong
/// per the same m16n8k16 lane-mapping bug that bit CSHA Tier B.1).
/// Future MoE MMA work should use the validated `matmul_mma::emit_load_*`
/// helpers directly. For now all configurations use the scalar 16x16
/// path; the `gpu_sm` argument is accepted but ignored to keep the
/// call sites stable.
pub fn select_moe_gemm_tile(
    _expert_m: usize,
    _expert_n: usize,
    _expert_k: usize,
    _gpu_sm: u32,
) -> (usize, usize) {
    (16, 16)
}

/// Generate PTX for the batched expert GEMM kernel with configurable tile size.
///
/// Uses blockIdx.z to select expert. Always emits the 16x16 scalar
/// `fma.rn.f32` body — the MMA path was deleted (see `select_moe_gemm_tile`).
/// The `tile_size` arg is kept so tests can exercise non-canonical sizes.
pub fn gen_expert_batched_gemm_ptx_tiled(tile_size: usize) -> Vec<u8> {
    use std::fmt::Write;

    let mut ptx = String::with_capacity(8192);

    writeln!(ptx, ".version 7.0").unwrap();
    writeln!(ptx, ".target sm_70").unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();

    writeln!(ptx, ".visible .entry expert_batched_gemm(").unwrap();
    writeln!(ptx, "    .param .u64 sorted_tokens_ptr,").unwrap();
    writeln!(ptx, "    .param .u64 expert_weights_ptr,").unwrap();
    writeln!(ptx, "    .param .u64 output_ptr,").unwrap();
    writeln!(ptx, "    .param .u64 boundaries_ptr,").unwrap();
    writeln!(ptx, "    .param .u32 num_experts,").unwrap();
    writeln!(ptx, "    .param .u32 hidden_dim,").unwrap();
    writeln!(ptx, "    .param .u32 intermediate_dim").unwrap();
    writeln!(ptx, ") {{").unwrap();

    writeln!(ptx, "    .reg .u32 %r<32>;").unwrap();
    writeln!(ptx, "    .reg .u64 %rd<24>;").unwrap();
    writeln!(ptx, "    .reg .f32 %f<32>;").unwrap();
    writeln!(ptx, "    .reg .pred %p<8>;").unwrap();

    let smem_elems = tile_size * tile_size;
    writeln!(ptx, "    .shared .align 4 .f32 tile_A[{smem_elems}];").unwrap();
    writeln!(ptx, "    .shared .align 4 .f32 tile_B[{smem_elems}];").unwrap();

    writeln!(ptx, "    mov.u32 %r0, %ctaid.z;  // expert_id").unwrap();
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;  // m-tile index").unwrap();
    writeln!(ptx, "    mov.u32 %r2, %ctaid.y;  // n-tile index").unwrap();
    writeln!(ptx, "    mov.u32 %r3, %tid.x;").unwrap();
    writeln!(ptx, "    mov.u32 %r4, %tid.y;").unwrap();
    writeln!(ptx).unwrap();
    writeln!(ptx, "    ld.param.u64 %rd0, [sorted_tokens_ptr];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rd1, [expert_weights_ptr];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rd2, [output_ptr];").unwrap();
    writeln!(ptx, "    ld.param.u64 %rd3, [boundaries_ptr];").unwrap();
    writeln!(ptx, "    ld.param.u32 %r5, [hidden_dim];").unwrap();
    writeln!(ptx, "    ld.param.u32 %r6, [intermediate_dim];").unwrap();
    writeln!(ptx).unwrap();

    writeln!(ptx, "    // Load expert boundaries").unwrap();
    writeln!(ptx, "    cvt.u64.u32 %rd4, %r0;").unwrap();
    writeln!(ptx, "    mul.lo.u64 %rd5, %rd4, 4;").unwrap();
    writeln!(ptx, "    add.u64 %rd6, %rd3, %rd5;").unwrap();
    writeln!(ptx, "    ld.global.u32 %r7, [%rd6];       // start").unwrap();
    writeln!(ptx, "    add.u64 %rd7, %rd6, 4;").unwrap();
    writeln!(ptx, "    ld.global.u32 %r8, [%rd7];       // end").unwrap();
    writeln!(ptx, "    sub.u32 %r9, %r8, %r7;           // num_tokens").unwrap();
    writeln!(ptx, "    setp.eq.u32 %p0, %r9, 0;").unwrap();
    writeln!(ptx, "    @%p0 bra GEMM_DONE;").unwrap();
    writeln!(ptx).unwrap();

    emit_scalar_gemm_body(&mut ptx, tile_size);

    writeln!(ptx, "GEMM_DONE:").unwrap();
    writeln!(ptx, "    ret;").unwrap();
    writeln!(ptx, "}}").unwrap();

    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

/// Emit the scalar (fma.rn.f32) tiled GEMM body.
/// This is the original 16x16 logic, parameterized by `tile_size`.
fn emit_scalar_gemm_body(ptx: &mut String, tile_size: usize) {
    use std::fmt::Write;

    // Thread mapping: row = blockIdx.x * tile_size + threadIdx.y
    writeln!(ptx, "    // Scalar tiled GEMM (tile_size={tile_size})").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r10, %r1, {tile_size};").unwrap();
    writeln!(ptx, "    add.u32 %r10, %r10, %r4;        // global row").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r11, %r2, {tile_size};").unwrap();
    writeln!(ptx, "    add.u32 %r11, %r11, %r3;        // global col").unwrap();
    writeln!(ptx).unwrap();
    writeln!(
        ptx,
        "    setp.ge.u32 %p1, %r10, %r9;     // row >= num_tokens?"
    )
    .unwrap();
    writeln!(
        ptx,
        "    setp.ge.u32 %p2, %r11, %r6;     // col >= intermediate_dim?"
    )
    .unwrap();
    writeln!(ptx, "    mov.f32 %f0, 0.0;               // accumulator").unwrap();
    writeln!(ptx).unwrap();

    // K-dimension tile loop
    writeln!(ptx, "    mov.u32 %r12, 0;                // k_tile_start").unwrap();
    writeln!(ptx, "TILE_LOOP:").unwrap();
    writeln!(
        ptx,
        "    setp.ge.u32 %p3, %r12, %r5;     // k >= hidden_dim?"
    )
    .unwrap();
    writeln!(ptx, "    @%p3 bra TILE_DONE;").unwrap();
    writeln!(ptx, "    bar.sync 0;").unwrap();
    writeln!(ptx).unwrap();

    // Load tile_A: A[row, k_tile_start + tid.x]
    writeln!(
        ptx,
        "    add.u32 %r13, %r12, %r3;        // k = k_tile_start + tid.x"
    )
    .unwrap();
    writeln!(ptx, "    setp.lt.u32 %p4, %r13, %r5;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %p5, %r10, %r9;").unwrap();
    writeln!(ptx, "    and.pred %p4, %p4, %p5;").unwrap();
    writeln!(ptx, "    @!%p4 mov.f32 %f1, 0.0;").unwrap();
    writeln!(ptx, "    @%p4 {{").unwrap();
    writeln!(
        ptx,
        "        add.u32 %r14, %r7, %r10;    // global_row = start + row"
    )
    .unwrap();
    writeln!(ptx, "        mul.lo.u32 %r14, %r14, %r5; // * hidden_dim").unwrap();
    writeln!(ptx, "        add.u32 %r14, %r14, %r13;   // + k").unwrap();
    writeln!(ptx, "        cvt.u64.u32 %rd8, %r14;").unwrap();
    writeln!(ptx, "        mul.lo.u64 %rd8, %rd8, 4;").unwrap();
    writeln!(ptx, "        add.u64 %rd9, %rd0, %rd8;").unwrap();
    writeln!(ptx, "        ld.global.f32 %f1, [%rd9];").unwrap();
    writeln!(ptx, "    }}").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r15, %r4, {tile_size};").unwrap();
    writeln!(ptx, "    add.u32 %r15, %r15, %r3;").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r15, %r15, 4;").unwrap();
    writeln!(ptx, "    st.shared.f32 [tile_A + %r15], %f1;").unwrap();
    writeln!(ptx).unwrap();

    // Load tile_B: B[k_tile_start + tid.y, col]
    writeln!(
        ptx,
        "    add.u32 %r16, %r12, %r4;        // k = k_tile_start + tid.y"
    )
    .unwrap();
    writeln!(ptx, "    setp.lt.u32 %p4, %r16, %r5;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %p5, %r11, %r6;").unwrap();
    writeln!(ptx, "    and.pred %p4, %p4, %p5;").unwrap();
    writeln!(ptx, "    @!%p4 mov.f32 %f2, 0.0;").unwrap();
    writeln!(ptx, "    @%p4 {{").unwrap();
    writeln!(
        ptx,
        "        mul.lo.u32 %r17, %r0, %r5;  // expert * hidden_dim"
    )
    .unwrap();
    writeln!(ptx, "        add.u32 %r17, %r17, %r16;   // + k").unwrap();
    writeln!(
        ptx,
        "        mul.lo.u32 %r17, %r17, %r6; // * intermediate_dim"
    )
    .unwrap();
    writeln!(ptx, "        add.u32 %r17, %r17, %r11;   // + col").unwrap();
    writeln!(ptx, "        cvt.u64.u32 %rd10, %r17;").unwrap();
    writeln!(ptx, "        mul.lo.u64 %rd10, %rd10, 4;").unwrap();
    writeln!(ptx, "        add.u64 %rd11, %rd1, %rd10;").unwrap();
    writeln!(ptx, "        ld.global.f32 %f2, [%rd11];").unwrap();
    writeln!(ptx, "    }}").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r18, %r4, {tile_size};").unwrap();
    writeln!(ptx, "    add.u32 %r18, %r18, %r3;").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r18, %r18, 4;").unwrap();
    writeln!(ptx, "    st.shared.f32 [tile_B + %r18], %f2;").unwrap();
    writeln!(ptx).unwrap();

    writeln!(ptx, "    bar.sync 0;").unwrap();
    writeln!(ptx).unwrap();

    // Inner accumulation loop
    writeln!(ptx, "    mov.u32 %r19, 0;").unwrap();
    writeln!(ptx, "INNER_LOOP:").unwrap();
    writeln!(ptx, "    setp.ge.u32 %p4, %r19, {tile_size};").unwrap();
    writeln!(ptx, "    @%p4 bra INNER_DONE;").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r20, %r4, {tile_size};").unwrap();
    writeln!(ptx, "    add.u32 %r20, %r20, %r19;").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r20, %r20, 4;").unwrap();
    writeln!(ptx, "    ld.shared.f32 %f3, [tile_A + %r20];").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r21, %r19, {tile_size};").unwrap();
    writeln!(ptx, "    add.u32 %r21, %r21, %r3;").unwrap();
    writeln!(ptx, "    mul.lo.u32 %r21, %r21, 4;").unwrap();
    writeln!(ptx, "    ld.shared.f32 %f4, [tile_B + %r21];").unwrap();
    writeln!(ptx, "    fma.rn.f32 %f0, %f3, %f4, %f0;").unwrap();
    writeln!(ptx, "    add.u32 %r19, %r19, 1;").unwrap();
    writeln!(ptx, "    bra INNER_LOOP;").unwrap();
    writeln!(ptx, "INNER_DONE:").unwrap();
    writeln!(ptx).unwrap();

    writeln!(ptx, "    add.u32 %r12, %r12, {tile_size};").unwrap();
    writeln!(ptx, "    bra TILE_LOOP;").unwrap();
    writeln!(ptx, "TILE_DONE:").unwrap();
    writeln!(ptx).unwrap();

    // Store result
    writeln!(ptx, "    or.pred %p4, %p1, %p2;").unwrap();
    writeln!(ptx, "    @%p4 bra GEMM_DONE;").unwrap();
    writeln!(ptx, "    add.u32 %r22, %r7, %r10;        // global_row").unwrap();
    writeln!(
        ptx,
        "    mul.lo.u32 %r22, %r22, %r6;     // * intermediate_dim"
    )
    .unwrap();
    writeln!(ptx, "    add.u32 %r22, %r22, %r11;       // + col").unwrap();
    writeln!(ptx, "    cvt.u64.u32 %rd12, %r22;").unwrap();
    writeln!(ptx, "    mul.lo.u64 %rd12, %rd12, 4;").unwrap();
    writeln!(ptx, "    add.u64 %rd13, %rd2, %rd12;").unwrap();
    writeln!(ptx, "    st.global.f32 [%rd13], %f0;").unwrap();
}


/// Generate PTX for the batched expert GEMM kernel (16x16 scalar).
///
/// This is the default entry point used by `all_moe_kernels()`.
pub fn gen_expert_batched_gemm_ptx() -> Vec<u8> {
    gen_expert_batched_gemm_ptx_tiled(16)
}

/// Generate PTX for the MoE weighted gather kernel.
///
/// Each thread processes one sorted assignment: loads the expert output row,
/// multiplies by the gating weight, and atomically adds to the output at the
/// original token position.
pub fn gen_moe_gather_ptx() -> Vec<u8> {
    let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry moe_gather(
    .param .u64 expert_outputs_ptr,
    .param .u64 reverse_indices_ptr,
    .param .u64 expert_weights_ptr,
    .param .u64 output_ptr,
    .param .u32 num_assigned,
    .param .u32 hidden_dim
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mul.lo.u32 %r3, %r0, %r1;
    add.u32 %r3, %r3, %r2;

    ld.param.u64 %rd0, [expert_outputs_ptr];
    ld.param.u64 %rd1, [reverse_indices_ptr];
    ld.param.u64 %rd2, [expert_weights_ptr];
    ld.param.u64 %rd3, [output_ptr];
    ld.param.u32 %r4, [num_assigned];
    ld.param.u32 %r5, [hidden_dim];

    // Each thread processes one sorted assignment
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra GATHER_DONE;

    // Load original token index
    cvt.u64.u32 %rd4, %r3;
    mul.lo.u64 %rd5, %rd4, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r6, [%rd6];

    // Load gating weight
    add.u64 %rd7, %rd2, %rd5;
    ld.global.f32 %f0, [%rd7];

    // For each dim: output[token, d] += weight * expert_output[tid, d]
    mov.u32 %r7, 0;
GATHER_LOOP:
    setp.ge.u32 %p1, %r7, %r5;
    @%p1 bra GATHER_DONE;

    // Load expert output element
    mul.lo.u32 %r8, %r3, %r5;
    add.u32 %r8, %r8, %r7;
    cvt.u64.u32 %rd8, %r8;
    mul.lo.u64 %rd8, %rd8, 4;
    add.u64 %rd9, %rd0, %rd8;
    ld.global.f32 %f1, [%rd9];

    // weighted value
    mul.f32 %f2, %f0, %f1;

    // Atomic add to output
    mul.lo.u32 %r9, %r6, %r5;
    add.u32 %r9, %r9, %r7;
    cvt.u64.u32 %rd10, %r9;
    mul.lo.u64 %rd10, %rd10, 4;
    add.u64 %rd11, %rd3, %rd10;
    atom.global.add.f32 %f3, [%rd11], %f2;

    add.u32 %r7, %r7, 1;
    bra GATHER_LOOP;

GATHER_DONE:
    ret;
}
"#;
    let mut bytes = ptx.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

/// All MoE PTX kernel names and their generators.
pub fn all_moe_kernels() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "moe_token_sort_histogram",
            gen_moe_token_sort_histogram_ptx(),
        ),
        ("moe_scatter", gen_moe_scatter_ptx()),
        ("expert_batched_gemm", gen_expert_batched_gemm_ptx()),
        ("moe_gather", gen_moe_gather_ptx()),
    ]
}

/// Compute the dynamic shared memory size in bytes for the histogram kernel.
/// Must be passed as `dynamicSmemBytes` when launching `moe_token_sort_histogram`.
pub fn histogram_dynamic_smem_bytes(num_experts: usize) -> usize {
    num_experts * std::mem::size_of::<u32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_histogram_no_hardcoded_64() {
        let ptx_bytes = gen_moe_token_sort_histogram_ptx();
        let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len() - 1]).unwrap();

        // Must NOT contain hardcoded [64] shared memory
        assert!(
            !ptx.contains("local_hist[64]"),
            "Histogram must not hardcode 64-element shared memory"
        );

        // Must use dynamic (extern) shared memory
        assert!(
            ptx.contains(".extern .shared"),
            "Histogram must use extern (dynamic) shared memory"
        );

        // Must have cooperative zero loop (handles num_experts > blockDim.x)
        assert!(
            ptx.contains("HIST_ZERO_LOOP"),
            "Must have cooperative zero initialization loop"
        );

        // Must have cooperative writeback loop
        assert!(
            ptx.contains("HIST_WRITE_LOOP"),
            "Must have cooperative writeback loop"
        );
    }

    #[test]
    fn test_moe_histogram_supports_256_experts() {
        // The histogram kernel uses num_experts from a parameter, not a compile-time constant.
        // 256 experts needs 256 * 4 = 1024 bytes of dynamic shared memory.
        let smem = histogram_dynamic_smem_bytes(256);
        assert_eq!(smem, 1024, "256 experts need 1024 bytes of shared memory");

        // Even 4096 experts would work
        let smem_big = histogram_dynamic_smem_bytes(4096);
        assert_eq!(smem_big, 16384);
    }

    #[test]
    fn test_moe_histogram_still_has_atomic_add() {
        let ptx_bytes = gen_moe_token_sort_histogram_ptx();
        let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len() - 1]).unwrap();

        assert!(
            ptx.contains("atom.shared.add.u32"),
            "Histogram must use atomic add to shared memory"
        );
    }

    #[test]
    fn test_moe_tile_selection_is_scalar_only() {
        // The MMA path was removed; tile selection always returns the
        // 16x16 scalar shape until a future MoE wiring uses the
        // validated `matmul_mma` helpers.
        assert_eq!(select_moe_gemm_tile(256, 512, 512, 80), (16, 16));
        assert_eq!(select_moe_gemm_tile(64, 128, 128, 80), (16, 16));
        assert_eq!(select_moe_gemm_tile(16, 32, 64, 80), (16, 16));
        assert_eq!(select_moe_gemm_tile(256, 512, 512, 70), (16, 16));
        assert_eq!(select_moe_gemm_tile(256, 512, 512, 90), (16, 16));
    }

    #[test]
    fn test_moe_batched_gemm_scalar_16x16() {
        let ptx_bytes = gen_expert_batched_gemm_ptx_tiled(16);
        let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len() - 1]).unwrap();

        assert!(ptx.contains("expert_batched_gemm"), "kernel name");
        assert!(ptx.contains("fma.rn.f32"), "scalar FMA matmul");
        assert!(ptx.contains("tile_A"), "shared memory tile A");
        assert!(ptx.contains("tile_B"), "shared memory tile B");
        assert!(
            ptx.contains("boundaries_ptr"),
            "expert boundaries parameter"
        );
        assert!(ptx.contains(".target sm_70"), "scalar targets sm_52");
        assert!(
            !ptx.contains("mma.sync.aligned"),
            "scalar should NOT have MMA"
        );
    }

    #[test]
    fn test_moe_batched_gemm_mma_path_is_gone() {
        // Belt-and-braces gate: no caller of moe_kernels can produce
        // mma.sync, regardless of tile size. If MMA support is added
        // back, it must go through `matmul_mma::emit_load_*_fragment_smem`
        // (validated end-to-end by `matmul_mma_single_mma_e2e` at 0.0e0
        // diff) — not the inline single-row pattern that previously
        // lived here.
        for &tile in &[16usize, 32, 64, 128] {
            let ptx_bytes = gen_expert_batched_gemm_ptx_tiled(tile);
            let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len() - 1]).unwrap();
            assert!(
                !ptx.contains("mma.sync"),
                "tile_size={tile}: MoE batched GEMM must not emit mma.sync (path was deleted)"
            );
        }
    }

    #[test]
    fn test_moe_batched_gemm_default_is_scalar_16() {
        // gen_expert_batched_gemm_ptx() should produce the 16x16 scalar version
        let ptx_bytes = gen_expert_batched_gemm_ptx();
        let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len() - 1]).unwrap();

        assert!(ptx.contains("fma.rn.f32"), "default should use scalar FMA");
        assert!(
            !ptx.contains("mma.sync.aligned"),
            "default should NOT use MMA"
        );
    }

    #[test]
    fn test_moe_all_kernels() {
        let kernels = all_moe_kernels();
        assert_eq!(kernels.len(), 4, "4 MoE kernels");
        let names: Vec<&str> = kernels.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"moe_token_sort_histogram"));
        assert!(names.contains(&"moe_scatter"));
        assert!(names.contains(&"expert_batched_gemm"));
        assert!(names.contains(&"moe_gather"));
    }
}
