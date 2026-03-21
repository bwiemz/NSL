//! Reduction fusion: detect softmax/layernorm/rmsnorm subgraphs and replace
//! with pre-optimized single-kernel implementations.
//! Uses hand-written PTX templates (like FlashAttention), not the KernelCompiler.

use std::collections::HashSet;
use crate::fusion_graph::{FusionGraph, FusionOp, NodeId, FusedKernelId};
use nsl_semantic::types::DType;

/// A matched reduction pattern in the fusion graph.
#[derive(Debug, Clone)]
pub struct ReductionMatch {
    pub pattern: &'static str,
    pub root_node: NodeId,
    pub input_nodes: Vec<NodeId>,
    pub all_matched_nodes: Vec<NodeId>,
    pub reduction_dim: i64,
    pub is_naive: bool,
    pub has_affine: bool,
}

/// Try to match all reduction patterns in the graph.
/// Returns matches sorted by root node ID.
pub fn detect_reduction_patterns(graph: &FusionGraph) -> Vec<ReductionMatch> {
    let mut matches = Vec::new();

    // Try each pattern at each node
    for node in &graph.nodes {
        if node.fused_into.is_some() {
            continue;
        }

        // Try softmax (look for div nodes that could be the tail)
        if matches!(node.op, FusionOp::Elementwise(ref s) if s == "div") {
            if let Some(m) = try_match_softmax(graph, node.id) {
                if is_valid_reduction_match(graph, &m) {
                    matches.push(m);
                }
            }
        }

        // Try layernorm (look for the final add with beta, or div without affine)
        if matches!(node.op, FusionOp::Elementwise(ref s) if s == "add" || s == "div") {
            if let Some(m) = try_match_layernorm(graph, node.id) {
                if is_valid_reduction_match(graph, &m) {
                    matches.push(m);
                }
            }
        }

        // Try rmsnorm (look for mul with gamma at the end)
        if matches!(node.op, FusionOp::Elementwise(ref s) if s == "mul") {
            if let Some(m) = try_match_rmsnorm(graph, node.id) {
                if is_valid_reduction_match(graph, &m) {
                    matches.push(m);
                }
            }
        }

        // Single-node builtins: softmax(), layernorm(), rmsnorm()
        if let FusionOp::Reduction(ref name) = node.op {
            match name.as_str() {
                "softmax" => {
                    matches.push(ReductionMatch {
                        pattern: "softmax",
                        root_node: node.id,
                        input_nodes: node.inputs.clone(),
                        all_matched_nodes: vec![node.id],
                        reduction_dim: -1,
                        is_naive: false,
                        has_affine: false,
                    });
                }
                "layernorm" => {
                    matches.push(ReductionMatch {
                        pattern: "layernorm",
                        root_node: node.id,
                        input_nodes: node.inputs.clone(),
                        all_matched_nodes: vec![node.id],
                        reduction_dim: -1,
                        is_naive: false,
                        has_affine: node.inputs.len() > 1,
                    });
                }
                "rmsnorm" => {
                    matches.push(ReductionMatch {
                        pattern: "rmsnorm",
                        root_node: node.id,
                        input_nodes: node.inputs.clone(),
                        all_matched_nodes: vec![node.id],
                        reduction_dim: -1,
                        is_naive: false,
                        has_affine: node.inputs.len() > 1,
                    });
                }
                _ => {}
            }
        }
    }

    matches
}

/// Internal vs external consumer rule: only the root node may have external consumers.
fn is_valid_reduction_match(graph: &FusionGraph, matched: &ReductionMatch) -> bool {
    let matched_set: HashSet<NodeId> = matched.all_matched_nodes.iter().copied().collect();

    for &node_id in &matched.all_matched_nodes {
        let node = &graph.nodes[node_id as usize];

        // Skip already-claimed nodes
        if node.fused_into.is_some() || node.no_fuse {
            return false;
        }

        for &consumer in &node.consumers {
            if !matched_set.contains(&consumer) {
                // External consumer — only the root node may have them
                if node_id != matched.root_node {
                    return false;
                }
            }
        }
    }
    true
}

/// Extract reduction dimension from a Reduction node, verifying it's the last dim.
fn get_reduction_dim(graph: &FusionGraph, node_id: NodeId) -> Option<i64> {
    let node = &graph.nodes[node_id as usize];
    // For now, assume reduction is on last dim (-1).
    // Full implementation would check the AST's dim argument.
    if matches!(node.op, FusionOp::Reduction(_)) {
        // Verify it's a contiguous last dimension
        if let Some(ref shape) = node.shape {
            let rank = shape.len() as i64;
            Some(rank - 1)
        } else {
            Some(-1)
        }
    } else {
        None
    }
}

/// Try to match the softmax pattern:
/// Stable: exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))
/// Naive: exp(x) / reduce_sum(exp(x))
fn try_match_softmax(graph: &FusionGraph, div_node: NodeId) -> Option<ReductionMatch> {
    let div = &graph.nodes[div_node as usize];
    if div.inputs.len() != 2 { return None; }

    let numerator_id = div.inputs[0];
    let denominator_id = div.inputs[1];

    let numerator = &graph.nodes[numerator_id as usize];
    let denominator = &graph.nodes[denominator_id as usize];

    // Denominator must be reduce_sum
    if !matches!(denominator.op, FusionOp::Reduction(ref s) if s == "reduce_sum") {
        return None;
    }

    // Numerator must be exp(something)
    if !matches!(numerator.op, FusionOp::Elementwise(ref s) if s == "exp") {
        return None;
    }

    // Check if it's the stable form: exp(x - max(x))
    let exp_input_id = numerator.inputs[0];
    let exp_input = &graph.nodes[exp_input_id as usize];

    let mut all_nodes = vec![div_node, numerator_id, denominator_id];
    let mut is_naive = true;
    let mut x_node = exp_input_id;

    if matches!(exp_input.op, FusionOp::Elementwise(ref s) if s == "sub") {
        // Stable form: sub(x, reduce_max(x))
        if exp_input.inputs.len() == 2 {
            let max_candidate = &graph.nodes[exp_input.inputs[1] as usize];
            if matches!(max_candidate.op, FusionOp::Reduction(ref s) if s == "reduce_max") {
                x_node = exp_input.inputs[0];
                all_nodes.push(exp_input_id);
                all_nodes.push(exp_input.inputs[1]);
                is_naive = false;
            }
        }
    }

    // reduce_sum's input should be the exp node
    if denominator.inputs[0] != numerator_id {
        return None;
    }

    let reduction_dim = get_reduction_dim(graph, denominator_id).unwrap_or(-1);

    Some(ReductionMatch {
        pattern: "softmax",
        root_node: div_node,
        input_nodes: vec![x_node],
        all_matched_nodes: all_nodes,
        reduction_dim,
        is_naive,
        has_affine: false,
    })
}

/// Try to match layernorm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
fn try_match_layernorm(graph: &FusionGraph, candidate: NodeId) -> Option<ReductionMatch> {
    let node = &graph.nodes[candidate as usize];

    // Check if this is the affine tail: add(mul(normalized, gamma), beta)
    if matches!(node.op, FusionOp::Elementwise(ref s) if s == "add") && node.inputs.len() == 2 {
        let mul_candidate = &graph.nodes[node.inputs[0] as usize];
        if matches!(mul_candidate.op, FusionOp::Elementwise(ref s) if s == "mul") {
            // Could be affine layernorm — check deeper
            let div_candidate_id = mul_candidate.inputs[0];
            let div_candidate = &graph.nodes[div_candidate_id as usize];
            if matches!(div_candidate.op, FusionOp::Elementwise(ref s) if s == "div") {
                // Continue checking from div node
                if let Some(inner) = try_match_layernorm_core(graph, div_candidate_id) {
                    let x_node = inner.0;
                    let mut all_nodes = vec![candidate, node.inputs[0]]; // add (beta), mul (gamma)
                    all_nodes.extend(inner.1);
                    let reduction_dim = inner.2;
                    return Some(ReductionMatch {
                        pattern: "layernorm",
                        root_node: candidate,
                        input_nodes: vec![x_node, mul_candidate.inputs[1], node.inputs[1]], // x, gamma, beta
                        all_matched_nodes: all_nodes,
                        reduction_dim,
                        is_naive: false,
                        has_affine: true,
                    });
                }
            }
        }
    }

    // Non-affine: just the div node
    if matches!(node.op, FusionOp::Elementwise(ref s) if s == "div") {
        if let Some(inner) = try_match_layernorm_core(graph, candidate) {
            let x_node = inner.0;
            let all_nodes = inner.1;
            return Some(ReductionMatch {
                pattern: "layernorm",
                root_node: candidate,
                input_nodes: vec![x_node],
                all_matched_nodes: all_nodes,
                reduction_dim: inner.2,
                is_naive: false,
                has_affine: false,
            });
        }
    }

    None
}

/// Match the core layernorm pattern: (x - mean(x)) / sqrt(var(x) + eps)
/// Returns (x_node, matched_node_ids, reduction_dim)
fn try_match_layernorm_core(graph: &FusionGraph, div_id: NodeId) -> Option<(NodeId, Vec<NodeId>, i64)> {
    let div = &graph.nodes[div_id as usize];
    if div.inputs.len() != 2 { return None; }

    // LHS: sub(x, mean(x))
    let sub_id = div.inputs[0];
    let sub = &graph.nodes[sub_id as usize];
    if !matches!(sub.op, FusionOp::Elementwise(ref s) if s == "sub") { return None; }
    if sub.inputs.len() != 2 { return None; }

    let x_node = sub.inputs[0];
    let mean_id = sub.inputs[1];
    let mean = &graph.nodes[mean_id as usize];
    if !matches!(mean.op, FusionOp::Reduction(ref s) if s == "mean") { return None; }

    // RHS: sqrt(var(x) + eps) or sqrt(add(var(x), eps))
    let sqrt_id = div.inputs[1];
    let sqrt = &graph.nodes[sqrt_id as usize];
    if !matches!(sqrt.op, FusionOp::Elementwise(ref s) if s == "sqrt") { return None; }

    let add_eps_id = sqrt.inputs[0];
    let add_eps = &graph.nodes[add_eps_id as usize];
    if !matches!(add_eps.op, FusionOp::Elementwise(ref s) if s == "add") { return None; }

    let var_id = add_eps.inputs[0];
    let var_node = &graph.nodes[var_id as usize];
    if !matches!(var_node.op, FusionOp::Reduction(ref s) if s == "var") { return None; }

    let reduction_dim = get_reduction_dim(graph, mean_id).unwrap_or(-1);

    let nodes = vec![div_id, sub_id, mean_id, sqrt_id, add_eps_id, var_id];
    Some((x_node, nodes, reduction_dim))
}

/// Try to match rmsnorm: x / sqrt(mean(x^2) + eps) * gamma
fn try_match_rmsnorm(graph: &FusionGraph, mul_node: NodeId) -> Option<ReductionMatch> {
    let mul = &graph.nodes[mul_node as usize];
    if mul.inputs.len() != 2 { return None; }

    // One input should be div(x, sqrt(mean(x^2) + eps)), other is gamma
    let div_id = mul.inputs[0];
    let div = &graph.nodes[div_id as usize];
    if !matches!(div.op, FusionOp::Elementwise(ref s) if s == "div") { return None; }
    if div.inputs.len() != 2 { return None; }

    let x_node = div.inputs[0];

    // sqrt(mean(x^2) + eps)
    let sqrt_id = div.inputs[1];
    let sqrt = &graph.nodes[sqrt_id as usize];
    if !matches!(sqrt.op, FusionOp::Elementwise(ref s) if s == "sqrt") { return None; }

    let add_eps_id = sqrt.inputs[0];
    let add_eps = &graph.nodes[add_eps_id as usize];
    if !matches!(add_eps.op, FusionOp::Elementwise(ref s) if s == "add") { return None; }

    let mean_id = add_eps.inputs[0];
    let mean = &graph.nodes[mean_id as usize];
    if !matches!(mean.op, FusionOp::Reduction(ref s) if s == "mean") { return None; }

    // mean input should be x^2 = mul(x, x)
    let sq_id = mean.inputs[0];
    let sq = &graph.nodes[sq_id as usize];
    if !matches!(sq.op, FusionOp::Elementwise(ref s) if s == "mul") { return None; }
    if sq.inputs[0] != x_node || sq.inputs[1] != x_node { return None; }

    let reduction_dim = get_reduction_dim(graph, mean_id).unwrap_or(-1);

    Some(ReductionMatch {
        pattern: "rmsnorm",
        root_node: mul_node,
        input_nodes: vec![x_node, mul.inputs[1]], // x, gamma
        all_matched_nodes: vec![mul_node, div_id, sqrt_id, add_eps_id, mean_id, sq_id],
        reduction_dim,
        is_naive: false,
        has_affine: true,
    })
}

// ─── PTX synthesis helpers ────────────────────────────────────────────────────

/// Emit warp-level shuffle-down reduction for a single f32 register `%freg`.
/// Uses shuttle registers %rs0/%rs1 to comply with PTX ISA (.b32 only for shfl).
fn emit_warp_reduce_f32(buf: &mut Vec<u8>, freg: &str, reduce_op: &str) {
    for offset in [16u32, 8, 4, 2, 1] {
        buf.extend_from_slice(
            format!(
                "    mov.b32 %rs0, {freg};\n\
                 shfl.sync.down.b32 %rs1, %rs0, {offset}, 31, 0xFFFFFFFF;\n\
                 mov.b32 %f_shfl, %rs1;\n\
                 {reduce_op}.f32 {freg}, {freg}, %f_shfl;\n"
            )
            .as_bytes(),
        );
    }
}

/// Emit cross-warp reduction via shared memory.
/// tid in %r_tid, warp_id = tid>>5, lane_id = tid&31.
/// After this, lane 0 of warp 0 holds the block-wide result in `%freg`.
fn emit_crosswarp_reduce(buf: &mut Vec<u8>, freg: &str, reduce_op: &str, n_warps: u32) {
    // Only lane 0 of each warp writes to shared mem
    buf.extend_from_slice(
        format!(
            "    // cross-warp reduction into shared memory\n\
             and.b32 %r_lane, %r_tid, 31;\n\
             shr.u32 %r_warp, %r_tid, 5;\n\
             setp.eq.u32 %p_lane0, %r_lane, 0;\n\
             @%p_lane0 st.shared.f32 [%smem + %r_warp * 4], {freg};\n\
             bar.sync 0;\n\
             // Only threads 0..{n_warps} load; others load identity
             setp.lt.u32 %p_warp_slot, %r_tid, {n_warps};\n\
             @%p_warp_slot ld.shared.f32 {freg}, [%smem + %r_tid * 4];\n\
             @!%p_warp_slot mov.f32 {freg}, 0.0;\n\
             bar.sync 0;\n"
        )
        .as_bytes(),
    );
    // Warp 0 does a final warp reduce across the partial sums
    emit_warp_reduce_f32(buf, freg, reduce_op);
}

/// Returns the store + optional dtype-convert instruction for a result in `%f_res`.
fn emit_store_with_dtype(buf: &mut Vec<u8>, dtype: DType, out_ptr_reg: &str) {
    match dtype {
        DType::Fp16 | DType::Bf16 => {
            buf.extend_from_slice(
                format!(
                    "    cvt.rn.f16.f32 %h_out, %f_res;\n\
                     st.global.b16 [{out_ptr_reg}], %h_out;\n"
                )
                .as_bytes(),
            );
        }
        _ => {
            buf.extend_from_slice(
                format!("    st.global.f32 [{out_ptr_reg}], %f_res;\n").as_bytes(),
            );
        }
    }
}

// ─── Public PTX synthesis functions ──────────────────────────────────────────

/// Synthesize a fused softmax PTX kernel for rows of `hidden_dim` elements.
/// Two-pass: find max → exp(x-max) → sum → normalize.
/// Returns null-terminated PTX bytes.
pub fn synthesize_fused_softmax_ptx(hidden_dim: u32, dtype: DType) -> Vec<u8> {
    let block_size: u32 = 256.min(hidden_dim);
    let n_warps: u32 = block_size / 32;
    let elems_per_thread: u32 = hidden_dim.div_ceil(block_size);
    let name = format!("fused_softmax_{hidden_dim}");

    let mut buf: Vec<u8> = Vec::new();

    // PTX header
    buf.extend_from_slice(
        b".version 7.0\n\
          .target sm_80\n\
          .address_size 64\n\n",
    );

    // Kernel signature
    buf.extend_from_slice(
        format!(
            ".visible .entry {name}(\n\
             .param .u64 param_out,\n\
             .param .u64 param_in,\n\
             .param .u32 param_rows\n\
             )\n{{\n"
        )
        .as_bytes(),
    );

    // Register declarations
    buf.extend_from_slice(
        format!(
            "    .reg .u64 %rd<8>;\n\
             .reg .u32 %r<16>;\n\
             .reg .f32 %f<32>;\n\
             .reg .f32 %f_shfl;\n\
             .reg .f32 %f_res;\n\
             .reg .b32 %rs<4>;\n\
             .reg .pred %p<8>;\n\
             .reg .pred %p_lane0;\n\
             .reg .pred %p_warp_slot;\n\
             .reg .u32 %r_tid;\n\
             .reg .u32 %r_lane;\n\
             .reg .u32 %r_warp;\n\
             .reg .b16 %h_out;\n\
             .shared .align 4 .b8 %smem[{smem_bytes}];\n\n"
        , smem_bytes = n_warps * 4)
        .as_bytes(),
    );

    // Load params
    buf.extend_from_slice(
        b"    ld.param.u64 %rd0, [param_out];\n\
          ld.param.u64 %rd1, [param_in];\n\
          ld.param.u32 %r0, [param_rows];\n\
          mov.u32 %r_tid, %tid.x;\n\
          mov.u32 %r1, %ctaid.x;  // row index\n\n"
    );

    // Row bounds check
    buf.extend_from_slice(
        b"    setp.ge.u32 %p0, %r1, %r0;\n\
          @%p0 ret;\n\n"
    );

    // Compute base pointers for this row
    buf.extend_from_slice(
        format!(
            "    // in_ptr = param_in + row * hidden_dim * sizeof(f32)\n\
             mul.wide.u32 %rd2, %r1, {row_bytes};\n\
             add.u64 %rd3, %rd1, %rd2;  // in row base\n\
             mul.wide.u32 %rd4, %r1, {row_bytes};\n\
             add.u64 %rd5, %rd0, %rd4;  // out row base\n\n"
        , row_bytes = hidden_dim * 4)
        .as_bytes(),
    );

    // ── Pass 1: find row max ──────────────────────────────────────────────────
    buf.extend_from_slice(b"    // Pass 1: find row max\n");
    buf.extend_from_slice(b"    mov.f32 %f0, 0fFF800000;  // -inf\n");

    for e in 0..elems_per_thread {
        let _idx = format!("%r_tid + {}", e * block_size);
        buf.extend_from_slice(
            format!(
                "    // element {e}\n\
                 add.u32 %r2, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r2, {hidden_dim};\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 @!%p1 mov.f32 %f{fp}, 0fFF800000;\n\
                 max.f32 %f0, %f0, %f{fp};\n"
            , off = e * block_size, fp = e + 1)
            .as_bytes(),
        );
    }

    buf.extend_from_slice(b"\n    // Warp reduce max\n");
    emit_warp_reduce_f32(&mut buf, "%f0", "max");
    emit_crosswarp_reduce(&mut buf, "%f0", "max", n_warps);
    // Broadcast row max to all threads
    buf.extend_from_slice(
        b"    // Broadcast max from thread 0 to all threads\n\
          mov.b32 %rs0, %f0;\n\
          shfl.sync.idx.b32 %rs1, %rs0, 0, 31, 0xFFFFFFFF;\n\
          mov.b32 %f0, %rs1;\n\n"
    );

    // ── Pass 1b: exp(x - max) and sum ────────────────────────────────────────
    buf.extend_from_slice(b"    // Pass 1b: exp(x - max) and accumulate sum\n");
    buf.extend_from_slice(b"    mov.f32 %f16, 0f00000000;  // sum = 0\n");
    // log2(e) constant: 0x3FB8AA3B
    buf.extend_from_slice(b"    mov.f32 %f17, 0f3FB8AA3B;  // log2(e)\n");

    for e in 0..elems_per_thread {
        buf.extend_from_slice(
            format!(
                "    add.u32 %r2, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r2, {hidden_dim};\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 @!%p1 mov.f32 %f{fp}, 0fFF800000;\n\
                 sub.f32 %f{fp}, %f{fp}, %f0;\n\
                 mul.f32 %f{fp}, %f{fp}, %f17;\n\
                 ex2.approx.f32 %f{fp}, %f{fp};\n\
                 add.f32 %f16, %f16, %f{fp};\n"
            , off = e * block_size, fp = e + 1)
            .as_bytes(),
        );
    }

    buf.extend_from_slice(b"\n    // Warp reduce sum\n");
    emit_warp_reduce_f32(&mut buf, "%f16", "add");
    emit_crosswarp_reduce(&mut buf, "%f16", "add", n_warps);
    // Broadcast sum
    buf.extend_from_slice(
        b"    mov.b32 %rs0, %f16;\n\
          shfl.sync.idx.b32 %rs1, %rs0, 0, 31, 0xFFFFFFFF;\n\
          mov.b32 %f16, %rs1;\n"
    );
    // 1/sum
    buf.extend_from_slice(b"    rcp.approx.f32 %f18, %f16;  // 1/sum\n\n");

    // ── Pass 2: write output ──────────────────────────────────────────────────
    buf.extend_from_slice(b"    // Pass 2: write output\n");
    for e in 0..elems_per_thread {
        buf.extend_from_slice(
            format!(
                "    add.u32 %r2, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r2, {hidden_dim};\n\
                 // recompute exp(x-max)\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 @!%p1 mov.f32 %f{fp}, 0f00000000;\n\
                 @%p1 sub.f32 %f{fp}, %f{fp}, %f0;\n\
                 @%p1 mul.f32 %f{fp}, %f{fp}, %f17;\n\
                 @%p1 ex2.approx.f32 %f{fp}, %f{fp};\n\
                 mul.f32 %f_res, %f{fp}, %f18;\n\
                 // compute output pointer\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd5, %rd6;\n"
            , off = e * block_size, fp = e + 1)
            .as_bytes(),
        );
        // Predicated store
        buf.extend_from_slice(b"    @%p1 ");
        emit_store_with_dtype(&mut buf, dtype, "%rd7");
    }

    buf.extend_from_slice(b"\n    ret;\n}\n");
    buf.push(0); // null terminator
    buf
}

/// Synthesize a fused layernorm PTX kernel using Welford's online algorithm.
/// Returns null-terminated PTX bytes.
///
/// TODO(M31-followup): The cross-warp Welford merge is simplified (adds means then
/// divides by n_warps). This is only correct when every thread processes the same
/// number of elements. For hidden_dim not divisible by block_size, use proper
/// parallel Welford merge: combined_mean = (n1*m1 + n2*m2)/(n1+n2), or switch
/// to sum-based reduction and divide by hidden_dim at the end.
pub fn synthesize_fused_layernorm_ptx(hidden_dim: u32, has_affine: bool, eps: f64, dtype: DType) -> Vec<u8> {
    let block_size: u32 = 256.min(hidden_dim);
    let n_warps: u32 = block_size / 32;
    let elems_per_thread: u32 = hidden_dim.div_ceil(block_size);
    let name = format!("fused_layernorm_{hidden_dim}");
    let eps_f32 = eps as f32;
    let eps_bits = eps_f32.to_bits();

    let mut buf: Vec<u8> = Vec::new();

    // PTX header
    buf.extend_from_slice(
        b".version 7.0\n\
          .target sm_80\n\
          .address_size 64\n\n",
    );

    // Kernel signature
    buf.extend_from_slice(b".visible .entry ");
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(b"(\n.param .u64 param_out,\n.param .u64 param_in,\n");
    if has_affine {
        buf.extend_from_slice(b".param .u64 param_gamma,\n.param .u64 param_beta,\n");
    }
    buf.extend_from_slice(b".param .u32 param_rows\n)\n{\n");

    // Register declarations
    buf.extend_from_slice(
        format!(
            "    .reg .u64 %rd<16>;\n\
             .reg .u32 %r<16>;\n\
             .reg .f32 %f<32>;\n\
             .reg .f32 %f_shfl;\n\
             .reg .f32 %f_res;\n\
             .reg .b32 %rs<4>;\n\
             .reg .pred %p<8>;\n\
             .reg .pred %p_lane0;\n\
             .reg .pred %p_warp_slot;\n\
             .reg .u32 %r_tid;\n\
             .reg .u32 %r_lane;\n\
             .reg .u32 %r_warp;\n\
             .reg .b16 %h_out;\n\
             .shared .align 4 .b8 %smem[{smem_bytes}];\n\n"
        , smem_bytes = n_warps * 8)  // 2 floats per warp: mean + M2
        .as_bytes(),
    );

    // Load params
    buf.extend_from_slice(
        b"    ld.param.u64 %rd0, [param_out];\n\
          ld.param.u64 %rd1, [param_in];\n"
    );
    if has_affine {
        buf.extend_from_slice(
            b"    ld.param.u64 %rd8, [param_gamma];\n\
              ld.param.u64 %rd9, [param_beta];\n"
        );
    }
    buf.extend_from_slice(
        b"    ld.param.u32 %r0, [param_rows];\n\
          mov.u32 %r_tid, %tid.x;\n\
          mov.u32 %r1, %ctaid.x;\n\n"
    );

    // Row bounds check
    buf.extend_from_slice(
        b"    setp.ge.u32 %p0, %r1, %r0;\n\
          @%p0 ret;\n\n"
    );

    // Row base pointers
    buf.extend_from_slice(
        format!(
            "    mul.wide.u32 %rd2, %r1, {row_bytes};\n\
             add.u64 %rd3, %rd1, %rd2;\n\
             mul.wide.u32 %rd4, %r1, {row_bytes};\n\
             add.u64 %rd5, %rd0, %rd4;\n\n"
        , row_bytes = hidden_dim * 4)
        .as_bytes(),
    );

    // ── Welford accumulation (simplified: compute mean + sum of squares) ──────
    buf.extend_from_slice(b"    // Welford: accumulate mean and M2\n");
    buf.extend_from_slice(b"    mov.f32 %f0, 0f00000000;  // mean\n");
    buf.extend_from_slice(b"    mov.f32 %f1, 0f00000000;  // M2 (sum of (x-mean)^2)\n");
    buf.extend_from_slice(b"    mov.u32 %r2, 0;            // count\n");

    for e in 0..elems_per_thread {
        buf.extend_from_slice(
            format!(
                "    add.u32 %r3, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r3, {hidden_dim};\n\
                 mul.wide.u32 %rd6, %r3, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 @!%p1 mov.f32 %f{fp}, 0f00000000;\n\
                 // Welford update: count++, delta = x - mean, mean += delta/count, M2 += delta*(x-mean_new)\n\
                 @%p1 add.u32 %r2, %r2, 1;\n\
                 @%p1 sub.f32 %f20, %f{fp}, %f0;\n\
                 cvt.rn.f32.u32 %f21, %r2;\n\
                 @%p1 rcp.approx.f32 %f22, %f21;\n\
                 @%p1 mul.f32 %f23, %f20, %f22;\n\
                 @%p1 add.f32 %f0, %f0, %f23;\n\
                 @%p1 sub.f32 %f24, %f{fp}, %f0;\n\
                 @%p1 mul.f32 %f24, %f20, %f24;\n\
                 @%p1 add.f32 %f1, %f1, %f24;\n"
            , off = e * block_size, fp = e + 2)
            .as_bytes(),
        );
    }

    // Warp reduce mean (sum partial means, then divide)
    buf.extend_from_slice(b"\n    // Warp + cross-warp reduce mean\n");
    emit_warp_reduce_f32(&mut buf, "%f0", "add");
    emit_crosswarp_reduce(&mut buf, "%f0", "add", n_warps);
    // Broadcast mean
    buf.extend_from_slice(
        b"    mov.b32 %rs0, %f0;\n\
          shfl.sync.idx.b32 %rs1, %rs0, 0, 31, 0xFFFFFFFF;\n\
          mov.b32 %f0, %rs1;\n"
    );
    // Divide by n_warps to get true mean
    buf.extend_from_slice(
        format!("    mul.f32 %f0, %f0, 0f{inv_warps:08X};\n\n", inv_warps = (1.0f32 / n_warps as f32).to_bits())
        .as_bytes(),
    );

    // Warp reduce M2 (variance numerator)
    buf.extend_from_slice(b"    // Warp + cross-warp reduce M2\n");
    emit_warp_reduce_f32(&mut buf, "%f1", "add");
    emit_crosswarp_reduce(&mut buf, "%f1", "add", n_warps);
    buf.extend_from_slice(
        b"    mov.b32 %rs0, %f1;\n\
          shfl.sync.idx.b32 %rs1, %rs0, 0, 31, 0xFFFFFFFF;\n\
          mov.b32 %f1, %rs1;\n"
    );
    // var = M2 / hidden_dim
    buf.extend_from_slice(
        format!("    mul.f32 %f1, %f1, 0f{inv_n:08X};  // var = M2/N\n", inv_n = (1.0f32 / hidden_dim as f32).to_bits())
        .as_bytes(),
    );
    // var + eps
    buf.extend_from_slice(
        format!("    add.f32 %f1, %f1, 0f{eps_bits:08X};  // var + eps\n")
        .as_bytes(),
    );
    // 1/sqrt(var+eps)
    buf.extend_from_slice(b"    rsqrt.approx.f32 %f2, %f1;  // 1/sqrt(var+eps)\n\n");

    // ── Write output ──────────────────────────────────────────────────────────
    buf.extend_from_slice(b"    // Normalize and write output\n");
    for e in 0..elems_per_thread {
        buf.extend_from_slice(
            format!(
                "    add.u32 %r3, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r3, {hidden_dim};\n\
                 mul.wide.u32 %rd6, %r3, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 // normalized = (x - mean) * rsqrt\n\
                 @%p1 sub.f32 %f{fp}, %f{fp}, %f0;\n\
                 @%p1 mul.f32 %f{fp}, %f{fp}, %f2;\n"
            , off = e * block_size, fp = e + 2)
            .as_bytes(),
        );
        if has_affine {
            buf.extend_from_slice(
                format!(
                    "                 // affine: gamma * normalized + beta\n\
                     add.u64 %rd10, %rd8, %rd6;\n\
                     @%p1 ld.global.f32 %f28, [%rd10];\n\
                     add.u64 %rd11, %rd9, %rd6;\n\
                     @%p1 ld.global.f32 %f29, [%rd11];\n\
                     @%p1 mul.f32 %f{fp}, %f{fp}, %f28;\n\
                     @%p1 add.f32 %f{fp}, %f{fp}, %f29;\n"
                , fp = e + 2)
                .as_bytes(),
            );
        }
        buf.extend_from_slice(
            format!(
                "    mov.f32 %f_res, %f{fp};\n\
                 mul.wide.u32 %rd6, %r3, 4;\n\
                 add.u64 %rd7, %rd5, %rd6;\n\
                 @%p1 "
            , fp = e + 2)
            .as_bytes(),
        );
        emit_store_with_dtype(&mut buf, dtype, "%rd7");
    }

    buf.extend_from_slice(b"\n    ret;\n}\n");
    buf.push(0);
    buf
}

/// Synthesize a fused RMSNorm PTX kernel.
/// Single-pass: accumulate x^2, reduce, rsqrt, normalize, optional gamma scale.
/// Returns null-terminated PTX bytes.
pub fn synthesize_fused_rmsnorm_ptx(hidden_dim: u32, has_affine: bool, eps: f64, dtype: DType) -> Vec<u8> {
    let block_size: u32 = 256.min(hidden_dim);
    let n_warps: u32 = block_size / 32;
    let elems_per_thread: u32 = hidden_dim.div_ceil(block_size);
    let name = format!("fused_rmsnorm_{hidden_dim}");
    let eps_f32 = eps as f32;
    let eps_bits = eps_f32.to_bits();

    let mut buf: Vec<u8> = Vec::new();

    // PTX header
    buf.extend_from_slice(
        b".version 7.0\n\
          .target sm_80\n\
          .address_size 64\n\n",
    );

    // Kernel signature
    buf.extend_from_slice(b".visible .entry ");
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(b"(\n.param .u64 param_out,\n.param .u64 param_in,\n");
    if has_affine {
        buf.extend_from_slice(b".param .u64 param_gamma,\n");
    }
    buf.extend_from_slice(b".param .u32 param_rows\n)\n{\n");

    // Register declarations
    buf.extend_from_slice(
        format!(
            "    .reg .u64 %rd<12>;\n\
             .reg .u32 %r<16>;\n\
             .reg .f32 %f<32>;\n\
             .reg .f32 %f_shfl;\n\
             .reg .f32 %f_res;\n\
             .reg .b32 %rs<4>;\n\
             .reg .pred %p<8>;\n\
             .reg .pred %p_lane0;\n\
             .reg .pred %p_warp_slot;\n\
             .reg .u32 %r_tid;\n\
             .reg .u32 %r_lane;\n\
             .reg .u32 %r_warp;\n\
             .reg .b16 %h_out;\n\
             .shared .align 4 .b8 %smem[{smem_bytes}];\n\n"
        , smem_bytes = n_warps * 4)
        .as_bytes(),
    );

    // Load params
    buf.extend_from_slice(
        b"    ld.param.u64 %rd0, [param_out];\n\
          ld.param.u64 %rd1, [param_in];\n"
    );
    if has_affine {
        buf.extend_from_slice(b"    ld.param.u64 %rd8, [param_gamma];\n");
    }
    buf.extend_from_slice(
        b"    ld.param.u32 %r0, [param_rows];\n\
          mov.u32 %r_tid, %tid.x;\n\
          mov.u32 %r1, %ctaid.x;\n\n"
    );

    // Row bounds check
    buf.extend_from_slice(
        b"    setp.ge.u32 %p0, %r1, %r0;\n\
          @%p0 ret;\n\n"
    );

    // Row base pointers
    buf.extend_from_slice(
        format!(
            "    mul.wide.u32 %rd2, %r1, {row_bytes};\n\
             add.u64 %rd3, %rd1, %rd2;\n\
             mul.wide.u32 %rd4, %r1, {row_bytes};\n\
             add.u64 %rd5, %rd0, %rd4;\n\n"
        , row_bytes = hidden_dim * 4)
        .as_bytes(),
    );

    // ── Pass 1: accumulate x^2 ────────────────────────────────────────────────
    buf.extend_from_slice(b"    // Accumulate sum of squares\n");
    buf.extend_from_slice(b"    mov.f32 %f0, 0f00000000;  // sum_sq\n");

    for e in 0..elems_per_thread {
        buf.extend_from_slice(
            format!(
                "    add.u32 %r2, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r2, {hidden_dim};\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 @!%p1 mov.f32 %f{fp}, 0f00000000;\n\
                 mul.f32 %f{sq}, %f{fp}, %f{fp};\n\
                 add.f32 %f0, %f0, %f{sq};\n"
            , off = e * block_size, fp = e + 1, sq = e + elems_per_thread + 1)
            .as_bytes(),
        );
    }

    // Warp + cross-warp reduce sum_sq
    buf.extend_from_slice(b"\n    // Warp + cross-warp reduce sum_sq\n");
    emit_warp_reduce_f32(&mut buf, "%f0", "add");
    emit_crosswarp_reduce(&mut buf, "%f0", "add", n_warps);
    buf.extend_from_slice(
        b"    mov.b32 %rs0, %f0;\n\
          shfl.sync.idx.b32 %rs1, %rs0, 0, 31, 0xFFFFFFFF;\n\
          mov.b32 %f0, %rs1;\n"
    );

    // mean_sq = sum_sq / hidden_dim
    buf.extend_from_slice(
        format!("    mul.f32 %f0, %f0, 0f{inv_n:08X};  // mean_sq = sum_sq / N\n", inv_n = (1.0f32 / hidden_dim as f32).to_bits())
        .as_bytes(),
    );
    // mean_sq + eps
    buf.extend_from_slice(
        format!("    add.f32 %f0, %f0, 0f{eps_bits:08X};  // mean_sq + eps\n")
        .as_bytes(),
    );
    // 1/sqrt(mean_sq + eps)
    buf.extend_from_slice(b"    rsqrt.approx.f32 %f16, %f0;  // 1/sqrt(mean_sq+eps)\n\n");

    // ── Pass 2: normalize and write ───────────────────────────────────────────
    buf.extend_from_slice(b"    // Normalize and write output\n");
    for e in 0..elems_per_thread {
        buf.extend_from_slice(
            format!(
                "    add.u32 %r2, %r_tid, {off};\n\
                 setp.lt.u32 %p1, %r2, {hidden_dim};\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd3, %rd6;\n\
                 @%p1 ld.global.f32 %f{fp}, [%rd7];\n\
                 @%p1 mul.f32 %f{fp}, %f{fp}, %f16;\n"
            , off = e * block_size, fp = e + 1)
            .as_bytes(),
        );
        if has_affine {
            buf.extend_from_slice(
                format!(
                    "                 // scale by gamma\n\
                     add.u64 %rd9, %rd8, %rd6;\n\
                     @%p1 ld.global.f32 %f28, [%rd9];\n\
                     @%p1 mul.f32 %f{fp}, %f{fp}, %f28;\n"
                , fp = e + 1)
                .as_bytes(),
            );
        }
        buf.extend_from_slice(
            format!(
                "    mov.f32 %f_res, %f{fp};\n\
                 mul.wide.u32 %rd6, %r2, 4;\n\
                 add.u64 %rd7, %rd5, %rd6;\n\
                 @%p1 "
            , fp = e + 1)
            .as_bytes(),
        );
        emit_store_with_dtype(&mut buf, dtype, "%rd7");
    }

    buf.extend_from_slice(b"\n    ret;\n}\n");
    buf.push(0);
    buf
}

/// Mark all nodes in detected reduction matches as fused.
pub fn apply_reduction_fusion(graph: &mut FusionGraph, matches: &[ReductionMatch], base_kernel_id: FusedKernelId) {
    for (i, m) in matches.iter().enumerate() {
        let kid = base_kernel_id + i as FusedKernelId;
        for &node_id in &m.all_matched_nodes {
            graph.nodes[node_id as usize].fused_into = Some(kid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_graph::FusionGraph;

    fn make_stable_softmax() -> (FusionGraph, NodeId) {
        // exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let rmax = g.add_node(FusionOp::Reduction("reduce_max".into()), vec![x]);
        g.set_type_info(rmax, vec![1024], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, rmax]);
        let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![sub]);
        let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
        g.set_type_info(rsum, vec![1024], DType::F32);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![exp, rsum]);
        g.mark_graph_output(div);
        g.build_consumers();
        (g, div)
    }

    fn make_naive_softmax() -> (FusionGraph, NodeId) {
        // exp(x) / reduce_sum(exp(x))
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![x]);
        let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
        g.set_type_info(rsum, vec![1024], DType::F32);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![exp, rsum]);
        g.mark_graph_output(div);
        g.build_consumers();
        (g, div)
    }

    #[test]
    fn test_detect_stable_softmax() {
        let (g, _) = make_stable_softmax();
        let matches = detect_reduction_patterns(&g);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern, "softmax");
        assert!(!matches[0].is_naive);
        assert!(matches[0].all_matched_nodes.len() >= 3);
    }

    #[test]
    fn test_detect_naive_softmax() {
        let (g, _) = make_naive_softmax();
        let matches = detect_reduction_patterns(&g);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern, "softmax");
        assert!(matches[0].is_naive);
    }

    #[test]
    fn test_layernorm_basic_pattern() {
        // Standard layernorm: (x - mean(x)) / sqrt(var(x) + eps)
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![x]);
        g.set_type_info(mean, vec![768], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, mean]);
        let var = g.add_node(FusionOp::Reduction("var".into()), vec![x]);
        g.set_type_info(var, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![var, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![sub, sqrt]);
        g.mark_graph_output(div);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let ln_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "layernorm").collect();
        assert_eq!(ln_matches.len(), 1);
    }

    #[test]
    fn test_layernorm_internal_multi_consumer() {
        // sub(x, mean) is consumed by BOTH div and the squaring step for variance.
        // This tests the internal multi-consumer rule: sub has 2 consumers but
        // both are inside the matched subgraph.
        // Pattern: mean, sub, mul(sub,sub), mean_sq, add_eps, sqrt, div
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![x]);
        g.set_type_info(mean, vec![768], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, mean]);
        // Manual variance: mean((x - mean)^2) — sub consumed by both sq and div
        let sq = g.add_node(FusionOp::Elementwise("mul".into()), vec![sub, sub]);
        let var_mean = g.add_node(FusionOp::Reduction("mean".into()), vec![sq]);
        g.set_type_info(var_mean, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![var_mean, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![sub, sqrt]);
        g.mark_graph_output(div);
        g.build_consumers();

        // sub has 3 consumer edges: sq uses sub twice (mul(sub, sub)) plus div uses it once.
        // build_consumers counts edges, not unique consumer nodes.
        assert_eq!(g.nodes[sub as usize].consumers.len(), 3);

        // NOTE: The current try_match_layernorm_core uses Reduction("var") not
        // this manual expansion. This test documents the desired behavior for
        // manual variance patterns. The implementer should extend the matcher
        // or accept that only var-builtin patterns are matched in M31.
    }

    #[test]
    fn test_layernorm_external_consumer_rejected() {
        // mean(x) consumed by sub AND an external node -> should be rejected
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![x]);
        g.set_type_info(mean, vec![768], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, mean]);
        let var = g.add_node(FusionOp::Reduction("var".into()), vec![x]);
        g.set_type_info(var, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![var, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![sub, sqrt]);
        g.mark_graph_output(div);
        // External consumer of mean (outside the layernorm subgraph)
        let _external = g.add_named_node("leak".into(), FusionOp::Elementwise("relu".into()), vec![mean]);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let ln_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "layernorm").collect();
        assert_eq!(ln_matches.len(), 0); // rejected due to external consumer on mean
    }

    #[test]
    fn test_single_node_softmax_builtin() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        let sm = g.add_node(FusionOp::Reduction("softmax".into()), vec![x]);
        g.mark_graph_output(sm);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern, "softmax");
        assert!(!matches[0].is_naive);
    }

    #[test]
    fn test_rmsnorm_pattern() {
        // x / sqrt(mean(x^2) + eps) * gamma
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let gamma = g.add_named_node("gamma".into(), FusionOp::Input, vec![]);
        let sq = g.add_node(FusionOp::Elementwise("mul".into()), vec![x, x]); // x^2
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![sq]);
        g.set_type_info(mean, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![mean, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_node(FusionOp::Elementwise("div".into()), vec![x, sqrt]);
        let out = g.add_named_node("out".into(), FusionOp::Elementwise("mul".into()), vec![div, gamma]);
        g.mark_graph_output(out);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let rms_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "rmsnorm").collect();
        assert_eq!(rms_matches.len(), 1);
        assert!(rms_matches[0].has_affine);
    }

    #[test]
    fn test_apply_reduction_marks_fused_into() {
        let (mut g, _) = make_stable_softmax();
        let matches = detect_reduction_patterns(&g);
        apply_reduction_fusion(&mut g, &matches, 100);

        // All matched nodes should be marked
        for &nid in &matches[0].all_matched_nodes {
            assert_eq!(g.nodes[nid as usize].fused_into, Some(100));
        }
    }

    #[test]
    fn test_already_claimed_skipped() {
        let (mut g, div) = make_stable_softmax();
        // Pre-claim the div node (the softmax tail)
        g.nodes[div as usize].fused_into = Some(99);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let softmax_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "softmax").collect();
        assert_eq!(softmax_matches.len(), 0);
    }

    #[test]
    fn test_synthesize_softmax_ptx() {
        let ptx = synthesize_fused_softmax_ptx(1024, DType::Fp16);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        assert!(ptx_str.contains(".entry fused_softmax_1024"));
        // Two-pass structure
        assert!(ptx_str.contains("shfl.sync.down.b32"));
        assert!(ptx_str.contains("bar.sync"));
        // Must have exp and div
        assert!(ptx_str.contains("ex2.approx.f32"));
        assert!(ptx_str.contains("rcp.approx.f32") || ptx_str.contains("div.rn.f32"));
        // Output dtype conversion
        assert!(ptx_str.contains("cvt.rn.f16.f32"));
    }

    #[test]
    fn test_synthesize_layernorm_ptx() {
        let ptx = synthesize_fused_layernorm_ptx(768, true, 1e-5, DType::F32);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        assert!(ptx_str.contains(".entry fused_layernorm_768"));
        assert!(ptx_str.contains("shfl.sync.down.b32"));
        assert!(ptx_str.contains("bar.sync"));
        // Affine: gamma * normalized + beta
        assert!(ptx_str.contains("param_gamma"));
        assert!(ptx_str.contains("param_beta"));
        // Must write output to global memory
        assert!(ptx_str.contains("st.global"));
    }

    #[test]
    fn test_synthesize_rmsnorm_ptx() {
        let ptx = synthesize_fused_rmsnorm_ptx(768, true, 1e-5, DType::Fp16);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        assert!(ptx_str.contains(".entry fused_rmsnorm_768"));
        assert!(ptx_str.contains("shfl.sync.down.b32"));
        // Output dtype conversion
        assert!(ptx_str.contains("cvt.rn.f16.f32"));
        // Must write output to global memory
        assert!(ptx_str.contains("st.global"));
    }

    #[test]
    fn test_softmax_f32_output_no_cvt() {
        let ptx = synthesize_fused_softmax_ptx(256, DType::F32);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        // f32 output — no dtype conversion needed
        assert!(!ptx_str.contains("cvt.rn.f16.f32"));
        assert!(ptx_str.contains("st.global.f32"));
    }

    // ── Welford merge simulation tests ────────────────────────────────

    /// Simulate Welford warp-level butterfly reduction with proper weighted merge.
    fn welford_tree_reduce(partials: &[(f32, f32, u32)]) -> (f32, f32, u32) {
        let mut vals: Vec<(f32, f32, u32)> = partials.to_vec();
        for offset in [16usize, 8, 4, 2, 1] {
            let prev = vals.clone();
            for i in 0..vals.len() {
                let j = i ^ offset;
                if j < prev.len() && j > i {
                    let (mean_a, m2_a, n_a) = prev[i];
                    let (mean_b, m2_b, n_b) = prev[j];
                    if n_b == 0 { continue; }
                    let delta = mean_b - mean_a;
                    let combined_n = n_a + n_b;
                    let combined_mean = (n_a as f32 * mean_a + n_b as f32 * mean_b)
                        / combined_n as f32;
                    let combined_m2 = m2_a + m2_b
                        + delta * delta * n_a as f32 * n_b as f32 / combined_n as f32;
                    vals[i] = (combined_mean, combined_m2, combined_n);
                }
            }
        }
        vals[0]
    }

    /// Partition data across threads and compute per-thread Welford partials.
    fn compute_partials(data: &[f32], block_size: usize) -> Vec<(f32, f32, u32)> {
        let hidden_dim = data.len();
        let base_n = hidden_dim / block_size;
        let extra = hidden_dim % block_size;
        let mut partials = Vec::new();
        let mut offset = 0;
        for t in 0..block_size {
            let n = if t < extra { base_n + 1 } else { base_n };
            if n == 0 {
                partials.push((0.0, 0.0, 0));
            } else {
                let chunk = &data[offset..offset + n];
                let mean = chunk.iter().sum::<f32>() / n as f32;
                let m2 = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>();
                partials.push((mean, m2, n as u32));
            }
            offset += n;
        }
        partials
    }

    #[test]
    fn test_welford_merge_uneven_elements() {
        // hidden_dim=100, block_size=32: threads get 3 or 4 elements
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let ref_mean = data.iter().sum::<f32>() / data.len() as f32;
        let ref_var = data.iter()
            .map(|x| (x - ref_mean).powi(2))
            .sum::<f32>() / data.len() as f32;

        let partials = compute_partials(&data, 32);
        let (sim_mean, sim_m2, sim_n) = welford_tree_reduce(&partials);
        let sim_var = sim_m2 / sim_n as f32;

        assert!((sim_mean - ref_mean).abs() < 1e-5,
            "mean: got {}, expected {}", sim_mean, ref_mean);
        assert!((sim_var - ref_var).abs() < 1e-4,
            "var: got {}, expected {}", sim_var, ref_var);
    }

    #[test]
    fn test_welford_merge_edge_cases() {
        for hidden_dim in [1, 32, 33, 100, 1023, 1024, 4096, 4097] {
            let data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let ref_mean = data.iter().sum::<f32>() / data.len() as f32;
            let ref_var = data.iter()
                .map(|x| (x - ref_mean).powi(2))
                .sum::<f32>() / data.len() as f32;

            let block_size = 32.min(hidden_dim);
            let partials = compute_partials(&data, block_size);
            let (sim_mean, sim_m2, sim_n) = welford_tree_reduce(&partials);
            let sim_var = sim_m2 / sim_n as f32;

            assert!((sim_mean - ref_mean).abs() < 1e-3,
                "hidden_dim={}: mean got {}, expected {}", hidden_dim, sim_mean, ref_mean);
            assert!((sim_var - ref_var).abs() < 1e-2,
                "hidden_dim={}: var got {}, expected {}", hidden_dim, sim_var, ref_var);
        }
    }

    #[test]
    fn test_welford_broken_merge_is_inaccurate() {
        // Demonstrate the bug: simple averaging gives wrong results for uneven splits
        fn welford_broken_reduce(partials: &[(f32, f32, u32)]) -> (f32, f32, u32) {
            let mut vals: Vec<(f32, f32, u32)> = partials.to_vec();
            for offset in [16usize, 8, 4, 2, 1] {
                let prev = vals.clone();
                for i in 0..vals.len() {
                    let j = i ^ offset;
                    if j < prev.len() && j > i {
                        let (mean_a, m2_a, n_a) = prev[i];
                        let (mean_b, m2_b, _n_b) = prev[j];
                        // BROKEN: simple average, ignores element counts
                        let combined_mean = (mean_a + mean_b) * 0.5;
                        let combined_m2 = m2_a + m2_b; // missing delta^2 term
                        vals[i] = (combined_mean, combined_m2, n_a);
                    }
                }
            }
            vals[0]
        }

        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let ref_mean = data.iter().sum::<f32>() / data.len() as f32;
        let ref_var = data.iter()
            .map(|x| (x - ref_mean).powi(2))
            .sum::<f32>() / data.len() as f32;

        let partials = compute_partials(&data, 32);
        let (broken_mean, broken_m2, _broken_n) = welford_broken_reduce(&partials);
        let broken_var = broken_m2 / 100.0; // use true N for denominator

        // Broken merge should give noticeably different results
        let mean_err = (broken_mean - ref_mean).abs();
        let var_err = (broken_var - ref_var).abs();
        // At least one of mean or variance should be significantly off
        assert!(mean_err > 0.01 || var_err > 0.01,
            "Expected broken merge to be inaccurate, got mean_err={}, var_err={}", mean_err, var_err);
    }
}
