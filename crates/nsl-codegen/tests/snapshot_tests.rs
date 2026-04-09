//! Snapshot tests for generated PTX and KIR output.
//!
//! These tests lock down the exact bytes the compiler generates for GPU kernels.
//! If a codegen change alters the PTX output, the test fails with a diff.
//!
//! To review and accept new snapshots:
//!   cargo insta review -p nsl-codegen
//!
//! Snapshots are stored in tests/snapshots/*.snap

// ---------------------------------------------------------------------------
// FlashAttention PTX snapshots
// ---------------------------------------------------------------------------

#[test]
fn snapshot_flash_attention_basic() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx[..ptx.len().saturating_sub(1)]); // strip null
    insta::assert_snapshot!("flash_attention_basic_causal", ptx_str);
}

#[test]
fn snapshot_flash_attention_paged() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: true,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx[..ptx.len().saturating_sub(1)]);
    insta::assert_snapshot!("flash_attention_paged_causal", ptx_str);
}

#[test]
fn snapshot_flash_attention_rope_gqa() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: true,
        rope_q: true,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 4,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx[..ptx.len().saturating_sub(1)]);
    insta::assert_snapshot!("flash_attention_rope_gqa4", ptx_str);
}

#[test]
fn snapshot_flash_attention_tree_mask() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: true,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: true,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx[..ptx.len().saturating_sub(1)]);
    insta::assert_snapshot!("flash_attention_tree_mask", ptx_str);
}

// ---------------------------------------------------------------------------
// FlashAttention logsumexp PTX tests (Phase 1 backward support)
// ---------------------------------------------------------------------------

#[test]
fn flash_attention_ptx_has_logsumexp_param() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx);

    // Verify param_logsumexp is declared in the kernel entry
    assert!(
        ptx_str.contains(".param .u64 param_logsumexp"),
        "PTX must declare param_logsumexp parameter"
    );
}

#[test]
fn flash_attention_ptx_loads_logsumexp_base() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx);

    // Verify logsumexp_base is loaded from param
    assert!(
        ptx_str.contains("ld.param.u64 %logsumexp_base, [param_logsumexp]"),
        "PTX must load logsumexp_base from param_logsumexp"
    );
}

#[test]
fn flash_attention_ptx_computes_logsumexp() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx);

    // Verify log2 approximation for computing ln(row_sum) via log2 * ln(2)
    assert!(
        ptx_str.contains("lg2.approx.f32 %log_sum, %row_sum"),
        "PTX must compute log2(row_sum)"
    );

    // Verify multiplication by ln(2) constant (IEEE 754 hex: 0x3F317218)
    assert!(
        ptx_str.contains("0F3F317218"),
        "PTX must multiply by ln(2) constant 0F3F317218"
    );

    // Verify L = row_max + log_sum
    assert!(
        ptx_str.contains("add.f32 %lse, %row_max, %log_sum"),
        "PTX must compute lse = row_max + log_sum"
    );

    // Verify null-guard: setp.ne for logsumexp_base
    assert!(
        ptx_str.contains("setp.ne.u64 %p_has_lse, %logsumexp_base, 0"),
        "PTX must null-check logsumexp_base before store"
    );

    // Verify global store of logsumexp value
    assert!(
        ptx_str.contains("st.global.f32 [%lse_addr], %lse"),
        "PTX must store logsumexp to global memory"
    );
}

#[test]
fn flash_attention_ptx_logsumexp_has_bounds_check() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx);

    // Verify bounds check: only store if (q_start + tid_x) < seq_len
    assert!(
        ptx_str.contains("SKIP_LSE_STORE"),
        "PTX must have SKIP_LSE_STORE label for bounds/null guard"
    );
}

#[test]
fn flash_attention_ptx_logsumexp_register_declarations() {
    let config = nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
    };
    let ptx = nsl_codegen::flash_attention::synthesize_flash_attention_ptx(&config);
    let ptx_str = String::from_utf8_lossy(&ptx);

    // Verify register declarations for logsumexp computation
    assert!(
        ptx_str.contains(".reg .u64 %logsumexp_base"),
        "PTX must declare %logsumexp_base register"
    );
    assert!(
        ptx_str.contains(".reg .f32 %log_sum, %lse"),
        "PTX must declare %log_sum and %lse registers"
    );
    assert!(
        ptx_str.contains(".reg .pred %p_has_lse"),
        "PTX must declare %p_has_lse predicate"
    );
}

// ---------------------------------------------------------------------------
// KIR -> PTX backend snapshots
// ---------------------------------------------------------------------------

#[test]
fn snapshot_kir_tensor_add_ptx() {
    use nsl_codegen::backend_ptx::lower_kir_to_ptx;
    use nsl_codegen::kernel_ir::*;

    // Build a tensor_add kernel in KIR
    let mut b = KirBuilder::new("nsl_tensor_add");
    let a_ptr = b.add_param(
        "a",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    let b_ptr = b.add_param(
        "b",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    let out_ptr = b.add_param(
        "out",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    let len = b.add_param("len", KirType::U32, AddressSpace::Local);

    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();

    b.set_block(entry);
    let tid = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(tid, 0));
    let cond = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(cond, tid, len, CmpOp::Lt));
    b.terminate(KirTerminator::CondBranch(cond, body, exit));

    b.set_block(body);
    let a_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(a_addr, a_ptr, tid));
    let a_val = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(a_val, a_addr, AddressSpace::Global));
    let b_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(b_addr, b_ptr, tid));
    let b_val = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(b_val, b_addr, AddressSpace::Global));
    let sum = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Add(sum, a_val, b_val));
    let out_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(out_addr, out_ptr, tid));
    b.emit(KirOp::Store(out_addr, sum, AddressSpace::Global));
    b.terminate(KirTerminator::Branch(exit));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);

    let ir = b.finalize();
    let ptx = lower_kir_to_ptx(&ir);
    let ptx_str = String::from_utf8_lossy(&ptx[..ptx.len().saturating_sub(1)]);
    insta::assert_snapshot!("kir_tensor_add_ptx", ptx_str);
}

#[test]
fn snapshot_kir_with_barrier_ptx() {
    use nsl_codegen::backend_ptx::lower_kir_to_ptx;
    use nsl_codegen::kernel_ir::*;

    let mut b = KirBuilder::new("barrier_kernel");
    b.add_param(
        "data",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    b.set_shared_mem(1024);

    let entry = b.new_block();
    b.set_block(entry);
    b.emit(KirOp::Barrier);
    b.terminate(KirTerminator::Return);

    let ir = b.finalize();
    let ptx = lower_kir_to_ptx(&ir);
    let ptx_str = String::from_utf8_lossy(&ptx[..ptx.len().saturating_sub(1)]);
    insta::assert_snapshot!("kir_barrier_kernel_ptx", ptx_str);
}

// ---------------------------------------------------------------------------
// Fusion graph structure snapshots
// ---------------------------------------------------------------------------

#[test]
fn snapshot_fusion_graph_transformer_block() {
    use nsl_codegen::fusion_graph::*;

    let mut graph = FusionGraph::new();

    // Simulate a transformer block: input -> matmul -> bias_add -> gelu -> layernorm -> output
    let input = graph.add_named_node("input".into(), FusionOp::Input, vec![]);
    let matmul = graph.add_named_node("matmul".into(), FusionOp::Matmul, vec![input]);
    let bias = graph.add_named_node(
        "bias_add".into(),
        FusionOp::Elementwise("add".into()),
        vec![matmul],
    );
    let gelu = graph.add_named_node(
        "gelu".into(),
        FusionOp::Elementwise("gelu".into()),
        vec![bias],
    );
    let ln = graph.add_named_node(
        "layernorm".into(),
        FusionOp::Reduction("layernorm".into()),
        vec![gelu],
    );

    graph.mark_graph_output(ln);
    graph.build_consumers();

    // Snapshot the graph structure
    let mut desc = String::new();
    for node in &graph.nodes {
        let name = node.name.as_deref().unwrap_or("?");
        let op = format!("{:?}", node.op);
        let inputs: Vec<_> = node
            .inputs
            .iter()
            .map(|i| graph.nodes[*i as usize].name.as_deref().unwrap_or("?"))
            .collect();
        let consumers: Vec<_> = node
            .consumers
            .iter()
            .map(|c| graph.nodes[*c as usize].name.as_deref().unwrap_or("?"))
            .collect();
        desc.push_str(&format!(
            "{name}: op={op}, inputs=[{}], consumers=[{}], output={}\n",
            inputs.join(", "),
            consumers.join(", "),
            node.is_graph_output,
        ));
    }
    insta::assert_snapshot!("fusion_graph_transformer_block", desc);
}
