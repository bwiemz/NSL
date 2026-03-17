//! M34: Context parallelism FFI.

#[no_mangle]
pub extern "C" fn nsl_cp_init(
    _ring_size: i64,
    _local_seq_len: i64,
    _num_heads: i64,
    _num_kv_heads: i64,
    _head_dim: i64,
    _dtype: i64,
) -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn nsl_sequence_partition(
    _input_ptr: i64,
    _batch: i64,
    _seq_len: i64,
    _hidden_dim: i64,
    _ring_size: i64,
    _rank: i64,
    _output_ptr: i64,
) -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn nsl_ring_attention(
    _ctx_handle: i64,
    _scale_bits: i64,
    _causal: i64,
    _block_table_ptr: i64,
    _k_pool_ptr: i64,
    _v_pool_ptr: i64,
    _block_size: i64,
    _output_ptr: i64,
    _ptx_ptr: i64,
    _name_ptr: i64,
    _block_q: i64,
    _block_kv: i64,
    _shared_mem_bytes: i64,
) -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn nsl_ring_send_recv(
    _send_buf_ptr: i64,
    _recv_buf_ptr: i64,
    _count: i64,
    _dtype: i64,
    _stream: i64,
) -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn nsl_sequence_gather(
    _local_output_ptr: i64,
    _full_output_ptr: i64,
    _batch: i64,
    _local_seq_len: i64,
    _hidden_dim: i64,
    _ring_size: i64,
    _dtype: i64,
    _stream: i64,
) -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn nsl_cp_destroy(_ctx_handle: i64) -> i64 {
    0
}
