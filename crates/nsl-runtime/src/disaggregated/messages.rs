//! M41: Inter-process message protocol for disaggregated inference.

/// Worker role — determined by NSL_ROLE environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerRole {
    Router = 0,
    Prefill = 1,
    Decode = 2,
}

impl WorkerRole {
    pub fn from_env() -> Self {
        match std::env::var("NSL_ROLE").as_deref() {
            Ok("router") => WorkerRole::Router,
            Ok("prefill") => WorkerRole::Prefill,
            Ok("decode") => WorkerRole::Decode,
            _ => WorkerRole::Router, // default: monolithic (router-only)
        }
    }

    pub fn local_rank() -> i32 {
        std::env::var("NSL_LOCAL_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }
}

/// Messages exchanged between router and workers via shared memory.
///
/// Layout: `[u32 tag][payload bytes]` — tag identifies the variant,
/// payload is the struct fields in order.
#[derive(Clone, Debug)]
pub enum RouterMessage {
    /// Router → Prefill: start processing this request's prompt.
    StartPrefill {
        request_id: u64,
        token_ids_ptr: i64,    // pointer to token array (in shared memory)
        num_tokens: u32,
        target_decode_rank: i32,
    },

    /// Prefill → Router: prefill complete, KV transferred to decode worker.
    PrefillComplete {
        request_id: u64,
        num_kv_blocks: u32,
    },

    /// Router → Decode: KV pages incoming, begin autoregressive generation.
    StartDecode {
        request_id: u64,
        prompt_len: u32,
        max_tokens: u32,
        temperature_bits: u64,  // f64 transmuted to u64 for C repr
        top_p_bits: u64,
    },

    /// Decode → Router: a token was generated.
    TokenGenerated {
        request_id: u64,
        token_id: i64,
        is_eos: u8,  // 0 = no, 1 = yes
    },

    /// Decode → Router: sequence complete, all resources freed.
    DecodeComplete {
        request_id: u64,
        total_tokens: u32,
    },
}

/// Request lifecycle state tracked by the router.
#[derive(Clone, Debug, PartialEq)]
pub enum DisaggRequestState {
    Queued,
    Prefilling { prefill_rank: i32 },
    Transferring { from_rank: i32, to_rank: i32 },
    Decoding { decode_rank: i32, tokens_generated: u32 },
    Complete { total_tokens: u32 },
}

/// Per-worker metadata tracked by the router.
pub struct WorkerHandle {
    pub rank: i32,
    pub role: WorkerRole,
    pub active_requests: u32,
    pub free_kv_blocks: u32,
}

impl WorkerHandle {
    pub fn new(rank: i32, role: WorkerRole, initial_kv_blocks: u32) -> Self {
        WorkerHandle {
            rank,
            role,
            active_requests: 0,
            free_kv_blocks: initial_kv_blocks,
        }
    }
}

// Serialization helpers for shared-memory message passing.

/// Serialize a RouterMessage into a byte buffer.
/// Format: [u32 tag][fields in repr(C) order]
pub fn serialize_message(msg: &RouterMessage) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    match msg {
        RouterMessage::StartPrefill { request_id, token_ids_ptr, num_tokens, target_decode_rank } => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&token_ids_ptr.to_le_bytes());
            buf.extend_from_slice(&num_tokens.to_le_bytes());
            buf.extend_from_slice(&target_decode_rank.to_le_bytes());
        }
        RouterMessage::PrefillComplete { request_id, num_kv_blocks } => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&num_kv_blocks.to_le_bytes());
        }
        RouterMessage::StartDecode { request_id, prompt_len, max_tokens, temperature_bits, top_p_bits } => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&prompt_len.to_le_bytes());
            buf.extend_from_slice(&max_tokens.to_le_bytes());
            buf.extend_from_slice(&temperature_bits.to_le_bytes());
            buf.extend_from_slice(&top_p_bits.to_le_bytes());
        }
        RouterMessage::TokenGenerated { request_id, token_id, is_eos } => {
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&token_id.to_le_bytes());
            buf.extend_from_slice(&[*is_eos, 0, 0, 0]); // pad to 4 bytes
        }
        RouterMessage::DecodeComplete { request_id, total_tokens } => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&total_tokens.to_le_bytes());
        }
    }
    buf
}

/// Deserialize a RouterMessage from a byte slice.
/// Returns None if the buffer is too short or tag is unknown.
pub fn deserialize_message(buf: &[u8]) -> Option<RouterMessage> {
    if buf.len() < 4 {
        return None;
    }
    let tag = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let payload = &buf[4..];

    match tag {
        0 if payload.len() >= 24 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let token_ids_ptr = i64::from_le_bytes(payload[8..16].try_into().ok()?);
            let num_tokens = u32::from_le_bytes(payload[16..20].try_into().ok()?);
            let target_decode_rank = i32::from_le_bytes(payload[20..24].try_into().ok()?);
            Some(RouterMessage::StartPrefill { request_id, token_ids_ptr, num_tokens, target_decode_rank })
        }
        1 if payload.len() >= 12 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let num_kv_blocks = u32::from_le_bytes(payload[8..12].try_into().ok()?);
            Some(RouterMessage::PrefillComplete { request_id, num_kv_blocks })
        }
        2 if payload.len() >= 32 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let prompt_len = u32::from_le_bytes(payload[8..12].try_into().ok()?);
            let max_tokens = u32::from_le_bytes(payload[12..16].try_into().ok()?);
            let temperature_bits = u64::from_le_bytes(payload[16..24].try_into().ok()?);
            let top_p_bits = u64::from_le_bytes(payload[24..32].try_into().ok()?);
            Some(RouterMessage::StartDecode { request_id, prompt_len, max_tokens, temperature_bits, top_p_bits })
        }
        3 if payload.len() >= 17 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let token_id = i64::from_le_bytes(payload[8..16].try_into().ok()?);
            let is_eos = payload[16];
            Some(RouterMessage::TokenGenerated { request_id, token_id, is_eos })
        }
        4 if payload.len() >= 12 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let total_tokens = u32::from_le_bytes(payload[8..12].try_into().ok()?);
            Some(RouterMessage::DecodeComplete { request_id, total_tokens })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_role_defaults() {
        // When NSL_ROLE is not set, default is Router
        std::env::remove_var("NSL_ROLE");
        assert_eq!(WorkerRole::from_env(), WorkerRole::Router);
    }

    #[test]
    fn message_roundtrip_start_prefill() {
        let msg = RouterMessage::StartPrefill {
            request_id: 42,
            token_ids_ptr: 0x1234_5678,
            num_tokens: 128,
            target_decode_rank: 3,
        };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::StartPrefill { request_id, token_ids_ptr, num_tokens, target_decode_rank } => {
                assert_eq!(request_id, 42);
                assert_eq!(token_ids_ptr, 0x1234_5678);
                assert_eq!(num_tokens, 128);
                assert_eq!(target_decode_rank, 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_prefill_complete() {
        let msg = RouterMessage::PrefillComplete { request_id: 7, num_kv_blocks: 16 };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::PrefillComplete { request_id, num_kv_blocks } => {
                assert_eq!(request_id, 7);
                assert_eq!(num_kv_blocks, 16);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_start_decode() {
        let temp: f64 = 0.7;
        let top_p: f64 = 0.9;
        let msg = RouterMessage::StartDecode {
            request_id: 99,
            prompt_len: 512,
            max_tokens: 256,
            temperature_bits: temp.to_bits(),
            top_p_bits: top_p.to_bits(),
        };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::StartDecode { request_id, prompt_len, max_tokens, temperature_bits, top_p_bits } => {
                assert_eq!(request_id, 99);
                assert_eq!(prompt_len, 512);
                assert_eq!(max_tokens, 256);
                assert_eq!(f64::from_bits(temperature_bits), 0.7);
                assert_eq!(f64::from_bits(top_p_bits), 0.9);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_token_generated() {
        let msg = RouterMessage::TokenGenerated { request_id: 5, token_id: 42, is_eos: 1 };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::TokenGenerated { request_id, token_id, is_eos } => {
                assert_eq!(request_id, 5);
                assert_eq!(token_id, 42);
                assert_eq!(is_eos, 1);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_decode_complete() {
        let msg = RouterMessage::DecodeComplete { request_id: 10, total_tokens: 200 };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::DecodeComplete { request_id, total_tokens } => {
                assert_eq!(request_id, 10);
                assert_eq!(total_tokens, 200);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn deserialize_short_buffer_returns_none() {
        assert!(deserialize_message(&[]).is_none());
        assert!(deserialize_message(&[0, 0, 0, 0]).is_none()); // tag 0 but no payload
    }

    #[test]
    fn deserialize_unknown_tag_returns_none() {
        let mut buf = 99u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 32]);
        assert!(deserialize_message(&buf).is_none());
    }

    #[test]
    fn worker_handle_construction() {
        let w = WorkerHandle::new(2, WorkerRole::Decode, 1024);
        assert_eq!(w.rank, 2);
        assert_eq!(w.role, WorkerRole::Decode);
        assert_eq!(w.active_requests, 0);
        assert_eq!(w.free_kv_blocks, 1024);
    }

    #[test]
    fn disagg_request_state_transitions() {
        let mut state = DisaggRequestState::Queued;
        assert_eq!(state, DisaggRequestState::Queued);

        state = DisaggRequestState::Prefilling { prefill_rank: 0 };
        assert!(matches!(state, DisaggRequestState::Prefilling { .. }));

        state = DisaggRequestState::Transferring { from_rank: 0, to_rank: 2 };
        assert!(matches!(state, DisaggRequestState::Transferring { .. }));

        state = DisaggRequestState::Decoding { decode_rank: 2, tokens_generated: 0 };
        assert!(matches!(state, DisaggRequestState::Decoding { .. }));

        state = DisaggRequestState::Complete { total_tokens: 50 };
        assert_eq!(state, DisaggRequestState::Complete { total_tokens: 50 });
    }
}
