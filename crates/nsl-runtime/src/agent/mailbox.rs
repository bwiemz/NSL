//! M56 PortMailbox — one mailbox per declared port. Holds at most one
//! `PortMessage` per logical time step.
//!
//! Spec §3.2, refinement #3 (struct-typed ports require enum payload).

use crate::tensor::NslTensor;

/// A port's wire payload. Tensor-typed ports carry an `NslTensor` by value;
/// struct-typed ports carry a heap-allocated struct payload (Q2: structs
/// move whole-struct-granularity through ports).
pub enum PortMessage {
    Tensor(NslTensor),
    Struct(Box<StructPayload>),
}

impl std::fmt::Debug for PortMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortMessage::Tensor(_) => f.debug_tuple("Tensor").finish(),
            PortMessage::Struct(s) => f.debug_tuple("Struct").field(s).finish(),
        }
    }
}

/// Opaque heap-allocated struct payload. The codegen emits payloads with
/// matching layout; the scheduler treats `bytes` opaquely.
///
/// `_descriptor_id` is a forward-compatibility hook for v2 — when codegen
/// emits structs with multiple distinct schemas, this field will identify
/// which schema this payload conforms to. v1 leaves it 0; tasks 17–19
/// will populate it.
#[derive(Debug)]
pub struct StructPayload {
    bytes: Vec<u8>,
    _descriptor_id: u64,
}

impl StructPayload {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, _descriptor_id: 0 }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// M56 v1 port mailbox. Holds at most one `PortMessage` per logical time
/// step. `stamped_time` records when a message was written; `expected_read_time`
/// is `stamped_time + 1`, encoding the "outputs at T visible at T+1" invariant
/// from spec §2.1.
///
/// Note: `#[repr(C)]` is intentionally omitted — `Option<PortMessage>` is not
/// `repr(C)` safe due to the `Box<StructPayload>` variant. The FFI surface
/// (Task 16) will operate on the mailbox by opaque pointer; C-compatible layout
/// is not required for v1.
#[derive(Debug)]
pub struct PortMailbox {
    slot: Option<PortMessage>,
    stamped_time: u64,
    expected_read_time: u64,
}

impl PortMailbox {
    pub fn new() -> Self {
        Self {
            slot: None,
            stamped_time: 0,
            expected_read_time: 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slot.is_none()
    }

    pub fn stamped_time(&self) -> u64 {
        self.stamped_time
    }

    pub fn expected_read_time(&self) -> u64 {
        self.expected_read_time
    }

    /// Write a payload, stamping it with the producer's current logical time.
    /// Sets `expected_read_time = time + 1`.
    pub fn write(&mut self, msg: PortMessage, time: u64) {
        self.slot = Some(msg);
        self.stamped_time = time;
        self.expected_read_time = time + 1;
    }

    /// Read and consume the payload, if present. Returns `None` if no
    /// message is currently in the mailbox.
    pub fn read(&mut self) -> Option<PortMessage> {
        self.slot.take()
    }
}

impl Default for PortMailbox {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::ptr::null_mut;

    use super::*;

    /// Construct a minimal zero-element NslTensor with no heap allocation.
    /// All pointer fields are null; this is safe only in test code that never
    /// dereferences them.
    fn make_null_tensor() -> NslTensor {
        NslTensor::new(
            null_mut(), // data
            null_mut(), // shape
            null_mut(), // strides
            0,          // ndim
            0,          // len
            0,          // device (CPU)
            0,          // dtype (f64)
            0,          // owns_data = 0 (borrowed — no free on drop)
            0,          // data_owner
        )
    }

    #[test]
    fn mailbox_round_trips_tensor_message() {
        let mut mb = PortMailbox::new();
        let tensor = make_null_tensor();
        assert!(mb.is_empty());
        mb.write(PortMessage::Tensor(tensor), /* time = */ 5);
        assert_eq!(mb.stamped_time(), 5);
        let msg = mb.read().expect("message missing");
        assert!(matches!(msg, PortMessage::Tensor(_)));
        assert!(mb.is_empty());
    }

    #[test]
    fn mailbox_carries_struct_payload() {
        let mut mb = PortMailbox::new();
        let payload = StructPayload::new(vec![1, 2, 3]);
        mb.write(PortMessage::Struct(Box::new(payload)), 3);
        let msg = mb.read().unwrap();
        assert!(matches!(msg, PortMessage::Struct(_)));
    }

    #[test]
    fn mailbox_visibility_at_t_plus_one() {
        // Spec §2.1: outputs written at T are visible at T+1.
        // The mailbox tracks expected_read_time = stamped_time + 1.
        let mut mb = PortMailbox::new();
        let tensor = make_null_tensor();
        mb.write(PortMessage::Tensor(tensor), 7);
        // expected_read_time should be 8 (= stamped_time + 1).
        assert_eq!(mb.expected_read_time(), 8);
    }

    #[test]
    fn mailbox_default_construction() {
        let mb = PortMailbox::default();
        assert!(mb.is_empty());
        assert_eq!(mb.stamped_time(), 0);
    }
}
