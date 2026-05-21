//! HIR ID types. Each fresh-allocated via `Id::fresh()` returning a unique value.
//!
//! Counters are **per-thread** (`thread_local!` Cell) so snapshot tests are
//! deterministic regardless of test discovery order or parallel-test sharding.
//! `Id::reset()` resets the current thread's counter to its initial value (1);
//! `KirToHirPass::lower` calls the resets at entry so each lowering invocation
//! produces stable IDs (M57.1 wire-array mini §6 snapshot stability).
//!
//! Per-thread (rather than process-global atomic) was chosen because:
//!   - cargo runs integration tests in parallel by default; a shared atomic
//!     counter would still interleave across threads even with a reset at
//!     `lower()` entry, breaking snapshots in the suite-wide run.
//!   - Within a single `lower()` call the lowerer is sequential, so per-thread
//!     state is sufficient — no cross-thread sharing of generated HIR IDs
//!     happens during lowering.

use std::cell::Cell;

macro_rules! define_id {
    ($name:ident, $counter:ident) => {
        thread_local! {
            // Per-thread counter. The initial value (1) is preserved on
            // reset; `0` is reserved for sentinel/default IDs.
            static $counter: Cell<u64> = const { Cell::new(1) };
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u64);

        impl $name {
            pub fn fresh() -> Self {
                $counter.with(|c| {
                    let v = c.get();
                    c.set(v + 1);
                    Self(v)
                })
            }

            /// Reset the current thread's counter to its initial value (1).
            /// Used by `KirToHirPass::lower` so that snapshot tests don't
            /// suffer inter-test counter bleed — see M57.1 wire-array mini §6.
            pub fn reset() {
                $counter.with(|c| c.set(1));
            }
        }
    };
}

define_id!(WireId, WIRE_ID_COUNTER);
define_id!(RegisterId, REGISTER_ID_COUNTER);
define_id!(GenvarId, GENVAR_ID_COUNTER);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClockDomainId(pub u64);
impl ClockDomainId {
    pub const DEFAULT: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResetSignalId(pub u64);
impl ResetSignalId {
    pub const DEFAULT: Self = Self(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_ids_are_unique() {
        let a = WireId::fresh();
        let b = WireId::fresh();
        assert_ne!(a, b);
    }

    #[test]
    fn fresh_ids_are_strictly_increasing() {
        let a = WireId::fresh();
        let b = WireId::fresh();
        assert!(b.0 > a.0);
    }

    #[test]
    fn default_clock_domain_is_zero() {
        assert_eq!(ClockDomainId::DEFAULT.0, 0);
    }

    #[test]
    fn default_reset_signal_is_zero() {
        assert_eq!(ResetSignalId::DEFAULT.0, 0);
    }
}
