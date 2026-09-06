//! The train block's lowering, one phase per submodule.
//!
//! `stmt.rs::compile_train_block_inner` is the driver: it still owns the
//! config extraction, the parameter / optimizer-state / accumulator lists
//! and the epoch + batch loops, and the bindings that flow between the
//! phases. Each submodule here is one phase peeled off that function
//! (roadmap A1), in the order the driver runs them:
//!
//!   - [`contract`] — section 2: the resolved optimizer / scheduler /
//!     callbacks contract, the `data:` section, the Muon perf-flag
//!     refusals and the FASE plan, returned as a [`contract::TrainContract`].
//!   - [`identity`] — the checkpoint-identity emission at setup: the
//!     resolved train/optimizer/scheduler record (item 4) and the
//!     full-state resume load (Milestone B).
//!   - [`epoch_close`] — the batch-loop seal, the `on_epoch` callbacks,
//!     the epoch increment and the jump back to the epoch header.
//!   - [`param_lists`] — the per-parameter runtime lists built at setup:
//!     the Muon/AdamW route flags, the weight-decay exemption flags and
//!     the gradient-accumulation buffers.
//!   - [`teardown`] — every emission after the epoch loop's exit block: free
//!     the lists, sweep the trailing CSLA window, restore streamed
//!     weights, print the CUDA-graphs banner.
//!
//! Every peel is a byte-for-byte move of the emission under the
//! train-block CLIF snapshots (`tests/train_clif_snapshots.rs`): the
//! instruction stream a fixture lowers to must not change. The CSLA
//! window helpers live beside this module in `stmt_csla.rs`; the FASE
//! optimizer-step emitters in `stmt_fase.rs`.

pub(crate) mod contract;
pub(crate) mod epoch_close;
pub(crate) mod identity;
pub(crate) mod param_lists;
pub(crate) mod teardown;
