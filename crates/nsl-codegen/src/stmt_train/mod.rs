//! The train block's lowering, one phase per submodule.
//!
//! `stmt.rs::compile_train_block_inner` is the driver: it still owns the
//! config extraction, the parameter / optimizer-state / accumulator lists
//! and the epoch + batch loops, and the bindings that flow between the
//! phases. Each submodule here is one phase peeled off that function
//! (roadmap A1), in the order the driver runs them:
//!
//!   - [`model_params`] — section 3: the model's layout, its tensor
//!     parameters as a runtime list, the CPDT dtype-code lists and Muon's
//!     mode table, returned as a [`model_params::ModelParams`].
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

pub(crate) mod model_params;
pub(crate) mod param_lists;
pub(crate) mod teardown;
