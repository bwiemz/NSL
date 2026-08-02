//! Roadmap item 2, step 4: **the pass bus, typed and observable.**
//!
//! # Why this exists
//!
//! Steps 1-3 made pass execution visible: [`crate::pass_trace`] records which
//! passes were reached ([`crate::pass_trace::record`]), which driver phase
//! invoked them ([`crate::pass_trace::enter_phase`]), and what each one did
//! ([`crate::pass_trace::record_disposition`]). What none of that answers is
//! how passes reach *each other*.
//!
//! They do it through `Compiler`. A pass stashes its output in a field —
//! `last_wrga_plan`, `last_csha_bridge`, `wggo_overrides` — and some later
//! stage reads it back. That is a real dataflow graph: **twelve channels, one
//! producing pass each**, and it was written down nowhere. Before this module
//! `Compiler` had **75 fields**, 74 of them `pub`, so nothing distinguished an
//! inter-pass channel from a scratch counter or a `Value`-keyed emission
//! cache.
//!
//! Those counts are checked, not asserted from memory:
//! `every_file_that_reads_a_channel_is_a_declared_consumer` in
//! `crates/nsl-codegen/tests/pass_bus_drift.rs` fails if a file reads a
//! channel its descriptor does not list, and
//! `every_pass_bus_field_is_a_declared_channel` fails the other way.
//!
//! Twelve is not yet all of them — see [`CHANNELS`] for what is deliberately
//! excluded and why. The first revision of this module claimed eight and said
//! the remainder were not channels; a review showed four of them plainly were,
//! including the one the ordering argument below depends on.
//!
//! # The ordering argument
//!
//! This is the step that makes a pass *manager* possible, and the reason is a
//! finding from step 1. [`crate::pass_registry::PipelineStage`] cannot order
//! passes: under `--wggo full` the observed order is `WGGO(OnWengert) ->
//! FASE(PreExtraction)`, an inversion, because WGGO is invoked from a driver
//! that runs before any train block exists. Stage says where a pass *acts*, not
//! when it is *called*.
//!
//! Data dependency does not have that problem. If CSHA publishes
//! [`Channel::CshaBackwardClaims`] and the source-AD reverse walk consumes it,
//! then CSHA precedes the reverse walk in any correct schedule — not by
//! declaration but by construction. The channels below are the edges a
//! scheduler would order by, which is why they are worth naming before anything
//! is scheduled.
//!
//! The WGGO-to-FASE edge in that very example is
//! [`Channel::WggoOverrides`]: WGGO fills it, FASE recipe selection reads it,
//! and that is why FASE must follow WGGO regardless of what their declared
//! stages imply. An ordering argument whose own worked example was not in the
//! channel list would have been an argument about nothing.
//!
//! # Why the fields are private
//!
//! Every field of [`PassBus`] is private to THIS module — not to
//! `compiler/mod.rs`, which defines `Compiler` and could therefore reach them.
//! So the accessors are the only way in, and that is enforced by the compiler
//! rather than by whoever adds the next call site.
//!
//! That distinction is not theoretical. Step 2 instrumented driver phases at
//! the call sites and covered 10 of 26; the fix was to move the instrumentation
//! to the callee, where being called is the only way to arrive. Privacy is the
//! same move applied to state: a new consumer of `wrga_plan` cannot be written
//! without going through [`PassBus::wrga_plan`], so it cannot be written
//! without going through the counter.
//!
//! Precisely: privacy guarantees every access goes through an ACCESSOR. It does
//! not guarantee every access is counted, because four accessors deliberately
//! do not count — [`PassBus::restore_csha_backward_claims`],
//! [`PassBus::clear_csha_backward_claims`], [`PassBus::clear_wggo_overrides`]
//! and [`PassBus::clear_cfie_serve_gen`] are lifecycle operations on a value
//! that was already published once, and counting them would make a channel
//! nobody consumes look busier than it is. Each says so at its definition. They
//! stay `pub` because the integration-test binaries are separate crates; the
//! guarantee that holds is *"no access without an accessor"*, and the accessor
//! is where the decision to count is written down and reviewable.
//!
//! # What the traffic counters find
//!
//! Two defect classes, neither visible from the pass trace alone:
//!
//! * **Dead output** — a pass published a channel that nothing read. This is
//!   the `find_best_variant` shape (an autotuner chooser with zero production
//!   callers, found only by a negative test written afterwards) at channel
//!   granularity, and it is now reported from any single run.
//! * **Silent default** — a consumer read an empty channel *although its
//!   producing pass ran*. The consumer took its `None` branch and carried on,
//!   which is indistinguishable from correct behaviour at every layer above.
//!   Detecting this needs both halves: the bus knows the read was empty, the
//!   pass trace knows the producer ran. Neither alone can say it.
//!
//! An empty read whose producer did NOT run is the ordinary case (the feature
//! is off) and is counted but not flagged.
//!
//! # Scope, and what that costs
//!
//! Counters are per PROCESS, not per compile — the same scope
//! [`crate::pass_trace`] settled on, and for the same reason: `nsl build`
//! compiles a program's imported modules through a different codegen entry
//! point than its own, so resetting at either entry would silently drop the
//! other's traffic. `nsl run` and `nsl build` compile once, so in practice the
//! two coincide.
//!
//! Where they do not, the findings are CONSERVATIVE — they under-report rather
//! than invent. Two compiles in one process share counters, so a channel read
//! by the first and left dead by the second shows `reads_full > 0` overall and
//! no `DeadOutput`. That is a false negative, and the acceptable direction: a
//! finding that fires on a compile the reader is not looking at would be
//! unactionable, and this whole campaign is about reports that can be trusted
//! at face value.
//!
//! # Why it cannot change the build
//!
//! The accessors move exactly the values the public fields moved, and the
//! counters are process-global atomics that nothing in codegen reads. The gate
//! in `crates/nsl-cli/tests/pass_bus_gate.rs` pins this by comparing loss
//! streams with `NSL_PASS_TRACE` on and off.
//!
//! Counters are [`AtomicUsize`], not a `Mutex`. Several accessors sit on
//! per-item paths — [`PassBus::adapter_sites`] is consulted on every member
//! access that might be a synthesized adapter field, and
//! [`PassBus::is_csha_claimed`] is shaped as a per-Wengert-op predicate — so a
//! lock would be a cost paid on every build to serve an env var almost nobody
//! sets. A relaxed atomic increment is not.
//!
//! (`is_csha_claimed` has no callers at all today, which the drift gate pins;
//! the point stands on `adapter_sites`, and the design should not have to
//! change when CSHA A.2.2 finally wires the other one up.)

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::pass_registry::CompilePhase;

/// One inter-pass channel.
///
/// The discriminant indexes [`TRAFFIC`] and [`CHANNELS`], so the three stay
/// aligned by construction; `channel_descriptors_are_in_enum_order` in
/// `crates/nsl-codegen/tests/pass_bus_drift.rs` pins that they do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// CSHA's per-layer bridge result: which layers were claimed and the
    /// `CshaExtras` each FlashAttention launch needs.
    CshaBridge,
    /// The Wengert op indices CSHA claimed, so op lowering can skip emitting a
    /// redundant launch for an op the fused kernel already covers.
    CshaClaimedOps,
    /// The chain-level dispatch map the source-AD reverse walk uses to route
    /// claimed ops through CSHA's fused backward instead of per-op AD rules.
    CshaBackwardClaims,
    /// The WRGA plan: adapter placement, freeze decisions, fusion groups.
    WrgaPlan,
    /// WRGA's prescan plan, produced before any train block so adapter
    /// injection can rewrite model methods that the main plan has not seen.
    AdapterPrescanPlan,
    /// CPKD's distillation facts, rendered as a build report.
    CpkdPlan,
    /// The CPDT plan, read by the precision-adaptive optimizer path.
    CpdtPlan,
    /// The CFIE plan for a serve block.
    CfiePlan,
    /// WGGO's per-layer decisions, as consumed by everything downstream.
    ///
    /// **The edge the module header's motivating example rests on.** FASE
    /// recipe selection, the per-parameter mode table, CSHA's input and WRGA's
    /// placement filter all read it; WGGO fills it. An earlier revision of this
    /// module asserted that WGGO published no channel at all *because* its
    /// output went through the Wengert list — false, and false about the one
    /// edge being used to argue that data dependency can order passes where
    /// `PipelineStage` cannot.
    WggoOverrides,
    /// WGGO's pre-pass plans, produced before any train block.
    WggoPreplans,
    /// The adapter injection sites WRGA resolved from decorators.
    AdapterSites,
    /// CFIE's `generate()` driver parameters, live only inside a CFIE-active
    /// serve body.
    CfieServeGen,
}

impl Channel {
    /// Every variant, in discriminant order.
    ///
    /// [`TRAFFIC`] is sized from `CHANNELS.len()` and every counter helper
    /// indexes it by discriminant, so a variant added WITHOUT a descriptor is
    /// an out-of-bounds panic at first use — and one added with a descriptor
    /// but no field slips past a gate that iterates `CHANNELS` rather than the
    /// enum. This list closes both: `ALL_CHANNELS_HAVE_A_DESCRIPTOR` below is a
    /// compile-time check, so the mistake cannot reach a test run.
    pub const ALL: &'static [Channel] = &[
        Channel::CshaBridge,
        Channel::CshaClaimedOps,
        Channel::CshaBackwardClaims,
        Channel::WrgaPlan,
        Channel::AdapterPrescanPlan,
        Channel::CpkdPlan,
        Channel::CpdtPlan,
        Channel::CfiePlan,
        Channel::WggoOverrides,
        Channel::WggoPreplans,
        Channel::AdapterSites,
        Channel::CfieServeGen,
    ];

    const fn idx(self) -> usize {
        self as usize
    }

    /// This channel's declaration.
    ///
    /// `expect` rather than a bare index so that adding a variant without a
    /// descriptor says what is wrong. The alternative is an out-of-bounds
    /// panic pointing at this line, which describes the symptom and not the
    /// omission.
    pub fn descriptor(self) -> &'static ChannelDescriptor {
        CHANNELS.get(self.idx()).unwrap_or_else(|| {
            panic!(
                "Channel::{self:?} has no ChannelDescriptor — CHANNELS has {} \
                 entries and must have one per variant, in discriminant order",
                CHANNELS.len()
            )
        })
    }
}

/// Whether a traffic pattern is a defect for THIS channel.
///
/// A sum type carrying a reason, not a `bool`, for the reason
/// [`crate::pass_registry::WikiCoverage`] is one: an exemption without a
/// recorded reason outlives the reason for it, and nothing can tell it apart
/// from an oversight. The drift gate requires the reason to be non-empty.
///
/// Both findings need this. Neither `published but never read` nor `applied
/// yet empty` is universally a defect — for several channels the tree does
/// exactly that on purpose — and a rule that fires on correct behaviour trains
/// its reader to stop reading it. Which is the failure this campaign keeps
/// finding, one level up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invariant {
    /// The pattern is a defect for this channel; report it.
    Enforced,
    /// Not a defect here, and why not.
    Exempt(&'static str),
}

impl Invariant {
    fn enforced(self) -> bool {
        matches!(self, Self::Enforced)
    }
}

/// What a channel carries, who fills it, who reads it, and which traffic
/// patterns are defects for it.
///
/// `consumers` lists repo-relative FILES rather than functions. Line numbers
/// rot on the first unrelated edit, and this tree has 17k-line modules where a
/// function name is not much better; the drift gate checks the file exists and
/// mentions the accessor, which is the strongest claim that stays true.
#[derive(Debug, Clone, Copy)]
pub struct ChannelDescriptor {
    pub channel: Channel,
    /// Field name on [`PassBus`], and the stem of its accessors.
    pub name: &'static str,
    /// The pass that fills this channel. MUST be a name in
    /// [`crate::pass_registry::PASSES`] — the drift gate checks it, so a
    /// renamed pass cannot leave a channel pointing at nothing.
    ///
    /// Names the pass whose WORK produces the value, which is not always the
    /// pass whose recorded invocation precedes it: WRGA's adapter prescan runs
    /// from the driver, before any train block, and does not
    /// `pass_trace::record("WRGA")`. See [`producer_phases`], which is
    /// deliberately documented as a property of the PASS and not a claim about
    /// where the publish happens.
    pub producer: &'static str,
    /// The type it carries, as written in the field declaration.
    pub carries: &'static str,
    /// Repo-relative files that read it.
    pub consumers: &'static [&'static str],
    /// What an empty read legitimately means. Recorded because "the channel is
    /// empty" is the single most load-bearing fact about a consumer's `None`
    /// branch, and it was previously implicit in every one of them.
    pub empty_means: &'static str,
    /// Is `published, never read` a defect here?
    pub dead_output: Invariant,
    /// Is `producer applied, yet the channel is empty` a defect here?
    pub applied_implies_published: Invariant,
}

/// Every inter-pass channel, in [`Channel`] discriminant order.
///
/// # What is deliberately NOT here
///
/// `Compiler` has other `Option<T>` fields that look similar and are not
/// channels between passes:
///
/// * `flash_attn_aux`, `flash_attn_bwd_cache`, `fused_ce_fwd_lse`,
///   `fused_ce_fwd_casts`, `fused_ce_bwd_cache`, `fused_kl_ce_*`,
///   `csha_fused_bwd_cache`, `csha_forward_saves` — forward-to-backward
///   EMISSION caches, keyed by `cranelift_codegen::ir::Value` or by layer name.
///   Both ends are op lowering, not a pass; they exist because a backward
///   lowering needs a buffer the forward already allocated.
/// * `wrga_inputs`, `fused_ce_configs`, `pca_user_strategies`, `cpdt_mode` —
///   compile OPTIONS forwarded from the CLI. The producer is the driver, and
///   they are populated before any pass runs, so "was it published" is not a
///   question that can fail.
/// * `fusion_plan_for_profile` — produced by the dev-tools profile pre-pass,
///   which is not a registered pass. It is a *consumer* of [`Channel::WrgaPlan`]
///   and is listed as one.
///
/// Drawing that line is the point of the list. A registry that swept in every
/// `Option` field would report traffic on twenty things and locate nothing.
///
/// # Known channels NOT yet migrated
///
/// Stated because the first revision of this list was wrong about exactly this,
/// and an omission recorded is an omission that can be finished:
///
/// * `active_fused_ce_config` / `active_distill_context` — scoped install and
///   restore around one train or distill block, not produce-then-consume. The
///   accessor pair would have to model a stack, which is a different shape from
///   everything here.
/// * `grad_live_results` — set and cleared around each adjoint-lowering call by
///   the same driver that reads it. A guard, not a channel.
/// * `last_csha_bridge`-adjacent emission state (`csha_forward_saves`,
///   `csha_fa_call_ordinal`) — written by op lowering, read by op lowering.
/// * `retention_arena_data_id` / `retention_offsets` / `adapter_prescan_plan`'s
///   sibling `synth_member_names` — codegen layout tables.
///
/// The first two are the ones most likely to be genuine channels on a closer
/// look. Neither is claimed here to be settled.
pub const CHANNELS: &[ChannelDescriptor] = &[
    ChannelDescriptor {
        channel: Channel::CshaBridge,
        name: "csha_bridge",
        producer: "CSHA",
        carries: "crate::csha_apply::BridgeResult",
        consumers: &[
            "crates/nsl-codegen/src/expr/advanced.rs",
            "crates/nsl-codegen/src/wengert_lower.rs",
            "crates/nsl-codegen/src/stmt.rs",
        ],
        empty_means: "CSHA is off, or ran and claimed no layers — the \
                      FlashAttention call site uses the non-CSHA FFI variant",
        // The producing site reads it back ~25 lines after publishing, to
        // build the backward claims (stmt.rs). So reads_full >= publishes
        // always holds and the finding is structurally unreachable. Recorded
        // rather than left to look enforced: a future cleanup that used the
        // local `bridge_out` instead of re-reading the channel would make
        // every declining `--csha auto` build report a dead output, for a
        // change that altered nothing.
        dead_output: Invariant::Exempt(
            "the publishing site re-reads it two statements later to derive \
             csha_backward_claims, so a full read is guaranteed",
        ),
        applied_implies_published: Invariant::Enforced,
    },
    ChannelDescriptor {
        channel: Channel::CshaClaimedOps,
        name: "csha_claimed_ops",
        producer: "CSHA",
        carries: "std::collections::HashSet<u32>",
        consumers: &["crates/nsl-codegen/src/compiler/mod.rs"],
        empty_means: "no op is claimed, so every op lowers normally — the \
                      empty set IS the correct answer, not a missing one",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Exempt(
            "collect_claimed_ops returns an empty set whenever no boundary \
             chain is claimed, which an applying CSHA does routinely; the \
             empty set is this channel's correct output, not an absence",
        ),
    },
    ChannelDescriptor {
        channel: Channel::CshaBackwardClaims,
        name: "csha_backward_claims",
        producer: "CSHA",
        carries: "crate::source_ad::CshaBackwardClaims",
        consumers: &[
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-codegen/src/wengert_lower.rs",
        ],
        empty_means: "the reverse walk applies per-op AD rules instead of \
                      CSHA's fused backward. Also the deliberate state during \
                      adjoint lowering — see take_csha_backward_claims",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Exempt(
            "published only when chain_marks is non-empty, and the backward \
             SMEM validator can reject every chain while CSHA still applies \
             the forward — so an applying CSHA legitimately leaves it empty",
        ),
    },
    ChannelDescriptor {
        channel: Channel::WrgaPlan,
        name: "wrga_plan",
        producer: "WRGA",
        carries: "crate::wrga::WrgaPlan",
        consumers: &[
            "crates/nsl-codegen/src/compiler/entry_points.rs",
            "crates/nsl-codegen/src/test_helpers.rs",
        ],
        empty_means: "no @train block compiled, or WRGA is disabled; \
                      `compile` returns `None` for the plan it hands back",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Enforced,
    },
    ChannelDescriptor {
        channel: Channel::AdapterPrescanPlan,
        name: "adapter_prescan_plan",
        producer: "WRGA",
        carries: "crate::wrga::WrgaPlan",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "the prescan did not run (no adapter decorators), so \
                      train-block adapter injection has nothing to apply",
        dead_output: Invariant::Exempt(
            "the prescan publishes unconditionally, and the train-block plan \
             supersedes it whenever that block has decorated placements — an \
             unread prescan plan is the designed-for case, not a defect",
        ),
        applied_implies_published: Invariant::Exempt(
            "published by the driver prescan, which never records WRGA as \
             having run, so the antecedent cannot be established for it",
        ),
    },
    ChannelDescriptor {
        channel: Channel::CpkdPlan,
        name: "cpkd_plan",
        producer: "CPKD",
        carries: "crate::cpkd::CpkdPlan",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "no distill block compiled, so no distillation report is \
                      rendered",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Exempt(
            "CPKD reports AdvisoryOnly and never Applied — it rewrites \
             nothing — so the antecedent never holds for this channel",
        ),
    },
    ChannelDescriptor {
        channel: Channel::CpdtPlan,
        name: "cpdt_plan",
        producer: "CPDT",
        carries: "crate::cpdt::CpdtPlan",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "CPDT is off; the optimizer runs its verbatim FP32 path",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Enforced,
    },
    ChannelDescriptor {
        channel: Channel::CfiePlan,
        name: "cfie_plan",
        producer: "CFIE",
        carries: "crate::cfie::CfiePlan",
        consumers: &["crates/nsl-codegen/src/compiler/entry_points.rs"],
        empty_means: "no serve block opted into CFIE",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Enforced,
    },
    ChannelDescriptor {
        channel: Channel::WggoOverrides,
        name: "wggo_overrides",
        producer: "WGGO",
        carries: "crate::wggo_overrides::WggoOverrides",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "no WGGO plan exists for this compile, so FASE falls back \
                      to fase::plan, the per-parameter mode table is skipped, \
                      and CSHA and WRGA receive no per-layer decisions",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Enforced,
    },
    ChannelDescriptor {
        channel: Channel::WggoPreplans,
        name: "wggo_preplans",
        producer: "WGGO",
        carries: "Vec<crate::wggo_prepass::WggoPrePlan>",
        consumers: &[
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-codegen/src/compiler/kernel.rs",
        ],
        empty_means: "the WGGO pre-pass produced no plan, so kernel synthesis \
                      and the train block see no pre-computed decisions",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Enforced,
    },
    ChannelDescriptor {
        channel: Channel::AdapterSites,
        name: "adapter_sites",
        producer: "WRGA",
        carries: "Vec<crate::wrga_adapter_inject::AdapterSite>",
        consumers: &[
            "crates/nsl-codegen/src/wrga_prescan.rs",
            "crates/nsl-codegen/src/wrga_adapter_rewrite.rs",
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-codegen/src/expr/access.rs",
        ],
        empty_means: "no adapter decorators resolved, so field access finds no \
                      synthesized adapter members and nothing is injected",
        dead_output: Invariant::Enforced,
        applied_implies_published: Invariant::Exempt(
            "also published by the driver prescan, which never records WRGA \
             as having run",
        ),
    },
    ChannelDescriptor {
        channel: Channel::CfieServeGen,
        name: "cfie_serve_gen",
        producer: "CFIE",
        carries: "crate::serve::CfieServeGen",
        consumers: &["crates/nsl-codegen/src/expr/calls.rs"],
        empty_means: "not inside a CFIE-active serve body, so a `generate()` \
                      call refuses cleanly instead of enqueueing into the \
                      dead M29 path",
        dead_output: Invariant::Exempt(
            "installed for the duration of a serve body and cleared on exit; \
             a serve block containing no generate() call legitimately never \
             reads it",
        ),
        applied_implies_published: Invariant::Enforced,
    },
];

/// Traffic on one channel.
///
/// Three counters rather than one "used" flag, because the three questions are
/// different: did anyone fill it, did anyone get something out, did anyone ask
/// and get nothing. Collapsing them loses exactly the distinction that makes
/// `read while empty` a finding.
struct Counters {
    publishes: AtomicUsize,
    reads_full: AtomicUsize,
    reads_empty: AtomicUsize,
}

impl Counters {
    const fn new() -> Self {
        Self {
            publishes: AtomicUsize::new(0),
            reads_full: AtomicUsize::new(0),
            reads_empty: AtomicUsize::new(0),
        }
    }
}

/// Per-channel traffic for the process, indexed by [`Channel`] discriminant.
static TRAFFIC: [Counters; CHANNELS.len()] = [const { Counters::new() }; CHANNELS.len()];

/// Compile-time: one descriptor per variant, and in the same order.
///
/// A test could check the count, but only after someone runs it — and the
/// failure mode is an out-of-bounds index inside `note_read`, which is reached
/// from ordinary compilation long before any gate. Failing the build is the
/// proportionate response.
const ALL_CHANNELS_HAVE_A_DESCRIPTOR: () = {
    assert!(
        Channel::ALL.len() == CHANNELS.len(),
        "every Channel variant needs a ChannelDescriptor in CHANNELS, and an \
         entry in Channel::ALL"
    );
    let mut i = 0;
    while i < CHANNELS.len() {
        assert!(
            CHANNELS[i].channel as usize == i,
            "CHANNELS is out of discriminant order — every channel would \
             report another's producer and consumers"
        );
        assert!(
            Channel::ALL[i] as usize == i,
            "Channel::ALL is out of discriminant order"
        );
        i += 1;
    }
};
const _: () = ALL_CHANNELS_HAVE_A_DESCRIPTOR;

/// A snapshot of one channel's traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelTraffic {
    pub publishes: usize,
    pub reads_full: usize,
    pub reads_empty: usize,
}

/// Read `channel`'s counters.
pub fn traffic(channel: Channel) -> ChannelTraffic {
    let c = &TRAFFIC[channel.idx()];
    ChannelTraffic {
        publishes: c.publishes.load(Relaxed),
        reads_full: c.reads_full.load(Relaxed),
        reads_empty: c.reads_empty.load(Relaxed),
    }
}

fn note_publish(channel: Channel) {
    TRAFFIC[channel.idx()].publishes.fetch_add(1, Relaxed);
}

/// Count a read, and hand the value straight back so that at every call site
/// the read and its accounting are one expression. A separate `note_read` the
/// accessor had to remember to call would be the call-site instrumentation
/// pattern this module exists to avoid.
fn note_read<T>(channel: Channel, v: Option<T>) -> Option<T> {
    let c = &TRAFFIC[channel.idx()];
    if v.is_some() {
        c.reads_full.fetch_add(1, Relaxed);
    } else {
        c.reads_empty.fetch_add(1, Relaxed);
    }
    v
}

/// Zero every counter. Test-only, for the same reason as
/// [`crate::pass_trace::reset`]: several compiles run in one test process.
///
/// Called BY `pass_trace::reset` so a test that resets one resets both. The
/// alternative — two resets a test must remember — is the shape of bug this
/// campaign keeps finding, and there is no case for clearing the pass trace
/// while keeping bus traffic from the compile that produced it.
pub fn reset() {
    for c in TRAFFIC.iter() {
        c.publishes.store(0, Relaxed);
        c.reads_full.store(0, Relaxed);
        c.reads_empty.store(0, Relaxed);
    }
}

/// The inter-pass channels, owned by `Compiler` as its `bus` field.
///
/// Every field is private to this module. See the module header for why that
/// is the mechanism rather than a convention.
#[derive(Default)]
pub struct PassBus {
    csha_bridge: Option<crate::csha_apply::BridgeResult>,
    csha_claimed_ops: std::collections::HashSet<u32>,
    csha_backward_claims: Option<crate::source_ad::CshaBackwardClaims>,
    wrga_plan: Option<crate::wrga::WrgaPlan>,
    adapter_prescan_plan: Option<crate::wrga::WrgaPlan>,
    cpkd_plan: Option<crate::cpkd::CpkdPlan>,
    cpdt_plan: Option<crate::cpdt::CpdtPlan>,
    cfie_plan: Option<crate::cfie::CfiePlan>,
    wggo_overrides: Option<crate::wggo_overrides::WggoOverrides>,
    wggo_preplans: Vec<crate::wggo_prepass::WggoPrePlan>,
    adapter_sites: Vec<crate::wrga_adapter_inject::AdapterSite>,
    cfie_serve_gen: Option<crate::serve::CfieServeGen>,
}

impl PassBus {
    // ── CSHA bridge ──────────────────────────────────────────────────

    /// Publish CSHA's per-layer bridge result.
    pub fn publish_csha_bridge(&mut self, v: crate::csha_apply::BridgeResult) {
        note_publish(Channel::CshaBridge);
        self.csha_bridge = Some(v);
    }

    pub fn csha_bridge(&self) -> Option<&crate::csha_apply::BridgeResult> {
        note_read(Channel::CshaBridge, self.csha_bridge.as_ref())
    }

    // ── CSHA claimed ops ─────────────────────────────────────────────

    /// Publish the Wengert op indices CSHA claimed.
    ///
    /// An EMPTY set is not counted as a publish. "CSHA claimed no ops" is the
    /// absence of a claim, not a claim of nothing, and the two must not read
    /// the same: CSHA calls this unconditionally after planning, so counting
    /// the empty case made every declining CSHA compile report a `DeadOutput`
    /// — a pass blamed for not being consumed on a set that had nothing in it
    /// to consume. The decline itself is already reported, by
    /// [`crate::pass_trace::record_disposition`].
    pub fn publish_csha_claimed_ops(&mut self, v: std::collections::HashSet<u32>) {
        if !v.is_empty() {
            note_publish(Channel::CshaClaimedOps);
        }
        self.csha_claimed_ops = v;
    }

    /// Is `op_idx` claimed by a CSHA boundary chain?
    ///
    /// Counted as a read of [`Channel::CshaClaimedOps`], and a read of the
    /// EMPTY set counts as an empty read even though `contains` answers it
    /// perfectly well. That is the useful accounting: a compile reporting
    /// thousands of empty reads is asking a question whose answer is fixed,
    /// which is worth being able to see.
    pub fn is_csha_claimed(&self, op_idx: u32) -> bool {
        let set = note_read(
            Channel::CshaClaimedOps,
            (!self.csha_claimed_ops.is_empty()).then_some(&self.csha_claimed_ops),
        );
        set.is_some_and(|s| s.contains(&op_idx))
    }

    // ── CSHA backward claims ─────────────────────────────────────────

    /// Publish the reverse-walk dispatch map.
    pub fn publish_csha_backward_claims(&mut self, v: crate::source_ad::CshaBackwardClaims) {
        note_publish(Channel::CshaBackwardClaims);
        self.csha_backward_claims = Some(v);
    }

    pub fn csha_backward_claims(&self) -> Option<&crate::source_ad::CshaBackwardClaims> {
        note_read(Channel::CshaBackwardClaims, self.csha_backward_claims.as_ref())
    }

    /// Hand the claims to the adjoint generator, emptying the channel.
    ///
    /// A CONSUME, counted as a read. The emptiness that follows is deliberate:
    /// `stmt.rs` (D2b) requires that the adjoint lowering never observe claims,
    /// and the pairing of this with [`Self::restore_csha_backward_claims`] and
    /// [`Self::clear_csha_backward_claims`] is what enforces the window.
    pub fn take_csha_backward_claims(&mut self) -> Option<crate::source_ad::CshaBackwardClaims> {
        note_read(Channel::CshaBackwardClaims, self.csha_backward_claims.take())
    }

    /// Put back what the adjoint generator handed back, for the forward
    /// lowering's fused-SDPA dispatch to read.
    ///
    /// NOT a publish. The value originated at the one real publish site; if
    /// this counted, a channel whose producer ran and whose consumers all
    /// ignored it would show two publishes and read as healthier than it is.
    pub fn restore_csha_backward_claims(
        &mut self,
        v: Option<crate::source_ad::CshaBackwardClaims>,
    ) {
        self.csha_backward_claims = v;
    }

    /// End the forward window: the adjoint/window lowering must not see claims.
    pub fn clear_csha_backward_claims(&mut self) {
        self.csha_backward_claims = None;
    }

    // ── WRGA plan ────────────────────────────────────────────────────

    pub fn publish_wrga_plan(&mut self, v: crate::wrga::WrgaPlan) {
        note_publish(Channel::WrgaPlan);
        self.wrga_plan = Some(v);
    }

    pub fn wrga_plan(&self) -> Option<&crate::wrga::WrgaPlan> {
        note_read(Channel::WrgaPlan, self.wrga_plan.as_ref())
    }

    // ── WRGA adapter prescan plan ────────────────────────────────────

    pub fn publish_adapter_prescan_plan(&mut self, v: crate::wrga::WrgaPlan) {
        note_publish(Channel::AdapterPrescanPlan);
        self.adapter_prescan_plan = Some(v);
    }

    pub fn adapter_prescan_plan(&self) -> Option<&crate::wrga::WrgaPlan> {
        note_read(Channel::AdapterPrescanPlan, self.adapter_prescan_plan.as_ref())
    }

    // ── CPKD plan ────────────────────────────────────────────────────

    pub fn publish_cpkd_plan(&mut self, v: crate::cpkd::CpkdPlan) {
        note_publish(Channel::CpkdPlan);
        self.cpkd_plan = Some(v);
    }

    /// Take the distillation facts for the build report. A consume: the report
    /// is rendered once per distill block.
    pub fn take_cpkd_plan(&mut self) -> Option<crate::cpkd::CpkdPlan> {
        note_read(Channel::CpkdPlan, self.cpkd_plan.take())
    }

    // ── CPDT plan ────────────────────────────────────────────────────

    pub fn publish_cpdt_plan(&mut self, v: crate::cpdt::CpdtPlan) {
        note_publish(Channel::CpdtPlan);
        self.cpdt_plan = Some(v);
    }

    pub fn cpdt_plan(&self) -> Option<&crate::cpdt::CpdtPlan> {
        note_read(Channel::CpdtPlan, self.cpdt_plan.as_ref())
    }

    // ── CFIE plan ────────────────────────────────────────────────────

    pub fn publish_cfie_plan(&mut self, v: crate::cfie::CfiePlan) {
        note_publish(Channel::CfiePlan);
        self.cfie_plan = Some(v);
    }

    pub fn cfie_plan(&self) -> Option<&crate::cfie::CfiePlan> {
        note_read(Channel::CfiePlan, self.cfie_plan.as_ref())
    }

    // ── WGGO overrides ───────────────────────────────────────────────

    pub fn publish_wggo_overrides(&mut self, v: crate::wggo_overrides::WggoOverrides) {
        note_publish(Channel::WggoOverrides);
        self.wggo_overrides = Some(v);
    }

    pub fn wggo_overrides(&self) -> Option<&crate::wggo_overrides::WggoOverrides> {
        note_read(Channel::WggoOverrides, self.wggo_overrides.as_ref())
    }

    /// Does a WGGO plan exist for this compile?
    ///
    /// A predicate rather than `wggo_overrides().is_some()` at the call site
    /// only so that the read is counted the same way either spelling.
    pub fn has_wggo_overrides(&self) -> bool {
        self.wggo_overrides().is_some()
    }

    /// Clear the overrides before compiling a train block that has no pre-plan.
    ///
    /// Not a publish. The clear is load-bearing rather than tidy: without it a
    /// train block with no pre-plan would inherit the PREVIOUS block's
    /// overrides, which is the stale-leak the pre-pass restructure was written
    /// to fix. Spelled as its own method so that intent survives.
    pub fn clear_wggo_overrides(&mut self) {
        self.wggo_overrides = None;
    }

    // ── WGGO pre-pass plans ──────────────────────────────────────────

    /// Publish the pre-pass plans. An EMPTY vec is not a publish, for the
    /// reason spelled out at [`Self::publish_csha_claimed_ops`].
    pub fn publish_wggo_preplans(&mut self, v: Vec<crate::wggo_prepass::WggoPrePlan>) {
        if !v.is_empty() {
            note_publish(Channel::WggoPreplans);
        }
        self.wggo_preplans = v;
    }

    pub fn wggo_preplans(&self) -> &[crate::wggo_prepass::WggoPrePlan] {
        note_read(
            Channel::WggoPreplans,
            (!self.wggo_preplans.is_empty()).then_some(&self.wggo_preplans),
        );
        &self.wggo_preplans
    }

    // ── WRGA adapter sites ───────────────────────────────────────────

    /// Publish the resolved adapter injection sites. An EMPTY vec is not a
    /// publish — WRGA writes this unconditionally, including when no decorator
    /// resolved to anything.
    pub fn publish_adapter_sites(&mut self, v: Vec<crate::wrga_adapter_inject::AdapterSite>) {
        if !v.is_empty() {
            note_publish(Channel::AdapterSites);
        }
        self.adapter_sites = v;
    }

    pub fn adapter_sites(&self) -> &[crate::wrga_adapter_inject::AdapterSite] {
        note_read(
            Channel::AdapterSites,
            (!self.adapter_sites.is_empty()).then_some(&self.adapter_sites),
        );
        &self.adapter_sites
    }

    // ── CFIE serve-generate context ──────────────────────────────────

    pub fn publish_cfie_serve_gen(&mut self, v: crate::serve::CfieServeGen) {
        note_publish(Channel::CfieServeGen);
        self.cfie_serve_gen = Some(v);
    }

    pub fn cfie_serve_gen(&self) -> Option<&crate::serve::CfieServeGen> {
        note_read(Channel::CfieServeGen, self.cfie_serve_gen.as_ref())
    }

    /// End the serve body's `generate()` window.
    ///
    /// Not a publish, and not counted — the same lifecycle-versus-production
    /// distinction as [`Self::clear_csha_backward_claims`].
    pub fn clear_cfie_serve_gen(&mut self) {
        self.cfie_serve_gen = None;
    }
}

/// A channel whose traffic pattern is a defect rather than a configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusFinding {
    /// The producer filled the channel and no consumer ever got a value out.
    DeadOutput { channel: Channel },
    /// A consumer read an empty channel although the producing pass ran and
    /// reported that it APPLIED a transformation. The consumer took its `None`
    /// branch, which looks identical to correct behaviour from every layer
    /// above — see the descriptor's `empty_means` for what that branch does.
    SilentDefault { channel: Channel, reads: usize },
}

/// Traffic patterns that indicate a defect.
///
/// Both findings need more than the counters, which is why this step comes
/// after steps 1-3 rather than before.
///
/// `SilentDefault` needs the pass trace AND the dispositions. "The producer
/// ran" is not enough on its own: a pass that runs and DECLINES leaves its
/// channel empty by definition, and on the CPU fixtures that is the common
/// case — CSHA admits no boundary chain, publishes no backward claims, and the
/// CCR site reads `None` and correctly falls back to per-op AD rules. Flagging
/// that would be flagging the pass for working. The finding is the
/// CONTRADICTION: a pass that says it applied a transformation, whose channel
/// is nevertheless empty when somebody reads it.
///
/// A producer with NO disposition is deliberately not flagged. That gap means
/// an exit with no `record_disposition` call — a real thing to fix, but a
/// finding about instrumentation, not about the build, and reporting it here
/// would attribute an unknown to a specific consumer's `None` branch.
///
/// Even with all of that, neither rule is universal — see [`Invariant`]. Both
/// consult the channel's own declaration, because for several channels the
/// tree does the flagged thing deliberately: `adapter_prescan_plan` is
/// published speculatively and superseded, and `csha_backward_claims` is
/// legitimately left empty by a CSHA that applies but whose chains the backward
/// SMEM validator rejects. Reporting those would be reporting correct code, and
/// a finding that cries wolf is a finding nobody reads.
pub fn findings() -> Vec<BusFinding> {
    let ran = crate::pass_trace::observed();
    let dispositions = crate::pass_trace::dispositions();
    let applied = |pass: &str| {
        matches!(
            dispositions.iter().find(|(p, _)| *p == pass),
            Some((_, crate::pass_trace::PassDisposition::Applied { .. }))
        )
    };
    CHANNELS
        .iter()
        .filter_map(|d| {
            let t = traffic(d.channel);
            if d.dead_output.enforced() && t.publishes > 0 && t.reads_full == 0 {
                return Some(BusFinding::DeadOutput { channel: d.channel });
            }
            // A publish with zero reads_full is already DeadOutput above; this
            // arm is the never-published case. `reads_empty > 0` keeps it to
            // channels somebody actually asked for.
            if d.applied_implies_published.enforced()
                && t.publishes == 0
                && t.reads_empty > 0
                && ran.contains(&d.producer)
                && applied(d.producer)
            {
                return Some(BusFinding::SilentDefault {
                    channel: d.channel,
                    reads: t.reads_empty,
                });
            }
            None
        })
        .collect()
}

/// The channels a pass publishes and the channels it consumes, as the edges a
/// scheduler would order by.
///
/// Consumers are attributed to a pass only where the consuming site IS a pass;
/// most consumers here are op lowering or the driver, which is why this returns
/// the producing edge and the consumer FILES rather than a pass-to-pass graph.
/// Claiming a pass-to-pass edge that does not exist would be worse than
/// claiming none: the whole point of the descriptor list is that it says only
/// what the tree supports.
pub fn edges_for(pass: &str) -> Vec<&'static ChannelDescriptor> {
    CHANNELS.iter().filter(|d| d.producer == pass).collect()
}

/// One line per channel that saw traffic, then one per finding.
///
/// Channels with no traffic at all are omitted: on a typical compile that is
/// six of eight, and printing them would bury the two that matter. The pass
/// trace's `did not run:` line already covers "this feature was off" — a bus
/// line saying the same thing for its channel adds nothing.
///
/// Uses the `[pass-bus]` prefix. Anything parsed positionally lives on its own
/// line for the reason spelled out at [`crate::pass_trace::report`]: the gates
/// split on markers, so a new line shape is safe where a wider line is not.
pub fn report() -> String {
    let mut s = String::new();
    for d in CHANNELS {
        let t = traffic(d.channel);
        if t.publishes == 0 && t.reads_full == 0 && t.reads_empty == 0 {
            continue;
        }
        s.push_str(&format!(
            "[pass-bus] {}: published {}x by {}, read {}x full, {}x empty\n",
            d.name, t.publishes, d.producer, t.reads_full, t.reads_empty
        ));
    }
    for f in findings() {
        match f {
            BusFinding::DeadOutput { channel } => {
                let d = channel.descriptor();
                s.push_str(&format!(
                    "[pass-bus] DEAD OUTPUT: {} published by {} and never read \
                     — the pass computed it for nobody\n",
                    d.name, d.producer
                ));
            }
            BusFinding::SilentDefault { channel, reads } => {
                let d = channel.descriptor();
                s.push_str(&format!(
                    "[pass-bus] SILENT DEFAULT: {} read {reads}x while empty \
                     although {} ran AND reported applying a transformation — \
                     consumers took the branch where {}\n",
                    d.name, d.producer, d.empty_means
                ));
            }
        }
    }
    s
}

/// The phases the producing PASS declares it is invoked from.
///
/// **Not a claim about where the publish happens**, and the difference is
/// real: WRGA's adapter prescan fills [`Channel::AdapterPrescanPlan`] and
/// [`Channel::AdapterSites`] from the driver, before any train block, without
/// calling `pass_trace::record("WRGA")` — so for those two edges this returns
/// `[TrainBlock]`, a phase in which the publish demonstrably does not occur.
/// Both descriptors record that, and both exempt themselves from
/// `applied_implies_published` for the same reason.
///
/// This is step 1's finding recurring one level down: the phase a pass is
/// *invoked* from is not the phase its *output* appears in, any more than the
/// stage it acts on is the phase it is called from. Attributing the publish
/// properly means recording it at the publishing driver, which is the next
/// increment, not this one. Saying so here is the alternative to a function
/// that quietly returns a plausible wrong answer.
///
/// A channel whose producer declares no phase at all cannot be ordered by this
/// edge; the drift gate asserts that only the `OutOfBand` passes are in that
/// position.
pub fn producer_phases(channel: Channel) -> &'static [CompilePhase] {
    crate::pass_registry::pass(channel.descriptor().producer)
        .map(|p| p.phases)
        .unwrap_or(&[])
}
