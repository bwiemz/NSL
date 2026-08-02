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
//! They do it through `Compiler`. A pass stashes its output in a public field
//! — `last_wrga_plan`, `last_csha_bridge`, `cpdt_plan` — and some later stage
//! reads it back. That is a real dataflow graph: eight channels, one producing
//! pass each, twenty-odd consumer sites across five files. It was written down
//! nowhere, and `Compiler` has 74 fields, so nothing distinguished an
//! inter-pass channel from a scratch counter.
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
//! without being counted.
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
//! # Why it cannot change the build
//!
//! The accessors move exactly the values the public fields moved, and the
//! counters are process-global atomics that nothing in codegen reads. The gate
//! in `crates/nsl-cli/tests/pass_bus_gate.rs` pins this by comparing loss
//! streams with `NSL_PASS_TRACE` on and off.
//!
//! Counters are [`AtomicUsize`], not a `Mutex`: [`PassBus::is_csha_claimed`] is
//! a per-Wengert-op predicate called thousands of times per compile, and taking
//! a lock there would be a real cost paid on every build to serve an env var
//! almost nobody sets. A relaxed atomic increment is not.

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
}

impl Channel {
    const fn idx(self) -> usize {
        self as usize
    }

    /// This channel's declaration.
    pub fn descriptor(self) -> &'static ChannelDescriptor {
        &CHANNELS[self.idx()]
    }
}

/// What a channel carries, who fills it, and who reads it.
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
    pub producer: &'static str,
    /// The type it carries, as written in the field declaration.
    pub carries: &'static str,
    /// Repo-relative files that read it.
    pub consumers: &'static [&'static str],
    /// What an empty read legitimately means. Recorded because "the channel is
    /// empty" is the single most load-bearing fact about a consumer's `None`
    /// branch, and it was previously implicit in every one of them.
    pub empty_means: &'static str,
}

/// Every inter-pass channel, in [`Channel`] discriminant order.
///
/// # What is deliberately NOT here
///
/// `Compiler` has other `Option<T>` fields that look similar and are not
/// channels between passes:
///
/// * `flash_attn_aux`, `flash_attn_bwd_cache`, `fused_ce_*`,
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
    },
    ChannelDescriptor {
        channel: Channel::CshaClaimedOps,
        name: "csha_claimed_ops",
        producer: "CSHA",
        carries: "std::collections::HashSet<u32>",
        consumers: &["crates/nsl-codegen/src/compiler/mod.rs"],
        empty_means: "no op is claimed, so every op lowers normally — the \
                      empty set IS the correct answer, not a missing one",
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
    },
    ChannelDescriptor {
        channel: Channel::AdapterPrescanPlan,
        name: "adapter_prescan_plan",
        producer: "WRGA",
        carries: "crate::wrga::WrgaPlan",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "the prescan did not run (no adapter decorators), so \
                      train-block adapter injection has nothing to apply",
    },
    ChannelDescriptor {
        channel: Channel::CpkdPlan,
        name: "cpkd_plan",
        producer: "CPKD",
        carries: "crate::cpkd::CpkdPlan",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "no distill block compiled, so no distillation report is \
                      rendered",
    },
    ChannelDescriptor {
        channel: Channel::CpdtPlan,
        name: "cpdt_plan",
        producer: "CPDT",
        carries: "crate::cpdt::CpdtPlan",
        consumers: &["crates/nsl-codegen/src/stmt.rs"],
        empty_means: "CPDT is off; the optimizer runs its verbatim FP32 path",
    },
    ChannelDescriptor {
        channel: Channel::CfiePlan,
        name: "cfie_plan",
        producer: "CFIE",
        carries: "crate::cfie::CfiePlan",
        consumers: &["crates/nsl-codegen/src/compiler/entry_points.rs"],
        empty_means: "no serve block opted into CFIE",
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
            if t.publishes > 0 && t.reads_full == 0 {
                return Some(BusFinding::DeadOutput { channel: d.channel });
            }
            // A publish with zero reads_full is already DeadOutput above; this
            // arm is the never-published case. `reads_empty > 0` keeps it to
            // channels somebody actually asked for.
            if t.publishes == 0
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

/// The driver phase a pass would have to run in to fill `channel` before its
/// consumers, expressed as the producer's declared phases.
///
/// Exists so the ordering claim in the module header is checkable rather than
/// prose: a channel whose producer declares no phase at all cannot be ordered
/// by this edge, and the drift gate asserts that only the two `OutOfBand`
/// passes are in that position.
pub fn producer_phases(channel: Channel) -> &'static [CompilePhase] {
    crate::pass_registry::pass(channel.descriptor().producer)
        .map(|p| p.phases)
        .unwrap_or(&[])
}
