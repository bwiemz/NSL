//! Roadmap item 2 step 4 — unit gates for the pass bus's counters and findings.
//!
//! # Why this is its own test binary
//!
//! [`nsl_codegen::pass_bus`] keeps per-channel counters in process-global
//! atomics, and [`nsl_codegen::pass_bus::findings`] cross-references the
//! equally process-global pass trace. Put these tests in the crate's `--lib`
//! binary and ~200 unrelated unit tests would be driving instrumented pass
//! entry points in parallel with them: step 1 shipped exactly that and got a
//! load-dependent flake, passing at the default thread count and failing at
//! `--test-threads=64`. A separate binary removes the polluters, because they
//! are in a different process.
//!
//! Within THIS binary the tests still share those globals, so they take
//! `TEST_LOCK` and reset first. Both are required: the lock stops them
//! interleaving, the reset stops one test's traffic being read as another's.

use std::sync::{Mutex, MutexGuard};

use nsl_codegen::pass_bus::{self, BusFinding, Channel, PassBus, CHANNELS};
use nsl_codegen::pass_trace::{self, DeclineReason, PassDisposition};

static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Serialize, then clear both globals. `pass_trace::reset` clears the bus too.
fn guard() -> MutexGuard<'static, ()> {
    let g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    pass_trace::reset();
    g
}

fn bridge() -> nsl_codegen::csha_apply::BridgeResult {
    nsl_codegen::csha_apply::BridgeResult::default()
}

#[test]
fn a_publish_and_a_full_read_are_counted_separately() {
    let _g = guard();
    let mut bus = PassBus::default();
    assert_eq!(pass_bus::traffic(Channel::CshaBridge).publishes, 0);

    bus.publish_csha_bridge(bridge());
    let t = pass_bus::traffic(Channel::CshaBridge);
    assert_eq!((t.publishes, t.reads_full, t.reads_empty), (1, 0, 0));

    assert!(bus.csha_bridge().is_some());
    let t = pass_bus::traffic(Channel::CshaBridge);
    assert_eq!(
        (t.publishes, t.reads_full, t.reads_empty),
        (1, 1, 0),
        "a read that found a value must count as full, not as a publish"
    );
}

#[test]
fn a_read_of_an_unpublished_channel_counts_as_empty() {
    let _g = guard();
    let bus = PassBus::default();
    assert!(bus.cpdt_plan().is_none());
    assert!(bus.cfie_plan().is_none());
    assert!(bus.cfie_plan().is_none());

    let t = pass_bus::traffic(Channel::CpdtPlan);
    assert_eq!((t.publishes, t.reads_full, t.reads_empty), (0, 0, 1));
    let t = pass_bus::traffic(Channel::CfiePlan);
    assert_eq!((t.publishes, t.reads_full, t.reads_empty), (0, 0, 2));
}

/// Counting an empty claim set as a publish made every DECLINING CSHA compile
/// report a dead output — the pass blamed for producing something nobody
/// consumed, on a set with nothing in it to consume. `csha auto` on the CPU
/// fixture reproduced it directly.
#[test]
fn publishing_an_empty_claim_set_is_not_a_publish() {
    let _g = guard();
    let mut bus = PassBus::default();

    bus.publish_csha_claimed_ops(std::collections::HashSet::new());
    assert_eq!(
        pass_bus::traffic(Channel::CshaClaimedOps).publishes,
        0,
        "an empty claim set is the absence of a claim, not a claim of nothing"
    );

    bus.publish_csha_claimed_ops([7u32].into_iter().collect());
    assert_eq!(pass_bus::traffic(Channel::CshaClaimedOps).publishes, 1);
    assert!(bus.is_csha_claimed(7));
    assert!(!bus.is_csha_claimed(8));
    let t = pass_bus::traffic(Channel::CshaClaimedOps);
    assert_eq!(
        (t.reads_full, t.reads_empty),
        (2, 0),
        "both queries hit a non-empty set, so both are full reads regardless \
         of the answer"
    );
}

/// A restore puts back a value that was already published once. Counting it
/// would make a channel whose producer ran and whose consumers all ignored it
/// show two publishes and read as healthier than it is.
#[test]
fn restoring_claims_is_not_a_publish() {
    let _g = guard();
    let mut bus = PassBus::default();
    bus.restore_csha_backward_claims(None);
    bus.clear_csha_backward_claims();
    assert_eq!(
        pass_bus::traffic(Channel::CshaBackwardClaims).publishes,
        0,
        "restore/clear are lifecycle operations on a value already published"
    );
}

#[test]
fn taking_a_channel_counts_as_a_read_and_empties_it() {
    let _g = guard();
    let mut bus = PassBus::default();
    bus.publish_csha_bridge(bridge());
    // `take` lives on the two channels whose consumer is a one-shot: the
    // adjoint-generator handoff and the distill report.
    assert!(bus.take_csha_backward_claims().is_none());
    let t = pass_bus::traffic(Channel::CshaBackwardClaims);
    assert_eq!((t.reads_full, t.reads_empty), (0, 1));

    assert!(bus.take_cpkd_plan().is_none());
    assert_eq!(pass_bus::traffic(Channel::CpkdPlan).reads_empty, 1);
}

/// A pass filled a channel and nothing ever got a value out of it.
///
/// Uses `csha_claimed_ops`, which ENFORCES `dead_output`. Not `csha_bridge`:
/// that channel is exempt, because its own producer reads it back two
/// statements after publishing, so the finding is unreachable there by
/// construction.
#[test]
fn dead_output_fires_when_a_published_channel_is_never_read() {
    let _g = guard();
    let mut bus = PassBus::default();
    pass_trace::record("CSHA");
    bus.publish_csha_claimed_ops([1u32, 2].into_iter().collect());

    assert_eq!(
        pass_bus::findings(),
        vec![BusFinding::DeadOutput { channel: Channel::CshaClaimedOps }]
    );

    // ...and stops firing the moment a consumer reads it.
    assert!(bus.is_csha_claimed(1));
    assert!(
        pass_bus::findings().is_empty(),
        "one full read is enough to make a channel not-dead"
    );
}

/// A channel that exempts itself from `dead_output` stays quiet.
///
/// `csha_bridge` is exempt because its own producer reads it back two
/// statements after publishing (`stmt.rs`), to derive the backward claims — so
/// `reads_full >= publishes` always holds and the finding is unreachable there
/// by construction. Recording that as an exemption rather than leaving it
/// looking enforced matters: a future cleanup that used the local `bridge_out`
/// instead of re-reading the channel would otherwise make every declining
/// `--csha auto` build report a dead output, for a change that altered nothing.
#[test]
fn an_exempt_channel_does_not_report_dead_output() {
    let _g = guard();
    let mut bus = PassBus::default();
    pass_trace::record("CSHA");
    bus.publish_csha_bridge(bridge());
    let t = pass_bus::traffic(Channel::CshaBridge);
    assert_eq!(
        (t.publishes, t.reads_full, t.reads_empty),
        (1, 0, 0),
        "the precondition must hold, or the exemption is not what keeps this \
         quiet"
    );
    assert!(
        pass_bus::findings().is_empty(),
        "csha_bridge is exempt from DeadOutput and must not fire: {:?}",
        pass_bus::findings()
    );
}

/// The sharp one: a pass that says it APPLIED a transformation, whose channel
/// is nevertheless empty when a consumer reads it.
///
/// Uses `csha_bridge`, which ENFORCES `applied_implies_published` — CSHA
/// publishes it unconditionally once it has a plan, so an applying CSHA with an
/// empty bridge is a genuine contradiction.
#[test]
fn silent_default_fires_only_when_the_producer_applied() {
    let _g = guard();
    let bus = PassBus::default();

    // 1. Producer never ran: an empty read is the ordinary case.
    assert!(bus.csha_bridge().is_none());
    assert!(
        pass_bus::findings().is_empty(),
        "an empty channel whose pass never ran is a disabled feature, not a \
         defect"
    );

    // 2. Producer ran and DECLINED: still ordinary — an empty channel is the
    //    definition of declining, and the decline is already reported by the
    //    pass trace. This is the case that made the first version of the rule
    //    fire on every CPU fixture.
    pass_trace::record("CSHA");
    pass_trace::record_disposition(
        "CSHA",
        PassDisposition::Declined {
            reason: DeclineReason::NoCandidates("no boundary chain admitted"),
        },
    );
    assert!(bus.csha_bridge().is_none());
    assert!(
        pass_bus::findings().is_empty(),
        "a declining pass must not be flagged for the empty channel its \
         decline necessarily produces"
    );

    // 3. Producer ran and APPLIED, channel still empty: a contradiction.
    pass_trace::record_disposition("CSHA", PassDisposition::Applied { rewrites: 3 });
    assert!(bus.csha_bridge().is_none());
    let f = pass_bus::findings();
    assert!(
        f.contains(&BusFinding::SilentDefault {
            channel: Channel::CshaBridge,
            reads: 3,
        }),
        "expected a SilentDefault for csha_bridge, got {f:?}"
    );
}

/// A channel that exempts itself from `applied_implies_published` stays quiet
/// under exactly the conditions that would otherwise fire.
///
/// Both CSHA channels here are legitimately empty on an APPLYING run:
/// `collect_claimed_ops` returns an empty set when no boundary chain is
/// claimed, and `csha_backward_claims` is published only when the backward SMEM
/// validator admits at least one chain. Before the exemptions, the first was
/// armed by the very empty-publish suppression that avoids a false DeadOutput,
/// and the second was reachable on GPU with no test covering it.
#[test]
fn exempt_channels_do_not_report_silent_default_on_an_applying_pass() {
    let _g = guard();
    let mut bus = PassBus::default();
    pass_trace::record("CSHA");
    pass_trace::record_disposition("CSHA", PassDisposition::Applied { rewrites: 7 });

    // Publishing an empty claim set leaves publishes == 0 — SilentDefault's
    // precondition — and the read below would have triggered it.
    bus.publish_csha_claimed_ops(std::collections::HashSet::new());
    assert!(!bus.is_csha_claimed(0));
    assert!(bus.csha_backward_claims().is_none());

    assert!(
        pass_bus::findings().is_empty(),
        "an applying CSHA that claims no ops and admits no backward chain is \
         correct, not a silent default: {:?}",
        pass_bus::findings()
    );
}

#[test]
fn reset_clears_every_counter() {
    let _g = guard();
    let mut bus = PassBus::default();
    bus.publish_csha_bridge(bridge());
    assert!(bus.csha_bridge().is_some());
    assert!(bus.cpdt_plan().is_none());

    pass_bus::reset();
    for d in CHANNELS {
        let t = pass_bus::traffic(d.channel);
        assert_eq!(
            (t.publishes, t.reads_full, t.reads_empty),
            (0, 0, 0),
            "channel `{}` survived reset",
            d.name
        );
    }
}

/// `pass_trace::reset` must clear the bus as well. If it did not, bus traffic
/// would outlive the trace it is judged against, and `SILENT DEFAULT` would
/// evaluate one compile's reads against another compile's pass list.
#[test]
fn resetting_the_trace_also_resets_the_bus() {
    let _g = guard();
    let mut bus = PassBus::default();
    bus.publish_csha_bridge(bridge());
    assert_eq!(pass_bus::traffic(Channel::CshaBridge).publishes, 1);

    pass_trace::reset();
    assert_eq!(
        pass_bus::traffic(Channel::CshaBridge).publishes,
        0,
        "pass_trace::reset left bus traffic behind"
    );
}

/// Channels with no traffic are omitted; channels with traffic are named once
/// with all three counts.
#[test]
fn the_report_shows_active_channels_and_omits_silent_ones() {
    let _g = guard();
    let mut bus = PassBus::default();
    bus.publish_csha_bridge(bridge());
    assert!(bus.csha_bridge().is_some());

    let r = pass_bus::report();
    assert!(
        r.contains("[pass-bus] csha_bridge: published 1x by CSHA, read 1x full, 0x empty"),
        "unexpected report:\n{r}"
    );
    for quiet in ["wrga_plan", "cpdt_plan", "cfie_plan", "cpkd_plan"] {
        assert!(
            !r.contains(quiet),
            "channel `{quiet}` had no traffic and should not be printed:\n{r}"
        );
    }
}

/// Both finding categories must be legible and distinct in the rendered
/// report. Two categories that read the same would be worse than one.
///
/// Uses channels that ENFORCE the invariant each finding tests:
/// `csha_claimed_ops` for DeadOutput, `csha_bridge` for SilentDefault.
#[test]
fn both_findings_render_distinguishably() {
    let _g = guard();
    let mut bus = PassBus::default();
    pass_trace::record("CSHA");
    pass_trace::record_disposition("CSHA", PassDisposition::Applied { rewrites: 1 });
    // published, never read -> DeadOutput (enforced on this channel)
    bus.publish_csha_claimed_ops([3u32].into_iter().collect());
    // empty + producer applied -> SilentDefault (enforced on this channel)
    assert!(bus.csha_bridge().is_none());

    let r = pass_bus::report();
    assert!(
        r.contains("DEAD OUTPUT: csha_claimed_ops"),
        "missing DeadOutput:\n{r}"
    );
    assert!(
        r.contains("SILENT DEFAULT: csha_bridge"),
        "missing SilentDefault:\n{r}"
    );
    let dead = r.lines().filter(|l| l.contains("DEAD OUTPUT")).count();
    let silent = r.lines().filter(|l| l.contains("SILENT DEFAULT")).count();
    assert_eq!((dead, silent), (1, 1), "each finding must appear once:\n{r}");
}

/// Every descriptor's `empty_means` says something. A blank one would make the
/// SilentDefault line read "consumers took the branch where ." — and the whole
/// value of that finding is telling the reader what the silent branch DOES.
#[test]
fn every_channel_explains_what_an_empty_read_means() {
    for d in CHANNELS {
        assert!(
            d.empty_means.len() > 20,
            "channel `{}` has no useful `empty_means`: {:?}",
            d.name,
            d.empty_means
        );
        assert!(!d.carries.is_empty(), "channel `{}` declares no type", d.name);
    }
}

// ─── Step 5: sequence stamps, ReadBeforePublish, dependency order ───────────

/// The stamps record ORDER, which the totals cannot: `publishes: 1,
/// reads_empty: 1` describes both interleavings, and only ask-before-answer
/// is the finding.
#[test]
fn read_before_publish_fires_only_when_the_ask_preceded_the_answer() {
    let _g = guard();

    // Ask, THEN answer, then ask again successfully: the early reader planned
    // against the empty default while the value existed later in the process.
    // `csha_claimed_ops` enforces the invariant, and the trailing full read
    // keeps DeadOutput (which would win otherwise) out of the picture.
    let mut bus = PassBus::default();
    assert!(!bus.is_csha_claimed(1)); // empty read, stamped
    bus.publish_csha_claimed_ops([1u32].into_iter().collect());
    assert!(bus.is_csha_claimed(1)); // full read: not DeadOutput
    let t = pass_bus::traffic(Channel::CshaClaimedOps);
    assert!(
        t.first_empty_read_seq != 0 && t.first_publish_seq > t.first_empty_read_seq,
        "the scenario must actually be ask-before-answer: {t:?}"
    );
    assert_eq!(
        pass_bus::findings(),
        vec![BusFinding::ReadBeforePublish { channel: Channel::CshaClaimedOps }]
    );

    // Control: answer, then ask. Same totals for publishes/reads_full, no
    // empty read before the publish — and no finding.
    pass_trace::reset();
    let mut bus = PassBus::default();
    bus.publish_csha_claimed_ops([1u32].into_iter().collect());
    assert!(bus.is_csha_claimed(1));
    assert!(
        pass_bus::findings().is_empty(),
        "publish-then-read is the healthy order and must not fire: {:?}",
        pass_bus::findings()
    );
}

/// A channel that EXEMPTS itself from ReadBeforePublish stays quiet under
/// exactly the pattern that would otherwise fire.
///
/// `wggo_overrides` is the production case the exemption exists for: with no
/// pre-plan, FASE's plan-selection read is empty (it falls back to
/// `fase::plan`, correctly) and the in-place replan's publish postdates it,
/// guarded by the driver's divergence refusal.
#[test]
fn an_exempt_channel_does_not_report_read_before_publish() {
    let _g = guard();
    let mut bus = PassBus::default();
    assert!(bus.wggo_overrides().is_none()); // FASE's empty read
    bus.publish_wggo_overrides(nsl_codegen::wggo_overrides::WggoOverrides {
        per_layer: vec![],
    }); // the in-place replan's late publish
    assert!(bus.wggo_overrides().is_some());
    let t = pass_bus::traffic(Channel::WggoOverrides);
    assert!(
        t.first_empty_read_seq != 0 && t.first_publish_seq > t.first_empty_read_seq,
        "the pattern must be present, or the exemption is not what keeps \
         this quiet: {t:?}"
    );
    assert!(
        pass_bus::findings().is_empty(),
        "wggo_overrides is exempt from ReadBeforePublish and must not fire: \
         {:?}",
        pass_bus::findings()
    );
}

/// `reset` must clear the stamps too, or a finding would compare one
/// compile's ask against another's answer.
#[test]
fn reset_clears_the_sequence_stamps() {
    let _g = guard();
    let bus = PassBus::default();
    assert!(!bus.is_csha_claimed(0));
    assert!(pass_bus::traffic(Channel::CshaClaimedOps).first_empty_read_seq != 0);
    pass_bus::reset();
    let t = pass_bus::traffic(Channel::CshaClaimedOps);
    assert_eq!(
        (t.first_publish_seq, t.first_empty_read_seq),
        (0, 0),
        "stamps survived reset"
    );
}

/// The invocation-order checker fires on a declared InvocationOrdered edge
/// whose consumer was reached first — and ONLY on that.
#[test]
fn dependency_order_violations_fire_in_exactly_the_declared_direction() {
    let _g = guard();

    // Consumer before producer: the violation.
    pass_trace::record("CSHA");
    pass_trace::record("WGGO");
    assert_eq!(
        pass_bus::dependency_order_violations(),
        vec![("WGGO", "CSHA", Channel::WggoOverrides)]
    );

    // Producer before consumer: the declared order.
    pass_trace::reset();
    pass_trace::record("WGGO");
    pass_trace::record("CSHA");
    pass_trace::record("WRGA");
    assert!(pass_bus::dependency_order_violations().is_empty());

    // Only one side ran: nothing to order.
    pass_trace::reset();
    pass_trace::record("CSHA");
    assert!(pass_bus::dependency_order_violations().is_empty());
    pass_trace::reset();
    pass_trace::record("WGGO");
    assert!(pass_bus::dependency_order_violations().is_empty());
}

/// THE pin for the OrderClaim distinction: FASE before WGGO is a legitimate
/// compile path (the in-place replan), so the FASE edge is ValueOrderedOnly
/// and must NOT produce an invocation-order violation. Flip that edge to
/// InvocationOrdered and this test is the one that says why not.
#[test]
fn a_value_ordered_only_edge_never_produces_an_invocation_violation() {
    let _g = guard();
    pass_trace::record("FASE");
    pass_trace::record("WGGO");
    assert!(
        pass_bus::dependency_order_violations().is_empty(),
        "FASE-then-WGGO is the in-place-replan path; the WGGO->FASE edge is \
         ValueOrderedOnly precisely so this is not flagged: {:?}",
        pass_bus::dependency_order_violations()
    );
}

/// The third finding renders distinctly, naming the channel, the producer,
/// and what the early reader's branch did.
#[test]
fn read_before_publish_renders_with_its_own_marker() {
    let _g = guard();
    let mut bus = PassBus::default();
    assert!(!bus.is_csha_claimed(1));
    bus.publish_csha_claimed_ops([1u32].into_iter().collect());
    assert!(bus.is_csha_claimed(1));
    let r = pass_bus::report();
    assert!(
        r.contains("READ BEFORE PUBLISH: csha_claimed_ops"),
        "missing or mis-attributed ReadBeforePublish:\n{r}"
    );
    assert!(
        r.contains("before CSHA first published it"),
        "the line must name the producer:\n{r}"
    );
}

/// `edges_for` is the scheduler-facing view: the channels a pass fills.
#[test]
fn edges_for_returns_a_passs_published_channels() {
    let csha: Vec<&str> = pass_bus::edges_for("CSHA").iter().map(|d| d.name).collect();
    assert_eq!(
        csha,
        vec!["csha_bridge", "csha_claimed_ops", "csha_backward_claims"]
    );
    // The first revision of this module asserted WGGO published nothing,
    // "because its output goes through the Wengert list". False, and false
    // about the very edge the ordering argument depends on: FASE recipe
    // selection reads `wggo_overrides`, which is why FASE must follow WGGO.
    let wggo: Vec<&str> = pass_bus::edges_for("WGGO").iter().map(|d| d.name).collect();
    assert_eq!(wggo, vec!["wggo_overrides", "wggo_preplans"]);
    assert!(
        pass_bus::edges_for("NotAPass").is_empty(),
        "an unknown pass must yield no edges rather than panicking"
    );
}
