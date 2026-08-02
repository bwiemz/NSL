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
#[test]
fn dead_output_fires_when_a_published_channel_is_never_read() {
    let _g = guard();
    let mut bus = PassBus::default();
    pass_trace::record("CSHA");
    bus.publish_csha_bridge(bridge());

    assert_eq!(
        pass_bus::findings(),
        vec![BusFinding::DeadOutput { channel: Channel::CshaBridge }]
    );

    // ...and stops firing the moment a consumer reads it.
    assert!(bus.csha_bridge().is_some());
    assert!(
        pass_bus::findings().is_empty(),
        "one full read is enough to make a channel not-dead"
    );
}

/// The sharp one: a pass that says it APPLIED a transformation, whose channel
/// is nevertheless empty when a consumer reads it.
#[test]
fn silent_default_fires_only_when_the_producer_applied() {
    let _g = guard();
    let bus = PassBus::default();

    // 1. Producer never ran: an empty read is the ordinary case.
    assert!(bus.csha_backward_claims().is_none());
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
    assert!(bus.csha_backward_claims().is_none());
    assert!(
        pass_bus::findings().is_empty(),
        "a declining pass must not be flagged for the empty channel its \
         decline necessarily produces"
    );

    // 3. Producer ran and APPLIED, channel still empty: a contradiction.
    pass_trace::record_disposition("CSHA", PassDisposition::Applied { rewrites: 3 });
    assert!(bus.csha_backward_claims().is_none());
    let f = pass_bus::findings();
    assert!(
        f.contains(&BusFinding::SilentDefault {
            channel: Channel::CshaBackwardClaims,
            reads: 3,
        }),
        "expected a SilentDefault for csha_backward_claims, got {f:?}"
    );
}

/// A producer with no disposition at all is deliberately NOT flagged: the gap
/// means an exit with no `record_disposition`, which is a fact about
/// instrumentation rather than about this build.
#[test]
fn a_producer_with_no_disposition_is_not_flagged() {
    let _g = guard();
    let bus = PassBus::default();
    pass_trace::record("CSHA");
    assert!(bus.csha_backward_claims().is_none());
    assert!(
        pass_bus::findings().is_empty(),
        "no disposition means unknown, and an unknown must not be reported as \
         a specific consumer's silent default"
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
#[test]
fn both_findings_render_distinguishably() {
    let _g = guard();
    let mut bus = PassBus::default();
    pass_trace::record("CSHA");
    pass_trace::record_disposition("CSHA", PassDisposition::Applied { rewrites: 1 });
    bus.publish_csha_bridge(bridge()); // published, never read -> DeadOutput
    assert!(bus.csha_backward_claims().is_none()); // empty + applied -> SilentDefault

    let r = pass_bus::report();
    assert!(r.contains("DEAD OUTPUT: csha_bridge"), "missing DeadOutput:\n{r}");
    assert!(
        r.contains("SILENT DEFAULT: csha_backward_claims"),
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

/// `edges_for` is the scheduler-facing view: the channels a pass fills.
#[test]
fn edges_for_returns_a_passs_published_channels() {
    let csha: Vec<&str> = pass_bus::edges_for("CSHA").iter().map(|d| d.name).collect();
    assert_eq!(
        csha,
        vec!["csha_bridge", "csha_claimed_ops", "csha_backward_claims"]
    );
    assert_eq!(
        pass_bus::edges_for("WGGO").len(),
        0,
        "WGGO publishes no bus channel — its output goes through the Wengert \
         list itself, not a stash field"
    );
    assert!(
        pass_bus::edges_for("NotAPass").is_empty(),
        "an unknown pass must yield no edges rather than panicking"
    );
}
