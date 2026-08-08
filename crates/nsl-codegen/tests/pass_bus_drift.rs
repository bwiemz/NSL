//! Roadmap item 2 step 4 — static gates for the pass bus.
//!
//! `pass_bus.rs` claims eight things about the tree: that each channel has a
//! field, that the field is private, that its producer is a registered pass,
//! that its consumers are files that exist and mention it. Prose cannot be
//! compiled, so these assert it instead — the same reasoning that produced
//! `pass_registry_drift.rs`, applied to the channel declarations.
//!
//! Static rather than runtime because the sites are ENUMERABLE. A runtime
//! assert only fires on paths some fixture covers, and no CPU fixture makes
//! CSHA claim a chain or CFIE compile a serve block; a scan of the source
//! checks every channel on every build. That is the same conclusion reached
//! for `pass_trace::record` in step 1.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use nsl_codegen::pass_bus::{Channel, CHANNELS};
use nsl_codegen::pass_registry::{CompilePhase, PASSES};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Drop `//` comment lines, then remove ALL whitespace.
///
/// Both halves are load-bearing, and each was added because the naive version
/// was wrong in a way that still passed:
///
/// * keeping comments made `wrga_adapter_init.rs` an undeclared consumer of
///   `wrga_plan` on the strength of a doc comment naming the accessor — a gate
///   that fires on prose trains its reader to add exemptions;
/// * matching line by line MISSED every multi-line call chain, and this tree
///   writes them constantly: `compiler\n.bus\n.wrga_plan()` in
///   `entry_points.rs`, `test_helpers.rs` and `wengert_lower.rs` are real
///   consumers that a line-based scan reported as clean. A completeness gate
///   with a blind spot for the dominant formatting style is worse than none,
///   because it certifies the thing it cannot see.
///
/// Removing all whitespace is safe here only because every pattern searched
/// for contains none.
fn code_only(text: &str) -> String {
    text.lines()
        .map(strip_trailing_comment)
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
        .split_whitespace()
        .collect()
}

/// Truncate a line at its trailing `//`.
///
/// Dropping only lines that START with `//` left a hole: `let x = foo(); // see
/// .wrga_plan()` satisfied the presence half from prose, which is the more
/// dangerous direction — it makes a channel look consumed when nothing consumes
/// it. The `:` guard keeps `https://` and `path::to::thing` intact; nothing
/// searched for here can appear after a `:` in a way that matters.
fn strip_trailing_comment(line: &str) -> &str {
    let b = line.as_bytes();
    for i in 0..b.len().saturating_sub(1) {
        if b[i] == b'/' && b[i + 1] == b'/' && (i == 0 || b[i - 1] != b':') {
            return &line[..i];
        }
    }
    line
}

fn pass_bus_source() -> String {
    std::fs::read_to_string(repo_root().join("crates/nsl-codegen/src/pass_bus.rs"))
        .expect("pass_bus.rs must be readable")
}

/// Every channel names a pass that exists.
///
/// The single cheapest gate here, and the one that catches the drift class
/// `pass_registry` was built for: a renamed pass leaves the channel pointing at
/// nothing, `findings()` silently stops asking whether the producer ran, and
/// `SILENT DEFAULT` quietly becomes unreportable.
#[test]
fn every_channel_producer_is_a_registered_pass() {
    let names: BTreeSet<&str> = PASSES.iter().map(|p| p.name).collect();
    for d in CHANNELS {
        assert!(
            names.contains(d.producer),
            "channel `{}` names producer `{}`, which is not in pass_registry::PASSES",
            d.name,
            d.producer
        );
    }
    assert!(
        CHANNELS.len() >= 12,
        "only {} channels — the list shrank; if a channel was genuinely \
         removed, lower this floor deliberately",
        CHANNELS.len()
    );
}

/// Every `Invariant::Exempt` carries a real reason.
///
/// The whole argument for a sum type over a `bool` is that an exemption without
/// a recorded reason outlives the reason for it and cannot be told apart from
/// an oversight — the same argument `pass_registry::WikiCoverage` makes. An
/// empty string would give back the `bool` while looking like the sum type.
#[test]
fn every_finding_exemption_records_why() {
    use nsl_codegen::pass_bus::Invariant;
    let mut exemptions = 0usize;
    for d in CHANNELS {
        for (what, inv) in [
            ("dead_output", d.dead_output),
            ("applied_implies_published", d.applied_implies_published),
            ("read_before_publish", d.read_before_publish),
        ] {
            if let Invariant::Exempt(reason) = inv {
                exemptions += 1;
                assert!(
                    reason.len() > 30,
                    "channel `{}` exempts itself from {what} with no useful \
                     reason: {reason:?}",
                    d.name
                );
            }
        }
    }
    assert!(
        exemptions >= 6,
        "only {exemptions} exemptions — if the finding rules became universal, \
         say so deliberately; this floor exists so the enum cannot quietly \
         decay into 'always Enforced'"
    );
}

/// Both findings must still be ENFORCED somewhere, or the rule is dead code
/// dressed as a check.
///
/// This is the failure mode `pass_registry` deleted its `status` field to
/// avoid: an enum implying a distinction nothing actually makes. If every
/// channel ends up exempt from `SilentDefault`, the honest move is to delete
/// the finding, not to keep a rule that can never fire.
#[test]
fn both_findings_are_enforced_by_at_least_one_channel() {
    use nsl_codegen::pass_bus::Invariant;
    let dead = CHANNELS.iter().filter(|d| d.dead_output == Invariant::Enforced).count();
    let applied = CHANNELS
        .iter()
        .filter(|d| d.applied_implies_published == Invariant::Enforced)
        .count();
    let ordered = CHANNELS
        .iter()
        .filter(|d| d.read_before_publish == Invariant::Enforced)
        .count();
    assert!(dead > 0, "no channel enforces DeadOutput — the finding is dead code");
    assert!(
        applied > 0,
        "no channel enforces SilentDefault — delete the finding rather than \
         keep one that can never fire"
    );
    assert!(
        ordered > 0,
        "no channel enforces ReadBeforePublish — delete the finding rather \
         than keep one that can never fire"
    );
}

/// `Channel::ALL` must list every variant, in order. The compile-time assert in
/// `pass_bus.rs` already checks the count and the ordering; this checks that
/// the list is not merely self-consistent but agrees with the descriptors.
#[test]
fn channel_all_agrees_with_the_descriptors() {
    assert_eq!(Channel::ALL.len(), CHANNELS.len());
    for (i, c) in Channel::ALL.iter().enumerate() {
        assert_eq!(
            c.descriptor().name,
            CHANNELS[i].name,
            "Channel::ALL[{i}] does not match CHANNELS[{i}]"
        );
    }
}

/// `Channel::descriptor` indexes `CHANNELS` by discriminant, so a descriptor
/// inserted out of order would make every channel report another's producer
/// and consumers — wrong, and wrong in a way that still renders plausibly.
#[test]
fn channel_descriptors_are_in_enum_order() {
    for (i, d) in CHANNELS.iter().enumerate() {
        assert_eq!(
            d.channel.descriptor().name,
            d.name,
            "CHANNELS[{i}] (`{}`) does not round-trip through its own \
             discriminant — the enum and the list are out of order",
            d.name
        );
    }
}

/// Every channel has a private field of the declared type, and an accessor.
///
/// Privacy is the mechanism the module depends on: it is what makes the
/// accessors the only way to reach a channel, and therefore what makes the
/// traffic counts complete. A `pub` slipping onto one field would silently
/// reopen the direct-access path for that channel alone, and nothing else in
/// the build would notice.
#[test]
fn every_channel_has_a_private_field_and_an_accessor() {
    let src = pass_bus_source();
    let body = src
        .split_once("pub struct PassBus {")
        .expect("PassBus struct must exist")
        .1
        .split_once("\n}")
        .expect("PassBus struct must close")
        .0;

    for d in CHANNELS {
        let decl = format!("    {}: ", d.name);
        assert!(
            body.contains(&decl),
            "channel `{}` has no field on PassBus (looked for `{}`)",
            d.name,
            decl.trim_end()
        );
        assert!(
            !body.contains(&format!("pub {}", d.name)),
            "PassBus field `{}` is public — the accessors are no longer the \
             only way in, so its traffic counts can no longer be trusted",
            d.name
        );
        assert!(
            src.contains(&format!("fn publish_{}(", d.name))
                || src.contains(&format!("fn publish_{}_", d.name)),
            "channel `{}` has no publish accessor",
            d.name
        );
        // The declared type must be the one the field actually holds.
        let field_line = body
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{}: ", d.name)))
            .unwrap();
        assert!(
            field_line.contains(d.carries),
            "channel `{}` declares it carries `{}` but the field reads `{}`",
            d.name,
            d.carries,
            field_line.trim()
        );
    }
}

/// No struct field on `PassBus` is missing a descriptor.
///
/// The direction that actually rots. Adding a channel and forgetting the
/// descriptor costs nothing at compile time and leaves the new channel
/// uncounted, unreported, and absent from the dependency picture the whole
/// module exists to provide — which is exactly how it would come to be
/// believed that the bus had been fully mapped.
#[test]
fn every_pass_bus_field_is_a_declared_channel() {
    let src = pass_bus_source();
    let body = src
        .split_once("pub struct PassBus {")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;

    let declared: BTreeSet<&str> = CHANNELS.iter().map(|d| d.name).collect();
    let mut fields = 0usize;
    for line in body.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        let Some((name, _)) = t.split_once(": ") else {
            continue;
        };
        fields += 1;
        assert!(
            declared.contains(name),
            "PassBus field `{name}` has no ChannelDescriptor — add one to \
             CHANNELS, or the channel is invisible to the report and the gates"
        );
    }
    assert_eq!(
        fields,
        CHANNELS.len(),
        "PassBus has {fields} fields but {} descriptors",
        CHANNELS.len()
    );
}

/// Declared consumer files exist and mention the channel's accessor.
///
/// Files rather than functions, and `mentions the accessor` rather than a line
/// number, because both of the tighter claims rot on the first unrelated edit
/// while this one stays true and still fails when a consumer is deleted or
/// moved — which is the case that matters, since a channel whose last consumer
/// leaves becomes dead output that nothing announces.
#[test]
fn declared_consumer_files_exist_and_read_the_channel() {
    let root = repo_root();
    for d in CHANNELS {
        assert!(
            !d.consumers.is_empty(),
            "channel `{}` declares no consumers — a channel nothing reads is \
             dead output, which should be recorded as such rather than left \
             blank",
            d.name
        );
        for c in d.consumers {
            let p = root.join(c);
            assert!(p.exists(), "channel `{}`: consumer file {c} does not exist", d.name);
            let src = code_only(&std::fs::read_to_string(&p).unwrap());
            // Comments are stripped in BOTH directions. This half asserts
            // presence, so prose mentioning an accessor would satisfy it
            // falsely — the more dangerous of the two failure modes, since it
            // makes a channel look consumed when nothing consumes it.
            let reads = src.contains(&format!("bus.{}(", d.name))
                || src.contains(&format!("bus.take_{}(", d.name))
                || (d.name == "csha_claimed_ops" && src.contains("bus.is_csha_claimed("));
            assert!(
                reads,
                "channel `{}`: {c} is declared a consumer but never calls its \
                 accessor",
                d.name
            );
        }
    }
}

/// A channel can only order its producer against its consumers if the producer
/// runs in a declared phase. The two `OutOfBand` passes are the exception, and
/// the gate pins that they are the ONLY exception — so a pass that loses its
/// phase declaration cannot quietly take its channel's ordering guarantee with
/// it.
#[test]
fn only_out_of_band_producers_lack_a_compile_phase() {
    for d in CHANNELS {
        let phases = nsl_codegen::pass_bus::producer_phases(d.channel);
        let pass = nsl_codegen::pass_registry::pass(d.producer).unwrap_or_else(|| {
            panic!(
                "channel `{}` names unregistered producer `{}` — see \
                 every_channel_producer_is_a_registered_pass",
                d.name, d.producer
            )
        });
        if phases.is_empty() {
            assert_eq!(
                pass.stage,
                nsl_codegen::pass_registry::PipelineStage::OutOfBand,
                "channel `{}`'s producer {} declares no CompilePhase but is \
                 not OutOfBand — nothing can be ordered by this edge",
                d.name,
                d.producer
            );
        } else {
            assert!(
                phases.iter().all(|p| *p != CompilePhase::OutOfBand),
                "channel `{}`'s producer {} mixes OutOfBand with real phases",
                d.name,
                d.producer
            );
        }
    }
}

/// **Pins a finding, so it cannot be lost.**
///
/// `Compiler::is_csha_claimed` — the sole consumer of
/// [`Channel::CshaClaimedOps`] — has no callers. CSHA computes the claimed-op
/// set on every plan and publishes it for a predicate nothing invokes. This is
/// the `find_best_variant` shape (an autotuner chooser with zero production
/// callers, pinned by a negative test after the fact) and it is what the bus's
/// `DEAD OUTPUT` finding reports at runtime whenever CSHA claims a chain.
///
/// It is not obviously a bug: the field was landed by 3bf484c1 as preparation
/// for CSHA A.2.2-A.2.4, which suppress redundant launches for claimed ops and
/// have not shipped. This test records the state so that wiring the first real
/// caller is a deliberate edit that fails here and updates the note, rather
/// than something discovered by grep months later.
#[test]
fn is_csha_claimed_still_has_no_production_callers() {
    let src_root = repo_root().join("crates/nsl-codegen/src");
    let mut callers = Vec::new();
    let mut scanned = 0usize;
    let mut dirs = vec![src_root.clone()];
    while let Some(dir) = dirs.pop() {
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            // The two definitions themselves: the accessor and the delegating
            // wrapper on `Compiler`.
            if p.ends_with("pass_bus.rs") || p.ends_with("compiler/mod.rs") {
                continue;
            }
            scanned += 1;
            let text = std::fs::read_to_string(&p).unwrap();
            for (i, line) in text.lines().enumerate() {
                if line.contains("is_csha_claimed(") && !line.trim_start().starts_with("//") {
                    callers.push(format!("{}:{}", p.display(), i + 1));
                }
            }
        }
    }
    assert!(
        scanned > 100,
        "only {scanned} files scanned — the walk failed, so an empty result \
         would prove nothing"
    );
    assert!(
        callers.is_empty(),
        "is_csha_claimed gained callers: {callers:?}. That is good news — the \
         CshaClaimedOps channel is no longer dead output. Update its \
         ChannelDescriptor consumers and delete this test."
    );
}

/// Every `PassBus` accessor either counts, or is a declared lifecycle
/// operation that says why it does not.
///
/// Privacy guarantees no access without an accessor. It does NOT guarantee the
/// accessor counts — a new `pub fn wggo_overrides_unchecked(&self)` returning
/// the field would compile, satisfy every other gate here, and silently make
/// that channel's traffic wrong in the direction that hides findings. This is
/// the static enforcement the privacy argument was resting on and did not have.
///
/// The lifecycle exemptions are operations on a value already published once:
/// counting a restore as a publish would make a channel nobody consumes look
/// busier than it is. Adding another is a deliberate edit here.
///
/// `is_published` is exempt on the opposite ground: it is the pass scheduler's
/// OCCUPANCY PROBE, and counting it would be self-defeating. The scheduler
/// consults it once per scheduled pass to check `applied_implies_published`;
/// if that consult counted as a `reads_full`, every channel the scheduler
/// watches would show a read it never had, and the `dead_output` finding —
/// "published, never read" — would be disarmed for exactly those channels. A
/// checker that breaks the checker underneath it is worse than no checker.
#[test]
fn every_accessor_counts_or_declares_why_not() {
    const UNCOUNTED: &[&str] = &[
        "restore_csha_backward_claims",
        "clear_csha_backward_claims",
        "clear_wggo_overrides",
        "clear_cpdt_plan",
        "clear_cfie_serve_gen",
        "is_published",
    ];

    let src = pass_bus_source();
    let body = src
        .split_once("impl PassBus {")
        .expect("impl PassBus must exist")
        .1;
    let body = &body[..body.find("\n}").expect("impl PassBus must close")];

    // Split into methods at each `    pub fn` / `    pub(crate) fn` header.
    let mut methods: Vec<(String, String)> = Vec::new();
    for line in body.lines() {
        let t = line.trim_start();
        if let Some(rest) = t
            .strip_prefix("pub fn ")
            .or_else(|| t.strip_prefix("pub(crate) fn "))
        {
            let name = rest.split('(').next().unwrap_or("").to_string();
            methods.push((name, String::new()));
        }
        if let Some(last) = methods.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }

    assert!(
        methods.len() >= 25,
        "only {} accessors parsed — the scan failed, so an empty result would \
         prove nothing",
        methods.len()
    );

    let mut uncounted = Vec::new();
    for (name, body) in &methods {
        let counts = body.contains("note_publish(")
            || body.contains("note_read(")
            // `has_wggo_overrides` delegates to the counting accessor rather
            // than reading the field, which is the pattern to encourage: one
            // narrow escape, not a general "mentions the field" clause that
            // would readmit everything this gate exists to catch.
            || body.contains("self.wggo_overrides()");
        if !counts && !UNCOUNTED.contains(&name.as_str()) {
            uncounted.push(name.clone());
        }
    }
    assert!(
        uncounted.is_empty(),
        "PassBus accessors that neither count nor are declared uncounted: \
         {uncounted:?} — call note_publish/note_read, or add the name to \
         UNCOUNTED here with a doc comment at the method saying why"
    );

    // The allowlist must not outlive its entries.
    for name in UNCOUNTED {
        assert!(
            methods.iter().any(|(n, _)| n == name),
            "UNCOUNTED lists `{name}`, which no longer exists — a stale \
             exemption is how a real gap gets waved through later"
        );
    }
}

/// The finding categories must render distinguishably. A report where two
/// categories produced the same text would be worse than one category.
#[test]
fn the_finding_categories_render_differently() {
    let src = pass_bus_source();
    assert!(src.contains("DEAD OUTPUT:"), "DeadOutput lost its marker");
    assert!(src.contains("SILENT DEFAULT:"), "SilentDefault lost its marker");
    assert!(
        src.contains("READ BEFORE PUBLISH:"),
        "ReadBeforePublish lost its marker"
    );
}

/// Every consumer file listed for a channel is one the tree actually has, AND
/// every file that calls a bus accessor is listed by the channel it reads.
///
/// The completeness direction: a new consumer added in a file nobody declared
/// leaves the descriptor understating the blast radius of a channel, which is
/// the fact a future scheduler would order by.
#[test]
fn every_file_that_reads_a_channel_is_a_declared_consumer() {
    let root = repo_root();
    let src_root = root.join("crates/nsl-codegen/src");
    let mut missing = Vec::new();
    let mut dirs = vec![src_root.clone()];
    let mut scanned = 0usize;
    while let Some(dir) = dirs.pop() {
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") || p.ends_with("pass_bus.rs") {
                continue;
            }
            scanned += 1;
            let text = std::fs::read_to_string(&p).unwrap();
            let rel = p.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
            let src = code_only(&text);
            for d in CHANNELS {
                // Reads only. A publish site is the producing pass's own file
                // and is described by `producer`, not `consumers`.
                let reads = src.contains(&format!("bus.{}(", d.name))
                    || src.contains(&format!("bus.take_{}(", d.name))
                    || (d.name == "csha_claimed_ops" && src.contains("bus.is_csha_claimed("));
                if reads && !d.consumers.contains(&rel.as_str()) {
                    missing.push(format!("{} reads `{}`", rel, d.name));
                }
            }
        }
    }
    assert!(scanned > 100, "only {scanned} files scanned — the walk failed");
    assert!(
        missing.is_empty(),
        "undeclared channel consumers: {missing:?} — add each file to its \
         ChannelDescriptor's `consumers`"
    );
}

/// `test_helpers.rs` is behind `cfg(any(test, feature = \"test-helpers\"))`, so
/// it is a consumer only under that feature. Recorded here rather than left as
/// a surprise for whoever wonders why the walk above sees it.
#[test]
fn the_wrga_plan_channel_documents_its_test_only_consumer() {
    let d = Channel::WrgaPlan.descriptor();
    assert!(
        d.consumers.contains(&"crates/nsl-codegen/src/test_helpers.rs"),
        "test_helpers.rs reads wrga_plan under the test-helpers feature and \
         must stay declared, or the completeness gate fails only in that \
         configuration"
    );
    let p = repo_root().join("crates/nsl-codegen/src/test_helpers.rs");
    let text = std::fs::read_to_string(p).unwrap();
    assert!(
        text.contains("#![cfg(any(test, feature = \"test-helpers\"))]"),
        "test_helpers.rs lost its cfg gate — this test's premise is stale"
    );
}

// ─── Step 5: the declared pass-to-pass dependency edges ─────────────────────

/// Every declared consumer pass exists, its `via` says something, and every
/// `ValueOrderedOnly` weakening records the mechanism that makes it safe.
///
/// The same three claims made for `producer`, `empty_means` and
/// `Invariant::Exempt` respectively, for the same reasons: a renamed pass must
/// not leave an edge pointing at nothing, and a weakened ordering claim
/// without its mechanism cannot be told apart from an oversight.
#[test]
fn every_declared_pass_consumer_is_registered_and_explained() {
    use nsl_codegen::pass_bus::OrderClaim;
    let names: BTreeSet<&str> = PASSES.iter().map(|p| p.name).collect();
    for d in CHANNELS {
        for c in d.consumed_by_passes {
            assert!(
                names.contains(c.pass),
                "channel `{}` names consumer pass `{}`, which is not in \
                 pass_registry::PASSES",
                d.name,
                c.pass
            );
            assert!(
                c.via.len() > 20,
                "channel `{}` -> `{}`: `via` records no useful mediation: {:?}",
                d.name,
                c.pass,
                c.via
            );
            if let OrderClaim::ValueOrderedOnly(reason) = c.order {
                assert!(
                    reason.len() > 30,
                    "channel `{}` -> `{}` weakens its ordering claim with no \
                     useful reason: {reason:?}",
                    d.name,
                    c.pass
                );
            }
        }
    }
}

/// The edge list must not decay: at least one edge of EACH OrderClaim variant
/// must exist, or the variant (and everything checking it) is dead code
/// dressed as a distinction — the `status`-field defect again.
///
/// The floors also pin the current graph's minimum: wggo_overrides carries
/// three real edges (FASE, CSHA, WRGA) and adapter_sites one self-edge. A
/// shrink below that is a deliberate edit here, not an accident.
#[test]
fn both_order_claims_are_used_and_the_edge_list_has_not_shrunk() {
    use nsl_codegen::pass_bus::OrderClaim;
    let all: Vec<_> = CHANNELS.iter().flat_map(|d| d.consumed_by_passes.iter()).collect();
    assert!(
        all.len() >= 4,
        "only {} declared pass-consumer edges — the graph shrank; if an edge \
         was genuinely removed, lower this floor deliberately",
        all.len()
    );
    let invocation = all
        .iter()
        .filter(|c| c.order == OrderClaim::InvocationOrdered)
        .count();
    let value_only = all.len() - invocation;
    assert!(
        invocation >= 2,
        "no InvocationOrdered edges left ({invocation}) — \
         dependency_order_violations can never fire and is dead code"
    );
    assert!(
        value_only >= 1,
        "no ValueOrderedOnly edges left — if invocation order now genuinely \
         holds universally, delete the variant rather than keep an unused one"
    );
}

/// Every non-self dependency edge shows type-level coupling: some module of
/// the consuming pass names the channel's carrying module.
///
/// This is the checkable HALF of the edge claim. Nearly every edge is
/// driver-mediated — `stmt.rs` reads the channel and hands the value into the
/// pass entry — so scanning the consuming pass for the ACCESSOR finds
/// nothing, and a gate that required it would force declaring the driver a
/// pass. What the consuming pass's own signature must show is the TYPE:
/// `csha.rs` takes `Option<&crate::wggo_overrides::WggoOverrides>`,
/// `wrga.rs` carries it as `WrgaInput.wggo_overrides`, `fase.rs` returns
/// `crate::wggo_overrides::OverrideDiagnostic`s. An edge with no type
/// coupling at all is either wrong or so indirect the descriptor should say
/// how, and the behavioural half is checked at runtime by
/// `dependency_order_violations` plus the ReadBeforePublish stamps.
///
/// Self-edges are skipped: a pass trivially couples to its own module.
#[test]
fn every_dependency_edge_shows_type_coupling_in_the_consuming_pass() {
    let root = repo_root();
    for d in CHANNELS {
        let non_self: Vec<_> =
            d.consumed_by_passes.iter().filter(|c| c.pass != d.producer).collect();
        if non_self.is_empty() {
            continue;
        }
        // "crate::wggo_overrides::WggoOverrides" -> "crate::wggo_overrides::"
        // (also inside "Vec<crate::wrga_adapter_inject::AdapterSite>"). Only
        // extracted for channels with a non-self edge: a channel like
        // csha_claimed_ops legitimately carries a std type, and demanding a
        // crate path from it would gate a claim nobody made.
        let start = d.carries.find("crate::").unwrap_or_else(|| {
            panic!(
                "channel `{}` declares a non-self pass consumer but carries a \
                 non-crate type ({}) — the type-coupling check cannot work; \
                 restructure the edge or this gate",
                d.name, d.carries
            )
        });
        // Truncate at the first generic/argument delimiter BEFORE taking the
        // last `::` — for a future `HashMap<crate::a::B, crate::c::D>` a bare
        // rfind on the whole string would span both arguments and produce a
        // needle no source file contains, failing with the misdiagnosing
        // "edge is stale" message.
        let tail = &d.carries[start..];
        let tail = &tail[..tail.find([',', '<', '>', ' ']).unwrap_or(tail.len())];
        let end = tail.rfind("::").unwrap() + 2;
        let carrying_module = &tail[..end];
        assert!(
            carrying_module.matches("::").count() >= 2,
            "channel `{}`: carrying-module extraction went wrong: {carrying_module:?}",
            d.name
        );

        for c in non_self {
            let pass = PASSES.iter().find(|p| p.name == c.pass).unwrap();
            let coupled = pass.source_files.iter().any(|f| {
                let text = std::fs::read_to_string(root.join(f))
                    .unwrap_or_else(|e| panic!("reading {f}: {e}"));
                code_only(&text).contains(carrying_module)
            });
            assert!(
                coupled,
                "channel `{}` declares consumer pass `{}`, but none of that \
                 pass's registered source files mentions `{carrying_module}` \
                 — either the edge is stale or the coupling moved to an \
                 unregistered module (register it)",
                d.name, c.pass
            );
        }
    }
}

/// The completeness direction for pass consumers: every consumer FILE that
/// belongs to a registered pass (by the same stem rule
/// `every_codegen_pass_module_belongs_to_a_registered_pass` uses) must appear
/// in `consumed_by_passes`.
///
/// This is what catches a pass module growing a direct bus read without the
/// edge being declared — the drift that would make the declared graph
/// understate the ordering constraints a scheduler must preserve. It cannot
/// see driver-mediated edges (stmt.rs is not a pass module); those are
/// declared from analysis, which is exactly why the floors above pin their
/// count.
#[test]
fn every_pass_module_consumer_file_is_a_declared_pass_consumer() {
    let pass_of_file = |rel: &str| -> Option<&'static str> {
        let stem = Path::new(rel).file_stem()?.to_string_lossy().to_string();
        PASSES
            .iter()
            .find(|p| {
                let low = p.name.to_lowercase();
                stem == low
                    || stem.starts_with(&format!("{low}_"))
                    || p.source_files.contains(&rel)
            })
            .map(|p| p.name)
    };
    let mut checked = 0usize;
    for d in CHANNELS {
        for f in d.consumers {
            let Some(pass) = pass_of_file(f) else { continue };
            checked += 1;
            assert!(
                d.consumed_by_passes.iter().any(|c| c.pass == pass),
                "channel `{}` lists consumer file {f}, which belongs to pass \
                 `{pass}`, but `{pass}` is not in consumed_by_passes — declare \
                 the edge (with its OrderClaim) or the dependency graph \
                 understates what a scheduler must preserve",
                d.name
            );
        }
    }
    assert!(
        checked >= 1,
        "no consumer file mapped to any pass — the stem rule broke, and \
         every assertion above was vacuous"
    );
}

/// Guards this file's own premise: the source it scans must be findable.
#[test]
fn the_scanned_sources_exist() {
    for f in [
        "crates/nsl-codegen/src/pass_bus.rs",
        "crates/nsl-codegen/src/compiler/mod.rs",
    ] {
        assert!(
            Path::new(&repo_root().join(f)).exists(),
            "missing {f} — every scan in this file would vacuously pass"
        );
    }
}
