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
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
        .split_whitespace()
        .collect()
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
        CHANNELS.len() >= 8,
        "only {} channels — the list shrank; if a channel was genuinely \
         removed, lower this floor deliberately",
        CHANNELS.len()
    );
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

/// The two findings must render distinguishably. A report where both
/// categories produced the same text would be worse than one category.
#[test]
fn the_two_finding_categories_render_differently() {
    let src = pass_bus_source();
    assert!(src.contains("DEAD OUTPUT:"), "DeadOutput lost its marker");
    assert!(src.contains("SILENT DEFAULT:"), "SilentDefault lost its marker");
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
