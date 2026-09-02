//! `nsl env list` / `nsl env current` end to end (roadmap A5).
//!
//! The registry's own agreement with the source tree is gated in
//! `crates/nsl-env/tests/registry_agreement.rs`; this file only proves the
//! CLI surface: every registered name reaches `list`, `--tier` filters and
//! refuses an unknown tier, and `current` distinguishes a registered
//! variable from a set-but-unknown one and fails under `--strict` only for
//! the latter. Every process starts with the inherited `NSL_*` variables
//! removed so a developer shell's knobs cannot leak into the expectations.

use std::process::{Command, Output};

use nsl_env::{Tier, REGISTRY};

fn nsl_env(args: &[&str], set: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("env").args(args);
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("NSL_") {
            cmd.env_remove(key);
        }
    }
    for (k, v) in set {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn nsl env")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn list_prints_every_registered_name() {
    let out = nsl_env(&["list"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    for var in REGISTRY {
        assert!(text.contains(var.name), "`nsl env list` omits {}", var.name);
    }
}

#[test]
fn list_tier_filters_and_json_parses() {
    let tier = Tier::Safety;
    let out = nsl_env(&["list", "--tier", tier.as_str(), "--json"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout(&out)).expect("`nsl env list --json` is valid JSON");
    let rows = parsed.as_array().expect("top-level array");
    let want: Vec<&str> = REGISTRY.iter().filter(|v| v.tier == tier).map(|v| v.name).collect();
    assert!(!want.is_empty(), "the registry has no {} entries", tier.as_str());
    let got: Vec<&str> = rows.iter().map(|r| r["name"].as_str().expect("name")).collect();
    assert_eq!(got, want, "--tier {} must list exactly that tier, in registry order", tier.as_str());
    for row in rows {
        assert_eq!(row["tier"].as_str(), Some(tier.as_str()));
    }
}

#[test]
fn list_refuses_an_unknown_tier() {
    let out = nsl_env(&["list", "--tier", "bogus"], &[]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("unknown tier 'bogus'"), "stderr: {err}");
    for tier in Tier::ALL {
        assert!(err.contains(tier.as_str()), "the refusal must name every tier; missing {}", tier.as_str());
    }
}

#[test]
fn markdown_is_the_whole_registry_and_refuses_a_tier_filter() {
    let out = nsl_env(&["list", "--markdown", "--tier", "safety"], &[]);
    assert!(!out.status.success(), "--markdown --tier must be refused, not silently ignored");
    let out = nsl_env(&["list", "--markdown"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    for var in REGISTRY {
        assert!(text.contains(&format!("`{}`", var.name)), "markdown omits {}", var.name);
    }
    // Placeholders like `<exe>` must not reach a GFM renderer as HTML tags.
    assert!(!text.contains("<exe"), "unescaped angle bracket in a cell");
}

#[test]
fn current_reports_nothing_when_no_nsl_var_is_set() {
    let out = nsl_env(&["current", "--strict"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out).trim(), "no NSL_* variables are set");
}

#[test]
fn current_flags_an_unregistered_name_and_strict_fails_on_it() {
    // Any registered name will do; take the first so the test never
    // hard-codes a variable that a later PR might retire.
    let known = REGISTRY[0];
    let set = [(known.name, "1"), ("NSL_THIS_IS_NOT_A_REAL_KNOB", "yes")];

    let out = nsl_env(&["current"], &set);
    assert!(out.status.success(), "without --strict an unknown name is reported, not fatal");
    let text = stdout(&out);
    assert!(text.contains(&format!("{}=1", known.name)), "stdout: {text}");
    assert!(text.contains(&format!("[{}]", known.tier.as_str())), "stdout: {text}");
    assert!(text.contains("NSL_THIS_IS_NOT_A_REAL_KNOB=yes"), "stdout: {text}");
    assert!(text.contains("[UNREGISTERED]"), "stdout: {text}");

    let out = nsl_env(&["current", "--strict"], &set);
    assert_eq!(out.status.code(), Some(1), "--strict must fail on the unregistered name");
    assert!(stderr(&out).contains("1 set NSL_* variable(s) are not registered"), "stderr: {}", stderr(&out));

    // --strict with only registered names set is clean.
    let out = nsl_env(&["current", "--strict"], &[(known.name, "1")]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
}

#[test]
fn current_json_marks_registration() {
    let known = REGISTRY[0];
    let out = nsl_env(
        &["current", "--json"],
        &[(known.name, "1"), ("NSL_THIS_IS_NOT_A_REAL_KNOB", "yes")],
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout(&out)).expect("`nsl env current --json` is valid JSON");
    let rows = parsed.as_array().expect("top-level array");
    assert_eq!(rows.len(), 2, "{parsed}");
    // Registered names come first, then unknown ones.
    assert_eq!(rows[0]["name"].as_str(), Some(known.name));
    assert_eq!(rows[0]["registered"].as_bool(), Some(true));
    assert_eq!(rows[0]["tier"].as_str(), Some(known.tier.as_str()));
    assert_eq!(rows[1]["name"].as_str(), Some("NSL_THIS_IS_NOT_A_REAL_KNOB"));
    assert_eq!(rows[1]["registered"].as_bool(), Some(false));
    assert_eq!(rows[1]["value"].as_str(), Some("yes"));
}
