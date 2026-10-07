//! Toolchain pinning (NSL V2 plan, Phase 0 item 0.1).
//!
//! The V2 redesign lands on `main` in phases, and production coder runs
//! (500M / 1B / 7B / RL) must not absorb that churn mid-campaign. They run on
//! a long-term-support toolchain instead: `release/0.10-lts`, cut from the
//! last certified commit, taking correctness-only backports. This module is
//! how a model directory says so and how `nsl` honours it.
//!
//! # The pin
//!
//! A model directory carries `nsl-toolchain.toml` ([`PIN_FILE`]):
//!
//! ```toml
//! [toolchain]
//! channel = "0.10-lts"
//! ```
//!
//! `nsl run` and `nsl build` look for it next to the input file and in every
//! directory above it ([`find_pin`]); the nearest one wins. Only those two
//! commands are pinned: they are what a production run executes. `nsl check`,
//! `nsl fmt` and the rest stay on whatever toolchain is invoked.
//!
//! What is pinned is the ENTRY file's real location: the input is
//! canonicalized first, so a symlink in a pinned directory that points
//! outside it is governed by its target's directory, and an unpinned entry
//! file that imports a pinned model's modules is not pinned. A production
//! entry point belongs in its model's directory.
//!
//! # What `nsl` does with it
//!
//! Every build belongs to one channel ([`CHANNEL`]; `"dev"` on `main`). When
//! the pin names a different channel, [`decide`] picks one of:
//!
//! - **hand over**: the pinned channel is installed at
//!   `~/.nsl/toolchains/<channel>/bin/nsl` ([`installed_toolchain`];
//!   `scripts/install-toolchain.sh` puts it there), so this process is
//!   replaced by that binary with the same arguments;
//! - **refuse**: it is not installed, so the run stops with an error that
//!   says how to install it;
//! - **proceed anyway**: `--ignore-toolchain-pin` was passed, so the run
//!   continues on this toolchain with a warning.
//!
//! A pin file that cannot be read or parsed is an error even under
//! `--ignore-toolchain-pin`: the flag overrides a channel mismatch, not a
//! broken file, and a pin that silently stopped pinning is what this exists
//! to prevent.

use std::path::{Path, PathBuf};
use std::process;

use serde::Deserialize;

/// The channel this build belongs to, as a macro so [`VERSION`] can splice it
/// into `concat!`.
///
/// `main` builds are `"dev"`. The `release/0.10-lts` branch changes this one
/// literal to `"0.10-lts"`, and that is the only change the mechanism needs on
/// that branch: a build from it then satisfies a `channel = "0.10-lts"` pin,
/// and `nsl --version` says which channel a binary is.
macro_rules! nsl_channel {
    () => {
        "0.10-lts"
    };
}

/// The channel this build belongs to (see `nsl_channel!` above).
pub const CHANNEL: &str = nsl_channel!();

/// What `nsl --version` prints after the binary name, e.g.
/// `0.10.0 (toolchain channel dev)`. `scripts/install-toolchain.sh` reads the
/// channel back out of this line before it installs a build.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (toolchain channel ",
    nsl_channel!(),
    ")"
);

/// The pin file's name.
pub const PIN_FILE: &str = "nsl-toolchain.toml";

/// The flag that runs a pinned model on the invoked toolchain anyway.
pub const IGNORE_FLAG: &str = "--ignore-toolchain-pin";

/// A pin found on disk: the channel it names and the file that named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolchainPin {
    pub channel: String,
    pub file: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinFileToml {
    toolchain: PinSectionToml,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinSectionToml {
    channel: String,
}

/// Check a channel name. It becomes a directory name under
/// `~/.nsl/toolchains/`, so it is restricted to `[A-Za-z0-9._-]+` and must
/// start with a letter or digit: that refuses `/`, `..`, `.` and anything
/// the install script could mistake for an option.
pub fn validate_channel(channel: &str) -> Result<(), String> {
    let mut bytes = channel.bytes();
    let first_ok = bytes.next().is_some_and(|b| b.is_ascii_alphanumeric());
    let rest_ok = bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if first_ok && rest_ok {
        Ok(())
    } else {
        Err(format!(
            "toolchain channel `{channel}` is not a valid channel name: it must match \
             [A-Za-z0-9][A-Za-z0-9._-]* (it names a directory under ~/.nsl/toolchains/)"
        ))
    }
}

/// Parse the text of a pin file. `file` is only used to name the file in
/// errors and in the returned pin.
pub fn parse_pin(text: &str, file: &Path) -> Result<ToolchainPin, String> {
    let parsed: PinFileToml = toml::from_str(text).map_err(|e| {
        format!(
            "`{}` is not a valid toolchain pin (expected `[toolchain]` with one key, \
             `channel = \"<name>\"`): {}",
            file.display(),
            e.to_string().trim_end()
        )
    })?;
    let channel = parsed.toolchain.channel;
    validate_channel(&channel).map_err(|e| format!("`{}`: {e}", file.display()))?;
    Ok(ToolchainPin {
        channel,
        file: file.to_path_buf(),
    })
}

/// Find the pin that governs `input`: canonicalize it, then look for
/// [`PIN_FILE`] in its directory and each directory above; the first one
/// found wins. `Ok(None)` when there is none. A pin that exists but cannot be
/// read or parsed is an `Err` naming the file, never skipped.
///
/// An input that does not exist (yet) is resolved against the current
/// directory instead of failing here; the command reports the missing file
/// itself, after the pin has been honoured.
pub fn find_pin(input: &Path) -> Result<Option<ToolchainPin>, String> {
    let resolved = match input.canonicalize() {
        Ok(p) => p,
        Err(_) => std::path::absolute(input).map_err(|e| {
            format!(
                "cannot resolve `{}` to look for {PIN_FILE}: {e}",
                input.display()
            )
        })?,
    };
    let Some(start) = resolved.parent() else {
        return Ok(None);
    };
    for dir in start.ancestors() {
        let candidate = dir.join(PIN_FILE);
        match candidate.try_exists() {
            Ok(false) => continue,
            Ok(true) => {
                let text = std::fs::read_to_string(&candidate).map_err(|e| {
                    format!("cannot read toolchain pin `{}`: {e}", candidate.display())
                })?;
                return parse_pin(&text, &candidate).map(Some);
            }
            Err(e) => {
                return Err(format!(
                    "cannot check for a toolchain pin at `{}`: {e}",
                    candidate.display()
                ));
            }
        }
    }
    Ok(None)
}

/// Where installed toolchains live: `$HOME/.nsl/toolchains`
/// (`%USERPROFILE%\.nsl\toolchains` on Windows). `None` when the home
/// variable is unset or empty.
pub fn toolchains_root() -> Option<PathBuf> {
    #[cfg(windows)]
    let home = std::env::var_os("USERPROFILE");
    #[cfg(not(windows))]
    let home = std::env::var_os("HOME");
    let home = home.filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".nsl").join("toolchains"))
}

/// The installed `nsl` for `channel`, if there is one:
/// `<toolchains_root>/<channel>/bin/nsl` (`nsl.exe` on Windows), when that
/// is a file.
pub fn installed_toolchain(channel: &str) -> Option<PathBuf> {
    validate_channel(channel).ok()?;
    let exe = if cfg!(windows) { "nsl.exe" } else { "nsl" };
    let path = toolchains_root()?.join(channel).join("bin").join(exe);
    path.is_file().then_some(path)
}

/// What to do about a pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinDecision {
    /// No pin, or the pin names this build's channel.
    Proceed,
    /// The pin names another channel and `--ignore-toolchain-pin` was
    /// passed: continue on this toolchain after printing the warning.
    ProceedIgnoring(String),
    /// Hand over to the pinned channel's installed `nsl`.
    Exec(PathBuf),
    /// Stop with this error.
    Refuse(String),
}

/// `0.10.0 (toolchain channel <channel>)` — [`VERSION`]'s shape for an
/// arbitrary channel, so [`decide`]'s messages agree with `nsl --version`.
fn version_for(channel: &str) -> String {
    format!("{} (toolchain channel {channel})", env!("CARGO_PKG_VERSION"))
}

/// Whether `candidate` is the running executable. Both sides are
/// canonicalized (a symlinked install, a relative `current_exe`); a path
/// that cannot be canonicalized is compared as given.
fn is_current_exe(candidate: &Path, current_exe: &Path) -> bool {
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    canon(candidate) == canon(current_exe)
}

/// Decide what to do about `pin` (pure apart from canonicalizing paths, for
/// unit testing).
///
/// - no pin, or `pin.channel == running_channel` → [`PinDecision::Proceed`];
/// - mismatch with `ignore_pin` → [`PinDecision::ProceedIgnoring`];
/// - mismatch with an installed toolchain for the pinned channel that is not
///   `current_exe` → [`PinDecision::Exec`];
/// - otherwise → [`PinDecision::Refuse`]. That includes an installed
///   toolchain that IS the running executable: a build of the wrong channel
///   installed under the pinned channel's directory would otherwise hand
///   over to itself forever. When `current_exe` is unknown, that cannot be
///   ruled out, so it refuses too.
pub fn decide(
    pin: Option<&ToolchainPin>,
    running_channel: &str,
    ignore_pin: bool,
    installed: impl Fn(&str) -> Option<PathBuf>,
    current_exe: Option<&Path>,
) -> PinDecision {
    let Some(pin) = pin else {
        return PinDecision::Proceed;
    };
    if pin.channel == running_channel {
        return PinDecision::Proceed;
    }
    let file = pin.file.display();
    let channel = &pin.channel;
    let running = version_for(running_channel);
    if ignore_pin {
        return PinDecision::ProceedIgnoring(format!(
            "`{file}` pins this model to toolchain channel `{channel}`, but {IGNORE_FLAG} \
             was passed: running on this `nsl` ({running}) instead"
        ));
    }
    let Some(target) = installed(channel) else {
        return PinDecision::Refuse(format!(
            "`{file}` pins this model to toolchain channel `{channel}`, but this `nsl` is \
             `{running}`. Install the pinned toolchain with `scripts/install-toolchain.sh \
             {channel} <git-ref>` (it lands in ~/.nsl/toolchains/{channel}/ and `nsl` hands \
             over to it automatically), or pass `{IGNORE_FLAG}` to run on this toolchain \
             anyway."
        ));
    };
    match current_exe {
        Some(current) if !is_current_exe(&target, current) => PinDecision::Exec(target),
        Some(_) => PinDecision::Refuse(format!(
            "`{file}` pins this model to toolchain channel `{channel}`, and the toolchain \
             installed for that channel at `{}` is this `nsl` itself, which is `{running}`: \
             the install is wrong. Reinstall it with `scripts/install-toolchain.sh {channel} \
             <git-ref>` (the script checks the channel of what it builds), or pass \
             `{IGNORE_FLAG}` to run on this toolchain anyway.",
            target.display()
        )),
        None => PinDecision::Refuse(format!(
            "`{file}` pins this model to toolchain channel `{channel}`, installed at `{}`, \
             but this `nsl` ({running}) cannot determine its own path, so it cannot rule \
             out handing over to itself. Run `{}` directly, or pass `{IGNORE_FLAG}` to run \
             on this toolchain anyway.",
            target.display(),
            target.display()
        )),
    }
}

/// Honour the pin governing `input` for `nsl run` / `nsl build`: find it,
/// [`decide`], and act. Returns only when the command should continue on
/// this toolchain; a handover replaces (or, off unix, waits on and exits
/// with) the pinned toolchain, and a refusal exits 1.
pub fn enforce_pin(input: &Path, ignore_pin: bool) {
    let pin = match find_pin(input) {
        Ok(Some(pin)) => pin,
        // Unpinned: nothing to honour, nothing to print.
        Ok(None) => return,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "error: {e}");
            process::exit(1);
        }
    };
    let current_exe = std::env::current_exe().ok();
    match decide(
        Some(&pin),
        CHANNEL,
        ignore_pin,
        installed_toolchain,
        current_exe.as_deref(),
    ) {
        PinDecision::Proceed => {
            nsl_log::nsl_log!(
                INFO,
                "cli",
                "note: toolchain channel `{}` (pinned by `{}`)",
                pin.channel,
                pin.file.display()
            );
        }
        PinDecision::ProceedIgnoring(msg) => {
            nsl_log::nsl_log!(WARN, "cli", "warning: {msg}");
        }
        PinDecision::Exec(target) => hand_over(&target, &pin),
        PinDecision::Refuse(msg) => {
            nsl_log::nsl_log!(ERROR, "cli", "error: {msg}");
            process::exit(1);
        }
    }
}

/// Re-run this invocation's arguments under `target`. Never returns.
fn hand_over(target: &Path, pin: &ToolchainPin) -> ! {
    nsl_log::nsl_log!(
        INFO,
        "cli",
        "note: handing over to {} (channel {}, pinned by {})",
        target.display(),
        pin.channel,
        pin.file.display()
    );
    let mut cmd = process::Command::new(target);
    cmd.args(std::env::args_os().skip(1));
    // These point `nsl` at a stdlib or runtime library OTHER than the
    // toolchain's own. Set for the invoking (usually a dev) toolchain, they
    // would make the pinned toolchain compile against the dev stdlib or link
    // the dev runtime, which is the churn the pin exists to keep out; the
    // pinned toolchain finds its own under its `lib/`. Literal names, so the
    // NSL_* registry gate (nsl-env) sees both reads.
    let toolchain_local = [
        ("NSL_STDLIB_PATH", std::env::var_os("NSL_STDLIB_PATH").is_some()),
        (
            "NSL_RUNTIME_LIB_PATH_OVERRIDE",
            std::env::var_os("NSL_RUNTIME_LIB_PATH_OVERRIDE").is_some(),
        ),
    ];
    for (var, set) in toolchain_local {
        if set {
            nsl_log::nsl_log!(
                WARN,
                "cli",
                "warning: not passing {var} to the pinned toolchain: it would replace that \
                 toolchain's own stdlib or runtime library with this one's"
            );
            cmd.env_remove(var);
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        // `exec` only returns on failure.
        let err = cmd.exec();
        nsl_log::nsl_log!(
            ERROR,
            "cli",
            "error: could not hand over to {}: {err}",
            target.display()
        );
        process::exit(1);
    }

    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(status) => process::exit(status.code().unwrap_or(1)),
            Err(err) => {
                nsl_log::nsl_log!(
                    ERROR,
                    "cli",
                    "error: could not hand over to {}: {err}",
                    target.display()
                );
                process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(channel: &str) -> ToolchainPin {
        ToolchainPin {
            channel: channel.to_string(),
            file: PathBuf::from("/models/m/nsl-toolchain.toml"),
        }
    }

    fn write_pin(dir: &Path, channel: &str) -> PathBuf {
        let file = dir.join(PIN_FILE);
        std::fs::write(&file, format!("[toolchain]\nchannel = \"{channel}\"\n")).unwrap();
        file
    }

    // ---- the version string ----------------------------------------------

    #[test]
    fn version_names_the_package_version_and_the_channel() {
        assert_eq!(
            VERSION,
            format!("{} (toolchain channel {CHANNEL})", env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(VERSION, version_for(CHANNEL));
        assert!(validate_channel(CHANNEL).is_ok());
    }

    // ---- parsing -----------------------------------------------------------

    #[test]
    fn a_well_formed_pin_parses() {
        let file = Path::new("/x/nsl-toolchain.toml");
        let text = "# comment\n[toolchain]\nchannel = \"0.10-lts\"\n";
        assert_eq!(
            parse_pin(text, file).unwrap(),
            ToolchainPin {
                channel: "0.10-lts".to_string(),
                file: file.to_path_buf()
            }
        );
    }

    #[test]
    fn an_unknown_key_is_refused_at_either_level() {
        let file = Path::new("/x/nsl-toolchain.toml");
        for text in [
            "[toolchain]\nchannel = \"dev\"\nversion = \"0.10\"\n",
            "[toolchain]\nchannel = \"dev\"\n[other]\nk = 1\n",
            "extra = 1\n[toolchain]\nchannel = \"dev\"\n",
        ] {
            let err = parse_pin(text, file).unwrap_err();
            assert!(err.contains("/x/nsl-toolchain.toml"), "{text:?}: {err}");
            assert!(err.contains("unknown field"), "{text:?}: {err}");
        }
    }

    #[test]
    fn a_missing_section_or_channel_is_refused() {
        let file = Path::new("/x/nsl-toolchain.toml");
        for text in ["", "channel = \"dev\"\n", "[toolchain]\n", "[toolchain]\nchannel = 3\n"] {
            let err = parse_pin(text, file).unwrap_err();
            assert!(err.contains("/x/nsl-toolchain.toml"), "{text:?}: {err}");
            assert!(err.contains("not a valid toolchain pin"), "{text:?}: {err}");
        }
    }

    #[test]
    fn a_channel_that_is_not_a_plain_path_component_is_refused() {
        let file = Path::new("/x/nsl-toolchain.toml");
        // TOML literal strings ('...'), so a backslash reaches the check
        // instead of failing as an escape sequence.
        for bad in ["", "..", ".", "a/b", "../etc", "-rf", ".hidden", "0.10 lts", "lts\\x", "é"] {
            let text = format!("[toolchain]\nchannel = '{bad}'\n");
            let err = parse_pin(&text, file).unwrap_err();
            assert!(err.contains("not a valid channel name"), "{bad:?}: {err}");
            assert!(err.contains("/x/nsl-toolchain.toml"), "{bad:?}: {err}");
        }
        for good in ["dev", "0.10-lts", "v1_2.3", "A"] {
            assert!(validate_channel(good).is_ok(), "{good:?}");
        }
    }

    // ---- finding -----------------------------------------------------------

    #[test]
    fn a_pin_in_a_grandparent_governs_the_file() {
        let root = tempfile::tempdir().unwrap();
        let file = write_pin(root.path(), "0.10-lts");
        let deep = root.path().join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        let input = deep.join("m.nsl");
        std::fs::write(&input, "").unwrap();
        let found = find_pin(&input).unwrap().unwrap();
        assert_eq!(found.channel, "0.10-lts");
        assert_eq!(found.file, file.canonicalize().unwrap());
    }

    #[test]
    fn the_nearest_pin_wins() {
        let root = tempfile::tempdir().unwrap();
        write_pin(root.path(), "outer");
        let inner = root.path().join("model");
        std::fs::create_dir_all(&inner).unwrap();
        write_pin(&inner, "inner");
        let input = inner.join("m.nsl");
        std::fs::write(&input, "").unwrap();
        assert_eq!(find_pin(&input).unwrap().unwrap().channel, "inner");
    }

    #[test]
    fn a_file_with_no_pin_above_it_is_unpinned() {
        // The walk goes to `/`, so this holds only while no ancestor of the
        // temp dir carries a pin. Checked first, so a stray pin there (or a
        // TMPDIR inside a pinned model directory) fails as what it is.
        let root = tempfile::tempdir().unwrap();
        let canon = root.path().canonicalize().unwrap();
        for dir in canon.ancestors() {
            assert!(
                !dir.join(PIN_FILE).exists(),
                "test precondition: {} has a {PIN_FILE}; point TMPDIR elsewhere",
                dir.display()
            );
        }
        let input = root.path().join("m.nsl");
        std::fs::write(&input, "").unwrap();
        assert_eq!(find_pin(&input).unwrap(), None);
    }

    #[test]
    fn a_missing_input_still_finds_its_directorys_pin() {
        let root = tempfile::tempdir().unwrap();
        write_pin(root.path(), "0.10-lts");
        let input = root.path().join("not_written_yet.nsl");
        assert_eq!(find_pin(&input).unwrap().unwrap().channel, "0.10-lts");
    }

    #[test]
    fn a_malformed_or_unreadable_pin_is_an_error_naming_it() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("m.nsl");
        std::fs::write(&input, "").unwrap();

        std::fs::write(root.path().join(PIN_FILE), "[toolchain\n").unwrap();
        let err = find_pin(&input).unwrap_err();
        assert!(err.contains(PIN_FILE), "{err}");

        // A directory in the pin's place cannot be read as one.
        std::fs::remove_file(root.path().join(PIN_FILE)).unwrap();
        std::fs::create_dir(root.path().join(PIN_FILE)).unwrap();
        let err = find_pin(&input).unwrap_err();
        assert!(err.contains("cannot read toolchain pin"), "{err}");
        assert!(err.contains(PIN_FILE), "{err}");
    }

    // ---- deciding ----------------------------------------------------------

    fn none_installed(_: &str) -> Option<PathBuf> {
        None
    }

    #[test]
    fn no_pin_proceeds() {
        let d = decide(None, "dev", false, none_installed, Some(Path::new("/bin/nsl")));
        assert_eq!(d, PinDecision::Proceed);
    }

    #[test]
    fn a_matching_pin_proceeds_without_looking_for_an_install() {
        let d = decide(
            Some(&pin("dev")),
            "dev",
            false,
            |_| panic!("a matching pin must not look up an installed toolchain"),
            None,
        );
        assert_eq!(d, PinDecision::Proceed);
    }

    #[test]
    fn a_mismatch_with_the_override_proceeds_with_a_warning() {
        let d = decide(Some(&pin("0.10-lts")), "dev", true, none_installed, None);
        let PinDecision::ProceedIgnoring(msg) = d else {
            panic!("expected ProceedIgnoring, got {d:?}");
        };
        assert!(msg.contains("/models/m/nsl-toolchain.toml"), "{msg}");
        assert!(msg.contains("`0.10-lts`"), "{msg}");
        assert!(msg.contains(IGNORE_FLAG), "{msg}");
        assert!(msg.contains("(toolchain channel dev)"), "{msg}");
    }

    #[test]
    fn the_override_wins_over_an_installed_toolchain() {
        let d = decide(
            Some(&pin("0.10-lts")),
            "dev",
            true,
            |_| Some(PathBuf::from("/elsewhere/nsl")),
            Some(Path::new("/bin/nsl")),
        );
        assert!(matches!(d, PinDecision::ProceedIgnoring(_)), "{d:?}");
    }

    #[test]
    fn a_mismatch_with_the_pinned_channel_installed_hands_over() {
        let dir = tempfile::tempdir().unwrap();
        let installed = dir.path().join("lts-nsl");
        let current = dir.path().join("dev-nsl");
        std::fs::write(&installed, "").unwrap();
        std::fs::write(&current, "").unwrap();
        let d = decide(
            Some(&pin("0.10-lts")),
            "dev",
            false,
            |c| {
                assert_eq!(c, "0.10-lts");
                Some(installed.clone())
            },
            Some(&current),
        );
        assert_eq!(d, PinDecision::Exec(installed.clone()));
    }

    #[test]
    fn a_mismatch_with_nothing_installed_refuses_with_the_fix() {
        let d = decide(Some(&pin("0.10-lts")), "dev", false, none_installed, None);
        let PinDecision::Refuse(msg) = d else {
            panic!("expected Refuse, got {d:?}");
        };
        assert!(msg.contains("/models/m/nsl-toolchain.toml"), "{msg}");
        assert!(msg.contains("toolchain channel `0.10-lts`"), "{msg}");
        assert!(msg.contains("(toolchain channel dev)"), "{msg}");
        assert!(msg.contains("scripts/install-toolchain.sh 0.10-lts <git-ref>"), "{msg}");
        assert!(msg.contains("~/.nsl/toolchains/0.10-lts/"), "{msg}");
        assert!(msg.contains(IGNORE_FLAG), "{msg}");
    }

    /// The exec-loop guard: a wrong-channel build installed under the pinned
    /// channel's directory is the running binary after the first handover,
    /// and handing over again would never end. Compared canonicalized, so a
    /// symlink to the running binary is caught too.
    #[test]
    fn an_installed_toolchain_that_is_this_executable_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("nsl");
        std::fs::write(&real, "").unwrap();
        let mut candidates = vec![real.clone()];
        #[cfg(unix)]
        {
            let link = dir.path().join("linked-nsl");
            std::os::unix::fs::symlink(&real, &link).unwrap();
            candidates.push(link);
        }
        for installed in candidates {
            let d = decide(
                Some(&pin("0.10-lts")),
                "dev",
                false,
                |_| Some(installed.clone()),
                Some(&real),
            );
            let PinDecision::Refuse(msg) = d else {
                panic!("{}: expected Refuse, got {d:?}", installed.display());
            };
            assert!(msg.contains("is this `nsl` itself"), "{msg}");
            assert!(msg.contains("the install is wrong"), "{msg}");
            assert!(msg.contains(IGNORE_FLAG), "{msg}");
        }
    }

    #[test]
    fn an_unknown_current_executable_refuses_rather_than_risk_a_loop() {
        let d = decide(
            Some(&pin("0.10-lts")),
            "dev",
            false,
            |_| Some(PathBuf::from("/home/u/.nsl/toolchains/0.10-lts/bin/nsl")),
            None,
        );
        let PinDecision::Refuse(msg) = d else {
            panic!("expected Refuse, got {d:?}");
        };
        assert!(msg.contains("cannot determine its own path"), "{msg}");
        assert!(msg.contains(IGNORE_FLAG), "{msg}");
    }

    #[test]
    fn an_invalid_channel_is_never_looked_up_on_disk() {
        // `installed_toolchain` re-validates, so a channel built by hand (not
        // through `parse_pin`) still cannot walk out of the toolchains root.
        assert_eq!(installed_toolchain(".."), None);
        assert_eq!(installed_toolchain("../../bin"), None);
    }
}
