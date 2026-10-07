//! Toolchain pinning (NSL V2 plan 0.1), end to end through the real binary.
//!
//! A model directory's `nsl-toolchain.toml` names the toolchain channel its
//! production runs use. `nsl run` / `nsl build` on a file under it must
//! either run on a matching toolchain, hand the invocation over to the
//! installed toolchain for the pinned channel, or refuse, and
//! `--ignore-toolchain-pin` must be the only way past a mismatch. The unit
//! tests in `src/toolchain.rs` cover each decision; these pin what the
//! process actually does: exit status, the lines a user reads, that the
//! program still runs when it should, and (unix) that a handover passes the
//! argv through unchanged.
//!
//! Every test writes its own temp dir. The ones that must not find a real
//! install point HOME (USERPROFILE on Windows) at an empty directory.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

const PROGRAM: &str = "print(\"pinned program ran\")\n";
const PROGRAM_OUTPUT: &str = "pinned program ran";

/// A model directory holding `prog.nsl` and, when `pin` is given, an
/// `nsl-toolchain.toml` with that exact text.
fn model_dir(pin: Option<&str>) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let prog = dir.path().join("prog.nsl");
    std::fs::write(&prog, PROGRAM).unwrap();
    if let Some(text) = pin {
        std::fs::write(dir.path().join("nsl-toolchain.toml"), text).unwrap();
    }
    (dir, prog)
}

fn pin_text(channel: &str) -> String {
    format!("[toolchain]\nchannel = \"{channel}\"\n")
}

/// `nsl <args>` from `cwd`, with the repo stdlib and, when given, HOME (and
/// USERPROFILE) replaced.
fn nsl(args: &[&str], prog: &Path, cwd: &Path, home: Option<&Path>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(args)
        .arg(prog)
        .current_dir(cwd)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"));
    if let Some(home) = home {
        cmd.env("HOME", home).env("USERPROFILE", home);
    }
    cmd.output().expect("spawn nsl")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n")
}

#[test]
fn a_mismatched_pin_with_nothing_installed_is_refused_with_the_fix() {
    let (dir, prog) = model_dir(Some(&pin_text("0.10-lts")));
    let home = tempfile::tempdir().unwrap();
    for sub in ["run", "build"] {
        let out = nsl(&[sub], &prog, dir.path(), Some(home.path()));
        let stderr = text(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "nsl {sub}:\n{stderr}");
        assert!(stderr.contains("nsl-toolchain.toml"), "names the pin file:\n{stderr}");
        let canon_dir = dir.path().canonicalize().unwrap();
        assert!(
            stderr.contains(&canon_dir.display().to_string()),
            "names the pin file's directory:\n{stderr}"
        );
        assert!(stderr.contains("toolchain channel `0.10-lts`"), "names the channel:\n{stderr}");
        assert!(stderr.contains("(toolchain channel dev)"), "names this toolchain:\n{stderr}");
        assert!(stderr.contains("--ignore-toolchain-pin"), "names the override:\n{stderr}");
        assert!(
            stderr.contains("scripts/install-toolchain.sh 0.10-lts <git-ref>"),
            "names the install command:\n{stderr}"
        );
        assert!(
            !text(&out.stdout).contains(PROGRAM_OUTPUT),
            "a refused run must not run the program"
        );
    }
}

#[test]
fn the_override_runs_a_mismatched_pin_on_this_toolchain() {
    let (dir, prog) = model_dir(Some(&pin_text("0.10-lts")));
    let out = nsl(&["run", "--ignore-toolchain-pin"], &prog, dir.path(), None);
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "nsl run --ignore-toolchain-pin:\n{stderr}");
    assert_eq!(text(&out.stdout).trim(), PROGRAM_OUTPUT, "stderr:\n{stderr}");
    assert!(
        stderr.contains("warning:") && stderr.contains("--ignore-toolchain-pin was passed"),
        "the override must say what it overrode:\n{stderr}"
    );
}

#[test]
fn a_pin_naming_this_channel_runs() {
    let (dir, prog) = model_dir(Some(&pin_text("dev")));
    let out = nsl(&["run"], &prog, dir.path(), None);
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "nsl run under a `dev` pin:\n{stderr}");
    assert_eq!(text(&out.stdout).trim(), PROGRAM_OUTPUT, "stderr:\n{stderr}");
    assert!(
        stderr.contains("note: toolchain channel `dev` (pinned by"),
        "a matching pin is reported once, as a note:\n{stderr}"
    );
    // Not `!contains("warning:")`: the system linker prints its own
    // `warning:` lines on some hosts.
    assert!(
        !stderr.contains("pins this model to toolchain channel"),
        "a matching pin is not a mismatch:\n{stderr}"
    );
}

#[test]
fn a_malformed_pin_is_refused_naming_the_file_even_with_the_override() {
    for (bad, why) in [
        ("[toolchain\nchannel = \"dev\"\n", "unterminated table header"),
        ("[toolchain]\nchannel = \"dev\"\nextra = 1\n", "unknown key"),
        ("[toolchain]\nchannel = \"../dev\"\n", "path-like channel"),
    ] {
        let (dir, prog) = model_dir(Some(bad));
        let home = tempfile::tempdir().unwrap();
        for args in [&["run"][..], &["run", "--ignore-toolchain-pin"][..]] {
            let out = nsl(args, &prog, dir.path(), Some(home.path()));
            let stderr = text(&out.stderr);
            assert_eq!(out.status.code(), Some(1), "{why} with {args:?}:\n{stderr}");
            assert!(stderr.contains("nsl-toolchain.toml"), "{why}: names the file:\n{stderr}");
            assert!(!text(&out.stdout).contains(PROGRAM_OUTPUT), "{why}: must not run");
        }
    }
}

#[test]
fn version_names_the_toolchain_channel() {
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("--version")
        .output()
        .expect("spawn nsl --version");
    assert!(out.status.success());
    let stdout = text(&out.stdout);
    assert_eq!(
        stdout.trim(),
        format!("nsl {} (toolchain channel dev)", env!("CARGO_PKG_VERSION")),
    );
}

/// The handover: with the pinned channel installed, `nsl` replaces itself
/// with the installed binary and passes the argv through unchanged. The
/// "installed toolchain" here is a shell script that prints what it got,
/// so the test observes exactly what a real pinned toolchain would receive.
#[cfg(unix)]
#[test]
fn an_installed_pinned_toolchain_receives_the_same_argv() {
    use std::os::unix::fs::PermissionsExt as _;

    let (dir, prog) = model_dir(Some(&pin_text("0.10-lts")));
    let home = tempfile::tempdir().unwrap();
    let bin = home.path().join(".nsl/toolchains/0.10-lts/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let fake = bin.join("nsl");
    std::fs::write(
        &fake,
        "#!/bin/sh\n\
         for a in \"$@\"; do printf 'ARG:%s\\n' \"$a\"; done\n\
         printf 'STDLIB:%s\\n' \"${NSL_STDLIB_PATH-unset}\"\n\
         exit 7\n",
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();

    let prog_arg = prog.display().to_string();
    for args in [
        vec!["run", "--deterministic"],
        vec!["build", "--emit-obj"],
    ] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
        cmd.args(&args).arg(&prog);
        if args[0] == "run" {
            cmd.args(["--", "--program-flag", "two words"]);
        }
        let out = cmd
            .current_dir(dir.path())
            .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
            .env("HOME", home.path())
            .output()
            .expect("spawn nsl");
        let stdout = text(&out.stdout);
        let stderr = text(&out.stderr);

        assert_eq!(
            out.status.code(),
            Some(7),
            "the handover must end with the pinned toolchain's exit status:\n{stdout}\n{stderr}"
        );
        let mut want: Vec<String> = args.iter().map(|a| format!("ARG:{a}")).collect();
        want.push(format!("ARG:{prog_arg}"));
        if args[0] == "run" {
            want.extend(["ARG:--", "ARG:--program-flag", "ARG:two words"].map(String::from));
        }
        let got: Vec<String> = stdout
            .lines()
            .filter(|l| l.starts_with("ARG:"))
            .map(String::from)
            .collect();
        assert_eq!(got, want, "argv must pass through unchanged:\n{stderr}");
        assert!(
            stderr.contains("handing over to") && stderr.contains(&fake.display().to_string()),
            "the handover must name the binary it hands over to:\n{stderr}"
        );
        assert!(
            stdout.contains("STDLIB:unset"),
            "NSL_STDLIB_PATH points at THIS toolchain's stdlib and must not reach the \
             pinned one:\n{stdout}\n{stderr}"
        );
        assert!(stderr.contains("not passing NSL_STDLIB_PATH"), "{stderr}");
    }
}

/// The exec-loop guard, end to end: a toolchain installed under the pinned
/// channel that is really this (dev) binary must be refused, not handed
/// over to (it would hand over to itself forever).
#[cfg(unix)]
#[test]
fn an_install_that_is_this_binary_is_refused_not_looped() {
    let (dir, prog) = model_dir(Some(&pin_text("0.10-lts")));
    let home = tempfile::tempdir().unwrap();
    let bin = home.path().join(".nsl/toolchains/0.10-lts/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_nsl"), bin.join("nsl")).unwrap();

    let out = nsl(&["run"], &prog, dir.path(), Some(home.path()));
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("is this `nsl` itself"), "{stderr}");
    assert!(!stderr.contains("handing over to"), "{stderr}");
}
