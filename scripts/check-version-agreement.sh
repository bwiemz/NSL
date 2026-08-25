#!/usr/bin/env bash
#
# check-version-agreement.sh
#
# Verify the single source-of-truth version — the root workspace's
# [workspace.package].version — agrees with every place a version string is
# duplicated by hand: the spec header, the README install snippet, and that the
# CLI crate still *inherits* the workspace version (rather than hardcoding a
# value that could silently drift). Also confirm the CHANGELOG has a landing
# section. Run in CI so a version bump can never leave a stale string behind.
#
# Source of truth : Cargo.toml [workspace.package].version
# Checked mirrors : crates/nsl-cli (must inherit), SPECIFICATION.md, README.md
# Advisory        : CHANGELOG.md must have an [Unreleased] or [<version>] head
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

fail=0
err() { printf '  \xE2\x9C\x97 %s\n' "$1" >&2; fail=1; }
ok() { printf '  \xE2\x9C\x93 %s\n' "$1"; }

# --- source of truth: [workspace.package].version in root Cargo.toml ---
version="$(awk '
  /^\[workspace\.package\]/ { in_wp = 1; next }
  /^\[/                     { in_wp = 0 }
  in_wp && /^version[[:space:]]*=/ {
    if (match($0, /"[0-9]+\.[0-9]+\.[0-9]+[^"]*"/)) {
      print substr($0, RSTART + 1, RLENGTH - 2)
      exit
    }
  }
' Cargo.toml)"

if [[ -z "${version}" ]]; then
  echo "FATAL: could not extract [workspace.package].version from Cargo.toml" >&2
  exit 2
fi
echo "Source-of-truth version ([workspace.package].version): ${version}"
echo

# --- CLI crate must inherit the workspace version (never hardcode it) ---
if grep -qE '^version\.workspace[[:space:]]*=[[:space:]]*true' crates/nsl-cli/Cargo.toml; then
  ok "crates/nsl-cli inherits workspace version (version.workspace = true)"
else
  err "crates/nsl-cli/Cargo.toml must set 'version.workspace = true' so 'nsl --version' cannot drift"
fi

# --- SPECIFICATION.md header must name the current version ---
if grep -qF "v${version}" SPECIFICATION.md; then
  ok "SPECIFICATION.md references v${version}"
else
  err "SPECIFICATION.md does not reference v${version} (stale spec header?)"
fi

# --- README install snippet must reference the current tarball version ---
readme_versions="$(grep -oE 'nsl-v[0-9]+\.[0-9]+\.[0-9]+' README.md | sort -u || true)"
if [[ -z "${readme_versions}" ]]; then
  ok "README.md pins no nsl-v<version> tarball reference (nothing to check)"
elif [[ "${readme_versions}" == "nsl-v${version}" ]]; then
  ok "README.md tarball references all match nsl-v${version}"
else
  err "README.md has nsl-v<version> reference(s) that don't all match nsl-v${version}: ${readme_versions//$'\n'/, }"
fi

# --- CHANGELOG must have a landing section (Unreleased or the current version) ---
version_re="${version//./\\.}"
if grep -qE "^## \[(Unreleased|${version_re})\]" CHANGELOG.md; then
  ok "CHANGELOG.md has an [Unreleased] or [${version}] section"
else
  err "CHANGELOG.md has neither an [Unreleased] nor a [${version}] section heading"
fi

# --- CHANGELOG's newest numbered release must not be AHEAD of Cargo ---
# The landing-section check above is satisfiable while a HIGHER version sits
# tagged in the changelog than the one Cargo declares — exactly what happened
# with a phantom [0.9.1] recorded 2026-03-26 for a release that was never
# tagged (item 19). sort -V puts the larger version last.
newest_numbered="$(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' CHANGELOG.md \
  | head -1 | tr -d '#[] ' || true)"
if [[ -z "${newest_numbered}" ]]; then
  ok "CHANGELOG.md has no numbered release sections (nothing to order-check)"
elif [[ "$(printf '%s\n%s\n' "${newest_numbered}" "${version}" | sort -V | tail -1)" == "${version}" ]]; then
  ok "CHANGELOG.md's newest numbered release (${newest_numbered}) <= ${version}"
else
  err "CHANGELOG.md's newest numbered release [${newest_numbered}] is AHEAD of Cargo's ${version} — a release that never happened, or a missed Cargo bump"
fi

# --- C API version string must derive from CARGO_PKG_VERSION ---
# nsl_model_get_version() shipped a hardcoded "NSL 0.2.0" for seven releases:
# it reaches Python users via NslModel.version, is absent from the generated C
# header (so the header gates never saw it), and its only unit test asserted
# starts_with("NSL"). Structural check: the literal must be built from the
# crate version, which cannot drift. The exact-equality unit test lives in
# c_api/mod.rs (test_version_string).
capi="crates/nsl-runtime/src/c_api/mod.rs"
if grep -qF 'concat!("NSL ", env!("CARGO_PKG_VERSION")' "${capi}"; then
  ok "C API version string derives from CARGO_PKG_VERSION (${capi})"
else
  err "${capi}: nsl_model_get_version must build its string with concat!(\"NSL \", env!(\"CARGO_PKG_VERSION\"), ...) — a hardcoded literal is how 'NSL 0.2.0' survived seven releases"
fi

# --- Python package version fields ---
py_toml="$(grep -oE '^version[[:space:]]*=[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"' python/pyproject.toml | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)"
if [[ "${py_toml}" == "${version}" ]]; then
  ok "python/pyproject.toml version == ${version}"
else
  err "python/pyproject.toml version is '${py_toml:-<missing>}', expected ${version}"
fi
py_init="$(grep -oE '^__version__[[:space:]]*=[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"' python/nslpy/__init__.py | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)"
if [[ "${py_init}" == "${version}" ]]; then
  ok "python/nslpy/__init__.py __version__ == ${version}"
else
  err "python/nslpy/__init__.py __version__ is '${py_init:-<missing>}', expected ${version}"
fi

# --- Wiki roadmap's "Current version" line ---
roadmap_ver="$(grep -oE '\*\*Current version:\*\* `[0-9]+\.[0-9]+\.[0-9]+`' docs/wiki/Roadmap.md | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)"
if [[ "${roadmap_ver}" == "${version}" ]]; then
  ok "docs/wiki/Roadmap.md 'Current version' == ${version}"
else
  err "docs/wiki/Roadmap.md 'Current version' is '${roadmap_ver:-<missing>}', expected ${version} (it claimed 0.9.1 — a version that was never tagged — until item 19)"
fi

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "Version agreement check FAILED — reconcile the drift above with Cargo.toml." >&2
  exit 1
fi
echo "Version agreement check passed (all references == ${version})."
