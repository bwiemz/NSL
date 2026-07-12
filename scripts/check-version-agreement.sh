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

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "Version agreement check FAILED — reconcile the drift above with Cargo.toml." >&2
  exit 1
fi
echo "Version agreement check passed (all references == ${version})."
