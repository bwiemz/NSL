#!/usr/bin/env bash
#
# install-toolchain.sh — build an `nsl` toolchain from a git ref and install
# it as a pinned channel (NSL V2 plan, Phase 0 item 0.1).
#
# Usage:
#   scripts/install-toolchain.sh <channel> <git-ref> [--prefix DIR]
#                                [--features LIST] [--target-dir DIR]
#
#   <channel>          the channel the build must report, e.g. 0.10-lts
#   <git-ref>          what to build, e.g. release/0.10-lts or a commit
#                      (resolved in this repository; fetch first if remote)
#   --prefix DIR       install root (default: $HOME/.nsl/toolchains); the
#                      toolchain lands in DIR/<channel>/
#   --features LIST    comma- or space-separated cargo features (default:
#                      cuda, since production coder runs are GPU runs). A
#                      bare name is an nsl-cli feature, which nsl-cli forwards
#                      to the runtime; write crate/feature for another crate's
#                      (e.g. nsl-runtime/strict-matmul). '' builds CPU-only.
#   --target-dir DIR   cargo target directory for this build (default:
#                      $HOME/.nsl/build-cache/<channel>). Never the caller's
#                      CARGO_TARGET_DIR: a target dir shared with a checkout
#                      of another commit can link that commit's crates.
#
# A model directory's `nsl-toolchain.toml` pins its runs to a channel;
# `nsl run` / `nsl build` of another channel hand over to
# <prefix>/<channel>/bin/nsl. This script is how that binary gets there.
#
# What it does:
#   1. Resolves <git-ref> to a commit and checks that the commit's source
#      declares <channel> (crates/nsl-cli/src/toolchain.rs) — a fast refusal
#      before a multi-minute build.
#   2. Checks the commit out into a temporary detached worktree (always
#      removed on exit).
#   3. Builds nsl-cli and nsl-runtime there with `--profile dist --locked`,
#      in ONE cargo invocation so the static runtime is built with exactly
#      the features the CLI's own runtime dependency has (cuda, interop).
#   4. Stages the release layout (release.yml): bin/nsl, lib/libnsl_runtime.a,
#      lib/stdlib/, plus toolchain-info.txt (channel, ref, commit, features).
#   5. Refuses to install unless the staged `bin/nsl --version` reports
#      `(toolchain channel <channel>)`, the runtime archive carries the CUDA
#      runtime when cuda was requested, and the staged toolchain can find its
#      stdlib and compile and link a one-line program from an empty directory.
#   6. Moves the stage into <prefix>/<channel>: an existing install is moved
#      aside first and deleted only after the new one is in place. That is
#      two renames, not one: a kill between them leaves no install (and a
#      .old-<channel>.* directory in the prefix), which fails closed — `nsl`
#      then refuses the pinned run with the install hint.
set -euo pipefail

usage() {
  sed -n '3,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
  printf 'install-toolchain: %s\n' "$1" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

channel=""
ref=""
prefix=""
features="cuda"
target_dir=""
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --prefix) [[ $# -ge 2 ]] || die "--prefix needs a directory"; prefix="$2"; shift 2 ;;
    --prefix=*) prefix="${1#--prefix=}"; shift ;;
    --features) [[ $# -ge 2 ]] || die "--features needs a list"; features="$2"; shift 2 ;;
    --features=*) features="${1#--features=}"; shift ;;
    --target-dir) [[ $# -ge 2 ]] || die "--target-dir needs a directory"; target_dir="$2"; shift 2 ;;
    --target-dir=*) target_dir="${1#--target-dir=}"; shift ;;
    --) shift; positional+=("$@"); break ;;
    -*) die "unknown option '$1' (see --help)" ;;
    *) positional+=("$1"); shift ;;
  esac
done
[[ ${#positional[@]} -eq 2 ]] || { usage >&2; exit 2; }
channel="${positional[0]}"
ref="${positional[1]}"

# Same rule as toolchain.rs's validate_channel: the channel is a directory
# name under the prefix, so no '/', no '..', nothing that reads as an option.
[[ "${channel}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "channel '${channel}' must match [A-Za-z0-9][A-Za-z0-9._-]*"

command -v git >/dev/null || die "git not found"
command -v cargo >/dev/null || die "cargo not found"
[[ -n "${HOME:-}" ]] || [[ -n "${prefix}" && -n "${target_dir}" ]] \
  || die "HOME is unset: pass --prefix and --target-dir"
prefix="${prefix:-${HOME}/.nsl/toolchains}"
target_dir="${target_dir:-${HOME}/.nsl/build-cache/${channel}}"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) exe=".exe"; runtime_archive="nsl_runtime.lib" ;;
  *) exe=""; runtime_archive="libnsl_runtime.a" ;;
esac

# --- features: bare names are nsl-cli's ------------------------------------
cargo_features=()
want_cuda=0
IFS=', ' read -r -a requested <<< "${features}"
# `${a[@]+"${a[@]}"}`: an empty array is "unbound" under `set -u` before
# bash 4.4 (macOS ships 3.2), and `--features ''` empties these two.
for f in ${requested[@]+"${requested[@]}"}; do
  [[ -n "${f}" ]] || continue
  [[ "${f}" == */* ]] || f="nsl-cli/${f}"
  cargo_features+=("${f}")
  if [[ "${f}" == */cuda || "${f}" == */nccl ]]; then
    want_cuda=1
  fi
done
feature_args=()
if [[ ${#cargo_features[@]} -gt 0 ]]; then
  joined="$(IFS=,; printf '%s' "${cargo_features[*]}")"
  feature_args=(--features "${joined}")
fi

# --- 1. resolve the ref; does its source declare this channel? -------------
commit="$(git -C "${repo_root}" rev-parse --verify --quiet "${ref}^{commit}")" \
  || die "'${ref}' does not name a commit in ${repo_root} (fetch it first?)"
channel_src="crates/nsl-cli/src/toolchain.rs"
if ! git -C "${repo_root}" cat-file -e "${commit}:${channel_src}" 2>/dev/null; then
  die "refusing: ${ref} (${commit:0:12}) predates toolchain channels (no ${channel_src}); its nsl cannot satisfy a channel pin"
fi
# awk reads to the end (no early `exit`): under pipefail, git dying of SIGPIPE
# would fail the pipeline and blank the answer.
declared="$(git -C "${repo_root}" show "${commit}:${channel_src}" | awk '
  /macro_rules! nsl_channel/ { in_macro = 1 }
  in_macro && !found && match($0, /"[^"]*"/) {
    print substr($0, RSTART + 1, RLENGTH - 2); found = 1
  }
')" || declared=""
if [[ -n "${declared}" && "${declared}" != "${channel}" ]]; then
  die "refusing: ${ref} (${commit:0:12}) is channel '${declared}', not '${channel}' (nsl_channel! in ${channel_src})"
fi

echo "install-toolchain: channel ${channel} from ${ref} (${commit:0:12})"
echo "install-toolchain: features: ${cargo_features[*]:-<none>}; target dir: ${target_dir}"

# --- 2. temporary detached worktree, removed on any exit ---------------------
mkdir -p "${prefix}"
work="$(mktemp -d "${TMPDIR:-/tmp}/nsl-toolchain-${channel}.XXXXXX")"
src="${work}/src"
stage=""
cleanup() {
  if [[ -d "${src}" ]]; then
    git -C "${repo_root}" worktree remove --force "${src}" >/dev/null 2>&1 || true
  fi
  git -C "${repo_root}" worktree prune >/dev/null 2>&1 || true
  rm -rf "${work}"
  if [[ -n "${stage}" ]]; then
    rm -rf "${stage}"
  fi
}
trap cleanup EXIT
git -C "${repo_root}" worktree add --detach --quiet "${src}" "${commit}"

# --- 3. build ------------------------------------------------------------------
# From inside the checkout, so its rust-toolchain.toml selects the compiler.
mkdir -p "${target_dir}"
(
  cd "${src}"
  CARGO_TARGET_DIR="${target_dir}" cargo build --profile dist --locked \
    -p nsl-cli -p nsl-runtime ${feature_args[@]+"${feature_args[@]}"}
)
built="${target_dir}/dist"
[[ -f "${built}/nsl${exe}" ]] || die "build produced no ${built}/nsl${exe}"
[[ -f "${built}/${runtime_archive}" ]] || die "build produced no ${built}/${runtime_archive}"

# --- 4. stage the release layout on the prefix's filesystem ------------------
stage="$(mktemp -d "${prefix}/.stage-${channel}.XXXXXX")"
# mktemp makes it 0700, and it becomes <prefix>/<channel>: give it the mode
# a plain mkdir would have under this umask.
chmod "$(printf '%o' $((0777 & ~0$(umask))))" "${stage}"
mkdir -p "${stage}/bin" "${stage}/lib"
cp "${built}/nsl${exe}" "${stage}/bin/nsl${exe}"
cp "${built}/${runtime_archive}" "${stage}/lib/${runtime_archive}"
cp -R "${src}/stdlib" "${stage}/lib/stdlib"
{
  printf 'channel  = %s\n' "${channel}"
  printf 'ref      = %s\n' "${ref}"
  printf 'commit   = %s\n' "${commit}"
  printf 'features = %s\n' "${cargo_features[*]:-<none>}"
} > "${stage}/toolchain-info.txt"

# --- 5. refuse anything that is not what was asked for -----------------------
version_line="$("${stage}/bin/nsl${exe}" --version)" \
  || die "refusing to install: the built nsl does not run (--version failed)"
if [[ "${version_line}" != *"(toolchain channel ${channel})"* ]]; then
  die "refusing to install: the build reports '${version_line}', not '(toolchain channel ${channel})'"
fi
if [[ "${want_cuda}" -eq 1 ]] \
  && ! LC_ALL=C grep -qaF gpu_fase_fused_adamw_step "${stage}/lib/${runtime_archive}"; then
  die "refusing to install: cuda was requested but ${runtime_archive} has no CUDA runtime"
fi
smoke="${work}/smoke"
mkdir -p "${smoke}"
printf 'print("toolchain smoke")\n' > "${smoke}/smoke.nsl"
# An empty directory and none of the variables that would point the staged
# toolchain at another stdlib or runtime: it must find its own. The pin
# override keeps a pin above the temp dir from handing the smoke build to
# some other installed toolchain (every ref that passed step 1 has the flag).
# (`&&`, not separate lines: `set -e` does not apply inside a tested `if`.)
if ! (
  cd "${smoke}" \
    && env -u NSL_STDLIB_PATH -u NSL_RUNTIME_LIB_PATH_OVERRIDE \
      "${stage}/bin/nsl${exe}" doc stdlib >/dev/null \
    && env -u NSL_STDLIB_PATH -u NSL_RUNTIME_LIB_PATH_OVERRIDE \
      "${stage}/bin/nsl${exe}" build --ignore-toolchain-pin smoke.nsl \
      -o "${smoke}/smoke${exe}"
); then
  die "refusing to install: the staged toolchain could not find its stdlib or build a one-line program"
fi
[[ -f "${smoke}/smoke${exe}" ]] || die "refusing to install: the smoke build produced no executable"

# --- 6. install ------------------------------------------------------------------
dest="${prefix}/${channel}"
old=""
if [[ -e "${dest}" ]]; then
  old="$(mktemp -d "${prefix}/.old-${channel}.XXXXXX")"
  mv "${dest}" "${old}/${channel}"
fi
if ! mv "${stage}" "${dest}"; then
  if [[ -n "${old}" ]]; then
    mv "${old}/${channel}" "${dest}" || true
    rm -rf "${old}"
  fi
  die "could not move the staged toolchain to ${dest}"
fi
stage=""
if [[ -n "${old}" ]]; then
  rm -rf "${old}"
fi

echo "install-toolchain: installed ${dest}"
echo "install-toolchain:   ${dest}/bin/nsl${exe} --version -> ${version_line}"
echo "install-toolchain: build cache kept at ${target_dir} (safe to delete)"
