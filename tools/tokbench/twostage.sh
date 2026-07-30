#!/usr/bin/env bash
# Sweep the point at which the pre-tokenizer boundary is relaxed.
#
# transition == vocab_size  -> ordinary word-bounded BPE
# transition == 0           -> line-level BPE from scratch
# in between                -> subword inventory first, cross-boundary units after
set -euo pipefail

BIN="${BIN:-./target/release/tokbench}"
D="${D:-../../data/tokcorpus}"
OUT="${OUT:-/tmp/claude-1000/-home-brandon-Projects-NSL/0f94ff91-005c-4cc6-81c0-d36c03606d87/scratchpad/two}"
VOCAB="${VOCAB:-49152}"
STAGE1="${STAGE1:-cl100k}"
RELAXED="${RELAXED:-line}"
MAXTOK="${MAXTOK:-0}"

mkdir -p "${OUT}"

bpt() { "${BIN}" eval --tokenizer "$1" --heldout "$2" --name x 2>/dev/null | awk '/BYTES\/TOKEN/{print $2}'; }

printf "%-10s %8s %8s %8s %9s %8s\n" transition nsl other ood "enc MiB/s" overfit
for t in "$@"; do
  tok="${OUT}/t${t}-v${VOCAB}-${STAGE1}-${RELAXED}-m${MAXTOK}.json"
  if [[ ! -f "${tok}" ]]; then
    "${BIN}" train2 --corpus "${D}/combined_train.txt" --out "${tok}" \
      --stage1 "${STAGE1}" --relaxed "${RELAXED}" --vocab-size "${VOCAB}" \
      --transition "${t}" --min-freq 2 --max-token-bytes "${MAXTOK}" >/dev/null 2>&1
  fi
  n=$(bpt "${tok}" "${D}/heldout.jsonl")
  o=$(bpt "${tok}" "${D}/other_heldout.jsonl")
  d=$(bpt "${tok}" "${D}/ood.jsonl")
  e=$("${BIN}" eval --tokenizer "${tok}" --heldout "${D}/ood.jsonl" --name x 2>/dev/null | awk '/encode /{print $2}')
  r=$(python3 -c "print(f'{${n}/${d}:.2f}x')")
  printf "%-10s %8s %8s %8s %9s %8s\n" "${t}" "${n}" "${o}" "${d}" "${e}" "${r}"
done
