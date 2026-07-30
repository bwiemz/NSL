#!/usr/bin/env bash
# Train each configuration on the combined corpus and score it on three sets:
#   nsl   - held-out files from the NSL repo            (in-domain)
#   other - held-out files from projects seen in training (in-domain, other langs)
#   ood   - every file from five projects never trained on (true transfer)
set -euo pipefail

BIN="${BIN:-./target/release/tokbench}"
DATA="${DATA:-../../data/tokcorpus}"
OUT="${OUT:-/tmp/claude-1000/-home-brandon-Projects-NSL/0f94ff91-005c-4cc6-81c0-d36c03606d87/scratchpad/combined}"
VOCAB="${VOCAB:-49152}"

mkdir -p "${OUT}"

bpt() {
  "${BIN}" eval --tokenizer "$1" --heldout "$2" --name x 2>/dev/null \
    | awk '/BYTES\/TOKEN/ {print $2}'
}
enc() {
  "${BIN}" eval --tokenizer "$1" --heldout "$2" --name x 2>/dev/null \
    | awk '/encode / {print $2}'
}

printf "%-14s %8s %8s %8s %10s %10s\n" config nsl other ood "enc MiB/s" overfit

for pre in "$@"; do
  tok="${OUT}/${pre}-v${VOCAB}.json"
  if [[ ! -f "${tok}" ]]; then
    "${BIN}" train --corpus "${DATA}/combined_train.txt" --out "${tok}" \
      --pretokenizer "${pre}" --vocab-size "${VOCAB}" --min-freq 1 --whole-docs >/dev/null
  fi
  n=$(bpt "${tok}" "${DATA}/heldout.jsonl")
  o=$(bpt "${tok}" "${DATA}/other_heldout.jsonl")
  d=$(bpt "${tok}" "${DATA}/ood.jsonl")
  e=$(enc "${tok}" "${DATA}/ood.jsonl")
  r=$(python3 -c "print(f'{${n}/${d}:.2f}x')")
  printf "%-14s %8s %8s %8s %10s %10s\n" "${pre}" "${n}" "${o}" "${d}" "${e}" "${r}"
done
