#!/usr/bin/env bash
# Sweep tokenizer configurations and report bytes/token on the held-out corpus.
set -euo pipefail

BIN="${BIN:-./target/release/tokbench}"
CORPUS="${CORPUS:-../../data/tokcorpus/train.txt}"
HELDOUT="${HELDOUT:-../../data/tokcorpus/heldout.jsonl}"
OUT="${OUT:-/tmp/claude-1000/-home-brandon-Projects-NSL/0f94ff91-005c-4cc6-81c0-d36c03606d87/scratchpad/sweep}"

mkdir -p "${OUT}"

run() {
  local name="$1" pre="$2" vocab="$3"
  shift 3
  local tok="${OUT}/${name}.json"
  if [[ ! -f "${tok}" ]]; then
    "${BIN}" train --corpus "${CORPUS}" --out "${tok}" \
      --pretokenizer "${pre}" --vocab-size "${vocab}" "$@"
  fi
  "${BIN}" eval --tokenizer "${tok}" --heldout "${HELDOUT}" \
    --name "${name}" --roundtrip --json-out "${OUT}/${name}.report.json"
}

# Pre-tokenizer comparison at the vocab size coder50m declares.
for pre in gpt2 cl100k o200k code; do
  run "pre-${pre}-v49152" "${pre}" 49152 --min-freq 1
done

# Vocabulary scaling with the code-oriented pattern.
for v in 16384 32768 65536 98304; do
  run "code-v${v}" code "${v}" --min-freq 1
done

echo
echo "=== summary (bytes/token, higher is better) ==="
for f in "${OUT}"/*.report.json; do
  python3 - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"{r['name']:<24} vocab={r['vocab_size']:<7} {r['bytes_per_token']:.4f} B/tok  "
      f"tokens={r['total_tokens']:<9} roundtrip={r['roundtrip_exact']}/{r['roundtrip_total']}")
PY
done
