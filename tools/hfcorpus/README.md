# `tools/hfcorpus` — Hugging Face pretraining corpus pipeline

Builds the pretraining and SFT corpora for `coder50m` / `coder500m` / `coder1b`
from two Hugging Face datasets plus this repository's own source corpus, in the
flat little-endian u16 token format `load_mmap(path, 3)` already mmaps.

```
fetch.py         HF  -> data/hf/**.parquet
extract.py       parquet -> data/text/<set>/*.txt   (NUL-separated documents)
train_tokenizer.py  sample -> data/tokenizer/nsl_mix_v2_*.json
pretokenize.py   text shards -> data/tokens/mix/*.bin  (+ .manifest.json)
```

Everything under `data/` is gitignored. Run from the repo root with the pinned
interpreter: `tools/hfcorpus/.venv/bin/python`.

## Sources

| set | source | size |
|---|---|---|
| code | `HuggingFaceCode/stack-v3-train`, 20 of 8,192 shards | 8.43 GB parquet, 28.67 GB text, 7.75B v2-tokens |
| web | `HuggingFaceFW/fineweb`, config `sample-10BT` | 15 parquet, 30.65 GB, 14.9M docs, ~8.05B v2-tokens |
| chat | `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`, config `canonical` | 57,937 traces, 0.22 GB rendered |
| nsl | `data/tokcorpus/combined_train.txt` (this repo + 15 unrelated local projects) | 48.23 MB, 11.6M tokens |

Stack v3's train subset is 3.54 TB / ~3.6T tokens over 8,192 shards — three
orders of magnitude past anything that fits here. `fetch.py --stack-shards N`
takes a **strided** sample rather than a prefix: the shards come out of a Spark
job whose partitioning order is not documented, so a prefix could be correlated
with repo size, language, or crawl order.

Each Stack row is a **repository** with contents inline under `files[].content`
(unlike The Stack v2, which shipped only blob ids and needed a separate S3
fetch). A row renders to one document of `<|file_sep|>path\ncontent` per file,
which is the point of the repo grouping — the model sees a project rather than
shuffled files. Repositories past `--max-doc-bytes` split across several
documents instead of being dropped, so one monorepo cannot dominate a shard.

`sample-10BT` rather than `sample-100BT`: at the measured 4.6K tok/s for 500M on
the RTX 4500, even 10B tokens is ~25 GPU-days, so the larger sample would buy
storage pressure and nothing else.

The `nsl` corpus is 11.6M tokens against ~8B of web and ~11B of code. It cannot
carry a meaningful share of the mixture without extreme repetition — 1% of a
20B-token corpus would be 69 epochs of it. It sits at ~0.25% (about 4 epochs,
inside the range where repeated data behaves like fresh data) and is better
spent again in a finetune stage than upsampled here.

The chat set is kept OUT of the base pretraining mix. It is chat-structured with
tool calls and reasoning traces, and at ~0.15B tokens it is ~1.5% of the corpus —
diluted to nothing during pretraining, and more useful as a separate SFT or
anneal stage.

## Traps this pipeline is built around

**The distillation repo ships two overlapping shard series in one directory.**
`data/canonical/` holds both `train-*-of-00005` (46,250 rows) and
`train-*-of-00006` (52,205 rows). They are not a reshard of the same content:
4,568 trace ids appear only in the older series. The README declares config
names with no `data_files`, so `datasets`' default `train-*` glob concatenates
both and duplicates 41,682 traces. `extract.py` names the `-of-00006` series
explicitly; 52,205 + 2,872 + 2,860 = 57,937 matches the dataset card exactly.

**A document can forge its own boundary.** A `tokenizers` `AddedToken` is matched
by surface at encode time regardless of `add_special_tokens`, so a crawled page
whose body contains the literal `<|endoftext|>` would inject a genuine
end-of-document id into the middle of the stream. `extract.py` neutralizes every
reserved surface in document text (U+2063 after `<|`, which breaks the match and
leaves the text readable), and `pretokenize.py` injects the separator
structurally by id rather than splicing a literal into the text. The reserved
list in `extract.py` and `train_tokenizer.py` must stay identical — a surface
reserved but not neutralized is a forgeable boundary.

**The v1 tokenizer was written to a wiped `/tmp` scratchpad and lost.**
`data/tokens/train_new.bin` was the only surviving artifact of it. It was
recovered by retraining the spec from `data/tokcorpus/combined_train.txt` and
verified byte-identical against `train_new.bin` (8,925,916 tokens, 5.403 B/tok),
which also proves `train_two_stage` is deterministic. Both tokenizers now live in
`models/tokenizers/` rather than in a scratch directory.

**Sharding is exact, not approximate.** `tokbench pretokenize` carries no encoder
state across documents, so encoding N shards and concatenating is byte-identical
to encoding one corpus file. This is load-bearing: tokbench reads the whole
corpus into memory and builds the entire token stream before writing, so a single
42 GB corpus file is not an option.

**The web split is cut at file granularity.** The last of the 15 parquet files is
held out whole. A row-offset cut inside a shared file would be index-disjoint but
not separated — adjacent FineWeb rows come from the same crawl segment and often
the same site — and would depend on a row count that moves if the set is
re-exported.

## Tokenizers

`v1` (`nsl_code_v1_t40960_v49152.json`) is the tokenizer of record for
`data/tokens/train_new.bin` and everything pinned against it, including
`pretrain_prod_agreement_gate` and the 5.403 B/tok provenance fingerprint. It is
unchanged and its corpora are untouched.

`v2` (`nsl_mix_v2_t40960_v49152.json`) is trained for the new corpus. It changes
exactly two things — the training mixture and seven reserved special tokens
(`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|tool_call|>`,
`<|tool_result|>`, `<|pad|>`, `<|file_sep|>` — the last is the corpus's most
frequent marker, one per rendered repository file; v1 reserved none, so it had
no representable document boundary). Two-stage cl100k→line training, vocab 49152 and min_freq 2
are held fixed so the two are comparable, and 49152 keeps the id space inside
u16.

Reserved surfaces and ids live in `models/tokenizers/special_tokens.json`
(committed; the training run rewrites it in place). Read them from there; do
not assume they are contiguous or last.

## Resuming, and what invalidates a cache

`pretokenize.py` keeps per-shard `.bin` parts under `data/tokens/mix/.<name>.parts/`
so a 91-shard run survives interruption. Two records make that safe:

* `cache_key.json` — the tokenizer's sha256 and the separator. If either
  changes, the parts are discarded. Parts with **no** key are also discarded:
  unknown provenance is exactly the case the key exists for, and accepting them
  because there is nothing to compare against would make the guard unreachable
  for its own motivating case.
* `roster.json` — the shard list, recorded on first use. Deriving it from
  `glob()` each run is what makes `--reap` and resume incompatible: a reaped
  shard vanishes from the glob, so a resumed run would concatenate only the
  shards not yet encoded and emit a manifest, a sha256, and an entirely
  plausible bytes/token for a silently truncated corpus.

Pass `--clean-parts` once a corpus is final; it removes both records with the
parts.

## Mixing

`load_mmap(path, 3)` takes a single file, so a mixture is materialized rather
than expressed as a loader weight. `make_mix.py` works on token ids, not text,
so changing the ratio costs one pass over the `.bin` files instead of a full
retokenization. Documents are located by the end-of-document id and emitted
whole, and injections land at seeded random positions among the base documents —
appending the minority corpus would give the model none of it until the end of
the epoch, and cutting at token offsets would splice unrelated documents into
single windows.

Repetition is reported rather than hidden: every `--add` entry's manifest
records how many epochs of that corpus its share works out to.

Those epoch counts describe the FILE, not what a run reads. The shipped mixture
is ~22.6B tokens and asks Stack for 1.75 epochs of its 7.75B unique tokens, but
a 2B-token run at 50M reads under 9% of the corpus — and because documents are
interleaved at seeded random positions rather than blocked, the odds of both
copies of a repository landing inside that prefix are small. Effective
repetition for the runs actually planned is close to 1.0. Judge a corpus by the
prefix a run consumes, not by the whole file.

## Wiring a recipe to these corpora

The recipes take a corpus by path and pin its size, so switching one is three
edits in `models/coder<size>/config.nsl` plus the matching literal in
`pretrain_prod.nsl`:

```
const PRETRAIN_DATA = "../../data/tokens/mix/pretrain_train.bin"
const PRETRAIN_VAL  = "../../data/tokens/mix/web_val.bin"
const CORPUS_TOKENS = <"tokens" from the corpus's .manifest.json>
```

`crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs` re-derives the schedule
from `CORPUS_TOKENS` and checks every declared `load_mmap` path is used exactly
once, so a stale count or a val loader pointed at the train slice is caught
there rather than 30 minutes into a run.

**`VOCAB_SIZE = 49152` does not need to change.** v2's largest id is 49149, so
the existing embedding covers it with two unused rows, and the
`@fused_lm_ce(vocab_size=49152, ...)` hint stays valid. Shrinking the constant to
49150 would save two rows and invalidate every pinned literal; it is not worth
it.

**The recipes' existing corpora are untouched.** `train_new.bin`,
`prod_train_slice.bin`, `prod_val_slice.bin` and the `sr_*` slices are still v1
and still match their gates. Point a recipe at `data/tokens/mix/` only when you
intend to switch it.

**Slices.** `models/benchmarks/make_prod_split.py` cuts a train prefix and a
held-out tail out of a single corpus with a deliberate gap. These corpora
already ship a separate `web_val.bin` cut at parquet-file granularity, so a
per-model run wants a prefix of `pretrain_train.bin` sized to its token budget
and `web_val.bin` as-is — not another tail cut.


## Corpus file names

| file | what it is |
|---|---|
| `stack_train.bin` | Stack v3 source code, repo-grouped |
| `web_train.bin` / `web_val.bin` | FineWeb, split at parquet-file granularity |
| `nsl_train.bin` | this repo's own sources (`combined_train.txt`) |
| `sft_train.bin` / `sft_val.bin` | distillation chat traces, separate SFT stage |
| `pretrain_train.bin` | the mixture `make_mix.py` builds from the above |

`nsl_train.bin` and `stack_train.bin` are both code and are deliberately named
apart: they differ by three orders of magnitude in size and by their role, and a
recipe that pointed at the wrong one would still run.
