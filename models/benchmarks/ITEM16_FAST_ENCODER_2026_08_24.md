# Item 16 — GigaToken-adapted fast encoder, same-token parity (2026-08-24)

Roadmap row: "GigaToken preprocessing backend — same-token parity + fast u16
output". GigaToken (github.com/marcelroed/gigatoken, MIT) is a Rust tokenizer
engine ~100-1000x faster than HF `tokenizers`; the roadmap intent was to take
the pieces useful to NSL and adapt them, not to depend on it (it pins nightly
Rust and its encode core is tiktoken-shaped).

## What was taken, and what was not

Taken and adapted into `crates/nsl-runtime/src/tokenizer_fast.rs`:

- **Byte-domain vocabulary** (their `ByteRemapping`): the ByteLevel
  byte→surrogate-char mapping is inverted over the vocab ONCE at load, so
  encoding runs on raw input bytes — the per-pretoken unicode expansion the
  HF path does per document disappears.
- **Heap + linked-list merge loop** (their `bpe_merge_symbols`): lazily
  invalidated min-heap of `(rank << 32 | pos)` — the same merge order as HF
  `Word::merge_all`, plus their small-sequence insight as a linear-scan path
  for lines ≤ 192 symbols (tiktoken-style rank array; the heap only pays off
  on pathological lines).
- **Memoized pretoken encodings** (their pretoken cache, simplified): NSL's
  pretokens are whole LINES, so the packed 32-byte-entry table does not fit;
  a byte-budgeted `FxHashMap<line, ids>` per worker (insert-only, 128-byte
  line cap) captures the code-corpus repetition instead.
- **Document-parallel encoding**: rayon over 64-doc chunks in tokbench,
  order-preserving concat (no encoder state crosses documents — the same
  property that makes shard-then-concatenate exact).

Not taken: the SIMD/SWAR pretokenizers (NSL's encode-time pretokenization is
line splitting, already a trivial byte scan — the cl100k machinery only
shaped *training*), the pyo3 surface, tiktoken/SentencePiece compat, and the
hugepage-backed cache table.

## Same-token parity — the load-bearing property

Layered, all against the committed tokenizer of record
(`models/tokenizers/nsl_mix_v2_t40960_v49152.json`):

1. **Unit** (`tokenizer_fast::tests`): byte-mapping bijection vs the crate's
   own alphabet; a trainer-built tiny tokenizer diffed against the crate;
   config refusals (the loader REFUSES any normalizer / pre-tokenizer /
   model flag combination it does not reproduce, and any id space that does
   not fit u16 — including added tokens whose declared id the crate would
   silently re-assign).
2. **Library differential, every CI build**
   (`fast_encoder_parity_gate.rs`): an edge-case battery (line-split seams,
   whole/partial/adjacent added-token surfaces, CJK/RTL/combining/NBSP,
   cache-interaction shapes) plus ~2 MB of seeded generated corpus — fast
   vs `tokenizers` crate, id-for-id, with a shared cache so cache poisoning
   cannot hide.
3. **Binary differential, cert lane** (`tokbench_reference_encoder_gate`):
   the built tokbench's two backends × two separator modes must produce
   byte-identical streams, and both refuse the over-wide tokenizer.
4. **Full corpus of record**: every set re-encoded from its text shards
   with `--backend fast`, sha256 vs `CORPUS_MANIFEST_v2.json`:

| set | tokens | result |
|---|---|---|
| `nsl_train` | 11,413,038 | **IDENTICAL** |
| `sft_train` | 41,020,908 | **IDENTICAL** (212,849 `<\|im_start\|>` extractions) |
| `web_train` | 8,993,175,632 | **IDENTICAL** |
| `stack_train` | 7,749,015,576 | **IDENTICAL** |

## Throughput and memory

Measured on this box (16 threads, Blackwell workstation), shard-level wall
including process spawn and IO:

| corpus | hf backend | fast backend |
|---|---|---|
| nsl (48 MB code) | 0.6 s (~80 MB/s) | 0.2 s (~240 MB/s) |
| sft (201 MB chat) | 2.2 s (~91 MB/s) | 0.8 s (~250 MB/s) |
| web 45.3 GB | — | 4.0 min (187 MB/s) |
| stack 28.7 GB | — | 3.7 min (130 MB/s) |

A full corpus re-encode is now ~8 minutes; CPU cost is ~3.7x lower than the
hf path (nsl shard: 2.2 s user vs 8.2 s). Peak memory drops from the hf
path's text×2 + every `Encoding` + whole output stream to text once
(borrowed doc slices) + encoded parts + per-worker caches. Remaining
per-core cost is dominated by pair-rank hash lookups; gigatoken's answer
(a ~99%-hit word cache) does not transfer because line-level pretokens on
web text are mostly unique — recorded here so nobody re-derives it.

`tokbench pretokenize --backend {fast,hf}` defaults to **fast** on the
strength of the table above; `hf` remains the reference the differentials
run against. `tools/hfcorpus/pretokenize.py --backend` passes through and
includes the backend in its part-cache key, so switching backends discards
cached parts instead of trusting the parity proof transitively.
