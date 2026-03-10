# Section 7 — Tokenization

## Design Rationale

Tokenization is the interface between text and tensors — a critical component that's usually
handled by external libraries (HuggingFace `tokenizers`, SentencePiece) with no integration
into the type system. NSL makes tokenizers a standard library module with deep integration:
tokenizer output types flow directly into tensor types, vocabulary sizes are checked at compile
time against embedding dimensions, and streaming tokenization for large corpora is native.

## Tokenize Block Grammar

```ebnf
tokenizer_def   ::= 'tokenizer' IDENT '(' tokenizer_config ')' ':' INDENT tokenizer_body DEDENT
tokenizer_config::= config_entry (',' config_entry)*
config_entry    ::= 'algorithm' '=' algorithm
                   | 'vocab_size' '=' INT
                   | 'model_file' '=' STRING

algorithm       ::= 'bpe' | 'wordpiece' | 'sentencepiece' | 'unigram' | 'char' | 'byte'

tokenizer_body  ::= (tokenizer_stmt NEWLINE)*
tokenizer_stmt  ::= special_tokens_def | normalization_def | pre_tokenize_def
                   | padding_def | truncation_def

special_tokens_def ::= 'special_tokens' ':' INDENT (token_def NEWLINE)+ DEDENT
token_def       ::= IDENT '=' STRING                # e.g., pad = "<pad>"

normalization_def ::= 'normalize' ':' norm_rule (',' norm_rule)*
norm_rule       ::= 'nfc' | 'nfkc' | 'lowercase' | 'strip_accents' | 'strip' | IDENT

pre_tokenize_def ::= 'pre_tokenize' ':' pre_rule (',' pre_rule)*
pre_rule        ::= 'whitespace' | 'punctuation' | 'digits' | 'byte_fallback' | IDENT

padding_def     ::= 'padding' ':' pad_config
pad_config      ::= 'side' '=' ('left' | 'right') (',' 'pad_to' '=' (INT | 'longest'))?

truncation_def  ::= 'truncation' ':' trunc_config
trunc_config    ::= 'max_length' '=' INT (',' 'strategy' '=' ('longest_first' | 'only_first' | 'only_second'))?

# Vocabulary type
vocab_type      ::= 'Vocab' '<' INT '>'              # e.g., Vocab<50257>
```

## Vocabulary as a Compile-Time Type

```nsl
# The vocabulary size is tracked in the type system.
# This means the compiler can verify that embedding dimensions match.

tokenizer my_tokenizer(algorithm=bpe, vocab_size=50257):
    # ...

model MyModel:
    # Compiler checks: Embedding vocab_size (50257) matches tokenizer vocab_size ✓
    embed: Embedding = Embedding(50257, 768)

    fn forward(tokens: Tensor<[batch, seq], int32>) -> Tensor<[batch, seq, 768], fp32>:
        return self.embed(tokens)

# If you change the tokenizer to vocab_size=32000 but forget to update the embedding:
# COMPILE ERROR: tokenizer 'my_tokenizer' has vocab_size=32000
#                but Embedding expects vocab_size=50257
```

## 4 Tokenization Examples

### Example 1: BPE Tokenizer from Scratch

```nsl
## Train a BPE tokenizer from scratch on a text corpus.
## BPE (Byte Pair Encoding) iteratively merges the most frequent byte pairs.

import nsl.tokenize.{BPETrainer, Tokenizer}

# Define the tokenizer configuration
tokenizer gpt_tokenizer(algorithm=bpe, vocab_size=50257):
    special_tokens:
        eos = "<|endoftext|>"
        pad = "<|pad|>"
        unk = "<|unk|>"

    normalize: nfkc, strip                     # unicode normalization, strip whitespace
    pre_tokenize: whitespace, byte_fallback    # split on whitespace, fallback to bytes

    padding:
        side = left
        pad_to = longest                        # pad to longest sequence in batch

    truncation:
        max_length = 2048
        strategy = longest_first

# Train the tokenizer on a corpus
let trainer = BPETrainer(
    vocab_size=50257,
    min_frequency=2,              # minimum frequency for a merge
    special_tokens=gpt_tokenizer.special_tokens,
    show_progress=true
)

# Train from files (streaming — doesn't load entire corpus into memory)
trainer.train_from_files(
    files=["data/corpus_part1.txt", "data/corpus_part2.txt"],
    output="tokenizers/gpt_bpe.json"
)

# Load the trained tokenizer
let tokenizer = gpt_tokenizer.load("tokenizers/gpt_bpe.json")

# Encode text to tensor
let text = "Hello, NeuralScript!"
let tokens: Tensor<[1, _], int32> = tokenizer.encode(text)
print(tokens)    # Tensor([15496, 11, 47041, 12345, 0])

# Decode back to text
let decoded: str = tokenizer.decode(tokens[0])
print(decoded)   # "Hello, NeuralScript!"
```

### Example 2: Fine-tuning an Existing Vocabulary

```nsl
## Add domain-specific tokens to an existing tokenizer.
## Useful for adapting a general-purpose tokenizer to scientific/medical text.

import nsl.tokenize.{Tokenizer, VocabExtender}

# Load existing tokenizer
let base_tokenizer = Tokenizer.load("tokenizers/gpt_bpe.json")
print(f"base vocab size: {base_tokenizer.vocab_size}")   # 50257

# Define domain-specific tokens to add
let medical_tokens = [
    "myocardial",
    "infarction",
    "electrocardiogram",
    "thrombocytopenia",
    "immunoglobulin",
    "pharmacokinetics",
    "pathophysiology"
]

# Extend vocabulary
let extended = VocabExtender(base_tokenizer):
    # Add specific tokens
    add_tokens(medical_tokens)

    # Or mine frequent subwords from a domain corpus
    mine_from_corpus(
        corpus="data/medical_papers.txt",
        max_new_tokens=5000,
        min_frequency=100
    )

print(f"extended vocab size: {extended.vocab_size}")  # 55257

# Save the extended tokenizer
extended.save("tokenizers/medical_bpe.json")

# IMPORTANT: resize the model's embedding layer to match
model.embed.resize(extended.vocab_size)
# New embeddings are initialized to the mean of existing embeddings
```

### Example 3: Batch Encoding with Padding and Attention Masks

```nsl
## Encode a batch of texts with proper padding and attention masks.
## Returns tensors ready for model consumption.

let tokenizer = Tokenizer.load("tokenizers/gpt_bpe.json")

# Batch of texts with different lengths
let texts = [
    "The quick brown fox",
    "jumps over the lazy dog on a sunny afternoon",
    "Hello world"
]

# Batch encode — returns a BatchEncoding struct
let encoded = tokenizer.encode_batch(
    texts,
    padding=true,           # pad shorter sequences
    truncation=true,         # truncate to max_length
    max_length=32,
    return_attention_mask=true,
    return_token_type_ids=false
)

# encoded.input_ids: Tensor<[3, 32], int32>
# encoded.attention_mask: Tensor<[3, 32], int32>  (1 for real tokens, 0 for padding)

print(f"input_ids shape: {encoded.input_ids.shape}")       # [3, 32]
print(f"attention_mask shape: {encoded.attention_mask.shape}")  # [3, 32]

# Directly feed into model
let logits = model.forward(encoded.input_ids, attention_mask=encoded.attention_mask)

# Decode a batch
let decoded = tokenizer.decode_batch(encoded.input_ids, skip_special_tokens=true)
for text in decoded:
    print(text)
```

### Example 4: Streaming Corpus Tokenization

```nsl
## Tokenize a massive corpus (hundreds of GB) in streaming fashion.
## Memory usage stays constant regardless of corpus size.
## Output is memory-mapped binary for fast training data loading.

import nsl.tokenize.{Tokenizer, StreamingTokenizer}

let tokenizer = Tokenizer.load("tokenizers/gpt_bpe.json")

# Create a streaming tokenizer that processes files chunk by chunk
let stream = StreamingTokenizer(
    tokenizer=tokenizer,
    chunk_size=1_000_000,        # process 1M characters at a time
    num_workers=8,                # parallel tokenization
    output_format=binary          # raw int32 binary (most compact)
)

# Process multiple large files
let input_files = glob("data/corpus/*.txt")
print(f"processing {input_files.len()} files...")

stream.tokenize_files(
    input_files,
    output="data/tokenized/corpus.bin",
    # Optional: insert document separators
    document_separator=tokenizer.eos_token_id,
    # Progress tracking
    progress=true
):
    # Optional per-document callback for filtering/transformation
    on_document(doc):
        # Skip very short documents
        if doc.len() < 100:
            return none    # skip this document
        return doc

# Statistics
let stats = stream.stats()
print(f"total tokens: {stats.total_tokens:,}")           # e.g., 15,234,567,890
print(f"total documents: {stats.total_documents:,}")     # e.g., 8,013,372
print(f"tokens per second: {stats.tokens_per_second:,.0f}")  # e.g., 5,000,000
print(f"output size: {stats.output_size_gb:.1f} GB")

# The output file can be directly used as a memory-mapped dataset
let train_data = dataset("corpus"):
    source = nsl.data.MemoryMapped("data/tokenized/corpus.bin", dtype=int32)
    sequence_length = 2048
    packing = true    # pack multiple documents into fixed-length sequences
```

## Design Tensions & Tradeoffs

1. **Built-in vs External tokenizers**: NSL builds in the most common algorithms (BPE,
   WordPiece, Unigram) but the tokenizer ecosystem evolves rapidly. The `Tokenizer.from_hf()`
   interop function can load any HuggingFace tokenizer, providing an escape hatch.

2. **Compile-time vocab checking**: Checking vocab_size at compile time catches mismatches
   early but requires that the tokenizer be available at compile time (not just at runtime).
   If loading a tokenizer from a file, the compiler reads the vocab size from the file during
   compilation.

3. **Streaming tokenization**: Processing huge corpora requires streaming, but streaming
   means you can't compute global statistics (like optimal BPE merges) in one pass.
   Tokenizer *training* is always a batch operation; tokenizer *application* can stream.
