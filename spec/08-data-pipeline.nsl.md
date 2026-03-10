# Section 8 — Data Pipeline

## Design Rationale

Data loading is often the bottleneck in ML training — PyTorch's `DataLoader` is single-threaded
in its collation, relies on Python multiprocessing (with its serialization overhead), and offers
no built-in support for sequence packing, memory mapping, or streaming from cloud storage. NSL's
`dataset` and `dataloader` keywords provide zero-copy data loading with native multi-threading
(no GIL), memory mapping, automatic batching with smart padding/packing, and prefetch pipelines.

## Dataset/Dataloader Grammar

```ebnf
dataset_def     ::= 'dataset' IDENT '(' STRING ')' ':' INDENT dataset_body DEDENT
dataset_body    ::= (dataset_stmt NEWLINE)*
dataset_stmt    ::= 'source' '=' source_expr
                   | 'format' '=' format_spec
                   | 'transform' '=' transform_chain
                   | 'filter' '=' filter_expr
                   | 'shuffle' '=' BOOL
                   | 'sequence_length' '=' INT
                   | 'packing' '=' BOOL
                   | 'max_samples' '=' INT

source_expr     ::= 'nsl.data.MemoryMapped' '(' STRING (',' kwargs)? ')'
                   | 'nsl.data.JSONL' '(' STRING ')'
                   | 'nsl.data.Parquet' '(' STRING ')'
                   | 'nsl.data.Arrow' '(' STRING ')'
                   | 'nsl.data.HuggingFace' '(' STRING (',' kwargs)? ')'
                   | 'nsl.data.CSV' '(' STRING ')'
                   | glob_expr

dataloader_def  ::= 'dataloader' IDENT '(' dataloader_config ')' ':' INDENT dl_body DEDENT
dataloader_config ::= config_entry (',' config_entry)*
dl_body         ::= (dl_stmt NEWLINE)*
dl_stmt         ::= 'batch_size' '=' INT
                   | 'shuffle' '=' BOOL
                   | 'num_workers' '=' INT
                   | 'pin_memory' '=' BOOL
                   | 'prefetch' '=' INT
                   | 'drop_last' '=' BOOL
                   | 'collate' '=' collate_fn
```

## 3 Data Pipeline Examples

### Example 1: LLM Pre-training Data Pipeline

```nsl
## Memory-mapped binary dataset with sequence packing for LLM pre-training.
## Sequence packing concatenates short documents to fill fixed-length sequences,
## maximizing GPU utilization by eliminating padding waste.

import nsl.data.{MemoryMapped, DataLoader, PackedSequenceDataset}

# Define the dataset — lazy, zero-copy memory mapping
dataset pretrain_data("openwebtext"):
    source = MemoryMapped("data/openwebtext.bin", dtype=int32)
    sequence_length = 2048

    # Sequence packing: concatenate documents with EOS separators
    # into fixed-length sequences. No padding waste.
    packing = true
    pack_separator = 50256      # EOS token id

    # Optional: skip sequences with too few unique tokens
    filter = |seq| seq.unique().len() > 100

# Create the data loader
let loader = DataLoader(
    dataset=pretrain_data,
    batch_size=32,
    shuffle=true,                # shuffle at sequence level
    num_workers=8,               # native threads, not processes
    pin_memory=true,             # pin to page-locked memory for faster GPU transfer
    prefetch=4,                  # prefetch 4 batches ahead
    drop_last=true               # drop incomplete last batch
)

# Iterate
for batch in loader:
    # batch.input_ids: Tensor<[32, 2048], int32, cpu_pinned>
    let gpu_batch = batch.input_ids.to(cuda, non_blocking=true)
    # non_blocking=true: transfer overlaps with computation
    let logits = model.forward(gpu_batch)
    # ...

# Dataset statistics
print(f"total tokens: {pretrain_data.total_tokens():,}")
print(f"total sequences: {pretrain_data.len():,}")
print(f"packing efficiency: {pretrain_data.packing_efficiency():.1%}")
```

### Example 2: Multi-Modal Dataset (Image + Text)

```nsl
## A dataset combining images and text captions from Parquet files.
## Demonstrates transforms, lazy loading, and custom collation.

import nsl.data.{Parquet, DataLoader, transforms as T}

# Image transforms — composed as a zero-copy pipeline
let image_transform = T.compose([
    T.Resize(256),                    # resize shortest edge to 256
    T.CenterCrop(224),               # crop center 224x224
    T.ToTensor(),                     # convert to [C, H, W] float tensor
    T.Normalize(                      # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset image_text_data("laion"):
    source = Parquet("data/laion_subset/*.parquet")
    # Lazy decode: images are decoded only when accessed
    # (Parquet stores them as binary blobs)

    transform = |sample| {
        image: image_transform(sample.image_bytes),
        caption: tokenizer.encode(sample.caption, max_length=77),
        metadata: sample.metadata
    }

    # Filter out samples with missing data
    filter = |sample| sample.image_bytes is not none and sample.caption.len() > 10

    # Shuffle buffer: for streaming datasets, maintain a shuffle buffer
    shuffle_buffer = 10000

let loader = DataLoader(
    dataset=image_text_data,
    batch_size=256,
    num_workers=16,
    pin_memory=true,
    prefetch=4,
    # Custom collation for variable-length captions
    collate = |samples| {
        images: stack([s.image for s in samples]),     # [256, 3, 224, 224]
        captions: pad_sequence(                         # [256, max_len]
            [s.caption for s in samples],
            padding_value=tokenizer.pad_token_id
        ),
        attention_mask: create_mask(
            [s.caption.len() for s in samples]
        )
    }
)
```

### Example 3: Streaming Dataset from HuggingFace Hub

```nsl
## Stream a dataset from HuggingFace Hub without downloading it entirely.
## Useful for massive datasets that don't fit on local disk.

import nsl.data.{HuggingFace, DataLoader, StreamingDataset}

# Stream directly from HuggingFace Hub
dataset wiki_data("wikipedia"):
    source = HuggingFace(
        path="wikipedia",
        name="20220301.en",
        split="train",
        streaming=true           # don't download, stream over HTTP
    )

    # Process each sample on-the-fly
    transform = |sample| {
        input_ids: tokenizer.encode(
            sample.text,
            max_length=2048,
            truncation=true
        ),
        title: sample.title
    }

    # Resume from a specific position (for fault tolerance)
    # resume_from = "shard_42:offset_1337"

# Streaming data loader — works with infinite/unknown-length datasets
let loader = DataLoader(
    dataset=wiki_data,
    batch_size=16,
    num_workers=4,
    prefetch=8,                  # prefetch more for streaming (hides network latency)
    # No shuffle (streaming doesn't support random access)
    # Use shuffle_buffer in dataset config instead
)

# Training loop with streaming data
for (step, batch) in loader.enumerate():
    let logits = model.forward(batch.input_ids)
    let loss = cross_entropy(logits, batch.input_ids)
    # ...

    if step >= 100000:
        break    # stop after 100K steps (streaming is infinite)

    # Checkpoint includes dataset position for resumability
    if step % 5000 == 0:
        let state = {
            model: model.state_dict(),
            optimizer: optimizer.state_dict(),
            data_position: loader.position()    # saves stream offset
        }
        save_checkpoint(f"ckpt_step{step}.nslm", state)
```

## Zero-Copy Semantics

NSL's data pipeline avoids copies at every stage:

```
File on disk (mmap) → Transform (in-place) → Batch (view) → Pin (DMA) → GPU (async)
         │                    │                   │              │            │
         └── 0 copies ────────┘                   │              │            │
                                    0 copies ─────┘              │            │
                                                   0 copies ─────┘            │
                                                                1 DMA transfer┘
```

Only one data transfer happens: DMA from pinned CPU memory to GPU memory, and it's
asynchronous (overlapped with computation).

## Design Tensions & Tradeoffs

1. **Packing vs Padding**: Sequence packing maximizes GPU utilization but complicates
   attention masks (you need block-diagonal masks to prevent cross-document attention).
   NSL's packing generates the appropriate masks automatically.

2. **Streaming vs Random access**: Streaming datasets can't be shuffled globally. NSL's
   `shuffle_buffer` provides approximate shuffling within a window. For full random access,
   use memory-mapped datasets.

3. **num_workers threads vs processes**: NSL uses native threads (no GIL), so worker threads
   share memory with the main thread. This is faster than Python's multiprocessing but means
   transforms must be thread-safe. The compiler verifies this for transforms written in NSL.
