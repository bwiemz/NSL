# M60: Exabyte-Scale Distributed Data Streaming — Design Spec

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M60
**Prerequisites:** M19 (Data Pipeline — basic data loading, done)
**Dependencies:** M58 (Fault Tolerance — checkpoint-aware data resumption), M59 (Topology — data placement awareness)

---

## Overview

Stream trillions of tokens from globally distributed storage into GPUs without ever letting the GPUs starve for data. The data pipeline uses RDMA (Remote Direct Memory Access) for kernel-bypass I/O, GPUDirect Storage for NIC-to-GPU transfers that skip CPU memory entirely, multi-stage prefetching to stay N steps ahead of GPU consumption, and distributed shuffle-without-materialization for epoch-level randomization across petabyte-scale datasets.

**Key design:** The `data` block in a `train` declaration becomes a first-class compiled entity. The compiler analyzes the data pipeline declaration, computes buffer sizes based on the model's batch consumption rate and the storage backend's bandwidth, and emits a multi-stage pipeline with compile-time-determined buffer depths. The prefetch depth is not a heuristic — it is computed from the topology's network bandwidth (M59) and the model's forward pass time (M37 cost model).

**Why this is different from PyTorch DataLoader:** PyTorch's DataLoader uses Python multiprocessing with pickle serialization, OS-level page cache, and kernel-space TCP sockets. Each of these adds latency: pickle serialization is 100-1000x slower than zero-copy, page cache doubles memory usage, and kernel TCP adds two context switches per packet. NSL's pipeline uses: (1) RDMA for zero-copy network transfer, (2) GPUDirect Storage for NIC-to-GPU without CPU involvement, (3) compiled decode/tokenize functions instead of Python callbacks, and (4) compile-time buffer sizing instead of runtime heuristics.

**Why this requires a compiler:** The compiler knows the model's batch size, sequence length, and dtype at compile time. Combined with the topology's network bandwidth (M59) and the model's step time (M37), it can compute exact prefetch depth: `depth = ceil(step_time / transfer_time)`. Python frameworks must guess this at runtime.

---

## Section 1: Language Surface

### `data` Block Declaration

```
data WebCorpus:
    source: "s3://training-data/web-corpus/"
    format: "jsonl"                          # jsonl, parquet, tfrecord, raw_binary
    shards: 10000                            # number of data shards
    tokenizer: BPE("tokenizer.json")

    # Streaming configuration
    prefetch_steps: 4                        # prefetch N batches ahead (auto if omitted)
    shuffle: "global"                        # none, shard, global
    shuffle_seed: 42                         # required if shuffle != none
    compression: "zstd"                      # none, gzip, zstd, lz4, snappy

    # Multi-modal support
    columns:
        text: str                            # tokenized to Tensor<[S], int32>
        image: image(224, 224)               # decoded + resized to Tensor<[3, 224, 224], f32>
        label: int                           # classification label

    # RDMA / GPUDirect configuration
    transfer: "gpudirect"                    # cpu, rdma, gpudirect
    io_threads: 4                            # CPU threads for decode/decompress

@fault_tolerant
train Pretraining:
    model: LLaMA70B
    data: WebCorpus                          # references the data block
    epochs: 1
    batch_size: 2048
    seq_len: 4096
    optimizer: AdamW(lr=3e-4)
    distribute: "dp=32, tp=8"
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | required | URI: `s3://`, `gs://`, `hdfs://`, `/local/path/` |
| `format` | `str` | `"jsonl"` | Data format: jsonl, parquet, tfrecord, raw_binary, webdataset |
| `shards` | `int` | auto-detect | Number of shards (files) in the dataset |
| `tokenizer` | `Tokenizer` | None | Tokenizer for text data |
| `prefetch_steps` | `int` | auto | Batches to prefetch (computed from bandwidth if omitted) |
| `shuffle` | `str` | `"shard"` | Shuffle strategy: none, shard, global |
| `shuffle_seed` | `int` | required if shuffle | Seed for deterministic shuffling |
| `compression` | `str` | `"none"` | Compression of source data |
| `columns` | `map` | required for multi-modal | Column name -> type mapping |
| `transfer` | `str` | `"cpu"` | Transfer mode: cpu, rdma, gpudirect |
| `io_threads` | `int` | 4 | CPU threads for decompression/decoding |
| `max_buffer_mb` | `int` | 4096 | Maximum prefetch buffer size in MB |
| `retry_count` | `int` | 3 | Retries for failed shard reads |
| `retry_delay_ms` | `int` | 1000 | Delay between retries |

### Multi-Source Data Mixing

```
data MixedCorpus:
    sources:
        - source: "s3://web-text/", weight: 0.7, format: "jsonl"
        - source: "s3://books/", weight: 0.2, format: "parquet"
        - source: "s3://code/", weight: 0.1, format: "jsonl"
    tokenizer: BPE("tokenizer.json")
    shuffle: "global"
    shuffle_seed: 42
```

The `weight` field controls the sampling probability for each source. The data pipeline interleaves samples from multiple sources according to these weights, producing mixed batches without requiring pre-materialized interleaving.

### Multi-Modal Streaming

```
data VisionLanguage:
    source: "s3://multimodal-data/"
    format: "webdataset"
    columns:
        text: str
        image: image(384, 384)
        video: video(frames=16, height=224, width=224)
        audio: audio(sample_rate=16000, duration_s=10.0)

    # Modality-specific configuration
    image_decode: "turbojpeg"                # turbojpeg, stb, pillow_compat
    video_decode: "ffmpeg"                   # ffmpeg, nvdec (GPU-accelerated)
    audio_decode: "minimp3"                  # minimp3, ffmpeg

    transfer: "gpudirect"
```

---

## Section 2: Architecture

### Pipeline Stages

```
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
│  Shard    │    │ Decompress │    │   Decode &   │    │   Transfer    │    │   GPU    │
│  Reader   │───>│   (CPU)    │───>│  Tokenize    │───>│ (RDMA/GDS)   │───>│  Buffer  │
│  (I/O)   │    │            │    │   (CPU)      │    │              │    │          │
└──────────┘    └────────────┘    └──────────────┘    └───────────────┘    └──────────┘
     │                │                  │                    │                  │
  io_thread[0]    io_thread[1]      io_thread[2]        DMA engine        GPU compute
  (async I/O)     (zstd/lz4)       (BPE/img_dec)      (NIC hardware)      (training)

Buffer:          Buffer:           Buffer:              Buffer:
raw_shards[]     decompressed[]    tokenized[]          gpu_prefetch[]
(ring buffer)    (ring buffer)     (ring buffer)        (pinned + mapped)
```

### Buffer Sizing (Compile-Time)

```rust
/// Compute optimal prefetch depth from model and topology information.
pub fn compute_prefetch_depth(
    batch_size: u64,
    seq_len: u64,
    dtype_bytes: u64,
    step_time_us: f64,         // from M37 cost model
    network_bw_gbps: f64,     // from M59 topology or user config
    transfer_mode: TransferMode,
) -> u32 {
    let batch_bytes = batch_size * seq_len * dtype_bytes;

    let transfer_time_us = match transfer_mode {
        TransferMode::Cpu => {
            // CPU path: network -> page cache -> memcpy -> GPU
            let net_time = (batch_bytes as f64 * 8.0) / (network_bw_gbps * 1e3);
            let memcpy_time = batch_bytes as f64 / (25e3); // ~25 GB/s PCIe
            net_time + memcpy_time
        }
        TransferMode::Rdma => {
            // RDMA: network -> host pinned memory -> GPU
            let net_time = (batch_bytes as f64 * 8.0) / (network_bw_gbps * 1e3);
            let gpu_time = batch_bytes as f64 / (25e3);
            net_time + gpu_time
        }
        TransferMode::GpuDirect => {
            // GPUDirect Storage: NIC -> GPU memory directly
            let transfer = (batch_bytes as f64 * 8.0) / (network_bw_gbps * 1e3);
            transfer // no CPU involvement
        }
    };

    // Need enough prefetched batches to cover the transfer pipeline
    let depth = (step_time_us / transfer_time_us).ceil() as u32;
    std::cmp::max(depth, 2) // minimum 2 for double-buffering
}
```

---

## Section 3: Shard Management

### Shard Index

Each dataset is split into shards (typically 256MB-1GB each). A shard index file maps shard IDs to storage URIs and metadata:

```rust
pub struct ShardIndex {
    pub shards: Vec<ShardInfo>,
    pub total_samples: u64,
    pub total_bytes: u64,
}

pub struct ShardInfo {
    pub id: u32,
    pub uri: String,                 // s3://bucket/shard_00042.jsonl.zst
    pub byte_offset: u64,            // offset within concatenated dataset
    pub byte_length: u64,            // compressed size
    pub num_samples: u32,            // number of records in this shard
    pub checksum: u32,               // CRC32 for integrity verification
}
```

### Distributed Shard Assignment

Each DP rank gets a disjoint subset of shards. The assignment is deterministic given the shuffle seed and world size:

```rust
/// Assign shards to DP ranks for the current epoch.
pub fn assign_shards(
    index: &ShardIndex,
    dp_rank: u16,
    dp_degree: u16,
    epoch: u32,
    shuffle_seed: u64,
) -> Vec<u32> {
    let mut shard_ids: Vec<u32> = (0..index.shards.len() as u32).collect();

    // Deterministic shuffle using seed + epoch
    let mut rng = Pcg64::seed_from_u64(shuffle_seed.wrapping_add(epoch as u64));
    shard_ids.shuffle(&mut rng);

    // Round-robin assignment to DP ranks
    shard_ids.into_iter()
        .enumerate()
        .filter(|(i, _)| (*i as u16) % dp_degree == dp_rank)
        .map(|(_, id)| id)
        .collect()
}
```

### Distributed Shuffle Without Materialization

Global shuffle across the entire dataset without loading all data into memory:

```
Algorithm: Two-Phase Distributed Shuffle

Phase 1: Intra-shard shuffle
  - Each rank reads its assigned shards sequentially
  - Samples within each shard are shuffled using a local RNG
  - Cost: O(shard_size) memory, no network

Phase 2: Inter-shard interleaving
  - Shards are read in shuffled order (from assign_shards)
  - Samples from consecutive shards are interleaved via a shuffle buffer
  - Buffer size: shuffle_buffer_samples (default: 10000)
  - As new samples arrive, they replace random positions in the buffer
  - Output samples are drawn randomly from the buffer

Result: Approximate global shuffle with O(buffer_size) memory
        Exact global shuffle would require O(dataset_size) memory
```

```rust
pub struct ShuffleBuffer<T> {
    buffer: Vec<Option<T>>,
    rng: Pcg64,
    filled: usize,
    capacity: usize,
}

impl<T> ShuffleBuffer<T> {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            rng: Pcg64::seed_from_u64(seed),
            filled: 0,
            capacity,
        }
    }

    /// Insert a sample and return a shuffled sample (or None during warmup).
    pub fn insert_and_yield(&mut self, sample: T) -> Option<T> {
        if self.filled < self.capacity {
            // Warmup phase: fill the buffer
            self.buffer[self.filled] = Some(sample);
            self.filled += 1;
            if self.filled == self.capacity {
                // Buffer full — yield first shuffled sample
                let idx = self.rng.gen_range(0..self.capacity);
                let out = self.buffer[idx].take();
                self.buffer[idx] = None;
                self.filled -= 1;
                return out;
            }
            return None;
        }

        // Steady state: replace a random position
        let idx = self.rng.gen_range(0..self.capacity);
        let out = self.buffer[idx].take();
        self.buffer[idx] = Some(sample);
        out
    }
}
```

---

## Section 4: RDMA Data Transfer

### GPUDirect Storage Path

For clusters with Mellanox ConnectX-6+ NICs and NVIDIA Magnum IO GPUDirect Storage, data flows directly from the NIC to GPU memory without touching CPU DRAM:

```
Storage Server                      Training Node
┌───────────┐                      ┌───────────────┐
│  NVMe SSD │──> RDMA NIC ──────> │  ConnectX NIC  │
└───────────┘    (RNIC)            │       │        │
                                   │   PCIe Switch  │
                                   │       │        │
                                   │     GPU VRAM   │
                                   │  (BAR1 mapped) │
                                   └───────────────┘
```

```rust
/// RDMA transfer context for GPUDirect Storage.
pub struct RdmaTransfer {
    /// libfabric endpoint for RDMA operations
    endpoint: *mut ffi::fi_endpoint,
    /// Completion queue for async transfer notification
    cq: *mut ffi::fi_cq,
    /// GPU memory regions registered with the NIC
    gpu_mr: Vec<RegisteredMemory>,
    /// Pinned host staging buffers (fallback path)
    host_staging: Vec<PinnedBuffer>,
}

pub struct RegisteredMemory {
    pub gpu_ptr: *mut c_void,       // cuMemAlloc'd GPU memory
    pub size: usize,
    pub mr_key: u64,                // RDMA memory region key
    pub fi_mr: *mut ffi::fi_mr,    // libfabric MR handle
}

impl RdmaTransfer {
    /// Register a GPU buffer for RDMA access (NIC can DMA directly to it).
    pub fn register_gpu_buffer(
        &mut self,
        gpu_ptr: *mut c_void,
        size: usize,
    ) -> Result<RegisteredMemory, RdmaError> {
        // 1. Ensure GPU memory is BAR1-mappable
        //    (cuMemGetAddressRange to verify it's a cuMemAlloc allocation)

        // 2. Register with libfabric for RDMA access
        //    fi_mr_reg(domain, gpu_ptr, size, FI_REMOTE_WRITE, ...)

        // 3. Return the registration with the MR key
        //    (remote side needs this key to RDMA write into our GPU buffer)

        Ok(RegisteredMemory { gpu_ptr, size, mr_key: 0, fi_mr: std::ptr::null_mut() })
    }

    /// Initiate an async RDMA read from remote storage into GPU memory.
    pub fn read_async(
        &self,
        remote_addr: u64,
        remote_key: u64,
        local_mr: &RegisteredMemory,
        offset: usize,
        length: usize,
    ) -> Result<TransferId, RdmaError> {
        // fi_read(endpoint, local_mr.gpu_ptr + offset, length,
        //         local_mr.fi_mr, remote_addr, remote_key, context)
        Ok(TransferId(0))
    }

    /// Poll for completion of an async RDMA transfer.
    pub fn poll_completion(&self) -> Option<TransferId> {
        // fi_cq_read(cq, ...) — non-blocking poll
        None
    }

    /// Block until a specific transfer completes.
    pub fn wait_completion(&self, id: TransferId) -> Result<(), RdmaError> {
        // fi_cq_sread(cq, ..., timeout) — blocking wait
        Ok(())
    }
}

pub struct TransferId(u64);
```

### Fallback CPU Path

When GPUDirect Storage is not available, the pipeline falls back to:

```
Storage ──> TCP/HTTP ──> OS Page Cache ──> User Buffer ──> Pinned Memory ──> GPU
```

This is 3-5x slower than GPUDirect but works on any hardware.

```rust
pub struct CpuTransfer {
    /// HTTP/S3 client for fetching shard data
    client: S3Client,
    /// Pinned host buffers for staging before GPU upload
    pinned_buffers: Vec<PinnedBuffer>,
    /// CUDA stream for async H2D transfers
    copy_stream: CUstream,
}

impl CpuTransfer {
    pub fn fetch_shard_to_gpu(
        &self,
        uri: &str,
        gpu_dst: *mut c_void,
        buffer_idx: usize,
    ) -> Result<(), TransferError> {
        // 1. HTTP GET shard data into pinned host buffer
        let data = self.client.get_object(uri)?;
        let pinned = &self.pinned_buffers[buffer_idx];
        pinned.copy_from_slice(&data);

        // 2. Async cuMemcpyHtoDAsync to GPU
        unsafe {
            cuMemcpyHtoDAsync(
                gpu_dst as u64,
                pinned.as_ptr() as *const c_void,
                data.len(),
                self.copy_stream,
            );
        }

        Ok(())
    }
}
```

---

## Section 5: Prefetch Pipeline

### Multi-Stage Pipeline Implementation

```rust
/// The compiled data pipeline: runs as a set of background threads feeding the GPU.
pub struct DataPipeline {
    /// Stage 1: Shard reader (async I/O)
    shard_reader: ShardReader,
    /// Stage 2: Decompressor
    decompressor: Decompressor,
    /// Stage 3: Decoder/tokenizer
    decoder: Decoder,
    /// Stage 4: Batcher + transfer to GPU
    batcher: Batcher,
    /// Backpressure: signal from GPU that it's not consuming fast enough
    backpressure: Arc<AtomicBool>,
    /// Channels between stages
    raw_tx: crossbeam::channel::Sender<RawShard>,
    raw_rx: crossbeam::channel::Receiver<RawShard>,
    decoded_tx: crossbeam::channel::Sender<DecodedSample>,
    decoded_rx: crossbeam::channel::Receiver<DecodedSample>,
    batch_tx: crossbeam::channel::Sender<GpuBatch>,
    batch_rx: crossbeam::channel::Receiver<GpuBatch>,
}

pub struct ShardReader {
    shard_queue: Vec<ShardInfo>,
    current_shard: usize,
    transfer: Box<dyn DataTransfer>,  // RdmaTransfer or CpuTransfer
    io_threads: usize,
}

pub struct Decompressor {
    codec: CompressionCodec,
    thread_pool: rayon::ThreadPool,
}

pub struct Decoder {
    tokenizer: Option<BpeTokenizer>,
    image_decoder: Option<ImageDecoder>,
    video_decoder: Option<VideoDecoder>,
    audio_decoder: Option<AudioDecoder>,
    thread_pool: rayon::ThreadPool,
}

pub struct Batcher {
    batch_size: u64,
    seq_len: u64,
    shuffle_buffer: ShuffleBuffer<DecodedSample>,
    gpu_buffers: Vec<GpuBuffer>,   // double/triple buffered
    current_buffer: usize,
}

pub struct GpuBatch {
    pub tensors: Vec<*mut c_void>,     // GPU memory pointers
    pub shapes: Vec<Vec<u64>>,
    pub dtypes: Vec<u8>,
    pub step: u64,                     // which training step this batch is for
}

pub struct GpuBuffer {
    pub ptr: *mut c_void,
    pub size: usize,
    pub ready: Arc<AtomicBool>,
}
```

### Pipeline Orchestration

```rust
impl DataPipeline {
    /// Start all pipeline stages as background threads.
    pub fn start(&mut self) {
        // Stage 1: Shard reader thread
        let shard_reader = self.shard_reader.clone();
        let raw_tx = self.raw_tx.clone();
        let backpressure = self.backpressure.clone();
        std::thread::spawn(move || {
            for shard in &shard_reader.shard_queue {
                // Backpressure: wait if GPU is not consuming
                while backpressure.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(10));
                }
                let data = shard_reader.transfer.fetch_shard(shard);
                raw_tx.send(RawShard { id: shard.id, data }).unwrap();
            }
        });

        // Stage 2: Decompressor threads
        let raw_rx = self.raw_rx.clone();
        let decoded_tx = self.decoded_tx.clone();
        let decompressor = self.decompressor.clone();
        std::thread::spawn(move || {
            while let Ok(raw) = raw_rx.recv() {
                let decompressed = decompressor.decompress(&raw.data);
                for sample in parse_samples(&decompressed, raw.id) {
                    decoded_tx.send(sample).unwrap();
                }
            }
        });

        // Stage 3: Decoder/tokenizer threads
        let decoded_rx = self.decoded_rx.clone();
        let batch_tx = self.batch_tx.clone();
        let decoder = self.decoder.clone();
        std::thread::spawn(move || {
            while let Ok(sample) = decoded_rx.recv() {
                let encoded = decoder.process(sample);
                batch_tx.send(encoded).unwrap();
            }
        });
    }

    /// Called by the training loop to get the next batch.
    /// Non-blocking if prefetch is keeping up; blocks if pipeline is behind.
    pub fn next_batch(&mut self) -> GpuBatch {
        self.batch_rx.recv().unwrap()
    }
}
```

### Backpressure

When the GPU is slower than the data pipeline (e.g., during a long backward pass), the pipeline must not overflow its buffers:

```rust
impl DataPipeline {
    /// Called by the training loop when it starts consuming a batch.
    pub fn notify_batch_consumed(&self) {
        // Free the oldest GPU buffer for reuse
        // If all buffers are free, release backpressure
        if self.all_gpu_buffers_free() {
            self.backpressure.store(false, Ordering::Relaxed);
        }
    }

    /// Called internally when all GPU prefetch buffers are full.
    fn activate_backpressure(&self) {
        self.backpressure.store(true, Ordering::Relaxed);
        // Shard reader thread will pause until a buffer is freed
    }
}
```

---

## Section 6: Checkpoint-Aware Resumption

### Data Position Tracking

The data pipeline's position is saved with every micro-checkpoint (M58):

```rust
pub struct DataPosition {
    pub epoch: u32,
    pub shard_index: u32,            // which shard in the shuffled order
    pub sample_offset_in_shard: u64, // byte offset within the current shard
    pub samples_consumed: u64,       // total samples consumed this epoch
    pub shuffle_seed: u64,
    pub rng_state: Vec<u8>,          // serialized RNG state for reproducibility
}

impl DataPipeline {
    /// Serialize current position for checkpointing.
    pub fn checkpoint_position(&self) -> DataPosition {
        DataPosition {
            epoch: self.current_epoch,
            shard_index: self.shard_reader.current_shard as u32,
            sample_offset_in_shard: self.shard_reader.current_offset,
            samples_consumed: self.total_samples,
            shuffle_seed: self.shuffle_seed,
            rng_state: self.shuffle_buffer.rng.to_bytes(),
        }
    }

    /// Resume from a checkpointed position.
    pub fn resume_from(&mut self, pos: &DataPosition) {
        self.current_epoch = pos.epoch;
        self.shard_reader.current_shard = pos.shard_index as usize;
        self.shard_reader.current_offset = pos.sample_offset_in_shard;
        self.total_samples = pos.samples_consumed;

        // Restore RNG state for shuffle buffer
        self.shuffle_buffer.rng = Pcg64::from_bytes(&pos.rng_state);

        // Re-compute shard assignment for this epoch + DP rank
        self.shard_reader.shard_queue = assign_shards(
            &self.shard_index,
            self.dp_rank,
            self.dp_degree,
            pos.epoch,
            pos.shuffle_seed,
        );

        // Skip to the correct shard
        self.shard_reader.shard_queue.drain(..pos.shard_index as usize);

        // Skip to the correct offset within the shard
        // (seek within the shard file)
    }
}
```

---

## Section 7: Multi-Modal Streaming

### Format-Aware Decoders

```rust
pub enum ModalityDecoder {
    Text(TextDecoder),
    Image(ImageDecoder),
    Video(VideoDecoder),
    Audio(AudioDecoder),
}

pub struct TextDecoder {
    pub tokenizer: BpeTokenizer,
    pub max_seq_len: u64,
    pub pad_token_id: u32,
    pub truncation: TruncationStrategy,
}

pub struct ImageDecoder {
    pub backend: ImageBackend,
    pub target_height: u32,
    pub target_width: u32,
    pub normalize: bool,             // normalize to [0, 1]
    pub mean: [f32; 3],             // per-channel mean for normalization
    pub std: [f32; 3],              // per-channel std for normalization
}

pub enum ImageBackend {
    TurboJpeg,    // fastest for JPEG (uses libjpeg-turbo via FFI)
    Stb,          // stb_image — single-file C library, handles PNG/BMP/GIF too
    NvJpeg,       // NVIDIA GPU-accelerated JPEG decode
}

pub struct VideoDecoder {
    pub backend: VideoBackend,
    pub num_frames: u32,
    pub frame_height: u32,
    pub frame_width: u32,
    pub fps_sample: f32,             // sample at this FPS (skip frames if source is higher)
}

pub enum VideoBackend {
    Ffmpeg,       // libavcodec via FFI
    NvDec,        // NVIDIA GPU-accelerated video decode
}

pub struct AudioDecoder {
    pub backend: AudioBackend,
    pub sample_rate: u32,
    pub duration_samples: u64,
    pub mono: bool,                  // convert to mono
}

pub enum AudioBackend {
    MiniMp3,      // tiny MP3 decoder
    Ffmpeg,       // libavcodec for all formats
}
```

### Multi-Modal Batching

Multi-modal data requires interleaved collation — text, images, and other modalities must be batched together while maintaining alignment:

```rust
pub struct MultiModalBatch {
    pub text_tokens: *mut c_void,     // Tensor<[B, S], int32> on GPU
    pub images: *mut c_void,          // Tensor<[B, 3, H, W], f32> on GPU
    pub image_positions: *mut c_void, // Tensor<[B, num_images], int32> — where images go in text
    pub labels: *mut c_void,          // Tensor<[B], int32> on GPU
    pub batch_size: u32,
    pub has_image: Vec<bool>,         // which samples in the batch have images
}

impl Decoder {
    pub fn collate_multimodal(
        &self,
        samples: Vec<DecodedSample>,
    ) -> MultiModalBatch {
        // 1. Pad text sequences to max length in batch
        // 2. Stack images (skip-pad for samples without images)
        // 3. Compute image position indices within text sequences
        // 4. Transfer all tensors to GPU
        todo!()
    }
}
```

---

## Section 8: Codegen Changes

### Data Block Compilation

**Module:** `crates/nsl-codegen/src/data_pipeline.rs` (NEW)

```rust
pub struct DataPipelineCompiler {
    config: DataBlockConfig,
    prefetch_depth: u32,
    buffer_size_bytes: u64,
    transfer_mode: TransferMode,
}

impl DataPipelineCompiler {
    /// Compile a `data` block declaration into pipeline initialization code.
    pub fn compile_data_block(
        &mut self,
        compiler: &mut Compiler,
        data_block: &DataBlock,
        train_block: &TrainBlock,
    ) {
        // 1. Compute prefetch depth from topology + cost model
        self.prefetch_depth = compute_prefetch_depth(
            train_block.batch_size,
            train_block.seq_len,
            4, // f32 = 4 bytes
            compiler.cost_model.step_time_us(),
            compiler.topology.as_ref().map_or(10.0, |t| t.data_bandwidth_gbps()),
            data_block.transfer_mode,
        );

        // 2. Emit pipeline initialization
        compiler.emit_call("nsl_data_stream_init", &[
            compiler.const_str(&data_block.source),
            compiler.const_u32(data_block.shards),
            compiler.const_u32(self.prefetch_depth),
            compiler.const_u8(data_block.transfer_mode as u8),
            compiler.const_u8(data_block.compression as u8),
            compiler.const_u32(data_block.io_threads),
        ]);

        // 3. If multi-modal, emit column decoders
        if let Some(columns) = &data_block.columns {
            for (name, col_type) in columns {
                compiler.emit_call("nsl_data_add_column", &[
                    compiler.const_str(name),
                    compiler.const_u8(col_type.to_id()),
                ]);
            }
        }

        // 4. Emit tokenizer initialization if present
        if let Some(tokenizer) = &data_block.tokenizer {
            compiler.emit_call("nsl_data_set_tokenizer", &[
                compiler.const_str(&tokenizer.vocab_path),
            ]);
        }
    }

    /// Emit the call to get the next batch inside the training loop.
    pub fn emit_next_batch(&self, compiler: &mut Compiler) -> Value {
        compiler.emit_call("nsl_data_stream_next_batch", &[])
    }

    /// Emit checkpoint position save.
    pub fn emit_save_position(&self, compiler: &mut Compiler) -> Value {
        compiler.emit_call("nsl_data_stream_checkpoint", &[])
    }
}
```

### Integration with Train Block

**Module:** `crates/nsl-codegen/src/stmt.rs`

```rust
fn compile_train_block(&mut self, train: &TrainBlock) {
    // ... existing setup ...

    // NEW: Compile data pipeline if data block is referenced
    if let Some(data_ref) = &train.data {
        let data_block = self.resolve_data_block(data_ref)?;
        let mut pipeline_compiler = DataPipelineCompiler::new(&data_block);
        pipeline_compiler.compile_data_block(self, &data_block, train);

        // Inside the training loop, replace data loading with pipeline fetch
        self.data_pipeline = Some(pipeline_compiler);
    }
}

fn compile_train_loop_body(&mut self, train: &TrainBlock) {
    // Get next batch from pipeline (non-blocking if prefetch is ahead)
    let batch = if let Some(pipeline) = &self.data_pipeline {
        pipeline.emit_next_batch(self)
    } else {
        self.compile_simple_data_load(train)
    };

    // ... forward, backward, optimizer step ...

    // Notify pipeline that batch was consumed (releases buffer for reuse)
    if self.data_pipeline.is_some() {
        self.emit_call("nsl_data_stream_batch_consumed", &[]);
    }
}
```

### AST Extension

**Module:** `crates/nsl-ast/src/lib.rs`

```rust
pub struct DataBlock {
    pub name: String,
    pub source: String,
    pub sources: Option<Vec<DataSource>>,   // for multi-source mixing
    pub format: DataFormat,
    pub shards: u32,
    pub tokenizer: Option<TokenizerConfig>,
    pub prefetch_steps: Option<u32>,         // None = auto-compute
    pub shuffle: ShuffleStrategy,
    pub shuffle_seed: Option<u64>,
    pub compression: CompressionKind,
    pub columns: Option<Vec<(String, ColumnType)>>,
    pub transfer: TransferMode,
    pub io_threads: u32,
    pub max_buffer_mb: u32,
    pub span: Span,
}

pub struct DataSource {
    pub source: String,
    pub weight: f64,
    pub format: DataFormat,
}

pub enum DataFormat {
    Jsonl,
    Parquet,
    TfRecord,
    RawBinary,
    WebDataset,
}

pub enum ShuffleStrategy {
    None,
    Shard,    // shuffle shard order only
    Global,   // shuffle within and across shards
}

pub enum TransferMode {
    Cpu,
    Rdma,
    GpuDirect,
}

pub enum ColumnType {
    Str,
    Int,
    Float,
    Image { height: u32, width: u32 },
    Video { frames: u32, height: u32, width: u32 },
    Audio { sample_rate: u32, duration_s: f32 },
}
```

---

## Section 9: Runtime FFI Functions

```rust
// crates/nsl-runtime/src/data_stream.rs

/// Initialize the data streaming pipeline.
#[no_mangle]
pub extern "C" fn nsl_data_stream_init(
    source: *const c_char,
    num_shards: u32,
    prefetch_depth: u32,
    transfer_mode: u8,
    compression: u8,
    io_threads: u32,
) { ... }

/// Add a column decoder to the pipeline (for multi-modal data).
#[no_mangle]
pub extern "C" fn nsl_data_add_column(
    name: *const c_char,
    column_type: u8,
) { ... }

/// Set the tokenizer for text data.
#[no_mangle]
pub extern "C" fn nsl_data_set_tokenizer(
    vocab_path: *const c_char,
) { ... }

/// Configure distributed shard assignment.
#[no_mangle]
pub extern "C" fn nsl_data_set_distributed(
    dp_rank: u16,
    dp_degree: u16,
    shuffle_seed: u64,
) { ... }

/// Start the background pipeline threads.
#[no_mangle]
pub extern "C" fn nsl_data_stream_start() { ... }

/// Get the next batch from the pipeline. Blocks if pipeline is behind.
/// Returns a pointer to a GpuBatch struct.
#[no_mangle]
pub extern "C" fn nsl_data_stream_next_batch() -> *mut GpuBatch { ... }

/// Notify the pipeline that the current batch has been consumed.
#[no_mangle]
pub extern "C" fn nsl_data_stream_batch_consumed() { ... }

/// Serialize the current data pipeline position for checkpointing.
/// Returns a pointer to serialized DataPosition (caller owns the memory).
#[no_mangle]
pub extern "C" fn nsl_data_stream_checkpoint() -> *mut DataPosition { ... }

/// Resume the data pipeline from a checkpointed position.
#[no_mangle]
pub extern "C" fn nsl_data_stream_resume(
    position: *const DataPosition,
) { ... }

/// Stop the pipeline and release all resources.
#[no_mangle]
pub extern "C" fn nsl_data_stream_stop() { ... }

/// Get pipeline statistics (for monitoring/debugging).
#[no_mangle]
pub extern "C" fn nsl_data_stream_stats() -> DataStreamStats { ... }

#[repr(C)]
pub struct DataStreamStats {
    pub samples_read: u64,
    pub bytes_transferred: u64,
    pub batches_prefetched: u32,
    pub gpu_stalls: u64,          // times GPU had to wait for data
    pub backpressure_events: u64, // times pipeline paused due to GPU being slow
    pub io_errors: u64,
    pub retries: u64,
}
```

---

## Section 10: File Changes

### New Files

| File | Description |
|------|-------------|
| `crates/nsl-runtime/src/data_stream.rs` | FFI entry points for data pipeline |
| `crates/nsl-runtime/src/shard_reader.rs` | Shard index parsing, shard assignment |
| `crates/nsl-runtime/src/rdma_transfer.rs` | RDMA/GPUDirect data transfer |
| `crates/nsl-runtime/src/shuffle_buffer.rs` | Distributed shuffle without materialization |
| `crates/nsl-runtime/src/modality_decoders.rs` | Image/video/audio decoders |
| `crates/nsl-runtime/src/backpressure.rs` | Backpressure and flow control |
| `crates/nsl-codegen/src/data_pipeline.rs` | Data block compilation, prefetch depth |
| `crates/nsl-semantic/src/data_block.rs` | Semantic validation for data blocks |
| `crates/nsl-ast/src/data_block.rs` | AST nodes for data block declarations |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-parser/src/lib.rs` | Parse `data` block syntax (source, columns, etc.) |
| `crates/nsl-ast/src/lib.rs` | Add DataBlock, DataSource, ColumnType enums |
| `crates/nsl-codegen/src/stmt.rs` | Integrate DataPipelineCompiler into train block codegen |
| `crates/nsl-codegen/src/context.rs` | Add data pipeline state to compilation context |
| `crates/nsl-runtime/src/lib.rs` | Export new FFI functions |
| `crates/nsl-runtime/src/micro_checkpoint.rs` | Include DataPosition in checkpoint metadata |
| `crates/nsl-codegen/src/linker.rs` | Link libfabric (RDMA), libjpeg-turbo, ffmpeg if used |
| `crates/nsl-cli/src/main.rs` | No new CLI flags (data config lives in the source file) |

---

## Section 11: Testing Strategy

### Unit Tests

**`crates/nsl-runtime/src/shard_reader.rs`:**
- `assign_shards` distributes shards evenly across DP ranks
- `assign_shards` with different epochs produces different orderings (same seed)
- `assign_shards` with same epoch + same seed is deterministic
- No shard assigned to two different DP ranks (disjoint partitioning)

**`crates/nsl-runtime/src/shuffle_buffer.rs`:**
- ShuffleBuffer warmup: first `capacity` inserts return None
- ShuffleBuffer steady state: every insert returns a sample
- ShuffleBuffer with same seed produces identical output sequences
- ShuffleBuffer output is not in insertion order (actually shuffled)

**`crates/nsl-runtime/src/data_stream.rs`:**
- Pipeline with 1 shard, no compression, CPU transfer: produces correct batches
- Pipeline backpressure: 0-depth prefetch blocks after one batch
- Pipeline stats: samples_read and bytes_transferred are accurate
- Checkpoint/resume: position round-trip produces identical sample sequence

**`crates/nsl-runtime/src/modality_decoders.rs`:**
- JPEG decode produces correct pixel values (compare against reference)
- Text tokenization matches standalone tokenizer output
- Image resize produces correct dimensions

**`crates/nsl-codegen/src/data_pipeline.rs`:**
- `compute_prefetch_depth` with high bandwidth returns small depth
- `compute_prefetch_depth` with low bandwidth returns large depth
- `compute_prefetch_depth` always returns >= 2 (double buffer minimum)
- Data block without tokenizer for non-text format: no error
- Data block with shuffle but no shuffle_seed: compile error

**`crates/nsl-semantic/src/data_block.rs`:**
- `data` block with invalid format string: compile error
- `data` block with `shuffle: "global"` but no `shuffle_seed`: compile error
- `data` block with `transfer: "gpudirect"` validates hardware support warning
- Multi-source data block with weights not summing to 1.0: warning

### E2E Tests

- **`examples/m60_data_basic.nsl`** — Single-shard JSONL data, tokenize, train for 5 steps, verify samples consumed
- **`examples/m60_data_multimodal.nsl`** — WebDataset with text + images, verify both modalities decoded
- **`examples/m60_data_checkpoint.nsl`** — Train for 10 steps, checkpoint, resume, verify no duplicate/missing samples
- **`examples/m60_data_shuffle.nsl`** — Two runs with same shuffle_seed produce identical sample ordering
- **`examples/m60_data_backpressure.nsl`** — Slow model (artificial delay), verify no OOM from unbounded prefetch

---

## Section 12: Deliverables

- `data` block declaration with source, format, shards, tokenizer, columns
- Multi-source data mixing with weighted sampling
- Multi-modal streaming: text, image, video, audio with format-aware decoders
- Compile-time prefetch depth computation from topology bandwidth + model step time
- RDMA data transfer with GPUDirect Storage (NIC-to-GPU, zero CPU copy)
- CPU fallback path for clusters without RDMA hardware
- Distributed shuffle without materialization (shuffle buffer algorithm)
- Backpressure: pipeline pauses when GPU cannot consume fast enough
- Checkpoint-aware resumption: data position saved/restored with model checkpoints
- Pipeline statistics: samples read, bytes transferred, GPU stalls, backpressure events
- Integration with M58 fault tolerance (data position in micro-checkpoints)

## Not in Scope

- Data preprocessing/augmentation (random crop, color jitter, etc. — future work)
- Custom user-defined decode functions (compiled callbacks in NSL)
- Data validation/quality filtering at streaming time
- Federated data access (cross-datacenter with privacy constraints)
- On-the-fly tokenizer training (tokenizer must be pre-trained)
- Streaming from real-time sources (Kafka, Kinesis)
- Data deduplication at streaming time
