# Section 10 — Standard Library

## Design Rationale

NSL's standard library is organized into focused modules that mirror the ML workflow:
data → tokenization → model → training → quantization → export. Each module provides
a complete, opinionated API surface — not a sprawling collection of every possible function,
but the 80% of operations that 95% of users need. Advanced users can extend via custom
implementations or the interop layer.

## Module Tree

```
nsl
├── nn          — Neural network primitives (layers, activations, losses)
├── optim       — Optimizers and learning rate schedulers
├── quant       — Quantization toolkit
├── data        — Dataset, DataLoader, transforms
├── tokenize    — Tokenization algorithms and vocabulary management
├── dist        — Distributed training utilities
├── export      — Model export: ONNX, CoreML, TFLite, QNF
├── bench       — Benchmarking, profiling, and memory analysis
└── compat      — PyTorch/HuggingFace interoperability layer
```

---

## nsl.nn — Neural Network Primitives

Top 10 public APIs:

```nsl
# 1. Linear layer
fn Linear(in_features: int, out_features: int, bias: bool = true) -> LinearLayer
    ## Fully connected linear transformation: y = xW^T + b

# 2. Multi-head attention
fn Attention(d_model: int, n_heads: int, dropout: f32 = 0.0,
             causal: bool = false, flash: bool = true) -> AttentionLayer
    ## Multi-head scaled dot-product attention with optional flash attention

# 3. Layer normalization
fn LayerNorm(normalized_shape: int | list<int>, eps: f64 = 1e-5) -> LayerNormLayer
    ## Applies layer normalization over the last D dimensions

# 4. RMS normalization
fn RMSNorm(dim: int, eps: f64 = 1e-6) -> RMSNormLayer
    ## Root mean square normalization (used in LLaMA, Gemma)

# 5. Embedding lookup
fn Embedding(num_embeddings: int, embedding_dim: int,
             padding_idx: int = none) -> EmbeddingLayer
    ## Lookup table for dense embeddings from integer indices

# 6. Convolutional layer
fn Conv2d(in_channels: int, out_channels: int, kernel_size: int | (int, int),
          stride: int = 1, padding: int | str = 0) -> Conv2dLayer
    ## 2D convolution over an input signal composed of input planes

# 7. Multi-layer perceptron
fn MLP(in_dim: int, hidden_dim: int, out_dim: int,
       activation: Activation = gelu, dropout: f32 = 0.0) -> MLPLayer
    ## Two-layer feedforward network: Linear → Activation → Dropout → Linear

# 8. Dropout regularization
fn Dropout(p: f32 = 0.1) -> DropoutLayer
    ## Randomly zeros elements with probability p during training

# 9. Cross-entropy loss
fn cross_entropy(input: Tensor<[batch, classes], _>,
                 target: Tensor<[batch], int32>,
                 weight: Tensor<[classes], _> = none,
                 ignore_index: int = -100,
                 label_smoothing: f32 = 0.0) -> Tensor<[], _>
    ## Computes cross-entropy loss between input logits and target labels

# 10. Softmax activation
fn softmax(input: Tensor, dim: int = -1) -> Tensor
    ## Applies softmax function along the specified dimension
```

Additional APIs: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `BatchNorm`, `GroupNorm`,
`Conv1d`, `ConvTranspose2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`,
`RotaryEmbed`, `MoERouter`, `flash_attention`, `mse_loss`, `l1_loss`,
`binary_cross_entropy`, `cosine_similarity`, `pairwise_distance`.

---

## nsl.optim — Optimizers and Schedulers

Top 10 public APIs:

```nsl
# 1. AdamW optimizer
fn AdamW(params: ParamGroup, lr: f64 = 1e-3, betas: (f64, f64) = (0.9, 0.999),
         eps: f64 = 1e-8, weight_decay: f64 = 0.01) -> AdamWOptimizer
    ## Adam with decoupled weight decay (Loshchilov & Hutter, 2019)

# 2. SGD optimizer
fn SGD(params: ParamGroup, lr: f64 = 0.01, momentum: f64 = 0.0,
       dampening: f64 = 0.0, nesterov: bool = false) -> SGDOptimizer
    ## Stochastic gradient descent with optional momentum

# 3. Lion optimizer
fn Lion(params: ParamGroup, lr: f64 = 1e-4,
        betas: (f64, f64) = (0.9, 0.99), weight_decay: f64 = 0.0) -> LionOptimizer
    ## Evolved sign momentum optimizer (Chen et al., 2023)

# 4. Muon optimizer
fn Muon(params: ParamGroup, lr: f64 = 0.02, momentum: f64 = 0.95,
        nesterov: bool = true, ns_steps: int = 5) -> MuonOptimizer
    ## Muon optimizer with orthogonalized momentum (Jordan et al., 2024)

# 5. Cosine annealing scheduler
fn CosineScheduler(optimizer: Optimizer, T_max: int,
                   eta_min: f64 = 0.0) -> CosineScheduler
    ## Cosine annealing learning rate schedule

# 6. Warmup + cosine scheduler
fn WarmupCosineScheduler(optimizer: Optimizer, warmup_steps: int,
                         total_steps: int, min_lr: f64 = 0.0) -> WCScheduler
    ## Linear warmup for warmup_steps, then cosine decay to min_lr

# 7. Linear decay scheduler
fn LinearDecayScheduler(optimizer: Optimizer, total_steps: int,
                        end_factor: f64 = 0.0) -> LinearScheduler
    ## Linearly decays learning rate from initial to end_factor * initial

# 8. OneCycle scheduler
fn OneCycleScheduler(optimizer: Optimizer, max_lr: f64,
                     total_steps: int, pct_start: f64 = 0.3) -> OneCycleScheduler
    ## Super-convergence one-cycle policy (Smith & Topin, 2019)

# 9. Gradient accumulator
fn GradAccumulator(params: ParamGroup, steps: int = 1) -> GradAccumulator
    ## Accumulates gradients over multiple micro-batches

# 10. Parameter group builder
fn ParamGroup(params: list<Param>, **overrides) -> ParamGroup
    ## Groups parameters with shared optimizer hyperparameters
```

---

## nsl.quant — Quantization Toolkit

Top 10 public APIs:

```nsl
# 1. Quantize a model
fn quantize(model: Model, scheme: QuantScheme, mode: QuantMode = static,
            calibration_data: Dataset = none) -> QuantizedModel
    ## Applies quantization to all eligible layers in the model

# 2. Quantize a single tensor
fn quantize_tensor(tensor: Tensor, scheme: QuantScheme,
                   granularity: Granularity = per_tensor) -> QuantizedTensor
    ## Quantizes a single tensor with specified scheme

# 3. Dequantize
fn dequantize(qtensor: QuantizedTensor) -> Tensor
    ## Converts quantized tensor back to floating point

# 4. Sensitivity analysis
fn analyze_sensitivity(model: Model, data: Dataset,
                       config: QuantConfig, metric: Metric) -> SensitivityReport
    ## Per-layer sensitivity analysis for quantization

# 5. GPTQ quantization
fn gptq(model: Model, calibration_data: Dataset, bits: int = 4,
        group_size: int = 128, act_order: bool = true) -> QuantizedModel
    ## GPTQ weight-only quantization using Hessian information

# 6. AWQ quantization
fn awq(model: Model, calibration_data: Dataset, bits: int = 4,
       group_size: int = 128) -> QuantizedModel
    ## Activation-Aware Weight Quantization

# 7. Compute quantization error
fn quant_error(original: Tensor, quantized: QuantizedTensor) -> QuantError
    ## Computes MSE, SNR, max absolute error between original and quantized

# 8. Calibrate observer
fn calibrate(model: Model, data: Dataset,
             method: CalibrationMethod = minmax) -> CalibrationResult
    ## Run calibration data through model to determine quantization scales

# 9. Create quantization config
fn QuantConfig(scheme: QuantScheme, granularity: Granularity = per_tensor,
               mode: QuantMode = static, **kwargs) -> QuantConfig
    ## Creates a quantization configuration for use with quantize()

# 10. Mixed quantization config builder
fn MixedQuantConfig() -> MixedQuantConfigBuilder
    ## Builder for per-layer mixed-precision quantization configurations
```

---

## nsl.data — Dataset and DataLoader

Top 10 public APIs:

```nsl
# 1. Memory-mapped dataset
fn MemoryMapped(path: str, dtype: DType = int32,
                shape: Shape = none) -> MemoryMappedDataset
    ## Zero-copy memory-mapped file for binary tensor data

# 2. JSONL dataset
fn JSONL(path: str, fields: list<str> = none) -> JSONLDataset
    ## Line-delimited JSON dataset with lazy parsing

# 3. Parquet dataset
fn Parquet(path: str | list<str>, columns: list<str> = none) -> ParquetDataset
    ## Apache Parquet columnar dataset with predicate pushdown

# 4. HuggingFace Hub dataset
fn HuggingFace(path: str, name: str = none, split: str = "train",
               streaming: bool = false) -> HFDataset
    ## Load datasets from HuggingFace Hub, with optional streaming

# 5. DataLoader
fn DataLoader(dataset: Dataset, batch_size: int, shuffle: bool = false,
              num_workers: int = 0, pin_memory: bool = false,
              prefetch: int = 2, drop_last: bool = false) -> DataLoader
    ## Multi-threaded data loading with prefetch and pinned memory

# 6. Transform composition
fn compose(transforms: list<Transform>) -> Transform
    ## Composes multiple transforms into a single transform

# 7. Sequence packer
fn SequencePacker(dataset: Dataset, seq_length: int,
                  separator_token: int, shuffle: bool = true) -> PackedDataset
    ## Packs variable-length sequences into fixed-length blocks

# 8. Pad sequence
fn pad_sequence(sequences: list<Tensor>, padding_value: int = 0,
                side: str = "right") -> Tensor
    ## Pads a list of variable-length tensors to the same length

# 9. Data sampler
fn DistributedSampler(dataset: Dataset, num_replicas: int,
                      rank: int, shuffle: bool = true) -> Sampler
    ## Distributes data across multiple processes for distributed training

# 10. Arrow dataset
fn Arrow(path: str) -> ArrowDataset
    ## Apache Arrow IPC format dataset with zero-copy reads
```

---

## nsl.tokenize — Tokenization

Top 10 public APIs:

```nsl
# 1. Load tokenizer
fn Tokenizer.load(path: str) -> Tokenizer
    ## Load a trained tokenizer from a JSON file

# 2. BPE trainer
fn BPETrainer(vocab_size: int, min_frequency: int = 2,
              special_tokens: list<str> = []) -> Trainer
    ## Train a Byte Pair Encoding tokenizer

# 3. Encode text
fn Tokenizer.encode(text: str, max_length: int = none,
                    truncation: bool = false) -> Tensor<[1, _], int32>
    ## Encode a string into token IDs

# 4. Decode tokens
fn Tokenizer.decode(ids: Tensor<[_], int32>,
                    skip_special_tokens: bool = true) -> str
    ## Decode token IDs back to a string

# 5. Batch encode
fn Tokenizer.encode_batch(texts: list<str>, padding: bool = false,
                          truncation: bool = false, max_length: int = none,
                          return_attention_mask: bool = true) -> BatchEncoding
    ## Encode a batch of strings with padding and attention masks

# 6. Batch decode
fn Tokenizer.decode_batch(ids: Tensor<[batch, seq], int32>,
                          skip_special_tokens: bool = true) -> list<str>
    ## Decode a batch of token ID sequences to strings

# 7. Vocabulary extender
fn VocabExtender(tokenizer: Tokenizer) -> VocabExtender
    ## Add new tokens to an existing tokenizer's vocabulary

# 8. Streaming tokenizer
fn StreamingTokenizer(tokenizer: Tokenizer, chunk_size: int = 1_000_000,
                      num_workers: int = 4) -> StreamingTokenizer
    ## Tokenize large files in streaming fashion

# 9. WordPiece trainer
fn WordPieceTrainer(vocab_size: int, min_frequency: int = 2) -> Trainer
    ## Train a WordPiece tokenizer (BERT-style)

# 10. SentencePiece trainer
fn SentencePieceTrainer(vocab_size: int, model_type: str = "unigram") -> Trainer
    ## Train a SentencePiece tokenizer (T5/mBART-style)
```

---

## nsl.dist — Distributed Training

Top 10 public APIs:

```nsl
# 1. Initialize distributed
fn init(backend: str = "nccl") -> DistContext
    ## Initialize distributed training environment

# 2. Get world size
fn world_size() -> int
    ## Returns the number of processes in the current group

# 3. Get rank
fn rank() -> int
    ## Returns the rank of the current process

# 4. All-reduce
fn all_reduce(tensor: Tensor, op: ReduceOp = sum) -> Tensor
    ## Reduces tensor across all processes and distributes the result

# 5. All-gather
fn all_gather(tensor: Tensor) -> list<Tensor>
    ## Gathers tensors from all processes

# 6. Broadcast
fn broadcast(tensor: Tensor, src: int = 0) -> Tensor
    ## Broadcasts tensor from source rank to all other ranks

# 7. Barrier
fn barrier() -> void
    ## Synchronization point — blocks until all processes reach it

# 8. FSDP wrapper
fn FSDP(model: Model, sharding: ShardingStrategy = full,
        mixed_precision: DType = none) -> FSDPModel
    ## Wraps model with Fully Sharded Data Parallel

# 9. DDP wrapper
fn DDP(model: Model, device_ids: list<int> = none) -> DDPModel
    ## Wraps model with Distributed Data Parallel

# 10. Pipeline parallel
fn PipelineParallel(model: Model, devices: list<Device>,
                    chunks: int = 1) -> PPModel
    ## Splits model across devices with pipeline parallelism
```

---

## nsl.export — Model Export

Top 10 public APIs:

```nsl
# 1. Export to ONNX
fn to_onnx(model: Model, input_shape: Shape, output: str,
           opset_version: int = 17) -> void
    ## Export model to ONNX format

# 2. Export to CoreML
fn to_coreml(model: Model, input_shape: Shape, output: str,
             compute_units: str = "all") -> void
    ## Export model to Apple CoreML format

# 3. Export to TFLite
fn to_tflite(model: Model, input_shape: Shape, output: str,
             quantize: bool = false) -> void
    ## Export model to TensorFlow Lite format

# 4. Export to QNF (Quadric Native Format)
fn to_qnf(model: Model, input_shape: Shape, output: str,
           target: NPUTarget = QuadricChimera) -> void
    ## Export quantized model to Quadric Native Format for NPU deployment

# 5. Import from ONNX
fn from_onnx(path: str) -> Model
    ## Import an ONNX model into NSL

# 6. Export to TorchScript
fn to_torchscript(model: Model, input_shape: Shape, output: str) -> void
    ## Export model to PyTorch TorchScript format for interop

# 7. Model summary
fn summary(model: Model, input_shape: Shape) -> ModelSummary
    ## Prints layer-by-layer summary: shapes, parameters, FLOPs

# 8. Trace model
fn trace(model: Model, inputs: Tensor) -> TracedModel
    ## Creates a traced version of the model for export

# 9. Validate export
fn validate_export(original: Model, exported_path: str,
                   test_input: Tensor, tolerance: f64 = 1e-5) -> ValidationResult
    ## Validates exported model produces same outputs as original

# 10. Optimize for inference
fn optimize_inference(model: Model, target: Device = cpu) -> Model
    ## Applies inference optimizations: constant folding, dead code elimination, fusion
```

---

## nsl.bench — Benchmarking and Profiling

Top 10 public APIs:

```nsl
# 1. Benchmark inference throughput
fn benchmark(model: Model, input_shape: Shape, num_iters: int = 100,
             warmup: int = 10) -> BenchmarkResult
    ## Measures inference throughput (tokens/sec, samples/sec, latency)

# 2. Profile memory usage
fn profile_memory(model: Model, input_shape: Shape) -> MemoryProfile
    ## Profiles peak memory, per-layer allocations, fragmentation

# 3. Count FLOPs
fn count_flops(model: Model, input_shape: Shape) -> FLOPCount
    ## Counts floating point operations per forward pass

# 4. Count parameters
fn count_params(model: Model, trainable_only: bool = false) -> int
    ## Counts model parameters (total or trainable only)

# 5. Generate flame graph
fn flame_graph(model: Model, input_shape: Shape, output: str) -> void
    ## Generates an interactive flame graph of execution time

# 6. Memory timeline
fn memory_timeline(model: Model, input_shape: Shape, output: str) -> void
    ## Generates a timeline of memory allocations during forward/backward

# 7. Kernel profiler
fn profile_kernels(model: Model, input_shape: Shape) -> KernelProfile
    ## Profiles individual kernel execution times

# 8. Roofline analysis
fn roofline(model: Model, input_shape: Shape, device: Device) -> RooflineResult
    ## Generates roofline model showing compute vs memory boundedness

# 9. Compare models
fn compare(models: list<(str, Model)>, input_shape: Shape) -> ComparisonTable
    ## Side-by-side comparison of multiple models on speed, memory, FLOPs

# 10. Hardware simulator
fn simulate(model: Model, target: Device, input_shape: Shape) -> SimulationResult
    ## Simulates execution on target hardware (e.g., NPU latency estimation)
```

---

## nsl.compat — Interoperability

Top 10 public APIs:

```nsl
# 1. Convert from PyTorch
fn from_torch(module: py.torch.nn.Module) -> Model
    ## Converts a PyTorch nn.Module to an NSL model

# 2. Convert to PyTorch
fn to_torch(model: Model) -> py.torch.nn.Module
    ## Converts an NSL model to a PyTorch nn.Module

# 3. Load HuggingFace model
fn from_hf(model_id: str, revision: str = "main") -> Model
    ## Loads a model from HuggingFace Hub into NSL

# 4. Load HuggingFace tokenizer
fn tokenizer_from_hf(model_id: str) -> Tokenizer
    ## Loads a HuggingFace tokenizer

# 5. Load safetensors weights
fn load_safetensors(path: str) -> dict<str, Tensor>
    ## Loads weights from safetensors format

# 6. Save safetensors weights
fn save_safetensors(state_dict: dict<str, Tensor>, path: str) -> void
    ## Saves weights in safetensors format

# 7. Python FFI call
fn py.call(module_fn: str, *args, **kwargs) -> PyObject
    ## Call any Python function from NSL

# 8. NumPy interop
fn from_numpy(arr: py.numpy.ndarray) -> Tensor
    ## Zero-copy conversion from NumPy array to NSL tensor

# 9. NumPy interop (reverse)
fn to_numpy(tensor: Tensor) -> py.numpy.ndarray
    ## Zero-copy conversion from NSL tensor to NumPy array

# 10. Load PyTorch checkpoint
fn load_torch_checkpoint(path: str) -> dict<str, Tensor>
    ## Loads a PyTorch .pt or .pth checkpoint (safe deserialization)
```
