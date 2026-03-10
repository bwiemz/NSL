# Section 6 — Quantization as a Language Feature

## Design Rationale

Quantization in PyTorch is an afterthought — a sprawling library of incompatible APIs
(`torch.quantization`, `torch.ao`, `bitsandbytes`, `auto-gptq`, `awq`) with no unified
interface. NSL makes quantization a **native language feature** with the `quant` keyword.
The compiler understands quantized types, validates scale/zero-point consistency, and
generates hardware-specific quantized kernels. Quantization errors surface at compile time.

## Quant DSL Grammar

```ebnf
quant_block     ::= 'quant' '(' quant_config ')' ':' INDENT quant_body DEDENT
quant_config    ::= config_entry (',' config_entry)*
config_entry    ::= 'scheme' '=' quant_scheme
                   | 'mode' '=' quant_mode
                   | 'granularity' '=' granularity
                   | 'calibration' '=' calibration_method
                   | 'target' '=' hardware_target

quant_scheme    ::= 'int8' | 'int4' | 'fp8_e4m3' | 'fp8_e5m2'
                   | 'nf4' | 'nf8' | 'gptq' | 'awq' | 'smoothquant'

quant_mode      ::= 'static'          # post-training, fixed scales
                   | 'dynamic'        # per-batch scale computation
                   | 'aware'          # quantization-aware training
                   | 'weight_only'    # only quantize weights, not activations

granularity     ::= 'per_tensor' | 'per_channel' | 'per_group' '(' 'size' '=' INT ')'
                   | 'per_token'

calibration_method ::= 'minmax' | 'percentile' '(' FLOAT ')' | 'entropy'
                      | 'mse' | 'histogram'

hardware_target ::= 'generic' | 'cuda_sm80' | 'cuda_sm90'
                   | 'npu' '<' IDENT '>'     # e.g., npu<QuadricChimera>

# Layer-level quantization annotation
layer_quant     ::= '@' 'quant' '(' quant_config ')'

# Scale/zero-point types
scale_type      ::= 'Scale' '<' dtype '>'
zeropoint_type  ::= 'ZeroPoint' '<' dtype '>'
```

## Scale and Zero-Point as First-Class Types

```nsl
# Quantized tensor representation
struct QuantizedTensor<Shape, QDtype, SDtype>:
    data: Tensor<Shape, QDtype>           # quantized integer data
    scale: Scale<SDtype>                   # scale factor
    zero_point: ZeroPoint<QDtype>          # zero point offset
    # Dequantized value: (data - zero_point) * scale

# The compiler tracks quantization metadata through operations
let w_q: QuantizedTensor<[768, 3072], int8, fp16>
# Compiler knows:
#   - Storage: int8 (1 byte per element vs 4 for fp32 — 4x compression)
#   - Scale: fp16 (one per channel or per group)
#   - Dequantized type: fp16

# Quantization error is a built-in metric
let error = quant_error(original=w_fp32, quantized=w_q)
# error.mse: mean squared error
# error.snr: signal-to-noise ratio
# error.max_abs: maximum absolute error
```

## 6 Quantization Examples

### Example 1: Post-Training Quantization (Static)

```nsl
## Static PTQ: calibrate scales on a small dataset, then quantize.
## Best for inference deployment when training data is available.

let model = TransformerDecoder.load("pretrained.nslm")

# Calibration dataset — representative inputs for determining quantization ranges
let calib_data = dataset("calibration"):
    source = nsl.data.JSONL("data/calibration.jsonl")
    max_samples = 512    # only need a small subset

# Apply static quantization
let quantized_model = quant(
    scheme=int8,
    mode=static,
    granularity=per_channel,
    calibration=minmax
):
    model = model
    calibration_data = calib_data
    # Quantize weights AND activations to INT8
    quantize_weights = true
    quantize_activations = true

# Verify quantization quality
let original_loss = evaluate_loss(model, test_data)
let quantized_loss = evaluate_loss(quantized_model, test_data)
print(f"degradation: {quantized_loss - original_loss:.4f}")

# Save quantized model — includes scales and zero points
quantized_model.save("model_int8.nslm")
```

### Example 2: Quantization-Aware Training (QAT)

```nsl
## QAT: simulate quantization during training so the model learns to be
## robust to quantization noise. Produces higher quality than PTQ.

let model = TransformerDecoder.load("pretrained.nslm").to(cuda)

# Prepare model for QAT — inserts fake quantization observers
let qat_model = quant(scheme=int8, mode=aware, granularity=per_channel):
    model = model
    # During training, weights and activations pass through fake-quant:
    #   x_fq = dequantize(quantize(x, scale, zp), scale, zp)
    # This simulates quantization error while keeping gradients flowing.

train(model=qat_model, epochs=5, precision=fp32):
    data:
        source = train_dataset
        batch_size = 32

    optimizer: AdamW(lr=1e-5)    # small LR for fine-tuning
    scheduler: WarmupCosine(warmup_steps=200, total_steps=5000)

    step(batch):
        let logits = model.forward(batch.input_ids)
        loss = cross_entropy(logits, batch.labels)

# Convert QAT model to actual quantized model (replace fake-quant with real quant)
let final_model = qat_model.convert()
final_model.save("model_qat_int8.nslm")
```

### Example 3: GPTQ (Post-Training, Weight-Only)

```nsl
## GPTQ: advanced weight-only quantization using approximate second-order
## information. Achieves near-lossless INT4 quantization for large LLMs.

let model = TransformerDecoder.load("llama-70b.nslm")

let calib_data = dataset("calibration"):
    source = nsl.data.JSONL("data/c4_sample.jsonl")
    max_samples = 128

let gptq_model = quant(
    scheme=gptq,
    mode=static,
    granularity=per_group(size=128),    # 128-element groups for INT4
    calibration=mse                      # optimize for minimum MSE
):
    model = model
    # GPTQ-specific settings
    bits = 4                             # 4-bit quantization
    damp_percent = 0.01                  # Hessian damping factor
    block_size = 128                     # columns processed at once
    calibration_data = calib_data
    # Act order: quantize columns in order of decreasing Hessian diagonal
    act_order = true
    # True sequential: quantize layers one at a time, updating subsequent layers
    true_sequential = true

# Report per-layer quantization error
for (name, error) in gptq_model.quant_errors():
    print(f"{name}: MSE={error.mse:.6f}, SNR={error.snr:.1f}dB")

gptq_model.save("llama-70b-gptq-4bit.nslm")
```

### Example 4: Mixed-Precision Quantization

```nsl
## Different layers quantized to different precisions.
## Attention layers are more sensitive — keep at INT8.
## FFN layers are less sensitive — quantize to INT4 for maximum compression.

let model = TransformerDecoder.load("pretrained.nslm")

# Layer-level quantization using @ annotation
model TransformerBlock_MixedQuant(d_model: int, n_heads: int, d_ff: int):
    # Attention: INT8 (more precision-sensitive)
    wq: Linear = Linear(d_model, d_model) @ quant(int8, per_channel)
    wk: Linear = Linear(d_model, d_model) @ quant(int8, per_channel)
    wv: Linear = Linear(d_model, d_model) @ quant(int8, per_channel)
    wo: Linear = Linear(d_model, d_model) @ quant(int8, per_channel)

    # FFN: INT4 (less sensitive, dominates parameter count)
    w_gate: Linear = Linear(d_model, d_ff) @ quant(int4, per_group(size=128))
    w_up: Linear   = Linear(d_model, d_ff) @ quant(int4, per_group(size=128))
    w_down: Linear = Linear(d_ff, d_model) @ quant(int4, per_group(size=128))

    # Norms: keep in fp16 (tiny parameter count, high sensitivity)
    attn_norm: RMSNorm = RMSNorm(d_model)    # not quantized
    ff_norm: RMSNorm = RMSNorm(d_model)      # not quantized

    fn forward(x: Tensor<[batch, seq, d_model], fp16>) -> Tensor<[batch, seq, d_model], fp16>:
        # Quantized layers automatically dequantize during forward pass
        let h = self.attn_norm(x)
        let q = self.wq(h)    # INT8 matmul, dequant to fp16
        # ... rest of forward pass identical to non-quantized version

# Model size comparison:
# Full fp16: 14GB (7B params * 2 bytes)
# Mixed INT4/INT8: ~4.5GB (FFN in INT4, attention in INT8)
```

### Example 5: Hardware-Targeted Quantization (Quadric Chimera)

```nsl
## Quantize specifically for the Quadric Chimera GPNPU, which has:
## - INT4x8 MAC arrays (4-bit weights, 8-bit activations)
## - Dedicated quantization units with hardware scale support
## - Tile-based execution with specific alignment requirements

let model = TransformerDecoder.load("small-lm.nslm")

let chimera_model = quant(
    scheme=int4,
    mode=static,
    granularity=per_group(size=64),     # matches Chimera tile width
    target=npu<QuadricChimera>           # target-specific optimizations
):
    model = model
    calibration_data = calib_data

    # Hardware-specific hints
    weight_bits = 4                      # INT4 weights for MAC arrays
    activation_bits = 8                  # INT8 activations
    tile_alignment = 64                  # align groups to 64-element tiles
    accumulator_bits = 32                # INT32 accumulator in MAC

    # Chimera-specific: asymmetric quantization with hardware scale format
    symmetric = false
    scale_format = chimera_native        # uses Chimera's built-in scale representation

# Export in Quadric's native format
chimera_model.export(
    format=qnf,                          # Quadric Native Format
    output="model_chimera.qnf",
    target=npu<QuadricChimera>,
    optimize_tiling = true                # optimize tensor layout for Chimera tiles
)

# Verify on Chimera simulator
let sim = nsl.bench.ChimeraSimulator()
let latency = sim.profile(chimera_model, input_shape=[1, 2048])
print(f"estimated latency: {latency.total_ms:.1f}ms")
print(f"MAC utilization: {latency.mac_util:.1%}")
```

### Example 6: Quantization Error Feedback Loop

```nsl
## Iterative quantization with error feedback: quantize, measure degradation,
## adjust granularity or precision for problematic layers, repeat.

let model = TransformerDecoder.load("pretrained.nslm")
let eval_data = dataset("eval"):
    source = nsl.data.JSONL("data/eval.jsonl")

# Start with aggressive quantization
let config = QuantConfig(
    scheme=int4,
    granularity=per_group(size=128),
    mode=static
)

# Sensitivity analysis: try quantizing each layer and measure impact
let sensitivity = quant.analyze_sensitivity(model, eval_data, config):
    # For each layer, quantize only that layer and measure perplexity change
    metric = perplexity
    baseline = evaluate_perplexity(model, eval_data)

print("Layer sensitivity ranking:")
for (layer_name, ppl_delta) in sensitivity.ranked():
    print(f"  {layer_name}: +{ppl_delta:.2f} perplexity")

# Build mixed-precision config based on sensitivity
let mixed_config = QuantConfig.mixed():
    # Sensitive layers (>0.5 perplexity increase) → INT8
    for (name, delta) in sensitivity.ranked():
        if delta > 0.5:
            set_layer(name, scheme=int8, granularity=per_channel)
        elif delta > 0.1:
            set_layer(name, scheme=int4, granularity=per_group(size=64))  # finer groups
        else:
            set_layer(name, scheme=int4, granularity=per_group(size=128))  # aggressive

# Apply the mixed config
let final_model = quant(config=mixed_config, mode=static):
    model = model
    calibration_data = calib_data

# Verify
let final_ppl = evaluate_perplexity(final_model, eval_data)
let original_ppl = evaluate_perplexity(model, eval_data)
print(f"perplexity: {original_ppl:.2f} -> {final_ppl:.2f} (+{final_ppl - original_ppl:.2f})")
print(f"model size: {model.size_mb():.0f}MB -> {final_model.size_mb():.0f}MB")
print(f"compression ratio: {model.size_mb() / final_model.size_mb():.1f}x")
```

## Design Tensions & Tradeoffs

1. **Built-in schemes vs Extensibility**: NSL builds in the most common quantization schemes
   (INT4/8, FP8, GPTQ, AWQ). New schemes (future research) require language updates. Mitigation:
   a `CustomQuantScheme` trait lets advanced users define new schemes, but they won't get the
   same compiler optimizations as built-in schemes.

2. **Compile-time vs Runtime scales**: Static quantization uses compile-time scales (fastest
   inference). Dynamic quantization computes scales per-batch (more accurate but slower). The
   `mode` parameter makes this tradeoff explicit.

3. **Hardware-specific code**: The `target=npu<QuadricChimera>` annotation generates
   hardware-specific code. Models quantized for one target may not run on another. The
   `target=generic` default produces portable quantized models at the cost of missing
   hardware-specific optimizations.
