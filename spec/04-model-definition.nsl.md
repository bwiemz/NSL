# Section 4 — Model Definition

## Design Rationale

In PyTorch, models are Python classes inheriting from `nn.Module` — a pattern that mixes
general-purpose OOP with ML-specific concerns (parameter registration, device placement,
serialization). NSL introduces `model` as a first-class keyword that declares a self-contained
neural network component. The compiler knows what a model is, enabling automatic parameter
discovery, native serialization (no pickle), and compile-time verification of layer compatibility.

## Model Block Grammar

```ebnf
model_def       ::= 'model' IDENT ('<' type_params '>')? ('(' model_params ')')? ':' INDENT model_body DEDENT
model_params    ::= param (',' param)*
model_body      ::= (model_member NEWLINE)*
model_member    ::= layer_decl | param_decl | buffer_decl | submodel_decl | method_def | forward_def
layer_decl      ::= IDENT ':' layer_type ('=' layer_init)? ('@' quant_annotation)?
layer_type      ::= 'Linear' | 'Attention' | 'Conv2d' | 'Conv1d' | 'LayerNorm' | 'RMSNorm'
                   | 'Embedding' | 'MLP' | 'Dropout' | 'RotaryEmbed' | 'MoERouter'
                   | 'BatchNorm' | 'GroupNorm' | 'MultiHeadAttention' | IDENT
layer_init      ::= layer_type '(' arg_list ')'
param_decl      ::= IDENT ':' 'Param' '<' shape ',' dtype '>' '=' init_expr
buffer_decl     ::= IDENT ':' 'Buffer' '<' shape ',' dtype '>' '=' init_expr
submodel_decl   ::= IDENT ':' IDENT '=' IDENT '(' arg_list ')'          # nested model
forward_def     ::= 'fn' 'forward' '(' param_list ')' ('->' type)? ':' INDENT block DEDENT
method_def      ::= 'fn' IDENT '(' param_list ')' ('->' type)? ':' INDENT block DEDENT

# Initialization expressions
init_expr       ::= 'init.xavier' '(' shape ')' | 'init.kaiming' '(' shape ')'
                   | 'init.normal' '(' shape ',' 'std' '=' FLOAT ')'
                   | 'init.zeros' '(' shape ')' | 'init.ones' '(' shape ')'
                   | 'init.uniform' '(' shape ',' FLOAT ',' FLOAT ')'
```

## Built-in Layer Types

| Layer           | Constructor Signature                                                | Description                          |
|-----------------|----------------------------------------------------------------------|--------------------------------------|
| `Linear`        | `Linear(in, out, bias=true)`                                         | Fully connected layer                |
| `Attention`     | `Attention(d_model, n_heads, dropout=0.0, causal=false)`             | Multi-head attention                 |
| `Conv2d`        | `Conv2d(in_ch, out_ch, kernel, stride=1, padding=0)`                 | 2D convolution                       |
| `Conv1d`        | `Conv1d(in_ch, out_ch, kernel, stride=1, padding=0)`                 | 1D convolution                       |
| `LayerNorm`     | `LayerNorm(dim)`                                                     | Layer normalization                  |
| `RMSNorm`       | `RMSNorm(dim, eps=1e-6)`                                            | Root mean square normalization       |
| `BatchNorm`     | `BatchNorm(num_features, momentum=0.1)`                              | Batch normalization                  |
| `GroupNorm`     | `GroupNorm(num_groups, num_channels)`                                 | Group normalization                  |
| `Embedding`     | `Embedding(vocab_size, dim)`                                         | Token embedding lookup               |
| `MLP`           | `MLP(in, hidden, out, activation=gelu)`                              | Two-layer feedforward                |
| `Dropout`       | `Dropout(p=0.1)`                                                     | Dropout regularization               |
| `RotaryEmbed`   | `RotaryEmbed(dim, max_seq=2048, base=10000)`                         | Rotary positional embeddings (RoPE)  |
| `MoERouter`     | `MoERouter(d_model, num_experts, top_k=2, capacity_factor=1.25)`     | Mixture of Experts routing           |

## Model Features

### Parameter Tying

```nsl
model LanguageModel(vocab_size: int, d_model: int):
    embed: Embedding = Embedding(vocab_size, d_model)
    # ... transformer layers ...

    # Tie output projection weights to embedding weights
    lm_head: Linear = Linear(d_model, vocab_size, bias=false)
        @tie_weights(self.embed.weight)    # shares the same parameter tensor

    fn forward(input_ids: Tensor<[batch, seq], int32>) -> Tensor<[batch, seq, vocab_size], fp32>:
        let h = self.embed(input_ids)
        # ... transformer forward ...
        return self.lm_head(h)
```

### Weight Sharing Between Sub-models

```nsl
model SiameseNetwork(backbone: ResNet):
    # Both branches share the same backbone weights
    branch_a: ResNet = backbone
    branch_b: ResNet = backbone    # same reference, shared parameters

    fn forward(x1: Tensor, x2: Tensor) -> (Tensor, Tensor):
        return (self.branch_a(x1), self.branch_b(x2))
```

### Lazy Initialization

```nsl
model LazyLinear:
    # Shape is inferred from the first forward pass
    weight: Param<[_, out_features], fp32> = init.lazy()

    fn forward(x: Tensor<[batch, in_features], fp32>) -> Tensor<[batch, out_features], fp32>:
        if not self.weight.is_materialized():
            self.weight = init.kaiming([x.shape[-1], self.out_features])
        return x @ self.weight
```

### Native Serialization

```nsl
# Save — binary format, no pickle, includes architecture + weights + metadata
model.save("checkpoint.nslm")

# Save with metadata
model.save("checkpoint.nslm", metadata={
    "epoch": 10,
    "loss": 0.023,
    "git_hash": "abc123"
})

# Load — type-safe, the compiler verifies the loaded model matches the declared type
let loaded: TransformerLM = TransformerLM.load("checkpoint.nslm")

# Partial load (e.g., load pretrained backbone, skip head)
let backbone = TransformerLM.load("checkpoint.nslm", keys=["encoder.*"])

# Format is:
# - Header: magic bytes, version, architecture hash
# - Metadata: JSON-encoded training info
# - Parameters: tensor data in a flat binary format with an index table
# - No arbitrary code execution (unlike pickle)
```

## Complete Transformer Decoder Block

```nsl
## A full transformer decoder block implementing:
## - Multi-head self-attention with rotary embeddings and KV-cache
## - RMSNorm pre-normalization (LLaMA-style)
## - SwiGLU feedforward network
## - Residual connections

model TransformerBlock(d_model: int, n_heads: int, d_ff: int, dropout: f32 = 0.0):
    # Pre-norm architecture (more stable training than post-norm)
    attn_norm: RMSNorm = RMSNorm(d_model)
    ff_norm: RMSNorm = RMSNorm(d_model)

    # Multi-head self-attention with rotary position embeddings
    wq: Linear = Linear(d_model, d_model, bias=false)
    wk: Linear = Linear(d_model, d_model, bias=false)
    wv: Linear = Linear(d_model, d_model, bias=false)
    wo: Linear = Linear(d_model, d_model, bias=false)
    rotary: RotaryEmbed = RotaryEmbed(d_model // n_heads)
    attn_drop: Dropout = Dropout(dropout)

    # SwiGLU feedforward: gate * swish(up) then down-project
    w_gate: Linear = Linear(d_model, d_ff, bias=false)
    w_up: Linear = Linear(d_model, d_ff, bias=false)
    w_down: Linear = Linear(d_ff, d_model, bias=false)
    ff_drop: Dropout = Dropout(dropout)

    # Constants
    n_heads: int = n_heads
    head_dim: int = d_model // n_heads

    fn forward(
        x: Tensor<[batch, seq, d_model], bf16>,
        mask: Tensor<[batch, 1, seq, seq], bool> = none,
        kv_cache: KVCache = none
    ) -> (Tensor<[batch, seq, d_model], bf16>, KVCache):

        # === Self-Attention with Pre-Norm ===
        let residual = x
        let h = self.attn_norm(x)

        # Project to Q, K, V and reshape for multi-head
        let q = self.wq(h).reshape([batch, seq, self.n_heads, self.head_dim]).transpose(1, 2)
        let k = self.wk(h).reshape([batch, seq, self.n_heads, self.head_dim]).transpose(1, 2)
        let v = self.wv(h).reshape([batch, seq, self.n_heads, self.head_dim]).transpose(1, 2)

        # Apply rotary positional embeddings
        let (q, k) = self.rotary(q, k, offset=kv_cache.seq_len() if kv_cache else 0)

        # Update KV cache for autoregressive generation
        let new_cache = if kv_cache:
            let (cached_k, cached_v) = kv_cache.get()
            let k = cat([cached_k, k], dim=-2)    # append new keys
            let v = cat([cached_v, v], dim=-2)    # append new values
            KVCache.update(k, v)
        else:
            KVCache.new(k, v)

        # Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt()
        let scores = (q @ k.transpose(-2, -1)) / scale

        if mask is not none:
            scores = scores.masked_fill(mask == false, -1e9)

        let attn_weights = softmax(scores, dim=-1)
        let attn_weights = self.attn_drop(attn_weights)
        let attn_out = attn_weights @ v

        # Reshape back and project
        let attn_out = attn_out.transpose(1, 2).reshape([batch, seq, d_model])
        let attn_out = self.wo(attn_out)

        # Residual connection
        let x = residual + attn_out

        # === Feedforward with Pre-Norm (SwiGLU) ===
        let residual = x
        let h = self.ff_norm(x)

        # SwiGLU: gate mechanism with swish activation
        let gate = self.w_gate(h).silu()    # swish = silu
        let up = self.w_up(h)
        let ff_out = self.w_down(gate * up)
        let ff_out = self.ff_drop(ff_out)

        # Residual connection
        let x = residual + ff_out

        return (x, new_cache)


## Full decoder model stacking multiple transformer blocks
model TransformerDecoder(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    max_seq: int = 2048,
    dropout: f32 = 0.0
):
    embed: Embedding = Embedding(vocab_size, d_model)
    layers: list<TransformerBlock> = [
        TransformerBlock(d_model, n_heads, d_ff, dropout)
        for _ in 0..n_layers
    ]
    norm: RMSNorm = RMSNorm(d_model)
    lm_head: Linear = Linear(d_model, vocab_size, bias=false)
        @tie_weights(self.embed.weight)

    fn forward(
        input_ids: Tensor<[batch, seq], int32>,
        kv_caches: list<KVCache> = none
    ) -> (Tensor<[batch, seq, vocab_size], bf16>, list<KVCache>):

        let h = self.embed(input_ids)
        let new_caches = []

        # Causal mask for autoregressive attention
        let mask = causal_mask(seq, device=h.device)

        for (i, layer) in self.layers.enumerate():
            let cache = kv_caches[i] if kv_caches else none
            let (h, new_cache) = layer.forward(h, mask=mask, kv_cache=cache)
            new_caches.append(new_cache)

        let h = self.norm(h)
        let logits = self.lm_head(h)

        return (logits, new_caches)

    ## Generate text autoregressively
    @no_grad
    fn generate(
        prompt: Tensor<[1, prompt_len], int32>,
        max_tokens: int = 100,
        temperature: f32 = 1.0,
        top_k: int = 50
    ) -> Tensor<[1, _], int32>:
        let tokens = prompt
        let caches: list<KVCache> = none

        for _ in 0..max_tokens:
            let input = if caches then tokens[:, -1:] else tokens
            let (logits, caches) = self.forward(input, kv_caches=caches)
            let next_logit = logits[:, -1, :] / temperature

            # Top-k sampling
            let (top_vals, top_idx) = next_logit.topk(top_k, dim=-1)
            let probs = softmax(top_vals, dim=-1)
            let sampled = probs.multinomial(1)
            let next_token = top_idx.gather(-1, sampled)

            tokens = cat([tokens, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == EOS_TOKEN:
                break

        return tokens
```

## Conditional Layer Execution

```nsl
# Layers can be conditionally executed based on configuration or runtime values

model AdaptiveTransformer(d_model: int, n_heads: int, d_ff: int, use_moe: bool = false):
    attn: Attention = Attention(d_model, n_heads)
    norm1: RMSNorm = RMSNorm(d_model)
    norm2: RMSNorm = RMSNorm(d_model)

    # Conditional layer: only allocated if use_moe is true
    ff: MLP = if not use_moe then MLP(d_model, d_ff, d_model) else none
    moe: MoEBlock = if use_moe then MoEBlock(d_model, d_ff, num_experts=8) else none

    fn forward(x: Tensor) -> Tensor:
        let h = x + self.attn(self.norm1(x))
        if self.use_moe:
            return h + self.moe(self.norm2(h))
        else:
            return h + self.ff(self.norm2(h))
```

## Dynamic Architecture (Mixture of Experts)

```nsl
model MoEBlock(d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
    router: MoERouter = MoERouter(d_model, num_experts, top_k=top_k)
    experts: list<MLP> = [
        MLP(d_model, d_ff, d_model) for _ in 0..num_experts
    ]

    fn forward(x: Tensor<[batch, seq, d_model], bf16>) -> Tensor<[batch, seq, d_model], bf16>:
        # Router computes expert assignments and weights
        let (expert_indices, expert_weights, aux_loss) = self.router(x)
        # expert_indices: [batch, seq, top_k] — which experts to use
        # expert_weights: [batch, seq, top_k] — how much to weight each expert

        # Dispatch tokens to experts
        let output = zeros_like(x)
        for k in 0..self.top_k:
            for (e, expert) in self.experts.enumerate():
                # Mask: which tokens go to this expert at this rank
                let mask = expert_indices[:, :, k] == e
                if mask.any():
                    let expert_input = x[mask]
                    let expert_output = expert(expert_input)
                    output[mask] += expert_weights[:, :, k][mask].unsqueeze(-1) * expert_output

        return output

    fn routing_loss() -> Tensor<[], fp32>:
        ## Auxiliary load-balancing loss to prevent expert collapse
        return self.router.load_balance_loss()
```

## Device Placement

```nsl
# Device placement is declared once at model instantiation — all parameters
# and buffers are placed on the specified device.

let model = TransformerDecoder(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072
).to(cuda)    # entire model on GPU

# Multi-GPU placement with pipeline parallelism
let model = TransformerDecoder(...).distribute(
    strategy=pipeline,
    devices=[cuda(0), cuda(1), cuda(2), cuda(3)],
    split_at=[3, 6, 9]    # split layers across 4 GPUs
)

# Per-layer device placement (manual)
model.layers[0..3].to(cuda(0))
model.layers[3..6].to(cuda(1))
model.layers[6..9].to(cuda(2))
model.layers[9..12].to(cuda(3))
```

## Design Tensions & Tradeoffs

1. **`model` keyword vs classes**: A dedicated `model` keyword means models aren't general-purpose
   objects — you can't inherit from arbitrary classes or mix in unrelated behavior. This is
   intentional: it keeps models serializable, inspectable, and optimizable. For general-purpose
   OOP, NSL has `struct` and `trait`.

2. **Layer list syntax**: `list<TransformerBlock>` with a comprehension is convenient but means
   all layers share constructor arguments. For heterogeneous layer stacks, users must construct
   each layer individually. This matches how most real models work.

3. **KV-cache as a return value**: Returning KV-cache from `forward` makes the API stateless
   (each call is pure), which is better for compilation and parallelism. The alternative
   (stateful KV-cache inside the model) would be simpler for users but harder to optimize.

4. **`@tie_weights`**: Weight tying is a compile-time directive. The compiler ensures tied
   parameters always point to the same memory and that gradients are accumulated correctly.
   This is more robust than PyTorch's manual `weight = other.weight` assignment.
