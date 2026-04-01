"""Fixed vocabulary for the NSL-Coder-RL 4096-token BPE tokenizer.

Tokens 0-399 are hand-assigned. Tokens 400-4095 are learned via BPE.
Both build_tokenizer.py and prepare_sft.py import from this file.
"""

# --- Special tokens (0-9) ---
SPECIAL_TOKENS = [
    "<pad>",          # 0
    "<bos>",          # 1
    "<eos>",          # 2
    "<code_start>",   # 3
    "<code_end>",     # 4
    "<error_start>",  # 5
    "<error_end>",    # 6
    "<fix_start>",    # 7
    "<task_start>",   # 8
    "<task_end>",     # 9
]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# --- NSL keywords (10-59) ---
NSL_KEYWORDS = [
    "fn", "let", "const", "model", "train", "return", "if", "else", "elif",
    "for", "while", "break", "continue", "in", "from", "import", "as",
    "true", "false", "and", "or", "not", "none", "pass",
    "grad", "kernel", "quant", "serve", "match", "case",
    "Tensor", "int", "float", "bool", "str", "list", "dict", "tuple",
    "self", "super", "class", "def", "with", "try", "except", "raise",
    "pub", "struct", "impl", "use",
]

# --- NSL builtins (60-139) ---
NSL_BUILTINS = [
    "randn", "zeros", "ones", "full", "arange", "linspace",
    "matmul", "softmax", "relu", "gelu", "silu", "sigmoid", "tanh",
    "exp", "log", "sqrt", "abs", "neg", "sign", "clamp",
    "sum", "mean", "reduce_max", "argmax",
    "reshape", "transpose", "squeeze", "unsqueeze", "contiguous", "expand",
    "embedding_lookup", "layernorm", "rmsnorm", "batchnorm",
    "cross_entropy", "mse_loss", "l1_loss", "bce_loss",
    "dropout", "bias_add", "gather", "scatter_add",
    "conv2d", "maxpool2d", "avgpool2d",
    "to", "cuda", "cpu", "shape", "ndim", "item",
    "print", "clock",
    "model_save", "model_load",
    "load_mmap", "DataLoader",
    "AdamW", "Adam", "SGD", "Lion",
    "warmup_cosine", "cosine_anneal", "linear_decay",
    "RMSNorm", "LayerNorm", "Linear", "Embedding",
    "GroupedQueryAttention", "SwiGLUFFN", "TransformerBlock",
    "byte_tokenizer_new", "tokenizer_encode", "tokenizer_decode",
    "tensor_slice", "causal_mask",
    "scaled_dot_product_attention", "rotate_half",
    "tensor_cos", "tensor_sin", "tensor_cat",
    "zeros_like", "ones_like", "full_like",
]

# --- Operators (140-169) ---
NSL_OPERATORS = [
    "@", "|>", "**", "+=", "-=", "*=", "/=",
    "->", ":", "=", "==", "!=", "<", ">", "<=", ">=",
    "+", "-", "*", "/", "%", ".", ",", ";",
    "(", ")", "[", "]", "{", "}",
]

# --- Structure tokens (170-179) ---
STRUCTURE_TOKENS = [
    "<indent>", "<dedent>", "<newline>",
    "<tab>", "<double_newline>", "<colon_newline>",
    "#", "...", "_", "<hashbang>",
]

# --- Number tokens (180-211) ---
NUMBER_TOKENS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "0.0", "1.0", "0.1", "0.01", "0.001", "0.0001",
    "0.5", "2.0", "0.02", "0.9", "0.95", "0.99",
    "32", "64", "128", "256", "512", "1024", "2048", "4096",
    "49152", "384",
]

# --- Common identifiers (212-399) ---
COMMON_IDENTS = [
    "x", "y", "z", "h", "w", "b", "m", "n", "k", "v", "q",
    "loss", "logits", "pred", "target", "labels", "input_ids",
    "hidden", "output", "result", "out",
    "batch", "seq", "dim", "size", "len", "num",
    "batch_size", "seq_len", "d_model", "d_ff", "n_heads", "n_kv_heads",
    "head_dim", "vocab_size", "num_layers", "dropout_p",
    "weight", "bias", "scale", "gamma", "beta", "eps",
    "lr", "epoch", "step", "param",
    "gate", "up", "down", "proj", "norm",
    "embed", "blocks", "attn", "ffn",
    "w_gate", "w_up", "w_down", "w_q", "w_k", "w_v", "w_o",
    "attn_norm", "ffn_norm", "final_norm",
    "training", "forward", "forward_train",
    "tokens", "loader",
    "running_loss", "total_tokens",
    "optimizer", "scheduler",
    "data", "callbacks",
    "on_step", "on_epoch",
]

def _pad_to(tokens: list[str], target_len: int, prefix: str) -> list[str]:
    while len(tokens) < target_len:
        tokens.append(f"<{prefix}_reserved_{len(tokens)}>")
    return tokens[:target_len]

def build_fixed_vocab() -> list[str]:
    """Return the complete fixed vocabulary (tokens 0-399) as a list."""
    vocab: list[str] = []
    vocab.extend(_pad_to(list(SPECIAL_TOKENS), 10, "special"))    # 0-9
    vocab.extend(_pad_to(list(NSL_KEYWORDS), 50, "kw"))           # 10-59
    vocab.extend(_pad_to(list(NSL_BUILTINS), 80, "builtin"))      # 60-139
    vocab.extend(_pad_to(list(NSL_OPERATORS), 30, "op"))          # 140-169
    vocab.extend(_pad_to(list(STRUCTURE_TOKENS), 10, "struct"))   # 170-179
    vocab.extend(_pad_to(list(NUMBER_TOKENS), 32, "num"))         # 180-211
    vocab.extend(_pad_to(list(COMMON_IDENTS), 188, "ident"))      # 212-399
    assert len(vocab) == 400, f"Fixed vocab should be 400, got {len(vocab)}"
    return vocab

VOCAB_SIZE = 4096
FIXED_VOCAB_SIZE = 400
BPE_VOCAB_SIZE = VOCAB_SIZE - FIXED_VOCAB_SIZE  # 3696
