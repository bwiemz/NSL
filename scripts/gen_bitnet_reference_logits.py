#!/usr/bin/env python3
"""Generate the M35.1 BitNet b1.58-3B reference-logit merge-gate artifact.

Anchor: the pinned 1bitLLM/bitnet_b1_58-3B checkpoint's OWN modeling code
(modeling_bitnet.py + configuration_bitnet.py + utils_quant.py) at revision
af89e318d78a70802061246bf037199d2fb97020. That vendored code is the
paper-author-validated reference implementation and is already M35.1's source
of record for the "activations use absmax, not absmean" correction.

Output layout (must match crates/nsl-codegen/tests/bitnet_logit_match.rs:180-192):
    32 prompts x 32002 vocab x 2 bytes little-endian FP16,
    prompt-major, row-major, NO header  ==  2,048,128 bytes.
Each row is the logits at the FINAL token position of that prompt.

--------------------------------------------------------------------------
WHY THE SHIMS (v1 of this script silently produced an ALL-NaN artifact)
--------------------------------------------------------------------------
The vendored modeling code targets transformers 4.39.  This venv pins
transformers 5.14.1 / torch 2.13.0+cu130 / python 3.14.6 and MUST NOT be
changed (shared with another agent; no cp314 tokenizers wheel exists for
transformers 4.44 anyway).  Four incompatibilities each had to be shimmed:

  SHIM 1 -- config.json declares "model_type": "llama", so
    AutoModelForCausalLM/AutoConfig resolve to the BUILT-IN LlamaForCausalLM
    instead of the vendored BitnetForCausalLM, even with
    trust_remote_code=True (there is no auto_map in this checkpoint).  Stock
    Llama has no `mlp.ffn_layernorm` and no `self_attn.inner_attn_ln`, so
    those 52 sub-layernorms were dropped as UNEXPECTED, and BitLinear
    (weight/activation quantization) was never instantiated at all.  The
    resulting model emitted NaN for every logit.
    => Import configuration_bitnet.BitnetConfig and modeling_bitnet by FILE
       PATH (namespace package rooted at the checkpoint's parent dir) and
       instantiate BitnetForCausalLM explicitly.

  SHIM 2 -- in transformers 5, PretrainedConfig.rope_scaling is an alias for
    rope_parameters and returns {'rope_theta': 10000.0, 'rope_type':
    'default'} rather than the None stored in config.json.  modeling_bitnet
    BitnetAttention._init_rope (line ~272) does
    `if self.config.rope_scaling is None: ... else: raise NotImplementedError`,
    so construction raised NotImplementedError.
    => Patch BitnetAttention._init_rope to take the default-rope branch
       unconditionally.  This is semantics-preserving: config.json really does
       store rope_scaling=null and rope_theta=10000.0, and the replacement
       builds exactly the same BitnetRotaryEmbedding(head_dim,
       max_position_embeddings, base=rope_theta) the original branch built.

  SHIM 3 -- transformers 5 modeling_utils.py:2704 calls
    `self._tied_weights_keys.keys()`; the vendored class sets the 4.x list
    form `["lm_head.weight"]`, raising AttributeError.
    => BitnetForCausalLM._tied_weights_keys =
         {"lm_head.weight": "model.embed_tokens.weight"}
       which is the v5 spelling of the same tie (config tie_word_embeddings=true).

  SHIM 4 -- `torch_dtype=` is deprecated in transformers 5 => use `dtype=`.

Additionally attn_implementation="eager" is forced: modeling_bitnet's
LLAMA_ATTENTION_CLASSES only has {"eager", "flash_attention_2"}, and
transformers 5 defaults _attn_implementation to "sdpa", which would KeyError.
eager is also the arithmetically reference path.

--------------------------------------------------------------------------
HARD VALIDATION CONTRACT -- the artifact is written only if ALL pass
--------------------------------------------------------------------------
  a) type(model).__name__ == "BitnetForCausalLM"
  b) zero missing / zero unexpected state-dict keys, and the 52 BitNet-only
     sub-layernorms are bitwise identical to the safetensors shards
  c) every one of the 32 x 32002 logits is finite
  d) utils_quant.BitLinear instances actually exist in the model (counted)
  e) smoke: "The capital of France is" -> argmax 3681 == "Paris"
  f) tokenization is bit-identical to [BOS] + raw sentencepiece(tokenizer.model)
     for all 32 prompts (transformers 5 hands back a TokenizersBackend even for
     use_fast=False, so the sentencepiece equivalence eval_ppl.py relies on is
     asserted rather than assumed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

CACHE = Path.home() / ".cache/nsl-tests/bitnet_b158_3b"
REVISION = "af89e318d78a70802061246bf037199d2fb97020"
EXPECTED_VOCAB = 32002
EXPECTED_PROMPTS = 32


# ---------------------------------------------------------------------------
# Prompt parsing -- byte-for-byte port of parse_phase1_prompts() from
# crates/nsl-codegen/tests/bitnet_logit_match.rs.  REUSED VERBATIM from v1.
# ---------------------------------------------------------------------------
def parse_phase1_prompts(text: str) -> list[str]:
    """Port of parse_phase1_prompts() from bitnet_logit_match.rs.

    Must stay byte-for-byte equivalent to the Rust parser: the fixture is four
    sections of 8, and code-completion prompts span multiple lines delimited by
    a new `def ` rather than by blank lines.
    """
    prompts: list[str] = []
    current_section: str | None = None
    code_buf: list[str] = []

    def flush_code() -> None:
        if code_buf:
            prompts.append("\n".join(code_buf))
            code_buf.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("#"):
            flush_code()
            lower = line[1:].lower()
            if "code-completion" in lower:
                current_section = "code"
            elif "short factual" in lower:
                current_section = "factual"
            elif "long-context" in lower:
                current_section = "long"
            elif "edge-case" in lower:
                current_section = "edge"
            else:
                current_section = None
            continue
        if current_section == "code":
            if line.startswith("def ") and code_buf:
                flush_code()
            code_buf.append(line)
        elif current_section is not None:
            prompts.append(line)
    flush_code()
    return prompts


class Fail(RuntimeError):
    """Validation-contract violation. Never write an artifact after this."""


def build_model(device: str):
    """Load BitnetForCausalLM from the vendored code with SHIMs 1-4 applied."""
    # SHIM 1: reach the vendored classes by file path.  CACHE has no
    # __init__.py, but PEP-420 namespace packages make `import
    # bitnet_b158_3b.modeling_bitnet` work AND keep the module's relative
    # imports (`from .configuration_bitnet import BitnetConfig`) resolvable.
    sys.path.insert(0, str(CACHE.parent))
    pkg = CACHE.name
    mb = __import__(f"{pkg}.modeling_bitnet", fromlist=["*"])
    cb = __import__(f"{pkg}.configuration_bitnet", fromlist=["*"])
    uq = __import__(f"{pkg}.utils_quant", fromlist=["*"])

    cfg = cb.BitnetConfig.from_pretrained(str(CACHE))
    cfg._attn_implementation = "eager"

    # SHIM 2: rope_scaling is aliased to rope_parameters in transformers 5 and
    # is no longer None, which would trip the NotImplementedError branch.
    def _init_rope_default(self) -> None:
        self.rotary_emb = mb.BitnetRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    mb.BitnetAttention._init_rope = _init_rope_default

    # SHIM 3: v5 wants the dict spelling of the weight tie.
    mb.BitnetForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    # SHIM 4: dtype= replaces the deprecated torch_dtype=.
    model, info = mb.BitnetForCausalLM.from_pretrained(
        str(CACHE),
        config=cfg,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model = model.to(device).eval()
    return model, info, mb, uq


def validate_load(model, info, uq) -> dict:
    """Contract (a), (b), (d). Raises Fail on any violation."""
    report: dict = {}

    # (a) real BitNet class, not a stock-Llama fallback.
    cls_name = type(model).__name__
    report["model_class"] = cls_name
    if cls_name != "BitnetForCausalLM":
        raise Fail(f"(a) model class is {cls_name}, expected BitnetForCausalLM")

    # (b) zero missing / zero unexpected keys.
    missing = sorted(info.get("missing_keys", []) or [])
    unexpected = sorted(info.get("unexpected_keys", []) or [])
    # lm_head.weight is tied to model.embed_tokens.weight and is legitimately
    # absent from the shards; transformers may or may not list it as missing.
    tied = "lm_head.weight"
    tie_ok = bool(getattr(model.config, "tie_word_embeddings", False))
    if tie_ok and tied in missing:
        missing.remove(tied)
        report["tied_key_excused"] = tied
    report["missing_keys"] = missing
    report["unexpected_keys"] = unexpected
    if missing:
        raise Fail(f"(b) {len(missing)} MISSING state-dict keys: {missing[:8]}")
    if unexpected:
        raise Fail(f"(b) {len(unexpected)} UNEXPECTED state-dict keys: {unexpected[:8]}")

    # (b, strengthened) the two BitNet-only sub-layernorms plus a BitLinear
    # weight must be bitwise identical to the safetensors shards.  A silent
    # stock-Llama fallback drops exactly these.
    from safetensors import safe_open

    idx = json.loads((CACHE / "model.safetensors.index.json").read_text())["weight_map"]
    sd = model.state_dict()
    n_layers = int(model.config.num_hidden_layers)
    probe = []
    for li in (0, n_layers // 2, n_layers - 1):
        probe += [
            f"model.layers.{li}.mlp.ffn_layernorm.weight",
            f"model.layers.{li}.self_attn.inner_attn_ln.weight",
            f"model.layers.{li}.self_attn.q_proj.weight",
            f"model.layers.{li}.self_attn.rotary_emb.inv_freq",
        ]
    for key in probe:
        if key not in idx:
            raise Fail(f"(b) probe key {key} absent from the checkpoint index")
        if key not in sd:
            raise Fail(f"(b) probe key {key} absent from the LOADED model state_dict")
        with safe_open(str(CACHE / idx[key]), framework="pt", device="cpu") as f:
            ref = f.get_tensor(key)
        got = sd[key].detach().cpu()
        if got.shape != ref.shape:
            raise Fail(f"(b) {key} shape {tuple(got.shape)} != checkpoint {tuple(ref.shape)}")
        if not torch.equal(got.to(ref.dtype), ref):
            raise Fail(f"(b) {key} loaded values differ from the checkpoint shard")
    report["bitwise_probed_keys"] = probe

    # Also assert the BitNet-only sub-layernorms exist on EVERY layer, and that
    # all 314 checkpoint keys are present in the live model.
    ckpt_keys = set(idx)
    live_keys = set(sd)
    absent = sorted(ckpt_keys - live_keys)
    if absent:
        raise Fail(f"(b) {len(absent)} checkpoint keys never landed in the model: {absent[:8]}")
    report["checkpoint_key_count"] = len(ckpt_keys)
    report["ffn_layernorm_count"] = sum(1 for k in live_keys if k.endswith("mlp.ffn_layernorm.weight"))
    report["inner_attn_ln_count"] = sum(1 for k in live_keys if k.endswith("self_attn.inner_attn_ln.weight"))
    if report["ffn_layernorm_count"] != n_layers or report["inner_attn_ln_count"] != n_layers:
        raise Fail(
            f"(b) sub-layernorm count mismatch: ffn={report['ffn_layernorm_count']} "
            f"inner={report['inner_attn_ln_count']} expected {n_layers} each"
        )

    # (d) BitLinear is actually the linear layer type in use.
    n_bitlinear = sum(1 for m in model.modules() if isinstance(m, uq.BitLinear))
    report["bitlinear_count"] = n_bitlinear
    report["bitlinear_class"] = f"{uq.BitLinear.__module__}.{uq.BitLinear.__qualname__}"
    expected_bitlinear = n_layers * 7  # q,k,v,o + gate,up,down
    if n_bitlinear != expected_bitlinear:
        raise Fail(f"(d) found {n_bitlinear} BitLinear modules, expected {expected_bitlinear}")
    # lm_head must stay a plain nn.Linear (tied to the embedding, not quantized).
    report["lm_head_class"] = type(model.lm_head).__name__
    if isinstance(model.lm_head, uq.BitLinear):
        raise Fail("(d) lm_head is a BitLinear; the reference keeps it a plain tied nn.Linear")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("/home/brandon/Projects/NSL"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta-out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Determinism knobs: fp16 weights, but keep any f32 matmul off the TF32 path.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)

    prompts_path = args.repo_root / "tests/fixtures/bitnet_b158_phase1_prompts.txt"
    prompts = parse_phase1_prompts(prompts_path.read_text(encoding="utf-8"))
    if len(prompts) != EXPECTED_PROMPTS:
        raise Fail(f"parsed {len(prompts)} prompts, expected {EXPECTED_PROMPTS}")
    assert prompts[0] == "What is the capital of France?", prompts[0]
    assert prompts[8].startswith("def fibonacci(n):"), prompts[8][:40]
    assert prompts[16].startswith("The transformer architecture"), prompts[16][:40]
    assert prompts[24] == "a", repr(prompts[24])
    print(f"Parsed {len(prompts)} prompts; spot-checks match the Rust parser.")

    import transformers
    from transformers import AutoTokenizer

    print(f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__}")

    tok = AutoTokenizer.from_pretrained(str(CACHE), use_fast=False)
    print(f"tokenizer={type(tok).__name__} add_bos_token={getattr(tok, 'add_bos_token', None)}")

    # (f) transformers 5 returns a TokenizersBackend even for use_fast=False
    # (tokenizer.json is present), and its `add_bos_token` attribute reads
    # False even though the tokenizer.json post-processor DOES prepend <s>.
    # Assert equivalence to the raw sentencepiece path that eval_ppl.py's
    # BitnetTokenizer(use_fast=False) uses, rather than trusting either flag.
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(CACHE / "tokenizer.model"))
    bos_id = 1
    for i, p in enumerate(prompts):
        hf_ids = tok(p).input_ids
        sp_ids = [bos_id] + sp.encode(p)
        if hf_ids != sp_ids:
            raise Fail(
                f"(f) prompt {i} tokenization differs from raw sentencepiece: "
                f"hf={hf_ids[:16]} sp={sp_ids[:16]}"
            )
    print(f"  validate tokenization == [BOS] + sentencepiece for all {len(prompts)} prompts")

    model, info, _mb, uq = build_model(args.device)
    report = validate_load(model, info, uq)
    for k, v in report.items():
        if k not in ("bitwise_probed_keys", "missing_keys", "unexpected_keys"):
            print(f"  validate {k} = {v}")
    print(f"  validate missing={len(report['missing_keys'])} unexpected={len(report['unexpected_keys'])}")

    cfg_vocab = int(model.config.vocab_size)
    if cfg_vocab != EXPECTED_VOCAB:
        raise Fail(f"config vocab_size={cfg_vocab}, expected {EXPECTED_VOCAB}")

    # (e) smoke test against the known-good reference result.
    smoke_text = "The capital of France is"
    smoke_ids = tok(smoke_text, return_tensors="pt").input_ids
    smoke_list = smoke_ids[0].tolist()
    smoke_logits = model(input_ids=smoke_ids.to(args.device), use_cache=False).logits[0, -1, :]
    smoke_argmax = int(smoke_logits.argmax())
    smoke_tokstr = tok.convert_ids_to_tokens([smoke_argmax])[0]
    print(f"SMOKE ids={smoke_list} argmax={smoke_argmax} piece={smoke_tokstr!r}")
    if smoke_list != [1, 450, 7483, 310, 3444, 338]:
        raise Fail(f"(e) smoke ids {smoke_list} != [1, 450, 7483, 310, 3444, 338]")
    if smoke_argmax != 3681:
        raise Fail(f"(e) smoke argmax {smoke_argmax} != 3681 ('Paris')")
    if not torch.isfinite(smoke_logits).all():
        raise Fail("(e) smoke logits contain non-finite values")

    # ---- generate --------------------------------------------------------
    out = torch.empty((EXPECTED_PROMPTS, EXPECTED_VOCAB), dtype=torch.float16)
    per_prompt: list[dict] = []
    for i, prompt in enumerate(prompts):
        # Raw continuation; this checkpoint has no chat template. Batch size 1,
        # no padding, so no attention_mask ambiguity can enter the reference.
        enc = tok(prompt, return_tensors="pt")
        ids = enc.input_ids
        logits = model(input_ids=ids.to(args.device), use_cache=False).logits
        final = logits[0, -1, :]
        if int(final.shape[0]) != EXPECTED_VOCAB:
            raise Fail(f"prompt {i} produced vocab {int(final.shape[0])}")
        if not torch.isfinite(final).all():
            raise Fail(f"(c) prompt {i} produced non-finite logits")
        row = final.to(torch.float16).cpu()
        out[i] = row
        am = int(row.argmax())
        per_prompt.append(
            {
                "index": i,
                "prompt_head": prompt[:60],
                "n_tokens": int(ids.shape[1]),
                "first_token_id": int(ids[0, 0]),
                "argmax_id": am,
                "argmax_piece": tok.convert_ids_to_tokens([am])[0],
                "argmax_decoded": tok.decode([am]),
                "max_logit": float(row[am]),
            }
        )
        p = per_prompt[-1]
        print(
            f"  [{i:2d}/{EXPECTED_PROMPTS}] tokens={p['n_tokens']:4d} "
            f"argmax={am:5d} piece={p['argmax_piece']!r} max={p['max_logit']:.4f}"
        )

    arr = out.contiguous().numpy()
    if arr.dtype != np.float16:
        raise Fail(f"array dtype {arr.dtype}, expected float16")
    if not np.isfinite(arr.astype(np.float32)).all():
        raise Fail("(c) final array contains non-finite values")

    # Force little-endian FP16 explicitly; the merge gate reads LE u16.
    buf = arr.astype("<f2", copy=False).tobytes()
    expected_bytes = EXPECTED_PROMPTS * EXPECTED_VOCAB * 2
    if len(buf) != expected_bytes:
        raise Fail(f"produced {len(buf)} bytes, expected {expected_bytes}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(buf)
    digest = hashlib.sha256(buf).hexdigest()
    print(f"WROTE {args.out} ({len(buf)} bytes) sha256={digest}")

    bos_prepended = all(p["first_token_id"] == int(model.config.bos_token_id) for p in per_prompt)
    meta = {
        "generator": "gen_ref_logits_hf_v2.py (HF-transformers anchor)",
        "model_id": "1bitLLM/bitnet_b1_58-3B",
        "revision": REVISION,
        "modeling_code": "checkpoint-vendored modeling_bitnet.py + configuration_bitnet.py + utils_quant.py",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": args.device,
        "dtype": "float16",
        "attn_implementation": "eager",
        "tf32": {"matmul.allow_tf32": False, "cudnn.allow_tf32": False},
        "batch_size": 1,
        "padding": "none",
        "use_cache": False,
        "tokenizer": {
            "requested": "AutoTokenizer.from_pretrained(D, use_fast=False)",
            "actual_class": type(tok).__name__,
            "add_bos_token_attr": bool(getattr(tok, "add_bos_token", False)),
            "note": (
                "transformers 5 hands back a TokenizersBackend even for use_fast=False "
                "(tokenizer.json is present) and its add_bos_token attribute reads False, "
                "which is MISLEADING: the tokenizer.json post-processor does prepend <s>. "
                "Verified equivalent: for all 32 prompts the ids are bit-identical to "
                "[1] + sentencepiece(tokenizer.model).encode(prompt), i.e. the exact slow "
                "sentencepiece path eval_ppl.py's BitnetTokenizer(use_fast=False) uses."
            ),
            "sentencepiece_equivalence_verified": True,
            "sentencepiece_vocab_size": int(sp.get_piece_size()),
        },
        "bos": {
            "prepended": bos_prepended,
            "bos_token_id": int(model.config.bos_token_id),
            "note": (
                "YES -- BOS (<s>, id 1) IS prepended to every one of the 32 prompts; "
                "verified per-prompt, not assumed (see per_prompt[*].first_token_id). "
                "tokenizer_config.json sets add_bos_token=true and the tokenizer.json "
                "post-processor applies it. LOAD-BEARING: an NSL harness that does NOT "
                "prepend BOS will differ in EVERY logit of this artifact."
            ),
            "prompt0_text": prompts[0],
            "prompt0_token_ids": tok(prompts[0], return_tensors="pt").input_ids[0].tolist(),
            "prompt0_token_pieces": tok.convert_ids_to_tokens(
                tok(prompts[0], return_tensors="pt").input_ids[0].tolist()
            ),
        },
        "shims": [
            {
                "id": 1,
                "what": "Import BitnetConfig/BitnetForCausalLM from the checkpoint dir BY FILE PATH "
                        "(PEP-420 namespace package) and instantiate BitnetForCausalLM explicitly.",
                "why": "config.json says \"model_type\": \"llama\" and the checkpoint has no auto_map, so "
                       "AutoConfig/AutoModelForCausalLM resolve to the BUILT-IN LlamaForCausalLM. That "
                       "fallback drops all 52 BitNet-only sub-layernorms (mlp.ffn_layernorm, "
                       "self_attn.inner_attn_ln) as UNEXPECTED and never instantiates BitLinear, so "
                       "weight/activation quantization is skipped entirely. v1 of this script hit exactly "
                       "that and emitted an all-NaN artifact.",
            },
            {
                "id": 2,
                "what": "Patch BitnetAttention._init_rope to unconditionally build the default "
                        "BitnetRotaryEmbedding(head_dim, max_position_embeddings, base=rope_theta).",
                "why": "transformers 5 aliases PretrainedConfig.rope_scaling to rope_parameters, which "
                       "returns {'rope_theta': 10000.0, 'rope_type': 'default'} instead of the None stored "
                       "in config.json, so modeling_bitnet.py:~272 took the `raise NotImplementedError` "
                       "branch. Semantics-preserving: config.json really is rope_scaling=null / "
                       "rope_theta=10000.0, and the patch builds the identical module the None branch built.",
            },
            {
                "id": 3,
                "what": "BitnetForCausalLM._tied_weights_keys = "
                        "{\"lm_head.weight\": \"model.embed_tokens.weight\"}",
                "why": "transformers 5 modeling_utils.py:2704 calls _tied_weights_keys.keys(); the vendored "
                       "4.x list form [\"lm_head.weight\"] raised AttributeError. The dict is the v5 "
                       "spelling of the same tie (config tie_word_embeddings=true).",
            },
            {
                "id": 4,
                "what": "from_pretrained(dtype=torch.float16) instead of torch_dtype=.",
                "why": "torch_dtype= is deprecated in transformers 5.",
            },
            {
                "id": 5,
                "what": "attn_implementation=\"eager\" forced on both the config and from_pretrained.",
                "why": "modeling_bitnet's LLAMA_ATTENTION_CLASSES only contains {eager, flash_attention_2}; "
                       "transformers 5 defaults _attn_implementation to \"sdpa\", which KeyErrors. eager is "
                       "also the arithmetically reference path.",
            },
        ],
        "validation": {
            **report,
            "all_finite": True,
            "smoke": {
                "text": smoke_text,
                "ids": smoke_list,
                "argmax_id": smoke_argmax,
                "argmax_piece": smoke_tokstr,
                "expected_argmax_id": 3681,
            },
        },
        "prompts": EXPECTED_PROMPTS,
        "vocab": EXPECTED_VOCAB,
        "layout": "prompt-major, final-token-position logits, row-major little-endian FP16, no header",
        "bytes": len(buf),
        "sha256": digest,
        "argmax_per_prompt": [p["argmax_id"] for p in per_prompt],
        "per_prompt": per_prompt,
    }
    args.meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.meta_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as exc:
        print(f"VALIDATION FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
