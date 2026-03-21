"""HuggingFace Hub integration for NSL models.

Download and compile HuggingFace models for NSL inference.

Usage::

    model = nslpy.hub.from_pretrained("meta-llama/Llama-3.2-1B")
    output = model(input_tokens)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from nslpy._core import NslModel


# Architecture name → NSL model template mapping
_HF_ARCH_MAP = {
    "LlamaForCausalLM": "llama",
    "GPT2LMHeadModel": "gpt2",
    "MistralForCausalLM": "mistral",
    "Phi3ForCausalLM": "phi3",
    "Qwen2ForCausalLM": "qwen2",
    "GemmaForCausalLM": "gemma",
    "StableLmForCausalLM": "stablelm",
}


def from_pretrained(
    model_id: str,
    *,
    revision: str = "main",
    cache_dir: Optional[str | Path] = None,
    nsl_template: Optional[str] = None,
    compile_flags: Optional[list[str]] = None,
) -> NslModel:
    """Download a model from HuggingFace Hub and load it as an NslModel.

    Args:
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B").
        revision: Git revision to download.
        cache_dir: Local cache directory (default: HF_HOME or ~/.cache/huggingface).
        nsl_template: NSL model architecture template. Auto-detected from config.json
                      if not specified.
        compile_flags: Additional flags for ``nsl build``.

    Returns:
        NslModel ready for inference.

    Example::

        model = from_pretrained("meta-llama/Llama-3.2-1B")
        output = model(tokens)
    """
    try:
        from huggingface_hub import hf_hub_download, model_info
    except ImportError:
        raise ImportError(
            "nslpy.hub requires huggingface_hub. Install with: pip install huggingface_hub"
        )

    # Download config.json to detect architecture
    config_path = hf_hub_download(
        model_id, "config.json",
        revision=revision,
        cache_dir=cache_dir,
    )
    with open(config_path) as f:
        config = json.load(f)

    # Auto-detect architecture
    arch = nsl_template
    if arch is None:
        architectures = config.get("architectures", [])
        for hf_arch in architectures:
            if hf_arch in _HF_ARCH_MAP:
                arch = _HF_ARCH_MAP[hf_arch]
                break
        if arch is None:
            raise ValueError(
                f"Cannot auto-detect NSL template for architectures {architectures}. "
                f"Pass nsl_template= explicitly. Supported: {list(_HF_ARCH_MAP.values())}"
            )

    # Download safetensors weights
    # Try model.safetensors first, then model-00001-of-XXXXX.safetensors shards
    try:
        weights_path = hf_hub_download(
            model_id, "model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception:
        # Try sharded format
        weights_path = hf_hub_download(
            model_id, "model-00001-of-00001.safetensors",
            revision=revision,
            cache_dir=cache_dir,
        )

    return NslModel(weights_path)


def list_supported_architectures() -> list[str]:
    """Return list of HuggingFace architectures that can be auto-detected."""
    return list(_HF_ARCH_MAP.keys())
