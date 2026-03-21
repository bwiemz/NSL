"""Tests for HuggingFace Hub integration."""

import pytest
from nslpy.hub import list_supported_architectures, _HF_ARCH_MAP


class TestHubArchMap:
    def test_supported_architectures_not_empty(self):
        archs = list_supported_architectures()
        assert len(archs) > 0

    def test_llama_supported(self):
        assert "LlamaForCausalLM" in _HF_ARCH_MAP
        assert _HF_ARCH_MAP["LlamaForCausalLM"] == "llama"

    def test_gpt2_supported(self):
        assert "GPT2LMHeadModel" in _HF_ARCH_MAP
        assert _HF_ARCH_MAP["GPT2LMHeadModel"] == "gpt2"

    def test_mistral_supported(self):
        assert "MistralForCausalLM" in _HF_ARCH_MAP

    def test_all_values_are_strings(self):
        for k, v in _HF_ARCH_MAP.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
