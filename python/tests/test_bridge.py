"""Tests for the DLPack bridge (defensive copy guard, alignment checks)."""

import pytest
from nslpy._bridge import _is_contiguous, _is_aligned, _needs_defensive_copy


class MockTensor:
    """Minimal mock for testing bridge validation logic."""

    def __init__(self, contiguous: bool = True, data_ptr_val: int = 0):
        self._contiguous = contiguous
        self._data_ptr = data_ptr_val

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self._data_ptr


class TestContiguityCheck:
    def test_contiguous_tensor(self):
        t = MockTensor(contiguous=True)
        assert _is_contiguous(t) is True

    def test_non_contiguous_tensor(self):
        t = MockTensor(contiguous=False)
        assert _is_contiguous(t) is False

    def test_unknown_type_assumes_contiguous(self):
        assert _is_contiguous(42) is True


class TestAlignmentCheck:
    def test_aligned_64(self):
        t = MockTensor(data_ptr_val=0)
        assert _is_aligned(t, 64) is True

    def test_aligned_at_boundary(self):
        t = MockTensor(data_ptr_val=128)
        assert _is_aligned(t, 64) is True

    def test_misaligned(self):
        t = MockTensor(data_ptr_val=17)
        assert _is_aligned(t, 64) is False

    def test_unknown_type_assumes_aligned(self):
        assert _is_aligned("not a tensor") is True


class TestDefensiveCopyDecision:
    def test_safe_tensor_no_copy(self):
        t = MockTensor(contiguous=True, data_ptr_val=64)
        assert _needs_defensive_copy(t) is False

    def test_non_contiguous_needs_copy(self):
        t = MockTensor(contiguous=False, data_ptr_val=64)
        assert _needs_defensive_copy(t) is True

    def test_misaligned_needs_copy(self):
        t = MockTensor(contiguous=True, data_ptr_val=7)
        assert _needs_defensive_copy(t) is True

    def test_mutating_input_needs_copy(self):
        t = MockTensor(contiguous=True, data_ptr_val=64)
        assert _needs_defensive_copy(t, mutates_input=True) is True

    def test_all_bad_flags(self):
        t = MockTensor(contiguous=False, data_ptr_val=3)
        assert _needs_defensive_copy(t, mutates_input=True) is True
