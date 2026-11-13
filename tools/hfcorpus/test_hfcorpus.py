"""Tests for the corpus tools that can run without the corpus.

    python -m unittest discover -s tools/hfcorpus -p 'test_*.py'

Needs numpy. The decontamination test that reads parquet files is skipped
when pyarrow is not installed; everything else runs on synthetic streams
built here. Expected values are worked out from the construction of each
stream, not taken from the code under test.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify

EOS = 7


def stream(documents: list[list[int]]) -> np.ndarray:
    """Documents joined into one u16 stream, each followed by EOS."""
    out: list[int] = []
    for doc in documents:
        out.extend(doc)
        out.append(EOS)
    return np.asarray(out, dtype="<u2")


def random_documents(rng: random.Random, count: int, low: int, high: int) -> list[list[int]]:
    # Token ids from 100 up, so no document contains EOS.
    return [[rng.randrange(100, 60000) for _ in range(rng.randrange(low, high))]
            for _ in range(count)]


class IterDocuments(unittest.TestCase):
    def test_spans_exclude_the_separator(self) -> None:
        s = np.asarray([1, 2, EOS, 3, EOS, EOS, 4, 5], dtype="<u2")
        # [1 2] EOS [3] EOS [] EOS [4 5]
        self.assertEqual(list(verify.iter_documents(s, EOS)),
                         [(0, 2), (3, 4), (5, 5), (6, 8)])

    def test_a_chunk_boundary_does_not_split_a_document(self) -> None:
        s = np.asarray([1, 2, 3, EOS, 4, 5, 6, 7 + 1, EOS], dtype="<u2")
        whole = list(verify.iter_documents(s, EOS))
        self.assertEqual(whole, [(0, 3), (4, 8)])
        for chunk in (1, 2, 3, 4, 5):
            self.assertEqual(list(verify.iter_documents(s, EOS, chunk=chunk)), whole)


class TrainValDisjointness(unittest.TestCase):
    def setUp(self) -> None:
        rng = random.Random(0)
        self.train_docs = random_documents(rng, 400, 40, 200)
        self.other_docs = random_documents(rng, 50, 40, 200)

    def test_a_validation_set_copied_from_the_middle_of_train_is_found(self) -> None:
        train = stream(self.train_docs)
        copied = self.train_docs[150:190]
        val = stream(copied)
        # The check this replaces compared the first and the last 2^20 tokens
        # of the two files, and passed on exactly this input.
        n = min(1 << 20, train.size, val.size)
        self.assertFalse(np.array_equal(train[:n], val[:n]))
        self.assertFalse(np.array_equal(train[-n:], val[-n:]))

        val_docs, shared = verify.shared_documents(train, val, EOS)
        self.assertEqual(val_docs, 40)
        self.assertEqual(shared, 40)

    def test_a_partly_leaked_validation_set_reports_the_leaked_count(self) -> None:
        train = stream(self.train_docs)
        val = stream(self.other_docs[:30] + self.train_docs[10:13])
        self.assertEqual(verify.shared_documents(train, val, EOS), (33, 3))

    def test_disjoint_sets_share_nothing(self) -> None:
        train = stream(self.train_docs)
        val = stream(self.other_docs)
        self.assertEqual(verify.shared_documents(train, val, EOS), (50, 0))

    def test_short_documents_are_not_counted(self) -> None:
        # The same three-token document on both sides is not a leak, and a
        # validation set made only of such documents has nothing to compare.
        short = [[101, 102, 103]]
        train = stream(self.train_docs + short)
        val = stream(short * 5)
        self.assertEqual(verify.shared_documents(train, val, EOS), (0, 0))
        # One token over the floor is counted.
        exact = [list(range(200, 200 + verify.MIN_DOC_TOKENS))]
        self.assertEqual(
            verify.shared_documents(stream(self.train_docs + exact), stream(exact), EOS),
            (1, 1),
        )

    def test_a_document_cut_at_the_edge_of_val_is_a_different_document(self) -> None:
        # val starts in the middle of a training document: that first piece
        # is not a whole training document, the ones after it are.
        train = stream(self.train_docs)
        first = self.train_docs[20]
        val = stream([first[5:]] + self.train_docs[21:25])
        val_docs, shared = verify.shared_documents(train, val, EOS)
        self.assertEqual(shared, 4)
        self.assertEqual(val_docs, 5 if len(first) - 5 >= verify.MIN_DOC_TOKENS else 4)


if __name__ == "__main__":
    unittest.main()
