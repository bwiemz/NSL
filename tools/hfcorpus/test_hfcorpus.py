"""Tests for the corpus tools that can run without the corpus.

    python -m unittest discover -s tools/hfcorpus -p 'test_*.py'

Needs numpy. The decontamination test that reads parquet files is skipped
when pyarrow is not installed; everything else runs on synthetic streams
built here. Expected values are worked out from the construction of each
stream, not taken from the code under test.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decontaminate
import make_mix
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


class MixPlacement(unittest.TestCase):
    """Base corpus: 2000 documents of 2 tokens, then 2000 of 200 tokens (each
    counting its separator). The first half of the documents holds
    2000 * 2 / (2000 * 2 + 2000 * 200) = 1/101 of the tokens."""

    def setUp(self) -> None:
        lengths = [2] * 2000 + [200] * 2000
        ends = np.cumsum(lengths)
        starts = ends - np.asarray(lengths)
        self.base_docs = np.stack([starts, ends], axis=1)
        self.base_tokens = int(ends[-1])

    def share_in_short_half(self, placement: str) -> float:
        rng = random.Random(1337)
        keys = [make_mix.injection_key(rng, self.base_docs, self.base_tokens, placement)
                for _ in range(20000)]
        self.assertTrue(all(0 <= k < len(self.base_docs) for k in keys))
        return sum(k < 2000 for k in keys) / len(keys)

    def test_document_placement_follows_the_document_index(self) -> None:
        # Half of the documents, so half of the injections: 50 times the
        # share of tokens those documents hold.
        self.assertAlmostEqual(self.share_in_short_half("document"), 0.5, delta=0.02)

    def test_token_placement_follows_the_token_share(self) -> None:
        self.assertAlmostEqual(self.share_in_short_half("token"), 1 / 101, delta=0.004)

    def test_document_placement_is_one_uniform_draw(self) -> None:
        # The draw sequence of the default must not change: the hash of a
        # mixture built before the option existed is pinned in a manifest.
        a, b = random.Random(5), random.Random(5)
        for _ in range(100):
            self.assertEqual(
                make_mix.injection_key(a, self.base_docs, self.base_tokens, "document"),
                b.uniform(0, len(self.base_docs)),
            )

    def test_an_unknown_placement_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            make_mix.injection_key(random.Random(0), self.base_docs, self.base_tokens, "bytes")


LONG_A = "    result = sorted(numbers, key=lambda value: (value % 7, -value))"
LONG_B = "    return [left + right for left, right in zip(first_values, second_values)]"
MID_A = "    total = sum(x * x for x in xs)"
MID_B = "    best = max(scores, default=None)"


class DecontaminationCoverage(unittest.TestCase):
    def lines_of(self, *texts: str) -> set[str]:
        out: set[str] = set()
        for text in texts:
            for raw in text.splitlines():
                norm = decontaminate.normalize(raw)
                if len(norm) >= decontaminate.MIN_LINE_CHARS:
                    out.add(norm)
        return out

    def test_the_fixture_lines_are_on_the_intended_side_of_each_threshold(self) -> None:
        n = decontaminate.normalize
        self.assertGreaterEqual(len(n(LONG_A)), decontaminate.STRONG_LINE_CHARS)
        self.assertGreaterEqual(len(n(LONG_B)), decontaminate.STRONG_LINE_CHARS)
        for mid in (MID_A, MID_B):
            self.assertGreaterEqual(len(n(mid)), decontaminate.MIN_LINE_CHARS)
            self.assertLess(len(n(mid)), decontaminate.STRONG_LINE_CHARS)

    def test_problems_made_of_short_lines_are_counted_as_undetectable(self) -> None:
        short_solution = "    if n < 2:\n        return n\n    return f(n - 1) + f(n - 2)"
        problems = [
            # One long line: detectable whole and by its solution.
            ("bench", "def a(numbers):\n" + LONG_A, LONG_A),
            # Two mid lines in the solution: detectable (two distinct lines).
            ("bench", "def b(xs, scores):\n" + MID_A + "\n" + MID_B, MID_A + "\n" + MID_B),
            # One mid line only: below the two-line rule, undetectable.
            ("bench", "def c(xs):\n" + MID_A, MID_A),
            # All lines short: nothing enters the line set at all.
            ("bench", "def f(n):\n" + short_solution, short_solution),
            # A long line in the prompt, a short solution: the whole problem
            # is detectable, the solution on its own is not.
            ("other", LONG_B + "\n    return 0", "    return 0"),
        ]
        lines = self.lines_of(*(whole for _, whole, _ in problems))
        got = decontaminate.coverage(problems, lines)
        self.assertEqual(got, {
            "bench": {"problems": 4, "undetectable_problems": 2, "undetectable_solutions": 2},
            "other": {"problems": 1, "undetectable_problems": 0, "undetectable_solutions": 1},
        })

    def test_the_count_agrees_with_what_the_scan_does_to_a_verbatim_copy(self) -> None:
        # A corpus document that IS the short-line solution passes the scan.
        short_solution = "    if n < 2:\n        return n\n    return f(n - 1) + f(n - 2)"
        lines = self.lines_of("def f(n):\n" + short_solution, LONG_A)
        self.assertFalse(decontaminate.doc_is_contaminated(short_solution, lines)[0])
        self.assertTrue(decontaminate.doc_is_contaminated("x = 1\n" + LONG_A, lines)[0])

    def test_benchmark_problems_reads_the_same_files_as_benchmark_lines(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            he = root / "data/hf/humaneval/openai_humaneval"
            he.mkdir(parents=True)
            pq.write_table(pa.table({
                "task_id": ["HumanEval/0", "HumanEval/1"],
                "prompt": ["def a(numbers):", "def f(n):"],
                "canonical_solution": [LONG_A, "    return n"],
                "test": ["assert a([3, 1]) == [1, 3]", "assert f(1) == 1"],
            }), he / "test-00000-of-00001.parquet")
            full = root / "data/hf/mbpp/full"
            full.mkdir(parents=True)
            pq.write_table(pa.table({
                "task_id": [1, 2],
                "text": ["Write a function to add two lists element by element.", "Add."],
                "code": [LONG_B, "    return a + b"],
                "test_list": [["assert add([1], [2]) == [3]"], ["assert add(1, 2) == 3"]],
            }), full / "train-00000-of-00001.parquet")
            sanitized = root / "data/hf/mbpp/sanitized"
            sanitized.mkdir(parents=True)
            pq.write_table(pa.table({
                "task_id": [2],
                "prompt": ["Add."],
                "code": ["    return a + b"],
                "test_list": [["assert add(1, 2) == 3"]],
            }), sanitized / "test-00000-of-00001.parquet")

            saved = decontaminate.REPO
            decontaminate.REPO = root
            try:
                counts, lines = decontaminate.benchmark_lines()
                problems = list(decontaminate.benchmark_problems())
            finally:
                decontaminate.REPO = saved

        self.assertEqual([p[0] for p in problems],
                         ["humaneval", "humaneval", "mbpp", "mbpp", "mbpp"])
        self.assertEqual(problems[0][2], LONG_A)
        self.assertIn("assert a([3, 1]) == [1, 3]", problems[0][1])
        self.assertEqual(problems[2][2], LONG_B)
        self.assertIn("assert add([1], [2]) == [3]", problems[2][1])
        # Every line benchmark_lines kept comes from some problem's text.
        for line in lines:
            self.assertTrue(
                any(line in {decontaminate.normalize(r) for r in whole.splitlines()}
                    for _, whole, _ in problems),
                line,
            )
        self.assertEqual(sum(counts.values()), len(lines))
        got = decontaminate.coverage(problems, lines)
        # HumanEval/0 and MBPP task 1 carry a line of 60+ characters;
        # HumanEval/1 and MBPP task 2 (twice, full and sanitized) have only
        # short lines.
        self.assertEqual(got, {
            "humaneval": {"problems": 2, "undetectable_problems": 1,
                          "undetectable_solutions": 1},
            "mbpp": {"problems": 3, "undetectable_problems": 2,
                     "undetectable_solutions": 2},
        })
        json.dumps(got)  # the report is written as JSON


if __name__ == "__main__":
    unittest.main()
