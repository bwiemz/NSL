#!/usr/bin/env python3
"""Checks on the arithmetic the benchmark's numbers rest on. No GPU, no torch.

    python3 tools/perfcheck/test_perfcheck.py

The references are hand-worked: FLOPs per token from the architecture written out
term by term, tok/s and spread from stamps chosen so the answer is exact, GPU-busy
time from intervals whose union is known.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nsys_summary as ns
import perfcheck as pc


class FlopsPerToken(unittest.TestCase):
    def test_coder_1b_by_hand(self) -> None:
        s = pc.read_shape("1b", None, None, None)
        # Per layer: q 2048x2048, k and v 2048x512 each, o 2048x2048, SwiGLU 3 x 2048x8192.
        per_layer = 2048 * 2048 + 2 * 2048 * 512 + 2048 * 2048 + 3 * 2048 * 8192
        self.assertEqual(per_layer, 60_817_408)
        matmul = 16 * per_layer + 2048 * 49152  # + tied LM head
        self.assertEqual(s.matmul_params(), matmul)
        attn = 6 * 16 * 32 * 64 * 2048  # causal score + value matmuls, fwd + bwd
        self.assertEqual(s.flops_per_token(), 6 * matmul + attn)
        self.assertEqual(matmul, 2**30)  # 1,073,741,824: Coder-1B is exactly 2^30 of them
        self.assertEqual(s.flops_per_token(), 6_845_104_128)

    def test_recipe_step_shapes(self) -> None:
        for size, mb, seq, acc in (("500m", 2, 1024, 8), ("1b", 2, 2048, 4)):
            s = pc.read_shape(size, None, None, None)
            self.assertEqual((s.micro_batch, s.seq, s.accum), (mb, seq, acc))
            self.assertEqual(s.tokens_per_update, 16384)

    def test_shapes_agree_with_the_pytorch_twin(self) -> None:
        # ref.py's SIZES and config.nsl are written independently; the harness reads
        # the latter, the PyTorch arms build the former.
        sys.path.insert(0, str(pc.REPO / "tools" / "refcheck"))
        try:
            import ref  # pyright: ignore[reportMissingImports]
        except ImportError:  # torch not installed here: the twin cannot be imported
            self.skipTest("tools/refcheck/ref.py needs torch")
        for size in ("500m", "1b"):
            s = pc.read_shape(size, None, None, None)
            c = ref.SIZES[size][1]
            self.assertEqual(
                (s.d_model, s.n_layers, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff, s.vocab),
                (c.d_model, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.d_ff, c.vocab),
            )


class Steady(unittest.TestCase):
    def test_window_after_warmup(self) -> None:
        # Two slow warm-up gaps, then four updates of exactly 2 s and one of 4 s.
        stamps = [0.0, 10.0, 15.0, 17.0, 19.0, 21.0, 23.0, 27.0]
        st = pc.steady(stamps, warmup=2, tokens_per_update=1000)
        assert st is not None
        self.assertEqual(st["updates_measured"], 5)
        self.assertAlmostEqual(st["tok_s"], 5 * 1000 / 12.0)
        self.assertAlmostEqual(st["update_s_median"], 2.0)
        self.assertAlmostEqual(st["update_s_max"], 4.0)
        mean = 12.0 / 5
        sd = ((4 * (2 - mean) ** 2 + (4 - mean) ** 2) / 5) ** 0.5
        self.assertAlmostEqual(st["update_s_cv"], sd / mean)

    def test_too_few_stamps(self) -> None:
        self.assertIsNone(pc.steady([0.0, 1.0, 2.0], warmup=1, tokens_per_update=1))


class Arms(unittest.TestCase):
    def test_builds_are_distinct_where_the_arithmetic_is(self) -> None:
        keys = {pc.flag_key(pc.arm(n).build_key) for n in ("nsl_tf32", "nsl_bf16")}
        self.assertEqual(len(keys), 2, "bf16 is a build-time setting: it needs its own binary")
        self.assertEqual(
            pc.flag_key(pc.arm("nsl_tf32").build_key),
            pc.flag_key(pc.arm("nsl_tf32_nodrain").build_key),
            "the drain switch is read at run time: same binary",
        )

    def test_torch_arm_names(self) -> None:
        a = pc.arm("torch_bf16_compile_sel")
        self.assertEqual(
            (a.kind, a.precision, a.compile, a.ckpt), ("torch", "bf16", True, "selective")
        )
        with self.assertRaises(SystemExit):
            pc.arm("torch_fp8_eager_sel")

    def test_toolchain_variants_are_their_own_builds(self) -> None:
        base, after = pc.arm("nsl_bf16_nodrain"), pc.arm("nsl_bf16_nodrain@after")
        self.assertEqual(after.name, "nsl_bf16_nodrain@after")
        self.assertEqual(after.variant, "after")
        # Same arm in every respect but the compiler that builds it.
        self.assertEqual(dataclasses.replace(after, name=base.name, variant=""), base)
        self.assertNotEqual(pc.flag_key(base.build_key), pc.flag_key(after.build_key))
        self.assertTrue(pc.flag_key(after.build_key).endswith("@after"))
        for bad in ("torch_tf32_eager_sel@after", "nsl_tf32@a/b", "nsl_tf32@"):
            with self.assertRaises(SystemExit, msg=bad):
                pc.arm(bad)


class Pairs(unittest.TestCase):
    @staticmethod
    def row(arm: str, rnd: int, tok_s: float, status: str = "ok") -> dict[str, object]:
        return {"arm": arm, "round": rnd, "status": status, "steady": {"tok_s": tok_s}}

    def test_ratios_are_taken_within_a_round(self) -> None:
        rows = {
            "a": [self.row("a", 0, 110.0), self.row("a", 1, 99.0), self.row("a", 2, 50.0)],
            # Round 1 is a machine-wide slow round: both arms ~10% down, ratio unchanged.
            "b": [self.row("b", 0, 100.0), self.row("b", 1, 90.0), self.row("b", 3, 1.0)],
        }
        # Rounds 2 and 3 have only one of the two arms: no ratio for them.
        self.assertEqual(pc.paired_ratios(rows, "a", "b"), [(0, 1.1), (1, 1.1)])

    def test_failed_runs_are_not_paired(self) -> None:
        rows = {
            "a": [self.row("a", 0, 110.0), self.row("a", 1, 0.0, status="oom")],
            "b": [self.row("b", 0, 100.0), self.row("b", 1, 100.0)],
        }
        self.assertEqual(pc.paired_ratios(rows, "a", "b"), [(0, 1.1)])


def iv(s: int, e: int, name: str = "k") -> ns.Interval:
    return ns.Interval(s, e, name)


class Timeline(unittest.TestCase):
    def test_busy_is_the_union(self) -> None:
        # [0,10) and [5,20) overlap -> 20; [30,40) clipped to [30,35) -> 5.
        self.assertEqual(ns.busy_ns([iv(0, 10), iv(5, 20), iv(30, 40)], 0, 35), 25)
        self.assertEqual(ns.busy_ns([iv(5, 20), iv(0, 10)], 2, 100), 18)

    def test_update_ends_split_on_gaps(self) -> None:
        ks = [iv(0, 5, "adam"), iv(6, 9, "adam"), iv(50, 60, "gemm"), iv(100, 110, "adam")]
        self.assertEqual(ns.update_ends(ks, "adam", gap_ns=20), [9, 110])

    def test_phase_cuts(self) -> None:
        ks = [
            iv(10, 11, "embed"), iv(12, 13, "gemm"), iv(14, 15, "lce_bwd"), iv(16, 17, "gemm"),
            iv(20, 21, "embed"), iv(22, 23, "lce_bwd"), iv(30, 31, "adam"),
        ]  # fmt: skip
        cuts = ns.phase_cuts(ks, 0, 40, "embed", "lce_bwd", "adam")
        self.assertEqual(
            cuts,
            [("pre", 0, 10), ("fwd", 10, 14), ("bwd", 14, 20), ("fwd", 20, 22),
             ("bwd", 22, 30), ("opt", 30, 40)],
        )  # fmt: skip

    def test_categories(self) -> None:
        self.assertEqual(ns.category("void pytorch_flash::flash_fwd_kernel<...>"), "attention")
        self.assertEqual(ns.category("nvjet_tst_128x256_64x4_2x1_v_bz_TNN"), "gemm")
        self.assertEqual(
            ns.category("multi_tensor_apply_kernel<FusedAdamMathFunctor>"), "optimizer"
        )


if __name__ == "__main__":
    unittest.main()
