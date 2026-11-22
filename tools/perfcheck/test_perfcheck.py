#!/usr/bin/env python3
"""Checks on the arithmetic and parsing the benchmark's numbers rest on. No GPU.

    VENV/bin/python tools/perfcheck/test_perfcheck.py

VENV is the environment the PyTorch arms run in: the twin checks import
`tools/refcheck/ref.py`, which needs torch, and FAIL (not skip) without it.

The references are hand-worked: FLOPs per token from the architecture written out
term by term, recipe shapes read from `pretrain_prod.nsl`'s own literals (the harness
reads `config.nsl`), tok/s and spread from stamps chosen so the answer is exact,
GPU-busy time from intervals whose union is known.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nsys_summary as ns
import perfcheck as pc


def import_twin() -> Any:
    sys.path.insert(0, str(pc.REPO / "tools" / "refcheck"))
    try:
        import ref  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise AssertionError(
            f"tools/refcheck/ref.py did not import ({e}); run these tests with the "
            "interpreter of the venv that holds torch"
        ) from e
    return ref


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

    def test_coder_500m_by_hand(self) -> None:
        s = pc.read_shape("500m", None, None, None)
        # d 1280, 20 query heads and 10 KV heads of 64, d_ff 3520, 24 layers, seq 1024.
        q = o = 1280 * 1280  # 1,638,400 each
        kv = 2 * 1280 * 640  # 1,638,400
        ffn = 3 * 1280 * 3520  # 13,516,800
        per_layer = q + kv + o + ffn
        self.assertEqual(per_layer, 18_432_000)
        head = 1280 * 49152  # 62,914,560
        matmul = 24 * per_layer + head
        self.assertEqual(matmul, 505_282_560)
        self.assertEqual(s.matmul_params(), matmul)
        attn = 6 * 24 * 20 * 64 * 1024  # 188,743,680
        self.assertEqual(s.flops_per_token(), 6 * matmul + attn)
        self.assertEqual(s.flops_per_token(), 3_220_439_040)


class RecipeShapes(unittest.TestCase):
    def test_step_matches_the_recipe_literals(self) -> None:
        # The harness reads config.nsl; the recipe states its step as literals in its
        # DataLoader and train() calls. Two sources, one answer.
        for size in ("50m", "500m", "1b"):
            s = pc.read_shape(size, None, None, None)
            src = (pc.REPO / "models" / s.directory / "pretrain_prod.nsl").read_text()
            dl = re.search(r"^let loader = DataLoader\(.*?batch_size=(\d+), seq_len=(\d+)", src,
                           re.MULTILINE)  # fmt: skip
            tr = re.search(r"^train\(.*?grad_accumulation=(\d+)", src, re.MULTILINE)
            assert dl is not None and tr is not None, size
            self.assertEqual(
                (s.micro_batch, s.seq, s.accum),
                (int(dl.group(1)), int(dl.group(2)), int(tr.group(1))),
                size,
            )
        for size in ("500m", "1b"):
            self.assertEqual(pc.read_shape(size, None, None, None).tokens_per_update, 16384)

    def test_the_twin_builds_the_same_model(self) -> None:
        # ref.py's SIZES and config.nsl are written independently; the harness reads
        # the latter for NSL, the PyTorch arms build the former.
        ref = import_twin()
        for size in ("500m", "1b"):
            s = pc.read_shape(size, None, None, None)
            c = ref.SIZES[size][1]
            cfg = (pc.REPO / "models" / s.directory / "config.nsl").read_text()

            def const(name: str, text: str = cfg) -> float:
                m = re.search(rf"^const {name} = ([0-9.]+)", text, re.MULTILINE)
                assert m is not None, name
                return float(m.group(1))

            self.assertEqual(
                (s.d_model, s.n_layers, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff, s.vocab),
                (c.d_model, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.d_ff, c.vocab),
            )
            self.assertEqual(const("ROPE_THETA"), c.theta, size)
            self.assertEqual(int(const("MAX_SEQ_LEN")), c.max_seq, size)
            self.assertGreaterEqual(c.max_seq, s.seq, "the twin's RoPE table covers the step")


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

    def test_paired_ratios_pair_by_round(self) -> None:
        def row(rnd: int, tok: float, status: str = "ok") -> dict[str, Any]:
            return {"round": rnd, "status": status, "steady": {"tok_s": tok}}

        rows = {
            "a": [row(0, 300.0), row(1, 330.0), row(2, 1.0, "error")],
            "b": [row(1, 110.0), row(0, 100.0), row(2, 100.0)],
        }
        self.assertEqual(pc.paired_ratios(rows, "a", "b"), [(0, 3.0), (1, 3.0)])


class Parsing(unittest.TestCase):
    def test_nsl_loss_lines(self) -> None:
        self.assertEqual(pc.parse_loss("tensor([11.229241371154785])"), 11.229241371154785)
        self.assertEqual(pc.parse_loss("10.5"), 10.5)
        nan = pc.parse_loss("tensor([NaN])")
        assert nan is not None
        self.assertTrue(math.isnan(nan))
        self.assertEqual(pc.parse_loss("tensor([-inf])"), -math.inf)
        self.assertIsNone(pc.parse_loss("OPT_STEP"))

    def test_loss_problem(self) -> None:
        self.assertIsNone(pc.loss_problem([11.2, 11.1, 11.0], 3))
        self.assertIn("losses for", pc.loss_problem([11.2, 11.0], 3) or "")
        self.assertIn("non-finite", pc.loss_problem([11.2, math.nan, 11.0], 3) or "")
        self.assertIn("did not fall", pc.loss_problem([11.0, 11.1, 11.2], 3) or "")

    def test_memstats_contexts_at_peak(self) -> None:
        text = (
            "         Peak reserved:    25.51 GB\n"
            "         Top contexts at peak:\n"
            "         <unknown>: 16.02 GB (776 blocks)\n"
            "         matmul_f32: 5.41 GB (98 blocks)\n"
            "         Top contexts:\n"
            "         <unknown>: 1.00 GB (3 blocks)\n"
        )
        self.assertEqual(
            pc.top_contexts_at_peak(text), [("<unknown>", "16.02 GB"), ("matmul_f32", "5.41 GB")]
        )

    def test_host_profile_medians(self) -> None:
        step = (
            "[host-profile] step: {t} ms attributed\n"
            "  pcie           h2d=1.1 MiB  d2h=0.0 MiB  total=1.1 MiB\n"
            "  memcpy             {m} ms      134 calls   5000.00 us/call   97.7%\n"
            "  kernel_launch       15.0 ms     {k} calls      9.00 us/call    2.0%\n"
        )
        text = "".join(step.format(t=t, m=m, k=k) for t, m, k in
                       ((800.0, 780.0, 1700), (950.0, 930.0, 1650), (700.0, 680.0, 1900)))  # fmt: skip
        hp = pc.host_profile_summary(text)  # medians, not means (800 vs 816.7)
        assert hp is not None
        self.assertEqual(hp["steps"], 3)
        self.assertEqual(hp["attributed_ms"], 800.0)
        self.assertEqual(hp["memcpy_ms"], 780.0)
        self.assertEqual(hp["memcpy_calls"], 134)
        self.assertEqual(hp["kernel_launch_calls"], 1700)


class ParamStorage(unittest.TestCase):
    def test_the_conversion_line_is_required_and_read(self) -> None:
        err = (
            "[nsl-matmul] parameter storage: bf16-split (train param_dtype: ...)\n"
            "[param-storage] 112 parameters stored bf16-split (1.812 GiB of bf16 parameters)\n"
        )
        self.assertEqual(pc.params_converted(err, "bf16-split"), 112)
        # The banner alone is not a conversion.
        self.assertEqual(pc.params_converted(err.splitlines()[0], "bf16-split"), 0)
        # Another mode's line does not count, and a prefix is not the mode.
        self.assertEqual(pc.params_converted(err, "bf16-master"), 0)
        self.assertEqual(
            pc.params_converted(
                "[param-storage] 3 parameters stored bf16-master-tz", "bf16-master"
            ),
            0,
        )
        # The compile-time plan line is not the run's conversion either.
        plan = "[param-storage] 112 of 146 parameters stored bf16-split (32 of the GEMMs ...)"
        self.assertEqual(pc.params_converted(plan, "bf16-split"), 0)


class Arms(unittest.TestCase):
    def test_builds_are_distinct_where_the_arithmetic_is(self) -> None:
        keys = {pc.flag_key(pc.arm(n).build_key) for n in ("nsl_tf32", "nsl_bf16")}
        self.assertEqual(len(keys), 2, "bf16 is a build-time setting: it needs its own binary")
        self.assertEqual(
            pc.flag_key(pc.arm("nsl_tf32").build_key),
            pc.flag_key(pc.arm("nsl_tf32_nodrain").build_key),
            "the drain switch is read at run time: same binary",
        )

    def test_the_bf16_storage_arm_is_its_own_program(self) -> None:
        # Same flags and environment as an f32-storage arm; only the train header
        # differs, so the program (and the binary) must too.
        s = pc.arm("nsl_bf16s")
        self.assertEqual((s.flags, s.build_env), (pc.arm("nsl_tf32").flags, ()))
        self.assertEqual(s.train_args, ', precision="bf16"')
        keys = {pc.flag_key(pc.arm(n).build_key) for n in ("nsl_tf32", "nsl_bf16", "nsl_bf16s")}
        self.assertEqual(len(keys), 3)
        self.assertTrue(pc.flag_key(pc.arm("nsl_bf16s@after").build_key).endswith("@after"))
        shape = pc.read_shape("1b", None, None, None)
        header = pc.program_text(
            shape, 24, Path("/t.bin"), grad_clip=True, train_args=s.train_args
        )
        self.assertIn('grad_clip=1.0, precision="bf16"):', header)

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

    def test_existing_build_directories_keep_their_names(self) -> None:
        # Results of earlier campaigns live under these directory names; adding the
        # grad_clip axis to the build key must not rename them.
        self.assertEqual(
            pc.flag_key(pc.arm("nsl_tf32").build_key),
            "nsl_sourcead_checkpointblocks_checkpointselective_fusermsnormbackward_fusewgradaccum",
        )
        self.assertEqual(
            pc.flag_key(pc.arm("nsl_bf16_compress@after").build_key),
            "nsl_sourcead_checkpointblocks_checkpointselective_fusermsnormbackward_"
            "fusewgradaccum_checkpointcompress_bf16__matmul_bf161@after",
        )


def feature_rules() -> list[tuple[str, str, str, bool]]:
    """(flag, Requires|Conflicts, partner, clap) for every rule in the compiler's
    composition registry, `crates/nsl-cli/src/feature_rules.rs`. Clap rules name the
    partner by its clap FIELD (`optim_state_offload`), source rules by its spelling
    (`grad_clip=`, `--weight-stream`, or prose for a non-flag partner)."""
    src = (pc.REPO / "crates" / "nsl-cli" / "src" / "feature_rules.rs").read_text()
    clap = re.findall(r'clap_rule\("(--[a-z0-9-]+)", RuleKind::(\w+), "(\w+)"\)', src)
    source = re.findall(r'src_rule\(\s*"(--[a-z0-9-]+)",\s*RuleKind::(\w+),\s*"([^"]+)"', src)
    assert len(clap) >= 15 and len(source) >= 20, "the registry's layout changed"
    return [(f, k, "--" + o.replace("_", "-"), True) for f, k, o in clap] + [
        (f, k, o, False) for f, k, o in source
    ]


class MemoryArms(unittest.TestCase):
    def test_arms_respect_the_compilers_composition_rules(self) -> None:
        rules = feature_rules()
        for prec in ("tf32", "bf16"):
            for suffix in [*pc.MEMORY_ARMS, *pc.OPTIM_ARMS]:
                a = pc.arm(f"nsl_{prec}_{suffix}")
                flags = set(a.flags)
                # A flag followed by a value (`--zero-stage 1`): its source rules are
                # conditions on the value ("--zero-stage 3 requires ..."), not on the
                # flag, which this table cannot express. Its clap rules still apply.
                valued = {f for f, v in itertools.pairwise(a.flags) if not v.startswith("--")}
                for flag, kind, partner, clap in rules:
                    if flag not in flags or (not clap and flag in valued):
                        continue
                    if not (partner.startswith("--") or partner == "grad_clip="):
                        continue  # prose partner ("FASE-Deferred plan")
                    if partner == "--source-ad-mode":  # clap group: either spelling
                        has = bool(flags & {"--source-ad", "--pretrain-optimized"})
                    elif partner == "grad_clip=":
                        has = a.grad_clip
                    else:
                        has = partner in flags
                    if kind == "Conflicts":
                        self.assertFalse(has, f"{a.name}: {flag} refuses {partner}")
                    elif kind == "Requires" and partner.startswith("--"):
                        self.assertTrue(has, f"{a.name}: {flag} requires {partner}")

    def test_the_offload_and_layerwise_arms_drop_only_what_is_refused(self) -> None:
        prod = set(pc.PROD_FLAGS)
        offload = set(pc.arm("nsl_bf16_offload").flags)
        self.assertEqual(prod - offload, {"--fuse-wgrad-accum"})
        self.assertEqual(offload - prod, {"--optim-state-offload"})
        lw = pc.arm("nsl_bf16_lw")
        self.assertEqual(
            set(lw.flags),
            {"--source-ad", "--checkpoint-blocks", "--fuse-rmsnorm-backward", "--layerwise-accum"},
        )
        self.assertFalse(lw.grad_clip)
        # Each layerwise arm has a control with the same drops and no option.
        self.assertEqual(
            set(pc.arm("nsl_bf16_blocks_nowgrad_noclip").flags) | {"--layerwise-accum"},
            set(lw.flags),
        )
        self.assertFalse(pc.arm("nsl_bf16_blocks_nowgrad_noclip").grad_clip)

    def test_clip_is_part_of_the_build(self) -> None:
        shape = pc.read_shape("1b", None, None, None)
        on = pc.program_text(shape, 24, Path("/t.bin"), grad_clip=True)
        off = pc.program_text(shape, 24, Path("/t.bin"), grad_clip=False)
        train = 'train(model=m, epochs=1, grad_accumulation=4, schedule_unit="optimizer_step"'
        self.assertIn(train + ", grad_clip=1.0):\n", on)
        self.assertIn(train + "):\n", off)
        self.assertNotIn("grad_clip=", off)
        clip, noclip = pc.arm("nsl_bf16_nowgrad"), pc.arm("nsl_bf16_nowgrad_noclip")
        self.assertEqual(clip.flags, noclip.flags)
        self.assertNotEqual(pc.flag_key(clip.build_key), pc.flag_key(noclip.build_key))

    def test_optimizer_arguments_are_part_of_the_build(self) -> None:
        shape = pc.read_shape("1b", None, None, None)
        base = pc.program_text(shape, 24, Path("/t.bin"), grad_clip=True)
        m1 = pc.program_text(
            shape, 24, Path("/t.bin"), grad_clip=True, optim_args='first_moment_dtype="bf16"'
        )
        adamw = "    optimizer: AdamW(lr=0.0001, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8"
        self.assertIn(adamw + ")\n", base)
        self.assertIn(adamw + ', first_moment_dtype="bf16")\n', m1)
        # Only the optimizer line differs.
        self.assertEqual(base.replace(adamw + ")", adamw + ', first_moment_dtype="bf16")'), m1)
        a, prod = pc.arm("nsl_bf16_m1bf16"), pc.arm("nsl_bf16")
        self.assertEqual((a.flags, a.build_env, a.env), (prod.flags, prod.build_env, prod.env))
        self.assertEqual(a.optim_args, 'first_moment_dtype="bf16"')
        self.assertEqual(
            pc.flag_key(a.build_key), pc.flag_key(prod.build_key) + "__firstmomentdtypebf16"
        )
        self.assertEqual(pc.arm("nsl_tf32_m1bf16@mem").variant, "mem")

    def test_run_time_switches_share_the_binary(self) -> None:
        # NSL_WS_RESIDENT and NSL_ASYNC_ALLOC are read by the runtime: same build.
        for opt, base in (("wsall", "ws"), ("asyncalloc", "")):
            a, b = pc.arm(f"nsl_bf16_{opt}"), pc.arm("nsl_bf16" + (f"_{base}" if base else ""))
            self.assertEqual(pc.flag_key(a.build_key), pc.flag_key(b.build_key), opt)
            self.assertNotEqual(a.env, b.env, opt)


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
        self.assertEqual(
            ns.category("vectorized_elementwise_kernel<silu_kernel(...)::[lambda() (instance 1)]>"),
            "elementwise",
            "a lambda-named kernel is not an optimizer",
        )


def synthetic_trace(path: Path) -> None:
    """Two updates of one micro-batch of a two-layer model, in Nsight Systems' schema.
    Layer 1's forward GEMMs take 2 us longer than layer 0's, so each projection's median
    is the midpoint; both backward layers have the same seven GEMMs."""
    us = 1000
    seq: list[tuple[str, int]] = [("adam", 1)]  # the burst that closes update 0
    for _ in range(2):
        seq += [("embed", 2)]
        for extra in (0, 2):
            seq += [(f"gemm_{r}", d + extra)
                    for r, d in zip(ns.ROLES, (10, 20, 30, 40, 50, 60, 70), strict=True)]  # fmt: skip
            seq += [("flash_fwd", 5)]
        seq += [("lce_bwd", 1), ("flash_bwd", 8)]
        for _layer in range(2):
            seq += [(f"gemm_b{i}", d) for i, d in enumerate((100, 90, 80, 70, 60, 50, 40))]
            seq += [("flash_bwd", 8)]
        seq += [("adam", 3)]
    con = sqlite3.connect(path)
    con.execute("create table StringIds (id integer, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, demangledName int)")
    names = {n: i for i, n in enumerate(dict.fromkeys(n for n, _ in seq))}
    con.executemany("insert into StringIds values (?, ?)", [(i, n) for n, i in names.items()])
    t = 0
    for name, dur in seq:
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?)",
                    (t, t + dur * us, names[name]))  # fmt: skip
        t += dur * us + us  # 1 us gap between kernels, far below the burst gap
    con.commit()
    con.close()


class LayerTables(unittest.TestCase):
    def test_roles_attention_and_backward_layer(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.sqlite"
            synthetic_trace(db)
            lt = ns.layer_tables(db, "adam", 0, 0.0005, ("embed", "lce_bwd", "adam", ""), 2,
                                 "flash_bwd")  # fmt: skip
        self.assertEqual(
            [lt[f"fwd_{r}_us"] for r in ns.ROLES], [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0]
        )
        self.assertEqual(lt["fwd_attention_per_layer_us"], 5.0)
        self.assertEqual(lt["bwd_layer_gemm_us"], 490.0)
        self.assertEqual(lt["bwd_layer_top6_gemm_us"], 450.0)


if __name__ == "__main__":
    unittest.main()
