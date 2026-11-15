#!/usr/bin/env python3
"""Where one optimizer update's time goes, from an Nsight Systems SQLite export.

    nsys_summary.py trace.sqlite --boundary 'adam|fase' [--skip 8] [--json out.json]

Updates are found in the trace itself: each burst of optimizer kernels (names matching
`--boundary`) ends one update, so the window of update i runs from the end of burst
i-1 to the end of burst i. The first `--skip` updates (warm-up) are dropped. Per
update, averaged over the kept ones:

* wall time, GPU-busy time (the union of kernel, memcpy and memset intervals), and
  the idle remainder: time the GPU waited on the host;
* kernel time by category (GEMM, attention, cross-entropy, norm, optimizer, ...)
  and by kernel, with launch counts;
* driver/runtime API calls: count and host time per name (launches, synchronising
  copies, explicit synchronisations, allocations).

With `--fwd-start` and `--bwd-start` (kernel-name regexes for the first kernel of a
micro-batch's forward and of its backward) each update is also cut into phases on the
GPU timeline: `pre` (update start to the first forward), `fwd` (forward and loss),
`bwd` (backward, recompute included), `opt` (from the first `--opt-start` kernel to
the end of the optimizer burst). Kernels in a backward that match `--recompute` are
counted separately as recompute.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORIES: list[tuple[str, str]] = [
    # torch.compile's generated kernels are named after every op they fuse
    # ("triton_poi_fused__scaled_dot_product_flash_attention_..._mul_neg" is RoPE),
    # so they get their own bucket rather than whichever op name matches first.
    ("triton_fused", r"^triton_"),
    ("attention", r"flash|fmha|sdpa|attention|attn"),
    ("cross_entropy", r"cross_entropy|crossentropy|lce|log_softmax|logsoftmax|nll|softmax"),
    # Not "lamb": PyTorch's elementwise kernels are named after their lambdas.
    ("optimizer", r"adam|fase_|multi_tensor_apply|FusedOptimizer|LpNorm|lpnorm_cleanup"),
    ("gemm", r"gemm|nvjet|cutlass|xmma|cublas|sgemm|splitk|sm\d+_xmma|ampere_|Kernel2"),
    ("norm", r"rmsnorm|rms_norm|layernorm|norm"),
    ("rope", r"rope|rotary"),
    ("reduce", r"reduce|sum_dim|sum_sq|argmax|max_dim|mean"),
    ("embedding", r"embedding|gather|index_select|indexSelect|scatter"),
    ("copy_cast", r"copy|cast|convert|contiguous|transpose|permute|to_bf16|to_f16|cat"),
    ("elementwise", r"elementwise|ew_|silu|swiglu|mul|add|sub|div|dropout|fill|triton|unrolled"),
]


def category(name: str) -> str:
    for cat, pat in CATEGORIES:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


@dataclass
class Interval:
    start: int
    end: int
    name: str


def load(db: Path) -> tuple[list[Interval], list[Interval], list[Interval]]:
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    strings = dict(con.execute("select id, value from StringIds"))
    kernels = [
        Interval(s, e, strings.get(n, str(n)))
        for s, e, n in con.execute(
            "select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start"
        )
    ]
    copies: list[Interval] = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        kinds = {0: "unknown", 1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
        copies += [
            Interval(s, e, f"[memcpy {kinds.get(k, k)}]")
            for s, e, k in con.execute(
                "select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start"
            )
        ]
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        copies += [
            Interval(s, e, "[memset]")
            for s, e in con.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET")
        ]
    copies.sort(key=lambda i: i.start)
    api: list[Interval] = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api = [
            Interval(s, e, strings.get(n, str(n)))
            for s, e, n in con.execute(
                "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"
            )
        ]
    return kernels, copies, api


def update_ends(kernels: list[Interval], boundary: str, gap_ns: int) -> list[int]:
    """End time of each burst of optimizer kernels."""
    pat = re.compile(boundary, re.IGNORECASE)
    ends: list[int] = []
    last_end: int | None = None
    for k in kernels:
        if not pat.search(k.name):
            continue
        if last_end is not None and k.start - last_end > gap_ns:
            ends.append(last_end)
        last_end = k.end if last_end is None else max(last_end, k.end)
    if last_end is not None:
        ends.append(last_end)
    return ends


def busy_ns(intervals: list[Interval], lo: int, hi: int) -> int:
    """Length of the union of `intervals` clipped to [lo, hi)."""
    total, cur_s, cur_e = 0, None, None
    for iv in sorted(intervals, key=lambda i: i.start):
        s, e = max(iv.start, lo), min(iv.end, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None and cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None and cur_s is not None:
        total += cur_e - cur_s
    return total


def within(items: list[Interval], lo: int, hi: int) -> list[Interval]:
    starts = [i.start for i in items]
    a, b = bisect.bisect_left(starts, lo), bisect.bisect_left(starts, hi)
    return items[a:b]


@dataclass
class Phase:
    wall_ms: float = 0.0
    busy_ms: float = 0.0
    kernel_ms: float = 0.0
    launches: float = 0.0
    recompute_ms: float = 0.0
    recompute_launches: float = 0.0


def phase_cuts(
    ks: list[Interval], lo: int, hi: int, fwd_start: str, bwd_start: str, opt_start: str
) -> list[tuple[str, int, int]]:
    """(label, start, end) regions of one update window."""
    opt_at = next((k.start for k in ks if re.search(opt_start, k.name, re.IGNORECASE)), hi)
    starts = [
        k.start for k in ks if k.start < opt_at and re.search(fwd_start, k.name, re.IGNORECASE)
    ]
    if not starts:
        return [("pre", lo, opt_at), ("opt", opt_at, hi)]
    cuts: list[tuple[str, int, int]] = [("pre", lo, starts[0])]
    bounds = [*starts, opt_at]
    for s0, s1 in itertools.pairwise(bounds):
        b = next(
            (
                k.start
                for k in ks
                if s0 < k.start < s1 and re.search(bwd_start, k.name, re.IGNORECASE)
            ),
            s1,
        )
        cuts += [("fwd", s0, b), ("bwd", b, s1)]
    cuts.append(("opt", opt_at, hi))
    return cuts


def summarise(
    db: Path,
    boundary: str,
    skip: int,
    gap_ms: float,
    top: int,
    marks: tuple[str, str, str, str] | None,
) -> dict[str, Any]:
    kernels, copies, api = load(db)
    ends = update_ends(kernels, boundary, int(gap_ms * 1e6))
    if len(ends) < skip + 2:
        raise SystemExit(f"{db}: {len(ends)} optimizer bursts found, need > {skip + 1}")
    windows = list(itertools.pairwise(ends[skip:]))
    n = len(windows)
    by_kernel: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_cat: dict[str, float] = defaultdict(float)
    by_api: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    phases: dict[str, Phase] = defaultdict(Phase)
    phase_cat: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    wall = busy = kern_total = copy_total = 0.0
    for lo, hi in windows:
        ks, cs = within(kernels, lo, hi), within(copies, lo, hi)
        wall += (hi - lo) / 1e6
        busy += busy_ns(ks + cs, lo, hi) / 1e6
        for k in ks:
            d = (k.end - k.start) / 1e6
            by_kernel[k.name][0] += d
            by_kernel[k.name][1] += 1
            by_cat[category(k.name)] += d
            kern_total += d
        for c in cs:
            d = (c.end - c.start) / 1e6
            by_kernel[c.name][0] += d
            by_kernel[c.name][1] += 1
            by_cat["memcpy_memset"] += d
            copy_total += d
        for x in within(api, lo, hi):
            by_api[x.name][0] += (x.end - x.start) / 1e6
            by_api[x.name][1] += 1
        if marks is None:
            continue
        fwd_start, bwd_start, opt_start, recompute = marks
        for label, s0, s1 in phase_cuts(ks, lo, hi, fwd_start, bwd_start, opt_start):
            pk = within(kernels, s0, s1)
            ph = phases[label]
            ph.wall_ms += (s1 - s0) / 1e6
            ph.busy_ms += busy_ns(pk + within(copies, s0, s1), s0, s1) / 1e6
            ph.kernel_ms += sum((k.end - k.start) / 1e6 for k in pk)
            ph.launches += len(pk)
            for k in pk:
                phase_cat[label][category(k.name)] += (k.end - k.start) / 1e6
                if label == "bwd" and recompute and re.search(recompute, k.name, re.IGNORECASE):
                    ph.recompute_ms += (k.end - k.start) / 1e6
                    ph.recompute_launches += 1
    res: dict[str, Any] = {
        "trace": str(db),
        "updates_found": len(ends),
        "updates_used": n,
        "per_update": {
            "wall_ms": wall / n,
            "gpu_busy_ms": busy / n,
            "gpu_idle_ms": (wall - busy) / n,
            "kernel_ms": kern_total / n,
            "memcpy_memset_ms": copy_total / n,
            "kernel_launches": sum(v[1] for k, v in by_kernel.items() if not k.startswith("[")) / n,
        },
        "categories_ms": {k: v / n for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "top_kernels": [
            {"name": k, "ms": v[0] / n, "count": v[1] / n, "category": category(k)}
            for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:top]
        ],
        "api": [
            {"name": k, "ms": v[0] / n, "count": v[1] / n}
            for k, v in sorted(by_api.items(), key=lambda kv: -kv[1][0])[:25]
        ],
    }
    if marks is not None:
        res["phases"] = {
            k: {f: getattr(v, f) / n for f in Phase.__dataclass_fields__}
            | {"categories_ms": {c: t / n for c, t in sorted(phase_cat[k].items(),
                                                             key=lambda kv: -kv[1])}}
            for k, v in phases.items()
        }  # fmt: skip
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trace", type=Path)
    ap.add_argument("--boundary", required=True, help="regex for optimizer kernel names")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--gap-ms", type=float, default=50.0)
    ap.add_argument("--fwd-start", help="regex: first kernel of a micro-batch's forward")
    ap.add_argument("--bwd-start", help="regex: first kernel of a micro-batch's backward")
    ap.add_argument("--opt-start", help="regex: first optimizer-phase kernel (default --boundary)")
    ap.add_argument("--recompute", default="", help="regex: forward kernels seen in a backward")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    marks = None
    if a.fwd_start and a.bwd_start:
        marks = (a.fwd_start, a.bwd_start, a.opt_start or a.boundary, a.recompute)
    res = summarise(a.trace, a.boundary, a.skip, a.gap_ms, a.top, marks)
    pu = res["per_update"]
    print(f"{a.trace.name}: {res['updates_used']} updates (of {res['updates_found']})")
    print(
        f"per update: wall {pu['wall_ms']:.1f} ms, GPU busy {pu['gpu_busy_ms']:.1f} ms "
        f"({100 * pu['gpu_busy_ms'] / pu['wall_ms']:.1f}%), idle {pu['gpu_idle_ms']:.1f} ms, "
        f"{pu['kernel_launches']:.0f} kernel launches"
    )
    print("\n| category | ms/update | share of wall |\n|---|---|---|")
    for k, v in res["categories_ms"].items():
        print(f"| {k} | {v:.1f} | {100 * v / pu['wall_ms']:.1f}% |")
    if "phases" in res:
        print(
            "\n| phase | wall ms | GPU busy ms | kernel ms | launches | recompute ms "
            "| top categories |\n|---|---|---|---|---|---|---|"
        )
        for k in ("pre", "fwd", "bwd", "opt"):
            v = res["phases"].get(k)
            if v is None:
                continue
            cats = ", ".join(f"{c} {t:.0f}" for c, t in list(v["categories_ms"].items())[:4])
            print(
                f"| {k} | {v['wall_ms']:.1f} | {v['busy_ms']:.1f} | {v['kernel_ms']:.1f} | "
                f"{v['launches']:.0f} | {v['recompute_ms']:.1f} | {cats} |"
            )
    print("\n| kernel | category | ms/update | launches/update |\n|---|---|---|---|")
    for k in res["top_kernels"]:
        print(f"| `{k['name'][:90]}` | {k['category']} | {k['ms']:.2f} | {k['count']:.0f} |")
    print("\n| API | ms/update | calls/update |\n|---|---|---|")
    for x in res["api"]:
        print(f"| {x['name']} | {x['ms']:.2f} | {x['count']:.0f} |")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
