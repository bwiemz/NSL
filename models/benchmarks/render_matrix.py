#!/usr/bin/env python3
"""Render matrix_bench.py's matrix_results.json as the item-6 markdown report.

Separate from the bench so a re-render never risks a re-measure, and so
several result files (e.g. the 2026-08-02 run plus a re-run of its dropped
cells) can be folded into one table. Later files win on (scale, seq,
microbatch) collisions.

Usage:
    python3 render_matrix.py results1.json [results2.json ...]
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cells: dict[tuple, dict] = {}
    gpu = ""
    for path in sys.argv[1:]:
        data = json.load(open(path))
        gpu = data.get("gpu", gpu)
        for c in data["cells"]:
            cells[(c["scale"], c["seq"], c["microbatch"])] = c

    print(f"# Item 6 — standardized throughput matrix ({gpu})\n")
    print("| scale | seq | micro | arm | ms/step | tokens/s | TFLOPS | MFU% "
          "(roofline) | peak alloc MB | peak driver MB | loss max-rel vs arm0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (scale, seq, mb), c in sorted(cells.items()):
        for arm, r in c["arms"].items():
            if not r.get("ok"):
                print(f"| {scale} | {seq} | {mb} | {arm} | — | — | — | — | — "
                      f"| — | FAILED: {r.get('why', '?')} |")
                continue
            div = r.get("loss_divergence_vs_first_arm")
            div_s = f"{div['max_rel']:.2e}" if div else "n/a"
            print(
                f"| {scale} | {seq} | {mb} | {arm} "
                f"| {r['ms_per_step_median']:.1f} "
                f"| {r['useful_tokens_per_s']:,.0f} "
                f"| {r['tflops']:.2f} "
                f"| {r['mfu_pct']:.1f} ({r['mfu_roofline']}) "
                f"| {r['peak_alloc_mb']} | {r['peak_driver_mb']} | {div_s} |"
            )
    print()
    # Attribution appendix: PCIe + admissions where captured.
    print("## Attribution (pass B — times discarded)\n")
    for (scale, seq, mb), c in sorted(cells.items()):
        for arm, r in c["arms"].items():
            att = r.get("attribution") if r.get("ok") else None
            if not att:
                continue
            pcie = att.get("pcie")
            pcie_s = (f"h2d {pcie['h2d_mib']:.0f} MiB, d2h {pcie['d2h_mib']:.1f} MiB"
                      if pcie else "n/a")
            adm = att.get("admissions", {})
            adm_s = ", ".join(f"{k}={v}" for k, v in adm.items())
            print(f"- **{scale}@{seq} mb{mb} {arm}** — PCIe {pcie_s}; {adm_s}")
            regions = att.get("regions", {})
            if regions:
                top = sorted(regions.items(), key=lambda kv: -kv[1]["ms"])[:4]
                print("  - regions: " + ", ".join(
                    f"{k} {v['ms']:.0f}ms/{v['pct']:.0f}%" for k, v in top))


if __name__ == "__main__":
    main()
