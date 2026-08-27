#!/usr/bin/env python3
"""Splice a training checkpoint's θ into a full model .nslm by tensor NAME.

Why: `model_load` reads a full-model file (every tensor, 338 at 1B —
params AND derived buffers) positionally; a `checkpoint_save` file carries
only the 146 TRAINED params, so it cannot be loaded for inference/val
directly, and the recipe's val loops only run at epoch end. Mid-chain
inspection gates (the staged 1B intermediate run) need held-out numbers on
a checkpoint. Both file kinds share one format (NSLM v1: 16-byte header +
JSON param table with per-tensor name/shape/dtype/offset/nbytes + packed
data), so this tool takes an architecture-identical full-model TEMPLATE,
overwrites every tensor whose name appears in the checkpoint, and writes a
model_load-able result. Names present in the template but not the
checkpoint (RoPE tables, masks — derived, training-invariant) pass through
from the template.

Refusals, not defaults: a checkpoint name missing from the template, or a
shape/dtype/nbytes mismatch on any spliced tensor, aborts. A silent skip
would score the template's weights under the checkpoint's name.

Known-answer validation (2026-08-27): splicing pilot_lr3e5_state.nslm
(opt-6000) into pilot_lr3e5_final.nslm (opt-6001) must reproduce the arm's
recorded VAL_LOSS_STACK 7.392 / VAL_LOSS_WEB 8.009 to within the one
min_lr update the two differ by.

Usage:
  python3 tools/nslm_splice.py CHECKPOINT.nslm TEMPLATE.nslm OUT.nslm
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

MAGIC = b"NSLM"


def canon_ckpt(name: str) -> str:
    """checkpoint_save names params through the model VARIABLE
    ('m.blocks.0.attn.wq'): drop that first segment unconditionally and
    rewrite dotted numeric segments as index brackets. Asymmetric with
    canon_tmpl on purpose — a generic one-function rule mangles template
    names like 'final_norm.weight'."""
    idx = re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", name)
    _, _, rest = idx.partition(".")
    return rest or idx


def canon_tmpl(name: str) -> str:
    """model_save names are already field paths ('blocks[0].attn.wq',
    'final_norm.weight'): rewrite any dotted numeric segments, nothing
    else."""
    return re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", name)


def read_nslm(path: Path) -> tuple[dict, bytes, int]:
    """Returns (header dict, whole file bytes, data section start)."""
    data = path.read_bytes()
    if data[0:4] != MAGIC:
        sys.exit(f"{path}: not an NSLM file")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 1:
        sys.exit(f"{path}: NSLM version {version}, this tool reads 1")
    hsize = struct.unpack_from("<Q", data, 8)[0]
    header = json.loads(data[16:16 + hsize])
    # The writer 64-aligns the data section after the header
    # (checkpoint.rs: data_start = total_header + padding). Missing this
    # shifted every splice by up to 63 bytes — caught by the known-answer
    # gate as garbage scalar fields and a zero-sized reshape.
    total = 16 + hsize
    return header, data, total + ((64 - (total % 64)) % 64)


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ckpt_path, tmpl_path, out_path = map(Path, sys.argv[1:4])

    ckpt_hdr, ckpt, ckpt_data0 = read_nslm(ckpt_path)
    tmpl_hdr, tmpl, tmpl_data0 = read_nslm(tmpl_path)

    tmpl_by_name = {canon_tmpl(p["name"]): p for p in tmpl_hdr["params"]}
    dup = len(tmpl_hdr["params"]) - len(tmpl_by_name)
    if dup:
        sys.exit(f"{tmpl_path}: {dup} duplicate tensor name(s) — name-keyed "
                 f"splicing is ambiguous; refusing")

    out = bytearray(tmpl)
    spliced = 0
    for p in ckpt_hdr["params"]:
        t = tmpl_by_name.get(canon_ckpt(p["name"]))
        if t is None:
            sys.exit(f"checkpoint tensor '{p['name']}' has no counterpart in "
                     f"the template — architecture mismatch; refusing")
        for k in ("shape", "dtype", "nbytes"):
            if p[k] != t[k]:
                sys.exit(f"'{p['name']}': {k} mismatch (checkpoint {p[k]} vs "
                         f"template {t[k]}); refusing")
        src0 = ckpt_data0 + p["offset"]
        dst0 = tmpl_data0 + t["offset"]
        n = p["nbytes"]
        out[dst0:dst0 + n] = ckpt[src0:src0 + n]
        spliced += 1

    Path(out_path).write_bytes(bytes(out))
    print(f"spliced {spliced}/{len(tmpl_hdr['params'])} tensors "
          f"({len(ckpt_hdr['params'])} in checkpoint) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
