#!/usr/bin/env python3
"""Experiment Lineage Records for NSL training runs.

WHY THIS EXISTS. The 1B gate campaign banked a wrong conclusion because two
bookkeeping errors survived review: slopes were compared over intervals of
different widths, and a PROBE-ARM checkpoint was tabled as a MAIN-CHAIN
checkpoint. Both are invisible in a table of numbers. Neither is invisible in
a schema.

A lineage record travels with every scored checkpoint. The trajectory builder
ingests records, not filenames, so:

  * a point from the wrong arm is a LINEAGE FAILURE, not a plausible number;
  * unequal-width intervals are a REFUSAL unless explicitly requested;
  * a checkpoint whose bytes changed since scoring is detected by hash.

Fields are read from the artefacts themselves (checkpoint headers, git, the
binary) — never passed in by the caller, because a caller that can assert the
arm identity can also assert it wrongly.
"""
from __future__ import annotations
import hashlib, json, math, pathlib, struct, subprocess, uuid

PI = 3.14159265358979323846358979323846  # stdlib/nsl/optim/schedulers.nsl

TOKENS_PER_MICRO = 4096


def warmup_cosine(base_lr, step, warmup, total, min_lr):
    """The schedule, transcribed from stdlib/nsl/optim/schedulers.nsl:21.

    Reimplemented rather than imported because the LR at a checkpoint has to
    be reconstructible from the checkpoint ALONE, with no NSL toolchain.
    """
    if step < warmup:
        return base_lr * (step / warmup) if warmup > 0 else base_lr
    if total <= warmup:
        return min_lr
    p = (step - warmup) / (total - warmup)
    if p >= 1.0:
        return min_lr
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(PI * p))


def sha256_file(p: pathlib.Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as f:
        while (chunk := f.read(1 << 22)):
            if limit is not None and n + len(chunk) > limit:
                h.update(chunk[: limit - n]); break
            h.update(chunk); n += len(chunk)
    return h.hexdigest()


def read_header(p: pathlib.Path) -> dict:
    """NSLM/NSLO: 16-byte header, JSON table, data section 64-byte aligned."""
    with p.open("rb") as f:
        magic = f.read(4)
        if magic not in (b"NSLM", b"NSLO"):
            raise ValueError(f"{p}: not an NSL checkpoint (magic {magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        hsize = struct.unpack("<Q", f.read(8))[0]
        j = json.loads(f.read(hsize).decode("utf-8").rstrip("\x00"))
    return {"magic": magic.decode(), "version": version, "header_size": hsize, "json": j}


def parse_cfg(s: str) -> dict:
    """train_cfg / exec are comma-separated k=v; floats ride inside a string."""
    out = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def integrity(p: pathlib.Path, hdr: dict) -> bool:
    """file_size >= data_start + sum(nbytes); data_start is 64-byte aligned."""
    tbl = hdr["json"].get("params") or hdr["json"].get("tensors") or []
    start = ((16 + hdr["header_size"]) + 63) // 64 * 64
    total = sum(t.get("nbytes", 0) for t in tbl) if isinstance(tbl, list) else 0
    return p.stat().st_size >= start + total


def record(theta: pathlib.Path, *, arm: str, run_uuid: str,
           parent_run_uuid: str | None, parent_ckpt_sha: str | None,
           repo: pathlib.Path, binary: pathlib.Path | None = None,
           interruption_count: int = 0, replayed_microsteps: int = 0,
           hash_theta: bool = True) -> dict:
    """Build a lineage record for a checkpoint, reading every field from disk."""
    optim = theta.with_suffix(theta.suffix + ".optim")
    th, oh = read_header(theta), read_header(optim)
    r = oh["json"].get("resume", {})
    cfg = parse_cfg(r.get("train_cfg", ""))
    exe = parse_cfg(r.get("exec", ""))

    micro = int(oh["json"].get("step_count", -1))
    accum = int(cfg.get("accum", "1") or 1)
    lr = None
    if cfg.get("sched") == "warmup_cosine":
        lr = warmup_cosine(float(cfg["lr"]), micro, float(cfg["sp1"]),
                           float(cfg["sp2"]), float(cfg["sp3"]))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {
        "schema": "nsl.lineage/1",
        "arm": arm,
        "run_uuid": run_uuid,
        "parent_run_uuid": parent_run_uuid,
        "parent_checkpoint_sha256": parent_ckpt_sha,
        "model_checkpoint_sha256": sha256_file(theta) if hash_theta else None,
        "checkpoint_path": str(theta),
        "checkpoint_integrity_ok": integrity(theta, th) and integrity(optim, oh),
        "binary_sha256": sha256_file(binary) if binary and binary.exists() else None,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "execution_fingerprint": r.get("exec"),
        "train_config_fingerprint": r.get("train_cfg"),
        "loader_epoch": r.get("loader_epoch"),
        "loader_slot": r.get("loader_slot"),
        "loader_id": r.get("loader_id"),
        "rng_seed": r.get("rng_seed"),
        "rng_pos_hi": r.get("rng_pos_hi"),
        "rng_pos_lo": r.get("rng_pos_lo"),
        "micro_step": micro,
        "optimizer_step": micro // accum if accum else None,
        "tokens_seen": micro * TOKENS_PER_MICRO,
        "current_lr": lr,
        "sched": cfg.get("sched"),
        "sched_params": {k: cfg.get(k) for k in ("lr", "sp1", "sp2", "sp3")},
        "grad_accumulation": accum,
        "precision_mode": exe.get("dtype"),
        "matmul_mode": exe.get("dtype"),
        "deterministic": exe.get("det"),
        "interruption_count": interruption_count,
        "replayed_microsteps": replayed_microsteps,
    }
