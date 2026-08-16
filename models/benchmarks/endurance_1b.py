#!/usr/bin/env python3
"""Milestone B endurance + certification harness for the canonical 1B@2048
single-GPU workload (models/coder1b/pretrain_1b2048.nsl).

One invocation runs the milestone's exit criteria as ASSERTIONS, not prose:

  phase 1  endurance   N optimizer steps (default 110) of the SR-BF16 arm,
                       checkpoint saved every 50:
                       - fused LM-CE proven active: compile witnesses
                         ([lm-head-fusion] inferred / [csla] fused-ce
                         tape-carry: 1 slots) AND runtime launch counters > 0
                       - VRAM high-water FLAT: the in-program
                         gpu_peak_bytes() stream must stop growing after the
                         warmup fraction — byte equality, not a trend eyeball
                       - no unexpected PCIe weight traffic: the streaming
                         stack's own byte counters read 0, and steady-state
                         [host-profile] pcie stays under --pcie-mib-per-step
                       - residency decision recorded machine-readably
                         (NSL_WS_DECISION_JSON) and validated against the
                         measured at-peak surfaces
                       - safety margin: allocator peak vs total VRAM must
                         clear --margin-gib
  phase 2  f32         the F32 reference arm, same schedule — banks the
                       reference trajectory next to the SR-BF16 one
  phase 3  resume      new process, checkpoint_load of phase 1's step-50
                       checkpoint, token stream sliced at the same boundary;
                       asserts the [checkpoint] resumed witness and loss
                       continuity against the parent tail (no re-warm spike).
                       Bit-identity is NOT asserted at 1B — TF32 GEMMs and
                       fused-CE backward atomics make that an environment
                       property, not a correctness property (see
                       long_run_drift_gpu_gate's control-vs-control
                       doctrine); the CPU gate train_checkpoint_gate.rs owns
                       bit-exactness.

Results: <out>/endurance_result.json plus per-phase workdirs with full
stdout/stderr, the loss+peak streams, and the residency-decision JSON. An
OOM or a failed assertion is a hard exit — this harness is the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from matrix_bench import ARMS, REPO, SCALES, build_binary  # noqa: E402

CANONICAL = REPO / "models" / "coder1b" / "pretrain_1b2048.nsl"
LOSS_RE = re.compile(r"^tensor\(\[([0-9.eE+-]+)\]\)\s*$")
PCIE_RE = re.compile(r"pcie\s+h2d=([0-9.]+) MiB\s+d2h=([0-9.]+) MiB")
PEAK_RESERVED_RE = re.compile(r"Peak reserved:\s+([0-9.]+) (GB|MB)")

MB, SEQ, ACCUM = 2, 2048, 2
TOKENS_PER_OPT_STEP = MB * SEQ * ACCUM


def gen_tokens(path: Path, count: int) -> None:
    # A stale file with a DIFFERENT count silently changes every phase's
    # step count and the checkpoint cadence the assertions are built on —
    # regenerate unless the size matches exactly (u16 = 2 bytes/token).
    if path.exists():
        if path.stat().st_size == count * 2:
            return
        print(f"tokens.bin is stale ({path.stat().st_size} B, want {count * 2}) "
              f"— regenerating", flush=True)
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {count} tokens at {path}", flush=True)
    subprocess.run(
        [sys.executable, str(REPO / "models/benchmarks/gen_tokens.py"),
         str(path), "--tokens", str(count)],
        check=True,
    )


def build_workdir(work: Path, tokens: Path, ckpt_args: str) -> Path:
    """Materialize the canonical program with markers rewritten."""
    work.mkdir(parents=True, exist_ok=True)
    src_dir = REPO / "models" / "coder1b"
    for name in ("model.nsl", "config.nsl"):
        (work / name).write_bytes((src_dir / name).read_bytes())
    text = CANONICAL.read_text(encoding="utf-8")
    for marker in ("B2048_TOKENS_PATH", "B2048_CKPT_ARGS"):
        # The markers also appear in the doc header — require one on a CODE
        # line, or a doc-only survivor would satisfy the guard while the
        # generated program silently trains without the rewrite.
        if not any(marker in ln for ln in text.splitlines()
                   if not ln.lstrip().startswith("#")):
            sys.exit(f"canonical file lost its {marker} marker on a code line")
    text = text.replace("B2048_TOKENS_PATH", tokens.as_posix())
    text = text.replace(" B2048_CKPT_ARGS", ckpt_args)
    prog = work / "pretrain_1b2048_generated.nsl"
    prog.write_text(text, encoding="utf-8")
    return prog


class SmiSampler:
    """Per-PID nvidia-smi peak sampler (residency_probe's approach): a LOWER
    BOUND on the driver-level peak, sampled every 100 ms."""

    def __init__(self) -> None:
        self.peak_mb = 0
        self._stop = threading.Event()
        self._pid: int | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and self._pid is not None \
                            and parts[0] == str(self._pid):
                        self.peak_mb = max(self.peak_mb, int(parts[1]))
            except Exception:
                pass
            time.sleep(0.1)

    def watch(self, pid: int) -> None:
        self._pid = pid
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        return self.peak_mb


def run_phase(tag: str, work: Path, binary: Path, env_extra: dict[str, str],
              timeout_s: int) -> dict:
    env = dict(os.environ)
    env["NSL_WS_COUNTER"] = "1"
    env["NSL_MEMSTATS"] = "1"
    env["NSL_DEBUG_MEM_ALL"] = "1"
    env["NSL_HOST_PROFILE"] = "1"
    env["NSL_WS_DECISION_JSON"] = str(work / "ws_decision.json")
    env.update(env_extra)

    # A stale decision file from an earlier aborted run would be banked as
    # this run's artifact — atexit only writes on clean-ish exits.
    (work / "ws_decision.json").unlink(missing_ok=True)

    sampler = SmiSampler()
    proc = subprocess.Popen(
        [str(binary)], cwd=work, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    sampler.watch(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Salvage whatever the pipes hold — the per-phase artifacts are the
        # whole point of this harness; a timeout without them is undebuggable.
        stdout, stderr = proc.communicate()
        (work / "run_stdout.txt").write_text(stdout or "")
        (work / "run_stderr.txt").write_text(stderr or "")
        sys.exit(f"[{tag}] timed out after {timeout_s}s — partial output at "
                 f"{work}/run_stdout.txt")
    smi_peak = sampler.stop()
    (work / "run_stdout.txt").write_text(stdout)
    (work / "run_stderr.txt").write_text(stderr)

    oom = re.search(r"out of memory", stderr, re.I) is not None
    if proc.returncode != 0 or oom:
        sys.exit(f"[{tag}] run failed (exit {proc.returncode}, oom={oom}) — "
                 f"see {work}/run_stderr.txt")
    if re.search(r"tensor\(\[\s*(nan|inf)", stdout, re.I):
        # The triplet parser below would silently DROP non-numeric losses —
        # a NaN'd run must fail here, not pass a shorter loss stream.
        sys.exit(f"[{tag}] non-finite loss in the stream — see "
                 f"{work}/run_stdout.txt")

    # (step, loss, peak_bytes) triplets from on_step.
    lines = stdout.splitlines()
    steps: list[int] = []
    losses: list[float] = []
    peaks: list[int] = []
    i = 0
    while i + 2 < len(lines):
        m = LOSS_RE.match(lines[i + 1].strip())
        if lines[i].strip().isdigit() and m and lines[i + 2].strip().isdigit():
            steps.append(int(lines[i].strip()))
            losses.append(float(m.group(1)))
            peaks.append(int(lines[i + 2].strip()))
            i += 3
        else:
            i += 1

    def witness_block(name: str) -> list[int]:
        try:
            a = lines.index(f"WITNESS_{name}_BEGIN") + 1
            b = lines.index(f"WITNESS_{name}_END")
        except ValueError:
            sys.exit(f"[{tag}] WITNESS_{name} block missing from stdout")
        return [int(x.strip()) for x in lines[a:b]]

    fused = witness_block("FUSED_LCE")
    ws = witness_block("WEIGHT_STREAM")
    peak_block = witness_block("PEAK")

    pcie_steps = [float(m.group(1))
                  for m in PCIE_RE.finditer(stderr)]

    reserved = None
    m = PEAK_RESERVED_RE.search(stderr)
    if m:
        reserved = float(m.group(1)) * (1024 if m.group(2) == "GB" else 1)

    peak_ctx: list[str] = []
    if "Top contexts at peak:" in stderr:
        tail = stderr.split("Top contexts at peak:", 1)[1].splitlines()[1:]
        for line in tail:
            if ":" in line and not line.strip().startswith("Top contexts"):
                peak_ctx.append(line.strip())
            else:
                break

    decision = None
    dj = work / "ws_decision.json"
    if dj.exists():
        decision = json.loads(dj.read_text())

    return {
        "tag": tag,
        "micro_batches": len(steps),
        "losses": list(zip(steps, losses)),
        "peak_bytes_stream": peaks,
        "fused_lce_counts": fused,
        "weight_stream_stat": ws,
        "witness_peak_block": peak_block,
        "pcie_h2d_mib_per_step": pcie_steps,
        "nvidia_smi_peak_mb_lower_bound": smi_peak,
        "peak_reserved_mb": reserved,
        "top_contexts_at_peak": peak_ctx,
        "ws_decision": decision,
        "compile_stderr": (work / "build_stderr.txt").read_text()
        if (work / "build_stderr.txt").exists() else "",
    }


def assert_fused_ce_active(tag: str, res: dict) -> None:
    cs = res["compile_stderr"]
    for needle in ("[lm-head-fusion] inferred:",
                   "[csla] fused-ce tape-carry: 1 slots"):
        if needle not in cs:
            sys.exit(f"[{tag}] EC1 FAIL: compile witness missing: {needle}")
    fwd = res["fused_lce_counts"][0] + res["fused_lce_counts"][1] \
        + res["fused_lce_counts"][3]
    bwd = res["fused_lce_counts"][2] + res["fused_lce_counts"][4]
    if fwd <= 0 or bwd <= 0:
        sys.exit(f"[{tag}] EC1 FAIL: fused-CE launch counters not live "
                 f"(fwd={fwd} bwd={bwd}) — compiled but never fired")
    print(f"[{tag}] EC1 OK: fused-CE active (fwd={fwd} bwd={bwd} launches)")


def assert_flat_high_water(tag: str, res: dict, warmup_frac: float) -> None:
    peaks = res["peak_bytes_stream"]
    if len(peaks) < 8:
        sys.exit(f"[{tag}] EC8 FAIL: too few steps to judge flatness")
    k = int(len(peaks) * warmup_frac)
    if peaks[-1] != peaks[k]:
        sys.exit(f"[{tag}] EC8 FAIL: VRAM high-water grew after warmup: "
                 f"{peaks[k]} B at micro-batch {k} -> {peaks[-1]} B at end")
    print(f"[{tag}] EC8 OK: high-water flat at {peaks[-1]/2**30:.2f} GiB "
          f"from micro-batch {k} to {len(peaks)}")


def assert_no_weight_traffic(tag: str, res: dict,
                             pcie_cap_mib: float | None) -> None:
    """EC5. `pcie_cap_mib=None` skips the GLOBAL per-step cap — used for the
    --optim-state-offload reference arm, whose ~8 GB/step of moment staging
    is MODELED traffic (the flag's whole point), not unexpected weight
    streaming. The weight-stream byte counters must read zero either way:
    they count only the θ-streaming stack's copies."""
    ws = res["weight_stream_stat"]
    h2d, d2h = ws[0], ws[1]
    if h2d != 0 or d2h != 0:
        sys.exit(f"[{tag}] EC5 FAIL: weight-stream moved bytes "
                 f"(h2d={h2d} d2h={d2h}) with a device-resident posture")
    if pcie_cap_mib is None:
        print(f"[{tag}] EC5 OK: ws traffic 0 B; global PCIe cap skipped "
              f"(offload arm — moment staging is modeled traffic)")
        return
    steady = res["pcie_h2d_mib_per_step"][2:]
    if len(steady) < 5:
        # host_profile suppresses intervals with zero attributed time, so a
        # short/empty series is a broken measurement, not a clean one — the
        # cap check must never pass vacuously.
        sys.exit(f"[{tag}] EC5 FAIL: only {len(steady)} steady-state "
                 f"[host-profile] pcie intervals captured — the PCIe check "
                 f"would be vacuous (is NSL_HOST_PROFILE reaching the run?)")
    worst = max(steady)
    if worst > pcie_cap_mib:
        sys.exit(f"[{tag}] EC5 FAIL: steady-state PCIe h2d {worst:.1f} "
                 f"MiB/step exceeds the {pcie_cap_mib} MiB cap")
    print(f"[{tag}] EC5 OK: ws traffic 0 B; steady-state pcie h2d "
          f"max {worst:.1f} MiB/step over {len(steady)} intervals")


def assert_srbf16_posture(tag: str, res: dict, work: Path) -> None:
    """The SR-BF16 arm's residency story: bf16-sr OWNS residency — its
    registration bypasses the weight-stream policy by design, so the
    modeled-decision record (EC3) belongs to the F32 phase. What this arm
    must prove instead: the bf16-authoritative mirrors actually engaged."""
    stderr = (work / "run_stderr.txt").read_text()
    m = re.search(r"\[sr-bf16\] teardown: (\d+) bf16-authoritative param\(s\), "
                  r"(\d+) SR optimizer step\(s\)", stderr)
    if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
        sys.exit(f"[{tag}] FAIL: SR-BF16 teardown witness missing or zero — "
                 f"the bf16-sr arm did not actually run bf16-authoritative")
    d = res["ws_decision"]
    if d and d.get("decided"):
        sys.exit(f"[{tag}] FAIL: the residency policy DECIDED under bf16-sr "
                 f"— the registration bypass has drifted")
    print(f"[{tag}] SR posture OK: {m.group(1)} bf16 params, "
          f"{m.group(2)} SR steps; policy correctly bypassed")


def assert_residency_decision(tag: str, res: dict) -> None:
    d = res["ws_decision"]
    if not d or not d.get("decided"):
        sys.exit(f"[{tag}] EC3 FAIL: no machine-readable residency decision")
    if d["registered"] <= 0 or d["pinned"] != d["registered"]:
        sys.exit(f"[{tag}] EC3 FAIL: expected pin-all posture on this card, "
                 f"got pinned={d['pinned']} of {d['registered']}")
    # Model-vs-reality: what actually had to fit in the free VRAM the
    # decision saw is the at-peak non-persistent surface. If this exceeds
    # free_at_decision, the pin-all decision was luck, not model.
    wb = res["witness_peak_block"]
    peak_total, w, m, v = wb[0], wb[1], wb[2], wb[3]
    # Everything at the peak that was NOT already resident when the decision
    # ran: θ (as weights/mirrors/widen views), m and v were all allocated
    # before first registration, so the decision's free_at_decision had
    # already paid for them. The rest — activations, m_partial, workspaces —
    # is what the reserve models.
    transient_need = peak_total - m - v - w
    if transient_need > d["free_at_decision_bytes"]:
        sys.exit(f"[{tag}] EC3 FAIL: at-peak transient surface "
                 f"{transient_need} B exceeds free-at-decision "
                 f"{d['free_at_decision_bytes']} B — the decision under-"
                 f"modeled the workload")
    print(f"[{tag}] EC3 OK: pin-all decision recorded (reserve "
          f"{d['reserve_bytes']/2**30:.1f} GiB, free-at-decision "
          f"{d['free_at_decision_bytes']/2**30:.1f} GiB, measured transient "
          f"need {transient_need/2**30:.1f} GiB)")


def assert_margin(tag: str, res: dict, margin_gib: float) -> None:
    total = res["weight_stream_stat"][8]
    peak = res["witness_peak_block"][0]
    margin_alloc = (total - peak) / 2**30
    smi_margin = (total / 2**20 - res["nvidia_smi_peak_mb_lower_bound"]) / 1024
    if margin_alloc < margin_gib:
        sys.exit(f"[{tag}] EC2 FAIL: allocator-peak margin "
                 f"{margin_alloc:.2f} GiB < required {margin_gib} GiB")
    print(f"[{tag}] EC2 OK: peak {peak/2**30:.2f} GiB of "
          f"{total/2**30:.2f} GiB — margin {margin_alloc:.2f} GiB "
          f"(driver-level lower-bound margin {smi_margin:.2f} GiB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsl", type=Path, default=REPO / "target/release/nsl")
    ap.add_argument("--out", type=Path, default=REPO / "target/b2048_endurance")
    ap.add_argument("--steps", type=int, default=120,
                    help="optimizer steps for the endurance phase (the token "
                         "stream gets steps+2 worth; the run consumes it all)")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--resume-steps", type=int, default=25)
    ap.add_argument("--margin-gib", type=float, default=3.0)
    ap.add_argument("--pcie-mib-per-step", type=float, default=8.0)
    ap.add_argument("--warmup-frac", type=float, default=0.3)
    ap.add_argument("--arm", default="layerwise_srbf16_fusedce_rn",
                    help="ARMS key for the endurance/SR-BF16 phase")
    ap.add_argument("--f32-arm", default="layerwise_f32_fusedce_rn",
                    help="ARMS key for the F32 reference phase")
    ap.add_argument("--skip-f32", action="store_true")
    ap.add_argument("--skip-resume", action="store_true")
    ap.add_argument("--timeout", type=int, default=5400)
    args = ap.parse_args()

    args.nsl = args.nsl.resolve()
    if not args.nsl.exists():
        sys.exit(f"compiler not found at {args.nsl} — build with:\n"
                 "  cargo build --release --features cuda --bin nsl")
    args.out.mkdir(parents=True, exist_ok=True)

    total_tokens = (args.steps + 2) * TOKENS_PER_OPT_STEP
    tokens = args.out / "tokens.bin"
    gen_tokens(tokens, total_tokens)

    result: dict = {"config": vars(args) | {"tokens": str(tokens)},
                    "phases": {}}
    for k in ("nsl", "out", "f32_arm"):
        result["config"][k] = str(result["config"][k])

    # ── phase 1: endurance (SR-BF16 canonical) ─────────────────────────
    arm = ARMS[args.arm]
    work = args.out / "endurance"
    ckpt = (f', checkpoint_save = "b2048.ckpt.nslm", '
            f'checkpoint_every = {args.ckpt_every}')
    prog = build_workdir(work, tokens, ckpt)
    binary = build_binary(args.nsl, prog, arm, args.timeout)
    if binary is None:
        sys.exit("endurance arm BUILD REFUSED — see build_stderr.txt")
    res = run_phase("endurance", work, binary, dict(arm.env), args.timeout)
    result["phases"]["endurance"] = res

    assert_fused_ce_active("endurance", res)
    assert_flat_high_water("endurance", res, args.warmup_frac)
    assert_no_weight_traffic("endurance", res, args.pcie_mib_per_step)
    assert_srbf16_posture("endurance", res, work)
    assert_margin("endurance", res, args.margin_gib)
    # Cadence from the MEASURED stream, not --steps: the token file carries
    # steps+2 worth and the run consumes all of it.
    measured_opt_steps = res["micro_batches"] // ACCUM
    n_saves = (work / "run_stderr.txt").read_text().count("[checkpoint] saved:")
    want_saves = measured_opt_steps // args.ckpt_every
    if n_saves < 1 or n_saves != want_saves:
        sys.exit(f"[endurance] EC7 FAIL: {measured_opt_steps} optimizer steps "
                 f"at every={args.ckpt_every} should save {want_saves} times, "
                 f"saw {n_saves}")
    last_save_step = want_saves * args.ckpt_every
    print(f"[endurance] checkpoints OK: {n_saves} boundary saves; the file on "
          f"disk is the step-{last_save_step} one")

    # ── phase 2: F32 reference trajectory ──────────────────────────────
    if not args.skip_f32:
        f32_arm = ARMS[args.f32_arm]
        work_f = args.out / "f32"
        prog_f = build_workdir(work_f, tokens, "")
        binary_f = build_binary(args.nsl, prog_f, f32_arm, args.timeout)
        if binary_f is None:
            sys.exit("f32 arm BUILD REFUSED — see build_stderr.txt")
        res_f = run_phase("f32", work_f, binary_f, dict(f32_arm.env),
                          args.timeout)
        result["phases"]["f32"] = res_f
        assert_fused_ce_active("f32", res_f)
        assert_flat_high_water("f32", res_f, args.warmup_frac)
        # EC3 lives HERE: the F32 arm is the one the resident-weight policy
        # governs (bf16-sr bypasses it by design — see assert_srbf16_posture).
        f32_cap = (None if "--optim-state-offload" in f32_arm.flags
                   else args.pcie_mib_per_step)
        assert_no_weight_traffic("f32", res_f, f32_cap)
        assert_residency_decision("f32", res_f)
        print(f"[f32] trajectory banked: {len(res_f['losses'])} micro-batch "
              f"losses, peak {res_f['witness_peak_block'][0]/2**30:.2f} GiB")

    # ── phase 3: checkpoint resume ─────────────────────────────────────
    if not args.skip_resume:
        # The checkpoint on disk is the LAST boundary save (each save
        # overwrites the same path), so both the token slice and the
        # step-numbering assertion anchor on last_save_step — slicing at the
        # FIRST boundary would replay steps the checkpoint already trained
        # and fail the numbering check.
        boundary_tokens = last_save_step * TOKENS_PER_OPT_STEP
        sliced = args.out / "tokens_resume.bin"
        raw = tokens.read_bytes()
        sliced.write_bytes(raw[boundary_tokens * 2:])  # u16 = 2 bytes/token
        work_r = args.out / "resume"
        prog_r = build_workdir(work_r, sliced,
                               ', checkpoint_load = "b2048.ckpt.nslm"')
        for suffix in ("", ".optim"):
            shutil.copy2(work / f"b2048.ckpt.nslm{suffix}",
                         work_r / f"b2048.ckpt.nslm{suffix}")
        binary_r = build_binary(args.nsl, prog_r, arm, args.timeout)
        if binary_r is None:
            sys.exit("resume arm BUILD REFUSED — see build_stderr.txt")
        res_r = run_phase("resume", work_r, binary_r, dict(arm.env),
                          args.timeout)
        result["phases"]["resume"] = res_r

        stderr_r = (work_r / "run_stderr.txt").read_text()
        if "[checkpoint] resumed:" not in stderr_r:
            sys.exit("[resume] EC7 FAIL: resume witness missing")
        # Loss continuity: the resumed run's first losses must sit in the
        # band of the parent's losses AROUND THE SAVE BOUNDARY (not the
        # parent's final tail — the parent trained past the checkpoint).
        boundary_mb = last_save_step * ACCUM
        around = [l for s, l in res["losses"]
                  if boundary_mb - 10 <= s <= boundary_mb]
        mu = statistics.mean(around)
        sd = statistics.pstdev(around)
        first_resumed = res_r["losses"][0][1]
        band = mu + max(5 * sd, 0.02 * mu)
        if first_resumed > band:
            sys.exit(f"[resume] EC7 FAIL: first resumed loss "
                     f"{first_resumed:.4f} above the parent boundary band "
                     f"{band:.4f} (boundary mean {mu:.4f} ± {sd:.4f}) — "
                     f"this is a re-warm, not a resume")
        # Step numbering must CONTINUE (restored counter), not restart.
        first_step = res_r["losses"][0][0]
        want_first = boundary_mb + 1
        if first_step != want_first:
            sys.exit(f"[resume] EC7 FAIL: first resumed micro-batch is "
                     f"{first_step}, expected {want_first}")
        print(f"[resume] EC7 OK: resumed at micro-batch {first_step}, first "
              f"loss {first_resumed:.4f} within parent tail band {band:.4f}")

    out_json = args.out / "endurance_result.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nALL MILESTONE B ASSERTIONS PASSED — results at {out_json}")


if __name__ == "__main__":
    main()
