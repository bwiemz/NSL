#!/usr/bin/env python3
"""Compare the uninterrupted and resumed arms field by field.

The claim under test is Brandon's: "an intentional interruption must not
change which mathematical training trajectory is REQUESTED". That splits into
two kinds of evidence and they are reported separately, because conflating
them is how a resume bug hides behind "GPU nondeterminism".

  REQUESTED TRAJECTORY (must be EXACT -- these are integers, strings and
  seeds, and no floating-point reduction can excuse a difference):
      loader_slot, loader_id, loader_epoch, rng_seed, rng_pos_hi/lo,
      gpu_dropout_ctr, step_count, train_cfg, exec fingerprint,
      and the scheduler LR those imply.

  NUMERICAL OUTCOME (loss stream, optimizer moments, theta): under
      --deterministic these should ALSO be bit-exact, and the UNINT1 vs UNINT2
      control establishes whether that holds today. Without that control a
      difference here is unattributable.
"""
import hashlib, json, math, pathlib, re, struct, sys

PI = 3.14159265358979323846358979323846

def sha(p, limit=None):
    h = hashlib.sha256(); n = 0
    with open(p, "rb") as f:
        while (c := f.read(1 << 22)):
            if limit and n + len(c) > limit: h.update(c[:limit-n]); break
            h.update(c); n += len(c)
    return h.hexdigest()

def hdr(p):
    with open(p, "rb") as f:
        magic = f.read(4); ver = struct.unpack("<I", f.read(4))[0]
        hs = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hs).decode().rstrip("\x00"))

def cfg(s):
    return dict(kv.split("=", 1) for kv in (s or "").split(",") if "=" in kv)

def wcos(base, step, warm, tot, mn):
    if step < warm: return base * (step/warm) if warm else base
    if tot <= warm: return mn
    p = (step-warm)/(tot-warm)
    return mn if p >= 1.0 else mn + (base-mn)*0.5*(1.0+math.cos(PI*p))

def losses(log):
    """Steps print as a bare integer; losses print as `tensor([<float>])`.

    An earlier version of this parser expected a bare float, matched NOTHING,
    and the comparison then declared two EMPTY lists bit-identical -- a
    vacuous pass reported as evidence. The caller asserts a non-zero count for
    that reason; see the both-sides-empty trap in PR #519.
    """
    out, prev = [], None
    for l in open(log, errors="ignore"):
        l = l.strip()
        if re.fullmatch(r"\d+", l): prev = int(l); continue
        m = re.fullmatch(r"tensor\(\[(-?[\d.]+(?:e-?\d+)?)\]\)", l)
        if prev is not None and m:
            out.append((prev, m.group(1))); prev = None   # keep the STRING
    return out

C = pathlib.Path(__file__).resolve().parent
arms = {}
for name in ("unint1", "unint2", "resume"):
    d = C / name
    th, op = d/"state.nslm", d/"state.nslm.optim"
    if not th.exists(): print(f"MISSING: {th}"); sys.exit(2)
    o = hdr(op); r = o.get("resume", {}); tc = cfg(r.get("train_cfg", ""))
    ls = losses(d/"run.log")
    if (d/"run2.log").exists(): ls = ls + losses(d/"run2.log")
    arms[name] = dict(theta=sha(th), optim=sha(op), step=o.get("step_count"),
                      resume=r, tc=tc, losses=ls,
                      lr=wcos(float(tc["lr"]), o.get("step_count", 0), float(tc["sp1"]),
                              float(tc["sp2"]), float(tc["sp3"])) if tc.get("sched")=="warmup_cosine" else None)

STATE = ["loader_slot","loader_id","loader_epoch","rng_seed","rng_pos_hi",
         "rng_pos_lo","gpu_dropout_ctr","train_epoch","exec","train_cfg"]

def cmp(a, b, label):
    A, B = arms[a], arms[b]
    print(f"\n=== {label}: {a} vs {b} ===")
    print("  -- REQUESTED TRAJECTORY (must be exact) --")
    ok = True
    for k in ["step_count"] + STATE:
        va = A["step"] if k=="step_count" else A["resume"].get(k)
        vb = B["step"] if k=="step_count" else B["resume"].get(k)
        same = va == vb; ok &= same
        if not same: print(f"    DIFFER {k}: {va!r} vs {vb!r}")
    la, lb = A["lr"], B["lr"]
    same = (la == lb); ok &= same
    if not same: print(f"    DIFFER scheduler_lr: {la!r} vs {lb!r}")
    print(f"    {'ALL EXACT' if ok else 'MISMATCH ABOVE'}  ({len(STATE)+2} fields)")

    print("  -- NUMERICAL OUTCOME --")
    na, nb = len(A["losses"]), len(B["losses"])
    if na == 0 or nb == 0:
        print(f"    loss stream: VACUOUS ({na} vs {nb} parsed) -- REFUSING to call this a match")
        return False
    print(f"    loss stream: {na} vs {nb} steps", end="")
    if na == nb:
        d = [(s, x, y) for (s, x), (_, y) in zip(A["losses"], B["losses"]) if x != y]
        print(f" -- {'BIT-IDENTICAL' if not d else f'{len(d)} differ, first at step {d[0][0]}: {d[0][1]} vs {d[0][2]}'}")
    else:
        print(" -- LENGTH MISMATCH")
    print(f"    theta sha256 : {'MATCH' if A['theta']==B['theta'] else 'DIFFER'}  {A['theta'][:16]} / {B['theta'][:16]}")
    print(f"    optim sha256 : {'MATCH' if A['optim']==B['optim'] else 'DIFFER'}  {A['optim'][:16]} / {B['optim'][:16]}")
    return ok

print("Final micro-step per arm:", {k: v["step"] for k, v in arms.items()})
c = cmp("unint1", "unint2", "CONTROL (two uninterrupted runs)")
e = cmp("unint1", "resume", "TEST (uninterrupted vs save/exit/resume)")
print("\n" + "="*66)
if not c:
    print("CONTROL FAILED: two uninterrupted runs disagree on requested state.")
    print("Nothing can be attributed to the interruption.")
else:
    print("Control holds: two uninterrupted runs request the identical trajectory.")
    print("=> the TEST comparison above is attributable to the interruption.")
