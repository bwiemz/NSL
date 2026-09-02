#!/usr/bin/env python3
"""Generate the three schedule-campaign arm recipes from pretrain_prod.nsl.

The arms must differ in EXACTLY the scheduler/optimizer LR line and the
checkpoint paths. Hand-copying three near-identical 350-line recipes is how
an arm silently acquires a second difference, so they are generated here and
the generator ASSERTS the realised diff is only the intended lines.

LR parameters are derived from the branch checkpoint's own train_cfg record
(read by read_branch_lr.py), never from a number in a comment.
"""
import math, pathlib, re, sys, json

PI = 3.14159265358979323846358979323846  # the literal in stdlib/nsl/optim/schedulers.nsl

def warmup_cosine(base_lr, step, warmup, total, min_lr):
    if step < warmup:
        return base_lr * (step / warmup) if warmup > 0 else base_lr
    if total <= warmup:
        return min_lr
    p = (step - warmup) / (total - warmup)
    if p >= 1.0:
        return min_lr
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(PI * p))

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "models/coder1b/pretrain_prod.nsl"
OUT = ROOT / "campaign-sched3"
LIN = json.loads((OUT / "branch_lineage.json").read_text())

BRANCH_STEP = LIN["micro_step"]                 # 64000
END_STEP    = LIN["arm_end_micro"]              # 96000
BASE, W, T_PROD, MIN = LIN["lr"], LIN["sp1"], LIN["sp2"], LIN["sp3"]
LR_BRANCH = warmup_cosine(BASE, BRANCH_STEP, W, T_PROD, MIN)

# Arm B: identical LR at the branch, cosine sized so the arm IS the remaining
# decay (ends exactly at min_lr at END_STEP).
T_SHORT = float(END_STEP)
k_b = 0.5 * (1.0 + math.cos(PI * (BRANCH_STEP - W) / (T_SHORT - W)))
BASE_B = MIN + (LR_BRANCH - MIN) / k_b
# Arm C: exactly 0.5x B pointwise. Halving base AND min_lr halves the whole
# curve, because lr = min + (base-min)*k is linear in (base, min) jointly.
BASE_C, MIN_C = BASE_B / 2.0, MIN / 2.0

def fmt(x):
    """Render a float at FULL float64 round-trip precision.

    A 12-decimal render moved arm B's branch LR by ~9 parts per billion --
    physically irrelevant, but there is no reason to carry an avoidable
    difference into an experiment whose whole point is that the arms differ in
    exactly one thing. 17 significant digits round-trips float64 exactly;
    `eps=1e-8` in the production recipe proves the parser takes exponent form.
    """
    s = f"{x:.17g}"
    assert float(s) == x, f"{x!r} does not round-trip through {s!r}"
    return s

ARMS = {
    "A": dict(lr=BASE,   warm=W, total=T_PROD,  minlr=MIN,
              why="control: the production schedule, unchanged"),
    "B": dict(lr=BASE_B, warm=W, total=T_SHORT, minlr=MIN,
              why="same LR at the branch, aggressive decay to min_lr"),
    "C": dict(lr=BASE_C, warm=W, total=T_SHORT, minlr=MIN_C,
              why="0.5x arm B pointwise: same shape, half the magnitude"),
}

src = SRC.read_text()
OPT_RE = re.compile(r"^(\s*optimizer: AdamW\(lr=)[0-9.eE+-]+(,.*)$", re.M)
SCH_RE = re.compile(r"^(\s*scheduler: warmup_cosine\(warmup_steps=)[0-9]+"
                    r"(, total_steps=)[0-9]+(, min_lr=)[0-9.eE+-]+(\).*)$", re.M)
CKPT_RE = re.compile(r'checkpoint_load="[^"]*", checkpoint_save="[^"]*"')

assert len(OPT_RE.findall(src)) == 1, "expected exactly one optimizer line"
assert len(SCH_RE.findall(src)) == 1, "expected exactly one scheduler line"
assert len(CKPT_RE.findall(src)) == 1, "expected exactly one checkpoint pair"

for name, a in ARMS.items():
    ck = f"{OUT}/ckpt_{name}/state.nslm"
    if name == "A":
        # THE CONTROL IS NOT REWRITTEN. Re-rendering 0.000015 as 1.5e-05 is
        # numerically identical but textually different, and the train_cfg
        # fingerprint is a STRING. A control whose schedule lines are
        # byte-identical to the production recipe cannot have drifted; one
        # that merely parses to the same numbers has to be argued about.
        # Assert the intended values instead of writing them.
        assert a['lr'] == BASE and a['warm'] == W \
               and a['total'] == T_PROD and a['minlr'] == MIN, \
               "arm A must be the production schedule verbatim"
        t = src
    else:
        t = OPT_RE.sub(lambda m: f"{m.group(1)}{fmt(a['lr'])}{m.group(2)}", src)
        t = SCH_RE.sub(lambda m: (f"{m.group(1)}{int(a['warm'])}{m.group(2)}"
                                  f"{int(a['total'])}{m.group(3)}{fmt(a['minlr'])}{m.group(4)}"), t)
    t = CKPT_RE.sub(f'checkpoint_load="{ck}", checkpoint_save="{ck}"', t)
    dst = ROOT / "models/coder1b" / f"sched3_arm_{name}.nsl"
    dst.write_text(t)

    # Assert the realised diff is ONLY the three intended lines.
    diff = [(i, x, y) for i, (x, y) in
            enumerate(zip(src.splitlines(), t.splitlines()), 1) if x != y]
    assert len(src.splitlines()) == len(t.splitlines()), f"arm {name}: line count changed"
    changed = {i for i, _, _ in diff}
    want = 1 if name == 'A' else 3
    assert len(changed) == want, f"arm {name}: {len(changed)} lines differ, expected {want}"
    for i, x, y in diff:
        assert ("optimizer: AdamW" in x) or ("scheduler: warmup_cosine" in x) \
               or ("checkpoint_load" in x), f"arm {name}: unexpected change on line {i}:\n{x}\n{y}"
    print(f"arm {name}: {len(changed)} line(s) changed  ({a['why']})")
    for s in (BRANCH_STEP, 72000, 80000, 88000, END_STEP):
        print(f"    lr({s}) = {warmup_cosine(a['lr'], s, a['warm'], a['total'], a['minlr']):.6e}")

# The branch LR must be IDENTICAL for A and B (that is what "same LR at the
# branch" means) and exactly half for C. Assert it rather than eyeball it.
la = warmup_cosine(ARMS['A']['lr'], BRANCH_STEP, ARMS['A']['warm'], ARMS['A']['total'], ARMS['A']['minlr'])
lb = warmup_cosine(ARMS['B']['lr'], BRANCH_STEP, ARMS['B']['warm'], ARMS['B']['total'], ARMS['B']['minlr'])
lc = warmup_cosine(ARMS['C']['lr'], BRANCH_STEP, ARMS['C']['warm'], ARMS['C']['total'], ARMS['C']['minlr'])
assert abs(lb - la) < 1e-15, f"B must start at the branch LR: {lb} vs {la}"
assert abs(lc - la / 2) < 1e-15, f"C must start at half the branch LR: {lc} vs {la/2}"
print(f"\nbranch-LR invariants OK: A=B={la:.9e}, C={lc:.9e} (= A/2)")
