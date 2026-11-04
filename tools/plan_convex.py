# Convex-cost per-layer DRAM budgets (Nsight finding: a MoE layer costs ~max(h*N_hbm, c*N_cold) per step, so
# cold experts are free up to a crossover and cost ~64 us each beyond it; the linear plan ignores this).
#   usage: plan_convex.py <routing.npz> <base_plan.json> <n_cold_blocks> <out.json|-> [budget_src=sharegpt] [within_src=mix]
# Per-layer block counts are chosen greedily (smallest marginal expected cost, weighted over conc 1/8/32 step times);
# within-layer membership and slot layout follow analyze_routing.py v2 (coldest experts by within_src, interleaved blocks).
import json, sys
import numpy as np
from math import erf, sqrt, pi, exp
L, E, K, B = 40, 384, 6, 16
NB = E // B
npz, base, nblk, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
bsrc = sys.argv[5] if len(sys.argv) > 5 else "sharegpt"
wsrc = sys.argv[6] if len(sys.argv) > 6 else "mix"
d = np.load(npz)
names = sorted({k.rsplit("_", 1)[0] for k in d.files})
def norm(c): c = c.astype(float); return c / (c[0].sum() / K)
P = {n: norm(d[f"{n}_decode"]) for n in names}
P["mix"] = sum(P[n] for n in names) / len(names)
def src(s):
    return sum(float(w) * P[n] for n, w in (x.split(":") for x in s.split("+"))) if ":" in s else P[s]
pw = src(wsrc)
OBJ = [(n, float(w)) for n, w in (x.split(":") for x in bsrc.split(","))] if ":" in bsrc else [(bsrc, 1.0)]
pb = P[OBJ[0][0]]
# regimes: (name, T_eff, h us per HBM expert, c us per cold expert, step ms, weight)
# T_eff calibrated so the model reproduces measured DRAM experts/forward (README: c1 50, c8 221, c32 431).
REG = [("c1", 6, 5.6, 64.0, 13.2), ("c8", 32, 4.0, 64.0, 28.9), ("c32", 64, 3.45, 64.0, 48.5)]
def Phi(x): return 0.5 * (1 + erf(x / sqrt(2)))
def phi(x): return exp(-x * x / 2) / sqrt(2 * pi)
srt_b = np.argsort(pb, axis=1, kind="stable")
def layer_cost(l, nb, T, h, c, p=None):
    p = pb if p is None else p
    srt = np.argsort(p[l], kind="stable")
    q = 1 - (1 - np.clip(p[l], 0, 1)) ** T
    cold = srt[: nb * B]; hot = srt[nb * B:]
    mu = c * q[cold].sum() - h * q[hot].sum()
    var = c * c * (q[cold] * (1 - q[cold])).sum() + h * h * (q[hot] * (1 - q[hot])).sum()
    s = sqrt(max(var, 1e-12))
    return h * q[hot].sum() + mu * Phi(mu / s) + s * phi(mu / s), q[cold].sum()
def calib(target, T0, nc_blocks, p):
    lo, hi = 1.0, 2000.0
    for _ in range(50):
        T = sqrt(lo * hi)
        D = sum((1 - (1 - np.clip(np.sort(p[l])[: nc_blocks[l] * B], 0, 1)) ** T).sum() for l in range(L))
        lo, hi = (T, hi) if D < target else (lo, T)
    return T
bp = json.load(open(base))
nb0 = np.array([(E - bp["layers"][str(l)]["n_hot"]) // B for l in range(L)])
meas = {"c1": 50, "c8": 221, "c32": 431}
regs = []
for name, T, h, c, step in REG:
    Te = calib(meas[name], T, nb0, pb)
    regs.append((name, Te, h, c, step))
print("calibrated T_eff:", {r[0]: round(r[1], 1) for r in regs}, "budget src", bsrc, "within src", wsrc)
tab = {}
for name, Te, h, c, step in regs:
    tab[name] = sum(w * np.array([[layer_cost(l, k, Te, h, c, P[n]) for k in range(NB + 1)] for l in range(L)])
                    for n, w in OBJ) / sum(w for _, w in OBJ)  # [L, NB+1, 2]
def total(nb, name):
    t = tab[name]; return t[np.arange(L), nb, 0].sum() / 1e3, t[np.arange(L), nb, 1].sum()
def obj(nb):  # sum of relative step-time increase over the regimes
    return sum(total(nb, r[0])[0] / r[4] for r in regs)
# greedy: add blocks where the weighted marginal cost is smallest (costs are convex in nb per layer)
nb = np.zeros(L, int)
for _ in range(nblk):
    best, bl = None, None
    for l in range(L):
        if nb[l] >= 16: continue
        dlt = sum((tab[r[0]][l, nb[l] + 1, 0] - tab[r[0]][l, nb[l], 0]) / r[4] for r in regs)
        if best is None or dlt < best: best, bl = dlt, l
    nb[bl] += 1
# local search: move single blocks between layers while it helps
improved = True
while improved:
    improved = False
    cur = obj(nb)
    for a in range(L):
        for b_ in range(L):
            if a == b_ or nb[a] == 0 or nb[b_] >= 16: continue
            nb[a] -= 1; nb[b_] += 1
            v = obj(nb)
            if v < cur - 1e-9: cur = v; improved = True
            else: nb[a] += 1; nb[b_] -= 1
# linear (analyze_routing) allocation for the same block count, for reference
cand = sorted((pw[l, np.sort(np.argsort(pw[l], kind="stable")[k * B:(k + 1) * B])].sum(), l) for l in range(L) for k in range(NB))
nbl = np.zeros(L, int)
for _, l in cand[:nblk]: nbl[l] += 1
print(f"blocks: base {nb0.sum()}  new {nb.sum()}")
for lab, x in (("base plan", nb0), ("linear same budget", nbl), ("convex", nb)):
    s = "  ".join(f"{r[0]} MoE-host-excess {total(x, r[0])[0]:6.2f} ms D {total(x, r[0])[1]:5.0f}" for r in regs)
    print(f"{lab:20s} {s}")
print("per-layer blocks base  :", " ".join(f"{v:2d}" for v in nb0))
print("per-layer blocks convex:", " ".join(f"{v:2d}" for v in nb))
# evaluate on other distributions (robustness): costs with the same T_eff, p from each workload
for n in names + ["mix"]:
    row = []
    for name, Te, h, c, step in regs:
        f = lambda x: sum(layer_cost(l, x[l], Te, h, c, P[n])[0] for l in range(L)) / 1e3
        row.append(f"{name} {f(nb0):6.2f}->{f(nb):6.2f} ms")
    print(f"  eval on {n:9s}: " + "  ".join(row))
if out == "-": sys.exit()
layers2 = {}
for l in range(L):
    c = int(nb[l])
    srt = np.argsort(pw[l], kind="stable")
    cold_ids = [int(e) for e in srt[: c * B]]
    cs = set(cold_ids)
    hot_ids = [int(e) for e in np.argsort(-pw[l], kind="stable") if e not in cs]
    cold_blocks = sorted({int((i + 0.5) * NB / c) for i in range(c)}) if c else []
    order, segs, ci, hi = [], [], 0, 0
    for b in range(NB):
        is_cold = b in cold_blocks
        ids = cold_ids[ci:ci + B] if is_cold else hot_ids[hi:hi + B]
        ci += B if is_cold else 0; hi += 0 if is_cold else B
        order += ids
        kind = "host" if is_cold else "hbm"
        if segs and segs[-1][2] == kind: segs[-1][1] = (b + 1) * B
        else: segs.append([b * B, (b + 1) * B, kind])
    assert sorted(order) == list(range(E)), l
    layers2[str(l)] = {"order": order, "segments": segs, "n_hot": E - c * B}
json.dump({"num_experts": E, "budget_gib": nblk * B * 16.875 / 1024, "source": npz, "block": B,
           "objective": f"convex budget_src={bsrc} within={wsrc}", "layers": layers2}, open(out, "w"))
print("written", out)
