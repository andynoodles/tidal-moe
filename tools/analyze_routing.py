# Analyze expert routing counts (from collect_routing.py) and build a hot/cold placement plan.
#   usage: python3 analyze_routing.py <routing.npz> <plan_out.json> [dram_budget_gib=61.17]
import json, sys
import numpy as np

src, plan_out = sys.argv[1], sys.argv[2]
BUDGET_GIB = float(sys.argv[3]) if len(sys.argv) > 3 else 61.17
L, E, K = 40, 384, 6
W13, W2 = 11.25 * 2**20, 5.625 * 2**20          # FP4 bytes per expert (w13, w2)
EXP = W13 + W2
d = np.load(src)
names = sorted({k.rsplit("_", 1)[0] for k in d.files})
dec = {n: d[f"{n}_decode"].astype(np.float64) for n in names}
pre = {n: d[f"{n}_prefill"].astype(np.float64) for n in names}


def share_top(c, frac):
    s = np.sort(c)[::-1]
    k = int(round(frac * len(s)))
    return s[:k].sum() / max(s.sum(), 1)


print("== 80/20 test: share of routed selections handled by the top X% of experts (per layer, decode tokens) ==")
print(f"{'workload':10} {'tokens':>8}  {'top10%':>14} {'top20%':>14} {'top50%':>14}   (mean [min-max] over 40 layers)")
for n in names + ["all"]:
    c = sum(dec.values()) if n == "all" else dec[n]
    row = []
    for f in (0.1, 0.2, 0.5):
        v = np.array([share_top(c[l], f) for l in range(L)])
        row.append(f"{v.mean()*100:4.1f} [{v.min()*100:3.0f}-{v.max()*100:3.0f}]")
    print(f"{n:10} {int(c[0].sum()//K):8d}  " + " ".join(f"{r:>14}" for r in row))
print("(uniform routing would give 10/20/50%)")
c = sum(dec.values())
print(f"\nexperts never selected in decode (all workloads): {(c == 0).sum()} of {L*E};"
      f"  per-layer max/mean load ratio: {np.mean(c.max(1) / c.mean(1)):.1f}x")

# hot-set stability between workloads (top 20% per layer)
print("\n== overlap of per-layer top-20% hot sets between workloads (Jaccard, mean over layers) ==")
k20 = int(0.2 * E)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        j = np.mean([len(set(np.argsort(-dec[a][l])[:k20]) & set(np.argsort(-dec[b][l])[:k20])) /
                     len(set(np.argsort(-dec[a][l])[:k20]) | set(np.argsort(-dec[b][l])[:k20])) for l in range(L)])
        print(f"  {a} vs {b}: {j:.2f}")


def norm(c):                     # per-token selection probability of each (layer, expert)
    return c / (c[0].sum() / K)


def plan_from(profile, budget_gib):
    """Cold set = the least-used (layer, expert) pairs globally until the DRAM byte budget is used."""
    n_cold = int(budget_gib * 2**30 // EXP)
    flat = np.argsort(profile.ravel(), kind="stable")[:n_cold]
    cold = np.zeros(L * E, bool); cold[flat] = True
    return cold.reshape(L, E)


def baseline_cold():
    """What vLLM's UVA offload does today: layers 0-8 fully, layer 9's w13 only (61.17 GiB)."""
    m = np.zeros((L, E)); m[:9] = 1.0; m[9] = W13 / EXP
    return m                     # fraction of each expert's bytes in DRAM


def dram_bytes_per_step(p, cold_frac, T):
    """Expected DRAM bytes read per decode step with T tokens/step: each distinct cold expert touched once."""
    touched = 1 - (1 - np.clip(p, 0, 1)) ** T
    return (touched * cold_frac * EXP).sum()


mix = sum(norm(dec[n]) for n in names) / len(names)   # equal weight per workload
cold = plan_from(mix, BUDGET_GIB)
print(f"\n== DRAM traffic, same {BUDGET_GIB:.2f} GiB of experts in DRAM ==")
print(f"hot/cold plan: {cold.sum()} cold experts ({cold.sum()/(L*E)*100:.1f}%), per layer min/max {cold.sum(1).min()}/{cold.sum(1).max()}")
print(f"{'eval on':10} {'layout':10} {'DRAM sel%':>9} " + " ".join(f"{'T='+str(T):>9}" for T in (1, 6, 16, 48, 64)) + "   (GiB read from DRAM per decode step, T tokens/step)")
for n in names + ["mix"]:
    p = mix if n == "mix" else norm(dec[n])
    for lay, cf in (("today", baseline_cold()), ("hot/cold", cold.astype(float))):
        sel = (p * cf).sum() / p.sum() * 100
        print(f"{n:10} {lay:10} {sel:8.1f}% " + " ".join(f"{dram_bytes_per_step(p, cf, T)/2**30:9.2f}" for T in (1, 6, 16, 48, 64)))

# leave-one-out: plan built without workload n, evaluated on n
print("\n== generalization: plan built from the other workloads, evaluated on the held-out one ==")
for n in names:
    others = [o for o in names if o != n]
    cl = plan_from(sum(norm(dec[o]) for o in others) / len(others), BUDGET_GIB)
    p = norm(dec[n])
    print(f"  held out {n:9}: DRAM sel {((p*cl).sum()/p.sum()*100):5.1f}%  (own-data plan {((p*plan_from(p, BUDGET_GIB)).sum()/p.sum()*100):5.1f}%, today {((p*baseline_cold()).sum()/p.sum()*100):5.1f}%)")

layers = {}
for l in range(L):
    hot = [int(e) for e in np.argsort(-mix[l], kind="stable") if not cold[l, e]]
    cl_ = [int(e) for e in np.argsort(-mix[l], kind="stable") if cold[l, e]]
    layers[str(l)] = {"order": hot + cl_, "n_hot": len(hot)}
json.dump({"num_experts": E, "budget_gib": BUDGET_GIB, "source": src, "layers": layers}, open(plan_out, "w"))
print(f"\nplan written: {plan_out}")


# ---------------------------------------------------------------------------------------------------------------
# Plan v2: cold experts in blocks of 16 slots (16 experts = 90 x 2MiB pages for w13, 45 for w2 -> no HBM waste),
# blocks spread evenly across each layer so the MoE kernel interleaves host-memory tiles with HBM tiles.
BLOCK = 16
NB = E // BLOCK
n_blocks = int(BUDGET_GIB * 2**30 // (EXP * BLOCK))
cand = []                                   # (cost, layer, k): k-th coldest block of a layer
srt = np.argsort(mix, axis=1, kind="stable")
for l in range(L):
    for k in range(NB):
        cand.append((mix[l, srt[l, k * BLOCK:(k + 1) * BLOCK]].sum(), l, k))
cand.sort()
nb_layer = np.zeros(L, int)
for cost, l, k in cand[:n_blocks]:
    nb_layer[l] += 1                        # costs increase within a layer, so chosen blocks are a coldest-prefix
layers2, cold2 = {}, np.zeros((L, E), bool)
for l in range(L):
    c = int(nb_layer[l])
    cold_ids = [int(e) for e in srt[l, :c * BLOCK]]
    hot_ids = [int(e) for e in np.argsort(-mix[l], kind="stable") if e not in set(cold_ids)]
    cold2[l, cold_ids] = True
    cold_blocks = sorted({int((i + 0.5) * NB / c) for i in range(c)}) if c else []
    order, segs, ci, hi = [], [], 0, 0
    for b in range(NB):
        is_cold = b in cold_blocks
        ids = cold_ids[ci:ci + BLOCK] if is_cold else hot_ids[hi:hi + BLOCK]
        ci += BLOCK if is_cold else 0
        hi += 0 if is_cold else BLOCK
        order += ids
        kind = "host" if is_cold else "hbm"
        if segs and segs[-1][2] == kind:
            segs[-1][1] = (b + 1) * BLOCK
        else:
            segs.append([b * BLOCK, (b + 1) * BLOCK, kind])
    assert sorted(order) == list(range(E)), l
    layers2[str(l)] = {"order": order, "segments": segs, "n_hot": E - c * BLOCK}
p = mix
print(f"\nplan v2 (blocks of {BLOCK}, interleaved): {cold2.sum()} cold experts, per-layer cold blocks min/max "
      f"{nb_layer.min()}/{nb_layer.max()}, DRAM sel {(p*cold2).sum()/p.sum()*100:.2f}% (v1 {(p*cold).sum()/p.sum()*100:.2f}%)")
out2 = plan_out.replace(".json", "-v2.json") if plan_out.endswith(".json") else plan_out + "-v2"
json.dump({"num_experts": E, "budget_gib": BUDGET_GIB, "source": src, "block": BLOCK, "layers": layers2}, open(out2, "w"))
print(f"plan v2 written: {out2}; example layer 0 segments: {layers2['0']['segments']}")
