#!/usr/bin/env python3
"""relayout.py <in_plan.json> <out_plan.json> [G=8]
Same cold expert set per layer as the input plan, but cold slots arranged as groups of G consecutive slots
(G-aligned) spread evenly over the layer, so the MoE kernel (CTAs dispatched in slot order) meets host tiles
in small, evenly spaced bursts. Hot experts keep their relative order, cold experts keep theirs.
With G=8, w13 groups are 2 MiB-aligned (8 x 11.25 MiB = 45 pages); each group's w2 has one half page shared
with a hot neighbour, which _alloc_segments keeps in HBM (1 MiB of HBM per group)."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 8
p = json.load(open(src)); E = p["num_experts"]; NG = E // G
out = dict(p); out["layers"] = {}; out["block"] = G; out["source"] = f"{src} relayout G={G}"
tot_groups = 0
for k, lp in p["layers"].items():
    order = lp["order"]
    host = [False] * E
    for a, b, kind in (lp.get("segments") or [[0, lp["n_hot"], "hbm"], [lp["n_hot"], E, "host"]]):
        if kind == "host":
            for i in range(a, b): host[i] = True
    cold = [order[i] for i in range(E) if host[i]]; hot = [order[i] for i in range(E) if not host[i]]
    assert len(cold) % G == 0, (k, len(cold))
    n = len(cold) // G; tot_groups += n
    pos = sorted({int((i + 0.5) * NG / n) for i in range(n)}) if n else []
    assert len(pos) == n
    new, segs, ci, hi = [], [], 0, 0
    for g in range(NG):
        c = g in pos
        new += cold[ci:ci + G] if c else hot[hi:hi + G]
        ci += G if c else 0; hi += 0 if c else G
        kind = "host" if c else "hbm"
        if segs and segs[-1][2] == kind: segs[-1][1] = (g + 1) * G
        else: segs.append([g * G, (g + 1) * G, kind])
    assert sorted(new) == list(range(E))
    out["layers"][k] = {"order": new, "segments": segs, "n_hot": lp["n_hot"]}
json.dump(out, open(dst, "w"))
print(f"{dst}: {tot_groups} groups of {G} = {tot_groups * G} cold experts; layer 10 segments {out['layers']['10']['segments'] if '10' in out['layers'] else ''}")
