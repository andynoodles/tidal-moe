"""Figures for the TidalMoE technical report.

All numbers are measurements from the tuning campaign (Nov 2026, single GB300); each block names its source.
Routing statistics come from data/routing-v1.npz, the adaptive-cache trace from report/data/adapt-zh-epochs.csv.
Run:  python3 report/figures/make_figures.py   (needs numpy + matplotlib) -> report/figures/*.pdf
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---- palette (validated: categorical slots 1-3 adjacent CVD dE >= 9.2; ordinal blue ramp 250..600) -------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal: later stage = darker
NEUTRAL = "#9b9a94"                                               # baseline (stock)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 2.0, "lines.markersize": 5, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.0f}", size=7):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=INK2)


# ---- 1. Throughput progression (A/B-verified stage results; README "Results") ---------------------------------
def fig_progression():
    points = ["ShareGPT\nconc 1", "ShareGPT\nconc 8", "ShareGPT\nconc 32", "Chinese\nconc 32"]
    stages = [
        ("Stock vLLM", [124, 282, 432, 419], NEUTRAL),
        ("Hot/cold placement", [216, 627, 931, 644], RAMP[0]),
        ("+ adaptive cache", [219, 735, 1170, 1150], RAMP[1]),
        ("+ layout, tactics, HBM reclaim", [259, 789, 1313, 1316], RAMP[2]),
        ("+ persistent MoE kernels", [302, 935, 1599, 1617], RAMP[3]),
        ("+ adaptive verification", [293, 982, 1862, 1821], RAMP[4]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    n = len(stages); w = 0.8 / n; x = np.arange(len(points))
    for i, (name, vals, col) in enumerate(stages):
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, w * 0.92, color=col, label=name, zorder=2)
        if i in (0, n - 1):
            bar_labels(ax, bars, size=6.5)
    ax.set_xticks(x, points)
    ax.set_ylabel("output tokens / s")
    ax.set_ylim(0, 2050)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.2, handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_progression.pdf")


# ---- 2. Throughput vs concurrency (stock and static-plan sweeps 2026-11-02/03; final = WF3 verification means) ---
def fig_concurrency():
    conc = [1, 2, 4, 8, 16, 32]
    series = [
        ("TidalMoE (final)", [293.1, 433.0, 653.4, 981.9, 1333.9, 1861.5], BLUE, "o"),
        ("Static hot/cold placement", [216.2, 330.5, 425.7, 627.2, 759.3, 930.8], AQUA, "s"),
        ("Stock vLLM", [124.0, 151.4, 203.4, 282.3, 363.5, 432.3], ORANGE, "D"),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for name, v, col, mk in series:
        ax.plot(conc, v, color=col, marker=mk, label=name, zorder=3)
        ax.annotate(f"{v[-1]:.0f}", (conc[-1], v[-1]), xytext=(4, -2), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log", base=2); ax.set_xticks(conc, [str(c) for c in conc])
    ax.set_xlabel("concurrent requests"); ax.set_ylabel("output tokens / s")
    ax.set_ylim(0, 2050); ax.set_xlim(0.85, 44)
    ax.legend(loc="upper left", fontsize=7.2)
    save(fig, "fig_concurrency.pdf")


# ---- 3. Routing skew (data/routing-v1.npz, decode-token routing counts per layer x expert) -----------------------
def routing_stats():
    d = np.load(os.path.join(REPO, "data", "routing-v1.npz"))
    out = {}
    for wl in ("sharegpt", "zh", "code"):
        c = d[f"{wl}_decode"].astype(float)                        # [40, 384]
        s = -np.sort(-c, axis=1)
        cum = np.cumsum(s, axis=1) / s.sum(axis=1, keepdims=True)  # per-layer cumulative share
        out[wl] = (c, cum.mean(axis=0))
    return d, out


def fig_routing():
    d, st = routing_stats()
    names = {"sharegpt": ("ShareGPT (English chat)", BLUE), "zh": ("ShareGPT-CN (Chinese)", ORANGE), "code": ("Code", AQUA)}
    E = 384; frac = np.arange(1, E + 1) / E
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.0, ls="--", zorder=1)
    for wl, (lab, col) in names.items():
        ax.plot(frac, st[wl][1], color=col, label=lab, zorder=3)
    ax.axvline(0.2, color=INK2, lw=0.8, ls=":")
    top20 = {wl: st[wl][1][int(0.2 * E) - 1] for wl in st}
    ax.annotate("top 20%", (0.2, 0.04), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK2)
    lo, hi = min(top20.values()), max(top20.values())
    ax.annotate(f"top-20% share:\n{lo:.0%}-{hi:.0%}", (0.2, 0.47), xytext=(3, 0), textcoords="offset points", ha="left",
                fontsize=7, color=INK2)
    ax.annotate("uniform routing", (0.78, 0.74), rotation=33, fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax.set_xlabel("fraction of experts (sorted by use, per layer)")
    ax.set_ylabel("share of decode routing")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.grid(axis="both")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.32), fontsize=7.2)
    save(fig, "fig_routing_skew.pdf")
    # numbers quoted in the report
    k = int(0.2 * E)
    jac = {}
    for a, b in (("sharegpt", "zh"), ("sharegpt", "code"), ("zh", "code")):
        ja = []
        for layer in range(40):
            ta = set(np.argsort(-st[a][0][layer])[:k]); tb = set(np.argsort(-st[b][0][layer])[:k])
            ja.append(len(ta & tb) / len(ta | tb))
        jac[(a, b)] = float(np.mean(ja))
    cold = {wl: 1 - st[wl][1][int(0.76 * E) - 1] for wl in st}
    pre = d["sharegpt_prefill"].sum(axis=1)
    print("top-20% share:", {k_: round(v, 3) for k_, v in top20.items()})
    print("coldest-24% share:", {k_: round(v, 3) for k_, v in cold.items()})
    print("top-20% Jaccard:", {f"{a}-{b}": round(v, 3) for (a, b), v in jac.items()})
    print("prefill selections enc(0-19) vs dec(20-39):", int(pre[:20].sum()), int(pre[20:].sum()))


# ---- 4. Bandwidth under load (bw-sequence 2026-11-03: Grace uncore PMU + GPM; docs/NSIGHT-BOTTLENECKS.md s7) ----
def fig_bandwidth():
    conc = ["1", "8", "16", "32"]; x = np.arange(4); w = 0.38
    c2c_stock = [165.5, 248.9, 258.0, 258.1]; c2c_tidal = [50.6, 128.7, 142.8, 170.3]
    hbm_stock = [14.3, 14.8, 14.9, 14.6]; hbm_tidal = [28.1, 40.2, 42.4, 37.6]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, a, b, ylab, top, fmt in ((axs[0], c2c_stock, c2c_tidal, "GPU reads of LPDDR5X over C2C (GB/s)", 300, "{:.0f}"),
                                     (axs[1], hbm_stock, hbm_tidal, "HBM DRAM activity (%)", 50, "{:.0f}")):
        b1 = ax.bar(x - w / 2, a, w * 0.94, color=ORANGE, label="Stock (layer offload)", zorder=2)
        b2 = ax.bar(x + w / 2, b, w * 0.94, color=BLUE, label="Hot/cold + adaptive cache", zorder=2)
        bar_labels(ax, b1, fmt); bar_labels(ax, b2, fmt)
        ax.set_xticks(x, conc); ax.set_xlabel("concurrent requests"); ax.set_ylabel(ylab); ax.set_ylim(0, top)
    axs[0].axhline(390, color=INK2, lw=0.8, ls=":")
    axs[0].set_ylim(0, 420)
    axs[0].annotate("measured host-read ceiling ~390 GB/s", (3.45, 390), xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=7, color=INK2)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=7.2)
    save(fig, "fig_bandwidth.pdf")


# ---- 5. Adaptive cache convergence (report/data/adapt-zh-epochs.csv) --------------------------------------------
def fig_adapt():
    rows = [r for r in csv.reader(l for l in open(os.path.join(REPO, "report", "data", "adapt-zh-epochs.csv"))
                                  if not l.startswith("#"))][1:]
    t = np.array([float(r[0]) for r in rows]); share = np.array([float(r[3]) for r in rows])
    swaps = np.array([float(r[4]) for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw={"wspace": 0.38})
    axs[0].plot(t, share, color=BLUE, lw=1.6)
    axs[0].set_xlabel("time since traffic switch (s)"); axs[0].set_ylabel("routed selections served\nfrom host memory (%)")
    axs[0].set_ylim(0, 30)
    axs[0].annotate(f"{share[0]:.1f}% at start", (t[0], share[0]), xytext=(40, 6), textcoords="offset points", fontsize=7,
                    color=INK2)
    axs[0].annotate(f"{share[-1]:.1f}%", (t[-1], share[-1]), xytext=(-4, 6), textcoords="offset points", ha="right",
                    fontsize=7, color=INK2)
    axs[1].plot(t, swaps, color=BLUE, lw=1.6)
    axs[1].set_xlabel("time since traffic switch (s)"); axs[1].set_ylabel("cumulative expert swaps")
    axs[1].annotate(f"{swaps[-1]:.0f}", (t[-1], swaps[-1]), xytext=(-8, 2), textcoords="offset points", ha="right",
                    fontsize=7, color=INK2)
    save(fig, "fig_adapt.pdf")


# ---- 6. Persistent kernel, one MoE layer standalone (kbench, production plan layout; ms per call) ---------------
def fig_persist():
    cold = [0, 4, 10, 18]
    data = {6: ([0.120, 0.279, 0.617, 1.113], [0.114, 0.201, 0.472, 0.837]),
            32: ([0.343, 0.396, 0.648, 1.129], [0.355, 0.358, 0.477, 0.844]),
            64: ([0.557, 0.545, 0.778, 1.149], [0.551, 0.528, 0.542, 0.843])}
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, (T, (trt, per)) in zip(axs, data.items()):
        ax.plot(cold, trt, color=ORANGE, marker="D", label="TRT-LLM + forced tactics")
        ax.plot(cold, per, color=BLUE, marker="o", label="TidalMoE persistent")
        ax.set_title(f"{T} tokens per call", color=INK)
        ax.set_xticks(cold); ax.set_xlim(-1.5, 20)
        gain = 1 - per[-1] / trt[-1]
        ax.annotate(f"-{gain:.0%}", (cold[-1], per[-1]), xytext=(0, -9), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=INK2)
    fig.supxlabel("host-resident (cold) experts touched in the layer", fontsize=8.5, y=-0.07)
    axs[0].set_ylabel("MoE layer time (ms)"); axs[0].set_ylim(0, 1.25)
    axs[0].legend(loc="upper left", fontsize=7)
    save(fig, "fig_persist.pdf")


# ---- 7. Adaptive verification (WF3 verification, interleaved runs, means of 2) ----------------------------------
def fig_av():
    conc = ["1", "2", "4", "8", "16", "32"]; x = np.arange(6); w = 0.38
    noav = [284.7, 425.5, 624.2, 930.0, 1214.0, 1633.8]
    av = [293.1, 433.0, 653.4, 981.9, 1333.9, 1861.5]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.bar(x - w / 2, noav, w * 0.94, color=AQUA, label="static draft schedule", zorder=2)
    b = ax.bar(x + w / 2, av, w * 0.94, color=BLUE, label="adaptive verification", zorder=2)
    for xi, (a0, a1), bb in zip(x, zip(noav, av), b):
        ax.annotate(f"{(a1 / a0 - 1) * 100:+.0f}%", (bb.get_x() + bb.get_width() / 2, a1), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, conc); ax.set_xlabel("concurrent requests (ShareGPT)"); ax.set_ylabel("output tokens / s")
    ax.set_ylim(0, 2100); ax.legend(loc="upper left", fontsize=7.2)
    save(fig, "fig_av.pdf")


if __name__ == "__main__":
    fig_progression(); fig_concurrency(); fig_routing(); fig_bandwidth(); fig_adapt(); fig_persist(); fig_av()
