"""Figures for the TidalMoE technical report.

All numbers are measurements from the tuning campaign (Nov 2026, single GB300); each block names its source.
Routing statistics come from data/routing-v1.npz, the adaptive-cache trace from report/data/adapt-zh-epochs.csv.
Run:  python3 report/figures/make_figures.py   (needs numpy + matplotlib) -> report/figures/*.pdf

Figures are drawn at the size they are printed (ICLR text width 5.5 in), so font sizes here are the printed sizes.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

FULL = 5.5            # ICLR \textwidth (in)
HALF = 2.68           # one of two side-by-side minipages (0.487\linewidth)
SKEW = 3.2            # routing-skew minipage (0.58\linewidth)

# ---- palette (validated: categorical slots 1-3 adjacent CVD dE >= 9.2; ordinal blue ramp 250..600) -------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#a9cbf4", "#86b6ef", "#5598e7", "#2a78d6", "#104281"]   # ordinal: later stage = darker
NEUTRAL = "#9b9a94"                                               # ablation baseline (feature off)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Consistent roles across figures: stock vLLM = orange, TidalMoE = blue, feature switched off = grey.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlepad": 4, "axes.labelpad": 2,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 1.8, "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def save(fig, name, width=FULL):
    """Save at exactly `width` inches after tight cropping, so the PDF is placed at 100% and fonts print as set."""
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(5):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
        if abs(w - width) < 0.005:
            break
        fig.set_size_inches(fig.get_figwidth() + (width - w), fig.get_figheight())
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.0f}", size=7, color=INK2):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=color)


# ---- 1. Throughput progression (A/B-verified stage results; README "Results") ---------------------------------
def fig_progression():
    # One panel per workload point, stages as rows: stage names are read once, no legend lookup.
    points = ["ShareGPT\n1 request", "ShareGPT\n8 requests", "ShareGPT\n32 requests", "Chinese\n32 requests"]
    stages = [
        ("Stock vLLM", [124, 282, 432, 419], ORANGE),
        ("Hot/cold placement", [216, 627, 931, 644], RAMP[0]),
        ("+ adaptive cache", [219, 735, 1170, 1150], RAMP[1]),
        ("+ layout, tactics, HBM", [259, 789, 1313, 1316], RAMP[2]),
        ("+ persistent kernels", [302, 935, 1599, 1617], RAMP[3]),
        ("+ adaptive verification", [293, 982, 1862, 1821], RAMP[4]),
    ]
    y = np.arange(len(stages))[::-1]
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
    for j, ax in enumerate(axs):
        vals = [s[1][j] for s in stages]
        ax.barh(y, vals, 0.72, color=[s[2] for s in stages], zorder=2)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v}", (v, yi), xytext=(2, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=7, color=INK2)
        ax.set_title(points[j], fontsize=7.5, linespacing=1.1)
        ax.set_xlim(0, 2350)
        ax.set_xticks([0, 1000, 2000])
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        speed = vals[-1] / vals[0]
        ax.annotate(f"{speed:.1f}×", (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8.5, fontweight="bold", color=RAMP[4])
    axs[0].set_yticks(y, [s[0] for s in stages])
    fig.supxlabel("output tokens / s", fontsize=8, y=-0.1)
    save(fig, "fig_progression.pdf")


# ---- 2. Throughput vs concurrency (stock and static-plan sweeps 2026-11-02/03; final = WF3 verification means) ---
def fig_concurrency():
    conc = [1, 2, 4, 8, 16, 32]
    series = [
        ("TidalMoE", [293.1, 433.0, 653.4, 981.9, 1333.9, 1861.5], BLUE, "o"),
        ("Static placement", [216.2, 330.5, 425.7, 627.2, 759.3, 930.8], AQUA, "s"),
        ("Stock vLLM", [124.0, 151.4, 203.4, 282.3, 363.5, 432.3], ORANGE, "D"),
    ]
    fig, ax = plt.subplots(figsize=(HALF, 2.15))
    fig.subplots_adjust(right=0.74)
    for name, v, col, mk in series:
        ax.plot(conc, v, color=col, marker=mk, zorder=3)
        # direct labels at the line ends instead of a legend
        ax.annotate(f"{name}\n{v[-1]:.0f}", (conc[-1], v[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color=col, fontweight="bold", linespacing=1.0)
    ax.set_xscale("log", base=2); ax.set_xticks(conc, [str(c) for c in conc]); ax.minorticks_off()
    ax.set_xlabel("concurrent requests"); ax.set_ylabel("output tokens / s")
    ax.set_ylim(0, 2000); ax.set_xlim(0.8, 34)
    save(fig, "fig_concurrency.pdf", HALF)


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
    names = {"sharegpt": ("English chat", BLUE), "zh": ("Chinese chat", ORANGE), "code": ("Code", AQUA)}
    E = 384; frac = np.arange(1, E + 1) / E
    k = int(0.2 * E)
    top20 = {wl: st[wl][1][k - 1] for wl in st}
    fig, ax = plt.subplots(figsize=(3.1, 2.25))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.9, ls="--", zorder=1)
    ax.annotate("uniform routing", (0.58, 0.635), rotation=45, fontsize=7, color=INK2, ha="center", va="bottom",
                rotation_mode="anchor", transform_rotates_text=True)
    for wl, (lab, col) in names.items():
        ax.plot(frac, st[wl][1], color=col, label=lab, zorder=3)
        ax.plot([0.2], [top20[wl]], "o", color=col, ms=4, zorder=4)
    ax.axvline(0.2, color=INK2, lw=0.8, ls=":", zorder=2)
    lo, hi = min(top20.values()), max(top20.values())
    ax.annotate(f"top 20% of experts get\n{lo:.0%}–{hi:.0%} of routing", (0.98, 0.04), xytext=(0, 0),
                textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("fraction of experts (sorted by use)")
    ax.set_ylabel("cumulative share of routing")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(axis="both")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.0,
              borderaxespad=0.1)
    save(fig, "fig_routing_skew.pdf", SKEW)
    # numbers quoted in the report
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
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 1.95), gridspec_kw={"wspace": 0.22})
    panels = ((axs[0], c2c_stock, c2c_tidal, "GPU reads of host memory (GB/s)", 430),
              (axs[1], hbm_stock, hbm_tidal, "HBM DRAM activity (%)", 50))
    for ax, a, b, title, top in panels:
        b1 = ax.bar(x - w / 2, a, w * 0.94, color=ORANGE, label="Stock vLLM (layer offload)", zorder=2)
        b2 = ax.bar(x + w / 2, b, w * 0.94, color=BLUE, label="Hot/cold placement + adaptive cache", zorder=2)
        bar_labels(ax, b1); bar_labels(ax, b2)
        ax.set_xticks(x, conc); ax.set_xlabel("concurrent requests"); ax.set_ylim(0, top)
        ax.set_title(title, loc="left")
    axs[0].axhline(390, color=INK2, lw=0.8, ls=":")
    axs[0].annotate("host-read ceiling ≈390 GB/s", (-0.45, 390), xytext=(0, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7, color=INK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.6)
    save(fig, "fig_bandwidth.pdf")


# ---- 5. Adaptive cache convergence (report/data/adapt-zh-epochs.csv) --------------------------------------------
def fig_adapt():
    rows = [r for r in csv.reader(l for l in open(os.path.join(REPO, "report", "data", "adapt-zh-epochs.csv"))
                                  if not l.startswith("#"))][1:]
    t = np.array([float(r[0]) for r in rows]) / 60.0; share = np.array([float(r[3]) for r in rows])
    swaps = np.array([float(r[4]) for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 1.8), gridspec_kw={"wspace": 0.22})
    axs[0].plot(t, share, color=BLUE, lw=1.5)
    axs[0].set_title("Selections served from host (%)", loc="left")
    axs[0].set_ylim(0, 30)
    axs[0].annotate(f"{share[0]:.1f}% at start", (t[0], share[0]), xytext=(1.25, 21), textcoords="data",
                    ha="left", va="center", fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=2))
    axs[0].annotate(f"{share[-1]:.1f}%", (t[-1], share[-1]), xytext=(0, 5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7, color=INK2)
    axs[1].plot(t, swaps, color=BLUE, lw=1.5)
    axs[1].set_title("Cumulative expert swaps", loc="left")
    axs[1].set_ylim(0, 6400)
    axs[1].annotate(f"{swaps[-1]:,.0f}", (t[-1], swaps[-1]), xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7, color=INK2)
    for ax in axs:
        ax.set_xlabel("minutes since traffic switch"); ax.set_xlim(0, 6.6)
    save(fig, "fig_adapt.pdf")


# ---- 6. Persistent kernel, one MoE layer standalone (kbench, production plan layout; ms per call) ---------------
def fig_persist():
    cold = [0, 4, 10, 18]
    data = {6: ([0.120, 0.279, 0.617, 1.113], [0.114, 0.201, 0.472, 0.837]),
            32: ([0.343, 0.396, 0.648, 1.129], [0.355, 0.358, 0.477, 0.844]),
            64: ([0.557, 0.545, 0.778, 1.149], [0.551, 0.528, 0.542, 0.843])}
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (T, (trt, per)) in zip(axs, data.items()):
        ax.plot(cold, trt, color=NEUTRAL, marker="D", label="TRT-LLM kernels (forced tactics)")
        ax.plot(cold, per, color=BLUE, marker="o", label="TidalMoE persistent kernels")
        ax.set_title(f"{T} tokens per call")
        ax.set_xticks(cold); ax.set_xlim(-1.5, 25)
        gain = 1 - per[-1] / trt[-1]
        ax.annotate("", xy=(19.3, per[-1]), xytext=(19.3, trt[-1]),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8, shrinkA=0, shrinkB=0))
        ax.annotate(f"−{gain:.0%}", (19.6, (per[-1] + trt[-1]) / 2), ha="left", va="center", fontsize=7,
                    color=INK2)
    fig.supxlabel("host-resident (cold) experts touched in the layer", fontsize=8, y=-0.08)
    axs[0].set_ylabel("MoE layer time (ms)"); axs[0].set_ylim(0, 1.25)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.8, columnspacing=1.6)
    save(fig, "fig_persist.pdf")


# ---- 7. Adaptive verification (WF3 verification, interleaved runs, means of 2) ----------------------------------
def fig_av():
    conc = ["1", "2", "4", "8", "16", "32"]; x = np.arange(6); w = 0.4
    noav = [284.7, 425.5, 624.2, 930.0, 1214.0, 1633.8]
    av = [293.1, 433.0, 653.4, 981.9, 1333.9, 1861.5]
    fig, ax = plt.subplots(figsize=(HALF, 2.15))
    ax.bar(x - w / 2, noav, w * 0.94, color=NEUTRAL, label="Static draft schedule", zorder=2)
    b = ax.bar(x + w / 2, av, w * 0.94, color=BLUE, label="Adaptive verification", zorder=2)
    for xi, (a0, a1), bb in zip(x, zip(noav, av), b):
        ax.annotate(f"{(a1 / a0 - 1) * 100:+.0f}%", (xi, a1), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, conc); ax.set_xlabel("concurrent requests"); ax.set_ylabel("output tokens / s")
    ax.set_ylim(0, 2000); ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.1)
    save(fig, "fig_av.pdf", HALF)


if __name__ == "__main__":
    fig_progression(); fig_concurrency(); fig_routing(); fig_bandwidth(); fig_adapt(); fig_persist(); fig_av()
