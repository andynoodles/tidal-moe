# usage: python3 bench/summarize.py <label> [<label> ...]  (results from bench/bench.sh, dir $RESULTS_DIR or ./results)
#  -> output tok/s, TPOT, TTFT per concurrency (+ ratio last/first)
import glob, json, os, sys

R = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
labs = sys.argv[1:]
rows = {}
for lab in labs:
    for f in glob.glob(f"{R}/{lab}/*.json"):
        d = json.load(open(f))
        rows.setdefault(d.get("max_concurrency"), {})[lab] = d

hdr = "out tok/s  TPOT  TTFT"
print("conc | " + " | ".join(f"{l[:24]:>24}" for l in labs) + ("  | last/first" if len(labs) > 1 else ""))
print("     | " + " | ".join(f"{hdr:>24}" for _ in labs))
for c in sorted(k for k in rows if k):
    cells = []
    for l in labs:
        d = rows[c].get(l)
        cells.append(f"{d['output_throughput']:9.1f} {d['mean_tpot_ms']:5.1f} {d['mean_ttft_ms']:6.0f}" if d else f"{'-':>24}")
    ratio = ""
    if len(labs) > 1 and labs[0] in rows[c] and labs[-1] in rows[c]:
        ratio = f"  | {rows[c][labs[-1]]['output_throughput'] / rows[c][labs[0]]['output_throughput']:.2f}x"
    print(f"{c:>4} | " + " | ".join(f"{x:>24}" for x in cells) + ratio)
