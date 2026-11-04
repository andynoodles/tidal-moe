#!/usr/bin/env bash
# Memory-bandwidth view of a running server under steady load, per concurrency:
#   * HBM: GPM "DRAM activity" counter (nvidia-smi dmon --gpm-metrics 10), plus SM / tensor activity
#   * Grace LPDDR5X: uncore PMUs via perf -- GPU reads of host memory over NVLink-C2C (nvidia_nvlink_c2c{0,1}_pmu_0
#     rd_bytes_loc) and the CPU's own DRAM traffic (nvidia_scf_pmu_0 cmem_*). Needs root or kernel.perf_event_paranoid<=0.
#   * plugin accounting of expert bytes read from DRAM / HBM, if the server runs with MOE_BW_TRACE=1
# GPU-side C2C/PCIe counters (DCGM/GPM) read 0 on GB300 with driver 595, hence the Grace PMUs.
#
#   bench/measure-bw.sh <label> <container> <conc...>       e.g. bench/measure-bw.sh tidal tidal-moe 1 8 32
#   DATASET=zh bench/measure-bw.sh ...
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$REPO/bench/common.sh"
LABEL=$1; CT=$2; shift 2
need_bench
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$RESULTS_DIR"; OUT=$RESULTS_DIR/$LABEL-bw.txt
DS=$(dataset_path "${DATASET:-sharegpt}")
for C in "$@"; do
  N=$(( C * 10 )); [ $N -lt 16 ] && N=16
  "$VLLM_BENCH" --backend openai-chat --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name sharegpt --dataset-path "$DS" --sharegpt-output-len 1024 --ignore-eos --temperature 0 \
    --seed 5 --num-prompts $N --max-concurrency $C --disable-tqdm > "$TMP/load.log" 2>&1 &
  BP=$!
  sleep 40                                   # let the load reach steady state
  T0=$(date -u +%s)
  E=nvidia_nvlink_c2c0_pmu_0/rd_bytes_loc/,nvidia_nvlink_c2c0_pmu_0/wr_bytes_loc/,nvidia_nvlink_c2c1_pmu_0/rd_bytes_loc/,nvidia_nvlink_c2c1_pmu_0/wr_bytes_loc/,nvidia_scf_pmu_0/cmem_rd_data/,nvidia_scf_pmu_0/cmem_wr_total_bytes/
  perf stat -a -x, -I 1000 -e $E -- sleep 30 2> "$TMP/perf.txt" &
  PP=$!
  nvidia-smi dmon --gpm-metrics 2,5,10 -d 1 -c 30 2>/dev/null > "$TMP/gpm.txt"
  wait $PP
  T1=$(date -u +%s)
  kill $BP 2>/dev/null; wait $BP 2>/dev/null
  docker logs --since "$T0" --until "$T1" "$CT" 2>&1 | grep "moe_placement\[bw\]" > "$TMP/bw.txt"
  python3 - "$LABEL" "$C" "$TMP" <<'PY' | tee -a "$OUT"
import re, statistics as st, sys, os
label, c, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
g = [l.split() for l in open(os.path.join(tmp, "gpm.txt")) if l.strip() and not l.startswith("#")]
num = lambda x: float(x) if re.match(r"^[0-9.]+$", x) else None
sm = [num(r[-3]) for r in g if num(r[-3]) is not None]
tn = [num(r[-2]) for r in g if num(r[-2]) is not None]
dr = [num(r[-1]) for r in g if num(r[-1]) is not None]
bw = []
for l in open(os.path.join(tmp, "bw.txt")):
    m = re.search(r"\(([0-9.]+)/s\).*?per forward: ([0-9.]+) DRAM \+ ([0-9.]+) HBM.*?DRAM ([0-9.]+) GB/s, HBM ([0-9.]+) GB/s", l)
    if m:
        bw.append([float(x) for x in m.groups()])
f = lambda v: st.mean(v) if v else float("nan")
pf = {}
for l in open(os.path.join(tmp, "perf.txt")):
    p = l.strip().split(",")
    if len(p) > 3 and re.match(r"^[0-9.]+$", p[0]) and re.match(r"^[0-9]+$", p[1]):
        pf.setdefault(p[3], []).append(float(p[1]))
pm = lambda k: (st.mean(pf[k]) if pf.get(k) else float("nan"))
c2c_rd = (pm("nvidia_nvlink_c2c0_pmu_0/rd_bytes_loc/") + pm("nvidia_nvlink_c2c1_pmu_0/rd_bytes_loc/")) / 1e9
c2c_wr = (pm("nvidia_nvlink_c2c0_pmu_0/wr_bytes_loc/") + pm("nvidia_nvlink_c2c1_pmu_0/wr_bytes_loc/")) / 1e9
cpu_rd = pm("nvidia_scf_pmu_0/cmem_rd_data/") * 32 / 1e9
cpu_wr = pm("nvidia_scf_pmu_0/cmem_wr_total_bytes/") / 1e9
cols = list(zip(*bw)) if bw else [[]] * 5
print(f"{label:16} c={c:>2} | GPM: SM {f(sm):5.1f}%  tensor {f(tn):4.1f}%  HBM DRAM-activity {f(dr):5.1f}% "
      f"(max {max(dr) if dr else float('nan'):.0f}%)")
print(f"{'':16}      | Grace PMU: GPU reads of LPDDR5X over C2C {c2c_rd:6.1f} GB/s, GPU writes {c2c_wr:5.1f} GB/s; "
      f"CPU's own LPDDR5X rd {cpu_rd:5.1f} wr {cpu_wr:5.1f} GB/s")
if bw:
    print(f"{'':16}      | plugin (MOE_BW_TRACE=1): {f(cols[0]):5.1f} steps/s, experts/forward DRAM {f(cols[1]):6.1f} "
          f"HBM {f(cols[2]):7.1f}, expert reads DRAM {f(cols[3]):6.1f} GB/s, HBM {f(cols[4]):7.1f} GB/s")
PY
done
