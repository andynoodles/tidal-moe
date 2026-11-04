#!/usr/bin/env bash
# Downloads the two benchmark datasets into datasets/ (not committed):
#   ShareGPT (English):   anon8231489123/ShareGPT_Vicuna_unfiltered -> ShareGPT_V3_unfiltered_cleaned_split.json
#   ShareGPT-CN (zh):     FreedomIntelligence/ShareGPT-CN -> sharegpt_zh_bench.json, 4000 first-turn conversations whose
#                         prompt is >= 30% CJK and 10-1500 chars, shuffled with seed 1 (the exact "zh" set in the README)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$REPO/bench/common.sh"
mkdir -p "$DATASETS_DIR"; cd "$DATASETS_DIR"
command -v hf >/dev/null || { echo "needs the Hugging Face CLI: pip install -U huggingface_hub" >&2; exit 1; }
[ -f ShareGPT_V3_unfiltered_cleaned_split.json ] || \
  hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir .
[ -f sharegpt-cn/sharegpt_zh_38K.json ] || \
  hf download FreedomIntelligence/ShareGPT-CN sharegpt_zh_38K.json --repo-type dataset --local-dir sharegpt-cn
python3 - <<'PY'
import json, random, re
cjk = re.compile(r"[一-鿿]")
rows = [json.loads(l) for l in open("sharegpt-cn/sharegpt_zh_38K.json") if l.strip()]
keep = []
for r in rows:
    c = r.get("conversations") or []
    if len(c) >= 2 and c[0].get("from") == "human" and c[1].get("from") == "gpt":
        t = c[0]["value"]
        if 10 < len(t) < 1500 and len(cjk.findall(t)) >= 0.3 * len(t):
            keep.append({"id": r["id"], "conversations": c[:2]})
random.Random(1).shuffle(keep)
json.dump(keep[:4000], open("sharegpt_zh_bench.json", "w"), ensure_ascii=False)
print(f"zh conversations kept: {len(keep)} -> sharegpt_zh_bench.json (4000)")
PY
ls -la "$DATASETS_DIR"
