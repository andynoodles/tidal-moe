# Teacher-forced logprob comparison: same fixed texts, per-token logprob of the actual next token (prompt_logprobs).
#   record:  python3 tf_logprobs.py record  correctness/my-tf-ref.json
#   compare: python3 tf_logprobs.py compare correctness/tf_final.json [label]
import json, os, sys, urllib.request

URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1/completions"
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017")
TEXTS = [
    "The transformer architecture replaced recurrence with self-attention, letting every token attend to every other "
    "token in a sequence. Mixture-of-experts layers then scaled model capacity by routing each token to a small subset "
    "of feed-forward experts, so the compute per token stays roughly constant while the total parameter count grows. "
    "In practice, serving such models is often limited by memory bandwidth rather than arithmetic, because every "
    "decode step has to stream the weights of the selected experts from memory. Placing frequently used experts in "
    "the fastest memory tier is therefore an attractive optimization for single-GPU deployments.",
    "台灣的半導體產業從一九八〇年代開始發展，經過數十年的努力，已經成為全球供應鏈中不可或缺的一環。晶圓代工模式讓"
    "設計公司不必自建工廠，就能把晶片交給專業的製造廠生產。這種分工降低了進入門檻，也讓研發資源可以更集中。近年來，"
    "先進製程的競爭越來越激烈，除了電晶體微縮之外，先進封裝與異質整合也成為提升效能的重要方向。",
    "def merge_intervals(intervals):\n    intervals.sort(key=lambda x: x[0])\n    merged = []\n    for start, end in "
    "intervals:\n        if merged and start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n            merged.append([start, end])\n    return merged\n\n\nclass LRUCache:\n    def __init__(self, "
    "capacity):\n        self.capacity = capacity\n        self.data = {}\n\n    def get(self, key):\n        if key not "
    "in self.data:\n            return -1\n        value = self.data.pop(key)\n        self.data[key] = value\n        "
    "return value\n",
]


def tf(text):
    body = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    c = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    out = []
    for d in c.get("prompt_logprobs") or []:
        if not d:
            continue
        (_, v), = list(d.items())[:1]
        out.append(v["logprob"] if isinstance(v, dict) else v)
    return out


mode, path = sys.argv[1], sys.argv[2]
if mode == "record":
    json.dump([tf(t) for t in TEXTS], open(path, "w"))
    print("recorded", path)
else:
    ref = json.load(open(path))
    label = sys.argv[3] if len(sys.argv) > 3 else "candidate"
    allv = []
    for i, (t, r) in enumerate(zip(TEXTS, ref)):
        c = tf(t)
        d = [abs(a - b) for a, b in zip(r, c)]
        allv += d
        print(f"[{label}] text {i}: {len(d)} tokens  mean|dlogprob|={sum(d)/len(d):.4f}  max={max(d):.4f}  "
              f"tokens with |d|>0.1: {sum(x > 0.1 for x in d)}  identical: {sum(x == 0 for x in d)}")
    allv.sort()
    print(f"[{label}] all {len(allv)} tokens: mean={sum(allv)/len(allv):.4f} p50={allv[len(allv)//2]:.4f} "
          f"p99={allv[int(len(allv)*.99)]:.4f} max={allv[-1]:.4f}")
