# Greedy outputs of the correctness prompts sent concurrently (30 requests, so DSpark adaptive verification (MOE_AV=1) prunes
# drafts while many requests run) vs. the single-request reference (prefix match), then a losslessness check:
# every generated token is re-scored teacher-forced (one request at a time, prompt_logprobs=1) and must be the
# argmax up to numerical near-ties (gap = top-1 logprob - logprob of the generated token).
#   python3 conc_check.py <ref.json> <label> [copies]
import json, os, sys, urllib.request, concurrent.futures as cf
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1"
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017")
PROMPTS = [
    "Explain how a CPU cache works, including cache lines and associativity.",
    "請用繁體中文說明台灣的四季氣候特色，並給旅遊建議。",
    "Write a Python function that returns the n-th Fibonacci number using matrix exponentiation.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes? Show your steps.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "You are a tool-using agent. The user asks: 'What is the weather in Taipei?' Reply with the JSON tool call you would make.",
]


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def ask(p):
    r = post("/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 160,
                                   "temperature": 0, "logprobs": True, "top_logprobs": 0, "return_token_ids": True})
    c = r["choices"][0]
    toks = [t["token"] for t in (c.get("logprobs") or {}).get("content") or []]
    return toks, r.get("prompt_token_ids") or c.get("prompt_token_ids"), c.get("token_ids")


ref = json.load(open(sys.argv[1]))
label = sys.argv[2]
copies = int(sys.argv[3]) if len(sys.argv) > 3 else 5
jobs = [i for i in range(len(PROMPTS)) for _ in range(copies)]
with cf.ThreadPoolExecutor(len(jobs)) as ex:
    outs = list(ex.map(lambda i: ask(PROMPTS[i]), jobs))
ident = 0
pref = []
for i, (b, _, _) in zip(jobs, outs):
    a = [t for t, _ in ref[i]["tokens"]]
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    ident += n == len(a) == len(b)
    pref.append(n)
by = {i: sorted(p for j, p in zip(jobs, pref) if j == i) for i in range(len(PROMPTS))}
print(f"[{label}] concurrent x{len(jobs)}: identical to single-request ref {ident}/{len(jobs)}, mean common prefix "
      f"{sum(pref)/len(pref):.1f} tokens; per prompt: " + " ".join(f"p{i}:{v}" for i, v in by.items()))
txt = "".join(outs[jobs.index(3)][0])
print(f"[{label}] sample p3: {txt[-120:]!r}")

# teacher-forced argmax check
gaps = []
nonmax = 0
for toks, pids, oids in outs:
    if not pids or not oids:
        print(f"[{label}] no token ids returned; skipping argmax check")
        break
    r = post("/completions", {"model": MODEL, "prompt": list(pids) + list(oids), "max_tokens": 1, "temperature": 0,
                              "prompt_logprobs": 1})
    pl = r["choices"][0].get("prompt_logprobs") or []
    for pos, tid in zip(range(len(pids), len(pids) + len(oids)), oids):
        d = pl[pos] if pos < len(pl) else None
        if not d:
            continue
        lp = {int(k): (v["logprob"] if isinstance(v, dict) else v) for k, v in d.items()}
        if tid not in lp:
            continue
        g = max(lp.values()) - lp[tid]
        gaps.append(g)
        nonmax += g > 0
if gaps:
    gaps.sort()
    print(f"[{label}] teacher-forced argmax check over {len(gaps)} generated tokens: not argmax {nonmax} "
          f"({100*nonmax/len(gaps):.2f}%), gap>0.1: {sum(g > 0.1 for g in gaps)}, gap>0.5: {sum(g > 0.5 for g in gaps)}, "
          f"gap>1: {sum(g > 1 for g in gaps)}, max gap {gaps[-1]:.3f}")
