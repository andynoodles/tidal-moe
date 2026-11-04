# Greedy-output equivalence check between two server configs (run one request at a time).
#   record:  python3 correctness.py record  correctness/my-ref.json
#   compare: python3 correctness.py compare correctness/my-ref.json [label]
import json, os, sys, urllib.request

URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017")
PROMPTS = [
    "Explain how a CPU cache works, including cache lines and associativity.",
    "請用繁體中文說明台灣的四季氣候特色，並給旅遊建議。",
    "Write a Python function that returns the n-th Fibonacci number using matrix exponentiation.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes? Show your steps.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "You are a tool-using agent. The user asks: 'What is the weather in Taipei?' Reply with the JSON tool call you would make.",
]


def ask(p):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 160,
            "temperature": 0, "logprobs": True, "top_logprobs": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    c = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]
    m = c["message"]
    toks = [(t["token"], t["logprob"]) for t in (c.get("logprobs") or {}).get("content") or []]
    return {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "",
            "tokens": toks}


mode, path = sys.argv[1], sys.argv[2]
if mode == "record":
    json.dump([ask(p) for p in PROMPTS], open(path, "w"), ensure_ascii=False)
    print("recorded", path)
else:
    ref = json.load(open(path))
    label = sys.argv[3] if len(sys.argv) > 3 else "candidate"
    same_all = True
    for i, (p, r) in enumerate(zip(PROMPTS, ref)):
        c = ask(p)
        a, b = r["tokens"], c["tokens"]
        n = 0
        while n < min(len(a), len(b)) and a[n][0] == b[n][0]:
            n += 1
        dl = [abs(x[1] - y[1]) for x, y in zip(a[:n], b[:n]) if x[1] is not None and y[1] is not None]
        same = n == len(a) == len(b)
        same_all &= same
        print(f"[{label}] prompt {i}: {'IDENTICAL' if same else f'diverges at token {n}/{len(a)}'}  "
              f"max|dlogprob| over common prefix={max(dl) if dl else 0:.4f}")
        if not same:
            print(f"     ref: {''.join(t for t, _ in a[max(0, n-5):n+10])!r}")
            print(f"     new: {''.join(t for t, _ in b[max(0, n-5):n+10])!r}")
    print(f"[{label}] {'ALL IDENTICAL' if same_all else 'DIFFERENCES FOUND'}")
