"""Greedy-output check against a running qwen36 server (one request at a time, so batching cannot change numerics).

  python qwen36/greedy_check.py save results/a3b-stock/greedy.json      # reference from one configuration
  python qwen36/greedy_check.py compare results/a3b-stock/greedy.json   # same prompts against another; diff report
"""
import json
import sys
import urllib.request

PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List five prime numbers greater than 100 and explain how you checked one of them.",
    "用两句话介绍一下长城。",
]


def ask(prompt, url="http://127.0.0.1:8000/v1/chat/completions"):
    body = {"model": "nvidia/Qwen3.6-35B-A3B-NVFP4", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96, "temperature": 0, "logprobs": True, "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    toks = [(t["token"], round(t["logprob"], 4)) for t in r["logprobs"]["content"]]
    return {"text": r["message"]["content"], "tokens": toks}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mode, path = sys.argv[1], sys.argv[2]
    out = [dict(prompt=p, **ask(p)) for p in PROMPTS]
    if mode == "save":
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"saved {len(out)} greedy outputs to {path}")
        return
    ref = json.load(open(path, encoding="utf-8"))
    for a, b in zip(ref, out):
        ta, tb = [t[0] for t in a["tokens"]], [t[0] for t in b["tokens"]]
        n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        same_lp = sum(1 for x, y in zip(a["tokens"][:n], b["tokens"][:n]) if x[1] == y[1])
        status = "IDENTICAL" if ta == tb else f"diverges at token {n}/{len(ta)}"
        print(f"{status:24s} logprobs equal on {same_lp}/{n} common tokens | {a['prompt'][:50]}")


if __name__ == "__main__":
    main()
