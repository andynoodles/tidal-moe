# Send one long prompt (~N tokens) and report success, prefill time; used to probe activation/KV peak memory.
import json, os, sys, time, urllib.request
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
para = ("In a quiet valley the river bends twice before it reaches the old mill, where the miller keeps careful "
        "records of every season, every flood and every harvest. ")
text = "".join(f"[{i}] " + para for i in range(n // 34 + 1))
body = {"model": os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017"), "prompt": text + "\nSummarize the records in one sentence:",
        "max_tokens": 16, "temperature": 0}
t = time.time()
req = urllib.request.Request(os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    print(f"long prompt OK: prompt_tokens={r['usage']['prompt_tokens']} in {time.time()-t:.1f}s -> {r['choices'][0]['text'][:60]!r}")
except Exception as e:
    print(f"long prompt FAILED after {time.time()-t:.1f}s: {e}")
