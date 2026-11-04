# Collect per-token expert routing from a vLLM server started with --enable-return-routed-experts.
# Runs inside the vLLM container (needs numpy). Writes counts[layer][expert] for prefill and decode tokens,
# per workload, to an .npz file.
#   usage: python3 collect_routing.py <sharegpt.json> <out.npz> [n_sharegpt] [max_tokens] [concurrency]
import base64, io, json, os, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SHAREGPT, OUT = sys.argv[1], sys.argv[2]
N_SHAREGPT = int(sys.argv[3]) if len(sys.argv) > 3 else 160
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 384
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 16
L, E = 40, 384

ZH = [
    "請用繁體中文解釋什麼是通貨膨脹，以及央行升息為什麼可以抑制通膨。",
    "幫我寫一封給客戶的道歉信，因為我們的出貨延遲了兩週。",
    "台灣的健保制度有哪些優缺點？請分點說明。",
    "請比較 Python 和 Go 在後端開發上的差異，給新手工程師的建議是什麼？",
    "我想在台北規劃三天兩夜的旅遊行程，預算一萬元，請幫我安排。",
    "請解釋半導體製程中的 EUV 微影技術，以及台積電為什麼需要它。",
    "幫我把以下句子改得更正式：老闆，那個案子我們這禮拜應該做不完。",
    "什麼是大型語言模型的幻覺問題？企業導入時要怎麼降低風險？",
    "請寫一首關於颱風夜的現代詩。",
    "員工請育嬰留職停薪有哪些法律規定？請以台灣勞基法為準說明。",
    "請說明碳費與碳稅的差別，以及對中小企業的影響。",
    "幫我整理一份新進員工到職第一週的待辦清單。",
    "用簡單的方式向國小學生解釋光合作用。",
    "請分析電動車在台灣普及的障礙與可能的解決方法。",
    "一間咖啡店想提高回客率，請提出五個具體行銷策略。",
    "請解釋什麼是 ESG，以及上市櫃公司為什麼要發布永續報告書。",
    "我和同事對專案分工有衝突，該怎麼溝通比較好？",
    "請用表格比較定期定額投資 ETF 與個股投資的風險和報酬。",
    "幫我寫一段產品介紹文案，產品是一款可以監測睡眠品質的智慧手環。",
    "請說明中醫和西醫在治療感冒上的觀念差異。",
    "台灣少子化對教育體系會造成哪些長期影響？",
    "請解釋區塊鏈的基本原理，並舉一個非加密貨幣的應用例子。",
    "幫我擬定一份為期一個月的英文口說練習計畫。",
    "請說明如何在 Excel 中使用樞紐分析表整理銷售資料。",
    "從資安角度來看，企業員工最常犯的錯誤有哪些？",
    "請簡述清朝統治台灣時期的重要政策與影響。",
    "幫我寫一份會議記錄範本，包含議程、決議事項與待辦追蹤。",
    "請解釋機器學習中的過擬合是什麼，以及如何避免。",
    "長照 2.0 提供哪些服務？家屬要怎麼申請？",
    "請規劃一份給上班族的一週健康便當菜單。",
    "什麼是供應鏈韌性？疫情後製造業如何調整？",
    "請說明勞保年金與國民年金的差異。",
    "幫我寫一篇三百字的部落格文章，主題是在家工作的時間管理。",
    "請解釋量子電腦和傳統電腦的差別，用生活化的比喻。",
    "公司要導入生成式 AI 客服，請列出評估指標與導入步驟。",
    "請比較高鐵和台鐵在城際交通上的優劣。",
    "幫我把這段文字翻譯成英文：本公司致力於提供高品質的在地化服務。",
    "請說明心肺復甦術（CPR）的正確步驟。",
    "如何判斷一則網路新聞是不是假訊息？",
    "請用繁體中文摘要說明再生能源的種類與台灣目前的發展狀況。",
]
CODE = [
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Implement an LRU cache in Go with O(1) get and put.",
    "Explain the difference between a mutex and a semaphore with a C example.",
    "Write a SQL query that returns the top 3 products by revenue per month.",
    "Refactor this JavaScript callback-based code to async/await: fs.readFile(p, (e, d) => { if (e) throw e; console.log(d) })",
    "Write a Rust function that parses a CSV line respecting quoted fields.",
    "How do I debug a memory leak in a Node.js service? Give concrete steps.",
    "Write a bash script that finds the 10 largest files under a directory.",
    "Implement binary search in TypeScript with generics and tests.",
    "Explain how Python's GIL affects multithreaded CPU-bound code and alternatives.",
    "Write a Dockerfile for a FastAPI app with multi-stage build.",
    "Write a Kubernetes Deployment and Service YAML for an nginx app with 3 replicas.",
    "Implement Dijkstra's algorithm in C++ using a priority queue.",
    "Write a regex that validates IPv4 addresses and explain each part.",
    "What is the difference between git rebase and git merge? When to use each?",
    "Write a PyTorch training loop for a simple CNN on MNIST.",
    "Explain CUDA warps, shared memory, and bank conflicts with an example kernel.",
    "Write a Go HTTP middleware that logs request latency and status code.",
    "Implement a thread-safe bounded queue in Java.",
    "Write unit tests with pytest for a function that converts Roman numerals to integers.",
    "Explain database indexing (B-tree vs hash) and when an index hurts performance.",
    "Write a React component with a debounced search input that calls an API.",
    "Implement quicksort in Haskell and explain laziness implications.",
    "How would you design a rate limiter for a public API? Provide code for a token bucket.",
    "Write a Python asyncio crawler that fetches 100 URLs with a concurrency limit of 10.",
    "Explain how TLS handshake works step by step.",
    "Write a C function to reverse a singly linked list, iteratively and recursively.",
    "Write a GitHub Actions workflow that runs tests on Python 3.10-3.12.",
    "Implement a trie with insert, search and prefix search in Python.",
    "Explain the CAP theorem with real database examples.",
]

def sharegpt_prompts(n):
    data = json.load(open(SHAREGPT))
    random.Random(0).shuffle(data)
    out = []
    for d in data:
        conv = d.get("conversations") or []
        if conv and conv[0].get("from") == "human":
            t = conv[0]["value"].strip()
            if 20 < len(t) < 2000 and t.isascii():
                out.append(t)
        if len(out) >= n:
            break
    return out

def run_one(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "temperature": 0.6}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    ch = r["choices"][0]
    re_b64 = ch.get("routed_experts") or (ch.get("message") or {}).get("routed_experts") or r.get("routed_experts")
    if re_b64 is None:
        raise RuntimeError(f"no routed_experts in response keys={list(ch)}")
    arr = np.load(io.BytesIO(base64.b64decode(re_b64)))           # (num_tokens-1, L, k)
    n_prompt = r["usage"]["prompt_tokens"]
    return arr, n_prompt

results = {}
for name, prompts in (("zh", ZH), ("code", CODE), ("sharegpt", sharegpt_prompts(N_SHAREGPT))):
    pre = np.zeros((L, E), np.int64); dec = np.zeros((L, E), np.int64)
    t0 = time.time(); done = [0]; lock = threading.Lock()
    def work(p):
        arr, n_prompt = run_one(p)
        assert arr.shape[1] == L, arr.shape
        p_part, d_part = arr[:n_prompt], arr[n_prompt:]
        cp = np.zeros((L, E), np.int64); cd = np.zeros((L, E), np.int64)
        for l in range(L):
            cp[l] = np.bincount(p_part[:, l, :].ravel().astype(np.int64), minlength=E)[:E]
            cd[l] = np.bincount(d_part[:, l, :].ravel().astype(np.int64), minlength=E)[:E]
        with lock:
            pre[:] += cp; dec[:] += cd; done[0] += 1
    with ThreadPoolExecutor(CONC) as ex:
        list(ex.map(work, prompts))
    results[f"{name}_prefill"] = pre; results[f"{name}_decode"] = dec
    print(f"{name}: {len(prompts)} prompts, prefill tokens={pre[0].sum()//6}, decode tokens={dec[0].sum()//6}, {time.time()-t0:.0f}s", flush=True)
np.savez(OUT, **results)
print("saved", OUT)
