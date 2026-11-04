# Re-planning for your traffic
The adaptive cache converges online, so re-planning only gives a better starting point after restarts:
1. Start a routing-capture server:
   ```bash
   MOE_PLACEMENT=off MOE_PERSIST=0 MOE_AV=0 VLLM_USE_RUST_FRONTEND=0 ./serve.sh --enable-return-routed-experts
   ```
   Returning routed experts needs the Python frontend.
2. Collect routing counts with `tools/collect_routing.py`. Edit its prompt lists first; it needs numpy, so run it
   inside the container:
   ```bash
   docker cp tools/collect_routing.py tidal-moe:/tmp/ && docker cp datasets/ShareGPT_V3_unfiltered_cleaned_split.json tidal-moe:/tmp/sg.json
   docker exec tidal-moe python3 /tmp/collect_routing.py /tmp/sg.json /tmp/routing.npz 200 384 16
   docker cp tidal-moe:/tmp/routing.npz data/routing-new.npz
   ```
3. Build the plan in the vLLM image:
   ```bash
   IMG=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
   docker run --rm --entrypoint python3 -v "$PWD:/w" $IMG /w/tools/plan_convex.py /w/data/routing-new.npz \
     /w/plans/plan-b16.json 207 /w/plans/plan-new.json sharegpt:0.5,zh:0.3,code:0.2 mix
   docker run --rm --entrypoint python3 -v "$PWD:/w" $IMG /w/tools/relayout.py /w/plans/plan-new.json /w/plans/plan-new-g8.json 8
   ```
   The `207` is the host budget in 16-expert-block units: 54.2 GiB of cold experts, which fits the default KV sizing.
4. Start with `PLAN=plan-new-g8.json ./serve.sh`.

