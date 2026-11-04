"""Hot/cold MoE expert placement for vLLM (DeepSeek-V4.1 on a single GB300).

Loaded as a vLLM general plugin (entry point group ``vllm.general_plugins``). It wraps the model's
``process_weights_after_loading`` so that, once the MoE kernel has re-laid-out its weights and before vLLM
profiles memory for the KV cache, every routed-expert layer is rewritten as:

  * experts permuted per layer so the hot ones come first (gate rows and routing biases permuted the same
    way, so routing decisions are unchanged), and
  * w13 / w2 rebuilt on one contiguous CUDA virtual-address range whose head (hot experts) is backed by HBM
    and whose tail (cold experts) is backed by pinned host DRAM (NUMA node 0), via cuMemCreate/cuMemMap.

The MoE kernel sees an ordinary CUDA tensor: hot experts are read at HBM speed, cold ones over NVLink-C2C,
like vLLM's UVA offload but per expert instead of per layer.

Env:
  MOE_PLACEMENT=off|inspect|apply   (default off)
  MOE_SPLIT=1                         run hot and cold experts as two concurrent TRT-LLM calls (needs a hot-prefix plan)
  MOE_ADAPT=1                         online adaptive expert cache (see _AdaptiveCache; knobs MOE_ADAPT_*)
  MOE_PLACEMENT_IDENTITY=1            control run: no permutation, only the per-layer HBM/DRAM split
  MOE_ADAPT_COUNT=sel|touch, MOE_ADAPT_MAX_TOKENS, MOE_ADAPT_FUSED=1   adaptive-cache counting (see _patch_adapt)
  MOE_HOST_PARAMS=<regex>             move matching gather-only params (token embedding) to host DRAM
  MOE_TACTICS="8=8:437,16=16:388,32=8:71,64=32:119"  force the TRT-LLM MoE tactic (tile_n:config) per token bucket (<=128)
  MOE_MHC_MEGA=1                      Mega-mHC for every token count (no TP4-tuned mHC overlap path at <=16 tokens)
  MOE_AV_CURVE=builtin|moe|online|file:<json>|scale:<a>, MOE_STEP_TRACE=1   DSpark adaptive-verification cost curve / step trace
  MOE_AV_MIN_REQS=<r>, MOE_AV_UNIFORM_CG=1   AV prunes drafts only at >= r requests; uniform decode graphs below r
  MOE_AV_MAX_K=<k>, MOE_AV_CAP_TOKENS=<n>, MOE_AV_DRAFT_MS=<ms|r:ms,..>, MOE_AV_VERIFY=<n:ms,..>, MOE_AV_CTL=<json file>
                                      AV per-batch caps / cost-curve overrides / runtime control (see _patch_av)
  MOE_PERSIST=1                       host-aware persistent MoE kernels for <= MOE_PERSIST_MAX_TOKENS tokens (persist_moe.py)
  MOE_PLACEMENT_PLAN=/path/plan.json  {"num_experts": 384, "layers": {"<idx>": {"order": [...], "n_hot": k}}}
                                      order[i] = logical expert stored in physical slot i; slots < n_hot -> HBM
"""
import gc
import json
import os
import time

_MODE = os.environ.get("MOE_PLACEMENT", "off").lower()


def _logger():
    """vLLM only wires handlers for the "vllm" logger tree; also mirror everything to /tuning/logs."""
    import logging
    from vllm.logger import init_logger
    log = init_logger("vllm.moe_placement")
    if not getattr(log, "_moe_file", False):
        try:
            os.makedirs("/tuning/logs", exist_ok=True)
            fh = logging.FileHandler(f"/tuning/logs/moe_placement-{os.getpid()}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(fh)
        except OSError:
            pass
        log._moe_file = True
    return log
_KEEP = []          # VMM handles + array-interface owners, alive for the process lifetime
_BIG = ("w13_weight", "w2_weight")
_SPLIT_STREAM = None
_ADAPT = None       # _AdaptiveCache when MOE_ADAPT=1


def register():
    if _MODE == "off":
        return
    import importlib
    log = _logger()
    patched = 0
    for modname in ("vllm.models.deepseek_v41.nvidia.vl_model", "vllm.models.deepseek_v41.nvidia.model"):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:                      # noqa: BLE001
            log.warning("moe_placement: cannot import %s: %s", modname, e)
            continue
        for name in dir(mod):
            cls = getattr(mod, name)
            # Only the outermost model: its hook runs after the MoE kernel's weight post-processing. The inner
            # language model's hook runs at the end of load_weights, before scales are converted / quant configs built.
            if (isinstance(cls, type) and name == "DeepseekV41ForCausalLM"
                    and "process_weights_after_loading" in cls.__dict__
                    and not getattr(cls, "_moe_placement_patched", False)):
                orig = cls.__dict__["process_weights_after_loading"]

                def wrapped(self, *a, _orig=orig, **kw):
                    out = _orig(self, *a, **kw)
                    if not getattr(self, "_moe_placement_done", False):
                        self._moe_placement_done = True
                        run(self)
                    return out
                cls.process_weights_after_loading = wrapped
                cls._moe_placement_patched = True
                patched += 1
                log.info("moe_placement: patched %s.%s (mode=%s)", modname, name, _MODE)
    if not patched:
        log.warning("moe_placement: nothing patched")
    if os.environ.get("MOE_SPLIT") == "1":
        _patch_split(log)
    if os.environ.get("MOE_PERSIST") == "1" and _MODE == "apply":
        from . import persist_moe          # wraps apply BEFORE the adaptive cache's counting wrapper
        persist_moe.patch_apply(log)
    if os.environ.get("MOE_ADAPT") == "1":
        _patch_adapt(log)
    if os.environ.get("MOE_TACTICS", "").strip():
        _patch_tactics(log)
    if os.environ.get("MOE_MHC_MEGA") == "1":
        _patch_mhc_mega(log)
    if (os.environ.get("MOE_AV_CURVE", "").strip() or os.environ.get("MOE_STEP_TRACE") == "1"
            or os.environ.get("MOE_AV_MIN_REQS", "").strip() or os.environ.get("MOE_AV_UNIFORM_CG") == "1"):
        _patch_av(log)


def _patch_tactics(log):
    """MOE_TACTICS="8=8:437,16=16:388,32=8:71,64=32:119": force the TRT-LLM-gen MoE tactic (tile_n, config) of the routed-expert
    op per num_tokens bucket (next power of two, <= 128 only). The autotuner's choice was profiled on a host/HBM mix
    unlike the served one; the override was picked offline with the hot/cold expert layout (kernel/kbench.py).
    The tactic is chosen on the host per call (fixed at CUDA-graph capture). A forced tactic that is not in the op's
    valid list for that shape is ignored. Not every tactic is bit-identical to the autotuned one: GEMM2 configs with a different split-K structure change
    the rounding of a few GEMM2 outputs (e.g. 8:468 vs the autotuned 8:272 at <=8 tokens). Only use tactics checked
    with kernel/exact2.py / kernel/exsweep.py (random fp32 biases, varied block scales, both finalize modes).
    Checked bit-identical on this model: 8=8:437,16=16:388,32=8:71,64=32:119 (5..64 tokens)."""
    table = {}
    for item in os.environ["MOE_TACTICS"].split(","):
        b, t = item.strip().split("=")
        tn, cfg = t.split(":")
        table[int(b)] = [int(tn), int(cfg)]
    from flashinfer.autotuner import AutoTuner
    if getattr(AutoTuner, "_moe_tactics_patched", False):
        return
    orig = AutoTuner.choose_one
    logged = set()

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kw):
        r, t = orig(self, custom_op, runners, tuning_config, inputs, **kw)
        if custom_op != "flashinfer::trtllm_fp4_block_scale_moe":
            return r, t
        try:
            n = int(inputs[4].shape[0])                  # MoeRunnerInputs.hidden_states
            if n > 128 or getattr(r, "num_local_experts", None) != 384 or getattr(r, "intermediate_size", None) != 2304:
                return r, t
            b = 1 << max(n - 1, 0).bit_length()
            want = table.get(b)
            if want is None:
                return r, t
            valid = r.get_valid_tactics(inputs, None)
            ok = any(list(v)[:2] == want for v in valid)
            key = (n, tuple(t) if isinstance(t, (list, tuple)) else t)
            if key not in logged:
                logged.add(key)
                _logger().info("moe_placement: MoE tactic for %d tokens (bucket %d): autotuned %s -> %s%s", n, b, t, want,
                               "" if ok else " (NOT valid for this shape: kept autotuned)")
            return (r, list(want)) if ok else (r, t)
        except Exception as e:                           # noqa: BLE001
            _logger().warning("moe_placement: tactic override failed (%s); keeping autotuned tactic", e)
            return r, t
    AutoTuner.choose_one = choose_one
    AutoTuner._moe_tactics_patched = True
    log.info("moe_placement: MoE tactic override per token bucket: %s", table)


def _patch_split(log):
    """Run each placed layer's routed experts as two TRT-LLM calls over contiguous expert ranges, like two EP ranks:
    hot slots [0, n_hot) on the current stream (HBM only) and cold slots [n_hot, E) on a side stream (host reads),
    concurrently, then add the two finalized outputs. Layers without ``_moe_split`` keep the stock path."""
    import torch
    from vllm.model_executor.layers.fused_moe.experts import trtllm_mxfp4_moe as tm
    cls = tm.TrtLlmMxfp4ExpertsModular
    if getattr(cls, "_moe_split_patched", False):
        return
    orig = cls.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        n_hot = getattr(self, "_moe_split", None)
        if n_hot is None or output is None or expert_map is not None:
            return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                        expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                        apply_router_weight_on_input)
        from flashinfer import trtllm_fp4_block_scale_routed_moe
        topk_ids = topk_ids.to(dtype=torch.int32)
        topk = topk_ids.size(-1)
        E = w1.size(0)
        x_scale = a1q_scale.view(torch.float8_e4m3fn) if a1q_scale is not None else None
        assert hidden_states.size(0) <= self._max_supported_tokens(topk, global_num_experts)

        def run(out, lo, hi):
            sl = slice(lo, hi)
            pick = lambda t: None if t is None else t[sl]          # noqa: E731
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=(topk_ids, topk_weights), routing_bias=None,
                hidden_states=hidden_states, hidden_states_scale=x_scale,
                gemm1_weights=w1[sl], gemm1_weights_scale=self.w1_scale[sl], gemm1_bias=pick(self.w1_bias),
                gemm1_alpha=pick(self.gemm1_alpha), gemm1_beta=pick(self.gemm1_beta),
                gemm1_clamp_limit=pick(self.gemm1_clamp_limit),
                gemm2_weights=w2[sl], gemm2_weights_scale=self.w2_scale[sl], gemm2_bias=pick(self.w2_bias),
                output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
                num_experts=global_num_experts, top_k=topk, n_group=None, topk_group=None,
                intermediate_size=self.intermediate_size_per_partition,
                local_expert_offset=lo, local_num_experts=hi - lo, routed_scaling_factor=None,
                routing_method_type=tm.RoutingMethodType.Renormalize, do_finalize=True, enable_pdl=True,
                activation_type=self._flashinfer_activation_type(activation), output=out,
                tune_max_num_tokens=tm.fi_moe_largest_bucket(self.moe_config))

        cur = torch.cuda.current_stream()
        aux = self._moe_split_stream
        tmp = torch.empty_like(output)
        aux.wait_stream(cur)
        with torch.cuda.stream(aux):
            run(tmp, n_hot, E)                                   # cold experts: reads host DRAM over C2C
        run(output, 0, n_hot)                                    # hot experts: HBM only
        cur.wait_stream(aux)
        tmp.record_stream(aux)
        output.add_(tmp)
        if not getattr(cls, "_moe_split_logged", False):
            cls._moe_split_logged = True
            _logger().info("moe_placement: split MoE active (first call: %d tokens, hot %d / cold %d experts)",
                           hidden_states.size(0), n_hot, E - n_hot)
        return None

    cls.apply = apply
    cls._moe_split_patched = True
    log.info("moe_placement: patched TrtLlmMxfp4ExpertsModular.apply for hot/cold split")


# ----------------------------------------------------------------------------------------------- discovery

def _moe_layers(model, num_experts):
    """(layer_idx, name, moe_module) for the main model's routed-expert MoE layers."""
    from vllm.model_executor.models.utils import extract_layer_index
    out = []
    for name, m in model.named_modules():
        if not (hasattr(m, "gate") and hasattr(m, "experts") and hasattr(m, "n_routed_experts")):
            continue
        if m.n_routed_experts != num_experts:
            continue                                   # e.g. the DSpark draft MoE (128 experts)
        try:
            idx = extract_layer_index(name)
        except Exception:                              # noqa: BLE001
            continue
        out.append((idx, name, m))
    out.sort(key=lambda t: t[0])
    return out


def _walk_tensors(root, max_depth=6):
    """Yield (owner, key, tensor, path) for tensors reachable from ``root`` without leaving it:
    nn.Modules are followed only if they are submodules of root; vllm.config objects are skipped."""
    import torch
    inside = {id(x) for x in root.modules()}
    seen = set()
    out = []

    def visit(obj, path, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            if id(obj) not in inside:
                return
            items = (list(obj._parameters.items()) + list(obj._buffers.items())
                     + [(k, v) for k, v in vars(obj).items() if not k.startswith("_")]
                     + list(obj._modules.items()))
        elif isinstance(obj, dict):
            items = list(obj.items())
        elif isinstance(obj, (list, tuple)):
            items = list(enumerate(obj))
        elif (hasattr(obj, "__dict__") and not isinstance(obj, type)
              and type(obj).__module__.startswith(("vllm", "flashinfer"))
              and not type(obj).__module__.startswith("vllm.config")):
            items = list(vars(obj).items())
        else:
            return
        for k, v in items:
            if isinstance(v, torch.Tensor):
                out.append((obj, k, v, f"{path}.{k}"))
            elif v is not None and not isinstance(v, (int, float, str, bool, bytes, torch.dtype, torch.device)):
                visit(v, f"{path}.{k}", depth + 1)

    visit(root, "moe", 0)
    visit = None                    # break the closure self-reference cycle (it would pin `out`'s tensors)
    return out


def _expert_groups(m, E):
    """Group per-expert tensors (dim0 == E, on CUDA) of MoE module ``m`` by storage."""
    groups = {}
    for owner, key, t, path in _walk_tensors(m):
        if t.dim() >= 1 and t.shape[0] == E and t.is_cuda:
            groups.setdefault(t.untyped_storage().data_ptr(), []).append((owner, key, t, path))
    return groups


def _is_host(t):
    from cuda.bindings import driver as cu
    err, mt = cu.cuPointerGetAttribute(cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, t.data_ptr())
    if err != cu.CUresult.CUDA_SUCCESS:
        return None
    return int(mt) == int(cu.CUmemorytype.CU_MEMORYTYPE_HOST)


# ----------------------------------------------------------------------------------------------- VMM

def _ck(ret):
    from cuda.bindings import driver as cu
    if isinstance(ret, tuple):
        err, *vals = ret
    else:
        err, vals = ret, []
    if err != cu.CUresult.CUDA_SUCCESS:
        _, msg = cu.cuGetErrorString(err)
        raise RuntimeError(f"CUDA driver error {err}: {msg}")
    return vals[0] if len(vals) == 1 else vals


class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None}


def _alloc_split(nbytes, hbm_bytes, dev, host_numa=0):
    """uint8 CUDA tensor of ``nbytes`` on one VA range: [0, round_up(hbm_bytes)) in HBM, the rest in host DRAM."""
    return _alloc_segments(nbytes, [(0, hbm_bytes, "hbm"), (hbm_bytes, nbytes, "host")], dev, host_numa)


def _alloc_segments(nbytes, segments, dev, host_numa=0):
    """uint8 CUDA tensor of ``nbytes`` on one VA range whose byte ``segments`` [(start, end, "hbm"|"host"), ...] are
    backed by HBM or pinned host DRAM. A page goes to host only if it lies entirely inside a host segment."""
    import torch
    from cuda.bindings import driver as cu

    def prop(loc_type, loc_id):
        p = cu.CUmemAllocationProp()
        p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        p.location.type = loc_type
        p.location.id = loc_id
        return p
    dprop = prop(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, dev)
    hprop = prop(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, host_numa)
    rec = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    g = max(int(_ck(cu.cuMemGetAllocationGranularity(dprop, rec))), int(_ck(cu.cuMemGetAllocationGranularity(hprop, rec))))
    size = -(-nbytes // g) * g
    n_pages = size // g
    on_host = [False] * n_pages
    for s, e, kind in segments:
        if kind != "host":
            continue
        for pg in range(-(-s // g), e // g):          # pages fully inside [s, e)
            on_host[pg] = True
    runs = []                                         # coalesced (offset, length, prop)
    for pg in range(n_pages):
        p = hprop if on_host[pg] else dprop
        if runs and runs[-1][2] is p:
            runs[-1][1] += g
        else:
            runs.append([pg * g, g, p])
    hb = sum(r[1] for r in runs if r[2] is dprop)
    base = int(_ck(cu.cuMemAddressReserve(size, g, 0, 0)))
    handles = []
    for off, length, p in runs:
        if length:
            try:
                h = _ck(cu.cuMemCreate(length, p, 0))
            except RuntimeError as e:
                where = "HBM" if p is dprop else "host DRAM"
                raise RuntimeError(f"cuMemCreate {length / 2**30:.2f} GiB in {where} failed "
                                   f"(GPU free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB): {e}") from None
            _ck(cu.cuMemMap(cu.CUdeviceptr(base + off), length, 0, h, 0))
            handles.append(h)
    acc = cu.CUmemAccessDesc()
    acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = dev
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _ck(cu.cuMemSetAccess(cu.CUdeviceptr(base), size, [acc], 1))
    cai = _CAI(base, nbytes)
    t = torch.as_tensor(cai, device=f"cuda:{dev}")
    assert t.data_ptr() == base
    _KEEP.append((base, size, handles, cai))
    return t, hb, size - hb


# ----------------------------------------------------------------------------------------------- main

def _inspect(layers, E, log):
    for idx, name, m in layers[:1] + layers[9:11] + layers[-1:]:
        for owner, key, t, path in _walk_tensors(m):
            if E in tuple(t.shape) or path.startswith("moe.gate"):
                log.info("moe_placement[inspect] L%d %s shape=%s dtype=%s contig=%s host=%s %.1fMiB ptr=%x",
                         idx, path, tuple(t.shape), t.dtype, t.is_contiguous(), _is_host(t),
                         t.numel() * t.element_size() / 2**20, t.untyped_storage().data_ptr())


def _place_layer(idx, m, lp, E, dev, log):
    import torch
    order = torch.tensor(lp["order"], dtype=torch.long, device=f"cuda:{dev}")
    n_hot = int(lp["n_hot"])
    assert sorted(lp["order"]) == list(range(E))
    hbm = host = 0
    names = []
    groups = _expert_groups(m, E)
    # small tensors first, then the big weights one at a time, dropping each old copy right after its replacement
    keys = sorted(groups, key=lambda k: any(r[1] in _BIG for r in groups[k]))
    for k in keys:
        refs = groups.pop(k)
        base = max(refs, key=lambda r: r[2].numel())[2]
        is_big = any(r[1] in _BIG for r in refs)
        names.append(refs[0][3].split(".")[-1] + ("*" if is_big else ""))
        if not is_big:                                 # scales, biases, gate rows, routing biases: in place
            base.copy_(base.index_select(0, order))
            continue
        assert base.is_contiguous() and base.storage_offset() == 0, (refs[0][3], base.stride())
        row = base[0].numel() * base.element_size()
        segs = lp.get("segments") or [[0, n_hot, "hbm"], [n_hot, E, "host"]]
        flat, hb, hs = _alloc_segments(row * E, [(s * row, e * row, k) for s, e, k in segs], dev)
        new = flat.view(base.dtype).view(base.shape)
        for i0 in range(0, E, 32):                     # permuted copy; source may be host (UVA) resident
            sl = order[i0:i0 + 32]
            new[i0:i0 + sl.numel()].copy_(base.index_select(0, sl))
        for owner, key, t, _p in refs:
            view = new if (t.shape == base.shape and t.stride() == base.stride() and t.storage_offset() == 0) \
                else torch.as_strided(new, t.shape, t.stride(), t.storage_offset())
            if isinstance(owner, torch.nn.Module) and key in owner._parameters:
                owner._parameters[key].data = view
                owner._parameters[key]._vllm_is_uva_offloaded = False
            elif isinstance(owner, (dict, list)):
                owner[key] = view
            else:
                setattr(owner, key, view)
        hbm += hb
        host += hs
        del refs, base, t, owner, new, flat, view
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    m._moe_placement_order = order                     # physical slot -> logical expert id
    if os.environ.get("MOE_SPLIT") == "1":
        segs = lp.get("segments") or [[0, n_hot, "hbm"], [n_hot, E, "host"]]
        assert [list(x) for x in segs] == [[0, n_hot, "hbm"], [n_hot, E, "host"]], "split needs a hot prefix"
        fe = m.experts.routed_experts.quant_method.moe_kernel.impl.fused_experts
        global _SPLIT_STREAM
        if _SPLIT_STREAM is None:
            _SPLIT_STREAM = torch.cuda.Stream(device=dev)
        fe._moe_split = n_hot
        fe._moe_split_stream = _SPLIT_STREAM
        for cfg in {id(c): c for c in (getattr(fe, "moe_config", None), getattr(m.experts, "moe_config", None))
                    if c is not None}.values():
            cfg.limit_deferred_moe_finalize(0)     # both halves finalize; no top-k reduction fused downstream
        assert not m.experts.moe_config.should_defer_moe_finalize(64)
    return hbm, host, names


def run(model):
    import torch
    log = _logger()
    t0 = time.time()
    E = 384
    layers = _moe_layers(model, E)
    log.info("moe_placement: %d MoE layers with %d experts", len(layers), E)
    if _MODE == "inspect":
        _inspect(layers, E, log)
        return
    plan = json.load(open(os.environ["MOE_PLACEMENT_PLAN"]))
    assert plan["num_experts"] == E
    res = os.environ.get("MOE_ADAPT_RESUME", "")
    if res and os.path.exists(res):
        # warm start: slot order saved by the adaptive cache (MOE_ADAPT_SAVE); only if the HBM/DRAM layout matches
        try:
            st = json.load(open(res))
            seg = lambda lp: [list(x) for x in (lp.get("segments") or [[0, lp["n_hot"], "hbm"], [lp["n_hot"], E, "host"]])]
            ok = (st.get("num_experts") == E and set(st["layers"]) == set(plan["layers"])
                  and all(seg(st["layers"][k]) == seg(plan["layers"][k]) and sorted(st["layers"][k]["order"]) == list(range(E))
                          for k in plan["layers"]))
            if ok:
                log.info("moe_placement: warm start from %s (saved epoch %s, %s)", res, st.get("epoch"), st.get("saved_at"))
                plan = st
            else:
                log.warning("moe_placement: %s does not match the plan's HBM/DRAM layout; ignoring it", res)
        except Exception as e:                          # noqa: BLE001
            log.warning("moe_placement: cannot use %s: %s", res, e)
    dev = torch.cuda.current_device()

    hbm_q, host_q = [], []                             # layers whose big weights are in HBM / on host now
    for idx, name, m in layers:
        lp = plan["layers"].get(str(idx))
        if lp is None:
            continue
        if os.environ.get("MOE_PLACEMENT_IDENTITY") == "1":
            # control: keep expert order, put the same number of (highest-id) experts in DRAM
            lp = {"order": list(range(E)), "n_hot": lp["n_hot"]}
        big = [r[2] for refs in _expert_groups(m, E).values() for r in refs if r[1] in _BIG]
        (host_q if any(_is_host(t) for t in big) else hbm_q).append((idx, m, lp))
    tot_hbm = tot_host = 0
    placed = []
    while hbm_q or host_q:
        # Take an on-host layer whenever free HBM can hold its hot part with margin; otherwise shrink an HBM layer.
        pick = None
        if host_q:
            idx, m, lp = host_q[0]
            need = sum(r[2][0].numel() * r[2].element_size() for refs in _expert_groups(m, E).values()
                       for r in refs[:1] if any(x[1] in _BIG for x in refs)) * int(lp["n_hot"])
            free, _ = torch.cuda.mem_get_info()
            if free - need > (6 << 30) or not hbm_q:     # keep room for an HBM layer's transient w13 copy
                pick = host_q.pop(0)
        if pick is None:
            pick = hbm_q.pop(0)
        idx, m, lp = pick
        placed.append(pick)
        free0 = torch.cuda.mem_get_info()[0]
        hb, hs, names = _place_layer(idx, m, lp, E, dev, log)
        tot_hbm += hb
        tot_host += hs
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
        free1 = torch.cuda.mem_get_info()[0]
        log.info("moe_placement: L%d n_hot=%d -> HBM %.2f GiB, DRAM %.2f GiB; GPU free %.2f -> %.2f GiB%s", idx,
                 lp["n_hot"], hb / 2**30, hs / 2**30, free0 / 2**30, free1 / 2**30,
                 f"; permuted {names}" if idx in (0, 39) else "")
    free, total = torch.cuda.mem_get_info()
    log.info("moe_placement: placed in %.1fs -> expert weights HBM %.2f GiB, host DRAM %.2f GiB; GPU free %.1f/%.1f GiB",
             time.time() - t0, tot_hbm / 2**30, tot_host / 2**30, free / 2**30, total / 2**30)
    _host_rebind(model, dev, log)
    if os.environ.get("MOE_PERSIST") == "1":
        from . import persist_moe
        persist_moe.bind(placed, E, dev, log)
    if os.environ.get("MOE_ADAPT") == "1":
        global _ADAPT
        _ADAPT = _AdaptiveCache(placed, E, dev, log)


def _host_rebind(model, dev, log):
    """MOE_HOST_PARAMS=<regex>: move matching large parameters that are only gathered (e.g. the token embedding:
    ~64 rows = 0.6 MB read per decode step) to pinned host DRAM, freeing HBM for hot experts. Bit-exact copy on the
    same kind of VMM host pages as cold experts; runs before memory profiling / graph capture so addresses are fixed.
    Never touches the LM head (read in full every step)."""
    import re
    import torch
    pat = os.environ.get("MOE_HOST_PARAMS", "")
    if not pat:
        return
    rx = re.compile(pat)
    seen, moved = set(), 0
    big = sorted(((p.numel() * p.element_size(), n, tuple(p.shape), str(p.dtype)) for n, p in model.named_parameters()
                  if p.is_cuda and ".experts." not in n and p.numel() * p.element_size() >= (256 << 20)), reverse=True)
    log.info("moe_placement: large non-expert CUDA params: %s",
             "; ".join(f"{n} {s} {d} {b / 2**30:.2f}GiB" for b, n, s, d in big[:8]))
    for name, p in model.named_parameters(remove_duplicate=False):
        if not rx.search(name) or "head" in name or not p.is_cuda or id(p) in seen:
            continue
        seen.add(id(p))
        nbytes = p.numel() * p.element_size()
        if nbytes < (64 << 20) or _is_host(p) or not p.is_contiguous():
            continue
        free0 = torch.cuda.mem_get_info()[0]
        flat, _hb, _hs = _alloc_segments(nbytes, [(0, nbytes, "host")], dev)
        new = flat.view(p.dtype).view(p.shape)
        new.copy_(p.data)
        torch.cuda.synchronize()
        assert torch.equal(new, p.data), name
        p.data = new
        moved += nbytes
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free1 = torch.cuda.mem_get_info()[0]
        log.info("moe_placement: %s %s -> host DRAM (%.2f GiB); GPU free %.2f -> %.2f GiB", name, tuple(p.shape),
                 nbytes / 2**30, free0 / 2**30, free1 / 2**30)
    if not moved:
        log.warning("moe_placement: MOE_HOST_PARAMS=%r matched nothing", pat)



# ----------------------------------------------------------------------------------------------- adaptive cache
#
# Online re-placement of experts between the HBM and DRAM slots of each layer, after the ideas in DALI (workload
# window scores), TriMoE (EMA load prediction, benefit-ranked migration within a budget), HybriMoE (decayed scores)
# and migration-cost gating: a swap is only done when the DRAM expert's smoothed per-step selection rate beats the
# HBM expert's by a margin, at most MOE_ADAPT_SWAPS swaps per epoch, most beneficial first.
#
# * Counting: every placed layer's TRT-LLM MoE call adds its routed slot ids into a [L, E] GPU counter
#   (index_add_, captured into the CUDA graphs like the rest of the layer).
# * Epoch (every MOE_ADAPT_EPOCH_S seconds, on the engine thread between steps): read + zero the counters,
#   rate = counts / steps, score = EMA(rate) with half-life MOE_ADAPT_HALFLIFE epochs.
# * Swap: exchange rows a (HBM slot) <-> b (DRAM slot) in every per-expert tensor of the layer (weights, scales,
#   gate rows, routing biases, clamp limits). Addresses never change, so CUDA graphs stay valid, and since the gate
#   rows move with the weights the routing decisions are unchanged.

# Counting knobs (defaults = the original behaviour):
#   MOE_ADAPT_COUNT=sel|touch   sel: every routed selection (+1 per token per expert, prefill included);
#                               touch: +1 per distinct expert per MoE call, i.e. what a step actually reads once.
#   MOE_ADAPT_MAX_TOKENS=N      only count calls with <= N tokens (decode / spec-verify steps; big prefill chunks
#                               touch nearly every expert regardless of placement). Default 128 for touch, 0 = all.
#   MOE_ADAPT_FUSED=1           one Triton kernel per MoE call instead of 7 small torch kernels.
_COUNT_TOUCH = os.environ.get("MOE_ADAPT_COUNT", "sel").lower() == "touch"
_COUNT_FUSED = os.environ.get("MOE_ADAPT_FUSED", "0") == "1"
_COUNT_MAX_TOK = int(os.environ.get("MOE_ADAPT_MAX_TOKENS", "128" if _COUNT_TOUCH else "0"))
_COUNT_KERNEL = None


def _count_kernel():
    global _COUNT_KERNEL
    if _COUNT_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit(do_not_specialize=["n_ids"])
        def _moe_count(ids_ptr, n_ids, cnt_ptr, n_e, TOUCH: tl.constexpr, BLOCK: tl.constexpr, NBINS: tl.constexpr):
            acc = tl.zeros([NBINS], dtype=tl.int32)
            for start in range(0, n_ids, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                m = offs < n_ids
                ids = tl.load(ids_ptr + offs, mask=m, other=0).to(tl.int32)
                ok = m & (ids >= 0) & (ids < n_e)            # dummy / padded rows can carry ids outside [0, E)
                acc += tl.histogram(tl.where(ok, ids, 0), NBINS, mask=ok)
            b = tl.arange(0, NBINS)
            bm = b < n_e
            if TOUCH:
                inc = (acc > 0).to(tl.float32)
            else:
                inc = acc.to(tl.float32)
            old = tl.load(cnt_ptr + b, mask=bm, other=0.0)
            tl.store(cnt_ptr + b, old + inc, mask=bm)
        _COUNT_KERNEL = _moe_count
    return _COUNT_KERNEL


def _count_fused(topk_ids, c, touch):
    ids = topk_ids.reshape(-1)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    n_e = c.numel()
    nbins = 1 << max(n_e - 1, 1).bit_length()
    _count_kernel()[(1,)](ids, ids.numel(), c, n_e, TOUCH=touch, BLOCK=1024, NBINS=nbins, num_warps=4)


def _patch_adapt(log):
    import torch
    from vllm.model_executor.layers.fused_moe.experts import trtllm_mxfp4_moe as tm
    cls = tm.TrtLlmMxfp4ExpertsModular
    if not getattr(cls, "_moe_adapt_patched", False):
        inner = cls.apply

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **kw):
            c = getattr(self, "_moe_counts", None)
            if c is not None:
                # static per captured graph size / per eager call -> capture-safe
                if _COUNT_MAX_TOK <= 0 or hidden_states.size(0) <= _COUNT_MAX_TOK:
                    if _COUNT_FUSED:
                        _count_fused(topk_ids, c, _COUNT_TOUCH)
                    elif _COUNT_TOUCH:
                        n_e = c.numel()
                        ids = topk_ids.reshape(-1).long()
                        ids = torch.where((ids >= 0) & (ids < n_e), ids, n_e)
                        hit = self._moe_hit
                        hit.zero_()
                        hit.index_fill_(0, ids, 1.0)
                        c.add_(hit[:n_e])
                    else:
                        ids = topk_ids.reshape(-1).long()
                        n_e = c.numel()
                        valid = ((ids >= 0) & (ids < n_e)).to(c.dtype)
                        c.index_add_(0, ids.clamp(0, n_e - 1), valid)
                tr = getattr(self, "_moe_trace", None)
                if tr is not None:
                    ids = topk_ids.reshape(-1).long()
                    n_e = c.numel()
                    valid = ((ids >= 0) & (ids < n_e)).to(c.dtype)
                    # distinct experts this call touches, split by memory tier
                    hit =torch.zeros_like(c).index_add_(0, ids.clamp(0, n_e - 1), valid) > 0
                    hm = self._moe_hostmask
                    tr[0] += (hit & hm).sum()
                    tr[1] += (hit & ~hm).sum()
                    tr[2] += 1
            return inner(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **kw)
        cls.apply = apply
        cls._moe_adapt_patched = True
    from vllm.v1.worker.gpu import model_runner as mr
    run_cls = mr.GPUModelRunner
    if not getattr(run_cls, "_moe_adapt_patched", False):
        orig = run_cls.execute_model

        def execute_model(self, scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
            if (_ADAPT is not None and not dummy
                    and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0):
                _ADAPT.on_step()
            return orig(self, scheduler_output, *a, **kw)
        run_cls.execute_model = execute_model
        run_cls._moe_adapt_patched = True
    log.info("moe_placement: adaptive expert cache hooks installed")


class _AdaptiveCache:
    def __init__(self, placed, E, dev, log):
        import numpy as np
        import torch
        self.E, self.log = E, log
        self.epoch_s = float(os.environ.get("MOE_ADAPT_EPOCH_S", "3"))
        self.max_swaps = int(os.environ.get("MOE_ADAPT_SWAPS", "64"))
        self.half_life = float(os.environ.get("MOE_ADAPT_HALFLIFE", "10"))
        self.margin = float(os.environ.get("MOE_ADAPT_MARGIN", "0.25"))
        self.min_gain = float(os.environ.get("MOE_ADAPT_MIN_GAIN", "0.002"))
        # policy "z" (default): EMA of raw counts + Poisson significance gate; "ratio": EMA of per-step rates (v1)
        self.policy = os.environ.get("MOE_ADAPT_POLICY", "z")
        self.z = float(os.environ.get("MOE_ADAPT_Z", "3"))
        self.layers = []
        for idx, m, lp in sorted(placed, key=lambda t: t[0]):
            n_hot = int(lp["n_hot"])
            segs = lp.get("segments") or [[0, n_hot, "hbm"], [n_hot, E, "host"]]
            host = np.zeros(E, bool)
            for a, b, kind in segs:
                if kind == "host":
                    host[a:b] = True
            tensors = []
            for refs in _expert_groups(m, E).values():
                if any(str(r[1]).startswith("_moe") for r in refs):
                    continue
                t = max(refs, key=lambda r: r[2].numel())[2]
                tensors.append(t)
            fe = m.experts.routed_experts.quant_method.moe_kernel.impl.fused_experts
            order = getattr(m, "_moe_placement_order", None)
            self.layers.append({"idx": idx, "host": host, "tensors": tensors, "fe": fe, "segs": [list(x) for x in segs],
                                "n_hot": n_hot, "score0": lp.get("score"),
                                "order": order.cpu().numpy().copy() if order is not None else np.arange(E)})
        L = len(self.layers)
        self.counts = torch.zeros((L, E), dtype=torch.float32, device=f"cuda:{dev}")
        ones = torch.ones(1 << 17, dtype=torch.float32, device=f"cuda:{dev}")
        for i, lay in enumerate(self.layers):
            lay["fe"]._moe_counts = self.counts[i]
            lay["fe"]._moe_ones = ones
            lay["fe"]._moe_hit = torch.zeros(E + 1, dtype=torch.float32, device=f"cuda:{dev}")
        if _COUNT_FUSED:                          # compile + sanity-check the counting kernel once, outside capture
            ids = torch.tensor([[0, 3, 3, E - 1, -1, E]], dtype=torch.int32, device=f"cuda:{dev}")
            tst = torch.zeros(E, dtype=torch.float32, device=f"cuda:{dev}")
            _count_fused(ids, tst, False)
            _count_fused(ids, tst, True)
            torch.cuda.synchronize()
            exp = torch.zeros(E); exp[0], exp[3], exp[E - 1] = 2, 3, 2
            assert torch.equal(tst.cpu(), exp), tst.nonzero().flatten().tolist()
        log.info("moe_placement: counting mode %s, max tokens/call %s, fused kernel %s",
                 "touch" if _COUNT_TOUCH else "sel", _COUNT_MAX_TOK or "all", _COUNT_FUSED)
        self.host = np.stack([lay["host"] for lay in self.layers])
        self.trace = None
        if os.environ.get("MOE_BW_TRACE") == "1":
            self.trace = torch.zeros((L, 3), dtype=torch.float32, device=f"cuda:{dev}")
            for i, lay in enumerate(self.layers):
                lay["fe"]._moe_trace = self.trace[i]
                lay["fe"]._moe_hostmask = torch.tensor(lay["host"], device=f"cuda:{dev}")
            import torch as _t
            per = [(t, t[0].numel() * t.element_size()) for t in self.layers[0]["tensors"] if t.dim() == 3]
            self.w_bytes = sum(b for t, b in per if t.dtype == _t.uint8)          # FP4 weights (live in the slot's tier)
            self.s_bytes = sum(b for t, b in per if t.dtype != _t.uint8)          # block scales (always HBM)
            log.info("moe_placement: bandwidth trace on (per expert: %.3f MiB weights, %.3f MiB scales)",
                     self.w_bytes / 2**20, self.s_bytes / 2**20)
        self.score = None
        # warm start: EMA scores saved with the slot order (same policy) keep the swap gate's long-term history
        sc0 = [lay.pop("score0", None) for lay in self.layers]
        if all(x is not None and len(x) == E for x in sc0) and os.environ.get("MOE_ADAPT_RESUME_SCORES", "1") == "1":
            self.score = np.array(sc0, dtype=np.float64)
            log.info("moe_placement: restored adaptive scores (%.0f counts total)", self.score.sum())
        self.started = False
        self.steps = 0
        self.epoch = 0
        self.total_swaps = 0
        self.t_last = time.time()
        # MOE_ADAPT_SAVE=<path>: every MOE_ADAPT_SAVE_EVERY epochs (default 100 = 5 min) write the current slot order
        # as a plan file; MOE_ADAPT_RESUME=<same path> at the next start restores it (warm start after restarts).
        self.save_path = os.environ.get("MOE_ADAPT_SAVE", "")
        self.save_every = int(os.environ.get("MOE_ADAPT_SAVE_EVERY", "100"))
        self.saved_swaps = 0
        log.info("moe_placement: adaptive cache on %d layers (%d per-expert tensors/layer); epoch %.1fs, <=%d swaps/epoch, "
                 "half-life %.0f epochs, margin %.2f, policy %s (z=%.1f)", L, len(self.layers[0]["tensors"]), self.epoch_s,
                 self.max_swaps, self.half_life, self.margin, self.policy, self.z)

    def on_step(self):
        import torch
        now = time.time()
        if not self.started:                      # drop counts from warm-up / graph capture
            torch.cuda.synchronize()
            self.counts.zero_()
            self.started, self.steps, self.t_last = True, 0, now
            return
        self.steps += 1
        if now - self.t_last >= self.epoch_s:
            self._epoch(now)

    def _save(self):
        t0 = time.time()
        st = {"num_experts": self.E, "block": 16, "source": "adaptive-cache state", "epoch": self.epoch,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "total_swaps": self.total_swaps,
              "policy": self.policy,
              "layers": {str(lay["idx"]): {"order": [int(x) for x in lay["order"]], "segments": lay["segs"],
                                           "n_hot": lay["n_hot"]} for lay in self.layers}}
        if self.score is not None:
            for li, lay in enumerate(self.layers):
                st["layers"][str(lay["idx"])]["score"] = [round(float(x), 2) for x in self.score[li]]
        try:
            d = os.path.dirname(self.save_path) or "."
            os.makedirs(d, exist_ok=True)
            tmp = f"{self.save_path}.tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, self.save_path)
            self.saved_swaps = self.total_swaps
            self.log.info("moe_placement[adapt] saved slot order to %s (epoch %d, %.1f ms)", self.save_path,
                          self.epoch, (time.time() - t0) * 1e3)
        except OSError as e:
            self.log.warning("moe_placement[adapt] cannot save state to %s: %s", self.save_path, e)

    def _swap(self, li, a, b):
        lay = self.layers[li]
        for t in lay["tensors"]:
            ra = t[a].clone()
            t[a].copy_(t[b])
            t[b].copy_(ra)
        o = lay["order"]
        o[a], o[b] = o[b], o[a]
        self.score[li, [a, b]] = self.score[li, [b, a]]

    def _epoch(self, now):
        import numpy as np
        import torch
        t0 = time.time()
        torch.cuda.synchronize()
        c = self.counts.cpu().numpy().astype(np.float64)
        self.counts.zero_()
        dt = max(now - self.t_last, 1e-6)
        if self.trace is not None:
            tr = self.trace.cpu().numpy().astype(np.float64)
            self.trace.zero_()
            calls = max(tr[:, 2].max(), 1)
            cold_t, hot_t = tr[:, 0].sum(), tr[:, 1].sum()
            self.log.info("moe_placement[bw] %.1fs: %d steps (%.1f/s), forwards/layer %d; per forward: %.1f DRAM + %.1f HBM "
                          "experts touched (all layers); expert reads DRAM %.1f GB/s, HBM %.1f GB/s",
                          dt, self.steps, self.steps / dt, calls, cold_t / calls, hot_t / calls,
                          cold_t * self.w_bytes / dt / 1e9, (hot_t * (self.w_bytes + self.s_bytes) + cold_t * self.s_bytes) / dt / 1e9)
        steps, self.steps, self.t_last = max(self.steps, 1), 0, now
        tot = c.sum()
        if tot < 6 * 40 * 64:                       # too little traffic this epoch to learn from
            return
        self.epoch += 1
        rate = c / steps
        beta = 0.5 ** (1.0 / self.half_life)
        if self.policy == "z":
            # effective counts over ~half-life window; comparable across epochs of similar duration
            self.score = c if self.score is None else beta * self.score + c
        else:
            self.score = rate if self.score is None else beta * self.score + (1 - beta) * rate
        cold_share = (c * self.host).sum() / tot
        pairs = []
        for li in range(len(self.layers)):
            sc, hm = self.score[li], self.host[li]
            cs = np.where(hm)[0]
            hs = np.where(~hm)[0]
            cs = cs[np.argsort(-sc[cs], kind="stable")]
            hs = hs[np.argsort(sc[hs], kind="stable")]
            for a_cold, b_hot in zip(cs, hs):
                x, y = sc[a_cold], sc[b_hot]
                if self.policy == "z":
                    ok = x > y * (1 + self.margin) and (x - y) > self.z * np.sqrt(x + y + 1.0)
                else:
                    ok = x > y * (1 + self.margin) + self.min_gain
                if ok:
                    pairs.append((x - y, li, int(b_hot), int(a_cold)))
                else:
                    break
        pairs.sort(reverse=True)
        todo = pairs[: self.max_swaps]
        for _gain, li, h, cslot in todo:
            self._swap(li, h, cslot)
        torch.cuda.synchronize()
        self.total_swaps += len(todo)
        pred = (self.score * self.host).sum() / max(self.score.sum(), 1e-9)
        if self.save_path and self.epoch % self.save_every == 0 and self.total_swaps != self.saved_swaps:
            self._save()
        if todo or self.epoch <= 20 or self.epoch % 20 == 0:
            self.log.info("moe_placement[adapt] epoch %d: %d steps, DRAM share of routed selections %.2f%%, "
                          "%d candidates, %d swaps (total %d) in %.1f ms; predicted DRAM share now %.2f%%",
                          self.epoch, steps, cold_share * 100, len(pairs), len(todo), self.total_swaps,
                          (time.time() - t0) * 1e3, pred * 100)


# ----------------------------------------------------------------------------------------------- tech-report leads
#
# MOE_MHC_MEGA=1     Lead 2: no mHC side stream at TP1, so every token count takes DeepGEMM Mega-mHC
#                    (sm100_mega_mhc_impl, 152 CTAs) instead of the 4-kernel overlap path (6 CTAs at c1) that vLLM
#                    tuned for GB200 TP4 and uses for <= 16 tokens in FULL graphs (MHC_OVERLAP_MAX_TOKENS).
# MOE_AV_CURVE=...   Lead 1: cost curve of DSpark adaptive verification (speculative config
#                    "enable_adaptive_verification": true). vLLM profiles it once at startup with padding tokens, which
#                    route to no expert, so the routed-expert MoE (~70% of a c32 step on this box) is missing from it.
#                      builtin (default)  vLLM's profiled curve, only logged
#                      moe[:scale]        profiled + MOE_AV_LAYERS (40) x per-layer MoE cost (MOE_AV_MOE_MS, ms by
#                                         tokens, from kernel/kbench.py on the hot/cold layout with production tactics)
#                      online[:scale]     moe seed, then every MOE_STEP_TRACE_S s the buckets with >= MOE_AV_MIN_N
#                                         measured decode steps are replaced by measured step time - draft cost
#                      file:<json>        {"verify": [[tokens, ms], ...], "draft": [[reqs, ms], ...]} (either key)
#                      scale:<a>          profiled verify curve x a
# MOE_STEP_TRACE=1   log the median GPU time of pure-decode steps per token count every MOE_STEP_TRACE_S seconds (a CUDA
#                    event at each execute_model entry; consecutive events bracket one step while the GPU stays busy).
#                    A second line lists the medians by (requests, tokens) seen in the last period.
# MOE_AV_MIN_REQS=r  adaptive verification prunes drafts only in batches of >= r requests; smaller batches verify every
#                    draft the per-batch-size schedule proposed (as without AV).
# MOE_AV_UNIFORM_CG=1  with AV: also capture uniform-decode FULL graphs for 1..r-1 requests (AV otherwise captures only
#                    variable-length decode graphs, ~1 ms slower at 8 requests x 4 tokens); see _patch_av_uniform_cg.
_AV_MOE_MS = ("1:0.05,2:0.075,4:0.125,6:0.174,8:0.2,12:0.257,16:0.31,24:0.413,32:0.47,40:0.6175,48:0.628,64:0.79,"
              "96:1.05,128:1.3,192:1.8")
_TRACE = None


def _patch_mhc_mega(log):
    import vllm.models.deepseek_v41.nvidia.model as m41
    if getattr(m41, "_moe_mhc_mega", False):
        return
    m41.supports_mhc_overlap = lambda *a, **k: False
    m41._moe_mhc_mega = True
    log.info("moe_placement: MOE_MHC_MEGA=1 -> no mHC overlap side stream; Mega-mHC for every token count")


def _moe_ms_fn(spec=None):
    import numpy as np
    spec = spec or os.environ.get("MOE_AV_MOE_MS", _AV_MOE_MS)
    pts = sorted((float(a), float(b)) for a, b in (x.split(":") for x in spec.split(",")))
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    tail = float(os.environ.get("MOE_AV_TAIL_SLOPE", "0.0005"))       # ms per extra token per layer past the table

    def f(n):
        return float(np.interp(n, xs, ys)) if n <= xs[-1] else float(ys[-1] + tail * (n - xs[-1]))
    return f


def _fmt_curve(c, keys=(1, 2, 4, 6, 8, 12, 16, 18, 24, 32, 40, 48, 64, 96, 128, 192, 256, 1024, 8192)):
    d = dict(c)
    return " ".join(f"{k}:{d[k]:.2f}" for k in keys if k in d)


def _model_verify(verify, scale, spec=None):
    n_layers = int(os.environ.get("MOE_AV_LAYERS", "40"))
    f = _moe_ms_fn(spec)
    return [(n, ms + scale * n_layers * f(n)) for n, ms in verify]


def _av_verify_curve(mode, draft, verify):
    """-> (draft, verify) to use, or None for vLLM's own curves."""
    if mode in ("", "builtin"):
        return None
    kind, _, arg = mode.partition(":")
    if kind in ("moe", "online"):
        return draft, _model_verify(verify, float(arg or 1.0))
    if kind == "scale":
        return draft, [(n, ms * float(arg)) for n, ms in verify]
    if kind == "file":
        d = json.load(open(arg))
        return ([tuple(x) for x in d.get("draft", draft)], [tuple(x) for x in d.get("verify", verify)])
    raise ValueError(f"MOE_AV_CURVE={mode!r}")


class _StepTrace:
    def __init__(self, log):
        self.log = log
        self.pending = []                       # [event, tokens, reqs, pure_decode]
        self.samples = {}                       # tokens -> [(ms, reqs)], newest last
        self.period_rt = {}                     # (reqs, tokens) -> [ms] in the current period
        self.period = float(os.environ.get("MOE_STEP_TRACE_S", "20"))
        self.min_n = int(os.environ.get("MOE_AV_MIN_N", "30"))
        self.t_last = time.time()
        self.mgr = None                         # AdaptiveVerificationManager (set at its curve init)
        self.base = None                        # (draft, model verify) curves the online mode starts from
        self.online = os.environ.get("MOE_AV_CURVE", "").startswith("online")

    def on_step(self, so):
        import torch
        reqs = getattr(so, "num_scheduled_tokens", {}) or {}
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.pending.append([ev, int(so.total_num_scheduled_tokens), len(reqs),
                             bool(reqs) and max(reqs.values()) <= 8])
        if time.time() - self.t_last >= self.period:
            self.t_last = time.time()
            self.flush()

    def set_tokens(self, n):
        if self.pending:
            self.pending[-1][1] = int(n)

    def flush(self):
        import numpy as np
        k = 0
        for i in range(len(self.pending) - 1):
            a, b = self.pending[i], self.pending[i + 1]
            if not b[0].query():
                break
            ms = a[0].elapsed_time(b[0])
            if a[3] and 0 < ms < 1000:
                _AVST["dms"] += ms
                _AVST["dms_n"] += 1
                lst = self.samples.setdefault(a[1], [])
                lst.append((ms, a[2]))
                self.period_rt.setdefault((a[2], a[1]), []).append(ms)
                if len(lst) > 2000:
                    del lst[:1000]
            k = i + 1
        self.pending = self.pending[k:]
        rows = [(n, len(v), float(np.median([x[0] for x in v[-400:]])), float(np.mean([x[0] for x in v[-400:]])))
                for n, v in sorted(self.samples.items()) if len(v) >= 5]
        rt = sorted((k, len(v), float(np.median(v))) for k, v in self.period_rt.items() if len(v) >= 10)
        self.period_rt = {}
        if rt:
            self.log.info("moe_placement[step] last period by reqs x tokens (r/n:median(count)): %s",
                          " ".join(f"{r}/{n}:{m:.1f}({c})" for (r, n), c, m in rt))
        if rows:
            self.log.info("moe_placement[step] decode step ms by tokens (n:median/mean of last 400 (count)): %s",
                          " ".join(f"{n}:{m:.1f}/{a:.1f}({c})" for n, c, m, a in rows))
        if self.online and self.mgr is not None and self.base is not None:
            self._update_curve()

    def _update_curve(self):
        import numpy as np
        draft, model = self.base
        tables = self.mgr.cost_tables
        if tables is None:
            return
        draft_tab = tables[0]
        xs = [n for n, _ in model]
        meas = {}
        for n, v in self.samples.items():
            v = v[-400:]
            if not v:
                continue
            j = int(np.searchsorted(xs, n, side="left"))
            if j >= len(xs):
                continue
            meas.setdefault(xs[j], []).extend(ms - float(draft_tab[min(r, len(draft_tab) - 1)]) for ms, r in v)
        meas = {b: float(np.median(v)) for b, v in meas.items() if len(v) >= self.min_n}
        if not meas:
            return
        mb = sorted(meas)
        md = dict(model)
        lx = np.log(np.array(mb, dtype=np.float64))
        res = np.array([meas[b] - md[b] for b in mb])
        new = []
        for n, ms in model:                      # measured buckets as measured; others: model + residual interpolated in log n
            new.append((n, meas[n]) if n in meas else (n, ms + float(np.interp(np.log(max(n, 1)), lx, res))))
        self.mgr.set_cost_curves(draft, new)
        self.log.info("moe_placement[av] online verify curve (measured buckets %s): %s",
                      " ".join(f"{b}:{meas[b]:.1f}" for b in mb), _fmt_curve(new))


# AV runtime control (av-persist). Knobs, all applied only to batches of >= min_reqs requests:
#   MOE_AV_MAX_K=k        admit at most k drafts per verifying request on average (budget <= k x requests)
#   MOE_AV_CAP_TOKENS=n   at most n target tokens per step (e.g. the largest CUDA-graph capture size)
#   MOE_AV_DRAFT_MS=c|r:ms,..   replace the profiled draft-cost curve (constant or points by requests)
#   MOE_AV_VERIFY=n:ms,..       replace the verify-cost curve (points by target tokens; step function between them)
#   MOE_AV_CTL=<json>     re-read when its mtime changes (checked once per second): keys min_reqs, max_k, cap_tokens,
#                         force_k (budget = k x requests) / force_tokens (calibration only), min_k, draft, verify, moe_ms, moe_scale,
#                         label. A key missing from the file falls back to the env value.
# With MOE_STEP_TRACE=1, MOE_AV_CTL or MOE_AV_LOG=1: every MOE_STEP_TRACE_S s (and at each ctl change) one
# "moe_placement[avdec]" line logs generated tok/s, pure-decode tokens/steps and GPU ms/steps (decode efficiency), and per
# request count the steps, the mean scheduled target tokens and the mean tokens AV kept.
#   MOE_AV_STATIC_SCHED=[[lo,hi,k],...]  per-request draft cap by batch size (ctl key static_sched; emulates a static
#                         dspark schedule inside an AV server, e.g. production [[1,4,5],[5,8,3],[9,16,2],[17,32,1]])
def _parse_curve(x, keys=None):
    if isinstance(x, (int, float)):
        return [(k, float(x)) for k in (keys or [1])]
    if isinstance(x, str):
        x = x.strip()
        if ":" not in x:
            return [(k, float(x)) for k in (keys or [1])]
        return sorted((int(a), float(b)) for a, b in (y.split(":") for y in x.split(",")))
    return sorted((int(a), float(b)) for a, b in x)


def _env_num(name, cast=float):
    v = os.environ.get(name, "").strip()
    return cast(v) if v else None


_AVC_ENV = {"min_reqs": int(os.environ.get("MOE_AV_MIN_REQS", "0") or 0), "max_k": _env_num("MOE_AV_MAX_K"),
            "cap_tokens": _env_num("MOE_AV_CAP_TOKENS", int), "force_k": None, "force_tokens": None,
            "min_k": None, "varlen_only": False,
            "static_sched": json.loads(os.environ["MOE_AV_STATIC_SCHED"]) if os.environ.get("MOE_AV_STATIC_SCHED") else None}
_AVC = dict(_AVC_ENV)
_AV_CTL_PATH = os.environ.get("MOE_AV_CTL", "").strip()
_AV_LOG = (os.environ.get("MOE_STEP_TRACE") == "1" or bool(_AV_CTL_PATH) or os.environ.get("MOE_AV_LOG") == "1")
_AVS = {"init": None, "mgr": None, "ctl_t": 0.0, "ctl_mtime": None, "curve_keys": False}
_AVST = {"t": time.time(), "rows": {}, "hist": {}, "gen": 0, "steps": 0, "dgen": 0, "dsteps": 0, "dms": 0.0,
         "dms_n": 0}


def _av_stats_flush(force=False):
    now = time.time()
    if not force and now - _AVST["t"] < float(os.environ.get("MOE_STEP_TRACE_S", "20")):
        return
    dt = max(1e-6, now - _AVST["t"])
    _AVST["t"] = now
    rows, hist, gen, steps = _AVST["rows"], _AVST["hist"], _AVST["gen"], _AVST["steps"]
    dg, ds, dms, dmn = _AVST["dgen"], _AVST["dsteps"], _AVST["dms"], _AVST["dms_n"]
    _AVST["rows"], _AVST["hist"], _AVST["gen"], _AVST["steps"] = {}, {}, 0, 0
    _AVST["dgen"], _AVST["dsteps"], _AVST["dms"], _AVST["dms_n"] = 0, 0, 0.0, 0
    if rows or gen:
        _logger().info("moe_placement[avdec] %.1f s: gen %.1f tok/s, %.1f steps/s, decode %d/%d tok/steps %.1f/%d ms/steps | "
                       "reqs:steps/sched/kept (mean target tokens): %s | kept-token hist (r>=%s): %s", dt, gen / dt, steps / dt,
                       dg, ds, dms, dmn,
                       " ".join(f"{r}:{v[0]}/{v[1] / v[0]:.1f}/{v[2] / v[0]:.1f}" for r, v in sorted(rows.items())),
                       _AVC["min_reqs"], " ".join(f"{n}:{c}" for n, c in sorted(hist.items())))


def _av_apply_curves(mgr, spec, log_label=""):
    init = _AVS["init"]
    if init is None:
        return
    draft0, verify0, cur0 = init                  # profiled draft, profiled verify, curves set at init
    draft, verify = cur0
    if "draft" in spec:
        draft = _parse_curve(spec["draft"], [r for r, _ in draft0])
    if "verify" in spec:
        verify = _parse_curve(spec["verify"])
    elif "moe_ms" in spec or "moe_scale" in spec:
        verify = _model_verify(verify0, float(spec.get("moe_scale", 1.0)), spec.get("moe_ms"))
    mgr.set_cost_curves(draft, verify)
    if _TRACE is not None:
        _TRACE.base = (draft, verify)
    _logger().info("moe_placement[avctl] %s curves: draft ms by reqs %s | verify ms by tokens %s", log_label,
                   _fmt_curve(draft), _fmt_curve(verify))


def _av_ctl_poll(mgr):
    path = _AV_CTL_PATH
    if not path:
        return
    now = time.time()
    if now - _AVS["ctl_t"] < 1.0:
        return
    _AVS["ctl_t"] = now
    try:
        mt = os.stat(path).st_mtime
    except OSError:
        return
    if mt == _AVS["ctl_mtime"]:
        return
    _AVS["ctl_mtime"] = mt
    try:
        spec = json.load(open(path))
    except Exception as e:                          # noqa: BLE001
        _logger().warning("moe_placement[avctl] cannot read %s: %r", path, e)
        return
    if _TRACE is not None:
        _TRACE.flush()
    _av_stats_flush(force=True)
    _AVC.clear()
    _AVC.update(_AVC_ENV)
    for k in ("min_reqs", "max_k", "cap_tokens", "force_k", "force_tokens", "min_k", "varlen_only", "static_sched"):
        if k in spec:
            _AVC[k] = spec[k]
    has_curve = any(k in spec for k in ("draft", "verify", "moe_ms", "moe_scale"))
    if has_curve or _AVS["curve_keys"]:
        _av_apply_curves(mgr, spec, spec.get("label", ""))
    _AVS["curve_keys"] = has_curve
    _logger().info("moe_placement[avctl] applied %s: %s", spec.get("label", ""), json.dumps(_AVC))


def _patch_av(log):
    global _TRACE
    import numpy as np
    from collections import defaultdict
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av
    mode = os.environ.get("MOE_AV_CURVE", "builtin").strip()
    trace_on = os.environ.get("MOE_STEP_TRACE") == "1" or mode.startswith("online")
    if trace_on and _TRACE is None:
        _TRACE = _StepTrace(log)
    cls = av.AdaptiveVerificationManager
    if not getattr(cls, "_moe_av_patched", False):
        orig_init = cls.set_initial_cost_curves
        orig_get = cls.get_num_tokens

        def set_initial_cost_curves(self, samples):
            def median_curve(points):
                g = defaultdict(list)
                for k, v in points:
                    g[k].append(v)
                return [(k, float(np.median(v))) for k, v in sorted(g.items())]
            draft = median_curve((s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph)
            verify = median_curve((s.num_target_tokens, s.forward_ms) for s in samples)
            _logger().info("moe_placement[av] profiled curves: draft ms by reqs %s | verify ms by tokens %s",
                           _fmt_curve(draft), _fmt_curve(verify))
            new = _av_verify_curve(mode, draft, verify) if draft and verify else None
            env_draft = os.environ.get("MOE_AV_DRAFT_MS", "").strip()
            env_verify = os.environ.get("MOE_AV_VERIFY", "").strip()
            if draft and verify and (env_draft or env_verify):
                d0, v0 = new if new is not None else (draft, verify)
                new = (_parse_curve(env_draft, [r for r, _ in draft]) if env_draft else d0,
                       _parse_curve(env_verify) if env_verify else v0)
            if new is None:
                orig_init(self, samples)
            else:
                self.set_cost_curves(*new)
                _logger().info("moe_placement[av] MOE_AV_CURVE=%s: draft ms by reqs %s | verify ms by tokens %s", mode,
                               _fmt_curve(new[0]), _fmt_curve(new[1]))
            if draft and verify:
                _AVS["init"] = (draft, verify, new if new is not None else (draft, verify))
            _AVS["mgr"] = self
            if _TRACE is not None:
                _TRACE.mgr = self
                _TRACE.base = new if new is not None else (draft, verify)
            if os.environ.get("MOE_AV_PREWARM", "1") == "1":
                _prewarm_budget(self)

        def get_num_tokens(self, *a, **kw):
            _av_ctl_poll(self)
            n = orig_get(self, *a, **kw)
            bb = self._batch_budget
            if bb is not None:
                nd, nn, b = bb
                r = len(nd)
                full = min(sum(nd.values()), max(0, self._max_total_logits - r * self.num_bonus_tokens))
                nond = sum(nn.values())
                c = _AVC
                nd_changed = False
                ss = c.get("static_sched")
                if ss:                      # per-request draft cap by batch size, e.g. [[1,4,5],[5,8,3],[9,16,2],[17,32,1]]
                    kc = next((int(k) for lo, hi, k in ss if int(lo) <= r <= int(hi)), None)
                    if kc is not None and any(v > kc for v in nd.values()):
                        nd = {q: min(v, kc) for q, v in nd.items()}
                        nd_changed = True
                        full = min(sum(nd.values()), full)
                        b = min(b, full)
                if c["min_reqs"] and r < c["min_reqs"]:
                    b2 = full               # small batch: verify every scheduled draft (as without AV)
                else:
                    b2 = b
                    nv = sum(1 for v in nd.values() if v > 0)
                    if c.get("force_k") is not None:
                        b2 = min(full, int(round(nv * float(c["force_k"]))))
                    if c.get("force_tokens") is not None:
                        b2 = min(full, max(0, int(c["force_tokens"]) - nond))
                    if c.get("min_k") is not None:
                        b2 = max(b2, min(full, int(round(nv * float(c["min_k"])))))
                    if c.get("max_k") is not None:
                        b2 = min(b2, int(round(nv * float(c["max_k"]))))
                    if c.get("cap_tokens"):
                        b2 = min(b2, max(0, int(c["cap_tokens"]) - nond))
                if b2 != b or nd_changed:
                    self._batch_budget = (nd, nn, b2)
                    n = nond + b2
                if _AV_LOG:
                    st = _AVST["rows"].setdefault(r, [0, 0, 0])
                    st[0] += 1
                    st[1] += nond + full
                    st[2] += n
                    if c["min_reqs"] and r >= c["min_reqs"] or not c["min_reqs"]:
                        _AVST["hist"][n] = _AVST["hist"].get(n, 0) + 1
            if _TRACE is not None:
                _TRACE.set_tokens(n)
            if _AV_LOG:
                _av_stats_flush()
            return n
        cls.set_initial_cost_curves = set_initial_cost_curves
        cls.get_num_tokens = get_num_tokens
        cls._moe_av_patched = True
    if trace_on:
        from vllm.v1.worker.gpu import model_runner as mr
        run_cls = mr.GPUModelRunner
        if not getattr(run_cls, "_moe_trace_patched", False):
            orig = run_cls.execute_model

            def execute_model(self, scheduler_output, *a, **kw):
                dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
                if (_TRACE is not None and not dummy
                        and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0):
                    _TRACE.on_step(scheduler_output)
                return orig(self, scheduler_output, *a, **kw)
            run_cls.execute_model = execute_model
            run_cls._moe_trace_patched = True
    if os.environ.get("MOE_AV_UNIFORM_CG") == "1":
        _patch_av_uniform_cg(log, int(os.environ.get("MOE_AV_MIN_REQS", "0") or 0) - 1)
    if _AV_LOG:
        try:
            from vllm.v1.core.sched import scheduler as sch
            scls = sch.Scheduler
            if not getattr(scls, "_moe_gen_patched", False):
                orig_upd = scls.update_from_output

                def update_from_output(self, scheduler_output, model_runner_output, *a, **kw):
                    st = getattr(model_runner_output, "sampled_token_ids", None)
                    if st:
                        g = sum(len(x) for x in st)
                        _AVST["gen"] += g
                        _AVST["steps"] += 1
                        nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
                        if nst and max(nst.values()) <= 8:
                            _AVST["dgen"] += g
                            _AVST["dsteps"] += 1
                    return orig_upd(self, scheduler_output, model_runner_output, *a, **kw)
                scls.update_from_output = update_from_output
                scls._moe_gen_patched = True
        except Exception as e:                      # noqa: BLE001
            log.warning("moe_placement[av] generated-token counter not installed: %r", e)
    log.info("moe_placement: adaptive-verification curve mode %r, step trace %s, AV min reqs %s, uniform graphs %s",
             mode, trace_on, os.environ.get("MOE_AV_MIN_REQS", "0"), os.environ.get("MOE_AV_UNIFORM_CG", "0"))


def _prewarm_budget(mgr):
    """Compile vLLM's torch.compile'd draft-slot ranking (_assign_draft_token_budget, dynamic shapes) at startup. It is
    otherwise compiled on the first step that prunes drafts (e.g. the first batch of >= MOE_AV_MIN_REQS requests) and
    stalls that step for ~4 s."""
    import torch
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av
    t0 = time.time()
    try:
        probs = torch.rand_like(mgr._confidence_probs)
        steps = mgr.num_speculative_steps
        nmax = probs.shape[0]
        for r in sorted({min(nmax, x) for x in (1, 2, 8, 17, 24, nmax)}):
            idx = torch.randperm(nmax, device=probs.device)[:r].to(torch.int64)
            cap = mgr._batch_draft_capacity[:r]
            cap.fill_(steps)
            av._assign_draft_token_budget_compiled(probs, idx, cap, max(1, r * steps // 2), steps)
        torch.cuda.synchronize()
        _logger().info("moe_placement[av] pre-compiled the draft-slot ranking kernel in %.1f s", time.time() - t0)
    except Exception as e:                          # noqa: BLE001
        _logger().warning("moe_placement[av] draft-slot ranking pre-compile failed: %r", e)


def _patch_av_uniform_cg(log, max_r):
    """With adaptive verification on, vLLM captures only variable-length decode graphs (num_reqs = num_tokens, any query
    length up to 6). Below MOE_AV_MIN_REQS every scheduled draft is verified, so those batches are uniform: also capture
    the uniform-decode FULL graphs vLLM uses without AV (one per batch size 1..max_r, exact request count) and dispatch a
    uniform batch to them first. Varlen graphs stay the fallback for every token count."""
    if max_r < 1:
        return
    from vllm.v1.worker.gpu import cudagraph_utils as cu
    cls = cu.CudaGraphManager
    if getattr(cls, "_moe_uniform_patched", False):
        return
    orig = cls._init_candidates
    orig_dispatch = cls.dispatch

    def _init_candidates(self):
        orig(self)
        try:
            _add_uniform_descs(self, cu, max_r, log)
        except Exception as e:                      # noqa: BLE001
            log.warning("moe_placement[av] uniform decode graphs not added: %r", e)

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, *a, **kw):
        # ctl "varlen_only": true -> the target model ignores the added uniform graphs (A/B of graph kinds, experiments)
        if uniform_token_count is not None and _AVC.get("varlen_only") and getattr(self, "_moe_uniform_added", False):
            uniform_token_count = None
        return orig_dispatch(self, num_reqs, num_tokens, uniform_token_count, *a, **kw)
    cls._init_candidates = _init_candidates
    cls.dispatch = dispatch
    cls._moe_uniform_patched = True


def _add_uniform_descs(self, cu, max_r, log):
    from vllm.config.compilation import CUDAGraphMode
    if not (self.varlen_decode and self._capture_descs) or list(self.lora_capture_cases) != [0] \
            or self.ubatch_runner is not None:
        return
    decode_mode = self.cudagraph_mode.decode_mode()
    if decode_mode != CUDAGraphMode.FULL or not self.cudagraph_mode.separate_routine():
        return
    spec = self.vllm_config.speculative_config
    nspec = self.vllm_config.num_speculative_tokens
    nnew = self.decode_query_len - nspec
    if spec is not None and spec.uses_dynamic_speculative_decoding() and nnew > 0:
        dense = cu.build_dynamic_sd_schedule_lookup(spec.num_speculative_tokens_per_batch_size,
                                                    vllm_max_batch_size=self.max_num_reqs,
                                                    vllm_num_speculative_tokens=nspec)
    else:
        dense = [nspec] * (self.max_num_reqs + 1)
    limit = self.compilation_config.max_cudagraph_capture_size or 0
    full = list(self._capture_descs.get(decode_mode, []))
    extra = []
    for r in range(1, min(max_r, self.max_num_reqs) + 1):
        q = int(dense[r]) + nnew
        d = cu.BatchExecutionDescriptor(cg_mode=decode_mode, num_tokens=r * q, num_reqs=r, uniform_token_count=q)
        if r * q <= limit and d not in full and d not in extra:
            extra.append(d)
    if not extra:
        return
    allf = full + extra
    asc = sorted(allf, key=lambda d: (d.num_tokens, d.uniform_token_count is None, d.num_reqs or 0))
    self._capture_descs[decode_mode] = sorted(allf, key=lambda d: d.num_tokens, reverse=True)
    old = self._candidates
    maxt = max([k[0] for k in old] + [asc[-1].num_tokens])
    new = {}
    for i in range(0, maxt + 1):                  # FULL: every graph that fits, smallest first, uniform before varlen
        fl = [d for d in asc if d.num_tokens >= i]
        rest = [d for d in old.get((i, 0), []) if d.cg_mode != decode_mode]
        if fl or rest:
            new[(i, 0)] = fl + rest
    self._candidates = new
    self._moe_uniform_added = True
    log.info("moe_placement[av] + %d uniform decode FULL graphs (reqs x tokens/req): %s", len(extra),
             " ".join(f"{d.num_reqs}x{d.uniform_token_count}" for d in extra))
