"""TidalMoE hot/cold expert placement + adaptive expert cache for Qwen3.6-35B-A3B (Qwen3.5-MoE) on stock vLLM 0.33.

Port of plugin/moe_placement (DeepSeek-V4.1 on GB300) to a consumer GPU over PCIe (RTX 5060 Ti, WSL2). Loaded as a vLLM
general plugin. After the MoE kernel has repacked its weights and before vLLM profiles memory, every routed-expert layer
(Qwen3NextSparseMoeBlock) is rewritten as:

  * experts permuted per layer so the hot ones come first; the router (GateLinear) rows are permuted the same way, so
    routing decisions are unchanged and the kernel simply sees slot ids, and
  * each large per-expert tensor (w13/w2 FP4 weights and their block scales) rebuilt on one contiguous CUDA
    virtual-address range whose head (hot experts) is backed by VRAM and whose tail (cold experts) is backed by pinned
    host memory (cuMemCreate/cuMemMap, verified to work under WSL2). The MoE kernel reads hot experts at VRAM speed
    and cold ones zero-copy over PCIe.

The adaptive cache counts which slots every MoE call touches (a capture-safe op inside the CUDA graphs) and, every
MOE_ADAPT_EPOCH_S seconds between steps, swaps cold experts that are used significantly more than hot ones (all
per-expert tensors incl. router rows, in place, so CUDA graphs stay valid). The slot order can be saved and restored
across restarts (MOE_ADAPT_SAVE / MOE_ADAPT_RESUME).

Not ported (DeepSeek/B300-specific): forced TRT-LLM tactics, SM100 persistent MoE kernels, mHC, kernel-aware cold groups.

Env:
  MOE_PLACEMENT=off|inspect|apply      default off; inspect = log the per-expert tensors and hook targets only
  MOE_HOT_GB=<GiB>                     VRAM for routed-expert tensors (weights + scales), split evenly over layers when
                                       there is no plan; default 7.0
  MOE_PLACEMENT_PLAN=/path/plan.json   {"num_experts": E, "layers": {"<idx>": {"order": [...], "n_hot": k}}}
                                       order[i] = logical expert stored in slot i; slots < n_hot -> VRAM
  MOE_BIG_MIN_MIB=8                    per-expert tensors at least this large per layer are split VRAM/host; smaller
                                       ones (router rows, global scales) are permuted in place and stay in VRAM
  MOE_ADAPT=1                          adaptive cache; knobs MOE_ADAPT_EPOCH_S (3), MOE_ADAPT_SWAPS (64),
                                       MOE_ADAPT_HALFLIFE (10), MOE_ADAPT_MARGIN (0.25), MOE_ADAPT_Z (3),
                                       MOE_ADAPT_COUNT=touch|sel (touch), MOE_ADAPT_MAX_TOKENS (256 for touch)
  MOE_ADAPT_SAVE=/path, MOE_ADAPT_RESUME=/path, MOE_ADAPT_SAVE_EVERY=20 (epochs)
  MOE_BW_TRACE=1                       log experts touched per forward by tier and the implied PCIe read rate
"""
import gc
import json
import os
import time

_MODE = os.environ.get("MOE_PLACEMENT", "off").lower()
_KEEP = []            # VMM handles + array-interface owners, alive for the process lifetime
_ADAPT = None
_DONE = False
_BIG_MIN = int(os.environ.get("MOE_BIG_MIN_MIB", "8")) << 20
_COUNT_TOUCH = os.environ.get("MOE_ADAPT_COUNT", "touch").lower() == "touch"
_COUNT_MAX_TOK = int(os.environ.get("MOE_ADAPT_MAX_TOKENS", "256" if _COUNT_TOUCH else "0"))


def _logger():
    from vllm.logger import init_logger
    return init_logger("vllm.tidal_qwen")


def _patch_uva_repack(log):
    """vLLM 0.33 bug fix (UVA weight offload + repacking quant methods): replace_parameter() keeps the Parameter object
    and rebinds .data to the repacked tensor in VRAM, but the object still carries _vllm_is_uva_offloaded=True, so
    device_loading_context() never re-offloads it. With ModelOpt NVFP4 MoE (Marlin repack) every "offloaded" expert
    ends up in VRAM (and WDDM silently spills the overflow to system memory). Re-offload such parameters right after
    each layer's repack, so --cpu-offload-gb behaves as documented."""
    import contextlib
    import torch
    from vllm.model_executor.model_loader import utils as lu
    if getattr(lu, "_tq_uva_fixed", False):
        return
    orig = lu.device_loading_context
    stats = {"n": 0, "bytes": 0}

    @contextlib.contextmanager
    def device_loading_context(module, target_device):
        with orig(module, target_device) as m:
            yield m
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        moved = False
        for _name, p in module.named_parameters():
            if getattr(p, "_vllm_is_uva_offloaded", False) and p.is_cuda and _is_host(p) is False:
                cpu = torch.empty_like(p.data, device="cpu", pin_memory=True).copy_(p.data)
                p.data = get_accelerator_view_from_cpu_tensor(cpu)
                stats["n"] += 1
                stats["bytes"] += cpu.numel() * cpu.element_size()
                moved = True
        if moved:
            torch.cuda.empty_cache()
            if stats["n"] % 32 == 0:
                log.info("tidal_qwen: re-offloaded %d repacked UVA params (%.2f GiB) so far", stats["n"],
                         stats["bytes"] / 2**30)
    lu.device_loading_context = device_loading_context
    lu._tq_uva_fixed = True
    log.info("tidal_qwen: UVA re-offload-after-repack fix installed")


def register():
    log = _logger()
    if os.environ.get("UVA_REPACK_FIX", "1") == "1":
        _patch_uva_repack(log)
    if _MODE == "off":
        return
    from vllm.model_executor.models import qwen3_5
    patched = 0
    for cname in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"):
        cls = getattr(qwen3_5, cname, None)
        if cls is None or getattr(cls, "_tq_patched", False):
            continue
        orig = getattr(cls, "process_weights_after_loading", None)   # model-level hook: after per-layer quant finalize

        def hook(self, *a, _orig=orig, **kw):
            out = _orig(self, *a, **kw) if _orig is not None else None
            global _DONE
            if not _DONE:
                _DONE = True
                run(self)
            return out
        cls.process_weights_after_loading = hook
        cls._tq_patched = True
        patched += 1
    log.info("tidal_qwen: patched %d Qwen3.5-MoE model classes (mode=%s)", patched, _MODE)
    if os.environ.get("MOE_ADAPT") == "1" and _MODE == "apply":
        _patch_step(log)


# ----------------------------------------------------------------------------------------------- discovery

def _routed(m):
    """The module holding the expert weights: vLLM 0.33 nests them as block.experts (MoERunner).routed_experts."""
    ex = getattr(m, "experts", None)
    if ex is None:
        return None
    re_ = getattr(ex, "routed_experts", None)
    if re_ is not None and hasattr(re_, "w13_weight"):
        return re_
    return ex if hasattr(ex, "w13_weight") else None


def _moe_blocks(model):
    """[(layer_idx, name, block, E)] for every routed-expert block (a module with .gate and routed experts)."""
    out = []
    for name, m in model.named_modules():
        if getattr(m, "gate", None) is None or type(m).__name__ in ("RoutedExperts", "MoERunner"):
            continue
        ex = _routed(m)
        if ex is None:
            continue
        E = getattr(ex, "global_num_experts", None) or getattr(ex, "num_experts", None)
        digits = [p for p in name.split(".") if p.isdigit()]
        if not E or not digits:
            continue
        out.append((int(digits[-1]), name, m, int(E)))
    out.sort(key=lambda t: t[0])
    return out


def _walk_tensors(root, max_depth=9):
    """(owner, key, tensor, path) for tensors reachable from ``root`` without leaving it (from plugin/moe_placement)."""
    import torch
    inside = {id(x) for x in root.modules()}
    seen, out = set(), []

    def visit(obj, path, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            if id(obj) not in inside:
                return
            items = (list(obj._parameters.items()) + list(obj._buffers.items())
                     + [(k, v) for k, v in vars(obj).items() if not k.startswith("_")] + list(obj._modules.items()))
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
    visit = None
    return out


def _expert_groups(m, E):
    """Per-expert tensors (dim0 in {E, E+1}: a fused shared expert sits in row E) grouped by storage."""
    groups = {}
    for owner, key, t, path in _walk_tensors(m):
        if t.dim() >= 1 and t.shape[0] in (E, E + 1) and t.is_cuda and not str(key).startswith("_tq"):
            groups.setdefault(t.untyped_storage().data_ptr(), []).append((owner, key, t, path))
    return groups


def _group_bytes(refs):
    base = max(refs, key=lambda r: r[2].numel())[2]
    return base.numel() * base.element_size()


def _is_host(t):
    from cuda.bindings import driver as cu
    err, mt = cu.cuPointerGetAttribute(cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, t.data_ptr())
    return None if err != cu.CUresult.CUDA_SUCCESS else int(mt) == int(cu.CUmemorytype.CU_MEMORYTYPE_HOST)


def _kernel_obj(ex):
    """The object whose apply() gets topk_ids inside the MoE custom op (modular kernel experts), else the router."""
    qm = getattr(ex, "quant_method", None)
    for path in ("moe_kernel.impl.fused_experts", "moe_kernel.fused_experts", "fused_experts", "moe_mk.fused_experts"):
        o = qm
        for p in path.split("."):
            o = getattr(o, p, None)
            if o is None:
                break
        if o is not None and hasattr(o, "apply"):
            return o, "apply"
    r = getattr(ex, "router", None)
    if r is not None and hasattr(r, "select_experts"):
        return r, "select_experts"
    return None, None


def _inspect(blocks, log):
    kos = {id(_kernel_obj(_routed(m))[0]) for _i, _n, m, _E in blocks}
    log.info("tidal_qwen[inspect] %d blocks, %d distinct kernel objects", len(blocks), len(kos))
    for idx, name, m, E in blocks[:1] + blocks[-1:]:
        ex = _routed(m)
        ko, meth = _kernel_obj(ex)
        log.info("tidal_qwen[inspect] L%d %s: E=%d block=%s runner=%s experts=%s quant_method=%s kernel=%s.%s router=%s",
                 idx, name, E, type(m).__name__, type(m.experts).__name__, type(ex).__name__,
                 type(getattr(ex, "quant_method", None)).__name__, type(ko).__name__ if ko is not None else None, meth,
                 type(getattr(m.experts, "router", None)).__name__)
        for owner, key, t, path in _walk_tensors(m):
            if E in tuple(t.shape) or (E + 1) in tuple(t.shape) or t.numel() * t.element_size() >= (4 << 20):
                log.info("tidal_qwen[inspect] L%d %s shape=%s dtype=%s contig=%s host=%s %.2fMiB ptr=%x", idx, path,
                         tuple(t.shape), t.dtype, t.is_contiguous(), _is_host(t), t.numel() * t.element_size() / 2**20,
                         t.untyped_storage().data_ptr())


# ----------------------------------------------------------------------------------------------- VMM

def _ck(ret):
    from cuda.bindings import driver as cu
    err, *vals = ret if isinstance(ret, tuple) else (ret,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}: {cu.cuGetErrorString(err)[1]}")
    return vals[0] if len(vals) == 1 else vals


class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None}


def _alloc_segments(nbytes, segments, dev):
    """uint8 CUDA tensor on one VA range whose byte segments [(start, end, "hbm"|"host")] are backed by VRAM or pinned
    host memory. A page goes to host only if it lies entirely inside a host segment (from plugin/moe_placement)."""
    import torch
    from cuda.bindings import driver as cu

    def prop(loc_type, loc_id):
        p = cu.CUmemAllocationProp()
        p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        p.location.type = loc_type
        p.location.id = loc_id
        return p
    dprop = prop(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, dev)
    hprop = prop(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, 0)
    rec = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    g = max(int(_ck(cu.cuMemGetAllocationGranularity(dprop, rec))), int(_ck(cu.cuMemGetAllocationGranularity(hprop, rec))))
    size = -(-nbytes // g) * g
    n_pages = size // g
    on_host = [False] * n_pages
    for s, e, kind in segments:
        if kind == "host":
            for pg in range(-(-s // g), e // g):
                on_host[pg] = True
    runs = []
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
        try:
            h = _ck(cu.cuMemCreate(length, p, 0))
        except RuntimeError as e:
            where = "VRAM" if p is dprop else "host memory"
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


# ----------------------------------------------------------------------------------------------- placement

def _segments(lp, E, rows):
    """Row segments for a per-expert tensor with ``rows`` rows (E, or E+1 with a fused shared expert kept in VRAM)."""
    segs = [list(x) for x in (lp.get("segments") or [[0, lp["n_hot"], "hbm"], [lp["n_hot"], E, "host"]])]
    if rows > E:
        segs.append([E, rows, "hbm"])
    return segs


def _place_layer(idx, m, lp, E, dev, log):
    import torch
    order = torch.tensor(lp["order"], dtype=torch.long, device=f"cuda:{dev}")
    assert sorted(lp["order"]) == list(range(E))
    hbm = host = 0
    names = []
    groups = _expert_groups(m, E)
    keys = sorted(groups, key=lambda k: _group_bytes(groups[k]))       # small tensors first, big ones one at a time
    for k in keys:
        refs = groups.pop(k)
        base = max(refs, key=lambda r: r[2].numel())[2]
        rows = base.shape[0]
        perm = order if rows == E else torch.cat([order, torch.arange(E, rows, device=order.device)])
        big = _group_bytes(refs) >= _BIG_MIN
        names.append(refs[0][3].split(".")[-1] + ("*" if big else ""))
        if not big:                                    # router rows, per-expert global/input scales: in place
            base.copy_(base.index_select(0, perm))
            continue
        assert base.is_contiguous() and base.storage_offset() == 0, (refs[0][3], base.stride())
        row = base[0].numel() * base.element_size()
        flat, hb, hs = _alloc_segments(row * rows, [(s * row, e * row, kd) for s, e, kd in _segments(lp, E, rows)], dev)
        new = flat.view(base.dtype).view(base.shape)
        for i0 in range(0, rows, 32):                  # permuted copy; the source may be host (UVA) resident
            sl = perm[i0:i0 + 32]
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
    m._tq_order = order
    return hbm, host, names


def _expert_row_bytes(m, E):
    return sum(_group_bytes(refs) // max(refs, key=lambda r: r[2].numel())[2].shape[0]
               for refs in _expert_groups(m, E).values() if _group_bytes(refs) >= _BIG_MIN)


def _default_plan(blocks, log):
    """Uniform split: MOE_HOT_GB of VRAM for routed experts, the same number of hot slots in every layer (a multiple
    of 4 so the 2 MiB VMM pages of the FP4 weights split cleanly; partial scale pages stay in VRAM); identity order."""
    E = blocks[0][3]
    per = _expert_row_bytes(blocks[0][2], E)
    hot_gb = float(os.environ.get("MOE_HOT_GB", "7.0"))
    n_hot = int(hot_gb * 2**30 / (per * len(blocks)))
    n_hot = max(0, min(E, n_hot - n_hot % 4))
    log.info("tidal_qwen: no plan file; uniform plan with %d/%d hot experts per layer (%.2f MiB per expert, "
             "MOE_HOT_GB=%.1f -> %.2f GiB in VRAM)", n_hot, E, per / 2**20, hot_gb, n_hot * per * len(blocks) / 2**30)
    return {"num_experts": E, "layers": {str(i): {"order": list(range(E)), "n_hot": n_hot} for i, _n, _m, _E in blocks}}


def _replan_from_scores(st, plan, E, log, src):
    """Plan with ``plan``'s hot-slot counts but the experts ranked by the learned scores of a saved adaptive state
    (score[slot] belongs to logical expert order[slot]); carries the scores over so the swap gate keeps its history."""
    if st.get("num_experts") != E:
        raise ValueError("different expert count")
    new = {"num_experts": E, "layers": {}}
    for k, lp in plan["layers"].items():
        sl = st["layers"].get(k)
        if sl is None or "score" not in sl:
            raise ValueError(f"layer {k} has no scores")
        by_expert = {int(e): float(s) for e, s in zip(sl["order"], sl["score"])}
        order = sorted(range(E), key=lambda e: -by_expert.get(e, 0.0))
        new["layers"][k] = {"order": order, "n_hot": lp["n_hot"], "score": [by_expert.get(e, 0.0) for e in order]}
    log.info("tidal_qwen: re-planned from learned scores in %s (epoch %s): hottest experts first, %d hot slots/layer",
             src, st.get("epoch"), next(iter(plan["layers"].values()))["n_hot"])
    return new


def run(model):
    import torch
    log = _logger()
    t0 = time.time()
    blocks = _moe_blocks(model)
    if not blocks:
        log.warning("tidal_qwen: no routed-expert blocks found; nothing to do")
        return
    E = blocks[0][3]
    log.info("tidal_qwen: %d MoE layers with %d experts", len(blocks), E)
    if _MODE == "inspect":
        _inspect(blocks, log)
        return
    path = os.environ.get("MOE_PLACEMENT_PLAN", "")
    plan = json.load(open(path)) if path and os.path.exists(path) else _default_plan(blocks, log)
    assert plan["num_experts"] == E
    res = os.environ.get("MOE_ADAPT_RESUME", "")
    if res and not os.path.exists(res):
        # no state for this layout yet: re-plan from the newest adaptive state saved with another hot budget
        d = os.path.dirname(res) or "."
        cands = sorted((os.path.join(d, f) for f in (os.listdir(d) if os.path.isdir(d) else [])
                        if f.startswith("adapt-") and f.endswith(".json")), key=os.path.getmtime, reverse=True)
        for c in cands:
            try:
                plan = _replan_from_scores(json.load(open(c)), plan, E, log, c)
                break
            except Exception as e:                      # noqa: BLE001
                log.warning("tidal_qwen: cannot re-plan from %s: %s", c, e)
    elif res:
        try:
            st = json.load(open(res))
            seg = lambda lp: [list(x) for x in (lp.get("segments") or [[0, lp["n_hot"], "hbm"], [lp["n_hot"], E, "host"]])]
            if (st.get("num_experts") == E and set(st["layers"]) == set(plan["layers"])
                    and all(seg(st["layers"][k]) == seg(plan["layers"][k]) for k in plan["layers"])):
                log.info("tidal_qwen: warm start from %s (epoch %s, %s)", res, st.get("epoch"), st.get("saved_at"))
                plan = st
            else:
                plan = _replan_from_scores(st, plan, E, log, res)
        except Exception as e:                          # noqa: BLE001
            log.warning("tidal_qwen: cannot use %s: %s", res, e)
    dev = torch.cuda.current_device()
    hbm_q, host_q = [], []
    for idx, _name, m, _E in blocks:
        lp = plan["layers"].get(str(idx))
        if lp is None:
            continue
        big = [r[2] for refs in _expert_groups(m, E).values() if _group_bytes(refs) >= _BIG_MIN for r in refs]
        (host_q if any(_is_host(t) for t in big) else hbm_q).append((idx, m, lp))
    tot_hbm = tot_host = 0
    placed = []
    per = _expert_row_bytes(blocks[0][2], E)
    while hbm_q or host_q:
        # an on-host (UVA-offloaded at load) layer when free VRAM can hold its hot part with margin, else shrink a
        # VRAM-resident layer first (its transient copy needs room too)
        pick = None
        if host_q:
            idx, m, lp = host_q[0]
            if torch.cuda.mem_get_info()[0] - per * int(lp["n_hot"]) > (1 << 30) or not hbm_q:
                pick = host_q.pop(0)
        if pick is None:
            pick = hbm_q.pop(0)
        idx, m, lp = pick
        placed.append(pick)
        free0 = torch.cuda.mem_get_info()[0]
        hb, hs, names = _place_layer(idx, m, lp, E, dev, log)
        tot_hbm += hb
        tot_host += hs
        torch.cuda.synchronize()
        free1 = torch.cuda.mem_get_info()[0]
        log.info("tidal_qwen: L%d n_hot=%d -> VRAM %.2f GiB, host %.2f GiB; GPU free %.2f -> %.2f GiB%s", idx,
                 lp["n_hot"], hb / 2**30, hs / 2**30, free0 / 2**30, free1 / 2**30,
                 f"; permuted {names}" if idx in (blocks[0][0], blocks[-1][0]) else "")
    free, total = torch.cuda.mem_get_info()
    log.info("tidal_qwen: placed in %.1fs -> routed experts VRAM %.2f GiB, host %.2f GiB; GPU free %.1f/%.1f GiB",
             time.time() - t0, tot_hbm / 2**30, tot_host / 2**30, free / 2**30, total / 2**30)
    if os.environ.get("MOE_ADAPT") == "1":
        global _ADAPT
        _ADAPT = _AdaptiveCache(placed, E, dev, log)


# ----------------------------------------------------------------------------------------------- adaptive cache

def _count(topk_ids, c, hit, touch):
    import torch
    n_e = c.numel()
    ids = topk_ids.reshape(-1).long()
    ok = (ids >= 0) & (ids < n_e)
    if touch:                                          # +1 per distinct expert per call = what a step reads once
        hit.zero_()
        hit.index_fill_(0, torch.where(ok, ids, n_e), 1.0)
        c.add_(hit[:n_e])
    else:                                              # +1 per routed selection
        c.index_add_(0, ids.clamp(0, n_e - 1), ok.to(c.dtype))


def _install_count_hook(obj, meth, log):
    cls = type(obj)
    if getattr(cls, f"_tq_{meth}_patched", False):
        return
    inner = getattr(cls, meth)

    if meth == "apply":
        def wrapped(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **kw):
            c = getattr(self, "_tq_counts", None)
            if c is not None and (_COUNT_MAX_TOK <= 0 or hidden_states.size(0) <= _COUNT_MAX_TOK):
                _count(topk_ids, c, self._tq_hit, _COUNT_TOUCH)
                tr = getattr(self, "_tq_trace", None)
                if tr is not None:
                    _trace(topk_ids, c.numel(), self._tq_hostmask, tr)
            return inner(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **kw)
    else:
        def wrapped(self, hidden_states, *a, **kw):
            out = inner(self, hidden_states, *a, **kw)
            c = getattr(self, "_tq_counts", None)
            if c is not None and (_COUNT_MAX_TOK <= 0 or hidden_states.size(0) <= _COUNT_MAX_TOK):
                topk_ids = out[1]
                _count(topk_ids, c, self._tq_hit, _COUNT_TOUCH)
                tr = getattr(self, "_tq_trace", None)
                if tr is not None:
                    _trace(topk_ids, c.numel(), self._tq_hostmask, tr)
            return out
    setattr(cls, meth, wrapped)
    setattr(cls, f"_tq_{meth}_patched", True)
    log.info("tidal_qwen: counting hook on %s.%s", cls.__name__, meth)


def _trace(topk_ids, n_e, hostmask, tr):
    import torch
    ids = topk_ids.reshape(-1).long()
    ok = ((ids >= 0) & (ids < n_e)).float()
    hit = torch.zeros(n_e, dtype=torch.float32, device=ids.device).index_add_(0, ids.clamp(0, n_e - 1), ok) > 0
    tr[0] += (hit & hostmask).sum()
    tr[1] += (hit & ~hostmask).sum()
    tr[2] += 1


def _patch_step(log):
    from vllm.v1.worker.gpu import model_runner as mr
    run_cls = mr.GPUModelRunner
    if getattr(run_cls, "_tq_patched", False):
        return
    orig = run_cls.execute_model

    def execute_model(self, scheduler_output, *a, **kw):
        dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
        if _ADAPT is not None and not dummy and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0:
            _ADAPT.on_step()
        return orig(self, scheduler_output, *a, **kw)
    run_cls.execute_model = execute_model
    run_cls._tq_patched = True
    log.info("tidal_qwen: adaptive-cache step hook on %s.execute_model", run_cls.__name__)


class _AdaptiveCache:
    """Port of plugin/moe_placement._AdaptiveCache (policy "z": EMA of counts + Poisson significance gate)."""

    def __init__(self, placed, E, dev, log):
        import numpy as np
        import torch
        self.E, self.log = E, log
        self.epoch_s = float(os.environ.get("MOE_ADAPT_EPOCH_S", "3"))
        self.max_swaps = int(os.environ.get("MOE_ADAPT_SWAPS", "64"))
        self.half_life = float(os.environ.get("MOE_ADAPT_HALFLIFE", "10"))
        self.margin = float(os.environ.get("MOE_ADAPT_MARGIN", "0.25"))
        self.z = float(os.environ.get("MOE_ADAPT_Z", "3"))
        self.layers = []
        for idx, m, lp in sorted(placed, key=lambda t: t[0]):
            n_hot = int(lp["n_hot"])
            segs = [list(x) for x in (lp.get("segments") or [[0, n_hot, "hbm"], [n_hot, E, "host"]])]
            host = np.zeros(E, bool)
            for a, b, kind in segs:
                if kind == "host":
                    host[a:b] = True
            tensors = [max(refs, key=lambda r: r[2].numel())[2] for refs in _expert_groups(m, E).values()]
            ko, meth = _kernel_obj(_routed(m))
            if ko is None:
                raise RuntimeError("tidal_qwen: no MoE kernel/router object to hook for counting")
            _install_count_hook(ko, meth, log)
            order = getattr(m, "_tq_order", None)
            self.layers.append({"idx": idx, "host": host, "tensors": tensors, "ko": ko, "segs": segs, "n_hot": n_hot,
                                "score0": lp.get("score"),
                                "order": order.cpu().numpy().copy() if order is not None else np.arange(E)})
        if len({id(l["ko"]) for l in self.layers}) != len(self.layers):
            raise RuntimeError("tidal_qwen: MoE layers share one kernel object; per-layer counting would mix layers")
        L = len(self.layers)
        self.counts = torch.zeros((L, E), dtype=torch.float32, device=f"cuda:{dev}")
        for i, lay in enumerate(self.layers):
            lay["ko"]._tq_counts = self.counts[i]
            lay["ko"]._tq_hit = torch.zeros(E + 1, dtype=torch.float32, device=f"cuda:{dev}")
        self.host = np.stack([lay["host"] for lay in self.layers])
        self.trace = None
        if os.environ.get("MOE_BW_TRACE") == "1":
            self.trace = torch.zeros((L, 3), dtype=torch.float32, device=f"cuda:{dev}")
            for i, lay in enumerate(self.layers):
                lay["ko"]._tq_trace = self.trace[i]
                lay["ko"]._tq_hostmask = torch.tensor(lay["host"], device=f"cuda:{dev}")
            self.row_bytes = _expert_row_bytes(placed[0][1], E)
        self.score = None
        sc0 = [lay.pop("score0", None) for lay in self.layers]
        if all(x is not None and len(x) == E for x in sc0):
            self.score = np.array(sc0, dtype=np.float64)
            log.info("tidal_qwen: restored adaptive scores (%.0f counts total)", self.score.sum())
        self.started, self.steps, self.epoch, self.total_swaps = False, 0, 0, 0
        self.t_last = time.time()
        self.save_path = os.environ.get("MOE_ADAPT_SAVE", "")
        self.save_every = int(os.environ.get("MOE_ADAPT_SAVE_EVERY", "20"))
        self.saved_swaps = 0
        log.info("tidal_qwen: adaptive cache on %d layers (%d per-expert tensors/layer, %d hot slots); counting %s "
                 "(<= %s tokens/call); epoch %.1fs, <= %d swaps/epoch, half-life %.0f epochs, margin %.2f, z %.1f",
                 L, len(self.layers[0]["tensors"]), self.layers[0]["n_hot"], "touch" if _COUNT_TOUCH else "sel",
                 _COUNT_MAX_TOK or "all", self.epoch_s, self.max_swaps, self.half_life, self.margin, self.z)

    def on_step(self):
        import torch
        now = time.time()
        if not self.started:                           # drop counts from warm-up / graph capture
            torch.cuda.synchronize()
            self.counts.zero_()
            if self.trace is not None:
                self.trace.zero_()
            self.started, self.steps, self.t_last = True, 0, now
            return
        self.steps += 1
        if now - self.t_last >= self.epoch_s:
            self._epoch(now)

    def _save(self):
        st = {"num_experts": self.E, "source": "tidal_qwen adaptive-cache state", "epoch": self.epoch,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "total_swaps": self.total_swaps,
              "layers": {str(l["idx"]): {"order": [int(x) for x in l["order"]], "segments": l["segs"],
                                         "n_hot": l["n_hot"]} for l in self.layers}}
        if self.score is not None:
            for li, l in enumerate(self.layers):
                st["layers"][str(l["idx"])]["score"] = [round(float(x), 2) for x in self.score[li]]
        try:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            tmp = f"{self.save_path}.tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, self.save_path)
            self.saved_swaps = self.total_swaps
            self.log.info("tidal_qwen[adapt] saved slot order to %s (epoch %d)", self.save_path, self.epoch)
        except OSError as e:
            self.log.warning("tidal_qwen[adapt] cannot save state to %s: %s", self.save_path, e)

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
            self.log.info("tidal_qwen[bw] %.1fs: %d steps (%.1f/s), forwards/layer %d; per forward %.1f host + %.1f VRAM "
                          "experts touched (all layers); expert reads over PCIe %.2f GB/s", dt, self.steps,
                          self.steps / dt, calls, cold_t / calls, hot_t / calls, cold_t * self.row_bytes / dt / 1e9)
        steps, self.steps, self.t_last = max(self.steps, 1), 0, now
        tot = c.sum()
        if tot < 4 * len(self.layers) * 8:             # too little traffic this epoch to learn from
            return
        self.epoch += 1
        beta = 0.5 ** (1.0 / self.half_life)
        self.score = c if self.score is None else beta * self.score + c
        cold_share = (c * self.host).sum() / tot
        pairs = []
        for li in range(len(self.layers)):
            sc, hm = self.score[li], self.host[li]
            cs, hs = np.where(hm)[0], np.where(~hm)[0]
            cs = cs[np.argsort(-sc[cs], kind="stable")]
            hs = hs[np.argsort(sc[hs], kind="stable")]
            for a_cold, b_hot in zip(cs, hs):
                x, y = sc[a_cold], sc[b_hot]
                if x > y * (1 + self.margin) and (x - y) > self.z * np.sqrt(x + y + 1.0):
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
            self.log.info("tidal_qwen[adapt] epoch %d: %d steps, host share of expert reads %.1f%%, %d candidates, "
                          "%d swaps (total %d) in %.1f ms; predicted host share now %.1f%%", self.epoch, steps,
                          cold_share * 100, len(pairs), len(todo), self.total_swaps, (time.time() - t0) * 1e3, pred * 100)
