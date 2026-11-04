"""Host-aware persistent MoE: binding + moe_placement plugin hook (MOE_PERSIST=1).

Per decode MoE call with T <= MOE_PERSIST_MAX_TOKENS tokens (else: the TRT-LLM-gen path, unchanged):
  front (Triton, persist_front.py)   record v2 + permuted / compact routing + gathered MXFP8 rows (xp, xpsf)
  GEMM1 (CuTe, persist_moe_cute.py)  H = MXFP8(clamped SwiGLU(xp @ w13[e]^T)), host-aware schedule, PDL
  GEMM2 (CuTe)                       rows = bf16(H @ w2[e]^T) in compact order, host-aware schedule, PDL
  finalize (Triton)                  out = sum_k w * rows[permuted]   -- or UnfinalizedMoEOutput when vLLM defers it
Env:
  MOE_PERSIST=1                 enable (needs MOE_PLACEMENT=apply: the host mask comes from the plan's segments)
  MOE_PERSIST_MAX_TOKENS=128    largest call taken (decode / spec-verify sizes); larger calls -> TRT-LLM
  MOE_PERSIST_STAGES=6,10       TMA pipeline depth GEMM1, GEMM2
  MOE_PERSIST_KC=4,16           min / max cold CTAs per GEMM (front picks per call)
  MOE_PERSIST_BW=385,6000,46    host GB/s, HBM GB/s, per-CTA GB/s of the split model
  MOE_PERSIST_LAYERS=all|0-9,.. restrict to some layers (bisecting)
  MOE_PERSIST_PDL=1             front / finalize launched with PDL (persist_front.py)
  MOE_PERSIST_WARM=1            run every capture size once at bind (compiles the Triton specialisations outside capture)
"""
import os
from functools import cache

import torch

HIDDEN, INTER, E, TOPK = 5120, 2304, 384, 6
_MAX_T = int(os.environ.get("MOE_PERSIST_MAX_TOKENS", "128"))
_ST = [int(x) for x in os.environ.get("MOE_PERSIST_STAGES", "6,10").split(",")]
_CK = [int(x) for x in os.environ.get("MOE_PERSIST_CTAK", "256,128").split(",")]
_KC = [int(x) for x in os.environ.get("MOE_PERSIST_KC", "4,16").split(",")]
_BW = [float(x) * 1e3 for x in os.environ.get("MOE_PERSIST_BW", "385,6000,46").split(",")]   # GB/s -> bytes/us


def _src():
    """persist_moe_cute / persist_front live next to this file in the plugin (copied from tuning/persist/src)."""
    import sys
    d = os.path.dirname(__file__)
    if d not in sys.path:
        sys.path.insert(0, d)


def _ptr(dtype, t, align):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _fake(dtype, align):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _kernels(dev):
    """Compile GEMM1 / GEMM2 once per device (outside CUDA-graph capture)."""
    _src()
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import make_fake_stream
    import persist_moe_cute as pm
    with torch.cuda.device(dev):
        g1 = cute.compile(pm.RoutedExpertGemm1(HIDDEN, INTER, INTER, cta_k=_CK[0], num_ab_stage=_ST[0]),
                          _fake(cutlass.Float4E2M1FN, 32), _fake(cutlass.Float8E8M0FNU, 16),
                          _fake(cutlass.Float8E4M3FN, 16), _fake(cutlass.Float8E8M0FNU, 16),
                          _fake(cutlass.Uint16, 16), _fake(cutlass.Uint8, 16), _fake(cutlass.Int32, 16),
                          _fake(cutlass.Float32, 16), Int32(8), Int32(E), Int32(4), Int32(8), Int32(1), make_fake_stream())
        g2 = cute.compile(pm.RoutedExpertGemm2(HIDDEN, INTER, cta_k=_CK[1], num_ab_stage=_ST[1]),
                          _fake(cutlass.Float4E2M1FN, 32), _fake(cutlass.Float8E8M0FNU, 16),
                          _fake(cutlass.Float8E4M3FN, 16), _fake(cutlass.Float8E8M0FNU, 16),
                          _fake(cutlass.BFloat16, 16), _fake(cutlass.Int32, 16),
                          Int32(8), Int32(6), Int32(E), Int32(4), Int32(1), make_fake_stream())
    return g1, g2


def _stream():
    import cuda.bindings.driver as cuda
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


class PersistMoE:
    """The persistent MoE of one placed layer (weights in the TRT-LLM-gen layout vLLM prepared, VMM hot/cold)."""

    def __init__(self, fe, w13, w2, host_mask_bool):
        assert tuple(w13.shape) == (E, 2 * INTER, HIDDEN // 2) and tuple(w2.shape) == (E, HIDDEN, INTER // 2), (w13.shape, w2.shape)
        assert fe.w1_bias is None and fe.w2_bias is None and fe.gemm1_alpha is None and fe.gemm1_beta is None
        assert fe.gemm1_clamp_limit is not None and fe.gemm1_clamp_limit.dtype == torch.float32
        self.fe, self.w13, self.w2 = fe, w13, w2
        self.dev = w13.device
        self.host = host_mask_bool.to(torch.int8).to(self.dev)
        self.sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
        _kernels(self.dev.index)

    def accepts(self, hidden_states, a1q_scale, expert_map, topk_ids):
        T = hidden_states.size(0)
        return (expert_map is None and a1q_scale is not None and hidden_states.dtype == torch.float8_e4m3fn
                and 1 <= T <= _MAX_T and hidden_states.is_contiguous() and tuple(hidden_states.shape) == (T, HIDDEN)
                and a1q_scale.numel() == T * (HIDDEN // 32) and a1q_scale.element_size() == 1 and a1q_scale.is_contiguous()
                and tuple(topk_ids.shape) == (T, TOPK) and self.fe.w1_scale is not None)

    def warm(self, log):
        """Run each decode capture size once (both finalize paths) so every Triton specialisation is compiled before
        CUDA-graph capture; checks the outputs are finite."""
        import time
        t0 = time.time()
        dev = self.dev
        sizes = [t for t in (1, 2, 4, 6, 8, 12, 16, 18, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 256) if t <= _MAX_T]
        k = torch.arange(TOPK, device=dev)
        bad = []
        for T in sizes:
            t = torch.arange(T, device=dev)[:, None]
            ids = ((t * 7 + k * 61) % E).to(torch.int32)
            x = (torch.randn(T, HIDDEN, device=dev) * 0.1).to(torch.float8_e4m3fn)
            xs = torch.full((T, HIDDEN // 32), 127, dtype=torch.uint8, device=dev)
            w = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32, device=dev)
            out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
            self(out, x, xs, ids, w)
            u = self(None, x, xs, ids, w)
            torch.cuda.synchronize(dev)
            if not (torch.isfinite(out.float()).all() and torch.isfinite(u.gemm2_permuted.float()).all()):
                bad.append(T)
        log.info("moe_placement: persistent MoE warmed %d sizes (<= %d tokens) in %.1fs%s", len(sizes), _MAX_T,
                 time.time() - t0, "; NON-FINITE at %s" % bad if bad else "")

    def __call__(self, output, x_q, x_scale, topk_ids, topk_weights):
        """output: [T, H] bf16 to finalize into, or None -> UnfinalizedMoEOutput (compact rows)."""
        _src()
        import cutlass
        from cutlass import Int32
        import persist_front as pf
        from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
        T = x_q.size(0)
        M = pf.max_tiles(T)
        C = pf.cta_words(M)
        dev = x_q.device
        ids = topk_ids.to(torch.int32).contiguous()
        xq8 = x_q.view(torch.uint8)
        xsf = x_scale.view(torch.uint8).reshape(T, HIDDEN // 32)
        rec = torch.empty(int(pf.HDR) + 4 * C + 8 * M, dtype=torch.int32, device=dev)
        perm = torch.empty((T, TOPK), dtype=torch.int32, device=dev)
        xp = torch.empty((8 * M, HIDDEN), dtype=torch.uint8, device=dev)
        xpsf = torch.empty((M, HIDDEN // 128 * 512), dtype=torch.uint8, device=dev)
        h = torch.empty((8 * M, INTER // 2), dtype=torch.uint16, device=dev)
        hs = torch.empty((M, INTER // 128 * 512), dtype=torch.uint8, device=dev)
        rows = torch.empty((T * TOPK, HIDDEN), dtype=torch.bfloat16, device=dev)
        pf.front(ids, xq8, xsf, self.host, rec, perm, xp, xpsf, G=self.sms, host_bw=_BW[0], hbm_bw=_BW[1],
                 cta_bw=_BW[2], kc_min=_KC[0], kc_max=_KC[1])
        g1, g2 = _kernels(dev.index)
        fe = self.fe
        s = _stream()
        g1(_ptr(cutlass.Float4E2M1FN, self.w13, 32), _ptr(cutlass.Float8E8M0FNU, fe.w1_scale, 16),
           _ptr(cutlass.Float8E4M3FN, xp, 16), _ptr(cutlass.Float8E8M0FNU, xpsf, 16),
           _ptr(cutlass.Uint16, h, 16), _ptr(cutlass.Uint8, hs, 16), _ptr(cutlass.Int32, rec, 16),
           _ptr(cutlass.Float32, fe.gemm1_clamp_limit, 16),
           Int32(8 * M), Int32(E), Int32(C), Int32(8 * M), Int32(self.sms), s)
        g2(_ptr(cutlass.Float4E2M1FN, self.w2, 32), _ptr(cutlass.Float8E8M0FNU, fe.w2_scale, 16),
           _ptr(cutlass.Float8E4M3FN, h, 16), _ptr(cutlass.Float8E8M0FNU, hs, 16),
           _ptr(cutlass.BFloat16, rows, 16), _ptr(cutlass.Int32, rec, 16),
           Int32(8 * M), Int32(T * TOPK), Int32(E), Int32(C), Int32(self.sms), s)
        if output is None:
            return UnfinalizedMoEOutput(gemm2_permuted=rows, expert_weights=topk_weights.contiguous().view(T, TOPK),
                                        expanded_idx_to_permuted_idx=perm)
        pf.finalize(rows, topk_weights.contiguous(), perm, output)
        return None


def _layer_filter():
    spec = os.environ.get("MOE_PERSIST_LAYERS", "all")
    if spec == "all":
        return lambda i: True
    keep = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return lambda i: i in keep


def _why_not(fe, m):
    """Why this layer's experts cannot take the persistent kernels (None: they can)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import TrtLlmMxfp4ExpertsModular
    if not isinstance(fe, TrtLlmMxfp4ExpertsModular):
        return "experts %s" % type(fe).__name__
    cfg, q = fe.moe_config, fe.quant_config
    if cfg.activation != MoEActivation.SILU:
        return "activation %s" % cfg.activation
    if fe.gemm1_alpha is not None or fe.gemm1_beta is not None or fe.w1_bias is not None or fe.w2_bias is not None:
        return "alpha/beta/bias"
    if fe.local_num_experts != cfg.num_experts or cfg.num_experts != E:
        return "expert parallel / %d experts" % cfg.num_experts
    if q.quant_dtype != "mxfp8" or q.is_scale_swizzled:
        return "activations not MXFP8 with linear scales (%s, swizzled %s)" % (q.quant_dtype, q.is_scale_swizzled)
    if fe.topk != TOPK:
        return "top-%d" % fe.topk
    if getattr(m.experts, "routed_scaling_factor", 1.0) not in (None, 1.0):
        return "routed scaling %s" % m.experts.routed_scaling_factor
    return None


def bind(placed, E_, dev, log):
    """After placement: attach a PersistMoE to every placed layer's TrtLlmMxfp4ExpertsModular (fe._moe_persist)."""
    import numpy as np
    want = _layer_filter()
    n = 0
    first = None
    for idx, m, lp in placed:
        if not want(idx):
            continue
        fe0 = m.experts.routed_experts.quant_method.moe_kernel.impl.fused_experts
        why = _why_not(fe0, m)
        if why is not None:
            log.warning("moe_placement: persistent MoE skips layer %d: %s", idx, why)
            continue
        n_hot = int(lp["n_hot"])
        segs = lp.get("segments") or [[0, n_hot, "hbm"], [n_hot, E_, "host"]]
        host = np.zeros(E_, bool)
        for a, b, kind in segs:
            if kind == "host":
                host[a:b] = True
        fe = m.experts.routed_experts.quant_method.moe_kernel.impl.fused_experts
        r = m.experts.routed_experts
        fe._moe_persist = PersistMoE(fe, r.w13_weight, r.w2_weight, torch.from_numpy(host))
        first = first or fe._moe_persist
        n += 1
    if first is not None and os.environ.get("MOE_PERSIST_WARM", "1") == "1":
        first.warm(log)
    log.info("moe_placement: persistent host-aware MoE bound on %d layers (<= %d tokens, stages %s, kc %s, bw %s)",
             n, _MAX_T, _ST, _KC, _BW)


def patch_apply(log):
    """Wrap TrtLlmMxfp4ExpertsModular.apply (install BEFORE the adaptive-cache wrapper so counting still runs)."""
    from vllm.model_executor.layers.fused_moe.experts import trtllm_mxfp4_moe as tm
    cls = tm.TrtLlmMxfp4ExpertsModular
    if getattr(cls, "_moe_persist_patched", False):
        return
    orig = cls.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        p = getattr(self, "_moe_persist", None)
        if (p is not None and not apply_router_weight_on_input and global_num_experts == E
                and p.accepts(hidden_states, a1q_scale, expert_map, topk_ids)):
            M = hidden_states.size(0)
            defer = self.moe_config.should_defer_moe_finalize(M)
            return p(None if defer else output, hidden_states, a1q_scale, topk_ids, topk_weights)
        return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                    expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                    apply_router_weight_on_input)

    cls.apply = apply
    cls._moe_persist_patched = True
    log.info("moe_placement: TrtLlmMxfp4ExpertsModular.apply -> persistent host-aware MoE for <= %d tokens", _MAX_T)
