"""Token embedding in pinned host memory for stock vLLM (consumer-GPU port of TidalMoE's MOE_HOST_PARAMS).

Loaded as a vLLM general plugin (entry point group ``vllm.general_plugins``). The token embedding is only gathered
(one row per token, ~10 KB per decoded token for Qwen3.6-27B), so it can live in pinned host memory and be read over
PCIe through a UVA view at negligible cost, freeing its VRAM (2.37 GiB of BF16 for Qwen3.6-27B) for weights that are
read in full every step. Unlike TidalMoE (cuMemCreate host pages on Grace), this uses torch pinned memory plus
vLLM's own UVA view helper, which also works under WSL2. The move happens in the layer's __init__, before weights are
loaded, so the embedding is never resident in VRAM. LM heads (``ParallelLMHead``, read in full every step) are skipped.

Env:
  HOST_EMBED=1                 enable (default off)
  HOST_EMBED_MIN_MIB=256       only move embeddings at least this large
"""
import os


def register():
    if os.environ.get("HOST_EMBED", "0") != "1":
        return
    from vllm.logger import init_logger
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe

    log = init_logger("vllm.host_embed")
    cls = vpe.VocabParallelEmbedding
    if getattr(cls, "_host_embed_patched", False):
        return
    min_bytes = int(os.environ.get("HOST_EMBED_MIN_MIB", "256")) << 20
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if isinstance(self, vpe.ParallelLMHead):
            return
        w = getattr(self, "weight", None)
        if w is None or not w.is_cuda or w.numel() * w.element_size() < min_bytes:
            return
        import torch
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        nbytes = w.numel() * w.element_size()
        cpu = torch.empty(w.shape, dtype=w.dtype, device="cpu", pin_memory=True)
        w.data = get_accelerator_view_from_cpu_tensor(cpu)
        w._host_embed_cpu = cpu  # keep the pinned buffer alive
        torch.cuda.empty_cache()
        log.info("host_embed: %s %s %s -> pinned host memory via UVA (%.2f GiB of VRAM freed)",
                 type(self).__name__, tuple(w.shape), str(w.dtype).replace("torch.", ""), nbytes / 2**30)

    cls.__init__ = __init__
    cls._host_embed_patched = True
    log.info("host_embed: patched VocabParallelEmbedding.__init__")
