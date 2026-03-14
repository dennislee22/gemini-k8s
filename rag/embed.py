import os
import logging

_log_rag = logging.getLogger("rag")

def _detect_gpu_count() -> int:
    explicit = os.getenv("NUM_GPU")
    if explicit is not None:
        return int(explicit)
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return n
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL)
        return len([l for l in out.decode().strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0

NUM_GPU = _detect_gpu_count()

_embedder_fn  = None

def _get_embedder():
    global _embedder_fn
    if _embedder_fn is not None:
        return _embedder_fn

    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5").strip()
    _log_rag.info(f"[Embed] Loading SentenceTransformer: {EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer
    import transformers as _tf
    _tf.logging.set_verbosity_error()

    if NUM_GPU > 0:
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                _log_rag.warning(
                    "[Embed] NUM_GPU=%d but torch.cuda.is_available()=False "
                    "(CUDA runtime issue?) — falling back to CPU", NUM_GPU
                )
                device = "cpu"
        except ImportError:
            pass
    else:
        device = "cpu"

    _log_rag.info(f"[Embed] device={device} (NUM_GPU={NUM_GPU})")
    _st = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)

    def _local(text: str) -> list:
        return _st.encode(text, normalize_embeddings=True).tolist()

    _embedder_fn = _local
    return _embedder_fn

def embed_text(text: str) -> list:
    return _get_embedder()(text)