from pathlib import Path
import config

_embedder_fn  = None
_lancedb_conn  = None
_docs_table    = None
_excel_table   = None
_EMBED_DIM = 768

def _get_embedder():
    global _embedder_fn
    if _embedder_fn is not None: return _embedder_fn

    config._log_rag.info(f"[Embed] Loading SentenceTransformer: {config.EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer
    import transformers as _tf
    _tf.logging.set_verbosity_error()

    if config.NUM_GPU > 0:
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                config._log_rag.warning("[Embed] NUM_GPU=%d but torch.cuda.is_available()=False — falling back to CPU", config.NUM_GPU)
                device = "cpu"
        except ImportError: pass
    else: device = "cpu"

    config._log_rag.info(f"[Embed] device={device} (NUM_GPU={config.NUM_GPU})")
    _st = SentenceTransformer(config.EMBED_MODEL, device=device, trust_remote_code=True)
    _embedder_fn = lambda text: _st.encode(text, normalize_embeddings=True).tolist()
    return _embedder_fn

def embed_text(text: str) -> list:
    return _get_embedder()(text)

def _get_lancedb():
    global _lancedb_conn, _docs_table, _excel_table
    if _lancedb_conn is not None: return _lancedb_conn, _docs_table, _excel_table

    import lancedb
    import pyarrow as pa

    Path(config.LANCEDB_DIR).mkdir(parents=True, exist_ok=True)
    config._log_rag.info(f"[LanceDB] Opening store: {config.LANCEDB_DIR}")
    _lancedb_conn = lancedb.connect(config.LANCEDB_DIR)

    docs_schema = pa.schema([
        pa.field("id", pa.utf8()), pa.field("vector", pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("text", pa.utf8()), pa.field("source", pa.utf8()),
        pa.field("doc_type", pa.utf8()), pa.field("chunk_index", pa.int32()), pa.field("file_hash", pa.utf8()),
    ])
    if "docs" in _lancedb_conn.table_names(): _docs_table = _lancedb_conn.open_table("docs")
    else:
        _docs_table = _lancedb_conn.create_table("docs", schema=docs_schema)
        config._log_rag.info("[LanceDB] Created table: docs")

    excel_schema = pa.schema([
        pa.field("id", pa.utf8()), pa.field("vector", pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("source_file", pa.utf8()), pa.field("file_hash", pa.utf8()),
        pa.field("sheet", pa.utf8()), pa.field("symptom", pa.utf8()),
        pa.field("issue_id", pa.utf8()), pa.field("category", pa.utf8()),
        pa.field("problem", pa.utf8()), pa.field("root_cause", pa.utf8()),
        pa.field("fix", pa.utf8()), pa.field("severity", pa.utf8()),
        pa.field("present", pa.utf8()), pa.field("jira", pa.utf8()),
        pa.field("discovered", pa.utf8()), pa.field("resolved", pa.utf8()),
        pa.field("notes", pa.utf8()), pa.field("do_text", pa.utf8()),
        pa.field("dont_text", pa.utf8()), pa.field("rationale", pa.utf8()),
        pa.field("prerequisite", pa.utf8()), pa.field("how_to_verify", pa.utf8()),
        pa.field("learning", pa.utf8()), pa.field("action_taken", pa.utf8()),
    ])
    if "excel_issues" in _lancedb_conn.table_names(): _excel_table = _lancedb_conn.open_table("excel_issues")
    else:
        _excel_table = _lancedb_conn.create_table("excel_issues", schema=excel_schema)
        config._log_rag.info("[LanceDB] Created table: excel_issues")

    return _lancedb_conn, _docs_table, _excel_table

def init_db():
    _get_lancedb()
    _get_embedder()
