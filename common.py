"""
ArxivSage — shared configuration and small helpers.

Every notebook does `from common import *` so that paths, model names and
hyper-parameters live in ONE place. Change a value here, re-run the notebooks.
"""
import os
from datetime import date, timedelta
from pathlib import Path

# Optional machine-specific overrides: KEY=VALUE lines in a git-ignored `.env` next to this file.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _key, _value = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _value.strip())

# Use a single GPU (default GPU 0); must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PAPERS_PATH = DATA_DIR / "papers.jsonl"              # step 1 output: one paper per line
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"              # step 2 output: one chunk per line
EMBEDDINGS_PATH = DATA_DIR / "chunk_embeddings.npy"  # step 2 output: row i = vector of chunk i
HF_DATASET = "Sagnik2003/arxivsage-cv-nlp"            # prebuilt data on the Hugging Face Hub (download_data.py)
WEAVIATE_DIR = PROJECT_DIR / "weaviate_data"         # Weaviate's on-disk storage
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Data scope (step 1)
# ----------------------------------------------------------------------------
CATEGORIES = ["cs.CV", "cs.CL"]          # Computer Vision + Computation and Language (NLP)
YEARS_BACK = 2                           # drop to 1 if disk is tight
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365 * YEARS_BACK)
OAI_URL = "https://oaipmh.arxiv.org/oai"

# ----------------------------------------------------------------------------
# Mixed chunking (step 2)
# ----------------------------------------------------------------------------
CHUNK_WORDS = 120                        # max chunk size (~160 BERT tokens, well under the 512 limit)
OVERLAP_WORDS = int(0.10 * CHUNK_WORDS)  # 10% overlap = 12 words

# ----------------------------------------------------------------------------
# Models (all local)
# ----------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # bi-encoder, 1024-d
# BGE recommends this prefix for *queries* only (passages are embedded as-is).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-v2-m3" # cross-encoder used for re-ranking
LLM_MODEL = "Qwen/Qwen3.5-9B"            # Hugging Face, loaded in bf16 (~18 GB VRAM)
FAST_LLM_MODEL = "Qwen/Qwen3.5-2B"       # notebook 04: small model for HyDE / query rewriting / summaries (~4.5 GB)

# ----------------------------------------------------------------------------
# Weaviate (step 2 + 3)
# ----------------------------------------------------------------------------
COLLECTION = "ArxivChunk"
WEAVIATE_PORT = 8099                     # non-default ports: avoid clashing with others on the box
WEAVIATE_GRPC_PORT = 50099


def get_weaviate_client():
    """Start (or attach to) the embedded Weaviate instance.

    Embedded Weaviate runs as a child process of the Python kernel and keeps
    its data in WEAVIATE_DIR. If another notebook kernel already started it,
    starting a second one fails, so we just connect to the running one.
    """
    import weaviate

    try:
        return weaviate.connect_to_embedded(
            persistence_data_path=str(WEAVIATE_DIR),
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            environment_variables={"DISABLE_TELEMETRY": "true", "LOG_LEVEL": "error"},
        )
    except Exception:
        return weaviate.connect_to_local(port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT)
