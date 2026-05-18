import os
from pathlib import Path

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nz_legal")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "120"))   # words per chunk (~480 chars)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "15"))  # words
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "20"))  # discard fragments shorter than this

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

NZLII_BASE = "https://www.nzlii.org"

COURTS = {
    "NZTT": "NZ Tenancy Tribunal",
    "NZHC": "NZ High Court",
    "NZCA": "NZ Court of Appeal",
    "NZSC": "NZ Supreme Court",
    "NZEmpC": "NZ Employment Court",
    "NZERA": "NZ Employment Relations Authority",
    "NZFC": "NZ Family Court",
    "NZEnvC": "NZ Environment Court",
    "NZACC": "NZ ACC Appeals",
    "NZCorC": "NZ Coroners Court",
    "NZLCDT": "NZ Lawyers and Conveyancers Disciplinary Tribunal",
    "NZHRRT": "NZ Human Rights Review Tribunal",
    "NZREADT": "NZ Real Estate Agents Disciplinary Tribunal",
    "NZLEG": "NZ Legislation",
}
