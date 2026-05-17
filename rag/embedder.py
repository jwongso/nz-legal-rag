"""
Embedding via sentence-transformers (nomic-embed-text-v1.5).

Runs directly in-process on CPU - no Ollama server required.
Model is downloaded from HuggingFace on first use (~274MB, cached after that).

# TODO: Once AI Max+ 395 is available, use GPU-accelerated encoding via
#        model.encode(..., device='cuda') for 10-20x batch throughput.
#
# TODO: If Ollama becomes stable on this machine, switching back to the
#        /api/embed batch endpoint is an option. Keep this as primary since
#        it removes a runtime dependency.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
_PROMPT = "search_document: "  # nomic-embed-text requires task prefix


class Embedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True, device="cpu")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_PROMPT + t for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    async def embed(self, text: str) -> list[float]:
        return self._encode([text])[0]

    async def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            results.extend(self._encode(texts[i : i + batch_size]))
        return results

    async def close(self) -> None:
        pass
