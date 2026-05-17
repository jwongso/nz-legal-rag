"""Embedding via Ollama nomic-embed-text (runs locally, no API key needed)."""

# TODO: Once AI Max+ 395 is available, run a dedicated embedding server on Node 3
#       and switch to a larger model (e.g. nomic-embed-text-v2 or mxbai-embed-large).
#       EMBED_DIM will need updating in config.py if the model changes.
#
# TODO: Once a Blackwell card is affordable, switch to a GPU-accelerated embedding
#       server (vLLM or TGI) and batch requests at 512+ for ingest throughput.

import httpx

import config


class Embedder:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=config.OLLAMA_URL, timeout=30)

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            "/api/embeddings",
            json={"model": config.EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    async def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(await self.embed(text))
        return results

    async def close(self) -> None:
        await self._client.aclose()
