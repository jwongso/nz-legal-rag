"""
LLM generation via OpenAI-compatible API (llama.cpp server, Ollama, vLLM, LM Studio).
No cloud dependency - points at localhost by default.

Current: Qwen3.6-35B-A3B MoE on single host GPU (14 GPU layers, 4096 ctx).

# TODO: Once AI Max+ 395 Node 1 is available (128GB unified memory), run the full
#       70B model (e.g. Qwen3-72B Q4_K_M) with all layers on GPU. Update
#       LLM_MODEL and GPU_LAYERS in llama-server.env accordingly.
#
# TODO: Once a fine-tuned NZ legal adapter (LoRA) is trained on Node 2, merge it
#       into the base GGUF and deploy on Node 1. Expected improvement: better
#       citation format, fewer hallucinated NZ case names.
#
# TODO: Once Blackwell is affordable, evaluate vLLM with speculative decoding for
#       sub-second time-to-first-token on the legal Q&A use case.
"""

import json
from typing import AsyncIterator

import httpx

import config

_SYSTEM_PROMPT = """You are a legal research assistant specialising in New Zealand law.

Rules:
- Answer only from the provided context. Do not invent cases, statutes, or dates.
- Always cite the source document for each claim (case name and year, or Act and section).
- If the context does not contain enough information to answer, say so clearly.
- Use plain English. Avoid legal jargon unless quoting directly from a source.
- Do not give legal advice. Remind the user to consult a qualified NZ lawyer for their specific situation.
"""


class Generator:
    def __init__(self, system_prompt: str | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=config.LLM_BASE_URL,
            timeout=120,
        )
        self._system_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT

    async def generate_stream(
        self, question: str, context_chunks: list[str], sources: list[dict],
        thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Yield raw token strings from the LLM as they arrive."""
        truncated = [c[:1500] for c in context_chunks]
        context_block = "\n\n---\n\n".join(
            f"[S{i + 1}] {chunk}" for i, chunk in enumerate(truncated)
        )
        source_header = "\n".join(
            f"  [S{i + 1}] {s.get('title', 'Unknown')} | {s.get('court_name', '')} | "
            f"{s.get('date', '')} | {s.get('url', '')}"
            for i, s in enumerate(sources)
        )
        user_message = (
            f"Source index:\n{source_header}\n\n"
            f"Context documents (numbered to match source index):\n\n{context_block}\n\n"
            f"---\n\nQuestion: {question}\n\n"
            f"Answer using only the context above. Cite sources with [SN] notation "
            f"(e.g. [S1], [S2]) matching the source index. Do not invent other citation formats."
        )
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "stream": True,
        }
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (KeyError, json.JSONDecodeError):
                    continue

    async def generate(self, question: str, context_chunks: list[str], sources: list[dict]) -> str:
        truncated = [c[:1500] for c in context_chunks]
        context_block = "\n\n---\n\n".join(
            f"[S{i + 1}] {chunk}" for i, chunk in enumerate(truncated)
        )
        source_list = "\n".join(
            f"  [S{i + 1}] {s.get('title', 'Unknown')} ({s.get('court_name', '')}, {s.get('date', '')})"
            f" - {s.get('url', '')}"
            for i, s in enumerate(sources)
        )

        source_header = "\n".join(
            f"  [S{i + 1}] {s.get('title', 'Unknown')} | {s.get('court_name', '')} | "
            f"{s.get('date', '')} | {s.get('url', '')}"
            for i, s in enumerate(sources)
        )

        user_message = (
            f"Source index:\n{source_header}\n\n"
            f"Context documents (numbered to match source index):\n\n{context_block}\n\n"
            f"---\n\nQuestion: {question}\n\n"
            f"Answer using only the context above. Cite sources with [SN] notation "
            f"(e.g. [S1], [S2]) matching the source index. Do not invent other citation formats."
        )

        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code == 500:
            # Retry once - 500 on first call is common when the model is still loading
            import asyncio
            await asyncio.sleep(2)
            resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()

        # Always append the authoritative source list - overrides any LLM-generated one
        if source_list:
            answer += f"\n\nSources:\n{source_list}"

        return answer

    async def close(self) -> None:
        await self._client.aclose()
