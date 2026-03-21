"""LLM client — Task 8 implementation.

Sends prompts to Ollama's streaming HTTP API and records TTFT and generation time.
Retries up to 3 times with 1-second delay on HTTP error.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    text: str
    total_tokens: int
    ttft_ms: float
    generation_ms: float
    failed: bool
    failure_reason: str | None


def build_prompt(chunks: list[str], query: str) -> str:
    """Build the prompt string from retrieved chunks and query."""
    return f"Context:\n{chr(10).join(chunks)}\n\nQuestion: {query}\nAnswer:"


def _parse_token_counts(final_obj: dict) -> int:
    """Return total_tokens = prompt_eval_count + eval_count from the final streaming object."""
    prompt_tokens = final_obj.get("prompt_eval_count", 0) or 0
    completion_tokens = final_obj.get("eval_count", 0) or 0
    return prompt_tokens + completion_tokens


def generate(
    query: str,
    chunks: list[str],
    llm_host: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
    max_retries: int = 3,
) -> LLMResponse:
    """Send prompt to Ollama and return LLMResponse with timing metrics."""
    model = os.environ.get("BENCHMARK_MODEL", model)
    prompt = build_prompt(chunks, query)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 256},
    }

    last_error: str | None = None

    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(1.0)

        try:
            text_parts: list[str] = []
            ttft_ms: float = 0.0
            total_tokens: int = 0
            first_token = True

            start = time.perf_counter()

            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST",
                    f"{llm_host}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue
                        obj = json.loads(line)
                        token = obj.get("response", "")
                        if token and first_token:
                            ttft_ms = (time.perf_counter() - start) * 1000.0
                            first_token = False
                        text_parts.append(token)
                        if obj.get("done", False):
                            total_tokens = _parse_token_counts(obj)
                            break

            generation_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="".join(text_parts),
                total_tokens=total_tokens,
                ttft_ms=ttft_ms,
                generation_ms=generation_ms,
                failed=False,
                failure_reason=None,
            )

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_error = str(exc)

    return LLMResponse(
        text="",
        total_tokens=0,
        ttft_ms=0.0,
        generation_ms=0.0,
        failed=True,
        failure_reason=f"Failed after {max_retries} retries: {last_error}",
    )
