"""Client for OpenAI and OpenAI-compatible endpoints (vLLM, Ollama, ...).

Self-hosted open-source models work by pointing ``base_url`` at the server,
e.g. ``http://localhost:8000/v1`` for vLLM or ``http://localhost:11434/v1``
for Ollama.
"""

from __future__ import annotations

from typing import Optional

from inter_cascade.llm.base import BaseLLMClient, LLMResponse, TokenLogprob


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI  # lazy: keep import cost out of --help paths

        if not api_key and not base_url:
            raise ValueError(
                "No API key found. Set the environment variable named by "
                "`api_key_env` in your config (default OPENAI_API_KEY), or set "
                "`base_url` for a local OpenAI-compatible server."
            )
        self.model = model
        # Local servers usually accept any non-empty key.
        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    def ask(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        logprobs: bool = True,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 1

        choice = self.client.chat.completions.create(**kwargs).choices[0]

        tokens = []
        if logprobs and choice.logprobs and choice.logprobs.content:
            tokens = [TokenLogprob(t.token, t.logprob) for t in choice.logprobs.content]
        return LLMResponse(text=choice.message.content or "", tokens=tokens)
