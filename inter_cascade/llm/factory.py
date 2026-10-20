from __future__ import annotations

from inter_cascade.config import LLMConfig
from inter_cascade.llm.base import BaseLLMClient


def build_client(cfg: LLMConfig) -> BaseLLMClient:
    if cfg.provider == "openai":
        from inter_cascade.llm.openai_client import OpenAIClient

        return OpenAIClient(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
    if cfg.provider == "gemini":
        from inter_cascade.llm.gemini_client import GeminiClient

        return GeminiClient(
            model=cfg.model,
            api_key=cfg.api_key,
            project_id=cfg.project_id,
            location=cfg.location,
        )
    raise ValueError(f"Unknown provider '{cfg.provider}' (expected 'openai' or 'gemini')")
