"""Client for Google Gemini, via API key or Vertex AI.

Vertex AI mode (``project_id`` set) requires the ``gcloud`` application
default credentials::

    gcloud auth application-default login
    gcloud config set project <your-project-id>

Requires the ``gemini`` extra: ``pip install inter-cascade[gemini]``.
"""

from __future__ import annotations

from typing import Optional

from inter_cascade.llm.base import BaseLLMClient, LLMResponse, TokenLogprob


class GeminiClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "global",
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "provider 'gemini' requires the google-genai package: "
                "pip install 'inter-cascade[gemini]'"
            ) from e

        self.model = model
        if project_id:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(
                "Gemini needs either `project_id` (Vertex AI) in the config or "
                "an API key in the environment variable named by `api_key_env` "
                "(default GEMINI_API_KEY)."
            )

    def ask(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        logprobs: bool = True,
    ) -> LLMResponse:
        from google.genai.types import GenerateContentConfig

        config = GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_logprobs=logprobs,
            logprobs=1 if logprobs else None,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )

        tokens = []
        candidate = response.candidates[0]
        if logprobs and candidate.logprobs_result:
            tokens = [
                TokenLogprob(c.token, c.log_probability)
                for c in candidate.logprobs_result.chosen_candidates
            ]
        return LLMResponse(text=response.text or "", tokens=tokens)
