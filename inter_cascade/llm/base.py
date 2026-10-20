"""Provider-agnostic LLM client interface.

Every provider returns the same ``LLMResponse`` shape: the generated text plus
per-token log-probabilities of the chosen tokens. Token logprobs are what the
cascade's confidence score is computed from, so ``logprobs=True`` is the
default everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenLogprob:
    token: str
    logprob: float


@dataclass
class LLMResponse:
    text: str
    tokens: List[TokenLogprob] = field(default_factory=list)


class BaseLLMClient(ABC):
    """Minimal chat interface: one prompt in, one response out."""

    model: str

    @abstractmethod
    def ask(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        logprobs: bool = True,
    ) -> LLMResponse:
        ...
