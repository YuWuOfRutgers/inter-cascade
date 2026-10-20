"""Confidence scores from answer-token log-probabilities.

The confidence of an answer is ``exp(sum of the [Answer] tokens' logprobs)``,
min-max normalized with parameters fitted on the calibration set (so test-time
confidences are on the same scale the deferral threshold was calibrated on).

The original research code extracted answer tokens two different ways (a
tokenizer-specific tail count for GPT, a token-stream scan for Gemini); here a
single provider-agnostic method reconstructs the text from the returned tokens
and takes every token after the final ``[Answer]:`` marker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from inter_cascade.llm.base import LLMResponse

ANSWER_MARKER = "[Answer]:"


@dataclass
class MinMaxNormalizer:
    """Min-max scaling fitted on calibration confidences.

    Test values are transformed with the *calibration* min/range (matching
    ``minmax01`` / ``minmax01_test`` in the original code), so they can fall
    slightly outside [0, 1]; the threshold comparison does not care.
    """

    x_min: float
    range: float

    @classmethod
    def fit(cls, values: Sequence[float], eps: float = 1e-12) -> "MinMaxNormalizer":
        if not values:
            raise ValueError("Cannot fit a normalizer on an empty value list")
        x_min = min(values)
        rng = max(max(values) - x_min, eps)
        return cls(x_min=x_min, range=rng)

    def transform(self, value: float) -> float:
        return (value - self.x_min) / self.range

    def to_dict(self) -> dict:
        return {"x_min": self.x_min, "range": self.range}

    @classmethod
    def from_dict(cls, data: dict) -> "MinMaxNormalizer":
        return cls(x_min=data["x_min"], range=data["range"])


def answer_logprob_sum(response: LLMResponse) -> Optional[float]:
    """Sum of logprobs of the tokens that form the [Answer] section.

    Returns None when the response carries no token logprobs or the marker is
    not found.
    """
    if not response.tokens:
        return None

    pieces = [t.token for t in response.tokens]
    full = "".join(pieces)
    marker_pos = full.rfind(ANSWER_MARKER)
    if marker_pos == -1:
        return None
    answer_start = marker_pos + len(ANSWER_MARKER)

    # Map the character offset back to token indices.
    offsets: List[int] = []
    pos = 0
    for piece in pieces:
        offsets.append(pos)
        pos += len(piece)

    logprobs = [
        t.logprob
        for t, start in zip(response.tokens, offsets)
        if start + len(t.token) > answer_start and t.token.strip() != ""
    ]
    if not logprobs:
        return None
    return sum(logprobs)


def confidence(logprob_sum: Optional[float], normalizer: MinMaxNormalizer) -> Optional[float]:
    if logprob_sum is None:
        return None
    return normalizer.transform(math.exp(logprob_sum))
