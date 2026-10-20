"""The Inter-Cascade inference pipeline (Algorithm 2 of the paper).

For each query:

1. **Retrieve** the top-k most similar strategies from the repository and
   build the weak model's prompt (zero-shot when nothing relevant is stored).
2. **Weak decision** - the weak model answers; if the confidence of its
   [Answer] tokens passes the calibrated threshold, its answer is final.
3. **Strong decision** - otherwise the query is deferred to the strong model
   (zero-shot). If the strong model's confidence passes its own threshold, its
   answer is final and its [Strategy] is written back to the repository, where
   it will help the weak model on future similar queries. If not, the query is
   rejected (uncovered).

The standard cascade baseline (Jung et al.) and the paper's ablations are the
same loop with retrieval/write switched off - see ``RepositoryConfig``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from inter_cascade.confidence import MinMaxNormalizer, answer_logprob_sum, confidence
from inter_cascade.llm.base import BaseLLMClient
from inter_cascade.parsing import extract_strategy_and_answer
from inter_cascade.prompts import build_prompt
from inter_cascade.repository import StrategyDoc, StrategyRepository


@dataclass
class Deferral:
    """Confidence-thresholded accept/defer rule for one model.

    ``threshold=None`` means the rule is disabled and every answer is
    accepted (the paper runs the strong model this way on MetaMath and
    NASA-History).
    """

    threshold: Optional[float]
    normalizer: Optional[MinMaxNormalizer]

    def confidence_of(self, logprob_sum: Optional[float]) -> Optional[float]:
        if self.normalizer is None:
            return None
        return confidence(logprob_sum, self.normalizer)

    def accept(self, conf: Optional[float]) -> bool:
        if self.threshold is None:
            return True
        return conf is not None and conf > self.threshold


@dataclass
class ModelTrace:
    """Everything one model did for one query (kept for analysis/metrics)."""

    prompt: str
    response: str
    strategy: Optional[str]
    answer: Optional[str]
    logprob_sum: Optional[float]
    confidence: Optional[float]
    accepted: bool
    seconds: float


@dataclass
class CascadeResult:
    question: str
    answer: Optional[str]  # None when both models rejected the query
    source: str  # "weak" | "strong" | "rejected"
    retrieved: List[StrategyDoc] = field(default_factory=list)
    weak: Optional[ModelTrace] = None
    strong: Optional[ModelTrace] = None
    retrieve_seconds: float = 0.0


class InterCascade:
    """Two-model interactive cascade.

    Build one from a YAML config with :func:`inter_cascade.runner.build_cascade`,
    or construct directly for programmatic use::

        cascade = InterCascade(weak, strong, repo, weak_deferral, strong_deferral)
        result = cascade.answer("A robe takes 2 bolts of blue fiber and ...")
    """

    def __init__(
        self,
        weak: BaseLLMClient,
        strong: BaseLLMClient,
        repository: Optional[StrategyRepository],
        weak_deferral: Deferral,
        strong_deferral: Deferral,
        top_k: int = 2,
        similarity_threshold: Optional[float] = None,
        retrieval: str = "similarity",  # "similarity" | "random" | "none"
        include_strategy: bool = True,
        write_repository: bool = True,
        answer_type: str = "numeric",
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        if retrieval not in ("similarity", "random", "none"):
            raise ValueError("retrieval must be 'similarity', 'random', or 'none'")
        if retrieval != "none" and repository is None:
            raise ValueError(f"retrieval='{retrieval}' requires a repository")
        self.weak = weak
        self.strong = strong
        self.repository = repository
        self.weak_deferral = weak_deferral
        self.strong_deferral = strong_deferral
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.retrieval = retrieval
        self.include_strategy = include_strategy
        self.write_repository = write_repository
        self.answer_type = answer_type
        self.temperature = temperature
        self._rng = random.Random(seed)

    # ------------------------------------------------------------ pipeline

    def answer(self, question: str) -> CascadeResult:
        t0 = time.time()
        docs = self._retrieve(question)
        retrieve_seconds = time.time() - t0

        weak_trace = self._ask(self.weak, question, docs, self.weak_deferral)
        result = CascadeResult(
            question=question,
            answer=weak_trace.answer,
            source="weak",
            retrieved=docs,
            weak=weak_trace,
            retrieve_seconds=retrieve_seconds,
        )
        if weak_trace.accepted:
            return result

        # Deferred: the strong model sees the plain question, zero-shot.
        strong_trace = self._ask(self.strong, question, [], self.strong_deferral)
        result.strong = strong_trace
        if strong_trace.accepted:
            result.answer = strong_trace.answer
            result.source = "strong"
            if self.write_repository and self.repository is not None and strong_trace.strategy:
                self.repository.add(
                    question=question,
                    strategy=strong_trace.strategy,
                    answer=strong_trace.answer or "",
                )
        else:
            result.answer = None
            result.source = "rejected"
        return result

    # ------------------------------------------------------------- helpers

    def _retrieve(self, question: str) -> List[StrategyDoc]:
        if self.retrieval == "none" or self.repository is None:
            return []
        if self.retrieval == "random":
            return self.repository.retrieve_random(self.top_k, rng=self._rng)
        docs = self.repository.retrieve(question, self.top_k)
        if self.similarity_threshold is not None:
            docs = [d for d in docs if d.score is not None and d.score > self.similarity_threshold]
        return docs

    def _ask(
        self,
        client: BaseLLMClient,
        question: str,
        docs: List[StrategyDoc],
        deferral: Deferral,
    ) -> ModelTrace:
        prompt = build_prompt(
            question,
            docs,
            answer_type=self.answer_type,
            include_strategy=self.include_strategy,
        )
        t0 = time.time()
        response = client.ask(prompt, temperature=self.temperature, logprobs=True)
        seconds = time.time() - t0

        strategy, answer = extract_strategy_and_answer(response.text)
        logprob = answer_logprob_sum(response) if answer is not None else None
        conf = deferral.confidence_of(logprob)
        accepted = answer is not None and deferral.accept(conf)
        return ModelTrace(
            prompt=prompt,
            response=response.text,
            strategy=strategy,
            answer=answer,
            logprob_sum=logprob,
            confidence=conf,
            accepted=accepted,
            seconds=seconds,
        )
