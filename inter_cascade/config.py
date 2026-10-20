"""YAML-backed configuration for the Inter-Cascade pipeline.

A single YAML file describes the two LLMs, the deferral rules, the strategy
repository, and the task. See ``configs/example.yaml`` for a commented
template. API keys are never stored in the config; each LLM section names an
environment variable (``api_key_env``) that holds the key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class LLMConfig:
    """One model endpoint (weak or strong).

    provider:
        ``openai``  - OpenAI or any OpenAI-compatible server (vLLM, Ollama,
        LM Studio, llama.cpp, ...). Point ``base_url`` at the server for
        self-hosted open-source models.
        ``gemini``  - Google Gemini via API key or Vertex AI (``project_id``).
    """

    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    project_id: Optional[str] = None  # Vertex AI mode for provider=gemini
    location: str = "global"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env) if self.api_key_env else None


@dataclass
class DeferralConfig:
    """Confidence-thresholded deferral rule for one model.

    ``alpha``/``delta`` are the risk tolerance and error level of the
    Clopper-Pearson calibration (Jung et al., 2024); ``inter-cascade
    calibrate`` derives ``threshold`` from them on the calibration set.
    Setting ``threshold`` explicitly skips calibration for this model.
    Setting ``threshold: null`` together with ``alpha: null`` disables the
    rule (every answer is accepted - the paper's MetaMath/NASA strong-model
    setting).
    """

    alpha: Optional[float] = None
    delta: Optional[float] = None
    threshold: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.threshold is not None or self.alpha is not None


@dataclass
class RepositoryConfig:
    """The strategy repository and how the weak model uses it.

    retrieval:
        ``similarity`` - top-k nearest questions (the Inter-Cascade default)
        ``random``     - random k strategies (paper ablation)
        ``none``       - no retrieval; the pipeline degenerates to a standard
        confidence-thresholded cascade (the Jung et al. baseline).
    """

    dir: str = "strategy_repo"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 2
    similarity_threshold: Optional[float] = None  # keep docs with score > this
    retrieval: str = "similarity"
    include_strategy: bool = True  # False = "No strategy" ablation
    write: bool = True  # add accepted strong answers back to the repository
    seed_from_calibration: bool = True  # warm-start from strong calibration run


@dataclass
class TaskConfig:
    """What kind of answers the task expects and how to grade them.

    answer_type:
        ``numeric`` / ``multiple_choice`` / ``freeform`` - controls the answer
        formatting instructions in the prompt and the matching rule.
    judge:
        ``strong`` - when a string/numeric comparison is inconclusive, ask the
        strong model to grade equivalence (used only during evaluation, where
        ground truth is available). ``none`` disables LLM grading.
    """

    answer_type: str = "numeric"
    judge: str = "strong"


@dataclass
class InterCascadeConfig:
    weak: LLMConfig = field(default_factory=LLMConfig)
    strong: LLMConfig = field(default_factory=LLMConfig)
    weak_deferral: DeferralConfig = field(default_factory=DeferralConfig)
    strong_deferral: DeferralConfig = field(default_factory=DeferralConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    calibration_file: Optional[str] = None  # JSON list of {question, ground_truth}
    artifacts_dir: str = "artifacts"  # calibration outputs (thresholds, records)

    @property
    def artifacts(self) -> Path:
        return Path(self.artifacts_dir)


def _build(cls, data: Dict[str, Any]):
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} for {cls.__name__}; expected {sorted(known)}")
    return cls(**data)


def load_config(path: str) -> InterCascadeConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sections = {
        "weak": LLMConfig,
        "strong": LLMConfig,
        "weak_deferral": DeferralConfig,
        "strong_deferral": DeferralConfig,
        "repository": RepositoryConfig,
        "task": TaskConfig,
    }
    kwargs: Dict[str, Any] = {}
    for key, cls in sections.items():
        if key in raw:
            kwargs[key] = _build(cls, raw.pop(key) or {})
    kwargs.update(raw)  # scalar top-level keys: calibration_file, artifacts_dir
    return _build(InterCascadeConfig, kwargs)
