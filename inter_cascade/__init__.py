"""Inter-Cascade: an online, interactive LLM cascade.

When the strong model resolves a query the weak model deferred, it also
produces a generalized problem-solving *strategy*. Strategies are stored in a
similarity-searchable repository and injected into the weak model's context on
future queries, so the weak model improves on the job without fine-tuning.

Reference: "Inter-Cascade: Interactive LLM Cascades with Strategy Repositories".
"""

from inter_cascade.cascade import CascadeResult, Deferral, InterCascade
from inter_cascade.config import InterCascadeConfig, load_config
from inter_cascade.confidence import MinMaxNormalizer
from inter_cascade.repository import StrategyRepository

__version__ = "0.1.0"

__all__ = [
    "InterCascade",
    "CascadeResult",
    "Deferral",
    "InterCascadeConfig",
    "load_config",
    "MinMaxNormalizer",
    "StrategyRepository",
]
