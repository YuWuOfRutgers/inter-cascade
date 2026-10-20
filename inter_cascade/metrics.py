"""Evaluation metrics, matching Table "Evaluation Metrics" in the paper.

Notation: T = total queries, U = uncovered (rejected by both models),
C_w / C_s = correctly answered queries that were also accepted by the weak /
strong model, C_w_total = all queries the weak model answered correctly,
N_s = strong model invocations.
"""

from __future__ import annotations

from typing import Dict, List


def compute_metrics(records: List[dict]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {}

    uncovered = sum(1 for r in records if r.get("source") == "rejected")
    covered = total - uncovered

    weak_correct_total = sum(1 for r in records if r.get("weak_correct"))
    weak_correct_accepted = sum(
        1 for r in records if r.get("weak_accepted") and r.get("weak_correct")
    )
    strong_called = sum(1 for r in records if r.get("strong_called"))
    strong_correct_accepted = sum(
        1 for r in records if r.get("strong_accepted") and r.get("strong_correct")
    )
    helped = sum(1 for r in records if r.get("weak_helped_by_strategy"))

    def ratio(num: int, den: int) -> float:
        return num / den if den else float("nan")

    return {
        "total_queries": total,
        "uncovered_queries": uncovered,
        "coverage": ratio(covered, total),
        "pipeline_accuracy": ratio(weak_correct_accepted + strong_correct_accepted, covered),
        "weak_accuracy": ratio(weak_correct_total, covered),
        "weak_correct_accepted": ratio(weak_correct_accepted, covered),
        "strong_call_rate": ratio(strong_called, total),
        "weak_helped_by_strategy_rate": ratio(helped, total),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    labels = {
        "total_queries": "Total queries",
        "uncovered_queries": "Uncovered queries",
        "coverage": "Coverage",
        "pipeline_accuracy": "Pipeline accuracy",
        "weak_accuracy": "Weak LLM accuracy",
        "weak_correct_accepted": "Weak correct & accepted",
        "strong_call_rate": "Strong LLM call rate",
        "weak_helped_by_strategy_rate": "Queries with retrieved strategies",
    }
    lines = []
    for key, label in labels.items():
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float) and 0 <= value <= 1 and key not in ("total_queries", "uncovered_queries"):
            lines.append(f"{label:<34}{value * 100:6.2f}%")
        else:
            lines.append(f"{label:<34}{value}")
    return "\n".join(lines)
