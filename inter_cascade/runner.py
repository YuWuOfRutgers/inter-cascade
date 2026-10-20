"""Wiring between the YAML config and the pipeline: calibration runs,
cascade construction, and streaming evaluation.

Calibration artifacts live in ``<artifacts_dir>/``:

- ``calibration_records_weak.json`` / ``..._strong.json`` - one record per
  calibration query (answer, correctness, answer-token logprob sum)
- ``calibration.json`` - per-model normalizer + calibrated threshold

``run_eval`` writes one result record per test query, saving after every item
so interrupted runs resume with ``--resume``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

from inter_cascade.calibration import calibrate_threshold, coverage, scan_alpha_delta
from inter_cascade.cascade import Deferral, InterCascade
from inter_cascade.config import DeferralConfig, InterCascadeConfig
from inter_cascade.confidence import MinMaxNormalizer, answer_logprob_sum
from inter_cascade.llm import BaseLLMClient, build_client
from inter_cascade.metrics import compute_metrics, format_metrics
from inter_cascade.parsing import AnswerChecker, extract_strategy_and_answer
from inter_cascade.prompts import build_prompt
from inter_cascade.repository import StrategyDoc, StrategyRepository


def _read_json(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------- calibration


def run_calibration(
    cfg: InterCascadeConfig,
    role: str,
    limit: Optional[int] = None,
    scan: bool = False,
) -> dict:
    """Run one model over the calibration set and calibrate its threshold.

    ``role`` is "weak" or "strong". Answers are graded (exact/numeric match,
    falling back to the strong model as judge) and each answer's token-logprob
    sum is recorded; the min-max normalizer and the Clopper-Pearson threshold
    are then fitted on these records.
    """
    if role not in ("weak", "strong"):
        raise ValueError("role must be 'weak' or 'strong'")
    if not cfg.calibration_file:
        raise ValueError("config needs `calibration_file` to calibrate")

    data = _read_json(cfg.calibration_file)
    if limit:
        data = data[:limit]

    client = build_client(cfg.weak if role == "weak" else cfg.strong)
    llm_cfg = cfg.weak if role == "weak" else cfg.strong
    judge = _build_judge(cfg)
    checker = AnswerChecker(judge_client=judge)

    records_path = cfg.artifacts / f"calibration_records_{role}.json"
    records: List[dict] = _read_json(records_path) if records_path.exists() else []
    done = {r["question"] for r in records}

    for i, item in enumerate(data):
        if item["question"] in done:
            continue
        prompt = build_prompt(item["question"], [], answer_type=cfg.task.answer_type)
        response = client.ask(prompt, temperature=llm_cfg.temperature, logprobs=True)
        strategy, answer = extract_strategy_and_answer(response.text)
        correct = checker.check(answer, item["ground_truth"])
        logprob = answer_logprob_sum(response) if answer is not None else None
        records.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": answer,
                "strategy": strategy,
                "correct": correct,
                "logprob_sum": logprob,
            }
        )
        _write_json(records_path, records)
        _log(f"[calibrate:{role}] {len(records)}/{len(data)} answered={answer!r} correct={correct}")

    usable = [r for r in records if r["logprob_sum"] is not None]
    if not usable:
        raise RuntimeError("No calibration record has a parseable answer with logprobs")
    confidences_raw = [math.exp(r["logprob_sum"]) for r in usable]
    labels = [int(r["correct"]) for r in usable]
    normalizer = MinMaxNormalizer.fit(confidences_raw)
    confidences = [normalizer.transform(c) for c in confidences_raw]

    entry = {
        "model": client.model,
        "normalizer": normalizer.to_dict(),
        "accuracy": sum(labels) / len(labels),
        "n_records": len(records),
    }

    dcfg = cfg.weak_deferral if role == "weak" else cfg.strong_deferral
    if dcfg.threshold is not None:
        entry.update({"threshold": dcfg.threshold, "source": "config"})
    elif dcfg.alpha is not None and dcfg.delta is not None:
        thr = calibrate_threshold(confidences, labels, dcfg.alpha, dcfg.delta)
        entry.update(
            {
                "threshold": thr,
                "alpha": dcfg.alpha,
                "delta": dcfg.delta,
                "coverage": coverage(confidences, thr),
                "source": "calibrated",
            }
        )
    else:
        entry.update({"threshold": None, "source": "disabled (always accept)"})

    if scan:
        entry["alpha_delta_scan"] = scan_alpha_delta(confidences, labels)

    calib_path = cfg.artifacts / "calibration.json"
    calib = _read_json(calib_path) if calib_path.exists() else {}
    calib[role] = entry
    _write_json(calib_path, calib)
    _log(
        f"[calibrate:{role}] accuracy={entry['accuracy']:.4f} "
        f"threshold={entry['threshold']} -> {calib_path}"
    )
    return entry


# ------------------------------------------------------------- construction


def _build_judge(cfg: InterCascadeConfig) -> Optional[BaseLLMClient]:
    return build_client(cfg.strong) if cfg.task.judge == "strong" else None


def _load_calibration(cfg: InterCascadeConfig) -> dict:
    calib_path = cfg.artifacts / "calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(
            f"{calib_path} not found - run `inter-cascade calibrate --config <cfg>` "
            "first (or set explicit `threshold` values plus calibration records)."
        )
    return _read_json(calib_path)


def _deferral_from(entry: Optional[dict], dcfg: DeferralConfig) -> Deferral:
    if entry is None:
        if dcfg.enabled:
            raise ValueError(
                "Deferral is configured (alpha/threshold set) but no calibration "
                "entry exists for this model; run `inter-cascade calibrate`."
            )
        return Deferral(threshold=None, normalizer=None)
    normalizer = MinMaxNormalizer.from_dict(entry["normalizer"]) if entry.get("normalizer") else None
    threshold = dcfg.threshold if dcfg.threshold is not None else entry.get("threshold")
    if threshold is not None and normalizer is None:
        raise ValueError("A threshold requires a fitted normalizer (re-run calibrate)")
    return Deferral(threshold=threshold, normalizer=normalizer)


def build_cascade(cfg: InterCascadeConfig, seed: Optional[int] = None) -> InterCascade:
    """Assemble an :class:`InterCascade` from config + calibration artifacts."""
    calib = _load_calibration(cfg)
    weak_deferral = _deferral_from(calib.get("weak"), cfg.weak_deferral)
    strong_deferral = _deferral_from(calib.get("strong"), cfg.strong_deferral)

    repository = None
    if cfg.repository.retrieval != "none" or cfg.repository.write:
        repository = StrategyRepository(
            dir=cfg.repository.dir,
            embedding_model=cfg.repository.embedding_model,
        )
        if cfg.repository.seed_from_calibration and len(repository) == 0:
            _seed_repository(cfg, repository)

    return InterCascade(
        weak=build_client(cfg.weak),
        strong=build_client(cfg.strong),
        repository=repository,
        weak_deferral=weak_deferral,
        strong_deferral=strong_deferral,
        top_k=cfg.repository.top_k,
        similarity_threshold=cfg.repository.similarity_threshold,
        retrieval=cfg.repository.retrieval,
        include_strategy=cfg.repository.include_strategy,
        write_repository=cfg.repository.write,
        answer_type=cfg.task.answer_type,
        temperature=cfg.weak.temperature,
        seed=seed,
    )


def _seed_repository(cfg: InterCascadeConfig, repository: StrategyRepository) -> None:
    """Warm-start the repository from the strong model's calibration run
    (the paper's default; skipped in the cold-start experiments)."""
    records_path = cfg.artifacts / "calibration_records_strong.json"
    if not records_path.exists():
        _log(f"[repository] no {records_path}; starting cold")
        return
    docs = [
        StrategyDoc(question=r["question"], strategy=r["strategy"], answer=r["answer"])
        for r in _read_json(records_path)
        if r.get("strategy") and r.get("answer")
    ]
    added = repository.add_many(docs)
    _log(f"[repository] seeded {added} strategies from the strong calibration run")


# ---------------------------------------------------------------- evaluation


def run_eval(
    cfg: InterCascadeConfig,
    input_file: str,
    output_file: str,
    limit: Optional[int] = None,
    resume: bool = False,
    seed: Optional[int] = None,
) -> List[dict]:
    """Stream the test set through the cascade, grading each final answer.

    The repository grows during the run exactly as it would in deployment;
    results are flushed to ``output_file`` after every query.
    """
    data = _read_json(input_file)
    if limit:
        data = data[:limit]

    cascade = build_cascade(cfg, seed=seed)
    checker = AnswerChecker(judge_client=_build_judge(cfg))

    records: List[dict] = []
    if resume and Path(output_file).exists():
        records = _read_json(output_file)
        _log(f"[run] resuming after {len(records)} completed queries")

    for item in data[len(records) :]:
        result = cascade.answer(item["question"])
        weak, strong = result.weak, result.strong
        record = {
            "question": item["question"],
            "ground_truth": item.get("ground_truth"),
            "source": result.source,
            "final_answer": result.answer,
            "weak_helped_by_strategy": bool(result.retrieved),
            "retrieved_scores": [d.score for d in result.retrieved],
            "weak_answer": weak.answer if weak else None,
            "weak_confidence": weak.confidence if weak else None,
            "weak_accepted": weak.accepted if weak else False,
            "weak_seconds": weak.seconds if weak else None,
            "strong_called": strong is not None,
            "strong_answer": strong.answer if strong else None,
            "strong_confidence": strong.confidence if strong else None,
            "strong_accepted": strong.accepted if strong else False,
            "strong_seconds": strong.seconds if strong else None,
            "retrieve_seconds": result.retrieve_seconds,
        }
        if item.get("ground_truth") is not None:
            record["weak_correct"] = (
                checker.check(weak.answer, item["ground_truth"]) if weak else False
            )
            record["strong_correct"] = (
                checker.check(strong.answer, item["ground_truth"]) if strong else False
            )
            record["final_correct"] = (
                record["weak_correct"] if result.source == "weak"
                else record["strong_correct"] if result.source == "strong"
                else False
            )
        records.append(record)
        _write_json(output_file, records)
        _log(
            f"[run] {len(records)}/{len(data)} source={result.source} "
            f"correct={record.get('final_correct')} "
            f"repo_size={len(cascade.repository) if cascade.repository else 0}"
        )

    metrics = compute_metrics(records)
    _log("\n" + format_metrics(metrics))
    return records
