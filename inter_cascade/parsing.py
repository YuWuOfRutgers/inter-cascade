"""Parsing model replies and grading answers."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from inter_cascade.llm.base import BaseLLMClient
from inter_cascade.prompts import JUDGE_PROMPT


def extract_strategy_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a ``[Strategy]: ... [Answer]: ...`` reply into its two sections.

    Returns ``(None, None)`` when the reply does not follow the format.
    """
    if not text:
        return None, None
    strategy_start = text.find("[Strategy]:")
    answer_start = text.rfind("[Answer]:")
    if strategy_start == -1 or answer_start == -1 or answer_start < strategy_start:
        return None, None
    strategy = text[strategy_start + len("[Strategy]:") : answer_start].strip()
    answer = text[answer_start + len("[Answer]:") :].strip()
    return strategy or None, answer or None


def _leading_option_letter(value: str) -> Optional[str]:
    m = re.match(r"^\s*([A-Z])(?:[.):\s]|$)", value)
    return m.group(1) if m else None


class AnswerChecker:
    """Grade a model answer against ground truth (evaluation only).

    Order of checks: exact string match, numeric equality, option-letter
    match, then (optionally) an LLM judge for formats a string comparison
    cannot resolve - e.g. "1/2" vs "50%". The judge is the strong model, per
    the paper's setup.
    """

    def __init__(self, judge_client: Optional[BaseLLMClient] = None):
        self.judge_client = judge_client

    def check(self, answer: Optional[str], ground_truth: str) -> bool:
        if not answer:
            return False
        answer, ground_truth = answer.strip(), str(ground_truth).strip()

        if answer == ground_truth:
            return True
        try:
            if float(answer) == float(ground_truth):
                return True
        except ValueError:
            pass
        letter, gt_letter = _leading_option_letter(answer), _leading_option_letter(ground_truth)
        if letter and gt_letter:
            return letter == gt_letter

        if self.judge_client is not None:
            prompt = JUDGE_PROMPT.format(llm_answer=answer, ground_truth=ground_truth)
            reply = self.judge_client.ask(prompt, logprobs=False)
            verdict = reply.text.strip().split()
            return bool(verdict) and verdict[-1].lower() == "true"
        return False
