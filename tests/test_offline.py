"""Offline unit tests - no network, no API keys, no model downloads.

Run with: python -m pytest tests/  (or python tests/test_offline.py)
"""

import math
import sys
import unittest

from inter_cascade.calibration import calibrate_threshold, coverage
from inter_cascade.confidence import MinMaxNormalizer, answer_logprob_sum
from inter_cascade.llm.base import LLMResponse, TokenLogprob
from inter_cascade.metrics import compute_metrics
from inter_cascade.parsing import AnswerChecker, extract_strategy_and_answer
from inter_cascade.prompts import build_prompt
from inter_cascade.repository import StrategyDoc


class TestParsing(unittest.TestCase):
    def test_extracts_both_sections(self):
        text = "[Strategy]: Isolate x by subtracting y.\n[Answer]: 6"
        strategy, answer = extract_strategy_and_answer(text)
        self.assertEqual(strategy, "Isolate x by subtracting y.")
        self.assertEqual(answer, "6")

    def test_malformed_returns_none(self):
        self.assertEqual(extract_strategy_and_answer("The answer is 6."), (None, None))
        self.assertEqual(extract_strategy_and_answer(""), (None, None))

    def test_checker_numeric_and_letter(self):
        checker = AnswerChecker(judge_client=None)
        self.assertTrue(checker.check("6", "6.0"))
        self.assertTrue(checker.check("B. True", "B"))
        self.assertFalse(checker.check("7", "6"))
        self.assertFalse(checker.check(None, "6"))


class TestConfidence(unittest.TestCase):
    def _response(self, pieces):
        return LLMResponse(
            text="".join(p for p, _ in pieces),
            tokens=[TokenLogprob(p, lp) for p, lp in pieces],
        )

    def test_answer_logprob_sum_after_marker(self):
        response = self._response(
            [("[Strategy]:", -0.5), (" reason", -0.1), ("\n[Answer]:", -0.2), (" ", -0.9), ("6", -0.3)]
        )
        # only the "6" token counts (whitespace-only tokens are skipped)
        self.assertAlmostEqual(answer_logprob_sum(response), -0.3)

    def test_no_marker_returns_none(self):
        response = self._response([("hello", -0.1)])
        self.assertIsNone(answer_logprob_sum(response))

    def test_normalizer_roundtrip(self):
        norm = MinMaxNormalizer.fit([0.2, 0.4, 0.8])
        self.assertAlmostEqual(norm.transform(0.2), 0.0)
        self.assertAlmostEqual(norm.transform(0.8), 1.0)
        restored = MinMaxNormalizer.from_dict(norm.to_dict())
        self.assertAlmostEqual(restored.transform(0.4), norm.transform(0.4))


class TestCalibration(unittest.TestCase):
    def test_threshold_separates_correct_from_incorrect(self):
        # High-confidence answers correct, low-confidence ones wrong. The
        # procedure returns the first grid point where the bound fails, i.e.
        # right at the top of the wrong cluster.
        confidences = [0.9] * 50 + [0.1] * 50
        labels = [1] * 50 + [0] * 50
        thr = calibrate_threshold(confidences, labels, alpha=0.2, delta=0.5)
        self.assertGreater(thr, 0.095)  # grid values carry float error
        self.assertLessEqual(thr, 0.9)
        # One grid step above the returned threshold, only correct answers remain.
        accepted = [l for c, l in zip(confidences, labels) if c >= thr + 0.001]
        self.assertTrue(accepted and all(accepted))

    def test_mixed_top_region_still_calibrates(self):
        # A wrong answer at the very top must not make calibration infeasible
        # (the paper's overconfident-weak-model case).
        confidences = [1.0] + [0.9] * 60 + [0.2] * 39
        labels = [0] + [1] * 60 + [0] * 39
        thr = calibrate_threshold(confidences, labels, alpha=0.3, delta=0.5)
        self.assertLess(thr, 0.9)
        self.assertGreater(thr, 0.15)

    def test_all_wrong_raises(self):
        with self.assertRaises(ValueError):
            calibrate_threshold([0.9] * 10, [0] * 10, alpha=0.05, delta=0.05)


class TestPrompts(unittest.TestCase):
    def test_zero_shot_and_context(self):
        zero = build_prompt("What is 2+2?", [], answer_type="numeric")
        self.assertIn("[Question]: What is 2+2?", zero)
        docs = [StrategyDoc(question="1+1?", strategy="Add.", answer="2")]
        ctx = build_prompt("What is 2+2?", docs, answer_type="numeric")
        self.assertIn("[Strategy]: Add.", ctx)
        no_strat = build_prompt("What is 2+2?", docs, answer_type="numeric", include_strategy=False)
        self.assertNotIn("[Strategy]: Add.", no_strat)


class TestMetrics(unittest.TestCase):
    def test_paper_metrics(self):
        records = [
            # weak accepted & correct
            {"source": "weak", "weak_correct": True, "weak_accepted": True,
             "strong_called": False, "weak_helped_by_strategy": True},
            # deferred; strong accepted & correct
            {"source": "strong", "weak_correct": False, "weak_accepted": False,
             "strong_called": True, "strong_correct": True, "strong_accepted": True},
            # rejected by both
            {"source": "rejected", "weak_correct": False, "weak_accepted": False,
             "strong_called": True, "strong_correct": False, "strong_accepted": False},
            # weak correct but deferred anyway; strong accepted & wrong
            {"source": "strong", "weak_correct": True, "weak_accepted": False,
             "strong_called": True, "strong_correct": False, "strong_accepted": True},
        ]
        m = compute_metrics(records)
        self.assertEqual(m["total_queries"], 4)
        self.assertEqual(m["uncovered_queries"], 1)
        self.assertAlmostEqual(m["coverage"], 3 / 4)
        self.assertAlmostEqual(m["pipeline_accuracy"], 2 / 3)
        self.assertAlmostEqual(m["weak_accuracy"], 2 / 3)
        self.assertAlmostEqual(m["strong_call_rate"], 3 / 4)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1], verbosity=2)
