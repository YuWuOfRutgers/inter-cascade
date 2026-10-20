"""Benchmark preparation.

Each command produces two JSON files in the standard record format the rest of
the pipeline consumes::

    [{"question": "...", "ground_truth": "..."}, ...]

- ``calibration_<name>.json`` - held-out set for threshold calibration
- ``test_<name>.json``        - evaluation stream

Bring-your-own-data: any JSON file in this format works; multiple-choice
options should be embedded in the question text (see ``nasa_history`` below).
Downloading the built-in benchmarks requires the ``data`` extra:
``pip install inter-cascade[data]``.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional


def _write(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(records):>6} records -> {path}")


def _gsm_ground_truth(answer: str) -> Optional[str]:
    m = re.search(r"(?:####|###)\s*([^\n]+)\s*$", answer)
    return m.group(1).strip() if m else None


def prepare_gsm_symbolic(output_dir: str, calib_ratio: float = 0.1, seed: int = 42) -> None:
    """apple/GSM-Symbolic (main): 1250 calibration / 11250 test, as in the paper."""
    import datasets

    ds = datasets.load_dataset("apple/GSM-Symbolic", "main")
    splits = ds["test"].train_test_split(test_size=calib_ratio, seed=seed)

    def convert(split) -> List[dict]:
        out = []
        for item in split:
            gt = _gsm_ground_truth(item["answer"])
            if gt is None:
                continue
            out.append({"question": item["question"], "ground_truth": gt})
        random.seed(seed)
        random.shuffle(out)
        return out

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_gsm_symbolic.json"), convert(splits["test"]))
    _write(os.path.join(output_dir, "test_gsm_symbolic.json"), convert(splits["train"]))


def prepare_gsm8k(output_dir: str, calib_ratio: float = 0.1, seed: int = 42) -> None:
    """openai/gsm8k (main), calibration split carved from the test set."""
    import datasets

    ds = datasets.load_dataset("openai/gsm8k", "main")
    items = [
        {"question": item["question"], "ground_truth": gt}
        for item in ds["test"]
        if (gt := _gsm_ground_truth(item["answer"])) is not None
    ]
    random.seed(seed)
    random.shuffle(items)
    n_calib = int(len(items) * calib_ratio)

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_gsm8k.json"), items[:n_calib])
    _write(os.path.join(output_dir, "test_gsm8k.json"), items[n_calib:])


def prepare_nasa_history(output_dir: str, calibration_size: int = 1000, seed: int = 42) -> None:
    """patrickfleith/NASA-History-MCQ, options shuffled into the question text."""
    import datasets

    ds = datasets.load_dataset("patrickfleith/NASA-History-MCQ")
    full = list(ds["train"])
    random.seed(seed)

    def convert(items) -> List[dict]:
        out = []
        for item in items:
            options = [
                item["correct_answer"],
                item["incorrect_answer_1"],
                item["incorrect_answer_2"],
                item["incorrect_answer_3"],
            ]
            random.shuffle(options)
            letter = chr(65 + options.index(item["correct_answer"]))
            options_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
            out.append(
                {
                    "question": f"{item['question']}\n{options_str}",
                    "ground_truth": letter,
                }
            )
        return out

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_nasa_history.json"), convert(full[:calibration_size]))
    _write(os.path.join(output_dir, "test_nasa_history.json"), convert(full[calibration_size:]))


PREPARERS: Dict[str, callable] = {
    "gsm_symbolic": prepare_gsm_symbolic,
    "gsm8k": prepare_gsm8k,
    "nasa_history": prepare_nasa_history,
}


def prepare(name: str, output_dir: str, **kwargs) -> None:
    if name not in PREPARERS:
        raise ValueError(f"Unknown dataset '{name}'. Built-ins: {sorted(PREPARERS)}. "
                         "For other benchmarks, produce JSON in the standard record format.")
    PREPARERS[name](output_dir, **kwargs)
