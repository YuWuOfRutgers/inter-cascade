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


def prepare_gsm_plus(output_dir: str, calib_ratio: float = 0.1, seed: int = 42) -> None:
    """qintongli/GSM-Plus: calibration stratified by perturbation type, as in the paper."""
    import datasets
    from collections import defaultdict

    ds = datasets.load_dataset("qintongli/GSM-Plus")
    groups = defaultdict(list)
    for item in ds["test"]:
        groups[item["perturbation_type"]].append(item)

    random.seed(seed)
    calibration, test = [], []
    for items in groups.values():
        random.shuffle(items)
        n_calib = max(1, int(len(items) * calib_ratio))
        calibration.extend(items[:n_calib])
        test.extend(items[n_calib:])

    def convert(items) -> List[dict]:
        return [
            {
                "question": item["question"],
                "ground_truth": item["answer"],
                "perturbation_type": item["perturbation_type"],
            }
            for item in items
        ]

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_gsm_plus.json"), convert(calibration))
    _write(os.path.join(output_dir, "test_gsm_plus.json"), convert(test))


def prepare_metamath(
    output_dir: str,
    calib_ratio: float = 0.2,
    seed: int = 42,
    calib_size: int = 1000,
    test_size: int = 20000,
) -> None:
    """fxmeng/MetaMath-GSM240K, capped at 1000 calibration / 20000 test (the paper's
    "MetaMath (20K)"). Each stream interleaves the rephrased MetaMath queries with the
    matching original GSM8K question (once per original), exactly as in the paper runs."""
    import datasets

    def metamath_ground_truth(response: str) -> Optional[str]:
        m = re.search(r"The answer is:\s*([^\n]+)", response, re.IGNORECASE)
        return m.group(1).strip() if m else None

    mm = datasets.load_dataset("fxmeng/MetaMath-GSM240K")["train"]
    gsm = datasets.load_dataset("openai/gsm8k", "main")["train"]
    gsm_q2a = {ex["question"]: _gsm_ground_truth(ex["answer"]) for ex in gsm}

    splits = mm.train_test_split(test_size=calib_ratio, seed=seed)
    seen_original: set = set()

    def convert(split) -> List[dict]:
        out = []
        for item in split:
            gt = metamath_ground_truth(item["response"])
            if gt is None:
                continue
            out.append({"source": "metamath", "question": item["query"], "ground_truth": gt})
            oq = item["original_question"]
            oa = gsm_q2a.get(oq)
            if oq and oa and oq not in seen_original:
                out.append({"source": "gsm", "question": oq, "ground_truth": oa})
                seen_original.add(oq)
        return out

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_metamath.json"), convert(splits["test"])[:calib_size])
    _write(os.path.join(output_dir, "test_metamath.json"), convert(splits["train"])[:test_size])


_BBH_EXCLUDED_SUBSETS = {"dyck_languages", "word_sorting"}


def _bbh_strip_tail_options(text: str, values: set) -> str:
    """Drop a trailing options block (and its 'Options:' header) listing ``values``."""
    lines = text.splitlines()
    i = len(lines) - 1
    while i >= 0:
        stripped = lines[i].strip().lower()
        if stripped.startswith(("- ", "* ")):
            stripped = stripped[2:].strip()
        if stripped == "" or re.sub(r"[^a-z]+", "", stripped) in values:
            i -= 1
            continue
        break
    while i >= 0 and re.sub(r"[^a-z]+", "", lines[i].strip().lower()) == "options":
        i -= 1
    return "\n".join(lines[: i + 1]).rstrip()


def _bbh_convert(input_text: str, target) -> tuple:
    """Normalize one BIG-Bench Hard item to (question, ground_truth)."""
    text = str(input_text or "").strip()
    if isinstance(target, bool):
        s = "true" if target else "false"
    else:
        s = str(target).strip()
    low = s.lower()

    if low in ("true", "false"):
        return f"{text}\n\nA. True\nB. False", "A" if low == "true" else "B"
    if low in ("yes", "no"):
        return (f"{_bbh_strip_tail_options(text, {'yes', 'no'})}\n\nA. Yes\nB. No",
                "A" if low == "yes" else "B")
    if low in ("valid", "invalid"):
        return (f"{_bbh_strip_tail_options(text, {'valid', 'invalid'})}\n\nA. valid\nB. invalid",
                "A" if low == "valid" else "B")
    if re.fullmatch(r"\([A-Za-z]\)", s):
        return re.sub(r"\(([A-Z])\)", r"\1", text, flags=re.IGNORECASE), s.strip("()").upper()
    if re.fullmatch(r"[A-Za-z]", s):
        return text, s.upper()
    return text, s  # numeric or short free-form target


def prepare_bigbench(output_dir: str, calib_ratio: float = 0.1, seed: int = 42) -> None:
    """lighteval/big_bench_hard, all subsets except dyck_languages and word_sorting.
    Targets are normalized to option letters where the subset is multiple-choice
    (True/False, Yes/No, valid/invalid, "(A)"-style); other targets stay as-is.
    The calibration split is stratified per subset; both files are shuffled."""
    import datasets
    from datasets import get_dataset_config_names

    random.seed(seed)
    calibration, test = [], []
    for subset in get_dataset_config_names("lighteval/big_bench_hard", trust_remote_code=True):
        if subset in _BBH_EXCLUDED_SUBSETS:
            continue
        ds = datasets.load_dataset("lighteval/big_bench_hard", subset, trust_remote_code=True)
        if "train" not in ds:
            continue
        rows = []
        for item in ds["train"]:
            question, gt = _bbh_convert(item.get("input", ""), item.get("target", ""))
            rows.append({"subset": subset, "question": question, "ground_truth": gt})
        random.shuffle(rows)
        n_calib = max(1, int(len(rows) * calib_ratio))
        calibration.extend(rows[:n_calib])
        test.extend(rows[n_calib:])

    random.shuffle(calibration)
    random.shuffle(test)
    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_bigbench.json"), calibration)
    _write(os.path.join(output_dir, "test_bigbench.json"), test)


def prepare_barexam(output_dir: str, calib_ratio: float = 0.1, seed: int = 42) -> None:
    """reglab/barexam_qa (qa config): prompt + stem + A-D options in the question text."""
    import datasets

    ds = datasets.load_dataset("reglab/barexam_qa", name="qa", trust_remote_code=True)

    def convert(item) -> dict:
        prompt = item.get("prompt")
        if prompt is None or str(prompt).strip().lower() == "nan":
            prompt = ""
        parts = [p for p in (str(prompt).strip(), (item.get("question") or "").strip()) if p]
        options = "\n".join(
            f"{letter}. {(item.get(f'choice_{letter.lower()}') or '').strip()}"
            for letter in "ABCD"
        )
        return {
            "question": "\n\n".join(parts + [options]).strip(),
            "ground_truth": item["answer"],
        }

    items = [convert(row) for row in ds["train"]]
    random.seed(seed)
    random.shuffle(items)
    n_calib = max(1, int(len(items) * calib_ratio))

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_barexam.json"), items[:n_calib])
    _write(os.path.join(output_dir, "test_barexam.json"), items[n_calib:])


def prepare_medmcqa(
    output_dir: str,
    calib_ratio: float = 0.1,
    seed: int = 42,
    calib_size: int = 2000,
    test_size: int = 8000,
) -> None:
    """openlifescienceai/medmcqa (train split): per-subject stratified calibration
    split, then a global random sample of 2000 calibration / 8000 test items,
    matching the paper's setup."""
    import datasets
    from collections import defaultdict

    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    ds = datasets.load_dataset("openlifescienceai/medmcqa")["train"]

    def convert(item) -> Optional[dict]:
        cop = item["cop"]
        if isinstance(cop, str):
            cop = cop.strip()
            if cop.lower() in ("a", "b", "c", "d"):
                letter = cop.upper()
            elif cop.isdigit():
                letter = letter_map.get(int(cop))
            else:
                return None
        else:
            letter = letter_map.get(cop)
        if letter is None:
            return None
        options = "\n".join(
            f"{l}. {(item['op' + l.lower()] or '').strip()}" for l in "ABCD"
        )
        return {
            "subject": item["subject_name"],
            "question": f"{(item['question'] or '').strip()}\n\n{options}",
            "ground_truth": letter,
        }

    groups = defaultdict(list)
    for item in ds:
        if (rec := convert(item)) is not None:
            groups[rec["subject"]].append(rec)

    random.seed(seed)
    calibration, test = [], []
    for rows in groups.values():
        random.shuffle(rows)
        n_calib = max(1, int(len(rows) * calib_ratio))
        calibration.extend(rows[:n_calib])
        test.extend(rows[n_calib:])

    random.shuffle(calibration)
    random.shuffle(test)
    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, "calibration_medmcqa.json"), calibration[:calib_size])
    _write(os.path.join(output_dir, "test_medmcqa.json"), test[:test_size])


PREPARERS: Dict[str, callable] = {
    "gsm_symbolic": prepare_gsm_symbolic,
    "gsm8k": prepare_gsm8k,
    "nasa_history": prepare_nasa_history,
    "gsm_plus": prepare_gsm_plus,
    "metamath": prepare_metamath,
    "bigbench": prepare_bigbench,
    "barexam": prepare_barexam,
    "medmcqa": prepare_medmcqa,
}


def prepare(name: str, output_dir: str, **kwargs) -> None:
    if name not in PREPARERS:
        raise ValueError(f"Unknown dataset '{name}'. Built-ins: {sorted(PREPARERS)}. "
                         "For other benchmarks, produce JSON in the standard record format.")
    PREPARERS[name](output_dir, **kwargs)
