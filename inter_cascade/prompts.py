"""Prompt templates.

Every model reply must follow the two-section format the pipeline parses::

    [Strategy]: <general approach for solving similar problems>
    [Answer]: <final answer only>

The [Strategy] section is what the strong model contributes to the repository;
the [Answer] section is what the confidence score is computed over. The
answer-formatting instructions vary by task type (numeric value vs. option
letter vs. free-form) and are substituted into shared templates, replacing the
per-dataset prompt copies of the original research code.
"""

from __future__ import annotations

from typing import List

ANSWER_FORMAT = {
    "numeric": (
        "containing ONLY the value. (1) Do NOT include units such as minutes, "
        "feet, etc.; (2) If the question asks for a percentage, ONLY provide "
        "the number (e.g., answer 20 instead of 20%); (3) Do NOT include any "
        "explanations; (4) If there is no answer, RETURN None as the value."
    ),
    "multiple_choice": (
        "containing ONLY the single capital letter of the correct option "
        "(exactly one of: A, B, C, D, ...). Do NOT include the option text or "
        "any explanation."
    ),
    "freeform": (
        "containing ONLY the final answer (e.g., a number, expression, or "
        "short phrase), with NO extra commentary."
    ),
}

EXAMPLE = {
    "numeric": (
        "[Question]: x + y = 10, y = 4, what is x?\n"
        "[Strategy]: To solve for x, isolate x by subtracting y from both "
        "sides of the equation. x = 10 - y = 10 - 4 = 6.\n"
        "[Answer]: 6"
    ),
    "multiple_choice": (
        "[Question]: x + y = 10, y = 4, what is x?\n"
        "A. 5\nB. 6\nC. 7\nD. 8\n"
        "[Strategy]: To solve for x, isolate x by subtracting y from both "
        "sides of the equation.\n"
        "[Answer]: B"
    ),
    "freeform": (
        "[Question]: x + y = 10, y = 4, what is x?\n"
        "[Strategy]: To solve for x, isolate x by subtracting y from both "
        "sides of the equation. x = 10 - y = 10 - 4 = 6.\n"
        "[Answer]: 6"
    ),
}

ZERO_SHOT_TEMPLATE = """
Based on the question below, please strictly follow this format when answering:
1. Start with [Strategy] section explaining the general approach for solving similar problems;
2. End with [Answer] section {answer_format}

Example:
{example}

Now answer this question:
[Question]: {question}
[Strategy]:
[Answer]:
"""

WITH_CONTEXT_TEMPLATE = """
Using the following problem examples:

{context}

Based on the question below, please strictly follow this format when answering:
1. Start with [Strategy] section explaining the general approach for solving similar problems;
2. End with [Answer] section {answer_format}

Now answer this question:
[Question]: {question}
[Strategy]:
[Answer]:
"""

JUDGE_PROMPT = """
Check if {llm_answer} and {ground_truth} are exactly the same:
- If both are numbers, compare them numerically (supporting formats like percentages, fractions, and scientific notation).
- If either value starts with a multiple-choice option letter (A-Z optionally followed by a period or text) extract ONLY the leading letter as the answer and compare only the letters.
If they match, return True; otherwise return False. ONLY return True or False.
"""


def format_context(docs: List, include_strategy: bool = True) -> str:
    """Render retrieved repository entries as few-shot context."""
    parts = []
    for i, doc in enumerate(docs):
        block = f"{i + 1}.[Question]: {doc.question}\n"
        if include_strategy and doc.strategy:
            block += f"[Strategy]: {doc.strategy}\n"
        block += f"[Answer]: {doc.answer}\n"
        parts.append(block)
    return "\n".join(parts)


def build_prompt(
    question: str,
    docs: List,
    answer_type: str = "numeric",
    include_strategy: bool = True,
) -> str:
    """Zero-shot prompt when ``docs`` is empty, strategy-augmented otherwise."""
    if answer_type not in ANSWER_FORMAT:
        raise ValueError(
            f"Unknown answer_type '{answer_type}' (expected one of {sorted(ANSWER_FORMAT)})"
        )
    if docs:
        return WITH_CONTEXT_TEMPLATE.format(
            context=format_context(docs, include_strategy=include_strategy),
            answer_format=ANSWER_FORMAT[answer_type],
            question=question,
        )
    return ZERO_SHOT_TEMPLATE.format(
        answer_format=ANSWER_FORMAT[answer_type],
        example=EXAMPLE[answer_type],
        question=question,
    )
