"""Minimal programmatic use of Inter-Cascade, without the CLI or YAML.

Requires OPENAI_API_KEY (weak + strong here for simplicity; mix providers as
you like). This example pins thresholds directly instead of calibrating -
fine for a demo, but use `inter-cascade calibrate` for deployments that need
the risk guarantee.
"""

import os

from inter_cascade import Deferral, InterCascade, MinMaxNormalizer, StrategyRepository
from inter_cascade.llm.openai_client import OpenAIClient

weak = OpenAIClient(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
strong = OpenAIClient(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

# An identity normalizer (x_min=0, range=1) means thresholds act on raw
# exp(answer logprob sum) in [0, 1].
normalizer = MinMaxNormalizer(x_min=0.0, range=1.0)

cascade = InterCascade(
    weak=weak,
    strong=strong,
    repository=StrategyRepository(dir="strategy_repo_demo"),
    weak_deferral=Deferral(threshold=0.8, normalizer=normalizer),
    strong_deferral=Deferral(threshold=None, normalizer=normalizer),  # always accept
    top_k=2,
    answer_type="numeric",
)

questions = [
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?",
    "A cloak takes 4 bolts of red fiber and half that much green fiber. How many bolts in total?",
]

for q in questions:
    result = cascade.answer(q)
    print(f"Q: {q}")
    print(f"A: {result.answer}   (answered by: {result.source}, "
          f"strategies retrieved: {len(result.retrieved)})\n")

# If the first query was deferred, the strong model's strategy is now in the
# repository - so the second, similar query is more likely to stay local.
