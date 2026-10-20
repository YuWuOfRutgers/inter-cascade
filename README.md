# Inter-Cascade

**An online, interactive LLM cascade: the strong model doesn't just answer the
hard queries — it teaches the weak model how to solve them next time.**

Standard LLM cascades send easy queries to a weak/cheap model and *defer*
uncertain ones to a strong/expensive model. They are static: faced with
repeated or semantically similar queries, they pay for the strong model again
and again. **Inter-Cascade** closes the loop. Whenever the strong model
resolves a deferred query, it also produces a generalized, reusable
problem-solving **strategy**. Strategies live in a self-evolving, similarity-
searchable repository and are injected into the weak model's context on future
queries — so the weak model improves *on the job*, with no fine-tuning, for
both API-based and open-source models.

Compared with a state-of-the-art confidence-calibrated cascade (Jung et al.,
2024), Inter-Cascade in our experiments:

- improves weak-model accuracy by up to **+33.06** points and overall system
  accuracy by up to **+6.35** points,
- cuts strong-model calls by up to **48.05%** and fees by up to **49.63%**,
- keeps a provable risk bound: with probability ≥ 1−δ, accepted answers are
  correct with rate ≥ 1−α,
- adds only ~0.1 s retrieval latency and ~10–20 MB storage per 10k+ queries.

<p align="center"><em>weak LLM ──(confidence ≥ λw?)──▶ answer<br>
&nbsp;&nbsp;│ defer&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▲ top-k strategies<br>
&nbsp;&nbsp;▼&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
strong LLM ──(answer + strategy)──▶ <b>strategy repository</b></em></p>

## Installation

```bash
git clone https://github.com/YuWuOfRutgers/inter-cascade.git
cd inter-cascade
pip install -e .            # core (OpenAI-compatible endpoints)
pip install -e ".[gemini]"  # + Google Gemini support
pip install -e ".[all]"     # + benchmark downloads
```

## Quickstart (one-command workflow)

```bash
# 1. Starter config + keys
inter-cascade init                        # writes inter_cascade.yaml
export OPENAI_API_KEY=sk-...              # weak model
export GEMINI_API_KEY=...                 # strong model

# 2. A calibration/validation set (built-in benchmark, or bring your own JSON)
inter-cascade prepare --dataset gsm8k --output-dir data

# 3. Calibrate deferral thresholds (provable risk bound from alpha/delta)
inter-cascade calibrate --config inter_cascade.yaml

# 4a. Serve single queries (what an agent system calls)
inter-cascade query --config inter_cascade.yaml --json \
  "Natalia sold clips to 48 friends in April, and half as many in May. How many altogether?"

# 4b. Or evaluate a whole test stream
inter-cascade run --config inter_cascade.yaml \
  --input data/test_gsm8k.json --output results.json
inter-cascade stats --results results.json
```

Every piece of state (calibration artifacts, the strategy repository, result
records) is a plain JSON/FAISS file on disk; runs are interruption-safe and
resumable (`run --resume`).

### Using open-source models

Any OpenAI-compatible server (vLLM, Ollama, LM Studio, llama.cpp, ...) can play
either role — the weak model on-device with a strong API model behind it is the
intended deployment:

```yaml
weak:
  provider: openai
  model: Qwen/Qwen2.5-7B-Instruct
  base_url: http://localhost:8000/v1   # vLLM
strong:
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
```

The only hard requirement is that the endpoint returns **token logprobs**
(vLLM and OpenAI do), which the confidence score is computed from.

### Bring your own task

A dataset is just a JSON list — embed multiple-choice options in the question
text and set `task.answer_type` accordingly:

```json
[{"question": "…", "ground_truth": "42"}, …]
```

### Python API

```python
from inter_cascade import load_config
from inter_cascade.runner import build_cascade

cascade = build_cascade(load_config("inter_cascade.yaml"))
result = cascade.answer("If x + y = 10 and y = 4, what is x?")
print(result.answer, result.source)   # e.g. "6 weak" — no strong-model fee
```

## Reproducing the paper

All eight benchmarks ship with a built-in downloader and a config pinning the
paper's hyperparameters. The recipe is the same three commands for each —
substitute any benchmark name from the table:

```bash
inter-cascade prepare   --dataset gsm_plus --output-dir data/gsm_plus
inter-cascade calibrate --config configs/paper/gsm_plus.yaml
inter-cascade run       --config configs/paper/gsm_plus.yaml \
    --input data/gsm_plus/test_gsm_plus.json --output results/gsm_plus.json
```

| Benchmark | `--dataset` / config | Weak α/δ | Strong α/δ | Answer type |
|---|---|---|---|---|
| GSM-Symbolic (main) | `gsm_symbolic` | 0.6 / 0.6 | 0.2 / 0.8 | numeric |
| GSM-Plus (main) | `gsm_plus` | 0.6 / 0.6 | 0.2 / 0.8 | numeric |
| MetaMath 20K (main) | `metamath` | 0.4 / 0.6 | — (always accept) | numeric |
| NASA-History (main) | `nasa_history` | 0.2 / 0.7 | — | multiple choice |
| GSM8K (appendix) | `gsm8k` | 0.5 / 0.5 | 0.2 / 0.8 | numeric |
| BigBench-Hard (appendix) | `bigbench` | 0.4 / 0.6 | — | freeform |
| BarExamQA (appendix) | `barexam` | 0.5 / 0.5 | — | multiple choice |
| MedMCQA (appendix) | `medmcqa` | 0.3 / 0.8 | — | multiple choice |

The paper's baseline and every ablation are config switches on the same
pipeline (see `configs/paper/gsm_symbolic_jung_baseline.yaml` and
`configs/paper/gsm_symbolic_ablations.yaml` for ready-made examples):

| Paper variant | `repository:` settings |
|---|---|
| **Inter-Cascade** (ours) | `retrieval: similarity`, `include_strategy: true`, `write: true` |
| Jung et al. baseline | `retrieval: none`, `write: false` |
| Random strategies | `retrieval: random` |
| No-strategy (Q/A only) | `include_strategy: false` |
| Cold start | `seed_from_calibration: false` |

```bash
bash scripts/reproduce_gsm_symbolic.sh   # prepare → calibrate → ours + baseline → stats
```

The LLM-pair and top-k ablation grids are in the paper; the GSM-Symbolic
headline numbers are Pipeline Acc. **70.37%** vs 66.04% (baseline) with
Strong-model calls **30.84%** vs 59.37%.

## How it works

1. **Retrieve.** Each query is embedded (SentenceTransformers,
   `all-MiniLM-L6-v2`) and the top-k most similar solved queries are fetched
   from the FAISS-backed strategy repository.
2. **Weak decision.** The weak model answers with retrieved strategies as
   context, in a `[Strategy]: … [Answer]: …` format. Confidence =
   exp(Σ logprob of the answer tokens), min-max normalized on the calibration
   set. Above its threshold λw → the answer is final.
3. **Strong decision.** Otherwise the strong model answers zero-shot. Above
   its threshold λs → its answer is returned **and its strategy is written to
   the repository**, teaching the weak model for future similar queries.
   Below → the query is rejected (abstention), preserving the risk guarantee.

Thresholds come from fixed-sequence testing with Clopper–Pearson bounds
(Jung et al., 2024): pick risk tolerance α and error level δ, and
`inter-cascade calibrate` finds the largest-coverage λ such that accepted
answers are ≥(1−α) accurate with probability ≥(1−δ). The paper further proves
that strategy injection improves the weak model's confidence calibration, so
coverage grows while the same bound holds.

## Repository layout

```
inter_cascade/
├── cascade.py      # the Inter-Cascade inference loop (paper Algorithm 2)
├── repository.py   # persistent FAISS strategy repository
├── calibration.py  # threshold calibration (paper Algorithm 1)
├── confidence.py   # answer-token logprob → normalized confidence
├── prompts.py      # [Strategy]/[Answer] templates per answer type
├── parsing.py      # response parsing + answer grading (incl. LLM judge)
├── metrics.py      # paper metrics (pipeline acc., strong call rate, …)
├── datasets.py     # benchmark → calibration/test JSON
├── runner.py       # config → calibration runs / cascade / streaming eval
├── llm/            # provider-agnostic clients (OpenAI-compatible, Gemini)
└── cli.py          # init / prepare / calibrate / run / query / stats
```

## Citation

If you use this code, please cite the Inter-Cascade paper:

```bibtex
@article{intercascade2026,
  title   = {From Deferral to Learning: Online In-Context Knowledge
             Distillation for LLM Cascades},
  author  = {...},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```

## License

MIT
