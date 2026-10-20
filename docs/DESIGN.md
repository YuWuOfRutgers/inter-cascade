# Design notes: from the research code to this library

This library is a from-scratch refactor of the original experiment code (the
`rebuttal/` research tree). This document maps the two, and records every
place where behavior intentionally differs.

## Mapping

| Research code | Here | Notes |
|---|---|---|
| `rag_setup.PersistentRAG` | `repository.StrategyRepository` | same store: SentenceTransformers + FAISS `IndexFlatIP` over L2-normalized embeddings, `documents.json`/`index.faiss`/`meta.json`, dedup by lowercased question |
| `find_threshold.threshold()` | `calibration.calibrate_threshold()` | same fixed-sequence testing with Clopper–Pearson upper bound |
| `find_threshold.minmax01/minmax01_test` | `confidence.MinMaxNormalizer` | fit on calibration, apply at test time |
| `finetune_hyperparameter.ipynb` (α/δ grid) | `calibration.scan_alpha_delta()`, exposed as `calibrate --scan` | |
| `model/request_gpt.ChatGPTClient` | `llm/openai_client.OpenAIClient` | adds `base_url` for vLLM/Ollama/any OpenAI-compatible server |
| `model/request_gemini.GeminiClient` | `llm/gemini_client.GeminiClient` | both API-key and Vertex modes kept |
| `model/prompts.py` (10 per-dataset copies) | `prompts.py` | one template pair parameterized by `answer_type` (numeric / multiple_choice / freeform) |
| `utils.extract_response_components` | `parsing.extract_strategy_and_answer` | returns `None` instead of the `"No strategy"` sentinel |
| per-dataset `our_pipeline_*.py` | `cascade.InterCascade` + `runner.run_eval` | one implementation; paper ablations are config switches |
| per-dataset `jung_pipeline_*.py` | same pipeline with `repository.retrieval: none, write: false` | |
| per-dataset `single_weak/strong_*.py` | `runner.run_calibration` | one command per role, resumable |
| `utils.stats_pipeline` | `metrics.compute_metrics` | same quantities, plus the paper's ratio metrics |
| `data_prepare_*.py` | `datasets.py` | gsm_symbolic / gsm8k / nasa_history built in; same splits and seeds |

## Faithfulness to the paper

- **Confidence** is `exp(Σ logprob of the [Answer] tokens)`, min-max
  normalized with calibration-set parameters — identical to the paper.
- **Threshold calibration** is an exact port of the research code's
  `threshold()` (the function the paper's λ values came from), including its
  handling of the degenerate top-of-grid region: windows that contain no
  errors yet, or where *all* answers are wrong (the original's NaN bound),
  are walked past untested, and the returned λ is the first grid point at
  which the Clopper–Pearson bound exceeds α.
- **Deferral** accepts strictly above the threshold (`conf > λ`), as in the
  research code.
- **Repository seeding**: by default the repository is warm-started from the
  strong model's calibration run, exactly like the experiments; the cold-start
  setting is `seed_from_calibration: false`.
- **Strong model answers zero-shot** (never with retrieved context), as in the
  paper's pipeline.

## Intentional deviations

1. **Answer-token extraction is provider-agnostic.** The research code counted
   answer tokens with `tiktoken` from the tail for GPT and scanned the token
   stream for `"Answer" "]:"` for Gemini. Here one method reconstructs the text
   from returned tokens and sums logprobs of everything after the final
   `[Answer]:` marker (whitespace-only tokens skipped). This removes the
   tokenizer dependency and works with any endpoint that returns logprobs;
   values may differ from the GPT-side original by whether the marker tokens
   straddle token boundaries, which is negligible in practice.
2. **Malformed strategies are never stored.** The research code could insert
   the literal string `"No strategy"` into the repository when parsing failed;
   here a strong answer is only stored when a strategy section was actually
   parsed.
3. **Grading order is unified**: exact match → numeric equality →
   option-letter match → LLM judge. (The research code varied slightly per
   dataset.) Grading only happens in evaluation; deployment paths never see
   ground truth.
4. **Secrets.** API keys come exclusively from environment variables named in
   the config. The research code embedded keys in scripts; those must be
   rotated before any public release.
5. **Resume** is first-class: calibration and evaluation write after every
   item and skip completed ones, replacing the manual `problems_indices`
   bookkeeping.

## Not (yet) ported

- Plotting scripts (`plot_figure_*.py`) — figures for the paper; straightforward
  to rebuild from the result records this library emits.
- The GMM similarity-threshold estimator (`gmm_separation_threshold`) — the
  paper's main experiments use no similarity cutoff (`similarity_threshold:
  null`); the estimator can return as a utility later.
- Token/cost accounting per provider price sheet — result records keep
  latencies and call counts; fee computation is provider-specific and was done
  in the analysis notebooks.
- The remaining benchmarks (`metamath`, `gsm_plus`, `bigbench`, `barexam`,
  `medmc`) — add a `prepare_*` function in
  `datasets.py`, or produce the standard JSON with any script.
