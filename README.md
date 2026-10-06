<div align="center">
<img src="static/images/OracleProto_Logo_Horizontal.svg" alt="OracleProto" width="360">

**A reproducible framework for benchmarking LLM native forecasting**

Knowledge cutoff · Temporal masking · Auditable retrieval

[![CI](https://github.com/MaYiding/OracleProto/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MaYiding/OracleProto/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-42558c?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-6752a8?style=flat-square)](LICENSE)

[Leaderboard](https://oracleproto.com) · [Dataset](https://huggingface.co/datasets/MaYiding/OracleProto) · [Paper](https://arxiv.org/abs/2605.03762) · [中文](README-ZH.md)

<img src="static/images/benchmark-overview.svg" alt="300 questions, 528 points, 3 repetitions, 27 configurations" width="100%">
</div>

## Forecasting with a declared information boundary

OracleProto reconstructs resolved events as forecasting tasks with a model knowledge cutoff and a time limit on retrieval. Search results pass through a separate leakage detector, provider-native browsing is disabled, and the run records the prompts, answers and resource usage needed to inspect a result. `event` and `end_time` identify the event; the retrieval cutoff stays in the tool layer.

The public paper benchmark contains **300 questions** resolved from **2026-03-12 to 2026-09-21**. The complete evaluation covers **27 model configurations** with **three repetitions**, for **24,300 answers**. [Result CSVs](data/results/score_summary.csv) and the [scoring contract](data/results/benchmark_manifest.json) accompany the dataset.

| Question type | Questions | Points per question | Available points |
| --- | ---: | ---: | ---: |
| Yes/no | 100 | 1 | 100 |
| Named binary | 9 | 1 | 9 |
| Single-answer multiple choice | 154 | 2 | 308 |
| Multi-answer multiple choice | 37 | 3 | 111 |
| **Total** | **300** | | **528** |

**Score is the primary metric.** Exact-answer questions earn all or none of their points. Multi-answer questions earn option-level F1 credit. Accuracy, selection errors, repeated answers and inference costs explain Score; they do not replace it. The [live leaderboard](https://oracleproto.com) presents the complete results.

The final filtered-retrieval audit uses **GPT 6.1 Sol's 137 leakage labels among 10,000 reviewed results, or 1.37%**. Human review confirmed all 137 positives and found no leakage in 1,000 sampled negative labels. This audit concerns retrieved content; it does not establish a model's training-data boundary.

<details>
<summary>View the framework</summary>

<img src="static/images/Framework.png" alt="OracleProto information boundaries and evaluation flow" width="100%">

</details>

---

## 1. Code map

```
forecast_eval/                       # core package
├─ runner.py                         # build_task_plan + scheduler
├─ react.py                          # ReAct loop + Tavily end_date injection
├─ leak_filter.py                    # retrieval-content auditor
├─ llm.py                            # OpenAI-compatible client; enforces no provider-side browsing
├─ search.py                         # Tavily wrapper
├─ analysis/                         # fixed Score, selection and repetition diagnostics
├─ prompts.py / parser.py            # input renderer R / output parser Ψ
├─ types.py / errors.py / config.py  # data models / typed exceptions / Settings
├─ db.py / loader.py                 # SQLite schema migrations / dataset sync
└─ tavily_keys.py / tools.py         # API-key rotation / tool schemas
evaluation.py                        # entrypoint
scripts/                             # dataset, panel, sensitivity and plotting tools
tests/                               # tests
runs/, logs/                         # run artefacts
forecast_eval_set_example.db         # bundled example dataset
```

The reusable scripts take dataset settings from `.env` or accept input and output paths as arguments.

| Script | Purpose |
| --- | --- |
| `scripts/build_forecast_eval_set.py` | Build and validate a source dataset using `SOURCE_PARQUET`, `SOURCE_DB`, `SOURCE_TABLE`, and `SOURCE_MIN_END_DATE`. |
| `scripts/build_panel_analysis.py` | Assemble model databases from declared runs using a [panel manifest](panels/README.md). |
| `scripts/fss_sensitivity.py` | Recompute FSS over selected false-positive and false-negative penalties. |
| `scripts/plot_analysis.py` | Plot available analysis CSV/JSON artifacts for a selected run. |

---

## 2. Quickstart

### 2.1 Environment

Use `uv` :

```bash
uv sync
source .venv/bin/activate
```

or use `Conda` :

```bash
conda env create -f environment.yml
conda activate oracleproto
```

### 2.2 Configure `.env`

```bash
cp .env.example .env
```

Fill `LLM_API_KEY`, `LLM_BASE_URL`, `MODELS`, `MODEL_TRAINING_CUTOFFS`,
`TAVILY_API_KEY`, `LEAK_DETECTOR_API_KEY`, `LEAK_DETECTOR_BASE_URL`,
`LEAK_DETECTOR_MODEL`. The inline notes in [`.env.example`](./.env.example)
cover the rest.

### 2.3 Tests

```bash
pytest tests/ -q
```

### 2.4 Run

```bash
python evaluation.py
```

Each invocation creates `runs/{run_id}/` with `run_id` of the form
`YYYYMMDD-HHMMSS-{4-char hex}`. Set `RUN_ID=<existing-id>` in `.env` to resume that
run in place; completed or ineligible questions are skipped, and transient errors
retry under the original backoff policy.

---

## 3. Bring your own dataset

The bundled `forecast_eval_set_example.db` is a SQLite source generated from
`futurex-ai/Futurex-Past`. It contains one table, `test_cases`, with seven
columns: `id`, `choice_type`, `question_type`, `event`, `options`, `answer`,
and `end_time`. The table has 300 curated Level 1/2 rows spanning 2026-03-19 to
2026-09-22. It contains 104 binary questions (97 yes/no and 7 named binary),
156 single-answer questions with at least three options, and 40 multi-answer
questions.

To plug in another corpus, create a SQLite table with the same seven columns
and point `SOURCE_DB` and `SOURCE_TABLE` at it in `.env`. A source DB does not
need `dataset_metadata`; when that table is absent, the loader uses
`forecast_eval.prompts.DEFAULT_PROMPT_TEMPLATES`. Add
`dataset_metadata.features_json.prompt_reconstruction` only when the dataset
must carry its own eleven prompt-template keys.

### Reproduce the paper benchmark

The bundled example and the paper benchmark are different 300-question sets. For the paper results, download the fixed [Hugging Face SQLite release](https://huggingface.co/datasets/MaYiding/OracleProto/resolve/main/forecast_eval_set_example.db) and use its stored shared prompt recipe:

```bash
curl -L https://huggingface.co/datasets/MaYiding/OracleProto/resolve/main/forecast_eval_set_example.db -o oracleproto_300.db
```

```dotenv
SOURCE_DB=./oracleproto_300.db
SOURCE_TABLE=test_cases
PROMPT_TEMPLATE_STYLE=shared
SAMPLING_N=3
REACT_MAX_STEPS=6
REACT_MAX_SEARCH_CALLS=4
TAVILY_MAX_RESULTS=5
```

The release manifest pins the source SHA-256 and scoring contract. Explicit reasoning settings and provider defaults remain separate configurations. `High` routes and thinking switches do not imply the same provider budget. The shared prompt's strict-answer instruction is preserved as collected; the published scorer awards F1 partial credit for multi-answer questions.

---

## 4. Outputs

The primary **Score** is the percentage of available question points earned. Yes/no and named binary questions are worth 1 point each; single-answer multiple choice is worth 2; multi-answer multiple choice is worth 3. The first three use exact answers. Multi-answer credit is $`3\cdot 2TP/(2TP+FP+FN)`$. Only an exact answer earns all three points.

With three repetitions, average each question's earned points across its three answers, sum those averages, and divide by the full paper points. A perfect model scores 100. Question counts affect each type's contribution; there is no separate type-weight fitting.

```text
runs/{run_id}/
├─ manifest.json
├─ db/{model_slug}.db
├─ analysis/score/         # Score, diagnostics, coverage, scoring_meta.json
└─ logs/{run_id}.log
```

Score a completed run with:

```bash
python -B -m forecast_eval.analysis runs/{run_id}
```

`score_report.md` and `score_summary.csv` report the common-question panel. `score_by_type.csv` shows all four types; `selection_diagnostics.csv` separates extra, missed, and substituted selections; `score_by_trial.csv` and the pass/vote columns describe repeated answers. Definitions are in the [scoring contract](data/results/benchmark_manifest.json); output fields follow the [report implementation](forecast_eval/analysis/score_report.py).

The scorer reads stored answers even when `correct` is NULL and never updates raw DBs. Declared reference DBs are joined by model/question/trial with conflict checks. Missing samples or infrastructure failures stop official scoring with exit code 2 and a coverage report. `--allow-incomplete` permits diagnostics while affected scores remain empty. Model refusals and invalid answers earn zero. Cutoff-excluded questions are excluded. The fixed scoring contract, source configuration hashes, observations, and implementation are fingerprinted in `scoring_meta.json`.

The default computes 2,000 shared question bootstrap replicates for Score only. `--bootstrap-iterations 0` skips intervals for a coverage check. FSS, fitted composites, rank stability, and probability calibration are outside this report. Files directly under `analysis/` belong to their own contracts; use the artifact list in `analysis/score/scoring_meta.json`.

## 5. Contact

For questions about code usage, dataset construction, or reproducing results, please reach out to the developers directly:
- **Yiding Ma**: [yidingma@bupt.edu.cn](mailto:yidingma@bupt.edu.cn)
- **Chengyun Ruan**: [ruanchengyun815@bupt.edu.cn](mailto:ruanchengyun815@bupt.edu.cn)

For joint research, dataset and benchmark co-development, or paper collaboration, please contact the principal investigators:
- **Kaibo Huang** (corresponding author): [huangkaibo@bupt.edu.cn](mailto:huangkaibo@bupt.edu.cn)
- **Zhongliang Yang** (corresponding author): [yangzl@bupt.edu.cn](mailto:yangzl@bupt.edu.cn)

---

## 6. Paper

View Our Paper: [arXiv](http://arxiv.org/abs/2605.03762)

---

## 7. Citation

If you use this project in your research, please cite our paper:

```
@article{OracleProto,
  title={OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking},
  author={Yiding Ma, Chengyun Ruan, Kaibo Huang, Zhongliang Yang, Linna Zhou},
  journal={arXiv preprint arXiv:2605.03762},
  year={2026}
}
```

---

<details>
<summary>Advanced collection controls</summary>

## Collect raw predictions with fixed reasoning profiles

`MODEL_PROFILES` maps experiment-arm IDs to provider model IDs and explicit reasoning parameters. `MODELS` lists the arm IDs. Keep `TAVILY_MAX_RESULTS=5`, `REACT_MAX_SEARCH_CALLS=4`, `REACT_MAX_STEPS=6`, and `SAMPLING_N=3` for the paper collection budget. A provider-default arm has no declared effort and remains distinct from an explicit `none` or `high` arm.

Set `SCORE_ANSWERS=false` and invoke `python evaluation.py --skip-analysis` to collect answers without computing correctness or metrics. `WRITE_REQUEST_AUDIT=true` keeps request attempts and responses inside each model DB; `REQUIRE_HEALTHY_RETRIEVAL=true` stops collection if search or the detector cannot fulfill the retrieval contract. Resuming requires the same source, prompts, profile parameters, and collection budgets.

`COLLECTION_MODEL` selects one profile for dispatch. `COLLECTION_SAMPLE_LIMIT` bounds the number of new samples in an invocation, with zero meaning unlimited. Set `RUN_ID` to resume pending slots in the same run. `COLLECTION_PAUSED_PROFILES` holds selected profiles while other profiles run. `MODEL_QUESTION_IDS` and `MODEL_SAMPLE_INDICES` restrict dispatch to declared question and sample slots.

`COLLECTION_DRAIN_ON_ERROR=true` stops admitting new samples on a terminal failure while samples already started finish. `COLLECTION_RETAIN_MODEL_REFUSALS=true` preserves forecast-provider content refusals and skips them on resume. `COLLECTION_DEFER_TOOL_FAILURES=true` continues other slots after provider-rejected tool generations exhaust their fixed budget, leaving failed slots pending. Refusal retention and tool-failure deferral require request auditing.

`PROMPT_TEMPLATE_STYLE=shared` selects a dataset with one shared outer template; the default `typed` requires the four question-type templates. `LEAK_DETECTOR_RESPONSE_FORMAT=json_object` requests JSON from a supporting detector endpoint. `LEAK_DETECTOR_MAX_TOKENS` sets its output cap. `LEAK_DETECTOR_DROP_CONTENT_POLICY=true` drops pages that the detector provider refuses to process, retaining the raw page and rejection. Other detector failures still stop strict collection. Detector format, output-cap, or refusal-handling changes require separate runs; `COLLECTION_REFERENCE_DBS` can reuse completed samples while retaining their source configurations.

</details>
