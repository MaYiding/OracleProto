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
scripts/                             # offline tooling
tests/                               # tests
runs/, logs/                         # run artefacts
forecast_eval_set_example.db         # bundled example dataset
```

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

After raw collection and delegated results are complete, run:

```bash
python -B -m forecast_eval.analysis runs/{run_id}
```

For collection across batches, use `python -B -m forecast_eval.analysis runs/collection_300/catalog.json --profiles PROFILE_ID ...`. Refresh the catalog in the collection workflow and explicitly select completed profiles; the catalog can also contain reasoning profiles that have not started. Scoring checks corpus and observation-export hashes and retains reference, continuation, repair, and detector strata. Output is `runs/collection_300/analysis/score/`.

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

`PROMPT_TEMPLATE_STYLE=shared` explicitly selects a dataset with one shared outer template; the default `typed` requires the four question-type templates. `python scripts/collect_forecast_panel.py run` runs the prepared continuation plan in small batches, prioritizing cheaper models. `COLLECTION_MODEL` and `COLLECTION_SAMPLE_LIMIT` restrict dispatch without changing sample budgets. The reasoning expansion is prepared separately and waits for additional quota.

To tune collection throughput, atomically replace `runs/collection_300/dispatch_control.json` with concurrency overrides, for example `{"LEAK_DETECTOR_CONCURRENCY": 10}`. The collector reads them before the next batch. `SIGTERM` requests a stop after the current batch finishes; the same run command resumes pending slots. Effective settings are retained in the run directory's `dispatches.jsonl`.

`COLLECTION_EXPENSIVE_CONCURRENCY` sets the separate sample limit for expensive profiles and accepts the same batch-boundary overrides. With `COLLECTION_DRAIN_ON_ERROR=true`, a terminal failure stops admission of pending samples while samples already started finish and retain their evidence. The failed slot remains unresolved, and the collector stops after those samples drain.

Deferred or delegated profiles can be excluded from the local queue with `runs/collection_300/local_queue_exclusions.json`, specifying `run_id` and `profiles`. A `local_queue_exclusions.json` in the run directory adds exclusions for that run without replacing another run's queue settings. Exclusions are checked before every batch; the run directory's `local_queue_state.json` records the startup queue and PID. Remaining excluded work keeps the phase status `local_queue_complete_pending_delegation`.

`COLLECTION_BATCH_SAMPLES` controls normal batches after the three-sample pilot, with a default of 60. Increasing it reduces waits for the last slow sample between batches; concurrent samples remain bounded separately. Expensive profiles retain 15-sample batches, and every batch remains limited by remaining slots and the search-quota reserve. The control file can override this setting at a batch boundary.

After receiving delegated runs, verify the sender’s checksums and confirm collection has stopped before setting each entry in `delegated_results.json` to `verified_return` with its project-relative `returned_artifacts` SHA-256 map. The catalog validates the fixed assignment, execution snapshots, raw journal, and inference contract before indexing those runs as independent sources. Unverified returns remain pending.

`COLLECTION_RETAIN_MODEL_REFUSALS=true` preserves forecast-provider content refusals and continues other slots. It requires request auditing and skips retained refusals on resume. The catalog reports them separately from predictions; other terminal errors stop strict collection unless generated-tool failure deferral is enabled.

`COLLECTION_PAUSED_PROFILES` lists authorized profiles to hold while other profiles continue. Their remaining slots stay pending until an explicit resume. Set this list in the collection plan’s `runtime`.

`COLLECTION_DEFER_TOOL_FAILURES=true` lets other slots continue when a model exhausts its fixed round budget on provider-rejected tool generations. Failed slots retain their request evidence and remain pending; they are neither counted as complete nor automatically retried. Request auditing is required. Search, authentication, quota, and other service failures still stop strict collection.

`LEAK_DETECTOR_RESPONSE_FORMAT=json_object` requests valid JSON from a supporting detector endpoint. `python scripts/catalog_collection.py` links reference and collected samples in `runs/collection_300/catalog.json` and exports observations with consistent field names, retaining source strata and raw values without scoring. Its coverage checklist identifies every pending model/question/sample slot. The observation export uses lossless gzip compression.

`LEAK_DETECTOR_DROP_CONTENT_POLICY=true` drops pages that the detector provider refuses to process. Their verdict remains `failed:content_policy`, with the raw page and rejection retained. Other detector failures still stop strict collection. This option requires its own collection stratum.

The 300-question collection sets `LEAK_DETECTOR_MAX_TOKENS=2048`. A different detector output cap requires a separate run directory; `COLLECTION_REFERENCE_DBS` can reuse completed samples from a stratum with a lower cap while retaining both configurations.

`python scripts/collect_forecast_panel.py plan --phase repair` prepares isolated retries for recorded reference failures. `MODEL_SAMPLE_INDICES` restricts each question to its failed sample slots, preserving completed samples and the declared sampling count. The repair run shares the collection lock and can start once the active collector releases it.

The anchored panel workflow uses `python scripts/prepare_collection.py`, `python scripts/collect_forecast_panel.py plan`, `python scripts/probe_collection.py`, and `python scripts/collect_forecast_panel.py run`. Its private plan and inventory live under `runs/collection_300/`; `python scripts/collect_forecast_panel.py status` reports collection counts without scoring. API credentials remain in `.env`.

</details>
