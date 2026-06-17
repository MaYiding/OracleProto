<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

</div>

$`\Large \text{Forecasting} = \text{Gathering} \times \text{Synthesis} \times \text{Judgment} \times \text{Decision}`$

<div align="center">

Traditional benchmarks ask: “Can you recall the answer?”<br>
OracleProto asks: “Can you predict the future?”

<b>May every forecast be reproducible, may AI truly become decision support</b><br>
In service of every person’s judgments and choices for a good life

![GitHub License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.12-brightgreen?style=for-the-badge)

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

View Our Paper: [arXiv](http://arxiv.org/abs/2605.03762)

Visit Our Leaderboards: [oracleproto.com](https://oracleproto.com)

</div>

---

## Overview

- **Background & Challenges:** Evaluating LLM forecasting faces a dilemma: live benchmarks **expire easily**, and retrospective benchmarks suffer from **data leakage**. Prompting cannot establish a genuine **knowledge boundary**.
  
- **Architecture & Methods:** The OracleProto framework combines model knowledge cutoffs and temporal masking to rigorously reconstruct historical events into **reproducible, time-bounded forecasting samples**.

- **Prompt Rendering:** Each task prompt separates `event` from `end_time`. `end_time` is the resolution date that identifies the event instance; the search cutoff $`\chi_i`$ remains injected only inside the tool layer.
  
- **Experimental Results:** Tests on six contemporary LLMs show that OracleProto distinguishes models' forecasting quality, stability, and cost efficiency. It reduces the leakage rate to 1%, providing a controlled signal source for **model comparison, supervised fine-tuning, and reinforcement learning**.

<div align="center">

<img src="static/images/Framework.png" alt="Framework of OracleProto" width="100%">

Framework of OracleProto

<img src="static/images/preview/2-EN.png" alt="Online Leaderboard Overview of OracleProto" width="100%">

Online Leaderboard Overview of OracleProto

</div>

---

## 1. Code map

```
forecast_eval/                       # core package
├─ runner.py                         # build_task_plan + scheduler
├─ react.py                          # ReAct loop + Tavily end_date injection
├─ leak_filter.py                    # retrieval-content auditor
├─ llm.py                            # OpenAI-compatible client; enforces no provider-side browsing
├─ search.py                         # Tavily wrapper
├─ analysis/                         # scoring and diagnostics: accuracy, FSS, BI, composite, behavior
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
and `end_time`. The table has 312 cleaned rows spanning 2026-03-19 to
2026-05-26: 114 `yes_no`, 8 `binary_named`, and 190 `multiple_choice` rows;
35 of the `multiple_choice` rows are multi-answer questions.

To plug in another corpus, create a SQLite table with the same seven columns
and point `SOURCE_DB` and `SOURCE_TABLE` at it in `.env`. A source DB does not
need `dataset_metadata`; when that table is absent, the loader uses
`forecast_eval.prompts.DEFAULT_PROMPT_TEMPLATES`. Add
`dataset_metadata.features_json.prompt_reconstruction` only when the dataset
must carry its own eleven prompt-template keys.

---

## 4. Outputs

```
runs/{run_id}/
├─ manifest.json          # run-level metadata and hash chain
├─ db/{model_slug}.db     # one SQLite per model, independently replayable
├─ analysis/              # CSV/JSON regenerated from the raw DB
└─ logs/{run_id}.log
```

The DB stores raw observations only. Every aggregate ($`\text{pass@1}`$, FSS, BI,
composite, …) is recomputed by `forecast_eval/analysis/`, which runs at the end of
`evaluation.py` and can also be invoked standalone:

```bash
python -m forecast_eval.analysis runs/{run_id}
```

When the run is a virtual panel (`manifest.is_virtual_panel = true`), the analysis path additionally emits the panel-wide statistical envelope. See [`panels/README.md`](./panels/README.md) for the field-by-field schema and [`panels/example.json`](./panels/example.json) for a 2-model template.

---

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
