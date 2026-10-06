<div align="center">
<img src="static/images/OracleProto_Logo_Horizontal.svg" alt="OracleProto" width="360">

**对 LLM 原生预测能力进行基准评测的可复现框架**

知识截止 · 时间掩码 · 可审计检索

[![CI](https://github.com/MaYiding/OracleProto/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MaYiding/OracleProto/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-42558c?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-6752a8?style=flat-square)](LICENSE)

[排行榜](https://oracleproto.com) · [数据集](https://huggingface.co/datasets/MaYiding/OracleProto) · [论文](https://arxiv.org/abs/2605.03762) · [English](README.md)

<img src="static/images/benchmark-overview.svg" alt="300 道题、528 分、三次重复、27 个配置" width="100%">
</div>

## 在声明的信息边界内评测预测能力

OracleProto 将已解析事件重构为预测题目，声明模型知识截止，并限制检索的时间范围。检索内容经过独立泄漏检测器，供应商原生浏览被禁用，运行记录保留提示词、答案和资源用量，以便核查结果。`event` 与 `end_time` 定位事件实例，检索截止仍在工具层注入。

公开的论文评测试卷包含 **300 道题**，解析日期为 **2026-03-12 至 2026-09-21**。完整评测涵盖 **27 个模型配置**，每题 **三次重复**，共 **24,300 次作答**。数据集配套提供[结果 CSV](data/results/score_summary.csv)和[评分契约](data/results/benchmark_manifest.json)。

| 题型 | 题数 | 单题分值 | 满分 |
| --- | ---: | ---: | ---: |
| 判断题 | 100 | 1 | 100 |
| 具名二选一 | 9 | 1 | 9 |
| 多选一 | 154 | 2 | 308 |
| 多选多 | 37 | 3 | 111 |
| **合计** | **300** | | **528** |

**Score 是核心指标。** 要求精确答案的题目按对错给全分或零分，多选多按选项 F1 给部分分。正确率、选项错误、重复作答和推理成本用于解释 Score。[在线排行榜](https://oracleproto.com)展示完整结果。

最终检索内容审计采用 **GPT 6.1 Sol 核验 10,000 条结果、标注 137 条泄漏的结论，即 1.37%**。人工核验确认 137 条正例均泄漏，并在抽查的 1,000 条负例中未发现泄漏。该审计针对检索内容，不能据此确认模型的训练数据边界。

<details>
<summary>查看框架图</summary>

<img src="static/images/Framework.png" alt="OracleProto 信息边界与评测流程" width="100%">

</details>

---

## 1. 代码结构

```
forecast_eval/                       # 核心代码
├─ runner.py                         # build_task_plan + 调度
├─ react.py                          # ReAct 循环 + Tavily end_date 注入
├─ leak_filter.py                    # 检索内容审计
├─ llm.py                            # OpenAI 兼容客户端；强制禁止供应商原生浏览
├─ search.py                         # Tavily 包装
├─ analysis/                         # 固定 Score、选项与重复作答诊断
├─ prompts.py / parser.py            # 输入渲染器 R / 输出解析器 Ψ
├─ types.py / errors.py / config.py  # 数据模型 / 类型化异常 / Settings
├─ db.py / loader.py                 # SQLite schema 迁移 / 数据集同步
└─ tavily_keys.py / tools.py         # API key 轮转 / 工具 schema
evaluation.py                        # 入口
scripts/                             # 数据集、面板、敏感性分析与绘图工具
tests/                               # 测试
runs/, logs/                         # 运行产物
forecast_eval_set_example.db         # 样例数据集
```

可复用脚本从 `.env` 读取数据集配置，或通过参数接收输入和输出路径。

| 脚本 | 用途 |
| --- | --- |
| `scripts/build_forecast_eval_set.py` | 使用 `SOURCE_PARQUET`、`SOURCE_DB`、`SOURCE_TABLE` 和 `SOURCE_MIN_END_DATE` 构建并校验源数据集。 |
| `scripts/build_panel_analysis.py` | 按[面板清单](panels/README-ZH.md)汇集声明运行中的模型数据库。 |
| `scripts/fss_sensitivity.py` | 按指定的错选和漏选惩罚重新计算 FSS。 |
| `scripts/plot_analysis.py` | 为指定运行中已有的分析 CSV/JSON 产物绘图。 |

---

## 2. 快速开始

### 2.1 环境

使用 `uv` ：

```bash
uv sync
source .venv/bin/activate
```

或使用 `Conda`：

```bash
conda env create -f environment.yml
conda activate oracleproto
```

### 2.2 配置 `.env`

```bash
cp .env.example .env
```

填入 `LLM_API_KEY`、`LLM_BASE_URL`、`MODELS`、`MODEL_TRAINING_CUTOFFS`、`TAVILY_API_KEY`、`LEAK_DETECTOR_API_KEY`、`LEAK_DETECTOR_BASE_URL`、`LEAK_DETECTOR_MODEL`。其他解释说明见 [`.env.example`](./.env.example) 中的注释。

### 2.3 测试

```bash
pytest tests/ -q
```

### 2.4 运行

```bash
python evaluation.py
```

每次调用创建 `runs/{run_id}/`，`run_id` 形如 `YYYYMMDD-HHMMSS-{4-char hex}`。
在 `.env` 中设置 `RUN_ID=<existing-id>` 即可在同一目录中续跑该运行；已完成的题目或不符合条件的题目将被跳过，瞬时错误按原退避策略重试。

---

## 3. 接入自有数据集

仓库随附的 `forecast_eval_set_example.db` 是由 `futurex-ai/Futurex-Past` 生成的 SQLite 源库。它只有一张 `test_cases` 表，包含七列：`id`、`choice_type`、`question_type`、`event`、`options`、`answer` 与 `end_time`。该表含 300 道精选 Level 1/2 题，日期为 2026-03-19 至 2026-09-22，包括 104 道二选一（97 道判断题、7 道具名二选一）、156 道至少三个选项的多选一，以及 40 道多选多。

接入其他语料时，先创建同样七列的 SQLite 表，再在 `.env` 中指向 `SOURCE_DB` 与 `SOURCE_TABLE`。源库不需要 `dataset_metadata`；缺少该表时，loader 使用 `forecast_eval.prompts.DEFAULT_PROMPT_TEMPLATES`。只有当数据集需要携带自己的 11 个 prompt-template 键时，才添加 `dataset_metadata.features_json.prompt_reconstruction`。

### 复现论文评测

仓库示例库和论文试卷是两套不同的 300 题题库。复现论文结果时，下载固定的 [Hugging Face SQLite 发布文件](https://huggingface.co/datasets/MaYiding/OracleProto/resolve/main/forecast_eval_set_example.db)，并使用其中保存的共用提示词：

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

发布清单固定源库 SHA-256 和评分契约。显式思考设置与供应商默认配置分别保存，High 路由和思考开关也不代表相同的供应商预算。共用提示词中的严格答案指令保持采集时原文，公开评分器对多选多采用 F1 部分分。

---

## 4. 输出

主指标 **Score** 是已得题目分数占试卷满分的百分比。判断题、二选一各 1 分，多选一 2 分，多选多 3 分。前三类按答案完全匹配计分；多选多得分为 $`3\cdot 2TP/(2TP+FP+FN)`$，只有答案完全正确才能拿满 3 分。

三次重复实验先对每题三次得分取平均，再将各题平均分相加，除以整张试卷的满分。全部答对时 Score 为 100。各题型的贡献由题数和单题分值决定，不再另行拟合题型权重。

```text
runs/{run_id}/
├─ manifest.json
├─ db/{model_slug}.db
├─ analysis/score/         # Score、诊断、覆盖率、scoring_meta.json
└─ logs/{run_id}.log
```

对已完成的运行执行评分：

```bash
python -B -m forecast_eval.analysis runs/{run_id}
```

`score_report.md` 与 `score_summary.csv` 报告共同题目上的模型表现。`score_by_type.csv` 列出四类题型；`selection_diagnostics.csv` 区分错选、漏选和替换；`score_by_trial.csv` 及 pass/vote 列描述重复作答。指标定义见[评分契约](data/results/benchmark_manifest.json)，输出字段对应[报告实现](forecast_eval/analysis/score_report.py)。

即使 `correct` 为 NULL，评分器也会读取已存答案重新判分，不回写原始 DB。声明的参考 DB 按模型、题目、采样序号联合读取，并检查冲突。缺样本或基础设施失败会以退出码 2 阻止正式评分，同时生成覆盖报告。`--allow-incomplete` 允许查看诊断，但受影响的分数仍留空。模型拒绝与无效答案计零分；因 cutoff 排除的题目不计入。固定评分契约、来源配置哈希、观测与实现指纹写入 `scoring_meta.json`。

默认只为 Score 计算 2,000 次共享的题目 bootstrap。检查覆盖情况时可用 `--bootstrap-iterations 0` 跳过区间。FSS、拟合的综合权重、排名稳定性和概率校准不属于这份报告。`analysis/` 根目录中的文件使用各自的契约；本次评分以 `analysis/score/scoring_meta.json` 的产物列表为准。

## 5. 联系与合作

如有代码使用、数据集构建、复现问题等，欢迎直接联系项目开发者：
- **马一丁**：[yidingma@bupt.edu.cn](mailto:yidingma@bupt.edu.cn)
- **阮承沄**：[ruanchengyun815@bupt.edu.cn](mailto:ruanchengyun815@bupt.edu.cn)

如需联合研究、数据与评测基准共建、论文合作等，请联系课题负责人：
- **黄凯博**（通讯作者）：[huangkaibo@bupt.edu.cn](mailto:huangkaibo@bupt.edu.cn)
- **杨忠良**（通讯作者）：[yangzl@bupt.edu.cn](mailto:yangzl@bupt.edu.cn)

---

## 6. 论文

查看我们的论文：[arXiv](http://arxiv.org/abs/2605.03762)

---
## 7. 引用

如果您在研究中使用了本项目，请引用我们的论文：

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
<summary>高级采集设置</summary>

## 使用固定思考配置采集原始预测

`MODEL_PROFILES` 将实验配置 ID 映射到供应商模型 ID 与明确的思考参数；`MODELS` 列出实验配置 ID。论文采集预算固定为 `TAVILY_MAX_RESULTS=5`、`REACT_MAX_SEARCH_CALLS=4`、`REACT_MAX_STEPS=6`、`SAMPLING_N=3`。供应商默认模式不声明 effort，与显式 `none` 或 `high` 配置分别保存。

设置 `SCORE_ANSWERS=false` 并执行 `python evaluation.py --skip-analysis`，即可采集答案而不计算正确性或指标。`WRITE_REQUEST_AUDIT=true` 将请求尝试与响应保存在每个模型 DB 内；`REQUIRE_HEALTHY_RETRIEVAL=true` 在搜索或过滤模型无法履行检索契约时停止采集。恢复运行要求题库、提示词、思考配置及采集预算一致。

`COLLECTION_MODEL` 选择一个配置进行调度。`COLLECTION_SAMPLE_LIMIT` 限制单次启动新增的样本数，零表示不限。设置 `RUN_ID` 即可在同一运行中恢复待处理槽位。`COLLECTION_PAUSED_PROFILES` 暂停指定配置，其余配置继续运行。`MODEL_QUESTION_IDS` 与 `MODEL_SAMPLE_INDICES` 将调度限制在声明的题目和采样槽位内。

`COLLECTION_DRAIN_ON_ERROR=true` 在终止性故障后停止接收新样本，已开始的样本继续完成。`COLLECTION_RETAIN_MODEL_REFUSALS=true` 保留预测供应商的内容拒绝，并在恢复时跳过这些槽位。`COLLECTION_DEFER_TOOL_FAILURES=true` 在供应商拒绝工具生成并耗尽固定预算后继续其他槽位，将失败槽位留为待处理。拒绝留存和工具失败延后处理均要求启用请求审计。

`PROMPT_TEMPLATE_STYLE=shared` 选择使用共用外层模板的数据集；默认 `typed` 要求四类题型各自的模板。`LEAK_DETECTOR_RESPONSE_FORMAT=json_object` 向支持该选项的过滤模型接口请求 JSON。`LEAK_DETECTOR_MAX_TOKENS` 设置其输出上限。`LEAK_DETECTOR_DROP_CONTENT_POLICY=true` 丢弃过滤模型供应商拒绝处理的网页，同时保留原文与拒绝记录；其他过滤失败仍停止严格采集。过滤响应格式、输出上限或拒绝处理规则不同的配置应使用独立运行；`COLLECTION_REFERENCE_DBS` 可复用已完成样本并保留其来源配置。

</details>
