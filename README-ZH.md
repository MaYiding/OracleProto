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
scripts/                             # 离线工具
tests/                               # 测试
runs/, logs/                         # 运行产物
forecast_eval_set_example.db         # 样例数据集
```

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

原始采集及委派结果全部齐备后执行：

```bash
python -B -m forecast_eval.analysis runs/{run_id}
```

分批采集的统一入口是 `python -B -m forecast_eval.analysis runs/collection_300/catalog.json --profiles PROFILE_ID ...`。先由采集会话刷新 catalog，再明确选定已完成的配置；catalog 可能同时列出尚未开始的 reasoning 配置。评分会校验题库与观测导出的哈希，保留参考、续跑、补录和过滤分层。结果写入 `runs/collection_300/analysis/score/`。

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

`PROMPT_TEMPLATE_STYLE=shared` 显式选择共用一个外层模板的题库；默认 `typed` 要求四个按题型区分的模板。`python scripts/collect_forecast_panel.py run` 按准备好的续跑计划分小批采集，优先运行便宜模型。`COLLECTION_MODEL` 与 `COLLECTION_SAMPLE_LIMIT` 限制调度范围，不改变单个样本预算。思考档位扩展单独准备，等待补充额度。

调整采集吞吐时，原子替换 `runs/collection_300/dispatch_control.json` 中的并发覆盖值，例如 `{"LEAK_DETECTOR_CONCURRENCY": 10}`。采集器在下一小批开始前读取。`SIGTERM` 请求在当前小批完成后停止；用相同运行命令续跑待完成样本槽。实际设置保存在运行目录的 `dispatches.jsonl`。

`COLLECTION_EXPENSIVE_CONCURRENCY` 单独设置昂贵配置的采样并发，也支持在小批边界覆盖。`COLLECTION_DRAIN_ON_ERROR=true` 时，终止性失败会停止接收待执行样本，已开始的样本继续完成并留存证据。失败槽位保持未解决状态，采集器等这些样本收尾后停止。

延后或交给其他执行者的配置可通过 `runs/collection_300/local_queue_exclusions.json` 从本机队列排除，文件指定 `run_id` 和 `profiles`。运行目录中的 `local_queue_exclusions.json` 可为该运行追加排除项，不替换另一个运行的队列设置。每个小批前检查排除列表；运行目录的 `local_queue_state.json` 记录启动队列及 PID。仍有排除的任务时，批次状态保持 `local_queue_complete_pending_delegation`。

`COLLECTION_BATCH_SAMPLES` 控制 3 次采样试跑之后的普通批次大小，默认 60。增大它可以减少批次间等待最后一个慢样本的频率；同时在途的样本数仍受独立并发上限约束。昂贵配置保持每批 15 次，所有批次仍受剩余槽位和搜索额度预留限制。控制文件可在批次边界覆盖此设置。

收到委派结果后，先核对发送方校验清单并确认采集已停止，再将 `delegated_results.json` 中对应项设为 `verified_return`，填写以项目相对路径为键的 `returned_artifacts` SHA-256 映射。catalog 核对固定任务、执行快照、原始日志和推理契约后，将回传运行作为独立来源收录。未经核对的回传仍记为待接收。

`COLLECTION_RETAIN_MODEL_REFUSALS=true` 留存预测供应商的内容拒绝，并继续其他样本槽。该开关要求启用请求审计，恢复时跳过已留存的拒绝。catalog 将拒绝与预测分开统计；除启用生成工具失败延后处理外，其他终止性错误仍停止严格采集。

`COLLECTION_PAUSED_PROFILES` 列出需要暂缓的已授权配置，其他配置可继续执行。暂缓配置的剩余槽位保留为待完成，等待明确恢复。在采集计划的 `runtime` 中设置该列表。

`COLLECTION_DEFER_TOOL_FAILURES=true` 允许在模型因供应商拒绝其生成的工具调用而耗尽固定轮数后，继续采集其他槽位。失败槽位保留请求证据并保持待处理状态，既不计为完成，也不自动重试。该选项要求请求审计。搜索、鉴权、额度及其他服务故障仍停止严格采集。

`LEAK_DETECTOR_RESPONSE_FORMAT=json_object` 向支持该选项的过滤模型接口请求有效 JSON。`python scripts/catalog_collection.py` 在 `runs/collection_300/catalog.json` 中关联参考样本和采集样本，按一致的字段名导出观测记录，保留来源分层和原始值，不计算评分。其中的覆盖清单标识每个尚未完成的模型、题目和采样序号组合。观测导出采用 gzip 无损压缩。

`LEAK_DETECTOR_DROP_CONTENT_POLICY=true` 丢弃过滤模型供应商拒绝处理的页面。判定保留为 `failed:content_policy`，原始页面与拒绝记录一并保存。其他过滤错误仍会停止严格采集。此选项使用独立采集分层。

300 题采集设置 `LEAK_DETECTOR_MAX_TOKENS=2048`。不同过滤输出上限使用独立运行目录；`COLLECTION_REFERENCE_DBS` 可复用较低上限分层中已完成的样本，同时保留两种配置。

`python scripts/collect_forecast_panel.py plan --phase repair` 为参考数据中已记录的失败准备独立补录。`MODEL_SAMPLE_INDICES` 将每题限制到失败样本槽，保留已完成样本和声明的采样次数。补录运行共用采集锁，在当前采集器释放锁后启动。

保留原题的面板流程依次使用 `python scripts/prepare_collection.py`、`python scripts/collect_forecast_panel.py plan`、`python scripts/probe_collection.py` 与 `python scripts/collect_forecast_panel.py run`。本地计划和来源清单位于 `runs/collection_300/`；`python scripts/collect_forecast_panel.py status` 只报告采集数量，不评分。API 凭据保存在 `.env`。

</details>
