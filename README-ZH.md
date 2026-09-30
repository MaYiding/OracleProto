<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>通过知识截止与时间掩码，对 LLM 原生预测能力进行基准评测的可复现框架</em>

</div>

$`\Large \text{预测} = \text{信息搜集} \times \text{证据整合} \times \text{情势研判} \times \text{行动决策}`$

<div align="center">

传统基准考的是“你记得住答案吗”
<br>
OracleProto 问的是“你能预测未来吗”

<b>愿每一次预测都可复现，愿 AI 真正走向辅助决策</b>
<br>
服务于每一个人对美好生活的判断与选择

![GitHub License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.12-brightgreen?style=for-the-badge)

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

查看我们的论文：[arXiv](http://arxiv.org/abs/2605.03762)

访问我们的排行榜：[oracleproto.com](https://oracleproto.com)

</div>

---

## 概述

- **背景与挑战：** LLM预测评估面临两难：实时测试**易失效**，回顾测试存在**数据泄露**。提示词无法建立真实的**知识边界**。

- **架构和方法：** OracleProto 框架结合模型知识截止与时间遮蔽，将历史事件严谨重构为**具有时间边界的可复现的预测样本**。

- **Prompt 渲染：** 每条题目把 `event` 与 `end_time` 分开给模型。`end_time` 是定位事件实例的 resolution date；检索 cutoff $`\chi_i`$ 仍只在工具层注入。

- **实验的效果：** 测试 6 种主流 LLM 表明，OracleProto 能区分模型的预测质量、稳定性与成本效益，将泄露率降至 1%，为**模型对比、监督微调和强化学习**提供了受控的信号源。

<div align="center">

<img src="static/images/Framework.png" alt="OracleProto 框架图" width="100%">

OracleProto 框架图

<img src="static/images/preview/2-ZH.png" alt="OracleProto 在线排行榜一览" width="100%">

OracleProto 在线排行榜速览

</div>

---
## 1. 代码结构

```
forecast_eval/                       # 核心代码
├─ runner.py                         # build_task_plan + 调度
├─ react.py                          # ReAct 循环 + Tavily end_date 注入
├─ leak_filter.py                    # 检索内容审计
├─ llm.py                            # OpenAI 兼容客户端；强制禁止供应商原生浏览
├─ search.py                         # Tavily 包装
├─ analysis/                         # 评分与诊断：accuracy、FSS、BI、composite、behavior
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

仓库随附的 `forecast_eval_set_example.db` 是由 `futurex-ai/Futurex-Past` 生成的 SQLite 源库。它只有一张 `test_cases` 表，包含七列：`id`、`choice_type`、`question_type`、`event`、`options`、`answer` 与 `end_time`。该表含 300 道精选 Level 1/2 题，日期为 2026-03-19 至 2026-09-22，包括 130 道二选一（是否题或两个具名选项）、130 道至少三个选项的多选一，以及 40 道多选多。体育和电竞合计 54 道，占 18%。

接入其他语料时，先创建同样七列的 SQLite 表，再在 `.env` 中指向 `SOURCE_DB` 与 `SOURCE_TABLE`。源库不需要 `dataset_metadata`；缺少该表时，loader 使用 `forecast_eval.prompts.DEFAULT_PROMPT_TEMPLATES`。只有当数据集需要携带自己的 11 个 prompt-template 键时，才添加 `dataset_metadata.features_json.prompt_reconstruction`。

---

## 4. 输出

```
runs/{run_id}/
├─ manifest.json          # 运行级元数据与哈希链
├─ db/{model_slug}.db     # 每模型一份 SQLite，可独立重放
├─ analysis/              # 由原始 DB 重算的 CSV/JSON
└─ logs/{run_id}.log
```

DB 仅存原始观测。每一项聚合（$`\text{pass@1}`$、FSS、BI、composite 等）由 `forecast_eval/analysis/` 重算，该步骤在 `evaluation.py` 末尾自动运行，亦可独立调用：

```bash
python -m forecast_eval.analysis runs/{run_id}
```

当运行是虚拟 panel（`manifest.is_virtual_panel = true`），analysis 路径额外产出 panel-wide 统计资料。字段级 schema 见 [`panels/README.md`](./panels/README-ZH.md)，最小可工作骨架见 [`panels/example.json`](./panels/example.json)。

---

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

## 使用固定思考配置采集原始预测

`MODEL_PROFILES` 将实验配置 ID 映射到供应商模型 ID 与明确的思考参数；`MODELS` 列出实验配置 ID。论文采集预算固定为 `TAVILY_MAX_RESULTS=5`、`REACT_MAX_SEARCH_CALLS=4`、`REACT_MAX_STEPS=6`、`SAMPLING_N=3`。供应商默认模式不声明 effort，与显式 `none` 或 `high` 配置分别保存。

设置 `SCORE_ANSWERS=false` 并执行 `python evaluation.py --skip-analysis`，即可采集答案而不计算正确性或指标。`WRITE_REQUEST_AUDIT=true` 将请求尝试与响应保存在每个模型 DB 内；`REQUIRE_HEALTHY_RETRIEVAL=true` 在搜索或过滤模型无法履行检索契约时停止采集。恢复运行要求题库、提示词、思考配置及采集预算一致。

`PROMPT_TEMPLATE_STYLE=shared` 显式选择共用一个外层模板的题库；默认 `typed` 要求四个按题型区分的模板。`python scripts/collect_forecast_panel.py run` 按准备好的续跑计划分小批采集，优先运行便宜模型。`COLLECTION_MODEL` 与 `COLLECTION_SAMPLE_LIMIT` 限制调度范围，不改变单个样本预算。思考档位扩展单独准备，等待补充额度。

调整采集吞吐时，原子替换 `runs/collection_300/dispatch_control.json` 中的并发覆盖值，例如 `{"LEAK_DETECTOR_CONCURRENCY": 10}`。采集器在下一小批开始前读取。`SIGTERM` 请求在当前小批完成后停止；用相同运行命令续跑待完成样本槽。实际设置保存在运行目录的 `dispatches.jsonl`。

交给其他执行者的配置可通过 `runs/collection_300/local_queue_exclusions.json` 从本机队列排除，文件指定 `run_id` 和 `profiles`。每个小批前检查排除列表；运行目录的 `local_queue_state.json` 记录启动队列及 PID。仍有委派工作时，批次状态保持 `local_queue_complete_pending_delegation`。

`COLLECTION_RETAIN_MODEL_REFUSALS=true` 留存预测供应商的内容拒绝，并继续其他样本槽。该开关要求启用请求审计，恢复时跳过已留存的拒绝。catalog 将拒绝与预测分开统计；其他终止性错误仍停止严格采集。

`LEAK_DETECTOR_RESPONSE_FORMAT=json_object` 向支持该选项的过滤模型接口请求有效 JSON。`python scripts/catalog_collection.py` 在 `runs/collection_300/catalog.json` 中关联参考样本和采集样本，按一致的字段名导出观测记录，保留来源分层和原始值，不计算评分。其中的覆盖清单标识每个尚未完成的模型、题目和采样序号组合。观测导出采用 gzip 无损压缩。

`LEAK_DETECTOR_DROP_CONTENT_POLICY=true` 丢弃过滤模型供应商拒绝处理的页面。判定保留为 `failed:content_policy`，原始页面与拒绝记录一并保存。其他过滤错误仍会停止严格采集。此选项使用独立采集分层。

300 题采集设置 `LEAK_DETECTOR_MAX_TOKENS=2048`。不同过滤输出上限使用独立运行目录；`COLLECTION_REFERENCE_DBS` 可复用较低上限分层中已完成的样本，同时保留两种配置。

`python scripts/collect_forecast_panel.py plan --phase repair` 为参考数据中已记录的失败准备独立补录。`MODEL_SAMPLE_INDICES` 将每题限制到失败样本槽，保留已完成样本和声明的采样次数。补录运行共用采集锁，在当前采集器释放锁后启动。

保留原题的面板流程依次使用 `python scripts/prepare_collection.py`、`python scripts/collect_forecast_panel.py plan`、`python scripts/probe_collection.py` 与 `python scripts/collect_forecast_panel.py run`。本地计划和来源清单位于 `runs/collection_300/`；`python scripts/collect_forecast_panel.py status` 只报告采集数量，不评分。API 凭据保存在 `.env`。
