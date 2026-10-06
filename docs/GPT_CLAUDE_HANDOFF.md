# GPT / Claude 五配置采集交接

本任务仅采集原始模型输出，不评分、不计算统计指标。代码随 Git 提交交付；题库和原始记录通过独立数据包私下传递，不上传公开仓库。接手者必须使用独立项目目录，不得在协调者正在采集的目录解压或运行。

## 范围与计数

参考面板有 18 个配置。明确移除的两个配置为 `DeepSeek-V3.2-Exp-Think` 和 `grok-4-1-fast-reasoning`。`gemini-3.1-pro-preview` 与 `gemini-3.1-flash-lite-preview` 的原接口也不在直接续跑范围，因此直接续跑为 14 个原配置。12 指额外思考配置的数量，不是直接续跑模型数。依据为协调者的 `logs/aihubmix_probe_20260929/CONTINUATION_TABLE.md`、`runs/collection_300/plan.json` 与参考面板清单。

本交接仅包含下表五项 provider-default 配置。每项保留完整的 80 × 3 = 240 条参考观测，仅新增 220 × 3 = 660 个采样槽。合计保留 1,200 条参考观测，新增 3,300 个采样槽。不要运行额外思考矩阵，也不要重跑前 80 题。

| 请求模型 ID | temperature | top_p 请求字段 | token 上限字段 | 实验 cutoff |
|---|---:|---|---|---|
| `gpt-5.4` | 0.7 | 1.0 | `max_completion_tokens` | 2026-03-05 |
| `gpt-5.4-high` | 1.0 | 1.0 | `max_completion_tokens` | 2026-03-05 |
| `gpt-5.3-codex` | 1.0 | 1.0 | `max_completion_tokens` | 2026-02-24 |
| `claude-sonnet-4-6` | 1.0 | 省略 | `max_tokens` | 2026-02-17 |
| `claude-opus-4-6-think` | 1.0 | 省略 | `max_tokens` | 2026-02-04 |

cutoff 是面板继承的保守实验边界，不宣称为新核实的训练截止日期。配置 ID 是请求模型 ID 加 `--provider-default`。五项均省略 `reasoning_effort` 与单独的 reasoning token budget；provider-default 不表示关闭思考。`gpt-5.4-high` 和 `claude-opus-4-6-think` 必须保留准确别名，不能改成基础 ID 再手填 effort。接手者私下填写 base URL 和 Key；端点必须支持本表相同模型 ID、参数语义和 Qwen 过滤模型，禁止自动换模型。

## 固定实验参数

包内 `jobs/*.json` 固定除 base URL 和密钥以外的全部 Settings 字段，是机器执行依据；不得通过接手者的 `.env` 覆盖实验协议。

| 参数 | 固定值 |
|---|---|
| 每题独立采样 | `SAMPLING_N=3`，槽位 0、1、2 |
| ReAct 轮数 | `REACT_MAX_STEPS=6` |
| 每个样本总搜索预算 | `REACT_MAX_SEARCH_CALLS=[4]` |
| 每次搜索结果数 | `TAVILY_MAX_RESULTS=[5]` |
| 每次预测请求输出 token 上限 | `LLM_MAX_TOKENS=12000` |
| 预测请求超时 | 240 秒 |
| 搜索时间偏移 | -1 天，保留 L1 可允许性过滤与 L2 统一 end_date 注入 |
| Tavily | basic；raw_content=markdown；include_answer=false |
| 上下文网页截断 | 8,000 字符；未截断搜索响应仍写审计记录 |
| 独立泄漏过滤模型 | `qwen3.8-flash`，AIHubMix，effort=none |
| 过滤器输出 | json_object；2048 tokens；temperature=0；90 秒超时 |
| 过滤失败处理 | drop；内容政策拒绝页面保留原文、原因与判定后丢弃，其他终止性故障停止采集 |
| reflection / belief | true / false |
| nudge / 最少搜索 | 2 / 0 |
| 额外 final-answer retry | false |
| 预算意识、超额移除工具、临近上限强制 final | 均 true；lookahead=2 |
| prompt 模板 | source.db 内固定模板；style=shared |
| 保存 / 评分 | messages trace=true；request audit=true；SCORE_ANSWERS=false；skip_analysis=true |
| 恢复与提交 | RESUME=true；DB_COMMIT_BATCH=1 |
| 检索健康 / 模型内容拒绝保留 | 均 true |

一次科学采样可以包含多次 API 请求；API 重试不增加 SAMPLING_N。模型解析失败、格式不符或内容拒绝是观测，不得为了获得答案反复重采。内容拒绝单列，完成 660 个已处理槽位不保证得到 660 个有效预测。概率向量未要求采集，后续指标不能假设它存在。

## 接收文件与环境

交付包含代码 Git bundle、私下数据包及校验清单。Git bundle 提供代码提交及其历史；私下数据包解压后位于 `runs/handoff_gpt_claude/`：

- `source.db`：固定 300 题与 prompt 模板。SHA-256 为 `67243aeadbab783202d0c715c5dd01cef265ab41bf729248ff0239d6bcb06c97`。
- `reference_db/`：五个只读参考库，每库 80 题、240 条已完成观测。
- `manifest.json`：文件与执行代码哈希、80/220 两组 ID、参考完整性检查、模型到 job 文件的映射。
- `jobs/`：五个固定运行 ID 和实验设置；恢复时始终使用同一文件，不重新生成。
- `profile_probes.json`：五项能力验证证据。
- `release.json`：协调者排除五项任务的交接确认。

不要用项目根目录的其他题库重建这 300 题；新增范围按 ID 固定，不是 SQL 返回行的第 81–300 行。五个参考库独立保留，不作为新增运行的可写数据库。无需接收其他模型正在运行的数据库。

在本地磁盘选择全新空目录，避开 iCloud / 同步盘。校验包的 SHA-256 与交付清单一致后：

```sh
git clone --branch dev /path/to/forecast-code.bundle Forecast-handoff
cd Forecast-handoff
# 解压私下数据包到本项目根目录；保留 runs/handoff_gpt_claude/ 路径。
tar -xzf /path/to/forecast-gpt-claude-data.tar.gz
```

Python 要求为 3.12。使用已安装项目依赖的环境，或由接手者按 `pyproject.toml` / `environment.yml` 配置环境。本交接不要求本地编译、构建或运行评分工具。以下命令中的 `.venv/bin/python` 可替换为已配置的 Python 3.12 解释器。

`.env` 只需私下填入：

```dotenv
LLM_BASE_URL=<预测及过滤模型共用的OpenAI兼容端点>
LLM_API_KEY=<预测及过滤模型凭证>
TAVILY_API_KEY=<分配给接手者的key1>,<key2>,<key3>
```

预测与过滤器共用填写的 base URL 和 Key。模型和过滤器选择均已固定，只有这两个接入字段及 Tavily Key 由接手者配置。密钥不要写进 Git、交接记录或命令行。Tavily Key 必须与其他执行机器的分配互斥；同一额度池在其他机器消费会破坏本机余额估计。五项搜索调用预算上限为 13,200 次，HTTP 重试可能额外消耗额度；脚本每批预留 100 次。没有模型账户 Manage Key 时不能声称已检查 LLM 余额，应由账户持有人确认可用额度。

## 检查、启动、恢复

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py verify
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py status
```

verify 不调用付费 API，会核对数据和代码哈希、题目范围、模型身份、重复次数与交接放行文件。缺少放行确认必须停止，联系协调者；不得自行编造 release.json。专用入口按互斥锁限制本机同时运行一个模型。不要运行 `prepare_collection.py`、`prepare_handoff.py` 或 `collect_forecast_panel.py run`；这些是协调者的准备入口，不能用于接手方续跑。

按以下顺序逐项执行，每条结束后检查 status 和 run 日志再运行下一条：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py run gpt-5.4
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py run gpt-5.4-high
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py run gpt-5.3-codex
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py run claude-sonnet-4-6
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/run_handoff.py run claude-opus-4-6-think
```

每个模型从最多 3 个样本的 pilot 开始，随后每批最多 15 个样本，模型并发始终为 1。搜索和过滤器并发不改变此模型并发限制。工具限制、余额检查、磁盘剩余不足 10 GiB 或终止性错误会阻止继续派发。不要将五条命令并发启动，也不要用无限重启循环掩盖错误。

需暂停时向该 Python 进程发送 `SIGTERM`，当前小批处理完毕后退出；不要 `kill -9`。恢复时重执行同一模型的 run 命令，程序按数据库中的已完成槽位跳过，保留所有中断 attempt。补充密钥只编辑 `.env`。已开始的运行不得切换供应商或 base URL；如确需改变接入来源，应联系协调者单独记录来源层，不能当作同一条件继续混合。若同一模型需要换机器，须先确认原机器进程已退出，并转移完整运行目录与整个 handoff 目录，不得在两台机器同时恢复同一运行 ID。

程序异常退出时先读运行目录中的日志、manifest 与 request_events，区分额度、限流、网络、供应商拒绝和协议错误。不要修改题目、prompts、cutoff、预算、过滤器或模型别名来绕过失败；这会改变实验条件。禁止供应商原生联网或自动替换模型。

## 保存和回传

每项输出写入 `runs/<jobs文件中的run_id>/`，包含 manifest、模型 DB、日志、代码快照、executions.jsonl、dispatches.jsonl 和 handoff_assignment.json。DB 的原始 request_events 保存预测与过滤请求/响应、搜索原始响应、过滤判定及丢弃理由；消息轨迹与每题结果也保留。不要只交 CSV 或最终答案，不删除失败记录，不进行答案评分或 SQL 拼接。

参考 80 题的过滤器为 `qwen3.6-flash-2026-04-16`，新增 220 题为 `qwen3.8-flash`，必须保留为两个来源层。参考库有最终观测和消息轨迹，但缺失的完整 HTTP 日志、被丢弃网页原文与理由无法从现有记录补回；不要把新增采集的审计完备性归给参考库。

回传前让采集进程在批次边界退出。复制完整五个运行目录、完整 handoff 目录及 `logs/collection_300/quota_latest.json`；如出现 SQLite `-wal` / `-shm`，必须与 DB 一起保留，禁止只复制正在写入的主 DB。确认没有其他写入者后打包并记录每个文件的 SHA-256，使用私下渠道传递，排除 `.env`、凭证文件和 Python 环境。

协调者按 `(profile_id, question_id, sample_idx)` 检查新增 220 × 3 槽位，并把预测、保留拒绝、待处理与错误分开计数。所有文件作为独立来源纳入目录索引，保留原始 DB、run_id、attempt_id、过滤器与契约哈希。模型 DB 不直接 INSERT 拼接；最终数据分析与评分由后续任务执行。

## 给接手 AI 的任务

> 阅读 docs/GPT_CLAUDE_HANDOFF.md 并核对交付清单。仅在本机独立项目目录执行 scripts/run_handoff.py 的 verify/status/run。使用私下提供的 .env 凭证，禁止输出密钥。先确认五项 release 有效、文件和代码哈希一致，再依文档顺序逐个模型采集新增 220 题 × 3 次；保留原 80 题和全部原始请求、搜索、过滤、消息轨迹、拒绝与错误。不要运行其他模型、额外思考档、重建题库、修改实验参数、评分或分析。余额不足或终止性故障时保留断点并报告；恢复使用原固定运行 ID。完成后提供各模型成功/拒绝/待处理槽位数及完整私下回传包的校验清单。
