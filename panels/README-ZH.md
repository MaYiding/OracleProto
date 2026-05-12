# Panel manifest

[English](./README.md) | [中文文档](./README-ZH.md)

Panel manifest 把分布在一个或多个 `runs/<run_id>/` 目录下的逐 model SQLite
产物集中到一个虚拟评测组合中。`scripts/build_panel_analysis.py` 读取
manifest，物化出形如普通 run 的 `runs/<panel_dir>/`（`manifest.json` +
`db/*.db`），随后 `forecast_eval.analysis` 在其上产出 panel 层级的统计资料
（逐 (model, metric) bootstrap CI、pairwise 显著性、rank 稳定性、家族内
paired Δ、Holm vs Benjamini–Hochberg 敏感性）。

每位评测者各自打包自己的 slate，因此 manifest 文件**不被 git 追踪**。
权威 schema 见 `panels/example.json`；本文件逐字段说明，让你能自己写一份。

## 字段说明

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `panel_id` | string | yes | 简短标识；用作 panel 目录后缀（`runs/panel_<panel_id>/`）。 |
| `created` | string (`YYYY-MM-DD`) | no | 仅用于审计；build 脚本不消费。 |
| `bootstrap_seed` | int | yes | paired bootstrap CI / pairwise 显著性 / rank 稳定性的随机种子。同一种子产出同一份 csv 字节。 |
| `bootstrap_iterations` | int | yes | 重采样次数（每次抽 panel-level 全量）。典型值：`5000`。 |
| `ci_alpha` | float in (0, 1) | yes | 双侧 CI 尾部质量。`0.05` 给出 95% CI。 |
| `composite_weights` | object | yes | 给 panel 侧 `composite` metric 的 3-bucket 权重；键必须正好是 `{"binary", "mc-single", "mc-multi"}`，值求和约为 1。 |
| `families` | object | yes | family 名 → `virtual_slug` 列表。驱动 `within_family_pairs.csv` 与写入 panel manifest 的 `model_family_map`。 |
| `models` | array | yes | 逐 model 条目（见下）。 |

### `models[]` 条目

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `display_label` | string | yes | 人类可读标签，用于下游报告；面板内必须唯一。 |
| `virtual_slug` | string | yes | 该 model 在源 run DB 中存储的 slug（通常形如 `<model>::r<R>::c<C>`）；面板内必须唯一。 |
| `source_run_id` | string | yes | `runs/` 下含该 model DB 的目录名。 |
| `db_filename` | string | yes | `runs/<source_run_id>/db/` 下的文件名。slug-safe 映射：`/` → `__`，其他非 `[A-Za-z0-9._-]` 字符 → `_`。 |
| `family` | string | yes | family bucket；必须是顶层 `families` 字典的某个 key。 |
| `training_cutoff` | string (`YYYY-MM-DD`) | no | 公开披露的知识截止；当源 run manifest 未记录 `model_training_cutoffs[virtual_slug]` 时作为 fallback 使用。 |

## 构建

```bash
python scripts/build_panel_analysis.py \
    --panel panels/<panel_id>.json \
    --out   runs/panel_<panel_id> \
    --mode  copy            # 或 symlink

python -m forecast_eval.analysis runs/panel_<panel_id>
```

build 脚本会把每条 `runs/<source_run_id>/db/<db_filename>` 复制（或软链接）
到 `runs/panel_<panel_id>/db/` 下，逐 DB 计算 SHA-256 digest，并写出一份带
`is_virtual_panel: true` 的 manifest，同时携带协议参数（`bootstrap_seed`、
`bootstrap_iterations`、`ci_alpha`、`composite_weights`、`families`、
`model_family_map`）。

当 `forecast_eval.analysis` 看到 `is_virtual_panel: true`，会额外产出：

- `analysis/per_model_ci.csv`
- `analysis/metric_pairwise_significance.csv`
- `analysis/rank_stability.csv` + `analysis/rank_stability_summary.json`
- `analysis/within_family_pairs.csv`
- `analysis/multi_comparison_sensitivity.csv`

并把 `bootstrap_seed`、`bootstrap_iterations`、`ci_alpha`、
`composite_weights_source`、`panel_metric_set` 一并写入
`analysis/analysis_meta.json`。

## 测试

`tests/test_panel_build.py` 在运行时发现 `panels/*.json` 的任意一份。任何由
评测者提供的 manifest 优先于 `panels/example.json`；example 仅作 fallback，
让 schema 测试始终对真实 JSON 跑一遍。若一份 manifest 都没找到，测试会 skip
而非 fail，新 clone 跑 `pytest` 无需任何本地准备。
