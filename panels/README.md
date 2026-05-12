# Panel manifests

A panel manifest bundles per-model SQLite outputs from one or more `runs/<run_id>/`
directories into a single virtual evaluation slate. `scripts/build_panel_analysis.py`
reads the manifest, materialises a `runs/<panel_dir>/` whose shape matches a normal
run (`manifest.json` + `db/*.db`), and `forecast_eval.analysis` then emits the
panel-wide statistical envelope (per-(model, metric) bootstrap CI, pairwise
significance, rank stability, within-family pairs, Holm-vs-BH sensitivity).

Each evaluator bundles their own slate, so manifest files are **not tracked**
by git. The canonical schema is `panels/example.json`; this README documents
every field so you can write your own.

## Field reference

### Top-level

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `panel_id` | string | yes | Short identifier; becomes the panel dir suffix (`runs/panel_<panel_id>/`). |
| `created` | string (`YYYY-MM-DD`) | no | Audit-only timestamp; not consumed by the build script. |
| `bootstrap_seed` | int | yes | Seed for paired bootstrap CI / pairwise significance / rank stability. Same seed reproduces the same csv bytes. |
| `bootstrap_iterations` | int | yes | Resample count (panel-level resampling per draw). Typical: `5000`. |
| `ci_alpha` | float in (0, 1) | yes | Two-sided CI tail mass. `0.05` gives 95% CI. |
| `composite_weights` | object | yes | 3-bucket weights for the panel-side `composite` metric. Keys must be exactly `{"binary", "mc-single", "mc-multi"}`; values sum to ~1. |
| `families` | object | yes | Map from family name to list of `virtual_slug`s. Drives `within_family_pairs.csv` and the `model_family_map` stamped into the panel manifest. |
| `models` | array | yes | Per-model entries (see below). |

### `models[]` entries

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `display_label` | string | yes | Human-readable label used in downstream tables. Must be unique within the panel. |
| `virtual_slug` | string | yes | The slug stored in the source run's DB (typically `<model>::r<R>::c<C>`). Must be unique within the panel. |
| `source_run_id` | string | yes | Folder name under `runs/` containing this model's DB. |
| `db_filename` | string | yes | Filename inside `runs/<source_run_id>/db/`. Slug-safe map: `/` → `__`, other non-`[A-Za-z0-9._-]` chars → `_`. |
| `family` | string | yes | Family bucket; must be a key of the top-level `families` map. |
| `training_cutoff` | string (`YYYY-MM-DD`) | no | Disclosed knowledge cutoff. Used as a fallback when the source run manifest does not record `model_training_cutoffs[virtual_slug]`. |

## Build

```bash
python scripts/build_panel_analysis.py \
    --panel panels/<panel_id>.json \
    --out   runs/panel_<panel_id> \
    --mode  copy            # or symlink

python -m forecast_eval.analysis runs/panel_<panel_id>
```

The build script copies (or symlinks) each `runs/<source_run_id>/db/<db_filename>`
into `runs/panel_<panel_id>/db/`, computes per-DB SHA-256 digests, and writes a
manifest with `is_virtual_panel: true` plus the protocol parameters
(`bootstrap_seed`, `bootstrap_iterations`, `ci_alpha`, `composite_weights`,
`families`, `model_family_map`).

When `forecast_eval.analysis` sees `is_virtual_panel: true` it additionally
emits:

- `analysis/per_model_ci.csv`
- `analysis/metric_pairwise_significance.csv`
- `analysis/rank_stability.csv` + `analysis/rank_stability_summary.json`
- `analysis/within_family_pairs.csv`
- `analysis/multi_comparison_sensitivity.csv`

and stamps `bootstrap_seed`, `bootstrap_iterations`, `ci_alpha`,
`composite_weights_source`, and `panel_metric_set` into `analysis/analysis_meta.json`.

## Tests

`tests/test_panel_build.py` discovers any `panels/*.json` at runtime. Any
evaluator-supplied manifest is preferred over `panels/example.json`; the
example only acts as a fallback so the schema tests still exercise real
JSON. If no manifest is found at all the tests skip rather than fail, so
a fresh clone passes `pytest` without any local setup.
