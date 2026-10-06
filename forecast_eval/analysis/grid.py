"""Grid-search analysis for `(real_model, R, C)` triplets.

Re-aggregates the per-virtual-slug analysis output produced by the main
flow into a triplet-keyed view that paper figures can consume directly.

The grid report uses these operations:

* `build_grid_summary` walks every virtual slug, decodes the
  `(real_model, R, C)` triplet via `db.parse_virtual_slug`, and bundles
  the per-cell `Aggregate` / `ModelProbabilisticAggregate` plus per-cell
  CIs into a `GridCell`.
* `marginal_along_C` / `marginal_along_R` filter the grid for paper
  figures 3 (saturation curves).
* `pareto_frontier` returns the non-dominated cell set for paper Fig 1.
* `paired_bootstrap_per_cell` runs single-variable bootstrap on per-
  question Brier scores via `inference.paired_bootstrap` (paired with a
  zero array) and reports the resulting BI 95% CI per slug.
* `winrate_matrix` counts (R, C) cells where real_model_a beats
  real_model_b in BI plus the subset that's statistically significant.
* `run_grid_analysis` is the entry point called from
  `analysis/__init__.py::run_analysis`. When `manifest["grid"]` is
  missing (legacy single-cell run), it returns an empty list — keeps
  the legacy regression path byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import db as dbmod
from .accuracy import Aggregate, _aggregate
from .flatten import SampleRow
from .inference import paired_bootstrap
from .probabilistic import _QuestionProbabilityRow
from .proper_score import ModelProbabilisticAggregate, brier_score_lab


# `grid_summary.csv` header — locked by `search-budget-grid` spec.
# Order matters: the writer literally writes these columns in this order.
_GRID_SUMMARY_HEADER: tuple[str, ...] = (
    "real_model",
    "R",
    "C",
    "n_eligible",
    "n_total",
    "acc_mean",
    "acc_ci_lo",
    "acc_ci_hi",
    "bi_mean",
    "bi_ci_lo",
    "bi_ci_hi",
    "nll_mean",
    "ece",
    "mean_search_calls",
    "mean_latency_ms",
    "parse_ok_rate",
    "belief_parse_ok_rate",
)


@dataclass(frozen=True)
class GridCell:
    """One `(real_model, R, C)` triplet's full metric bundle.

    `accuracy_aggregate` and `probabilistic_aggregate` are the stable
    aggregates we already compute for every virtual slug — we just keep a
    handle to them so the writer can pick whichever fields it wants. CI
    fields come from `paired_bootstrap_per_cell` (BI) and a parallel
    bootstrap on per-resolvable-sample correctness (Acc).
    """

    real_model: str
    R: int
    C: int
    accuracy_aggregate: Aggregate
    probabilistic_aggregate: ModelProbabilisticAggregate
    n_eligible: int
    n_total: int
    mean_search_calls: float | None
    mean_latency_ms: float | None
    parse_ok_rate: float | None
    belief_parse_ok_rate: float | None
    bi_ci_lo: float | None
    bi_ci_hi: float | None
    acc_ci_lo: float | None
    acc_ci_hi: float | None


@dataclass(frozen=True)
class WinrateRow:
    """Pairwise win count summary across the (R, C) grid for two real models.

    Cell-level paired-bootstrap counts (`sig_cells_a` / `sig_cells_b`) only
    increment when `paired_bootstrap.p_two_sided < alpha` AND the sign of
    the mean delta points to that side, so the two columns are mutually
    exclusive: a single cell can't be "significantly better for both".
    """

    model_a: str
    model_b: str
    total_cells: int
    wins_a: int
    wins_b: int
    ties: int
    sig_cells_a: int
    sig_cells_b: int


def _bs_to_bi(mean_bs: float) -> float:
    """$100\\bigl(1 - \\sqrt{\\overline{BS}}\\bigr)$ — same convention as
    `proper_score.brier_index`. We import via numeric value (rather than
    re-importing brier_index) because we need to convert bootstrap CI
    endpoints, not just an aggregate."""
    return 100.0 * (1.0 - math.sqrt(max(0.0, mean_bs)))


def _bi_ci_from_bs_array(
    bs_per_q: list[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float] | None:
    """Single-variable bootstrap on per-question BS, returned in BI domain.

    We reuse `inference.paired_bootstrap` paired against a zero array —
    `bs_a - bs_b == bs_per_q` so `delta_mean / ci_low / ci_high` are the
    mean BS and its 95% CI. BI is monotone-decreasing in mean BS, so the
    BI lower bound corresponds to the BS upper bound and vice versa.
    """
    if not bs_per_q:
        return None
    zeros = [0.0] * len(bs_per_q)
    res = paired_bootstrap(bs_per_q, zeros, n_bootstrap=n_bootstrap, seed=seed)
    bi_mean = _bs_to_bi(res.delta_mean)
    bi_lo = _bs_to_bi(res.ci_high)
    bi_hi = _bs_to_bi(res.ci_low)
    return bi_mean, bi_lo, bi_hi


def _acc_ci_for_samples(
    samples: list[SampleRow],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float] | None:
    """Single-variable bootstrap on per-resolvable-sample 0/1 correctness.

    The point estimate matches `Aggregate.pass_at_1_avg`; this only
    contributes the CI half. Returns `None` when the cell has no
    resolvable samples (CI is undefined)."""
    correct_arr = [
        1.0 if s.correct == 1 else 0.0
        for s in samples
        if s.is_resolvable
    ]
    if not correct_arr:
        return None
    zeros = [0.0] * len(correct_arr)
    res = paired_bootstrap(correct_arr, zeros, n_bootstrap=n_bootstrap, seed=seed)
    return res.ci_low, res.ci_high


def _parse_ok_rate(samples: list[SampleRow]) -> float | None:
    """Eligible-sample parse_ok rate. Distinct from `Aggregate.parse_failure_rate`,
    which only counts parse_ok==0 rows where `error is None` (i.e. the LLM
    returned a response but we couldn't extract a boxed answer); this
    function reports the positive-side rate over ALL eligible rows so a
    grid cell with many transient errors doesn't get a misleadingly high
    parse_ok_rate.
    """
    eligible = [s for s in samples if s.is_eligible]
    if not eligible:
        return None
    return sum(1 for s in eligible if s.parse_ok == 1) / len(eligible)


def _belief_parse_ok_rate(samples: list[SampleRow]) -> float | None:
    """Eligible-sample belief_parse_ok rate. Mirrors `_parse_ok_rate` but
    on the belief column."""
    eligible = [s for s in samples if s.is_eligible]
    if not eligible:
        return None
    return sum(1 for s in eligible if s.belief_parse_ok == 1) / len(eligible)


def paired_bootstrap_per_cell(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float] | None]:
    """Per-virtual-slug BI 95% CI via single-variable bootstrap on per-question BS.

    Returns `{slug: (bi_mean, bi_ci_lo, bi_ci_hi)}` or `{slug: None}` when
    the slug has no eligible probability rows. Keys are virtual slugs
    (`{real}::r{R}::c{C}`) so the caller can map them back to triplets via
    `db.parse_virtual_slug` if needed.
    """
    out: dict[str, tuple[float, float, float] | None] = {}
    for slug, rows in rows_by_model.items():
        bs_per_q = [brier_score_lab(r.probs, r.obs) for r in rows]
        out[slug] = _bi_ci_from_bs_array(
            bs_per_q, n_bootstrap=n_bootstrap, seed=seed,
        )
    return out


def build_grid_summary(
    samples_by_model: dict[str, list[SampleRow]],
    gt_map_global: dict[str, frozenset[str]],
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    manifest_grid: dict[str, Any],
) -> dict[tuple[str, int, int], GridCell]:
    """Re-aggregate per-virtual-slug data into `(real_model, R, C)` cells.

    Iterates over every key in `samples_by_model` (== virtual slug list).
    Slugs that don't parse via `parse_virtual_slug` are silently skipped —
    they're either real-slug fallbacks from a malformed manifest or junk
    data that wouldn't fit the grid anyway. The returned dict is keyed by
    triplet so downstream consumers (writers, plot funcs) can pivot
    however they like.
    """
    bi_ci_per_slug = paired_bootstrap_per_cell(rows_by_model)

    out: dict[tuple[str, int, int], GridCell] = {}
    for slug, samples in samples_by_model.items():
        triplet = dbmod.parse_virtual_slug(slug)
        if triplet is None:
            continue
        real, R, C = triplet

        # Aggregate (accuracy side). gt_map filtered to questions this slug
        # touches isn't strictly necessary — _aggregate tolerates extras —
        # but keeps the ABI denominator sane for cutoff-heavy cells.
        gt_subset: dict[str, frozenset[str]] = {
            qid: gt for qid, gt in gt_map_global.items()
        }
        sampling_n = max(
            (s.sample_idx + 1 for s in samples), default=1
        )
        agg_acc = _aggregate(samples, sampling_n, gt_map=gt_subset)

        # Probabilistic aggregate — reconstruct from rows_by_model so the
        # path is self-contained. We don't have crowd γ here (the
        # crowd would need to span virtual slugs, which doesn't make
        # sense), so abi_crowd / abi_uniform are computed inside
        # `proper_score.aggregate_probabilistic` from per-question scores
        # alone (uniform γ from `obs`). Grid cells remain independent.
        rows = rows_by_model.get(slug, [])
        prob_agg = _grid_cell_probabilistic_aggregate(rows)

        # Behaviour columns
        eligible = [s for s in samples if s.is_eligible]
        n_eligible = len(eligible)
        n_total = len(samples)

        bi_ci = bi_ci_per_slug.get(slug)
        bi_ci_lo = bi_ci[1] if bi_ci is not None else None
        bi_ci_hi = bi_ci[2] if bi_ci is not None else None

        acc_ci = _acc_ci_for_samples(samples)
        acc_ci_lo = acc_ci[0] if acc_ci is not None else None
        acc_ci_hi = acc_ci[1] if acc_ci is not None else None

        cell = GridCell(
            real_model=real,
            R=R,
            C=C,
            accuracy_aggregate=agg_acc,
            probabilistic_aggregate=prob_agg,
            n_eligible=n_eligible,
            n_total=n_total,
            mean_search_calls=agg_acc.avg_tool_calls,
            mean_latency_ms=agg_acc.avg_latency_ms,
            parse_ok_rate=_parse_ok_rate(samples),
            belief_parse_ok_rate=_belief_parse_ok_rate(samples),
            bi_ci_lo=bi_ci_lo,
            bi_ci_hi=bi_ci_hi,
            acc_ci_lo=acc_ci_lo,
            acc_ci_hi=acc_ci_hi,
        )
        out[(real, R, C)] = cell
    return out


def _grid_cell_probabilistic_aggregate(
    rows: list[_QuestionProbabilityRow],
) -> ModelProbabilisticAggregate:
    """Compute `ModelProbabilisticAggregate` for one cell from its question rows.

    Cells are isolated — there's no per-cell crowd because crowds across
    virtual slugs would mix different (R, C) experimental conditions. So
    we only compute uniform γ here, and `aggregate_probabilistic` will
    set `abi_crowd = abi_uniform` per its single-model fallback rule.
    """
    from .probabilistic import (
        _per_question_scores_from_rows,
        _build_uniform_gammas,
    )
    from .proper_score import aggregate_probabilistic

    per_q = _per_question_scores_from_rows(rows)
    uniform_gammas = _build_uniform_gammas(rows)
    return aggregate_probabilistic(
        per_q, crowd_gammas=None, uniform_gammas=uniform_gammas,
    )


def marginal_along_C(
    grid: dict[tuple[str, int, int], GridCell],
    fix_R: int,
) -> list[GridCell]:
    """Return cells with `R == fix_R`, sorted by (real_model asc, C asc).

    Empty list when no matching cell (e.g. caller passed a `fix_R` not in
    the manifest's `r_list`)."""
    cells = [c for c in grid.values() if c.R == fix_R]
    cells.sort(key=lambda c: (c.real_model, c.C))
    return cells


def marginal_along_R(
    grid: dict[tuple[str, int, int], GridCell],
    fix_C: int,
) -> list[GridCell]:
    """Symmetric: cells with `C == fix_C`, sorted by (real_model asc, R asc)."""
    cells = [c for c in grid.values() if c.C == fix_C]
    cells.sort(key=lambda c: (c.real_model, c.R))
    return cells


def _cell_x_value(cell: GridCell, x_axis: str) -> float | None:
    if x_axis == "mean_search_calls":
        return cell.mean_search_calls
    if x_axis == "mean_latency_ms":
        return cell.mean_latency_ms
    if x_axis == "C":
        return float(cell.C)
    raise ValueError(
        f"x_axis must be one of 'mean_search_calls', 'mean_latency_ms', 'C'; "
        f"got {x_axis!r}"
    )


def _cell_y_value(cell: GridCell, y_axis: str) -> float | None:
    if y_axis == "bi_mean":
        return cell.probabilistic_aggregate.bi
    if y_axis == "nll_mean":
        return cell.probabilistic_aggregate.nll
    raise ValueError(
        f"y_axis must be one of 'bi_mean', 'nll_mean'; got {y_axis!r}"
    )


def pareto_frontier(
    grid: dict[tuple[str, int, int], GridCell],
    *,
    x_axis: str = "mean_search_calls",
    y_axis: str = "bi_mean",
) -> list[GridCell]:
    """Pareto-optimal cell set on (cost, quality).

    Cost (`x_axis`) is lower-is-better. Quality (`y_axis`) is
    higher-is-better for `bi_mean` and lower-is-better for `nll_mean`.
    A cell `c` is **dominated** if some other cell `o` has weakly better
    cost AND weakly better quality with at least one strictly better.

    Cells with `None` on either axis are skipped — they can't be placed
    on the (x, y) plane. The returned list is sorted by
    `(real_model, R, C)` for stable downstream output.
    """
    cells_with_xy = []
    for c in grid.values():
        x = _cell_x_value(c, x_axis)
        y = _cell_y_value(c, y_axis)
        if x is None or y is None:
            continue
        cells_with_xy.append((c, x, y))

    y_minimize = (y_axis == "nll_mean")

    pareto: list[GridCell] = []
    for c, cx, cy in cells_with_xy:
        dominated = False
        for o, ox, oy in cells_with_xy:
            if o is c:
                continue
            x_weak = ox <= cx
            y_weak = (oy <= cy) if y_minimize else (oy >= cy)
            x_strict = ox < cx
            y_strict = (oy < cy) if y_minimize else (oy > cy)
            if x_weak and y_weak and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    pareto.sort(key=lambda c: (c.real_model, c.R, c.C))
    return pareto


def winrate_matrix(
    grid: dict[tuple[str, int, int], GridCell],
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> list[WinrateRow]:
    """Pairwise win count over (R, C) cells for every ordered pair (a, b) with a < b.

    For each (R, C) cell common to both models, compare BI and run a
    paired bootstrap on per-question BS to flag statistical significance.
    "wins_a" counts cells where `cell_a.bi > cell_b.bi`; "sig_cells_a"
    is the subset where the paired bootstrap is also significant in a's
    favor (`p_two_sided < alpha` AND `delta_mean < 0`, i.e. a's BS is
    smaller hence a is better).
    """
    real_models = sorted({rm for rm, _, _ in grid.keys()})
    out: list[WinrateRow] = []
    for i, ma in enumerate(real_models):
        for mb in real_models[i + 1:]:
            cells_a = {(R, C): cell for (rm, R, C), cell in grid.items() if rm == ma}
            cells_b = {(R, C): cell for (rm, R, C), cell in grid.items() if rm == mb}
            common_rc = sorted(set(cells_a.keys()) & set(cells_b.keys()))
            wins_a = wins_b = ties = sig_a = sig_b = 0
            for (R, C) in common_rc:
                a = cells_a[(R, C)]
                b = cells_b[(R, C)]
                a_bi = a.probabilistic_aggregate.bi
                b_bi = b.probabilistic_aggregate.bi
                if a_bi is not None and b_bi is not None:
                    if a_bi > b_bi:
                        wins_a += 1
                    elif a_bi < b_bi:
                        wins_b += 1
                    else:
                        ties += 1

                slug_a = dbmod.compose_virtual_slug(ma, R, C)
                slug_b = dbmod.compose_virtual_slug(mb, R, C)
                rows_a = rows_by_model.get(slug_a, [])
                rows_b = rows_by_model.get(slug_b, [])
                bs_a_by_q = {
                    r.question_id: brier_score_lab(r.probs, r.obs)
                    for r in rows_a
                }
                bs_b_by_q = {
                    r.question_id: brier_score_lab(r.probs, r.obs)
                    for r in rows_b
                }
                common_q = sorted(set(bs_a_by_q.keys()) & set(bs_b_by_q.keys()))
                if not common_q:
                    continue
                bs_a_arr = [bs_a_by_q[q] for q in common_q]
                bs_b_arr = [bs_b_by_q[q] for q in common_q]
                res = paired_bootstrap(
                    bs_a_arr, bs_b_arr,
                    n_bootstrap=n_bootstrap, seed=seed,
                )
                if res.p_two_sided < alpha:
                    if res.delta_mean < 0:
                        sig_a += 1
                    elif res.delta_mean > 0:
                        sig_b += 1
            out.append(WinrateRow(
                model_a=ma,
                model_b=mb,
                total_cells=len(common_rc),
                wins_a=wins_a,
                wins_b=wins_b,
                ties=ties,
                sig_cells_a=sig_a,
                sig_cells_b=sig_b,
            ))
    return out


def run_grid_analysis(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    samples_by_model: dict[str, list[SampleRow]],
    gt_map_global: dict[str, frozenset[str]],
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    analysis_dir: Path,
) -> list[Path]:
    """Grid-analysis entry point. Returns the list of CSV paths written.

    Returns an empty list when `manifest["grid"]` is missing — that's the
    legacy single-cell run path, where existing analysis output is
    already byte-equivalent to pre-change behavior. The caller
    (`run_analysis`) still appends our return value to its `written`
    list, so the empty-list path is the no-op needed by D7.
    """
    from .writers import (
        _write_grid_marginal_C_csv,
        _write_grid_marginal_R_csv,
        _write_grid_pareto_csv,
        _write_grid_summary_csv,
        _write_grid_winrate_csv,
    )

    grid_meta = manifest.get("grid")
    if not grid_meta:
        return []

    grid = build_grid_summary(
        samples_by_model, gt_map_global, rows_by_model, grid_meta,
    )

    written: list[Path] = []

    written.append(_write_grid_summary_csv(
        analysis_dir / "grid_summary.csv", grid,
    ))

    r_list = list(grid_meta.get("r_list", []))
    c_list = list(grid_meta.get("c_list", []))
    default_r = grid_meta.get("default_r")
    if default_r is None and r_list:
        default_r = r_list[0]
    default_c = grid_meta.get("default_c")
    if default_c is None and c_list:
        default_c = c_list[0]

    if default_r is not None:
        marg_c = marginal_along_C(grid, int(default_r))
        written.append(_write_grid_marginal_C_csv(
            analysis_dir / "grid_marginal_C.csv", marg_c, int(default_r),
        ))
    if default_c is not None:
        marg_r = marginal_along_R(grid, int(default_c))
        written.append(_write_grid_marginal_R_csv(
            analysis_dir / "grid_marginal_R.csv", marg_r, int(default_c),
        ))

    pareto = pareto_frontier(grid)
    written.append(_write_grid_pareto_csv(
        analysis_dir / "grid_pareto.csv", pareto, grid,
    ))

    winrate = winrate_matrix(grid, rows_by_model)
    written.append(_write_grid_winrate_csv(
        analysis_dir / "grid_winrate.csv", winrate,
    ))

    return written


__all__ = [
    "GridCell",
    "WinrateRow",
    "_GRID_SUMMARY_HEADER",
    "build_grid_summary",
    "marginal_along_C",
    "marginal_along_R",
    "pareto_frontier",
    "paired_bootstrap_per_cell",
    "winrate_matrix",
    "run_grid_analysis",
]
