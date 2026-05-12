"""Within-family pair enumeration and significance filtering.

Reads `model_family_map: {virtual_slug: family_id}` from the panel
manifest and yields all `C(|family|, 2)` pairs per family. Downstream
callers feed each pair into `pairwise_metric_significance` to obtain
paired delta + CI + p_holm + Cohen's d per metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class WithinFamilyPair:
    family: str
    model_a: str
    model_b: str


def enumerate_within_family_pairs(
    family_map: dict[str, str],
) -> list[WithinFamilyPair]:
    """Enumerate (model_a < model_b) pairs within each family.

    Pairs are emitted in (family, model_a, model_b) alphabetical order.
    Families with only one member contribute zero pairs.
    """
    by_family: dict[str, list[str]] = {}
    for model, fam in family_map.items():
        by_family.setdefault(fam, []).append(model)

    out: list[WithinFamilyPair] = []
    for fam in sorted(by_family.keys()):
        members = sorted(by_family[fam])
        for a, b in combinations(members, 2):
            out.append(WithinFamilyPair(family=fam, model_a=a, model_b=b))
    return out


def filter_significance_to_within_family(
    significance_rows,
    family_map: dict[str, str],
):
    """Filter rows to (model_a, model_b) pairs sharing a family.

    Accepts either ordering of (a, b) for robustness against alternate
    pairwise enumerations elsewhere in the codebase.
    """
    pairs = {(p.model_a, p.model_b) for p in enumerate_within_family_pairs(family_map)}
    pairs |= {(b, a) for a, b in pairs}
    return [r for r in significance_rows if (r.model_a, r.model_b) in pairs]


__all__ = [
    "WithinFamilyPair",
    "enumerate_within_family_pairs",
    "filter_significance_to_within_family",
]
