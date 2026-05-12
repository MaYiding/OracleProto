"""Pin tests for within-family pair enumeration."""
from __future__ import annotations

from forecast_eval.analysis.within_family import (
    WithinFamilyPair,
    enumerate_within_family_pairs,
)


def test_enumerate_pairs_5_families():
    family_map = {
        "gpt-a": "gpt", "gpt-b": "gpt", "gpt-c": "gpt",
        "claude-a": "claude", "claude-b": "claude",
        "gemini-a": "gemini", "gemini-b": "gemini",
        "qwen-a": "qwen", "qwen-b": "qwen", "qwen-c": "qwen",
        "ds-a": "deepseek", "ds-b": "deepseek",
    }
    pairs = enumerate_within_family_pairs(family_map)
    # C(3,2) + C(2,2) + C(2,2) + C(3,2) + C(2,2) = 3 + 1 + 1 + 3 + 1 = 9
    assert len(pairs) == 9
    for p in pairs:
        assert isinstance(p, WithinFamilyPair)
        assert family_map[p.model_a] == family_map[p.model_b]


def test_enumerate_skips_single_family():
    family_map = {"solo": "other"}
    pairs = enumerate_within_family_pairs(family_map)
    assert pairs == []


def test_pairs_alphabetical_ordering():
    family_map = {"z": "gpt", "a": "gpt"}
    pairs = enumerate_within_family_pairs(family_map)
    assert len(pairs) == 1
    assert pairs[0].model_a == "a"
    assert pairs[0].model_b == "z"
