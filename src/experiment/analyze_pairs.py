"""Paired analysis of high- and low-overlap source-evidence retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import random
from typing import Any

from .evaluate_retrieval import QUERY_TYPES
from .retrievers import METHODS


TRANSITIONS = ("hit→hit", "hit→miss", "miss→hit", "miss→miss")
_METRICS = (
    "high_source_evidence_hit_at_5",
    "low_source_evidence_hit_at_5",
    "hit_drop",
    "high_source_evidence_mrr_at_5",
    "low_source_evidence_mrr_at_5",
    "mrr_drop",
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    left = int(position)
    fraction = position - left
    if left + 1 == len(sorted_values):
        return sorted_values[left]
    return sorted_values[left] * (1.0 - fraction) + sorted_values[left + 1] * fraction


def _paired_rows(scores: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    pair_ids: set[str] = set()
    run_ids: set[str] = set()
    config_hashes: set[str] = set()
    for row in scores:
        pair_id, method, query_type = row.get("pair_id"), row.get("method"), row.get("query_type")
        if not isinstance(pair_id, str) or not pair_id or method not in METHODS or query_type not in QUERY_TYPES:
            raise ValueError("Score rows need a valid pair_id, method, and query_type")
        key = (pair_id, str(method), str(query_type))
        if key in indexed:
            raise ValueError(f"Duplicate score row: {key}")
        indexed[key] = row
        pair_ids.add(pair_id)
        run_ids.add(str(row.get("run_id")))
        config_hashes.add(str(row.get("config_hash")))
    if not pair_ids:
        raise ValueError("No evaluated pairs")
    if len(run_ids) != 1 or len(config_hashes) != 1:
        raise ValueError("Paired analysis cannot mix runs or configurations")
    sorted_ids = sorted(pair_ids)
    expected = {
        (pair_id, method, query_type)
        for pair_id in sorted_ids for method in METHODS for query_type in QUERY_TYPES
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"Incomplete pair/method grid, e.g. {missing[:3]}")

    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for pair_id in sorted_ids:
        reference_high = indexed[(pair_id, METHODS[0], QUERY_TYPES[0])]
        reference_low = indexed[(pair_id, METHODS[0], QUERY_TYPES[1])]
        if reference_high["query_id"] == reference_low["query_id"]:
            raise ValueError(f"High and low queries share an ID for {pair_id}")
        if set(reference_high["evidence_node_ids"]) != set(reference_low["evidence_node_ids"]):
            raise ValueError(f"High and low gold sets differ for {pair_id}")
        if reference_high.get("source_job_family") != reference_low.get("source_job_family"):
            raise ValueError(f"High and low source families differ for {pair_id}")
        for method in METHODS:
            high = indexed[(pair_id, method, QUERY_TYPES[0])]
            low = indexed[(pair_id, method, QUERY_TYPES[1])]
            if high["query_id"] != reference_high["query_id"] or low["query_id"] != reference_low["query_id"]:
                raise ValueError(f"Methods used different query IDs for {pair_id}")
            if set(high["evidence_node_ids"]) != set(reference_high["evidence_node_ids"]) or set(low["evidence_node_ids"]) != set(reference_low["evidence_node_ids"]):
                raise ValueError(f"Methods used different gold labels for {pair_id}")
            high_hit = int(high["source_evidence_hit_at_5"])
            low_hit = int(low["source_evidence_hit_at_5"])
            if high_hit not in (0, 1) or low_hit not in (0, 1):
                raise ValueError("Source evidence hit must be binary")
            high_rr = float(high["source_evidence_rr_at_5"])
            low_rr = float(low["source_evidence_rr_at_5"])
            if not 0 <= high_rr <= 1 or not 0 <= low_rr <= 1:
                raise ValueError("Source evidence reciprocal rank must be in [0,1]")
            transition = f"{'hit' if high_hit else 'miss'}→{'hit' if low_hit else 'miss'}"
            by_method[method].append({
                "pair_id": pair_id,
                "method": method,
                "source_job_family": high.get("source_job_family"),
                "high_query_id": high["query_id"],
                "low_query_id": low["query_id"],
                "high_source_evidence_hit_at_5": high_hit,
                "low_source_evidence_hit_at_5": low_hit,
                "hit_drop": high_hit - low_hit,
                "high_source_evidence_mrr_at_5": high_rr,
                "low_source_evidence_mrr_at_5": low_rr,
                "mrr_drop": high_rr - low_rr,
                "transition": transition,
            })
    return sorted_ids, by_method


def summarize_pairs(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize absolute scores, drops, transitions, and paired intervals.

    Each bootstrap draw resamples pair IDs and keeps both query variants and
    all three methods together. Positive drop means low overlap scored worse.
    """

    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    pair_ids, by_method = _paired_rows(score_rows)
    n_pairs = len(pair_ids)
    main_table: list[dict[str, Any]] = []
    pair_outcomes: list[dict[str, Any]] = []
    transition_summary: list[dict[str, Any]] = []
    for method in METHODS:
        rows = by_method[method]
        main_table.append({
            "method": method,
            "pair_count": n_pairs,
            **{metric: _mean([float(row[metric]) for row in rows]) for metric in _METRICS},
        })
        pair_outcomes.extend(rows)
        counts = Counter(row["transition"] for row in rows)
        transition_summary.extend({
            "method": method,
            "transition": transition,
            "count": counts[transition],
            "fraction": counts[transition] / n_pairs,
        } for transition in TRANSITIONS)

    alpha = (1 - confidence) / 2
    rng = random.Random(seed)
    observations = {
        method: {metric: [float(row[metric]) for row in by_method[method]] for metric in _METRICS}
        for method in METHODS
    }
    replicates: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in _METRICS} for method in METHODS
    }
    diff_replicates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    comparisons = (("hybrid", "bm25"), ("hybrid", "dense"), ("dense", "bm25"))
    for _ in range(bootstrap_samples):
        sampled_indices = [rng.randrange(n_pairs) for _ in range(n_pairs)]
        sample_means: dict[str, dict[str, float]] = {}
        for method in METHODS:
            sample_means[method] = {}
            for metric in _METRICS:
                values = observations[method][metric]
                value = sum(values[index] for index in sampled_indices) / n_pairs
                replicates[method][metric].append(value)
                sample_means[method][metric] = value
        for method_a, method_b in comparisons:
            for metric in ("hit_drop", "mrr_drop"):
                diff_replicates[(method_a, method_b, metric)].append(
                    sample_means[method_a][metric] - sample_means[method_b][metric]
                )

    bootstrap_intervals: list[dict[str, Any]] = []
    estimates = {row["method"]: row for row in main_table}
    for method in METHODS:
        for metric in _METRICS:
            values = sorted(replicates[method][metric])
            bootstrap_intervals.append({
                "method": method,
                "metric": metric,
                "estimate": estimates[method][metric],
                "ci_lower": _percentile(values, alpha),
                "ci_upper": _percentile(values, 1 - alpha),
                "confidence": confidence,
                "bootstrap_samples": bootstrap_samples,
                "pair_count": n_pairs,
            })
    method_drop_differences: list[dict[str, Any]] = []
    for method_a, method_b in comparisons:
        for metric in ("hit_drop", "mrr_drop"):
            values = sorted(diff_replicates[(method_a, method_b, metric)])
            method_drop_differences.append({
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "drop_difference": estimates[method_a][metric] - estimates[method_b][metric],
                "ci_lower": _percentile(values, alpha),
                "ci_upper": _percentile(values, 1 - alpha),
                "confidence": confidence,
                "bootstrap_samples": bootstrap_samples,
                "pair_count": n_pairs,
            })

    family_descriptive: list[dict[str, Any]] = []
    for method in METHODS:
        family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_method[method]:
            family_rows[str(row["source_job_family"])].append(row)
        for family in sorted(family_rows):
            rows = family_rows[family]
            family_descriptive.append({
                "method": method,
                "source_job_family": family,
                "pair_count": len(rows),
                **{metric: _mean([float(row[metric]) for row in rows]) for metric in _METRICS},
            })
    first = score_rows[0]
    return {
        "run_id": first["run_id"],
        "config_hash": first["config_hash"],
        "pair_count": n_pairs,
        "query_count": n_pairs * 2,
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": confidence,
        "main_table": main_table,
        "pair_outcomes": pair_outcomes,
        "transition_summary": transition_summary,
        "bootstrap_intervals": bootstrap_intervals,
        "method_drop_differences": method_drop_differences,
        "family_descriptive": family_descriptive,
        "interpretation_limit": (
            "Source Evidence Hit@5 and MRR@5 measure retrieval of the generated "
            "query's originating evidence nodes only. Other relevant JDs were not "
            "labeled, so these numbers are not Recall@5, Precision@5, or RAG "
            "answer quality. A retrieved alternative JD may be reasonable even "
            "when the source evidence is absent."
        ),
    }
