"""Additive V3 reranking over a verified, frozen V2 run.

The source run and qrels are read only. This module has no dependency on V2's
entry point and cannot rewrite a V2 artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any

from .evaluate_retrieval import QUERY_TYPES, evaluate_results, score_one
from .retrievers import METHODS


RERANK_METHOD = "rerank_bm25_dense"
ALL_METHODS = (*METHODS, RERANK_METHOD)
METRICS = (
    "high_source_evidence_hit_at_5", "low_source_evidence_hit_at_5", "hit_drop",
    "high_source_evidence_mrr_at_5", "low_source_evidence_mrr_at_5", "mrr_drop",
)


def candidate_ids(by_method: Mapping[str, Mapping[str, Any]], *, top_k: int = 20) -> list[str]:
    """Preserve BM25 then dense order while deduplicating by stable node ID."""
    if set(by_method) != set(METHODS) or top_k != 20:
        raise ValueError("V3 requires the complete V2 method grid and each Top-20 list")
    seen: set[str] = set()
    merged: list[str] = []
    for method in ("bm25", "dense"):
        ranked = by_method[method].get("ranked_node_ids")
        if (not isinstance(ranked, list) or len(ranked) != top_k
                or any(not isinstance(node_id, str) or not node_id for node_id in ranked)
                or len(set(ranked)) != top_k):
            raise ValueError(f"Invalid V2 {method} Top-20 list")
        for node_id in ranked:
            if node_id not in seen:
                seen.add(node_id)
                merged.append(node_id)
    return merged


def rerank_queries(
    results: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    model: Any,
    *,
    run_id: str,
    config_hash: str,
) -> list[dict[str, Any]]:
    """Score query/node pairs locally and return all candidates in rank order."""
    node_text = {row["node_id"]: row["retrieval_text"] for row in nodes}
    if len(node_text) != len(nodes) or any(not isinstance(value, str) for value in node_text.values()):
        raise ValueError("Frozen nodes need unique IDs and retrieval_text")
    by_query: dict[str, dict[str, Mapping[str, Any]]] = {}
    for result in results:
        query_id, method = result["query_id"], result["method"]
        group = by_query.setdefault(query_id, {})
        if method in group:
            raise ValueError(f"Duplicate V2 result: {query_id}/{method}")
        group[method] = result
    output: list[dict[str, Any]] = []
    if set(by_query) != {row["query_id"] for row in queries}:
        raise ValueError("V2 result/query IDs differ")
    for query in sorted(queries, key=lambda row: row["query_id"]):
        query_id = query["query_id"]
        ids = candidate_ids(by_query[query_id])
        if any(node_id not in node_text for node_id in ids):
            raise ValueError(f"V2 result refers to an unknown node: {query_id}")
        question = query["query"]
        if not isinstance(question, str) or not question:
            raise ValueError(f"Invalid query text: {query_id}")
        started = perf_counter()
        raw_scores = model.compute_score([(question, node_text[node_id]) for node_id in ids])
        elapsed_ms = (perf_counter() - started) * 1000
        if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) != len(ids):
            raise ValueError(f"Reranker returned wrong score count for {query_id}")
        scores = [float(score) for score in raw_scores]
        if any(not math.isfinite(score) for score in scores):
            raise ValueError(f"Reranker returned non-finite score for {query_id}")
        ordered = sorted(zip(ids, scores, strict=True), key=lambda item: (-item[1], item[0]))
        output.append({
            "run_id": run_id,
            "query_id": query_id,
            "method": RERANK_METHOD,
            "config_hash": config_hash,
            "candidate_node_ids": ids,
            "candidate_count": len(ids),
            "ranked_node_ids": [node_id for node_id, _ in ordered],
            "ranked_results": [
                {"node_id": node_id, "rank": rank, "score": score}
                for rank, (node_id, score) in enumerate(ordered, 1)
            ],
            "rerank_elapsed_ms": elapsed_ms,
        })
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    index = int(position)
    fraction = position - index
    if index + 1 == len(ordered):
        return ordered[index]
    return ordered[index] * (1 - fraction) + ordered[index + 1] * fraction


def analyze_v3(
    v2_results: Sequence[Mapping[str, Any]],
    reranked: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Compare frozen V2 scores and V3 scores with pair-level resampling."""
    if bootstrap_samples < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid paired-bootstrap settings")
    base_scores = evaluate_results(v2_results, qrels, queries, top_k=5)
    query_by_id = {row["query_id"]: row for row in queries}
    gold_by_id = {row["query_id"]: row["evidence_node_ids"] for row in qrels}
    if len(query_by_id) != len(queries) or set(query_by_id) != set(gold_by_id):
        raise ValueError("Frozen queries/qrels are incomplete")
    if len(reranked) != len(queries) or {row["query_id"] for row in reranked} != set(query_by_id):
        raise ValueError("Reranked results do not cover every query exactly once")
    scores_by_query_method: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base_scores:
        scores_by_query_method[(row["query_id"], row["method"])] = dict(row)
    for row in reranked:
        query_id = row["query_id"]
        if row.get("method") != RERANK_METHOD:
            raise ValueError("Unexpected V3 method")
        ranked = row.get("ranked_node_ids")
        candidates = row.get("candidate_node_ids")
        if (not isinstance(ranked, list) or not isinstance(candidates, list)
                or len(ranked) != len(candidates) or set(ranked) != set(candidates)
                or len(ranked) != len(set(ranked)) or len(ranked) < 5):
            raise ValueError(f"Invalid V3 ranking for {query_id}")
        hit, reciprocal_rank, _ = score_one(ranked, gold_by_id[query_id], top_k=5)
        query = query_by_id[query_id]
        key = (query_id, RERANK_METHOD)
        if key in scores_by_query_method:
            raise ValueError(f"Duplicate V3 result for {query_id}")
        scores_by_query_method[key] = {
            "query_id": query_id, "pair_id": query["pair_id"],
            "query_type": query["query_type"], "method": RERANK_METHOD,
            "source_evidence_hit_at_5": hit,
            "source_evidence_rr_at_5": reciprocal_rank,
        }
    pairs: dict[str, dict[str, str]] = {}
    for query in queries:
        pair = pairs.setdefault(query["pair_id"], {})
        typ = query["query_type"]
        if typ not in QUERY_TYPES or typ in pair:
            raise ValueError("Invalid or duplicated high/low pair")
        pair[typ] = query["query_id"]
    if not pairs or any(set(pair) != set(QUERY_TYPES) for pair in pairs.values()):
        raise ValueError("Incomplete high/low pairs")
    pair_ids = sorted(pairs)
    per_pair: dict[str, list[dict[str, float]]] = {}
    main_table: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        method_pairs: list[dict[str, float]] = []
        for pair_id in pair_ids:
            high_id = pairs[pair_id][QUERY_TYPES[0]]
            low_id = pairs[pair_id][QUERY_TYPES[1]]
            high = scores_by_query_method[(high_id, method)]
            low = scores_by_query_method[(low_id, method)]
            high_hit, low_hit = high["source_evidence_hit_at_5"], low["source_evidence_hit_at_5"]
            high_rr, low_rr = high["source_evidence_rr_at_5"], low["source_evidence_rr_at_5"]
            method_pairs.append({
                "high_source_evidence_hit_at_5": float(high_hit),
                "low_source_evidence_hit_at_5": float(low_hit),
                "hit_drop": float(high_hit - low_hit),
                "high_source_evidence_mrr_at_5": float(high_rr),
                "low_source_evidence_mrr_at_5": float(low_rr),
                "mrr_drop": float(high_rr - low_rr),
            })
        per_pair[method] = method_pairs
        main_table.append({
            "method": method, "pair_count": len(pair_ids),
            **{metric: sum(row[metric] for row in method_pairs) / len(pair_ids) for metric in METRICS},
        })

    by_query = {(row["query_id"], row["method"]): row for row in v2_results}
    rerank_by_query = {row["query_id"]: row for row in reranked}
    candidate_coverage = []
    for typ in QUERY_TYPES:
        ids = [query_id for query_id, row in query_by_id.items() if row["query_type"] == typ]
        for method in (*METHODS, "bm25_dense_union"):
            hits = 0
            for query_id in ids:
                ranked = (rerank_by_query[query_id]["candidate_node_ids"] if method == "bm25_dense_union"
                          else by_query[(query_id, method)]["ranked_node_ids"])
                hits += bool(set(ranked) & set(gold_by_id[query_id]))
            candidate_coverage.append({
                "query_type": typ, "candidate_source": method,
                "query_count": len(ids), "source_evidence_in_candidates": hits,
                "source_evidence_candidate_coverage": hits / len(ids),
            })

    rng = random.Random(seed)
    alpha = (1 - confidence) / 2
    replicates = {(method, metric): [] for method in ALL_METHODS for metric in METRICS}
    differences = {(baseline, metric): [] for baseline in METHODS for metric in METRICS}
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(pair_ids)) for _ in pair_ids]
        sampled = {
            (method, metric): sum(per_pair[method][i][metric] for i in indices) / len(indices)
            for method in ALL_METHODS for metric in METRICS
        }
        for key, value in sampled.items():
            replicates[key].append(value)
        for baseline in METHODS:
            for metric in METRICS:
                differences[(baseline, metric)].append(
                    sampled[(RERANK_METHOD, metric)] - sampled[(baseline, metric)]
                )
    intervals = []
    comparisons = []
    estimates = {row["method"]: row for row in main_table}
    for method in ALL_METHODS:
        for metric in METRICS:
            values = replicates[(method, metric)]
            intervals.append({
                "method": method, "metric": metric, "estimate": estimates[method][metric],
                "ci_lower": _percentile(values, alpha), "ci_upper": _percentile(values, 1 - alpha),
                "confidence": confidence, "bootstrap_samples": bootstrap_samples,
            })
    for baseline in METHODS:
        for metric in METRICS:
            values = differences[(baseline, metric)]
            comparisons.append({
                "new_method": RERANK_METHOD, "baseline": baseline, "metric": metric,
                "difference": estimates[RERANK_METHOD][metric] - estimates[baseline][metric],
                "ci_lower": _percentile(values, alpha), "ci_upper": _percentile(values, 1 - alpha),
                "confidence": confidence, "bootstrap_samples": bootstrap_samples,
            })
    return {
        "pair_count": len(pair_ids), "query_count": len(queries),
        "main_table": main_table, "candidate_coverage": candidate_coverage,
        "bootstrap_intervals": intervals, "paired_differences": comparisons,
        "mean_rerank_ms_per_query": sum(row["rerank_elapsed_ms"] for row in reranked) / len(reranked),
        "interpretation_limit": (
            "Only the generated query's originating evidence nodes are labeled. "
            "Candidate coverage is an oracle upper bound for source-evidence Hit@5 "
            "within that candidate pool, not an observed reranker result, all-relevant "
            "Recall@5, or answer quality."
        ),
    }


def write_v3_report(analysis: Mapping[str, Any], report_dir: Path) -> dict[str, Path]:
    """Write a new V3 report directory without replacing any existing file."""
    if report_dir.exists():
        raise FileExistsError(f"V3 report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True)
    files = {
        "main_table": ("main_table.csv", analysis["main_table"],
                       ("method", "pair_count", *METRICS)),
        "candidate_coverage": ("candidate_coverage.csv", analysis["candidate_coverage"],
                               ("query_type", "candidate_source", "query_count",
                                "source_evidence_in_candidates", "source_evidence_candidate_coverage")),
        "bootstrap_intervals": ("bootstrap_intervals.csv", analysis["bootstrap_intervals"],
                                ("method", "metric", "estimate", "ci_lower", "ci_upper",
                                 "confidence", "bootstrap_samples")),
        "paired_differences": ("paired_differences.csv", analysis["paired_differences"],
                               ("new_method", "baseline", "metric", "difference", "ci_lower",
                                "ci_upper", "confidence", "bootstrap_samples")),
    }
    paths: dict[str, Path] = {}
    for name, (filename, rows, columns) in files.items():
        path = report_dir / filename
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = path
    from .io import write_json

    paths["analysis"] = report_dir / "analysis.json"
    write_json(dict(analysis), paths["analysis"])
    lines = [
        "# JD evidence retrieval V3: local reranking extension", "",
        f"Frozen V2 pairs: {analysis['pair_count']}; queries: {analysis['query_count']}.", "",
        "| Method | High Hit@5 | Low Hit@5 | Hit drop | High MRR@5 | Low MRR@5 | MRR drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["main_table"]:
        lines.append("| " + " | ".join([
            row["method"], *[f"{row[metric]:.3f}" for metric in METRICS]
        ]) + " |")
    lines.extend(["", f"Mean reranker inference: {analysis['mean_rerank_ms_per_query']:.1f} ms/query.",
                  "", "Candidate coverage and paired intervals are in the CSV files.",
                  "", analysis["interpretation_limit"], ""])
    paths["summary"] = report_dir / "summary.md"
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")
    return paths
