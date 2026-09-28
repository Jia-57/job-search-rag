"""Source-evidence retrieval scoring over frozen query labels.

These metrics only ask whether the originating evidence node is returned. They
do not measure recall of all relevant JDs or answer quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .retrievers import METHODS


QUERY_TYPES = ("high_lexical_overlap", "low_lexical_overlap")


def score_one(
    ranked_node_ids: Sequence[str],
    evidence_node_ids: Sequence[str],
    *,
    top_k: int = 5,
) -> tuple[int, float, int | None]:
    """Return Source Evidence Hit, reciprocal rank, and first gold rank."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    gold = set(evidence_node_ids)
    if not gold:
        raise ValueError("Source evidence labels must be nonempty")
    for rank, node_id in enumerate(ranked_node_ids[:top_k], start=1):
        if node_id in gold:
            return 1, 1.0 / rank, rank
    return 0, 0.0, None


def _unique_rows(rows: Sequence[Mapping[str, Any]], key: str, name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} row missing {key}")
        if value in indexed:
            raise ValueError(f"Duplicate {name} {key}: {value}")
        indexed[value] = row
    return indexed


def evaluate_results(
    results: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Evaluate a complete three-method run against source-only qrels.

    A missing method, duplicate result, inconsistent frozen gold set, or mixed
    run fails fast instead of silently changing a denominator.
    """

    if top_k != 5:
        raise ValueError("V2 metrics are fixed at Top-5")
    query_by_id = _unique_rows(queries, "query_id", "query")
    qrel_by_id = _unique_rows(qrels, "query_id", "qrel")
    if not query_by_id or set(query_by_id) != set(qrel_by_id):
        raise ValueError("Queries and qrels must contain exactly the same query IDs")
    gold_by_id: dict[str, list[str]] = {}
    for query_id, qrel in qrel_by_id.items():
        gold = qrel.get("evidence_node_ids")
        if not isinstance(gold, list) or not gold or any(not isinstance(value, str) or not value for value in gold):
            raise ValueError(f"Invalid or empty source evidence IDs for {query_id}")
        if len(gold) != len(set(gold)):
            raise ValueError(f"Duplicate source evidence IDs for {query_id}")
        query_gold = query_by_id[query_id].get("evidence_node_ids")
        if query_gold is not None and set(query_gold) != set(gold):
            raise ValueError(f"Query and qrel source evidence IDs disagree for {query_id}")
        gold_by_id[query_id] = gold

    expected = {(query_id, method) for query_id in query_by_id for method in METHODS}
    observed: set[tuple[str, str]] = set()
    run_ids: set[str] = set()
    config_hashes: set[str] = set()
    scores: list[dict[str, Any]] = []
    for result in results:
        query_id = result.get("query_id")
        method = result.get("method")
        key = (query_id, method)
        if key not in expected:
            raise ValueError(f"Unexpected result key: {key}")
        if key in observed:
            raise ValueError(f"Duplicate result key: {key}")
        observed.add(key)
        run_id = result.get("run_id")
        config_hash = result.get("config_hash")
        if not isinstance(run_id, str) or not run_id or not isinstance(config_hash, str) or not config_hash:
            raise ValueError("Each result needs run_id and config_hash")
        run_ids.add(run_id)
        config_hashes.add(config_hash)
        ranked = result.get("ranked_node_ids")
        if not isinstance(ranked, list) or any(not isinstance(node_id, str) or not node_id for node_id in ranked):
            raise ValueError(f"Invalid node ranking for {key}")
        if len(ranked) != len(set(ranked)):
            raise ValueError(f"Duplicate node ID in ranking for {key}")
        ranked_details = result.get("ranked_results")
        if ranked_details is not None:
            if [item.get("node_id") for item in ranked_details] != ranked:
                raise ValueError(f"Rank detail/ID mismatch for {key}")
            if [item.get("rank") for item in ranked_details] != list(range(1, len(ranked) + 1)):
                raise ValueError(f"Noncontiguous ranks for {key}")
        hit, rr, first_rank = score_one(ranked, gold_by_id[str(query_id)], top_k=top_k)
        query = query_by_id[str(query_id)]
        if query.get("query_type") not in QUERY_TYPES:
            raise ValueError(f"Invalid query type for {query_id}")
        if not isinstance(query.get("pair_id"), str) or not query.get("pair_id"):
            raise ValueError(f"Missing pair_id for {query_id}")
        scores.append({
            "run_id": run_id,
            "config_hash": config_hash,
            "query_id": query_id,
            "pair_id": query["pair_id"],
            "query_type": query["query_type"],
            "source_job_family": query.get("source_job_family"),
            "method": method,
            "source_evidence_hit_at_5": hit,
            "source_evidence_rr_at_5": rr,
            "first_source_evidence_rank": first_rank,
            "top5_node_ids": ranked[:top_k],
            "evidence_node_ids": gold_by_id[str(query_id)],
        })
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"Incomplete result grid; missing {len(missing)} query/method rows, e.g. {missing[:3]}")
    if len(run_ids) != 1 or len(config_hashes) != 1:
        raise ValueError("Cannot evaluate results from multiple runs or configurations")
    return sorted(scores, key=lambda row: (str(row["query_id"]), METHODS.index(str(row["method"]))))
