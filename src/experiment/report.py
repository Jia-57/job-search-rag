"""Write reproducible V2 tables and source-evidence failure cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate_retrieval import evaluate_results
from .retrievers import METHODS


MAIN_COLUMNS = (
    "method",
    "pair_count",
    "high_source_evidence_hit_at_5",
    "low_source_evidence_hit_at_5",
    "hit_drop",
    "high_source_evidence_mrr_at_5",
    "low_source_evidence_mrr_at_5",
    "mrr_drop",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _short_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _failure_cases(
    results: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    *,
    failure_limit: int,
) -> list[dict[str, Any]]:
    query_by_id = {str(row["query_id"]): row for row in queries}
    node_by_id = {str(row["node_id"]): row for row in nodes}
    result_by_key = {(str(row["query_id"]), str(row["method"])): row for row in results}
    score_by_key = {(str(row["query_id"]), str(row["method"])): row for row in scores}
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for query_id, query in query_by_id.items():
        hits = [score_by_key[(query_id, method)]["source_evidence_hit_at_5"] for method in METHODS]
        if all(hits):
            continue
        is_low = query.get("query_type") == "low_lexical_overlap"
        priority = 0 if not any(hits) and is_low else 1 if not any(hits) else 2 if is_low else 3
        top5: dict[str, list[dict[str, Any]]] = {}
        diagnoses: dict[str, str] = {}
        gold_ids = set(score_by_key[(query_id, METHODS[0])]["evidence_node_ids"])
        source_job_id = str(query["source_job_id"])
        for method in METHODS:
            result = result_by_key[(query_id, method)]
            ranked_ids = result["ranked_node_ids"]
            if any(node_id not in node_by_id for node_id in ranked_ids):
                raise ValueError(f"Result references an unknown node ID: {query_id}/{method}")
            gold_ranks = [rank for rank, node_id in enumerate(ranked_ids, 1)
                          if node_id in gold_ids]
            source_ranks = [rank for rank, node_id in enumerate(ranked_ids, 1)
                            if node_by_id[node_id]["job_id"] == source_job_id]
            if gold_ranks and gold_ranks[0] <= 5:
                diagnoses[method] = f"Source evidence found at rank {gold_ranks[0]}."
            elif gold_ranks:
                diagnoses[method] = (
                    f"Source evidence found at rank {gold_ranks[0]}, outside Top-5."
                )
            elif source_ranks and source_ranks[0] <= 5:
                diagnoses[method] = (
                    "Top-5 contains a different chunk from the source JD; "
                    "the labeled evidence is absent from Top-20."
                )
            elif source_ranks:
                diagnoses[method] = (
                    f"The source JD first appears at rank {source_ranks[0]}, "
                    "but the labeled evidence is absent from Top-20."
                )
            else:
                diagnoses[method] = "The source JD is absent from Top-20."
            rows: list[dict[str, Any]] = []
            for rank, node_id in enumerate(ranked_ids[:5], start=1):
                if node_id not in node_by_id:
                    raise ValueError(f"Result references unknown node ID: {node_id}")
                node = node_by_id[node_id]
                rows.append({
                    "rank": rank,
                    "node_id": node_id,
                    "job_id": node.get("job_id"),
                    "title": node.get("title"),
                    "company": node.get("company"),
                    "section_heading": node.get("section_heading"),
                    "excerpt": _short_text(node.get("content_text")),
                })
            top5[method] = rows
        candidates.append((priority, query_id, {
            "query_id": query_id,
            "pair_id": query.get("pair_id"),
            "query_type": query.get("query_type"),
            "query": query.get("query"),
            "source_job_id": query.get("source_job_id"),
            "evidence_text": query.get("evidence_text"),
            "evidence_node_ids": score_by_key[(query_id, METHODS[0])]["evidence_node_ids"],
            "method_hits": {method: score_by_key[(query_id, method)]["source_evidence_hit_at_5"] for method in METHODS},
            "method_diagnoses": diagnoses,
            "top5": top5,
            "judgment": (
                "A method is counted as a source-evidence miss when none of its Top-5 "
                "node IDs belong to the frozen source evidence ID set. Other retrieved "
                "JDs may be relevant but are not labeled by this benchmark."
            ),
        }))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates[:failure_limit]]


def _render_failures(cases: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# V2 source-evidence retrieval failure cases",
        "",
        "These cases use frozen source evidence labels. A different retrieved JD may also answer the query; the benchmark does not label that possibility.",
        "",
    ]
    if not cases:
        lines.extend(("No source-evidence misses were found in this run.", ""))
    for index, case in enumerate(cases, start=1):
        lines.extend((
            f"## {index}. {case['query_id']} ({case['query_type']})",
            "",
            f"- Pair: `{case['pair_id']}`; source JD: `{case['source_job_id']}`",
            f"- Query: {json.dumps(case['query'], ensure_ascii=False)}",
            f"- Source evidence: {json.dumps(case['evidence_text'], ensure_ascii=False)}",
            f"- Source evidence node IDs: {', '.join(f'`{node_id}`' for node_id in case['evidence_node_ids'])}",
            f"- Judgment: {case['judgment']}",
            "",
        ))
        for method in METHODS:
            status = "hit" if case["method_hits"][method] else "miss"
            lines.extend((f"### {method} — {status}", "", case["method_diagnoses"][method], ""))
            for item in case["top5"][method]:
                title = _short_text(item["title"], limit=75)
                company = _short_text(item["company"], limit=55)
                section = _short_text(item["section_heading"], limit=50)
                lines.append(
                    f"{item['rank']}. `{item['node_id']}` | JD `{item['job_id']}` | "
                    f"{title} ({company}), section: {section}. "
                    f"Excerpt: {json.dumps(item['excerpt'], ensure_ascii=False)}"
                )
            lines.append("")
    return "\n".join(lines)


def _render_summary(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# JD evidence retrieval experiment V2",
        "",
        f"Run: `{analysis['run_id']}`; pairs: {analysis['pair_count']}; queries: {analysis['query_count']}.",
        "",
        "| Method | High source Hit@5 | Low source Hit@5 | Hit drop | High source MRR@5 | Low source MRR@5 | MRR drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["main_table"]:
        lines.append(
            f"| {row['method']} | {row['high_source_evidence_hit_at_5']:.3f} | "
            f"{row['low_source_evidence_hit_at_5']:.3f} | {row['hit_drop']:.3f} | "
            f"{row['high_source_evidence_mrr_at_5']:.3f} | "
            f"{row['low_source_evidence_mrr_at_5']:.3f} | {row['mrr_drop']:.3f} |"
        )
    lines.extend((
        "",
        "Positive drop means the low-overlap rewrite scored worse than the high-overlap query for the same source fact.",
        "",
        "## Interpretation limit",
        "",
        str(analysis["interpretation_limit"]),
        "",
        "Bootstrap intervals resample pair IDs, preserving both query variants and all three methods in every draw. Family-level tables are descriptive; each family has a small sample.",
        "",
    ))
    return "\n".join(lines)


def write_reports(
    analysis: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    failure_limit: int = 12,
) -> dict[str, Path]:
    """Write paper tables and diagnostic cases from a complete frozen run."""

    if failure_limit < 0:
        raise ValueError("failure_limit cannot be negative")
    scores = evaluate_results(results, qrels, queries, top_k=5)
    if (
        analysis.get("run_id") != scores[0]["run_id"]
        or analysis.get("config_hash") != scores[0]["config_hash"]
        or analysis.get("query_count") != len(queries)
        or analysis.get("pair_count") != len(queries) // 2
    ):
        raise ValueError("Analysis does not match the retrieval results and benchmark")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tables: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("main_table", "main_table.csv", MAIN_COLUMNS),
        ("pair_outcomes", "pair_transitions.csv", (
            "pair_id", "method", "source_job_family", "high_query_id", "low_query_id",
            "high_source_evidence_hit_at_5", "low_source_evidence_hit_at_5", "transition",
            "high_source_evidence_mrr_at_5", "low_source_evidence_mrr_at_5", "hit_drop", "mrr_drop",
        )),
        ("transition_summary", "transition_summary.csv", ("method", "transition", "count", "fraction")),
        ("bootstrap_intervals", "bootstrap_intervals.csv", (
            "method", "metric", "estimate", "ci_lower", "ci_upper", "confidence", "bootstrap_samples", "pair_count",
        )),
        ("method_drop_differences", "method_drop_differences.csv", (
            "method_a", "method_b", "metric", "drop_difference", "ci_lower", "ci_upper", "confidence", "bootstrap_samples", "pair_count",
        )),
        ("family_descriptive", "family_descriptive.csv", (
            "method", "source_job_family", "pair_count", *MAIN_COLUMNS[2:],
        )),
    )
    for key, filename, columns in tables:
        path = output / filename
        _write_csv(path, analysis[key], columns)
        paths[key] = path

    score_path = output / "per_query_scores.jsonl"
    with score_path.open("w", encoding="utf-8") as stream:
        for score in scores:
            stream.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
    paths["per_query_scores"] = score_path

    cases = _failure_cases(results, scores, queries, nodes, failure_limit=failure_limit)
    failure_path = output / "failure_cases.md"
    failure_path.write_text(_render_failures(cases), encoding="utf-8")
    paths["failure_cases"] = failure_path
    summary_path = output / "summary.md"
    summary_path.write_text(_render_summary(analysis), encoding="utf-8")
    paths["summary"] = summary_path

    json_path = output / "analysis.json"
    json_path.write_text(json.dumps({**analysis, "failure_cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["analysis"] = json_path
    return paths
