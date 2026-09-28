"""Static integrity checks for frozen source-evidence benchmark records."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .retrievers import tokenize_bm25


QUERY_TYPES = ("high_lexical_overlap", "low_lexical_overlap")
FAMILIES = (
    "software_engineering", "data_science_analytics",
    "ai_engineering", "product_management",
)
_NUMERIC = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")
_URL = re.compile(r"(?:https?://|www\.|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+/[^\s]+)", re.I)
_LEXICAL_STOP = frozenset(
    "a an and are as at be by can do for from have in is it of on or our the "
    "their this to with will you your we which what where who role saved job "
    "requires require requiring".split()
)
_COMPANY_SUFFIXES = frozenset({
    "inc", "llc", "ltd", "limited", "corp", "corporation", "company",
    "group", "gmbh", "ag", "plc", "co", "the",
})
_TITLE_MODIFIERS = re.compile(
    r"^(?:(?:senior|sr|junior|jr|principal|staff|lead|associate|"
    r"mid.level|entry.level)\s+)+",
    re.I,
)


def _as_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    raise TypeError(f"expected mapping or Pydantic model, got {type(record)!r}")


def _has_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.strip()
    return bool(phrase) and bool(re.search(
        rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I,
    ))


def _lexical_terms(text: str) -> set[str]:
    return {
        term for term in tokenize_bm25(text)
        if term not in _LEXICAL_STOP
    }


def _source_leaks(query: str, job: Mapping[str, Any]) -> list[str]:
    leaks: list[str] = []
    if _URL.search(query):
        leaks.append("url")
    for field in ("company", "title", "id", "source_job_id"):
        value = str(job.get(field) or "").strip()
        if value and _has_phrase(query, value):
            leaks.append(field)
    company_words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", str(job.get("company") or ""))
    company_core = [
        word for word in company_words
        if len(word) >= 4 and word.casefold() not in _COMPANY_SUFFIXES
    ]
    if any(_has_phrase(query, word) for word in company_core):
        leaks.append("company")
    title = str(job.get("title") or "")
    title_core = _TITLE_MODIFIERS.sub("", title).split(",", 1)[0].strip()
    title_core = re.sub(r"\s+(?:I|II|III|IV|V)$", "", title_core)
    if title_core and title_core.casefold() != title.casefold() and _has_phrase(query, title_core):
        leaks.append("title")
    source_url = str(job.get("url") or "")
    if source_url and source_url.casefold() in query.casefold():
        leaks.append("url")
    hostname = urlsplit(source_url).hostname
    if hostname and re.search(rf"(?<![\w.]){re.escape(hostname)}(?![\w.])", query, re.I):
        leaks.append("url")
    return sorted(set(leaks))


def static_pair_errors(
    generated: Mapping[str, Any], evidence_text: str, source_job: Mapping[str, Any]
) -> list[str]:
    """Cheap, deterministic checks before the separate semantic model prompt."""
    errors: list[str] = []
    high = generated.get("high_query")
    low = generated.get("low_query")
    anchor = generated.get("anchor_term")
    if any(not isinstance(value, str) or not value.strip() for value in (high, low, anchor)):
        return ["missing_or_non_string_query_or_anchor"]
    assert isinstance(high, str) and isinstance(low, str) and isinstance(anchor, str)
    high, low, anchor = high.strip(), low.strip(), anchor.strip()
    if not 20 <= len(high) <= 320 or not 20 <= len(low) <= 320:
        errors.append("query_length")
    if high.casefold() == low.casefold():
        errors.append("identical_questions")
    if (len(anchor) < 3 or len(anchor) > 80 or len(anchor.split()) > 3
            or not _has_phrase(evidence_text, anchor)):
        errors.append("anchor_absent_or_invalid")
    if not _has_phrase(high, anchor):
        errors.append("high_missing_anchor")
    if _has_phrase(low, anchor):
        errors.append("low_repeats_anchor")
    evidence_terms = _lexical_terms(evidence_text)
    if len(_lexical_terms(high) & evidence_terms) <= len(
        _lexical_terms(low) & evidence_terms
    ):
        errors.append("low_not_lower_lexical_overlap")
    normalized_evidence = re.sub(r"\s+", " ", evidence_text).strip().casefold()
    for label, query in (("high", high), ("low", low)):
        if not query.endswith("?"):
            errors.append(f"{label}_not_question")
        if normalized_evidence and normalized_evidence in re.sub(r"\s+", " ", query).casefold():
            errors.append(f"{label}_copies_whole_evidence")
        for leak in _source_leaks(query, source_job):
            errors.append(f"{label}_leaks_{leak}")
        source_numbers = set(_NUMERIC.findall(evidence_text))
        missing_numbers = source_numbers - set(_NUMERIC.findall(query))
        if missing_numbers:
            errors.append(f"{label}_drops_numeric_constraint:{','.join(sorted(missing_numbers))}")
    return errors


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    pair_count: int
    query_count: int
    qrel_count: int
    source_jobs_by_family: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            sample = "\n".join(f"- {error}" for error in self.errors[:20])
            raise ValueError(f"benchmark integrity check failed ({len(self.errors)} errors):\n{sample}")


def validate_benchmark(
    jobs: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int | None = None,
    expected_per_family: int | None = None,
    strict: bool = True,
) -> ValidationReport:
    """Validate source offsets, complete gold sets, pair linkage and leakage.

    The checks use only the frozen corpus and nodes. Retrieval scores must never
    enter this function. Semantic equality is established by the independent
    model validation recorded during benchmark construction.
    """
    errors: list[str] = []
    job_by_id: dict[str, Mapping[str, Any]] = {}
    for job in jobs:
        row = _as_mapping(job)
        job_by_id[str(row["id"])] = row
    nodes_by_job: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        row = _as_mapping(node)
        nodes_by_job[str(row["job_id"])].append(row)
        job = job_by_id.get(str(row["job_id"]))
        if job is None:
            errors.append(f"node {row['node_id']}: unknown source job")
            continue
        start, end = int(row["start_offset"]), int(row["end_offset"])
        description = str(job["description_clean"])
        if (
            not 0 <= start < end <= len(description)
            or row.get("content_text") != description[start:end]
        ):
            errors.append(f"node {row['node_id']}: content/offset mismatch")
    query_by_id: dict[str, Mapping[str, Any]] = {}
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for query in queries:
        row = _as_mapping(query)
        query_id = str(row.get("query_id", ""))
        pair_id = str(row.get("pair_id", ""))
        query_type = str(row.get("query_type", ""))
        if not query_id or query_id in query_by_id:
            errors.append(f"empty or duplicate query_id: {query_id!r}")
        query_by_id[query_id] = row
        if query_type not in QUERY_TYPES or query_type in pairs[pair_id]:
            errors.append(f"invalid or duplicate query type for {pair_id}: {query_type}")
        pairs[pair_id][query_type] = row
    qrel_by_id: dict[str, Mapping[str, Any]] = {}
    for qrel in qrels:
        row = _as_mapping(qrel)
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in qrel_by_id:
            errors.append(f"empty or duplicate qrel query_id: {query_id!r}")
        qrel_by_id[query_id] = row
    if set(query_by_id) != set(qrel_by_id):
        errors.append("query_id sets in queries and qrels differ")
    if expected_pairs is not None and len(pairs) != expected_pairs:
        errors.append(f"expected {expected_pairs} pairs, found {len(pairs)}")
    if len(queries) != 2 * len(pairs):
        errors.append("each pair must have exactly two queries")
    source_jobs_by_family: dict[str, set[str]] = defaultdict(set)
    seen_source_jobs: set[str] = set()
    for pair_id, typed in sorted(pairs.items()):
        if not pair_id or set(typed) != set(QUERY_TYPES):
            errors.append(f"{pair_id!r} lacks one high/low query")
            continue
        high, low = (typed[query_type] for query_type in QUERY_TYPES)
        shared_fields = (
            "source_job_id", "source_job_family", "evidence_text",
            "evidence_start", "evidence_end", "anchor_term", "evidence_node_ids",
        )
        for field in shared_fields:
            if high.get(field) != low.get(field):
                errors.append(f"{pair_id}: high/low disagree on {field}")
        source_job_id = str(high.get("source_job_id", ""))
        if source_job_id in seen_source_jobs:
            errors.append(f"{pair_id}: source JD selected more than once")
        seen_source_jobs.add(source_job_id)
        job = job_by_id.get(source_job_id)
        if job is None:
            errors.append(f"{pair_id}: unknown source job")
            continue
        family = str(high.get("source_job_family", ""))
        if family != job["job_family"]:
            errors.append(f"{pair_id}: source family mismatch")
        source_jobs_by_family[family].add(source_job_id)
        evidence = high.get("evidence_text")
        start, end = high.get("evidence_start"), high.get("evidence_end")
        description = str(job["description_clean"])
        if (
            not isinstance(evidence, str) or not evidence
            or not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool)
            or not 0 <= start < end <= len(description)
        ):
            errors.append(f"{pair_id}: invalid evidence or offsets")
            continue
        if description[start:end] != evidence or description.count(evidence) != 1:
            errors.append(f"{pair_id}: evidence not uniquely present at exact offsets")
        expected_node_ids = sorted({
            str(node["node_id"]) for node in nodes_by_job[source_job_id]
            if int(node["start_offset"]) <= start and int(node["end_offset"]) >= end
        })
        if not expected_node_ids:
            errors.append(f"{pair_id}: no node completely contains evidence")
        if sorted(high.get("evidence_node_ids") or []) != expected_node_ids:
            errors.append(f"{pair_id}: gold IDs are not all and only enclosing nodes")
        for query_type, row in typed.items():
            query_id = str(row.get("query_id", ""))
            qrel = qrel_by_id.get(query_id)
            if qrel is None:
                errors.append(f"{pair_id}: missing qrel for {query_id}")
                continue
            if qrel.get("pair_id") != pair_id:
                errors.append(f"{pair_id}: qrel pair_id mismatch")
            if str(qrel.get("source_job_id", "")) != source_job_id:
                errors.append(f"{pair_id}: qrel source job mismatch")
            if sorted(qrel.get("evidence_node_ids") or []) != expected_node_ids:
                errors.append(f"{pair_id}: qrel gold IDs mismatch")
            if row.get("validation_status") != "passed":
                errors.append(f"{query_id}: validation not passed")
            semantic = row.get("semantic_validation")
            if not isinstance(semantic, Mapping) or any(
                semantic.get(flag) is not True for flag in (
                    "role_related_atomic_fact", "self_contained_evidence",
                    "same_information_need", "high_supported", "low_supported",
                    "constraints_preserved", "natural_questions",
                    "low_is_conceptual_paraphrase",
                )
            ):
                errors.append(f"{query_id}: missing or failed semantic validation")
            if not isinstance(row.get("query"), str) or not row["query"].strip():
                errors.append(f"{query_id}: empty question")
        generated = {
            "high_query": high.get("query"), "low_query": low.get("query"),
            "anchor_term": high.get("anchor_term"),
        }
        errors.extend(f"{pair_id}: {error}" for error in static_pair_errors(generated, evidence, job))
    counts = {family: len(source_jobs_by_family.get(family, set())) for family in FAMILIES}
    if expected_per_family is not None:
        for family, count in counts.items():
            if count != expected_per_family:
                errors.append(f"{family}: expected {expected_per_family} source jobs, found {count}")
    report = ValidationReport(tuple(errors), len(pairs), len(queries), len(qrels), counts)
    if strict:
        report.require_valid()
    return report
