"""Extract exact, source-traceable JD facts before any retrieval is run.

The rules here only propose candidates. The independent model check in
``build_benchmark`` decides whether a candidate can support a question pair.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Sequence


_LINE = re.compile(r"[^\n]+")
_BULLET = re.compile(r"^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)")
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+.#/\-]*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do for from have in is it of on or our the their "
    "this to with will you your we work working role team teams experience skills "
    "ability able knowledge strong excellent required requirements good including "
    "using use develop support build provide manage across ensure within about "
    "position candidate candidates job years year plus must should preferred".split()
)
_BANNED_SECTION = frozenset({"about_company", "benefits", "additional_information"})
_BANNED_HEADING = re.compile(
    r"\b(?:benefits?|perks?|rewards?|compensation|salary|about us|about our "
    r"company|equal opportunity|privacy|legal|how to apply|application process|"
    r"accommodation|diversity|life at)\b",
    re.I,
)
_TEMPLATE = re.compile(
    r"\b(?:equal opportunity employer|reasonable accommodation|all qualified "
    r"applicants|privacy policy|terms and conditions|drug.free workplace|"
    r"competitive salary|comprehensive benefits|401\s?\(?k\)?|paid time off|"
    r"health insurance|click (?:here|apply)|apply now|recruitment process|"
    r"fast.paced environment|excellent communication skills|strong communication "
    r"skills|team player|passion for (?:our|the) mission)\b",
    re.I,
)


def _mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    raise TypeError(f"expected mapping or Pydantic model, got {type(record)!r}")


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOPWORDS
    }


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[end - 1].isspace():
        end -= 1
    return start, end


def _line_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Return whole bullet/line spans, splitting only long lines at sentence ends."""
    spans: list[tuple[int, int]] = []
    for match in _LINE.finditer(text):
        line_start, line_end = _trimmed_span(text, *match.span())
        if line_start == line_end or _MARKDOWN_HEADING.match(text[line_start:line_end]):
            continue
        bullet = _BULLET.match(text[line_start:line_end])
        if bullet:
            line_start += bullet.end()
            line_start, line_end = _trimmed_span(text, line_start, line_end)
        if line_start == line_end:
            continue
        if line_end - line_start <= max_chars:
            spans.append((line_start, line_end))
            continue
        cursor = line_start
        for separator in _SENTENCE_BREAK.finditer(text, line_start, line_end):
            part_start, part_end = _trimmed_span(text, cursor, separator.start())
            if part_start < part_end:
                spans.append((part_start, part_end))
            cursor = separator.end()
        part_start, part_end = _trimmed_span(text, cursor, line_end)
        if part_start < part_end:
            spans.append((part_start, part_end))
    return spans


@dataclass(frozen=True)
class EvidenceConfig:
    min_chars: int = 40
    max_chars: int = 420
    min_words: int = 8
    max_words: int = 70
    max_document_frequency: float = 0.35

    def __post_init__(self) -> None:
        if not 0 < self.min_chars <= self.max_chars:
            raise ValueError("invalid evidence character limits")
        if not 0 < self.min_words <= self.max_words:
            raise ValueError("invalid evidence word limits")
        if not 0 < self.max_document_frequency <= 1:
            raise ValueError("max_document_frequency must be in (0, 1]")


@dataclass
class EvidenceExtractionResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def extract_evidence_candidates(
    jobs: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    config: EvidenceConfig | None = None,
) -> EvidenceExtractionResult:
    """Select exact JD substrings fully enclosed by one or more prepared nodes.

    Candidate ranking never uses BM25, Dense or Hybrid results. Rejections are
    retained with explicit reasons for the benchmark manifest and audit file.
    """
    config = config or EvidenceConfig()
    job_rows = [_mapping(job) for job in jobs]
    node_rows = [_mapping(node) for node in nodes]
    by_job: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in node_rows:
        by_job[str(node["job_id"])].append(node)
    document_frequency: Counter[str] = Counter()
    for job in job_rows:
        document_frequency.update(_terms(str(job["description_clean"])))
    total_jobs = len(job_rows)
    result = EvidenceExtractionResult()
    reasons: Counter[str] = Counter()
    provisional: list[dict[str, Any]] = []

    def reject(job_id: str, start: int, end: int, text: str, reason: str) -> None:
        reasons[reason] += 1
        result.rejections.append({
            "source_job_id": job_id,
            "evidence_start": start,
            "evidence_end": end,
            "evidence_text": text,
            "reason": reason,
        })

    for job in job_rows:
        job_id = str(job["id"])
        description = str(job["description_clean"])
        seen_text: set[str] = set()
        source_nodes = by_job.get(job_id, [])
        for start, end in _line_spans(description, config.max_chars):
            evidence = description[start:end]
            words = _WORD.findall(evidence)
            if not config.min_chars <= len(evidence) <= config.max_chars:
                reject(job_id, start, end, evidence, "length_chars")
                continue
            if not config.min_words <= len(words) <= config.max_words:
                reject(job_id, start, end, evidence, "length_words")
                continue
            if _TEMPLATE.search(evidence):
                reject(job_id, start, end, evidence, "template_or_non_role_fact")
                continue
            company = str(job.get("company") or "").strip()
            if len(company) >= 3 and re.search(
                rf"(?<!\w){re.escape(company)}(?!\w)", evidence, re.I,
            ):
                reject(job_id, start, end, evidence, "mentions_source_company")
                continue
            normalized = re.sub(r"\s+", " ", evidence).strip().casefold()
            if normalized in seen_text or description.count(evidence) != 1:
                reject(job_id, start, end, evidence, "ambiguous_or_duplicate_source_text")
                continue
            seen_text.add(normalized)
            enclosing = [
                node for node in source_nodes
                if int(node["start_offset"]) <= start and int(node["end_offset"]) >= end
            ]
            if not enclosing:
                reject(job_id, start, end, evidence, "no_complete_node")
                continue
            node = enclosing[0]
            section = str(node.get("canonical_section") or "unsectioned")
            heading = str(node.get("section_heading") or "")
            if section in _BANNED_SECTION or (
                section in {"unsectioned", "unmapped"} and _BANNED_HEADING.search(heading)
            ):
                reject(job_id, start, end, evidence, "non_role_section")
                continue
            distinctive = sorted(
                term for term in _terms(evidence)
                if document_frequency[term] / max(total_jobs, 1)
                <= config.max_document_frequency
            )
            if not distinctive and not re.search(r"\b\d+(?:[.+%-]|\b)", evidence):
                reject(job_id, start, end, evidence, "generic_corpus_vocabulary")
                continue
            key = f"{job_id}\x1f{start}\x1f{end}\x1f{evidence}"
            provisional.append({
                "candidate_id": sha256(key.encode("utf-8")).hexdigest()[:20],
                "source_job_id": job_id,
                "source_job_family": str(job["job_family"]),
                "evidence_text": evidence,
                "evidence_start": start,
                "evidence_end": end,
                "evidence_node_ids": sorted({str(node["node_id"]) for node in enclosing}),
                "section_heading": heading or None,
                "canonical_section": section,
                "distinctive_terms": distinctive,
                "distinctiveness_score": round(
                    len(distinctive) / max(len(words) ** 0.5, 1)
                    + (1 if re.search(r"\b\d", evidence) else 0)
                    + (0.5 if section in {"qualifications", "responsibilities"} else 0),
                    3,
                ),
            })

    passage_frequency: Counter[str] = Counter(
        re.sub(r"\s+", " ", row["evidence_text"]).casefold()
        for row in provisional
    )
    for row in provisional:
        normalized = re.sub(r"\s+", " ", row["evidence_text"]).casefold()
        if passage_frequency[normalized] > 3:
            reject(
                row["source_job_id"], row["evidence_start"], row["evidence_end"],
                row["evidence_text"], "repeated_across_jobs",
            )
            continue
        result.candidates.append(row)
    result.candidates.sort(key=lambda row: (row["source_job_family"], row["source_job_id"], row["evidence_start"]))
    result.counts = {
        "candidate_spans": len(provisional) + sum(reasons.values()) - reasons["repeated_across_jobs"],
        "accepted_candidates": len(result.candidates),
        **{f"rejected_{reason}": count for reason, count in sorted(reasons.items())},
    }
    return result
