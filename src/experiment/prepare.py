"""Turn cleaned job descriptions into stable, offset-preserving retrieval nodes.

Only visible heading lines in ``description_clean`` establish section boundaries.
The text of a heading remains in its section's first chunk, so every character
of the source description is represented in at least one node. Offsets always
refer to the unchanged ``description_clean`` string, never to retrieval text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analysis.sections import HeadingNormalizer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HEADINGS = _REPO_ROOT / "config" / "section_headings.yaml"
_MARKDOWN_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•][ \t]+|\d+[.)][ \t]+)")
_WORD = re.compile(r"\b[\w]+(?:[’'-][\w]+)*\b", re.UNICODE)


@dataclass(frozen=True)
class NodeConfig:
    """Fixed character splitting and heading rules for one benchmark version."""

    chunk_size: int = 1000
    chunk_overlap: int = 100
    heading_config: str | Path | None = None
    include_heading_in_retrieval: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise ValueError("chunk_size must be a positive integer")
        if isinstance(self.chunk_overlap, bool) or not isinstance(self.chunk_overlap, int):
            raise ValueError("chunk_overlap must be a non-negative integer")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
        if not isinstance(self.include_heading_in_retrieval, bool):
            raise ValueError("include_heading_in_retrieval must be boolean")
        if self.heading_config is not None and not isinstance(self.heading_config, (str, Path)):
            raise ValueError("heading_config must be a path")


def _config(value: NodeConfig | Mapping[str, Any] | None) -> NodeConfig:
    if value is None:
        return NodeConfig()
    if isinstance(value, NodeConfig):
        return value
    return NodeConfig(**value)


def _field(job: Mapping[str, Any] | object, name: str) -> Any:
    if isinstance(job, Mapping):
        return job[name]
    return getattr(job, name)


def _heading(line: str, normalizer: HeadingNormalizer) -> tuple[str, str] | None:
    """Recognize Markdown, known aliases, or explicitly styled plain lines.

    A short line ending in a colon or a short all-caps line is visible heading
    syntax even if the alias file cannot classify it. Ordinary short sentences
    and bullet items never become headings merely because they look topical.
    """
    if _BULLET.match(line):
        return None
    markdown = _MARKDOWN_HEADING.match(line)
    if markdown:
        raw = re.sub(r"[ \t]+#+[ \t]*$", "", markdown.group(1)).strip()
        if raw:
            return raw, normalizer.classify(raw) or "unmapped"
        return None

    raw = line.strip()
    if not raw or len(raw) > 120:
        return None
    known = normalizer.classify(raw)
    if known:
        return raw, known

    words = _WORD.findall(raw)
    if not words or "://" in raw:
        return None
    colon_heading = raw.endswith(":") and len(raw) <= 100 and len(words) <= 12
    uppercase_heading = (
        raw.isupper() and not raw.endswith(".") and len(raw) <= 80
        and 2 <= len(words) <= 10 and sum(char.isalpha() for char in raw) >= 4
    )
    if colon_heading or uppercase_heading:
        return raw, "unmapped"
    return None


def _sections(
    text: str, normalizer: HeadingNormalizer
) -> Iterable[tuple[int, int, str | None, str]]:
    """Yield disjoint, exhaustive character ranges; each heading starts a range."""
    start = 0
    cursor = 0
    heading: str | None = None
    canonical = "unsectioned"
    for line in text.splitlines(keepends=True):
        candidate = _heading(line.rstrip("\r\n"), normalizer)
        if candidate is not None:
            if cursor > start:
                yield start, cursor, heading, canonical
            start = cursor
            heading, canonical = candidate
        cursor += len(line)
    if start < len(text):
        yield start, len(text), heading, canonical


def _split_end(text: str, start: int, section_end: int, chunk_size: int) -> int:
    """Prefer paragraph, line, then word boundaries near the target size."""
    limit = min(start + chunk_size, section_end)
    if limit == section_end:
        return limit
    lower = start + max(1, chunk_size // 2)
    for separator in ("\n\n", "\n", " ", "\t"):
        position = text.rfind(separator, lower, limit)
        if position >= 0:
            return position + len(separator)
    return limit


def _chunk_ranges(
    text: str, section_start: int, section_end: int, config: NodeConfig
) -> Iterable[tuple[int, int]]:
    start = section_start
    while start < section_end:
        end = _split_end(text, start, section_end, config.chunk_size)
        yield start, end
        if end == section_end:
            break
        next_start = max(start + 1, end - config.chunk_overlap)
        # A nearby preceding boundary keeps overlap from starting mid-word.
        if config.chunk_overlap and next_start > start + 1:
            boundary = max(
                text.rfind(" ", max(start + 1, next_start - 24), next_start),
                text.rfind("\n", max(start + 1, next_start - 24), next_start),
                text.rfind("\t", max(start + 1, next_start - 24), next_start),
            )
            if boundary >= start + 1:
                next_start = boundary + 1
        start = next_start


def _node_id(job_id: str, start: int, end: int, content: str) -> str:
    encoded = json.dumps(
        ["jd_v2_node_v1", job_id, start, end, content],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_nodes(
    jobs: Iterable[Mapping[str, Any] | object],
    config: NodeConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic nodes from Job objects or equivalent mappings.

    Jobs are sorted by source ID, so input iteration order cannot alter output.
    A repeated ID, empty description, or missing required field fails loudly.
    ``content_text == description_clean[start_offset:end_offset]`` for every
    node. The union of node ranges covers every source character.
    """
    settings = _config(config)
    heading_path = Path(settings.heading_config or _DEFAULT_HEADINGS)
    if not heading_path.is_absolute():
        heading_path = _REPO_ROOT / heading_path
    normalizer = HeadingNormalizer.from_yaml(heading_path)

    by_id: dict[str, Mapping[str, Any] | object] = {}
    for job in jobs:
        job_id = _field(job, "id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("every job needs a non-empty string id")
        if job_id in by_id:
            raise ValueError(f"duplicate job id: {job_id}")
        by_id[job_id] = job

    nodes: list[dict[str, Any]] = []
    for job_id in sorted(by_id):
        job = by_id[job_id]
        description = _field(job, "description_clean")
        if not isinstance(description, str) or not description:
            raise ValueError(f"job {job_id} has an empty description_clean")
        metadata = {
            "job_id": job_id,
            "job_family": _field(job, "job_family"),
            "company": _field(job, "company"),
            "title": _field(job, "title"),
        }
        for section_start, section_end, heading, canonical in _sections(description, normalizer):
            for chunk_index, (start, end) in enumerate(
                _chunk_ranges(description, section_start, section_end, settings)
            ):
                content = description[start:end]
                retrieval = (
                    f"Section: {heading}\n{content}"
                    if settings.include_heading_in_retrieval and heading else content
                )
                nodes.append({
                    "node_id": _node_id(job_id, start, end, content),
                    **metadata,
                    "section_heading": heading,
                    "canonical_section": canonical,
                    "chunk_index": chunk_index,
                    "start_offset": start,
                    "end_offset": end,
                    "content_text": content,
                    "retrieval_text": retrieval,
                })
    return nodes


def write_nodes(nodes: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write nodes in their supplied deterministic order as UTF-8 JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for node in nodes:
            handle.write(json.dumps(node, ensure_ascii=False, separators=(",", ":")) + "\n")
    return output
