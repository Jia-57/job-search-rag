"""Conservative, auditable JD heading and section extraction for corpus analysis."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import yaml

from src.preprocessing.normalize import clean_description


_WORD = re.compile(r"\b[\w]+(?:[’'-][\w]+)*\b", re.UNICODE)
_MARKUP = re.compile(r"<\s*(?:h[1-6]|p|div|ul|li|section|strong|b|br)\b", re.I)
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)")
_IGNORED = {"script", "style"}


def word_count(text: str) -> int:
    """Unicode-aware lexical word count, not model-token count."""
    return len(_WORD.findall(text))


def normalize_heading(text: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(text))
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value).strip().strip("# ")
    return value.rstrip(" :–—-.").casefold()


@dataclass(frozen=True)
class SectionType:
    key: str
    label: str
    patterns: tuple[re.Pattern[str], ...]


class HeadingNormalizer:
    def __init__(self, types: tuple[SectionType, ...]) -> None:
        self.types = types

    @classmethod
    def from_yaml(cls, path: Path) -> HeadingNormalizer:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("sections"), dict):
            raise ValueError("section heading config needs a sections mapping")
        types: list[SectionType] = []
        for key, definition in payload["sections"].items():
            if not isinstance(definition, dict) or not isinstance(definition.get("label"), str):
                raise ValueError(f"invalid section type: {key}")
            patterns = definition.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise ValueError(f"section type {key} needs heading patterns")
            types.append(SectionType(
                key=key, label=definition["label"],
                patterns=tuple(re.compile(pattern, re.I) for pattern in patterns),
            ))
        return cls(tuple(types))

    def classify(self, heading: str) -> str | None:
        normalized = normalize_heading(heading)
        return next(
            (section.key for section in self.types
             if any(pattern.fullmatch(normalized) for pattern in section.patterns)),
            None,
        )


@dataclass(frozen=True)
class TextBlock:
    kind: str
    text: str
    level: int = 0
    strong_only: bool = False


class _BlockParser(HTMLParser):
    """Retain visible blocks and explicit h-tags / standalone bold paragraphs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[TextBlock] = []
        self._active_tag: str | None = None
        self._parts: list[tuple[str, bool]] = []
        self._loose: list[str] = []
        self._strong_depth = 0
        self._ignored_depth = 0

    def _finish_active(self) -> None:
        if self._active_tag is None:
            return
        text = re.sub(r"\s+", " ", "".join(part for part, _ in self._parts)).strip()
        if text:
            nonstrong = "".join(part for part, strong in self._parts if not strong)
            strong_only = bool(self._parts) and bool(
                re.sub(r"[\s:–—-]", "", "".join(
                    part for part, strong in self._parts if strong
                ))
            ) and not re.sub(r"[\s:–—-]", "", nonstrong)
            kind = "heading" if re.fullmatch(r"h[1-6]", self._active_tag) else (
                "bullet" if self._active_tag == "li" else "paragraph"
            )
            # Some ATS templates encode every list item as a bold <p>- item</p>.
            # Such paragraphs are bullets, never section headings.
            if kind == "paragraph" and _BULLET.match(text):
                kind = "bullet"
                text = _BULLET.sub("", text)
            level = int(self._active_tag[1]) if kind == "heading" else 0
            self.blocks.append(TextBlock(kind, text, level, strong_only))
        self._active_tag = None
        self._parts.clear()

    def _finish_loose(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._loose)).strip()
        if text:
            self.blocks.append(TextBlock("paragraph", text))
        self._loose.clear()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _IGNORED:
            self._finish_active()
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in {"strong", "b"}:
            self._strong_depth += 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._finish_active()
            self._finish_loose()
            self._active_tag = tag
        elif tag == "br":
            if self._active_tag is not None:
                self._parts.append((" ", False))
            else:
                self._loose.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED:
            self._ignored_depth = max(self._ignored_depth - 1, 0)
            return
        if self._ignored_depth:
            return
        if tag in {"strong", "b"}:
            self._strong_depth = max(self._strong_depth - 1, 0)
        elif tag == self._active_tag:
            self._finish_active()
        elif tag in {"div", "section", "article", "td", "tr"} and self._active_tag is None:
            self._finish_loose()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._active_tag is None:
            self._loose.append(data)
        else:
            self._parts.append((data, self._strong_depth > 0))

    def finish(self) -> list[TextBlock]:
        self._finish_active()
        self._finish_loose()
        return self.blocks


def _blocks(raw: str) -> list[TextBlock]:
    decoded = raw
    for _ in range(2):
        if _MARKUP.search(decoded):
            break
        decoded = html.unescape(decoded)
    if _MARKUP.search(decoded):
        parser = _BlockParser()
        parser.feed(decoded)
        parser.close()
        return parser.finish()
    blocks: list[TextBlock] = []
    for line in clean_description(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        heading = _MARKDOWN_HEADING.match(line)
        if heading:
            blocks.append(TextBlock("heading", heading.group(2), len(heading.group(1))))
        else:
            blocks.append(TextBlock("bullet" if _BULLET.match(line) else "paragraph",
                                    _BULLET.sub("", line)))
    return blocks


@dataclass(frozen=True)
class Section:
    raw_heading: str
    canonical_type: str | None
    heading_level: int
    body: str
    word_count: int
    bullet_count: int


@dataclass(frozen=True)
class ExtractedStructure:
    sections: tuple[Section, ...]
    preamble_word_count: int


@dataclass
class _SectionBuilder:
    raw_heading: str
    canonical_type: str | None
    heading_level: int
    body_parts: list[str] = field(default_factory=list)
    bullet_count: int = 0

    def finish(self) -> Section:
        body = "\n".join(self.body_parts)
        return Section(self.raw_heading, self.canonical_type, self.heading_level,
                       body, word_count(body), self.bullet_count)


def extract_sections(raw: str, normalizer: HeadingNormalizer) -> ExtractedStructure:
    """Split only at explicit headings or short paragraphs matching known aliases."""
    sections: list[Section] = []
    current: _SectionBuilder | None = None
    preamble: list[str] = []
    for block in _blocks(raw):
        known_type = normalizer.classify(block.text) if len(block.text) <= 120 else None
        heading = block.kind == "heading" or (
            block.kind == "paragraph" and len(block.text) <= 120
            and (known_type is not None or (
                block.strong_only and word_count(block.text) <= 12
                and not block.text.rstrip().endswith(".")
            ))
        )
        if heading:
            if current is not None:
                sections.append(current.finish())
            current = _SectionBuilder(block.text, known_type, block.level or 2)
        elif current is None:
            preamble.append(block.text)
        else:
            current.body_parts.append(block.text)
            current.bullet_count += block.kind == "bullet"
    if current is not None:
        sections.append(current.finish())
    return ExtractedStructure(tuple(sections), word_count("\n".join(preamble)))
