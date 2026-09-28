"""Deterministic text, timestamp, and location normalization for ATS jobs."""

from __future__ import annotations

import html
import math
import re
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from typing import Literal


_TAG_NAMES = (
    r"a|article|b|blockquote|br|div|em|h[1-6]|i|li|ol|p|section|"
    r"span|strong|table|tbody|td|th|tr|ul|script|style"
)
_HTML_TAG = re.compile(rf"<\s*/?\s*(?:{_TAG_NAMES})\b[^>]*>", re.IGNORECASE)
_ENCODED_HTML_TAG = re.compile(
    rf"&lt;\s*/?\s*(?:{_TAG_NAMES})\b", re.IGNORECASE
)
_BLOCK_TAGS = {
    "article", "blockquote", "div", "p", "section", "table", "tbody",
    "td", "th", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
}
_IGNORED_TAGS = {"script", "style"}


class _DescriptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._parts: list[str] = []
        self._prefix = ""
        self._lists: list[tuple[str, int]] = []
        self._ignored_depth = 0

    def _finish_line(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._parts)).strip()
        if text:
            self.lines.append(f"{self._prefix}{text}")
        self._parts.clear()
        self._prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _IGNORED_TAGS:
            self._finish_line()
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in _BLOCK_TAGS or tag == "br":
            self._finish_line()
            if re.fullmatch(r"h[1-6]", tag):
                self._prefix = f"{'#' * int(tag[1])} "
        elif tag in {"ul", "ol"}:
            self._finish_line()
            self._lists.append((tag, 0))
        elif tag == "li":
            self._finish_line()
            indent = "  " * max(len(self._lists) - 1, 0)
            if self._lists and self._lists[-1][0] == "ol":
                number = self._lists[-1][1] + 1
                self._lists[-1] = ("ol", number)
                self._prefix = f"{indent}{number}. "
            else:
                self._prefix = f"{indent}- "

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED_TAGS:
            self._ignored_depth = max(self._ignored_depth - 1, 0)
            return
        if self._ignored_depth:
            return
        if tag in _BLOCK_TAGS or tag == "li":
            self._finish_line()
        elif tag in {"ul", "ol"}:
            self._finish_line()
            if self._lists:
                self._lists.pop()

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        self._finish_line()
        return "\n".join(self.lines)


def _plain_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    compact: list[str] = []
    for line in lines:
        if line or (compact and compact[-1]):
            compact.append(line)
    return "\n".join(compact)


def clean_description(description: str | None) -> str:
    """Turn ATS HTML or plain text into readable text without rewriting it."""
    if not description:
        return ""
    if _HTML_TAG.search(description):
        markup = description
    else:
        decoded = html.unescape(description)
        if _HTML_TAG.search(decoded):
            markup = decoded
        elif _ENCODED_HTML_TAG.search(decoded):
            markup = html.unescape(decoded)
        else:
            return _plain_text(decoded)
    parser = _DescriptionParser()
    parser.feed(markup)
    parser.close()
    return parser.text()


def normalize_datetime(
    value: datetime | date | str | int | float | None,
    *,
    epoch_unit: Literal["seconds", "milliseconds"] | None = None,
) -> datetime | None:
    """Return an aware UTC timestamp; leave date-only or naive values unknown."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return None
    elif isinstance(value, str):
        raw = value.strip()
        if not raw or re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"unsupported timestamp: {value}") from exc
    elif isinstance(value, bool):
        raise TypeError("boolean is not a timestamp")
    elif isinstance(value, (int, float)):
        if epoch_unit is None:
            raise ValueError("numeric timestamps require an explicit epoch_unit")
        if not math.isfinite(value):
            raise ValueError("numeric timestamp must be finite")
        seconds = value / 1000 if epoch_unit == "milliseconds" else value
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
    else:
        raise TypeError(f"unsupported timestamp type: {type(value).__name__}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def normalize_location(value: str | None) -> str | None:
    """Standardize whitespace without inferring a city, country, or remote status."""
    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip() or None
