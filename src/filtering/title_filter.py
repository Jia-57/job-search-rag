"""Deterministic title-based filtering shared by all ATS providers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.models.config import JobFiltersConfig
from src.models.job import JobFamily


FilterStatus = Literal["accepted", "internship", "ambiguous", "non_target"]


@dataclass(frozen=True)
class FilterDecision:
    status: FilterStatus
    job_family: JobFamily | None = None
    reason: str | None = None


class TitleFilter:
    """Classify clear titles; leave ambiguous cases for a later review stage."""

    def __init__(self, config: JobFiltersConfig) -> None:
        self.api_query_terms = tuple(config.api_query_terms)
        self.internship_patterns = self._compile(config.internship_title_patterns)
        self.ambiguous_patterns = self._compile(config.ambiguous_title_patterns)
        self.family_patterns = {
            family: (
                self._compile(rules.positive_title_patterns),
                self._compile(rules.negative_title_patterns),
            )
            for family, rules in config.families.items()
        }

    @staticmethod
    def _compile(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _matches(patterns: tuple[re.Pattern[str], ...], value: str) -> bool:
        return any(pattern.search(value) for pattern in patterns)

    def evaluate(self, title: str, employment_type: str | None = None) -> FilterDecision:
        """Use only explicit title/employment signals, never infer from JD prose."""
        if self._matches(self.internship_patterns, title) or (
            employment_type is not None
            and self._matches(self.internship_patterns, employment_type)
        ):
            return FilterDecision("internship", reason="internship title or employment type")

        if self._matches(self.ambiguous_patterns, title):
            return FilterDecision("ambiguous", reason="configured ambiguous title")

        matches: list[JobFamily] = []
        for family, (positive, negative) in self.family_patterns.items():
            if self._matches(positive, title) and not self._matches(negative, title):
                matches.append(family)

        if len(matches) == 1:
            return FilterDecision("accepted", job_family=matches[0])
        if len(matches) > 1:
            return FilterDecision("ambiguous", reason="matches multiple job families")
        return FilterDecision("non_target", reason="no target family title rule matched")
