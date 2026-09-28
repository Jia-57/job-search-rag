"""Validated configuration contracts for ATS boards and title rules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.job import AtsSource, JobFamily, NonEmptyText


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: NonEmptyText
    provider: AtsSource
    board_slug: NonEmptyText
    region: str | None = None
    priority_group: Literal["germany", "global"] = "global"
    enabled: bool = True


class DiscoveryConfig(BaseModel):
    """Company-independent ATS board discovery and bounded scanning."""

    model_config = ConfigDict(extra="forbid")

    domains: dict[AtsSource, list[NonEmptyText]]
    max_boards_per_run: int = Field(default=400, ge=1)
    min_boards_before_stop: int = Field(default=100, ge=1)
    index_pages_per_domain: int = Field(default=3, ge=1, le=100)

    @model_validator(mode="after")
    def validate_domains(self) -> DiscoveryConfig:
        if not self.domains or any(not values for values in self.domains.values()):
            raise ValueError("discovery needs at least one domain per selected ATS")
        if self.min_boards_before_stop > self.max_boards_per_run:
            raise ValueError("min_boards_before_stop exceeds max_boards_per_run")
        return self


class SourcesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceConfig] = Field(default_factory=list)
    discovery: DiscoveryConfig | None = None

    @model_validator(mode="after")
    def unique_boards(self) -> SourcesConfig:
        keys = [(item.provider, item.board_slug) for item in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("sources contains duplicate provider/board_slug pairs")
        if not self.sources and self.discovery is None:
            raise ValueError("sources or discovery must be configured")
        return self


class FamilyRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_title_patterns: list[NonEmptyText] = Field(min_length=1)
    negative_title_patterns: list[NonEmptyText] = Field(default_factory=list)


class JobFiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internship_title_patterns: list[NonEmptyText] = Field(min_length=1)
    ambiguous_title_patterns: list[NonEmptyText] = Field(default_factory=list)
    api_query_terms: list[NonEmptyText] = Field(default_factory=list)
    families: dict[JobFamily, FamilyRules]

    @model_validator(mode="after")
    def validate_patterns_and_families(self) -> JobFiltersConfig:
        expected = {
            "software_engineering",
            "data_science_analytics",
            "ai_engineering",
            "product_management",
        }
        if set(self.families) != expected:
            raise ValueError("job filters must define all four target families")
        patterns = self.internship_title_patterns + self.ambiguous_title_patterns
        for rules in self.families.values():
            patterns.extend(rules.positive_title_patterns)
            patterns.extend(rules.negative_title_patterns)
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid title pattern: {pattern}") from exc
        return self


class DatasetConfig(BaseModel):
    """Research corpus size and repeatable sampling settings."""

    model_config = ConfigDict(extra="forbid")

    target_per_family: int = Field(ge=1, le=100)
    max_jobs_per_company: int = Field(default=4, ge=1)
    sampling_seed: NonEmptyText
    preferred_countries: list[NonEmptyText] = Field(default_factory=list)


def _load_yaml(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_sources_config(path: str | Path) -> SourcesConfig:
    return SourcesConfig.model_validate(_load_yaml(path))


def load_job_filters_config(path: str | Path) -> JobFiltersConfig:
    return JobFiltersConfig.model_validate(_load_yaml(path))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.model_validate(_load_yaml(path))
