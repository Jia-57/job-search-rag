"""Shared job records used before and after job-family classification."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


AtsSource: TypeAlias = Literal[
    "greenhouse", "lever", "ashby", "smartrecruiters"
]
JobFamily: TypeAlias = Literal[
    "software_engineering",
    "data_science_analytics",
    "ai_engineering",
    "product_management",
]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def stable_job_id(source: AtsSource, board_slug: str, source_job_id: str) -> str:
    """Identify one provider posting independently of its title or company name."""
    parts = (source, board_slug, source_job_id)
    if any(not part.strip() for part in parts):
        raise ValueError("source, board_slug and source_job_id must be non-empty")
    key = "\x1f".join(part.strip() for part in parts)
    return sha256(key.encode("utf-8")).hexdigest()


class SharedJobFields(BaseModel):
    """Fields with the same meaning across all four ATS providers."""

    model_config = ConfigDict(extra="forbid")

    source: AtsSource
    source_job_id: NonEmptyText
    company: NonEmptyText
    title: NonEmptyText
    url: NonEmptyText
    location: str | None = None
    description_raw: str = Field(min_length=1)
    posted_at: datetime | None = None
    fetched_at: datetime
    workplace_type: str | None = None
    employment_type: str | None = None
    department: str | None = None
    team: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    seniority: str | None = None
    source_metadata: dict[str, Any] | None = None

    @field_validator("url")
    @classmethod
    def require_public_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("url must be an absolute HTTP(S) URL")
        return value

    @field_validator("posted_at", "fetched_at")
    @classmethod
    def require_timezone_and_convert_to_utc(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("job timestamps must include a timezone")
        return value.astimezone(timezone.utc)


class CandidateJob(SharedJobFields):
    """Provider-mapped record before final classification and text cleaning."""

    board_slug: NonEmptyText
    raw_file: str | None = None

    @property
    def stable_id(self) -> str:
        return stable_job_id(self.source, self.board_slug, self.source_job_id)


class Job(SharedJobFields):
    """Canonical record written to jobs.jsonl after Phase 1 validation."""

    id: str = Field(pattern=r"^[0-9a-f]{64}$")
    job_family: JobFamily
    description_clean: NonEmptyText
