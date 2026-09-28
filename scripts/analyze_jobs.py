"""Create reproducible corpus/section statistics and three vector figures.

Run from the project root: python -m scripts.analyze_jobs
Only reads canonical JSONL; no ATS requests or modifications to dataset files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from xml.sax.saxutils import escape

from src.analysis.sections import HeadingNormalizer, extract_sections, normalize_heading, word_count
from src.filtering.title_filter import TitleFilter
from src.models.config import load_job_filters_config
from src.models.job import Job


ROOT = Path(__file__).resolve().parents[1]
FAMILY_LABELS = {
    "software_engineering": "Software Engineering",
    "data_science_analytics": "Data Scientist / Analyst",
    "ai_engineering": "Applied AI / AI Engineering",
    "product_management": "Product Manager / Owner",
}
SECTION_DISPLAY_ORDER = (
    "about_company", "about_role", "role_description", "responsibilities",
    "qualifications", "preferred_qualifications", "benefits", "additional_information",
)
PALETTE = ["#2F6B9A", "#3C9D86", "#8264A9", "#D47742", "#668AA0", "#A27758", "#757BB2", "#5D9681"]


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {key: 0 for key in ("count", "min", "q1", "median", "q3", "max")}
    return {
        "count": len(values), "min": min(values),
        "q1": round(_percentile(values, .25), 1),
        "median": round(statistics.median(values), 1),
        "q3": round(_percentile(values, .75), 1),
        "max": max(values),
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_jobs(path: Path, title_filter: TitleFilter) -> list[Job]:
    jobs: list[Job] = []
    seen_ids: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        job = Job.model_validate_json(line)
        if job.id in seen_ids:
            raise ValueError(f"duplicate job id at JSONL line {number}: {job.id}")
        decision = title_filter.evaluate(job.title, job.employment_type)
        if decision.status != "accepted" or decision.job_family != job.job_family:
            raise ValueError(f"title/family mismatch or internship at line {number}: {job.title}")
        seen_ids.add(job.id)
        jobs.append(job)
    if not jobs:
        raise ValueError("canonical JSONL contains no jobs")
    return jobs


def analyze(jobs: list[Job], normalizer: HeadingNormalizer, dataset_hash: str) -> tuple[dict, dict[str, list[dict]], dict[str, list[int]]]:
    """Return summary, CSV tables, and figure-ready distributions."""
    canonical_keys = [section.key for section in normalizer.types]
    canonical_labels = {section.key: section.label for section in normalizer.types}
    provider_counts = Counter(job.source for job in jobs)
    family_counts = Counter(job.job_family for job in jobs)
    company_counts = Counter(job.company for job in jobs)
    optional_fields = (
        "location", "posted_at", "workplace_type", "employment_type",
        "department", "team", "salary_min", "salary_max", "salary_currency", "seniority",
    )
    length_by_family: dict[str, list[int]] = {family: [] for family in FAMILY_LABELS}
    length_by_section: dict[str, list[int]] = {key: [] for key in canonical_keys}
    occurrences: dict[str, list[dict]] = defaultdict(list)
    variant_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    variant_jobs: dict[tuple[str, str], set[str]] = defaultdict(set)
    prevalence: dict[str, set[str]] = defaultdict(set)
    prevalence_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    job_rows: list[dict] = []
    section_rows: list[dict] = []
    all_section_counts: list[int] = []
    canonical_section_counts: list[int] = []
    no_heading_jobs = 0
    total_preamble_words = 0
    total_canonical_body_words = 0
    total_unmapped_body_words = 0

    for job in jobs:
        total_words = word_count(job.description_clean)
        length_by_family[job.job_family].append(total_words)
        extracted = extract_sections(job.description_raw, normalizer)
        content_sections = [section for section in extracted.sections if section.word_count > 0]
        known_sections = [section for section in content_sections if section.canonical_type]
        total_preamble_words += extracted.preamble_word_count
        total_canonical_body_words += sum(section.word_count for section in known_sections)
        total_unmapped_body_words += sum(
            section.word_count for section in content_sections if not section.canonical_type
        )
        all_section_counts.append(len(content_sections))
        canonical_section_counts.append(len(known_sections))
        no_heading_jobs += not extracted.sections
        job_rows.append({
            "id": job.id, "source": job.source, "company": job.company,
            "job_family": job.job_family, "jd_words": total_words,
            "sections": len(content_sections), "canonical_sections": len(known_sections),
            "preamble_words": extracted.preamble_word_count,
        })
        for index, section in enumerate(extracted.sections, start=1):
            key = section.canonical_type or "unmapped"
            normalized = normalize_heading(section.raw_heading)
            variant = (key, normalized)
            variant_counts[variant][section.raw_heading] += 1
            variant_jobs[variant].add(job.id)
            if section.word_count == 0:
                continue
            row = {
                "job_id": job.id, "source": job.source, "job_family": job.job_family,
                "section_index": index, "canonical_section": key,
                "raw_heading": section.raw_heading, "heading_level": section.heading_level,
                "words": section.word_count, "bullets": section.bullet_count,
            }
            section_rows.append(row)
            occurrences[key].append(row)
            prevalence[key].add(job.id)
            prevalence_by_source[(job.source, key)].add(job.id)
            if key != "unmapped":
                length_by_section[key].append(section.word_count)

    section_stats: list[dict] = []
    for key in [*canonical_keys, "unmapped"]:
        rows = occurrences[key]
        words = [row["words"] for row in rows]
        bullets = [row["bullets"] for row in rows]
        stats = _distribution(words)
        section_stats.append({
            "canonical_section": key,
            "label": canonical_labels.get(key, "Unmapped heading"),
            "jd_count": len(prevalence[key]),
            "jd_coverage_percent": round(100 * len(prevalence[key]) / len(jobs), 1),
            "section_occurrences": len(rows),
            "median_words": stats["median"], "q1_words": stats["q1"],
            "q3_words": stats["q3"], "max_words": stats["max"],
            "median_bullets": round(statistics.median(bullets), 1) if bullets else 0,
            "heading_variants": sum(k == key for k, _ in variant_counts),
        })

    heading_rows = [
        {
            "canonical_section": key,
            "normalized_heading": normalized,
            "example_raw_heading": counts.most_common(1)[0][0],
            "occurrences": sum(counts.values()),
            "jd_count": len(variant_jobs[(key, normalized)]),
        }
        for (key, normalized), counts in variant_counts.items()
    ]
    heading_rows.sort(key=lambda row: (row["canonical_section"], -row["occurrences"], row["normalized_heading"]))

    source_rows = [
        {
            "source": source, "canonical_section": key,
            "source_jds": count, "jd_count": len(prevalence_by_source[(source, key)]),
            "jd_coverage_percent": round(100 * len(prevalence_by_source[(source, key)]) / count, 1),
        }
        for source, count in sorted(provider_counts.items())
        for key in [*canonical_keys, "unmapped"]
    ]
    company_rows = [
        {
            "company": company, "jobs": count,
            "job_families": len({job.job_family for job in jobs if job.company == company}),
            "sources": ", ".join(sorted({job.source for job in jobs if job.company == company})),
        }
        for company, count in sorted(company_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    family_source_rows = [
        {"source": source, "job_family": family,
         "jobs": sum(job.source == source and job.job_family == family for job in jobs)}
        for source in sorted(provider_counts)
        for family in FAMILY_LABELS
    ]
    summary = {
        "dataset_sha256": dataset_hash,
        "job_count": len(jobs),
        "distinct_companies": len({job.company.casefold() for job in jobs}),
        "max_jobs_per_company": max(company_counts.values()),
        "jobs_by_family": dict(sorted(family_counts.items())),
        "jobs_by_source": dict(sorted(provider_counts.items())),
        "optional_field_non_null_counts": {
            field: sum(getattr(job, field) is not None for job in jobs)
            for field in optional_fields
        },
        "jd_words": _distribution([row["jd_words"] for row in job_rows]),
        "jd_words_by_family": {family: _distribution(values) for family, values in length_by_family.items()},
        "sections_per_jd": _distribution(all_section_counts),
        "canonical_sections_per_jd": _distribution(canonical_section_counts),
        "jobs_without_detected_headings": no_heading_jobs,
        "jobs_with_contentful_canonical_section": len(set().union(*(prevalence[k] for k in canonical_keys))),
        "jobs_with_explicit_responsibilities_and_qualifications": len(
            prevalence["responsibilities"] & prevalence["qualifications"]
        ),
        "preamble_body_words": total_preamble_words,
        "canonical_section_body_words": total_canonical_body_words,
        "unmapped_section_body_words": total_unmapped_body_words,
        "total_heading_variants": len(variant_counts),
        "unmapped_heading_variants": sum(key == "unmapped" for key, _ in variant_counts),
        "section_definition": "Explicit HTML h1-h6, standalone bold paragraphs, or short paragraphs matching configured aliases; nonempty body required for prevalence and length.",
        "word_definition": "Unicode lexical words; not model tokens. Section lengths exclude headings.",
    }
    tables = {
        "jobs": job_rows, "sections": section_rows,
        "section_stats": section_stats, "heading_variants": heading_rows,
        "prevalence_by_source": source_rows,
        "companies": company_rows, "family_by_source": family_source_rows,
    }
    figures = {**{f"family:{key}": values for key, values in length_by_family.items()},
               **{f"section:{key}": values for key, values in length_by_section.items()}}
    return summary, tables, figures


def _nice_ceiling(value: float) -> int:
    if value <= 0:
        return 1
    base = 10 ** math.floor(math.log10(value / 5))
    step = next(multiplier * base for multiplier in (1, 2, 5, 10) if multiplier * base >= value / 5)
    return max(1, math.ceil(value / step) * step)


def _svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="40" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="#17324D">{escape(title)}</text>',
        f'<text x="40" y="65" font-family="Arial,sans-serif" font-size="13" fill="#53677B">{escape(subtitle)}</text>',
    ]


def _axis(parts: list[str], *, left: int, right: int, top: int, bottom: int,
          max_value: int, axis_label: str) -> None:
    span = right - left
    for tick in range(6):
        value = max_value * tick / 5
        x = left + span * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#E2E9EF"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#53677B">{value:,.0f}</text>')
    parts.append(f'<text x="{(left + right) / 2:.1f}" y="{bottom + 52}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#17324D">{escape(axis_label)}</text>')


def _boxplot_svg(title: str, subtitle: str, rows: list[tuple[str, list[int]]]) -> str:
    width, left, right, top, row_height = 1120, 275, 1025, 95, 62
    height = top + row_height * len(rows) + 105
    bottom = top + row_height * len(rows)
    max_value = _nice_ceiling(max((max(values) for _, values in rows if values), default=1))
    parts = _svg_start(width, height, title, subtitle)
    _axis(parts, left=left, right=right, top=top - 10, bottom=bottom,
          max_value=max_value, axis_label="Word count")
    scale = (right - left) / max_value
    for index, (label, values) in enumerate(rows):
        y = top + row_height * (index + .5)
        parts.append(f'<text x="{left - 18}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" fill="#17324D">{escape(label)}</text>')
        if not values:
            parts.append(f'<text x="{left + 8}" y="{y + 5:.1f}" font-family="Arial,sans-serif" font-size="12" fill="#8794A0">No sections</text>')
            continue
        q1, median, q3 = (_percentile(values, .25), _percentile(values, .5), _percentile(values, .75))
        iqr = q3 - q1
        lower = min(value for value in values if value >= q1 - 1.5 * iqr)
        upper = max(value for value in values if value <= q3 + 1.5 * iqr)
        color = PALETTE[index % len(PALETTE)]
        x = lambda value: left + value * scale
        parts.append(f'<line x1="{x(lower):.1f}" y1="{y:.1f}" x2="{x(upper):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>')
        for bound in (lower, upper):
            parts.append(f'<line x1="{x(bound):.1f}" y1="{y - 11:.1f}" x2="{x(bound):.1f}" y2="{y + 11:.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{x(q1):.1f}" y="{y - 16:.1f}" width="{max(x(q3) - x(q1), 1):.1f}" height="32" rx="3" fill="{color}" fill-opacity="0.24" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x(median):.1f}" y1="{y - 16:.1f}" x2="{x(median):.1f}" y2="{y + 16:.1f}" stroke="{color}" stroke-width="3"/>')
        for value in values:
            if value < lower or value > upper:
                parts.append(f'<circle cx="{x(value):.1f}" cy="{y:.1f}" r="2.7" fill="{color}" fill-opacity="0.48"/>')
        parts.append(f'<text x="{right + 12}" y="{y + 5:.1f}" font-family="Arial,sans-serif" font-size="12" fill="#53677B">n={len(values)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _prevalence_svg(rows: list[dict], total_jobs: int) -> str:
    rows = sorted((row for row in rows if row["canonical_section"] != "unmapped"),
                  key=lambda row: (-row["jd_coverage_percent"], row["label"]))
    width, left, right, top, row_height = 1120, 285, 1010, 100, 55
    bottom = top + row_height * len(rows)
    parts = _svg_start(width, bottom + 105, "Canonical section prevalence",
                       f"Share of {total_jobs} JDs with a matched heading and nonempty section body")
    _axis(parts, left=left, right=right, top=top - 10, bottom=bottom,
          max_value=100, axis_label="JD coverage (%)")
    for index, row in enumerate(rows):
        y = top + row_height * (index + .5)
        percent = row["jd_coverage_percent"]
        color = PALETTE[index % len(PALETTE)]
        parts.append(f'<text x="{left - 18}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" fill="#17324D">{escape(row["label"])}</text>')
        parts.append(f'<rect x="{left}" y="{y - 15:.1f}" width="{(right - left) * percent / 100:.1f}" height="30" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{right + 12}" y="{y + 5:.1f}" font-family="Arial,sans-serif" font-size="12" fill="#17324D">{percent:.1f}%</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze canonical JD and section structure")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "canonical" / "jobs.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "dataset_analysis")
    parser.add_argument("--headings", type=Path, default=ROOT / "config" / "section_headings.yaml")
    parser.add_argument("--filters", type=Path, default=ROOT / "config" / "job_filters.yaml")
    args = parser.parse_args()
    title_filter = TitleFilter(load_job_filters_config(args.filters))
    normalizer = HeadingNormalizer.from_yaml(args.headings)
    jobs = _read_jobs(args.input, title_filter)
    summary, tables, figures = analyze(jobs, normalizer, sha256(args.input.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "jobs.csv", tables["jobs"],
               ["id", "source", "company", "job_family", "jd_words", "sections", "canonical_sections", "preamble_words"])
    _write_csv(args.output_dir / "sections.csv", tables["sections"],
               ["job_id", "source", "job_family", "section_index", "canonical_section", "raw_heading", "heading_level", "words", "bullets"])
    _write_csv(args.output_dir / "section_stats.csv", tables["section_stats"],
               ["canonical_section", "label", "jd_count", "jd_coverage_percent", "section_occurrences", "median_words", "q1_words", "q3_words", "max_words", "median_bullets", "heading_variants"])
    _write_csv(args.output_dir / "heading_variants.csv", tables["heading_variants"],
               ["canonical_section", "normalized_heading", "example_raw_heading", "occurrences", "jd_count"])
    _write_csv(args.output_dir / "section_prevalence_by_source.csv", tables["prevalence_by_source"],
               ["source", "canonical_section", "source_jds", "jd_count", "jd_coverage_percent"])
    _write_csv(args.output_dir / "company_counts.csv", tables["companies"],
               ["company", "jobs", "job_families", "sources"])
    _write_csv(args.output_dir / "family_by_source.csv", tables["family_by_source"],
               ["source", "job_family", "jobs"])
    (args.output_dir / "figure_1_jd_length_by_family.svg").write_text(_boxplot_svg(
        "JD length by job family", "Clean JD word count; boxes show IQR, centre line median, whiskers 1.5×IQR",
        [(label, figures[f"family:{key}"]) for key, label in FAMILY_LABELS.items()]
    ), encoding="utf-8")
    (args.output_dir / "figure_2_section_prevalence.svg").write_text(
        _prevalence_svg(tables["section_stats"], len(jobs)), encoding="utf-8"
    )
    (args.output_dir / "figure_3_section_length.svg").write_text(_boxplot_svg(
        "Section length by canonical type", "Section body words only; multiple sections per JD possible",
        [(section.label, figures[f"section:{section.key}"])
         for section in sorted(normalizer.types, key=lambda item: (
             SECTION_DISPLAY_ORDER.index(item.key)
             if item.key in SECTION_DISPLAY_ORDER else len(SECTION_DISPLAY_ORDER)
         ))]
    ), encoding="utf-8")
    print(json.dumps({
        "jobs": len(jobs), "sources": summary["jobs_by_source"],
        "families": summary["jobs_by_family"],
        "sections_per_jd_median": summary["sections_per_jd"]["median"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
