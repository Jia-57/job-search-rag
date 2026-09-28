# Local-first Job Search Knowledge Base with RAG

This repository contains a frozen research corpus of public job descriptions and code for corpus analysis and retrieval experiments. Live postings change over time, so the included dataset is the reference snapshot for reproducing the experiments.

## Current corpus

The frozen snapshot in `data/canonical/jobs.jsonl` contains 240 unique, non-internship JDs, with 60 in each of four job families. It was assembled from public ATS postings and represents openings available at collection time; those postings may since have changed or disappeared.

Families: Software Engineering; Data Scientist / Data Analyst; AI Architect / AI Engineer / AI Automation; Product Manager / Product Owner. The published JSONL is the fixed input for the analysis and retrieval workflows below.

## Dissertation dataset analysis

Generate corpus statistics, section-heading vocabulary, provider-stratified prevalence, and three SVG figures without modifying the JD dataset:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_jobs
```

Outputs and the measurement protocol are documented in `reports/dataset_analysis/README.md`. Heading aliases live in `config/section_headings.yaml`; the analysis does not implement chunking or retrieval.

## JD evidence retrieval experiment V2

The V2 study builds a frozen, automatically generated source-evidence benchmark
from the canonical JD corpus, then compares BM25, local BGE-M3 dense retrieval,
and reciprocal rank fusion on paired high/low lexical-overlap questions. It
measures retrieval of the originating evidence, not all relevant jobs or final
answer quality. Use Python 3.11 or newer and install `requirements.txt`. The
models and revisions are pinned in `config/experiment_v2.yaml`. Run
`python -m scripts.run_experiment cache-models --config config/experiment_v2.yaml`
to cache their weights, then use the `smoke-models`, `prepare`, `benchmark`,
`validate`, `retrieve`, and `report` stages in order. Model inference uses local
weights. The cluster-specific Slurm scripts and detailed run notes stay local.

## JD evidence retrieval experiment V3 extension

V3 reads the frozen V2 benchmark and Top-20 results, then reranks the union of
BM25 and dense candidates with a local BGE reranker. It writes only to
`data/benchmarks/jd_v3/`, `runs/jd_v3/`, and `reports/experiment/jd_v3/`.
See `RETRIEVAL_EXPERIMENT_V3_IMPLEMENTATION.md` for the design and use
`config/experiment_v3.yaml` with `scripts/run_rerank_v3.py` for the new stages.
