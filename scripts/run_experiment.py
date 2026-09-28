"""Staged command line entry point for the frozen JD evidence retrieval V2 study.

Run compute stages through Slurm on the cluster. This module never reads raw ATS
records and never sends inference requests to a hosted model service.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.io import (
    file_sha256, installed_versions, load_config, object_sha256,
    model_snapshot_fingerprint, read_jsonl, write_json, write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiment_v2.yaml"
FAMILIES = (
    "software_engineering", "data_science_analytics",
    "ai_engineering", "product_management",
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MODEL_ALLOW = ["*.json", "*.safetensors", "*.bin", "*.pt", "*.model", "*.txt"]
_MODEL_IGNORE = ["onnx/*", "imgs/*"]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {key: _path(value) for key, value in config["paths"].items()}


def _load_settings(path: Path) -> dict[str, Any]:
    config = load_config(path)
    if config.get("experiment_version") != "jd_v2":
        raise ValueError("experiment_version must be jd_v2")
    if config["benchmark"]["per_family"] != 20:
        raise ValueError("V2 fixes benchmark size at 20 source jobs per family")
    if (config["retrieval"]["top_k"], config["retrieval"]["rrf_k"],
        config["evaluation"]["top_k"]) != (20, 60, 5):
        raise ValueError("V2 fixes retrieval Top-20, RRF k=60, evaluation Top-5")
    if config["models"]["dense"].get("normalize") is not True:
        raise ValueError("V2 dense embeddings must use L2 normalization")
    dense = config["models"]["dense"]
    if dense.get("pooling_method") != "cls" or dense.get("similarity") != "cosine":
        raise ValueError("V2 BGE-M3 uses CLS pooling and cosine similarity")
    generator = config["models"]["generator"]
    if (generator.get("do_sample") is not False
            or generator.get("enable_thinking") is not False
            or generator.get("device_map") != "auto"):
        raise ValueError("V2 question generation uses deterministic non-thinking decoding")
    if config["benchmark"].get("prompt_version") != "jd_v2_generation_2":
        raise ValueError("benchmark prompt_version does not match the implemented prompt")
    retrieval = config["retrieval"]
    if (retrieval.get("tokenizer"), retrieval.get("lowercase"),
        retrieval.get("stemming"), retrieval.get("stopwords")) != (
            "technical_terms_v1", True, False, "none"
        ):
        raise ValueError("BM25 tokenizer/lowercase/stemming/stopwords must match implementation")
    for model in config["models"].values():
        if not model.get("model_id") or not model.get("revision"):
            raise ValueError("each model needs an explicit model_id and revision")
    return config


def _corpus(config: dict[str, Any], paths: dict[str, Path]) -> tuple[list[dict[str, Any]], str]:
    corpus_path = paths["corpus"]
    digest = file_sha256(corpus_path)
    if digest != config["dataset_sha256"]:
        raise ValueError(f"corpus SHA-256 differs from frozen V2 value: {digest}")
    summary_path = paths["dataset_summary"]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dataset_sha256") != digest:
            raise ValueError("dataset analysis summary and corpus SHA-256 disagree")
    rows = read_jsonl(corpus_path)
    counts = Counter(row.get("job_family") for row in rows)
    if len(rows) != 240 or any(counts.get(family) != 60 for family in FAMILIES):
        raise ValueError(f"expected 240 JDs, 60 per family; found {dict(counts)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("canonical corpus has empty or duplicate Job.id values")
    if any(not isinstance(row.get("description_clean"), str)
           or not row["description_clean"].strip() for row in rows):
        raise ValueError("canonical corpus has an empty description_clean")
    return rows, digest


def _manifest_path(paths: dict[str, Path]) -> Path:
    return paths["benchmark_dir"] / "manifest.json"


def _read_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    path = _manifest_path(paths)
    if not path.exists():
        raise FileNotFoundError(f"run 'prepare' first: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark manifest must be a JSON object")
    return value


def _save_manifest(paths: dict[str, Path], value: dict[str, Any]) -> None:
    write_json(value, _manifest_path(paths), overwrite=True)


def _model_record(model_config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model_config.items() if key != "local_path"}


def _assert_benchmark_settings(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Keep generation settings fixed after nodes/benchmark are prepared."""
    expected = {
        "seed": config["seed"],
        "chunk": config["chunk"],
        "heading_rules_sha256": file_sha256(_path(config["chunk"]["heading_config"])),
        "evidence_config": config["evidence"],
        "benchmark_config": config["benchmark"],
        "generator_model": _model_record(config["models"]["generator"]),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"current {field} differs from the prepared/frozen benchmark manifest")


def _base_manifest(config: dict[str, Any], corpus_digest: str,
                   nodes_digest: str, node_count: int) -> dict[str, Any]:
    return {
        "benchmark_version": config["experiment_version"],
        "dataset_sha256": corpus_digest,
        "seed": config["seed"],
        "node_count": node_count,
        "nodes_sha256": nodes_digest,
        "chunk": config["chunk"],
        "heading_rules_sha256": file_sha256(_path(config["chunk"]["heading_config"])),
        "evidence_config": config["evidence"],
        "benchmark_config": config["benchmark"],
        "generator_model": _model_record(config["models"]["generator"]),
        "dense_model": _model_record(config["models"]["dense"]),
        "retrieval": config["retrieval"],
        "evaluation": config["evaluation"],
        "dependencies": installed_versions(),
        "gold_scope": "Only nodes containing the generated query's originating evidence span are labeled relevant.",
        "frozen": False,
    }


def _nodes(paths: dict[str, Path], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = paths["benchmark_dir"] / "nodes.jsonl"
    digest = file_sha256(path)
    if digest != manifest.get("nodes_sha256"):
        raise ValueError("nodes.jsonl differs from the benchmark manifest")
    nodes = read_jsonl(path)
    if len(nodes) != manifest.get("node_count"):
        raise ValueError("node count differs from the benchmark manifest")
    return nodes


def _frozen(config: dict[str, Any], paths: dict[str, Path],
            jobs: list[dict[str, Any]], nodes: list[dict[str, Any]],
            manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from src.experiment.validate_benchmark import validate_benchmark

    if manifest.get("frozen") is not True:
        raise ValueError("benchmark has not been frozen; run 'benchmark' first")
    _assert_benchmark_settings(config, manifest)
    output_dir = paths["benchmark_dir"]
    query_path = output_dir / "queries.jsonl"
    qrel_path = output_dir / "qrels.jsonl"
    if (file_sha256(query_path) != manifest.get("queries_sha256")
            or file_sha256(qrel_path) != manifest.get("qrels_sha256")):
        raise ValueError("frozen benchmark files changed since the manifest was written")
    queries, qrels = read_jsonl(query_path), read_jsonl(qrel_path)
    validate_benchmark(
        jobs, nodes, queries, qrels,
        expected_pairs=4 * config["benchmark"]["per_family"],
        expected_per_family=config["benchmark"]["per_family"],
    )
    return queries, qrels


def _local_model(config: dict[str, Any]) -> Path:
    """Return an existing pinned snapshot; inference never downloads weights."""
    local_path = config.get("local_path")
    if local_path:
        path = _path(local_path)
        if path.parent.name == "snapshots" and path.name != config["revision"]:
            raise ValueError("local Hugging Face snapshot path does not match pinned revision")
    else:
        from huggingface_hub import snapshot_download

        try:
            path = Path(snapshot_download(
                repo_id=config["model_id"], revision=config["revision"],
                local_files_only=True,
                allow_patterns=_MODEL_ALLOW, ignore_patterns=_MODEL_IGNORE,
            ))
        except Exception as exc:
            raise FileNotFoundError(
                f"pinned {config['model_id']} snapshot is not cached; "
                "set models.*.local_path or run the cache-models stage on an approved node"
            ) from exc
    if not path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {path}")
    return path.resolve()


def stage_cache_models(config: dict[str, Any]) -> None:
    """Explicit network download stage, separate from all inference stages."""
    from huggingface_hub import snapshot_download

    for name in ("generator", "dense"):
        model = config["models"][name]
        if model.get("local_path"):
            print(f"{name}: using local directory {_local_model(model)}")
            continue
        snapshot = snapshot_download(
            repo_id=model["model_id"], revision=model["revision"],
            local_files_only=False,
            allow_patterns=_MODEL_ALLOW, ignore_patterns=_MODEL_IGNORE,
        )
        print(f"{name}: cached pinned snapshot {snapshot}")


def stage_smoke_models(config: dict[str, Any]) -> None:
    """Exercise one local Qwen pair and one local BGE-M3 ranking on a GPU node."""
    import gc
    from time import perf_counter

    from src.experiment.build_benchmark import LocalQwenGenerator
    from src.experiment.retrievers import DenseRetriever
    from src.experiment.validate_benchmark import static_pair_errors

    example = (
        "At least 3 years of experience deploying machine-learning services "
        "on Kubernetes."
    )
    generator_config = config["models"]["generator"]
    generator = LocalQwenGenerator(
        model_name_or_path=str(_local_model(generator_config)),
        revision=generator_config["revision"],
        local_files_only=generator_config["local_files_only"],
        dtype=generator_config["dtype"],
        max_new_tokens=generator_config["max_new_tokens"],
        prompt_version=config["benchmark"]["prompt_version"],
    )
    started = perf_counter()
    smoke_job = {"id": "smoke-source", "company": "Sample Company",
                 "title": "ML Engineer", "url": "https://example.org/smoke"}
    required_flags = (
        "role_related_atomic_fact", "self_contained_evidence",
        "same_information_need", "high_supported", "low_supported",
        "constraints_preserved", "natural_questions",
        "low_is_conceptual_paraphrase",
    )
    previous_error: str | None = None
    for attempt in range(1, config["benchmark"]["max_attempts_per_candidate"] + 1):
        try:
            proposal = generator.generate_pair(example, rejection_reason=previous_error)
        except ValueError as exc:
            previous_error = f"invalid generation JSON: {exc}"
            print(f"Qwen smoke attempt {attempt}: {previous_error}", flush=True)
            continue
        print(f"Qwen smoke attempt {attempt} proposal: "
              f"{json.dumps(proposal, ensure_ascii=False)}", flush=True)
        static_errors = static_pair_errors(proposal, example, smoke_job)
        if static_errors:
            previous_error = ", ".join(static_errors)
            print(f"Qwen smoke attempt {attempt} static rejection: {previous_error}", flush=True)
            continue
        try:
            assessment = generator.validate_pair(
                example, proposal["high_query"], proposal["low_query"], proposal["anchor_term"],
            )
        except ValueError as exc:
            previous_error = f"invalid semantic JSON: {exc}"
            print(f"Qwen smoke attempt {attempt}: {previous_error}", flush=True)
            continue
        failed_flags = [flag for flag in required_flags if assessment.get(flag) is not True]
        if failed_flags:
            previous_error = ", ".join(failed_flags)
            print(f"Qwen smoke attempt {attempt} semantic rejection: {previous_error}; "
                  f"assessment={json.dumps(assessment, ensure_ascii=False)}", flush=True)
            continue
        break
    else:
        raise ValueError(
            "Qwen smoke pair failed automatic validation after "
            f"{config['benchmark']['max_attempts_per_candidate']} attempts: {previous_error}"
        )
    print("Qwen local generation:", json.dumps(proposal, ensure_ascii=False))
    print("Qwen independent check:", json.dumps(assessment, ensure_ascii=False))
    print(f"Qwen attempts needed: {attempt}")
    print(f"Qwen generation plus validation: {perf_counter() - started:.1f}s")
    del generator
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    dense_config = config["models"]["dense"]
    nodes = [
        {"node_id": "smoke_kubernetes", "retrieval_text": example},
        {"node_id": "smoke_unrelated", "retrieval_text": "Design product roadmaps and pricing."},
    ]
    dense = DenseRetriever(
        nodes, model_path=_local_model(dense_config), device=dense_config["device"],
        use_fp16=dense_config["use_fp16"], batch_size=2,
        query_max_length=dense_config["query_max_length"],
        passage_max_length=dense_config["max_length"],
    )
    rankings, _, _ = dense.rank_many([proposal["low_query"]], top_k=2)
    if rankings[0][0]["node_id"] != "smoke_kubernetes":
        raise ValueError("BGE-M3 smoke retrieval did not rank the evidence first")
    print("BGE-M3 local dense ranking:", json.dumps(rankings[0], ensure_ascii=False))


def stage_prepare(config: dict[str, Any], paths: dict[str, Path]) -> None:
    from src.experiment.prepare import NodeConfig, prepare_nodes

    jobs, digest = _corpus(config, paths)
    chunk = config["chunk"]
    nodes = prepare_nodes(jobs, NodeConfig(
        chunk_size=chunk["chunk_size"],
        chunk_overlap=chunk["chunk_overlap"],
        heading_config=_path(chunk["heading_config"]),
        include_heading_in_retrieval=chunk["include_heading_in_retrieval"],
    ))
    output = paths["benchmark_dir"] / "nodes.jsonl"
    if output.exists():
        if read_jsonl(output) != nodes:
            raise ValueError("existing nodes differ; a frozen benchmark cannot be silently rebuilt")
    else:
        write_jsonl(nodes, output)
    old = _read_manifest(paths) if _manifest_path(paths).exists() else None
    new = _base_manifest(config, digest, file_sha256(output), len(nodes))
    if old is not None:
        if old.get("dataset_sha256") != digest or old.get("nodes_sha256") != new["nodes_sha256"]:
            raise ValueError("existing manifest points to a different corpus or nodes")
        if old.get("frozen"):
            _assert_benchmark_settings(config, old)
            print(f"prepared nodes already frozen: {len(nodes)} nodes")
            return
    _save_manifest(paths, new)
    print(f"prepared {len(nodes)} deterministic nodes from {len(jobs)} jobs")


def stage_benchmark(config: dict[str, Any], paths: dict[str, Path]) -> None:
    from src.experiment.build_benchmark import (
        BenchmarkConfig, LocalQwenGenerator, build_benchmark, write_benchmark,
    )
    from src.experiment.extract_evidence import EvidenceConfig, extract_evidence_candidates
    from src.experiment.validate_benchmark import validate_benchmark

    jobs, digest = _corpus(config, paths)
    manifest = _read_manifest(paths)
    if manifest.get("dataset_sha256") != digest:
        raise ValueError("manifest and canonical corpus hashes differ")
    _assert_benchmark_settings(config, manifest)
    nodes = _nodes(paths, manifest)
    output_dir = paths["benchmark_dir"]
    query_path, qrel_path = output_dir / "queries.jsonl", output_dir / "qrels.jsonl"
    if query_path.exists() or qrel_path.exists():
        if not (query_path.exists() and qrel_path.exists() and manifest.get("frozen")):
            raise FileExistsError("partial or unmanifested benchmark files already exist")
        _frozen(config, paths, jobs, nodes, manifest)
        print("benchmark already frozen; existing query/qrel files were validated")
        return
    evidence = extract_evidence_candidates(jobs, nodes, EvidenceConfig(**config["evidence"]))
    candidate_path = output_dir / "evidence_candidates.jsonl"
    if candidate_path.exists():
        if read_jsonl(candidate_path) != evidence.candidates:
            raise ValueError("existing evidence candidates differ from current rules")
    else:
        write_jsonl(evidence.candidates, candidate_path)
    write_jsonl(evidence.rejections, output_dir / "evidence_rejections.jsonl", overwrite=True)

    generator_config = config["models"]["generator"]
    generator_path = _local_model(generator_config)
    generator_snapshot = model_snapshot_fingerprint(
        generator_path, generator_config["revision"]
    )
    generator = LocalQwenGenerator(
        model_name_or_path=str(generator_path),
        revision=generator_config["revision"],
        local_files_only=generator_config["local_files_only"],
        dtype=generator_config["dtype"],
        max_new_tokens=generator_config["max_new_tokens"],
        prompt_version=config["benchmark"]["prompt_version"],
    )
    result = build_benchmark(
        jobs, nodes, evidence.candidates, generator,
        BenchmarkConfig(**config["benchmark"]), seed=config["seed"],
    )
    # Snapshot paths are machine-specific. Frozen question metadata records the
    # published model ID and immutable revision instead.
    for query in result.queries:
        query["generator_model"] = generator_config["model_id"]
        query["generator_revision"] = generator_config["revision"]
    write_jsonl(result.rejections, output_dir / "benchmark_rejections.jsonl", overwrite=True)
    audit = {"evidence_counts": evidence.counts, "benchmark_counts": result.counts,
             "complete": result.complete}
    write_json(audit, output_dir / "selection_audit.json", overwrite=True)
    result.require_complete()
    validate_benchmark(
        jobs, nodes, result.queries, result.qrels,
        expected_pairs=4 * config["benchmark"]["per_family"],
        expected_per_family=config["benchmark"]["per_family"],
    )
    sample_size = config["benchmark"]["review_sample_size"]
    sampled_pairs = set(sorted(
        {row["pair_id"] for row in result.queries},
        key=lambda pair_id: object_sha256([config["seed"], pair_id]),
    )[:sample_size])
    write_jsonl(
        [row for row in result.queries if row["pair_id"] in sampled_pairs],
        output_dir / "review_sample.jsonl", overwrite=True,
    )
    write_benchmark(result, output_dir)
    manifest.update({
        "frozen": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_candidates_sha256": file_sha256(candidate_path),
        "queries_sha256": file_sha256(query_path),
        "qrels_sha256": file_sha256(qrel_path),
        "evidence_counts": evidence.counts,
        "benchmark_counts": result.counts,
        "generator_snapshot": generator_snapshot,
        "query_count": len(result.queries),
        "pair_count": len(result.queries) // 2,
        "dependencies": installed_versions(),
    })
    _save_manifest(paths, manifest)
    print(f"froze {manifest['pair_count']} pairs / {manifest['query_count']} queries")


def stage_validate(config: dict[str, Any], paths: dict[str, Path]) -> None:
    jobs, digest = _corpus(config, paths)
    manifest = _read_manifest(paths)
    if manifest.get("dataset_sha256") != digest:
        raise ValueError("manifest and canonical corpus hashes differ")
    nodes = _nodes(paths, manifest)
    queries, qrels = _frozen(config, paths, jobs, nodes, manifest)
    print(f"validated {len(nodes)} nodes and {len(queries)} frozen queries / {len(qrels)} qrels")


def _run_id(value: str | None, config_hash: str) -> str:
    run_id = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + config_hash[:8]
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run ID must contain only letters, numbers, dots, dashes, or underscores")
    return run_id


def _verify_result_nodes(results: list[dict[str, Any]],
                         nodes: list[dict[str, Any]], top_k: int) -> None:
    valid_ids = {node["node_id"] for node in nodes}
    for row in results:
        ranked = row.get("ranked_node_ids")
        if not isinstance(ranked, list) or len(ranked) != top_k:
            raise ValueError(f"{row.get('query_id')}/{row.get('method')}: expected Top-{top_k}")
        if len(set(ranked)) != top_k or any(node_id not in valid_ids for node_id in ranked):
            raise ValueError(f"{row.get('query_id')}/{row.get('method')}: invalid ranked node IDs")


def stage_retrieve(config: dict[str, Any], paths: dict[str, Path], run_id_arg: str | None) -> None:
    from src.experiment.evaluate_retrieval import evaluate_results
    from src.experiment.retrievers import run_retrieval

    jobs, digest = _corpus(config, paths)
    manifest = _read_manifest(paths)
    if manifest.get("dataset_sha256") != digest:
        raise ValueError("manifest and canonical corpus hashes differ")
    nodes = _nodes(paths, manifest)
    queries, qrels = _frozen(config, paths, jobs, nodes, manifest)
    config_hash = object_sha256(config)
    run_id = _run_id(run_id_arg, config_hash)
    run_dir = paths["runs_dir"] / run_id
    if run_dir.exists():
        raise FileExistsError(f"run ID already exists: {run_dir}")
    model = config["models"]["dense"]
    dense_path = _local_model(model)
    dense_snapshot = model_snapshot_fingerprint(dense_path, model["revision"])
    retrieval = config["retrieval"]
    results = run_retrieval(
        nodes, queries, run_id=run_id, config_hash=config_hash,
        dense_model_path=dense_path, dense_model_id=model["model_id"],
        dense_device=model["device"],
        dense_use_fp16=model["use_fp16"],
        dense_batch_size=model["batch_size"],
        dense_query_max_length=model.get("query_max_length", 128),
        dense_passage_max_length=model["max_length"],
        bm25_k1=retrieval["bm25_k1"], bm25_b=retrieval["bm25_b"],
        top_k=retrieval["top_k"], rrf_k=retrieval["rrf_k"],
    )
    _verify_result_nodes(results, nodes, retrieval["top_k"])
    evaluate_results(results, qrels, queries, top_k=config["evaluation"]["top_k"])
    result_path = run_dir / "retrieval_results.jsonl"
    write_jsonl(results, result_path)
    run_manifest = {
        "run_id": run_id,
        "run_started_or_saved_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "benchmark_manifest_sha256": file_sha256(_manifest_path(paths)),
        "dataset_sha256": digest,
        "nodes_sha256": manifest["nodes_sha256"],
        "queries_sha256": manifest["queries_sha256"],
        "qrels_sha256": manifest["qrels_sha256"],
        "retrieval_results_sha256": file_sha256(result_path),
        "retrieval": retrieval,
        "dense_model": _model_record(model),
        "dense_local_path": str(dense_path),
        "dense_snapshot": dense_snapshot,
        "dependencies": installed_versions(),
        "result_row_count": len(results),
    }
    write_json(run_manifest, run_dir / "manifest.json")
    print(f"saved {len(results)} Top-20 result rows in {result_path}")


def stage_report(config: dict[str, Any], paths: dict[str, Path], run_id_arg: str | None) -> None:
    from src.experiment.analyze_pairs import summarize_pairs
    from src.experiment.evaluate_retrieval import evaluate_results
    from src.experiment.report import write_reports

    if run_id_arg is None:
        raise ValueError("report requires --run-id")
    run_id = _run_id(run_id_arg, object_sha256(config))
    jobs, digest = _corpus(config, paths)
    benchmark_manifest = _read_manifest(paths)
    nodes = _nodes(paths, benchmark_manifest)
    queries, qrels = _frozen(config, paths, jobs, nodes, benchmark_manifest)
    run_dir = paths["runs_dir"] / run_id
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("config_sha256") != object_sha256(config):
        raise ValueError("report config differs from the retrieval run config")
    if run_manifest.get("benchmark_manifest_sha256") != file_sha256(_manifest_path(paths)):
        raise ValueError("benchmark manifest differs from the retrieval run")
    result_path = run_dir / "retrieval_results.jsonl"
    if (run_manifest.get("dataset_sha256") != digest
            or run_manifest.get("nodes_sha256") != benchmark_manifest["nodes_sha256"]
            or run_manifest.get("queries_sha256") != benchmark_manifest["queries_sha256"]
            or run_manifest.get("qrels_sha256") != benchmark_manifest["qrels_sha256"]
            or run_manifest.get("retrieval_results_sha256") != file_sha256(result_path)):
        raise ValueError("run inputs/results no longer match their frozen manifests")
    results = read_jsonl(result_path)
    _verify_result_nodes(results, nodes, config["retrieval"]["top_k"])
    eval_config = config["evaluation"]
    scores = evaluate_results(results, qrels, queries, top_k=eval_config["top_k"])
    analysis = summarize_pairs(
        scores, bootstrap_samples=eval_config["bootstrap_samples"],
        seed=config["seed"], confidence=eval_config["bootstrap_confidence"],
    )
    report_dir = paths["report_dir"] / run_id
    output_paths = write_reports(
        analysis, results, queries, qrels, nodes, report_dir,
        failure_limit=eval_config["failure_case_limit"],
    )
    score_path = run_dir / "per_query_scores.jsonl"
    write_jsonl(scores, score_path, overwrite=True)
    run_manifest["per_query_scores_sha256"] = file_sha256(score_path)
    run_manifest["report_files"] = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in output_paths.items()
    }
    write_json(run_manifest, run_manifest_path, overwrite=True)
    print(f"wrote source-evidence reports in {report_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("cache-models", "smoke-models", "prepare", "benchmark", "validate", "retrieve", "report"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name in {"retrieve", "report"}:
            subparser.add_argument("--run-id")
    args = parser.parse_args(argv)
    config = _load_settings(args.config)
    paths = _paths(config)
    if args.stage == "cache-models":
        stage_cache_models(config)
    elif args.stage == "smoke-models":
        stage_smoke_models(config)
    elif args.stage == "prepare":
        stage_prepare(config, paths)
    elif args.stage == "benchmark":
        stage_benchmark(config, paths)
    elif args.stage == "validate":
        stage_validate(config, paths)
    elif args.stage == "retrieve":
        stage_retrieve(config, paths, args.run_id)
    else:
        stage_report(config, paths, args.run_id)


if __name__ == "__main__":
    main()
