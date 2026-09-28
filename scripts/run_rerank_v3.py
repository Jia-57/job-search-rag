"""Run the additive JD V3 reranker from a frozen V2 benchmark and run.

Stages: cache-model (network), smoke (GPU), run (GPU, offline). No V2 file is
opened for writing, and a V3 run ID cannot replace an existing result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

from src.experiment.evaluate_retrieval import evaluate_results
from src.experiment.io import (
    file_sha256, installed_versions, load_config, model_snapshot_fingerprint,
    object_sha256, read_jsonl, write_json, write_jsonl,
)
from src.experiment.rerank_v3 import analyze_v3, rerank_queries, write_v3_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "experiment_v3.yaml"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MODEL_ALLOW = ["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _settings(path: Path) -> dict[str, Any]:
    config = load_config(path)
    if config.get("experiment_version") != "jd_v3_rerank_1":
        raise ValueError("V3 requires experiment_version=jd_v3_rerank_1")
    candidate = config["candidate"]
    if (candidate.get("methods"), candidate.get("top_k_per_method"),
            candidate.get("deduplicate_by")) != (["bm25", "dense"], 20, "node_id"):
        raise ValueError("V3 fixes candidates to the union of BM25 and Dense Top-20 by node_id")
    evaluation = config["evaluation"]
    if evaluation.get("top_k") != 5:
        raise ValueError("V3 uses the same Top-5 metric as V2")
    reranker = config["reranker"]
    if (reranker.get("model_id") != "BAAI/bge-reranker-v2-m3"
            or not re.fullmatch(r"[0-9a-f]{40}", str(reranker.get("revision", "")))
            or reranker.get("normalize") is not False
            or reranker.get("batch_size", 0) < 1
            or reranker.get("max_length", 0) < 1):
        raise ValueError("V3 reranker settings must be pinned and valid")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("V3 run_id must use letters, numbers, dots, dashes or underscores")
    source = config["source"]
    paths = config["paths"]
    v2_benchmark = _path(source["benchmark_dir"]).resolve()
    v2_run = _path(source["run_dir"]).resolve()
    v3_benchmark = _path(paths["benchmark_snapshot_dir"]).resolve()
    v3_run = (_path(paths["runs_dir"]) / run_id).resolve()
    v3_report = (_path(paths["reports_dir"]) / run_id).resolve()
    for output in (v3_benchmark, v3_run, v3_report):
        if output == v2_benchmark or output == v2_run or output.is_relative_to(v2_benchmark) or output.is_relative_to(v2_run):
            raise ValueError("V3 outputs must be separate from V2 inputs")
    if len({v3_benchmark, v3_run, v3_report}) != 3:
        raise ValueError("V3 output directories must be distinct")
    return config


def _model_path(config: dict[str, Any], *, download: bool) -> Path:
    model = config["reranker"]
    local_path = model.get("local_path")
    if local_path:
        path = _path(local_path)
        if path.parent.name == "snapshots" and path.name != model["revision"]:
            raise ValueError("Local reranker snapshot path does not match pinned revision")
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(
            repo_id=model["model_id"], revision=model["revision"],
            local_files_only=not download, allow_patterns=_MODEL_ALLOW,
        ))
    if not path.is_dir():
        raise FileNotFoundError(f"Reranker model directory missing: {path}")
    for required in ("config.json", "tokenizer_config.json"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"Reranker snapshot lacks {required}: {path}")
    if not any((path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"Reranker snapshot lacks model weights: {path}")
    return path.resolve()


def _load_v2(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any],
                                                  list[dict[str, Any]], list[dict[str, Any]],
                                                  list[dict[str, Any]], list[dict[str, Any]]]:
    source = config["source"]
    benchmark_dir = _path(source["benchmark_dir"])
    run_dir = _path(source["run_dir"])
    paths = {
        "benchmark_dir": benchmark_dir,
        "run_dir": run_dir,
        "benchmark_manifest": benchmark_dir / "manifest.json",
        "nodes": benchmark_dir / "nodes.jsonl",
        "queries": benchmark_dir / "queries.jsonl",
        "qrels": benchmark_dir / "qrels.jsonl",
        "run_manifest": run_dir / "manifest.json",
        "results": run_dir / "retrieval_results.jsonl",
    }
    benchmark_manifest = json.loads(paths["benchmark_manifest"].read_text(encoding="utf-8"))
    run_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    if benchmark_manifest.get("benchmark_version") != "jd_v2" or benchmark_manifest.get("frozen") is not True:
        raise ValueError("Source benchmark is not frozen V2")
    expected = {
        "nodes": benchmark_manifest.get("nodes_sha256"),
        "queries": benchmark_manifest.get("queries_sha256"),
        "qrels": benchmark_manifest.get("qrels_sha256"),
    }
    for name, digest in expected.items():
        if not isinstance(digest, str) or file_sha256(paths[name]) != digest:
            raise ValueError(f"V2 {name} changed since the benchmark was frozen")
    if (run_manifest.get("benchmark_manifest_sha256") != file_sha256(paths["benchmark_manifest"])
            or run_manifest.get("nodes_sha256") != expected["nodes"]
            or run_manifest.get("queries_sha256") != expected["queries"]
            or run_manifest.get("qrels_sha256") != expected["qrels"]
            or run_manifest.get("dataset_sha256") != benchmark_manifest.get("dataset_sha256")
            or run_manifest.get("retrieval_results_sha256") != file_sha256(paths["results"])):
        raise ValueError("V2 run is inconsistent with its frozen benchmark")
    if run_manifest.get("run_id") != run_dir.name or not isinstance(run_manifest.get("config_sha256"), str):
        raise ValueError("V2 run manifest identity is invalid")
    nodes, queries, qrels, results = (
        read_jsonl(paths[name]) for name in ("nodes", "queries", "qrels", "results")
    )
    if (len(nodes) != benchmark_manifest.get("node_count")
            or len(queries) != benchmark_manifest.get("query_count")
            or len(qrels) != len(queries)
            or len(results) != 3 * len(queries)
            or len(queries) != 2 * benchmark_manifest.get("pair_count", -1)):
        raise ValueError("V2 row counts are inconsistent")
    valid_ids = {row["node_id"] for row in nodes}
    if len(valid_ids) != len(nodes):
        raise ValueError("V2 node IDs are duplicated")
    for row in results:
        ranked = row.get("ranked_node_ids")
        if (not isinstance(ranked, list) or len(ranked) != 20
                or len(set(ranked)) != 20 or any(node_id not in valid_ids for node_id in ranked)):
            raise ValueError("V2 Top-20 results contain invalid node IDs")
        if (row.get("run_id") != run_manifest["run_id"]
                or row.get("config_hash") != run_manifest["config_sha256"]):
            raise ValueError("V2 result rows do not match their run manifest")
    evaluate_results(results, qrels, queries, top_k=5)
    return paths, benchmark_manifest, run_manifest, nodes, queries, qrels, results


def _snapshot_v2_inputs(v2_paths: dict[str, Path], v3_dir: Path) -> dict[str, Path]:
    """Keep byte-for-byte frozen copies in the new V3 data directory."""
    v3_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, Path] = {}
    for name in ("benchmark_manifest", "nodes", "queries", "qrels"):
        dest_name = "source_v2_manifest.json" if name == "benchmark_manifest" else v2_paths[name].name
        destination = v3_dir / dest_name
        if destination.exists():
            if file_sha256(destination) != file_sha256(v2_paths[name]):
                raise ValueError(f"Existing V3 input snapshot differs from V2: {destination}")
        else:
            with v2_paths[name].open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            if file_sha256(destination) != file_sha256(v2_paths[name]):
                raise ValueError(f"V3 input snapshot copy failed: {destination}")
        copies[name] = destination
    return copies


def _reranker(config: dict[str, Any], model_path: Path) -> Any:
    import torch
    from FlagEmbedding import FlagReranker

    settings = config["reranker"]
    device = settings["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("V3 reranking requires a GPU on the requested CUDA device")
    return FlagReranker(
        str(model_path), use_fp16=settings["use_fp16"], devices=device,
        batch_size=settings["batch_size"], max_length=settings["max_length"],
        normalize=settings["normalize"], trust_remote_code=False,
    )


def stage_smoke(config: dict[str, Any]) -> None:
    import math

    model = _reranker(config, _model_path(config, download=False))
    scores = model.compute_score([
        ("Which role uses Kubernetes?", "The role requires Kubernetes experience."),
        ("Which role uses Kubernetes?", "The role plans retail pricing."),
    ])
    if len(scores) != 2 or not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Reranker smoke check returned invalid scores")
    if float(scores[0]) <= float(scores[1]):
        raise ValueError("Reranker smoke check ranked the unrelated text higher")
    print(f"V3 local reranker smoke passed: {scores}")


def stage_run(config: dict[str, Any]) -> None:
    v2_paths, benchmark_manifest, v2_manifest, nodes, queries, qrels, results = _load_v2(config)
    run_id = config["run_id"]
    v3_run_dir = _path(config["paths"]["runs_dir"]) / run_id
    v3_report_dir = _path(config["paths"]["reports_dir"]) / run_id
    if v3_run_dir.exists() or v3_report_dir.exists():
        raise FileExistsError("V3 run/report ID already exists; choose a new run_id")
    model_path = _model_path(config, download=False)
    model_fingerprint = model_snapshot_fingerprint(model_path, config["reranker"]["revision"])
    config_hash = object_sha256(config)
    model = _reranker(config, model_path)
    reranked = rerank_queries(results, queries, nodes, model, run_id=run_id, config_hash=config_hash)
    analysis = analyze_v3(
        results, reranked, queries, qrels,
        bootstrap_samples=config["evaluation"]["bootstrap_samples"],
        confidence=config["evaluation"]["bootstrap_confidence"],
        seed=config["evaluation"]["seed"],
    )
    snapshot = _snapshot_v2_inputs(v2_paths, _path(config["paths"]["benchmark_snapshot_dir"]))
    v3_run_dir.mkdir(parents=True)
    result_path = v3_run_dir / "reranked_results.jsonl"
    write_jsonl(reranked, result_path)
    report_files = write_v3_report(analysis, v3_report_dir)
    manifest = {
        "experiment_version": config["experiment_version"],
        "run_id": run_id,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "source_v2_run_id": v2_manifest["run_id"],
        "source_v2_benchmark_manifest_sha256": file_sha256(v2_paths["benchmark_manifest"]),
        "source_v2_run_manifest_sha256": file_sha256(v2_paths["run_manifest"]),
        "source_v2_results_sha256": file_sha256(v2_paths["results"]),
        "source_dataset_sha256": benchmark_manifest["dataset_sha256"],
        "v3_input_snapshot": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in snapshot.items()
        },
        "candidate": config["candidate"],
        "reranker": {key: value for key, value in config["reranker"].items() if key != "local_path"},
        "reranker_local_path": str(model_path),
        "reranker_snapshot": model_fingerprint,
        "evaluation": config["evaluation"],
        "dependencies": installed_versions(),
        "reranked_results_sha256": file_sha256(result_path),
        "result_row_count": len(reranked),
        "report_files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in report_files.items()
        },
    }
    write_json(manifest, v3_run_dir / "manifest.json")
    print(f"V3 reranked {len(reranked)} frozen queries; results: {result_path}")
    print(f"V3 report: {v3_report_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache-model", "smoke", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = _settings(args.config)
    if args.stage == "cache-model":
        print(f"Cached pinned reranker: {_model_path(config, download=True)}")
    elif args.stage == "smoke":
        stage_smoke(config)
    else:
        stage_run(config)


if __name__ == "__main__":
    main()
