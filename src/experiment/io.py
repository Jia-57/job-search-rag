"""Small, deterministic file and manifest helpers for the V2 experiment."""

from __future__ import annotations

import json
import os
import tempfile
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import yaml


PACKAGE_NAMES = (
    "job-search-rag", "pydantic", "PyYAML", "numpy",
    "torch", "transformers", "accelerate", "FlagEmbedding",
    "huggingface-hub", "pytest",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _atomic_bytes(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.",
                                                   suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace existing artifact: {path}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path,
                *, overwrite: bool = False) -> None:
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")
    _atomic_bytes(path, data, overwrite=overwrite)


def write_json(value: Any, path: Path, *, overwrite: bool = False) -> None:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n").encode("utf-8")
    _atomic_bytes(path, data, overwrite=overwrite)


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_snapshot_fingerprint(path: Path, revision: str) -> dict[str, Any]:
    """Hash the actual local model files, including weights, for provenance.

    A copied model directory may not carry its Hub commit identity. File hashes
    remain an exact record of the bytes loaded even in that case.
    """
    model_files = sorted(
        file for file in path.rglob("*")
        if file.is_file()
        and not any(part in {"onnx", "imgs", ".git", ".cache"}
                    for part in file.relative_to(path).parts)
        and file.suffix.lower() in {
            ".json", ".safetensors", ".bin", ".pt", ".model", ".txt",
        }
    )
    if not model_files:
        raise ValueError(f"no model/config/tokenizer files found in {path}")
    files = {
        str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": file_sha256(file)}
        for file in model_files
    }
    return {
        "configured_revision": revision,
        "hub_snapshot_path_matches_revision": (
            path.parent.name == "snapshots" and path.name == revision
        ),
        "files": files,
        "combined_sha256": object_sha256(files),
    }
