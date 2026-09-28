"""Local BM25, BGE-M3 dense, and rank-fusion retrieval over frozen JD nodes.

All methods rank the same ``retrieval_text`` field. No retrieval method sees the
source evidence labels. The dense model must already exist on the cluster; this
module never downloads weights or calls a hosted inference endpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any


METHODS = ("bm25", "dense", "hybrid")
TOKENIZER_PATTERN = (
    r"(?<![A-Za-z0-9_])(?:c\+\+(?:\d+)?|c#(?:\d+)?|\.net|node\.js|ci/cd)"
    r"(?![A-Za-z0-9_])|[A-Za-z0-9]+(?:[._/+:-][A-Za-z0-9]+)*"
)
_TOKEN_RE = re.compile(TOKENIZER_PATTERN, re.IGNORECASE)


def tokenize_bm25(text: str) -> list[str]:
    """Case-fold terms; retain technical forms, with no stemming or stopwords."""

    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _node_rows(nodes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(nodes)
    if not rows:
        raise ValueError("The node corpus is empty")
    ids = [row.get("node_id") for row in rows]
    if any(not isinstance(node_id, str) or not node_id for node_id in ids):
        raise ValueError("Every node needs a nonempty string node_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate node_id in the corpus")
    if any(not isinstance(row.get("retrieval_text"), str) for row in rows):
        raise ValueError("Every node needs retrieval_text")
    return rows


class BM25Retriever:
    """BM25 with Robertson IDF, k1=1.5, b=0.75, and binary query terms.

    IDF is ``log(1 + (N - df + 0.5) / (df + 0.5))``. The implementation keeps
    a local in-memory inverted index and orders equal scores by stable node ID.
    """

    def __init__(
        self,
        nodes: Sequence[Mapping[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.nodes = _node_rows(nodes)
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = float(k1)
        self.b = float(b)
        self.ids = [str(row["node_id"]) for row in self.nodes]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for index, row in enumerate(self.nodes):
            counts = Counter(tokenize_bm25(str(row["retrieval_text"])))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((index, tf))
        self.avg_length = sum(self.lengths) / len(self.lengths)

    def rank(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = [0.0] * len(self.ids)
        n = len(self.ids)
        # Each distinct query term contributes once; repeated prose cannot
        # inflate the score of a document by repeating a word.
        for term in dict.fromkeys(tokenize_bm25(query)):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for index, tf in postings:
                length_ratio = self.lengths[index] / self.avg_length if self.avg_length else 0.0
                denominator = tf + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += idf * tf * (self.k1 + 1) / denominator
        indices = sorted(range(n), key=lambda index: (-scores[index], self.ids[index]))
        return [
            {"node_id": self.ids[index], "rank": rank, "score": scores[index]}
            for rank, index in enumerate(indices[:top_k], start=1)
        ]


def _normalized_vectors(vectors: Any, *, expected_rows: int) -> Any:
    """Return finite float32 unit vectors, independently of model defaults."""

    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] == 0:
        raise ValueError(f"Expected {expected_rows} nonempty dense vectors")
    if not np.isfinite(array).all():
        raise ValueError("Dense vectors contain NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Dense vectors contain a zero vector")
    return array / norms


class DenseRetriever:
    """Exact cosine search using locally loaded BGE-M3 dense embeddings."""

    def __init__(
        self,
        nodes: Sequence[Mapping[str, Any]],
        *,
        model_path: str | Path,
        device: str = "cuda:0",
        use_fp16: bool = True,
        batch_size: int = 16,
        query_max_length: int = 128,
        passage_max_length: int = 1024,
    ) -> None:
        self.nodes = _node_rows(nodes)
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"BGE-M3 local model directory not found: {path}. "
                "Download a pinned model snapshot before running retrieval."
            )
        if batch_size < 1 or query_max_length < 1 or passage_max_length < 1:
            raise ValueError("Dense batch size and token limits must be positive")
        if not device:
            raise ValueError("Dense device must be explicit, e.g. cuda:0 or cpu")
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            str(path.resolve()),
            devices=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
            pooling_method="cls",
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            trust_remote_code=False,
        )
        self.ids = [str(row["node_id"]) for row in self.nodes]
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        start = perf_counter()
        output = self.model.encode_corpus(
            [str(row["retrieval_text"]) for row in self.nodes],
            batch_size=batch_size,
            max_length=passage_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        self.corpus_vectors = _normalized_vectors(output["dense_vecs"], expected_rows=len(self.nodes))
        self.corpus_encoding_ms = (perf_counter() - start) * 1000

    def rank_many(
        self, queries: Sequence[str], *, top_k: int = 20
    ) -> tuple[list[list[dict[str, Any]]], float, float]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_list = list(queries)
        if not query_list:
            return [], 0.0, 0.0
        import numpy as np

        start = perf_counter()
        output = self.model.encode_queries(
            query_list,
            batch_size=self.batch_size,
            max_length=self.query_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_vectors = _normalized_vectors(output["dense_vecs"], expected_rows=len(query_list))
        encoding_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        similarities = np.matmul(query_vectors, self.corpus_vectors.T)
        rankings: list[list[dict[str, Any]]] = []
        for row in similarities:
            indices = sorted(range(len(self.ids)), key=lambda index: (-float(row[index]), self.ids[index]))
            rankings.append([
                {"node_id": self.ids[index], "rank": rank, "score": float(row[index])}
                for rank, index in enumerate(indices[:top_k], start=1)
            ])
        ranking_ms = (perf_counter() - start) * 1000
        return rankings, encoding_ms, ranking_ms


def rrf_fuse(
    bm25_results: Sequence[Mapping[str, Any]],
    dense_results: Sequence[Mapping[str, Any]],
    *,
    k: int = 60,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion of two Top-20 lists, deduplicated by node ID."""

    if k < 1 or top_k < 1:
        raise ValueError("RRF k and top_k must be positive")
    scores: dict[str, float] = defaultdict(float)
    rank_maps: list[dict[str, int]] = []
    for rows in (bm25_results, dense_results):
        seen: set[str] = set()
        ranks: dict[str, int] = {}
        for expected_rank, row in enumerate(rows, start=1):
            node_id = row.get("node_id")
            if not isinstance(node_id, str) or not node_id or node_id in seen:
                raise ValueError("An input ranking contains a missing or duplicate node_id")
            if row.get("rank", expected_rank) != expected_rank:
                raise ValueError("Input ranks must be contiguous and start at one")
            seen.add(node_id)
            ranks[node_id] = expected_rank
            scores[node_id] += 1.0 / (k + expected_rank)
        rank_maps.append(ranks)
    ids = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))
    return [
        {
            "node_id": node_id,
            "rank": rank,
            "score": scores[node_id],
            "bm25_rank": rank_maps[0].get(node_id),
            "dense_rank": rank_maps[1].get(node_id),
        }
        for rank, node_id in enumerate(ids[:top_k], start=1)
    ]


def run_retrieval(
    nodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    config_hash: str,
    dense_model_path: str | Path,
    dense_model_id: str = "BAAI/bge-m3",
    dense_device: str = "cuda:0",
    dense_use_fp16: bool = True,
    dense_batch_size: int = 16,
    dense_query_max_length: int = 128,
    dense_passage_max_length: int = 1024,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Run all three methods and return deterministic-order JSONL-ready rows.

    ``elapsed_ms`` for dense includes its per-query share of batched query
    encoding and ranking. Corpus encoding is reported separately because it is
    shared across the entire run. This avoids timing each query as if it had
    loaded and encoded the corpus again.
    """

    node_rows = _node_rows(nodes)
    query_rows = list(queries)
    if not query_rows or len({row.get("query_id") for row in query_rows}) != len(query_rows):
        raise ValueError("Queries must be nonempty and have unique query_id values")
    if any(not isinstance(row.get("query_id"), str) or not isinstance(row.get("query"), str) for row in query_rows):
        raise ValueError("Each query needs string query_id and query")
    if not run_id or not config_hash:
        raise ValueError("run_id and config_hash are required")
    if top_k != 20 or rrf_k != 60:
        raise ValueError("V2 fixes first-stage/output top_k=20 and RRF k=60")

    start = perf_counter()
    bm25 = BM25Retriever(node_rows, k1=bm25_k1, b=bm25_b)
    bm25_index_ms = (perf_counter() - start) * 1000
    dense = DenseRetriever(
        node_rows,
        model_path=dense_model_path,
        device=dense_device,
        use_fp16=dense_use_fp16,
        batch_size=dense_batch_size,
        query_max_length=dense_query_max_length,
        passage_max_length=dense_passage_max_length,
    )
    dense_rankings, query_encoding_ms, dense_ranking_ms = dense.rank_many(
        [str(row["query"]) for row in query_rows], top_k=top_k
    )
    amortized_dense_ms = (query_encoding_ms + dense_ranking_ms) / len(query_rows)
    rows: list[dict[str, Any]] = []
    for query, dense_ranking in zip(query_rows, dense_rankings, strict=True):
        start = perf_counter()
        bm25_ranking = bm25.rank(str(query["query"]), top_k=top_k)
        bm25_elapsed_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        hybrid_ranking = rrf_fuse(bm25_ranking, dense_ranking, k=rrf_k, top_k=top_k)
        hybrid_elapsed_ms = (perf_counter() - start) * 1000
        for method, ranking, elapsed_ms in (
            ("bm25", bm25_ranking, bm25_elapsed_ms),
            ("dense", dense_ranking, amortized_dense_ms),
            ("hybrid", hybrid_ranking, bm25_elapsed_ms + amortized_dense_ms + hybrid_elapsed_ms),
        ):
            rows.append({
                "run_id": run_id,
                "query_id": query["query_id"],
                "method": method,
                "ranked_node_ids": [item["node_id"] for item in ranking],
                "ranked_results": ranking,
                "config_hash": config_hash,
                "elapsed_ms": elapsed_ms,
                "timing_note": "Query encoding is batched and apportioned equally; shared index/corpus setup excluded",
                "bm25_index_ms": bm25_index_ms,
                "dense_corpus_encoding_ms": dense.corpus_encoding_ms,
                "dense_query_encoding_batch_ms": query_encoding_ms,
                "dense_ranking_batch_ms": dense_ranking_ms,
                "dense_model_id": dense_model_id,
            })
    return rows
