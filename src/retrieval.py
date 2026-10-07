"""Retrieval: vector search (ChromaDB), keyword search (BM25) and hybrid search (Reciprocal Rank Fusion).

All three modes search over chunks and return ARTICLES (best first), each with its best matching chunks.
  vector   semantic similarity – finds "Datenpanne melden" even if the law says "Verletzung des Schutzes …"
  keyword  BM25 – exact terms, article numbers, abbreviations ("72 Stunden", "Art. 22", "FLOP")
  hybrid   both lists fused with RRF – the default, compared against the others in src/evaluate.py
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking import Chunk, chunk_records
from src.embeddings import Embedder

MODES = ("vector", "keyword", "hybrid")
CANDIDATE_CHUNKS = 60  # chunks fetched per method before grouping them into articles
FUSION_DEPTH = 20  # articles per method that enter the fusion

STOP_DE = set("der die das den dem des ein eine einer eines einem einen und oder ist sind wird werden nicht "
              "für mit von zu im in am an auf aus bei als wie was welche welcher welches wer wann ob es sie "
              "er auch nach über unter muss müssen darf dürfen kann können gilt gelten".split())
STOP_EN = set("the a an and or is are be been was were not for with of to in on at by as what which who when "
              "whether it they he she also after under must may can shall does do how".split())


def load_articles(path: Path | str = "data/articles.jsonl") -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_STEMMERS: dict = {}


def tokenize(text: str, lang: str | None = None) -> list[str]:
    """Lower case, drop stop words, reduce words to their stem ("Bußgelder" → "bussgeld", "fines" → "fine")."""
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in STOP_DE and w not in STOP_EN]
    if lang is None:
        return words
    if lang not in _STEMMERS:
        import snowballstemmer

        _STEMMERS[lang] = snowballstemmer.stemmer("german" if lang == "de" else "english")
    return _STEMMERS[lang].stemWords(words)


def detect_lang(text: str) -> str:
    """Very small language guess for the question: 'de' or 'en' (ties → 'de')."""
    words = re.findall(r"\w+", text.lower())
    de = sum(w in STOP_DE for w in words) + len(re.findall(r"[äöüß]", text.lower()))
    en = sum(w in STOP_EN for w in words)
    return "en" if en > de else "de"


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60, top_n: int = 5) -> list[tuple[str, float]]:
    """Combine ranked lists of ids: score(d) = Σ 1 / (k + rank_i(d)), rank starting at 1.

    k = 60 as in the original paper (Cormack, Clarke & Büttcher, SIGIR 2009). Only ranks matter, so the very
    different score scales of BM25 and cosine similarity need no normalisation.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def group_by_article(chunk_ids: Sequence[str]) -> dict[str, list[str]]:
    """Ranked chunk ids → ranked articles (first appearance counts), each with its chunks in rank order."""
    groups: dict[str, list[str]] = {}
    for cid in chunk_ids:
        groups.setdefault(cid.split("#")[0], []).append(cid)
    return groups


@dataclass
class Hit:
    doc_id: str
    score: float
    chunk_ids: list[str] = field(default_factory=list)


class Retriever:
    """One retrieval object for the API, the UI and the evaluation, so all three behave identically."""

    def __init__(
        self,
        records: list[dict],
        embedder: Embedder | None = None,
        chroma_dir: str = "chroma_db",
        collection: str = "eu_law",
    ):
        from rank_bm25 import BM25Okapi

        self.records = {r["id"]: r for r in records}
        self.chunks: dict[str, Chunk] = {c.id: c for c in chunk_records(records)}
        self._embedder = embedder
        self._chroma_dir, self._collection_name = chroma_dir, collection
        self._collection_obj = None
        self._bm25 = {}
        for lang in sorted({c.lang for c in self.chunks.values()}):
            ids = [cid for cid, c in self.chunks.items() if c.lang == lang]
            tokens = [tokenize(self.chunks[i].text, lang) for i in ids]
            self._bm25[lang] = (BM25Okapi(tokens), ids, [set(t) for t in tokens])

    # ---------- index ----------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from src.embeddings import FastEmbedder

            self._embedder = FastEmbedder()
        return self._embedder

    def _client(self):
        import chromadb

        return chromadb.PersistentClient(path=self._chroma_dir)

    @property
    def collection(self):
        if self._collection_obj is None:
            col = self._client().get_collection(self._collection_name)
            built_with = (col.metadata or {}).get("embedding_model")
            if built_with != self.embedder.name:
                raise RuntimeError(
                    f"Index was built with '{built_with}', but EMBEDDING_MODEL is '{self.embedder.name}'. "
                    "Rebuild it: python -m src.build_index --rebuild"
                )
            self._collection_obj = col
        return self._collection_obj

    def build_index(self, rebuild: bool = False, batch_size: int = 64, progress=print) -> int:
        client = self._client()
        if rebuild and self._collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(self._collection_name)
        col = client.get_or_create_collection(
            self._collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedder.name},
            embedding_function=None,
        )
        existing = set(col.get(include=[])["ids"])
        todo = [c for cid, c in self.chunks.items() if cid not in existing]
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            col.upsert(
                ids=[c.id for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[{"doc_id": c.doc_id, "lang": c.lang} for c in batch],
            )
            progress(f"  {min(start + batch_size, len(todo))}/{len(todo)} chunks embedded")
        self._collection_obj = None
        return col.count()

    # ---------- search ----------
    def vector_chunks(self, query: str, lang: str, n: int = CANDIDATE_CHUNKS) -> list[tuple[str, float]]:
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=n,
            where={"lang": lang},
            include=["distances"],
        )
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0], strict=True)]

    def keyword_chunks(self, query: str, lang: str, n: int = CANDIDATE_CHUNKS) -> list[tuple[str, float]]:
        if lang not in self._bm25:
            return []
        bm25, ids, token_sets = self._bm25[lang]
        q = tokenize(query, lang)
        scores = bm25.get_scores(q)
        matching = [i for i in range(len(ids)) if token_sets[i] & set(q)]  # at least one query word
        best = sorted(matching, key=lambda i: scores[i], reverse=True)[:n]
        return [(ids[i], float(scores[i])) for i in best]

    def search(self, query: str, mode: str = "hybrid", k: int = 5, lang: str | None = None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        lang = lang or detect_lang(query)
        vec = self.vector_chunks(query, lang) if mode in ("vector", "hybrid") else []
        kw = self.keyword_chunks(query, lang) if mode in ("keyword", "hybrid") else []
        v_groups, k_groups = group_by_article([c for c, _ in vec]), group_by_article([c for c, _ in kw])
        best = {}  # best chunk score per article, for display
        for cid, score in vec + kw:
            best.setdefault(cid.split("#")[0], score)

        if mode == "vector":
            ranked = [(d, best[d]) for d in list(v_groups)[:k]]
        elif mode == "keyword":
            ranked = [(d, best[d]) for d in list(k_groups)[:k]]
        else:
            ranked = reciprocal_rank_fusion([list(v_groups)[:FUSION_DEPTH], list(k_groups)[:FUSION_DEPTH]], top_n=k)

        hits = []
        for doc_id, score in ranked:
            chunks = list(dict.fromkeys(v_groups.get(doc_id, [])[:1] + k_groups.get(doc_id, [])[:1]
                                        + v_groups.get(doc_id, []) + k_groups.get(doc_id, [])))
            hits.append(Hit(doc_id, round(score, 4), chunks[:2]))
        return hits

    def get(self, doc_id: str) -> dict:
        return self.records[doc_id]
