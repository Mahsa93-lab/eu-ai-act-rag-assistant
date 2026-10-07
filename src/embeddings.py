"""Text embeddings – computed locally, so no legal text or question leaves the computer for the search step.

Default model: intfloat/multilingual-e5-large (≈ 2.2 GB, downloaded once into ./models, ~100 languages).
Smaller alternative for weak laptops: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (0.2 GB).
E5 models were trained with the prefixes "query: " and "passage: " – leaving them out costs retrieval quality.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL = "intfloat/multilingual-e5-large"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """ONNX runtime via fastembed – no PyTorch needed, runs on any CPU (also inside Docker)."""

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported here so unit tests run without the model

        self.name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
        self._is_e5 = "e5" in self.name.lower()
        self._model = TextEmbedding(self.name, cache_dir=cache_dir or os.getenv("MODEL_CACHE", "models"))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" if self._is_e5 else t for t in texts]
        return [v.tolist() for v in self._model.embed(prefixed, batch_size=16)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([f"query: {text}" if self._is_e5 else text]))).tolist()


class HashEmbedder:
    """Tiny deterministic stand-in for tests and CI: bag of hashed words, L2-normalised. Not for real use."""

    name = "hash-test-embedder"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
