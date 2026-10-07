"""Build the search index: data/articles.jsonl → chunks → local embeddings → ChromaDB (./chroma_db).

  python -m src.build_index            # adds missing chunks (safe to re-run)
  python -m src.build_index --rebuild  # after changing EMBEDDING_MODEL or the chunk size

First run downloads the embedding model (~2.2 GB for multilingual-e5-large) and takes several minutes on a CPU.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from dotenv import load_dotenv

from src.retrieval import Retriever, load_articles


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rebuild", action="store_true", help="delete and rebuild the collection")
    args = p.parse_args()

    start = time.perf_counter()
    retriever = Retriever(load_articles())
    per_lang = Counter(c.lang for c in retriever.chunks.values())
    print(f"{len(retriever.records)} articles/annexes → {len(retriever.chunks)} chunks {dict(per_lang)}")
    print(f"Embedding model: {retriever.embedder.name}")
    total = retriever.build_index(rebuild=args.rebuild)
    print(f"Index ready: {total} chunks in ./chroma_db  ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
