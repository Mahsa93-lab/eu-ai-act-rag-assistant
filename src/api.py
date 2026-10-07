"""REST API – used by the Streamlit UI and later by the KPI agent (project 4).

  uvicorn src.api:app --reload      → http://127.0.0.1:8000/docs

  GET  /health   index size, models
  POST /search   retrieval only (no LLM, no API key needed)
  POST /ask      answer with citations and guardrails
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm import LLM
from src.rag import answer, mask_pii, sources_of
from src.retrieval import Retriever, detect_lang, load_articles

load_dotenv()
app = FastAPI(title="EU AI Act & GDPR Assistant", version="1.0.0")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Was regelt Artikel 4 der KI-Verordnung?"])
    mode: Literal["vector", "hybrid", "keyword"] = "vector"  # best in the evaluation
    lang: Literal["de", "en"] | None = None
    k: int = Field(5, ge=1, le=10)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_articles())


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM()


@app.get("/health")
def health() -> dict:
    r = get_retriever()
    return {"status": "ok", "articles": len(r.records), "chunks": len(r.chunks), "embedding_model": r.embedder.name}


@app.post("/search")
def search(q: Question) -> dict:
    r = get_retriever()
    clean_q = mask_pii(q.question)
    lang = q.lang or detect_lang(clean_q)
    hits = r.search(clean_q, mode=q.mode, k=q.k, lang=lang)
    return {"lang": lang, "mode": q.mode, "sources": sources_of(r, hits)}


@app.post("/ask")
def ask(q: Question) -> dict:
    try:
        llm = get_llm()
    except RuntimeError as exc:  # CHAT_MODEL missing
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return answer(q.question, get_retriever(), llm, mode=q.mode, k=q.k, lang=q.lang).to_dict()
