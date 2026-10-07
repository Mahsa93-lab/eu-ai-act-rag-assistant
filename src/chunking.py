"""Split long articles into chunks for search – the article stays the unit that is cited.

Embedding models read at most ~512 tokens. Article 3 of the AI Act (definitions) has ~17,000 characters, so
embedding the whole article would only "see" its first definitions. Each chunk therefore gets
  * a header with law, article and title (so every chunk knows where it comes from), and
  * whole paragraphs only (split at line breaks, very long paragraphs at sentence ends).
Search runs over chunks; results are grouped back to articles ("small-to-big" retrieval).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 1400  # ≈ 350–450 tokens of German legal text, leaves room for the header within 512 tokens


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#c<n>"
    doc_id: str
    lang: str
    header: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.header}\n{self.body}"


def header(rec: dict) -> str:
    h = f"{rec['label']}: {rec['title']}" if rec.get("title") else rec["label"]
    return f"{h} ({rec['chapter']})" if rec.get("chapter") else h


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Paragraph longer than max_chars → pieces at sentence/clause ends; hard cut only as last resort."""
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", paragraph):
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_record(rec: dict, max_chars: int = MAX_CHARS) -> list[Chunk]:
    paragraphs = []
    for line in rec["text"].split("\n"):
        paragraphs += _split_long(line, max_chars) if len(line) > max_chars else [line]

    bodies, current = [], ""
    for p in paragraphs:
        if current and len(current) + 1 + len(p) > max_chars:
            bodies.append(current)
            current = p
        else:
            current = f"{current}\n{p}" if current else p
    if current:
        bodies.append(current)

    h = header(rec)
    return [Chunk(f"{rec['id']}#c{i}", rec["id"], rec["lang"], h, body) for i, body in enumerate(bodies)]


def chunk_records(records: list[dict], max_chars: int = MAX_CHARS) -> list[Chunk]:
    return [c for rec in records for c in chunk_record(rec, max_chars)]
