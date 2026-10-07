"""Answer generation with citations and guardrails.

Flow:  question → mask personal data → search (vector / keyword / hybrid) → context with article labels
       → LLM with strict system prompt → guardrails check the answer → answer + sources + cost.

Guardrails (deterministic code, not left to the model):
  1. PII masking   e-mail, phone numbers and IBANs are replaced before search and before the LLM call
  2. No context    no search hit → fixed "no answer" sentence, no LLM call
  3. No source     an answer without at least one citation of a retrieved article is replaced by the
                   "no answer" sentence – an uncited statement about the law is worse than no answer
  4. Unknown cite  citations of articles that were NOT in the context are flagged (possible hallucination)
  5. Facts check   every date and every amount (≥ 4 digits) in the answer must also appear in the context –
                   otherwise the model used its own memory and the answer is replaced by the "no answer"
                   sentence (found in the evaluation: a correct date, cited to an article that does not contain it)
  6. Disclaimer    "keine Rechtsberatung" is always appended by the code, in the language of the question
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.laws import LAW_ALIASES
from src.llm import LLM, cost_usd
from src.retrieval import Retriever, detect_lang

SYSTEM_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
NO_ANSWER = {
    "de": "Dazu finde ich keine Grundlage in den vorliegenden Artikeln.",
    "en": "I cannot find a basis for this in the provided articles.",
}
ANSWER_LANGUAGE = {"de": "Deutsch", "en": "Englisch (answer in English)"}
DISCLAIMER = {"de": "Hinweis: Dies ist keine Rechtsberatung.", "en": "Note: This is not legal advice."}

PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),  # e-mail
    re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"),  # IBAN
    re.compile(r"(?<![\w/])(?:\+|00)\d{1,3}[ /-]?\d[\d /-]{6,}\d"),  # +49 176 1234567
    re.compile(r"(?<![\w/.])0\d{2,5}[ /-]?\d[\d /-]{4,}\d"),  # 0176 1234567, 089/12345678
]
_LAW = r"(AI Act|KI-Verordnung|KI-VO|DSGVO|GDPR)"
ARTICLE_CITE = re.compile(_LAW + r"\s*(?:Art\.?|Artikel|Article)\s*(\d+)", re.IGNORECASE)
ANNEX_CITE = re.compile(_LAW + r"\s*(?:Anhang|Annex)\s+([IVXLC]+)\b", re.IGNORECASE)


_MONTHS = ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober",
           "november", "dezember", "january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
MONTH_NO = {m: i % 12 + 1 for i, m in enumerate(_MONTHS)}
DATE = re.compile(r"\b(\d{1,2})\.?\s+(" + "|".join(dict.fromkeys(_MONTHS)) + r")\s+(\d{4})\b", re.IGNORECASE)
NUM_DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
NUMBER = re.compile(r"\d[\d\s.,\u202f\u00a0]*\d|\d")
REFERENCE = re.compile(r"\[[^\]]*\]|\b\d{2,4}/\d{1,4}\b|"  # citations, regulation numbers like 2024/1689
                       r"(?:Art(?:ikel|icle)?\.?|Abs(?:atz)?\.?|Nr\.?|point|paragraph|Anhang|Annex|\(EU\)|"
                       r"(?:EU|EG|EWG)\)?\s*Nr\.?|Verordnung|Regulation|Richtlinie|Directive)\s*[\d/()a-z]*",
                       re.IGNORECASE)


def mask_pii(text: str) -> str:
    for pattern in PII_PATTERNS:
        text = pattern.sub("[entfernt]", text)
    return text


def parse_citations(text: str) -> list[str]:
    """All citations inside [...] as language-independent keys, e.g. ['ai_act|art5', 'gdpr|art33']."""
    keys = []
    for inner in re.findall(r"\[([^\[\]]+)\]", text):
        for law, no in ARTICLE_CITE.findall(inner):
            keys.append(f"{LAW_ALIASES[law.lower()]}|art{no}")
        for law, no in ANNEX_CITE.findall(inner):
            keys.append(f"{LAW_ALIASES[law.lower()]}|annex_{no.lower()}")
    return list(dict.fromkeys(keys))


def checkable_facts(text: str) -> set[str]:
    """Dates and amounts in a text, normalised: '2. Februar 2025' → 'date:2025-02-02', '7 500 000' → 'num:7500000'.
    Article numbers, regulation numbers and plain years are ignored."""
    facts = {f"date:{int(y)}-{MONTH_NO[m.lower()]:02d}-{int(d):02d}" for d, m, y in DATE.findall(text)}
    facts |= {f"date:{int(y)}-{int(m):02d}-{int(d):02d}" for d, m, y in NUM_DATE.findall(text)}
    rest = REFERENCE.sub(" ", DATE.sub(" ", NUM_DATE.sub(" ", text)))
    for match in NUMBER.findall(rest):
        digits = re.sub(r"\D", "", match)
        if len(digits) >= 4 and not (len(digits) == 4 and digits[:2] in ("19", "20")):
            facts.add(f"num:{digits}")
    return facts


def unsupported_facts(answer_text: str, context: str) -> list[str]:
    return sorted(checkable_facts(answer_text) - checkable_facts(context))


def _is_no_answer(text: str) -> bool:
    plain = re.sub(r"[„“\"'.]", "", text).lower()
    return any(re.sub(r"[.]", "", s).lower() in plain for s in NO_ANSWER.values())


@dataclass
class GuardrailResult:
    text: str
    cited: list[str]
    invalid_citations: list[str]
    abstained: bool
    guardrail: str | None
    unsupported: list[str] = field(default_factory=list)


def apply_guardrails(text: str, allowed_keys: set[str], lang: str, context: str = "") -> GuardrailResult:
    for d in DISCLAIMER.values():  # the model was told not to add it; remove it if it did anyway
        text = text.replace(d, "").strip()
    cited = parse_citations(text)
    invalid = [c for c in cited if c not in allowed_keys]
    valid = [c for c in cited if c in allowed_keys]
    unsupported = unsupported_facts(text, context) if context else []
    guardrail = None
    if _is_no_answer(text):
        text, abstained = NO_ANSWER[lang], True
    elif not valid:
        text, abstained, guardrail = NO_ANSWER[lang], True, "no_valid_citation"
    elif unsupported:
        text, abstained, guardrail = NO_ANSWER[lang], True, "unsupported_fact"
    else:
        abstained = False
        guardrail = "unknown_citation" if invalid else None
    return GuardrailResult(f"{text}\n\n{DISCLAIMER[lang]}", valid, invalid, abstained, guardrail, unsupported)


@dataclass
class Answer:
    question: str
    text: str
    lang: str
    mode: str
    sources: list[dict] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_facts: list[str] = field(default_factory=list)
    abstained: bool = False
    guardrail: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    context: str = ""  # what the LLM saw – needed by the evaluation (faithfulness), not sent by the API

    def to_dict(self, with_context: bool = False) -> dict:
        d = asdict(self)
        if not with_context:
            d.pop("context")
        return d


def build_context(retriever: Retriever, hits) -> str:
    blocks = []
    for hit in hits:
        rec = retriever.get(hit.doc_id)
        body = "\n[…]\n".join(retriever.chunks[c].body for c in sorted(hit.chunk_ids, key=_chunk_no))
        blocks.append(f"[{rec['label']}] {rec['title']}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _chunk_no(chunk_id: str) -> int:
    return int(chunk_id.rsplit("#c", 1)[1])


def sources_of(retriever: Retriever, hits) -> list[dict]:
    out = []
    for hit in hits:
        rec = retriever.get(hit.doc_id)
        excerpt = retriever.chunks[hit.chunk_ids[0]].body if hit.chunk_ids else ""
        out.append({"id": rec["id"], "key": rec["key"], "label": rec["label"], "title": rec["title"],
                    "url": rec["url"], "score": hit.score, "excerpt": excerpt[:600]})
    return out


def answer(question: str, retriever: Retriever, llm: LLM, mode: str = "vector", k: int = 5,
           lang: str | None = None) -> Answer:
    start = time.perf_counter()  # latency = search + LLM, as the user experiences it
    clean_q = mask_pii(question)
    lang = lang or detect_lang(clean_q)
    hits = retriever.search(clean_q, mode=mode, k=k, lang=lang)
    if not hits:
        return Answer(question, f"{NO_ANSWER[lang]}\n\n{DISCLAIMER[lang]}", lang, mode, abstained=True,
                      guardrail="no_context")

    context = build_context(retriever, hits)
    # the answer language is set explicitly: with a German system prompt the model sometimes answered
    # English questions in German (seen in the first evaluation run, q24)
    user = f"KONTEXT:\n{context}\n\nFRAGE:\n{clean_q}\n\nANTWORTSPRACHE: {ANSWER_LANGUAGE[lang]}"
    res = llm.chat(SYSTEM_PROMPT, user)
    sources = sources_of(retriever, hits)
    g = apply_guardrails(res.text, {s["key"] for s in sources}, lang, context)
    return Answer(
        question=question, text=g.text, lang=lang, mode=mode, sources=sources, cited=g.cited,
        invalid_citations=g.invalid_citations, unsupported_facts=g.unsupported, abstained=g.abstained,
        guardrail=g.guardrail,
        latency_s=round(time.perf_counter() - start, 2),
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        cost_usd=round(cost_usd(res.prompt_tokens, res.completion_tokens), 6),
        context=context,
    )
