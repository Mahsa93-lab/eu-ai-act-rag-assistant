"""The two regulations in the corpus: identifiers, display names, source URLs and expected sizes.

Everything that depends on "which law / which language" lives here, so the parser, the retriever,
the guardrails and the UI all use the same names and the same citation labels.
"""
from __future__ import annotations

LAWS = {
    "ai_act": {
        "celex": "32024R1689",
        "name": {"de": "AI Act", "en": "AI Act"},
        "full_name": {"de": "Verordnung (EU) 2024/1689 (KI-Verordnung)", "en": "Regulation (EU) 2024/1689 (AI Act)"},
        "articles": 113,
        "annexes": 13,
    },
    "gdpr": {
        "celex": "32016R0679",
        "name": {"de": "DSGVO", "en": "GDPR"},
        "full_name": {"de": "Verordnung (EU) 2016/679 (DSGVO)", "en": "Regulation (EU) 2016/679 (GDPR)"},
        "articles": 99,
        "annexes": 0,
    },
}
LANGS = ("de", "en")

# all names a citation may use for a law (an English answer may still say "DSGVO")
LAW_ALIASES = {"ai act": "ai_act", "ki-verordnung": "ai_act", "ki-vo": "ai_act", "dsgvo": "gdpr", "gdpr": "gdpr"}


def eurlex_url(law: str, lang: str, anchor: str | None = None) -> str:
    """Official HTML page of the regulation; `anchor` jumps to an article (art_5) or annex (anx_III)."""
    url = f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{LAWS[law]['celex']}"
    return f"{url}#{anchor}" if anchor else url


def doc_key(law: str, kind: str, number: str) -> str:
    """Language-independent key, e.g. 'ai_act|art5' or 'ai_act|annex_iii'. Used in the test set and in metrics."""
    return f"{law}|art{number}" if kind == "article" else f"{law}|annex_{number.lower()}"


def doc_id(law: str, lang: str, kind: str, number: str) -> str:
    """Unique id of one article in one language, e.g. 'ai_act|de|art5'."""
    law_, rest = doc_key(law, kind, number).split("|")
    return f"{law_}|{lang}|{rest}"


def key_of(doc_id_: str) -> str:
    """'ai_act|de|art5' → 'ai_act|art5'."""
    law, _lang, rest = doc_id_.split("|")
    return f"{law}|{rest}"


def label(law: str, lang: str, kind: str, number: str) -> str:
    """Citation label as the model must write it: 'AI Act Art. 5', 'DSGVO Art. 33', 'AI Act Anhang III'."""
    unit = "Art." if kind == "article" else ("Anhang" if lang == "de" else "Annex")
    return f"{LAWS[law]['name'][lang]} {unit} {number}"
