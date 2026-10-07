import pytest

from src.embeddings import HashEmbedder
from src.laws import doc_id, doc_key, eurlex_url, label


def rec(law, lang, kind, number, title, text, chapter=""):
    return {"id": doc_id(law, lang, kind, number), "key": doc_key(law, kind, number),
            "label": label(law, lang, kind, number), "url": eurlex_url(law, lang), "law": law, "lang": lang,
            "kind": kind, "number": number, "title": title, "chapter": chapter, "text": text}


RECORDS = [
    rec("gdpr", "de", "article", "33", "Meldung von Verletzungen des Schutzes personenbezogener Daten",
        "(1) Im Falle einer Verletzung des Schutzes personenbezogener Daten meldet der Verantwortliche unverzüglich "
        "und möglichst binnen 72 Stunden der Aufsichtsbehörde."),
    rec("gdpr", "en", "article", "33", "Notification of a personal data breach",
        "1. In the case of a personal data breach, the controller shall without undue delay and, where feasible, "
        "not later than 72 hours after having become aware of it, notify the supervisory authority."),
    rec("ai_act", "de", "article", "4", "KI-Kompetenz",
        "Die Anbieter und Betreiber von KI-Systemen ergreifen Maßnahmen, um sicherzustellen, dass ihr Personal "
        "über ein ausreichendes Maß an KI-Kompetenz verfügt."),
    rec("ai_act", "en", "article", "4", "AI literacy",
        "Providers and deployers of AI systems shall take measures to ensure a sufficient level of AI literacy "
        "of their staff."),
    rec("ai_act", "en", "article", "99", "Penalties",
        "3. Non-compliance with the prohibition of the AI practices referred to in Article 5 shall be subject to "
        "administrative fines of up to EUR 35000000 or up to 7 % of its total worldwide annual turnover."),
    rec("ai_act", "en", "annex", "III", "High-risk AI systems referred to in Article 6(2)",
        "4. Employment, workers management: (a) AI systems intended to be used for the recruitment or selection "
        "of natural persons, to analyse and filter job applications, and to evaluate candidates;"),
]


@pytest.fixture()
def records():
    return [dict(r) for r in RECORDS]


@pytest.fixture()
def retriever(records, tmp_path):
    from src.retrieval import Retriever

    r = Retriever(records, embedder=HashEmbedder(), chroma_dir=str(tmp_path / "chroma"))
    r.build_index(progress=lambda _msg: None)
    return r
