from src.llm import ChatResult
from src.rag import DISCLAIMER, NO_ANSWER, answer, apply_guardrails, mask_pii, parse_citations


def test_pii_is_masked_but_legal_numbers_are_kept():
    text = mask_pii("Mail max.muster@example.de, Tel. +49 176 1234567 oder 089/12345678, "
                    "IBAN DE89 3704 0044 0532 0130 00 – gilt Verordnung (EU) 2024/1689 und Art. 99 (EUR 35000000)?")
    assert "example.de" not in text and "1234567" not in text and "12345678" not in text and "3704" not in text
    assert text.count("[entfernt]") == 4
    assert "2024/1689" in text and "35000000" in text


def test_parse_citations_all_formats():
    text = ("Verboten [AI Act Art. 5]; gilt ab 2025 [AI Act Art. 113, AI Act Artikel 5]. Meldung binnen 72 h "
            "[DSGVO Art. 33 Abs. 1]. Fine [GDPR Art. 83(5)]. Hochrisiko [AI Act Anhang III] / [AI Act Annex III].")
    assert parse_citations(text) == ["ai_act|art5", "ai_act|art113", "gdpr|art33", "gdpr|art83", "ai_act|annex_iii"]


def test_guardrail_keeps_cited_answer_and_adds_disclaimer_once():
    g = apply_guardrails("Binnen 72 Stunden [DSGVO Art. 33]. Hinweis: Dies ist keine Rechtsberatung.",
                         {"gdpr|art33"}, "de")
    assert g.text == f"Binnen 72 Stunden [DSGVO Art. 33].\n\n{DISCLAIMER['de']}"
    assert (g.cited, g.abstained, g.guardrail) == (["gdpr|art33"], False, None)


def test_guardrail_replaces_uncited_answer():
    g = apply_guardrails("The fine is 35 million euros.", {"ai_act|art99"}, "en")
    assert g.text.startswith(NO_ANSWER["en"])
    assert (g.abstained, g.guardrail) == (True, "no_valid_citation")


def test_guardrail_flags_citation_outside_context():
    g = apply_guardrails("Up to 7 % [AI Act Art. 99]; see also [AI Act Art. 101].", {"ai_act|art99"}, "en")
    assert (g.cited, g.invalid_citations, g.guardrail) == (["ai_act|art99"], ["ai_act|art101"], "unknown_citation")


def test_model_refusal_is_recognised():
    g = apply_guardrails("„Dazu finde ich keine Grundlage in den vorliegenden Artikeln.“", set(), "de")
    assert g.abstained and g.guardrail is None and g.text.startswith(NO_ANSWER["de"])


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, system, user, temperature=0.0):
        self.calls.append(user)
        return ChatResult(self.reply, prompt_tokens=1000, completion_tokens=100, latency_s=0.5)


def test_answer_end_to_end(retriever, monkeypatch):
    monkeypatch.setenv("PRICE_INPUT_PER_M", "0.25")
    monkeypatch.setenv("PRICE_OUTPUT_PER_M", "1.50")
    llm = FakeLLM("Unverzüglich, möglichst binnen 72 Stunden [DSGVO Art. 33].")
    a = answer("Ich bin max@example.de – wann muss eine Datenpanne gemeldet werden? 72 Stunden?", retriever, llm,
               mode="hybrid", k=2)
    assert a.lang == "de" and not a.abstained
    assert a.sources[0]["key"] == "gdpr|art33" and a.cited == ["gdpr|art33"]
    assert "[DSGVO Art. 33] Meldung von Verletzungen" in llm.calls[0]  # context block with citation label
    assert "max@example.de" not in llm.calls[0]  # PII never reaches the LLM
    assert llm.calls[0].endswith("ANTWORTSPRACHE: Deutsch")
    assert a.cost_usd == round((1000 * 0.25 + 100 * 1.50) / 1e6, 6)
    assert "context" not in a.to_dict() and a.to_dict(with_context=True)["context"]


def test_checkable_facts_dates_and_amounts():
    from src.rag import checkable_facts

    text = ("Ab dem 2. Februar 2025 bzw. 2 August 2026 (02.08.2027): bis zu 7 500 000 EUR oder 35000000 EUR, "
            "gemäß Art. 99 Abs. 5 der Verordnung (EU) 2024/1689 im Jahr 2024; 10^25 FLOP; 72 Stunden.")
    assert checkable_facts(text) == {"date:2025-02-02", "date:2026-08-02", "date:2027-08-02", "num:7500000",
                                     "num:35000000"}


def test_guardrail_rejects_date_that_is_not_in_the_context():
    # real case from the evaluation: correct date from the model's memory, cited to an article without it
    context = "[AI Act Art. 111] … bis zum 2. August 2027 … bis zum 31. Dezember 2030 …"
    g = apply_guardrails("Die Verbote gelten ab dem 2. Februar 2025. [AI Act Art. 111]", {"ai_act|art111"}, "de",
                         context)
    assert (g.abstained, g.guardrail, g.unsupported) == (True, "unsupported_fact", ["date:2025-02-02"])
    assert g.text.startswith(NO_ANSWER["de"])


def test_guardrail_accepts_amount_written_differently():
    context = "[AI Act Art. 99] … Geldbußen von bis zu 7 500 000 EUR oder … bis zu 1 % …"
    g = apply_guardrails("Bis zu 7.500.000 EUR oder 1 % [AI Act Art. 99].", {"ai_act|art99"}, "de", context)
    assert (g.abstained, g.guardrail, g.unsupported) == (False, None, [])
