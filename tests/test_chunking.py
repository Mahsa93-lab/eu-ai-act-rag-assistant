from src.chunking import chunk_record, header


def make(text, title="Begriffsbestimmungen", chapter=""):
    return {"id": "ai_act|de|art3", "lang": "de", "label": "AI Act Art. 3", "title": title, "chapter": chapter,
            "text": text}


def test_short_article_is_one_chunk_with_header():
    chunks = chunk_record(make("(1) KI-System …"))
    assert len(chunks) == 1
    assert chunks[0].id == "ai_act|de|art3#c0"
    assert chunks[0].text == "AI Act Art. 3: Begriffsbestimmungen\n(1) KI-System …"


def test_long_article_split_at_paragraphs_never_above_limit():
    paragraphs = [f"({i}) " + "Wort " * 60 for i in range(1, 30)]  # ~300 chars each
    chunks = chunk_record(make("\n".join(paragraphs)), max_chars=1000)
    assert len(chunks) > 5
    assert all(len(c.body) <= 1000 for c in chunks)
    assert all(c.body.startswith("(") for c in chunks)  # every chunk starts at a paragraph
    assert "".join(c.body.replace("\n", "") for c in chunks) == "".join(paragraphs)  # nothing lost


def test_very_long_paragraph_split_at_sentence_end():
    text = " ".join(f"Satz {i} endet hier." for i in range(200))
    chunks = chunk_record(make(text), max_chars=500)
    assert all(len(c.body) <= 500 for c in chunks)
    assert all(c.body.endswith("hier.") for c in chunks)


def test_header_contains_chapter():
    assert header(make("x", chapter="KAPITEL I – ALLGEMEINE BESTIMMUNGEN")) == (
        "AI Act Art. 3: Begriffsbestimmungen (KAPITEL I – ALLGEMEINE BESTIMMUNGEN)")
