import pytest

from src.retrieval import detect_lang, group_by_article, reciprocal_rank_fusion, tokenize


def test_rrf_rewards_documents_found_by_both_methods():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "a"]], top_n=4)
    assert [d for d, _ in fused][:2] == ["a", "c"]  # rank 1+3 beats rank 2 alone
    assert {d for d, _ in fused} == {"a", "b", "c", "d"}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_respects_top_n():
    assert len(reciprocal_rank_fusion([["a", "b", "c"]], top_n=2)) == 2


def test_group_chunks_by_article_keeps_rank_order():
    groups = group_by_article(["x|de|art5#c2", "x|de|art3#c0", "x|de|art5#c0"])
    assert list(groups) == ["x|de|art5", "x|de|art3"]
    assert groups["x|de|art5"] == ["x|de|art5#c2", "x|de|art5#c0"]


def test_tokenize_stems_and_drops_stop_words():
    assert tokenize("Die Bußgelder für KI-Systeme", "de") == ["bussgeld", "ki", "system"]
    assert tokenize("What are the fines?", "en") == ["fine"]


def test_detect_lang():
    assert detect_lang("Wie lange müssen Protokolle aufbewahrt werden?") == "de"
    assert detect_lang("What is the maximum fine under the GDPR?") == "en"


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_finds_article_in_question_language(retriever, mode):
    hits = retriever.search("Verletzung personenbezogener Daten 72 Stunden melden", mode=mode, k=2, lang="de")
    assert hits[0].doc_id == "gdpr|de|art33"
    assert all("|de|" in h.doc_id for h in hits)  # language filter
    assert hits[0].chunk_ids == ["gdpr|de|art33#c0"]


def test_search_detects_language_and_finds_annex(retriever):
    hits = retriever.search("Is AI for recruitment and filtering job applications high-risk?", k=3)
    assert "ai_act|en|annex_iii" in [h.doc_id for h in hits]


def test_index_is_tied_to_embedding_model(retriever, records, tmp_path):
    from src.embeddings import HashEmbedder
    from src.retrieval import Retriever

    other = HashEmbedder()
    other.name = "another-model"
    r2 = Retriever(records, embedder=other, chroma_dir=str(tmp_path / "chroma"))
    with pytest.raises(RuntimeError, match="--rebuild"):
        r2.search("KI-Kompetenz", mode="vector", lang="de")
    assert r2.build_index(rebuild=True, progress=lambda _m: None) == len(r2.chunks)
