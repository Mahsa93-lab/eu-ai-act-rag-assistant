from fastapi.testclient import TestClient

from src import api
from tests.test_rag import FakeLLM


def test_search_and_ask(retriever, monkeypatch):
    monkeypatch.setattr(api, "get_retriever", lambda: retriever)
    monkeypatch.setattr(api, "get_llm", lambda: FakeLLM("Providers must ensure AI literacy [AI Act Art. 4]."))
    client = TestClient(api.app)

    assert client.get("/health").json()["chunks"] == len(retriever.chunks)

    r = client.post("/search", json={"question": "What does the AI Act say about AI literacy of staff?", "k": 2})
    assert r.status_code == 200 and r.json()["lang"] == "en"
    assert r.json()["sources"][0]["label"] == "AI Act Art. 4"

    r = client.post("/ask", json={"question": "What does the AI Act say about AI literacy of staff?"}).json()
    assert r["cited"] == ["ai_act|art4"] and r["text"].endswith("Note: This is not legal advice.")
    assert "context" not in r

    assert client.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422
