"""Streamlit chat UI.   streamlit run app.py   (the API must run: uvicorn src.api:app)"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
# "Ab wann gelten die Verbote …?" is deliberately not an example: it needs Art. 113, which the search does not
# find (known gap, see README) – the demo should show what the system does well, the README shows what it does not
EXAMPLES = [
    "Müssen Nutzer informiert werden, wenn sie mit einem KI-Chatbot sprechen?",
    "Gilt ein KI-System, das Bewerbungen filtert, als Hochrisiko-KI-System?",
    "Innerhalb welcher Frist muss eine Datenschutzverletzung gemeldet werden?",
    "What is the maximum fine for prohibited AI practices?",
]
GUARDRAIL_TEXT = {
    "no_valid_citation": "Die Antwort des Modells enthielt keine überprüfbare Quelle und wurde deshalb verworfen.",
    "unknown_citation": "Die Antwort zitiert Artikel, die nicht gefunden wurden – bitte im Gesetzestext prüfen.",
    "no_context": "Keine passenden Artikel gefunden.",
    "unsupported_fact": "Die Antwort des Modells enthielt ein Datum oder einen Betrag, der nicht in den gefundenen "
    "Artikeln steht, und wurde deshalb verworfen.",
}

st.set_page_config(page_title="EU AI Act & DSGVO Assistant", page_icon="⚖️", layout="centered")
st.title("EU AI Act & DSGVO Assistant")
st.caption("Antworten nur auf Basis der Gesetzestexte, mit Artikelangabe und Link zu EUR-Lex. Keine Rechtsberatung.")

with st.sidebar:
    mode = st.radio("Suche", ["vector", "hybrid", "keyword"], help="vector = beste Variante in der Evaluation")
    lang = st.selectbox("Sprache", ["automatisch", "de", "en"])
    k = st.slider("Artikel im Kontext", 3, 8, 5)
    st.divider()
    st.markdown("**Beispiele**")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.pending = ex
    if st.button("Verlauf löschen"):
        st.session_state.history = []

st.session_state.setdefault("history", [])


def show_answer(r: dict) -> None:
    st.markdown(r["text"])
    if r.get("guardrail"):
        st.warning(GUARDRAIL_TEXT.get(r["guardrail"], r["guardrail"]))
    if r.get("abstained"):  # nothing was used – show the searched articles in one line, not as sources
        st.caption("Durchsuchte Artikel: " + ", ".join(s["label"] for s in r["sources"]))
        return
    for s in r["sources"]:
        marker = "✅ " if s["key"] in r.get("cited", []) else ""
        with st.expander(f"{marker}{s['label']} – {s['title']}"):
            st.write(s["excerpt"] + " …")
            st.markdown(f"[Volltext auf EUR-Lex]({s['url']})")
    st.caption(f"{r['latency_s']} s · {r['prompt_tokens'] + r['completion_tokens']} Tokens · "
               f"{r['mode']} · ✅ = in der Antwort zitiert")


for turn in st.session_state.history:
    st.chat_message("user").write(turn["question"])
    with st.chat_message("assistant"):
        show_answer(turn["response"])

question = st.chat_input("Frage stellen, z. B. „Was regelt Artikel 4 der KI-Verordnung?“")
question = question or st.session_state.pop("pending", None)
if question:
    st.chat_message("user").write(question)
    payload = {"question": question, "mode": mode, "k": k, "lang": None if lang == "automatisch" else lang}
    with st.chat_message("assistant"):
        try:
            with st.spinner("Suche in den Artikeln …"):
                resp = requests.post(f"{API}/ask", json=payload, timeout=180)
            resp.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"API nicht erreichbar oder Fehler: {exc}")
            st.stop()
        result = resp.json()
        show_answer(result)
    st.session_state.history.append({"question": question, "response": result})
