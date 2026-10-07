# EU AI Act & GDPR Assistant (RAG · DE/EN)

**English** · [Deutsch](README.de.md)

> Ask questions about the EU AI Act and the GDPR in German or English. Every statement cites the article, the code rejects answers without a verifiable source, and quality is measured on 30 test questions.

![Architecture](docs/architecture.png)

## Business problem
Since February 2025 companies using AI must meet the first duties of the EU AI Act (e.g. AI literacy, Art. 4) – on top of the GDPR. Business departments have many questions and need answers they can **check**. A chatbot is only useful here if it is reliable: correct sources, no invented content, a clear "I don't know" and measured quality instead of a demo impression.

## Key results
1. **The simpler search won – measured, not assumed.** Dense retrieval with a multilingual model found the right article for **26 of 26** questions (hit rate@5 100 %, MRR 0.90). Hybrid search was *worse* (96 %, 0.86): BM25 cannot match German compounds ("Datenschutzverletzung" vs. "Verletzung des Schutzes personenbezogener Daten"), and Reciprocal Rank Fusion then ranked articles that were mediocre in both lists above the correct one. Vector search is therefore the default.
2. **No invented sources.** Citation precision 100 % in all runs, all 4 out-of-scope questions refused (incl. a prompt-injection attempt), 85 % of the answers fully correct in the final run.
3. **The evaluation found a hallucination that neither the citation check nor the LLM judge caught** – a correct date from the model's memory, cited to an article that does not contain it. A new deterministic check (every date and amount in the answer must appear in the retrieved text) now rejects such answers.
4. **Cheap and fast:** ≈ 2 s per answer and **$0.03 per 100 questions** (`gpt-6-luna`); embeddings run locally, so the legal texts never leave the machine for the search step.

## Data
- Regulation (EU) 2024/1689 (AI Act: 113 articles, 13 annexes) and Regulation (EU) 2016/679 (GDPR: 99 articles), German and English, official HTML from [EUR-Lex](https://eur-lex.europa.eu) – reuse permitted with source acknowledgement (Decision 2011/833/EU)
- Parsed **per article / annex** → 450 records (`data/articles.jsonl`); the parser checks the counts per file

## Approach
1. **Parsing** (`src/parse_articles.py`): one stream of paragraphs and table rows, a small state machine opens a record at every article/annex heading – works for the newer (AI Act) and older (GDPR) EUR-Lex layout; footnotes, recitals and signatures are excluded, `10^25` stays readable
2. **Chunking** (`src/chunking.py`): long articles (Art. 3 AI Act ≈ 17,000 characters) are split at paragraph boundaries; every chunk carries law, article and title. Search runs on chunks, the **article** stays the unit that is cited
3. **Retrieval** (`src/retrieval.py`): local embeddings (`multilingual-e5-large`, fastembed/ONNX) in ChromaDB, BM25 with German/English stemming, and **hybrid search** with Reciprocal Rank Fusion; results are filtered to the language of the question
4. **Generation** (`src/rag.py`, `prompts/system_prompt.md`): any OpenAI-compatible model (default: OpenAI API; also tested setup for the Gemini free tier and a local Ollama). Context only, citation after every statement, fixed "no answer" sentence
5. **Guardrails in code**, not only in the prompt: personal data masked before search and LLM · answer without a valid citation → "no answer" · every date and amount must appear in the retrieved text, otherwise "no answer" · citations of articles that were not retrieved are flagged · answer language set explicitly · "not legal advice" note always added
6. **Evaluation** (`src/evaluate.py`): 30 questions (15 DE / 15 EN, 4 out of scope, 1 prompt injection), retrieval metrics without LLM and answer quality with an LLM judge
7. **Delivery**: FastAPI (`/ask`, `/search`, `/health`), Streamlit chat with sources and EUR-Lex links, Docker Compose, GitHub Actions

## Results
Final run (`gpt-6-luna`, reasoning effort low, k = 5 articles, 26 answerable + 4 out-of-scope questions):

| Mode | Hit rate @5 | MRR | Correct | Faithfulness | Citation precision | Out of scope refused | False refusals | Latency | Cost / 100 questions |
|---|---|---|---|---|---|---|---|---|---|
| **vector** (default) | **100 %** | **0.904** | **85 %** | 98 % | 100 % | 4/4 | 1/26 | 2.1 s | $0.03 |
| keyword (BM25) | 92 % | 0.792 | – | – | – | – | – | – | – |
| hybrid (RRF) | 96 % | 0.856 | 81 % | 100 % | 100 % | 4/4 | 3/26 | 1.9 s | $0.03 |

Full table and every answer: [docs/eval_results.md](docs/eval_results.md) · [docs/eval_per_question.csv](docs/eval_per_question.csv)

**Three evaluation runs – what changed and why**

| Run | Change | Correct (vector / hybrid) | Finding |
|---|---|---|---|
| 1 | first version | 96 % / 77 % | one English question answered in German → answer language now set explicitly |
| 2 | answer language | 100 % / 85 % | "100 %" included a correct date that was **not** in the retrieved text (q01) |
| 3 | facts check | **85 % / 81 %** | q01 now refused; remaining errors are incomplete answers (q10, q24) and one inverted condition (q13) |

Vector search was better than hybrid in every run. The spread between runs (up to 15 points with identical retrieval) shows the variance of the LLM on 26 questions – the numbers are tendencies, not exact rates. Known retrieval gap: "Ab wann gelten die Verbote …?" needs Art. 113 (dates of application), which no search mode ranks in the top 5. In the evaluation runs the system refused; in a live test it answered "2 August 2025" – a date that **is** in the retrieved Art. 111, but belongs to a different rule (general-purpose AI models). The facts check cannot catch a correct-looking date taken from the wrong provision; only better retrieval fixes this (see next steps).

## Screenshots
| Answer with sources and EUR-Lex links | Refusal outside the law |
|---|---|
| ![Chat](images/01_chat.png) | ![Refusal](images/02_refusal.png) |

REST API (FastAPI / OpenAPI): [images/03_api_docs.png](images/03_api_docs.png)

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env                      # add LLM_API_KEY, CHAT_MODEL and prices
python -m src.download_eurlex             # 4 HTML files → data/raw/
python -m src.parse_articles              # → data/articles.jsonl (450 records, counts checked)
python -m src.build_index                 # local embeddings → chroma_db/ (first run downloads the model)
python -m src.evaluate retrieval          # free, no LLM
python -m src.evaluate answers            # with LLM, cached and resumable
python -m src.evaluate report             # → docs/eval_results.md
uvicorn src.api:app                       # http://127.0.0.1:8000/docs
streamlit run app.py                      # http://localhost:8501
docker compose up --build                 # API + UI in containers (index from the host)
```
Expected output of every step: [docs/expected-results.md](docs/expected-results.md)

## Tests
`pytest` – 39 tests without network or model: both EUR-Lex layouts (footnotes, nested lists, quoted articles, annexes after the signature), chunking, RRF and stemming, search in all three modes on a temporary ChromaDB, guardrails (citations, dates and amounts) and PII masking, LLM client (incl. reasoning models without `temperature`), metrics, API. CI runs `ruff` and `pytest` on every push.

## Limitations
- Not legal advice. Recitals and later amendments/corrigenda are not indexed in version 1
- The LLM judge is the same model family as the answering model; 30 questions show tendencies, not statistical certainty
- The facts check covers dates and amounts, not other statements; one incomplete or inverted condition per run remains (see q10, q13)
- PII masking covers e-mail, phone numbers and IBANs, not names
- Data protection: personal data is masked before any LLM call; the OpenAI API does not use API data for training by default (inputs kept up to 30 days for abuse monitoring). The Gemini free tier may use inputs to improve Google's products – fine for public law, not for company data; fully local: Ollama

## What I would do next
- Index the provisions on dates of application with extra weight, or add a "When does … apply?" router (fixes q01)
- German compound splitting for BM25 and weighted RRF – tested on a **separate** question set, not tuned on these 30
- Larger test set written together with legal experts; a second, different judge model

## Project structure
```
src/        download_eurlex · parse_articles · chunking · embeddings · retrieval · build_index · llm · rag · evaluate · api
prompts/    system_prompt.md
data/       eval_set.jsonl (30 questions) · articles.jsonl (generated) · raw/ (downloaded, not committed)
tests/      pytest suite
docs/       architecture, expected results, evaluation results
images/     screenshots
app.py      Streamlit UI
```

---
*Author: Mahsa Ahmadi · Uses only public legal texts. Not legal advice.*
