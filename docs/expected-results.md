# Expected results per step

Values marked ✔ are fixed (structure of the law, code). Values marked ≈ depend on the PC, the embedding model
or the LLM and are filled in from the first real run.

## 1 · Tests (no network, no model)
```
ruff check .   → All checks passed!
pytest -q      → 36 passed
```

## 2 · Download – `python -m src.download_eurlex`
Four files in `data/raw/`: `ai_act_de.html`, `ai_act_en.html`, `gdpr_de.html`, `gdpr_en.html`.
If EUR-Lex answers with a bot check, the script prints the links: open in the browser → Ctrl+S →
"Webseite, nur HTML" → save under the printed name, then run the script again (it skips existing files).

## 3 · Parsing – `python -m src.parse_articles`
| file | articles ✔ | annexes ✔ | characters |
|---|---|---|---|
| ai_act_de.html | 113 | 13 | ≈ 360,000–400,000 |
| ai_act_en.html | 113 | 13 | ≈ 330,000 ✔ (official Formex text: 329,751) |
| gdpr_de.html | 99 | 0 | ≈ 200,000–220,000 |
| gdpr_en.html | 99 | 0 | ≈ 185,000 ✔ (official Formex text: 185,126) |

`450 records → data/articles.jsonl`, every line `OK`. Spot checks in `data/articles.jsonl`:
- `ai_act|en|art51` contains `greater than 10^25`
- `ai_act|de|art113` ends with the date 2 August 2027 (Art. 6 Abs. 1) and does **not** contain "Geschehen zu Brüssel"
- `ai_act|de|annex_iii` title: "Hochrisiko-KI-Systeme gemäß Artikel 6 Absatz 2"
- `gdpr|de|art33` title: "Meldung von Verletzungen des Schutzes personenbezogener Daten an die Aufsichtsbehörde"

## 4 · Index – `python -m src.build_index`
- English text gives ≈ 517 chunks (measured on the official text), German ≈ 550–620 → ≈ 1,100 chunks in total
- First run downloads `intfloat/multilingual-e5-large` (≈ 2.2 GB) into `models/`
- Duration ≈ 5–20 minutes on a laptop CPU; a second run adds nothing (`0/0 chunks`)

## 5 · Retrieval evaluation – `python -m src.evaluate retrieval`
- Measured in the sandbox on the official English text, keyword mode (BM25 with stemming), 13 English answerable
  questions: hit rate@5 = 100 %, MRR = 0.91 ✔
- All three modes on 26 answerable questions (DE + EN): ≈ from the first run

## 6 · LLM check – `python -m src.llm --list-models` and `--ping`
`answer='OK'  tokens=…  … s`

## 7 · Answer evaluation – `python -m src.evaluate answers` → `report`
- 30 questions × 2 modes = 60 answers + 52 judge calls ≈ 112 LLM calls; with a paid OpenAI key a few minutes and a few cents, on a free tier (`LLM_MIN_INTERVAL_S=4`) ≈ 10–15 minutes
- Interrupted (e.g. rate limit for the day)? Run the same command again – finished questions are skipped
- Out-of-scope questions (q14, q15, q29, q30) should all be refused: abstention `4/4`

## 8 · API and UI
- `GET /health` → `{"status": "ok", "articles": 450, "chunks": ≈1100, "embedding_model": "intfloat/multilingual-e5-large"}`
- `POST /search` with "Innerhalb welcher Frist muss eine Datenschutzverletzung gemeldet werden?" → first source `DSGVO Art. 33`
- Streamlit: answer with `[DSGVO Art. 33]`, the source expander marked ✅, link to EUR-Lex jumps to the article
