# Evaluation results

30 questions (15 DE / 15 EN, 4 out of scope), k = 5 articles. Retrieval needs no LLM; answer columns only for modes evaluated with `python -m src.evaluate answers`.

| Mode | Hit rate @5 | Recall @5 | MRR | Correct | Faithfulness | Relevance | Citation precision | Abstention (out of scope) | False refusals | Avg latency | Cost / 100 questions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector | 100% | 98% | 0.904 | 85% | 98% | 100% | 100% | 4/4 | 1/26 | 2.076 s | $0.03 |
| keyword | 92% | 90% | 0.792 | – | – | – | – | – | – | – | – |
| hybrid | 96% | 94% | 0.856 | 81% | 100% | 98% | 100% | 4/4 | 3/26 | 1.862 s | $0.03 |

Per question: [eval_per_question.csv](eval_per_question.csv)
