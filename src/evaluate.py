"""Evaluation on 30 test questions (data/eval_set.jsonl): 15 German, 15 English, 4 of them the law does NOT answer.

  python -m src.evaluate retrieval   # no LLM, free, ~1 min: hit rate, recall and MRR for vector / keyword / hybrid
  python -m src.evaluate answers     # with LLM: answers + LLM-as-judge for vector and hybrid (cached, resumable)
  python -m src.evaluate report      # writes docs/eval_results.md and docs/eval_per_question.csv

Retrieval metrics (answerable questions only, k = 5 articles):
  hit rate@5   share of questions where at least one expected article is among the 5 results
  recall@5     share of all expected articles that were found
  MRR@5        mean of 1/rank of the first expected article (1.0 = always on rank 1)
Answer metrics:
  correct          judge: answer contains the key facts of the reference answer (answerable questions)
  faithfulness     judge: share of statements supported by the retrieved context (answered questions)
  relevance        judge: answer addresses the question (answered questions)
  citation prec.   share of citations that point to a retrieved article (deterministic, no judge)
  abstention       out-of-scope questions answered with the fixed "no answer" sentence
  false refusals   answerable questions where the system refused
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from src.laws import key_of

EVAL_SET = Path("data/eval_set.jsonl")
RUN_DIR = Path("data/eval_runs")
DOCS = Path("docs")
K = 5

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system for EU law.
You receive the CONTEXT (the only allowed source), the QUESTION, the system's ANSWER and a REFERENCE ANSWER.
Return only a JSON object: {"faithfulness": <0..1>, "relevance": <0..1>, "correct": <0 or 1>, "reason": "<short>"}
- faithfulness: share of the statements in ANSWER that are supported by CONTEXT (1 = all supported).
- relevance: how well ANSWER addresses the QUESTION (1 = fully, 0 = not at all).
- correct: 1 if ANSWER contains the key facts of the REFERENCE ANSWER (deadlines, amounts, conditions) and does
  not contradict it, otherwise 0. Ignore citation format, language and the legal-advice note."""


# ---------- retrieval metrics ----------
def hit_rate_at_k(retrieved: Sequence[str], expected: Sequence[str]) -> float | None:
    """1.0 if any expected article is retrieved; None for questions without an answer in the law."""
    if not expected:
        return None
    return 1.0 if set(retrieved) & set(expected) else 0.0


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    return len(set(retrieved) & set(expected)) / len(set(expected))


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    for rank, key in enumerate(retrieved, start=1):
        if key in expected:
            return 1.0 / rank
    return 0.0


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def evaluate_retrieval(questions: list[dict], search: Callable[[dict], list[str]]) -> tuple[dict, list[dict]]:
    """`search(question_record)` returns ranked language-independent keys like 'gdpr|art33'."""
    rows = []
    for q in questions:
        retrieved = search(q)
        rows.append({"id": q["id"], "retrieved": retrieved, "hit": hit_rate_at_k(retrieved, q["expected"]),
                     "recall": recall_at_k(retrieved, q["expected"]), "rr": reciprocal_rank(retrieved, q["expected"])})
    answerable = [r for r in rows if r["hit"] is not None]
    summary = {"questions": len(answerable), "hit_rate": _mean(r["hit"] for r in answerable),
               "recall": _mean(r["recall"] for r in answerable), "mrr": _mean(r["rr"] for r in answerable)}
    return summary, rows


# ---------- judge ----------
def parse_judge(text: str) -> dict:
    """Robust against ```json fences and extra text; values are clamped to [0, 1]."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    out = {}
    for key in ("faithfulness", "relevance", "correct"):
        try:
            out[key] = min(1.0, max(0.0, float(data[key])))
        except (KeyError, TypeError, ValueError):
            out[key] = None
    out["reason"] = str(data.get("reason", ""))[:300]
    return out


def answer_metrics(rows: list[dict]) -> dict:
    """rows = one dict per question with 'expected', 'abstained', 'cited', 'invalid_citations', judge scores …"""
    answerable = [r for r in rows if r["expected"]]
    out_of_scope = [r for r in rows if not r["expected"]]
    answered = [r for r in answerable if not r["abstained"]]
    n_valid = sum(len(r["cited"]) for r in rows)
    n_invalid = sum(len(r["invalid_citations"]) for r in rows)
    return {
        "correct": _mean((r.get("correct") or 0.0) if not r["abstained"] else 0.0 for r in answerable),
        "faithfulness": _mean(r.get("faithfulness") for r in answered),
        "relevance": _mean(r.get("relevance") for r in answered),
        "citation_precision": round(n_valid / (n_valid + n_invalid), 3) if n_valid + n_invalid else None,
        "abstention": f"{sum(r['abstained'] for r in out_of_scope)}/{len(out_of_scope)}",
        "false_refusals": f"{len(answerable) - len(answered)}/{len(answerable)}",
        "latency_s": _mean(r["latency_s"] for r in rows),
        "tokens_per_question": round(statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows)),
        "cost_per_100_usd": round(100 * statistics.mean(r["cost_usd"] for r in rows), 4),
    }


def judge_input(rec: dict, q: dict) -> str:
    return (f"CONTEXT:\n{rec['context']}\n\nQUESTION:\n{q['question']}\n\nANSWER:\n{rec['text']}\n\n"
            f"REFERENCE ANSWER:\n{q['reference_answer']}")


# ---------- I/O ----------
def load_questions(path: Path = EVAL_SET) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def _fmt(v) -> str:
    return "–" if v is None else (f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v))


# ---------- commands ----------
def run_retrieval(retriever, questions: list[dict]) -> dict:
    results = {}
    for mode in ("vector", "keyword", "hybrid"):
        start = time.perf_counter()
        def search(q: dict, mode: str = mode) -> list[str]:
            return [key_of(h.doc_id) for h in retriever.search(q["question"], mode, K, q["lang"])]

        summary, rows = evaluate_retrieval(questions, search)
        summary["ms_per_query"] = round(1000 * (time.perf_counter() - start) / len(questions))
        results[mode] = {"summary": summary, "rows": rows}
        print(f"{mode:<8} hit@5={_fmt(summary['hit_rate'])}  recall@5={_fmt(summary['recall'])}  "
              f"MRR={summary['mrr']}  {summary['ms_per_query']} ms/query")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "retrieval.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return results


def run_answers(retriever, llm, questions: list[dict], modes: Sequence[str], fresh: bool = False) -> None:
    from src.rag import answer

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        path = RUN_DIR / f"answers_{mode}.jsonl"
        done = {} if fresh else _read_jsonl(path)
        with path.open("w", encoding="utf-8") as f:
            for i, q in enumerate(questions, 1):
                rec = done.get(q["id"])
                if rec is None:
                    a = answer(q["question"], retriever, llm, mode=mode, k=K, lang=q["lang"])
                    rec = {"id": q["id"], "expected": q["expected"], **a.to_dict(with_context=True)}
                if q["expected"] and not rec["abstained"] and rec.get("faithfulness") is None:
                    judge = llm.chat(JUDGE_PROMPT, judge_input(rec, q))
                    rec.update(parse_judge(judge.text))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                flag = "refused" if rec["abstained"] else f"correct={rec.get('correct')}"
                print(f"[{mode} {i:>2}/{len(questions)}] {q['id']} {flag}  {rec['latency_s']} s")


def write_report(questions: list[dict]) -> str:
    retrieval = json.loads((RUN_DIR / "retrieval.json").read_text(encoding="utf-8"))
    lines = ["| Mode | Hit rate @5 | Recall @5 | MRR | Correct | Faithfulness | Relevance | Citation precision "
             "| Abstention (out of scope) | False refusals | Avg latency | Cost / 100 questions |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_question = {q["id"]: {"id": q["id"], "lang": q["lang"], "question": q["question"],
                              "expected": " ".join(q["expected"])} for q in questions}
    for mode in ("vector", "keyword", "hybrid"):
        r = retrieval[mode]["summary"]
        for row in retrieval[mode]["rows"]:
            per_question[row["id"]][f"{mode}_top5"] = " ".join(row["retrieved"])
            per_question[row["id"]][f"{mode}_rr"] = row["rr"]
        answers = _read_jsonl(RUN_DIR / f"answers_{mode}.jsonl")
        a = answer_metrics(list(answers.values())) if len(answers) == len(questions) else {}
        for qid, rec in answers.items():
            per_question[qid][f"{mode}_answer"] = rec["text"].rsplit("\n\n", 1)[0]  # without the disclaimer
            per_question[qid][f"{mode}_correct"] = rec.get("correct")
            per_question[qid][f"{mode}_guardrail"] = rec.get("guardrail")
        latency = f"{a['latency_s']} s" if a else "–"
        cost = f"${a['cost_per_100_usd']:.2f}" if a else "–"
        lines.append(
            f"| {mode} | {_fmt(r['hit_rate'])} | {_fmt(r['recall'])} | {r['mrr']} | {_fmt(a.get('correct'))} "
            f"| {_fmt(a.get('faithfulness'))} | {_fmt(a.get('relevance'))} | {_fmt(a.get('citation_precision'))} "
            f"| {a.get('abstention', '–')} | {a.get('false_refusals', '–')} | {latency} | {cost} |"
        )
    table = "\n".join(lines)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "eval_results.md").write_text(
        "# Evaluation results\n\n30 questions (15 DE / 15 EN, 4 out of scope), k = 5 articles. "
        "Retrieval needs no LLM; answer columns only for modes evaluated with `python -m src.evaluate answers`.\n\n"
        + table + "\n\nPer question: [eval_per_question.csv](eval_per_question.csv)\n", encoding="utf-8")
    first = ["id", "lang", "question", "expected"]
    fields = first + sorted({k for row in per_question.values() for k in row} - set(first))
    with (DOCS / "eval_per_question.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_question.values())
    return table


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Evaluate retrieval and answers on data/eval_set.jsonl")
    p.add_argument("command", choices=["retrieval", "answers", "report"])
    p.add_argument("--modes", nargs="+", default=["vector", "hybrid"], help="answers: modes to evaluate")
    p.add_argument("--fresh", action="store_true", help="answers: ignore cached results")
    args = p.parse_args()
    questions = load_questions()

    if args.command == "report":
        print(write_report(questions))
        return
    from src.retrieval import Retriever, load_articles

    retriever = Retriever(load_articles())
    if args.command == "retrieval":
        run_retrieval(retriever, questions)
    else:
        from src.llm import LLM

        run_answers(retriever, LLM(), questions, args.modes, fresh=args.fresh)
    print("\nNext: python -m src.evaluate report")


if __name__ == "__main__":
    main()
