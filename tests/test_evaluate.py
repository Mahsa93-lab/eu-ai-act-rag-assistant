import pytest

from src.evaluate import answer_metrics, evaluate_retrieval, hit_rate_at_k, parse_judge, recall_at_k, reciprocal_rank


def test_retrieval_metrics():
    assert hit_rate_at_k(["a", "b"], ["b"]) == 1.0
    assert hit_rate_at_k(["a", "b"], ["c"]) == 0.0
    assert hit_rate_at_k(["a"], []) is None  # "no answer" questions are excluded
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert recall_at_k(["a", "b"], ["b", "x"]) == 0.5


def test_evaluate_retrieval_aggregates():
    questions = [{"id": "1", "question": "q1", "expected": ["x"]}, {"id": "2", "question": "q2", "expected": ["y"]},
                 {"id": "3", "question": "q3", "expected": []}]
    fake_index = {"q1": ["x", "z"], "q2": ["z", "y"], "q3": ["x"]}
    summary, rows = evaluate_retrieval(questions, lambda q: fake_index[q["question"]])
    assert summary == {"questions": 2, "hit_rate": 1.0, "recall": 1.0, "mrr": 0.75}
    assert len(rows) == 3


def test_parse_judge_is_robust():
    assert parse_judge('```json\n{"faithfulness": 0.8, "relevance": 1, "correct": 1, "reason": "ok"}\n```') == {
        "faithfulness": 0.8, "relevance": 1.0, "correct": 1.0, "reason": "ok"}
    assert parse_judge("no json at all")["correct"] is None
    assert parse_judge('{"faithfulness": 7, "relevance": "x", "correct": 0}')["faithfulness"] == 1.0


def row(expected, abstained, correct=None, cited=(), invalid=()):
    return {"expected": expected, "abstained": abstained, "correct": correct, "faithfulness": correct,
            "relevance": correct, "cited": list(cited), "invalid_citations": list(invalid), "latency_s": 2.0,
            "prompt_tokens": 900, "completion_tokens": 100, "cost_usd": 0.0004}


def test_answer_metrics():
    rows = [row(["a"], False, 1.0, cited=["a"]), row(["b"], False, 0.0, cited=["b"], invalid=["z"]),
            row(["c"], True), row([], True), row([], False, cited=["q"])]
    m = answer_metrics(rows)
    assert m["correct"] == pytest.approx(1 / 3, abs=1e-3)  # refusal on an answerable question counts as wrong
    assert m["faithfulness"] == 0.5  # only answered questions
    assert m["citation_precision"] == 0.75
    assert (m["abstention"], m["false_refusals"]) == ("1/2", "1/3")
    assert m["tokens_per_question"] == 1000 and m["cost_per_100_usd"] == 0.04
