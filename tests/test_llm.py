from types import SimpleNamespace

from src.llm import LLM, cost_usd


class FakeCompletions:
    def __init__(self, reject_temperature):
        self.reject_temperature, self.calls = reject_temperature, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_temperature and "temperature" in kwargs:
            raise ValueError("Error code: 400 – Unsupported value: 'temperature' does not support 0")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" OK "))], usage=usage)


def fake_llm(reject_temperature=False):
    completions = FakeCompletions(reject_temperature)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLM(client=client, model="test-model", min_interval_s=0), completions


def test_chat_returns_text_and_tokens():
    llm, completions = fake_llm()
    r = llm.chat("system", "user")
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ("OK", 10, 2)
    assert completions.calls[0]["temperature"] == 0.0


def test_reasoning_model_without_temperature_is_retried_once():
    llm, completions = fake_llm(reject_temperature=True)
    assert llm.chat("s", "u").text == "OK"
    assert llm.chat("s", "u").text == "OK"
    assert ["temperature" in c for c in completions.calls] == [True, False, False]  # remembered


def test_cost(monkeypatch):
    monkeypatch.setenv("PRICE_INPUT_PER_M", "0.20")
    monkeypatch.setenv("PRICE_OUTPUT_PER_M", "1.20")
    assert cost_usd(1_000_000, 100_000) == 0.20 + 0.12
