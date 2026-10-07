"""Chat model access through the OpenAI-compatible API – works with OpenAI, Google Gemini (free tier),
Mistral or a local Ollama; only the settings in .env change (LLM_BASE_URL empty = OpenAI).

  python -m src.llm --list-models   # which model ids does my key allow?
  python -m src.llm --ping          # one short test call
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv


def _client():
    from openai import OpenAI

    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL") or None,  # None → https://api.openai.com/v1
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing-key",
        max_retries=6,  # 429 "rate limit" and 5xx are retried with exponential backoff
        timeout=120,
    )


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Cost at the PAID price of the model (USD per 1 million tokens, from .env) – also shown on the free tier,
    so the README can state what 100 questions would cost in production."""
    price_in = float(os.getenv("PRICE_INPUT_PER_M", "0") or 0)
    price_out = float(os.getenv("PRICE_OUTPUT_PER_M", "0") or 0)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


class LLM:
    def __init__(self, client=None, model: str | None = None, min_interval_s: float | None = None):
        load_dotenv()
        self.model = model or os.getenv("CHAT_MODEL", "").strip()
        if not self.model:
            raise RuntimeError("CHAT_MODEL is empty – set it in .env (python -m src.llm --list-models shows the ids)")
        self.client = client or _client()
        # free tiers allow only a few requests per minute – keep a minimum gap between calls
        self.min_interval_s = float(os.getenv("LLM_MIN_INTERVAL_S", "0") if min_interval_s is None else min_interval_s)
        self.reasoning_effort = os.getenv("REASONING_EFFORT") or None
        self._last_call = 0.0
        self._send_temperature = True  # reasoning models only accept their default temperature

    def chat(self, system: str, user: str, temperature: float = 0.0) -> ChatResult:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                **kwargs, **({"temperature": temperature} if self._send_temperature else {})
            )
        except Exception as exc:  # e.g. 400 "Unsupported value: 'temperature'" → retry once without it
            if not self._send_temperature or "temperature" not in str(exc).lower():
                raise
            self._send_temperature = False
            resp = self.client.chat.completions.create(**kwargs)
        self._last_call = time.monotonic()
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=round(time.perf_counter() - start, 2),
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Check the LLM connection")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--ping", action="store_true")
    args = p.parse_args()
    load_dotenv()
    if args.list_models:
        for m in sorted(m.id for m in _client().models.list()):
            print(m)
    if args.ping:
        r = LLM().chat("Antworte nur mit dem Wort OK.", "Test")
        print(f"answer={r.text!r}  tokens={r.prompt_tokens}+{r.completion_tokens}  {r.latency_s} s")


if __name__ == "__main__":
    main()
