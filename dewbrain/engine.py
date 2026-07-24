"""
Engine — pluggable text-generation backend for the vmPFC.

The core (retrieval / law / critique loop) is engine-agnostic: it builds a prompt and asks
for text. WHERE that text comes from is a swappable part, exactly like stitchu keeps the
drafting engine separate from its frontends. Three backends share one interface:

  ApiEngine   -> Anthropic API. Metered (per-token), autonomous (runs headless, no session),
                 fully robust (retry/backoff/stream). Use for unattended batch runs.
  MaxEngine   -> Claude Code session (Max subscription). Zero extra cost, but NOT autonomous:
                 it writes each prompt to a hand-off file for the running session to answer.
                 Use while a session is open and cost must stay zero.
  LocalEngine -> (future) own fine-tuned model on GPU. Slots into the same interface once the
                 gold corpus reaches ~100+. Then the rented engine is gone = full ownership.

Selection is one config field (DEWBRAIN_ENGINE=api|max). Nothing else in the system changes.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Callable, Protocol

import anthropic


class EngineError(RuntimeError):
    """A generation backend could not produce text."""


class Engine(Protocol):
    """One method: turn a prompt into text. Every backend implements exactly this."""
    def complete(self, prompt: str, log: Callable[[str], None]) -> str: ...


# --------------------------------------------------------------------------------------
# ApiEngine — Anthropic API, fully robust (retry / backoff / stream / adaptive thinking)
# --------------------------------------------------------------------------------------
class ApiEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EngineError(
                "engine=api but ANTHROPIC_API_KEY not set.\n"
                "  export ANTHROPIC_API_KEY=...   (or use engine=max for the session backend)"
            )
        # SDK retries off; OUR loop owns backoff (single, observable retry policy).
        self.client = anthropic.Anthropic(api_key=key, timeout=cfg.api.timeout, max_retries=0)

    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        c = self.cfg.api
        if retry_after is not None:
            delay = min(retry_after, c.max_delay)
        else:
            delay = min(c.base_delay * (2 ** attempt), c.max_delay)
            delay = random.uniform(0, delay)  # full jitter: no thundering herd
        time.sleep(delay)
        return delay

    def complete(self, prompt: str, log: Callable[[str], None]) -> str:
        c = self.cfg.api
        last_exc: Exception | None = None
        # max_retries RETRIES => max_retries + 1 total attempts.
        for attempt in range(c.max_retries + 1):
            try:
                with self.client.messages.stream(
                    model=c.model,
                    max_tokens=c.max_tokens,
                    thinking={"type": "adaptive"},
                    output_config={"effort": c.effort},
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for _ in stream.text_stream:
                        pass
                    msg = stream.get_final_message()

                if msg.stop_reason == "refusal":
                    detail = getattr(msg, "stop_details", None)
                    cat = getattr(detail, "category", None) if detail else None
                    raise EngineError(f"Claude refused (category={cat}). Not retried.")

                text = "".join(
                    b.text for b in msg.content if getattr(b, "type", None) == "text"
                )
                if not text.strip():
                    raise EngineError(f"Empty response (stop_reason={msg.stop_reason}).")
                return text.strip()

            except anthropic.RateLimitError as e:
                last_exc = e
                ra = None
                try:
                    ra = float(e.response.headers.get("retry-after", "")) if e.response else None
                except (ValueError, AttributeError):
                    ra = None
                waited = self._backoff(attempt, retry_after=ra)
                log(f"  [api] rate limited, retry {attempt + 1}/{c.max_retries} in {waited:.1f}s")
            except (anthropic.APIStatusError, anthropic.APIConnectionError,
                    anthropic.APITimeoutError) as e:
                status = getattr(e, "status_code", None)
                if isinstance(e, anthropic.APIStatusError) and status is not None and status < 500:
                    raise EngineError(f"Non-retryable API error {status}: {e}") from e
                last_exc = e
                waited = self._backoff(attempt)
                log(f"  [api] {type(e).__name__}, retry {attempt + 1}/{c.max_retries} in {waited:.1f}s")

        raise EngineError(f"API call failed after {c.max_retries} retries: {last_exc}") from last_exc


# --------------------------------------------------------------------------------------
# MaxEngine — Claude Code session backend (zero extra cost, session-bound)
# --------------------------------------------------------------------------------------
class MaxEngine:
    """
    Uses the running Claude Code session (Max subscription) as the generator, so no API is
    metered. It cannot call itself autonomously: it writes the prompt to a hand-off file and
    blocks until the answer file appears (the session, i.e. Claude, fills it in). This makes
    'engine=max' work today at zero cost while a session is open; swap to 'api' for headless.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        base = cfg.history_path.parent / "handoff"
        base.mkdir(parents=True, exist_ok=True)
        self.req = base / "request.txt"
        self.resp = base / "response.txt"
        self.poll = float(os.environ.get("DEWBRAIN_MAX_POLL", "2.0"))
        self.wait_ceiling = float(os.environ.get("DEWBRAIN_MAX_WAIT", "1800"))

    def complete(self, prompt: str, log: Callable[[str], None]) -> str:
        # publish the request, clear any stale answer
        if self.resp.exists():
            self.resp.unlink()
        self.req.write_text(prompt, encoding="utf-8")
        log(f"  [max] prompt written -> {self.req}")
        log("  [max] waiting for the Claude Code session to answer (write response.txt)...")
        waited = 0.0
        while waited < self.wait_ceiling:
            if self.resp.exists():
                text = self.resp.read_text(encoding="utf-8").strip()
                if text:
                    self.resp.unlink()
                    self.req.unlink(missing_ok=True)
                    return text
            time.sleep(self.poll)
            waited += self.poll
        raise EngineError(
            f"engine=max timed out after {self.wait_ceiling}s waiting for {self.resp}. "
            "Is a Claude Code session running to answer the hand-off?"
        )


# --------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------
def build_engine(cfg) -> Engine:
    name = (cfg.api.engine or "api").lower()
    if name == "api":
        return ApiEngine(cfg)
    if name == "max":
        return MaxEngine(cfg)
    raise EngineError(f"unknown engine '{name}' (use 'api' or 'max')")
