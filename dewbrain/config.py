"""
Config — NOT embedded in the engine. vmPFC and the API layer read from here.

Why a separate module: model / k / beta / max-iter / retry policy are operating
parameters, not engine logic. Keeping them here means tuning the loop (or swapping
the model) never touches brain.py, and env overrides work for CI / experiments.

Neuro mapping: these are the vmPFC's control knobs — how many traces to recall (k),
how sharp the Hopfield retrieval is (beta), how many critique->repair cycles the
prefrontal loop may run before it must stop (max_iter).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def _env_opt_float(name: str, default: Optional[float]) -> Optional[float]:
    """None means 'calibrate from the corpus'. An explicit env value pins it."""
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v else default
    except ValueError:
        return default


@dataclass(frozen=True)
class APIConfig:
    """Anthropic API surface — model + robustness policy."""
    # engine = which text-generation backend the vmPFC calls.
    #   "api" -> anthropic API (metered, autonomous, runs headless without a session)
    #   "max" -> Claude Code session (Max subscription, zero extra cost, needs a live session)
    # The core (retrieval/law/loop) is engine-agnostic; only this pluggable part changes.
    # A future "local" (own fine-tuned model, GPU) slots into the same interface.
    engine: str = field(default_factory=lambda: _env_str("DEWBRAIN_ENGINE", "api"))
    model: str = field(default_factory=lambda: _env_str("DEWBRAIN_MODEL", "claude-opus-4-8"))
    # max_tokens: generate/repair produce a full post; give room, stream to dodge HTTP timeout.
    max_tokens: int = field(default_factory=lambda: _env_int("DEWBRAIN_MAX_TOKENS", 8000))
    # adaptive thinking is the only on-mode for opus-4-8; effort tunes depth/cost.
    effort: str = field(default_factory=lambda: _env_str("DEWBRAIN_EFFORT", "high"))
    # per-call retry policy for rate-limit / 5xx / timeout (on top of the SDK's own retries).
    max_retries: int = field(default_factory=lambda: _env_int("DEWBRAIN_MAX_RETRIES", 5))
    base_delay: float = field(default_factory=lambda: _env_float("DEWBRAIN_BASE_DELAY", 1.0))
    max_delay: float = field(default_factory=lambda: _env_float("DEWBRAIN_MAX_DELAY", 60.0))
    # request timeout in seconds; streaming keeps the connection alive, this is the ceiling.
    timeout: float = field(default_factory=lambda: _env_float("DEWBRAIN_TIMEOUT", 600.0))


@dataclass(frozen=True)
class LoopConfig:
    """vmPFC coordinator — recall + critique->repair loop control."""
    k: int = field(default_factory=lambda: _env_int("DEWBRAIN_K", 3))
    # beta = None -> HippocampalRetrieval calibrates it from the corpus geometry
    # (the whole point of calibrate_beta). A hard 8.0 here silently overrode that
    # calibration in the production path; it must stay None unless explicitly pinned.
    beta: Optional[float] = field(default_factory=lambda: _env_opt_float("DEWBRAIN_BETA", None))
    # loop-until-clean: max critique->repair cycles before we stop and keep the best draft.
    max_iter: int = field(default_factory=lambda: _env_int("DEWBRAIN_MAX_ITER", 4))
    embedding_model: str = field(
        default_factory=lambda: _env_str("DEWBRAIN_EMBED_MODEL", "all-mpnet-base-v2")
    )


@dataclass(frozen=True)
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    # gold-accumulation history: accepted output + verdict + gold-used appended here (JSONL).
    history_path: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "DEWBRAIN_HISTORY",
                str(Path(__file__).resolve().parent.parent / "runs" / "history.jsonl"),
            )
        )
    )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["history_path"] = str(self.history_path)
        return d


def load_config() -> Config:
    """Single entry point. Env vars override defaults; nothing is baked into the engine."""
    return Config()
