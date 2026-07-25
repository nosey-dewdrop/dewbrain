"""
sources.py — dewbrain's SOURCE REGISTRY (every trace Damla produces feeds the brain).

Damla (25 Tem): the brain is not fed from one channel. Its episodic memory is
EVERYTHING she produces — writing, decisions, audits, project maps, ideas, and
(when pointed at them) personal notes / an Obsidian vault. The brain distills
"who Damla is" from the whole trail, not a single file.

Engineering discipline (root cause, not a patch): binding each channel to its
own bespoke loader is fragile — every new channel would mean new code. Instead
this is ONE registry. A channel is a small spec (where the files are, how to
turn each into episodic traces, what weight/kind it carries). Adding Obsidian
or a notes folder later = ONE entry here, not a new module.

Channel taxonomy (all EPISODIC — raw experience the brain recalls and later
consolidates; none are pre-approved gold, which stays *_verified.md only):
  writing    -> dewrites/dewthinks/dewlog/dewideo raw          (memory.load_raw)
  decision   -> DECISIONS.md across active repos               (decisions.py)
  report     -> reports/*.md|*.txt (audits, strategy, research)
  projectdoc -> every repo's CLAUDE.md (what + status + ideas)
  idea       -> IDEA_PARKING.md (commercial-idea inventory)
  notes      -> (optional) personal notes / Obsidian vault, added on request

Each channel yields the legacy {title, body, source, kind, section, note,
channel, date, salience} dict shape so retrieval.py / neocortex embed them
uniformly. Nothing here is voice-gold; this is the raw episodic stream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import memory
import decisions

PROJECTS = Path.home() / "damla_projects_2026"
REPORTS = PROJECTS / "reports"
IDEA_PARKING = PROJECTS / "IDEA_PARKING.md"

# repos/paths that are clones, worktrees or build noise — never a real source.
_EXCLUDE = ("/.audit-clones/", "/worktrees/", "/node_modules/", "/.venv/",
            "/__pycache__/", "/dewbrain/")


def _excluded(p: Path) -> bool:
    s = str(p)
    return any(x in s for x in _EXCLUDE)


# ---------------------------------------------------------------------------
# generic single-doc -> traces splitter (reports, CLAUDE.md, idea parking)
# splits a markdown file into "## section" traces; keeps the file title as
# context so a short section still carries where it came from.
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _split_markdown_sections(path: Path, source: str, channel: str,
                             min_chars: int = 40) -> list[dict]:
    """A doc -> list of {section-heading + body} traces. The file's top '#'
    title is the shared context; each '## ' opens a trace. Files with no '##'
    become a single trace (the whole doc). Code-fence aware so '##' inside a
    code block is not mistaken for a heading."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    lines = text.splitlines()

    file_title = ""
    for ln in lines[:5]:
        if ln.strip().startswith("# ") and not ln.strip().startswith("## "):
            file_title = ln.strip()[2:].strip()
            break
    if not file_title:
        file_title = path.stem

    date = ""
    dm = _DATE_RE.search(path.name)
    if dm:
        date = dm.group(1)

    traces: list[dict] = []
    cur_head: Optional[str] = None
    cur_body: list[str] = []
    in_fence = False

    def _flush():
        nonlocal cur_head, cur_body
        body = "\n".join(cur_body).strip()
        if body and len(body) >= min_chars:
            head = cur_head or file_title
            traces.append({
                "title": f"{file_title} — {head}" if cur_head else file_title,
                "body": body,
                "source": source,
                "kind": "episodic",
                "section": file_title,
                "note": "",
                "channel": channel,
                "date": date,
                "salience": 0.0,
            })
        cur_head, cur_body = None, []

    for ln in lines:
        if _FENCE.match(ln):
            in_fence = not in_fence
            cur_body.append(ln)
            continue
        if not in_fence and ln.strip().startswith("## "):
            _flush()
            cur_head = ln.strip()[3:].strip()
            continue
        if not in_fence and ln.strip().startswith("# ") and not ln.strip().startswith("## "):
            # top title line; already captured as file_title, skip as body
            continue
        cur_body.append(ln)
    _flush()
    return traces


# ---------------------------------------------------------------------------
# channel loaders
# ---------------------------------------------------------------------------
def _load_writing() -> list[dict]:
    """dewrites/dewthinks/dewlog/dewideo raw = episodic writing pool."""
    return memory.load_raw()


def _load_decisions() -> list[dict]:
    """DECISIONS.md across active repos = episodic decisions (world-model fuel)."""
    return decisions.load_decision_memory()


def _load_reports() -> list[dict]:
    """reports/*.md|*.txt = audits, strategy, research (analysis reasoning)."""
    out: list[dict] = []
    if not REPORTS.exists():
        return out
    for p in sorted(REPORTS.glob("*")):
        if p.suffix.lower() not in (".md", ".txt") or _excluded(p):
            continue
        out.extend(_split_markdown_sections(p, source=f"report:{p.stem}",
                                             channel="report"))
    return out


def _load_projectdocs() -> list[dict]:
    """every repo's CLAUDE.md = project map (what + status + ideas)."""
    out: list[dict] = []
    for p in sorted(PROJECTS.rglob("CLAUDE.md")):
        if _excluded(p):
            continue
        proj = p.parent.name
        out.extend(_split_markdown_sections(p, source=f"projectdoc:{proj}",
                                            channel="projectdoc"))
    return out


def _load_ideas() -> list[dict]:
    """IDEA_PARKING.md = commercial-idea inventory (what-to-build instinct)."""
    return _split_markdown_sections(IDEA_PARKING, source="idea:parking",
                                    channel="idea", min_chars=20)


# Desktop holds Damla's real idea/thesis/note material (video-fikirleri = how she
# thinks, content-creator gece-* = her writing, damla.md = who she is). It is NOT
# under damla_projects_2026, so the registry pulls it explicitly. WHITELIST only:
# code homework (LeetCode, CS102), assets, and raw chat logs are not her voice.
DESKTOP = Path.home() / "Desktop"
_DESKTOP_INCLUDE = [
    "video-fikirleri",        # idea/thesis material (Pink Floyd->loneliness, Zimbardo->power)
    "content-creator-agent/gece-yazilar.md",
    "content-creator-agent/gece-linkedin.md",
    "content-creator-agent/gece-reels.md",
    # damla.md + damumya/ DELIBERATELY EXCLUDED: those are raw, un-curated personal
    # notes. Damla's publishable self already lives, curated, in the content system
    # (dewrites/dewlog/dewthinks/dewideo). The brain learns her voice from the
    # curated channel, not from raw private notes (her call, 25 Tem).
]


def _load_desktop() -> list[dict]:
    """Damla's Desktop idea/note material (whitelisted paths only)."""
    out: list[dict] = []
    for rel in _DESKTOP_INCLUDE:
        p = DESKTOP / rel
        if p.is_dir():
            for f in sorted(p.rglob("*.md")):
                if _excluded(f):
                    continue
                out.extend(_split_markdown_sections(f, source=f"desktop:{f.stem}",
                                                    channel="desktop", min_chars=30))
        elif p.is_file():
            out.extend(_split_markdown_sections(p, source=f"desktop:{p.stem}",
                                                channel="desktop", min_chars=30))
    return out


def _load_notes_from(root_env: str) -> Callable[[], list[dict]]:
    """Factory: a notes/Obsidian loader over a directory given by env/arg.
    Not wired by default — Damla points it at her vault when she wants it in."""
    def _load() -> list[dict]:
        import os
        root = os.environ.get(root_env, "")
        if not root:
            return []
        base = Path(root).expanduser()
        if not base.exists():
            return []
        out: list[dict] = []
        for p in sorted(base.rglob("*.md")):
            if _excluded(p):
                continue
            out.extend(_split_markdown_sections(p, source=f"notes:{p.stem}",
                                                channel="notes"))
        return out
    return _load


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    """One data channel feeding the brain. weight scales its retrieval salience
    prior (a decision is worth more per-trace than a report line); enabled lets
    a channel be parked without deleting its spec."""
    name: str
    loader: Callable[[], list[dict]]
    weight: float = 1.0
    enabled: bool = True
    note: str = ""


REGISTRY: list[Channel] = [
    Channel("writing", _load_writing, weight=1.0,
            note="dewrites/thinks/log/ideo raw — episodic voice pool"),
    Channel("decision", _load_decisions, weight=1.4,
            note="DECISIONS.md — world-model fuel, highest per-trace value"),
    Channel("report", _load_reports, weight=0.9,
            note="reports/ audits+strategy+research — analysis reasoning"),
    Channel("projectdoc", _load_projectdocs, weight=0.8,
            note="every CLAUDE.md — project map, what+status+ideas"),
    Channel("idea", _load_ideas, weight=1.1,
            note="IDEA_PARKING — what-to-build instinct"),
    Channel("desktop", _load_desktop, weight=1.2,
            note="Desktop idea/thesis/notes — how Damla thinks (video-fikirleri, gece-*, damla.md)"),
    # Parked until Damla points it at her vault (DEWBRAIN_NOTES_DIR / Obsidian).
    Channel("notes", _load_notes_from("DEWBRAIN_NOTES_DIR"), weight=1.0,
            enabled=False, note="personal notes / Obsidian — set DEWBRAIN_NOTES_DIR"),
]


# retrieval embeds a trace as ONE vector; a 300 KB report section has no usable
# single embedding (it averages to mush). Cap a trace's embeddable body: the
# core idea sits at the top of a section, so the head is the honest signal.
# Full text stays in `body_full` for later (consolidation reads the whole thing).
_MAX_BODY = 4000
_MIN_BODY = 30


def _cap_body(t: dict) -> dict:
    body = t.get("body", "")
    if len(body) > _MAX_BODY:
        t["body_full"] = body
        t["body"] = body[:_MAX_BODY].rsplit(" ", 1)[0] + " …"
    return t


def load_all(only: Optional[list[str]] = None) -> list[dict]:
    """Every enabled channel's episodic traces, tagged with channel weight.
    `only` restricts to named channels (for testing one at a time). Traces are
    body-capped for embeddability and drop-filtered below a minimum length."""
    out: list[dict] = []
    for ch in REGISTRY:
        if not ch.enabled and (only is None or ch.name not in (only or [])):
            continue
        if only is not None and ch.name not in only:
            continue
        try:
            traces = ch.loader()
        except Exception as e:  # a bad channel must not sink the whole brain
            print(f"[sources] channel '{ch.name}' failed: {e}")
            continue
        for t in traces:
            # the registry channel name is authoritative (writing/decision/...).
            # a sub-source's own tag (LinkedIn/Substack) is kept as subchannel so
            # channel_counts is correct and retrieval can still see the surface.
            sub = t.get("channel", "")
            if sub and sub != ch.name:
                t["subchannel"] = sub
            t["channel"] = ch.name
            t["channel_weight"] = ch.weight
        out.extend(traces)
    # cap oversized bodies for embeddability; drop traces too thin to teach.
    out = [_cap_body(t) for t in out if len(t.get("body", "").strip()) >= _MIN_BODY]
    return out


def channel_counts(only: Optional[list[str]] = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in load_all(only):
        counts[t.get("channel", "?")] = counts.get(t.get("channel", "?"), 0) + 1
    return counts


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import statistics
    traces = load_all()
    counts = channel_counts()
    print(f"dewbrain episodik bellek — TÜM kanallar: {len(traces)} iz\n")
    for ch in REGISTRY:
        state = "" if ch.enabled else "  (parked)"
        print(f"  {ch.name:11} {counts.get(ch.name, 0):5}  w={ch.weight}{state}")
        print(f"              {ch.note}")
    lens = sorted(len(t["body"]) for t in traces)
    if lens:
        print(f"\nbody uzunluğu: min={lens[0]} medyan={statistics.median(lens):.0f} "
              f"max={lens[-1]}")
    print("\n--- örnek iz (her kanaldan biri) ---")
    seen = set()
    for t in traces:
        c = t.get("channel")
        if c in seen:
            continue
        seen.add(c)
        print(f"\n[{c}] {t['title'][:60]}")
        print("  " + t["body"][:160].replace("\n", " "))
