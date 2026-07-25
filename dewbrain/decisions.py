"""
decisions.py — dewbrain's DECISION memory (the world-model's fuel).

dewbrain's north star is NOT a writing tool. It is a Damla-model that sees her
future and decides / works / builds in her place (CLAUDE.md: "şahsi Palantir";
BRAIN_MAP: world-model "Damla bu durumda ne yapardı, hangi sonuca giderdi").

A world-model needs training data that is DECISIONS, not prose. Damla's real
engineering judgment lives in the DECISIONS.md files scattered across every
project: each line is a real choice with its reason and its reversal cost. That
is exactly the (context -> choice -> why -> outcome) signal a world-model learns
"what would Damla do" from. This module turns that scattered record into
structured decision traces the brain can recall and (later) roll forward.

Where the raw voice corpus (dewrites/dewthinks) is EPISODIC-writing, these are
EPISODIC-DECISIONS: what Damla decided, in which situation, for what reason, at
what reversal cost, and (when recorded) the outcome. They feed:
  - neocortex: consolidated decision PATTERNS (how Damla weighs a call)
  - world-model (Faz 2+): rollout "given this state, Damla picks X because Y"
  - meta-cognition: reversal-cost is a native confidence signal (cheap-to-undo
    calls are made fast/loose; expensive ones demand certainty)

TWO on-disk formats exist in Damla's repos; this parser handles BOTH without a
sync step (single source = the .md files themselves, same law.py discipline):

  INLINE  (stitchu): one bullet, fields inline ->
      "- 2026-07-19 <title>: <situation>. KARAR: <choice> / NEDEN: <reason> /
       GERI ALMA: <cost>. <trailing note>"
  SECTION (gymgyme): a "## <date · phase · title>" heading, then labelled
      lines "Durum: ... / Karar: ... / Gerekçe: ... / Etki: ...".

Field synonyms are normalized: choice<-KARAR/Karar, reason<-NEDEN/Neden/
Gerekçe, cost<-GERI ALMA/Geri alma/Etki, context<-Durum. Robust like memory.py:
code-fence aware, tolerant of missing fields (a decision with only a choice is
still a decision), never crashes on a malformed entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --- decision sources on this machine (active repos; audit-clones/worktrees excluded) ---
PROJECTS = Path.home() / "damla_projects_2026"
DECISION_FILES = [
    PROJECTS / "stitchu" / "DECISIONS.md",
    PROJECTS / "00_currently_on_working" / "stitchu" / "DECISIONS.md",
    PROJECTS / "00_currently_on_working" / "gymgyme" / "DECISIONS.md",
    PROJECTS / "inkbee" / "DECISIONS.md",
    PROJECTS / "cs_school_projects" / "vibematch_cs" / "DECISIONS.md",
]

# --- field label synonyms -> canonical slot (case-insensitive) ---
# choice = what was decided; reason = why; cost = reversal cost / effect; context = situation.
_LABELS = {
    "karar": "choice",
    "neden": "reason",
    "gerekçe": "reason",
    "gerekce": "reason",
    "geri alma": "cost",
    "geri alma maliyeti": "cost",
    "etki": "cost",
    "durum": "context",
    "hakem": "outcome",
    "kanit": "outcome",
    "kanıt": "outcome",
    "pin disiplini": "outcome",
}

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})|(\b\d{1,2}\s+(?:Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara))")
_FENCE = re.compile(r"^\s*```")
# an inline bullet that carries at least one uppercase KARAR/NEDEN/GERI field.
_INLINE_FIELD = re.compile(r"\b(KARAR|NEDEN|GERI ALMA|GERİ ALMA)\b")
# "Label: value" at line start (section format), with an optional leading "- "
# bullet (stitchu writes "- Karar: ...", gymgyme writes "Karar: ..."). Label is
# letters/spaces before the colon.
_LABEL_LINE = re.compile(r"^-?\s*([A-Za-zÇĞİÖŞÜçğıöşü ]{3,30}?):\s*(.*)$")


@dataclass
class Decision:
    """One resolved decision. Missing fields stay empty; a bare choice still counts."""
    title: str
    choice: str = ""          # KARAR / Karar — what was decided
    reason: str = ""          # NEDEN / Gerekçe — why
    cost: str = ""            # GERI ALMA / Etki — reversal cost / effect
    context: str = ""         # Durum — the situation that forced the call
    outcome: str = ""         # Hakem / Kanıt — measured result, if recorded
    project: str = ""         # which repo it came from
    date: str = ""            # ISO date or Turkish "19 Tem" if present
    source: str = ""          # file path
    raw: str = ""             # the full original text (audit / re-parse)

    def as_dict(self) -> dict:
        return asdict(self)

    def as_memory_body(self) -> str:
        """Render as a compact trace the retrieval/neocortex layer can embed.
        Kept in Damla's own words (choice/reason verbatim), law-clean shape."""
        parts = []
        if self.context:
            parts.append(f"durum: {self.context}")
        if self.choice:
            parts.append(f"karar: {self.choice}")
        if self.reason:
            parts.append(f"neden: {self.reason}")
        if self.cost:
            parts.append(f"geri alma: {self.cost}")
        if self.outcome:
            parts.append(f"sonuç: {self.outcome}")
        return "\n".join(parts) if parts else self.raw.strip()


def _project_of(path: Path) -> str:
    """Human-readable project handle from the file path."""
    p = path.parent.name
    # 00_currently_on_working/stitchu -> stitchu ; keep it simple and stable.
    return p


def _norm_label(label: str) -> Optional[str]:
    return _LABELS.get(label.strip().lower())


def _extract_date(text: str) -> str:
    m = _DATE_RE.search(text)
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip()


# ---------------------------------------------------------------------------
# INLINE format: "- <date> <title>: <situation>. KARAR: ... / NEDEN: ... / GERI ALMA: ..."
# ---------------------------------------------------------------------------
def _parse_inline_bullet(text: str, source: Path) -> Optional[Decision]:
    """Split a single inline bullet into slots. The fields are separated by ' / '
    and each starts with a KARAR/NEDEN/GERI-ALMA label; text before the first
    label is the title + situation."""
    body = text.strip().lstrip("-").strip()
    if not body:
        return None

    # Find the first field label; everything before it is title/context.
    m = _INLINE_FIELD.search(body)
    head = body[: m.start()].strip() if m else body
    tail = body[m.start():] if m else ""

    # head = "<date> <title>: <situation>"
    date = _extract_date(head)
    title, context = head, ""
    if ":" in head:
        title, context = head.split(":", 1)
        title, context = title.strip(), context.strip()
    title = re.sub(r"^\d{4}-\d{2}-\d{2}\s*", "", title).strip()

    dec = Decision(title=title or head[:60], context=context,
                   date=date, project=_project_of(source),
                   source=str(source), raw=text.strip())

    # tail = "KARAR: ... / NEDEN: ... / GERI ALMA: ..." — split on ' / ' but keep
    # slashes inside values by splitting only where a known label follows.
    if tail:
        # segment at each label boundary
        segments = re.split(r"\s*/\s*(?=(?:KARAR|NEDEN|GERI ALMA|GERİ ALMA)\b)", tail)
        for seg in segments:
            lm = re.match(r"(KARAR|NEDEN|GERI ALMA|GERİ ALMA)\s*:?\s*(.*)$", seg.strip(), re.DOTALL)
            if not lm:
                continue
            slot = _norm_label(lm.group(1))
            val = lm.group(2).strip()
            if slot and val:
                setattr(dec, slot, (getattr(dec, slot) + " " + val).strip())

    # Inline bullets sometimes state the decision in the title/situation and only
    # carry NEDEN/GERI ALMA (e.g. "review.X -> flat.style (Damla karari: ...)").
    # Don't drop the choice — fall back to the title so the trace stays complete.
    if not dec.choice.strip():
        dec.choice = dec.title
    return dec if (dec.choice or dec.reason or dec.context) else None


# ---------------------------------------------------------------------------
# SECTION format: "## <date · phase · title>" then "Label: value" lines.
# ---------------------------------------------------------------------------
def _parse_section(title_line: str, body_lines: list[str], source: Path) -> Optional[Decision]:
    title = title_line.lstrip("#").strip()
    dec = Decision(title=title, date=_extract_date(title),
                   project=_project_of(source), source=str(source),
                   raw="\n".join([title_line] + body_lines).strip())

    cur_slot: Optional[str] = None
    for ln in body_lines:
        lm = _LABEL_LINE.match(ln.strip())
        slot = _norm_label(lm.group(1)) if lm else None
        if slot:
            cur_slot = slot
            val = lm.group(2).strip()
            if val:
                setattr(dec, slot, (getattr(dec, slot) + " " + val).strip())
        elif cur_slot and ln.strip():
            # continuation line of the current field (wrapped value)
            setattr(dec, cur_slot, (getattr(dec, cur_slot) + " " + ln.strip()).strip())

    # Some section headings ARE the decision (e.g. "buton rengi TERS DÖNDÜ",
    # "MIHENK-05 v7 seçildi") with no separate Karar: line. Don't lose the
    # choice — the title carries it. Strip the "date · phase ·" prefix so the
    # choice is the actual decision, not the metadata.
    if not dec.choice.strip():
        t = re.sub(r"^.*?·\s*[^·]*·\s*", "", title).strip()  # drop "date · phase ·"
        dec.choice = t or title
    return dec if (dec.choice or dec.reason or dec.context) else None


# ---------------------------------------------------------------------------
# file walker — handles both formats in one pass, code-fence aware
# ---------------------------------------------------------------------------
def parse_decision_file(path: Path) -> list[Decision]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()

    out: list[Decision] = []
    in_fence = False
    # section accumulator
    sec_title: Optional[str] = None
    sec_body: list[str] = []

    def _flush_section():
        nonlocal sec_title, sec_body
        if sec_title is not None:
            d = _parse_section(sec_title, sec_body, path)
            if d:
                out.append(d)
        sec_title, sec_body = None, []

    for ln in lines:
        if _FENCE.match(ln):
            in_fence = not in_fence
            if sec_title is not None:
                sec_body.append(ln)
            continue
        if in_fence:
            if sec_title is not None:
                sec_body.append(ln)
            continue

        stripped = ln.strip()

        # a "## " heading opens a SECTION-format decision (or closes the previous).
        if stripped.startswith("## "):
            _flush_section()
            sec_title = stripped
            continue
        # top-of-file "# " title resets any open section.
        if stripped.startswith("# ") and not stripped.startswith("## "):
            _flush_section()
            continue

        # an inline bullet carrying an uppercase field is an INLINE-format decision
        # ONLY when no section is open. If a "## " heading is currently open, its
        # "- KARAR:/- YÖNTEM:" bullets belong to THAT section (stitchu writes both
        # styles); routing them to the standalone inline parser would shatter the
        # section (bullet split off, its indented continuation lines left behind).
        if sec_title is None and stripped.startswith("- ") and _INLINE_FIELD.search(stripped):
            d = _parse_inline_bullet(stripped, path)
            if d:
                out.append(d)
            continue

        # otherwise, if a section is open, this is body (context/continuation).
        if sec_title is not None:
            sec_body.append(ln)

    _flush_section()
    return out


def load_decisions(dedup: bool = True) -> list[Decision]:
    """All decisions across active repos. Dedups identical (title+choice) pairs
    (stitchu appears twice on disk: root + 00_currently_on_working)."""
    all_dec: list[Decision] = []
    for f in DECISION_FILES:
        all_dec.extend(parse_decision_file(f))
    if not dedup:
        return all_dec
    seen: set[tuple[str, str]] = set()
    unique: list[Decision] = []
    for d in all_dec:
        key = (d.title.strip().lower(), d.choice.strip().lower()[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# bridge to the memory layer: decisions as episodic traces retrieval can recall.
# body = the decision trace; kind = "episodic" (raw decision, not distilled gold).
# ---------------------------------------------------------------------------
def load_decision_memory() -> list[dict]:
    """Legacy {title, body, source, ...} shape so retrieval.py can embed these
    exactly like dewrites episodes. Decision = a first-class episodic trace."""
    out = []
    for d in load_decisions():
        out.append({
            "title": d.title,
            "body": d.as_memory_body(),
            "source": f"decision:{d.project}",
            "kind": "episodic",
            "section": d.project,
            "note": (f"reversal-cost: {d.cost}" if d.cost else ""),
            "channel": "decision",
            "date": d.date,
            "salience": 0.0,
        })
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    decs = load_decisions()
    by_proj: dict[str, int] = {}
    for d in decs:
        by_proj[d.project] = by_proj.get(d.project, 0) + 1
    print(f"kararlar (dedup'lu): {len(decs)}")
    for p, n in sorted(by_proj.items(), key=lambda x: -x[1]):
        print(f"  {p:16} {n}")

    # completeness: how many carry each field (fuel quality).
    full = sum(1 for d in decs if d.choice and d.reason)
    with_cost = sum(1 for d in decs if d.cost)
    with_ctx = sum(1 for d in decs if d.context)
    print(f"\nkarar+neden dolu: {full}/{len(decs)}   "
          f"geri-alma dolu: {with_cost}   durum dolu: {with_ctx}")

    print("\n--- örnek 3 karar-izi ---")
    for d in decs[:3]:
        print(f"\n[{d.project}] {d.title[:60]}")
        print("  " + d.as_memory_body().replace("\n", "\n  ")[:400])
