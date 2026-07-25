"""
memory.py — dewbrain's memory layer.

Two stores, one clean boundary (research c, brain-architecture.md):
  EPISODIC  = raw traces (dewrites/dewthinks/dewlog/dewideo .md)      -> what happened
  SEMANTIC  = gold, Damla-approved voice (*_verified.md + grown gold) -> distilled voice

Design pillars, each tied to a verified deep-research finding:

  1. ROBUST PARSE (not fragile regex). The old corpus.py did
     `re.split("^### ")` then `split("\n---")` and threw away the `>` note.
     That breaks the moment a body contains "---" (thematic break),
     a nested "###", or a fenced code block with "###" comments. Here a
     small state-machine tokenizer walks the file line by line, tracks
     fenced ``` blocks, and only treats `##`/`###`/`---`/`>` as structure
     OUTSIDE code fences. The `>` provenance note is CAPTURED, not dropped
     (it is the salience/why-verified signal, research c: salience score).

  2. EPISODIC vs SEMANTIC as first-class objects (MemoryEntry), not bare
     dicts, so consolidation can attach salience, provenance, embeddings,
     and lineage without breaking callers.

  3. CONSOLIDATION BRIDGE (research c: memory-stream + reflection + salience,
     arXiv:2304.03442; ExpeL add/edit/vote, parameter-free). An approved
     output is APPENDED to the grown-gold store (data/gold_grown.jsonl) with
     its provenance (cue, law verdict, reflection). Nothing is silently
     overwritten (ExpeL add, not replace). This is how the ~9-gold corpus
     grows without a GPU, feeding Faz 2's future fine-tune.

  4. REFLECTION / DISTILLATION (research c: raw episodic -> semantic).
     `reflect()` produces the salience-scored, provenance-carrying record
     that gets stored. Salience (poignancy 1-10) gates what is worth keeping
     as semantic gold vs what stays raw episodic.

Backwards compatible: load_gold() / load_raw() still return the plain
{title, body, source} dicts brain.py + retrieval.py already consume.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

# --- source data (this machine) ---
ICERIK = Path.home() / "damla_projects_2026" / "icerik"

# --- consolidation store (gitignored data/, grows over time) ---
DATA = Path(__file__).resolve().parent.parent / "data"
GOLD_GROWN = DATA / "gold_grown.jsonl"        # HUMAN-approved outputs -> semantic gold
GOLD_CANDIDATES = DATA / "gold_candidates.jsonl"  # clean-but-unconfirmed, await Damla
EPISODIC_LOG = DATA / "episodic_stream.jsonl"  # every generation attempt (memory stream)

# verified files = semantic gold; raw files = episodic pool.
GOLD_SOURCES = ["dewrites_verified", "dewthinks_verified",
                "dewlog_verified", "dewideo_verified"]
RAW_SOURCES = ["dewrites", "dewthinks", "dewlog", "dewideo"]

# Headings that are LAW/DOC scaffolding inside the .md, never voice samples.
# e.g. "A. YEDİ OMURGA", "B. MEKANİK KURALLAR", "C. DAMLA-KOKUYOR MU?", "D. SÜREÇ".
_LAW_HEADING = re.compile(r"^[A-Z]\.\s")
# Placeholder stubs like "_(henüz verified reel yok ...)_" are not entries.
_PLACEHOLDER = re.compile(r"^_\(.*\)_\s*$")
_FENCE = re.compile(r"^\s*```")


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    """One trace. Episodic (raw) or semantic (gold) is decided by `kind`."""
    title: str
    body: str
    source: str                      # which file / origin it came from
    kind: str = "semantic"           # "semantic" (gold) | "episodic" (raw)
    section: str = ""                # "## [project]" it lived under
    note: str = ""                   # the `>` provenance note (why verified)
    channel: str = ""                # LinkedIn / Substack / reel ... (from title tail)
    date: str = ""                   # GG.AA.YYYY if present in the title
    salience: float = 0.0            # poignancy 1-10 (0 = unscored raw)
    provenance: dict = field(default_factory=dict)  # lineage for grown gold

    def as_dict(self) -> dict:
        # brain.py/retrieval.py only read title/body/source; keep them present.
        return asdict(self)


# ---------------------------------------------------------------------------
# robust parser (state machine, code-fence aware, > note capturing)
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")


def _split_title(raw_title: str) -> tuple[str, str, str]:
    """
    "on kitsch — Substack — VERIFIED (23.07.2026)"
      -> ("on kitsch — Substack — VERIFIED (23.07.2026)", channel, date)
    We keep the full title (Damla writes em-dash in titles as metadata, not body),
    but pull out channel + date as retrieval hints without mangling the title.
    """
    date = ""
    m = _DATE_RE.search(raw_title)
    if m:
        date = m.group(1)
    channel = ""
    for ch in ("LinkedIn", "Substack", "reel", "carousel", "vlog", "video", "Instagram"):
        if re.search(rf"\b{re.escape(ch)}\b", raw_title, re.IGNORECASE):
            channel = ch
            break
    return raw_title.strip(), channel, date


def parse_file(md_path: Path, source: str, kind: str) -> list[MemoryEntry]:
    """
    Walk the markdown as a token stream. Structure markers (##, ###, ---, >)
    are only honored OUTSIDE fenced code blocks, so a body containing a
    thematic '---' or a '###' comment does not shatter the parse.

    An entry = a '### heading' + optional consecutive '>' note lines + body,
    terminated by the next '###'/'##' heading, a top-of-file '#' heading, or a
    bare '---' rule that sits OUTSIDE the entry body (we only cut on '---' when
    it is on its own line at fence-depth 0 AND a heading already opened).
    """
    if not md_path.exists():
        return []
    lines = md_path.read_text(encoding="utf-8").splitlines()

    entries: list[MemoryEntry] = []
    section = ""            # current "## [project]"
    cur: Optional[dict] = None  # open entry accumulator
    note_phase = False     # are we still collecting '>' lines right after a heading?
    in_fence = False

    def _flush():
        nonlocal cur
        if cur is None:
            return
        body = "\n".join(cur["body"]).strip()
        # strip a trailing '---' rule if it slipped in as the last body line
        body = re.sub(r"\n?-{3,}\s*$", "", body).strip()
        title = cur["title"].strip()
        if body and not _LAW_HEADING.match(title) and not _PLACEHOLDER.match(title):
            full_title, channel, date = _split_title(title)
            entries.append(MemoryEntry(
                title=full_title, body=body, source=source, kind=kind,
                section=cur["section"], note=cur["note"].strip(),
                channel=channel, date=date,
            ))
        cur = None

    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            if cur is not None:
                cur["body"].append(line)
                note_phase = False
            continue

        if in_fence:
            if cur is not None:
                cur["body"].append(line)
            continue

        stripped = line.strip()

        # top-level file title -> resets; not an entry
        if stripped.startswith("# ") and not stripped.startswith("## "):
            _flush()
            continue

        # section header "## [project]"
        if stripped.startswith("## "):
            _flush()
            section = stripped[3:].strip()
            continue

        # entry header "### ..."
        if stripped.startswith("### "):
            _flush()
            cur = {"title": stripped[4:], "section": section,
                   "note": "", "body": []}
            note_phase = True
            continue

        # provenance note: consecutive '>' lines immediately after a heading
        if cur is not None and note_phase and stripped.startswith(">"):
            cur["note"] += " " + stripped[1:].strip()
            continue

        # a bare horizontal rule closes the current entry (outside fence)
        if re.fullmatch(r"-{3,}", stripped):
            _flush()
            continue

        # any non-empty, non-'>' line ends the note phase and becomes body
        if cur is not None:
            if stripped:
                note_phase = False
            cur["body"].append(line)

    _flush()
    return entries


# ---------------------------------------------------------------------------
# episodic / semantic loaders
# ---------------------------------------------------------------------------
def load_semantic(include_grown: bool = True) -> list[MemoryEntry]:
    """Gold voice: *_verified.md + grown gold from approved outputs."""
    out: list[MemoryEntry] = []
    for name in GOLD_SOURCES:
        out.extend(parse_file(ICERIK / f"{name}.md", source=name, kind="semantic"))
    if include_grown:
        out.extend(_load_grown_gold())
    return out


def load_episodic() -> list[MemoryEntry]:
    """Raw pool: the working .md files. Ideas/angles, not approved voice."""
    out: list[MemoryEntry] = []
    for name in RAW_SOURCES:
        out.extend(parse_file(ICERIK / f"{name}.md", source=name, kind="episodic"))
    return out


def _load_grown_gold() -> list[MemoryEntry]:
    if not GOLD_GROWN.exists():
        return []
    out = []
    for ln in GOLD_GROWN.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        out.append(MemoryEntry(
            title=d.get("title", ""), body=d["body"],
            source="gold_grown", kind="semantic",
            note=d.get("note", ""), salience=d.get("salience", 0.0),
            provenance=d.get("provenance", {}),
        ))
    return out


# ---------------------------------------------------------------------------
# backwards-compatible shims (brain.py + retrieval.py already use these)
# ---------------------------------------------------------------------------
def load_gold() -> list[dict]:
    """Legacy shape: list of {title, body, source, ...}. Gold = semantic."""
    return [e.as_dict() for e in load_semantic()]


def load_raw() -> list[dict]:
    """Legacy shape for the episodic pool."""
    return [e.as_dict() for e in load_episodic()]


# ---------------------------------------------------------------------------
# consolidation bridge: approved output -> grown gold  (research c)
# ---------------------------------------------------------------------------
def _substance_prior(body: str) -> float:
    """Poignancy WITHOUT a law verdict (research c, generative agents). The old
    code returned a flat 5.0 when no verdict existed, so every gold candidate
    scored identical 0.50 — salience did no ranking at all. This measures how
    'strong/memorable' a text is from deterministic signals that track Damla's
    own voice law: a carried THESIS, binary CONTRAST, concrete NUMBERS (her law:
    'sayı = güç'), and a HUMAN mark. Returns 1..10, no model, no GPU.
    """
    text = (body or "").strip()
    if not text:
        return 1.0
    words = text.split()
    n = len(words)
    low = text.lower()

    score = 3.0  # neutral floor for a real-but-plain trace

    # length band: too short can't teach voice; a full post is substantive.
    if n >= 40:
        score += 1.0
    if n >= 120:
        score += 1.0
    if n < 20:
        score -= 1.5

    # concrete numbers / measurements = evidence (Damla: numbers are the hook).
    import re as _re
    n_numbers = len(_re.findall(r"\b\d[\d.,]*\b", text))
    if n_numbers >= 1:
        score += 0.7
    if n_numbers >= 3:
        score += 0.6

    # binary contrast — her sharpening blade ("not X but Y", "değil ... ama").
    contrast = ("not because", "but because", "instead of", " not a ", " not the ",
                "değil", " ama ", "yerine", " vs ", "rather than")
    if any(c in low for c in contrast):
        score += 0.8

    # human mark / stance — the lines that make it hers, not a report.
    human = ("i just love", "i wish", "the truth is", "i stopped", "i kept",
             "ben ", "kendi", "bu yüzden", "aslında")
    if any(h in low for h in human):
        score += 0.6

    # a single-clause one-liner rarely carries a full thesis.
    if text.count(".") + text.count("\n") <= 1 and n < 30:
        score -= 1.0

    return max(1.0, min(10.0, score))


def salience(entry_body: str, verdict: Optional[dict] = None,
             human_approved: bool = False) -> float:
    """
    Poignancy 1-10 (research c, generative agents). Cheap, deterministic,
    parameter-free (no GPU). Combines:
      - law pass rate (a clean pass is a strong keep signal), when a verdict exists
      - a SUBSTANCE prior from the text itself when there is no verdict (so gold
        candidates actually RANK instead of all scoring a flat 5.0)
      - explicit human approval (Damla said "this is me") = ceiling
    Salience gates what is worth consolidating into semantic gold.
    """
    if verdict and verdict.get("items"):
        items = verdict["items"]
        passed = sum(1 for i in items if i.get("pass"))
        score = 1.0 + 9.0 * (passed / max(1, len(items)))
        # NOTE: `damla_kokuyor` is the MODEL judging its own output. It must not
        # be able to floor salience into the auto-consolidation zone — that would
        # let the engine self-certify its own voice into gold. "This is me" is
        # Damla's call (human_approved), not the critic's. Model judgment nudges
        # ranking, it does not certify identity.
        if verdict.get("damla_kokuyor") is True:
            score = min(9.0, score + 0.5)  # nudge, capped below the human ceiling
    else:
        # no verdict (e.g. ranking existing raw traces as gold candidates):
        # score from the text's own substance instead of a flat constant.
        score = _substance_prior(entry_body)

    words = len(entry_body.split())
    if words < 40:              # too thin to teach voice
        score -= 2.0
    if human_approved:
        score = 10.0
    return max(1.0, min(10.0, round(score, 1)))


def reflect(cue: str, final_text: str, verdict: Optional[dict],
            gold_used: Optional[list[str]] = None,
            human_approved: bool = False) -> dict:
    """
    Distill a raw generation episode into a semantic-gold candidate record
    (research c: reflection turns episodic stream -> semantic knowledge).
    Returns the record; does NOT store it (call consolidate() to persist).
    """
    sal = salience(final_text, verdict, human_approved)
    return {
        "title": _infer_title(final_text),
        "body": final_text.strip(),
        "salience": sal,
        "note": _reflection_note(verdict, human_approved),
        "provenance": {
            "cue": cue.strip()[:400],
            "gold_used": gold_used or [],
            "law_pass": _pass_summary(verdict),
            "human_approved": human_approved,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }


def consolidate(record: dict, min_salience: float = 8.0,
                human_approved: bool = False) -> str:
    """
    ExpeL-style add (arXiv, parameter-free), but voice-safe.

    ROOT PROBLEM this guards: gold is the ONLY source of the few-shot voice
    (CLAUDE.md: "ses konsolidasyonu SADECE verified'dan"). If the engine's own
    clean-but-generic outputs auto-promote into gold, the voice recursively
    drifts to the model's mean and collapses — model self-certifying its own
    identity. So identity ("bu ben miyim") requires DAMLA, not the critic.

    Routing (always logs to the episodic stream first, so nothing is lost):
      - human_approved      -> GOLD_GROWN     (real semantic gold, feeds voice)
      - clean & high sal.    -> GOLD_CANDIDATES (queued, awaits Damla; NOT voice)
      - otherwise            -> episodic log only

    Returns "gold" | "candidate" | "logged".
    """
    DATA.mkdir(parents=True, exist_ok=True)
    _append_jsonl(EPISODIC_LOG, {**record, "consolidated": False})

    if human_approved:
        if _is_duplicate(GOLD_GROWN, record["body"]):
            return "gold"  # idempotent
        _append_jsonl(GOLD_GROWN, record)
        return "gold"

    # Model-clean but human-unconfirmed: queue for review, do NOT feed the voice.
    if record.get("salience", 0.0) >= min_salience:
        if _is_duplicate(GOLD_CANDIDATES, record["body"]):
            return "candidate"
        _append_jsonl(GOLD_CANDIDATES, record)
        return "candidate"

    return "logged"


def _is_duplicate(path: Path, body: str) -> bool:
    """Guard the add path so consolidate() is idempotent per body, per store."""
    if not path.exists():
        return False
    key = body.strip()
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            if json.loads(ln).get("body", "").strip() == key:
                return True
    return False


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _pass_summary(verdict: Optional[dict]) -> str:
    if not verdict or not verdict.get("items"):
        return "unscored"
    items = verdict["items"]
    p = sum(1 for i in items if i.get("pass"))
    return f"{p}/{len(items)}"


def _reflection_note(verdict: Optional[dict], human_approved: bool) -> str:
    if human_approved:
        return "Damla approved (this is me). Consolidated into semantic gold."
    if verdict and verdict.get("items"):
        weak = verdict.get("en_zayif") or ""
        return f"auto-consolidated, law {_pass_summary(verdict)}. weakest: {weak}".strip()
    return "auto-consolidated (no law verdict attached)."


def _infer_title(text: str) -> str:
    """A stable, human-readable handle for a grown-gold record."""
    first = text.strip().splitlines()[0] if text.strip() else "untitled"
    first = re.sub(r"\s+", " ", first).strip()
    return (first[:70] + "…") if len(first) > 70 else first


# ---------------------------------------------------------------------------
# few-shot style scaffold for the ~9-gold regime  (research e)
# ---------------------------------------------------------------------------
def style_context(gold_hits: Iterable[dict], law_snippets: bool = True) -> str:
    """
    research e: with ~9 gold, a raw model drifts to GENERIC. The fix at this
    tiny-corpus scale is NOT fine-tuning (kills the voice, research b) but
    explicit style anchoring: (1) contrastive framing (learn the VOICE, do not
    copy the topic), (2) always carry the provenance `>` note so the model sees
    WHY each sample is Damla, not just the text. This packs the few gold we
    have as maximally informative style exemplars for the generator prompt.
    """
    blocks = []
    for g in gold_hits:
        note = g.get("note", "")
        head = "--- ÖRNEK (Damla'nın onaylı sesi) ---"
        why = f"\n[neden Damla: {note}]" if (law_snippets and note) else ""
        blocks.append(f"{head}{why}\n{g['body']}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sem = load_semantic(include_grown=False)
    epi = load_episodic()
    print(f"semantic gold (verified): {len(sem)}")
    for e in sem:
        print(f"  [{e.source}] sal-hint={'✓note' if e.note else '   '} "
              f"{e.channel or '?':9} {e.title[:52]}")
    print(f"\nepisodic raw (pool): {len(epi)}")
    for e in epi[:10]:
        print(f"  [{e.source}] {e.section[:14]:14} {e.title[:48]}")

    # consolidation bridge dry-run (no API needed)
    fake_verdict = {
        "items": [{"key": f"k{i}", "pass": True} for i in range(9)]
                 + [{"key": "k9", "pass": False, "note": "em dash slipped"}],
        "damla_kokuyor": True,
        "en_zayif": "one clause reads slightly corporate",
    }
    rec = reflect(
        cue="raw idea: I could not 'do marketing' but I could write one honest post",
        final_text=("For a long time I said I can't about things that had nothing to do "
                    "with skill. I could not do marketing, but I could write one honest "
                    "post about one real number, and that was the whole difference."),
        verdict=fake_verdict, gold_used=["fifty projects", "neden farklı yazıyorum"],
    )
    print(f"\n[reflect] salience={rec['salience']}  title={rec['title']!r}")
    print(f"[reflect] note={rec['note']}")
