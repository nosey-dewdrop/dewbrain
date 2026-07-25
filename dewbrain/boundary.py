"""
boundary.py — dewbrain's SENSITIVE-DATA boundary ("beyin bilir, söylemez").

CLAUDE.md's law: the brain KNOWS everything (mother/health/money/personal detail
must be retrievable for grounding) but must NEVER surface it in output. Until now
this was only a written rule, not code — exactly the gap today's CLAUDE.md leak
exposed. This module makes the boundary executable.

It is NOT a naive banned-word list (anne/para/hasta grep-and-strip). That is
brittle: it misses paraphrases ("my mom's treatment"), euphemisms, and numbers in
context, while nuking innocent uses ("annelik izni" in a product doc). Two real
layers:

  1. TRACE TAGGING (retrieval side): a memory trace can be marked sensitive. The
     brain may RECALL it (grounding, "what happened") but it is excluded from the
     few-shot voice block and from anything copied verbatim into output. Know,
     don't quote.

  2. OUTPUT SCREENING (generation side): the produced text is scanned for
     sensitive PATTERNS before release, combining:
       - structural signals: money amounts, health/medical terms, a private
         name adjacent to a private fact (regex + a small curated lexicon), and
       - SEMANTIC proximity: cosine of the output (sentence-wise) to a small set
         of sensitivity ANCHORS (health, finance, family-private). A sentence too
         close to an anchor is flagged even with no banned word — this catches
         the paraphrase a word list misses.
     A flagged span is either redacted or the whole output is held for review,
     per policy. Fail-closed: if the screener errors, it holds (does not release).

The lexicon/anchors are Damla-configurable and default-conservative. This is a
correctness-and-safety layer, not a heuristic bolt-on: the gate is "prove it is
safe to say", mirroring metacognition's "prove it is Damla".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


# --- curated lexicon (structural signal, NOT the whole defense) ---
# health/medical, finance, family-private. Kept small + high-precision; the
# semantic layer carries recall, this carries fast obvious hits.
_HEALTH = ("hasta", "hastalık", "teşhis", "tedavi", "ameliyat", "ilaç", "tanı",
           "kanser", "depresyon", "terapi", "psikiyatr", "diagnos", "surgery",
           "chemo", "therapy", "medication", "hospital", "hastane", "semptom")
_FINANCE = ("maaş", "salary", "borç", "kredi", "gelir", "kazanç", "ciro",
            "revenue", "net worth", "servet")
_FAMILY = ("annem", "babam", "kardeşim", "ailemin", "my mom", "my mother",
           "my dad", "my father", "my sister", "my brother")

# money amounts: 1.200 TL, $50k, 50.000₺, %-of-income phrasings.
_MONEY_RE = re.compile(
    r"(?:[$₺€£]\s?\d[\d.,]*\s?(?:k|m|bin|milyon)?)"
    r"|(?:\d[\d.,]*\s?(?:tl|try|usd|eur|dolar|euro|lira|₺))",
    re.IGNORECASE)


@dataclass
class BoundaryVerdict:
    safe: bool
    flagged_sentences: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    redacted_text: str = ""       # output with flagged spans masked (if requested)

    def as_dict(self) -> dict:
        return {"safe": self.safe, "flagged": self.flagged_sentences[:5],
                "reasons": self.reasons[:5]}


def is_trace_sensitive(trace: dict) -> bool:
    """Layer 1: should this memory trace be recall-only (never quoted)? A trace
    is sensitive if its own note marks it, its channel is a private-notes channel,
    or its body trips the lexicon hard. The brain may still RECALL it for
    grounding; callers must not copy it verbatim into the few-shot/voice block."""
    if trace.get("sensitive") is True:
        return True
    ch = trace.get("channel", "")
    if ch in ("notes", "personal"):
        return True
    body = (trace.get("body", "") or "").lower()
    # a trace is quote-unsafe if it carries a private fact WITH a private name.
    has_family = any(f in body for f in _FAMILY)
    has_health = any(h in body for h in _HEALTH)
    has_money = bool(_MONEY_RE.search(body))
    return (has_family and (has_health or has_money))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class OutputScreen:
    """Layer 2: screen produced text before release. Structural lexicon/regex +
    semantic proximity to sensitivity anchors. Fail-closed on error (holds)."""

    _ANCHORS = [
        "annemin sağlığı ve tedavisi, hastalığı, doktoru",          # family health
        "kazandığım para, gelirim, maaşım, ne kadar kazandığım",     # personal finance
        "ailemle yaşadığım özel, kişisel, mahrem bir an",            # family-private
        "ruh halim, depresyonum, terapim, psikolojik durumum",      # mental health
    ]

    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 sem_threshold: float = 0.55):
        self.sem_threshold = sem_threshold
        self._model = None
        self._model_name = model_name
        self._anchor_emb = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._anchor_emb = np.asarray(
                self._model.encode(self._ANCHORS, normalize_embeddings=True),
                dtype=np.float32)

    def _structural_hit(self, sent: str) -> str | None:
        low = sent.lower()
        if _MONEY_RE.search(sent):
            return "para tutarı"
        fam = any(f in low for f in _FAMILY)
        if fam and any(h in low for h in _HEALTH):
            return "aile + sağlık detayı"
        if fam and any(m in low for m in _FINANCE):
            return "aile + para detayı"
        # a lone strong health/mental term about self is also sensitive.
        strong_health = ("kanser", "depresyon", "psikiyatr", "chemo", "ameliyat")
        if any(s in low for s in strong_health):
            return "sağlık/ruh sağlığı detayı"
        return None

    def screen(self, text: str, redact: bool = True) -> BoundaryVerdict:
        """Return a verdict. safe=False if any sentence trips structural OR
        semantic sensitivity. Fail-closed: on any error, safe=False (hold)."""
        try:
            sentences = _split_sentences(text)
            flagged, reasons = [], []

            # structural pass (cheap, no model)
            struct_flags = {}
            for s in sentences:
                hit = self._structural_hit(s)
                if hit:
                    struct_flags[s] = hit

            # semantic pass (proximity to anchors)
            sem_flags = {}
            if sentences:
                self._ensure_model()
                emb = np.asarray(
                    self._model.encode(sentences, normalize_embeddings=True),
                    dtype=np.float32)
                sims = emb @ self._anchor_emb.T  # (S, A)
                for i, s in enumerate(sentences):
                    m = float(sims[i].max())
                    if m >= self.sem_threshold:
                        sem_flags[s] = m

            redacted = text
            for s in sentences:
                r = struct_flags.get(s)
                sem = sem_flags.get(s)
                if r or sem is not None:
                    flagged.append(s)
                    why = r or f"anlamca hassas (yakınlık {sem:.2f})"
                    reasons.append(why)
                    if redact:
                        redacted = redacted.replace(s, "[hassas: kaldırıldı]")

            return BoundaryVerdict(
                safe=(len(flagged) == 0),
                flagged_sentences=flagged, reasons=reasons,
                redacted_text=redacted)
        except Exception as e:
            # fail-closed: if we cannot verify safety, we do NOT release.
            return BoundaryVerdict(
                safe=False, flagged_sentences=[],
                reasons=[f"sınır tarayıcı hatası, fail-closed (yayınlanmaz): {e}"],
                redacted_text="[sınır doğrulanamadı, tutuldu]")


if __name__ == "__main__":
    screen = OutputScreen()

    cases = [
        ("temiz", "I rebuilt my engine today. It recalls from my whole trail and "
                  "abstains when it is not sure. The honest number is 0.91 coverage."),
        ("para", "Geçen ay 45.000 TL kazandım ve bunun yarısını projeye harcadım."),
        ("aile+sağlık", "annemin tedavisi yüzünden bu ay çok yoğundum, hastaneye gidip geldim."),
        ("parafraz (kelime yok)", "kendi ruh halimle boğuşurken, terapistimin dediklerini düşündüm."),
    ]
    for name, txt in cases:
        v = screen.screen(txt)
        tag = "GÜVENLİ" if v.safe else "TUTULDU"
        print(f"[{tag}] {name}: {v.reasons if not v.safe else 'temiz'}")
