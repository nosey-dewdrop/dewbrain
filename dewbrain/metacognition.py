"""
metacognition.py — dewbrain's CONFIDENCE + ABSTAIN layer.

Damla's sharpest law is "kanıtla, iddia etme" (prove, don't claim). Her single
most-hated failure is a system that says "bu sensin / this is done / ship-ready"
with no error bar. BRAIN_MAP flagged this as the KRİTİK missing layer: the loop's
`accepted = best_fails == 0` is a bare first-order verdict. It can only say
clean/not-clean. It cannot say "I am 70% sure this is you" or "I don't know —
abstaining." A brain does the second thing. Without it, dewbrain does the exact
thing Damla hates: asserts identity with no calibration.

This layer turns the loop's raw signals into a CALIBRATED confidence and, below
a floor, an ABSTAIN (the model's right to say "bilmiyorum"). It reads only
signals already produced — no extra API call, deterministic, auditable.

Signals (each a real uncertainty source, not a vibe):
  1. law pass    — weighted rubric pass rate (0 violations != full confidence if
                   it barely passed; weighted so an em-dash fail hurts more).
  2. voice fit   — how cleanly gold separated for this cue (retrieval top-mass).
                   A blended recall (two golds at 0.5/0.5) means the voice anchor
                   was ambiguous -> lower confidence in "this is Damla".
  3. grounding   — did episodic context actually support this cue, or was the
                   output ungrounded (invention risk = the fabrication law).
  4. effort      — how many repair rounds it took. Clean on round 1 is confident;
                   scraping clean on the last allowed round is fragile.
  5. self-judge  — the critic's holistic damla_kokuyor, used only as a NUDGE
                   (the model judging itself must never certify identity; that is
                   Damla's call, mirrored from memory.consolidate's guard).

Output is a Confidence(score 0..1, verdict: sure|tentative|abstain, reasons).
The gate to call something "bu sensin" (identity claim) is DELIBERATELY high and
still ultimately Damla's — this layer decides when the brain is allowed to SPEAK
vs must ABSTAIN, not when a draft becomes gold.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# thresholds — a brain that over-claims is worse than one that abstains, so the
# abstain floor is generous and the "sure" bar is high (prove, don't claim).
ABSTAIN_FLOOR = 0.45      # below this the brain says "bilmiyorum" and holds
SURE_BAR = 0.78           # at/above this it may state "this reads like Damla"


@dataclass
class Confidence:
    score: float                       # 0..1 calibrated
    verdict: str                       # "sure" | "tentative" | "abstain"
    components: dict = field(default_factory=dict)  # each signal's contribution
    reasons: list[str] = field(default_factory=list)  # human-readable why
    weakest: str = ""                  # the signal dragging confidence down

    @property
    def abstains(self) -> bool:
        return self.verdict == "abstain"

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "verdict": self.verdict,
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "reasons": self.reasons,
            "weakest": self.weakest,
        }


def _weighted_pass_rate(verdict: dict, weights: dict | None) -> float:
    """Fraction of rubric WEIGHT that passed (not raw item count). A heavy fail
    (em-dash w=3) costs more than a light one, matching the law's own weights."""
    items = (verdict or {}).get("items") or []
    if not items:
        return 0.0
    w = weights or {}
    total = sum(w.get(i.get("key"), 1.0) for i in items)
    if total == 0:
        return 0.0
    got = sum(w.get(i.get("key"), 1.0) for i in items if i.get("pass"))
    return got / total


def assess(
    verdict: dict,
    *,
    gold_hits: list[dict] | None = None,
    context_hits: list[dict] | None = None,
    iterations: int = 1,
    max_iter: int = 4,
    weights: dict | None = None,
    abstain_floor: float = ABSTAIN_FLOOR,
    sure_bar: float = SURE_BAR,
) -> Confidence:
    """Fuse the loop's signals into a calibrated confidence + speak/abstain call.

    Pure function of already-computed signals: no API, deterministic, so the same
    run always yields the same confidence (auditable). Each component is in 0..1;
    the final score is a weighted blend, then the self-judge nudges within a cap.
    """
    comp: dict[str, float] = {}
    reasons: list[str] = []

    # 1. law pass (weighted). The dominant signal — voice purity is the law.
    law = _weighted_pass_rate(verdict, weights)
    comp["law"] = law
    if law < 1.0:
        reasons.append(f"yasa {law:.0%} ağırlıklı geçti (bir ihlal kaldı)")

    # 2. voice fit — retrieval top-mass. Blended recall => ambiguous voice anchor.
    if gold_hits:
        top = max((h.get("weight", 0.0) for h in gold_hits), default=0.0)
        comp["voice_fit"] = top
        if top < 0.6:
            reasons.append(f"ses çapası bulanık (gold {top:.2f} mass, blend riski)")
    else:
        comp["voice_fit"] = 0.4  # no gold recalled = weak anchor, not zero
        reasons.append("ses çapası yok (gold getirilemedi)")

    # 3. grounding — did episodic context back this cue? Ungrounded = invention risk.
    if context_hits:
        gtop = max((h.get("score", 0.0) for h in context_hits), default=0.0)
        # scores are unbounded-ish; squash to 0..1 with a soft knee at ~0.5.
        comp["grounding"] = min(1.0, gtop / 0.5)
        if comp["grounding"] < 0.5:
            reasons.append("geçmiş bağlam zayıf (uydurma riski)")
    else:
        comp["grounding"] = 0.5  # neutral: absence of context isn't a strong signal
        reasons.append("geçmiş bağlam getirilmedi (nötr)")

    # 4. effort — clean early = confident; scraping clean late = fragile.
    effort = 1.0 - (max(0, iterations - 1) / max(1, max_iter))
    comp["effort"] = effort
    if iterations >= max_iter:
        reasons.append(f"son turda ({iterations}) zar zor temizlendi (kırılgan)")

    # weighted blend. law dominates (it IS the voice law); the rest calibrate.
    score = (0.45 * comp["law"]
             + 0.22 * comp["voice_fit"]
             + 0.18 * comp["grounding"]
             + 0.15 * comp["effort"])

    # 5. self-judge NUDGE only. The critic saying "damla_kokuyor" cannot certify
    # identity (that is Damla's call, mirrored from memory.consolidate's guard);
    # it may nudge +-, capped, never floor/ceiling the score.
    smells = (verdict or {}).get("damla_kokuyor")
    if smells is True:
        score = min(1.0, score + 0.03)
    elif smells is False:
        score = max(0.0, score - 0.08)
        reasons.append("iç denetçi 'Damla kokmuyor' dedi")

    weakest = min(comp, key=comp.get) if comp else ""

    # hard rule (prove, don't claim): an identity claim ("bu sensin") requires
    # ZERO law violations. Any remaining violation caps the verdict at tentative
    # no matter how high the blended score — you cannot be "sure it's Damla" while
    # a rule she never breaks is still broken. This is the law as a gate, not a
    # score, mirroring prefrontal.Verdict.clean.
    any_fail = any(not i.get("pass", False) for i in (verdict or {}).get("items", []))

    if score < abstain_floor:
        verdict_label = "abstain"
        reasons.insert(0, "güven eşiğin altında — çekimser (bilmiyorum, iddia etmiyorum)")
    elif score >= sure_bar and not any_fail:
        verdict_label = "sure"
    else:
        verdict_label = "tentative"
        if score >= sure_bar and any_fail:
            reasons.insert(0, "skor yüksek ama bir ihlal var — 'sen' demiyorum (sıfır-ihlal şart)")

    return Confidence(score=score, verdict=verdict_label,
                      components=comp, reasons=reasons, weakest=weakest)


def confidence_line(c: Confidence) -> str:
    """One honest human line — the brain stating its own certainty out loud.
    This is the opposite of the failure Damla hates: it always carries the bar."""
    if c.abstains:
        return (f"çekimserim (güven {c.score:.0%}). Bunun sen olduğunu iddia "
                f"etmiyorum; en zayıf sinyal: {c.weakest}.")
    if c.verdict == "sure":
        return f"bu büyük olasılıkla sen (güven {c.score:.0%})."
    return (f"olabilir ama emin değilim (güven {c.score:.0%}); "
            f"en zayıf sinyal: {c.weakest}.")


if __name__ == "__main__":
    # dry-run: three synthetic verdicts show sure / tentative / abstain.
    clean = {"items": [{"key": f"k{i}", "pass": True} for i in range(16)],
             "damla_kokuyor": True}
    one_fail = {"items": [{"key": f"k{i}", "pass": i != 0} for i in range(16)],
                "damla_kokuyor": True}
    messy = {"items": [{"key": f"k{i}", "pass": i % 3 != 0} for i in range(16)],
             "damla_kokuyor": False}

    strong_gold = [{"weight": 0.9, "score": 0.8}]
    blended_gold = [{"weight": 0.5, "score": 0.4}]
    ctx = [{"score": 0.9}]

    print("--- SURE (temiz, güçlü çapa, 1 tur) ---")
    c = assess(clean, gold_hits=strong_gold, context_hits=ctx, iterations=1)
    print(" ", c.as_dict()); print(" ", confidence_line(c))

    print("\n--- TENTATIVE (1 ihlal, bulanık çapa, 3 tur) ---")
    c = assess(one_fail, gold_hits=blended_gold, context_hits=ctx, iterations=3)
    print(" ", c.as_dict()); print(" ", confidence_line(c))

    print("\n--- ABSTAIN (dağınık, çapa yok, son tur) ---")
    c = assess(messy, gold_hits=None, context_hits=None, iterations=4, max_iter=4)
    print(" ", c.as_dict()); print(" ", confidence_line(c))
