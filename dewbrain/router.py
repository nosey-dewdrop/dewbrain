"""
router.py — dewbrain's DUAL-PROCESS controller (System 1 / System 2).

Damla's ask: stop being a router, be an orchestrator. The old brain.run ran
EVERY cue through the identical fixed pipeline (recall -> generate -> critique
-> repair, same max_iter for all). A real brain does not. It routes: a familiar,
low-stakes cue goes the fast path (System 1: few passes, trust the anchor); a
novel or high-stakes cue goes the slow path (System 2: more critique rounds,
higher bar to speak, more context pulled). Kahneman's dual-process, and the
BRAIN_MAP "alışkanlık/dual-process" layer.

This decides the CONTROL KNOBS before generation, from signals available up
front (no API spend to decide):
  - familiarity: how cleanly the cue matches past traces (retrieval top-mass +
    context strength). High = seen this before = fast path.
  - stakes: what KIND of cue it is. A decision/idea ("Damla ne yapmalı") is
    higher-stakes than a writing cue — a wrong decision costs more than a wrong
    sentence, so it earns more scrutiny and a higher abstain floor.
  - novelty: low familiarity => slow path even if low stakes (unknown terrain).

Output is a Route: max_iter, k (how many traces to recall), and confidence
thresholds tuned for this cue. brain.run consumes it instead of raw config.
The point: same engine, DIFFERENT effort matched to the cue — orchestration,
not one-size pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Route:
    path: str            # "fast" (System 1) | "slow" (System 2)
    max_iter: int        # critique->repair rounds allowed
    k: int               # traces to recall (context width)
    abstain_floor: float # confidence below this -> abstain (raised for high stakes)
    sure_bar: float      # confidence at/above this -> may claim "this is Damla"
    reason: str          # human-readable why this route

    def as_dict(self) -> dict:
        return {"path": self.path, "max_iter": self.max_iter, "k": self.k,
                "abstain_floor": round(self.abstain_floor, 2),
                "sure_bar": round(self.sure_bar, 2), "reason": self.reason}


# stakes by cue channel/kind. A decision or idea is a higher-consequence output
# than a piece of writing: getting "what would Damla do" wrong is worse than a
# clumsy sentence, so it earns System 2 and a higher bar to speak.
_STAKES = {
    "decision": 0.9,
    "idea": 0.8,
    "report": 0.6,
    "projectdoc": 0.5,
    "writing": 0.4,
    "": 0.5,
}


def _familiarity(gold_hits, context_hits) -> float:
    """0..1 how strongly this cue resembles past traces. High top-mass in gold
    (clean voice anchor) + strong context recall = 'I have been here before'."""
    voice = max((h.get("weight", 0.0) for h in (gold_hits or [])), default=0.0)
    ctx = max((h.get("score", 0.0) for h in (context_hits or [])), default=0.0)
    ctx = min(1.0, ctx / 0.6)  # soft-normalize context score to 0..1
    # voice fit dominates familiarity (it is the 'have I said this' signal),
    # context grounds it.
    return 0.65 * voice + 0.35 * ctx


def route(
    cue_channel: str,
    gold_hits: list[dict] | None,
    context_hits: list[dict] | None,
    *,
    base_max_iter: int = 4,
    base_k: int = 3,
) -> Route:
    """Pick fast vs slow path and its knobs from up-front signals.

    Fast path (System 1): familiar + low stakes -> fewer rounds, standard bar.
    Slow path (System 2): novel OR high stakes -> more rounds, more context,
    higher abstain floor (be more willing to say 'bilmiyorum' when it matters).
    """
    fam = _familiarity(gold_hits, context_hits)
    stakes = _STAKES.get(cue_channel, 0.5)

    # decision boundary: high familiarity AND low stakes -> fast. Anything novel
    # (fam < 0.5) or consequential (stakes >= 0.7) -> slow.
    go_fast = fam >= 0.6 and stakes < 0.7

    if go_fast:
        return Route(
            path="fast",
            max_iter=max(2, base_max_iter - 2),      # trust the anchor, fewer rounds
            k=base_k,
            abstain_floor=0.45,
            sure_bar=0.78,
            reason=(f"tanıdık (fam={fam:.2f}) + düşük risk (stakes={stakes:.1f}) "
                    f"-> hızlı yol, az tur"),
        )

    # slow path — scale scrutiny with how novel/consequential it is.
    extra = 1 if (fam < 0.4 or stakes >= 0.85) else 0
    return Route(
        path="slow",
        max_iter=base_max_iter + extra,              # more critique rounds
        k=base_k + 2,                                # pull wider context
        abstain_floor=0.45 + 0.15 * stakes,          # higher stakes = quicker to abstain
        sure_bar=0.78 + 0.07 * stakes,               # higher stakes = harder to claim
        reason=(f"{'yeni' if fam < 0.5 else 'tanıdık'} (fam={fam:.2f}) / "
                f"{'yüksek' if stakes >= 0.7 else 'orta'} risk (stakes={stakes:.1f}) "
                f"-> yavaş yol, derin eleştiri"),
    )


if __name__ == "__main__":
    strong = [{"weight": 0.9, "score": 0.8}]
    weak = [{"weight": 0.3, "score": 0.2}]

    print("--- yazı, tanıdık -> beklenen FAST ---")
    print(" ", route("writing", strong, strong).as_dict())

    print("\n--- karar, tanıdık -> beklenen SLOW (yüksek risk) ---")
    print(" ", route("decision", strong, strong).as_dict())

    print("\n--- yazı, yeni -> beklenen SLOW (novelty) ---")
    print(" ", route("writing", weak, weak).as_dict())

    print("\n--- fikir, yeni -> beklenen SLOW+extra (yeni + yüksek risk) ---")
    print(" ", route("idea", weak, weak).as_dict())
