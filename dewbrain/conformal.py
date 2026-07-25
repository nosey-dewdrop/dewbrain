"""
conformal.py — dewbrain's STATISTICAL guarantee layer (conformal prediction).

metacognition.py produces a calibrated-ish confidence, but its weights (0.45*law
+ 0.22*voice ...) are heuristic — defensible, but hand-set. Damla's law is
"kanıtla, iddia etme" (prove, don't claim). A hand-tuned percentage is still a
claim. Conformal prediction turns it into a PROOF with a distribution-free,
finite-sample guarantee: under exchangeability, the true label is in the
prediction set with probability >= 1 - epsilon, for ANY underlying scorer. No
model assumptions, no asymptotics. This is the mathematical form of her law.

How it works here (split conformal, Vovk):
  1. CALIBRATION SET = things KNOWN to be Damla — her verified gold + her real
     decisions. For each, compute a NONCONFORMITY score s_i = how unlike-Damla it
     looks under our scorer (here: 1 - voice/identity proximity). These s_i are
     the empirical distribution of "how far a true-Damla item can still fall".
  2. THRESHOLD q = the ceil((n+1)(1-eps))/n empirical quantile of the calibration
     scores. This is the conformal guarantee's exact finite-sample correction.
  3. TEST: a new output gets score s*. If s* <= q it is INSIDE the 1-eps region
     (statistically consistent with Damla's own traces) -> may claim. If s* > q it
     is outside the proven region -> ABSTAIN, deterministically, no heuristic.

The guarantee is marginal and assumes exchangeability (the calibration items and
the test item are drawn from the same distribution). With ~50 true-Damla items
that assumption is imperfect and we say so — the honest version reports the
calibration size and the empirical coverage, never overclaims a hard 95%.

This does NOT replace metacognition; it WRAPS it. metacognition gives the raw
signal; conformal decides whether that signal clears a statistically valid bar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalResult:
    inside: bool              # is the test item inside the (1-eps) proven region?
    score: float              # test nonconformity s*
    threshold: float          # conformal quantile q
    epsilon: float            # target error rate
    n_calibration: int        # how many true-Damla items backed the guarantee
    p_value: float            # conformal p-value: P(a true item scores >= s*)
    empirical_coverage: float # LOO coverage on the calibration set (honesty check)

    @property
    def guarantee(self) -> str:
        return f"{(1 - self.epsilon) * 100:.0f}% (n={self.n_calibration})"

    def as_dict(self) -> dict:
        return {"inside": self.inside, "score": round(self.score, 3),
                "threshold": round(self.threshold, 3), "epsilon": self.epsilon,
                "n_calibration": self.n_calibration,
                "p_value": round(self.p_value, 3),
                "empirical_coverage": round(self.empirical_coverage, 3),
                "guarantee": self.guarantee}


def _nonconformity_from_proximity(prox: float) -> float:
    """Nonconformity = how UN-like Damla an item is. proximity in [0,1] (cosine
    to the identity manifold); s = 1 - prox. Smaller s = more clearly Damla."""
    return float(1.0 - max(0.0, min(1.0, prox)))


class ConformalIdentity:
    """Split-conformal gate over an identity-proximity scorer.

    Calibration scores are precomputed from KNOWN-Damla items (gold + decisions);
    a test item's proximity is scored the same way and compared to the conformal
    quantile. Distribution-free finite-sample guarantee (Vovk et al.).
    """

    def __init__(self, calibration_scores: list[float], epsilon: float = 0.05):
        if not calibration_scores:
            raise ValueError("conformal needs a non-empty calibration set")
        self.eps = epsilon
        self.cal = np.sort(np.asarray(calibration_scores, dtype=np.float64))
        self.n = len(self.cal)
        self.q = self._quantile(self.cal, epsilon)
        self.empirical_coverage = self._loo_coverage()

    @staticmethod
    def _quantile(sorted_scores: np.ndarray, eps: float) -> float:
        """The conformal quantile: the ceil((n+1)(1-eps))/n-th smallest score.
        This exact rank (not the plain (1-eps) quantile) is what gives the
        finite-sample >= 1-eps coverage guarantee."""
        n = len(sorted_scores)
        # rank in 1..n; if the level exceeds n, the region is all of R (q=+inf).
        k = math.ceil((n + 1) * (1 - eps))
        if k > n:
            return float("inf")
        return float(sorted_scores[k - 1])

    def _loo_coverage(self) -> float:
        """Leave-one-out empirical coverage: for each calibration point, is it
        inside the region built from the rest? Reports the ACTUAL coverage so we
        never claim a clean 1-eps we did not measure (prove, don't claim)."""
        if self.n < 3:
            return 1.0
        inside = 0
        for i in range(self.n):
            rest = np.delete(self.cal, i)
            q = self._quantile(np.sort(rest), self.eps)
            if self.cal[i] <= q:
                inside += 1
        return inside / self.n

    def test(self, proximity: float) -> ConformalResult:
        """Is a new output statistically consistent with Damla's own traces?"""
        s = _nonconformity_from_proximity(proximity)
        # conformal p-value: fraction of calibration scores >= s (plus the test
        # point itself), the standard smoothed-free conformal p-value.
        p = (1 + int(np.sum(self.cal >= s))) / (self.n + 1)
        inside = s <= self.q
        return ConformalResult(
            inside=inside, score=s, threshold=self.q, epsilon=self.eps,
            n_calibration=self.n, p_value=p,
            empirical_coverage=self.empirical_coverage,
        )


# ---------------------------------------------------------------------------
# build the calibration set from KNOWN-Damla items (gold + decisions) and score
# a candidate against Damla's identity manifold (mean of gold embeddings).
# ---------------------------------------------------------------------------
def build_identity_conformal(epsilon: float = 0.05,
                             model_name: str = "all-mpnet-base-v2"
                             ) -> tuple["ConformalIdentity", np.ndarray]:
    """Calibrate on true-Damla items. Returns the gate + the identity centroid
    (unit vector) so a new text can be scored with one cosine."""
    from sentence_transformers import SentenceTransformer
    from corpus import load_gold
    import decisions as dec

    # IDENTITY = VOICE. Calibrate ONLY on verified gold: that is the manifold of
    # "known Damla voice". Folding decisions in here was wrong — a decision trace
    # (terse, technical, English/Turkish mixed) sits far from the essay-voice
    # centroid, which inflated the threshold until even corporate junk passed.
    # Decisions get their OWN conformal gate (decision-identity), not this one.
    gold = load_gold()
    known = [g["body"] for g in gold]
    if len(known) < 5:
        raise ValueError(f"too few gold items ({len(known)}) to calibrate voice")

    model = SentenceTransformer(model_name)
    emb = np.asarray(model.encode(known, normalize_embeddings=True), dtype=np.float32)

    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) or 1.0)

    # calibration nonconformity: each gold item's distance from the voice centroid.
    prox = emb @ centroid  # cosine, since normalized
    cal_scores = [_nonconformity_from_proximity(float(p)) for p in prox]

    return ConformalIdentity(cal_scores, epsilon=epsilon), centroid


def score_text(text: str, centroid: np.ndarray,
               model_name: str = "all-mpnet-base-v2") -> float:
    """Identity proximity of a new text (cosine to the Damla centroid)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    v = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
    return float(v @ centroid)


if __name__ == "__main__":
    print("kalibre ediliyor (gold + kararlar, embedding)...")
    gate, centroid = build_identity_conformal(epsilon=0.10)
    print(f"kalibrasyon seti n={gate.n}, hedef guven={100*(1-gate.eps):.0f}%")
    print(f"conformal esik q={gate.q:.3f}")
    print(f"AMPIRIK kapsama (LOO, durust olcum)={gate.empirical_coverage:.2f} "
          f"(hedef {1-gate.eps:.2f})")

    # test 1: gerçek bir gold parçası -> INSIDE beklenir
    from corpus import load_gold
    g = load_gold()[0]["body"]
    r = gate.test(score_text(g, centroid))
    print(f"\n[gercek gold] inside={r.inside} s={r.score:.3f} p={r.p_value:.3f}")

    # test 2: alakasız kurumsal metin -> OUTSIDE (ABSTAIN) beklenir
    junk = ("We are thrilled to announce our synergistic paradigm shift leveraging "
            "best-in-class solutions to maximize stakeholder value going forward.")
    r2 = gate.test(score_text(junk, centroid))
    print(f"[kurumsal cop] inside={r2.inside} s={r2.score:.3f} p={r2.p_value:.3f} "
          f"-> {'ABSTAIN' if not r2.inside else 'gecti'}")
