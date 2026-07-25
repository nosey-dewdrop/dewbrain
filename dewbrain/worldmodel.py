"""
worldmodel.py — dewbrain's DECISION model ("what would Damla do", honestly).

This is the layer Damla actually wants: not a writing tool, a decision core that
reflects her own pattern back at her. It is NOT a fortune teller. A fortune teller
invents a probability with no ground ("go on this date and you'll burn out 44%").
That is exactly the fabricated-certainty Damla's law forbids. This does the
opposite: it grounds every statement in her REAL past decisions and REFUSES to
speak when the past does not support the present.

At a decision threshold ("should I apply to X", "should I kill this project"),
the model:
  1. RECALLS the most similar past decisions from the whole trail (1397 traces,
     42 explicit DECISIONS carrying choice/reason/reversal-cost/outcome).
  2. REPORTS what she actually chose, why, at what reversal cost, and how it went.
  3. GATES with conformal prediction: is this new situation statistically
     consistent with her decision manifold? If the nearest matches are too far
     (outside the 1-eps region), it ABSTAINS: "you have no comparable past here,
     I will not guide you" — the honest failure, not a made-up number.
  4. Only when grounded does it state the PATTERN: "in N similar situations you
     chose X, K of them were cheap-to-reverse" — real counts, not predictions.

The honesty rule (Damla, 25 Tem, explicit: "fal değil"): every number is a COUNT
over real traces (how many similar, how many reversible), never a probability
pulled from thin air. With 42 decisions the sample is small and the model SAYS
so. A world-model that oversells its confidence at n=42 is a fortune teller with
a math costume. This one carries its own error bar and stays silent past it.

MCTS / multi-step rollout (the external suggestion) is deliberately NOT built
here: it needs a learned reward over far more than 42 labelled decisions, and
faking that reward is the fortune-teller trap. This is the honest baseline it
would sit on top of once the decision corpus is large enough (Faz 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimilarDecision:
    """A past decision recalled as precedent for the current one."""
    title: str
    choice: str
    reason: str
    cost: str            # reversal cost / effect
    outcome: str
    project: str
    similarity: float

    def as_dict(self) -> dict:
        return {"title": self.title, "choice": self.choice[:120],
                "reason": self.reason[:120], "cost": self.cost[:80],
                "similarity": round(self.similarity, 3), "project": self.project}


@dataclass
class DecisionReflection:
    """The model's honest reflection on a decision cue. Either grounded pattern
    or an explicit abstain — never an invented probability."""
    cue: str
    abstained: bool
    reason_line: str               # one honest sentence (pattern or why abstained)
    precedents: list[SimilarDecision]
    n_similar: int                 # real count of comparable past decisions
    n_reversible: int              # of those, how many were cheap-to-reverse
    conformal: dict = field(default_factory=dict)  # the statistical gate result

    def as_dict(self) -> dict:
        return {
            "cue": self.cue[:100],
            "abstained": self.abstained,
            "reason": self.reason_line,
            "n_similar": self.n_similar,
            "n_reversible": self.n_reversible,
            "precedents": [p.as_dict() for p in self.precedents],
            "conformal": self.conformal,
        }


_REVERSIBLE_HINTS = ("ucuz", "düşük", "dusuk", "cheap", "low", "kolay", "revert")


def _is_reversible(cost: str) -> bool:
    c = (cost or "").lower()
    return any(h in c for h in _REVERSIBLE_HINTS)


class WorldModel:
    """Honest decision reflection over Damla's real decision history.

    Recall precedents, gate with conformal (is this situation in-distribution for
    her decisions?), and report a grounded pattern or abstain. No rollout, no
    invented probabilities — counts over real traces plus a statistical bar.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2", epsilon: float = 0.1):
        from sentence_transformers import SentenceTransformer
        import decisions as dec

        self.decs = dec.load_decisions()
        if len(self.decs) < 5:
            raise ValueError(f"too few decisions ({len(self.decs)}) for a world-model")
        self.model = SentenceTransformer(model_name)
        bodies = [d.as_memory_body() for d in self.decs]
        self.emb = np.asarray(self.model.encode(bodies, normalize_embeddings=True),
                              dtype=np.float32)
        self._sim_floor = 0.30  # a precedent must clear this to count as comparable

        # conformal calibration on a kNN manifold score, NOT centroid distance.
        # Root cause found (25 Tem): centroid proximity measures "is this a
        # decision-shaped question", not "is this Damla's kind of problem" — the
        # sentence embedding latches onto the 'X mi yoksa Y mi' FORM, so an
        # off-domain personal cue ("tatile mi çıksam") scored as close as a real
        # technical one. Fix: score a point by its similarity to its nearest REAL
        # decisions (kNN), and calibrate leave-one-out. An off-domain cue is far
        # from every real decision even if it is decision-shaped.
        self._knn = 3
        from conformal import ConformalIdentity, _nonconformity_from_proximity
        cal = [_nonconformity_from_proximity(self._knn_proximity(i))
               for i in range(len(self.decs))]
        self.gate = ConformalIdentity(cal, epsilon=epsilon)
        self._nonconf = _nonconformity_from_proximity

    def _knn_proximity(self, leave_out: int) -> float:
        """Mean similarity of decision `leave_out` to its k nearest OTHER
        decisions. Leave-one-out so calibration is honest (a point never counts
        itself). This is 'how well-supported is this point by real neighbours'."""
        sims = self.emb @ self.emb[leave_out]
        sims[leave_out] = -1.0  # drop self
        top = np.sort(sims)[-self._knn:]
        return float(np.mean(top))

    def _cue_proximity(self, q: np.ndarray) -> float:
        """kNN proximity of a query to the real decisions (mean of top-k sims)."""
        sims = self.emb @ q
        top = np.sort(sims)[-self._knn:]
        return float(np.mean(top))

    def reflect(self, cue: str, k: int = 4) -> DecisionReflection:
        """Reflect on a decision cue: grounded precedent pattern, or abstain."""
        q = np.asarray(self.model.encode([cue], normalize_embeddings=True)[0],
                       dtype=np.float32)

        # conformal gate: is this cue supported by REAL nearest decisions (kNN),
        # not just decision-shaped? An off-domain cue is far from every real one.
        prox_to_manifold = self._cue_proximity(q)
        cres = self.gate.test(prox_to_manifold)

        sims = self.emb @ q
        order = np.argsort(-sims)[:k]
        precedents = []
        for i in order:
            d = self.decs[int(i)]
            precedents.append(SimilarDecision(
                title=d.title, choice=d.choice, reason=d.reason, cost=d.cost,
                outcome=d.outcome, project=d.project, similarity=float(sims[i])))

        # count real precedents that clear a similarity floor (comparable, not
        # just top-k). This is the COUNT the model reports, not a probability.
        comparable = [p for p in precedents if p.similarity >= 0.25]
        n_similar = len(comparable)
        n_reversible = sum(1 for p in comparable if _is_reversible(p.cost))

        if not cres.inside:
            # outside the decision manifold -> no comparable past -> abstain.
            return DecisionReflection(
                cue=cue, abstained=True,
                reason_line=(
                    f"çekimserim. bu karar geçmiş kararlarınla istatistiksel olarak "
                    f"yeterince örtüşmüyor (conformal p={cres.p_value:.2f}, eşik "
                    f"{cres.epsilon:.2f}). karşılaştırılabilir bir geçmişin yok, "
                    f"kendi örüntünle yol gösteremem."),
                precedents=precedents, n_similar=n_similar,
                n_reversible=n_reversible, conformal=cres.as_dict())

        if n_similar == 0:
            return DecisionReflection(
                cue=cue, abstained=True,
                reason_line=("çekimserim. konu tanıdık ama yakın bir emsal karar "
                             "bulamadım (en yakın benzerlik düşük). uydurmam."),
                precedents=precedents, n_similar=0, n_reversible=0,
                conformal=cres.as_dict())

        # grounded pattern — real counts, no invented probability.
        top = comparable[0]
        rev_note = (f"{n_reversible}/{n_similar}'i geri alması ucuzdu"
                    if n_similar else "")
        line = (f"geçmişte {n_similar} benzer kararında örüntün şu. en yakını "
                f"({top.project}) '{top.choice[:80]}' dedin"
                f"{', çünkü ' + top.reason[:80] if top.reason else ''}. "
                f"{rev_note}. bu bir tahmin değil, kendi kararlarının sayımı "
                f"(n={n_similar} az, kesinlik iddia etmiyorum).")
        return DecisionReflection(
            cue=cue, abstained=False, reason_line=line, precedents=precedents,
            n_similar=n_similar, n_reversible=n_reversible,
            conformal=cres.as_dict())


if __name__ == "__main__":
    print("world-model kuruluyor (kararlar + conformal)...")
    wm = WorldModel(epsilon=0.15)
    print(f"{len(wm.decs)} karar üzerine kuruldu.\n")

    tests = [
        "yeni bir özellik canlı diye yazayım mı yoksa deploy edilene kadar bekleyeyim mi",
        "bu pinli stile dokunayım mı, golden bozulur mu",
        "sevgilimle tatile mi çıksam yoksa çalışsam mı",  # alan dışı -> ABSTAIN beklenir
    ]
    for t in tests:
        r = wm.reflect(t)
        tag = "ÇEKİMSER" if r.abstained else "ÖRÜNTÜ"
        print(f"[{tag}] cue: {t[:55]}")
        print(f"  {r.reason_line}")
        if r.precedents and not r.abstained:
            p = r.precedents[0]
            print(f"  en yakın emsal: [{p.project}] sim={p.similarity:.2f} {p.choice[:50]}")
        print()
