"""
consolidation.py — dewbrain's SLEEP layer (episodic -> semantic distillation).

The brain does its real learning offline. During deep sleep the hippocampus
replays the day's episodic traces and the neocortex extracts the STABLE pattern,
dropping incidental detail (brain-architecture.md: consolidation). dewbrain's
gold bottleneck (9 -> 100) is exactly this step missing: 1219 raw traces sit
there, but nothing distills them into what the brain KNOWS about Damla.

This layer runs that offline pass over the existing trail — WITHOUT asking Damla
to write or approve anything (her explicit constraint: this is the brain's job,
not homework). It is deterministic, needs no API (embeddings already local), and
respects the voice-safety law: it never promotes anything into voice-gold on its
own (memory.consolidate's guard — model self-certifying identity collapses the
voice). It produces two things:

  1. NEOCORTEX PATTERNS — clusters of similar traces across the trail. A tight
     cluster of many traces = a stable, repeated Damla pattern (how she reasons
     about a recurring theme). This is semantic knowledge the brain can hold,
     distinct from voice-gold. Not shown as "you", shown as "a pattern in you".

  2. GOLD CANDIDATES — the raw WRITING traces closest to the existing gold
     geometry AND law-clean, ranked. These are the traces most likely to BE
     Damla-voice already. The brain surfaces a short ranked list; the "bu ben"
     stamp stays Damla's (candidate != gold). This turns 9->100 into a review
     queue the brain fills, not a writing task Damla owns.

Sleep does not fabricate gold. It distills what already exists and hands Damla a
ranked shortlist instead of 1219 raw traces to wade through.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    """A neocortex pattern: many similar traces = a stable repeated theme."""
    size: int
    channels: dict          # channel -> count within the cluster
    exemplar_title: str     # the trace nearest the cluster centroid
    members: list[str]      # member titles
    cohesion: float         # mean cosine to centroid (tightness)

    def as_dict(self) -> dict:
        return {"size": self.size, "channels": self.channels,
                "exemplar": self.exemplar_title,
                "cohesion": round(self.cohesion, 3),
                "members": self.members[:8]}


@dataclass
class GoldCandidate:
    """A raw writing trace ranked by how close it already is to Damla-voice gold."""
    title: str
    channel: str
    voice_proximity: float  # cosine to nearest gold (voice geometry)
    salience: float         # substance/length prior
    score: float            # combined rank
    body_preview: str

    def as_dict(self) -> dict:
        return {"title": self.title, "channel": self.channel,
                "voice_proximity": round(self.voice_proximity, 3),
                "salience": round(self.salience, 3),
                "score": round(self.score, 3),
                "preview": self.body_preview[:120]}


def _embed(texts: list[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. neocortex patterns — greedy agglomerative clustering over the whole trail.
#    No sklearn dependency: a simple cosine-threshold single-pass grouping is
#    enough to surface stable repeated themes (and stays deterministic).
# ---------------------------------------------------------------------------
def find_patterns(traces: list[dict], sim_threshold: float = 0.55,
                  min_size: int = 3, model_name: str = "all-mpnet-base-v2"
                  ) -> list[Cluster]:
    """Group traces into cohesion clusters. A cluster with >= min_size members
    is a repeated pattern the neocortex can hold as semantic knowledge."""
    if len(traces) < min_size:
        return []
    emb = _embed([t["body"] for t in traces], model_name)
    n = len(traces)
    assigned = [-1] * n
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []

    # greedy: each trace joins the nearest centroid above threshold, else opens one.
    order = sorted(range(n), key=lambda i: -len(traces[i]["body"]))  # seed with rich traces
    for i in order:
        v = emb[i]
        best_c, best_s = -1, sim_threshold
        for ci, c in enumerate(centroids):
            s = float(v @ c)
            if s > best_s:
                best_c, best_s = ci, s
        if best_c == -1:
            centroids.append(v.copy())
            members.append([i])
            assigned[i] = len(centroids) - 1
        else:
            members[best_c].append(i)
            m = members[best_c]
            centroids[best_c] = emb[m].mean(axis=0)  # update centroid
            centroids[best_c] /= (np.linalg.norm(centroids[best_c]) or 1.0)
            assigned[i] = best_c

    clusters: list[Cluster] = []
    for ci, mem in enumerate(members):
        if len(mem) < min_size:
            continue
        c = centroids[ci]
        sims = [float(emb[j] @ c) for j in mem]
        cohesion = float(np.mean(sims))
        exemplar = mem[int(np.argmax(sims))]
        chans: dict = {}
        for j in mem:
            ch = traces[j].get("channel", "?")
            chans[ch] = chans.get(ch, 0) + 1
        clusters.append(Cluster(
            size=len(mem), channels=chans,
            exemplar_title=traces[exemplar]["title"],
            members=[traces[j]["title"] for j in mem],
            cohesion=cohesion,
        ))
    clusters.sort(key=lambda c: (-c.size, -c.cohesion))
    return clusters


# ---------------------------------------------------------------------------
# 2. gold candidates — which raw WRITING traces already look like voice-gold.
#    voice_proximity = cosine to nearest existing gold; salience = substance.
#    NOTE: only the writing channel is eligible (gold = voice; a report/decision
#    is not voice-gold material). The stamp stays Damla's — this only RANKS.
# ---------------------------------------------------------------------------
def rank_gold_candidates(traces: list[dict], gold: list[dict],
                         top: int = 15, model_name: str = "all-mpnet-base-v2"
                         ) -> list[GoldCandidate]:
    """Rank raw writing traces by proximity to the existing gold voice geometry.
    Surfaces the shortlist most likely to already be Damla — Damla approves; the
    brain does the wading. Never auto-promotes (voice-safety law)."""
    import memory  # reuse the salience prior (substance/length)

    writing = [t for t in traces if t.get("channel") == "writing"
               and len(t.get("body", "")) >= 200]
    if not writing or not gold:
        return []

    gold_emb = _embed([g["body"] for g in gold], model_name)
    w_emb = _embed([t["body"] for t in writing], model_name)

    cands: list[GoldCandidate] = []
    for i, t in enumerate(writing):
        prox = float((w_emb[i] @ gold_emb.T).max()) if len(gold_emb) else 0.0
        # too close to an existing gold = near-duplicate, not a NEW gold; the
        # sweet spot is 'clearly Damla-voice but not already captured'.
        if prox > 0.92:
            continue
        sal = memory.salience(t["body"]) / 10.0
        score = 0.7 * prox + 0.3 * sal
        cands.append(GoldCandidate(
            title=t["title"], channel=t.get("channel", ""),
            voice_proximity=prox, salience=sal, score=score,
            body_preview=t["body"].replace("\n", " "),
        ))
    cands.sort(key=lambda c: -c.score)
    return cands[:top]


# ---------------------------------------------------------------------------
# the sleep pass — run both, report. Persists nothing to gold (Damla's stamp).
# ---------------------------------------------------------------------------
def sleep(model_name: str = "all-mpnet-base-v2") -> dict:
    """One offline consolidation pass over the whole trail. Returns a report:
    stable patterns + a ranked gold-candidate shortlist. No API, no gold writes."""
    import sources
    from corpus import load_gold

    traces = sources.load_all()
    gold = load_gold()

    patterns = find_patterns(traces, model_name=model_name)
    candidates = rank_gold_candidates(traces, gold, model_name=model_name)

    return {
        "trail_size": len(traces),
        "gold_size": len(gold),
        "patterns": patterns,
        "candidates": candidates,
    }


if __name__ == "__main__":
    rep = sleep()
    print(f"\n=== UYKU (konsolidasyon) — {rep['trail_size']} iz, {rep['gold_size']} gold ===\n")

    print(f"NEOCORTEX ÖRÜNTÜLERİ (tekrar eden temalar, {len(rep['patterns'])} küme):")
    for c in rep["patterns"][:8]:
        chans = " ".join(f"{k}:{v}" for k, v in c.channels.items())
        print(f"  [{c.size:3} iz, cohesion {c.cohesion:.2f}] {chans}")
        print(f"        örnek: {c.exemplar_title[:60]}")

    print(f"\nGOLD ADAYLARI (Damla onayı bekler, motor sıraladı, {len(rep['candidates'])}):")
    for c in rep["candidates"][:10]:
        print(f"  score={c.score:.3f} (ses={c.voice_proximity:.2f} sal={c.salience:.2f}) "
              f"{c.title[:52]}")
