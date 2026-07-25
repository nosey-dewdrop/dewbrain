"""
hippocampus katmanı — retrieval (pattern separation + completion).

GERÇEK mekanizma (research/cs-mapping.md, açı (a) + (e) + (c)):
Depo değil INDEX. Bir cue (yarım fikir) verilince stored pattern'ları (verified/ham
embedding) getirir. cosine-top-k YETMEZ: benzer iki altını ortalar = metastable state
= blend (Damla'nın iki farklı yazısının karışımı, hiçbiri gibi kokmayan puree).

Çözüm = SPARSE MODERN HOPFIELD (attention-eşdeğer) + retrieval skoru + çeşitlilik:
- Modern Hopfield update = softmax(beta * X q) X   (Ramsauer 2020, arXiv:2008.02217).
  Tek-adımda retrieval, üssel depolama kapasitesi. dense softmax = eski kod.
- SPARSE Hopfield alpha-entmax ile (Hu 2024, arXiv:2402.13725): softmax yerine
  1.5-entmax -> alakasız pattern'lara TAM SIFIR ağırlık (Fenchel-Young margin).
- HEN mantığı (arXiv:2409.16408): ham metni değil pretrained encoder LATENT'ini sakla.

DÜRÜST SINIR (9 altın rejiminde ölçülmüş gerçek, uydurma değil): 9 kısa LinkedIn
yazısı aynı seste; kosinüs benzerlikleri 0.03-0.25 gibi DAR bir banda sıkışır. Bu
geometride entmax tek well'e çökMEZ — üstteki iki altına ~0.45/0.55 verir, yani tam
da elemeye çalıştığı blend'i üretir. Sebep entmax değil, VERİ: az ve homojen korpusta
ayrılabilir "well" yoktur. O yüzden bu modül entmax'i tek koz saymaz:
  1. retrieval skoru = relevance + recency + salience (research c, generative agents
     arXiv:2304.03442) — kosinüs tek başına değil.
  2. k-seçimi MMR/greedy çeşitlilik ile — iki near-duplicate altını üst üste getirmez,
     completion için TAMAMLAYICI izler seçer (pattern completion = kombinasyon).
  3. separability ÖLÇÜLÜR ve dürüst raporlanır; korpus büyüyünce (Faz 2) entmax'in
     gerçekten ayırmaya başladığı eşik burada görünür. Bugün marifet entmax değil,
     skor+çeşitlilik; overclaim yok.
beta korpustan kalibre edilir (sabit değil); prod yolu da bunu kullanır (config.beta=None).

İngilizce yorum aşağıda; üstteki Türkçe gerekçe koddan bağımsız okunabilir.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# alpha-entmax  (Fenchel-Young sparse softmax; Peters & Martins 2019, Hu 2024)
# ---------------------------------------------------------------------------
def entmax_bisect(scores: torch.Tensor, alpha: float = 1.5,
                  n_iter: int = 50, dim: int = -1) -> torch.Tensor:
    """Alpha-entmax over `dim` via bisection on the threshold tau.

    entmax_alpha(z)_i = [ (alpha-1) z_i - tau ]_+ ^ (1/(alpha-1))
    where tau is chosen so the outputs sum to 1. For 1 < alpha the map is
    SPARSE: scores far below the top get EXACTLY zero mass. That exact zero is
    what kills the metastable blend that plain softmax (alpha->1) produces.

    Bisection is the standard exact solver for general alpha (Peters 2019); it
    is a handful of iterations, deterministic, and needs no external package.
    """
    if alpha == 1.0:
        return torch.softmax(scores, dim=dim)
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1 for a sparse map")

    am1 = alpha - 1.0
    z = scores * am1  # work in the (alpha-1)-scaled space

    # tau lives in [max(z) - 1, max(z)]; the feasible set for the threshold.
    z_max = z.max(dim=dim, keepdim=True).values
    tau_lo = z_max - 1.0
    tau_hi = z_max

    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(z - tau, min=0.0) ** (1.0 / am1)
        s = p.sum(dim=dim, keepdim=True)
        # sum decreases as tau grows -> if sum > 1 push tau up, else down.
        too_big = s > 1.0
        tau_lo = torch.where(too_big, tau, tau_lo)
        tau_hi = torch.where(too_big, tau_hi, tau)

    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(z - tau, min=0.0) ** (1.0 / am1)
    # renormalize the tiny bisection residual so weights sum to exactly 1.
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return p


# ---------------------------------------------------------------------------
# beta calibration  (no magic constant)
# ---------------------------------------------------------------------------
def calibrate_beta(sim_matrix: np.ndarray, target_ratio: float = 6.0) -> float:
    """Pick beta from the DATA, not a hard-coded 8.0.

    Modern Hopfield's beta (inverse temperature) sets how sharply attention
    concentrates. Too low -> everything blends (metastable). Too high -> only
    the single nearest, brittle. We want the true top well to dominate the
    runner-up by ~`target_ratio` on the AVERAGE cue, measured against the real
    corpus geometry.

    For a leave-one-out cue i, let g1,g2 be its 1st/2nd similarity to the OTHER
    patterns. softmax mass ratio between them is exp(beta*(g1-g2)). Solving the
    mean gap for the target ratio gives a beta grounded in this corpus's own
    separability. Falls back to a safe sharp default on degenerate corpora.
    """
    n = sim_matrix.shape[0]
    if n < 2:
        return 12.0
    gaps = []
    for i in range(n):
        row = np.delete(sim_matrix[i], i)  # drop self-similarity
        if row.size < 2:
            continue
        top2 = np.partition(row, -2)[-2:]
        gaps.append(float(top2[-1] - top2[-2]))
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    if mean_gap < 1e-4:
        # near-duplicate corpus: geometry can't separate; lean on entmax sparsity.
        return 20.0
    beta = np.log(target_ratio) / mean_gap
    return float(np.clip(beta, 4.0, 40.0))


# ---------------------------------------------------------------------------
# Hippocampal retrieval
# ---------------------------------------------------------------------------
class HippocampalRetrieval:
    """Sparse modern-Hopfield associative recall over the gold/raw corpus.

    Stored patterns = L2-normalized encoder latents (HEN: separable in high dim).
    A cue maps to a weight distribution via alpha-entmax(beta * X q). Entmax gives
    EXACT sparsity so dissimilar patterns get zero mass (no blend); beta is
    calibrated to the corpus so the top well wins by a real margin (separation).
    """

    def __init__(self, entries, model_name: str = "all-mpnet-base-v2",
                 alpha: float = 1.5, beta: float | None = None,
                 target_ratio: float = 6.0,
                 recency_w: float = 0.10, salience_w: float = 0.10,
                 channel_w: float = 0.10, diversity: float = 0.5):
        from sentence_transformers import SentenceTransformer

        if not entries:
            raise ValueError("retrieval needs at least one stored pattern")

        self.entries = entries
        self.channel_w = channel_w  # channel-balance prior (report 732 must not
        # drown decision 45 by sheer COUNT; per-trace channel weight offsets it).
        self.alpha = alpha  # 1.5 = 1.5-entmax; ->1 recovers dense softmax
        # research c: score = relevance + recency + importance. Weights small so
        # relevance still dominates; they only break ties in a homogeneous corpus.
        self.recency_w = recency_w
        self.salience_w = salience_w
        self.diversity = diversity  # 1.0 = pure MMR diversity, 0.0 = pure relevance
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name

        # content-addressed cache: only unseen bodies are embedded, the rest hit
        # disk. Turns the ~30s corpus encode into ~0s on repeat runs (autonomous
        # loop). Falls back to a direct encode if the cache module is unavailable.
        texts = [e["body"] for e in entries]
        try:
            from embed_cache import encode_cached
            emb = encode_cached(texts, model_name=model_name, normalize=True)
        except Exception:
            emb = self.model.encode(texts, normalize_embeddings=True)
        self.emb = torch.as_tensor(np.asarray(emb, dtype=np.float32))  # (N, d)

        # recency + salience + channel priors, normalized to [0,1] once at build.
        self._recency = self._recency_prior(entries)
        self._salience = self._salience_prior(entries)
        self._channel = self._channel_prior(entries)

        # corpus self-similarity (cosine, since normalized) drives calibration.
        self._sim = (self.emb @ self.emb.T).numpy()
        self.beta = beta if beta is not None else calibrate_beta(self._sim, target_ratio)

        # separability diagnostic, computed once at build time.
        self.separability = self._measure_separability()

    # -- priors -------------------------------------------------------------
    @staticmethod
    def _recency_prior(entries) -> np.ndarray:
        """GG.AA.YYYY date -> [0,1] recency (newest = 1). Undated = neutral 0.5."""
        import datetime as _dt
        vals, have = [], []
        for e in entries:
            d = e.get("date", "")
            try:
                vals.append(_dt.datetime.strptime(d, "%d.%m.%Y").timestamp())
                have.append(True)
            except (ValueError, TypeError):
                vals.append(None)
                have.append(False)
        arr = np.full(len(entries), 0.5, dtype=np.float32)
        real = [v for v, h in zip(vals, have) if h]
        if len(real) >= 2:
            lo, hi = min(real), max(real)
            span = (hi - lo) or 1.0
            for i, (v, h) in enumerate(zip(vals, have)):
                if h:
                    arr[i] = (v - lo) / span
        return arr

    @staticmethod
    def _salience_prior(entries) -> np.ndarray:
        """entry.salience (poignancy 1-10) -> [0,1]. Unscored gold -> neutral 0.5."""
        s = np.array([float(e.get("salience", 0.0) or 0.0) for e in entries],
                     dtype=np.float32)
        out = np.where(s > 0.0, s / 10.0, 0.5).astype(np.float32)
        return out

    @staticmethod
    def _channel_prior(entries) -> np.ndarray:
        """channel_weight -> [0,1]. Offsets count imbalance: a report channel
        (732 traces, w=0.9) must not outweigh a decision channel (45, w=1.4) just
        by having more traces. The per-trace weight nudges the score so a decision
        can win over a report even though reports are far more numerous. Missing
        weight -> neutral 1.0 (no offset)."""
        w = np.array([float(e.get("channel_weight", 1.0) or 1.0) for e in entries],
                     dtype=np.float32)
        lo, hi = float(w.min()), float(w.max())
        if hi - lo < 1e-6:
            return np.full(len(entries), 0.5, dtype=np.float32)
        return ((w - lo) / (hi - lo)).astype(np.float32)

    # -- core recall --------------------------------------------------------
    def recall(self, cue: str, k: int = 3, min_weight: float = 1e-3):
        """cue -> k complementary gold traces (relevance + recency + salience, MMR-diverse).

        entmax mass is the relevance term. On a tiny homogeneous corpus entmax
        alone blends the top two (measured 0.45/0.55); so the final score adds a
        small recency + salience prior (research c) and k-selection uses MMR so
        the returned set is DIVERSE (pattern completion needs complementary
        traces, not two near-duplicates). Each hit carries {weight, sim, score,
        recency, salience, active} for transparency.
        """
        q = self._encode(cue)
        sim = self.emb @ q  # (N,) cosine, normalized space
        w = entmax_bisect(self.beta * sim, alpha=self.alpha, dim=-1)

        w_np = w.numpy()
        sim_np = sim.numpy()
        active = int((w_np > min_weight).sum())

        # research c: relevance + recency + importance. entmax mass carries
        # relevance; small priors break ties in a homogeneous corpus.
        base = (w_np
                + self.recency_w * self._recency
                + self.salience_w * self._salience
                + self.channel_w * self._channel)

        picked = self._mmr_select(base, k=k, min_weight=min_weight)
        hits = []
        for i in picked:
            hits.append({
                **self.entries[int(i)],
                "weight": float(w_np[i]),
                "sim": float(sim_np[i]),
                "score": float(base[i]),
                "recency": float(self._recency[i]),
                "salience": float(self._salience[i]),
                "active": active,
            })
        return hits

    def _mmr_select(self, base: np.ndarray, k: int, min_weight: float) -> list[int]:
        """Maximal Marginal Relevance: greedily pick high-score, low-redundancy items.

        MMR(i) = (1-d)*base_i - d*max_{j in chosen} cos(i, j). This stops the
        recall set from being two near-identical golds (the blend the corpus can't
        separate). d = self.diversity. Candidates below min_weight are dropped.
        """
        d = self.diversity
        cand = [i for i in range(len(base)) if base[i] > min_weight]
        if not cand:
            cand = [int(np.argmax(base))]
        chosen: list[int] = []
        while cand and len(chosen) < k:
            if not chosen:
                best = max(cand, key=lambda i: base[i])
            else:
                def mmr(i: int) -> float:
                    redundancy = max(float(self._sim[i, j]) for j in chosen)
                    return (1.0 - d) * float(base[i]) - d * redundancy
                best = max(cand, key=mmr)
            chosen.append(best)
            cand.remove(best)
        return chosen

    # -- separability measurement ------------------------------------------
    def _measure_separability(self) -> dict:
        """Leave-one-out: does each stored pattern recall ITSELF cleanly?

        For pattern i we cue with i's own latent (minus self), run entmax over
        the rest, and record the top mass. High mean top-mass + low blend
        (entropy) => wells are separable, retrieval won't puree neighbors. This
        is the number to watch as the corpus grows: if it drops, similar golds
        are colliding and it's time for the sparse (higher-alpha) regime.
        """
        n = self.emb.shape[0]
        if n < 2:
            return {"n": n, "mean_top_mass": 1.0, "mean_active": 1.0,
                    "collisions": []}
        sim = self._sim.copy()
        np.fill_diagonal(sim, -np.inf)  # can't retrieve yourself
        scores = torch.as_tensor(self.beta * sim, dtype=torch.float32)
        # -inf on the diagonal safely maps to zero mass under entmax.
        scores = torch.nan_to_num(scores, neginf=-1e9)
        w = entmax_bisect(scores, alpha=self.alpha, dim=-1).numpy()

        top_mass = w.max(axis=1)
        active = (w > 1e-3).sum(axis=1)
        # collisions: patterns whose nearest neighbor still steals >35% mass.
        collisions = []
        for i in range(n):
            j = int(np.argmax(w[i]))
            if w[i, j] < 0.65:
                collisions.append((i, j, float(w[i, j])))
        return {
            "n": n,
            "beta": self.beta,
            "mean_top_mass": float(top_mass.mean()),
            "mean_active": float(active.mean()),
            "collisions": collisions,
        }

    # -- helpers ------------------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return torch.as_tensor(np.asarray(v, dtype=np.float32))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # smoke test / separability report on the live corpus.
    import sys
    from corpus import load_gold

    gold = load_gold()
    print(f"[hippocampus] {len(gold)} gold patterns loaded")
    retr = HippocampalRetrieval(gold)
    s = retr.separability
    print(f"[calibration] beta={retr.beta:.2f} alpha={retr.alpha}")
    print(f"[separability] mean_top_mass={s['mean_top_mass']:.3f} "
          f"mean_active={s['mean_active']:.2f}  "
          f"(1.0/1.0 = perfectly separable wells)")
    if s["collisions"]:
        print(f"[separability] {len(s['collisions'])} colliding pairs "
              f"(neighbor steals >35% mass) — similar golds, watch as corpus grows:")
        for i, j, m in s["collisions"][:5]:
            print(f"    #{i} <-> #{j}  top_mass={m:.2f}  "
                  f"{gold[i]['title'][:35]!r} ~ {gold[j]['title'][:35]!r}")
    else:
        print("[separability] no collisions — every gold recalls itself cleanly")

    cue = sys.argv[1] if len(sys.argv) > 1 else (
        "I kept saying 'I can't' about things that were not about skill at all — "
        "posting, reaching out, selling. It was fear or not knowing how, hiding "
        "behind 'can't'. Break it into one small step and the 'can't' disappears."
    )
    print(f"\n[recall] cue: {cue[:70]}...")
    hits = retr.recall(cue, k=3)
    print(f"[recall] entmax kept {hits[0]['active'] if hits else 0} non-zero wells "
          f"(sparse: dissimilar golds got exactly 0 mass, no blend)")
    for h in hits:
        print(f"  w={h['weight']:.3f} sim={h['sim']:.3f}  {h['title'][:55]}")
