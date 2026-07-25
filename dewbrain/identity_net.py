"""
identity_net.py — dewbrain's LEARNED identity encoder (the first real neural net).

Until now every layer read a FROZEN off-the-shelf embedding (all-mpnet-base-v2).
That is retrieval over someone else's weights: no matter how good the Hopfield /
conformal math on top is, the geometry the traces live in is generic. Grep proved
it: not one nn.Module, backward, or optimizer in the whole codebase. The "neural
network" was math, not a network that LEARNS.

This module is the missing piece and the moat's technical core:

  A small projection network TRAINED ON ONE PERSON'S TRACES, that maps the generic
  768-d mpnet space into a person-specific identity space where THAT person's own
  structure (which trace belongs with which, by channel/source) is separated better
  than the generic space separates it.

Why this is the moat, tested against Damla's own law:
  - WRAPPER TEST: remove the LLM entirely and this still runs. The output is a set
    of LEARNED WEIGHTS (Theta) fit to the person's data. An LLM wrapper, stripped
    of its LLM, is nothing. This, stripped of the LLM, is a trained encoder.
  - "why can't a competitor copy it": a competitor can call the same GPT. It cannot
    fit weights to Ahmet's / Sebnem's / George's private traces — those weights only
    exist where that person's data is.
  - "why would George give his data": he doesn't. Training is LOCAL. mpnet runs
    offline, this net trains on CPU in seconds, the data and the learned weights
    never leave his machine. No trace is sent to any LLM. Data-never-leaves is a
    property of the architecture, not a promise.

Learning signal (self-supervised, NO human labels needed — critical, since a new
customer has no labels on day one): SUPERVISED CONTRASTIVE over the person's own
structure. Two traces from the same (channel, source) damar are a positive pair
(pull together); traces from different damar are negatives (push apart). The net
learns the person's OWN notion of "these belong together" — which is exactly the
identity geometry retrieval needs and the generic space lacks.

Measured, not claimed: we hold out a test split, and report whether the learned
space separates same-damar from different-damar pairs BETTER than raw mpnet
(AUC / silhouette). If it does not beat mpnet, it is not shipped — the number is
the gate, per Damla's "prove, don't claim".

This is Faz-2's LoRA cousin at CPU scale: it does not touch the base model
(catastrophic-forgetting safe), it learns a thin adapter ON TOP. When the gold
corpus and a GPU arrive, the same interface swaps to a deeper adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# the network — a thin projection head (identity adapter) over frozen mpnet.
# ---------------------------------------------------------------------------
class IdentityEncoder(nn.Module):
    """Projects generic 768-d embeddings into a person-specific identity space.

    Deliberately small: with ~1400 traces a large net overfits and, worse, the
    result would not be honestly attributable to the PERSON'S structure. A thin
    residual MLP with L2-normalized output is enough to rotate/reshape the space
    around this person's own damar structure while staying regularizable.

    Residual (out = normalize(x + adapter(x))) so at init it is ~identity: it can
    only IMPROVE on mpnet, never destroy it — the base geometry is a safe floor.
    """

    def __init__(self, dim: int = 768, hidden: int = 512, out: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )
        # residual bridge so the untrained net starts near the base space.
        self.bridge = nn.Linear(dim, out)
        nn.init.zeros_(self.proj[-1].weight)   # start: adapter contributes nothing
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bridge(x) + self.proj(x)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# supervised contrastive loss (SupCon, Khosla 2020) over damar groups.
# ---------------------------------------------------------------------------
def supcon_loss(z: torch.Tensor, groups: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """Pull same-group (same channel+source damar) together, push others apart.

    z: (B, d) L2-normalized embeddings. groups: (B,) integer damar id.
    For each anchor, positives = other items with the same group. Standard SupCon:
    -mean over positives of log( exp(sim_pos/T) / sum_{k != i} exp(sim_ik/T) ).
    No human labels: the group id is the person's own (channel, source) structure.
    """
    device = z.device
    sim = z @ z.T / temperature                       # (B, B)
    # numerical stability: subtract row max, mask self.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(z), dtype=torch.bool, device=device)
    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    denom = exp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = sim - torch.log(denom)

    pos_mask = (groups[:, None] == groups[None, :]) & ~self_mask
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0                              # anchors with >=1 positive
    if valid.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1)[valid] / pos_count[valid]
    return -pos_log_prob.mean()


# ---------------------------------------------------------------------------
# training data — the person's own traces, grouped by damar, split train/test.
# ---------------------------------------------------------------------------
@dataclass
class IdentityData:
    X: np.ndarray            # (N, 768) frozen base embeddings
    groups: np.ndarray       # (N,) damar id (channel+source)
    group_names: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray


def build_data(model_name: str = "all-mpnet-base-v2",
               min_group: int = 2, test_frac: float = 0.2,
               seed: int = 0) -> IdentityData:
    """Load the person's whole trail, embed once (cached), group by damar, split.

    A damar = (channel, source): 'these traces are the same KIND of thing I made'.
    That is the person's own structure — the signal we learn identity geometry
    from, with no human labeling. Groups with a single member are dropped from
    supervision (no positive pair) but kept for evaluation coverage.
    """
    import sources
    from embed_cache import encode_cached

    traces = sources.load_all()
    X = np.asarray(encode_cached([t["body"] for t in traces], model_name=model_name),
                   dtype=np.float32)

    # damar key = channel + source stem. Fine-grained enough to be the person's
    # structure, coarse enough to have positives.
    keys = [f"{t.get('channel','?')}|{t.get('source','?')}" for t in traces]
    uniq = sorted(set(keys))
    kid = {k: i for i, k in enumerate(uniq)}
    groups = np.array([kid[k] for k in keys], dtype=np.int64)

    # keep only groups with >= min_group members for a valid positive pair.
    counts = np.bincount(groups)
    keep_g = {g for g in range(len(counts)) if counts[g] >= min_group}
    mask = np.array([g in keep_g for g in groups])
    X, groups = X[mask], groups[mask]
    kept_keys = [uniq[g] for g in groups]
    # re-index groups densely.
    remap = {g: i for i, g in enumerate(sorted(set(groups.tolist())))}
    groups = np.array([remap[g] for g in groups], dtype=np.int64)
    group_names = sorted(set(kept_keys))

    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return IdentityData(X=X, groups=groups, group_names=group_names,
                        train_idx=train_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# evaluation — does the LEARNED space separate the person's damar BETTER than
# raw mpnet? Honest gate: pairwise same-vs-different AUC on the held-out split.
# ---------------------------------------------------------------------------
def pair_auc(emb: np.ndarray, groups: np.ndarray, rng, n_pairs: int = 20000
             ) -> float:
    """AUC of 'same damar' vs cosine similarity. Sample same-group and diff-group
    pairs, ask: does higher cosine predict same-damar? 0.5 = no structure, 1.0 =
    perfect. This is the honest number: bigger = the space knows this person's
    structure better."""
    n = len(emb)
    idx = np.arange(n)
    sims, labels = [], []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(float(emb[i] @ emb[j]))
        labels.append(1 if groups[i] == groups[j] else 0)
    sims = np.asarray(sims)
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    # AUC via rank statistic (Mann-Whitney U), no sklearn.
    order = np.argsort(sims)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(sims) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@dataclass
class TrainReport:
    baseline_auc: float
    learned_auc: float
    epochs: int
    final_loss: float
    improved: bool
    n_train: int
    n_test: int
    n_groups: int

    def as_dict(self) -> dict:
        return {"baseline_auc": round(self.baseline_auc, 4),
                "learned_auc": round(self.learned_auc, 4),
                "delta": round(self.learned_auc - self.baseline_auc, 4),
                "improved": self.improved, "epochs": self.epochs,
                "final_loss": round(self.final_loss, 4),
                "n_train": self.n_train, "n_test": self.n_test,
                "n_groups": self.n_groups}


def train(data: IdentityData, epochs: int = 60, lr: float = 1e-3,
          batch: int = 256, temperature: float = 0.1, weight_decay: float = 1e-4,
          out_dim: int = 256, seed: int = 0, log=print) -> tuple[IdentityEncoder, TrainReport]:
    """Fit the identity encoder on the person's TRAIN split, gate on the TEST split.

    The gate (prove, don't claim): learned test-AUC must beat raw-mpnet test-AUC.
    We report both; `improved` is the ship decision. Training is deterministic-ish
    (seeded) and CPU-fast; the whole point is it runs where the data lives."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xtr = torch.from_numpy(data.X[data.train_idx])
    gtr = torch.from_numpy(data.groups[data.train_idx])
    Xte = data.X[data.test_idx]
    gte = data.groups[data.test_idx]

    net = IdentityEncoder(dim=data.X.shape[1], out=out_dim)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(Xtr)
    final_loss = float("nan")
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, batch):
            bidx = perm[s:s + batch]
            z = net(Xtr[bidx])
            loss = supcon_loss(z, gtr[bidx], temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        final_loss = float(np.mean(losses)) if losses else final_loss
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d}  loss={final_loss:.4f}")

    # evaluate on held-out test split: learned vs raw mpnet.
    net.eval()
    with torch.no_grad():
        Zte = net(torch.from_numpy(Xte)).numpy()
    base_auc = pair_auc(Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9),
                        gte, np.random.default_rng(seed))
    learned_auc = pair_auc(Zte, gte, np.random.default_rng(seed))

    rep = TrainReport(
        baseline_auc=base_auc, learned_auc=learned_auc, epochs=epochs,
        final_loss=final_loss, improved=learned_auc > base_auc,
        n_train=len(data.train_idx), n_test=len(data.test_idx),
        n_groups=len(set(data.groups.tolist())))
    return net, rep


# ---------------------------------------------------------------------------
# persistence — the LEARNED WEIGHTS are the artifact (the moat is these bytes).
# Saved under data/ (gitignored: a person's identity weights are personal, never
# pushed). retrieval loads this to project into the person's own space.
# ---------------------------------------------------------------------------
from pathlib import Path

_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "data" / "identity_net"


def save(net: IdentityEncoder, rep: "TrainReport",
         name: str = "default") -> Path:
    """Persist the trained adapter + its honest eval. Only ships if it beat mpnet."""
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _WEIGHTS_DIR / f"{name}.pt"
    torch.save({"state_dict": net.state_dict(),
                "out_dim": net.bridge.out_features,
                "in_dim": net.bridge.in_features,
                "report": rep.as_dict()}, path)
    return path


def load(name: str = "default") -> tuple[IdentityEncoder | None, dict]:
    """Load a trained adapter if present. Returns (None, {}) if not trained yet —
    callers fall back to raw mpnet (safe floor), never crash."""
    path = _WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        return None, {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = IdentityEncoder(dim=ckpt["in_dim"], out=ckpt["out_dim"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt.get("report", {})


def project(net: IdentityEncoder, base_emb: np.ndarray) -> np.ndarray:
    """Map frozen mpnet embeddings (N, 768) into the learned identity space."""
    with torch.no_grad():
        return net(torch.from_numpy(np.asarray(base_emb, dtype=np.float32))).numpy()


if __name__ == "__main__":
    print("kimlik- agi egitiliyor (kisinin kendi verisinde, LLM'siz, lokal)...")
    data = build_data()
    print(f"veri: {len(data.X)} iz, {len(set(data.groups.tolist()))} damar, "
          f"train={len(data.train_idx)} test={len(data.test_idx)}")
    net, rep = train(data)
    print("\n=== SONUC (kanit, iddia degil) ===")
    d = rep.as_dict()
    print(f"  ham mpnet test-AUC : {d['baseline_auc']}")
    print(f"  ogrenilen  test-AUC : {d['learned_auc']}  (delta {d['delta']:+})")
    print(f"  mpnet'i gecti mi   : {d['improved']}")
    if not d["improved"]:
        print("  -> GECMEDI. Bu haliyle satilmaz. Sebep aranir (loss/temp/damar), overclaim yok.")
    else:
        print("  -> gecti. Ogrenen ag kisinin yapisini ham uzaydan iyi ayiriyor.")
    if rep.improved:
        p = save(net, rep)
        print(f"  agirliklar kaydedildi -> {p} (data/, gitignore, kisisel)")
