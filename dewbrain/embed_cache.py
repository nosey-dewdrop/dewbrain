"""
embed_cache.py — content-addressed embedding cache (shared across all layers).

Every layer that retrieves (retrieval, consolidation, conformal, worldmodel)
re-encodes its corpus from scratch on each run — ~1400 traces, ~30s of GPU/CPU,
paid EVERY time. For an autonomous loop that runs many times, that is the
dominant cost. This caches embeddings by CONTENT HASH: a text's embedding is
computed once, stored on disk, and reused until the text itself changes.

Content-addressed (not index-addressed) so it is correct under edits: if a trace
body changes, its hash changes, so it re-embeds only THAT trace; unchanged traces
hit the cache. Adding a channel re-embeds only the new traces. No manual
invalidation, no stale vectors.

Keyed by (model_name, sha1(text)). Stored as a single .npz per model under data/
(gitignored — embeddings of personal traces are personal). Thread-unsafe by
design (single-process loop); the loop owns it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embed_cache"


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Content-addressed embedding store for one model. encode() returns the
    same (N, d) array the model would, but only computes the misses."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None
        self._store: dict[str, np.ndarray] = {}
        safe = model_name.replace("/", "_")
        self._path = _CACHE_DIR / f"{safe}.npz"
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = np.load(self._path, allow_pickle=False)
                self._store = {k: data[k] for k in data.files}
            except Exception:
                self._store = {}  # corrupt cache = start clean, never crash

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez needs str keys; hashes already are.
        np.savez(self._path, **self._store)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Return embeddings for texts, computing only the cache misses.
        Order-preserving: output[i] corresponds to texts[i]."""
        keys = [_key(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._store]

        if missing_idx:
            self._ensure_model()
            to_embed = [texts[i] for i in missing_idx]
            fresh = np.asarray(
                self._model.encode(to_embed, normalize_embeddings=normalize),
                dtype=np.float32)
            for j, i in enumerate(missing_idx):
                self._store[keys[i]] = fresh[j]
            self._save()

        return np.stack([self._store[k] for k in keys]).astype(np.float32)

    def stats(self) -> dict:
        return {"model": self.model_name, "cached": len(self._store),
                "path": str(self._path)}


# process-wide cache per model, so all layers share one store.
_CACHES: dict[str, EmbeddingCache] = {}


def get_cache(model_name: str = "all-mpnet-base-v2") -> EmbeddingCache:
    if model_name not in _CACHES:
        _CACHES[model_name] = EmbeddingCache(model_name)
    return _CACHES[model_name]


def encode_cached(texts: list[str], model_name: str = "all-mpnet-base-v2",
                  normalize: bool = True) -> np.ndarray:
    """Convenience: encode texts through the shared cache for model_name."""
    return get_cache(model_name).encode(texts, normalize=normalize)


if __name__ == "__main__":
    import time
    import sources
    texts = [t["body"] for t in sources.load_all()][:300]
    print(f"{len(texts)} metin, cache testi...")

    cache = EmbeddingCache()
    # temizle (test için) — gerçek kullanımda cache kalıcı.
    cache._store = {}

    t0 = time.time()
    a = cache.encode(texts)
    t1 = time.time()
    print(f"ilk geçiş (cache boş): {t1 - t0:.1f}s, shape={a.shape}")

    t0 = time.time()
    b = cache.encode(texts)
    t1 = time.time()
    print(f"ikinci geçiş (cache dolu): {t1 - t0:.2f}s")
    print(f"aynı sonuç mu: {np.allclose(a, b)}")
    print(f"stats: {cache.stats()}")
