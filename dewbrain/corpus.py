"""
corpus.py — thin compatibility layer over memory.py.

The old fragile regex parser + two-source drift lived here. Memory (episodic
raw + semantic gold), robust parsing, and the consolidation bridge now live in
ONE place: memory.py. This module only re-exports the legacy entry points so
existing callers (brain.py, retrieval.py) keep working with no change.

Use memory.load_semantic() / memory.load_episodic() for the rich MemoryEntry
objects (salience, provenance note, channel, section); use load_gold/load_raw
below for the legacy {title, body, source} dict shape.
"""
from memory import load_gold, load_raw  # noqa: F401  (re-export)

__all__ = ["load_gold", "load_raw"]


if __name__ == "__main__":
    g, r = load_gold(), load_raw()
    print(f"gold (semantic, ses altını): {len(g)}")
    for e in g:
        print(f"  [{e['source']}] {e['title'][:60]}")
    print(f"\nham (episodic, kobay havuzu): {len(r)}")
    for e in r[:15]:
        print(f"  [{e['source']}] {e['title'][:60]}")
