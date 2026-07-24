"""
dewbrain vmPFC — production coordinator (recall -> generate -> critique -> repair, loop-until-clean).

Real neuro loop (CLAUDE.md / cs-mapping.md):
  hippocampus (recall gold)  -> HippocampalRetrieval (Hopfield/attention)
  neocortex + vmPFC (produce) -> generate() with law + few-shot gold
  prefrontal  (critique)      -> critique() rubric-judge, every item binary a_k
  prefrontal  (repair)        -> repair() targets only the failed items
  vmPFC coordinates the critique<->repair cycle until clean (or max_iter), keeps the best draft.

Engine = Claude (Anthropic API, claude-opus-4-8). This layer is the coordinator; the API
calls it makes are made ROBUST here: retry with backoff, rate-limit aware, streaming for the
long generate/repair turns, JSON-parse guarded critique, hard timeout.

Accepted output + verdict + gold-used are appended to a JSONL history, because an approved
output becomes NEW GOLD -> that flywheel grows the few-shot corpus from 9 toward 100 (Faz 2 fuel).

Config (model/k/beta/max-iter/retry) lives in config.py, never embedded here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Config, load_config
from law import YASA_METNI, RUBRIK, WEIGHTS
from corpus import load_gold
from retrieval import HippocampalRetrieval
from engine import build_engine, Engine, EngineError

# Backward-compat alias so old references keep working. The robust API/session backends
# now live in engine.py; brain.py is engine-agnostic and only calls engine.complete().
DewbrainAPIError = EngineError


# --------------------------------------------------------------------------------------
# The robust API / session backends live in engine.py. brain.py is engine-agnostic:
# it builds prompts and calls engine.complete(prompt). Swap api<->max<->local via config.
# --------------------------------------------------------------------------------------

def _parse_verdict_json(raw: str) -> dict[str, Any]:
    """
    Guarded JSON extraction for the critique output. Claude may wrap the JSON in prose or a
    code fence; we slice to the outermost braces and parse. Raises a clear error on failure
    so the coordinator can decide (rather than crashing mid-loop like the old version did).
    """
    txt = raw.strip()
    if txt.startswith("```"):
        # removeprefix/suffix only touch the fences; strip("`") would delete backticks INSIDE
        # the content (e.g. a code sample in a note), corrupting the JSON.
        txt = txt.removeprefix("```").removesuffix("```").strip()
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip()
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise DewbrainAPIError(f"Critique returned no JSON object:\n{raw[:300]}")
    try:
        return json.loads(txt[start:end + 1])
    except json.JSONDecodeError as e:
        raise DewbrainAPIError(f"Critique JSON invalid: {e}\n{raw[:300]}") from e


# --------------------------------------------------------------------------------------
# neocortex + prefrontal — prompt construction (generate / critique / repair)
# --------------------------------------------------------------------------------------

def _few_shot(gold_hits: list[dict]) -> str:
    """Retrieved gold -> few-shot example block (neocortex semantic reference)."""
    return "\n\n".join(
        f"--- EXAMPLE (Damla's approved voice) ---\n{g['body']}" for g in gold_hits
    )


def generate(cue: str, gold_hits: list[dict], engine: Engine, log) -> str:
    """neocortex + vmPFC: produce a full post from the cue, conditioned on law + retrieved gold."""
    prompt = f"""{YASA_METNI}

Aşağıda Damla'nın KENDİ onayladığı yazılarından örnekler var. Kelime seçimini, ritmini,
cümle dokusunu, düşünce yapısını bunlardan öğren. TAKLİT etme, Damla gibi DÜŞÜN ve YAZ.

{_few_shot(gold_hits)}

--- GÖREV ---
Damla'nın şu ham fikrini, yukarıdaki sesle TAM bir yazıya çevir. Yasaya birebir uy.
Ham fikir:
{cue}

Sadece yazının kendisini ver, başka açıklama yok."""
    return engine.complete(prompt, log)


def critique(draft: str, gold_hits: list[dict], engine: Engine, log) -> dict[str, Any]:
    """prefrontal: adversarial self-critique against the law. Every rubric item is binary a_k.

    K2 fix (cs-mapping (e), few-shot style): the judge sees Damla's approved voice as a
    POSITIVE anchor, not only the do/don't list. Without it the judge is a mere violation-hunter
    that passes 'clean but generic' text — the exact 'yoksa generic' failure the law warns of.
    Voice separates from positive style examples, not from a bare prohibition list.
    """
    items = "\n".join(f"- {key}: {desc}" for key, desc in RUBRIK)
    prompt = f"""Sen Damla'nın sesinin acımasız denetçisisin. Aşağıdaki yazıyı Damla YAZMIŞ MI diye
her maddeyi tek tek kontrol et. Yalaka olma, ihlali gördüğün yeri SÖYLE. Sen YAZAN değil
DENETLEYEN'sin, taslağı düşman gözüyle oku.

Damla'nın onaylı sesi (pozitif çapa, yazı buna benzemeli):
{_few_shot(gold_hits)}

Denetlenecek yazı:
{draft}

Her madde için pass=true/false ve ihlalse tek cümle kanıt ver. damla_kokuyor sadece ihlal
yokluğu DEĞİL, yukarıdaki sesle gerçekten örtüşüp örtüşmediğidir (temiz ama generic ise false).
SADECE JSON döndür, başka hiçbir şey:
{{"items": [{{"key": "...", "pass": true, "note": "..."}}], "damla_kokuyor": true, "en_zayif": "..."}}

Maddeler:
{items}"""
    raw = engine.complete(prompt, log)
    verdict = _parse_verdict_json(raw)
    if "items" not in verdict or not isinstance(verdict["items"], list):
        raise DewbrainAPIError(f"Critique JSON missing 'items' list:\n{raw[:300]}")
    return verdict


def repair(draft: str, verdict: dict, gold_hits: list[dict], engine: Engine, log) -> str:
    """prefrontal repair: rewrite targeting ONLY the failed items, leaving the rest intact."""
    fails = [i for i in verdict["items"] if not i.get("pass", False)]
    if not fails:
        return draft
    fixlist = "\n".join(f"- {i.get('key')}: {i.get('note', '')}" for i in fails)
    prompt = f"""{YASA_METNI}

Damla'nın onaylı sesi (referans):
{_few_shot(gold_hits)}

Aşağıdaki yazı şu maddelerde Damla'nın sesinden SAPMIŞ. SADECE bu ihlalleri düzelt,
geri kalanı bozma, yazıyı yeniden yazma. İhlaller:
{fixlist}

Yazı:
{draft}

Düzeltilmiş yazıyı ver, sadece metin."""
    return engine.complete(prompt, log)


# --------------------------------------------------------------------------------------
# gold-accumulation flywheel — persist accepted output + verdict + gold-used
# --------------------------------------------------------------------------------------

def _fail_count(verdict: dict) -> int:
    """Kaç madde ihlal edildi (gate için: sıfır ihlal = temiz)."""
    return sum(1 for i in verdict.get("items", []) if not i.get("pass", False))


def _weighted_fail(verdict: dict) -> float:
    """RUBRIC-ARROW ağırlıklı ihlal skoru (cs-mapping (d)). best-draft SEÇİMİ bununla yapılır.
    1 em-dash taşıyan taslak (w=3.0) 2 hafif ihlalden (2x w=1.0) DAHA KÖTÜ; düz sayım bunu
    ters seçerdi. Gate hâlâ 'sıfır ihlal' (_fail_count), ama best-draft ağırlıklı."""
    return sum(WEIGHTS.get(i.get("key"), 1.0)
               for i in verdict.get("items", []) if not i.get("pass", False))


def persist_run(result: dict, cfg: Config) -> Path:
    """
    Append the run to the JSONL history. An accepted output (clean verdict) is a candidate for
    NEW GOLD — this is the corpus flywheel (9 -> 100). We record every run (accepted or not)
    with an `accepted` flag so consolidation can later promote the clean ones into verified gold.
    """
    path = cfg.history_path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": cfg.api.model,
        "config": cfg.as_dict(),
        **result,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------------------
# vmPFC coordinator — the loop
# --------------------------------------------------------------------------------------

def run(cue: str, cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    """
    Full vmPFC pass: recall -> generate -> (critique -> repair)* until clean or max_iter,
    keeping the best draft seen. Persists the run to history for the gold flywheel.
    """
    cfg = cfg or load_config()
    log = (lambda m: print(m)) if verbose else (lambda m: None)

    engine = build_engine(cfg)
    log(f"[engine] backend={cfg.api.engine} model={cfg.api.model}")

    # hippocampus: recall the most separable gold traces (Hopfield/attention, not cosine-top-k).
    gold = load_gold()
    if not gold:
        raise DewbrainAPIError("No gold found (verified corpus empty) — cannot condition voice.")
    beta_src = "config" if cfg.loop.beta is not None else "calibrated-from-corpus"
    log(f"[hippocampus] {len(gold)} gold loaded, building retrieval (beta={beta_src})...")
    # beta=None => HippocampalRetrieval calibrates from corpus geometry (the point
    # of calibrate_beta). Passing a hard 8.0 here would silently kill that.
    retr = HippocampalRetrieval(gold, model_name=cfg.loop.embedding_model, beta=cfg.loop.beta)
    hits = retr.recall(cue, k=cfg.loop.k)
    sep = retr.separability
    log(f"[hippocampus] beta={retr.beta:.2f} separability mean_top_mass={sep['mean_top_mass']:.2f} "
        f"({len(sep.get('collisions', []))} colliding golds — blend risk while corpus is small)")
    log("[hippocampus] retrieved gold (score = relevance + recency + salience, MMR-diverse):")
    for h in hits:
        log(f"  score={h['score']:.3f} w={h['weight']:.3f} sim={h['sim']:.3f}  {h['title'][:50]}")

    # neocortex + vmPFC: first draft.
    log("[neocortex+vmPFC] generating first draft...")
    draft = generate(cue, hits, engine, log)

    # prefrontal: loop-until-clean. Re-critique after every repair; keep the best draft by
    # WEIGHTED violation (RUBRIC-ARROW), not raw count — a single em-dash beats two light fails.
    best_draft = draft
    best_verdict: dict[str, Any] | None = None
    best_weight = float("inf")
    best_fails = 10**9
    iterations = 0

    current = draft
    for i in range(cfg.loop.max_iter):
        iterations = i + 1
        log(f"[prefrontal] critique pass {iterations}/{cfg.loop.max_iter}...")
        verdict = critique(current, hits, engine, log)
        fails = _fail_count(verdict)
        wfail = _weighted_fail(verdict)
        log(f"  violations={fails} weighted={wfail:.1f} damla_kokuyor={verdict.get('damla_kokuyor')}")

        if wfail < best_weight:
            best_weight, best_fails, best_draft, best_verdict = wfail, fails, current, verdict

        if fails == 0:
            log("[prefrontal] clean — loop done.")
            break
        if iterations >= cfg.loop.max_iter:
            log("[prefrontal] max_iter reached — keeping best draft.")
            break

        log(f"[prefrontal] repairing {fails} violation(s) (weighted {wfail:.1f})...")
        current = repair(current, verdict, hits, engine, log)

    accepted = best_fails == 0
    result = {
        "cue": cue,
        "gold_used": [h["title"] for h in hits],
        "final": best_draft,
        "verdict": best_verdict,
        "ihlal_sayisi": best_fails,
        "weighted_fail": best_weight,
        "iterations": iterations,
        "accepted": accepted,
    }

    hist = persist_run(result, cfg)
    log(f"[flywheel] run saved -> {hist} (accepted={accepted})")
    result["history_path"] = str(hist)
    return result


if __name__ == "__main__":
    import sys

    cue_in = sys.argv[1] if len(sys.argv) > 1 else (
        "For a long time I said 'I can't' about things that had nothing to do with skill. "
        "I can't post this, I can't reach out, I can't sell. Then I saw that every time, "
        "'I can't' was hiding something else: I did not know how, or I was afraid. The moment "
        "someone broke it into a small concrete step, the 'can't' disappeared. I could not "
        "'do marketing' but I could write one honest post about one real number."
    )
    out = run(cue_in)
    print("\n" + "=" * 60)
    print("HAM FİKİR:", out["cue"][:80], "...")
    print("KULLANILAN ALTIN:", out["gold_used"])
    print(f"İHLAL: {out['ihlal_sayisi']}  tur: {out['iterations']}  accepted: {out['accepted']}")
    if out["verdict"]:
        print("damla_kokuyor:", out["verdict"].get("damla_kokuyor"))
    print("=" * 60)
    print("\n### FİNAL (bu ben miyim?):\n")
    print(out["final"])
