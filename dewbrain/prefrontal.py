"""
prefrontal / basal ganglia — yasa-reward model (öz-eleştiri + düzeltme).

Görev: bir taslağı Damla'nın yasasına karşı test et (RUBRIC-ARROW judge), ihlalleri
bul, hedefli düzelt, temizlenene kadar döngüle (loop-until-clean). Üretim kalitesi:
bozuk JSON'da ÇÖKMEZ, retry'lı, çıktılar/verdict'ler diske biriktirilir (altın yakıtı).

Nöromimari eşleme (cs-mapping.md (d)):
  prefrontal reward model = rubrik-koşullu judge. Her rubrik maddesi binary a_k in {0,1};
  ağırlıklı ortalama pointwise reward (RUBRIC-ARROW, arXiv:2605.29156). Bugün prompt-
  tabanlı judge (model rubriği okuyup her maddeye pass/fail verir); veri büyüyünce
  (100+ altın, GPU) öğrenilebilir reward model / DPO buraya girer, arayüz aynı kalır.

TASARIM İLKELERİ (kök sebep, yama değil):
  1. JSON parse çökmez. Model bozuk JSON dönerse: çok-stratejili çıkarım → şema
     doğrulama → coerce → tek turluk "JSON'ı onar" retry → güvenli fallback (fail-closed:
     çıkarılamayan madde "ihlal var" sayılır, "geçti" değil — sessiz geçiş YASAK).
  2. loop-until-clean. repair sonrası YENİDEN critique; ihlal kalırsa tekrar düzelt,
     max_iter'e kadar. Tek-tur değil.
  3. API çağrısı retry'lı (geçici hata / boş içerik / bozuk JSON'da yeniden dene).
  4. Her tur diske yazılır (data/prefrontal/): altın biriktirme + denetlenebilirlik.
  5. Yasa TEK KAYNAKTAN (law.py → dewrites.md). Burada yasa kopyası YOK.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import anthropic

from law import LAW, Law, RubricItem

MODEL = "claude-opus-4-8"  # ses inceliği + yargı için en güçlü
_JUDGE_MAX_TOKENS = 2000
_REPAIR_MAX_TOKENS = 2000
_API_RETRIES = 3
_API_BACKOFF = 2.0  # saniye, üstel

# Verdict/taslak arşivi. data/ gitignore'da (kişisel veri).
_ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "prefrontal"


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------
@dataclass
class ItemVerdict:
    """Tek rubrik maddesinin binary yargısı (RUBRIC-ARROW a_k)."""

    key: str
    passed: bool  # a_k: True=1 (geçti), False=0 (ihlal)
    note: str = ""  # ihlalse tek cümle kanıt
    parsed: bool = True  # judge çıktısından güvenle okunabildi mi (fail-closed izi)


@dataclass
class Verdict:
    """Bir taslağın tam yasa yargısı."""

    items: list[ItemVerdict]
    smells_like_damla: bool  # judge'ın bütünsel "Damla mı?" hükmü
    weakest: str = ""  # en zayıf madde açıklaması
    raw: str = ""  # ham judge çıktısı (denetim için)

    @property
    def fails(self) -> list[ItemVerdict]:
        return [i for i in self.items if not i.passed]

    @property
    def clean(self) -> bool:
        """Geçme kapısı = SIFIR ihlal (skor değil sınır; yasa reçete değil sınırdır)."""
        return len(self.fails) == 0

    def score(self, law: Law) -> float:
        """Ağırlıklı pointwise reward (RUBRIC-ARROW). 0..1. Rapor/izleme için;
        geçme kararı clean üstünden (sıfır-tolerans), skor kalite göstergesi."""
        w_by_key = {i.key: i.weight for i in law.items}
        total = sum(w_by_key.get(i.key, 1.0) for i in self.items)
        if total == 0:
            return 0.0
        got = sum(w_by_key.get(i.key, 1.0) for i in self.items if i.passed)
        return got / total


# ---------------------------------------------------------------------------
# API çağrısı — retry'lı, boş/hatalı içerikte yeniden dener
# ---------------------------------------------------------------------------
def _call(client: anthropic.Anthropic, prompt: str, max_tokens: int) -> str:
    """Tek turluk mesaj çağrısı, geçici hatalarda üstel backoff'la retry.
    Adaptive thinking açık (yasa yargısı ince akıl ister)."""
    last_err: Exception | None = None
    for attempt in range(_API_RETRIES):
        try:
            r = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in r.content if b.type == "text").strip()
            if text:
                return text
            last_err = RuntimeError("boş yanıt (metin bloğu yok)")
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_err = e
        if attempt < _API_RETRIES - 1:
            time.sleep(_API_BACKOFF * (2**attempt))
    raise RuntimeError(f"API çağrısı {_API_RETRIES} denemede başarısız: {last_err}")


# ---------------------------------------------------------------------------
# JSON çıkarım — çok stratejili, ÇÖKMEZ
# ---------------------------------------------------------------------------
def _strip_code_fence(s: str) -> str:
    """```json ... ``` çitini soy."""
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    return m.group(1).strip() if m else s


def _balanced_object(s: str) -> str | None:
    """İlk '{' ile onu dengeleyen '}' arasını (string-farkında) çıkar.
    rfind('}') naif; string içindeki süslü parantez onu bozar. Bu tarayıcı
    tırnak/kaçış-farkında sayar."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None  # dengelenmemiş (kesik yanıt)


def _try_json(raw: str) -> dict | None:
    """Sırayla dene: düz → çitsiz → dengeli-obje. İlk geçerli dict'i döndür."""
    for candidate in (raw, _strip_code_fence(raw)):
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    balanced = _balanced_object(raw)
    if balanced:
        try:
            obj = json.loads(balanced)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _as_bool(v: Any) -> bool | None:
    """Model 'true'/'evet'/1/'pass' gibi çeşitli formlarda dönebilir; tolere et."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "yes", "evet", "pass", "1", "geçti", "gecti"}:
            return True
        if t in {"false", "no", "hayır", "hayir", "fail", "0", "ihlal"}:
            return False
    return None


def _coerce_verdict(obj: dict, law: Law) -> Verdict:
    """JSON dict'i şemaya oturt. FAIL-CLOSED: bir madde okunamıyorsa 'ihlal' say
    (sessiz geçiş YASAK — okunamayan yargı geçmiş sayılmaz).

    Judge çıktı şeması:
      {"items": [{"key","pass","note"}], "damla_kokuyor": bool, "en_zayif": str}
    """
    raw_items = obj.get("items")
    by_key: dict[str, dict] = {}
    if isinstance(raw_items, list):
        for it in raw_items:
            if isinstance(it, dict) and isinstance(it.get("key"), str):
                by_key[it["key"]] = it

    items: list[ItemVerdict] = []
    for law_item in law.items:
        entry = by_key.get(law_item.key)
        if entry is None:
            # Judge bu maddeyi hiç döndürmedi → fail-closed.
            items.append(
                ItemVerdict(key=law_item.key, passed=False,
                            note="judge bu maddeyi döndürmedi (fail-closed)", parsed=False)
            )
            continue
        b = _as_bool(entry.get("pass"))
        if b is None:
            items.append(
                ItemVerdict(key=law_item.key, passed=False,
                            note="judge pass alanı okunamadı (fail-closed)", parsed=False)
            )
            continue
        note = entry.get("note")
        items.append(
            ItemVerdict(key=law_item.key, passed=b,
                        note=str(note).strip() if note else "", parsed=True)
        )

    smells = _as_bool(obj.get("damla_kokuyor"))
    # Bütünsel hüküm okunamadıysa, madde-temelli sonuca düş (tüm maddeler geçtiyse True).
    if smells is None:
        smells = all(i.passed for i in items)
    weakest = obj.get("en_zayif")
    return Verdict(
        items=items,
        smells_like_damla=bool(smells),
        weakest=str(weakest).strip() if weakest else "",
        raw=obj.get("__raw__", ""),
    )


# ---------------------------------------------------------------------------
# Judge (critique) — rubrik-koşullu, her madde binary
# ---------------------------------------------------------------------------
def _judge_prompt(draft: str, law: Law) -> str:
    lines = "\n".join(f'- key="{i.key}" [{i.kind}]: {i.desc}' for i in law.items)
    keys = ", ".join(f'"{i.key}"' for i in law.items)
    return f"""Sen Damla'nın sesinin acımasız denetçisisin (yalaka DEĞİL). Aşağıdaki yazıyı
Damla YAZMIŞ MI diye her maddeyi tek tek kontrol et. İhlali gördüğün yeri açıkça söyle,
şüphede kalırsan İHLAL say (ses saflığı taviz vermez).

Yazı:
\"\"\"
{draft}
\"\"\"

Aşağıdaki maddelerin HER BİRİ için karar ver. Sadece geçerli JSON döndür, başka hiçbir
şey yazma. Şema (her madde için tam olarak bu key'ler, {len(law.items)} madde eksiksiz):
{{"items": [{{"key": <bu key'lerden biri: {keys}>, "pass": true|false, "note": "<ihlalse tek cümle kanıt, geçtiyse boş>"}}],
  "damla_kokuyor": true|false,
  "en_zayif": "<en zayıf maddenin tek cümle özeti>"}}

Maddeler:
{lines}"""


def critique(
    client: anthropic.Anthropic, draft: str, law: Law = LAW
) -> Verdict:
    """Taslağı yasaya karşı yargıla. Bozuk JSON'da ÇÖKMEZ: parse edilemezse tek
    turluk 'JSON'ı onar' retry, o da olmazsa fail-closed verdict."""
    prompt = _judge_prompt(draft, law)
    raw = _call(client, prompt, _JUDGE_MAX_TOKENS)
    obj = _try_json(raw)

    if obj is None:
        # Kök sebep: model bazen JSON'ı bozuk/eksik döndürür. Yama (regex ile
        # değer kazımak) YASAK; modele kendi çıktısını GEÇERLİ JSON'a çevirtiriz.
        repair_prompt = (
            "Aşağıdaki metni GEÇERLİ tek bir JSON nesnesine çevir. Sadece JSON döndür, "
            "açıklama/çit yok. Şema aynı kalsın "
            '({"items":[{"key","pass","note"}],"damla_kokuyor":bool,"en_zayif":str}):\n\n'
            + raw
        )
        try:
            raw2 = _call(client, repair_prompt, _JUDGE_MAX_TOKENS)
            obj = _try_json(raw2)
            if obj is not None:
                raw = raw2
        except RuntimeError:
            obj = None

    if obj is None:
        # Son çare: hiçbir şey parse edilemedi → tüm maddeler fail-closed.
        # (Sessizce "geçti" demektense DURDURUCU ihlal üret; insan görsün.)
        obj = {"items": [], "damla_kokuyor": False,
               "en_zayif": "judge çıktısı JSON'a çevrilemedi (fail-closed)"}

    obj["__raw__"] = raw
    return _coerce_verdict(obj, law)


# ---------------------------------------------------------------------------
# Repair — ihlalleri hedefli düzelt (yeniden yazMA)
# ---------------------------------------------------------------------------
def repair(
    client: anthropic.Anthropic,
    draft: str,
    verdict: Verdict,
    law: Law = LAW,
    few_shot: str = "",
) -> str:
    """Yalnız ihlal edilen maddeleri düzelt; geri kalanı bozma. few_shot verilirse
    (getirilen altınlar) referans olarak eklenir (neocortex ses çapası)."""
    fails = verdict.fails
    if not fails:
        return draft
    fixlist = "\n".join(f"- {i.key}: {i.note or law_desc(law, i.key)}" for i in fails)
    ref = f"\nDamla'nın onaylı sesi (referans):\n{few_shot}\n" if few_shot else ""
    prompt = f"""{law.text}
{ref}
Aşağıdaki yazı şu maddelerde Damla'nın sesinden SAPMIŞ. SADECE bu ihlalleri düzelt,
geri kalanı bozma, yazıyı sıfırdan yeniden yazma. Yasaya birebir uy.

İhlaller:
{fixlist}

Yazı:
\"\"\"
{draft}
\"\"\"

Düzeltilmiş yazının SADECE kendisini ver, başka açıklama yok."""
    return _call(client, prompt, _REPAIR_MAX_TOKENS)


def law_desc(law: Law, key: str) -> str:
    for i in law.items:
        if i.key == key:
            return i.desc
    return key


# ---------------------------------------------------------------------------
# loop-until-clean — critique → repair → RE-critique, ihlal bitene/max_iter'e kadar
# ---------------------------------------------------------------------------
@dataclass
class Enforcement:
    """Bir taslağa yasa uygulama sürecinin tam kaydı."""

    initial_draft: str
    final_draft: str
    iterations: list[dict] = field(default_factory=list)  # her tur: draft/verdict/skor
    clean: bool = False
    rounds: int = 0

    @property
    def final_verdict(self) -> Verdict | None:
        return self.iterations[-1]["verdict"] if self.iterations else None


def enforce(
    client: anthropic.Anthropic,
    draft: str,
    law: Law = LAW,
    few_shot: str = "",
    max_iter: int = 4,
    on_round: Callable[[int, Verdict], None] | None = None,
    persist: bool = True,
) -> Enforcement:
    """loop-until-clean: yazıyı yasaya sok, temizlenene ya da max_iter'e kadar döngüle.

    Her tur: critique (yargıla) → temizse dur → değilse repair → tekrar. En iyi
    (en az ihlal / en yüksek skor) taslak korunur; son tur bozarsa geri düşmeyiz.
    """
    current = draft
    enf = Enforcement(initial_draft=draft, final_draft=draft)
    best_draft = draft
    best_fails = None  # (fail_sayısı, -skor) küçük daha iyi

    for rnd in range(1, max_iter + 1):
        verdict = critique(client, current, law)
        score = verdict.score(law)
        enf.iterations.append({
            "round": rnd,
            "draft": current,
            "verdict": verdict,
            "score": score,
            "fails": [i.key for i in verdict.fails],
        })
        enf.rounds = rnd
        if on_round:
            on_round(rnd, verdict)

        key = (len(verdict.fails), -score)
        if best_fails is None or key < best_fails:
            best_fails, best_draft = key, current

        if verdict.clean:
            enf.clean = True
            enf.final_draft = current
            break

        if rnd == max_iter:
            # Tükendi; en iyi turu final yap (son tur en kötü olabilir).
            enf.final_draft = best_draft
            break

        current = repair(client, current, verdict, law, few_shot)
    else:
        enf.final_draft = best_draft

    if persist:
        _archive(enf, law)
    return enf


# ---------------------------------------------------------------------------
# Arşiv — her uygulama diske yazılır (altın yakıtı + denetim)
# ---------------------------------------------------------------------------
def _archive(enf: Enforcement, law: Law) -> Path:
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fid = f"{stamp}-{uuid.uuid4().hex[:8]}"
    path = _ARCHIVE_DIR / f"{fid}.json"
    payload = {
        "id": fid,
        "created_at": stamp,
        "model": MODEL,
        "law_source": str(law.source_path),
        "clean": enf.clean,
        "rounds": enf.rounds,
        "initial_draft": enf.initial_draft,
        "final_draft": enf.final_draft,
        "iterations": [
            {
                "round": it["round"],
                "score": it["score"],
                "fails": it["fails"],
                "damla_kokuyor": it["verdict"].smells_like_damla,
                "weakest": it["verdict"].weakest,
                "items": [
                    {"key": iv.key, "pass": iv.passed, "note": iv.note, "parsed": iv.parsed}
                    for iv in it["verdict"].items
                ],
                "draft": it["draft"],
            }
            for it in enf.iterations
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI: bir dosya/metin ver, yasaya sok
# ---------------------------------------------------------------------------
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY yok. export ANTHROPIC_API_KEY=...")
    return anthropic.Anthropic(api_key=key)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        text = Path(sys.argv[1]).read_text(encoding="utf-8")
    elif len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        # Kasıtlı ihlalli örnek (em dash + kanca + CTA) — döngü bunları söksün.
        text = (
            "Here's what nobody tells you about building products — most of them fail. "
            "I did not know why, but I felt it before I could name it. "
            "DM me if you want to know more, thoughts?"
        )

    client = _client()
    print(f"[yasa] kaynak: {LAW.source_path}, {len(LAW.items)} madde")

    def _report(rnd: int, v: Verdict) -> None:
        f = ", ".join(i.key for i in v.fails) or "yok"
        print(f"[tur {rnd}] skor={v.score(LAW):.2f} damla_kokuyor={v.smells_like_damla} ihlal: {f}")

    enf = enforce(client, text, on_round=_report)
    print("=" * 60)
    print(f"temiz mi: {enf.clean}  ({enf.rounds} tur)")
    print("=" * 60)
    print("\n### FİNAL:\n")
    print(enf.final_draft)
