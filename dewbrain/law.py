"""
prefrontal katmanı — YASA (tek kaynak).

Bu modül Damla'nın ses yasasını TEK KAYNAKTAN türetir: icerik/dewrites.md
başındaki "DAMLA'NIN SESİ + FORMAT — DEWRITES YASASI" bölümü. O bölüm hem insan-
okunur yasa metnini (üretim promptu için) hem de makine-okunur rubriği (RUBRIC-ARROW
judge için, her madde binary a_k) besler.

NEDEN parse, neden gömülü kopya değil (kök sebep, yama değil):
Eski law.py yasayı elle paraphrase edip gömüyordu. verified.md / CLAUDE.md / law.py
üç ayrı yerde aynı yasanın üç ayrı sürümü vardı; biri değişince ötekiler kayardı
(iki-kaynak drift). Çözüm senkron scripti değil, TEK KAYNAK: yasa dewrites.md'de
yaşar, buradan PARSE edilir. dewrites.md değişince yasa otomatik değişir; burada
kopya YOK, senkron edilecek bir şey YOK.

RUBRIC-ARROW (arXiv:2605.29156): rubrik pointwise reward'a çevrilir; her madde
rubrik-koşullu judge tarafından binary skorlanır (a_k in {0,1}), ağırlıklı ortalama
skor olur. Bu modül rubriği (madde listesi + ağırlık) verir; skorlama prefrontal.py'de.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Tek kaynak: yasanın yaşadığı dosya + bölüm başlığı.
# ---------------------------------------------------------------------------
LAW_SOURCE = Path.home() / "damla_projects_2026" / "icerik" / "dewrites.md"

# Bölüm başlığını yakalayan desen. Başlık metni değişse bile "DEWRITES YASASI"
# çapası kalırsa bulunur.
_SECTION_HEADER = re.compile(r"^##[^\n]*DEWRITES YASASI[^\n]*$", re.MULTILINE)

# Alt bölüm başlıkları (A/B/C/D). Numara + nokta esnek.
_SUBHEAD = re.compile(r"^###\s*([A-D])\.\s*(.+?)\s*$", re.MULTILINE)

# A bölümü omurga maddesi: "1. **Başlık.** açıklama..."
_OMURGA_ITEM = re.compile(
    r"^(?P<num>\d+)\.\s+\*\*(?P<title>.+?)\*\*\s*(?P<rest>.*?)(?=^\d+\.\s+\*\*|\Z)",
    re.MULTILINE | re.DOTALL,
)

# B bölümü mekanik maddesi: "- **Başlık.** açıklama..."
_MEKANIK_ITEM = re.compile(
    r"^-\s+\*\*(?P<title>.+?)\*\*\s*(?P<rest>.*?)(?=^-\s+\*\*|\Z)",
    re.MULTILINE | re.DOTALL,
)


class LawSourceError(RuntimeError):
    """Yasa kaynağı bulunamadı/parse edilemedi. Sessiz drift YASAK: tek kaynaktan
    türetilemezse motor DURUR, gömülü kopyaya düşmez."""


@dataclass(frozen=True)
class RubricItem:
    """Rubriğin tek maddesi. RUBRIC-ARROW'da her madde binary bir a_k üretir."""

    key: str  # makine anahtarı (kararlı, prompt/JSON eşlemesi için)
    title: str  # Damla'nın yasasındaki başlık
    desc: str  # judge'a verilen kontrol tanımı (ihlal ölçütü)
    kind: str  # "omurga" | "mekanik"
    weight: float = 1.0  # ağırlıklı ortalamadaki payı


@dataclass(frozen=True)
class Law:
    """Tek kaynaktan türetilmiş yasa. Değişmez (frozen) — çalışırken mutasyona uğramaz."""

    text: str  # üretim promptuna gömülen tam yasa metni (verbatim, Damla'nın dili)
    items: tuple[RubricItem, ...]  # 12 omurga + 4 mekanik
    gate: tuple[str, ...]  # C bölümü 10 soruluk çıkış kapısı
    source_path: Path

    @property
    def omurga(self) -> list[RubricItem]:
        return [i for i in self.items if i.kind == "omurga"]

    @property
    def mekanik(self) -> list[RubricItem]:
        return [i for i in self.items if i.kind == "mekanik"]

    def total_weight(self) -> float:
        return sum(i.weight for i in self.items)


# ---------------------------------------------------------------------------
# Kararlı anahtarlar. Yasadaki başlık metni Damla'nın; JSON/prompt eşlemesi için
# başlıktan bağımsız kararlı bir key gerekir. Sıraya göre eşlenir (yasa sırası sabit).
# Yeni madde eklenirse burada da bir anahtar eklenir (aksi halde otomatik türetilir).
# ---------------------------------------------------------------------------
_OMURGA_KEYS = [
    "acilis_duz_kanca_yok",
    "itiraf_var_magduriyet_yok",
    "tek_tasinabilir_tez",
    "karsitlik_metafor_degil",
    "insani_muhur",
    "idol_kanit_degil_sus",
    "kapanis_durus_degil_cta",
    "brag_sahte_tevazu_yok",
    "uydurma_canli_ozellik_yok",
    "rakip_gomme_yok",
    "konu_neyse_ses",
    "teknikse_sayi_hook",
]
_MEKANIK_KEYS = [
    "em_dash_iki_nokta_yok",
    "akan_paragraf_liste_yok",
    "ovunme_terim_sayi",
    "amac_once_urun_sonra",
]

# Ağırlık: mekanik ihlaller (em dash, iki nokta) SIFIR-tolerans, tek başına yazıyı
# öldürür → yüksek ağırlık. Omurga maddeleri eşit taban. Reçete DEĞİL sınır: skor
# bir kalite göstergesi, geçme kapısı "sıfır ihlal" (aşağıda passes()).
_WEIGHTS: dict[str, float] = {
    "em_dash_iki_nokta_yok": 3.0,
    "akan_paragraf_liste_yok": 2.0,
    "uydurma_canli_ozellik_yok": 2.0,
    "rakip_gomme_yok": 2.0,
    "acilis_duz_kanca_yok": 1.5,
    "kapanis_durus_degil_cta": 1.5,
}


def _clean(s: str) -> str:
    """Çok satırlı açıklamayı tek satıra indir, iç boşlukları topla."""
    return re.sub(r"\s+", " ", s).strip()


def _derive_key(prefer: list[str], idx: int, title: str) -> str:
    """Sıraya göre kararlı anahtar; liste biterse başlıktan türet."""
    if idx < len(prefer):
        return prefer[idx]
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or f"item_{idx}"


def _extract_section(text: str) -> str:
    """Yasa bölümünü (başlıktan bir sonraki '---' ayracına kadar) kes."""
    m = _SECTION_HEADER.search(text)
    if not m:
        raise LawSourceError(
            f"'DEWRITES YASASI' bölümü {LAW_SOURCE} içinde bulunamadı. "
            "Yasa tek kaynaktan türetilir; bölüm başlığı taşınmış/silinmiş olabilir."
        )
    start = m.start()
    # Bölümü kapatan ilk yatay ayraç ('\n---\n') ya da dosya sonu.
    rest = text[m.end():]
    end_rel = rest.find("\n---")
    section = rest if end_rel == -1 else rest[:end_rel]
    return section


def _parse_section(section: str) -> tuple[list[RubricItem], list[str]]:
    """A (omurga) + B (mekanik) + C (kapı) alt bölümlerini ayrıştır."""
    # Alt bölüm sınırlarını bul.
    heads = list(_SUBHEAD.finditer(section))
    blocks: dict[str, str] = {}
    for i, h in enumerate(heads):
        letter = h.group(1)
        body_start = h.end()
        body_end = heads[i + 1].start() if i + 1 < len(heads) else len(section)
        blocks[letter] = section[body_start:body_end]

    if "A" not in blocks:
        raise LawSourceError("Yasa bölümünde 'A.' (omurga) alt başlığı yok.")
    if "B" not in blocks:
        raise LawSourceError("Yasa bölümünde 'B.' (mekanik) alt başlığı yok.")

    items: list[RubricItem] = []

    for idx, m in enumerate(_OMURGA_ITEM.finditer(blocks["A"])):
        title = _clean(m.group("title"))
        desc = _clean(m.group("rest"))
        key = _derive_key(_OMURGA_KEYS, idx, title)
        items.append(
            RubricItem(
                key=key,
                title=title,
                desc=desc or title,
                kind="omurga",
                weight=_WEIGHTS.get(key, 1.0),
            )
        )
    if not items:
        raise LawSourceError("A bölümünden hiç omurga maddesi çıkarılamadı (format değişmiş).")

    for idx, m in enumerate(_MEKANIK_ITEM.finditer(blocks["B"])):
        title = _clean(m.group("title"))
        desc = _clean(m.group("rest"))
        key = _derive_key(_MEKANIK_KEYS, idx, title)
        items.append(
            RubricItem(
                key=key,
                title=title,
                desc=desc or title,
                kind="mekanik",
                weight=_WEIGHTS.get(key, 1.0),
            )
        )

    # C bölümü: 10 soruluk kapı (varsa). Yoksa boş — omurga+mekanik zaten kapı.
    gate: list[str] = []
    if "C" in blocks:
        for gm in re.finditer(r"^\d+\.\s+(.+?)\s*$", blocks["C"], re.MULTILINE):
            gate.append(_clean(gm.group(1)))

    return items, gate


def load_law(source: Path | None = None) -> Law:
    """Yasayı tek kaynaktan (dewrites.md) türet. Kaynak yok/bozuksa DURUR."""
    path = source or LAW_SOURCE
    if not path.exists():
        raise LawSourceError(
            f"Yasa kaynağı yok: {path}. dewbrain yasasını buradan türetir; "
            "gömülü kopya bilerek tutulmuyor (drift önlemi)."
        )
    text = path.read_text(encoding="utf-8")
    section = _extract_section(text)
    items, gate = _parse_section(section)

    # Üretim promptuna gömülen yasa metni = bölümün kendisi (verbatim, Damla'nın dili).
    # Paraphrase YOK; ne yazdıysa o. Başlığı da içerir ki bağlam net olsun.
    header = _SECTION_HEADER.search(text)
    law_text = text[header.start(): header.start() + len(section) + (header.end() - header.start())]
    law_text = law_text.strip()

    return Law(
        text=law_text,
        items=tuple(items),
        gate=tuple(gate),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Modül yüklenirken bir kez türet; import edenler hazır Law nesnesi alır.
# Türetilemezse import PATLAR (sessiz gömülü-kopyaya düşme YOK).
# ---------------------------------------------------------------------------
LAW: Law = load_law()


def _sanitize(s: str) -> str:
    """K1 kök çözümü: yasanın KENDİSİ em dash + iki nokta içeremez, çünkü bu metin
    üretim promptuna gömülüyor ve modele 'asla em dash kullanma' derken em dash saçmak
    çelişkili sinyal + few-shot kontaminasyonu (model prompttaki — 'ı normalize eder).
    Em dash -> nokta boşluk, cümle-bağlayan iki nokta -> nokta. Damla yasasına birebir."""
    s = s.replace(" — ", ". ").replace("—", ". ")
    s = re.sub(r":\s+", ". ", s)  # ": " (yasak) -> ". "
    s = re.sub(r"[.\s]*\.[.\s]*\.[.\s]*", ". ", s)  # ardışık nokta/boşluk kümesi -> tek ". "
    return s.strip()


def prompt_law() -> str:
    """Üretim/onarım promptuna gömülen TEMİZ yasa metni. LAW.text'i ham gömmek yerine
    (o başlık-formatlı meta-doküman, em dash + iki nokta dolu) yasayı parse edilmiş
    madde desc'lerinden yeniden kurar. Tek kaynak korunur (desc'ler dewrites.md'den),
    ama çıktı yasanın kendi kurallarına uyar."""
    lines = ["DAMLA'NIN SESİ (çıkılmaz yasa, birebir uy):"]
    for it in LAW.items:
        title = _sanitize(it.title).rstrip(". ")  # title zaten '.' ile bitebilir, çift nokta olmasın
        desc = _sanitize(it.desc)
        lines.append(f"- {title}. {desc}")
    return "\n".join(lines)


# Geriye dönük uyumluluk + K1 düzeltmesi: YASA_METNI artık TEMİZ prose (ham LAW.text değil).
YASA_METNI: str = prompt_law()
RUBRIK: list[tuple[str, str]] = [(i.key, i.desc) for i in LAW.items]
# RUBRIC-ARROW: her maddenin ağırlığı (brain best-draft seçimi ağırlıklı kullanır).
WEIGHTS: dict[str, float] = {i.key: i.weight for i in LAW.items}


if __name__ == "__main__":
    law = LAW
    print(f"kaynak: {law.source_path}")
    print(f"toplam madde: {len(law.items)} (omurga {len(law.omurga)} + mekanik {len(law.mekanik)})")
    print(f"toplam ağırlık: {law.total_weight():.1f}")
    print(f"çıkış kapısı: {len(law.gate)} soru\n")
    for i in law.items:
        print(f"[{i.kind:7} w={i.weight:.1f}] {i.key}")
        print(f"    {i.desc[:100]}")
    print(f"\nyasa metni ilk 200 karakter:\n{law.text[:200]}...")
