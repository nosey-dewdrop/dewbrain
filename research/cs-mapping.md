# nöroscience prensibi → gerçek CS mekanizması (kanıtlı)

deep-research çıktısı (24 Tem, 26 kaynak, 123 iddia, 18 doğrulanmış çoğu 3-0 oy). Metafor DEĞİL, her eşleme kaynakla. Sentez adımı session limitine takıldı, ham doğrulanmış bulgular burada damıtıldı.

UYARI: aşağıda "doğrulanamadı" işaretli olanlar YANLIŞ değil — doğrulayıcı ajanlar session limitine (23:20 reset) takılıp çekimser kaldı (0-0 abstain). Reset sonrası teyit edilecek, fikir sağlam görünüyor.

---

## (a) pattern separation + completion → MODERN HOPFIELD / SPARSE HOPFIELD

Nöroscience: hippocampus benzer anıları AYIRT eder (separation) + yarım ipucundan TAMI çağırır (completion).

Gerçek CS (doğrulanmış 3-0):
- **Modern Hopfield network** (Ramsauer 2020, arXiv:2008.02217): boyutta ÜSSEL kapasite, TEK adımda retrieval, üssel küçük hata. Ve KRİTİK: yeni update kuralı transformer ATTENTION ile matematiksel olarak AYNI şey. Yani attention zaten bir çağrışımsal-bellek retrieval'ı. Bizim retrieval'ımız cosine similarity değil, Hopfield/attention tabanlı olacak.
- **Sparse Hopfield** (alpha-entmax/normmax, arXiv:2402.13725): TAM retrieval (attractor = saklanan pattern birebir), Fenchel-Young loss margin'i varsa sıfır hata. "Metastable states" (birden çok pattern'i KARIŞTIRAN durumlar) elenir → benzer iki dewthinks'i BLEND etmeden ayırır. Bu tam pattern separation.
- **Hopfield-Fenchel-Young** (arXiv:2411.08590): dense+sparse tek enerji çatısı, SparseMAP ile pattern KOMBİNASYONU çağırır (birden çok parçadan tam anı kurar = gerçek completion).
- **HEN / Hopfield Encoding Networks** (arXiv:2409.16408): saklamadan önce pretrained encoder ile latent uzaya gömer, çağırırken decode eder. Ayrılabilirliği yüksek-boyutlu uzaya taşıyarak artırır → cosine'in yapamadığı separation.

Çekirdek ders: retrieval = yüksek-boyut ayrılabilirlik + sparse Hopfield. Cosine similarity YETMEZ (Damla'nın benzer yazıları karışır).

## (b) complementary learning systems + catastrophic forgetting → LoRA ADAPTER + IMPORTANCE-REG + ADAPTIVE REPLAY

Nöroscience: hızlı hippocampus (yeniyi kapar) + yavaş neocortex (genellemeyi korur). Problem: yeni verified eklerken eski ses BOZULMASIN.

Gerçek CS (doğrulanmış 3-0):
- Problem gerçek: SGD ile sıralı eğitim = catastrophic forgetting, yeni ağırlık eskiyi EZER (arXiv:2112.14146). LLM fine-tune pretrain bilgisini kaybettirir (arXiv:2501.13669). Yani naif "yeni veri geldi fine-tune et" = sesi öldürür.
- **O-LoRA / Orthogonal Subspace LoRA** (arXiv:2310.14152): yeni görevi ORTOGONAL alt-uzayda öğrenir, girişim/unutma azalır. Continual learning için LoRA varyantı.
- **Importance-weighted regularization / EWC ailesi** (arXiv:2501.13669): her parametrenin önemini ölçer, genel bilgi için kritik parametrelerin güncellenmesini KISITLAR, yeni göreve cross-entropy ile uyar. Eski sesi koruyan regularizasyon.
- **Adaptive Memory Replay** (IBM): geçmiş veriden hangisini tekrar oynatacağını multi-armed bandit + Boltzmann sampling ile SEÇER (uniform değil). Unutmayı %10'a kadar azaltır, ek maliyet yok.

Çekirdek ders: yeni verified eklerken TAM fine-tune YASAK. LoRA adapter (tercihen ortogonal) + eski altını replay. Hippocampus=adapter (hızlı/yeni), neocortex=base model (yavaş/korunan).

## (c) replay + salience consolidation → MEMORY STREAM + REFLECTION (teyit bekliyor)

Nöroscience: hippocampal replay + önem-ağırlıklı konsolidasyon, episodik ham → semantik kural.

Gerçek CS:
- **Memory stream** (Generative Agents, arXiv:2304.03442, doğrulandı 2-0): tüm deneyim doğal-dil gözlem olarak saklanır (ham episodik kayıt). Bu bizim ham dewrites/dewthinks havuzumuz.
- Doğrulanamadı (limit, YANLIŞ değil): reflection (LLM periyodik olarak ham gözlemlerden yüksek-seviye içgörü damıtır), poignancy 1-10 salience skoru, ExpeL (add/edit/vote ile ham deneyimden semantik kural, parametre güncellemesiz). Bunlar consolidation adayı, reset sonrası teyit.

Çekirdek ders: ham havuz = memory stream, damıtma = reflection benzeri bir adım (ham dewrites → yasa örüntüsü). Teyit edilecek.

## (d) yasa → öğrenilebilir reward/critic → RUBRIC REWARD + DPO (teyit bekliyor)

Nöroscience: prefrontal öz-eleştiri, sapmayı cezalar.

Gerçek CS:
- Doğrulanamadı (limit, YANLIŞ değil): RUBRIC-ARROW (arXiv:2605.29156) — rubrik (do/don't checklist) → öğrenilebilir POINTWISE reward model, rubrik-koşullu judge her kritere binary a_k ∈{0,1} verir (yasanın her maddesi ayrı puanlanır), ağırlıklı ortalama = skor. Bu tam bizim 12 omurga + 10 kapı'nın öğrenilebilir critic'e dönüşü. Reset sonrası teyit + oku.
- Not: az örnekte DPO / RLAIF / constitutional AI de aday, araştırma açıları taradı, tam eşleme reset sonrası netleşecek.

Çekirdek ders: yasa (12 omurga) → başta PROMPT-tabanlı rubrik-judge (her madde binary), veri büyüyünce öğrenilebilir reward model. Bu prefrontal katman.

## (e) few-shot style (tiny corpus, generic'e kaçmadan) → STYLE EMBEDDING + LoRA (kaynaklar var, teyit bekliyor)

Kaynaklar geldi (LUAR authorship embedding github.com/LLNL/LUAR, Bit-LoRA style personalization, contrastive style, fine-tune-vs-prompt tradeoff blogları) ama iddia doğrulaması limite takıldı. Reset sonrası bu açı öncelikli okunacak — çünkü BUGÜNKÜ durum tam bu (~8 altın).

---

## NE ÖNCE (8 altınla bugün) vs NE AÇILIR (100+)

**Bugün (~8 verified, fine-tune İMKANSIZ, veri yok):**
- retrieval = sparse Hopfield / attention-tabanlı çağrışımsal bellek, cosine DEĞİL (separation için).
- yasa = prompt-tabanlı rubrik-judge (12 omurga her madde binary kontrol) + öz-eleştiri döngüsü.
- ham havuz = memory stream (episodik), verified = semantik altın.
- fine-tune YOK, RAG + retrieval + rubrik-critic ile üretim.

**100+ altın olunca AÇILIR (Faz 2, GPU):**
- LoRA adapter (ortogonal) ile ses fine-tune, importance-reg + adaptive replay ile eski sesi koruyarak.
- yasa → öğrenilebilir reward model (RUBRIC-ARROW / DPO), prompt-judge yerine trained critic.
- consolidation → reflection ile ham→semantik otomatik damıtma.

## kaynaklar (primary)
- 2008.02217 modern Hopfield = attention · 2402.13725 sparse Hopfield · 2411.08590 Hopfield-Fenchel-Young · 2409.16408 HEN
- 2112.14146 catastrophic forgetting · 2501.13669 importance-reg · O-LoRA 2310.14152 · IBM adaptive memory replay
- 2304.03442 generative agents memory stream · ExpeL · 2605.29156 RUBRIC-ARROW · LUAR authorship · Bit-LoRA style
- devam: 2607.19219, 2605.08061, 2602.13576, 2410.12757, 2409.04574, 2411.00027 (reset sonrası oku)

## SIRADAKI (reset 23:20 sonrası)
Doğrulanamayan (d)(e)(c) açılarını tekrar doğrula/oku — özellikle (e) few-shot style çünkü bugünkü durum o. Sonra ilk kesit mimarisini bu bulgulara göre kilitle.
