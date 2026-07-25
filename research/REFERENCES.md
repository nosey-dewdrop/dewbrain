# dewbrain — bilimsel kaynaklar (teoriler, tezler, makaleler)

Bu dosya dewbrain'in DAYANDIĞI her teori, makale ve tezi tek yerde toplar. Her
kaynak için: ne diyor, dewbrain'in NERESİNDE kullanıldı, senin ne öğrenmen
gerek. Ezber için değil, savunmak için. SF'de/jüride "bu neye dayanıyor" sorusuna
tek tek cevap.

Okuma sırası önerisi: A (nöroscience temeli) → B (Hopfield/retrieval) →
C (öğrenme/unutma) → D (bellek mimarisi) → E (güven/istatistik). Her başlığın
sonunda **ÖNCE ŞUNU OKU** ile başlangıç makalesi işaretli.

---

## A. NÖROSCIENCE — beynin fonksiyonel mimarisi

dewbrain'in çerçevesi: beyin KONUYA göre değil FONKSİYONA göre bölünür. Her sınıf
bir beyin fonksiyonuna eşlenir (hippocampus=retrieval, neokorteks=bilgi,
prefrontal=karar). Bu analoji değil, mühendislik kararı — bu makaleler temeli.

**A1. Complementary Learning Systems (CLS) teorisi**
- Kim: McClelland, McNaughton, O'Reilly (1995), güncelleme O'Reilly 2011.
- Ne der: beyin İKİ öğrenme sistemi kullanır. Hızlı HIPPOCAMPUS (yeni olayı hemen
  kapar, tek seferde) + yavaş NEOKORTEKS (tekrarla genelleme çıkarır). İkisi
  ayrı çünkü tek sistem olsaydı yeni bilgi eskiyi ezerdi (catastrophic forgetting).
- dewbrain'de: `memory.py`'nin episodik (hızlı/ham) vs semantik (yavaş/damıtılmış)
  ayrımı; `consolidation.py`'nin uyku damıtması. Faz 2'de LoRA=hippocampus,
  base model=neokorteks.
- Sen öğren: neden yeni veri geldiğinde tam fine-tune "sesi öldürür" (bu teorinin
  doğrudan sonucu).

**A2. Hippocampal indexing theory + consolidation**
- Kim: Teyler & DiScenna (1986); güncel: Shift from Hippocampal to Neocortical
  Retrieval, J Neurosci https://www.jneurosci.org/content/29/32/10087 ;
  The Consolidation and Transformation of Memory, Neuron (2015).
- Ne der: hippocampus bir DEPO değil, bir INDEX'tir. Anıyı kendi tutmaz, dağıtık
  neokorteks izlerini yeniden aktive eder. Uyku/dinlenmede tekrarlı yeniden
  aktivasyonla anı yavaşça neokortekse taşınır (episodik → semantik).
- dewbrain'de: `retrieval.py` (index, depo değil), `consolidation.py` (uyku
  damıtması), `memory.py` (episodik→semantik köprü).

**A3. Cognitive maps — hippocampus + entorhinal cortex**
- Ne der: beyin soyut değişkenleri ve görev yapısını harita-formatında kodlar;
  ilişkili kavramlar birbirine yakın konumlanır. AI karşılığı = vektör uzayı /
  ilişkisel graf → transfer + planlama.
- dewbrain'de: embedding uzayı (yakın kavramlar yakın vektör), retrieval'ın temeli.

**A4. AI ↔ beyin bellek yakınsaması (2025-26)**
- Kaynaklar: AI Meets Brain (arXiv:2512.23343), Cognitive Architectures for
  Language Agents (arXiv:2309.02427), Thinking Beyond Tokens: Brain-Inspired to
  Cognitive Foundations for AGI (arXiv:2507.00951).
- Ne der: modern LLM-ajan bellek mimarisi (episodik/semantik/prosedürel) gerçek
  nöroscience'la AYNI yere yakınsıyor. dewbrain ikisini birleştirir.
- **ÖNCE ŞUNU OKU:** A1 (CLS) — her şeyin temeli budur.

---

## B. MODERN HOPFIELD AĞLARI — retrieval'ın kalbi

dewbrain'in retrieval'ı cosine similarity DEĞİL. Modern Hopfield / sparse
attention. Bu, projenin en akademik ve en savunulabilir kısmı (Selim Aksoy'a
gösterilecek). `retrieval.py` + `hopfield_energy.py`.

**B1. Modern Hopfield Networks = Attention** ⭐ EN ÖNEMLİ
- Kim: Ramsauer et al. 2020, "Hopfield Networks is All You Need",
  **arXiv:2008.02217**.
- Ne der: klasik Hopfield ağı ikili örüntüler saklar; MODERN hali sürekli
  örüntüler için üssel saklama kapasitesi verir, TEK adımda geri çağırır. Ve ana
  teorem: modern Hopfield güncelleme kuralı = softmax(β·Xξ)·X = transformer
  ATTENTION ile matematiksel olarak AYNI. Yani attention zaten çağrışımsal-bellek
  geri çağırmasıdır.
- dewbrain'de: `retrieval.py` recall = bu güncelleme; `hopfield_energy.py` enerji
  fonksiyonu E = -(1/β)logsumexp(β·Xξ) + ½‖ξ‖².
- Sen öğren: enerji fonksiyonu, β (inverse temperature) ne yapar, tek-adım
  yakınsama neden mümkün, "attention = associative memory" iddiasının ispatı.
- **ÖNCE ŞUNU OKU.**

**B2. Sparse Hopfield (α-entmax)**
- Kim: Hu et al. 2024, "Sparse and Structured Hopfield Networks",
  **arXiv:2402.13725**.
- Ne der: softmax her örüntüye ufak da olsa ağırlık verir → benzer iki anı
  BLEND olur (metastable state). α-entmax alakasıza TAM SIFIR ağırlık verir
  (Fenchel-Young margin) → blend elenir = pattern separation.
- dewbrain'de: `retrieval.py` `entmax_bisect()`; `hopfield_energy.py`
  `dynamic_alpha()` (α'yı bağlam homojenliğine göre büker).
- Sen öğren: softmax vs entmax farkı, "tam sıfır ağırlık" neden separation demek.

**B3. Hopfield-Fenchel-Young + pattern completion**
- Kim: arXiv:2411.08590.
- Ne der: dense+sparse tek enerji çatısı; SparseMAP ile pattern KOMBİNASYONU
  çağırır (birden çok parçadan tam anı = gerçek completion).
- dewbrain'de: retrieval'ın "tamamlayıcı izler getir" (MMR çeşitlilik) mantığının
  teorik temeli.

**B4. HEN — Hopfield Encoding Networks**
- Kim: arXiv:2409.16408.
- Ne der: ham metni değil, pretrained encoder LATENT'ini sakla. Yüksek-boyutlu
  uzayda ayrılabilirlik artar (cosine'in yapamadığı separation).
- dewbrain'de: sentence-transformers embedding'i saklamak = tam bu.

**B5. Lyapunov kararlılığı (klasik dinamik sistem teorisi)**
- Ne der: bir dinamik sistemin bir enerji (Lyapunov) fonksiyonu her adımda monoton
  azalıyorsa, sistem kararlı bir minimuma yakınsar, kaotik/sahte durumlar üretmez.
- dewbrain'de: `hopfield_energy.py` `stability_report()` — retrieval'ın enerjisinin
  düştüğünü gösterip "sahte anı (spurious state) üretmiyor" ispatı. Ölçüldü:
  spurious 0, self-recall 1.0.
- Sen öğren: Lyapunov fonksiyonu nedir, "monoton azalış = kararlılık" neden.

---

## C. ÖĞRENME + UNUTMA — Faz 2'nin temeli (henüz kodda değil)

Yeni veri geldiğinde eski sesi bozmadan öğrenme. Bugün kodda yok ama world-model
+ fine-tune bunlara dayanacak.

**C1. Catastrophic forgetting**
- Kaynak: arXiv:2112.14146; LLM'de arXiv:2501.13669.
- Ne der: SGD ile sıralı eğitim yeni ağırlığın eskiyi EZMESİNE yol açar. Naif
  "yeni veri geldi, fine-tune et" = önceki bilgiyi/sesi kaybettirir.
- dewbrain'de: neden Faz 2'de tam fine-tune YASAK, LoRA gerekli.

**C2. O-LoRA — Orthogonal Subspace LoRA**
- Kaynak: arXiv:2310.14152.
- Ne der: yeni görevi ORTOGONAL bir alt-uzayda öğren → eski bilgiyle girişim
  minimum. Continual learning için LoRA varyantı.
- dewbrain'de: Faz 2 ses fine-tune yolu (base model=korunan, adapter=yeni).

**C3. Importance-weighted regularization (EWC ailesi)**
- Kaynak: arXiv:2501.13669.
- Ne der: her parametrenin önemini ölç, kritik olanların güncellenmesini kısıtla.
- dewbrain'de: eski sesi koruyan regularizasyon (Faz 2).

---

## D. AJAN BELLEK MİMARİSİ — episodik/semantik/prosedürel

**D1. Generative Agents — memory stream + reflection** ⭐
- Kim: Park et al. 2023, **arXiv:2304.03442**.
- Ne der: bir ajan tüm deneyimi doğal-dil gözlem olarak saklar (memory stream);
  retrieval = relevance + recency + importance (poignancy 1-10 salience);
  periyodik "reflection" ile ham gözlemlerden yüksek-seviye içgörü damıtır.
- dewbrain'de: `sources.py` (memory stream = 1400 iz), `retrieval.py` skoru
  (relevance+recency+salience), `consolidation.py` (reflection = damıtma),
  `memory.py` `salience()`.
- **ÖNCE ŞUNU OKU** (bu bölümde).

**D2. Çok-katmanlı ajan belleği**
- Kaynaklar: arXiv:2603.29194, Episodic-Semantic Memory (arXiv:2605.17625).
- Ne der: working/episodik/semantik ayrımı → cross-session drift kontrolü,
  bounded context. Her katmanın retrieval semantiği farklı.
- dewbrain'de: memory.py'nin katman ayrımı + brain.py'nin iki-katmanlı retrieval
  (gold voice + episodik context).

**D3. ExpeL — deneyimden kural (parametre güncellemesiz)**
- Ne der: ham deneyimden add/edit/vote ile semantik kural çıkar, model
  ağırlıklarını değiştirmeden.
- dewbrain'de: `memory.consolidate()` (onaylı çıktı → grown gold, ExpeL-add).

---

## E. GÜVEN + İSTATİSTİK — "kanıtla, iddia etme"nin matematiği

Damla'nın en keskin yasasının (kanıtla-iddia-etme) matematiksel hali.
`metacognition.py` + `conformal.py`.

**E1. Conformal Prediction** ⭐ EN ÖNEMLİ (bu bölümde)
- Kim: Vovk, Gammerman, Shafer, "Algorithmic Learning in a Random World" (kitap);
  pratik giriş: Angelopoulos & Bates, "A Gentle Introduction to Conformal
  Prediction" (arXiv:2107.07511).
- Ne der: HERHANGİ bir skorlayıcı için, dağılım varsayımı OLMADAN, sonlu örnekte
  "gerçek etiket %(1-ε) olasılıkla tahmin kümesinde" garantisi. Nasıl: kalibrasyon
  setindeki nonconformity skorlarının (n+1)(1-ε)/n empirik kuantili = eşik.
- dewbrain'de: `conformal.py` — güveni keyfi yüzdeden istatistiksel eşiğe çevirir,
  geçmezse ABSTAIN. `worldmodel.py` alan-dışı kararda çekimser kalma.
- Sen öğren: nonconformity score, empirical quantile, coverage garantisi,
  exchangeability varsayımı (ve dewbrain'in n=9-47'de bunu neden "dürüstçe" söyler).
- **ÖNCE ŞUNU OKU:** arXiv:2107.07511 (Gentle Introduction) — kitaptan önce.

**E2. RUBRIC-ARROW — rubrikten öğrenilebilir reward**
- Kaynak: arXiv:2605.29156.
- Ne der: bir rubrik (do/don't checklist) → öğrenilebilir POINTWISE reward model;
  rubrik-koşullu judge her kritere binary aₖ∈{0,1} verir, ağırlıklı ortalama = skor.
- dewbrain'de: `law.py` (12 omurga + 4 mekanik = rubrik), `prefrontal.py` (judge),
  `metacognition.py` (ağırlıklı pass-rate). Faz 2'de prompt-judge → trained reward.

**E3. Dual-process teorisi (System 1 / System 2)**
- Kim: Kahneman, "Thinking, Fast and Slow".
- Ne der: hızlı sezgisel System 1 + yavaş analitik System 2. Tanıdık → hızlı,
  yeni/riskli → yavaş.
- dewbrain'de: `router.py` — cue'ya göre hızlı/yavaş yol seçer.

---

## OKUMA PLANI (bir akşam + sonrası)

**Bu akşam (temel):**
1. B1 Ramsauer 2020 (Hopfield=attention) — projenin kalbi.
2. D1 Generative Agents (memory stream) — bellek mantığı.
3. E1 Angelopoulos Gentle Intro (conformal) — güvenin matematiği.

**Sonra (derinlik):**
4. A1 CLS (McClelland) — neden iki bellek sistemi.
5. B2 Sparse Hopfield (entmax) — separation.
6. B5 Lyapunov — kararlılık ispatı.

**İleri (Faz 2 geldiğinde):**
7. C1-C3 (forgetting + LoRA + EWC), E2 (RUBRIC-ARROW).

Her makaleyi okurken tek soru sor: "bu, dewbrain'in HANGİ sınıfında yaşıyor?"
Cevabı yukarıda. Böylece teori soyut kalmaz, kendi kodunda görürsün.
