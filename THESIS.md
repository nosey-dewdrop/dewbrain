# dewbrain — bilimsel tez ve çalışma dokümanı

Bu doküman dewbrain'in NE OLDUĞUNU, hangi bilimsel temele dayandığını, her sınıfın
ne yaptığını ve senin (Damla) neyi öğrenmen gerektiğini anlatır. Amaç bancılık
değil: bu sistemi bir jürinin, bir hocanın (Selim Aksoy), bir YC ortağının
karşısında SEN savunacaksın. O yüzden burada her iddia bir kaynağa, her sınıf bir
mekanizmaya bağlıdır. Ezber değil, anlama.

---

## 0. TEZ CÜMLESİ (tek cümlede ne)

> Bir insanın ürettiği tüm izlerden (yazı, karar, rapor, fikir) o insanın
> **fonksiyonel** bilişsel mimarisini taklit eden, modern çağrışımsal-bellek
> (Hopfield/attention) üstüne kurulu, **kendi çıktısını istatistiksel olarak
> reddedebilen** (conformal ABSTAIN) bir karar-çekirdeği.

Fal değil (uydurma olasılık yok). Wrapper değil (RAG+prompt değil, gerçek NN
retrieval + öğrenilebilir mimari). Whole-brain-emulation değil (nöron simülasyonu
imkansız, connectome yok). **Arada:** beynin FONKSİYONUNU (hangi bölge ne iş
yapar) gerçek CS mekanizmalarıyla taklit eder, maddesini değil.

Moat = mekanizma değil (bilim herkese açık), **birleşim**: gerçek nöromimari ×
Damla'nın verisi × kimsenin kurmadığı entegrasyon. Kod public, veri gizli (tıpkı
bir LLM'in eğitim kodu açık, eğitim verisi kapalı olması gibi).

---

## 1. BİLİMSEL TEMEL — üç sütun

dewbrain üç literatürün kesişimidir. Her birini öğrenmen gerekir:

### A. Nöroscience: beyin FONKSİYONA göre bölünür (konuya göre değil)
- Hippocampus bir DEPO değil, bir INDEX'tir. Anıyı saklamaz, dağıtık neokorteks
  izlerini yeniden aktive eder. → retrieval katmanımız.
- Consolidation: yeni anı önce hippocampus'a bağımlı, uyku/dinlenmede tekrarlı
  yeniden-aktivasyonla neokortekse taşınır (episodik → semantik). → uyku katmanı.
- Prefrontal korteks: karar + öz-eleştiri, feedback'ten öğrenir (RL). → critique.
- **Öğren:** Complementary Learning Systems teorisi (McClelland 1995), hippocampal
  indexing theory. Kaynaklar: `research/brain-architecture.md`.

### B. Modern Hopfield ağları = attention (çağrışımsal bellek)
- Klasik Hopfield ağı: örüntüleri enerji minimumları olarak saklayan bir sinir
  ağı. Yarım bir ipucundan tam anıyı geri çağırır (pattern completion).
- **Modern** Hopfield (Ramsauer 2020): üssel saklama kapasitesi, TEK adımda
  retrieval, ve KRİTİK bulgu — güncelleme kuralı transformer ATTENTION ile
  matematiksel olarak AYNI. Yani attention zaten bir çağrışımsal-bellek geri
  çağırmasıdır.
- Sparse Hopfield (α-entmax, Hu 2024): softmax yerine α-entmax → alakasız
  örüntülere TAM SIFIR ağırlık → "metastable blend" (iki anının bulanık
  ortalaması) elenir = pattern separation.
- **Öğren:** softmax, logsumexp, enerji fonksiyonu, Lyapunov kararlılığı, entmax.
  Kaynaklar: `research/cs-mapping.md`, arXiv:2008.02217, arXiv:2402.13725.

### C. Conformal prediction = dağılımdan-bağımsız güven garantisi
- Sıradan bir güven skoru ("%88 eminim") bir modelin sezgisidir, garanti değil.
- Conformal prediction (Vovk): kalibrasyon setindeki hatalara bakarak, HERHANGİ
  bir skorlayıcı için, dağılım varsayımı olmadan, sonlu örnekte "gerçek etiket
  %(1-ε) olasılıkla tahmin kümesinde" garantisi verir.
- Bizde: "bu çıktı Damla'nın izleriyle istatistiksel tutarlı mı" sorusunu keyfi
  yüzdeden matematiksel eşiğe çevirir. Geçemezse ABSTAIN.
- **Öğren:** nonconformity score, empirical quantile, exchangeability varsayımı,
  p-value. Kaynak: Vovk, Gammerman, Shafer "Algorithmic Learning in a Random World".

---

## 2. SINIFLAR — her modül ne yapar, nasıl konuşur

Sistem `dewbrain/` altında 15 Python modülü. Fonksiyonel beyin bölgelerine eşlenir.

### VERİ KATMANI (beynin girdisi)

**`memory.py` — episodik/semantik bellek ayrımı**
- Ne yapar: Damla'nın içerik dosyalarını (dewrites/dewthinks/dewlog/dewideo)
  parse eder. İki tür: EPISODIC (ham izler, ne yaşandı) ve SEMANTIC (verified
  gold, onaylı ses). Kod-fence farkında, sağlam bir durum-makinesi parser'ı.
- Nasıl konuşur: `load_gold()` → 9 onaylı ses örneği; `load_raw()` → 237 ham iz;
  `salience()` → bir metnin "ne kadar akılda kalıcı" skoru (1-10, konsolidasyon
  kapısı).
- Bilim: memory-stream + reflection (Generative Agents, arXiv:2304.03442).

**`decisions.py` — karar-belleği (world-model yakıtı)**
- Ne yapar: dağınık `DECISIONS.md` dosyalarını (iki farklı format: stitchu inline,
  gymgyme section) parse eder. Her karar: seçim / neden / bağlam / geri-alma
  maliyeti / sonuç. Bu, "Damla ne yapardı"nın verisi.
- Nasıl konuşur: `load_decisions()` → 45 yapılandırılmış karar; her biri bir
  `Decision` nesnesi.

**`sources.py` — kaynak-kayıt (beynin tüm kanalları)**
- Ne yapar: beyin tek dosyadan beslenmez. Bir REGISTRY: writing + decision +
  report + projectdoc + idea + desktop = ~1400 episodik iz. Yeni kanal eklemek =
  registry'ye bir satır, kod değil.
- Nasıl konuşur: `load_all()` → tüm izler; `channel_counts()` → kanal başına sayı.
- Mühendislik dersi: kanalı ayrı loader'a gömmek kırılgan; registry pattern doğru
  soyutlama. Her iz body-cap'li (>4000 char embed edilemez) ve min-length filtreli.

### RETRIEVAL KATMANI (hippocampus)

**`retrieval.py` — sparse modern Hopfield (çağrışımsal bellek)**
- Ne yapar: bir cue (yarım fikir) verilince en ilgili izleri getirir. Cosine
  DEĞİL: α-entmax(β·Xq) ile sparse Hopfield. β korpustan kalibre edilir (sabit
  değil). Getirmede MMR çeşitliliği (iki near-duplicate getirme) + recency +
  salience prior.
- Nasıl konuşur: `HippocampalRetrieval(entries).recall(cue, k=3)` → k ilgili iz,
  her biri {weight, sim, score}. `separability` → korpusun ne kadar ayrılabilir
  olduğunun ölçümü (0.88 ölçüldü).
- Dürüst sınır: 9 gold homojen (cosine 0.03-0.25 dar bant), entmax tek başına
  ayıramaz; o yüzden skor + çeşitlilik + salience birlikte. Overclaim yok.

**`hopfield_energy.py` — enerji + Lyapunov kararlılık + dinamik α (akademik omurga)**
- Ne yapar: retrieval'ın enerji fonksiyonunu, iteratif yakınsamasını ve
  kararlılığını ANALİZ eder. E = -(1/β)logsumexp(β·Xξ) + ½‖ξ‖². Lyapunov testi:
  enerji her adımda düşer mi (düşmezse "sahte anı" = spurious, bayrak kaldırır).
- Nasıl konuşur: `stability_report()` → {all_lyapunov_ok, self_recall_rate,
  mean_iterations}. Ölçüldü: 9/9 gold self-recall 1.0, spurious 0, ortalama 3.4
  iterasyon. `dynamic_alpha()` → bağlam homojenliğine göre α (homojen → yüksek α,
  daha keskin ayırma; dış öneri bunu TERS yapmıştı, düzeltildi).
- Selim Aksoy'a gösterilecek: enerji-vs-iterasyon ve separability-vs-α grafikleri.

### YASA + ÖZ-ELEŞTİRİ KATMANI (prefrontal)

**`law.py` — ses yasası (tek kaynaktan)**
- Ne yapar: Damla'nın ses yasasını TEK kaynaktan (dewrites.md) parse eder. 12
  omurga + 4 mekanik madde. Gömülü kopya YOK (drift önlemi): dewrites.md değişince
  yasa otomatik değişir.
- Nasıl konuşur: `LAW.items` → 16 rubrik maddesi; `YASA_METNI` → üretim promptuna
  gömülen temiz yasa; `WEIGHTS` → her maddenin ağırlığı (em-dash w=3, ağır).
- Bilim: RUBRIC-ARROW (arXiv:2605.29156) — rubrik → pointwise reward, her madde
  binary aₖ∈{0,1}.

**`prefrontal.py` — RUBRIC judge + repair (öz-eleştiri döngüsü)**
- Ne yapar: bir taslağı yasaya karşı denetler (her madde pass/fail), ihlalleri
  hedefli düzeltir, temizlenene kadar döngüler (loop-until-clean). Bozuk JSON'da
  ÇÖKMEZ (fail-closed: okunamayan madde "ihlal" sayılır, sessiz geçiş yok).
- Nasıl konuşur: `enforce(client, draft)` → `Enforcement` (her tur draft+verdict).

### KARAR + GÜVEN KATMANI (meta-biliş)

**`metacognition.py` — kalibre güven + ABSTAIN**
- Ne yapar: bir çıktının güvenini dört sinyalden hesaplar (law-pass, voice-fit,
  grounding, effort). Sıfır-ihlal + yüksek güven yoksa "bu sensin" DEMEZ. Üç
  hüküm: sure / tentative / abstain.
- Nasıl konuşur: `assess(verdict, ...)` → `Confidence(score, verdict, reasons)`.
- Bilim = Damla'nın "kanıtla, iddia etme" yasasının kodu.

**`conformal.py` — istatistiksel garanti (conformal prediction)**
- Ne yapar: metacognition'ın keyfi yüzdesini istatistiksel garantiye çevirir.
  Kalibrasyon = bilinen-Damla izleri (gold). Nonconformity = kimlik-manifolduna
  uzaklık. Eşik = (n+1)(1-ε)/n empirik kuantil. Test dışındaysa ABSTAIN.
- Nasıl konuşur: `build_identity_conformal()` → gate + centroid; `gate.test(prox)`
  → {inside, p_value, empirical_coverage}. Ölçüldü: kurumsal çöp ABSTAIN,
  gerçek gold geçer, ampirik kapsama 0.91 (hedef 0.90).
- Dürüstlük: garanti marjinal ve exchangeability varsayar; n=9-47'de bunu SÖYLER,
  sert %95 iddia etmez.

### ORKESTRASYON KATMANI (vmPFC)

**`router.py` — dual-process (System 1/2)**
- Ne yapar: cue'ya göre çaba seçer. Tanıdık + düşük-risk → hızlı yol (az tur).
  Yeni veya yüksek-risk (karar, fikir) → yavaş yol (çok tur, geniş bağlam, yüksek
  abstain eşiği). Sabit boru hattı değil, orkestrasyon.
- Nasıl konuşur: `route(channel, gold_hits, context_hits)` → `Route(path, max_iter,
  k, abstain_floor, sure_bar)`.
- Bilim: Kahneman dual-process; familiarity retrieval top-mass'tan, stakes kanaldan.

**`brain.py` — vmPFC koordinatör (tam döngü)**
- Ne yapar: hepsini bağlar. recall(gold voice + episodik context) → route →
  generate(ses + bağlam) → critique→repair (loop-until-clean) → metacognition.
  Çıktıyı + verdict'i + güveni history'e yazar (gold flywheel).
- Nasıl konuşur: `run(cue)` → tam sonuç {final, verdict, confidence, abstained}.

**`consolidation.py` — uyku (episodik → semantik damıtma)**
- Ne yapar: ~1400 izi offline damıtır. (1) neocortex örüntüleri: benzer izleri
  kümeler (43 tema kümesi), tekrar eden Damla-örüntülerini bulur. (2) gold-adayı
  sıralama: gold'a en yakın ham YAZI izlerini sıralar (Damla onayı bekler; motor
  kendi çıktısını gold'a çeviremez — ses çöker). Gold 9→100'ü Damla'ya iş
  çıkarmadan çözer.
- Nasıl konuşur: `sleep()` → {patterns, candidates}.

**`worldmodel.py` — karar-modeli ("Damla ne yapardı", DÜRÜST)**
- Ne yapar: bir karar eşiğinde geçmiş benzer kararları getirir, ne seçtiğini +
  sonucunu + geri-alma-maliyetini gösterir, conformal ile "bu durum geçmişine
  uyuyor mu" der. Uymuyorsa ABSTAIN. FAL DEĞİL: her sayı gerçek izlerin SAYIMI
  (kaç benzer, kaçı geri-alınabilir), uydurma olasılık yok.
- Nasıl konuşur: `WorldModel().reflect(cue)` → `DecisionReflection(abstained,
  reason_line, precedents, n_similar, n_reversible)`.
- **AÇIK DÜRÜST SINIR + DENEY BULGUSU (25 Tem):** 45 homojen teknik karar ile
  alan-dışı ayrımı GÜVENİLİR DEĞİL (saçma cue bile "içeride" çıkabiliyor). ÖNCE
  embedding gücü sandık, ÖLÇTÜK: `all-mpnet-base-v2` (kararlar arası benzerlik
  ort=0.43) vs `multilingual-e5-large` (ort=0.83). e5 DAHA KÖTÜ ayırdı — teknik
  cue, "tatile mi çıksam" ve "kediler neden mavi rüya görür" ÜÇÜ DE 0.80 çıktı.
  **Bulgu: kök sebep embedding değil, VERİ.** 45 kararın hepsi stitchu/gymgyme
  teknik-mühendislik (kalıp/render/golden) = o kadar dar bir alan ki hiçbir
  embedding onları "farklı problemler" diye ayıramaz. Dar-alan manifold ayrımı
  embedding gücüyle değil, KARAR ÇEŞİTLİLİĞİYLE çözülür. Bu, "önce ölç, sonra
  iddia et" disiplininin somut örneği: güçlü modeli varsaymadık, denedik, zayıf
  çıktı, sebebini bulduk. mpnet'te kalındı (e5 kanıtlı daha kötü).
- Mimari doğru (retrieval + conformal + sayım), YAKIT (çeşitli karar) yetersiz.
  MCTS/rollout BİLEREK kurulmadı (45 kararla öğrenilebilir reward yok = fal tuzağı).

---

## 3. VERİ AKIŞI (tam döngü, tek şema)

```
cue (Damla'nın ham fikri / kararı)
   │
   ├─► retrieval (gold)      → ses çapası (few-shot)         [hippocampus]
   ├─► retrieval (episodik)  → geçmiş bağlam (ne yaptı)      [hippocampus]
   │
   ├─► router                → hızlı/yavaş yol seç           [dual-process]
   │
   ├─► generate              → yasa + ses + bağlamla üret    [neocortex+vmPFC]
   │
   ├─► critique → repair     → yasaya karşı, temizlenene dek [prefrontal RL]
   │
   ├─► metacognition         → kalibre güven                 [meta-biliş]
   ├─► conformal             → istatistiksel eşik, geçmezse ABSTAIN
   │
   └─► çıktı + güven + (abstain?) → history (gold flywheel)
```

---

## 4. ÖĞRENMEN GEREKENLER (öncelik sırasıyla)

Bu sistemi savunmak için şunları anlaman gerek. Sıra, en temelden en ileriye:

1. **Embedding nedir** — metni vektöre çeviren model; benzerlik = cosine. Neden
   homojen korpusta ayıramıyor (dewbrain'in şu anki darboğazı). → sentence-transformers.
2. **Softmax + logsumexp** — attention'ın ve Hopfield enerjisinin kalbi. Sıcaklık
   (β) ne yapar, neden kalibre edilir.
3. **α-entmax** — softmax'ın sparse hali; neden "tam sıfır ağırlık" pattern
   separation demek. Fenchel-Young loss. → arXiv:2402.13725.
4. **Modern Hopfield = attention** — Ramsauer 2020'nin ana teoremi. Enerji
   fonksiyonu, tek-adım yakınsama, üssel kapasite. → arXiv:2008.02217.
5. **Lyapunov kararlılığı** — bir dinamik sistemin bir enerji fonksiyonu monoton
   düşüyorsa kararlı minimuma yakınsar. "Sahte anı üretmez" ispatı bu.
6. **Conformal prediction** — nonconformity, empirical quantile, coverage
   garantisi, exchangeability. Neden dağılımdan-bağımsız. → Vovk et al.
7. **Complementary Learning Systems** — hızlı hippocampus + yavaş neokorteks,
   catastrophic forgetting, neden tam fine-tune sesi öldürür. → McClelland 1995.
8. **Conformal + karar teorisi** — world-model'in olgun hali; MCTS/MDP neden
   veri yeterli olunca gelir, şimdi neden nearest-decision baseline.

---

## 5. YOL HARİTASI (dürüst)

**Bugün kodda + ölçülü:** retrieval (Hopfield/entmax, sep 0.88), law parse,
critique/repair, metacognition, conformal (kapsama 0.91), router, consolidation,
Lyapunov kararlılık (spurious 0), dinamik α, sources (1400 iz), decisions (45),
world-model baseline.

**Açık (dürüst eksik):** world-model alan-dışı ayrımı (embedding + veri sınırı),
salience prior sabit 0.50, report kanalı sayıca baskın, uçtan-uca üretim API/max
gerektiriyor.

**Faz 2 (veri büyüyünce, GPU):** LoRA adapter (ortogonal) ile ses fine-tune,
öğrenilebilir reward model (RUBRIC-ARROW/DPO), world-model'e MCTS/MDP rollout,
spiking/active-inference derin katmanlar. → `BRAIN_MAP.md`.

**Faz 3 (braindot):** motor (kod) sabit kalır, veri soyutlanır → herkes kendi
Palantir'ini kurar. Kod public, herkesin verisi kendinde.
```
