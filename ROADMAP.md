# dewbrain — roadmap

Vizyon SABİT, adımlar PAY-AS-YOU-GO. Büyük hedef değişmez (seni modelleyen karar-
çekirdeği → braindot). Ama her katman bir öncekinin KANITLANMASINA ve VERİYE bağlı
açılır — çünkü kanıtlanmamış çekirdeğin üstüne katman koymak jürinin 4/4 reddettiği
şey ("kağıtta çok katman, kodda az"). O yüzden roadmap bir söz listesi değil, bir
KAPI dizisi: her kapı bir kill-gate (geçmezse sonraki açılmaz).

Durum işaretleri: ✅ kodda+ölçülü · 🟡 kodda ama sınırlı/veri-bekliyor · ⬜ tasarım · 🔒 veri/GPU kilidi

---

## FAZ 0 — ÇEKİRDEK (bugün, 25 Tem) ✅ BİTTİ

Amaç: yazı-motorunu karar-çekirdeğine çevir, API'siz kanıtla.

- ✅ `sources.py` — kaynak-kayıt, 1400 iz (tüm trail)
- ✅ `decisions.py` — karar-belleği, 45 karar
- ✅ `retrieval.py` — sparse Hopfield/entmax, separability 0.88
- ✅ `hopfield_energy.py` — Lyapunov kararlılık (spurious 0), dinamik α
- ✅ `metacognition.py` — kalibre güven + ABSTAIN
- ✅ `conformal.py` — istatistiksel garanti, kapsama 0.91
- ✅ `router.py` — dual-process (hızlı/yavaş)
- ✅ `consolidation.py` — uyku, örüntü kümeleri + gold-aday
- 🟡 `worldmodel.py` — karar-modeli baseline (alan-dışı ayrımı VERİ bekliyor)
- ✅ uçtan-uca döngü çalıştı (engine=max, "I can't" yazısı, güven %100)
- ✅ public repo + THESIS + REFERENCES

**Kill-gate (geçildi):** motor uçtan uca dönüyor mu? EVET. Bu olmadan Faz 1 açılmazdı.

---

## FAZ 1 — GÜÇLENDİRME (sonraki oturumlar, veri gerektirmez ya da AZ) 🟡

Amaç: bugünkü çekirdeğin zayıf yerlerini kapat. Her biri bağımsız, pay-as-you-go.

1. ⬜ **API key ile otonom uçtan-uca** — engine=api, gece loop, kendi kendine
   üretir + conformal filtreler. Şu an engine=max (session-bağlı). *(gerek: key)*
2. 🟡 **world-model çeşitliliği** — 45 karar hepsi teknik/stitchu, dar-alan.
   DENEY BULGUSU (25 Tem): embedding gücü çözmedi (e5 mpnet'ten kötü), VERİ sorunu.
   Çözüm: farklı alanlardan karar (iş, içerik, strateji) topla → manifold ayrışır.
   *(gerek: çeşitli karar verisi — DECISIONS.md'leri zenginleştir)*
3. ⬜ **salience prior gerçek** — şu an gold-adaylarda sabit 0.50. Metnin gerçek
   substance/poignancy'sini ölç (uzunluk+ders yoğunluğu+duygu). *(veri gerekmez)*
4. ⬜ **kanal-dengeleme** — report 732 sayıca writing 237'yi eziyor; retrieval'da
   audit dili sesi bastırabilir. Kanal-dengeli örnekleme. *(veri gerekmez)*
5. ⬜ **retrieval cache** — her run 1400 iz embed ediyor (~30sn). Bir kez kur,
   diske cache'le. *(veri gerekmez, sadece hız)*
6. ⬜ **sınır filtresi kodda** — hassas veri (anne/para/hasta) çıktıya asla
   girmesin diye retrieval-sonrası filtre. Şu an yasa var, kod yok. *(veri gerekmez)*

**Kill-gate:** world-model alan-dışını güvenilir ayırıyor mu? Ayırana kadar
"karar veren" iddiası yapılmaz (over-claim = Damla'nın nefreti).

---

## FAZ 2 — ÖĞRENEN MODEL (100+ onaylı gold + GPU) 🔒

Amaç: prompt-tabanlı sistemi ÖĞRENİLMİŞ modele çevir. Faz 1 bunun yakıtını
biriktirir (consolidation gold büyütür). VERİ + GPU kilidi.

1. 🔒 **LoRA ses fine-tune** — O-LoRA (ortogonal) + importance-reg + adaptive
   replay. Base model=korunan neokorteks, adapter=yeni. Tam fine-tune YASAK (ses
   ölür, catastrophic forgetting). *(gerek: 100+ gold, GPU)* → C1-C3 REFERENCES.
2. 🔒 **öğrenilebilir reward model** — prompt-judge yerine trained critic
   (RUBRIC-ARROW/DPO). Yasa 16 madde → öğrenilmiş pointwise reward. *(gerek: etiketli
   çıktı verisi)* → E2 REFERENCES.
3. 🔒 **world-model MCTS/MDP rollout** — nearest-decision baseline'ın üstüne
   gerçek ağaç-arama. UYARI: öğrenilebilir reward olmadan MCTS = fal tuzağı.
   Ancak 100+ çeşitli karar + reward olunca. *(gerek: çok karar + reward)*

**Kill-gate:** fine-tuned ses, prompt-tabanlı sesi geçiyor mu (blind A/B, Damla
hakem)? Geçmezse LoRA satılmaz, prompt kalır.

---

## FAZ 3 — DERİN NÖRON KATMANLARI (araştırma-inşa, beklemeden) 🔒

BRAIN_MAP'te tasarlanmış, jüriden geçmiş derin katmanlar. Damla'nın "research
değil, inşa" emri: akademisyen anlamak için okur, Damla kurmak için öğrenir.

- ⬜ **Hippocampus derin** — DG pattern-separation + CA3 attractor + successor-repr.
  retrieval'ı statik cosine'den dinamik associative-memory + karar-grafına çevirir.
- 🔒 **Active inference / EFE** — "Damla hangi modda, ne zaman araştırır ne zaman
  kill eder" tek denklemde (pymdp, Friston). *(mod'lar DECISIONS clustering)*
- 🔒 **Spiking karar** — snnTorch, N→N+1 karar dinamiği. *(SNN-vs-dense ablasyon zorunlu)*
- 🔒 **Neuromodülasyon** — statik gain'leri (β, law weights) Damla ödül/duygu
  geçmişinden öğrenilen skalerlere bağla.

**Kill-gate:** her derin katman baseline'ı YENMELİ, yoksa "beyin-terimi giydirilmiş X".

---

## FAZ 4 — EKSİK BİLİŞ KATMANLARI (eksik-avı, 100gold) 🔒

Beynin OLMAYAN ama gereken fonksiyonları (8 ajan buldu, kodla teyit "sıfır"):

- ⬜ **Dikkat/GWT** — "50 projeden hangisine enerji, hangisini öldür". cue'yu KİM
  üretir (şu an dışarıdan geliyor). Damla'nın asıl darboğazı.
- ⬜ **Sosyal-biliş/ToM** — yatırımcı/işveren için ayrı zihin-slotu ("a-için-yaz
  b-yi-hesaba-kat"). Müzakere katmanının ön-şartı.
- 🔒 **İnterosepsiyon/tükenme** — "şu tempo N gün sürerse geçmiş çöküş eşiğini
  kırarsın". git log tempo + enerji-state. *(bakıcılık değil, sayı-ayna)*
- 🔒 **Otobiyografik benlik/zaman** — "gelecek hatasına uyar" bunsuz imkansız.
  *(korpusta zaman ekseni gerek)*

---

## FAZ 5 — braindot (motor sabit, veri soyut) 🔒

Kod (motor) değişmez, veri katmanı soyutlanır → herkes kendi Palantir'ini kurar.
- ⬜ yolları config'e çıkar (herkes kendi verisiyle çalıştırsın)
- ⬜ kişisel-veri kurulum akışı (kullanıcı kendi izlerini bağlar)
- 🔒 SF/YC pitch: "kurucunun kendisinde 1 yıl çalıştı" (bugün başladı)

---

## KARAR: roadmap mı, pay-as-you-go mu?

**İkisi birden.** Bu harita SABİT (Faz 0→5 sırası değişmez, her faz öncekinin
kill-gate'ine bağlı). Ama faz İÇİNDEKİ adımlar pay-as-you-go: hangi veri/kaynak
gelirse o adım açılır. Faz 1 adımları bağımsız (istediğin sıradan yapılır). Faz
2+ VERİ/GPU kilitli — o kilitler açılana kadar zorlanmaz (yoksa fal/kostüm olur).

**Bir sonraki oturumun net adayları (Faz 1, veri gerektirmez):** salience prior,
kanal-dengeleme, retrieval cache, sınır filtresi. world-model çeşitliliği veri
ister. Sıra Damla'nın.
