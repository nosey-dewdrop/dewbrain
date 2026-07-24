# brain architecture — dewbrain'in nöromimari temeli

Damla'nın emri (24 Tem): dewbrain uydurma "konu lobları" değil, GERÇEK beyin fonksiyonel mimarisi. Mühendislik + biyoteknoloji + nöroscience. Bu dosya kaynak, sonraki session'da tekrar aranmasın.

## çekirdek bulgu

Beyin KONUYA göre değil FONKSİYONA göre bölünür. Ve gerçek nöroscience (hippocampus/neocortex/consolidation) ile 2025-26 LLM-ajan bellek mimarisi (episodik/semantik/prosedürel) AYNI yere yakınsıyor. dewbrain ikisini birleştirir.

## gerçek beyin mekaniği

- **Hippocampus depo değil, INDEX.** Yeni bilgi dağıtık neocortex bölgelerinde kodlanır, hippocampus'ta bütünleşir. Hippocampus izleri yeniden aktive eder (index gibi), depo gibi tutmaz.
- **Consolidation.** Yeni bellek başta hippocampus'a bağımlı, zamanla tekrarlı yeniden-aktivasyonla (derin uyku + uyanık dinlenme) neocortex'e taşınır, bağlar güçlenir, gereksiz/bağlamsal detay düşer. Episodik → semantik dönüşüm.
- **vmPFC** hippocampus ↔ cortex arası aracı, bellekleri entegre eder ve neocortex'te stabilize eder.
- **Prefrontal karar (RL).** Etkileşim ve feedback'ten öğrenme prefrontal-modellenir.

## AI ↔ beyin eşlemesi (literatür)

- RNN ↔ hippocampal temporal/sequential + memory
- RL ↔ prefrontal karar, feedback'ten öğrenme
- attention/CNN ↔ görsel korteks pattern
- cognitive maps: hippocampus + entorhinal cortex soyut değişkenleri ve görev yapısını harita-formatında kodlar; ilişkili kavramlar birbirine yakın konumlanır. AI'da relational graph / vector space karşılığı → transfer + planlama.

## 2025-26 ajan bellek mimarisi (yakınsama)

Üç katman standardı: **episodik / semantik / prosedürel.**
- Working / episodic / semantic AYRILIR → cross-session drift kontrol, bounded context.
- Olaylar tam bağlamla kodlanır, dış db'de saklanır, similarity + recency + salience ile getirilir, periyodik olarak reflection ile semantik bilgiye konsolide edilir.
- Katman ayrımı (Weaviate context-engineering): memory-layer (geçmiş etkileşim) / knowledge-layer (alan bilgisi) / working-memory (anlık durum), her birinin retrieval semantiği farklı.
- Açık zorluk: retrieval isabeti + stability/plasticity dengesi (yeni öğrenirken eskiyi bozmama).

## dewbrain'e ne düşer

- retrieval = hippocampus (verified/ham izlerden similarity+salience getir)
- neocortex = konsolide semantik (yasa + ses + Damla bilgisi, tek havuz)
- consolidation = ham → yasa/ses damıtma adımı
- öz-eleştiri = prefrontal RL (Damla feedback'i = ödül/ceza sinyali)
- stability/plasticity: yeni verified eklenince eski ses bozulmamalı (Damla'nın nefret ettiği generic = plasticity fazlası)

## kaynaklar

- Shift from Hippocampal to Neocortical Retrieval with Consolidation — J Neurosci https://www.jneurosci.org/content/29/32/10087
- The Consolidation and Transformation of Memory — Neuron https://www.cell.com/neuron/fulltext/S0896-6273(15)00761-8
- AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents — arXiv https://arxiv.org/pdf/2512.23343
- Cognitive Architectures for Language Agents — arXiv https://arxiv.org/pdf/2309.02427
- Multi-Layered Memory Architectures for LLM Agents — arXiv https://arxiv.org/html/2603.29194v1
- Episodic-Semantic Memory Architecture for Long-Horizon Agents — arXiv https://arxiv.org/pdf/2605.17625
- Thinking Beyond Tokens: Brain-Inspired to Cognitive Foundations for AGI — arXiv https://arxiv.org/pdf/2507.00951
