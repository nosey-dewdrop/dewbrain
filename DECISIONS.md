
## 2026-07-25 kaynak-kayıt (sources.py) — beynin tüm veri kanalları
- Karar: dewbrain'in episodik belleği tek kaynaktan (9 gold) değil, Damla'nın ÜRETTİĞİ her izden beslenir. Tek registry (sources.py): writing 237 + decision 42 + report 732 + projectdoc 203 + idea 5 = 1219 iz. notes/Obsidian parked (DEWBRAIN_NOTES_DIR ile açılır). Yeni kanal = registry'ye 1 Channel satırı, kod değil.
- Neden: Damla emri "sadece kod değil bütün yazılar olabilir". Beyin tek dosyadan değil tüm trailden "sen kimsin" çıkarır. Her kanalı ayrı loader'a bağlamak kırılgan (her yeni kanal = yeni kod); registry pattern moat değil ama doğru mühendislik.
- Geri alma: ucuz (yeni dosya sources.py, mevcut memory/decisions/retrieval'a dokunulmadı). Channel.enabled ile bir kanal silinmeden park edilir.
- Açık pürüz (NOT): report 732 sayıca writing 237'yi eziyor → retrieval'da audit dili sesi bastırabilir. Ağırlık (report w=0.9, decision w=1.4) kısmi denge; tam denge konsolidasyon katmanında (kanal-dengeli örnekleme). entmax+MMR alakasızı zaten eliyor. Bugün çözülmedi, işaretlendi.
