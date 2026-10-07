# Geliştirme Notları

README'ye girecek bulgular ve açık işler.

## Bulgular (ölçülmüş)

1. **`localhost` gecikmesi:** Ollama adresi `localhost` iken her istekte ~2 sn ek gecikme vardı.
   Windows önce IPv6'yı (::1) deniyor, Ollama ise IPv4'te dinliyor. `127.0.0.1` yapınca
   ilk token ~2.6 sn → ~0.3 sn oldu.
2. **Retrieval eşiği (min_score 0.80):** İlgili sorular 0.84–0.90, ilgisiz sorular
   0.66–0.785 skor aldı. "Bugün hava nasıl?" 0.785 ile sınıra yakın.
3. **Diller arası arama çalışıyor:** İngilizce soru, Türkçe dokümanda doğru parçayı buldu (0.852).
   İngilizce doküman eklemeye şimdilik gerek yok.
4. **Ada-7 / Mini Ada karışması:** "Ada-7 kaç kilo?" sorusunda Mini Ada parçası 1., Ada-7 3. sırada.
   İki robotun metni çok benzer; top_k=3 olduğu için cevap yine doğru.

5. **STT CPU vs GPU (Whisper, Piper ile üretilmiş test cümleleri):**
   small/CPU/int8 ~3900 ms · small/GPU/int8 ~900 ms · small/GPU/float16 ~2800 ms (GTX 1650'de
   float16 yavaş) · base/GPU ~340 ms ama yanlış tanıma. Sebep: Whisper her sesi 30 sn'ye
   tamamlayıp işliyor, kısa soru da 30 sn'lik iş. Seçim: small + GPU + int8.
6. **Hotwords:** Ürün isimleri Whisper'a ipucu olarak verildi. "A'da 7" → "Ada-7",
   "Arap" → "ARAT" düzeldi. base modelde ise hotword'leri uydurdu ("AKINCI-5, AMR, AROS").
7. **Gürültüyle test yanıltıcı:** Rastgele gürültüde Whisper 10+ sn takıldı (dili "nn" algıladı).
   Bu yüzden VAD şart: sadece konuşma içeren ses STT'ye gitmeli.
8. **TTS (Piper):** RTF ~0.055 (3 sn sesi ~160 ms'de üretiyor), CPU'da. Türkçe ses: tr_TR-dfki-medium
   (planlanan fahrettin sesi Piper listesinde yok).

## Açık işler

- [ ] **Dil karışması:** İngilizce soruya bir kez Türkçe ve yanlış cevap verdi (şarj süresi ile
      çalışma süresini karıştırdı), bir kez doğru. Sebep: temperature 0.2 + Türkçe bağlam.
      Plan: `temperature: 0` + dil talimatını sorunun yanına da koymak. Tekrarlı testle ölçülecek.
- [ ] **İlk token dalgalanması:** 327 ms ile 2464 ms arasında. Tahmin: sistem promptu değişince
      (TR↔EN) Ollama önbelleği kullanılamıyor. Benchmark ile doğrulanacak.
- [ ] **Bozuk Türkçe:** "Ada-7 uçabilir değil." 3B modelin dil sınırı.
- [ ] **Eksik cevap:** AMR sorusunda sadece v500 söylendi, v1000 atlandı.
- [ ] **Offline mod:** Embedding modeli her açılışta Hugging Face'e soruyor (HF_TOKEN uyarısı).
      Docker'da `HF_HUB_OFFLINE=1` açılacak.
- [ ] **Ollama symlink sorunu:** Ollama 0.40 model kaydını symlink olarak tutuyor; sandbox'tan
      indirilince açılamadı. Modeli kullanıcının kendi terminalinden indirmesi çözdü.
