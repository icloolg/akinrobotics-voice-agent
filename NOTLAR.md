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

9. **Uçtan uca ilk test:** VAD → STT → RAG+LLM → cümle cümle TTS çalışıyor. Kullanıcı sustuktan
   ilk sese ~4.5 sn: STT ~1.1 sn, LLM prompt işleme ~3 sn, TTS ~0.4 sn.
10. **Darboğaz LLM prompt işleme:** GTX 1650 (tensor core yok) ~200 token/sn. Denenen ve etkisiz:
    Flash Attention kapatma, num_batch 64/128, Whisper'ı GPU'dan çıkarma. Donanım sınırı.
11. **D: ortak sistem promptu + dil talimatı sonda:** 6 karışık TR/EN soruda ortalama prompt
    işleme 2246 → 1891 ms. Mini Ada İngilizce sorusu düzeldi (2.5 sa ❌ → 8 sa ✅). Süre bağlamdaki
    token sayısıyla doğrusal (~5 ms/token): 3 parçalı sorular ~3.5 sn, 1-2 parçalı ~1 sn.

12. **İlk gerçek sesli test (kendi mikrofonumla):** İlk ses 1.25–2.6 sn. Whisper "Ada-7"yi
    "other 7" duydu ama anlamsal arama yine Ada-7 parçasını buldu, cevap doğru → STT hatalarına dayanıklı.
13. **Sohbet sorunu:** "Sen kimsin?", "How are you?" → "bilgi bulamadım". Güvenli ama doğal değil.
14. **Anlamsal yönlendirme (router):** Soru, sohbet örnek cümleleriyle karşılaştırılıyor (aynı e5 modeli,
    ek model yok). Tek kural ("sohbet skoru > bilgi skoru") "Robotlarınız neler yapabilir?" sorusunu
    yanlışlıkla sohbete gönderiyordu (0.916 vs 0.863). İki koşul: sohbet ≥ 0.90 VE fark ≥ 0.10.
    20 test sorusunda: sohbet 7/8, bilgi 7/7, kapsam dışı 5/5 doğru yolda. Sohbet cevapları ~150–600 ms.
15. **Sohbet promptunun dili:** İngilizce prompt ile Türkçe cevaplarda dil karıştı ("robots hakkında",
    "robotuddur"). Her dil için kendi dilinde prompt + hazır tanıtım cümlesi → düzeldi.

16. **Kendi sesimle 18 cümlelik STT test seti** (tests/voice/sentences.csv, scripts/eval_stt.py):
    - Mikrofon seviyesi 65: dil 13/18, WER 0.62, tam doğru 5/18. Kayıtlar çok sessizdi (tepe ~0.05).
    - Mikrofon seviyesi 100: dil 16/18, WER 0.38, tam doğru 8/18. Kod değişmeden hata %39 azaldı.
    - Yazılımla ses yükseltme (normalize) fayda etmedi: gürültü de büyüyor.
    - Hotwords kapatınca WER arttı (0.62 → 0.67/0.77): faydalı, kalıyor.
17. **Dil seçim kuralı** (aynı 18 kayıtta karşılaştırıldı):
    - "Emin değilse Türkçe" (eşik 0.7): İngilizce cümleleri Türkçeye zorladı, Whisper ÇEVİRİ yaptı
      ("Hello how are you" → "Merhaba, nasılsınız?"). 16/18.
    - "tr/en hangisi yüksekse": kısa Türkçe kelimeleri İngilizce sandı ("Nerede" → "Merida"). 15/18.
    - **Seçilen:** varsayılan Türkçe; İngilizceye ancak P(en) ≥ 0.3 VE P(en) > 2×P(tr) ise geç → **18/18, WER 0.30.**
18. **Kalan STT hataları** çoğunlukla kelime düzeyinde ("kalp taktayanıyor", "BENDİ VEN"). Daha büyük
    model (medium/turbo) denemesi indirme takıldığı için yarım kaldı; 4 GB VRAM'de LLM ile sığması da şüpheli.

19. **Parçalama (chunking) hatası:** 600 karakterlik paketleme ilgisiz bölümleri tek parçada birleştirdi
    ("Misyon ve Vizyon" + "Ürün Gamı" + bir sonraki bölümün yarım başlığı). Karışık parçanın vektörü
    hiçbir konuya tam uymadı → "Hangi robotlarınız var?" sadece "ARAT ve AKINCI-5" dedi.
    **Bölüm bazlı parçalama** (her `##` bölümü bir parça, başında "doküman - bölüm" başlığı, kaynak URL
    meta veri olarak): robot listesi sorularında Ürün Gamı 0/4 → 4/4 ilk 3'te; 9 bilgi sorusunun 9'unda
    1. sıradaki parça doğru bölüm (önceden "Ada-7 kaç kilo?" için Mini Ada 1. sıradaydı). Yönlendirme
    ve eşik değişmeden 21/21 doğru. Cevap artık 9 robotu sayıyor.

20. **Ajan değerlendirme seti** (tests/agent_questions.csv, scripts/eval_agent.py): 23 soru —
    16 bilgi (beklenen değer cevapta var mı), 3 sohbet (doğru yönlendirme), 4 kapsam dışı (bilgi uydurmuyor mu).
21. **Ölçüm tuzağı — Ollama önbelleği:** Aynı test ikinci kez çalışınca toplam süre 2416 → 1075 ms düştü,
    oysa gerçek bir değişiklik yoktu (Ollama eski promptları bellekte tutuyor). Çözüm: her test çalıştırmasında
    sistem promptunun başına rastgele etiket → her çalıştırma soğuk başlıyor. Doğrulama: aynı ayar iki kez
    → 2513 / 2515 ms.
22. **top_k ve geçmiş (soğuk önbellekle):** top_k=3/geçmiş=3: ilk cümle 3268 ms, en kötü 4745 ms, 21/23.
    **top_k=2/geçmiş=1: 2515 ms (−%23), en kötü 3615 ms, 20–21/23.** top_k=1: çok hızlı ama doğruluk 19/23.
    Doğruluk çalıştırmalar arasında ±1 oynuyor (temperature 0.2).
23. **Cümle bölme:** 100 karakteri geçen cümle son virgülden bölünüyor; satır sonu da cümle sonu;
    liste işaretleri temizlenip kısa maddeler virgülle birleşiyor ("Eğitim, Sağlık").
24. **Halüsinasyon örneği:** "Ada-7 uçabilir mi?" → Ada-7 bölümü eşiği geçiyor (konu ilgili) ve model
    "Ada-7 uçabilir, ancak..." diyor. Eşik tek başına yetmiyor; açık iş.

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
