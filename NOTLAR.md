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

25. **Araç (tool) desteği — case'in "yeni araç eklemek kolay olmalı" maddesi:** LLM function-calling yerine
    anlamsal yönlendirme (3B model function-calling'de güvenilir değil + ek LLM çağrısı ~1-2 sn). Araç çıktısı
    BAĞLAM olarak aynı "sadece bağlamı kullan" promptuyla LLM'e gidiyor. Araçlar: `datetime` (saat/tarih),
    `robot_status` (HTTP API, sahte filo servisi /mock/robots). Yeni araç = Tool alt sınıfı + factory'de 1 satır + config.
26. **Robot durumu vs teknik özellik ayrımı:** "Ada-7 kaç saat çalışır?" (doküman) ile "Robotların şarj durumu?"
    (canlı) anlamca çok yakın; "Robotlarınız neler yapabilir?" araca 0.931 skor verdi. Sadece anlamsal skorla
    ayrılamadı → **hibrit yönlendirme**: anlamsal skor + canlı veri kelimesi ("şu an", "durum", "now"...).
    Farklı ifadelerle yazılmış test: durum soruları 8/10 araca, bilgi soruları 0/12 araca (kesinlik öncelikli).
    Tüm yönlendirme: 33/35.
27. **Araç hata yönetimi:** API kapalıyken LLM "servise ulaşılamıyor" mesajını "servis altında durumda" diye
    çarpıttı. Çözüm: araç `ToolUnavailable` fırlatıyor, mesaj LLM'siz aynen söyleniyor. Robot verisi uydurulmuyor.
28. **Ajan değerlendirmesi (28 soru):** bilgi 15/16, sohbet 3/3, kapsam dışı 3-4/4, araç 5/5 → 27/28.
    Araç cevapları 0.5-1 sn (kısa bağlam).

29. **Hotwords biçimi** (21 kayıt; 19-21 araç cümleleri eklendi): isim listesi WER 0.35 / tam 9;
    **Türkçe cümle içinde WER 0.30 / tam 11 (seçilen)**; hotwords yok WER 0.45 / tam 9.
    "Saat kaç?" + günlük kelimeler varyantı WER 0.25 verdi ama kelimeler test cümleleriyle örtüştüğü için
    (overfitting riski) seçilmedi.

30. **Alan içi cevapsız sorular (halüsinasyonun asıl kaynağı):** "Ada-7 uçabilir mi?", "ARAT'ın fiyatı?"
    gibi sorular konuyla ilgili olduğu için eşiği geçiyor; reddetmek LLM'e kalıyor. 7 soruluk set eklendi.
    Önce: **2/7** ("Mini Ada yüzebilir.", "Servis Robotu V3 5 yıl garantili." — uydurma; "CONTEXT'da
    belirtilmemiştir" — promptun iç yapısı sızıyor).
31. **Promptun son satırına hazır red cümlesi:** "Bağlam açıkça cevaplamıyorsa tahmin etme, evet/hayır deme,
    sadece şunu söyle: '...bilgi bulamadım.'" → **cevapsız 6-7/7**, toplam 28/34 → 32/34.
    Bedel: ilk cümle +~300 ms (talimat her soruda işleniyor) ve bir çalıştırmada cevabı olan bir soruya
    aşırı temkinli red. Case'in doğruluk/halüsinasyon kriteri nedeniyle kabul edildi.
32. **temperature 0:** aynı ayarla iki çalıştırma birebir aynı sonuç (0.2'de çalıştırmalar arası ±1).
    Hız farkı yok. Bilgi asistanında yaratıcılık değil tutarlılık isteniyor.

33. **Karşılaştırma soruları RAG ile cevaplanamıyor:** "En hafif robot?", "En hızlı?", "8 saatten fazla
    çalışanlar?" → 5 sorudan 0 doğru (3 "bilgi yok", 2 yanlış). Sebep: RAG 2 parça getiriyor, cevap için
    8 robotun hepsi gerekiyor.
34. **Veritabanı aracı (SQLite, robot_specs):** knowledge/robot_specs.csv → data/robots.db (ingest).
    Text-to-SQL yok (3B model güvenilmez + güvenlik riski): sorudaki kelime sabit bir sütun listesinden
    birini seçiyor, sorgu sabit (ORDER BY). Sayı filtresi ("8 saatten fazla") SQL'de parametreyle
    yapılıyor — LLM'e bırakınca 8 saatlik robotu "8'den fazla" saydı. Ajan koduna dokunulmadı: aynı Tool
    arayüzü + factory'de 1 satır + config.
    Yönlendirme: karşılaştırma 8/10 araca, tek robot soruları 10/10 RAG'de, durum soruları 3/3 API'de.
    "charges the fastest" önce hız sütununu seçti ("fast") → şarj sütunu önce kontrol ediliyor.
35. **Ajan değerlendirmesi (40 soru):** bilgi 14/16, sohbet 3/3, kapsam dışı 3/3, cevapsız 6/7,
    araç 11/11 → **37/40**. Veritabanı cevapları ~0.8-1.5 sn.
36. **Aşırı temkin (red cümlesinin yan etkisi):** cevabı bağlamda olan sorulara ara sıra "bilgi bulamadım"
    ("AKINSOFT'un kurucusu", bir çalıştırmada "Robotlar şu anda ne durumda?"). Uydurmaya göre güvenli
    taraf. Not: temperature 0 olsa da, test her çalıştırmada promptun başına farklı etiket koyduğu için
    çalıştırmalar arasında ±1-2 soru oynuyor.

## Açık işler

- [ ] **Bir cümlede birden çok sayı:** "8 saat çalışır, 2,5 saatte şarj olur, otonom şarjla 24 saat" →
      model Mini Ada bataryası için 2,5 veya 24 diyor. 3B modelin sınırı; dokümanı teste göre yeniden
      yazmak yerine sınırlama olarak raporlanacak.

- [ ] **"Saat kaç?" STT'de tanınmıyor** ("HATKAT", "Hot cut"); çok kısa cümle, Whisper small'un sınırı.
      Hotwords ile ayar denendi, test setine göre ayar riski nedeniyle reddedildi.

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

## Teslim hazırlığı (case'e göre eklenenler)

37. **Gerçek yanıt gecikmesi:** Zamanlayıcı VAD "sustu" dedikten sonra başlıyordu; kullanıcının hissettiği süreye
    VAD'in beklediği `min_silence_ms` (500 ms) dahil değildi. Artık her tur `endpoint_ms` ve `response_ms`
    (= 500 + ilk ses) kaydediyor. `scripts/benchmark.py` logs/turns.jsonl'dan tabloyu ve donanımı çıkarıyor.
    44 turda yanıt gecikmesi medyanı: RAG 5,1 sn · araç 2,7 sn · sohbet 2,2 sn · LLM'siz 1,8 sn.
38. **Söz kesme (barge-in):** Sunucu cevap verirken de dinliyor; 250 ms sesli konuşmada `speech_start` gönderip
    turu iptal ediyor (LLM'in HTTP akışı da kapanıyor). Cevap sırasında daha kısa sesler yok sayılıyor.
    VAD'de sesli süre takibi için Silero modeli sarmalandı (olasılığı VADIterator dışarı vermiyor).
    Kayıt 06/11/15 ile denendi: tek parça, 1,3–2,1 sn sesli süre; 60 ms'lik tık parça üretmiyor.
39. **Numaralı liste hatası:** "1. Eğitim" cümle bölücüde "1." sonrasından kesiliyor, TTS rakamları okuyordu
    (tests/test_text.py yakaladı). Satır başındaki "rakam + nokta" artık cümle sonu sayılmıyor.
40. **Bağlantı başına geçmiş:** Tek Agent nesnesi bütün istemcilerce paylaşılıyordu; `Agent.fork()` ile her
    WebSocket kendi geçmişini tutuyor, GPU'da turlar bir kilitle sıraya giriyor.
41. **VAD tamponu:** Parça bittiğinde `reset()` aynı pakette gelen sonraki sesi de siliyordu; düzeltildi.
42. **Sesli test bulgusu — araç yanlış eşleşirse:** "En akıllı robot hangisi?" ("en" kelimesi) veritabanı
    aracına gitti; sütun yok → bütün tablo bağlam oldu → ilk ses **8.5 sn**. Çözüm: araç
    `ToolNotApplicable` fırlatıyor, soru RAG'e düşüyor → 3.2 sn, doğru red. Genel mekanizma (her araç için).
43. **"Nasılsın?" → "Nasılsın?" (tekrar):** sohbet promptuna "kullanıcının sözlerini tekrar etme, cevap ver"
    + örnek → "İyiyim, teşekkür ederim." "Merhaba nasılsın?" hala bazen tekrar ediyor.
44. **Denendi ve geri alındı — "yardımcı red":** "Bilgi yok de, sonra bağlamdan ilgili tek bir bilgi ekle."
    Amaç: "AKINCI-5 uçabilir mi?" sorusuna çıplak red yerine "...yürüyebilen bir insansı robottur".
    Sonuç kötüleşti: "ARAT merdiven çıkabilir mi?" → "Hayır." (yanlış), "Hangi robotlarınız var?" ve
    "8 saatten fazla" reddedildi, "Related fact:" cevaba sızdı. 3B model iki parçalı koşullu talimatı
    uygulayamıyor → basit kurala dönüldü. Neden "Hayır" denmiyor: dokümanlar "uçamaz" da demiyor;
    genel bilgiyle "hayır" yanlış olabilir (ADA robotlarının yemek yapabildiği haberlerde geçiyor).
45. **Değerlendirme (42 soru): 39/42** — bilgi 14/16, sohbet 4/4, kapsam dışı 3/3, cevapsız 7/8, araç 11/11.
47. **Whisper ipucu cümlesini tekrarladı (tarayıcı testinde):** Kullanıcı konuşmadığı halde transkript
    "AKINCI-5, ARAT, ARAT, AKINCI-5, AMR ve AROS hakkında sorular." — belirsiz seste (gürültü, yankı)
    Whisper verilen hotwords metnini yazıyor. Filtre: 3+ kelimelik bir transkriptin bütün kelimeleri ipucu
    cümlesinden geliyorsa boş sayılır. "ARAT", "Mini Ada" gibi kısa cevaplar korunuyor. STT seti etkilenmedi
    (21/21, WER 0.30).
48. **Aşırı temkin (tarayıcı testinde göze battı):** "Hangi yazılım servisleri var?", "AKINSOFT nedir?" →
    "bilgi bulamadım". 3 gerçek soru eval setine eklendi (45 soru). Olumlu ifadeli kural denendi ("cevap
    bağlamdaysa onunla cevap ver, hiç yoksa ... de") → bilgi 14-15/19 → 13-14/19 ve "Mini Ada yüzebilir."
    geri geldi → geri alındı. Arama incelemesi gerçek sebepleri gösterdi:
    - "yazılım servisleri": doğru parça 1. sırada ama 2. sıradaki "Servis Robotu V3" ("servis" kelimesi) okundu;
    - "What software does AKINROBOTICS offer?": AROS parçası 4. sırada, modele hiç gitmedi — şirket
      dokümanında firmanın yazılım geliştirdiği yazmıyor (bilgi tabanı eksiği);
    - "AKINSOFT nedir?": cevap 1. parçada var, model yanlış okudu (3B sınırı).
    Sonraki adım: şirket dokümanına "Yazılımlar" bölümü (sitede var); gerisi sınırlama olarak raporlanacak.
49. **Tarayıcı arayüzüne robot görselleri:** akinrobotics.com'dan 7 görsel (kullanıcı onayıyla), kayan şerit +
    cevapta adı geçen robotun görseli. README'ye telif notu. Görseller çıkmıyordu: port 8000'de diğer
    pencereden kalma eski sunucu süreci çalışıyordu (yeni sunucu açılamamıştı); süreç durduruldu.
50. **AKINSOFT dokümanı eklendi** (knowledge/akinsoft_tr.md, akinsoft.com.tr/as/hakkimizda, 8 bölüm: kuruluş,
    yaygınlık, ürünler, bulut/e-dönüşüm, bağlı kuruluşlar, tarihçe, misyon, sertifikalar). AKINROBOTICS
    dokümanına kuruluş yılı (2015) eklendi. Tek tek: "AKINSOFT nedir?", "ne zaman kuruldu?", "kaç ülkede?",
    "İstanbul Robot Müzesi nedir?" doğru; "What products does AKINSOFT have?" ürünler bölümü ilk 2'ye girmedi.
    Regresyon: 25 parça, 39/45 (önce 38-40) — eşikler ve yönlendirme etkilenmedi.
    Gözlem: "AKINSOFT nedir?" tek başına doğru, değerlendirmede (bir önceki soru reddedilmişken) reddedildi →
    1 turluk geçmiş bir önceki reddi taşıyabiliyor.
51. **Sunucu çalışırken yeniden indeksleme cevapları kesti:** `scripts.ingest` koleksiyonu silip yeniden
    oluşturuyor; çalışan sunucu silinmiş koleksiyonu tutuyordu → canlı testte hiçbir soruya cevap gelmedi.
    Çözüm: arama hata verirse koleksiyon bir kez yeniden açılıyor (hata yeniden üretilip düzeltme doğrulandı).
52. **"AKINSOFT" STT'de tanınmıyordu** ("Akins of Nedir", "AKINSOFTNESS" — İngilizce sanıldı). Sözlüğe
    eklendi. 2 AKINSOFT cümlesi kaydedildi (23 kayıt): sözlüksüz WER 0.29 / tam 12, sözlüklü WER 0.30 /
    tam 11 (fark gürültü seviyesinde); sözlüklü versiyonda iki AKINSOFT cümlesi de tam doğru.
    İlke: yeni bilgi kaynağının özel isimleri STT sözlüğüne de eklenir.
53. **Takip soruları (konu takibi, app/agent/topics.py):** "AKINSOFT nedir?" → "Ne zaman kurulmuş?" reddediliyordu
    (her soru tek başına aranıyor). LLM ile soruyu yeniden yazmak +1-2 sn olacağı için hafif yöntem: bilinen
    isimler (config `followup.topics`) metinde aranıyor; isim yoksa son isim soruya ekleniyor. Eval'e 4 takip
    çifti eklendi (53 soru). Başlangıç takip 1/4.
    - İlk deneme (her isimsiz soruya ekle): takip 3/4 ama **araçlar 10/11 → 2/11** ("Mini Ada: Saat kaç?").
    - Ölçüm: takip soruları tek başına 0.785-0.823 skor alıyor, bağımsız sorular 0.869-0.890 → kural: isim yok
      + araç/sohbet değil + skor < 0.85 ise ekle. Sonuç: takip 3/4, araçlar 9-11/11, toplam 43 → 45/53.
    - Yan bulgu: "Peki kaç saat çalışır?" saat aracına gidiyordu ("kaç saat" süre, "saat kaç" saat). Saat aracına
      anahtar kelime şartı eklendi ("saat kaç", "tarih", "gün", "time", "date"...).
    - ~~Geçmişi kapatmak (history=0) denendi: 44/53~~ — GEÇERSİZ ÖLÇÜM, bkz. 54.
    - Test düzeltmesi: "What is ARAT?" cevabı "quadruped" diyordu, beklenen sadece "four" idi → "four|quadruped".
54. **Geçmiş aynı konuyu ikinci kez cevaplatmıyordu + history=0 hatası:** Sesli testte "AKINSOFT nedir?" iki kez
    reddedildi. Yeniden üretildi: taze konuşmada cevaplıyor; aynı konuda bir sorudan sonra (İngilizcesi ya da aynı
    soru tekrar) reddediyor — model geçmişte cevabı görünce tekrar cevaplamıyor. Ayrıca `history[-2*n:]` n=0 için
    `history[0:]` = bütün liste → "history=0" aslında sınırsız geçmişti (önceki ölçüm geçersiz). Düzeltildi
    (`max(0, len-2n)`). Doğru ölçüm: **history=0: 49/53, takip 4/4, ilk cümle 1683 ms**; history=1: 46/53, 3/4,
    2978 ms. Konu takibi geçmişin işini görüyor; geçmiş prompt önbelleğini de bozuyordu → history_turns: 0.
    "Mini Ada yüzebilir mi?" iki ayar arasında gidip geliyor (cevapsız 7-8/8).
    Ayrıca: İngilizce cevapta "4 Aralık 1996" → "April 4, 1996" (Türkçe bağlamı çevirirken ay hatası, 3B sınırı).
55. **Sohbet cevapları:** Tek örnekli prompt ("Nasılsın?" → "İyiyim") model tarafından her şeye uygulanıyordu:
    "Teşekkürler"/"Görüşürüz" → "İyiyim, teşekkür ederim"; "Thank you"/"Goodbye" → kendini tanıtma. İltifatlar
    ("Harikasın") sohbet olarak tanınmıyordu. Her tür için ayrı örnek (selam, hal hatır, teşekkür, iltifat, veda,
    kim olduğu) + yönlendiriciye iltifat örnekleri → 14 sohbet cümlesinde ~6 → 13 uygun cevap. Kalan: "Sağ ol" →
    "İyiyim". Not: noktalamasız "Harikasın" 0.900 skor aldı (eşik 0.90), Whisper'ın ürettiği "Harikasın." 0.928.
    Regresyon: 49/53 değişmedi.
56. **Sohbet: tek tek örnek yerine benzerlikle kategori (app/agent/smalltalk.py):** 6 kategori (selam, hal hatır,
    teşekkür, iltifat, veda, kimlik), her birinde birkaç örnek + TR/EN hazır cevap; yeni cümle en benzer örneğin
    kategorisine gidiyor, cevap LLM'siz. Örneklerde OLMAYAN 24 ifade: 22/24 doğru kategori ("Sağ ol", "Thanks a
    lot", "Süpersin", "You rock", "Kimsin sen?"...). Eşik 0.88 (doğrular 0.869-0.968); altında kalan LLM'e gider.
    - Yönlendiricinin "sohbet mi?" kapısı yeni ifadeleri kaçırıyordu (0.90/0.10). 16 yeni sohbet ifadesi + 12 bilgi
      sorusuyla yeniden ölçüldü: fark değeri ayırıyor (sohbet +0.092..+0.172, bilgi en çok +0.054), mutlak skor
      ayırmıyor → min 0.86, fark 0.08.
    - Yan etki: "How fast is it?" sohbet sanıldı (0.886, "how is it going"a benziyor) → takip 4/4 → 2/4. Çözüm:
      ölçü soran kelimeler ("kaç", "ne kadar", "how fast/much/long"...) varsa sohbet sayılmıyor → takip 4/4.
    - Sonuç: bilinen 14 sohbet cümlesi 14/14, yeni 12 ifadede 10 uygun; sohbet cevabı ~1 sn → ~20 ms.
      Regresyon 48/53 (hiçbir bilgi sorusu sohbete gitmedi; fark "Hangi robotlarınız var?" dalgalanması).
57. **PDF kaynak desteği (app/agent/pdf.py):** knowledge/ klasörüne PDF bırakmak yeterli. Resmi Robot Kol kataloğu
    (akinsoft.com.tr, 40 MB, 4 sayfa) indirildi — robot kol bilgi tabanında hiç yoktu. pypdf 0 karakter verdi: metin
    resim olarak gömülü → sayfa görsele çevrilip EasyOCR (tr+en) ile okunuyor (~7 sn/sayfa, data/ocr_cache'te önbellek).
    - Düz OCR iki sütunlu tabloyu "Ağırlık, Taşıma Kapasitesi, 27 kg, 5 kg" diye okudu → İngilizce soruya "payload
      27 kg" (yanlış), Türkçe sorular reddedildi: 0/7.
    - Düzene duyarlı OCR: kutu konumlarıyla her değer, üstündeki aynı hizadaki küçük etiketle eşleşiyor
      ("Ağırlık / Weight: 27 kg"). İlk sürüm "27 kg"yi 205 px'lik "Robot Kol" başlığının kopyası sanıp attı → kopya
      penceresi yarım satıra daraltıldı. 7 özelliğin 7'si doğru eşleşiyor. Cevaplar: 3/7 (payload 5 kg, kontrol, gripper
      açılma boyu); kalan hatalar gürültülü OCR metni (TR/EN iç içe, "Tasıma", başlıksız liste) + 3B modelin temkini.
    - Veri tutarsızlığı: Robot Kol kataloğundaki ağırlık (27 kg) ve ölçüler (44×75×58) sitede ARAT için yazanla aynı.
    - Regresyon: 57 soruda 50/57 (eski 53 soru aynı seviyede, robot kol 2/4).
58. **Embedding karşılaştırması (scripts/compare_embeddings.py, eşikten bağımsız):** 27 etiketli soru + 8 konu dışı.
    e5-small (118M): 1. sıra 24/27, ilk 2 25/27, boşluk −0.006 (3 cevaplanabilir soru konu dışından düşük), 13 ms/soru,
    indeks 1.7 sn. BGE-M3 (568M, ~2.3 GB): 25/27, 26/27, boşluk **+0.061 (çakışma yok)**, 94 ms/soru, indeks 14.8 sn.
    BGE-M3'ün asıl kazancı "bilmiyorum" eşiğinin güvenilirliği; doğrulukta +1. Geçiş config'te 2 satır (model + önekler,
    `rag.query_prefix/passage_prefix`) ama 7 benzerlik eşiğinin hepsi e5 ölçeğine göre (BGE skorları 0.39-0.45 aralığında)
    → yeniden ölçüm gerekir. Teslim işleri öncelikli olduğu için e5'te kalındı; ilk iyileştirme adayı.
59. **Çıkarım + NLI cevap doğrulaması (app/agent/verify.py):** "AKINSOFT ile AKINROBOTICS farkı nedir?" iki doküman
    bulunduğu halde reddediliyordu (sıkı kural birleştirmeyi engelliyor). Literatürden: RAGAS "faithfulness" / NLI ile
    doğrulama. Prompt "parçaları birleştirebilirsin" diye gevşedi; her cümle seslendirilmeden önce çok dilli NLI
    (mDeBERTa-v3-base-xnli) ile bağlama karşı kontrol ediliyor, desteklenmeyen cümle söylenmiyor.
    - 12 cümlelik testte 11/12 doğru karar. PyTorch'ta 1.6 sn/kontrol (DeBERTa dikkat katmanı bu sürümde yavaş;
      int8 nicemleme hata verdi), ONNX Runtime'a çevrilince 142 ms (gerçek bağlamlarla ~340 ms). Küçük MiniLM-NLI:
      13 ms ama 9/12.
    - Eval'e 5 çıkarım sorusu (62 soru). Eşik 0.5 ile: bilgi 21 → 13/27, toplam 40/62 — doğru cümleler eleniyordu.
    - Kalibrasyon (scripts/calibrate_verifier.py): doğru 55 cümlenin skorları 0.01-1.00 arası (sayılar, sıralı liste,
      bozuk Türkçe, Türkçe bağlama İngilizce cevap; "UV-C 70 metrekare" birebir dokümanda olduğu halde 0.01). Eşik 0.5'te
      42/55, 0.10'da 52/55 geçiyor. Sınırlama: bu çalıştırmada uydurma cümle tek (0.00); ilk testte uydurmalar 0.02-0.29.
    - Sonuç (62 soru): A sıkı kural 51/62, çıkarım 2/5, cevapsız 7/8, ilk cümle 1449 ms · B gevşek 53/62, 4/5, **6/8**,
      1634 ms · **C gevşek+NLI@0.10: 52/62, 3/5, cevapsız 8/8, araç 11/11, ilk cümle 2077 ms (+~0.6 sn)**. C seçildi:
      halüsinasyon case'te ayrı kriter. Not: test çıkarım cevaplarını sert değerlendiriyor ("Roboliza AROS ailesinde
      bir yazılımdır" doğru ama "bulut" aranıyor).
60. **Temiz sesli ölçüm oturumu (son ayarlar, NLI açık, 44 tur, logs/turns.jsonl):** yanıt gecikmesi medyanı
    RAG 3.6 sn (eski 5.1), araç 3.3, sohbet 1.8, cevap yok 1.8; p95 RAG 5.5. RTF STT 0.54, TTS 0.06. Planlanan 28
    sorudan: cevapsız 6/6 uydurma yok; hataların çoğu STT ("Harikasın" → "Haydi Kasım" 4 kez, "arızalı" → "arzanın",
    "What time is it?" → Türkçe sanılıp "Ne zaman bu?" diye ÇEVRİLDİ, "Saat kaç?" → "Alright, let's cut").
    Bulgular: (1) Türkçe öncelikli dil kuralı (kayıtlı sette 18/18) canlıda kısa İngilizce soruları Türkçeye çekiyor;
    (2) NLI sadakati kontrol ediyor, ilgiyi değil: yanlış duyulan "Hangi robot arzanın?" → dokümandaki AR-AS anlatıldı.
61. **Ayrılmış test seti (tests/agent_questions_heldout.csv, 36 soru, geliştirmede hiç kullanılmadı):** bir kez
    çalıştırıldı, ardından ayar YAPILMADI. **31/36 (%86)** — geliştirme seti 52/62 (%84): ayarlar genelleşiyor.
    Hatalar: 4 aşırı temkin (Servis Robotu V3 şarj süresi, Roboliza, AKINSOFT kaç ülke (EN), Ar-Mobil (EN));
    1 sınırda ("camera on its back?" → "AKINCI-5 has a depth camera": uydurma değil ama red de değil).
62. **Serbest sesli test (kullanıcının kendi soruları, 12 tur):** 4 doğru (AKINSOFT özeti, 120+ yazılım, ERP
    ürünleri), 4 konuda dürüst red veya kısmi (WOLVOX MRP detayı, dil seçeneği sayısı, saha personeli, ödül sayısı),
    **0 hatalı/uydurma**. Hepsi bilgi tabanı KAPSAMI eksiğiydi (AKINSOFT özetimde yoktu; sitede var) — model doğru
    davrandı. Sonrasında akinsoft.com.tr ana sayfasından "AKINSOFT Rakamlarla" (16 dil, 112 ödül, 4.957 saha
    personeli, 143 sektör...) ve "WOLVOX MRP" bölümleri eklendi; WOLVOX sözlüğe ve konu listesine eklendi ("VOLVOX"
    diye duyulmuştu). Dört soru artık doğru; İngilizce "How many awards?" ilgili bölümü bulamıyor (diller arası arama).
    Regresyon: STT 23/23, WER 0.31; ajan 51/62 (±1). README'ye serbest test EKLEMEDEN ÖNCEKİ haliyle yazılacak.
63. **Ada-7 dil listesi ve sesler eklendi (kullanıcının sitedeki metni). NLI liste sorunu:** virgülden bölünen liste
    parçaları ("bağırma ve sessizlik gibi.") tek başına elendi. Parçayı cümlenin önceki parçalarıyla birlikte kontrol
    etmek yetmedi: NLI, dokümandan KELİMESİ KELİMESİNE kopyalanmış iki noktalı liste cümlesine 0.05 verdi (aynı bilgi düz
    cümleyle 0.87). Çözüm: önce kelime örtüşmesi — cümlenin kelimelerinin ≥%85'i bağlamda varsa NLI'sız kabul. İstisna:
    tek başına sayı içeren cümleler ("24 saat dayanır" — kelimeler bağlamda var ama sayı yanlış eşlenmiş) her zaman
    NLI'a gider; "Ada-7"/"V3" içindeki rakam sayı sayılmaz. Uydurmalar bağlamda olmayan kelime içerdiği için NLI'a gider.
    Sonuç: **54/62** (en iyi), bilgi 21/27, cevapsız 8/8, araç 11/11; ses listesi tam cevaplanıyor.
64. **Otomatik web yükleyici (scripts/fetch_web.py):** iki sitenin llms.txt'indeki 37 sayfa + llms-full.txt arşivleri;
    trafilatura ile ana metin, sayfalar arası tekrar eden 92 paragraf, sayfa içi (≥%85 aynı) tekrarlar, etiketsiz tablo
    hücreleri ve URL'ler temizlendi. robots.txt uyuldu — Python robotparser kendi varsayılan kimliğiyle reddedilip
    "hepsi yasak" diyordu (yanlış alarm), dosya kendi kimliğimizle okunuyor.
    - Parçalama hataları bulundu: başlıkla metin arasında boş satır yoksa bütün blok "başlık" sayılıyordu (13.745
      karakterlik parça, ilk cümle 17 sn) → başlık = ilk satır; uzun paragraflar satır/cümle sınırından bölünüyor.
    - Ölçüm (62 soru): elle hazırlanmış 32 parça **54/62, 2.1 sn** · +37 sayfa (172 parça) 51/62, 2.9 sn ·
      +arşivler (848 parça) dense 49/62, 3.3 sn; hibrit 51/62, 3.5 sn. Sebepler: gürültü, eski haber bilgisi
      ("AKINSOFT 24 yıllık" — doğrusu 31), İngilizce sorularda Türkçe sayfalara kelime eşleşmesi yok.
    - Karar: web içeriği varsayılan indekste değil (knowledge_web/, rag.extra_dirs ile eklenebilir). Eksik bilgiler
      elle hazırlanmış dokümanlara eklendi. "Daha fazla veri her zaman daha iyi değil."
65. **Hibrit arama (BM25 + vektör, RRF):** Türkçe için kelimenin ilk 5 harfi kök (Can vd., 2008). Eşikler değişmesin
    diye skorlar vektör benzerliği; hibrit sadece sıralamayı değiştiriyor. scripts/eval_retrieval.py (LLM'siz: beklenen
    cevap ilk 2 parçada mı?): 848 parçada geliştirme seti dense 27/32 → hibrit 30/32 (ayrılmış set 22 → 20/22, raporlandı;
    seçim sadece geliştirme setiyle yapıldı); 32 parçada ikisi eşit (50/54). Admin panelinden doküman eklendikçe bilgi
    tabanı büyüyeceği için hibrit varsayılan.
66. **AkınVoice: dil seçimi ve admin paneli.**
    - Konuşma dili sayfadan seçilebiliyor (otomatik/TR/EN). Seçilirse Whisper dil algılamayı atlar. Kayıt 14
      ("Hello, how are you?") için: otomatik → en, en → en, tr → "Merhaba, nasılsınız?" (Whisper zorlanan dile
      çeviriyor; bu yüzden varsayılan otomatik). Arayüz dili (TR/EN) bundan ayrı bir seçim.
    - `/admin`: dosya/URL ekle, sil, yeniden indeksle, etkin bileşenleri gör. `ADMIN_TOKEN` yoksa kapalı.
      multipart bağımlılığı eklememek için dosya ham gövde olarak gönderiliyor. Dosya adı temizleniyor
      (`../` yok), PDF imzası ve UTF-8 kontrol ediliyor; yalnızca indekslenen klasörlerdeki belgeler silinebiliyor.
    - Sunucunun yeni indeksi fark etmesi parça sayısına bakarak yapılıyordu. Belge düzenlenince sayı aynı kalabilir,
      bu yüzden artık koleksiyon kimliğine bakılıyor (arama 14 ms, fark yok). İndeksleme tur kilidiyle yapılıyor,
      yani yarım indeksten cevap verilmiyor.
    - Model/araç değiştirme bilerek panelde yok: modeller açılışta yüklenip ısıtılıyor, değişiklik zaten yeniden
      başlatma istiyor; `config.yaml` tek doğru kaynak olarak kalıyor.
67. **Doğrulayıcıda iki yöntem hatası (sesli testte bulundu).**
    - Doğru cümle silindi: "AKINCI-5'in en yüksek hızı saniyede 2,5 metredir." AKINCI-5 parçasına göre 0.994, iki parça
      birleştirilince 0.008 (öbür parçadaki başka robotların hızları çelişki gibi okunuyor). Artık önce parça parça
      skorlanıyor; birleşik bağlam yalnızca hiçbir parça tek başına desteklemezse deneniyor.
    - Yanlış cümle geçti: "…166 santimetre veya 1,66 metre olup, bu 5,43 metreye eşittir." 0.97 aldı; NLI ana iddiaya
      bakıp eklenen sayıyı kaçırıyor. Yeni kural: cümledeki her bağımsız sayı bağlamda geçmeli (2,5 = 2.5).
    - Eşik değişmedi (0.10). Geliştirme seti, aynı koşullarda eski/yeni: **54 → 56/62**, bilgi 21 → 23/27,
      cevapsız 8/8 korundu. Kazanılan: AKINCI-5 boyu, UV-C alanı, Mini Ada boyutları. Kaybedilen: "Hangi yazılım
      servisleriniz var?" (eski "doğru" cevap zaten zayıftı: "AROS servis yazılımlarını kullanır."; yenisi "bilgi yok").
    - Bedel: ilk cümle ortalaması 2278 → 2495 ms (+0.2 sn), parça başına ayrı NLI çağrısı yüzünden. İki parça tek
      ONNX çağrısında (batch) skorlanarak azaltılabilir.
68. **Konu takibi cevaptan da güncelleniyor.** Sesli testte: "En hızlı robot hangisi?" → "AKINCI-5" (araç), ardından
    "Ne kadar hızlı?" → Ada-7'nin hızı söylendi; konu yalnızca sorulardan alındığı için bir önceki sorudaki Ada-7
    kalmıştı. Kural: soru hiçbir ad içermiyor ve cevap tam olarak bir ad içeriyorsa konu o ad olur; kullanıcının
    sorudaki adı her zaman önceliklidir, birden çok ad (liste) belirsiz sayılıp konuyu değiştirmez.
    Geliştirme seti: 56/62 (değişmedi), takip 4/4. Aynı akış metinle tekrarlandı: "AKINCI-5: Ne kadar hızlı?" → 2,5 m/s.
    Gözlem: konuşma dili "Türkçe" seçilince STT 1.2 → 0.7-0.8 sn (otomatik modda belirsiz dil ikinci çözümleme yapıyor).
69. **Soru olmayan tepkiler ("Süper!").** "AKINCI-5: Süper!" olarak arandı ve AKINCI-5 bilgisi okundu. Embedding
    ayırmıyor: tepkiler bilgi tabanına 0.80-0.83 (takip soruları gibi), sohbet skorları da çakışıyor (0.870-0.936 /
    0.814-0.886). Dilbilgisel kural (`router.is_request`): soru işareti, soru eki, soru kelimesi veya istek fiili.
    Geliştirme setinin 62 sorusunun hepsi soru sayıldı; görülmemiş 12 tepkinin hiçbiri. Soru olmayan, ad içermeyen
    ve yönlendirilmeyen ifade → sabit onay cevabı (LLM yok). Sohbet sınıflandırıcısı "Süper!"i selamlama sanıyordu,
    bu yüzden kategori cevabı değil tek nötr cevap. Geliştirme seti 56/62; bir koşuda cevapsız 7/8 çıktı, tek başına
    ve tekrar koşuda 8/8 (koşular arası ±1, o sırada eşzamanlı sesli test de vardı).
70. **Whisper halüsinasyonları.** Sesli testte gürültüden "Bu videoyu izlediğiniz için teşekkürler." yazıldı (Whisper'ın
    video altyazılarından öğrendiği bilinen hata). İki genel önlem: openai-whisper'ın varsayılan sessizlik kuralı
    (no_speech_prob > 0.6 ve avg_logprob < -1.0 → segment atılır) ve bilinen altyazı kalıpları listesi. eval_stt:
    WER 0.31, dil 23/23 (değişmedi). Kayıt 19 ("Saat kaç?") GPU'da deterministik değil: bazen "AKINCI-5", bazen
    hotwords cümlesinin yankısı (var olan yankı filtresi siler).
71. **AkınVoice kendini anlatıyor.** "What is the AKIN voice?" cevapsızdı. `knowledge/akinvoice_tr.md` eklendi; konu
    listesine "Akın Voice" (boşluk isteğe bağlı: "AKIN voice", "AkınVoice" eşleşir). İlk sürümde model "AKINROBOTICS
    tarafından geliştirilen" dedi ve NLI geçirdi; geliştiren bilgisi tanım cümlesinin içine alındı ("bir adayın
    değerlendirme görevi olarak geliştirdiği ...; AKINROBOTICS veya AKINSOFT'un ürünü değildir"). Türkçe cevaplar
    doğru; İngilizce cevaba ilgisiz ve yanlış çevrilmiş bir AKINSOFT cümlesi ekleniyor ("cybersecurity") — NLI
    alakayı ölçmüyor, diller arası çeviri zayıf: İngilizce kaynak eklemek gelecek adım.
72. **Arka planda indeksleme ve konu önerileri (kullanıcı geri bildirimi: "indeksleme uzun sürdü, config'e ad
    eklemek kullanıcı dostu değil").**
    - Mini Ada kılavuzu (20 sayfa, hiçbirinde metin katmanı yok) OCR ile ~3.5 dk sürdü ve bu sırada tur kilidi
      tutulduğu için asistan cevap veremedi. Yeni düzen: hazırlık (okuma, OCR, parçalama, embedding) kilitsiz, yalnızca
      indeks değişimi (~1 sn) kilit altında; yükleme/silme indekslemeyi kendisi başlatır; panel ilerlemeyi gösterir.
      İkinci indeksleme (OCR önbellekte) 15 sn.
    - Otomatik konu çıkarma ölçüldü: adların çoğunu buldu ama çıktının ~%40'ı gürültü ("Durum", "QR", "Teknik
      Çizimler"); gürültülü bir konu sonraki soruları sessizce yeniden yazar. Bu yüzden öneri + tek tıkla onay;
      onaylananlar data/topics.json'da, sunucu yeniden başlamadan geçerli. Öneriler yalnızca yeni/değişen belgeler için.
    - Refaktör: belge okuma/PDF/parçalama/web `app/knowledge/` paketine taşındı (Retriever yalnızca indeks ve arama);
      `search()` paylaşılan `best_score` alanı yerine `SearchResult` döndürüyor; `Agent.answer` adımlara bölündü;
      factory'de STT/LLM/TTS/araçlar aynı kayıt (registry) kalıbında; admin artık private alanlara ve scripts'e
      bağımlı değil. Bulunan hata: senkron silme uç noktası arka plan görevini başlatamıyordu (500) — async yapıldı.
73. **Son temizlik ve Docker.**
    - Mini Ada kılavuzu bilgi tabanından çıkarıldı: temiz ölçümde kılavuzsuz 55/62, kılavuzla 49/62 (3 araç sorusu
      o koşuda sunucu yeniden başlatıldığı için düştü; geri kalan kayıp gerçek: takip, çıkarım ve "yüzebilir" uydurması).
      Yalnızca kılavuzda olan bilgiler (acil durum, KVKK, pil tipi) kaybedildi; panel demosu için kullanılır.
    - "Ekranı." gibi soru işareti düşmüş eksiltili sorular tepki sanılıyordu. Ayırt edici özellik: bilgi tabanındaki bir
      kelimeyi içermesi (tepkilerde 13/14 yok, eksiltili sorularda 7/8 var; "tamam" gibi söylem belirleyiciler hariç).
    - Promptlardaki kimlik "AKINROBOTICS'in sesli asistanı" → "AkınVoice". Geliştirme seti 57/62 (bilgi 25/27,
      cevapsız 8/8). Yorumlardaki günlük dili ("seen live") kaldırıldı; scripts/test_stt.py ve test_tts.py silindi.
    - Docker: requirements.txt'te rank-bm25 eksikti (imaj açılışta çökerdi); torchvision CPU dizininden kuruluyor
      (yoksa pip torch'u CUDA sürümüyle değiştiriyor); CUDA kütüphaneleri (~1.2 GB) requirements-gpu.txt'e ayrıldı,
      yalnızca GPU imajına kurulur; indeksleme derleme sırasında bir kez yapılıp OCR önbelleği imaja alınır.
74. **Değerlendiricinin kurulumu.** GitHub'dan boş klasöre klonlayıp `--no-cache` derleme ile sıfırdan denendi:
    derleme ~15 dk, Ollama imajı 3,8 GB (~13 dk), model 2 GB (~7 dk), açılış ~70 sn — toplam ~36 dk (CPU modu).
    En büyük kalem Ollama'nın resmi imajı. Kararlar: (1) hazır imaj GHCR'da (`ghcr.io/icloolg/akinvoice`, public),
    `docker compose up` derlemek yerine onu çeker; (2) tek imaj hem GPU hem CPU (CUDA kütüphaneleri imajda,
    7,2 GB); `docker-compose.yml` GPU varsayılan, `docker-compose.cpu.yml` GPU'suz yedek — yalnızca GPU'ya bağlı
    bir kurulum NVIDIA kartı olmayan değerlendiricide hiç açılmazdı; (3) bilgisayarda Ollama varsa
    `OLLAMA_BASE_URL` ile Ollama imajı ve model indirmesi atlanır. Docker GPU (WSL2) doğrulandı: Whisper cuda,
    Ollama %100 GPU, STT 0,7–0,8 sn, sohbet 1,3 sn, bilgi 2,7–4,7 sn (CPU modunda STT 2–4,5 sn).
    Sorun: C: diski dolunca (6,9 GB boş) Docker motoru CUDA paketlerini açarken kilitlendi; durdurulan görevlerin
    docker-compose süreçleri arkada çalışmaya devam ediyordu. Eski imajlar silinip Docker yeniden başlatıldı.
    Dockerfile'da iki requirements dosyası ayrı katmana alındı (birindeki değişiklik diğerini yeniden kurdurmasın).
75. **Demo provasında iki sorun.** (1) "Ada-7'in boyu kaç cm?" → 19,8 sn ve "bilgi bulamadım". İlk token 16,7 sn:
    Ollama modeli 30 dk boşta kalınca bellekten atmış, yeniden yükleme Docker'da ~15 sn. `keep_alive: -1` ile model
    hiç boşaltılmıyor. (2) Model "166 santimetre veya 1,66 metre olup, bu 5,43 metreye eşittir" yazdı; sayı kontrolü
    cümleyi doğru şekilde sildi ama doğru bilgi de gitti. Prompta "birim dönüştürme, hesap yapma" kuralı denendi:
    3B model yine dönüştürdü ("0,0007 kilometre") ve kural Çinceye kaymaya yol açtı ("AKINCI-5机器人是最快的。";
    kurallı 1/8, kuralsız 0/8) — kural geri alındı. Değerlendirme betiği bunu yakalamadı (beklenen kelime "AKINCI-5"
    geçiyordu), NLI de çok dilli olduğu için geçirdi: Latin dışı harf içeren cümle artık hiç seslendirilmiyor.
    Asıl çözüm: reddedilen cümle bağlaçlardan ("veya", "olup",
    "yani", virgül) cümleciklere bölünür ve aynı kontrolleri geçen en uzun baş kısım söylenir → "Ada-7'in boyu 166
    santimetre." Cevapsız sorularda kurtarılacak destekli parça olmadığı için güvenlik değişmedi. Geliştirme seti
    57/62 (cevapsız 8/8, araç 11/11). Dockerfile: modeller koddan önce indiriliyor; kod değişikliği 1,5 GB'lık
    model katmanını geçersiz kılmıyor (sonraki derlemeler 15–40 dk yerine ~1,5 dk).
76. **Durdur/Başlat sohbeti sıfırlıyordu.** Demo provasında "Ada-7'in boyu kaç cm?" → (Durdur, anlatım, Başlat) →
    "Peki kaç kilogram?" Ada-7 yerine AMR/ARAT'ın yükünü cevapladı. Log: istemci bağlantıyı kapatıp yeniden açmış,
    sunucu yeni bağlantıyı yeni sohbet saymış, konu kaybolmuş. Kullanıcı için "Durdur" mikrofonu kapatmaktır,
    sohbeti bitirmek değil. Artık her sekme bir sohbet kimliği tutuyor (sessionStorage) ve `/ws?conversation=<id>`
    ile bağlanıyor; sunucu geçmişi ve konuyu kimliğe göre saklıyor (en fazla 100 sohbet, 1 saat kullanılmayan silinir).
    Sohbeti yalnızca "Sıfırla" bitiriyor. Ayrıca LLM ısıtması 120 sn'de zaman aşımına uğrayıp açılışı düşürmüştü
    (GPU dolu, model yavaş yüklendi): ısıtma için zaman aşımı 10 dk, konteynere `restart: unless-stopped`.
46. **Aynı klasörde iki düzenleme oturumu:** İkinci oturum `agent.py`'yi eski kopyasıyla ezdi; 42. maddenin
    kodu ve bu notların 42-45'i kayboldu, inceleme sırasında fark edilip geri getirildi. Ders: tek oturum yazar.

Açık (ölçülecek):
- [ ] `rag.chunk_max_chars: 450` → bölümler paragraf bazında bölünür, bağlam ~yarıya iner; ingest + eval_agent
      ile doğruluk/gecikme karşılaştırılacak (darboğaz bağlam token sayısı, ~5 ms/token).
- [ ] `ollama ps` çıktısında modelin %100 GPU'da olduğu doğrulanacak; değilse `llm.num_ctx: 2048`.
- [x] robot_specs: sütun eşleşmeyince ("en akıllı robot") bütün tablo bağlama giriyor → ilk ses 8,6 sn. (42. madde)
- [ ] README'deki gecikme tablosu son ayarlarla temiz bir oturumdan yeniden üretilecek.
