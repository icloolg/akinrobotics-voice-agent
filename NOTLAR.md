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
    + örnek → "İyiyim, teşekkür ederim." "Merhaba nasılsın?" hâlâ bazen tekrar ediyor.
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
46. **Aynı klasörde iki düzenleme oturumu:** İkinci oturum `agent.py`'yi eski kopyasıyla ezdi; 42. maddenin
    kodu ve bu notların 42-45'i kayboldu, inceleme sırasında fark edilip geri getirildi. Ders: tek oturum yazar.

Açık (ölçülecek):
- [ ] `rag.chunk_max_chars: 450` → bölümler paragraf bazında bölünür, bağlam ~yarıya iner; ingest + eval_agent
      ile doğruluk/gecikme karşılaştırılacak (darboğaz bağlam token sayısı, ~5 ms/token).
- [ ] `ollama ps` çıktısında modelin %100 GPU'da olduğu doğrulanacak; değilse `llm.num_ctx: 2048`.
- [x] robot_specs: sütun eşleşmeyince ("en akıllı robot") bütün tablo bağlama giriyor → ilk ses 8,6 sn. (42. madde)
- [ ] README'deki gecikme tablosu son ayarlarla temiz bir oturumdan yeniden üretilecek.
