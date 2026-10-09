# AkınVoice — Teknik Tasarım Ayrıntıları

README'nin özetlediği kararların ayrıntıları. Her kararın ham ölçümü ve denenip vazgeçilen seçenekler [`NOTLAR.md`](../NOTLAR.md) dosyasındadır.

## İçindekiler

1. [Kod düzeni](#kod-düzeni)
2. [Bileşen seçimleri](#bileşen-seçimleri)
3. [Kullanım](#kullanım)
4. [Halüsinasyon kontrolü](#halüsinasyon-kontrolü)
5. [Gecikme analizi](#gecikme-analizi)
6. [Doğruluk ölçümleri](#doğruluk-ölçümleri)
7. [Bilgi tabanının kapsamı](#bilgi-tabanının-kapsamı)
8. [Konuşma deneyimi](#konuşma-deneyimi)
9. [Özelleştirme ve yönetim paneli](#özelleştirme-ve-yönetim-paneli)
10. [Testler ve ölçüm betikleri](#testler-ve-ölçüm-betikleri)
11. [Bilinen sınırlamalar ve sonraki adımlar](#bilinen-sınırlamalar-ve-sonraki-adımlar)

## Kod düzeni

```
app/
  main.py              FastAPI + WebSocket sunucusu, tur yönetimi, söz kesme
  pipeline.py          bir tur: STT → ajan → cümle bölücü → doğrulama → TTS, süre ölçümü
  admin.py             /admin API: belgeler, arka planda indeksleme, konu adları, bileşenler
  factory.py           bileşenleri config.yaml'dan kuran kayıt (registry) tabloları
  agent/               ajan, yönlendirici, hibrit arama, konu takibi, sohbet, doğrulama, promptlar
  knowledge/           belge okuma, PDF/OCR, parçalama, web, konu önerileri, indeksleyici
  tools/               saat, robot durumu (HTTP API), robot özellikleri (SQLite)
  providers/           VAD, STT, LLM, TTS arayüzleri ve uygulamaları
  static/              tarayıcı istemcisi, yönetim paneli, logo, robot görselleri
client/                Python istemcisi (mikrofon + hoparlör)
knowledge/             bilgi kaynakları (Markdown, PDF, robot özellikleri CSV)
scripts/               indeksleme, model indirme, değerlendirme, benchmark, site tarama
tests/                 birim testleri, değerlendirme soru setleri, ses test seti
config.yaml            bütün bileşen ve eşik ayarları
```

- **Sağlayıcı arayüzleri:** STT, LLM, TTS ve VAD `app/providers/base.py` arayüzlerini uygular. Ajan ve boru hattı yalnızca bu arayüzleri bilir.
- **Kayıt tabanlı fabrika:** `app/factory.py`, `config.yaml`'daki sağlayıcı adını bir kurucu fonksiyona eşleyen tablolar tutar (STT, LLM, TTS, araçlar). Kütüphaneler yalnızca seçilen sağlayıcı için içe aktarılır.
- **Ajan adımları:** `Agent.answer` sırasıyla arar ve yönlendirir (`_retrieve`), tepkileri ve takip sorularını çözer (`_is_remark`, `_resolve_followup`), ardından yola göre cevaplar (`_answer_tool`, `_answer_chitchat`, `_answer_rag`).
- **Arama durumsuzdur:** `Retriever.search()` paylaşılan bir alan yazmaz, `SearchResult(chunks, best_score)` döndürür; aynı retriever'ı paylaşan oturumlar birbirini etkilemez.
- **Bilgi tabanı paketi:** belge okuma, PDF/OCR, parçalama ve web indirme `app/knowledge/` paketindedir; retriever yalnızca hazır parçaları indeksler ve arar.

## Bileşen seçimleri

| Bileşen | Seçim | Neden |
|---|---|---|
| VAD | Silero VAD (ONNX, CPU) | 32 ms'lik parçaları ~1 ms'de puanlar; STT'ye yalnızca konuşma gider (gürültüde Whisper 10+ sn takılıyordu). 100 ms'den kısa sesler gönderilmez. |
| STT | faster-whisper `small`, GPU, int8 | Dili aynı geçişte algılar. `base` ürün adlarını yanlış duyuyor; `small` CPU'da 3,9 sn, GPU int8'de 0,9 sn; float16 GTX 1650'de 3 kat yavaş. Alan kelimeleri `hotwords` ile verilir. |
| Dil seçimi | Varsayılan Türkçe; İngilizceye yalnızca P(en) ≥ 0,3 ve P(en) > 2·P(tr) ise geçilir | Kısa Türkçe kelimeler düşük güvenle başka dil sanılıyordu. Üç kural 18 kayıtta karşılaştırıldı: düz algılama 16/18, "tr/en'den olasısı" 15/18, seçilen kural 18/18. Kullanıcı dili sayfadan seçerse algılama atlanır. |
| Embedding | `multilingual-e5-small` (CPU) | Tek modelle Türkçe ve İngilizce; İngilizce soru Türkçe dokümanda doğru bölümü buluyor. Yönlendirmede de kullanılır. |
| Arama | Hibrit: embedding + BM25 (Türkçe için 5 harf kök), Reciprocal Rank Fusion | Ürün adı ve sayı gibi birebir eşleşmeleri embedding kaçırabiliyor. Eşikler embedding ölçeğinde kalır; hibrit yalnızca sıralamayı değiştirir. |
| Parçalama | Markdown bölümü başına bir parça (en fazla 900 karakter) | Sabit uzunluklu pencereler ilgisiz konuları birleştiriyor, yanlış robotun değeri geliyordu. |
| Vektör deposu | Chroma (gömülü) | Ayrı servis gerektirmez; bu ölçekte arama ~20 ms. |
| LLM | `qwen2.5:3b`, Ollama, `temperature: 0` | 4 GB VRAM'e Whisper ile birlikte sığan, Türkçesi kullanılabilir en büyük model. |
| Doğrulama | mDeBERTa-v3-base-xnli (NLI), ONNX Runtime, CPU | Her cümle seslendirilmeden önce kaynağa karşı kontrol edilir. PyTorch'ta 1,6 sn/kontrol, ONNX'te 0,14–0,34 sn. |
| TTS | Piper (VITS/ONNX, CPU) | RTF ≈ 0,06–0,10; GPU'yu LLM ve STT'ye bırakır. Dil başına bir ses. |
| Araç seçimi | Anlamsal yönlendirme + anahtar kelime (LLM function-calling yok) | 3B model function-calling'de güvenilir değil ve fazladan LLM çağrısı 1–2 sn ekler. Yönlendirme ek model gerektirmez, ~0 ms. |
| Veritabanı aracı | Sabit sorgu + sütun beyaz listesi (text-to-SQL yok) | Konuşmadan gelen metni SQL'e koymak güvenlik riski; sayı filtresi SQL'de yapılır, LLM'de değil. |
| Takip soruları | Konu takibi (bilinen adlar), LLM ile yeniden yazma yok | "Kaç saat çalışır?" son konuşulan adla aranır: ~0 ms, takip sorularında 4/4. LLM ile yeniden yazma her soruya 1–2 sn ekler. |

## Kullanım

- Arayüz dili (TR/EN) ve konuşma dili (otomatik / Türkçe / İngilizce) sayfanın üstünden seçilir. Konuşma dili seçilirse Whisper dil algılamayı atlar: kısa sorularda hem doğruluk hem hız artar (STT ~1,2 → ~0,8 sn).
- Her cevabın altında kaynağı ve gecikme dökümü (STT, LLM ilk token, ilk cümle+TTS) görünür.
- Örnek sorular: "Ada-7 kaç kilogram?", "What is AROS?", "En hızlı robot hangisi?" → "Ne kadar hızlı?", "Şu an hangi robot arızalı?", "Ada-7 uçabilir mi?" (cevapsız), "Bugün hava nasıl?" (kapsam dışı).
- Tarayıcı yerine Python istemcisi: `pip install -r client/requirements.txt && python client/voice_client.py`. Mikrofon, tarayıcı kuralı gereği yalnızca `localhost` veya HTTPS üzerinden açılır.
- Docker: tek imaj hem GPU'da (`docker-compose.yml`) hem CPU'da (`docker-compose.cpu.yml`, `STT_DEVICE=cpu`) çalışır; CUDA kütüphaneleri imajın içindedir. Hazır imaj `ghcr.io/icloolg/akinvoice`; kaynaktan derleme ~15–40 dk (bağlantıya bağlı); konteyner ~70 sn'de hazır. Docker'sız kurulumda GPU yoksa `config.yaml` içinde `stt.device: cpu`.

## Halüsinasyon kontrolü

Tek bir önlem yetmediği için katmanlıdır; her katman ölçülerek eklenmiştir.

1. **Erişim eşiği.** En iyi parçanın embedding benzerliği 0,80'in altındaysa LLM hiç çağrılmaz, sabit "bilgi bulamadım" cümlesi söylenir.
2. **Yalnızca bağlam.** Sistem promptu dış bilgiyi yasaklar; araç çıktıları da aynı "yalnızca bağlam" kuralıyla LLM'e verilir.
3. **Cümle cümle doğrulama (NLI).** Her cümle seslendirilmeden önce çok dilli bir doğal dil çıkarımı modeliyle (mDeBERTa-v3-base-xnli, ONNX) kaynak parçalara karşı kontrol edilir; desteklenmeyen cümle söylenmez, hiçbiri kalmazsa "bilgi bulamadım" denir. Bu, RAG değerlendirme araçlarındaki *faithfulness* ölçütünün canlı hâlidir ve promptun farklı parçalardaki bilgileri birleştirmesine izin verir ("AKINSOFT ile AKINROBOTICS farkı nedir?").
   - Her cümle önce **parça parça** skorlanır: iki parça birleştirilince model öbür parçadaki başka robotların değerlerini çelişki sanıyordu (doğru bir cümle 0,994 yerine 0,008 aldı).
   - Eşik 0,10'dur, 0,5 değil: doğru cümlelerin skorları 0,01–1,00 arasında dağılıyor (sayılar, listeler, Türkçe kaynağa İngilizce cevap); 0,5'te 55 doğru cümlenin 13'ü eleniyordu (`scripts/calibrate_verifier.py`).
4. **Sayı kontrolü.** Cümledeki her sayı kaynakta geçmelidir. NLI ana iddiaya bakıp eklenmiş bir sayıyı kaçırabiliyor ("166 santimetre veya 1,66 metre olup, bu 5,43 metreye eşittir" 0,97 aldı).
5. **Hesap LLM'e bırakılmaz.** Karşılaştırma ve sayı filtreleri SQL'de yapılır; saat ve robot durumu araçtan gelir.
6. **Araç cevaplayamıyorsa zorlanmaz.** Araçta karşılığı olmayan soru (`ToolNotApplicable`) doküman yoluna düşer; araç hatası (`ToolUnavailable`) LLM'den geçmeden aynen söylenir.
7. **Sohbet olguya dönüşmez.** Selam, teşekkür, "Sen kimsin?" hazır cevaplarla; soru olmayan tepkiler ("Süper!", "Tamam.") sabit bir onayla cevaplanır. Tepki ayrımı dilbilgiseldir: soru işareti, soru eki, soru kelimesi veya istek fiili yoksa ve bilgi tabanındaki hiçbir kelime geçmiyorsa ifade tepkidir (geliştirme setinin 62 sorusunun hiçbiri tepki sayılmadı; görülmemiş 14 tepkinin 13'ü ayrıldı).
8. **STT halüsinasyonları elenir.** Whisper'ın sessizlik kuralı (no_speech_prob > 0,6 ve avg_logprob < −1) ve bilinen altyazı kalıpları ("izlediğiniz için teşekkürler") konuşma sayılmaz.
9. **İzlenebilirlik.** Her turda yol, kaynak dosya, skor ve elenen cümleler loglanır; tarayıcı her cevabın kaynağını ve gecikme dökümünü gösterir.

## Gecikme analizi

**Yanıt gecikmesi** = VAD'in beklediği 500 ms sessizlik + ilk cevap sesinin hazır olması.

Sürenin nereye gittiği (bilgi sorusu): VAD beklemesi 500 ms · STT ~0,9–1,2 sn · arama + yönlendirme ~20–50 ms · LLM'in bağlamı okuması 0,1–4 sn · ilk cümlenin yazılması + NLI doğrulaması + TTS ~0,9–2,5 sn.

- **Darboğaz LLM'in bağlamı okumasıdır.** GTX 1650'de prompt işleme ~200 token/sn'dir ve süre bağlamdaki token sayısıyla doğrusaldır.
- **Ollama'nın prompt önbelleği belirleyicidir.** Aynı belge parçaları daha önce okunduysa ilk token 0,1 sn'de gelir ("AKINSOFT nedir?": 94 ms); bağlam ilk kez okunuyorsa 2,3–4 sn sürer. README'deki tekrar üretilebilir ölçümde her kayıt yeni bir sohbettir, bu yüzden bağlamların çoğu ilk kez okunmuştur. Canlı oturumlarda konular tekrar ettiği için bilgi soruları çoğunlukla 2,4–3,7 sn'de cevaplanır.
- **STT hataları gecikmeyi de etkiler.** Bozuk yazıya dökülmüş bir soruda ilk cümle doğrulamada elenebilir; ilk ses bir sonraki cümleyi bekler.
- **STT süresi konuşma uzunluğundan neredeyse bağımsızdır:** Whisper her sesi 30 sn'lik pencereye tamamlayarak işler.
- Cümle cümle akış sayesinde "ilk ses" ile "toplam" arasındaki fark kullanıcıya bekleme olarak yansımaz.

Gecikme için yapılanlar ve ölçülen etkileri:

| Değişiklik | Etki |
|---|---|
| Ollama adresi `localhost` → `127.0.0.1` (Windows IPv6 beklemesi) | ilk token 2,6 sn → 0,3 sn |
| Whisper CPU → GPU int8 | 3,9 sn → 0,9 sn |
| top_k 3 → 2 | ilk cümle 3268 → 2515 ms (doğruluk ±1 soru) |
| Konuşma geçmişi yerine konu takibi | ilk cümle 2978 → 1683 ms, doğruluk 46 → 49/53, takip soruları 3/4 → 4/4 |
| Tek ortak sistem promptu (Ollama önbelleği), dil talimatı sonda | prompt işleme 2246 → 1891 ms |
| 100 karakteri geçen cümleyi virgülden bölme | 250 karakterlik tek cümle ilk sesi 7 sn'ye çıkarıyordu |
| Sohbet ve tepkiler için hazır cevap (LLM'siz) | ~1 sn → ~20 ms |
| Konuşma dili sayfadan seçilince | STT ~1,2 → ~0,8 sn (Docker CPU modunda 4,5 → 2,3 sn) |
| Açılışta LLM, STT ve TTS'i ısıtma (warm-up) | ilk turda model yükleme beklemesi yok |
| *(maliyet)* NLI cümle doğrulaması | bilgi cevaplarında ilk cümleye ~0,4–0,6 sn |

## Doğruluk ölçümleri

- Değerlendirme betiği (`scripts/eval_agent.py`) konuşulan cevabı ölçer, yani doğrulamadan geçen cümleleri. Cevapsız sorularda beklenen yasak kelimelerin ("uçabilir", "lira"…) geçmemesi kontrol edilir.
- **Ayrılmış set** (36 soru) geliştirme boyunca iki kez çalıştırıldı: ara sürümde 31/36, son sürümde 35/36. Sonuçlarına bakılarak hiçbir ayar yapılmadı. Tek hata: "Does AKINCI-5 have a camera on its back?" → "Yes, AKINCI-5 has a deep vision camera." Kaynakta derinlik kamerası var ama yeri yazmıyor; cümle kaynağa sadık olduğu için doğrulamadan geçti.
- GPU'da LLM çıktıları koşudan koşuya birebir aynı değildir: aynı set ±1 soru oynar (aynı araç sorusu iki ardışık koşuda bir doğru, bir yanlış çıktı).
- STT hataları cevabı çoğu zaman bozmaz: "Ada-7" yanlış duyulduğunda bile anlamsal arama doğru bölümü bulur. Kısa tek kelimelik sorular ("Saat kaç?") en zayıf noktadır.
- Embedding karşılaştırması (`scripts/compare_embeddings.py`, 27 etiketli soru): e5-small doğru bölüm 24/27, 13 ms/soru; BGE-M3 25/27 ve "bilgi yok" eşiğinde daha temiz ayrım, ama 568M parametre, 94 ms/soru ve bütün eşiklerin yeniden ölçülmesi gerekiyor.

## Bilgi tabanının kapsamı

Bilgi tabanını büyütmek her zaman iyi değil:

| Eklenen | Parça | Geliştirme seti | Not |
|---|---:|---|---|
| — (elle derlenmiş belgeler) | 37 | 55–57/62 | |
| Şirket sitelerinden 37 sayfa | +172 | 51/62 | gürültü, eski haberler; +0,8 sn gecikme |
| Mini Ada kullanım kılavuzu (20 taranmış sayfa, OCR) | +30 | 49/62 | "yüz tanıma" metni "Mini Ada yüzebilir" uydurmasına yol açtı |

Bu yüzden sitelerden indirilen sayfalar (`knowledge_web/`, `python -m scripts.fetch_web`) varsayılan olarak indekslenmez; yeni bir kaynak eklendiğinde değerlendirme seti yeniden çalıştırılır.

## Konuşma deneyimi

- **Söz kesme (barge-in):** Sunucu cevap verirken de dinler. Kullanıcı araya girerse (250 ms sesli konuşma) cevap durur, LLM üretimi iptal edilir ve yeni soru dinlenir. Daha kısa sesler (yankı, tıkırtı) yok sayılır. Tarayıcının yankı engelleyicisi gerekir; Python istemcisi konuşurken mikrofonu kapatır (yarı çift yönlü). `vad.barge_in: false` ile kapatılabilir.
- **Takip soruları:** "ARAT kaç kilo?" → "Kaç saat çalışır?" sorusu "ARAT: Kaç saat çalışır?" olarak aranır. Soru hiçbir ad içermiyorsa ve cevapta tek bir ad geçiyorsa ("En hızlı robot hangisi?" → "AKINCI-5"), konu o ad olur. LLM ile soru yeniden yazma seçilmedi: her takip sorusuna 1–2 sn ekler ve 3B model yeniden yazarken hata yapıyor.
- Cevabın altında adı geçen robotun görseli ve gecikme dökümü (STT, LLM ilk token, ilk cümle+TTS, kaynak) gösterilir.
- 100 ms'den kısa sesler STT'ye gönderilmez; konuşma hızı `tts.length_scale` ile ayarlanır; liste ve markdown işaretleri seslendirilmeden temizlenir.

## Özelleştirme ve yönetim paneli

- **Model değiştirmek:** `config.yaml`. Örneğin başka bir LLM sunucusu için `llm.provider: openai_compat`, `base_url`, `model`. Modeller açılışta yüklenip ısıtıldığı için yeniden başlatma gerekir.
- **Yeni sağlayıcı (STT/LLM/TTS):** `app/providers/base.py` arayüzünü uygulayan bir sınıf + `app/factory.py`'deki kayıt tablosuna bir satır.
- **Yeni araç:** `app/tools/base.py` içindeki `Tool`'dan türeyen sınıf (`examples`, isteğe bağlı `keywords`, `run()`), `app/factory.py`'deki `TOOLS` tablosuna bir satır, `config.yaml`'da `tools.enabled`. Ajan koduna dokunulmaz.
- **Yönetim paneli (`/admin`):** `ADMIN_TOKEN` ortam değişkeniyle açılır (yoksa kapalıdır; anahtar sabit sürede karşılaştırılır).
  - `.md` / `.txt` / `.pdf` yükleme veya web sayfası ekleme (ana metin alınır, robots.txt'ye uyulur), belgeleri parça sayılarıyla listeleme ve silme.
  - Ekleme ve silme arka planda yeniden indekslemeyi kendiliğinden başlatır. Okuma, OCR ve embedding kilitsiz çalışır, asistan önceki indeksle cevap vermeye devam eder; yalnızca indeks değişimi (~1 sn) kilit altındadır. Panel ilerlemeyi gösterir ("OCR 12/20"). OCR sonucu önbelleğe alınır.
  - **Konu adları:** yeni belgelerdeki aday adlar önerilir, doğru olanlar tek tıkla eklenir (`data/topics.json`) ve yeniden başlatmadan geçerli olur. Tam otomatik çıkarma ölçüldü: önerilerin ~%40'ı gürültüydü ("Durum", "QR"). Gürültülü bir ad sonraki soruları sessizce bozacağı için son karar insandadır.
  - Etkin bileşenler (STT, LLM, TTS, embedding, arama, doğrulama, araçlar) görüntülenir.
- **API dokümantasyonu:** Swagger UI `/docs` adresindedir; uç noktalar Yönetim, Sistem ve Örnek filo API'si olarak gruplanmıştır, WebSocket protokolü sayfanın başında anlatılır. Üretimde `DOCS_ENABLED=0` ile kapatılır.
- **Yeni bilgi kaynağı (panelsiz):** `knowledge/` klasörüne dosya, ardından `python -m scripts.ingest`. Kaynakta yeni özel adlar varsa `stt.hotwords` cümlesine de eklenmelidir. Ardından `tests/agent_questions.csv`'ye birkaç soru eklenip `python -m scripts.eval_agent` ile eski soruların bozulmadığı kontrol edilir.

## Testler ve ölçüm betikleri

```bash
pip install -r requirements-dev.txt
pytest                                   # model/GPU gerektirmez: metin, araçlar, doğrulama kuralları, sunucu, söz kesme, admin API
python -m scripts.eval_agent             # 62 soruluk doğruluk + gecikme (Ollama gerekir)
python -m scripts.eval_retrieval         # LLM'siz arama doğruluğu (dense / BM25 / hibrit)
python -m scripts.eval_stt               # kayıtlı seslerle WER, dil, RTF
python -m scripts.replay_benchmark       # kayıtları tam boru hattından geçirip gecikme ölçer
python -m scripts.benchmark              # logs/*.jsonl özeti + donanım bilgisi
python -m scripts.chat_text              # ajanla yazarak konuşma
```

## Bilinen sınırlamalar ve sonraki adımlar

- **Gecikme:** bilgi sorularında çoğu süre LLM'in bağlamı okumasıdır ve GPU ile ölçeklenir. Daha kısa parçalar (`rag.chunk_max_chars`) ve NLI kontrollerini tek ONNX çağrısında toplamak sonraki adaylardır.
- **STT:** Whisper `small` çok kısa sorularda ve marka adlarında hata yapabiliyor ("AKINSOFT" → "Bakın SOFT"); daha büyük model 4 GB VRAM'e LLM ile birlikte sığmıyor. Bilinen adlarla bulanık eşleştirme (ASR sonrası düzeltme) denenebilir.
- **3B model:** Türkçe dilbilgisi hataları ("santimetre'dür"), bir cümlede birden çok sayıyı karıştırma ve aşırı temkin ("AKINCI-5 yüzer mi?" → "AKINCI-5 yürüyebilir.").
- **Doğrulama alaka ölçmez:** NLI cümlenin kaynağa sadık olduğunu kontrol eder, soruyla ilgili olduğunu değil.
- **Diller arası:** kaynaklar Türkçe; İngilizce cevaplarda çeviri hatası olabiliyor ("bilişim" → "cybersecurity"). İngilizce kaynak eklemek bir sonraki adımdır.
- **Taranmış PDF'ler:** OCR metni gürültülüdür; üretimde düzen modeli (docling, PP-Structure) veya insan kontrolünden geçmiş metin önerilir.
- **Konu önerileri:** kurala dayalı; aynı onay akışıyla LLM'li öneri ölçülerek karşılaştırılabilir (indeksleme arka planda olduğu için kullanıcı gecikmesine etkisi yoktur).
- **Embedding:** BGE-M3'e geçiş, eşiklerin yeniden ölçülmesiyle birlikte ilk iyileştirme adayıdır.
- Robot durum API'si örnek bir servistir (`/mock/robots`); gerçek kullanımda filo yönetim sisteminin adresi verilir.
- GPU'da aynı anda tek tur işlenir; birden çok istemci sıraya girer (her istemcinin konuşma geçmişi ve konusu ayrıdır).
- Söz kesme, hoparlör çok yüksekse yankı yüzünden yanlış tetiklenebilir; kulaklıkla sorun olmaz.
