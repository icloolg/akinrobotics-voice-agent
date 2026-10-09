# AkınVoice Sesli Asistan

Kaynak: http://localhost:8000 (AkınVoice proje açıklaması)

## AkınVoice Nedir

AkınVoice (Akın Voice), bir adayın değerlendirme görevi olarak geliştirdiği sesli yapay zeka asistanıdır; AKINROBOTICS veya AKINSOFT'un ürünü değildir. AkınVoice, AKINROBOTICS robotları, AKINSOFT yazılımları ve bu iki şirket hakkındaki soruları sesli olarak cevaplar. Kullanıcı konuşur, AkınVoice soruyu anlar, cevabı bilgi kaynaklarından bulur ve cevabı sesli olarak söyler.

## AkınVoice'u Kim Geliştirdi

AkınVoice bir aday projesidir: AKINROBOTICS'teki Yapay Zeka Yazılım Mühendisi pozisyonunun değerlendirme görevi olarak bir aday tarafından geliştirilmiştir. AkınVoice, AKINROBOTICS veya AKINSOFT'un resmi bir ürünü değildir.

## AkınVoice Neleri Cevaplar

AkınVoice; Ada-7, Mini Ada, Servis Robotu V3, Fabrika Servis Robotu, UV-C Sterilizasyon Robotu, AMR, robot kol, ARAT ve AKINCI-5 robotlarının özelliklerini, robotların anlık durumunu, AROS yazılımlarını, AKINSOFT ürünlerini ve şirket bilgilerini cevaplar. Saati ve tarihi de söyleyebilir. Bilgi kaynaklarında olmayan bir soruya bilgi uydurmaz; bilgi bulamadığını söyler.

## AkınVoice Hangi Dilleri Konuşur

AkınVoice Türkçe ve İngilizce konuşur. Soru hangi dilde sorulursa cevabı o dilde verir.

## AkınVoice Nasıl Çalışır

AkınVoice tamamen yerel ve açık kaynak bileşenlerle çalışır. Konuşmayı Whisper modeli yazıya döker, soru bilgi kaynaklarında aranır, cevabı Qwen dil modeli kaynaklara dayanarak yazar, her cümle bir doğrulama modeliyle kaynağa göre kontrol edilir ve Piper ses modeli cevabı seslendirir. Cevap cümle cümle seslendirildiği için kullanıcı ilk cümleyi model diğer cümleleri yazarken duyar.
