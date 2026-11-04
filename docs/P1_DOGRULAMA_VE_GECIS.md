# P1 takip düzeltmesi: doğrulama ve geçiş

Bu değişiklik PR #7'nin ilk sürümündeki doğrulama açıklarını giderir. Strateji
listesi ve incelemesi `P1_STRATEJI_ENVANTERI.md` dosyasındadır. Strateji
eşikleri, ağırlıklar ve maliyet oranları optimize edilmedi.

## Düzelen doğrulanmış sorunlar

- Eski sürecin yeni pozisyonları nakit etkisi olmadan birleştirip para yaratması:
  OS kilidi altında sürüm ve dosya özeti karşılaştırılır. Eski hesap reddedilir.
  `force=True` korumayı atlatmaz. Yeniden denemek için tüm karar yeniden
  hesaplanmalıdır. Dosya atomik yazılır, öncesinde ileriye dönük defter denetlenir.
- Rapor normalizasyonunda lot ve parasal kârın kaybolması: P1 alanları korunur.
  Kısmi satışlar kimlikli tamamlanan pozisyonlar üzerinde birleştirilir.
  Kimliksiz eski kayıtların ölçümü hâlâ olay bazlıdır ve bu kapsam belirtilir.
- Eski günlük kapanışın yeni alım/acil çıkış fiyatı olması: işlem uygunluğu
  sayısal fiyat geçerliliğinden ayrılır. Sorgulama zamanı veri zamanı değildir.
- Acil satışta nakde komisyonsuz tutar eklenirken deftere komisyonlu tutar
  yazılması: aynı net nakit tutarı kullanılır. Alış komisyonu kısmi satışlara
  lot oranında dağıtılır; net parasal kâr nakit değişimiyle eşleşir.
- Test bağımlılığı eksikliği: pytest/PyYAML ayrı test gereksinimlerine alındı;
  CI tam Git geçmişini alır ve JUnit sonuç belgesi yayınlar.
- Yanlış sabit test bütünlüğü: canlı state'in Ağustos değerine dönmesi
  beklenmez. Test öncesi ve sonrasındaki üretim state/kaynak özetleri karşılaştırılır.
  Frozen araştırma sabitleri değiştirilmedi. Golden test verileri üretimden ayrıdır.
- Ortak concurrency grubundaki paralel pending job'lar: writer job'ları açık
  bağımlılık zincirine alındı. Atlanan moda rağmen zincir ilerler; bir üst iş
  başarısızsa sonraki işlem kararları devam etmez. P1 uzak state değişmişse
  eski hesabı soft-reset ile zorla göndermek yerine hata verir.
- Sabah hazırlanan P1 adayına ertesi gün tarihi verilmesi: sabah adayları o
  gün için geçerlidir. Yeni alım gerçek P1 giriş yolunda seans/tatil kapısından geçer.
- Makro zamanı bozuk veya eski olduğunda yüksek skorun acil satışa sızması:
  geçersiz karar yeni alımı engeller ve tasfiye tetikleyen geçerli skor taşımaz.

## Veri politikası ve sınırlamalar

Yeni alım ve acil satış için zaman damgalı bir dakikalık fiyatın yaş sınırı
20 dakikadır (gecikmeli yedek veri için açık simülasyon varsayımı). Bir dakikalık
bar fiyatı gerçek broker gerçekleşmesi değildir. TradingView float arayüzü ve
fast_info gerçek fiyat zamanını sağlamadığından şu an P1 işlem uygunluğu alamaz;
yalnız sorgu zamanı ile “güncel” ilan edilmez. Bu politika alım sayısını azaltabilir.
Zaman damgalı TradingView adaptörü ayrı ve doğrulanmış veri sözleşmesiyle eklenebilir.

Saatlik mum en fazla 90 dakika önce kapanmış olmalıdır. Günlük göstergeler için
seans kapanışından sonra 10 dakika tamamlanma payı vardır. Bu eşikler ayrı
araştırma varsayımlarıdır, doğrulanmış optimumlar değildir.

Fiyatı doğrulanamayan açık pozisyonda aylık getiri `None` olur. Günlük equity
kaydında `equity=None`, `degerleme_eksik` ve maliyet temelli yardımcı tahmin
ayrıdır. Bunlar Sharpe/drawdown serisine geçerli piyasa değeri diye alınmamalıdır.
Eski portföy özetinin yardımcı tahmini ayrıca yeniden tasarlanmadı; güvenilir
getiri kaynağı aylık blok ve doğrulanmış günlük equity'dir.

BIST takvimi 2026 dini bayram tarihlerini içerir. Gelecek yıl takvimi resmi
kaynaktan güncellenmeden o yılın tatil doğruluğu iddia edilmez.
Resmi kaynak: https://www.borsaistanbul.com/files/pay-piyasasi-2026-yili-tatil-tablosu.pdf

Workflow seviyesindeki `mott-daily` kilidi korunur. Çok sayıda üst üste
workflow tetiklenirse GitHub'ın pending run sınırı hâlâ geçerlidir. Bu değişiklik
tek workflow içindeki writer çakışmasını çözer; zamanında çalışma SLA'sı değildir.

## Kabul testleri

`tests/unit/test_p1_acceptance.py` üretim fiyatına/ağına ihtiyaç duymaz:

- 100.000 TL'den 110.000 TL yaratan eski kayıt senaryosu artık reddedilir.
- Aynı sürümden iki eşzamanlı yerel yazıcının yalnız biri kaydeder.
- Eski yazıcı kapanmış pozisyonu geri getiremez.
- Gerçek normalize→rapor hattında 995 TL / 101 TL örneği PF=9,85 verir.
- Tek lot TP1 tamamlanan pozisyon sayılır; kısmi kâr/son zarar aynı kimlikte toplanır.
- Eski, gelecek tarihli, zamansız ve günlük fiyatla işlem uygunluğu verilmez.
- Bozuk makro zamanı tasfiye skoru taşımaz.
- Alış→TP1→trailing ve acil tasfiye nakit/net kârı bağımsız aritmetik ile karşılaştırılır.
- Aynı mum tekrarı ek nakit/lot hareketi yapmaz.
- Atomik değiştirme öncesi kesinti eski dosyayı korur.
- Yinelenen defter olayları ve negatif lotlar reddedilir.
- Üretim P1 alım yolunda resmi tatil işlem oluşturamaz.
- Açık/gelecek günlük satırlar geçmiş indikatör girdisini değiştiremez.
- Likidite alanı tarayıcıdan son aday puanına kadar taşınır.
- Ortak P2 geçmiş kaydı P1 maliyet/defter semantiğine geçirilmez.

Test sayısı tek başına doğruluk kanıtı değildir. Bu senaryolar da bütün olası
piyasa durumlarını kapsamaz. Gerçek broker/likidite ve dağıtık sistem kesintileri
için ayrı doğrulama gerekir.

Yerel doğrulama: Windows/Python3.11, dış baseline dizini kapalı, NETWORK=OFF.
Tam koşu: **506 geçti, 0 hata, 13 atlandı**. Yeni kabul senaryoları: **26/26**.
Atlananlar eski harici baseline dosyalarına bağlıdır; kritik P1 kabul
senaryolarında atlama yoktur. JUnit belgesi CI'da `pytest-results` artifact'idir.
Bu yerel sayı Ubuntu CI başarısı yerine geçmez; PR kontrolünün ayrı geçmesi gerekir.

## State geçişi

Üretim JSON dosyaları bu geliştirmede değiştirilmedi. İlk yeni ekonomik olay
öncesinde `muhasebe_acilis` oluşturulur: mevcut gerçek nakit, açık lotlar ve
eski defter sonu indeksi. Mutabakat bu sınırdan sonraki olaylar için geçerlidir.
Tarihsel −4.430,78 TL fark bu başlangıçla silinmez veya açıklanmış sayılmaz.
Eski olaylar, bakiye ve başlangıç sermayesi yeniden yazılmaz.

Eski pozisyonun alış komisyonu bilinmiyorsa tahmin edilmez; `maliyet_bilgisi_tam`
false olur. Yeni alışlarda komisyon toplamı ve giriş lotu saklanır.

Üretim geçişinden önce son state yedeklenmeli, açık pozisyon/defter ilk durumu
kontrol edilmeli ve dry-run çıktısı incelenmelidir. Geri alma yalnız kod uyumu
olarak görülmemeli: yeni maliyet ve defter anlamı korunacak şekilde güncel
state üzerinden değerlendirilmelidir. Eski bakiye yedeğini geri yüklemek,
arada gerçekleşen yeni ekonomik olayları silebilir.

PR birleştirilmedi ve üretim botu çalıştırılmadı. Kod/test düzeyindeki başarı,
stratejinin kârlılığına veya gerçek para getirisine ilişkin kanıt değildir.
