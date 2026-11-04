# P1 strateji envanteri ve mantık incelemesi

İnceleme referansı: PR #7, `f373dedc9c48184967859cd231044bbaad6429a4`.
Kullanıcının “ajan” ifadesi, ZKN ve KBM gibi tarama kurallarını anlatmaktadır.
MA/LightGBM/piyasa yönü/AlphaTrend ek değerlendirmeleri bu strateji listesiyle
karıştırılmamalıdır.

## Mevcut stratejiler

`scanner_p1.run_scan` sekiz kuralı çağırır. Aktif dört kural seçen bir ayar
bulunmuyor. Ağırlıklar `portfoy_yonetici.STRATEGY_WEIGHTS` içindedir.

| Kod | İsim | Temel koşullar | Ağırlık | Alım etkisi |
|---|---|---|---:|---|
| ZKN | İyi alım yeri | Fiyat EMA50 üstünde; RSI 40–58; StochRSI <40; CMF >−0,1; göreli hacim ≥0,8 | 14 | Aday ve puan |
| KBM | Karışık BB MACD | Fiyat Bollinger orta çizgisinin üstünde; MACD sinyali yukarı keser; RSI >48; CMF >0; göreli hacim ≥0,9 | 8 | Aday ve puan |
| GT | Güçlü trend | EMA8>EMA21>EMA50; RSI 50–65; ADX>20; DI+>DI−; CMF>0; hacim ≥1 | 18 | Aday ve puan |
| GTD | Güçlü trend devam | Aynı EMA sırası; RSI 65–78; ADX>25; CMF>0,05; hacim ≥1,2 | 14 | Aday ve puan |
| ALPHA | ATR kanalına göre geçiş | Fiyat tarayıcı kanalını yukarı geçer; RSI>45; ADX>18; CMF>−0,05; hacim ≥1 | 16 | Aday ve puan |
| ZT3 | ZTAN3 | ADX>25; DI+>DI−; RSI>50; MACD yukarı kesişimi; fiyat tarayıcı kanalının üstünde; hacim ≥1 | 18 | Aday ve puan |
| MR | Mini ralli | RSI>55; MACD sinyal üstünde ve pozitif; fiyat SMA20 üstünde; hacim ≥1,1; günlük değişim pozitif | 10 | Aday ve puan |
| DIP | Dip taraması | RSI<30; StochRSI<20; fiyat alt Bollinger altında; CMF<−0,1; değişim >−1%; hacim ≥1 | 12 | Tek başına alım adaylarından çıkarılır |

Her kural en az 10 milyon TL ortalama günlük işlem tutarı ister. Bu değer
piyasa kapitalizasyonu değildir: son fiyat × son 20 gün ortalama hacimdir.
GT, GTD ve ZKN ayrıca EMA200 kontrolü yapar; 200 bar yoksa mevcut politika
bu filtreyi atlar. İlk PR bunu görünür hale getirmiştir, bu takip değişikliği
filtre politikasını değiştirmemektedir.

İncelenen kalıcı tarama günlüğündeki 149 hisse-tarama kaydında MR 93, ZKN 24,
ALPHA 18, GTD 17, GT 11, DIP 9, KBM 6 ve ZT3 2 kez görünür. Son kayıt
25.09.2026 20:49'dur. Aynı kayıtta birden fazla strateji olabilir; sayılar
bağımsız işlem sayısı veya başarı oranı değildir.

“MD” adlı ayrı bir kural bu sürümde bulunmadı. Erişilebilen dal geçmişinde
P1 dosyaları için bağımsız `MD` kelimesinin eklendiği/silindiği commit de
bulunmadı. Hangi eski sürümün dört ajan içerdiği bu kanıtla kesinleştirilemez.
Bu nedenle dört ismi tahmin edip diğer kuralları kaldırmadık.

## Mantık değerlendirmesi

ZKN trend içindeki geri çekilmeyi, KBM pozitif MACD geçişini, GT/GTD trendin
iki güç seviyesini, ZT3 teyitli MACD geçişini ve MR kısa süreli devamı arar.
Koşulların yönleri kendi amaçlarıyla tutarlıdır. Bu, gelecekte kâr üretecekleri
veya parametrelerin en iyi değerler olduğu anlamına gelmez.

Önemli bulgular:

1. **Likidite alanı aktarım hatası düzeltildi.** Tarayıcı işlem tutarını
   hesaplıyor, fakat sinyal kaydına koymuyordu. Son puanlama eksik alanı sıfır
   sayıp likit hisselere de −4 puan uyguluyordu. İşlem tutarı artık kayıt ve
   aday hattında korunuyor; eşik/ağırlık değiştirilmedi.
2. **DIP alım için fiilen kapalıdır.** Diğer yedi kural en az RSI40 veya daha
   yüksek RSI isterken DIP RSI<30 ister. Dolayısıyla mevcut koşullarla DIP
   diğer kuralla birleşemez ve tek başına çıkarıldığı için alım üretemez.
   “DIP kaldırılsın” ile “DIP yalnız bilgi olarak taransın” farklı kararlardır;
   mevcut bilgi amaçlı davranış korundu.
3. **GT ve GTD RSI=65 sınırında birlikte tetiklenebilir.** Diğer şartlar da
   sağlanırsa aynı trend iki oy ve iki ağırlık alır. Bu açık bir tasarım
   tercihi olabilir; araştırma yapmadan eşikleri değiştirmedik.
4. **KBM, ZT3, MR ve GT ailesi bağımsız oylar değildir.** EMA/MACD/hacim
   bilgisini tekrar kullanırlar. Çoklu teyit bonusunun bağımsız dört ajan
   onayı gibi yorumlanması doğru değildir. Ağırlıklar kalibrasyon kanıtı
   taşımayan sabitlerdir; ayrı dönemlerde ablation karşılaştırması gerekir.
5. **ALPHA ile ek AlphaTrend aynı formül değildir.** Tarayıcı orta fiyat,
   ATR×1,5 ve fiyat/kanal ilişkisini kullanır. Portföy ek bileşeni MFI14,
   ATR×1,0 ve kanalın iki barlık eğimini kullanır. Biri diğerinin birebir
   teyidi olarak tanıtılmamalıdır; formüller bu değişiklikte birleştirilmedi.
6. **Günlük verinin bilinirlik zamanı düzeltildi.** Tarayıcı ve ek puanlama
   bileşenleri açık günlük mumu veya gelecek tarihli günlük satırı kullanmaz.
   Seans kapanışından sonra 10 dakika veri tamamlama payı uygulanır.
7. **Skor bir olasılık değildir.** Strateji ağırlığı, teyit sayısı, MA bonusu,
   model katkısı, piyasa yönü ve AlphaTrend bonusu toplanır. Örneğin 80 puan
   “%80 kazanma ihtimali” anlamına gelmez.
8. **LightGBM mevcut depoda pasiftir.** Model dosyası bulunmuyor. Model sonradan
   eklendiğinde artık pozitif sınıf, eksik özellik ve sonlu [0,1] olasılık
   kontrolleri yapılır; bilinmeyen özellik sessizce sıfırlanmaz.

## Ek değerlendirmeler

- MA bileşeni: Motor A EMA12/26 yukarı kesişimi; Motor B EMA8/21'e aşağıdan
  yaklaşma, kapanan mesafe. Açı, hacim ve EMA50 teyidiyle bonus verir.
- Piyasa yönü: adı VİOP olsa da çoğu zaman günlük BIST100/BIST30 veya hisse
  sepeti değişimini kullanır. Güncel vadeli kontrat tahmini değildir.
- AlphaTrend: yukarıda açıklanan ayrı MFI tabanlı bonus.
- LightGBM: model varsa ayrı filtre ve bonus; bugün pasif.

Bu ek bileşenler de birbirinden bağımsız ajanlar olarak doğrulanmış değildir.

## Sonraki araştırma

Strateji eşiklerini değiştirmenin öncesinde ortak dönem/veri/masraf varsayımıyla
her kural tek başına, mevcut birleşim ve her kural çıkarılmış birleşim
karşılaştırılmalıdır. Sinyal üretim zamanı ile alış gerçekleşme zamanı ayrı
olmalı; örtüşen işlemler ve gelecekte bilinen veriler ayıklanmalıdır.

Yeni çıkış kayıtları `source_signal` taşır. Eski kapanış kayıtlarında sinyal
kökeni tam olmadığı için geçmiş toplam getiriyi sekiz ajana güvenilir biçimde
bölmek mümkün değildir. İleriye dönük katkı ölçümü bu yeni kayıtlardan yapılabilir.

## Kaynaklar

- `scanner_p1.py`: get_indicators, sekiz strategy fonksiyonu, run_scan.
- `portfoy_yonetici.py`: STRATEGY_WEIGHTS, tarama_calistir, final_signal_score.
- `scan_history_p1.jsonl`: salt okunur tarama günlüğü.
- BIST 2026 tatil kaynağı: https://www.borsaistanbul.com/files/pay-piyasasi-2026-yili-tatil-tablosu.pdf
