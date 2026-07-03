# MOTT — BIST Çok Stratejili Takip Sistemi

BIST hisseleri için 5 bağımsız stratejiyi GitHub Actions üzerinde çalıştıran,
işlem olduğunda Telegram'a bildirim gönderen kağıt-portföy (paper trading)
sistemi. Her portföy 100.000 TL sanal sermaye ile başlar.

## Stratejiler

| Kod | Ad | Ana dosyalar | State dosyası |
|-----|----|--------------|---------------|
| P1 | Momentum | `scanner_p1.py`, `portfoy_yonetici.py` | `portfoy.json`, `state_p1.json` |
| P2 | SMC (Smart Money Concepts) | `scanner_smc.py`, `portfoy_yonetici.py` | `portfoy_p2.json`, `state_p2.json` |
| P3 | DSP (sinyal işleme) | `simulate_dsp.py` (+`dsp_core`, `dsp_trend`, `dsp_strategies`) | `portfolio_state.json` |
| P4 | Meta Optimizer (IC ağırlıklı, P1+P2+P3 sinyallerinden) | `meta_portfolio.py` | `state_p4.json` |
| P5 | Komite (muhafazakâr konsensüs, min. 2 strateji onayı) | `p5_committee.py` | `state_p5.json` |

Ortak modüller:

- `mott_mode.py` — TSİ'ye göre çalışma modunu belirler (aşağıya bakın).
- `mott_telegram.py` — tüm Telegram gönderimi burada: retry (3 deneme),
  429 rate-limit bekleme, Markdown hatasında düz metne düşme, 4096 karakter
  üstünde otomatik parçalama. Mesaj yalnızca gerçek işlem (alım/satım/STOP/TP)
  olduğunda gönderilir.
- `mott_fiyat.py` — canlı fiyat: önce TradingView screener (gerçek zamanlı,
  TV grafikleriyle uyumlu), erişilemezse yfinance yedeği.
- `mott_state.py` — 5 farklı state şemasını tek ortak şemaya çeviren
  normalizasyon katmanı (raporlama araçları için).
- `mott_aylik_rapor.py` — ayın 1'inde 5 portföyün karşılaştırma raporu.
- `mott_performans_analiz.py` — geçmiş performans analizi (manuel araç).

## Çalışma modları (TSİ)

`mott_mode.py` saati moda çevirir; workflow'daki her job yalnızca ilgili
modda çalışır:

| Mod | Pencere | Ne yapılır |
|-----|---------|-----------|
| `sabah` | 08:45–09:45 | P1 sabah değerlendirmesi |
| `alim` | 10:00–11:20 | P1 alım denemesi + çıkış kontrolü; P2 teyit; P3/P4/P5 canlı takip |
| `takip` | 11:20–16:59 | P1/P2 pozisyon takibi; P3/P4/P5 canlı takip |
| `kapani` | 17:00–18:15 | P1 kapanış kontrolü; P3/P4/P5 canlı takip |
| `aksam` | 18:30–21:30 | Tam tarama: P1, P2, P3 → ardından P4 ve P5 |

## Zamanlama

- 09:00 TSİ — sabah değerlendirme
- 10:00–18:00 TSİ, saat başı — gün içi çalıştırma (mod otomatik belirlenir);
  açık pozisyonlar için STOP / TP / MAX_GUN kontrolü ve anında Telegram
- ~19:05 TSİ — akşam tam tarama

Not: GitHub zamanlanmış tetikleri "en iyi çaba" ile çalıştırır; birkaç dakika
gecikme normaldir.

## Risk kuralları

- STOP: −%5 (P3'te gün içi acil stop −%8), TP: +%10, MAX_GUN: 10 takvim günü.
- Elde tutma süresi giriş **tarihinden** hesaplanır — cron atlansa bile şaşmaz.
- Bölünme/bedelsiz koruması: BIST günlük taban limiti %10 olduğundan, son
  kayıtlı fiyata göre %10'u aşan düşüş piyasa hareketi sayılmaz; bedelsiz/split
  varsayılıp giriş fiyatı oranla düzeltilir ve Telegram'a uyarı gönderilir
  (KAP'tan elle doğrulanmalı).

## Kurulum / sırlar

Bağımlılıklar `requirements.txt` içinde **sabit sürümlü**dür. Güncellerken
önce tek stratejide deneyin.

GitHub Secrets: `TELEGRAM_BOT_TOKEN` (veya `TELEGRAM_TOKEN`) ve
`TELEGRAM_CHAT_ID`. Her workflow job'ı başarısız olursa Telegram'a hata
bildirimi de gönderilir.

## Yapılacaklar

- [ ] **15 dakikalık canlı takip** — GitHub cron sık tetikleri atladığı için
  saatliğe çekildi. Gerçek 15 dk sıklık için harici bir zamanlayıcı
  (ör. cron-job.org) fine-grained PAT ile
  `POST /repos/{owner}/{repo}/actions/workflows/mott_daily.yml/dispatches`
  çağırmalı. PAT'e yalnızca bu depo için `actions: write` izni verin.
- [ ] GitHub Pages kullanılmıyorsa kapatın (Settings → Pages → Disable):
  her push'ta gereksiz "pages build and deployment" çalışıyor ve ara ara
  hata e-postası üretiyor.
- [ ] `portfoy_yonetici.py` (~78 KB) zamanla P1/P2 ortak çekirdek + strateji
  modüllerine bölünebilir.

## Notlar

- Depo public'tir — Actions dakikaları ücretsiz ve sınırsızdır. Private'a
  çevrilirse aylık tüketim 2.000 dakikalık ücretsiz limiti aşar.
- `reports/history/` son 30 raporla sınırlı tutulur (P3 akşam job'ı temizler).
- Bu sistem kağıt portföydür; yatırım tavsiyesi değildir.
