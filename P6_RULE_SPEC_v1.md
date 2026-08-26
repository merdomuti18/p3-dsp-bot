# P6_RULE_SPEC_v1 — Etap 1 (ZKN + altyapı)

```
DURUM:       ETAP 1 UYGULAMA (ZKN parity kapısı)
CODE:        scanner_p6.py + tests (P1 dokunulmaz)
PRODUCTION:  scanner_p1.py / portfoy_yonetici.py = 0 değişiklik
```

Bu belge onaylı pakettir: Uygulayıcı 1 başarı kriterleri (A/B/C/D) +
Uygulayıcı 2 teknik yamaları. Freeze kararı uygulayıcıya ait değildir;
uygulama kapısı spec sahibi tarafından açılmıştır.

**Bu PR kapsamı:** spec + `get_indicators_p6` iskeleti + `strategy_zkn(ind)` +
orkestratör evren kapısı + ZKN parity + 5+5 altyapısı (ZKN doldurulur).
WYC / SQZ / CRSI / MD **strateji fonksiyonları yazılmaz**.

---

## 1. Mimari

```
ORCHESTRATOR
    universe = aktif_hisse_listesi()          # P1 ile aynı liste kaynağı
    mc = close × vol_ma20                     # P1 fetch_ohlcv ile aynı
    filtre: mc >= 10_000_000
    filtre: len(df) >= 50                     # P1 cache + fetch_ohlcv
    asof_date = son tam kapanmış BIST günü
    df, asof'dan sonraki barları içermez
        │
        ▼
get_indicators_p6(df)                         # P1 get_indicators'i DEĞİŞTİRMEZ
        │
        ▼
strategy_zkn(ind) -> bool                     # mc yok
        │
        ▼
TRUE ise ayrı satır kaydı (score yok)
```

Director = kaydedici. Seçmez, sıralamaz, Top-N yok.

---

## 2. Üç açık nokta — kilit cevaplar

### 2.1 SQZ `rel_vol >= 1.2` — v1'de YOK

Onaylı kaynak **gösterilemedi**.

- Chartist S4 “hacimli kırılım” der; sayısal `>= 1.2` yazmaz.
- Açık kullanıcı kilidi: `sqz-v1 içinde rel_vol yok` (hacim ayrı deney).
- `1.2` yalnızca donmamış taslakta vardı; sonraki metne uygulayıcı kaymasıyla girdi.

**`sqz-kc20-1.5-v1` = BB⊂KC + `close > bb_up` + `squeeze_on.iloc[-6:-1].any()`.**
`rel_vol` yok. Hacim = `sqz-v2` deneyi. Bu PR’de `strategy_sqz` zaten yazılmaz.

### 2.2 5+5 veri üretimi — iki katman

| Katman | Girdi | Amaç |
|--------|--------|------|
| Boolean unit (5+5) | Sentetik `ind` sözlüğü → `strategy_xxx(ind)` | Kural kimliği |
| Gösterge | Fixture/gerçek OHLCV → `get_indicators_p6` | Formül parity |

Karıştırılmaz. Look-ahead mutasyonu üçüncü, ayrı testtir.

**5+5 tanımı:** her ajan için 5 YES (kural True) + 5 NO (tam **bir** zorunlu şart kırık).
Sınır değerler açık. `mc` ajan NO’su değildir (orkestratör).

### 2.3 Warm-up

`len(df) >= 50` **yalnızca orkestratör**. Ajan kendi NaN alanında `False` döner.

`get_indicators_p6` (Seçenek B): her zaman complete-schema **dict**; `None` yok; `{}` yok.
P1 `get_indicators` `None` dönerse P6 bunu NaN’lı şema dict’ine çevirir. P1 kodu değişmez.

P1 kaynak (iki satır):

- `veri_hazirla`: `if len(df) >= 50` cache
- `fetch_ohlcv`: `if df is None or len(df) < 50: return None, None`

---

## 3. ZKN — `zkn-p1-birebir-v1`

P1 `strategy_zkn` Boolean’ı **kopyalanır**, import edilmez. `mc` ajan içinde yok.

```
close > ema50
and (isna(ema200) or close > ema200)
and 40 <= rsi <= 58
and stochrsi < 40
and cmf > -0.1
and rel_vol >= 0.8
```

Anahtarlar P1 ile aynı: `close, ema50, ema200, rsi, stochrsi, cmf, rel_vol`.

P1 formülleri (parity-kritik):

- EMA: `ewm(span=p, adjust=False)`
- RSI: rolling-mean kazanç/kayıp (Wilder değil) — `scanner_p1._rsi`
- StochRSI: `_stochrsi` (14, K=3)
- CMF: `_cmf` (20)
- `rel_vol`: `vol[-1] / vol[-20:].mean()`
- `ema200`: `len < 200` ise `NaN`, `pd.isna` kolu

`get_indicators_p6` bu yedi alanı P1 `get_indicators` çıktısından alır
(sarmalayıcı; P1 fonksiyonu değiştirilmez). P1 `None` ise alanlar dict içinde NaN kalır.

---

## 4. Orkestratör

```
mc = close.iloc[-1] * volume.iloc[-20:].mean()
eligible = len(df) >= 50 and mc >= 10_000_000
asof_date = df indeksinin son tarihi (çağıran, kapanmamış barı kesmiş olmalı)
```

---

## 5. Kayıt formatı

Zorunlu: `symbol, asof_date, strategy, rule_version, indicators, trigger_conditions`.
`scan_time` ops damgasıdır, kimlik değildir.

Yasak: `score, weight, rank, final_score, strategies: [...]`.
`mc` ajan girdisi ve (Etap 1) kayıt alanı değildir.

Aynı gün/sembol/farklı ajan = ayrı satırlar. Overlap yalnız log (bu PR’de
çoklu ajan olmadığı için overlap motoru yok).

---

## 6. Henüz yazılmayan ajanlar (spek donuk, kod yok)

| Kural | Versiyon | Not |
|-------|----------|-----|
| WYC | wyc-v1 | tek bar spring; `support = low.rolling(10).min().shift(1)`; `range==0 → False` |
| SQZ | sqz-kc20-1.5-v1 | BB⊂KC; `close>bb_up`; `iloc[-6:-1]`; **rel_vol yok** |
| CRSI | crsi-1d-v1 | 4 AND tanh/ölçek; Stoch %K (14,3,3); Fisher/EMA5/ADX/MTF yok |
| MD | md-v1 | `rsi<40` **ve** `rsi>rsi_ema10` **ve** `rsi<rsi_fib50` **ve** `rsi_mom` sıfır kesişimi **ve** `rel_vol>=1.2` |

---

## 7. Etap 1 başarı kriterleri (Uygulayıcı 1 — somut, soyut 5 başlık değil)

### A — Ajan (ZKN bu PR; diğerleri ajan yazılınca)

| ID | Kontrol | Hedef |
|----|---------|-------|
| A1 | İmza | `strategy_xxx(ind) -> bool`; `mc` parametresi yok |
| A2 | Dönüş | yalnızca `True`/`False` |
| A3 | Runtime | 0 exception (geçerli ve NaN `ind`) |
| A4 | Warm-up | zorunlu alan NaN → `False`; `get_indicators_p6` asla `None`/`{}` değil |
| A5 | `rule_version` | kayıtta mevcut |
| A6 | `trigger_conditions` | kural kolları bool |
| A7 | 5+5 | 5 YES True, 5 NO False (direkt `ind`) |
| A8 | Look-ahead | `Data[t+1]` mutasyonu `sinyal[t]` değiştirmez |

### B — Sistem

| ID | Kontrol | Hedef |
|----|---------|-------|
| B1 | Evren | `aktif_hisse_listesi()` |
| B2 | `mc` | yalnız orkestratör, `>= 10_000_000` |
| B3 | Bar kapısı | `len(df) >= 50` orkestratör (P1 iki satır) |
| B4 | `asof_date` | son kapanmış BIST günü; 11:05 kısmi bar yok |
| B5 | `get_indicators_p6` | P1 `get_indicators` değiştirilmez; extras ZKN’yi öldürmez |
| B6 | Satır modeli | bir tetikleme = bir satır |
| B7 | Skor alanları | yok |
| B8 | Overlap | yalnız ölç (ajan yokken N/A) |
| B9 | P1 dokunulmaz | `scanner_p1.py`, `portfoy_yonetici.py` hash aynı |

### C — ZKN parity (ilk kapı)

| ID | Kontrol | Hedef |
|----|---------|-------|
| C1 | 7 gösterge | P1 vs P6 aynı OHLCV’de eşit |
| C2 | True kümesi | P1 `strategy_zkn(ind, mc)` Boolean ∩ evren vs P6 |
| C3 | Yer gerçeği | kapanmış günlük OHLCV; `tarama_listesi` değil |
| C4 | `ema200` | P1 `pd.isna` davranışı |

C fail → WYC/SQZ/CRSI/MD kodlanmaz.

### D — Dokümantasyon

| ID | Kontrol | Hedef |
|----|---------|-------|
| D1 | Bu spec dosyası | repoda |
| D2 | Versiyon etiketleri | ZKN kilitli; diğerleri spekte |
| D3 | Yasak / var listesi | bölüm 8 |
| D4 | 5+5 + dual test yolu | bölüm 2.2 |

KPI yok: forward return, alpha, kota, 10–40 sinyal, ajan payı.

---

## 8. Yasak / var

**YOK:** score, STRATEGY_WEIGHTS, combo, Top-N, kota, 10–40 hedef, forward-return KPI,
alpha KPI, Risk, Execution, LGBM, MA/VIOP, ADX ekleme, EMA5 giriş, MTF, yeni ajan,
P1 scanner/portföy değişikliği, `tarama_listesi` parity kaynağı, SQZ-v1 `rel_vol`,
`get_indicators_p6` `None` veya `{}` (P1 `None` → NaN şema dict).

**VAR:** `strategy_xxx(ind)->bool`, `mc` orkestratör, ayrı satır, `rule_version`,
`asof_date`, `trigger_conditions`, 5+5 (direkt ind), gösterge testi (OHLCV).

---

## 9. Uygulama sırası

1. Bu spec  
2. `get_indicators_p6` + ZKN Boolean + orkestratör  
3. **ZKN PARITY** — FAIL ise DUR  
4. Sonraki PR’ler: WYC → SQZ → CRSI → MD (her biri 5+5 + look-ahead)  
5. Director log tam seti, overlap, A/B/C/D kapısı  

---

## 10. ZKN 5+5 (direkt `ind`)

Taban YES: `close=100, ema50=99, ema200=90, rsi=50, stochrsi=30, cmf=0, rel_vol=1`.

| ID | Değişiklik | Beklenen |
|----|------------|----------|
| YES-1 | taban | True |
| YES-2 | rsi=40 | True (dahil) |
| YES-3 | rsi=58 | True (dahil) |
| YES-4 | rel_vol=0.8 | True (dahil) |
| YES-5 | ema200=NaN | True (P1 isna) |
| NO-1 | rsi=39 | False |
| NO-2 | rsi=59 | False |
| NO-3 | stochrsi=40 | False (`<` sıkı) |
| NO-4 | cmf=-0.1 | False (`>` sıkı) |
| NO-5 | rel_vol=0.799 | False |
