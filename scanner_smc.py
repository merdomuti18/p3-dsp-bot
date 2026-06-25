# -*- coding: utf-8 -*-
"""
Cloud Function BIST SMC Scanner — Portföy 2
============================================
3 katmanlı çalışma:

  20:50 → run_scan_smc()        : 1h→1D resample + 1W HTF, ana sinyal
  11:00 → saatlik_teyit_smc()  : 1h direkt, teyit bonusu + alım
  12-17 → saatlik_teyit_smc()  : saatlik tekrar, stop/TP VM tarafında

Skor sistemi:
  Akşam sinyali  : CHoCH/BOS/OB/Sweep(high=±3.0) FVG(medium=±1.5)
                   PD zone, RVOL, 1W HTF bonusları
  Gün içi teyit  : +2.0 (1h CHoCH/BOS), +1.0 (trend), +1.0 (RVOL)

Portföy 2 risk parametreleri (P1 ile aynı):
  Sermaye: 100k TL | Max: 7 hisse | Limit: 20k TL
  Stop: -5% | TP1: +8% | Trailing: -5% | MaxGün: 10
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")

# GitHub Actions'ta state dosyası repo kök dizininde
BASE_DIR   = Path(os.environ.get("MOTT_BASE_DIR", "."))
TMP_BASE   = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
SIGNAL_LOG = TMP_BASE / "scan_smc_history.jsonl"

# SMC parametreleri
SWING_LENGTH    = 3
ISLEM_TL_MIN    = 5_000_000
SCORE_GUCLU_AL  = 6.0
SCORE_AL        = 3.0

# Teyit bonusları
TEYIT_CHOCH_BOS = 2.0
TEYIT_TREND     = 1.0
TEYIT_RVOL      = 1.0

try:
    yf.set_tz_cache_location("/tmp/yf_tz")
except Exception:
    pass

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome110")
    def _ticker(symbol: str):
        return yf.Ticker(f"{symbol}.IS", session=_SESSION)
except Exception:
    _SESSION = None
    def _ticker(symbol: str):
        return yf.Ticker(f"{symbol}.IS")

try:
    from smartmoneyconcepts import smc as _smc
    SMC_OK = True
except ImportError:
    SMC_OK = False
    log.error("smartmoneyconcepts kurulu degil!")

_VARSAYILAN_SEMBOLLER = [
    "AKBNK","AKSA","AKSEN","AKGRT","ALKIM","ANSGR","ARCLK","ARDYZ","ARSAN","ASELS",
    "ASTOR","ASUZU","AYGAZ","BAGFS","BERA","BFREN","BIMAS","BORLS","BRISA","BRSAN",
    "BTCIM","BUCIM","BURCE","BVSAN","CASA","CATES","CCOLA","CELHA","CEMTS","CIMSA",
    "CLEBI","CWENE","DAGI","DENGE","DEVA","DITAS","DOAS","DOFER","DOKTA","ECILC",
    "ECZYT","EFOR","EGEEN","EGSER","EKGYO","ELITE","ENKAI","ENSRI","ENTRA","EREGL",
    "ERSU","FADE","FENER","FONET","FROTO","GARAN","GARFA","GEDZA","GESAN","GLCVY",
    "GLYHO","GOODY","GRSEL","GSDHO","GUBRF","HATEK","HALKB","HATSN","HEKTS","HOROZ",
    "ICBCT","IEYHO","IHLGM","IMASM","INDES","INFO","INGRM","INTEM","INVEO","ISBIR",
    "ISCTR","ISDMR","ISGSY","ISKPL","KAREL","KARSN","KARTN","KCHOL","KGYO","KLGYO",
    "KLMSN","KMPUR","KNFRT","KONYA","KORDS","KRDMA","KRDMB","KRDMD","KRGYO","KSTUR",
    "KTSKR","KUYAS","LIDFA","LINK","LOGO","MAKIM","MANAS","MARTI","MAVI","MEGMT",
    "MGROS","MNDRS","MOBTL","NATEN","NETAS","NTGAZ","NUHCM","OBASE","ODAS","ORGE",
    "PAGYO","PENGD","PETKM","PGSUS","PKENT","PNLSN","PRKAB","RALYH","RAYSG","REEDR",
    "RUZYE","SAFKR","SAHOL","SANKO","SARKY","SASA","SILVR","SKTAS","SNICA","TABGD",
    "TARKM","THYAO","TKFEN","TKNSA","TOASO","TRGYO","TRMET","TSGYO","TTKOM","TTRAK",
    "TUPRS","TUREX","TURGG","TURSG","UFUK","ULUFA","ULUSE","UNLU","VAKBN","VAKKO",
    "VKING","YATAS","YIGIT","ZOREN","ULKER","BRKSN","GWIND","ALARK","PAHOL","BALSU",
    "SISE","KCAER","MIATK","MOGAN","GLRMK","BORSK","PATEK","PSGYO","UCAYM","OBAMS",
    "KLYPV",
]

def _hisse_listesi_yukle() -> list[str]:
    txt = BASE_DIR / "hisseler.txt"
    if txt.exists():
        try:
            return list(dict.fromkeys(
                l.strip().upper() for l in txt.read_text("utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#")
            ))
        except Exception as e:
            log.warning("hisseler.txt: %s", e)
    return list(_VARSAYILAN_SEMBOLLER)

# ── Veri önbelleği ─────────────────────────────────────────────────────────────
_CACHE_1H: dict[str, pd.DataFrame] = {}   # ham 1h veri (her şeyin tabanı)

def veri_hazirla(semboller: list[str], period: str = "60d") -> int:
    """
    1h toplu indirme — tüm resample'ların tabanı.
    period="60d"  → akşam tarama (1D resample için yeterli)
    period="5d"   → gün içi teyit (1h direkt, hızlı)
    """
    global _CACHE_1H
    _CACHE_1H = {}
    tickers = [f"{s}.IS" for s in semboller]
    log.info("SMC 1h veri çekiliyor (%s): %s hisse", period, len(tickers))
    try:
        kwargs = dict(
            tickers=tickers, period=period, interval="1h",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
        if _SESSION:
            kwargs["session"] = _SESSION
        raw = yf.download(**kwargs)
    except Exception as e:
        log.error("1h toplu indirme: %s", e)
        return 0

    if raw is None or raw.empty:
        return 0

    ok = 0
    for sym in semboller:
        ticker = f"{sym}.IS"
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker][["Open","High","Low","Close","Volume"]].copy()
            else:
                df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = [c.lower() for c in df.columns]
            df["volume"] = df["volume"].ffill()
            df = df.dropna(subset=["open","high","low","close"])
            if len(df) >= 8:   # en az 1 günlük veri
                # UTC+3'e çevir, timezone-naive yap (1W resample gün sınırı için)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("Europe/Istanbul").tz_localize(None)
                _CACHE_1H[sym] = df
                ok += 1
        except Exception:
            continue
    log.info("1h indirme tamamlandi: %s/%s", ok, len(semboller))
    return ok

# ── Resample fonksiyonları ─────────────────────────────────────────────────────

def _resample_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    1h → 1D günlük mumlar.
    Orijinal SMCScanner mantığı: resample('1D').
    17:45 veya 20:50'de çalıştırıldığında o günün tüm 1h barları
    dahil olur → tam günlük mum oluşur.
    """
    df = df_1h.resample("1D").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return df

def _resample_1w(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    1h → 1W haftalık mumlar.
    HTF EMA200 filtresi için — her zaman kapanmış ✓
    """
    df = df_1h.resample("W-FRI").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return df

# ── İndikatörler ───────────────────────────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

# ── HTF trend (1W EMA200) ──────────────────────────────────────────────────────

def get_tv_ema200_bulk(symbols: list) -> dict:
    """TradingView screener'dan EMA200 1W degerlerini toplu ceker."""
    try:
        from tradingview_screener import Query, col
        count, df = (Query()
            .set_markets('turkey')
            .select('name', 'EMA200|1W', 'close')
            .where(col('name').isin(symbols))
            .get_scanner_data())
        result = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('EMA200|1W')) and pd.notna(row.get('close')):
                result[row['name']] = {
                    'ema200': float(row['EMA200|1W']),
                    'close':  float(row['close']),
                }
        log.info('TV EMA200 bulk: %s/%s sembol alindi', len(result), len(symbols))
        return result
    except Exception as exc:
        log.warning('TV EMA200 bulk hatasi, fallback aktif: %s', exc)
        return {}

def get_htf_trend(df_1h: pd.DataFrame, sym: str = '', tv_cache: dict = None) -> str:
    """HTF EMA200 trend filtresi. TV cache varsa kullan, yoksa yfinance resample."""
    if tv_cache and sym in tv_cache:
        d = tv_cache[sym]
        return 'Bullish' if d['close'] > d['ema200'] else 'Bearish'
    df_1w = _resample_1w(df_1h)
    if len(df_1w) < 10:
        return 'Neutral'
    ema200 = float(_ema(df_1w['close'], 200).iloc[-1])
    last   = float(df_1w['close'].iloc[-1])
    return 'Bullish' if last > ema200 else 'Bearish'
# ── SMC sinyal hesabı ──────────────────────────────────────────────────────────

def _smc_hesapla(df: pd.DataFrame, mod: str = "daily") -> dict | None:
    """
    df üzerinde SMC sinyallerini hesaplar.
    mod="daily" → 1D resample üzerinde (akşam tarama)
    mod="hourly" → 1h direkt (gün içi teyit)
    """
    if not SMC_OK or len(df) < 10:
        return None

    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    n = len(df)

    # Pencere: daily'de son 5 bar, hourly'de son 3 bar
    pencere_hl  = 5 if mod == "daily" else 3
    pencere_fvg = 3 if mod == "daily" else 2

    last_close = float(c.iloc[-1])
    last_vol   = float(v.iloc[-1])
    prev_close = float(c.iloc[-2]) if n > 1 else last_close

    rsi_val   = float(_rsi(c).iloc[-1])
    ema200_val = float(_ema(c, 200).iloc[-1]) if n >= 200 else float(_ema(c, n).iloc[-1])
    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    macd_l = ema12 - ema26; macd_sig = _ema(macd_l, 9)

    trend   = "Bullish" if last_close > ema200_val else "Bearish"
    high_20 = float(h.tail(20).max())
    low_20  = float(l.tail(20).min())
    mid_pt  = (high_20 + low_20) / 2
    pd_zone = "Premium" if last_close > mid_pt else "Discount"

    avg_vol  = float(v.tail(20).mean())
    rvol     = round(last_vol / (avg_vol + 1e-9), 2)
    vol_conf = rvol > 1.5
    islem_tl = last_close * last_vol
    chg_pct  = round((last_close / prev_close - 1) * 100, 2) if prev_close else 0

    try:
        last_time_str = df.index[-1].strftime("%d/%m %H:%M")
    except Exception:
        last_time_str = ""

    signals = []

    # MACD Cross
    if n >= 3:
        if float(macd_l.iloc[-1]) > float(macd_sig.iloc[-1]) and \
           float(macd_l.iloc[-2]) <= float(macd_sig.iloc[-2]):
            signals.append({"type":"Bullish MACD Cross","level":round(float(macd_l.iloc[-1]),3),
                            "time":last_time_str,"importance":"medium"})
        elif float(macd_l.iloc[-1]) < float(macd_sig.iloc[-1]) and \
             float(macd_l.iloc[-2]) >= float(macd_sig.iloc[-2]):
            signals.append({"type":"Bearish MACD Cross","level":round(float(macd_l.iloc[-1]),3),
                            "time":last_time_str,"importance":"medium"})

    try:
        swing_hl = _smc.swing_highs_lows(df, swing_length=SWING_LENGTH)

        # CHoCH
        bc = _smc.bos_choch(df, swing_hl)
        choch_rows = bc.dropna(subset=["CHOCH"])
        if not choch_rows.empty and int(choch_rows.iloc[-1].name) > n - pencere_hl - 1:
            r = choch_rows.iloc[-1]
            signals.append({"type":"Bullish CHoCH" if r["CHOCH"]==1 else "Bearish CHoCH",
                            "level":round(float(r["Level"]),2),"time":last_time_str,"importance":"high"})

        # BOS
        bos_rows = bc.dropna(subset=["BOS"])
        if not bos_rows.empty and int(bos_rows.iloc[-1].name) > n - pencere_hl - 1:
            r = bos_rows.iloc[-1]
            signals.append({"type":"Bullish BOS" if r["BOS"]==1 else "Bearish BOS",
                            "level":round(float(r["Level"]),2),"time":last_time_str,"importance":"high"})

        # Liquidity Sweep
        liq = _smc.liquidity(df, swing_hl)
        liq_rows = liq.dropna(subset=["Liquidity"])
        if not liq_rows.empty and int(liq_rows.iloc[-1].name) > n - pencere_fvg - 1:
            r = liq_rows.iloc[-1]
            signals.append({"type":"Bullish Sweep" if r["Liquidity"]==1 else "Bearish Sweep",
                            "level":round(float(r["Level"]),2),"time":last_time_str,"importance":"high"})

        # Order Block
        ob = _smc.ob(df, swing_hl)
        ob_rows = ob.dropna(subset=["OB"])
        if not ob_rows.empty and int(ob_rows.iloc[-1].name) > n - pencere_hl - 1:
            r = ob_rows.iloc[-1]
            signals.append({"type":"Bullish OB" if r["OB"]==1 else "Bearish OB",
                            "level":round(float(r["Top"]),2),"time":last_time_str,"importance":"high"})

        # FVG
        fvg = _smc.fvg(df)
        fvg_rows = fvg.dropna()
        if not fvg_rows.empty and int(fvg_rows.iloc[-1].name) > n - pencere_fvg - 1:
            r = fvg_rows.iloc[-1]
            signals.append({"type":"Bullish FVG" if r["FVG"]==1 else "Bearish FVG",
                            "level":round(float(r["Top"]),2),"time":last_time_str,"importance":"medium"})

    except Exception as e:
        log.debug("SMC hesap %s: %s", mod, e)

    return {
        "signals": signals, "pd_zone": pd_zone, "rvol": rvol,
        "volume_confirmation": vol_conf,
        "indicators": {"rsi": round(rsi_val,2), "trend": trend},
        "last_price": round(last_close,2), "last_time": last_time_str,
        "islem_tl": round(islem_tl,0), "change_pct": chg_pct,
    }

# ── Verdict (akşam skoru) ──────────────────────────────────────────────────────

def calculate_verdict(result: dict, htf_trend: str) -> dict:
    score = 0.0
    reasons = []

    for sig in result["signals"]:
        val = 3.0 if sig["importance"] == "high" else 1.5
        score += val if "Bullish" in sig["type"] else -val

    has_bull = any("Bullish" in s["type"] for s in result["signals"])
    has_bear = any("Bearish" in s["type"] for s in result["signals"])

    if has_bull:
        if result["pd_zone"] == "Discount":
            score += 2.0; reasons.append("Discount bölge")
        else:
            score -= 1.0; reasons.append("Premium risk")
    if has_bear:
        score += -2.0 if result["pd_zone"] == "Premium" else 1.0

    if result["volume_confirmation"]:
        score += 2.0 if has_bull else -2.0
        reasons.append(f"RVOL {result['rvol']:.1f}x")

    # 1W HTF
    htf_aligned = htf_trend == result["indicators"]["trend"] and htf_trend != "Neutral"
    if htf_aligned and has_bull:
        score += 1.0; reasons.append("1W HTF uyumu")

    score = round(score, 1)
    verdict = ("GÜÇLÜ AL" if score >= SCORE_GUCLU_AL else
               "AL"       if score >= SCORE_AL       else
               "GÜÇLÜ SAT" if score <= -SCORE_GUCLU_AL else
               "SAT"       if score <= -SCORE_AL     else "İZLE")

    return {"score": score, "verdict": verdict, "reasons": reasons[:3],
            "htf_aligned": htf_aligned, "htf_trend": htf_trend,
            "teyit_skoru": 0.0, "teyit_var": False}

# ── Gün içi 1h teyit ──────────────────────────────────────────────────────────

def teyit_skoru_hesapla(symbol: str) -> dict:
    """
    1h direkt veri ile teyit bonusu hesapla.
    Akşam listesindeki adaya +bonus ekler.
    """
    df_1h = _CACHE_1H.get(symbol)
    if df_1h is None or len(df_1h) < 5:
        return {"teyit_skoru": 0.0, "teyit_var": False, "teyit_sinyaller": []}

    result_1h = _smc_hesapla(df_1h, mod="hourly")
    if result_1h is None:
        return {"teyit_skoru": 0.0, "teyit_var": False, "teyit_sinyaller": []}

    bonus = 0.0
    has_bull_1h = any("Bullish" in s["type"] for s in result_1h["signals"]
                      if s["importance"] == "high")

    if has_bull_1h:
        bonus += TEYIT_CHOCH_BOS

    if result_1h["indicators"]["trend"] == "Bullish":
        bonus += TEYIT_TREND

    if result_1h["volume_confirmation"]:
        bonus += TEYIT_RVOL

    return {
        "teyit_skoru":    round(bonus, 1),
        "teyit_var":      bonus >= TEYIT_CHOCH_BOS,
        "teyit_sinyaller": [s["type"] for s in result_1h["signals"] if "Bullish" in s["type"]],
    }

# ── Ana tarama (20:50) ─────────────────────────────────────────────────────────

def run_scan_smc(manual: bool = False) -> tuple[list, list]:
    """
    Akşam 20:50 taraması.
    manual=True → şu an tetiklenmiş, etiketi farklı.
    """
    if not SMC_OK:
        send_telegram("⚠️ <b>P2-SMC</b>\nsmartmoneyconcepts kurulu değil.")
        return [], []

    tz_tr     = timezone(timedelta(hours=3))
    now       = datetime.now(tz_tr)
    scan_time = now.strftime("%d.%m.%Y %H:%M")
    scan_label = "Manuel Tetikleme → 20:50 Modu" if manual else "20:50 Akşam Tarama"

    semboller = _hisse_listesi_yukle()
    ok = veri_hazirla(semboller, period="60d")
    if ok == 0:
        send_telegram(f"⚠️ <b>P2-SMC</b>\nVeri çekilemedi.")
        return [], []

    guclu_al, al, hatali = [], [], 0
    tv_ema200 = get_tv_ema200_bulk(semboller)

    for i, sym in enumerate(semboller):
        try:
            df_1h = _CACHE_1H.get(sym)
            if df_1h is None:
                continue

            # 1D resample — orijinal SMCScanner mantığı
            df_1d = _resample_1d(df_1h)
            if len(df_1d) < 10:
                continue

            result = _smc_hesapla(df_1d, mod="daily")
            if result is None or not result["signals"]:
                continue
            if result["islem_tl"] < ISLEM_TL_MIN:
                continue
            if not any("Bullish" in s["type"] for s in result["signals"]):
                continue

            htf = get_htf_trend(df_1h, sym, tv_ema200)
            ai  = calculate_verdict(result, htf)

            if ai["verdict"] not in ("AL", "GÜÇLÜ AL"):
                continue

            result["symbol"] = sym
            result["ai_analysis"] = ai

            if ai["verdict"] == "GÜÇLÜ AL":
                guclu_al.append(result)
            else:
                al.append(result)

        except Exception as e:
            hatali += 1
            log.debug("scan %s: %s", sym, e)

        if (i + 1) % 20 == 0:
            time.sleep(0.3)

    guclu_al.sort(key=lambda x: -x["ai_analysis"]["score"])
    al.sort(key=lambda x: -x["ai_analysis"]["score"])

    log.info("SMC akşam tarama: GÜÇLÜ AL=%s AL=%s hata=%s", len(guclu_al), len(al), hatali)

    all_res = guclu_al + al
    records = _build_records(all_res, scan_time, scan_label)
    _append_signal_log(records)
    vm_gonder_p2(records, scan_time, scan_label, mod="aksam")

    msg = format_aksam(guclu_al, al, scan_time, scan_label, ok)
    send_telegram(msg)
    return guclu_al, al

# ── Gün içi teyit tarama (11:00-17:00) ────────────────────────────────────────

def saatlik_teyit_smc() -> dict:
    """
    11:00-17:00 arası saatlik çalışır.
    VM'deki bekleyen_al listesindeki P2 adaylarına teyit bonusu ekler.
    Döner: teyit güncellemesi + VM'ye gönderir.
    """
    if not SMC_OK:
        return {}

    tz_tr     = timezone(timedelta(hours=3))
    now       = datetime.now(tz_tr)
    scan_time = now.strftime("%d.%m.%Y %H:%M")

    # VM'den P2 bekleyen listesini al
    bekleyenler = _vm_bekleyen_al_p2()
    if not bekleyenler:
        log.info("P2 teyit: bekleyen aday yok")
        return {}

    semboller = list({b["symbol"] for b in bekleyenler})
    ok = veri_hazirla(semboller, period="5d")
    if ok == 0:
        return {}

    teyitler = {}
    for sym in semboller:
        t = teyit_skoru_hesapla(sym)
        teyitler[sym] = t
        log.info("Teyit %s: bonus=%.1f var=%s sig=%s",
                 sym, t["teyit_skoru"], t["teyit_var"], t["teyit_sinyaller"])

    # VM'ye teyit güncellemesi gönder
    payload = {
        "scan_time": scan_time,
        "mod": "teyit",
        "teyitler": teyitler,
    }
    vm_gonder_p2(payload, scan_time, "Saatlik Teyit", mod="teyit")

    # Teyit var olanları Telegram'a bildir
    teyit_alanlar = [s for s, t in teyitler.items() if t["teyit_var"]]
    if teyit_alanlar:
        lines = [f"🟣 <b>P2-SMC Saatlik Teyit — {scan_time}</b>", ""]
        for sym in teyit_alanlar:
            t = teyitler[sym]
            sig_str = " + ".join(
                s.replace("Bullish ","") for s in t["teyit_sinyaller"]
            )
            lines.append(f"   ✅ <b>{sym}</b> +{t['teyit_skoru']:.1f} | {sig_str}")
        lines += ["", "⏰ Alım denemesi VM tarafında yapılacak."]
        send_telegram("\n".join(lines))

    return teyitler

# ── VM iletişim ────────────────────────────────────────────────────────────────

def _vm_bekleyen_al_p2() -> list:
    """state_p2.json'dan bekleyen_al listesini oku — GitHub Actions uyumlu."""
    state_file = BASE_DIR / "state_p2.json"
    try:
        if state_file.exists():
            with open(state_file, encoding="utf-8") as fh:
                return json.load(fh).get("bekleyen_al", [])
    except Exception as e:
        log.warning("state_p2.json okuma hatası: %s", e)
    return []

def vm_gonder_p2(payload: dict | list, scan_time: str, scan_label: str, mod: str = "aksam") -> dict:
    """state_p2.json'a yaz — GitHub Actions uyumlu, Flask gerektirmez."""
    state_file = BASE_DIR / "state_p2.json"
    try:
        existing = {}
        if state_file.exists():
            with open(state_file, encoding="utf-8") as fh:
                existing = json.load(fh)
        if mod == "teyit":
            teyitler = payload if isinstance(payload, dict) else {}
            bekleyen = existing.get("bekleyen_al", [])
            for item in bekleyen:
                sym = item["symbol"]
                if sym in teyitler:
                    t = teyitler[sym]
                    item["teyit_skoru"]     = t.get("teyit_skoru", 0)
                    item["teyit_var"]       = t.get("teyit_var", False)
                    item["teyit_sinyaller"] = t.get("teyit_sinyaller", [])
                    item["final_score"]     = item.get("score", 0) + t.get("teyit_skoru", 0)
            existing["bekleyen_al"] = bekleyen
        else:
            signals = payload if isinstance(payload, list) else payload.get("signals", [])
            existing["tarama"]      = {"scan_time": scan_time, "scan_label": scan_label, "signals": signals, "mod": mod}
            existing["bekleyen_al"] = signals
            existing["last_scan"]   = scan_time
        with open(state_file, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)
        n = len(payload) if isinstance(payload, list) else len(payload.get("signals", []))
        log.info("P2 state_p2.json yazıldı: %d sinyal, mod=%s", n, mod)
        return {"status": "ok", "n": n, "mod": mod}
    except Exception as e:
        log.warning("state_p2.json yazma hatası: %s", e)
        return {}

# ── Telegram ───────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    from datetime import datetime
    import pytz
    _now = datetime.now(pytz.timezone("Europe/Istanbul"))
    _header = f"[P2] {_now.strftime('%d.%m.%Y | %H:%M')}\n"
    message = _header + message
    if not BOT_TOKEN or not CHAT_ID:
        print(message); return
    max_len = 3900
    parcalar, parca = [], ""
    for satir in message.split("\n"):
        if len(parca) + len(satir) + 1 > max_len:
            if parca.strip(): parcalar.append(parca.strip())
            parca = satir + "\n"
        else:
            parca += satir + "\n"
    if parca.strip(): parcalar.append(parca.strip())
    for idx, p in enumerate(parcalar, 1):
        if len(parcalar) > 1:
            p = f"({idx}/{len(parcalar)})\n{p}"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": p,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=15,
            )
            if not resp.ok:
                log.error("Telegram %s: %s", resp.status_code, resp.text[:200])
            time.sleep(0.4)
        except Exception as e:
            log.error("Telegram: %s", e)

# ── Mesaj formatı ──────────────────────────────────────────────────────────────

_IKON = {
    "Bullish CHoCH":"🔄","Bearish CHoCH":"🔄",
    "Bullish BOS":"💥","Bearish BOS":"💥",
    "Bullish Sweep":"🌊","Bearish Sweep":"🌊",
    "Bullish OB":"📦","Bearish OB":"📦",
    "Bullish FVG":"🕳","Bearish FVG":"🕳",
    "Bullish MACD Cross":"📈","Bearish MACD Cross":"📉",
}

def format_aksam(guclu_al, al, scan_time, scan_label, n_symbols) -> str:
    lines = [
        f"🟣 <b>P2-SMC — {scan_label}</b>",
        f"📅 {scan_time}  |  1h→1D resample + 1W HTF",
        "",
    ]

    def bolum(baslik, emoji, results):
        rows = [f"{emoji} <b>{baslik}</b>  ({len(results)} hisse)"]
        if not results:
            rows.append("   - sinyal yok")
        else:
            for r in results[:7]:
                ai = r["ai_analysis"]
                sig_str = " + ".join(
                    f"{_IKON.get(s['type'],'•')}{s['type'].replace('Bullish ','').replace('Bearish ','')}"
                    for s in r["signals"] if "Bullish" in s["type"]
                )
                pd_tag  = "🟢D" if r["pd_zone"] == "Discount" else "🔴P"
                htf_tag = " 1W✓" if ai.get("htf_aligned") else ""
                rows.append(
                    f"   <b>{r['symbol']}</b> skor:<b>{ai['score']:+.1f}</b> "
                    f"RSI:{r['indicators']['rsi']:.0f} RV:{r['rvol']:.1f} {pd_tag}{htf_tag}"
                )
                if sig_str:
                    rows.append(f"   └ {sig_str}")
        rows.append("")
        return rows

    lines += bolum("GÜÇLÜ AL 🏆", "💎", guclu_al)
    lines += bolum("AL ✅", "✅", al)
    lines.append("--------------------")

    all_res = guclu_al + al
    def cnt(k): return sum(1 for r in all_res if any(k in s["type"] for s in r["signals"]))
    htf_cnt  = sum(1 for r in all_res if r["ai_analysis"].get("htf_aligned"))
    disc_cnt = sum(1 for r in all_res if r["pd_zone"] == "Discount")

    lines += [
        f"📊 Taranan:{n_symbols} | Sinyal:{len(all_res)} (💎{len(guclu_al)} ✅{len(al)})",
        f"🔄 CHoCH:{cnt('CHoCH')}  💥 BOS:{cnt('BOS')}  🌊 Sweep:{cnt('Sweep')}  📦 OB:{cnt('OB')}  🕳 FVG:{cnt('FVG')}",
        f"📐 Discount:{disc_cnt}  |  1W HTF uyumlu:{htf_cnt}",
        "",
        "⏰ Giriş: Ertesi 11:00  |  Saatlik teyit: 11:00-17:00",
        "🟣 P2-SMC  Stop:-5%  TP1:+8%  Trail:-5%  MaxGün:10",
        "",
        "⚠️ <i>Yatırım tavsiyesi değildir.</i>",
    ]
    return "\n".join(lines)

# ── Yardımcı ───────────────────────────────────────────────────────────────────

def _build_records(results, scan_time, scan_label):
    records = []
    for r in results:
        records.append({
            "symbol":      r["symbol"],
            "scan_time":   scan_time,
            "scan_label":  scan_label,
            "verdict":     r["ai_analysis"]["verdict"],
            "score":       r["ai_analysis"]["score"],
            "pd_zone":     r["pd_zone"],
            "rvol":        r["rvol"],
            "rsi":         r["indicators"]["rsi"],
            "htf_aligned": r["ai_analysis"].get("htf_aligned", False),
            "htf_trend":   r["ai_analysis"].get("htf_trend", ""),
            "signals":     [s["type"] for s in r["signals"]],
            "islem_tl":    r["islem_tl"],
            "change_pct":  r["change_pct"],
        })
    return records

def _append_signal_log(records):
    try:
        with SIGNAL_LOG.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("signal log: %s", e)

# ── Entry points ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "aksam"
    if cmd == "aksam":
        g, a = run_scan_smc(manual=False)
        print(f"GÜÇLÜ AL: {[r['symbol'] for r in g]}")
        print(f"AL:       {[r['symbol'] for r in a]}")
    elif cmd == "manuel":
        # İlk sefere özel: şimdi tara, sanki 20:50 gibi değerlendir
        g, a = run_scan_smc(manual=True)
        print(f"GÜÇLÜ AL: {[r['symbol'] for r in g]}")
        print(f"AL:       {[r['symbol'] for r in a]}")
    elif cmd == "teyit":
        t = saatlik_teyit_smc()
        print(json.dumps(t, ensure_ascii=False, indent=2))
    else:
        print("Kullanim: python scanner_smc.py [aksam|manuel|teyit]")
