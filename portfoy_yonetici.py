# -*- coding: utf-8 -*-
"""
BIST paper trading + portfolio manager.

Gün içi uygulama akışı:
  09:00 -> dünkü sinyaller + makro/VIOP/model -> "bugün şunları al" mesajı
  11:00 -> ilk alım denemesi
  11:30 -> portföy özeti
  12:00-17:00 -> saatlik risk kontrolü + tekrar alım denemesi
  17:30 -> kapanış özeti
  20:50 -> Cloud tarama (değişmez)
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import pickle
import uuid
import threading
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

import mott_risk
from mott_bist_takvim import to_tsi, is_bist_islem_gunu, is_bist_seans_acik
from p1_safety import StateConflict, file_digest, state_lock, start_ledger, reconcile, closed_daily


def _p1_now(now=None):
    return to_tsi(now)

try:
    _tz_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "yf_tz"
    yf.set_tz_cache_location(str(_tz_cache))
except Exception:
    pass


def _ticker(sym: str):
    return yf.Ticker(sym)


def _market_ticker(sym: str):
    return yf.Ticker(sym)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("portfoy_yonetici.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_TOKEN", ""))
CHAT_ID       = os.environ.get("TELEGRAM_CHAT_ID", "")
# FLASK_PORT kaldırıldı — GitHub Actions Flask sunucusu kullanmaz
FLASK_API_KEY = os.environ.get("FLASK_API_KEY", "")
VIOP_TICKER   = os.environ.get("VIOP_TICKER", "XU030.IS")

BASE_DIR             = Path(__file__).parent
PORTFOY_FILE         = BASE_DIR / "portfoy.json"
STATE_P1_FILE        = BASE_DIR / "state_p1.json"
TARAMA_FILE          = BASE_DIR / "tarama_listesi.json"
LGBM_MODEL_FILE      = BASE_DIR / "lgbm_model.pkl"
LGBM_META_FILE       = BASE_DIR / "lgbm_ozellikler.json"
DURUM_FILE           = BASE_DIR / "son_durum.json"
SIGNAL_AUDIT_FILE    = BASE_DIR / "signal_audit.jsonl"
PORTFOY_AUDIT_FILE   = BASE_DIR / "portfolio_actions.jsonl"
HISSELER_FILE        = BASE_DIR / "hisseler.txt"
TELEGRAM_OFFSET_FILE = BASE_DIR / "telegram_offset.txt"
PORTFOY_P2_FILE       = BASE_DIR / "portfoy_p2.json"
TARAMA_P2_FILE        = BASE_DIR / "tarama_listesi_p2.json"
SIGNAL_AUDIT_P2_FILE  = BASE_DIR / "signal_audit_p2.jsonl"
PORTFOY_AUDIT_P2_FILE = BASE_DIR / "portfolio_actions_p2.jsonl"


SERMAYE_BASLANGIC        = 100_000
P2_SERMAYE_BASLANGIC = 100_000
P2_MAX_HISSE         = 7
P2_HISSE_LIMIT       = 20_000
P2_STOP_PCT          = -0.05
P2_TP1_PCT           = 0.08
P2_TRAILING_PCT      = -0.05
P2_MAX_GUN           = 10
P2_MIN_SCORE         = 3.0
P2_TEYIT_BONUS_ESIK  = 2.0

MAX_HISSE                = 7
HEDEF_HISSE              = 5
HISSE_LIMIT              = 20_000
STOP_PCT                 = -0.05
TP1_PCT                  = 0.08
TRAILING_PCT             = -0.05
MAX_GUN                  = 10
MAX_GUN_EXTENSION        = 5
LGBM_MIN_SKOR            = 60
WAITING_EXPIRES_HOUR     = 17
EMERGENCY_LIQUIDATION_SCORE = 80
RETRY_REASONS = {"veri_yok", "exception", "lot_yetersiz", "nakit_yetersiz"}

# ── Merkezi İşlem Maliyeti ve Kayma Ayarları ────────────────────────────────
VARSAYILAN_KOMISYON_ORANI  = 0.0005  # onbinde 5 (%0.05) varsayılan komisyon
VARSAYILAN_KAYMA_ORANI     = 0.0010  # binde 1 (%0.10) varsayılan slippage
MALIYET_VARSAYIMI_ACIKLAMA = "varsayilan_onbinde_5_komisyon_binde_1_kayma"


def hesapla_cikis_kayma(tetik_f: float, open_f: float, kayma_orani: float = VARSAYILAN_KAYMA_ORANI) -> float:
    """Satış işleminde kayma ve gap down hesabı:
      Açılış stop/tetik seviyesinin altındaysa (gap down) açılıştan kayma düşülür.
    """
    if open_f < tetik_f:
        return round(open_f * (1.0 - kayma_orani), 4)
    return round(tetik_f * (1.0 - kayma_orani), 4)


def hesapla_tp_kayma(tetik_f: float, open_f: float, kayma_orani: float = VARSAYILAN_KAYMA_ORANI) -> float:
    """TP satışında: eğer açılış gap up ise (open_f > tetik_f), satış open_f üzerinden gerçekleşir."""
    f = max(tetik_f, open_f)
    return round(f * (1.0 - kayma_orani), 4)


def hesapla_net_tutar(lotlar: int, fiyat: float, komisyon_orani: float = VARSAYILAN_KOMISYON_ORANI, islem: str = "satis") -> tuple[float, float]:
    """Net nakit tutarı ve komisyonu hesaplar."""
    brut = lotlar * fiyat
    komisyon = round(brut * komisyon_orani, 4)
    if islem == "satis":
        return round(brut - komisyon, 4), komisyon
    return round(brut + komisyon, 4), komisyon


def _parse_tarih(t_val) -> date | None:
    if not t_val:
        return None
    if isinstance(t_val, date) and not isinstance(t_val, datetime):
        return t_val
    if isinstance(t_val, datetime):
        return t_val.date()
    s = str(t_val).strip().split(" ")[0].split("T")[0]
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _parse_datetime(dt_val) -> datetime | None:
    if not dt_val:
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    s = str(dt_val).strip()
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _elde_tutma_gunu(giris_t: str, now_date: date | None = None) -> int:
    """Pozisyonun kaç takvim günüdür açık olduğunu giriş tarihinden hesapla.

    Not: pos['gun'] sayacı yalnızca main() içindeki gün-sonu döngüsünde
    artıyordu; GitHub Actions bu döngüyü hiç çalıştırmadığı için MAX_GUN
    çıkışı hiçbir zaman tetiklenmiyordu. Bunun yerine giriş tarihinden
    itibaren geçen gerçek gün sayısını hesaplıyoruz — kontrol sıklığından
    (saatlik / 15 dk) bağımsız, her zaman doğru sonuç verir.
    """
    try:
        giris_tarih = _parse_tarih(giris_t)
        if not giris_tarih:
            return 0
        ref = now_date if now_date is not None else date.today()
        return (ref - giris_tarih).days
    except Exception:
        return 0


def _max_gun_date_hesapla_p1(giris_t: str, max_gun: int = MAX_GUN, now_date: date | None = None) -> str:
    """P1: Entry tarihinden MAX_GUN deadline hesapla (DD.MM.YYYY format)."""
    giris = _parse_tarih(giris_t)
    ref = now_date if now_date is not None else date.today()
    if giris:
        return (giris + timedelta(days=max_gun)).strftime("%d.%m.%Y")
    return (ref + timedelta(days=max_gun)).strftime("%d.%m.%Y")


def _p1_alim_listesi_durum(now: datetime | None = None) -> tuple[set[str], str, dict]:
    """P1 canonical ALIM listesi durumunu ve sembol kümesini döndürür.

    Dönüş: (symbols_set, status, metadata)
    status:
      'valid'   : Taze ve geçerli tarama listesi
      'stale'   : Süresi geçmiş tarama listesi (> 96 saat veya gelecek tarih)
      'missing' : Dosya veya scan_time yok
      'corrupt' : JSON veya veri bozuk
      'empty'   : Geçerli tarama fakat sinyal yok
    """
    ref_now = now if now is not None else datetime.now()
    if not TARAMA_FILE.exists():
        return set(), "missing", {"reason": "tarama_listesi_yok"}
    try:
        tarama = tarama_listesi_yukle()
    except Exception as exc:
        return set(), "corrupt", {"reason": f"json_hatasi:{exc}"}

    scan_time_raw = tarama.get("scan_time", "")
    if not scan_time_raw:
        return set(), "missing", {"reason": "scan_time_yok"}

    scan_dt = _parse_datetime(scan_time_raw)
    if scan_dt is None:
        return set(), "corrupt", {"reason": f"gecersiz_scan_time:{scan_time_raw}"}

    if scan_dt.tzinfo is not None and ref_now.tzinfo is None:
        scan_dt = scan_dt.replace(tzinfo=None)
    elif scan_dt.tzinfo is None and ref_now.tzinfo is not None:
        ref_now = ref_now.replace(tzinfo=None)

    diff = ref_now - scan_dt
    if diff.total_seconds() < -300:
        return set(), "stale", {"reason": "gelecek_tarihli_tarama", "scan_time": scan_time_raw}

    if diff.total_seconds() > 96 * 3600:
        return set(), "stale", {
            "reason": "bayat_tarama_96h_asildi",
            "scan_time": scan_time_raw,
            "gecen_saat": round(diff.total_seconds() / 3600, 1),
        }

    signals = tarama.get("signals", [])
    symbols = {s["symbol"] for s in signals if isinstance(s, dict) and "symbol" in s}
    if not symbols:
        return set(), "empty", {"reason": "sinyal_yok", "scan_time": scan_time_raw}

    return symbols, "valid", {"scan_time": scan_time_raw, "signal_count": len(symbols)}


def _p1_alim_listesi(now: datetime | None = None) -> set[str]:
    """P1 canonical ALIM listesi — bugünkü tarama.
    Stale veya hata durumunda boş küme döner (güvenli EXIT)."""
    symbols, status, _ = _p1_alim_listesi_durum(now=now)
    return symbols if status == "valid" else set()


def _p2_alim_listesi() -> set[str]:
    """P2 canonical ALIM listesi — bugünkü tarama.
    Stale veya hata durumunda boş küme döner (güvenli EXIT)."""
    try:
        p2_tarama = BASE_DIR / "tarama_listesi_p2.json"
        if not p2_tarama.exists():
            return set()
        with open(p2_tarama, encoding="utf-8") as fh:
            data = json.load(fh)
        scan_time = data.get("scan_time", "")
        bugun = date.today().isoformat()
        if isinstance(data, list):
            return set(data) if scan_time and scan_time[:10] == bugun else set()
        if scan_time and scan_time[:10] != bugun:
            return set()
        signals = data.get("signals", [])
        if isinstance(signals, list) and signals:
            if isinstance(signals[0], str):
                return set(signals)
            return {s["symbol"] for s in signals}
        return set()
    except Exception:
        return set()


def _p2_trade_kaydet(portfoy: dict, sym: str, pos: dict, cikis_f: float,
                  neden: str, lotlar: int | None = None):
    """Kapanan (veya kısmi kapanan) işlemi portföy JSON'ındaki trade_history'ye
    yaz — P4/P5 şemasıyla uyumlu. Audit JSONL'e ek olarak tutulur; kalıcı
    performans raporu ve cooldown kontrolü bu listeden beslenir."""
    giris_f = pos.get("giris_f", 0) or 0
    pnl = (cikis_f - giris_f) / giris_f * 100 if giris_f else 0.0
    giris_t = str(pos.get("giris_t", ""))
    giris_iso = ""
    try:
        giris_iso = datetime.strptime(giris_t.split(" ")[0], "%d.%m.%Y").date().isoformat()
    except Exception:
        giris_iso = giris_t[:10]
    portfoy.setdefault("trade_history", []).append({
        "symbol":       sym,
        "giris_fiyat":  giris_f,
        "cikis_fiyat":  round(float(cikis_f), 4),
        "lotlar":       lotlar if lotlar is not None else pos.get("lotlar", 0),
        "pnl_pct":      round(pnl, 2),
        "gun":          _elde_tutma_gunu(giris_t),
        "neden":        neden,
        "giris_tarih":  giris_iso,
        "cikis_tarih":  date.today().isoformat(),
    })


def _trade_kaydet(
    portfoy: dict,
    sym: str,
    pos: dict,
    cikis_f: float,
    neden: str,
    lotlar: int | None = None,
    tp1_tetik_fiyat: float | None = None,
    ambiguity: str | None = None,
    fiyat_kaynak: str | None = None,
    fiyat_zaman: str | None = None,
    likidite_teyitli: bool = True,
    event_id: str | None = None,
    now: datetime | None = None,
):
    """Kapanan (veya kısmi kapanan) işlemi portföy JSON'ındaki trade_history'ye
    ve işlem defterine yaz — P1/P4/P5 şemasıyla uyumlu. Audit JSONL'e ek olarak tutulur;
    kalıcı performans raporu, denetim ve cooldown kontrolü bu listeden beslenir."""
    ref_now = now if now is not None else datetime.now()
    giris_f = pos.get("giris_f", 0) or 0
    actual_lots = int(lotlar if lotlar is not None else pos.get("lotlar", 0))
    pnl = (cikis_f - giris_f) / giris_f * 100 if giris_f else 0.0
    giris_t = str(pos.get("giris_t", ""))
    giris_iso = ""
    try:
        giris_iso = datetime.strptime(giris_t.split(" ")[0], "%d.%m.%Y").date().isoformat()
    except Exception:
        giris_iso = giris_t[:10]

    pos_id = pos.setdefault("position_id", "P1_" + uuid.uuid5(uuid.NAMESPACE_URL, f"P1/{sym}/{giris_t}/{giris_f}").hex)
    evt_id = event_id or "EVT_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{pos_id}/{neden}/{pos.get('decision_bar_time', ref_now.isoformat())}/{actual_lots}").hex
    if any(t.get("event_id") == evt_id for t in portfoy.get("trade_history", [])):
        raise ValueError("Duplicate P1 exit event; economic change must not be repeated")

    net_tutar, komisyon = hesapla_net_tutar(actual_lots, cikis_f, islem="satis")
    brut_tutar = actual_lots * cikis_f
    maliyet = actual_lots * giris_f
    entry_lots = int(pos.get("entry_lotlar", actual_lots))
    buy_fee = float(pos.get("entry_komisyon", 0)) * actual_lots / entry_lots
    net_tl_kar = round(net_tutar - maliyet - buy_fee, 4)
    net_pnl_pct = (net_tutar - maliyet - buy_fee) / (maliyet + buy_fee) * 100 if maliyet else 0.0
    brut_tl_kar = round(brut_tutar - maliyet, 2)

    rec = {
        "symbol":       sym,
        "giris_fiyat":  giris_f,
        "cikis_fiyat":  round(float(cikis_f), 4),
        "lotlar":       actual_lots,
        "pnl_pct":      round(pnl, 2),
        "net_pnl_pct":  round(net_pnl_pct, 4),
        "gun":          _elde_tutma_gunu(giris_t, now_date=ref_now.date()),
        "neden":        neden,
        "giris_tarih":  giris_iso,
        "cikis_tarih":  ref_now.date().isoformat(),
        # Ek muhasebe ve denetim alanları
        "position_id":       pos_id,
        "event_id":          evt_id,
        "tl_kar":            net_tl_kar,
        "brut_tl_kar":       brut_tl_kar,
        "komisyon":          komisyon,
        "alis_komisyon_payi": buy_fee,
        "position_closed": pos.get("lotlar", 0) == 0 or neden != "TP1",
        "maliyet_bilgisi_tam": "entry_komisyon" in pos,
        "source_signal": dict(pos.get("source_signal", {})),
        "kayma_orani":       VARSAYILAN_KAYMA_ORANI,
        "maliyet_varsayimi": MALIYET_VARSAYIMI_ACIKLAMA,
        "likidite_teyitli":  likidite_teyitli,
    }
    if tp1_tetik_fiyat is not None:
        rec["tp1_trigger_price"] = tp1_tetik_fiyat
    if ambiguity is not None:
        rec["ambiguity"] = ambiguity
    if fiyat_kaynak:
        rec["fiyat_kaynak"] = fiyat_kaynak
    if fiyat_zaman:
        rec["fiyat_zaman"] = fiyat_zaman

    portfoy.setdefault("trade_history", []).append(rec)

    # İşlem defteri (Ledger): nakit ve lot hareketlerinin yeniden hesaplanabilmesi için
    portfoy.setdefault("islem_defteri", []).append({
        "event_id":     evt_id,
        "position_id":  pos_id,
        "symbol":       sym,
        "islem_tipi":   f"SATIS_{neden}",
        "lot":          actual_lots,
        "fiyat":        round(float(cikis_f), 4),
        "brut_tutar":   round(brut_tutar, 4),
        "komisyon":     komisyon,
        "nakit_etkisi": round(net_tutar, 4),
        "zaman":        ref_now.isoformat(),
    })

# ── LGBM global (uygulama başında bir kez yüklenir) ──────────────────────────
_LGBM_MODEL: object  = None
_LGBM_STATUS: str    = "pasif"

STRATEGY_WEIGHTS = {
    "GT":    18, "ZT3":   18, "ALPHA": 16,
    "GTD":   14, "ZKN":   14, "DIP":   12,
    "MR":    10, "KBM":    8,
}

MAKRO_KURALLAR = [
    ("^GSPC",      1, "down", -2.0, 15.0),
    ("^IXIC",      1, "down", -3.0, 12.0),
    ("^VIX",       1, "up",    3.0, 20.0),
    ("^N225",      0, "down", -2.0, 12.0),
    ("^HSI",        0, "down", -2.0, 10.0),
    ("000001.SS",  0, "down", -2.0,  8.0),
    ("GC=F",       1, "up",    2.0,  8.0),
]
ASYA_TICKERLAR = ["^N225", "^HSI", "000001.SS"]
ASYA_ESIK      = -2.0
ASYA_MIN_TETIK = 2


# ── Makro skor cache (60 dakika) ─────────────────────────────────────────────
_MAKRO_CACHE: dict = {"value": None, "fetched_at": None}
MAKRO_CACHE_TTL = 3600  # saniye


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def sonraki_islem_gunu(ref_date: date) -> date:
    nxt = ref_date + timedelta(days=1)
    while not is_bist_islem_gunu(nxt):
        nxt += timedelta(days=1)
    return nxt


def lgbm_model_yukle():
    """Global cache'den döner — disk okuma sadece ilk çağrıda yapılır."""
    global _LGBM_MODEL, _LGBM_STATUS
    if _LGBM_STATUS != "pasif" or _LGBM_MODEL is not None:
        return _LGBM_MODEL, _LGBM_STATUS
    if not LGBM_MODEL_FILE.exists():
        _LGBM_STATUS = "pasif"
        return None, "pasif"
    try:
        with open(LGBM_MODEL_FILE, "rb") as fh:
            model = pickle.load(fh)
        if not hasattr(model, "predict_proba"):
            _LGBM_STATUS = "pasif"
            return None, "pasif"
        _LGBM_MODEL  = model
        _LGBM_STATUS = "aktif"
        log.info("LGBM model yuklendi (global cache)")
        return _LGBM_MODEL, _LGBM_STATUS
    except Exception as exc:
        log.warning("LGBM model yuklenemedi: %s", exc)
        _LGBM_STATUS = "pasif"
        return None, "pasif"


def guncel_makro_skoru() -> float:
    if not DURUM_FILE.exists():
        return 0.0
    try:
        with open(DURUM_FILE, encoding="utf-8") as fh:
            return float(json.load(fh).get("makro_skor", 0.0) or 0.0)
    except Exception:
        return 0.0


_FIYAT_CACHE: dict = {}


def _quote_quality(quote: dict, now: datetime) -> dict:
    """A numeric quote is not necessarily a tradable or current quote."""
    result = dict(quote)
    result["observed_at"] = _p1_now(now).isoformat()
    stamp = _parse_datetime(quote.get("time"))
    known = bool(stamp) and quote.get("source") == "yfinance_1m"
    age = (_p1_now(now) - to_tsi(stamp)).total_seconds() if known else None
    fresh = known and -60 <= age <= 20*60
    result["source_time_known"] = known
    result["age_seconds"] = age
    result["trade_eligible"] = bool(quote.get("valid")) and fresh and is_bist_seans_acik(_p1_now(now))
    result["valuation_valid"] = bool(quote.get("valid")) and fresh
    if not known:
        result["reason"] = "source_time_unverified" if quote.get("source") != "yfinance_1d_close" else "daily_close_not_execution_price"
    elif not fresh:
        result["reason"] = "stale_or_future_quote"
    return result


def guncel_fiyat_detayli(symbol: str, cache: bool = True, now: datetime | None = None) -> dict:
    ref_now = _p1_now(now)
    cached = _FIYAT_CACHE.get(symbol)
    if cache and cached and 0 <= (ref_now - to_tsi(cached["_cached_at"])).total_seconds() < 60:
        return _quote_quality({k: v for k, v in cached.items() if not k.startswith("_")}, ref_now)
    result = _guncel_fiyat_detayli_cek(symbol, ref_now)
    result = _quote_quality(result, ref_now)
    if cache and result.get("valid"):
        _FIYAT_CACHE[symbol] = {**result, "_cached_at": ref_now}
    return result


def _guncel_fiyat_detayli_cek(symbol: str, ref_now: datetime) -> dict:
    candidates = []
    def add(price, source, timestamp=None):
        if price is not None and np.isfinite(float(price)) and float(price) > 0:
            candidates.append({"symbol": symbol, "price": float(price), "source": source,
                               "time": timestamp, "valid": True, "reason": "ok"})
    try:
        from mott_fiyat import tv_fiyatlar
        add(tv_fiyatlar([symbol]).get(symbol), "tradingview")
    except Exception:
        pass
    ticker = _ticker(f"{symbol}.IS")
    try:
        df = ticker.history(period="1d", interval="1m")
        if not df.empty:
            add(df["Close"].iloc[-1], "yfinance_1m", df.index[-1].isoformat())
            if _quote_quality(candidates[-1], ref_now)["valuation_valid"]:
                return candidates[-1]
    except Exception:
        pass
    try:
        add(ticker.fast_info.get("last_price") or ticker.fast_info.get("lastPrice"), "yfinance_fast_info")
    except Exception:
        pass
    try:
        df = ticker.history(period="5d", interval="1d")
        if not df.empty:
            add(df["Close"].iloc[-1], "yfinance_1d_close", df.index[-1].isoformat())
    except Exception:
        pass
    return candidates[0] if candidates else {"symbol": symbol, "price": None, "source": "none", "time": None, "valid": False, "reason": "fiyat_bulunamadi"}


def guncel_fiyat(symbol: str, cache: bool = True) -> float | None:
    res = guncel_fiyat_detayli(symbol, cache=cache)
    return res["price"] if res.get("valid") else None


def saatlik_bar(symbol: str, now: datetime | None = None) -> dict | None:
    """Son kapanmış saatlik mumu zaman bilgisine göre seçer (iloc[-2] varsayımına dayanmaz).

    Bir mum [T, T + 1 saat) aralığını kapsar; ancak T + 1 saat tamamlandıktan
    sonra kapanmış sayılır. ref_now'dan önce tamamlanmış en son bar seçilir.
    """
    ref_now = _p1_now(now)
    try:
        df = _ticker(f"{symbol}.IS").history(period="5d", interval="60m")
        if df is None or len(df) == 0:
            return None

        closed_bars = []
        for idx, row in df.iterrows():
            bar_start = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            bar_start = to_tsi(bar_start)

            bar_end = bar_start + timedelta(hours=1)
            if bar_end <= ref_now and (ref_now-bar_end).total_seconds() <= 90*60 and is_bist_islem_gunu(bar_start.date()):
                c = float(row["Close"])
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                if min(c, o, h, l) > 0 and np.isfinite([c,o,h,l]).all() and l <= min(o,c) <= max(o,c) <= h:
                    closed_bars.append({
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": float(row.get("Volume", 0) or 0),
                        "bar_time": bar_start.isoformat(),
                        "bar_end": bar_end.isoformat(),
                        "source": "yfinance_60m",
                    })

        if not closed_bars:
            return None

        return closed_bars[-1]
    except Exception as exc:
        log.debug("saatlik_bar %s: %s", symbol, exc)
        return None




# ─────────────────────────────────────────────────────────────────────────────
# P2-SMC Portföy Fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def p2_portfoy_yukle() -> dict:
    if PORTFOY_P2_FILE.exists():
        with open(PORTFOY_P2_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"pozisyonlar": {}, "nakit": P2_SERMAYE_BASLANGIC, "baslangic": P2_SERMAYE_BASLANGIC}
    data.setdefault("bekleyen_al", [])
    data.setdefault("open_attempts_today", [])
    data.setdefault("last_hourly_check_time", "")
    data.setdefault("trade_history", [])
    return data


def p2_portfoy_kaydet(portfoy: dict):
    with open(PORTFOY_P2_FILE, "w", encoding="utf-8") as fh:
        json.dump(portfoy, fh, indent=2, ensure_ascii=False)


def p2_portfolio_preview(portfoy: dict) -> dict:
    nakit  = portfoy.get("nakit", 0)
    equity = nakit
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        equity += pos["lotlar"] * (f if f else pos["giris_f"])
    kazanc_pct = (equity - portfoy.get("baslangic", P2_SERMAYE_BASLANGIC)) / portfoy.get("baslangic", P2_SERMAYE_BASLANGIC) * 100
    return {
        "cash": round(nakit, 2), "equity": round(equity, 2),
        "n_positions": len(portfoy.get("pozisyonlar", {})),
        "baslangic": portfoy.get("baslangic", P2_SERMAYE_BASLANGIC),
        "kazanc_pct": round(kazanc_pct, 2),
    }


def p2_yeni_pozisyon_ac(portfoy: dict, adaylar: list, makro_karar: str) -> tuple:
    if makro_karar == "GIRME":
        return portfoy, [], [], [{"symbol": a["symbol"], "reason": "makro_girme"} for a in adaylar]
    mesajlar, alinan, alinmayan = [], [], []
    mevcut   = portfoy["pozisyonlar"]
    nakit    = portfoy["nakit"]
    bos_slot = max(0, P2_MAX_HISSE - len(mevcut))
    rejim_limit = mott_risk.rejim_slot_limiti(makro_karar)
    if rejim_limit is not None:
        bos_slot = min(bos_slot, rejim_limit)
    if bos_slot == 0 or not adaylar:
        return portfoy, [], [], []
    adaylar_s  = sorted(adaylar, key=lambda x: -x.get("final_score", 0))
    secilenler = [a for a in adaylar_s if a.get("final_score", 0) >= P2_MIN_SCORE][:bos_slot]
    toplam_skor = sum(max(a.get("final_score", 1), 1) for a in secilenler) or 1
    for aday in secilenler:
        sym = aday["symbol"]
        if sym in mevcut:
            alinmayan.append({"symbol": sym, "reason": "already_open"})
            continue
        if mott_risk.cooldown_da(portfoy.get("trade_history"), sym):
            alinmayan.append({"symbol": sym, "reason": "cooldown"})
            append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "cooldown"})
            continue
        if mott_risk.kitap_limiti_asildi(sym, haric="P2"):
            alinmayan.append({"symbol": sym, "reason": "kitap_limiti"})
            append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "kitap_limiti"})
            continue
        try:
            giris_f = guncel_fiyat(sym)
            if giris_f is None:
                alinmayan.append({"symbol": sym, "reason": "veri_yok"})
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "veri_yok"})
                continue
            alloc  = min(P2_HISSE_LIMIT, nakit * (aday.get("final_score", 1) / toplam_skor))
            lotlar = int(alloc / max(giris_f, 0.01))
            if lotlar < 1:
                alinmayan.append({"symbol": sym, "reason": "lot_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "lot_yetersiz"})
                continue
            maliyet = lotlar * giris_f
            if maliyet > nakit:
                alinmayan.append({"symbol": sym, "reason": "nakit_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "nakit_yetersiz"})
                continue
            nakit -= maliyet
            mevcut[sym] = {
                "giris_f": round(giris_f, 4), "giris_t": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "tepe_f": round(giris_f, 4), "lotlar": lotlar, "gun": 0, "tp1_yapildi": False,
                "max_gun_date": (date.today() + timedelta(days=P2_MAX_GUN)).strftime("%d.%m.%Y"),
                "smc_score": aday.get("score", 0), "teyit_skoru": aday.get("teyit_skoru", 0),
                "signals": aday.get("signals", []),
            }
            mesajlar.append(
                f"\U0001f6a8 <b>P2-AL - {sym}</b>\n"
                f"   {lotlar} lot @ {giris_f:.2f} TL\n"
                f"   SMC:{aday.get('score',0):.1f} teyit:+{aday.get('teyit_skoru',0):.1f}"
            )
            append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_success", "symbol": sym,
                                                   "price": giris_f, "lots": lotlar,
                                                   "smc_score": aday.get("score", 0)})
            alinan.append(sym)
        except Exception as exc:
            log.exception("P2 pozisyon HATA %s", sym)
            alinmayan.append({"symbol": sym, "reason": "exception"})
            append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": f"exception:{exc}"})
    portfoy["nakit"] = nakit
    return portfoy, mesajlar, alinan, alinmayan


def p2_pozisyon_kontrol(portfoy: dict) -> tuple:
    mesajlar, kapatilacak = [], []
    for sym, pos in list(portfoy["pozisyonlar"].items()):
        try:
            bar = saatlik_bar(sym)
            if bar is None:
                continue
            high, low, close = bar["high"], bar["low"], bar["close"]
            giris_f = pos["giris_f"]
            lotlar  = pos["lotlar"]
            pos["tepe_f"] = max(pos.get("tepe_f", giris_f), high)
            if (low - giris_f) / giris_f <= P2_STOP_PCT:
                cikis_f = round(giris_f * (1 + P2_STOP_PCT), 4)
                portfoy["nakit"] += lotlar * cikis_f
                _p2_trade_kaydet(portfoy, sym, pos, cikis_f, "STOP")
                kapatilacak.append(sym)
                mesajlar.append(f"\U0001f6d1 <b>P2-STOP - {sym}</b>\n   {giris_f:.2f} \u2192 {cikis_f:.2f}")
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "stop", "symbol": sym,
                                                      "exit_price": cikis_f, "return_pct": P2_STOP_PCT * 100})
                continue
            if (high - giris_f) / giris_f >= P2_TP1_PCT and not pos.get("tp1_yapildi"):
                yari = max(1, lotlar // 2)
                portfoy["nakit"] += yari * close
                pos["lotlar"]    -= yari
                pos["tp1_yapildi"] = True
                _p2_trade_kaydet(portfoy, sym, pos, close, "TP1", lotlar=yari)
                mesajlar.append(f"\U0001f3af <b>P2-TP1 - {sym}</b>\n   {yari} lot @ {close:.2f} (+{P2_TP1_PCT*100:.0f}%)")
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "tp1", "symbol": sym,
                                                      "price": close, "remaining": pos["lotlar"]})
            elif pos.get("tp1_yapildi"):
                trail_ret = (low - pos["tepe_f"]) / pos["tepe_f"]
                if trail_ret <= P2_TRAILING_PCT:
                    cikis_f = round(pos["tepe_f"] * (1 + P2_TRAILING_PCT), 4)
                    portfoy["nakit"] += pos["lotlar"] * cikis_f
                    _p2_trade_kaydet(portfoy, sym, pos, cikis_f, "TRAILING")
                    kapatilacak.append(sym)
                    ret_g = (cikis_f - giris_f) / giris_f
                    mesajlar.append(f"\U0001f4c9 <b>P2-TRAIL - {sym}</b>\n   {cikis_f:.2f} | {ret_g*100:+.1f}%")
                    append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "trailing", "symbol": sym,
                                                          "exit_price": cikis_f, "return_pct": ret_g * 100})
                    continue
            if _elde_tutma_gunu(pos.get("giris_t", "")) >= P2_MAX_GUN:
                # Rolling extension: MAX_GUN gününde P2 ALIM listesi kontrolü
                mgd_str = pos.get("max_gun_date") or _max_gun_date_hesapla_p1(pos.get("giris_t", ""), P2_MAX_GUN)
                try:
                    mgd = datetime.strptime(mgd_str, "%d.%m.%Y").date()
                except Exception:
                    mgd = date.today()
                if date.today() >= mgd:
                    alim = _p2_alim_listesi()
                    if sym in alim:
                        pos["max_gun_date"] = (date.today() + timedelta(days=MAX_GUN_EXTENSION)).strftime("%d.%m.%Y")
                        continue
                    # else: EXIT below
                portfoy["nakit"] += pos["lotlar"] * close
                _p2_trade_kaydet(portfoy, sym, pos, close, "MAX_GUN")
                kapatilacak.append(sym)
                gun_ret = (close - giris_f) / giris_f
                mesajlar.append(f"\u23f0 <b>P2-MAXGUN - {sym}</b>\n   {close:.2f} | {gun_ret*100:+.1f}%")
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "max_day", "symbol": sym,
                                                      "exit_price": close, "return_pct": gun_ret * 100})
        except Exception as exc:
            log.debug("P2 kontrol %s: %s", sym, exc)
    for sym in kapatilacak:
        portfoy["pozisyonlar"].pop(sym, None)
    return portfoy, mesajlar


def p2_ozet_mesaji(portfoy: dict, saat_label: str) -> str:
    prev = p2_portfolio_preview(portfoy)
    lines = [
        f"\U0001f7e3 <b>P2-SMC PORTFOY - {saat_label}</b>",
        "--------------------",
        f"\U0001f4b0 Baslangic : {prev['baslangic']:,.0f} TL",
        f"\U0001f4ca Guncel    : {prev['equity']:,.0f} TL ({prev['kazanc_pct']:+.1f}%)",
        f"\U0001f4b5 Nakit     : {prev['cash']:,.0f} TL",
    ]
    if portfoy["pozisyonlar"]:
        for sym, pos in portfoy["pozisyonlar"].items():
            f = guncel_fiyat(sym)
            ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f else 0
            tp1_tag = " TP1✓" if pos.get("tp1_yapildi") else ""
            lines.append(
                f"   \u2022 <b>{sym}</b> {pos['lotlar']}lot @ {pos['giris_f']:.2f}"
                f" \u2192 {f:.2f if f else '?'} ({ret:+.1f}%) gun:{pos.get('gun',0)}{tp1_tag}"
            )
    else:
        lines.append("   - Acik pozisyon yok")
    bekleyen = portfoy.get("bekleyen_al", [])
    if bekleyen:
        lines.append(f"\U0001f4cb Bekleyen: {', '.join(b['symbol'] for b in bekleyen[:7])}")
    return "\n".join(lines)


def _p2_acik_detay(portfoy: dict) -> list[dict]:
    detay = []
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f and pos.get("giris_f") else pos.get("pnl_pct")
        detay.append({"symbol": sym, "pnl_pct": ret})
    return detay


def _p2_mott_portfoy(portfoy: dict) -> dict:
    prev = p2_portfolio_preview(portfoy)
    return {
        "pozisyonlar":   portfoy.get("pozisyonlar", {}),
        "trade_history": portfoy.get("trade_history", []),
        "baslangic":     portfoy.get("baslangic", P2_SERMAYE_BASLANGIC),
        "equity":        prev["equity"],
        "acik_detay":    _p2_acik_detay(portfoy),
    }


def _p2_telegram_islem(
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list | None = None,
) -> None:
    """Yalnızca alım/satım varsa P2 formatında Telegram gönder."""
    try:
        from mott_telegram import telegram_islem_gonder, yukle_p2_sinyaller
        telegram_islem_gonder(
            "P2",
            sinyaller=yukle_p2_sinyaller(),
            portfoy=_p2_mott_portfoy(portfoy),
            giris=giris or [],
            cikis=cikis or [],
            mesajlar=mesajlar,
        )
    except Exception as exc:
        log.warning("P2 Telegram hatasi: %s", exc)


def p2_saatlik_kontrol(makro_karar: str):
    now = datetime.now()
    p2_teyit_senkronize_et()
    portfoy = p2_portfoy_yukle()
    onceki = set(portfoy["pozisyonlar"].keys())
    mesajlar = []
    if portfoy["pozisyonlar"]:
        portfoy, islem_msg = p2_pozisyon_kontrol(portfoy)
        mesajlar.extend(islem_msg)
    bekleyen = portfoy.get("bekleyen_al", [])
    if bekleyen and makro_karar != "GIRME":
        portfoy, al_msg, alinan, _ = p2_yeni_pozisyon_ac(portfoy, bekleyen, makro_karar)
        mesajlar.extend(al_msg)
        portfoy["bekleyen_al"] = [b for b in bekleyen if b["symbol"] not in set(alinan)]
    portfoy["last_hourly_check_time"] = now.strftime("%d.%m.%Y %H:%M")
    p2_portfoy_kaydet(portfoy)
    if mesajlar:
        sonra = set(portfoy["pozisyonlar"].keys())
        _p2_telegram_islem(
            portfoy,
            giris=list(sonra - onceki),
            cikis=list(onceki - sonra),
            mesajlar=mesajlar,
        )


def p2_gun_sonu_guncelle():
    portfoy = p2_portfoy_yukle()
    if portfoy["pozisyonlar"]:
        for pos in portfoy["pozisyonlar"].values():
            pos["gun"] = pos.get("gun", 0) + 1
        p2_portfoy_kaydet(portfoy)


def p2_adaylari_yukle_ve_hazirla():
    """
    Akşam SMC taramasından (state_p2.json) adayları oku,
    portfoy_p2.json'un bekleyen_al listesine aktar.

    scanner_smc.py yalnızca tarama yapar (state_p2.json yazar) — gerçek
    pozisyon açma/kapama işlemleri burada, portfoy_p2.json üzerinden yapılır.
    Bu köprü olmadan P2 hiçbir zaman gerçek işlem yapamaz.
    """
    state_file = BASE_DIR / "state_p2.json"
    if not state_file.exists():
        log.info("P2 aday hazırlama: state_p2.json yok")
        return
    try:
        with open(state_file, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as exc:
        log.warning("P2 state okunamadı: %s", exc)
        return

    sinyaller = state.get("tarama", {}).get("signals", [])
    portfoy = p2_portfoy_yukle()
    mevcut = set(portfoy["pozisyonlar"].keys())
    bekleyen = []
    for s in sinyaller:
        sym = s.get("symbol")
        if not sym or sym in mevcut:
            continue
        skor = float(s.get("score", 0))
        if skor < P2_MIN_SCORE:
            continue
        bekleyen.append({
            "symbol": sym,
            "score": skor,
            "final_score": skor,
            "verdict": s.get("verdict", ""),
            "signals": s.get("signals", []),
            "teyit_skoru": 0.0,
            "teyit_var": False,
            "queued_at": state.get("last_scan", ""),
            "attempt_count": 0,
            "last_attempt_reason": "",
        })
    portfoy["bekleyen_al"] = bekleyen
    portfoy["open_attempts_today"] = []
    p2_portfoy_kaydet(portfoy)
    log.info("P2 aday hazırlandı: %d aday bekleyen_al'a eklendi", len(bekleyen))


def p2_teyit_senkronize_et():
    """state_p2.json'daki (scanner_smc.py teyit modu) teyit skorlarını
    portfoy_p2.json'un bekleyen_al listesine aktarır."""
    state_file = BASE_DIR / "state_p2.json"
    if not state_file.exists():
        return
    try:
        with open(state_file, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return
    teyit_map = {b["symbol"]: b for b in state.get("bekleyen_al", [])}
    if not teyit_map:
        return
    portfoy = p2_portfoy_yukle()
    for item in portfoy.get("bekleyen_al", []):
        t = teyit_map.get(item["symbol"])
        if t:
            item["teyit_skoru"] = t.get("teyit_skoru", 0)
            item["teyit_var"] = t.get("teyit_var", False)
            item["final_score"] = item.get("score", 0) + t.get("teyit_skoru", 0)
    p2_portfoy_kaydet(portfoy)

def portfoy_yukle() -> dict:
    if PORTFOY_FILE.exists():
        raw = PORTFOY_FILE.read_bytes()
        data = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
    else:
        data = {"pozisyonlar": {}, "nakit": SERMAYE_BASLANGIC, "baslangic": SERMAYE_BASLANGIC}
        digest = None
    data.setdefault("bekleyen_al", [])
    data.setdefault("open_attempts_today", [])
    data.setdefault("last_open_attempt_summary", {})
    data.setdefault("last_hourly_check_time", "")
    data.setdefault("trade_history", [])
    data.setdefault("islem_defteri", [])
    data.setdefault("veri_hatalari", [])
    data["_initial_gen"] = data.get("_gen", 0)
    data["_initial_digest"] = digest
    return data


def veri_hatasi_kaydet(
    portfoy: dict,
    symbol: str,
    data_time: str,
    stage: str,
    reason: str,
    now: datetime | None = None,
) -> dict:
    """Veri hatalarını sessizce geçmeyip yapılandırılmış biçimde kaydeder."""
    ref_now = now if now is not None else datetime.now()
    run_id = str(os.environ.get("GITHUB_RUN_ID") or f"run_{ref_now.strftime('%Y%m%d_%H%M%S')}")
    err_entry = {
        "event": "data_error",
        "symbol": symbol,
        "data_time": str(data_time),
        "run_id": run_id,
        "stage": stage,
        "reason": reason,
        "timestamp": ref_now.isoformat(),
    }
    log.warning("Veri Hatasi [%s] sembol=%s veri_zamani=%s sebep=%s run_id=%s", stage, symbol, data_time, reason, run_id)
    append_jsonl(PORTFOY_AUDIT_FILE, err_entry)
    hatalar = portfoy.setdefault("veri_hatalari", [])
    hatalar.append(err_entry)
    if len(hatalar) > 50:
        portfoy["veri_hatalari"] = hatalar[-50:]
    return portfoy


def gunluk_equity_kaydet(portfoy: dict, now: datetime | None = None) -> dict:
    """Günlük equity kaydını ileriye dönük oluşturur (geçmiş günlük equity'yi tahminle üretmez)."""
    ref_now = _p1_now(now)
    bugun = ref_now.strftime("%Y-%m-%d")
    tarihce = portfoy.setdefault("gunluk_equity_tarihcesi", [])
    nakit = float(portfoy.get("nakit", 0.0) or 0.0)
    acik_deger = 0.0
    fiyat_eksik = []
    for sym, pos in (portfoy.get("pozisyonlar") or {}).items():
        if isinstance(pos, dict):
            lot = int(pos.get("lotlar", 0) or 0)
            detail = guncel_fiyat_detayli(sym, now=ref_now)
            if not detail.get("valuation_valid"):
                fiyat_eksik.append(sym)
            f = detail.get("price") if detail.get("valuation_valid") else float(pos.get("giris_f", 0.0) or 0.0)
            acik_deger += lot * f
    equity = round(nakit + acik_deger, 2)
    idx = next((i for i, k in enumerate(tarihce) if isinstance(k, dict) and k.get("tarih") == bugun), None)
    rec = {
        "tarih": bugun,
        "equity": None if fiyat_eksik else equity,
        "equity_cost_estimate": equity,
        "degerleme_eksik": fiyat_eksik,
        "nakit": round(nakit, 2),
        "acik_deger": round(acik_deger, 2),
        "zaman": ref_now.isoformat(),
    }
    if idx is not None:
        tarihce[idx] = rec
    else:
        tarihce.append(rec)
    return portfoy


def portfoy_kaydet(portfoy: dict, force: bool = False) -> dict:
    """Reject stale writes under an OS lock; retry requires a fresh decision.

    force is retained for API compatibility, but cannot bypass a conflict.
    """
    from mott_state_coordination import atomic_write_json, stamp_state
    with state_lock(PORTFOY_FILE):
        disk = json.loads(PORTFOY_FILE.read_text(encoding="utf-8")) if PORTFOY_FILE.exists() else None
        expected = portfoy.get("_initial_gen", portfoy.get("_gen", 0))
        if disk is not None and disk.get("_gen", 0) != expected:
            raise StateConflict("P1 stale state: reload and recompute; no merge performed")
        if "_initial_digest" in portfoy and file_digest(PORTFOY_FILE) != portfoy["_initial_digest"]:
            raise StateConflict("P1 state bytes changed: reload and recompute")
        reconcile(portfoy)
        gunluk_equity_kaydet(portfoy)
        stamp_state(portfoy)
        clean = {k: v for k, v in portfoy.items() if k not in ("_initial_gen", "_initial_digest")}
        atomic_write_json(PORTFOY_FILE, clean)
        portfoy["_initial_gen"] = portfoy["_gen"]
        portfoy["_initial_digest"] = file_digest(PORTFOY_FILE)
    return portfoy


def tarama_listesi_yukle() -> dict:
    if TARAMA_FILE.exists():
        with open(TARAMA_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"signals": [], "scan_time": "", "scan_label": ""}
    if isinstance(data, list):
        return {"signals": [{"symbol": x, "score_count": 1, "strategies": [],
                             "scan_time": "", "scan_label": ""} for x in data]}
    if "signals" not in data and "semboller" in data:
        data["signals"] = [{"symbol": x, "score_count": 1, "strategies": [],
                            "scan_time": "", "scan_label": ""} for x in data["semboller"]]
    data.setdefault("signals", [])
    return data


def durum_kaydet(durum: dict):
    with open(DURUM_FILE, "w", encoding="utf-8") as fh:
        json.dump(durum, fh, indent=2, ensure_ascii=False)


def hisse_listesi_yukle() -> list:
    if not HISSELER_FILE.exists():
        return []
    try:
        return list(dict.fromkeys(
            line.strip().upper()
            for line in HISSELER_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ))
    except Exception as exc:
        log.warning("hisseler.txt okunamadi: %s", exc)
        return []


def hisse_listesi_kaydet(semboller: list) -> list:
    unique = list(dict.fromkeys(str(x).strip().upper() for x in semboller if str(x).strip()))
    HISSELER_FILE.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    return unique


def bekleyen_adayi_hazirla(aday: dict, ref_dt=None, valid_for_date=None) -> dict:
    ref_dt = ref_dt or datetime.now()
    valid_for = valid_for_date or sonraki_islem_gunu(ref_dt.date())
    hazir = dict(aday)
    hazir["queued_at"]           = ref_dt.strftime("%d.%m.%Y %H:%M")
    hazir["valid_for_date"]      = valid_for.isoformat()
    hazir["expires_at_hour"]     = WAITING_EXPIRES_HOUR
    hazir["attempt_count"]       = 0
    hazir["last_attempt_time"]   = ""
    hazir["last_attempt_reason"] = ""
    return hazir


def ayikla_suresi_dolan_bekleyenler(bekleyenler: list, now=None):
    now = now or datetime.now()
    active, expired = [], []
    current_date = now.date()
    current_hour = now.hour
    for item in bekleyenler:
        valid_for_str = item.get("valid_for_date")
        try:
            valid_for = date.fromisoformat(valid_for_str) if valid_for_str else current_date
        except Exception:
            valid_for = current_date
        expires_hour = item.get("expires_at_hour", WAITING_EXPIRES_HOUR)
        is_expired = (current_date > valid_for or
                      (current_date == valid_for and current_hour >= expires_hour))
        if is_expired:
            expired.append({"symbol": item.get("symbol", "?"), "reason": "expired"})
        else:
            active.append(item)
    return active, expired


def retry_bekleyenleri_filtrele(bekleyenler: list) -> list:
    return [b for b in bekleyenler
            if b.get("last_attempt_reason", "") in RETRY_REASONS
            or b.get("attempt_count", 0) == 0]


def _p1_acik_detay(portfoy: dict) -> list[dict]:
    detay = []
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f and pos.get("giris_f") else None
        detay.append({"symbol": sym, "pnl_pct": ret})
    return detay


def _p1_mott_portfoy(portfoy: dict) -> dict:
    prev = portfolio_preview(portfoy)
    return {
        "pozisyonlar":   portfoy.get("pozisyonlar", {}),
        "trade_history": portfoy.get("trade_history", []),
        "baslangic":     portfoy.get("baslangic", SERMAYE_BASLANGIC),
        "equity":        prev["equity"],
        "acik_detay":    _p1_acik_detay(portfoy),
    }


def _p1_telegram_islem(
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list | None = None,
) -> None:
    """Yalnızca alım/satım varsa P1 formatında Telegram gönder."""
    try:
        from mott_telegram import telegram_islem_gonder, yukle_p1_sinyaller
        telegram_islem_gonder(
            "P1",
            sinyaller=yukle_p1_sinyaller(),
            portfoy=_p1_mott_portfoy(portfoy),
            giris=giris or [],
            cikis=cikis or [],
            mesajlar=mesajlar,
        )
    except Exception as exc:
        log.warning("P1 Telegram hatasi: %s", exc)


def send_telegram(msg: str, parse_mode: str = "HTML"):
    from datetime import datetime
    import pytz
    _now = datetime.now(pytz.timezone("Europe/Istanbul"))
    _header = f"[PORTF\u00d6Y] {_now.strftime('%d.%m.%Y | %H:%M')}\n"
    msg = _header + msg
    if not BOT_TOKEN or not CHAT_ID:
        log.info("[TELEGRAM SIMULE]\n%s", msg)
        return
    # Gönderim mott_telegram üzerinden — retry + parçalama merkezî olarak orada
    try:
        from mott_telegram import telegram_gonder
        telegram_gonder(msg, parse_mode=parse_mode)
    except Exception as exc:
        log.error("Telegram hatasi: %s", exc)


def telegram_api_call(method: str, payload=None):
    if not BOT_TOKEN:
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/{method}",
                          json=payload or {}, timeout=20)
        return r.json() if r.ok else None
    except Exception:
        return None


def _authorized_chat(update: dict) -> bool:
    try:
        return str(update["message"]["chat"]["id"]) == str(CHAT_ID)
    except Exception:
        return False


def _telegram_offset_yukle() -> int:
    if TELEGRAM_OFFSET_FILE.exists():
        try:
            return int(TELEGRAM_OFFSET_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return 0


def _telegram_offset_kaydet(offset: int):
    TELEGRAM_OFFSET_FILE.write_text(str(int(offset)), encoding="utf-8")


def telegram_komut_isle(text: str) -> str:
    raw = (text or "").strip()
    cmd, _, arg = raw.partition(" ")
    symbol = arg.strip().upper()
    mevcut = hisse_listesi_yukle()
    if cmd == "/ekle":
        if not symbol:
            return "Kullanim: /ekle ASELS"
        if symbol in mevcut:
            return f"{symbol} zaten listede."
        mevcut.append(symbol)
        hisse_listesi_kaydet(mevcut)
        return f"\u2705 {symbol} eklendi.\nToplam: {len(mevcut)} hisse"
    if cmd == "/sil":
        if not symbol:
            return "Kullanim: /sil ASELS"
        if symbol not in mevcut:
            return f"{symbol} listede yok."
        hisse_listesi_kaydet([x for x in mevcut if x != symbol])
        return f"\U0001f5d1\ufe0f {symbol} silindi."
    if cmd == "/liste":
        if not mevcut:
            return "Liste bos."
        preview = ", ".join(mevcut[:40])
        extra = f"\n... +{len(mevcut)-40} hisse" if len(mevcut) > 40 else ""
        return f"\U0001f4cb {len(mevcut)} hisse:\n{preview}{extra}"
    if cmd == "/yardim":
        return "Komutlar:\n/ekle ASELS\n/sil ASELS\n/liste"
    return ""


def telegram_komutlarini_kontrol_et():
    if not BOT_TOKEN or not CHAT_ID:
        return
    offset = _telegram_offset_yukle()
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"offset": offset + 1, "timeout": 1, "allowed_updates": ["message"]},
            timeout=5,
        )
        if not r.ok:
            return
        for update in r.json().get("result", []):
            update_id = int(update.get("update_id", 0))
            offset = max(offset, update_id)
            if not _authorized_chat(update):
                continue
            text = (update.get("message", {}).get("text") or "").strip()
            if not text.startswith("/"):
                continue
            yanit = telegram_komut_isle(text)
            if yanit:
                send_telegram(yanit)
                append_jsonl(PORTFOY_AUDIT_FILE, {"event": "telegram_command", "command": text})
        _telegram_offset_kaydet(offset)
    except Exception as exc:
        log.debug("Telegram komut kontrol hatasi: %s", exc)


def makro_risk_skoru():
    global _MAKRO_CACHE
    now = datetime.now()
    # Cache geçerliyse direkt dön
    if (_MAKRO_CACHE["value"] is not None and _MAKRO_CACHE["fetched_at"] is not None
            and (now - _MAKRO_CACHE["fetched_at"]).total_seconds() < MAKRO_CACHE_TTL):
        log.debug("Makro skor cache'den dondu")
        return _MAKRO_CACHE["value"]

    detaylar = []
    piyasa_ret = {}
    for ticker, lag, yon, esik, agirlik in MAKRO_KURALLAR:
        try:
            df = _market_ticker(ticker).history(period="5d", interval="1d")
            if len(df) < 2:
                continue
            ret = float(df["Close"].pct_change().dropna().iloc[-(lag + 1)]) * 100
            piyasa_ret[ticker] = ret
            tetiklendi = ret < esik if yon == "down" else ret > esik
            detaylar.append({"endeks": ticker, "deger_pct": round(ret, 2),
                             "esik": esik, "agirlik": agirlik, "tetiklendi": tetiklendi})
        except Exception as exc:
            log.debug("Makro %s: %s", ticker, exc)

    sp_ret     = piyasa_ret.get("^GSPC",     0.0)
    nasdaq_ret = piyasa_ret.get("^IXIC",     0.0)
    vix_ret    = piyasa_ret.get("^VIX",      0.0)
    altin_ret  = piyasa_ret.get("GC=F",      0.0)
    abd_kotu   = sp_ret <= -2.0 or nasdaq_ret <= -3.0
    abd_orta   = -2.0 < sp_ret <= -1.0 or -3.0 < nasdaq_ret <= -1.5
    vix_sert   = vix_ret >= 3.0
    vix_hafif  = 2.0 <= vix_ret < 3.0
    altin_risk = altin_ret >= 2.0 and (sp_ret <= -1.0 or vix_ret >= 2.0)
    asya_kirmizi = sum(1 for t in ASYA_TICKERLAR if piyasa_ret.get(t, 0.0) <= ASYA_ESIK)

    if abd_kotu and vix_sert and asya_kirmizi >= ASYA_MIN_TETIK:
        karar = "GIRME"
    elif abd_orta or vix_hafif or asya_kirmizi >= 1 or altin_risk:
        karar = "DIKKATLI"
    else:
        karar = "NORMAL"

    toplam_skor = sum(d["agirlik"] for d in detaylar if d["tetiklendi"])
    if karar == "GIRME":
        toplam_skor = max(toplam_skor, 60.0)
    elif karar == "DIKKATLI":
        toplam_skor = max(toplam_skor, 30.0)

    sonuc = round(toplam_skor, 1), detaylar, karar, piyasa_ret
    _MAKRO_CACHE["value"]      = sonuc
    _MAKRO_CACHE["fetched_at"] = now
    return sonuc


def _makro_cache_temizle():
    """Sabah 09:00 akışında cache'i zorla yenile."""
    global _MAKRO_CACHE
    _MAKRO_CACHE = {"value": None, "fetched_at": None}


def makro_karar_olustur(
    skor: float,
    karar: str,
    detaylar: list,
    piyasa_ret: dict,
    kaynak: str = "sabah_09_akisi",
    now: datetime | None = None,
) -> dict:
    """Sabah veya gün içi değerlendirilen makro kararını standart metadata ile paketler."""
    ref_now = now if now is not None else datetime.now()
    run_id = os.environ.get("GITHUB_RUN_ID") or f"run_{ref_now.strftime('%Y%m%d_%H%M%S')}"
    return {
        "karar": karar,
        "skor": round(float(skor), 2),
        "zaman": ref_now.isoformat(),
        "gecerlilik_tarih": ref_now.strftime("%Y-%m-%d"),
        "kaynak": kaynak,
        "run_id": str(run_id),
        "piyasa_ret": {k: round(float(v), 2) for k, v in piyasa_ret.items()} if piyasa_ret else {},
        "detay_sayisi": len(detaylar) if detaylar else 0,
        "valid": True,
    }


def get_aktif_makro_karar(
    portfoy: dict | None = None,
    now: datetime | None = None,
) -> tuple[str, float, dict]:
    """Aktif makro kararını, skorunu ve doğrulama metadata'sını döndürür.

    Doğrulama kuralları:
      - Karar 'NORMAL', 'DIKKATLI' veya 'GIRME' olmalıdır.
      - gecerlilik_tarih bugünün tarihi ile eşleşmelidir.
      - Karar zamanı son 14 saat içinde olmalıdır.
      - Eksik, bozuk, parse edilemeyen veya süresi geçmiş kararda:
        -> 'GIRME' döndürülür, valid=False olarak işaretlenir.
        -> Yeni alım engellenir, mevcut pozisyon risk kontrolleri devam eder.
    """
    ref_now = now if now is not None else datetime.now()
    bugun_str = ref_now.strftime("%Y-%m-%d")

    data = None
    if portfoy and isinstance(portfoy.get("makro_karar"), dict):
        data = portfoy["makro_karar"]
    elif PORTFOY_FILE.exists():
        try:
            with open(PORTFOY_FILE, encoding="utf-8") as fh:
                pj = json.load(fh)
                if isinstance(pj.get("makro_karar"), dict):
                    data = pj["makro_karar"]
        except Exception:
            pass

    if data is None and STATE_P1_FILE.exists():
        try:
            with open(STATE_P1_FILE, encoding="utf-8") as fh:
                sp = json.load(fh)
                if isinstance(sp.get("makro_karar"), dict):
                    data = sp["makro_karar"]
        except Exception:
            pass

    if data is None and DURUM_FILE.exists():
        try:
            with open(DURUM_FILE, encoding="utf-8") as fh:
                df = json.load(fh)
                if df.get("makro_karar"):
                    data = {
                        "karar": df.get("makro_karar"),
                        "skor": df.get("makro_skor", 0.0),
                        "zaman": df.get("tarih", ""),
                        "gecerlilik_tarih": df.get("tarih", "")[:10] if df.get("tarih") else "",
                        "kaynak": "son_durum_fallback",
                        "run_id": "legacy_son_durum",
                    }
        except Exception:
            pass

    if not data:
        meta = {"valid": False, "reason": "makro_karar_yok", "karar": "GIRME", "skor": 0.0}
        return "GIRME", 0.0, meta

    karar = str(data.get("karar", "")).upper()
    if karar not in ("NORMAL", "DIKKATLI", "GIRME"):
        meta = {"valid": False, "reason": "bozuk_karar_degeri", "raw": data, "karar": "GIRME", "skor": 0.0}
        return "GIRME", 0.0, meta

    gecerlilik = str(data.get("gecerlilik_tarih", ""))
    zaman_raw = str(data.get("zaman", ""))
    try:
        skor = float(data.get("skor", 0.0) or 0.0)
        if not np.isfinite(skor):
            raise ValueError("nonfinite macro score")
    except (TypeError, ValueError):
        return "GIRME", 0.0, {"valid": False, "reason": "bozuk_makro_skor"}

    tarih_uyusuyor = False
    if gecerlilik == bugun_str:
        tarih_uyusuyor = True
    elif zaman_raw:
        z_dt = _parse_datetime(zaman_raw)
        if z_dt and z_dt.date() == ref_now.date():
            tarih_uyusuyor = True

    if not tarih_uyusuyor:
        meta = {
            "valid": False,
            "reason": "suresi_gecmis_karar",
            "gecerlilik_tarih": gecerlilik,
            "bugun": bugun_str,
            "karar": "GIRME",
            "skor": skor,
            "run_id": data.get("run_id", ""),
        }
        return "GIRME", 0.0, meta

    z_dt = _parse_datetime(zaman_raw)
    if z_dt is None:
        return "GIRME", 0.0, {"valid": False, "reason": "makro_zamani_gecersiz"}
    if z_dt:
        diff = _p1_now(ref_now) - to_tsi(z_dt)
        if diff.total_seconds() < -300:
            meta = {"valid": False, "reason": "gelecek_tarihli_karar", "karar": "GIRME", "skor": skor}
            return "GIRME", 0.0, meta
        if diff.total_seconds() > 14 * 3600:
            meta = {"valid": False, "reason": "karar_14h_asildi", "karar": "GIRME", "skor": skor}
            return "GIRME", 0.0, meta

    meta = dict(data)
    meta["valid"] = True
    return karar, skor, meta


def viop_bias_hesapla() -> dict:
    """
    BIST yön tahmini için proxy zinciri:
      1. VIOP_TICKER (.env override — kullanıcı tanımlı)
      2. ^XU100      (Yahoo Finance BIST100 endeks sembolü)
      3. XU100.IS    (alternatif format)
      4. XU030.IS    (BIST30 ETF)
      5. basket      (10 büyük hisse ortalaması — son çare)
    Her adımda len(df) >= 2 ve NaN kontrolü yapılır.
    """
    def _ret_from_ticker(sym: str):
        try:
            df = _ticker(sym).history(period="5d", interval="1d")
            df = df.dropna(subset=["Close"])
            if len(df) < 2:
                return None
            ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            if not np.isfinite(ret):
                return None
            return ret
        except Exception as exc:
            log.debug("VIOP proxy %s: %s", sym, exc)
            return None

    def _bias_from_ret(ret: float, ticker: str, source: str) -> dict:
        if ret >= 1.5:
            return {"label": "Long bias g\u00fc\u00e7l\u00fc", "score": 5, "size_factor": 1.15,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        if ret >= 0.5:
            return {"label": "Normal", "score": 2, "size_factor": 1.05,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        if ret <= -1.5:
            return {"label": "K\u00fc\u00e7\u00fck boyut / nakit a\u011f\u0131rl\u0131k", "score": -5, "size_factor": 0.55,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        if ret <= -0.5:
            return {"label": "Temkinli long", "score": -2, "size_factor": 0.80,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        return {"label": "Normal", "score": 0, "size_factor": 1.0,
                "ret": round(ret, 2), "ticker": ticker, "source": source}

    # Proxy zinciri — sırayla dene
    proxy_zinciri = []
    # 1. Kullanıcı override (.env) — boş veya varsayılan değilse ekle
    if VIOP_TICKER and VIOP_TICKER not in ("XU100.IS", "XU030.IS"):
        proxy_zinciri.append((VIOP_TICKER, "env_override"))
    # 2-4. Sabit fallback'ler
    proxy_zinciri += [
        ("^XU100",   "bist100_index"),
        ("XU100.IS", "bist100_etf"),
        ("XU030.IS", "bist30_etf"),
    ]

    for ticker, source in proxy_zinciri:
        ret = _ret_from_ticker(ticker)
        if ret is not None:
            log.debug("VIOP proxy basarili: %s (%s) ret=%.2f", ticker, source, ret)
            return _bias_from_ret(ret, ticker, source)

    log.warning("VIOP proxy zinciri basarisiz, basket'e geciliyor")

    # 5. Basket proxy — son çare
    basket = ["AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS", "KCHOL.IS",
              "PGSUS.IS", "SAHOL.IS", "SISE.IS",  "THYAO.IS", "TUPRS.IS"]
    rets = []
    for ticker in basket:
        ret = _ret_from_ticker(ticker)
        if ret is not None:
            rets.append(ret)

    if rets:
        ret = float(np.mean(rets))
        log.info("VIOP basket proxy: %d/%d hisse, ret=%.2f", len(rets), len(basket), ret)
        if ret >= 1.2:
            return {"label": "Long bias g\u00fc\u00e7l\u00fc", "score": 4, "size_factor": 1.10,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        if ret >= 0.3:
            return {"label": "Normal", "score": 1, "size_factor": 1.03,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        if ret <= -1.2:
            return {"label": "K\u00fc\u00e7\u00fck boyut / nakit a\u011f\u0131rl\u0131k", "score": -4, "size_factor": 0.65,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        if ret <= -0.3:
            return {"label": "Temkinli long", "score": -1, "size_factor": 0.85,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        return {"label": "Normal", "score": 0, "size_factor": 1.0,
                "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}

    log.error("VIOP: tum proxy'ler basarisiz, Veri yok donuyor")
    return {"label": "Veri yok", "score": 0, "size_factor": 1.0,
            "ret": None, "ticker": None, "source": "none"}


def ema_calc(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_angle(fast_ma: pd.Series, lookback: int = 3) -> pd.Series:
    delta = (fast_ma / fast_ma.shift(lookback) - 1) * 100
    return np.degrees(np.arctan(delta / lookback))


def calc_distance(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    return (slow_ma - fast_ma) / (slow_ma + 1e-9) * 100


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def ma_motor_skoru(symbol: str):
    try:
        df = _ticker(f"{symbol}.IS").history(period="6mo", interval="1d")
        df = closed_daily(df, _p1_now())
        if len(df) < 60:
            return None
        c, v  = df["Close"], df["Volume"]
        vol_ratio = v / (v.rolling(20).mean() + 1e-9)
        ema50 = ema_calc(c, 50)
        scores = {"MotorA": 0.0, "MotorB": 0.0}
        for motor_ad, fast, slow, a_min, a_max, pre_cross in [
            ("MotorA", 12, 26, 20, 30, False),
            ("MotorB",  8, 21, 30, 45, True),
        ]:
            fast_ma = ema_calc(c, fast)
            slow_ma = ema_calc(c, slow)
            angle   = calc_angle(fast_ma)
            dist    = calc_distance(fast_ma, slow_ma)
            speed   = dist.shift(1) - dist
            above   = (fast_ma > slow_ma).astype(int)
            crossover = ((above == 1) & (above.shift(1) == 0)).astype(int)
            has_signal = (bool(crossover.iloc[-1]) if not pre_cross
                         else float(dist.iloc[-1]) > 0 and float(dist.iloc[-1]) <= 1.0 and float(speed.iloc[-1]) > 0)
            if not has_signal:
                continue
            aci = float(angle.iloc[-1])
            vol_r = float(vol_ratio.iloc[-1])
            if a_min <= aci < a_max and vol_r >= 1.5 and float(c.iloc[-1]) > float(ema50.iloc[-1]):
                scores[motor_ad] = round(min(aci, 90) * min(vol_r, 5), 2)
        toplam = scores["MotorA"] + scores["MotorB"]
        if scores["MotorA"] > 0 and scores["MotorB"] > 0:
            toplam *= 1.5
        scores["toplam"] = round(toplam, 2)
        return scores
    except Exception as exc:
        log.debug("MA motor %s: %s", symbol, exc)
        return None


def lgbm_skor_hesapla(symbol: str, model):
    if model is None:
        return None
    try:
        df = _ticker(f"{symbol}.IS").history(period="1y", interval="1d")
        df = closed_daily(df, _p1_now())
        if len(df) < 60:
            return None
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        feats = {}
        for p in [5,8,9,12,13,20,21,26,34,50,100,200]:
            feats[f"ema{p}_ratio"] = float((c / (ema_calc(c, p) + 1e-9)).iloc[-1])
        e9,e21,e12,e26,e50,e200 = (ema_calc(c,p) for p in [9,21,12,26,50,200])
        feats.update({
            "ema9_gt_21": int(e9.iloc[-1]>e21.iloc[-1]),
            "ema12_gt_26": int(e12.iloc[-1]>e26.iloc[-1]),
            "ema21_gt_50": int(e21.iloc[-1]>e50.iloc[-1]),
            "ema50_gt_200": int(e50.iloc[-1]>e200.iloc[-1]),
        })
        for p,lb in [(9,3),(12,3),(21,5),(50,10)]:
            feats[f"angle{p}"] = float(calc_angle(ema_calc(c,p),lb).iloc[-1])
        def _dist(fp,sp):
            return (ema_calc(c,sp)-ema_calc(c,fp))/(ema_calc(c,sp)+1e-9)*100
        feats.update({
            "dist_9_21": float(_dist(9,21).iloc[-1]),
            "dist_12_26": float(_dist(12,26).iloc[-1]),
            "dist_8_21": float(_dist(8,21).iloc[-1]),
            "speed_9_21": float(_dist(9,21).diff().iloc[-1]),
            "speed_12_26": float(_dist(12,26).diff().iloc[-1]),
            "rsi14": float(_rsi(c,14).iloc[-1]),
            "rsi7":  float(_rsi(c,7).iloc[-1]),
        })
        ml = ema_calc(c,12)-ema_calc(c,26); ms = ema_calc(ml,9)
        feats.update({
            "macd": float(ml.iloc[-1]), "macd_sig": float(ms.iloc[-1]),
            "macd_hist": float((ml-ms).iloc[-1]),
            "macd_gt_0": int(ml.iloc[-1]>0), "macd_gt_sig": int(ml.iloc[-1]>ms.iloc[-1]),
        })
        mid=c.rolling(20).mean(); std=c.rolling(20).std()
        bb_l=mid-2*std; bb_u=mid+2*std
        feats.update({
            "bb_pct": float(((c-bb_l)/(bb_u-bb_l+1e-9)).iloc[-1]),
            "bb_width": float(((bb_u-bb_l)/(mid+1e-9)).iloc[-1]),
            "bb_gt_mid": int(c.iloc[-1]>mid.iloc[-1]),
        })
        vm10,vm20 = v.rolling(10).mean(),v.rolling(20).mean()
        feats.update({
            "rel_vol_10": float((v/(vm10+1e-9)).iloc[-1]),
            "rel_vol_20": float((v/(vm20+1e-9)).iloc[-1]),
            "vol_trend":  float((vm10/(vm20+1e-9)).iloc[-1]),
        })
        tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        a14=tr.rolling(14).mean()
        feats.update({
            "atr_pct": float((a14/(c+1e-9)*100).iloc[-1]),
            "atr_ratio": float((a14/(a14.rolling(20).mean()+1e-9)).iloc[-1]),
            "ret1d": float(c.pct_change(1).iloc[-1]*100),
            "ret3d": float(c.pct_change(3).iloc[-1]*100),
            "ret5d": float(c.pct_change(5).iloc[-1]*100),
            "ret10d": float(c.pct_change(10).iloc[-1]*100),
        })
        mfv=((c-l)-(h-c))/(h-l+1e-9)*v
        feats["cmf20"] = float((mfv.rolling(20).sum()/(v.rolling(20).sum()+1e-9)).iloc[-1])
        r14=_rsi(c,14); lo=r14.rolling(14).min(); hi=r14.rolling(14).max()
        feats["stochrsi"] = float((100*(r14-lo)/(hi-lo+1e-9)).rolling(3).mean().iloc[-1])
        pivot=(h.shift(1)+l.shift(1)+c.shift(1))/3
        feats.update({
            "price_gt_fibr1": int(c.iloc[-1]>(pivot+0.382*(h.shift(1)-l.shift(1))).iloc[-1]),
            "price_gt_pivot": int(c.iloc[-1]>pivot.iloc[-1]),
            "motorA_sinyal": int(e12.iloc[-1]>e26.iloc[-1]),
            "motorB_sinyal": int(e9.iloc[-1]>e21.iloc[-1]),
        })
        feat_cols = (json.loads(LGBM_META_FILE.read_text(encoding="utf-8")).get("feature_cols")
                     if LGBM_META_FILE.exists() else None)
        X = (pd.DataFrame([[feats[col] for col in feat_cols]],columns=feat_cols)
             if feat_cols else pd.DataFrame([feats]))
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("LGBM features are incomplete")
        classes = list(getattr(model, "classes_", []))
        if 1 not in classes:
            raise ValueError("LGBM positive class 1 is not declared")
        prob = float(model.predict_proba(X)[0][classes.index(1)])
        if not np.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("LGBM prediction outside probability range")
        return round(prob*100, 1)
    except Exception as exc:
        log.debug("LGBM %s: %s", symbol, exc)
        return None


def alpha_trend_analiz(symbol: str) -> dict:
    try:
        df = _ticker(f"{symbol}.IS").history(period="8mo", interval="1d")
        df = closed_daily(df, _p1_now())
        if len(df) < 60:
            return {"state": "none", "flip_up": False, "bonus": 0, "tag": ""}
        c,h,l,v = df["Close"],df["High"],df["Low"],df["Volume"]
        ap,coeff = 14,1.0
        atr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1).rolling(ap).mean()
        tp = (h+l+c)/3.0; rmf = tp*v
        pos_mf = rmf.where(tp>tp.shift(1),0.0); neg_mf = rmf.where(tp<tp.shift(1),0.0)
        mfi = 100-(100/(1+pos_mf.rolling(ap).sum()/(neg_mf.rolling(ap).sum()+1e-9)))
        up_t,dn_t = l-atr*coeff, h+atr*coeff
        alpha = pd.Series(index=df.index, dtype=float)
        alpha.iloc[0] = c.iloc[0]
        for i in range(1, len(df)):
            prev = alpha.iloc[i-1]
            alpha.iloc[i] = max(up_t.iloc[i],prev) if mfi.iloc[i]>=50 else min(dn_t.iloc[i],prev)
        bullish = bool(alpha.iloc[-1]>alpha.iloc[-3])
        prev_bullish = bool(alpha.iloc[-2]>alpha.iloc[-4]) if len(alpha)>=4 else False
        flip_up = bullish and not prev_bullish
        ema21,ema50 = ema_calc(c,21),ema_calc(c,50)
        above = bool(c.iloc[-1]>ema21.iloc[-1] and ema21.iloc[-1]>=ema50.iloc[-1])
        if flip_up and above:
            return {"state":"bullish","flip_up":True,"bonus":4,"tag":"AT\u2191"}
        if bullish:
            return {"state":"bullish","flip_up":False,"bonus":2,"tag":"AT+"}
        return {"state":"bearish","flip_up":False,"bonus":0,"tag":""}
    except Exception as exc:
        log.debug("Alpha Trend %s: %s", symbol, exc)
        return {"state":"none","flip_up":False,"bonus":0,"tag":""}


def _strategy_combo_score(strategies: list) -> float:
    return sum(STRATEGY_WEIGHTS.get(x, 0) for x in dict.fromkeys(strategies))


def final_signal_score(signal: dict, ma_score, lgbm_score, viop_bias: dict, alpha_bonus: float) -> float:
    base = _strategy_combo_score(signal.get("strategies", []))
    base += min(signal.get("score_count", 1) * 6, 20)
    if ma_score:
        base += min(ma_score / 10, 20)
    if lgbm_score is not None:
        base += max((lgbm_score - 50) / 3, -10)
    base += viop_bias.get("score", 0)
    base += min(alpha_bonus, 4)
    rsi = signal.get("rsi", 50)
    rel_vol = signal.get("rel_vol", 1)
    strategies = signal.get("strategies", [])
    is_dip_only = len(strategies) >= 1 and all(s == "DIP" for s in strategies)
    if is_dip_only:
        if rsi >= 75:
            base -= 3
    else:
        if rsi >= 82:
            base -= 6
        elif rsi >= 75:
            base -= 3
    if rel_vol < 1:
        base -= 4
    if signal.get("islem_tl", 0) < 10_000_000:
        base -= 4
    return round(max(0, min(base, 100)), 2)


def build_candidate(symbol: str, signal: dict, model, viop_bias: dict) -> dict:
    ma_result  = ma_motor_skoru(symbol)
    ma_total   = ma_result["toplam"] if ma_result else 0.0
    lgbm_score = lgbm_skor_hesapla(symbol, model) if model else None
    alpha      = alpha_trend_analiz(symbol)
    return {
        "symbol":            symbol,
        "score_count":       signal.get("score_count", 1),
        "strategies":        signal.get("strategies", []),
        "tarama_score":      _strategy_combo_score(signal.get("strategies", [])),
        "ma_score":          round(ma_total, 2),
        "agent_status": {"ma": "ok" if ma_result else "data_missing", "lgbm": "scored" if lgbm_score is not None else ("model_missing" if model is None else "invalid_prediction"), "market": viop_bias.get("source", "none"), "alpha": alpha["state"]},
        "lgbm_score":        lgbm_score,
        "viop_bias":         viop_bias["label"],
        "viop_score":        viop_bias["score"],
        "alpha_trend_state": alpha["state"],
        "alpha_trend_flip":  alpha["flip_up"],
        "alpha_trend_bonus": alpha["bonus"],
        "alpha_tag":         alpha["tag"],
        "final_score":       final_signal_score(signal, ma_total, lgbm_score, viop_bias, alpha["bonus"]),
        "change_pct":        signal.get("change_pct", 0),
        "rsi":               signal.get("rsi", 0),
        "rel_vol":           signal.get("rel_vol", 0),
        "cmf":               signal.get("cmf", 0),
        "islem_tl":          signal.get("islem_tl", 0),
        "scan_time":         signal.get("scan_time", ""),
        "scan_label":        signal.get("scan_label", ""),
    }


def tarama_calistir(saat_label: str, makro_karar: str, viop_bias: dict):
    model, model_status = lgbm_model_yukle()
    signals  = tarama_listesi_yukle().get("signals", [])
    adaylar, elinenler = [], []
    for signal in signals:
        aday = build_candidate(signal["symbol"], signal, model, viop_bias)
        if len(aday.get("strategies", [])) == 1 and "DIP" in aday.get("strategies", []):
            append_jsonl(SIGNAL_AUDIT_FILE, {**aday, "event": "dip_info_only"})
            continue
        if aday["lgbm_score"] is not None and aday["lgbm_score"] < LGBM_MIN_SKOR:
            elinenler.append({"symbol": signal["symbol"], "reason": "score_low"})
            continue
        adaylar.append(aday)
    adaylar.sort(key=lambda x: (-x["final_score"], x["symbol"]))
    for aday in adaylar:
        append_jsonl(SIGNAL_AUDIT_FILE, {**aday, "event": "candidate"})
    return adaylar, model_status, elinenler


def portfolio_preview(portfoy: dict) -> dict:
    nakit = portfoy.get("nakit", 0)
    equity = nakit
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        equity += pos["lotlar"] * (f if f else pos["giris_f"])
    return {"cash": round(nakit, 2), "equity": round(equity, 2),
            "n_positions": len(portfoy.get("pozisyonlar", {}))}


def yeni_pozisyon_ac(portfoy: dict, adaylar: list, makro_karar: str, viop_bias: dict, makro_meta: dict | None = None, now: datetime | None = None):
    ref_now = _p1_now(now)
    if not is_bist_seans_acik(ref_now):
        return portfoy, [], [], [{"symbol": a["symbol"], "reason": "seans_kapali"} for a in adaylar]
    start_ledger(portfoy, ref_now.isoformat())
    if makro_karar == "GIRME":
        sub_reason = "makro_gecersiz" if (makro_meta and not makro_meta.get("valid", True)) else "makro_girme"
        return portfoy, [], [], [{"symbol": a["symbol"], "reason": sub_reason} for a in adaylar]
    mesajlar, alinan, alinmayan = [], [], []
    mevcut   = portfoy["pozisyonlar"]
    nakit    = portfoy["nakit"]
    bos_slot = max(0, MAX_HISSE - len(mevcut))
    rejim_limit = mott_risk.rejim_slot_limiti(makro_karar)
    if rejim_limit is not None:
        bos_slot = min(bos_slot, rejim_limit)
    if bos_slot == 0 or not adaylar:
        return portfoy, [], [], []
    secilenler  = [a for a in adaylar if a["final_score"] >= 30][:bos_slot]
    toplam_skor = sum(max(a["final_score"], 1) for a in secilenler) or 1
    size_factor = viop_bias.get("size_factor", 1.0) * (0.6 if makro_karar == "DIKKATLI" else 1.0)
    for aday in secilenler:
        sym = aday["symbol"]
        if sym in mevcut:
            alinmayan.append({"symbol": sym, "reason": "already_open"})
            continue
        if mott_risk.cooldown_da(portfoy.get("trade_history"), sym):
            alinmayan.append({"symbol": sym, "reason": "cooldown"})
            append_jsonl(PORTFOY_AUDIT_FILE, {"event": "buy_failed", "symbol": sym, "reason": "cooldown"})
            continue
        if mott_risk.kitap_limiti_asildi(sym, haric="P1"):
            alinmayan.append({"symbol": sym, "reason": "kitap_limiti"})
            append_jsonl(PORTFOY_AUDIT_FILE, {"event": "buy_failed", "symbol": sym, "reason": "kitap_limiti"})
            continue
        try:
            f_detay = guncel_fiyat_detayli(sym, now=ref_now)
            giris_f = f_detay["price"] if f_detay.get("trade_eligible", False) else None
            if giris_f is None or giris_f <= 0:
                alinmayan.append({"symbol": sym, "reason": "veri_yok"})
                append_jsonl(PORTFOY_AUDIT_FILE, {
                    "event": "buy_failed",
                    "symbol": sym,
                    "reason": "veri_yok",
                    "fiyat_meta": f_detay,
                })
                continue
            alloc  = min(HISSE_LIMIT, nakit * size_factor * (aday["final_score"] / toplam_skor))
            lotlar = int(alloc / max(giris_f, 0.01))
            if lotlar < 1:
                alinmayan.append({"symbol": sym, "reason": "lot_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":"lot_yetersiz"})
                continue
            maliyet = lotlar * giris_f
            net_alis_tutari, buy_komisyon = hesapla_net_tutar(lotlar, giris_f, islem="alis")
            if net_alis_tutari > nakit:
                alinmayan.append({"symbol": sym, "reason": "nakit_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":"nakit_yetersiz"})
                continue
            nakit -= net_alis_tutari
            now_dt = ref_now
            iso_now = now_dt.isoformat()
            pos_id = f"P1_{sym}_" + uuid.uuid4().hex
            buy_evt_id = "EVT_BUY_" + pos_id
            mevcut[sym] = {
                "position_id": pos_id,
                "entry_event_id": buy_evt_id,
                "entry_lotlar": lotlar, "entry_komisyon": buy_komisyon,
                "giris_f": round(giris_f, 4), "giris_t": now_dt.strftime("%d.%m.%Y %H:%M"),
                "fiyat_kaynak": f_detay.get("source", "unknown"),
                "fiyat_zaman": f_detay.get("time", ""),
                "tepe_f": round(giris_f, 4), "lotlar": lotlar, "gun": 0, "tp1_yapildi": False,
                "max_gun_date": (ref_now.date() + timedelta(days=MAX_GUN)).strftime("%d.%m.%Y"),
                "final_score": aday.get("final_score", 0),
                "ma_score": aday.get("ma_score", 0),
                "lgbm_score": aday.get("lgbm_score", None),
                "source_signal": {"score_count": aday.get("score_count", 0), "strategies": aday.get("strategies", []),
                                  "viop_bias": aday.get("viop_bias", "NEUTRAL"), "alpha_tag": aday.get("alpha_tag",""),
                                  "alpha_trend_bonus": aday.get("alpha_trend_bonus",0)},
            }
            portfoy.setdefault("islem_defteri", []).append({
                "event_id":     buy_evt_id,
                "position_id":  pos_id,
                "symbol":       sym,
                "islem_tipi":   "ALIS",
                "lot":          lotlar,
                "fiyat":        round(giris_f, 4),
                "brut_tutar":   round(maliyet, 4),
                "komisyon":     buy_komisyon,
                "nakit_etkisi": -round(net_alis_tutari, 4),
                "zaman":        iso_now,
            })
            mesajlar.append(
                f"\U0001f6a8 <b>AL - {sym}</b>\n"
                f"   {lotlar} lot @ {giris_f:.2f} TL\n"
                f"   Skor: {aday['final_score']:.1f} {aday.get('alpha_tag','')} | V\u0130OP: {aday.get('viop_bias', '')}"
            )
            append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_success","symbol":sym,"price":giris_f,
                                               "lots":lotlar,"final_score":aday["final_score"]})
            alinan.append(sym)
        except Exception as exc:
            log.exception("Yeni pozisyon HATA %s", sym)
            alinmayan.append({"symbol": sym, "reason": "exception"})
            append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":f"exception:{exc}"})
    portfoy["nakit"] = nakit
    return portfoy, mesajlar, alinan, alinmayan


def pozisyon_guncelle_saatlik(portfoy: dict, makro_karar: str, makro_skor: float | None = None, now: datetime | None = None):
    mesajlar, kapatilacak = [], []
    ref_now = _p1_now(now)
    ref_date = ref_now.date()
    start_ledger(portfoy, ref_now.isoformat())
    for sym, pos in list(portfoy["pozisyonlar"].items()):
        try:
            bar = saatlik_bar(sym, now=ref_now)
            if bar is None:
                continue
            high, low, close = bar["high"], bar["low"], bar["close"]
            pos["decision_bar_time"] = str(bar.get("bar_time", ref_now.isoformat()))
            giris_f = pos["giris_f"]
            lotlar  = pos["lotlar"]

            # Aynı pozisyon ve mum için tekrar çalıştırma kontrolü (mum idempotency)
            bar_time = str(bar.get("bar_time", ""))
            already_evaluated_bar = bool(bar_time) and (pos.get("last_bar_time") == bar_time)

            if not already_evaluated_bar:
                prev_tepe_f = float(pos.get("tepe_f", giris_f))
                stop_seviyesi = round(giris_f * (1 + STOP_PCT), 4)
                tp1_seviyesi = round(giris_f * (1 + TP1_PCT), 4)
                is_stop = low <= stop_seviyesi
                is_tp1 = (high >= tp1_seviyesi) and not pos.get("tp1_yapildi")
                likidite_teyitli = bool(bar.get("volume", 0) > 0)

                # Belirsizlik: STOP ve TP1 aynı mumda görüldüyse muhafazakâr stop önceliği
                if is_stop and is_tp1:
                    ambiguity = "both_stop_and_tp_in_bar"
                    if bar["open"] <= stop_seviyesi:
                        cikis_f = round(bar["open"] * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                    else:
                        cikis_f = round(stop_seviyesi * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                    net_tutar, komisyon = hesapla_net_tutar(lotlar, cikis_f, islem="satis")
                    portfoy["nakit"] += net_tutar
                    _trade_kaydet(portfoy, sym, pos, cikis_f, "STOP",
                                  ambiguity=ambiguity, likidite_teyitli=likidite_teyitli, now=ref_now)
                    kapatilacak.append(sym)
                    ret_g = (cikis_f - giris_f) / giris_f
                    mesajlar.append(
                        f"\U0001f6d1 <b>STOP (Belirsiz Mum) - {sym}</b>\n"
                        f"   Giriş: {giris_f:.2f} \u2192 Çıkış: {cikis_f:.2f} ({ret_g*100:+.1f}%) [Aynı barda TP1 ve STOP görüldü]"
                    )
                    append_jsonl(PORTFOY_AUDIT_FILE, {
                        "event": "stop",
                        "symbol": sym,
                        "exit_price": cikis_f,
                        "return_pct": ret_g * 100,
                        "ambiguity": ambiguity,
                        "likidite_teyitli": likidite_teyitli,
                    })
                    pos["last_bar_time"] = bar_time
                    continue

                # Bağımsız STOP kontrolü
                if is_stop:
                    if bar["open"] <= stop_seviyesi:
                        # Gap down: açılış stop seviyesinin altında
                        cikis_f = round(bar["open"] * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                    else:
                        cikis_f = round(stop_seviyesi * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                    net_tutar, komisyon = hesapla_net_tutar(lotlar, cikis_f, islem="satis")
                    portfoy["nakit"] += net_tutar
                    _trade_kaydet(portfoy, sym, pos, cikis_f, "STOP", likidite_teyitli=likidite_teyitli, now=ref_now)
                    kapatilacak.append(sym)
                    ret_g = (cikis_f - giris_f) / giris_f
                    mesajlar.append(f"\U0001f6d1 <b>STOP - {sym}</b>\n   Giriş: {giris_f:.2f} \u2192 Çıkış: {cikis_f:.2f} ({ret_g*100:+.1f}%)")
                    append_jsonl(PORTFOY_AUDIT_FILE, {
                        "event": "stop",
                        "symbol": sym,
                        "exit_price": cikis_f,
                        "return_pct": ret_g * 100,
                        "gap_down": bool(bar["open"] <= stop_seviyesi),
                        "likidite_teyitli": likidite_teyitli,
                    })
                    pos["last_bar_time"] = bar_time
                    continue

                # TP1 kontrolü
                if is_tp1:
                    if bar["open"] >= tp1_seviyesi:
                        tp1_cikis_f = round(bar["open"] * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                    else:
                        tp1_cikis_f = round(tp1_seviyesi * (1 - VARSAYILAN_KAYMA_ORANI), 4)

                    realized_pnl_pct = round((tp1_cikis_f - giris_f) / giris_f * 100, 2)

                    if lotlar <= 1:
                        yari = 1
                        pos["lotlar"] = 0
                        kapatilacak.append(sym)
                    else:
                        yari = lotlar // 2
                        pos["lotlar"] -= yari

                    pos["tp1_yapildi"] = True
                    net_tutar, komisyon = hesapla_net_tutar(yari, tp1_cikis_f, islem="satis")
                    portfoy["nakit"] += net_tutar
                    _trade_kaydet(portfoy, sym, pos, tp1_cikis_f, "TP1", lotlar=yari,
                                  tp1_tetik_fiyat=tp1_seviyesi, likidite_teyitli=likidite_teyitli, now=ref_now)
                    realized_pnl_pct = portfoy["trade_history"][-1]["net_pnl_pct"]

                    mesajlar.append(
                        f"\U0001f3af <b>TP1 - {sym}</b>\n"
                        f"   {yari} lot @ {tp1_cikis_f:.2f} satıldı (net {realized_pnl_pct:+.2f}%) [Tetik: {tp1_seviyesi:.2f}]"
                    )
                    append_jsonl(PORTFOY_AUDIT_FILE, {
                        "event": "tp1",
                        "symbol": sym,
                        "tp1_trigger_price": tp1_seviyesi,
                        "simulated_fill_price": tp1_cikis_f,
                        "realized_return_pct": realized_pnl_pct,
                        "sold_lots": yari,
                        "remaining_lots": pos["lotlar"],
                        "likidite_teyitli": likidite_teyitli,
                    })
                    if pos["lotlar"] == 0:
                        pos["last_bar_time"] = bar_time
                        continue

                # TRAILING kontrolü (Önceki mumlardan bilinen prev_tepe_f kullanılır)
                elif pos.get("tp1_yapildi"):
                    trail_ret = (low - prev_tepe_f) / prev_tepe_f
                    if trail_ret <= TRAILING_PCT:
                        trail_seviyesi = round(prev_tepe_f * (1 + TRAILING_PCT), 4)
                        if bar["open"] <= trail_seviyesi:
                            cikis_f = round(bar["open"] * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                        else:
                            cikis_f = round(trail_seviyesi * (1 - VARSAYILAN_KAYMA_ORANI), 4)
                        net_tutar, komisyon = hesapla_net_tutar(pos["lotlar"], cikis_f, islem="satis")
                        portfoy["nakit"] += net_tutar
                        _trade_kaydet(portfoy, sym, pos, cikis_f, "TRAILING", likidite_teyitli=likidite_teyitli, now=ref_now)
                        kapatilacak.append(sym)
                        ret_g = (cikis_f - giris_f) / giris_f
                        mesajlar.append(f"\U0001f4c9 <b>TRAILING - {sym}</b>\n   Çıkış: {cikis_f:.2f} | Getiri: {ret_g*100:+.1f}%")
                        append_jsonl(PORTFOY_AUDIT_FILE, {
                            "event": "trailing",
                            "symbol": sym,
                            "exit_price": cikis_f,
                            "return_pct": ret_g * 100,
                            "likidite_teyitli": likidite_teyitli,
                        })
                        pos["last_bar_time"] = bar_time
                        continue

                pos["tepe_f"] = max(prev_tepe_f, high)
                pos["last_bar_time"] = bar_time

            # MAX GUN (Takvim günü bazlı kontrol)
            if _elde_tutma_gunu(pos.get("giris_t", ""), now_date=ref_date) >= MAX_GUN:
                # Rolling extension: MAX_GUN gününde P1 ALIM listesi kontrolü
                mgd_str = pos.get("max_gun_date") or _max_gun_date_hesapla_p1(pos.get("giris_t", ""), now_date=ref_date)
                mgd = _parse_tarih(mgd_str) or ref_date
                if ref_date >= mgd:
                    alim, status, meta = _p1_alim_listesi_durum(now=ref_now)
                    if status == "valid" and sym in alim:
                        pos["max_gun_date"] = (ref_date + timedelta(days=MAX_GUN_EXTENSION)).strftime("%d.%m.%Y")
                        pos["extension_count"] = pos.get("extension_count", 0) + 1
                        append_jsonl(PORTFOY_AUDIT_FILE, {
                            "event": "max_gun_extended",
                            "symbol": sym,
                            "new_max_gun_date": pos["max_gun_date"],
                            "extension_count": pos["extension_count"],
                            "scan_meta": meta,
                        })
                        continue
                    # Aday listede yok veya veri hatası -> Güvenli MAX_GUN çıkışı
                    sub_reason = "not_in_candidate_list" if status == "valid" else f"data_error_{status}"
                    if status != "valid":
                        append_jsonl(PORTFOY_AUDIT_FILE, {
                            "event": "max_gun_data_error",
                            "symbol": sym,
                            "status": status,
                            "scan_meta": meta,
                        })
                else:
                    # Uzatılmış max_gun_date henüz gelmedi (örn: 11., 12., 13., 14. gün).
                    # Erken MAX_GUN satışı yapma, pozisyonu tutmaya devam et!
                    continue

                net_tutar, komisyon = hesapla_net_tutar(pos["lotlar"], close, islem="satis")
                portfoy["nakit"] += net_tutar
                _trade_kaydet(portfoy, sym, pos, close, "MAX_GUN", likidite_teyitli=bool(bar.get("volume", 0) > 0), now=ref_now)
                kapatilacak.append(sym)
                gun_ret = (close - giris_f) / giris_f
                mesajlar.append(f"\u23f0 <b>MAX GÜN - {sym}</b>\n   Çıkış: {close:.2f} | Getiri: {gun_ret*100:+.1f}%")
                append_jsonl(PORTFOY_AUDIT_FILE, {
                    "event": "max_day",
                    "symbol": sym,
                    "exit_price": close,
                    "return_pct": gun_ret * 100,
                    "sub_reason": sub_reason,
                })
        except Exception as exc:
            log.debug("Pozisyon guncelle %s: %s", sym, exc)
    for sym in kapatilacak:
        portfoy["pozisyonlar"].pop(sym, None)
    # Acil likidasyon: Yalnızca makro_karar ve makro_skor aynı tutarlı makro değerlendirmesinde GIRME ve skor >= EMERGENCY_LIQUIDATION_SCORE ise
    if makro_skor is not None and makro_karar == "GIRME" and portfoy["pozisyonlar"] and makro_skor >= EMERGENCY_LIQUIDATION_SCORE:
        semboller = list(portfoy["pozisyonlar"].keys())
        for sym in semboller:
            pos = portfoy["pozisyonlar"][sym]
            f_detay = guncel_fiyat_detayli(sym, now=ref_now)
            if f_detay.get("trade_eligible", False) and f_detay.get("price") and f_detay["price"] > 0:
                cikis_f = round(float(f_detay["price"]) * (1-VARSAYILAN_KAYMA_ORANI), 4)
                portfoy["pozisyonlar"].pop(sym, None)
                kapatilacak.append(sym)
                net_tutar, _ = hesapla_net_tutar(pos["lotlar"], cikis_f)
                portfoy["nakit"] += net_tutar
                _trade_kaydet(portfoy, sym, pos, cikis_f, "ACIL_NAKIT", now=ref_now, fiyat_kaynak=f_detay.get("source"), fiyat_zaman=f_detay.get("time"))
                mesajlar.append(f"\U0001f6a8 <b>ACİL NAKİT - {sym}</b>\n   Çıkış: {cikis_f:.2f}")
                append_jsonl(PORTFOY_AUDIT_FILE, {
                    "event": "risk_off_liquidation_symbol",
                    "symbol": sym,
                    "cikis_fiyat": cikis_f,
                    "kaynak": f_detay.get("source"),
                    "zaman": f_detay.get("time"),
                })
            else:
                # Fiyat yoksa POZİSYONU KORU, giriş fiyatından satma!
                # Tasfiye talebini bekleyen durum olarak kaydet.
                pos["pending_liquidation"] = {
                    "requested_at": ref_now.isoformat(),
                    "reason": "acil_nakit_fiyat_yok",
                    "makro_skor": makro_skor,
                    "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                }
                log.warning("Acil tasfiye: %s icin guncel fiyat yok, pozisyon korundu ve beklemeye alindi", sym)
                append_jsonl(PORTFOY_AUDIT_FILE, {
                    "event": "liquidation_deferred_no_price",
                    "symbol": sym,
                    "reason": "guncel_fiyat_yok",
                    "fiyat_meta": f_detay,
                })
    return portfoy, mesajlar
    return portfoy, mesajlar


def makro_ozet_metni(skor: float, karar: str, detaylar: list, piyasa_ret: dict) -> str:
    def fmt(v): return f"{v:+.1f}%" if v != 0.0 else "\u2014"
    sp   = piyasa_ret.get("^GSPC",    0.0)
    nq   = piyasa_ret.get("^IXIC",    0.0)
    vix  = piyasa_ret.get("^VIX",     0.0)
    n225 = piyasa_ret.get("^N225",    0.0)
    hsi  = piyasa_ret.get("^HSI",      0.0)
    sse  = piyasa_ret.get("000001.SS",0.0)
    altin = piyasa_ret.get("GC=F",    0.0)
    emoji = {"GIRME":"\U0001f534","DIKKATLI":"\U0001f7e1","NORMAL":"\U0001f7e2"}.get(karar,"\u26aa")
    return (
        "\U0001f30d <b>K\u00fcresel G\u00f6r\u00fcn\u00fcm</b>\n"
        f"   \U0001f1fa\U0001f1f8 S&P500: {fmt(sp)} | Nasdaq: {fmt(nq)} | VIX: {fmt(vix)}\n"
        f"   \U0001f30f Nikkei: {fmt(n225)} | Hang Seng: {fmt(hsi)} | Shanghai: {fmt(sse)}\n"
        f"   \U0001f947 Alt\u0131n: {fmt(altin)}\n"
        f"\U0001f4ca Makro Skor: {skor:.1f} \u2192 {emoji} <b>{karar}</b>"
    )


def portfoy_ozet_mesaji(portfoy: dict, saat_label: str, model_status: str = "pasif") -> str:
    prev = portfolio_preview(portfoy)
    lines = [
        f"\U0001f4bc <b>PORTF\u00d6Y - {saat_label}</b>",
        "--------------------",
        f"\U0001f4b0 Ba\u015flang\u0131\u00e7 : {portfoy.get('baslangic', SERMAYE_BASLANGIC):,.0f} TL",
        f"\U0001f4ca G\u00fcncel    : {prev['equity']:,.0f} TL",
        f"\U0001f4b5 Nakit     : {prev['cash']:,.0f} TL",
        f"\U0001f916 Model     : {model_status}",
    ]
    if portfoy["pozisyonlar"]:
        for sym, pos in portfoy["pozisyonlar"].items():
            f = guncel_fiyat(sym)
            ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f else 0
            lines.append(
                f"   \u2022 <b>{sym}</b> {pos['lotlar']}lot @ {pos['giris_f']:.2f}"
                f" \u2192 {f:.2f if f else '?'} ({ret:+.1f}%) g\u00fcn:{pos.get('gun',0)}"
            )
    else:
        lines.append("   - A\u00e7\u0131k pozisyon yok")
    waiting = portfoy.get("bekleyen_al", [])
    if waiting:
        lines.append(f"\U0001f4cb Bekleyen: {', '.join(x['symbol'] for x in waiting[:7])}")
    summary = portfoy.get("last_open_attempt_summary", {})
    if summary:
        lines.append(
            f"\U0001fa9f Son al\u0131m: bekleyen={summary.get('pending',0)} | "
            f"al\u0131nd\u0131={summary.get('bought',0)} | "
            f"al\u0131namad\u0131={summary.get('failed',0)} | "
            f"s\u00fcresi dolan={summary.get('expired',0)}"
        )
    return "\n".join(lines)


def alim_denemesi(portfoy: dict, makro_karar: str, viop_bias: dict, now: datetime, makro_meta: dict | None = None):
    aktif, expired = ayikla_suresi_dolan_bekleyenler(portfoy.get("bekleyen_al", []), now)
    portfoy["bekleyen_al"] = aktif
    portfoy, al_mesajlari, alinanlar, alinmayanlar = yeni_pozisyon_ac(
        portfoy, aktif, makro_karar, viop_bias, makro_meta=makro_meta, now=now)
    portfoy["bekleyen_al"] = [x for x in aktif if x["symbol"] not in set(alinanlar)]
    # Deneme kaydını güncelle
    for item in portfoy["bekleyen_al"]:
        sym = item["symbol"]
        neden = next((a["reason"] for a in alinmayanlar if a["symbol"] == sym), "")
        item["attempt_count"]       = item.get("attempt_count", 0) + 1
        item["last_attempt_time"]   = now.strftime("%d.%m.%Y %H:%M")
        item["last_attempt_reason"] = neden
    summary = {
        "pending":  len(aktif),
        "bought":   len(alinanlar),
        "failed":   len(alinmayanlar),
        "expired":  len(expired),
        "failed_reasons": (alinmayanlar + expired)[:10],
        "time":     now.strftime("%d.%m.%Y %H:%M"),
    }
    portfoy["last_open_attempt_summary"] = summary
    portfoy.setdefault("open_attempts_today", []).append({
        "saat":     now.strftime("%H:%M"),
        "tarih":    now.strftime("%Y-%m-%d"),
        "alinan":   alinanlar,
        "alinmayan": [a["symbol"] for a in alinmayanlar],
        "nedenler": {a["symbol"]: a["reason"] for a in alinmayanlar},
    })
    # Son 7 günü tut, eskiyi sil
    yedi_gun_once = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    portfoy["open_attempts_today"] = [
        a for a in portfoy["open_attempts_today"]
        if a.get("tarih", "9999-99-99") >= yedi_gun_once
    ]
    mesajlar = al_mesajlari
    if expired:
        mesajlar += [f"\u23f3 <b>S\u00fcresi dolan:</b> {', '.join(x['symbol'] for x in expired[:7])}"]
    return portfoy, mesajlar, summary


def sabah_09_akisi():
    now = _p1_now()
    if not is_bist_islem_gunu(now.date()):
        return "GIRME"
    saat_label = now.strftime("%d.%m.%Y %H:%M")
    _makro_cache_temizle()  # her sabah taze veri çek
    skor, detaylar, karar, piyasa_ret = makro_risk_skoru()
    viop_bias = viop_bias_hesapla()
    _, model_status = lgbm_model_yukle()
    adaylar, _, elinenler = tarama_calistir(saat_label, karar, viop_bias)
    portfoy = portfoy_yukle()
    bekleyen = []
    for aday in adaylar[:MAX_HISSE]:
        if aday["symbol"] in portfoy["pozisyonlar"]:
            continue
        bekleyen.append(bekleyen_adayi_hazirla(aday, now, valid_for_date=now.date()))
    portfoy["bekleyen_al"] = bekleyen
    portfoy["open_attempts_today"] = []
    makro_payload = makro_karar_olustur(skor, karar, detaylar, piyasa_ret, kaynak="sabah_09_akisi", now=now)
    portfoy["makro_karar"] = makro_payload
    portfoy_kaydet(portfoy)
    durum_kaydet({
        "tarih": saat_label,
        "makro_skor": skor,
        "makro_karar": karar,
        "makro_detaylar": detaylar,
        "piyasa_ret": piyasa_ret,
        "viop": {
            "label": viop_bias.get("label"),
            "score": viop_bias.get("score"),
            "ticker": viop_bias.get("ticker"),
        },
    })
    lines = [
        f"\u2600\ufe0f <b>09:00 Sabah De\u011flendirmesi \u2014 {saat_label}</b>", "",
        makro_ozet_metni(skor, karar, detaylar, piyasa_ret), "",
        f"\U0001f4cc V\u0130OP: <b>{viop_bias['label']}</b> ({viop_bias['score']:+.0f}p) | "
        f"Kaynak: {viop_bias.get('ticker','?')}", "",
    ]
    if karar == "GIRME":
        lines.append("\U0001f534 <b>GIRME karar\u0131 \u2014 bug\u00fcn yeni al\u0131m yap\u0131lmayacak.</b>")
    elif bekleyen:
        lines.append(f"\U0001f4cb <b>Bug\u00fcn \u015funlar\u0131 alal\u0131m</b> ({len(bekleyen)} aday, 11:00'da i\u015flem):")
        for aday in bekleyen[:7]:
            lines.append(
                f"   \u2022 <b>{aday['symbol']}</b> {aday.get('alpha_tag','')} "
                f"skor:{aday['final_score']:.1f} puan:{aday['score_count']} "
                f"LGBM:{aday['lgbm_score'] if aday['lgbm_score'] is not None else '-'} [{','.join(aday.get('strategies',[]))}]"
            )
    else:
        lines.append("   - Bug\u00fcn g\u00fc\u00e7l\u00fc aday yok")
    if elinenler:
        lines.append(f"\n\U0001f6ab LGBM filtresiyle elinen: {', '.join(x['symbol'] for x in elinenler[:5])}")
    try:
        from mott_sabah_telegram import gunluk_plan_gonder
        gunluk_plan_gonder()
    except Exception as exc:
        log.warning("Sabah günlük plan Telegram: %s", exc)
    log.info("Sabah degerlendirmesi tamamlandi")
    return karar


def saat_11_alim(makro_karar: str | None = None, makro_skor: float | None = None, makro_meta: dict | None = None, now: datetime | None = None):
    ref_now = now if now is not None else datetime.now()
    viop_bias = viop_bias_hesapla()
    portfoy = portfoy_yukle()
    if makro_karar is None:
        makro_karar, makro_skor, makro_meta = get_aktif_makro_karar(portfoy, now=ref_now)
    onceki = set(portfoy["pozisyonlar"].keys())
    mesajlar = []
    # Piyasa açılışından itibaren (10:00 TSİ) STOP/TP kontrolü de burada
    # yapılır — yalnızca "takip" penceresini (11:20+) beklemek, sabah erken
    # saatlerde taşınan pozisyonların saatlerce izlenmeden kalmasına yol açardı.
    if portfoy["pozisyonlar"]:
        portfoy, islem_msg = pozisyon_guncelle_saatlik(portfoy, makro_karar, makro_skor=makro_skor, now=ref_now)
        mesajlar.extend(islem_msg)
    portfoy, al_msg, summary = alim_denemesi(portfoy, makro_karar, viop_bias, ref_now, makro_meta=makro_meta)
    mesajlar.extend(al_msg)
    portfoy["last_hourly_check_time"] = ref_now.strftime("%d.%m.%Y %H:%M")
    portfoy_kaydet(portfoy)
    if mesajlar:
        sonra = set(portfoy["pozisyonlar"].keys())
        _p1_telegram_islem(
            portfoy,
            giris=list(sonra - onceki),
            cikis=list(onceki - sonra),
            mesajlar=mesajlar,
        )
    # P1 Paper Trading Hook (Feature Flag P1_PAPER=on)
    if os.environ.get("P1_PAPER", "").lower() in ("1", "true", "on", "yes"):
        try:
            import p1_paper
            p1_paper.run_phase_acilis(as_of_date=pd.to_datetime(ref_now.date()))
        except Exception as p_exc:
            log.warning("P1 Paper acilis hook hatasi: %s", p_exc)


def saat_1130_ozeti():
    """11:30 özeti — işlem yoksa Telegram gönderilmez."""
    log.info("11:30 ozet: islem bazli politika nedeniyle Telegram atlanir")


def saatlik_kontrol(makro_karar: str | None = None, makro_skor: float | None = None, makro_meta: dict | None = None, now: datetime | None = None):
    ref_now = now if now is not None else datetime.now()
    viop_bias = viop_bias_hesapla()
    portfoy = portfoy_yukle()
    if makro_karar is None:
        makro_karar, makro_skor, makro_meta = get_aktif_makro_karar(portfoy, now=ref_now)
    onceki = set(portfoy["pozisyonlar"].keys())
    mesajlar = []
    if portfoy["pozisyonlar"]:
        portfoy, islem_msg = pozisyon_guncelle_saatlik(portfoy, makro_karar, makro_skor=makro_skor, now=ref_now)
        mesajlar.extend(islem_msg)
    portfoy["bekleyen_al"] = retry_bekleyenleri_filtrele(portfoy.get("bekleyen_al", []))
    if portfoy["bekleyen_al"]:
        portfoy, al_msg, _ = alim_denemesi(portfoy, makro_karar, viop_bias, ref_now, makro_meta=makro_meta)
        mesajlar.extend(al_msg)
    portfoy["last_hourly_check_time"] = ref_now.strftime("%d.%m.%Y %H:%M")
    portfoy_kaydet(portfoy)
    append_jsonl(PORTFOY_AUDIT_FILE, {
        "event": "hourly_check", "saat": ref_now.strftime("%H:%M"),
        "pozisyon_sayisi": len(portfoy["pozisyonlar"]),
        "bekleyen_sayisi": len(portfoy.get("bekleyen_al", [])),
    })
    if mesajlar:
        sonra = set(portfoy["pozisyonlar"].keys())
        _p1_telegram_islem(
            portfoy,
            giris=list(sonra - onceki),
            cikis=list(onceki - sonra),
            mesajlar=mesajlar,
        )
    # P1 Paper Trading Hook (Feature Flag P1_PAPER=on)
    if os.environ.get("P1_PAPER", "").lower() in ("1", "true", "on", "yes"):
        try:
            import p1_paper
            p1_paper.run_phase_takip(as_of_date=pd.to_datetime(ref_now.date()))
        except Exception as p_exc:
            log.warning("P1 Paper takip hook hatasi: %s", p_exc)


def kapanis_ozeti_1730():
    """17:30 — bekleyenleri temizle; Telegram yalnızca işlem anında gönderilir."""
    now = datetime.now()
    portfoy = portfoy_yukle()
    _, expired = ayikla_suresi_dolan_bekleyenler(portfoy.get("bekleyen_al", []), now)
    portfoy["bekleyen_al"] = []
    portfoy_kaydet(portfoy)
    log.info("17:30 kapanis: bekleyen temizlendi (Telegram: islem yoksa atlanir)")


def preview_from_signals(signals: list, scan_time: str, scan_label: str) -> dict:
    model, model_status = lgbm_model_yukle()
    viop_bias = viop_bias_hesapla()
    preview = []
    for signal in signals[:15]:
        preview.append(build_candidate(signal["symbol"], signal, model, viop_bias))
    preview.sort(key=lambda x: (-x["final_score"], x["symbol"]))
    portfoy = portfoy_yukle()
    return {
        "preview_candidates": preview[:5],
        "portfolio":          portfolio_preview(portfoy),
        "waiting_symbols":    [x["symbol"] for x in preview[:MAX_HISSE]],
        "viop_bias":          viop_bias,
        "scan_time":          scan_time,
        "scan_label":         scan_label,
        "model_status":       model_status,
    }


def flask_baslat():
    from flask import Flask, jsonify, request as freq
    app = Flask(__name__)

    def api_key_ok():
        gelen = freq.headers.get("X-API-Key") or freq.args.get("api_key", "")
        return not FLASK_API_KEY or gelen == FLASK_API_KEY

    @app.route("/tarama", methods=["POST"])
    def tarama_al():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        data    = freq.get_json(silent=True) or {}
        signals = data.get("signals")
        if signals is None:
            semboller = data.get("semboller", [])
            signals   = [{"symbol": s, "score_count": 1, "strategies": [],
                          "scan_time": data.get("scan_time",""), "scan_label": data.get("scan_label","")}
                         for s in semboller]
        clean = []
        for s in signals[:200]:
            symbol = str(s.get("symbol","")).upper().strip()
            if not symbol:
                continue
            clean.append({
                "symbol":     symbol,
                "score_count": int(s.get("score_count",1)),
                "strategies":  list(s.get("strategies",[])),
                "change_pct":  float(s.get("change_pct",0)),
                "rsi":         float(s.get("rsi",0)),
                "rel_vol":     float(s.get("rel_vol",0)),
                "cmf":         float(s.get("cmf",0)),
                "islem_tl":    float(s.get("islem_tl",0)),
                "scan_time":   s.get("scan_time", data.get("scan_time","")),
                "scan_label":  s.get("scan_label", data.get("scan_label","")),
            })
        payload = {"scan_time": data.get("scan_time",""), "scan_label": data.get("scan_label",""), "signals": clean}
        with open(TARAMA_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        for item in clean:
            append_jsonl(SIGNAL_AUDIT_FILE, {**item, "event": "scanner_signal"})
        summary = preview_from_signals(clean, payload["scan_time"], payload["scan_label"])
        return jsonify({"status": "ok", "n": len(clean), **summary})

    
    @app.route("/tarama_p2", methods=["POST"])
    def tarama_p2_al():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        data = freq.get_json(silent=True) or {}
        mod  = data.get("mod", "aksam")

        if mod == "teyit":
            teyitler = data.get("teyitler", {})
            portfoy  = p2_portfoy_yukle()
            guncellenen = []
            for item in portfoy.get("bekleyen_al", []):
                sym = item["symbol"]
                if sym in teyitler:
                    t = teyitler[sym]
                    item["teyit_skoru"]     = t.get("teyit_skoru", 0)
                    item["teyit_var"]       = t.get("teyit_var", False)
                    item["teyit_sinyaller"] = t.get("teyit_sinyaller", [])
                    item["final_score"]     = item.get("score", 0) + t.get("teyit_skoru", 0)
                    guncellenen.append(sym)
            p2_portfoy_kaydet(portfoy)
            log.info("P2 teyit guncellendi: %s", guncellenen)
            return jsonify({"status": "ok", "mod": "teyit", "guncellenen": guncellenen})

        else:
            signals = data.get("signals", [])
            clean   = []
            for s in signals[:200]:
                symbol = str(s.get("symbol", "")).upper().strip()
                if not symbol:
                    continue
                clean.append({
                    "symbol":          symbol,
                    "score":           float(s.get("score", 0)),
                    "final_score":     float(s.get("score", 0)),
                    "verdict":         s.get("verdict", "AL"),
                    "pd_zone":         s.get("pd_zone", ""),
                    "rvol":            float(s.get("rvol", 1)),
                    "rsi":             float(s.get("rsi", 50)),
                    "htf_aligned":     bool(s.get("htf_aligned", False)),
                    "signals":         list(s.get("signals", [])),
                    "islem_tl":        float(s.get("islem_tl", 0)),
                    "change_pct":      float(s.get("change_pct", 0)),
                    "scan_time":       s.get("scan_time", data.get("scan_time", "")),
                    "scan_label":      s.get("scan_label", data.get("scan_label", "")),
                    "teyit_skoru":     0.0,
                    "teyit_var":       False,
                    "teyit_sinyaller": [],
                })
            payload = {
                "scan_time":  data.get("scan_time", ""),
                "scan_label": data.get("scan_label", ""),
                "signals":    clean,
            }
            with open(TARAMA_P2_FILE, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            portfoy = p2_portfoy_yukle()
            mevcut_semboller = set(portfoy["pozisyonlar"].keys())
            bekleyen = []
            for s in clean:
                if s["symbol"] not in mevcut_semboller:
                    item = dict(s)
                    item["queued_at"]           = data.get("scan_time", "")
                    item["attempt_count"]       = 0
                    item["last_attempt_reason"] = ""
                    bekleyen.append(item)
            portfoy["bekleyen_al"]         = bekleyen
            portfoy["open_attempts_today"] = []
            p2_portfoy_kaydet(portfoy)
            for item in clean:
                append_jsonl(SIGNAL_AUDIT_P2_FILE, {**item, "event": "p2_scanner_signal"})
            log.info("P2 tarama alindi: %s sinyal, mod=%s", len(clean), mod)
            return jsonify({
                "status": "ok", "n": len(clean), "mod": mod,
                "portfolio": p2_portfolio_preview(portfoy),
            })

    @app.route("/durum_p2", methods=["GET"])
    def durum_p2():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        portfoy = p2_portfoy_yukle()
        return jsonify({
            "status":      "ok",
            "portfolio":   p2_portfolio_preview(portfoy),
            "pozisyonlar": list(portfoy["pozisyonlar"].keys()),
            "bekleyen_al": portfoy.get("bekleyen_al", []),
        })

    @app.route("/durum", methods=["GET"])
    def durum():
        p = portfoy_yukle()
        return jsonify(portfolio_preview(p) | {"baslangic": SERMAYE_BASLANGIC,
                                                "semboller": list(p["pozisyonlar"].keys())})

    @app.route("/semboller", methods=["GET"])
    def semboller():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        syms = hisse_listesi_yukle()
        return jsonify({"status": "ok", "symbols": syms, "n": len(syms)})

    @app.route("/saglik", methods=["GET"])
    def saglik():
        return jsonify({"status": "ok", "zaman": datetime.now().strftime("%H:%M:%S")})

    log.info("Flask basliyor: port %s", FLASK_PORT)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)


def main():
    log.info("portfoy_yonetici.py basliyor")
    # Flask kaldırıldı — GitHub Actions tek seferlik çalışır
    makro_karar       = "NORMAL"
    son_saat          = ""
    saat_11_yapildi   = False
    saat_1130_yapildi = False
    saat_1730_yapildi = False

    while True:
        now  = _p1_now()
        saat = now.strftime("%H:%M")
        gun  = now.weekday()

        telegram_komutlarini_kontrol_et()

        if not is_bist_islem_gunu(now.date()):
            time.sleep(300)
            continue

        # Gün başı sıfırla
        if saat == "00:01" and son_saat != "00:01":
            saat_11_yapildi = saat_1130_yapildi = saat_1730_yapildi = False
            son_saat = saat

        # 09:00 Sabah değerlendirmesi
        if saat == "09:00" and son_saat != "09:00":
            makro_karar = sabah_09_akisi()
            son_saat = saat
            time.sleep(61)

        # 11:00 İlk alım
        elif saat == "11:00" and not saat_11_yapildi:
            karar, skor, meta = get_aktif_makro_karar(portfoy_yukle(), now=now)
            saat_11_alim(karar, makro_skor=skor, makro_meta=meta, now=now)
            p2_saatlik_kontrol(karar)
            saat_11_yapildi = True
            son_saat = saat
            time.sleep(61)

        # 11:30 Portföy özeti
        elif saat == "11:30" and not saat_1130_yapildi:
            saat_1130_ozeti()
            saat_1130_yapildi = True
            son_saat = saat
            time.sleep(61)

        # 12:00-17:00 Saatlik kontrol
        elif now.minute == 0 and 12 <= now.hour <= 17 and son_saat != saat:
            karar, skor, meta = get_aktif_makro_karar(portfoy_yukle(), now=now)
            saatlik_kontrol(karar, makro_skor=skor, makro_meta=meta, now=now)
            p2_saatlik_kontrol(karar)
            son_saat = saat
            time.sleep(61)

        # 17:30 Kapanış özeti
        elif saat == "17:30" and not saat_1730_yapildi:
            kapanis_ozeti_1730()
            saat_1730_yapildi = True
            son_saat = saat
            time.sleep(61)

        # 21:00 Gün sonu pozisyon güncelle
        elif saat == "21:00" and son_saat != "21:00":
            portfoy = portfoy_yukle()
            if portfoy["pozisyonlar"]:
                for pos in portfoy["pozisyonlar"].values():
                    pos["gun"] = pos.get("gun", 0) + 1
                portfoy_kaydet(portfoy)
            p2_gun_sonu_guncelle()
            son_saat = saat
            time.sleep(61)

        else:
            time.sleep(20)


def _cli_p1_paper(args: list):
    import p1_paper
    sub = args[0] if args else "status"
    if sub == "aksam":
        p1_paper.run_phase_aksam()
    elif sub in ("acilis", "alim"):
        p1_paper.run_phase_acilis()
    elif sub in ("takip", "kapani"):
        p1_paper.run_phase_takip()
    elif sub == "replay":
        from p1_signal_parity import run_60_session_replay
        res = run_60_session_replay()
        print(json.dumps(res, indent=2))
    elif sub == "status":
        st = p1_paper.load_paper_state()
        print(f"P1 Paper Status (20k): Nakit={st.get('cash_20k',0):,.2f} TL, Pozisyon={len(st.get('positions_20k',{}))}")
        print(f"P1 Paper Status (10k Eq): Nakit={st.get('cash_10k',0):,.2f} TL, Pozisyon={len(st.get('positions_10k',{}))}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        cmds = {
            "sabah":  sabah_09_akisi,
            "alim":   lambda: saat_11_alim(),
            "ozet":   saat_1130_ozeti,
            "kapani": kapanis_ozeti_1730,
            "takip":  lambda: saatlik_kontrol(),
            "durum":  lambda: print(json.dumps(portfoy_yukle(), indent=2, ensure_ascii=False)),
            "viop":   lambda: print(json.dumps(viop_bias_hesapla(), ensure_ascii=False)),
            "makro":  lambda: [print(f"Skor: {s} -> {k}") or [print(f"  {t}: {v:+.2f}%") for t,v in pr.items()]
                               for s,_,k,pr in [makro_risk_skoru()]],
            # ── P1 — Paper Trading hook ──────────────────────────────────
            "p1_paper": lambda: _cli_p1_paper(sys.argv[2:]),
            # ── P2 — SMC portföy köprüsü ──────────────────────────────────
            "p2_aksam": p2_adaylari_yukle_ve_hazirla,
            "p2_takip": lambda: p2_saatlik_kontrol("NORMAL"),
            "p2_durum": lambda: print(json.dumps(p2_portfoy_yukle(), indent=2, ensure_ascii=False)),
        }
        if cmd in cmds:
            cmds[cmd]()
        else:
            print("Kullanim: python portfoy_yonetici.py [sabah|alim|ozet|kapani|takip|durum|viop|makro|p1_paper|p2_aksam|p2_takip|p2_durum]")
    else:
        main()
