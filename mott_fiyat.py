"""
mott_fiyat.py — Ortak canlı fiyat modülü
=========================================
Öncelik: TradingView screener (BIST verisi gerçek zamanlı ve TV grafikleriyle
birebir uyumlu). Yedek: yfinance (BIST'te ~15 dk gecikmeli ve zaman zaman
TV'den sapıyor — bu yüzden yalnızca TV erişilemezse kullanılır).

FAZ 6.1: In-memory price cache — aynı sembol grubu için tekrarlı API
cagrilarini onler. Cache TTL = 60 saniye.

Kullanım:
    from mott_fiyat import canli_fiyat, canli_fiyatlar, tv_fiyatlar

    canli_fiyatlar(["GARAN", "THYAO"])   # {"GARAN": 12.34, ...} TV + yf yedek
    canli_fiyat("GARAN")                 # 12.34 | None
    tv_fiyatlar(["GARAN"])               # yalnızca TV, yedek yok
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FAZ 6.1 — In-Memory Price Cache
# ---------------------------------------------------------------------------

CACHE_TTL = 60  # saniye — fiyat cache süresi
_PRICE_CACHE: dict[str, dict] = {}  # {symbol: {price, timestamp, source}}


def clear_cache() -> None:
    """Tüm fiyat cache'ini temizle."""
    _PRICE_CACHE.clear()


def cache_stats() -> dict:
    """Cache durumu — test/debug için."""
    now = time.time()
    fresh = sum(1 for v in _PRICE_CACHE.values() if now - v["timestamp"] < CACHE_TTL)
    stale = len(_PRICE_CACHE) - fresh
    return {"total": len(_PRICE_CACHE), "fresh": fresh, "stale": stale}


def _temiz(sym: str) -> str:
    return sym[:-3] if sym.endswith(".IS") else sym


def tv_fiyatlar(semboller: list[str]) -> dict[str, float]:
    """TradingView screener'dan toplu güncel fiyat. Hata/eksikte boş döner."""
    semboller = [_temiz(s) for s in semboller]
    if not semboller:
        return {}
    try:
        from tradingview_screener import Query, col
        _, df = (Query()
                 .set_markets("turkey")
                 .select("name", "close")
                 .where(col("name").isin(semboller))
                 .get_scanner_data())
        out: dict[str, float] = {}
        for _, row in df.iterrows():
            c = row.get("close")
            # NaN kontrolü: NaN != NaN
            if c is not None and c == c and float(c) > 0:
                out[str(row["name"])] = round(float(c), 4)
        log.info("TV fiyat: %d/%d sembol alındı", len(out), len(semboller))
        return out
    except Exception as e:
        log.warning("TV fiyat hatası (yfinance yedeğine düşülecek): %s", e)
        return {}


def _yf_fiyat(sym: str) -> float | None:
    """yfinance son kapanış — yedek kaynak."""
    try:
        import yfinance as yf
        ticker = f"{_temiz(sym)}.IS"
        df = yf.Ticker(ticker).history(period="5d")
        df = df.dropna(subset=["Close"])
        if not df.empty:
            p = float(df["Close"].iloc[-1])
            if p == p and p > 0:
                return round(p, 4)
    except Exception as e:
        log.debug("yfinance %s: %s", sym, e)
    return None


def canli_fiyatlar(semboller: list[str], *, use_cache: bool = True) -> dict[str, float]:
    """TV öncelikli toplu fiyat; TV'de bulunamayanlar yfinance ile tamamlanır.

    FAZ 6.1: use_cache=True ise in-memory cache kullanılır (TTL = CACHE_TTL).
    use_cache=False ise her zaman raw API çağrısı yapılır.
    """
    semboller = [_temiz(s) for s in semboller]
    if not semboller:
        return {}

    now = time.time()
    fiyatlar: dict[str, float] = {}
    to_fetch: list[str] = []

    # 1. Cache'den okunabilenleri al
    if use_cache:
        for s in semboller:
            entry = _PRICE_CACHE.get(s)
            if entry and (now - entry["timestamp"]) < CACHE_TTL:
                fiyatlar[s] = entry["price"]
            else:
                to_fetch.append(s)
    else:
        to_fetch = list(semboller)

    # 2. Eksik sembolleri raw fetch ile çek
    if to_fetch:
        tv_fiyatlari = tv_fiyatlar(to_fetch)
        for s in to_fetch:
            if s in tv_fiyatlari:
                fiyatlar[s] = tv_fiyatlari[s]
                if use_cache:
                    _PRICE_CACHE[s] = {
                        "price": tv_fiyatlari[s],
                        "timestamp": now,
                        "source": "tv",
                    }
            else:
                p = _yf_fiyat(s)
                if p is not None:
                    fiyatlar[s] = p
                    if use_cache:
                        _PRICE_CACHE[s] = {
                            "price": p,
                            "timestamp": now,
                            "source": "yf",
                        }

    return fiyatlar


def canli_fiyat(sym: str, *, use_cache: bool = True) -> float | None:
    """Tek sembol için canlı fiyat (TV → yfinance).

    FAZ 6.1: use_cache parametresi ile cache kontrolü.
    """
    result = canli_fiyatlar([sym], use_cache=use_cache)
    return result.get(_temiz(sym))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["GARAN", "THYAO", "ASELS"]
    print(canli_fiyatlar(syms))
