"""
mott_fiyat.py — Ortak canlı fiyat modülü
=========================================
Öncelik: TradingView screener (BIST verisi gerçek zamanlı ve TV grafikleriyle
birebir uyumlu). Yedek: yfinance (BIST'te ~15 dk gecikmeli ve zaman zaman
TV'den sapıyor — bu yüzden yalnızca TV erişilemezse kullanılır).

Kullanım:
    from mott_fiyat import canli_fiyat, canli_fiyatlar, tv_fiyatlar

    canli_fiyatlar(["GARAN", "THYAO"])   # {"GARAN": 12.34, ...} TV + yf yedek
    canli_fiyat("GARAN")                 # 12.34 | None
    tv_fiyatlar(["GARAN"])               # yalnızca TV, yedek yok
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


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


def canli_fiyatlar(semboller: list[str]) -> dict[str, float]:
    """TV öncelikli toplu fiyat; TV'de bulunamayanlar yfinance ile tamamlanır."""
    semboller = [_temiz(s) for s in semboller]
    fiyatlar = tv_fiyatlar(semboller)
    for s in semboller:
        if s not in fiyatlar:
            p = _yf_fiyat(s)
            if p is not None:
                fiyatlar[s] = p
    return fiyatlar


def canli_fiyat(sym: str) -> float | None:
    """Tek sembol için canlı fiyat (TV → yfinance)."""
    return canli_fiyatlar([sym]).get(_temiz(sym))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["GARAN", "THYAO", "ASELS"]
    print(canli_fiyatlar(syms))
