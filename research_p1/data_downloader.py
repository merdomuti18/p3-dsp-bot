# -*- coding: utf-8 -*-
"""
research_p1/data_downloader.py
==============================
BIST100 hisselerinin 2020-2026 OHLCV verilerini Yahoo Finance üzerinden
indirip yerel Parquet/CSV dosyalarında önbelleğe alır.

Dönem:
  Ana dönem: 06.10.2021 – 05.10.2026 (5 yıl)
  Isınma dönemi: 01.01.2020 – 05.10.2021 (EMA200 ve 200 barlık indikatörler için)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "ohlcv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# BIST 100 Listesi (KAP 2026-Q4 resmi portföy dağılımı + geçmiş dönem ana bileşenleri)
BIST100_SYMBOLS = [
    "AEFES", "AGHOL", "AHGAZ", "AKBNK", "AKCNS", "AKFYE", "AKSA", "AKSEN", "ALARK", "ALBRK",
    "ALTNY", "ANHYT", "ANSGR", "ARCLK", "ASELS", "ASTOR", "AYGAZ", "BALSU", "BERA", "BIMAS",
    "BINHO", "BRSAN", "BRYAT", "BSOKE", "BTCIM", "CANTE", "CCOLA", "CIMSA", "CVKMD", "CWENE",
    "DAPGM", "DOAS", "DOHOL", "DSTKF", "ECILC", "ECZYT", "EFOR", "EGEEN", "EGGUB", "EKGYO",
    "ENERY", "ENJSA", "ENKAI", "ENTRA", "EREGL", "ESEN", "EUPWR", "EUREN", "FENER", "FROTO",
    "GARAN", "GENIL", "GESAN", "GLRMK", "GLYHO", "GRSEL", "GRTHO", "GSRAY", "GUBRF", "GWIND",
    "HALKB", "HEKTS", "IEYHO", "ISCTR", "ISDMR", "ISMEN", "IZENR", "KARSN", "KATMR", "KCAER",
    "KCHOL", "KLRHO", "KORDS", "KRDMD", "KTLEV", "KUYAS", "LMKDC", "MAGEN", "MAVI", "MGROS",
    "MIATK", "MPARK", "OBAMS", "ODAS", "ODINE", "OTKAR", "OYAKC", "PAHOL", "PASEU", "PATEK",
    "PETKM", "PGSUS", "PSGYO", "QUAGR", "RALYH", "REEDR", "RGYAS", "RYSAS", "SAHOL", "SARKY",
    "SASA", "SISE", "SKBNK", "SNGYO", "SOKM", "TABGD", "TAVHL", "TCKRC", "TCELL", "THYAO",
    "TKFEN", "TOASO", "TRALT", "TRENJ", "TRGYO", "TRMET", "TSKB", "TTKOM", "TTRAK", "TUKAS",
    "TUPRS", "TURSG", "ULKER", "VAKBN", "VESTL", "YEOTK", "YKBNK", "ZOREN"
]
# Tekil semboller
BIST100_SYMBOLS = sorted(list(dict.fromkeys(BIST100_SYMBOLS)))


def fetch_yahoo_ohlcv(symbol: str, start_dt: str = "2020-01-01", end_dt: str = "2026-10-06") -> tuple[pd.DataFrame | None, dict]:
    """Yahoo Finance chart API'sinden OHLCV, temettü ve split verilerini çeker."""
    p1 = int(datetime.strptime(start_dt, "%Y-%m-%d").timestamp())
    p2 = int(datetime.strptime(end_dt, "%Y-%m-%d").timestamp()) + 86400
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}.IS?period1={p1}&period2={p2}&interval=1d&events=div,splits"

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if not r.ok:
            return None, {"error": f"HTTP_{r.status_code}"}

        data = r.json()
        result = data.get("chart", {}).get("result")
        if not result or len(result) == 0:
            return None, {"error": "no_result"}

        res0 = result[0]
        timestamps = res0.get("timestamp", [])
        if not timestamps:
            return None, {"error": "no_timestamps"}

        quotes = res0.get("indicators", {}).get("quote", [{}])[0]
        adj = res0.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

        dates = [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps]
        df = pd.DataFrame({
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
            "adjclose": adj if len(adj) == len(timestamps) else quotes.get("close", []),
        }, index=pd.to_datetime(dates))

        # Temizlik
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        events = res0.get("events", {})
        splits = list(events.get("splits", {}).values()) if "splits" in events else []
        dividends = list(events.get("dividends", {}).values()) if "dividends" in events else []

        meta = {
            "symbol": symbol,
            "currency": res0.get("meta", {}).get("currency", "TRY"),
            "bars_count": len(df),
            "first_date": str(df.index[0].date()) if len(df) > 0 else "",
            "last_date": str(df.index[-1].date()) if len(df) > 0 else "",
            "splits_count": len(splits),
            "dividends_count": len(dividends),
            "splits": splits,
            "dividends": dividends,
        }
        return df, meta
    except Exception as exc:
        return None, {"error": str(exc)}


def download_and_cache_all(symbols: list[str] | None = None) -> dict:
    target_symbols = symbols or BIST100_SYMBOLS
    log.info("BIST100 veri indirme başlıyor: %d sembol", len(target_symbols))
    summary = {}
    success_count = 0

    for i, sym in enumerate(target_symbols, 1):
        cache_file = CACHE_DIR / f"{sym}.parquet"
        meta_file = CACHE_DIR / f"{sym}_meta.json"

        if cache_file.exists() and meta_file.exists():
            log.info("[%d/%d] %s önbellekten yüklendi.", i, len(target_symbols), sym)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            summary[sym] = meta
            success_count += 1
            continue

        df, meta = fetch_yahoo_ohlcv(sym)
        if df is not None and len(df) > 0:
            df.to_parquet(cache_file)
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            summary[sym] = meta
            success_count += 1
            log.info("[%d/%d] %s indirildi: %d bar (%s -> %s)", i, len(target_symbols), sym, len(df), meta["first_date"], meta["last_date"])
        else:
            log.warning("[%d/%d] %s indirilemedi: %s", i, len(target_symbols), sym, meta.get("error"))
            summary[sym] = {"error": meta.get("error", "unknown")}

        time.sleep(0.15)

    log.info("İndirme tamamlandı: %d/%d başarılı", success_count, len(target_symbols))
    summary_file = DATA_DIR / "download_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def load_cached_ohlcv(symbol: str) -> tuple[pd.DataFrame | None, dict]:
    cache_file = CACHE_DIR / f"{symbol}.parquet"
    meta_file = CACHE_DIR / f"{symbol}_meta.json"
    if not cache_file.exists() or not meta_file.exists():
        return None, {}
    df = pd.read_parquet(cache_file)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return df, meta


if __name__ == "__main__":
    download_and_cache_all()
