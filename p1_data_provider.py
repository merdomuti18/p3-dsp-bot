# -*- coding: utf-8 -*-
"""
p1_data_provider.py — Dakikalık bar / açılış VWAP sağlayıcı arayüzü
==================================================================
Paper dolum vekili, seans açılışının ilk 5 bir-dakikalık barından hesaplanır.

Tipik fiyat tanımı (belgelenmiş, kopya yok):
    typical_price = (high + low + close) / 3

VWAP:
    VWAP = Σ(typical_price × volume) / Σ volume
    İlk 5 bar: Europe/Istanbul 10:00–10:05 (yarım günde de 10:00 başlangıç).

Kaynaklar (repoda mevcut olanlar; uydurma yok):
  1) Yahoo / yfinance  (requirements.txt yfinance==1.5.1)
     - 1m bar: evet
     - geçmiş derinlik: ~7–8 takvim günü (yfinance 1m max ≈ 8 gün)
     - BIST gecikmesi: ~15 dk; canlı paper için 10:05 penceresinde kullanılabilir
  2) TradingView tvDatafeed  (requirements.txt tradingview-datafeed)
     - scanner_dsp.py Interval.in_daily kullanır; 1m Interval.in_1_minute ile mümkün
     - geçmiş: oturum/plana bağlı, bu repoda 1m canlı çekim yok — denenen yedek
  3) TradingView screener (mott_fiyat.py / tradingview-screener)
     - yalnızca anlık last/close; 1m OHLCV bar YOK
  4) research_p1.data_downloader
     - yalnızca 1d OHLCV; 1m YOK
  5) borsapy
     - bu repoda yok (requirements / import yok)
  6) Osmanlı / aracı kurum API
     - bu repoda broker emir veya 1m bar istemcisi yok

Veri alınamazsa çağıran taraf ERROR yazar, notlar="VWAP verisi yok".
1m geçmiş kısa olduğu için sonradan doldurulamaz.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger("p1_data_provider")

TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")
TYPICAL_PRICE_DEFINITION = "(high + low + close) / 3"
OPENING_MINUTE_BARS = 5
SESSION_OPEN_TSI = time(10, 0)
VWAP_WINDOW_END_TSI = time(10, 5)

PROVIDER_REPORT = [
    {
        "kaynak": "yfinance / Yahoo Finance",
        "modül": "yfinance (portfoy_yonetici, scanner_p1, mott_fiyat yedek)",
        "1m_bar": True,
        "gecmis": "yaklaşık 7–8 takvim günü (interval=1m)",
        "kullanim": "varsayılan 1m sağlayıcı",
    },
    {
        "kaynak": "tvDatafeed / TradingView",
        "modül": "tv_auth + scanner_dsp.DataClient",
        "1m_bar": True,
        "gecmis": "oturum ve plana bağlı; repoda yalnızca günlük get_hist kullanılıyor",
        "kullanim": "Yahoo başarısızsa yedek denemesi",
    },
    {
        "kaynak": "tradingview-screener",
        "modül": "mott_fiyat.tv_fiyatlar",
        "1m_bar": False,
        "gecmis": "anlık last; bar geçmişi yok",
        "kullanim": "1m VWAP için uygun değil",
    },
    {
        "kaynak": "research_p1.data_downloader",
        "modül": "Yahoo chart API interval=1d",
        "1m_bar": False,
        "gecmis": "2020–2026 günlük",
        "kullanim": "1m VWAP için uygun değil",
    },
    {
        "kaynak": "borsapy",
        "modül": None,
        "1m_bar": False,
        "gecmis": "repoda yok",
        "kullanim": "bağlanmadı",
    },
    {
        "kaynak": "Osmanlı / aracı kurum",
        "modül": None,
        "1m_bar": False,
        "gecmis": "repoda yok",
        "kullanim": "bağlanmadı",
    },
]


def typical_price_series(bars: pd.DataFrame) -> pd.Series:
    """typical_price = (high + low + close) / 3"""
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    return (high + low + close) / 3.0


def opening_vwap(bars: pd.DataFrame, n_bars: int = OPENING_MINUTE_BARS) -> Optional[float]:
    """İlk n 1m bar üzerinden hacim ağırlıklı tipik fiyat VWAP. Veri yoksa None."""
    if bars is None or len(bars) < n_bars:
        return None
    cols = {c.lower(): c for c in bars.columns}
    need = ("high", "low", "close", "volume")
    if any(k not in cols for k in need):
        return None
    renamed = bars.rename(columns={cols[k]: k for k in need})
    sub = renamed.head(n_bars)
    vol = sub["volume"].astype(float)
    denom = float(vol.sum())
    if denom <= 0:
        return None
    tp = typical_price_series(sub)
    return float((tp * vol).sum() / denom)


def first_n_session_bars(bars: pd.DataFrame, session_day: date, n_bars: int = OPENING_MINUTE_BARS) -> pd.DataFrame:
    """TSİ 10:00 sonrası aynı gün barlarından ilk n tanesini keser."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ_ISTANBUL, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(TZ_ISTANBUL)
    day_mask = df.index.date == session_day
    after_open = df.index.time >= SESSION_OPEN_TSI
    return df.loc[day_mask & after_open].head(n_bars)


class MinuteBarProvider(ABC):
    """1 dakikalık OHLCV soyut arayüzü."""

    name: str = "abstract"

    @abstractmethod
    def fetch_opening_1m_bars(self, symbol: str, session_day: date, n_bars: int = OPENING_MINUTE_BARS) -> Optional[pd.DataFrame]:
        """Sembol için seans açılışı 1m barları. Başarısızsa None (sessiz boş kolon yok; çağıran ERROR yazar)."""


class YahooMinuteBarProvider(MinuteBarProvider):
    name = "yfinance_1m"

    def fetch_opening_1m_bars(self, symbol: str, session_day: date, n_bars: int = OPENING_MINUTE_BARS) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except Exception as exc:
            log.error("yfinance import edilemedi: %s", exc)
            return None
        ticker = symbol if str(symbol).endswith(".IS") else f"{symbol}.IS"
        try:
            raw = yf.Ticker(ticker).history(period="5d", interval="1m", auto_adjust=False)
        except Exception as exc:
            log.error("Yahoo 1m çekilemedi %s: %s", ticker, exc)
            return None
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={c: c.lower() for c in raw.columns})
        cut = first_n_session_bars(raw, session_day, n_bars=n_bars)
        if len(cut) < n_bars:
            return None
        return cut[["open", "high", "low", "close", "volume"]]


class TradingViewMinuteBarProvider(MinuteBarProvider):
    name = "tvdatafeed_1m"

    def fetch_opening_1m_bars(self, symbol: str, session_day: date, n_bars: int = OPENING_MINUTE_BARS) -> Optional[pd.DataFrame]:
        try:
            from tv_auth import is_authenticated, make_tv_client
            from tvDatafeed import Interval
        except Exception as exc:
            log.error("tvDatafeed/tv_auth kullanılamadı: %s", exc)
            return None
        try:
            client, _diag = make_tv_client(require_auth=True)
            if not is_authenticated(client):
                return None
            sym = str(symbol).replace(".IS", "").upper()
            raw = client.get_hist(sym, "BIST", interval=Interval.in_1_minute, n_bars=400)
        except Exception as exc:
            log.error("TV 1m çekilemedi %s: %s", symbol, exc)
            return None
        if raw is None or getattr(raw, "empty", True):
            return None
        raw = raw.rename(columns={c: str(c).lower() for c in raw.columns})
        cut = first_n_session_bars(raw, session_day, n_bars=n_bars)
        if len(cut) < n_bars:
            return None
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in cut.columns]
        return cut[keep] if len(keep) == 5 else None


class CompositeMinuteBarProvider(MinuteBarProvider):
    """Yahoo birincil, TV yedek. Hiçbiri vermezse None."""

    name = "composite_yahoo_tv"

    def __init__(self, providers: Optional[list] = None):
        self.providers = providers or [YahooMinuteBarProvider(), TradingViewMinuteBarProvider()]

    def fetch_opening_1m_bars(self, symbol: str, session_day: date, n_bars: int = OPENING_MINUTE_BARS) -> Optional[pd.DataFrame]:
        for prov in self.providers:
            bars = prov.fetch_opening_1m_bars(symbol, session_day, n_bars=n_bars)
            if bars is not None and len(bars) >= n_bars:
                return bars
        return None


_DEFAULT_PROVIDER: Optional[MinuteBarProvider] = None


def get_default_provider() -> MinuteBarProvider:
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = CompositeMinuteBarProvider()
    return _DEFAULT_PROVIDER


def set_default_provider(provider: Optional[MinuteBarProvider]) -> None:
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider


def fetch_opening_minute_bars(symbol: str, session_day: date, n_bars: int = OPENING_MINUTE_BARS) -> Optional[pd.DataFrame]:
    return get_default_provider().fetch_opening_1m_bars(symbol, session_day, n_bars=n_bars)
