"""
scanner_dsp.py — P3-DSP Sembol Tarayıcı
=========================================
Çalıştır:
  python scanner_dsp.py                  # Mock veri
  python scanner_dsp.py --real           # TradingView (günlük)
  python scanner_dsp.py --real --yf      # yfinance
  python scanner_dsp.py --real --symbol GARAN
"""

from __future__ import annotations

import numpy as np
import time
import logging
from dataclasses import dataclass
from datetime import date
from typing import Protocol, Optional
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dsp_core import design_butterworth, apply_causal, warmup_length, AdaptiveFilterState
from dsp_strategies import XoverParams, XoverBacktest, generate_xover_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sembol Listesi
# ---------------------------------------------------------------------------

def load_symbols(path: str = None) -> list[str]:
    """hisse_listesi.txt dosyasından sembol yükler. # yorum, boş satır atlanır."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "hisse_listesi.txt")
    if not os.path.exists(path):
        logger.warning(f"hisse_listesi.txt bulunamadı — yedek liste kullanılıyor")
        return _FALLBACK_SYMBOLS
    symbols = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.append(line.upper())
    return symbols if symbols else _FALLBACK_SYMBOLS

_FALLBACK_SYMBOLS = [
    "GARAN", "AKBNK", "YKBNK", "HALKB", "VAKBN",
    "THYAO", "PGSUS", "TUPRS", "EREGL", "KRDMD",
    "ASELS", "AEFES", "BIMAS", "MGROS", "SOKM",
    "KCHOL", "SAHOL", "SISE",  "PETKM", "TOASO",
]

SYMBOL_UNIVERSE_SAMPLE = load_symbols()

# ---------------------------------------------------------------------------
# Veri Adaptörleri
# ---------------------------------------------------------------------------

class DataAdapter(Protocol):
    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]: ...


class MockDataAdapter:
    """Sentetik veri — test için."""
    def __init__(self, seed_base: int = 42):
        self.seed_base = seed_base

    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]:
        seed = self.seed_base + hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        vol, returns = 0.018, []
        for i in range(lookback_days):
            shock = rng.normal(0, vol)
            vol = np.clip(0.0003 + 0.85*vol + 0.12*abs(shock), 0.008, 0.055)
            returns.append(0.0008*np.sin(i/80) + rng.choice([-1,1])*0.0002 + shock)
        return 10.0 * np.cumprod(1 + np.array(returns))


class TvDataAdapter:
    """
    TradingView günlük veri — anonim bağlantı.
    Kurulum: pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
    """
    def __init__(self, username: str = "", password: str = "", exchange: str = "BIST"):
        self.exchange = exchange
        self._username = username
        self._password = password
        self._tv = None

    def _connect(self):
        if self._tv is not None:
            return
        from tvDatafeed import TvDatafeed, Interval
        self._Interval = Interval
        self._tv = TvDatafeed(self._username, self._password) if self._username \
                   else TvDatafeed()

    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]:
        try:
            self._connect()
            df = self._tv.get_hist(
                symbol=symbol, exchange=self.exchange,
                interval=self._Interval.in_daily, n_bars=lookback_days + 50,
            )
            if df is None or df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df = df.drop(columns=["symbol"], errors="ignore")
            close = df["close"].dropna().values.astype(float)
            return close[-lookback_days:] if len(close) >= 60 else None
        except Exception as e:
            logger.warning(f"{symbol} TV hatası: {e}")
            return None


class YfDataAdapter:
    """yfinance yedek adaptör. Sembol: 'GARAN' → otomatik 'GARAN.IS' eklenir."""
    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]:
        try:
            import yfinance as yf, pandas as pd
            ticker = symbol if symbol.endswith(".IS") else f"{symbol}.IS"
            df = yf.Ticker(ticker).history(period=f"{lookback_days+50}d")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            close = df["Close"].dropna().values.astype(float)
            return close[-lookback_days:] if len(close) >= 60 else None
        except Exception as e:
            logger.warning(f"{symbol} yf hatası: {e}")
            return None

# ---------------------------------------------------------------------------
# Sinyal Skoru
# ---------------------------------------------------------------------------

@dataclass
class SymbolScore:
    symbol: str
    signal: int
    score: float
    crossover_margin: float
    momentum_3w: float
    fast_line: float
    slow_line: float
    warmup_ok: bool
    scan_time_ms: float
    error: Optional[str] = None

    @property
    def is_long(self) -> bool:
        return self.signal == 1 and self.warmup_ok

    @property
    def signal_str(self) -> str:
        if not self.warmup_ok: return "WARM-UP"
        return "LONG" if self.signal == 1 else "FLAT"


def score_symbol(symbol: str, prices: np.ndarray, params: XoverParams) -> SymbolScore:
    t0 = time.perf_counter()
    prices = np.asarray(prices, dtype=float)
    sig = generate_xover_signal(prices, params)
    vf  = sig.valid_from
    warmup_ok    = len(prices) > vf + 5
    last_fast    = float(sig.fast_line[-1])
    last_slow    = float(sig.slow_line[-1])
    last_signal  = int(sig.signal[-1])
    crossover_margin = (last_fast - last_slow) / (last_slow + 1e-9)
    momentum_3w  = float(np.log(prices[-1] / prices[-16])) if len(prices) >= 16 else 0.0
    score = 0.7 * crossover_margin + 0.3 * momentum_3w
    if last_signal < 0:
        score -= 0.05
    return SymbolScore(
        symbol=symbol, signal=last_signal if warmup_ok else 0,
        score=score, crossover_margin=crossover_margin, momentum_3w=momentum_3w,
        fast_line=last_fast, slow_line=last_slow, warmup_ok=warmup_ok,
        scan_time_ms=round((time.perf_counter()-t0)*1000, 2),
    )

# ---------------------------------------------------------------------------
# Tarayıcı
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    scan_date: date
    total_symbols: int
    scanned: int
    errors: int
    long_signals: int
    top_longs: list[SymbolScore]
    all_scores: list[SymbolScore]
    elapsed_sec: float
    params: XoverParams

    def summary_line(self) -> str:
        return (
            f"[P3-DSP] {self.scan_date} | "
            f"Taranan: {self.scanned}/{self.total_symbols} | "
            f"Long sinyal: {self.long_signals} | "
            f"Top-5: {', '.join(s.symbol for s in self.top_longs[:5])} | "
            f"{self.elapsed_sec:.1f}s"
        )


class DspScanner:
    def __init__(self, data_adapter, params=None, lookback_days=300, top_n=10, symbols=None):
        self.adapter       = data_adapter
        self.params        = params or XoverParams(fast_period=15, slow_period=40, order=3)
        self.lookback_days = lookback_days
        self.top_n         = top_n
        self.symbols       = symbols or SYMBOL_UNIVERSE_SAMPLE

    def run(self, verbose=False) -> ScanResult:
        t0, today, scores, errors = time.perf_counter(), date.today(), [], 0
        for symbol in self.symbols:
            try:
                prices = self.adapter.fetch(symbol, self.lookback_days)
                if prices is None or len(prices) < 60:
                    scores.append(SymbolScore(
                        symbol=symbol, signal=0, score=-999,
                        crossover_margin=0, momentum_3w=0,
                        fast_line=0, slow_line=0, warmup_ok=False,
                        scan_time_ms=0, error="Yetersiz veri",
                    ))
                    errors += 1
                else:
                    sc = score_symbol(symbol, prices, self.params)
                    scores.append(sc)
                if verbose:
                    sc = scores[-1]
                    flag = "▲ LONG" if sc.is_long else "  flat"
                    print(f"  {flag}  {symbol:<10} skor={sc.score:+.4f}  "
                          f"marj={sc.crossover_margin:+.3f}  mom={sc.momentum_3w:+.3f}")
            except Exception as e:
                logger.error(f"{symbol} hata: {e}")
                errors += 1

        long_scores = sorted([s for s in scores if s.is_long], key=lambda x: x.score, reverse=True)
        all_sorted  = sorted(scores, key=lambda x: x.score, reverse=True)
        return ScanResult(
            scan_date=today, total_symbols=len(self.symbols),
            scanned=len(self.symbols)-errors, errors=errors,
            long_signals=len(long_scores), top_longs=long_scores[:self.top_n],
            all_scores=all_sorted, elapsed_sec=round(time.perf_counter()-t0, 2),
            params=self.params,
        )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="P3-DSP Sembol Tarayıcı")
    parser.add_argument("--top",     type=int,  default=10)
    parser.add_argument("--symbol",  type=str,  default=None)
    parser.add_argument("--real",    action="store_true", help="Gerçek veri")
    parser.add_argument("--yf",      action="store_true", help="yfinance kullan")
    parser.add_argument("--tv-user", type=str,  default="")
    parser.add_argument("--tv-pass", type=str,  default="")
    args = parser.parse_args()

    if args.real and args.yf:
        adapter = YfDataAdapter()
        print("Veri: yfinance")
    elif args.real:
        adapter = TvDataAdapter(username=args.tv_user, password=args.tv_pass)
        print("Veri: TradingView (günlük)")
    else:
        adapter = MockDataAdapter()
        print("Veri: Mock (sentetik)")

    symbols = [args.symbol.upper()] if args.symbol else load_symbols()

    scanner = DspScanner(
        data_adapter=adapter, top_n=args.top, symbols=symbols,
        params=XoverParams(fast_period=15, slow_period=40, order=3),
    )

    print(f"P3-DSP Tarama başlıyor... ({len(scanner.symbols)} sembol)\n")
    result = scanner.run(verbose=True)

    print(f"\n{'─'*55}")
    print(result.summary_line())
    print(f"{'─'*55}")
    print(f"\nTop {args.top} Long Adayı:")
    print(f"  {'Sembol':<12} {'Skor':>8} {'Marj':>8} {'Mom':>8}  Signal")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}  {'─'*6}")
    for s in result.top_longs:
        print(f"  {s.symbol:<12} {s.score:>+8.4f} "
              f"{s.crossover_margin:>+8.3f} {s.momentum_3w:>+8.3f}  {s.signal_str}")