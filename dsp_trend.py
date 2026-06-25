"""
dsp_trend.py — P3-DSP Faz 4: DSP-TREND Stratejisi
====================================================
DSP-XOVER'dan bağımsız, tek Butterworth low-pass + eğim (slope) tabanlı
trend takip stratejisi.

Sinyal mantığı:
  filtered  = causal Butterworth(prices, period)
  slope     = (filtered[-1] - filtered[-slope_window]) / filtered[-slope_window]
  signal    = +1 if slope > threshold else -1
  score     = slope  (normalize edilmiş eğim oranı)

Faz 4 görevi:
  - Walk-forward ile XOVER parametreleriyle karşılaştır
  - Katkı kanıtlanırsa scanner_dsp.py'e entegre et

KRİTİK KURAL: filtfilt KULLANILMAZ — sadece apply_causal / lfilter.
"""

from __future__ import annotations

import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_core import (
    design_butterworth,
    apply_causal,
    warmup_length,
)


# ---------------------------------------------------------------------------
# 1. Parametre Seti
# ---------------------------------------------------------------------------

@dataclass
class TrendParams:
    """
    DSP-TREND parametre seti.

    period       : Butterworth low-pass cutoff (gün). Gürültüyü eler.
    slope_window : Eğim hesabı için geriye bakış (bar). Trend hızını belirler.
    threshold    : Minimum eğim oranı — sinyal üretmek için geçilmesi gereken eşik.
                   0.0 → sadece yön yeterli, >0 → daha seçici.
    order        : Butterworth mertebesi (DSP-XOVER ile tutarlı: 3).
    max_lag_ratio: Faz gecikme bütçesi (XOVER ile aynı kural).
    """
    period: float       = 30.0
    slope_window: int   = 10
    threshold: float    = 0.0
    order: int          = 3
    max_lag_ratio: float = 0.40

    def label(self) -> str:
        return f"p{int(self.period)}_s{self.slope_window}_t{self.threshold:.3f}"


# Faz 4 tarama grid'i
TREND_PARAM_GRID = [
    TrendParams(period=p, slope_window=s, threshold=0.0, order=3)
    for p, s in itertools.product(
        [20, 25, 30, 35, 40],   # period seçenekleri
        [5, 7, 10, 15],          # slope_window seçenekleri
    )
]


# ---------------------------------------------------------------------------
# 2. Sinyal Üreteci
# ---------------------------------------------------------------------------

@dataclass
class TrendSignal:
    """
    DSP-TREND sinyal dizisi — XoverSignal ile paralel yapı.
    """
    signal:       np.ndarray    # +1 (long) / -1 (flat), prices ile aynı uzunluk
    filtered:     np.ndarray    # Butterworth low-pass çıktısı
    slope_series: np.ndarray    # Her bar için normalize eğim
    valid_from:   int           # Warm-up tamamlandıktan sonraki ilk geçerli bar
    params:       TrendParams


def generate_trend_signal(
    prices: np.ndarray,
    params: Optional[TrendParams] = None,
) -> TrendSignal:
    """
    Tek Butterworth low-pass + eğim tabanlı trend sinyali üretir.

    Tüm filtreler causal — look-ahead yok.

    Parameters
    ----------
    prices : Kapanış fiyatları (1-D, en eski → en yeni).
    params : TrendParams, None ise default değerler kullanılır.

    Returns
    -------
    TrendSignal
    """
    if params is None:
        params = TrendParams()

    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    b, a = design_butterworth(params.period, params.order)
    filtered = apply_causal(b, a, prices)

    # Warm-up: en az order*period bar + slope_window ek
    vf = warmup_length(params.order, params.period) + params.slope_window

    # Eğim serisi: normalize oransal eğim
    slope_series = np.zeros(n)
    for i in range(params.slope_window, n):
        base = filtered[i - params.slope_window]
        if abs(base) > 1e-9:
            slope_series[i] = (filtered[i] - base) / base
        else:
            slope_series[i] = 0.0

    # Sinyal
    signal = np.where(slope_series > params.threshold, 1.0, -1.0)
    signal[:vf] = 0.0   # warm-up barlarını sıfırla

    return TrendSignal(
        signal=signal,
        filtered=filtered,
        slope_series=slope_series,
        valid_from=vf,
        params=params,
    )


# ---------------------------------------------------------------------------
# 3. Backtest Motoru
# ---------------------------------------------------------------------------

@dataclass
class TrendBacktestResult:
    """
    TrendBacktest.run() dönüş tipi — XoverBacktest ile aynı alanlar.
    """
    total_return_pct:   float
    buy_hold_return_pct: float
    sharpe:             float
    max_drawdown_pct:   float
    n_trades:           int
    win_rate:           float
    n_bars:             int
    params:             TrendParams
    equity_curve:       np.ndarray = field(repr=False)
    daily_returns:      np.ndarray = field(repr=False)


class TrendBacktest:
    """
    DSP-TREND için tek pencere backtest motoru.
    XoverBacktest ile birebir aynı varsayımlar:
      - Kapanışta sinyal, ertesi bar etkili (simplifikasyon: aynı bar)
      - Kısa pozisyon yok (paper portföy)
      - İşlem maliyeti: 0
    """

    def __init__(self, params: Optional[TrendParams] = None):
        self.params = params or TrendParams()

    def run(self, prices: np.ndarray) -> TrendBacktestResult:
        prices = np.asarray(prices, dtype=float)
        sig = generate_trend_signal(prices, self.params)
        vf  = sig.valid_from

        p = prices[vf:]
        s = sig.signal[vf:]

        if len(p) < 10:
            raise ValueError(
                f"Warm-up sonrası yeterli bar yok (vf={vf}, n={len(prices)})"
            )

        log_ret  = np.diff(np.log(p))
        position = np.where(s[:-1] > 0, 1.0, 0.0)   # flat = 0, long = 1

        strat_ret = position * log_ret
        bh_ret    = log_ret

        equity    = np.exp(np.cumsum(strat_ret))
        bh_equity = np.exp(np.cumsum(bh_ret))

        total_ret  = (equity[-1] - 1) * 100
        bh_ret_pct = (bh_equity[-1] - 1) * 100

        sharpe = (
            strat_ret.mean() / (strat_ret.std() + 1e-9) * np.sqrt(252)
        )

        running_max = np.maximum.accumulate(equity)
        drawdown    = (equity - running_max) / running_max
        max_dd      = drawdown.min() * 100

        trades = self._count_trades(position, strat_ret)

        return TrendBacktestResult(
            total_return_pct=round(total_ret, 2),
            buy_hold_return_pct=round(bh_ret_pct, 2),
            sharpe=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            n_trades=trades["n_trades"],
            win_rate=round(trades["win_rate"], 3),
            n_bars=len(log_ret),
            params=self.params,
            equity_curve=equity,
            daily_returns=strat_ret,
        )

    @staticmethod
    def _count_trades(position: np.ndarray, daily_returns: np.ndarray) -> dict:
        changes    = np.diff(np.concatenate([[0], position]))
        entry_bars = np.where(changes == 1)[0]
        exit_bars  = np.where(changes == -1)[0]
        n_trades   = len(entry_bars)
        if n_trades == 0:
            return {"n_trades": 0, "win_rate": 0.0}
        wins = 0
        for i, entry in enumerate(entry_bars):
            exit_ = exit_bars[i] if i < len(exit_bars) else len(daily_returns)
            if daily_returns[entry:exit_].sum() > 0:
                wins += 1
        return {"n_trades": n_trades, "win_rate": wins / n_trades}


# ---------------------------------------------------------------------------
# 4. Walk-Forward — tek parametre seti
# ---------------------------------------------------------------------------

@dataclass
class TrendWFResult:
    """Tek pencere walk-forward sonucu."""
    window_idx:     int
    test_return_pct: float
    test_sharpe:    float
    test_max_dd_pct: float
    bh_return_pct:  float
    params:         TrendParams


@dataclass
class TrendWFSummary:
    """Tüm pencereler istatistiksel özet."""
    params:             TrendParams
    n_windows:          int
    mean_test_return:   float
    std_test_return:    float
    positive_windows:   int
    mean_sharpe:        float
    mean_max_dd:        float
    consistency_ratio:  float
    is_consistent:      bool        # > 0.55
    windows:            list[TrendWFResult]

    def one_line(self) -> str:
        status = "✅" if self.is_consistent else "⚠️ "
        return (
            f"{status} {self.params.label():<20} "
            f"ret={self.mean_test_return:+6.2f}%  "
            f"std={self.std_test_return:5.2f}%  "
            f"sharpe={self.mean_sharpe:+5.3f}  "
            f"consist={self.consistency_ratio:.0%} "
            f"({self.positive_windows}/{self.n_windows})"
        )


def walk_forward_trend(
    prices: np.ndarray,
    params: Optional[TrendParams] = None,
    train_window: int = 252,
    test_window:  int = 63,
    step:         int = 21,
) -> TrendWFSummary:
    """
    DSP-TREND için rolling walk-forward validasyon.
    XoverBacktest'teki walk_forward_validate() ile aynı pencere mantığı.
    """
    if params is None:
        params = TrendParams()

    prices = np.asarray(prices, dtype=float)
    bt     = TrendBacktest(params)
    wf_min = warmup_length(params.order, params.period) + params.slope_window + 5
    windows = []
    idx = window_num = 0

    while idx + train_window + test_window <= len(prices):
        test_start = idx + train_window
        test_end   = test_start + test_window

        # Warm-up için test penceresinden önce context al
        context_start = max(0, test_start - wf_min)
        context_seg   = prices[context_start:test_end]

        if len(context_seg) > wf_min + 10:
            try:
                r = bt.run(context_seg)
                # Sadece test bölümünü yeniden değerlendir
                pure_test = prices[test_start:test_end]
                if len(pure_test) > wf_min:
                    r = bt.run(pure_test)
                windows.append(TrendWFResult(
                    window_idx=window_num,
                    test_return_pct=r.total_return_pct,
                    test_sharpe=r.sharpe,
                    test_max_dd_pct=r.max_drawdown_pct,
                    bh_return_pct=r.buy_hold_return_pct,
                    params=params,
                ))
                window_num += 1
            except ValueError:
                pass

        idx += step

    if not windows:
        raise ValueError(
            f"Hiç pencere oluşturulamadı — prices uzunluğu ({len(prices)}) "
            f"en az {train_window + test_window + wf_min} olmalı."
        )

    returns = np.array([w.test_return_pct for w in windows])
    sharpes = np.array([w.test_sharpe     for w in windows])
    dds     = np.array([w.test_max_dd_pct for w in windows])
    pos     = int((returns > 0).sum())
    ratio   = pos / len(windows)

    return TrendWFSummary(
        params=params,
        n_windows=len(windows),
        mean_test_return=round(float(returns.mean()), 3),
        std_test_return=round(float(returns.std()),   3),
        positive_windows=pos,
        mean_sharpe=round(float(sharpes.mean()), 3),
        mean_max_dd=round(float(dds.mean()),     3),
        consistency_ratio=round(ratio, 3),
        is_consistent=(ratio > 0.55),
        windows=windows,
    )


# ---------------------------------------------------------------------------
# 5. Parametre Tarama — tüm grid
# ---------------------------------------------------------------------------

def scan_trend_params(
    prices: np.ndarray,
    param_grid: Optional[list[TrendParams]] = None,
    train_window: int = 252,
    test_window:  int = 63,
    step:         int = 21,
    verbose:      bool = True,
) -> list[TrendWFSummary]:
    """
    TREND_PARAM_GRID üzerinde walk-forward tarama.
    Consistency ratio'ya göre sıralı sonuç listesi döner.

    Parameters
    ----------
    prices     : Kapanış fiyat serisi (en az ~800 bar önerilir).
    param_grid : None ise TREND_PARAM_GRID kullanılır.
    verbose    : Her parametre setini yazdır.

    Returns
    -------
    list[TrendWFSummary] — consistency_ratio azalan sırada.
    """
    if param_grid is None:
        param_grid = TREND_PARAM_GRID

    results: list[TrendWFSummary] = []

    if verbose:
        print(f"\nDSP-TREND Parametre Taraması ({len(param_grid)} kombinasyon)")
        print(f"Fiyat serisi: {len(prices)} bar | "
              f"train={train_window} test={test_window} step={step}")
        print("─" * 75)

    for i, p in enumerate(param_grid):
        try:
            summary = walk_forward_trend(
                prices, p, train_window, test_window, step
            )
            results.append(summary)
            if verbose:
                print(f"  [{i+1:2d}/{len(param_grid)}] {summary.one_line()}")
        except ValueError as e:
            if verbose:
                print(f"  [{i+1:2d}/{len(param_grid)}] {p.label():<20} ATLA: {e}")

    # Consistency ratio → ikinci kriter: ortalama getiri
    results.sort(key=lambda x: (x.consistency_ratio, x.mean_test_return), reverse=True)

    if verbose:
        print("\n" + "─" * 75)
        print("🏆 TOP 5 PARAMETRE SETİ (consistency_ratio ↓)")
        print("─" * 75)
        for r in results[:5]:
            print(f"  {r.one_line()}")
        print()

    return results


# ---------------------------------------------------------------------------
# 6. XOVER vs TREND Karşılaştırma
# ---------------------------------------------------------------------------

def compare_xover_trend(
    prices: np.ndarray,
    xover_params=None,
    trend_params: Optional[TrendParams] = None,
    train_window: int = 252,
    test_window:  int = 63,
    step:         int = 21,
) -> dict:
    """
    DSP-XOVER ve DSP-TREND'i aynı veri üzerinde karşılaştırır.
    Entegrasyon kararı için temel metrikler yan yana.

    Returns dict:
      xover   : WalkForwardSummary
      trend   : TrendWFSummary
      verdict : 'trend_better' | 'xover_better' | 'similar'
    """
    from dsp_strategies import walk_forward_validate, XoverParams

    xp = xover_params or XoverParams(fast_period=15, slow_period=40, order=3)
    tp = trend_params  or TrendParams(period=30, slope_window=10)

    xover_wf = walk_forward_validate(prices, xp, train_window, test_window, step)
    trend_wf  = walk_forward_trend(prices, tp, train_window, test_window, step)

    # Karar: consistency ratio farkı > 0.05 ise anlamlı üstünlük
    diff = trend_wf.consistency_ratio - xover_wf.consistency_ratio
    if diff > 0.05:
        verdict = "trend_better"
    elif diff < -0.05:
        verdict = "xover_better"
    else:
        verdict = "similar"

    return {
        "xover": xover_wf,
        "trend": trend_wf,
        "verdict": verdict,
        "consistency_diff": round(diff, 3),
    }


# ---------------------------------------------------------------------------
# 7. CLI — Hızlı Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P3-DSP DSP-TREND Faz 4")
    parser.add_argument("--symbol",  default="GARAN",  help="Sembol (yfinance)")
    parser.add_argument("--bars",    type=int, default=1200, help="Geçmiş bar sayısı")
    parser.add_argument("--compare", action="store_true", help="XOVER ile karşılaştır")
    args = parser.parse_args()

    # Veri: yfinance veya sentetik
    try:
        import yfinance as yf
        ticker = f"{args.symbol}.IS"
        df = yf.Ticker(ticker).history(period="max")
        prices = df["Close"].dropna().values[-args.bars:]
        print(f"Veri: {ticker} — {len(prices)} bar (yfinance)")
    except Exception:
        rng = np.random.default_rng(42)
        prices = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.018, args.bars))
        print(f"Veri: Sentetik — {len(prices)} bar")

    if args.compare:
        print(f"\n{'='*55}")
        print("DSP-XOVER vs DSP-TREND Karşılaştırması")
        print(f"{'='*55}")
        cmp = compare_xover_trend(prices)
        xw  = cmp["xover"]
        tw  = cmp["trend"]
        print(f"\n  {'Metrik':<22} {'XOVER':>10} {'TREND':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10}")
        print(f"  {'Consistency':22} {xw.consistency_ratio:>10.0%} {tw.consistency_ratio:>10.0%}")
        print(f"  {'Ort. Test Getiri':22} {xw.mean_test_return:>+10.2f}% {tw.mean_test_return:>+10.2f}%")
        print(f"  {'Std Test Getiri':22} {xw.std_test_return:>10.2f}% {tw.std_test_return:>10.2f}%")
        print(f"  {'Ort. Sharpe':22} {xw.mean_sharpe:>+10.3f} {tw.mean_sharpe:>+10.3f}")
        print(f"  {'Ort. Max DD':22} {xw.mean_max_dd:>10.2f}% {tw.mean_max_dd:>10.2f}%")
        print(f"\n  Karar: {cmp['verdict'].upper()} "
              f"(consistency fark: {cmp['consistency_diff']:+.1%})")
    else:
        results = scan_trend_params(prices, verbose=True)
        best = results[0]
        print(f"En iyi parametre: period={best.params.period}, "
              f"slope_window={best.params.slope_window}")
        print(f"Consistency: {best.consistency_ratio:.0%} | "
              f"Ort. getiri: {best.mean_test_return:+.2f}% | "
              f"Sharpe: {best.mean_sharpe:+.3f}")