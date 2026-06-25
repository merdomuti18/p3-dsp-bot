"""
compare_strategies.py — P3-DSP Faz 4
DSP-XOVER vs DSP-TREND: 10 sembol, farklı sektörler

Çalıştır:
    python compare_strategies.py
    python compare_strategies.py --bars 1500
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp_trend import (
    TrendParams,
    walk_forward_trend,
    compare_xover_trend,
)
from dsp_strategies import (
    XoverParams,
    walk_forward_validate,
)

# ---------------------------------------------------------------------------
# Sembol Evreni — 10 hisse, farklı sektörler
# ---------------------------------------------------------------------------

SYMBOLS = [
    ("GARAN",  "Bankacılık"),
    ("AKBNK",  "Bankacılık"),
    ("THYAO",  "Ulaşım"),
    ("EREGL",  "Demir Çelik"),
    ("ASELS",  "Savunma"),
    ("BIMAS",  "Perakende"),
    ("TUPRS",  "Enerji"),
    ("KCHOL",  "Holding"),
    ("TCELL",  "Telekom"),
    ("TOASO",  "Otomotiv"),
]

# Sabit parametreler (GARAN walk-forward kazananı)
XOVER_PARAMS = XoverParams(fast_period=15, slow_period=40, order=3)
TREND_PARAMS  = TrendParams(period=40, slope_window=10, order=3)

# ---------------------------------------------------------------------------
# Veri Çekici
# ---------------------------------------------------------------------------

def fetch_prices(symbol: str, bars: int) -> np.ndarray | None:
    try:
        import yfinance as yf
        ticker = f"{symbol}.IS"
        df = yf.Ticker(ticker).history(period="max")
        prices = df["Close"].dropna().values
        if len(prices) < 400:
            return None
        return prices[-bars:]
    except Exception as e:
        print(f"  ⚠️  {symbol} veri hatası: {e}")
        return None


# ---------------------------------------------------------------------------
# Ana Karşılaştırma
# ---------------------------------------------------------------------------

def run_comparison(bars: int = 1500):
    print(f"\n{'='*65}")
    print(f"  DSP-XOVER vs DSP-TREND — {len(SYMBOLS)} Sembol Karşılaştırması")
    print(f"  XOVER: fast={XOVER_PARAMS.fast_period} slow={XOVER_PARAMS.slow_period} order={XOVER_PARAMS.order}")
    print(f"  TREND: period={TREND_PARAMS.period} slope={TREND_PARAMS.slope_window} order={TREND_PARAMS.order}")
    print(f"{'='*65}\n")

    results = []

    for symbol, sector in SYMBOLS:
        print(f"  [{symbol:<6}] {sector:<15} ", end="", flush=True)
        prices = fetch_prices(symbol, bars)

        if prices is None:
            print("VERİ YOK — atlandı")
            continue

        try:
            xover_wf = walk_forward_validate(
                prices, XOVER_PARAMS,
                train_window=252, test_window=63, step=21
            )
            trend_wf = walk_forward_trend(
                prices, TREND_PARAMS,
                train_window=252, test_window=63, step=21
            )

            diff = trend_wf.consistency_ratio - xover_wf.consistency_ratio
            if diff > 0.05:
                verdict = "TREND ↑"
            elif diff < -0.05:
                verdict = "XOVER ↑"
            else:
                verdict = "BENZER"

            results.append({
                "symbol":  symbol,
                "sector":  sector,
                "xover_consist": xover_wf.consistency_ratio,
                "trend_consist": trend_wf.consistency_ratio,
                "xover_ret":     xover_wf.mean_test_return,
                "trend_ret":     trend_wf.mean_test_return,
                "xover_sharpe":  xover_wf.mean_sharpe,
                "trend_sharpe":  trend_wf.mean_sharpe,
                "xover_dd":      xover_wf.mean_max_dd,
                "trend_dd":      trend_wf.mean_max_dd,
                "consist_diff":  diff,
                "verdict":       verdict,
                "n_windows":     xover_wf.n_windows,
            })

            print(
                f"XOVER={xover_wf.consistency_ratio:.0%} "
                f"TREND={trend_wf.consistency_ratio:.0%}  "
                f"→ {verdict}"
            )

        except Exception as e:
            print(f"HATA: {e}")

    if not results:
        print("Hiç sonuç alınamadı.")
        return

    # -------------------------------------------------------------------
    # Detaylı Tablo
    # -------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print(f"  {'Sembol':<7} {'Sektör':<15} "
          f"{'X.Cons':>7} {'T.Cons':>7} "
          f"{'X.Ret':>7} {'T.Ret':>7} "
          f"{'X.Shrp':>7} {'T.Shrp':>7} "
          f"{'Karar'}")
    print(f"  {'─'*7} {'─'*15} "
          f"{'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} "
          f"{'─'*8}")

    for r in results:
        xc = f"{r['xover_consist']:.0%}"
        tc = f"{r['trend_consist']:.0%}"
        xr = f"{r['xover_ret']:+.1f}%"
        tr = f"{r['trend_ret']:+.1f}%"
        xs = f"{r['xover_sharpe']:+.3f}"
        ts = f"{r['trend_sharpe']:+.3f}"
        print(f"  {r['symbol']:<7} {r['sector']:<15} "
              f"{xc:>7} {tc:>7} "
              f"{xr:>7} {tr:>7} "
              f"{xs:>7} {ts:>7}  "
              f"{r['verdict']}")

    # -------------------------------------------------------------------
    # Özet İstatistikler
    # -------------------------------------------------------------------
    n = len(results)
    avg_x_consist = np.mean([r["xover_consist"] for r in results])
    avg_t_consist = np.mean([r["trend_consist"] for r in results])
    avg_x_ret     = np.mean([r["xover_ret"]     for r in results])
    avg_t_ret     = np.mean([r["trend_ret"]     for r in results])
    avg_x_sharpe  = np.mean([r["xover_sharpe"]  for r in results])
    avg_t_sharpe  = np.mean([r["trend_sharpe"]  for r in results])
    avg_x_dd      = np.mean([r["xover_dd"]      for r in results])
    avg_t_dd      = np.mean([r["trend_dd"]      for r in results])

    trend_wins  = sum(1 for r in results if r["verdict"] == "TREND ↑")
    xover_wins  = sum(1 for r in results if r["verdict"] == "XOVER ↑")
    similar     = sum(1 for r in results if r["verdict"] == "BENZER")

    print(f"\n{'─'*65}")
    print(f"  ORTALAMA ({n} sembol)")
    print(f"  {'Metrik':<22} {'XOVER':>10} {'TREND':>10} {'Fark':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Consistency':22} {avg_x_consist:>10.1%} {avg_t_consist:>10.1%} "
          f"{avg_t_consist - avg_x_consist:>+10.1%}")
    print(f"  {'Ort. Test Getiri':22} {avg_x_ret:>+10.2f}% {avg_t_ret:>+10.2f}% "
          f"{avg_t_ret - avg_x_ret:>+10.2f}%")
    print(f"  {'Ort. Sharpe':22} {avg_x_sharpe:>+10.3f} {avg_t_sharpe:>+10.3f} "
          f"{avg_t_sharpe - avg_x_sharpe:>+10.3f}")
    print(f"  {'Ort. Max DD':22} {avg_x_dd:>+10.2f}% {avg_t_dd:>+10.2f}% "
          f"{avg_t_dd - avg_x_dd:>+10.2f}%")

    print(f"\n  Sembol bazında karar:")
    print(f"  TREND üstün  : {trend_wins}/{n} sembol")
    print(f"  XOVER üstün  : {xover_wins}/{n} sembol")
    print(f"  Benzer       : {similar}/{n} sembol")

    # -------------------------------------------------------------------
    # Nihai Karar
    # -------------------------------------------------------------------
    consist_diff = avg_t_consist - avg_x_consist
    sharpe_diff  = avg_t_sharpe  - avg_x_sharpe

    print(f"\n{'='*65}")
    if consist_diff > 0.05 and sharpe_diff > 0:
        final = "✅ TREND ENTEGRE ET — birleşik skor önerilir"
    elif consist_diff > 0.05 and sharpe_diff <= 0:
        final = "⚠️  TREND consistency iyi ama Sharpe düşük — dikkatli entegre"
    elif consist_diff < -0.05:
        final = "❌ XOVER belirgin üstün — TREND ekleme"
    else:
        final = "🔄 BENZER — birleşik skor opsiyonel, zarar vermez"
    print(f"  NİHAİ KARAR: {final}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=1500, help="Geçmiş bar sayısı")
    args = parser.parse_args()
    run_comparison(bars=args.bars)