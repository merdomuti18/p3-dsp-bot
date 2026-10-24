# -*- coding: utf-8 -*-
"""
P1 / GTD deneyleri: (1) tek basina walk-forward, (3) RSI bandi taramasi (sadece IS),
(4) maliyet stresi + portfoy max drawdown + yil kirilimi + top-10 katkisi.

ADAPTOR: run_gtd() backtest_engine'e baglanmistir.
Beklenen cikti: trade DataFrame, kolonlar:
  symbol, entry_date, exit_date, net_ret_pct (masraf dusulmus %), pnl_tl
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from research_p1.backtest_engine import (
    simulate_trades,
    precompute_signals_for_agents,
)
from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS = ("2021-10-06", "2024-10-05")
OOS1 = ("2024-10-06", "2025-10-05")
OOS2 = ("2025-10-06", "2026-10-05")
FULL = ("2021-10-06", "2026-10-05")
BASE_RSI = (65, 78)
BASE_COST = (0.20, 0.20)  # (komisyon, slipaj) yuzde, tek yon
RNG = np.random.default_rng(42)

_SYMBOL_DATA: Optional[Dict[str, pd.DataFrame]] = None
_PRECOMP_GTD: Optional[Dict[str, Dict[str, pd.Series]]] = None


def _get_symbol_data() -> Dict[str, pd.DataFrame]:
    global _SYMBOL_DATA
    if _SYMBOL_DATA is None:
        _SYMBOL_DATA = {}
        for s in BIST100_SYMBOLS:
            df, meta = load_cached_ohlcv(s)
            if df is not None and len(df) >= 50:
                _SYMBOL_DATA[s] = compute_all_indicators(df)
    return _SYMBOL_DATA


def _get_precomp_gtd() -> Dict[str, Dict[str, pd.Series]]:
    global _PRECOMP_GTD
    if _PRECOMP_GTD is None:
        sym_data = _get_symbol_data()
        _PRECOMP_GTD = precompute_signals_for_agents(sym_data, ["GTD"])
    return _PRECOMP_GTD


def run_gtd(start: str, end: str, rsi_band: Tuple[int, int] = BASE_RSI, cost: Tuple[float, float] = BASE_COST) -> pd.DataFrame:
    """ADAPTOR: backtest_engine çağrısı."""
    sym_data = _get_symbol_data()
    comm_rate = cost[0] / 100.0
    slip_rate = cost[1] / 100.0
    lo, hi = rsi_band

    if (lo, hi) == (65, 78):
        precomp = _get_precomp_gtd()
        agent_key = "GTD"
    else:
        precomp = {}
        agent_key = f"GTD_{lo}_{hi}"
        for sym, df in sym_data.items():
            c = df["close"]
            ema8, ema21, ema50, ema200 = df["ema8"], df["ema21"], df["ema50"], df["ema200"]
            adx, di_p, di_n, cmf, rel_vol = df["adx"], df["di_p"], df["di_n"], df["cmf"], df["rel_vol"]
            rsi = df["rsi"]
            cond = (
                (ema8 > ema21) & (ema21 > ema50) &
                (ema200.isna() | (c > ema200)) &
                (rsi >= lo) & (rsi <= hi) &
                (adx > 25) &
                (di_p > di_n) &
                (cmf > 0.05) &
                (rel_vol >= 1.2)
            )
            precomp[sym] = {agent_key: cond}

    trades, rejected, metrics = simulate_trades(
        sym_data, [agent_key],
        precomputed_signals=precomp,
        start_date=start, end_date=end,
        gap_tolerance=0.0001,
        commission_rate=comm_rate, slippage_rate=slip_rate,
        mode="single_agent"
    )

    df_trades = pd.DataFrame(trades)
    if len(df_trades) > 0:
        df_trades["net_ret_pct"] = df_trades["net_return_pct"]
        df_trades["pnl_tl"] = df_trades["net_profit_tl"]
    else:
        df_trades = pd.DataFrame(columns=["symbol", "entry_date", "exit_date", "net_ret_pct", "pnl_tl"])
    return df_trades


def boot_ci(x, n=5000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 5:
        return (np.nan, np.nan)
    means = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def summarize(t, label):
    if len(t) == 0:
        return dict(label=label, n=0)
    r = t["net_ret_pct"]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    lo, hi = boot_ci(r)
    return dict(label=label, n=len(t), win=(r > 0).mean() * 100,
                mean=r.mean(), median=r.median(),
                pf=gp / gl if gl > 0 else np.inf,
                total_tl=t["pnl_tl"].sum(), ci_lo=lo, ci_hi=hi)


def max_drawdown(trades, capital=100_000.0):
    """Basit portfoy DD: kapanis tarihine gore kumulatif TL pnl uzerinden."""
    t = trades.sort_values("exit_date")
    eq = capital + t["pnl_tl"].cumsum()
    peak = eq.cummax()
    return ((eq - peak) / peak).min() * 100


def exp1_walk_forward():
    rows, oos = [], []
    for name, (s, e) in [("IS", IS), ("OOS1", OOS1), ("OOS2", OOS2)]:
        t = run_gtd(s, e)
        rows.append(summarize(t, f"GTD_{name}"))
        if name != "IS":
            oos.append(t)
    rows.append(summarize(pd.concat(oos), "GTD_OOS1+OOS2"))
    return pd.DataFrame(rows)


def exp3_rsi_scan():
    """SADECE IS. Secim kurali: en yuksek hucre degil, 3x3 komsuluk ortalamasi en yuksek hucre (plato)."""
    los, his = [55, 60, 65, 70], [72, 75, 78, 82, 85]
    rows = []
    for lo in los:
        for hi in his:
            if hi <= lo + 5:
                continue
            t = run_gtd(*IS, rsi_band=(lo, hi))
            s = summarize(t, f"{lo}-{hi}")
            s.update(lo=lo, hi=hi)
            rows.append(s)
    df = pd.DataFrame(rows)
    grid = df.pivot(index="lo", columns="hi", values="mean")
    smooth = grid.copy()
    for i, lo in enumerate(grid.index):
        for hi in grid.columns:
            nb = grid.iloc[max(0, i - 1):i + 2][[c for c in grid.columns if abs(c - hi) <= 3 or c == hi]]
            smooth.loc[lo, hi] = np.nanmean(nb.values)
    df["smoothed_mean"] = [smooth.loc[r.lo, r.hi] for r in df.itertuples()]
    return df.sort_values("smoothed_mean", ascending=False), grid


def exp4_cost_stress_and_breakdown():
    rows, extra = [], {}
    for c in [(0.20, 0.20), (0.30, 0.30), (0.40, 0.40)]:
        # 1. Continuous 5-year run (taban kontrolü: 497 işlem, +1.24%)
        t_cont = run_gtd(*FULL, cost=c)
        s_cont = summarize(t_cont, f"cost_{c[0]}+{c[1]}_roundtrip_{2*(c[0]+c[1]):.1f}_continuous5Y")
        s_cont["max_dd_pct"] = max_drawdown(t_cont)
        rows.append(s_cont)

        # 2. Split concatenated run (IS + OOS1 + OOS2)
        t_split = pd.concat([run_gtd(*p, cost=c) for p in (IS, OOS1, OOS2)])
        s_split = summarize(t_split, f"cost_{c[0]}+{c[1]}_roundtrip_{2*(c[0]+c[1]):.1f}_concat_IS_OOS")
        s_split["max_dd_pct"] = max_drawdown(t_split)
        rows.append(s_split)

        if c == BASE_COST:
            t_base = t_cont.copy()
            t_base["year"] = pd.to_datetime(t_base["exit_date"]).dt.year
            extra["by_year"] = t_base.groupby("year")["net_ret_pct"].agg(["count", "mean", "median"])
            top10 = t_base.nlargest(10, "pnl_tl")["pnl_tl"].sum()
            extra["top10_share_of_profit"] = top10 / t_base["pnl_tl"].sum()
            extra["by_symbol_top"] = t_base.groupby("symbol")["pnl_tl"].sum().nlargest(10)

    return pd.DataFrame(rows), extra


if __name__ == "__main__":
    print("=" * 60)
    print("GTD Deneyleri Calistiriliyor...")
    print("=" * 60)

    # E1: Walk-forward
    print("\n[E1] Walk-Forward Calistiriliyor...")
    df_e1 = exp1_walk_forward()
    df_e1.to_csv(RESULTS_DIR / "exp_e1_gtd_walk_forward.csv", index=False)
    print(df_e1.to_string())

    # E3: RSI Scan (Sadece IS)
    print("\n[E3] RSI Bandi Taramasi (Sadece IS) Calistiriliyor...")
    scan, grid = exp3_rsi_scan()
    scan.to_csv(RESULTS_DIR / "exp_e3_rsi_band_scan_IS.csv", index=False)
    grid.to_csv(RESULTS_DIR / "exp_e3_rsi_grid_mean_IS.csv")
    print("\n--- RSI Grid (Ortalama Net Getiri %) ---")
    print(grid.to_string())
    print("\n--- En Iyi 5 Plato Hucresi ---")
    print(scan[["label", "n", "mean", "smoothed_mean", "pf"]].head(5).to_string())

    # E4: Cost stress & breakdown
    print("\n[E4] Maliyet Stresi ve Kirilimlar Calistiriliyor...")
    cost, extra = exp4_cost_stress_and_breakdown()
    cost.to_csv(RESULTS_DIR / "exp_e4_gtd_cost_stress.csv", index=False)
    extra["by_year"].to_csv(RESULTS_DIR / "exp_e4_gtd_by_year.csv")
    extra["by_symbol_top"].to_csv(RESULTS_DIR / "exp_e4_gtd_top_symbols.csv")

    print("\n--- Maliyet Stres Testi ---")
    print(cost[["label", "n", "win", "mean", "median", "pf", "total_tl", "max_dd_pct", "ci_lo", "ci_hi"]].to_string())

    print("\nTop-10 islemin toplam kara orani:", round(extra["top10_share_of_profit"], 3))
    print("\nYil bazinda kirilim:")
    print(extra["by_year"].to_string())
    print("\nEn karli 10 hisse:")
    print(extra["by_symbol_top"].to_string())
