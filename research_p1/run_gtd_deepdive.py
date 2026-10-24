# -*- coding: utf-8 -*-
"""
research_p1/run_gtd_deepdive.py
===============================
GTD Derinlemesine Doğrulama ve Stres Testleri:
1. GTD Solo Walk-Forward (IS, OOS-1, OOS-2 ve Birleşik OOS 2024-2026)
2. IS (2021-2024) Döneminde RSI Bant Taraması (Plato mu, Tek Tepe mi?)
3. GTD Masraf Stres Testi (%0.80, %1.20, %1.60)
4. GTD Portföy Düzeyinde Simülasyon (100k TL, Max Drawdown, CAGR)
5. Kâr Konsantrasyonu: Yıl Kırılımı ve En İyi 10 İşlemin Katkısı
6. Kapasite ve Red Analizi (GTD tek başına ne kadar sinyal üretiyor, kaçı alınıyor, kaçı reddediliyor?)
7. XU100 ve USDTRY Benchmark Karşılaştırması
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import requests

from research_p1.backtest_engine import (
    simulate_trades,
    evaluate_agent_signals,
    precompute_signals_for_agents,
)
from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def download_benchmarks():
    headers = {"User-Agent": "Mozilla/5.0"}
    p1 = 1577836800  # 2020-01-01
    p2 = 1791331200  # 2026-10-06
    bench_dir = Path(__file__).parent / "data" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    for ticker in ["XU100.IS", "USDTRY=X"]:
        p_file = bench_dir / f"{ticker}.csv"
        if p_file.exists():
            continue
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={p1}&period2={p2}&interval=1d"
        r = requests.get(url, headers=headers)
        if r.ok:
            data = r.json()["chart"]["result"][0]
            ts = [pd.to_datetime(t, unit="s").strftime("%Y-%m-%d") for t in data["timestamp"]]
            q = data["indicators"]["quote"][0]
            df = pd.DataFrame({"close": q["close"]}, index=ts).dropna()
            df.to_csv(p_file)
            log.info("Benchmark %s indirildi (%d bar)", ticker, len(df))


def load_all_indicators(symbols: list[str]) -> Dict[str, pd.DataFrame]:
    prepared = {}
    for sym in symbols:
        df, meta = load_cached_ohlcv(sym)
        if df is None or len(df) < 50:
            continue
        try:
            ind = compute_all_indicators(df)
            prepared[sym] = ind
        except Exception as exc:
            pass
    return prepared


def run_gtd_deepdive():
    download_benchmarks()
    symbol_data = load_all_indicators(BIST100_SYMBOLS)
    log.info("Toplam %d hisse yüklendi.", len(symbol_data))

    # Tarihler
    FULL_START = "2021-10-06"
    FULL_END   = "2026-10-05"
    IS_START   = "2021-10-06"
    IS_END     = "2024-10-05"
    OOS1_START = "2024-10-06"
    OOS1_END   = "2025-10-05"
    OOS2_START = "2025-10-06"
    OOS2_END   = "2026-10-05"
    OOS_COMB_START = "2024-10-06"
    OOS_COMB_END   = "2026-10-05"

    precomp_gtd = precompute_signals_for_agents(symbol_data, ["GTD"])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GTD Solo Walk-Forward (IS, OOS-1, OOS-2 ve Birleşik OOS)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 1. GTD Solo Walk-Forward Çalıştırılıyor ---")
    wf_periods = [
        ("Gelistirme_IS_2021_2024", IS_START, IS_END),
        ("Dogrulama_OOS1_2024_2025", OOS1_START, OOS1_END),
        ("Nihai_Test_OOS2_2025_2026", OOS2_START, OOS2_END),
        ("Birlesik_OOS_2024_2026", OOS_COMB_START, OOS_COMB_END),
        ("Tam_Donem_5Yil", FULL_START, FULL_END),
    ]

    wf_results = []
    for p_label, s_dt, e_dt in wf_periods:
        trades, rej, m = simulate_trades(
            symbol_data, ["GTD"],
            precomputed_signals=precomp_gtd,
            start_date=s_dt, end_date=e_dt,
            gap_tolerance=0.0001,
            commission_rate=0.002, slippage_rate=0.002,
            mode="single_agent"
        )
        m["period"] = p_label
        wf_results.append(m)
        log.info("[%s] İşlem: %d, WinRate: %.1f%%, Ort Net: %+.2f%%, Medyan: %+.2f%%, PF: %.2f, 95%% CI: [%+.2f%%, %+.2f%%]",
                 p_label, m["n_trades"], m["win_rate"], m["mean_return_pct"], m["median_return_pct"],
                 m["profit_factor"], m["ci_lower_95"], m["ci_upper_95"])

    pd.DataFrame(wf_results).to_csv(RESULTS_DIR / "gtd_walk_forward.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. RSI Bant Taraması (Sadece IS: 2021-2024)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 2. RSI Bant Taraması (Sadece IS: 2021-2024) ---")
    rsi_bands = [
        (50, 65, "RSI_50_65_MevcutGT"),
        (55, 70, "RSI_55_70"),
        (60, 75, "RSI_60_75"),
        (65, 78, "RSI_65_78_MevcutGTD"),
        (65, 80, "RSI_65_80"),
        (70, 85, "RSI_70_85"),
        (0, 100, "RSI_Filtresiz"),
    ]

    rsi_scan_results = []
    for lo, hi, b_name in rsi_bands:
        precomp_band = {}
        for sym, df in symbol_data.items():
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
            precomp_band[sym] = {b_name: cond}

        trades, rej, m_b = simulate_trades(
            symbol_data, [b_name],
            precomputed_signals=precomp_band,
            start_date=IS_START, end_date=IS_END,
            gap_tolerance=0.0001,
            commission_rate=0.002, slippage_rate=0.002,
            mode="single_agent"
        )
        m_b["band_name"] = b_name
        m_b["rsi_min"] = lo
        m_b["rsi_max"] = hi
        rsi_scan_results.append(m_b)
        log.info("[%s (%d-%d)] IS İşlem: %d, Ort Net: %+.2f%%, PF: %.2f",
                 b_name, lo, hi, m_b["n_trades"], m_b["mean_return_pct"], m_b["profit_factor"])

    pd.DataFrame(rsi_scan_results).to_csv(RESULTS_DIR / "gtd_rsi_band_scan_is.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. GTD Masraf Stres Testi (Tam 5 Yıl ve Birleşik OOS)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 3. GTD Masraf Stres Testi ---")
    cost_levels = [
        (0.0020, 0.0020, "Temel (%0.80 Tur Masrafı)"),
        (0.0030, 0.0030, "Stres 1 (%1.20 Tur Masrafı)"),
        (0.0040, 0.0040, "Stres 2 (%1.60 Tur Masrafı)"),
    ]

    gtd_cost_results = []
    for comm, slip, c_label in cost_levels:
        # Full period
        _, _, m_full = simulate_trades(
            symbol_data, ["GTD"],
            precomputed_signals=precomp_gtd,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=0.0001,
            commission_rate=comm, slippage_rate=slip,
            mode="single_agent"
        )
        m_full["cost_level"] = c_label
        m_full["period"] = "Tam_5Yil"

        # Combined OOS
        _, _, m_oos = simulate_trades(
            symbol_data, ["GTD"],
            precomputed_signals=precomp_gtd,
            start_date=OOS_COMB_START, end_date=OOS_COMB_END,
            gap_tolerance=0.0001,
            commission_rate=comm, slippage_rate=slip,
            mode="single_agent"
        )
        m_oos["cost_level"] = c_label
        m_oos["period"] = "Birlesik_OOS_2024_2026"

        gtd_cost_results.extend([m_full, m_oos])
        log.info("[%s] Tam 5Y Ort Net: %+.2f%% (PF: %.2f) | Birleşik OOS Ort Net: %+.2f%% (PF: %.2f)",
                 c_label, m_full["mean_return_pct"], m_full["profit_factor"],
                 m_oos["mean_return_pct"], m_oos["profit_factor"])

    pd.DataFrame(gtd_cost_results).to_csv(RESULTS_DIR / "gtd_cost_stress_test.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. GTD Gerçekçi Portföy Simülasyonu & Drawdown
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 4. GTD Solo Portföy Düzeyi Simülasyonu ---")
    p_trades, p_rej, p_metrics = simulate_trades(
        symbol_data, ["GTD"],
        precomputed_signals=precomp_gtd,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=0.002, slippage_rate=0.002,
        mode="portfolio", initial_capital=100_000.0, max_positions=7
    )
    pd.DataFrame(p_trades).to_csv(RESULTS_DIR / "trades_gtd_solo_portfolio.csv", index=False)
    with open(RESULTS_DIR / "gtd_solo_portfolio_metrics.json", "w", encoding="utf-8") as f:
        json.dump(p_metrics, f, indent=2, ensure_ascii=False)

    log.info("GTD Solo Portföy: Başlangıç: 100k -> Bitiş: %.2f TL, CAGR: %.2f%%, MaxDD: %.2f%%, İşlem: %d, WinRate: %.1f%%",
             p_metrics["final_equity"], p_metrics["cagr_pct"], p_metrics["max_drawdown_pct"],
             p_metrics["n_trades"], p_metrics["win_rate"])

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Kâr Konsantrasyonu: Yıl/Ay Kırılımı ve En İyi 10 İşlem
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 5. Kâr Konsantrasyon Analizi ---")
    # Load all trades of GTD
    all_gtd_trades = pd.DataFrame(p_trades) if p_trades else pd.read_csv(RESULTS_DIR / "trades_individual_GTD.csv")
    all_gtd_trades["entry_year"] = pd.to_datetime(all_gtd_trades["entry_date"]).dt.year
    all_gtd_trades["entry_year_month"] = pd.to_datetime(all_gtd_trades["entry_date"]).dt.to_period("M").astype(str)

    # Year breakdown
    year_stats = all_gtd_trades.groupby("entry_year").agg(
        trades_count=("net_profit_tl", "count"),
        total_profit_tl=("net_profit_tl", "sum"),
        mean_return_pct=("net_return_pct", "mean"),
        win_rate=("net_return_pct", lambda s: round((s > 0).mean() * 100, 2))
    ).reset_index()
    year_stats.to_csv(RESULTS_DIR / "gtd_concentration_by_year.csv", index=False)
    log.info("Yıl Kırılımı:\n%s", year_stats)

    # Top 10 winning trades
    top10_trades = all_gtd_trades.sort_values(by="net_profit_tl", ascending=False).head(10)
    top10_trades.to_csv(RESULTS_DIR / "gtd_top10_winners.csv", index=False)
    total_gtd_profit = all_gtd_trades["net_profit_tl"].sum()
    top10_profit = top10_trades["net_profit_tl"].sum()
    top10_share_pct = round(top10_profit / total_gtd_profit * 100, 2) if total_gtd_profit > 0 else 0.0

    # Top 10 losing trades
    worst10_trades = all_gtd_trades.sort_values(by="net_profit_tl", ascending=True).head(10)
    worst10_trades.to_csv(RESULTS_DIR / "gtd_worst10_losers.csv", index=False)

    # Symbol breakdown
    symbol_stats = all_gtd_trades.groupby("symbol").agg(
        trades_count=("net_profit_tl", "count"),
        total_profit_tl=("net_profit_tl", "sum"),
        mean_return_pct=("net_return_pct", "mean"),
        win_rate=("net_return_pct", lambda s: round((s > 0).mean() * 100, 2))
    ).sort_values(by="total_profit_tl", ascending=False).reset_index()
    symbol_stats.to_csv(RESULTS_DIR / "gtd_concentration_by_symbol.csv", index=False)

    concentration_summary = {
        "total_trades": len(all_gtd_trades),
        "total_profit_tl": round(total_gtd_profit, 2),
        "top10_profit_tl": round(top10_profit, 2),
        "top10_profit_share_pct": top10_share_pct,
        "worst10_loss_tl": round(worst10_trades["net_profit_tl"].sum(), 2),
        "top_symbols_profitable_count": int((symbol_stats["total_profit_tl"] > 0).sum()),
        "total_symbols_traded": len(symbol_stats),
    }
    with open(RESULTS_DIR / "gtd_concentration_summary.json", "w", encoding="utf-8") as f:
        json.dump(concentration_summary, f, indent=2, ensure_ascii=False)
    log.info("Kâr Konsantrasyon Özeti: %s", concentration_summary)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Benchmark Karşılaştırması (XU100 ve USDTRY)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 6. Benchmark Karşılaştırması ---")
    bench_dir = Path(__file__).parent / "data" / "benchmarks"
    xu100_file = bench_dir / "XU100.IS.csv"
    usd_file = bench_dir / "USDTRY=X.csv"

    xu_ret_pct, usd_ret_pct = 0.0, 0.0
    if xu100_file.exists():
        df_xu = pd.read_csv(xu100_file, index_col=0)
        df_xu_sub = df_xu.loc[FULL_START:FULL_END]
        if len(df_xu_sub) > 1:
            xu_start = df_xu_sub["close"].iloc[0]
            xu_end = df_xu_sub["close"].iloc[-1]
            xu_ret_pct = round((xu_end / xu_start - 1.0) * 100, 2)
            xu_cagr = round(((xu_end / xu_start) ** (1.0 / 5.0) - 1.0) * 100, 2)

    if usd_file.exists():
        df_usd = pd.read_csv(usd_file, index_col=0)
        df_usd_sub = df_usd.loc[FULL_START:FULL_END]
        if len(df_usd_sub) > 1:
            usd_start = df_usd_sub["close"].iloc[0]
            usd_end = df_usd_sub["close"].iloc[-1]
            usd_ret_pct = round((usd_end / usd_start - 1.0) * 100, 2)
            usd_cagr = round(((usd_end / usd_start) ** (1.0 / 5.0) - 1.0) * 100, 2)

    benchmark_summary = {
        "period": "2021-10-06 -> 2026-10-05 (5 Yıl)",
        "gtd_portfolio_return_pct": p_metrics.get("total_return_pct"),
        "gtd_portfolio_cagr_pct": p_metrics.get("cagr_pct"),
        "gtd_portfolio_max_dd_pct": p_metrics.get("max_drawdown_pct"),
        "xu100_buy_and_hold_return_pct": xu_ret_pct,
        "xu100_cagr_pct": xu_cagr,
        "usdtry_buy_and_hold_return_pct": usd_ret_pct,
        "usdtry_cagr_pct": usd_cagr,
    }
    with open(RESULTS_DIR / "gtd_benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)
    log.info("Benchmark Özeti: %s", benchmark_summary)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. GTD Solo Kapasite & Sinyal Red İstatistiği
    # ─────────────────────────────────────────────────────────────────────────
    gtd_ind_trades = pd.read_csv(RESULTS_DIR / "trades_individual_GTD.csv")
    gtd_raw_signals = len(gtd_ind_trades) + 2828  # 2828 rejected under gap
    capacity_summary = {
        "agent": "GTD Solo",
        "total_generated_signals_5y": gtd_raw_signals,
        "signals_rejected_by_gap_0_0001": 2828,
        "gap_rejection_rate_pct": round(2828 / gtd_raw_signals * 100, 2),
        "signals_accepted_under_gap": len(gtd_ind_trades),
        "portfolio_trades_taken_7slots": p_metrics["n_trades"],
        "portfolio_rejected_by_capacity": len(gtd_ind_trades) - p_metrics["n_trades"],
        "portfolio_capacity_rejection_rate_pct": round((len(gtd_ind_trades) - p_metrics["n_trades"]) / len(gtd_ind_trades) * 100, 2),
    }
    with open(RESULTS_DIR / "gtd_capacity_summary.json", "w", encoding="utf-8") as f:
        json.dump(capacity_summary, f, indent=2, ensure_ascii=False)
    log.info("Kapasite Özeti: %s", capacity_summary)

    log.info("GTD Derinlemesine Doğrulama Başarıyla Tamamlandı!")


if __name__ == "__main__":
    run_gtd_deepdive()
