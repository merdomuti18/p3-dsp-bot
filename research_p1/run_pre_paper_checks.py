# -*- coding: utf-8 -*-
"""
research_p1/run_pre_paper_checks.py
===================================
Paper Trading Öncesi 3 Kritik Analiz & Temiz Şablon Üretimi:

1. ALPHA_B_SIMPLE Walk-Forward Analizi:
   - IS (2021-2024), OOS1 (2024-2025), OOS2 (2025-2026) ve Birleşik OOS (2024-2026).
   - Filtreler tüm örneklemde seçilmişti; OOS'ta performansı test edilir.

2. 3-Ajanlı Portföyün Yıl Yıl ve IS/OOS Kırılımı (20k Boyut, 5 Slot, %1.2 Maliyet):
   - Yıl bazında: 2021, 2022, 2023, 2024, 2025, 2026 (özellikle 2023-2025 medyan incelemesi).
   - IS / OOS1 / OOS2 dönem kırılımı.

3. GT'nin RSI Dilim Analizi (GT_NO_RSI):
   - Gerçekleşen işlemlerin RSI dilimlerine göre getirisi:
     * RSI < 65 (Eski GT bölgesi)
     * 65 <= RSI <= 78 (GTD bölgesi)
     * RSI > 78 (Aşırı momentum)
   - Kârı üreten işlemlerin gerçekten hangi RSI diliminden geldiğinin tespiti.

4. Güncellenmiş Temiz Paper Trading Log Şablonu:
   - Sahte örnek satır çıkarılmış, istenen tüm alanlar eklenmiş boş CSV.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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
from research_p1.run_p1_revision_backtest import evaluate_custom_agent_signals, simulate_priority_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

COMM_RATE = 0.0030  # %0.30
SLIP_RATE = 0.0030  # %0.30 -> Tur başı %1.20


def calc_ci(returns: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    if len(returns) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot = [rng.choice(returns, size=len(returns), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def run_checks():
    log.info("BIST100 verisi yükleniyor...")
    symbol_data = {}
    for s in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            symbol_data[s] = compute_all_indicators(df)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ALPHA_B_SIMPLE WALK-FORWARD ANALİZİ
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 1. ALPHA_B_SIMPLE Walk-Forward Analizi Başlıyor ---")
    precomp_alpha = {}
    for sym, df in symbol_data.items():
        precomp_alpha[sym] = {"ALPHA_B_SIMPLE": evaluate_custom_agent_signals(df, "ALPHA_B_SIMPLE")}

    wf_alpha_periods = [
        ("IS_2021_2024", IS_START, IS_END),
        ("OOS1_2024_2025", OOS1_START, OOS1_END),
        ("OOS2_2025_2026", OOS2_START, OOS2_END),
        ("OOS_COMB_2024_2026", OOS_COMB_START, OOS_COMB_END),
        ("FULL_5Y", FULL_START, FULL_END),
    ]

    alpha_wf_rows = []
    for p_name, s_dt, e_dt in wf_alpha_periods:
        trades, rej, m = simulate_trades(
            symbol_data, ["ALPHA_B_SIMPLE"],
            precomputed_signals=precomp_alpha,
            start_date=s_dt, end_date=e_dt,
            gap_tolerance=0.0001,
            commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
            mode="single_agent", position_size=20_000.0
        )
        rets = np.array([t["net_return_pct"] for t in trades]) if trades else np.array([])
        ci_lo, ci_hi = calc_ci(rets) if len(rets) >= 5 else (np.nan, np.nan)
        alpha_wf_rows.append({
            "period": p_name,
            "n_trades": len(trades),
            "win_rate": m["win_rate"],
            "mean_ret_pct": m["mean_return_pct"],
            "median_ret_pct": m["median_return_pct"],
            "profit_factor": m["profit_factor"],
            "total_pnl_tl": m["total_profit_tl"],
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })

    df_alpha_wf = pd.DataFrame(alpha_wf_rows)
    df_alpha_wf.to_csv(RESULTS_DIR / "alpha_b_simple_walk_forward.csv", index=False)
    log.info("ALPHA_B_SIMPLE Walk-Forward:\n%s", df_alpha_wf.to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 3-AJANLI PORTFÖYÜN YIL YIL VE IS/OOS KIRILIMI
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 2. 3-Ajanlı Portföy (20k Boyut, 5 Slot, %1.2 Maliyet) Yıl & IS/OOS Kırılımı ---")
    trades_file = RESULTS_DIR / "p1_rev_S1_20kBoyut_Maliyet1.2pct_trades.csv"
    if not trades_file.exists():
        log.warning("Trades dosyası bulunamadı, yeniden üretiliyor...")
        from research_p1.run_p1_revision_backtest import run_full_p1_revision
        run_full_p1_revision()

    df_trades = pd.read_csv(trades_file)
    df_trades["entry_dt"] = pd.to_datetime(df_trades["entry_date"])
    df_trades["year"] = df_trades["entry_dt"].dt.year

    # Yıl bazında kırılım
    year_rows = []
    for yr, grp in df_trades.groupby("year"):
        rets = grp["net_return_pct"].values
        pnls = grp["net_profit_tl"].values
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls < 0].sum())
        pf = round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0)
        ci_lo, ci_hi = calc_ci(rets)
        year_rows.append({
            "year": yr,
            "n_trades": len(grp),
            "win_rate": round(float((rets > 0).mean() * 100), 2),
            "mean_ret_pct": round(float(rets.mean()), 4),
            "median_ret_pct": round(float(np.median(rets)), 4),
            "profit_factor": pf,
            "total_pnl_tl": round(float(pnls.sum()), 2),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })

    df_by_year = pd.DataFrame(year_rows)
    df_by_year.to_csv(RESULTS_DIR / "p1_3agents_by_year_breakdown.csv", index=False)
    log.info("3-Ajanlı Portföy Yıl Kırılımı:\n%s", df_by_year.to_string())

    # IS / OOS Kırılımı
    period_conditions = [
        ("IS_2021_2024", df_trades["entry_date"].between(IS_START, IS_END)),
        ("OOS1_2024_2025", df_trades["entry_date"].between(OOS1_START, OOS1_END)),
        ("OOS2_2025_2026", df_trades["entry_date"].between(OOS2_START, OOS2_END)),
        ("OOS_COMB_2024_2026", df_trades["entry_date"].between(OOS_COMB_START, OOS_COMB_END)),
    ]
    period_rows = []
    for p_name, cond in period_conditions:
        grp = df_trades[cond]
        rets = grp["net_return_pct"].values
        pnls = grp["net_profit_tl"].values
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls < 0].sum())
        pf = round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0)
        ci_lo, ci_hi = calc_ci(rets)
        period_rows.append({
            "period": p_name,
            "n_trades": len(grp),
            "win_rate": round(float((rets > 0).mean() * 100), 2),
            "mean_ret_pct": round(float(rets.mean()), 4),
            "median_ret_pct": round(float(np.median(rets)), 4),
            "profit_factor": pf,
            "total_pnl_tl": round(float(pnls.sum()), 2),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })

    df_by_period = pd.DataFrame(period_rows)
    df_by_period.to_csv(RESULTS_DIR / "p1_3agents_is_oos_breakdown.csv", index=False)
    log.info("3-Ajanlı Portföy IS/OOS Kırılımı:\n%s", df_by_period.to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # 3. GT (RSI'SIZ) GERÇEKLEŞEN İŞLEMLERİNİN RSI DİLİM ANALİZİ
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 3. GT (RSI'sız) İşlemlerinin Gerçekleşen RSI Dilim Analizi ---")
    precomp_gt_no_rsi = {}
    for sym, df in symbol_data.items():
        precomp_gt_no_rsi[sym] = {"GT_NO_RSI": evaluate_custom_agent_signals(df, "GT_NO_RSI")}

    trades_gt_raw, _, m_gt_raw = simulate_trades(
        symbol_data, ["GT_NO_RSI"],
        precomputed_signals=precomp_gt_no_rsi,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="single_agent", position_size=20_000.0
    )

    # Her işlemin sinyal günündeki RSI değerini eşleştir
    enriched_gt_trades = []
    for t in trades_gt_raw:
        sym = t["symbol"]
        sig_dt = pd.to_datetime(t["signal_date"])
        df = symbol_data.get(sym)
        if df is not None and sig_dt in df.index:
            t["signal_rsi"] = float(df.loc[sig_dt, "rsi"])
        else:
            t["signal_rsi"] = np.nan
        enriched_gt_trades.append(t)

    df_gt_trades = pd.DataFrame(enriched_gt_trades).dropna(subset=["signal_rsi"])

    # Dilimleme: <65 (Eski GT), 65-78 (GTD bölgesi), >78 (Aşırı momentum)
    bin_conditions = [
        ("RSI < 65 (Eski GT Bandi & Alti)", df_gt_trades["signal_rsi"] < 65.0),
        ("65 <= RSI <= 78 (GTD Bandi)", (df_gt_trades["signal_rsi"] >= 65.0) & (df_gt_trades["signal_rsi"] <= 78.0)),
        ("RSI > 78 (Asiri Momentum)", df_gt_trades["signal_rsi"] > 78.0),
        ("RSI >= 65 (GTD + Asiri)", df_gt_trades["signal_rsi"] >= 65.0),
        ("Tum_Islemler_GT_NO_RSI", pd.Series(True, index=df_gt_trades.index)),
    ]

    rsi_bin_rows = []
    for b_label, cond in bin_conditions:
        sub = df_gt_trades[cond]
        rets = sub["net_return_pct"].values
        pnls = sub["net_profit_tl"].values
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls < 0].sum())
        pf = round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0)
        ci_lo, ci_hi = calc_ci(rets)
        rsi_bin_rows.append({
            "rsi_bin": b_label,
            "n_trades": len(sub),
            "pct_of_all_trades": round(len(sub) / len(df_gt_trades) * 100, 2),
            "win_rate": round(float((rets > 0).mean() * 100), 2),
            "mean_ret_pct": round(float(rets.mean()), 4),
            "median_ret_pct": round(float(np.median(rets)), 4),
            "profit_factor": pf,
            "total_pnl_tl": round(float(pnls.sum()), 2),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })

    df_rsi_bins = pd.DataFrame(rsi_bin_rows)
    df_rsi_bins.to_csv(RESULTS_DIR / "gt_no_rsi_trade_performance_by_rsi_bins.csv", index=False)
    log.info("GT_NO_RSI İşlemlerinin RSI Dilim Analizi:\n%s", df_rsi_bins.to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. GÜNCELLENMİŞ TEMİZ PAPER TRADING ŞABLONU
    # ─────────────────────────────────────────────────────────────────────────
    clean_columns = [
        "sinyal_tarihi",
        "giris_tarihi",
        "sembol",
        "birincil_ajan",
        "tum_ajanlar",
        "ajan_bazli_rel_vol_sirasi",
        "rel_vol",
        "sinyal_kapanis_fiyati",
        "beklenen_acilis_fiyati",
        "gerceklesen_acilis_fiyati",
        "gap_orani_pct",
        "gerceklesen_slipaj_pct",
        "onaylanan_lot",
        "tutar_tl",
        "kapasite_durumu",
        "xu100_rejim_boga",
        "cikis_tarihi",
        "cikis_fiyati",
        "cikis_nedeni",
        "net_getiri_pct",
        "net_pnl_tl",
        "net_pnl_10k_esdegeri_tl",
        "reddedilen_sonradan_getiri_pct",
        "notlar"
    ]
    df_template = pd.DataFrame(columns=clean_columns)
    df_template.to_csv(RESULTS_DIR / "paper_trading_log_template.csv", index=False)
    log.info("Temiz Paper Trading Şablonu yazıldı (Örnek satırsız, tam format): %s", RESULTS_DIR / "paper_trading_log_template.csv")


if __name__ == "__main__":
    run_checks()
