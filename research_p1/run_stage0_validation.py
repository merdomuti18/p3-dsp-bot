# -*- coding: utf-8 -*-
"""
research_p1/run_stage0_validation.py
====================================
P1 (GTD) Aşama 0: Paper Öncesi Doğrulama ve Kapı Testleri

Konfigürasyon:
- Çekirdek Ajan: GTD (RSI 65-78, mevcut filtreler)
- İşlem Başı Boyut: 10.000 TL (100.000 TL sermayenin %10'u)
- Eşzamanlı Pozisyon: En fazla 6 (Maksimum maruziyet <= %60)
- Maliyet: Tur başı %1.20 (Komisyon: %0.30, Slipaj: %0.30)
- Sinyal / Dolum: Sinyal günü kapanışta, ertesi gün açılışta emir

Aşama 0 Görevleri:
1. Günlük Equity Drawdown (Açık pozisyonlar MTM dahil, 10k TL boyutla)
2. Aylık Blok Bootstrap (Birleşik OOS 2024-2026 dönemi)
3. Hisse Dayanıklılığı (Top-5 ve Top-10 çıkarma, Likidite & Slipaj stresi)
4. Referans Karşılaştırma (XU100 al-tut, USD bazlı getiri, Hayatta kalma yanlılığı)
5. Aşama 0 Geçiş Kapısı Değerlendirmesi
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

CAPITAL = 100_000.0
POS_SIZE = 10_000.0
MAX_POS = 6
COMM_RATE = 0.0030  # %0.30
SLIP_RATE = 0.0030  # %0.30 -> Tur başı %1.20


def load_universe():
    symbol_data = {}
    for s in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            symbol_data[s] = compute_all_indicators(df)
    return symbol_data


def run_stage0():
    log.info("Aşama 0: BIST100 verisi yükleniyor...")
    symbol_data = load_universe()
    precomp_gtd = precompute_signals_for_agents(symbol_data, ["GTD"])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GÜNLÜK EQUITY DRAWDOWN (MTM, 10.000 TL Boyut, 6 Slot, %1.2 Maliyet)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 1. Günlük Equity DD (Portföy Modu: 10k TL Boyut, 6 Slot, %1.2 Maliyet) ---")
    p_trades, p_rej, p_metrics = simulate_trades(
        symbol_data, ["GTD"],
        precomputed_signals=precomp_gtd,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="portfolio", initial_capital=CAPITAL,
        max_positions=MAX_POS, position_size=POS_SIZE
    )

    eq_series = p_metrics.get("equity_series", [])
    df_equity = pd.DataFrame(eq_series)
    df_equity["peak"] = df_equity["equity"].cummax()
    df_equity["drawdown_pct"] = (df_equity["equity"] - df_equity["peak"]) / df_equity["peak"] * 100.0
    df_equity.to_csv(RESULTS_DIR / "stage0_daily_equity_gtd_10k.csv", index=False)

    max_daily_dd = float(df_equity["drawdown_pct"].min())
    final_equity = float(df_equity["equity"].iloc[-1])
    total_ret_pct = (final_equity / CAPITAL - 1.0) * 100.0
    cagr_pct = ((final_equity / CAPITAL) ** (1.0 / 5.0) - 1.0) * 100.0

    # OOS Dönemindeki Günlük Drawdown
    df_eq_oos = df_equity[(df_equity["date"] >= OOS_COMB_START) & (df_equity["date"] <= OOS_COMB_END)].copy()
    df_eq_oos["peak_oos"] = df_eq_oos["equity"].cummax()
    df_eq_oos["dd_oos"] = (df_eq_oos["equity"] - df_eq_oos["peak_oos"]) / df_eq_oos["peak_oos"] * 100.0
    max_daily_dd_oos = float(df_eq_oos["dd_oos"].min())

    log.info("Günlük MTM Equity DD (Tam 5 Yıl): %.2f%%", max_daily_dd)
    log.info("Günlük MTM Equity DD (Birleşik OOS 2024-2026): %.2f%%", max_daily_dd_oos)
    log.info("Final Equity: %.2f TL, Toplam Getiri: %+.2f%%, CAGR: %.2f%%", final_equity, total_ret_pct, cagr_pct)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. AYLIK BLOK BOOTSTRAP (BİRLEŞİK OOS 2024-2026)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 2. Aylık Blok Bootstrap (Birleşik OOS: 2024-10-06 -> 2026-10-05) ---")
    # Tekil trade bazında (kapasite kısıtlamasız saf GTD işlemleri, %1.2 maliyetle)
    t_oos, _, m_oos = simulate_trades(
        symbol_data, ["GTD"],
        precomputed_signals=precomp_gtd,
        start_date=OOS_COMB_START, end_date=OOS_COMB_END,
        gap_tolerance=0.0001,
        commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="single_agent", position_size=POS_SIZE
    )
    df_t_oos = pd.DataFrame(t_oos)
    df_t_oos["month"] = pd.to_datetime(df_t_oos["entry_date"]).dt.to_period("M").astype(str)

    months_list = sorted(df_t_oos["month"].unique().tolist())
    log.info("OOS Döneminde İşlem İçeren Ay Sayısı: %d", len(months_list))

    # 10.000 İterasyonluk Aylık Blok Bootstrap
    rng = np.random.default_rng(42)
    n_boot = 10_000
    boot_means = []

    month_groups = {m: df_t_oos[df_t_oos["month"] == m]["net_return_pct"].values for m in months_list}

    for _ in range(n_boot):
        # 24 aylık blokları yerine koyarak örnekle
        sampled_months = rng.choice(months_list, size=len(months_list), replace=True)
        sampled_returns = []
        for sm in sampled_months:
            sampled_returns.extend(month_groups[sm])
        boot_means.append(np.mean(sampled_returns))

    block_ci_lo = float(np.percentile(boot_means, 2.5))
    block_ci_hi = float(np.percentile(boot_means, 97.5))
    block_median = float(np.median(boot_means))

    log.info("Aylık Blok Bootstrap (10.000 Iterasyon):")
    log.info("  Ortalama Net Getiri: %+.2f%%", m_oos["mean_return_pct"])
    log.info("  95%% Blok Bootstrap GA: [%+.2f%%, %+.2f%%]", block_ci_lo, block_ci_hi)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. HİSSE DAYANIKLILIĞI, LİKİDİTE FİLTRESİ VE KÜÇÜK HİSSE SLİPAJ STRESİ
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 3. Hisse Dayanıklılığı & Likidite Stresi ---")
    # Tam dönem tekil işlemler (%1.2 maliyetle)
    t_full_single, _, m_full_single = simulate_trades(
        symbol_data, ["GTD"],
        precomputed_signals=precomp_gtd,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="single_agent", position_size=POS_SIZE
    )
    df_trades_full = pd.DataFrame(t_full_single)

    # Hisse bazında kâr sıralaması
    sym_profit = df_trades_full.groupby("symbol")["net_profit_tl"].sum().sort_values(ascending=False)
    top5_syms = sym_profit.head(5).index.tolist()
    top10_syms = sym_profit.head(10).index.tolist()

    # Top-5 çıkarılınca
    df_no_top5 = df_trades_full[~df_trades_full["symbol"].isin(top5_syms)]
    mean_ret_no_top5 = float(df_no_top5["net_return_pct"].mean())
    pf_no_top5 = float(df_no_top5[df_no_top5["net_profit_tl"] > 0]["net_profit_tl"].sum() /
                       abs(df_no_top5[df_no_top5["net_profit_tl"] < 0]["net_profit_tl"].sum()))
    total_tl_no_top5 = float(df_no_top5["net_profit_tl"].sum())

    # Top-10 çıkarılınca
    df_no_top10 = df_trades_full[~df_trades_full["symbol"].isin(top10_syms)]
    mean_ret_no_top10 = float(df_no_top10["net_return_pct"].mean())
    pf_no_top10 = float(df_no_top10[df_no_top10["net_profit_tl"] > 0]["net_profit_tl"].sum() /
                        abs(df_no_top10[df_no_top10["net_profit_tl"] < 0]["net_profit_tl"].sum()))
    total_tl_no_top10 = float(df_no_top10["net_profit_tl"].sum())

    log.info("Tüm İşlemler: %d adet, Ort: %+.2f%%, Toplam: %+.2f TL",
             len(df_trades_full), df_trades_full["net_return_pct"].mean(), df_trades_full["net_profit_tl"].sum())
    log.info("Top-5 Hisse Çıkarılınca (%s): %d adet, Ort Net: %+.2f%%, PF: %.2f, Toplam: %+.2f TL",
             ", ".join(top5_syms), len(df_no_top5), mean_ret_no_top5, pf_no_top5, total_tl_no_top5)
    log.info("Top-10 Hisse Çıkarılınca: %d adet, Ort Net: %+.2f%%, PF: %.2f, Toplam: %+.2f TL",
             len(df_no_top10), mean_ret_no_top10, pf_no_top10, total_tl_no_top10)

    # Likidite (ADV) Hesabı
    # Her hissenin ortalama günlük TL işlem hacmi (ADV = Close * Volume)
    adv_map = {}
    for sym, df in symbol_data.items():
        sub_c = df.loc[FULL_START:FULL_END]
        adv_map[sym] = float((sub_c["close"] * sub_c["volume"]).mean())

    df_trades_full["adv_tl"] = df_trades_full["symbol"].map(adv_map)
    trades_adv_under_50m = df_trades_full[df_trades_full["adv_tl"] < 50_000_000]
    trades_adv_over_50m = df_trades_full[df_trades_full["adv_tl"] >= 50_000_000]

    log.info("Likidite Kırılımı (50M TL Günlük Hacim Eşiği):")
    log.info("  ADV >= 50M TL: %d işlem, Ort Net: %+.2f%%", len(trades_adv_over_50m), trades_adv_over_50m["net_return_pct"].mean())
    log.info("  ADV < 50M TL: %d işlem, Ort Net: %+.2f%%", len(trades_adv_under_50m), trades_adv_under_50m["net_return_pct"].mean())

    # Küçük Hisselerde Daha Yüksek Slipaj Stresi
    # ADV < 50M TL olan hisselerde slipaj %0.30 yerine %0.60 (Tur başı maliyet %1.80)
    slip_map_stress = {sym: (0.0060 if adv < 50_000_000 else 0.0030) for sym, adv in adv_map.items()}
    _, _, m_slip_stress = simulate_trades(
        symbol_data, ["GTD"],
        precomputed_signals=precomp_gtd,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        slippage_rate_map=slip_map_stress,
        mode="single_agent", position_size=POS_SIZE
    )
    log.info("Küçük Hisselerde Ağır Slipaj Stresi (ADV<50M için %%1.80 tur masrafı):")
    log.info("  İşlem: %d, Ort Net: %+.2f%%, PF: %.2f, Toplam: %+.2f TL",
             m_slip_stress["n_trades"], m_slip_stress["mean_return_pct"], m_slip_stress["profit_factor"], m_slip_stress["total_profit_tl"])

    # ─────────────────────────────────────────────────────────────────────────
    # 4. REFERANS KARŞILAŞTIRMA & HAYATTA KALMA YANLILIĞI
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 4. Referans Karşılaştırma & Hayatta Kalma Yanlılığı ---")
    bench_dir = Path(__file__).parent / "data" / "benchmarks"
    df_xu = pd.read_csv(bench_dir / "XU100.IS.csv", index_col=0).loc[FULL_START:FULL_END]
    df_usd = pd.read_csv(bench_dir / "USDTRY=X.csv", index_col=0).loc[FULL_START:FULL_END]

    xu_ret = (df_xu["close"].iloc[-1] / df_xu["close"].iloc[0] - 1.0) * 100.0
    xu_cagr = ((df_xu["close"].iloc[-1] / df_xu["close"].iloc[0]) ** (1.0 / 5.0) - 1.0) * 100.0

    usd_ret = (df_usd["close"].iloc[-1] / df_usd["close"].iloc[0] - 1.0) * 100.0
    usd_cagr = ((df_usd["close"].iloc[-1] / df_usd["close"].iloc[0]) ** (1.0 / 5.0) - 1.0) * 100.0

    # Portföyün USD bazlı getirisi
    usd_start_rate = float(df_usd["close"].iloc[0])
    usd_end_rate = float(df_usd["close"].iloc[-1])
    port_usd_start = CAPITAL / usd_start_rate
    port_usd_end = final_equity / usd_end_rate
    port_usd_ret = (port_usd_end / port_usd_start - 1.0) * 100.0
    port_usd_cagr = ((port_usd_end / port_usd_start) ** (1.0 / 5.0) - 1.0) * 100.0

    log.info("BIST100 Al-Tut Getirisi: %+.2f%% (CAGR: %.2f%%)", xu_ret, xu_cagr)
    log.info("USD/TRY Değişimi: %+.2f%% (CAGR: %.2f%%)", usd_ret, usd_cagr)
    log.info("P1 (GTD 10k) USD Bazlı Değeri: $%.2f -> $%.2f (%+.2f%%, CAGR: %+.2f%%)",
             port_usd_start, port_usd_end, port_usd_ret, port_usd_cagr)

    # Hayatta Kalma Yanlılığı Kontrolü
    # 2021 öncesi işlem gören 93 hisse ile test
    symbols_5y = [s for s, df in symbol_data.items() if str(df.index[0].date()) <= "2021-10-06"]
    sym_data_surv = {s: symbol_data[s] for s in symbols_5y}
    precomp_surv = {s: precomp_gtd[s] for s in symbols_5y}

    _, _, m_surv = simulate_trades(
        sym_data_surv, ["GTD"],
        precomputed_signals=precomp_surv,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="single_agent", position_size=POS_SIZE
    )
    log.info("Hayatta Kalma Yanlılığı Kontrolü (Sadece 2021 öncesi 93 köklü hisse):")
    log.info("  İşlem: %d, Ort Net: %+.2f%%, PF: %.2f, Toplam: %+.2f TL",
             m_surv["n_trades"], m_surv["mean_return_pct"], m_surv["profit_factor"], m_surv["total_profit_tl"])

    # ─────────────────────────────────────────────────────────────────────────
    # 5. AŞAMA 0 GEÇİŞ KAPISI KONTROLÜ
    # ─────────────────────────────────────────────────────────────────────────
    # Eşik 1: Blok bootstrap GA alt sınırı >= 0
    gate1_pass = bool(block_ci_lo >= 0.0)

    # Eşik 2: Top-5 çıkarılınca ortalama net getiri > 0
    gate2_pass = bool(mean_ret_no_top5 > 0.0)

    # Eşik 3: Günlük DD 1.5 - 2.0 kat çarpanla -%20'yi aşmıyor mu?
    # Gerçek günlük DD = max_daily_dd
    # Stresli DD = max_daily_dd * 1.5 ve * 2.0
    stressed_dd_1_5 = max_daily_dd * 1.5
    stressed_dd_2_0 = max_daily_dd * 2.0
    gate3_pass = bool(stressed_dd_2_0 >= -20.0 or stressed_dd_1_5 >= -20.0)

    all_gates_pass = gate1_pass and gate2_pass and gate3_pass

    gate_summary = {
        "gate_1_block_bootstrap_oos": {
            "requirement": "Birleşik OOS 95% GA alt sınırı >= 0",
            "ci_lower": block_ci_lo,
            "ci_upper": block_ci_hi,
            "passed": gate1_pass,
        },
        "gate_2_top5_removed_robustness": {
            "requirement": "Top-5 hisse çıkarılınca ortalama net getiri > 0",
            "mean_return_no_top5_pct": mean_ret_no_top5,
            "profit_factor_no_top5": pf_no_top5,
            "passed": gate2_pass,
        },
        "gate_3_daily_equity_drawdown": {
            "requirement": "Günlük MTM DD 1.5x-2x çarpanla -%20 sınırını aşmamalı",
            "realized_daily_max_dd_pct": max_daily_dd,
            "realized_daily_max_dd_oos_pct": max_daily_dd_oos,
            "stressed_dd_1_5x_pct": round(stressed_dd_1_5, 2),
            "stressed_dd_2_0x_pct": round(stressed_dd_2_0, 2),
            "passed": gate3_pass,
        },
        "overall_stage_0_passed": all_gates_pass,
        "recommendation": "Paper Trading (Aşama 1)'e geçiş ONAYLANDI" if all_gates_pass else "Parametre revizyonu gerekli",
    }

    with open(RESULTS_DIR / "stage0_gate_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(gate_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("AŞAMA 0 GEÇİŞ KAPISI SONUÇLARI:")
    print("=" * 60)
    print(json.dumps(gate_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_stage0()
