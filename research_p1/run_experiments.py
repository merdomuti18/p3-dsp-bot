# -*- coding: utf-8 -*-
"""
research_p1/run_experiments.py
==============================
6 Ajanın Tüm Araştırma Deneylerini Çalıştırır:
1. Ön Hazırlık: BIST100 hisselerinin indikatör serilerini önbellekleme
2. Deney A1: 6 Ajan Bireysel Testi (ZKN, GTD, GT, ALPHA_A, ALPHA_B, ZT3, KBM)
3. Deney A2: 6 Ajan Birlikte (Kombinasyon) Testi
4. Deney A3: Leave-One-Out (Birleşimden her ajan tek tek çıkarılmış 6 deney)
5. Deney A4: Sinyal Örtüşme ve Benzerlik Matrisi (Jaccard & Cosine)
6. Deney B1: Filtre Ablasyonu (EMA200, RSI, ADX, DI, MACD, CMF, Hacim)
7. Deney B2: ALPHA Varyant A (Özgün) vs Varyant B (Filtreli) Karşılaştırması
8. Deney C1: Gap Duyarlılığı (%0.00, %0.25, %0.50, %1.00)
9. Deney C2: Masraf Dayanıklılığı (%0.20+%0.20, %0.30+%0.30, %0.40+%0.40)
10. Deney D: Walk-Forward Geliştirme (2021-2024), Doğrulama (2024-2025) ve Dondurulmuş Modelin Dokunulmamış Nihai Test Yılı (2025-2026) Değerlendirmesi
11. Deney E: Gerçekçi Portföy Testi (100.000 TL, 7 Slot, Nakit Kısıtı)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from research_p1.backtest_engine import simulate_trades, evaluate_agent_signals
from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_all_data() -> Dict[str, pd.DataFrame]:
    log.info("BIST100 hisselerinin indikatörleri hesaplanıyor...")
    prepared = {}
    for sym in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(sym)
        if df is None or len(df) < 50:
            continue
        try:
            ind = compute_all_indicators(df)
            prepared[sym] = ind
        except Exception as exc:
            log.warning("Hata %s: %s", sym, exc)
    log.info("Toplam %d hisse indikatör serisi hazırlandı.", len(prepared))
    return prepared


def run_all_experiments():
    symbol_data = load_and_prepare_all_data()

    # Dönem Sınırları
    FULL_START = "2021-10-06"
    FULL_END = "2026-10-05"

    DEV_START = "2021-10-06"
    DEV_END = "2024-10-05"

    VAL_START = "2024-10-06"
    VAL_END = "2025-10-05"

    TEST_START = "2025-10-06"
    TEST_END = "2026-10-05"

    agents_list = ["ZKN", "GTD", "GT", "ALPHA_A", "ALPHA_B", "ZT3", "KBM"]

    from research_p1.backtest_engine import precompute_signals_for_agents
    log.info("Ajan sinyalleri önceden hesaplanıyor...")
    precomputed = precompute_signals_for_agents(symbol_data, agents_list)
    log.info("Ajan sinyalleri hazırlandı.")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DENEY A1: 6 Ajan Bireysel Testi (Tam 5 Yıl, Gap Yok, %0.2+%0.2)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney A1: Bireysel Ajan Performansları Başlıyor ---")
    a1_results = []
    all_agent_trades = {}

    for ag in agents_list:
        trades, rejected, m = simulate_trades(
            symbol_data, [ag],
            precomputed_signals=precomputed,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=0.0001,
            commission_rate=0.002, slippage_rate=0.002,
            mode="single_agent"
        )
        all_agent_trades[ag] = trades
        # Save CSV
        pd.DataFrame(trades).to_csv(RESULTS_DIR / f"trades_individual_{ag}.csv", index=False)
        m["agent"] = ag
        a1_results.append(m)
        log.info("Ajan %s: %d islem, Ort Net Getiri: %+.2f%%, WinRate: %.1f%%, PF: %.2f",
                 ag, m["n_trades"], m["mean_return_pct"], m["win_rate"], m["profit_factor"])

    a1_df = pd.DataFrame(a1_results)
    a1_df.to_csv(RESULTS_DIR / "exp_a1_individual_agents.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DENEY A2: 6 Ajan Birlikte (Kombinasyon) Testi
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney A2: 6 Ajan Birlikte (Kombinasyon) Testi ---")
    core_6 = ["ZKN", "GTD", "GT", "ALPHA_A", "ZT3", "KBM"]
    trades_all, rej_all, m_all = simulate_trades(
        symbol_data, core_6,
        precomputed_signals=precomputed,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=0.002, slippage_rate=0.002,
        mode="single_agent"
    )
    pd.DataFrame(trades_all).to_csv(RESULTS_DIR / "trades_combined_6agents.csv", index=False)
    pd.DataFrame([m_all]).to_csv(RESULTS_DIR / "exp_a2_combined_6agents.csv", index=False)
    log.info("6 Ajan Birlikte: %d islem, Ort Net: %+.2f%%, PF: %.2f",
             m_all["n_trades"], m_all["mean_return_pct"], m_all["profit_factor"])

    # ─────────────────────────────────────────────────────────────────────────
    # 3. DENEY A3: Leave-One-Out (Bileşimden Tek Tek Çıkarma)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney A3: Leave-One-Out Testi ---")
    a3_results = []
    for ag_out in core_6:
        sub_agents = [a for a in core_6 if a != ag_out]
        _, _, m_sub = simulate_trades(
            symbol_data, sub_agents,
            precomputed_signals=precomputed,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=0.0001,
            commission_rate=0.002, slippage_rate=0.002,
            mode="single_agent"
        )
        m_sub["excluded_agent"] = ag_out
        m_sub["remaining_agents"] = ",".join(sub_agents)
        a3_results.append(m_sub)
        log.info("Haric %s: %d islem, Ort Net: %+.2f%%, PF: %.2f",
                 ag_out, m_sub["n_trades"], m_sub["mean_return_pct"], m_sub["profit_factor"])

    pd.DataFrame(a3_results).to_csv(RESULTS_DIR / "exp_a3_leave_one_out.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DENEY A4: Sinyal Örtüşme ve Benzerlik Matrisi (Jaccard)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney A4: Sinyal Örtüşme Analizi ---")
    agent_signals_set = {ag: set() for ag in core_6}
    for sym, df in symbol_data.items():
        sub_df = df.loc[FULL_START:FULL_END]
        for ag in core_6:
            sig = evaluate_agent_signals(df, ag).loc[FULL_START:FULL_END]
            for dt, val in sig.items():
                if val:
                    agent_signals_set[ag].add((sym, str(dt.date())))

    overlap_matrix = pd.DataFrame(index=core_6, columns=core_6, dtype=float)
    for a1 in core_6:
        for a2 in core_6:
            s1 = agent_signals_set[a1]
            s2 = agent_signals_set[a2]
            jaccard = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0.0
            overlap_matrix.loc[a1, a2] = round(jaccard * 100, 2)

    overlap_matrix.to_csv(RESULTS_DIR / "exp_a4_signal_overlap_jaccard.csv")
    log.info("Sinyal Örtüşme (Jaccard %%):\n%s", overlap_matrix)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. DENEY C1: Gap Duyarlılığı Analizi (%0.00, %0.25, %0.50, %1.00)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney C1: Gap Duyarlılığı Testleri ---")
    c1_results = []
    gap_tolerances = [0.0001, 0.0025, 0.0050, 0.0100]
    for g_tol in gap_tolerances:
        _, rej, m_gap = simulate_trades(
            symbol_data, core_6,
            precomputed_signals=precomputed,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=g_tol,
            commission_rate=0.002, slippage_rate=0.002,
            mode="single_agent"
        )
        m_gap["gap_tolerance"] = g_tol
        c1_results.append(m_gap)
        log.info("Gap Tol %.4f: %d islem, Red: %d, Ort Net: %+.2f%%",
                 g_tol, m_gap["n_trades"], m_gap["rejected_signals"], m_gap["mean_return_pct"])

    pd.DataFrame(c1_results).to_csv(RESULTS_DIR / "exp_c1_gap_sensitivity.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. DENEY C2: Masraf ve Slipaj Dayanıklılığı
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney C2: Masraf Dayanıklılığı ---")
    c2_results = []
    cost_scenarios = [
        (0.002, 0.002, "Temel: %0.20 Komisyon + %0.20 Slipaj"),
        (0.003, 0.003, "Dayanıklılık 1: %0.30 Komisyon + %0.30 Slipaj"),
        (0.004, 0.004, "Dayanıklılık 2: %0.40 Komisyon + %0.40 Slipaj"),
    ]
    for comm, slip, name in cost_scenarios:
        _, _, m_c = simulate_trades(
            symbol_data, core_6,
            precomputed_signals=precomputed,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=0.0001,
            commission_rate=comm, slippage_rate=slip,
            mode="single_agent"
        )
        m_c["scenario_name"] = name
        m_c["commission"] = comm
        m_c["slippage"] = slip
        c2_results.append(m_c)
        log.info("%s: Ort Net: %+.2f%%, PF: %.2f", name, m_c["mean_return_pct"], m_c["profit_factor"])

    pd.DataFrame(c2_results).to_csv(RESULTS_DIR / "exp_c2_cost_robustness.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. DENEY D: Walk-Forward Dönem Ayrımı (Geliştirme, Doğrulama, Nihai Test)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney D: Walk-Forward Dönem Değerlendirmesi ---")
    periods = [
        ("Gelistirme_2021_2024", DEV_START, DEV_END),
        ("Dogrulama_2024_2025", VAL_START, VAL_END),
        ("Nihai_Test_2025_2026", TEST_START, TEST_END),
    ]
    d_results = []
    candidate_models = {
        "M1_Tam_6Ajan": core_6,
        "M2_GT_GTD": ["GT", "GTD"],
        "M3_ALPHA_A_Only": ["ALPHA_A"],
        "M4_Sade_GT_GTD_ALPHA": ["GT", "GTD", "ALPHA_A"],
        "M5_ZKN_Only": ["ZKN"],
    }

    for p_name, p_start, p_end in periods:
        for m_name, ag_set in candidate_models.items():
            _, _, m_p = simulate_trades(
                symbol_data, ag_set,
                precomputed_signals=precomputed,
                start_date=p_start, end_date=p_end,
                gap_tolerance=0.0001,
                commission_rate=0.002, slippage_rate=0.002,
                mode="single_agent"
            )
            m_p["period"] = p_name
            m_p["model_name"] = m_name
            d_results.append(m_p)
            log.info("[%s] %s: %d islem, Ort Net: %+.2f%%, PF: %.2f",
                     p_name, m_name, m_p["n_trades"], m_p["mean_return_pct"], m_p["profit_factor"])

    pd.DataFrame(d_results).to_csv(RESULTS_DIR / "exp_d_walk_forward_models.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. DENEY E: Gerçekçi Portföy Simülasyonu (100.000 TL, 7 Slot)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Deney E: Gerçekçi Portföy Simülasyonu ---")
    p_trades, _, p_metrics = simulate_trades(
        symbol_data, ["GT", "GTD", "ALPHA_A"],
        precomputed_signals=precomputed,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=0.002, slippage_rate=0.002,
        mode="portfolio", initial_capital=100_000.0, max_positions=7
    )
    pd.DataFrame(p_trades).to_csv(RESULTS_DIR / "trades_portfolio_sade_model.csv", index=False)
    with open(RESULTS_DIR / "exp_e_portfolio_metrics.json", "w", encoding="utf-8") as f:
        json.dump(p_metrics, f, indent=2, ensure_ascii=False)

    log.info("Portföy Sonuçları: Final Equity: %.2f TL, CAGR: %.2f%%, MaxDD: %.2f%%",
             p_metrics.get("final_equity", 0), p_metrics.get("cagr_pct", 0), p_metrics.get("max_drawdown_pct", 0))

    log.info("Tüm deneyler başarıyla tamamlandı ve CSV kayıtları results/ klasörüne yazıldı.")


if __name__ == "__main__":
    run_all_experiments()
