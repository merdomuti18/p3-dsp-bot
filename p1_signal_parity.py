# -*- coding: utf-8 -*-
"""
p1_signal_parity.py — P1 Sinyal Uyum ve Replay Doğrulama Modülü
==============================================================
Her seans sonunda botun ürettiği sinyaller ile backtest motorunun
aynı veri üzerindeki sinyallerini karşılaştırır ve uyum oranını kaydeder.
Başlamadan önce son 60 seanslık geçmiş REPLAY ile sistem doğrulanır.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from p1_paper_config import (
    AGENTS_PRIORITY,
    SHADOW_AGENTS,
    SIGNAL_PARITY_FILE,
)
from research_p1.run_p1_revision_backtest import evaluate_custom_agent_signals

log = logging.getLogger(__name__)

PARITY_LOG_COLUMNS = [
    "tarih",
    "toplam_bot_sinyal",
    "toplam_backtest_sinyal",
    "ortusen_sinyal",
    "uyum_orani_pct",
    "uyusmazliklar",
]


def init_signal_parity_log() -> None:
    """Parity CSV log dosyasını başlıklarıyla başlatır."""
    if not SIGNAL_PARITY_FILE.exists() or SIGNAL_PARITY_FILE.stat().st_size == 0:
        SIGNAL_PARITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_PARITY_FILE, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(PARITY_LOG_COLUMNS)


def compare_daily_signals(
    tarih_str: str,
    bot_signals: Set[Tuple[str, str]],       # {(symbol, agent), ...}
    backtest_signals: Set[Tuple[str, str]],  # {(symbol, agent), ...}
) -> dict:
    """Verilen gün için bot sinyalleri ile backtest sinyallerini karşılaştırır."""
    init_signal_parity_log()

    total_bot = len(bot_signals)
    total_bt = len(backtest_signals)
    intersection = bot_signals & backtest_signals
    n_common = len(intersection)

    if total_bot == 0 and total_bt == 0:
        parity_rate = 100.0
        discrepancies = []
    else:
        # Dice / F1 benzerlik katsayısı: 2 * |A n B| / (|A| + |B|)
        parity_rate = round((2.0 * n_common / (total_bot + total_bt)) * 100.0, 2)
        diff_bot = bot_signals - backtest_signals
        diff_bt = backtest_signals - bot_signals
        discrepancies = [f"+BOT:{s}_{a}" for s, a in diff_bot] + [f"+BT:{s}_{a}" for s, a in diff_bt]

    row = {
        "tarih": tarih_str,
        "toplam_bot_sinyal": total_bot,
        "toplam_backtest_sinyal": total_bt,
        "ortusen_sinyal": n_common,
        "uyum_orani_pct": parity_rate,
        "uyusmazliklar": ";".join(discrepancies) if discrepancies else "YOK",
    }

    with open(SIGNAL_PARITY_FILE, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PARITY_LOG_COLUMNS)
        writer.writerow(row)

    if parity_rate < 95.0:
        log.warning("SİNYAL UYUM UYARISI [%s]: Uyum oranı %%%.2f < %%95.0! Uyuşmazlıklar: %s",
                    tarih_str, parity_rate, row["uyusmazliklar"])
    else:
        log.info("Sinyal uyum teyit edildi [%s]: %%%.2f uyum (%d/%d sinyal)",
                 tarih_str, parity_rate, n_common, max(total_bot, total_bt))

    return row


def evaluate_backtest_signals_for_day(
    symbol_data: Dict[str, pd.DataFrame],
    dt: pd.Timestamp,
    agents: Optional[List[str]] = None,
) -> Set[Tuple[str, str]]:
    """Backtest motoru ile belirli bir gündeki (t) sinyalleri değerlendirir."""
    target_agents = agents or (AGENTS_PRIORITY + SHADOW_AGENTS)
    signals = set()
    for sym, df in symbol_data.items():
        if dt not in df.index:
            continue
        for ag in target_agents:
            sig_s = evaluate_custom_agent_signals(df, ag)
            if sig_s.loc[dt]:
                signals.add((sym, ag))
    return signals


def run_60_session_replay(
    symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
    n_sessions: int = 60,
) -> dict:
    """Son 60 seanslık geçmiş üzerinde bot ve backtest sinyal uyumunu replay ile doğrular."""
    from p1_paper_config import RESEARCH_DIR
    from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
    from research_p1.p1_engine import compute_all_indicators

    if symbol_data is None:
        log.info("60 Seans Replay: BIST100 önbellek verileri yükleniyor...")
        symbol_data = {}
        for s in BIST100_SYMBOLS:
            df, meta = load_cached_ohlcv(s)
            if df is not None and len(df) >= 50:
                symbol_data[s] = compute_all_indicators(df)

    # Ortak tarih takvimi
    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    if len(all_dates) < n_sessions:
        raise ValueError(f"Yetersiz seans sayısı: {len(all_dates)} < {n_sessions}")

    replay_dates = all_dates[-n_sessions:]
    log.info("60 Seans Replay başlıyor: %s -> %s (%d seans)",
             replay_dates[0].strftime("%Y-%m-%d"), replay_dates[-1].strftime("%Y-%m-%d"), len(replay_dates))

    # Log dosyasını temizleyip yeniden başlatalım
    if SIGNAL_PARITY_FILE.exists():
        SIGNAL_PARITY_FILE.unlink()
    init_signal_parity_log()

    parity_rates = []
    total_signals_count = 0
    discrepant_days = 0

    from p1_paper import generate_bot_signals_for_day

    for cur_dt in replay_dates:
        t_str = cur_dt.strftime("%Y-%m-%d")
        # Bot tarafı sinyalleri
        bot_sigs, _ = generate_bot_signals_for_day(symbol_data, cur_dt)
        # Backtest motoru sinyalleri
        bt_sigs = evaluate_backtest_signals_for_day(symbol_data, cur_dt)

        res = compare_daily_signals(t_str, bot_sigs, bt_sigs)
        parity_rates.append(res["uyum_orani_pct"])
        total_signals_count += res["toplam_backtest_sinyal"]
        if res["uyum_orani_pct"] < 95.0:
            discrepant_days += 1

    overall_parity_pct = round(sum(parity_rates) / len(parity_rates), 2)
    summary = {
        "n_sessions": len(replay_dates),
        "start_date": replay_dates[0].strftime("%Y-%m-%d"),
        "end_date": replay_dates[-1].strftime("%Y-%m-%d"),
        "total_signals_generated": total_signals_count,
        "overall_parity_pct": overall_parity_pct,
        "discrepant_days_count": discrepant_days,
        "status": "PASSED" if overall_parity_pct >= 95.0 else "FAILED",
    }

    log.info("60 Seans Replay Tamamlandı: Ortalama Uyum = %%%.2f, Durum: %s",
             overall_parity_pct, summary["status"])

    if overall_parity_pct < 95.0:
        raise RuntimeError(f"60 Seans Replay Sinyal Uyumu Yetersiz: %{overall_parity_pct} < %95.0! Detaylar için signal_parity_log.csv'ye bakın.")

    return summary
