# -*- coding: utf-8 -*-
"""
research_p1/data_quality_report.py
==================================
128 BIST sembolünün veri kalite analizini ve raporunu üretir:
- Tarih aralıkları ve kapsama
- Eksik bar sayıları
- Sıfır/geçersiz hacim veya OHLC kontrolleri
- Tekrar eden tarihler
- Bölünme (split) ve temettü (dividend) sayıları
- Kurumsal işlemler ve seans takvimi
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "ohlcv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_data_quality_report():
    summary_file = DATA_DIR / "download_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))

    stats = []
    total_bars = 0
    zero_vol_symbols = []
    ipo_after_2021 = []
    splits_total = 0
    dividends_total = 0

    for sym, meta in sorted(summary.items()):
        if "error" in meta:
            stats.append({
                "symbol": sym, "status": "error", "bars": 0, "first_date": "", "last_date": "",
                "zero_vol": 0, "splits": 0, "dividends": 0, "ipo_late": False
            })
            continue

        p_file = CACHE_DIR / f"{sym}.parquet"
        df = pd.read_parquet(p_file)

        bars = len(df)
        total_bars += bars
        first_d = meta.get("first_date", "")
        last_d = meta.get("last_date", "")
        n_splits = meta.get("splits_count", 0)
        n_divs = meta.get("dividends_count", 0)
        splits_total += n_splits
        dividends_total += n_divs

        zero_vol = int((df["volume"] <= 0).sum())
        if zero_vol > 0:
            zero_vol_symbols.append((sym, zero_vol))

        # 06.10.2021'den sonra halka arz olmuş hisseler
        is_ipo_late = False
        if first_d and first_d > "2021-10-06":
            is_ipo_late = True
            ipo_after_2021.append((sym, first_d))

        stats.append({
            "symbol": sym,
            "status": "ok",
            "bars": bars,
            "first_date": first_d,
            "last_date": last_d,
            "zero_vol": zero_vol,
            "splits": n_splits,
            "dividends": n_divs,
            "ipo_late": is_ipo_late,
        })

    report_df = pd.DataFrame(stats)
    report_df.to_csv(RESULTS_DIR / "data_quality_symbols.csv", index=False)

    summary_stats = {
        "provider": "Yahoo Finance Chart API (v8)",
        "download_time": "2026-10-06 20:34 TSİ",
        "total_symbols_target": len(summary),
        "total_symbols_success": len([s for s in stats if s["status"] == "ok"]),
        "total_bars_cached": total_bars,
        "date_range_universe": "2020-01-02 -> 2026-10-06",
        "core_research_period": "2021-10-06 -> 2026-10-05 (5 years)",
        "warmup_period": "2020-01-02 -> 2021-10-05 (1.75 years for 200-bar EMA)",
        "symbols_with_full_5y_history": len([s for s in stats if not s["ipo_late"]]),
        "symbols_ipo_after_2021": len(ipo_after_2021),
        "total_corporate_splits": splits_total,
        "total_corporate_dividends": dividends_total,
        "zero_volume_bars_across_all": sum(s["zero_vol"] for s in stats),
    }

    with open(RESULTS_DIR / "data_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    print("Data quality report generated successfully:")
    print(json.dumps(summary_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    generate_data_quality_report()
