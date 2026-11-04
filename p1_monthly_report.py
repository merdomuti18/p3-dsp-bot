# -*- coding: utf-8 -*-
"""
p1_monthly_report.py — P1 Paper Trading Aylık Raporlama Scripti
==============================================================
Kullanım:
    python p1_monthly_report.py YYYYMM
Örnek:
    python p1_monthly_report.py 202609

Çıktı dosyası:
    results/monthly_report_YYYYMM.md

İçerik:
- İşlem sayısı
- Ajan bazlı getiri ve kâr tablosu
- Hareketli ortalama dolum maliyeti (komisyon ve slipaj ayrı)
- Kapasite reddi sayısı ve reddedilenlerin sonradan getirisi
- Günlük MTM Drawdown (20k ve 10k)
- Rejim bayrağına göre kırılım (XU100 > EMA200)
- Sinyal uyum oranı
- Kesinlikle performans yorumu içermez, sadece rakamları verir.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from p1_paper_config import (
    COMMISSION_RATE,
    PAPER_LOG_FILE,
    PAPER_STATE_FILE,
    RESULTS_DIR,
    ROUNDTRIP_COST_PCT,
    SIGNAL_PARITY_FILE,
)
from p1_monitoring import evaluate_cost_tier

log = logging.getLogger("p1_monthly_report")


def generate_monthly_report(yyyymm: str) -> Path:
    """Belirtilen ay (YYYYMM) için aylık markdown raporunu üretir."""
    year_str = yyyymm[:4]
    month_str = yyyymm[4:6]
    target_month_prefix = f"{year_str}-{month_str}"

    out_file = RESULTS_DIR / f"monthly_report_{yyyymm}.md"

    # 1. Log tablosunu yükle
    trades_in_month = []
    rejected_in_month = []

    if PAPER_LOG_FILE.exists() and PAPER_LOG_FILE.stat().st_size > 0:
        with open(PAPER_LOG_FILE, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                sig_date = r.get("sinyal_tarihi", "")
                entry_date = r.get("giris_tarihi", "")
                exit_date = r.get("cikis_tarihi", "")

                # Ay eşleşmesi (giriş veya çıkış bu aya aitse)
                in_month = (entry_date and entry_date.startswith(target_month_prefix)) or \
                           (exit_date and exit_date.startswith(target_month_prefix)) or \
                           (sig_date and sig_date.startswith(target_month_prefix))

                if in_month:
                    if r.get("kapasite_durumu") == "ISLEME_ALINDI" and r.get("cikis_nedeni"):
                        trades_in_month.append(r)
                    elif r.get("kapasite_durumu") in ("KAPASITE_REDDI", "GAP_REDDI"):
                        rejected_in_month.append(r)

    # 2. State dosyasından günlük equity serisini yükle
    daily_eq_in_month = []
    if PAPER_STATE_FILE.exists():
        try:
            with open(PAPER_STATE_FILE, "r", encoding="utf-8") as fh:
                st = json.load(fh)
                for de in st.get("daily_equity", []):
                    if de.get("tarih", "").startswith(target_month_prefix):
                        daily_eq_in_month.append(de)
        except Exception:
            pass

    # 3. Parity log dosyasından sinyal uyum oranını yükle
    parity_rates = []
    discrepancies_count = 0
    if SIGNAL_PARITY_FILE.exists():
        try:
            with open(SIGNAL_PARITY_FILE, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for pr in reader:
                    if pr.get("tarih", "").startswith(target_month_prefix):
                        try:
                            parity_rates.append(float(pr.get("uyum_orani_pct", 100.0)))
                            if pr.get("uyusmazliklar") and pr.get("uyusmazliklar") != "YOK":
                                discrepancies_count += len(pr.get("uyusmazliklar").split(";"))
                        except ValueError:
                            pass
        except Exception:
            pass

    avg_parity_pct = round(sum(parity_rates) / len(parity_rates), 2) if parity_rates else 100.0

    # 4. İstatistiklerin Hesaplanması
    n_trades = len(trades_in_month)
    df_m_trades = pd.DataFrame(trades_in_month) if trades_in_month else pd.DataFrame()

    total_pnl_20k = 0.0
    total_pnl_10k = 0.0
    avg_net_ret = 0.0
    win_rate = 0.0

    if not df_m_trades.empty:
        df_m_trades["net_return_pct"] = pd.to_numeric(df_m_trades["net_getiri_pct"], errors="coerce").fillna(0.0)
        df_m_trades["net_pnl_tl"] = pd.to_numeric(df_m_trades["net_pnl_tl"], errors="coerce").fillna(0.0)
        df_m_trades["net_pnl_10k"] = pd.to_numeric(df_m_trades["net_pnl_10k_esdegeri_tl"], errors="coerce").fillna(0.0)

        total_pnl_20k = float(df_m_trades["net_pnl_tl"].sum())
        total_pnl_10k = float(df_m_trades["net_pnl_10k"].sum())
        avg_net_ret = float(df_m_trades["net_return_pct"].mean())
        win_rate = float((df_m_trades["net_return_pct"] > 0).mean() * 100.0)

    # Dolum maliyeti & slipaj vekili
    slip_proxies = []
    for r in trades_in_month:
        sp = r.get("gerceklesen_slipaj_pct")
        if sp != "" and sp is not None:
            try:
                slip_proxies.append(float(sp))
            except ValueError:
                pass

    avg_slip_proxy = round(sum(slip_proxies) / len(slip_proxies), 4) if slip_proxies else 0.0
    comm_pct = round(COMMISSION_RATE * 100.0, 2)
    rolling_fill_cost_pct = round(2.0 * (comm_pct + avg_slip_proxy), 4) if slip_proxies else ROUNDTRIP_COST_PCT
    cost_tier, cost_tier_msg = evaluate_cost_tier(rolling_fill_cost_pct)

    # Kapasite reddi ve sonradan getiri
    cap_rejected = [r for r in rejected_in_month if r.get("kapasite_durumu") == "KAPASITE_REDDI"]
    gap_rejected = [r for r in rejected_in_month if r.get("kapasite_durumu") == "GAP_REDDI"]
    n_cap_rejected = len(cap_rejected)
    n_gap_rejected = len(gap_rejected)

    subsequent_returns = []
    for r in cap_rejected:
        sr = r.get("reddedilen_sonradan_getiri_pct")
        if sr != "" and sr is not None:
            try:
                subsequent_returns.append(float(sr))
            except ValueError:
                pass
    avg_subsequent_ret = round(sum(subsequent_returns) / len(subsequent_returns), 4) if subsequent_returns else 0.0

    # Günlük MTM Drawdown
    dd_20k = 0.0
    dd_10k = 0.0
    if daily_eq_in_month:
        df_de = pd.DataFrame(daily_eq_in_month)
        if "equity_20k" in df_de.columns:
            peak_20 = df_de["equity_20k"].cummax()
            dd_20k = float(((df_de["equity_20k"] - peak_20) / peak_20 * 100.0).min())
        if "equity_10k" in df_de.columns:
            peak_10 = df_de["equity_10k"].cummax()
            dd_10k = float(((df_de["equity_10k"] - peak_10) / peak_10 * 100.0).min())

    # 5. Markdown Raporunun Oluşturulması
    lines = [
        f"# P1 Paper Trading Aylık Raporu — {year_str}-{month_str}",
        "",
        "## 1. Genel Özet Tablosu",
        "",
        "| Metrik | Değer |",
        "|---|---|",
        f"| **Dönem** | {year_str}-{month_str} |",
        f"| **Kapanan İşlem Sayısı** | {n_trades} |",
        f"| **Kazanma Oranı (%)** | %{win_rate:.2f} |",
        f"| **İşlem Başı Ortalama Net Getiri (%)** | %{avg_net_ret:+.2f} |",
        f"| **Toplam Net Kâr (20k Boyut)** | {total_pnl_20k:+,.2f} TL |",
        f"| **Toplam Net Kâr (10k Eşdeğeri)** | {total_pnl_10k:+,.2f} TL |",
        f"| **Kapasite Reddi Sayısı** | {n_cap_rejected} |",
        f"| **Gap Reddi Sayısı** | {n_gap_rejected} |",
        f"| **Sinyal Uyum Oranı (%)** | %{avg_parity_pct:.2f} |",
        "",
        "## 2. Ajan Bazlı Getiri ve Kâr Dağılımı",
        "",
        "| Ajan | İşlem Sayısı | Win Rate (%) | Ort. Net Getiri (%) | Toplam Kâr 20k (TL) | Toplam Kâr 10k (TL) |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]

    if not df_m_trades.empty:
        for ag, grp in df_m_trades.groupby("birincil_ajan"):
            n_ag = len(grp)
            w_ag = (grp["net_return_pct"] > 0).mean() * 100.0
            m_ag = grp["net_return_pct"].mean()
            pnl20_ag = grp["net_pnl_tl"].sum()
            pnl10_ag = grp["net_pnl_10k"].sum()
            lines.append(f"| **{ag}** | {n_ag} | %{w_ag:.2f} | %{m_ag:+.2f} | {pnl20_ag:+,.2f} TL | {pnl10_ag:+,.2f} TL |")
    else:
        lines.append("| — | 0 | %0.00 | %0.00 | 0.00 TL | 0.00 TL |")

    lines.extend([
        "",
        "## 3. Dolum Maliyeti ve Slipaj Dağılımı",
        "",
        "| Bileşen | Oran (%) | Not |",
        "|---|:---:|---|",
        f"| **Komisyon Oranı (Tek Yön)** | %{comm_pct:.2f} | Sabit |",
        f"| **Ortalama Slipaj Vekili (5dk VWAP)** | %{avg_slip_proxy:.4f} | {'VWAP verisi yok' if not slip_proxies else 'Hesaplandı'} |",
        f"| **Hareketli Ortalama Tur Başı Maliyet** | %{rolling_fill_cost_pct:.2f} | 2 × (Komisyon + Slipaj) |",
        f"| **Maliyet Kademe Durumu** | **{cost_tier}** | {cost_tier_msg} |",
        "",
        "## 4. Kapasite Reddi ve Reddedilen Sinyallerin Sonradan Getirisi",
        "",
        "| Metrik | Değer |",
        "|---|---|",
        f"| **Kapasite Reddi Sayısı (5 Slot Tavanı)** | {n_cap_rejected} |",
        f"| **Reddedilenlerin Ortalama Sonradan Getirisi (%)** | %{avg_subsequent_ret:+.2f} |",
        f"| **Tamamlanan Sanal Takip Sayısı** | {len(subsequent_returns)} |",
        "",
        "## 5. Günlük MTM Drawdown",
        "",
        "| Portföy Modeli | Aylık Gerçekleşen Max MTM DD (%) |",
        "|---|:---:|",
        f"| **20.000 TL Pozisyon Boyutu** | %{dd_20k:.2f} |",
        f"| **10.000 TL Eşdeğeri** | %{dd_10k:.2f} |",
        "",
        "## 6. XU100 > EMA200 Rejim Kırılımı",
        "",
        "| Rejim | İşlem Sayısı | Win Rate (%) | Ort. Net Getiri (%) | Toplam Net Kâr 20k (TL) |",
        "|---|:---:|:---:|:---:|:---:|",
    ])

    if not df_m_trades.empty and "xu100_rejim_boga" in df_m_trades.columns:
        for reg_val, grp in df_m_trades.groupby("xu100_rejim_boga"):
            reg_label = "Boğa (XU100 > EMA200)" if str(reg_val).lower() in ("true", "1") else "Ayı (XU100 <= EMA200)"
            lines.append(f"| **{reg_label}** | {len(grp)} | %{(grp['net_return_pct']>0).mean()*100:.2f} | %{grp['net_return_pct'].mean():+.2f} | {grp['net_pnl_tl'].sum():+,.2f} TL |")
    else:
        lines.append("| — | 0 | %0.00 | %0.00 | 0.00 TL |")

    lines.extend([
        "",
        "## 7. Sinyal Uyum Oranı",
        "",
        "| Metrik | Değer |",
        "|---|---|",
        f"| **Ay İçi Kontrol Edilen Gün Sayısı** | {len(parity_rates)} |",
        f"| **Ortalama Sinyal Uyum Oranı (%)** | %{avg_parity_pct:.2f} |",
        f"| **Tespit Edilen Uyuşmazlık Sayısı** | {discrepancies_count} |",
        "",
        "---",
        f"*Rapor Oluşturma Zamanı: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    content = "\n".join(lines) + "\n"
    out_file.write_text(content, encoding="utf-8")
    log.info("Aylık Rapor oluşturuldu: %s", out_file)
    return out_file


if __name__ == "__main__":
    if len(sys.argv) > 1:
        yyyymm_arg = sys.argv[1].replace("-", "").strip()
    else:
        yyyymm_arg = pd.Timestamp.now().strftime("%Y%m")

    report_path = generate_monthly_report(yyyymm_arg)
    print(f"Rapor yazıldı -> {report_path}")
