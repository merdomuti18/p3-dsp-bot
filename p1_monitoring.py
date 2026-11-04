# -*- coding: utf-8 -*-
"""
p1_monitoring.py — P1 Paper Trading İzleme ve Uyarı Sistemi
============================================================
Paper trading sürecinde izleme ve uyarı mantığı:
- Dolum maliyeti hareketli ortalaması (son 20-30 işlem, n >= 20 olunca):
    * Yeşil:  <= %1.20 (Hedef maliyet aralığı)
    * Sarı:   %1.20 - %1.50 (Maliyet sınırda, dikkat)
    * Kırmızı: > %1.50 (Maliyet yüksek, strateji nakdin altına düşebilir)
- Günlük MTM Drawdown uyarıları (açık pozisyonlar dahil):
    * Seviye 1: -%10.0
    * Seviye 2: -%15.0
    * Seviye 3: -%20.0 (Hard-stop eşiği)

NOT: Paper modunda bu kontroller YALNIZCA log ve uyarı üretir, otomatik işlem iptali yapmaz.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from p1_paper_config import (
    ALERTS_LOG_FILE,
    COST_ALERT_GREEN_MAX,
    COST_ALERT_YELLOW_MAX,
    DD_ALERT_THRESHOLDS,
)

log = logging.getLogger(__name__)


def evaluate_cost_tier(rolling_cost_pct: float) -> Tuple[str, str]:
    """Dolum maliyeti kademesini belirler.

    Dönüş: (KADEME, MESAJ)
      - YESIL:   <= %1.20
      - SARI:    %1.20 - %1.50
      - KIRMIZI: > %1.50
    """
    if rolling_cost_pct <= COST_ALERT_GREEN_MAX:
        return "YESIL", f"Dolum maliyeti hedef aralıkta: %{rolling_cost_pct:.2f} <= %{COST_ALERT_GREEN_MAX:.2f}"
    elif rolling_cost_pct <= COST_ALERT_YELLOW_MAX:
        return "SARI", f"UYARI (SARI): Dolum maliyeti sınırda: %{rolling_cost_pct:.2f} (%{COST_ALERT_GREEN_MAX:.2f} - %{COST_ALERT_YELLOW_MAX:.2f})"
    else:
        return "KIRMIZI", f"KRİTİK UYARI (KIRMIZI): Dolum maliyeti yüksek: %{rolling_cost_pct:.2f} > %{COST_ALERT_YELLOW_MAX:.2f}"


def _measured_proxy_cost(trade: dict) -> Optional[float]:
    """Yalnızca ölçülen dolum vekili. Simüle slipaj (açılış×1,003 / %1,20) kullanılmaz."""
    c_val = trade.get("dolum_vekili_tur_basi_maliyet_pct")
    if c_val is None or c_val == "":
        return None
    if isinstance(c_val, float) and np_isnan(c_val):
        return None
    try:
        return float(c_val)
    except (ValueError, TypeError):
        return None


def check_fill_cost_alerts(
    trades: List[dict],
    window: int = 20,
    timestamp: Optional[datetime] = None,
) -> Optional[dict]:
    """Son 20–30 işlemin ölçülen vekil maliyet ortalaması (n >= 20 ölçüm şart).

    Simüle slipaj ve boş VWAP satırları hesaba katılmaz. n < 20 ise uyarı üretilmez.
    """
    now_dt = timestamp or datetime.now()
    window = min(max(int(window), 20), 30)

    measured_costs: List[float] = []
    for t in reversed(list(trades)):
        val = _measured_proxy_cost(t)
        if val is None:
            continue
        measured_costs.append(val)
        if len(measured_costs) >= window:
            break

    if len(measured_costs) < 20:
        return None

    avg_cost = sum(measured_costs) / len(measured_costs)
    tier, msg = evaluate_cost_tier(avg_cost)

    alert = {
        "timestamp": now_dt.isoformat(),
        "type": "FILL_COST_ALERT",
        "tier": tier,
        "rolling_window": window,
        "average_cost_pct": round(avg_cost, 4),
        "message": msg,
    }

    _append_alert_log(alert)
    if tier in ("SARI", "KIRMIZI"):
        log.warning("[%s] %s", tier, msg)
    else:
        log.info("[%s] %s", tier, msg)

    return alert


def check_drawdown_alerts(
    current_equity: float,
    peak_equity: float,
    size_label: str = "20k",
    timestamp: Optional[datetime] = None,
) -> List[dict]:
    """Günlük MTM Drawdown eşiklerini kontrol eder (-10%, -15%, -20%)."""
    now_dt = timestamp or datetime.now()
    if peak_equity <= 0:
        return []

    dd_pct = (current_equity - peak_equity) / peak_equity * 100.0
    alerts = []

    for threshold in sorted(DD_ALERT_THRESHOLDS):  # [-20.0, -15.0, -10.0]
        if dd_pct <= threshold:
            severity = "HARD_STOP_UYARI" if threshold <= -20.0 else ("KRITIK_UYARI" if threshold <= -15.0 else "DIKKAT_UYARI")
            msg = f"{severity} ({size_label}): Günlük MTM Drawdown eşiği aşıldı! Mevcut DD: %{dd_pct:.2f} <= %{threshold:.2f} (Tepe: {peak_equity:,.2f} TL, Mevcut: {current_equity:,.2f} TL)"
            alert = {
                "timestamp": now_dt.isoformat(),
                "type": "DRAWDOWN_ALERT",
                "size_label": size_label,
                "threshold_pct": threshold,
                "current_dd_pct": round(dd_pct, 2),
                "peak_equity": round(peak_equity, 2),
                "current_equity": round(current_equity, 2),
                "message": msg,
            }
            alerts.append(alert)
            _append_alert_log(alert)
            log.warning("[%s] %s", size_label, msg)

    return alerts


def _append_alert_log(alert: dict) -> None:
    try:
        ALERTS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTS_LOG_FILE, "a", encoding="utf-8") as fh:
            import json
            fh.write(json.dumps(alert, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("Alert log yazılamadı: %s", exc)


def warn_vwap_missing(
    symbol: str,
    session_date: str,
    timestamp: Optional[datetime] = None,
) -> dict:
    """1m VWAP alınamadı. Sonradan doldurulamaz (Yahoo 1m ~7 gün)."""
    now_dt = timestamp or datetime.now()
    msg = (
        f"VWAP verisi yok: {symbol} {session_date}. "
        "Ölçülen dolum vekili boş bırakıldı; sonradan doldurulamaz."
    )
    alert = {
        "timestamp": now_dt.isoformat(),
        "type": "VWAP_VERISI_YOK",
        "symbol": symbol,
        "session_date": session_date,
        "message": msg,
    }
    _append_alert_log(alert)
    log.error("%s", msg)
    return alert


def np_isnan(val) -> bool:
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False
