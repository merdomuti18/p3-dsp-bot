"""
notifier_dsp.py — P3-DSP Telegram Bildirimi + Flask Dashboard Route
====================================================================
İki bağımsız bileşen tek dosyada:

  1. DspNotifier   — Mevcut Telegram bot'una [P3-DSP] etiketli mesaj gönderir
  2. build_dashboard_route() — Mevcut Flask uygulamasına /p3-dsp route ekler

VM'deki entegrasyon:
  # Telegram:
  from notifier_dsp import DspNotifier
  notifier = DspNotifier(token=os.getenv("TELEGRAM_BOT_TOKEN"),
                          chat_id=os.getenv("TELEGRAM_CHAT_ID"))
  notifier.send(portfolio.telegram_message(actions))

  # Flask (mevcut app.py içinde):
  from notifier_dsp import build_dashboard_route
  build_dashboard_route(app, portfolio)    # /p3-dsp route eklenir
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Telegram Bildirici
# ---------------------------------------------------------------------------

class DspNotifier:
    """
    Mevcut Telegram bot altyapısını kullanan P3-DSP bildirici.

    Yeni bir bot kurmaya gerek yok — mevcut bota [P3-DSP] etiketli
    mesaj formatı eklenir (Mimari Plan Bölüm 7.2).

    Bağımlılık:
        pip install requests   (muhtemelen zaten kurulu)

    Kullanım:
        notifier = DspNotifier(
            token=os.getenv("TELEGRAM_BOT_TOKEN"),
            chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        )
        notifier.send("📡 *[P3-DSP]* Test mesajı")
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Parameters
        ----------
        token   : Telegram bot token. None ise TELEGRAM_BOT_TOKEN env'den alır.
        chat_id : Hedef chat ID. None ise TELEGRAM_CHAT_ID env'den alır.
        dry_run : True ise gerçek istek atmaz, sadece loglar (test için).
        """
        self.token   = token   or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.dry_run = dry_run

        if not self.token and not dry_run:
            logger.warning("TELEGRAM_BOT_TOKEN bulunamadı — dry_run moduna geçildi")
            self.dry_run = True

    def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Telegram mesajı gönder.

        Returns True başarılı, False hata.
        """
        if self.dry_run:
            logger.info(f"[DRY-RUN] Telegram mesajı:\n{text}")
            print(f"\n── Telegram (dry-run) ──────────────────────────\n{text}\n{'─'*50}")
            return True

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram mesajı gönderildi")
                return True
            else:
                logger.error(f"Telegram hata: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Telegram gönderim hatası: {e}")
            return False

    def send_scan_result(self, scan_result, portfolio_msg: str) -> bool:
        """
        Tarama özeti + portföy değişiklikleri — tek mesajda gönder.
        Değişiklik yoksa sessiz kal (spam önleme).
        """
        # "Değişiklik yok" satırı varsa gönderme
        if "Değişiklik yok" in portfolio_msg and "Giriş" not in portfolio_msg:
            logger.info("Değişiklik yok — Telegram mesajı atlandı")
            return True

        full_msg = (
            f"{portfolio_msg}\n\n"
            f"📈 Tarama: {scan_result.scanned}/{scan_result.total_symbols} sembol | "
            f"{scan_result.long_signals} long sinyal | {scan_result.elapsed_sec:.1f}s"
        )
        return self.send(full_msg)


# ---------------------------------------------------------------------------
# 2. Flask Dashboard Route
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>P3-DSP Dashboard</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9;
            max-width: 960px; margin: 0 auto; padding: 24px; }}
    h1   {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
    h2   {{ color: #79c0ff; margin-top: 32px; }}
    .badge {{ display:inline-block; padding: 2px 10px; border-radius: 12px;
              font-size: 0.85em; font-weight: 600; }}
    .badge-go    {{ background:#238636; color:#fff; }}
    .badge-flat  {{ background:#30363d; color:#8b949e; }}
    table {{ width:100%; border-collapse:collapse; margin-top: 12px; }}
    th    {{ background:#161b22; color:#8b949e; padding:8px 12px;
             text-align:left; font-size:0.85em; text-transform:uppercase; }}
    td    {{ padding:8px 12px; border-bottom:1px solid #21262d; }}
    tr:hover td {{ background:#161b22; }}
    .pos  {{ color:#3fb950; font-weight:600; }}
    .neg  {{ color:#f85149; font-weight:600; }}
    .meta {{ color:#8b949e; font-size:0.85em; margin-top:8px; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
             padding:16px 20px; margin-bottom:16px; }}
    .kpi  {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
    .kpi-item {{ text-align:center; }}
    .kpi-val  {{ font-size:1.6em; font-weight:700; color:#58a6ff; }}
    .kpi-lbl  {{ font-size:0.8em; color:#8b949e; margin-top:2px; }}
  </style>
</head>
<body>
  <h1>📡 P3-DSP Paper Portföy</h1>
  <p class="meta">Son tarama: <strong>{last_scan}</strong> &nbsp;|&nbsp;
                  Strateji: <strong>DSP-XOVER (15/40, order=3)</strong></p>

  <div class="card">
    <div class="kpi">
      <div class="kpi-item">
        <div class="kpi-val">{n_open}/{max_pos}</div>
        <div class="kpi-lbl">Açık Pozisyon</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-val">{n_closed}</div>
        <div class="kpi-lbl">Kapanan Trade</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-val {wr_cls}">{win_rate}%</div>
        <div class="kpi-lbl">Kazanma Oranı</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-val {pnl_cls}">{total_pnl}%</div>
        <div class="kpi-lbl">Toplam P&L</div>
      </div>
    </div>
  </div>

  <h2>Açık Pozisyonlar</h2>
  {open_table}

  <h2>Son 20 Trade</h2>
  {trade_table}

  <p class="meta">P3-DSP &nbsp;·&nbsp; P1 ve P2-SMC'den bağımsız, paper only</p>
</body>
</html>"""


def build_dashboard_route(app, portfolio, scan_result_getter=None):
    """
    Mevcut Flask uygulamasına /p3-dsp route ekler.

    Mimari Plan Bölüm 7.3: Flask port 5002, /p3-dsp route.
    P1/P2 dashboard route'larıyla aynı pattern.

    Parameters
    ----------
    app               : Flask app instance (mevcut app.py'deki)
    portfolio         : DspPortfolio instance
    scan_result_getter: () → ScanResult | None  (opsiyonel, en son tarama)

    VM'deki entegrasyon (app.py içine):
        from notifier_dsp import build_dashboard_route
        from portfolio_dsp import DspPortfolio
        dsp_portfolio = DspPortfolio()
        build_dashboard_route(app, dsp_portfolio)
    """

    @app.route("/p3-dsp")
    def p3_dsp_dashboard():
        try:
            from flask import Response
            perf = portfolio.performance_summary()
            html = _render_dashboard(perf, portfolio.trade_history)
            return Response(html, mimetype="text/html")
        except Exception as e:
            logger.error(f"Dashboard render hatası: {e}")
            return f"<pre>Hata: {e}</pre>", 500

    return p3_dsp_dashboard


def _render_dashboard(perf: dict, trade_history: list) -> str:
    """HTML dashboard'u doldur."""

    # Açık pozisyon tablosu
    if perf["open_positions"]:
        rows = []
        for p in perf["open_positions"]:
            pnl = p.get("pnl_pct")
            pnl_str = f"{pnl:+.1f}%" if pnl is not None else "—"
            cls = "pos" if (pnl or 0) >= 0 else "neg"
            rows.append(
                f"<tr>"
                f"<td><strong>{p['symbol'].replace('.IS','')}</strong></td>"
                f"<td>{p['entry_date']}</td>"
                f"<td>{p['entry_price']:.3f}</td>"
                f"<td>{p.get('current_price') or '—'}</td>"
                f"<td class='{cls}'>{pnl_str}</td>"
                f"<td>{p['score']:+.4f}</td>"
                f"</tr>"
            )
        open_table = (
            "<table><tr>"
            "<th>Sembol</th><th>Giriş</th><th>Giriş Fiyat</th>"
            "<th>Güncel</th><th>P&L</th><th>Skor</th>"
            "</tr>" + "".join(rows) + "</table>"
        )
    else:
        open_table = "<p style='color:#8b949e'>Açık pozisyon yok.</p>"

    # Trade geçmişi (son 20)
    recent_trades = list(reversed(trade_history))[:20]
    if recent_trades:
        rows = []
        for t in recent_trades:
            cls = "pos" if t.pnl_pct >= 0 else "neg"
            rows.append(
                f"<tr>"
                f"<td>{t.symbol.replace('.IS','')}</td>"
                f"<td>{t.entry_date}</td>"
                f"<td>{t.exit_date}</td>"
                f"<td>{t.entry_price:.3f}</td>"
                f"<td>{t.exit_price:.3f}</td>"
                f"<td class='{cls}'>{t.pnl_pct:+.1f}%</td>"
                f"<td>{t.holding_days}g</td>"
                f"<td>{t.exit_reason}</td>"
                f"</tr>"
            )
        trade_table = (
            "<table><tr>"
            "<th>Sembol</th><th>Giriş</th><th>Çıkış</th>"
            "<th>G.Fiyat</th><th>Ç.Fiyat</th><th>P&L</th>"
            "<th>Süre</th><th>Neden</th>"
            "</tr>" + "".join(rows) + "</table>"
        )
    else:
        trade_table = "<p style='color:#8b949e'>Henüz kapanan trade yok.</p>"

    wr  = perf["win_rate"] * 100
    pnl = perf["total_pnl_pct"]

    return DASHBOARD_HTML.format(
        last_scan=perf["last_scan"] or "—",
        n_open=perf["n_open"],
        max_pos=perf["max_positions"],
        n_closed=perf["n_closed_trades"],
        win_rate=f"{wr:.0f}",
        wr_cls="pos" if wr >= 50 else "neg",
        total_pnl=f"{pnl:+.1f}",
        pnl_cls="pos" if pnl >= 0 else "neg",
        open_table=open_table,
        trade_table=trade_table,
    )
