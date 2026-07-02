"""
simulate_dsp.py — P3-DSP Günlük Simülasyon Motoru
===================================================
Her gün 19:00'da GitHub Actions tarafından çalıştırılır.

Görev:
  1. Gerçek veri ile 20 sembol tara (yfinance → tvdatafeed yedek)
  2. Portfolio state'i güncelle (giriş/çıkış kararları)
  3. HTML simülasyon raporu üret → reports/latest.html
  4. Geçmiş raporu reports/history/ altına arşivle

State dosyası: portfolio_state.json (repo'da saklanır, her run güncellenir)
"""

from __future__ import annotations

import json
import os
import sys
import logging
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scanner_dsp import (
    DspScanner, RealDataAdapter,
    MockDataAdapter, SYMBOL_UNIVERSE_SAMPLE, ScanResult
)
from dsp_strategies import XoverParams
from nonstationarity_monitor import NonstationarityMonitor, fetch_prices_for_monitor
from correlation_analysis import PortfolioCorrelation, run_portfolio_correlation

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

STATE_FILE   = PROJECT_ROOT / "portfolio_state.json"
REPORT_DIR   = PROJECT_ROOT / "reports"
HISTORY_DIR  = REPORT_DIR / "history"
MAX_POS      = 5
POS_SIZE_PCT = 20.0   # her pozisyon %20
EMERGENCY_STOP_PCT = -8.0   # tek pozisyon acil durdurma eşiği (%)

PARAMS = XoverParams(fast_period=15, slow_period=40, order=3)


# ---------------------------------------------------------------------------
# State Yönetimi
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "created": date.today().isoformat(),
        "positions": {},       # {symbol: {entry_date, entry_price, score}}
        "history":   [],       # kapanan trade listesi
        "scan_log":  [],       # günlük tarama özeti (son 30)
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Portfolio Güncelleme
# ---------------------------------------------------------------------------

def update_portfolio(state: dict, scan: ScanResult) -> dict:
    """
    Tarama sonucuna göre giriş/çıkış kararları ver.
    Fiyat bilgisi olmadığı için entry/exit price yfinance'ten alınır.
    """
    today      = date.today().isoformat()
    long_set   = {s.symbol for s in scan.top_longs}
    positions  = state["positions"]
    actions    = {"entries": [], "exits": [], "holds": [], "skipped": []}

    # Çıkışlar
    for sym in list(positions.keys()):
        if sym not in long_set:
            pos = positions.pop(sym)
            exit_price = _get_price(sym)
            pnl = 0.0
            if exit_price and pos.get("entry_price"):
                pnl = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

            entry_d = date.fromisoformat(pos["entry_date"])
            holding = (date.today() - entry_d).days

            state["history"].append({
                "symbol":      sym,
                "entry_date":  pos["entry_date"],
                "exit_date":   today,
                "entry_price": pos.get("entry_price"),
                "exit_price":  exit_price,
                "pnl_pct":     round(pnl, 2),
                "holding_days": holding,
            })
            actions["exits"].append(sym)
        else:
            actions["holds"].append(sym)

    # Girişler
    for sc in scan.top_longs:
        sym = sc.symbol
        if sym in positions:
            continue
        if len(positions) >= MAX_POS:
            actions["skipped"].append(sym)
            continue
        price = _get_price(sym)
        positions[sym] = {
            "entry_date":  today,
            "entry_price": price,
            "score":       round(sc.score, 4),
            "margin":      round(sc.crossover_margin, 4),
        }
        actions["entries"].append(sym)

    # Günlük log (son 60 gün sakla)
    state["scan_log"].append({
        "date":         today,
        "scanned":      scan.scanned,
        "long_signals": scan.long_signals,
        "top5":         [s.symbol for s in scan.top_longs[:5]],
        "entries":      actions["entries"],
        "exits":        actions["exits"],
    })
    state["scan_log"] = state["scan_log"][-60:]

    return actions


def _get_price(symbol: str) -> float | None:
    """Son kapanış fiyatını yfinance'ten çek."""
    try:
        import yfinance as yf
        ticker = f"{symbol}.IS" if not symbol.endswith(".IS") else symbol
        df = yf.Ticker(ticker).history(period="5d")
        df = df.dropna(subset=["Close"])
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            if price == price:  # NaN kontrolü (NaN != NaN)
                return round(price, 3)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Performans Hesaplama
# ---------------------------------------------------------------------------

def calc_performance(state: dict) -> dict:
    history = state["history"]
    if not history:
        return {
            "n_closed": 0, "win_rate": 0, "avg_pnl": 0,
            "total_pnl": 0, "avg_hold": 0, "best": None, "worst": None
        }
    pnls  = [t["pnl_pct"] for t in history if t.get("pnl_pct") is not None and not (t["pnl_pct"] != t["pnl_pct"])]
    holds = [t["holding_days"] for t in history if t.get("holding_days")]
    wins  = sum(1 for p in pnls if p > 0)

    sorted_trades = sorted(history, key=lambda x: x.get("pnl_pct", 0))
    return {
        "n_closed":  len(history),
        "win_rate":  round(wins / len(pnls) * 100) if pnls else 0,
        "avg_pnl":   round(sum(pnls) / len(pnls), 2) if pnls else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_hold":  round(sum(holds) / len(holds), 1) if holds else 0,
        "best":      sorted_trades[-1] if sorted_trades else None,
        "worst":     sorted_trades[0]  if sorted_trades else None,
    }


# ---------------------------------------------------------------------------
# HTML Rapor
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>P3-DSP Simülasyon Raporu</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117;
            color: #c9d1d9; padding: 20px; max-width: 1100px; margin: 0 auto; }}
    h1   {{ color: #58a6ff; padding: 16px 0 8px; border-bottom: 1px solid #30363d; }}
    h2   {{ color: #79c0ff; margin: 28px 0 10px; font-size: 1.05em; }}
    .meta {{ color: #8b949e; font-size: 0.85em; margin: 6px 0 20px; }}
    .kpi  {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr));
             gap: 10px; margin-bottom: 24px; }}
    .kpi-card {{ background: #161b22; border: 1px solid #30363d;
                 border-radius: 8px; padding: 14px; text-align: center; }}
    .kpi-val  {{ font-size: 1.7em; font-weight: 700; color: #58a6ff; }}
    .kpi-lbl  {{ font-size: 0.75em; color: #8b949e; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    th    {{ background: #161b22; color: #8b949e; padding: 8px 10px;
             text-align: left; font-size: 0.8em; text-transform: uppercase; }}
    td    {{ padding: 8px 10px; border-bottom: 1px solid #21262d; }}
    tr:hover td {{ background: #161b22; }}
    .pos  {{ color: #3fb950; font-weight: 600; }}
    .neg  {{ color: #f85149; font-weight: 600; }}
    .neu  {{ color: #8b949e; }}
    .badge-long {{ background: #238636; color: #fff; padding: 2px 8px;
                   border-radius: 10px; font-size: 0.8em; }}
    .badge-flat {{ background: #30363d; color: #8b949e; padding: 2px 8px;
                   border-radius: 10px; font-size: 0.8em; }}
    .scan-row {{ background: #161b22; border-radius: 6px; padding: 8px 12px;
                 margin: 4px 0; font-size: 0.85em; display: flex;
                 justify-content: space-between; align-items: center; }}
    .scan-date {{ color: #58a6ff; min-width: 90px; }}
    footer {{ color: #30363d; font-size: 0.75em; margin-top: 40px;
              text-align: center; border-top: 1px solid #21262d; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>📡 P3-DSP Simülasyon Raporu</h1>
  <p class="meta">
    Son güncelleme: <strong>{update_time}</strong> &nbsp;·&nbsp;
    Strateji: <strong>DSP-XOVER (fast=15g, slow=40g, order=3)</strong> &nbsp;·&nbsp;
    Paper trading — gerçek sermaye yok
  </p>

  <div class="kpi">
    <div class="kpi-card">
      <div class="kpi-val">{n_open}/{max_pos}</div>
      <div class="kpi-lbl">Açık Pozisyon</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">{n_closed}</div>
      <div class="kpi-lbl">Kapanan Trade</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val {wr_cls}">{win_rate}%</div>
      <div class="kpi-lbl">Kazanma Oranı</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val {pnl_cls}">{total_pnl}%</div>
      <div class="kpi-lbl">Toplam P&L</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">{avg_hold}g</div>
      <div class="kpi-lbl">Ort. Süre</div>
    </div>
  </div>

  <h2>📂 Açık Pozisyonlar</h2>
  {open_table}

  <h2>📋 Son Tarama — {scan_date}</h2>
  {scan_table}

  <h2>📈 Trade Geçmişi</h2>
  {history_table}

  <h2>🗓️ Günlük Tarama Günlüğü (son 14 gün)</h2>
  {scan_log}

  <h2>🔬 Non-Stationarity Monitör</h2>
  {monitor_section}

  <h2>🔗 Portföy Korelasyon Analizi</h2>
  {correlation_section}

  <footer>
    P3-DSP · Frekans domeni tabanlı paper strateji ·
    P1 ve P2-SMC'den bağımsız · {update_time}
  </footer>
</body>
</html>"""


def build_report(
    state: dict,
    scan: ScanResult,
    actions: dict,
    perf: dict,
    monitor_result=None,
    corr_result=None,
) -> str:
    today  = date.today().isoformat()
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Açık pozisyonlar
    positions = state["positions"]
    if positions:
        rows = []
        for sym, pos in positions.items():
            price    = _get_price(sym)
            ep       = pos.get("entry_price")
            pnl      = round((price - ep) / ep * 100, 2) if (price and ep) else None
            pnl_str  = f"{pnl:+.1f}%" if pnl is not None else "—"
            cls      = "pos" if (pnl or 0) >= 0 else "neg"
            entry_flag = "🟢 YENİ" if sym in actions["entries"] else ""
            ep_str = f"{ep:.3f}" if ep else "—"
            pr_str = f"{price:.3f}" if price else "—"
            rows.append(f"""<tr>
              <td><strong>{sym}</strong> {entry_flag}</td>
              <td>{pos['entry_date']}</td>
              <td>{ep_str}</td>
              <td>{pr_str}</td>
              <td class='{cls}'>{pnl_str}</td>
              <td class='neu'>{pos.get('score',0):+.4f}</td>
            </tr>""")
        open_table = f"""<table>
          <tr><th>Sembol</th><th>Giriş</th><th>G.Fiyat</th>
              <th>Güncel</th><th>P&L</th><th>Skor</th></tr>
          {''.join(rows)}</table>"""
    else:
        open_table = "<p class='neu' style='padding:8px'>Açık pozisyon yok.</p>"

    # Son tarama sinyalleri
    all_scores = scan.all_scores
    scan_rows = []
    for sc in all_scores:
        if sc.score == -999:
            continue
        badge = "<span class='badge-long'>LONG</span>" if sc.is_long \
                else "<span class='badge-flat'>flat</span>"
        cls = "pos" if sc.score > 0 else "neg"
        scan_rows.append(f"""<tr>
          <td><strong>{sc.symbol}</strong></td>
          <td>{badge}</td>
          <td class='{cls}'>{sc.score:+.4f}</td>
          <td class='{cls}'>{sc.crossover_margin:+.3f}</td>
          <td class='{cls}'>{sc.momentum_3w:+.3f}</td>
        </tr>""")
    scan_table = f"""<table>
      <tr><th>Sembol</th><th>Sinyal</th><th>Skor</th>
          <th>Marj</th><th>Mom(15g)</th></tr>
      {''.join(scan_rows)}</table>"""

    # Trade geçmişi (son 20, en yeni üstte)
    history = list(reversed(state["history"]))[:20]
    if history:
        h_rows = []
        for t in history:
            pnl = t.get("pnl_pct", 0) or 0
            cls = "pos" if pnl >= 0 else "neg"
            h_rows.append(f"""<tr>
              <td>{t['symbol']}</td>
              <td>{t['entry_date']}</td>
              <td>{t['exit_date']}</td>
              <td>{t.get('entry_price') or '—'}</td>
              <td>{t.get('exit_price') or '—'}</td>
              <td class='{cls}'>{pnl:+.1f}%</td>
              <td>{t.get('holding_days','—')}g</td>
            </tr>""")
        history_table = f"""<table>
          <tr><th>Sembol</th><th>Giriş</th><th>Çıkış</th>
              <th>G.Fiyat</th><th>Ç.Fiyat</th><th>P&L</th><th>Süre</th></tr>
          {''.join(h_rows)}</table>"""
    else:
        history_table = "<p class='neu' style='padding:8px'>Henüz kapanan trade yok.</p>"

    # Günlük log (son 14)
    log_items = list(reversed(state["scan_log"]))[:14]
    log_html = ""
    for lg in log_items:
        entries_str = f"🟢 {', '.join(lg['entries'])}" if lg.get('entries') else ""
        exits_str   = f"⬜ {', '.join(lg['exits'])}"   if lg.get('exits')   else ""
        top5_str    = ", ".join(lg.get('top5', []))
        log_html += f"""<div class='scan-row'>
          <span class='scan-date'>{lg['date']}</span>
          <span>{lg['scanned']} sembol · {lg['long_signals']} long</span>
          <span style='color:#58a6ff'>↑ {top5_str}</span>
          <span class='pos'>{entries_str}</span>
          <span class='neg'>{exits_str}</span>
        </div>"""
    if not log_html:
        log_html = "<p class='neu' style='padding:8px'>Henüz log yok.</p>"

    # Korelasyon bölümü
    if corr_result is not None:
        corr_obj = PortfolioCorrelation()
        correlation_section = corr_obj.html_section(corr_result)
    else:
        correlation_section = (
            "<p class='neu' style='padding:8px'>"
            "Korelasyon analizi için yeterli açık pozisyon yok (min 2)."
            "</p>"
        )

    # Non-stationarity monitör bölümü
    if monitor_result is not None:
        from nonstationarity_monitor import NonstationarityMonitor
        monitor_section = NonstationarityMonitor().html_section(monitor_result)
        drift_kpi = f"{monitor_result.drifted_count}/{monitor_result.total_checked}"
    else:
        monitor_section = "<p class='neu' style='padding:8px'>Monitör bu çalıştırmada devre dışı.</p>"
        drift_kpi = "—"

    # KPI renkleri
    wr  = perf["win_rate"]
    pnl = perf["total_pnl"]

    return HTML_TEMPLATE.format(
        update_time=now,
        n_open=len(positions),
        max_pos=MAX_POS,
        n_closed=perf["n_closed"],
        win_rate=wr,
        wr_cls="pos" if wr >= 50 else "neg",
        total_pnl=f"{pnl:+.1f}",
        pnl_cls="pos" if pnl >= 0 else "neg",
        avg_hold=perf["avg_hold"],
        scan_date=today,
        open_table=open_table,
        scan_table=scan_table,
        history_table=history_table,
        scan_log=log_html,
        monitor_section=monitor_section,
        correlation_section=correlation_section,
    )


# ---------------------------------------------------------------------------
# Gün İçi Acil Stop Kontrolü (hafif — tam tarama yapmaz)
# ---------------------------------------------------------------------------

def monitor() -> dict:
    """
    Gün içi hafif kontrol (her ~15 dk çağrılır): P3'ün normalde tek karar
    mekanizması olan günlük sinyal rotasyonudur (yalnızca akşam çalışır),
    yani gün içinde STOP/TP kavramı yoktu. Burada, açık bir pozisyon
    beklenmedik şekilde büyük düşerse (ör. %8+) günü beklemeden ACİL STOP
    ile pozisyonu kapatıp anında Telegram gönderen bir güvenlik ağı
    ekliyoruz. Tam 79 sembollük günlük tarama/rotasyon burada YAPILMAZ.
    """
    state = load_state()
    positions = state["positions"]
    if not positions:
        print("P3 monitor: açık pozisyon yok")
        return {"kapanan": []}

    today = date.today().isoformat()
    kapananlar = []
    for sym in list(positions.keys()):
        pos = positions[sym]
        price = _get_price(sym)
        ep = pos.get("entry_price")
        if not price or not ep:
            continue
        pnl = (price - ep) / ep * 100
        if pnl <= EMERGENCY_STOP_PCT:
            positions.pop(sym)
            try:
                entry_d = date.fromisoformat(pos["entry_date"])
                holding = (date.today() - entry_d).days
            except Exception:
                holding = None
            state["history"].append({
                "symbol":       sym,
                "entry_date":   pos["entry_date"],
                "exit_date":    today,
                "entry_price":  ep,
                "exit_price":   price,
                "pnl_pct":      round(pnl, 2),
                "holding_days": holding,
                "reason":       "ACIL_STOP",
            })
            kapananlar.append({"symbol": sym, "pnl_pct": round(pnl, 2)})
            print(f"🛑 ACİL STOP: {sym} {pnl:+.1f}%")

    if kapananlar:
        save_state(state)
        try:
            from mott_telegram import telegram_islem_gonder
            mesajlar = [f"🛑 *ACİL STOP - {k['symbol']}*\n   Getiri: {k['pnl_pct']:+.1f}%" for k in kapananlar]
            telegram_islem_gonder(
                "P3", sinyaller=[], portfoy=state,
                giris=[], cikis=[k["symbol"] for k in kapananlar],
                top_longs=[], mesajlar=mesajlar,
            )
        except Exception as e:
            print(f"P3 monitor Telegram hatası: {e}")
    else:
        print("P3 monitor: tetiklenen acil stop yok")

    return {"kapanan": kapananlar}


# ---------------------------------------------------------------------------
# Ana Akış
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*55}")
    print(f"  P3-DSP Günlük Simülasyon — {date.today()}")
    print(f"{'='*55}")

    # 1. Veri adaptörü seç
    # GitHub Actions'da yfinance daha güvenilir
    # Yerel çalıştırmada tvdatafeed önce denenir
    use_tv = os.getenv("DSP_USE_TV", "0") == "1"

    if use_tv:
        print("Veri: TradingView")
        adapter = RealDataAdapter()
    else:
        print("Veri: yfinance")
        adapter = RealDataAdapter()

    # 2. Tara
    scanner = DspScanner(
        data_adapter=adapter,
        params=PARAMS,
        top_n=MAX_POS,
        symbols=SYMBOL_UNIVERSE_SAMPLE,
    )
    print(f"{len(scanner.symbols)} sembol taranıyor...")
    scan = scanner.run(verbose=True)
    print(f"\n{scan.summary_line()}")

    # 3. State yükle + güncelle
    state   = load_state()
    actions = update_portfolio(state, scan)
    perf    = calc_performance(state)

    # 3b. Non-stationarity monitör — her günlük taramada çalışır
    print("\nNon-stationarity kontrolü...")
    try:
        monitor       = NonstationarityMonitor(drift_threshold=0.35)
        prices_map    = fetch_prices_for_monitor(
            SYMBOL_UNIVERSE_SAMPLE, lookback_days=300
        )
        monitor_result = monitor.run(prices_map, state)
        print(monitor_result.summary_line())

        if monitor_result.has_alerts():
            print(f"⚠️  Drift tespit edildi: "
                  f"{[a.symbol for a in monitor_result.alerts]}")
        else:
            print("✅ Tüm semboller stabil.")
    except Exception as e:
        logger.warning(f"Non-stationarity monitör hatası: {e}")
        monitor_result = None

    save_state(state)

    # 3c. Portföy korelasyon analizi
    print("\nKorelasyon analizi...")
    try:
        corr_result = run_portfolio_correlation(state)
        if corr_result:
            print(corr_result.summary_line())
        else:
            print("Korelasyon: yeterli pozisyon yok (min 2).")
    except Exception as e:
        logger.warning(f"Korelasyon analizi hatası: {e}")
        corr_result = None

    print(f"\nGiriş : {actions['entries'] or '—'}")
    print(f"Çıkış : {actions['exits']   or '—'}")
    print(f"Devam : {actions['holds']   or '—'}")
    print(f"Skip  : {actions['skipped'] or '—'}")
    print(f"\nAçık pozisyon: {len(state['positions'])}/{MAX_POS}")
    print(f"Toplam P&L   : {perf['total_pnl']:+.1f}%  "
          f"({perf['n_closed']} trade, win={perf['win_rate']}%)")

    # 4. HTML rapor üret
    REPORT_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)

    html = build_report(state, scan, actions, perf, monitor_result, corr_result)

    report_path = REPORT_DIR / "latest.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nRapor: {report_path}")

    # Arşiv kopyası
    archive = HISTORY_DIR / f"{date.today().isoformat()}.html"
    archive.write_text(html, encoding="utf-8")

    # P3 Telegram — yalnızca akşam modunda ve alım/satım varsa
    if os.getenv("MOTT_MODE", "aksam") == "aksam":
        try:
            from mott_telegram import telegram_islem_gonder
            telegram_islem_gonder(
                "P3",
                sinyaller=[],
                portfoy=state,
                giris=actions.get("entries"),
                cikis=actions.get("exits"),
                top_longs=scan.top_longs,
                monitor_alert=monitor_result.has_alerts() if monitor_result else False,
                corr_risk=corr_result.risk_level if corr_result else "LOW",
            )
        except Exception as e:
            print(f"P3 Telegram hatası: {e}")
    print(f"{'='*55}\nSimülasyon tamamlandı ✅\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor()
    else:
        main()

