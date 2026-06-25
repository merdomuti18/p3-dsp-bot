"""
nonstationarity_monitor.py — P3-DSP Faz 4
==========================================
Dominant cycle periyot kayması tespiti + adaptif filtre güncelleme.

Çalışma mantığı:
  1. Her sembol için son 60 günlük rolling Welch analizi → dominant periyot
  2. Tarihsel referans periyot ile karşılaştır (baseline = ilk 120 bar ortalaması)
  3. Kayma > %35 ise:
       a. Telegram uyarısı gönder   (pasif: her zaman)
       b. AdaptiveFilterState.refit() çağır  (aktif: eşik aşılınca)
  4. Sonuçları state'e logla (son 30 gün saklanır)

simulate_dsp.py entegrasyonu:
    from nonstationarity_monitor import NonstationarityMonitor
    monitor = NonstationarityMonitor()
    alerts  = monitor.run(symbols_prices, state)
    if alerts:
        notifier.send(monitor.telegram_message(alerts))

KRİTİK KURAL: filtfilt KULLANILMAZ.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

DRIFT_THRESHOLD   = 0.35    # %35 periyot kayması → uyarı + adaptasyon
BASELINE_WINDOW   = 120     # Referans periyot için ilk N bar
DETECTION_WINDOW  = 60      # Günlük Welch analizi penceresi (bar)
PERIOD_MIN        = 15.0    # Cycle arama alt sınırı (gün)
PERIOD_MAX        = 60.0    # Cycle arama üst sınırı (gün)
FALLBACK_PERIOD   = 30.0    # Cycle güvenilir değilse varsayılan (gün)
LOG_KEEP_DAYS     = 30      # State'de kaç günlük log saklanır


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class SymbolCycleState:
    """
    Tek sembol için cycle izleme durumu.
    State JSON'da 'cycle_monitor' anahtarı altında saklanır.
    """
    symbol:           str
    baseline_period:  float         # İlk ölçüm — referans
    last_period:      float         # En son ölçüm
    drift_pct:        float         # (last - baseline) / baseline
    is_drifted:       bool          # drift_pct > DRIFT_THRESHOLD
    peakiness:        float         # Welch güvenilirlik skoru
    is_reliable:      bool          # peakiness > 2.5
    refit_applied:    bool          # Bu günde refit yapıldı mı
    check_date:       str           # Kontrol tarihi (ISO)

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "baseline_period": round(self.baseline_period, 2),
            "last_period":     round(self.last_period, 2),
            "drift_pct":       round(self.drift_pct, 4),
            "is_drifted":      self.is_drifted,
            "peakiness":       round(self.peakiness, 3),
            "is_reliable":     self.is_reliable,
            "refit_applied":   self.refit_applied,
            "check_date":      self.check_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolCycleState":
        return cls(**d)


@dataclass
class MonitorResult:
    """
    monitor.run() dönüş tipi — günlük özet.
    """
    check_date:      str
    total_checked:   int
    reliable_count:  int            # peakiness > 2.5 olan sembol sayısı
    drifted_count:   int            # Eşik aşan sembol sayısı
    refit_count:     int            # Filtre güncellenen sembol sayısı
    alerts:          list[SymbolCycleState]    # Drift tespit edilenler
    stable:          list[SymbolCycleState]    # Stabil kalanlar
    elapsed_sec:     float

    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    def summary_line(self) -> str:
        return (
            f"[Non-Stat] {self.check_date} | "
            f"Kontrol: {self.total_checked} | "
            f"Güvenilir: {self.reliable_count} | "
            f"Drift: {self.drifted_count} | "
            f"Refit: {self.refit_count} | "
            f"{self.elapsed_sec:.1f}s"
        )


# ---------------------------------------------------------------------------
# Ana Monitör
# ---------------------------------------------------------------------------

class NonstationarityMonitor:
    """
    P3-DSP Non-stationarity Monitörü.

    Kullanım (simulate_dsp.py içinde):
        monitor = NonstationarityMonitor()
        result  = monitor.run(symbols_prices, state)
        if result.has_alerts():
            notifier.send(monitor.telegram_message(result))
        # state otomatik güncellendi, save_state() çağır
    """

    def __init__(
        self,
        drift_threshold:  float = DRIFT_THRESHOLD,
        baseline_window:  int   = BASELINE_WINDOW,
        detection_window: int   = DETECTION_WINDOW,
        period_min:       float = PERIOD_MIN,
        period_max:       float = PERIOD_MAX,
        fallback_period:  float = FALLBACK_PERIOD,
        filter_order:     int   = 3,
    ):
        self.drift_threshold  = drift_threshold
        self.baseline_window  = baseline_window
        self.detection_window = detection_window
        self.period_min       = period_min
        self.period_max       = period_max
        self.fallback_period  = fallback_period
        self.filter_order     = filter_order

        # AdaptiveFilterState — aktif adaptasyon için
        from dsp_core import AdaptiveFilterState
        self._adaptive = AdaptiveFilterState(
            refit_interval_days=1,      # Her gün refit yapabilir
            lookback_days=detection_window,
            filter_order=filter_order,
            period_min=period_min,
            period_max=period_max,
            fallback_period=fallback_period,
        )

    # --- Ana Akış ---

    def run(
        self,
        symbols_prices: dict[str, np.ndarray],
        state: dict,
    ) -> MonitorResult:
        """
        Tüm sembolleri kontrol et, state'i güncelle, sonuç döndür.

        Parameters
        ----------
        symbols_prices : {symbol: prices_array} — scanner'dan gelen fiyat verileri
        state          : simulate_dsp.py'nin portfolio_state dict'i
                         'cycle_monitor' anahtarı altında güncellenir.

        Returns
        -------
        MonitorResult
        """
        import time
        t0    = time.perf_counter()
        today = date.today().isoformat()

        # State altında cycle_monitor bölümü
        if "cycle_monitor" not in state:
            state["cycle_monitor"] = {
                "baselines": {},    # {symbol: baseline_period}
                "log":       [],    # günlük kayıt (son 30)
            }
        cm = state["cycle_monitor"]

        alerts  : list[SymbolCycleState] = []
        stable  : list[SymbolCycleState] = []
        reliable_count = refit_count = 0

        for symbol, prices in symbols_prices.items():
            prices = np.asarray(prices, dtype=float)
            if len(prices) < self.baseline_window + 10:
                continue

            # 1. Mevcut dominant periyodu tespit et
            current = self._detect_period(prices)

            # 2. Baseline: ilk ölçüm — bir kez hesaplanır, saklanır
            if symbol not in cm["baselines"]:
                baseline_p = self._detect_period(
                    prices[:self.baseline_window]
                )
                cm["baselines"][symbol] = {
                    "period":   round(baseline_p.period, 2),
                    "set_date": today,
                }

            baseline_period = cm["baselines"][symbol]["period"]

            # 3. Drift hesapla
            drift_pct = (current.period - baseline_period) / (baseline_period + 1e-9)
            is_drifted = abs(drift_pct) > self.drift_threshold

            if current.is_reliable:
                reliable_count += 1

            # 4. Aktif adaptasyon — eşik aşıldıysa refit
            refit_applied = False
            if is_drifted and current.is_reliable:
                try:
                    self._adaptive.refit(symbol, prices[-self.detection_window:], date.today())
                    refit_applied = True
                    refit_count  += 1
                    logger.info(
                        f"{symbol}: Refit — baseline={baseline_period:.1f}g "
                        f"→ current={current.period:.1f}g (drift={drift_pct:+.1%})"
                    )
                except Exception as e:
                    logger.warning(f"{symbol} refit hatası: {e}")

            cs = SymbolCycleState(
                symbol=symbol,
                baseline_period=baseline_period,
                last_period=current.period,
                drift_pct=drift_pct,
                is_drifted=is_drifted,
                peakiness=current.peakiness,
                is_reliable=current.is_reliable,
                refit_applied=refit_applied,
                check_date=today,
            )

            if is_drifted:
                alerts.append(cs)
            else:
                stable.append(cs)

        # Drift büyüklüğüne göre sırala (en kritik önce)
        alerts.sort(key=lambda x: abs(x.drift_pct), reverse=True)

        elapsed = round(time.perf_counter() - t0, 2)

        result = MonitorResult(
            check_date=today,
            total_checked=len(symbols_prices),
            reliable_count=reliable_count,
            drifted_count=len(alerts),
            refit_count=refit_count,
            alerts=alerts,
            stable=stable,
            elapsed_sec=elapsed,
        )

        # 5. Günlük log — son 30 gün sakla
        cm["log"].append({
            "date":          today,
            "total":         result.total_checked,
            "reliable":      result.reliable_count,
            "drifted":       result.drifted_count,
            "refit":         result.refit_count,
            "alert_symbols": [a.symbol for a in alerts],
        })
        cm["log"] = cm["log"][-LOG_KEEP_DAYS:]

        return result

    # --- Cycle Tespiti ---

    @dataclass
    class _CycleDetection:
        period:      float
        peakiness:   float
        is_reliable: bool

    def _detect_period(self, prices: np.ndarray) -> "_CycleDetection":
        """
        Rolling pencerede Welch → dominant periyot.
        Güvenilir değilse fallback_period döner.
        """
        from dsp_core import detect_dominant_cycle
        import warnings

        window = prices[-self.detection_window:]

        if len(window) < 32:
            return self._CycleDetection(
                period=self.fallback_period,
                peakiness=0.0,
                is_reliable=False,
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = detect_dominant_cycle(
                    window,
                    period_min=self.period_min,
                    period_max=self.period_max,
                )
            period = float(np.clip(
                result.dominant_period, self.period_min, self.period_max
            )) if result.is_reliable else self.fallback_period

            return self._CycleDetection(
                period=period,
                peakiness=result.peakiness,
                is_reliable=result.is_reliable,
            )
        except Exception as e:
            logger.debug(f"Cycle tespit hatası: {e}")
            return self._CycleDetection(
                period=self.fallback_period,
                peakiness=0.0,
                is_reliable=False,
            )

    # --- Getter: güncellenmiş filtre katsayıları ---

    def get_adapted_coeffs(self, symbol: str):
        """
        Refit yapılmış sembol için güncellenmiş (b, a) döndürür.
        Refit yapılmamışsa None.
        """
        try:
            return self._adaptive.get_coeffs(symbol)
        except ValueError:
            return None

    # --- Telegram Mesajı ---

    def telegram_message(self, result: MonitorResult) -> str:
        """
        Telegram'a gönderilecek non-stationarity uyarı mesajı.
        Sadece drift tespit edildiğinde çağrılır.
        """
        lines = [
            f"⚠️ *[P3-DSP] Non-Stationarity Uyarısı* — {result.check_date}",
            f"Kontrol: {result.total_checked} sembol | "
            f"Drift: {result.drifted_count} | "
            f"Refit: {result.refit_count}",
            "",
        ]

        for a in result.alerts[:8]:   # Maksimum 8 sembol listele
            direction = "↑" if a.drift_pct > 0 else "↓"
            reliable  = "🔵" if a.is_reliable else "⚪"
            refit_tag = " ♻️ refit" if a.refit_applied else ""
            lines.append(
                f"{reliable} *{a.symbol}* {direction}{abs(a.drift_pct):.0%}  "
                f"({a.baseline_period:.0f}g → {a.last_period:.0f}g){refit_tag}"
            )

        if result.drifted_count > 8:
            lines.append(f"_...ve {result.drifted_count - 8} sembol daha_")

        lines += [
            "",
            f"🔵 = Welch güvenilir (peakiness>2.5)  ⚪ = zayıf sinyal",
            f"♻️ = Filtre katsayıları güncellendi",
        ]

        return "\n".join(lines)

    # --- HTML Rapor Bölümü ---

    def html_section(self, result: MonitorResult) -> str:
        """
        simulate_dsp.py HTML raporuna eklenecek non-stationarity bölümü.
        """
        if not result.has_alerts():
            return (
                "<p style='color:#3fb950;padding:8px'>"
                "✅ Tüm semboller stabil — periyot kayması tespit edilmedi."
                "</p>"
            )

        rows = []
        for a in result.alerts:
            drift_cls  = "neg" if abs(a.drift_pct) > 0.50 else "neu"
            direction  = "↑" if a.drift_pct > 0 else "↓"
            refit_tag  = "<span style='color:#58a6ff'>♻️</span>" if a.refit_applied else ""
            reliable   = "🔵" if a.is_reliable else "⚪"
            rows.append(
                f"<tr>"
                f"<td><strong>{a.symbol}</strong></td>"
                f"<td>{a.baseline_period:.0f}g</td>"
                f"<td>{a.last_period:.0f}g</td>"
                f"<td class='{drift_cls}'>{direction}{abs(a.drift_pct):.0%}</td>"
                f"<td>{a.peakiness:.2f} {reliable}</td>"
                f"<td>{refit_tag}</td>"
                f"</tr>"
            )

        return (
            "<table>"
            "<tr><th>Sembol</th><th>Baseline</th><th>Güncel</th>"
            "<th>Kayma</th><th>Peakiness</th><th>Refit</th></tr>"
            + "".join(rows)
            + "</table>"
        )


# ---------------------------------------------------------------------------
# simulate_dsp.py Entegrasyon Yardımcısı
# ---------------------------------------------------------------------------

def fetch_prices_for_monitor(
    symbols: list[str],
    lookback_days: int = 300,
) -> dict[str, np.ndarray]:
    """
    simulate_dsp.py'de kullanım için yfinance'ten fiyat çek.
    YfDataAdapter ile aynı mantık — monitöre özel hafif versiyon.
    """
    import yfinance as yf

    result = {}
    for sym in symbols:
        try:
            ticker = f"{sym}.IS" if not sym.endswith(".IS") else sym
            df = yf.Ticker(ticker).history(period="max")
            prices = df["Close"].dropna().values
            if len(prices) >= lookback_days:
                result[sym] = prices[-lookback_days:]
            elif len(prices) >= 130:
                result[sym] = prices
        except Exception as e:
            logger.debug(f"{sym} fiyat hatası: {e}")
    return result


# ---------------------------------------------------------------------------
# CLI — Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="P3-DSP Non-Stationarity Monitör")
    parser.add_argument("--symbols", nargs="+",
                        default=["GARAN", "AKBNK", "THYAO", "EREGL",
                                 "ASELS", "BIMAS", "TUPRS", "KCHOL",
                                 "TCELL", "TOASO"],
                        help="Test edilecek semboller")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD,
                        help="Drift eşiği (varsayılan: 0.35)")
    parser.add_argument("--bars", type=int, default=300,
                        help="Geçmiş bar sayısı")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  P3-DSP Non-Stationarity Monitör Testi")
    print(f"  Semboller: {', '.join(args.symbols)}")
    print(f"  Eşik: %{args.threshold*100:.0f} | Bars: {args.bars}")
    print(f"{'='*60}\n")

    print("Veri çekiliyor...")
    prices_map = fetch_prices_for_monitor(args.symbols, lookback_days=args.bars)
    print(f"{len(prices_map)} sembol yüklendi.\n")

    monitor = NonstationarityMonitor(drift_threshold=args.threshold)
    state   = {}   # Boş state — ilk çalıştırma
    result  = monitor.run(prices_map, state)

    print(result.summary_line())
    print()

    # Tüm sembol detayı
    all_states = result.alerts + result.stable
    all_states.sort(key=lambda x: abs(x.drift_pct), reverse=True)

    print(f"  {'Sembol':<10} {'Baseline':>9} {'Güncel':>7} {'Kayma':>8} "
          f"{'Peak':>6} {'Güv':>4} {'Drift':>6} {'Refit':>5}")
    print(f"  {'─'*10} {'─'*9} {'─'*7} {'─'*8} "
          f"{'─'*6} {'─'*4} {'─'*6} {'─'*5}")

    for cs in all_states:
        drift_flag = "⚠️ " if cs.is_drifted else "  "
        rel_flag   = "🔵" if cs.is_reliable else "⚪"
        refit_flag = "♻️ " if cs.refit_applied else "  "
        direction  = "↑" if cs.drift_pct > 0 else "↓"
        print(f"  {cs.symbol:<10} {cs.baseline_period:>8.1f}g "
              f"{cs.last_period:>6.1f}g "
              f"{direction}{abs(cs.drift_pct):>6.0%}  "
              f"{cs.peakiness:>6.2f} {rel_flag}  "
              f"{drift_flag}  {refit_flag}")

    print(f"\n{'─'*60}")
    if result.has_alerts():
        print("\n📨 Telegram Mesajı Önizleme:")
        print("─" * 40)
        print(monitor.telegram_message(result))
    else:
        print("✅ Tüm semboller stabil.")

    # State özeti
    print(f"\n{'─'*60}")
    print(f"State cycle_monitor baselines: "
          f"{len(state.get('cycle_monitor', {}).get('baselines', {}))} sembol")