"""
correlation_analysis.py — P3-DSP Faz 4
========================================
Portföy korelasyon analizi — iki katman:

  1. P3 İç Korelasyon:
     "Portföydeki 5 hisse gerçekten çeşitlendirilmiş mi?"
     → Günlük getiri korelasyon matrisi
     → Ortalama pairwise korelasyon
     → Uyarı: ort. korelasyon > 0.70 ise konsantrasyon riski

  2. Evren Korelasyonu (opsiyonel):
     Tüm tarama evreninde korelasyon kümeleri
     → Hangi sektörler birlikte hareket ediyor?
     → Portföy seçimi için çeşitlendirme skoru

  Not: P1/P2 verisi geldiğinde strateji korelasyonu
       buraya eklenecek (Faz 4 → Faz 5 geçişi).

simulate_dsp.py entegrasyonu:
    from correlation_analysis import PortfolioCorrelation
    corr = PortfolioCorrelation()
    result = corr.run(open_positions, all_symbols)
    html  = corr.html_section(result)

Çalıştır:
    python correlation_analysis.py
    python correlation_analysis.py --symbols GARAN AKBNK THYAO EREGL ASELS
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

CORR_WARN_THRESHOLD  = 0.70   # Ortalama korelasyon uyarı eşiği
CORR_HIGH_THRESHOLD  = 0.85   # Çok yüksek korelasyon — kritik
LOOKBACK_DAYS        = 120    # Korelasyon hesabı için geriye bakış (bar)
MIN_BARS             = 60     # Minimum veri gereksinimi


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class PairCorrelation:
    """İki sembol arasındaki korelasyon."""
    symbol_a:    str
    symbol_b:    str
    correlation: float
    is_high:     bool      # > CORR_WARN_THRESHOLD

    def label(self) -> str:
        if self.correlation > CORR_HIGH_THRESHOLD:
            return "🔴 KRİTİK"
        elif self.correlation > CORR_WARN_THRESHOLD:
            return "⚠️  YÜKSEK"
        elif self.correlation > 0.40:
            return "🟡 ORTA"
        else:
            return "🟢 DÜŞÜK"


@dataclass
class CorrelationResult:
    """run() dönüş tipi."""
    check_date:       str
    symbols:          list[str]
    corr_matrix:      np.ndarray          # N×N korelasyon matrisi
    avg_correlation:  float               # Ortalama pairwise korelasyon
    max_correlation:  float               # En yüksek çift korelasyonu
    min_correlation:  float               # En düşük çift korelasyonu
    pairs:            list[PairCorrelation]   # Sıralı çift listesi
    risk_level:       str                 # 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    diversification_score: float          # 0-100 (100 = tam çeşitlendirilmiş)
    n_high_pairs:     int                 # Eşik üstü çift sayısı
    lookback_days:    int

    def summary_line(self) -> str:
        return (
            f"[Korelasyon] {self.check_date} | "
            f"{len(self.symbols)} sembol | "
            f"Ort={self.avg_correlation:.2f} | "
            f"Max={self.max_correlation:.2f} | "
            f"Risk={self.risk_level} | "
            f"Çeşitlendirme={self.diversification_score:.0f}/100"
        )


# ---------------------------------------------------------------------------
# Ana Sınıf
# ---------------------------------------------------------------------------

class PortfolioCorrelation:
    """
    P3-DSP Portföy Korelasyon Analizörü.

    Kullanım:
        corr   = PortfolioCorrelation()
        result = corr.run(["GARAN", "THYAO", "EREGL", "ASELS", "BIMAS"])
        print(result.summary_line())
    """

    def __init__(
        self,
        lookback_days:       int   = LOOKBACK_DAYS,
        warn_threshold:      float = CORR_WARN_THRESHOLD,
        high_threshold:      float = CORR_HIGH_THRESHOLD,
    ):
        self.lookback_days  = lookback_days
        self.warn_threshold = warn_threshold
        self.high_threshold = high_threshold

    # --- Ana Akış ---

    def run(
        self,
        symbols:    list[str],
        prices_map: Optional[dict[str, np.ndarray]] = None,
    ) -> CorrelationResult:
        """
        Sembol listesi için korelasyon analizi.

        Parameters
        ----------
        symbols    : Analiz edilecek semboller.
        prices_map : {symbol: prices_array} — None ise yfinance'ten çekilir.

        Returns
        -------
        CorrelationResult
        """
        today = date.today().isoformat()

        # Veri yükle
        if prices_map is None:
            prices_map = self._fetch_prices(symbols)

        # Ortak veri — hepsinde olan barları hizala
        returns_map = {}
        for sym in symbols:
            if sym not in prices_map:
                continue
            p = np.asarray(prices_map[sym], dtype=float)
            if len(p) < MIN_BARS:
                logger.warning(f"{sym}: yetersiz veri ({len(p)} bar), atlandı")
                continue
            p = p[-self.lookback_days:]
            log_ret = np.diff(np.log(p))
            returns_map[sym] = log_ret

        valid_symbols = list(returns_map.keys())
        if len(valid_symbols) < 2:
            raise ValueError(
                f"Korelasyon için en az 2 sembol gerekli "
                f"(geçerli: {len(valid_symbols)})"
            )

        # Getiri matrisini hizala (en kısa seriye göre)
        min_len = min(len(r) for r in returns_map.values())
        ret_matrix = np.array([
            returns_map[s][-min_len:] for s in valid_symbols
        ])  # shape: (n_symbols, n_bars)

        # Korelasyon matrisi
        corr_matrix = np.corrcoef(ret_matrix)

        # Çift korelasyonları çıkar (üst üçgen, diagonal hariç)
        pairs = []
        n = len(valid_symbols)
        pairwise_corrs = []

        for i in range(n):
            for j in range(i + 1, n):
                c = float(corr_matrix[i, j])
                pairwise_corrs.append(c)
                pairs.append(PairCorrelation(
                    symbol_a=valid_symbols[i],
                    symbol_b=valid_symbols[j],
                    correlation=round(c, 3),
                    is_high=abs(c) > self.warn_threshold,
                ))

        # Yüksekten düşüğe sırala
        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)

        avg_corr = float(np.mean(pairwise_corrs))
        max_corr = float(np.max(pairwise_corrs))
        min_corr = float(np.min(pairwise_corrs))
        n_high   = sum(1 for c in pairwise_corrs if abs(c) > self.warn_threshold)

        # Risk seviyesi
        if avg_corr > self.high_threshold:
            risk_level = "CRITICAL"
        elif avg_corr > self.warn_threshold:
            risk_level = "HIGH"
        elif avg_corr > 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Çeşitlendirme skoru: 0-100
        # avg_corr=1.0 → skor=0, avg_corr=0.0 → skor=100
        # avg_corr=-1.0 → skor=100 (negatif korelasyon da çeşitlendirme)
        div_score = max(0.0, min(100.0, (1.0 - avg_corr) * 100))

        return CorrelationResult(
            check_date=today,
            symbols=valid_symbols,
            corr_matrix=corr_matrix,
            avg_correlation=round(avg_corr, 3),
            max_correlation=round(max_corr, 3),
            min_correlation=round(min_corr, 3),
            pairs=pairs,
            risk_level=risk_level,
            diversification_score=round(div_score, 1),
            n_high_pairs=n_high,
            lookback_days=min_len,
        )

    # --- Veri Çekici ---

    def _fetch_prices(self, symbols: list[str]) -> dict[str, np.ndarray]:
        """yfinance'ten fiyat çek."""
        import yfinance as yf
        result = {}
        for sym in symbols:
            try:
                ticker = f"{sym}.IS" if not sym.endswith(".IS") else sym
                df = yf.Ticker(ticker).history(period="max")
                prices = df["Close"].dropna().values
                if len(prices) >= MIN_BARS:
                    result[sym] = prices
                else:
                    logger.warning(f"{sym}: yetersiz veri")
            except Exception as e:
                logger.warning(f"{sym} veri hatası: {e}")
        return result

    # --- Telegram Mesajı ---

    def telegram_message(self, result: CorrelationResult) -> str:
        """Korelasyon uyarısı — sadece HIGH/CRITICAL durumda gönder."""
        risk_emoji = {
            "LOW":      "🟢",
            "MEDIUM":   "🟡",
            "HIGH":     "⚠️ ",
            "CRITICAL": "🔴",
        }
        emoji = risk_emoji.get(result.risk_level, "⚠️ ")

        lines = [
            f"{emoji} *[P3-DSP] Portföy Korelasyonu* — {result.check_date}",
            f"Ort. korelasyon: *{result.avg_correlation:.2f}* | "
            f"Risk: *{result.risk_level}* | "
            f"Çeşitlendirme: *{result.diversification_score:.0f}/100*",
            "",
        ]

        if result.n_high_pairs > 0:
            lines.append("🔗 *Yüksek korelasyonlu çiftler:*")
            for p in result.pairs[:5]:
                if p.is_high:
                    lines.append(
                        f"  {p.symbol_a} ↔ {p.symbol_b}: "
                        f"{p.correlation:+.2f} {p.label()}"
                    )
        else:
            lines.append("✅ Yüksek korelasyonlu çift yok — portföy çeşitlendirilmiş.")

        return "\n".join(lines)

    # --- HTML Rapor Bölümü ---

    def html_section(self, result: CorrelationResult) -> str:
        """simulate_dsp.py HTML raporuna eklenecek korelasyon bölümü."""

        risk_colors = {
            "LOW":      "#3fb950",
            "MEDIUM":   "#d29922",
            "HIGH":     "#f85149",
            "CRITICAL": "#ff0000",
        }
        color = risk_colors.get(result.risk_level, "#8b949e")

        # KPI satırı
        kpi = (
            f"<div style='display:flex;gap:16px;margin-bottom:12px'>"
            f"<span>Ort. Korelasyon: <strong style='color:{color}'>"
            f"{result.avg_correlation:.2f}</strong></span>"
            f"<span>Risk: <strong style='color:{color}'>{result.risk_level}</strong></span>"
            f"<span>Çeşitlendirme: <strong>{result.diversification_score:.0f}/100</strong></span>"
            f"<span>Yüksek Çift: <strong>{result.n_high_pairs}</strong></span>"
            f"<span>Lookback: <strong>{result.lookback_days} bar</strong></span>"
            f"</div>"
        )

        # Çift tablosu
        rows = []
        for p in result.pairs:
            c = p.correlation
            bar_width = int(abs(c) * 80)
            bar_color = "#f85149" if abs(c) > self.warn_threshold else \
                        "#d29922" if abs(c) > 0.40 else "#3fb950"
            bar = (
                f"<div style='background:{bar_color};width:{bar_width}px;"
                f"height:8px;border-radius:4px;display:inline-block'></div>"
            )
            rows.append(
                f"<tr>"
                f"<td><strong>{p.symbol_a}</strong></td>"
                f"<td><strong>{p.symbol_b}</strong></td>"
                f"<td>{c:+.3f}</td>"
                f"<td>{bar}</td>"
                f"<td>{p.label()}</td>"
                f"</tr>"
            )

        table = (
            "<table>"
            "<tr><th>Sembol A</th><th>Sembol B</th>"
            "<th>Korelasyon</th><th>Güç</th><th>Seviye</th></tr>"
            + "".join(rows)
            + "</table>"
        )

        # Korelasyon matrisi ısı haritası (ASCII tablo)
        n = len(result.symbols)
        header = "<tr><th></th>" + "".join(
            f"<th style='font-size:0.75em'>{s}</th>"
            for s in result.symbols
        ) + "</tr>"

        matrix_rows = []
        for i, sym in enumerate(result.symbols):
            cells = [f"<td><strong style='font-size:0.8em'>{sym}</strong></td>"]
            for j in range(n):
                c = result.corr_matrix[i, j]
                if i == j:
                    bg = "#30363d"
                    txt = "—"
                else:
                    intensity = int(abs(c) * 200)
                    if c > 0:
                        bg = f"rgb({80+intensity//2},{30},{30})" \
                             if c > self.warn_threshold else \
                             f"rgb({40+intensity//3},{40+intensity//3},{20})"
                    else:
                        bg = f"rgb(20,{60+intensity//2},{40})"
                    txt = f"{c:+.2f}"
                cells.append(
                    f"<td style='background:{bg};text-align:center;"
                    f"font-size:0.8em;padding:4px 6px'>{txt}</td>"
                )
            matrix_rows.append(f"<tr>{''.join(cells)}</tr>")

        matrix_table = (
            "<div style='overflow-x:auto;margin-top:12px'>"
            f"<table style='border-collapse:collapse'>"
            f"{header}{''.join(matrix_rows)}</table></div>"
        )

        return (
            f"{kpi}"
            f"<h3 style='color:#79c0ff;font-size:0.95em;margin:12px 0 6px'>"
            f"Pairwise Korelasyonlar</h3>"
            f"{table}"
            f"<h3 style='color:#79c0ff;font-size:0.95em;margin:16px 0 6px'>"
            f"Korelasyon Matrisi</h3>"
            f"{matrix_table}"
        )


# ---------------------------------------------------------------------------
# simulate_dsp.py Entegrasyon Yardımcısı
# ---------------------------------------------------------------------------

def run_portfolio_correlation(
    state: dict,
    prices_map: Optional[dict[str, np.ndarray]] = None,
) -> Optional[CorrelationResult]:
    """
    simulate_dsp.py'den çağrılır.
    Açık pozisyonlar için korelasyon analizi yapar.

    Parameters
    ----------
    state      : portfolio_state dict
    prices_map : Varsa scanner'dan gelen fiyat verisi (opsiyonel)
    """
    positions = list(state.get("positions", {}).keys())
    if len(positions) < 2:
        logger.info("Korelasyon analizi için yeterli pozisyon yok (min 2).")
        return None

    try:
        corr = PortfolioCorrelation()
        return corr.run(positions, prices_map)
    except Exception as e:
        logger.warning(f"Korelasyon analizi hatası: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="P3-DSP Korelasyon Analizi")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["YKBNK", "AKBNK", "SKBNK", "ODAS", "GLYHO"],
        help="Analiz edilecek semboller (varsayılan: mevcut açık pozisyonlar)"
    )
    parser.add_argument(
        "--lookback", type=int, default=120,
        help="Geriye bakış süresi (bar, varsayılan: 120)"
    )
    parser.add_argument(
        "--universe", action="store_true",
        help="Tüm tarama evreni için korelasyon analizi"
    )
    args = parser.parse_args()

    if args.universe:
        # Tüm evren
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scanner_dsp import load_symbols
        symbols = load_symbols()
        print(f"Evren analizi: {len(symbols)} sembol")
    else:
        symbols = [s.upper() for s in args.symbols]

    print(f"\n{'='*60}")
    print(f"  P3-DSP Portföy Korelasyon Analizi")
    print(f"  Semboller : {', '.join(symbols)}")
    print(f"  Lookback  : {args.lookback} bar")
    print(f"{'='*60}\n")

    print("Veri çekiliyor (yfinance)...")
    corr    = PortfolioCorrelation(lookback_days=args.lookback)
    result  = corr.run(symbols)

    print(f"\n{result.summary_line()}\n")

    # Pairwise tablo
    print(f"  {'Sembol A':<10} {'Sembol B':<10} {'Korelasyon':>12} {'Seviye'}")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*12}")
    for p in result.pairs:
        print(f"  {p.symbol_a:<10} {p.symbol_b:<10} "
              f"{p.correlation:>+12.3f}  {p.label()}")

    # Korelasyon matrisi
    n = len(result.symbols)
    print(f"\n  Korelasyon Matrisi ({result.lookback_days} bar):")
    header = f"  {'':12}" + "".join(f"{s:>8}" for s in result.symbols)
    print(header)
    for i, sym in enumerate(result.symbols):
        row = f"  {sym:<12}" + "".join(
            f"{'─':>8}" if i == j else f"{result.corr_matrix[i,j]:>+8.2f}"
            for j in range(n)
        )
        print(row)

    # Özet
    print(f"\n{'─'*60}")
    print(f"  Ortalama korelasyon : {result.avg_correlation:+.3f}")
    print(f"  Maksimum korelasyon : {result.max_correlation:+.3f}")
    print(f"  Minimum korelasyon  : {result.min_correlation:+.3f}")
    print(f"  Yüksek çift sayısı  : {result.n_high_pairs}")
    print(f"  Çeşitlendirme skoru : {result.diversification_score:.0f}/100")

    risk_icons = {
        "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "⚠️ ", "CRITICAL": "🔴"
    }
    print(f"  Risk seviyesi       : "
          f"{risk_icons.get(result.risk_level,'')} {result.risk_level}")

    if result.risk_level in ("HIGH", "CRITICAL"):
        print(f"\n  ⚠️  Portföy konsantrasyon riski yüksek!")
        print(f"  Öneri: Düşük korelasyonlu sektörlerden sembol ekle.")
    else:
        print(f"\n  ✅ Portföy çeşitlendirmesi yeterli.")

    print(f"\n{'─'*60}")
    print("\n📨 Telegram Mesajı Önizleme:")
    print("─" * 40)
    print(corr.telegram_message(result))