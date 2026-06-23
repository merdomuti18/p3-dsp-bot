"""
portfolio_dsp.py — P3-DSP Paper Portföy State Yönetimi
========================================================
P2-SMC'nin portfolio_smc.py'sine paralel yapı.

Görev:
  - Açık pozisyonları tut (JSON state dosyası)
  - Yeni tarama sonuçlarına göre giriş/çıkış kararı ver
  - Her işlemi logla
  - Telegram bildirimi için mesaj üret

State dosyası: ~/bist_ai/p3_dsp_portfolio.json
Log dosyası  : ~/bist_ai/logs/p3_dsp.log
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

DEFAULT_STATE_FILE = os.path.expanduser("~/bist_ai/p3_dsp_portfolio.json")
DEFAULT_LOG_FILE   = os.path.expanduser("~/bist_ai/logs/p3_dsp.log")

MAX_POSITIONS   = 5      # Aynı anda maksimum açık pozisyon
POSITION_SIZE_PCT = 0.20 # Her pozisyon portföyün %20'si (eşit ağırlık)


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    entry_date: str         # ISO format
    entry_price: float
    entry_score: float
    size_pct: float         # Portföy ağırlığı
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100


@dataclass
class Trade:
    """Kapanmış işlem kaydı."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int
    exit_reason: str    # "signal_exit" | "manual" | "rebalance"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**d)


# ---------------------------------------------------------------------------
# Portfolio State
# ---------------------------------------------------------------------------

class DspPortfolio:
    """
    P3-DSP Paper Portföy.

    State JSON'da saklanır — her çalıştırmada yüklenir, güncellenir,
    tekrar kaydedilir. P2-SMC'nin portfolio_smc.py ile aynı pattern.

    Kullanım (scanner_dsp.py'den):
        portfolio = DspPortfolio()
        actions = portfolio.update(scan_result, current_prices)
        msg = portfolio.telegram_message(actions)
    """

    def __init__(
        self,
        state_file: str = DEFAULT_STATE_FILE,
        max_positions: int = MAX_POSITIONS,
        position_size_pct: float = POSITION_SIZE_PCT,
    ):
        self.state_file = state_file
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct

        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self.created_at: str = datetime.utcnow().isoformat()
        self.last_scan: Optional[str] = None

        self._load()

    # --- State Yönetimi ---

    def _load(self) -> None:
        """State dosyasından yükle."""
        if not os.path.exists(self.state_file):
            logger.info(f"State dosyası bulunamadı, yeni oluşturulacak: {self.state_file}")
            return

        try:
            with open(self.state_file, "r") as f:
                raw = f.read().strip()
            if not raw:
                return
            data = json.loads(raw)
            self.positions = {
                k: Position.from_dict(v)
                for k, v in data.get("positions", {}).items()
            }
            self.trade_history = [
                Trade.from_dict(t) for t in data.get("trade_history", [])
            ]
            self.created_at = data.get("created_at", self.created_at)
            self.last_scan  = data.get("last_scan")
            logger.info(f"State yüklendi: {len(self.positions)} açık pozisyon")
        except Exception as e:
            logger.error(f"State yükleme hatası: {e}")

    def save(self) -> None:
        """State dosyasına kaydet."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        data = {
            "created_at":    self.created_at,
            "last_scan":     self.last_scan,
            "positions":     {k: v.to_dict() for k, v in self.positions.items()},
            "trade_history": [t.to_dict() for t in self.trade_history],
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"State kaydedildi: {self.state_file}")

    # --- Portföy Güncelleme ---

    @dataclass
    class PortfolioActions:
        """update() dönüş tipi — ne yapıldığını özetler."""
        entries:  list[str] = None   # Yeni girilen semboller
        exits:    list[str] = None   # Çıkılan semboller
        holds:    list[str] = None   # Devam eden pozisyonlar
        skipped:  list[str] = None   # Slot dolu, sinyal var ama girilmedi
        date:     str = ""

        def __post_init__(self):
            self.entries  = self.entries  or []
            self.exits    = self.exits    or []
            self.holds    = self.holds    or []
            self.skipped  = self.skipped  or []
            self.date     = self.date or date.today().isoformat()

        @property
        def has_changes(self) -> bool:
            return bool(self.entries or self.exits)

    def update(
        self,
        scan_result,                          # ScanResult (scanner_dsp.py)
        current_prices: dict[str, float],     # {symbol: son_fiyat}
    ) -> "DspPortfolio.PortfolioActions":
        """
        Tarama sonucuna göre pozisyonları güncelle.

        Mantık:
          1. Çıkış: mevcut pozisyonlarda sinyal +1 değilse kapat
          2. Giriş: long listesinde olup pozisyonda olmayan semboller,
                    boş slot varsa al (skor sırasıyla)
          3. Hold:  hem pozisyonda hem sinyal +1 → değişiklik yok
        """
        today = date.today().isoformat()
        self.last_scan = today
        actions = self.PortfolioActions(date=today)

        # Long sinyal seti
        long_set = {s.symbol for s in scan_result.top_longs}

        # 1. Çıkış kararları
        for symbol in list(self.positions.keys()):
            if symbol not in long_set:
                price = current_prices.get(symbol)
                self._close_position(symbol, price, "signal_exit", today)
                actions.exits.append(symbol)
            else:
                actions.holds.append(symbol)

        # 2. Giriş kararları
        for sc in scan_result.top_longs:
            symbol = sc.symbol
            if symbol in self.positions:
                continue  # Zaten açık

            if len(self.positions) >= self.max_positions:
                actions.skipped.append(symbol)
                continue

            price = current_prices.get(symbol)
            if price is None:
                logger.warning(f"{symbol}: fiyat yok, giriş atlandı")
                continue

            pos = Position(
                symbol=symbol,
                entry_date=today,
                entry_price=price,
                entry_score=sc.score,
                size_pct=self.position_size_pct,
                notes=f"score={sc.score:.4f} marj={sc.crossover_margin:.3f}",
            )
            self.positions[symbol] = pos
            actions.entries.append(symbol)
            logger.info(f"GIRIŞ: {symbol} @ {price:.3f}")

        self.save()
        return actions

    def _close_position(
        self,
        symbol: str,
        exit_price: Optional[float],
        reason: str,
        today: str,
    ) -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        entry_d = date.fromisoformat(pos.entry_date)
        exit_d  = date.fromisoformat(today)
        holding = (exit_d - entry_d).days

        pnl = 0.0
        if exit_price:
            pnl = pos.pnl_pct(exit_price)

        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=today,
            entry_price=pos.entry_price,
            exit_price=exit_price or 0.0,
            pnl_pct=round(pnl, 2),
            holding_days=holding,
            exit_reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(f"ÇIKIŞ: {symbol} pnl={pnl:+.2f}% ({holding}g) [{reason}]")

    # --- Telegram Mesaj Üretici ---

    def telegram_message(self, actions: "DspPortfolio.PortfolioActions") -> str:
        """
        Telegram'a gönderilecek mesajı üret.

        [P3-DSP] etiketi ile — mevcut bot altyapısına eklenir.
        Değişiklik yoksa kısa özet; değişiklik varsa detaylı mesaj.
        """
        lines = [f"📡 *[P3-DSP]* {actions.date}"]

        if actions.entries:
            entries_str = ", ".join(s.replace(".IS", "") for s in actions.entries)
            lines.append(f"🟢 *Giriş:* {entries_str}")

        if actions.exits:
            exit_parts = []
            for sym in actions.exits:
                trade = next(
                    (t for t in reversed(self.trade_history) if t.symbol == sym), None
                )
                if trade:
                    pnl_emoji = "✅" if trade.pnl_pct >= 0 else "🔴"
                    exit_parts.append(
                        f"{sym.replace('.IS','')} {pnl_emoji}{trade.pnl_pct:+.1f}%"
                    )
                else:
                    exit_parts.append(sym.replace(".IS", ""))
            lines.append(f"⬜ *Çıkış:* {', '.join(exit_parts)}")

        if actions.holds:
            holds_str = ", ".join(s.replace(".IS", "") for s in actions.holds)
            lines.append(f"⏸️ *Devam:* {holds_str}")

        if actions.skipped:
            skip_str = ", ".join(s.replace(".IS", "") for s in actions.skipped)
            lines.append(f"⚠️ *Slot dolu (atlandı):* {skip_str}")

        # Özet satırı
        n_open = len(self.positions)
        lines.append(f"\n📊 Açık pozisyon: {n_open}/{self.max_positions}")

        if not actions.has_changes:
            lines.append("_Değişiklik yok_")

        return "\n".join(lines)

    # --- Performans Özeti ---

    def performance_summary(
        self,
        current_prices: Optional[dict[str, float]] = None,
    ) -> dict:
        """
        Dashboard ve Telegram için özet istatistikler.
        """
        closed = self.trade_history
        n_closed = len(closed)

        if n_closed > 0:
            pnls = [t.pnl_pct for t in closed]
            win_rate = sum(1 for p in pnls if p > 0) / n_closed
            avg_pnl  = float(np.mean(pnls))
            total_pnl = float(np.sum(pnls))
            avg_hold  = float(np.mean([t.holding_days for t in closed]))
        else:
            win_rate = avg_pnl = total_pnl = avg_hold = 0.0

        open_positions = []
        for sym, pos in self.positions.items():
            price = (current_prices or {}).get(sym)
            open_positions.append({
                "symbol":      sym,
                "entry_date":  pos.entry_date,
                "entry_price": pos.entry_price,
                "current_price": price,
                "pnl_pct":     pos.pnl_pct(price) if price else None,
                "score":       pos.entry_score,
            })

        return {
            "last_scan":       self.last_scan,
            "n_open":          len(self.positions),
            "max_positions":   self.max_positions,
            "n_closed_trades": n_closed,
            "win_rate":        round(win_rate, 3),
            "avg_pnl_pct":     round(avg_pnl, 2),
            "total_pnl_pct":   round(total_pnl, 2),
            "avg_hold_days":   round(avg_hold, 1),
            "open_positions":  open_positions,
        }
