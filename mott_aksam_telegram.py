"""
mott_aksam_telegram.py — Akşam modunda P1→P5 sıralı Telegram raporları
======================================================================
Akşam taramasında işlem mesajları kuyruğa yazılır; bu modül tüm job'lar
bittikten sonra P1, P2, P3, P4, P5 sırasıyla gönderir (işlem yoksa portföy özeti).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)
BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", "."))
PENDING_DIR = BASE_DIR / "reports" / "telegram_pending"
SIRALAMA = ("P1", "P2", "P3", "P4", "P5")
BEKLEME_SN = 2


def pending_yolu(strateji: str) -> Path:
    return PENDING_DIR / f"{strateji}.json"


def pending_kaydet(strateji: str, mesaj: str, meta: dict | None = None) -> None:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"strateji": strateji, "mesaj": mesaj, "meta": meta or {}}
    pending_yolu(strateji).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("[%s] Akşam Telegram kuyruğa yazıldı", strateji)


def _portfoy_yukle(strateji: str) -> dict:
    dosyalar = {
        "P1": "portfoy.json",
        "P2": "portfoy_p2.json",
        "P3": "portfolio_state.json",
        "P4": "state_p4.json",
        "P5": "state_p5.json",
    }
    path = BASE_DIR / dosyalar.get(strateji, "")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("%s portföy okunamadı: %s", strateji, exc)
    return {}


def _sinyaller_yukle(strateji: str) -> list[dict]:
    try:
        if strateji == "P1":
            from mott_telegram import yukle_p1_sinyaller
            return yukle_p1_sinyaller()
        if strateji == "P2":
            from mott_telegram import yukle_p2_sinyaller
            return yukle_p2_sinyaller()
        if strateji == "P3":
            path = BASE_DIR / "portfolio_state.json"
            if path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                log_items = d.get("scan_log", [])
                if log_items:
                    top = log_items[-1].get("top5", log_items[-1].get("top_longs", []))
                    return [{"symbol": s, "score": 0.0} for s in top[:8]]
        if strateji == "P4":
            from meta_portfolio import p1_sinyalleri_yukle, p2_sinyalleri_yukle, p3_sinyalleri_yukle, ic_agirlikli_birlestir
            p1, _ = p1_sinyalleri_yukle()
            p2, _ = p2_sinyalleri_yukle()
            p3, _ = p3_sinyalleri_yukle()
            adaylar = ic_agirlikli_birlestir(p1, p2, p3, 0.0, 0.0, 0.0)
            return [
                {
                    "symbol": a.get("symbol"),
                    "score": a.get("meta_score", 0),
                    "kaynaklar": a.get("kaynaklar", []),
                }
                for a in adaylar[:8]
            ]
        if strateji == "P5":
            from p5_committee import komite_adaylari
            adaylar, _ = komite_adaylari()
            return adaylar[:8]
    except Exception as exc:
        log.warning("%s sinyal yüklenemedi: %s", strateji, exc)
    return []


def portfoy_ozet_mesaji(strateji: str) -> str:
    """İşlem olmasa da gönderilecek akşam portföy özeti."""
    from mott_telegram import p1_mesaj, p2_mesaj, p3_mesaj, p4_mesaj, p5_mesaj

    portfoy = _portfoy_yukle(strateji)
    sinyaller = _sinyaller_yukle(strateji)

    if strateji == "P1":
        return p1_mesaj(sinyaller, portfoy, giris=[], cikis=[])
    if strateji == "P2":
        return p2_mesaj(sinyaller, portfoy, giris=[], cikis=[])
    if strateji == "P3":
        return p3_mesaj(sinyaller, portfoy, giris=[], cikis=[])
    if strateji == "P4":
        secilen = [
            {
                "symbol": s.get("symbol", "?"),
                "meta_score": s.get("score", 0),
                "kaynaklar": s.get("kaynaklar", s.get("strategies", [])),
            }
            for s in sinyaller
        ]
        return p4_mesaj(secilen, portfoy, ic_scores={}, giris=[], cikis=[])
    if strateji == "P5":
        secilen = [
            {
                "symbol": s.get("symbol", "?"),
                "komite_skor": s.get("score", s.get("komite_skor", 0)),
                "kaynaklar": s.get("kaynaklar", s.get("strategies", [])),
            }
            for s in sinyaller
        ]
        return p5_mesaj(secilen, portfoy, giris=[], cikis=[])
    return ""


def aksam_raporlari_gonder() -> int:
    """P1→P5 sırayla Telegram gönder. Dönen: gönderilen mesaj sayısı."""
    from mott_telegram import telegram_gonder

    gonderilen = 0
    for strateji in SIRALAMA:
        path = pending_yolu(strateji)
        mesaj = ""
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                mesaj = data.get("mesaj", "")
            except Exception as exc:
                log.warning("[%s] kuyruk okunamadı: %s", strateji, exc)

        if not mesaj:
            mesaj = portfoy_ozet_mesaji(strateji)
            if mesaj:
                log.info("[%s] İşlem yok — portföy özeti gönderiliyor", strateji)

        if not mesaj:
            log.info("[%s] Atlandı (mesaj üretilemedi)", strateji)
            continue

        if telegram_gonder(mesaj):
            gonderilen += 1
            log.info("[%s] Telegram gönderildi", strateji)
        else:
            log.warning("[%s] Telegram gönderilemedi", strateji)

        if strateji != SIRALAMA[-1]:
            time.sleep(BEKLEME_SN)

    return gonderilen


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="MOTT akşam sıralı Telegram")
    parser.add_argument("--gonder", action="store_true", help="P1→P5 sırayla gönder")
    args = parser.parse_args()
    if args.gonder:
        n = aksam_raporlari_gonder()
        print(f"Akşam Telegram: {n} mesaj gönderildi")
