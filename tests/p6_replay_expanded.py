# -*- coding: utf-8 -*-
"""
P6 Etap 1.9 — expanded historical as-of replay (ANALYSIS ONLY).

Reuses tests/p6_replay.py (Etap 1.7) evaluate_symbol + truncate_to_asof.
Does not modify scanner_p1.veri_hazirla / scanner_p6 / spec.

Does not compute forward returns. Horizon counts are bar-availability only.
Does not overwrite analysis/p6_signal_ledger.jsonl.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scanner_p6
from p6_replay import (
    LEDGER_PURPOSE,
    ReplayDataError,
    frames_look_ahead_equal,
    ledgers_equal,
    load_ohlcv_pickle,
    normalize_ohlcv,
    production_universe,
    replay_asof,
    write_ledger,
    write_meta,
)

EXPANDED_PURPOSE = "HISTORICAL REPLAY / ANALYSIS ONLY — ETAP 1.9 COVERAGE EXPANSION"
HORIZONS = (1, 3, 5, 10, 20)
ORIGINAL_LEDGER = _REPO_ROOT / "analysis" / "p6_signal_ledger.jsonl"
EXPANDED_LEDGER_NAME = "p6_signal_ledger_expanded.jsonl"
EXPANDED_META_NAME = "p6_replay_expanded_meta.json"
ANALYSIS_DOWNLOAD_PERIOD = "8mo"
ANALYSIS_DOWNLOAD_INTERVAL = "1d"


def last_closed_asof_cutoff(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Last fully closed BIST cash session (18:00 + 10m). Naive midnight timestamp."""
    if now is None:
        now = pd.Timestamp.now(tz="Europe/Istanbul")
    elif now.tzinfo is None:
        now = now.tz_localize("Europe/Istanbul")
    else:
        now = now.tz_convert("Europe/Istanbul")
    closed_today = (now.hour > 18) or (now.hour == 18 and now.minute >= 10)
    day = pd.Timestamp(now.date())
    if not closed_today:
        day = day - pd.Timedelta(days=1)
    return day


def apply_closed_cutoff(
    frames: Mapping[str, pd.DataFrame],
    cutoff: Optional[pd.Timestamp] = None,
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    if cutoff is None:
        cutoff = last_closed_asof_cutoff()
    cutoff = pd.Timestamp(cutoff).normalize()
    out: dict[str, pd.DataFrame] = {}
    for symbol, df in frames.items():
        clipped = df.loc[df.index <= cutoff]
        if clipped.empty:
            continue
        out[symbol] = clipped
    return out, cutoff


def download_daily_frames(
    symbols: list[str],
    *,
    period: str = ANALYSIS_DOWNLOAD_PERIOD,
    interval: str = ANALYSIS_DOWNLOAD_INTERVAL,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Analysis-only daily OHLCV. Does not call scanner_p1.veri_hazirla.
    No volume ffill, no missing-bar fill, no price impute.
    Duplicate timestamps raise via normalize_ohlcv.
    """
    import yfinance as yf

    tickers = [f"{s}.IS" for s in symbols]
    kwargs = dict(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    raw = yf.download(**kwargs)
    frames: dict[str, pd.DataFrame] = {}
    dropped_nan_ohlc: dict[str, int] = {}
    empty: list[str] = []
    errors: list[str] = []
    if raw is None or getattr(raw, "empty", True):
        raise ReplayDataError("analysis download returned empty frame")
    for sym in symbols:
        ticker = f"{sym}.IS"
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = raw.columns.get_level_values(0)
                if ticker not in set(level0):
                    empty.append(sym)
                    continue
                df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = [str(c).strip().lower() for c in df.columns]
            before = len(df)
            df = df.dropna(subset=["open", "high", "low", "close"])
            dropped = before - len(df)
            if dropped:
                dropped_nan_ohlc[sym] = int(dropped)
            if df.empty:
                empty.append(sym)
                continue
            frames[sym] = normalize_ohlcv(df, symbol=sym)
        except ReplayDataError:
            raise
        except Exception as exc:
            errors.append(f"{sym}: {exc}")
            continue
    report = {
        "period": period,
        "interval": interval,
        "auto_adjust": True,
        "volume_ffill": False,
        "price_impute": False,
        "bar_forward_fill": False,
        "production_veri_hazirla_called": False,
        "requested_symbols": len(symbols),
        "frames_n": len(frames),
        "empty_or_missing": empty,
        "dropped_nan_ohlc": dropped_nan_ohlc,
        "errors": errors,
    }
    return frames, report


def save_frames_pickle(path: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(dict(frames), fh)


def future_horizon_available(df: pd.DataFrame, signal_date: str, period: int) -> str:
    """Bar-availability only. Does not compute log/percent return."""
    ts = pd.Timestamp(signal_date).normalize()
    if ts not in df.index:
        return "missing_signal_date"
    pos = int(df.index.get_indexer([ts])[0])
    if pos < 0:
        return "missing_signal_date"
    if pos + period >= len(df):
        return "insufficient_future"
    c1 = float(df["close"].iloc[pos + 1])
    cp = float(df["close"].iloc[pos + period])
    if pd.isna(c1) or pd.isna(cp) or c1 <= 0 or cp <= 0:
        return "invalid_future_close"
    return "available"


def horizon_coverage(
    rows: list[dict],
    frames: Mapping[str, pd.DataFrame],
    *,
    agent: Optional[str] = None,
) -> dict[str, Any]:
    sub = [r for r in rows if agent is None or r["agent"] == agent]
    out: dict[str, Any] = {"raw_signals": len(sub), "horizons": {}}
    for period in HORIZONS:
        st: Counter = Counter()
        for r in sub:
            if r["symbol"] not in frames:
                st["missing_symbol"] += 1
                continue
            st[future_horizon_available(frames[r["symbol"]], r["date"], period)] += 1
        out["horizons"][str(period)] = {
            "usable": int(st.get("available", 0)),
            "status": dict(st),
        }
    return out


def look_ahead_scan(frames: Mapping[str, pd.DataFrame]) -> list[dict]:
    """All symbols with MIN_BARS+1 bars; T = second-to-last index."""
    fails: list[dict] = []
    keys = (
        "agents_equal",
        "sqz_signal_equal",
        "recent_squeeze_equal",
        "bb_up_equal",
        "squeeze_on_prefix_equal",
        "zkn_equal",
        "wyc_equal",
    )
    n = 0
    for symbol, df in frames.items():
        if len(df) < scanner_p6.MIN_BARS + 1:
            continue
        n += 1
        asof = df.index[-2]
        chk = frames_look_ahead_equal(symbol, df, asof)
        if not all(chk[k] for k in keys):
            fails.append({"symbol": symbol, "asof": str(pd.Timestamp(asof).date()), **chk})
    return fails


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P6 Etap 1.9 expanded as-of replay (ANALYSIS ONLY).")
    parser.add_argument("--ohlcv", default="", help="Existing pickle. If empty, analysis download.")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "analysis"))
    parser.add_argument("--cache-pickle", default="/tmp/p6_etap19/ohlcv.pkl")
    parser.add_argument("--period", default=ANALYSIS_DOWNLOAD_PERIOD)
    args = parser.parse_args(argv)
    original = ORIGINAL_LEDGER
    original_hash = original.read_bytes() if original.exists() else None
    universe = production_universe()
    download_report: Optional[dict] = None
    src_kind = "existing_ohlcv_pickle"
    src_path = args.ohlcv
    if args.ohlcv:
        frames = load_ohlcv_pickle(Path(args.ohlcv))
    else:
        cache = Path(args.cache_pickle)
        if cache.exists():
            frames = load_ohlcv_pickle(cache)
            src_kind = "analysis_cache_pickle"
            src_path = str(cache)
        else:
            frames, download_report = download_daily_frames(universe, period=args.period)
            save_frames_pickle(cache, frames)
            src_kind = "analysis_yfinance_download"
            src_path = str(cache)
    frames, closed_cutoff = apply_closed_cutoff(frames)
    rows_a, stats = replay_asof(frames, universe=universe)
    rows_b, stats_b = replay_asof(frames, universe=universe)
    if not ledgers_equal(rows_a, rows_b):
        print("FAIL — NON-DETERMINISTIC REPLAY", file=sys.stderr)
        return 1
    if stats["signal_count"] != stats_b["signal_count"]:
        print("FAIL — NON-DETERMINISTIC REPLAY (meta counts)", file=sys.stderr)
        return 1
    la_fail = look_ahead_scan(frames)
    if la_fail:
        print("FAIL — LOOK-AHEAD VIOLATION", file=sys.stderr)
        print(json.dumps(la_fail[:5], ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    coverage = {
        "ALL": horizon_coverage(rows_a, frames),
        "SQZ": horizon_coverage(rows_a, frames, agent="SQZ"),
        "ZKN": horizon_coverage(rows_a, frames, agent="ZKN"),
        "WYC": horizon_coverage(rows_a, frames, agent="WYC"),
    }
    out_dir = Path(args.out_dir)
    write_ledger(out_dir / EXPANDED_LEDGER_NAME, rows_a)
    extra = {
        "purpose": EXPANDED_PURPOSE,
        "source": src_path,
        "source_kind": src_kind,
        "download_report": download_report,
        "determinism_pass": True,
        "look_ahead_fail_n": len(la_fail),
        "look_ahead_checked_symbols": sum(
            1 for df in frames.values() if len(df) >= scanner_p6.MIN_BARS + 1
        ),
        "production_scan_history": False,
        "ranking": False,
        "score": False,
        "top_k": False,
        "forward_return_calculated": False,
        "execution": False,
        "original_ledger_preserved": True,
        "original_ledger_path": str(original),
        "expanded_ledger_path": str(out_dir / EXPANDED_LEDGER_NAME),
        "universe_source": "scanner_p6.aktif_hisse_listesi -> scanner_p1",
        "hisseler_txt_present": (_REPO_ROOT / "hisseler.txt").exists(),
        "horizon_coverage_bar_availability_only": coverage,
        "closed_asof_cutoff": str(closed_cutoff.date()),
        "min_bars": scanner_p6.MIN_BARS,
        "rule_versions": {
            "SQZ": scanner_p6.RULE_VERSION_SQZ,
            "ZKN": scanner_p6.RULE_VERSION_ZKN,
            "WYC": scanner_p6.RULE_VERSION_WYC,
        },
    }
    write_meta(out_dir / EXPANDED_META_NAME, stats, extra)
    meta_path = out_dir / EXPANDED_META_NAME
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["purpose"] = EXPANDED_PURPOSE
    payload["source_kind"] = src_kind
    payload["source_provenance"] = {
        "period": ANALYSIS_DOWNLOAD_PERIOD,
        "interval": ANALYSIS_DOWNLOAD_INTERVAL,
        "auto_adjust": True,
        "volume_ffill": False,
        "price_impute": False,
        "bar_forward_fill": False,
        "production_veri_hazirla_called": False,
        "cache_pickle": args.cache_pickle,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if original.exists() and original_hash is not None and original.read_bytes() != original_hash:
        print("FAIL — original Etap 1.7 ledger was modified", file=sys.stderr)
        return 1
    print(json.dumps({
        "purpose": EXPANDED_PURPOSE,
        "frames": len(frames),
        "signals": stats["signal_count"],
        "sqz": stats["sqz_true"],
        "zkn": stats["zkn_true"],
        "wyc": stats["wyc_true"],
        "valid_dates": stats["valid_replay_dates"],
        "sqz_5d_usable": coverage["SQZ"]["horizons"]["5"]["usable"],
        "determinism_pass": True,
        "look_ahead_fail_n": 0,
        "forward_return_calculated": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
