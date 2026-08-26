# -*- coding: utf-8 -*-
"""
P6 Etap 1.7 — historical as-of replay / signal ledger (ANALYSIS ONLY).

HISTORICAL REPLAY / ANALYSIS ONLY

Does not change P6 Boolean contracts. Replays existing
scanner_p6.evaluate_symbol + truncate_to_asof.

No ranking, score, Top-K, forward return, execution, or exit.
Does not write production scan_history / state / workflows.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scanner_p6

LEDGER_PURPOSE = "HISTORICAL REPLAY / ANALYSIS ONLY"
REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")
AGENTS = (scanner_p6.STRATEGY_ZKN, scanner_p6.STRATEGY_WYC, scanner_p6.STRATEGY_SQZ)
RULE_VERSIONS = {
    scanner_p6.STRATEGY_ZKN: scanner_p6.RULE_VERSION_ZKN,
    scanner_p6.STRATEGY_WYC: scanner_p6.RULE_VERSION_WYC,
    scanner_p6.STRATEGY_SQZ: scanner_p6.RULE_VERSION_SQZ,
}


class ReplayDuplicateError(RuntimeError):
    """Same (date, symbol, agent) appeared twice — do not drop silently."""


class ReplayDataError(ValueError):
    """Source OHLCV cannot be used without inventing or silently cleaning bars."""


def _json_num(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _json_bool(val: Any) -> Optional[bool]:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None
    return bool(val)


def normalize_ohlcv(df: pd.DataFrame, *, symbol: str = "") -> pd.DataFrame:
    """Lowercase OHLC columns; naive midnight index. Duplicate timestamps raise."""
    if df is None or df.empty:
        raise ReplayDataError(f"{symbol or '?'}: empty OHLCV")
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in REQUIRED_OHLCV if c not in out.columns]
    if missing:
        raise ReplayDataError(f"{symbol or '?'}: missing columns {missing}")
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    dup = int(out.index.duplicated().sum())
    if dup:
        raise ReplayDataError(f"{symbol or '?'}: {dup} duplicate timestamps — not dropped")
    out = out.sort_index()
    return out[list(REQUIRED_OHLCV)]


def load_ohlcv_pickle(path: Path) -> dict[str, pd.DataFrame]:
    with Path(path).open("rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, dict) or not raw:
        raise ReplayDataError(f"{path}: expected non-empty dict[symbol, DataFrame]")
    return {str(sym): normalize_ohlcv(frame, symbol=str(sym)) for sym, frame in raw.items()}


def production_universe() -> list[str]:
    """P6 universe = scanner_p1.aktif_hisse_listesi via scanner_p6."""
    return list(scanner_p6.aktif_hisse_listesi())


def trading_dates(frames: Mapping[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for df in frames.values():
        dates.update(pd.Timestamp(ts).normalize() for ts in df.index)
    return sorted(dates)


def valid_replay_dates(
    frames: Mapping[str, pd.DataFrame],
    *,
    min_bars: int = scanner_p6.MIN_BARS,
) -> list[pd.Timestamp]:
    """Dates T where at least one symbol has a bar on T and len(prefix)<=T >= min_bars."""
    out: list[pd.Timestamp] = []
    for ts in trading_dates(frames):
        for df in frames.values():
            if ts not in df.index:
                continue
            prefix = scanner_p6.truncate_to_asof(df, ts)
            if len(prefix) >= min_bars:
                out.append(ts)
                break
    return out


def ledger_row_from_record(rec: dict) -> dict:
    ind = rec.get("indicators") or {}
    row = {
        "date": rec["asof_date"],
        "symbol": rec["symbol"],
        "agent": rec["strategy"],
        "signal": True,
        "rule_version": rec["rule_version"],
        "close": _json_num(ind.get("close")),
        "bb_up": _json_num(ind.get("bb_up")),
        "recent_squeeze": _json_bool(ind.get("recent_squeeze")),
        "support": _json_num(ind.get("support")),
        "analysis_only": True,
        "purpose": LEDGER_PURPOSE,
    }
    forbidden = scanner_p6.FORBIDDEN_RECORD_KEYS.intersection(row)
    if forbidden:
        raise ReplayDataError(f"forbidden ledger fields: {sorted(forbidden)}")
    expected = RULE_VERSIONS.get(row["agent"])
    if expected and row["rule_version"] != expected:
        raise ReplayDataError(
            f"rule_version mismatch {row['agent']}: {row['rule_version']} != {expected}"
        )
    return row


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["date"]), str(row["symbol"]), str(row["agent"]))


def assert_no_duplicates(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            raise ReplayDuplicateError(f"duplicate primary key {key}")
        seen.add(key)


def sort_ledger(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    return sorted((dict(r) for r in rows), key=_row_key)


def frames_look_ahead_equal(
    symbol: str,
    df: pd.DataFrame,
    asof: Any,
) -> dict[str, Any]:
    """prefix(T) vs truncate_to_asof(full, T) — signals and SQZ prefix state."""
    asof_ts = pd.Timestamp(asof).normalize()
    prefix = scanner_p6.truncate_to_asof(df, asof_ts)
    rec_prefix = scanner_p6.evaluate_symbol(symbol, prefix)
    rec_full = scanner_p6.evaluate_symbol(symbol, df, asof=asof_ts)
    ind_a = scanner_p6.get_indicators_p6(prefix)
    truncated = scanner_p6.truncate_to_asof(df, asof_ts)
    ind_b = scanner_p6.get_indicators_p6(truncated)
    sq_a = scanner_p6.squeeze_on_series(prefix)
    sq_b = scanner_p6.squeeze_on_series(truncated)
    agents_a = sorted(r["strategy"] for r in rec_prefix)
    agents_b = sorted(r["strategy"] for r in rec_full)
    squeeze_equal = sq_a.equals(sq_b)
    return {
        "agents_equal": agents_a == agents_b,
        "agents_prefix": agents_a,
        "agents_full": agents_b,
        "sqz_signal_equal": bool(scanner_p6.strategy_sqz(ind_a))
        is bool(scanner_p6.strategy_sqz(ind_b)),
        "recent_squeeze_equal": bool(ind_a.get("recent_squeeze"))
        is bool(ind_b.get("recent_squeeze")),
        "bb_up_equal": _json_num(ind_a.get("bb_up")) == _json_num(ind_b.get("bb_up")),
        "squeeze_on_prefix_equal": bool(squeeze_equal),
        "zkn_equal": bool(scanner_p6.strategy_zkn(ind_a))
        is bool(scanner_p6.strategy_zkn(ind_b)),
        "wyc_equal": bool(scanner_p6.strategy_wyc(ind_a))
        is bool(scanner_p6.strategy_wyc(ind_b)),
    }


def replay_asof(
    frames: Mapping[str, pd.DataFrame],
    *,
    universe: Optional[list[str]] = None,
    dates: Optional[list[pd.Timestamp]] = None,
) -> tuple[list[dict], dict]:
    """
    For each asof T and symbol: data_asof = full[index <= T], then evaluate_symbol.

    Ledger is TRUE rows only. Denominator lives in stats (evaluation_count).
    """
    if universe is None:
        universe = production_universe()
    frames_n = {s: frames[s] for s in frames}
    all_dates = dates if dates is not None else trading_dates(frames_n)
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    missing_ohlcv = 0
    missing_bar = 0
    insufficient_bars = 0
    below_mc = 0
    evaluations = 0
    nan_indicator = 0
    skipped_warmup_dates = 0

    valid = set(valid_replay_dates(frames_n))
    for ts in all_dates:
        if ts not in valid:
            skipped_warmup_dates += 1

        for symbol in universe:
            df = frames_n.get(symbol)
            if df is None:
                missing_ohlcv += 1
                continue
            if ts not in df.index:
                missing_bar += 1
                continue
            prefix = scanner_p6.truncate_to_asof(df, ts)
            if len(prefix) < scanner_p6.MIN_BARS:
                insufficient_bars += 1
                continue
            if not scanner_p6.universe_ok(prefix):
                below_mc += 1
                continue
            evaluations += 1
            ind = scanner_p6.get_indicators_p6(prefix)
            if any(pd.isna(ind.get(k)) for k in ("close", "open", "high", "low")):
                nan_indicator += 1
            recs = scanner_p6.evaluate_symbol(symbol, df, asof=ts)
            for rec in recs:
                row = ledger_row_from_record(rec)
                if row["date"] != pd.Timestamp(ts).date().isoformat():
                    raise ReplayDataError(
                        f"asof_date {row['date']} != T {pd.Timestamp(ts).date().isoformat()} "
                        f"for {symbol}"
                    )
                key = _row_key(row)
                if key in seen:
                    raise ReplayDuplicateError(f"duplicate primary key {key}")
                seen.add(key)
                rows.append(row)

    rows = sort_ledger(rows)
    by_ds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_ds[(row["date"], row["symbol"])].add(row["agent"])
    agent_counts = Counter(r["agent"] for r in rows)
    sqz_rows = [r for r in rows if r["agent"] == scanner_p6.STRATEGY_SQZ]
    sqz_by_date = Counter(r["date"] for r in sqz_rows)
    sqz_by_symbol = Counter(r["symbol"] for r in sqz_rows)
    stats = {
        "purpose": LEDGER_PURPOSE,
        "min_bars": scanner_p6.MIN_BARS,
        "universe_n": len(universe),
        "frames_n": len(frames_n),
        "replay_calendar_dates": len(all_dates),
        "valid_replay_dates": len(valid),
        "earliest_data_date": all_dates[0].date().isoformat() if all_dates else None,
        "latest_data_date": all_dates[-1].date().isoformat() if all_dates else None,
        "earliest_valid_p6_date": min(valid).date().isoformat() if valid else None,
        "latest_valid_p6_date": max(valid).date().isoformat() if valid else None,
        "evaluation_count": evaluations,
        "signal_count": len(rows),
        "sqz_true": int(agent_counts.get(scanner_p6.STRATEGY_SQZ, 0)),
        "zkn_true": int(agent_counts.get(scanner_p6.STRATEGY_ZKN, 0)),
        "wyc_true": int(agent_counts.get(scanner_p6.STRATEGY_WYC, 0)),
        "unique_signal_dates": sorted({r["date"] for r in rows}),
        "unique_signal_symbols": sorted({r["symbol"] for r in rows}),
        "unique_signal_date_count": len({r["date"] for r in rows}),
        "unique_signal_symbol_count": len({r["symbol"] for r in rows}),
        "missing_ohlcv": missing_ohlcv,
        "missing_bar": missing_bar,
        "insufficient_bars": insufficient_bars,
        "below_mc": below_mc,
        "nan_indicator": nan_indicator,
        "duplicates": 0,
        "skipped_warmup_date_passes": skipped_warmup_dates,
        "overlap_sqz_zkn": sum(
            1 for agents in by_ds.values() if {"SQZ", "ZKN"} <= agents
        ),
        "overlap_sqz_wyc": sum(
            1 for agents in by_ds.values() if {"SQZ", "WYC"} <= agents
        ),
        "overlap_zkn_wyc": sum(
            1 for agents in by_ds.values() if {"ZKN", "WYC"} <= agents
        ),
        "overlap_sqz_zkn_wyc": sum(
            1 for agents in by_ds.values() if {"SQZ", "ZKN", "WYC"} <= agents
        ),
        "sqz_signals_per_date": dict(sorted(sqz_by_date.items())),
        "sqz_signals_per_symbol": dict(sorted(sqz_by_symbol.items())),
        "sqz_first_signal_date": min((r["date"] for r in sqz_rows), default=None),
        "sqz_last_signal_date": max((r["date"] for r in sqz_rows), default=None),
        "universe_missing_in_frames": sorted(s for s in universe if s not in frames_n),
        "frames_not_in_universe": sorted(s for s in frames_n if s not in set(universe)),
        "rule_versions": dict(RULE_VERSIONS),
    }
    if sqz_by_symbol:
        top_sym, top_n = sqz_by_symbol.most_common(1)[0]
        stats["sqz_top_symbol"] = top_sym
        stats["sqz_top_symbol_count"] = int(top_n)
        stats["sqz_top_symbol_share"] = round(top_n / len(sqz_rows), 4) if sqz_rows else 0.0
    else:
        stats["sqz_top_symbol"] = None
        stats["sqz_top_symbol_count"] = 0
        stats["sqz_top_symbol_share"] = 0.0
    return rows, stats


def write_ledger(path: Path, rows: list[dict]) -> None:
    assert_no_duplicates(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in sort_ledger(rows):
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_meta(path: Path, stats: dict, extra: Optional[dict] = None) -> None:
    payload = dict(stats)
    if extra:
        payload.update(extra)
    payload["purpose"] = LEDGER_PURPOSE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ledgers_equal(a: list[dict], b: list[dict]) -> bool:
    return sort_ledger(a) == sort_ledger(b)


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="P6 as-of historical replay (ANALYSIS ONLY). No new price download."
    )
    parser.add_argument(
        "--ohlcv",
        required=True,
        help="Existing OHLCV pickle dict[symbol, DataFrame] (no download).",
    )
    parser.add_argument("--out-dir", required=True, help="Offline analysis output directory.")
    args = parser.parse_args(argv)
    src = Path(args.ohlcv)
    if not src.exists():
        print("BLOCKED — HISTORICAL DATA SOURCE REQUIRED", file=sys.stderr)
        print(f"missing OHLCV pickle: {src}", file=sys.stderr)
        return 2
    frames = load_ohlcv_pickle(src)
    universe = production_universe()
    rows_a, stats = replay_asof(frames, universe=universe)
    rows_b, _ = replay_asof(frames, universe=universe)
    if not ledgers_equal(rows_a, rows_b):
        print("FAIL — NON-DETERMINISTIC REPLAY", file=sys.stderr)
        return 1
    lookahead_fail = []
    for symbol, df in list(frames.items())[: min(8, len(frames))]:
        if len(df) < scanner_p6.MIN_BARS + 1:
            continue
        asof = df.index[-2]
        chk = frames_look_ahead_equal(symbol, df, asof)
        if not all(
            chk[k]
            for k in (
                "agents_equal",
                "sqz_signal_equal",
                "recent_squeeze_equal",
                "bb_up_equal",
                "squeeze_on_prefix_equal",
                "zkn_equal",
                "wyc_equal",
            )
        ):
            lookahead_fail.append({"symbol": symbol, "asof": str(asof.date()), **chk})
    if lookahead_fail:
        print("FAIL — LOOK-AHEAD VIOLATION", file=sys.stderr)
        print(json.dumps(lookahead_fail, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    write_ledger(out_dir / "p6_signal_ledger.jsonl", rows_a)
    extra = {
        "source": str(src),
        "source_kind": "existing_ohlcv_pickle",
        "price_download_called": False,
        "determinism_pass": True,
        "look_ahead_sample_fail": lookahead_fail,
        "look_ahead_sample_n": min(8, len(frames)),
        "production_scan_history": False,
        "ranking": False,
        "score": False,
        "top_k": False,
        "forward_return": False,
        "execution": False,
        "universe_source": "scanner_p6.aktif_hisse_listesi -> scanner_p1",
        "hisseler_txt_present": (Path("hisseler.txt")).exists(),
        "universe_fallback": "scanner_p1._VARSAYILAN_SEMBOLLER" if not Path("hisseler.txt").exists() else None,
    }
    write_meta(out_dir / "p6_replay_meta.json", stats, extra)
    print(json.dumps({"purpose": LEDGER_PURPOSE, "signal_count": stats["signal_count"],
                      "evaluation_count": stats["evaluation_count"],
                      "sqz_true": stats["sqz_true"], "zkn_true": stats["zkn_true"],
                      "wyc_true": stats["wyc_true"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
