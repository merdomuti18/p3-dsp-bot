# -*- coding: utf-8 -*-
"""
P6 Etap 1.8 — forward-return / selection-edge forensic (ANALYSIS ONLY).

Uses existing ledger + existing OHLCV cache.
Does not call evaluate_symbol, does not download prices, does not rank.

fwd_return(t, period) = meta_portfolio._fwd_return_hesapla
    = log(close[t+period] / close[t+1])
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import meta_portfolio
from p6_replay import ReplayDataError, ReplayDuplicateError, load_ohlcv_pickle

PURPOSE = "HISTORICAL REPLAY / ANALYSIS ONLY — ETAP 1.8 FORENSIC"
HORIZONS = (1, 3, 5, 10, 20)
AGENTS = ("SQZ", "ZKN", "WYC")
WIN_RULE = "win = (fwd_return > 0); zero is not a win"


class LedgerIntegrityError(RuntimeError):
    pass


def load_ledger(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["date"]), str(row["symbol"]), str(row["agent"]))
            if key in seen:
                raise ReplayDuplicateError(f"duplicate ledger key {key} line {line_no}")
            seen.add(key)
            rows.append(row)
    return rows


def ledger_counts(rows: Sequence[Mapping[str, Any]]) -> dict:
    agents = Counter(r["agent"] for r in rows)
    return {
        "total": len(rows),
        "SQZ": int(agents.get("SQZ", 0)),
        "ZKN": int(agents.get("ZKN", 0)),
        "WYC": int(agents.get("WYC", 0)),
        "all_true": all(r.get("signal") is True for r in rows),
    }


def close_series(df: pd.DataFrame) -> pd.Series:
    col = "close" if "close" in df.columns else "Close"
    seri = df[col].astype(float)
    seri.index = pd.to_datetime(seri.index).normalize()
    return seri.sort_index()


def fwd_return(df: pd.DataFrame, signal_date: str, period: int) -> Optional[float]:
    """Repo formula via meta_portfolio._fwd_return_hesapla. None = cannot align."""
    seri = close_series(df)
    return meta_portfolio._fwd_return_hesapla(seri, date.fromisoformat(signal_date), int(period))


def match_status(
    frames: Mapping[str, pd.DataFrame],
    symbol: str,
    signal_date: str,
    period: int,
) -> str:
    if symbol not in frames:
        return "missing_symbol"
    df = frames[symbol]
    ts = pd.Timestamp(signal_date).normalize()
    if ts not in df.index:
        return "missing_signal_date"
    seri = close_series(df)
    if int(seri.index.duplicated().sum()) > 0:
        return "duplicate_price_timestamp"
    pos_arr = seri.index.get_indexer([ts], method=None)
    if len(pos_arr) == 0 or pos_arr[0] < 0:
        return "missing_signal_date"
    pos = int(pos_arr[0])
    if pos + 1 >= len(seri) or pos + period >= len(seri):
        return "insufficient_future_horizon"
    c1 = float(seri.iloc[pos + 1])
    cp = float(seri.iloc[pos + period])
    if math.isnan(c1) or math.isnan(cp):
        return "nan_close"
    if c1 <= 0 or cp <= 0:
        return "invalid_return"
    return "matched"


def summarize(values: Sequence[float]) -> dict:
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 0:
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "win_pct": None, "p25": None, "p75": None, "min": None, "max": None,
        }
    a = np.asarray(arr, dtype=float)
    wins = int(np.sum(a > 0.0))
    zeros = int(np.sum(a == 0.0))
    return {
        "n": n,
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=1)) if n > 1 else 0.0,
        "win_pct": float(wins / n),
        "win_count": wins,
        "zero_count": zeros,
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def winsorize(values: Sequence[float], p: float = 0.05) -> list[float]:
    """ROBUSTNESS ONLY. Not a primary result."""
    a = np.asarray(list(values), dtype=float)
    if len(a) == 0:
        return []
    lo, hi = np.quantile(a, [p, 1.0 - p])
    return [float(np.clip(x, lo, hi)) for x in a]


def mfe_mae(df: pd.DataFrame, signal_date: str, period: int) -> tuple[Optional[float], Optional[float]]:
    """Descriptive path: log(ext / close[t+1]) on bars t+1..t+period. Not execution PnL."""
    ts = pd.Timestamp(signal_date).normalize()
    if ts not in df.index:
        return None, None
    pos = int(df.index.get_indexer([ts])[0])
    if pos < 0 or pos + period >= len(df):
        return None, None
    window = df.iloc[pos + 1 : pos + period + 1]
    close_t1 = float(df["close"].iloc[pos + 1])
    if close_t1 <= 0 or window.empty:
        return None, None
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    if hi <= 0 or lo <= 0 or math.isnan(hi) or math.isnan(lo):
        return None, None
    return float(np.log(hi / close_t1)), float(np.log(lo / close_t1))


def reconstruct_eval_pairs(
    frames: Mapping[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    min_bars: int,
) -> list[tuple[str, str]]:
    """(date, symbol) with a bar on T and len(prefix)<=T >= min_bars. No agent scan."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    pairs: list[tuple[str, str]] = []
    for symbol, df in frames.items():
        for ts in df.index:
            ts = pd.Timestamp(ts).normalize()
            if ts < start_ts or ts > end_ts:
                continue
            if ts not in df.index:
                continue
            prefix_len = int((df.index <= ts).sum())
            if prefix_len < min_bars:
                continue
            pairs.append((str(ts.date()), symbol))
    return sorted(set(pairs))


def overlap_sets(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> dict:
    sa = {(r["date"], r["symbol"]) for r in rows if r["agent"] == a}
    sb = {(r["date"], r["symbol"]) for r in rows if r["agent"] == b}
    inter = sa & sb
    union = sa | sb
    jaccard = (len(inter) / len(union)) if union else None
    return {
        "a": a,
        "b": b,
        "a_only": len(sa - sb),
        "b_only": len(sb - sa),
        "intersection": len(inter),
        "union": len(union),
        "jaccard": jaccard,
        "overlap_pct_of_a": (len(inter) / len(sa)) if sa else None,
        "overlap_pct_of_b": (len(inter) / len(sb)) if sb else None,
        "intersection_keys": sorted(list(inter)),
    }


def pair_returns(
    pairs: Iterable[tuple[str, str]],
    frames: Mapping[str, pd.DataFrame],
    period: int,
) -> tuple[list[float], Counter]:
    vals: list[float] = []
    status: Counter = Counter()
    for d, sym in pairs:
        st = match_status(frames, sym, d, period)
        status[st] += 1
        if st != "matched":
            continue
        val = fwd_return(frames[sym], d, period)
        if val is None:
            status["invalid_return"] += 1
            status["matched"] -= 1
            continue
        vals.append(float(val))
    return vals, status


def agent_horizon_table(
    rows: Sequence[Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
    agent: str,
) -> dict:
    sub = [r for r in rows if r["agent"] == agent]
    out: dict[str, Any] = {"raw_signals": len(sub), "horizons": {}}
    for period in HORIZONS:
        status: Counter = Counter()
        vals: list[float] = []
        matched_rows: list[dict] = []
        for r in sub:
            st = match_status(frames, r["symbol"], r["date"], period)
            status[st] += 1
            if st != "matched":
                continue
            val = fwd_return(frames[r["symbol"]], r["date"], period)
            if val is None:
                status["invalid_return"] += 1
                status["matched"] -= 1
                continue
            vals.append(float(val))
            matched_rows.append({"date": r["date"], "symbol": r["symbol"], "fwd": float(val)})
        stats = summarize(vals)
        robust = summarize(winsorize(vals)) if vals else summarize([])
        out["horizons"][str(period)] = {
            "status": dict(status),
            "matched": int(status.get("matched", 0)),
            "stats": stats,
            "robustness_winsor_p05": robust,
            "matched_rows": matched_rows,
        }
    return out


def seeded_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 18,
) -> Optional[dict]:
    """EXPLORATORY only. Deterministic seed. Not a production-edge test."""
    a = np.asarray(list(values), dtype=float)
    if len(a) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(a, size=len(a), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "label": "EXPLORATORY",
        "n": len(a),
        "n_boot": n_boot,
        "seed": seed,
        "mean": float(np.mean(a)),
        "ci95": [float(lo), float(hi)],
    }


def run_forensic(
    rows: list[dict],
    frames: dict[str, pd.DataFrame],
    meta: dict,
) -> dict:
    counts = ledger_counts(rows)
    expected = {"total": 106, "SQZ": 31, "ZKN": 41, "WYC": 34}
    if counts["total"] != expected["total"] or any(counts[k] != expected[k] for k in ("SQZ", "ZKN", "WYC")):
        raise LedgerIntegrityError(f"ledger counts {counts} != expected {expected}")

    start = meta["earliest_valid_p6_date"]
    end = meta["latest_valid_p6_date"]
    min_bars = int(meta["min_bars"])
    eval_pairs = reconstruct_eval_pairs(frames, start=start, end=end, min_bars=min_bars)
    control_ok = len(eval_pairs) == int(meta["evaluation_count"])

    sqz_pairs = {(r["date"], r["symbol"]) for r in rows if r["agent"] == "SQZ"}
    zkn_pairs = {(r["date"], r["symbol"]) for r in rows if r["agent"] == "ZKN"}
    wyc_pairs = {(r["date"], r["symbol"]) for r in rows if r["agent"] == "WYC"}
    non_sqz_pairs = sorted(set(eval_pairs) - sqz_pairs) if control_ok else []

    agent_tables = {ag: agent_horizon_table(rows, frames, ag) for ag in AGENTS}

    uplift = {}
    if control_ok:
        for period in HORIZONS:
            sqz_vals = [x["fwd"] for x in agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]]
            ctrl_vals, ctrl_status = pair_returns(non_sqz_pairs, frames, period)
            su = summarize(sqz_vals)
            cu = summarize(ctrl_vals)
            uplift[str(period)] = {
                "sqz_mean": su["mean"],
                "sqz_median": su["median"],
                "non_sqz_mean": cu["mean"],
                "non_sqz_median": cu["median"],
                "mean_uplift": (None if su["mean"] is None or cu["mean"] is None else su["mean"] - cu["mean"]),
                "median_uplift": (
                    None if su["median"] is None or cu["median"] is None else su["median"] - cu["median"]
                ),
                "sqz_n": su["n"],
                "control_n": cu["n"],
                "control_status": dict(ctrl_status),
            }
    else:
        uplift = {"identifiable": False, "reason": "NON-SQZ CONTROL = NOT IDENTIFIABLE"}

    # Pool ablation on TRUE pairs only (no ranking).
    base_pairs = sorted(zkn_pairs | wyc_pairs)
    plus_pairs = sorted(zkn_pairs | wyc_pairs | sqz_pairs)
    ablation = {}
    for period in HORIZONS:
        b_vals, b_st = pair_returns(base_pairs, frames, period)
        p_vals, p_st = pair_returns(plus_pairs, frames, period)
        bs, ps = summarize(b_vals), summarize(p_vals)
        ablation[str(period)] = {
            "base_n": bs["n"],
            "base_mean": bs["mean"],
            "plus_n": ps["n"],
            "plus_mean": ps["mean"],
            "incremental_uplift": (
                None if bs["mean"] is None or ps["mean"] is None else ps["mean"] - bs["mean"]
            ),
            "base_status": dict(b_st),
            "plus_status": dict(p_st),
        }

    # Time stability on SQZ 5/10/20 using signal date thirds of unique SQZ dates.
    sqz_dates = sorted({r["date"] for r in rows if r["agent"] == "SQZ"})
    n_d = len(sqz_dates)
    if n_d >= 3:
        cuts = [n_d // 3, 2 * n_d // 3]
        buckets = {
            "early": set(sqz_dates[: cuts[0]]),
            "middle": set(sqz_dates[cuts[0] : cuts[1]]),
            "late": set(sqz_dates[cuts[1] :]),
        }
    else:
        buckets = {"early": set(sqz_dates), "middle": set(), "late": set()}
    time_stab = {"date_buckets": {k: sorted(v) for k, v in buckets.items()}, "horizons": {}}
    for period in (5, 10, 20):
        time_stab["horizons"][str(period)] = {}
        matched = agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]
        for name, dset in buckets.items():
            vals = [m["fwd"] for m in matched if m["date"] in dset]
            time_stab["horizons"][str(period)][name] = summarize(vals)

    # Symbol concentration vs returns.
    sqz_sym_counts = Counter(r["symbol"] for r in rows if r["agent"] == "SQZ")
    top3 = [s for s, _ in sqz_sym_counts.most_common(3)]
    conc = {"sqz_symbol_counts": dict(sqz_sym_counts), "top3": top3, "horizons": {}}
    for period in HORIZONS:
        matched = agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]
        all_v = [m["fwd"] for m in matched]
        ex_netas = [m["fwd"] for m in matched if m["symbol"] != "NETAS"]
        ex_top3 = [m["fwd"] for m in matched if m["symbol"] not in set(top3)]
        conc["horizons"][str(period)] = {
            "all": summarize(all_v),
            "excluding_NETAS": summarize(ex_netas),
            "excluding_top3": summarize(ex_top3),
        }

    # MFE/MAE descriptive for SQZ.
    mfe_tab = {}
    for period in (1, 5, 10, 20):
        mfes, maes = [], []
        st: Counter = Counter()
        for r in rows:
            if r["agent"] != "SQZ":
                continue
            ms = match_status(frames, r["symbol"], r["date"], period)
            st[ms] += 1
            if ms != "matched" or r["symbol"] not in frames:
                continue
            mfe, mae = mfe_mae(frames[r["symbol"]], r["date"], period)
            if mfe is None or mae is None:
                st["mfe_unavailable"] += 1
                continue
            mfes.append(mfe)
            maes.append(mae)
        mfe_tab[str(period)] = {
            "status": dict(st),
            "mfe": summarize(mfes),
            "mae": summarize(maes),
            "mfe_gt_0p05": (sum(1 for x in mfes if x > math.log(1.05)) / len(mfes)) if mfes else None,
            "mfe_gt_0p08": (sum(1 for x in mfes if x > math.log(1.08)) / len(mfes)) if mfes else None,
            "mfe_gt_0p10": (sum(1 for x in mfes if x > math.log(1.10)) / len(mfes)) if mfes else None,
        }

    unions = {}
    for name, s in (
        ("SQZ_OR_ZKN", sqz_pairs | zkn_pairs),
        ("SQZ_OR_WYC", sqz_pairs | wyc_pairs),
        ("ZKN_OR_WYC", zkn_pairs | wyc_pairs),
    ):
        unions[name] = {}
        for period in HORIZONS:
            vals, st = pair_returns(sorted(s), frames, period)
            unions[name][str(period)] = {"status": dict(st), "stats": summarize(vals)}

    sqz5 = [x["fwd"] for x in agent_tables["SQZ"]["horizons"]["5"]["matched_rows"]]
    sqz3 = [x["fwd"] for x in agent_tables["SQZ"]["horizons"]["3"]["matched_rows"]]

    price_cov = {
        "symbols": len(frames),
        "min_date": min(str(df.index.min().date()) for df in frames.values()) if frames else None,
        "max_date": max(str(df.index.max().date()) for df in frames.values()) if frames else None,
        "duplicate_timestamps": int(sum(int(df.index.duplicated().sum()) for df in frames.values())),
        "nan_close": int(sum(int(df["close"].isna().sum()) for df in frames.values())),
        "benchmark_symbols_in_cache": [
            s for s in frames if any(x in s.upper() for x in ("XU100", "XU030", "BIST", "XUTUM"))
        ],
        "tz": str(getattr(next(iter(frames.values())).index, "tz", None)) if frames else None,
        "columns": sorted({c for df in frames.values() for c in df.columns}),
    }

    return {
        "purpose": PURPOSE,
        "formula": "fwd_return(t, period) = log(close[t+period] / close[t+1])",
        "formula_source": "meta_portfolio._fwd_return_hesapla",
        "win_rule": WIN_RULE,
        "period_1_identity": "period=1 => log(close[t+1]/close[t+1]) = 0 when matched",
        "ranking": False,
        "score": False,
        "top_k": False,
        "execution": False,
        "signal_generation": False,
        "price_download": False,
        "ledger_counts": counts,
        "price_coverage": price_cov,
        "eval_grid": {
            "reconstructed_n": len(eval_pairs),
            "meta_evaluation_count": int(meta["evaluation_count"]),
            "checksum_ok": control_ok,
            "control_identifiable": control_ok,
            "non_sqz_n": len(non_sqz_pairs) if control_ok else None,
        },
        "agents": agent_tables,
        "overlap_sqz_zkn": overlap_sets(rows, "SQZ", "ZKN"),
        "overlap_sqz_wyc": overlap_sets(rows, "SQZ", "WYC"),
        "overlap_zkn_wyc": overlap_sets(rows, "ZKN", "WYC"),
        "selection_uplift": uplift,
        "ablation_true_pool": ablation,
        "unions": unions,
        "time_stability": time_stab,
        "symbol_concentration": conc,
        "mfe_mae_descriptive": mfe_tab,
        "alpha": {
            "computable": False,
            "reason": "NO XU100/BIST100 (or other index) series in existing price cache; download forbidden",
        },
        "exploratory_bootstrap": {
            "sqz_3d": seeded_bootstrap_mean_ci(sqz3),
            "sqz_5d": seeded_bootstrap_mean_ci(sqz5),
        },
        "score_predictiveness": "NOT APPLICABLE",
        "top_k_selection_uplift": "NOT APPLICABLE",
    }


def strip_matched_rows(payload: dict) -> dict:
    """Compact JSON without per-row fwd list (still deterministic)."""
    out = json.loads(json.dumps(payload))
    for ag in out.get("agents", {}):
        for hz in out["agents"][ag].get("horizons", {}):
            out["agents"][ag]["horizons"][hz].pop("matched_rows", None)
    return out


def _cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="P6 Etap 1.8 forensic (ledger + existing OHLCV only).")
    p.add_argument("--ledger", default=str(_REPO_ROOT / "analysis" / "p6_signal_ledger.jsonl"))
    p.add_argument("--meta", default=str(_REPO_ROOT / "analysis" / "p6_replay_meta.json"))
    p.add_argument("--ohlcv", default="/tmp/p1_forensic/ohlcv.pkl")
    p.add_argument("--out-dir", default=str(_REPO_ROOT / "analysis"))
    args = p.parse_args(argv)
    ledger_path, meta_path, ohlcv_path = Path(args.ledger), Path(args.meta), Path(args.ohlcv)
    if not ledger_path.exists():
        print("BLOCKED — USER DECISION REQUIRED: ledger missing", file=sys.stderr)
        return 2
    if not ohlcv_path.exists():
        print("BLOCKED — USER DECISION REQUIRED: price cache missing", file=sys.stderr)
        return 2
    rows = load_ledger(ledger_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = load_ohlcv_pickle(ohlcv_path)
    payload = run_forensic(rows, frames, meta)
    compact = strip_matched_rows(payload)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p6_etap18_results.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "purpose": PURPOSE,
        "eval_checksum_ok": payload["eval_grid"]["checksum_ok"],
        "sqz_5d_n": payload["agents"]["SQZ"]["horizons"]["5"]["stats"]["n"],
        "sqz_5d_mean": payload["agents"]["SQZ"]["horizons"]["5"]["stats"]["mean"],
        "alpha": payload["alpha"]["computable"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
