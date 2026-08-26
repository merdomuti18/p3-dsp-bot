# -*- coding: utf-8 -*-
"""
P6 Etap 2.0 — SQZ-v1 selection-edge forensic (ANALYSIS ONLY).

Reuses Etap 1.8 forward-return methodology from tests/p6_fwd.py.
Does not call evaluate_symbol, does not change frozen Booleans,
does not rank / execute / write production state.

fwd_return(t, period) = log(close[t+period] / close[t+1])
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scanner_p6
from p6_fwd import (
    AGENTS,
    HORIZONS,
    WIN_RULE,
    agent_horizon_table,
    ledger_counts,
    load_ledger,
    match_status,
    mfe_mae,
    overlap_sets,
    pair_returns,
    reconstruct_eval_pairs,
    strip_matched_rows,
    summarize,
)
from p6_replay import load_ohlcv_pickle
from p6_replay_expanded import apply_closed_cutoff, last_closed_asof_cutoff

PURPOSE = "HISTORICAL REPLAY / ANALYSIS ONLY — ETAP 2.0 SQZ SELECTION FORENSIC"
BOOTSTRAP_SEED = 18
BOOTSTRAP_N = 5000
INCONCLUSIVE_N = 30
EXPECTED_COUNTS = {"total": 1830, "SQZ": 425, "ZKN": 608, "WYC": 797}
EXPECTED_SQZ_USABLE_BARS = {"1": 423, "3": 413, "5": 399, "10": 389, "20": 371}
EXPECTED_RULE_VERSIONS = {
    "SQZ": "sqz-kc20-1.5-v1",
    "ZKN": "zkn-p1-birebir-v1",
    "WYC": "wyc-v1",
}
BENCHMARK_TICKERS = (("^XU100", "yahoo_bist100_index"), ("XU100.IS", "yahoo_xu100_is"))
BENCHMARK_CACHE = Path("/tmp/p6_etap20/xu100.pkl")


class LedgerIntegrityError(RuntimeError):
    pass


def rule_version_audit(rows: Sequence[Mapping[str, Any]]) -> dict:
    by_agent: dict[str, Counter] = {ag: Counter() for ag in AGENTS}
    for r in rows:
        by_agent[str(r["agent"])][str(r.get("rule_version"))] += 1
    frozen = True
    for ag, expected in EXPECTED_RULE_VERSIONS.items():
        if set(by_agent[ag]) != {expected}:
            frozen = False
    return {
        "by_agent": {ag: dict(c) for ag, c in by_agent.items()},
        "expected": dict(EXPECTED_RULE_VERSIONS),
        "frozen_ok": frozen,
    }


def reconstruct_eval_pairs_p6(
    frames: Mapping[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    min_bars: int,
    apply_universe_ok: bool = True,
) -> list[tuple[str, str]]:
    """
    Evaluation-universe (date, symbol) pairs.

    Same as-of grid as Etap 1.9 replay: bar on T, prefix<=T has MIN_BARS,
    and (optionally) scanner_p6.universe_ok. Does not call evaluate_symbol.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    pairs: list[tuple[str, str]] = []
    for symbol, df in frames.items():
        idx = df.index
        for ts in idx:
            ts = pd.Timestamp(ts).normalize()
            if ts < start_ts or ts > end_ts:
                continue
            prefix_len = int((idx <= ts).sum())
            if prefix_len < min_bars:
                continue
            if apply_universe_ok:
                prefix = df.loc[idx <= ts]
                if not scanner_p6.universe_ok(prefix):
                    continue
            pairs.append((str(ts.date()), symbol))
    return sorted(set(pairs))


def unique_pairs(rows: Sequence[Mapping[str, Any]], agent: str) -> set[tuple[str, str]]:
    return {(r["date"], r["symbol"]) for r in rows if r["agent"] == agent}


def overlap_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    sqz = unique_pairs(rows, "SQZ")
    zkn = unique_pairs(rows, "ZKN")
    wyc = unique_pairs(rows, "WYC")
    return {
        "SQZ_only": sqz - zkn - wyc,
        "SQZ_and_ZKN": (sqz & zkn) - wyc,
        "SQZ_and_WYC": (sqz & wyc) - zkn,
        "SQZ_and_ZKN_and_WYC": sqz & zkn & wyc,
        "SQZ": sqz,
        "ZKN": zkn,
        "WYC": wyc,
        "BASE": zkn | wyc,
        "BASE_plus_SQZ": zkn | wyc | sqz,
        "BASE_WYC": set(wyc),
        "BASE_WYC_plus_ZKN": wyc | zkn,
        "BASE_SQZ": set(sqz),
        "BASE_SQZ_plus_ZKN": sqz | zkn,
    }


def split_date_thirds(dates: Sequence[str]) -> dict[str, Any]:
    """Equal-count thirds of sorted unique as-of dates (expanded eval window)."""
    uniq = sorted(set(dates))
    n = len(uniq)
    if n == 0:
        empty: set[str] = set()
        return {
            "method": "equal-count unique as-of dates (EARLY/MIDDLE/LATE)",
            "n_dates": 0,
            "cuts": None,
            "ranges": {},
            "buckets": {"EARLY": empty, "MIDDLE": empty, "LATE": empty},
        }
    i1 = n // 3
    i2 = 2 * n // 3
    buckets = {
        "EARLY": set(uniq[:i1]),
        "MIDDLE": set(uniq[i1:i2]),
        "LATE": set(uniq[i2:]),
    }
    ranges = {}
    for name, s in buckets.items():
        ordered = sorted(s)
        ranges[name] = {
            "n_dates": len(ordered),
            "start": ordered[0] if ordered else None,
            "end": ordered[-1] if ordered else None,
        }
    return {
        "method": (
            "sorted unique as-of dates in the expanded evaluation window, "
            "split into three equal-count buckets (n//3, 2n//3)"
        ),
        "n_dates": n,
        "cuts": {
            "early_end_exclusive": uniq[i1] if i1 < n else None,
            "middle_end_exclusive": uniq[i2] if i2 < n else None,
        },
        "ranges": ranges,
        "buckets": buckets,
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Hedges-uncorrected Cohen's d (pooled sample SD, ddof=1)."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    nx, ny = len(x), len(y)
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0.0 or math.isnan(pooled):
        return None
    return float((np.mean(x) - np.mean(y)) / pooled)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Robust stochastic superiority: P(A>B) - P(A<B). Not a production proof."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    # Vectorised pairwise via searchsorted (O(n log m)).
    y_sorted = np.sort(y)
    n = len(y_sorted)
    # For each x: # of y < x  => A > B;  # of y > x => A < B.
    n_lt_y = np.searchsorted(y_sorted, x, side="left")
    n_le_y = np.searchsorted(y_sorted, x, side="right")
    n_gt = float(np.sum(n_lt_y))
    n_lt = float(np.sum(n - n_le_y))
    denom = float(len(x) * len(y))
    return float((n_gt - n_lt) / denom)


def seeded_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[dict]:
    """Two-sample bootstrap CI for mean and median difference. EXPLORATORY."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    rng = np.random.default_rng(seed)
    mean_diffs = np.empty(n_boot, dtype=float)
    median_diffs = np.empty(n_boot, dtype=float)
    nx, ny = len(x), len(y)
    for i in range(n_boot):
        xs = rng.choice(x, size=nx, replace=True)
        ys = rng.choice(y, size=ny, replace=True)
        mean_diffs[i] = float(np.mean(xs) - np.mean(ys))
        median_diffs[i] = float(np.median(xs) - np.median(ys))
    obs_mean = float(np.mean(x) - np.mean(y))
    obs_median = float(np.median(x) - np.median(y))
    # Two-sided helper p: share of bootstrap diffs on the other side of 0.
    p_mean = float(2.0 * min(np.mean(mean_diffs <= 0.0), np.mean(mean_diffs >= 0.0)))
    p_mean = min(1.0, p_mean)
    mean_lo, mean_hi = np.percentile(mean_diffs, [2.5, 97.5])
    med_lo, med_hi = np.percentile(median_diffs, [2.5, 97.5])
    return {
        "label": "EXPLORATORY",
        "n_a": nx,
        "n_b": ny,
        "n_boot": n_boot,
        "seed": seed,
        "mean_diff": obs_mean,
        "median_diff": obs_median,
        "mean_ci95": [float(mean_lo), float(mean_hi)],
        "median_ci95": [float(med_lo), float(med_hi)],
        "mean_ci_crosses_zero": bool(mean_lo <= 0.0 <= mean_hi),
        "median_ci_crosses_zero": bool(med_lo <= 0.0 <= med_hi),
        "helper_p_mean_two_sided": p_mean,
        "not_production_edge_proof": True,
    }


def pool_horizon_stats(
    pairs: Iterable[tuple[str, str]],
    frames: Mapping[str, pd.DataFrame],
    period: int,
) -> dict:
    vals, status = pair_returns(pairs, frames, period)
    stats = summarize(vals)
    return {"status": dict(status), "stats": stats, "values": vals}


def incremental_block(
    base_pairs: set[tuple[str, str]],
    plus_pairs: set[tuple[str, str]],
    frames: Mapping[str, pd.DataFrame],
) -> dict:
    out: dict[str, Any] = {}
    for period in HORIZONS:
        base = pool_horizon_stats(sorted(base_pairs), frames, period)
        plus = pool_horizon_stats(sorted(plus_pairs), frames, period)
        bs, ps = base["stats"], plus["stats"]
        out[str(period)] = {
            "base_n": bs["n"],
            "base_mean": bs["mean"],
            "base_median": bs["median"],
            "base_win_pct": bs["win_pct"],
            "plus_n": ps["n"],
            "plus_mean": ps["mean"],
            "plus_median": ps["median"],
            "plus_win_pct": ps["win_pct"],
            "delta_mean": (
                None if bs["mean"] is None or ps["mean"] is None else ps["mean"] - bs["mean"]
            ),
            "delta_median": (
                None
                if bs["median"] is None or ps["median"] is None
                else ps["median"] - bs["median"]
            ),
            "delta_win": (
                None
                if bs["win_pct"] is None or ps["win_pct"] is None
                else ps["win_pct"] - bs["win_pct"]
            ),
            "base_status": base["status"],
            "plus_status": plus["status"],
        }
    return out


def group_horizon_stats(
    pairs: set[tuple[str, str]],
    frames: Mapping[str, pd.DataFrame],
    horizons: Sequence[int] = (3, 5, 10, 20),
) -> dict:
    n_raw = len(pairs)
    flag = "INCONCLUSIVE" if n_raw < INCONCLUSIVE_N else "OK"
    out: dict[str, Any] = {
        "n_pairs": n_raw,
        "sample_flag": flag,
        "inconclusive_n_threshold": INCONCLUSIVE_N,
        "horizons": {},
    }
    for period in horizons:
        packed = pool_horizon_stats(sorted(pairs), frames, period)
        st = packed["stats"]
        out["horizons"][str(period)] = {
            "status": packed["status"],
            "stats": {k: v for k, v in st.items() if k != "values"},
            "sample_flag": (
                "INCONCLUSIVE" if st["n"] < INCONCLUSIVE_N else flag
            ),
        }
    return out


def symbol_concentration(rows: Sequence[Mapping[str, Any]], agent: str = "SQZ") -> dict:
    sub = [r for r in rows if r["agent"] == agent]
    counts = Counter(r["symbol"] for r in sub)
    n = len(sub)
    unique = len(counts)
    ranked = counts.most_common()
    top1 = ranked[:1]
    top3 = ranked[:3]
    top5 = ranked[:5]

    def share(items: list[tuple[str, int]]) -> Optional[float]:
        if n == 0:
            return None
        return float(sum(c for _, c in items) / n)

    return {
        "n_signals": n,
        "unique_symbols": unique,
        "top1": [{"symbol": s, "n": c} for s, c in top1],
        "top3": [{"symbol": s, "n": c} for s, c in top3],
        "top5": [{"symbol": s, "n": c} for s, c in top5],
        "top1_share": share(top1),
        "top3_share": share(top3),
        "top5_share": share(top5),
        "counts": dict(counts),
    }


def sensitivity_drop_symbols(
    matched_rows: Sequence[Mapping[str, Any]],
    drop: set[str],
) -> dict:
    kept = [float(m["fwd"]) for m in matched_rows if m["symbol"] not in drop]
    dropped_n = sum(1 for m in matched_rows if m["symbol"] in drop)
    return {"dropped_n": dropped_n, "stats": summarize(kept)}


def mfe_mae_table(
    rows: Sequence[Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
    agent: str = "SQZ",
) -> dict:
    tab: dict[str, Any] = {}
    for period in HORIZONS:
        mfes: list[float] = []
        maes: list[float] = []
        st: Counter = Counter()
        for r in rows:
            if r["agent"] != agent:
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
        tab[str(period)] = {
            "status": dict(st),
            "n": len(mfes),
            "mean_mfe": None if not mfes else float(np.mean(mfes)),
            "median_mfe": None if not mfes else float(np.median(mfes)),
            "mean_mae": None if not maes else float(np.mean(maes)),
            "median_mae": None if not maes else float(np.median(maes)),
            "mfe_gt_0p05": (sum(1 for x in mfes if x > math.log(1.05)) / len(mfes)) if mfes else None,
            "mfe_gt_0p08": (sum(1 for x in mfes if x > math.log(1.08)) / len(mfes)) if mfes else None,
            "mfe_gt_0p10": (sum(1 for x in mfes if x > math.log(1.10)) / len(mfes)) if mfes else None,
            "not_execution_pnl": True,
        }
    return tab


def try_download_xu100(cutoff: pd.Timestamp) -> tuple[Optional[pd.Series], dict]:
    """
    Analysis-only XU100 close series. Does not call scanner_p1.veri_hazirla.
    No impute. Returns None if a reliable series cannot be obtained.
    """
    report: dict[str, Any] = {
        "attempted": True,
        "production_veri_hazirla_called": False,
        "tickers_tried": [t for t, _ in BENCHMARK_TICKERS],
        "chosen": None,
        "cache_path": str(BENCHMARK_CACHE),
        "cutoff": str(pd.Timestamp(cutoff).date()),
    }
    if BENCHMARK_CACHE.exists():
        try:
            raw = pd.read_pickle(BENCHMARK_CACHE)
            if isinstance(raw, dict) and "close" in raw and "ticker" in raw:
                seri = pd.Series(raw["close"])
                seri.index = pd.to_datetime(seri.index).normalize()
                seri = seri.sort_index()
                seri = seri.loc[seri.index <= pd.Timestamp(cutoff).normalize()]
                seri = seri[seri > 0].dropna()
                if len(seri) >= 40:
                    report["chosen"] = {
                        "ticker": raw.get("ticker"),
                        "source": raw.get("source"),
                        "from_cache": True,
                        "n": int(len(seri)),
                        "min_date": str(seri.index.min().date()),
                        "max_date": str(seri.index.max().date()),
                    }
                    return seri, report
        except Exception as exc:
            report["cache_error"] = str(exc)

    try:
        import yfinance as yf
    except Exception as exc:
        report["error"] = f"yfinance import failed: {exc}"
        return None, report

    errors: list[str] = []
    for ticker, source in BENCHMARK_TICKERS:
        try:
            hist = yf.Ticker(ticker).history(period="8mo", interval="1d", auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns:
                errors.append(f"{ticker}: empty")
                continue
            seri = hist["Close"].astype(float)
            idx = pd.to_datetime(seri.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(None)
            seri.index = idx.normalize()
            seri = seri.sort_index()
            seri = seri.loc[seri.index <= pd.Timestamp(cutoff).normalize()]
            seri = seri[seri > 0].dropna()
            dup = int(seri.index.duplicated().sum())
            if dup:
                errors.append(f"{ticker}: {dup} duplicate timestamps — not used")
                continue
            if len(seri) < 40:
                errors.append(f"{ticker}: too short n={len(seri)}")
                continue
            BENCHMARK_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(
                {"close": seri, "ticker": ticker, "source": source},
                BENCHMARK_CACHE,
            )
            report["chosen"] = {
                "ticker": ticker,
                "source": source,
                "from_cache": False,
                "n": int(len(seri)),
                "min_date": str(seri.index.min().date()),
                "max_date": str(seri.index.max().date()),
                "auto_adjust": True,
            }
            report["errors"] = errors
            return seri, report
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
    report["errors"] = errors
    report["chosen"] = None
    return None, report


def benchmark_fwd(seri: pd.Series, signal_date: str, period: int) -> Optional[float]:
    import meta_portfolio

    return meta_portfolio._fwd_return_hesapla(seri, date.fromisoformat(signal_date), int(period))


def alpha_values(
    matched_rows: Sequence[Mapping[str, Any]],
    bench: pd.Series,
    period: int,
) -> tuple[list[float], Counter]:
    vals: list[float] = []
    st: Counter = Counter()
    for m in matched_rows:
        stock = float(m["fwd"])
        b = benchmark_fwd(bench, m["date"], period)
        if b is None:
            st["benchmark_unavailable"] += 1
            continue
        st["matched"] += 1
        vals.append(stock - float(b))
    return vals, st


def _sign(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    if x > 0:
        return "pos"
    if x < 0:
        return "neg"
    return "zero"


def classify_sqz(payload: Mapping[str, Any]) -> dict:
    """
    Transparent decision-matrix application. Does not use a single metric.
    1D is an identity (always 0 when matched) and is not used for A/B/C/D.
    """
    focus = ("5", "10", "20")
    uplift = payload.get("selection_uplift") or {}
    incremental = (payload.get("incremental_base_vs_base_plus_sqz") or {})
    time_stab = (payload.get("time_stability") or {}).get("horizons") or {}
    conc = payload.get("symbol_sensitivity") or {}
    reasons: list[str] = []
    blockers: list[str] = []

    if not payload.get("eval_grid", {}).get("checksum_ok"):
        return {
            "class": "F",
            "recommendation": "INCONCLUSIVE",
            "reasons": ["eval-grid checksum failed; NON-SQZ control not identifiable"],
            "blockers": ["control_not_identifiable"],
        }

    mean_u = []
    median_u = []
    win_u = []
    ci_cross = []
    incr_mean = []
    for h in focus:
        u = uplift.get(h) or {}
        inc = incremental.get(h) or {}
        mean_u.append(u.get("mean_uplift"))
        median_u.append(u.get("median_uplift"))
        sqz_w = u.get("sqz_win_pct")
        ctrl_w = u.get("control_win_pct")
        win_u.append(None if sqz_w is None or ctrl_w is None else sqz_w - ctrl_w)
        boot = u.get("bootstrap") or {}
        ci_cross.append(bool(boot.get("mean_ci_crosses_zero", True)))
        incr_mean.append(inc.get("delta_mean"))

    if any(v is None for v in mean_u + median_u):
        return {
            "class": "F",
            "recommendation": "INCONCLUSIVE",
            "reasons": ["missing 5D/10D/20D mean or median uplift"],
            "blockers": ["missing_horizon_stats"],
        }

    n_mean_pos = sum(1 for v in mean_u if v is not None and v > 0)
    n_mean_neg = sum(1 for v in mean_u if v is not None and v < 0)
    n_med_pos = sum(1 for v in median_u if v is not None and v > 0)
    n_med_neg = sum(1 for v in median_u if v is not None and v < 0)
    n_win_pos = sum(1 for v in win_u if v is not None and v > 0)
    n_win_neg = sum(1 for v in win_u if v is not None and v < 0)
    n_ci_cross = sum(1 for c in ci_cross if c)
    n_incr_pos = sum(1 for v in incr_mean if v is not None and v > 0)
    n_incr_neg = sum(1 for v in incr_mean if v is not None and v < 0)

    reasons.append(
        f"focus 5/10/20 mean_uplift signs pos={n_mean_pos} neg={n_mean_neg}; "
        f"median pos={n_med_pos} neg={n_med_neg}; win% pos={n_win_pos} neg={n_win_neg}; "
        f"mean CI crosses 0 on {n_ci_cross}/3; incremental mean pos={n_incr_pos} neg={n_incr_neg}"
    )

    # Time stability: 5D mean per period when N>=threshold.
    period_means = []
    for name in ("EARLY", "MIDDLE", "LATE"):
        block = ((time_stab.get("5") or {}).get(name)) or {}
        n = block.get("n") or 0
        if n < INCONCLUSIVE_N:
            period_means.append(None)
            reasons.append(f"{name} 5D N={n} < {INCONCLUSIVE_N} → INCONCLUSIVE bucket")
        else:
            period_means.append(block.get("mean"))
    period_signs = [_sign(v) for v in period_means if v is not None]
    mixed_periods = len(set(period_signs)) > 1 if period_signs else True
    if mixed_periods:
        blockers.append("time_stability_mixed_or_thin")

    # Symbol sensitivity: 5D mean sign flip after dropping top-3.
    s5 = (conc.get("5") or {})
    all_mean = ((s5.get("all") or {}).get("mean"))
    ex3_mean = ((s5.get("excluding_top3") or {}).get("mean"))
    conc_explains = (
        all_mean is not None
        and ex3_mean is not None
        and ((all_mean > 0 > ex3_mean) or (all_mean < 0 < ex3_mean))
    )
    if conc_explains:
        blockers.append("symbol_concentration_sign_flip")
        reasons.append("5D mean sign flips after dropping top-3 symbols")

    overlap_low = (
        (payload.get("overlap_sqz_zkn") or {}).get("intersection", 0) == 0
        and (payload.get("overlap_sqz_wyc") or {}).get("intersection", 0) == 0
    )
    # Low overlap is complementarity candidate, NOT automatic D.
    reasons.append(
        "low SQZ∩ZKN / SQZ∩WYC is not by itself D (redundant); "
        "D requires weak incremental pool value"
    )

    consistent_pos = n_mean_pos == 3 and n_med_pos == 3 and n_win_pos >= 2
    consistent_neg = n_mean_neg == 3 and n_med_neg == 3
    incremental_pos = n_incr_pos >= 2 and n_incr_neg == 0
    incremental_neg = n_incr_neg >= 2 and n_incr_pos == 0
    incremental_near_zero = (
        all(v is not None and abs(v) < 0.001 for v in incr_mean)
        and n_incr_pos <= 1
    )

    if consistent_neg and incremental_neg:
        klass = "C"
        rec = "DROP-AS-CANDIDATE"
        reasons.append("5D/10D/20D mean and median vs control are jointly negative; pool incremental is negative")
    elif consistent_pos and incremental_near_zero and not overlap_low:
        klass = "D"
        rec = "DROP-AS-CANDIDATE"
        reasons.append("SQZ may look non-negative vs control but BASE+SQZ incremental is ~0")
    elif consistent_pos and incremental_pos and n_ci_cross == 0 and not mixed_periods and not conc_explains:
        klass = "A"
        rec = "RETAIN"
        reasons.append("5D/10D/20D aligned positive vs control; CI excludes 0; incremental positive; periods stable")
    elif consistent_pos and incremental_pos and not conc_explains:
        klass = "B"
        rec = "RETAIN-AS-CANDIDATE"
        reasons.append(
            "generally positive 5D/10D/20D selection signal with incremental pool value, "
            "but CI/period/win alignment is not A-strong"
        )
    elif n_mean_pos >= 2 and n_med_pos >= 2 and incremental_pos and not conc_explains:
        klass = "B"
        rec = "RETAIN-AS-CANDIDATE"
        reasons.append("majority of 5D/10D/20D mean+median positive with incremental value")
    elif n_mean_neg >= 2 and n_med_neg >= 2 and (incremental_neg or n_win_neg >= 2):
        klass = "C"
        rec = "DROP-AS-CANDIDATE"
        reasons.append("majority of focus horizons jointly negative vs control")
    elif mixed_periods or n_ci_cross == 3 or conc_explains:
        klass = "F"
        rec = "INCONCLUSIVE"
        reasons.append("horizons/periods/CI/concentration do not support A/B/C/D")
    elif n_mean_pos > 0 and n_mean_neg > 0:
        klass = "F"
        rec = "INCONCLUSIVE"
        reasons.append("focus-horizon mean uplifts disagree in sign")
    else:
        klass = "F"
        rec = "INCONCLUSIVE"
        reasons.append("no decision-matrix cell fully satisfied")

    return {
        "class": klass,
        "recommendation": rec,
        "reasons": reasons,
        "blockers": blockers,
        "focus_horizons": list(focus),
        "mean_uplift": mean_u,
        "median_uplift": median_u,
        "win_uplift": win_u,
        "incremental_delta_mean": incr_mean,
        "mean_ci_crosses_zero": ci_cross,
        "note": (
            "Classification uses 5D/10D/20D jointly; 1D is the period=1 identity "
            "(always 0 when matched) and is not a performance claim. "
            "Overlapping horizons are not independent observations."
        ),
    }


def run_etap20(
    rows: list[dict],
    frames: dict[str, pd.DataFrame],
    meta: dict,
    *,
    benchmark: Optional[pd.Series] = None,
    benchmark_meta: Optional[dict] = None,
    expected_counts: Optional[Mapping[str, int]] = None,
) -> dict:
    counts = ledger_counts(rows)
    expected = dict(expected_counts) if expected_counts is not None else dict(EXPECTED_COUNTS)
    if counts["total"] != expected["total"] or any(
        counts[k] != expected[k] for k in ("SQZ", "ZKN", "WYC")
    ):
        raise LedgerIntegrityError(f"ledger counts {counts} != expected {expected}")
    if not counts["all_true"]:
        raise LedgerIntegrityError("ledger contains non-TRUE signal rows")

    rv = rule_version_audit(rows)
    if not rv["frozen_ok"]:
        raise LedgerIntegrityError(f"rule_version drift: {rv}")

    start = meta["earliest_valid_p6_date"]
    end = meta["latest_valid_p6_date"]
    min_bars = int(meta["min_bars"])
    eval_pairs = reconstruct_eval_pairs_p6(
        frames, start=start, end=end, min_bars=min_bars, apply_universe_ok=True
    )
    eval_pairs_min_bars_only = reconstruct_eval_pairs(
        frames, start=start, end=end, min_bars=min_bars
    )
    control_ok = len(eval_pairs) == int(meta["evaluation_count"])

    groups = overlap_groups(rows)
    sqz_pairs = groups["SQZ"]
    non_sqz_pairs = sorted(set(eval_pairs) - sqz_pairs) if control_ok else []

    agent_tables = {ag: agent_horizon_table(rows, frames, ag) for ag in AGENTS}

    # SQZ usable vs Etap 1.9 bar-availability expectation.
    sqz_usable = {
        h: int(agent_tables["SQZ"]["horizons"][h]["stats"]["n"]) for h in ("1", "3", "5", "10", "20")
    }
    coverage_delta = {
        h: sqz_usable[h] - EXPECTED_SQZ_USABLE_BARS[h] for h in EXPECTED_SQZ_USABLE_BARS
    }

    uplift: dict[str, Any] = {}
    if control_ok:
        for period in HORIZONS:
            sqz_vals = [x["fwd"] for x in agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]]
            ctrl_vals, ctrl_status = pair_returns(non_sqz_pairs, frames, period)
            su = summarize(sqz_vals)
            cu = summarize(ctrl_vals)
            boot = seeded_bootstrap_diff_ci(sqz_vals, ctrl_vals)
            uplift[str(period)] = {
                "sqz_n": su["n"],
                "sqz_mean": su["mean"],
                "sqz_median": su["median"],
                "sqz_win_pct": su["win_pct"],
                "sqz_p25": su["p25"],
                "sqz_p75": su["p75"],
                "control_n": cu["n"],
                "control_mean": cu["mean"],
                "control_median": cu["median"],
                "control_win_pct": cu["win_pct"],
                "control_p25": cu["p25"],
                "control_p75": cu["p75"],
                "mean_uplift": (
                    None if su["mean"] is None or cu["mean"] is None else su["mean"] - cu["mean"]
                ),
                "median_uplift": (
                    None
                    if su["median"] is None or cu["median"] is None
                    else su["median"] - cu["median"]
                ),
                "win_uplift": (
                    None
                    if su["win_pct"] is None or cu["win_pct"] is None
                    else su["win_pct"] - cu["win_pct"]
                ),
                "cohens_d": cohens_d(sqz_vals, ctrl_vals),
                "cliffs_delta": cliffs_delta(sqz_vals, ctrl_vals),
                "effect_size_method": "Cohen_d_pooled_sd; robust_companion=Cliff_delta",
                "bootstrap": boot,
                "control_status": dict(ctrl_status),
                "sqz_stats": su,
                "control_stats": cu,
            }
    else:
        uplift = {
            "identifiable": False,
            "reason": "NON-SQZ CONTROL = NOT IDENTIFIABLE (eval-grid checksum failed)",
        }

    incremental = incremental_block(groups["BASE"], groups["BASE_plus_SQZ"], frames)
    zkn_vs_wyc = incremental_block(groups["BASE_WYC"], groups["BASE_WYC_plus_ZKN"], frames)
    zkn_vs_sqz = incremental_block(groups["BASE_SQZ"], groups["BASE_SQZ_plus_ZKN"], frames)

    asof_dates = sorted({d for d, _ in eval_pairs})
    thirds = split_date_thirds(asof_dates)
    time_stab = {
        "method": thirds["method"],
        "n_dates": thirds["n_dates"],
        "cuts": thirds["cuts"],
        "ranges": thirds["ranges"],
        "inconclusive_n_threshold": INCONCLUSIVE_N,
        "horizons": {},
    }
    for period in (5, 10, 20):
        matched = agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]
        time_stab["horizons"][str(period)] = {}
        for name, dset in thirds["buckets"].items():
            vals = [m["fwd"] for m in matched if m["date"] in dset]
            stats = summarize(vals)
            time_stab["horizons"][str(period)][name] = {
                **stats,
                "sample_flag": "INCONCLUSIVE" if stats["n"] < INCONCLUSIVE_N else "OK",
            }

    conc_meta = symbol_concentration(rows, "SQZ")
    top1_syms = {x["symbol"] for x in conc_meta["top1"]}
    top3_syms = {x["symbol"] for x in conc_meta["top3"]}
    sqz5_rows = agent_tables["SQZ"]["horizons"]["5"]["matched_rows"]
    sqz5_counts = Counter(m["symbol"] for m in sqz5_rows)
    n5 = len(sqz5_rows)
    top5_in_5d = sqz5_counts.most_common(5)
    sensitivity = {
        "all_sqz": conc_meta,
        "sqz_5d_usable": {
            "n": n5,
            "unique_symbols": len(sqz5_counts),
            "top5": [{"symbol": s, "n": c, "share": (c / n5 if n5 else None)} for s, c in top5_in_5d],
            "top1_share": (sqz5_counts.most_common(1)[0][1] / n5) if n5 and sqz5_counts else None,
            "top3_share": (
                sum(c for _, c in sqz5_counts.most_common(3)) / n5 if n5 else None
            ),
        },
        "horizons": {},
    }
    for period in HORIZONS:
        matched = agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]
        all_v = [m["fwd"] for m in matched]
        sensitivity["horizons"][str(period)] = {
            "all": summarize(all_v),
            "excluding_top1": sensitivity_drop_symbols(matched, top1_syms)["stats"],
            "excluding_top3": sensitivity_drop_symbols(matched, top3_syms)["stats"],
            "dropped_top1": sorted(top1_syms),
            "dropped_top3": sorted(top3_syms),
        }
    # flatten 5 for classifier convenience
    sensitivity["5"] = sensitivity["horizons"]["5"]

    overlap_group_stats = {
        name: group_horizon_stats(groups[name], frames)
        for name in ("SQZ_only", "SQZ_and_ZKN", "SQZ_and_WYC", "SQZ_and_ZKN_and_WYC")
    }

    # Alpha
    alpha_block: dict[str, Any]
    if benchmark is None or len(benchmark) < 40:
        alpha_block = {
            "computable": False,
            "status": "NOT COMPUTABLE",
            "reason": (benchmark_meta or {}).get(
                "reason",
                "no reliable XU100/BIST100 series (missing, too short, or duplicates)",
            ),
            "benchmark": benchmark_meta,
            "note": "raw stock forward returns are NOT labeled alpha",
        }
    else:
        alpha_block = {
            "computable": True,
            "status": "COMPUTED",
            "formula": "alpha(t,period)=stock_fwd_return(t,period)-benchmark_fwd_return(t,period)",
            "benchmark": benchmark_meta,
            "horizons": {},
        }
        for period in HORIZONS:
            sqz_rows_h = agent_tables["SQZ"]["horizons"][str(period)]["matched_rows"]
            a_vals, a_st = alpha_values(sqz_rows_h, benchmark, period)
            alpha_block["horizons"][str(period)] = {
                "status": dict(a_st),
                "stats": summarize(a_vals),
            }

    price_cov = {
        "symbols": len(frames),
        "min_date": min(str(df.index.min().date()) for df in frames.values()) if frames else None,
        "max_date": max(str(df.index.max().date()) for df in frames.values()) if frames else None,
        "duplicate_timestamps": int(sum(int(df.index.duplicated().sum()) for df in frames.values())),
        "nan_close": int(sum(int(df["close"].isna().sum()) for df in frames.values())),
        "tz": str(getattr(next(iter(frames.values())).index, "tz", None)) if frames else None,
        "columns": sorted({c for df in frames.values() for c in df.columns}),
        "source": "/tmp/p6_etap19/ohlcv.pkl",
        "source_kind": "etap19_expanded_cache",
    }

    payload = {
        "purpose": PURPOSE,
        "formula": "fwd_return(t, period) = log(close[t+period] / close[t+1])",
        "formula_source": "meta_portfolio._fwd_return_hesapla",
        "win_rule": WIN_RULE,
        "period_1_identity": "period=1 => log(close[t+1]/close[t+1]) = 0 when matched",
        "return_unit": "LOG RETURN",
        "impute": False,
        "ranking": False,
        "score": False,
        "top_k": False,
        "execution": False,
        "signal_generation": False,
        "price_download_production": False,
        "layer": "A_SELECTION_NOT_EXECUTION",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_N, "label": "EXPLORATORY"},
        "ledger_counts": counts,
        "rule_versions": rv,
        "sqz_usable_vs_etap19_bar_availability": {
            "expected": EXPECTED_SQZ_USABLE_BARS,
            "observed_matched": sqz_usable,
            "delta": coverage_delta,
            "note": (
                "Etap 1.9 usable counts were bar-availability only. "
                "Observed here additionally requires price>0 at t+1 and t+period."
            ),
        },
        "price_coverage": price_cov,
        "eval_grid": {
            "reconstructed_n_with_universe_ok": len(eval_pairs),
            "reconstructed_n_min_bars_only": len(eval_pairs_min_bars_only),
            "meta_evaluation_count": int(meta["evaluation_count"]),
            "meta_below_mc": int(meta.get("below_mc", 0)),
            "checksum_ok": control_ok,
            "control_identifiable": control_ok,
            "non_sqz_n": len(non_sqz_pairs) if control_ok else None,
            "construction": (
                "same expanded as-of window; bar on T; prefix<=T len>=MIN_BARS; "
                "scanner_p6.universe_ok (mc>=10M); SQZ FALSE. "
                "evaluate_symbol is not re-run."
            ),
        },
        "agents": agent_tables,
        "overlap_sqz_zkn": overlap_sets(rows, "SQZ", "ZKN"),
        "overlap_sqz_wyc": overlap_sets(rows, "SQZ", "WYC"),
        "overlap_zkn_wyc": overlap_sets(rows, "ZKN", "WYC"),
        "selection_uplift": uplift,
        "incremental_base_vs_base_plus_sqz": incremental,
        "incremental_wyc_vs_wyc_plus_zkn": zkn_vs_wyc,
        "incremental_sqz_vs_sqz_plus_zkn": zkn_vs_sqz,
        "overlap_groups": overlap_group_stats,
        "time_stability": time_stab,
        "symbol_concentration": conc_meta,
        "symbol_sensitivity": sensitivity,
        "mfe_mae_descriptive": mfe_mae_table(rows, frames, "SQZ"),
        "alpha": alpha_block,
        "multiple_horizon_caution": {
            "horizons": list(HORIZONS),
            "note": (
                "1D/3D/5D/10D/20D are tested together (multiple testing). "
                "10D/20D observations overlap 5D paths on the same names/dates "
                "and are not independent. Do not treat five significant-looking "
                "horizons as five independent confirmations."
            ),
        },
        "score_predictiveness": "NOT APPLICABLE",
        "top_k_selection_uplift": "NOT APPLICABLE",
    }
    payload["classification"] = classify_sqz(payload)
    return payload


def compact_payload(payload: dict) -> dict:
    out = strip_matched_rows(payload)
    for key in ("overlap_sqz_zkn", "overlap_sqz_wyc", "overlap_zkn_wyc"):
        if key in out and "intersection_keys" in out[key]:
            keys = out[key]["intersection_keys"]
            out[key]["intersection_keys_n"] = len(keys)
            # Keep keys only when tiny (audit); else drop to keep JSON small.
            if len(keys) > 32:
                out[key]["intersection_keys"] = None
    return out


def _cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="P6 Etap 2.0 SQZ selection forensic (analysis only).")
    p.add_argument("--ledger", default=str(_REPO_ROOT / "analysis" / "p6_signal_ledger_expanded.jsonl"))
    p.add_argument("--meta", default=str(_REPO_ROOT / "analysis" / "p6_replay_expanded_meta.json"))
    p.add_argument("--ohlcv", default="/tmp/p6_etap19/ohlcv.pkl")
    p.add_argument("--out", default=str(_REPO_ROOT / "analysis" / "p6_etap20_results.json"))
    p.add_argument("--skip-benchmark-download", action="store_true")
    args = p.parse_args(argv)

    ledger_path, meta_path, ohlcv_path = Path(args.ledger), Path(args.meta), Path(args.ohlcv)
    if not ledger_path.exists():
        print("BLOCKED — expanded ledger missing", file=sys.stderr)
        return 2
    if not ohlcv_path.exists():
        print("BLOCKED — expanded OHLCV cache missing", file=sys.stderr)
        return 2
    rows = load_ledger(ledger_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = load_ohlcv_pickle(ohlcv_path)
    # Keep the same closed-session cutoff as Etap 1.9 if the cache still has a later bar.
    cutoff = pd.Timestamp(meta.get("closed_asof_cutoff") or last_closed_asof_cutoff()).normalize()
    frames, _applied = apply_closed_cutoff(frames, cutoff)

    bench = None
    bench_meta: dict[str, Any]
    if args.skip_benchmark_download:
        bench_meta = {
            "attempted": False,
            "reason": "benchmark download skipped by flag",
            "production_veri_hazirla_called": False,
        }
    else:
        bench, bench_meta = try_download_xu100(cutoff)
        if bench is None:
            bench_meta["reason"] = bench_meta.get("reason") or (
                "XU100/BIST100 series not obtained from cache or analysis-only yfinance; "
                "no impute; ALPHA = NOT COMPUTABLE"
            )

    payload = run_etap20(rows, frames, meta, benchmark=bench, benchmark_meta=bench_meta)
    compact = compact_payload(payload)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    out_path.write_text(text, encoding="utf-8")
    # Determinism: second in-memory compact of the same payload.
    again = json.dumps(compact_payload(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if again != text:
        print("FAIL — non-deterministic compact payload", file=sys.stderr)
        return 1
    klass = payload["classification"]
    print(json.dumps({
        "purpose": PURPOSE,
        "eval_checksum_ok": payload["eval_grid"]["checksum_ok"],
        "sqz_5d_n": payload["agents"]["SQZ"]["horizons"]["5"]["stats"]["n"],
        "sqz_5d_mean": payload["agents"]["SQZ"]["horizons"]["5"]["stats"]["mean"],
        "alpha": payload["alpha"]["computable"],
        "class": klass["class"],
        "recommendation": klass["recommendation"],
        "out": str(out_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
