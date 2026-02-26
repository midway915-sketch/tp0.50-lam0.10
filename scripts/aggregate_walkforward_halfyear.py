#!/usr/bin/env python3
# scripts/aggregate_walkforward_halfyear.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


# ----------------------------
# helpers
# ----------------------------
def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _read_any(path_parq: Path, path_csv: Path) -> pd.DataFrame:
    if path_parq.exists():
        return pd.read_parquet(path_parq)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    raise FileNotFoundError(f"missing: {path_parq} (or {path_csv})")


def _try_read_any(path_parq: Path, path_csv: Path) -> Optional[pd.DataFrame]:
    try:
        return _read_any(path_parq, path_csv)
    except FileNotFoundError:
        return None


def _parse_float_token(tok: str) -> float:
    """
    "0p10" -> 0.10, "1p0" -> 1.0, "50" -> 50.0
    """
    s = str(tok).strip()
    if not s:
        return float("nan")
    s = s.replace("p", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _calc_cagr(start_equity: float, end_equity: float, days: float) -> float:
    if not np.isfinite(start_equity) or not np.isfinite(end_equity) or start_equity <= 0 or end_equity <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float((end_equity / start_equity) ** (1.0 / years) - 1.0)


def _clean_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()

    c = curve.copy()
    if "Date" not in c.columns or "Equity" not in c.columns:
        return pd.DataFrame()

    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")

    # InCycle / InPosition -> InCycle(0/1)
    if "InCycle" in c.columns:
        inc = pd.to_numeric(c["InCycle"], errors="coerce").fillna(0).astype(int)
    elif "InPosition" in c.columns:
        inc = pd.to_numeric(c["InPosition"], errors="coerce").fillna(0).astype(int)
    else:
        inc = pd.Series([0] * len(c), index=c.index, dtype=int)
    c["InCycle"] = (inc > 0).astype(int)

    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    c = c.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return c


def _max_drawdown_stats(equity: pd.Series) -> Tuple[float, int, int]:
    """
    returns: (max_dd, max_underwater_days, max_recovery_days)
      - max_dd: positive number, e.g. 0.25 means -25%
    """
    eq = pd.to_numeric(equity, errors="coerce").astype(float).values
    if len(eq) < 2 or not np.isfinite(eq).all():
        return (float("nan"), 0, 0)

    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)
    max_dd = float(np.nanmax(dd))

    # underwater days: consecutive dd>0
    uw = dd > 1e-12
    max_uw = 0
    cur = 0
    for v in uw:
        if v:
            cur += 1
            max_uw = max(max_uw, cur)
        else:
            cur = 0

    # recovery days: from max-dd trough until new high
    i_trough = int(np.nanargmax(dd))
    trough_peak = float(peak[i_trough])
    rec = 0
    for j in range(i_trough, len(eq)):
        if eq[j] >= trough_peak - 1e-12:
            break
        rec += 1

    return (max_dd, int(max_uw), int(rec))


def _daily_risk_stats(curve: pd.DataFrame) -> Tuple[float, float, float]:
    """
    returns: (daily_vol, sharpe0, sortino0) using daily returns of Equity
    """
    if curve is None or curve.empty or "Equity" not in curve.columns:
        return (float("nan"), float("nan"), float("nan"))

    eq = pd.to_numeric(curve["Equity"], errors="coerce")
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 5:
        return (float("nan"), float("nan"), float("nan"))

    vol = float(r.std(ddof=1))
    mean = float(r.mean())

    if vol > 0:
        sharpe = float(mean / vol)
    else:
        sharpe = float("nan")

    downside = r[r < 0]
    dvol = float(downside.std(ddof=1)) if len(downside) >= 3 else float("nan")
    sortino = float(mean / dvol) if np.isfinite(dvol) and dvol > 0 else float("nan")

    return (vol, sharpe, sortino)


def _is_badexit_reason(reason: str) -> int:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return 1
    if r.startswith("GRACE_END_EXIT"):
        return 1
    return 0


def _badexit_reason_bucket(reason: str) -> str:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return "RevalFail"
    if r.startswith("GRACE_END_EXIT"):
        return "GraceEnd"
    return "Other"


def _split_tickers(cell: Any) -> List[str]:
    ticks = [t.strip().upper() for t in str(cell or "").split(",") if t.strip()]
    return ticks


def _parse_cap_from_suffix(suffix: str) -> Tuple[str, str]:
    s = str(suffix or "")
    m = re.search(r"(.*)_cap(none|h2|total)$", s)
    if not m:
        return (s, "")
    return (m.group(1), m.group(2))


def _parse_suffix_params(suffix: str) -> Dict[str, Any]:
    """
    Try extracting:
      ps_min, tail_threshold, utility_quantile, lambda_tail, topk, badexit_max
      trail_stop, tp1_frac  (optional)
    """
    s = str(suffix or "")

    def m_float(pattern: str) -> float:
        m = re.search(pattern, s)
        return _parse_float_token(m.group(1)) if m else float("nan")

    def m_int(pattern: str) -> float:
        m = re.search(pattern, s)
        return float(int(m.group(1))) if m else float("nan")

    out: Dict[str, Any] = {}
    out["ps_min"] = m_float(r"_ps(\d+p\d+)")
    out["tail_threshold"] = m_float(r"_t(\d+p\d+)")
    out["utility_quantile"] = m_float(r"_q(\d+p\d+)")
    out["lambda_tail"] = m_float(r"_lam(\d+p\d+)")
    out["badexit_max"] = m_float(r"_be(\d+p\d+)")
    out["topk"] = m_int(r"_k(\d+)")

    # optional but useful
    out["trail_stop"] = m_float(r"_tr(\d+p\d+)")
    # TP1 fraction could be in suffix as tp50 meaning 0.50 OR tp0p5 — handle both
    m_tp = re.search(r"_tp(\d+)$", s)
    if m_tp:
        out["tp1_frac"] = float(int(m_tp.group(1))) / 100.0
    else:
        out["tp1_frac"] = m_float(r"_tp(\d+p\d+)")

    return out


def _discover_files(period_dir: Path) -> List[Dict[str, Any]]:
    """
    For each config within a period dir, find matching curve and trades file.

    Expected patterns:
      sim_engine_trades_<tag>_gate_<suffix>.parquet/csv
      sim_engine_curve_<tag>_gate_<suffix>.parquet/csv
    """
    rows: List[Dict[str, Any]] = []

    trade_files = sorted(list(period_dir.glob("sim_engine_trades_*_gate_*.parquet")) +
                         list(period_dir.glob("sim_engine_trades_*_gate_*.csv")))
    if not trade_files:
        return rows

    rx = re.compile(r"sim_engine_trades_(?P<tag>.+?)_gate_(?P<suffix>.+?)\.(csv|parquet)$", re.IGNORECASE)

    for tf in trade_files:
        m = rx.match(tf.name)
        if not m:
            continue
        tag = m.group("tag")
        suffix = m.group("suffix")

        curve_parq = period_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"
        curve_csv = period_dir / f"sim_engine_curve_{tag}_gate_{suffix}.csv"

        rows.append({
            "tag": tag,
            "suffix": suffix,
            "trades_file": tf,
            "curve_parq": curve_parq,
            "curve_csv": curve_csv,
        })

    return rows


def _load_prices_optional(raw_prices_dir: Path) -> Optional[pd.DataFrame]:
    parq = raw_prices_dir / "prices.parquet"
    csv = raw_prices_dir / "prices.csv"
    df = _try_read_any(parq, csv)
    if df is None or df.empty:
        return None

    need = {"Date", "Ticker"}
    if not need.issubset(set(df.columns)) or "Close" not in df.columns:
        return None

    df = df.copy()
    df["Date"] = _safe_to_dt(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def _qqq_stats(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, float]:
    if prices is None or prices.empty:
        return (float("nan"), float("nan"))

    q = prices.loc[prices["Ticker"] == "QQQ"].copy()
    if q.empty:
        return (float("nan"), float("nan"))

    q = q.sort_values("Date")

    q_start = q.loc[q["Date"] >= start]
    q_end = q.loc[q["Date"] <= end]
    if q_start.empty or q_end.empty:
        return (float("nan"), float("nan"))

    p0 = float(q_start["Close"].iloc[0])
    p1 = float(q_end["Close"].iloc[-1])
    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1) or p1 <= 0:
        return (float("nan"), float("nan"))

    mult = p1 / p0
    days = float((end - start).days)
    cagr = _calc_cagr(1.0, mult, days)
    return (float(mult), float(cagr))


# ----------------------------
# per-config metrics
# ----------------------------
def _compute_metrics_one(
    period: str,
    period_dir: Path,
    tag: str,
    suffix: str,
    trades_path: Path,
    curve_path: Optional[Path],
    prices: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    base_suffix, cap = _parse_cap_from_suffix(suffix)
    params = _parse_suffix_params(suffix)

    out["period"] = period
    out["tag"] = tag
    out["suffix"] = suffix
    out["base_suffix"] = base_suffix
    out["cap"] = cap

    # required summary columns (even if NaN)
    out["ps_min"] = params.get("ps_min", float("nan"))
    out["tail_threshold"] = params.get("tail_threshold", float("nan"))
    out["utility_quantile"] = params.get("utility_quantile", float("nan"))
    out["lambda_tail"] = params.get("lambda_tail", float("nan"))
    out["topk"] = params.get("topk", float("nan"))
    out["badexit_max"] = params.get("badexit_max", float("nan"))

    out["curve_file"] = str(curve_path) if curve_path else ""
    out["trades_file"] = str(trades_path)

    # ---- trades
    trades = pd.read_parquet(trades_path) if trades_path.suffix.lower() == ".parquet" else pd.read_csv(trades_path)
    if trades is None or trades.empty:
        # trades metrics
        out.update({
            "TradeCount": 0,
            "TradesPerYear": float("nan"),
            "BadExitRate_Row": float("nan"),
            "BadExitRate_Ticker": float("nan"),
            "BadExitReasonShare_RevalFail": float("nan"),
            "BadExitReasonShare_GraceEnd": float("nan"),
            "BadExitReturnMean": float("nan"),
            "NonBadExitReturnMean": float("nan"),
            "BadExitReturnDiff": float("nan"),
            "HoldDays_Mean": float("nan"),
            "HoldDays_Median": float("nan"),
            "HoldDays_P90": float("nan"),
        })
        warmup_end = pd.NaT
    else:
        # date parse
        if "EntryDate" in trades.columns:
            trades["EntryDate"] = _safe_to_dt(trades["EntryDate"])
        if "ExitDate" in trades.columns:
            trades["ExitDate"] = _safe_to_dt(trades["ExitDate"])

        # TradeCount / TradesPerYear
        out["TradeCount"] = int(len(trades))

        # infer time span from EntryDate
        entry = trades["EntryDate"].dropna() if "EntryDate" in trades.columns else pd.Series([], dtype="datetime64[ns]")
        if not entry.empty:
            warmup_end = entry.min()
            t0 = entry.min()
            t1 = entry.max()
            days_span = float((t1 - t0).days) if pd.notna(t0) and pd.notna(t1) else float("nan")
            years_span = days_span / 365.0 if np.isfinite(days_span) and days_span > 0 else float("nan")
            out["TradesPerYear"] = float(out["TradeCount"] / years_span) if np.isfinite(years_span) and years_span > 0 else float("nan")
        else:
            warmup_end = pd.NaT
            out["TradesPerYear"] = float("nan")

        # BadExit row rate + reason shares
        if "Reason" in trades.columns:
            reason = trades["Reason"].astype(str)
            bad_row = reason.apply(_is_badexit_reason).astype(int)
            out["BadExitRate_Row"] = float(bad_row.mean()) if len(bad_row) else float("nan")

            buckets = reason.apply(_badexit_reason_bucket)
            be = bad_row == 1
            if be.any():
                be_b = buckets[be]
                out["BadExitReasonShare_RevalFail"] = float((be_b == "RevalFail").mean())
                out["BadExitReasonShare_GraceEnd"] = float((be_b == "GraceEnd").mean())
            else:
                out["BadExitReasonShare_RevalFail"] = 0.0
                out["BadExitReasonShare_GraceEnd"] = 0.0
        else:
            out["BadExitRate_Row"] = float("nan")
            out["BadExitReasonShare_RevalFail"] = float("nan")
            out["BadExitReasonShare_GraceEnd"] = float("nan")
            bad_row = pd.Series([0] * len(trades), dtype=int)

        # BadExit ticker-rate: split Tickres
        if "Tickers" in trades.columns and "Reason" in trades.columns and len(trades) > 0:
            rows = []
            for _, r in trades.iterrows():
                d = r.get("EntryDate", pd.NaT)
                if pd.isna(d):
                    continue
                ticks = _split_tickers(r.get("Tickers", ""))
                if not ticks:
                    continue
                be = _is_badexit_reason(r.get("Reason", ""))
                for t in ticks:
                    rows.append((d, t, be))

            if rows:
                tmp = pd.DataFrame(rows, columns=["Date", "Ticker", "BadExit"])
                out["BadExitRate_Ticker"] = float(tmp["BadExit"].mean())
            else:
                out["BadExitRate_Ticker"] = float("nan")
        else:
            out["BadExitRate_Ticker"] = float("nan")

        # BadExit vs NonBadExit return difference (if Return exists)
        ret_col = None
        for cand in ["Return", "ret", "PnL_pct", "pnl_pct", "CycleReturn", "cycle_return"]:
            if cand in trades.columns:
                ret_col = cand
                break

        if ret_col and "Reason" in trades.columns:
            rr = pd.to_numeric(trades[ret_col], errors="coerce")
            be = trades["Reason"].astype(str).apply(_is_badexit_reason).astype(int)
            be_rr = rr[be == 1].dropna()
            nb_rr = rr[be == 0].dropna()
            out["BadExitReturnMean"] = float(be_rr.mean()) if len(be_rr) else float("nan")
            out["NonBadExitReturnMean"] = float(nb_rr.mean()) if len(nb_rr) else float("nan")
            if np.isfinite(out["BadExitReturnMean"]) and np.isfinite(out["NonBadExitReturnMean"]):
                out["BadExitReturnDiff"] = float(out["BadExitReturnMean"] - out["NonBadExitReturnMean"])
            else:
                out["BadExitReturnDiff"] = float("nan")
        else:
            out["BadExitReturnMean"] = float("nan")
            out["NonBadExitReturnMean"] = float("nan")
            out["BadExitReturnDiff"] = float("nan")

        # holding days
        hold = None
        if "HoldingDays" in trades.columns:
            hold = pd.to_numeric(trades["HoldingDays"], errors="coerce")
        elif "EntryDate" in trades.columns and "ExitDate" in trades.columns:
            hold = (trades["ExitDate"] - trades["EntryDate"]).dt.days
        if hold is not None:
            hold = pd.to_numeric(hold, errors="coerce").dropna()
            out["HoldDays_Mean"] = float(hold.mean()) if len(hold) else float("nan")
            out["HoldDays_Median"] = float(hold.median()) if len(hold) else float("nan")
            out["HoldDays_P90"] = float(np.nanpercentile(hold.values, 90)) if len(hold) else float("nan")
        else:
            out["HoldDays_Mean"] = float("nan")
            out["HoldDays_Median"] = float("nan")
            out["HoldDays_P90"] = float("nan")

    # ---- curve / after-warmup
    curve = pd.DataFrame()
    if curve_path and curve_path.exists():
        curve_raw = pd.read_parquet(curve_path) if curve_path.suffix.lower() == ".parquet" else pd.read_csv(curve_path)
        curve = _clean_curve(curve_raw)

    if curve.empty:
        out.update({
            "WarmupEndDate": "",
            "Days_AfterWarmup": float("nan"),
            "SeedMultiple_AfterWarmup": float("nan"),
            "CAGR_AfterWarmup": float("nan"),
            "QQQ_SeedMultiple_SamePeriod": float("nan"),
            "QQQ_CAGR_SamePeriod": float("nan"),
            "ExcessSeedMultiple_AfterWarmup": float("nan"),
            "ExcessCAGR_AfterWarmup": float("nan"),
            "MaxDD_AfterWarmup": float("nan"),
            "MaxUnderwaterDays_AfterWarmup": float("nan"),
            "MaxDDRecoveryDays_AfterWarmup": float("nan"),
            "DailyVol": float("nan"),
            "Sharpe0": float("nan"),
            "Sortino0": float("nan"),
            "ActiveDaysAfterWarmup": float("nan"),
            "IdleDaysAfterWarmup": float("nan"),
            "IdlePctAfterWarmup": float("nan"),
        })
        return out

    if pd.isna(warmup_end):
        warmup_end = pd.Timestamp(curve["Date"].iloc[0])

    cur2 = curve.loc[curve["Date"] >= warmup_end].copy()
    if cur2.empty:
        cur2 = curve.copy()

    out["WarmupEndDate"] = str(pd.Timestamp(warmup_end).date()) if pd.notna(warmup_end) else ""

    start_date = pd.Timestamp(cur2["Date"].iloc[0])
    end_date = pd.Timestamp(cur2["Date"].iloc[-1])
    days = float((end_date - start_date).days)
    out["Days_AfterWarmup"] = days

    start_eq = float(cur2["Equity"].iloc[0])
    end_eq = float(cur2["Equity"].iloc[-1])
    seed_mult = float(end_eq / start_eq) if (np.isfinite(start_eq) and start_eq > 0 and np.isfinite(end_eq) and end_eq > 0) else float("nan")
    out["SeedMultiple_AfterWarmup"] = seed_mult
    out["CAGR_AfterWarmup"] = _calc_cagr(start_eq, end_eq, days)

    # idle/active
    total_days = int(cur2["Date"].nunique())
    active_days = int((cur2["InCycle"] > 0).sum())
    idle_days = int(max(0, total_days - active_days))
    idle_pct = float(idle_days / total_days) if total_days > 0 else float("nan")
    out["ActiveDaysAfterWarmup"] = active_days
    out["IdleDaysAfterWarmup"] = idle_days
    out["IdlePctAfterWarmup"] = idle_pct

    # drawdown / underwater / recovery
    max_dd, max_uw, max_rec = _max_drawdown_stats(cur2["Equity"])
    out["MaxDD_AfterWarmup"] = max_dd
    out["MaxUnderwaterDays_AfterWarmup"] = float(max_uw)
    out["MaxDDRecoveryDays_AfterWarmup"] = float(max_rec)

    # daily vol/sharpe/sortino
    vol, sharpe, sortino = _daily_risk_stats(cur2)
    out["DailyVol"] = vol
    out["Sharpe0"] = sharpe
    out["Sortino0"] = sortino

    # QQQ benchmark (optional)
    qqq_mult, qqq_cagr = _qqq_stats(prices, start_date, end_date) if prices is not None else (float("nan"), float("nan"))
    out["QQQ_SeedMultiple_SamePeriod"] = qqq_mult
    out["QQQ_CAGR_SamePeriod"] = qqq_cagr

    out["ExcessSeedMultiple_AfterWarmup"] = float(seed_mult / qqq_mult) if np.isfinite(seed_mult) and np.isfinite(qqq_mult) and qqq_mult > 0 else float("nan")
    out["ExcessCAGR_AfterWarmup"] = float(out["CAGR_AfterWarmup"] - qqq_cagr) if np.isfinite(out["CAGR_AfterWarmup"]) and np.isfinite(qqq_cagr) else float("nan")

    return out


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="root dir containing period subdirs (e.g. data/signals/walkforward2)")
    ap.add_argument("--out", required=True, type=str, help="output summary csv path")
    ap.add_argument("--raw-prices-dir", default="data/raw", type=str, help="optional dir containing prices.parquet/csv (QQQ benchmark)")

    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    # prices are OPTIONAL now (this is the fix)
    prices = _load_prices_optional(Path(args.raw_prices_dir))

    # discover periods
    period_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not period_dirs:
        raise FileNotFoundError(f"no period dirs under: {root}")

    rows: List[Dict[str, Any]] = []

    for pd_dir in period_dirs:
        period = pd_dir.name
        items = _discover_files(pd_dir)
        if not items:
            continue

        for it in items:
            tag = it["tag"]
            suffix = it["suffix"]
            trades_file: Path = it["trades_file"]

            # curve file preference: parquet -> csv -> none
            curve_path = it["curve_parq"] if it["curve_parq"].exists() else (it["curve_csv"] if it["curve_csv"].exists() else None)

            try:
                row = _compute_metrics_one(
                    period=period,
                    period_dir=pd_dir,
                    tag=tag,
                    suffix=suffix,
                    trades_path=trades_file,
                    curve_path=curve_path,
                    prices=prices,
                )
                rows.append(row)
            except Exception as e:
                # keep summary generation robust: one bad file shouldn't kill the run
                rows.append({
                    "period": period,
                    "tag": tag,
                    "suffix": suffix,
                    "cap": _parse_cap_from_suffix(suffix)[1],
                    "curve_file": str(curve_path) if curve_path else "",
                    "trades_file": str(trades_file),
                    "error": str(e),
                    # required cols
                    "ps_min": float("nan"),
                    "tail_threshold": float("nan"),
                    "utility_quantile": float("nan"),
                    "lambda_tail": float("nan"),
                    "topk": float("nan"),
                    "badexit_max": float("nan"),
                })

    if not rows:
        raise RuntimeError("no summary rows produced (no matching trades files)")

    out = pd.DataFrame(rows)

    # enforce required columns always present
    required = [
        "period", "curve_file", "suffix", "cap",
        "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max",
    ]
    for c in required:
        if c not in out.columns:
            out[c] = np.nan if c not in ["period", "curve_file", "suffix", "cap"] else ""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote summary: {out_path} rows={len(out)}")
    if prices is None:
        print("[INFO] raw prices missing -> QQQ benchmark columns will be NaN (this is ok)")


if __name__ == "__main__":
    main()