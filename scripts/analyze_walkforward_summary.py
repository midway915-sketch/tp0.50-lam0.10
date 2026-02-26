#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _calc_cagr_from_mult_days(mult: float, days: float) -> float:
    if not np.isfinite(mult) or mult <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float(mult ** (1.0 / years) - 1.0)


def _wavg(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    m = x.notna() & (w > 0)
    if not m.any():
        return float("nan")
    return float((x[m] * w[m]).sum() / w[m].sum())


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_curve_path(curve_file: str, summary_path: Path) -> Path:
    """
    curve_file may be:
      - absolute
      - relative to cwd
      - relative to summary file directory
    """
    p = Path(str(curve_file)).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    # try relative to summary file folder
    p2 = (summary_path.parent / p).resolve()
    if p2.exists():
        return p2
    return p  # return original (will fail later)


def _clean_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "Date" not in df.columns or "Equity" not in df.columns:
        return pd.DataFrame()

    c = df.copy()
    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")

    # InCycle / InPosition fallback
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


def _fill_start_end_mult_days_from_curve(df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    """
    Ensure per-row:
      - StartDate_AfterWarmup
      - EndDate_AfterWarmup
      - Days_AfterWarmup (recomputed from curve)
      - SeedMultiple_AfterWarmup (recomputed from curve)
      - CAGR_AfterWarmup (recomputed from mult/days)
    Uses WarmupEndDate + curve_file.
    """
    if df.empty:
        return df

    for c in ["WarmupEndDate", "curve_file"]:
        if c not in df.columns:
            df[c] = ""

    # create columns if missing
    for c in ["StartDate_AfterWarmup", "EndDate_AfterWarmup"]:
        if c not in df.columns:
            df[c] = ""

    # normalize warmup end dt
    warm = _safe_to_dt(df["WarmupEndDate"])
    df["_WarmupEndDT"] = warm

    # numeric cols
    df["Days_AfterWarmup"] = pd.to_numeric(df.get("Days_AfterWarmup", np.nan), errors="coerce")
    df["SeedMultiple_AfterWarmup"] = pd.to_numeric(df.get("SeedMultiple_AfterWarmup", np.nan), errors="coerce")

    # cache curves to avoid repeated load
    curve_cache: dict[str, pd.DataFrame] = {}

    starts = []
    ends = []
    days2 = []
    mult2 = []
    cagr2 = []

    for _, r in df.iterrows():
        curve_file = str(r.get("curve_file", "") or "").strip()
        wdt = r.get("_WarmupEndDT", pd.NaT)

        # if no curve or no warmup end, keep original numbers
        if not curve_file or pd.isna(wdt):
            starts.append(r.get("StartDate_AfterWarmup", ""))
            ends.append(r.get("EndDate_AfterWarmup", ""))
            days2.append(r.get("Days_AfterWarmup", np.nan))
            mult2.append(r.get("SeedMultiple_AfterWarmup", np.nan))
            cagr2.append(r.get("CAGR_AfterWarmup", np.nan))
            continue

        # load curve
        key = curve_file
        if key in curve_cache:
            curve = curve_cache[key]
        else:
            p = _resolve_curve_path(curve_file, summary_path)
            if not p.exists():
                curve_cache[key] = pd.DataFrame()
                curve = curve_cache[key]
            else:
                try:
                    curve_raw = _read_any(p)
                    curve = _clean_curve(curve_raw)
                except Exception:
                    curve = pd.DataFrame()
                curve_cache[key] = curve

        if curve is None or curve.empty:
            starts.append(r.get("StartDate_AfterWarmup", ""))
            ends.append(r.get("EndDate_AfterWarmup", ""))
            days2.append(r.get("Days_AfterWarmup", np.nan))
            mult2.append(r.get("SeedMultiple_AfterWarmup", np.nan))
            cagr2.append(r.get("CAGR_AfterWarmup", np.nan))
            continue

        cur2 = curve.loc[curve["Date"] >= pd.Timestamp(wdt)].copy()
        if cur2.empty:
            starts.append(r.get("StartDate_AfterWarmup", ""))
            ends.append(r.get("EndDate_AfterWarmup", ""))
            days2.append(r.get("Days_AfterWarmup", np.nan))
            mult2.append(r.get("SeedMultiple_AfterWarmup", np.nan))
            cagr2.append(r.get("CAGR_AfterWarmup", np.nan))
            continue

        sd = pd.Timestamp(cur2["Date"].iloc[0])
        ed = pd.Timestamp(cur2["Date"].iloc[-1])
        d = float((ed - sd).days)

        se0 = float(cur2["Equity"].iloc[0])
        se1 = float(cur2["Equity"].iloc[-1])
        m = float(se1 / se0) if np.isfinite(se0) and se0 > 0 and np.isfinite(se1) and se1 > 0 else float("nan")
        cg = _calc_cagr_from_mult_days(m, d)

        starts.append(str(sd.date()))
        ends.append(str(ed.date()))
        days2.append(d)
        mult2.append(m)
        cagr2.append(cg)

    df["StartDate_AfterWarmup"] = starts
    df["EndDate_AfterWarmup"] = ends
    df["Days_AfterWarmup"] = pd.to_numeric(days2, errors="coerce")
    df["SeedMultiple_AfterWarmup"] = pd.to_numeric(mult2, errors="coerce")
    df["CAGR_AfterWarmup"] = pd.to_numeric(cagr2, errors="coerce")

    df = df.drop(columns=["_WarmupEndDT"], errors="ignore")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--group-cols", default="suffix,cap", type=str, help="comma-separated group keys")
    args = ap.parse_args()

    summ = Path(args.summary)
    if not summ.exists():
        raise FileNotFoundError(f"missing summary: {summ}")

    df = pd.read_csv(summ)
    if df.empty:
        raise RuntimeError("summary is empty")

    # required group keys
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            df[c] = ""

    # Ensure required numeric columns exist
    for c in ["suffix", "cap", "SeedMultiple_AfterWarmup", "Days_AfterWarmup"]:
        if c not in df.columns:
            df[c] = np.nan

    # If per-period start/end missing, reconstruct from curve file
    df = _fill_start_end_mult_days_from_curve(df, summ)

    # weights
    df["TradeCount"] = pd.to_numeric(df.get("TradeCount", np.nan), errors="coerce").fillna(0.0)

    g = df.groupby(group_cols, dropna=False)

    rows = []
    for key, part in g:
        part = part.copy()

        # TOTAL seed multiple = product across periods (after-warmup)
        mults = pd.to_numeric(part.get("SeedMultiple_AfterWarmup", np.nan), errors="coerce").dropna()
        seed_total = float(np.prod(mults.values)) if len(mults) > 0 else float("nan")

        # TOTAL days = sum of per-period days (walkforward slices)
        days_ser = pd.to_numeric(part.get("Days_AfterWarmup", np.nan), errors="coerce").dropna()
        days_total = float(days_ser.sum()) if len(days_ser) > 0 else float("nan")

        cagr_total = _calc_cagr_from_mult_days(seed_total, days_total)

        # Start/End total from reconstructed per-period dates
        sdt = _safe_to_dt(part.get("StartDate_AfterWarmup", pd.Series([], dtype=str)))
        edt = _safe_to_dt(part.get("EndDate_AfterWarmup", pd.Series([], dtype=str)))
        start_total = sdt.min() if sdt.notna().any() else pd.NaT
        end_total = edt.max() if edt.notna().any() else pd.NaT

        # QQQ total (optional)
        qqq_mults = pd.to_numeric(part.get("QQQ_SeedMultiple_SamePeriod", np.nan), errors="coerce").dropna()
        qqq_seed_total = float(np.prod(qqq_mults.values)) if len(qqq_mults) > 0 else float("nan")
        qqq_cagr_total = _calc_cagr_from_mult_days(qqq_seed_total, days_total) if np.isfinite(days_total) else float("nan")
        excess_cagr_total = float(cagr_total - qqq_cagr_total) if np.isfinite(cagr_total) and np.isfinite(qqq_cagr_total) else float("nan")

        # badexit weighted rates
        bad_row = _wavg(part.get("BadExitRate_Row", np.nan), part["TradeCount"])
        bad_tkr = _wavg(part.get("BadExitRate_Ticker", np.nan), part["TradeCount"])
        reval_share = _wavg(part.get("BadExitReasonShare_RevalFail", np.nan), part["TradeCount"])
        grace_share = _wavg(part.get("BadExitReasonShare_GraceEnd", np.nan), part["TradeCount"])

        # badexit return stats (weighted by TradeCount)
        bad_ret = _wavg(part.get("BadExitReturnMean", np.nan), part["TradeCount"])
        good_ret = _wavg(part.get("NonBadExitReturnMean", np.nan), part["TradeCount"])
        diff_ret = _wavg(part.get("BadExitReturnDiff", np.nan), part["TradeCount"])

        # dd style: max across periods (conservative)
        max_dd = pd.to_numeric(part.get("MaxDD_AfterWarmup", np.nan), errors="coerce")
        max_dd_total = float(np.nanmax(max_dd.values)) if max_dd.notna().any() else float("nan")

        uw = pd.to_numeric(part.get("MaxUnderwaterDays_AfterWarmup", np.nan), errors="coerce")
        uw_total = float(np.nanmax(uw.values)) if uw.notna().any() else float("nan")

        rec = pd.to_numeric(part.get("MaxDDRecoveryDays_AfterWarmup", np.nan), errors="coerce")
        rec_total = float(np.nanmax(rec.values)) if rec.notna().any() else float("nan")

        trade_total = float(part["TradeCount"].sum())

        # carry representative params: take first non-null
        def first_nonnull(col: str):
            if col not in part.columns:
                return np.nan
            s = part[col].dropna()
            return s.iloc[0] if len(s) else np.nan

        row = {}
        if isinstance(key, tuple):
            for i, c in enumerate(group_cols):
                row[c] = key[i]
        else:
            row[group_cols[0]] = key

        row.update({
            "Periods": int(part["period"].nunique()) if "period" in part.columns else int(len(part)),
            "StartDate_Total": str(start_total.date()) if pd.notna(start_total) else "",
            "EndDate_Total": str(end_total.date()) if pd.notna(end_total) else "",

            "DaysTotal_AfterWarmup": days_total,
            "SeedMultiple_Total": seed_total,
            "CAGR_Total": cagr_total,

            "QQQ_SeedMultiple_Total": qqq_seed_total,
            "QQQ_CAGR_Total": qqq_cagr_total,
            "ExcessCAGR_Total": excess_cagr_total,

            "TradeCount_Total": trade_total,

            "BadExitRate_Row_Total": bad_row,
            "BadExitRate_Ticker_Total": bad_tkr,
            "BadExitReasonShare_RevalFail_Total": reval_share,
            "BadExitReasonShare_GraceEnd_Total": grace_share,

            "BadExitReturnMean_Total": bad_ret,
            "NonBadExitReturnMean_Total": good_ret,
            "BadExitReturnDiff_Total": diff_ret,

            "MaxDD_Total": max_dd_total,
            "MaxUnderwaterDays_Total": uw_total,
            "MaxDDRecoveryDays_Total": rec_total,

            # helpful params
            "ps_min": first_nonnull("ps_min"),
            "tail_threshold": first_nonnull("tail_threshold"),
            "utility_quantile": first_nonnull("utility_quantile"),
            "lambda_tail": first_nonnull("lambda_tail"),
            "topk": first_nonnull("topk"),
            "badexit_max": first_nonnull("badexit_max"),
        })

        # simple ranking score (prioritize CAGR, penalize DD)
        if np.isfinite(row["CAGR_Total"]) and np.isfinite(row["MaxDD_Total"]):
            row["Score"] = float(row["CAGR_Total"] / (1.0 + row["MaxDD_Total"]))
        elif np.isfinite(row["CAGR_Total"]):
            row["Score"] = float(row["CAGR_Total"])
        else:
            row["Score"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)

    # sort best first
    sort_cols = [c for c in ["Score", "CAGR_Total", "SeedMultiple_Total"] if c in out.columns]
    if sort_cols:
        asc = [False] * len(sort_cols)
        out = out.sort_values(sort_cols, ascending=asc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # print a tiny debug headline (so you can see dates/cagr in logs)
    if len(out):
        best = out.iloc[0].to_dict()
        print("=" * 60)
        print("[BEST]")
        print(f"{group_cols}={tuple(best.get(c) for c in group_cols)}")
        print(f"Start={best.get('StartDate_Total')} End={best.get('EndDate_Total')} Days={best.get('DaysTotal_AfterWarmup')}")
        print(f"SeedTotal={best.get('SeedMultiple_Total')} CAGR={best.get('CAGR_Total')}")
        print("=" * 60)

    print(f"[DONE] wrote analysis: {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()