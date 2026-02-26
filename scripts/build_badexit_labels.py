#!/usr/bin/env python3
# scripts/build_badexit_labels.py
from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import re
from pathlib import Path
import pandas as pd

from scripts.feature_spec import get_feature_cols


DATA_DIR = Path("data")
SIGNALS_DIR = DATA_DIR / "signals"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

FEAT_PARQ = FEAT_DIR / "features_model.parquet"
FEAT_CSV = FEAT_DIR / "features_model.csv"


# ----------------------------
# utils
# ----------------------------
def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _is_badexit_reason(reason: str) -> int:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return 1
    if r.startswith("GRACE_END_EXIT"):
        return 1
    return 0


def _parse_trades_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "BadExit"])

    need = {"EntryDate", "Tickers", "Reason"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"trades file missing cols={miss}: {path}")

    df = df.copy()
    df["EntryDate"] = _norm_date(df["EntryDate"])
    df["Reason"] = df["Reason"].astype(str)
    df["BadExit"] = df["Reason"].apply(_is_badexit_reason).astype(int)

    rows = []
    for _, r in df.iterrows():
        d = r["EntryDate"]
        if pd.isna(d):
            continue
        tickers = [t.strip().upper() for t in str(r["Tickers"]).split(",") if t.strip()]
        for t in tickers:
            rows.append({"Date": d, "Ticker": t, "BadExit": int(r["BadExit"])})

    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "BadExit"])

    out = pd.DataFrame(rows)
    out["Date"] = _norm_date(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()

    out = (
        out.dropna(subset=["Date", "Ticker"])
        .groupby(["Date", "Ticker"], as_index=False)["BadExit"]
        .max()
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return out


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build BadExit labels (tag-safe).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--signals-dir", type=str, default=str(SIGNALS_DIR))
    ap.add_argument("--also-read-csv", action="store_true")

    args = ap.parse_args()

    pt100 = int(round(args.profit_target * 100))
    sl100 = int(round(abs(args.stop_level) * 100))
    EX_TAG = f"pt{pt100}_h{args.max_days}_sl{sl100}_ex{args.max_extend_days}"

    out_parq = LABEL_DIR / f"labels_badexit_{EX_TAG}.parquet"
    out_csv = LABEL_DIR / f"labels_badexit_{EX_TAG}.csv"

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- SSOT 18개 강제
    feature_cols = get_feature_cols(sector_enabled=True)
    if len(feature_cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(feature_cols)}")

    feats = read_table(FEAT_PARQ, FEAT_CSV).copy()
    feats["Date"] = _norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    feats = (
        feats.dropna(subset=["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"features_model missing SSOT cols: {missing}")

    # ---- 정확히 해당 EX_TAG trades만 ingest
    signals_dir = Path(args.signals_dir)
    if not signals_dir.exists():
        raise FileNotFoundError(f"signals_dir not found: {signals_dir}")

    pattern = f"sim_engine_trades_{EX_TAG}*.parquet"
    paths = sorted(signals_dir.glob(pattern))

    if args.also_read_csv:
        pattern_csv = f"sim_engine_trades_{EX_TAG}*.csv"
        paths += sorted(signals_dir.glob(pattern_csv))

    if not paths:
        raise FileNotFoundError(f"No trades files found for tag={EX_TAG} in {signals_dir}")

    parts = []
    for p in paths:
        try:
            part = _parse_trades_file(p)
            if not part.empty:
                parts.append(part)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if not parts:
        raise RuntimeError("No BadExit labels produced.")

    labels = (
        pd.concat(parts, ignore_index=True)
        .groupby(["Date", "Ticker"], as_index=False)["BadExit"]
        .max()
    )

    merged = feats[["Date", "Ticker"] + feature_cols].merge(
        labels,
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )

    merged.to_parquet(out_parq, index=False)
    merged.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(merged)}")
    print(f"[INFO] BadExit rate={merged['BadExit'].mean():.4f}")
    print(f"[INFO] EX_TAG={EX_TAG}")


if __name__ == "__main__":
    main()