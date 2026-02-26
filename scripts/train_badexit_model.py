#!/usr/bin/env python3
# scripts/train_badexit_model.py
from __future__ import annotations

# ✅ FIX: "python scripts/xxx.py"로 실행될 때도 scripts.* import가 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from scripts.feature_spec import get_feature_cols


DATA_DIR = Path("data")
LABEL_DIR = DATA_DIR / "labels"
META_DIR = DATA_DIR / "meta"
APP_DIR = Path("app")

IN_PARQ = LABEL_DIR / "labels_badexit.parquet"  # legacy default
IN_CSV = LABEL_DIR / "labels_badexit.csv"        # legacy default


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train BadExit (p_badexit) model from labels_badexit(_{tag}).")

    # ✅ NEW: tag convenience (recommended)
    # If --tag is set and you keep defaults, we will use tagged dataset/artifacts.
    ap.add_argument("--tag", type=str, default="", help="e.g. pt10_h40_sl10_ex20")
    ap.add_argument("--in-parq", type=str, default=str(IN_PARQ))
    ap.add_argument("--in-csv", type=str, default=str(IN_CSV))

    ap.add_argument("--out-model", type=str, default="app/badexit_model.pkl")
    ap.add_argument("--out-scaler", type=str, default="app/badexit_scaler.pkl")
    ap.add_argument("--out-report", type=str, default="data/meta/train_badexit_report.json")

    ap.add_argument("--valid-frac", type=float, default=0.20, help="time-based split by date (last X frac = valid)")
    ap.add_argument("--min-rows", type=int, default=500, help="fail fast if too small dataset")
    args = ap.parse_args()

    tag = (args.tag or "").strip()

    # If tag is given and inputs are untouched defaults -> switch to tagged labels
    if tag and (args.in_parq == str(IN_PARQ)) and (args.in_csv == str(IN_CSV)):
        args.in_parq = str(LABEL_DIR / f"labels_badexit_{tag}.parquet")
        args.in_csv = str(LABEL_DIR / f"labels_badexit_{tag}.csv")

    # If tag is given and outputs are untouched defaults -> write tagged artifacts
    if tag and (args.out_model == "app/badexit_model.pkl"):
        args.out_model = f"app/badexit_model_{tag}.pkl"
    if tag and (args.out_scaler == "app/badexit_scaler.pkl"):
        args.out_scaler = f"app/badexit_scaler_{tag}.pkl"
    if tag and (args.out_report == "data/meta/train_badexit_report.json"):
        args.out_report = f"data/meta/train_badexit_report_{tag}.json"

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # ✅ 18개 SSOT 강제(섹터 포함)
    feature_cols = get_feature_cols(sector_enabled=True)
    if len(feature_cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(feature_cols)}: {feature_cols}")

    df = read_table(Path(args.in_parq), Path(args.in_csv)).copy()
    if df.empty:
        raise RuntimeError("labels_badexit dataset is empty")

    if "Date" not in df.columns or "Ticker" not in df.columns or "BadExit" not in df.columns:
        raise ValueError("labels_badexit must include Date, Ticker, BadExit")

    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["BadExit"] = pd.to_numeric(df["BadExit"], errors="coerce").fillna(0).astype(int)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"labels_badexit missing SSOT feature cols: {missing}")

    df = (
        df.dropna(subset=["Date", "Ticker", "BadExit"])
        .sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    if len(df) < int(args.min_rows):
        raise RuntimeError(f"Too few rows for training: {len(df)} < {int(args.min_rows)}")

    # ---- time-based split (last valid-frac dates)
    uniq_dates = np.array(sorted(df["Date"].dropna().unique()))
    if len(uniq_dates) < 10:
        raise RuntimeError(f"Too few unique dates for time split: {len(uniq_dates)}")

    cut_idx = int(max(1, np.floor(len(uniq_dates) * (1.0 - float(args.valid_frac)))))
    cut_idx = min(cut_idx, len(uniq_dates) - 1)
    cut_date = pd.Timestamp(uniq_dates[cut_idx])

    train_df = df.loc[df["Date"] < cut_date].copy()
    valid_df = df.loc[df["Date"] >= cut_date].copy()

    if train_df.empty or valid_df.empty:
        raise RuntimeError(f"Bad time split: train={len(train_df)} valid={len(valid_df)} cut_date={cut_date.date()}")

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_train = train_df["BadExit"].to_numpy(dtype=int)

    X_valid = valid_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_valid = valid_df["BadExit"].to_numpy(dtype=int)

    # ---- model
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_valid)

    # class_weight balanced: BadExit(1) 희소할 가능성 큼
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(Xtr, y_train)

    # ---- metrics
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]

    # threshold=0.5 기본 (나중에 gate에서 max로 쓰면 threshold 필요없음)
    yhat_tr = (p_tr >= 0.5).astype(int)
    yhat_va = (p_va >= 0.5).astype(int)

    report = {
        "tag": tag,
        "built_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "badexit_rate_total": float(df["BadExit"].mean()),
            "badexit_rate_train": float(train_df["BadExit"].mean()),
            "badexit_rate_valid": float(valid_df["BadExit"].mean()),
            "date_min": str(df["Date"].min().date()),
            "date_max": str(df["Date"].max().date()),
            "valid_cut_date": str(cut_date.date()),
        },
        "model": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "features": feature_cols,
        },
        "metrics_train": {
            "auc": _safe_auc(y_train, p_tr),
            "accuracy": float(accuracy_score(y_train, yhat_tr)),
            "precision": float(precision_score(y_train, yhat_tr, zero_division=0)),
            "recall": float(recall_score(y_train, yhat_tr, zero_division=0)),
            "f1": float(f1_score(y_train, yhat_tr, zero_division=0)),
        },
        "metrics_valid": {
            "auc": _safe_auc(y_valid, p_va),
            "accuracy": float(accuracy_score(y_valid, yhat_va)),
            "precision": float(precision_score(y_valid, yhat_va, zero_division=0)),
            "recall": float(recall_score(y_valid, yhat_va, zero_division=0)),
            "f1": float(f1_score(y_valid, yhat_va, zero_division=0)),
        },
    }

    out_model = Path(args.out_model)
    out_scaler = Path(args.out_scaler)
    out_report = Path(args.out_report)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_scaler.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, out_model)
    joblib.dump(scaler, out_scaler)

    # ✅ Compatibility: also refresh untagged defaults used by older code
    try:
        joblib.dump(clf, Path("app/badexit_model.pkl"))
        joblib.dump(scaler, Path("app/badexit_scaler.pkl"))
        print("[INFO] also wrote compatibility copies: app/badexit_model.pkl / app/badexit_scaler.pkl")
    except Exception as e:
        print(f"[WARN] failed to write compatibility copies: {e}")

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote model : {out_model}")
    print(f"[DONE] wrote scaler: {out_scaler}")
    print(f"[DONE] wrote report: {out_report}")
    print(
        f"[INFO] valid AUC={report['metrics_valid']['auc']:.4f} "
        f"acc={report['metrics_valid']['accuracy']:.4f} "
        f"prec={report['metrics_valid']['precision']:.4f} "
        f"rec={report['metrics_valid']['recall']:.4f}"
    )


if __name__ == "__main__":
    main()