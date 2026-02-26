#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


DATA_DIR = Path("data")
APP_DIR = Path("app")
META_DIR = DATA_DIR / "meta"


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def read_labels_by_tag(tag: str) -> tuple[pd.DataFrame, str]:
    """
    Priority:
      1) labels_model_{tag}.parquet/csv
      2) labels_model.parquet/csv (fallback)
    """
    if tag:
        p = DATA_DIR / "labels" / f"labels_model_{tag}.parquet"
        c = DATA_DIR / "labels" / f"labels_model_{tag}.csv"
        if p.exists():
            return pd.read_parquet(p), str(p)
        if c.exists():
            return pd.read_csv(c), str(c)

    p = DATA_DIR / "labels" / "labels_model.parquet"
    c = DATA_DIR / "labels" / "labels_model.csv"

    if p.exists():
        return pd.read_parquet(p), str(p)
    if c.exists():
        return pd.read_csv(c), str(c)

    raise FileNotFoundError("No labels_model file found.")


# ------------------------------------------------------------
# feature resolution
# ------------------------------------------------------------
def resolve_feature_cols(args_features: str):
    override = [x.strip() for x in str(args_features).split(",") if x.strip()]
    if override:
        return override, "--features"

    cols_meta, _ = read_feature_cols_meta()
    if cols_meta:
        return cols_meta, "feature_cols.json"

    return get_feature_cols(sector_enabled=False), "feature_spec fallback"


# ------------------------------------------------------------
# date split
# ------------------------------------------------------------
def date_split(df: pd.DataFrame, date_col: str, ratio: float):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    uniq = sorted(dates.dropna().unique())
    cut = uniq[int(len(uniq) * ratio)]
    train = df[df[date_col] <= cut]
    test = df[df[date_col] > cut]
    return train, test


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--target-col", default="Success")
    ap.add_argument("--features", default="")
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)
    args = ap.parse_args()

    tag = (args.tag or "").strip()

    df, labels_src = read_labels_by_tag(tag)

    if args.target_col not in df.columns:
        raise ValueError(f"Missing target col: {args.target_col}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feat_cols, feat_src = resolve_feature_cols(args.features)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    train_df, test_df = date_split(df, "Date", args.train_ratio)

    X_train = train_df[feat_cols].values
    y_train = train_df[args.target_col].astype(int).values

    X_test = test_df[feat_cols].values
    y_test = test_df[args.target_col].astype(int).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=args.max_iter)

    # 안정적인 CV
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    try:
        model = CalibratedClassifierCV(estimator=base, cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, cv=tscv)

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    out_model = APP_DIR / (f"model_{tag}.pkl" if tag else "model.pkl")
    out_scaler = APP_DIR / (f"scaler_{tag}.pkl" if tag else "scaler.pkl")

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    report = {
        "tag": tag,
        "labels_source": labels_src,
        "feature_source": feat_src,
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "auc": float(auc) if np.isfinite(auc) else None,
    }

    meta_path = META_DIR / (f"train_model_report_{tag}.json" if tag else "train_model_report.json")
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[TRAIN DONE]")
    print("AUC:", auc)
    print("labels_src:", labels_src)
    print("model:", out_model)
    print("=" * 60)


if __name__ == "__main__":
    main()