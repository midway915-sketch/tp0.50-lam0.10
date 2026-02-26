#!/usr/bin/env python3
"""
Generate daily picks (TopK) for gate.
Writes:
- picks_{tag}_gate_{suffix}.csv
- picks_{tag}_gate_{suffix}.debug.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _load_features(features_path: str) -> pd.DataFrame:
    if features_path:
        p = Path(features_path)
        if not p.exists():
            raise FileNotFoundError(f"--features-path not found: {p}")
        df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    else:
        p_parq = FEATURES_DIR / "features_scored.parquet"
        p_csv = FEATURES_DIR / "features_scored.csv"
        df = read_table(p_parq, p_csv)

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("features must include Date,Ticker")

    df = df.copy()
    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)
    return df


def _resolve_model_dir(model_dir_arg: str) -> Path:
    md = (model_dir_arg or "").strip()
    if not md:
        md = os.getenv("MODEL_DIR", "").strip()
    return Path(md) if md else Path("app")


def _load_models(model_dir: Path):
    import joblib

    model_p = model_dir / "model.pkl"
    scaler_p = model_dir / "scaler.pkl"
    if not model_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(f"Missing model/scaler: {model_p} {scaler_p}")
    return joblib.load(model_p), joblib.load(scaler_p)


def _load_tail_models(model_dir: Path):
    import joblib

    model_p = model_dir / "tail_model.pkl"
    scaler_p = model_dir / "tail_scaler.pkl"
    if not model_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(f"Missing tail model/scaler: {model_p} {scaler_p}")
    return joblib.load(model_p), joblib.load(scaler_p)


def _load_badexit_models(model_dir: Path):
    """
    Supports both:
      - model_dir/badexit_model.pkl + badexit_scaler.pkl (staged)
      - model_dir/badexit_model_{tag}.pkl etc. (not used here, but staged should exist)
    """
    import joblib

    model_p = model_dir / "badexit_model.pkl"
    scaler_p = model_dir / "badexit_scaler.pkl"
    if not model_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(f"Missing badexit model/scaler: {model_p} {scaler_p}")
    return joblib.load(model_p), joblib.load(scaler_p)


def _has_valid_numeric_col(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    return np.isfinite(s).any()


def _load_ssot_feature_cols() -> list[str] | None:
    """
    data/meta/feature_cols.json supports:
      - ["c1","c2",...]
      - {"feature_cols":[...]}
      - {"cols":[...]}
      - {"features":[...]}
      - {"p_success_cols":[...]}  (legacy fallback)
    """
    meta = DATA_DIR / "meta" / "feature_cols.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        cols = None
        if isinstance(payload, list):
            cols = payload
        elif isinstance(payload, dict):
            for k in ("feature_cols", "cols", "features", "p_success_cols"):
                v = payload.get(k)
                if isinstance(v, list) and v:
                    cols = v
                    break
        if isinstance(cols, list) and cols:
            out = [str(c).strip() for c in cols if str(c).strip()]
            return out if out else None
    except Exception:
        return None
    return None


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    ssot = _load_ssot_feature_cols()
    if ssot:
        missing = [c for c in ssot if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols.json contains missing cols in features: {missing[:20]}")
        return ssot

    drop = {
        "Date",
        "Ticker",
        "p_success",
        "p_tail",
        "p_badexit",
        "tau_H",
        "tau_class",
        "tau_pmax",
        "ret_score",
        "utility",
        "utility_raw",
        "tail_pen",
        "utility_qpass",
        "score",
    }
    cols = [c for c in df.columns if c not in drop]
    out: list[str] = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
        else:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                out.append(c)

    if not out:
        raise RuntimeError("No numeric feature cols found.")
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def _apply_regime_filter(
    df: pd.DataFrame,
    mode: str,
    dd_max: float,
    ret20_min: float,
    atr_max: float,
    leverage_mult: float,
) -> tuple[pd.DataFrame, dict]:
    m = (mode or "off").strip().lower()
    if m == "off":
        return df, {"enabled": False, "mode": "off"}

    audit: dict = {
        "enabled": True,
        "mode": m,
        "dd_max": float(dd_max),
        "ret20_min": float(ret20_min),
        "atr_max": float(atr_max),
        "leverage_mult": float(leverage_mult),
        "used_cols": [],
        "n_before": int(len(df)),
        "n_after": None,
    }

    need_dd = "Market_Drawdown"
    need_ret = "Market_ret_20"
    need_atr = "Market_ATR_ratio"

    lev = float(leverage_mult) if np.isfinite(leverage_mult) and leverage_mult > 0 else 1.0
    dd_thr = -float(dd_max) / lev
    ret_thr = float(ret20_min) / lev
    atr_thr = float(atr_max) / lev

    out = df

    def has_col(c: str) -> bool:
        return c in out.columns and pd.api.types.is_numeric_dtype(out[c])

    if m in ("dd", "basic", "combo"):
        if has_col(need_dd):
            audit["used_cols"].append(need_dd)
            out = out[out[need_dd].astype(float) >= dd_thr].copy()
        else:
            audit.setdefault("warnings", []).append(f"missing {need_dd} -> dd filter skipped")

    if m in ("trend", "combo"):
        if has_col(need_ret):
            audit["used_cols"].append(need_ret)
            out = out[out[need_ret].astype(float) >= ret_thr].copy()
        else:
            audit.setdefault("warnings", []).append(f"missing {need_ret} -> trend filter skipped")

    if m in ("basic", "combo"):
        if has_col(need_atr):
            audit["used_cols"].append(need_atr)
            out = out[out[need_atr].astype(float) <= atr_thr].copy()
        else:
            audit.setdefault("warnings", []).append(f"missing {need_atr} -> atr filter skipped")

    audit["n_after"] = int(len(out))
    audit["effective_thresholds"] = {
        "dd_thr_market": float(dd_thr),
        "ret20_thr_market": float(ret_thr),
        "atr_thr_market": float(atr_thr),
    }
    return out, audit


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--ps-min", default=0.0, type=float)
    ap.add_argument("--badexit-max", default=None, type=float)

    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", required=True)

    ap.add_argument("--exclude-tickers", default="")
    ap.add_argument("--features-path", default="")

    ap.add_argument("--out-dir", default="data/signals", help="output directory for picks/meta")
    ap.add_argument("--require-files", default="", help="comma-separated file paths that must exist; missing -> exit code 2")

    ap.add_argument("--regime-mode", default="off", choices=["off", "basic", "trend", "dd", "combo"])
    ap.add_argument("--regime-dd-max", default=0.20, type=float)
    ap.add_argument("--regime-ret20-min", default=0.00, type=float)
    ap.add_argument("--regime-atr-max", default=1.30, type=float)
    ap.add_argument("--regime-leverage-mult", default=3.0, type=float)

    # WF date filter
    ap.add_argument("--date-from", default="", type=str, help="optional inclusive date filter YYYY-MM-DD")
    ap.add_argument("--date-to", default="", type=str, help="optional inclusive date filter YYYY-MM-DD")

    # model directory override (also supports env MODEL_DIR)
    ap.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Directory containing model/scaler files. If empty, uses env MODEL_DIR, else falls back to app/.",
    )

    args = ap.parse_args()

    # require-files fail-fast
    req = [x.strip() for x in str(args.require_files or "").split(",") if x.strip()]
    if req:
        missing = [p for p in req if not Path(p).exists()]
        if missing:
            print(f"[ERROR] Missing required files: {missing}")
            sys.exit(2)

    model_dir = _resolve_model_dir(args.model_dir)
    feats = _load_features(args.features_path)

    # apply date range early (inclusive)
    if str(args.date_from).strip():
        d0 = pd.to_datetime(args.date_from, errors="coerce")
        if pd.isna(d0):
            raise ValueError(f"--date-from invalid: {args.date_from}")
        feats = feats[feats["Date"] >= d0].copy()
    if str(args.date_to).strip():
        d1 = pd.to_datetime(args.date_to, errors="coerce")
        if pd.isna(d1):
            raise ValueError(f"--date-to invalid: {args.date_to}")
        feats = feats[feats["Date"] <= d1].copy()

    # ---- p_success
    use_scored_ps = _has_valid_numeric_col(feats, "p_success")
    if use_scored_ps:
        feats2 = feats.copy()
        feats2["p_success"] = pd.to_numeric(feats2["p_success"], errors="coerce").fillna(0.0).astype(float)
        scored_mode_ps = "use_features_col"
        feat_cols: list[str] | None = None
    else:
        model, scaler = _load_models(model_dir)
        feat_cols = _get_feature_cols(feats)
        feats2 = _coerce_numeric(feats, feat_cols)
        X = feats2[feat_cols].to_numpy(dtype=float)
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)
        feats2["p_success"] = (proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]).astype(float)
        scored_mode_ps = "model_predict"

    # ---- p_tail
    use_scored_pt = _has_valid_numeric_col(feats2, "p_tail")
    need_tail = str(args.mode) in ("tail", "tail_utility")
    if use_scored_pt:
        feats2["p_tail"] = pd.to_numeric(feats2.get("p_tail"), errors="coerce").fillna(0.0).astype(float)
        scored_mode_pt = "use_features_col"
    else:
        if need_tail:
            if feat_cols is None:
                feat_cols = _get_feature_cols(feats2)
                feats2 = _coerce_numeric(feats2, feat_cols)
            X = feats2[feat_cols].to_numpy(dtype=float)
            tail_model, tail_scaler = _load_tail_models(model_dir)
            Xt = tail_scaler.transform(X)
            tproba = tail_model.predict_proba(Xt)
            feats2["p_tail"] = (tproba[:, 1] if tproba.shape[1] >= 2 else tproba[:, 0]).astype(float)
            scored_mode_pt = "model_predict"
        else:
            feats2["p_tail"] = 0.0
            scored_mode_pt = "filled_0_not_needed"

    # ---- p_badexit (compute if needed for filtering)
    scored_mode_be = "not_used"
    if args.badexit_max is not None:
        use_scored_be = _has_valid_numeric_col(feats2, "p_badexit")
        if use_scored_be:
            feats2["p_badexit"] = pd.to_numeric(feats2["p_badexit"], errors="coerce").fillna(0.0).astype(float)
            scored_mode_be = "use_features_col"
        else:
            # only compute if we actually want to filter
            try:
                if feat_cols is None:
                    feat_cols = _get_feature_cols(feats2)
                    feats2 = _coerce_numeric(feats2, feat_cols)
                X = feats2[feat_cols].to_numpy(dtype=float)
                be_model, be_scaler = _load_badexit_models(model_dir)
                Xb = be_scaler.transform(X)
                bproba = be_model.predict_proba(Xb)
                feats2["p_badexit"] = (bproba[:, 1] if bproba.shape[1] >= 2 else bproba[:, 0]).astype(float)
                scored_mode_be = "model_predict"
            except FileNotFoundError as e:
                # if model missing, degrade gracefully to 0.0 so pipeline doesn't die
                feats2["p_badexit"] = 0.0
                scored_mode_be = f"filled_0_missing_models ({e})"

    # exclude tickers
    if args.exclude_tickers:
        ex = [x.strip().upper() for x in args.exclude_tickers.split(",") if x.strip()]
        if ex:
            feats2 = feats2[~feats2["Ticker"].isin(ex)].copy()

    # ps_min filter
    feats2 = feats2[feats2["p_success"] >= float(args.ps_min)].copy()

    # regime filter
    feats2, regime_audit = _apply_regime_filter(
        feats2,
        mode=str(args.regime_mode),
        dd_max=float(args.regime_dd_max),
        ret20_min=float(args.regime_ret20_min),
        atr_max=float(args.regime_atr_max),
        leverage_mult=float(args.regime_leverage_mult),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    debug_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.debug.json"

    if feats2.empty:
        pd.DataFrame(columns=["Date", "Ticker"]).to_csv(picks_path, index=False)
        debug_path.write_text(
            json.dumps(
                {
                    "empty": True,
                    "reason": "filters_or_date_range",
                    "ps_min": float(args.ps_min),
                    "date_from": args.date_from,
                    "date_to": args.date_to,
                    "regime": regime_audit,
                    "p_success_source": scored_mode_ps,
                    "p_tail_source": scored_mode_pt,
                    "p_badexit_source": scored_mode_be,
                    "model_dir": str(model_dir),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[WARN] empty picks -> wrote {picks_path}")
        return

    mode = args.mode
    tmax = float(args.tail_threshold)
    uq = float(args.utility_quantile)
    lam = float(args.lambda_tail)

    feats2["ret_score"] = feats2.get("ret_score", np.nan)
    feats2["utility_raw"] = feats2.get("utility", np.nan)

    # fallback utility
    if "utility" not in feats2.columns or feats2["utility_raw"].isna().all():
        feats2["utility_raw"] = feats2["p_success"].astype(float)

    # tail penalty (only matters for tail modes; harmless otherwise)
    feats2["tail_pen"] = np.maximum(0.0, feats2["p_tail"] - tmax)

    qv = feats2["utility_raw"].quantile(uq) if np.isfinite(uq) else feats2["utility_raw"].min()
    feats2["utility_qpass"] = feats2["utility_raw"] >= qv

    if mode == "none":
        feats2["score"] = feats2["utility_raw"]
    elif mode == "tail":
        feats2["score"] = feats2["utility_raw"] - lam * feats2["tail_pen"]
    elif mode == "utility":
        feats2 = feats2[feats2["utility_qpass"]].copy()
        feats2["score"] = feats2["utility_raw"]
    elif mode == "tail_utility":
        feats2 = feats2[feats2["utility_qpass"]].copy()
        feats2["score"] = feats2["utility_raw"] - lam * feats2["tail_pen"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if args.rank_by == "p_success":
        feats2["score"] = feats2["p_success"]
    elif args.rank_by == "ret_score":
        if "ret_score" not in feats2.columns or feats2["ret_score"].isna().all():
            feats2["ret_score"] = feats2["utility_raw"]
        feats2["score"] = feats2["ret_score"].fillna(-np.inf)

    # badexit filter
    if args.badexit_max is not None:
        feats2["p_badexit"] = pd.to_numeric(feats2.get("p_badexit", 0.0), errors="coerce").fillna(0.0).astype(float)
        feats2 = feats2[feats2["p_badexit"] <= float(args.badexit_max)].copy()

    feats2 = feats2.sort_values(["Date", "score"], ascending=[True, False]).reset_index(drop=True)

    picks = (
        feats2.groupby("Date", group_keys=False)
        .head(int(args.topk))[["Date", "Ticker", "score", "p_success", "p_tail"]]
        .copy()
    )

    # output: only Date,Ticker (as you intended)
    picks[["Date", "Ticker"]].to_csv(picks_path, index=False)

    dbg = {
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": mode,
        "tail_threshold": tmax,
        "utility_quantile": uq,
        "rank_by": args.rank_by,
        "lambda_tail": lam,
        "topk": int(args.topk),
        "ps_min": float(args.ps_min),
        "badexit_max": float(args.badexit_max) if args.badexit_max is not None else None,
        "exclude_tickers": args.exclude_tickers,
        "features_path": args.features_path or "data/features/features_scored.(parquet/csv)",
        "out_dir": str(out_dir),
        "n_rows_features": int(len(feats2)),
        "n_rows_picks": int(len(picks)),
        "regime": regime_audit,
        "p_success_source": scored_mode_ps,
        "p_tail_source": scored_mode_pt,
        "p_badexit_source": scored_mode_be,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "model_dir": str(model_dir),
    }
    debug_path.write_text(json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote picks: {picks_path} rows={len(picks)}")
    print(f"[DONE] wrote debug: {debug_path}")


if __name__ == "__main__":
    main()