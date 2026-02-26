#!/usr/bin/env python3
# scripts/score_features.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
META_DIR = DATA_DIR / "meta"


# build_features.py 기준(16) + (옵션) RelStrength(1)
# ✅ NOTE: Sector_Ret_20 -> Market_ret_20 로 교체(기존 대화에서 수정했던 SSOT와 맞춤)
DEFAULT_FEATURES = [
    # base (9)
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
    "ret_score",
    # new (7)
    "ret_5",
    "ret_10",
    "ret_20",
    "breakout_20",
    "vol_surge",
    "trend_align",
    "beta_60",
    # optional / derived
    "Market_ret_20",
    "RelStrength",
]


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_features_exist(df: pd.DataFrame, feat_cols: list[str], *, warn_prefix: str = "") -> pd.DataFrame:
    """
    scoring 단계는 운영 편의상:
      - 컬럼 없으면 0.0으로 생성
      - 단, report/SSOT 기반일 때는 경고 출력
    """
    out = df.copy()
    missing = [c for c in feat_cols if c not in out.columns]
    if missing:
        pfx = f"{warn_prefix} " if warn_prefix else ""
        print(f"[WARN]{pfx}missing feature cols -> filled with 0.0: {missing}")
        for c in missing:
            out[c] = 0.0

    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def _load_feature_cols_from_report(report_path: Path) -> list[str] | None:
    if not report_path.exists():
        return None
    try:
        j = json.loads(report_path.read_text(encoding="utf-8"))
        cols = j.get("feature_cols")
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols if str(c).strip()]
    except Exception:
        return None
    return None


def _load_feature_cols_from_ssot() -> list[str] | None:
    """
    data/meta/feature_cols.json (SSOT) 우선 사용.

    ✅ 지원 형식:
      - ["c1","c2",...]
      - {"feature_cols":[...]}
      - {"cols":[...]}
      - {"features":[...]}
      - {"p_success_cols":[...]}  (legacy)
    """
    p = META_DIR / "feature_cols.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
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


def load_ps_feature_cols(tag: str) -> list[str] | None:
    return _load_feature_cols_from_report(META_DIR / f"train_model_report_{tag}.json")


def load_tail_feature_cols(tag: str) -> list[str] | None:
    return _load_feature_cols_from_report(META_DIR / f"train_tail_report_{tag}.json")


def load_tau_feature_cols(tag: str) -> list[str] | None:
    return _load_feature_cols_from_report(META_DIR / f"train_tau_report_{tag}.json")


def load_badexit_feature_cols(tag: str) -> list[str] | None:
    """
    ✅ workflow에서 저장하는 형태:
      data/meta/train_badexit_report_${TAG}.json
    """
    return _load_feature_cols_from_report(META_DIR / f"train_badexit_report_{tag}.json")


def parse_tau_h_map(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        return [30, 40, 50]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(float(p)))
        except Exception:
            pass
    return out if out else [30, 40, 50]


def class_to_h(cls: int, hmap: list[int]) -> int:
    if not hmap:
        return 40
    if cls < 0:
        return hmap[0]
    if cls >= len(hmap):
        return hmap[-1]
    return hmap[cls]


def parse_csv_cols(s: str) -> list[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


def _strip_h_from_tag(tag: str) -> str:
    """
    ex) pt10_h40_sl10_ex30 -> pt10_sl10_ex30
    """
    core = re.sub(r"_h\d+", "", str(tag))
    core = core.replace("__", "_").strip("_")
    return core


def _predict_proba_1(model, X: np.ndarray) -> np.ndarray:
    # robust proba extraction
    try:
        p = model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)
    except Exception:
        # fallback: if decision_function exists, squash
        if hasattr(model, "decision_function"):
            z = model.decision_function(X)
            z = np.asarray(z, dtype=float)
            return 1.0 / (1.0 + np.exp(-z))
        # last resort: predict as class
        y = model.predict(X)
        y = np.asarray(y, dtype=float)
        return np.clip(y, 0.0, 1.0)


def _resolve_model_dir(args) -> Path | None:
    md = (getattr(args, "model_dir", "") or "").strip()
    if not md:
        md = (os.getenv("MODEL_DIR", "") or "").strip()
    if not md:
        return None
    return Path(md)


def _resolve_tau_paths(tag: str, base_model: str, base_scaler: str, model_dir: Path | None) -> tuple[Path, Path, str]:
    """
    ✅ tau 로딩 우선순위:
      1) model_dir/tau_model_{tag}.pkl, model_dir/tau_scaler_{tag}.pkl
      2) app/tau_model_{tag}.pkl, app/tau_scaler_{tag}.pkl   (구버전 호환)
      3) model_dir/tau_model.pkl, model_dir/tau_scaler.pkl
      4) args --tau-model/--tau-scaler (base)
    """
    tag = (tag or "").strip()

    if model_dir is not None and tag:
        cand_m = model_dir / f"tau_model_{tag}.pkl"
        cand_s = model_dir / f"tau_scaler_{tag}.pkl"
        if cand_m.exists() and cand_s.exists():
            return cand_m, cand_s, "tagged(model_dir/tau_*_{tag}.pkl)"

    if tag:
        cand_m = Path("app") / f"tau_model_{tag}.pkl"
        cand_s = Path("app") / f"tau_scaler_{tag}.pkl"
        if cand_m.exists() and cand_s.exists():
            return cand_m, cand_s, "tagged(app/tau_*_{tag}.pkl)"

    if model_dir is not None:
        cand_m = model_dir / "tau_model.pkl"
        cand_s = model_dir / "tau_scaler.pkl"
        if cand_m.exists() and cand_s.exists():
            return cand_m, cand_s, "base(model_dir/tau_model.pkl)"

    return Path(base_model), Path(base_scaler), "base(--tau-model/--tau-scaler)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score features_model with p_success, p_tail, p_badexit, and tau_class/tau_H -> save features_scored."
    )
    ap.add_argument("--tag", required=True, type=str, help="e.g. pt10_h40_sl10_ex30")

    ap.add_argument("--features-parq", default=str(FEATURE_DIR / "features_model.parquet"), type=str)
    ap.add_argument("--features-csv", default=str(FEATURE_DIR / "features_model.csv"), type=str)

    ap.add_argument("--out-parq", default=str(FEATURE_DIR / "features_scored.parquet"), type=str)
    ap.add_argument("--out-csv", default=str(FEATURE_DIR / "features_scored.csv"), type=str)

    # ✅ NEW: model namespace (per-period)
    ap.add_argument(
        "--model-dir",
        default=os.getenv("MODEL_DIR", "").strip(),
        type=str,
        help="Directory containing model/scaler files (overrides app/*). Example: data/models/wf_2018H1",
    )

    # p_success (base paths only used for fallback if model-dir doesn't have files)
    ap.add_argument("--ps-model", default="app/model.pkl", type=str)
    ap.add_argument("--ps-scaler", default="app/scaler.pkl", type=str)
    ap.add_argument(
        "--ps-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_model_report_{tag}) -> SSOT(meta/feature_cols.json) -> DEFAULT_FEATURES",
    )

    # p_tail
    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)
    ap.add_argument("--tail-scaler", default="app/tail_scaler.pkl", type=str)
    ap.add_argument(
        "--tail-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_tail_report_{tag}) -> SSOT -> ps-features",
    )

    # p_badexit
    ap.add_argument("--badexit-model", default="app/badexit_model.pkl", type=str)
    ap.add_argument("--badexit-scaler", default="app/badexit_scaler.pkl", type=str)
    ap.add_argument(
        "--badexit-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_badexit_report_{TAG}.json) -> SSOT -> ps-features",
    )

    # tau (tagged 우선)
    ap.add_argument("--tau-model", default="app/tau_model.pkl", type=str)
    ap.add_argument("--tau-scaler", default="app/tau_scaler.pkl", type=str)
    ap.add_argument(
        "--tau-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_tau_report_{tag}) -> SSOT -> ps-features",
    )
    ap.add_argument(
        "--tau-h-map",
        default="30,40,50",
        type=str,
        help="tau_class->H mapping. e.g. '30,40,50' means class0=30 class1=40 class2=50",
    )

    args = ap.parse_args()
    core_tag = _strip_h_from_tag(args.tag)
    hmap = parse_tau_h_map(args.tau_h_map)

    model_dir = _resolve_model_dir(args)
    if model_dir is not None:
        print(f"[INFO] MODEL_DIR={model_dir}")
    else:
        print("[INFO] MODEL_DIR=<none> (fallback to app/* + explicit paths)")

    feats = read_table(Path(args.features_parq), Path(args.features_csv)).copy()
    print("[DEBUG] features_model cols(head):", list(feats.columns)[:30], " index.name:", feats.index.name)

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date,Ticker")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    ssot_cols = _load_feature_cols_from_ssot()

    # -------------------------
    # feature columns resolve (ps baseline)
    # -------------------------
    if args.ps_features.strip():
        ps_cols = parse_csv_cols(args.ps_features)
        ps_cols_src = "--ps-features"
    else:
        ps_cols = load_ps_feature_cols(args.tag)
        if ps_cols:
            ps_cols_src = f"report(train_model_report_{args.tag}.json)"
        elif ssot_cols:
            ps_cols = ssot_cols
            ps_cols_src = "SSOT(data/meta/feature_cols.json)"
        else:
            ps_cols = DEFAULT_FEATURES
            ps_cols_src = "DEFAULT_FEATURES"
    ps_cols = [c for c in ps_cols if c]

    # -------------------------
    # tau_class / tau_H (tagged/model_dir 우선)
    # -------------------------
    tau_model_path, tau_scaler_path, tau_src = _resolve_tau_paths(args.tag, args.tau_model, args.tau_scaler, model_dir)

    if tau_model_path.exists() and tau_scaler_path.exists():
        tau_model = joblib.load(str(tau_model_path))
        tau_scaler = joblib.load(str(tau_scaler_path))

        if args.tau_features.strip():
            tau_cols = parse_csv_cols(args.tau_features)
            tau_cols_src = "--tau-features"
        else:
            tau_cols = load_tau_feature_cols(args.tag)
            if tau_cols:
                tau_cols_src = f"report(train_tau_report_{args.tag}.json)"
            elif ssot_cols:
                tau_cols = ssot_cols
                tau_cols_src = "SSOT(data/meta/feature_cols.json)"
            else:
                tau_cols = ps_cols
                tau_cols_src = "fallback(ps_cols)"

        tau_cols = [c for c in tau_cols if c]
        feats_tau = ensure_features_exist(feats, tau_cols, warn_prefix=f"[tau:{tau_cols_src}]")
        X_tau = feats_tau[tau_cols].to_numpy(dtype=float)
        X_tau_s = tau_scaler.transform(X_tau)

        try:
            tau_class = tau_model.predict(X_tau_s)
            tau_class = np.asarray(tau_class).astype(int)
        except Exception:
            proba = tau_model.predict_proba(X_tau_s)
            tau_class = np.argmax(proba, axis=1).astype(int)

        feats["tau_class"] = tau_class.astype(int)

        try:
            proba = tau_model.predict_proba(X_tau_s)
            feats["tau_pmax"] = np.max(proba, axis=1).astype(float)
        except Exception:
            feats["tau_pmax"] = 0.0

        feats["tau_H"] = feats["tau_class"].apply(lambda x: class_to_h(int(x), hmap)).astype(int)
    else:
        feats["tau_class"] = -1
        feats["tau_pmax"] = 0.0
        feats["tau_H"] = int(hmap[1]) if (hmap and len(hmap) >= 2) else 40
        tau_cols_src = "disabled(missing tau model/scaler)"

    # -------------------------
    # helper: per-H inference
    # -------------------------
    def _score_by_h(
        *,
        col_out: str,
        model_stem: str,
        scaler_stem: str,
        base_model_path: str,
        base_scaler_path: str,
        feat_cols: list[str],
        feat_cols_src: str,
        enabled: bool,
        default_value: float = 0.0,
    ) -> tuple[pd.Series, dict]:
        if not enabled:
            return pd.Series([default_value] * len(feats), index=feats.index, dtype=float), {"enabled": False}

        feats_x = ensure_features_exist(feats, feat_cols, warn_prefix=f"[{col_out}:{feat_cols_src}]")
        X = feats_x[feat_cols].to_numpy(dtype=float)

        # ✅ search order:
        # 1) model_dir/{stem}_{core_tag}_h{H}.pkl (and scaler)
        # 2) app/ parent of base_model_path (legacy)
        # 3) single fallback: model_dir/{stem}.pkl (and scaler) then base paths
        models: dict[int, tuple[Path, Path]] = {}

        search_dirs: list[Path] = []
        if model_dir is not None:
            search_dirs.append(model_dir)
        search_dirs.append(Path(base_model_path).parent)

        for sd in search_dirs:
            for h in sorted(set(hmap)):
                m_path = sd / f"{model_stem}_{core_tag}_h{int(h)}.pkl"
                s_path = sd / f"{scaler_stem}_{core_tag}_h{int(h)}.pkl"
                if m_path.exists() and s_path.exists():
                    models[int(h)] = (m_path, s_path)

            if models:
                audit_dir = sd
                break
        else:
            audit_dir = search_dirs[0] if search_dirs else Path(base_model_path).parent

        if not models:
            # single fallback priority: model_dir/{stem}.pkl
            if model_dir is not None:
                bm = model_dir / f"{model_stem}.pkl"
                bs = model_dir / f"{scaler_stem}.pkl"
                if bm.exists() and bs.exists():
                    model = joblib.load(str(bm))
                    scaler = joblib.load(str(bs))
                    Xs = scaler.transform(X)
                    p = _predict_proba_1(model, Xs)
                    return pd.Series(p, index=feats.index, dtype=float), {
                        "enabled": True,
                        "mode": "single_fallback_model_dir",
                        "model": str(bm),
                        "scaler": str(bs),
                        "core_tag": core_tag,
                    }

            bm = Path(base_model_path)
            bs = Path(base_scaler_path)
            if bm.exists() and bs.exists():
                model = joblib.load(str(bm))
                scaler = joblib.load(str(bs))
                Xs = scaler.transform(X)
                p = _predict_proba_1(model, Xs)
                return pd.Series(p, index=feats.index, dtype=float), {
                    "enabled": True,
                    "mode": "single_fallback_base_paths",
                    "model": str(bm),
                    "scaler": str(bs),
                    "core_tag": core_tag,
                }

            print(f"[WARN] no models found for {col_out} (core_tag={core_tag}) -> fill {default_value}")
            return pd.Series([default_value] * len(feats), index=feats.index, dtype=float), {
                "enabled": True,
                "mode": "missing_models",
                "core_tag": core_tag,
                "searched_dirs": [str(d) for d in search_dirs],
            }

        tauH = pd.to_numeric(feats["tau_H"], errors="coerce").fillna(int(hmap[1]) if len(hmap) >= 2 else 40).astype(int)
        out = np.full(len(feats), default_value, dtype=float)

        audit = {
            "enabled": True,
            "mode": "per_H",
            "core_tag": core_tag,
            "model_dir_used": str(audit_dir),
            "models_found": {str(k): {"model": str(v[0]), "scaler": str(v[1])} for k, v in models.items()},
        }

        # score where tau_H exactly matches
        for h, (m_path, s_path) in models.items():
            idx = np.where(tauH.to_numpy() == int(h))[0]
            if idx.size == 0:
                continue
            model = joblib.load(str(m_path))
            scaler = joblib.load(str(s_path))
            Xs = scaler.transform(X[idx])
            out[idx] = _predict_proba_1(model, Xs)

        # fallback for tau_H values without a model: nearest-H
        covered = set(models.keys())
        missing_idx = np.where(~np.isin(tauH.to_numpy(), list(covered)))[0]
        if missing_idx.size > 0:
            avail = sorted(covered)
            for i in missing_idx:
                th = int(tauH.iloc[i])
                nearest = min(avail, key=lambda x: abs(int(x) - th))
                m_path, s_path = models[int(nearest)]
                model = joblib.load(str(m_path))
                scaler = joblib.load(str(s_path))
                Xs = scaler.transform(X[i : i + 1])
                out[i] = float(_predict_proba_1(model, Xs)[0])
            audit["fallback_for_missing_tauH"] = True
        else:
            audit["fallback_for_missing_tauH"] = False

        return pd.Series(out, index=feats.index, dtype=float), audit

    # -------------------------
    # p_success (per tau_H)
    # -------------------------
    ps_series, ps_audit = _score_by_h(
        col_out="p_success",
        model_stem="model",
        scaler_stem="scaler",
        base_model_path=args.ps_model,
        base_scaler_path=args.ps_scaler,
        feat_cols=ps_cols,
        feat_cols_src=ps_cols_src,
        enabled=True,
        default_value=0.0,
    )
    feats["p_success"] = ps_series.astype(float)

    # -------------------------
    # p_tail (per tau_H)
    # -------------------------
    if args.tail_features.strip():
        tail_cols = parse_csv_cols(args.tail_features)
        tail_cols_src = "--tail-features"
    else:
        tail_cols = load_tail_feature_cols(args.tag)
        if tail_cols:
            tail_cols_src = f"report(train_tail_report_{args.tag}.json)"
        elif ssot_cols:
            tail_cols = ssot_cols
            tail_cols_src = "SSOT(data/meta/feature_cols.json)"
        else:
            tail_cols = ps_cols
            tail_cols_src = "fallback(ps_cols)"
    tail_cols = [c for c in tail_cols if c]

    tail_series, tail_audit = _score_by_h(
        col_out="p_tail",
        model_stem="tail_model",
        scaler_stem="tail_scaler",
        base_model_path=args.tail_model,
        base_scaler_path=args.tail_scaler,
        feat_cols=tail_cols,
        feat_cols_src=tail_cols_src,
        enabled=True,
        default_value=0.0,
    )
    feats["p_tail"] = tail_series.astype(float)

    # -------------------------
    # p_badexit (per tau_H)
    # -------------------------
    if args.badexit_features.strip():
        bad_cols = parse_csv_cols(args.badexit_features)
        bad_cols_src = "--badexit-features"
    else:
        bad_cols = load_badexit_feature_cols(f"wf_{os.getenv('WF_PERIOD','')}".strip())  # unused; just keep safe
        # 위 줄은 실수 방지용인데, 실제로는 아래에서 TAG="wf_${WF_PERIOD}"로 호출하니까 args.tag가 그걸 가지고 있음.
        # 그래서 args.tag를 우선으로 다시 덮는다.
        bad_cols = load_badexit_feature_cols(args.tag) or None

        if bad_cols:
            bad_cols_src = f"report(train_badexit_report_{args.tag}.json)"
        elif ssot_cols:
            bad_cols = ssot_cols
            bad_cols_src = "SSOT(data/meta/feature_cols.json)"
        else:
            bad_cols = ps_cols
            bad_cols_src = "fallback(ps_cols)"
    bad_cols = [c for c in (bad_cols or []) if c]

    bad_series, bad_audit = _score_by_h(
        col_out="p_badexit",
        model_stem="badexit_model",
        scaler_stem="badexit_scaler",
        base_model_path=args.badexit_model,
        base_scaler_path=args.badexit_scaler,
        feat_cols=bad_cols if bad_cols else ps_cols,  # 마지막 안전장치
        feat_cols_src=bad_cols_src if bad_cols else "fallback(ps_cols:empty_bad_cols)",
        enabled=True,
        default_value=0.0,
    )
    feats["p_badexit"] = bad_series.astype(float)

    # -------------------------
    # write outputs
    # -------------------------
    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    try:
        feats.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote: {out_parq} rows={len(feats)}")
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv")
        feats.to_csv(out_csv, index=False)
        print(f"[DONE] wrote: {out_csv} rows={len(feats)}")

    print("=" * 60)
    print("[INFO] scoring summary")
    print("tag:", args.tag)
    print("core_tag:", core_tag)
    print("MODEL_DIR:", str(model_dir) if model_dir is not None else "<none>")

    print("tau model/scaler src:", tau_src, "|", str(tau_model_path), "/", str(tau_scaler_path))
    print("tau cols source:", tau_cols_src)
    print("tau_h_map:", hmap)

    print("p_success cols source:", ps_cols_src, "cols:", ps_cols)
    print("p_success audit:", ps_audit)
    print("p_tail audit:", tail_audit)
    print("p_badexit audit:", bad_audit)
    print("=" * 60)


if __name__ == "__main__":
    main()