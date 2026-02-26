#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

# ✅ split별 signals 폴더로 분리 (예: data/signals/taucur, data/signals/tauq255025)
TAU_SPLIT="${TAU_SPLIT:-}"
BASE_OUT_DIR="${OUT_DIR:-data/signals}"
if [ -n "$TAU_SPLIT" ]; then
  OUT_DIR="${BASE_OUT_DIR}/${TAU_SPLIT}"
else
  OUT_DIR="${BASE_OUT_DIR}"
fi
mkdir -p "$OUT_DIR"

TAG="${LABEL_KEY:-run}"
echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TAU_SPLIT=${TAU_SPLIT:-<none>}"

# Required envs
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"
: "${TRAIL_STOPS:?}"
: "${TP1_FRAC:?}"
: "${ENABLE_TRAILING:?}"
: "${TOPK_CONFIGS:?}"
: "${PS_MINS:?}"
: "${BADEXIT_MAXES:?}"
: "${MAX_LEVERAGE_PCT:?}"

# ✅ optional (빈값/미설정이어도 OK)
EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-}"
REQUIRE_FILES="${REQUIRE_FILES:-}"

DATE_FROM="${WF_DATE_FROM:-}"
DATE_TO="${WF_DATE_TO:-}"

# ✅ NEW: Regime filter env (optional; defaults are safe)
REGIME_MODE="${REGIME_MODE:-off}"                   # off|basic|trend|dd|combo
REGIME_DD_MAX="${REGIME_DD_MAX:-0.20}"              # levered 기준
REGIME_RET20_MIN="${REGIME_RET20_MIN:-0.00}"
REGIME_ATR_MAX="${REGIME_ATR_MAX:-1.30}"            # levered 기준
REGIME_LEVERAGE_MULT="${REGIME_LEVERAGE_MULT:-3.0}" # 3x universe 고려
echo "[INFO] REGIME_MODE=$REGIME_MODE DD_MAX=$REGIME_DD_MAX RET20_MIN=$REGIME_RET20_MIN ATR_MAX=$REGIME_ATR_MAX LEV_MULT=$REGIME_LEVERAGE_MULT"

# re-eval thresholds (engine args)
REVAL_PS_STRONG="${REVAL_PS_STRONG:-0.70}"
REVAL_PT_STRONG="${REVAL_PT_STRONG:-0.20}"
REVAL_PS_PASS="${REVAL_PS_PASS:-0.60}"
REVAL_PT_PASS="${REVAL_PT_PASS:-0.35}"
echo "[INFO] REVAL_PS_STRONG=$REVAL_PS_STRONG REVAL_PT_STRONG=$REVAL_PT_STRONG REVAL_PS_PASS=$REVAL_PS_PASS REVAL_PT_PASS=$REVAL_PT_PASS"

# dedupe on/off (default: true)
DEDUP_PICKS="${DEDUP_PICKS:-true}"
DEDUP_PICKS="$(echo "$DEDUP_PICKS" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$DEDUP_PICKS" != "true" && "$DEDUP_PICKS" != "false" ]]; then
  echo "[ERROR] DEDUP_PICKS must be true/false (got: $DEDUP_PICKS)"
  exit 1
fi
echo "[INFO] DEDUP_PICKS=$DEDUP_PICKS"

# MAX_EXTEND_DAYS는 "있으면 사용", 없으면 H//2 자동계산
if [ -z "${MAX_EXTEND_DAYS:-}" ]; then
  MAX_EXTEND_DAYS="$(python - <<PY
H=int("${MAX_DAYS}")
print(max(1, H//2))
PY
)"
  export MAX_EXTEND_DAYS
fi
echo "[INFO] MAX_EXTEND_DAYS=$MAX_EXTEND_DAYS (auto: H//2 when not provided)"

# 옵션B 제거: tp1-trail-unlimited는 true만 사용
TP1_TRAIL_UNLIMITED="true"

# cap 모드 제어
CAP_COMPARE="${CAP_COMPARE:-false}"
CAP_COMPARE="$(echo "$CAP_COMPARE" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$CAP_COMPARE" != "true" && "$CAP_COMPARE" != "false" ]]; then
  echo "[ERROR] CAP_COMPARE must be true/false (got: $CAP_COMPARE)"
  exit 1
fi

TP1_HOLD_CAP_SINGLE="${TP1_HOLD_CAP_SINGLE:-none}"
TP1_HOLD_CAP_SINGLE="$(echo "$TP1_HOLD_CAP_SINGLE" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$TP1_HOLD_CAP_SINGLE" != "none" && "$TP1_HOLD_CAP_SINGLE" != "h2" && "$TP1_HOLD_CAP_SINGLE" != "total" ]]; then
  echo "[ERROR] TP1_HOLD_CAP_SINGLE must be none|h2|total (got: $TP1_HOLD_CAP_SINGLE)"
  exit 1
fi

TP1_HOLD_CAP_MODES="${TP1_HOLD_CAP_MODES:-none,h2,total}"
echo "[INFO] CAP_COMPARE=$CAP_COMPARE"
echo "[INFO] TP1_HOLD_CAP_SINGLE=$TP1_HOLD_CAP_SINGLE"
echo "[INFO] TP1_HOLD_CAP_MODES=$TP1_HOLD_CAP_MODES"

# ✅ NEW: Tau/DCA 옵션 (엔진이 지원할 때만 켜서 전달)
USE_TAU_H="${USE_TAU_H:-false}"
USE_TAU_H="$(echo "$USE_TAU_H" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$USE_TAU_H" != "true" && "$USE_TAU_H" != "false" ]]; then
  echo "[ERROR] USE_TAU_H must be true/false (got: $USE_TAU_H)"
  exit 1
fi

ENABLE_DCA="${ENABLE_DCA:-false}"
ENABLE_DCA="$(echo "$ENABLE_DCA" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$ENABLE_DCA" != "true" && "$ENABLE_DCA" != "false" ]]; then
  echo "[ERROR] ENABLE_DCA must be true/false (got: $ENABLE_DCA)"
  exit 1
fi

# DCA 조건(옵션)
DCA_MAX_ADDS="${DCA_MAX_ADDS:-}"
DCA_GAP_DAYS="${DCA_GAP_DAYS:-}"
DCA_TRIGGER="${DCA_TRIGGER:-}"
DCA_ADD_FRAC="${DCA_ADD_FRAC:-}"

echo "[INFO] USE_TAU_H=$USE_TAU_H ENABLE_DCA=$ENABLE_DCA DCA_MAX_ADDS=${DCA_MAX_ADDS:-<unset>} DCA_GAP_DAYS=${DCA_GAP_DAYS:-<unset>} DCA_TRIGGER=${DCA_TRIGGER:-<unset>} DCA_ADD_FRAC=${DCA_ADD_FRAC:-<unset>}"

# ----- helpers
split_csv() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
for p in parts:
  print(p)
PY
}

split_scsv() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(";") if p.strip()]
for p in parts:
  print(p)
PY
}

suffix_float() {
  python - <<PY
x=float("$1")
print(str(x).replace(".","p").replace("-","m"))
PY
}

first_csv_item() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
print(parts[0] if parts else "")
PY
}

picks_hash() {
  local file="$1"
  python - <<PY
import hashlib, pandas as pd
from pathlib import Path
p = Path("$file")
if not p.exists():
    print("")
    raise SystemExit(0)
df = pd.read_csv(p)
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).astype(str)
if "Ticker" in df.columns:
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
cols = [c for c in ["Date","Ticker"] if c in df.columns]
df = df[cols].dropna().sort_values(cols).reset_index(drop=True)
payload = df.to_csv(index=False).encode("utf-8")
print(hashlib.sha1(payload).hexdigest())
PY
}

# cap modes iterator
iter_cap_modes() {
  if [ "$CAP_COMPARE" = "true" ]; then
    split_csv "$TP1_HOLD_CAP_MODES"
  else
    echo "$TP1_HOLD_CAP_SINGLE"
  fi
}

# ----- print config
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MINS=$PS_MINS BADEXIT_MAXES=$BADEXIT_MAXES MAX_LEVERAGE_PCT=$MAX_LEVERAGE_PCT"
echo "[INFO] TP1_TRAIL_UNLIMITED=$TP1_TRAIL_UNLIMITED"
echo "[INFO] EXCLUDE_TICKERS=${EXCLUDE_TICKERS:-<empty>}"
echo "[INFO] REQUIRE_FILES=${REQUIRE_FILES:-<empty>}"

DEFAULT_TMAX="$(first_csv_item "$P_TAIL_THRESHOLDS")"
DEFAULT_UQ="$(first_csv_item "$UTILITY_QUANTILES")"
DEFAULT_BE="$(first_csv_item "$BADEXIT_MAXES")"
if [ -z "$DEFAULT_TMAX" ] || [ -z "$DEFAULT_UQ" ] || [ -z "$DEFAULT_BE" ]; then
  echo "[ERROR] P_TAIL_THRESHOLDS / UTILITY_QUANTILES / BADEXIT_MAXES must be non-empty CSV."
  exit 1
fi

HASH_DIR="$OUT_DIR/_picks_hash"
mkdir -p "$HASH_DIR"
seen_hash_file="$HASH_DIR/seen_hashes.txt"
touch "$seen_hash_file"

is_hash_seen() { local h="$1"; [ -n "$h" ] && grep -q "^$h$" "$seen_hash_file"; }
mark_hash_seen() { local h="$1"; [ -n "$h" ] && echo "$h" >> "$seen_hash_file"; }

# ----- main loops
while read -r mode; do
  mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | xargs)"
  [ -z "$mode" ] && continue

  if [ "$mode" = "none" ]; then
    TMAX_LIST="$DEFAULT_TMAX"
    UQ_LIST="$DEFAULT_UQ"
  elif [ "$mode" = "tail" ]; then
    TMAX_LIST="$P_TAIL_THRESHOLDS"
    UQ_LIST="$DEFAULT_UQ"
  elif [ "$mode" = "utility" ]; then
    TMAX_LIST="$DEFAULT_TMAX"
    UQ_LIST="$UTILITY_QUANTILES"
  elif [ "$mode" = "tail_utility" ]; then
    TMAX_LIST="$P_TAIL_THRESHOLDS"
    UQ_LIST="$UTILITY_QUANTILES"
  else
    echo "[ERROR] Unknown mode: $mode"
    exit 1
  fi

  while read -r tmax; do
    while read -r uq; do
      while read -r rank_by; do
        while read -r psmin; do
          while read -r be; do
            while read -r topk_line; do
              K="${topk_line%%|*}"
              W="${topk_line#*|}"

              while read -r trail; do
                t_s="$(suffix_float "$tmax")"
                uq_s="$(suffix_float "$uq")"
                lam_s="$(suffix_float "$LAMBDA_TAIL")"
                ps_s="$(suffix_float "$psmin")"
                be_s="$(suffix_float "$be")"
                tr_s="$(suffix_float "$trail")"
                tp_pct="$(python - <<PY
f=float("$TP1_FRAC")
print(int(round(f*100)))
PY
)"
                tu_s="tu1"

                base_suffix="${mode}_${tu_s}_t${t_s}_q${uq_s}_r${rank_by}_lam${lam_s}_ps${ps_s}_be${be_s}_k${K}_w$(echo "$W" | tr ',' '_')_tp${tp_pct}_tr${tr_s}"

                echo "=============================="
                echo "[RUN] mode=$mode tail_max=$tmax u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL ps_min=$psmin badexit_max=$be topk=$K weights=$W trail=$trail base_suffix=$base_suffix"
                echo "=============================="

                python "$PRED" \
                  --profit-target "$PROFIT_TARGET" \
                  --max-days "$MAX_DAYS" \
                  --stop-level "$STOP_LEVEL" \
                  --max-extend-days "$MAX_EXTEND_DAYS" \
                  --mode "$mode" \
                  --tail-threshold "$tmax" \
                  --utility-quantile "$uq" \
                  --rank-by "$rank_by" \
                  --lambda-tail "$LAMBDA_TAIL" \
                  --topk "$K" \
                  --ps-min "$psmin" \
                  --badexit-max "$be" \
                  --tag "$TAG" \
                  --suffix "$base_suffix" \
                  --exclude-tickers "$EXCLUDE_TICKERS" \
                  --out-dir "$OUT_DIR" \
                  --require-files "$REQUIRE_FILES" \
                  --regime-mode "$REGIME_MODE" \
                  --regime-dd-max "$REGIME_DD_MAX" \
                  --regime-ret20-min "$REGIME_RET20_MIN" \
                  --regime-atr-max "$REGIME_ATR_MAX" \
                  --regime-leverage-mult "$REGIME_LEVERAGE_MULT" \
                  ${DATE_FROM:+--date-from "$DATE_FROM"} \
                  ${DATE_TO:+--date-to "$DATE_TO"}

                picks_path="$OUT_DIR/picks_${TAG}_gate_${base_suffix}.csv"
                if [ ! -f "$picks_path" ]; then
                  echo "[WARN] picks missing -> skip simulate/summarize (base_suffix=$base_suffix)"
                  continue
                fi

                rows="$(python - <<PY
import pandas as pd
try:
  df=pd.read_csv("$picks_path")
  print(len(df))
except Exception:
  print(0)
PY
)"
                if [ "${rows:-0}" = "0" ]; then
                  echo "[INFO] picks rows=0 -> skip simulate/summarize (base_suffix=$base_suffix)"
                  continue
                fi

                if [ "$DEDUP_PICKS" = "true" ]; then
                  h="$(picks_hash "$picks_path")"
                  if is_hash_seen "$h"; then
                    echo "[INFO] duplicate picks hash=$h -> proceed (base_suffix=$base_suffix)"
                  else
                    mark_hash_seen "$h"
                  fi
                fi

                while read -r cap_mode; do
                  cap_mode="$(echo "$cap_mode" | tr '[:upper:]' '[:lower:]' | xargs)"
                  [ -z "$cap_mode" ] && continue
                  if [[ "$cap_mode" != "none" && "$cap_mode" != "h2" && "$cap_mode" != "total" ]]; then
                    echo "[ERROR] invalid cap_mode=$cap_mode (must be none|h2|total)"
                    exit 1
                  fi

                  cap_suffix="${base_suffix}_cap${cap_mode}"
                  echo "[SIM] cap_mode=$cap_mode cap_suffix=$cap_suffix"

                  SIM_EXTRA_ARGS=()
                  if [ "$USE_TAU_H" = "true" ]; then
                    SIM_EXTRA_ARGS+=( --use-tau-h "true" )
                  fi
                  if [ "$ENABLE_DCA" = "true" ]; then
                    SIM_EXTRA_ARGS+=( --enable-dca "true" )
                    [ -n "$DCA_MAX_ADDS" ] && SIM_EXTRA_ARGS+=( --dca-max-adds "$DCA_MAX_ADDS" )
                    [ -n "$DCA_GAP_DAYS" ] && SIM_EXTRA_ARGS+=( --dca-gap-days "$DCA_GAP_DAYS" )
                    [ -n "$DCA_TRIGGER" ] && SIM_EXTRA_ARGS+=( --dca-trigger "$DCA_TRIGGER" )
                    [ -n "$DCA_ADD_FRAC" ] && SIM_EXTRA_ARGS+=( --dca-add-frac "$DCA_ADD_FRAC" )
                  fi

                  python "$SIM" \
                    --picks-path "$picks_path" \
                    --profit-target "$PROFIT_TARGET" \
                    --max-days "$MAX_DAYS" \
                    --stop-level "$STOP_LEVEL" \
                    --max-extend-days "$MAX_EXTEND_DAYS" \
                    --max-leverage-pct "$MAX_LEVERAGE_PCT" \
                    --enable-trailing "$ENABLE_TRAILING" \
                    --tp1-frac "$TP1_FRAC" \
                    --trail-stop "$trail" \
                    --tp1-trail-unlimited "$TP1_TRAIL_UNLIMITED" \
                    --tp1-hold-cap "$cap_mode" \
                    --reval-ps-strong "$REVAL_PS_STRONG" \
                    --reval-pt-strong "$REVAL_PT_STRONG" \
                    --reval-ps-pass "$REVAL_PS_PASS" \
                    --reval-pt-pass "$REVAL_PT_PASS" \
                    --topk "$K" \
                    --weights "$W" \
                    --tag "$TAG" \
                    --suffix "$cap_suffix" \
                    --out-dir "$OUT_DIR" \
                    "${SIM_EXTRA_ARGS[@]}"

                  trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${cap_suffix}.parquet"

                  python "$SUM" \
                    --trades-path "$trades_path" \
                    --tag "$TAG" \
                    --suffix "$cap_suffix" \
                    --profit-target "$PROFIT_TARGET" \
                    --max-days "$MAX_DAYS" \
                    --stop-level "$STOP_LEVEL" \
                    --max-extend-days "$MAX_EXTEND_DAYS" \
                    --out-dir "$OUT_DIR"

                done < <(iter_cap_modes)

              done < <(split_csv "$TRAIL_STOPS")
            done < <(split_scsv "$TOPK_CONFIGS")
          done < <(split_csv "$BADEXIT_MAXES")
        done < <(split_csv "$PS_MINS")
      done < <(split_csv "$RANK_METRICS")
    done < <(split_csv "$UQ_LIST")
  done < <(split_csv "$TMAX_LIST")

done < <(split_csv "$GATE_MODES")

echo "[DONE] grid finished"
if [ "$DEDUP_PICKS" = "true" ]; then
  echo "[INFO] unique picks hashes: $(wc -l < "$seen_hash_file" | tr -d ' ')"
fi
ls -la "$OUT_DIR" | sed -n '1,200p'