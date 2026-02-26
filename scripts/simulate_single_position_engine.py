#!/usr/bin/env python3
# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _failfast_require_files(require_files: str) -> None:
    req = [x.strip() for x in str(require_files or "").split(",") if x.strip()]
    if not req:
        return
    missing = [p for p in req if not Path(p).exists()]
    if missing:
        print(f"[ERROR] Missing required files: {missing}")
        raise SystemExit(2)


def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    desired : amount we'd like to spend today (>=0)
    """
    if desired <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


@dataclass
class Leg:
    ticker: str
    weight: float

    shares: float = 0.0
    invested: float = 0.0

    tp1_done: bool = False
    peak: float = 0.0  # for trailing after TP1

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan

    def value(self, close_px: float) -> float:
        if not np.isfinite(close_px) or close_px <= 0:
            return 0.0
        return float(self.shares) * float(close_px)


@dataclass
class CycleState:
    in_cycle: bool = False
    entry_date: pd.Timestamp | None = None

    seed: float = 0.0         # cash, can go negative
    entry_seed: float = 0.0   # S0 at entry

    H: int = 0
    unit: float = 0.0         # daily buy budget = entry_seed / H
    holding_days: int = 0

    # extend / grace
    pending_reval: bool = False
    ret_H: float = np.nan
    extending: bool = False
    grace_days_total: int = 0
    grace_days_left: int = 0
    recovery_threshold: float = np.nan
    reval_strength: str = ""
    reval_ps: float = np.nan
    reval_pt: float = np.nan

    # risk / stats
    max_leverage_pct: float = 0.0
    max_equity: float = 0.0
    max_dd: float = 0.0

    # A-version flags
    dca_locked: bool = False
    trailing_started: bool = False
    trailing_entries: int = 0
    tp1_first_holding_day: int = 0
    cycle_max_return: float = np.nan

    # realized totals
    cycle_buy_total: float = 0.0
    cycle_sell_total: float = 0.0

    # DCA control (cycle-level)
    dca_add_count: int = 0
    last_dca_holding_day: int = 0  # 0 means "never"

    legs: list[Leg] = None  # type: ignore

    def equity(self, prices: dict[str, float]) -> float:
        v = 0.0
        if self.legs:
            for leg in self.legs:
                px = prices.get(leg.ticker, np.nan)
                if np.isfinite(px):
                    v += leg.value(float(px))
        return float(self.seed) + float(v)

    def update_dd(self, prices: dict[str, float]) -> None:
        eq = self.equity(prices)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd

    def update_lev(self, max_cap: float) -> None:
        if self.entry_seed <= 0:
            return
        lev = max(0.0, -self.seed) / self.entry_seed
        if lev > self.max_leverage_pct:
            self.max_leverage_pct = float(lev)
        if self.max_leverage_pct > max_cap + 1e-9:
            self.max_leverage_pct = float(max_cap)

    def update_cycle_max_return(self, prices_close: dict[str, float]) -> None:
        if not self.in_cycle or self.entry_seed <= 0:
            return
        eq = self.equity(prices_close)
        r = (eq - self.entry_seed) / self.entry_seed
        if not np.isfinite(self.cycle_max_return):
            self.cycle_max_return = float(r)
        else:
            self.cycle_max_return = float(max(self.cycle_max_return, r))


def parse_weights(weights: str, topk: int) -> list[float]:
    parts = [p.strip() for p in str(weights).split(",") if p.strip()]
    ws = [float(p) for p in parts]
    if len(ws) != topk:
        raise ValueError(f"--weights must have {topk} numbers (got {len(ws)}): {weights}")
    s = sum(ws)
    if s <= 0:
        raise ValueError("--weights sum must be > 0")
    return [w / s for w in ws]


def compute_cycle_return_today(st: CycleState, day_close: dict[str, float]) -> float:
    rets = []
    for leg in st.legs:
        if leg.shares <= 0:
            continue
        px = day_close.get(leg.ticker, np.nan)
        avg = leg.avg_price()
        if np.isfinite(px) and np.isfinite(avg) and avg > 0:
            rets.append((px - avg) / avg)
    return float(np.mean(rets)) if rets else float("nan")


def load_feat_map(features_parq: str, features_csv: str) -> pd.DataFrame:
    p = Path(features_parq)
    c = Path(features_csv)
    if not p.exists() and not c.exists():
        return pd.DataFrame()
    df = read_table(str(p), str(c)).copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        return pd.DataFrame()
    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    keep = [x for x in ["Date", "Ticker", "p_success", "p_tail", "tau_H", "tau_class"] if x in df.columns]
    df = df[keep].dropna(subset=["Date", "Ticker"]).drop_duplicates(subset=["Date", "Ticker"]).reset_index(drop=True)
    df = df.set_index(["Date", "Ticker"], drop=True)
    return df


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Single-cycle engine with TopK(1~2), TP1 partial, trailing stop, leverage cap on ALL buys.\n"
            "Workflow-compat hardened version.\n"
        )
    )

    ap.add_argument("--picks-path", required=True, type=str, help="CSV with columns Date,Ticker (TopK rows/day).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--features-scored-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--features-scored-csv", default="data/features/features_scored.csv", type=str)
    # ✅ optional explicit features path (parquet/csv)
    ap.add_argument("--features-path", default="", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)  # kept for metadata compatibility
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--max-leverage-pct", default=1.0, type=float)

    ap.add_argument("--enable-trailing", default="true", type=str)
    ap.add_argument("--tp1-frac", default=0.50, type=float)
    ap.add_argument("--trail-stop", default=0.10, type=float)

    # compat only
    ap.add_argument("--tp1-trail-unlimited", default="true", type=str)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--weights", default="1.0", type=str)

    ap.add_argument("--tp1-hold-cap", default="none", type=str, choices=["none", "h2", "total"])
    ap.add_argument("--tau-h-map", default="30,40,50", type=str)

    # ✅ workflow compat flags
    ap.add_argument("--use-tau-h", default="true", type=str)
    ap.add_argument("--enable-dca", default="true", type=str)
    ap.add_argument("--dca-max-adds", default=0, type=int, help="0 means unlimited (cycle-level DCA events)")
    ap.add_argument("--dca-gap-days", default=1, type=int, help="minimum holding-day gap between DCA events")
    ap.add_argument("--dca-trigger", default="legacy", type=str, help="currently only 'legacy' is supported")
    ap.add_argument("--dca-add-frac", default=1.0, type=float, help="scale factor applied to DCA daily budget")

    # re-eval thresholds
    ap.add_argument("--reval-ps-strong", default=0.70, type=float)
    ap.add_argument("--reval-pt-strong", default=0.20, type=float)
    ap.add_argument("--reval-ps-pass", default=0.60, type=float)
    ap.add_argument("--reval-pt-pass", default=0.35, type=float)

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    # ✅ NEW: fail-fast guard (like predict_gate)
    ap.add_argument("--require-files", default="", type=str, help="comma-separated must-exist files; missing -> exit 2")

    args = ap.parse_args()

    _failfast_require_files(args.require_files)

    enable_trailing = _parse_bool(args.enable_trailing)
    _ = _parse_bool(args.tp1_trail_unlimited)  # compat only
    use_tau_h = _parse_bool(args.use_tau_h)
    enable_dca = _parse_bool(args.enable_dca)

    dca_max_adds = int(args.dca_max_adds)
    dca_gap_days = int(max(1, args.dca_gap_days))
    dca_add_frac = float(args.dca_add_frac)
    dca_trigger = str(args.dca_trigger or "legacy").strip().lower()
    if dca_trigger != "legacy":
        print(f"[WARN] unsupported --dca-trigger={dca_trigger}. Falling back to 'legacy'.")
        dca_trigger = "legacy"

    hmap = parse_tau_h_map(args.tau_h_map)

    topk = int(args.topk)
    if topk < 1 or topk > 2:
        raise ValueError("--topk should be 1 or 2")
    weights = parse_weights(args.weights, topk=topk)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # features map
    if args.features_path:
        fp = Path(args.features_path)
        if not fp.exists():
            raise FileNotFoundError(f"--features-path not found: {fp}")
        if fp.suffix.lower() == ".parquet":
            tmp = pd.read_parquet(fp)
        else:
            tmp = pd.read_csv(fp)
        # write temp into map loader form
        if "Date" not in tmp.columns or "Ticker" not in tmp.columns:
            feat_map = pd.DataFrame()
        else:
            tmp = tmp.copy()
            tmp["Date"] = _norm_date(tmp["Date"])
            tmp["Ticker"] = tmp["Ticker"].astype(str).str.upper().str.strip()
            keep = [x for x in ["Date", "Ticker", "p_success", "p_tail", "tau_H", "tau_class"] if x in tmp.columns]
            tmp = tmp[keep].dropna(subset=["Date", "Ticker"]).drop_duplicates(subset=["Date", "Ticker"]).reset_index(drop=True)
            feat_map = tmp.set_index(["Date", "Ticker"], drop=True)
    else:
        feat_map = load_feat_map(args.features_scored_parq, args.features_scored_csv)

    picks_path = Path(args.picks_path)
    if not picks_path.exists():
        raise FileNotFoundError(f"Missing picks file: {picks_path}")

    picks = pd.read_csv(picks_path) if picks_path.suffix.lower() != ".parquet" else pd.read_parquet(picks_path)
    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date,Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)
    picks = picks.groupby("Date", group_keys=False).head(topk).reset_index(drop=True)

    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError("prices must have Date,Ticker")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    picks_by_date: dict[pd.Timestamp, list[str]] = {}
    for d, g in picks.groupby("Date"):
        picks_by_date[d] = g["Ticker"].tolist()

    grouped = prices.groupby("Date", sort=True)
    all_dates = list(grouped.groups.keys())
    last_date = all_dates[-1] if all_dates else None

    st = CycleState(
        seed=float(args.initial_seed),
        max_equity=float(args.initial_seed),
        max_dd=0.0,
        legs=[]
    )

    cooldown_today = False
    trades: list[dict] = []
    curve: list[dict] = []

    def liquidate_all_legs(day_prices_close: dict[str, float]) -> float:
        proceeds = 0.0
        for leg in st.legs:
            px = float(day_prices_close.get(leg.ticker, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            if leg.shares > 0:
                proceeds += leg.shares * px
                leg.shares = 0.0
                leg.invested = 0.0
        return float(proceeds)

    def close_cycle(exit_date: pd.Timestamp, day_prices_close: dict[str, float], reason: str) -> None:
        nonlocal cooldown_today, st, trades

        liq_proceeds = liquidate_all_legs(day_prices_close)
        st.cycle_sell_total += float(liq_proceeds)

        invested_total = float(st.cycle_buy_total)
        proceeds_total = float(st.cycle_sell_total)

        cycle_return = (proceeds_total - invested_total) / invested_total if invested_total > 0 else 0.0
        win = int(cycle_return > 0)
        cmr = float(st.cycle_max_return) if np.isfinite(st.cycle_max_return) else np.nan

        H_half = int(max(1, int(st.H) // 2)) if int(st.H) > 0 else 0
        total_cap = int(st.H + H_half) if int(st.H) > 0 else 0
        tp1_cap = int(H_half) if int(st.H) > 0 else 0

        trades.append({
            "CycleType": "MAIN",
            "EntryDate": st.entry_date,
            "ExitDate": exit_date,
            "Tickers": ",".join([l.ticker for l in st.legs]),
            "Weights": ",".join([f"{l.weight:.4f}" for l in st.legs]),
            "EntrySeed": st.entry_seed,
            "H": int(st.H),
            "ProfitTarget": float(args.profit_target),
            "TP1_Frac": float(args.tp1_frac),
            "TrailStop": float(args.trail_stop) if enable_trailing else np.nan,
            "MaxDaysInput": int(args.max_days),
            "StopLevelInput": float(args.stop_level),
            "MaxExtendDaysInput": int(args.max_extend_days),

            "TP1HoldCapMode": str(args.tp1_hold_cap),
            "TP1HoldCapDays": tp1_cap,
            "TotalHoldCapDays": total_cap,
            "TP1FirstHoldingDay": int(st.tp1_first_holding_day),

            "TauHMap": ",".join([str(x) for x in hmap]),
            "UseTauH": int(use_tau_h),

            "GraceDays": int(st.grace_days_total),
            "RevalStrength": st.reval_strength,
            "RecoveryThreshold": float(st.recovery_threshold) if np.isfinite(st.recovery_threshold) else np.nan,
            "MaxLeveragePctCap": float(args.max_leverage_pct),
            "MaxLeveragePct": float(st.max_leverage_pct),

            "Invested": invested_total,
            "Proceeds": proceeds_total,
            "CycleReturn": float(cycle_return),

            "CycleMaxReturn": cmr,
            "HoldingDays": st.holding_days,
            "Extending": int(st.extending),
            "Reason": reason,
            "MaxDrawdown": st.max_dd,
            "Win": win,

            "TrailingEntries": int(st.trailing_entries),
            "DcaLocked": int(st.dca_locked),

            "EnableDCA": int(enable_dca),
            "DcaMaxAdds": int(dca_max_adds),
            "DcaGapDays": int(dca_gap_days),
            "DcaAddFrac": float(dca_add_frac),
            "DcaAddsUsed": int(st.dca_add_count),
        })

        st.seed += float(liq_proceeds)

        # reset cycle core
        st.in_cycle = False
        st.entry_date = None
        st.entry_seed = 0.0
        st.H = 0
        st.unit = 0.0
        st.holding_days = 0

        st.pending_reval = False
        st.ret_H = np.nan
        st.extending = False
        st.grace_days_total = 0
        st.grace_days_left = 0
        st.recovery_threshold = np.nan
        st.reval_strength = ""
        st.reval_ps = np.nan
        st.reval_pt = np.nan

        st.max_leverage_pct = 0.0
        st.legs = []

        st.dca_locked = False
        st.trailing_started = False
        st.trailing_entries = 0
        st.tp1_first_holding_day = 0
        st.cycle_max_return = np.nan

        st.cycle_buy_total = 0.0
        st.cycle_sell_total = 0.0

        st.dca_add_count = 0
        st.last_dca_holding_day = 0

        cooldown_today = True

    def reval_strength_for_day(date: pd.Timestamp) -> tuple[str, float, float]:
        if feat_map.empty:
            return "FAIL", 0.0, 1.0

        ps = []
        pt = []
        for leg in st.legs:
            try:
                row = feat_map.loc[(date, leg.ticker)]
            except Exception:
                continue
            if "p_success" in row.index:
                ps.append(float(row["p_success"]))
            if "p_tail" in row.index:
                pt.append(float(row["p_tail"]))

        if not ps or not pt:
            return "FAIL", float(np.mean(ps)) if ps else 0.0, float(np.mean(pt)) if pt else 1.0

        ps_m = float(np.mean(ps))
        pt_m = float(np.mean(pt))

        if ps_m >= float(args.reval_ps_strong) and pt_m <= float(args.reval_pt_strong):
            return "STRONG", ps_m, pt_m
        if ps_m >= float(args.reval_ps_pass) and pt_m <= float(args.reval_pt_pass):
            return "WEAK", ps_m, pt_m
        return "FAIL", ps_m, pt_m

    def should_force_close_after_tp1() -> tuple[bool, str]:
        if not st.in_cycle:
            return False, ""
        tp1_hold_cap = str(args.tp1_hold_cap).lower().strip()
        if tp1_hold_cap == "none":
            return False, ""
        if st.tp1_first_holding_day <= 0:
            return False, ""

        H = int(st.H)
        if H <= 0:
            return False, ""

        H_half = int(max(1, H // 2))

        if tp1_hold_cap == "h2":
            cap_day = int(st.tp1_first_holding_day + H_half)
            if int(st.holding_days) >= cap_day:
                return True, "TP1_H2_CAP_EXIT"
            return False, ""

        if tp1_hold_cap == "total":
            cap_day = int(H + H_half)
            if int(st.holding_days) >= cap_day:
                return True, "TP1_TOTAL_CAP_EXIT"
            return False, ""

        return False, ""

    # simulate
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        day_prices_close: dict[str, float] = {}
        day_prices_high: dict[str, float] = {}
        day_prices_low: dict[str, float] = {}

        for t in day_df.index:
            r = day_df.loc[t]
            day_prices_close[t] = float(r["Close"])
            day_prices_high[t] = float(r["High"])
            day_prices_low[t] = float(r["Low"])

        # ----- in cycle: update
        if st.in_cycle:
            st.holding_days += 1

            # (0) pending re-eval executes on T+1 close
            if st.pending_reval:
                strength, ps_m, pt_m = reval_strength_for_day(date)
                st.reval_strength = strength
                st.reval_ps = float(ps_m)
                st.reval_pt = float(pt_m)

                if strength == "FAIL":
                    close_cycle(date, day_prices_close, reason=f"REVAL_FAIL(ps={ps_m:.3f},pt={pt_m:.3f})")
                else:
                    st.extending = True
                    st.pending_reval = False

                    st.grace_days_total = int(min(max(1, st.H // 4), 15))
                    st.grace_days_left = int(st.grace_days_total)

                    base = float(st.ret_H) + 0.05 if np.isfinite(st.ret_H) else -0.05
                    if strength == "STRONG":
                        st.recovery_threshold = float(max(base, -0.05))
                    else:
                        st.recovery_threshold = float(min(base, -0.05))

            if st.in_cycle:
                # 1) TP1 + trailing per leg
                if enable_trailing:
                    for leg in st.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        high_px = day_prices_high[leg.ticker]
                        low_px = day_prices_low[leg.ticker]
                        avg = leg.avg_price()

                        if (not leg.tp1_done) and np.isfinite(avg) and high_px >= avg * (1.0 + float(args.profit_target)):
                            tp_px = avg * (1.0 + float(args.profit_target))
                            sell_shares = leg.shares * float(args.tp1_frac)
                            sell_shares = float(min(leg.shares, max(0.0, sell_shares)))
                            proceeds = sell_shares * tp_px

                            before_shares = leg.shares
                            leg.shares -= sell_shares
                            if before_shares > 0:
                                leg.invested *= (leg.shares / before_shares)
                            else:
                                leg.invested = 0.0

                            st.seed += proceeds
                            st.cycle_sell_total += float(proceeds)

                            leg.tp1_done = True
                            leg.peak = float(max(high_px, tp_px))

                            if not st.trailing_started:
                                st.trailing_started = True
                                st.trailing_entries = 1
                                st.tp1_first_holding_day = int(st.holding_days)

                            st.dca_locked = True
                            st.update_lev(float(args.max_leverage_pct))

                        if leg.tp1_done and leg.shares > 0:
                            if high_px > leg.peak:
                                leg.peak = float(high_px)
                            stop_px = leg.peak * (1.0 - float(args.trail_stop))
                            if np.isfinite(low_px) and low_px <= stop_px:
                                proceeds = leg.shares * stop_px
                                st.seed += proceeds
                                st.cycle_sell_total += float(proceeds)
                                leg.shares = 0.0
                                leg.invested = 0.0

                # 2) if all legs emptied -> close
                if st.in_cycle and all((leg.shares <= 0 for leg in st.legs)):
                    close_cycle(date, day_prices_close, reason="TRAIL_EXIT_ALL")

            # 2.5) TP1-period cap
            if st.in_cycle:
                force, why = should_force_close_after_tp1()
                if force:
                    close_cycle(date, day_prices_close, reason=why)

            # 3) H reached: set pending_reval only if no TP1 anywhere
            if st.in_cycle and (not st.extending) and (not st.pending_reval):
                any_tp1 = any((leg.tp1_done for leg in st.legs))
                if (not any_tp1) and (st.H > 0) and (st.holding_days >= st.H):
                    st.ret_H = compute_cycle_return_today(st, day_prices_close)
                    st.pending_reval = True

            # 4) Grace mode (no TP1 case only)
            if st.in_cycle and st.extending:
                any_tp1 = any((leg.tp1_done for leg in st.legs))
                if not any_tp1:
                    st.grace_days_left = max(0, int(st.grace_days_left) - 1)

                    thr = float(st.recovery_threshold) if np.isfinite(st.recovery_threshold) else -0.05
                    hit = False
                    for leg in st.legs:
                        if leg.shares <= 0 or leg.tp1_done:
                            continue
                        px = day_prices_close.get(leg.ticker, np.nan)
                        avg = leg.avg_price()
                        if np.isfinite(px) and np.isfinite(avg) and avg > 0:
                            r = (px - avg) / avg
                            if r >= thr:
                                hit = True
                                break

                    if hit:
                        close_cycle(
                            date,
                            day_prices_close,
                            reason=f"GRACE_RECOVERY_EXIT(str={st.reval_strength},ps={st.reval_ps:.3f},pt={st.reval_pt:.3f})",
                        )
                    else:
                        if st.in_cycle and st.grace_days_left <= 0:
                            close_cycle(
                                date,
                                day_prices_close,
                                reason=f"GRACE_END_EXIT(str={st.reval_strength},ps={st.reval_ps:.3f},pt={st.reval_pt:.3f})",
                            )

                if st.in_cycle:
                    st.update_dd(day_prices_close)
                    st.update_cycle_max_return(day_prices_close)

            # 5) Normal mode DCA (optional)
            if st.in_cycle and (not st.extending) and (not st.pending_reval):
                if enable_dca and (not st.dca_locked):
                    if dca_max_adds > 0 and st.dca_add_count >= dca_max_adds:
                        pass
                    else:
                        if st.last_dca_holding_day > 0 and (st.holding_days - st.last_dca_holding_day) < dca_gap_days:
                            pass
                        else:
                            desired_total = float(st.unit) * float(max(0.0, dca_add_frac))
                            any_invest = False

                            for leg in st.legs:
                                if leg.ticker not in day_df.index:
                                    continue
                                close_px = day_prices_close[leg.ticker]
                                if not np.isfinite(close_px) or close_px <= 0:
                                    continue

                                avg = leg.avg_price()
                                desired_leg = desired_total * float(leg.weight)

                                if np.isfinite(avg) and avg > 0:
                                    if close_px <= avg:
                                        pass
                                    elif close_px <= avg * 1.05:
                                        desired_leg = desired_leg / 2.0
                                    else:
                                        desired_leg = 0.0

                                invest = clamp_invest_by_leverage(
                                    st.seed, st.entry_seed, desired_leg, float(args.max_leverage_pct)
                                )
                                if invest > 0:
                                    st.seed -= invest
                                    st.cycle_buy_total += float(invest)
                                    leg.invested += invest
                                    leg.shares += invest / close_px
                                    st.update_lev(float(args.max_leverage_pct))
                                    any_invest = True

                            if any_invest:
                                st.dca_add_count += 1
                                st.last_dca_holding_day = int(st.holding_days)

                st.update_dd(day_prices_close)
                st.update_cycle_max_return(day_prices_close)

        # ----- entry: not in cycle and not cooldown day
        if (not st.in_cycle) and (not cooldown_today):
            picks_today = picks_by_date.get(date, [])
            if picks_today:
                valid = [t for t in picks_today if t in day_df.index and np.isfinite(day_prices_close.get(t, np.nan))]
                if len(valid) >= 1:
                    chosen = valid[:topk]

                    if not use_tau_h:
                        H_eff = int(max(1, int(args.max_days)))
                    else:
                        Hs: list[int] = []
                        if not feat_map.empty:
                            for t in chosen:
                                try:
                                    row = feat_map.loc[(date, t)]
                                    if "tau_H" in row.index:
                                        hh = int(row["tau_H"])
                                        if hh > 0:
                                            Hs.append(hh)
                                    elif "tau_class" in row.index:
                                        cls = int(row["tau_class"])
                                        hh = int(class_to_h(cls, hmap))
                                        if hh > 0:
                                            Hs.append(hh)
                                except Exception:
                                    pass

                        fallback_H = int(hmap[1] if len(hmap) >= 2 else (hmap[0] if hmap else 40))
                        H_eff = int(max(Hs)) if Hs else int(fallback_H)
                        H_eff = int(max(1, H_eff))

                    st.cycle_buy_total = 0.0
                    st.cycle_sell_total = 0.0
                    st.dca_add_count = 0
                    st.last_dca_holding_day = 0

                    S0 = float(st.seed)
                    unit = (S0 / float(H_eff)) if H_eff > 0 else 0.0

                    legs = [Leg(ticker=t, weight=float(weights[i])) for i, t in enumerate(chosen)]

                    invested_total = 0.0
                    for leg in legs:
                        px = day_prices_close[leg.ticker]
                        desired = float(unit) * float(leg.weight)
                        invest = clamp_invest_by_leverage(st.seed, S0, desired, float(args.max_leverage_pct))
                        if invest > 0:
                            st.seed -= invest
                            st.cycle_buy_total += float(invest)
                            leg.invested += invest
                            leg.shares += invest / px
                            invested_total += invest

                    if invested_total > 0:
                        st.in_cycle = True
                        st.entry_date = date
                        st.entry_seed = S0
                        st.H = int(H_eff)
                        st.unit = float(unit)
                        st.holding_days = 1

                        st.pending_reval = False
                        st.ret_H = np.nan
                        st.extending = False
                        st.grace_days_total = 0
                        st.grace_days_left = 0
                        st.recovery_threshold = np.nan
                        st.reval_strength = ""
                        st.reval_ps = np.nan
                        st.reval_pt = np.nan

                        st.max_leverage_pct = 0.0
                        st.legs = legs

                        st.dca_locked = False
                        st.trailing_started = False
                        st.trailing_entries = 0
                        st.tp1_first_holding_day = 0
                        st.cycle_max_return = np.nan

                        st.update_lev(float(args.max_leverage_pct))
                        st.update_dd(day_prices_close)
                        st.update_cycle_max_return(day_prices_close)

        # ----- record curve
        prices_for_eq = day_prices_close if st.in_cycle else {}
        eq = st.equity(prices_for_eq)

        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": st.seed,
            "InCycle": int(st.in_cycle),
            "Tickers": ",".join([l.ticker for l in st.legs]) if st.in_cycle else "",
            "HoldingDays": st.holding_days if st.in_cycle else 0,
            "H": int(st.H) if st.in_cycle else 0,
            "PendingReval": int(st.pending_reval),
            "Extending": int(st.extending),
            "GraceLeft": int(st.grace_days_left),
            "RevalStrength": st.reval_strength,
            "RecoveryThreshold": float(st.recovery_threshold) if np.isfinite(st.recovery_threshold) else np.nan,
            "MaxLeveragePctCycle": st.max_leverage_pct if st.in_cycle else 0.0,
            "MaxDrawdownPortfolio": st.max_dd,

            "DcaLocked": int(st.dca_locked) if st.in_cycle else 0,
            "TrailingEntriesCycle": int(st.trailing_entries) if st.in_cycle else 0,
            "TP1FirstHoldingDay": int(st.tp1_first_holding_day) if st.in_cycle else 0,
            "TP1HoldCapMode": str(args.tp1_hold_cap) if st.in_cycle else "",

            "CycleMaxReturnSoFar": float(st.cycle_max_return) if (st.in_cycle and np.isfinite(st.cycle_max_return)) else np.nan,

            "CycleBuyTotal": float(st.cycle_buy_total) if st.in_cycle else 0.0,
            "CycleSellTotal": float(st.cycle_sell_total) if st.in_cycle else 0.0,

            "UseTauH": int(use_tau_h),
            "EnableDCA": int(enable_dca),
            "DcaAddsUsed": int(st.dca_add_count) if st.in_cycle else 0,
        })

        # ----- force close on last day
        if last_date is not None and date == last_date and st.in_cycle:
            close_cycle(date, day_prices_close, reason="FINAL_CLOSE")

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)
    if not curve_df.empty:
        curve_df["SeedMultiple"] = curve_df["Equity"] / float(args.initial_seed)

    tag = args.tag if args.tag else "run"
    suffix = args.suffix if args.suffix else picks_path.stem.replace("picks_", "")

    trades_path = Path(args.out_dir) / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
    curve_path = Path(args.out_dir) / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"

    trades_df.to_parquet(trades_path, index=False)
    curve_df.to_parquet(curve_path, index=False)

    trades_csv_path = trades_path.with_suffix(".csv")
    curve_csv_path = curve_path.with_suffix(".csv")
    trades_df.to_csv(trades_csv_path, index=False)
    curve_df.to_csv(curve_csv_path, index=False)

    print(f"[DONE] wrote trades: {trades_path} rows={len(trades_df)}")
    print(f"[DONE] wrote trades: {trades_csv_path} rows={len(trades_df)}")
    print(f"[DONE] wrote curve : {curve_path} rows={len(curve_df)}")
    print(f"[DONE] wrote curve : {curve_csv_path} rows={len(curve_df)}")
    if not curve_df.empty:
        last_sm = float(curve_df["SeedMultiple"].iloc[-1])
        print(f"[INFO] final SeedMultiple={last_sm:.4f} maxDD={float(st.max_dd):.4f}")


if __name__ == "__main__":
    main()