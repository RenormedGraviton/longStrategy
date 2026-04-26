"""Backtest — pure online, no caching.

Mirrors what production would do per tick: open files, slice the past
N days, compute features, predict. No pre-load, no in-memory archive.

Structure:
    for each hour ts in [start, end):
        for each symbol:
            load past `lookback_hours` from disk
            compute features
            apply 3-day hard filter
            if pred >= threshold and not held: BUY
            if held: walk closes, check stop_loss / trail_stop / time_exit

CLI: python squeeze_pump_v4b/backtest.py
All knobs in config.yaml under `backtest:`.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yaml

from core import (LiveLoader, PumpModel, compute_features, load_config,
                  process_tick)

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
_OUT_COLS = ["ticker", "datetime", "long", "short", "model_pred",
             "side", "exit_reason", "entry_close", "exit_close", "ret"]


# ── replay-mode shadow folder helpers ───────────────────────────────
# Production keeps a SLIDING WINDOW per (sym, tag) folder: the past
# `_KEEP_PAST_DAYS` closed days as YYYY-MM-DD.csv plus today's partial
# bars as latest.csv. The shadow mirrors this exactly. At each UTC
# midnight, the oldest date-CSV rotates out, yesterday gets promoted
# from latest.csv to a date-CSV, and today's CSV becomes the new
# latest.csv.

_KEEP_PAST_DAYS = 4   # production's sliding window size


def _initial_shadow(orig_root: Path, shadow_root: Path, day0_str: str) -> None:
    """Mirror production at the START of `day0_str`:
        - past _KEEP_PAST_DAYS closed days as YYYY-MM-DD.csv (symlinks)
        - day0 as latest.csv (symlink)
       Future days don't exist on the prod fs yet — skipped here too."""
    shadow_root.mkdir(parents=True, exist_ok=True)
    day0_date = pd.Timestamp(day0_str).date()
    keep_dates = [(day0_date - pd.Timedelta(days=i)).isoformat()
                  for i in range(1, _KEEP_PAST_DAYS + 1)]

    for orig_sub in orig_root.iterdir():
        if not orig_sub.is_dir():
            continue
        shadow_sub = shadow_root / orig_sub.name
        shadow_sub.mkdir(exist_ok=True)
        # Past N closed days.
        for d in keep_dates:
            orig_f = orig_sub / f"{d}.csv"
            if orig_f.is_file():
                lnk = shadow_sub / orig_f.name
                if not lnk.exists() and not lnk.is_symlink():
                    lnk.symlink_to(orig_f.resolve())
        # Today as latest.csv.
        orig_today = orig_sub / f"{day0_str}.csv"
        if orig_today.is_file():
            latest = shadow_sub / "latest.csv"
            if not latest.exists() and not latest.is_symlink():
                latest.symlink_to(orig_today.resolve())


def _advance_shadow(orig_root: Path, shadow_root: Path,
                    prev_str: str, new_str: str) -> None:
    """Roll the shadow at the UTC day boundary:
        1. Delete the date-CSV that's now older than _KEEP_PAST_DAYS.
        2. Drop the old latest.csv symlink.
        3. Promote prev_str (yesterday) to a YYYY-MM-DD.csv symlink.
        4. Re-point latest.csv at today (new_str)."""
    new_date = pd.Timestamp(new_str).date()
    drop_str = (new_date - pd.Timedelta(days=_KEEP_PAST_DAYS + 1)).isoformat()

    for shadow_sub in shadow_root.iterdir():
        if not shadow_sub.is_dir():
            continue
        # 1. Drop the now-too-old date file.
        too_old = shadow_sub / f"{drop_str}.csv"
        if too_old.is_symlink() or too_old.exists():
            too_old.unlink()
        # 2. Remove old latest.csv.
        latest = shadow_sub / "latest.csv"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # 3. Promote yesterday → date-CSV.
        orig_prev = orig_root / shadow_sub.name / f"{prev_str}.csv"
        if orig_prev.is_file():
            shadow_prev = shadow_sub / f"{prev_str}.csv"
            if not shadow_prev.exists() and not shadow_prev.is_symlink():
                shadow_prev.symlink_to(orig_prev.resolve())
        # 4. New latest.csv = today.
        orig_new = orig_root / shadow_sub.name / f"{new_str}.csv"
        if orig_new.is_file():
            latest.symlink_to(orig_new.resolve())


def _row_from_signal(s: dict) -> dict:
    """Flatten a process_tick BUY/SELL dict into the unified CSV schema."""
    out = {k: "" for k in _OUT_COLS}
    out["ticker"] = s.get("symbol", "")
    if s.get("side") == "BUY":
        out.update({
            "datetime": s["entry_ts"],
            "long": 1, "short": 0,
            "model_pred": s["pred_xgb"], "side": "BUY",
            "entry_close": s["entry_close"],
        })
    else:  # SELL
        out.update({
            "datetime": s["exit_ts"],
            "long": 0, "short": -1, "model_pred": "",
            "side": "SELL", "exit_reason": s["exit_reason"],
            "entry_close": s["entry_close"],
            "exit_close": s["exit_close"], "ret": s["exit_ret"],
        })
    return out


def _walk_exit(pos: dict, closes: pd.Series, now_ts: pd.Timestamp,
               trail_pct: float, stop_pct: float, max_hold_h: int):
    entry_ts = pos["entry_ts"]
    time_limit = entry_ts + pd.Timedelta(hours=max_hold_h)
    window_end = min(now_ts, time_limit)
    try:
        window = closes.loc[entry_ts + pd.Timedelta(hours=1):window_end]
    except KeyError:
        return None
    peak = pos["peak"]
    stop_price = pos["entry_close"] * (1.0 - stop_pct)
    for sub_ts, p in window.items():
        if not np.isfinite(p):
            continue
        peak = max(peak, p)
        if p <= stop_price:
            return "stop_loss", sub_ts, float(p)
        if p <= peak * (1.0 - trail_pct):
            return "trail_stop", sub_ts, float(p)
        if sub_ts >= time_limit:
            return "time_exit", sub_ts, float(p)
    pos["peak"] = peak
    return None


def _run_replay(data_dir: Path, model, cfg, start_ts, end_ts, n_ticks,
                out_path: Path, log) -> None:
    """Replay the entire window through the LIVE code path
    (`core.process_tick`). Today's bars are exposed via `latest.csv` in a
    temp shadow folder so the production-style file layout is exercised.
    State is persisted to cfg.state_path each tick (matches run_live)."""
    cfg.max_wait_min = 0          # don't sleep on missing canary symbols
    if cfg.state_path.is_file():
        cfg.state_path.unlink()   # clean slate per replay run

    shadow_root = Path(tempfile.mkdtemp(prefix="sp_replay_shadow_"))
    log.info("replay shadow at %s", shadow_root)

    import csv
    out_f = open(out_path, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=_OUT_COLS)
    writer.writeheader()

    cur_day_str: str | None = None
    n_buys = n_sells = n_timeout = 0
    t0 = time.time()
    tick_i = 0
    try:
        ts = start_ts
        while ts < end_ts:
            day_str = ts.normalize().date().isoformat()
            if cur_day_str is None:
                log.info("building initial shadow for %s ...", day_str)
                _initial_shadow(data_dir, shadow_root, day_str)
            elif day_str != cur_day_str:
                _advance_shadow(data_dir, shadow_root, cur_day_str, day_str)
            cur_day_str = day_str

            loader = LiveLoader(shadow_root, cfg.known_tags)
            # ↓ EXACTLY the call run_live.py makes every tick.
            tick_t0 = time.time()
            result = process_tick(loader, model, cfg, ts)
            tick_dt = time.time() - tick_t0
            if result.get("data_status") == "timeout":
                n_timeout += 1
            for s in result.get("close", []):
                writer.writerow(_row_from_signal(s)); n_sells += 1
            for s in result.get("open", []):
                writer.writerow(_row_from_signal(s)); n_buys += 1

            tick_i += 1
            el = time.time() - t0
            rate = tick_i / el if el else 0
            eta = (n_ticks - tick_i) / rate if rate else float("inf")
            tick_buys = len(result.get("open", []))
            tick_sells = len(result.get("close", []))
            log.info(
                "  %s  tick %d/%d  hour_took=%.2fs  rate=%.2f t/s  ETA=%.0fs  "
                "this_tick: buy=%d sell=%d  totals: buys=%d sells=%d timeout=%d open=%d  "
                "data_status=%s",
                ts.isoformat(), tick_i, n_ticks, tick_dt, rate, eta,
                tick_buys, tick_sells, n_buys, n_sells, n_timeout,
                result.get("open_positions", 0),
                result.get("data_status", ""),
            )
            ts += pd.Timedelta(hours=1)
    finally:
        out_f.close()
        shutil.rmtree(shadow_root, ignore_errors=True)

    log.info("replay done — buys=%d sells=%d timeout_ticks=%d → %s   total %.1fs",
             n_buys, n_sells, n_timeout, out_path, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(Path(__file__).resolve().parent / "config.yaml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("squeeze_pump_v4b.bt")

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    raw_yaml = yaml.safe_load(cfg_path.read_text())
    bt = raw_yaml["backtest"]
    yaml_dir = cfg_path.parent
    data_dir = (yaml_dir / bt["old_data_folder"]).resolve()
    out_path = (yaml_dir / bt["out"]).resolve()
    max_symbols = int(bt.get("max_symbols", 0)) or None
    mode = str(bt.get("mode", "default")).lower()
    if mode not in ("default", "replay"):
        raise SystemExit(f"backtest.mode must be 'default' or 'replay', got {mode!r}")

    start_ts = pd.Timestamp(str(bt["start"]), tz="UTC").normalize()
    end_ts   = pd.Timestamp(str(bt["end"]),   tz="UTC").normalize() + pd.Timedelta(days=1)
    log.info("backtest mode=%s  %s → %s  data=%s",
             mode, start_ts, end_ts, data_dir)

    cfg.live_data_path = data_dir          # default-mode loader root
    model = PumpModel(cfg.model_path)
    n_ticks = int((end_ts - start_ts) / pd.Timedelta(hours=1))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "replay":
        return _run_replay(data_dir, model, cfg, start_ts, end_ts, n_ticks,
                           out_path, log)

    # ── default mode: simple online loop ─────────────────────────────
    loader = LiveLoader(data_dir, cfg.known_tags)
    symbols = loader.list_symbols(cfg.primary_tag)
    if max_symbols:
        symbols = symbols[:max_symbols]
    log.info("default mode — running on %d symbols", len(symbols))

    records: list[dict] = []
    positions: dict[str, dict] = {}
    last_buy_ts: dict[str, pd.Timestamp] = {}   # 4h re-entry cooldown
    cooldown = pd.Timedelta(hours=4)
    cur = start_ts
    t0 = time.time()
    tick_i = 0
    while cur < end_ts:
        # ── exit pass ─────────────────────────────────────────────────
        for sym in list(positions):
            bars = loader.load(sym, cfg.primary_tag, cur, cfg.lookback_hours)
            if bars.empty:
                continue
            closes = bars.set_index("ts")["close"]
            res = _walk_exit(positions[sym], closes, cur,
                             cfg.trail_pct, cfg.stop_loss_pct, cfg.max_hold_hours)
            if res is None:
                continue
            reason, exit_ts, exit_close = res
            pos = positions.pop(sym)
            records.append({
                "ticker": sym, "datetime": exit_ts.isoformat(),
                "long": 0, "short": -1, "model_pred": pos["entry_pred"],
                "side": "SELL", "exit_reason": reason,
                "entry_close": pos["entry_close"], "exit_close": exit_close,
                "ret": exit_close / pos["entry_close"] - 1.0,
            })

        # ── entry pass ────────────────────────────────────────────────
        # Per-symbol: load bundle → features → 3-day filter → predict.
        btc = None
        if cfg.include_btc_features:
            btc = loader.load(cfg.btc_symbol, cfg.btc_source_tag,
                              cur, cfg.lookback_hours)
        for sym in symbols:
            if sym in positions:
                continue
            # 4h cooldown: even if the prior long has already closed, no
            # re-entry until 4 hours after the most recent BUY for this symbol.
            last_buy = last_buy_ts.get(sym)
            if last_buy is not None and cur - last_buy < cooldown:
                continue
            bundle = loader.load_bundle(sym, cur, cfg.lookback_hours)
            if cfg.primary_tag not in bundle:
                continue
            feats = compute_features(bundle, btc, sym, cfg)
            if feats.empty:
                continue
            row_df = feats[feats["ts"] == cur]
            if row_df.empty:
                continue
            row = row_df.iloc[-1]
            if not (row["oi_coin_3d_chg_pct"] > cfg.oi_coin_3d_chg_pct_min
                    and row["ret_3d_pct"]      > cfg.ret_3d_pct_min):
                continue
            pred = float(model.predict_proba(row_df, cfg.features)[-1])
            if pred < cfg.entry_threshold:
                continue
            positions[sym] = {
                "entry_ts": cur, "entry_close": float(row["close"]),
                "peak": float(row["close"]), "entry_pred": pred,
            }
            last_buy_ts[sym] = cur
            records.append({
                "ticker": sym, "datetime": cur.isoformat(),
                "long": 1, "short": 0, "model_pred": pred,
                "side": "BUY", "exit_reason": "",
                "entry_close": float(row["close"]),
                "exit_close": np.nan, "ret": np.nan,
            })

        tick_i += 1
        if tick_i % 24 == 0:
            elapsed = time.time() - t0
            rate = tick_i / elapsed if elapsed else 0
            eta = (n_ticks - tick_i) / rate if rate else float("inf")
            log.info("  %s  tick %d/%d  rate=%.2f ticks/s  ETA=%.0fs  "
                     "open=%d  records=%d",
                     cur.isoformat(), tick_i, n_ticks, rate, eta,
                     len(positions), len(records))
        cur += pd.Timedelta(hours=1)

    out_df = pd.DataFrame(records, columns=_OUT_COLS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("done — buys=%d sells=%d still_open=%d → %s   total %.1fs",
             (out_df["side"] == "BUY").sum(),
             (out_df["side"] == "SELL").sum(),
             len(positions), out_path, time.time() - t0)


if __name__ == "__main__":
    main()
