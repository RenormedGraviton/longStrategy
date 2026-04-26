"""End-to-end smoke test for the consolidated layout.

1. Picks ~40 parquet-covered symbols.
2. Generates pseudo-live CSVs.
3. Trains a tiny XGBoost on the 25-feature (BTC-disabled) slice so we
   have a real model artifact to load.
4. Writes a temp config.yaml pointing at the pseudo-live folder.
5. Runs run_live.py --once on a 4h-aligned bar that passes the filter.
6. Runs backtest.py over a 5-day slice ending at the same bar.

Run directly:
    python /path/to/squeeze_pump_v4b/tests/smoke_test.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from core import (LiveLoader, compute_features, has_required_history,
                  load_config, passes_hard_filter)
from tools.make_pseudo_live import generate

REPO = Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO / "data"
SP_DIR = Path(__file__).resolve().parents[1]


def _pick_symbols(n: int = 40) -> list[str]:
    out = []
    for p in sorted(PARQUET_DIR.glob("*_binance_futures.parquet")):
        sym = p.stem.replace("_binance_futures", "")
        if (PARQUET_DIR / f"{sym}_binance_spot.parquet").is_file() \
           and (PARQUET_DIR / f"{sym}_bybit_perps.parquet").is_file():
            out.append(sym)
            if len(out) >= n:
                break
    return out


def _train_tiny_model(model_path: Path, syms: list[str], now_ts: pd.Timestamp,
                      cfg) -> None:
    rows = []
    for sym in syms:
        bundle = {}
        for pq in sorted(PARQUET_DIR.glob(f"{sym}_*.parquet")):
            tag = pq.stem[len(sym) + 1:]
            df = pd.read_parquet(pq)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df[df["ts"] <= now_ts]
            if not df.empty:
                bundle[tag] = df
        if cfg.primary_tag not in bundle:
            continue
        feats = compute_features(bundle, None, sym, cfg)
        if feats.empty:
            continue
        closes = feats["close"].values
        n = len(closes)
        label = np.zeros(n, dtype=int)
        h = 12  # 48h on a 4h grid
        for i in range(n - h):
            if closes[i] > 0 and closes[i + h] / closes[i] - 1 > 0.05:
                label[i] = 1
        feats["y"] = label
        rows.append(feats)
    if not rows:
        raise RuntimeError("no training rows")
    df = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    X = df[cfg.features].fillna(0).values
    y = df["y"].values
    print(f"  training rows: {len(df)}, positive rate: {y.mean():.3f}")
    clf = xgb.XGBClassifier(
        n_estimators=60, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(model_path))


def _scan_filter_passing(syms: list[str], start_ts: pd.Timestamp,
                         max_steps: int, cfg) -> pd.Timestamp | None:
    """Walk back from start_ts in 4h steps until at least one symbol's
    latest 4h-grid row passes the hard filter."""
    ts = start_ts
    for _ in range(max_steps):
        for sym in syms:
            bundle = {}
            for pq in sorted(PARQUET_DIR.glob(f"{sym}_*.parquet")):
                tag = pq.stem[len(sym) + 1:]
                if tag not in (cfg.primary_tag, "binance_spot", "bybit_perps"):
                    continue
                df = pd.read_parquet(pq)
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df = df[df["ts"] <= ts]
                if not df.empty:
                    bundle[tag] = df
            if cfg.primary_tag not in bundle:
                continue
            primary = bundle[cfg.primary_tag]
            if not has_required_history(primary, ts, cfg.required_history_days):
                continue
            feats = compute_features(bundle, None, sym, cfg)
            row = feats[feats["ts"] == ts]
            if row.empty:
                continue
            if passes_hard_filter(row.iloc[-1], cfg):
                return ts
        ts -= pd.Timedelta(hours=4)
    return None


def main() -> None:
    syms = _pick_symbols(40)
    if not syms:
        raise SystemExit("no symbols with full triple coverage in data/")
    print(f"smoke: {len(syms)} symbols")

    work = Path(tempfile.mkdtemp(prefix="sp_v4b_smoke_"))
    try:
        live = work / "live"
        cfg_path = work / "config.yaml"
        model_path = work / "model.json"
        state_path = work / "state" / "positions.json"
        log_path = work / "log" / "signals.jsonl"
        bt_out = work / "bt.csv"

        # Build a temp config (BTC disabled — no BTC parquet locally).
        cfg_dict = yaml.safe_load((SP_DIR / "config.yaml").read_text())
        cfg_dict["live_data_path"] = str(live)
        cfg_dict["model_path"] = str(model_path)
        cfg_dict["state_path"] = str(state_path)
        cfg_dict["log_path"] = str(log_path)
        cfg_dict["include_btc_features"] = False
        cfg_dict["entry_threshold"] = 0.05    # relaxed for smoke
        cfg_dict["freshness"]["max_wait_min"] = 0  # don't wait — we're offline
        cfg_path.write_text(yaml.safe_dump(cfg_dict))
        cfg = load_config(cfg_path)
        print(f"smoke: include_btc={cfg.include_btc_features}, "
              f"|features|={len(cfg.features)}, threshold={cfg.entry_threshold}")

        # Pick "now" = latest hour common across all selected symbols.
        max_ts = []
        for s in syms:
            df = pd.read_parquet(PARQUET_DIR / f"{s}_{cfg.primary_tag}.parquet",
                                 columns=["ts"])
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            max_ts.append(df["ts"].max())
        now_ts = min(max_ts).floor("h")
        while now_ts.hour % 4 != 0:
            now_ts -= pd.Timedelta(hours=1)
        print(f"smoke: initial now_ts = {now_ts.isoformat()}")

        # Scan back for a filter-passing bar so the open path is exercised.
        hit = _scan_filter_passing(syms, now_ts, max_steps=60 * 24 // 4, cfg=cfg)
        if hit is not None:
            now_ts = hit
            print(f"smoke: using {now_ts.isoformat()} (filter passed)")
        else:
            print("smoke: no filter-passing bar found in 60d — using initial")

        # 1) Pseudo-live data.
        print("\n1) Generating pseudo-live CSVs ...")
        generate(PARQUET_DIR, live, now_ts, syms,
                 [cfg.primary_tag, "binance_spot", "bybit_perps"])

        # 2) Train tiny model.
        print("\n2) Training tiny XGBoost ...")
        _train_tiny_model(model_path, syms, now_ts, cfg)

        # 3) run_live --once.
        print("\n3) python run_live.py --once --now ...")
        cmd = [sys.executable, str(SP_DIR / "run_live.py"),
               "--config", str(cfg_path), "--once",
               "--now", now_ts.isoformat()]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(textwrap.indent(out.stdout, "    "))
        if out.returncode != 0:
            print("STDERR:", out.stderr); raise SystemExit("run_live failed")

        # 4) backtest over the past 5d ending at now_ts.
        print("4) python backtest.py ...")
        bt_start = (now_ts - pd.Timedelta(days=5)).isoformat()
        bt_end = (now_ts + pd.Timedelta(hours=1)).isoformat()
        cmd = [sys.executable, str(SP_DIR / "backtest.py"),
               "--config", str(cfg_path),
               "--old-data-folder", str(live),
               "--start", bt_start, "--end", bt_end,
               "--out", str(bt_out), "--include-scored"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        for line in out.stdout.splitlines()[-3:] + out.stderr.splitlines()[-3:]:
            print("    " + line)
        if out.returncode != 0:
            print("STDERR:", out.stderr); raise SystemExit("backtest failed")
        bt_df = pd.read_csv(bt_out)
        print(f"\n   backtest CSV: {len(bt_df)} rows, sides={bt_df['side'].value_counts().to_dict()}")
        print(bt_df.head(10).to_string(index=False))

        print("\nSMOKE OK")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
