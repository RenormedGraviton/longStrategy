"""Generate pseudo-live CSVs from the parquet archive.

Reads `data/{RAWSYMBOL}_{TAG}.parquet` and writes the date-partitioned
layout the loader expects:

    OUT_DIR/{NORMSYMBOL}_{TAG}/
        YYYY-MM-DD.csv     # closed UTC days
        latest.csv         # partial current UTC day (only at --now's date)

Usage:
    python /path/to/squeeze_pump_v4b/tools/make_pseudo_live.py \
        --parquet-dir data \
        --out-dir data/pseudo_live \
        --now 2026-04-12T18:00:00Z \
        --symbols BTCUSDT,0GUSDT \
        --tags binance_futures,binance_spot,bybit_perps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core import _normalize_symbol


_DEFAULT_TAGS = [
    "binance_futures", "bybit_perps", "okex_swap",
    "binance_spot", "bybit_spot", "okex_spot",
    "bitget_spot", "bitget_futures",
]


def generate(parquet_dir: Path, out_dir: Path, now_ts: pd.Timestamp,
             symbols: list[str] | None, tags: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    now_date = now_ts.normalize().date()
    for pq in sorted(parquet_dir.glob("*.parquet")):
        stem = pq.stem
        matched = next((t for t in tags if stem.endswith("_" + t)), None)
        if matched is None:
            continue
        raw_sym = stem[: -(len(matched) + 1)]
        norm_sym = _normalize_symbol(raw_sym, matched)
        if symbols and norm_sym not in symbols:
            continue

        df = pd.read_parquet(pq)
        if df.empty or "ts" not in df.columns:
            continue
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df[df["ts"] <= now_ts].sort_values("ts").reset_index(drop=True)
        if df.empty:
            continue

        folder = out_dir / f"{norm_sym}_{matched}"
        folder.mkdir(parents=True, exist_ok=True)
        df["_date"] = df["ts"].dt.date
        for d, g in df.groupby("_date"):
            g = g.drop(columns=["_date"])
            target = folder / ("latest.csv" if d == now_date else f"{d.isoformat()}.csv")
            g.to_csv(target, index=False)
        print(f"  {norm_sym}_{matched}: {len(df)} rows, "
              f"{df['_date'].nunique()} day files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--now", required=True)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--tags", default=",".join(_DEFAULT_TAGS))
    args = ap.parse_args()

    now_ts = pd.Timestamp(args.now)
    now_ts = now_ts.tz_convert("UTC") if now_ts.tzinfo else now_ts.tz_localize("UTC")
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    generate(args.parquet_dir.resolve(), args.out_dir.resolve(),
             now_ts, symbols, tags)


if __name__ == "__main__":
    main()
