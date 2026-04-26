"""Convert the parquet archive in data/ into the date-partitioned CSV
layout the production loader expects.

Input  : data/{RAW_SYMBOL}_{TAG}.parquet
         RAW_SYMBOL may be hyphenated (OKEx: '0G-USDT-SWAP', 'A-USDT')
         or already normalized (Binance/Bybit/Bitget: '0GUSDT').
Output : data_standard/{NORM_SYMBOL}_{TAG}/YYYY-MM-DD.csv
         Symbols are aligned via the notebook's `normalize_symbol`:
            OKEx '0G-USDT-SWAP'  → '0GUSDT'
            OKEx 'A-USDT'        → 'AUSDT'
            others               → unchanged
         If two parquets map to the same (NORM_SYMBOL, TAG) — e.g. an
         already-normalized file plus a hyphenated OKEx file for the
         same exchange — they are merged on ts (last-wins).

Usage (from repo root):
    python codes/convert_data.py
    python codes/convert_data.py --data-dir data --out-dir data_standard
    python codes/convert_data.py --overwrite           # replace existing day files
    python codes/convert_data.py --symbols BTCUSDT,ETHUSDT
    python codes/convert_data.py --tags binance_futures,binance_spot
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def normalize_symbol(raw: str, tag: str) -> str:
    """Same rule as notebook & codes/toolbox.py — only OKEx symbols change."""
    s = raw.upper()
    if "okex" in tag.lower():
        s = s.replace("-USDT-SWAP", "USDT").replace("-USDT", "USDT")
    return s


def parse_filename(stem: str, known_tags: list[str]) -> tuple[str, str] | None:
    """Split '{RAW_SYMBOL}_{TAG}' by trying every known tag as suffix."""
    for tag in known_tags:
        suffix = "_" + tag
        if stem.endswith(suffix):
            return stem[: -len(suffix)], tag
    return None


def group_files(data_dir: Path, known_tags: list[str],
                symbols_filter: set[str] | None,
                tags_filter: set[str] | None
                ) -> dict[tuple[str, str], list[Path]]:
    """Bucket parquets by (NORM_SYMBOL, TAG). Multi-bucket files = merge later."""
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    skipped = 0
    for pq in sorted(data_dir.glob("*.parquet")):
        parsed = parse_filename(pq.stem, known_tags)
        if parsed is None:
            skipped += 1
            continue
        raw_sym, tag = parsed
        if tags_filter and tag not in tags_filter:
            continue
        norm = normalize_symbol(raw_sym, tag)
        if symbols_filter and norm not in symbols_filter:
            continue
        groups[(norm, tag)].append(pq)
    if skipped:
        print(f"  skipped {skipped} parquets with unrecognized tag suffix")
    return groups


def merge_parquets(paths: list[Path]) -> pd.DataFrame:
    """Read + concat. For multi-file groups (rare — only when both a
    hyphenated and pre-normalized OKEx file exist for the same exchange)
    we drop ts duplicates keeping last. Single-file groups are NEVER
    deduped — every row from the source parquet is preserved."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        if "ts" not in df.columns:
            print(f"    ! {p.name}: no 'ts' column — skipping")
            continue
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if len(paths) > 1:
        before = len(df)
        df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        if before != len(df):
            print(f"    [merge] dedup dropped {before - len(df)} duplicate-ts rows")
    return df


# %.17g preserves full float64 round-trip precision on the WRITE side
# (IEEE 754 needs 17 significant decimal digits to losslessly recover
# any double). On the READ side, callers MUST pass
# `float_precision='round_trip'` to pd.read_csv — pandas' default C
# parser truncates ~1 ULP and the regen→read round-trip is otherwise
# not bit-exact. The production loader (squeeze_pump_v4b/core.py)
# already does this; if you write a new reader, pass it explicitly.
_FLOAT_FMT = "%.17g"


def write_day_csvs(df: pd.DataFrame, out_dir: Path, overwrite: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["_date"] = df["ts"].dt.date
    written = 0
    for d, g in df.groupby("_date"):
        g = g.drop(columns=["_date"])
        path = out_dir / f"{d.isoformat()}.csv"
        if path.exists() and not overwrite:
            continue
        g.to_csv(path, index=False, float_format=_FLOAT_FMT)
        written += 1
    return written


# ── CLI ──────────────────────────────────────────────────────────────
DEFAULT_TAGS = [
    "binance_futures", "bybit_perps", "okex_swap",
    "binance_spot", "bybit_spot", "okex_spot",
    "bitget_spot", "bitget_futures",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data_standard")
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS),
                    help="Comma-separated exchange tags to convert")
    ap.add_argument("--symbols", default="",
                    help="Comma-separated NORMALIZED symbols (e.g. 0GUSDT,BTCUSDT)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing YYYY-MM-DD.csv files")
    ap.add_argument("--verify", action="store_true",
                    help="After writing, re-read each (symbol, tag) folder "
                         "and assert bit-exact equality of every numeric "
                         "column against its source parquet")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not data_dir.is_dir():
        sys.exit(f"data dir not found: {data_dir}")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    syms = {s.strip() for s in args.symbols.split(",") if s.strip()} or None

    print(f"convert_data: {data_dir}  →  {out_dir}")
    print(f"  tags: {tags}")
    if syms:
        print(f"  symbols filter: {sorted(syms)}")

    groups = group_files(data_dir, tags, syms, set(tags))
    if not groups:
        sys.exit("no parquets matched")
    print(f"  {len(groups)} (symbol, tag) groups; "
          f"{sum(len(v) for v in groups.values())} parquets total")

    n_groups = total_days = total_rows = 0
    merge_count = 0
    for (norm_sym, tag), paths in sorted(groups.items()):
        n_groups += 1
        if len(paths) > 1:
            merge_count += 1
            print(f"  [merge] {norm_sym}_{tag} ← {[p.name for p in paths]}")
        df = merge_parquets(paths)
        if df.empty:
            continue
        total_rows += len(df)
        wrote = write_day_csvs(df, out_dir / f"{norm_sym}_{tag}", args.overwrite)
        total_days += wrote
        if n_groups % 200 == 0:
            print(f"  … {n_groups}/{len(groups)} groups, {total_days} day files written")

    print(f"\nDONE: {n_groups} groups → {total_days} day CSVs "
          f"({total_rows:,} rows total). Merged {merge_count} groups across "
          f"normalized variants.")
    print(f"  output: {out_dir}")

    if args.verify:
        print("\nverifying bit-exact round-trip ...")
        verify(groups, out_dir)


def verify(groups: dict, out_dir: Path) -> None:
    """Re-read each output folder and bit-compare with the source parquet.
    Single-file groups must be byte-equal across all numeric columns."""
    import re
    DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
    NUM_COLS = ["close", "volume_usdt", "cvd_usdt", "open_interest",
                "funding_rate", "funding_interval_hours", "high", "low"]

    n_ok = n_bad = 0
    for (norm_sym, tag), paths in sorted(groups.items()):
        if len(paths) != 1:
            continue  # skip merged groups (true source is union, harder to compare)
        orig = pd.read_parquet(paths[0])
        orig["ts"] = pd.to_datetime(orig["ts"], utc=True)
        orig = orig.sort_values("ts").reset_index(drop=True)

        folder = out_dir / f"{norm_sym}_{tag}"
        frames = []
        for f in sorted(folder.iterdir()):
            if DATE_RE.match(f.name):
                df = pd.read_csv(f, float_precision="round_trip")
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                frames.append(df)
        back = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)

        bad: list[str] = []
        if len(orig) != len(back):
            bad.append(f"row {len(orig)} != {len(back)}")
        else:
            for c in NUM_COLS:
                if c not in orig.columns:
                    continue
                a = orig[c].to_numpy()
                b = back[c].to_numpy()
                if a.tobytes() != b.tobytes():
                    bad.append(c)
        if bad:
            n_bad += 1
            print(f"  BAD {norm_sym}_{tag}: {bad}")
        else:
            n_ok += 1
    print(f"verify: {n_ok} OK, {n_bad} BAD out of {n_ok + n_bad} single-file groups")


if __name__ == "__main__":
    main()
