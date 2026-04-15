#!/usr/bin/env python3
"""
Build features for all 195 symbols and save the IS/OOS splits to disk.
This makes future runs reproducible without needing to re-load + re-split.

Outputs (in data/splits/):
  is_data.parquet     — in-sample DataFrame (raw features, no capping)
  oos_data.parquet    — out-of-sample DataFrame
  is_symbols.txt      — IS symbol list
  oos_symbols.txt     — OOS symbol list
  cap_thresholds.json — p1/p99 thresholds computed on IS only (apply later)
  manifest.json       — metadata: seed, date, row counts, etc.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from adaboost_clean import load_all, resample_4h, compute_features, FEATURE_NAMES

SEED = 42
OUT_DIR = 'data/splits'
MIN_ABS_RET = 0.008  # filter: KEEP symbols whose mean |1h return| > 0.8%


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading 1h bars...')
    symbols = load_all('data/bars')
    print(f'Loaded {len(symbols)} symbols')

    # Per-symbol volatility filter: KEEP if mean |1h return| > MIN_ABS_RET
    # Computed from each symbol's own 1h close prices, no cross-symbol info.
    print(f'\nApplying filter: mean |1h return| > {MIN_ABS_RET} (keep volatile symbols)')
    kept_symbols = {}
    dropped = []
    for sym, df_1h in symbols.items():
        ret = df_1h['close'].pct_change().dropna()
        if len(ret) < 100:
            dropped.append((sym, f'too few bars ({len(ret)})'))
            continue
        ar = ret.abs().mean()
        if ar <= MIN_ABS_RET:
            dropped.append((sym, f'abs_ret={ar:.4f}'))
            continue
        kept_symbols[sym] = df_1h
    print(f'  Kept:    {len(kept_symbols)} symbols')
    print(f'  Dropped: {len(dropped)} symbols (sample: {dropped[:5]})')

    # Build features for kept symbols
    print('\nComputing features...')
    all_features = []
    for sym, df_1h in kept_symbols.items():
        bars4h = resample_4h(df_1h)
        if len(bars4h) < 10:
            continue
        f = compute_features(bars4h)
        f['symbol'] = sym
        all_features.append(f)

    combined = pd.concat(all_features)
    combined = combined.dropna(subset=FEATURE_NAMES)
    print(f'Total: {len(combined):,} rows, {combined.symbol.nunique()} symbols')

    # Random 50/50 symbol split, seed=42
    all_syms = sorted(combined['symbol'].unique())
    rng = np.random.default_rng(SEED)
    shuffled = list(rng.permutation(all_syms))
    half = len(shuffled) // 2
    is_syms = sorted(shuffled[:half])
    oos_syms = sorted(shuffled[half:])

    is_df = combined[combined['symbol'].isin(is_syms)].copy()
    oos_df = combined[combined['symbol'].isin(oos_syms)].copy()

    print(f'\nSplit (seed={SEED}):')
    print(f'  IS:  {len(is_syms):>3} symbols / {len(is_df):>7,} rows')
    print(f'  OOS: {len(oos_syms):>3} symbols / {len(oos_df):>7,} rows')

    # Compute p1/p99 cap thresholds from IS only
    cap_thresholds = {}
    is_with_target = is_df[is_df['target_24h'].notna()]
    for fc in FEATURE_NAMES:
        cap_thresholds[fc] = {
            'p1':  float(is_with_target[fc].quantile(0.01)),
            'p99': float(is_with_target[fc].quantile(0.99)),
        }

    # Save data
    is_path = os.path.join(OUT_DIR, 'is_data.parquet')
    oos_path = os.path.join(OUT_DIR, 'oos_data.parquet')
    is_df.to_parquet(is_path, index=True)
    oos_df.to_parquet(oos_path, index=True)
    print(f'\nSaved:')
    print(f'  {is_path:<35} ({os.path.getsize(is_path)/1024:.0f} KB)')
    print(f'  {oos_path:<35} ({os.path.getsize(oos_path)/1024:.0f} KB)')

    # Save symbol lists
    is_syms_path = os.path.join(OUT_DIR, 'is_symbols.txt')
    oos_syms_path = os.path.join(OUT_DIR, 'oos_symbols.txt')
    with open(is_syms_path, 'w') as f:
        f.write('\n'.join(is_syms) + '\n')
    with open(oos_syms_path, 'w') as f:
        f.write('\n'.join(oos_syms) + '\n')
    print(f'  {is_syms_path}')
    print(f'  {oos_syms_path}')

    # Save cap thresholds
    cap_path = os.path.join(OUT_DIR, 'cap_thresholds.json')
    with open(cap_path, 'w') as f:
        json.dump(cap_thresholds, f, indent=2)
    print(f'  {cap_path}')

    # Save manifest
    manifest = {
        'created':              datetime.now().isoformat(timespec='seconds'),
        'seed':                 SEED,
        'split_method':         'random_50_50_by_symbol',
        'features':             FEATURE_NAMES,
        'target':               'target_24h',
        'bar_size':             '4h',
        'forward_horizon_bars': 6,
        'forward_horizon_hours': 24,
        'symbol_filter':        f'mean |1h return| > {MIN_ABS_RET} (keep volatile)',
        'n_symbols_loaded':     len(symbols),
        'n_symbols_dropped':    len(dropped),
        'n_symbols_kept':       len(all_syms),
        'n_is_symbols':         len(is_syms),
        'n_oos_symbols':        len(oos_syms),
        'n_is_rows':            len(is_df),
        'n_oos_rows':           len(oos_df),
        'capping_note':         'Features in parquet files are RAW (uncapped). Apply cap_thresholds.json before fitting.',
        'description':          f'Phase 3 AdaBoost data splits — frozen 2026-04-11 for reproducibility. Filter applied: mean |1h return| > {MIN_ABS_RET} (keep volatile symbols).',
    }
    manifest_path = os.path.join(OUT_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'  {manifest_path}')

    # Print loader snippet
    print(f'\n{"="*70}')
    print('  HOW TO LOAD NEXT TIME')
    print(f'{"="*70}')
    print('''
import pandas as pd
import json

is_df  = pd.read_parquet("data/splits/is_data.parquet")
oos_df = pd.read_parquet("data/splits/oos_data.parquet")

with open("data/splits/cap_thresholds.json") as f:
    caps = json.load(f)

# Apply p1/p99 capping (computed on IS only)
for fc, bounds in caps.items():
    is_df[fc]  = is_df[fc].clip(bounds["p1"], bounds["p99"])
    oos_df[fc] = oos_df[fc].clip(bounds["p1"], bounds["p99"])
''')


if __name__ == '__main__':
    main()
