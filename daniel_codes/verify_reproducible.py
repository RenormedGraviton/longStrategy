#!/usr/bin/env python3
"""
Verification script: loads everything from disk and reproduces the winning
AdaBoost classifier's OOS metrics. Exits 0 if matches, exits 1 if not.

Run this after restarting Claude to confirm the model and data are intact.

Usage:
    cd /home/dbai/perp-rf-strategy
    python3 verify_reproducible.py
"""

import json
import sys
import os
import joblib
import numpy as np
import pandas as pd

# --- Paths (absolute from /home/dbai/perp-rf-strategy/) ---
MODEL_DIR  = 'models/winner_clf_20260411'
MODEL_PATH = f'{MODEL_DIR}/model.joblib'
META_PATH  = f'{MODEL_DIR}/metadata.json'
IS_PATH    = 'data/splits/is_data.parquet'
OOS_PATH   = 'data/splits/oos_data.parquet'
CAPS_PATH  = 'data/splits/cap_thresholds.json'

# --- Expected numbers (from metadata at save time) ---
EXPECTED = {
    'sharpe':    3.044,
    'long_bps':  88.07,
    'short_bps': 82.59,
    'n_long':    2749,
    'n_short':   2752,
}
TOLERANCE = 0.02   # 2% relative


def check_file(path, label):
    if not os.path.exists(path):
        print(f'  [FAIL] {label} NOT FOUND at {path}')
        return False
    size = os.path.getsize(path)
    print(f'  [ OK ] {label}  {path}  ({size:,} bytes)')
    return True


def main():
    print('='*70)
    print('  REPRODUCIBILITY VERIFICATION')
    print('='*70)
    print(f'Working dir: {os.getcwd()}')
    print()

    # Step 1: check all required files exist
    print('Step 1: checking files exist...')
    ok = True
    ok &= check_file(MODEL_PATH, 'Model file')
    ok &= check_file(META_PATH, 'Metadata')
    ok &= check_file(IS_PATH, 'IS parquet')
    ok &= check_file(OOS_PATH, 'OOS parquet')
    ok &= check_file(CAPS_PATH, 'Cap thresholds')
    if not ok:
        print('\n[FAIL] Missing files — cannot verify.')
        sys.exit(1)

    # Step 2: load
    print('\nStep 2: loading model + data...')
    model = joblib.load(MODEL_PATH)
    meta  = json.load(open(META_PATH))
    oos   = pd.read_parquet(OOS_PATH)
    caps  = json.load(open(CAPS_PATH))
    print(f'  Model type:     {type(model).__name__}')
    print(f'  Trees fitted:   {len(model.estimators_)}')
    print(f'  Target:         {meta["target"]}')
    print(f'  Features:       {meta["features"]}')
    print(f'  Hyperparameters: {meta["hyperparameters"]}')
    print(f'  OOS rows raw:   {len(oos):,}')

    # Step 3: apply capping from frozen thresholds
    print('\nStep 3: applying p1/p99 capping from cap_thresholds.json...')
    for fc, b in caps.items():
        oos[fc] = oos[fc].clip(b['p1'], b['p99'])
    oos = oos[oos[meta['target']].notna()]
    print(f'  OOS rows after dropna: {len(oos):,}')

    # Step 4: predict + tail select
    print('\nStep 4: predicting + selecting top/bottom 5%...')
    features = meta['features']
    X = oos[features].values
    y = oos[meta['target']].values
    score = model.predict_proba(X)[:, 1] - 0.5
    p_lo = np.percentile(score, 5)
    p_hi = np.percentile(score, 95)
    long_mask  = score >= p_hi
    short_mask = score <= p_lo
    print(f'  Long trades:  {int(long_mask.sum()):,}')
    print(f'  Short trades: {int(short_mask.sum()):,}')

    # Step 5: compute metrics
    cost = 10 / 1e4
    long_pnl  = y[long_mask]  - cost
    short_pnl = -y[short_mask] - cost
    allp = np.concatenate([long_pnl, short_pnl])
    sharpe = allp.mean() / allp.std() * np.sqrt(6 * 365)
    long_bps = long_pnl.mean() * 1e4
    short_bps = short_pnl.mean() * 1e4

    print(f'\nStep 5: metrics\n')
    print(f'  {"metric":<25} {"expected":>12} {"actual":>12} {"match":>8}')
    print(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*8}')
    results = [
        ('per-trade Sharpe (ann)', EXPECTED['sharpe'], sharpe),
        ('long bps/trade',         EXPECTED['long_bps'], long_bps),
        ('short bps/trade',        EXPECTED['short_bps'], short_bps),
        ('n long trades',          EXPECTED['n_long'], int(long_mask.sum())),
        ('n short trades',         EXPECTED['n_short'], int(short_mask.sum())),
    ]
    all_match = True
    for name, exp, act in results:
        rel_err = abs(act - exp) / (abs(exp) + 1e-9)
        match = rel_err < TOLERANCE
        marker = 'PASS' if match else 'FAIL'
        print(f'  {name:<25} {exp:>12.3f} {act:>12.3f} {marker:>8}')
        if not match:
            all_match = False

    print()
    print('='*70)
    if all_match:
        print('  [SUCCESS] All metrics match within 2% — model reproduces exactly.')
        print('='*70)
        sys.exit(0)
    else:
        print('  [FAIL] Some metrics do not match — investigate before trusting model.')
        print('='*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
