#!/usr/bin/env python3
"""
AdaBoost sweep that loads from the frozen splits in data/splits/.
Same grid as adaboost_clean.py but uses the pre-saved IS/OOS parquet files
+ cap thresholds, so it's fully reproducible across runs.
"""

import os
import time
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import argparse


SPLITS_DIR = 'data/splits'
ANNUALIZE = np.sqrt(6 * 365)
COST_BPS = 10
SEED = 42
# TARGET is set from CLI arg (default target_24h) — see main()
TARGET = 'target_24h'

FEATURE_NAMES = [
    'price_change_4h',
    'oi_change_pct',
    'oi_change_dollar',
    'volume_change_pct',
    'volume_dollar',
    'cvd_dollar',
]


def load_splits(splits_dir):
    """Load IS and OOS, apply p1/p99 capping from cap_thresholds.json."""
    is_df  = pd.read_parquet(os.path.join(splits_dir, 'is_data.parquet'))
    oos_df = pd.read_parquet(os.path.join(splits_dir, 'oos_data.parquet'))

    with open(os.path.join(splits_dir, 'cap_thresholds.json')) as f:
        caps = json.load(f)

    for fc, bounds in caps.items():
        is_df[fc]  = is_df[fc].clip(bounds['p1'], bounds['p99'])
        oos_df[fc] = oos_df[fc].clip(bounds['p1'], bounds['p99'])

    with open(os.path.join(splits_dir, 'manifest.json')) as f:
        manifest = json.load(f)

    return is_df, oos_df, manifest


def fit_and_eval(X_tr, y_tr, y_cont_tr, sym_tr,
                 X_te, y_te, y_cont_te, sym_te,
                 n_est, depth, leaf, lr, model_type):
    cost = COST_BPS / 1e4

    if model_type == 'classifier':
        base = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        model = AdaBoostClassifier(
            estimator=base, n_estimators=n_est, learning_rate=lr,
            random_state=SEED)
        model.fit(X_tr, y_tr)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        sig_tr = np.where(pred_tr == 1, 1.0, -1.0)
        sig_te = np.where(pred_te == 1, 1.0, -1.0)
        acc_te = (pred_te == y_te).mean()
    else:
        base = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        model = AdaBoostRegressor(
            estimator=base, n_estimators=n_est, learning_rate=lr,
            random_state=SEED)
        model.fit(X_tr, y_cont_tr)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        sig_tr = np.sign(pred_tr)
        sig_te = np.sign(pred_te)
        acc_te = ((sig_te > 0) == (y_cont_te > 0)).mean()

    pnl_tr = sig_tr * y_cont_tr - cost
    pnl_te = sig_te * y_cont_te - cost

    sharpe_tr = pnl_tr.mean() / pnl_tr.std() * ANNUALIZE if pnl_tr.std() > 0 else 0
    sharpe_te = pnl_te.mean() / pnl_te.std() * ANNUALIZE if pnl_te.std() > 0 else 0

    cum = np.cumsum(pnl_te)
    total_ret = cum[-1]
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    n_prof = 0
    sym_sharpes = []
    for s in np.unique(sym_te):
        m = sym_te == s
        if m.sum() < 10:
            continue
        ps = pnl_te[m]
        sh = ps.mean() / ps.std() * ANNUALIZE if ps.std() > 0 else 0
        sym_sharpes.append(sh)
        if sh > 0:
            n_prof += 1

    return {
        'model_type': model_type,
        'n_estimators': n_est, 'max_depth': depth,
        'min_samples_leaf': leaf, 'learning_rate': lr,
        'sharpe_is': round(sharpe_tr, 3),
        'sharpe_oos': round(sharpe_te, 3),
        'acc_oos': round(acc_te, 4),
        'mean_bps_oos': round(pnl_te.mean() * 1e4, 2),
        'total_ret_oos': round(total_ret, 5),
        'max_dd_oos': round(max_dd, 5),
        'n_profitable': n_prof,
        'n_symbols': len(sym_sharpes),
        'median_sym_sharpe': round(np.median(sym_sharpes), 3) if sym_sharpes else 0,
        'mean_sym_sharpe': round(np.mean(sym_sharpes), 3) if sym_sharpes else 0,
    }


def main():
    global TARGET
    p = argparse.ArgumentParser()
    p.add_argument('--splits_dir', default=SPLITS_DIR)
    p.add_argument('--target', default='target_24h',
                   choices=['target_4h', 'target_8h', 'target_12h', 'target_24h'],
                   help='Forward-return target column to predict')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    TARGET = args.target
    if args.out is None:
        args.out = f'results/adaboost_volfilt_sweep_{TARGET}.csv'

    print(f'Loading frozen splits from {args.splits_dir}/')
    is_df, oos_df, manifest = load_splits(args.splits_dir)
    print(f'Manifest: {manifest["description"]}')
    print(f'  Filter: {manifest.get("symbol_filter", "(none)")}')
    print(f'  IS:  {manifest["n_is_symbols"]} symbols / {manifest["n_is_rows"]:,} rows')
    print(f'  OOS: {manifest["n_oos_symbols"]} symbols / {manifest["n_oos_rows"]:,} rows')

    # Prepare arrays once
    is_df  = is_df[is_df[TARGET].notna()]
    oos_df = oos_df[oos_df[TARGET].notna()]

    X_tr = is_df[FEATURE_NAMES].values
    y_cont_tr = is_df[TARGET].values
    y_tr = (y_cont_tr > 0).astype(int)
    sym_tr = is_df['symbol'].values

    X_te = oos_df[FEATURE_NAMES].values
    y_cont_te = oos_df[TARGET].values
    y_te = (y_cont_te > 0).astype(int)
    sym_te = oos_df['symbol'].values

    print(f'\nTrain: {len(y_tr):,} rows  |  Test: {len(y_te):,} rows')

    # Sweep grid (same as adaboost_clean.py)
    grid = {
        'model_type':       ['classifier', 'regressor'],
        'n_estimators':     [3, 5, 10, 15, 20, 30, 50, 100],
        'max_depth':        [1, 2, 3, 4, 5, 6],
        'min_samples_leaf': [100, 200, 500, 1000, 2000, 5000],
        'learning_rate':    [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
    }
    combos = list(itertools.product(
        grid['model_type'], grid['n_estimators'], grid['max_depth'],
        grid['min_samples_leaf'], grid['learning_rate'],
    ))
    print(f'\nSweeping {len(combos)} combinations...\n')
    print(f"{'#':>5}  {'type':>5} {'n':>3} {'d':>2} {'leaf':>5} {'lr':>5}  "
          f"{'IS':>6} {'OOS':>6} {'acc':>5} {'bps':>6} {'prof':>6}")
    print("-" * 75)

    results = []
    best_sharpe = -999
    t_start = time.time()

    for i, (mtype, n_est, depth, leaf, lr) in enumerate(combos):
        try:
            r = fit_and_eval(
                X_tr, y_tr, y_cont_tr, sym_tr,
                X_te, y_te, y_cont_te, sym_te,
                n_est, depth, leaf, lr, mtype)
            results.append(r)
            marker = ""
            if r['sharpe_oos'] > best_sharpe:
                best_sharpe = r['sharpe_oos']
                marker = " ***"
            mt = 'clf' if mtype == 'classifier' else 'reg'
            print(f"{i+1:>5}  {mt:>5} {n_est:>3} {depth:>2} {leaf:>5} {lr:>5.2f}  "
                  f"{r['sharpe_is']:>+6.2f} {r['sharpe_oos']:>+6.2f} "
                  f"{r['acc_oos']:>5.3f} {r['mean_bps_oos']:>+6.1f} "
                  f"{r['n_profitable']:>3}/{r['n_symbols']:<2}{marker}")
        except Exception as e:
            print(f"{i+1:>5}  ERROR: {e}")

        if (i + 1) % 100 == 0 and results:
            pd.DataFrame(results).to_csv(args.out, index=False)

    total = time.time() - t_start
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*75}")
    print(f"Done: {len(results)} configs in {total/60:.1f} min")
    print(f"Saved: {args.out}")

    # Top per model type
    for mtype in ['classifier', 'regressor']:
        sub = df[df.model_type == mtype]
        if sub.empty: continue
        print(f"\n{'='*75}")
        print(f"  TOP 15 {mtype.upper()} by OOS Sharpe")
        print(f"{'='*75}")
        top = sub.sort_values('sharpe_oos', ascending=False).head(15)
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(f"  {rank:>2}. n={int(r.n_estimators):>3} d={int(r.max_depth)} "
                  f"leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  "
                  f"IS={r.sharpe_is:+.2f} OOS={r.sharpe_oos:+.3f} "
                  f"acc={r.acc_oos:.3f} bps={r.mean_bps_oos:+.1f} "
                  f"prof={int(r.n_profitable)}/{int(r.n_symbols)} "
                  f"med_sh={r.median_sym_sharpe:+.2f}")

    # Best per depth
    print(f"\n{'='*75}")
    print(f"  BEST OOS SHARPE PER DEPTH")
    print(f"{'='*75}")
    for d in sorted(df.max_depth.unique()):
        sub = df[df.max_depth == d]
        best = sub.loc[sub.sharpe_oos.idxmax()]
        mt = 'clf' if best.model_type == 'classifier' else 'reg'
        print(f"  d={d}  {mt} n={int(best.n_estimators):>3} "
              f"leaf={int(best.min_samples_leaf):>5} lr={best.learning_rate:.2f}  "
              f"OOS={best.sharpe_oos:+.3f} bps={best.mean_bps_oos:+.1f} "
              f"prof={int(best.n_profitable)}/{int(best.n_symbols)}")


if __name__ == '__main__':
    main()
