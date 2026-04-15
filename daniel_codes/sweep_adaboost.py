#!/usr/bin/env python3
"""
Sweep AdaBoost hyperparameters for target_24h.
Loads data ONCE via fit_simple.stage3_universe_split, then iterates.
"""

import sys
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Reuse data loading from fit_simple
from fit_simple import stage3_universe_split

# ── Sweep grid ──────────────────────────────────────────────────────
GRID = {
    'n_estimators':      [3, 5, 10, 15, 20, 30, 50, 100],
    'max_depth':         [1, 2, 3, 4, 5, 6],
    'min_samples_leaf':  [100, 200, 500, 1000, 2000, 5000],
    'learning_rate':     [0.05, 0.1, 0.3, 0.5, 1.0],
}

TARGET_COL = 'target_24h'
COST_BPS = 10
SEED = 42
ANNUALIZE = np.sqrt(6 * 365)   # 6 bars/day × 365 days

FEATURE_COLS = ['price_change_4h', 'oi_change_pct', 'oi_change_dollar',
                'volume_change_pct', 'volume_dollar', 'cvd_dollar']


def evaluate_config(X_train, y_train, y_cont_train, sym_train,
                    X_test, y_test, y_cont_test, sym_test,
                    n_est, depth, leaf, lr):
    """Fit one AdaBoost config, return metrics dict."""
    base = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=SEED,
    )
    model = AdaBoostClassifier(
        estimator=base,
        n_estimators=n_est,
        learning_rate=lr,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    cost = COST_BPS / 1e4

    # ── In-sample ──
    pred_tr = model.predict(X_train)
    sig_tr = np.where(pred_tr == 1, 1.0, -1.0)
    pnl_tr = sig_tr * y_cont_train
    pnl_tr_net = pnl_tr - cost
    sharpe_tr = pnl_tr_net.mean() / pnl_tr_net.std() * ANNUALIZE if pnl_tr_net.std() > 0 else 0

    # ── OOS ──
    pred_te = model.predict(X_test)
    sig_te = np.where(pred_te == 1, 1.0, -1.0)
    pnl_te = sig_te * y_cont_test
    pnl_te_net = pnl_te - cost
    sharpe_te = pnl_te_net.mean() / pnl_te_net.std() * ANNUALIZE if pnl_te_net.std() > 0 else 0

    acc_te = (pred_te == y_test).mean()
    win_te = (pnl_te_net > 0).mean()

    cum = np.cumsum(pnl_te_net)
    total_ret = cum[-1]
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    # Per-symbol stats
    unique_syms = np.unique(sym_test)
    n_profitable = 0
    sym_sharpes = []
    for s in unique_syms:
        mask = sym_test == s
        if mask.sum() < 10:
            continue
        pnl_s = pnl_te_net[mask]
        sh_s = pnl_s.mean() / pnl_s.std() * ANNUALIZE if pnl_s.std() > 0 else 0
        sym_sharpes.append(sh_s)
        if sh_s > 0:
            n_profitable += 1

    mean_sym_sharpe = np.mean(sym_sharpes) if sym_sharpes else 0
    median_sym_sharpe = np.median(sym_sharpes) if sym_sharpes else 0

    up_rate_tr = pred_tr.mean()
    up_rate_te = pred_te.mean()

    return {
        'n_estimators': n_est,
        'max_depth': depth,
        'min_samples_leaf': leaf,
        'learning_rate': lr,
        'sharpe_is': round(sharpe_tr, 3),
        'sharpe_oos': round(sharpe_te, 3),
        'acc_oos': round(acc_te, 4),
        'win_oos': round(win_te, 4),
        'total_ret_oos': round(total_ret, 5),
        'max_dd_oos': round(max_dd, 5),
        'mean_bps_oos': round(pnl_te_net.mean() * 1e4, 2),
        'up_rate_is': round(up_rate_tr, 3),
        'up_rate_oos': round(up_rate_te, 3),
        'n_profitable': n_profitable,
        'n_symbols': len(sym_sharpes),
        'mean_sym_sharpe': round(mean_sym_sharpe, 3),
        'median_sym_sharpe': round(median_sym_sharpe, 3),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bars_dir', default='data/bars')
    p.add_argument('--cap_pct', type=float, default=1,
                   help='Feature capping percentile (1 = p1/p99)')
    p.add_argument('--filter_through_date', default='2026-02-07')
    p.add_argument('--max_abs_ret', type=float, default=0.010)
    p.add_argument('--max_skew', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--demean', action='store_true', default=False)
    p.add_argument('--out', default='results/adaboost_sweep.csv')
    args = p.parse_args()

    # ── Load data once ──
    print("Loading universe and splitting...")
    t0 = time.time()
    _, train, test, _, _, _ = stage3_universe_split(args)
    print(f"Data loaded in {time.time()-t0:.1f}s\n")

    # Prepare arrays once
    tr_mask = train[TARGET_COL].notna()
    X_train = train.loc[tr_mask, FEATURE_COLS].values
    y_cont_train = train.loc[tr_mask, TARGET_COL].values
    y_train = (y_cont_train > 0).astype(int)
    sym_train = train.loc[tr_mask, 'symbol'].values

    te_mask = test[TARGET_COL].notna()
    X_test = test.loc[te_mask, FEATURE_COLS].values
    y_cont_test = test.loc[te_mask, TARGET_COL].values
    y_test = (y_cont_test > 0).astype(int)
    sym_test = test.loc[te_mask, 'symbol'].values

    print(f"Train: {len(y_train):,} rows | Test: {len(y_test):,} rows")
    print(f"Features: {FEATURE_COLS}\n")

    # ── Sweep ──
    combos = list(itertools.product(
        GRID['n_estimators'], GRID['max_depth'],
        GRID['min_samples_leaf'], GRID['learning_rate'],
    ))
    print(f"Sweeping {len(combos)} combinations...\n")
    print(f"{'#':>4}  {'n_est':>5} {'depth':>5} {'leaf':>6} {'lr':>5}  "
          f"{'IS_Sh':>6} {'OOS_Sh':>6} {'acc':>5} {'ret%':>7} {'DD%':>6} "
          f"{'bps':>6} {'prof':>5}  {'elapsed':>7}")
    print("-" * 95)

    results = []
    t_start = time.time()
    best_sharpe = -999
    best_idx = -1

    for i, (n_est, depth, leaf, lr) in enumerate(combos):
        t1 = time.time()
        try:
            r = evaluate_config(
                X_train, y_train, y_cont_train, sym_train,
                X_test, y_test, y_cont_test, sym_test,
                n_est, depth, leaf, lr,
            )
            results.append(r)

            marker = ""
            if r['sharpe_oos'] > best_sharpe:
                best_sharpe = r['sharpe_oos']
                best_idx = len(results) - 1
                marker = " ★"

            elapsed = time.time() - t1
            print(f"{i+1:>4}  {n_est:>5} {depth:>5} {leaf:>6} {lr:>5.2f}  "
                  f"{r['sharpe_is']:>+6.2f} {r['sharpe_oos']:>+6.2f} "
                  f"{r['acc_oos']:>5.3f} {r['total_ret_oos']*100:>+7.2f} "
                  f"{r['max_dd_oos']*100:>6.2f} {r['mean_bps_oos']:>+6.2f} "
                  f"{r['n_profitable']:>3}/{r['n_symbols']:<2} {elapsed:>6.1f}s{marker}")
        except Exception as e:
            print(f"{i+1:>4}  {n_est:>5} {depth:>5} {leaf:>6} {lr:>5.2f}  ERROR: {e}")

        # Save checkpoint every 50 combos
        if (i + 1) % 50 == 0 and results:
            df_tmp = pd.DataFrame(results)
            df_tmp.to_csv(args.out, index=False)

    total_time = time.time() - t_start

    # ── Save final results ──
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"\n{'='*95}")
    print(f"Sweep complete: {len(results)} configs in {total_time:.0f}s "
          f"({total_time/len(results):.1f}s avg)")
    print(f"Saved to {args.out}")

    # ── Top 20 by OOS Sharpe ──
    df_sorted = df.sort_values('sharpe_oos', ascending=False)
    print(f"\n{'='*95}")
    print(f"  TOP 20 by OOS Sharpe (net of {COST_BPS}bp cost)")
    print(f"{'='*95}")
    print(f"{'rank':>4}  {'n_est':>5} {'depth':>5} {'leaf':>6} {'lr':>5}  "
          f"{'IS_Sh':>6} {'OOS_Sh':>7} {'acc':>5} {'ret%':>7} {'DD%':>6} "
          f"{'bps':>6} {'prof':>5} {'med_sh':>6}")
    print("-" * 95)
    for rank, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
        print(f"{rank:>4}  {int(row.n_estimators):>5} {int(row.max_depth):>5} "
              f"{int(row.min_samples_leaf):>6} {row.learning_rate:>5.2f}  "
              f"{row.sharpe_is:>+6.2f} {row.sharpe_oos:>+7.3f} "
              f"{row.acc_oos:>5.3f} {row.total_ret_oos*100:>+7.2f} "
              f"{row.max_dd_oos*100:>6.2f} {row.mean_bps_oos:>+6.2f} "
              f"{int(row.n_profitable):>3}/{int(row.n_symbols):<2} "
              f"{row.median_sym_sharpe:>+6.2f}")

    # ── Worst 5 (sanity check) ──
    print(f"\n  WORST 5:")
    for rank, (_, row) in enumerate(df_sorted.tail(5).iterrows(), 1):
        print(f"  {int(row.n_estimators):>5} {int(row.max_depth):>5} "
              f"{int(row.min_samples_leaf):>6} {row.learning_rate:>5.2f}  "
              f"OOS_Sh={row.sharpe_oos:>+.3f}  ret={row.total_ret_oos*100:>+.2f}%")


if __name__ == '__main__':
    main()
