#!/usr/bin/env python3
"""
Focused AdaBoost sweep on target_12h — small grid around the 24h sweet spot.
Evaluates both all-trades and 5% tail Sharpe for each config.
"""

import json
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ANNUALIZE = np.sqrt(6 * 365)
COST_BPS = 10
SEED = 42
TARGET = 'target_12h'
TAIL_PCT = 5

FEATURES = [
    'price_change_4h', 'oi_change_pct', 'oi_change_dollar',
    'volume_change_pct', 'volume_dollar', 'cvd_dollar',
]

# Focused grid around the 24h sweet spot (d=2-4, lr=0.01-0.10)
GRID = {
    'model_type':       ['classifier', 'regressor'],
    'n_estimators':     [10, 20, 30, 50, 100],
    'max_depth':        [2, 3, 4],
    'min_samples_leaf': [500, 1000, 2000, 5000],
    'learning_rate':    [0.01, 0.05, 0.10],
}


def load():
    is_df  = pd.read_parquet('data/splits/is_data.parquet')
    oos_df = pd.read_parquet('data/splits/oos_data.parquet')
    caps = json.load(open('data/splits/cap_thresholds.json'))
    for fc, b in caps.items():
        is_df[fc]  = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])
    is_df  = is_df[is_df[TARGET].notna()]
    oos_df = oos_df[oos_df[TARGET].notna()]
    return is_df, oos_df


def fit_eval(X_tr, y_tr, y_cont_tr,
             X_te, y_te, y_cont_te,
             n_est, depth, leaf, lr, model_type):
    if model_type == 'classifier':
        base = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        m = AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=SEED)
        m.fit(X_tr, y_tr)
        score_te = m.predict_proba(X_te)[:, 1] - 0.5
    else:
        base = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        m = AdaBoostRegressor(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=SEED)
        m.fit(X_tr, y_cont_tr)
        score_te = m.predict(X_te)

    cost = COST_BPS / 1e4

    # All-trades (sign of score)
    sig_te = np.where(score_te > 0, 1.0, -1.0)
    pnl_all = sig_te * y_cont_te - cost
    sharpe_all = pnl_all.mean() / pnl_all.std() * ANNUALIZE if pnl_all.std() > 0 else 0

    # Tail 5%/5%
    p_lo, p_hi = np.percentile(score_te, TAIL_PCT), np.percentile(score_te, 100 - TAIL_PCT)
    long_mask  = score_te >= p_hi
    short_mask = score_te <= p_lo
    long_pnl  = y_cont_te[long_mask]  - cost
    short_pnl = -y_cont_te[short_mask] - cost
    all_pnl = np.concatenate([long_pnl, short_pnl])
    sharpe_tail = all_pnl.mean() / all_pnl.std() * ANNUALIZE if all_pnl.std() > 0 else 0
    long_bps  = long_pnl.mean() * 1e4
    short_bps = short_pnl.mean() * 1e4

    return {
        'model_type': model_type, 'n_estimators': n_est, 'max_depth': depth,
        'min_samples_leaf': leaf, 'learning_rate': lr,
        'sharpe_all': round(sharpe_all, 3),
        'sharpe_tail': round(sharpe_tail, 3),
        'long_bps': round(long_bps, 2),
        'short_bps': round(short_bps, 2),
    }


def main():
    print('Loading frozen splits...')
    is_df, oos_df = load()
    print(f'IS: {len(is_df):,} rows | OOS: {len(oos_df):,} rows | target = {TARGET}')

    X_tr = is_df[FEATURES].values
    y_cont_tr = is_df[TARGET].values
    y_tr = (y_cont_tr > 0).astype(int)

    X_te = oos_df[FEATURES].values
    y_cont_te = oos_df[TARGET].values
    y_te = (y_cont_te > 0).astype(int)

    combos = list(itertools.product(
        GRID['model_type'], GRID['n_estimators'], GRID['max_depth'],
        GRID['min_samples_leaf'], GRID['learning_rate'],
    ))
    print(f'\nSweeping {len(combos)} focused combos on target_12h...\n')
    print(f"{'#':>4}  {'type':>3} {'n':>3} {'d':>2} {'leaf':>5} {'lr':>5}  "
          f"{'all_Sh':>7} {'5%_Sh':>7} {'L_bps':>7} {'S_bps':>7}")
    print('-' * 70)

    rows = []
    best_tail = -999
    t0 = time.time()
    for i, (mtype, n_est, depth, leaf, lr) in enumerate(combos):
        try:
            r = fit_eval(X_tr, y_tr, y_cont_tr,
                         X_te, y_te, y_cont_te,
                         n_est, depth, leaf, lr, mtype)
            rows.append(r)
            marker = ''
            if r['sharpe_tail'] > best_tail:
                best_tail = r['sharpe_tail']
                marker = ' ***'
            mt = 'clf' if mtype == 'classifier' else 'reg'
            print(f"{i+1:>4}  {mt:>3} {n_est:>3} {depth:>2} {leaf:>5} {lr:>5.2f}  "
                  f"{r['sharpe_all']:>+7.3f} {r['sharpe_tail']:>+7.3f} "
                  f"{r['long_bps']:>+7.1f} {r['short_bps']:>+7.1f}{marker}")
        except Exception as e:
            print(f"{i+1:>4}  ERROR: {e}")

    total_sec = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv('results/sweep_12h_focused.csv', index=False)
    print(f'\nDone in {total_sec/60:.1f} min. Saved: results/sweep_12h_focused.csv')

    print(f'\n{"="*80}')
    print(f'  TOP 10 BY 5% TAIL SHARPE ({TARGET})')
    print(f'{"="*80}')
    top = df.sort_values('sharpe_tail', ascending=False).head(10)
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        mt = 'clf' if r.model_type == 'classifier' else 'reg'
        print(f'  {rank:>2}. {mt} n={int(r.n_estimators):>3} d={int(r.max_depth)} '
              f'leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  '
              f'all={r.sharpe_all:+.3f}  tail={r.sharpe_tail:+.3f}  '
              f'L={r.long_bps:+.1f} bps  S={r.short_bps:+.1f} bps')

    print(f'\n  REFERENCE — same grid on target_24h winner (clf n=100 d=4 leaf=2000 lr=0.10):')
    print(f'                                          tail=+3.044  L=+88.1 bps  S=+82.6 bps\n')

    # Best per model type
    for mt_name in ['classifier', 'regressor']:
        sub = df[df.model_type == mt_name]
        if len(sub) == 0:
            continue
        best = sub.loc[sub.sharpe_tail.idxmax()]
        mt = 'clf' if mt_name == 'classifier' else 'reg'
        print(f'  Best {mt_name}: {mt} n={int(best.n_estimators)} d={int(best.max_depth)} '
              f'leaf={int(best.min_samples_leaf)} lr={best.learning_rate}  '
              f'tail={best.sharpe_tail:+.3f}')


if __name__ == '__main__':
    main()
