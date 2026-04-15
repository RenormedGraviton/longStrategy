#!/usr/bin/env python3
"""
Tail evaluation (5% and 10%) for top CLASSIFIER configs specifically.
Top N classifiers from each sweep — one list from the unfiltered sweep,
one from the volatile-filtered sweep.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ANNUALIZE = np.sqrt(6 * 365)
COST_BPS = 10
SEED = 42
TARGET = 'target_24h'

FEATURE_NAMES = [
    'price_change_4h', 'oi_change_pct', 'oi_change_dollar',
    'volume_change_pct', 'volume_dollar', 'cvd_dollar',
]


def load_splits(splits_dir):
    is_df = pd.read_parquet(os.path.join(splits_dir, 'is_data.parquet'))
    oos_df = pd.read_parquet(os.path.join(splits_dir, 'oos_data.parquet'))
    with open(os.path.join(splits_dir, 'cap_thresholds.json')) as f:
        caps = json.load(f)
    for fc, b in caps.items():
        is_df[fc] = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])
    return is_df, oos_df


def fit_classifier(train, n_est, depth, leaf, lr):
    Xtr = train[FEATURE_NAMES].values
    ytr = (train[TARGET].values > 0).astype(int)
    base = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
    m = AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=SEED)
    m.fit(Xtr, ytr)
    return m


def eval_tail(model, test, pct):
    te = test[test[TARGET].notna()]
    X = te[FEATURE_NAMES].values
    y = te[TARGET].values
    score = model.predict_proba(X)[:, 1] - 0.5
    p_lo = np.percentile(score, pct)
    p_hi = np.percentile(score, 100 - pct)
    long_mask = score >= p_hi
    short_mask = score <= p_lo
    cost = COST_BPS / 1e4
    lp = y[long_mask] - cost
    sp = -y[short_mask] - cost
    allp = np.concatenate([lp, sp])
    sh = allp.mean() / allp.std() * ANNUALIZE if allp.std() > 0 else 0
    ls = lp.mean() / lp.std() * ANNUALIZE if len(lp) > 1 and lp.std() > 0 else 0
    ss = sp.mean() / sp.std() * ANNUALIZE if len(sp) > 1 and sp.std() > 0 else 0
    return {
        'sharpe': round(sh, 3),
        'long_sh': round(ls, 3),
        'short_sh': round(ss, 3),
        'long_bps': round(lp.mean() * 1e4, 1),
        'short_bps': round(sp.mean() * 1e4, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_csv', required=True)
    p.add_argument('--splits_dir', default='data/splits')
    p.add_argument('--top_n', type=int, default=20)
    args = p.parse_args()

    is_df, oos_df = load_splits(args.splits_dir)
    is_df = is_df[is_df[TARGET].notna()]

    sweep = pd.read_csv(args.sweep_csv)
    clf_only = sweep[sweep.model_type == 'classifier'].sort_values('sharpe_oos', ascending=False)
    top = clf_only.head(args.top_n)

    print(f'Splits: {args.splits_dir}')
    print(f'Sweep: {args.sweep_csv}')
    print(f'Top {args.top_n} CLASSIFIER configs by all-trades OOS Sharpe:\n')

    print(f"{'#':>3}  {'n':>3} {'d':>2} {'leaf':>5} {'lr':>5}  "
          f"{'orig':>6}  "
          f"{'5pct_Sh':>8} {'L5_Sh':>6} {'S5_Sh':>6} {'L5bps':>6} {'S5bps':>6}  "
          f"{'10pct_Sh':>9} {'L10_Sh':>7} {'S10_Sh':>7}")
    print('-' * 120)

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        n_est = int(r.n_estimators)
        d = int(r.max_depth)
        leaf = int(r.min_samples_leaf)
        lr = r.learning_rate
        m = fit_classifier(is_df, n_est, d, leaf, lr)
        e5 = eval_tail(m, oos_df, 5)
        e10 = eval_tail(m, oos_df, 10)
        rows.append({'rank': rank, 'n_estimators': n_est, 'max_depth': d,
                     'min_samples_leaf': leaf, 'learning_rate': lr,
                     'orig_sharpe': r.sharpe_oos,
                     'sharpe_5pct': e5['sharpe'], 'long_sh_5pct': e5['long_sh'],
                     'short_sh_5pct': e5['short_sh'], 'long_bps_5pct': e5['long_bps'],
                     'short_bps_5pct': e5['short_bps'],
                     'sharpe_10pct': e10['sharpe'], 'long_sh_10pct': e10['long_sh'],
                     'short_sh_10pct': e10['short_sh']})
        print(f'{rank:>3}  {n_est:>3} {d:>2} {leaf:>5} {lr:>5.2f}  '
              f'{r.sharpe_oos:>+6.2f}  '
              f'{e5["sharpe"]:>+8.3f} {e5["long_sh"]:>+6.2f} {e5["short_sh"]:>+6.2f} '
              f'{e5["long_bps"]:>+6.0f} {e5["short_bps"]:>+6.0f}  '
              f'{e10["sharpe"]:>+9.3f} {e10["long_sh"]:>+7.2f} {e10["short_sh"]:>+7.2f}')

    out = pd.DataFrame(rows)
    out_path = args.sweep_csv.replace('.csv', '_clf_tail.csv')
    out.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    best5 = out.sort_values('sharpe_5pct', ascending=False).head(3)
    print('\nTOP 3 CLASSIFIER BY 5% TAIL SHARPE:')
    for _, r in best5.iterrows():
        print(f'  n={int(r.n_estimators):>3} d={int(r.max_depth)} leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  '
              f'orig={r.orig_sharpe:+.2f}  5pct={r.sharpe_5pct:+.3f} '
              f'(L={r.long_sh_5pct:+.2f}/{r.long_bps_5pct:+.0f}, S={r.short_sh_5pct:+.2f}/{r.short_bps_5pct:+.0f})')


if __name__ == '__main__':
    main()
