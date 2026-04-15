#!/usr/bin/env python3
"""
For the top configs from a sweep CSV, re-fit and evaluate at top/bottom
5% AND 10% quantiles of OOS predictions. Reads frozen splits from
data/splits/.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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


def fit_model(train, n_est, depth, leaf, lr, model_type):
    Xtr = train[FEATURE_NAMES].values
    ytr_cont = train[TARGET].values
    if model_type == 'regressor':
        base = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        m = AdaBoostRegressor(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=SEED)
        m.fit(Xtr, ytr_cont)
    else:
        ytr = (ytr_cont > 0).astype(int)
        base = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        m = AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=SEED)
        m.fit(Xtr, ytr)
    return m


def get_score(model, X, model_type):
    if model_type == 'regressor':
        return model.predict(X)
    return model.predict_proba(X)[:, 1] - 0.5


def evaluate_tail(model, test, model_type, pct):
    te = test[test[TARGET].notna()]
    Xte = te[FEATURE_NAMES].values
    y_cont = te[TARGET].values

    score = get_score(model, Xte, model_type)
    p_lo = np.percentile(score, pct)
    p_hi = np.percentile(score, 100 - pct)

    long_mask = score >= p_hi
    short_mask = score <= p_lo

    cost = COST_BPS / 1e4
    long_pnl = y_cont[long_mask] - cost
    short_pnl = -y_cont[short_mask] - cost

    all_pnl = np.concatenate([long_pnl, short_pnl])
    sharpe = all_pnl.mean() / all_pnl.std() * ANNUALIZE if all_pnl.std() > 0 else 0
    long_sh = long_pnl.mean() / long_pnl.std() * ANNUALIZE if len(long_pnl) > 1 and long_pnl.std() > 0 else 0
    short_sh = short_pnl.mean() / short_pnl.std() * ANNUALIZE if len(short_pnl) > 1 and short_pnl.std() > 0 else 0

    return {
        f'sharpe_{pct}pct':   round(sharpe, 3),
        f'long_sh_{pct}pct':  round(long_sh, 3),
        f'short_sh_{pct}pct': round(short_sh, 3),
        f'long_bps_{pct}pct': round(long_pnl.mean() * 1e4, 1),
        f'short_bps_{pct}pct': round(short_pnl.mean() * 1e4, 1),
        f'long_win_{pct}pct': round((long_pnl > 0).mean(), 4),
        f'short_win_{pct}pct': round((short_pnl > 0).mean(), 4),
        f'n_long_{pct}pct':  int(long_mask.sum()),
        f'n_short_{pct}pct': int(short_mask.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_csv', required=True,
                   help='Sweep results CSV (e.g., results/adaboost_volfilt_sweep.csv)')
    p.add_argument('--splits_dir', default='data/splits')
    p.add_argument('--top_n', type=int, default=20)
    p.add_argument('--out', default='results/tail_eval.csv')
    args = p.parse_args()

    print(f'Loading splits from {args.splits_dir}/')
    is_df, oos_df = load_splits(args.splits_dir)
    is_df = is_df[is_df[TARGET].notna()]
    print(f'IS: {len(is_df):,} rows | OOS: {len(oos_df[oos_df[TARGET].notna()]):,} rows')

    sweep = pd.read_csv(args.sweep_csv)
    top = sweep.sort_values('sharpe_oos', ascending=False).head(args.top_n)
    print(f'\nTop {args.top_n} configs from {args.sweep_csv}:\n')

    print(f"{'#':>3}  {'type':>3} {'n':>3} {'d':>2} {'leaf':>5} {'lr':>5}  "
          f"{'orig':>6}  "
          f"{'5pct_Sh':>8} {'L5_Sh':>6} {'S5_Sh':>6}  "
          f"{'10pct_Sh':>9} {'L10_Sh':>7} {'S10_Sh':>7}")
    print('-' * 115)

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        n_est = int(r.n_estimators)
        d = int(r.max_depth)
        leaf = int(r.min_samples_leaf)
        lr = r.learning_rate
        mtype = r.model_type

        model = fit_model(is_df, n_est, d, leaf, lr, mtype)
        ev5 = evaluate_tail(model, oos_df, mtype, 5)
        ev10 = evaluate_tail(model, oos_df, mtype, 10)

        row = {
            'rank': rank, 'model_type': mtype, 'n_estimators': n_est,
            'max_depth': d, 'min_samples_leaf': leaf, 'learning_rate': lr,
            'orig_sharpe': r.sharpe_oos, **ev5, **ev10,
        }
        rows.append(row)

        mt = 'reg' if mtype == 'regressor' else 'clf'
        print(f'{rank:>3}  {mt:>3} {n_est:>3} {d:>2} {leaf:>5} {lr:>5.2f}  '
              f'{r.sharpe_oos:>+6.2f}  '
              f'{ev5["sharpe_5pct"]:>+8.3f} {ev5["long_sh_5pct"]:>+6.2f} {ev5["short_sh_5pct"]:>+6.2f}  '
              f'{ev10["sharpe_10pct"]:>+9.3f} {ev10["long_sh_10pct"]:>+7.2f} {ev10["short_sh_10pct"]:>+7.2f}')

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out, index=False)
    print(f'\nSaved: {args.out}')

    # Best by 5pct and 10pct
    print()
    print('='*100)
    print('  TOP 5 BY 5% TAIL SHARPE')
    print('='*100)
    for _, r in df_out.sort_values('sharpe_5pct', ascending=False).head(5).iterrows():
        mt = 'reg' if r.model_type == 'regressor' else 'clf'
        print(f"  {mt} n={int(r.n_estimators):>3} d={int(r.max_depth)} leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  "
              f"orig={r.orig_sharpe:+.2f}  "
              f"5pct={r.sharpe_5pct:+.3f} (L={r.long_sh_5pct:+.2f}/{r.long_bps_5pct:+.0f}bps, S={r.short_sh_5pct:+.2f}/{r.short_bps_5pct:+.0f}bps)")

    print()
    print('='*100)
    print('  TOP 5 BY 10% TAIL SHARPE')
    print('='*100)
    for _, r in df_out.sort_values('sharpe_10pct', ascending=False).head(5).iterrows():
        mt = 'reg' if r.model_type == 'regressor' else 'clf'
        print(f"  {mt} n={int(r.n_estimators):>3} d={int(r.max_depth)} leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  "
              f"orig={r.orig_sharpe:+.2f}  "
              f"10pct={r.sharpe_10pct:+.3f} (L={r.long_sh_10pct:+.2f}/{r.long_bps_10pct:+.0f}bps, S={r.short_sh_10pct:+.2f}/{r.short_bps_10pct:+.0f}bps)")


if __name__ == '__main__':
    main()
