#!/usr/bin/env python3
"""
For the top configs from the AdaBoost sweep, re-fit and evaluate
using ONLY the top 5% (longs) and bottom 5% (shorts) of OOS predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from adaboost_clean import load_all, build_universe, FEATURE_NAMES

ANNUALIZE = np.sqrt(6 * 365)
COST_BPS = 10
SEED = 42
TARGET = 'target_24h'


def fit_model(train, n_est, depth, leaf, lr, model_type):
    Xtr = train[FEATURE_NAMES].values
    ytr_cont = train[TARGET].values

    if model_type == 'regressor':
        base = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                     random_state=SEED)
        m = AdaBoostRegressor(estimator=base, n_estimators=n_est,
                              learning_rate=lr, random_state=SEED)
        m.fit(Xtr, ytr_cont)
    else:
        ytr = (ytr_cont > 0).astype(int)
        base = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                      random_state=SEED)
        m = AdaBoostClassifier(estimator=base, n_estimators=n_est,
                               learning_rate=lr, random_state=SEED)
        m.fit(Xtr, ytr)
    return m


def get_score(model, X, model_type):
    """Continuous score: regressor → predict, classifier → P(up)−0.5."""
    if model_type == 'regressor':
        return model.predict(X)
    else:
        return model.predict_proba(X)[:, 1] - 0.5


def evaluate_top5pct(model, test, model_type, pct=5):
    """
    Long the top {pct}% of predictions, short the bottom {pct}%.
    Returns dict of metrics.
    """
    te = test[test[TARGET].notna()]
    Xte = te[FEATURE_NAMES].values
    y_cont = te[TARGET].values
    sym = te['symbol'].values

    score = get_score(model, Xte, model_type)

    p_lo = np.percentile(score, pct)
    p_hi = np.percentile(score, 100 - pct)

    long_mask = score >= p_hi
    short_mask = score <= p_lo

    cost = COST_BPS / 1e4

    # Combined long+short PnL
    long_pnl = y_cont[long_mask] - cost
    short_pnl = -y_cont[short_mask] - cost

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    n_total = n_long + n_short

    all_pnl = np.concatenate([long_pnl, short_pnl])
    all_y = np.concatenate([y_cont[long_mask], -y_cont[short_mask]])  # signed

    sharpe = all_pnl.mean() / all_pnl.std() * ANNUALIZE if all_pnl.std() > 0 else 0
    win = (all_pnl > 0).mean()
    cum = np.cumsum(all_pnl)
    total_ret = cum[-1]
    max_dd = (np.maximum.accumulate(cum) - cum).max()

    # Per-leg
    long_sharpe = long_pnl.mean() / long_pnl.std() * ANNUALIZE if len(long_pnl) > 1 and long_pnl.std() > 0 else 0
    short_sharpe = short_pnl.mean() / short_pnl.std() * ANNUALIZE if len(short_pnl) > 1 and short_pnl.std() > 0 else 0

    return {
        'n_long': n_long, 'n_short': n_short, 'n_total': n_total,
        'long_sharpe': round(long_sharpe, 3),
        'short_sharpe': round(short_sharpe, 3),
        'combined_sharpe': round(sharpe, 3),
        'mean_bps': round(all_pnl.mean() * 1e4, 2),
        'long_bps': round(long_pnl.mean() * 1e4, 2),
        'short_bps': round(short_pnl.mean() * 1e4, 2),
        'win_pct': round(win, 4),
        'long_win_pct': round((long_pnl > 0).mean(), 4),
        'short_win_pct': round((short_pnl > 0).mean(), 4),
        'total_ret': round(total_ret, 4),
        'max_dd': round(max_dd, 4),
        'p_lo': float(p_lo),
        'p_hi': float(p_hi),
    }


def main():
    print('Loading data...')
    symbols = load_all('data/bars')
    train, test, _, _ = build_universe(symbols, seed=SEED)
    train = train[train[TARGET].notna()]
    print(f'Train {len(train):,} | Test {len(test[test[TARGET].notna()]):,}')

    # Read the sweep results, take top 20 by OOS Sharpe
    sweep = pd.read_csv('results/adaboost_clean_sweep.csv')
    top = sweep.sort_values('sharpe_oos', ascending=False).head(20)

    print(f'\nRe-fitting top 20 configs and evaluating top/bottom 5% on OOS...')
    print(f'(Long top 5%, short bottom 5%, net of {COST_BPS}bp cost)\n')

    print(f'{"#":>3}  {"type":>3} {"n":>3} {"d":>2} {"leaf":>5} {"lr":>5}  '
          f'{"orig_Sh":>7}  {"long_Sh":>7} {"short_Sh":>8} {"combo_Sh":>8}  '
          f'{"L_bps":>6} {"S_bps":>6}  {"L_win":>5} {"S_win":>5}  '
          f'{"ret":>7} {"DD":>6}')
    print('-' * 130)

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        n_est = int(r.n_estimators)
        d = int(r.max_depth)
        leaf = int(r.min_samples_leaf)
        lr = r.learning_rate
        mtype = r.model_type

        m = fit_model(train, n_est, d, leaf, lr, mtype)
        ev = evaluate_top5pct(m, test, mtype, pct=5)
        ev.update({'orig_oos_sharpe': r.sharpe_oos, 'rank': rank,
                   'n_estimators': n_est, 'max_depth': d,
                   'min_samples_leaf': leaf, 'learning_rate': lr,
                   'model_type': mtype})
        rows.append(ev)

        mt = 'reg' if mtype == 'regressor' else 'clf'
        print(f'{rank:>3}  {mt:>3} {n_est:>3} {d:>2} {leaf:>5} {lr:>5.2f}  '
              f'{r.sharpe_oos:>+7.3f}  '
              f'{ev["long_sharpe"]:>+7.3f} {ev["short_sharpe"]:>+8.3f} {ev["combined_sharpe"]:>+8.3f}  '
              f'{ev["long_bps"]:>+6.1f} {ev["short_bps"]:>+6.1f}  '
              f'{ev["long_win_pct"]*100:>4.1f}% {ev["short_win_pct"]*100:>4.1f}%  '
              f'{ev["total_ret"]*100:>+6.1f}% {ev["max_dd"]*100:>5.1f}%')

    df_out = pd.DataFrame(rows)
    df_out.to_csv('results/top5pct_eval.csv', index=False)
    print(f'\nSaved: results/top5pct_eval.csv')

    # Best by combined sharpe at top/bottom 5%
    print(f'\n{"="*100}')
    print('  TOP 5 BY COMBINED 5% SHARPE')
    print('=' * 100)
    df_out_sorted = df_out.sort_values('combined_sharpe', ascending=False).head(5)
    for _, r in df_out_sorted.iterrows():
        mt = 'reg' if r.model_type == 'regressor' else 'clf'
        print(f'  {mt} n={int(r.n_estimators):>3} d={int(r.max_depth)} leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  '
              f'orig_Sh={r.orig_oos_sharpe:+.3f}  '
              f'5pct_Sh={r.combined_sharpe:+.3f}  '
              f'L={r.long_sharpe:+.2f}/{r.long_bps:+.0f}bps  '
              f'S={r.short_sharpe:+.2f}/{r.short_bps:+.0f}bps')


if __name__ == '__main__':
    main()
