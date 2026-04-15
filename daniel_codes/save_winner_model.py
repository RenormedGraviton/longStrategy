#!/usr/bin/env python3
"""
Fit the winning AdaBoost classifier and save it (+ metadata) to disk.
Lets next session load the trained model directly without re-fitting.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn

FEATURE_NAMES = [
    'price_change_4h', 'oi_change_pct', 'oi_change_dollar',
    'volume_change_pct', 'volume_dollar', 'cvd_dollar',
]
TARGET = 'target_24h'
COST_BPS = 10
SEED = 42

# Winner config
N_EST = 100
DEPTH = 4
LEAF = 2000
LR = 0.10
TAIL_PCT = 5

OUT_DIR = 'models/winner_clf_20260411'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load frozen splits
    print('Loading frozen splits...')
    is_df  = pd.read_parquet('data/splits/is_data.parquet')
    oos_df = pd.read_parquet('data/splits/oos_data.parquet')
    caps = json.load(open('data/splits/cap_thresholds.json'))
    for fc, b in caps.items():
        is_df[fc]  = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])
    is_df  = is_df[is_df[TARGET].notna()]
    oos_df = oos_df[oos_df[TARGET].notna()]

    # Fit
    print(f'Fitting AdaBoostClassifier(n={N_EST}, d={DEPTH}, leaf={LEAF}, lr={LR})...')
    X_tr = is_df[FEATURE_NAMES].values
    y_tr = (is_df[TARGET].values > 0).astype(int)
    base = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=SEED)
    model = AdaBoostClassifier(estimator=base, n_estimators=N_EST,
                                learning_rate=LR, random_state=SEED)
    model.fit(X_tr, y_tr)
    print(f'  Fitted {len(model.estimators_)} trees')

    # Compute the IS-percentile thresholds for tail selection
    # These let you apply the same 5% cutoff in production without peeking at OOS
    is_score = model.predict_proba(X_tr)[:, 1] - 0.5
    is_p_lo  = float(np.percentile(is_score, TAIL_PCT))
    is_p_hi  = float(np.percentile(is_score, 100 - TAIL_PCT))

    # Also compute OOS thresholds (for reference — what the backtest used)
    X_te = oos_df[FEATURE_NAMES].values
    y_te = oos_df[TARGET].values
    oos_score = model.predict_proba(X_te)[:, 1] - 0.5
    oos_p_lo  = float(np.percentile(oos_score, TAIL_PCT))
    oos_p_hi  = float(np.percentile(oos_score, 100 - TAIL_PCT))

    # Evaluate on OOS with the OOS thresholds (what we reported)
    long_mask  = oos_score >= oos_p_hi
    short_mask = oos_score <= oos_p_lo
    cost = COST_BPS / 1e4
    long_pnl  = y_te[long_mask]  - cost
    short_pnl = -y_te[short_mask] - cost
    all_pnl   = np.concatenate([long_pnl, short_pnl])
    per_trade_sharpe = all_pnl.mean() / all_pnl.std() * np.sqrt(6 * 365) if all_pnl.std() > 0 else 0

    # Save model
    model_path = os.path.join(OUT_DIR, 'model.joblib')
    joblib.dump(model, model_path, compress=3)
    model_size = os.path.getsize(model_path) / 1024
    print(f'Saved model: {model_path} ({model_size:.0f} KB)')

    # Save metadata
    meta = {
        'created':        datetime.now().isoformat(timespec='seconds'),
        'name':           'winner_clf_20260411',
        'description':    'Best AdaBoost classifier from Phase 3.5 sweep on volatile-filtered universe. Top/bottom 5% tail-trading.',
        'model_type':     'AdaBoostClassifier',
        'hyperparameters': {
            'n_estimators':     N_EST,
            'max_depth':        DEPTH,
            'min_samples_leaf': LEAF,
            'learning_rate':    LR,
            'random_state':     SEED,
        },
        'features':       FEATURE_NAMES,
        'target':         TARGET,
        'bar_size':       '4h',
        'forward_horizon_hours': 24,
        'data_splits_dir':      'data/splits/',
        'trade_rule':           f'Long top {TAIL_PCT}% / Short bottom {TAIL_PCT}% of OOS predictions',
        'tail_pct':             TAIL_PCT,
        'is_thresholds': {
            'p_lo': is_p_lo,
            'p_hi': is_p_hi,
            'note': 'Use these as production cutoffs to avoid look-ahead.',
        },
        'oos_thresholds': {
            'p_lo': oos_p_lo,
            'p_hi': oos_p_hi,
            'note': 'Backtest used these (mild data snooping).',
        },
        'oos_metrics': {
            'per_trade_sharpe_annualized':  round(per_trade_sharpe, 3),
            'mean_bps_per_trade':           round(all_pnl.mean() * 1e4, 2),
            'long_bps':                     round(long_pnl.mean() * 1e4, 2),
            'short_bps':                    round(short_pnl.mean() * 1e4, 2),
            'n_long_trades':                int(long_mask.sum()),
            'n_short_trades':               int(short_mask.sum()),
            'win_rate':                     round(float((all_pnl > 0).mean()), 4),
            'daily_sharpe_annualized':      5.373,  # from plot_winner_eqwt.py
            'compounded_total_return':      3.1604, # +316%
            'max_drawdown':                 -0.1115,
            'window_start':                 '2025-10-05',
            'window_end':                   '2026-04-04',
            'cost_bps_round_trip':          COST_BPS,
        },
        'sklearn_version': sklearn.__version__,
        'n_is_rows':       len(is_df),
        'n_oos_rows':      len(oos_df),
    }
    meta_path = os.path.join(OUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved metadata: {meta_path}')

    # Save loader helper
    loader_path = os.path.join(OUT_DIR, 'load_and_predict.py')
    with open(loader_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Loader for the saved AdaBoost classifier — predicts & trades on new data."""
import json
import joblib
import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
MODEL = joblib.load(f"{HERE}/model.joblib")
META  = json.load(open(f"{HERE}/metadata.json"))
FEATURES = META["features"]

def predict_signals(df, use_is_thresholds=True):
    """
    Given a DataFrame with FEATURES columns, return (signal, score).
    signal:  +1 = long, -1 = short, 0 = no trade (middle of the distribution)
    score:   raw probability - 0.5

    If use_is_thresholds=True, applies the production thresholds learned
    from IS (no look-ahead). If False, computes thresholds from the
    current batch (backtest mode).
    """
    X = df[FEATURES].values
    score = MODEL.predict_proba(X)[:, 1] - 0.5
    if use_is_thresholds:
        p_lo = META["is_thresholds"]["p_lo"]
        p_hi = META["is_thresholds"]["p_hi"]
    else:
        p_lo = np.percentile(score, META["tail_pct"])
        p_hi = np.percentile(score, 100 - META["tail_pct"])
    signal = np.where(score >= p_hi,  1,
             np.where(score <= p_lo, -1, 0))
    return signal, score

if __name__ == "__main__":
    # Smoke test: load OOS split and predict
    oos = pd.read_parquet("data/splits/oos_data.parquet")
    caps = json.load(open("data/splits/cap_thresholds.json"))
    for fc, b in caps.items():
        oos[fc] = oos[fc].clip(b["p1"], b["p99"])
    oos = oos[oos["target_24h"].notna()]
    sig, sc = predict_signals(oos)
    print(f"Long  (+1): {(sig == +1).sum():,}")
    print(f"Short (-1): {(sig == -1).sum():,}")
    print(f"Flat   (0): {(sig ==  0).sum():,}")
''')
    print(f'Saved loader: {loader_path}')

    print(f'\n{"="*70}')
    print(f'  SANITY CHECK — OOS metrics (top/bottom {TAIL_PCT}%)')
    print(f'{"="*70}')
    print(f'  Per-trade Sharpe: {per_trade_sharpe:+.3f}')
    print(f'  Mean bps/trade:   {all_pnl.mean()*1e4:+.2f}')
    print(f'  Long trades:  {int(long_mask.sum())}  ({long_pnl.mean()*1e4:+.1f} bps)')
    print(f'  Short trades: {int(short_mask.sum())}  ({short_pnl.mean()*1e4:+.1f} bps)')
    print(f'  Win rate:     {(all_pnl > 0).mean()*100:.1f}%')

    print(f'\n{"="*70}')
    print('  HOW TO LOAD NEXT TIME')
    print(f'{"="*70}')
    print(f'''
import joblib, json
model = joblib.load("{OUT_DIR}/model.joblib")
meta  = json.load(open("{OUT_DIR}/metadata.json"))

# Or use the convenience loader:
import sys; sys.path.insert(0, "{OUT_DIR}")
from load_and_predict import predict_signals
signal, score = predict_signals(oos_df)
''')


if __name__ == '__main__':
    main()
