#!/usr/bin/env python3
"""
Clean AdaBoost pipeline from scratch.

6 factors (4h bars) → 24h forward return.
Random 50/50 symbol split, p1/p99 capping.
"""

import os, glob, time, itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
import argparse


# ── Step 1: Load & resample ────────────────────────────────────────

def load_all(bars_dir):
    """Load all symbol CSVs (1h bars), return dict of raw DataFrames."""
    symbols = {}
    for fp in sorted(glob.glob(os.path.join(bars_dir, '*.csv'))):
        sym = os.path.basename(fp).replace('.csv', '')
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) < 25:
            continue
        required = ['close', 'volume', 'buy_volume', 'open_interest']
        if all(c in df.columns for c in required):
            symbols[sym] = df
    return symbols


def resample_4h(df_1h):
    """Resample 1h bars → 4h bars."""
    agg = {
        'close': 'last',
        'volume': 'sum',
        'buy_volume': 'sum',
        'open_interest': 'last',
    }
    return df_1h.resample('4h').agg(agg).dropna()


# ── Step 2: Compute features + target ──────────────────────────────

FEATURE_NAMES = [
    'price_change_4h',      # close pct_change
    'oi_change_pct',        # OI pct_change
    'oi_change_dollar',     # OI diff × close
    'volume_change_pct',    # volume pct_change
    'volume_dollar',        # volume × close (level, not change)
    'cvd_dollar',           # (2×buy_vol - vol) × close
]

def compute_features(bars4h):
    """From 4h OHLCV+OI bars, compute 6 features + forward-return targets.

    Targets (on 4h bars):
      target_4h  = 1 bar  ahead (next 4h)
      target_8h  = 2 bars ahead
      target_12h = 3 bars ahead
      target_24h = 6 bars ahead
    """
    f = pd.DataFrame(index=bars4h.index)

    f['price_change_4h']   = bars4h['close'].pct_change(1)
    f['oi_change_pct']     = bars4h['open_interest'].pct_change(1)
    f['oi_change_dollar']  = bars4h['open_interest'].diff(1) * bars4h['close']
    f['volume_change_pct'] = bars4h['volume'].pct_change(1)
    f['volume_dollar']     = bars4h['volume'] * bars4h['close']
    f['cvd_dollar']        = (2.0 * bars4h['buy_volume'] - bars4h['volume']) * bars4h['close']

    # Forward-return targets (all horizons)
    f['target_4h']  = bars4h['close'].pct_change(1).shift(-1)
    f['target_8h']  = bars4h['close'].pct_change(2).shift(-2)
    f['target_12h'] = bars4h['close'].pct_change(3).shift(-3)
    f['target_24h'] = bars4h['close'].pct_change(6).shift(-6)

    return f


# ── Step 3: Filter, split, cap ─────────────────────────────────────

def build_universe(symbols_dict, seed=42, cap_pct=1.0):
    """
    1. Use ALL symbols (no filtering)
    2. Random 50/50 symbol split
    3. p1/p99 feature capping from IS symbols only
    """
    surviving = []

    for sym, df_1h in symbols_dict.items():
        bars4h = resample_4h(df_1h)
        if len(bars4h) < 10:
            continue
        feats = compute_features(bars4h)
        feats['symbol'] = sym
        surviving.append(feats)

    combined = pd.concat(surviving)
    # Drop rows where any feature is NaN (keep NaN targets — they get filtered per-target)
    combined = combined.dropna(subset=FEATURE_NAMES)

    all_syms = sorted(combined['symbol'].unique())
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(all_syms))
    half = len(shuffled) // 2
    is_syms = sorted(shuffled[:half])
    oos_syms = sorted(shuffled[half:])

    train = combined[combined['symbol'].isin(is_syms)].copy()
    test = combined[combined['symbol'].isin(oos_syms)].copy()

    # p1/p99 capping computed from IS only
    if cap_pct:
        lo_q = cap_pct / 100.0
        hi_q = 1.0 - lo_q
        for fc in FEATURE_NAMES:
            lo = float(train[fc].quantile(lo_q))
            hi = float(train[fc].quantile(hi_q))
            train[fc] = train[fc].clip(lo, hi)
            test[fc] = test[fc].clip(lo, hi)

    print(f"Universe: {len(all_syms)} symbols → {len(is_syms)} IS / {len(oos_syms)} OOS")
    print(f"Rows: {len(train):,} IS / {len(test):,} OOS")

    return train, test, is_syms, oos_syms


# ── Step 4: Fit & evaluate ─────────────────────────────────────────

ANNUALIZE = np.sqrt(6 * 365)  # 6 bars/day

def fit_and_eval(train, test, n_est, depth, leaf, lr, cost_bps=10, seed=42,
                 model_type='classifier'):
    """Fit one AdaBoost config, return metrics dict."""
    target = 'target_24h'

    tr = train[train[target].notna()]
    te = test[test[target].notna()]

    X_tr = tr[FEATURE_NAMES].values
    y_cont_tr = tr[target].values
    sym_tr = tr['symbol'].values

    X_te = te[FEATURE_NAMES].values
    y_cont_te = te[target].values
    sym_te = te['symbol'].values

    cost = cost_bps / 1e4

    if model_type == 'classifier':
        y_tr = (y_cont_tr > 0).astype(int)
        y_te = (y_cont_te > 0).astype(int)

        base = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, random_state=seed)
        model = AdaBoostClassifier(
            estimator=base, n_estimators=n_est,
            learning_rate=lr, random_state=seed)
        model.fit(X_tr, y_tr)

        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        sig_tr = np.where(pred_tr == 1, 1.0, -1.0)
        sig_te = np.where(pred_te == 1, 1.0, -1.0)
        acc_te = (pred_te == y_te).mean()
    else:
        # Regressor: sign of prediction = direction
        base = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=leaf, random_state=seed)
        model = AdaBoostRegressor(
            estimator=base, n_estimators=n_est,
            learning_rate=lr, random_state=seed)
        model.fit(X_tr, y_cont_tr)

        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        sig_tr = np.sign(pred_tr)
        sig_te = np.sign(pred_te)
        acc_te = ((sig_te > 0) == (y_cont_te > 0)).mean()

    # PnL
    pnl_tr = sig_tr * y_cont_tr - cost
    pnl_te = sig_te * y_cont_te - cost

    sharpe_tr = pnl_tr.mean() / pnl_tr.std() * ANNUALIZE if pnl_tr.std() > 0 else 0
    sharpe_te = pnl_te.mean() / pnl_te.std() * ANNUALIZE if pnl_te.std() > 0 else 0

    cum = np.cumsum(pnl_te)
    total_ret = cum[-1]
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    # Per-symbol
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
    }, model


# ── Main ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bars_dir', default='data/bars')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cost_bps', type=float, default=10)
    p.add_argument('--out', default='results/adaboost_clean_sweep.csv')
    args = p.parse_args()

    # Load data once
    print("Loading bars...")
    t0 = time.time()
    symbols = load_all(args.bars_dir)
    print(f"Loaded {len(symbols)} symbols in {time.time()-t0:.1f}s")

    train, test, is_syms, oos_syms = build_universe(
        symbols, seed=args.seed)

    # ── Sweep grid (focused on d=2, d=3, but also d=1,4,5,6 for comparison) ──
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
    print(f"\nSweeping {len(combos)} combinations...")
    print(f"{'#':>5}  {'type':>5} {'n_est':>5} {'d':>2} {'leaf':>5} {'lr':>5}  "
          f"{'IS':>6} {'OOS':>6} {'acc':>5} {'bps':>6} {'prof':>5}")
    print("-" * 75)

    results = []
    best_sharpe = -999
    t_start = time.time()

    for i, (mtype, n_est, depth, leaf, lr) in enumerate(combos):
        try:
            r, _ = fit_and_eval(train, test, n_est, depth, leaf, lr,
                                cost_bps=args.cost_bps, seed=args.seed,
                                model_type=mtype)
            results.append(r)

            marker = ""
            if r['sharpe_oos'] > best_sharpe:
                best_sharpe = r['sharpe_oos']
                marker = " ***"

            mt = 'clf' if mtype == 'classifier' else 'reg'
            print(f"{i+1:>5}  {mt:>5} {n_est:>5} {depth:>2} {leaf:>5} {lr:>5.2f}  "
                  f"{r['sharpe_is']:>+6.2f} {r['sharpe_oos']:>+6.2f} "
                  f"{r['acc_oos']:>5.3f} {r['mean_bps_oos']:>+6.1f} "
                  f"{r['n_profitable']:>3}/{r['n_symbols']:<2}{marker}")
        except Exception as e:
            print(f"{i+1:>5}  ERROR: {e}")

        if (i + 1) % 100 == 0 and results:
            pd.DataFrame(results).to_csv(args.out, index=False)
            elapsed = time.time() - t_start
            remaining = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  ... checkpoint {len(results)} saved, "
                  f"~{remaining/60:.0f}min remaining")

    total = time.time() - t_start
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*75}")
    print(f"Done: {len(results)} configs in {total/60:.1f} min")
    print(f"Saved: {args.out}")

    # ── Results ──
    for mtype in ['classifier', 'regressor']:
        sub = df[df.model_type == mtype]
        if sub.empty:
            continue
        print(f"\n{'='*75}")
        print(f"  TOP 15 {mtype.upper()} by OOS Sharpe")
        print(f"{'='*75}")
        top = sub.sort_values('sharpe_oos', ascending=False).head(15)
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(f"  {rank:>2}. n={int(r.n_estimators):>3} d={int(r.max_depth)} "
                  f"leaf={int(r.min_samples_leaf):>5} lr={r.learning_rate:.2f}  "
                  f"IS={r.sharpe_is:+.2f}  OOS={r.sharpe_oos:+.3f}  "
                  f"acc={r.acc_oos:.3f}  bps={r.mean_bps_oos:+.1f}  "
                  f"prof={int(r.n_profitable)}/{int(r.n_symbols)}  "
                  f"med_sh={r.median_sym_sharpe:+.2f}")

    # ── Best per depth ──
    print(f"\n{'='*75}")
    print(f"  BEST OOS SHARPE PER DEPTH (both model types)")
    print(f"{'='*75}")
    for d in sorted(df.max_depth.unique()):
        sub = df[df.max_depth == d]
        best = sub.loc[sub.sharpe_oos.idxmax()]
        mt = 'clf' if best.model_type == 'classifier' else 'reg'
        print(f"  d={d}  {mt} n={int(best.n_estimators):>3} "
              f"leaf={int(best.min_samples_leaf):>5} lr={best.learning_rate:.2f}  "
              f"OOS={best.sharpe_oos:+.3f}  bps={best.mean_bps_oos:+.1f}  "
              f"prof={int(best.n_profitable)}/{int(best.n_symbols)}")


if __name__ == '__main__':
    main()
