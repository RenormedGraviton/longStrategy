#!/usr/bin/env python3
"""
Train Random Forest to predict next-hour return on Binance perp.

Features (computed at end of hour t):
  - price_change_1h:    (close_t - close_{t-1}) / close_{t-1}
  - volume_change_1h:   (volume_t - volume_{t-1}) / volume_{t-1}
  - oi_change_1h:       (oi_t - oi_{t-1}) / oi_{t-1}
  - funding_change_1h:  funding_t - funding_{t-1}
                        Note: Tardis's `funding_rate` for binance-futures IS the
                        LIVE predicted/estimated rate (from Binance markPriceUpdate
                        `r` field, updates ~60s). It is NOT the stepwise realized
                        rate. This was verified empirically (BTC: 1022 changes/day).
  - funding_rate:       current live predicted funding rate (level)
  - Plus: longer-window changes (3h, 6h, 12h, 24h) for context

Target:
  - next_return = (close_{t+1} - close_t) / close_t

Model: RandomForestRegressor with walk-forward backtest.

Strategy (for backtest):
  - Long when prediction > +threshold
  - Short when prediction < -threshold
  - Hold cash otherwise
  - Position sized by signal strength (optional)
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix


def build_features(df, horizon=1):
    """Build feature DataFrame from raw hourly bars (single symbol).

    If `horizon` > 1, the 1h bars are first resampled to N-hour bars
    (OHLC aggregated, volume summed, OI/funding last). All "1h" feature
    names refer to ONE BAR, which is `horizon` hours after resampling.
    Target becomes the next-bar return = next N-hour return.
    """
    if horizon > 1:
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
               'volume': 'sum', 'open_interest': 'last', 'funding_rate': 'last'}
        if 'buy_volume' in df.columns:
            agg['buy_volume'] = 'sum'
        if 'quote_volume' in df.columns:
            agg['quote_volume'] = 'sum'
        df = df.resample(f'{horizon}h').agg(agg).dropna()

    f = pd.DataFrame(index=df.index)

    # Core 1-bar changes (= horizon hours)
    f['price_change_1h'] = df['close'].pct_change(1)
    f['volume_change_1h'] = df['volume'].pct_change(1)
    f['oi_change_1h'] = df['open_interest'].pct_change(1)
    # Tardis's `funding_rate` for binance-futures IS the LIVE predicted rate from
    # Binance's markPriceUpdate WebSocket (~60s updates), NOT the stepwise realized
    # rate. Tardis's separate `predicted_funding_rate` column is always NaN here.
    f['funding_change_1h'] = df['funding_rate'].diff(1)

    # Funding level (perp basis pressure) — live predicted, not realized
    f['funding_rate'] = df['funding_rate']

    # Multi-horizon changes
    for lag in [3, 6, 12, 24]:
        f[f'price_change_{lag}h'] = df['close'].pct_change(lag)
        f[f'volume_change_{lag}h'] = df['volume'].pct_change(lag)
        f[f'oi_change_{lag}h'] = df['open_interest'].pct_change(lag)

    # Realized volatility (last 24h)
    f['vol_24h'] = df['close'].pct_change().rolling(24).std()
    f['vol_6h'] = df['close'].pct_change().rolling(6).std()

    # Volume ratio (taker buy / total)
    if 'buy_volume' in df.columns:
        f['buy_ratio'] = df['buy_volume'] / df['volume'].replace(0, np.nan)

    # Range / spread proxy
    f['hl_range_pct'] = (df['high'] - df['low']) / df['close']

    # Interaction: OI change × price change (positive = trend confirmation)
    f['oi_x_price'] = f['oi_change_1h'] * f['price_change_1h']

    # NOTE: v2 added 11 more features (funding extremes/stats, OI vs MA,
    # liquidation proxies). They cannibalized signal from vol_24h and
    # collapsed pred std by 40%, dropping pooled Sharpe from 0.62 to 0.18.
    # Reverted to v1 22-feature baseline.

    # Target: next hour return
    f['target_return'] = df['close'].pct_change().shift(-1)

    return f


def load_universe(bars_dir='data/bars', min_bars=25,
                  max_abs_ret=None, max_skew=None,
                  min_oi_usd=None, max_oi_usd=None,
                  filter_through_date=None, horizon=1):
    """Load all per-symbol bar CSVs and combine into one DataFrame.

    Each row gets a 'symbol' column. Features are computed PER SYMBOL
    (so pct_change/diff don't cross symbol boundaries), then concatenated.

    `min_bars` defaults to 25 (24h horizon + 1 row) — just enough to compute
    one valid feature row. Lower threshold lets new-listing symbols (which
    have only post-split-date data) contribute their OOS rows to the pooled
    evaluation set, even though they have no training data.

    Optional structural filters (use to define a tradable universe):
      max_abs_ret  — drop symbols whose mean |hourly return| exceeds this.
                     Strongest empirical separator: 0.010 keeps the
                     low-vol symbols where the model has edge.
      max_skew     — drop symbols where |return skew| exceeds this. 3.0
                     drops fat-tailed pump-prone tokens.
      min_oi_usd   — drop symbols with mean USD open interest below this.
      max_oi_usd   — drop symbols with mean USD open interest above this.
    """
    files = sorted(glob.glob(os.path.join(bars_dir, '*.csv')))
    print(f"\nLoading {len(files)} symbol files from {bars_dir}...")
    if any(x is not None for x in [max_abs_ret, max_skew, min_oi_usd, max_oi_usd]):
        scope = f"in-sample only (< {filter_through_date})" if filter_through_date else "full period"
        print(f"  Filters [{scope}]: max_abs_ret={max_abs_ret}, max_|skew|={max_skew}, "
              f"oi_usd in [{min_oi_usd},{max_oi_usd}]")
    filt_dt = pd.Timestamp(filter_through_date) if filter_through_date else None

    all_features = []
    skipped = []
    filtered = []

    for f in files:
        symbol = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(df) < min_bars:
                skipped.append((symbol, len(df)))
                continue
            if 'open_interest' not in df.columns or 'funding_rate' not in df.columns:
                skipped.append((symbol, 'no OI/funding'))
                continue

            # Structural filter stats: full period by default, OR an
            # in-sample-only slice if filter_through_date is set (avoids
            # look-ahead — filter only "knows" about pre-OOS history).
            if filt_dt is not None:
                df_filt = df[df.index < filt_dt]
                if len(df_filt) < 100:
                    filtered.append((symbol, f'no in-sample (only {len(df_filt)} bars)'))
                    continue
            else:
                df_filt = df

            ret = df_filt['close'].pct_change()
            if max_abs_ret is not None:
                v = ret.abs().mean()
                if v > max_abs_ret:
                    filtered.append((symbol, f'abs_ret={v:.4f}'))
                    continue
            if max_skew is not None:
                v = ret.skew()
                if abs(v) > max_skew:
                    filtered.append((symbol, f'skew={v:.2f}'))
                    continue
            if min_oi_usd is not None or max_oi_usd is not None:
                oi_usd = (df_filt['open_interest'] * df_filt['close']).mean()
                if min_oi_usd is not None and oi_usd < min_oi_usd:
                    filtered.append((symbol, f'oi_usd=${oi_usd/1e6:.1f}M'))
                    continue
                if max_oi_usd is not None and oi_usd > max_oi_usd:
                    filtered.append((symbol, f'oi_usd=${oi_usd/1e6:.1f}M'))
                    continue

            features = build_features(df, horizon=horizon)
            features['symbol'] = symbol
            all_features.append(features)
        except Exception as e:
            skipped.append((symbol, str(e)[:40]))

    if not all_features:
        raise RuntimeError(f"No valid symbol files in {bars_dir}")

    combined = pd.concat(all_features)
    print(f"  Loaded:  {len(all_features)} symbols, {len(combined)} total rows")
    if filtered:
        print(f"  Filtered out: {len(filtered)} symbols (structural filter)")
        print(f"    Sample: {filtered[:8]}")
    if skipped:
        print(f"  Skipped: {len(skipped)} symbols (sample: {skipped[:5]})")

    return combined


def walk_forward_backtest(features, train_window=720, step=24, test_window=24,
                          model_type='regressor', threshold=0.001):
    """
    Walk-forward training: train on [t-train_window, t), test on [t, t+test_window).
    Slides forward by `step` hours per iteration.
    """
    feature_cols = [c for c in features.columns if c != 'target_return']
    X_all = features[feature_cols].values
    y_all = features['target_return'].values
    times = features.index

    n = len(features)
    predictions = np.full(n, np.nan)
    actuals = np.full(n, np.nan)

    feature_imps = []
    iter_count = 0

    print(f"\nWalk-forward backtest:")
    print(f"  Train window: {train_window} hours ({train_window/24:.1f} days)")
    print(f"  Test window:  {test_window} hours")
    print(f"  Step:         {step} hours")
    print(f"  Total bars:   {n}")
    print(f"  Features:     {len(feature_cols)}")

    start = train_window
    while start + test_window <= n:
        train_idx = slice(start - train_window, start)
        test_idx = slice(start, start + test_window)

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # Skip if too many NaNs
        train_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        test_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)

        if train_mask.sum() < 100:
            start += step
            continue

        if model_type == 'regressor':
            model = RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=20,
                n_jobs=-1, random_state=42)
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=20,
                n_jobs=-1, random_state=42)
            y_train_clf = (y_train > 0).astype(int)
            model.fit(X_train[train_mask], y_train_clf[train_mask])
            preds = model.predict_proba(X_test[test_mask])[:, 1] - 0.5  # center at 0
            test_indices = np.where(test_mask)[0]
            predictions[start + test_indices] = preds
            actuals[start + test_indices] = y_test[test_mask]
            feature_imps.append(model.feature_importances_)
            iter_count += 1
            start += step
            continue

        model.fit(X_train[train_mask], y_train[train_mask])
        preds = model.predict(X_test[test_mask])

        test_indices = np.where(test_mask)[0]
        predictions[start + test_indices] = preds
        actuals[start + test_indices] = y_test[test_mask]
        feature_imps.append(model.feature_importances_)

        iter_count += 1
        start += step

    print(f"  Iterations:   {iter_count}")

    # Aggregate feature importance
    avg_imp = np.mean(feature_imps, axis=0) if feature_imps else None

    return predictions, actuals, feature_cols, avg_imp


def apply_vol_gate(test_df, test_pred, filter_through_date,
                   vol_mult=2.0, lookback_bars=42):
    """Per-symbol vol regime gate: zero predictions when a symbol's recent
    realized vol exceeds `vol_mult` × its in-sample baseline vol.

    Baseline = stdev of realized returns from each symbol's bars BEFORE
    `filter_through_date` (mirrors the structural filter's in-sample window,
    so no look-ahead). Current vol = rolling std of realized returns over
    `lookback_bars`, computed causally (using `target_return.shift(1)`).

    Symbols without any in-sample data (newer listings) are NOT gated —
    they trade as usual. This is conservative; we don't penalize new tokens.

    Designed to address Mode A (SIRENUSDT-style symbol-specific blowups
    that the bull regime gate doesn't fully catch).
    """
    df = test_df.copy()
    df['realized'] = df.groupby('symbol')['target_return'].shift(1)

    filt_dt = pd.Timestamp(filter_through_date) if filter_through_date else None
    if filt_dt is None:
        # Fall back to first 60% of each symbol's history if no filter date
        baseline = df.groupby('symbol')['realized'].apply(
            lambda x: x.iloc[:max(1, int(len(x) * 0.6))].std()
        )
    else:
        insample = df[df.index < filt_dt]
        baseline = insample.groupby('symbol')['realized'].std()

    # Per-symbol rolling current vol (causal — operates on .shift(1) realized)
    min_p = max(5, lookback_bars // 2)
    df['curr_vol'] = df.groupby('symbol')['realized'].transform(
        lambda x: x.rolling(lookback_bars, min_periods=min_p).std()
    )
    df['baseline_vol'] = df['symbol'].map(baseline)
    df['vol_ratio'] = df['curr_vol'] / df['baseline_vol']
    gate_on = (df['vol_ratio'] > vol_mult).fillna(False)

    n_gated = int(gate_on.sum())
    pct = n_gated / max(len(df), 1) * 100
    n_with_baseline = baseline.notna().sum()
    print(f"\n  Vol gate: lookback={lookback_bars} bars, threshold=current_vol > {vol_mult}× baseline")
    print(f"    Symbols with baseline: {n_with_baseline}/{baseline.size}")
    print(f"    Median baseline vol:   {baseline.median():.5f}")
    print(f"    Gated {n_gated}/{len(df)} rows ({pct:.1f}%)")
    if n_gated > 0:
        sym_gated = df.assign(g=gate_on.values).groupby('symbol')['g'].sum().sort_values(ascending=False)
        sym_gated = sym_gated[sym_gated > 0]
        print(f"    Top 5 most-gated symbols:")
        for sym, n in sym_gated.head(5).items():
            print(f"      {sym:20s} {int(n):>5d} bars")

    return np.where(gate_on.values, 0.0, test_pred)


def apply_bull_regime_gate(test_df, test_pred, lookback_bars, threshold):
    """Zero out predictions during sustained cross-sectional bull regimes.

    Logic: at each timestamp, compute the equal-weight average of *realized*
    returns across all active symbols. Smooth with a rolling mean over
    `lookback_bars`. When this rolling mean exceeds `threshold`, the entire
    universe is gated to flat for that bar.

    Realized returns are computed as `target_return.shift(1)` per symbol —
    i.e., the return that has already happened by the start of bar t. This
    is causally available at decision time, no look-ahead.

    Mutates nothing; returns the gated prediction array.
    """
    df = test_df.copy()
    df['pred_orig'] = test_pred
    df['realized'] = df.groupby('symbol')['target_return'].shift(1)

    cs = df.groupby(df.index)['realized'].mean()
    min_p = max(5, lookback_bars // 2)
    cs_roll = cs.rolling(lookback_bars, min_periods=min_p).mean()
    gate_on = (cs_roll > threshold).fillna(False)

    df['_gate'] = df.index.map(gate_on).astype(bool)
    gated_pred = np.where(df['_gate'].values, 0.0, test_pred)

    pct_bars = gate_on.sum() / max(len(cs), 1) * 100
    pct_rows = df['_gate'].sum() / max(len(df), 1) * 100
    print(f"\n  Bull regime gate: lookback={lookback_bars} bars, threshold={threshold:+.5f}")
    print(f"    Gated {gate_on.sum()}/{len(cs)} timestamps ({pct_bars:.1f}%)")
    print(f"    Gated {df['_gate'].sum()}/{len(df)} rows ({pct_rows:.1f}%)")
    return gated_pred


def evaluate(predictions, actuals, feature_cols, avg_imp, threshold=0.001,
             cost_bps=0.0, horizon=1):
    """Compute metrics + simple long/short backtest PnL.

    `cost_bps` is the round-trip cost in bps (taker fees + slippage). Each
    leg (entry or exit) costs `cost_bps / 2`. A long→short flip costs a
    full round-trip. Default 0 = gross, no costs.

    `horizon` is the bar size in hours (for Sharpe annualization).
    """
    valid = ~np.isnan(predictions) & ~np.isnan(actuals)
    p = predictions[valid]
    a = actuals[valid]

    print(f"\n{'='*70}")
    print(f"  EVALUATION ({valid.sum()} valid predictions, cost={cost_bps:.1f} bps round-trip)")
    print(f"{'='*70}")

    # Regression metrics
    mae = mean_absolute_error(a, p)
    r2 = r2_score(a, p)
    print(f"  MAE:           {mae:.6f}")
    print(f"  R²:            {r2:.6f}")
    print(f"  Pred std:      {p.std():.6f}")
    print(f"  Actual std:    {a.std():.6f}")

    # Direction accuracy
    direction_correct = (np.sign(p) == np.sign(a)).mean()
    print(f"  Dir accuracy:  {direction_correct:.4f}  ({direction_correct*100:.1f}%)")

    # Simple long/short strategy
    # Long when prediction > threshold, short when < -threshold
    positions = np.zeros_like(p)
    positions[p > threshold] = 1
    positions[p < -threshold] = -1
    gross_returns = positions * a

    # Costs: each unit of |position change| = one leg = cost_bps/2.
    # NOTE: this flat array spans symbol boundaries. The boundary edges
    # over-charge by ~95 legs out of ~7000 — small enough to ignore here.
    pos_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
    cost_per_leg = (cost_bps / 2.0) / 1e4
    costs = pos_changes * cost_per_leg
    strategy_returns = gross_returns - costs
    total_legs = int(pos_changes.sum())
    total_cost = float(costs.sum())

    n_trades = (positions != 0).sum()
    win_rate = ((positions != 0) & (strategy_returns > 0)).sum() / max(n_trades, 1)
    avg_return = strategy_returns[positions != 0].mean() if n_trades > 0 else 0
    total_return = strategy_returns.sum()
    cum_returns = np.cumsum(strategy_returns)
    max_dd = np.min(cum_returns - np.maximum.accumulate(cum_returns))
    bars_per_year = (24 / horizon) * 365
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year) if strategy_returns.std() > 0 else 0

    print(f"\n  STRATEGY (threshold={threshold}, cost={cost_bps:.1f} bps RT, horizon={horizon}h):")
    print(f"  N bars long/short: {n_trades}")
    print(f"  N legs (entry/exit): {total_legs}")
    print(f"  Total cost paid:   {total_cost*100:+.2f}%")
    print(f"  Win rate:      {win_rate:.4f}  ({win_rate*100:.1f}%)")
    print(f"  Avg return:    {avg_return*10000:+.2f} bps per held bar (net)")
    print(f"  Total return:  {total_return*100:+.2f}%")
    print(f"  Max drawdown:  {max_dd*100:+.2f}%")
    print(f"  Sharpe (ann):  {sharpe:.2f}")

    # Buy & hold benchmark
    bh_return = a.sum()
    print(f"\n  BUY & HOLD:    {bh_return*100:+.2f}%")

    # Top features
    if avg_imp is not None:
        print(f"\n  TOP FEATURES (by avg importance):")
        imp_pairs = sorted(zip(feature_cols, avg_imp), key=lambda x: -x[1])
        for name, imp in imp_pairs[:10]:
            bar = '█' * int(imp * 200)
            print(f"    {name:<25} {imp:.4f}  {bar}")

    return {
        'mae': mae, 'r2': r2,
        'direction_acc': direction_correct,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'cum_returns': cum_returns,
        'positions': positions,
        'predictions': p,
        'actuals': a,
    }


def make_charts(results, out='results/backtest.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative returns
    ax = axes[0, 0]
    ax.plot(results['cum_returns'] * 100, color='#2ecc71', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Strategy Cumulative Return (%)', fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(alpha=0.3)

    # Prediction vs Actual scatter
    ax = axes[0, 1]
    ax.scatter(results['predictions'] * 10000, results['actuals'] * 10000,
               alpha=0.3, s=8, color='#3498db')
    lim = max(abs(results['predictions']).max(), abs(results['actuals']).max()) * 10000
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=0.8, alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f"Prediction vs Actual (dir acc {results['direction_acc']*100:.1f}%)",
                 fontweight='bold')
    ax.set_xlabel('Predicted return (bps)')
    ax.set_ylabel('Actual return (bps)')
    ax.grid(alpha=0.3)

    # Position histogram
    ax = axes[1, 0]
    pos_counts = pd.Series(results['positions']).value_counts().sort_index()
    colors = {-1: '#e74c3c', 0: '#95a5a6', 1: '#2ecc71'}
    ax.bar(pos_counts.index, pos_counts.values,
           color=[colors[p] for p in pos_counts.index], edgecolor='black')
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['SHORT', 'FLAT', 'LONG'])
    ax.set_title('Position Distribution', fontweight='bold')
    ax.set_ylabel('Hours')
    ax.grid(axis='y', alpha=0.3)

    # Returns histogram
    ax = axes[1, 1]
    ax.hist(results['actuals'] * 10000, bins=60, alpha=0.5, label='Actual',
            color='#3498db', edgecolor='black')
    ax.hist(results['predictions'] * 10000, bins=60, alpha=0.5, label='Predicted',
            color='#e67e22', edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Return Distribution', fontweight='bold')
    ax.set_xlabel('Return (bps)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nCharts saved: {out}")


def symbol_split(features, n_train_symbols=100, seed=42, model_type='regressor'):
    """
    Symbol-based train/test split: randomly pick `n_train_symbols` for in-sample
    training, use the remaining symbols entirely as out-of-sample.

    Tests cross-symbol generalization: can a model trained on one set of perps
    predict next-hour returns on perps it's never seen before? Each symbol's
    full time series goes to one side or the other (not split chronologically).

    `model_type`: 'regressor' fits actual return; 'classifier' fits binary
    direction (sign of target_return). Classifier predictions are returned as
    centered probabilities in [-0.5, +0.5] so the same threshold semantics
    work in evaluate() (positive = bullish, negative = bearish).
    """
    feature_cols = [c for c in features.columns if c not in ('target_return', 'symbol')]
    clean = features.dropna(subset=feature_cols + ['target_return'])

    all_syms = sorted(clean['symbol'].unique())
    rng = np.random.default_rng(seed)
    n_train = min(n_train_symbols, len(all_syms))
    train_syms = set(rng.choice(all_syms, size=n_train, replace=False).tolist())
    test_syms  = [s for s in all_syms if s not in train_syms]

    train = clean[clean['symbol'].isin(train_syms)]
    test  = clean[clean['symbol'].isin(test_syms)]

    print(f"\n{'='*70}")
    print(f"  SYMBOL-BASED TRAIN/TEST SPLIT (random, seed={seed}, {model_type})")
    print(f"{'='*70}")
    print(f"  Total symbols available: {len(all_syms)}")
    print(f"  Train symbols ({len(train_syms)}): {sorted(train_syms)[:10]}{'...' if len(train_syms)>10 else ''}")
    print(f"  Test  symbols ({len(test_syms)}):  {test_syms[:10]}{'...' if len(test_syms)>10 else ''}")
    print(f"  Train: {len(train):>8} rows, time range {train.index.min()} → {train.index.max()}")
    print(f"  Test:  {len(test):>8} rows, time range {test.index.min()} → {test.index.max()}")

    X_train = train[feature_cols].values
    y_train = train['target_return'].values
    X_test = test[feature_cols].values
    y_test = test['target_return'].values

    if model_type == 'classifier':
        # Binary direction target. Drop zero-return rows (no direction signal).
        nonzero = y_train != 0
        X_train_clf = X_train[nonzero]
        y_train_clf = (y_train[nonzero] > 0).astype(int)
        print(f"\n  Training RandomForestClassifier (300 trees, max_depth=10)...")
        print(f"  Class balance (train): up={y_train_clf.mean():.4f}")
        model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=50,
            n_jobs=8, random_state=seed, verbose=0)
        model.fit(X_train_clf, y_train_clf)
        # Centered probabilities: [-0.5, +0.5]; positive = bullish
        train_pred = model.predict_proba(X_train)[:, 1] - 0.5
        test_pred  = model.predict_proba(X_test)[:, 1]  - 0.5
    elif model_type == 'hybrid':
        # Hybrid: classifier predicts DIRECTION, regressor predicts MAGNITUDE.
        # Final pred = regressor's value, ZEROED OUT when classifier disagrees.
        # The threshold filter still operates on regressor units, so the cost
        # sweep / threshold tuning carries over from the regressor baseline.
        print(f"\n  Training hybrid (regressor + classifier, 300 trees each)...")
        reg = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=50,
            n_jobs=8, random_state=seed, verbose=0)
        reg.fit(X_train, y_train)

        nonzero = y_train != 0
        y_clf = (y_train[nonzero] > 0).astype(int)
        print(f"  Class balance (train): up={y_clf.mean():.4f}")
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=50,
            n_jobs=8, random_state=seed, verbose=0)
        clf.fit(X_train[nonzero], y_clf)

        reg_pred_train = reg.predict(X_train)
        reg_pred_test  = reg.predict(X_test)
        clf_pred_train = clf.predict_proba(X_train)[:, 1] - 0.5
        clf_pred_test  = clf.predict_proba(X_test)[:, 1]  - 0.5

        # Agreement: same sign (or one is zero). Where they disagree, go flat.
        agree_train = np.sign(reg_pred_train) == np.sign(clf_pred_train)
        agree_test  = np.sign(reg_pred_test)  == np.sign(clf_pred_test)
        train_pred = np.where(agree_train, reg_pred_train, 0.0)
        test_pred  = np.where(agree_test,  reg_pred_test,  0.0)

        agree_pct_test = agree_test.mean()
        print(f"  Models agree on direction: train={agree_train.mean()*100:.1f}%, test={agree_pct_test*100:.1f}%")
        # Use regressor's importances as the headline (magnitude carries it)
        model = reg
    else:
        print(f"\n  Training RandomForestRegressor (300 trees, max_depth=10)...")
        model = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=50,
            n_jobs=8, random_state=seed, verbose=0)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

    train_dir = (np.sign(train_pred) == np.sign(y_train)).mean()
    test_dir  = (np.sign(test_pred)  == np.sign(y_test)).mean()
    print(f"\n  In-sample direction accuracy:     {train_dir*100:.2f}%")
    print(f"  Out-of-sample direction accuracy: {test_dir*100:.2f}%")
    if train_dir - test_dir > 0.05:
        print(f"  *** WARNING: Overfitting gap {(train_dir-test_dir)*100:.1f}%")

    return test_pred, y_test, feature_cols, model.feature_importances_, test


def time_split(features, split_date):
    """
    Chronological train/test split for pooled multi-symbol data.
    Train: rows BEFORE split_date.
    Test:  rows AT/AFTER split_date.
    Cross-symbol — same time cut applies to all symbols.
    """
    feature_cols = [c for c in features.columns if c not in ('target_return', 'symbol')]
    clean = features.dropna(subset=feature_cols + ['target_return'])

    split_dt = pd.Timestamp(split_date)
    train = clean[clean.index < split_dt]
    test = clean[clean.index >= split_dt]

    print(f"\n{'='*70}")
    print(f"  CHRONOLOGICAL TRAIN/TEST SPLIT (pooled, multi-symbol)")
    print(f"{'='*70}")
    print(f"  Split date: {split_dt}")
    print(f"  Train: {len(train):>8} rows from {train.index.min()} to {train.index.max()}")
    print(f"         {train['symbol'].nunique()} symbols")
    print(f"  Test:  {len(test):>8} rows from {test.index.min()} to {test.index.max()}")
    print(f"         {test['symbol'].nunique()} symbols")

    X_train = train[feature_cols].values
    y_train = train['target_return'].values
    X_test = test[feature_cols].values
    y_test = test['target_return'].values

    print(f"\n  Training RandomForestRegressor (300 trees, max_depth=10)...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=50,
        n_jobs=8, random_state=42, verbose=0)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_dir = (np.sign(train_pred) == np.sign(y_train)).mean()
    test_dir = (np.sign(test_pred) == np.sign(y_test)).mean()
    print(f"\n  In-sample direction accuracy:     {train_dir*100:.2f}%")
    print(f"  Out-of-sample direction accuracy: {test_dir*100:.2f}%")
    if train_dir - test_dir > 0.05:
        print(f"  *** WARNING: Overfitting gap {(train_dir-test_dir)*100:.1f}%")

    return test_pred, y_test, feature_cols, model.feature_importances_, test


def evaluate_per_symbol(test_df, predictions, threshold=0.001, min_trades=30,
                        cost_bps=0.0, horizon=1):
    """Per-symbol breakdown of OOS prediction quality + strategy PnL.

    Symbols with fewer than `min_trades` non-zero positions over the OOS
    window are excluded — single-trade Sharpes (from symbols that fired
    once and won) were polluting the aggregates.

    `cost_bps` is the round-trip cost in bps. Each leg (entry or exit)
    costs `cost_bps / 2`. Computed correctly per-symbol (no boundary spillover).
    """
    print(f"\n{'='*70}")
    print(f"  PER-SYMBOL OOS PERFORMANCE  (min_trades={min_trades}, cost={cost_bps:.1f} bps RT)")
    print(f"{'='*70}")

    df = test_df.copy()
    df['pred'] = predictions
    df['actual'] = df['target_return']
    cost_per_leg = (cost_bps / 2.0) / 1e4

    rows = []
    for sym, g in df.groupby('symbol'):
        if len(g) < 50:
            continue
        positions = np.zeros(len(g))
        positions[g['pred'].values > threshold] = 1
        positions[g['pred'].values < -threshold] = -1
        gross_returns = positions * g['actual'].values

        # Per-symbol position-change cost
        pos_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
        costs = pos_changes * cost_per_leg
        strategy_returns = gross_returns - costs

        n_trades = (positions != 0).sum()
        if n_trades < min_trades:
            continue

        dir_acc = (np.sign(g['pred']) == np.sign(g['actual'])).mean()
        win_rate = ((positions != 0) & (strategy_returns > 0)).sum() / n_trades
        total_return = strategy_returns.sum()
        bars_per_year = (24 / horizon) * 365
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(bars_per_year) if strategy_returns.std() > 0 else 0

        rows.append({
            'symbol': sym, 'n_bars': len(g), 'n_trades': int(n_trades),
            'dir_acc': dir_acc, 'win_rate': win_rate,
            'total_return': total_return, 'sharpe': sharpe,
        })

    summary = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
    print(f"\n  TOP 20 by Sharpe:")
    print(summary.head(20).to_string(index=False))
    print(f"\n  BOTTOM 10 by Sharpe:")
    print(summary.tail(10).to_string(index=False))
    print(f"\n  AGGREGATE:")
    print(f"    Symbols traded:  {len(summary)}")
    print(f"    Profitable syms: {(summary['total_return'] > 0).sum()} ({(summary['total_return'] > 0).mean()*100:.1f}%)")
    print(f"    Mean total ret:  {summary['total_return'].mean()*100:+.2f}%")
    print(f"    Median total:    {summary['total_return'].median()*100:+.2f}%")
    print(f"    Mean Sharpe:     {summary['sharpe'].mean():.2f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bars_dir', default='data/bars',
                        help='Directory of per-symbol bar CSVs')
    parser.add_argument('--split_mode', default='symbols', choices=['symbols', 'time'],
                        help='symbols=random symbol holdout (default), time=chronological')
    parser.add_argument('--n_train_symbols', type=int, default=100,
                        help='[symbols mode] Number of symbols for in-sample training')
    parser.add_argument('--seed', type=int, default=42,
                        help='[symbols mode] RNG seed for symbol selection')
    parser.add_argument('--split_date', default='2026-02-07',
                        help='[time mode] Train/test cutoff (4mo in-sample / 2mo OOS)')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Signal threshold for long/short positions')
    parser.add_argument('--min_bars', type=int, default=25,
                        help='Min bars per symbol to include (default 25 = 1d horizon + 1)')
    parser.add_argument('--min_trades', type=int, default=30,
                        help='Min OOS trades per symbol for per-symbol summary (default 30)')
    parser.add_argument('--cost_bps', type=float, default=0.0,
                        help='Round-trip cost in bps (fees + slippage). Default 0 = gross')
    parser.add_argument('--max_abs_ret', type=float, default=None,
                        help='Filter: drop symbols with mean |return| above this (e.g. 0.010)')
    parser.add_argument('--max_skew', type=float, default=None,
                        help='Filter: drop symbols with |return skew| above this (e.g. 3.0)')
    parser.add_argument('--min_oi_usd_mil', type=float, default=None,
                        help='Filter: drop symbols with mean USD OI below this (in $M)')
    parser.add_argument('--max_oi_usd_mil', type=float, default=None,
                        help='Filter: drop symbols with mean USD OI above this (in $M)')
    parser.add_argument('--filter_through_date', default=None,
                        help='Compute filter stats only on data BEFORE this date (avoids look-ahead). e.g. 2026-02-07')
    parser.add_argument('--target_horizon', type=int, default=1,
                        help='Bar size in hours (resamples 1h bars to N-hour). Target = next N-hour return. Default 1')
    parser.add_argument('--model_type', default='regressor',
                        choices=['regressor', 'classifier', 'hybrid'],
                        help='regressor fits return; classifier fits binary direction; hybrid uses classifier as a directional gate on the regressor')
    parser.add_argument('--save_predictions', default=None,
                        help='Save per-bar test predictions to this parquet path')
    parser.add_argument('--bull_gate_threshold', type=float, default=None,
                        help='Bull regime gate: zero predictions when rolling cross-sectional realized return > this')
    parser.add_argument('--bull_gate_lookback', type=int, default=42,
                        help='Bull regime gate lookback in bars (default 42 = 7 days at 4h horizon)')
    parser.add_argument('--vol_gate_mult', type=float, default=None,
                        help='Per-symbol vol gate: zero predictions when current vol > mult × in-sample baseline (e.g. 2.0)')
    parser.add_argument('--vol_gate_lookback', type=int, default=42,
                        help='Vol gate lookback in bars (default 42 = 7 days at 4h horizon)')
    parser.add_argument('--out', default='results/backtest.png')
    args = parser.parse_args()

    # Load all symbols and compute features per symbol
    features = load_universe(
        args.bars_dir, min_bars=args.min_bars,
        max_abs_ret=args.max_abs_ret,
        max_skew=args.max_skew,
        min_oi_usd=(args.min_oi_usd_mil * 1e6 if args.min_oi_usd_mil else None),
        max_oi_usd=(args.max_oi_usd_mil * 1e6 if args.max_oi_usd_mil else None),
        filter_through_date=args.filter_through_date,
        horizon=args.target_horizon,
    )
    feature_cols = [c for c in features.columns if c not in ('target_return', 'symbol')]
    print(f"\n  Features ({len(feature_cols)}): {feature_cols}")

    if args.split_mode == 'symbols':
        test_pred, y_test, fcols, imp, test_df = symbol_split(
            features, n_train_symbols=args.n_train_symbols, seed=args.seed,
            model_type=args.model_type)

    # Optional gates (applied AFTER predictions, BEFORE evaluation)
    if args.bull_gate_threshold is not None and args.split_mode == 'symbols':
        test_pred = apply_bull_regime_gate(
            test_df, test_pred,
            lookback_bars=args.bull_gate_lookback,
            threshold=args.bull_gate_threshold)
    if args.vol_gate_mult is not None and args.split_mode == 'symbols':
        test_pred = apply_vol_gate(
            test_df, test_pred,
            filter_through_date=args.filter_through_date,
            vol_mult=args.vol_gate_mult,
            lookback_bars=args.vol_gate_lookback)
    else:
        test_pred, y_test, fcols, imp, test_df = time_split(features, args.split_date)

    # Aggregate evaluation
    results = evaluate(test_pred, y_test, fcols, imp,
                       threshold=args.threshold, cost_bps=args.cost_bps,
                       horizon=args.target_horizon)
    make_charts(results, args.out.replace('.png', '_oos.png'))

    # Per-symbol breakdown
    summary = evaluate_per_symbol(test_df, test_pred,
                                  threshold=args.threshold,
                                  min_trades=args.min_trades,
                                  cost_bps=args.cost_bps,
                                  horizon=args.target_horizon)

    # Optional: save per-bar test predictions for downstream analysis
    if args.save_predictions:
        out = test_df[['symbol', 'target_return']].copy()
        out['pred'] = test_pred
        out['threshold'] = args.threshold
        out['cost_bps'] = args.cost_bps
        out['horizon'] = args.target_horizon
        out.to_parquet(args.save_predictions)
        print(f"\nPer-bar predictions saved: {args.save_predictions} ({len(out)} rows)")
    summary.to_csv('results/per_symbol_oos.csv', index=False)
    print(f"\n  Per-symbol summary saved: results/per_symbol_oos.csv")


if __name__ == '__main__':
    main()
