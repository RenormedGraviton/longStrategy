#!/usr/bin/env python3
"""
Simplified fit pipeline using ONLY 4 features at the 4-hour horizon:
  1. price_change_4h     — pct change in close
  2. oi_change_4h        — pct change in open interest
  3. volume_change_4h    — pct change in total volume
  4. cvd_4h              — Coinglass-style per-bar delta, (2*buy - vol) / vol
                            ∈ [-1, +1]

Targets (forward returns at four horizons, in 4h-bar units):
  target_4h   = next 1 bar  (close[t+1]/close[t] - 1)
  target_8h   = next 2 bars (close[t+2]/close[t] - 1)
  target_12h  = next 3 bars
  target_24h  = next 6 bars

Stages (use --stage to run up to N):
  1 = load one symbol's bars (sanity check)
  2 = compute features + 4 targets for one symbol (sanity check)
  3 = compute features across the universe + RANDOM 50/50 train/test split
  4 = fit AdaBoost regressor — one model per target horizon
  5 = OOS evaluation across all 4 horizons

Run: python3 fit_simple.py --stage 1
     python3 fit_simple.py --stage 2 --probe_symbol BTCUSDT
     python3 fit_simple.py --stage 5     (runs everything)
"""

import argparse
import os
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BAR_HOURS = 4
RESAMPLE = f'{BAR_HOURS}h'

# Forward target horizons in HOURS. Each maps to (h_bars = h_hours / BAR_HOURS).
TARGET_HORIZONS_H = [4, 8, 12, 24]
TARGET_BAR_STEPS = [h // BAR_HOURS for h in TARGET_HORIZONS_H]   # [1, 2, 3, 6]
TARGET_COLS = [f'target_{h}h' for h in TARGET_HORIZONS_H]


# ────────────────────────────────────────────────────────────────────
# Stage 1 — Load one symbol's bars
# ────────────────────────────────────────────────────────────────────

def load_bars(symbol, bars_dir='data/bars'):
    path = os.path.join(bars_dir, f'{symbol}.csv')
    if not os.path.exists(path):
        sys.exit(f'missing {path}')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def show_bars(df, symbol):
    print(f"\n{'='*70}")
    print(f"  STAGE 1 — Loaded {symbol}: {len(df)} hourly bars")
    print(f"{'='*70}")
    print(f"  Time range: {df.index.min()} → {df.index.max()}")
    print(f"  Columns:    {list(df.columns)}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())
    print(f"\n  Summary:")
    print(df[['close', 'volume', 'buy_volume', 'open_interest', 'funding_rate']].describe().round(4).to_string())


# ────────────────────────────────────────────────────────────────────
# Stage 2 — Build the 4 features (per symbol)
# ────────────────────────────────────────────────────────────────────

def build_features_simple(df):
    """Resample 1h → 4h and compute exactly 4 features + 4 forward-return targets.

    Aggregations on resample:
      - close       : last price in the 4h window
      - volume      : sum of hourly volumes
      - buy_volume  : sum of hourly taker-buy volumes
      - open_interest: last (snapshot at end of bar)

    Forward targets:
      target_4h  = close[t+1]/close[t] - 1   (next 1 bar)
      target_8h  = close[t+2]/close[t] - 1   (next 2 bars)
      target_12h = close[t+3]/close[t] - 1   (next 3 bars)
      target_24h = close[t+6]/close[t] - 1   (next 6 bars)
    Note: targets are OVERLAPPING across bars (e.g. target_24h at t and t+1
    share 5/6 of their window). Fine for training; matters for evaluation.
    """
    agg = {
        'close':         'last',
        'volume':        'sum',
        'buy_volume':    'sum',
        'open_interest': 'last',
        'funding_rate':  'last',   # snapshot at end of 4h bar
    }
    bars = df.resample(RESAMPLE).agg(agg).dropna()

    f = pd.DataFrame(index=bars.index)
    f['price_change_4h']  = bars['close'].pct_change(1)
    # OI — both percentage and dollar
    f['oi_change_pct']    = bars['open_interest'].pct_change(1)
    f['oi_change_dollar'] = bars['open_interest'].diff(1) * bars['close']
    # Volume — both percentage change and dollar level
    f['volume_change_pct'] = bars['volume'].pct_change(1)
    f['volume_dollar']     = bars['volume'] * bars['close']
    # CVD in USD: net taker buying in dollar terms over the 4h bar
    f['cvd_dollar']        = (2.0 * bars['buy_volume'] - bars['volume']) * bars['close']
    # Funding rate level — live predicted rate, updates ~60s on Binance.
    # Positive = longs pay shorts (bullish crowded), negative = shorts pay longs.
    f['funding_rate'] = bars['funding_rate']

    # Multi-horizon forward targets
    for h_hours, h_bars in zip(TARGET_HORIZONS_H, TARGET_BAR_STEPS):
        f[f'target_{h_hours}h'] = bars['close'].pct_change(h_bars).shift(-h_bars)

    return f


def show_features(f, symbol):
    feature_cols = ['price_change_4h', 'oi_change_pct', 'oi_change_dollar',
                    'volume_change_pct', 'volume_dollar', 'cvd_dollar']
    print(f"\n{'='*70}")
    print(f"  STAGE 2 — Features + 4 targets for {symbol}: {len(f)} 4h bars")
    print(f"{'='*70}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Targets  ({len(TARGET_COLS)}): {TARGET_COLS}")

    print(f"\n  First 3 valid rows (showing features + all 4 targets):")
    print(f.dropna().head(3).round(5).to_string())

    print(f"\n  Distribution stats:")
    print(f[feature_cols + TARGET_COLS].describe().round(5).to_string())

    print(f"\n  Feature correlations with each target:")
    print(f"  {'feature':<22}  {'target_4h':>10}  {'target_8h':>10}  {'target_12h':>10}  {'target_24h':>10}")
    for c in feature_cols:
        row = []
        for tc in TARGET_COLS:
            clean = f[[c, tc]].dropna()
            corr = clean[c].corr(clean[tc]) if len(clean) > 50 else float('nan')
            row.append(corr)
        print(f"  {c:<22}  " + "  ".join(f'{r:+10.4f}' for r in row))


# ────────────────────────────────────────────────────────────────────
# Stage 3 — Universe + train/test split (TODO in next stage)
# ────────────────────────────────────────────────────────────────────

def stage3_universe_split(args):
    """Apply walk-forward structural filter, build features for the
    surviving universe, do a random symbol-based train/test split."""
    print(f"\n{'='*70}")
    print(f"  STAGE 3 — Universe + train/test split")
    print(f"{'='*70}")

    files = sorted(glob.glob(os.path.join(args.bars_dir, '*.csv')))
    filt_dt = pd.Timestamp(args.filter_through_date)
    print(f"  Found {len(files)} symbol files")
    print(f"  Filter (computed on data < {filt_dt}):")
    print(f"    abs_ret <= {args.max_abs_ret}")
    print(f"    |skew|  <= {args.max_skew}")

    all_features = []
    skipped = []
    filtered = []

    for fp in files:
        symbol = os.path.basename(fp).replace('.csv', '')
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df) < 25:
                skipped.append((symbol, f'only {len(df)} bars')); continue
            if 'open_interest' not in df.columns or 'buy_volume' not in df.columns:
                skipped.append((symbol, 'no OI/buy_vol')); continue

            # Walk-forward filter: stats from in-sample (< filter_through_date) only
            df_filt = df[df.index < filt_dt]
            if len(df_filt) < 100:
                filtered.append((symbol, f'no in-sample ({len(df_filt)} bars)')); continue

            ret = df_filt['close'].pct_change()
            abs_ret = float(ret.abs().mean())
            sk = float(ret.skew())
            if abs_ret > args.max_abs_ret:
                filtered.append((symbol, f'abs_ret={abs_ret:.4f}')); continue
            if abs(sk) > args.max_skew:
                filtered.append((symbol, f'skew={sk:.2f}')); continue

            f_df = build_features_simple(df)
            f_df['symbol'] = symbol
            all_features.append(f_df)
        except Exception as e:
            skipped.append((symbol, str(e)[:40]))

    if not all_features:
        sys.exit('no surviving symbols after filter')

    combined = pd.concat(all_features)
    print(f"\n  Loaded:   {len(all_features)} symbols, {len(combined):>7} 4h-bar rows")
    print(f"  Filtered: {len(filtered)} symbols (sample: {filtered[:5]})")
    if skipped:
        print(f"  Skipped:  {len(skipped)} symbols (sample: {skipped[:3]})")

    feature_cols = ['price_change_4h', 'oi_change_pct', 'oi_change_dollar',
                    'volume_change_pct', 'volume_dollar', 'cvd_dollar']
    # Drop only rows where features are NaN — keep rows where SOME targets are
    # NaN (longer-horizon targets are NaN at the tail of each symbol's series)
    clean = combined.dropna(subset=feature_cols)
    print(f"\n  After feature dropna: {len(clean)} rows ({len(clean)/len(combined)*100:.0f}% kept)")

    # p1/p99 capping: compute percentiles on IN-SAMPLE data only,
    # then apply the same clip bounds to the FULL dataset (walk-forward safe).
    if args.cap_pct:
        lo_q = args.cap_pct / 100.0
        hi_q = 1.0 - lo_q
        insample_clean = clean[clean.index < pd.Timestamp(args.filter_through_date)]
        print(f"\n  Feature capping at p{args.cap_pct}/p{100-args.cap_pct} "
              f"(computed on {len(insample_clean):,} in-sample rows):")
        for fc in feature_cols:
            lo = float(insample_clean[fc].quantile(lo_q))
            hi = float(insample_clean[fc].quantile(hi_q))
            n_before = ((clean[fc] < lo) | (clean[fc] > hi)).sum()
            clean.loc[:, fc] = clean[fc].clip(lo, hi)
            print(f"    {fc:<22}  [{lo:+.6f}, {hi:+.6f}]  clipped {n_before:,} bars ({n_before/len(clean)*100:.1f}%)")

    # Cross-sectional demeaning: subtract the contemporaneous universe mean
    # from each target, so the model predicts RELATIVE outperformance, not
    # absolute return. Produces both positive and negative targets regardless
    # of whether the market was bullish or bearish overall.
    if args.demean:
        print(f"\n  Cross-sectional demeaning (--demean):")
        for tc in TARGET_COLS:
            before_mean = clean[tc].mean()
            cs_mean = clean.groupby(clean.index)[tc].transform('mean')
            clean.loc[:, tc] = clean[tc] - cs_mean
            after_mean = clean[tc].mean()
            pct_pos = (clean[tc] > 0).mean() * 100
            print(f"    {tc:<12} mean: {before_mean*1e4:+8.2f} → {after_mean*1e4:+8.2f} bps  "
                  f"({pct_pos:.1f}% positive after demean)")
    print(f"  Per-target available rows:")
    for tc in TARGET_COLS:
        n_valid = clean[tc].notna().sum()
        print(f"    {tc:<12}  {n_valid:>7d}  ({n_valid/len(clean)*100:.0f}%)")

    # Random 50/50 symbol split (in-sample / out-of-sample halves)
    all_syms = sorted(clean['symbol'].unique())
    rng = np.random.default_rng(args.seed)
    shuffled = [str(s) for s in rng.permutation(all_syms)]
    half = len(shuffled) // 2          # for 91 syms → 45 in-sample, 46 OOS
    in_sample_syms = sorted(shuffled[:half])
    oos_syms       = sorted(shuffled[half:])

    train = clean[clean['symbol'].isin(in_sample_syms)]
    test  = clean[clean['symbol'].isin(oos_syms)]

    print(f"\n  Random 50/50 symbol split (seed={args.seed}):")
    print(f"    Total surviving symbols: {len(all_syms)}")
    print(f"    In-sample: {len(in_sample_syms)} symbols / {len(train):>6} rows")
    print(f"    OOS:       {len(oos_syms)} symbols / {len(test):>6} rows")
    print(f"\n  In-sample symbols (first 12): {in_sample_syms[:12]}")
    print(f"  OOS       symbols (first 12): {oos_syms[:12]}")

    return clean, train, test, feature_cols, in_sample_syms, oos_syms


# ────────────────────────────────────────────────────────────────────
# Stage 4 — Fit a single shallow decision tree on target_4h
# ────────────────────────────────────────────────────────────────────

def stage4_fit_simple_tree(args, train, feature_cols):
    """Fit ONE DecisionTreeRegressor on a chosen forward-return target.
    Shallow tree, capped at args.tree_max_leaves leaf nodes — should be
    interpretable enough to read the rules off the tree directly."""
    target_col = args.target_col

    if args.exclude_features:
        dropped = [f.strip() for f in args.exclude_features.split(',') if f.strip()]
        feature_cols = [f for f in feature_cols if f not in dropped]
        print(f"\n  Excluded features: {dropped}")

    print(f"\n{'='*70}")
    print(f"  STAGE 4 — Simple decision tree (target = {target_col})")
    print(f"{'='*70}")
    print(f"  Features used ({len(feature_cols)}): {feature_cols}")
    print(f"  Tree config:")
    print(f"    max_depth         = {args.tree_max_depth}")
    print(f"    max_leaf_nodes    = {args.tree_max_leaves}")
    print(f"    min_samples_leaf  = {args.tree_min_samples_leaf:,}")
    print(f"    random_state      = {args.seed}")

    mask = train[target_col].notna()
    X = train.loc[mask, feature_cols].values
    y = train.loc[mask, target_col].values
    print(f"\n  In-sample rows: {len(y):,}  ({mask.sum()}/{len(train)})")
    print(f"  Target std:     {y.std():.6f}")
    print(f"  Target mean:    {y.mean():+.6f}")

    model = DecisionTreeRegressor(
        max_depth=args.tree_max_depth,
        max_leaf_nodes=args.tree_max_leaves,
        min_samples_leaf=args.tree_min_samples_leaf,
        random_state=args.seed,
    )
    model.fit(X, y)

    train_pred = model.predict(X)
    nz = train_pred != 0
    dir_acc_all   = (np.sign(train_pred) == np.sign(y)).mean()
    dir_acc_signed = ((np.sign(train_pred[nz]) == np.sign(y[nz])).mean()
                      if nz.any() else float('nan'))
    sse  = float(((y - train_pred) ** 2).sum())
    sst  = float(((y - y.mean()) ** 2).sum())
    r2   = 1 - sse / sst if sst > 0 else 0.0
    mae  = float(np.abs(y - train_pred).mean())

    print(f"\n  Tree shape (after fit):")
    print(f"    actual depth:    {model.get_depth()}")
    print(f"    leaf nodes:      {model.get_n_leaves()}")

    print(f"\n  In-sample fit quality:")
    print(f"    R²:                  {r2:+.6f}")
    print(f"    MAE:                 {mae:.6f}  ({mae*1e4:.2f} bps)")
    print(f"    Direction accuracy:  {dir_acc_all:.4f} (incl. zero preds)")
    print(f"    Direction accuracy:  {dir_acc_signed:.4f} (signed preds only)")
    print(f"    Pred std:            {train_pred.std():.6f}")
    print(f"    Pred range:          [{train_pred.min():+.6f}, {train_pred.max():+.6f}]")
    print(f"    Distinct pred vals:  {len(np.unique(train_pred))} (one per leaf)")

    print(f"\n  Feature importances:")
    imps = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    for name, imp in imps:
        bar = '█' * int(imp * 50)
        print(f"    {name:<22} {imp:.4f}  {bar}")

    print(f"\n  Tree (ASCII, with sample counts + node-mean target):")
    print_tree_pretty(model, feature_cols)

    print(f"\n  Rules (one per leaf, sorted by sample count):")
    print_tree_rules(model, feature_cols)

    out_pdf = args.tree_pdf_out
    save_tree_visual(model, feature_cols, target_col, out_pdf)

    return model


def save_tree_visual(model, feature_names, target_name, out_pdf):
    """Render the fitted tree using sklearn's plot_tree, save as PDF + PNG.

    Post-processes the default sklearn labels to show:
      - feature comparison (e.g. "price_change_4h <= +0.01156")
      - sample share ("38.6% of data")
      - prediction in bps ("predict −3.10 bps") at each leaf
    """
    fig, ax = plt.subplots(figsize=(20, 11))
    annotations = plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=12,
        precision=5,
        impurity=False,   # hide squared_error
        proportion=True,  # samples shown as fraction
    )

    # Rewrite each node's text in friendlier units
    for ann in annotations:
        text = ann.get_text()
        new_lines = []
        for line in text.split('\n'):
            s = line.strip()
            if s.startswith('value = '):
                try:
                    raw = s.replace('value = ', '').strip('[]').strip()
                    val = float(raw)
                    new_lines.append(f'predict {val * 1e4:+.2f} bps')
                except (ValueError, AttributeError):
                    new_lines.append(line)
            elif s.startswith('samples = '):
                try:
                    frac = float(s.replace('samples = ', ''))
                    new_lines.append(f'{frac * 100:.1f}% of data')
                except (ValueError, AttributeError):
                    new_lines.append(line)
            else:
                new_lines.append(line)
        ann.set_text('\n'.join(new_lines))

    fig.suptitle(
        f'Decision tree fit on {target_name}  '
        f'(depth={model.get_depth()}, leaves={model.get_n_leaves()}, '
        f'in-sample n={int(model.tree_.n_node_samples[0]):,})',
        fontsize=15, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_pdf) or '.', exist_ok=True)
    plt.savefig(out_pdf, bbox_inches='tight', format='pdf')
    out_png = out_pdf.replace('.pdf', '.png')
    plt.savefig(out_png, bbox_inches='tight', format='png', dpi=140)
    plt.close(fig)

    print(f"\n  Tree visualization saved:")
    print(f"    PDF: {out_pdf}")
    print(f"    PNG: {out_png}")


# ────────────────────────────────────────────────────────────────────
# Tree printers
# ────────────────────────────────────────────────────────────────────

def print_tree_pretty(model, feature_names):
    """Box-drawing ASCII tree with per-node sample counts and target means."""
    tree = model.tree_
    total_n = int(tree.n_node_samples[0])

    def walk(node_id, prefix, is_root, is_last):
        n = int(tree.n_node_samples[node_id])
        pct = n / total_n * 100
        mean_target = float(tree.value[node_id][0][0])

        if is_root:
            line_prefix = '    '
            child_prefix = '    '
        else:
            line_prefix = prefix + ('└── ' if is_last else '├── ')
            child_prefix = prefix + ('    ' if is_last else '│   ')

        if tree.feature[node_id] >= 0:  # internal split node
            feat = feature_names[tree.feature[node_id]]
            thr = float(tree.threshold[node_id])
            print(f"{line_prefix}{feat} ≤ {thr:+.5f}    "
                  f"[n={n:>6,d}  {pct:>5.1f}%  μ={mean_target*1e4:+8.2f} bps]")
            walk(int(tree.children_left[node_id]),  child_prefix, False, False)
            walk(int(tree.children_right[node_id]), child_prefix, False, True)
        else:  # leaf
            print(f"{line_prefix}LEAF → pred = {mean_target*1e4:+9.2f} bps    "
                  f"[n={n:>6,d}  {pct:>5.1f}%]")

    walk(0, '', True, True)


def print_tree_rules(model, feature_names):
    """One row per leaf: collapsed feature ranges + prediction + sample count.
    Sorted by sample count descending so the dominant rules show first."""
    tree = model.tree_
    rules = []

    def walk(node_id, conds):
        if tree.feature[node_id] < 0:  # leaf
            n = int(tree.n_node_samples[node_id])
            pred = float(tree.value[node_id][0][0])
            rules.append((n, pred, list(conds)))
            return
        feat = feature_names[tree.feature[node_id]]
        thr = float(tree.threshold[node_id])
        walk(int(tree.children_left[node_id]),  conds + [(feat, '<=', thr)])
        walk(int(tree.children_right[node_id]), conds + [(feat, '>',  thr)])

    walk(0, [])

    def collapse(conds):
        from collections import defaultdict
        bounds = defaultdict(lambda: {'lo': -np.inf, 'hi': np.inf})
        for feat, op, thr in conds:
            if op == '<=':
                bounds[feat]['hi'] = min(bounds[feat]['hi'], thr)
            else:
                bounds[feat]['lo'] = max(bounds[feat]['lo'], thr)
        parts = []
        for feat, bb in bounds.items():
            lo, hi = bb['lo'], bb['hi']
            if lo == -np.inf and hi == np.inf:
                continue
            if lo == -np.inf:
                parts.append(f"{feat} ≤ {hi:+.4f}")
            elif hi == np.inf:
                parts.append(f"{feat} > {lo:+.4f}")
            else:
                parts.append(f"{lo:+.4f} < {feat} ≤ {hi:+.4f}")
        return parts

    rules.sort(key=lambda r: -r[0])
    total = sum(r[0] for r in rules)
    print(f"    {'leaf':>4}  {'n':>8}  {'pct':>7}  {'pred (bps)':>11}   rule")
    print(f"    {'-'*4}  {'-'*8}  {'-'*7}  {'-'*11}   {'-'*60}")
    for i, (n, pred, conds) in enumerate(rules, 1):
        parts = collapse(conds)
        rule_str = '  AND  '.join(parts) if parts else '(always true)'
        pct = n / total * 100
        print(f"    {i:>4}  {n:>8,d}  {pct:>6.2f}%  {pred*1e4:>+11.2f}   {rule_str}")


# ────────────────────────────────────────────────────────────────────
# Stage 4b — AdaBoost Classifier
# ────────────────────────────────────────────────────────────────────

def stage4b_adaboost(args, train, test, feature_cols):
    """Fit AdaBoostClassifier (binary up/down) on in-sample symbols,
    evaluate on OOS symbols. Uses shallow DecisionTreeClassifier as
    base estimator."""
    target_col = args.target_col

    if args.exclude_features:
        dropped = [f.strip() for f in args.exclude_features.split(',') if f.strip()]
        feature_cols = [f for f in feature_cols if f not in dropped]

    print(f"\n{'='*70}")
    print(f"  STAGE 4b — AdaBoost Classifier (target = {target_col})")
    print(f"{'='*70}")
    print(f"  Features used ({len(feature_cols)}): {feature_cols}")
    print(f"  AdaBoost config:")
    print(f"    n_estimators        = {args.ada_n_estimators}")
    print(f"    base tree depth     = {args.ada_max_depth}")
    print(f"    base min_samples_leaf = {args.ada_min_samples_leaf:,}")
    print(f"    learning_rate       = {args.ada_learning_rate}")
    print(f"    random_state        = {args.seed}")

    # --- Prepare data ---
    tr_mask = train[target_col].notna()
    X_train = train.loc[tr_mask, feature_cols].values
    y_cont_train = train.loc[tr_mask, target_col].values
    y_train = (y_cont_train > 0).astype(int)  # 1=up, 0=down
    sym_train = train.loc[tr_mask, 'symbol'].values
    idx_train = train.loc[tr_mask].index

    te_mask = test[target_col].notna()
    X_test = test.loc[te_mask, feature_cols].values
    y_cont_test = test.loc[te_mask, target_col].values
    y_test = (y_cont_test > 0).astype(int)
    sym_test = test.loc[te_mask, 'symbol'].values
    idx_test = test.loc[te_mask].index

    print(f"\n  In-sample:  {len(y_train):,} rows, {len(np.unique(sym_train))} symbols, "
          f"{y_train.mean()*100:.1f}% positive")
    print(f"  OOS:        {len(y_test):,} rows, {len(np.unique(sym_test))} symbols, "
          f"{y_test.mean()*100:.1f}% positive")

    # --- Fit ---
    base_tree = DecisionTreeClassifier(
        max_depth=args.ada_max_depth,
        min_samples_leaf=args.ada_min_samples_leaf,
        random_state=args.seed,
    )
    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=args.ada_n_estimators,
        learning_rate=args.ada_learning_rate,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    # --- In-sample metrics ---
    pred_train = model.predict(X_train)
    prob_train = model.predict_proba(X_train)[:, 1]  # P(up)
    acc_train = (pred_train == y_train).mean()
    dir_acc_train = (pred_train == (y_cont_train > 0).astype(int)).mean()

    print(f"\n  In-sample results:")
    print(f"    Accuracy:            {acc_train:.4f}")
    print(f"    Direction accuracy:  {dir_acc_train:.4f}")
    print(f"    Predict-up rate:     {pred_train.mean()*100:.1f}%")

    # PnL: go long when predict up, short when predict down
    signal_train = np.where(pred_train == 1, 1.0, -1.0)
    pnl_train = signal_train * y_cont_train
    sharpe_train = pnl_train.mean() / pnl_train.std() * np.sqrt(6 * 365) if pnl_train.std() > 0 else 0
    cost_bps = args.cost_bps / 1e4
    pnl_train_net = pnl_train - cost_bps
    sharpe_train_net = pnl_train_net.mean() / pnl_train_net.std() * np.sqrt(6 * 365) if pnl_train_net.std() > 0 else 0

    print(f"    Mean return/trade:   {pnl_train.mean()*1e4:+.2f} bps (gross)")
    print(f"    Mean return/trade:   {pnl_train_net.mean()*1e4:+.2f} bps (net {args.cost_bps}bp RT)")
    print(f"    Sharpe (gross):      {sharpe_train:.2f}")
    print(f"    Sharpe (net):        {sharpe_train_net:.2f}")

    # --- OOS metrics ---
    pred_test = model.predict(X_test)
    prob_test = model.predict_proba(X_test)[:, 1]
    acc_test = (pred_test == y_test).mean()

    signal_test = np.where(pred_test == 1, 1.0, -1.0)
    pnl_test = signal_test * y_cont_test
    pnl_test_net = pnl_test - cost_bps
    sharpe_test = pnl_test.mean() / pnl_test.std() * np.sqrt(6 * 365) if pnl_test.std() > 0 else 0
    sharpe_test_net = pnl_test_net.mean() / pnl_test_net.std() * np.sqrt(6 * 365) if pnl_test_net.std() > 0 else 0

    cum_pnl_test = np.cumsum(pnl_test_net)
    max_dd = 0.0
    peak = cum_pnl_test[0]
    for v in cum_pnl_test:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    print(f"\n  OOS results:")
    print(f"    Accuracy:            {acc_test:.4f}")
    print(f"    Predict-up rate:     {pred_test.mean()*100:.1f}%")
    print(f"    Mean return/trade:   {pnl_test.mean()*1e4:+.2f} bps (gross)")
    print(f"    Mean return/trade:   {pnl_test_net.mean()*1e4:+.2f} bps (net {args.cost_bps}bp RT)")
    print(f"    Sharpe (gross):      {sharpe_test:.2f}")
    print(f"    Sharpe (net):        {sharpe_test_net:.2f}")
    print(f"    Total return (net):  {cum_pnl_test[-1]*100:+.2f}%")
    print(f"    Max drawdown:        {max_dd*100:.2f}%")
    print(f"    Win rate:            {(pnl_test > 0).mean()*100:.1f}%")
    print(f"    # trades:            {len(pnl_test):,}")

    # --- Per-symbol OOS breakdown ---
    print(f"\n  Per-symbol OOS breakdown (sorted by Sharpe):")
    print(f"    {'symbol':<14} {'n':>5} {'dir_acc':>8} {'mean_bps':>9} {'sharpe':>7} {'win%':>6}")
    print(f"    {'-'*14} {'-'*5} {'-'*8} {'-'*9} {'-'*7} {'-'*6}")
    sym_results = []
    for sym in sorted(np.unique(sym_test)):
        mask_s = sym_test == sym
        if mask_s.sum() < 10:
            continue
        pnl_s = pnl_test_net[mask_s]
        pred_s = pred_test[mask_s]
        y_s = y_test[mask_s]
        da = (pred_s == y_s).mean()
        sh = pnl_s.mean() / pnl_s.std() * np.sqrt(6 * 365) if pnl_s.std() > 0 else 0
        wr = (pnl_s > 0).mean()
        sym_results.append((sym, mask_s.sum(), da, pnl_s.mean() * 1e4, sh, wr))

    sym_results.sort(key=lambda x: -x[4])
    n_pos = sum(1 for r in sym_results if r[4] > 0)
    for sym, n, da, mbps, sh, wr in sym_results:
        print(f"    {sym:<14} {n:>5} {da:>8.3f} {mbps:>+9.2f} {sh:>+7.2f} {wr*100:>5.1f}%")

    print(f"\n  Summary: {n_pos}/{len(sym_results)} symbols profitable "
          f"({n_pos/len(sym_results)*100:.0f}%)")

    # --- Estimator weights ---
    print(f"\n  Estimator weights:")
    for i, w in enumerate(model.estimator_weights_):
        print(f"    Tree #{i+1}: weight={w:.4f}")

    # --- Feature importances (weighted by tree weights) ---
    fi = np.zeros(len(feature_cols))
    for est, w in zip(model.estimators_, model.estimator_weights_):
        fi += w * est.feature_importances_
    fi /= fi.sum()
    print(f"\n  Feature importances (weighted):")
    for name, imp in sorted(zip(feature_cols, fi), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"    {name:<22} {imp:.4f}  {bar}")

    # --- Visualize all trees ---
    n_trees = len(model.estimators_)
    cols = min(3, n_trees)
    rows = (n_trees + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n_trees == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (est, w) in enumerate(zip(model.estimators_, model.estimator_weights_)):
        ax = axes[i]
        plot_tree(est, feature_names=feature_cols, filled=True, rounded=True,
                  ax=ax, fontsize=8, precision=5, impurity=False, proportion=True)
        ax.set_title(f'Tree #{i+1} (weight={w:.3f})', fontsize=11, fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f'AdaBoost Classifier — Real {n_trees} trees (d={args.ada_max_depth}, '
        f'leaf>{args.ada_min_samples_leaf:,}, n={len(y_train):,})\n'
        f'OOS Sharpe(net)={sharpe_test_net:.2f}, Acc={acc_test:.3f}, '
        f'{n_pos}/{len(sym_results)} syms profitable',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_pdf = 'results/adaboost_trees.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', format='pdf')
    plt.savefig(out_pdf.replace('.pdf', '.png'), bbox_inches='tight', format='png', dpi=140)
    plt.close(fig)
    print(f"\n  Tree visualization saved: {out_pdf} / .png")

    return model


def main():

    p = argparse.ArgumentParser()
    p.add_argument('--stage',          type=int,   default=2, choices=[1, 2, 3, 4, 5])
    p.add_argument('--probe_symbol',   default='AIAUSDT',
                   help='Symbol used for stage 1/2 sanity checks')
    p.add_argument('--bars_dir',       default='data/bars')
    # Stage 3 args
    p.add_argument('--filter_through_date', default='2026-02-07')
    p.add_argument('--max_abs_ret',    type=float, default=0.010)
    p.add_argument('--max_skew',       type=float, default=3.0)
    p.add_argument('--seed',           type=int,   default=42)
    # Stage 4 args (simple tree)
    p.add_argument('--tree_max_depth',         type=int, default=4)
    p.add_argument('--tree_max_leaves',        type=int, default=8)
    p.add_argument('--tree_min_samples_leaf',  type=int, default=10000)
    p.add_argument('--tree_pdf_out',           default='results/tree_4h.pdf')
    p.add_argument('--exclude_features',       default='',
                   help='Comma-separated features to drop before fitting')
    p.add_argument('--target_col',             default='target_4h',
                   choices=['target_4h', 'target_8h', 'target_12h', 'target_24h'],
                   help='Which forward-return target to fit')
    p.add_argument('--demean', action='store_true', default=False,
                   help='Cross-sectionally demean targets (predict relative outperformance)')
    p.add_argument('--cap_pct', type=float, default=None,
                   help='Cap features at this percentile from each tail (e.g. 1 = p1/p99). Computed from in-sample only.')
    # AdaBoost args
    p.add_argument('--adaboost', action='store_true', default=False,
                   help='Run AdaBoost classifier instead of single tree in stage 4')
    p.add_argument('--ada_n_estimators', type=int, default=6)
    p.add_argument('--ada_max_depth', type=int, default=3)
    p.add_argument('--ada_min_samples_leaf', type=int, default=5000)
    p.add_argument('--ada_learning_rate', type=float, default=1.0)
    p.add_argument('--cost_bps', type=float, default=10,
                   help='Round-trip cost in bps for net PnL calculation')
    args = p.parse_args()

    if args.stage >= 1:
        df = load_bars(args.probe_symbol, args.bars_dir)
        show_bars(df, args.probe_symbol)

    if args.stage >= 2:
        f = build_features_simple(df)
        show_features(f, args.probe_symbol)

    train = test = feature_cols = None
    if args.stage >= 3:
        _, train, test, feature_cols, _, _ = stage3_universe_split(args)

    if args.stage >= 4:
        if train is None:
            sys.exit("Stage 4 needs Stage 3 to run first")
        if args.adaboost:
            stage4b_adaboost(args, train, test, feature_cols)
        else:
            stage4_fit_simple_tree(args, train, feature_cols)


if __name__ == '__main__':
    main()
