#!/usr/bin/env python3
"""5-fold chronological CV training of 3 models on agg_futures_data features.

Models:
  1. pump_label  binary, scale_pos_weight = 0.1 * neg/pos, 100 trees
  2. sign(ret_24h) binary (target = (sign(ret_24h)+1)/2), 300 trees
  3. sigmoid(ret_24h/0.1) continuous target with cross_entropy loss, 500 trees

All models: learning_rate=0.01, chronological 5-fold on O['date'].

Usage (from project root, with conda env ml):
    python3 scripts/cv_3models.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import codes.toolbox as tb

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DATA_FOLDER = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DATASET = os.path.join(RESULTS_DIR, 'cv_dataset.pkl')
OUT_MODELS    = os.path.join(RESULTS_DIR, 'cv_models.pkl')
OUT_EVENTS    = os.path.join(RESULTS_DIR, 'cv_events.pkl')

MARKETS     = 'binance'
INSTRUMENTS = 'futures'
BAR_FREQ    = 4
SAMPLE_FREQ = 1
HORIZONS    = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 72]

PUMP_THR    = 1.5
PUMP_FWD_H  = 7 * 24
PUMP_MIN_H  = 3 * 24

N_FOLDS         = 5
COOLDOWN_HOURS  = 24
LONG_CUT_Q      = 0.95
EVENT_FWD_HOURS = 14 * 24   # 336

FEATURES = [
    'dprice_es4', 'dprice_ratio_es4', 'dprice_es8', 'dprice_ratio_es8',
    'dprice_es24', 'dprice_ratio_es24',
    'oi_es4', 'doi_es4', 'doi_ratio_es4',
    'oi_native_es4', 'doi_native_es4', 'doi_native_ratio_es4',
    'cvd_es4', 'cvd_es8', 'cvd_es24',
    'dcvd_es4', 'dcvd_es8', 'dcvd_es24',
    'dcvd_ratio_es4', 'dcvd_ratio_es8', 'dcvd_ratio_es24',
    'cvd_vol_es4', 'cvd_vol_es8', 'dcvd_vol_es4', 'dcvd_vol_es8',
    'abs_cvd_ema240',
    'cvd_es4_norm', 'dcvd_es4_norm',
    'cvd_es8_norm', 'dcvd_es8_norm',
    'cvd_es24_norm', 'dcvd_es24_norm',
    'd_abs_cvd_es_bar_es4', 'd_abs_cvd_es_bar_ratio_es4', 'd_abs_cvd_es_bar_es4_norm',
    'spot_cvd_es4', 'spot_cvd_es8', 'spot_cvd_es24',
    'dspot_cvd_es4', 'dspot_cvd_es8', 'dspot_cvd_es24',
    'dspot_cvd_ratio_es4', 'dspot_cvd_ratio_es8', 'dspot_cvd_ratio_es24',
    'spot_cvd_es4_norm', 'dspot_cvd_es4_norm',
    'spot_cvd_es8_norm', 'dspot_cvd_es8_norm',
    'spot_cvd_es24_norm', 'dspot_cvd_es24_norm',
    'd_abs_spot_cvd_es_bar_es4', 'd_abs_spot_cvd_es_bar_ratio_es4',
    'd_abs_spot_cvd_es_bar_es4_norm',
    'funding_rate', 'funding_rate_avg',
    'spot_fut_vol_ratio', 'spot_fut_vol_es24_ratio',
]

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ts_from_date_hour(date_col, hour_col):
    """Vector of UTC timestamps from date (int YYYYMMDD) and hour."""
    d = pd.Series(date_col).astype(int).astype(str).values
    h = pd.Series(hour_col).astype(int).values
    return pd.to_datetime(d, format='%Y%m%d', utc=True) + pd.to_timedelta(h, unit='h')


# ──────────────────────────────────────────────────────────────
# Step 1+2 — load cached dataset or build
# ──────────────────────────────────────────────────────────────
def build_or_load_dataset():
    if os.path.exists(CACHE_DATASET):
        print(f"[cache] loading dataset from {CACHE_DATASET}")
        d = tb.loadPickle(CACHE_DATASET)
        return d['X'], d['Y'], d['O']

    print("[build] loading DataLoader and computing agg_futures_data (slow)...")
    dl = tb.DataLoader(DATA_FOLDER)
    print(f"  {dl}")
    print(f"  building features on {len(dl.symbols)} symbols...")
    t0 = time.time()
    df = tb.agg_futures_data(dl, dl.symbols, MARKETS, INSTRUMENTS,
                             barFreq=BAR_FREQ, sampleFreq=SAMPLE_FREQ)
    print(f"  agg_futures_data: {df.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    returns = tb.makeReturns(
        dl, df,
        horizons=HORIZONS,
        adjustFundingRate=True,
        addPumpLabel=True,
        pumpThreshold=PUMP_THR,
        pumpForwardHours=PUMP_FWD_H,
        pumpMinHistoryHours=PUMP_MIN_H,
    )
    print(f"  makeReturns: {returns.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    X, Y, O = tb.makeXY(df, returns, FEATURES, appender=['volume_usdt'])
    print(f"  makeXY: X={X.shape}, Y={Y.shape}, O={O.shape} in {time.time()-t0:.1f}s")

    tb.savePickle({'X': X, 'Y': Y, 'O': O}, CACHE_DATASET)
    print(f"[cache] saved to {CACHE_DATASET}")
    return X, Y, O


# ──────────────────────────────────────────────────────────────
# Step 3 — chronological 5-fold date split
# ──────────────────────────────────────────────────────────────
def build_folds(O, n_folds=N_FOLDS):
    unique_dates = sorted(O['date'].unique())
    chunks = np.array_split(unique_dates, n_folds)
    folds = []
    for k, val_dates in enumerate(chunks):
        val_dates = set(int(d) for d in val_dates)
        train_dates = set(int(d) for d in unique_dates) - val_dates
        train_mask = O['date'].isin(train_dates).values
        val_mask   = O['date'].isin(val_dates).values
        folds.append({
            'k': k,
            'train_dates': sorted(train_dates),
            'val_dates':   sorted(val_dates),
            'train_mask':  train_mask,
            'val_mask':    val_mask,
        })
        print(f"  fold {k}: train={train_mask.sum()} rows ({len(train_dates)} days), "
              f"val={val_mask.sum()} rows ({len(val_dates)} days) "
              f"[{min(val_dates)} .. {max(val_dates)}]")
    return folds


# ──────────────────────────────────────────────────────────────
# Step 4 — fit three models per fold
# ──────────────────────────────────────────────────────────────
BASE_CLS = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': 2,
    'min_child_samples': 50,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'colsample_bynode': 1.0,
    'num_threads': -1,
    'verbose': -1,
}
BASE_REG = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': 4,
    'min_child_samples': 20,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'colsample_bynode': 1.0,
    'num_threads': -1,
    'verbose': -1,
}


def fit_three_models(X_tr_n, X_v_n, X_all_n, Y_tr, Y_v):
    """Fit all three models on one fold, return models + full predictions."""
    # ── Model 1: pump_label ──
    pump_tr = Y_tr['pump_label'].values
    ok1 = ~np.isnan(pump_tr)
    pos = float(np.sum(pump_tr[ok1] == 1))
    neg = float(np.sum(pump_tr[ok1] == 0))
    scale_pos = 0.1 * neg / max(pos, 1.0)
    p1 = {**BASE_CLS,
          'objective': 'binary', 'metric': ['binary_logloss', 'auc'],
          'n_estimators': 100, 'scale_pos_weight': scale_pos}
    print(f"    model 1: pos={int(pos)}, neg={int(neg)}, scale_pos_weight={scale_pos:.2f}")
    m1, _ = tb.fitLgb(X_tr_n[ok1], pump_tr[ok1], p1, log_period=0)

    # ── Model 2: sign ──
    ret_tr = Y_tr['ret_24h'].values
    ok2 = ~np.isnan(ret_tr)
    sign_target = (np.sign(ret_tr[ok2]) + 1.0) / 2.0
    # drop sign==0 bars (ambiguous)
    keep2 = ret_tr[ok2] != 0
    p2 = {**BASE_CLS,
          'objective': 'binary', 'metric': ['binary_logloss', 'auc'],
          'n_estimators': 300}
    m2, _ = tb.fitLgb(X_tr_n[ok2][keep2], sign_target[keep2], p2, log_period=0)

    # ── Model 3: sigmoid(ret/0.1) ──
    ok3 = ~np.isnan(ret_tr)
    y3_tr = sigmoid(ret_tr[ok3] / 0.1)
    p3 = {**BASE_REG,
          'objective': 'cross_entropy', 'metric': ['cross_entropy'],
          'n_estimators': 500}
    m3, _ = tb.fitLgb(X_tr_n[ok3], y3_tr, p3, log_period=0)

    # predict on ALL rows (train + val + any row in X_all) — needed for Step 6 lookup
    p1_all = m1.predict(X_all_n)
    p2_all = m2.predict(X_all_n)
    p3_all = m3.predict(X_all_n)

    return m1, m2, m3, p1_all, p2_all, p3_all


# ──────────────────────────────────────────────────────────────
# Step 5 — diagnostics plots
# ──────────────────────────────────────────────────────────────
def _precision_curve(pred, y):
    """Return (thresholds, precision, TP, FP, n_pred_pos) truncated at the tail."""
    mask = ~(np.isnan(pred) | np.isnan(y))
    pred, y = pred[mask], y[mask].astype(int)
    order = np.argsort(-pred)
    p_sorted = pred[order]
    y_sorted = y[order]
    TP = np.cumsum(y_sorted == 1)
    FP = np.cumsum(y_sorted == 0)
    n_pred_pos = np.arange(1, len(y_sorted) + 1)
    precision = TP / n_pred_pos
    total_pos = int(y.sum())
    fp_tp = np.where(TP > 0, FP / np.maximum(TP, 1), np.inf)
    stop_mask = (fp_tp > 5) & (n_pred_pos > 2 * max(total_pos, 1))
    if stop_mask.any():
        stop = np.argmax(stop_mask) + 1
        p_sorted = p_sorted[:stop]
        precision = precision[:stop]
        TP = TP[:stop]
        FP = FP[:stop]
    base_rate = y.sum() / max(len(y), 1)
    return p_sorted, precision, TP, FP, base_rate


def plot_precision_row(preds_by_fold, labels_by_fold, title, out_path):
    n = len(preds_by_fold)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for k, (p, y) in enumerate(zip(preds_by_fold, labels_by_fold)):
        ax = axes[0][k]
        thr, prec, TP, FP, base = _precision_curve(p, y)
        if len(thr):
            ax.plot(thr, prec, color='steelblue', linewidth=1.3)
            ax.axhline(base, color='gray', linestyle='--', label=f'base={base:.4f}')
            ax.invert_xaxis()
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        ax.set_title(f'fold {k}')
        ax.set_xlabel('threshold')
        if k == 0:
            ax.set_ylabel('precision')
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_model3_real_over_stocks(fold_results, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for fr in fold_results:
        tickers_sorted = fr['m3_stocks']  # list of (ticker, real)
        ys = np.array([r for _, r in tickers_sorted], dtype=float)
        ys = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(ys)
        if n == 0:
            continue
        cum_avg = np.cumsum(ys) / n
        ax.plot(range(n), cum_avg, linewidth=1.0, alpha=0.9,
                label=f"fold {fr['k']} (n={n}, avg={cum_avg[-1]:.4f})")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6)
    ax.set_xlabel('ticker rank (descending by avg volume)')
    ax.set_ylabel("cumsum(real) / total tickers")
    ax.set_title('Model 3 — cumulative real by ticker (val, sorted by volume)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_model3_real_over_days(fold_results, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for fr in fold_results:
        series = fr['m3_days'].sort_index()
        vals = np.nan_to_num(series.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(vals)
        if n == 0:
            continue
        cum_avg = np.cumsum(vals) / n
        ax.plot(series.index.astype(str), cum_avg, linewidth=1.0,
                label=f"fold {fr['k']} (n={n}, avg={cum_avg[-1]:.4f})")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6)
    ax.set_xlabel('date')
    ax.set_ylabel("cumsum(real) / total days")
    ax.set_title('Model 3 — cumulative real by day (val)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"    saved {out_path}")


# ──────────────────────────────────────────────────────────────
# Step 6 — event logger
# ──────────────────────────────────────────────────────────────
def build_events_for_fold(dl, fr, O, p1_all, p2_all_c, p3_all_c):
    """Emit one DataFrame per long-entry event."""
    val_mask = fr['val_mask']
    train_mask = fr['train_mask']
    p1_tr = p1_all[train_mask]
    p1_v = p1_all[val_mask]

    cutoff = float(np.quantile(p1_tr, LONG_CUT_Q))
    long_raw = np.where(p1_v > cutoff, 1.0, 0.0)

    O_val = O.loc[val_mask].reset_index(drop=True)
    long_signal = tb.causal_long_short(long_raw, O_val, cooldown_hours=COOLDOWN_HOURS)

    # Build per-fold lookup of predictions by (ticker, ts)
    ts_all = ts_from_date_hour(O['date'].values, O['hoursSinceMidnight'].values)
    pred_lookup = pd.DataFrame({
        'ticker': O['ticker'].values,
        'ts': ts_all,
        'm1': p1_all,
        'm2': p2_all_c,
        'm3': p3_all_c,
    })
    pred_lookup = pred_lookup.drop_duplicates(['ticker', 'ts']).set_index(['ticker', 'ts'])

    # cache 1h raw per ticker (across all spot markets for spot_cvd)
    ticker_cache = {}

    EMA_H = 480  # 20 days at 1h cadence

    def get_ticker_data(ticker):
        if ticker in ticker_cache:
            return ticker_cache[ticker]
        try:
            perp = dl.get(ticker, 'binance', 'futures', barFreqInHours=1)
        except (FileNotFoundError, KeyError):
            ticker_cache[ticker] = None
            return None
        # aggregated spot cvd
        spot_cvd = None
        for sm in dl.marketForTicker(ticker, 'spot'):
            try:
                sr = dl.get(ticker, sm, 'spot', barFreqInHours=1)
            except (FileNotFoundError, KeyError):
                continue
            aligned = sr['cvd_usdt'].reindex(perp.index).fillna(0.0)
            spot_cvd = aligned if spot_cvd is None else spot_cvd + aligned
        if spot_cvd is None:
            spot_cvd = pd.Series(np.nan, index=perp.index)

        # 20-day EMAs computed on the FULL 1h ticker series (proper warmup)
        oi_over_price = perp['open_interest'] / perp['close'].replace(0, np.nan)
        abs_future_cvd_ema = tb._ema(perp['cvd_usdt'].abs(), EMA_H)
        abs_spot_cvd_ema   = tb._ema(spot_cvd.abs(), EMA_H)
        oi_ema             = tb._ema(perp['open_interest'], EMA_H)
        oi_over_price_ema  = tb._ema(oi_over_price, EMA_H)

        ticker_cache[ticker] = {
            'perp': perp,
            'spot_cvd': spot_cvd,
            'abs_future_cvd_ema': abs_future_cvd_ema,
            'abs_spot_cvd_ema':   abs_spot_cvd_ema,
            'oi_ema':             oi_ema,
            'oi_over_price_ema':  oi_over_price_ema,
        }
        return ticker_cache[ticker]

    events = []
    entry_rows = np.where(long_signal == 1)[0]
    print(f"    fold {fr['k']}: cutoff={cutoff:.4f}, triggered {len(entry_rows)} long events")

    for i in tqdm(entry_rows, desc=f"fold {fr['k']} events", leave=False):
        ticker = str(O_val['ticker'].iloc[i])
        entry_date = int(O_val['date'].iloc[i])
        entry_hour = int(O_val['hoursSinceMidnight'].iloc[i])
        entry_ts = pd.Timestamp(f"{entry_date}", tz='UTC') + pd.Timedelta(hours=entry_hour)

        td = get_ticker_data(ticker)
        if td is None:
            continue
        perp = td['perp']
        spot_cvd = td['spot_cvd']

        grid = pd.date_range(start=entry_ts, periods=EVENT_FWD_HOURS, freq='1h')
        price = perp['close'].reindex(grid)
        funding_rate = perp['funding_rate'].reindex(grid)
        funding_ih = perp['funding_interval_hours'].reindex(grid)
        oi = perp['open_interest'].reindex(grid)
        future_cvd = perp['cvd_usdt'].reindex(grid)
        spot_cvd_v = spot_cvd.reindex(grid)
        abs_future_cvd_ema480 = td['abs_future_cvd_ema'].reindex(grid)
        abs_spot_cvd_ema480   = td['abs_spot_cvd_ema'].reindex(grid)
        oi_ema480             = td['oi_ema'].reindex(grid)
        oi_over_price_ema480  = td['oi_over_price_ema'].reindex(grid)

        # model preds aligned on grid (may have holes → NaN)
        try:
            sub = pred_lookup.loc[ticker]
            m1 = sub['m1'].reindex(grid)
            m2 = sub['m2'].reindex(grid)
            m3 = sub['m3'].reindex(grid)
        except KeyError:
            nan_arr = np.full(len(grid), np.nan)
            m1 = pd.Series(nan_arr, index=grid)
            m2 = pd.Series(nan_arr, index=grid)
            m3 = pd.Series(nan_arr, index=grid)

        oi_over_price = oi.values / np.where(price.values == 0, np.nan, price.values)

        ev = pd.DataFrame({
            'fold': fr['k'],
            'ticker': ticker,
            'entry_time': entry_ts,
            'hour_offset': np.arange(EVENT_FWD_HOURS),
            'ts': grid,
            'price': price.values,
            'funding_rate': funding_rate.values,
            'funding_interval_hours': funding_ih.values,
            'open_interest': oi.values,
            'oi_over_price': oi_over_price,
            'future_cvd': future_cvd.values,
            'spot_cvd': spot_cvd_v.values,
            'abs_future_cvd_ema480': abs_future_cvd_ema480.values,
            'abs_spot_cvd_ema480':   abs_spot_cvd_ema480.values,
            'oi_ema480':             oi_ema480.values,
            'oi_over_price_ema480':  oi_over_price_ema480.values,
            'model1_pred': m1.values,
            'model2_pred': m2.values,
            'model3_pred': m3.values,
        })
        events.append(ev)

    return events, cutoff


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("5-fold CV: 3 models on agg_futures_data features")
    print("=" * 70)

    # Step 1+2 — dataset
    X, Y, O = build_or_load_dataset()
    print(f"\n[dataset] X={X.shape}, rows_with_pump_label={int((~Y['pump_label'].isna()).sum())}")

    # DataLoader is needed for Step 6 regardless of cache
    dl = tb.DataLoader(DATA_FOLDER)

    # Step 3 — folds
    print("\n[step 3] chronological 5-fold split")
    folds = build_folds(O, N_FOLDS)

    # Steps 4-6 per fold
    fold_results = []
    all_events = []
    cutoffs = []

    for fr in folds:
        k = fr['k']
        print(f"\n{'-'*60}\n[fold {k}] training 3 models\n{'-'*60}")

        X_tr = X[fr['train_mask']]
        Y_tr = Y.loc[fr['train_mask']].reset_index(drop=True)
        O_tr = O.loc[fr['train_mask']].reset_index(drop=True)
        X_v  = X[fr['val_mask']]
        Y_v  = Y.loc[fr['val_mask']].reset_index(drop=True)
        O_v  = O.loc[fr['val_mask']].reset_index(drop=True)

        # normalizer
        norm = tb.FeatureStater(group_mode='all', quantiles=(0.005, 0.995))
        norm.fit(X_tr, O_tr)
        X_tr_n = norm.transform(X_tr, O_tr, clip=True)
        X_v_n  = norm.transform(X_v,  O_v,  clip=True)
        X_all_n = norm.transform(X, O, clip=True)

        # Step 4 — fit
        m1, m2, m3, p1_all, p2_all, p3_all = fit_three_models(
            X_tr_n, X_v_n, X_all_n, Y_tr, Y_v
        )

        # center model 2 and 3 by train median
        med2 = float(np.median(p2_all[fr['train_mask']]))
        med3 = float(np.median(p3_all[fr['train_mask']]))
        p2_all_c = p2_all - med2
        p3_all_c = p3_all - med3

        # Step 5 per-fold summaries for plotting
        p1_v = p1_all[fr['val_mask']]
        p2_v_c = p2_all_c[fr['val_mask']]
        p3_v_c = p3_all_c[fr['val_mask']]

        # Model-3 diagnostics: 'real' by ticker and by date on validation
        ret_v = Y_v['ret_24h'].values
        ok_v  = ~np.isnan(ret_v)
        # by ticker
        tickers_v = O_v['ticker'].values[ok_v]
        vols_v    = O_v['volume_usdt'].values[ok_v]
        p3v_c_ok = p3_v_c[ok_v]
        ret_v_ok = ret_v[ok_v]
        tmp_df = pd.DataFrame({
            'ticker': tickers_v,
            'vol': vols_v,
            'p': p3v_c_ok,
            'y': ret_v_ok,
        })
        stock_stats = []
        for t, g in tmp_df.groupby('ticker'):
            if len(g) < 2:
                continue
            _, _, real, _ = tb.stats(g['p'].values, g['y'].values)
            stock_stats.append((t, g['vol'].mean(), real))
        stock_stats.sort(key=lambda r: -r[1])  # descending volume
        m3_stocks = [(t, r) for t, _, r in stock_stats]

        # by date
        tmp_df2 = pd.DataFrame({
            'date': O_v['date'].values[ok_v],
            'p': p3v_c_ok,
            'y': ret_v_ok,
        })
        day_rows = []
        for d, g in tmp_df2.groupby('date'):
            if len(g) < 2:
                continue
            _, _, real, _ = tb.stats(g['p'].values, g['y'].values)
            day_rows.append((d, real))
        day_rows.sort(key=lambda r: r[0])
        m3_days = pd.Series([r for _, r in day_rows], index=[d for d, _ in day_rows])

        fold_results.append({
            'k': k,
            'p1_v': p1_v,
            'p2_v_c': p2_v_c,
            'p3_v_c': p3_v_c,
            'pump_v': Y_v['pump_label'].values,
            'sign_v': (np.sign(ret_v) + 1.0) / 2.0,
            'ret_v':  ret_v,
            'm3_stocks': m3_stocks,
            'm3_days':   m3_days,
            'med2': med2,
            'med3': med3,
        })

        # Step 6 events
        print(f"[fold {k}] building events (cutoff from p1_train @ {LONG_CUT_Q*100:.0f}%)")
        evts, cutoff = build_events_for_fold(dl, fr, O, p1_all, p2_all_c, p3_all_c)
        all_events.extend(evts)
        cutoffs.append(cutoff)

        # save models (append-safe)
        if k == 0:
            models_dump = {'folds': []}
        else:
            models_dump = tb.loadPickle(OUT_MODELS) if os.path.exists(OUT_MODELS) else {'folds': []}
        models_dump['folds'].append({
            'k': k,
            'm1': m1, 'm2': m2, 'm3': m3,
            'med2': med2, 'med3': med3,
            'cutoff_m1': cutoff,
        })
        tb.savePickle(models_dump, OUT_MODELS)

    # ── Step 5 plots after all folds ──
    print("\n[step 5] diagnostic plots")
    plot_precision_row(
        [fr['p1_v'] for fr in fold_results],
        [fr['pump_v'] for fr in fold_results],
        'Model 1 (pump_label) — precision on val by fold',
        os.path.join(RESULTS_DIR, 'cv_precision_model1.png'),
    )
    plot_precision_row(
        [fr['p2_v_c'] for fr in fold_results],
        [fr['sign_v'] for fr in fold_results],
        'Model 2 (sign ret_24h) — precision on val by fold (centered pred)',
        os.path.join(RESULTS_DIR, 'cv_precision_model2.png'),
    )
    plot_model3_real_over_stocks(
        fold_results,
        os.path.join(RESULTS_DIR, 'cv_model3_real_over_stocks.png'),
    )
    plot_model3_real_over_days(
        fold_results,
        os.path.join(RESULTS_DIR, 'cv_model3_real_over_days.png'),
    )

    # save events
    tb.savePickle(
        {'events': all_events, 'cutoffs': cutoffs, 'n_folds': N_FOLDS},
        OUT_EVENTS,
    )
    print(f"\n[done] events saved to {OUT_EVENTS} (n_events={len(all_events)})")
    print(f"       models saved to {OUT_MODELS}")


if __name__ == '__main__':
    main()
