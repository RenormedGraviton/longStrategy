import os
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from tqdm.notebook import tqdm


class DataLoader:
    """Loader for a folder of parquet files named {symbol}_{exchange}_{instrument}.parquet."""

    def __init__(self, data_folder):
        self.data_folder = os.path.abspath(data_folder)
        files = glob.glob(os.path.join(self.data_folder, '*.parquet'))
        self._files = []
        exchanges = set()
        instruments = set()
        symbols = set()
        for f in files:
            parts = os.path.basename(f).replace('.parquet', '').split('_')
            sym, exch, inst = parts[0], parts[1], parts[2]
            self._files.append((sym, exch, inst))
            symbols.add(sym)
            exchanges.add(exch)
            instruments.add(inst)
        self.symbols = sorted(symbols)
        self.exchanges = sorted(exchanges)
        self.instruments = sorted(instruments)
        self.columns = pq.read_schema(files[0]).names
        ts = pd.to_datetime(pq.read_table(files[0], columns=['ts']).column('ts').to_pandas(), utc=True)
        dates = sorted(ts.dt.strftime('%Y%m%d').unique())
        self.dates = [int(d) for d in dates]

    def availSymbols(self, exchange, instrument):
        """Return sorted list of symbols available for a given exchange and instrument."""
        return sorted({sym for sym, exch, inst in self._files
                       if exch == exchange and inst == instrument})

    def get(self, symbol, exchange, instrument, startDate=None, endDate=None, barFreqInHours=1):
        """Load bars for a symbol, aggregate to a non-overlapping frequency.

        Equivalent to get_sophi with sampleFreqInHours == barFreqInHours.
        """
        return self.get_sophi(symbol, exchange, instrument, startDate, endDate,
                              barFreqInHours=barFreqInHours, sampleFreqInHours=barFreqInHours)

    def _load_raw(self, symbol, exchange, instrument, startDate=None, endDate=None):
        """Load raw 1h bars, filtered by date range."""
        fname = f"{symbol}_{exchange}_{instrument}.parquet"
        path = os.path.join(self.data_folder, fname)
        df = pd.read_parquet(path)
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        df = df.sort_values('ts').set_index('ts')
        if startDate is not None:
            df = df[df.index >= pd.Timestamp(str(startDate), tz='UTC')]
        if endDate is not None:
            df = df[df.index <= pd.Timestamp(str(endDate) + ' 23:59:59', tz='UTC')]
        # open_interest is already in dollar terms in data2
        if 'funding_interval_hours' not in df.columns:
            df['funding_interval_hours'] = float('nan')
        return df

    @staticmethod
    def _agg_window(df):
        """Aggregate a chunk of 1h rows into one bar."""
        return pd.Series({
            'close': df['close'].iloc[-1],
            'volume_usdt': df['volume_usdt'].sum(),
            'cvd_usdt': df['cvd_usdt'].sum(),
            'open_interest_chg': df['open_interest'].iloc[-1] - df['open_interest'].iloc[0],
            'open_interest': df['open_interest'].iloc[-1],
            'open_interest_avg': df['open_interest'].mean(),
            'funding_rate_chg': df['funding_rate'].iloc[-1] - df['funding_rate'].iloc[0],
            'funding_rate': df['funding_rate'].iloc[-1],
            'funding_rate_avg': df['funding_rate'].mean(),
            'funding_interval_hours': df['funding_interval_hours'].iloc[-1],
        })

    def get_sophi(self, symbol, exchange, instrument, startDate=None, endDate=None,
                  barFreqInHours=1, sampleFreqInHours=None):
        """Like get() but with sliding-window aggregation.

        Parameters
        ----------
        barFreqInHours : int – window size in hours (how many 1h bars to aggregate)
        sampleFreqInHours : int or None – step size in hours (how often to output a bar).
            Defaults to barFreqInHours (no overlap). When < barFreqInHours, bars overlap.
        """
        if sampleFreqInHours is None:
            sampleFreqInHours = barFreqInHours

        raw = self._load_raw(symbol, exchange, instrument, startDate, endDate)
        window = int(barFreqInHours)
        step = int(sampleFreqInHours)

        # sample points: every `step` hours, starting from the first index where
        # a full window is available
        sample_idx = raw.index[window - 1::step]
        rows = []
        for ts in sample_idx:
            chunk = raw.loc[(ts - pd.Timedelta(hours=window - 1)):ts]
            if len(chunk) < window:
                continue
            rows.append(self._agg_window(chunk))

        agg = pd.DataFrame(rows, index=sample_idx)
        agg.index.name = 'ts'
        agg['date'] = agg.index.strftime('%Y%m%d').astype(int)
        agg['hoursSinceMidnight'] = agg.index.hour
        return agg

    def __repr__(self):
        first, last = self.symbols[0], self.symbols[-1]
        n = len(self.symbols)
        mkts = ', '.join(self.exchanges)
        insts = ', '.join(self.instruments)
        d0, d1, nd = self.dates[0], self.dates[-1], len(self.dates)
        return f"DataLoader(symbols={first} .. {last} ({n} symbols), exchanges=({mkts}), instruments=({insts}), dates={d0} .. {d1} ({nd} days))"


def agg_daniel_data(dl, tickers, exchanges, instruments,
                    startDate=None, endDate=None,
                    barFreq=4, sampleFreq=None):
    """Build derived feature DataFrames for all ticker/exchange/instrument combos.

    Parameters
    ----------
    dl : DataLoader
    tickers, exchanges, instruments : str or list of str
    startDate, endDate : str/int or None
    barFreq : int – window size in hours
    sampleFreq : int or None – step size in hours (defaults to barFreq)

    Returns
    -------
    pd.DataFrame with columns:
        ticker, exchange, instrument,
        dprice, dvolume_usdt, volume_usdt, dcvd_usdt, cvd_usdt, doi, oi,
        date, hoursSinceMidnight
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(exchanges, str):
        exchanges = [exchanges]
    if isinstance(instruments, str):
        instruments = [instruments]
    if sampleFreq is None:
        sampleFreq = barFreq

    shift = barFreq // sampleFreq  # rows back for barFreq-wide diffs

    combos = [(t, e, i) for t in tickers for e in exchanges for i in instruments]
    parts = []
    for ticker, exch, inst in tqdm(combos, desc='Loading'):
        try:
            raw = dl.get_sophi(ticker, exch, inst, startDate, endDate,
                               barFreqInHours=barFreq, sampleFreqInHours=sampleFreq)
        except (FileNotFoundError, KeyError):
            continue

        out = pd.DataFrame(index=raw.index)
        out.index.name = 'ts'
        out['ticker'] = ticker
        out['exchange'] = exch
        out['instrument'] = inst
        out['price'] = raw['close']
        out['dprice'] = raw['close'] - raw['close'].shift(shift)
        out['dprice_ratio'] = out['dprice'] / raw['close'].shift(shift)
        out['dvolume_usdt'] = raw['volume_usdt'] - raw['volume_usdt'].shift(shift)
        out['dvolume_usdt_ratio'] = out['dvolume_usdt'] / (raw['volume_usdt'] + raw['volume_usdt'].shift(shift))
        out['volume_usdt'] = raw['volume_usdt']
        out['dcvd_usdt'] = raw['cvd_usdt'] - raw['cvd_usdt'].shift(shift)
        out['cvd_usdt'] = raw['cvd_usdt']
        out['doi'] = raw['open_interest'] - raw['open_interest'].shift(shift)
        out['doi_ratio'] = out['doi'] / raw['open_interest'].shift(shift)
        out['oi'] = raw['open_interest']
        out['funding_rate'] = raw['funding_rate']
        out['funding_rate_avg'] = raw['funding_rate_avg']
        out['funding_interval_hours'] = raw['funding_interval_hours']
        out['date'] = raw['date']
        out['hoursSinceMidnight'] = raw['hoursSinceMidnight']
        parts.append(out)

    return pd.concat(parts) if parts else pd.DataFrame()


def makeReturns(dl, df, horizons=1, adjustFundingRate=False):
    """Compute forward returns matching the timestamps in df.

    Parameters
    ----------
    dl : DataLoader
    df : pd.DataFrame – output of agg_daniel_data (must have ticker, exchange,
         instrument columns and a ts index)
    horizons : int or list of int – forward return horizons in hours
    adjustFundingRate : bool – if True, subtract cumulative per-hour funding cost:
        ret_h = (price[t+h]-price[t])/price[t] - sum(funding_rate/funding_interval_hours)[t+1:t+h+1]

    Returns
    -------
    pd.DataFrame indexed by ts with columns:
        ticker, exchange, instrument, price, ret_{h}h for each horizon,
        date, hoursSinceMidnight
    """
    if isinstance(horizons, int):
        horizons = [horizons]

    groups = df.groupby(['ticker', 'exchange', 'instrument'])
    parts = []
    for (ticker, exch, inst), grp in tqdm(groups, desc='Returns'):
        try:
            raw = dl.get_sophi(ticker, exch, inst,
                               barFreqInHours=1, sampleFreqInHours=1)
        except (FileNotFoundError, KeyError):
            continue

        price = raw['close']
        if adjustFundingRate:
            funding_per_hour = raw['funding_rate'] / raw['funding_interval_hours']
        else:
            funding_per_hour = None
        # compute returns at all 1h timestamps
        all_ret = pd.DataFrame(index=price.index)
        all_ret.index.name = 'ts'
        all_ret['price'] = price.values
        for h in horizons:
            future_price = price.shift(-h).values
            ret = (future_price - price.values) / price.values
            if adjustFundingRate:
                # sum of (funding_rate/funding_interval_hours)[t+1] + ... + [t+h]
                cum_cost = funding_per_hour.rolling(h, min_periods=h).sum().shift(-h).values
                ret = ret - cum_cost
            all_ret[f'ret_{h}h'] = ret

        # keep only timestamps present in df for this group
        matched = all_ret.reindex(grp.index)
        matched['ticker'] = ticker
        matched['exchange'] = exch
        matched['instrument'] = inst
        matched['date'] = grp['date']
        matched['hoursSinceMidnight'] = grp['hoursSinceMidnight']
        parts.append(matched)

    return pd.concat(parts) if parts else pd.DataFrame()


def makeXY(df1, returns, features, appender=None):
    """Match df1 (features) with returns and split into X, Y, O.

    Parameters
    ----------
    df1 : pd.DataFrame – output of agg_daniel_data
    returns : pd.DataFrame – output of makeReturns
    features : list of str – column names from df1 to use as X

    Returns
    -------
    X : pd.DataFrame – feature columns from matched rows
    Y : pd.DataFrame – ret_*h columns from matched rows
    O : pd.DataFrame – remaining columns from matched returns (price, date, etc.)
    """
    keys = ['date', 'hoursSinceMidnight', 'ticker', 'exchange', 'instrument']
    merged = df1.merge(returns, on=keys, suffixes=('', '_ret'))

    # drop rows with any NaN in features or return columns
    ret_cols = [c for c in returns.columns if c.startswith('ret_')]
    check_cols = features + ret_cols
    merged[check_cols] = merged[check_cols].replace([float('inf'), float('-inf')], float('nan'))
    merged = merged.dropna(subset=check_cols)

    X = merged[features]
    Y = merged[ret_cols]
    o_cols = keys + [c for c in returns.columns if c not in ret_cols and c not in keys]
    O = merged[[c if c in merged.columns else c + '_ret' for c in o_cols]]
    O.columns = o_cols

    if appender is not None:
        if isinstance(appender, str):
            appender = [appender]
        for col in appender:
            O[col] = merged[col].values

    return X, Y, O


class FeatureStater:
    """Compute and apply per-group statistics for feature clipping and normalization."""

    def __init__(self, quantiles=(0.01, 0.99), group_mode='ticker_agg'):
        """
        Parameters
        ----------
        quantiles : tuple of 2 floats – lower and upper quantile for clipping
        group_mode : str – 'ticker_agg', 'tickerwise', or 'all'
        """
        assert group_mode in ('ticker_agg', 'tickerwise', 'all')
        self.quantiles = quantiles
        self.group_mode = group_mode
        self.stats_ = {}  # keyed by group key -> {feature -> {stat_name: value}}

    def _group_key_cols(self):
        if self.group_mode == 'ticker_agg':
            return ['ticker']
        elif self.group_mode == 'tickerwise':
            return ['ticker', 'exchange', 'instrument']
        else:
            return None

    @staticmethod
    def _compute_stats(sub, quantiles):
        """Compute stats dict for each feature column."""
        stats = {}
        for col in sub.columns:
            s = sub[col].dropna()
            stats[col] = {
                'mean': s.mean(),
                'median': s.median(),
                'std': s.std(),
                'min': s.min(),
                'max': s.max(),
                'q_lo': s.quantile(quantiles[0]),
                'q_hi': s.quantile(quantiles[1]),
            }
        return stats

    def fit(self, X, O):
        """Compute per-group statistics from X, using O for group columns.

        Parameters
        ----------
        X : pd.DataFrame – feature columns
        O : pd.DataFrame – must contain ticker (and exchange, instrument for tickerwise)
        """
        self.features_ = list(X.columns)
        combined = X.copy()
        combined['ticker'] = O['ticker'].values if 'ticker' in O.columns else None
        if 'exchange' in O.columns:
            combined['exchange'] = O['exchange'].values
        if 'instrument' in O.columns:
            combined['instrument'] = O['instrument'].values

        key_cols = self._group_key_cols()
        if key_cols is None:
            # group_mode == 'all'
            self.stats_['__all__'] = self._compute_stats(combined[self.features_], self.quantiles)
        else:
            for grp_key, grp in combined.groupby(key_cols):
                if isinstance(grp_key, str):
                    grp_key = (grp_key,)
                self.stats_[grp_key] = self._compute_stats(grp[self.features_], self.quantiles)
        return self

    def _resolve_key(self, row_ticker, row_exchange=None, row_instrument=None):
        if self.group_mode == 'all':
            return '__all__'
        elif self.group_mode == 'ticker_agg':
            return (row_ticker,)
        else:
            return (row_ticker, row_exchange, row_instrument)

    def transform(self, X, O, clip=True, centerMode='no', normMode='no'):
        """Clip, center, and/or normalize X using fitted statistics.

        Parameters
        ----------
        X : pd.DataFrame
        O : pd.DataFrame – must contain group columns matching group_mode
        clip : bool – clip to fitted quantile range
        centerMode : 'no', 'mean', 'median'
        normMode : 'no', 'std', 'minMax'

        Returns
        -------
        pd.DataFrame – transformed X (same shape/index)
        """
        out = X.copy()
        key_cols = self._group_key_cols()

        if key_cols is None:
            # group_mode == 'all'
            st = self.stats_['__all__']
            for col in self.features_:
                if clip:
                    out[col] = out[col].clip(lower=st[col]['q_lo'], upper=st[col]['q_hi'])
                if centerMode == 'mean':
                    out[col] = out[col] - st[col]['mean']
                elif centerMode == 'median':
                    out[col] = out[col] - st[col]['median']
                if normMode == 'std':
                    out[col] = out[col] / st[col]['std']
                elif normMode == 'minMax':
                    out[col] = out[col] / (st[col]['max'] - st[col]['min'])
        else:
            # build group keys for each row
            if self.group_mode == 'ticker_agg':
                row_keys = list(zip(O['ticker'].values))
            else:
                row_keys = list(zip(O['ticker'].values, O['exchange'].values, O['instrument'].values))

            # validate all keys exist
            unique_keys = set(row_keys)
            missing = unique_keys - set(self.stats_.keys())
            if missing:
                raise KeyError(f"Unseen groups in transform: {missing}")

            # process per group
            combined = out.copy()
            if self.group_mode == 'ticker_agg':
                combined['__grp__'] = O['ticker'].values
                grp_cols = ['__grp__']
            else:
                combined['__t__'] = O['ticker'].values
                combined['__e__'] = O['exchange'].values
                combined['__i__'] = O['instrument'].values
                grp_cols = ['__t__', '__e__', '__i__']

            for grp_key, idx in combined.groupby(grp_cols).groups.items():
                if isinstance(grp_key, str):
                    grp_key = (grp_key,)
                st = self.stats_[grp_key]
                for col in self.features_:
                    vals = out.loc[idx, col]
                    if clip:
                        vals = vals.clip(lower=st[col]['q_lo'], upper=st[col]['q_hi'])
                    if centerMode == 'mean':
                        vals = vals - st[col]['mean']
                    elif centerMode == 'median':
                        vals = vals - st[col]['median']
                    if normMode == 'std':
                        vals = vals / st[col]['std']
                    elif normMode == 'minMax':
                        vals = vals / (st[col]['max'] - st[col]['min'])
                    out.loc[idx, col] = vals

        return out


def subplots(arrs, ncols=3, figsize=None):
    """Generator that yields (idx, value) while setting the active subplot.

    Usage:
        for idx, val in subplots(values):
            plt.plot(func(val), label=idx)

    Parameters
    ----------
    arrs : list, dict, or iterable – if dict, yields (key, value); if list, yields (i, value)
    ncols : int – number of columns in the grid
    figsize : tuple or None – defaults to (4*ncols, 3.5*nrows)
    """
    import matplotlib.pyplot as plt
    import math

    if isinstance(arrs, dict):
        items = list(arrs.items())
    else:
        items = list(enumerate(arrs))

    n = len(items)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, (key, val) in enumerate(items):
        r, c = divmod(i, ncols)
        plt.sca(axes[r][c])
        plt.title(str(key))
        yield key, val

    # hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()


def fitLgb(X, Y, params, X_val=None, Y_val=None, num_boost_round=1000, log_period=0):
    """Fit a LightGBM model and record per-iteration loss.

    Parameters
    ----------
    X : array-like (n, p) – training features
    Y : array-like (n,) – training target
    params : dict – LightGBM parameters. Keys 'n_estimators' and
        'early_stopping_rounds' are extracted and handled separately.
    X_val : array-like or None – validation features (defaults to X)
    Y_val : array-like or None – validation target (defaults to Y)
    num_boost_round : int – max boosting iterations (overridden by n_estimators in params)

    Returns
    -------
    model : lgb.Booster – fitted model
    loss : dict – {dataset_name: {metric_name: [loss_per_iteration]}}
    """
    params = params.copy()
    # extract sklearn-style keys that lgb.train doesn't accept
    num_boost_round = params.pop('n_estimators', num_boost_round)
    early_stopping = params.pop('early_stopping_rounds', None)
    params['verbose'] = -1

    has_val = X_val is not None and Y_val is not None
    dtrain = lgb.Dataset(X, label=Y)
    if has_val:
        dval = lgb.Dataset(X_val, label=Y_val, reference=dtrain)
    else:
        dval = lgb.Dataset(X, label=Y, reference=dtrain)

    loss = {}

    def _print_eval(period):
        def _callback(env):
            if period > 0 and (env.iteration + 1) % period == 0:
                parts = []
                for ds, metric, val, _ in env.evaluation_result_list:
                    parts.append(f"{ds}'s {metric}: {val:.6f}")
                print(f"[{env.iteration + 1}]\t" + "\t".join(parts))
        return _callback

    callbacks = [lgb.record_evaluation(loss), _print_eval(log_period)]
    # only use early stopping when a real validation set is provided
    if early_stopping and has_val:
        callbacks.append(lgb.early_stopping(early_stopping))

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )
    return model, loss


def stats(x, y):
    """Compute signal statistics.

    Parameters
    ----------
    x, y : array-like of shape (n,)

    Returns
    -------
    (max, exp, real, counts) : tuple
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mx = np.sqrt(np.sum(y * y) / n)
    exp = np.sqrt(np.sum(x * x) / n)
    real = np.sum(x * y) / np.sqrt(np.sum(x * x)*n)
    return mx, exp, real, n


def evaluateHorizonsPred(ypred, y, o, mode='volumeWeighted'):
    """Evaluate a single prediction against multiple return horizons.

    Parameters
    ----------
    ypred : array-like (n,)
    y : pd.DataFrame – columns in 'ret_{h}h' format
    o : pd.DataFrame – must contain 'ticker'; 'volume_usdt' required for volumeWeighted
    mode : 'all', 'equalWeight', or 'volumeWeighted'

    Returns
    -------
    np.ndarray of shape (n_horizons, 4) – (max, exp, real, counts) per horizon
    """
    ret_cols = sorted(y.columns, key=lambda c: int(c.split('_')[1].replace('h', '')))
    return np.array([evaluateSingleHorizonPred(ypred, y, o, col, mode=mode, agg=True)
                     for col in ret_cols])


def evaluateSingleHorizonPred(pred, Y, O, horizon, mode='volumeWeighted', agg=True):
    """Evaluate a prediction against a single return horizon.

    Parameters
    ----------
    pred : array-like (n,)
    Y : pd.DataFrame – must contain `horizon` column
    O : pd.DataFrame – must contain 'ticker'; 'volume_usdt' required for volumeWeighted
    horizon : str – column name in Y, e.g. 'ret_4h'
    mode : 'all', 'equalWeight', or 'volumeWeighted' (ignored when agg=False)
    agg : bool – if False, return per-ticker stats (n_tickers, 4)

    Returns
    -------
    np.ndarray – shape (4,) if agg=True, shape (n_tickers, 4) if agg=False
        columns are (max, exp, real, counts).
        When agg=False, tickers are sorted alphabetically.
    """
    assert horizon in Y.columns, f"'{horizon}' not in Y.columns: {list(Y.columns)}"
    pred = np.asarray(pred, dtype=float)
    yvals = Y[horizon].values

    tickers = O['ticker'].values
    unique_tickers = np.sort(np.unique(tickers))
    ticker_masks = {t: (tickers == t) for t in unique_tickers}

    if not agg:
        vol = O['volume_usdt'].values if 'volume_usdt' in O.columns else None
        group_stats = []
        for t in unique_tickers:
            m = ticker_masks[t]
            mx, exp, real, n = stats(pred[m], yvals[m])
            avg_vol = vol[m].mean() if vol is not None else 0.0
            group_stats.append((mx, exp, real, n, avg_vol))
        return unique_tickers, np.array(group_stats)

    if mode == 'all':
        return np.array(stats(pred, yvals))

    group_stats = []
    for t in unique_tickers:
        m = ticker_masks[t]
        group_stats.append(stats(pred[m], yvals[m]))
    group_stats = np.array(group_stats)

    if mode == 'equalWeight':
        return group_stats.mean(axis=0)

    elif mode == 'volumeWeighted':
        assert 'volume_usdt' in O.columns, "O must contain 'volume_usdt' for volumeWeighted mode"
        vol = O['volume_usdt'].values
        weights = np.array([np.sqrt(vol[ticker_masks[t]].mean()) for t in unique_tickers])
        weights = weights / weights.sum()
        return (weights[:, None] * group_stats).sum(axis=0)


def MainEvaluateHorizons(pred, Y, O, cutoff_values, eval_horizon='ret_24h',
                         statCol=2, cutoff_names=None):
    """Plot evaluation across horizons for different prediction cutoffs.

    4 rows:
      Row 1: mode='all', one line per cutoff
      Row 2: mode='equalWeight', one line per cutoff
      Row 3: mode='volumeWeighted', one line per cutoff
      Row 4: per-ticker cumsum of stats sorted by volume for eval_horizon

    Parameters
    ----------
    pred : array-like (n,)
    Y : pd.DataFrame – ret columns
    O : pd.DataFrame – must contain 'ticker', 'volume_usdt'
    cutoff_values : list of float – abs(pred) > cutoff thresholds
    eval_horizon : str – horizon column for row 4
    statCol : int – which stat to plot in rows 1-3 (0=max, 1=exp, 2=real, 3=counts)
    cutoff_names : list of str or None – labels for cutoffs
    """
    import matplotlib.pyplot as plt

    pred = np.asarray(pred, dtype=float)
    if cutoff_names is None:
        cutoff_names = [f'{c:.4f}' for c in cutoff_values]

    ret_cols = sorted(Y.columns, key=lambda c: int(c.split('_')[1].replace('h', '')))
    stat_names = ['max', 'exp', 'real', 'counts']
    modes = ['all', 'equalWeight', 'volumeWeighted']

    fig, axes = plt.subplots(4, 1, figsize=(10, 16))

    # Rows 1-3: per-mode, one line per cutoff
    for row, mode in enumerate(modes):
        plt.sca(axes[row])
        for ci, cutoff in enumerate(cutoff_values):
            mask = np.abs(pred) > cutoff
            if mask.sum() == 0:
                continue
            output = evaluateHorizonsPred(pred[mask], Y.loc[mask], O.loc[mask], mode=mode)
            plt.plot(output[:, statCol], label=f'{cutoff_names[ci]} (n={mask.sum()})', marker='o')
        plt.xticks(np.arange(len(ret_cols)), ret_cols, rotation=90, fontsize=7)
        plt.title(f'agg={mode} horizon plots for statCol={stat_names[statCol]}')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    # Row 4: per-ticker cumsum sorted by volume, one line per cutoff
    plt.sca(axes[3])
    for ci, cutoff in enumerate(cutoff_values):
        mask = np.abs(pred) > cutoff
        if mask.sum() == 0:
            continue
        tickers, outputs = evaluateSingleHorizonPred(
            pred[mask], Y.loc[mask], O.loc[mask], eval_horizon, agg=False)
        order = np.argsort(outputs[:, 4])
        outputs = outputs[order]
        n = outputs.shape[0]
        plt.plot(np.cumsum(outputs[:, statCol]) / n,
                 label=f'{cutoff_names[ci]} (n={n} tickers)')
    plt.xlabel('stocks -> increasing volume')
    plt.title(f'Per-ticker cumsum ({eval_horizon}) for statCol={stat_names[statCol]}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()


def evaluateTimes(pred, Y, O, cutoff_q=0.0, statCol=2, aggMode='all'):
    """Evaluate prediction across horizons, broken down by hoursSinceMidnight.

    Parameters
    ----------
    pred : array-like (n,)
    Y : pd.DataFrame – ret columns
    O : pd.DataFrame – must contain 'hoursSinceMidnight', 'ticker'
    cutoff_q : float – quantile on |pred| for filtering (0 = no filter)
    statCol : int – 0=max, 1=exp, 2=real, 3=counts
    aggMode : str – 'all', 'equalWeight', or 'volumeWeighted'
    """
    import matplotlib.pyplot as plt

    pred = np.asarray(pred, dtype=float)
    stat_names = ['max', 'exp', 'real', 'counts']
    ret_cols = sorted(Y.columns, key=lambda c: int(c.split('_')[1].replace('h', '')))

    # apply quantile cutoff
    threshold = np.quantile(np.abs(pred), cutoff_q)
    mask = np.abs(pred) > threshold
    pred = pred[mask]
    Y = Y.loc[mask]
    O = O.loc[mask]

    hours = sorted(O['hoursSinceMidnight'].unique())

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for h in hours:
        hmask = O['hoursSinceMidnight'].values == h
        if hmask.sum() == 0:
            continue
        output = evaluateHorizonsPred(pred[hmask], Y.loc[hmask], O.loc[hmask], mode=aggMode)
        ax.plot(output[:, statCol], label=f'hour={h} (n={hmask.sum()})', marker='o')

    ax.set_xticks(np.arange(len(ret_cols)))
    ax.set_xticklabels(ret_cols, rotation=90, fontsize=7)
    ax.set_title(f'agg={aggMode} by hoursSinceMidnight, statCol={stat_names[statCol]}, cutoff_q={cutoff_q}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def getDailyReturn(ypred_train, ypred_valid, y_valid, o_valid,
                   cutoff_quantiles, horizon='ret_24h', aggMode='sign'):
    """Compute daily returns per hoursSinceMidnight for each cutoff quantile.

    Parameters
    ----------
    ypred_train : array-like (n_train,) – training predictions (for computing cutoffs)
    ypred_valid : array-like (n_valid,) – validation predictions
    y_valid : pd.DataFrame – must contain `horizon` column
    o_valid : pd.DataFrame – must contain 'hoursSinceMidnight', 'date'
    cutoff_quantiles : list of float – quantiles on |ypred_train|
    horizon : str – column name in y_valid
    aggMode : str – 'sign' or 'value'
        'sign': daily = mean(y * sign(pred))
        'value': daily = sum(y * pred) / sum(|pred|)

    Returns
    -------
    dict[int, dict[float, pd.Series]] – keyed by hoursSinceMidnight, then by
        cutoff quantile. Each pd.Series is indexed by date with daily return.
    """
    ypred_train = np.asarray(ypred_train, dtype=float)
    ypred_valid = np.asarray(ypred_valid, dtype=float)
    yvals = y_valid[horizon].values
    sign_pred = np.sign(ypred_valid)
    abs_pred = np.abs(ypred_valid)

    cutoff_values = [np.quantile(np.abs(ypred_train), q) for q in cutoff_quantiles]
    hours = sorted(o_valid['hoursSinceMidnight'].unique())

    result = {}
    for h in hours:
        hmask = o_valid['hoursSinceMidnight'].values == h
        hour_dict = {}
        for q, cutoff in zip(cutoff_quantiles, cutoff_values):
            cmask = hmask & (abs_pred > cutoff)
            if cmask.sum() == 0:
                hour_dict[q] = (pd.Series(dtype=float), 0.0)
                continue
            dates = o_valid['date'].values[cmask]
            if aggMode == 'sign':
                daily_ret = pd.Series(yvals[cmask] * sign_pred[cmask], index=dates)
                counts_per_day = daily_ret.groupby(daily_ret.index).count().mean()
                daily_ret = daily_ret.groupby(daily_ret.index).mean()
            elif aggMode == 'value':
                numer = pd.Series(yvals[cmask] * ypred_valid[cmask], index=dates)
                denom = pd.Series(abs_pred[cmask], index=dates)
                counts_per_day = numer.groupby(numer.index).count().mean()
                daily_ret = numer.groupby(numer.index).sum() / denom.groupby(denom.index).sum()
            hour_dict[q] = (daily_ret, counts_per_day)
        result[h] = hour_dict

    return result


def plotDailyReturn(rets, agg=False):
    """Plot cumulative daily returns from getDailyReturn output.

    Parameters
    ----------
    rets : dict[int, dict[float, (pd.Series, float)]] – output of getDailyReturn
    agg : bool – if True, average all hours into a single plot
    """
    import matplotlib.pyplot as plt
    import math

    hours = sorted(rets.keys())

    if agg:
        # average across all hours per quantile
        quantiles = list(rets[hours[0]].keys())
        # collect all dates across all hours and quantiles
        all_dates = sorted(set().union(
            *(s.index for h in hours for s, _ in rets[h].values() if len(s) > 0)))

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for q in quantiles:
            hour_vals = []
            cpd_sum = 0.0
            for h in hours:
                series, cpd = rets[h][q]
                if len(series) == 0:
                    continue
                aligned = series.reindex(all_dates)
                vals = np.nan_to_num(aligned.values, nan=0.0, posinf=0.0, neginf=0.0)
                hour_vals.append(vals)
                cpd_sum += cpd
            if len(hour_vals) == 0:
                continue
            avg_vals = np.mean(hour_vals, axis=0)
            avg_cpd = cpd_sum / len(hour_vals)
            ax.plot(np.cumsum(avg_vals) / len(avg_vals), label=f'q={q} (n={avg_cpd:.0f}/d)')
        ax.axhline(0, color='black', linestyle='dashed')
        ax.set_title('average')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return

    n = len(hours)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             sharey=True, squeeze=False)

    for i, h in enumerate(hours):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        # collect all dates across cutoffs for this hour
        all_dates = sorted(set().union(
            *(s.index for s, _ in rets[h].values() if len(s) > 0)))
        for q, (series, cpd) in rets[h].items():
            if len(series) == 0:
                continue
            aligned = series.reindex(all_dates)
            vals = np.nan_to_num(aligned.values, nan=0.0, posinf=0.0, neginf=0.0)
            ax.plot(np.cumsum(vals) / len(vals), label=f'q={q} (n={cpd:.0f}/d)')
        ax.axhline(0, color='black', linestyle='dashed')
        ax.set_title(f'hour={h}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()


def showStockSelection(ypred_train, ypred_valid, o_valid, cutoff_quantile):
    """Show a heatmap of long/short/flat positions: stocks x days.

    Parameters
    ----------
    ypred_train : array-like – training predictions (for cutoff)
    ypred_valid : array-like – validation predictions
    o_valid : pd.DataFrame – must contain 'ticker', 'date', 'volume_usdt'
    cutoff_quantile : float – quantile on |ypred_train|
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ypred_train = np.asarray(ypred_train, dtype=float)
    ypred_valid = np.asarray(ypred_valid, dtype=float)

    cutoff = np.quantile(np.abs(ypred_train), cutoff_quantile)

    pos = np.where(ypred_valid > cutoff, 1,
                   np.where(ypred_valid < -cutoff, -1, 0))

    tickers = o_valid['ticker'].values
    dates = o_valid['date'].values

    vol_per_ticker = o_valid.groupby('ticker')['volume_usdt'].mean()

    unique_dates = sorted(np.unique(dates))
    unique_tickers = vol_per_ticker.sort_values(ascending=False).index.tolist()

    ticker_to_row = {t: i for i, t in enumerate(unique_tickers)}
    date_to_col = {d: j for j, d in enumerate(unique_dates)}

    matrix = np.zeros((len(unique_tickers), len(unique_dates)))
    for k in range(len(pos)):
        t, d = tickers[k], dates[k]
        if t in ticker_to_row and d in date_to_col:
            r, c = ticker_to_row[t], date_to_col[d]
            if pos[k] != 0:
                matrix[r, c] = pos[k]

    cmap = ListedColormap(['green', 'white', 'red'])
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('date')
    ax.set_ylabel('ticker (large vol -> top)')
    ax.set_title(f'Stock selection (cutoff_q={cutoff_quantile}, cutoff={cutoff:.6f})')

    n_dates = len(unique_dates)
    step = max(1, n_dates // 15)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels([unique_dates[j] for j in range(0, n_dates, step)],
                       rotation=90, fontsize=6)

    n_tickers = len(unique_tickers)
    tstep = max(1, n_tickers // 30)
    ax.set_yticks(range(0, n_tickers, tstep))
    ax.set_yticklabels([unique_tickers[j] for j in range(0, n_tickers, tstep)], fontsize=6)

    fig.tight_layout()
