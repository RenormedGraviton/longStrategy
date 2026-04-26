import os
import glob
import pickle
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb


def savePickle(obj, path):
    """Save an object (dict or anything picklable) to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def loadPickle(path):
    """Load an object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)
from tqdm.notebook import tqdm


def normalize_symbol(sym, exchange):
    """Normalize a symbol to a standard form.

    E.g. OKEx '0G-USDT-SWAP' or '0G-USDT' -> '0GUSDT'; Binance/Bybit/Bitget
    unchanged.
    """
    s = sym.upper()
    if "okex" in exchange.lower():
        s = s.replace("-USDT-SWAP", "USDT").replace("-USDT", "USDT")
    return s


class DataLoader:
    """Loader for a folder of parquet files named {symbol}_{exchange}_{instrument}.parquet.

    Symbol names are normalized (e.g. OKEx '0G-USDT-SWAP' -> '0GUSDT') so all
    public APIs use the normalized form. The original filename is tracked
    internally via `_file_lookup`.
    """

    def __init__(self, data_folder):
        self.data_folder = os.path.abspath(data_folder)
        files = glob.glob(os.path.join(self.data_folder, '*.parquet'))
        self._files = []
        self._file_lookup = {}  # (norm_sym, exchange, instrument) -> raw_sym
        exchanges = set()
        instruments = set()
        symbols = set()
        for f in files:
            parts = os.path.basename(f).replace('.parquet', '').split('_')
            raw_sym, exch, inst = parts[0], parts[1], parts[2]
            norm_sym = normalize_symbol(raw_sym, exch)
            self._files.append((norm_sym, exch, inst))
            self._file_lookup[(norm_sym, exch, inst)] = raw_sym
            symbols.add(norm_sym)
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

    def marketForTicker(self, ticker, instrument):
        """Return sorted list of exchanges that have the given ticker and instrument."""
        return sorted({exch for sym, exch, inst in self._files
                       if sym == ticker and inst == instrument})

    def get(self, symbol, exchange, instrument, startDate=None, endDate=None, barFreqInHours=1):
        """Load bars for a symbol, aggregate to a non-overlapping frequency.

        Equivalent to get_sophi with sampleFreqInHours == barFreqInHours.
        """
        return self.get_sophi(symbol, exchange, instrument, startDate, endDate,
                              barFreqInHours=barFreqInHours, sampleFreqInHours=barFreqInHours)

    def _load_raw(self, symbol, exchange, instrument, startDate=None, endDate=None):
        """Load raw 1h bars, filtered by date range. Accepts normalized symbol."""
        key = (symbol, exchange, instrument)
        if key not in self._file_lookup:
            raise FileNotFoundError(f"No data for {symbol} {exchange} {instrument}")
        raw_sym = self._file_lookup[key]
        fname = f"{raw_sym}_{exchange}_{instrument}.parquet"
        path = os.path.join(self.data_folder, fname)
        df = pd.read_parquet(path)
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        df = df.sort_values('ts').set_index('ts')
        df = df[~df.index.duplicated(keep='first')]
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


def _exp_sum(series, H):
    """Exponential sum: x(t) = f(t) + alpha * x(t-1), alpha = exp(-1/H), x(0)=f(0)."""
    alpha = np.exp(-1.0 / H)
    vals = series.values.astype(float)
    out = np.zeros_like(vals)
    prev = 0.0
    started = False
    for i in range(len(vals)):
        v = vals[i]
        if np.isnan(v):
            out[i] = np.nan if not started else prev
            continue
        if not started:
            out[i] = v
            prev = v
            started = True
        else:
            prev = v + alpha * prev
            out[i] = prev
    return pd.Series(out, index=series.index)


def _ema(series, H):
    """True EMA: x(t) = (1-alpha)*f(t) + alpha*x(t-1), alpha=exp(-1/H), x(0)=f(0)."""
    alpha = np.exp(-1.0 / H)
    vals = series.values.astype(float)
    out = np.zeros_like(vals)
    prev = 0.0
    started = False
    for i in range(len(vals)):
        v = vals[i]
        if np.isnan(v):
            out[i] = np.nan if not started else prev
            continue
        if not started:
            out[i] = v
            prev = v
            started = True
        else:
            prev = (1 - alpha) * v + alpha * prev
            out[i] = prev
    return pd.Series(out, index=series.index)


def _safe_div(numer, denom):
    """Divide two Series, returning 0 where denom==0 or result is inf/nan."""
    result = numer / denom.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def agg_futures_data(dl, tickers, exchanges, instruments='futures',
                     startDate=None, endDate=None,
                     barFreq=4, sampleFreq=None):
    """Build futures features at 1h resolution, apply exponential sums, subsample to sampleFreq.

    Loads raw 1h data, computes per-hour features, applies exponential sums / true EMA,
    then subsamples to the desired sampleFreq. Spot CVD is aggregated across all spot
    markets for each ticker.

    All ratio and normalization divisions are zero-protected (0 when denom == 0).
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(exchanges, str):
        exchanges = [exchanges]
    if isinstance(instruments, str):
        instruments = [instruments]
    if sampleFreq is None:
        sampleFreq = barFreq

    H1, H2, H3 = barFreq, 2 * barFreq, 24

    combos = [(t, e, i) for t in tickers for e in exchanges for i in instruments]
    parts = []
    for ticker, exch, inst in tqdm(combos, desc='Loading futures'):
        try:
            # always load at 1h resolution
            raw = dl.get_sophi(ticker, exch, inst, startDate, endDate,
                               barFreqInHours=1, sampleFreqInHours=1)
        except (FileNotFoundError, KeyError):
            continue

        # spot CVD & volume: sum across all spot markets for this ticker (at 1h)
        spot_markets = dl.marketForTicker(ticker, 'spot')
        spot_cvd = None
        spot_vol = None
        for sm in spot_markets:
            try:
                spot_raw = dl.get_sophi(ticker, sm, 'spot', startDate, endDate,
                                        barFreqInHours=1, sampleFreqInHours=1)
            except (FileNotFoundError, KeyError):
                continue
            aligned_cvd = spot_raw['cvd_usdt'].reindex(raw.index).fillna(0.0)
            aligned_vol = spot_raw['volume_usdt'].reindex(raw.index).fillna(0.0)
            spot_cvd = aligned_cvd if spot_cvd is None else spot_cvd + aligned_cvd
            spot_vol = aligned_vol if spot_vol is None else spot_vol + aligned_vol

        has_spot = spot_cvd is not None
        if not has_spot:
            spot_cvd = pd.Series(np.nan, index=raw.index)
            spot_vol = pd.Series(np.nan, index=raw.index)

        out = pd.DataFrame(index=raw.index)
        out.index.name = 'ts'
        out['ticker'] = ticker
        out['exchange'] = exch
        out['instrument'] = inst

        price = raw['close']
        vol = raw['volume_usdt']
        oi = raw['open_interest']
        cvd = raw['cvd_usdt']

        out['price'] = price
        out['volume_usdt'] = vol

        # 1. dprice, dprice_ratio with ES at H1, H2, H3
        dprice = price - price.shift(1)
        dprice_ratio = _safe_div(dprice, price.shift(1))
        for H in [H1, H2, H3]:
            out[f'dprice_es{H}'] = _exp_sum(dprice, H)
            out[f'dprice_ratio_es{H}'] = _exp_sum(dprice_ratio, H)

        # 2. oi exp sum at H1
        out[f'oi_es{H1}'] = _exp_sum(oi, H1)

        # 3. doi exp sum at H1
        doi = oi - oi.shift(1)
        out[f'doi_es{H1}'] = _exp_sum(doi, H1)

        # 4. doi_ratio exp sum at H1
        doi_ratio = _safe_div(doi, oi.shift(1))
        out[f'doi_ratio_es{H1}'] = _exp_sum(doi_ratio, H1)

        # 5. oi_native = oi/price, mirror 2-4
        oi_native = _safe_div(oi, price)
        out[f'oi_native_es{H1}'] = _exp_sum(oi_native, H1)
        doi_native = oi_native - oi_native.shift(1)
        out[f'doi_native_es{H1}'] = _exp_sum(doi_native, H1)
        doi_native_ratio = _safe_div(doi_native, oi_native.shift(1))
        out[f'doi_native_ratio_es{H1}'] = _exp_sum(doi_native_ratio, H1)

        # 6. cvd exp sums at H1, H2, H3
        cvd_es = {}
        for H in [H1, H2, H3]:
            cvd_es[H] = _exp_sum(cvd, H)
            out[f'cvd_es{H}'] = cvd_es[H]

        # 7. dcvd exp sums at H1, H2, H3
        dcvd = cvd - cvd.shift(1)
        dcvd_es = {}
        for H in [H1, H2, H3]:
            dcvd_es[H] = _exp_sum(dcvd, H)
            out[f'dcvd_es{H}'] = dcvd_es[H]

        # volume exp sums at H1, H2 (for cvd_vol / dcvd_vol ratios)
        vol_es = {}
        for H in [H1, H2]:
            vol_es[H] = _exp_sum(vol, H)

        # cvd_vol: ES(cvd)/ES(vol) at H1, H2
        for H in [H1, H2]:
            out[f'cvd_vol_es{H}'] = _safe_div(cvd_es[H], vol_es[H])

        # dcvd_vol: ES(dcvd)/ES(vol) at H1, H2
        for H in [H1, H2]:
            out[f'dcvd_vol_es{H}'] = _safe_div(dcvd_es[H], vol_es[H])

        # 8. dcvd_ratio exp sums at H1, H2, H3
        dcvd_ratio = _safe_div(dcvd, cvd.shift(1).abs())
        for H in [H1, H2, H3]:
            out[f'dcvd_ratio_es{H}'] = _exp_sum(dcvd_ratio, H)

        # 9. abs(cvd) true EMA at H=240, normalize features 6 and 7
        abs_cvd_ema240 = _ema(cvd.abs(), 240)
        out['abs_cvd_ema240'] = abs_cvd_ema240
        for H in [H1, H2, H3]:
            out[f'cvd_es{H}_norm'] = _safe_div(cvd_es[H], abs_cvd_ema240)
            out[f'dcvd_es{H}_norm'] = _safe_div(dcvd_es[H], abs_cvd_ema240)

        # 10. |cvd_es{H1}| diff, exp sum at H1
        abs_cvd_es_bar = cvd_es[H1].abs()
        d_abs_cvd_es_bar = abs_cvd_es_bar - abs_cvd_es_bar.shift(1)
        d_abs_cvd_es_bar_es = _exp_sum(d_abs_cvd_es_bar, H1)
        out[f'd_abs_cvd_es_bar_es{H1}'] = d_abs_cvd_es_bar_es

        # 11. ratio version
        d_abs_cvd_es_bar_ratio = _safe_div(d_abs_cvd_es_bar, abs_cvd_es_bar.shift(1))
        out[f'd_abs_cvd_es_bar_ratio_es{H1}'] = _exp_sum(d_abs_cvd_es_bar_ratio, H1)

        # 12. normalize feature 10 by feature 9
        out[f'd_abs_cvd_es_bar_es{H1}_norm'] = _safe_div(d_abs_cvd_es_bar_es, abs_cvd_ema240)

        # 13. Mirror 6-12 for spot_cvd
        scvd_es = {}
        for H in [H1, H2, H3]:
            scvd_es[H] = _exp_sum(spot_cvd, H)
            out[f'spot_cvd_es{H}'] = scvd_es[H]

        dspot_cvd = spot_cvd - spot_cvd.shift(1)
        dscvd_es = {}
        for H in [H1, H2, H3]:
            dscvd_es[H] = _exp_sum(dspot_cvd, H)
            out[f'dspot_cvd_es{H}'] = dscvd_es[H]

        dspot_cvd_ratio = _safe_div(dspot_cvd, spot_cvd.shift(1).abs())
        for H in [H1, H2, H3]:
            out[f'dspot_cvd_ratio_es{H}'] = _exp_sum(dspot_cvd_ratio, H)

        abs_spot_cvd_ema240 = _ema(spot_cvd.abs(), 240)
        out['abs_spot_cvd_ema240'] = abs_spot_cvd_ema240
        for H in [H1, H2, H3]:
            out[f'spot_cvd_es{H}_norm'] = _safe_div(scvd_es[H], abs_spot_cvd_ema240)
            out[f'dspot_cvd_es{H}_norm'] = _safe_div(dscvd_es[H], abs_spot_cvd_ema240)

        abs_spot_cvd_es_bar = scvd_es[H1].abs()
        d_abs_spot_cvd_es_bar = abs_spot_cvd_es_bar - abs_spot_cvd_es_bar.shift(1)
        d_abs_spot_cvd_es_bar_es = _exp_sum(d_abs_spot_cvd_es_bar, H1)
        out[f'd_abs_spot_cvd_es_bar_es{H1}'] = d_abs_spot_cvd_es_bar_es

        d_abs_spot_cvd_es_bar_ratio = _safe_div(d_abs_spot_cvd_es_bar, abs_spot_cvd_es_bar.shift(1))
        out[f'd_abs_spot_cvd_es_bar_ratio_es{H1}'] = _exp_sum(d_abs_spot_cvd_es_bar_ratio, H1)

        out[f'd_abs_spot_cvd_es_bar_es{H1}_norm'] = _safe_div(d_abs_spot_cvd_es_bar_es, abs_spot_cvd_ema240)

        # spot/futures volume features
        out['spot_volume_usdt'] = spot_vol
        out['spot_fut_vol_ratio'] = _safe_div(spot_vol, vol)
        fut_vol_es24 = _exp_sum(vol, 24)
        spot_vol_es24 = _exp_sum(spot_vol, 24)
        out['fut_volume_es24'] = fut_vol_es24
        out['spot_volume_es24'] = spot_vol_es24
        out['spot_fut_vol_es24_ratio'] = _safe_div(spot_vol_es24, fut_vol_es24)

        out['funding_rate'] = raw['funding_rate']
        out['funding_rate_avg'] = raw['funding_rate_avg']
        out['funding_interval_hours'] = raw['funding_interval_hours']
        out['date'] = raw['date']
        out['hoursSinceMidnight'] = raw['hoursSinceMidnight']

        # subsample to sampleFreq (end-of-window convention, same as get_sophi)
        out = out.iloc[sampleFreq - 1::sampleFreq]
        parts.append(out)

    return pd.concat(parts) if parts else pd.DataFrame()


def agg_withJustin_data(dl, tickers, exchanges, instruments='futures',
                        startDate=None, endDate=None,
                        barFreq=4, sampleFreq=None):
    """Same as agg_futures_data but extended with Justin's unique features.

    Adds (on top of agg_futures_data):
      - log_oi_level
      - funding_7d_avg (168h rolling mean)
      - Simple pct_change features at 1h/4h/24h/3d/7d horizons for price, volume,
        OI (dollar), OI-coin (oi/price)
      - Simple CVD diffs at 1h/4h/24h/7d for both perp and spot
      - Multi-exchange spot CVD 7d diff
      - Multi-exchange perp OI 7d diff (sum across all perp-like exchanges)
      - total_cvd_7d (perp + spot combined)
      - spot_price_vs_perp basis (spot avg close vs perp close)
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(exchanges, str):
        exchanges = [exchanges]
    if isinstance(instruments, str):
        instruments = [instruments]
    if sampleFreq is None:
        sampleFreq = barFreq

    H1, H2, H3 = barFreq, 2 * barFreq, 24
    PERP_INSTS = {'futures', 'perps', 'swap'}

    combos = [(t, e, i) for t in tickers for e in exchanges for i in instruments]
    parts = []
    for ticker, exch, inst in tqdm(combos, desc='Loading w/ Justin'):
        try:
            raw = dl.get_sophi(ticker, exch, inst, startDate, endDate,
                               barFreqInHours=1, sampleFreqInHours=1)
        except (FileNotFoundError, KeyError):
            continue

        # spot CVD, volume, close: aggregate across all spot markets at 1h
        spot_markets = dl.marketForTicker(ticker, 'spot')
        spot_cvd = None
        spot_vol = None
        spot_close_sum = None
        spot_close_cnt = pd.Series(0, index=raw.index)
        for sm in spot_markets:
            try:
                spot_raw = dl.get_sophi(ticker, sm, 'spot', startDate, endDate,
                                        barFreqInHours=1, sampleFreqInHours=1)
            except (FileNotFoundError, KeyError):
                continue
            aligned_cvd = spot_raw['cvd_usdt'].reindex(raw.index).fillna(0.0)
            aligned_vol = spot_raw['volume_usdt'].reindex(raw.index).fillna(0.0)
            aligned_close = spot_raw['close'].reindex(raw.index)
            spot_cvd = aligned_cvd if spot_cvd is None else spot_cvd + aligned_cvd
            spot_vol = aligned_vol if spot_vol is None else spot_vol + aligned_vol
            # sum only non-NaN closes, count them for mean
            mask = aligned_close.notna()
            spot_close_sum = aligned_close.fillna(0.0) if spot_close_sum is None \
                else spot_close_sum + aligned_close.fillna(0.0)
            spot_close_cnt = spot_close_cnt + mask.astype(int)

        has_spot = spot_cvd is not None
        if not has_spot:
            spot_cvd = pd.Series(np.nan, index=raw.index)
            spot_vol = pd.Series(np.nan, index=raw.index)
            spot_close_mean = pd.Series(np.nan, index=raw.index)
        else:
            spot_close_mean = _safe_div(spot_close_sum, spot_close_cnt.replace(0, np.nan))
            # restore NaN where nothing was observed
            spot_close_mean = spot_close_mean.where(spot_close_cnt > 0)

        # multi-exchange PERP OI: sum diff(7d) across all perp-like markets
        # (futures, perps, swap) for this ticker, in addition to the primary
        multi_oi_7d = pd.Series(0.0, index=raw.index)
        has_any_perp = False
        for inst_candidate in ['futures', 'perps', 'swap']:
            for pm in dl.marketForTicker(ticker, inst_candidate):
                if pm == exch and inst_candidate == inst:
                    continue  # primary handled below
                try:
                    perp_raw = dl.get_sophi(ticker, pm, inst_candidate, startDate, endDate,
                                            barFreqInHours=1, sampleFreqInHours=1)
                except (FileNotFoundError, KeyError):
                    continue
                aligned_oi = perp_raw['open_interest'].reindex(raw.index)
                multi_oi_7d = multi_oi_7d.add(aligned_oi.diff(7 * 24), fill_value=0)
                has_any_perp = True

        out = pd.DataFrame(index=raw.index)
        out.index.name = 'ts'
        out['ticker'] = ticker
        out['exchange'] = exch
        out['instrument'] = inst

        price = raw['close']
        vol = raw['volume_usdt']
        oi = raw['open_interest']
        cvd = raw['cvd_usdt']

        out['price'] = price
        out['volume_usdt'] = vol

        # ────── my original agg_futures_data features ──────
        dprice = price - price.shift(1)
        dprice_ratio = _safe_div(dprice, price.shift(1))
        for H in [H1, H2, H3]:
            out[f'dprice_es{H}'] = _exp_sum(dprice, H)
            out[f'dprice_ratio_es{H}'] = _exp_sum(dprice_ratio, H)

        out[f'oi_es{H1}'] = _exp_sum(oi, H1)
        doi = oi - oi.shift(1)
        out[f'doi_es{H1}'] = _exp_sum(doi, H1)
        doi_ratio = _safe_div(doi, oi.shift(1))
        out[f'doi_ratio_es{H1}'] = _exp_sum(doi_ratio, H1)

        oi_native = _safe_div(oi, price)
        out[f'oi_native_es{H1}'] = _exp_sum(oi_native, H1)
        doi_native = oi_native - oi_native.shift(1)
        out[f'doi_native_es{H1}'] = _exp_sum(doi_native, H1)
        doi_native_ratio = _safe_div(doi_native, oi_native.shift(1))
        out[f'doi_native_ratio_es{H1}'] = _exp_sum(doi_native_ratio, H1)

        cvd_es = {}
        for H in [H1, H2, H3]:
            cvd_es[H] = _exp_sum(cvd, H)
            out[f'cvd_es{H}'] = cvd_es[H]

        dcvd = cvd - cvd.shift(1)
        dcvd_es = {}
        for H in [H1, H2, H3]:
            dcvd_es[H] = _exp_sum(dcvd, H)
            out[f'dcvd_es{H}'] = dcvd_es[H]

        vol_es = {}
        for H in [H1, H2]:
            vol_es[H] = _exp_sum(vol, H)
        for H in [H1, H2]:
            out[f'cvd_vol_es{H}'] = _safe_div(cvd_es[H], vol_es[H])
            out[f'dcvd_vol_es{H}'] = _safe_div(dcvd_es[H], vol_es[H])

        dcvd_ratio = _safe_div(dcvd, cvd.shift(1).abs())
        for H in [H1, H2, H3]:
            out[f'dcvd_ratio_es{H}'] = _exp_sum(dcvd_ratio, H)

        abs_cvd_ema240 = _ema(cvd.abs(), 240)
        out['abs_cvd_ema240'] = abs_cvd_ema240
        for H in [H1, H2, H3]:
            out[f'cvd_es{H}_norm'] = _safe_div(cvd_es[H], abs_cvd_ema240)
            out[f'dcvd_es{H}_norm'] = _safe_div(dcvd_es[H], abs_cvd_ema240)

        abs_cvd_es_bar = cvd_es[H1].abs()
        d_abs_cvd_es_bar = abs_cvd_es_bar - abs_cvd_es_bar.shift(1)
        d_abs_cvd_es_bar_es = _exp_sum(d_abs_cvd_es_bar, H1)
        out[f'd_abs_cvd_es_bar_es{H1}'] = d_abs_cvd_es_bar_es
        d_abs_cvd_es_bar_ratio = _safe_div(d_abs_cvd_es_bar, abs_cvd_es_bar.shift(1))
        out[f'd_abs_cvd_es_bar_ratio_es{H1}'] = _exp_sum(d_abs_cvd_es_bar_ratio, H1)
        out[f'd_abs_cvd_es_bar_es{H1}_norm'] = _safe_div(d_abs_cvd_es_bar_es, abs_cvd_ema240)

        scvd_es = {}
        for H in [H1, H2, H3]:
            scvd_es[H] = _exp_sum(spot_cvd, H)
            out[f'spot_cvd_es{H}'] = scvd_es[H]

        dspot_cvd = spot_cvd - spot_cvd.shift(1)
        dscvd_es = {}
        for H in [H1, H2, H3]:
            dscvd_es[H] = _exp_sum(dspot_cvd, H)
            out[f'dspot_cvd_es{H}'] = dscvd_es[H]

        dspot_cvd_ratio = _safe_div(dspot_cvd, spot_cvd.shift(1).abs())
        for H in [H1, H2, H3]:
            out[f'dspot_cvd_ratio_es{H}'] = _exp_sum(dspot_cvd_ratio, H)

        abs_spot_cvd_ema240 = _ema(spot_cvd.abs(), 240)
        out['abs_spot_cvd_ema240'] = abs_spot_cvd_ema240
        for H in [H1, H2, H3]:
            out[f'spot_cvd_es{H}_norm'] = _safe_div(scvd_es[H], abs_spot_cvd_ema240)
            out[f'dspot_cvd_es{H}_norm'] = _safe_div(dscvd_es[H], abs_spot_cvd_ema240)

        abs_spot_cvd_es_bar = scvd_es[H1].abs()
        d_abs_spot_cvd_es_bar = abs_spot_cvd_es_bar - abs_spot_cvd_es_bar.shift(1)
        d_abs_spot_cvd_es_bar_es = _exp_sum(d_abs_spot_cvd_es_bar, H1)
        out[f'd_abs_spot_cvd_es_bar_es{H1}'] = d_abs_spot_cvd_es_bar_es
        d_abs_spot_cvd_es_bar_ratio = _safe_div(d_abs_spot_cvd_es_bar, abs_spot_cvd_es_bar.shift(1))
        out[f'd_abs_spot_cvd_es_bar_ratio_es{H1}'] = _exp_sum(d_abs_spot_cvd_es_bar_ratio, H1)
        out[f'd_abs_spot_cvd_es_bar_es{H1}_norm'] = _safe_div(d_abs_spot_cvd_es_bar_es, abs_spot_cvd_ema240)

        out['spot_volume_usdt'] = spot_vol
        out['spot_fut_vol_ratio'] = _safe_div(spot_vol, vol)
        fut_vol_es24 = _exp_sum(vol, 24)
        spot_vol_es24 = _exp_sum(spot_vol, 24)
        out['fut_volume_es24'] = fut_vol_es24
        out['spot_volume_es24'] = spot_vol_es24
        out['spot_fut_vol_es24_ratio'] = _safe_div(spot_vol_es24, fut_vol_es24)

        # ────── Justin's unique features (no BTC) ──────

        # level
        out['log_oi_level'] = np.log1p(oi)

        # funding 7d avg
        out['funding_7d_avg'] = raw['funding_rate'].rolling(7 * 24, min_periods=7 * 12).mean()

        # simple pct returns
        out['ret_1h_pct'] = price.pct_change(1) * 100
        out['ret_4h_pct'] = price.pct_change(4) * 100
        out['ret_1d_pct'] = price.pct_change(24) * 100
        out['ret_3d_pct'] = price.pct_change(3 * 24) * 100

        # simple volume pct changes
        out['vol_1h_pct'] = vol.pct_change(1) * 100
        out['vol_24h_pct'] = vol.pct_change(24) * 100
        vol_4h = vol.rolling(4).sum()
        vol_4h_prev = vol.shift(4).rolling(4).sum()
        out['vol_chg_4h_pct'] = _safe_div(vol_4h, vol_4h_prev).replace(0, np.nan).fillna(1.0).sub(1.0) * 100

        # OI pct changes
        out['oi_chg_4h_pct'] = oi.pct_change(4) * 100
        out['oi_3d_chg_pct'] = oi.pct_change(3 * 24) * 100
        out['oi_7d_chg_pct'] = oi.pct_change(7 * 24) * 100

        # OI-coin pct changes (oi / price)
        out['oi_coin_1h_chg_pct'] = oi_native.pct_change(1) * 100
        out['oi_coin_chg_4h_pct'] = oi_native.pct_change(4) * 100
        out['oi_coin_24h_chg_pct'] = oi_native.pct_change(24) * 100
        out['oi_coin_3d_chg_pct'] = oi_native.pct_change(3 * 24) * 100
        out['oi_coin_7d_chg_pct'] = oi_native.pct_change(7 * 24) * 100

        # simple CVD diffs (perp)
        out['perp_cvd_1h'] = cvd.diff(1)
        out['perp_cvd_4h'] = cvd.diff(4)
        out['perp_cvd_24h'] = cvd.diff(24)
        out['perp_cvd_7d'] = cvd.diff(7 * 24)

        # simple CVD diffs (spot, summed)
        out['spot_cvd_1h'] = spot_cvd.diff(1)
        out['spot_cvd_4h'] = spot_cvd.diff(4)
        out['spot_cvd_24h'] = spot_cvd.diff(24)
        out['spot_cvd_7d'] = spot_cvd.diff(7 * 24)

        # multi-exchange spot CVD 7d (same as spot_cvd_7d since spot_cvd
        # already sums across spot markets) — alias for clarity
        out['multi_exchange_spot_cvd_7d'] = out['spot_cvd_7d']

        # total CVD 7d (perp + spot)
        out['total_cvd_7d'] = out['perp_cvd_7d'].fillna(0) + out['spot_cvd_7d'].fillna(0)

        # multi-exchange perp OI 7d change (own + others)
        own_oi_7d = oi.diff(7 * 24)
        out['multi_exchange_oi_7d_chg'] = own_oi_7d.fillna(0) + (multi_oi_7d if has_any_perp else 0)

        # spot-perp basis
        out['spot_price_vs_perp'] = _safe_div(spot_close_mean - price, price)

        out['funding_rate'] = raw['funding_rate']
        out['funding_rate_avg'] = raw['funding_rate_avg']
        out['funding_interval_hours'] = raw['funding_interval_hours']
        out['date'] = raw['date']
        out['hoursSinceMidnight'] = raw['hoursSinceMidnight']

        out = out.iloc[sampleFreq - 1::sampleFreq]
        parts.append(out)

    return pd.concat(parts) if parts else pd.DataFrame()


def agg_justin_data(dl, tickers, exchanges, instruments='futures',
                    startDate=None, endDate=None,
                    barFreq=4, sampleFreq=None):
    """Justin's feature set only (no exponential sums, no BTC context).

    Features:
      - Level: log_oi_level
      - Funding: funding_rate, funding_7d_avg
      - Returns (simple pct_change): ret_1h_pct, ret_4h_pct, ret_1d_pct, ret_3d_pct
      - Volume pct: vol_1h_pct, vol_24h_pct, vol_chg_4h_pct
      - OI pct: oi_chg_4h_pct, oi_3d_chg_pct, oi_7d_chg_pct
      - OI-coin pct (oi/price): oi_coin_1h/4h/24h/3d/7d_chg_pct
      - Perp CVD diffs: perp_cvd_1h, cvd_chg_4h, perp_cvd_24h, perp_cvd_7d
      - Spot CVD diffs: spot_cvd_1h/4h/24h/7d (spot aggregated across all spot markets)
      - Spot/perp ratios: spot_perp_vol_ratio (24h), spot_price_vs_perp
      - Multi-exchange: multi_exchange_spot_cvd_7d, total_cvd_7d, multi_exchange_oi_7d_chg

    Skipped from Justin's original list:
      - market_cap_proxy, oi_mcap_ratio: degenerate since OI in data2 is in dollars
      - BTC returns: no BTC data available in data2
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(exchanges, str):
        exchanges = [exchanges]
    if isinstance(instruments, str):
        instruments = [instruments]
    if sampleFreq is None:
        sampleFreq = barFreq

    combos = [(t, e, i) for t in tickers for e in exchanges for i in instruments]
    parts = []
    for ticker, exch, inst in tqdm(combos, desc='Loading justin'):
        try:
            raw = dl.get_sophi(ticker, exch, inst, startDate, endDate,
                               barFreqInHours=1, sampleFreqInHours=1)
        except (FileNotFoundError, KeyError):
            continue

        # aggregate spot: cvd, volume, close (mean) across spot markets
        spot_markets = dl.marketForTicker(ticker, 'spot')
        spot_cvd = None
        spot_vol = None
        spot_close_sum = None
        spot_close_cnt = pd.Series(0, index=raw.index)
        for sm in spot_markets:
            try:
                spot_raw = dl.get_sophi(ticker, sm, 'spot', startDate, endDate,
                                        barFreqInHours=1, sampleFreqInHours=1)
            except (FileNotFoundError, KeyError):
                continue
            aligned_cvd = spot_raw['cvd_usdt'].reindex(raw.index).fillna(0.0)
            aligned_vol = spot_raw['volume_usdt'].reindex(raw.index).fillna(0.0)
            aligned_close = spot_raw['close'].reindex(raw.index)
            spot_cvd = aligned_cvd if spot_cvd is None else spot_cvd + aligned_cvd
            spot_vol = aligned_vol if spot_vol is None else spot_vol + aligned_vol
            mask = aligned_close.notna()
            spot_close_sum = aligned_close.fillna(0.0) if spot_close_sum is None \
                else spot_close_sum + aligned_close.fillna(0.0)
            spot_close_cnt = spot_close_cnt + mask.astype(int)

        has_spot = spot_cvd is not None
        if not has_spot:
            spot_cvd = pd.Series(np.nan, index=raw.index)
            spot_vol = pd.Series(np.nan, index=raw.index)
            spot_close_mean = pd.Series(np.nan, index=raw.index)
        else:
            spot_close_mean = _safe_div(spot_close_sum, spot_close_cnt.replace(0, np.nan))
            spot_close_mean = spot_close_mean.where(spot_close_cnt > 0)

        # multi-exchange perp OI 7d change (sum across all perp-like markets)
        multi_oi_7d = pd.Series(0.0, index=raw.index)
        has_any_other_perp = False
        for inst_candidate in ['futures', 'perps', 'swap']:
            for pm in dl.marketForTicker(ticker, inst_candidate):
                if pm == exch and inst_candidate == inst:
                    continue
                try:
                    perp_raw = dl.get_sophi(ticker, pm, inst_candidate, startDate, endDate,
                                            barFreqInHours=1, sampleFreqInHours=1)
                except (FileNotFoundError, KeyError):
                    continue
                aligned_oi = perp_raw['open_interest'].reindex(raw.index)
                multi_oi_7d = multi_oi_7d.add(aligned_oi.diff(7 * 24), fill_value=0)
                has_any_other_perp = True

        out = pd.DataFrame(index=raw.index)
        out.index.name = 'ts'
        out['ticker'] = ticker
        out['exchange'] = exch
        out['instrument'] = inst

        price = raw['close']
        vol = raw['volume_usdt']
        oi = raw['open_interest']
        cvd = raw['cvd_usdt']
        oi_native = _safe_div(oi, price)

        out['price'] = price
        out['volume_usdt'] = vol

        # level
        out['log_oi_level'] = np.log1p(oi)

        # funding
        out['funding_rate'] = raw['funding_rate']
        out['funding_7d_avg'] = raw['funding_rate'].rolling(7 * 24, min_periods=7 * 12).mean()

        # simple price returns
        out['ret_1h_pct'] = price.pct_change(1) * 100
        out['ret_4h_pct'] = price.pct_change(4) * 100
        out['ret_1d_pct'] = price.pct_change(24) * 100
        out['ret_3d_pct'] = price.pct_change(3 * 24) * 100

        # simple volume pct changes
        out['vol_1h_pct'] = vol.pct_change(1) * 100
        out['vol_24h_pct'] = vol.pct_change(24) * 100
        vol_4h = vol.rolling(4).sum()
        vol_4h_prev = vol.shift(4).rolling(4).sum()
        out['vol_chg_4h_pct'] = _safe_div(vol_4h, vol_4h_prev).replace(0, np.nan).fillna(1.0).sub(1.0) * 100

        # OI pct changes (dollar)
        out['oi_chg_4h_pct'] = oi.pct_change(4) * 100
        out['oi_3d_chg_pct'] = oi.pct_change(3 * 24) * 100
        out['oi_7d_chg_pct'] = oi.pct_change(7 * 24) * 100

        # OI-coin pct changes
        out['oi_coin_1h_chg_pct'] = oi_native.pct_change(1) * 100
        out['oi_coin_chg_4h_pct'] = oi_native.pct_change(4) * 100
        out['oi_coin_24h_chg_pct'] = oi_native.pct_change(24) * 100
        out['oi_coin_3d_chg_pct'] = oi_native.pct_change(3 * 24) * 100
        out['oi_coin_7d_chg_pct'] = oi_native.pct_change(7 * 24) * 100

        # perp CVD diffs
        out['perp_cvd_1h'] = cvd.diff(1)
        out['cvd_chg_4h'] = cvd.diff(4)
        out['perp_cvd_24h'] = cvd.diff(24)
        out['perp_cvd_7d'] = cvd.diff(7 * 24)

        # spot CVD diffs (from aggregated spot cvd)
        out['spot_cvd_1h'] = spot_cvd.diff(1)
        out['spot_cvd_4h'] = spot_cvd.diff(4)
        out['spot_cvd_24h'] = spot_cvd.diff(24)
        out['spot_cvd_7d'] = spot_cvd.diff(7 * 24)

        # multi-exchange aggregates
        out['multi_exchange_spot_cvd_7d'] = out['spot_cvd_7d']
        out['total_cvd_7d'] = out['perp_cvd_7d'].fillna(0) + out['spot_cvd_7d'].fillna(0)
        own_oi_7d = oi.diff(7 * 24)
        out['multi_exchange_oi_7d_chg'] = own_oi_7d.fillna(0) + (multi_oi_7d if has_any_other_perp else 0)

        # spot/perp ratios
        spot_vol_24h = spot_vol.rolling(24, min_periods=12).sum()
        perp_vol_24h = vol.rolling(24, min_periods=12).sum()
        out['spot_perp_vol_ratio'] = _safe_div(spot_vol_24h, perp_vol_24h)
        out['spot_price_vs_perp'] = _safe_div(spot_close_mean - price, price)

        out['funding_interval_hours'] = raw['funding_interval_hours']
        out['date'] = raw['date']
        out['hoursSinceMidnight'] = raw['hoursSinceMidnight']

        out = out.iloc[sampleFreq - 1::sampleFreq]
        parts.append(out)

    return pd.concat(parts) if parts else pd.DataFrame()


def makeReturns(dl, df, horizons=1, adjustFundingRate=False,
                addPumpLabel=False, pumpThreshold=1.5,
                pumpForwardHours=14*24, pumpMinHistoryHours=3*24):
    """Compute forward returns matching the timestamps in df.

    Parameters
    ----------
    dl : DataLoader
    df : pd.DataFrame – output of agg_daniel_data (must have ticker, exchange,
         instrument columns and a ts index)
    horizons : int or list of int – forward return horizons in hours
    adjustFundingRate : bool – if True, subtract cumulative per-hour funding cost
    addPumpLabel : bool – if True, add 'pump_label' (1 if peak forward return
        within pumpForwardHours >= pumpThreshold)
    pumpThreshold : float – return threshold (e.g. 1.5 = 150%)
    pumpForwardHours : int – look-forward window in hours (default 14 days)
    pumpMinHistoryHours : int – min prior history required (default 3 days);
        rows earlier than this within a group get NaN label

    Returns
    -------
    pd.DataFrame indexed by ts with columns:
        ticker, exchange, instrument, price, ret_{h}h for each horizon,
        date, hoursSinceMidnight, and optionally pump_label
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

        # pump label: peak forward return over pumpForwardHours >= pumpThreshold
        if addPumpLabel:
            closes = price.values
            n = len(closes)
            labels = np.full(n, np.nan)
            for i in range(n):
                if i < pumpMinHistoryHours:
                    continue
                fwd_end = min(i + pumpForwardHours + 1, n)
                if fwd_end <= i + 1:
                    continue
                fwd = closes[i + 1:fwd_end]
                if len(fwd) == 0 or np.isnan(closes[i]) or closes[i] == 0:
                    continue
                peak = np.nanmax(fwd)
                if np.isnan(peak):
                    continue
                peak_ret = peak / closes[i] - 1
                labels[i] = 1.0 if peak_ret >= pumpThreshold else 0.0
            all_ret['pump_label'] = labels

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
    ret_cols = [c for c in returns.columns if c.startswith('ret_') or c == 'pump_label'] 
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


def findBumpAndDump(prices, left=24, right=24, bumpThreshold=0.5, dumpThreshold=0.6):
    """Detect pump-and-dump events in a single ticker's price series.

    Parameters
    ----------
    prices : pd.Series – price series indexed by timestamp
    left : int – lookback window in bars
    right : int – lookforward window in bars
    bumpThreshold : float – minimum rise ratio (e.g. 0.5 = 50%)
    dumpThreshold : float – minimum fall ratio (e.g. 0.6 = 60%)

    Returns
    -------
    (peak_time, context_prices) or (None, None) if no event found
        peak_time : timestamp of the peak
        context_prices : pd.Series sliced ±20 days around the peak
    """
    p = prices.values
    idx = prices.index
    n = len(p)

    candidates = []
    for t in range(left, n - right):
        rise = (p[t] - p[t - left]) / p[t - left]
        fall = (p[t] - p[t + right]) / p[t]
        if rise > bumpThreshold and fall > dumpThreshold:
            candidates.append((t, p[t]))

    if not candidates:
        return None, None

    # pick the candidate with the highest price
    best_t = max(candidates, key=lambda x: x[1])[0]
    peak_time = idx[best_t]

    # slice ±20 days around peak
    start = peak_time - pd.Timedelta(days=20)
    end = peak_time + pd.Timedelta(days=20)
    context = prices[(prices.index >= start) & (prices.index <= end)]

    return peak_time, context


def findBumpAndDumpV2(prices, threshold=2):
    """Detect pump-and-dump by checking if max price exceeds threshold * median.

    Parameters
    ----------
    prices : pd.Series – price series indexed by timestamp
    threshold : float – max price must be > threshold * median price

    Returns
    -------
    (peak_time, context_prices) or (None, None) if no event found
        peak_time : timestamp of the peak
        context_prices : pd.Series sliced ±20 days around the peak
    """
    peak_idx = prices.idxmax()
    max_price = prices[peak_idx]
    median_before = prices[prices.index < peak_idx].median()

    if np.isnan(median_before) or max_price < threshold * median_before:
        return None, None

    peak_time = peak_idx
    start = peak_time - pd.Timedelta(days=20)
    end = peak_time + pd.Timedelta(days=20)
    context = prices[(prices.index >= start) & (prices.index <= end)]

    return peak_time, context


def peakCvdProfile(raw_df, threshold=2, left_days=20, right_days=5):
    """Extract normalized CVD profile around a detected pump-and-dump peak.

    Parameters
    ----------
    raw_df : pd.DataFrame – output of get/get_sophi with 'close' and 'cvd_usdt' columns,
        indexed by timestamp (1h bars)
    threshold : float – passed to findBumpAndDumpV2
    left_days : int – days before peak to include
    right_days : int – days after peak to include

    Returns
    -------
    result : np.ndarray of shape (left_days*24 + right_days*24,) – normalized CVD
    price_profile : np.ndarray of same shape – price / price_max
    peak_time : timestamp
    Returns (None, None, None) if no peak found.
    """
    peak_time, _ = findBumpAndDumpV2(raw_df['close'], threshold=threshold)
    if peak_time is None:
        return None, None, None

    total_len = left_days * 24 + right_days * 24
    start = peak_time - pd.Timedelta(days=left_days)
    end = peak_time + pd.Timedelta(days=right_days)

    # context window
    context_mask = (raw_df.index >= start) & (raw_df.index < end)
    context_cvd = raw_df.loc[context_mask, 'cvd_usdt']

    # denominator: 80th percentile of |cvd_usdt| outside the context
    outside_cvd = raw_df.loc[~context_mask, 'cvd_usdt']
    denom = np.quantile(np.abs(outside_cvd.dropna().values), 0.8)
    if denom == 0:
        denom = 1.0

    normalized = context_cvd.values / denom

    # price profile: price / price_max
    context_price = raw_df.loc[context_mask, 'close']
    price_max = raw_df['close'].max()
    norm_price = context_price.values / price_max

    # pad to fixed length: left_days*24 bars before peak, right_days*24 after
    actual_start = context_cvd.index[0] if len(context_cvd) > 0 else peak_time
    left_pad = max(0, int((actual_start - start).total_seconds() / 3600))
    right_pad = max(0, total_len - left_pad - len(normalized))

    result = np.concatenate([
        np.zeros(left_pad),
        normalized,
        np.zeros(right_pad),
    ])[:total_len]
    if len(result) < total_len:
        result = np.concatenate([result, np.zeros(total_len - len(result))])

    price_profile = np.concatenate([
        np.zeros(left_pad),
        norm_price,
        np.zeros(right_pad),
    ])[:total_len]
    if len(price_profile) < total_len:
        price_profile = np.concatenate([price_profile, np.zeros(total_len - len(price_profile))])

    return result, price_profile, peak_time


def peakPredProfiles(pred, O, peak_ratio=1.5, left_days=10, right_days=5):
    """Extract normalized price and prediction profiles around each ticker's peak.

    For each ticker in O:
      1. Find the max-price point.
      2. If max_price / avg_price > peak_ratio, include this ticker.
      3. Extract a window from [peak - left_days, peak + right_days].
      4. Pad edges by nearest available value (ffill/bfill).
      5. Normalize price by max_price.

    Returns two matrices (n_tickers, T) sorted by max/avg ratio (largest first).

    Parameters
    ----------
    pred : np.ndarray (n,) – prediction aligned with O's rows (by position)
    O : pd.DataFrame – must have 'ticker', 'price', 'date', 'hoursSinceMidnight'
    peak_ratio : float – inclusion threshold on max_price/avg_price
    left_days, right_days : int – window around peak in days

    Returns
    -------
    price_matrix : np.ndarray (n_events, T) – normalized prices (peak=1)
    pred_matrix : np.ndarray (n_events, T) – corresponding predictions
    tickers : list – ticker names in sorted order (matching rows)
    peak_times : list – peak timestamps
    """
    pred = np.asarray(pred, dtype=float)

    # reconstruct ts from date + hoursSinceMidnight; attach pred
    work = O.copy().reset_index(drop=True)
    ts = pd.to_datetime(work['date'].astype(int).astype(str), format='%Y%m%d', utc=True) \
        + pd.to_timedelta(work['hoursSinceMidnight'].astype(int), unit='h')
    work = work.assign(_ts=ts, _pred=pred).set_index('_ts').sort_index()

    # detect bar spacing from first ticker
    first_ticker = work['ticker'].iloc[0]
    first_idx = work[work['ticker'] == first_ticker].sort_index().index
    if len(first_idx) < 2:
        raise ValueError("Not enough rows to detect bar spacing")
    bar_hours = (first_idx[1] - first_idx[0]).total_seconds() / 3600
    bars_per_day = 24 / bar_hours
    T = int(round((left_days + right_days) * bars_per_day))

    results = []  # list of (ratio, price_row, pred_row, ticker, peak_time)
    for ticker, grp in work.groupby('ticker'):
        grp = grp.sort_index()
        grp = grp[~grp.index.duplicated(keep='first')]
        prices = grp['price']
        if len(prices) == 0:
            continue
        max_price = prices.max()
        avg_price = prices.mean()
        if avg_price == 0 or np.isnan(avg_price):
            continue
        ratio = max_price / avg_price
        if ratio <= peak_ratio:
            continue

        peak_time = prices.idxmax()
        start = peak_time - pd.Timedelta(days=left_days)

        # regular grid at bar_hours spacing
        grid = pd.date_range(start=start, periods=T, freq=f'{int(bar_hours)}h')

        price_aligned = prices.reindex(grid, method='nearest',
                                       tolerance=pd.Timedelta(hours=int(bar_hours)))
        price_aligned = price_aligned.ffill().bfill()

        pred_aligned = grp['_pred'].reindex(grid, method='nearest',
                                            tolerance=pd.Timedelta(hours=int(bar_hours)))
        pred_aligned = pred_aligned.ffill().bfill()

        norm_price = price_aligned.values / max_price
        results.append((ratio, norm_price, pred_aligned.values, ticker, peak_time))

    if not results:
        return (np.zeros((0, T)), np.zeros((0, T)), [], [])

    # sort by ratio descending
    results.sort(key=lambda x: x[0], reverse=True)

    price_matrix = np.stack([r[1] for r in results])
    pred_matrix = np.stack([r[2] for r in results])
    tickers = [r[3] for r in results]
    peak_times = [r[4] for r in results]

    return price_matrix, pred_matrix, tickers, peak_times


def getDailyReturnExact(long_short, y_valid, o_valid, horizon='ret_24h',
                        masked_tickers=None):
    """Per-date daily return aggregated as mean-of-hourly-means.

    For each date:
      1. Group by hoursSinceMidnight
      2. For each hour, compute mean(return * long_short) across all tickers
      3. Average the hourly means to get the daily return

    Parameters
    ----------
    long_short : array-like (n,) – position vector (+1/-1/0)
    y_valid : pd.DataFrame – must contain `horizon` column
    o_valid : pd.DataFrame – must contain 'date', 'hoursSinceMidnight'
    horizon : str – column name in y_valid
    masked_tickers : list of str or None – tickers whose returns should be
        zeroed out (rows with these tickers contribute 0 to the signal*return
        numerator but still count in the position denominator)

    Returns
    -------
    pd.Series – indexed by date, daily return values
    """
    ls = np.asarray(long_short, dtype=float).copy()
    if masked_tickers is not None and len(masked_tickers) > 0:
        mask = o_valid['ticker'].isin(masked_tickers).values
        ls[mask] = 0.0
    y = y_valid[horizon].values
    df = pd.DataFrame({
        'date': o_valid['date'].values,
        'hour': o_valid['hoursSinceMidnight'].values,
        'num': ls * y,
        'den': np.abs(ls),
    })
    # sum per (date, hour)
    grp = df.groupby(['date', 'hour']).agg(num=('num', 'sum'), den=('den', 'sum'))
    # per-hour return: sum(ls*ret) / sum(|ls|), or 0 if no position
    grp['r'] = np.where(grp['den'] > 0, grp['num'] / grp['den'].replace(0, np.nan), 0.0)
    # active flag: include only hours where we actually had a position
    grp['active'] = grp['den'] > 0

    # average only active hours within each date
    def _mean_active(g):
        active = g[g['active']]
        if len(active) == 0:
            return 0.0
        return active['r'].mean()

    daily = grp.groupby('date').apply(_mean_active)
    return daily


def causal_long_short(long_short, O, cooldown_hours=48):
    """Enforce per-direction cooldown on a long_short signal vector.

    After a +1 signal at time t, any subsequent +1 within `cooldown_hours`
    (same ticker) is zeroed out. Same for -1. A +1 does not affect subsequent
    -1 signals and vice versa (direction flips are allowed).

    Parameters
    ----------
    long_short : array-like (n,) – values in {+1, -1, 0}
    O : pd.DataFrame – must contain 'date' and 'hoursSinceMidnight'; if 'ticker'
        is present, the cooldown is applied per ticker independently
    cooldown_hours : int – cooldown window in hours (default 48)

    Returns
    -------
    np.ndarray (n,) – filtered long_short vector (same order as input)
    """
    ls = np.asarray(long_short, dtype=float).copy()
    n = len(ls)

    # reconstruct ts hours since epoch
    date = O['date'].astype(int).astype(str).values
    hour = O['hoursSinceMidnight'].astype(int).values
    ts = pd.to_datetime(date, format='%Y%m%d', utc=True) + pd.to_timedelta(hour, unit='h')
    ts_hours = (ts.astype('int64') // (3600 * 10**9)).to_numpy()  # epoch hours

    ticker = O['ticker'].values if 'ticker' in O.columns else np.zeros(n, dtype=object)

    # process per ticker
    out = ls.copy()
    for t_val in pd.unique(ticker):
        mask = (ticker == t_val)
        idx = np.where(mask)[0]
        # sort within ticker by time
        order = np.argsort(ts_hours[idx])
        sorted_idx = idx[order]
        sorted_hours = ts_hours[sorted_idx]
        sorted_ls = out[sorted_idx].copy()

        last_long = -np.inf
        last_short = -np.inf
        for k, i in enumerate(sorted_idx):
            h = sorted_hours[k]
            if sorted_ls[k] == 1:
                if h - last_long < cooldown_hours:
                    out[i] = 0
                else:
                    last_long = h
            elif sorted_ls[k] == -1:
                if h - last_short < cooldown_hours:
                    out[i] = 0
                else:
                    last_short = h

    return out


def showFeatureImportance(model, feature_names=None, importance_type='gain', top=None):
    """Plot LightGBM feature importance as a horizontal bar chart.

    Parameters
    ----------
    model : lgb.Booster – fitted model from fitLgb
    feature_names : list of str or None – if None, uses model.feature_name()
    importance_type : 'gain' or 'split' – gain = total loss reduction,
        split = number of times feature was used
    top : int or None – show only the top N features (default: all)

    Returns
    -------
    pd.DataFrame – columns [feature, importance], sorted descending
    """
    import matplotlib.pyplot as plt

    if feature_names is None:
        feature_names = model.feature_name()
    importance = model.feature_importance(importance_type=importance_type)

    df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    plot_df = df.head(top) if top else df
    n = len(plot_df)

    fig, ax = plt.subplots(1, 1, figsize=(8, max(3, 0.3 * n)))
    ax.barh(np.arange(n)[::-1], plot_df['importance'].values)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels(plot_df['feature'].values, fontsize=8)
    ax.set_xlabel(f'importance ({importance_type})')
    ax.set_title(f'Feature importance ({importance_type})')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    return df


def showLongShortStockSelection(long_short, O_valid):
    """Heatmap of long/short/flat positions: stocks x dates.

    For rows with multiple (ticker, date) entries (different hours), the sign
    of the sum is shown.

    Parameters
    ----------
    long_short : array-like (n,) – values in {+1, -1, 0}
    O_valid : pd.DataFrame – must contain 'ticker', 'date'
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ls = np.asarray(long_short, dtype=float)
    tickers = O_valid['ticker'].values
    dates = O_valid['date'].values

    df = pd.DataFrame({'ticker': tickers, 'date': dates, 'ls': ls})
    # aggregate multi-hour entries by sum, then take sign
    agg = df.groupby(['ticker', 'date'])['ls'].sum().reset_index()
    agg['pos'] = np.sign(agg['ls']).astype(int)

    unique_dates = sorted(agg['date'].unique())
    # sort tickers by total activity (|long_short| sum) descending
    activity = df.assign(abs_ls=df['ls'].abs()).groupby('ticker')['abs_ls'].sum()
    unique_tickers = activity.sort_values(ascending=False).index.tolist()

    ticker_to_row = {t: i for i, t in enumerate(unique_tickers)}
    date_to_col = {d: j for j, d in enumerate(unique_dates)}

    matrix = np.zeros((len(unique_tickers), len(unique_dates)))
    for _, row in agg.iterrows():
        r = ticker_to_row[row['ticker']]
        c = date_to_col[row['date']]
        matrix[r, c] = row['pos']

    cmap = ListedColormap(['green', 'white', 'red'])
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('date')
    ax.set_ylabel('ticker (more active -> top)')
    ax.set_title('Long/short stock selection')

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


def getLongShortInfo(long_short, O_val):
    """Return a DataFrame of nonzero long/short signals with timestamps.

    Parameters
    ----------
    long_short : array-like (n,) – values in {+1, -1, 0}
    O_val : pd.DataFrame – must contain 'ticker', 'date', 'hoursSinceMidnight'

    Returns
    -------
    pd.DataFrame with columns [ts, date, hoursSinceMidnight, ticker, signal]
        sorted by ts, only rows where signal != 0
    """
    ls = np.asarray(long_short, dtype=float)
    date = O_val['date'].astype(int).astype(str).values
    hour = O_val['hoursSinceMidnight'].astype(int).values
    ts = pd.to_datetime(date, format='%Y%m%d', utc=True) + pd.to_timedelta(hour, unit='h')

    df = pd.DataFrame({
        'ts': ts,
        'date': O_val['date'].values,
        'hoursSinceMidnight': O_val['hoursSinceMidnight'].values,
        'ticker': O_val['ticker'].values,
        'signal': ls.astype(int),
    })
    df = df[df['signal'] != 0].sort_values('ts').reset_index(drop=True)
    return df


def plotDailyReturnExactTrades(long_short, Y, O, horizon='ret_24h'):
    """Plot cumsum of per-trade P&L with date-anchored x-axis.

    Each trade = one row with long_short != 0.
    x-axis: each unique date is mapped to an integer (0, 1, 2, ...). Trades
    within a date are uniformly spaced in [date_idx, date_idx+1) — i.e., if
    a date has N trades they land at date_idx + i/N for i=0..N-1.

    Parameters
    ----------
    long_short : array-like (n,) – values in {+1, -1, 0}
    Y : pd.DataFrame – must contain `horizon` column
    O : pd.DataFrame – must contain 'date', 'hoursSinceMidnight'
    horizon : str – return column in Y

    Returns
    -------
    (x, cum_pnl) : tuple of np.ndarray – x positions and cumulative P&L
    """
    import matplotlib.pyplot as plt

    ls = np.asarray(long_short, dtype=float)
    y = Y[horizon].values
    pnl_all = ls * y

    df = pd.DataFrame({
        'date': O['date'].values,
        'hour': O['hoursSinceMidnight'].values,
        'ls': ls,
        'pnl': pnl_all,
    })
    df = df[df['ls'] != 0].sort_values(['date', 'hour']).reset_index(drop=True)
    if len(df) == 0:
        print("No trades.")
        return np.array([]), np.array([])

    # map each unique date to integer position
    unique_dates = sorted(df['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    # for each trade compute x = date_idx + i / N_that_date
    xs = np.zeros(len(df))
    for d, grp in df.groupby('date'):
        n = len(grp)
        idx = date_to_idx[d]
        xs[grp.index.values] = idx + np.arange(n) / n

    n_trades = len(df)
    cum = np.cumsum(df['pnl'].fillna(0).values) / n_trades

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(xs, cum, marker='.', markersize=3, linewidth=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax.set_xlabel('date index (fractional within each day)')
    ax.set_ylabel('cumsum(pnl) / n_trades')
    ax.set_title(f'Per-trade cumsum/n ({horizon}, {n_trades} trades, {len(unique_dates)} days, avg={cum[-1]:.4f})')
    ax.grid(True, alpha=0.3)

    # mark integer date boundaries
    step = max(1, len(unique_dates) // 15)
    tick_positions = [i for i in range(0, len(unique_dates), step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(unique_dates[i]) for i in tick_positions], rotation=90, fontsize=7)
    plt.tight_layout()

    return xs, cum


def long_short_tradeReturn(long, short, O, stoploss, min_horizon=4, max_horizon=24):
    """Simulate long-only trades with stop-loss and short-signal exit.

    For each ticker (processed independently):
      - Walk rows in chronological order.
      - If not in position and long[i] == 1, enter at that row's price.
      - While in position, exit when ANY of:
          (a) time elapsed >= max_horizon hours
          (b) current return <= -stoploss
          (c) short[i] == -1 AND time elapsed >= min_horizon
      - No double-long: additional +1 signals while in position are ignored.

    Parameters
    ----------
    long : array-like (n,) – values in {+1, 0}
    short : array-like (n,) – values in {-1, 0}
    O : pd.DataFrame – must contain 'ticker', 'price', 'date', 'hoursSinceMidnight'
    stoploss : float – positive magnitude, exit when return <= -stoploss
    min_horizon : int – hours; short signals before this don't close the long
    max_horizon : int – hours; hard exit after this much time

    Returns
    -------
    pd.DataFrame with columns:
        ticker, startTime, endTime, priceStart, priceEnd, return, date
    """
    long = np.asarray(long, dtype=float)
    short = np.asarray(short, dtype=float)

    # reconstruct ts from date + hoursSinceMidnight (O index is RangeIndex)
    date = O['date'].astype(int).astype(str).values
    hour = O['hoursSinceMidnight'].astype(int).values
    ts = pd.to_datetime(date, format='%Y%m%d', utc=True) + pd.to_timedelta(hour, unit='h')

    work = pd.DataFrame({
        'ticker': O['ticker'].values,
        'ts': ts,
        'date': O['date'].values,
        'price': O['price'].values,
        'long': long,
        'short': short,
    }).reset_index(drop=True)

    trades = []
    for ticker, grp in work.groupby('ticker'):
        grp = grp.sort_values('ts').reset_index(drop=True)
        ts_arr = grp['ts'].values
        price = grp['price'].values
        l = grp['long'].values
        s = grp['short'].values
        n = len(grp)

        in_pos = False
        entry_i = -1
        entry_ts = None
        entry_price = None

        for i in range(n):
            if not in_pos:
                if l[i] == 1 and not np.isnan(price[i]):
                    in_pos = True
                    entry_i = i
                    entry_ts = ts_arr[i]
                    entry_price = price[i]
            else:
                # elapsed hours since entry
                elapsed_h = (ts_arr[i] - entry_ts) / np.timedelta64(1, 'h')
                if np.isnan(price[i]) or entry_price == 0:
                    continue
                cur_ret = (price[i] - entry_price) / entry_price

                exit_now = False
                if elapsed_h >= max_horizon:
                    exit_now = True
                elif cur_ret <= -stoploss:
                    exit_now = True
                elif s[i] == -1 and elapsed_h >= min_horizon:
                    exit_now = True

                if exit_now:
                    trades.append({
                        'ticker': ticker,
                        'startTime': pd.Timestamp(entry_ts),
                        'endTime': pd.Timestamp(ts_arr[i]),
                        'priceStart': entry_price,
                        'priceEnd': price[i],
                        'return': cur_ret,
                        'date': int(grp['date'].iloc[entry_i]),
                    })
                    in_pos = False
                    entry_i = -1
                    entry_ts = None
                    entry_price = None

        # if still in position at end of data, close at last available price
        if in_pos and entry_price is not None:
            last_price = price[-1]
            if not np.isnan(last_price):
                trades.append({
                    'ticker': ticker,
                    'startTime': pd.Timestamp(entry_ts),
                    'endTime': pd.Timestamp(ts_arr[-1]),
                    'priceStart': entry_price,
                    'priceEnd': last_price,
                    'return': (last_price - entry_price) / entry_price,
                    'date': int(grp['date'].iloc[entry_i]),
                })

    result = pd.DataFrame(trades, columns=[
        'ticker', 'startTime', 'endTime', 'priceStart', 'priceEnd', 'return', 'date'
    ])
    if len(result) > 0:
        result['duration'] = (result['endTime'] - result['startTime']) / pd.Timedelta(hours=1)
    else:
        result['duration'] = []
    return result


def showTrade(dl, out, ticker, long_pred=None, short_pred=None, O=None):
    """Plot Binance-futures 1h price + trade markers, optionally with pred panels.

    Parameters
    ----------
    dl : DataLoader
    out : pd.DataFrame – output of long_short_tradeReturn
    ticker : str
    long_pred, short_pred : array-like or None – predictions aligned with O's rows
    O : pd.DataFrame or None – must contain 'ticker', 'date', 'hoursSinceMidnight';
        required when long_pred/short_pred are passed
    """
    import matplotlib.pyplot as plt

    raw = dl.get(ticker, 'binance', 'futures', barFreqInHours=1)
    price = raw['close']
    trades = out[out['ticker'] == ticker].copy()

    has_preds = (long_pred is not None) and (short_pred is not None) and (O is not None)

    if has_preds:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                                 gridspec_kw={'height_ratios': [3, 1, 1]},
                                 sharex=True)
        ax_price, ax_long, ax_short = axes
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=(14, 5))

    ax_price.plot(price.index, price.values, color='steelblue', linewidth=0.8, label='price (1h)')

    if len(trades) > 0:
        ax_price.scatter(trades['startTime'], trades['priceStart'],
                         marker='^', color='red', s=80, zorder=5, label='open long')
        ax_price.scatter(trades['endTime'], trades['priceEnd'],
                         marker='v', color='green', s=80, zorder=5, label='close')

    ax_price.set_ylabel('price')
    ax_price.set_title(f'{ticker} — binance futures 1h ({len(trades)} trades)')
    ax_price.legend(fontsize=8)
    ax_price.grid(True, alpha=0.3)

    if has_preds:
        lp_df = getTickerPred(long_pred, O, ticker)
        sp_df = getTickerPred(short_pred, O, ticker)

        ax_long.plot(lp_df['ts'], lp_df['pred'], color='red', linewidth=0.8)
        ax_long.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax_long.set_ylabel('long_pred')
        ax_long.grid(True, alpha=0.3)

        ax_short.plot(sp_df['ts'], sp_df['pred'], color='green', linewidth=0.8)
        ax_short.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax_short.set_ylabel('short_pred')
        ax_short.set_xlabel('time')
        ax_short.grid(True, alpha=0.3)
    else:
        ax_price.set_xlabel('time')

    plt.tight_layout()


def getTickerPred(pred, O, ticker):
    """Return a DataFrame of predictions for a single ticker, sorted by time.

    Parameters
    ----------
    pred : array-like (n,) – prediction aligned with O's rows
    O : pd.DataFrame – must contain 'ticker', 'date', 'hoursSinceMidnight'
    ticker : str

    Returns
    -------
    pd.DataFrame with columns [ts, date, hoursSinceMidnight, pred]
        sorted by ts, filtered to the given ticker
    """
    pred = np.asarray(pred, dtype=float)
    mask = (O['ticker'].values == ticker)
    date = O.loc[mask, 'date'].astype(int).astype(str).values
    hour = O.loc[mask, 'hoursSinceMidnight'].astype(int).values
    ts = pd.to_datetime(date, format='%Y%m%d', utc=True) + pd.to_timedelta(hour, unit='h')

    df = pd.DataFrame({
        'ts': ts,
        'date': O.loc[mask, 'date'].values,
        'hoursSinceMidnight': O.loc[mask, 'hoursSinceMidnight'].values,
        'pred': pred[mask],
    }).sort_values('ts').reset_index(drop=True)
    return df
