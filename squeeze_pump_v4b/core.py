"""squeeze_pump_v4b — production core.

Single file: config + loader + features + filter + model + positions +
data-freshness checks + the per-tick orchestrator. Used by both
`run_live.py` (production) and `backtest.py` (historical replay) via
`from core import ...` since they sit next to this file.

Live data layout (per user spec):
    LIVE_DATA_PATH/
        {SYMBOL}_{TAG}/
            YYYY-MM-DD.csv      # closed UTC days, hourly bars
            latest.csv          # current partial UTC day

CSV columns (extras like high/low pass through):
    ts, close, volume_usdt, cvd_usdt,
    open_interest, funding_rate, funding_interval_hours
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

log = logging.getLogger("squeeze_pump_v4b")


# ════════════════════════════════════════════════════════════════════════
# Feature catalog (training-derived — do not edit unless you retrain).
# ════════════════════════════════════════════════════════════════════════
_NON_BTC_FEATURES: list[str] = [
    "spot_perp_vol_ratio", "spot_price_vs_perp",
    "log_oi_level", "market_cap_proxy", "oi_mcap_ratio",
    "oi_3d_chg_pct", "oi_coin_3d_chg_pct",
    "funding",
    "ret_1d_pct", "ret_3d_pct",
    "ret_4h_pct", "cvd_chg_4h", "oi_chg_4h_pct",
    "oi_coin_chg_4h_pct", "vol_chg_4h_pct",
    "ret_1h_pct",
    "oi_coin_1h_chg_pct", "oi_coin_24h_chg_pct",
    "vol_1h_pct", "vol_24h_pct",
    "perp_cvd_1h", "perp_cvd_24h",
    "spot_cvd_1h", "spot_cvd_4h", "spot_cvd_24h",
]
_BTC_FEATURES: list[str] = [
    "btc_ret_1h", "btc_ret_4h", "btc_ret_24h", "btc_ret_3d",
]
# Filter columns are computed but not fed to the model.
_FILTER_COLS: list[str] = [
    "oi_coin_3d_chg_pct", "oi_coin_7d_chg_pct",
    "total_cvd_7d", "ret_3d_pct",
]


# ════════════════════════════════════════════════════════════════════════
# Config (YAML).
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # paths
    live_data_path: Path
    model_path: Path
    state_path: Path
    log_path: Path
    # universe
    primary_tag: str
    spot_tags: list[str]
    perp_tags: list[str]
    known_tags: list[str]
    # btc
    include_btc_features: bool
    btc_symbol: str
    btc_source_tag: str
    # thresholds
    entry_threshold: float
    trail_pct: float
    stop_loss_pct: float
    max_hold_hours: int
    # hard filter (3-day only)
    oi_coin_3d_chg_pct_min: float
    ret_3d_pct_min: float
    # lookback
    lookback_hours: int
    required_history_days: int
    # freshness
    recheck_interval_sec: int
    max_wait_min: int
    probe_symbols: list[str]
    probe_tag: str
    # optional / defaulted last (Python dataclass rule)
    oi_coin_7d_chg_pct_min: float | None = None   # legacy, unused
    total_cvd_7d_min: float | None = None         # legacy, unused
    features: list[str] = field(default_factory=list)


def load_config(path: str | Path) -> Config:
    """Read YAML config. Relative paths inside resolve to the YAML's folder."""
    p = Path(path).expanduser().resolve()
    raw = yaml.safe_load(p.read_text())
    base = p.parent

    def _resolve(s: str) -> Path:
        q = Path(s)
        return q if q.is_absolute() else (base / q).resolve()

    hf = raw["hard_filter"]
    fr = raw["freshness"]
    cfg = Config(
        live_data_path=_resolve(raw["live_data_path"]),
        model_path=_resolve(raw["model_path"]),
        state_path=_resolve(raw["state_path"]),
        log_path=_resolve(raw["log_path"]),
        primary_tag=raw["primary_tag"],
        spot_tags=list(raw["spot_tags"]),
        perp_tags=list(raw["perp_tags"]),
        known_tags=list(raw["known_tags"]),
        include_btc_features=bool(raw["include_btc_features"]),
        btc_symbol=raw["btc_symbol"],
        btc_source_tag=raw["btc_source_tag"],
        entry_threshold=float(raw["entry_threshold"]),
        trail_pct=float(raw["trail_pct"]),
        stop_loss_pct=float(raw["stop_loss_pct"]),
        max_hold_hours=int(raw["max_hold_hours"]),
        oi_coin_3d_chg_pct_min=float(hf["oi_coin_3d_chg_pct_min"]),
        ret_3d_pct_min=float(hf["ret_3d_pct_min"]),
        oi_coin_7d_chg_pct_min=(float(hf["oi_coin_7d_chg_pct_min"])
                                if "oi_coin_7d_chg_pct_min" in hf else None),
        total_cvd_7d_min=(float(hf["total_cvd_7d_min"])
                          if "total_cvd_7d_min" in hf else None),
        lookback_hours=int(raw["lookback_hours"]),
        required_history_days=int(raw["required_history_days"]),
        recheck_interval_sec=int(fr["recheck_interval_sec"]),
        max_wait_min=int(fr["max_wait_min"]),
        probe_symbols=list(fr["probe_symbols"]),
        probe_tag=fr["probe_tag"],
    )
    cfg.features = _NON_BTC_FEATURES + (
        _BTC_FEATURES if cfg.include_btc_features else []
    )
    return cfg


# ════════════════════════════════════════════════════════════════════════
# CSV loader.
# ════════════════════════════════════════════════════════════════════════
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
_REQUIRED_COLS = ["ts", "close", "volume_usdt", "cvd_usdt",
                  "open_interest", "funding_rate", "funding_interval_hours"]
_OPTIONAL_COLS = ["high", "low"]


def _normalize_symbol(sym: str, tag: str) -> str:
    s = sym.upper()
    if "okex" in tag.lower():
        s = s.replace("-USDT-SWAP", "USDT").replace("-USDT", "USDT")
    return s


def _read_one_csv(path: Path) -> pd.DataFrame:
    # float_precision='round_trip' is required to bit-exactly recover
    # the original float64 values written by convert_data.py with
    # float_format='%.17g'. The default C parser loses ~1 ULP.
    df = pd.read_csv(path, float_precision="round_trip")
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in _REQUIRED_COLS[1:]:
        if c not in df.columns:
            df[c] = np.nan
    keep = _REQUIRED_COLS + [c for c in _OPTIONAL_COLS if c in df.columns]
    return df[keep]


class LiveLoader:
    def __init__(self, root: str | Path, known_tags: list[str]):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data folder not found: {self.root}")
        self.known_tags = list(known_tags)

    def _folder(self, symbol: str, tag: str) -> Path:
        return self.root / f"{symbol}_{tag}"

    def list_symbols(self, tag: str) -> list[str]:
        suffix = f"_{tag}"
        out: set[str] = set()
        for p in self.root.iterdir():
            if p.is_dir() and p.name.endswith(suffix):
                raw = p.name[: -len(suffix)]
                out.add(_normalize_symbol(raw, tag))
        return sorted(out)

    def list_tags(self, symbol: str) -> list[str]:
        prefix = f"{symbol}_"
        out: list[str] = []
        for p in self.root.iterdir():
            if p.is_dir() and p.name.startswith(prefix):
                tag = p.name[len(prefix):]
                if tag in self.known_tags:
                    out.append(tag)
        return sorted(out)

    def load(self, symbol: str, tag: str, now_ts: pd.Timestamp,
             lookback_hours: int) -> pd.DataFrame:
        folder = self._folder(symbol, tag)
        if not folder.is_dir():
            return pd.DataFrame(columns=_REQUIRED_COLS)

        now_ts = _ensure_utc(now_ts)
        start_ts = now_ts - pd.Timedelta(hours=lookback_hours)
        d_lo, d_hi = start_ts.normalize().date(), now_ts.normalize().date()

        frames: list[pd.DataFrame] = []
        for entry in folder.iterdir():
            if not entry.is_file():
                continue
            m = _DATE_RE.match(entry.name)
            if not m:
                continue
            d = pd.Timestamp(m.group(1)).date()
            if d_lo <= d <= d_hi:
                frames.append(_read_one_csv(entry))

        latest = folder / "latest.csv"
        if latest.is_file():
            frames.append(_read_one_csv(latest))

        if not frames:
            return pd.DataFrame(columns=_REQUIRED_COLS)

        df = pd.concat(frames, ignore_index=True)
        df = (df.dropna(subset=["ts"])
                .sort_values("ts")
                .drop_duplicates(subset=["ts"], keep="last")
                .reset_index(drop=True))
        df = df[(df["ts"] >= start_ts) & (df["ts"] <= now_ts)]
        return df.reset_index(drop=True)

    def load_bundle(self, symbol: str, now_ts: pd.Timestamp,
                    lookback_hours: int) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for tag in self.list_tags(symbol):
            df = self.load(symbol, tag, now_ts, lookback_hours)
            if not df.empty:
                out[tag] = df
        return out


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


class CachedLoader:
    """Drop-in replacement for LiveLoader optimized for backtest.

    Loads every (symbol, tag) folder under `root` once at construction and
    serves load() / load_bundle() from in-memory slices. ~2 orders of
    magnitude faster than LiveLoader for repeated ticks on the same data.
    Used only by `backtest.py` mode='default'. run_live and replay still
    use LiveLoader so they exercise the production CSV-reading path.
    """

    def __init__(self, root: str | Path, known_tags: list[str],
                 verbose: bool = True):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data folder not found: {self.root}")
        self.known_tags = list(known_tags)
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._tags_by_sym: dict[str, list[str]] = {}
        self._syms_by_tag: dict[str, list[str]] = {}
        if verbose:
            log.info("CachedLoader: pre-loading %s ...", self.root)
        n_folders = n_rows = 0
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            matched = next((t for t in self.known_tags
                            if sub.name.endswith("_" + t)), None)
            if matched is None:
                continue
            raw = sub.name[: -(len(matched) + 1)]
            sym = _normalize_symbol(raw, matched)
            frames: list[pd.DataFrame] = []
            for f in sub.iterdir():
                if not f.is_file():
                    continue
                if not (_DATE_RE.match(f.name) or f.name == "latest.csv"):
                    continue
                df = _read_one_csv(f)
                if not df.empty:
                    frames.append(df)
            if not frames:
                continue
            df = (pd.concat(frames, ignore_index=True)
                    .dropna(subset=["ts"]).sort_values("ts")
                    .drop_duplicates(subset=["ts"], keep="last")
                    .reset_index(drop=True))
            self._cache[(sym, matched)] = df
            self._tags_by_sym.setdefault(sym, []).append(matched)
            self._syms_by_tag.setdefault(matched, []).append(sym)
            n_folders += 1
            n_rows += len(df)
        for sym in self._tags_by_sym:
            self._tags_by_sym[sym] = sorted(set(self._tags_by_sym[sym]))
        for tag in self._syms_by_tag:
            self._syms_by_tag[tag] = sorted(set(self._syms_by_tag[tag]))
        if verbose:
            log.info("CachedLoader: %d (symbol, tag) frames, %d rows total",
                     n_folders, n_rows)

    def list_symbols(self, tag: str) -> list[str]:
        return list(self._syms_by_tag.get(tag, []))

    def list_tags(self, symbol: str) -> list[str]:
        return list(self._tags_by_sym.get(symbol, []))

    def load(self, symbol: str, tag: str, now_ts: pd.Timestamp,
             lookback_hours: int) -> pd.DataFrame:
        df = self._cache.get((symbol, tag))
        if df is None:
            return pd.DataFrame(columns=_REQUIRED_COLS)
        now_ts = _ensure_utc(now_ts)
        start_ts = now_ts - pd.Timedelta(hours=lookback_hours)
        m = (df["ts"] >= start_ts) & (df["ts"] <= now_ts)
        return df.loc[m].reset_index(drop=True)

    def load_bundle(self, symbol: str, now_ts: pd.Timestamp,
                    lookback_hours: int) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for tag in self._tags_by_sym.get(symbol, []):
            sub = self.load(symbol, tag, now_ts, lookback_hours)
            if not sub.empty:
                out[tag] = sub
        return out


# ════════════════════════════════════════════════════════════════════════
# Feature engineering — verbatim from squeeze_pump_detection_v4b.ipynb.
# Computes 1h frame, then subsamples to the 4h grid.
# ════════════════════════════════════════════════════════════════════════
def compute_features(bundle: dict[str, pd.DataFrame],
                     btc_bars: pd.DataFrame | None,
                     symbol: str,
                     cfg: Config) -> pd.DataFrame:
    if cfg.primary_tag not in bundle or bundle[cfg.primary_tag].empty:
        return pd.DataFrame(
            columns=["ts", "symbol", "close"] + cfg.features + _FILTER_COLS)

    g = bundle[cfg.primary_tag].copy()
    g["ts"] = pd.to_datetime(g["ts"], utc=True)
    g = (g.sort_values("ts")
          .drop_duplicates(subset=["ts"], keep="last")
          .set_index("ts"))

    # Primary perp features.
    g["funding"] = g["funding_rate"]
    g["oi_3d_chg_pct"] = g["open_interest"].pct_change(3 * 24) * 100
    g["oi_7d_chg_pct"] = g["open_interest"].pct_change(7 * 24) * 100
    oi_coins = g["open_interest"] / g["close"].replace(0, np.nan)
    g["oi_coin_3d_chg_pct"] = oi_coins.pct_change(3 * 24) * 100
    g["oi_coin_7d_chg_pct"] = oi_coins.pct_change(7 * 24) * 100
    g["log_oi_level"] = np.log1p(g["open_interest"])
    g["market_cap_proxy"] = g["close"] * g["open_interest"]
    g["oi_mcap_ratio"] = g["open_interest"] / g["market_cap_proxy"].replace(0, np.nan)
    g["ret_1d_pct"] = g["close"].pct_change(24) * 100
    g["ret_3d_pct"] = g["close"].pct_change(3 * 24) * 100
    g["ret_1h_pct"] = g["close"].pct_change(1) * 100
    g["oi_coin_1h_chg_pct"] = oi_coins.pct_change(1) * 100
    g["oi_coin_24h_chg_pct"] = oi_coins.pct_change(24) * 100
    g["vol_1h_pct"] = g["volume_usdt"].pct_change(1) * 100
    g["vol_24h_pct"] = g["volume_usdt"].pct_change(24) * 100
    g["perp_cvd_1h"] = g["cvd_usdt"].diff(1)
    g["perp_cvd_24h"] = g["cvd_usdt"].diff(24)
    g["ret_4h_pct"] = g["close"].pct_change(4) * 100
    g["cvd_chg_4h"] = g["cvd_usdt"].diff(4)
    g["oi_chg_4h_pct"] = g["open_interest"].pct_change(4) * 100
    g["oi_coin_chg_4h_pct"] = oi_coins.pct_change(4) * 100
    vol_4h = g["volume_usdt"].rolling(4).sum()
    vol_4h_prev = g["volume_usdt"].shift(4).rolling(4).sum()
    g["vol_chg_4h_pct"] = (vol_4h / vol_4h_prev - 1) * 100
    g["perp_cvd_7d"] = g["cvd_usdt"].diff(7 * 24)

    # BTC context (optional).
    if cfg.include_btc_features:
        if btc_bars is not None and not btc_bars.empty:
            btc = btc_bars.copy()
            btc["ts"] = pd.to_datetime(btc["ts"], utc=True)
            btc = (btc.sort_values("ts")
                      .drop_duplicates(subset=["ts"], keep="last")
                      .set_index("ts"))["close"]
            ba = btc.reindex(g.index)
            g["btc_ret_1h"]  = ba.pct_change(1)  * 100
            g["btc_ret_4h"]  = ba.pct_change(4)  * 100
            g["btc_ret_24h"] = ba.pct_change(24) * 100
            g["btc_ret_3d"]  = ba.pct_change(72) * 100
        else:
            for c in _BTC_FEATURES:
                g[c] = np.nan

    # Binance spot features.
    bs = bundle.get("binance_spot")
    if bs is not None and not bs.empty:
        bs = bs.copy()
        bs["ts"] = pd.to_datetime(bs["ts"], utc=True)
        bs = (bs.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
                .set_index("ts"))
        spot_cvd = bs["cvd_usdt"].reindex(g.index)
        spot_vol = bs["volume_usdt"].reindex(g.index)
        spot_close = bs["close"].reindex(g.index)
        g["spot_cvd_7d"] = spot_cvd.diff(7 * 24)
        g["spot_cvd_1h"] = spot_cvd.diff(1)
        g["spot_cvd_4h"] = spot_cvd.diff(4)
        g["spot_cvd_24h"] = spot_cvd.diff(24)
        spot_vol_24h = spot_vol.rolling(24, min_periods=12).sum()
        perp_vol_24h = g["volume_usdt"].rolling(24, min_periods=12).sum()
        g["spot_perp_vol_ratio"] = spot_vol_24h / perp_vol_24h.replace(0, np.nan)
        g["spot_price_vs_perp"] = (spot_close - g["close"]) / g["close"].replace(0, np.nan)
    else:
        for c in ["spot_cvd_7d", "spot_cvd_1h", "spot_cvd_4h", "spot_cvd_24h",
                  "spot_perp_vol_ratio", "spot_price_vs_perp"]:
            g[c] = np.nan

    # Multi-exchange spot CVD 7d sum.
    multi = pd.Series(0.0, index=g.index)
    has_any_spot = False
    for tag in cfg.spot_tags:
        df = bundle.get(tag)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = (df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
                .set_index("ts"))
        chg = df["cvd_usdt"].reindex(g.index).diff(7 * 24)
        multi = multi.add(chg, fill_value=0.0)
        has_any_spot = True
    g["multi_exchange_spot_cvd_7d"] = multi if has_any_spot else np.nan

    total_cvd = g["perp_cvd_7d"].fillna(0)
    if has_any_spot:
        total_cvd = total_cvd + multi.fillna(0)
    g["total_cvd_7d"] = total_cvd

    # Hourly grid throughout — no 4h downsample. Aggregations are still
    # 4h rolling (vol_4h, ret_4h_pct, etc.), but every hour gets a row.
    g = g.reset_index()
    g["symbol"] = symbol

    keep: list[str] = []
    for c in ["ts", "symbol", "close"] + cfg.features + _FILTER_COLS:
        if c not in keep:
            keep.append(c)
    for c in keep:
        if c not in g.columns:
            g[c] = np.nan
    return g[keep].replace([np.inf, -np.inf], np.nan)


# ════════════════════════════════════════════════════════════════════════
# Hard filter.
# ════════════════════════════════════════════════════════════════════════
def passes_hard_filter(row: pd.Series, cfg: Config) -> bool:
    """3-day-only filter. 7-day columns were dropped — features all use ≤ 3
    days lookback, so the production pipeline only needs 4 days of bars."""
    return bool(
        row["oi_coin_3d_chg_pct"] > cfg.oi_coin_3d_chg_pct_min
        and row["ret_3d_pct"] > cfg.ret_3d_pct_min
    )


# ════════════════════════════════════════════════════════════════════════
# XGBoost model wrapper.
# ════════════════════════════════════════════════════════════════════════
class PumpModel:
    def __init__(self, model_path: str | Path):
        p = Path(model_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"model not found: {p}")
        self.path = p
        if p.suffix in (".json", ".ubj", ".bin"):
            self._clf = xgb.XGBClassifier()
            self._clf.load_model(str(p))
            self._is_sklearn = True
            self._booster = None
        else:
            import pickle
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, xgb.XGBClassifier):
                self._clf, self._booster = obj, None
                self._is_sklearn = True
            elif isinstance(obj, xgb.Booster):
                self._clf, self._booster = None, obj
                self._is_sklearn = False
            else:
                raise TypeError(f"unsupported model type: {type(obj)}")

    def predict_proba(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        X = df.reindex(columns=features).astype(float).fillna(0).values
        if self._is_sklearn:
            return self._clf.predict_proba(X)[:, 1]
        return self._booster.predict(xgb.DMatrix(X, feature_names=features))


# ════════════════════════════════════════════════════════════════════════
# Position registry + exit rules.
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Position:
    symbol: str
    tag: str
    entry_ts: str
    entry_close: float
    peak: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_state(path: Path) -> tuple[list[Position], dict[str, str]]:
    """Returns (positions, last_buys). last_buys[sym] = ISO ts of most
    recent BUY signal for that symbol, used for the 4h re-entry cooldown.
    Backward-compat: if the file is a bare list, treat it as legacy
    positions-only with empty last_buys."""
    if not path.is_file():
        return [], {}
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return [Position(**r) for r in raw], {}
    return ([Position(**r) for r in raw.get("positions", [])],
            dict(raw.get("last_buys", {})))


def save_state(positions: list[Position], last_buys: dict[str, str],
               path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"positions": [p.to_json() for p in positions],
                       "last_buys": last_buys}, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def open_position(positions: list[Position], symbol: str, tag: str,
                  ts: pd.Timestamp, close: float) -> Position:
    pos = Position(symbol=symbol, tag=tag,
                   entry_ts=_ensure_utc(ts).isoformat(),
                   entry_close=float(close), peak=float(close))
    positions.append(pos)
    return pos


def _decide_exit(pos: Position, bars: pd.DataFrame, now_ts: pd.Timestamp,
                 cfg: Config):
    """Returns (reason, exit_ts, exit_close, peak) or None. Mutates peak only."""
    entry_ts = _ensure_utc(pd.Timestamp(pos.entry_ts))
    fwd = bars[(bars["ts"] > entry_ts) & (bars["ts"] <= now_ts)].reset_index(drop=True)
    peak = pos.peak
    stop_price = pos.entry_close * (1.0 - cfg.stop_loss_pct)
    time_limit = entry_ts + pd.Timedelta(hours=cfg.max_hold_hours)
    for _, bar in fwd.iterrows():
        p = float(bar["close"])
        ts = pd.Timestamp(bar["ts"])
        peak = max(peak, p)
        if p <= stop_price:
            return "stop_loss", ts, p, peak
        if p <= peak * (1.0 - cfg.trail_pct):
            return "trail_stop", ts, p, peak
        if ts >= time_limit:
            return "time_exit", ts, p, peak
    pos.peak = peak  # advance peak even if no exit
    return None


def tick_exits(positions: list[Position], loader: LiveLoader,
               now_ts: pd.Timestamp, cfg: Config
               ) -> tuple[list[Position], list[dict]]:
    still_open: list[Position] = []
    signals: list[dict] = []
    for pos in positions:
        bars = loader.load(pos.symbol, pos.tag, now_ts, cfg.lookback_hours)
        if bars.empty:
            log.warning("no bars for open position %s/%s — keeping open",
                        pos.symbol, pos.tag)
            still_open.append(pos)
            continue
        # Freshness: latest bar must be at or after now_ts. If stale we
        # keep the position open (will retry next tick) — never blind-exit.
        if bars["ts"].max() < now_ts:
            log.warning("stale data for open position %s/%s (last %s, now %s) — "
                        "keeping open", pos.symbol, pos.tag,
                        bars["ts"].max(), now_ts)
            still_open.append(pos)
            continue
        res = _decide_exit(pos, bars, now_ts, cfg)
        if res is None:
            still_open.append(pos)
            continue
        reason, exit_ts, exit_close, peak = res
        pos.peak = peak
        signals.append({
            "signal": -1, "side": "SELL",
            "symbol": pos.symbol, "tag": pos.tag,
            "entry_ts": pos.entry_ts, "entry_close": pos.entry_close,
            "exit_ts": exit_ts.isoformat(), "exit_close": exit_close,
            "exit_ret": exit_close / pos.entry_close - 1.0,
            "exit_reason": reason, "data_status": "fresh",
        })
    return still_open, signals


# ════════════════════════════════════════════════════════════════════════
# Data freshness helpers.
# ════════════════════════════════════════════════════════════════════════
def has_required_history(df: pd.DataFrame, now_ts: pd.Timestamp,
                         required_days: int) -> bool:
    if df.empty:
        return False
    needed = _ensure_utc(now_ts) - pd.Timedelta(days=required_days)
    return df["ts"].min() <= needed


def is_symbol_fresh(loader: LiveLoader, symbol: str, tag: str,
                    now_ts: pd.Timestamp, lookback_hours: int) -> bool:
    df = loader.load(symbol, tag, now_ts, lookback_hours=2)
    return (not df.empty) and df["ts"].max() >= _ensure_utc(now_ts)


def freshness_probe(loader: LiveLoader, now_ts: pd.Timestamp,
                    cfg: Config) -> tuple[bool, list[str]]:
    """Returns (all_fresh, missing). missing = probe symbols not yet at now_ts."""
    missing: list[str] = []
    for sym in cfg.probe_symbols:
        if not is_symbol_fresh(loader, sym, cfg.probe_tag,
                               now_ts, cfg.lookback_hours):
            missing.append(f"{sym}_{cfg.probe_tag}")
    return (len(missing) == 0), missing


def wait_for_fresh_data(loader: LiveLoader, now_ts: pd.Timestamp,
                        cfg: Config) -> tuple[bool, list[str]]:
    """Poll every `recheck_interval_sec` until probes are fresh OR
    `max_wait_min` elapses. Returns (ok, missing_probes_at_timeout)."""
    deadline = time.time() + cfg.max_wait_min * 60
    while True:
        ok, missing = freshness_probe(loader, now_ts, cfg)
        if ok:
            return True, []
        if time.time() >= deadline:
            return False, missing
        log.warning("data not fresh yet at %s (missing=%s) — retry in %ds",
                    now_ts.isoformat(), missing, cfg.recheck_interval_sec)
        time.sleep(cfg.recheck_interval_sec)


# ════════════════════════════════════════════════════════════════════════
# Per-tick orchestrator. Pure: takes positions in, returns positions + signals.
# ════════════════════════════════════════════════════════════════════════
def is_entry_hour(ts: pd.Timestamp) -> bool:
    """Kept for back-compat — strategy now evaluates every hour."""
    return True


_COOLDOWN = pd.Timedelta(hours=4)   # min gap between two BUYs on the same symbol


def run_tick(now_ts: pd.Timestamp, loader: LiveLoader, model: PumpModel,
             positions: list[Position], last_buys: dict[str, str],
             cfg: Config, data_status: str = "fresh") -> dict:
    """One tick of the engine. Pure — caller-owned state:
        positions  : list of currently-open Position objects (mutated in place).
        last_buys  : symbol -> ISO ts of most recent BUY signal (mutated in place).
                     Used for the 4 h re-entry cooldown.
    """
    now_ts = _ensure_utc(now_ts)

    # ── Exit pass (every hour) ────────────────────────────────────────
    # tick_exits returns a NEW list of survivors. We mutate `positions`
    # in place so the caller's reference stays current across ticks.
    still_open, close_signals = tick_exits(positions, loader, now_ts, cfg)
    positions[:] = still_open

    # ── Entry pass (every hour) ───────────────────────────────────────
    open_signals: list[dict] = []
    skipped: list[dict] = []
    btc = None
    if cfg.include_btc_features:
        btc = loader.load(cfg.btc_symbol, cfg.btc_source_tag,
                          now_ts, cfg.lookback_hours)
        if not has_required_history(btc, now_ts, cfg.required_history_days):
            log.warning("BTC history < %d days — entry pass skipped",
                        cfg.required_history_days)
            btc = pd.DataFrame()  # forces NaN BTC features → filter rejects
    held = {(p.symbol, p.tag) for p in positions}

    for sym in loader.list_symbols(cfg.primary_tag):
        if (sym, cfg.primary_tag) in held:
            continue
        # 4 h cooldown: even if a prior long has already closed, no
        # re-entry until 4 h after the most recent BUY for this symbol.
        prior = last_buys.get(sym)
        if prior is not None:
            prior_ts = pd.Timestamp(prior)
            if prior_ts.tzinfo is None:
                prior_ts = prior_ts.tz_localize("UTC")
            if now_ts - prior_ts < _COOLDOWN:
                continue

        bundle = loader.load_bundle(sym, now_ts, cfg.lookback_hours)
        primary = bundle.get(cfg.primary_tag, pd.DataFrame())

        # Per-symbol freshness + history checks.
        if primary.empty:
            skipped.append({"symbol": sym, "reason": "no_data"})
            continue
        if primary["ts"].max() < now_ts:
            skipped.append({"symbol": sym, "reason": "stale",
                            "last_ts": primary["ts"].max().isoformat()})
            continue
        if not has_required_history(primary, now_ts, cfg.required_history_days):
            skipped.append({"symbol": sym, "reason": "insufficient_history",
                            "first_ts": primary["ts"].min().isoformat()})
            continue

        feats = compute_features(bundle, btc, sym, cfg)
        latest = feats[feats["ts"] == now_ts]
        if latest.empty:
            skipped.append({"symbol": sym, "reason": "no_row_at_now"})
            continue
        row = latest.iloc[-1]
        if not passes_hard_filter(row, cfg):
            continue

        prob = float(model.predict_proba(latest, cfg.features)[-1])
        if prob >= cfg.entry_threshold:
            pos = open_position(positions, sym, cfg.primary_tag,
                                row["ts"], row["close"])
            last_buys[sym] = pos.entry_ts
            open_signals.append({
                "signal": 1, "side": "BUY",
                "symbol": sym, "tag": cfg.primary_tag,
                "entry_ts": pos.entry_ts, "entry_close": pos.entry_close,
                "pred_xgb": prob, "data_status": data_status,
            })

    return {
        "now": now_ts.isoformat(),
        "data_status": data_status,
        "open": open_signals,
        "close": close_signals,
        "skipped": skipped,
        "open_positions": len(positions),
    }


# ════════════════════════════════════════════════════════════════════════
# Per-TICK orchestrator — shared by run_live.py and backtest.py(replay).
#
# Both entrypoints call this for every UTC integer hour. They differ only
# in:
#   1. how the LiveLoader's data root is laid out
#      (production: today→latest.csv; replay: shadow that mirrors that),
#   2. how the result is written downstream (JSONL vs CSV),
#   3. wall-clock pacing (live sleeps to :00, replay walks ts in a loop).
# Everything between data load and signal emission is shared.
# ════════════════════════════════════════════════════════════════════════
def process_tick(loader: "LiveLoader", model: "PumpModel", cfg: Config,
                 now_ts: pd.Timestamp,
                 strict_on_timeout: bool = False) -> dict:
    """One full tick: freshness probe → load positions → run_tick →
    persist positions. Returns the run_tick result enriched with a
    `data_status` field (fresh|timeout). On timeout + strict mode,
    skips run_tick entirely and returns a sentinel."""
    now_ts = _ensure_utc(now_ts)
    ok, missing = wait_for_fresh_data(loader, now_ts, cfg)
    data_status = "fresh" if ok else "timeout"
    if not ok and strict_on_timeout:
        return {
            "now": now_ts.isoformat(),
            "data_status": "timeout",
            "missing_probes": missing,
            "open": [], "close": [], "skipped": [],
            "open_positions": 0,
            "strict_skipped": True,
        }
    positions, last_buys = load_state(cfg.state_path)
    result = run_tick(now_ts, loader, model, positions, last_buys, cfg,
                      data_status=data_status)
    save_state(positions, last_buys, cfg.state_path)
    if not ok:
        result["missing_probes"] = missing
    return result


# ════════════════════════════════════════════════════════════════════════
# Misc.
# ════════════════════════════════════════════════════════════════════════
def utc_now_floor_hour() -> pd.Timestamp:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return pd.Timestamp(now)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
