"""Production live runner.

Wakes at every UTC integer hour, polls upstream for fresh data
(every 5s, up to 10min — both knobs are in config.yaml), then runs
one tick of the engine. Emits BUY/SELL signal records to stdout and
to a JSONL log so downstream can consume them.

Usage (run directly, no install):
    python /path/to/squeeze_pump_v4b/run_live.py \
        --config /path/to/config.yaml \
        --loop                                    # forever

    python .../run_live.py --config .../config.yaml --once
    python .../run_live.py --config .../config.yaml --once --now 2026-04-25T16:00:00Z

Data-issue handling (per user spec):
- After the integer hour we poll the probe symbols (default BTCUSDT) every
  `freshness.recheck_interval_sec` (default 5s) for up to
  `freshness.max_wait_min` (default 10min). If probes go fresh in time, we
  proceed with `data_status="fresh"`. Otherwise we emit a
  `data_status="timeout"` record (so downstream knows the signal is
  potentially stale) and STILL run the tick — this protects exit decisions
  on already-open positions. Per-symbol freshness/history checks inside the
  entry pass cause stale/short-history symbols to be skipped (see
  `result["skipped"]`).

If 10 min elapses with no data:
  - Default behaviour: log a `data_status="timeout"` record and continue
    (still run the tick). Stale symbols are skipped from entry decisions
    automatically.
  - If you want a stricter mode (skip the entire tick on timeout) set the
    `--strict-on-timeout` flag below.
  - Operationally, here are the things you might want to do at this point:
      1. Page upstream — the data feeder is down.
      2. Switch the strategy to "monitor-only" (already the default).
      3. If you have a backup data source, point `live_data_path` at it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Run-as-script bootstrap (no install required, works from any cwd).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from core import (LiveLoader, PumpModel, append_jsonl, load_config,
                  process_tick, utc_now_floor_hour)


def _sleep_until_next_hour(buffer_sec: int = 1) -> None:
    """Sleep until N seconds past the next UTC hour. We only need a tiny
    margin to clear the boundary — the per-tick freshness loop polls
    upstream every `recheck_interval_sec` for up to `max_wait_min`, so
    waiting longer here is redundant."""
    now = datetime.now(timezone.utc)
    nxt = (now.replace(minute=0, second=0, microsecond=0)
           + timedelta(hours=1, seconds=buffer_sec))
    delay = (nxt - now).total_seconds()
    if delay > 0:
        time.sleep(delay)


def _one_tick(now_ts: pd.Timestamp, cfg, model: "PumpModel",
              strict_on_timeout: bool) -> dict:
    """Production tick. Pure I/O around the shared `process_tick` core —
    backtest replay calls the same `process_tick` against a shadow folder."""
    log = logging.getLogger("squeeze_pump_v4b")
    loader = LiveLoader(cfg.live_data_path, cfg.known_tags)

    log.info("tick %s — waiting for fresh upstream data", now_ts.isoformat())
    result = process_tick(loader, model, cfg, now_ts,
                          strict_on_timeout=strict_on_timeout)

    if result.get("data_status") == "timeout":
        log.error("DATA TIMEOUT after %d min — probe %s still missing at %s. "
                  "Signals carry data_status='timeout'.",
                  cfg.max_wait_min, result.get("missing_probes"),
                  now_ts.isoformat())

    append_jsonl(cfg.log_path, result)
    log.info("tick %s done — open=%d close=%d skipped=%d total_open=%d (%s)",
             result["now"], len(result.get("open", [])),
             len(result.get("close", [])), len(result.get("skipped", [])),
             result.get("open_positions", 0), result.get("data_status"))
    for s in result.get("close", []) + result.get("open", []):
        print(json.dumps(s))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--now", default=None,
                    help="Replay a specific UTC hour, e.g. '2026-04-25T16:00:00Z'")
    ap.add_argument("--strict-on-timeout", action="store_true",
                    help="Skip the entire tick (incl. exits) on data timeout")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--once", action="store_true")
    grp.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    model = PumpModel(cfg.model_path)   # load once, reuse every tick

    if args.once:
        ts = pd.Timestamp(args.now) if args.now else utc_now_floor_hour()
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        _one_tick(ts, cfg, model, args.strict_on_timeout)
        return

    while True:
        try:
            _one_tick(utc_now_floor_hour(), cfg, model, args.strict_on_timeout)
        except Exception:
            logging.getLogger("squeeze_pump_v4b").exception("tick failed")
        _sleep_until_next_hour()


if __name__ == "__main__":
    main()
