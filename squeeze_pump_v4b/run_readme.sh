#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# squeeze_pump_v4b — example commands.
#
# This file is for reference. Copy/paste the line you want into your
# terminal — it is NOT meant to be executed end-to-end.
#
# All three entry points share config.yaml; only `backtest.mode` differs
# between default vs replay backtests.
# ─────────────────────────────────────────────────────────────────────

# Activate the conda env that has xgboost, pandas, pyyaml.
conda activate ml

# Move into the package folder (so relative paths in config.yaml resolve).
cd squeeze_pump_v4b


# ── 1. Default backtest ──────────────────────────────────────────────
# Fast in-memory walk over historical CSVs. Set `backtest.mode: default`
# in config.yaml. Output → notebooks/bt_signals.csv.
python backtest.py --config config.yaml


# ── 2. Replay backtest ───────────────────────────────────────────────
# Bit-identical mirror of run_live.py (shadow folder with latest.csv
# layout, persisted state). Slower but verifies the live code path.
# Set `backtest.mode: replay` in config.yaml.
# Clear leftover live state first so the replay starts cold:
rm -f state/positions.json log/signals.json
python backtest.py --config config.yaml


# ── 3. Production live — one tick at the current UTC hour ────────────
python run_live.py --config config.yaml --once


# ── 4. Production live — replay one specific UTC hour ────────────────
# Useful for debugging a missed signal at a known timestamp.
python run_live.py --config config.yaml --once --now 2026-04-25T16:00:00Z


# ── 5. Production live — forever loop ────────────────────────────────
# Wakes every UTC hour at :00:01, runs one tick, sleeps. The per-tick
# freshness loop (every `recheck_interval_sec`, up to `max_wait_min`)
# handles upstream lag, so the wake-up margin is just 1s. Use nohup or
# a process supervisor (systemd, launchd, pm2 …) to keep it alive.
python run_live.py --config config.yaml --loop
