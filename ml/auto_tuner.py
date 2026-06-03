"""
Auto-Tuner — ml/auto_tuner.py

Inspired by FreqAI's Hyperopt with Optuna.
Uses Bayesian optimisation to find the best bot parameters
by running mini-backtests and maximising Sharpe ratio.

Parameters tuned:
  - MIN_SIGNAL_SCORE       (how strict to be about signals)
  - ATR_STOP_MULTIPLIER    (how wide the stop loss is)
  - ATR_TARGET_MULTIPLIER  (how far the take profit is)
  - ADX_THRESHOLD          (trend strength filter)
  - VOLUME_SURGE_MULTIPLIER (volume confirmation threshold)

All logic written from scratch using Optuna.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# Silence optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    best_params:    dict
    best_sharpe:    float
    best_return:    float
    n_trials:       int
    improvement:    str     # e.g. "MIN_SIGNAL_SCORE: 5.5 → 6.2"


def _mini_backtest(df: pd.DataFrame,
                   analyse_fn: Callable,
                   symbol: str,
                   params: dict) -> dict:
    """
    Run a lightweight backtest with given parameters.
    Returns metrics: sharpe, total_return, win_rate, profit_factor.

    Designed to be fast — runs entirely in-memory on a pre-fetched DataFrame.
    """
    import config as _cfg

    # Temporarily override config for this trial
    orig = {k: getattr(_cfg, k) for k in params}
    try:
        for k, v in params.items():
            setattr(_cfg, k, v)

        window     = 60
        capital    = 1000.0
        curr_cap   = capital
        in_trade   = False
        entry = stop = target = qty = 0.0
        side  = ""
        pnls  = []

        for i in range(window, len(df) - 1):
            slice_  = df.iloc[:i]
            current = df.iloc[i]
            price   = current["close"]

            # Check exit
            if in_trade:
                hit_sl = (side == "long"  and current["low"]  <= stop) or \
                         (side == "short" and current["high"] >= stop)
                hit_tp = (side == "long"  and current["high"] >= target) or \
                         (side == "short" and current["low"]  <= target)
                if hit_tp or hit_sl:
                    ep = target if hit_tp else stop
                    pnl = (ep - entry) * qty if side == "long" else (entry - ep) * qty
                    curr_cap += pnl
                    pnls.append(pnl)
                    in_trade = False

            # Check entry
            if not in_trade:
                try:
                    sig = analyse_fn(symbol, slice_, include_sentiment=False)
                except Exception:
                    continue

                if (sig.direction in ("long", "short") and
                        sig.score >= _cfg.MIN_SIGNAL_SCORE and
                        sig.atr > 0):

                    atr = sig.atr
                    if sig.direction == "long":
                        s = price - _cfg.ATR_STOP_MULTIPLIER * atr
                        t = price + _cfg.ATR_TARGET_MULTIPLIER * atr
                    else:
                        s = price + _cfg.ATR_STOP_MULTIPLIER * atr
                        t = price - _cfg.ATR_TARGET_MULTIPLIER * atr

                    risk_unit = abs(price - s)
                    if risk_unit > 0:
                        risk_usd = curr_cap * 0.02
                        qty    = risk_usd / risk_unit
                        entry  = price
                        stop   = s
                        target = t
                        side   = sig.direction
                        in_trade = True

        if not pnls:
            return {"sharpe": -999, "total_return": 0, "win_rate": 0, "profit_factor": 0}

        pnl_arr    = np.array(pnls)
        total_ret  = (curr_cap - capital) / capital
        wins       = pnl_arr[pnl_arr > 0]
        losses     = pnl_arr[pnl_arr <= 0]
        win_rate   = len(wins) / len(pnls)
        pf         = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
        sharpe     = (pnl_arr.mean() / (pnl_arr.std() + 1e-8)) * np.sqrt(252) if pnl_arr.std() > 0 else 0

        return {"sharpe": sharpe, "total_return": total_ret,
                "win_rate": win_rate, "profit_factor": pf, "n_trades": len(pnls)}

    finally:
        # Restore original config values
        for k, v in orig.items():
            setattr(_cfg, k, v)


def tune_parameters(df: pd.DataFrame,
                    analyse_fn: Callable,
                    symbol: str,
                    n_trials: int = 50,
                    progress_callback: Callable | None = None) -> TuningResult:
    """
    Use Optuna to find optimal bot parameters for the given symbol and data.

    analyse_fn  : the analyse() function from core.strategies
    n_trials    : number of parameter combinations to try (more = better, slower)
    progress_callback : called with (trial_num, n_trials, best_sharpe) each trial
    """
    import config as _cfg

    # Record starting values for comparison
    start_params = {
        "MIN_SIGNAL_SCORE":       _cfg.MIN_SIGNAL_SCORE,
        "ATR_STOP_MULTIPLIER":    _cfg.ATR_STOP_MULTIPLIER,
        "ATR_TARGET_MULTIPLIER":  _cfg.ATR_TARGET_MULTIPLIER,
        "ADX_THRESHOLD":          _cfg.ADX_THRESHOLD,
        "VOLUME_SURGE_MULTIPLIER":_cfg.VOLUME_SURGE_MULTIPLIER,
    }

    trial_count = {"n": 0}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "MIN_SIGNAL_SCORE":        trial.suggest_float("MIN_SIGNAL_SCORE",       4.5, 7.5, step=0.25),
            "ATR_STOP_MULTIPLIER":     trial.suggest_float("ATR_STOP_MULTIPLIER",    1.0, 2.5, step=0.25),
            "ATR_TARGET_MULTIPLIER":   trial.suggest_float("ATR_TARGET_MULTIPLIER",  2.0, 5.0, step=0.5),
            "ADX_THRESHOLD":           trial.suggest_float("ADX_THRESHOLD",          15,  35,  step=1.0),
            "VOLUME_SURGE_MULTIPLIER": trial.suggest_float("VOLUME_SURGE_MULTIPLIER",1.0, 2.5, step=0.25),
        }
        result = _mini_backtest(df, analyse_fn, symbol, params)
        trial_count["n"] += 1
        if progress_callback:
            progress_callback(trial_count["n"], n_trials, result.get("sharpe", -999))
        return result.get("sharpe", -999)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),   # Bayesian TPE sampler
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_result = _mini_backtest(df, analyse_fn, symbol, best)

    # Build improvement summary
    improvements = []
    for k, v_old in start_params.items():
        v_new = best.get(k, v_old)
        if abs(v_new - v_old) > 0.01:
            improvements.append(f"{k}: {v_old} → {v_new}")
    improvement_str = "\n".join(improvements) if improvements else "No significant changes"

    logger.info("Auto-tune complete for %s | best_sharpe=%.2f | best_return=%.1f%%",
                symbol, best_result.get("sharpe", 0), best_result.get("total_return", 0)*100)

    return TuningResult(
        best_params=best,
        best_sharpe=round(best_result.get("sharpe", 0), 3),
        best_return=round(best_result.get("total_return", 0) * 100, 2),
        n_trials=n_trials,
        improvement=improvement_str,
    )


def apply_best_params(result: TuningResult) -> None:
    """Apply the tuned parameters to config (runtime only — not saved to file)."""
    import config as _cfg
    for k, v in result.best_params.items():
        if hasattr(_cfg, k):
            setattr(_cfg, k, v)
            logger.info("Applied tuned param: %s = %s", k, v)
