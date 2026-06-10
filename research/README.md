# research/ — investigation archive

Reproducible scripts from the multi-day edge-discovery investigation. **Run from
the repo root** (e.g. `python3 research/risk_intelligence.py`) — they `sys.path`
into the project for `exchange/`, `core/`, etc.

These are **archived research, not part of the live bot.** The bot is
`bot.py` / `sma_bot.py` / `config.py` (root). The only research artifact promoted
to "permanent infrastructure" is **`validate.py`** (root) — the deployability gate.

## What's here (grouped)

**Edge validation / forensics**
`forensic_audit`, `forensic_all`, `forensic_final`, `forensic_rediscovery`,
`edge_validation`, `edge_inventory`, `signal_discovery`, `horizon_discovery`,
`mr_validation`, `ema_validate`, `weeklysma_deep`

**Strategy backtests (all failed OOS/cost)**
`best_pick_backtest`, `breakout_backtest`, `daily_backtest`,
`daily_portfolio_backtest`, `tf_portfolio_backtest`, `htf_systems`,
`cross_sectional_adversarial`, `vwap_ema_test`, `vwap_ema_daily_test`, `tune`,
`compare_fees`

**Regime / risk research**
`regime_adaptive`, `regime_switch_test`, `regime_walkforward`,
`unsupervised_regimes`, `risk_intelligence`, `fragility_audit`, `exposure_engine`

**Alt-data studies (funding / OI / on-chain / cross-asset / ML)**
`funding_study`, `oi_study`, `onchain_study`, `redundancy_benchmark`,
`cross_asset_leadlag`, `ml_interactions`

## Verdict
Every approach failed an honest out-of-sample + cost gate. See
`../RESEARCH_FINDINGS.md` for the full write-up. The deployable conclusion is a
risk-managed crypto-beta system (200-day SMA, `sma_bot.py`) — drawdown control,
not alpha.
