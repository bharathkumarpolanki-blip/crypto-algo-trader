"""
Integrated ROTATION + PORTFOLIO + REGIME strategy (the evidence-driven redesign).

Combines the three things the robustness audit said actually help:
  • RELATIVE-STRENGTH ROTATION (Phase 9 winner): hold the Top-N assets ranked by
    12-week momentum — concentrate in what's strongest.
  • TREND GATE (Phase 7): only hold an asset while its weekly close > 30-week SMA.
  • REGIME GATE (targets the Phase-12 out-of-sample failure): scale the whole
    book to cash when BTC is in a confirmed downtrend (below a rising 200-day SMA).
  • PORTFOLIO (Phase 11): equal-weight the held names to diversify drawdown.

Long-only spot, weekly rebalance, fees + slippage. Daily candles.
This is RESEARCH until it passes validate.py — no live wiring.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config
from exchange.market_data import fetch_ohlcv

FEE  = config.FEE_RATE_PCT / 100.0
SLIP = 0.05 / 100.0
RT   = FEE + SLIP                      # per-side cost
CAP  = 1000.0

DEFAULT_UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
                    "DOGE/USD", "XRP/USD", "ADA/USD", "DOT/USD"]
WK_SMA   = 30      # weekly trend filter (weeks)
MOM_DAYS = 84      # 12-week momentum lookback
REGIME_SMA = 200   # BTC daily regime filter
REGIME_SLOPE = 20  # slope window for "rising"

_cache: dict[str, pd.DataFrame] = {}


def _load(sym: str) -> pd.DataFrame | None:
    if sym not in _cache:
        df = fetch_ohlcv(sym, "1d", limit=2700)
        _cache[sym] = df if (df is not None and not df.empty and len(df) > 250) else None
    return _cache[sym]


def _weekly_trend_up(close: pd.Series) -> pd.Series:
    """Daily boolean: weekly close > 30-week SMA, forward-filled to daily index."""
    w = close.resample("1W").last().dropna()
    up = w > w.rolling(WK_SMA).mean()
    return up.reindex(close.index, method="ffill").fillna(False)


def _regime_risk_on(btc_close: pd.Series) -> pd.Series:
    """Daily boolean: BTC above a RISING 200-day SMA (risk-on)."""
    sma = btc_close.rolling(REGIME_SMA).mean()
    rising = sma.diff(REGIME_SLOPE) > 0
    return ((btc_close > sma) & rising).fillna(False)


def backtest(universe: list[str] | None = None,
             top_n: int = 2,
             use_regime_gate: bool = True,
             start: str | None = None,
             end: str | None = None,
             fee_mult: float = 1.0) -> dict:
    """
    Run the integrated strategy. Returns equity curve + metrics + trades.
    fee_mult scales the per-side cost (for cost-stress testing).
    """
    rt = RT * fee_mult
    universe = universe or DEFAULT_UNIVERSE
    data = {s: _load(s) for s in universe}
    data = {k: v for k, v in data.items() if v is not None}
    if "BTC/USD" not in data or len(data) < 2:
        return {"error": "insufficient data"}

    # UNION index (driven by BTC's full history). Newer coins are simply absent
    # (NaN) until they list — they become eligible only once they have data, so
    # short-history assets don't truncate the whole backtest.
    idx = data["BTC/USD"].index
    for v in data.values():
        idx = idx.union(v.index)
    idx = idx.sort_values()
    if start: idx = idx[idx >= start]
    if end:   idx = idx[idx < end]
    if len(idx) < 260:
        return {"error": "not enough history"}

    close = pd.DataFrame({k: v["close"].reindex(idx) for k, v in data.items()})
    # BTC must be present throughout (it anchors the regime gate)
    close = close[close["BTC/USD"].notna()]
    idx = close.index
    mom = close.pct_change(MOM_DAYS)
    trend_up = pd.DataFrame({k: _weekly_trend_up(close[k]) for k in close.columns})
    risk_on = _regime_risk_on(close["BTC/USD"]) if use_regime_gate else pd.Series(True, index=idx)

    cash = CAP
    held: dict[str, dict] = {}          # sym -> {units, entry}
    eq, trades = [], []
    rebal_days = set(idx[::7])          # weekly rebalance

    for d in idx:
        px = close.loc[d]
        if d in rebal_days:
            if not risk_on.loc[d]:
                pick = []               # regime risk-off → all cash
            else:
                elig = [c for c in close.columns if trend_up.loc[d, c] and not np.isnan(mom.loc[d, c])]
                ranked = mom.loc[d, elig].sort_values(ascending=False)
                pick = list(ranked.index[:top_n])
            # liquidate anything not in the new pick
            for c in list(held):
                if c not in pick:
                    cash += held[c]["units"] * px[c] * (1 - rt)
                    trades.append((px[c] - held[c]["entry"]) / held[c]["entry"])
                    del held[c]
            # allocate equally across picks (rebalance cash into equal weights)
            if pick:
                # mark-to-market current holdings, then equalize
                total = cash + sum(held[c]["units"] * px[c] for c in held)
                target = total / len(pick)
                # sell overweights / buy underweights to hit equal target
                for c in pick:
                    cur_val = held[c]["units"] * px[c] if c in held else 0.0
                    if c not in held:
                        units = (target * (1 - rt)) / px[c]
                        held[c] = {"units": units, "entry": px[c]}
                        cash -= target
                    # (skip intra-rebalance trimming to keep fees realistic/low)
        eq.append(cash + sum(held[c]["units"] * px[c] for c in held))

    equity = pd.Series(eq, index=idx)
    return {"equity": equity, "trades": trades, **_metrics(equity, trades)}


def _metrics(eq: pd.Series, trades: list, ann: int = 365) -> dict:
    idx = eq.index
    yrs = (idx[-1] - idx[0]).days / 365.25 if len(idx) > 1 else 1
    end = eq.iloc[-1]
    ret = (end / CAP - 1) * 100
    cagr = ((end / CAP) ** (1 / yrs) - 1) * 100 if yrs > 0 and end > 0 else -100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0.0
    dn = r[r < 0]
    sortino = r.mean() / dn.std() * np.sqrt(ann) if len(dn) > 1 and dn.std() > 0 else 0.0
    wins = [t for t in trades if t > 0]
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    losssum = abs(sum(t for t in trades if t <= 0))
    pf = sum(wins) / losssum if losssum > 0 else (float("inf") if wins else 0.0)
    return {"ret": ret, "cagr": cagr, "maxdd": dd, "sharpe": sharpe,
            "sortino": sortino, "wr": wr, "pf": pf, "trades_n": len(trades),
            "years": round(yrs, 2)}


def buy_hold(sym: str = "BTC/USD") -> dict:
    df = _load(sym)
    if df is None: return {}
    eq = (CAP * (1 - FEE)) / df["close"].iloc[0] * df["close"]
    return _metrics(eq, [])
