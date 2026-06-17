"""
Delta-neutral carry harvester — PAPER engine.

Per token we hold a "book": long spot + short perp of EQUAL coin quantity. Because
both legs are the same coin size, the book is delta-neutral — price moves cancel
(spot gains == perp-short losses), so P&L comes from FUNDING, not direction.

  • Funding accrues prorated by elapsed time (rate is per interval → per-hour =
    rate/interval_h). Over a day this equals the sum of the day's settlements; the
    cumulative is identical, only the granularity is smoothed — honest for paper.
  • "Smart" mode (config.CARRY_SMART) flips the book FLAT when funding turns
    negative, so we never PAY funding. Hysteresis (enter ≥ MIN_APR, exit ≤
    EXIT_APR) prevents churn around zero — and every flip pays real round-trip
    fees, which is exactly the friction Phase 1 exists to measure.
  • Basis residual (spot−perp price gap drift) is tracked explicitly; it's the
    small non-funding P&L that a real delta-neutral book actually carries.

No live execution here by design — this module only mutates an in-memory/JSON book.
"""
import logging
import config

logger = logging.getLogger("carry.engine")
HOUR_MS = 3_600_000.0


def new_book(token: str) -> dict:
    return {
        "token": token, "status": "FLAT",
        "spot_qty": 0.0, "spot_entry": 0.0,
        "perp_qty": 0.0, "perp_entry": 0.0,
        "funding_income": 0.0,     # cumulative USD funding earned (the edge)
        "fees_paid": 0.0,          # cumulative USD trading fees (flip cost)
        "realized_residual": 0.0,  # basis P&L locked in on closes
        "last_accrual_ms": None,
        "opened_ms": None,
        "flips": 0,                # number of ON/FLAT transitions (churn meter)
        "last_rate": 0.0, "last_apr": 0.0,
    }


def _spot_fee(notional: float) -> float:
    return abs(notional) * config.CARRY_SPOT_FEE_PCT / 100.0


def _perp_fee(notional: float) -> float:
    return abs(notional) * config.CARRY_PERP_FEE_PCT / 100.0


def _open(book: dict, mkt: dict, now_ms: int) -> None:
    n = config.CARRY_NOTIONAL_USD
    book["spot_qty"] = n / mkt["spot"]
    book["spot_entry"] = mkt["spot"]
    book["perp_qty"] = n / mkt["mark"]
    book["perp_entry"] = mkt["mark"]
    book["fees_paid"] += _spot_fee(n) + _perp_fee(n)
    book["status"] = "ON"
    book["opened_ms"] = now_ms
    book["last_accrual_ms"] = now_ms
    book["flips"] += 1
    logger.info("%s: OPEN carry  spot %.6f @ %.2f | short perp %.6f @ %.2f | APR %.1f%%",
                book["token"], book["spot_qty"], mkt["spot"],
                book["perp_qty"], mkt["mark"], mkt["apr"] * 100)


def _close(book: dict, mkt: dict) -> None:
    resid = (book["spot_qty"] * (mkt["spot"] - book["spot_entry"])
             + book["perp_qty"] * (book["perp_entry"] - mkt["mark"]))
    book["realized_residual"] += resid
    book["fees_paid"] += (_spot_fee(book["spot_qty"] * mkt["spot"])
                          + _perp_fee(book["perp_qty"] * mkt["mark"]))
    book["status"] = "FLAT"
    book["spot_qty"] = book["perp_qty"] = 0.0
    book["spot_entry"] = book["perp_entry"] = 0.0
    book["opened_ms"] = None
    book["last_accrual_ms"] = None
    book["flips"] += 1
    logger.info("%s: CLOSE carry  basis residual %+.4f USD | APR now %.1f%%",
                book["token"], resid, mkt["apr"] * 100)


def _accrue(book: dict, mkt: dict, now_ms: int) -> None:
    if book["status"] != "ON" or book["last_accrual_ms"] is None:
        return
    elapsed_h = (now_ms - book["last_accrual_ms"]) / HOUR_MS
    if elapsed_h <= 0:
        return
    perp_notional = book["perp_qty"] * mkt["mark"]
    per_hour_rate = mkt["rate"] / mkt["interval_h"]   # rate is per funding interval
    book["funding_income"] += per_hour_rate * elapsed_h * perp_notional
    book["last_accrual_ms"] = now_ms


def step(book: dict, mkt: dict, now_ms: int) -> dict:
    """Advance one book by one market snapshot: accrue, then apply smart decision."""
    book["last_rate"] = mkt["rate"]
    book["last_apr"] = mkt["apr"]
    # 1) bank funding for the period just elapsed (incl. any negative drag while ON)
    _accrue(book, mkt, now_ms)
    # 2) decide ON/FLAT with hysteresis
    if book["status"] == "FLAT":
        if mkt["apr"] >= config.CARRY_MIN_APR:
            _open(book, mkt, now_ms)
    else:  # ON
        if config.CARRY_SMART and mkt["apr"] <= config.CARRY_EXIT_APR:
            _close(book, mkt)
    return book


# ── read-only derived metrics ────────────────────────────────────────────────
def unrealized_residual(book: dict, mkt: dict) -> float:
    if book["status"] != "ON":
        return 0.0
    return (book["spot_qty"] * (mkt["spot"] - book["spot_entry"])
            + book["perp_qty"] * (book["perp_entry"] - mkt["mark"]))


def net_pnl(book: dict, mkt: dict) -> float:
    return (book["funding_income"] + book["realized_residual"]
            + unrealized_residual(book, mkt) - book["fees_paid"])


def capital_deployed(book: dict) -> float:
    """Spot leg (fully paid) + perp margin (notional/leverage) when ON, else 0."""
    if book["status"] != "ON":
        return 0.0
    n = config.CARRY_NOTIONAL_USD
    return n + n / max(config.CARRY_LEVERAGE, 1e-9)


def margin_health(book: dict, mkt: dict) -> dict:
    """
    Honest margin/liquidation read for the SHORT perp leg.

    Cross-margined against the spot long, the book is naturally safe (spot gains
    fund the short's losses). We still surface the ISOLATED-margin liquidation
    price — the level the short alone would liquidate at given CARRY_LEVERAGE —
    because that's the number a live operator must watch. With the spot hedge,
    hitting it isn't fatal; without it, it is.
    """
    if book["status"] != "ON":
        return {"liq_price": None, "dist_pct": None}
    lev = max(config.CARRY_LEVERAGE, 1e-9)
    maint = 0.005                                  # ~0.5% maintenance margin (OKX-ish)
    liq = book["perp_entry"] * (1 + 1 / lev - maint)   # short liquidates as price rises
    dist = (liq - mkt["mark"]) / mkt["mark"] * 100
    return {"liq_price": liq, "dist_pct": dist}
