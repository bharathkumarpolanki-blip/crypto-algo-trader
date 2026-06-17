#!/usr/bin/env python3
"""
CARRY HARVESTER — delta-neutral funding harvest (PAPER).  Phase 1.

Long spot + short perp of equal coin qty per token → delta-neutral. Earns the
perp funding premium (BTC ~+12%/yr, ETH ~+14%/yr net in 2020-2026 feasibility;
positive through the 2022 bear). "Smart" mode sits FLAT when funding goes negative.

  python3 carry_bot.py            # run the live PAPER harvester loop
  python3 carry_bot.py --once     # run a single cycle and exit
  python3 carry_bot.py --status   # print the saved book (no network) and exit

PAPER ONLY: there is NO live-execution path in this file. It reads OKX public data
and simulates the book — it cannot place a real order regardless of any config flag.
The honest catches the backtest can't show (counterparty/exchange failure, real
liquidation, capital-efficiency drag, current-regime funding compression, tax) are
what Phase 1 measures and Phase 2 must engineer around. See RESEARCH_FINDINGS.md.
"""
import sys
import time
import logging
from datetime import datetime, timezone

import config
from carry import data, engine, store

try:                                              # core module; guard just in case
    from notifications.notifier import send_telegram
except Exception:                                 # pragma: no cover
    def send_telegram(_msg: str) -> bool:         # fallback no-op
        return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("carry")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _telegram(msg: str) -> None:
    """Best-effort, non-blocking alert (send_telegram dispatches on a daemon thread,
    so this never blocks the harvest loop). No-op if Telegram isn't configured."""
    try:
        send_telegram(msg)
    except Exception:
        pass


# ── one harvest cycle ─────────────────────────────────────────────────────────
def gather_market(tokens: list[str]) -> dict:
    out = {}
    for t in tokens:
        try:
            out[t] = data.snapshot(t, config.CARRY_QUOTE)
        except Exception as e:
            logger.warning("%s: market fetch failed (%s) — book left untouched", t, e)
    return out


def run_once(state: dict) -> dict:
    tokens = config.CARRY_TOKENS
    store.ensure_books(state, tokens)
    markets = gather_market(tokens)
    now = _now_ms()
    for t in tokens:
        mkt = markets.get(t)
        if not mkt:                       # fetch failed → don't touch the book
            continue
        book = state["books"][t]
        prev = book["status"]
        engine.step(book, mkt, now)
        if book["status"] != prev:        # ON/FLAT flip → alert (off the hot path)
            if book["status"] == "ON":
                _telegram(f"🪙 *{t} carry OPENED* (PAPER) — funding APR "
                          f"{mkt['apr']*100:+.1f}%, ${config.CARRY_NOTIONAL_USD:.0f}/leg.")
            else:
                _telegram(f"💤 *{t} carry CLOSED* (PAPER) — funding APR fell to "
                          f"{mkt['apr']*100:+.1f}% (smart exit). Funding banked "
                          f"${book['funding_income']:+.2f}.")
        book["last_mkt"] = mkt            # cache for --status (offline)
    state["last_cycle"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state["cycles"] = state.get("cycles", 0) + 1
    store.save(state)
    return markets


# ── reporting ─────────────────────────────────────────────────────────────────
def _book_mkt(state: dict, t: str) -> dict | None:
    b = state["books"].get(t, {})
    return b.get("last_mkt")


def portfolio_summary(state: dict) -> dict:
    days = max((_now_ms() - state.get("started_ms", _now_ms())) / 86_400_000.0, 1e-6)
    tot_fund = tot_fee = tot_net = tot_cap = 0.0
    rows = []
    for t, b in state["books"].items():
        mkt = b.get("last_mkt")
        if not mkt:
            continue
        net = engine.net_pnl(b, mkt)
        cap = engine.capital_deployed(b)
        tot_fund += b["funding_income"]; tot_fee += b["fees_paid"]
        tot_net += net; tot_cap += cap
        # Annualizing over a sub-day window is meaningless (and blows up); only
        # report a run-to-date APR once enough time has elapsed to be informative.
        est_apr = ((net / config.CARRY_NOTIONAL_USD) * (365.0 / days) * 100
                   if days >= 0.5 else None)
        rows.append({
            "token": t, "status": b["status"], "apr_now": mkt["apr"] * 100,
            "funding": b["funding_income"], "fees": b["fees_paid"],
            "residual": b["realized_residual"] + engine.unrealized_residual(b, mkt),
            "net": net, "flips": b["flips"], "est_apr": est_apr,
            "liq": engine.margin_health(b, mkt),
        })
    net_apr = (tot_net / tot_cap * 365.0 / days * 100
               if (tot_cap > 0 and days >= 0.5) else None)
    return {"days": days, "rows": rows, "tot_funding": tot_fund, "tot_fees": tot_fee,
            "tot_net": tot_net, "tot_cap": tot_cap, "net_apr": net_apr}


def print_status(state: dict) -> None:
    s = portfolio_summary(state)
    mode = "PAPER"   # Phase 1 has no live path
    print(f"\n{'='*78}")
    print(f"  CARRY HARVESTER ({mode}) — delta-neutral funding harvest")
    print(f"  run {s['days']:.2f}d | cycles {state.get('cycles',0)} | "
          f"smart={'on' if config.CARRY_SMART else 'off'} | "
          f"notional ${config.CARRY_NOTIONAL_USD:.0f}/leg | lev {config.CARRY_LEVERAGE:g}x")
    print(f"{'='*78}")
    if not s["rows"]:
        print("  (no market data yet — run a cycle first)\n"); return
    print(f"  {'TOK':4} {'STAT':4} {'APRnow':>7} {'funding':>9} {'fees':>7} "
          f"{'basis':>7} {'NET$':>8} {'~netAPR':>8} {'flips':>5} {'liqDist':>8}")
    for r in s["rows"]:
        liq = r["liq"]["dist_pct"]
        liqs = f"{liq:+.1f}%" if liq is not None else "   —"
        apr = f"{r['est_apr']:+7.1f}%" if r["est_apr"] is not None else "   warmup"
        print(f"  {r['token']:4} {r['status']:4} {r['apr_now']:+6.1f}% "
              f"{r['funding']:+8.3f} {r['fees']:6.3f} {r['residual']:+6.3f} "
              f"{r['net']:+7.3f} {apr:>8} {r['flips']:5d} {liqs:>8}")
    print(f"  {'-'*76}")
    napr = f"{s['net_apr']:+.1f}%" if s["net_apr"] is not None else "warming up"
    print(f"  TOTAL  funding {s['tot_funding']:+.3f}  fees {s['tot_fees']:.3f}  "
          f"NET ${s['tot_net']:+.3f}  | capital ${s['tot_cap']:.0f}  | "
          f"blended net APR {napr}")
    print(f"  Note: counterparty/exchange-failure risk is NOT in these numbers — it is")
    print(f"  the dominant real-world risk and the core of Phase 2.  (paper, no real money)\n")


# ── runner ────────────────────────────────────────────────────────────────────
def _maybe_start_dashboard() -> None:
    """Serve the web dashboard from this process (open the '🪙 Carry' tab), so you
    don't need bot.py just for the UI. Non-fatal: a busy port (bot.py/sma_bot
    already serving) or a missing dep just logs and the harvester runs headless."""
    if not getattr(config, "CARRY_DASHBOARD", False):
        return
    port = getattr(config, "CARRY_DASHBOARD_PORT", 8081)
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            logger.warning("Dashboard port %d already in use (bot.py/sma_bot running?) — "
                           "skipping. The 🪙 Carry tab is already served there.", port)
            return
    finally:
        s.close()
    try:
        from ui.dashboard import start_server
        start_server(port=port)
        print(f"  🖥  Dashboard: http://localhost:{port}   (open the 🪙 Carry tab)")
    except Exception as e:
        logger.warning("Could not start dashboard: %s (running headless)", e)


def run() -> None:
    state = store.load()
    logger.info("Carry harvester starting | tokens=%s | PAPER | poll every %ds",
                config.CARRY_TOKENS, config.CARRY_POLL_SECONDS)
    _maybe_start_dashboard()
    _telegram(f"🪙 Carry harvester started (PAPER) — {', '.join(config.CARRY_TOKENS)}")
    try:
        while True:
            try:
                run_once(state)
                print_status(state)
            except Exception as e:
                logger.error("Cycle error: %s", e)
            slept, total = 0, config.CARRY_POLL_SECONDS
            while slept < total:                  # short slices → prompt Ctrl+C
                time.sleep(min(1, total - slept))
                slept += 1
    except KeyboardInterrupt:
        store.save(state)
        s = portfolio_summary(state)
        _telegram(f"🛑 Carry harvester stopped (PAPER) — net ${s['tot_net']:+.2f} "
                  f"over {s['days']:.1f}d, funding ${s['tot_funding']:+.2f}.")
        logger.info("Carry harvester stopped (Ctrl+C). State saved.")
        print("\n👋 Carry harvester stopped. State saved — restart any time.")


def main() -> None:
    args = set(sys.argv[1:])
    if "--status" in args:
        print_status(store.load()); return
    if "--once" in args:
        state = store.load()
        run_once(state)
        print_status(state)
        return
    run()


if __name__ == "__main__":
    main()
