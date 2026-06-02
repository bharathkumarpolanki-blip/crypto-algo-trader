"""Telegram notifications — rich trade and signal messages."""

import logging
import requests
from datetime import datetime, timezone
import config

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id":    config.TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }, timeout=10)
        if resp.status_code != 200:
            logger.warning("Telegram error %s: %s", resp.status_code, resp.text[:200])
        return resp.status_code == 200
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _pnl_emoji(pnl: float) -> str:
    if pnl > 0:   return "🟢"
    if pnl < 0:   return "🔴"
    return "⚪"


def _dir_emoji(direction: str) -> str:
    return "📈" if direction == "long" else "📉"


def _score_bar(score: float) -> str:
    """Visual score bar e.g. ████░░░░░░ 7.2/10"""
    filled = round(score)
    return "█" * filled + "░" * (10 - filled) + f" {score:.1f}/10"


# ── Trade opened ──────────────────────────────────────────────────────────────

def notify_trade_open(symbol: str, direction: str, entry: float,
                      qty: float, stop: float, take_profit: float,
                      risk_usd: float, score: float,
                      candle_patterns: dict | None = None,
                      sentiment_label: str = "neutral") -> None:
    rr = round(abs(take_profit - entry) / abs(stop - entry), 2) if abs(stop - entry) > 0 else 0
    invested = round(entry * qty, 2)
    patterns_str = ""
    if candle_patterns:
        names = ", ".join(k.replace("_", " ").title() for k in candle_patterns)
        patterns_str = f"\n📊 *Patterns:* {names}"

    msg = (
        f"{_dir_emoji(direction)} *TRADE OPENED — {direction.upper()}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Symbol:*      `{symbol}`\n"
        f"📥 *Buy Price:*   `${entry:,.4f}`\n"
        f"📦 *Quantity:*    `{qty:.6f}` units\n"
        f"💵 *Invested:*    `${invested:,.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛑 *Stop Loss:*   `${stop:,.4f}`  (-${abs(entry - stop) * qty:,.2f})\n"
        f"🎯 *Take Profit:* `${take_profit:,.4f}`  (+${abs(take_profit - entry) * qty:,.2f})\n"
        f"⚖️  *Risk:*        `${risk_usd:,.2f}` | R:R `{rr:.2f}x`\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 *Signal Score:* {_score_bar(score)}\n"
        f"📰 *Sentiment:*   {sentiment_label.capitalize()}"
        f"{patterns_str}\n"
        f"🕐 {_now()}"
    )
    send_telegram(msg)


# ── Trade closed ──────────────────────────────────────────────────────────────

def notify_trade_close(symbol: str, direction: str,
                       entry: float, exit_price: float,
                       qty: float, pnl: float, reason: str,
                       hold_duration_mins: int | None = None) -> None:
    invested   = round(entry * qty, 2)
    pnl_pct    = round(pnl / invested * 100, 2) if invested > 0 else 0
    emoji      = _pnl_emoji(pnl)
    reason_map = {
        "take_profit": "🎯 Take Profit hit",
        "stop_loss":   "🛑 Stop Loss hit",
        "trailing":    "📌 Trailing Stop hit",
        "manual":      "✋ Manually closed",
        "EOD":         "⏱ End of session",
    }
    reason_str  = reason_map.get(reason, f"📋 {reason}")
    duration_str = f"\n⏱ *Held:*         `{hold_duration_mins} min`" if hold_duration_mins else ""

    msg = (
        f"{emoji} *TRADE CLOSED — {direction.upper()}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Symbol:*      `{symbol}`\n"
        f"📥 *Bought at:*   `${entry:,.4f}`\n"
        f"📤 *Sold at:*     `${exit_price:,.4f}`\n"
        f"📦 *Quantity:*    `{qty:.6f}` units\n"
        f"💵 *Invested:*    `${invested:,.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *P&L:*          `{'+'if pnl>=0 else ''}{pnl:,.2f} USD`  "
        f"(`{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%`)\n"
        f"📋 *Reason:*      {reason_str}"
        f"{duration_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {_now()}"
    )
    send_telegram(msg)


# ── Signal alert (no trade opened — just informational) ───────────────────────

def notify_signal(signal) -> None:
    patterns = list((signal.candle_patterns or {}).keys())
    pat_str  = ", ".join(p.replace("_", " ").title() for p in patterns[:3]) if patterns else "None"
    top_comp = sorted(signal.components.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    comp_str = "  ".join(f"{k.replace('_',' ')}={v:+.1f}" for k, v in top_comp)

    msg = (
        f"{_dir_emoji(signal.direction)} *SIGNAL — {signal.direction.upper()}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Symbol:*      `{signal.symbol}`\n"
        f"📊 *Score:*       {_score_bar(signal.score)}\n"
        f"📥 *Entry:*       `${signal.entry_price:,.4f}`\n"
        f"🛑 *Stop:*        `${signal.stop_loss:,.4f}`\n"
        f"🎯 *Target:*      `${signal.take_profit:,.4f}`\n"
        f"⚖️  *R:R:*         `{signal.risk_reward:.2f}x`\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📰 *Sentiment:*   {signal.sentiment.get('label','neutral').capitalize()}\n"
        f"📊 *Patterns:*    {pat_str}\n"
        f"🔍 *Top factors:* `{comp_str}`\n"
        f"🕐 {_now()}"
    )
    send_telegram(msg)


# ── Portfolio summary ─────────────────────────────────────────────────────────

def notify_portfolio_summary(capital: float, start_capital: float,
                              open_positions: list, closed_trades: list) -> None:
    pnl       = capital - start_capital
    pnl_pct   = round(pnl / start_capital * 100, 2) if start_capital > 0 else 0
    wins      = [t for t in closed_trades if (t.get("pnl") or 0) > 0]
    losses    = [t for t in closed_trades if (t.get("pnl") or 0) <= 0]
    win_rate  = round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0
    emoji     = _pnl_emoji(pnl)

    positions_str = ""
    for p in open_positions[:5]:
        p_pnl = p.get("pnl", 0)
        positions_str += (
            f"\n  • `{p['symbol']}` {p['side'].upper()} "
            f"@ ${p['entry']:,.2f} → "
            f"{'+'if p_pnl>=0 else ''}{p_pnl:,.2f} USD"
        )

    msg = (
        f"📋 *PORTFOLIO SUMMARY*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 *Capital:*     `${capital:,.2f}`\n"
        f"{emoji} *Total P&L:*   `{'+'if pnl>=0 else ''}{pnl:,.2f} USD` "
        f"(`{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%`)\n"
        f"📊 *Win Rate:*    `{win_rate}%` "
        f"({len(wins)}W / {len(losses)}L)\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔓 *Open Positions:* {len(open_positions)}"
        f"{positions_str}\n"
        f"🕐 {_now()}"
    )
    send_telegram(msg)


# ── Error alert ───────────────────────────────────────────────────────────────

def notify_error(msg: str) -> None:
    send_telegram(f"⚠️ *BOT ERROR*\n`{msg}`\n🕐 {_now()}")


# ── Bot started/stopped ───────────────────────────────────────────────────────

def notify_bot_started(capital: float, watchlist: list) -> None:
    send_telegram(
        f"🚀 *BOT STARTED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 *Capital:*  `${capital:,.2f} USD`\n"
        f"🌍 *Watching:* `{len(watchlist)} symbols`\n"
        f"🔍 *Mode:*     `{'DRY RUN (paper)' if config.DRY_RUN else 'LIVE TRADING'}`\n"
        f"⏱  *Interval:* every `{config.SCAN_INTERVAL_SECONDS}s`\n"
        f"🕐 {_now()}"
    )


def notify_bot_stopped(capital: float, start_capital: float,
                        total_trades: int) -> None:
    pnl     = capital - start_capital
    pnl_pct = round(pnl / start_capital * 100, 2) if start_capital > 0 else 0
    send_telegram(
        f"🛑 *BOT STOPPED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 *Final Capital:* `${capital:,.2f}`\n"
        f"{_pnl_emoji(pnl)} *Session P&L:*   `{'+'if pnl>=0 else ''}{pnl:,.2f} USD` "
        f"(`{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%`)\n"
        f"📊 *Total Trades:*  `{total_trades}`\n"
        f"🕐 {_now()}"
    )


# ── Backward-compat wrapper (used by old bot.py call sites) ──────────────────

def notify_trade(action: str, symbol: str, price: float,
                 reason: str, pnl: float | None = None) -> None:
    """Legacy wrapper — kept so existing bot.py calls don't break."""
    if action == "open":
        send_telegram(
            f"📥 *OPENED* `{symbol}` @ `${price:,.4f}`\n"
            f"Reason: {reason}\n🕐 {_now()}"
        )
    else:
        emoji = _pnl_emoji(pnl or 0)
        pnl_str = f"\n{emoji} *P&L:* `{'+'if (pnl or 0)>=0 else ''}{pnl:,.2f} USD`" if pnl is not None else ""
        send_telegram(
            f"📤 *CLOSED* `{symbol}` @ `${price:,.4f}`\n"
            f"Reason: {reason}{pnl_str}\n🕐 {_now()}"
        )
