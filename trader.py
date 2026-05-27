"""
Paper Trading System — 8 strategies running simultaneously.
Runs daily. Checks signals on schedule. Logs everything to state.json.
Generates HTML dashboard at docs/index.html.

Strategies:
1. TQQQ+SOXL, 20d momentum, 5d check (FLAGSHIP — best OOS performer)
2. TQQQ+SOXL, 30d momentum, 10d check (slower, fewer trades)
3. TQQQ+SOXL, 20d momentum, 10d check (conservative)
4. SPY RSI(2) mean reversion (buy RSI<10, sell RSI>60 or 15-day cap)
5. QQQ 5d momentum > 3%, hold 20 days
6. UPRO+FAS, 20d momentum, 5d check (non-tech rotation)
7. KORU+NAIL+ERX, 20d momentum, 5d check (PROBATIONARY — added 2026-05-25,
   validated train +32% / test +95% on 2023-2026 unseen)
8. KORU+DPST, 20d momentum, 5d check (PROBATIONARY — added 2026-05-25,
   validated train +19% / test +54% on 2023-2026 unseen)
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Try importing dependencies
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "yfinance", "pandas", "numpy", "-q"])
    import yfinance as yf
    import pandas as pd
    import numpy as np

import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "state.json"
HTML_FILE = SCRIPT_DIR / "docs" / "index.html"
STARTING_CAPITAL = 1000.0


# ============================================================
# DATA
# ============================================================

def fetch_prices(ticker, days=60):
    """Fetch daily OHLC data for a ticker. Robust to yfinance API changes:
    - Pins auto_adjust=True explicitly (default changed across versions)
    - Handles both MultiIndex and flat column structures
    - Handles 'Date' / 'Datetime' / unnamed index from different yfinance versions"""
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if data is None or len(data) == 0:
        raise ValueError(f"empty response from yfinance")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Locate the date column — yfinance versions differ
    date_col = None
    for candidate in ('Date', 'Datetime', 'date', 'datetime', 'index'):
        if candidate in data.columns:
            date_col = candidate
            break
    if date_col is None and len(data.columns) > 0:
        # Last resort: assume first column post reset_index is the date
        date_col = data.columns[0]

    rename_map = {date_col: 'date', 'Close': 'close', 'Open': 'open',
                  'High': 'high', 'Low': 'low'}
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    if 'date' not in data.columns or 'close' not in data.columns:
        raise ValueError(f"missing required columns after rename. Got: {list(data.columns)}")

    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    cols = ['date', 'close'] + (['open'] if 'open' in data.columns else [])
    return data[cols].sort_values('date').reset_index(drop=True)


def get_all_prices():
    tickers = ['TQQQ', 'SOXL', 'SPY', 'QQQ', 'UPRO', 'FAS',
               'KORU', 'NAIL', 'ERX', 'DPST']
    prices = {}
    for t in tickers:
        try:
            df = fetch_prices(t)
            if len(df) > 0:
                prices[t] = df
        except Exception as e:
            print(f"  WARN: Failed to fetch {t}: {type(e).__name__}: {e}")
    return prices


def compute_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(2).mean()
    avg_loss = loss.rolling(2).mean()
    # Standard RSI; handle div-by-zero so RSI is always defined:
    # - no losses (avg_loss == 0) means only up days -> RSI = 100 (max overbought)
    # - no gains (avg_gain == 0) means only down days -> RSI = 0 (max oversold)
    # - both zero (no movement) -> RSI = 50 (neutral)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_2'] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), 'rsi_2'] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), 'rsi_2'] = 0
    df.loc[(avg_gain == 0) & (avg_loss == 0), 'rsi_2'] = 50
    return df


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================

def eval_momentum_rotation(prices, tickers, mom_period):
    best = None
    best_mom = -999

    for t in tickers:
        if t not in prices or len(prices[t]) < mom_period + 1:
            continue
        df = prices[t]
        current = df.iloc[-1]['close']
        past = df.iloc[-(mom_period + 1)]['close']
        ma = df['close'].tail(mom_period).mean()
        mom = (current / past) - 1

        if current > ma and mom > 0 and mom > best_mom:
            best_mom = mom
            best = t

    return best, best_mom


def eval_rsi_spy(prices):
    if 'SPY' not in prices or len(prices['SPY']) < 5:
        return None
    df = compute_indicators(prices['SPY'])
    current_rsi = df.iloc[-1]['rsi_2']
    if pd.isna(current_rsi):
        return None
    return current_rsi


def eval_qqq_momentum(prices):
    if 'QQQ' not in prices or len(prices['QQQ']) < 6:
        return None
    df = prices['QQQ']
    current = df.iloc[-1]['close']
    past = df.iloc[-6]['close']
    mom_5d = (current / past) - 1
    return mom_5d


# ============================================================
# STATE MANAGEMENT
# ============================================================

def get_default_strategies():
    """Single source of truth for strategy definitions. Adding a strategy here
    auto-applies to fresh state AND to existing state via load_state migration."""
    return {
        "S1_TQQQ_SOXL_20d_5d": {
            "name": "TQQQ+SOXL 20d mom, 5d check",
            "type": "momentum_rotation",
            "tickers": ["TQQQ", "SOXL"],
            "mom_period": 20,
            "check_interval": 5,
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_since_check": 999,
            "trades": []
        },
        "S2_TQQQ_SOXL_30d_10d": {
            "name": "TQQQ+SOXL 30d mom, 10d check",
            "type": "momentum_rotation",
            "tickers": ["TQQQ", "SOXL"],
            "mom_period": 30,
            "check_interval": 10,
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_since_check": 999,
            "trades": []
        },
        "S3_TQQQ_SOXL_20d_10d": {
            "name": "TQQQ+SOXL 20d mom, 10d check",
            "type": "momentum_rotation",
            "tickers": ["TQQQ", "SOXL"],
            "mom_period": 20,
            "check_interval": 10,
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_since_check": 999,
            "trades": []
        },
        "S4_SPY_RSI": {
            "name": "SPY RSI(2) mean reversion",
            "type": "rsi_reversion",
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_in_trade": 0,
            "trades": []
        },
        "S5_QQQ_MOM": {
            "name": "QQQ 5d momentum, hold 20d",
            "type": "qqq_momentum",
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_in_trade": 0,
            "trades": []
        },
        "S6_UPRO_FAS_20d_5d": {
            "name": "UPRO+FAS 20d mom, 5d check",
            "type": "momentum_rotation",
            "tickers": ["UPRO", "FAS"],
            "mom_period": 20,
            "check_interval": 5,
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_since_check": 999,
            "trades": []
        },
        "S7_KORU_NAIL_ERX_20d_5d": {
            "name": "KORU+NAIL+ERX 20d mom, 5d check",
            "type": "momentum_rotation",
            "tickers": ["KORU", "NAIL", "ERX"],
            "mom_period": 20,
            "check_interval": 5,
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_since_check": 999,
            "trades": []
        },
        "S8_KORU_DPST_20d_5d": {
            "name": "KORU+DPST 20d mom, 5d check",
            "type": "momentum_rotation",
            "tickers": ["KORU", "DPST"],
            "mom_period": 20,
            "check_interval": 5,
            "holding": None,
            "entry_price": 0,
            "entry_date": None,
            "equity": STARTING_CAPITAL,
            "shares": 0,
            "days_since_check": 999,
            "trades": []
        }
    }


def load_state():
    defaults = get_default_strategies()

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Migration: add any strategy slots defined in defaults but missing from
        # saved state. Lets us roll out new strategies without losing existing state.
        for sid, default_strat in defaults.items():
            if sid not in state["strategies"]:
                state["strategies"][sid] = default_strat
                print(f"  Added new strategy slot: {sid}")
        return state

    return {
        "created": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "last_run": None,
        "run_count": 0,
        "strategies": defaults
    }


def save_state(state):
    try:
        from zoneinfo import ZoneInfo
        et_now = datetime.now(ZoneInfo("America/New_York"))
        state["last_run"] = et_now.strftime('%Y-%m-%d %H:%M ET')
    except Exception:
        state["last_run"] = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    state["run_count"] = state.get("run_count", 0) + 1

    # Daily equity snapshot — used to compute "Today's Move" on the dashboard.
    # Keep one entry per trading day; if the same day runs again (manual retrigger),
    # overwrite. Cap history at 365 entries so state.json stays small.
    total_eq = sum(s.get("equity", 0) for s in state.get("strategies", {}).values())
    snapshot_date = state.get("last_run_date") or "unknown"
    history = state.setdefault("history", [])
    if history and history[-1].get("date") == snapshot_date:
        history[-1]["total_equity"] = round(total_eq, 2)
    else:
        history.append({"date": snapshot_date, "total_equity": round(total_eq, 2)})
    if len(history) > 365:
        state["history"] = history[-365:]

    # Backup before writing — if the write crashes, we can recover
    backup = SCRIPT_DIR / "state.backup.json"
    if STATE_FILE.exists():
        import shutil
        shutil.copy2(STATE_FILE, backup)

    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        # Restore from backup if write failed
        if backup.exists():
            import shutil
            shutil.copy2(backup, STATE_FILE)
        raise


# ============================================================
# STRATEGY EXECUTION
# ============================================================

def run_momentum_rotation(strat, prices, today_str):
    strat["days_since_check"] = strat.get("days_since_check", 0) + 1

    # Update equity using stored shares
    if strat["holding"] and strat["holding"] in prices:
        current_price = prices[strat["holding"]].iloc[-1]['close']
        shares = strat.get("shares", 0)
        if shares > 0:
            strat["equity"] = shares * current_price

    if strat["days_since_check"] < strat["check_interval"]:
        return None

    strat["days_since_check"] = 0

    best, best_mom = eval_momentum_rotation(prices, strat["tickers"], strat["mom_period"])

    action = None

    if best != strat["holding"]:
        # Sell at close
        if strat["holding"] and strat["holding"] in prices:
            exit_price = prices[strat["holding"]].iloc[-1]['close']
            pnl_pct = (exit_price / strat["entry_price"] - 1) * 100 if strat["entry_price"] > 0 else 0
            strat["trades"].append({
                "date": today_str, "action": "SELL", "ticker": strat["holding"],
                "price": round(exit_price, 2), "pnl_pct": round(pnl_pct, 2),
                "equity": round(strat["equity"], 2)
            })
            action = f"SELL {strat['holding']} ({pnl_pct:+.1f}%)"

        # Buy at close
        if best and best in prices:
            entry_price = prices[best].iloc[-1]['close']
            strat["holding"] = best
            strat["entry_price"] = entry_price
            strat["entry_date"] = today_str
            strat["shares"] = strat["equity"] / entry_price
            strat["trades"].append({
                "date": today_str, "action": "BUY", "ticker": best,
                "price": round(entry_price, 2), "equity": round(strat["equity"], 2)
            })
            if action:
                action += f", BUY {best}"
            else:
                action = f"BUY {best}"
        else:
            strat["holding"] = None
            strat["entry_price"] = 0
            strat["entry_date"] = None
            strat["shares"] = 0
            if action:
                action += ", CASH"
            else:
                action = "CASH (no momentum)"
    else:
        action = f"HOLD {strat['holding'] or 'CASH'}"

    return action


def run_rsi_strategy(strat, prices, today_str):
    rsi = eval_rsi_spy(prices)
    if rsi is None:
        return None

    spy_price = prices['SPY'].iloc[-1]['close']
    action = None

    # Update equity using stored shares
    if strat["holding"]:
        shares = strat.get("shares", 0)
        if shares > 0:
            strat["equity"] = shares * spy_price
        strat["days_in_trade"] = strat.get("days_in_trade", 0) + 1

        if rsi > 60 or strat["days_in_trade"] >= 15:
            pnl_pct = (spy_price / strat["entry_price"] - 1) * 100 if strat["entry_price"] > 0 else 0
            strat["trades"].append({
                "date": today_str, "action": "SELL", "ticker": "SPY",
                "price": round(spy_price, 2), "pnl_pct": round(pnl_pct, 2),
                "equity": round(strat["equity"], 2)
            })
            strat["holding"] = None
            strat["entry_price"] = 0
            strat["shares"] = 0
            strat["days_in_trade"] = 0
            action = f"SELL SPY ({pnl_pct:+.1f}%)"
        else:
            action = f"HOLD SPY (RSI={rsi:.0f}, day {strat['days_in_trade']})"
    else:
        if rsi < 10:
            strat["holding"] = "SPY"
            strat["entry_price"] = spy_price
            strat["entry_date"] = today_str
            strat["shares"] = strat["equity"] / spy_price
            strat["days_in_trade"] = 0
            strat["trades"].append({
                "date": today_str, "action": "BUY", "ticker": "SPY",
                "price": round(spy_price, 2), "equity": round(strat["equity"], 2)
            })
            action = f"BUY SPY (RSI={rsi:.0f})"
        else:
            action = f"WAIT (RSI={rsi:.0f}, need <10)"

    return action


def run_qqq_momentum(strat, prices, today_str):
    mom = eval_qqq_momentum(prices)
    if mom is None:
        return None

    qqq_price = prices['QQQ'].iloc[-1]['close']
    action = None

    # Update equity using stored shares
    if strat["holding"]:
        shares = strat.get("shares", 0)
        if shares > 0:
            strat["equity"] = shares * qqq_price
        strat["days_in_trade"] = strat.get("days_in_trade", 0) + 1

        if strat["days_in_trade"] >= 20:
            pnl_pct = (qqq_price / strat["entry_price"] - 1) * 100 if strat["entry_price"] > 0 else 0
            strat["trades"].append({
                "date": today_str, "action": "SELL", "ticker": "QQQ",
                "price": round(qqq_price, 2), "pnl_pct": round(pnl_pct, 2),
                "equity": round(strat["equity"], 2)
            })
            strat["holding"] = None
            strat["entry_price"] = 0
            strat["shares"] = 0
            strat["days_in_trade"] = 0
            action = f"SELL QQQ ({pnl_pct:+.1f}%)"
        else:
            action = f"HOLD QQQ (day {strat['days_in_trade']}/20)"
    else:
        if mom > 0.03:
            strat["holding"] = "QQQ"
            strat["entry_price"] = qqq_price
            strat["entry_date"] = today_str
            strat["shares"] = strat["equity"] / qqq_price
            strat["days_in_trade"] = 0
            strat["trades"].append({
                "date": today_str, "action": "BUY", "ticker": "QQQ",
                "price": round(qqq_price, 2), "equity": round(strat["equity"], 2)
            })
            action = f"BUY QQQ (5d mom={mom*100:.1f}%)"
        else:
            action = f"WAIT (5d mom={mom*100:.1f}%, need >3%)"

    return action


# ============================================================
# HTML DASHBOARD
# ============================================================

def generate_html(state):
    now = state.get("last_run") or "never (no runs yet)"
    last_market_date = state.get("last_run_date")
    strategies = state["strategies"]

    rows = ""
    detail_sections = ""

    sorted_strats = sorted(strategies.items(),
                           key=lambda x: x[1]["equity"], reverse=True)

    FLAGSHIP = "S1_TQQQ_SOXL_20d_5d"
    N_STRATS = len(strategies)
    TOTAL_DEPOSITED = STARTING_CAPITAL * N_STRATS

    # Portfolio totals
    total_equity = sum(s["equity"] for s in strategies.values())
    total_pnl_dollar = total_equity - TOTAL_DEPOSITED
    total_return = (total_equity / TOTAL_DEPOSITED - 1) * 100

    # Today's move (from daily equity snapshots)
    history = state.get("history", [])
    if len(history) >= 2:
        prev_eq = history[-2].get("total_equity", TOTAL_DEPOSITED)
        today_move_dollar = total_equity - prev_eq
        today_move_pct = (total_equity / prev_eq - 1) * 100 if prev_eq > 0 else 0.0
        today_available = True
    else:
        today_move_dollar = 0.0
        today_move_pct = 0.0
        today_available = False

    # Top and worst performers by $ P/L
    perfs = []
    for sid, s in strategies.items():
        short_id = sid.split('_')[0]
        pnl = s["equity"] - STARTING_CAPITAL
        pct = (s["equity"] / STARTING_CAPITAL - 1) * 100
        perfs.append((short_id, s["name"], s["equity"], pnl, pct))
    perfs_sorted = sorted(perfs, key=lambda x: x[3], reverse=True)
    top = perfs_sorted[0]
    worst = perfs_sorted[-1]

    # Group strategies by check schedule for the schedule card.
    # Merge strategies sharing the same frequency, even if their underlying
    # rule differs — the rule is already visible in the strategy name.
    schedule_groups = {}  # interval_label -> [strategy_short_id]
    for sid, s in strategies.items():
        short_id = sid.split('_')[0]  # e.g. "S1" from "S1_TQQQ_SOXL_20d_5d"
        stype = s.get("type")
        if stype == "momentum_rotation":
            interval = s.get("check_interval", 5)
            label = f"every {interval} trading days"
        elif stype in ("rsi_reversion", "qqq_momentum"):
            label = "daily"
        else:
            label = "unknown"
        schedule_groups.setdefault(label, []).append(short_id)

    # Sort groups: daily first, then by interval ascending
    def group_sort_key(item):
        label = item[0]
        if "daily" in label:
            return (0, label)
        if "trading days" in label:
            try:
                n = int(label.split()[1])
                return (1, n)
            except Exception:
                return (2, label)
        return (3, label)

    schedule_parts = []
    for label, sids in sorted(schedule_groups.items(), key=group_sort_key):
        sids_str = " ".join(sorted(sids))
        short_label = label.replace("every ", "every ").replace(" trading days", "d")
        schedule_parts.append(f"{sids_str} {short_label}")
    schedule_line = " · ".join(schedule_parts)

    def pl_color(v):
        if v > 0.5:
            return "#22c55e"
        if v < -0.5:
            return "#ef4444"
        return "#a3a3a3"

    def fmt_dollar(v):
        # Format as "+$50.30" / "-$23.45" / "$0.00"
        if v > 0:
            return f"+${v:,.2f}"
        if v < 0:
            return f"-${abs(v):,.2f}"
        return "$0.00"

    total_color = pl_color(total_pnl_dollar)
    today_color = pl_color(today_move_dollar) if today_available else "#a3a3a3"
    top_color = pl_color(top[3])
    worst_color = pl_color(worst[3])
    total_pnl_str = fmt_dollar(total_pnl_dollar)
    today_move_str = fmt_dollar(today_move_dollar)
    top_pnl_str = fmt_dollar(top[3])
    worst_pnl_str = fmt_dollar(worst[3])

    for sid, s in sorted_strats:
        eq = s["equity"]
        ret = (eq / STARTING_CAPITAL - 1) * 100
        holding_ticker = s.get("holding")
        n_closed = len([t for t in s["trades"] if t["action"] == "SELL"])
        wins = len([t for t in s["trades"] if t["action"] == "SELL" and t.get("pnl_pct", 0) > 0])

        # Win rate: "—" when no closed trades, percent otherwise
        wr_str = f"{(wins/n_closed*100):.0f}%" if n_closed > 0 else "—"

        if ret > 0:
            color = "#22c55e"
        elif ret < 0:
            color = "#ef4444"
        else:
            color = "#a3a3a3"

        is_flagship = sid == FLAGSHIP
        flag_badge = ' <span class="badge">BEST</span>' if is_flagship else ""
        row_class = ' class="flagship"' if is_flagship else ""

        # Position cell: CASH (gray) or ticker + entry info
        if holding_ticker:
            entry_price = s.get("entry_price", 0)
            entry_date = s.get("entry_date", "")
            days_held = ""
            if entry_date and last_market_date:
                try:
                    d1 = datetime.strptime(entry_date, "%Y-%m-%d")
                    d2 = datetime.strptime(last_market_date, "%Y-%m-%d")
                    n_days = (d2 - d1).days
                    days_held = f" · {n_days}d" if n_days > 0 else " · today"
                except Exception:
                    pass

            # Unrealized P/L: equity / (shares * entry_price) - 1
            # Treat sub-display-precision values as exactly zero so float-rounding
            # noise (e.g. -1e-13%) doesn't show as "-0.0%" in red while another
            # strategy shows "+0.0%" in green — both are effectively flat.
            shares = s.get("shares", 0)
            if entry_price > 0 and shares > 0:
                entry_value = shares * entry_price
                unr_pct = (eq / entry_value - 1) * 100
                if abs(unr_pct) < 0.05:  # below 0.1% display precision
                    unr_pct = 0.0
                    unr_color = "#a3a3a3"
                elif unr_pct > 0:
                    unr_color = "#22c55e"
                else:
                    unr_color = "#ef4444"
                unr_str = f'<span style="color:{unr_color}">{unr_pct:+.1f}%</span>'
            else:
                unr_str = "—"

            entry_str = f"${entry_price:,.2f}" if entry_price > 0 else "—"
            position_cell = (
                f'<strong>{holding_ticker}</strong>'
                f'<span style="color:#737373;font-size:0.85em">{days_held}</span>'
            )
        else:
            position_cell = '<span style="color:#737373">CASH</span>'
            entry_str = "—"
            unr_str = "—"

        rows += f"""
        <tr{row_class}>
            <td>{s['name']}{flag_badge}</td>
            <td>{position_cell}</td>
            <td style="color:#a3a3a3">{entry_str}</td>
            <td>{unr_str}</td>
            <td style="color:{color};font-weight:bold">${eq:,.0f}</td>
            <td style="color:{color};font-weight:bold">{ret:+.1f}%</td>
            <td>{n_closed}</td>
            <td>{wr_str}</td>
        </tr>"""

        # Trade log for this strategy
        recent_trades = s["trades"][-10:]
        trade_rows = ""
        if not recent_trades:
            trade_rows = (
                '<tr><td colspan="6" style="color:#737373;font-style:italic;'
                'text-align:center;padding:20px">'
                'No trades yet — waiting for first signal.'
                '</td></tr>'
            )
        else:
            for t in recent_trades:
                pnl = t.get("pnl_pct")
                # Distinguish SELL (always show pnl, even 0.0%) from BUY (no pnl)
                if t["action"] == "SELL" and isinstance(pnl, (int, float)):
                    pnl_str = f"{pnl:+.1f}%"
                    if pnl > 0:
                        pnl_color = "#22c55e"
                    elif pnl < 0:
                        pnl_color = "#ef4444"
                    else:
                        pnl_color = "#a3a3a3"
                else:
                    pnl_str = ""
                    pnl_color = "#888"
                trade_rows += f"""
                <tr>
                    <td>{t['date']}</td>
                    <td>{t['action']}</td>
                    <td>{t.get('ticker','')}</td>
                    <td>${t.get('price',0):,.2f}</td>
                    <td style="color:{pnl_color}">{pnl_str}</td>
                    <td>${t.get('equity',0):,.0f}</td>
                </tr>"""

        flag_label = ' <span class="badge">BEST</span>' if is_flagship else ""
        detail_sections += f"""
        <h3>{s['name']}{flag_label}</h3>
        <table>
            <tr><th>Date</th><th>Action</th><th>Ticker</th><th>Price</th><th>P&L</th><th>Equity</th></tr>
            {trade_rows}
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Trader Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0a0a0a; color: #e5e5e5; padding: 20px; }}
        h1 {{ color: #fff; margin-bottom: 5px; }}
        h2 {{ color: #a3a3a3; margin-top: 30px; margin-bottom: 10px; font-size: 1.1em; }}
        h3 {{ color: #d4d4d4; margin-top: 25px; margin-bottom: 8px; font-size: 1em; }}
        .meta {{ color: #737373; margin-bottom: 20px; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 0.9em; }}
        th {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #333;
             color: #a3a3a3; font-weight: 500; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #1a1a1a; }}
        tr:hover {{ background: #111; }}
        tr.flagship {{ background: #0f1d0f; border-left: 3px solid #22c55e; }}
        tr.flagship:hover {{ background: #142214; }}
        .badge {{ background: #22c55e; color: #000; font-size: 0.7em; font-weight: 700;
                 padding: 2px 8px; border-radius: 4px; margin-left: 8px;
                 text-transform: uppercase; letter-spacing: 0.5px; }}
        .flagship-note {{ background: #0f1d0f; border: 1px solid #1a3a1a; border-radius: 6px;
                         padding: 12px 16px; margin-bottom: 20px; font-size: 0.85em; color: #86efac; }}
        .note {{ color: #737373; font-size: 0.85em; margin-top: 30px;
                padding-top: 15px; border-top: 1px solid #222; }}
        .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                    gap: 12px; margin-bottom: 25px; }}
        .stat {{ background: #141414; border: 1px solid #222; border-radius: 6px;
                padding: 12px 14px; }}
        .stat .label {{ color: #737373; font-size: 0.75em; text-transform: uppercase;
                       letter-spacing: 0.5px; margin-bottom: 4px; }}
        .stat .value {{ color: #fff; font-size: 1.4em; font-weight: 600; }}
        .stat .sub {{ color: #737373; font-size: 0.75em; margin-top: 2px; }}
        .schedule-line {{ color: #a3a3a3; font-size: 0.85em; margin-bottom: 25px; }}
    </style>
</head>
<body>
    <h1>Paper Trader</h1>
    <div class="meta">Last updated: {now} | Run #{state.get('run_count', 0)} |
    Started: {state.get('created', 'unknown')}</div>

    <div class="overview">
        <div class="stat">
            <div class="label">Total P/L</div>
            <div class="value" style="color:{total_color}">{total_pnl_str}</div>
            <div class="sub">{total_return:+.1f}% · account ${total_equity:,.0f}</div>
        </div>
        <div class="stat">
            <div class="label">Today's Move</div>
            <div class="value" style="color:{today_color}">{today_move_str}</div>
            <div class="sub">{today_move_pct:+.2f}%{'' if today_available else ' · (starts on 2nd run)'}</div>
        </div>
        <div class="stat">
            <div class="label">Top</div>
            <div class="value" style="color:{top_color}">{top_pnl_str}</div>
            <div class="sub">{top[0]} · {top[4]:+.1f}%</div>
        </div>
        <div class="stat">
            <div class="label">Worst</div>
            <div class="value" style="color:{worst_color}">{worst_pnl_str}</div>
            <div class="sub">{worst[0]} · {worst[4]:+.1f}%</div>
        </div>
    </div>

    <div class="schedule-line">
        Runs 3:30 PM ET daily · Equity updates daily · Checks: {schedule_line}
    </div>

    <h2>Strategy Performance</h2>
    <div class="flagship-note">
        <strong>S1 (TQQQ+SOXL 20d/5d)</strong> is the flagship strategy.
        Returned +568% on unseen 2025-2026 data while surviving 2022 bear with +3%.
        S2-S6 are comparison strategies. S7-S8 are probationary additions
        (validated on 2023-2026 unseen data, monitored live before promotion).
    </div>
    <table>
        <tr><th>Strategy</th><th>Position</th><th>Entry</th><th>Unrealized</th>
        <th>Equity</th><th>Return</th><th>Closed</th><th>WR</th></tr>
        {rows}
    </table>

    <h2>Recent Trades</h2>
    {detail_sections}

    <div class="note">
        All trades are paper (simulated). Starting capital: $1,000 per strategy
        (${TOTAL_DEPOSITED:,.0f} total simulated deposit).
        Data from Yahoo Finance. Strategies evaluate at their own check intervals;
        "Closed" counts completed round-trip trades. "Unrealized" is the change since entry
        on currently-open positions.
    </div>
</body>
</html>"""

    os.makedirs(HTML_FILE.parent, exist_ok=True)
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)


# ============================================================
# MAIN
# ============================================================

def is_trading_day(prices):
    """Check if today is a real trading day by looking at the actual data.
    If the latest price data is from today's ET market date, the market
    is/was open. Uses ET because Yahoo returns market dates (ET), and
    in CI the system clock is UTC — comparing UTC date vs ET market date
    creates a false-skip after midnight UTC (= late evening ET prior day)."""
    try:
        from zoneinfo import ZoneInfo
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        today_wd = datetime.now(ZoneInfo("America/New_York")).weekday()
    except Exception:
        today_et = datetime.now().date()
        today_wd = datetime.now().weekday()

    if today_wd >= 5:
        return False

    # Check if any ticker has data from today's ET trading date
    for ticker, df in prices.items():
        latest_date = df.iloc[-1]['date'].date()
        if latest_date == today_et:
            return True

    # No ticker has today's data — market is closed (holiday)
    return False


def main():
    print(f"Paper Trader - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    force_run = "--force" in sys.argv

    # Check if we're in the right time window (3:15-4:15 PM Eastern)
    if not force_run:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        eastern = datetime.now(ZoneInfo("America/New_York"))
        et_hour, et_min = eastern.hour, eastern.minute
        et_total = et_hour * 60 + et_min

        if eastern.weekday() >= 5:
            print(f"Weekend ({eastern.strftime('%A')}). Skipping.")
            return

        # Allow any run after 3:15 PM ET on a trading day. The lower bound
        # blocks premarket execution (so we don't trade on incomplete data).
        # No upper bound: GitHub Actions cron can be delayed 1-2+ hours,
        # and running AFTER market close is fine — Yahoo's "Close" by then
        # is the actual closing price, which matches MOC-order execution.
        if et_total < 15 * 60 + 15:
            print(f"Before trading window (current ET: {eastern.strftime('%I:%M %p')}). Skipping.")
            return

        print(f"Eastern time: {eastern.strftime('%I:%M %p')} - within trading window.")

    print("Fetching prices...")
    prices = get_all_prices()
    if not prices:
        print("ERROR: No price data fetched. Exiting.")
        sys.exit(1)

    # Holiday check
    if not is_trading_day(prices) and not force_run:
        print("Market closed today (holiday). Skipping.")
        return

    latest_market_date = max(df.iloc[-1]['date'].date() for df in prices.values())
    today_str = str(latest_market_date)

    # Prevent double execution on same trading day
    state = load_state()
    if state.get("last_run_date") == today_str and not force_run:
        print(f"Already ran for {today_str}. Skipping.")
        return
    state["last_run_date"] = today_str

    print(f"  Got: {', '.join(prices.keys())}")
    for t, df in prices.items():
        print(f"    {t}: ${df.iloc[-1]['close']:.2f}")
    print()

    strategies = state["strategies"]

    print("Running strategies...")
    print()

    for sid, strat in strategies.items():
        stype = strat["type"]

        if stype == "momentum_rotation":
            action = run_momentum_rotation(strat, prices, today_str)
        elif stype == "rsi_reversion":
            action = run_rsi_strategy(strat, prices, today_str)
        elif stype == "qqq_momentum":
            action = run_qqq_momentum(strat, prices, today_str)
        else:
            action = "UNKNOWN TYPE"

        ret = (strat["equity"] / STARTING_CAPITAL - 1) * 100
        print(f"  {strat['name']:<35} ${strat['equity']:>8,.0f} ({ret:>+6.1f}%) | {action}")

    print()

    save_state(state)
    print(f"State saved to {STATE_FILE}")

    generate_html(state)
    print(f"Dashboard saved to {HTML_FILE}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
