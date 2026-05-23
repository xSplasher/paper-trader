"""
Paper Trading System — 6 strategies running simultaneously.
Runs daily. Checks signals on schedule. Logs everything to state.json.
Generates HTML dashboard at docs/index.html.

Strategies:
1. TQQQ+SOXL, 20d momentum, 5d check (best OOS performer)
2. TQQQ+SOXL, 30d momentum, 10d check (slower, fewer trades)
3. TQQQ+SOXL, 20d momentum, 10d check (conservative)
4. SPY RSI(2) mean reversion (buy dips, hold ~5 days)
5. QQQ 5d momentum > 3%, hold 20 days
6. UPRO+FAS, 20d momentum, 5d check (non-tech rotation)
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
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data = data.rename(columns={'Date': 'date', 'Close': 'close',
                                 'High': 'high', 'Low': 'low', 'Open': 'open'})
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    return data[['date', 'close', 'open']].sort_values('date').reset_index(drop=True)


def get_all_prices():
    tickers = ['TQQQ', 'SOXL', 'SPY', 'QQQ', 'UPRO', 'FAS']
    prices = {}
    for t in tickers:
        try:
            df = fetch_prices(t)
            if len(df) > 0:
                prices[t] = df
        except Exception as e:
            print(f"  WARN: Failed to fetch {t}: {e}")
    return prices


def compute_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(2).mean()
    avg_loss = loss.rolling(2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_2'] = 100 - (100 / (1 + rs))
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

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)

    strategies = {
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
            "days_since_check": 999,
            "trades": []
        }
    }

    return {
        "created": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "last_run": None,
        "run_count": 0,
        "strategies": strategies
    }


def save_state(state):
    state["last_run"] = datetime.now().strftime('%Y-%m-%d %H:%M')
    state["run_count"] = state.get("run_count", 0) + 1

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

    if strat["days_since_check"] < strat["check_interval"]:
        # Update equity if holding
        if strat["holding"] and strat["holding"] in prices:
            current_price = prices[strat["holding"]].iloc[-1]['close']
            shares = strat["equity"] / strat["entry_price"] if strat["entry_price"] > 0 else 0
            strat["equity"] = shares * current_price if shares > 0 else strat["equity"]
        return None

    strat["days_since_check"] = 0

    best, best_mom = eval_momentum_rotation(prices, strat["tickers"], strat["mom_period"])

    action = None

    if best != strat["holding"]:
        # Sell current position at close
        if strat["holding"] and strat["holding"] in prices:
            exit_price = prices[strat["holding"]].iloc[-1]['close']
            pnl_pct = (exit_price / strat["entry_price"] - 1) * 100 if strat["entry_price"] > 0 else 0
            shares = strat["equity"] / strat["entry_price"] if strat["entry_price"] > 0 else 0
            strat["equity"] = shares * exit_price if shares > 0 else strat["equity"]

            strat["trades"].append({
                "date": today_str,
                "action": "SELL",
                "ticker": strat["holding"],
                "price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "equity": round(strat["equity"], 2)
            })
            action = f"SELL {strat['holding']} ({pnl_pct:+.1f}%)"

        # Buy new position at close (executed 30 min before close in real life)
        if best and best in prices:
            entry_price = prices[best].iloc[-1]['close']
            strat["holding"] = best
            strat["entry_price"] = entry_price
            strat["entry_date"] = today_str

            strat["trades"].append({
                "date": today_str,
                "action": "BUY",
                "ticker": best,
                "price": round(entry_price, 2),
                "equity": round(strat["equity"], 2)
            })
            if action:
                action += f", BUY {best}"
            else:
                action = f"BUY {best}"
        else:
            strat["holding"] = None
            strat["entry_price"] = 0
            strat["entry_date"] = None
            if action:
                action += ", CASH"
            else:
                action = "CASH (no momentum)"
    else:
        # Update equity
        if strat["holding"] and strat["holding"] in prices:
            current_price = prices[strat["holding"]].iloc[-1]['close']
            shares = strat["equity"] / strat["entry_price"] if strat["entry_price"] > 0 else 0
            strat["equity"] = shares * current_price if shares > 0 else strat["equity"]
        action = f"HOLD {strat['holding'] or 'CASH'}"

    return action


def run_rsi_strategy(strat, prices, today_str):
    rsi = eval_rsi_spy(prices)
    if rsi is None:
        return None

    spy_price = prices['SPY'].iloc[-1]['close']
    action = None

    if strat["holding"]:
        strat["days_in_trade"] = strat.get("days_in_trade", 0) + 1
        shares = strat["equity"] / strat["entry_price"] if strat["entry_price"] > 0 else 0
        strat["equity"] = shares * spy_price if shares > 0 else strat["equity"]

        if rsi > 60 or strat["days_in_trade"] >= 15:
            pnl_pct = (spy_price / strat["entry_price"] - 1) * 100 if strat["entry_price"] > 0 else 0
            strat["trades"].append({
                "date": today_str,
                "action": "SELL",
                "ticker": "SPY",
                "price": round(spy_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "equity": round(strat["equity"], 2)
            })
            strat["holding"] = None
            strat["entry_price"] = 0
            strat["days_in_trade"] = 0
            action = f"SELL SPY ({pnl_pct:+.1f}%)"
        else:
            action = f"HOLD SPY (RSI={rsi:.0f}, day {strat['days_in_trade']})"
    else:
        if rsi < 10:
            strat["holding"] = "SPY"
            strat["entry_price"] = spy_price
            strat["entry_date"] = today_str
            strat["days_in_trade"] = 0
            strat["trades"].append({
                "date": today_str,
                "action": "BUY",
                "ticker": "SPY",
                "price": round(spy_price, 2),
                "equity": round(strat["equity"], 2)
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

    if strat["holding"]:
        strat["days_in_trade"] = strat.get("days_in_trade", 0) + 1
        shares = strat["equity"] / strat["entry_price"] if strat["entry_price"] > 0 else 0
        strat["equity"] = shares * qqq_price if shares > 0 else strat["equity"]

        if strat["days_in_trade"] >= 20:
            pnl_pct = (qqq_price / strat["entry_price"] - 1) * 100 if strat["entry_price"] > 0 else 0
            strat["trades"].append({
                "date": today_str,
                "action": "SELL",
                "ticker": "QQQ",
                "price": round(qqq_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "equity": round(strat["equity"], 2)
            })
            strat["holding"] = None
            strat["entry_price"] = 0
            strat["days_in_trade"] = 0
            action = f"SELL QQQ ({pnl_pct:+.1f}%)"
        else:
            action = f"HOLD QQQ (day {strat['days_in_trade']}/20)"
    else:
        if mom > 0.03:
            strat["holding"] = "QQQ"
            strat["entry_price"] = qqq_price
            strat["entry_date"] = today_str
            strat["days_in_trade"] = 0
            strat["trades"].append({
                "date": today_str,
                "action": "BUY",
                "ticker": "QQQ",
                "price": round(qqq_price, 2),
                "equity": round(strat["equity"], 2)
            })
            action = f"BUY QQQ (5d mom={mom*100:.1f}%)"
        else:
            action = f"WAIT (5d mom={mom*100:.1f}%, need >3%)"

    return action


# ============================================================
# HTML DASHBOARD
# ============================================================

def generate_html(state):
    now = state.get("last_run", "unknown")
    strategies = state["strategies"]

    rows = ""
    detail_sections = ""

    sorted_strats = sorted(strategies.items(),
                           key=lambda x: x[1]["equity"], reverse=True)

    FLAGSHIP = "S1_TQQQ_SOXL_20d_5d"

    for sid, s in sorted_strats:
        eq = s["equity"]
        ret = (eq / STARTING_CAPITAL - 1) * 100
        holding = s.get("holding") or "CASH"
        n_trades = len([t for t in s["trades"] if t["action"] == "SELL"])
        wins = len([t for t in s["trades"] if t["action"] == "SELL" and t.get("pnl_pct", 0) > 0])
        wr = (wins / n_trades * 100) if n_trades > 0 else 0

        color = "#22c55e" if ret > 0 else "#ef4444"

        is_flagship = sid == FLAGSHIP
        flag_badge = ' <span class="badge">BEST</span>' if is_flagship else ""
        row_class = ' class="flagship"' if is_flagship else ""

        rows += f"""
        <tr{row_class}>
            <td>{s['name']}{flag_badge}</td>
            <td><strong>{holding}</strong></td>
            <td style="color:{color};font-weight:bold">${eq:,.0f}</td>
            <td style="color:{color};font-weight:bold">{ret:+.1f}%</td>
            <td>{n_trades}</td>
            <td>{wr:.0f}%</td>
        </tr>"""

        # Trade log for this strategy
        recent_trades = s["trades"][-10:]
        trade_rows = ""
        for t in recent_trades:
            pnl = t.get("pnl_pct", "")
            pnl_str = f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) and pnl != 0 else ""
            pnl_color = "#22c55e" if isinstance(pnl, (int, float)) and pnl > 0 else "#ef4444" if isinstance(pnl, (int, float)) and pnl < 0 else "#888"
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
    </style>
</head>
<body>
    <h1>Paper Trader</h1>
    <div class="meta">Last updated: {now} | Run #{state.get('run_count', 0)} |
    Started: {state.get('created', 'unknown')}</div>

    <h2>Strategy Performance</h2>
    <div class="flagship-note">
        <strong>S1 (TQQQ+SOXL 20d/5d)</strong> is the flagship strategy.
        It returned +568% on unseen 2025-2026 data while surviving the 2022 bear market with +3%.
        The others are comparison strategies running in parallel to validate.
    </div>
    <table>
        <tr><th>Strategy</th><th>Holding</th><th>Equity</th><th>Return</th>
        <th>Trades</th><th>Win Rate</th></tr>
        {rows}
    </table>

    <h2>Recent Trades</h2>
    {detail_sections}

    <div class="note">
        All trades are paper (simulated). Starting capital: $1,000 per strategy.
        Data from Yahoo Finance. System checks daily, strategies evaluate on their own schedules.
    </div>
</body>
</html>"""

    os.makedirs(HTML_FILE.parent, exist_ok=True)
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)


# ============================================================
# MAIN
# ============================================================

def is_trading_day():
    today = datetime.now()
    if today.weekday() >= 5:
        return False
    return True


def main():
    print(f"Paper Trader - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    force_run = "--force" in sys.argv
    if not is_trading_day() and not force_run:
        print("Weekend - skipping. Use --force to override.")
        return

    print("Fetching prices...")
    prices = get_all_prices()
    if not prices:
        print("ERROR: No price data fetched. Exiting.")
        sys.exit(1)

    today_str = datetime.now().strftime('%Y-%m-%d')

    print(f"  Got: {', '.join(prices.keys())}")
    for t, df in prices.items():
        print(f"    {t}: ${df.iloc[-1]['close']:.2f}")
    print()

    state = load_state()
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
