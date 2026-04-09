#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_FILE = os.path.join(SCRIPT_DIR, 'probe_trade_rows.json')
OUT_FILE = os.path.join(SCRIPT_DIR, 'probe_standardized_result.json')
DEBUG_FILE = os.path.join(SCRIPT_DIR, 'probe_standardized_debug.json')

sys.path.insert(0, SCRIPT_DIR)
from tv_mcp_client import make_client

MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


# Unified trade-row regex that handles:
#   - any currency: USD, EUR, USDT, GBP, JPY, ...  (2-5 uppercase letters)
#   - intraday dates with time:  "Sep 15, 2025, 08:00"
#   - daily dates without time:  "Oct 03, 1983"
#   - comma-separated thousands:  "45,938.2"
#
# Group map:
#   1=trade_num  2=direction
#   3=exit_mon  4=exit_day  5=exit_year  6=exit_time(optional)
#   7=entry_mon 8=entry_day 9=entry_year 10=entry_time(optional)
#   11=price_a  12=currency_a  13=price_b  14=currency_b
_ROW_RE = re.compile(
    r'^(\d+)(Long|Short)\s+Exit\s+Entry\s+'
    r'([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})(?:,\s+(\d{2}:\d{2}))?\s+'  # exit date
    r'([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})(?:,\s+(\d{2}:\d{2}))?\s+'  # entry date
    r'.*?'                                                                   # skip junk (ISO blobs etc.)
    r'([0-9,]+(?:\.\d+)?)\s+([A-Z]{2,5})\s+'                              # price_a  currency_a
    r'([0-9,]+(?:\.\d+)?)\s+([A-Z]{2,5})',                                 # price_b  currency_b
    re.DOTALL,
)


def _make_dt(month_str, day, year, hm):
    """Build a UTC datetime; hm is 'HH:MM' or None (daily → midnight UTC)."""
    h, mi = (int(hm[:2]), int(hm[3:])) if hm else (0, 0)
    return datetime(year, MONTHS[month_str], day, h, mi, tzinfo=timezone.utc)


def parse_trade_row(row):
    cleaned = row.replace(' | ', ' ')
    m = _ROW_RE.search(cleaned)
    if not m:
        return None

    trade_num   = int(m.group(1))
    direction   = m.group(2).lower()

    exit_dt  = _make_dt(m.group(3),  int(m.group(4)),  int(m.group(5)),  m.group(6))
    entry_dt = _make_dt(m.group(7),  int(m.group(8)),  int(m.group(9)),  m.group(10))

    a = float(m.group(11).replace(',', ''))
    b = float(m.group(13).replace(',', ''))

    # Heuristic: for a Long the entry is the lower of the two prices; for a Short the higher.
    entry_price = max(a, b) if direction == 'short' else min(a, b)
    exit_price  = min(a, b) if direction == 'short' else max(a, b)

    return {
        'trade_num':         trade_num,
        'direction':         direction,
        'entry_time':        entry_dt.isoformat(),
        'exit_time':         exit_dt.isoformat(),
        'entry_ts':          int(entry_dt.timestamp()),
        'exit_ts':           int(exit_dt.timestamp()),
        'raw_price_a':       a,
        'raw_price_b':       b,
        'entry_price_guess': entry_price,
        'exit_price_guess':  exit_price,
    }


def true_range(curr, prev_close):
    return max(curr['high'] - curr['low'], abs(curr['high'] - prev_close), abs(curr['low'] - prev_close))


def compute_atr(bars, period=14):
    atr = [None] * len(bars)
    trs = [None] * len(bars)
    for i, bar in enumerate(bars):
        if i == 0:
            trs[i] = bar['high'] - bar['low']
        else:
            trs[i] = true_range(bar, bars[i-1]['close'])
        if i == period - 1:
            atr[i] = sum(trs[:period]) / period
        elif i >= period:
            atr[i] = ((atr[i-1] * (period - 1)) + trs[i]) / period
    return atr


def find_bar_index_at_or_before(bars, ts):
    idx = None
    for i, bar in enumerate(bars):
        if bar['time'] <= ts:
            idx = i
        else:
            break
    return idx


def simulate_trade(entry, bars, atr_values):
    idx = find_bar_index_at_or_before(bars, entry['entry_ts'])
    if idx is None or idx >= len(bars):
        return None
    if atr_values[idx] is None:
        return None

    direction = entry['direction']
    entry_price = entry['entry_price_guess']
    risk = 3.0 * atr_values[idx]
    if not risk or risk <= 0:
        return None

    if direction == 'long':
        stop = entry_price - risk
    else:
        stop = entry_price + risk

    reached_1r = reached_2r = reached_3r = False
    max_r = 0.0
    exit_price = None
    exit_ts = None

    for j in range(idx + 1, len(bars)):
        bar = bars[j]
        high = bar['high']
        low = bar['low']
        pending_stop = stop

        if direction == 'long':
            # Existing stop is active for the full current bar.
            if low <= stop:
                exit_price = stop
                exit_ts = bar['time']
                break
            max_r = max(max_r, (high - entry_price) / risk)
            if not reached_1r and high >= entry_price + 1 * risk:
                reached_1r = True
                pending_stop = max(pending_stop, entry_price)
            if not reached_2r and high >= entry_price + 2 * risk:
                reached_2r = True
                pending_stop = max(pending_stop, entry_price + 1 * risk)
            if not reached_3r and high >= entry_price + 3 * risk:
                reached_3r = True
                pending_stop = max(pending_stop, entry_price + 2 * risk)
        else:
            # Existing stop is active for the full current bar.
            if high >= stop:
                exit_price = stop
                exit_ts = bar['time']
                break
            max_r = max(max_r, (entry_price - low) / risk)
            if not reached_1r and low <= entry_price - 1 * risk:
                reached_1r = True
                pending_stop = min(pending_stop, entry_price)
            if not reached_2r and low <= entry_price - 2 * risk:
                reached_2r = True
                pending_stop = min(pending_stop, entry_price - 1 * risk)
            if not reached_3r and low <= entry_price - 3 * risk:
                reached_3r = True
                pending_stop = min(pending_stop, entry_price - 2 * risk)

        # Stop upgrades only take effect from the next bar onward.
        stop = pending_stop

    if exit_price is None:
        last = bars[-1]
        exit_price = last['close']
        exit_ts = last['time']

    pnl_r = ((exit_price - entry_price) / risk) if direction == 'long' else ((entry_price - exit_price) / risk)

    return {
        'trade_num': entry['trade_num'],
        'direction': direction,
        'entry_time': entry['entry_time'],
        'entry_price': entry_price,
        'risk': risk,
        'exit_time': datetime.fromtimestamp(exit_ts, tz=timezone.utc).isoformat(),
        'exit_price': exit_price,
        'pnl_r': pnl_r,
        'max_r': max_r,
        'reached_1r': reached_1r,
        'reached_2r': reached_2r,
        'reached_3r': reached_3r,
    }


def compute_metrics(sim_trades):
    if not sim_trades:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'reach_1r_count': 0,
            'reach_2r_count': 0,
            'reach_3r_count': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
        }

    wins = [t['pnl_r'] for t in sim_trades if t['pnl_r'] > 0]
    losses = [t['pnl_r'] for t in sim_trades if t['pnl_r'] < 0]
    breakeven = [t['pnl_r'] for t in sim_trades if t['pnl_r'] == 0]
    equity = 0.0
    peak = 0.0
    trough = 0.0
    max_dd = 0.0
    for t in sim_trades:
        equity += t['pnl_r']
        peak = max(peak, equity)
        trough = min(trough, equity)
        max_dd = max(max_dd, peak - equity)

    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (999.0 if wins else 0.0)
    win_rate = 100.0 * len(wins) / len(sim_trades)

    return {
        'trades': len(sim_trades),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 4),
        'max_drawdown': round(max_dd, 4),
        'reach_1r_count': sum(1 for t in sim_trades if t['reached_1r']),
        'reach_2r_count': sum(1 for t in sim_trades if t['reached_2r']),
        'reach_3r_count': sum(1 for t in sim_trades if t['reached_3r']),
        'wins': len(wins),
        'losses': len(losses),
        'breakeven': len(breakeven),
        'ending_equity_r': round(equity, 4),
        'lowest_equity_r': round(trough, 4),
    }


def timeframe_to_ccxt(timeframe):
    mapping = {
        '15': '15m', '60': '1h', '240': '4h', 'D': '1d', 'W': '1w', 'M': '1M'
    }
    return mapping.get(str(timeframe), str(timeframe))


def symbol_to_ccxt(symbol):
    if symbol.upper().startswith('BINANCE:'):
        base = symbol.split(':', 1)[1].upper()
        if base.endswith('USDT') and len(base) > 4:
            return base[:-4] + '/USDT'
    raise ValueError(f'Unsupported ccxt symbol mapping for {symbol}')


def symbol_to_yfinance(symbol):
    s = symbol.upper()
    mapping = {
        'FX:GBPUSD': 'GBPUSD=X',
        'FXCM:GBPUSD': 'GBPUSD=X',
        'OANDA:GBPUSD': 'GBPUSD=X',
        'FX:XAUUSD': 'GC=F',       # XAUUSD=X removed from Yahoo; GC=F (COMEX gold futures) is best proxy
        'FXCM:XAUUSD': 'GC=F',
        'OANDA:XAUUSD': 'GC=F',
        'FX:US30': '^DJI',
        'FXCM:US30': '^DJI',
        'OANDA:US30USD': '^DJI',
        'CAPITALCOM:US30': '^DJI',
        'TVC:DJI': '^DJI',
        'FX:GER30': '^GDAXI',
        'FX:GER40': '^GDAXI',
        'FXCM:GER30': '^GDAXI',
        'CAPITALCOM:GER40': '^GDAXI',
        'OANDA:DE30EUR': '^GDAXI',
        'TVC:DEU40': '^GDAXI',
    }
    if s in mapping:
        return mapping[s]
    raise ValueError(f'Unsupported yfinance symbol mapping for {symbol}')


def fetch_bars_via_ccxt(symbol, timeframe, oldest_entry_ts=None):
    import ccxt
    ex = ccxt.binance({'enableRateLimit': True})
    market = symbol_to_ccxt(symbol)
    tf = timeframe_to_ccxt(timeframe)
    since = None
    if oldest_entry_ts is not None:
        since = max(0, (oldest_entry_ts - 60 * 60 * 24 * 30) * 1000)
    all_rows = []
    cursor = since
    limit = 1000
    for _ in range(10):
        rows = ex.fetch_ohlcv(market, timeframe=tf, since=cursor, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        next_cursor = last_ts + 1
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < limit:
            break
    dedup = []
    seen = set()
    for r in all_rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        dedup.append({
            'time': int(r[0] // 1000),
            'open': float(r[1]),
            'high': float(r[2]),
            'low': float(r[3]),
            'close': float(r[4]),
            'volume': float(r[5]),
        })
    dedup.sort(key=lambda x: x['time'])
    return dedup


def _to_float(v, default=0.0):
    """Safely coerce a value that may be a numpy scalar, pandas Series, or plain float."""
    try:
        if hasattr(v, 'item'):          # numpy scalar / 0-d array
            return float(v.item())
        if hasattr(v, 'iloc'):          # pandas Series (MultiIndex row slice)
            return float(v.iloc[0])
        return float(v)
    except (TypeError, ValueError):
        return default


def fetch_bars_via_yfinance(symbol, timeframe, oldest_entry_ts=None):
    import yfinance as yf
    import pandas as pd
    ticker = symbol_to_yfinance(symbol)

    # yfinance has no native 4h interval; fetch 1h and resample.
    # '240' means 4h in TradingView.
    tf_str = str(timeframe)
    resample_4h = (tf_str == '240')
    tf_map = {
        '15':  ('15m', '60d'),
        '60':  ('60m', '730d'),
        '240': ('1h',  '730d'),   # will resample below
        'D':   ('1d',  '730d'),
    }
    interval, period = tf_map.get(tf_str, ('1d', '730d'))

    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return []

    # yfinance >= 0.2.x returns MultiIndex columns: ('Open', 'GBPUSD=X'), etc.
    # Flatten to a plain Index so row['Open'] gives a scalar, not a Series.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Guard against duplicate column names after flattening (rare but possible).
    df = df.loc[:, ~df.columns.duplicated()]

    # Resample 1h bars → 4h bars for the '240' timeframe.
    if resample_4h and interval == '1h':
        df = df.resample('4h').agg({
            'Open':  'first',
            'High':  'max',
            'Low':   'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna(subset=['Open', 'Close'])

    bars = []
    for idx, row in df.iterrows():
        ts = int(idx.to_pydatetime().timestamp())
        o = _to_float(row.get('Open'))
        h = _to_float(row.get('High'))
        lo = _to_float(row.get('Low'))
        c = _to_float(row.get('Close'))
        vol_raw = row.get('Volume', 0)
        vol = _to_float(vol_raw) if (vol_raw == vol_raw) else 0.0  # NaN check
        if o == 0.0 and h == 0.0 and lo == 0.0 and c == 0.0:
            continue  # skip empty/all-zero bars (weekends etc.)
        bars.append({'time': ts, 'open': o, 'high': h, 'low': lo, 'close': c, 'volume': vol})
    return bars


def parse_args():
    p = argparse.ArgumentParser(description='Run proof-of-concept standardized ATR/trailing-R simulation from cached trade rows.')
    p.add_argument('--input', default=PROBE_FILE, help='input JSON with trusted trade rows')
    p.add_argument('--output', default=OUT_FILE, help='output result JSON')
    p.add_argument('--debug-output', default=DEBUG_FILE, help='debug output JSON')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f'Missing {args.input}. Run extraction first.')

    with open(args.input) as f:
        data = json.load(f)

    rows = data.get('trade_rows', {}).get('rows', [])
    parsed = []
    for row in rows:
        item = parse_trade_row(row)
        if item:
            parsed.append(item)

    strategy_name = data.get('strategy_name_requested') or data.get('strategy_name') or data.get('report', {}).get('strategy_name_reported')
    oldest_entry_ts = min((p['entry_ts'] for p in parsed), default=None)

    symbol = data['symbol']
    timeframe = data['timeframe']
    try:
        if str(symbol).upper().startswith('BINANCE:'):
            bars = fetch_bars_via_ccxt(symbol, timeframe, oldest_entry_ts=oldest_entry_ts)
        else:
            bars = fetch_bars_via_yfinance(symbol, timeframe, oldest_entry_ts=oldest_entry_ts)
    except Exception as e:
        print(f'[OHLCV ERROR] fetch failed for {symbol}/{timeframe}: {e}', file=sys.stderr)
        raise SystemExit(1) from e

    if not bars:
        print(f'[OHLCV ERROR] zero bars returned for {symbol}/{timeframe}', file=sys.stderr)
        raise SystemExit(1)
    try:
        atr_values = compute_atr(bars, period=14)
    except Exception as e:
        print(f'[SIMULATOR ERROR] ATR computation failed for {symbol}/{timeframe}: {e}', file=sys.stderr)
        raise SystemExit(1) from e

    sim_trades = []
    skipped = []
    for entry in parsed:
        try:
            sim = simulate_trade(entry, bars, atr_values)
        except Exception as e:
            print(f'[SIMULATOR ERROR] simulate_trade failed for trade {entry.get("trade_num")}: {e}', file=sys.stderr)
            sim = None
        if sim:
            sim_trades.append(sim)
        else:
            idx = find_bar_index_at_or_before(bars, entry['entry_ts']) if bars else None
            skipped.append({
                'trade_num': entry['trade_num'],
                'entry_time': entry['entry_time'],
                'entry_ts': entry['entry_ts'],
                'bar_index': idx,
                'atr_at_index': (atr_values[idx] if idx is not None and idx < len(atr_values) else None),
            })

    debug = {
        'strategy_name': strategy_name,
        'symbol': data.get('symbol'),
        'timeframe': data.get('timeframe'),
        'parsed_count': len(parsed),
        'bars_count': len(bars),
        'oldest_bar_ts': (bars[0]['time'] if bars else None),
        'newest_bar_ts': (bars[-1]['time'] if bars else None),
        'oldest_bar_iso': (datetime.fromtimestamp(bars[0]['time'], tz=timezone.utc).isoformat() if bars else None),
        'newest_bar_iso': (datetime.fromtimestamp(bars[-1]['time'], tz=timezone.utc).isoformat() if bars else None),
        'oldest_entry_ts': oldest_entry_ts,
        'oldest_entry_iso': (datetime.fromtimestamp(oldest_entry_ts, tz=timezone.utc).isoformat() if oldest_entry_ts else None),
        'first_parsed_entry': (parsed[0] if parsed else None),
        'first_10_bar_times': [
            {
                'ts': b['time'],
                'iso': datetime.fromtimestamp(b['time'], tz=timezone.utc).isoformat(),
                'open': b['open'],
                'close': b['close'],
            }
            for b in bars[:10]
        ],
        'last_10_bar_times': [
            {
                'ts': b['time'],
                'iso': datetime.fromtimestamp(b['time'], tz=timezone.utc).isoformat(),
                'open': b['open'],
                'close': b['close'],
            }
            for b in bars[-10:]
        ] if bars else [],
        'skipped': skipped[:20],
    }

    result = {
        'strategy_name': strategy_name,
        'symbol': data.get('symbol'),
        'timeframe': data.get('timeframe'),
        'parsed_entries': parsed,
        'simulated_trades': sim_trades,
        'metrics': compute_metrics(sim_trades),
        'note': 'Proof-of-concept standardized simulation using TradingView OHLCV, 3xATR(14) initial stop, 1R->0R, 2R->1R, 3R->2R trailing.'
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    with open(args.debug_output, 'w') as f:
        json.dump(debug, f, indent=2)

    print(json.dumps({
        'success': True,
        'out_file': args.output,
        'debug_file': args.debug_output,
        'parsed_entries': len(parsed),
        'simulated_trades': len(sim_trades),
        'metrics': result['metrics'],
        'bars_count': len(bars),
        'oldest_bar_iso': debug['oldest_bar_iso'],
        'newest_bar_iso': debug['newest_bar_iso'],
        'strategy_name': result['strategy_name'],
        'symbol': result['symbol'],
        'timeframe': result['timeframe'],
    }, indent=2))


if __name__ == '__main__':
    main()
