import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    results = []
    trade_num = 1

    close = df['close']
    high = df['high']
    low = df['low']
    open_arr = df['open']
    volume = df['volume']
    time = df['time']

    n = len(df)
    if n < 3:
        return results

    # London time windows: 7:45-9:45 and 14:45-16:45
    in_trading_window = pd.Series(False, index=df.index)
    for i in range(n):
        dt = datetime.fromtimestamp(time.iloc[i], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_min = hour * 60 + minute
        # Morning: 7:45 (465) to 9:45 (585)
        # Afternoon: 14:45 (885) to 16:45 (1005)
        in_morning = 465 <= total_min < 585
        in_afternoon = 885 <= total_min < 1005
        in_trading_window.iloc[i] = in_morning or in_afternoon

    # Volume filter
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5

    # ATR filter (Wilder)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)

    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG: low > high[2]
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    # Bearish FVG: high < low[2]
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts

    # Entry conditions
    long_cond = in_trading_window & bfvg
    short_cond = in_trading_window & sfvg

    # Iterate and collect entries
    for i in range(2, n):
        if long_cond.iloc[i]:
            ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return results