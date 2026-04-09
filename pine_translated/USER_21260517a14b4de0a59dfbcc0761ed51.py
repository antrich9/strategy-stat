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
    entries = []
    trade_num = 1

    # London time windows
    london_start_morning = 7 * 3600 + 45 * 60
    london_end_morning = 9 * 3600 + 45 * 60
    london_start_afternoon = 14 * 3600 + 45 * 60
    london_end_afternoon = 16 * 3600 + 45 * 60

    # Calculate time of day in seconds from midnight UTC
    df['_hour'] = df['time'] // 3600 % 24
    df['_minute'] = (df['time'] // 60) % 60
    df['_second'] = df['time'] % 60
    df['_time_of_day'] = df['_hour'] * 3600 + df['_minute'] * 60 + df['_second']

    isWithinMorningWindow = (df['_time_of_day'] >= london_start_morning) & (df['_time_of_day'] < london_end_morning)
    isWithinAfternoonWindow = (df['_time_of_day'] >= london_start_afternoon) & (df['_time_of_day'] < london_end_afternoon)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow

    # Volume filter
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5

    # ATR filter (Wilder)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    # Bearish FVG
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Combined entry conditions
    long_condition = bfvg & in_trading_window
    short_condition = sfvg & in_trading_window

    for i in range(len(df)):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]) or pd.isna(locfiltb.iloc[i]) or pd.isna(locfilts.iloc[i]):
            continue

        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries