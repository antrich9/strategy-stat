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
    # Indicator parameters from the Pine Script
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult = 2.0

    close = df['close']

    # MACD (fast EMA - slow EMA)
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()
    macd = fastEMA - slowEMA
    macd_prev = macd.shift(1)

    # t1 = (MACD change) * sensitivity
    t1 = (macd - macd_prev) * sensitivity

    # Bollinger Band width (e1) = 2 * mult * standard deviation
    rolling_std = close.rolling(window=channelLength).std()
    e1 = 2 * mult * rolling_std

    # Trend up: clip negative values to zero
    trendUp = t1.clip(lower=0)

    # Track whether an entry has already been made
    entry_made = False
    trade_num = 1
    entries = []

    # Start from index 1 to have previous bar values available
    for i in range(1, len(df)):
        # Skip bars where required indicators are NaN
        if (pd.isna(trendUp.iloc[i]) or pd.isna(e1.iloc[i]) or
            pd.isna(trendUp.iloc[i-1]) or pd.isna(e1.iloc[i-1])):
            continue

        # Reset entry flag when trendUp goes below e1
        if trendUp.iloc[i] < e1.iloc[i]:
            entry_made = False

        # Long entry condition:
        # not entry_made AND trendUp > e1 AND trendUp[1] <= e1[1]
        if (not entry_made) and (trendUp.iloc[i] > e1.iloc[i]) and (trendUp.iloc[i-1] <= e1.iloc[i-1]):
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

            trade_num += 1
            entry_made = True

    return entries