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
    if len(df) < 3:
        return []

    # FVG detection using current low/high vs high/low from 2 bars back
    bull_fvg = df['low'] >= df['high'].shift(2)
    bear_fvg = df['high'] <= df['low'].shift(2)

    # Swing detection (local high/low)
    swing_high = (df['high'].shift(1) < df['high']) & (df['high'].shift(3) < df['high']) & (df['high'].shift(4) < df['high'])
    swing_low = (df['low'].shift(1) > df['low']) & (df['low'].shift(3) > df['low']) & (df['low'].shift(4) > df['low'])

    # London time window check (hour, minute)
    def is_in_london_window(hour, minute):
        mins = hour * 60 + minute
        morning = 8 * 60 <= mins < 10 * 60
        afternoon = 14 * 60 <= mins < 17 * 60
        return morning or afternoon

    # Fill NaN with False for boolean Series
    bull_fvg = bull_fvg.fillna(False)
    bear_fvg = bear_fvg.fillna(False)
    swing_high = swing_high.fillna(False)
    swing_low = swing_low.fillna(False)

    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        in_window = is_in_london_window(dt.hour, dt.minute)

        if bull_fvg.iloc[i] and in_window:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if bear_fvg.iloc[i] and in_window:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries