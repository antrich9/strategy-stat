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
    trade_num = 0

    # Calculate EMAs
    ema_fast = df['close'].ewm(span=10, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    ema_bullish = ema_fast > ema_slow
    ema_bearish = ema_fast < ema_slow

    # Time window conditions (London time)
    is_within_time_window = pd.Series(False, index=df.index)
    for idx in range(len(df)):
        dt = datetime.fromtimestamp(df['time'].iloc[idx], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        in_morning = (hour == 7 and minute >= 45) or (8 <= hour < 9) or (hour == 9 and minute <= 45)
        in_afternoon = (hour == 14 and minute >= 45) or (hour == 15) or (hour == 16 and minute <= 45)
        is_within_time_window.iloc[idx] = in_morning or in_afternoon

    # Helper functions for OB detection
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]

    def is_ob_up(idx):
        if idx + 1 >= len(df):
            return False
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]

    def is_ob_down(idx):
        if idx + 1 >= len(df):
            return False
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]

    def is_fvg_up(idx):
        if idx + 2 >= len(df):
            return False
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]

    def is_fvg_down(idx):
        if idx + 2 >= len(df):
            return False
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]

    # Build boolean series for OB and FVG
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        ob_up.iloc[i] = is_ob_up(i)
        ob_down.iloc[i] = is_ob_down(i)
        fvg_up.iloc[i] = is_fvg_up(i)
        fvg_down.iloc[i] = is_fvg_down(i)

    # Build entry conditions
    # Long: ema_bullish AND in_time_window AND ob_up AND fvg_up
    # Short: ema_bearish AND in_time_window AND ob_down AND fvg_down
    long_condition = ema_bullish & is_within_time_window & ob_up & fvg_up
    short_condition = ema_bearish & is_within_time_window & ob_down & fvg_down

    # Iterate through bars
    for i in range(len(df)):
        # Check long entries
        if i > 0 and long_condition.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        # Check short entries
        if i > 0 and short_condition.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return results