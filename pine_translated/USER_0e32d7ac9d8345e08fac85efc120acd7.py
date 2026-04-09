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
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # --- Indicators ---
    # Simple Moving Average (20‑period)
    sma20 = df['close'].rolling(20).mean()

    # Hour of the day (UTC) derived from UNIX timestamp
    df['hour'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour

    # Kill‑Zone definition (default parameters from the script)
    london_start, london_end = 3, 4
    ny_start, ny_end = 10, 11

    in_kill_zone = (
        ((df['hour'] >= london_start) & (df['hour'] < london_end)) |
        ((df['hour'] >= ny_start) & (df['hour'] < ny_end))
    )

    # Crossover / Crossunder detection (close vs SMA20)
    close = df['close']
    prev_close = close.shift(1)
    prev_sma20 = sma20.shift(1)

    long_condition = (close > sma20) & (prev_close <= prev_sma20) & in_kill_zone
    short_condition = (close < sma20) & (prev_close >= prev_sma20) & in_kill_zone

    # Guard against NaN in the indicator
    long_condition = long_condition & sma20.notna() & close.notna()
    short_condition = short_condition & sma20.notna() & close.notna()

    # --- Build entry list ---
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries