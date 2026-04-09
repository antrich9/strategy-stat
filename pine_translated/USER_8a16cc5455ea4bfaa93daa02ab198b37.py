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
    # Strategy parameters (hard‑coded as in the original Pine Script)
    fast_length = 8
    medium_length = 20
    slow_length = 50

    close = df['close']

    # EMAs (exponential moving averages)
    fast_ema = close.ewm(span=fast_length, adjust=False).mean()
    medium_ema = close.ewm(span=medium_length, adjust=False).mean()
    slow_ema = close.ewm(span=slow_length, adjust=False).mean()

    # Crossover / crossunder detection
    crossover_fast = (close > fast_ema) & (close.shift(1) <= fast_ema.shift(1))
    crossunder_fast = (close < fast_ema) & (close.shift(1) >= fast_ema.shift(1))

    # Fill NaNs that arise from shift operations
    crossover_fast = crossover_fast.fillna(False)
    crossunder_fast = crossunder_fast.fillna(False)

    # Ensure EMA values are available (ignore bars with missing EMAs)
    valid_ema = fast_ema.notna() & medium_ema.notna() & slow_ema.notna()

    # Entry conditions
    long_cond = (crossover_fast & (close > medium_ema) & (close > slow_ema) & valid_ema).fillna(False)
    short_cond = (crossunder_fast & (close < medium_ema) & (close < slow_ema) & valid_ema).fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1

    return entries