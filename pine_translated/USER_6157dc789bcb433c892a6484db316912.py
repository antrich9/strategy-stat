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
    # SSL Channel parameters
    len_period = 10

    # Simple moving averages for high and low
    sma_high = df['high'].rolling(window=len_period).mean()
    sma_low = df['low'].rolling(window=len_period).mean()

    # Determine Hlv state (1 when close > sma_high, -1 when close < sma_low, else hold prior)
    mask_up = df['close'] > sma_high
    mask_down = df['close'] < sma_low

    Hlv = pd.Series(np.nan, index=df.index)
    Hlv.iloc[0] = 0  # initial state
    Hlv.loc[mask_up] = 1
    Hlv.loc[mask_down] = -1
    Hlv = Hlv.ffill()

    # SSL Down and Up lines
    ssl_down = pd.Series(np.where(Hlv < 0, sma_high, sma_low), index=df.index)
    ssl_up = pd.Series(np.where(Hlv < 0, sma_low, sma_high), index=df.index)

    # Crossover (long) and crossunder (short) conditions
    long_condition = (ssl_up > ssl_down) & (ssl_up.shift(1) <= ssl_down.shift(1))
    short_condition = (ssl_up < ssl_down) & (ssl_up.shift(1) >= ssl_down.shift(1))

    # Ensure boolean type and handle NaNs
    long_condition = long_condition.fillna(False).astype(bool)
    short_condition = short_condition.fillna(False).astype(bool)

    entries = []
    trade_num = 1

    for i, row in df.iterrows():
        if long_condition.loc[i]:
            entry_ts = int(row['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(row['close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.loc[i]:
            entry_ts = int(row['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(row['close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries