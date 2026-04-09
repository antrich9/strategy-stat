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
    # Default filters are disabled (inputs set to false) → act as always true
    volfilt = pd.Series(True, index=df.index)
    atrfilt = pd.Series(True, index=df.index)
    locfiltb = pd.Series(True, index=df.index)
    locfilts = pd.Series(True, index=df.index)

    # Bullish and Bearish Fair Value Gap detection
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Replace NaNs with False to avoid truthiness issues
    bfvg = bfvg.fillna(False)
    sfvg = sfvg.fillna(False)

    entries = []
    trade_num = 1
    last_fvg = 0  # 0 = none, 1 = bullish, -1 = bearish

    # Start after enough bars for shift(2)
    start_idx = 2
    for i in range(start_idx, len(df)):
        if bfvg.iloc[i]:
            if last_fvg == -1:
                entry_ts = int(df['time'].iloc[i])
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                }
                entries.append(entry)
                trade_num += 1
            last_fvg = 1
        elif sfvg.iloc[i]:
            if last_fvg == 1:
                entry_ts = int(df['time'].iloc[i])
                entry = {
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                }
                entries.append(entry)
                trade_num += 1
            last_fvg = -1
    return entries