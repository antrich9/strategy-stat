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
    close = df['close']
    high = df['high']
    low = df['low']
    time = df['time']
    
    # Calculate EMAs
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    # Calculate Wilder ATR manually
    period = 14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=tr.index, dtype=float)
    atr.iloc[period - 1] = tr.iloc[:period].mean()
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    
    # Build condition Series
    ema8_above_ema20 = (ema8 > ema20).astype(int)
    ema8_below_ema20 = (ema8 < ema20).astype(int)
    
    crossover = (ema8_above_ema20.diff() == 1) & (ema20 > ema50)
    crossunder = (ema8_below_ema20.diff() == 1) & (ema20 < ema50)
    
    # Skip bars where indicators are NaN
    valid_mask = ~(ema8.isna() | ema20.isna() | ema50.isna() | atr.isna())
    
    entries = []
    trade_num = 1
    position_open = False
    
    for i in range(1, len(df)):
        if not valid_mask.iloc[i]:
            continue
        
        if not position_open:
            if crossover.iloc[i]:
                entry_price = close.iloc[i] - atr.iloc[i]
                entry_ts = int(time.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                position_open = True
            elif crossunder.iloc[i]:
                entry_price = close.iloc[i] + atr.iloc[i]
                entry_ts = int(time.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                position_open = True
        else:
            position_open = False
    
    return entries