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
    fvgTH = 0.5
    atr_len = 144
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - low.shift(1)).abs()
    tr3 = (high.shift(1) - low).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    atr_filtered = atr * fvgTH
    
    bullG = close > high.shift(1)
    bearG = close < low.shift(1)
    
    bullG_filled = bullG.fillna(False)
    bearG_filled = bearG.fillna(False)
    
    bull = (low - high.shift(2)) > atr_filtered
    bear = (low.shift(2) - high) > atr_filtered
    
    entry_conditions = bullG_filled & bull & ~bullG_filled.shift(1).fillna(False)
    entry_conditions |= bearG_filled & bear & ~bearG_filled.shift(1).fillna(False)
    
    trade_num = 0
    entries = []
    
    for i in range(len(df)):
        if entry_conditions.iloc[i]:
            if bullG_filled.iloc[i] if i == 0 else (bullG_filled.iloc[i] and (not bullG_filled.iloc[i-1] if i > 0 else True)):
                trade_num += 1
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
            elif bearG_filled.iloc[i] if i == 0 else (bearG_filled.iloc[i] and (not bearG_filled.iloc[i-1] if i > 0 else True)):
                trade_num += 1
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
    
    return entries