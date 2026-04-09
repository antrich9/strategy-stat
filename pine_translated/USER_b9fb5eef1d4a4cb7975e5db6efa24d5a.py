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
    # Calculate EMAs (ta.ema(close, length) = close.ewm(span=length, adjust=False).mean())
    fastEMA = df['close'].ewm(span=50, adjust=False).mean()
    slowEMA = df['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate Wilder ATR manually
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    # True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR initialization and calculation
    period = 14
    atr = pd.Series(np.nan, index=df.index)
    atr.iloc[period - 1] = tr.iloc[0:period].mean()
    
    for i in range(period, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    
    # Detect crossover (fastEMA crosses above slowEMA)
    crossOver = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    
    # Generate entries on crossover
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if crossOver.iloc[i]:
            # Skip bars where required indicators are NaN
            if pd.isna(fastEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]) or pd.isna(atr.iloc[i]):
                continue
            
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            
            trade_num += 1
    
    return entries