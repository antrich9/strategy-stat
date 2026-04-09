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
    
    # Calculate EMAs (8, 20, 50)
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate ATR (14) using Wilder's method
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Pivot Period
    PP = 5
    
    # Calculate pivot highs and lows
    pivot_high = df['high'].rolling(window=PP+1, min_periods=1).max()
    pivot_high = pivot_high.where(df['high'] == pivot_high, other=np.nan)
    pivot_high = pivot_high.shift(PP)
    
    pivot_low = df['low'].rolling(window=PP+1, min_periods=1).min()
    pivot_low = pivot_low.where(df['low'] == pivot_low, other=np.nan)
    pivot_low = pivot_low.shift(PP)
    
    # Track major high and low levels
    last_major_high = np.nan
    last_major_low = np.nan
    
    # Market structure signals
    bullish_bos = pd.Series(False, index=df.index)
    bearish_bos = pd.Series(False, index=df.index)
    bullish_choch = pd.Series(False, index=df.index)
    bearish_choch = pd.Series(False, index=df.index)
    
    # Iterate to fill market structure
    for i in range(PP, len(df)):
        if pd.notna(pivot_high.iloc[i]):
            last_major_high = df['high'].iloc[i]
        if pd.notna(pivot_low.iloc[i]):
            last_major_low = df['low'].iloc[i]
        
        if pd.notna(last_major_high) and df['close'].iloc[i] > last_major_high and ema8.iloc[i] > ema20.iloc[i] > ema50.iloc[i]:
            bullish_bos.iloc[i] = True
        
        if pd.notna(last_major_low) and df['close'].iloc[i] < last_major_low and ema8.iloc[i] < ema20.iloc[i] < ema50.iloc[i]:
            bearish_bos.iloc[i] = True
        
        if i > 0 and ema8.iloc[i] > ema20.iloc[i] and ema8.iloc[i-1] <= ema20.iloc[i-1] and atr.iloc[i] > atr.iloc[i-1]:
            bullish_choch.iloc[i] = True
        
        if i > 0 and ema8.iloc[i] < ema20.iloc[i] and ema8.iloc[i-1] >= ema20.iloc[i-1] and atr.iloc[i] > atr.iloc[i-1]:
            bearish_choch.iloc[i] = True
    
    # Entry conditions
    long_condition = bullish_bos | bullish_choch
    short_condition = bearish_bos | bearish_choch
    
    # Build entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
        
        if short_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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